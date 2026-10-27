"""Moving a geometry scene without rebuilding it.

The scene's meshes all share one point buffer (`shared_points`), so a
frame is three steps: compute the offsets of the moving nodes, write
them through a numpy view of VTK's memory, and mark the buffer modified.
Nothing is allocated or rebuilt per frame, which is what allows large
models to be animated.

The animators use no Qt; the caller drives them. The `animate_*`
functions play an animation in a window, or save a still or an MP4.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..units import DEFAULT_SYSTEM
from .geometry import (
    add_geometry,
    display_points,
    place_view,
    shared_points,
    shared_scalars,
)

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.geometry import Geometry
    from ..core.shapes import ShapeSet
    from ..deform import Deflection
    from ..units import UnitSystem


#: frames a second: the window's playback and a saved animation alike,
#: so a movie is the window's own animation written down
MOVIE_FPS = 30

#: every mode animates at this many seconds a cycle, whatever its Hz
SECONDS_PER_CYCLE = 2.0

#: a time record, or a sweep of lines, plays through in about this long
SECONDS_PER_RECORD = 10.0

#: the least a saved animation of a cycle lasts: whole cycles to at
#: least this, since a video player does not loop the way the window does
MOVIE_SECONDS = 6.0


def cycle_parameters(cycle_seconds: float = SECONDS_PER_CYCLE,
                     fps: int = MOVIE_FPS,
                     seconds: float = MOVIE_SECONDS) -> np.ndarray:
    """The phases, in radians, that a saved mode-shape animation steps
    through.

    Whole cycles lasting at least `seconds` in total, at `fps` frames a
    second and at the speed the window plays (`cycle_seconds` a cycle).
    The last frame stops one step short of 2π, so a player set to loop
    shows no repeated frame where the file wraps.
    """
    per_cycle = max(2, round(cycle_seconds * fps))
    cycles = max(1, int(np.ceil(seconds * fps / per_cycle - 1e-9)))
    return 2 * np.pi * np.arange(per_cycle * cycles) / per_cycle


def sweep_parameters(count: int, seconds: float = SECONDS_PER_RECORD,
                     fps: int = MOVIE_FPS) -> np.ndarray:
    """The sample (or line) indices a saved sweep steps through: 0 to
    `count` - 1, once, in `seconds` at `fps` frames a second, the way the
    window plays a record. Indices are spaced evenly, so some are skipped
    when there are more of them than frames, and some repeat when there
    are fewer."""
    frames = max(1, round(seconds * fps))
    return (np.arange(frames) * count // frames).astype(int)


def movie_frames(plotter: Any, step: Any, parameters: Any) -> Any:
    """Yield the scene rendered at each of `parameters`, as images: the
    frames of a movie, one at a time. `step(parameter)` moves the scene
    there; it is an animator's `set_parameter`, or a function that also
    moves whatever else goes with it (a caption, the window's plot
    cursor)."""
    for parameter in parameters:
        step(parameter)
        plotter.render()
        yield plotter.screenshot(return_img=True)


def _color_range(deflection: Deflection) -> float:
    """The top of the color scale: the furthest any node ever travels.

    Fixed for the whole animation rather than rescaled per frame, so the
    top color marks the one instant the model is at its furthest — a scale
    that followed each frame would paint some node yellow in every frame,
    including the ones where nothing is moving.
    """
    peak = float(abs(deflection.peak_magnitude))
    return peak if peak and np.isfinite(peak) else 1.0


class PairedAnimator:
    """Two animated copies, moving in step.

    Each copy is a `GeometryAnimator` with its own point buffer and its
    own automatic scale, so two mode shapes with different
    normalizations are compared by shape, which is the purpose of
    overlaying them. The copies may be on the same geometry or on two.
    """

    def __init__(self, first: GeometryAnimator,
                 second: GeometryAnimator) -> None:
        self.first: GeometryAnimator = first
        self.second: GeometryAnimator = second
        self.axis_unit: str = first.axis_unit

    @property
    def deflection(self) -> Any:
        """The first copy's deflection, which for a pair sharing one
        deflection (the envelope) is the pair's."""
        return self.first.deflection

    @property
    def parameter(self) -> float:
        """The first copy's current phase or index."""
        return self.first.parameter

    def set_scale(self, user_scale: float) -> None:
        """Set both copies' scale, as a multiple of each one's own
        automatic scale."""
        self.first.set_scale(user_scale)
        self.second.set_scale(user_scale)

    def set_parameter(self, parameter: float) -> None:
        """Move both copies to this phase or index."""
        self.first.set_parameter(parameter)
        self.second.set_parameter(parameter)

    def follow(self, drawn: Any) -> None:
        """Move a mesh drawn at nodes with the first copy: the +
        deflection of an envelope, the first set of a comparison."""
        self.first.follow(drawn)

    def render(self) -> None:
        """Render the scene the copies share."""
        self.first.render()

    def reset(self) -> None:
        """Put both copies back at their undeformed positions."""
        self.first.reset()
        self.second.reset()


class GeometryAnimator:
    """A geometry scene whose nodes move with a deflection.

    The geometry is drawn into `plotter` on construction (`add_geometry`,
    with any extra keyword arguments passed on). `deflection` supplies
    offsets for the nodes it moves, and every other node stays at its
    base position. The parameter is a phase in radians for a mode shape,
    and a sample index for time data.

    The automatic scale makes the largest deflection `fraction` of the
    model's size (the diagonal of its bounding box), and `set_scale`
    multiplies it. With `colormap` the nodes are colored by the size of
    their offset, under the array name `scalar_name`, on a color range
    of `clim`; by default the range runs from zero to the largest offset
    any node ever reaches, fixed for the whole animation.
    """

    def __init__(self, plotter: Any, geometry: Geometry,
                 deflection: Deflection,
                 unit_system: UnitSystem | None = None,
                 fraction: float = 0.1, colormap: bool = False,
                 clim: tuple[float, float] | None = None,
                 scalar_name: str = 'displacement',
                 **draw: Any) -> None:
        self.plotter: Any = plotter
        self.geometry: Geometry = geometry
        self.deflection: Any = deflection
        self.unit_system: UnitSystem = unit_system or DEFAULT_SYSTEM

        base, self.axis_unit = display_points(geometry, self.unit_system)
        self.base: np.ndarray = np.ascontiguousarray(base, dtype=np.float64)
        self.points_source, self.view = shared_points(self.base)
        self.meshes: list[Any] = []
        #: per-node deflection magnitude for the color map, and the
        #: one VTK array every mesh shares — None when not coloring
        self.magnitude: np.ndarray | None = None
        self.scalars_source: Any = None
        if colormap:
            self.magnitude, self.scalars_source = shared_scalars(
                len(self.base), scalar_name)
            draw = {**draw, 'scalars': self.scalars_source,
                    'clim': clim or (0.0, _color_range(deflection))}
        add_geometry(plotter, geometry, unit_system=self.unit_system,
                     points_source=self.points_source, meshes=self.meshes,
                     **draw)

        # everything here is in display units, including the yardstick the
        # deflection is scaled against
        self.model_size: float = float(
            np.linalg.norm(self.base.max(axis=0) - self.base.min(axis=0))) or 1.0
        peak = float(abs(deflection.peak))
        self.auto: float = (fraction * self.model_size / peak
                     if peak and np.isfinite(peak) else 1.0)
        self.user_scale: float = 1.0
        self._rows = deflection.rows
        self._base_rows = self.base[self._rows] if len(self._rows) else None
        self._norms = np.empty(len(self._rows))
        #: meshes drawn at nodes that move with them (`follow`)
        self._followers: list[tuple[Any, np.ndarray, np.ndarray, Any]] = []
        self.parameter: float = 0.0
        self.set_parameter(0.0)

    def follow(self, drawn: Any) -> None:
        """Move a mesh drawn at nodes with those nodes from now on: a
        `viz.sensors.SensorMesh`, or anything with its `node_rows`,
        `base`, `points` and `view`. Each point keeps its offset from its
        node, so a body translates with its node and keeps its
        orientation."""
        rows = np.asarray(drawn.node_rows, dtype=np.intp)
        relative = np.ascontiguousarray(drawn.base - self.base[rows])
        self._followers.append((drawn.points, drawn.view, rows, relative))
        self._place_followers()

    def _place_followers(self) -> None:
        # a gather into VTK's own memory and an add in place: nothing is
        # allocated per frame, as for the nodes themselves
        for points, view, rows, relative in self._followers:
            np.take(self.view, rows, axis=0, out=view)
            view += relative
            points.Modified()

    @property
    def scale(self) -> float:
        """The factor offsets are drawn at: the automatic scale times the
        user's."""
        return self.auto * self.user_scale

    def set_scale(self, user_scale: float) -> None:
        """Set the user's scale, a multiple of the automatic one, and
        redraw at the current parameter."""
        self.user_scale = float(user_scale)
        self.set_parameter(self.parameter)

    def set_parameter(self, parameter: float) -> None:
        """Place the nodes for this phase or sample index."""
        self.parameter = parameter
        if self._base_rows is None:
            return
        offsets = self.deflection.offsets(parameter)
        # one write into VTK's own memory, seen by every mesh at once
        self.view[self._rows] = self._base_rows + offsets * self.scale
        self.points_source.Modified()
        self._place_followers()
        if self.magnitude is not None:
            # |offset| per node, without allocating: einsum for the sum of
            # squares, sqrt in place
            np.einsum('ij,ij->i', offsets, offsets, out=self._norms)
            np.sqrt(self._norms, out=self._norms)
            self.magnitude[self._rows] = self._norms
            self.scalars_source.Modified()

    def render(self) -> None:
        """Render the plotter."""
        self.plotter.render()

    def reset(self) -> None:
        """Put every node back at its base position, and its color at
        zero."""
        self.view[:] = self.base
        self.points_source.Modified()
        self._place_followers()
        if self.magnitude is not None:
            self.magnitude[:] = 0.0
            self.scalars_source.Modified()


class EnvelopeAnimator(GeometryAnimator):
    """One copy of a PSD envelope. The envelope is drawn as a pair of
    these, one deflected + and one -.

    `sign` (+1 or -1) multiplies the scale, so the window's one Scale
    control still drives both copies through `PairedAnimator`. The
    parameter is the abscissa line, since an envelope has no phase to
    sweep. Color is absolute: dB below the loudest node at any line,
    between the deflection's `FLOOR_DB` and 0. Every line deflects at
    full size, so the color is the only thing that shows level; a node
    with no data is drawn at the floor.
    """

    def __init__(self, plotter: Any, geometry: Geometry,
                 deflection: Any, sign: float = 1.0,
                 **kwargs: Any) -> None:
        self.sign: float = float(sign)
        super().__init__(plotter, geometry, deflection,
                         clim=(deflection.FLOOR_DB, 0.0),
                         scalar_name='dB', **kwargs)
        if self.magnitude is not None:
            # zero is the *top* of a dB-below scale: nodes with no data
            # must start at the floor, not painted as the loudest
            self.magnitude[:] = deflection.FLOOR_DB
            self.magnitude[self._rows] = deflection.node_decibels()
            self.scalars_source.Modified()

    @property
    def scale(self) -> float:
        """The automatic scale times the user's, times `sign`."""
        return self.auto * self.user_scale * self.sign

    def set_parameter(self, parameter: float) -> None:
        """Move to this abscissa line, and recolor the nodes for it."""
        self.deflection.line = int(parameter)
        super().set_parameter(parameter)
        if self.magnitude is not None:
            self.magnitude[self._rows] = self.deflection.node_decibels()
            self.scalars_source.Modified()


def _scene(plotter: Any, build: Any, colors: Any,
           caption: Any = None) -> tuple[Any, Any]:
    """The moving scene in `plotter`: `build(plotter)` puts the geometry
    in and returns its animator, the axes are annotated, and `caption`
    (parameter -> text) writes the corner caption the window shows,
    rewritten in place as the parameter moves. Returns the animator and
    the `step(parameter)` that moves the scene, caption and all."""
    from .geometry import annotate_scene

    animator = build(plotter)
    annotate_scene(plotter, animator.axis_unit, colors)
    corner = (plotter.add_text(caption(animator.parameter),
                               position='upper_left', font_size=11,
                               color=colors['scene_text'],
                               name='scene-caption')
              if caption is not None else None)

    def step(parameter: Any) -> None:
        animator.set_parameter(parameter)
        if corner is not None:
            # 2 is the upper-left corner; one actor rewritten, not one
            # rebuilt a frame
            corner.SetText(2, caption(parameter))
    return animator, step


def _render_offscreen(geometry: Geometry, build: Any, *, colors: Any,
                      still: Any, parameters: Any,
                      screenshot: str | None, movie: str | None,
                      caption: Any = None, plot: Any = None) -> Any:
    """Every `animate_*`'s file output: the scene at `still` for
    `screenshot`, or stepped through `parameters` (`cycle_parameters`,
    `sweep_parameters`) as an MP4 for `movie` — the path returned. With
    `plot`, a function that makes `(widget, move)` for a scene of a given
    width, the movie carries that plot under the scene and `move(index)`
    walks its cursor with the frames (`gui.movie.over_plot`)."""
    import pyvista as pv

    if screenshot is not None and movie is not None:
        raise ValueError('ask for a screenshot or a movie, not both')
    plotter = pv.Plotter(off_screen=True)
    widget = None
    try:
        plotter.set_background(colors['scene_background'])
        _animator, step = _scene(plotter, build, colors, caption)
        place_view(plotter, geometry.opening_view, render=False)
        if screenshot is not None:
            step(still)
            return plotter.screenshot(str(screenshot))
        from ..gui.movie import over_plot, write_movie

        frames = movie_frames(plotter, step, parameters)
        if plot is not None:
            widget, move = plot(plotter.window_size[0])

            def walked(parameter: Any) -> None:
                step(parameter)
                move(parameter)

            frames = over_plot(movie_frames(plotter, walked, parameters),
                               widget)
        write_movie(movie, frames)
        return str(movie)
    finally:
        if widget is not None:
            widget.close()
        plotter.close()


def _record_plot(data: Any, records: Sequence[int],
                 unit_system: UnitSystem | None, theme: Any) -> Any:
    """The `plot` a movie that walks a record carries under its scene, a
    time history's samples or a PSD's lines: its records drawn as the
    window draws them (`plot.build_plots`), with a cursor at the sample
    or line each frame shows. A function of the scene's width, as
    `_render_offscreen` asks for it."""
    def make(width: int) -> tuple[Any, Any]:
        import pyqtgraph as pg

        from ..plot import _application, build_plots
        from ..theme import theme as resolve_theme

        app = _application()
        widget = pg.GraphicsLayoutWidget()
        widget.resize(int(width), int(0.35 * width))
        build_plots(widget, [(None, data, list(records))],
                    unit_system=unit_system, theme=theme)
        abscissa = np.asarray(data.display_abscissa(unit_system
                                                    or DEFAULT_SYSTEM))
        cursor = pg.InfiniteLine(
            pos=float(abscissa[0]), angle=90,
            pen=pg.mkPen(resolve_theme(theme)['scene_highlight'], width=1))
        widget.getItem(0, 0).addItem(cursor)
        widget.show()
        app.processEvents()

        def move(index: Any) -> None:
            cursor.setValue(float(abscissa[int(index)]))
        return widget, move
    return make


def _play(geometry: Geometry, build: Any, *, colors: Any, title: str,
          show: bool, caption: Any = None, per_second: float = 0.0,
          period: float | None = None, whole: bool = False) -> Any:
    """Every `animate_*`'s window: the scene, advanced `per_second` (a
    phase in radians, or samples) on a wall clock at the window's frame
    rate, so it reads the same however fast the machine renders —
    wrapping at `period`, in `whole` steps for an index. Still when
    `per_second` is zero."""
    from pyvistaqt import BackgroundPlotter

    from . import undeferred

    plotter = undeferred(BackgroundPlotter(title=title, show=show))
    plotter.set_background(colors['scene_background'])
    animator, step = _scene(plotter, build, colors, caption)
    place_view(plotter, geometry.opening_view)
    if per_second:
        interval = 1000 // MOVIE_FPS
        position = [float(animator.parameter)]

        def advance() -> None:
            position[0] += per_second * interval / 1000.0
            if period:
                position[0] %= period
            step(int(position[0]) if whole else position[0])

        plotter.add_callback(advance, interval=interval)
    return plotter


def _single(geometry: Geometry, deflection: Any, *,
            unit_system: UnitSystem | None, scale: float,
            colormap: bool) -> Any:
    """The builder for one deflected geometry."""
    def build(plotter: Any) -> GeometryAnimator:
        animator = GeometryAnimator(plotter, geometry, deflection,
                                    unit_system=unit_system,
                                    colormap=colormap)
        animator.set_scale(scale)
        return animator
    return build


def _sensing(build: Any, sensors: Any, size: float) -> Any:
    """`build` with the sensors a channel table places drawn at their
    nodes and moving with them; `build` itself when there is no table."""
    if sensors is None:
        return build

    def built(plotter: Any) -> Any:
        from .sensors import add_sensors

        animator = build(plotter)
        lead = getattr(animator, 'first', animator)
        drawn, _notes = add_sensors(plotter, lead.geometry, sensors,
                                    lead.unit_system, size)
        if drawn is not None:
            animator.follow(drawn)
        return animator
    return built


def _cycled(geometry: Geometry, build: Any, *, colors: Any,
            cycle_seconds: float, title: str, show: bool,
            screenshot: str | None, movie: str | None,
            caption: Any = None) -> Any:
    """A swept phase — a mode, an ODS, a pair — to a file or a window:
    the still at peak phase, whole cycles as a movie, or played."""
    if screenshot is not None or movie is not None:
        return _render_offscreen(
            geometry, build, colors=colors, still=0.0,
            parameters=cycle_parameters(cycle_seconds),
            screenshot=screenshot, movie=movie, caption=caption)
    return _play(geometry, build, colors=colors, title=title, show=show,
                 caption=caption, per_second=2 * np.pi / cycle_seconds)


def build_pair(plotter: Any, first: tuple[Geometry, Any],
               second: tuple[Geometry, Any], *,
               colors: tuple[str | None, str | None],
               alphas: tuple[float, float],
               unit_system: UnitSystem | None = None,
               text_color: str = '#000000') -> PairedAnimator:
    """Two (geometry, deflection) copies moving in step in one scene,
    each on its own meshes in one flat color (`colors`, None for the
    geometry's own colors) at its own opacity (`alphas`). This is the
    overlay that the window, the report and `animate_pair` all show."""
    return PairedAnimator(*(
        GeometryAnimator(plotter, geometry, deflection,
                         unit_system=unit_system, color_override=color,
                         opacity=alpha, text_color=text_color)
        for (geometry, deflection), color, alpha
        in ((first, colors[0], alphas[0]), (second, colors[1], alphas[1]))))


def pair_deflections(a: ShapeSet, i: int, b: ShapeSet, j: int,
                     a_geometry: Geometry, b_geometry: Geometry,
                     projected: ShapeSet | None = None,
                     on_basis: bool = False) -> tuple[tuple, tuple]:
    """The two (geometry, deflection) copies of an overlaid pair: mode
    `i` of `a` on a's geometry, and mode `j` of `b`, phase-aligned to it
    (`paired_mode`), on b's geometry. With `on_basis`, the second copy
    is instead the projection of b drawn on a's geometry, both sampled
    at the basis DOFs (this needs `projected`). `projected` is b carried
    onto a's DOFs, and is needed when the two geometries differ."""
    from ..core.shapes import aligned_mode, paired_mode
    from ..deform import ShapeDeflection

    first = (a_geometry, ShapeDeflection(a_geometry, a.coordinate,
                                         a.shape_matrix[i]))
    if on_basis and projected is not None:
        second = (a_geometry, ShapeDeflection(
            a_geometry, projected.coordinate,
            aligned_mode(a, i, projected, j)))
    else:
        second = (b_geometry, ShapeDeflection(
            b_geometry, b.coordinate,
            paired_mode(a, i, b, j,
                        None if a_geometry is b_geometry else projected)))
    return first, second


def animate_pair(first: tuple[Geometry, Any], second: tuple[Geometry, Any],
                 *, colors: tuple[str | None, str | None] | None = None,
                 alphas: tuple[float, float] | None = None,
                 caption: str = '',
                 unit_system: UnitSystem | None = None,
                 scale: float = 1.0,
                 cycle_seconds: float = SECONDS_PER_CYCLE,
                 theme: Any = None, screenshot: str | None = None,
                 movie: str | None = None, show: bool = True) -> Any:
    """Two deflections overlaid and moving together, as the window
    animates a picked MAC cell. `pair_deflections` makes the pair.

    Each copy is scaled to its own peak, so the picture compares shape
    and not level. `caption`, when given, is written in the corner of
    every frame; `core.shapes.pair_caption` makes one that includes a
    note on the two scales.
    `colors` default to the theme's pair, and `alphas` to the first
    copy solid and the second translucent (`theme.overlay_alphas`).
    `scale` multiplies both automatic scales, and `cycle_seconds` is the
    time of one cycle. `screenshot` saves the still at peak phase, and
    `movie` saves whole cycles as an MP4 and returns its path; otherwise
    a window plays it and is returned.
    """
    from ..theme import PAIR_COLORS, overlay_alphas
    from ..theme import theme as resolve_theme

    palette = resolve_theme(theme)

    def build(plotter: Any) -> PairedAnimator:
        pair = build_pair(plotter, first, second,
                          colors=colors or PAIR_COLORS,
                          alphas=alphas or overlay_alphas(None, None),
                          unit_system=unit_system,
                          text_color=palette['scene_text'])
        pair.set_scale(scale)
        return pair

    return _cycled(first[0], build, colors=palette,
                   cycle_seconds=cycle_seconds, title='Overlaid modes',
                   show=show, screenshot=screenshot, movie=movie,
                   caption=(lambda _phase: caption) if caption else None)


def animate_shape(geometry: Geometry, shapes: ShapeSet, mode: int = 0, *,
                  unit_system: UnitSystem | None = None,
                  scale: float = 1.0, colormap: bool = True,
                  cycle_seconds: float = SECONDS_PER_CYCLE,
                  theme: Any = None,
                  screenshot: str | None = None,
                  movie: str | None = None,
                  show: bool = True, sensors: Any = None,
                  sensor_size: float = 1.0) -> Any:
    """One mode shape (index `mode` of `shapes`) moving on a geometry, as
    the app animates it.

    With `colormap` the nodes are colored by displacement on viridis, as
    in the app's scene. `scale` multiplies the automatic scale. With
    `screenshot` the deflected shape is rendered to the file at peak
    phase, the frame that shows the shape best. With `movie` it is saved
    as an MP4 (H.264) of whole cycles at `cycle_seconds` a cycle, and
    the path is returned (`gui.movie.write_movie`). Otherwise a window
    plays it and is returned.

    With `sensors`, a channel table, the sensors it places are drawn at
    their nodes and move with them (`viz.sensors`), `sensor_size` times
    true size.
    """
    from ..deform import ShapeDeflection
    from ..theme import theme as resolve_theme

    deflection = ShapeDeflection(geometry, shapes.coordinate,
                                 shapes.shape_matrix[mode])
    return _cycled(geometry, _sensing(_single(geometry, deflection,
                                              unit_system=unit_system,
                                              scale=scale, colormap=colormap),
                                      sensors, sensor_size),
                   colors=resolve_theme(theme), cycle_seconds=cycle_seconds,
                   title=shapes.mode_label(mode), show=show,
                   screenshot=screenshot, movie=movie)


def animate_envelope(geometry: Geometry, data: Any,
                     records: Any = None, *,
                     frequency: float | None = None,
                     quantity: str | None = None,
                     unit_system: UnitSystem | None = None,
                     scale: float = 1.0, colormap: bool = True,
                     seconds: float = SECONDS_PER_RECORD,
                     theme: Any = None,
                     screenshot: str | None = None,
                     movie: str | None = None,
                     show: bool = True, sensors: Any = None,
                     sensor_size: float = 1.0) -> Any:
    """A PSD's envelope on a geometry, as the app shows it: two copies
    deflected by plus and minus sqrt(PSD) at one line, colored in dB
    below the loudest node.

    One `quantity` at a time, because newtons and meters per second
    squared cannot share a normalization; by default the most common
    one among the `records`. Cross records are left out, since their
    phase belongs to an operating deflection shape and not an envelope.
    The still is at `frequency` (the first line at or above it), or at
    the strongest line when none is given.

    `screenshot` saves the still. `movie` sweeps every line once in
    `seconds`, as Play does in the window, with the line's frequency
    written on each frame, and returns the path. Otherwise a window
    plays it and is returned.

    Raises ValueError when no auto record measures the quantity, or
    when none of the records is at a node of this geometry.

    With `sensors`, a channel table, the sensors it places are drawn at
    their nodes and move with them (`viz.sensors`), `sensor_size` times
    true size.
    """
    from ..deform import (
        EnvelopeDeflection,
        animation_records,
        envelope_records,
    )
    from ..theme import theme as resolve_theme

    of_kind, quantity, _crossed, _left_out = envelope_records(
        data, records, quantity)
    if not of_kind:
        raise ValueError(f'no auto records measure {quantity!r}')
    indices, _notes = animation_records(data, of_kind)
    dofs = [data.response_dof[i] for i in indices]
    deflection = EnvelopeDeflection(geometry, dofs,
                                    np.asarray(data.ordinate)[indices].real)
    if not len(deflection.rows):
        raise ValueError('none of the records land on this geometry')
    abscissa = np.asarray(data.abscissa, dtype=float)
    line = (deflection.strongest_line if frequency is None
            else int(np.clip(np.searchsorted(abscissa, float(frequency)),
                             0, len(abscissa) - 1)))
    colors = resolve_theme(theme)

    def caption(at: Any) -> str:
        # an envelope frame without its frequency says nothing
        return f'Envelope — {abscissa[int(at)]:.5g} Hz'

    def build(plotter: Any) -> PairedAnimator:
        # the undeflected geometry first, faint — the zero reference
        add_geometry(plotter, geometry, unit_system=unit_system,
                     opacity=0.25, color_override=colors['scene_muted'],
                     node_size=4.0, line_width=1.0)
        pair = PairedAnimator(*(
            EnvelopeAnimator(plotter, geometry, deflection, sign=sign,
                             unit_system=unit_system, colormap=colormap,
                             opacity=0.6) for sign in (1.0, -1.0)))
        pair.set_scale(scale)
        pair.set_parameter(line)
        return pair

    build = _sensing(build, sensors, sensor_size)
    if screenshot is not None or movie is not None:
        return _render_offscreen(
            geometry, build, colors=colors, still=line,
            parameters=sweep_parameters(len(abscissa), seconds),
            screenshot=screenshot, movie=movie, caption=caption,
            plot=_record_plot(data, indices, unit_system, theme))
    return _play(geometry, build, colors=colors, title=caption(line),
                 show=show, caption=caption)


def animate_ods(geometry: Geometry, data: Any,
                records: Any = None, *,
                frequency: float | None = None,
                quantity: str | None = None,
                unit_system: UnitSystem | None = None,
                scale: float = 1.0, colormap: bool = True,
                cycle_seconds: float = SECONDS_PER_CYCLE,
                theme: Any = None,
                screenshot: str | None = None,
                movie: str | None = None,
                show: bool = True, sensors: Any = None,
                sensor_size: float = 1.0) -> Any:
    """The operating deflection shape of complex spectra at one line, as
    the app animates it when the object is selected with a geometry.

    The shape is the records' complex values at `frequency` (the first
    line at or above it), or at the strongest line when none is given,
    swept in phase like a complex mode. One `quantity` at a time, as
    the envelope is, by default the most common one among the
    `records` (`quantity_records`). The records follow
    `animation_records`: one record per DOF, and the first reference of a
    whole FRF. `screenshot` and `movie` save it as `animate_shape` does,
    and otherwise a window plays it.

    Raises ValueError for real data, which has no phase, when no record
    measures the quantity, and when none of the records is at a node of
    this geometry.

    With `sensors`, a channel table, the sensors it places are drawn at
    their nodes and move with them (`viz.sensors`), `sensor_size` times
    true size.
    """
    from ..deform import (
        OdsDeflection,
        animation_records,
        first_capture,
        quantity_records,
    )
    from ..theme import theme as resolve_theme

    if not np.iscomplexobj(data.ordinate):
        raise ValueError('an operating deflection needs phase — '
                         'this data is real')
    captured, _note = first_capture(data, records)
    of_kind, quantity, _left_out = quantity_records(data, captured, quantity)
    if not of_kind:
        raise ValueError(f'no records measure {quantity!r}')
    indices, _notes = animation_records(data, of_kind)
    dofs = [data.response_dof[i] for i in indices]
    deflection = OdsDeflection(geometry, dofs,
                               np.asarray(data.ordinate)[list(indices)])
    if not len(deflection.rows):
        raise ValueError('none of the records land on this geometry')
    abscissa = np.asarray(data.abscissa, dtype=float)
    deflection.line = (deflection.strongest_line if frequency is None
                       else int(np.clip(
                           np.searchsorted(abscissa, float(frequency)),
                           0, len(abscissa) - 1)))
    return _cycled(geometry, _sensing(_single(geometry, deflection,
                                              unit_system=unit_system,
                                              scale=scale, colormap=colormap),
                                      sensors, sensor_size),
                   colors=resolve_theme(theme), cycle_seconds=cycle_seconds,
                   title=f'ODS — {abscissa[deflection.line]:.5g} Hz',
                   show=show, screenshot=screenshot, movie=movie)


def animate_time(geometry: Geometry, data: Any, records: Any = None, *,
                 quantity: str | None = None, shapes: Any = None,
                 unit_system: UnitSystem | None = None,
                 scale: float = 1.0, colormap: bool = True,
                 seconds: float = SECONDS_PER_RECORD, theme: Any = None,
                 screenshot: str | None = None,
                 movie: str | None = None,
                 show: bool = True, sensors: Any = None,
                 sensor_size: float = 1.0) -> Any:
    """Time records moving on a geometry, as the app plays them: the
    record once through in `seconds`, one record per DOF
    (`animation_records`) and one `quantity`, by default the most common
    one among the `records` (`quantity_records`). A window plays it on
    a loop.

    With `screenshot` the still is the sample where some record is
    furthest from zero, because the first sample is usually at rest and
    a still of the structure at rest shows nothing. With `movie` the
    record is saved once through as an MP4 (H.264) and the path is
    returned.

    With `shapes`, a shape set, the records played are the modal
    coordinates of that set (`M1` … `Mn`, such as the modal responses
    computed from forces), and each moves the geometry by its mode's
    shape scaled by its value at each sample (`deform.modal_deflection`):
    the physical response is never formed, so a large model plays at the
    cost of its modes rather than of a record per DOF. Several modes play
    as their sum; a pick of some of them plays their contribution.

    Raises ValueError when no record measures the quantity, or none of
    the records is at a node of this geometry.

    With `sensors`, a channel table, the sensors it places are drawn at
    their nodes and move with them (`viz.sensors`), `sensor_size` times
    true size.
    """
    from ..core.validate import modal_coordinate
    from ..deform import (
        TimeDeflection,
        animation_records,
        first_capture,
        modal_deflection,
        quantity_records,
    )
    from ..theme import theme as resolve_theme

    captured, _note = first_capture(data, records)
    if shapes is not None:
        captured = [i for i in captured
                    if modal_coordinate(data.response_dof[i]) is not None]
    of_kind, quantity, _left_out = quantity_records(data, captured, quantity)
    if not of_kind:
        raise ValueError(f'no records measure {quantity!r}')
    indices, _notes = animation_records(data, of_kind)
    ordinate = np.asarray(data.ordinate)[list(indices)].real
    if shapes is not None:
        deflection = modal_deflection(geometry, shapes, data, indices)
    else:
        dofs = [data.response_dof[i] for i in indices]
        deflection = TimeDeflection(geometry, dofs, ordinate)
    if not len(deflection.rows):
        raise ValueError('none of the records land on this geometry')
    colors = resolve_theme(theme)
    samples = deflection.num_frames
    build = _sensing(_single(geometry, deflection, unit_system=unit_system,
                             scale=scale, colormap=colormap),
                     sensors, sensor_size)
    if screenshot is not None or movie is not None:
        return _render_offscreen(
            geometry, build, colors=colors,
            still=int(np.abs(ordinate).max(axis=0).argmax()),
            parameters=sweep_parameters(samples, seconds),
            screenshot=screenshot, movie=movie,
            plot=_record_plot(data, indices, unit_system, theme))
    return _play(geometry, build, colors=colors, title='Time history',
                 show=show, per_second=samples / seconds, period=samples,
                 whole=True)
