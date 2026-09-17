"""Moving a geometry scene without rebuilding it.

The scene's meshes all share one point buffer (see `shared_points`), so a
frame is: compute the offsets for the moving nodes, write them through a
numpy view of VTK's memory, and mark it modified. Nothing is allocated and
nothing is rebuilt, which is what keeps large models animatable.

Knows nothing about Qt — the caller drives it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..units import DEFAULT_SYSTEM
from .geometry import add_geometry, display_points, shared_points, shared_scalars

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.geometry import Geometry
    from ..core.shapes import ShapeSet
    from ..deform import Deflection
    from ..units import UnitSystem


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
    """Two deflections of one geometry, moving in step.

    Each copy keeps its own point buffer and its own automatic scale, so
    two mode shapes of different normalization compare by *shape* — which
    is what overlaying them is for.
    """

    def __init__(self, first: GeometryAnimator,
                 second: GeometryAnimator) -> None:
        self.first: GeometryAnimator = first
        self.second: GeometryAnimator = second
        self.axis_unit: str = first.axis_unit

    @property
    def deflection(self) -> Any:
        """The first copy's — for a pair sharing one deflection (the
        envelope), the pair's."""
        return self.first.deflection

    @property
    def parameter(self) -> float:
        return self.first.parameter

    def set_scale(self, user_scale: float) -> None:
        self.first.set_scale(user_scale)
        self.second.set_scale(user_scale)

    def set_parameter(self, parameter: float) -> None:
        self.first.set_parameter(parameter)
        self.second.set_parameter(parameter)

    def render(self) -> None:
        self.first.render()

    def reset(self) -> None:
        self.first.reset()
        self.second.reset()


class GeometryAnimator:
    """A geometry scene whose nodes move with a deflection.

    `deflection` supplies offsets for the nodes it moves; everything else
    holds its base position. `parameter` means phase in radians for a mode
    shape and a sample index for time data.
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
        self.parameter: float = 0.0
        self.set_parameter(0.0)

    @property
    def scale(self) -> float:
        return self.auto * self.user_scale

    def set_scale(self, user_scale: float) -> None:
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
        if self.magnitude is not None:
            # |offset| per node, without allocating: einsum for the sum of
            # squares, sqrt in place
            np.einsum('ij,ij->i', offsets, offsets, out=self._norms)
            np.sqrt(self._norms, out=self._norms)
            self.magnitude[self._rows] = self._norms
            self.scalars_source.Modified()

    def render(self) -> None:
        self.plotter.render()

    def reset(self) -> None:
        """Put every node back where it started."""
        self.view[:] = self.base
        self.points_source.Modified()
        if self.magnitude is not None:
            self.magnitude[:] = 0.0
            self.scalars_source.Modified()


class EnvelopeAnimator(GeometryAnimator):
    """One copy of a PSD envelope; a ± pair of these is the picture.

    `sign` mirrors the pattern through the scale, so the window's one
    Scale control still drives both copies through `PairedAnimator`.
    The parameter is the abscissa line — the envelope has no phase to
    sweep — and color is absolute: dB below the loudest node at any
    line, the one channel left to carry level once every line deflects
    at full size.
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
        return self.auto * self.user_scale * self.sign

    def set_parameter(self, parameter: float) -> None:
        self.deflection.line = int(parameter)
        super().set_parameter(parameter)
        if self.magnitude is not None:
            self.magnitude[self._rows] = self.deflection.node_decibels()
            self.scalars_source.Modified()


def animate_shape(geometry: Geometry, shapes: ShapeSet, mode: int = 0, *,
                  unit_system: UnitSystem | None = None,
                  scale: float = 1.0, colormap: bool = True,
                  cycle_seconds: float = 2.0, theme: Any = None,
                  screenshot: str | None = None,
                  show: bool = True) -> Any:
    """One mode shape moving on a geometry, as the GUI animates it.

    Colored by displacement on viridis like the desktop scene. With
    `screenshot` the deflected shape is rendered to a file at peak
    phase instead — an animation has no still to save, so the still is
    the one that shows the shape.
    """
    import numpy as np

    from ..deform import ShapeDeflection
    from ..theme import theme as resolve_theme
    from .geometry import annotate_scene

    colors = resolve_theme(theme)
    deflection = ShapeDeflection(geometry, shapes.coordinate,
                                 shapes.shape_matrix[mode])
    if screenshot is not None:
        import pyvista as pv

        plotter = pv.Plotter(off_screen=True)
        plotter.set_background(colors['scene_background'])
        animator = GeometryAnimator(plotter, geometry, deflection,
                                    unit_system=unit_system,
                                    colormap=colormap)
        animator.set_scale(scale)
        animator.set_parameter(0.0)
        annotate_scene(plotter, animator.axis_unit, colors)
        image = plotter.screenshot(str(screenshot))
        plotter.close()
        return image

    from pyvistaqt import BackgroundPlotter

    from . import undeferred

    plotter = undeferred(BackgroundPlotter(title=shapes.mode_label(mode),
                                            show=show))
    plotter.set_background(colors['scene_background'])
    animator = GeometryAnimator(plotter, geometry, deflection,
                                unit_system=unit_system, colormap=colormap)
    animator.set_scale(scale)
    annotate_scene(plotter, animator.axis_unit, colors)
    plotter.reset_camera()

    # the desktop's own two seconds a cycle, stepped on a wall clock so
    # the swing reads the same however fast the machine renders
    interval = 33
    step = 2 * np.pi * interval / (cycle_seconds * 1000.0)

    def advance() -> None:
        animator.set_parameter(animator.parameter + step)

    plotter.add_callback(advance, interval=interval)
    return plotter


def animate_envelope(geometry: Geometry, data: Any,
                     records: Any = None, *,
                     frequency: float | None = None,
                     quantity: str | None = None,
                     unit_system: UnitSystem | None = None,
                     scale: float = 1.0, colormap: bool = True,
                     theme: Any = None,
                     screenshot: str | None = None,
                     show: bool = True) -> Any:
    """A PSD's envelope on a geometry, as the GUI shows it: two copies
    deflected ±sqrt(PSD) at one line, colored dB below the loudest.

    One quantity at a time — newtons and meters per second squared
    cannot share a normalization — defaulting to the commonest among
    the records. Cross records are left out: their phase belongs to an
    operating deflection shape, not an envelope.
    """
    import numpy as np

    from ..deform import (
        EnvelopeDeflection,
        animation_records,
        envelope_records,
    )
    from ..theme import theme as resolve_theme
    from .geometry import annotate_scene

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
    caption = f'Envelope — {abscissa[line]:.5g} Hz'
    colors = resolve_theme(theme)

    def build(plotter):
        # the undeflected geometry first, faint — the zero reference
        add_geometry(plotter, geometry, unit_system=unit_system,
                     opacity=0.25, color_override=colors['scene_muted'],
                     node_size=4.0, line_width=1.0)
        first = EnvelopeAnimator(plotter, geometry, deflection, sign=1.0,
                                 unit_system=unit_system, colormap=colormap,
                                 opacity=0.6)
        pair = PairedAnimator(first, EnvelopeAnimator(
            plotter, geometry, deflection, sign=-1.0,
            unit_system=unit_system, colormap=colormap, opacity=0.6))
        pair.set_scale(scale)
        pair.set_parameter(line)
        annotate_scene(plotter, pair.axis_unit, colors)
        return pair

    if screenshot is not None:
        import pyvista as pv

        plotter = pv.Plotter(off_screen=True)
        plotter.set_background(colors['scene_background'])
        build(plotter)
        image = plotter.screenshot(str(screenshot))
        plotter.close()
        return image

    from pyvistaqt import BackgroundPlotter

    from . import undeferred

    plotter = undeferred(BackgroundPlotter(title=caption, show=show))
    plotter.set_background(colors['scene_background'])
    build(plotter)
    plotter.reset_camera()
    return plotter


def animate_ods(geometry: Geometry, data: Any,
                records: Any = None, *,
                frequency: float | None = None,
                unit_system: UnitSystem | None = None,
                scale: float = 1.0, colormap: bool = True,
                cycle_seconds: float = 2.0, theme: Any = None,
                screenshot: str | None = None,
                show: bool = True) -> Any:
    """The operating deflection shape of complex spectra at one line,
    as the GUI animates it when the object is selected with a geometry.

    The pattern is the records' complex values at the chosen frequency
    — the strongest line when none is named — swept in phase like a
    complex mode. Record choice follows `animation_records`: one record
    per DOF, first reference of a whole FRF.
    """
    import numpy as np

    from ..deform import OdsDeflection, animation_records
    from ..theme import theme as resolve_theme
    from .geometry import annotate_scene

    if not np.iscomplexobj(data.ordinate):
        raise ValueError('an operating deflection needs phase — '
                         'this data is real')
    colors = resolve_theme(theme)
    indices, _notes = animation_records(data, records)
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
    caption = f'ODS — {abscissa[deflection.line]:.5g} Hz'

    if screenshot is not None:
        import pyvista as pv

        plotter = pv.Plotter(off_screen=True)
        plotter.set_background(colors['scene_background'])
        animator = GeometryAnimator(plotter, geometry, deflection,
                                    unit_system=unit_system,
                                    colormap=colormap)
        animator.set_scale(scale)
        animator.set_parameter(0.0)
        annotate_scene(plotter, animator.axis_unit, colors)
        image = plotter.screenshot(str(screenshot))
        plotter.close()
        return image

    from pyvistaqt import BackgroundPlotter

    from . import undeferred

    plotter = undeferred(BackgroundPlotter(title=caption, show=show))
    plotter.set_background(colors['scene_background'])
    animator = GeometryAnimator(plotter, geometry, deflection,
                                unit_system=unit_system, colormap=colormap)
    animator.set_scale(scale)
    annotate_scene(plotter, animator.axis_unit, colors)
    plotter.reset_camera()

    interval = 33
    step = 2 * np.pi * interval / (cycle_seconds * 1000.0)

    def advance() -> None:
        animator.set_parameter(animator.parameter + step)

    plotter.add_callback(advance, interval=interval)
    return plotter
