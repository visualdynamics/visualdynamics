"""Readings drawn over the waterfall stage: averaging frames, shock
windows, the truncation, the filter and octave previews, and a decade
axis.

The 3-D counterpart of `plot/averaging.py` and `plot/shocks.py`. The
numbers are the same ones (`Averaging.frame_bounds`, `.levels`,
`window_shape`, the shocks' own windows), drawn as stage geometry
instead of plot items, so the two views cannot disagree about where a
frame is.

A span edge is a rimmed plane where the flat plot draws a line, and an
analyzed stretch is a translucent slab through every channel. The
averaging's window rail is drawn on the back wall, above the stage's
top, because every channel shares one time axis, so one rail serves all
of them as it does in 2-D. The averaging span, the truncation and the
shock windows carry grab disks on their front edge; this module only
draws them, and the window's dragging is `gui.stage_drag`. Nothing
here uses Qt, like the rest of `viz`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.averaging import Averaging, window_shape
from ..theme import theme as resolve_theme
from .waterfall import STAGE, _finite_runs

# The marks began read-only (Brandon, 2026-08-23), with the side panel
# as the editor in both views; the grab disks and gui/stage_drag.py were
# added later.

#: the rail's proportions, in fractions of the stage height — the same
#: shape as the 2-D rail's view fractions, restated for a stage whose
#: ceiling is fixed at STAGE's z rather than wherever the view is
RAIL_GAP = 0.06
RAIL_SLOT = 0.10
RAIL_BAND = 0.34
GLYPH_SHARE = 0.78
#: how far behind the last station the back wall stands
WALL_SET_BACK = 1.04
#: translucency of the span and shock slabs
BAND_OPACITY = 0.18
EDGE_OPACITY = 0.55
#: where a frame band's fade tops out, at the window's full weight
FADE_OPACITY = 0.45
#: window glyphs are smooth; a handful of stops draws them faithfully
GLYPH_POINTS = 64
#: the tick under each end of every window, as a fraction of the
#: glyph height — the 2-D rail's own caps, telling a tapering
#: window's stop from its neighbor's start
CAP_SHARE = 0.35
#: the grab disks on a window's floor edge, as a fraction of the
#: stage width — small enough to leave the stage draggable for the
#: camera, big enough to hit (0.022 read as oversized on Brandon's
#: screen, 2026-08-23)
HANDLE_RADIUS = 0.015


#: how far the handle disks lean toward the viewer, so their
#: engravings read at the stage's working camera angle instead of
#: foreshortening flat on the floor
TILT = np.deg2rad(40.0)


def _tilted(x, local, seat, forward=0.0):
    """Points in a disk's own plane, leaned back by TILT about x.

    `local` rows are (dx, dy, dz) around the disk center; the disk
    sits with its lowest rim grazing the floor, pushed `forward`
    (negative y) so it stands clear of the slab it adjusts — half a
    disk buried in the translucent box was half a disk that could
    not be clicked (Brandon, 2026-08-24).
    """
    s, c = np.sin(TILT), np.cos(TILT)
    local = np.asarray(local, dtype=np.float64)
    out = np.empty_like(local)
    out[:, 0] = x + local[:, 0]
    out[:, 1] = forward + local[:, 1] * c - local[:, 2] * s
    out[:, 2] = seat + local[:, 1] * s + local[:, 2] * c
    return out.astype(np.float32)


def _handle_icon(role, x, radius, lift, seat, forward):
    """The engraving on a handle's disk, in the disk's own plane.

    An arrow on an edge handle, pointing the way that edge extends
    the window; a three-line grip on the center, the classic 'this
    slides' glyph (Brandon, 2026-08-24).
    """
    import pyvista as pv

    if role == 'move':
        points, lines = [], []
        for dy in (-0.35, 0.0, 0.35):
            first = len(points)
            points.extend([[-0.5 * radius, dy * radius, lift],
                           [0.5 * radius, dy * radius, lift]])
            lines.extend([2, first, first + 1])
        return pv.PolyData(_tilted(x, points, seat, forward),
                           lines=lines)
    sign = -1.0 if role in ('start', 'open') else 1.0
    points = [[sign * 0.55 * radius, 0.0, lift],
              [-sign * 0.30 * radius, -0.45 * radius, lift],
              [-sign * 0.30 * radius, 0.45 * radius, lift]]
    return pv.PolyData(_tilted(x, points, seat, forward),
                       faces=[3, 0, 1, 2])


def _handles(plotter, xl, xh, colors, prefix, roles=('start', 'move',
                                                     'stop')):
    """Three grab disks on the slab's front floor edge: left edge,
    center, right edge — resize, move, resize, exactly the 2-D
    region's own grammar. Gray buttons with engraved icons rather
    than bare spheres (Brandon, 2026-08-24); the icons are not
    pickable, so a click on the engraving grabs the disk under it.
    """
    import pyvista as pv

    sx = STAGE[0]
    radius = HANDLE_RADIUS * sx
    seat = radius * float(np.sin(TILT)) + 0.003 * sx
    # wholly in front of the stage: the rearmost rim grazes y = 0
    forward = -(radius * float(np.cos(TILT))) - 0.004 * sx
    for role, x in zip(roles, (xl, (xl + xh) / 2.0, xh)):
        x = float(x)
        rim = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        disk_local = np.column_stack([
            np.append(0.0, radius * np.cos(rim)),
            np.append(0.0, radius * np.sin(rim)),
            np.zeros(len(rim) + 1)])
        faces = []
        for k in range(len(rim)):
            faces.extend([3, 0, 1 + k, 1 + (k + 1) % len(rim)])
        disk = pv.PolyData(_tilted(x, disk_local, seat, forward),
                           faces=faces)
        _add(plotter, disk, color=colors['specification_curve'],
             pickable=True, name=f'{prefix}-handle-{role}')
        _add(plotter,
             _handle_icon(role, x, radius, 0.006 * sx, seat, forward),
             color=colors['scene_background'], line_width=3,
             name=f'{prefix}-handle-{role}-icon')


def _add(plotter, mesh, **kwargs):
    """`add_mesh`, with the actor excluded from the scene's bounds.

    The rail rides above the stage ceiling and the wall stands behind
    the last station, and pyvista refits the cube axes to the scene
    every time an actor arrives — so the first mark stretched the
    axes' box while their labeled ranges stayed put, and the time
    axis read wrong values (Brandon, 2026-08-24). Marks describe the
    stage; they must never resize it. The exclusion lands after the
    add (the refit inside `add_mesh` has already seen the actor), so
    `_repin_axes` restores the box once the marks are all in.
    """
    # one render per drawing, never one per actor: each add briefly
    # holds the axes in their inflated state (the refit runs before
    # the exclusion can land), and add_mesh's default render=True put
    # every one of those intermediate frames on screen — a drag
    # preview visibly stuttered through them before the repin's
    # corrected frame (Brandon, 2026-08-24). The caller renders once,
    # after the marks are whole.
    kwargs.setdefault('render', False)
    # marks are drawings, not drag targets — only the handle disks
    # opt back in, so a click near a slab can never pick the slab
    # instead of the handle behind it
    kwargs.setdefault('pickable', False)
    actor = plotter.add_mesh(mesh, **kwargs)
    actor.UseBoundsOff()
    return actor


def _repin_axes(plotter, extents):
    """Refit the cube axes to the actors that still count, and restate
    the labeled ranges from the extents the stage was drawn with.

    `ComputeVisiblePropBounds` skips everything `UseBoundsOff` marked,
    so this undoes the stretch the adds above caused — pyvista's own
    refit only ever saw the marks while they still counted. The axes
    actor itself is stood down for the measurement: it is a prop too,
    and it *claims the inflated box* — computing bounds with it in
    reads the stale answer straight back.

    The ranges cannot be read and restored: pyvista's per-add refit
    rewrites them to the bounds long before this runs, so a read-back
    restores the corruption (the shock project's time axis went
    0..1.6 stage units — Brandon, 2026-08-24). The extents are the
    same authority `show_bounds` was given, so the labels go back to
    the truth rather than to whatever the refit left. The depth
    axis's labels are hidden (the channel names on the curves are its
    labels), so its range stays wherever it is.
    """
    axes = getattr(plotter.renderer, 'cube_axes_actor', None)
    if axes is None:
        return
    axes.UseBoundsOff()
    try:
        bounds = plotter.renderer.ComputeVisiblePropBounds()
    finally:
        axes.UseBoundsOn()
    # through pyvista's own setters, never the raw VTK ones: pyvista
    # builds the drawn label text itself in the property setters, so
    # SetXAxisRange leaves GetXAxisRange *reading* the truth while the
    # screen keeps showing the stale strings — every probe that read
    # the range back said fine, and Brandon's screenshot said 1.6
    x0, x1, z0, z1 = extents
    axes.bounds = bounds
    axes.x_axis_range = (x0, x1)
    axes.z_axis_range = (z0, z1)


def _hold(plotter, mesh):
    """Keep a scalar-mapped mesh alive for the plotter's lifetime.

    pyvista's mapper does not take ownership of a mesh added with
    ``scalars``: drop the last Python reference and the *next*
    ``add_mesh`` call collects it, leaving the actor drawing an empty
    dataset. Found the hard way — a debugging spy that captured the
    call's arguments fixed the bug by accidentally keeping this very
    reference. Plain-color meshes are unaffected.
    """
    held = getattr(plotter, '_marks_held', None)
    if held is None:
        held = plotter._marks_held = []
    held.append(mesh)


def _mapper(extents):
    """Seconds -> stage x, from the extents the stage was drawn with.

    Time is seconds in every display system, so the averaging's and
    the shocks' own seconds land directly on the drawn axis.
    """
    x0, x1, _z0, _z1 = extents
    sx = STAGE[0]
    span = (x1 - x0) or 1.0
    return lambda t: (np.asarray(t, dtype=float) - x0) / span * sx


def _edge_rim(plotter, xn, color, name):
    """A span edge's outline: the flat plot's vertical line, given
    depth — drawn as the rim of the slab it bounds."""
    import pyvista as pv

    _sx, sy, sz = STAGE
    corners = np.array([[xn, 0.0, 0.0], [xn, sy, 0.0],
                        [xn, sy, sz], [xn, 0.0, sz]], dtype=np.float32)
    rim = pv.lines_from_points(np.vstack([corners, corners[:1]]))
    _add(plotter, rim, color=color, opacity=EDGE_OPACITY,
                     line_width=2, name=name)


def averaging_stage_geometry(averaging: Averaging, sample_rate: float,
                             extents: Sequence[float],
                             origin: float = 0.0) -> dict[str, Any]:
    """The averaging as plain stage geometry, with no VTK and no Qt.

    These are the numbers that `add_averaging_marks` puts into a plotter
    and that the report's canvas figure draws for itself: the analyzed
    span, and for each frame the rail's baseline, the stage x of the
    window's sampled points, their weights, the glyph height at each,
    and the x of the two end caps. They are computed in one place so the
    two views cannot disagree about where a frame is.

    `extents` are the stage's real ranges (x0, x1, z0, z1), and
    `origin` is the record's start time in seconds, since frames are
    placed on the record's own clock. Everything returned is in stage
    coordinates: x across, ``wall`` for the depth of the back wall the
    rail is drawn on, z up. Returns ``span``, ``wall``, ``cap`` (the end
    caps' half height), ``fade`` (`FADE_OPACITY`) and ``frames``.
    """
    to_x = _mapper(extents)
    _sx, sy, sz = STAGE
    wall = sy * WALL_SET_BACK
    # on the record's clock, like everything the stage draws
    # (`Averaging.stop` on the origin)
    first = origin + averaging.start_sample(sample_rate) / sample_rate
    last = averaging.stop(sample_rate, origin)
    levels, count = averaging.levels(sample_rate)
    slot = min(RAIL_SLOT, RAIL_BAND / count) * sz
    glyph = slot * GLYPH_SHARE
    foot = sz * (1.0 + RAIL_GAP)
    n = averaging.frame_length
    shape = window_shape(averaging.window, n,
                         averaging.window_parameter)
    # the glyph is smooth: a fixed sampling draws it without carrying
    # a vertex per sample of a long frame onto the stage
    at = np.linspace(0, n - 1, min(GLYPH_POINTS, n)).astype(int)
    weights = shape[at]
    frames = []
    for (low, high), level in zip(
            averaging.frame_bounds(sample_rate, origin), levels):
        baseline = foot + level * slot
        frames.append({
            'baseline': float(baseline),
            'xs': [float(v) for v in to_x(low + at / sample_rate)],
            'weights': [float(w) for w in weights],
            'glyph': [float(baseline + w * glyph) for w in weights],
            'caps': [float(to_x(low)), float(to_x(high))]})
    return {'span': [float(to_x(first)), float(to_x(last))],
            'wall': float(wall), 'cap': float(glyph * CAP_SHARE),
            'fade': FADE_OPACITY, 'frames': frames}


def shock_stage_geometry(shocks: Sequence[Any],
                         extents: Sequence[float],
                         origin: float = 0.0) -> dict[str, Any]:
    """Each shock's analysis window as plain stage geometry: a dict whose
    ``windows`` each hold the slab's ``span`` in stage x and its
    ``label``, the event's number counted from 1. The 3-D counterpart of
    the regions in `plot/shocks.py`, used by both the app's stage and
    the report's canvas, as the averaging's geometry is. The shocks'
    times are measured from `origin`, the record's start."""
    to_x = _mapper(extents)
    return {'windows': [
        {'span': [float(to_x(origin + shock.start)),
                  float(to_x(origin + shock.stop))],
         'label': str(index + 1)}
        for index, shock in enumerate(shocks)]}


def add_averaging_marks(plotter: Any, averaging: Averaging,
                        sample_rate: float, extents: Sequence[float],
                        theme: Any = None,
                        origin: float = 0.0) -> dict[str, int]:
    """Draw the averaging on the stage: the span, and each frame's band
    and window on the rail.

    The analyzed span is one translucent slab with rimmed edges and grab
    disks, drawn the same way as a shock's window, because both show
    which stretch of the record is analyzed. On the back wall each
    frame has a band shaded by the window's weight, the window's shape,
    and a tick at each end. Arguments are those of
    `averaging_stage_geometry`. Returns counts for a test to check:
    ``frames`` drawn on the rail, and ``edges``, always two (the span's
    start and stop rims).
    """
    # One slab like a shock window (Brandon, 2026-08-24): two lone edge
    # planes read as different objects.
    import pyvista as pv

    colors = resolve_theme(theme)
    geometry = averaging_stage_geometry(averaging, sample_rate, extents,
                                        origin)
    _sx, sy, sz = STAGE
    wall = geometry['wall']

    x_first, x_last = geometry['span']
    span = pv.Box(bounds=(x_first, x_last, 0.0, sy, 0.0, sz))
    _add(plotter, span, color=colors['averaging_band'],
                     opacity=BAND_OPACITY, name='marks-averaging-span')
    for which, xn in (('start', x_first), ('stop', x_last)):
        _edge_rim(plotter, xn, colors['averaging_window'],
                  f'marks-averaging-edge-{which}')
    _handles(plotter, x_first, x_last, colors, 'marks-averaging')

    band_points, band_faces, band_weights = [], [], []
    glyph_points, glyph_lines = [], []
    cap_points, cap_lines = [], []
    cap = geometry['cap']
    total = 0
    for frame in geometry['frames']:
        baseline = frame['baseline']
        xs = np.asarray(frame['xs'])
        weights = np.asarray(frame['weights'])
        # the band is subdivided along the frame and carries the
        # window's own value per column, so its shading *is* the
        # weight each moment carries — the 2-D gradient brush, as
        # per-point translucency (Brandon, 2026-08-24)
        base = len(band_points)
        for x, w in zip(xs, weights):
            band_points.extend([[x, wall, 0.0], [x, wall, baseline]])
            band_weights.extend([w, w])
        for column in range(len(xs) - 1):
            a = base + 2 * column
            band_faces.extend([4, a, a + 2, a + 3, a + 1])
        start = len(glyph_points)
        glyph_points.extend(np.column_stack(
            [xs, np.full(len(xs), wall), frame['glyph']]).tolist())
        glyph_lines.append(np.concatenate(
            [[len(xs)], np.arange(start, start + len(xs))]))
        # a tick at each end of the window — a tapering window comes
        # back to its baseline, so without these nothing says where a
        # hann frame stopped and the next began (the 2-D rail's caps)
        for xe in frame['caps']:
            first = len(cap_points)
            cap_points.extend([[xe, wall, baseline - cap],
                               [xe, wall, baseline + cap]])
            cap_lines.append([2, first, first + 1])
        total += 1

    if total:
        bands = pv.PolyData(np.asarray(band_points, dtype=np.float32),
                            faces=band_faces)
        weight = np.asarray(band_weights)
        bands.point_data['weight'] = weight
        # one color, translucency by weight: zero vanishes, the
        # window's full value stands at the fade ceiling. Baked as
        # per-point RGBA rather than a cmap + opacity transfer —
        # that path leaves the mapper without ownership of the mesh,
        # and the actor emptied the moment this local went out of
        # scope (found by a spy wrapper that fixed the bug by
        # accidentally keeping a reference)
        rgba = np.empty((len(weight), 4), dtype=np.uint8)
        rgba[:, :3] = pv.Color(colors['averaging_band']).int_rgb
        # the magnitude, not the signed value (Brandon, 2026-08-24): a
        # flattop's shoulders dip slightly negative, and a negatively
        # weighted moment still carries weight — its energy enters the
        # average as the square, sign gone. Cast signed, the negative
        # alpha wrapped through uint8 to ~255 and the band went solid
        # exactly where the window nearly vanishes; |w| shades it at
        # the few counts it deserves. Clipped above at one for any
        # window that overshoots.
        rgba[:, 3] = np.round(np.clip(np.abs(weight), 0.0, 1.0)
                              * FADE_OPACITY * 255).astype(np.uint8)
        bands.point_data['fade'] = rgba
        _hold(plotter, bands)
        _add(plotter, bands, scalars='fade', rgba=True,
                         name='marks-averaging-bands')
        glyphs = pv.PolyData(np.asarray(glyph_points, dtype=np.float32),
                             lines=np.concatenate(glyph_lines))
        _add(plotter, glyphs, color=colors['averaging_window'],
                         line_width=2, name='marks-averaging-windows')
        caps = pv.PolyData(np.asarray(cap_points, dtype=np.float32),
                           lines=np.concatenate(cap_lines))
        _add(plotter, caps, color=colors['averaging_window'],
                         line_width=2, name='marks-averaging-caps')
    _repin_axes(plotter, extents)
    return {'frames': total, 'edges': 2}


def add_truncation_marks(plotter: Any, truncation: Any,
                         first: float, last: float,
                         extents: Sequence[float],
                         theme: Any = None) -> dict[str, int]:
    """Draw the truncation on the stage: the discarded ends gray, the
    kept stretch clear, with grab disks on its edges.

    Each end that will be cut gets a gray translucent slab, gray as in
    the flat overlay because the data cut away is only for reference.
    There is a rim at each cut time, and the kept stretch has the same
    three grab disks as the averaging span (resize, move, resize).
    `first` and `last` are the record's first and last times, and an
    end with nothing cut gets no slab. Returns ``shaded``, how many end
    slabs were drawn, and ``edges``, always two.
    """
    import pyvista as pv

    colors = resolve_theme(theme)
    to_x = _mapper(extents)
    _sx, sy, sz = STAGE
    x_start = float(to_x(truncation.start))
    x_stop = float(to_x(truncation.stop))
    x_first = float(to_x(first))
    x_last = float(to_x(last))
    shaded = 0
    for name, (xa, xb) in (('head', (x_first, x_start)),
                           ('tail', (x_stop, x_last))):
        if xb - xa <= 0.0:
            # nothing discarded at this end: no slab — a zero-width
            # box would still draw a seam at the wall. Removed by
            # name, not merely skipped: a drag preview redraws these
            # marks without clearing the scene, so a slab from the
            # last preview would linger at its old width
            plotter.remove_actor(f'marks-truncation-{name}')
            continue
        slab = pv.Box(bounds=(xa, xb, 0.0, sy, 0.0, sz))
        _add(plotter, slab, color=colors['specification_curve'],
             opacity=BAND_OPACITY, name=f'marks-truncation-{name}')
        shaded += 1
    for which, xn in (('start', x_start), ('stop', x_stop)):
        _edge_rim(plotter, xn, colors['plot_foreground'],
                  f'marks-truncation-edge-{which}')
    _handles(plotter, x_start, x_stop, colors, 'marks-truncation')
    _repin_axes(plotter, extents)
    return {'shaded': shaded, 'edges': 2}


def add_shock_marks(plotter: Any, shocks: Sequence[Any],
                    extents: Sequence[float],
                    theme: Any = None,
                    locked: bool = False,
                    origin: float = 0.0) -> dict[str, int]:
    """Draw each shock's analysis window as a numbered translucent slab.

    The flat plot marks a window with a region. Here the slab spans
    every channel, because the window does. `locked` leaves off the grab
    disks, as the 2-D regions become fixed, for a record the controller
    already cut into frames, whose windows are not to be changed.
    `origin` is the record's start time. Returns ``windows``, how many
    were drawn.
    """
    import pyvista as pv

    colors = resolve_theme(theme)
    _sx, sy, sz = STAGE
    spots, names = [], []
    for index, window in enumerate(
            shock_stage_geometry(shocks, extents, origin)['windows']):
        xl, xh = window['span']
        slab = pv.Box(bounds=(xl, xh, 0.0, sy, 0.0, sz))
        _add(plotter, slab, color=colors['averaging_band'],
                         opacity=BAND_OPACITY,
                         name=f'marks-shock-{index}')
        for which, x in (('open', xl), ('close', xh)):
            rim = pv.lines_from_points(np.array(
                [[x, 0.0, 0.0], [x, sy, 0.0], [x, sy, sz],
                 [x, 0.0, sz], [x, 0.0, 0.0]], dtype=np.float32))
            _add(plotter, rim, color=colors['averaging_window'],
                             opacity=EDGE_OPACITY, line_width=2,
                             name=f'marks-shock-{index}-{which}')
        if not locked:
            _handles(plotter, xl, xh, colors,
                     f'marks-shock-{index}',
                     roles=('open', 'move', 'close'))
        spots.append([(xl + xh) / 2.0, sy / 2.0, sz * 1.05])
        names.append(f'{index + 1}')
    if names:
        plotter.add_point_labels(
            np.asarray(spots), names, font_size=12, always_visible=True,
            text_color=colors['scene_text'], shape=None, fill_shape=False,
            show_points=False, name='marks-shock-numbers', render=False)
    _repin_axes(plotter, extents)
    return {'windows': len(names)}


def add_filter_preview(plotter: Any, arrays: dict[str, Any],
                       extents: Sequence[float], stations: int,
                       theme: Any = None) -> dict[str, int]:
    """The filter previewed on the stage: each record's filtered copy,
    drawn over the raw curve at that record's own station.

    The filtered data is drawn on the level colormap, as an unfiltered
    stage is, and the raw record is drawn in gray behind it
    (`add_waterfall`'s `color`). That is the order the paired stage
    uses: the curve being judged is in color and its reference is gray.

    `arrays` is the filtered data's `waterfall_arrays`. It is
    normalized against the raw stage's `extents`, passed in and not
    recomputed, because the filtered curves would otherwise be scaled to
    their own range: a filtered record is quieter than the raw one, and
    stretched to the same height the comparison would show nothing. The
    color scale is fixed to those extents for the same reason, so a
    level has the same height and color on either curve. `stations` is
    how many stations the raw stage has, so each copy lands at the same
    depth even when the filter turns every sample of a record into a
    gap.

    Returns ``runs``, the polylines drawn, and ``points``, their
    vertices.
    """
    # The first version had it the other way round, raw in viridis and
    # the filtered copy in one flat color, which asked the reader to
    # judge the filtered record from the duller of the two lines; and
    # the banded stage showed the same day that two things drawn
    # together must be normalized together (Brandon, 2026-08-25).
    import pyvista as pv

    x0, x1, z0, z1 = (float(v) for v in extents)
    sx, sy, sz = STAGE
    xspan = (x1 - x0) or 1.0
    zspan = (z1 - z0) or 1.0
    points: list[np.ndarray] = []
    lines: list[np.ndarray] = []
    levels: list[np.ndarray] = []
    total = 0
    for k, (cx, cz) in enumerate(arrays['curves']):
        station = (k / max(stations - 1, 1)) * sy
        values = np.asarray(cz, dtype=float)
        xn = (np.asarray(cx, dtype=float) - x0) / xspan * sx
        zn = (values - z0) / zspan * sz
        for start, stop in _finite_runs(values):
            count = stop - start
            points.append(np.column_stack([
                xn[start:stop], np.full(count, station), zn[start:stop]]))
            lines.append(np.concatenate(
                [[count], np.arange(total, total + count)]))
            levels.append(values[start:stop])
            total += count
    if not total:
        return {'runs': 0, 'points': 0}
    mesh = pv.PolyData(np.vstack(points).astype(np.float32),
                       lines=np.concatenate(lines))
    mesh.point_data['level'] = np.concatenate(levels)
    _add(plotter, mesh, scalars='level', cmap='viridis', line_width=2,
         clim=(z0, z1), show_scalar_bar=False, name='marks-filter')
    _repin_axes(plotter, extents)
    return {'runs': len(lines), 'points': total}


def decade_labels(x0: float, x1: float) -> list[tuple[float, str]]:
    """The decades inside a log10 range [`x0`, `x1`], as (exponent,
    label) pairs, each labeled as the 2-D plot labels it: ``1`` for
    10⁰, and ``10¹``, ``10²`` and so on otherwise."""
    from ..units import _superscript

    low, high = int(np.ceil(float(x0) - 1e-9)), int(np.floor(float(x1) + 1e-9))
    return [(float(n), '1' if n == 0 else f'10{_superscript(str(n))}')
            for n in range(low, high + 1)]


def add_decade_axis(plotter: Any, extents: Sequence[float],
                    theme: Any = None) -> dict[str, Any]:
    """The frequency axis in decades, drawn as the 2-D plot draws it: a
    grid line and a label at every power of ten, and nothing at the even
    divisions of the range.

    The cube axes can only label even divisions of a range, so on a log
    axis they would print exponents such as -0.6, 0.45 and 1.5, which
    match nothing on the flat plot. Their frequency labels, ticks and
    grid lines are turned off, and the decades are drawn as stage
    geometry: a line up the back wall and across the floor at each, with
    its label below the front edge, in the same way the depth axis shows
    channel names instead of numbers. `extents` are the stage's
    ``(x0, x1, z0, z1)``, with the x range in log10.

    Returns ``labels``, the labels drawn, and ``exponents``, their
    decades; with no decade in the range, only an empty ``labels``.
    """
    # Brandon, 2026-09-05: the cube axes' exponents at -0.6, 0.45, 1.5
    # matched nothing on the flat plot.
    import pyvista as pv

    colors = resolve_theme(theme)
    x0, x1, _z0, _z1 = (float(v) for v in extents)
    sx, sy, sz = STAGE
    xspan = (x1 - x0) or 1.0
    axes = getattr(plotter.renderer, 'cube_axes_actor', None)
    if axes is not None:
        axes.x_label_visibility = False
        # VTK's own call, not a snake-case name: pyvista defines no
        # `x_axis_tick_visibility` on its CubeAxesActor (only the
        # minor-tick one), and the assignment reached VTK's property
        # only through the wrappers' alias — which a Windows install of
        # pyvista 0.49.0 with vtk 9.7.0 refused with a
        # PyVistaAttributeError, taking every banded stage render down
        # (Kevin Cross, 2026-09-19, by mail; not reproduced on macOS
        # with the same pair, nor on Linux CI)
        axes.XAxisTickVisibilityOff()
        axes.x_axis_minor_tick_visibility = False
        axes.SetDrawXGridlines(False)
    decades = decade_labels(x0, x1)
    if not decades:
        return {'labels': []}
    points, lines, spots, names = [], [], [], []
    for k, (exponent, label) in enumerate(decades):
        xn = (exponent - x0) / xspan * sx
        base = 4 * k
        # up the back wall, then across the floor to the front edge
        points.extend([[xn, sy, 0.0], [xn, sy, sz], [xn, 0.0, 0.0],
                       [xn, sy, 0.0]])
        lines.extend([2, base, base + 1, 2, base + 2, base + 3])
        spots.append([xn, -0.06 * sy, 0.0])
        names.append(label)
    mesh = pv.PolyData(np.asarray(points, dtype=np.float32),
                       lines=np.asarray(lines))
    grid = pv.Color(colors['scene_text'], opacity=0.35)
    _add(plotter, mesh, color=grid, opacity=0.35, line_width=1,
         name='decade-lines')
    plotter.add_point_labels(
        np.asarray(spots), names, font_size=12, always_visible=True,
        text_color=colors['scene_text'], shape=None, fill_shape=False,
        show_points=False, name='decade-labels', render=False)
    _repin_axes(plotter, extents)
    return {'labels': names, 'exponents': [e for e, _l in decades]}


def add_octave_preview(plotter: Any, curves: Sequence[tuple[Any, Any]],
                       extents: Sequence[float], stations: int,
                       theme: Any = None) -> dict[str, int]:
    """The octave-band conversion previewed on the stage: each drawn
    record's steps at that record's own station, in the flat plot's
    preview color. It is one flat color because the steps are a proposed
    result drawn over the data, not data.

    `curves` is one ``(x, z)`` pair per drawn record, already in the
    stage's own values (log10 where the axes are; `octave_steps` with
    `stage`). They are normalized against the raw stage's `extents`, as
    the filter preview is, because two things drawn together must be
    normalized together, and `stations` is how many stations the raw
    stage has. Returns ``runs``, the polylines drawn, and ``points``,
    their vertices.
    """
    import pyvista as pv

    colors = resolve_theme(theme)
    x0, x1, z0, z1 = (float(v) for v in extents)
    sx, sy, sz = STAGE
    xspan = (x1 - x0) or 1.0
    zspan = (z1 - z0) or 1.0
    points: list[np.ndarray] = []
    lines: list[np.ndarray] = []
    total = 0
    for k, (cx, cz) in enumerate(curves):
        station = (k / max(stations - 1, 1)) * sy
        values = np.asarray(cz, dtype=float)
        xn = (np.asarray(cx, dtype=float) - x0) / xspan * sx
        zn = (values - z0) / zspan * sz
        for start, stop in _finite_runs(values):
            count = stop - start
            points.append(np.column_stack([
                xn[start:stop], np.full(count, station), zn[start:stop]]))
            lines.append(np.concatenate(
                [[count], np.arange(total, total + count)]))
            total += count
    if not total:
        return {'runs': 0, 'points': 0}
    mesh = pv.PolyData(np.vstack(points).astype(np.float32),
                       lines=np.concatenate(lines))
    _add(plotter, mesh, color=colors['filter_preview'], line_width=2,
         name='marks-octave')
    _repin_axes(plotter, extents)
    return {'runs': len(lines), 'points': total}


# ---- the readings, for a script as the window draws them -----------------

#: the readings a plot's toggles draw over the data — flat
#: (`plot_data(marks=...)`) and on the stage (`plot_waterfall(marks=...)`)
READINGS = ('averaging', 'shocks', 'truncation', 'filter', 'octave')


def held_reading(data: Any, reading: str) -> Any:
    """What `reading` (one of `READINGS`) draws for `data`, as the window
    opens on it: the setting the object carries, or else what its
    detector suggests. That is the averaging frames
    (`TimeHistory.default_averaging`), the shock windows, the
    truncation or the filter of a time history. 'octave' reads a PSD and
    has no setting here, so it returns None.

    Raises
    ------
    ValueError
        For a reading that is not one.
    TypeError
        For a reading of the wrong kind of object.
    """
    from ..core.data import Psd, TimeHistory

    if reading not in READINGS:
        raise ValueError(f'{reading!r} is not a reading: '
                         + ', '.join(repr(r) for r in READINGS))
    if reading == 'octave':
        if not isinstance(data, Psd):
            raise TypeError('octave bands read a PSD, not '
                            f'{type(data).__name__}')
        return None
    if not isinstance(data, TimeHistory):
        raise TypeError(f'{reading} marks belong on a time history, not '
                        f'on {type(data).__name__}')
    if reading == 'averaging':
        return data.default_averaging()
    if reading == 'shocks':
        from ..core.shocks import suggest

        return tuple(data.shocks or suggest(data))
    if reading == 'truncation':
        return data.truncation or data.suggest_truncation()
    return data.filtering or data.suggest_filtering()


def octave_steps(psd: Any, per_octave: int | None = None,
                 unit_system: Any = None, *, stage: bool = False
                 ) -> tuple[Any, np.ndarray, np.ndarray]:
    """The octave-band conversion of `psd` as step outlines, to draw over
    the narrowband spectrum it integrates.

    Returns (banded, x, rows): the banded PSD (`Psd.to_octave` with
    `per_octave`), and the outline's abscissa and rows, the rows as
    magnitudes in the display units of `unit_system`. With `stage` they
    are transformed the way `waterfall_arrays` transforms the stage they
    are drawn on: log10 of the frequency and of the level where those
    axes are logarithmic. The window's flat and 3-D previews and a
    script all draw these same steps.

    Raises
    ------
    ValueError
        When the spectrum cannot be banded (`Psd.to_octave`'s refusal).
    """
    from ..plot import step_outline

    banded = psd.to_octave(per_octave)
    x, rows = step_outline(banded.abscissa,
                           banded.display_ordinate(unit_system),
                           banded.bin_edges())
    rows = np.abs(np.atleast_2d(rows))
    if stage:
        if getattr(banded, 'log_abscissa', False):
            x = np.log10(np.where(x > 0, x, np.nan))
        if banded.log_scaled():
            rows = np.log10(np.where(rows > 0, rows, np.nan))
    return banded, x, rows


def stage_filter_preview(plotter: Any, data: Any, filtering: Any,
                         info: dict[str, Any], unit_system: Any = None,
                         theme: Any = None) -> None:
    """Filter a staged record with `filtering` and draw the filtered copy
    over its stage (`add_filter_preview`), against the raw stage's
    extents and at the same records' stations. `info` is what
    `add_waterfall` returned for the raw stage. This is the window's 3-D
    filter view, and `plot_waterfall(marks='filter')`.

    Raises
    ------
    ValueError
        When the filter cannot run on this record (a corner past
        Nyquist).
    """
    from ..core.filters import filtered
    from .waterfall import waterfall_arrays

    twin = waterfall_arrays(filtered(data, filtering), info['drawn'],
                            unit_system, page=0)
    add_filter_preview(plotter, twin, info['extents'], info['stations'],
                       theme=theme)


def add_stage_reading(plotter: Any, data: Any, info: dict[str, Any],
                      reading: str, *, per_octave: int | None = None,
                      unit_system: Any = None, theme: Any = None) -> None:
    """Draw one reading (one of `READINGS`) of a staged record over its
    stage, as the window's toggles draw it, with the setting from
    `held_reading`. `info` is what `add_waterfall` returned for the
    stage, and 'octave' is banded `per_octave`. The grab disks are drawn
    as in the window, but nothing here makes them draggable (that is
    `gui.stage_drag`).
    """
    held = held_reading(data, reading)
    extents = info['extents']
    if reading == 'octave':
        _banded, x, rows = octave_steps(data, per_octave, unit_system,
                                        stage=True)
        add_octave_preview(plotter, [(x, rows[k]) for k in info['drawn']],
                           extents, info['stations'], theme=theme)
    elif reading == 'filter':
        stage_filter_preview(plotter, data, held, info, unit_system, theme)
    elif reading == 'truncation':
        abscissa = np.asarray(data.abscissa, dtype=float)
        add_truncation_marks(plotter, held, float(abscissa[0]),
                             float(abscissa[-1]), extents, theme=theme)
    elif reading == 'averaging':
        add_averaging_marks(plotter, held, data.sample_rate, extents,
                            theme=theme, origin=float(data.abscissa[0]))
    else:
        add_shock_marks(plotter, held, extents, theme=theme,
                        locked=bool(data.split_into_frames),
                        origin=float(data.abscissa[0]))
