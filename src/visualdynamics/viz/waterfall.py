"""The 3-D reading of a data object: its records as a waterfall.

Many channels on one 2-D axis hide each other exactly where it matters
most — resonances line up, and the tenth curve lands on the first nine.
The waterfall spreads the records along a depth axis instead: one
polyline per record at its own station, wearing the same label the
grids and the legend use, colored by level so a peak reads across the
whole set at once.

numpy and PyVista only — no Qt, like the rest of `viz`. Two decisions
carry the size ceiling:

- **Points are peak-decimated per record** before VTK sees them
  (`decimate.peak_decimate`, the same reading the 2-D plot and the
  report use), so a million-sample time history arrives as the few
  thousand points that keep every peak.
- **The whole set of curves is one PolyData under one actor**, whatever
  the record count — per-record actors are what make large scenes slow,
  which the animation learned at 202k nodes.

The scene is drawn on a fixed stage (`STAGE`) rather than in data
coordinates — a PSD spans decades while its frequency axis spans
kilohertz, and raw coordinates would draw a needle — and the axes are
relabeled with the real ranges (`axes_ranges`), the vertical one in
log10 for data that reads on a log axis, which the axis title says.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..decimate import peak_decimate, peak_decimate_rows
from ..plot import as_power_law, drawing_shape, step_outline
from ..theme import theme as resolve_theme
from ..units import DEFAULT_SYSTEM, UNKNOWN
from ._quiet import one_render

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..core.data import DataArray
    from ..units import UnitSystem

#: points a record may bring to the scene after decimation. 4096 keeps
#: every peak a screen could show and puts a 320-record stack of
#: million-sample records at ~1.3M line vertices, which VTK renders in
#: tenths of a second — the whole pipeline for that 2.6 GB worst case
#: is ~4 s, measured, and nearly all of it is converting the block to
#: display units, which any reading of the object pays.
POINT_BUDGET = 4096

#: vertices one scene may hold, across every record on it. Measured
#: on this machine: 10M builds in ~0.3 s, draws in ~0.6 s and costs
#: ~2 GB; 30M costs 5.5 GB and 65M costs 9.1 GB, so **memory is the
#: wall, not framerate** — time stays sub-second-ish well past the
#: point RAM stops being reasonable.
#:
#: Spent as a *record cap*, never as more decimation. Thinning a
#: stepped curve is what destroyed the treads and the area with them,
#: so what is drawn is always drawn exactly; what does not fit goes
#: onto the next page instead, and the bar says which page you are on.
#: A full CPSD from a 30-channel survey is 900 records — an ordinary
#: modal survey, not a stress test, and 29.5M vertices at 16k lines.
SCENE_BUDGET = 10_000_000

#: the stage everything is normalized onto: abscissa span, depth span,
#: height. Wide and shallow, so the abscissa stays the axis being read.
STAGE = (1.6, 0.9, 0.5)

#: at most this many channel-name labels; past it, every nth record is
#: named. Naming all of a 900-record CPSD letters every curve into an
#: unreadable smear, and thinning names is honest where thinning
#: *curves* would not be — the unnamed ones are still drawn.
LABEL_LIMIT = 40


def waterfall_groups(data: DataArray,
                     records: Sequence[int] | None = None
                     ) -> list[tuple[tuple[str, str | None], list[int]]]:
    """Every (dimension, hint) group among the records asked for.

    In the object's own order — what the quantity box offers, and what
    `waterfall_group` chooses from. Same grouping key as the 2-D plot,
    minus the per-plot axis preferences that stacked axes need and one
    floor does not.
    """
    wanted = (range(data.num_records) if records is None
              else [int(i) for i in records])
    groups: dict[tuple[str, str | None], list[int]] = {}
    for i in wanted:
        key = (data.ordinate_dim[i], data.dimension_hint[i])
        groups.setdefault(key, []).append(i)
    return list(groups.items())


def waterfall_group(data: DataArray,
                    records: Sequence[int] | None = None,
                    quantity: tuple[str, str | None] | None = None
                    ) -> tuple[list[int], list[int]]:
    """Which of the asked-for records draw together, and which wait.

    One vertical axis cannot honestly hold accelerations and volts at
    once. The 2-D plot answers with one stacked axis per quantity; a
    scene has one floor, so the waterfall draws one (dimension, hint)
    group — the one `quantity` names, else the largest (the earliest
    on a tie) — and reports the rest for the quantity box to offer.
    A `quantity` no longer present falls back to the largest rather
    than refusing: it is a sticky view choice outliving the selection
    it was made on, not a claim about this object.

    Returns (drawn, left_out), both in the object's record order.
    """
    groups = dict(waterfall_groups(data, records))
    if not groups:
        return [], []
    chosen = (groups[quantity] if quantity in groups
              else max(groups.values(), key=len))
    drawn = set(chosen)
    left_out = [i for members in groups.values() for i in members
                if i not in drawn]
    return chosen, sorted(left_out)


def records_per_page(data: DataArray, drawn_count: int,
                     samples: int, shape: str,
                     budget: int = SCENE_BUDGET,
                     points: int = POINT_BUDGET) -> int:
    """How many records fit in one scene, from what each one costs.

    A stepped record costs two points per bin and is never thinned, so
    a 16k-line PSD is 32k vertices however the budget feels about it.
    Anything else is thinned to `points` first, so it costs at most
    that. Derived from vertices rather than fixed at a record count
    because line counts vary by orders of magnitude — 300 records of
    16k lines and 4800 of 1k lines are the same scene.

    `points` is the decimation budget actually in use, not always this
    module's: the report draws the same stage on a far smaller one,
    and costing its records at the app's budget answered 26 records
    where 48 fit. On screen the two are the same number, which is why
    this went unnoticed.

    Never returns zero: one record always draws, even if it is alone
    over budget, because a page showing nothing is not a page.
    """
    per_record = 2 * samples if shape == 'steps' else min(samples, points)
    return max(1, min(drawn_count or 1, budget // max(per_record, 1)))


def quantity_label(key: tuple[str, str | None]) -> str:
    """The word the quantity box shows for one group.

    The response factor of the dimension, in the interface's own
    vocabulary — a PSD group of acceleration**2/frequency reads
    'acceleration', an FRF group by what it measures. A hinted group
    keeps the undefined-units flag the axes carry.
    """
    from ..core.data import channel_quantities
    from ..core.unit_choices import shown_dimension

    dim, hint = key
    named = dim if dim != UNKNOWN else (hint or UNKNOWN)
    if named == UNKNOWN:
        return 'units undefined'
    response, _reference = channel_quantities(named)
    word = shown_dimension(response)
    return word if dim != UNKNOWN else f'{word} [units undefined]'


def _component_values(data: DataArray, y: np.ndarray,
                      component: str) -> tuple[np.ndarray, bool, bool]:
    """(values, log reading, signed component applied).

    Mirrors the 2-D plot: anything but the magnitude of complex
    frequency data is signed, so the log reading gives way to a linear
    one — and a signed component is never a density, which is why the
    third answer feeds `drawing_shape`. Real data ignores the
    component, as the component box does.
    """
    applies = (component != 'magnitude' and np.iscomplexobj(y)
               and data.abscissa_dim == 'frequency')
    if applies and component == 'real':
        return np.asarray(y).real, False, True
    if applies and component == 'imag':
        return np.asarray(y).imag, False, True
    if applies and component == 'phase':
        return np.degrees(np.angle(y)), False, True
    values = np.abs(y) if np.iscomplexobj(y) else np.asarray(y)
    return values, data.log_scaled(), False


def waterfall_arrays(data: DataArray,
                     records: Sequence[int] | None = None,
                     unit_system: UnitSystem | None = None,
                     component: str = 'magnitude',
                     budget: int = POINT_BUDGET,
                     quantity: tuple[str, str | None] | None = None,
                     page: int = 0,
                     scene_budget: int | None = None) -> dict[str, Any]:
    """The waterfall's numbers, before any VTK touches them.

    Decimated per-record curves in display units, the log mapping, the
    labels — everything the scene draws, in a form a test can hold
    without a render window. Returns a dict:

    - ``curves``: one ``(x, z)`` pair per drawn record, peak-decimated;
      ``z`` is log10 of the value when ``log_scaled``, with non-positive
      values as NaN gaps
    - ``labels``: each drawn record's `record_label`
    - ``drawn`` / ``left_out``: record indices, from `waterfall_group`
    - ``log_scaled``: whether ``z`` is the log10 reading
    - ``xlabel`` / ``zlabel``: axis titles, plain text for VTK
    """
    us = unit_system or DEFAULT_SYSTEM
    chosen, left_out = waterfall_group(data, records, quantity)
    x = data.display_abscissa(us)
    # the abscissa's own reading, the same flag the 2-D plot and the
    # report consult: an SRS lays its natural frequencies out in
    # decades, and the stage stands in log10 of them exactly as its
    # vertical axis stands in log10 of a density. The transform is
    # applied to each curve *after* it is built, never to the lines
    # first: a band's edges and a power law's fill-in are worked out
    # in hertz, and taking them from exponents put an octave PSD's
    # steps in the wrong places the day the axis became switchable
    # (Brandon, 2026-09-05)
    log_abscissa = bool(getattr(data, 'log_abscissa', False))

    def decades(coords):
        if not log_abscissa:
            return coords
        coords = np.asarray(coords, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(coords > 0,
                            np.log10(np.where(coords > 0, coords, 1.0)),
                            np.nan)

    # Page *before* converting: display_ordinate on nine hundred
    # records is most of the cost of a scene nobody asked to see all
    # of, and the shape decides what a record costs.
    shape_guess = drawing_shape(
        data, component != 'magnitude'
        and np.iscomplexobj(data.ordinate)
        and data.abscissa_dim == 'frequency')
    # read through the module rather than defaulted in the signature:
    # a default binds at definition, so the constant would stop being
    # the one place the budget lives the moment anyone changed it
    per_page = records_per_page(
        data, len(chosen), len(x), shape_guess,
        SCENE_BUDGET if scene_budget is None else scene_budget, budget)
    pages = max(1, -(-len(chosen) // per_page)) if chosen else 1
    page = max(0, min(int(page), pages - 1))
    drawn = chosen[page * per_page:(page + 1) * per_page]

    # the component and the axis reading are the object's, not a
    # block's, so they are settled once from a row of nothing
    _empty, log_scaled, tagged = _component_values(
        data, np.empty((0, len(x)), dtype=data.ordinate.dtype), component)

    def logged(rows):
        if not log_scaled:
            return rows
        # zero is a gap on a log axis, not a value 300 decades down —
        # the report's axis learned this the hard way
        rows = np.where(rows > 0, rows, np.nan)
        with np.errstate(invalid='ignore'):
            return np.log10(rows)

    # the hard rule, in the 3-D reading as in the 2-D: a density draws
    # as the energy beneath it, so the RMS is the plain area under the
    # trace on the stage. One shape decision serves both readings
    shape = drawing_shape(data, tagged)
    if shape in ('law', 'steps') and drawn:
        # spectra: a few thousand lines a record, converted whole
        y = data.display_ordinate(us, drawn)
        values, _log, _tag = _component_values(data, y, component)
    if shape == 'law' and drawn:
        # a specification's breakpoints mean the power law between
        # them; each record fills in on its own grid
        curves = []
        for k in range(len(drawn)):
            cx, cv = as_power_law(x, values[k])
            cx, cv = peak_decimate(decades(cx), logged(cv), budget)
            curves.append((cx, cv))
    elif shape == 'steps' and len(x) > 1 and drawn:
        # flat across each bin, on the bins the object itself declares
        # — an octave PSD lands on its own band edges.
        # the object's own drawn edges: an end band is drawn over the
        # part its source covered (2026-09-19)
        own = getattr(data, 'bin_edges', None)
        widths = own() if own is not None and getattr(
            data, 'bandwidth', None) is not None else None
        x, values = step_outline(x, values, widths)
        x = decades(x)
        values = np.atleast_2d(values)
        # **Not decimated**, exactly as the 2-D plot does not decimate
        # a step curve, and for the same reason it gives: a spectrum is
        # thousands of lines where a record is hundreds of thousands.
        #
        # It used to be, and that was a correctness bug rather than a
        # cosmetic one. The outline is two points per bin, so it
        # crosses the budget at half the line count a plain curve
        # would: a 3073-line PSD came to 6146 points and was thinned
        # back to 3073 — and peak decimation keeps each slice's
        # extremes, not its bin edges, so the treads it had just built
        # were exactly what got dropped. The staircase came back a
        # polyline, and worse, the area under it stopped being
        # sum(G*df), which is the one thing the shape exists to
        # guarantee. Measured worst case in stressdata: 688k vertices
        # for 28 records of 12289 lines, against the 1.3M the drone
        # already draws.
        curves = [(x, row) for row in logged(values)]
    else:
        # converted and thinned a block of records at a time, so the
        # stage never holds a converted copy of a long run, let alone
        # the decimator's padded one beside it — the whole-object path
        # peaked at four times the data and took a 22 GB import down at
        # 67 GB (2026-09-18); the stage's cost is the block now
        curves = []
        stage_x = decades(x)
        for _start, _stop, block in data.display_blocks(us, drawn):
            rows, _log, _tag = _component_values(data, block, component)
            curves.extend(peak_decimate_rows(stage_x, logged(rows), budget))
    dim = data.ordinate_dim[drawn[0]] if drawn else UNKNOWN
    hint = data.dimension_hint[drawn[0]] if drawn else None
    # label_ascii, not label_text: VTK's axis titles silently drop
    # unicode superscripts, and (in/s²)²/Hz reading (in/s)/Hz is wrong
    # by two squarings
    if dim == UNKNOWN:
        zlabel = (f'{hint} [units undefined]' if hint
                  else 'units undefined')
    else:
        zlabel = us.label_ascii(dim)
        if zlabel == '1':
            # a dimensionless ordinate's unit is the bare '1', which
            # is true and tells a reader nothing — a coherence stage
            # came out with its vertical axis titled '1'. With no unit
            # worth printing, name the measurement instead.
            from ..names import display_name

            zlabel = display_name(type(data).__name__).lower()
    if log_scaled:
        zlabel = f'log10 {zlabel}'.rstrip()
    xlabel = (f'{data.abscissa_dim.capitalize()} '
              f'({us.label_ascii(data.abscissa_dim)})')
    if log_abscissa:
        # the same honesty as the vertical axis: the coordinates are
        # exponents, and the title says so for the VTK stage, whose
        # axis prints them as they are. The report's canvas prints
        # real values at decade ticks instead and drops the prefix.
        xlabel = f'log10 {xlabel}'
    return {'curves': curves, 'labels': [data.record_label(i) for i in drawn],
            'drawn': drawn, 'left_out': left_out, 'log_scaled': log_scaled,
            'log_abscissa': log_abscissa,
            'xlabel': xlabel, 'zlabel': zlabel,
            # what the bar's page stepper reads: which page is on the
            # stage, how many there are, and how many records this
            # quantity has in total across all of them
            'page': page, 'pages': pages, 'per_page': per_page,
            'of_quantity': len(chosen)}


def _finite_runs(z: np.ndarray) -> list[tuple[int, int]]:
    """[(start, stop)] maximal runs of finite values, two points or more.

    A gap splits a record into separate polylines rather than drawing a
    line across it — the same honesty as the 2-D plot's `gapless`, done
    with connectivity because a scene can have as many lines as it
    needs. A lone finite point between gaps draws no line and is
    dropped.
    """
    finite = np.isfinite(z)
    if not finite.any():
        return []
    edges = np.flatnonzero(np.diff(finite.astype(np.int8)))
    starts = ([0] if finite[0] else []) + [int(e) + 1 for e in edges[~finite[edges]]]
    stops = [int(e) + 1 for e in edges[finite[edges]]] + ([len(z)] if finite[-1] else [])
    return [(a, b) for a, b in zip(starts, stops) if b - a >= 2]


def nice_axis(low: float, high: float,
              budget: float = 0.05) -> tuple[float, float, int]:
    """(low, high, label count) — the range widened to round ticks,
    when the widening is nearly free.

    The cube axes place their labels at even divisions of the range,
    so a 0.4998 s record divided five ways puts ticks at 0.125 and
    0.375 — and the default one-decimal format then *prints* them as
    0.1 and 0.4, numbers that do not sit where they claim (Brandon,
    2026-08-28: the record read as 0.6 s long against the truncation
    boxes' honest 0.5). Widening to 0..0.5 with six labels makes
    every tick round *and* true.

    Only when it costs at most `budget` of the span in dead axis: a
    1024 Hz Nyquist would widen to 1200, a fifth of the stage holding
    no data, so it keeps its exact range — the label format prints
    those positions truthfully instead.
    """
    span = float(high) - float(low)
    if not span > 0.0 or not np.isfinite(span):
        return float(low), float(high), 5
    raw = span / 5.0
    magnitude = 10.0 ** np.floor(np.log10(raw))
    step = next(m * magnitude for m in (1.0, 2.0, 2.5, 5.0, 10.0)
                if raw <= m * magnitude)
    wide_low = np.floor(low / step) * step
    wide_high = np.ceil(high / step) * step
    if (wide_high - wide_low) - span > budget * span:
        return float(low), float(high), 5
    intervals = round(float((wide_high - wide_low) / step))
    return float(wide_low), float(wide_high), intervals + 1


def stage_curves(arrays: dict[str, Any],
                 ordinate_limits: tuple[float, float] | None = None
                 ) -> dict[str, Any]:
    """The waterfall's curves placed on the unit stage.

    Takes `waterfall_arrays`' output and answers where every point
    stands in stage coordinates — abscissa across, records receding,
    level up — plus the real ranges that stage stands in for and the
    runs of finite values each curve breaks into (a gap is a gap; a
    polyline drawn through one would invent a value).

    Here rather than inside `add_waterfall` because the report draws
    the same stage in a canvas, and two normalizations would be two
    pictures of one measurement. Returns a dict of ``runs`` (each
    ``{'station', 'points', 'levels'}``), ``extents`` (x0, x1, z0,
    z1), ``stations`` and ``labels`` — plain floats, no VTK.
    """
    curves = arrays['curves']
    n = len(curves)
    sx, sy, sz = STAGE
    x0 = min((float(np.nanmin(cx)) for cx, _cz in curves if len(cx)),
             default=0.0)
    x1 = max((float(np.nanmax(cx)) for cx, _cz in curves if len(cx)),
             default=1.0)
    # widened to round tick values where that is nearly free, before
    # anything is normalized against the range — the axis, the curves,
    # the marks and the report's canvas all read these extents, so the
    # widening has to happen at the one place they are set
    x0, x1, x_label_count = nice_axis(x0, x1)
    finite = [cz[np.isfinite(cz)] for _cx, cz in curves]
    finite = [cz for cz in finite if len(cz)]
    z0 = min((float(cz.min()) for cz in finite), default=0.0)
    z1 = max((float(cz.max()) for cz in finite), default=1.0)
    # an object that pins its own axis pins the stage the same way —
    # the 2-D plot holds a coherence to 0..1.05 so that a wall of ones
    # reads as the wall it is (too few averages), and normalized to its
    # own tiny range here it would read as mountains instead
    if ordinate_limits is not None and not arrays['log_scaled']:
        z0, z1 = ordinate_limits
    xspan = (x1 - x0) or 1.0
    zspan = (z1 - z0) or 1.0
    runs = []
    for k, (cx, cz) in enumerate(curves):
        station = (k / max(n - 1, 1)) * sy
        xn = (cx - x0) / xspan * sx
        zn = (cz - z0) / zspan * sz
        for start, stop in _finite_runs(cz):
            runs.append({
                'record': k, 'station': float(station),
                'points': np.column_stack([
                    xn[start:stop],
                    np.full(stop - start, station),
                    zn[start:stop]]),
                'levels': np.asarray(cz[start:stop], dtype=float),
                'first': int(start)})
    # the axis titles ride along, so this and `paired_stage_curves`
    # answer with the same shape and a reader of either does not have
    # to know which one it asked
    return {'runs': runs, 'extents': (x0, x1, z0, z1), 'stations': n,
            'x_label_count': x_label_count,
            'labels': arrays['labels'], 'xlabel': arrays['xlabel'],
            'log_abscissa': arrays.get('log_abscissa', False),
            'zlabel': arrays['zlabel'],
            'xn': lambda v: ((np.asarray(v, dtype=float) - x0) / xspan * sx),
            'zn': lambda v: ((np.asarray(v, dtype=float) - z0)
                             / zspan * sz)}


@one_render
def add_waterfall(plotter: Any, data: DataArray,
                  records: Sequence[int] | None = None,
                  unit_system: UnitSystem | None = None,
                  theme: Any = None, component: str = 'magnitude',
                  budget: int = POINT_BUDGET,
                  quantity: tuple[str, str | None] | None = None,
                  page: int = 0,
                  scene_budget: int | None = None,
                  color: str | None = None) -> dict[str, Any]:
    """Draw the waterfall into a plotter: one mesh, labels, axes.

    `quantity` picks which (dimension, hint) group of a mixed object
    stands on the floor — the quantity box's choice; None means the
    largest. Returns `waterfall_arrays`' dict with ``points`` (how many
    vertices the scene got — the number a size-ceiling test pins) and
    ``named`` (which drawn positions got a channel label) added.

    `color` draws the ribbons flat in one color instead of on the
    level colormap: the stood-back reading, for a record shown as the
    reference for something else drawn over it — what the filter
    preview does with the raw record (Brandon, 2026-08-25). The
    extents, the axes and the stations are unchanged, so the thing
    drawn over it lands in the same space.
    """
    import pyvista as pv

    us = unit_system or DEFAULT_SYSTEM
    colors = resolve_theme(theme)
    arrays = waterfall_arrays(data, records, us, component, budget,
                              quantity, page, scene_budget)
    curves = arrays['curves']
    n = len(curves)
    sx, sy, sz = STAGE
    del sz
    staged = stage_curves(arrays, data.ordinate_limits)
    x0, x1, z0, z1 = staged['extents']

    points, lines, scalars = [], [], []
    label_spots, label_names, named = [], [], []
    # past the limit, every nth name — the curves all still draw
    step = max(1, -(-n // LABEL_LIMIT))
    total = 0
    for run in staged['runs']:
        count = len(run['points'])
        lines.append(np.concatenate([[count],
                                     np.arange(total, total + count)]))
        points.append(run['points'])
        scalars.append(run['levels'])
        total += count
    for k, (cx, cz) in enumerate(curves):
        if k % step == 0 and len(cx):
            xn, zn = staged['xn'](cx), staged['zn'](cz)
            station = (k / max(n - 1, 1)) * sy
            first = int(np.flatnonzero(np.isfinite(cz))[0]) if np.isfinite(
                cz).any() else 0
            label_spots.append([xn[first] - 0.03 * sx, station,
                                zn[first] if np.isfinite(cz[first]) else 0.0])
            label_names.append(arrays['labels'][k])
            named.append(k)

    if total:
        mesh = pv.PolyData(np.vstack(points).astype(np.float32),
                           lines=np.concatenate(lines))
        mesh.point_data['level'] = np.concatenate(scalars)
        if color is not None:
            plotter.add_mesh(mesh, color=color, line_width=2,
                             show_scalar_bar=False, name='waterfall')
        else:
            # clim is the stage's own range, so the color scale and the
            # vertical axis are one scale — for pinned data (a
            # coherence's 0..1.05) as much as for data spanning its own
            # extremes
            plotter.add_mesh(mesh, scalars='level', cmap='viridis',
                             line_width=2, clim=(z0, z1),
                             show_scalar_bar=False, name='waterfall')
    if label_names:
        plotter.add_point_labels(
            np.asarray(label_spots), label_names, font_size=12,
            always_visible=True, text_color=colors['scene_text'],
            shape=None, fill_shape=False, show_points=False,
            name='waterfall-labels')
    if total:
        # the stage relabeled with the real ranges. The depth axis
        # keeps its grid but not its numbers: the channel names on the
        # curves are its labels
        # use_3d_text=False: the default vector text is ASCII-only and
        # silently drops the superscripts, so (in/s²)²/Hz read (in/s)/Hz
        # — wrong by two squarings
        plotter.show_bounds(
            axes_ranges=(x0, x1, 0, max(n - 1, 1), z0, z1),
            n_xlabels=staged['x_label_count'],
            # the house number format: a label that cannot be round is
            # printed where it actually stands, never rounded to a
            # place it does not (the default is one decimal)
            fmt='%.4g',
            # a space, not '': VTK builds a text actor for the depth axis
        # either way and logs 'vtkVectorText: Text is not set!' twice
        # per draw for an empty one — two error lines on every render,
        # which is exactly the noise a real error later hides in
        xtitle=arrays['xlabel'], ytitle=' ', ztitle=arrays['zlabel'],
            show_ylabels=False, grid='back', location='outer',
            use_3d_text=False, color=colors['scene_text'])
        if arrays.get('log_abscissa'):
            # decades, drawn as the 2-D plot draws them, in place of
            # the cube axes' even divisions of the exponent range
            from .marks import add_decade_axis

            arrays['decades'] = add_decade_axis(
                plotter, (x0, x1, z0, z1), theme=theme)['labels']
    arrays['points'] = total
    arrays['named'] = named
    # the real ranges the stage's unit cube stands in for, so marks
    # drawn later (averaging frames, shock windows) can land at the
    # data positions they describe
    arrays['extents'] = (x0, x1, z0, z1)
    arrays['stations'] = n
    return arrays


def waterfall_scene(data: DataArray,
                    records: Sequence[int] | None = None,
                    unit_system: UnitSystem | None = None,
                    plotter: Any = None, off_screen: bool = False,
                    theme: Any = None, component: str = 'magnitude',
                    budget: int = POINT_BUDGET,
                    quantity: tuple[str, str | None] | None = None) -> Any:
    """Build (or add to) a PyVista plotter showing the waterfall.

    Same shape as `geometry_scene`: `theme` is 'light', 'dark' or a
    colors dict; returns the plotter — `.show()` it, or `.screenshot()`
    if off_screen. The camera is placed once, here: from the front-left
    and above, abscissa reading left to right, records receding.
    """
    import pyvista as pv

    colors = resolve_theme(theme)
    if plotter is None:
        plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background(colors['scene_background'])
    add_waterfall(plotter, data, records, unit_system, colors, component,
                  budget, quantity)
    place_camera(plotter)
    return plotter


def place_camera(plotter: Any) -> None:
    """The waterfall's home view, placed — never re-placed on a redraw.

    From the front-left and above: abscissa left to right, records
    receding. Far enough out that the vertical axis title clears the
    frame — one notch closer clipped it. The window calls this when the
    displayed object changes and at no other time, per the standing
    camera rule; the stage is a fixed size, so one home view fits every
    object.
    """
    sx, sy, sz = STAGE
    plotter.camera_position = [
        (-sx * 0.5, -sy * 2.3, sz * 3.0),
        (sx * 0.5, sy * 0.5, sz * 0.2),
        (0.0, 0.0, 1.0)]


def stage_basis() -> tuple[np.ndarray, np.ndarray]:
    """(right, up) — the stage's home view, as a parallel basis.

    Derived from the very camera `place_camera` sets, so everything
    that draws the stage without a camera — the app icon's tile, the
    report's 3-D figure — opens on the app's own view of data rather
    than on a second isometric that merely looks similar. Parallel
    rather than the camera's perspective: both of those readings are
    small, and perspective at that size is a distortion nobody reads
    as depth.
    """
    sx, sy, sz = STAGE
    position = np.array([-sx * 0.5, -sy * 2.3, sz * 3.0])
    focus = np.array([sx * 0.5, sy * 0.5, sz * 0.2])
    view = focus - position
    view /= np.linalg.norm(view)
    right = np.cross(view, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    return right, np.cross(right, view)


def ribbon(xn: Any, yn: Any, z_low: Any, z_high: Any) -> Any:
    """A translucent surface between two curves over one path on the
    stage — the banded and sine views' zone fill, built once here so
    the two cannot drift apart. `yn` may be one depth or one per point.
    """
    import numpy as np
    import pyvista as pv

    n = len(xn)
    yn = np.broadcast_to(np.asarray(yn, dtype=float), (n,))
    points = np.concatenate([
        np.column_stack([xn, yn, z_low]),
        np.column_stack([xn, yn, z_high])])
    faces = np.column_stack([
        np.full(n - 1, 4),
        np.arange(n - 1), np.arange(1, n),
        np.arange(n + 1, 2 * n),
        np.arange(n, 2 * n - 1)]).ravel()
    return pv.PolyData(points.astype(np.float32), faces=faces)


def finish_scene(plotter: Any, screenshot: str | None, show: bool) -> Any:
    """End a stage's headless call the one way every stage does: render
    to `screenshot` and hand back the image, or show, or hand back the
    plotter for the caller to drive."""
    if screenshot is not None:
        image = plotter.screenshot(screenshot)
        plotter.close()
        return image
    if show:
        plotter.show()
    return plotter


def plot_waterfall(data: DataArray,
                   records: Sequence[int] | None = None, *,
                   screenshot: str | None = None,
                   unit_system: UnitSystem | None = None,
                   theme: Any = None, component: str = 'magnitude',
                   show: bool = True,
                   quantity: tuple[str, str | None] | None = None) -> Any:
    """Show the waterfall interactively, or render it to `screenshot`.

    The scriptable face of the plot bar's 3-D reading, like
    `plot_geometry` for the geometry scene. Returns the pane (its
    `.plotter` is the PyVista one), or the image array when rendering
    to a file.
    """
    if screenshot is not None:
        plotter = waterfall_scene(data, records, unit_system,
                                  off_screen=True, theme=theme,
                                  component=component, quantity=quantity)
        img = plotter.screenshot(screenshot)
        plotter.close()
        return img
    from ..gui.windows import scene_window

    return scene_window(
        lambda plotter: waterfall_scene(data, records, unit_system,
                                        plotter=plotter, theme=theme,
                                        component=component,
                                        quantity=quantity),
        theme=theme, axis_unit='', title='Waterfall', show=show)
