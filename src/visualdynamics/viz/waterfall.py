"""The 3-D view of a data object: its records as a waterfall.

Many channels on one 2-D axis hide each other where it matters most:
resonances line up, and the tenth curve is drawn over the first nine.
The waterfall spreads the records along a depth axis instead, one
polyline per record at its own station, with the same label the grids
and the legend use, and colored by level so that a peak can be followed
across the whole set.

It uses numpy and PyVista only, with no Qt, like the rest of `viz`. Two
choices set how large a scene can be:

- Each record's points are peak-decimated before VTK gets them
  (`decimate.peak_decimate`, the same reduction the 2-D plot and the
  report use), so a million-sample time history arrives as a few
  thousand points that keep every peak.
- All of the curves are one PolyData under one actor, whatever the
  record count, because an actor per record is what makes a large scene
  slow.

The scene is drawn on a fixed stage (`STAGE`) instead of in data
coordinates, because a PSD's level spans decades while its frequency
spans kilohertz, and drawn in raw coordinates it would be a thin needle.
The axes are relabeled with the real ranges (`axes_ranges`). The
vertical axis is in log10 for data read on a log axis, and its title
says so.
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

# One actor for every curve: the animation learned at 202k nodes that
# per-record (per-mesh) actors are what make a large scene slow.

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
    """Every (dimension, hint) group among `records` (all of them when
    None), as [(key, record indices)].

    In the object's own order. These are what the quantity box offers,
    and what `waterfall_group` chooses from. The grouping key is the 2-D
    plot's, without the per-plot axis preferences that stacked axes need
    and a single stage does not.
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
    """Which of the requested records are drawn together, and which are
    left out.

    One vertical axis cannot hold accelerations and volts at once. The
    2-D plot uses one stacked axis per quantity; a scene has one stage,
    so the waterfall draws one (dimension, hint) group, the one
    `quantity` names or else the largest (the earliest on a tie), and
    reports the rest for the quantity box to offer. A `quantity` that is
    not present falls back to the largest group instead of raising,
    because it is a view choice kept from an earlier selection and not a
    statement about this object.

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
    """How many records fit in one scene, from the vertices each one
    costs.

    `samples` is the number of abscissa values and `shape` the drawing
    shape (`plot.drawing_shape`). A stepped record ('steps') costs two
    points per bin and is never thinned, so a 16k-line PSD is 32k
    vertices whatever the budget. Any other record is thinned to
    `points` first, so it costs at most that. The count comes from
    vertices and not from a fixed number of records because line counts
    vary by orders of magnitude: 300 records of 16k lines and 4800 of 1k
    lines are the same size of scene. `budget` is the scene's vertex
    limit, and the result is at most `drawn_count`.

    `points` must be the decimation budget actually in use, which is not
    always this module's: the report draws the same stage with a much
    smaller one.

    Never returns zero. One record is always drawn, even when it alone
    is over the budget.
    """
    # Costing the report's records at the app's budget once gave 26
    # records a page where 48 fit; on screen the two budgets are the
    # same number, which is why it went unnoticed.
    per_record = 2 * samples if shape == 'steps' else min(samples, points)
    return max(1, min(drawn_count or 1, budget // max(per_record, 1)))


def quantity_label(key: tuple[str, str | None]) -> str:
    """The word the quantity box shows for one (dimension, hint) group.

    It is the response quantity of the dimension, in the words the
    interface uses: a PSD group of acceleration**2/frequency shows
    'acceleration', and an FRF group shows what it measures. A group
    whose units are undefined is named the same way from its hint, with
    '[units undefined]' after it as the axes show, or is
    'units undefined' when it has no hint.
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
    """The waterfall's numbers, before VTK is involved.

    Everything the scene draws, in a form a test can check without a
    render window: the per-record curves in display units, decimated to
    `budget` points (a stepped spectrum is not decimated), the log
    mapping and the labels. `component` ('magnitude', 'real', 'imag' or
    'phase') applies to complex frequency data, as in the 2-D plot.
    `quantity` picks the group drawn (`waterfall_group`). The records
    are split into pages of as many as fit in `scene_budget` vertices
    (`SCENE_BUDGET` by default, `records_per_page`), and `page` picks
    one, clipped to the pages there are. Returns a dict:

    - ``curves``: one ``(x, z)`` pair per drawn record. ``z`` is log10
      of the value when ``log_scaled``, with non-positive values as NaN
      gaps, and ``x`` is log10 of the abscissa when ``log_abscissa``
    - ``labels``: each drawn record's `record_label`
    - ``drawn`` / ``left_out``: record indices; ``drawn`` is this page
      of the group from `waterfall_group`
    - ``log_scaled`` / ``log_abscissa``: whether ``z`` and ``x`` are
      log10 values
    - ``xlabel`` / ``zlabel``: axis titles, in plain text for VTK
    - ``page``, ``pages``, ``per_page``: the page drawn, how many there
      are, and the records on each
    - ``of_quantity``: how many records the group has over all pages
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
    """(low, high, label count): the range widened to round tick values,
    when that adds little empty axis.

    The cube axes place their labels at even divisions of the range, so
    a 0.4998 s record divided five ways has ticks at 0.125 and 0.375,
    which a one-decimal format prints as 0.1 and 0.4, values that are
    not where they are drawn. Widened to 0 to 0.5 with six labels, every
    tick is round and correct.

    The range is widened only when that adds at most `budget` of the
    span as empty axis. A 1024 Hz Nyquist would widen to 1200, leaving a
    fifth of the stage with no data, so it keeps its exact range with
    five labels, and the label format prints those positions as they
    are. A range that is empty or not finite is returned unchanged.
    """
    # Brandon, 2026-08-28: a record read as 0.6 s long against the
    # truncation boxes' correct 0.5, because the ticks at 0.125 and 0.375
    # were printed as 0.1 and 0.4.
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


def drawn_span(curves: Any, default: tuple[float, float] = (0.0, 1.0)
               ) -> tuple[float, float]:
    """(low, high) of the abscissa where something is drawn, over the
    points whose level is finite, never over the whole abscissa a curve
    is carried on. A PSD written the way a controller writes a
    specification (every line to Nyquist, zero outside its band) is
    drawn only over its band, since zero is a gap on a log level, and
    the stage spans only that band. `curves` are ``(x, z)`` pairs, and
    `default` is returned when nothing is finite."""
    # Brandon, 2026-10-09: a stage that spanned every line drew 20 to
    # 400 Hz as a small fraction of a 0 to 2048 Hz floor.
    lows, highs = [], []
    for cx, cz in curves:
        cx, cz = np.asarray(cx, dtype=float), np.asarray(cz, dtype=float)
        shown = cx[np.isfinite(cz) & np.isfinite(cx)]
        if shown.size:
            lows.append(float(shown.min()))
            highs.append(float(shown.max()))
    if not lows:
        return default
    return min(lows), max(highs)


def stage_curves(arrays: dict[str, Any],
                 ordinate_limits: tuple[float, float] | None = None
                 ) -> dict[str, Any]:
    """The waterfall's curves placed on the stage.

    Takes the output of `waterfall_arrays` and gives the position of
    every point in stage coordinates (abscissa across, records receding,
    level up), the real ranges the stage stands for, and the runs of
    finite values each curve breaks into. A gap stays a gap, since a
    polyline drawn through one would show a value that is not there.
    The abscissa range is widened to round ticks (`nice_axis`). When
    `ordinate_limits` is given and the level is not log-scaled, it fixes
    the vertical range, as the 2-D plot fixes a coherence to 0 to 1.05.

    This is separate from `add_waterfall` because the report draws the
    same stage in a canvas, and two normalizations would give two
    pictures of one measurement. Returns a dict, with no VTK in it:

    - ``runs``: each ``{'record', 'station', 'points', 'levels',
      'first'}``, where ``first`` is the run's first index in its curve
    - ``extents``: (x0, x1, z0, z1), the real ranges
    - ``stations``: how many records are on the stage
    - ``x_label_count``: how many abscissa labels to draw
    - ``labels``, ``xlabel``, ``zlabel``, ``log_abscissa``: from the
      arrays
    - ``xn``, ``zn``: functions taking real values to stage
      coordinates
    """
    curves = arrays['curves']
    n = len(curves)
    sx, sy, sz = STAGE
    x0, x1 = drawn_span(curves)
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
    """Draw the waterfall into a plotter: one mesh, the channel labels
    and the axes.

    The arguments are those of `waterfall_arrays`. `quantity` picks
    which (dimension, hint) group of a mixed object is drawn, as the
    quantity box does; None means the largest. Past `LABEL_LIMIT`
    records, only every nth record gets a channel label, so that at most
    `LABEL_LIMIT` are labeled; every curve is still drawn.
    A log abscissa is labeled in decades (`marks.add_decade_axis`).

    `color` draws the curves in one flat color instead of on the level
    colormap. That is for a record shown as the reference for something
    drawn over it, as the filter preview does with the raw record. The
    extents, the axes and the stations are unchanged, so what is drawn
    over it lands in the same space.

    Returns the dict from `waterfall_arrays` with these added:
    ``points`` (how many vertices the scene has), ``named`` (which drawn
    positions got a channel label), ``extents`` ((x0, x1, z0, z1), the
    real ranges the stage stands for, so marks drawn later land at the
    data positions they describe), ``stations`` (how many records are
    on the stage) and, for a log abscissa, ``decades`` (the decade
    labels drawn).
    """
    # The flat `color` was added for the filter preview's raw record
    # (Brandon, 2026-08-25).
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
                    quantity: tuple[str, str | None] | None = None,
                    marks: str | None = None,
                    per_octave: int | None = None) -> Any:
    """Build a PyVista plotter showing the waterfall, or add it to
    `plotter`.

    Called the same way as `geometry_scene`: `theme` is 'light', 'dark'
    or a colors dict, and the plotter is returned, to ``.show()``, or to
    ``.screenshot()`` when it is off screen. `records`, `component`,
    `budget` and `quantity` are those of `waterfall_arrays`. The camera
    is placed here, once (`place_camera`). `marks` draws one of the
    record's readings over the stage (`viz.marks.add_stage_reading`),
    with `per_octave` for 'octave'.
    """
    import pyvista as pv

    colors = resolve_theme(theme)
    if plotter is None:
        plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background(colors['scene_background'])
    info = add_waterfall(plotter, data, records, unit_system, colors,
                         component, budget, quantity)
    if marks is not None:
        from .marks import add_stage_reading

        add_stage_reading(plotter, data, info, marks, per_octave=per_octave,
                          unit_system=unit_system, theme=colors)
    place_camera(plotter)
    return plotter


def place_camera(plotter: Any) -> None:
    """Place the camera at the waterfall's home view.

    From the front-left and above, with the abscissa reading left to
    right and the records receding, and far enough out that the
    vertical axis title stays inside the frame (one step closer cut it
    off). The window calls this when the displayed object changes and at
    no other time, so a redraw never moves the camera. The stage is a
    fixed size, so one home view fits every object.
    """
    sx, sy, sz = STAGE
    plotter.camera_position = [
        (-sx * 0.5, -sy * 2.3, sz * 3.0),
        (sx * 0.5, sy * 0.5, sz * 0.2),
        (0.0, 0.0, 1.0)]


def stage_basis() -> tuple[np.ndarray, np.ndarray]:
    """(right, up): the stage's home view, as unit vectors for a
    parallel projection.

    Computed from the camera `place_camera` sets, so everything that
    draws the stage without a camera (the app icon's tile, the report's
    3-D figure) shows the same view as the app, not a second isometric
    view that looks similar. It is parallel, not the camera's
    perspective, because both of those figures are small, and at that
    size perspective looks like distortion rather than depth.
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
    """A surface between two curves over one path on the stage, as a
    PolyData of quads: the zone fill of the banded and sine stages,
    built here once so the two stay the same. `xn`, `z_low` and `z_high`
    have one value per point, and `yn` is one depth or one per point.
    The caller sets its color and opacity.
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
    """Finish a stage's headless call, the same way for every stage:
    with a `screenshot` path, render to it, close the plotter and return
    the image; otherwise show the plotter when `show` is true, and
    return the plotter."""
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
                   quantity: tuple[str, str | None] | None = None,
                   marks: str | None = None,
                   per_octave: int | None = None) -> Any:
    """Show the waterfall interactively, or render it to `screenshot`.

    The script version of the plot bar's 3-D view, as `plot_geometry`
    is for the geometry scene. `records`, `component` and `quantity`
    are those of `waterfall_arrays`. Returns the pane (its `.plotter` is
    the PyVista one), or the image array when rendering to a file.

    `marks` draws one of the record's readings on the stage, as the plot
    bar's toggles do in 3-D: 'averaging', 'shocks', 'truncation' or
    'filter' over a time history, each taken from the history's own
    setting or else from what its detector suggests, or 'octave' over a
    PSD, banded `per_octave`.
    """
    if screenshot is not None:
        plotter = waterfall_scene(data, records, unit_system,
                                  off_screen=True, theme=theme,
                                  component=component, quantity=quantity,
                                  marks=marks, per_octave=per_octave)
        img = plotter.screenshot(screenshot)
        plotter.close()
        return img
    from ..gui.windows import scene_window

    return scene_window(
        lambda plotter: waterfall_scene(data, records, unit_system,
                                        plotter=plotter, theme=theme,
                                        component=component,
                                        quantity=quantity, marks=marks,
                                        per_octave=per_octave),
        theme=theme, axis_unit='', title='Waterfall', show=show)
