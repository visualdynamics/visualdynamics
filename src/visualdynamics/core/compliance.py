"""How a measured response compares with the specification it was controlled to.

A random vibration test is judged by comparing the response PSD with its
specification, and a report needs numbers from that comparison: the
power specified, the power measured, and how much of the band fell
outside the warning and abort limits.

The two rarely share a frequency axis. A specification is often written
at a few breakpoints, or on the controller's lines rather than the
analysis lines. A specification written at breakpoints is read log-log,
the way it is drawn, so the curve between two breakpoints is the
straight line on log axes, a power law. Interpolated linearly instead, a
segment one decade wide reads several dB above the specification through
the middle of the segment.

The comparison is made cell by cell (`cells`): on the measurement's lines
against a breakpoint curve, and on the coarser of the two grids when both
are densities on bins. Only frequencies both objects are written over are
judged. Outside the specification's band nothing is specified, and a
response there neither passes nor fails.

At the two ends of that band a line's bin is usually partly inside and
partly outside. Counting the whole bin credits the response with power
the specification did not ask for, and dropping it loses power that was
asked for. So a line at an end is compared over the part of its bin that
the specification covers, the same part the plot shades. Both sides of
the comparison are then areas over the same range of frequency, which is
what an RMS error needs.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import DataArray, Specification


def log_interpolate(frequencies: ArrayLike, spec_frequencies: ArrayLike,
                    spec_values: ArrayLike) -> np.ndarray:
    """A specification's values at `frequencies`, interpolated log-log.

    Points that are zero, negative or NaN are dropped first (a controller
    writes zero on the lines outside its band), and the result is NaN
    below the lowest and above the highest remaining point: extending the
    end segments would add a requirement nobody wrote. A dropped point
    between two remaining ones is bridged by the line between them. Fewer
    than two remaining points gives NaN everywhere.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    spec_frequencies = np.asarray(spec_frequencies, dtype=float)
    spec_values = np.asarray(np.real(spec_values), dtype=float)

    usable = (np.isfinite(spec_values) & (spec_values > 0.0)
              & np.isfinite(spec_frequencies) & (spec_frequencies > 0.0))
    out = np.full(frequencies.shape, np.nan)
    if usable.sum() < 2:
        return out
    known_x = np.log10(spec_frequencies[usable])
    known_y = np.log10(spec_values[usable])
    order = np.argsort(known_x)
    known_x, known_y = known_x[order], known_y[order]

    low, high = spec_frequencies[usable].min(), spec_frequencies[usable].max()
    inside = (frequencies > 0.0) & (frequencies >= low) & (frequencies <= high)
    if inside.any():
        out[inside] = 10.0 ** np.interp(np.log10(frequencies[inside]),
                                        known_x, known_y)
    return out


def log_log_areas(frequencies: ArrayLike, values: ArrayLike,
                  lows: ArrayLike, highs: ArrayLike) -> np.ndarray:
    """The exact area under the log-log curve through these points, over
    each of several bands at once: `log_log_area` for many bands.

    The points are breakpoints of a continuous curve, and between two of
    them the curve is the straight line they make on log axes, a power
    law W = C f**n. Its integral has a closed form, so the area does not
    depend on a grid or on how finely anything else was measured. For a
    segment from (f1, W1) to (f2, W2), with n the slope in log-log, the
    area is

        W1 f1 ln(f2/f1)                                   when n is -1
        W1 / f1**n * (f2**(n+1) - f1**(n+1)) / (n + 1)    otherwise

    The cumulative area at every breakpoint is computed once, and each
    band's area is the cumulative area at its high edge less that at its
    low edge, with the partial segments at the two ends taken from the
    same power law. A thousand bands cost about what one does. Only
    positive, finite points are used. A band that reaches past them is
    cut at them, a band wholly outside them is zero, and fewer than two
    points gives NaN for every band.
    """
    # vectorized because the comparison is judged cell by cell
    # (2026-09-19), and a thousand cells must cost what one did
    frequencies = np.asarray(frequencies, dtype=float)
    values = np.asarray(np.real(values), dtype=float)
    lows = np.atleast_1d(np.asarray(lows, dtype=float))
    highs = np.atleast_1d(np.asarray(highs, dtype=float))
    usable = (np.isfinite(frequencies) & (frequencies > 0.0)
              & np.isfinite(values) & (values > 0.0))
    if usable.sum() < 2:
        return np.full(lows.shape, np.nan)
    f = frequencies[usable]
    w = values[usable]
    order = np.argsort(f)
    f, w = f[order], w[order]
    keep = np.concatenate([[True], np.diff(f) > 0.0])
    f, w = f[keep], w[keep]
    if f.size < 2:
        return np.full(lows.shape, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        n = np.log(w[1:] / w[:-1]) / np.log(f[1:] / f[:-1])
    # the whole of each segment, then the running total at every point
    with np.errstate(divide='ignore', invalid='ignore'):
        whole = np.where(
            np.abs(n + 1.0) < 1e-12,
            w[:-1] * f[:-1] * np.log(f[1:] / f[:-1]),
            # written as w1·f1·((f2/f1)**(n+1) - 1)/(n+1): the exponent
            # is bounded by the data where f1**n alone is not (a steep
            # segment of a dense specification overflowed the other
            # form)
            w[:-1] * f[:-1] * ((f[1:] / f[:-1]) ** (n + 1.0) - 1.0)
            / (n + 1.0))
    cumulative = np.concatenate([[0.0], np.cumsum(whole)])

    def total(x: np.ndarray) -> np.ndarray:
        """The area from the first point to `x`, `x` cut to the curve."""
        x = np.clip(x, f[0], f[-1])
        k = np.clip(np.searchsorted(f, x, side='right') - 1, 0, f.size - 2)
        f1, w1, nk = f[k], w[k], n[k]
        with np.errstate(divide='ignore', invalid='ignore'):
            partial = np.where(
                np.abs(nk + 1.0) < 1e-12,
                w1 * f1 * np.log(x / f1),
                w1 * f1 * ((x / f1) ** (nk + 1.0) - 1.0) / (nk + 1.0))
        return cumulative[k] + np.where(x > f1, partial, 0.0)

    out = total(highs) - total(lows)
    return np.where(highs > lows, np.maximum(out, 0.0), 0.0)


def log_log_area(frequencies: ArrayLike, values: ArrayLike,
                 low: float | None = None,
                 high: float | None = None) -> float:
    """The exact area under the log-log curve through these points, over
    one band, or over the whole curve when no band is given.

    One band of `log_log_areas`, cut to the written points. Zero when
    the band is empty, NaN when fewer than two points are positive and
    finite."""
    frequencies = np.asarray(frequencies, dtype=float)
    values = np.asarray(np.real(values), dtype=float)
    usable = (np.isfinite(frequencies) & (frequencies > 0.0)
              & np.isfinite(values) & (values > 0.0))
    if usable.sum() < 2:
        return float('nan')
    f = frequencies[usable]
    low = float(f.min()) if low is None else max(float(low), float(f.min()))
    high = float(f.max()) if high is None else min(float(high), float(f.max()))
    if not high > low:
        return 0.0
    return float(log_log_areas(frequencies, values, [low], [high])[0])


def written_band(frequencies: ArrayLike,
                 values: ArrayLike) -> tuple[float, float] | None:
    """(low, high), the range one written curve covers: its lowest and
    highest positive, finite points. None when there are fewer than two."""
    frequencies = np.asarray(frequencies, dtype=float)
    values = np.asarray(np.real(values), dtype=float)
    usable = (np.isfinite(frequencies) & (frequencies > 0.0)
              & np.isfinite(values) & (values > 0.0))
    if usable.sum() < 2:
        return None
    return float(frequencies[usable].min()), float(frequencies[usable].max())


def coverage(spectrum: Any, record: int = 0) -> list[tuple[float, float]]:
    """The frequency ranges one record of a spectrum is written over,
    merged.

    For a density on bins, the bins of the lines the object counts as
    written (`written`). For a specification, that is finite and
    positive, so a controller's zero and NaN lines are gaps, not a
    requirement of zero. For a curve between breakpoints, the segments
    between consecutive written breakpoints. Either way the result is
    (low, high) pairs in ascending order, with touching ranges joined. A
    comparison is judged only over the part of a cell inside these
    ranges.

    Parameters
    ----------
    spectrum : Psd or Specification
        The spectrum.
    record : int, default 0
        Which record.

    Returns
    -------
    list of (float, float)
        Empty when nothing is written above zero hertz.
    """
    frequencies = np.asarray(spectrum.abscissa, dtype=float)
    said = spectrum.written(record) & np.isfinite(frequencies)
    if getattr(spectrum, 'interpolation', 'bin') == 'log_log':
        said &= frequencies > 0.0
        points = np.flatnonzero(said)
        pairs = [(float(frequencies[a]), float(frequencies[b]))
                 for a, b in itertools.pairwise(points)
                 if b == a + 1 and frequencies[b] > frequencies[a]]
    else:
        left, right = spectrum.bin_bounds()
        said &= right > 0.0
        pairs = [(float(max(left[i], 0.0)), float(right[i]))
                 for i in np.flatnonzero(said) if right[i] > max(left[i], 0.0)]
    merged: list[tuple[float, float]] = []
    for low, high in sorted(pairs):
        if merged and low <= merged[-1][1] * (1.0 + 1e-12):
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def _pieces(low: float, high: float,
            stretches: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """(low, high) cut to `stretches`: the parts of it inside them."""
    out = []
    for a, b in stretches:
        lo, hi = max(low, a), min(high, b)
        if hi > lo:
            out.append((lo, hi))
    return out


def comparable(specification: Specification, measured: DataArray) -> str | None:
    """Why this pair cannot be compared, or None when it can.

    A specification on octave bands compares only with a response on
    bands of the same fraction, and a response on octave bands only with
    such a specification: a band's power cannot be assigned to part of
    its width, and a requirement on bands says nothing about the lines
    under them. A narrowband response compares with a curve at
    breakpoints or with a specification on lines. The reason is a phrase
    the table and the status bar can show.

    Parameters
    ----------
    specification, measured : Specification, DataArray
        The pair.

    Returns
    -------
    str or None
        The reason, or None.
    """
    # bands pair only with bands of the same fraction (Brandon,
    # 2026-09-19)
    from .octave import per_octave_of

    on_bands = getattr(specification, 'bandwidth', None) is not None
    banded = getattr(measured, 'bandwidth', None) is not None
    if on_bands and not banded:
        return ('a specification on octave bands compares only with a '
                'response on the same bands; band the response the same way')
    if banded and not on_bands:
        return ('a response on octave bands compares only with a '
                'specification on the same bands; band the specification '
                'the same way')
    if on_bands and banded:
        asked = per_octave_of(specification.abscissa)
        held = per_octave_of(measured.abscissa)
        if asked != held:
            return (f'a specification on 1/{asked}-octave bands compares only '
                    f'with a response on 1/{asked}-octave bands, not 1/{held}')
    return None


def cells(specification: Specification, measured: DataArray,
          spec_record: int = 0, measured_record: int = 0
          ) -> list[dict[str, Any]]:
    """The cells one comparison is judged over.

    Against a curve at breakpoints, the cells are the measurement's own
    lines. When both are densities on bins, the cells are the bins of the
    coarser grid, by median bin width: a requirement written per octave
    band is a requirement on the band's power, not on every line under
    it. A banded measurement compares only with a specification on bands
    of the same fraction, never with a breakpoint curve (`comparable`),
    and a pair that cannot be compared has no cells.

    Each cell is cut to what both objects are written over: the
    specification's ranges (`coverage`, where a controller's zero lines
    are gaps) and the measurement's. A cell with nothing written in it is
    not judged, and a cell written over part of its width is judged over
    that part, both sides integrated over the same range. The same rule
    applies at the ends of the band and in the middle.

    Parameters
    ----------
    specification, measured : Specification, DataArray
        The pair.
    spec_record, measured_record : int
        Which record of each.

    Returns
    -------
    list of dict
        Each with 'low' and 'high' (the cell's outer reach), 'pieces'
        (the stretches judged, inside it), 'width' (their sum) and
        'span' (the whole bin it came from, judged or not).
    """
    # one rule at the ends and in the middle alike (Brandon,
    # 2026-09-19; PLAN.md "A comparison is an area against an area")
    if comparable(specification, measured) is not None:
        return []
    asked = coverage(specification, spec_record)
    held = coverage(measured, measured_record)
    if not asked or not held:
        return []
    if getattr(specification, 'interpolation', 'bin') == 'log_log':
        grid = measured.bin_bounds()
    else:
        sl, sr = specification.bin_bounds()
        ml, mr = measured.bin_bounds()
        s_said = specification.written(spec_record)
        m_said = measured.written(measured_record)
        s_width = float(np.median((sr - sl)[s_said])) if s_said.any() else 0.0
        m_width = float(np.median((mr - ml)[m_said])) if m_said.any() else 0.0
        grid = (sl, sr) if s_width > m_width * (1.0 + 1e-9) else (ml, mr)
    out = []
    for low, high in zip(*grid):
        pieces = [piece for a, b in _pieces(float(low), float(high), asked)
                  for piece in _pieces(a, b, held)]
        if pieces:
            out.append({'low': pieces[0][0], 'high': pieces[-1][1],
                        'pieces': pieces,
                        'width': float(sum(b - a for a, b in pieces)),
                        # the bin this cell came from, whole. `width`
                        # over `span` is how much of one line or one
                        # band was judged, which is what a share of
                        # the comparison counts in.
                        'span': float(high) - float(low)})
    return out


def judge(specification: Specification, measured: DataArray,
          spec_record: int = 0, measured_record: int = 0,
          limit: str | None = 'abort_upper', over: bool = True,
          scale_db: float | None = None) -> dict[str, Any]:
    """Which cells of a comparison fall outside one limit, and where.

    For each cell, the measurement's power (its area, integrated the way
    it is drawn, times the comparison's scale) is compared with the
    limit's power over the same range. The limit is integrated the way
    its specification is: the exact area under the power law between
    breakpoints, or a density per line or per band. A cell is out when
    it holds more than an upper limit (`over` True) or less than a lower
    one (`over` False). `limit` None judges against the target itself.

    Parameters
    ----------
    specification, measured : Specification, DataArray
        The pair.
    spec_record, measured_record : int, default 0
        Which record of each.
    limit : str or None, default 'abort_upper'
        The limit curve, one of `Specification.LIMITS`, or None for
        the target.
    over : bool, default True
        True when holding more than the curve is out, False when
        holding less is.
    scale_db : float, optional
        The comparison's scale in dB, added to the measurement;
        resolved through `comparison_scale_db` when omitted.

    Returns
    -------
    dict
        Per cell: 'cells' (as `cells` gives them), 'asked' and 'held'
        (the limit's and the measurement's power), 'judged' (True where
        both are finite and the limit's power is positive) and 'out'.
        Per line of the measurement: 'lines' (True where the line's cell
        is out), 'level' (the limit's mean density over that cell on the
        marked lines, NaN on the others; where the plot draws a mark),
        'start' and 'stop' (the judged range within each marked line's
        bin, the bin's own edges elsewhere). And 'scale_db', the scale
        used.
    """
    if scale_db is None:
        scale_db = comparison_scale_db(specification, measured)
    scale = 10.0 ** (float(scale_db) / 10.0)
    found = cells(specification, measured, spec_record, measured_record)
    # every piece of every cell at once — one vectorized read of each
    # object — then summed back per cell
    owner = np.array([k for k, cell in enumerate(found)
                      for _piece in cell['pieces']], dtype=int)
    lows = np.array([a for cell in found for a, _b in cell['pieces']])
    highs = np.array([b for cell in found for _a, b in cell['pieces']])
    count = len(found)
    if count:
        wanted = (specification.areas(spec_record, lows, highs) if limit is None
                  else specification.limit_areas(limit, spec_record, lows, highs))
        got = measured.areas(measured_record, lows, highs) * scale
        asked = np.bincount(owner, weights=np.nan_to_num(wanted), minlength=count)
        asked[np.bincount(owner, weights=~np.isfinite(wanted), minlength=count) > 0] = np.nan
        held = np.bincount(owner, weights=np.nan_to_num(got), minlength=count)
        held[np.bincount(owner, weights=~np.isfinite(got), minlength=count) > 0] = np.nan
    else:
        asked = held = np.zeros(0)
    with np.errstate(invalid='ignore'):
        judged = np.isfinite(asked) & (asked > 0.0) & np.isfinite(held)
        out = judged & ((held > asked) if over else (held < asked))
    left, right = measured.bin_bounds()
    lines = np.zeros(left.shape, dtype=bool)
    level = np.full(left.shape, np.nan)
    start, stop = left.copy(), right.copy()
    # the lines each out cell reaches, found by where the cell's ends
    # fall among the sorted bins: a bin that ends where the cell
    # begins shares an edge and no width, and is not in it
    for k in np.flatnonzero(out):
        cell = found[k]
        first = int(np.searchsorted(right, cell['low'] * (1.0 + 1e-12), side='right'))
        last = int(np.searchsorted(left, cell['high'] * (1.0 - 1e-12), side='left'))
        if last <= first:
            continue
        inside = slice(first, last)
        lines[inside] = True
        level[inside] = asked[k] / cell['width']
        start[inside] = np.maximum(left[inside], cell['low'])
        stop[inside] = np.minimum(right[inside], cell['high'])
    return {'cells': found, 'asked': asked, 'held': held, 'out': out,
            'judged': judged, 'lines': lines, 'level': level,
            'start': start, 'stop': stop, 'scale_db': float(scale_db)}


def covered(lines: ArrayLike, low: float, high: float,
            widths: ArrayLike | None = None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each measured bin cut to the part of it inside the band from `low`
    to `high`.

    A line stands for its whole bin, and at the ends of a specification
    a bin extends past the edge. Returns (start, stop, width, cut): where
    each bin lies inside the band, how wide that part is, and which bins
    the band's edge cuts. A bin wholly outside has zero width and is not
    cut. With `widths` given, the bins are proportional bands with edges
    from `octave.bin_bounds`; without, the bins end at the midpoints
    between neighbors. Fewer than two lines gives arrays of zeros.
    """
    lines = np.asarray(lines, dtype=float)
    if lines.size < 2:
        empty = np.zeros(lines.shape)
        return empty, empty, empty, np.zeros(lines.shape, dtype=bool)
    if widths is None:
        widths = np.gradient(lines)
        left, right = lines - widths / 2.0, lines + widths / 2.0
    else:
        from .octave import bin_bounds

        widths = np.asarray(widths, dtype=float)
        left, right = bin_bounds(lines, widths)
    start = np.maximum(left, low)
    stop = np.minimum(right, high)
    width = np.clip(stop - start, 0.0, None)
    return start, stop, width, (width > 0.0) & (width < widths * (1.0 - 1e-9))


def outside(lines: ArrayLike, values: ArrayLike,
            spec_frequencies: ArrayLike, limit_values: ArrayLike,
            over: bool = True, reading: str = 'log_log',
            spec_widths: ArrayLike | None = None,
            widths: ArrayLike | None = None) -> np.ndarray:
    """Which measured lines fall outside one limit curve, for a limit and
    a measurement held as plain arrays: the array form of `judge`.

    The limit is read as `reading`: 'log_log' between its points, or
    'bin' as a density per bin of `spec_widths`. The measurement is a
    density per bin of `widths` (bins ending at the midpoints between
    neighbors when None). The two go through the same cell rule as every
    other comparison, against the limit itself with no scale. Returns a
    bool per line, all False when there are fewer than two lines or
    limit points, or when only one of the two widths is given (a banded
    object does not compare with an unbanded one).
    """
    from .data import Psd, Specification

    lines = np.asarray(lines, dtype=float)
    values = np.asarray(np.real(values), dtype=float)
    spec_frequencies = np.asarray(spec_frequencies, dtype=float)
    limit_values = np.asarray(np.real(limit_values), dtype=float)
    if lines.size < 2 or spec_frequencies.size < 2:
        return np.zeros(lines.shape, dtype=bool)
    if (spec_widths is None) != (widths is None):
        return np.zeros(lines.shape, dtype=bool)   # not comparable
    curve = Specification(spec_frequencies, np.atleast_2d(limit_values),
                          response_dof=['1X+'], ordinate_dim=['unknown'],
                          bandwidth=spec_widths)
    curve.interpolation = reading
    held = Psd(lines, np.atleast_2d(values), response_dof=['1X+'],
               ordinate_dim=['unknown'], bandwidth=widths)
    return judge(curve, held, 0, 0, limit=None, over=over, scale_db=0)['lines']


def band_of(specification: Specification,
            record: int = 0) -> tuple[float, float] | None:
    """(low, high), the range the specification is written over: the
    outer ends of its `coverage`. None when nothing is written."""
    stretches = coverage(specification, record)
    if not stretches:
        return None
    return stretches[0][0], stretches[-1][1]


def specification_rms(specification: Specification, record: int = 0,
                      low: float | None = None,
                      high: float | None = None) -> float:
    """A specification's RMS, from its own points and nothing else.

    It is not computed on a measurement's frequency grid. The level a
    specification asks for belongs to the specification, and it should
    not change in the fourth decimal place because a frame length
    changed.

    The integral is the specification's own `area`, taken the way the
    object is drawn: the exact power-law area for a specification
    written at breakpoints, the sum over bins for one computed from a
    record (the PSD of a transient target is a `Specification` of
    thousands of density lines). `low` and `high`, in Hz, limit the band.
    """
    # This used to call `log_log_area` directly, which is right for a
    # specification written at breakpoints and wrong for one computed
    # from a record.
    return float(np.sqrt(specification.area(record, low, high)))


def rms(frequencies: ArrayLike, values: ArrayLike) -> float:
    """The RMS a PSD carries over the lines given: the square root of
    its total.

    Each line times the width of its own bin, summed. A discrete
    spectrum is a density per bin, so this is the total the lines hold,
    exactly as Parseval's theorem gives it, where the trapezoid rule
    halves the two end bins and reads a little low.

    The widths come from the whole axis before any line is dropped, so
    a NaN or negative line contributes nothing and its bin is not added
    to its neighbors'; bridging a gap would assume what is in it. NaN
    when there are fewer than two lines or no usable one.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    values = np.asarray(np.real(values), dtype=float)
    if frequencies.size < 2:
        return float('nan')
    # gradient is the spacing itself on an even axis, ends included, and
    # a sensible bin either side of a line on an uneven one
    widths = np.gradient(frequencies)
    good = np.isfinite(values) & np.isfinite(frequencies) & (values >= 0.0)
    if good.sum() < 1:
        return float('nan')
    return float(np.sqrt(np.sum(values[good] * widths[good])))


def bounds(specification: Specification, record: int, pair: str,
           frequencies: ArrayLike
           ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(lower, upper) of one pair of limits, interpolated log-log onto
    `frequencies`.

    `pair` is 'warning' or 'abort'. None for a limit the specification
    does not carry: nothing can be exceeded where nothing was specified.
    """
    out = []
    for edge in (f'{pair}_lower', f'{pair}_upper'):
        values = specification.limits.get(edge)
        out.append(None if values is None else log_interpolate(
            frequencies, specification.abscissa, values[record]))
    return out[0], out[1]


#: how far below a channel's own specification peak a line still
#: carries information about *level*. Beyond it the requirement has
#: essentially no content, and the measurement there is the article's
#: noise floor rather than a reading of how hard it was driven.
#:
#: Measured on the drone transient (Brandon, 2026-08-25): its
#: specification is the PSD of a target waveform that stops at about
#: 2 kHz, so above that it sits at 5e-15 against a measurement at
#: 5e-11 — a 40 dB difference that says nothing about level and
#: occupied **51% of the frequency lines**. The whole-band median was
#: dragged from about -4 dB to -26. Restricting to the band the
#: specification actually occupies changed no other project on file,
#: at any threshold from 20 to 60 dB.
DETECTION_RANGE_DB = 40.0


def significant_band(want: np.ndarray, good: np.ndarray) -> np.ndarray:
    """`good`, narrowed to the lines where the specification has real
    content.

    Lines within `DETECTION_RANGE_DB` of this channel's own peak in
    `want`. Relative to the peak rather than a fixed level, because a
    specification's units and size are its own; per channel, because
    each channel is specified at its own level.
    """
    if not good.any():
        return good
    peak = float(np.nanmax(want[good]))
    if not peak > 0.0:
        return good
    return good & (want >= peak / 10.0 ** (DETECTION_RANGE_DB / 10.0))


#: the ladder a commanded level sits on, in decibels: runs are run at
#: +3, 0, -3, -6, -9, -12 … and the detected offset snaps to it
#: (Brandon, 2026-09-19, from the whole decibels of 2026-08). A typed
#: scale is still any whole decibel.
SCALE_STEP_DB = 3

#: the commanded levels a detected scale may stand for, as what is
#: *added* to the measurement: a run at -12 dB needs +12, a run at
#: +6 dB needs -6. Controllers command on a ladder near 0 dB — run-ups
#: from -12 or -24, now and then +3 or +6 — and nobody commands +54.
#: Past this a measurement that far off its specification is a fault to
#: show, never a level to divide out (2026-09-26: a controller that blew
#: up ran 50 dB hot on every channel, eight of them agreeing within
#: 2 dB, and was charted -54 dB "scaled" — an overtest read as an
#: undertest). Lopsided on purpose: a measurement hotter than its
#: specification is the most dangerous thing a compliance check
#: reports, so that side gets the least room.
DETECTION_MIN_DB = -6
DETECTION_MAX_DB = 24


class ScaleWarning(UserWarning):
    """Warned when a comparison scaled the measurement by a level it
    detected from the data rather than one it was given. The charts
    built from `channel_errors` do not show the scale, so the warning
    does."""


#: how tightly the bands must agree before their mode is read as a
#: commanded level: at least this share of every compared band within
#: half a ladder step of the mode. This is the spread test, and the only
#: one: a standard deviation (tried first, at 2 dB) is dominated by tails,
#: and real control has them — on 22 real runs held on level, a few deep
#: bands of under-driven out-of-plane channels put σ at 2.4-4.7 dB with
#: 89% of the bands within 1.5 dB, and σ showed them unscaled. A robust
#: spread (the scaled MAD) fixed that, but with this share and the rung
#: test below it could never decide alone — its one possible case, two
#: groups straddling a rung, puts the median off the rung — and the
#: campaign's 242 scaled copies read the same with it or without it: 210
#: exact, none wrong (the other session's check, 2026-09-26).
DETECTION_AGREEMENT = 0.75

#: and the bands must center on the rung: their median within this of
#: it. Tight bands between two rungs are a controller that held the
#: wrong level — 1.6 dB low sits 1.4 dB from +3, inside half a step of
#: it on every band, and without this reads as a run at -3 dB.
DETECTION_RUNG_DB = 0.75


def detect_scale_db(specification: Specification, measured: DataArray,
                    spec_records: Sequence[int] | None = None,
                    measured_records: Sequence[int] | None = None,
                    controls: Sequence[str] | None = None) -> int:
    """The offset in dB, a multiple of 3 dB, that best lays the
    measurement on the specification, or zero when the data does not
    show one clearly.

    A run captured at -6 dB needs +6 dB added before it is compared with
    the 0 dB specification. Both objects are banded to sixth octaves
    (`octave.PER_OCTAVE`); a pair already banded is used as it is. Every
    band of every compared channel gives the dB it would need added,
    rounded to the nearest multiple of `SCALE_STEP_DB`, and the most
    common value (the mode) is the candidate. It is taken as the test's
    level only if all of these hold:

    - it lies within `DETECTION_MIN_DB` to `DETECTION_MAX_DB`, a level a
      controller would command (a run 50 dB hot is a failure, not a
      level);
    - at least `DETECTION_AGREEMENT` of the bands lie within half a step
      of it, and their median is within `DETECTION_RUNG_DB` of it, so the
      bands agree on one level and that level is on a step;
    - the total power agrees with it within `ERROR_DB`. A change of level
      moves the band median and the total power together. A run that is
      low between its resonances and hot at them has most bands agreeing
      on a low level while its power is at the peaks; the agreement test
      does not see the peaks, which are a minority of the bands, and the
      total power does.

    Otherwise the result is zero and the comparison is shown as
    measured. Scaling on doubtful evidence is worse than showing the
    mismatch, because a wrong scale can turn a failed test into a passed
    one. The Scaling field is where the true level is typed.

    The mode is used because a controller scales every channel together,
    so the level is one number for the whole test. The control channels
    fall on the same step, while monitor channels, each its own distance
    under its envelope, spread over many; the mode finds the controls,
    where a median of all the bands finds wherever the monitors happen
    to fall. Bands are used because in narrowband each line is a noisy
    estimate and many lines are at anti-resonances no controller holds:
    runs at full level read +14, +9 and +4 dB on lines and 0 dB in
    sixth-octave bands.

    A full-level test the controller held 3 dB low everywhere looks the
    same in the data as a test commanded at -3 dB, and is read as one.
    The commanded level is not in a Rattlesnake file, so a detected level
    is always reported where it is used: the report's test-level box,
    the captions and `ScaleWarning`.

    `controls` names the control channels (by DOF, or by the label
    `matched_records` gives) when the caller knows them, and only those
    are counted. Without it, a specification that also bounds many
    monitor channels spreads its bands and reads zero. Only the bands
    where the specification has real content are used
    (`significant_band`, per channel).
    """
    # The mode of the band levels, when the bands agree on it (Brandon,
    # 2026-09-26). It replaced, the same day, a per-channel vote with a
    # floor veto and gates (three hot controls outvoted twenty-one), then
    # a pooled median (monitors outvoted the controls). The +14, +9 and
    # +4 dB narrowband readings are from a 22-run campaign scored in
    # another session. Banding also matches the verdict, which is read
    # off the octave comparison, so the level and the verdict come from
    # the same bands. `significant_band` is there because a waveform's
    # spectrum ending at 2 kHz left half a transient's bands 40 dB apart,
    # saying nothing about level.
    from .octave import PER_OCTAVE

    if comparable(specification, measured) is not None:
        return 0
    if getattr(measured, 'bandwidth', None) is None:
        if not (hasattr(specification, 'to_octave')
                and hasattr(measured, 'to_octave')):
            return 0
        specification = specification.to_octave(PER_OCTAVE)
        measured = measured.to_octave(PER_OCTAVE)
    wanted = None if controls is None else {str(c) for c in controls}
    offsets, asked_total, held_total = [], 0.0, 0.0
    for label, spec_index, measured_index in matched_records(
            specification, measured, spec_records, measured_records):
        if wanted is not None and label not in wanted and str(
                specification.response_dof[spec_index]) not in wanted:
            continue
        # band by band, the same cells the comparison is judged on —
        # each a density: the power asked and held over the same band,
        # over its width
        level = judge(specification, measured, spec_index, measured_index,
                      None, True, scale_db=0)
        width = np.array([cell['width'] for cell in level['cells']],
                         dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            want = level['asked'] / width
            got = level['held'] / width
        good = np.isfinite(want) & (want > 0.0) & np.isfinite(got) & (got > 0.0)
        # the power over every finite band, as `compare` sums it
        whole = np.isfinite(level['asked']) & np.isfinite(level['held'])
        asked_total += float(np.sum(level['asked'][whole]))
        held_total += float(np.sum(level['held'][whole]))
        good = significant_band(want, good)
        offsets.append(10.0 * np.log10(want[good] / got[good]))
    offsets = np.concatenate(offsets) if offsets else np.array([])
    if not offsets.size or not asked_total > 0.0 or not held_total > 0.0:
        return 0
    rungs = np.round(offsets / SCALE_STEP_DB) * SCALE_STEP_DB
    values, counts = np.unique(rungs, return_counts=True)
    # the most common rung; a tie goes to the one nearest zero, the
    # answer that scales least
    best = counts.max()
    candidate = int(min(values[counts == best], key=abs))
    agreement = float(np.mean(np.abs(offsets - candidate)
                              <= SCALE_STEP_DB / 2))
    if (not DETECTION_MIN_DB <= candidate <= DETECTION_MAX_DB
            or agreement < DETECTION_AGREEMENT
            or abs(float(np.median(offsets)) - candidate) > DETECTION_RUNG_DB
            or abs(10.0 * np.log10(asked_total / held_total) - candidate)
            > ERROR_DB):
        return 0
    return candidate


def comparison_scale_db(specification: Specification, measured: DataArray,
                        spec_records: Sequence[int] | None = None,
                        measured_records: Sequence[int] | None = None,
                        controls: Sequence[str] | None = None) -> int:
    """The dB every comparison adds to `measured`: the value set on the
    object (`scale_db`, zero included), or the detected one
    (`detect_scale_db`) when none is set. `controls`, when known, names
    the channels the detection uses. Every comparison gets its scale
    here, so the drawn curves, the error metrics and the report cannot
    disagree."""
    held = getattr(measured, 'scale_db', None)
    if held is not None:
        return int(held)
    return detect_scale_db(specification, measured,
                           spec_records, measured_records, controls)


def matched_records(specification: Specification, measured: DataArray,
                    spec_records: Sequence[int] | None = None,
                    measured_records: Sequence[int] | None = None
                    ) -> list[tuple[str, int, int]]:
    """[(label, spec record, measured record)] for the channels the two
    have in common.

    Records are paired by response and reference DOF and by quantity, in
    the specification's order. A record of unknown quantity pairs with
    any quantity, and an exact match is preferred. The label is the DOF,
    or 'response/reference' for a cross term. `compare_all` and the
    scale detection both pair channels here, so the detection uses
    exactly the channels the comparison compares."""
    def pairs(data: DataArray,
              records: Sequence[int] | None) -> dict[tuple, int]:
        # keyed by DOF pair *and* quantity: a drive point's force
        # record shares its DOF name with the acceleration channel
        # measured beside it, and before the quantity joined the key
        # the pairing only came out right when the acceleration
        # happened to be stored first
        wanted = range(data.num_records) if records is None else records
        out = {}
        for i in wanted:
            i = int(i)
            reference = (data.reference_dof[i]
                         if data.reference_dof is not None
                         else data.response_dof[i])
            out.setdefault((data.response_dof[i], reference,
                            data.known_dim(i)), i)
        return out

    theirs = pairs(measured, measured_records)
    rows = []
    for pair, index in pairs(specification, spec_records).items():
        # 'unknown' matches anything: an undefined-units record has
        # made no claim to contradict
        candidates = [key for key in theirs
                      if key[:2] == pair[:2]
                      and (key[2] == pair[2]
                           or 'unknown' in (key[2], pair[2]))]
        if not candidates:
            continue
        exact = [key for key in candidates if key[2] == pair[2]]
        chosen = (exact or candidates)[0]
        label = pair[0] if pair[0] == pair[1] else f'{pair[0]}/{pair[1]}'
        rows.append((label, index, theirs[chosen]))
    return rows


def exceedances(specification: Specification, measured: DataArray,
                spec_record: int = 0, measured_record: int = 0,
                pair: str = 'abort',
                scale_db: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(over, under): which measured lines are above the upper and below
    the lower limit of one pair.

    `pair` is 'warning' or 'abort'. These are the lines the plot marks,
    so a line out of tolerance can be found by looking at the plot. The
    lines judged are those of the scaled measurement, the same curve
    the comparison draws; `scale_db` is resolved through
    `comparison_scale_db` unless given. A limit the specification does
    not carry gives all False.
    """
    if scale_db is None:
        scale_db = comparison_scale_db(specification, measured)
    marks = []
    for edge, over in ((f'{pair}_upper', True), (f'{pair}_lower', False)):
        if specification.limits.get(edge) is None:
            marks.append(np.zeros(np.shape(measured.abscissa), dtype=bool))
        else:
            marks.append(judge(specification, measured, spec_record,
                               measured_record, edge, over, scale_db)['lines'])
    return marks[0], marks[1]


def compare(specification: Specification, measured: DataArray,
            spec_record: int = 0,
            measured_record: int = 0,
            scale_db: float | None = None) -> dict[str, Any]:
    """How one measured record compares with one record of a
    specification.

    Returns a dict: 'lines' (how many cells were judged), 'band' (low
    and high, Hz), 'scale_db', 'specification_rms' and 'measured_rms',
    'difference_percent' and 'difference_db' (the measured RMS against
    the specification's; 20 log10 for the dB), and for each pair of
    limits the specification carries, 'warning_lines' and
    'warning_percent' or 'abort_lines' and 'abort_percent': how many
    judged cells fell outside the pair and their share of the
    comparison. A pair the specification does not carry is absent
    rather than zero. The share counts cells, each weighted by the
    fraction of its bin that was judged, so a band at the top of the
    range counts the same as one at the bottom.

    A pair of objects that cannot be compared (`comparable`) returns
    {'lines': 0, 'refused': reason}, and one with nothing to compare
    returns {'lines': 0}.

    Every number is of the scaled measurement: `scale_db` is resolved
    through `comparison_scale_db` unless given, and is returned so a
    table can say what was compared.
    """
    if scale_db is None:
        scale_db = comparison_scale_db(specification, measured)
    # the level, cell by cell: both integrated over the very same
    # stretches, the specification from its own points exactly and the
    # measurement from its bins, so a response sitting on its
    # specification comes out at nothing rather than a quarter of a
    # percent over
    refused = comparable(specification, measured)
    if refused is not None:
        return {'lines': 0, 'refused': refused}
    level = judge(specification, measured, spec_record, measured_record,
                  None, True, scale_db)
    found = level['cells']
    good = np.isfinite(level['asked']) & np.isfinite(level['held'])
    if not good.any():
        return {'lines': 0}
    low, high = found[0]['low'], found[-1]['high']
    spec_rms = float(np.sqrt(np.sum(level['asked'][good])))
    got_rms = float(np.sqrt(np.sum(level['held'][good])))
    counted = good & level['judged']
    out = {
        'lines': int(counted.sum()),
        'band': (low, high),
        'scale_db': float(scale_db),
        'specification_rms': spec_rms,
        'measured_rms': got_rms,
        'difference_percent': (
            float('nan') if not spec_rms or not np.isfinite(spec_rms)
            else 100.0 * (got_rms - spec_rms) / spec_rms),
        # the same difference in dB, which is how a tolerance on it is
        # written and read. An RMS is an amplitude, so twenty log ten —
        # and the two forms disagree either side of zero (+3 dB is
        # +41.3%, -3 dB is -29.2%), which is exactly why a bar chart
        # with a symmetric threshold wants this one.
        'difference_db': (
            float('nan') if not spec_rms or not got_rms
            or not np.isfinite(spec_rms) or not np.isfinite(got_rms)
            else 20.0 * np.log10(got_rms / spec_rms)),
    }
    # the cells outside a pair of limits, and the share of the
    # comparison they make.
    #
    # Counted in cells, each weighted by how much of itself was judged
    # — `width` over `span`. This used to be a share of frequency
    # *width* in hertz, on the reasoning that it reads the same
    # whether the cells are lines or bands. It does not (Brandon,
    # 2026-09-21). Proportional bands are equal in *log* frequency, so
    # in hertz they grow geometrically: in a sixth-octave set from
    # 10 Hz to 2 kHz the lowest band is 1.2 Hz wide and the highest
    # 243 Hz, and six bands outside abort read 0.45 % at the bottom
    # against 44.7 % at the top. The same six bands, a hundredfold
    # apart, and a channel that should have tripped the ten-percent
    # rule did not.
    #
    # Narrowband numbers do not move, to every digit: those bins are
    # all one width, so weighting by the fraction judged and weighting
    # by hertz are the same ratio. Cells *are* cut — at the ends of
    # every written stretch the outermost line is judged over half its
    # own bin — which is why this is a weighted count and not a plain
    # one.
    share = np.array([cell['width'] / cell['span'] if cell['span'] else 0.0
                      for cell in found], dtype=float)
    for pair in ('warning', 'abort'):
        if not any(f'{pair}_{edge}' in specification.limits
                   for edge in ('lower', 'upper')):
            continue
        beyond = np.zeros(len(found), dtype=bool)
        for edge, over in ((f'{pair}_upper', True), (f'{pair}_lower', False)):
            if specification.limits.get(edge) is not None:
                beyond |= judge(specification, measured, spec_record,
                                measured_record, edge, over, scale_db)['out']
        beyond &= counted
        out[f'{pair}_lines'] = int(beyond.sum())
        out[f'{pair}_percent'] = (100.0 * float(share[beyond].sum())
                                  / float(share[counted].sum())
                                  if counted.any() else 0.0)
    return out


#: how far the RMS may be out before a channel is called out, in dB.
#: Three is the tolerance a random vibration specification is usually
#: written with, and it is a starting point rather than a rule — the
#: bar charts let it be dragged.
ERROR_DB = 3.0

#: and how much of a channel's band may sit outside abort before the
#: same is said of it, as a percentage of the lines compared
LINES_PERCENT = 10.0


#: the share of control channels out on either reading past which the
#: test failed (Brandon, 2026-09-20): a fifth of the channels with
#: more than `LINES_PERCENT` of their band outside the abort limits,
#: or a tenth of them more than `ERROR_DB` off in RMS. At the number
#: exactly, failed — "20% or more is a failure".
LINES_CHANNELS_FAIL_PERCENT = 20.0
RMS_CHANNELS_FAIL_PERCENT = 10.0


def verdict(rows: Sequence[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Whether a run passed, from the rows `compare_all` returned.

    Two shares of the channels judged: those with more than
    `LINES_PERCENT` of their band outside the abort limits, and those
    more than `ERROR_DB` off in RMS. The run passes when the first is
    under `LINES_CHANNELS_FAIL_PERCENT` and the second under
    `RMS_CHANNELS_FAIL_PERCENT`; either one at or over its threshold
    fails it. Channels with nothing to compare are not counted, and with
    no channels at all there is no verdict ('passed' None).

    Parameters
    ----------
    rows : sequence of (str, dict)
        What `compare_all` returned.

    Returns
    -------
    dict
        'passed' (True, False or None), 'channels' (how many were
        judged), 'lines_percent' and 'rms_percent' (the two shares),
        and the four thresholds they were read against.
    """
    # the pass rule (Brandon, 2026-09-20)
    judged =[result for _label, result in rows if result.get('lines')]
    out = {'channels': len(judged),
           'lines_limit': LINES_PERCENT, 'rms_limit': ERROR_DB,
           'lines_fail_percent': LINES_CHANNELS_FAIL_PERCENT,
           'rms_fail_percent': RMS_CHANNELS_FAIL_PERCENT}
    if not judged:
        return {**out, 'passed': None, 'lines_percent': float('nan'),
                'rms_percent': float('nan')}
    lines_out = sum(1 for r in judged
                    if float(r.get('abort_percent') or 0.0) > LINES_PERCENT)
    rms_out = sum(1 for r in judged
                  if abs(float(r.get('difference_db', 0.0))) > ERROR_DB)
    lines_percent = 100.0 * lines_out / len(judged)
    rms_percent = 100.0 * rms_out / len(judged)
    passed = (lines_percent < LINES_CHANNELS_FAIL_PERCENT
              and rms_percent < RMS_CHANNELS_FAIL_PERCENT)
    return {**out, 'passed': passed, 'lines_percent': lines_percent,
            'rms_percent': rms_percent}


def channel_errors(rows: Sequence[tuple[str, dict[str, Any]]]
                   ) -> list[tuple[str, float, float]]:
    """[(label, dB out, percent of lines outside abort)] for a bar chart.

    One row per channel, in the specification's order, from the rows
    `compare_all` returned. Channels with nothing to compare are left
    out, since a bar at zero would read as a channel that matched.
    """
    out = []
    for label, result in rows:
        if not result.get('lines'):
            continue
        out.append((label, float(result.get('difference_db', float('nan'))),
                    float(result.get('abort_percent') or 0.0)))
    return out


def outside_fraction(values: Sequence[float], low: float,
                     high: float | None = None) -> float:
    """The percentage of these values outside the threshold.

    With `high` None the test is one-sided, values at or above `low`:
    the lines-out chart has a ceiling and no floor, because staying
    inside the abort limits is never a fault. Otherwise a value is
    outside when it is above `high` or below `low`. Values that are not
    finite are ignored, and none left gives zero.
    """
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return 0.0
    if high is None:
        beyond = sum(1 for v in values if v >= low)
    else:
        beyond = sum(1 for v in values if v > high or v < low)
    return 100.0 * beyond / len(values)


def compare_all(specification: Specification, measured: DataArray,
                spec_records: Sequence[int] | None = None,
                measured_records: Sequence[int] | None = None,
                scale_db: float | None = None
                ) -> list[tuple[str, dict[str, Any]]]:
    """[(label, result)] for every channel the two have in common, each
    result from `compare`.

    A specification bounds the control channels, and a measurement
    usually holds those and many more. Channels are paired by DOF, the
    response and the reference where there is one (`matched_records`),
    so a cross spectrum is compared with the specification's cross term
    if it has one and is left out if it does not.

    Rows are in the specification's order, the order a control room
    reads its channels in. The scale is resolved once for the whole set,
    so one measurement gets one scale, never a different value per
    channel. A caller comparing a derived form of the data (the report's
    own octave banding) passes the scale it resolved on the original, so
    the two cannot round to different values.

    A nonzero scale detected here (none passed and none set on the
    measurement) is warned as a `ScaleWarning`: every row carries it as
    'scale_db', but `channel_errors` and the charts built from it do
    not. Pass `scale_db=0` to compare as measured.
    """
    # the warning is there because a whole campaign was once scored on
    # shifted numbers without a word (2026-09-26)
    if scale_db is None:
        scale_db = comparison_scale_db(specification, measured,
                                       spec_records, measured_records)
        if scale_db and getattr(measured, 'scale_db', None) is None:
            import warnings

            warnings.warn(
                f'the measurement was scaled by {scale_db:+g} dB, detected '
                f'from the data as a run at {-scale_db:+g} dB; pass '
                'scale_db=0 to compare it as measured, or set '
                'measured.scale_db to the level it was run at',
                ScaleWarning, stacklevel=2)
    return [(label, compare(specification, measured, spec_index,
                            measured_index, scale_db=scale_db))
            for label, spec_index, measured_index in matched_records(
                specification, measured, spec_records, measured_records)]


#: The RMS dB deviation an SRS comparison is shaded past, either side.
#: The same three decibels the level error either side of a
#: specification uses, because it is the same judgment in the same
#: unit — how far a spectrum sits from what was asked for, and
#: over-testing and under-testing are not the same fault.
SRS_ERROR_DB = 3.0


def srs_errors(specification: DataArray, measured: DataArray
               ) -> list[tuple[str, str | None, float]]:
    """[(label, event, RMS dB deviation)] for a shock response spectrum
    against the one it was required to meet.

    The deviation is in dB, RMS across the band, and signed
    (`signed_rms_db` on each channel's lines). An SRS spans decades and
    is read on log axes. A relative norm, `||m - s|| / ||s||` as a time
    waveform gets, is dominated by the loudest bands and does not see a
    large ratio error where the target is small. In dB every band counts
    the same, which is how a shock is judged, and the threshold beside it
    (`SRS_ERROR_DB`) is the same 3 dB a level error gets.

    RMS rather than the worst band, because the worst band is one point
    and one point can be noise: a curve 4 dB out everywhere and a curve
    with a single 4 dB peak are different results. The worst band is
    still visible on the plot.

    The sign is the sign of the mean dB deviation, the side of the
    requirement the spectrum is mostly on, so an under-test reads
    negative. A curve on both sides of the requirement (a large RMS with
    a mean near zero) takes the side its mean is on; the magnitude is
    what fails it, and either bound is the same 3 dB away.

    The specification is interpolated log-log onto the measurement's own
    frequencies, and only where it is written. Channels are paired by
    response DOF, and the event is the measurement's block label for the
    record, None when it has none.
    """
    # Why dB: on the airplane run the relative norm passed every channel
    # at 20 % while one of them was 25 dB over, 99.5 % of its norm coming
    # from the top two decades. Why signed: an unsigned RMS read an
    # under-test as an over-test. A spectrum 5 dB low everywhere reported
    # +5, the bar chart shaded it past the ceiling, and the reader
    # concluded the shock was too hard when the machine had under-hit
    # (Brandon, on the drone stress set, 2026-08-20).
    wanted = {}
    for j in range(specification.num_records):
        wanted.setdefault(str(specification.response_dof[j]), j)
    out = []
    for i in range(measured.num_records):
        dof = str(measured.response_dof[i])
        if dof not in wanted:
            continue
        target = log_interpolate(
            measured.abscissa, specification.abscissa,
            specification.ordinate[wanted[dof]])
        got = np.real(measured.ordinate[i])
        good = (np.isfinite(target) & np.isfinite(got)
                & (target > 0.0) & (got > 0.0))
        if not good.any():
            continue
        block = None if measured.block is None else measured.block[i]
        out.append((dof, block, signed_rms_db(got[good], target[good])))
    return out


#: the same three decibels either side that the SRS deviation and the
#: random level error use — one convention, three readings
SINE_ERROR_DB = 3.0


def signed_rms_db(got: np.ndarray, wanted: np.ndarray) -> float:
    """The RMS of the line-by-line deviation in dB, signed by the mean
    deviation: positive when over, negative when under.

    The SRS and the sine comparisons both use this one definition, so
    the two cannot come to differ.

    Parameters
    ----------
    got, wanted : ndarray
        Measured and target, positive and finite, line for line.

    Returns
    -------
    float
        The signed RMS deviation in dB.
    """
    deviation = 20.0 * np.log10(np.asarray(got) / np.asarray(wanted))
    magnitude = float(np.sqrt(np.mean(deviation ** 2)))
    return float(np.copysign(magnitude, deviation.mean()))


def sine_errors(specification, levels):
    """[(dof, tone, RMS dB deviation)] for extracted sine levels
    against the tones they were controlled to.

    The same value `srs_errors` gives a shock spectrum, for the same
    reasons: in dB so every part of the sweep counts alike, RMS so one
    noisy line does not fail a channel, and signed by the mean so an
    under-test reads negative. The target is the tone's own breakpoint
    interpolation (`SineTone.target`: linear in frequency on linear
    segments, linear in log frequency on log ones), read only at the
    frequencies the level covers. Lines where the tone was under the
    noise floor (`SineLevel.resolved` False) are not scored, and neither
    are zero readings. `levels` is a `SineLevelSet` or a sequence of
    `SineLevel`; a tone or channel the specification does not have is
    skipped.
    """
    levels = getattr(levels, 'levels', levels)
    out = []
    for level in levels:
        try:
            tone = specification.tone(level.tone)
        except KeyError:
            continue
        target = tone.target(level.abscissa)
        for row, dof in enumerate(level.response_dof):
            try:
                column = specification.response_dof.index(str(dof))
            except ValueError:
                continue
            wanted = target[:, column]
            got = np.abs(level.ordinate[row])
            # a reading the tone was under the noise at is reported at
            # the floor and flagged (2026-09-30): it says how much was
            # not seen, not how much was there, so it is not scored;
            # a zero from an older level is the same 'nothing detected'
            good = np.isfinite(wanted) & (wanted > 0) & (got > 0)
            resolved = getattr(level, 'resolved', None)
            if resolved is not None:
                good &= np.asarray(resolved)[row]
            if not good.any():
                continue
            out.append((str(dof), level.tone,
                        signed_rms_db(got[good], wanted[good])))
    return out
