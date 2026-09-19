"""How a measured response compares with what it was controlled to.

A specification and the PSD that answered it are the comparison a random
vibration test exists to make, and looking at the two curves says only
so much. What a report needs is numbers: how much energy was asked for,
how much arrived, and how much of the band strayed outside the bands the
controller was told to warn and abort on.

The two rarely share a frequency axis. A specification is written at a
handful of breakpoints, or on the controller's lines rather than the
analysis's, so it is interpolated onto the measurement's lines — in
log-log, because a specification is drawn that way and read that way,
and a straight line between two breakpoints on log axes is a power law.
Interpolated linearly instead, a decade-wide segment runs several dB
above the specification through the middle of its own span.

Only the measurement's own lines are counted, and only those inside the
specification's band. Outside it the specification says nothing, and a
response there is neither passing nor failing.

At the two ends of that band a line is usually half in and half out,
and neither answer is right: counted whole it credits the response with
power the specification never asked for, dropped it throws away power
that was asked for. So a line at an end is compared over the part of
its own bin that the specification covers — the same half-bin the plot
shades — which is also what makes an RMS error the comparison of two
areas over exactly the same stretch of frequency.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import DataArray, Specification


def log_interpolate(frequencies: ArrayLike, spec_frequencies: ArrayLike,
                    spec_values: ArrayLike) -> np.ndarray:
    """A specification's values at `frequencies`, interpolated log-log.

    NaN wherever the specification does not reach. Outside its band it
    says nothing, and extending the end segments would invent a
    requirement nobody wrote; a line written at zero — which is how a
    controller writes one outside its band — says nothing either.
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


def log_log_area(frequencies: ArrayLike, values: ArrayLike,
                 low: float | None = None,
                 high: float | None = None) -> float:
    """The area under the power law through these points, exactly.

    A specification's points are breakpoints of a continuous curve, and
    the curve between two of them is the straight line they make on log
    axes — which is a power law, W = C f**n. Its integral has a closed
    form, so there is nothing to approximate: no grid, no rule, and no
    dependence on how finely anything else happened to be measured.

    For a segment from (f1, W1) to (f2, W2), n is the slope in log-log
    and the area is

        W1 f1 ln(f2/f1)                      when n is -1
        W1 / f1**n * (f2**(n+1) - f1**(n+1)) / (n + 1)   otherwise

    `low` and `high` clip the band. A segment straddling an edge is cut
    there and its value at the cut taken from the same power law, so a
    comparison over part of a specification integrates exactly that
    part.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    values = np.asarray(np.real(values), dtype=float)
    usable = (np.isfinite(frequencies) & (frequencies > 0.0)
              & np.isfinite(values) & (values > 0.0))
    if usable.sum() < 2:
        return float('nan')
    f = frequencies[usable]
    w = values[usable]
    order = np.argsort(f)
    f, w = f[order], w[order]

    low = f[0] if low is None else max(float(low), f[0])
    high = f[-1] if high is None else min(float(high), f[-1])
    if not high > low:
        return 0.0

    total = 0.0
    for (f1, w1), (f2, w2) in zip(zip(f, w), zip(f[1:], w[1:])):
        if f2 <= low or f1 >= high or f2 <= f1:
            continue
        n = np.log(w2 / w1) / np.log(f2 / f1)
        a, b = max(f1, low), min(f2, high)
        # the same power law evaluated at wherever the band cuts it
        wa = w1 * (a / f1) ** n
        if abs(n + 1.0) < 1e-12:
            total += wa * a * np.log(b / a)
        else:
            # written as wa·a·((b/a)**(n+1) - 1)/(n+1) rather than
            # wa/a**n · (b**(n+1) - a**(n+1))/(n+1). The two are the
            # same expression, and only this one survives a steep
            # segment: (b/a)**(n+1) is exp((n+1)·ln(b/a)), whose
            # exponent works out to ln(w2/w1) + ln(f2/f1) and is
            # therefore bounded by the data, where a**n alone is not.
            #
            # It matters because a specification is no longer always a
            # handful of breakpoints spanning decades. The PSD of a
            # transient target is a Specification with four thousand
            # lines a quarter-hertz apart, and neighboring lines two
            # decades apart over a frequency ratio of 1.00025 give an
            # exponent near twenty thousand: a**n overflowed, the
            # subtraction went inf - inf, and every channel's RMS error
            # came back NaN.
            total += wa * a * ((b / a) ** (n + 1.0) - 1.0) / (n + 1.0)
    return float(total)


def written_band(frequencies: ArrayLike,
                 values: ArrayLike) -> tuple[float, float] | None:
    """(low, high) one written curve actually says something over."""
    frequencies = np.asarray(frequencies, dtype=float)
    values = np.asarray(np.real(values), dtype=float)
    usable = (np.isfinite(frequencies) & (frequencies > 0.0)
              & np.isfinite(values) & (values > 0.0))
    if usable.sum() < 2:
        return None
    return float(frequencies[usable].min()), float(frequencies[usable].max())


def band_of(specification: Specification,
            record: int = 0) -> tuple[float, float] | None:
    """(low, high) the specification actually says something over.

    A specification on bands says something over each band's whole
    width, so its range is its outer bin edges — not its first and last
    centers, which sit inside them and would shrink every comparison
    against it by half a band at each end.
    """
    frequencies = np.asarray(specification.abscissa, dtype=float)
    values = np.asarray(np.real(specification.ordinate[record]), dtype=float)
    band = written_band(frequencies, values)
    if band is None or getattr(specification, 'bandwidth', None) is None:
        return band
    left, right = specification.bin_bounds()
    usable = (np.isfinite(frequencies) & (frequencies > 0.0)
              & np.isfinite(values) & (values > 0.0))
    return float(left[usable].min()), float(right[usable].max())


def covered(lines: ArrayLike, low: float, high: float,
            widths: ArrayLike | None = None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each measured bin cut to the part of it a band covers.

    A line stands for its whole bin, half a width either side, and at
    the ends of a specification a bin hangs over the edge. Returns
    (start, stop, width, cut): where each bin lies inside the band, how
    wide that is, and which bins the edge went through.

    A bin wholly outside has zero width and is neither cut nor covered.
    """
    lines = np.asarray(lines, dtype=float)
    if lines.size < 2:
        empty = np.zeros(lines.shape)
        return empty, empty, empty, np.zeros(lines.shape, dtype=bool)
    # a banded measurement's bins are its own, geometric ones
    widths = np.gradient(lines) if widths is None else np.asarray(widths)
    start = np.maximum(lines - widths / 2.0, low)
    stop = np.minimum(lines + widths / 2.0, high)
    width = np.clip(stop - start, 0.0, None)
    return start, stop, width, (width > 0.0) & (width < widths * (1.0 - 1e-9))


def _written_area(frequencies, values, low, high, reading, widths=None):
    """The area under a written curve, read the way its owner is.

    A limit belongs to a specification and is written the way the
    specification is, so it is integrated the same way. Reaching
    straight for the log-log form would be right for one written at
    breakpoints and wrong for one whose owner is a density — the same
    assumption that had the PSD of a transient target drawn as a power
    law through four thousand of its own lines.
    """
    if reading != 'bin':
        return log_log_area(frequencies, values, low, high)
    from .octave import bin_bounds

    # a banded specification carries its own bins — geometric, from a
    # standard — and the midpoints between its centers are not them
    left, right = bin_bounds(np.asarray(frequencies, dtype=float), widths)
    if low is not None:
        left, right = np.maximum(left, low), np.maximum(right, low)
    if high is not None:
        left, right = np.minimum(left, high), np.minimum(right, high)
    width = np.maximum(right - left, 0.0)
    values = np.asarray(np.real(values), dtype=float)
    good = np.isfinite(values) & (values >= 0.0)
    if not good.any():
        return float('nan')
    return float(np.sum(values[good] * width[good]))


def outside(lines: ArrayLike, values: ArrayLike,
            spec_frequencies: ArrayLike, limit_values: ArrayLike,
            over: bool = True, reading: str = 'log_log',
            spec_widths: ArrayLike | None = None,
            widths: ArrayLike | None = None) -> np.ndarray:
    """Which measured lines fell outside one written limit curve.

    In the middle of the band this is the plain comparison: the density
    measured on a line against the density the limit asks for there.

    At the two ends it cannot be. A bin straddling the edge is covered
    by the specification for part of its width and by nothing for the
    rest, and reading the limit at the bin's center either invents a
    requirement past the edge or drops a bin that is mostly inside. So
    a cut bin is judged on the part that is covered: the power the
    measurement holds over that stretch, `G` times the covered width,
    against the power the limit asks for over the same stretch, which
    is the exact log-log area under it. Both sides then span one
    identical piece of frequency, which is the only way the comparison
    means anything at an edge.
    """
    lines = np.asarray(lines, dtype=float)
    values = np.asarray(np.real(values), dtype=float)
    limit = log_interpolate(lines, spec_frequencies, limit_values)
    real = np.isfinite(values)
    out = (np.isfinite(limit) & real
           & ((values > limit) if over else (values < limit)))

    band = written_band(spec_frequencies, limit_values)
    if band is None:
        return out
    start, stop, width, cut = covered(lines, *band, widths)
    for i in np.flatnonzero(cut & real):
        asked = _written_area(spec_frequencies, limit_values,
                              start[i], stop[i], reading, spec_widths)
        if not np.isfinite(asked):
            continue
        held = values[i] * width[i]
        out[i] = held > asked if over else held < asked
    return out


def specification_rms(specification: Specification, record: int = 0,
                      low: float | None = None,
                      high: float | None = None) -> float:
    """A specification's RMS, from its own points and nothing else.

    Not from whatever grid a measurement happened to be computed on:
    the level a specification asks for is a property of the
    specification, and it should not move in the fourth decimal because
    somebody changed a frame length.

    The integral is the object's, so it is taken the way the object is
    drawn. This used to reach straight for `log_log_area`, which is
    right for a specification written at breakpoints and wrong for one
    computed from a record — and the second kind exists: the PSD of a
    transient target is a `Specification` of four thousand density
    lines.
    """
    return float(np.sqrt(specification.area(record, low, high)))


def rms(frequencies: ArrayLike, values: ArrayLike) -> float:
    """The RMS a PSD carries over the lines given: sqrt of its total.

    Each line times the width of its own bin, summed. A discrete
    spectrum is a density per bin, so this is the total the lines
    actually hold — Parseval's, exactly — where a trapezoid halves the
    two end bins and reads a little under.

    The widths come from the whole axis before any line is dropped, so
    a line that says nothing contributes nothing rather than having its
    bin quietly widened onto its neighbors. Bridging a gap would be
    assuming what is in it.
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
    """(lower, upper) of one pair of limits, on `frequencies`.

    None for a limit the specification does not carry — nothing was
    exceeded there because nothing was asked.
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

#: how large a detected offset must be before it is believed. Runs are
#: commanded in steps of 3 dB and 6; a detected 1 or 2 dB is not a
#: commanded level but the ordinary spread of a test that held its
#: channels imperfectly — and dividing that out would erase exactly
#: the control error the comparison exists to show. Below this the
#: answer is zero, and the Scaling field is there for anyone who
#: really did command a 2 dB step.
DETECTION_DEADBAND_DB = 2


def significant_band(want: np.ndarray, good: np.ndarray) -> np.ndarray:
    """`good`, narrowed to where the specification has real content.

    Lines within `DETECTION_RANGE_DB` of this channel's own peak. The
    peak rather than a fixed level, because a specification's units
    and size are its own; per channel rather than across them, because
    channels are bounded at their own levels.
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


def detect_scale_db(specification: Specification, measured: DataArray,
                    spec_records: Sequence[int] | None = None,
                    measured_records: Sequence[int] | None = None) -> int:
    """The offset, on the 3 dB ladder from zero, that best lays the
    measurement on the specification — what a run captured at -6 dB
    needs added to be compared against the 0 dB requirement.

    Each common channel answers with the median dB difference across
    the lines it shares with the specification — the median because a
    resonance or a notch is exactly the kind of departure the
    comparison exists to show, and a mean would let it vote on the
    level — snapped to the nearest multiple of `SCALE_STEP_DB`, since
    that is the ladder runs are commanded on; a level error of a
    decibel or two reads as its nearest step, and the Scaling field is
    there to type the truth. (Whole decibels until 2026-09-19.)

    Across channels the answer is the **smallest offset at least two
    channels agree on**, not the median of all of them. A specification
    usually bounds monitors as well as controls, each monitor sitting
    its own distance under its envelope, and the median of that spread
    lands wherever the monitors happen to pile up — measured on a real
    36-channel run it said +8 dB for a section commanded at -6. The
    controls are the channels *on* their specification, every one at
    exactly the commanded offset, which makes the commanded level the
    smallest value with corroboration; requiring two keeps one broken
    channel from answering alone. No agreement anywhere, the median of
    the channel answers; no channels, zero.

    And one physical veto over whichever answer wins: **a commanded
    level is a floor**. In a genuine run-up every channel sits at or
    above the commanded offset — controls exactly on it, monitors above
    it — so a channel whose band-median lands well *below* the
    candidate falsifies the scaled-run premise outright — provided
    that channel is itself behaving (a channel sitting over its own
    envelope is a fault, and a fault vetoes nothing). The plate's
    demonstration run is the case that demanded it: two shakers cannot
    hold eight channels to the specification, the drive point ended
    dead on spec and the rest scattered 3-14 dB low, and two of them
    happening to agree at +4 read as a well-controlled -4 dB run. When
    the floor contradicts the candidate the answer is zero: scaling a
    comparison silently on doubtful evidence is worse than showing the
    mismatch, and the Scaling field is right there to be typed in.
    """
    per_channel = []
    for _label, spec_index, measured_index in matched_records(
            specification, measured, spec_records, measured_records):
        lines = np.asarray(measured.abscissa, dtype=float)
        got = np.asarray(np.real(measured.ordinate[measured_index]),
                         dtype=float)
        want = log_interpolate(lines, specification.abscissa,
                               specification.ordinate[spec_index])
        good = np.isfinite(want) & (want > 0.0) & np.isfinite(got) & (got > 0.0)
        # only where the requirement says something about level —
        # see `significant_band`
        good = significant_band(want, good)
        if good.any():
            median = float(np.median(10.0 * np.log10(want[good] / got[good])))
            per_channel.append(round(median / SCALE_STEP_DB) * SCALE_STEP_DB)
    if not per_channel:
        return 0
    agreed = [value for value in set(per_channel)
              if per_channel.count(value) >= 2]
    candidate = (min(agreed) if agreed
                 else round(float(np.median(per_channel))))
    # The floor veto listens only to channels behaving like channels:
    # an offset below -2 means the channel sits over its own envelope
    # on a band-median basis, which is a fault (or a wild resonance),
    # not a statement about the commanded level — the broken-channel
    # test pins that such a channel cannot answer alone, and it must
    # not veto alone either. 2 dB of slack both ways lets a slightly
    # hot control sit under the commanded level without canceling a
    # real run-up.
    plausible = [value for value in per_channel if value >= -2]
    if plausible and min(plausible) < candidate - 2:
        return 0
    # and a deadband: a level nobody commanded is not a level. The
    # smallest-agreed rule reads a control/monitor split, and a run
    # whose channels all sit within a couple of dB of their
    # specification has no such split to read — it is simply a test
    # that held. Answering the dB two channels happened to share
    # there (the drone transient's -1) would divide out the control
    # error rather than show it.
    if abs(candidate) <= DETECTION_DEADBAND_DB:
        return 0
    return candidate


def comparison_scale_db(specification: Specification, measured: DataArray,
                        spec_records: Sequence[int] | None = None,
                        measured_records: Sequence[int] | None = None) -> int:
    """The decibels every comparison adds to `measured`: the value the
    user holds on the object (`scale_db`, 0 included), or the detected
    one when nothing is held. The one resolver, so the drawn curves,
    the error metrics and the report cannot disagree."""
    held = getattr(measured, 'scale_db', None)
    if held is not None:
        return int(held)
    return detect_scale_db(specification, measured,
                           spec_records, measured_records)


def matched_records(specification: Specification, measured: DataArray,
                    spec_records: Sequence[int] | None = None,
                    measured_records: Sequence[int] | None = None
                    ) -> list[tuple[str, int, int]]:
    """[(label, spec record, measured record)] for the channels the two
    have in common, paired by DOF in the specification's own order —
    the pairing `compare_all` has always used, named so the scale
    detection walks exactly the channels the comparison will."""
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
    """(over, under): which lines went outside a pair of limits, and how.

    What the plot marks, so a line out of tolerance is found by looking
    rather than by reading a percentage and hunting for it. The lines
    judged are the *scaled* measurement — the same curve the comparison
    draws — resolved through `comparison_scale_db` unless the caller
    already did.
    """
    if scale_db is None:
        scale_db = comparison_scale_db(specification, measured)
    lines = np.asarray(measured.abscissa, dtype=float)
    got = (np.asarray(np.real(measured.ordinate[measured_record]),
                      dtype=float) * 10.0 ** (scale_db / 10.0))
    marks = []
    for edge, over in ((f'{pair}_upper', True), (f'{pair}_lower', False)):
        written = specification.limits.get(edge)
        marks.append(np.zeros(lines.shape, dtype=bool) if written is None
                     else outside(lines, got, specification.abscissa,
                                  written[spec_record], over,
                                  getattr(specification, 'interpolation',
                                          'log_log'),
                                  getattr(specification, 'bandwidth', None),
                                  getattr(measured, 'bandwidth', None)))
    return marks[0], marks[1]


def compare(specification: Specification, measured: DataArray,
            spec_record: int = 0,
            measured_record: int = 0,
            scale_db: float | None = None) -> dict[str, Any]:
    """How one measured record answers one record of a specification.

    The two RMS levels and the difference between them as a percentage,
    how many lines were compared and over what band, and for each pair
    of limits how many of those lines fell outside it. A limit the
    specification does not carry is absent rather than zero.

    Every number is of the **scaled** measurement — `scale_db` resolved
    through `comparison_scale_db` unless the caller already did — and
    `scale_db` is echoed in the result so a table can say what was
    compared.
    """
    if scale_db is None:
        scale_db = comparison_scale_db(specification, measured)
    lines = np.asarray(measured.abscissa, dtype=float)
    got = (np.asarray(np.real(measured.ordinate[measured_record]),
                      dtype=float) * 10.0 ** (scale_db / 10.0))
    written = band_of(specification, spec_record)
    if written is None or lines.size < 2:
        return {'lines': 0}
    # the bins the spectrum actually has. An octave-band one carries
    # its own — geometric, from a standard — and reading them off the
    # centers would be a hair out at every band and wrong at the ends.
    own = getattr(measured, 'bin_widths', None)

    # A measured line stands for its whole bin, half a width either
    # side. The band compared is where those bins and the specification
    # overlap, and a bin straddling the edge is counted for the part of
    # it that is inside — so the two are summed and integrated over one
    # identical stretch, and a response sitting exactly on its
    # specification comes out at nothing rather than a quarter of a
    # percent over.
    widths = np.gradient(lines) if own is None else np.asarray(own())
    left, right = lines - widths / 2.0, lines + widths / 2.0
    low = max(written[0], float(left.min()))
    high = min(written[1], float(right.max()))
    if not high > low:
        return {'lines': 0}
    overlap = np.clip(np.minimum(right, high) - np.maximum(left, low),
                      0.0, None)
    inside = np.isfinite(got) & (got >= 0.0) & (overlap > 0.0)
    if inside.sum() < 1:
        return {'lines': 0}

    response = got[inside]
    # the specification from its own points, exactly; the measurement
    # from its bins, which is what a discrete spectrum holds. Each
    # integrated as its own points mean, over the one band.
    spec_rms = specification_rms(specification, spec_record, low, high)
    got_rms = float(np.sqrt(np.sum(response * overlap[inside])))
    out = {
        'lines': int(inside.sum()),
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
    for pair in ('warning', 'abort'):
        if not any(f'{pair}_{edge}' in specification.limits
                   for edge in ('lower', 'upper')):
            continue
        over, under = exceedances(specification, measured, spec_record,
                                  measured_record, pair, scale_db=scale_db)
        beyond = int(((over | under) & inside).sum())
        out[f'{pair}_lines'] = beyond
        out[f'{pair}_percent'] = 100.0 * beyond / out['lines']
    return out


#: how far the RMS may be out before a channel is called out, in dB.
#: Three is the tolerance a random vibration specification is usually
#: written with, and it is a starting point rather than a rule — the
#: bar charts let it be dragged.
ERROR_DB = 3.0

#: and how much of a channel's band may sit outside abort before the
#: same is said of it, as a percentage of the lines compared
LINES_PERCENT = 10.0


def channel_errors(rows: Sequence[tuple[str, dict[str, Any]]]
                   ) -> list[tuple[str, float, float]]:
    """[(label, dB out, percent of lines outside abort)] for a bar chart.

    A row per control channel, in the specification's own order, from
    whatever `compare_all` returned. Channels with nothing to compare
    are left out rather than drawn at zero, which would read as a
    channel that matched.
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
    """What share of these fell outside the threshold, as a percentage.

    One-sided when `high` is None — the lines-out chart has a ceiling
    and no floor, because no amount of *staying inside* the abort
    limits is a fault.
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
    """[(label, result)] for every channel the two have in common.

    A specification bounds control channels, and a measurement holds
    those and usually many more. Pairing is by DOF — the response, and
    the reference where there is one — so a cross spectrum is compared
    against the cross term of the specification if it carries one, and
    against nothing if it does not.

    In the specification's own order, which is the order a control room
    reads its channels in. The scale is resolved once for the whole
    set — one measurement gets one scaling, never a different number
    per channel — and a caller comparing a *derived* form of the data
    (the report's own octave banding) passes the scale it resolved on
    the original, so the two gridings cannot round to different
    decibels.
    """
    if scale_db is None:
        scale_db = comparison_scale_db(specification, measured,
                                       spec_records, measured_records)
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
    """[(label, event, RMS dB deviation)] for a shock spectrum against
    the one it was required to meet.

    In **decibels**, and root-mean-square across the band. An SRS spans
    decades and is read on log axes, so the obvious distance — the one
    a time waveform gets, `||m - s|| / ||s||` — is dominated by whatever
    bands happen to be loudest and goes blind to a large ratio error
    anywhere the target is small. Measured on the airplane run that
    reading passed every channel at 20% while one of them sat 25 dB
    over: 99.5% of its norm came from the top two decades. In dB every
    band counts the same, which is how a shock is judged and why the
    threshold beside it is the same three decibels a level error gets.

    RMS rather than the worst band, because the worst band is one point
    and one point can be noise; a curve that is 4 dB out everywhere and
    a curve with a single 4 dB spike are not the same result and should
    not read as one. The worst is still visible on the plot.

    **Signed**: the magnitude is the RMS, the sign is which side of the
    requirement the spectrum predominantly sits — the sign of the mean
    dB deviation. An unsigned RMS read an under-test as an over-test:
    a spectrum 5 dB low everywhere reported +5, the bar chart shaded
    it past the ceiling, and the reader concluded the shock was too
    hard when the machine had under-hit (Brandon, on the drone stress
    set, 2026-08-20). A curve genuinely astride the requirement — big
    RMS, mean near zero — takes whichever side its mean leans, which
    is honest: the magnitude is what convicts it, and either bound is
    the same three decibels away.

    The specification is interpolated onto the measurement's own
    frequencies, log-log, the way a written specification is — and only
    where it reaches. Outside its band it says nothing.
    """
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
    """The RMS of the line-by-line deviation in decibels, signed by
    which way the mean falls: over is positive, under is negative.
    The one metric the SRS and the sine readings share — a definition
    written once, so the two cannot drift apart.

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

    The same reading `srs_errors` gives a shock spectrum, for the same
    reasons: decibels so every part of the sweep counts alike, RMS so
    one noisy line does not convict a channel, **signed** by the mean
    so an under-test reads under. The target is the tone's own
    breakpoint interpolation (`SineTone.target` — linear in f on
    linear segments, log-f on log ones), evaluated only where the
    level actually swept: coverage is the comparison's, not assumed.
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
            # the debiased magnitude clamps at zero where the noise
            # accounted for everything measured; a zero is 'nothing
            # detected there', not minus infinity decibels
            good = np.isfinite(wanted) & (wanted > 0) & (got > 0)
            if not good.any():
                continue
            out.append((str(dof), level.tone,
                        signed_rms_db(got[good], wanted[good])))
    return out
