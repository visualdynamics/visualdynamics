"""A replicated record judged segment by segment, each by the rule for
its kind of motion.

`replication.compare` reads a replication the way a controller's screen
does: one waveform error over the whole record, one worst deviation of
a maximax spectrum, one level. Over a long record those figures are
dominated by whatever carries most of its mean square, and a short
event carries very little of it. A shock a fraction of a second long
in a minute of road can be missed by several decibels across part of
its spectrum while the whole-record error still reads under five
percent, because the shock holds a percent or two of the record's
mean square. The whole-record spectrum cannot see it either: a
maximax over the record keeps each oscillator's largest response
anywhere in it, and that may come from a different event.

So the record is cut into segments, each marked as one of three kinds
of motion, and each is held to the tolerance of its kind, the way test
documents commonly write a replication's tolerance:

- **stationary**: a stretch whose statistics hold steady (a rough
  road at constant speed). Judged in frequency by its ASD against the
  reference's, line by line within a band (±3 dB by default), and in
  time by its mean square about zero against the reference's (within
  +10 percent by default; an undertest shows in the ASD band).
- **transient**: a long non-stationary stretch whose spectrum holds
  its shape while its level changes (braking, a run-up). Judged by
  its short-time rms envelope, window by window, against the
  reference's (within 1 dB by default), and by its normalized ASD, the
  spectrum with the segment's own mean square divided out, within a
  band (±3 dB by default).
- **shock**: a short event. Judged by its shock response spectrum in
  each polarity against the reference's, point by point (−6/+3 dB by
  default), and by its major peaks, the reference's largest crests and
  troughs, each read against the control at the same sample (within
  20 percent by default).

The tolerances are examples of what test documents ask for, not any
one document's requirement; every one is an argument.

A stationary or transient segment is weighted by a tapered cosine
window before its ASD and its mean square are computed (10 percent of
it by default, half at each end), so the segment's abrupt ends do not
enter its spectrum as content the record does not have. A shock's SRS
is computed on the segment as recorded: the SRS is a peak response,
not an average, and a mark drawn tight on the event would put the
event's first and largest cycles under the taper, weighting down the
reference and the control alike there and hiding a miss in the
ratio. The two local readings, the envelope and the peaks, are read
on the segment as recorded too: they are ratios of the two traces at
one place, from which a weight applied to both cancels.

A segment is judged only on samples that are numbers. A trace holding
a NaN or an infinity is refused, and a control that is zero where the
reference asks for something reads minus infinity decibels there and
fails, so a dead channel never reads as a pass.

The comparison assumes the two traces are on like terms. The reference
and the control should be band-limited alike to the test band
(`core.filters`) before they are judged; `replication_segments` aligns
them by the one integer shift `replication.lag_of` finds, and
`judge_segments` takes arrays already aligned. The whole-record rms
error is reported beside the segments, so the reading that hides a
segment's miss is on the page next to the one that shows it.

References
----------
The segments and the forms of their tolerances follow the first of
these; the readings are standard estimators, set out in the others.
Each shock segment's spectra come from `core.srs`, which cites its
filter.

1. Department of Defense (2019). *MIL-STD-810H, Environmental
   Engineering Considerations and Laboratory Tests*. Method 525.2,
   paragraph 4.2.2c: a replication judged segment by segment, each
   stretch stationary, a short transient or a long transient, a
   stationary or long one tapered at its ends by a 10 percent cosine
   window, and each held to tolerances of the forms used here.
2. Welch, P. D. (1967). "The Use of Fast Fourier Transform for the
   Estimation of Power Spectra: A Method Based on Time Averaging Over
   Short, Modified Periodograms." *IEEE Transactions on Audio and
   Electroacoustics*, 15(2), 70-73.
   [doi:10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901)
   The averaged, windowed periodogram each ASD here is, the estimator
   `TimeHistory.compute_psds` computes.
3. Harris, F. J. (1978). "On the Use of Windows for Harmonic Analysis
   with the Discrete Fourier Transform." *Proceedings of the IEEE*,
   66(1), 51-83.
   [doi:10.1109/PROC.1978.10837](https://doi.org/10.1109/PROC.1978.10837)
   The tapered cosine window a stationary or transient segment is
   weighted by.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import DataArray, TimeHistory

#: the kinds of motion a segment can be marked as, each with its rule
SEGMENT_KINDS = ('stationary', 'transient', 'shock')

#: the share of each segment, in percent, under the tapered cosine
#: window's two ends together: half of it rises at the start, half
#: falls at the end, and the middle is weighted by one
TAPER_PERCENT = 10.0

#: a stationary segment's ASD and a transient segment's normalized ASD
#: against the reference's, in dB below and above, at every line judged
ASD_DB = (-3.0, 3.0)

#: a stationary segment's mean square against the reference's, in
#: percent below and above. Test documents commonly write only the
#: upper edge (+10 percent); a variance that falls short is an
#: undertest, and the ASD band already holds it from below
VARIANCE_PERCENT = (-np.inf, 10.0)

#: a transient segment's short-time rms against the reference's, in dB
#: below and above, in every window
ENVELOPE_DB = (-1.0, 1.0)

#: the envelope's window, and its step, as a share of the whole record
#: in percent, when no window is given. Test documents commonly ask
#: that the window be no longer than one percent of the record, which
#: keeps it short against the transient's own changes of level
ENVELOPE_PERCENT = 1.0

#: a shock segment's SRS against the reference's, in dB below and
#: above, at every natural frequency judged, each polarity on its own
SRS_DB = (-6.0, 3.0)

#: how far each major peak of the control may be from the reference's,
#: in percent of the reference's
PEAK_PERCENT = 20.0

#: how many of the reference's largest peaks are read each way (crests
#: and troughs)
PEAKS = 2

#: the largest share, in percent, of all the reference's peaks in the
#: segment that may be counted as major. A segment with few peaks reads
#: fewer than `PEAKS` each way, never fewer than one
MAJOR_SHARE_PERCENT = 10.0

#: the ASD's line spacing in Hz, when none is given: frames one second
#: long, under a Hann window at half overlap
LINES_HZ = 1.0

#: how far below its own highest line, in dB of power, the reference's
#: ASD may fall and still be judged. A ratio against a line that holds
#: nothing reports nothing: past the test band the reference is the
#: filter's skirt, and the control's leakage there is not a miss
ASD_FLOOR_DB = 40.0

#: how far past a tolerance's edge a reading still counts as on it
EDGE = 1e-9


class Segment(NamedTuple):
    """One stretch of the record and the kind of motion it is.

    `start` and `end` are in seconds on the reference's own clock, from
    its first sample. `kind` is one of `SEGMENT_KINDS`. `name` is what
    the readout calls it; when empty it is called by its kind and its
    place among the segments of that kind ('shock 1').
    """

    start: float
    end: float
    kind: str
    name: str = ''


@dataclass(frozen=True, eq=False)
class SegmentCheck:
    """One reading of one segment against its tolerance.

    `reading` names it ('ASD', 'variance', 'envelope', 'normalized ASD',
    'SRS positive', 'SRS negative', 'major peaks'). `where` and `values`
    are each judged item and its reading: lines in Hz for the spectra,
    window starts in seconds for the envelope, peak times in seconds
    for the peaks (both on the reference's clock), and one item for the
    variance; `values` are in `unit`, 'dB' or '%', as distances from
    the reference. `low` and `high` are the lowest and highest reading,
    `tolerance` the band it was held to, `outside` how many items fell
    outside it. `passed` is None when nothing could be judged.
    """

    reading: str
    unit: str
    tolerance: tuple[float, float]
    where: np.ndarray
    values: np.ndarray
    outside: int
    passed: bool | None
    #: how many items had to lie inside for a pass, when that is not
    #: every one of them (the ASD's share); None means every one
    needed: int | None = None

    @property
    def judged(self) -> int:
        """How many items were read."""
        return int(np.size(self.values))

    @property
    def low(self) -> float:
        """The lowest reading, NaN when there is none."""
        return float(np.min(self.values)) if self.judged else float('nan')

    @property
    def high(self) -> float:
        """The highest reading, NaN when there is none."""
        return float(np.max(self.values)) if self.judged else float('nan')

    def describe(self) -> str:
        """The check in a line: the reading, its range, the tolerance,
        and the verdict, as in 'SRS negative −7.4 to +0.1 dB (−6/+3 dB),
        4 of 65 outside: fails'."""
        if not self.judged:
            return f'{self.reading}: nothing to judge'
        span = (f'{_signed(self.low, self.unit)}' if self.judged == 1
                else f'{_signed(self.low, self.unit)} to '
                     f'{_signed(self.high, self.unit)}')
        text = f'{self.reading} {span} ({band_text(self.tolerance, self.unit)})'
        if self.judged > 1:
            text += f', {self.outside} of {self.judged} outside'
        verdict = {True: 'passes', False: 'fails', None: 'not judged'}[
            self.passed]
        return f'{text}: {verdict}'


@dataclass(frozen=True, eq=False)
class SegmentResult:
    """One segment judged: where it is, what share of the record it
    holds, and each of its checks.

    `first` and `samples` place it in the aligned traces. `share` is its
    share of the reference's mean square over the whole record, and
    `rms_error` its own rms error, both in percent and both read on the
    segment as recorded: together they say how much the segment can
    move the whole-record figure. `passed` is False when any check
    fails, None when none fails and one or more could not be judged,
    and True only when every check passes.
    """

    segment: Segment
    label: str
    first: int
    samples: int
    share: float
    rms_error: float
    checks: tuple[SegmentCheck, ...]
    passed: bool | None

    def describe(self) -> str:
        """The segment in a line: its name, kind, span and verdict, and
        which checks failed."""
        seg = self.segment
        verdict = {True: 'passes', False: 'fails', None: 'not judged'}[
            self.passed]
        text = (f'{self.label} ({seg.kind}, {seg.start:g} to {seg.end:g} s): '
                f'{verdict}')
        failed = [check.reading for check in self.checks
                  if check.passed is False]
        if failed:
            text += ' on ' + ', '.join(failed)
        return text


@dataclass(frozen=True, eq=False)
class SegmentedReplication:
    """A replicated record judged segment by segment.

    `segments` are the results in the order the marks were given.
    `rms_error` is the whole record's rms error in percent, the figure a
    controller reports, kept beside the segments because it is the one
    that hides a segment's miss. `passed` is False when any segment
    fails, None when none fails and one or more could not be judged (or
    none is marked), and True only when every segment passes. `channel` and `lag` say which record was
    read and how far it was shifted to align it (None and 0 for arrays
    given directly). `settings` are the tolerances it was judged under,
    as `judge_segments` took them.
    """

    segments: tuple[SegmentResult, ...]
    rms_error: float
    sample_rate: float
    samples: int
    passed: bool | None
    settings: dict[str, Any] = field(default_factory=dict)
    channel: str | None = None
    lag: int = 0

    @property
    def failed(self) -> list[SegmentResult]:
        """The segments that failed, in order."""
        return [result for result in self.segments if result.passed is False]

    @property
    def not_judged(self) -> list[SegmentResult]:
        """The segments that could not be judged, in order."""
        return [result for result in self.segments if result.passed is None]

    def describe(self) -> str:
        """The record in a line: the whole-record rms error, and the
        verdict by segment. The counts are of the segments judged; a
        segment that could not be judged is named on its own."""
        head = f'whole-record rms error {self.rms_error:.2f} %'
        if not self.segments:
            return f'{head}; no segment marked'
        failed, unjudged = self.failed, self.not_judged
        judged = len(self.segments) - len(unjudged)
        if failed:
            names = ', '.join(result.label for result in failed)
            text = (f'{head}; {len(failed)} of {judged} segments fail '
                    f'({names})')
        elif not judged:
            text = f'{head}; not judged'
        elif not unjudged:
            return f'{head}; all {judged} segments pass'
        else:
            text = f'{head}; {judged} of {len(self.segments)} segments pass'
        if unjudged and judged:
            names = ', '.join(result.label for result in unjudged)
            text += f', {len(unjudged)} not judged ({names})'
        return text


def _signed(value: float, unit: str) -> str:
    """A reading with its sign and unit, minus as the minus sign."""
    return f'{value:+.2f} {unit}'.replace('-', '\u2212')


def band_text(tolerance: tuple[float, float], unit: str) -> str:
    """A tolerance in words: '±3 dB', '−6/+3 dB', 'up to +10 %'."""
    below, above = tolerance
    if np.isinf(below) and below < 0:
        text = f'up to {above:+g} {unit}'
    elif np.isinf(above):
        text = f'from {below:+g} {unit}'
    elif np.isclose(-below, above):
        text = f'\u00b1{above:g} {unit}'
    else:
        text = f'{below:+g}/{above:+g} {unit}'
    return text.replace('-', '\u2212')


def _band(name: str, band: Any) -> tuple[float, float]:
    """(below, above), checked: two numbers, the first under the second."""
    try:
        below, above = (float(v) for v in band)
    except (TypeError, ValueError):
        raise ValueError(f'the {name} tolerance is (below, above), '
                         f'not {band!r}') from None
    if not below < above:
        raise ValueError(f'the {name} tolerance runs from {below:g} to '
                         f'{above:g}, which is not a band')
    return below, above


def tapered_cosine(samples: int, percent: float = TAPER_PERCENT
                   ) -> np.ndarray:
    """The weights of a tapered cosine (Tukey) window over `samples`
    samples.

    `percent` of the window is under the cosine, half rising from zero
    at the start and half falling to zero at the end; the middle is
    weighted by one. 0 is a rectangle and 100 a Hann window. This is
    the symmetric form, the one a segment cut from a record is weighted
    by, where `averaging.window_shape` gives the periodic form spectral
    frames use.

    Raises
    ------
    ValueError
        For fewer than one sample, or a share outside 0 to 100.
    """
    from scipy.signal import windows

    samples = int(samples)
    if samples < 1:
        raise ValueError('a window needs at least one sample')
    share = float(percent) / 100.0
    if not 0.0 <= share <= 1.0:
        raise ValueError(f'{percent:g} percent is not a share of a segment')
    # scipy's, not a second one written here (2026-10-16 review)
    return windows.tukey(samples, share, sym=True)


def _asd(x: np.ndarray, sample_rate: float, frame: int
         ) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies, ASD) of `x`: `TimeHistory.compute_psds` over Hann
    frames of `frame` samples at half overlap, as many as fit.

    Each trace is made a one-channel history of its own, because the
    reference and the control are records of one channel, which one
    history holding both would pool into one average.
    """
    from ..units import UNKNOWN
    from .averaging import Averaging
    from .data import TimeHistory

    # 2026-10-16 review: this was a second Welch estimator beside
    # compute_psds, held to it only by a test; now it is compute_psds
    rate = float(sample_rate)
    shape = Averaging(frame, 0.5, 'hann')
    averaging = replace(shape, frames=max(shape.most_frames(x.size, rate),
                                          1))
    history = TimeHistory(np.arange(x.size) / rate, x[None],
                          response_dof=['1X+'], ordinate_dim=[UNKNOWN])
    psd = history.compute_psds(averaging)
    return (np.asarray(psd.abscissa, dtype=float),
            np.asarray(psd.ordinate[0].real, dtype=float))


def _judged_lines(frequencies: np.ndarray, reference: np.ndarray,
                  band: tuple[float, float] | None) -> np.ndarray:
    """The lines a spectral check reads: inside the test band (above DC
    when none is given) and within `ASD_FLOOR_DB` of the reference's
    highest line there."""
    keep = frequencies > 0.0
    if band is not None:
        keep &= (frequencies >= band[0]) & (frequencies <= band[1])
    keep &= np.isfinite(reference) & (reference > 0.0)
    if keep.any():
        keep &= reference >= reference[keep].max() * 10.0 ** (-ASD_FLOOR_DB / 10.0)
    return keep


def _check(reading: str, unit: str, tolerance: tuple[float, float],
           where: np.ndarray, values: np.ndarray,
           percent: float = 100.0) -> SegmentCheck:
    """A check from its items: how many fall outside the tolerance, and
    whether at least `percent` of them are inside."""
    where = np.asarray(where, dtype=float)
    values = np.asarray(values, dtype=float)
    low, high = tolerance
    outside = (values < low - EDGE) | (values > high + EDGE)
    count = int(np.sum(outside))
    if not values.size:
        return SegmentCheck(reading, unit, tolerance, where, values, 0, None)
    inside = values.size - count
    if percent >= 100.0:
        return SegmentCheck(reading, unit, tolerance, where, values, count,
                            count == 0)
    # counted, not divided, as the SRS points rule counts
    needed = int(np.ceil(percent * values.size / 100.0 - 1e-9))
    return SegmentCheck(reading, unit, tolerance, where, values, count,
                        inside >= needed, needed)


def _spectral(reading: str, reference: np.ndarray, control: np.ndarray,
              sample_rate: float, frame: int, band, tolerance, percent,
              normalized: bool) -> SegmentCheck:
    """The ASD (or, `normalized`, the ASD with each trace's own mean
    square divided out) of the control against the reference's."""
    frequencies, want = _asd(reference, sample_rate, frame)
    _f, got = _asd(control, sample_rate, frame)
    if normalized:
        df = frequencies[1] - frequencies[0]
        want = want / (np.sum(want) * df)
        # a control with no power has no shape to compare; it stays
        # zero, and reads minus infinity at every line
        total = np.sum(got) * df
        got = got / total if total > 0.0 else got
    # judged where the reference asks for something, whatever the
    # control holds there: a dead control is minus infinity and fails
    # (2026-10-16 review: filtering on the control too let an all-zero
    # control pass on no lines)
    keep = _judged_lines(frequencies, want, band)
    return _check(reading, 'dB', tolerance, frequencies[keep],
                  _db(got[keep], want[keep], 10.0), percent)


def _db(got: np.ndarray, want: np.ndarray, factor: float) -> np.ndarray:
    """`factor`·log10(got/want), minus infinity where `got` is zero."""
    with np.errstate(divide='ignore'):
        return factor * np.log10(np.asarray(got, dtype=float)
                                 / np.asarray(want, dtype=float))


def _envelope(reference: np.ndarray, control: np.ndarray, window: int,
              first: int, sample_rate: float, tolerance) -> SegmentCheck:
    """The control's short-time rms over the reference's, in dB, in
    whole windows stepped by their own length from the segment's start.
    A tail shorter than a window is not read."""
    count = reference.size // window
    starts = np.arange(count) * window
    db = []
    kept = []
    for s in starts:
        want = float(np.mean(reference[s:s + window] ** 2))
        got = float(np.mean(control[s:s + window] ** 2))
        if want > 0.0:
            db.append(float(_db(got, want, 10.0)))
            kept.append(s)
    return _check('envelope', 'dB', tolerance,
                  (first + np.asarray(kept, dtype=float)) / sample_rate, db)


def _srs_checks(reference: np.ndarray, control: np.ndarray,
                sample_rate: float, band, tolerance, q: float,
                per_octave: int) -> list[SegmentCheck]:
    """Each polarity's shock response spectrum against the reference's,
    point by point."""
    from .replication import SRS_FLOOR_DB
    from .srs import octave_frequencies, peaks, spectra

    low = sample_rate / reference.size if band is None else band[0]
    high = sample_rate / 5.0 if band is None else min(band[1],
                                                      sample_rate / 5.0)
    if not high > low:
        return [SegmentCheck(f'SRS {kind}', 'dB', tolerance, np.empty(0),
                             np.empty(0), 0, None)
                for kind in ('positive', 'negative')]
    frequencies = octave_frequencies(low, high, per_octave)
    wanted = spectra(*peaks(reference, frequencies, sample_rate, q=q))
    gotten = spectra(*peaks(control, frequencies, sample_rate, q=q))
    out = []
    for kind in ('positive', 'negative'):
        want, got = wanted[kind], gotten[kind]
        # where the reference asks for something, whatever the control
        # holds: a control with no response there fails
        keep = np.isfinite(want) & (want > 0.0)
        if keep.any():
            # only where the reference asks for something, as the
            # whole-record comparison judges (`replication._srs_db`)
            keep &= want >= want[keep].max() * 10.0 ** (-SRS_FLOOR_DB / 20.0)
        got = np.clip(got, 0.0, None)
        out.append(_check(f'SRS {kind}', 'dB', tolerance,
                          frequencies[keep], _db(got[keep], want[keep], 20.0)))
    return out


def major_peaks(reference: ArrayLike, each: int = PEAKS,
                share_percent: float = MAJOR_SHARE_PERCENT) -> np.ndarray:
    """The samples of the reference's major peaks: its largest crests
    (local maxima above zero) and deepest troughs (local minima below
    zero), up to `each` of each, largest first, crests then troughs.

    No more than `share_percent` of all its peaks are counted as major,
    and never fewer than one each way where there is one, so a segment
    with few peaks is read on its very largest.
    """
    x = np.asarray(reference, dtype=float)
    if x.size < 3:
        return np.empty(0, dtype=int)
    middle = x[1:-1]
    crests = np.flatnonzero((middle > x[:-2]) & (middle >= x[2:])
                            & (middle > 0.0)) + 1
    troughs = np.flatnonzero((middle < x[:-2]) & (middle <= x[2:])
                             & (middle < 0.0)) + 1
    allowed = int(share_percent / 100.0 * (crests.size + troughs.size) / 2.0)
    take = min(int(each), max(allowed, 1))
    top = crests[np.argsort(x[crests])[::-1][:take]]
    bottom = troughs[np.argsort(x[troughs])[:take]]
    return np.concatenate([top, bottom]).astype(int)


def _peak_check(reference: np.ndarray, control: np.ndarray, first: int,
                sample_rate: float, each: int,
                percent: float) -> SegmentCheck:
    """The control at each of the reference's major peaks, as a
    distance from the peak in percent of it."""
    at = major_peaks(reference, each)
    deviation = 100.0 * (control[at] / reference[at] - 1.0)
    return _check('major peaks', '%', (-percent, percent),
                  (first + at) / sample_rate, deviation)


def _label(segments: Sequence[Segment]) -> list[str]:
    """What each segment is called: its name, or its kind and its place
    among the segments of that kind."""
    seen: dict[str, int] = {}
    labels = []
    for segment in segments:
        seen[segment.kind] = seen.get(segment.kind, 0) + 1
        labels.append(segment.name or f'{segment.kind} {seen[segment.kind]}')
    return labels


def as_segments(marks: Sequence[Any]) -> list[Segment]:
    """Segment marks as `Segment`s: each a `Segment`, or a tuple of
    (start, end, kind) with an optional name.

    Raises
    ------
    ValueError
        For a kind that is not one of `SEGMENT_KINDS`, or a segment
        that does not end after it starts.
    """
    out = []
    for mark in marks:
        segment = Segment(float(mark[0]), float(mark[1]), str(mark[2]),
                          str(mark[3]) if len(mark) > 3 else '')
        if segment.kind not in SEGMENT_KINDS:
            raise ValueError(f'{segment.kind!r} is not a kind of segment; '
                             'the kinds are ' + ', '.join(SEGMENT_KINDS))
        if not segment.end > segment.start:
            raise ValueError(f'a segment from {segment.start:g} to '
                             f'{segment.end:g} s does not end after it '
                             'starts')
        out.append(segment)
    return out


def judge_segments(reference: ArrayLike, control: ArrayLike,
                   sample_rate: float, segments: Sequence[Any], *,
                   taper_percent: float = TAPER_PERCENT,
                   band: tuple[float, float] | None = None,
                   lines: float = LINES_HZ,
                   asd_db: tuple[float, float] = ASD_DB,
                   asd_percent: float = 100.0,
                   variance_percent: tuple[float, float] = VARIANCE_PERCENT,
                   envelope_db: tuple[float, float] = ENVELOPE_DB,
                   window: float | None = None,
                   srs_db: tuple[float, float] = SRS_DB,
                   q: float | None = None,
                   per_octave: int | None = None,
                   peak_percent: float = PEAK_PERCENT,
                   peaks: int = PEAKS) -> SegmentedReplication:
    """A replicated record judged segment by segment, for a reference
    and a control trace held as plain arrays, aligned: the array form
    of `replication_segments`.

        result = judge_segments(reference, control, 2048.0, [
            (6.0, 30.0, 'stationary', 'rough road'),
            (31.9, 32.7, 'shock', 'pothole'),
            (41.0, 51.0, 'transient', 'braking')], band=(5.0, 400.0))
        result.rms_error, result.passed, result.failed

    Each segment is held to the rule for its kind (see the module):
    a stationary one by its ASD and its mean square, a transient one by
    its short-time rms envelope and its normalized ASD, a shock by its
    SRS in each polarity and its major peaks. Every tolerance is an
    example of what test documents ask for and is an argument here.

    Parameters
    ----------
    reference, control : array-like
        The two traces, one value per sample, aligned, in the same
        units. Ratios are taken, so the units do not matter.
    sample_rate : float
        Samples per second.
    segments : sequence
        The marks, each a `Segment` or (start, end, kind[, name]),
        start and end in seconds from the first sample.
    taper_percent : float, default `TAPER_PERCENT`
        The share of a stationary or transient segment under its
        tapered cosine window's ends, before its ASD and mean square
        are computed. A shock's SRS is read untapered.
    band : (float, float), optional
        The test band in Hz: the lines an ASD is judged at and the
        natural frequencies an SRS is computed at. By default an ASD is
        judged at every line above DC and an SRS from one cycle in the
        segment up to a fifth of the sample rate. Either way an SRS
        stops at a fifth of the sample rate, and a line or a point more
        than `ASD_FLOOR_DB` (or `replication.SRS_FLOOR_DB`) below the
        reference's highest is not judged.
    lines : float, default `LINES_HZ`
        The ASD's line spacing in Hz; a segment shorter than one frame
        is one frame.
    asd_db, asd_percent
        The ASD band in dB, for a stationary segment's ASD and a
        transient segment's normalized ASD, and the share of the judged
        lines that must lie inside it (every one by default).
    variance_percent : (float, float), default `VARIANCE_PERCENT`
        A stationary segment's mean square against the reference's, in
        percent below and above.
    envelope_db : (float, float), default `ENVELOPE_DB`
        A transient segment's short-time rms against the reference's in
        every window, in dB.
    window : float, optional
        The envelope's window and step in seconds;
        `ENVELOPE_PERCENT` of the whole record by default.
    srs_db : (float, float), default `SRS_DB`
        A shock segment's SRS against the reference's, each polarity,
        at every natural frequency judged.
    q, per_octave
        The SRS's quality factor and spacing, `srs.DEFAULT_Q` and
        `srs.PER_OCTAVE` by default.
    peak_percent, peaks
        How far each major peak may be from the reference's, in
        percent, and how many are read each way (`major_peaks`).

    Returns
    -------
    SegmentedReplication

    Raises
    ------
    ValueError
        For traces of different lengths or holding samples that are not
        numbers, a segment outside the record or too short to judge, an
        unknown kind, or a tolerance that is not a band.
    """
    from .srs import DEFAULT_Q, PER_OCTAVE

    r = np.asarray(np.real(reference), dtype=float).ravel()
    c = np.asarray(np.real(control), dtype=float).ravel()
    if r.size != c.size:
        raise ValueError(f'the reference has {r.size} samples and the '
                         f'control {c.size}; align them first')
    # a NaN would be dropped from a ratio as if it were not there, and
    # a run with a dead half would pass on the half that is
    # (2026-10-16 review)
    for name, x in (('reference', r), ('control', c)):
        bad = int(np.sum(~np.isfinite(x)))
        if bad:
            raise ValueError(f'the {name} holds {bad} samples that are '
                             'not numbers')
    rate = float(sample_rate)
    marks = as_segments(segments)
    settings = {
        'taper_percent': float(taper_percent),
        'band': None if band is None else _band('band', band),
        'lines': float(lines),
        'asd_db': _band('ASD', asd_db), 'asd_percent': float(asd_percent),
        'variance_percent': _band('variance', variance_percent),
        'envelope_db': _band('envelope', envelope_db),
        'window': None if window is None else float(window),
        'srs_db': _band('SRS', srs_db),
        'q': DEFAULT_Q if q is None else float(q),
        'per_octave': PER_OCTAVE if per_octave is None else int(per_octave),
        'peak_percent': float(peak_percent), 'peaks': int(peaks)}
    if settings['band'] is not None and settings['band'][0] <= 0.0:
        raise ValueError('the test band starts above 0 Hz')
    if not 0.0 < settings['asd_percent'] <= 100.0:
        raise ValueError(f'{asd_percent:g} percent is not a share of the '
                         'lines')
    if not settings['lines'] > 0.0:
        raise ValueError('the line spacing is above 0 Hz')
    if not settings['peak_percent'] > 0.0 or settings['peaks'] < 1:
        raise ValueError('the peaks are read one or more each way, '
                         'within a share above 0 percent')
    total = float(np.sum(r ** 2))
    whole = (100.0 * float(np.linalg.norm(c - r)) / np.sqrt(total)
             if total > 0.0 else float('nan'))
    envelope = (settings['window'] if settings['window'] is not None
                else ENVELOPE_PERCENT / 100.0 * r.size / rate)
    span = max(round(envelope * rate), 2)
    frame = max(round(rate / settings['lines']), 2)

    results = []
    for segment, label in zip(marks, _label(marks)):
        first = round(segment.start * rate)
        stop = round(segment.end * rate)
        if first < 0 or stop > r.size:
            raise ValueError(f'{label} runs {segment.start:g} to '
                             f'{segment.end:g} s, outside the record\'s '
                             f'{r.size / rate:g} s')
        n = stop - first
        if n < 8:
            raise ValueError(f'{label} holds {n} samples, too few to judge')
        rs, cs = r[first:stop], c[first:stop]
        weights = tapered_cosine(n, settings['taper_percent'])
        rw, cw = rs * weights, cs * weights
        own = float(np.sum(rs ** 2))
        share = 100.0 * own / total if total > 0.0 else float('nan')
        error = (100.0 * float(np.linalg.norm(cs - rs)) / np.sqrt(own)
                 if own > 0.0 else float('nan'))
        if segment.kind == 'stationary':
            want = float(np.mean(rw ** 2))
            got = float(np.mean(cw ** 2))
            ratio = (100.0 * (got / want - 1.0) if want > 0.0
                     else float('nan'))
            variance = (_check('variance', '%', settings['variance_percent'],
                               [segment.start], [ratio])
                        if np.isfinite(ratio) else
                        _check('variance', '%', settings['variance_percent'],
                               [], []))
            checks = [_spectral('ASD', rw, cw, rate, min(frame, n),
                                settings['band'], settings['asd_db'],
                                settings['asd_percent'], False),
                      variance]
        elif segment.kind == 'transient':
            checks = [_envelope(rs, cs, min(span, n), first, rate,
                                settings['envelope_db']),
                      _spectral('normalized ASD', rw, cw, rate,
                                min(frame, n), settings['band'],
                                settings['asd_db'], settings['asd_percent'],
                                True)]
        else:
            # untapered: a taper over a mark drawn tight on the event
            # weights down its largest cycles in both traces and hides
            # a miss in their ratio (2026-10-16 review)
            checks = [*_srs_checks(rs, cs, rate, settings['band'],
                                   settings['srs_db'], settings['q'],
                                   settings['per_octave']),
                      _peak_check(rs, cs, first, rate, settings['peaks'],
                                  settings['peak_percent'])]
        results.append(SegmentResult(
            segment, label, first, n, share, error, tuple(checks),
            _verdict([check.passed for check in checks])))
    return SegmentedReplication(
        tuple(results), whole, rate, int(r.size),
        _verdict([result.passed for result in results]), settings)


def _verdict(verdicts: Sequence[bool | None]) -> bool | None:
    """Failing when any fails; otherwise None when any could not be
    judged (or there is none), and passing only when every one passes.
    A reading that could not be made is not a pass."""
    if any(v is False for v in verdicts):
        return False
    if not verdicts or any(v is None for v in verdicts):
        return None
    return True


def aligned_traces(measured: TimeHistory, specification: DataArray,
                   channel: str | None = None, lag: int | None = None,
                   playing: int = 0
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, int]:
    """(time, reference, control, channel, lag): one channel of a run
    and of the reference it was controlled to, aligned, over one playing.

    The reference is the specification's record at `channel` (its first
    DOF by default), the control the run's record at the same DOF and
    dimension. The two are aligned by `replication.lag_of`'s one shift,
    unless `lag` is given, and the control is cut to the `playing`'th
    whole playing of the reference (`replication.playings`): for a
    reference as long as the run, the one playing there is. `time` is
    the reference's own clock, from zero, in seconds. Values are as
    stored (SI).

    Raises
    ------
    ValueError
        When the reference names no such channel, the run has no record
        at it, or the run does not hold that playing.
    """
    from . import replication as rep

    dofs = [str(dof) for dof in specification.response_dof]
    dof = dofs[0] if channel is None else str(channel)
    if dof not in dofs:
        raise ValueError(f'the reference names no channel {dof}')
    index = dofs.index(dof)
    dim = specification.ordinate_dim[index]
    (row,) = rep._rows_by_dof(measured, [dof], [dim])
    if row is None:
        raise ValueError(f'the run has no {dim} record at {dof}')
    averaging = rep.averaging_for(measured, specification)
    if averaging is None:
        raise ValueError('the run is shorter than one playing of the '
                         'reference')
    if lag is None:
        lag = rep.lag_of(measured, specification, averaging)
    found = [p for p in rep.playings(measured, specification, averaging, lag)
             if p['frame'] == int(playing)]
    if not found:
        raise ValueError(f'the run holds no whole playing {int(playing) + 1} '
                         'of the reference')
    start, n = found[0]['start'], found[0]['samples']
    reference = np.real(np.asarray(specification.ordinate[index][:n],
                                   dtype=complex)).astype(float)
    control = np.real(np.asarray(measured.ordinate[row, start:start + n],
                                 dtype=complex)).astype(float)
    time = np.arange(n) / float(specification.sample_rate)
    return time, reference, control, dof, int(lag)


def replication_segments(measured: TimeHistory, specification: DataArray,
                         segments: Sequence[Any], *,
                         channel: str | None = None, lag: int | None = None,
                         playing: int = 0,
                         **settings: Any) -> SegmentedReplication:
    """A replicated run judged segment by segment against the reference
    it was controlled to.

        reference = TransientSpecification(...)   # the field record
        result = replication_segments(run, reference, [
            (6.0, 30.0, 'stationary', 'rough road'),
            (31.9, 32.7, 'shock', 'pothole'),
            (41.0, 51.0, 'transient', 'braking')], band=(5.0, 400.0))
        print(result.describe())
        for segment in result.segments:
            print(segment.describe())

    `replication.compare` reads the run as a whole, which hides a short
    event's miss; this reads each marked segment by the rule for its
    kind (`judge_segments`) and reports the whole-record rms error
    beside them. The two traces are taken at `channel` (the reference's
    first DOF by default), over the `playing`'th whole playing of the
    reference in the run, aligned by `replication.lag_of` unless `lag`
    is given (`aligned_traces`). Segment times are seconds on the
    reference's clock, from its first sample.

    Parameters
    ----------
    measured : TimeHistory
        The run.
    specification : TransientSpecification or TimeHistory
        The reference it was controlled to.
    segments : sequence
        The marks, each a `Segment` or (start, end, kind[, name]).
    channel, lag, playing
        As for `aligned_traces`.
    **settings
        The tolerances, as `judge_segments` takes them.

    Returns
    -------
    SegmentedReplication
        With `channel` and `lag` set.
    """
    _time, reference, control, dof, shift = aligned_traces(
        measured, specification, channel, lag, playing)
    result = judge_segments(reference, control,
                            float(specification.sample_rate), segments,
                            **settings)
    return replace(result, channel=dof, lag=shift)
