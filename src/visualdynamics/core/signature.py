"""Two signatures of one article compared mode by mode.

A structural signature is a low-level measurement of the article at one
response point (an FRF, a transmissibility, or the level a tracked sine
sweep reads), taken the same way before a test and after it. Laid over
each other, the two say whether the article changed: a mode that has
dropped in frequency has lost stiffness, and one that has lost height
has gained damping or moved its motion elsewhere. An overlay shows
that; a criterion decides it. Test documents write the criterion per
mode, as a limit on how far each mode that counts may move in frequency
and in amplitude, and they name which modes count (usually the global
modes, the ones with a large share of the article's mass moving; a
local mode, a bracket or a cable, moves when a clamp is re-seated and
means little).

`compare_signatures` does that for two records, and `signature_shifts`
for two curves held as plain arrays. Each mode is named by a frequency
near it. The mode is found on each curve as the tallest line inside a
window about that frequency, its top placed between lines and its
half-power width read, and the two readings give the mode's frequency
shift and amplitude shift in percent of the reading before. A mode that
counts is judged against both limits and passes or is flagged; a mode
that does not count is reported with its shifts and not judged, because
the reader still wants to see it move. `find_modes` names the modes when
nobody has: the most prominent peaks of the curve before the test,
ranked by the rule the modal fit's suggestion ranks them by; a shape
set the modal fit has confirmed names its modes too.

The limits default to plus or minus 5 percent in frequency and plus or
minus 20 percent in amplitude, an example of what test documents ask
for and not any one document's requirement; the test plan states the
numbers, and either limit can be written as a pair when the plan holds
a drop and a rise to different values. A shift on a limit's edge is
inside it.

The two readings are of the curve, not of a fitted model: a peak read
on a transmissibility sits a little below the mode's natural frequency
(by about one part in 4 Q squared), and it sits there the same way
before and after, so the shift between the two is the mode's. A mode
that moves out of its window between the two signatures is read as
whatever peak is tallest in the window after; the window is wide (a
fifth of the frequency either side by default) and is narrowed to stop
halfway, on a log scale, to the next mode named, so two modes named
are never read as one. What a reading cannot be trusted for is said in
its notes, and the comparison gathers them: a peak on its window's
edge, a top placed outside the window, lines that hold no value, a
half-power width that cannot be read, and two modes read on one peak.
Two signatures of different quantities (an accelerance and a
transmissibility) are refused, since their heights are not one scale.

References
----------
The peak reading is a standard method; this is where it is described.

1. Ewins, D. J. (2000). *Modal Testing: Theory, Practice and
   Application*, 2nd ed. Research Studies Press. The peak-amplitude
   (half-power point) reading of a mode's frequency and Q from its
   peak, which each mode here is read with before and after.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

#: the frequency limit, in percent of the frequency before the test,
#: either way: an example of what test documents ask for
FREQUENCY_LIMIT = 5.0

#: the amplitude limit, in percent of the peak's height before the test,
#: either way: an example of what test documents ask for
AMPLITUDE_LIMIT = 20.0

#: how far either side of the frequency a mode is named at its window
#: reaches, as a share of that frequency. Wide, because the mode after
#: the test must still be inside it: a local mode that has dropped 8.5
#: percent is still in a window of 20, and a neighbor named beside it
#: narrows the window rather than being read in its place
SEARCH = 0.2

#: how far past a limit's edge, in percent, a shift still counts as on
#: it: a shift computed as exactly 5 percent reads 5.000000000000004 in
#: floating point, and a shift on the edge is inside
EDGE_PERCENT = 1e-9

@dataclass(frozen=True)
class SignatureMode:
    """A mode named for the comparison: a frequency near it, and
    whether it counts.

    `frequency` is in Hz and need not be exact; the mode is read as the
    tallest peak in a window about it. `within`, a (low, high) band in
    Hz, sets that window outright, for a mode with a taller neighbor
    close by. `name` labels the mode in the table ('first bending',
    'panel'); empty, the mode is labeled by its frequency.
    """

    frequency: float
    counts: bool = True
    within: tuple[float, float] | None = None
    name: str = ''

    def label(self) -> str:
        """What the table calls the mode: its name, or its frequency."""
        return self.name or f'{self.frequency:.4g} Hz'


@dataclass(frozen=True)
class PeakReading:
    """One mode read off one curve: the peak's frequency and height,
    and its half-power crossings.

    `low` and `high` are None where no half-power crossing can be read
    on that side, and `q` is then None. `notes` says what the reading
    cannot be trusted for, as text.
    """

    frequency: float
    height: float
    low: float | None
    high: float | None
    notes: tuple[str, ...] = ()

    @property
    def q(self) -> float | None:
        """Q: the peak frequency over the half-power width."""
        if self.low is None or self.high is None or self.high <= self.low:
            return None
        return self.frequency / (self.high - self.low)


@dataclass(frozen=True)
class ModeShift:
    """One mode before and after, its two shifts and the verdict.

    The shifts are in percent of the reading before the test, negative
    for a drop. The limits are (below, above) in percent, as the
    comparison was made with them. `passed` is None for a mode that
    does not count, which is reported and not judged. `notes` holds the
    two readings' notes, each said 'before the test' or 'after the
    test', and a note when this mode is read on the same peak as
    another.
    """

    mode: SignatureMode
    window: tuple[float, float]
    before: PeakReading
    after: PeakReading
    frequency_limit: tuple[float, float]
    amplitude_limit: tuple[float, float]
    notes: tuple[str, ...] = ()

    @property
    def frequency_shift(self) -> float:
        """The frequency shift in percent of the frequency before."""
        return 100.0 * (self.after.frequency / self.before.frequency - 1.0)

    @property
    def amplitude_shift(self) -> float:
        """The amplitude shift in percent of the height before."""
        return 100.0 * (self.after.height / self.before.height - 1.0)

    @property
    def frequency_inside(self) -> bool:
        """Whether the frequency shift is inside its limit."""
        return _inside(self.frequency_shift, self.frequency_limit)

    @property
    def amplitude_inside(self) -> bool:
        """Whether the amplitude shift is inside its limit."""
        return _inside(self.amplitude_shift, self.amplitude_limit)

    @property
    def passed(self) -> bool | None:
        """True inside both limits, False outside either, None for a
        mode that does not count."""
        if not self.mode.counts:
            return None
        return self.frequency_inside and self.amplitude_inside

    @property
    def verdict(self) -> str:
        """'pass', 'flag', or 'not judged' for a mode that does not
        count."""
        return {True: 'pass', False: 'flag', None: 'not judged'}[self.passed]

    def box(self) -> tuple[float, float, float, float]:
        """The window the limits draw about the mode before the test:
        (low frequency, high frequency, low height, high height). The
        mode after the test passes when its peak is inside it."""
        f, h = self.before.frequency, self.before.height
        return (f * (1.0 + self.frequency_limit[0] / 100.0),
                f * (1.0 + self.frequency_limit[1] / 100.0),
                h * (1.0 + self.amplitude_limit[0] / 100.0),
                h * (1.0 + self.amplitude_limit[1] / 100.0))

    def describe(self) -> str:
        """The mode in one line, as the table and the status line give
        it: '84.0 to 81.2 Hz (−3.3 %), height 18.0 to 13.0 (−27.6 %,
        outside): flag'. A shift past its limit on a mode that counts is
        marked 'outside'."""
        def shift(value, inside):
            text = f'{value:+.1f} %'.replace('-', '−')
            return text if inside or not self.mode.counts \
                else f'{text}, outside'

        return (f'{self.before.frequency:.1f} to {self.after.frequency:.1f} Hz '
                f'({shift(self.frequency_shift, self.frequency_inside)}), '
                f'height {self.before.height:#.3g} to {self.after.height:#.3g} '
                f'({shift(self.amplitude_shift, self.amplitude_inside)}): '
                f'{self.verdict}')


@dataclass(frozen=True)
class SignatureComparison:
    """Every mode named, compared: what `compare_signatures` and
    `signature_shifts` return.

    `shifts` holds one `ModeShift` per mode, in ascending frequency.
    `dof` names the record compared when the comparison was of records,
    and is empty for plain arrays. `notes` gathers every mode's notes,
    each led by the mode's label.
    """

    shifts: tuple[ModeShift, ...]
    frequency_limit: tuple[float, float]
    amplitude_limit: tuple[float, float]
    dof: str = ''
    notes: tuple[str, ...] = field(default=())

    @property
    def judged(self) -> tuple[ModeShift, ...]:
        """The modes that count."""
        return tuple(s for s in self.shifts if s.mode.counts)

    @property
    def flagged(self) -> tuple[ModeShift, ...]:
        """The modes that count and are outside a limit."""
        return tuple(s for s in self.shifts if s.passed is False)

    @property
    def passed(self) -> bool | None:
        """True when every mode that counts is inside both limits,
        False when any is flagged, None when no mode counts."""
        if not self.judged:
            return None
        return not self.flagged

    def describe(self) -> str:
        """The verdict in one line: 'fails: 1 of 1 mode that counts
        flagged; 1 not judged', and how many notes the readings carry
        ('; 2 notes')."""
        judged, flagged = len(self.judged), len(self.flagged)
        others = len(self.shifts) - judged
        tail = f'; {others} not judged' if others else ''
        if self.notes:
            count = len(self.notes)
            tail += f'; {count} note' if count == 1 else f'; {count} notes'
        if not judged:
            return f'no mode counts{tail}'
        verdict = 'passes' if not flagged else 'fails'
        word = 'mode that counts' if judged == 1 else 'modes that count'
        return f'{verdict}: {flagged} of {judged} {word} flagged{tail}'


def _limit(name: str, value: Any) -> tuple[float, float]:
    """(below, above) in percent from a symmetric limit or a pair,
    checked: the first not above zero, the second not below."""
    try:
        if np.ndim(value) == 0:
            below, above = -abs(float(value)), abs(float(value))
        else:
            below, above = (float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f'the {name} limit is a percentage or a pair '
                         f'(below, above), not {value!r}') from None
    if not (np.isfinite(below) and np.isfinite(above)
            and below <= 0.0 <= above):
        raise ValueError(f'the {name} limit runs from {below:g} to '
                         f'{above:g} percent, which does not hold zero')
    return below, above


def _inside(shift: float, limit: tuple[float, float]) -> bool:
    return limit[0] - EDGE_PERCENT <= shift <= limit[1] + EDGE_PERCENT


def as_modes(modes: Any) -> list[SignatureMode]:
    """The modes named, as `SignatureMode`s in ascending frequency.

    Each may be a `SignatureMode`, a frequency in Hz (a mode that
    counts), or a (frequency, counts) pair. A shape set (anything with a
    `frequency` array, as the modal fit makes) names every mode it
    holds, all counting, except a rigid-body mode at 0 Hz, which has no
    peak to compare and is left out. A mode named by number at a
    frequency that is not positive is refused.
    """
    if hasattr(modes, 'frequency') and not isinstance(modes, SignatureMode):
        modes = [float(f) for f in np.atleast_1d(modes.frequency)
                 if float(f) > 0.0]
    elif isinstance(modes, (SignatureMode, int, float, np.floating)):
        modes = [modes]
    out = []
    for mode in modes:
        if isinstance(mode, SignatureMode):
            out.append(mode)
        elif np.ndim(mode) == 0:
            out.append(SignatureMode(float(mode)))
        else:
            frequency, counts = mode
            out.append(SignatureMode(float(frequency), bool(counts)))
    for mode in out:
        if not (np.isfinite(mode.frequency) and mode.frequency > 0.0):
            raise ValueError(f'a mode is named at {mode.frequency!r} Hz; '
                             'a mode is named at a positive frequency')
    return sorted(out, key=lambda m: m.frequency)


def mode_windows(modes: Sequence[SignatureMode],
                 search: float = SEARCH) -> list[tuple[float, float]]:
    """The window each mode is read in, in Hz, in the order given.

    A mode's own `within` is its window. Otherwise the window reaches
    `search` of the frequency either side, and stops halfway, on a log
    scale, to the next mode named on either side, so two modes named
    close together are each read on their own peak.
    """
    frequencies = [m.frequency for m in modes]
    out = []
    for k, mode in enumerate(modes):
        if mode.within is not None:
            low, high = sorted(float(v) for v in mode.within)
            out.append((low, high))
            continue
        low = mode.frequency * (1.0 - search)
        high = mode.frequency * (1.0 + search)
        below = [f for f in frequencies[:k] if f < mode.frequency]
        above = [f for f in frequencies[k + 1:] if f > mode.frequency]
        if below:
            low = max(low, float(np.sqrt(below[-1] * mode.frequency)))
        if above:
            high = min(high, float(np.sqrt(above[0] * mode.frequency)))
        out.append((low, high))
    return out


def read_peak(frequencies: ArrayLike, magnitude: ArrayLike,
              within: tuple[float, float]) -> PeakReading:
    """The tallest peak inside a band, read by its top and its
    half-power width.

    The reading is `core.half_power.read_half_power`'s for the band:
    the peak is the tallest line between `within`'s two frequencies,
    its top is placed between lines by the parabola through the
    logarithms of that line and its two neighbors, and the two
    crossings of 0.707 of that height are interpolated linearly, on the
    whole curve, not only inside the band. Its notes are kept, and
    three more are said here. A tallest line on the band's edge is a
    peak outside it still rising. A top placed outside the band, which
    a peak on the edge can have, is a reading of something outside the
    window. A line that holds no value (NaN or infinite) is no part of
    the peak, and one inside the band or between the crossings is
    said, since the top or a crossing beside it is placed without it.

    Parameters
    ----------
    frequencies : array-like
        The curve's frequencies in Hz, ascending.
    magnitude : array-like
        The curve; complex values are read by their magnitude.
    within : tuple of float
        The band (low, high) in Hz the peak is in.

    Returns
    -------
    PeakReading
        The peak's frequency, height and crossings.

    Raises
    ------
    ValueError
        When no line of the curve lies inside the band, or no line
        there stands above zero, so there is no peak to read.
    """
    from .half_power import read_half_power

    f = np.asarray(frequencies, dtype=np.float64).ravel()
    m = np.abs(np.asarray(magnitude)).astype(np.float64).ravel()
    low, high = sorted(float(v) for v in within)
    band = np.flatnonzero((f >= low) & (f <= high))
    if not len(band):
        raise ValueError(f'no line of the curve lies between {low:.4g} and '
                         f'{high:.4g} Hz')
    finite = np.isfinite(m)
    # a band of zeros (or of lines that hold no value) has no peak, and
    # a height of zero before the test would divide the amplitude shift
    if not np.any(finite[band] & (m[band] > 0.0)):
        raise ValueError(f'no peak can be read between {low:.4g} and '
                         f'{high:.4g} Hz: no line there stands above zero')
    # the tallest line as read_half_power finds it, a line with no value
    # never the tallest, for the edge note
    i = int(band[np.argmax(np.where(finite, m, -np.inf)[band])])
    reading = read_half_power(f, m, f[i], within=(low, high), warn=False)
    notes = []
    outside = not low <= reading.frequency <= high
    if i in (band[0], band[-1]) and len(band) > 1:
        note = (f'the tallest line between {low:.4g} and {high:.4g} Hz '
                f'is on the window\'s edge, at {f[i]:.4g} Hz: the peak '
                'there may be outside the window')
        if outside:
            note += (f', and its top, placed between lines, is at '
                     f'{reading.frequency:.4g} Hz, outside it')
        notes.append(note)
    elif outside:
        notes.append(f'the top, placed between lines, is at '
                     f'{reading.frequency:.4g} Hz, outside the window '
                     f'{low:.4g} to {high:.4g} Hz')
    # what the reading touched: the band, and out to the crossings
    span_low = low if reading.low is None else min(low, reading.low)
    span_high = high if reading.high is None else max(high, reading.high)
    unread = int(np.count_nonzero(~finite & (f >= span_low)
                                  & (f <= span_high)))
    if unread:
        lines, hold, are = (('line', 'holds', 'is') if unread == 1
                            else ('lines', 'hold', 'are'))
        notes.append(f'{unread} {lines} between {span_low:.4g} and '
                     f'{span_high:.4g} Hz {hold} no value (NaN or infinite) '
                     f'and {are} read as no part of the peak')
    return PeakReading(reading.frequency, reading.height, reading.low,
                       reading.high, tuple(notes) + reading.notes)


def _top_line(f: np.ndarray, m: np.ndarray, frequency: float) -> float:
    """The top of the peak a reading at `frequency` sits on, found by
    climbing uphill from the nearest line: two readings with the same
    top are on one peak."""
    from .half_power import read_half_power

    return read_half_power(f, m, frequency, warn=False).frequency


def signature_shifts(frequencies: ArrayLike, before: ArrayLike,
                     after: ArrayLike, modes: Any, *,
                     after_frequencies: ArrayLike | None = None,
                     frequency_limit: Any = FREQUENCY_LIMIT,
                     amplitude_limit: Any = AMPLITUDE_LIMIT,
                     search: float = SEARCH) -> SignatureComparison:
    """Two signatures held as plain arrays, compared mode by mode: the
    array form of `compare_signatures`.

        result = signature_shifts(f, before, after, [84.0, (412.0, False)])
        result.passed, [s.describe() for s in result.shifts]

    Each mode is read on each curve in its window (`mode_windows`) by
    `read_peak`; its frequency shift and amplitude shift are in percent
    of the reading before, and a mode that counts passes when both are
    inside their limits. The two curves may be on different
    frequencies (a sine sweep run at another rate, a random at another
    resolution); neither needs to be ascending.

    Parameters
    ----------
    frequencies : array-like
        The frequencies of `before`, in Hz.
    before, after : array-like
        The two curves; complex values are read by their magnitude.
    modes : float, SignatureMode, (float, bool), sequence, or shape set
        The modes compared, as `as_modes` reads them.
    after_frequencies : array-like, optional
        The frequencies of `after`, when not those of `before`.
    frequency_limit, amplitude_limit : float or (float, float)
        The limits in percent: a number is that far either way, a pair
        is (below, above). `FREQUENCY_LIMIT` and `AMPLITUDE_LIMIT` by
        default.
    search : float, default `SEARCH`
        How far either side of its frequency a mode's window reaches,
        as a share of the frequency.

    Returns
    -------
    SignatureComparison
        One `ModeShift` per mode, the verdict over the modes that
        count, and the readings' notes.

    Raises
    ------
    ValueError
        For a limit that does not hold zero, a mode at a frequency that
        is not positive, curves whose lengths do not match their
        frequencies, or a mode's window with no line of a curve in it or
        no line above zero.
    """
    f_limit = _limit('frequency', frequency_limit)
    a_limit = _limit('amplitude', amplitude_limit)
    if not 0.0 < float(search) < 1.0:
        raise ValueError(f'a search of {search!r} is not a share of the '
                         'frequency between 0 and 1')
    curves = []
    for name, x, y in (('before', frequencies, before),
                       ('after', frequencies if after_frequencies is None
                        else after_frequencies, after)):
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y).ravel()
        if len(x) != len(y):
            raise ValueError(f'{len(x)} frequencies for {len(y)} values '
                             f'{name} the test')
        # a sweep run downward holds its frequencies descending
        order = np.argsort(x, kind='stable')
        curves.append((x[order], y[order]))
    named = as_modes(modes)
    windows = mode_windows(named, search)
    readings = [[read_peak(x, y, window) for x, y in curves]
                for window in windows]
    notes: list[list[str]] = [
        [f'{when} the test: {note}'
         for when, reading in zip(('before', 'after'), pair)
         for note in reading.notes] for pair in readings]
    # two modes read on one peak: a mode that has moved into its
    # neighbor's window is read as the neighbor, or the neighbor as it,
    # and neither shift is the mode's own. Said, on both modes.
    for side, (when, (x, y)) in enumerate(zip(('before', 'after'), curves)):
        tops = [_top_line(x, y, pair[side].frequency) for pair in readings]
        for k, top in enumerate(tops):
            for j, other in enumerate(tops):
                if j != k and other == top:
                    notes[k].append(
                        f'{when} the test: read on the same peak as '
                        f'{named[j].label()}, which tops at {top:.4g} Hz')
    shifts = tuple(
        ModeShift(mode, window, pair[0], pair[1], f_limit, a_limit,
                  tuple(said))
        for mode, window, pair, said in zip(named, windows, readings, notes))
    return SignatureComparison(
        shifts, f_limit, a_limit,
        notes=tuple(f'{s.mode.label()}, {note}' for s in shifts
                    for note in s.notes))


def matching_record(data: Any, record: int, other: Any) -> int:
    """The record of `other` at the same response and reference DOF as
    record `record` of `data`, else the same index."""
    def key(obj, i):
        ref = getattr(obj, 'reference_dof', None)
        return (str(obj.response_dof[i]),
                None if ref is None else str(ref[i]))

    wanted = key(data, record)
    for i in range(other.num_records):
        if key(other, i) == wanted:
            return i
    if record < other.num_records:
        return record
    raise ValueError(f'the signature after the test has no record at '
                     f'{wanted[0]} and no record {record}')


def compare_signatures(before: Any, after: Any, modes: Any, *,
                       record: int = 0, after_record: int | None = None,
                       frequency_limit: Any = FREQUENCY_LIMIT,
                       amplitude_limit: Any = AMPLITUDE_LIMIT,
                       search: float = SEARCH) -> SignatureComparison:
    """One record of a signature before the test and the same record
    after it, compared mode by mode against frequency and amplitude
    limits.

        result = compare_signatures(before, after, [84.0, (412.0, False)])
        result.describe()          # 'fails: 1 of 1 mode that counts ...'

    `before` and `after` are any curves of one quantity against
    frequency: two FRFs, two transmissibilities, or the levels two
    tracked sine sweeps read. The record after is the one at the same
    response (and reference) DOF as the record before, or the same
    index when there is none. The comparison itself is
    `signature_shifts`'s, which says how each mode is read and judged;
    `find_modes` names the modes from the curve before when the test
    plan does not. The two records must be of one quantity: an
    accelerance and a transmissibility are not one scale, and a pair
    whose declared dimensions differ is refused. A record whose units
    are undeclared is taken as the other's.

    Parameters
    ----------
    before, after : DataArray
        The two signatures.
    modes : float, SignatureMode, (float, bool), sequence, or shape set
        The modes compared, as `as_modes` reads them.
    record : int, default 0
        Which record of `before`.
    after_record : int, optional
        Which record of `after`; by DOF as above when omitted.
    frequency_limit, amplitude_limit, search
        As for `signature_shifts`.

    Returns
    -------
    SignatureComparison
        One `ModeShift` per mode, the verdict, and `dof`, the record's
        response DOF.

    Raises
    ------
    ValueError
        For the two records' declared dimensions differing, and for
        anything `signature_shifts` refuses.
    """
    from ..units import UNKNOWN

    if after_record is None:
        after_record = matching_record(before, record, after)
    dims = (str(before.ordinate_dim[record]),
            str(after.ordinate_dim[after_record]))
    if UNKNOWN not in dims and dims[0] != dims[1]:
        raise ValueError(f'the signature before is {dims[0]} and the one '
                         f'after is {dims[1]}: two signatures of one '
                         'quantity are compared')
    result = signature_shifts(
        before.abscissa, np.atleast_2d(before.ordinate)[record],
        np.atleast_2d(after.ordinate)[after_record], modes,
        after_frequencies=after.abscissa,
        frequency_limit=frequency_limit, amplitude_limit=amplitude_limit,
        search=search)
    return SignatureComparison(result.shifts, result.frequency_limit,
                               result.amplitude_limit,
                               dof=str(before.response_dof[record]),
                               notes=result.notes)


def find_modes(data: Any, record: int = 0, count: int = 2, *,
               bounds: tuple[float, float] | None = None) -> list[float]:
    """The frequencies of the `count` most prominent peaks of one
    record, ascending: the modes to compare when the test plan does not
    name them.

    Peaks are ranked as the modal fit's suggestion ranks them
    (`modal_fit.prominence`): by how far each stands above the median
    of its surroundings on a log scale, not by height, so a curve that
    rises with frequency still offers its first mode first. A peak
    inside the half-power width of one already taken is the same mode
    and is passed over. The ranking reads the magnitude alone, so a
    tracked sweep's level, which carries no phase to fit, is read the
    same way as an FRF. To compare the modes a modal fit has confirmed
    instead, pass its shape set as the modes (`Project.fit_modes`).
    These are suggestions; which modes count is the test plan's to say.

    Parameters
    ----------
    data : DataArray
        The signature, usually the one before the test.
    record : int, default 0
        Which record.
    count : int, default 2
        How many modes to find.
    bounds : tuple of float, optional
        (low, high) in Hz to find them within; the whole curve when
        omitted.

    Returns
    -------
    list of float
        The peaks' frequencies in Hz, placed between lines, ascending.
        Fewer than `count` when the curve has fewer peaks.
    """
    from .half_power import HALF_POWER, crossings, read_half_power
    from .modal_fit import prominence, prominence_window

    f = np.asarray(data.abscissa, dtype=np.float64)
    order = np.argsort(f, kind='stable')
    f = f[order]
    m = np.abs(np.atleast_2d(data.ordinate)[record][order]).astype(np.float64)
    m = np.where(np.isfinite(m), m, 0.0)
    low, high = (f[0], f[-1]) if bounds is None else sorted(bounds)
    inside = np.flatnonzero((f >= low) & (f <= high))
    # every local maximum at once: a noisy 16k-line curve has thousands
    i = inside[(inside > 0) & (inside < len(m) - 1)]
    peaks = i[(m[i] > 0.0) & (m[i] >= m[i - 1]) & (m[i] > m[i + 1])].tolist()
    if not peaks:
        return []
    rank = prominence(m, peaks, prominence_window(len(inside)))
    taken: list[tuple[float, float, float]] = []
    for k in np.argsort(-rank, kind='stable'):
        if len(taken) >= int(count):
            break
        i = peaks[int(k)]
        if any(lo <= f[i] <= hi for _f, lo, hi in taken):
            continue
        top = read_half_power(f, m, f[i], within=(f[i], f[i]),
                              warn=False).frequency
        # the width a neighbor is passed over inside is measured down
        # from the line, not the top placed between lines: a top that
        # stands more than 3 dB over its line would give no crossing,
        # and a width of the whole curve would pass over every peak
        lo, hi = crossings(f, m, i, m[i] * HALF_POWER)
        taken.append((top,
                      f[0] if lo is None else lo,
                      f[-1] if hi is None else hi))
    return sorted(frequency for frequency, _lo, _hi in taken)
