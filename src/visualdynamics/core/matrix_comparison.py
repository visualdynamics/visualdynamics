"""A measured cross-spectral density matrix judged against the one it
was controlled to, cell by cell.

A multi-input random test is controlled to a matrix, not to a set of
separate spectra. The diagonal holds each control channel's
autospectrum, its level; the cross terms hold how each pair of channels
moves together. A run can hold every level inside its band and still
not be the test that was asked for, if two channels that were to move
together moved independently, or moved in opposition. So the matrix is
judged in two parts, each in the quantity it is specified in.

The diagonal
------------
Each autospectrum is compared with its specification exactly as a
single channel is (`compliance.compare`, through `compliance.compare_all`):
the measured power against the specified power cell by cell, the share
of the band outside the abort limits, and the RMS error in dB. A level
is out by the same two states the channel's figure shows
(`compliance.lines_state`, `compliance.rms_state`), and the diagonal as
a whole passes or fails by the same rule a set of channels does
(`compliance.verdict`).

The cross terms
---------------
A cross term is complex, and it is not a level: its real part is not the
power of either channel, and it moves with the phase. It is judged as
the two quantities it is written in. The ordinary coherence of a pair,

    gamma^2 = |S_pq|^2 / (S_pp S_qq),

is the share of either channel's power that moves linearly with the
other, 0 to 1 and free of units. Its phase,

    phi = arg S_pq,

is the angle by which they move apart, in degrees. Both are read from
the specification and from the measurement at every measured line. The
specification is read the way it is drawn: between breakpoints the
magnitude of each term log-log and its phase linearly, the way the
authoring sheet reads a cross term, and on a specification of lines
or bands the value of the bin the line falls in. A pair of terms that
are power laws with one coherence between breakpoints keeps that
coherence exactly at every line between them.

A line is out when the measured coherence is further than the coherence
tolerance from the specified one (an absolute difference, since
coherence is already a ratio), or when the phase differs by more than
the phase tolerance, the difference taken the short way round the
circle. The phase is judged only where the specified coherence is at
least the floor: two channels that are asked to be nearly independent
have no phase to hold, and the estimate of it scatters without bound
as the coherence falls (its random error is sqrt(1 - gamma^2) /
(|gamma| sqrt(2 n)) radians at `n` averages). The coherence estimate
scatters too, by sqrt(2) (1 - gamma^2) / (|gamma| sqrt(n)) of its value:
about 0.05 at a coherence of 0.5 and a hundred averages, so a run that
is exactly its specification puts a few percent of its lines past a
tolerance of 0.1 by chance. A cross term is therefore out when either
reading is out on more than `compliance.LINES_PERCENT` of the lines it
judged, the share a level is allowed outside its abort limits, and not
on its worst line. And where the tolerance is inside twice the
estimate's scatter at the measurement's averages (`SCATTER_SIGMAS`),
the line is not judged at all: with too few averages for the tolerance
asked, a run that is its specification fails by chance, and the test
needs more averages, not a verdict.

The defaults (`COHERENCE_TOLERANCE` of 0.1 either way, `PHASE_TOLERANCE`
of 10 degrees either way, judged at a specified coherence of at least
`COHERENCE_FLOOR`, 0.5) are examples of what test documents ask for.
The test plan sets its own.

A pair the specification writes no cross term for is not judged and is
said to be not specified: an unstated pair is not a statement that the
channels are independent (`SpecificationDraft` makes the same rule).
The whole matrix passes when the diagonal passes and no cross term is
out, and has no verdict when nothing was judged.

The per-line summary
--------------------
For watching a run, or a step-up, at a glance, each measured line
carries the RMS over the control channels of their level errors in dB
(each channel's error read from its own comparison cells,
`compliance.judge`, and spread over the lines each cell covers), and
how many cross terms are out on either reading at that line.

`compare_matrix` reads a specification and a CPSD (`compute_cpsds`)
and returns a `MatrixComparison`; `cross_term_lines` is the per-line
judgment of one cross term on plain arrays.

References
----------
The cross-spectral density matrix, the ordinary coherence and the random
errors of coherence and phase estimates are Bendat and Piersol's.

1. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The cross-spectral density
   matrix of several records, the ordinary coherence read from it, and
   the normalized random errors of the coherence and the phase
   estimates.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..units import UNKNOWN
from .compliance import record_at
from .matrix_checks import channel_labels
from .progress import Ticker

#: how far the measured coherence may be from the specified one, either
#: way, as an absolute difference; an example of what test documents ask
#: for
COHERENCE_TOLERANCE = 0.1

#: how far the measured phase may be from the specified one, either way,
#: in degrees; an example of what test documents ask for
PHASE_TOLERANCE = 10.0

#: the specified coherence a cross term's phase is judged from: below
#: it the two channels are asked to be nearly independent, and the phase
#: between them is the estimate's scatter. At 0.5 and 50 averages one
#: standard deviation of the phase estimate is about 6 degrees
COHERENCE_FLOOR = 0.5

#: the slack a reading may sit past a tolerance and still be inside it:
#: a coherence or a phase exactly on its tolerance is inside, and the
#: arithmetic of reading it must not move it out
EDGE = 1e-9

#: a line is judged only where its tolerance is at least this many
#: standard deviations of the estimate's scatter at the measurement's
#: averages. Inside it, a run that is exactly its specification puts
#: more than 5 percent of its lines past the tolerance by chance, half
#: the share a cross term is allowed, and the rule would fail a correct
#: test: at a coherence of 0.5 and about sixty averages the coherence
#: estimate scatters by 0.065, and a tolerance of 0.1 failed ten runs in
#: ten that were exactly their specification
SCATTER_SIGMAS = 2.0


def _tolerances(coherence_tolerance: float, phase_tolerance: float,
                coherence_floor: float) -> tuple[float, float, float]:
    """The three settings checked: a coherence tolerance over 0 and at
    most 1, a phase tolerance over 0 and at most 180 degrees, a floor
    from 0 to under 1. Refused, naming the setting, otherwise."""
    try:
        values = (float(coherence_tolerance), float(phase_tolerance),
                  float(coherence_floor))
    except (TypeError, ValueError):
        raise ValueError(
            'the tolerances and the floor are numbers, not '
            f'{coherence_tolerance!r}, {phase_tolerance!r} and '
            f'{coherence_floor!r}') from None
    coherence, phase, floor = values
    if not (np.isfinite(coherence) and 0.0 < coherence <= 1.0):
        raise ValueError(f'the coherence tolerance is over 0 and at most 1, '
                         f'not {coherence_tolerance!r}')
    if not (np.isfinite(phase) and 0.0 < phase <= 180.0):
        raise ValueError(f'the phase tolerance is over 0 and at most 180 '
                         f'degrees, not {phase_tolerance!r}')
    if not (np.isfinite(floor) and 0.0 <= floor < 1.0):
        raise ValueError(f'the coherence floor is from 0 to under 1, not '
                         f'{coherence_floor!r}')
    return coherence, phase, floor


def level_text(difference_db: float) -> str:
    """A level error as the grid and the table write it, '+0.4 dB',
    rounded before its sign is read so that -0.04 is not written
    '-0.0 dB'; '—' for one that was not read."""
    value = float(difference_db)
    if not np.isfinite(value):
        return '—'
    return f'{round(value, 1) + 0.0:+.1f} dB'


def wrapped_degrees(angle: Any) -> np.ndarray:
    """An angle in degrees taken the short way round the circle, into
    -180 up to but not including 180: a phase 350 degrees off is 10
    degrees off."""
    angle = np.asarray(angle, dtype=float)
    return (angle + 180.0) % 360.0 - 180.0


def cross_term_lines(specified_coherence: Any, specified_phase: Any,
                     measured_coherence: Any, measured_phase: Any, *,
                     coherence_tolerance: float = COHERENCE_TOLERANCE,
                     phase_tolerance: float = PHASE_TOLERANCE,
                     coherence_floor: float = COHERENCE_FLOOR,
                     averages: float | None = None
                     ) -> dict[str, np.ndarray]:
    """One cross term judged line by line, on plain arrays: the array
    form of the judgment `compare_matrix` makes of each pair.

    A line's coherence is judged where both coherences are numbers, and
    is out when they differ by more than `coherence_tolerance`. Its
    phase is judged where the specified coherence is at least
    `coherence_floor`, the measured coherence is a number and both
    phases are, and is out when the two differ by more than
    `phase_tolerance` the short way round (`wrapped_degrees`). A
    reading exactly on its tolerance is inside.

    With `averages`, the number of averages the measured estimate holds
    (half its statistical degrees of freedom), a line is judged only
    where each tolerance is at least `SCATTER_SIGMAS` times the
    estimate's random error at the specified coherence: sqrt(2)
    (1 - gamma^2) |gamma| / sqrt(n) for the coherence and sqrt(1 -
    gamma^2) / (|gamma| sqrt(2 n)) radians for the phase.

    Parameters
    ----------
    specified_coherence, measured_coherence : array-like
        The ordinary coherence at each line, 0 to 1; not a number where
        it is undefined.
    specified_phase, measured_phase : array-like
        The phase at each line, in degrees.
    coherence_tolerance : float, default `COHERENCE_TOLERANCE`
        How far the coherence may be off, either way.
    phase_tolerance : float, default `PHASE_TOLERANCE`
        How far the phase may be off, in degrees either way.
    coherence_floor : float, default `COHERENCE_FLOOR`
        The specified coherence the phase is judged from.
    averages : float, optional
        The averages the measurement holds; when omitted no line is set
        aside for the scatter of its estimates.

    Returns
    -------
    dict of str to numpy.ndarray
        'coherence_error' (measured less specified) and 'phase_error'
        (measured less specified, wrapped), each not a number where it
        cannot be read; and the booleans 'coherence_judged',
        'coherence_out', 'phase_judged' and 'phase_out', and
        'coherence_scattered' and 'phase_scattered', the lines set aside
        because the tolerance is inside the estimate's scatter.

    Raises
    ------
    ValueError
        If a tolerance or the floor is out of range, the averages are
        not a positive number, or the four arrays are not one shape.
    """
    coherence_tolerance, phase_tolerance, coherence_floor = _tolerances(
        coherence_tolerance, phase_tolerance, coherence_floor)
    arrays = [np.asarray(a, dtype=float) for a in (
        specified_coherence, specified_phase, measured_coherence,
        measured_phase)]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError('the specified and measured coherence and phase '
                         'are one value per line, the same lines; their '
                         f'shapes are {[a.shape for a in arrays]}')
    s_coh, s_phase, m_coh, m_phase = arrays
    with np.errstate(invalid='ignore'):
        coherence_error = m_coh - s_coh
        phase_error = wrapped_degrees(m_phase - s_phase)
        coherence_judged = np.isfinite(coherence_error)
        phase_judged = (np.isfinite(phase_error) & np.isfinite(m_coh)
                        & (s_coh >= coherence_floor - EDGE))
        coherence_scattered = np.zeros(coherence_judged.shape, dtype=bool)
        phase_scattered = np.zeros(phase_judged.shape, dtype=bool)
        if averages is not None:
            averages = float(averages)
            if not averages > 0.0:
                raise ValueError('the averages a measurement holds are a '
                                 f'positive number, not {averages:g}')
            gamma = np.sqrt(np.clip(s_coh, 0.0, 1.0))
            with np.errstate(divide='ignore'):
                coherence_scatter = (np.sqrt(2.0) * (1.0 - gamma ** 2)
                                     * gamma / np.sqrt(averages))
                phase_scatter = np.degrees(np.sqrt(1.0 - gamma ** 2)
                                           / (gamma * np.sqrt(2.0 * averages)))
            coherence_scattered = coherence_judged & (
                coherence_tolerance + EDGE < SCATTER_SIGMAS * coherence_scatter)
            phase_scattered = phase_judged & (
                phase_tolerance + EDGE < SCATTER_SIGMAS * phase_scatter)
            coherence_judged &= ~coherence_scattered
            phase_judged &= ~phase_scattered
        coherence_out = coherence_judged & (
            np.abs(coherence_error) > coherence_tolerance + EDGE)
        phase_out = phase_judged & (
            np.abs(phase_error) > phase_tolerance + EDGE)
    return {'coherence_error': np.where(coherence_judged, coherence_error,
                                        np.nan),
            'phase_error': np.where(np.isfinite(phase_error), phase_error,
                                    np.nan),
            'coherence_judged': coherence_judged,
            'coherence_out': coherence_out,
            'phase_judged': phase_judged, 'phase_out': phase_out,
            'coherence_scattered': coherence_scattered,
            'phase_scattered': phase_scattered}


# ---- the records, laid out as a matrix ---------------------------------------

def _channel_of(data: Any, record: int) -> tuple[tuple[str, str],
                                                  tuple[str, str]]:
    """(row channel, column channel) of one record, each (DOF,
    quantity): a cross term 'acceleration*force/frequency' runs from an
    acceleration channel to a force channel, so a drive point's
    accelerometer and its force gauge are two channels."""
    from .data import channel_quantities

    response, reference = data.record_pair(record)
    out, into = channel_quantities(data.known_dim(record))
    return (str(response), out), (str(reference), into)


# `record_grid` gives which record holds each entry, not its values:
# the comparison reads the specification at the measured lines and
# names the records in its results.
def record_grid(data: Any, records: Sequence[int] | None = None
                ) -> tuple[list[tuple[str, str]], list[list[int | None]]]:
    """The records of a spectral matrix laid out as the matrix: its
    channels, and which record holds each (row, column) entry.

    The channels are the autospectra's, (DOF, quantity) in record
    order; a cross term between two of them takes the entry at its row
    and column, and one that names a channel with no autospectrum is
    left out. An entry no record holds is None. Two records for one
    entry are refused: which of them is the matrix is not in the data.

    Parameters
    ----------
    data : Psd or Specification
        The matrix's records.
    records : sequence of int, optional
        Which records; all of them when omitted.

    Returns
    -------
    tuple of (list of (str, str), list of list of int or None)
        The channels, and the record at each entry, row by row.

    Raises
    ------
    ValueError
        If two records hold one entry.
    """
    wanted = (range(data.num_records) if records is None
              else [int(i) for i in records])
    channels: list[tuple[str, str]] = []
    for i in wanted:
        if data.is_autospectrum(i):
            row, _column = _channel_of(data, i)
            if row not in channels:
                channels.append(row)
    index = {channel: k for k, channel in enumerate(channels)}
    grid: list[list[int | None]] = [[None] * len(channels)
                                    for _ in channels]
    for i in wanted:
        row, column = _channel_of(data, i)
        if row in index and column in index:
            r, c = index[row], index[column]
            if grid[r][c] is not None:
                raise ValueError(
                    f'{data.record_label(grid[r][c])} and '
                    f'{data.record_label(i)} are both the entry of '
                    f'{row[0]} against {column[0]}; keep one')
            grid[r][c] = i
    return channels, grid


def _disagrees(term: np.ndarray, mirror: np.ndarray) -> bool:
    """Whether a cross term and its mirror, both read at the same lines,
    are not each other's conjugate, to a part in a million, on the lines
    where both are numbers (a line that is not one is said by its own
    note)."""
    term = np.asarray(term, dtype=complex)
    other = np.conj(np.asarray(mirror, dtype=complex))
    finite = np.isfinite(term) & np.isfinite(other)
    if not finite.any():
        return False
    term, other = term[finite], other[finite]
    scale = float(np.max(np.abs(np.concatenate([term, other]))))
    return not np.allclose(term, other, rtol=1e-6, atol=1e-9 * scale)


# The specification is read at the measured lines by `compliance.record_at`,
# the one reader the drive prediction shares (its own reader read a
# zero at a breakpoint as a gap, and the two disagreed, 2026-10-28).


# ---- the readings -------------------------------------------------------------

@dataclass(frozen=True)
class LevelCheck:
    """One control channel's autospectrum against its specification,
    the diagonal entry of a `MatrixComparison`.

    `result` is `compliance.compare`'s, empty when the channel was not
    measured. `state` is 'in' or 'out' (by `compliance.lines_state` and
    `compliance.rms_state`), 'not judged' when nothing was compared,
    'not measured' when the measurement holds no autospectrum for the
    channel, or 'refused' with the reason in `reason`. `error_db` is the
    channel's level error at every measured line, measured over
    specified in dB, not a number where it was not judged. `lines_out`
    marks the measured lines outside the abort limits, as the channel's
    own comparison marks them (`compliance.exceedances`), None where the
    channel was not compared.
    """

    channel: str
    spec_record: int
    measured_record: int | None
    result: dict[str, Any]
    state: str
    reason: str = ''
    error_db: np.ndarray = field(default_factory=lambda: np.zeros(0),
                                 repr=False, compare=False)
    lines_out: np.ndarray | None = field(default=None, repr=False,
                                         compare=False)

    def words(self) -> str:
        """The verdict in words, as the panel lists it: 'in: 2.1 % of
        lines outside abort, +0.4 dB rms', the two numbers the
        channel's own figure carries (`compliance.readouts`)."""
        from .compliance import readouts

        if self.state == 'refused':
            return f'not compared: {self.reason}'
        if self.state in ('not judged', 'not measured'):
            return f'{self.state}: {self.reason}' if self.reason \
                else self.state
        parts = []
        for reading in readouts(self.result):
            if reading['name'] == 'rms':
                parts.append(f'{level_text(self.result["difference_db"])} '
                             'rms')
            else:
                parts.append(f'{reading["text"]} % of lines outside abort')
        if not np.isfinite(float(self.result.get('difference_db',
                                                  float('nan')))):
            parts.append('no power measured')
        return f'{self.state}: ' + ', '.join(parts)


@dataclass(frozen=True)
class CrossCheck:
    """One pair of control channels' cross term, its coherence and its
    phase, judged at every measured line.

    `row` and `column` are the pair's channels (`row` < `column`);
    above the diagonal the grid shows the coherence and below it the
    phase, of this one cross term, which is the term of `row` against
    `column` (the term the other way round is its conjugate, with the
    phase reversed). The arrays are one value per measured line: the
    specified and measured coherence and phase (degrees) and what
    `cross_term_lines` made of them.

    `coherence_state` and `phase_state` are 'in' or 'out' (out when
    more than `compliance.LINES_PERCENT` of the lines judged are out),
    'not judged' when no line was (the reason in `reason`), 'not
    specified' when the specification writes no cross term for the
    pair, 'not measured' when the measurement holds none, or 'refused'.
    `notes` say what was not judged on lines that were. `averages` is
    the number of averages the measured cross term holds, None when the
    measurement does not say.
    """

    row: int
    column: int
    label: str
    spec_record: int | None
    measured_record: int | None
    coherence_state: str
    phase_state: str
    reason: str = ''
    specified_coherence: np.ndarray = field(
        default_factory=lambda: np.zeros(0), repr=False, compare=False)
    measured_coherence: np.ndarray = field(
        default_factory=lambda: np.zeros(0), repr=False, compare=False)
    specified_phase: np.ndarray = field(
        default_factory=lambda: np.zeros(0), repr=False, compare=False)
    measured_phase: np.ndarray = field(
        default_factory=lambda: np.zeros(0), repr=False, compare=False)
    lines: dict[str, np.ndarray] = field(default_factory=dict, repr=False,
                                         compare=False)
    notes: tuple[str, ...] = ()
    averages: float | None = None

    def _scattered(self, reading: str) -> int:
        return int(np.count_nonzero(self.lines.get(f'{reading}_scattered',
                                                   False)))

    def _scatter_words(self, reading: str) -> str:
        count = self._scattered(reading)
        return (f'{count} line{"s" * (count != 1)} set aside: the tolerance '
                "is inside twice the estimate's scatter at "
                f'{self.averages:.0f} averages')

    def _count(self, reading: str) -> tuple[int, int]:
        judged = self.lines.get(f'{reading}_judged')
        if judged is None:
            return 0, 0
        return (int(np.count_nonzero(self.lines[f'{reading}_out'])),
                int(np.count_nonzero(judged)))

    @property
    def coherence_lines(self) -> int:
        """How many lines the coherence was judged on."""
        return self._count('coherence')[1]

    @property
    def phase_lines(self) -> int:
        """How many lines the phase was judged on."""
        return self._count('phase')[1]

    @property
    def coherence_percent(self) -> float:
        """The share of the lines judged whose coherence is out, in
        percent; not a number when none was judged."""
        out, judged = self._count('coherence')
        return 100.0 * out / judged if judged else float('nan')

    @property
    def phase_percent(self) -> float:
        """The share of the lines judged whose phase is out, in
        percent; not a number when none was judged."""
        out, judged = self._count('phase')
        return 100.0 * out / judged if judged else float('nan')

    @property
    def state(self) -> str:
        """The pair's one state: 'out' when either reading is, 'in'
        when a reading was judged and none is out, else the coherence's
        state (why nothing was judged)."""
        states = (self.coherence_state, self.phase_state)
        if 'out' in states:
            return 'out'
        if 'in' in states:
            return 'in'
        return self.coherence_state

    def words(self, reading: str, coherence_tolerance: float = COHERENCE_TOLERANCE,
              phase_tolerance: float = PHASE_TOLERANCE,
              coherence_floor: float = COHERENCE_FLOOR) -> str:
        """One reading's verdict in words, as the grid writes it over
        the cell: 'coherence in: 0 of 397 lines past ±0.10', 'phase out:
        23.4 % of 397 lines past ±10°', 'phase not judged: specified
        coherence under 0.50'. `reading` is 'coherence' or 'phase'."""
        state = getattr(self, f'{reading}_state')
        if state in ('in', 'out'):
            out, judged = self._count(reading)
            limit = (f'±{coherence_tolerance:.2f}' if reading == 'coherence'
                     else f'±{phase_tolerance:g}°')
            share = (f'{out} of {judged}' if state == 'in'
                     else f'{100.0 * out / judged:.1f} % of {judged}')
            words = f'{reading} {state}: {share} lines past {limit}'
            if self._scattered(reading):
                words += f'; {self._scatter_words(reading)}'
            return words
        reason = self.reason
        if state == 'not judged' and self._scattered(reading):
            reason = self._scatter_words(reading)
        elif reading == 'phase' and state == 'not judged' \
                and self.coherence_state in ('in', 'out'):
            reason = f'specified coherence under {coherence_floor:.2f}'
        return f'{reading} {state}: {reason}' if reason \
            else f'{reading} {state}'


@dataclass(frozen=True)
class MatrixCell:
    """What one cell of the grid shows: its place, which reading it is
    ('level' on the diagonal, 'coherence' above it, 'phase' below it),
    its state and its verdict in words."""

    row: int
    column: int
    reading: str
    label: str
    state: str
    words: str


@dataclass(frozen=True)
class MatrixComparison:
    """A measured cross-spectral matrix judged against its
    specification: what `compare_matrix` returns.

    `channels` are the control channels the two share, in the
    specification's order, labeled (`channel_labels`); `levels` holds
    one `LevelCheck` per channel and `pairs` one `CrossCheck` per pair,
    row by row above the diagonal. `frequency` is the measured lines,
    and the per-line summary is `level_rms_db` (the RMS over the
    channels of their level errors in dB) and `cross_out` (how many
    cross terms are out on either reading at each line).
    """

    channels: tuple[str, ...]
    levels: tuple[LevelCheck, ...]
    pairs: tuple[CrossCheck, ...]
    frequency: np.ndarray = field(repr=False, compare=False)
    scale_db: float = 0.0
    coherence_tolerance: float = COHERENCE_TOLERANCE
    phase_tolerance: float = PHASE_TOLERANCE
    coherence_floor: float = COHERENCE_FLOOR
    notes: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        """How many channels: the grid is this many on a side."""
        return len(self.channels)

    def pair(self, row: int, column: int) -> CrossCheck:
        """The cross term of two channels, either way round.

        Parameters
        ----------
        row, column : int
            The two channels, by position; not the same one.

        Returns
        -------
        CrossCheck
            The pair's judgment.
        """
        a, b = sorted((int(row), int(column)))
        if a == b:
            raise ValueError('a channel against itself is its level, not a '
                             'cross term')
        return next(p for p in self.pairs if (p.row, p.column) == (a, b))

    def cell(self, row: int, column: int) -> MatrixCell:
        """What one cell of the grid shows: on the diagonal the
        channel's level, above it the pair's coherence, below it the
        pair's phase.

        Parameters
        ----------
        row, column : int
            The cell, by position.

        Returns
        -------
        MatrixCell
            Its reading, state and words.
        """
        row, column = int(row), int(column)
        if row == column:
            level = self.levels[row]
            return MatrixCell(row, column, 'level', level.channel,
                              level.state, level.words())
        pair = self.pair(row, column)
        reading = 'coherence' if row < column else 'phase'
        label = (pair.label if row < column
                 else f'{self.channels[row]}/{self.channels[column]}')
        return MatrixCell(row, column, reading, label,
                          getattr(pair, f'{reading}_state'),
                          pair.words(reading, self.coherence_tolerance,
                                     self.phase_tolerance,
                                     self.coherence_floor))

    @property
    def level_rms_db(self) -> np.ndarray:
        """The RMS over the control channels of their level errors in
        dB, at each measured line; not a number where no channel was
        judged."""
        if not self.levels:
            return np.full(np.shape(self.frequency), np.nan)
        errors = np.array([level.error_db for level in self.levels])
        finite = np.isfinite(errors)
        count = finite.sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.where(finite, errors, 0.0) ** 2
            out = np.sqrt(mean.sum(axis=0) / count)
        return np.where(count > 0, out, np.nan)

    @property
    def cross_out(self) -> np.ndarray:
        """How many cross terms are out on either reading at each
        measured line."""
        out = np.zeros(np.shape(self.frequency), dtype=int)
        for pair in self.pairs:
            if pair.lines:
                out += (pair.lines['coherence_out']
                        | pair.lines['phase_out']).astype(int)
        return out

    @property
    def level_verdict(self) -> dict[str, Any]:
        """The diagonal's verdict by the rule a set of channels is
        judged by (`compliance.verdict`)."""
        from .compliance import verdict

        return verdict([(level.channel, level.result)
                        for level in self.levels])

    @property
    def passed(self) -> bool | None:
        """True when the diagonal passes and no cross term judged is
        out, False when either fails, None when nothing was judged."""
        levels = self.level_verdict['passed']
        judged = [p for p in self.pairs if p.state in ('in', 'out')]
        if levels is False or any(p.state == 'out' for p in judged):
            return False
        if levels is None and not judged:
            return None
        return True

    def with_tolerances(self, coherence_tolerance: float | None = None,
                        phase_tolerance: float | None = None,
                        coherence_floor: float | None = None
                        ) -> MatrixComparison:
        """The same matrix with its cross terms judged again at other
        tolerances.

        Nothing is read again: the levels and every coherence and phase
        are kept, and only the judgment moves, so a tolerance changed
        in the app redraws without comparing the levels a second time.

        Parameters
        ----------
        coherence_tolerance, phase_tolerance, coherence_floor : float, optional
            The new settings; this comparison's for any not given.

        Returns
        -------
        MatrixComparison
            The comparison at those tolerances.

        Raises
        ------
        ValueError
            If a setting is out of range.
        """
        settings = _tolerances(
            self.coherence_tolerance if coherence_tolerance is None
            else coherence_tolerance,
            self.phase_tolerance if phase_tolerance is None
            else phase_tolerance,
            self.coherence_floor if coherence_floor is None
            else coherence_floor)
        # a pair with readings is judged again; one not specified, not
        # measured or refused has none, and stays as it is
        pairs = tuple(_judged(pair, *settings)
                      if pair.specified_coherence.size else pair
                      for pair in self.pairs)
        return dataclasses.replace(
            self, pairs=pairs, coherence_tolerance=settings[0],
            phase_tolerance=settings[1], coherence_floor=settings[2])

    def table(self) -> list[dict[str, str]]:
        """The summary table: one row per reading of the matrix, the
        levels first, then each pair's coherence and phase.

        Returns
        -------
        list of dict of str to str
            Each with 'label', 'reading' ('level', 'coherence' or
            'phase'), 'value' (the RMS error, or the share of lines
            out, as shown), 'state' and 'words'.
        """
        rows = []
        for k, level in enumerate(self.levels):
            difference = float(level.result.get('difference_db',
                                                float('nan')))
            value = level_text(difference)
            rows.append({'label': level.channel, 'reading': 'level',
                         'value': value, 'state': level.state,
                         'words': self.cell(k, k).words})
        for pair in self.pairs:
            for reading in ('coherence', 'phase'):
                percent = getattr(pair, f'{reading}_percent')
                cell = (self.cell(pair.row, pair.column)
                        if reading == 'coherence'
                        else self.cell(pair.column, pair.row))
                rows.append({
                    'label': pair.label, 'reading': reading,
                    'value': (f'{percent:.1f} % out'
                              if np.isfinite(percent) else '—'),
                    'state': getattr(pair, f'{reading}_state'),
                    'words': cell.words})
        return rows

    def describe(self) -> str:
        """The verdict in one line: 'fails: levels pass on 3 channels;
        1 of 3 cross terms out (coherence 1, phase 0); 1 pair not
        specified'."""
        levels = self.level_verdict
        judged = [p for p in self.pairs if p.state in ('in', 'out')]
        out = [p for p in judged if p.state == 'out']
        unspecified = sum(1 for p in self.pairs
                          if p.state == 'not specified')
        verdict = {True: 'passes', False: 'fails',
                   None: 'nothing judged'}[self.passed]
        parts = []
        if levels['channels']:
            word = 'channel' if levels['channels'] == 1 else 'channels'
            state = 'pass' if levels['passed'] else 'fail'
            parts.append(f'levels {state} on {levels["channels"]} {word}')
        if judged:
            coherence = sum(1 for p in out if p.coherence_state == 'out')
            phase = sum(1 for p in out if p.phase_state == 'out')
            word = 'cross term' if len(judged) == 1 else 'cross terms'
            parts.append(f'{len(out)} of {len(judged)} {word} out '
                         f'(coherence {coherence}, phase {phase})')
        if unspecified:
            word = 'pair' if unspecified == 1 else 'pairs'
            parts.append(f'{unspecified} {word} not specified')
        # what was asked for and not judged is part of the verdict: a
        # pass over the rest is not a pass over these
        unread = sum(1 for p in self.pairs
                     if p.state in ('not judged', 'not measured', 'refused'))
        if unread:
            word = 'cross term' if unread == 1 else 'cross terms'
            parts.append(f'{unread} {word} not judged')
        dark = sum(1 for level in self.levels
                   if level.state not in ('in', 'out'))
        if dark:
            word = 'level' if dark == 1 else 'levels'
            parts.append(f'{dark} {word} not judged')
        return f'{verdict}: ' + '; '.join(parts) if parts else verdict


def _judged(check: CrossCheck, coherence_tolerance: float,
            phase_tolerance: float, coherence_floor: float) -> CrossCheck:
    """A cross term's readings judged at these tolerances: its lines
    (`cross_term_lines`) and its two states. Out when more than
    `compliance.LINES_PERCENT` of the lines judged are out, the share
    rule a level is held to (`compliance.lines_state`). `compare_matrix`
    and `MatrixComparison.with_tolerances` both judge a pair here."""
    from .compliance import lines_state

    judged = cross_term_lines(
        check.specified_coherence, check.specified_phase,
        check.measured_coherence, check.measured_phase,
        coherence_tolerance=coherence_tolerance,
        phase_tolerance=phase_tolerance, coherence_floor=coherence_floor,
        averages=check.averages)
    states = []
    for reading in ('coherence', 'phase'):
        count = int(np.count_nonzero(judged[f'{reading}_judged']))
        if not count:
            states.append('not judged')
            continue
        share = 100.0 * np.count_nonzero(judged[f'{reading}_out']) / count
        states.append('out' if lines_state(share) == 'over' else 'in')
    reason = ''
    checked = dataclasses.replace(check, lines=judged)
    if states[0] == 'not judged':
        reason = (checked._scatter_words('coherence')
                  if checked._scattered('coherence')
                  else check.notes[-1] if check.notes
                  else 'no measured line lies where the cross term is '
                  'written')
    return dataclasses.replace(checked, coherence_state=states[0],
                               phase_state=states[1], reason=reason)


def _non_finite_note(label: str, count: int) -> str:
    return (f'{count} line{"s" * (count != 1)} of the measured {label} '
            f'{"are" if count != 1 else "is"} not a number and not judged')


def compare_matrix(specification: Any, measured: Any, *,
                   coherence_tolerance: float = COHERENCE_TOLERANCE,
                   phase_tolerance: float = PHASE_TOLERANCE,
                   coherence_floor: float = COHERENCE_FLOOR,
                   spec_records: Sequence[int] | None = None,
                   measured_records: Sequence[int] | None = None,
                   scale_db: float | None = None,
                   progress: Callable[[int, int], None] | None = None
                   ) -> MatrixComparison:
    """A measured cross-spectral matrix judged against its
    specification: the levels on the diagonal, the coherence and phase
    of every cross term the specification writes.

    The control channels are the specification's autospectra, in its
    order, paired with the measurement's records by DOF and quantity
    (`compliance.matched_records`). Each level is
    `compliance.compare_all`'s, at one scale for the whole matrix
    (resolved and warned as `compare_all` does). Each cross term is
    judged line by line on the measured lines (`cross_term_lines`) and
    is out when more than `compliance.LINES_PERCENT` of the lines
    judged are out on either reading.

    Parameters
    ----------
    specification : Specification
        What the run was controlled to: its autospectra and the cross
        terms written for any pairs.
    measured : Psd
        The run's cross-spectral matrix (`TimeHistory.compute_cpsds`),
        with reference DOFs.
    coherence_tolerance : float, default `COHERENCE_TOLERANCE`
        How far the coherence may be off, either way.
    phase_tolerance : float, default `PHASE_TOLERANCE`
        How far the phase may be off, in degrees either way.
    coherence_floor : float, default `COHERENCE_FLOOR`
        The specified coherence a phase is judged from.
    spec_records, measured_records : sequence of int, optional
        Which records of each to use; all when omitted.
    scale_db : float, optional
        The comparison's scale in dB, added to the measured levels
        (coherence and phase do not change with it); resolved as
        `compare_all` resolves it when omitted.
    progress : callable, optional
        Told ``(done, total)`` as the levels and the cross terms are
        judged, one count each; what it raises stops the comparison.
        The app's progress bar uses it.

    Returns
    -------
    MatrixComparison
        Every cell's judgment, the per-line summary and the verdict.

    Raises
    ------
    ValueError
        If the measurement holds no cross terms, either selection of
        records is empty, the two cannot be compared
        (`compliance.comparable`), they share fewer than two control
        channels, or a tolerance is out of range.
    """
    # The cross terms were read by their real part through `compare`
    # until 2026-10-24; they are judged here in
    # the quantities a specification writes them in, and `compare` now
    # refuses them.
    from .compliance import (
        cell_lines,
        comparable,
        compare_all,
        coverage,
        exceedances,
        judge,
        lines_state,
        matched_records,
        resolved_scale_db,
        rms_state,
        units_check,
    )
    from .data import ordinary_coherence

    coherence_tolerance, phase_tolerance, coherence_floor = _tolerances(
        coherence_tolerance, phase_tolerance, coherence_floor)
    for records, which in ((spec_records, 'specification'),
                           (measured_records, 'measurement')):
        if records is not None and not len(records):
            raise ValueError(f'no records of the {which} are selected; '
                             'there is no matrix to judge')
    if getattr(measured, 'reference_dof', None) is None:
        raise ValueError('a matrix comparison needs a cross-spectral matrix: '
                         'records with a reference as well as a response '
                         '(compute_cpsds); a set of autospectra is compared '
                         'channel by channel')
    refused = comparable(specification, measured)
    if refused is not None:
        raise ValueError(refused)

    channels, grid = record_grid(specification, spec_records)
    labels = channel_labels(channels)
    n = len(channels)
    pairs = {spec: theirs for _label, spec, theirs in matched_records(
        specification, measured, spec_records, measured_records)}
    if sum(1 for k in range(n) if grid[k][k] in pairs) < 2:
        raise ValueError('a matrix comparison needs at least two control '
                         'channels that the specification and the '
                         'measurement share; these share '
                         f'{sum(1 for k in range(n) if grid[k][k] in pairs)}')

    lines = np.asarray(measured.abscissa, dtype=float)
    left, right = measured.bin_bounds()
    ordinate = np.asarray(measured.ordinate)
    notes: list[str] = []
    ticker = Ticker(progress)
    ticker.add(n + n * (n - 1) // 2)

    # ---- the diagonal: the existing comparison, at one scale
    diagonal = [grid[k][k] for k in range(n)]
    shared = [i for i in diagonal if i in pairs]
    scale = float(resolved_scale_db(specification, measured, shared,
                                    measured_records, scale_db,
                                    stacklevel=3))
    compared = dict(zip(shared, (result for _label, result in compare_all(
        specification, measured, shared, measured_records,
        scale_db=scale)), strict=True))
    levels = []
    for k, spec_index in enumerate(diagonal):
        ticker.tick()
        theirs = pairs.get(spec_index)
        error = np.full(lines.shape, np.nan)
        if theirs is None or not measured.is_autospectrum(theirs):
            levels.append(LevelCheck(
                labels[k], spec_index, None, {}, 'not measured',
                'the measurement holds no autospectrum for this channel',
                error))
            continue
        # a line that is not a number would be counted as no power,
        # reading the channel low without a word: refused instead
        inside = np.zeros(lines.shape, dtype=bool)
        for low, high in coverage(specification, spec_index):
            inside |= (right > low) & (left < high)
        bad = int(np.count_nonzero(~np.isfinite(ordinate[theirs]) & inside))
        if bad:
            levels.append(LevelCheck(
                labels[k], spec_index, theirs, {}, 'refused',
                _non_finite_note(labels[k], bad).replace(
                    ' and not judged', '; a level is not read across them'),
                error))
            continue
        result = compared[spec_index]
        if result.get('refused'):
            levels.append(LevelCheck(labels[k], spec_index, theirs, result,
                                     'refused', result['refused'], error))
            continue
        if not result.get('lines'):
            levels.append(LevelCheck(labels[k], spec_index, theirs, result,
                                     'not judged',
                                     'no line of the band was compared',
                                     error))
            continue
        out = (lines_state(result.get('abort_percent') or 0.0) == 'over'
               or rms_state(result.get('difference_db', 0.0)) != 'in')
        # the channel's error at every line, from its own comparison
        # cells, spread over the lines each covers
        level = judge(specification, measured, spec_index, theirs, None,
                      True, scale)
        with np.errstate(divide='ignore', invalid='ignore'):
            cell_db = 10.0 * np.log10(level['held'] / level['asked'])
        for cell, db, counted in zip(level['cells'], cell_db,
                                     level['judged'], strict=True):
            reach = cell_lines(cell, left, right)
            if reach is not None and counted and np.isfinite(db):
                error[reach] = db
        over, under = exceedances(specification, measured, spec_index,
                                  theirs, 'abort', scale)
        levels.append(LevelCheck(labels[k], spec_index, theirs, result,
                                 'out' if out else 'in', '', error,
                                 over | under))
        if result.get('note') and result['note'] not in notes:
            notes.append(result['note'])

    # ---- the cross terms: coherence and phase, line by line
    autos_measured = {k: levels[k].measured_record for k in range(n)}
    autos_specified = {k: record_at(specification, diagonal[k], lines)
                       for k in range(n)}
    # the measured cross terms found by the channels the autospectra
    # were paired with, either way round: a term held only the other way
    # is the conjugate of the one wanted
    measured_channels, measured_grid = record_grid(measured, measured_records)
    where = {key: k for k, key in enumerate(measured_channels)}

    def measured_term(p: int, q: int) -> tuple[int | None, float]:
        a = where.get(_channel_of(measured, p)[0])
        b = where.get(_channel_of(measured, q)[0])
        if a is None or b is None:
            return None, 1.0
        if measured_grid[a][b] is not None:
            return measured_grid[a][b], 1.0
        if measured_grid[b][a] is not None:
            return measured_grid[b][a], -1.0
        return None, 1.0

    # what each measured term is worth: half its statistical degrees of
    # freedom, the averages its scatter is read at
    counts = getattr(measured, 'statistical_dof', None)
    if counts is None:
        notes.append('the measurement does not say how many averages it '
                     'holds, so no line is set aside for the scatter of '
                     'its coherence and phase estimates')
    checks = []
    for r in range(n):
        for c in range(r + 1, n):
            ticker.tick()
            label = f'{labels[r]}/{labels[c]}'
            spec_index, flip = grid[r][c], 1.0
            if spec_index is None and grid[c][r] is not None:
                spec_index, flip = grid[c][r], -1.0
            if spec_index is None:
                checks.append(CrossCheck(
                    r, c, label, None, None, 'not specified',
                    'not specified',
                    'the specification writes no cross term for the pair'))
                continue
            p, q = autos_measured[r], autos_measured[c]
            theirs, measured_flip = ((None, 1.0) if p is None or q is None
                                     else measured_term(p, q))
            missing = None
            if p is None or q is None:
                missing = ('the measurement holds no autospectrum for '
                           f'{labels[r] if p is None else labels[c]}')
            elif theirs is None:
                missing = 'the measurement holds no cross term for the pair'
            if missing is not None:
                checks.append(CrossCheck(r, c, label, spec_index, None,
                                         'not measured', 'not measured',
                                         missing))
                continue
            # a term and its mirror are one statement; two that disagree
            # leave the pair unread rather than read from whichever
            # record the grid met first
            mirrors = []
            both = grid[r][c] is not None and grid[c][r] is not None
            if both and _disagrees(
                    record_at(specification, grid[r][c], lines),
                    record_at(specification, grid[c][r], lines)):
                mirrors.append('specification')
            a = where[_channel_of(measured, p)[0]]
            b = where[_channel_of(measured, q)[0]]
            held = (measured_grid[a][b], measured_grid[b][a])
            if None not in held and _disagrees(ordinate[held[0]],
                                               ordinate[held[1]]):
                mirrors.append('measurement')
            if mirrors:
                checks.append(CrossCheck(
                    r, c, label, spec_index, theirs, 'refused', 'refused',
                    f'the {" and the ".join(mirrors)} '
                    f'{"hold" if len(mirrors) > 1 else "holds"} the cross '
                    'term of the pair both ways round, and the two are not '
                    "each other's conjugate, so which is meant cannot be "
                    'read'))
                continue
            refusal, note = units_check(specification.ordinate_dim[spec_index],
                                        measured.ordinate_dim[theirs],
                                        wanted_name='the specified cross term',
                                        held_name='the measured one')
            # a coherence is the cross term over its two autospectra, so
            # the three are declared alike or the ratio is of numbers in
            # different scales
            for side, data, held in (
                    ('specified', specification,
                     (spec_index, diagonal[r], diagonal[c])),
                    ('measured', measured, (theirs, p, q))):
                declared = {str(data.ordinate_dim[i]) != UNKNOWN
                            for i in held}
                if refusal is None and len(declared) > 1:
                    refusal = (f'the {side} cross term and its autospectra '
                               'are not all declared in units: a coherence '
                               'of numbers in different scales is not one')
            if refusal is not None:
                checks.append(CrossCheck(r, c, label, spec_index, theirs,
                                         'refused', 'refused', refusal))
                continue
            if note is not None and note not in notes:
                notes.append(note)
            # the term of r against c: a record held the other way round
            # is the conjugate, its phase reversed
            spec_term = record_at(specification, spec_index, lines)
            spec_term = spec_term if flip > 0 else np.conj(spec_term)
            held_term = (ordinate[theirs] if measured_flip > 0
                         else np.conj(ordinate[theirs]))
            specified_coherence = ordinary_coherence(
                autos_specified[r], autos_specified[c], spec_term)
            measured_coherence = ordinary_coherence(
                ordinate[p], ordinate[q], held_term)
            with np.errstate(invalid='ignore'):
                specified_phase = np.degrees(np.angle(spec_term))
                measured_phase = np.degrees(np.angle(held_term))
            specified_phase[~np.isfinite(specified_coherence)] = np.nan
            measured_phase[~np.isfinite(measured_coherence)] = np.nan
            pair_notes = []
            asked = np.isfinite(specified_coherence)
            for name, record in ((label, theirs), (labels[r], p),
                                 (labels[c], q)):
                bad = int(np.count_nonzero(
                    ~np.isfinite(ordinate[record]) & asked))
                if bad:
                    pair_notes.append(_non_finite_note(name, bad))
            for name, record in ((labels[r], p), (labels[c], q)):
                values = np.real(ordinate[record])[asked]
                if (values.size and np.all(np.isfinite(values))
                        and not np.any(values > 0.0)):
                    pair_notes.append(f'the measured {name} recorded no '
                                      'power, so the pair has no coherence')
            averages = (None if counts is None
                        else float(np.asarray(counts)[theirs]) / 2.0)
            check = _judged(CrossCheck(
                r, c, label, spec_index, theirs, '', '', '',
                specified_coherence, measured_coherence, specified_phase,
                measured_phase, {}, tuple(pair_notes), averages),
                coherence_tolerance, phase_tolerance, coherence_floor)
            checks.append(check)
    return MatrixComparison(
        tuple(labels), tuple(levels), tuple(checks), lines, scale,
        coherence_tolerance, phase_tolerance, coherence_floor, tuple(notes))
