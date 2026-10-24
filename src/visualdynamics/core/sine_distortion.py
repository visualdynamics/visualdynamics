"""The sine distortion check: what is on a channel besides the tone.

A sine controller reads its control channel through a narrow band that
follows the drive (`core.sine_tracking`), so the level it holds is the
fundamental and nothing else. Whatever else the channel carries is
outside the band: harmonics from a rattle, a loose fixture bolt, a joint
that claps or an amplifier at its limit, a fixture resonance excited at a
multiple of the drive, noise. The controller does not see it, and the
article feels it. The check reads it directly. The tones the extraction
tracked are rebuilt on the record's own samples and taken away from it,
and what is left is everything that is not a tone. With r the RMS of
that residual and x_1 the RMS of the rebuilt tones, both over the same
span of the record, the distortion is

    D = r / x_1,

reported as a fraction (a percentage on screen) and in dB, 20 log10 D.

The tolerance is usually written on a ratio rather than on D: the
unfiltered RMS no more than some share above the filtered one. Where
the residual is orthogonal to the tones their mean squares add, so the
ratio is sqrt(1 + D**2) and D = sqrt(ratio**2 - 1) (`distortion`). Five
percent is an example of what test documents ask for, and it is the
default here (`DISTORTION_EXCESS`). Five percent over is a distortion of
sqrt(1.05**2 - 1) = 0.320, 32 % or -9.9 dB (`distortion_limit`), which is
the line the check draws and the figure a line is flagged above.

The residual is read directly because D taken from two readings, the
unfiltered RMS over the tone's, rests on two things that need not
hold. The tones must be orthogonal to each other for their powers to
add, and two tones sweeping through each other are not orthogonal where
they cross. And the two readings must cover the same stretch of the
record: a level read at an instant and an RMS read over a span differ
wherever the level moves inside the span, as it does through a sharp
resonance. The residual is read over the same
samples as the tones it is measured against, and two tones that beat or
cross are rebuilt together and subtracted together, so neither
assumption is made.

The check is not sensitive, and it is not meant to be. A tone with a
third harmonic at 10 % of it reads a ratio of 1.005, at 20 % 1.020, and
only at 32 % does it reach 1.05: it catches distortion large enough to
change the test. It hardly sees clipping at all. A sine clipped flat at
four fifths of its peak reads 1.004, because cutting the tops of a sine
puts most of the lost power back into the fundamental, so flat tops are
found in the time history and its crest factor, not here.

The tones are rebuilt by the estimator the extraction uses
(`core.sine.vold_kalman`): every tone the levels hold that plays over
the checked tone's stretch, solved jointly on the record's own samples
with the extraction's smoothing, each tone's complex envelope on its
own sweep argument. The envelope follows the level as it moves, a
resonance included, so the rebuilt tone is the tone the controller's
band would pass, sample by sample. Where the extraction corrected the
sweep clock (`SineLevel.drift_hz` not zero) the rebuild fits the same
kind of correction to its own envelope's phase and solves again, so a
drifted sweep is not rebuilt off its tone. The residual and the tones
are then read over `cycles` cycles of the tone's phase centered on each
line's instant (the extraction's smoothing by default), moved inward
where a span would run off either end of the tone's stretch.

The record's mean over the tone's stretch is taken out first: a sensor's
offset is not motion the article feels. Anything slower than the span
that is not constant is counted, as it should be, since it is not the
tone either. Content up to the Nyquist frequency is counted in full;
above it the record holds nothing, so a sweep whose third harmonic
passes the Nyquist frequency says so in the notes.

A line whose span reaches a tone of the specification the levels do not
hold cannot be judged, and neither can a line the extraction found under
its noise floor (`SineLevel.below_floor`), nor a line with no tone on
the channel to judge against (a dead channel), nor any line of a channel
whose stretch holds a sample that is not a finite number. All are kept,
counted in the notes and left out of the verdict
(`DistortionCheck.judged`), and a channel with no judged line does not
pass.

References
----------
The definition of the distortion and the estimator of the tone it is
read against are set out in these.

1. Department of Defense (2019). *MIL-STD-810H, Environmental
   Engineering Considerations and Laboratory Tests*. Method 514.8,
   Table 514.8-III, note 3: the distortion check on a sinusoidal test,
   the unfiltered RMS within 5 percent of the filtered RMS, stated as
   about 32 percent distortion.
2. Brandt, A. (2011). *Noise and Vibration Analysis: Signal Analysis and
   Experimental Procedures*. Wiley. Order tracking, the Vold-Kalman
   filter among it: a component on a known frequency history read as a
   slowly varying amplitude by least squares.
3. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The mean square of a sum of
   orthogonal components as the sum of their mean squares.
"""

# The first version read D from two readings, the unfiltered RMS over
# the tone's level; two clean tones crossing read near 100 % and so did
# the far side of a clean sweep through a sharp resonance (a Q of 50 at
# 20 Hz/s read 8 % where the residual reads under 1 %). The residual
# replaced it on review (Brandon, 2026-10-23).

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..units import UNKNOWN
from .data import _even_steps
from .sine import SineLevelSet, SineSweepSpecification, extract_sine, vold_kalman
from .sine_tracking import _laid

#: the share the unfiltered RMS may stand above the fundamental's
#: before a line is flagged: 5 %, an example of what test documents ask
#: for, which is a distortion of 32 % (`distortion_limit`)
DISTORTION_EXCESS = 0.05

#: the harmonic a note is written for when it climbs past the Nyquist
#: frequency: the third, the first a symmetric clip or a rattle puts on
#: a channel
HARMONIC = 3
HARMONIC_WORD = 'third'


def distortion(unfiltered: Any, fundamental: Any) -> np.ndarray:
    """The distortion from two RMS readings, where what is not the tone
    is orthogonal to it: sqrt((unfiltered / fundamental)**2 - 1), the RMS
    of everything that is not the tone over the RMS of the tone.

    This is the conversion between a tolerance written on the ratio and
    a distortion. The check itself reads the residual directly
    (`check_distortion`), since two tones that cross are not orthogonal
    and two readings over different stretches are not comparable.

    A ratio under one, which only the scatter of the two readings gives
    (the unfiltered record holds the tone and more), reads as no
    distortion rather than as the root of a negative number. A
    fundamental of zero reads as not a number.

    Parameters
    ----------
    unfiltered : float or array-like
        The channel's RMS, unfiltered.
    fundamental : float or array-like
        The tone's RMS, in the same units.

    Returns
    -------
    ndarray
        D as a fraction (0.32 is 32 %), the shape the two broadcast to.
    """
    unfiltered = np.asarray(unfiltered, dtype=float)
    fundamental = np.asarray(fundamental, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(fundamental > 0.0, unfiltered / fundamental, np.nan)
    return np.sqrt(np.maximum(ratio * ratio - 1.0, 0.0))


def distortion_limit(excess: float = DISTORTION_EXCESS) -> float:
    """The distortion a tolerance on the ratio allows: sqrt((1 +
    excess)**2 - 1), 0.320 for 5 %.

    Raises ValueError unless `excess` is more than zero.

    Parameters
    ----------
    excess : float, default `DISTORTION_EXCESS`
        The share the unfiltered RMS may stand above the fundamental's.

    Returns
    -------
    float
        D as a fraction.
    """
    excess = float(excess)
    if not excess > 0.0:
        raise ValueError(f'a tolerance of {excess:g} on the ratio must be '
                         'more than zero')
    return float(np.sqrt((1.0 + excess) ** 2 - 1.0))


def decibels(value: Any) -> np.ndarray:
    """A distortion in dB, 20 log10 D; minus infinity where it is zero.

    Parameters
    ----------
    value : float or array-like
        D as a fraction.

    Returns
    -------
    ndarray
        dB, the shape of `value`.
    """
    with np.errstate(divide='ignore'):
        return 20.0 * np.log10(np.asarray(value, dtype=float))


@dataclass(frozen=True, kw_only=True, eq=False)
class DistortionCheck:
    """One tone's distortion along its sweep, on each channel read: what
    `check_distortion` returns.

    `frequency` and `seconds` are the extraction's lines, ascending in
    frequency, each with the second it was read, from the recording's
    first sample. `unfiltered`, `fundamental` and `residual` are channels
    x lines of RMS in SI over `cycles` cycles about each line: the
    channel itself, the tones rebuilt on it, and the channel less the
    tones. `judged` is channels x lines, False where a line cannot be
    judged (the tone under its noise floor, a tone the levels do not
    hold live beside it, no tone on the channel, or a sample that is not
    a finite number). `excess` is the tolerance on the ratio. `notes`
    says what was not judged and why, empty when every line was. Two of
    these compare equal only if they are the same object, since the
    arrays have no single truth value.
    """

    tone: str
    dofs: list[str]
    dimensions: list[str]
    units: list[str | None]
    frequency: np.ndarray
    seconds: np.ndarray
    unfiltered: np.ndarray
    fundamental: np.ndarray
    residual: np.ndarray
    judged: np.ndarray
    excess: float
    cycles: float
    notes: tuple[str, ...] = field(default=())

    @property
    def distortion(self) -> np.ndarray:
        """Channels x lines: D as a fraction, the residual's RMS over the
        rebuilt tones'; not a number where there is no tone."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(self.fundamental > 0.0,
                            self.residual / self.fundamental, np.nan)

    @property
    def ratio(self) -> np.ndarray:
        """Channels x lines: the ratio the distortion stands for,
        sqrt(1 + D**2), the unfiltered RMS over the tone's where the rest
        is orthogonal to the tones; the number a tolerance like 'within
        5 %' is written on."""
        return np.sqrt(1.0 + self.distortion ** 2)

    @property
    def decibels(self) -> np.ndarray:
        """Channels x lines: D in dB (`decibels`)."""
        return decibels(self.distortion)

    @property
    def limit(self) -> float:
        """The distortion the tolerance allows, as a fraction
        (`distortion_limit`)."""
        return distortion_limit(self.excess)

    @property
    def over(self) -> np.ndarray:
        """Channels x lines, True where a judged line's distortion is
        above the limit."""
        with np.errstate(invalid='ignore'):
            return self.judged & (self.distortion > self.limit)

    @property
    def passed(self) -> bool:
        """Whether every channel read has a judged line and no judged
        line on any channel is over the limit. A channel with nothing
        judged (dead, not a number, or every line unjudged) has not
        passed: there is no reading to pass."""
        if not self.judged.any(axis=1).all():
            return False
        return not bool(self.over.any())

    def row(self, dof: str | None = None) -> int:
        """The row of channel `dof`; the first when None. Raises
        ValueError when the check did not read it."""
        if dof is None:
            return 0
        if str(dof) not in self.dofs:
            raise ValueError(f'{dof} was not read; the check holds '
                             f'{self.dofs}')
        return self.dofs.index(str(dof))

    def worst(self, dof: str | None = None) -> tuple[float, float] | None:
        """The largest judged distortion and the frequency it is at, on
        channel `dof` or, when None, on any channel: (Hz, D as a
        fraction), or None when no line was judged.

        Parameters
        ----------
        dof : str, optional
            The channel; every channel read when omitted.

        Returns
        -------
        tuple of float, or None
            (frequency in Hz, distortion as a fraction).
        """
        rows = (range(len(self.dofs)) if dof is None else [self.row(dof)])
        best = None
        values = self.distortion
        for r in rows:
            seen = self.judged[r] & np.isfinite(values[r])
            if not seen.any():
                continue
            k = int(np.flatnonzero(seen)[np.argmax(values[r][seen])])
            if best is None or values[r, k] > best[1]:
                best = (float(self.frequency[k]), float(values[r, k]))
        return best

    def describe(self, dof: str | None = None) -> str:
        """The verdict in one line, such as 'Up at 101Z+: worst 12.3 %
        (-18.2 dB) at 412 Hz, within the 32.0 % limit'. Every channel's
        worst when `dof` is None, naming any channel with no line
        judged."""
        where = self.tone + ('' if dof is None else f' at {dof}')
        worst = self.worst(dof)
        if worst is None:
            return f'{where}: no line could be judged'
        rows = (range(len(self.dofs)) if dof is None else [self.row(dof)])
        over = int(sum(self.over[r].sum() for r in rows))
        judged = int(sum(self.judged[r].sum() for r in rows))
        frequency, value = worst
        words = (f'{where}: worst {100.0 * value:.1f} % '
                 f'({float(decibels(value)):.1f} dB) at {frequency:.4g} Hz, ')
        if over:
            words += (f'{over} of {judged} lines over the '
                      f'{100.0 * self.limit:.1f} % limit')
        else:
            words += f'within the {100.0 * self.limit:.1f} % limit'
        silent = [self.dofs[r] for r in rows if not self.judged[r].any()]
        if silent:
            words += f'; no line judged on {", ".join(silent)}'
        return words


def _live_windows(specification: SineSweepSpecification, chosen: Any,
                  onset: float) -> dict[str, tuple[float, float]]:
    """Each tone's span in recording seconds, placed on the clock the
    read tone was found on: the environment starts at its onset less its
    own start time, and every tone starts its own start time after that."""
    origin = float(onset) - chosen.start_time
    return {tone.name: (origin + tone.start_time,
                        origin + tone.start_time + tone.duration())
            for tone in specification.tones}


def _span(phase: np.ndarray, k: int, cycles: float) -> tuple[int, float]:
    """The span of exactly `cycles` cycles of `phase` centered on sample
    `k`, moved inward where it would run off either end: (its last
    sample, the phase it starts at)."""
    turn = 2.0 * np.pi * cycles
    last = len(phase) - 1
    end = min(int(np.searchsorted(phase, phase[k] + turn / 2.0)), last)
    if phase[end] - turn < phase[0]:
        end = min(int(np.searchsorted(phase, phase[0] + turn)), last)
    return end, float(max(phase[end] - turn, phase[0]))


def _overlap(t0: np.ndarray, t1: np.ndarray, lo: float,
             hi: float) -> np.ndarray:
    """The share of each span from `t0` to `t1` that lies between `lo`
    and `hi`."""
    inside = np.clip(np.minimum(t1, hi) - np.maximum(t0, lo), 0.0, None)
    return inside / np.maximum(t1 - t0, 1e-300)


def _row_of(history: Any, dof: str, dimension: str) -> int:
    """The record of `history` at `dof` measuring `dimension`, the
    quantity its levels were read in: the first such, as the extraction
    takes the first record of a control DOF. A DOF can carry an
    accelerometer and a force gauge both, and a check of one against the
    other's levels would measure the wrong channel against the tone."""
    rows = [r for r, name in enumerate(history.response_dof)
            if str(name) == dof]
    if not rows:
        raise ValueError(f'the recording carries no record for {dof}')
    if dimension == UNKNOWN:
        return rows[0]
    same = [r for r in rows if history.ordinate_dim[r] == dimension]
    if not same:
        found = sorted({str(history.ordinate_dim[r]) for r in rows})
        raise ValueError(
            f'the levels at {dof} were read in {dimension} and the '
            f'recording carries {", ".join(found)} there: they are not of '
            'this record')
    return same[0]


#: how far, in radians, a tone's argument may stand from a quadratic in
#: time and still have its envelope's turning fitted as one: a tone off
#: its schedule by any share under one turns its envelope from the
#: quadratic by less than this
_QUADRATIC_ARGUMENT = 1e-3


def _unwound(argument: np.ndarray, envelope: np.ndarray, dt: float,
             frequency: np.ndarray) -> np.ndarray:
    """`argument` with the slow phase its envelope turned through taken
    into it, fitted to the envelope's unwrapped phase weighted by its
    power. Solved again on the corrected argument the envelope sits
    still, so the filter passes the tone in full.

    A tone whose argument is a quadratic in time (a linear sweep, a
    dwell) has the turning fitted as a quadratic in time, the kind of
    clock correction the extraction applies. Any other (a log sweep, a
    sweep of segments at different rates) has it fitted on its own
    argument and angular frequency, `frequency` in Hz at each sample: a
    tone run at s times its schedule turns its envelope by (s - 1) times
    the argument, and one late by d seconds by d times the angular
    frequency, whatever the law. A quadratic in time misses a log
    sweep's departure, and near the start of the stretch the part it
    misses is read through the filter's band."""
    weight = np.abs(envelope) ** 2
    if not weight.sum() > 0.0 or len(argument) < 3:
        return argument
    u = (np.arange(len(argument)) - len(argument) / 2.0) * dt
    phase = np.unwrap(np.angle(envelope))
    root = np.sqrt(weight)
    quadratic = np.stack([np.ones_like(u), u, u * u], axis=1)
    shape, *_rest = np.linalg.lstsq(quadratic, argument, rcond=None)
    if np.max(np.abs(argument - quadratic @ shape)) <= _QUADRATIC_ARGUMENT:
        basis = quadratic * root[:, None]
        coefficients, *_rest = np.linalg.lstsq(basis, phase * root,
                                               rcond=None)
        return argument + coefficients[1] * u + coefficients[2] * u * u
    # each column scaled to one, so the cut-off below reads the two as
    # shapes: on a log sweep the argument is the angular frequency times
    # a constant less a constant, the two are one column, and lstsq takes
    # the least-norm answer, whose fitted turning is the same
    middle = len(argument) // 2
    columns = [argument - argument[middle],
               2.0 * np.pi * (frequency - frequency[middle])]
    columns = [c / np.max(np.abs(c)) for c in columns]
    basis = np.stack([np.ones_like(u)] + columns, axis=1)
    coefficients, *_rest = np.linalg.lstsq(basis * root[:, None],
                                           phase * root, rcond=1e-9)
    return argument + basis[:, 1:] @ coefficients[1:]


def _mean_squares(values: np.ndarray, phase: np.ndarray,
                  ends: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """The mean of `values` squared over each span of the tone's phase,
    from the phase `starts` to the sample `ends`: the trapezoid over
    phase from one running sum, the span's first piece interpolated
    where it starts between two samples, divided by the phase covered.

    Over phase rather than samples, so a span is exactly its cycles of
    the tone wherever they fall between samples: counted in whole
    samples, ten cycles at forty samples a cycle read a sine's mean
    square up to a quarter of a percent off, more at fewer samples, and
    differently for the tone and for what is not the tone.
    """
    square = values * values
    running = np.concatenate(([0.0], np.cumsum(
        0.5 * (square[1:] + square[:-1]) * np.diff(phase))))
    at = np.clip(np.searchsorted(phase, starts), 1, len(phase) - 1)
    width = phase[at] - phase[at - 1]
    share = np.where(width > 0.0, (phase[at] - starts) / width, 0.0)
    begin = square[at] - share * (square[at] - square[at - 1])
    piece = 0.5 * (begin + square[at]) * (phase[at] - starts)
    covered = phase[ends] - starts
    return (running[ends] - running[at] + piece) / covered


@dataclass(frozen=True, kw_only=True, eq=False)
class _Lines:
    """What a reading of one tone at its extraction's lines works from,
    laid on the recording: the level and the tone read, the channels and
    their rows, the tone's stretch of the record and its phase, the span
    of `cycles` cycles about each line, which lines reach a tone the
    levels do not hold, and every tone the levels hold over the stretch,
    laid on the record's samples (the read tone first) as (phase,
    frequency, offset into the stretch, whether its clock drifted). The
    distortion check and the frequency check (`core.sine_frequency`)
    read the same lines of the same tones, so they are laid here once."""

    dt: float
    level: Any
    chosen: Any
    cycles: float
    smoothing: float
    wanted: list[str]
    picks: list[int]
    rows: list[int]
    first: int
    last: int
    frequency: np.ndarray
    phase: np.ndarray
    seconds: np.ndarray
    ends: np.ndarray
    starts: np.ndarray
    shared: np.ndarray
    laid: list[tuple[np.ndarray, np.ndarray, int, bool]]


def _tone_lines(history: Any, specification: SineSweepSpecification,
                levels: SineLevelSet | None, *, tone: str | None,
                channels: Sequence[str] | None, cycles: float | None,
                doing: str, reading: str) -> _Lines:
    """Lay one tone's lines on the recording for a reading over `cycles`
    cycles about each (`_Lines`), extracting the levels first when none
    are given, as the project verb would. `doing` names the reading in an
    uneven record's refusal and `reading` what is read over the span, in
    the refusals that name it. Raises ValueError for every refusal
    `check_distortion` lists but the tolerance's."""
    dt = _even_steps(np.asarray(history.abscissa, dtype=float), doing)
    if levels is None:
        setting = (history.sine_extraction
                   or history.suggest_sine_extraction())
        levels = extract_sine(history, specification, cycles=setting.cycles,
                              refine=setting.refine,
                              target_db=setting.target_db)
    if not len(levels):
        raise ValueError('the levels hold no tone; extract them first')
    if tone is not None and tone not in levels.tone_names:
        raise ValueError(f'the levels hold no tone {tone!r}; they hold '
                         f'{levels.tone_names}')
    level = levels.levels[0] if tone is None else levels.tone(tone)
    chosen = specification.tone(level.tone)
    if cycles is None:
        cycles = levels.cycles
    if cycles is None:
        raise ValueError('these levels do not say what smoothing they '
                         'were read with; give cycles=')
    cycles = float(cycles)
    if not cycles > 0.0:
        raise ValueError(f'{reading} is read over more than zero cycles')
    smoothing = float(levels.cycles if levels.cycles is not None
                      else cycles)
    held = [str(dof) for dof in level.response_dof]
    wanted = held if channels is None else [str(dof) for dof in channels]
    missing = [dof for dof in wanted if dof not in held]
    if missing:
        raise ValueError(f'the levels hold no reading for {missing}; they '
                         f'hold {held}')
    picks = [held.index(dof) for dof in wanted]
    level_dims = list(level.ordinate_dim)
    rows = [_row_of(history, dof, str(level_dims[p]))
            for dof, p in zip(wanted, picks, strict=True)]

    samples = history.ordinate.shape[1]
    first, last, frequency, phase = _laid(history, chosen, level.onset, dt)
    if phase[-1] - phase[0] < 2.0 * np.pi * cycles:
        raise ValueError(
            f'{chosen.name}: the recording holds '
            f'{(phase[-1] - phase[0]) / (2.0 * np.pi):.3g} cycles of the '
            f'tone, less than the {cycles:g} {reading} is read over')
    if level.seconds is None:
        raise ValueError(
            f'{chosen.name}: the levels do not say when each line was '
            'read, so there is no span of the recording to check. Extract '
            'them from the recording again')
    seconds = np.asarray(level.seconds, dtype=float)
    at = np.round(seconds / dt).astype(np.int64)
    if len(at) and (at.min() < first or at.max() >= min(last, samples)):
        raise ValueError(
            f'{chosen.name}: the levels were read at times from '
            f'{seconds.min():.4g} to {seconds.max():.4g} s, outside the '
            f"tone's stretch of this recording; they are of another "
            'record, or of this one before it changed. Extract them again')
    lines = np.clip(at - first, 0, last - first - 1)
    spans = [_span(phase, int(k), cycles) for k in lines]
    ends = np.array([end for end, _start in spans], dtype=np.int64)
    starts = np.array([start for _end, start in spans], dtype=float)
    # each span's ends in recording seconds, for the other tones' share
    t0 = (first + np.interp(starts, phase, np.arange(len(phase)))) * dt
    t1 = (first + ends) * dt

    # a tone of the specification the levels do not hold cannot be
    # rebuilt, so a span reaching it at all is not judged
    windows = _live_windows(specification, chosen, level.onset)
    extracted = set(levels.tone_names)
    shared = np.zeros(len(seconds), dtype=bool)
    for name, (lo, hi) in windows.items():
        if name not in extracted:
            shared |= _overlap(t0, t1, lo, hi) > 0.0

    # every tone the levels hold over the checked tone's stretch, laid on
    # the record's samples from its own onset, the checked tone first
    laid = []
    for other in [level] + [o for o in levels.levels if o is not level]:
        if other is level:
            o_first, o_last, o_frequency, o_phase = (first, last,
                                                     frequency, phase)
        else:
            try:
                o_first, o_last, o_frequency, o_phase = _laid(
                    history, specification.tone(other.tone), other.onset, dt)
            except ValueError:
                continue                       # none of it on the record
        lo, hi = max(o_first, first), min(o_last, last)
        if hi - lo < 2:
            continue
        laid.append((o_phase[lo - o_first:hi - o_first],
                     o_frequency[lo - o_first:hi - o_first], lo - first,
                     bool(other.drift_hz)))
    return _Lines(dt=dt, level=level, chosen=chosen, cycles=cycles,
                  smoothing=smoothing, wanted=wanted, picks=picks, rows=rows,
                  first=first, last=last, frequency=frequency, phase=phase,
                  seconds=seconds, ends=ends, starts=starts, shared=shared,
                  laid=laid)


def _rebuilt(x: np.ndarray, laid: Sequence[tuple], dt: float,
             smoothing: float, unwind: Sequence[bool]
             ) -> tuple[list[np.ndarray], list[int], list[np.ndarray]]:
    """Every laid tone solved jointly on one channel's stretch `x` by the
    extraction's estimator with its smoothing: (the arguments the
    envelopes sit on, their offsets into the stretch, the complex
    envelopes). Each tone `unwind` marks has the slow phase its envelope
    turned through fitted and taken into its argument (`_unwound`), and
    the solve is repeated: the distortion check does it for a tone whose
    clock the extraction moved, the frequency check for every tone, so a
    tone off its specified frequency is held inside the filter's band."""
    arguments = [argument for argument, _f, _o, _d in laid]
    frequencies = [f for _a, f, _o, _d in laid]
    offsets = [offset for _a, _f, offset, _d in laid]
    envelopes = vold_kalman(x, arguments, frequencies, offsets, dt, smoothing)
    if any(unwind):
        arguments = [_unwound(argument, envelope, dt, f) if moved
                     else argument
                     for argument, envelope, f, moved
                     in zip(arguments, envelopes, frequencies, unwind,
                            strict=True)]
        envelopes = vold_kalman(x, arguments, frequencies, offsets, dt,
                                smoothing)
    return arguments, offsets, envelopes


def check_distortion(history: Any, specification: SineSweepSpecification,
                     levels: SineLevelSet | None = None, *,
                     tone: str | None = None,
                     channels: Sequence[str] | None = None,
                     excess: float = DISTORTION_EXCESS,
                     cycles: float | None = None) -> DistortionCheck:
    """Check one tone's distortion along its sweep: the RMS of the record
    less the tracked tones against the RMS of the tones, at every line of
    the extraction.

    The tones come from `levels`, a `SineLevelSet` that `extract_sine`
    (Extract Sine Levels) read from this recording against this
    specification; with none, the extraction is run here with the
    recording's own `sine_extraction` setting (or its suggestion), as the
    project verb runs it. Over the checked tone's stretch, every tone the
    levels hold is rebuilt on each channel's samples by the joint
    Vold-Kalman solve with the extraction's smoothing, the record's mean
    taken out, and the rebuilt tones subtracted. At each line the residual
    and the rebuilt tones are read over `cycles` cycles of the tone
    centered on the line's instant, and the distortion is the one RMS
    over the other. A line is flagged where it is above the limit
    `excess` sets (`distortion_limit`).

    Raises ValueError when `excess` or `cycles` is not more than zero, when
    the recording is not evenly sampled, when the levels hold no such tone
    or channel, when the recording carries no record of a channel in the
    quantity its levels were read in, when the levels do not say when
    each line was read or were read at times the recording does not
    hold, or when the recording holds less of the
    tone than one span.

    Parameters
    ----------
    history : TimeHistory
        The recording, evenly sampled.
    specification : SineSweepSpecification
        The sweep the drive followed.
    levels : SineLevelSet, optional
        The extracted levels; extracted here when omitted.
    tone : str, optional
        Which tone, by name; the first the levels hold by default.
    channels : sequence of str, optional
        Which control channels; every channel the levels hold by default.
    excess : float, default `DISTORTION_EXCESS`
        The share the unfiltered RMS may stand above the fundamental's;
        5 %, a distortion of 32 %, by default.
    cycles : float, optional
        The span the RMS readings are taken over, in cycles of the tone;
        the extraction's smoothing by default.

    Returns
    -------
    DistortionCheck
        The three RMS readings at every line on every channel read, which
        lines could be judged, and the tolerance.
    """
    distortion_limit(excess)                  # refuses a bad tolerance
    lines = _tone_lines(history, specification, levels, tone=tone,
                        channels=channels, cycles=cycles,
                        doing='Checking sine distortion', reading='the RMS')
    level, chosen, cycles = lines.level, lines.chosen, lines.cycles
    wanted, picks, rows = lines.wanted, lines.picks, lines.rows
    first, last, phase = lines.first, lines.last, lines.phase
    ends, starts, seconds = lines.ends, lines.starts, lines.seconds
    shared, laid = lines.shared, lines.laid
    dt = lines.dt

    count = len(seconds)
    unfiltered = np.full((len(rows), count), np.nan)
    fundamental = np.full((len(rows), count), np.nan)
    residual = np.full((len(rows), count), np.nan)
    notes = []
    for c, row in enumerate(rows):
        x = np.real(np.asarray(history.ordinate[row, first:last],
                               dtype=float))
        bad = int((~np.isfinite(x)).sum())
        if bad:
            notes.append(f'{wanted[c]}: {bad} sample{"s" * (bad != 1)} over '
                         "the tone's stretch "
                         f'{"are" if bad != 1 else "is"} not a finite '
                         'number, so no line is judged')
            continue
        x = x - np.mean(x)
        arguments, offsets, envelopes = _rebuilt(
            x, laid, dt, lines.smoothing,
            [drifted for *_rest, drifted in laid])
        tones = np.zeros(len(x))
        for argument, offset, envelope in zip(arguments, offsets,
                                              envelopes, strict=True):
            tones[offset:offset + len(argument)] += np.real(
                envelope * np.exp(1j * argument))
        unfiltered[c] = np.sqrt(_mean_squares(x, phase, ends, starts))
        fundamental[c] = np.sqrt(_mean_squares(tones, phase, ends, starts))
        residual[c] = np.sqrt(np.maximum(
            _mean_squares(x - tones, phase, ends, starts), 0.0))

    floor = (np.zeros((len(rows), count), dtype=bool)
             if level.below_floor is None
             else np.asarray(level.below_floor, dtype=bool)[picks])
    toneless = np.isfinite(fundamental) & ~(fundamental > 0.0)
    judged = (~floor & ~shared[None, :] & np.isfinite(residual)
              & (fundamental > 0.0))
    if floor.any():
        notes.append(f'{int(floor.sum())} line{"s" * int(floor.sum() != 1)} '
                     'under the noise floor, not judged')
    if shared.any():
        notes.append(f'{int(shared.sum())} line{"s" * int(shared.sum() != 1)} '
                     'shared with a tone the levels do not hold, not judged')
    for c in range(len(rows)):
        none = int(toneless[c].sum())
        if none:
            notes.append(f'{wanted[c]}: {none} line{"s" * (none != 1)} '
                         'with no tone on the channel to judge against, '
                         'not judged')
    reach = 0.5 / dt / HARMONIC
    if float(np.max(level.abscissa)) > reach:
        notes.append(f'above {reach:.4g} Hz the {HARMONIC_WORD} harmonic is '
                     'past the Nyquist frequency, so the record cannot hold '
                     'it and the check cannot see it')
    return DistortionCheck(
        tone=chosen.name, dofs=wanted,
        dimensions=[history.ordinate_dim[row] for row in rows],
        units=[history.ordinate_unit[row] for row in rows],
        frequency=np.asarray(level.abscissa, dtype=float), seconds=seconds,
        unfiltered=unfiltered, fundamental=fundamental, residual=residual,
        judged=judged, excess=float(excess), cycles=cycles,
        notes=tuple(notes))


__all__ = ['DISTORTION_EXCESS', 'DistortionCheck', 'check_distortion',
           'decibels', 'distortion', 'distortion_limit']
