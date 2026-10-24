"""The sine frequency check: the tone's frequency on the channel against
the frequency the profile asks for, along the sweep.

A sine test is held to its frequency as well as its level. Test
documents write the band as a share of the frequency; MIL-STD-810H holds
it to plus or minus 0.1 percent. The default here is 2 percent
(`FREQUENCY_TOLERANCE`), for the governing document's own figure to
replace, and the band can be given a floor in hertz (`floor`) for a
document that writes one. At a sharp resonance the frequency band is also
a level band: a tone held 2 percent off a mode with a Q of 40 drives it
to about half the response it would at the center, since the one-mass
response falls as 1 / sqrt(1 + (2 Q d)**2) for a fractional offset d.

The check reads the tone's frequency off the record. The tones the
extraction tracked are rebuilt on the record's own samples by the
estimator the extraction uses (`core.sine.vold_kalman`), each a complex
envelope riding on its sweep's argument, the phase the specification
gives at each sample. A tone off its specified frequency turns its
envelope: the envelope's phase gains 2 pi times the offset every second.
So the tone's own phase on the record is its argument plus its
envelope's unwrapped phase, and its frequency over a span is that phase's
advance over the span divided by 2 pi times the span's length. With
psi the tone's phase and phi the specification's,

    f_tone = (psi(t1) - psi(t0)) / (2 pi (t1 - t0)),
    f_spec = (phi(t1) - phi(t0)) / (2 pi (t1 - t0)),

read over `cycles` cycles of the tone centered on each line of the
extraction (its smoothing by default), the same spans the distortion
check reads (`core.sine_distortion`). The error is f_tone - f_spec, in
hertz and as a share of f_spec, and a line is flagged where it is wider
than the band: the tolerance times f_spec, or the floor if that is
wider. Before the phase is read each envelope's slow turning is fitted
and taken into its argument and the tone solved again, as the extraction
corrects a drifted sweep clock, so a tone well off its frequency is held
inside the estimator's band rather than read through its skirt; the
frequency read is the same either way, since the correction is added to
the argument and taken out of the envelope. On a linear sweep or a dwell
the turning is fitted as a quadratic in time. On any other sweep, a log
sweep among them, it is fitted on the sweep's own argument and angular
frequency, which is how a tone run at a share off its schedule, or late
on it, turns its envelope whatever the law.

What the check reads, and what it does not:

- The specified frequency is the specification's, on the clock the
  extraction found the sweep on (`SineLevel.onset`). A sweep is a
  schedule of frequency against time, so a sweep run at the wrong rate
  arrives at each frequency early or late, and the check reads that as a
  frequency error that grows along the sweep. A rate within its own
  tolerance can carry a frequency outside a tight band by the end of a
  long sweep; the sweep's rate is read by `core.sweep_rate`, and a
  dwell, held at one frequency, is where the frequency band is read most
  directly.
- The record's own phase is read, and a channel's phase moves when the
  sweep passes a resonance of what it sits on: through a mode of quality
  factor Q at natural frequency f_n, swept at r Hz/s, the phase turns at
  up to 2 Q r / f_n radians per second, which reads as an offset of
  Q r / (pi f_n) Hz. At 84 Hz, a Q of 21 and 2 oct/min (1.9 Hz/s) that
  is 0.15 Hz, 0.18 percent. A control channel rarely sits on a sharp
  mode, and where it does the offset is real: it is the frequency the
  channel carried.
- A line under the noise floor (`SineLevel.below_floor`), a line whose
  span reaches a tone of the specification the levels do not hold, a
  line with no tone on the channel (a dead channel) and every line of a
  channel whose stretch holds a sample that is not a finite number are
  not judged; the notes count them and a channel with nothing judged
  does not pass.

References
----------
The estimator of the tone and the frequency band it is judged against
are set out in these.

1. Brandt, A. (2011). *Noise and Vibration Analysis: Signal Analysis and
   Experimental Procedures*. Wiley. Order tracking, the Vold-Kalman
   filter among it: a component on a known frequency history read as a
   slowly varying complex amplitude by least squares, whose phase
   carries the component's departure from that history.
2. Department of Defense (2019). *MIL-STD-810H, Environmental
   Engineering Considerations and Laboratory Tests*. Method 514.8,
   Table 514.8-III: the frequency of a sinusoidal test held to plus or
   minus 0.1 percent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .sine import SineLevelSet, SineSweepSpecification
from .sine_distortion import _rebuilt, _tone_lines

#: the share of the specified frequency the tone may stand from it before
#: a line is flagged: 2 %, an example of what test documents ask for
FREQUENCY_TOLERANCE = 0.02


def _band(tolerance: float, floor: float) -> tuple[float, float]:
    """The tolerance and floor, refused unless they are finite numbers,
    the tolerance more than zero and the floor zero or more."""
    tolerance, floor = float(tolerance), float(floor)
    if not (np.isfinite(tolerance) and tolerance > 0.0):
        raise ValueError(f'a frequency tolerance of {tolerance!r}: it is a '
                         'share of the frequency, more than zero')
    if not (np.isfinite(floor) and floor >= 0.0):
        raise ValueError(f'a floor of {floor!r} Hz: it is zero or more')
    return tolerance, floor


def allowed(frequency: Any, tolerance: float = FREQUENCY_TOLERANCE,
            floor: float = 0.0) -> np.ndarray:
    """The frequency band at each specified frequency, Hz either side:
    `tolerance` times the frequency, or `floor` where that is wider ('2 %
    or 1 Hz, whichever is greater' is tolerance 0.02, floor 1.0).

    Raises ValueError unless `tolerance` is a finite number more than
    zero and `floor` one of zero or more.

    Parameters
    ----------
    frequency : float or array-like
        Hz.
    tolerance : float, default `FREQUENCY_TOLERANCE`
        A share of the frequency.
    floor : float, default 0.0
        Hz.

    Returns
    -------
    ndarray
        Hz, the shape of `frequency`.
    """
    tolerance, floor = _band(tolerance, floor)
    return np.maximum(tolerance * np.abs(np.asarray(frequency, dtype=float)),
                      floor)


@dataclass(frozen=True, kw_only=True, eq=False)
class FrequencyCheck:
    """One tone's frequency along its sweep, on each channel read: what
    `check_frequency` returns.

    `frequency` and `seconds` are the extraction's lines, ascending in
    frequency, each with the second it was read from the recording's
    first sample. `specified` is the specification's frequency over each
    line's span, Hz, and `measured` channels x lines of the tone's own
    frequency over the same span, not a number where it could not be
    read. `judged` is channels x lines, False where a line cannot be
    judged (under the noise floor, beside a tone the levels do not hold,
    no tone on the channel, or a sample that is not a finite number).
    `tolerance` and `floor` set the band (`allowed`). `notes` says what
    was not judged and why. Two of these compare equal only if they are
    the same object, since the arrays have no single truth value.
    """

    tone: str
    dofs: list[str]
    frequency: np.ndarray
    seconds: np.ndarray
    specified: np.ndarray
    measured: np.ndarray
    judged: np.ndarray
    tolerance: float
    floor: float
    cycles: float
    notes: tuple[str, ...] = field(default=())

    @property
    def error(self) -> np.ndarray:
        """Channels x lines: the tone's frequency less the specified,
        Hz."""
        return self.measured - self.specified[None, :]

    @property
    def relative(self) -> np.ndarray:
        """Channels x lines: the error as a share of the specified
        frequency (0.02 is 2 %)."""
        return self.error / self.specified[None, :]

    @property
    def allowed(self) -> np.ndarray:
        """Lines: the band either side, Hz (`allowed`)."""
        return allowed(self.specified, self.tolerance, self.floor)

    @property
    def share(self) -> np.ndarray:
        """Channels x lines: the error as a share of the band, its size
        over the band's half-width; over one is outside."""
        return np.abs(self.error) / self.allowed[None, :]

    @property
    def over(self) -> np.ndarray:
        """Channels x lines, True where a judged line's error is wider
        than the band."""
        with np.errstate(invalid='ignore'):
            return self.judged & (self.share > 1.0)

    @property
    def passed(self) -> bool:
        """Whether every channel read has a judged line and no judged
        line is outside the band. A channel with nothing judged has not
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

    def _rows(self, dof: str | None) -> list[int]:
        return list(range(len(self.dofs))) if dof is None else [self.row(dof)]

    def worst(self, dof: str | None = None
              ) -> tuple[float, float, float] | None:
        """The judged line whose error takes the most of its band, on
        channel `dof` or, when None, on any channel: (specified Hz, error
        Hz, error as a share of the frequency), or None when no line was
        judged."""
        best, most = None, -1.0
        share = self.share
        for r in self._rows(dof):
            seen = self.judged[r] & np.isfinite(share[r])
            if not seen.any():
                continue
            k = int(np.flatnonzero(seen)[np.argmax(share[r][seen])])
            if share[r, k] > most:
                most = float(share[r, k])
                best = (float(self.specified[k]), float(self.error[r, k]),
                        float(self.relative[r, k]))
        return best

    def band_words(self) -> str:
        """The band in words: '±2 %' or '±2 % or 1 Hz, whichever is
        greater'."""
        words = f'±{100.0 * self.tolerance:g} %'
        if self.floor > 0.0:
            words += f' or {self.floor:g} Hz, whichever is greater'
        return words

    def describe(self, dof: str | None = None) -> str:
        """The verdict in one line, such as 'Up at 101Z+: worst +1.52 %
        (+1.52 Hz) at 100 Hz, 40 of 40 lines outside ±1.25 %'."""
        where = self.tone + ('' if dof is None else f' at {dof}')
        worst = self.worst(dof)
        if worst is None:
            return f'{where}: no line could be judged'
        rows = self._rows(dof)
        over = int(sum(self.over[r].sum() for r in rows))
        judged = int(sum(self.judged[r].sum() for r in rows))
        frequency, error, relative = worst
        words = (f'{where}: worst {100.0 * relative:+.3g} % '
                 f'({error:+.3g} Hz) at {frequency:.4g} Hz, ')
        if over:
            words += f'{over} of {judged} lines outside {self.band_words()}'
        else:
            words += f'within {self.band_words()}'
        silent = [self.dofs[r] for r in rows if not self.judged[r].any()]
        if silent:
            words += f'; no line judged on {", ".join(silent)}'
        return words


def _phase_at(phase: np.ndarray, position: np.ndarray) -> np.ndarray:
    """`phase` read at fractional sample positions, linearly between
    samples."""
    return np.interp(position, np.arange(len(phase)), phase)


def check_frequency(history: Any, specification: SineSweepSpecification,
                    levels: SineLevelSet | None = None, *,
                    tone: str | None = None,
                    channels: Sequence[str] | None = None,
                    tolerance: float = FREQUENCY_TOLERANCE,
                    floor: float = 0.0,
                    cycles: float | None = None) -> FrequencyCheck:
    """Check one tone's frequency along its sweep: the tone's own
    frequency on each channel against the specification's, at every line
    of the extraction.

    The tones come from `levels`, a `SineLevelSet` that `extract_sine`
    read from this recording against this specification; with none, the
    extraction is run here with the recording's own `sine_extraction`
    setting (or its suggestion), as the project verb runs it. Every tone
    the levels hold is rebuilt on each channel's stretch by the joint
    Vold-Kalman solve with the extraction's smoothing, each envelope's
    slow turning fitted into its argument and the solve repeated, and the
    checked tone's phase (argument plus envelope phase) is read over
    `cycles` cycles about each line. A line is flagged where the error is
    wider than the band (`allowed`).

    Raises ValueError when `tolerance` is not more than zero, `floor` is
    under zero, `cycles` is not more than zero, the recording is not
    evenly sampled, the levels hold no such tone or channel, the
    recording carries no record of a channel in the quantity its levels
    were read in, the levels do not say when each line was read or were
    read at times the recording does not hold, or the recording holds
    less of the tone than one span.

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
    tolerance : float, default `FREQUENCY_TOLERANCE`
        The band, a share of the specified frequency either side.
    floor : float, default 0.0
        The band's least half-width, Hz.
    cycles : float, optional
        The span each frequency is read over, in cycles of the tone; the
        extraction's smoothing by default.

    Returns
    -------
    FrequencyCheck
        The specified and measured frequency at every line on every
        channel read, which lines could be judged, and the band.
    """
    tolerance, floor = _band(tolerance, floor)
    lines = _tone_lines(history, specification, levels, tone=tone,
                        channels=channels, cycles=cycles,
                        doing='Checking sine frequency',
                        reading='the frequency')
    dt, phase, ends, starts = lines.dt, lines.phase, lines.ends, lines.starts
    first, last = lines.first, lines.last
    # where each span starts, as a fractional sample of the stretch, and
    # how long it is: the same for the tone and for the specification
    begin = np.interp(starts, phase, np.arange(len(phase)))
    seconds_spanned = (ends - begin) * dt
    specified = (phase[ends] - starts) / (2.0 * np.pi * seconds_spanned)

    count = len(lines.seconds)
    measured = np.full((len(lines.rows), count), np.nan)
    strength = np.full((len(lines.rows), count), np.nan)
    notes = []
    for c, row in enumerate(lines.rows):
        x = np.real(np.asarray(history.ordinate[row, first:last],
                               dtype=float))
        bad = int((~np.isfinite(x)).sum())
        if bad:
            notes.append(f'{lines.wanted[c]}: {bad} sample{"s" * (bad != 1)} '
                         "over the tone's stretch "
                         f'{"are" if bad != 1 else "is"} not a finite '
                         'number, so no line is judged')
            continue
        x = x - np.mean(x)
        arguments, _offsets, envelopes = _rebuilt(
            x, lines.laid, dt, lines.smoothing, [True] * len(lines.laid))
        # the checked tone is laid first, over the whole stretch
        envelope = envelopes[0]
        psi = arguments[0] + np.unwrap(np.angle(envelope))
        measured[c] = ((psi[ends] - _phase_at(psi, begin))
                       / (2.0 * np.pi * seconds_spanned))
        power = np.abs(envelope) ** 2
        running = np.concatenate(([0.0], np.cumsum(power)))
        whole = np.floor(begin).astype(np.int64)
        strength[c] = ((running[ends + 1] - running[whole])
                       / np.maximum(ends + 1 - whole, 1))

    level = lines.level
    floor_lines = (np.zeros((len(lines.rows), count), dtype=bool)
                   if level.below_floor is None
                   else np.asarray(level.below_floor, dtype=bool)[lines.picks])
    toneless = np.isfinite(strength) & ~(strength > 0.0)
    judged = (~floor_lines & ~lines.shared[None, :] & np.isfinite(measured)
              & (strength > 0.0))
    if floor_lines.any():
        n = int(floor_lines.sum())
        notes.append(f'{n} line{"s" * (n != 1)} under the noise floor, not '
                     'judged')
    if lines.shared.any():
        n = int(lines.shared.sum())
        notes.append(f'{n} line{"s" * (n != 1)} shared with a tone the levels '
                     'do not hold, not judged')
    for c in range(len(lines.rows)):
        none = int(toneless[c].sum())
        if none:
            notes.append(f'{lines.wanted[c]}: {none} line{"s" * (none != 1)} '
                         'with no tone on the channel to read, not judged')
    if level.drift_hz:
        notes.append(f'the extraction moved this sweep\'s clock by '
                     f'{level.drift_hz:+.3g} Hz at its end; the check reads '
                     'the record against the specification, so that is '
                     'counted here')
    return FrequencyCheck(
        tone=lines.chosen.name, dofs=list(lines.wanted),
        frequency=np.asarray(level.abscissa, dtype=float),
        seconds=lines.seconds, specified=specified, measured=measured,
        judged=judged, tolerance=tolerance, floor=floor, cycles=lines.cycles,
        notes=tuple(notes))


__all__ = ['FREQUENCY_TOLERANCE', 'FrequencyCheck', 'allowed',
           'check_frequency']
