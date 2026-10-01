"""Sine sweep specifications: named tones, each its own time-boxed sweep.

Phase B of the sine sweep arc (PLAN.md). A sine test's requirement is
not a curve over one abscissa — it is a *set of tones*, each a
breakpoint table with its own sweep law and its own start time, played
simultaneously and only partially overlapping in the ordinary case
(measured on the Phase A fixture: four tones playing 0-15, 2-13, 3-14
and 1-8 s of one 15 s environment). That shape does not fit a
DataArray, so it gets its own class rather than a forced one.

The sweep law here is the one the Phase A runs pinned against the
controller's own records, not assumed from documentation:

- `segment_rate[i]` governs the segment from `frequency[i]` to
  `frequency[i+1]` — the leading convention, matched to 0.0009 Hz
  over a 14 s sweep.
- A linear segment's rate is in Hz/s; a logarithmic segment's rate is
  in **oct/min** (one octave at rate 10 transited in 5.87 s ~= 6 s;
  the controller's own docstrings disagree with each other and the
  measurement settled it).
- A descending segment takes a negative rate. A rate whose sign
  contradicts its breakpoints is refused by name here, because the
  controller refuses the same thing as a negative sweep time — at
  initialization, deep in its log.

Amplitude between breakpoints interpolates linearly *in time*, which
is linear in frequency on a linear segment and linear in log-frequency
on a logarithmic one — `target()` interpolates that way per segment,
so the compliance curve is the curve the controller was actually
chasing.

One more alignment fact for the extraction to lean on, measured on the
Phase A fixture: the controller ramps each tone up for the
environment's `ramp_time` before sweeping, so the recorded trajectory
leaves its start frequency at `start_time + ramp_time` — all four
fixture tones rebuilt against the controller's own record to 0.0000 Hz
with exactly that lag.

Everything is SI at rest, like every other object in core.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data import NAMED_CLASSES, Bounded, Spectrum
from .progress import Ticker

LINEAR = 0
LOG = 1


class SineTone:
    """One tone: breakpoints, a sweep law, and its own window in time.

    Attributes:
        name: What the controller called it ('Sine Tone 1').
        start_time: Seconds after the environment started that this
            tone turns on. Tones are silent outside their own span —
            never held at an end frequency.
        frequency: Breakpoints, Hz, shape (n,). Ascending or descending
            per segment; each segment's rate sign must agree.
        amplitude: Target amplitude at each breakpoint per control
            channel, shape (n, m), SI.
        phase: Radians at each breakpoint per channel, shape (n, m).
        segment_type: (n-1,), LINEAR or LOG per segment.
        segment_rate: (n-1,), Hz/s for a linear segment, oct/min for a
            logarithmic one; negative for a descending segment.
        warning_lower, warning_upper, abort_lower, abort_upper:
            Band curves at the breakpoints, shape (n, m), or None where
            the specification carries no such limit.
    """

    LIMITS = ('warning_lower', 'warning_upper', 'abort_lower',
              'abort_upper')

    def __init__(self, name: str, start_time: float,
                 frequency: Any, amplitude: Any,
                 segment_type: Any, segment_rate: Any,
                 phase: Any = None,
                 warning_lower: Any = None, warning_upper: Any = None,
                 abort_lower: Any = None, abort_upper: Any = None) -> None:
        self.name: str = str(name)
        self.start_time: float = float(start_time)
        self.frequency: np.ndarray = np.asarray(frequency,
                                                dtype=np.float64)
        self.amplitude: np.ndarray = np.atleast_2d(
            np.asarray(amplitude, dtype=np.float64))
        n = len(self.frequency)
        if self.amplitude.shape[0] != n:
            raise ValueError(
                f'{self.name}: {n} breakpoints but amplitude has '
                f'{self.amplitude.shape[0]} rows')
        self.phase: np.ndarray = (
            np.zeros_like(self.amplitude) if phase is None
            else np.atleast_2d(np.asarray(phase, dtype=np.float64)))
        self.segment_type: np.ndarray = np.asarray(segment_type,
                                                   dtype=np.int64)
        self.segment_rate: np.ndarray = np.asarray(segment_rate,
                                                   dtype=np.float64)
        if len(self.segment_type) != n - 1 or len(self.segment_rate) != n - 1:
            raise ValueError(
                f'{self.name}: {n} breakpoints take {n - 1} segments; '
                f'got {len(self.segment_type)} types and '
                f'{len(self.segment_rate)} rates')
        self.limits: dict[str, np.ndarray] = {}
        for limit, values in (('warning_lower', warning_lower),
                              ('warning_upper', warning_upper),
                              ('abort_lower', abort_lower),
                              ('abort_upper', abort_upper)):
            if values is None:
                continue
            values = np.atleast_2d(np.asarray(values, dtype=np.float64))
            if values.shape != self.amplitude.shape:
                raise ValueError(
                    f'{self.name}: {limit} has shape {values.shape}, '
                    f'expected {self.amplitude.shape} to match the '
                    'amplitude')
            self.limits[limit] = values
        self._check_rates()

    def _check_rates(self):
        for i in range(len(self.segment_rate)):
            f0, f1 = self.frequency[i], self.frequency[i + 1]
            rate = self.segment_rate[i]
            if f0 == f1:
                raise ValueError(
                    f'{self.name}: segment {i} holds {f0} Hz — a dwell '
                    'has no sweep time; give the segment a frequency '
                    'span, however small')
            if rate == 0.0 or (f1 > f0) != (rate > 0):
                direction = 'ascending' if f1 > f0 else 'descending'
                raise ValueError(
                    f'{self.name}: segment {i} sweeps {direction} '
                    f'({f0} to {f1} Hz) but its rate is {rate} — the '
                    'rate sign must match the direction, negative for '
                    'a descending sweep')

    # ---- the sweep law ---------------------------------------------------

    def segment_seconds(self) -> np.ndarray:
        """How long each segment takes, from its span and its rate."""
        seconds = np.empty(len(self.segment_rate))
        for i in range(len(seconds)):
            f0, f1 = self.frequency[i], self.frequency[i + 1]
            if self.segment_type[i] == LINEAR:
                seconds[i] = (f1 - f0) / self.segment_rate[i]
            else:
                # oct/min, measured — see the module docstring
                seconds[i] = np.log2(f1 / f0) / self.segment_rate[i] * 60.0
        return seconds

    def duration(self) -> float:
        return float(self.segment_seconds().sum())

    def span(self) -> tuple[float, float]:
        """When this tone plays, in environment time."""
        return (self.start_time, self.start_time + self.duration())

    def trajectory(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Frequency versus time over the tone's own span.

        Returns (t, f): t in seconds from the tone's start (add
        `start_time` for environment time), f in Hz. This is the
        rebuild that matched the controller's own record to 0.0009 Hz.
        """
        seconds = self.segment_seconds()
        pieces_t: list[np.ndarray] = [np.array([0.0])]
        pieces_f: list[np.ndarray] = [self.frequency[:1]]
        start = 0.0
        for i, length in enumerate(seconds):
            n = max(round(length / dt), 1)
            t = start + length * np.arange(1, n + 1) / n
            f0, f1 = self.frequency[i], self.frequency[i + 1]
            fraction = np.arange(1, n + 1) / n
            if self.segment_type[i] == LINEAR:
                f = f0 + (f1 - f0) * fraction
            else:
                f = f0 * (f1 / f0) ** fraction
            pieces_t.append(t)
            pieces_f.append(f)
            start += length
        return np.concatenate(pieces_t), np.concatenate(pieces_f)

    def argument(self, dt: float) -> np.ndarray:
        """The cosine argument over the tone's span: 2*pi*integral(f),
        at the sample times `trajectory` returns — `phase_at` on them,
        so a slice of the sweep (`argument_slice`) is the same numbers
        as the whole (2026-10-01)."""
        t, _f = self.trajectory(dt)
        return self.phase_at(t)

    def _segments(self):
        """(start second, length, f0, f1, log?) per segment."""
        out, start = [], 0.0
        for i, length in enumerate(self.segment_seconds()):
            out.append((start, float(length), float(self.frequency[i]),
                        float(self.frequency[i + 1]),
                        int(self.segment_type[i]) != LINEAR))
            start += float(length)
        return out

    def frequency_at(self, t: Any) -> np.ndarray:
        """The instantaneous frequency, Hz, at seconds `t` from the
        tone's start — the same law `trajectory` lays on its grid,
        evaluated anywhere, so a slice of a long sweep costs the slice.

        Parameters
        ----------
        t : array-like
            Seconds from the tone's start.

        Returns
        -------
        ndarray
        """
        t = np.asarray(t, dtype=np.float64)
        f = np.full(t.shape, float(self.frequency[-1]))
        f[t <= 0.0] = float(self.frequency[0])
        for start, length, f0, f1, log in self._segments():
            inside = (t > start) & (t <= start + length)
            fraction = (t[inside] - start) / length
            f[inside] = (f0 * (f1 / f0) ** fraction if log
                         else f0 + (f1 - f0) * fraction)
        return f

    def phase_at(self, t: Any) -> np.ndarray:
        """The cosine argument, radians, at seconds `t` from the
        tone's start: 2*pi times the integral of the frequency law,
        segment by segment in closed form — a linear sweep's chirp, a
        log sweep's exponential.

        Parameters
        ----------
        t : array-like
            Seconds from the tone's start.

        Returns
        -------
        ndarray
        """
        t = np.asarray(t, dtype=np.float64)
        phase = np.zeros(t.shape)
        carried = 0.0            # cycles at the start of each segment
        for start, length, f0, f1, log in self._segments():
            local = np.clip(t - start, 0.0, length)
            if log:
                ratio = f1 / f0
                cycles = f0 * length / np.log(ratio) * (ratio ** (local / length) - 1.0)
                whole = f0 * length / np.log(ratio) * (ratio - 1.0)
            else:
                cycles = f0 * local + (f1 - f0) * local ** 2 / (2.0 * length)
                whole = (f0 + f1) * length / 2.0
            phase += cycles
            carried += whole
        # past the last segment the phase holds, as the frequency does
        over = t > sum(length for _s, length, *_r in self._segments())
        phase[over] = carried
        return 2.0 * np.pi * phase

    def grid(self, dt: float, first: int, last: int) -> np.ndarray:
        """The sample times `trajectory(dt)` lays down, from sample
        `first` to `last` (exclusive): the whole grid's own numbers,
        for a slice of it.

        Parameters
        ----------
        dt : float
            The sample interval.
        first, last : int
            The slice of the sweep's samples.

        Returns
        -------
        ndarray
        """
        # the grid is one sample at t=0, then each segment's samples
        # evenly over its own length — rebuilt per segment so a slice
        # never allocates the whole
        edges = []
        for start, length, *_rest in self._segments():
            n = max(round(length / dt), 1)
            edges.append((start, length, n))
        total = 1 + sum(n for _s, _l, n in edges)
        first, last = max(int(first), 0), min(int(last), total)
        out = np.empty(max(last - first, 0))
        if last <= first:
            return out
        index = 0
        if first == 0:
            out[0] = 0.0
            index = 1
        offset = 1
        for start, length, n in edges:
            lo, hi = max(first, offset), min(last, offset + n)
            if hi > lo:
                k = np.arange(lo - offset + 1, hi - offset + 1)
                out[index:index + hi - lo] = start + length * k / n
                index += hi - lo
            offset += n
        return out

    def grid_at(self, dt: float, samples: Any) -> np.ndarray:
        """The sample times `trajectory(dt)` lays down, at the given
        sample indices — `grid` for a handful of samples picked out of
        a long sweep, without laying the whole sweep down.

        Parameters
        ----------
        dt : float
            The sample interval.
        samples : array-like of int
            Sample indices into the sweep's grid.

        Returns
        -------
        ndarray
        """
        samples = np.asarray(samples, dtype=np.int64)
        out = np.zeros(samples.shape)
        offset = 1
        for start, length, *_rest in self._segments():
            n = max(round(length / dt), 1)
            inside = (samples >= offset) & (samples < offset + n)
            out[inside] = start + length * (samples[inside] - offset + 1) / n
            offset += n
        return out

    def target(self, frequencies: Any,
               curve: str = 'amplitude') -> np.ndarray:
        """The specified level at given frequencies, shape (len, m).

        `curve` is 'amplitude' or one of the limit names. Amplitude
        interpolates linearly in time — linear in frequency on a
        linear segment, linear in log-frequency on a logarithmic one —
        because that is the target the controller chases. Frequencies
        the tone never sweeps come back NaN: the specification says
        nothing there, and NaN says so where zero would lie.
        """
        values = (self.amplitude if curve == 'amplitude'
                  else self.limits.get(curve))
        if values is None:
            raise ValueError(f'{self.name} carries no {curve}')
        frequencies = np.asarray(frequencies, dtype=np.float64)
        out = np.full((len(frequencies), values.shape[1]), np.nan)
        for i in range(len(self.segment_rate)):
            f0, f1 = self.frequency[i], self.frequency[i + 1]
            lo, hi = min(f0, f1), max(f0, f1)
            inside = (frequencies >= lo) & (frequencies <= hi)
            if not inside.any():
                continue
            if self.segment_type[i] == LINEAR:
                fraction = (frequencies[inside] - f0) / (f1 - f0)
            else:
                fraction = (np.log(frequencies[inside] / f0)
                            / np.log(f1 / f0))
            out[inside] = (values[i]
                           + (values[i + 1] - values[i]) * fraction[:, None])
        return out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SineTone):
            return NotImplemented
        same = (self.name == other.name
                and self.start_time == other.start_time
                and np.array_equal(self.frequency, other.frequency)
                and np.array_equal(self.amplitude, other.amplitude)
                and np.array_equal(self.phase, other.phase)
                and np.array_equal(self.segment_type, other.segment_type)
                and np.array_equal(self.segment_rate, other.segment_rate)
                and set(self.limits) == set(other.limits))
        return same and all(
            np.allclose(values, other.limits[name], equal_nan=True)
            for name, values in self.limits.items())

    def __repr__(self) -> str:
        lo, hi = self.frequency.min(), self.frequency.max()
        t0, t1 = self.span()
        return (f'<SineTone {self.name!r}: {lo:g}-{hi:g} Hz, '
                f'{t0:g}-{t1:g} s>')


class SineSweepSpecification:
    """What a sine test was controlled to: tones over control channels.

    One object per sine environment, mirroring the controller's file:
    the tones (each with its own breakpoints, sweep law, bands and
    start time) and the control DOFs their amplitude columns belong
    to. Simultaneous tones are the ordinary case, not a variant.

    Attributes:
        tones: The SineTone list, in the file's order.
        response_dof: The control channel DOF strings — the columns of
            every tone's amplitude table, in order.
        ordinate_dim: What the amplitudes are ('acceleration' for a
            controller run on accelerometers).
        ordinate_unit: The SI unit the amplitudes are stored in, or
            None while undeclared — the same convention every DataArray
            follows.
        comment: One line about the set as a whole.
    """

    def __init__(self, tones: Sequence[SineTone],
                 response_dof: Sequence[str],
                 ordinate_dim: str = 'acceleration',
                 ordinate_unit: str | None = None,
                 comment: str = '') -> None:
        self.tones: list[SineTone] = list(tones)
        self.response_dof: list[str] = list(response_dof)
        m = len(self.response_dof)
        for tone in self.tones:
            if tone.amplitude.shape[1] != m:
                raise ValueError(
                    f'{tone.name} carries {tone.amplitude.shape[1]} '
                    f'amplitude columns for {m} control channels')
        names = [tone.name for tone in self.tones]
        if len(set(names)) != len(names):
            raise ValueError(f'tone names repeat: {sorted(names)}')
        self.ordinate_dim: str = ordinate_dim
        self.ordinate_unit: str | None = ordinate_unit
        self.comment: str = comment

    def tone(self, name: str) -> SineTone:
        for tone in self.tones:
            if tone.name == name:
                return tone
        raise KeyError(f'no tone named {name!r}; '
                       f'this specification holds {[t.name for t in self.tones]}')

    def span(self) -> tuple[float, float]:
        """When any tone is playing: first start to last end."""
        starts, ends = zip(*(tone.span() for tone in self.tones))
        return (min(starts), max(ends))

    def tone_curve(self, name: str, lines: int = 400) -> SineTarget:
        """One tone's requirement as a plottable, comparable curve.

        The abscissa follows the tone's own sweep — `lines` points
        spaced the way the sweep dwells, ascending whichever way it
        swept — and the bands ride along as Bounded limits.
        """
        tone = self.tone(name)
        _t, f = tone.trajectory(tone.duration() / lines)
        order = np.argsort(f)
        frequencies = f[order]
        target = tone.target(frequencies)
        limits = {limit: tone.target(frequencies, curve=limit).T
                  for limit in tone.limits}
        return SineTarget(
            abscissa=frequencies, ordinate=target.T,
            response_dof=list(self.response_dof),
            ordinate_dim=[self.ordinate_dim] * len(self.response_dof),
            ordinate_unit=[self.ordinate_unit] * len(self.response_dof),
            comment=[f'{name} requirement at {dof}'
                     for dof in self.response_dof],
            tone=name, **limits)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SineSweepSpecification):
            return NotImplemented
        return (self.tones == other.tones
                and self.response_dof == other.response_dof
                and self.ordinate_dim == other.ordinate_dim
                and self.ordinate_unit == other.ordinate_unit)

    def __repr__(self) -> str:
        lo = min(tone.frequency.min() for tone in self.tones)
        hi = max(tone.frequency.max() for tone in self.tones)
        return (f'<SineSweepSpecification: {len(self.tones)} tones, '
                f'{lo:g}-{hi:g} Hz, {len(self.response_dof)} control '
                'channels>')



class SineTarget(Bounded, Spectrum):
    """One tone's requirement laid over frequency, as a curve.

    What `SineSweepSpecification.tone_curve` hands the plot and the
    report: the tone's amplitude interpolated along its own sweep,
    with the warning and abort bands as the four limit curves every
    Bounded object carries — so the comparison against an extracted
    level draws with the same zones, the same shading and the same
    pairing a random specification gets, from the same machinery.
    Derived on demand; the specification object stays the one source.
    """

    def __init__(self, *args: Any, tone: str = '', **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tone: str = str(tone)


class SineLevel(Spectrum):
    """A sine sweep's measured level: amplitude against frequency.

    What `extract_sine` reads out of a recording for one tone — the
    demodulated complex amplitude of that tone at each control channel,
    sampled along the sweep and laid over frequency. The magnitude is
    the tracked amplitude the controller was steering; the angle is the
    phase relative to the reconstructed sweep argument, meaningful
    between channels rather than absolutely.

    The frequency coverage is the coverage: a run stopped early, or a
    tone the instructions windowed, extracts fewer lines, and the
    comparison against the specification sees exactly how much of the
    required range was actually run rather than being told everything
    was.

    Attributes:
        tone: The specification tone this level was extracted for.
        onset: Seconds into the recording where the tone's sweep proper
            was found (matched filter, or the caller's override) —
            reported so the alignment is auditable.
        seconds: When each line was measured, in recording seconds —
            the sweep's own clock, one entry per abscissa line, which
            is what lets the level stand on the 3-D stage without the
            specification beside it. None on a level from before the
            clock was kept.
        floor: The noise floor beside each reading, channels × lines,
            in the level's own units: the amplitude a tone would need
            to stand clear of the noise the smoothing let through
            (2026-09-30). None on a level from before it was kept.
        below_floor: Channels × lines, True where the tone was under
            its floor and the reading is reported *at* the floor
            rather than at zero — a mark on the plot and a line the
            comparison does not judge. None on an older level.
        drift_hz: The sweep-clock correction applied at the end of the
            tone's sweep, in Hz, when the recording's sweep had
            drifted from the specification's; 0 when none was needed.
    """

    def __init__(self, *args: Any, tone: str = '', onset: float = 0.0,
                 seconds: Any = None, floor: Any = None,
                 below_floor: Any = None, drift_hz: float = 0.0,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tone: str = str(tone)
        self.onset: float = float(onset)
        self.seconds: np.ndarray | None = (
            None if seconds is None
            else np.asarray(seconds, dtype=np.float64))
        self.floor: np.ndarray | None = (
            None if floor is None else np.asarray(floor, dtype=np.float64))
        self.below_floor: np.ndarray | None = (
            None if below_floor is None
            else np.asarray(below_floor, dtype=bool))
        self.drift_hz: float = float(drift_hz)

    @property
    def resolved(self) -> np.ndarray:
        """Channels × lines, True where the tone stood above its floor
        — every line, on a level that kept no floor."""
        if self.below_floor is None:
            return np.ones(np.shape(self.ordinate), dtype=bool)
        return ~self.below_floor


def _tone_score(records: Any, dt: float, tone: SineTone) -> np.ndarray:
    """Matched-filter correlation power for one tone at every lag.

    `records` is a sequence of one-dimensional records (views onto the
    history, never copies). Overlap-add convolution rather than one
    FFT of the whole record, so the memory is a block's, not the
    record's (2026-10-01: a 23 GB run)."""
    from scipy.fft import set_workers
    from scipy.signal import oaconvolve

    records = [np.asarray(record) for record in records]
    length = len(records[0])
    template = np.cos(tone.argument(dt))
    if length <= len(template):
        # a recording that stopped mid-sweep still holds the sweep's
        # opening, and the opening aligns it: correlate on the half
        # that fits, leaving the other half as search room. Anything
        # shorter than one second of tone has nothing to lock onto.
        template = template[:length // 2]
        if len(template) < 1.0 / dt:
            raise ValueError(
                f'{tone.name}: the recording '
                f'({length * dt:.2f} s) holds less than a '
                'second of the tone; nothing to align')
    score = np.zeros(length - len(template) + 1)
    flipped = template[::-1]
    # the FFTs inside run on every core; nothing is copied for it
    with set_workers(_workers()):
        for record in records:
            score += oaconvolve(np.asarray(record, dtype=float), flipped,
                                mode='valid') ** 2
    return score


def _workers(wanted: int | None = None) -> int:
    """How many workers to use: `wanted`, else the cores up to
    `MAX_WORKERS`."""
    import os

    if wanted is not None:
        return max(int(wanted), 1)
    return max(1, min(MAX_WORKERS, os.cpu_count() or 1))


class SineLevelSet:
    """One extraction, one object: the tones' levels, grouped the way
    the specification groups its tones (Brandon, 2026-08-22).

    Each tone sweeps its own frequencies on its own clock, so the
    levels stay separate `SineLevel`s inside — different abscissas
    cannot share a DataArray — but the *project* holds one thing, it
    expands into one row per tone, and picking rows plots a subset,
    exactly as the specification does.

    Attributes:
        levels: The per-tone SineLevels, in the specification's order.
        cycles: The smoothing the levels were read with, in cycles of
            each tone's instantaneous frequency; None on a set from
            before it was kept.
    """

    def __init__(self, levels: Sequence[SineLevel],
                 cycles: float | None = None) -> None:
        self.levels: list[SineLevel] = list(levels)
        self.cycles: float | None = None if cycles is None else float(cycles)
        names = [level.tone for level in self.levels]
        if len(set(names)) != len(names):
            raise ValueError(f'tone names repeat: {sorted(names)}')

    @property
    def tone_names(self) -> list[str]:
        return [level.tone for level in self.levels]

    @property
    def response_dof(self) -> list[str]:
        return list(self.levels[0].response_dof) if self.levels else []

    def tone(self, name: str) -> SineLevel:
        for level in self.levels:
            if level.tone == name:
                return level
        raise KeyError(f'no level for tone {name!r}; this set holds '
                       f'{self.tone_names}')

    def __iter__(self):
        return iter(self.levels)

    def __len__(self) -> int:
        return len(self.levels)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SineLevelSet):
            return NotImplemented
        return self.levels == other.levels

    def __repr__(self) -> str:
        return (f'<SineLevelSet: {len(self.levels)} tones, '
                f'{len(self.response_dof)} channels>')


def _records(records: Any) -> list[np.ndarray]:
    """One-dimensional records, whatever was handed in: a 2-D array's
    rows, or a sequence of records — views, never copies."""
    if isinstance(records, np.ndarray):
        return list(np.atleast_2d(records))
    return [np.asarray(record) for record in records]


def find_tone(records: np.ndarray, dt: float, tone: SineTone,
              search: tuple[float, float] | None = None) -> float:
    """Where a tone's sweep begins in a recording, by matched filter.

    Correlates the reconstructed sweep template against every record
    and sums the correlation power across them — the processing gain
    over a whole sweep is what finds a tone under a random excitation
    much louder than it (measured: a clean find under +15.6 dB of
    random). `search` bounds the onset in seconds when the caller
    knows roughly where to look. Returns the onset in seconds.
    """
    records = _records(records)
    score = _tone_score(records, dt, tone)
    if search is not None:
        lo = max(round(search[0] / dt), 0)
        hi = min(round(search[1] / dt) + 1, len(score))
        if lo >= hi:
            raise ValueError(f'{tone.name}: the search window '
                             f'{search} holds no onset candidates')
        return float((lo + int(np.argmax(score[lo:hi]))) * dt)
    return float(int(np.argmax(score)) * dt)


def find_environment(records: np.ndarray, dt: float,
                     tones: Sequence[SineTone], ticker=None) -> float:
    """Where the tones' shared clock starts, by joint matched filter.

    Every tone in one environment begins at its own `start_time` on
    one clock, so there is one unknown — the clock's position in the
    recording — and every tone's correlation votes on it at its own
    lag. The sharp votes carry the ambiguous ones: a near-dwell or a
    log sweep that mis-locks alone (measured: half a second off in a
    four-tone mix) is pinned by the linear sweeps beside it. Returns
    the clock origin in seconds; tone i's sweep begins at
    `origin + start_time_i`.
    """
    records = _records(records)
    joint = None
    if ticker is not None:
        ticker.add(len(tones))
    for tone in tones:
        lag = round(tone.start_time / dt)
        score = _tone_score(records, dt, tone)
        if ticker is not None:
            ticker.tick()
        if len(score) <= lag:
            continue
        vote = score[lag:]
        # summed as it comes, one vote alive at a time (2026-10-01)
        if joint is None:
            joint = vote.copy()
        else:
            length = min(len(joint), len(vote))
            joint = joint[:length] + vote[:length]
    if joint is None:
        raise ValueError('no tone fits the recording at its own start '
                         'time; nothing to align')
    return float(int(np.argmax(joint)) * dt)


def _smooth(values: np.ndarray, half_windows: np.ndarray) -> np.ndarray:
    """Centered moving average whose window varies per sample.

    The window follows the instantaneous frequency, so a sweep is
    averaged over a fixed number of *cycles* everywhere rather than a
    fixed number of seconds. Cumulative-sum implementation, one pass.
    Used for the noise estimate beside the Vold-Kalman envelope.
    """
    n = len(values)
    sums = np.concatenate(([0.0 + 0.0j], np.cumsum(values)))
    lo = np.clip(np.arange(n) - half_windows, 0, n).astype(np.int64)
    hi = np.clip(np.arange(n) + half_windows + 1, 0, n).astype(np.int64)
    return (sums[hi] - sums[lo]) / np.maximum(hi - lo, 1)


#: the -3 dB bandwidth of a centered average over L samples is 0.443 fs/L;
#: the Vold-Kalman penalty is tuned to the same bandwidth for the same
#: `cycles`, so the setting means what it meant under the moving average
_AVERAGE_BANDWIDTH = 0.443
#: the two-sided integral of 1/(1+x^4)^2 — the second-order filter's
#: equivalent noise bandwidth in units of its corner, 3*pi/(4*sqrt(2))
_NOISE_INTEGRAL = 3.0 * np.pi / (4.0 * np.sqrt(2.0))


#: samples a chunk of the Vold-Kalman solve keeps, and the margin
#: solved beyond it on each side, in smoothing windows of the widest
#: tone live at the edge. The solve's memory is linear in the span
#: (measured 2026-09-30: 1.2 kB per sample for one tone, 4.6 kB for
#: three), so a 19-minute sweep solved whole wanted tens of gigabytes
#: per channel and took the machine down (Brandon). The margin is
#: measured, not derived — the seam error at a 4096-sample chunk read
#: 2e-6 of the envelope at eight windows, 1.3e-7 at sixteen, 5e-9 at
#: twenty-four — and `test_extract_sine` pins the interior within 1e-5
#: of the whole-record solve at the smallest chunk. Twenty-four was
#: the first choice; eight is a third of the piece for a seam error
#: of 2e-5 dB, nothing against a reading (2026-10-01).
#: The streaming reader (`_read_levels`) takes the same count as
#: cycles of the tone (`MARGIN_WINDOWS * cycles`) rather than samples
#: of a window at the edge, since at the low end of a log sweep the
#: window is millions of samples and the chunk no longer bounded
#: anything (2026-10-01).
CHUNK = 1 << 16
MARGIN_WINDOWS = 8.0
#: tones are solved together only where they are close in frequency:
#: within this many smoothing bandwidths of each other (the window's
#: bandwidth is about f/cycles) at some sample both are live, or
#: crossing. Farther apart the filter separates them by frequency
#: alone, and a joint solve costs the square of the tones it holds —
#: nine tones in nine bands as one system, at the rate the highest
#: needed, was 11 kB a sample before the solver's workspace and a
#: 90 GB worker on a 23 GB run (Brandon, 2026-10-01); apart, each is
#: one tone, decimated to its own rate
SEPARATION_BANDWIDTHS = 4.0
#: where the floor is measured: this many filter corners to either
#: side of the tone along its sweep — outside the notch the fit cuts
#: in the residual at the tone (2.5 % of the noise taken at 2.5
#: corners, 1.2 % at 3), inside the separation that keeps another
#: group's tone away (4 bandwidths is 9 corners)
SIDE_CORNERS = 3.0
#: the fewest samples per cycle of the highest tone a piece is solved
#: at: each piece is decimated to a rate that gives the highest tone
#: live in it at least this many, since a 5 Hz tone sampled at
#: 20 kHz carries four thousand samples a cycle its envelope never
#: needs (2026-10-01: a 23 GB run's first piece was 63 million samples
#: at full rate and the banded solve 189 GB — the chunk was bounded,
#: the margin at the low end of a log sweep under heavy smoothing was
#: not; in cycles it is, and decimated it is small)
SAMPLES_PER_CYCLE = 8
#: the most memory the pool's in-flight pieces may take together, in
#: bytes — the banded solve costs ~3 kB per decimated sample, so a
#: wide piece is sent to fewer workers at once rather than all eight
POOL_BUDGET = 8 * 2 ** 30

#: the smoothing the extraction starts from — ten cycles of the
#: instantaneous frequency, the tracking-filter convention that
#: follows a resonance closely — and the ladder the automatic choice
#: climbs from it. Measured on a planted flat sweep (2026-09-30): the
#: scatter of the readings falls about 3 dB each time the smoothing
#: doubles, so the ladder's steps are the useful ones.
BASE_CYCLES = 10.0
CYCLES_LADDER = (10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0,
                 240.0, 320.0, 480.0, 640.0)
#: the scatter the automatic smoothing aims to hold the readings
#: under, one standard deviation, in dB
TARGET_SCATTER_DB = 1.0
#: how many stretches of each tone the sampled solve reads — enough to
#: see the sweep's range of signal-to-noise, cheap enough to run on
#: every edit of the setting
SAMPLE_WINDOWS = 8
#: the frequency offset, as a fraction of the filter's bandwidth, past
#: which the measured sweep clock is corrected against the
#: specification's and the solve repeated: the second-order filter is
#: within 0.2 % of flat there, and a sweep that drifted further would
#: begin to read low
REFINE_FRACTION = 0.2
#: the most worker processes an extraction spreads its pieces over
#: (Brandon, 2026-10-01: use the cores). Eight: the solve scaled to
#: 3.5x on eight workers of a twelve-core Mac and fell back at twelve,
#: when the efficiency cores joined; and the work it pays for — a
#: piece's worth of samples times channels under which the workers'
#: start-up (a second each, importing this package) costs more than
#: it saves, measured on the test recordings
MAX_WORKERS = 8
PARALLEL_FLOOR = 4_000_000


@dataclass(frozen=True)
class SineExtraction:
    """How a sweep's levels are read out of a recording — the settings
    that ride the time history, the way `Averaging` and `Filtering`
    do (the sine view sets them; Extract Sine Levels reads whatever is
    there).

    `cycles` is the smoothing: how many cycles of the tone's own
    instantaneous frequency each reading averages over. Shorter
    follows a resonance closely; longer holds the random environment
    out of the reading. None means *automatic*: the extraction samples
    the recording at `BASE_CYCLES`, measures how much noise the
    smoothing lets through against the tone it finds, and climbs
    `CYCLES_LADDER` until the predicted scatter of the readings is
    under `target_db` (Brandon, 2026-09-30: a 0.5 g sweep under a
    2.3 g RMS random environment read with a 17 dB spread and a fifth
    of its points at zero at ten cycles; the lever is the smoothing,
    and the data can say how much). `chosen` records what the
    automatic picked, so the setting read back says what was used.

    `refine` corrects the specification's sweep clock against the
    recording: the envelope's residual phase is fitted and the sweep
    argument moved by it, then solved again, so a controller whose
    sweep drifted from the commanded rate over a long run does not
    read low once the drift leaves the filter's bandwidth.

    Frozen like the other settings, and for the same reason: the
    staleness fingerprint is the fields, and a mutable setting would
    be a fingerprint that lies.
    """

    cycles: float | None = None
    target_db: float = TARGET_SCATTER_DB
    refine: bool = True
    chosen: float | None = field(default=None, compare=False)

    @property
    def automatic(self) -> bool:
        """Whether the smoothing is chosen from the data."""
        return self.cycles is None

    def effective_cycles(self) -> float | None:
        """The smoothing that applies: the one set, else the one the
        automatic chose, else None until an extraction has run."""
        return self.cycles if self.cycles is not None else self.chosen

    def describe(self) -> str:
        """One line for a status bar."""
        if self.cycles is not None:
            return f'{self.cycles:g} cycles of smoothing'
        if self.chosen is not None:
            return (f'automatic smoothing, {self.chosen:g} cycles chosen '
                    f'for {self.target_db:g} dB')
        return f'automatic smoothing for {self.target_db:g} dB'


def vold_kalman(signal: np.ndarray, arguments: Sequence[np.ndarray],
                frequencies: Sequence[np.ndarray],
                starts: Sequence[int], dt: float,
                cycles: float = BASE_CYCLES,
                chunk: int | None = None) -> list[np.ndarray]:
    """Every tone's complex envelope at every sample, solved jointly.

    Solved in chunks (`CHUNK` samples, or `chunk`), each with a margin
    of `MARGIN_WINDOWS` smoothing windows of the slowest tone active
    at its edges solved beyond it and discarded: the penalty's memory
    is a few windows, so the interior is the whole-record answer and
    the memory is the chunk's, not the record's. `_vold_kalman_whole`
    is the solve itself, on one span; `_solve_piece` cuts one piece
    out and solves it, which the sampled reading uses too.
    """
    signal = np.asarray(signal, dtype=float)
    spans = _spans(arguments, starts, len(signal))
    lo = min(a for a, _b in spans)
    hi = max(b for _a, b in spans)
    if hi - lo <= 0:
        raise ValueError('no tone overlaps the recording')
    chunk = CHUNK if chunk is None else int(chunk)
    if hi - lo <= chunk:
        return _vold_kalman_whole(signal, arguments, frequencies, starts,
                                  dt, cycles)
    envelopes = [np.zeros(b - a, dtype=complex) for a, b in spans]
    start = lo
    while start < hi:
        end = min(start + chunk, hi)
        solved = _solve_piece(signal, arguments, frequencies, spans, dt,
                              cycles, start, end)
        for k, (a, _b) in enumerate(spans):
            piece = solved[k]
            if piece is None:
                continue
            entry, envelope = piece
            keep_lo = max(start, entry)
            keep_hi = min(end, entry + len(envelope))
            if keep_hi > keep_lo:
                envelopes[k][keep_lo - a:keep_hi - a] = \
                    envelope[keep_lo - entry:keep_hi - entry]
        start = end
    return envelopes


def _spans(arguments, starts, length):
    """Per tone, (first sample, one past the last) of its span clipped
    to the record."""
    spans = []
    for k in range(len(arguments)):
        n = min(len(arguments[k]), length - starts[k])
        spans.append((starts[k], starts[k] + max(n, 0)))
    return spans


def _window_at(sample, spans, frequencies, dt, cycles):
    """The widest smoothing window, in samples, of any tone live at
    this sample: the margin follows the slowest tone, which decays
    slowest."""
    widest = 1.0
    for k, (a, b) in enumerate(spans):
        if a <= sample < b:
            f = max(float(frequencies[k][sample - a]), 1e-9)
            widest = max(widest, cycles / f / dt)
    return widest


def _solve_piece(signal, arguments, frequencies, spans, dt, cycles,
                 start, end):
    """Solve the record between `start` and `end` with the margin on
    each side, and return per tone `(entry, envelope)` — the sample
    where the solved piece enters the tone's span and the envelope
    from there, margin included — or None for a tone absent from the
    piece. The caller keeps the interior it asked for."""
    lo = min(a for a, _b in spans)
    hi = max(b for _a, b in spans)
    before = int(MARGIN_WINDOWS * _window_at(max(start, lo), spans,
                                             frequencies, dt, cycles))
    after = int(MARGIN_WINDOWS * _window_at(min(end, hi) - 1, spans,
                                            frequencies, dt, cycles))
    piece_lo, piece_hi = max(start - before, lo), min(end + after, hi)
    args, freqs, offsets, live = [], [], [], []
    for k, (a, b) in enumerate(spans):
        entry, leave = max(a, piece_lo), min(b, piece_hi)
        if leave <= entry:
            continue
        args.append(arguments[k][entry - a:leave - a])
        freqs.append(frequencies[k][entry - a:leave - a])
        offsets.append(entry - piece_lo)
        live.append(k)
    out: list[Any] = [None] * len(spans)
    if not live:
        return out
    solved = _vold_kalman_whole(signal[piece_lo:piece_hi], args, freqs,
                                offsets, dt, cycles)
    for k, envelope in zip(live, solved):
        out[k] = (max(spans[k][0], piece_lo), envelope)
    return out


def _vold_kalman_whole(signal: np.ndarray, arguments: Sequence[np.ndarray],
                       frequencies: Sequence[np.ndarray],
                       starts: Sequence[int], dt: float,
                       cycles: float = BASE_CYCLES) -> list[np.ndarray]:
    """Every tone's complex envelope at every sample, solved jointly,
    on one span — `vold_kalman` without the chunking.

    The second-order Vold-Kalman filter: the record is modeled as the
    sum of the tones, each a slowly varying complex amplitude on its
    own known sweep argument, and the amplitudes are found by least
    squares — the data equation pulling the sum onto the record, a
    smoothness penalty on each envelope's second difference pulling
    it towards a slow curve. Solved at once for all tones, so where
    two sweeps cross the solver separates them by their different
    frequency histories on either side, rather than each reading the
    other as noise: a planted 1.0 crossed by a 3.0 read 0.4 dB low at
    the median under the tracking demodulation this replaced
    (Brandon, 2026-09-03: the last analysis before a report should use
    the best estimator there is).

    The penalty weight follows each tone's instantaneous frequency so
    the envelope is smoothed over `cycles` cycles everywhere — the
    filter's -3 dB bandwidth matched to the centered average of the
    same span, so the setting keeps its meaning. Outside a tone's own
    span its envelope is pinned to zero.

    Parameters
    ----------
    signal : ndarray
        One channel, evenly sampled.
    arguments, frequencies : sequences of ndarray
        Per tone, the cosine argument and the instantaneous frequency
        (Hz) over the tone's span, as `SineTone.argument` and
        `SineTone.trajectory` give them.
    starts : sequence of int
        Per tone, the sample at which its span begins in `signal`.
    dt : float
        The sample interval in seconds.
    cycles : float
        Cycles of smoothing.

    Returns
    -------
    list of ndarray
        Per tone, the complex envelope over its span (clipped to the
        record): magnitude is the peak amplitude, angle the phase
        against the reconstructed sweep.
    """
    from scipy.linalg import solve_banded

    signal = np.asarray(signal, dtype=float)
    fs = 1.0 / dt
    K = len(arguments)
    spans = _spans(arguments, starts, len(signal))
    lo = min(a for a, _b in spans)
    hi = max(b for _a, b in spans)
    N = hi - lo
    if N <= 0:
        raise ValueError('no tone overlaps the recording')
    W = 2 * K                                   # unknowns per sample
    y = signal[lo:hi]

    # the data equation's per-sample coefficients, zero off a tone's span
    cos = np.zeros((K, N)); sin = np.zeros((K, N)); weight = np.zeros((K, N))
    for k, (a, b) in enumerate(spans):
        n = b - a
        if n <= 0:
            continue
        theta = arguments[k][:n]
        cos[k, a - lo:b - lo] = np.cos(theta)
        sin[k, a - lo:b - lo] = -np.sin(theta)
        f = np.maximum(frequencies[k][:n], 1e-9)
        bandwidth = _AVERAGE_BANDWIDTH * f / cycles       # Hz
        weight[k, a - lo:b - lo] = (fs / (2.0 * np.pi * bandwidth)) ** 2

    # The system is banded — the data term couples the unknowns of one
    # sample, the penalty a sample with its neighbors — so it is built
    # in LAPACK's banded layout and solved by its banded LU
    # (2026-10-01): ab[u + i - j, j] = A[i, j], u = 2W the bandwidth.
    # It was a COO list of triplets factored by SuperLU, 4.6 kB per
    # sample for three tones and the slowest thing in the extraction;
    # this is (4W + 1) * W doubles per sample and a fraction of the time.
    u = 2 * W
    size = N * W
    ab = np.zeros((2 * u + 1, size))
    rhs = np.zeros(size)
    coeff = np.empty((N, W))
    coeff[:, 0::2] = cos.T
    coeff[:, 1::2] = sin.T
    # Every unknown's column is sample * W + slot, so the unknowns of
    # one slot are a strided view `slot::W` of a row of ab — filled as
    # slices rather than through index arrays (2026-10-01: the index
    # arrays were half of each solve's time, and a scatter through
    # them holds the interpreter lock, which is why threads gained
    # nothing).
    # A^T A: one WxW block per sample, c c^T with c = [cos1, -sin1, ...]
    for i in range(W):
        rhs[i::W] = coeff[:, i] * y
        for j in range(W):
            ab[u + i - j, j::W] += coeff[:, i] * coeff[:, j]
    # the smoothness penalty D^T R^2 D per tone and component, R the
    # per-sample weight; and a unit pull to zero where a tone is absent
    for k in range(K):
        # the data term's diagonal averages 1/2 per component (cos^2,
        # sin^2 over a cycle), so the penalty is halved to put the
        # filter's -3 dB corner where the weight says — measured: the
        # noise variance came out 0.82 of the formula's before this
        w2 = 0.5 * weight[k] ** 2
        absent = (weight[k] == 0.0).astype(float)
        for comp in (0, 1):
            slot = 2 * k + comp
            diagonal = ab[u, slot::W]                 # one entry per sample
            diagonal += absent
            if N < 3:
                continue
            wm = w2[1:N - 1]
            # (x[n-1] - 2x[n] + x[n+1]) for n = 1..N-2, weighted by w[n]:
            # left is sample n-1, center n, right n+1
            diagonal[0:N - 2] += wm
            diagonal[1:N - 1] += 4.0 * wm
            diagonal[2:N] += wm
            # left-center and center-right: offsets -W (above) and +W
            above, below_ = ab[u - W, slot::W], ab[u + W, slot::W]
            above[1:N - 1] -= 2.0 * wm            # column = center
            below_[0:N - 2] -= 2.0 * wm           # column = left
            above[2:N] -= 2.0 * wm                # column = right
            below_[1:N - 1] -= 2.0 * wm           # column = center
            # left-right: offsets -2W and +2W
            ab[u - 2 * W, slot::W][2:N] += wm     # column = right
            ab[u + 2 * W, slot::W][0:N - 2] += wm  # column = left
    x = solve_banded((u, u), ab, rhs, overwrite_ab=True, overwrite_b=True,
                     check_finite=False)
    envelopes = []
    for k, (a, b) in enumerate(spans):
        real = x[2 * k::W][a - lo:b - lo]
        imag = x[2 * k + 1::W][a - lo:b - lo]
        envelopes.append(real + 1j * imag)
    return envelopes


# ---- reading the levels ---------------------------------------------------
#
# Nothing here holds a sweep-length array (2026-10-01): a 23 GB run on
# a 64 GB machine crashed at the extraction, which had carried the
# sweep's argument and frequency per tone, every tone's envelope per
# channel and the debiasing's residual at every sample — some 400
# bytes a sample on top of the record. The tone's law is closed-form
# (`SineTone.phase_at`, `frequency_at`, `grid`), so a chunk of the
# sweep is made when the chunk is solved and dropped after; the
# readings are taken at their centers inside the chunk and the clock
# drift is accumulated as per-chunk phase slopes. What stays is the
# record itself (read as views, cast one chunk at a time) and the
# readings.


def _even_steps(abscissa, dt):
    steps = np.diff(abscissa)
    if len(steps) < 1 or not np.allclose(steps, steps[0], rtol=1e-6):
        raise ValueError('extract_sine needs an evenly sampled time '
                         'history')
    return float(steps[0])


def _control_rows(history, specification):
    history_dofs = list(history.response_dof)
    rows, missing = [], []
    for dof in specification.response_dof:
        try:
            rows.append(history_dofs.index(dof))
        except ValueError:
            missing.append(dof)
    if missing:
        raise ValueError(
            'the recording carries no record for control '
            f'channel{"s" * (len(missing) != 1)} {missing}; the '
            'specification judges channels that were not measured')
    return rows


class _Laid:
    """One tone laid onto the recording: where its sweep starts, how
    many samples it spans, the smoothing, and the clock correction
    found so far — no per-sample arrays. `argument` and `f` give any
    slice of the sweep on demand from the tone's closed-form law."""

    def __init__(self, tone, onset, start, n, dt, cycles):
        self.tone, self.onset, self.start, self.n = tone, float(onset), int(start), int(n)
        self.dt, self.cycles = float(dt), float(cycles)
        #: the clock correction, radians per second and per second
        #: squared about the span's middle, and the offset it applies
        #: at the sweep's end in Hz — zero until the refinement moves it
        self.c1, self.c2, self.drift_hz = 0.0, 0.0, 0.0

    @property
    def end(self) -> int:
        return self.start + self.n

    def seconds(self, lo, hi) -> np.ndarray:
        """Seconds from the tone's start at its samples lo..hi."""
        return self.tone.grid(self.dt, lo, hi)

    def f(self, lo, hi) -> np.ndarray:
        return self.tone.frequency_at(self.seconds(lo, hi))

    def argument(self, lo, hi) -> np.ndarray:
        t = self.seconds(lo, hi)
        phase = self.tone.phase_at(t)
        if self.c1 or self.c2:
            u = t - self.n * self.dt / 2.0
            phase = phase + self.c1 * u + self.c2 * u * u
        return phase

    def window(self, f, dt=None) -> np.ndarray:
        """The smoothing window in samples at frequency `f`, at the
        record's rate or at a decimated one."""
        dt = self.dt if dt is None else dt
        return np.maximum(self.cycles / np.maximum(np.asarray(f, dtype=float),
                                                   1e-9) / dt, 1.0)

    def samples_in(self, f, dt=None) -> np.ndarray:
        """The filter's equivalent averaging length in samples, from
        its noise bandwidth: what a moving average of that length would
        do to white noise, the debiasing formula's N."""
        dt = self.dt if dt is None else dt
        bandwidth = _AVERAGE_BANDWIDTH * np.asarray(f, dtype=float) / self.cycles
        corner = 2.0 * np.pi * bandwidth * dt               # rad/sample
        return np.maximum(2.0 * np.pi / (corner * _NOISE_INTEGRAL), 1.0)

    def max_f(self, lo, hi) -> float:
        """The highest frequency the tone reaches over its samples
        lo..hi: at the ends, and at any breakpoint inside."""
        lo, hi = max(int(lo), 0), min(int(hi), self.n)
        if hi <= lo:
            return float(self.tone.frequency.max())
        t_lo, t_hi = (float(t) for t in self.tone.grid_at(self.dt, [lo, hi - 1]))
        highest = max(float(self.tone.frequency_at([t_lo])[0]),
                      float(self.tone.frequency_at([t_hi])[0]))
        at = 0.0
        for start_s, length, f0, f1, _log in self.tone._segments():
            if t_lo < start_s < t_hi:
                highest = max(highest, f0)
            at = start_s + length
            if t_lo < at < t_hi:
                highest = max(highest, f1)
        return highest

    def reach(self, k, cycles_count, direction) -> int:
        """The tone's sample `cycles_count` cycles before (`direction`
        -1) or after (+1) sample `k`, from the closed-form phase, clipped
        to the span: the margin a chunk is solved with, in cycles of the
        tone rather than samples, so the low end of a sweep does not
        reach across the whole of it."""
        k = min(max(int(k), 0), self.n - 1)
        here = float(self.tone.phase_at([self.tone.grid_at(self.dt, [k])[0]])[0])
        target = here + direction * 2.0 * np.pi * float(cycles_count)
        lo_t, hi_t = 0.0, max(self.n - 1, 1) * self.dt
        phase_lo = float(self.tone.phase_at([lo_t])[0])
        phase_hi = float(self.tone.phase_at([hi_t])[0])
        if target <= phase_lo:
            return 0
        if target >= phase_hi:
            return self.n
        for _ in range(60):
            mid = 0.5 * (lo_t + hi_t)
            if float(self.tone.phase_at([mid])[0]) < target:
                lo_t = mid
            else:
                hi_t = mid
        return min(max(round(hi_t / self.dt), 0), self.n)

    def window_at(self, k) -> float:
        k = min(max(int(k), 0), self.n - 1)
        return float(self.window(self.f(k, k + 1))[0])

    def seconds_at(self, samples) -> np.ndarray:
        """Seconds from the tone's start at the given sweep samples."""
        return self.tone.grid_at(self.dt, samples)


def _lay_tones(signals, specification, wanted, onsets, dt, cycles,
               ticker=None):
    """Where every wanted tone sits in the recording, found jointly
    unless given, and each tone laid onto the samples."""
    length = len(signals[0])
    searching = [tone for tone in wanted
                 if (onsets or {}).get(tone.name) is None]
    origin = (find_environment(signals, dt, searching, ticker)
              if searching else 0.0)
    laid = []
    for tone in wanted:
        onset = (onsets or {}).get(tone.name)
        if onset is None:
            onset = origin + tone.start_time
        start = round(onset / dt)
        total = 1 + sum(max(round(length_s / dt), 1)
                        for length_s in tone.segment_seconds())
        n = min(total, length - start)
        placed = _Laid(tone, onset, start, max(n, 0), dt, cycles)
        if n < 1 or n < int(placed.window_at(0)):
            raise ValueError(
                f'{tone.name}: the recording holds {max(n, 0) * dt:.2f} s '
                'of the tone, less than one smoothing window — nothing '
                'to extract')
        laid.append(placed)
    return laid


def _centers(laid: _Laid, points_per_window: float) -> np.ndarray:
    """Where a tone is read: sample centers tiling the sweep, each one
    window/points apart, each an (almost) independent reading."""
    centers = []
    k = int(laid.window_at(0) / 2.0)
    while k < laid.n - int(laid.window_at(min(k, laid.n - 1)) / 2.0):
        centers.append(k)
        k += max(int(laid.window_at(k) / points_per_window), 1)
    if not centers:
        raise ValueError(f'{laid.tone.name}: no whole smoothing window '
                         'fits the recorded span')
    return np.asarray(centers, dtype=np.int64)


def _piece_task(y, args, freqs, offsets, dt, cycles, reads, blocks):
    """One channel through one piece — the work a worker process does
    (2026-10-01). `y` is the piece at its own rate, `dt` that rate's
    interval. `reads` per live tone: (local centers, half-windows,
    debiasing lengths) or None when the chunk reads none of that
    tone's centers; `blocks` per live tone: (seconds from the tone's
    start at each sample of the slice, interior lo, interior hi) for
    the clock slopes. Returns per live tone `(amplitude, floor, below,
    slopes)` — the first three at the read centers, the slopes a list
    of (seconds, slope, weight)."""
    y = np.asarray(y, dtype=float)
    rotors = [np.exp(1j * a) for a in args]
    envelopes = _vold_kalman_whole(y, args, freqs, offsets, dt, cycles)
    model = np.zeros(len(y))
    for e in range(len(args)):
        span = slice(offsets[e], offsets[e] + len(args[e]))
        model[span] += np.real(envelopes[e] * rotors[e])
    residual = y - model
    out = []
    for e in range(len(args)):
        a = args[e]
        span = slice(offsets[e], offsets[e] + len(a))
        envelope = envelopes[e]
        amplitude = floor = below = None
        if reads[e] is not None:
            local, half, samples_in = reads[e]
            # the noise the smoothing lets through, measured beside the
            # tone rather than assumed from the residual's whole power:
            # the residual demodulated along the sweep `SIDE_CORNERS`
            # filter corners above and below the tone and averaged over
            # the debiasing length. White noise gives variance/N as the
            # whole-power formula did, but a tone of another group left
            # in the residual (solved apart, 2026-10-01) is attenuated
            # by the average the way the filter attenuates it, where
            # the whole-power formula counted it as noise and read the
            # tone half a decibel low (nine bands: -0.50 dB). Beside the
            # tone, not at it: at the tone the fit has taken the
            # in-band noise into the envelope and the residual is
            # notched, which read a loud floor at half its size.
            z = residual[span] * np.conj(rotors[e])
            f_e = freqs[e]
            noise_power = np.empty(len(local))
            for c, (center, h, n_avg) in enumerate(zip(local, half, samples_in)):
                center, h, n_avg = int(center), int(h), int(n_avg)
                lo_w, hi_w = max(center - h, 0), min(center + h + 1, len(a))
                lo_z, hi_z = max(lo_w - n_avg, 0), min(hi_w + n_avg, len(a))
                i = np.arange(lo_z, hi_z)
                omega = (2.0 * np.pi * SIDE_CORNERS * _AVERAGE_BANDWIDTH
                         * float(f_e[center]) / cycles * dt)
                lo_i = np.clip(np.arange(lo_w, hi_w) - n_avg // 2, lo_z, hi_z) - lo_z
                hi_i = np.clip(np.arange(lo_w, hi_w) + n_avg - n_avg // 2, lo_z, hi_z) - lo_z
                through = 0.0
                for side in (-1.0, 1.0):
                    shifted = z[lo_z:hi_z] * np.exp(-1j * side * omega * (i - center))
                    running = np.concatenate(([0.0], np.cumsum(shifted)))
                    averaged = (running[hi_i] - running[lo_i]) / np.maximum(hi_i - lo_i, 1)
                    through += np.mean(np.abs(averaged) ** 2)
                noise_power[c] = 4.0 * through / 2.0
            power = np.abs(envelope[local]) ** 2 - noise_power
            below = power <= 0.0
            shown = np.where(below, np.sqrt(noise_power),
                             np.sqrt(np.maximum(power, 0.0)))
            amplitude = shown * np.exp(1j * np.angle(envelope[local]))
            floor = np.sqrt(noise_power)
        # the clock: the envelope's phase slope over the chunk's
        # interior, in sub-blocks so a sweep shorter than one chunk
        # still gives the curve its slopes are fitted with, each
        # weighted by its power so a stretch under the noise says
        # nothing
        seconds, int_lo, int_hi = blocks[e]
        slopes = []
        block = max((int_hi - int_lo) // 8, 256)
        for b_lo in range(int_lo, int_hi, block):
            b_hi = min(b_lo + block, int_hi)
            if b_hi - b_lo < 4:
                continue
            weight = np.abs(envelope[b_lo:b_hi]) ** 2
            if weight.sum() <= 0.0:
                continue
            phase = np.unwrap(np.angle(envelope[b_lo:b_hi]))
            t = seconds[b_lo:b_hi]
            t_mean = np.average(t, weights=weight)
            p_mean = np.average(phase, weights=weight)
            spread = np.sum(weight * (t - t_mean) ** 2)
            if spread > 0.0:
                slope = np.sum(weight * (t - t_mean) * (phase - p_mean)) / spread
                slopes.append((float(t_mean), float(slope), float(weight.sum())))
        out.append((amplitude, floor, below, slopes))
    return out


def _run_piece(*args):
    """What the pool is handed: looks the task up in the worker's own
    module, so a task replaced in the parent (a test proving the work
    left the process) is not what the worker runs."""
    return _piece_task(*args)


def _read_levels(signals, laid, dt, cycles, centers, chunk=None, pieces=None,
                 pool=None, ticker=None):
    """The readings at each tone's centers, one chunk of the record at
    a time, every channel through each chunk before the next — in
    worker processes when a `pool` is given, each channel of each
    piece one task, the pool kept fed with a few pieces at once.

    Per chunk `[start, end)` — the chunks tile the tones' joint span,
    or are the `pieces` given — the margin the chunked solve needs is
    added each side, every tone live in the piece gives its slice of
    argument and frequency, and per channel the piece is solved
    (`_vold_kalman_whole`), the model of every tone subtracted for the
    residual, and each center inside the chunk read: the envelope's
    power less the noise power the smoothing let through, measured in
    the residual demodulated like the tone over the center's own
    window (`_debias` before this), the floor beside it, and whether
    the tone stood above it. Along the way the envelope's phase slope
    over the chunk's interior is fitted per tone and channel, which is
    what the clock refinement reads.

    Returns (amplitude, floor, below, slopes): the first three per tone
    `(channels, centers)` arrays (amplitude complex, at the floor and
    flagged where the tone was under it), `slopes` per tone a list per
    channel of `(seconds from the tone's start, phase slope, weight)`.
    """
    K, C = len(laid), len(signals)
    lo = min(placed.start for placed in laid)
    hi = max(placed.end for placed in laid)
    amplitude = [np.zeros((C, len(c)), dtype=complex) for c in centers]
    floor = [np.zeros((C, len(c))) for c in centers]
    below = [np.zeros((C, len(c)), dtype=bool) for c in centers]
    slopes = [[[] for _i in range(C)] for _k in range(K)]

    fs = 1.0 / dt
    margin_cycles = MARGIN_WINDOWS * cycles

    def live_at(sample):
        return [placed for placed in laid if placed.start <= sample < placed.end]

    def factor(a, b):
        """The decimation a stretch of the record takes: enough samples
        per cycle of the highest tone live in it."""
        highest = max([placed.max_f(a - placed.start, b - placed.start)
                       for placed in laid
                       if placed.start < b and placed.end > a] or [fs])
        return max(1, int(fs / (SAMPLES_PER_CYCLE * highest)))

    def chunk_end(start):
        """Where the chunk from `start` ends: CHUNK samples at the rate
        the chunk is solved at, settled against the highest frequency
        it reaches."""
        rate = factor(start, start + 1)
        end = min(start + CHUNK * rate, hi)
        for _ in range(3):
            lower = factor(start, end)
            if lower >= rate:
                break
            rate = lower
            end = min(start + CHUNK * rate, hi)
        return end

    def prepare(start, end):
        """The piece around a chunk — the margin in cycles of each live
        tone on either side, the whole decimated to the rate its highest
        tone needs — and what each channel's task needs: slices of
        argument and frequency per live tone on the decimated grid, the
        centers read, the seconds for the clock. None when no tone is
        live. Returns also the decimation and the piece's bounds."""
        piece_lo, piece_hi = start, end
        for placed in live_at(start) + live_at(end - 1):
            piece_lo = min(piece_lo, placed.start + placed.reach(
                start - placed.start, margin_cycles, -1))
            piece_hi = max(piece_hi, placed.start + placed.reach(
                end - 1 - placed.start, margin_cycles, +1))
        piece_lo, piece_hi = max(piece_lo, lo), min(piece_hi, hi)
        step = factor(piece_lo, piece_hi)
        dt_piece = dt * step
        count = -(-(piece_hi - piece_lo) // step)      # ceil: resample_poly's length
        live, args, freqs, offsets, reads, blocks = [], [], [], [], [], []
        for k, placed in enumerate(laid):
            # the decimated samples m whose record sample lies in the span
            m_lo = max(-(-(placed.start - piece_lo) // step), 0)
            m_hi = min(-(-(placed.end - piece_lo) // step), count)
            if m_hi <= m_lo:
                continue
            sweep = piece_lo + np.arange(m_lo, m_hi) * step - placed.start
            seconds = placed.tone.grid_at(dt, sweep)
            a = placed.tone.phase_at(seconds)
            if placed.c1 or placed.c2:
                u = seconds - placed.n * placed.dt / 2.0
                a = a + placed.c1 * u + placed.c2 * u * u
            f = placed.tone.frequency_at(seconds)
            absolute = centers[k] + placed.start
            j = np.flatnonzero((absolute >= start) & (absolute < end))
            if len(j):
                local = np.clip(np.rint((absolute[j] - piece_lo) / step).astype(np.int64)
                                - m_lo, 0, len(a) - 1)
                fc = f[local]
                reads.append((local,
                              (placed.window(fc, dt_piece) / 2.0).astype(np.int64),
                              placed.samples_in(fc, dt_piece)))
            else:
                reads.append(None)
            int_lo = max(-(-(start - piece_lo) // step), m_lo) - m_lo
            int_hi = min(-(-(end - piece_lo) // step), m_hi) - m_lo
            blocks.append((seconds, int_lo, int_hi))
            args.append(a)
            freqs.append(f)
            offsets.append(m_lo)
            live.append((k, j))
        if not live:
            return None
        return (piece_lo, piece_hi, step, dt_piece, live, args, freqs, offsets,
                reads, blocks)

    def channel_piece(i, piece_lo, piece_hi, step):
        """One channel's piece at the piece's rate: the record's samples,
        low-passed and decimated when the piece asks for it."""
        y = np.asarray(signals[i][piece_lo:piece_hi], dtype=float)
        if step == 1:
            return y
        from scipy.signal import resample_poly

        return resample_poly(y, 1, step)

    def keep(i, live, result):
        for (k, j), (amp, flo, bel, found) in zip(live, result):
            if amp is not None:
                amplitude[k][i, j] = amp
                floor[k][i, j] = flo
                below[k][i, j] = bel
            slopes[k][i].extend(found)

    if pieces is None:
        bounds = []
        start = lo
        while start < hi:
            end = chunk_end(start)
            bounds.append((start, end))
            start = end
    else:
        bounds = [(max(int(a), lo), min(int(b), hi)) for a, b in pieces]
        bounds = [(a, b) for a, b in bounds if b > a]
    if ticker is not None:
        ticker.add(len(bounds) * C)

    def tick():
        if ticker is not None:
            ticker.tick()

    if pool is None:
        for start, end in bounds:
            made = prepare(start, end)
            if made is None:
                tick()
                continue
            (piece_lo, piece_hi, step, dt_piece, live, args, freqs, offsets,
             reads, blocks) = made
            for i in range(C):
                keep(i, live, _piece_task(
                    channel_piece(i, piece_lo, piece_hi, step), args, freqs,
                    offsets, dt_piece, cycles, reads, blocks))
                tick()
        return amplitude, floor, below, slopes
    # a few pieces in flight at once, so every worker stays busy when
    # the channels alone are fewer than the workers — fewer when the
    # pieces are wide, within POOL_BUDGET; results are kept as they
    # land, in any order
    from concurrent.futures import FIRST_COMPLETED, wait

    in_flight = {}
    queue = iter(bounds)
    pending = True
    limit = max(2, -(-pool._max_workers // max(C, 1)) + 1) * C
    while pending or in_flight:
        while pending and len(in_flight) < limit:
            try:
                start, end = next(queue)
            except StopIteration:
                pending = False
                break
            made = prepare(start, end)
            if made is None:
                tick(C)
                continue
            (piece_lo, piece_hi, step, dt_piece, live, args, freqs, offsets,
             reads, blocks) = made
            # the banded solve's ~3 kB per decimated sample, per task
            cost = 3000 * (-(-(piece_hi - piece_lo) // step)) * (2 * len(live)) ** 2 / 16
            limit = max(1, min(limit, int(POOL_BUDGET // max(cost, 1))))
            for i in range(C):
                while len(in_flight) >= limit:
                    done, _rest = wait(list(in_flight), return_when=FIRST_COMPLETED)
                    for future in done:
                        ii, ll = in_flight.pop(future)
                        keep(ii, ll, future.result())
                        tick()
                future = pool.submit(
                    _run_piece, channel_piece(i, piece_lo, piece_hi, step),
                    args, freqs, offsets, dt_piece, cycles, reads, blocks)
                in_flight[future] = (i, live)
        if not in_flight:
            break
        done, _rest = wait(list(in_flight), return_when=FIRST_COMPLETED)
        for future in done:
            i, live = in_flight.pop(future)
            keep(i, live, future.result())
            tick()
    return amplitude, floor, below, slopes


def _tone_groups(laid, cycles):
    """Which placements share a solve: tones joined through pairs that
    come within `SEPARATION_BANDWIDTHS` smoothing bandwidths of each
    other, or cross, at some sample both are live. Returns index lists
    in first-appearance order; a tone far from every other is a group
    of one, solved at its own rate."""
    K = len(laid)
    parent = list(range(K))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in range(K):
        for b in range(a + 1, K):
            pa, pb = laid[a], laid[b]
            lo, hi = max(pa.start, pb.start), min(pa.end, pb.end)
            if hi <= lo:
                continue
            samples = np.linspace(lo, hi - 1, min(hi - lo, 4096)).astype(np.int64)
            fa = pa.tone.frequency_at(pa.tone.grid_at(pa.dt, samples - pa.start))
            fb = pb.tone.frequency_at(pb.tone.grid_at(pb.dt, samples - pb.start))
            gap = fa - fb
            bandwidth = (fa + fb) / (2.0 * cycles)
            if (np.any(np.abs(gap) < SEPARATION_BANDWIDTHS * bandwidth)
                    or np.any(np.sign(gap[1:]) != np.sign(gap[:-1]))):
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for k in range(K):
        groups.setdefault(find(k), []).append(k)
    return list(groups.values())


def _read_grouped(signals, laid, dt, cycles, centers, pieces_by_tone=None,
                  pool=None, ticker=None):
    """`_read_levels` a group of tones at a time (`_tone_groups`), the
    results put back in the placements' order. `pieces_by_tone`, when
    given, is a list per placement of the spans to solve for it."""
    K = len(laid)
    amplitude, floor, below, slopes = [None] * K, [None] * K, [None] * K, [None] * K
    for group in _tone_groups(laid, cycles):
        pieces = (None if pieces_by_tone is None
                  else [span for k in group for span in pieces_by_tone[k]])
        a, f, b, s = _read_levels(
            signals, [laid[k] for k in group], dt, cycles,
            [centers[k] for k in group], pieces=pieces, pool=pool,
            ticker=ticker)
        for j, k in enumerate(group):
            amplitude[k], floor[k], below[k], slopes[k] = a[j], f[j], b[j], s[j]
    return amplitude, floor, below, slopes


def _clock_correction(placed: _Laid, slopes_by_channel):
    """(c1, c2) for one tone from its channels' chunk slopes: per
    channel a weighted line through slope against time about the
    span's middle, then the median across channels — the structure's
    own phase differs channel to channel, the clock's does not. The
    correction phase is c1·u + c2·u², u seconds from the middle, so
    the slope it adds is c1 + 2·c2·u."""
    middle = placed.n * placed.dt / 2.0
    fits = []
    for points in slopes_by_channel:
        if not points:
            continue
        u = np.array([t for t, _s, _w in points]) - middle
        slope = np.array([s for _t, s, _w in points])
        weight = np.array([w for _t, _s, w in points])
        if len(points) < 2 or np.ptp(u) == 0.0:
            fits.append((float(np.average(slope, weights=weight)), 0.0))
            continue
        u_mean = np.average(u, weights=weight)
        s_mean = np.average(slope, weights=weight)
        p1 = (np.sum(weight * (u - u_mean) * (slope - s_mean))
              / np.sum(weight * (u - u_mean) ** 2))
        p0 = s_mean - p1 * u_mean
        fits.append((float(p0), float(p1) / 2.0))
    if not fits:
        return 0.0, 0.0
    c1, c2 = np.median(np.asarray(fits), axis=0)
    return float(c1), float(c2)


def sample_levels(history: Any, specification: SineSweepSpecification,
                  cycles: float = BASE_CYCLES,
                  windows: int = SAMPLE_WINDOWS,
                  tones: Sequence[str] | None = None,
                  onsets: dict[str, float] | None = None,
                  ticker: Any = None) -> list[dict[str, Any]]:
    """The levels read at `windows` stretches of each tone, cheaply.

    The sine view's preview and the automatic smoothing's measurement:
    rather than solve the whole record, solve one smoothing window's
    worth at each of `windows` evenly spaced points along each tone
    (with the chunked solve's margin around it), and read the
    amplitude and the noise floor at the center. Costs a few short
    solves per channel whatever the record's length, so it can run on
    every edit of the setting.

    Returns, per wanted tone, a dict: 'tone', 'frequency' (the sampled
    points), 'seconds', 'amplitude' and 'floor' (channels × points,
    debiased the way the full extraction is), 'scatter_db' (channels ×
    points, the predicted standard deviation of a reading at this
    smoothing, from the floor against the amplitude; infinite where
    the tone is under the floor), and 'cycles'.
    """
    dt = _even_steps(np.asarray(history.abscissa, dtype=float), None)
    rows = _control_rows(history, specification)
    signals = [history.ordinate[row] for row in rows]
    wanted = (specification.tones if tones is None
              else [specification.tone(name) for name in tones])
    laid = _lay_tones(signals, specification, wanted, onsets, dt, cycles,
                      ticker)
    centers, pieces = [], []
    for placed in laid:
        first = int(placed.window_at(0) / 2.0)
        last = placed.n - 1 - int(placed.window_at(placed.n - 1) / 2.0)
        count = max(1, min(windows, placed.n))
        chosen = (np.linspace(first, last, count).astype(np.int64)
                  if last > first else np.array([placed.n // 2]))
        centers.append(chosen)
        own = []
        for c in chosen:
            half = int(placed.window_at(c) / 2.0) + 1
            own.append((placed.start + c - half, placed.start + c + half + 1))
        pieces.append(own)
    amplitude, floor, below, _slopes = _read_grouped(
        signals, laid, dt, cycles, centers, pieces_by_tone=pieces,
        ticker=ticker)
    out = []
    for k, placed in enumerate(laid):
        seconds = placed.seconds_at(centers[k])
        # a reading under its floor is nothing seen; a weak reading that
        # stood above it is still a reading, and says its own scatter
        magnitude = np.where(below[k], 0.0, np.abs(amplitude[k]))
        out.append({'tone': placed.tone.name,
                    'frequency': placed.tone.frequency_at(seconds),
                    'seconds': placed.onset + centers[k] * dt,
                    'amplitude': magnitude,
                    'floor': floor[k],
                    'scatter_db': scatter_db(magnitude, floor[k]),
                    'cycles': float(cycles)})
    return out


def scatter_db(amplitude: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """The predicted standard deviation of a reading, in dB, from the
    debiased amplitude and the noise floor beside it.

    The envelope is the tone plus a complex noise of power floor²;
    half of that power lies along the tone, so the amplitude's
    standard deviation is floor/√2, and in decibels of the amplitude
    8.686 times their ratio. Infinite where the tone is under the
    floor — there is no reading to scatter around.
    """
    amplitude = np.asarray(amplitude, dtype=float)
    floor = np.asarray(floor, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(amplitude > 0.0, floor / np.sqrt(2.0)
                         / np.maximum(amplitude, 1e-300), np.inf)
    return 20.0 / np.log(10.0) * ratio


def suggest_cycles(history: Any, specification: SineSweepSpecification,
                   target_db: float = TARGET_SCATTER_DB,
                   tones: Sequence[str] | None = None,
                   onsets: dict[str, float] | None = None,
                   ticker: Any = None) -> float:
    """The smoothing the data asks for: the first rung of
    `CYCLES_LADDER` at which the predicted scatter of the readings is
    under `target_db` (one standard deviation).

    Measured at `BASE_CYCLES` with `sample_levels`, where the noise
    the smoothing lets through is read beside the tone it finds; the
    scatter falls as the square root of the smoothing, so the rung
    follows from one measurement. The median over the sampled points
    and channels speaks for a tone — a resonance or a dropout should
    not set the smoothing for a whole sweep — and the widest-asking
    tone speaks for the recording, since one setting rides it. The
    ladder is capped where a tone's smoothing window at its lowest
    frequency would reach a quarter of its own span: past that the
    reading is the sweep's mean, not a level along it. A tone under
    the floor at the base smoothing asks for the top of the ladder.
    """
    sampled = sample_levels(history, specification, BASE_CYCLES,
                            tones=tones, onsets=onsets, ticker=ticker)
    wanted = (specification.tones if tones is None
              else [specification.tone(name) for name in tones])
    asked = BASE_CYCLES
    cap = CYCLES_LADDER[-1]
    for reading, tone in zip(sampled, wanted):
        scatter = reading['scatter_db']
        finite = scatter[np.isfinite(scatter)]
        # the median reading under the floor: the tone was not seen
        # at the base smoothing, and only the longest smoothing can say
        if finite.size < scatter.size / 2.0:
            need = cap
        else:
            # the square-root law, with room: the scatter in decibels
            # grows faster than the noise-to-tone ratio once the noise
            # is a fair fraction of the tone, and the rung the ratio
            # alone chose for a 1.39 dB reading measured 1.25 dB
            # (2026-09-30); half again on the smoothing brings it under
            need = (1.5 * BASE_CYCLES
                    * (float(np.median(finite)) / target_db) ** 2)
        asked = max(asked, need)
        lowest = max(float(np.min(tone.frequency)), 1e-9)
        cap = min(cap, lowest * tone.duration() / 4.0)
    rung = next((c for c in CYCLES_LADDER if c >= asked), CYCLES_LADDER[-1])
    return float(max(BASE_CYCLES, min(rung, cap)))


def _passes(signals, laid, dt, cycles, centers, refine, pool, ticker=None):
    """The solve, and with `refine` the clock checked and the solve
    repeated while it moves, up to three times."""
    for _pass in range(3 if refine else 1):
        amplitude, floor, below, slopes = _read_grouped(
            signals, laid, dt, cycles, centers, pool=pool, ticker=ticker)
        if not refine:
            break
        moved = False
        for k, placed in enumerate(laid):
            c1, c2 = _clock_correction(placed, slopes[k])
            half_span = placed.n * placed.dt / 2.0
            ends = np.array([-half_span, half_span])
            offset_hz = (c1 + 2.0 * c2 * ends) / (2.0 * np.pi)
            f_ends = placed.f(0, 1)[0], placed.f(placed.n - 1, placed.n)[0]
            bandwidth = _AVERAGE_BANDWIDTH * np.array(f_ends) / cycles
            if np.max(np.abs(offset_hz) / bandwidth) > REFINE_FRACTION:
                placed.c1 += c1
                placed.c2 += c2
                placed.drift_hz += float(offset_hz[-1])
                moved = True
        if not moved:
            break
    return amplitude, floor, below, slopes


def extract_sine(history: Any, specification: SineSweepSpecification,
                 tones: Sequence[str] | None = None,
                 onsets: dict[str, float] | None = None,
                 cycles: float | None = None,
                 points_per_window: float = 2.0,
                 refine: bool = True,
                 target_db: float = TARGET_SCATTER_DB,
                 workers: int | None = None,
                 progress: Callable[[int, int], None] | None = None
                 ) -> SineLevelSet:
    """Read each tone's level out of a recording, against its own sweep.

    Reconstruct every wanted tone's sweep from the specification's
    breakpoints, find where the environment's clock begins in the
    recording (joint matched filter; `onsets` overrides per tone name,
    and the found value rides the result as `.onset`), then solve for
    every tone's complex envelope on every control channel at once
    with the second-order **Vold-Kalman filter** (`vold_kalman`): the
    record modeled as the sum of the tones on their known sweeps, each
    envelope held to a slow curve over `cycles` cycles of its own
    instantaneous frequency. Joint, so crossing sweeps are separated
    by their frequency histories rather than each reading the other
    as noise — the documented limit of the tracking demodulation this
    replaced (2026-09-03). Solved one chunk of the record at a time
    (`_read_levels`), so the memory is a chunk's whatever the record's
    length (2026-10-01).

    `cycles` None is **automatic** (`suggest_cycles`): the smoothing
    is climbed until the predicted scatter of the readings is under
    `target_db`, measured from the recording itself. The smoothing
    used rides the result as `SineLevelSet.cycles`.

    The magnitude is *debiased*: a noisy envelope's magnitude reads
    high, so the noise power left in the residual around each
    reading, scaled by the filter's equivalent averaging length, is
    subtracted from the squared magnitude before the square root — a
    planted amplitude under 4x its own RMS of noise reads back within
    a fraction of a dB. Where the tone is under that noise the reading
    is reported at the floor and flagged (`SineLevel.below_floor`)
    rather than at zero. The controller's own live tracker carries the
    raw bias, which is worth remembering when the two are compared.

    With `refine`, the sweep clock is checked against the recording:
    each tone's envelope phase slope is fitted chunk by chunk
    (`_clock_correction`) and, where the implied frequency offset
    reaches `REFINE_FRACTION` of the filter's bandwidth at either end
    of the sweep, the argument is moved by it and the solve repeated,
    up to three times. The offset applied at the end of each tone's
    sweep rides the result as `SineLevel.drift_hz`, zero when none was
    needed.

    `workers` is how many processes the pieces are spread over
    (Brandon, 2026-10-01: use the cores): None picks the cores up to
    `MAX_WORKERS`, and spreads only when the work is more than
    `PARALLEL_FLOOR` samples times channels, since the workers take a
    second each to start; 1 does everything here. The numbers are the
    same either way — each piece and channel is one task, and a task
    is the serial code.

    The envelope is read every window/`points_per_window` along the
    sweep, one (almost) independent reading each. Returns a
    `SineLevelSet` — one object, one `SineLevel` per tone inside,
    frequencies ascending whichever way the tone swept, each line
    stamped with the second it was measured. A recording that ends
    before a tone does yields the lines it reached — the coverage the
    comparison reports.
    """
    dt = _even_steps(np.asarray(history.abscissa, dtype=float), None)
    rows = _control_rows(history, specification)
    signals = [history.ordinate[row] for row in rows]
    ticker = Ticker(progress)
    if cycles is None:
        cycles = suggest_cycles(history, specification, target_db,
                                tones=tones, onsets=onsets, ticker=ticker)
    cycles = float(cycles)
    wanted = (specification.tones if tones is None
              else [specification.tone(name) for name in tones])
    laid = _lay_tones(signals, specification, wanted, onsets, dt, cycles,
                      ticker)
    centers = [_centers(placed, points_per_window) for placed in laid]

    span = max(p.end for p in laid) - min(p.start for p in laid)
    count = _workers(workers)
    if workers is None and span * len(rows) < PARALLEL_FLOOR:
        count = 1
    pool = None
    if count > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # spawned, on every platform: a forked worker inherits the
        # parent's BLAS threads and locks and can hang in them
        pool = ProcessPoolExecutor(
            max_workers=count, mp_context=multiprocessing.get_context('spawn'))
    try:
        amplitude, floor, below, _slopes = _passes(
            signals, laid, dt, cycles, centers, refine, pool, ticker)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    out = []
    for k, placed in enumerate(laid):
        frequencies = placed.tone.frequency_at(placed.seconds_at(centers[k]))
        seconds = placed.onset + centers[k] * dt
        order = np.argsort(frequencies)
        dims = [history.ordinate_dim[row] for row in rows]
        units = [history.ordinate_unit[row] for row in rows]
        out.append(SineLevel(
            abscissa=frequencies[order], ordinate=amplitude[k][:, order],
            response_dof=list(specification.response_dof),
            ordinate_dim=dims, ordinate_unit=units,
            comment=[f'{placed.tone.name} at {dof}'
                     for dof in specification.response_dof],
            tone=placed.tone.name, onset=float(placed.onset),
            seconds=seconds[order],
            floor=floor[k][:, order],
            below_floor=below[k][:, order],
            drift_hz=placed.drift_hz))
    return SineLevelSet(out, cycles=cycles)


# A SineLevel shares Spectrum's function type — dataset 58 has no code
# for an extracted sweep — so the native format names the class
# outright, exactly as the specifications do. Registered here rather
# than in data.py because data.py cannot import this module back.
NAMED_CLASSES['SineLevel'] = SineLevel
NAMED_CLASSES['SineTarget'] = SineTarget
