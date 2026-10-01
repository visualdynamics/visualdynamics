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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data import NAMED_CLASSES, Bounded, Spectrum

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
        """The cosine argument over the tone's span: 2*pi*integral(f)."""
        _t, f = self.trajectory(dt)
        return 2.0 * np.pi * np.cumsum(f) * dt

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


def _tone_score(records: np.ndarray, dt: float,
                tone: SineTone) -> np.ndarray:
    """Matched-filter correlation power for one tone at every lag."""
    from scipy.signal import fftconvolve

    template = np.cos(tone.argument(dt))
    if records.shape[1] <= len(template):
        # a recording that stopped mid-sweep still holds the sweep's
        # opening, and the opening aligns it: correlate on the half
        # that fits, leaving the other half as search room. Anything
        # shorter than one second of tone has nothing to lock onto.
        template = template[:records.shape[1] // 2]
        if len(template) < 1.0 / dt:
            raise ValueError(
                f'{tone.name}: the recording '
                f'({records.shape[1] * dt:.2f} s) holds less than a '
                'second of the tone; nothing to align')
    score = np.zeros(records.shape[1] - len(template) + 1)
    for record in records:
        score += fftconvolve(record, template[::-1], mode='valid') ** 2
    return score


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
    records = np.atleast_2d(records)
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
                     tones: Sequence[SineTone]) -> float:
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
    records = np.atleast_2d(records)
    votes = []
    for tone in tones:
        lag = round(tone.start_time / dt)
        score = _tone_score(records, dt, tone)
        if len(score) > lag:
            votes.append(score[lag:])
    if not votes:
        raise ValueError('no tone fits the recording at its own start '
                         'time; nothing to align')
    length = min(len(vote) for vote in votes)
    joint = np.zeros(length)
    for vote in votes:
        joint += vote[:length]
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
#: twenty-four — and `test_extract_sine` pins the interior within 1e-7
#: of the whole-record solve at the smallest chunk. The margin costs
#: little beside the chunk: a few thousand samples against 65 536.
CHUNK = 1 << 16
MARGIN_WINDOWS = 24.0

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
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu

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

    rows, cols, vals = [], [], []
    rhs = np.zeros(N * W)
    sample = np.arange(N)
    # A^T A: one WxW block per sample, c c^T with c = [cos1, -sin1, ...]
    coeff = np.empty((N, W))
    coeff[:, 0::2] = cos.T
    coeff[:, 1::2] = sin.T
    for i in range(W):
        rhs[sample * W + i] = coeff[:, i] * y
        for j in range(W):
            rows.append(sample * W + i); cols.append(sample * W + j)
            vals.append(coeff[:, i] * coeff[:, j])
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
            idx = sample * W + 2 * k + comp
            rows.append(idx); cols.append(idx); vals.append(absent)
            # (x[n-1] - 2x[n] + x[n+1]) for n = 1..N-2, weighted by w[n]
            m = np.arange(1, N - 1)
            wm = w2[m]
            center = m * W + 2 * k + comp
            left, right = center - W, center + W
            for (r, c, v) in ((left, left, wm), (center, center, 4 * wm),
                              (right, right, wm), (left, center, -2 * wm),
                              (center, left, -2 * wm), (center, right, -2 * wm),
                              (right, center, -2 * wm), (left, right, wm),
                              (right, left, wm)):
                rows.append(r); cols.append(c); vals.append(v)
    matrix = coo_matrix((np.concatenate(vals),
                         (np.concatenate(rows), np.concatenate(cols))),
                        shape=(N * W, N * W)).tocsc()
    x = splu(matrix).solve(rhs)
    envelopes = []
    for k, (a, b) in enumerate(spans):
        u = x[(sample * W + 2 * k)][a - lo:b - lo]
        v = x[(sample * W + 2 * k + 1)][a - lo:b - lo]
        envelopes.append(u + 1j * v)
    return envelopes


# ---- reading the levels ---------------------------------------------------


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


class _Tone:
    """One tone laid onto the recording: where it starts, its sweep
    argument and frequency sample by sample, its smoothing window and
    the debiasing length that follow from `cycles`."""

    def __init__(self, tone, onset, start, f, argument, n, dt, cycles):
        self.tone, self.onset, self.start, self.n = tone, onset, start, n
        self.f, self.argument = f[:n], argument[:n]
        self.dt, self.cycles = dt, cycles
        self.window: np.ndarray = np.maximum(cycles / self.f / dt, 1.0)
        # the filter's equivalent averaging length in samples, from its
        # noise bandwidth: what a moving average of that length would
        # do to white noise, the debiasing formula's N
        bandwidth = _AVERAGE_BANDWIDTH * self.f / cycles
        corner = 2.0 * np.pi * bandwidth * dt          # rad/sample
        self.samples_in: np.ndarray = np.maximum(
            2.0 * np.pi / (corner * _NOISE_INTEGRAL), 1.0)
        self.half: np.ndarray = (self.window / 2.0).astype(np.int64)

    def shifted(self, phase):
        """The same tone with `phase` (radians, per sample) added to
        its argument — the clock refinement."""
        out = _Tone.__new__(_Tone)
        out.__dict__.update(self.__dict__)
        out.argument = self.argument + phase
        return out


def _lay_tones(signals, specification, wanted, onsets, dt, cycles):
    """Where every wanted tone sits in the recording, found jointly
    unless given, and each tone laid onto the samples."""
    searching = [tone for tone in wanted
                 if (onsets or {}).get(tone.name) is None]
    origin = (find_environment(signals, dt, searching)
              if searching else 0.0)
    laid = []
    for tone in wanted:
        onset = (onsets or {}).get(tone.name)
        if onset is None:
            onset = origin + tone.start_time
        start = round(onset / dt)
        _t, f = tone.trajectory(dt)
        argument = tone.argument(dt)
        n = min(len(argument), signals.shape[1] - start)
        window = np.maximum(cycles / f[:n] / dt, 1.0) if n else np.zeros(0)
        if n < 1 or n < int(window[0]):
            raise ValueError(
                f'{tone.name}: the recording holds {max(n, 0) * dt:.2f} s '
                'of the tone, less than one smoothing window — nothing '
                'to extract')
        laid.append(_Tone(tone, onset, start, f, argument, n, dt, cycles))
    return laid


def _debias(signal, envelopes, laid, t_index, lo=0, hi=None):
    """(amplitude, floor) at every sample of tone `t_index` on one
    channel, between `lo` and `hi` of its span.

    The magnitude is *debiased*: a noisy envelope's magnitude reads
    high, so the noise power left in the residual around each sample
    — after every tone is removed, demodulated like the tone,
    measured in the same window the old average used — scaled by the
    filter's equivalent averaging length, is subtracted from the
    squared magnitude before the square root. The floor is that noise
    power's square root in the tone's own units: the amplitude a tone
    would need to stand clear of the noise at that reading. Where the
    squared magnitude is under it the tone was not resolved, and the
    amplitude is reported *at* the floor rather than at zero, flagged
    (Brandon, 2026-09-30: a reading of zero is a hole in the curve; a
    reading at the floor says how much was not seen).
    """
    this = laid[t_index]
    hi = this.n if hi is None else hi
    a = envelopes[t_index][lo:hi]
    model = np.zeros(hi - lo)
    for other, that in enumerate(laid):
        lo_s = max(this.start + lo, that.start)
        hi_s = min(this.start + hi, that.start + that.n)
        if hi_s > lo_s:
            e = envelopes[other][lo_s - that.start:hi_s - that.start]
            model[lo_s - this.start - lo:hi_s - this.start - lo] += np.real(
                e * np.exp(1j * that.argument[lo_s - that.start:
                                              hi_s - that.start]))
    residual = signal[this.start + lo:this.start + hi] - model
    z = residual * np.exp(-1j * this.argument[lo:hi])
    variance = _smooth(np.abs(z) ** 2 + 0.0j, this.half[lo:hi]).real
    noise_power = 4.0 * variance / this.samples_in[lo:hi]
    floor = np.sqrt(noise_power)
    power = np.abs(a) ** 2 - noise_power
    return np.sqrt(np.maximum(power, 0.0)), floor


def _solve_channel(signal, laid, dt, cycles):
    return vold_kalman(signal, [t.argument for t in laid],
                       [t.f for t in laid], [t.start for t in laid],
                       dt, cycles=cycles)


def sample_levels(history: Any, specification: SineSweepSpecification,
                  cycles: float = BASE_CYCLES,
                  windows: int = SAMPLE_WINDOWS,
                  tones: Sequence[str] | None = None,
                  onsets: dict[str, float] | None = None
                  ) -> list[dict[str, Any]]:
    """The levels read at `windows` stretches of each tone, cheaply.

    The sine view's preview and the automatic smoothing's measurement:
    rather than solve the whole record, solve one smoothing window's
    worth at each of `windows` evenly spaced points along each tone
    (with the chunked solve's margin around it), and read the
    amplitude and the noise floor at the center. Costs a few short
    solves per channel whatever the record's length, so it can run on
    every edit of the setting.

    Returns, per wanted tone, a dict: 'tone', 'frequency' (the sampled
    points), 'amplitude' and 'floor' (channels × points, debiased the
    way the full extraction is), 'scatter_db' (channels × points, the
    predicted standard deviation of a reading at this smoothing, from
    the floor against the amplitude; infinite where the tone is under
    the floor), and 'cycles'.
    """
    abscissa = np.asarray(history.abscissa, dtype=float)
    dt = _even_steps(abscissa, None)
    rows = _control_rows(history, specification)
    signals = np.asarray(history.ordinate, dtype=float)[rows]
    wanted = (specification.tones if tones is None
              else [specification.tone(name) for name in tones])
    laid = _lay_tones(signals, specification, wanted, onsets, dt, cycles)
    spans = [(t.start, t.start + t.n) for t in laid]
    out = []
    for k, this in enumerate(laid):
        # the sampled centers, each a window clear of the span's ends
        half_first = int(this.window[0] / 2.0)
        half_last = int(this.window[-1] / 2.0)
        first, last = half_first, this.n - 1 - half_last
        count = max(1, min(windows, this.n))
        centers = (np.linspace(first, last, count).astype(np.int64)
                   if last > first else np.array([this.n // 2]))
        amplitude = np.zeros((len(rows), len(centers)))
        floor = np.zeros((len(rows), len(centers)))
        for j, center in enumerate(centers):
            half = int(this.window[center] / 2.0) + 1
            piece_lo = this.start + max(center - half, 0)
            piece_hi = this.start + min(center + half + 1, this.n)
            for i in range(len(rows)):
                solved = _solve_piece(
                    signals[i], [t.argument for t in laid],
                    [t.f for t in laid], spans, dt, cycles,
                    piece_lo, piece_hi)
                # every tone's envelope over the piece, as the full
                # solve would hold it, so the residual is the same
                envelopes = []
                for t_index, that in enumerate(laid):
                    envelope = np.zeros(that.n, dtype=complex)
                    if solved[t_index] is not None:
                        entry, piece = solved[t_index]
                        envelope[entry - that.start:
                                 entry - that.start + len(piece)] = piece
                    envelopes.append(envelope)
                lo_s = max(center - this.half[center], 0)
                hi_s = min(center + this.half[center] + 1, this.n)
                debiased, noise = _debias(signals[i], envelopes, laid, k,
                                          lo_s, hi_s)
                at = center - lo_s
                amplitude[i, j], floor[i, j] = debiased[at], noise[at]
        out.append({'tone': this.tone.name, 'frequency': this.f[centers],
                    'seconds': this.onset + centers * dt,
                    'amplitude': amplitude, 'floor': floor,
                    'scatter_db': scatter_db(amplitude, floor),
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
                   onsets: dict[str, float] | None = None) -> float:
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
                            tones=tones, onsets=onsets)
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
        _t, f = tone.trajectory(1.0 / float(history.sample_rate))
        lowest = max(float(np.min(f)), 1e-9)
        cap = min(cap, lowest * tone.duration() / 4.0)
    rung = next((c for c in CYCLES_LADDER if c >= asked), CYCLES_LADDER[-1])
    return float(max(BASE_CYCLES, min(rung, cap)))


def _fit_drift(envelopes_by_channel, laid, t_index, dt):
    """The sweep clock's drift against the specification, for one
    tone: a quadratic phase (radians against seconds) common to the
    channels, as (c1, c2) — the linear and quadratic coefficients.

    Each channel's envelope phase is unwrapped and fitted by weighted
    least squares, the weight the envelope's power so a stretch where
    the tone is under the noise says nothing; the median across
    channels is the drift, since the structure's own phase differs
    channel to channel and the clock's does not. A smooth structural
    phase trend that survives the median is harmless: moving the
    argument by it rotates the envelope and leaves the magnitude.
    """
    this = laid[t_index]
    t = np.arange(this.n) * dt
    t = t - t.mean()
    fits = []
    for envelopes in envelopes_by_channel:
        a = envelopes[t_index]
        weight = np.abs(a) ** 2
        if not np.any(weight > 0.0):
            continue
        phase = np.unwrap(np.angle(a))
        basis = np.stack([np.ones_like(t), t, t ** 2], axis=1)
        sqrt_w = np.sqrt(weight)[:, None]
        coefficients, *_rest = np.linalg.lstsq(basis * sqrt_w,
                                               phase * sqrt_w[:, 0],
                                               rcond=None)
        fits.append(coefficients[1:])
    if not fits:
        return 0.0, 0.0, t
    c1, c2 = np.median(np.asarray(fits), axis=0)
    return float(c1), float(c2), t


def extract_sine(history: Any, specification: SineSweepSpecification,
                 tones: Sequence[str] | None = None,
                 onsets: dict[str, float] | None = None,
                 cycles: float | None = None,
                 points_per_window: float = 2.0,
                 refine: bool = True,
                 target_db: float = TARGET_SCATTER_DB) -> SineLevelSet:
    """Read each tone's level out of a recording, against its own sweep.

    Reconstruct every wanted tone's sweep argument from the
    specification's breakpoints, find where the environment's clock
    begins in the recording (joint matched filter; `onsets` overrides
    per tone name, and the found value rides the result as `.onset`),
    then solve for every tone's complex envelope on every control
    channel at once with the second-order **Vold-Kalman filter**
    (`vold_kalman`): the record modeled as the sum of the tones on
    their known sweeps, each envelope held to a slow curve over
    `cycles` cycles of its own instantaneous frequency. Joint, so
    crossing sweeps are separated by their frequency histories rather
    than each reading the other as noise — the documented limit of the
    tracking demodulation this replaced (2026-09-03).

    `cycles` None is **automatic** (`suggest_cycles`): the smoothing
    is climbed until the predicted scatter of the readings is under
    `target_db`, measured from the recording itself. The smoothing
    used rides the result as `SineLevelSet.cycles`.

    The magnitude is *debiased* (`_debias`): a noisy envelope's
    magnitude reads high, so the noise power left in the residual
    around each sample, scaled by the filter's equivalent averaging
    length, is subtracted from the squared magnitude before the square
    root — a planted amplitude under 4x its own RMS of noise reads
    back within a fraction of a dB. Where the tone is under that noise
    the reading is reported at the floor and flagged
    (`SineLevel.below_floor`) rather than at zero. The controller's
    own live tracker carries the raw bias, which is worth remembering
    when the two are compared.

    With `refine`, the sweep clock is checked against the recording:
    each tone's residual envelope phase is fitted (`_fit_drift`) and,
    where the implied frequency offset reaches `REFINE_FRACTION` of
    the filter's bandwidth anywhere along the sweep, the argument is
    moved by it and the solve repeated, up to three times. The offset
    applied at the end of each tone's sweep rides the result as
    `SineLevel.drift_hz`, zero when none was needed.

    The envelope is sampled every window/`points_per_window` along the
    sweep, one (almost) independent reading each. Returns a
    `SineLevelSet` — one object, one `SineLevel` per tone inside,
    frequencies ascending whichever way the tone swept, each line
    stamped with the second it was measured. A recording that ends
    before a tone does yields the lines it reached — the coverage the
    comparison reports.
    """
    abscissa = np.asarray(history.abscissa, dtype=float)
    dt = _even_steps(abscissa, None)
    rows = _control_rows(history, specification)
    signals = np.asarray(history.ordinate, dtype=float)[rows]
    if cycles is None:
        cycles = suggest_cycles(history, specification, target_db,
                                tones=tones, onsets=onsets)
    cycles = float(cycles)
    wanted = (specification.tones if tones is None
              else [specification.tone(name) for name in tones])
    laid = _lay_tones(signals, specification, wanted, onsets, dt, cycles)

    # where each tone is read, settled before any solving
    readings = []
    for this in laid:
        # sample centers tile the sweep: each one window/points apart,
        # each an (almost) independent reading of the tracked amplitude
        centers = []
        k = int(this.window[0] / 2.0)
        while k < this.n - int(this.window[min(k, this.n - 1)] / 2.0):
            centers.append(k)
            k += max(int(this.window[k] / points_per_window), 1)
        centers = np.asarray(centers, dtype=np.int64)
        if not len(centers):
            raise ValueError(f'{this.tone.name}: no whole smoothing '
                             'window fits the recorded span')
        readings.append(centers)

    drift = [0.0] * len(laid)
    for _pass in range(3 if refine else 1):
        # the joint solve, one channel at a time: every tone's envelope
        # together, and only this channel's alive — a long run's
        # envelopes for every channel at once were a second way to run
        # out of memory after the solve itself (2026-09-30)
        amplitudes = [np.empty((len(rows), len(c)), dtype=complex)
                      for c in readings]
        floors = [np.empty((len(rows), len(c))) for c in readings]
        flagged = [np.zeros((len(rows), len(c)), dtype=bool) for c in readings]
        fits: list[list[Any]] = []
        for i in range(len(rows)):
            envelopes = _solve_channel(signals[i], laid, dt, cycles)
            if refine:
                fits.append(envelopes)
            for t_index, (this, centers) in enumerate(zip(laid, readings)):
                debiased, floor = _debias(signals[i], envelopes, laid, t_index)
                a = envelopes[t_index]
                below = debiased <= 0.0
                shown = np.where(below, floor, debiased)
                phase = np.exp(1j * np.angle(a))
                amplitudes[t_index][i] = (shown * phase)[centers]
                floors[t_index][i] = floor[centers]
                flagged[t_index][i] = below[centers]
        if not refine:
            break
        # the clock against the specification: move the argument where
        # the drift would pull the reading down, and solve again
        moved = False
        for t_index, this in enumerate(laid):
            c1, c2, t = _fit_drift(fits, laid, t_index, dt)
            offset_hz = (c1 + 2.0 * c2 * t) / (2.0 * np.pi)
            bandwidth = _AVERAGE_BANDWIDTH * this.f / cycles
            if np.max(np.abs(offset_hz) / bandwidth) > REFINE_FRACTION:
                laid[t_index] = this.shifted(c1 * t + c2 * t ** 2)
                drift[t_index] += float(offset_hz[-1])
                moved = True
        del fits
        if not moved:
            break

    out = []
    for t_index, (this, centers) in enumerate(zip(laid, readings)):
        frequencies = this.f[centers]
        seconds = this.onset + centers * dt
        order = np.argsort(frequencies)
        dims = [history.ordinate_dim[row] for row in rows]
        units = [history.ordinate_unit[row] for row in rows]
        out.append(SineLevel(
            abscissa=frequencies[order], ordinate=amplitudes[t_index][:, order],
            response_dof=list(specification.response_dof),
            ordinate_dim=dims, ordinate_unit=units,
            comment=[f'{this.tone.name} at {dof}'
                     for dof in specification.response_dof],
            tone=this.tone.name, onset=float(this.onset),
            seconds=seconds[order],
            floor=floors[t_index][:, order],
            below_floor=flagged[t_index][:, order],
            drift_hz=drift[t_index]))
    return SineLevelSet(out, cycles=cycles)


# A SineLevel shares Spectrum's function type — dataset 58 has no code
# for an extracted sweep — so the native format names the class
# outright, exactly as the specifications do. Registered here rather
# than in data.py because data.py cannot import this module back.
NAMED_CLASSES['SineLevel'] = SineLevel
NAMED_CLASSES['SineTarget'] = SineTarget
