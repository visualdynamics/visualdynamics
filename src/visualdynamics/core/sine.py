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
    """

    def __init__(self, *args: Any, tone: str = '', onset: float = 0.0,
                 seconds: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tone: str = str(tone)
        self.onset: float = float(onset)
        self.seconds: np.ndarray | None = (
            None if seconds is None
            else np.asarray(seconds, dtype=np.float64))


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
    """

    def __init__(self, levels: Sequence[SineLevel]) -> None:
        self.levels: list[SineLevel] = list(levels)
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


def vold_kalman(signal: np.ndarray, arguments: Sequence[np.ndarray],
                frequencies: Sequence[np.ndarray],
                starts: Sequence[int], dt: float,
                cycles: float = 10.0) -> list[np.ndarray]:
    """Every tone's complex envelope at every sample, solved jointly.

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
    spans = []
    for k in range(K):
        n = min(len(arguments[k]), len(signal) - starts[k])
        spans.append((starts[k], starts[k] + max(n, 0)))
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


def extract_sine(history: Any, specification: SineSweepSpecification,
                 tones: Sequence[str] | None = None,
                 onsets: dict[str, float] | None = None,
                 cycles: float = 10.0,
                 points_per_window: float = 2.0) -> SineLevelSet:
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

    The magnitude is *debiased*: a noisy envelope's magnitude reads
    high, so the noise power left in the residual around each sample,
    scaled by the filter's equivalent averaging length, is subtracted
    from the squared magnitude before the square root — a planted
    amplitude under 4x its own RMS of noise reads back within a
    fraction of a dB. The controller's own live tracker carries the
    raw bias, which is worth remembering when the two are compared.

    The envelope is sampled every window/`points_per_window` along the
    sweep, one (almost) independent reading each. Returns a
    `SineLevelSet` — one object, one `SineLevel` per tone inside,
    frequencies ascending whichever way the tone swept, each line
    stamped with the second it was measured. A recording that ends
    before a tone does yields the lines it reached — the coverage the
    comparison reports.
    """
    abscissa = np.asarray(history.abscissa, dtype=float)
    steps = np.diff(abscissa)
    if len(steps) < 1 or not np.allclose(steps, steps[0], rtol=1e-6):
        raise ValueError('extract_sine needs an evenly sampled time '
                         'history')
    dt = float(steps[0])
    history_dofs = list(history.response_dof)
    rows = []
    missing = []
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
    signals = np.asarray(history.ordinate, dtype=float)[rows]
    wanted = (specification.tones if tones is None
              else [specification.tone(name) for name in tones])
    # one clock for the whole environment: every tone's correlation
    # votes on the shared origin at its own lag, so a near-dwell or a
    # log sweep that would mis-lock alone is pinned by the sweeps
    # beside it. An explicit onset per tone still overrides.
    searching = [tone for tone in wanted
                 if (onsets or {}).get(tone.name) is None]
    origin = (find_environment(signals, dt, searching)
              if searching else 0.0)
    per_tone = []
    for tone in wanted:
        onset = (onsets or {}).get(tone.name)
        if onset is None:
            onset = origin + tone.start_time
        start = round(onset / dt)
        _t, f = tone.trajectory(dt)
        argument = tone.argument(dt)
        n = min(len(argument), signals.shape[1] - start)
        window = np.maximum(cycles / f[:n] / dt, 1.0)
        if n < int(window[0]):
            raise ValueError(
                f'{tone.name}: the recording holds {n * dt:.2f} s of '
                'the tone, less than one smoothing window — nothing '
                'to extract')
        per_tone.append((tone, onset, start, f[:n], argument[:n], window, n))

    # the joint solve, once per channel: every tone's envelope together
    envelopes = [vold_kalman(signals[i],
                             [p[4] for p in per_tone], [p[3] for p in per_tone],
                             [p[2] for p in per_tone], dt, cycles=cycles)
                 for i in range(len(rows))]
    out = []
    for t_index, (tone, onset, start, f, argument, window, n) in enumerate(per_tone):
        half = (window / 2.0).astype(np.int64)
        # sample centers tile the sweep: each one window/points apart,
        # each an (almost) independent reading of the tracked amplitude
        centers = []
        k = int(window[0] / 2.0)
        while k < n - int(window[min(k, n - 1)] / 2.0):
            centers.append(k)
            k += max(int(window[k] / points_per_window), 1)
        centers = np.asarray(centers, dtype=np.int64)
        if not len(centers):
            raise ValueError(f'{tone.name}: no whole smoothing window '
                             'fits the recorded span')
        # the filter's equivalent averaging length in samples, from its
        # noise bandwidth: what a moving average of that length would
        # do to white noise, the debiasing formula's N
        bandwidth = _AVERAGE_BANDWIDTH * f / cycles
        corner = 2.0 * np.pi * bandwidth * dt          # rad/sample
        samples_in = np.maximum(2.0 * np.pi / (corner * _NOISE_INTEGRAL), 1.0)
        rotor = np.exp(-1j * argument)
        ordinate = np.empty((len(rows), len(centers)), dtype=complex)
        for i in range(len(rows)):
            a = envelopes[i][t_index]
            # the residual after every tone is removed — what the
            # envelope's noise was drawn from — measured in the same
            # window the old average used, demodulated like the tone
            model = np.zeros(n)
            for other in range(len(per_tone)):
                o_start, o_n = per_tone[other][2], per_tone[other][6]
                o_arg = per_tone[other][4]
                lo_s, hi_s = max(start, o_start), min(start + n, o_start + o_n)
                if hi_s > lo_s:
                    e = envelopes[i][other][lo_s - o_start:hi_s - o_start]
                    model[lo_s - start:hi_s - start] += np.real(
                        e * np.exp(1j * o_arg[lo_s - o_start:hi_s - o_start]))
            residual = signals[i][start:start + n] - model
            z = residual * rotor
            variance = _smooth(np.abs(z) ** 2 + 0.0j, half).real
            # a = 2*mean of the demodulated tone: the old formula in a's scale
            debiased = np.sqrt(np.maximum(
                np.abs(a) ** 2 - 4.0 * variance / samples_in, 0.0))
            phase = np.exp(1j * np.angle(a))
            ordinate[i] = (debiased * phase)[centers]
        frequencies = f[centers]
        seconds = onset + centers * dt
        order = np.argsort(frequencies)
        dims = [history.ordinate_dim[row] for row in rows]
        units = [history.ordinate_unit[row] for row in rows]
        out.append(SineLevel(
            abscissa=frequencies[order], ordinate=ordinate[:, order],
            response_dof=list(specification.response_dof),
            ordinate_dim=dims, ordinate_unit=units,
            comment=[f'{tone.name} at {dof}'
                     for dof in specification.response_dof],
            tone=tone.name, onset=float(onset),
            seconds=seconds[order]))
    return SineLevelSet(out)


# A SineLevel shares Spectrum's function type — dataset 58 has no code
# for an extracted sweep — so the native format names the class
# outright, exactly as the specifications do. Registered here rather
# than in data.py because data.py cannot import this module back.
NAMED_CLASSES['SineLevel'] = SineLevel
NAMED_CLASSES['SineTarget'] = SineTarget
