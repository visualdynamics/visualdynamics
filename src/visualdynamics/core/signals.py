"""Generated reference signals: random, burst random, periodic random,
an impulse or classical pulse, sine, and a shock synthesized to a shock
response spectrum.

Each generator returns a `TimeHistory` at the reference DOFs, in the
quantity a specification is the density of (a force, a voltage, an
acceleration), in SI. They are the inputs a response is later computed
from, and the drives a test plan is tried with before the test.

Random signals are shaped to a power spectral density. A frame of N
samples is built in the frequency domain: at line k, a complex amplitude
of magnitude N·sqrt(G_k·df/2) with a random phase, so that the frame's
one-sided PSD is G_k and, by the central limit theorem over many lines,
its samples are Gaussian. DC and the Nyquist line are left at zero. A
stationary record is made of independent frames, each windowed by a
half-cycle sine and overlapped by half a frame: the squares of two
overlapping sine windows sum to one at every sample, so the variance is
the same throughout, there is no seam, and the record does not repeat.
The window smooths the spectrum over about three lines, so a target is
drawn as it would be read with a frame of the same length.

Several DOFs are drawn from a cross-spectral density matrix S at each
line. With S factored as F Fᴴ (its eigenvectors scaled by the square
root of their eigenvalues, so a matrix of lower rank is drawn too), the
amplitudes F·z for independent unit-magnitude random phases z have the
cross spectrum F Fᴴ. This package stores a CPSD record (a, b) as
conj(X_a)·X_b (`TimeHistory.compute_cpsds`), so the matrix factored is
the conjugate of the one stored. Uncorrelated DOFs are the diagonal
matrix, the same rule.

Each record carries the averaging it is meant to be read with, so the
analysis settings opened on it match it: random in frames of the length
it was drawn at, under a Hann window, overlapped by half; burst random in
one rectangular frame per burst, and periodic random in one per period,
so each is one leakage-free average; an impulse or a pulse as the whole
record in one rectangular frame. A sine has none of its own.

An impulse is a half-sine of a chosen width, the force a hammer tip
makes: a pulse with a finite width has a finite bandwidth, where a
single sample is flat to the Nyquist frequency whatever the sample rate.
Its spectrum, |F(f)| = (2PT/π)·|cos(πfT)/(1 - 4f²T²)| for a peak P and
a width T, is flat at low frequency and falls 3 dB by 0.59/T, 10 dB by
1.02/T, and to its first zero at 1.5/T (`impulse_rolloff`), so the width
is chosen for the band the impulse should excite.

A shock specification gives a shock synthesized to its spectrum, a sum
of wavelets a shaker can play, checked against the shaker's shock
ratings (`core.shock_synthesis`).

A specification is read between its breakpoints as it is everywhere in
the package (`author.read_at_lines`: the magnitude log-log, the phase
linearly), or per bin when it holds a controller's lines, and is zero
outside the band it is written on.

References
----------
.. [1] J. S. Bendat and A. G. Piersol, *Random Data: Analysis and
   Measurement Procedures*, 4th ed., Wiley, 2010, ch. 5 and 11.
.. [2] D. O. Smallwood and T. L. Paez, "A frequency domain method for
   the generation of partially coherent normal stationary time domain
   signals", *Shock and Vibration*, vol. 1, no. 1, pp. 45–53, 1993.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

import numpy as np

from ..units import UNKNOWN
from .classical_pulse import ClassicalPulse
from .data import Psd, Specification, Spectrum, TimeHistory
from .shock_synthesis import HALF_CYCLES
from .sine import SineSweepSpecification

#: what follows a density's quantity, which the signal's quantity is
#: the square root of
_DENSITY = '**2/frequency'

#: the share of a burst frame's on-time each cosine taper takes, at
#: either end of the burst, by default
BURST_TAPER = 0.05

#: the longest frame chosen when none is given: a target written down to
#: 0 Hz would otherwise ask for lines without end, and the per-line
#: matrices of a frame of 2**25 took gigabytes (2026-10-24, found when a
#: mutation test ran four at once and the machine ran out of memory)
DEFAULT_FRAME_MAX = 2 ** 18

#: frames made and added at once: a record's frames together are twice
#: its samples, which for ten minutes of eight channels at 51.2 kHz is
#: several gigabytes; drawn in turn, the random stream is the same
FRAMES_AT_ONCE = 32


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    return (seed if isinstance(seed, np.random.Generator)
            else np.random.default_rng(seed))


def _quantity(dimension: str) -> str:
    """The quantity a density is the density of, or a refusal."""
    if dimension == UNKNOWN:
        raise ValueError('the spectrum does not say what it is the density '
                         'of; declare its units first')
    if not dimension.endswith(_DENSITY):
        raise ValueError(f'a signal is drawn from a density, and '
                         f'{dimension!r} is not one')
    return dimension[:-len(_DENSITY)]


def _level_at(spectrum: Psd, record: int, lines: np.ndarray) -> np.ndarray:
    """A record of `spectrum` at `lines`, zero outside its written band."""
    from .author import read_at_lines

    f = np.asarray(spectrum.abscissa, dtype=float)
    values = np.asarray(spectrum.ordinate[record])
    magnitude = np.abs(values)
    written = np.isfinite(magnitude) & (magnitude > 0.0)
    out = np.zeros(len(lines), dtype=values.dtype)
    if not written.any():
        return out
    f, values = f[written], values[written]
    if getattr(spectrum, 'interpolation', 'bin') == 'log_log':
        inside = (lines >= f[0]) & (lines <= f[-1])
        if inside.any():
            out[inside] = read_at_lines(f, values, lines[inside])
        return out
    # per bin: each written line holds the density over its own width,
    # half way to its neighbors (and as far again past the end lines)
    step = np.diff(f)
    edges = np.concatenate([[f[0] - (step[0] if step.size else 0.5) / 2],
                            (f[:-1] + f[1:]) / 2,
                            [f[-1] + (step[-1] if step.size else 0.5) / 2]])
    k = np.searchsorted(edges, lines, side='right') - 1
    inside = (k >= 0) & (k < len(f))
    out[inside] = values[k[inside]]
    return out


def _matrix(spectrum: Psd, dofs: list[str], lines: np.ndarray,
            correlated: bool) -> np.ndarray:
    """The (lines, n, n) target at `lines` over `dofs`, as stored:
    S[a, b] the record (a, b), its missing half the conjugate of the
    other. Without `correlated`, only the diagonal."""
    entries: dict[tuple[str, str], int] = {}
    for i in range(spectrum.num_records):
        entries.setdefault(spectrum.record_pair(i), i)
    n = len(dofs)
    block = np.zeros((len(lines), n, n), dtype=np.complex128)
    for a, da in enumerate(dofs):
        for b, db in enumerate(dofs):
            if a != b and not correlated:
                continue
            i, j = entries.get((da, db)), entries.get((db, da))
            if i is not None:
                block[:, a, b] = _level_at(spectrum, i, lines)
            elif j is not None:
                block[:, a, b] = np.conj(_level_at(spectrum, j, lines))
            elif a == b:
                raise ValueError(f'the spectrum holds no autospectrum at {da}')
    return block


def _factors(block: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """F with F Fᴴ = conj(S) at every line (the conjugate because a
    record (a, b) is conj(X_a)·X_b), or a refusal naming the first line
    whose matrix no set of signals has."""
    diagonal = not np.any(block - np.einsum('kii->ki', block)[..., None]
                          * np.eye(block.shape[1])[None])
    if diagonal:
        autos = np.real(np.einsum('kii->ki', block))
        return np.sqrt(np.clip(autos, 0.0, None))[..., None] \
            * np.eye(block.shape[1])[None]
    from .matrix_checks import judge_lines

    # a matrix no set of signals has is refused, not drawn, on exactly
    # the lines the realizability check fails
    verdict = judge_lines(lines, block, range(block.shape[1]))
    if verdict.failing.any():
        k = int(np.flatnonzero(verdict.failing)[0])
        raise ValueError(
            f'at {lines[k]:.4g} Hz the target is not a matrix any set of '
            f'signals has (its coherence matrix has an eigenvalue of '
            f'{verdict.smallest[k]:.3g} against {verdict.largest[k]:.3g}): '
            'its coherences cannot all hold together')
    values, vectors = np.linalg.eigh(np.conj(block))
    return vectors * np.sqrt(np.clip(values, 0.0, None))[:, None, :]


def _plan(spectrum: Psd, sample_rate: float, frame_length: int | None,
          dofs: Sequence[str] | None, correlated: bool):
    """The DOFs, their quantities, the lines and the per-line factors
    of a random signal."""
    if not sample_rate > 0.0:
        raise ValueError('the sample rate has to be above zero')
    autos = [i for i in range(spectrum.num_records)
             if spectrum.is_autospectrum(i)]
    if not autos:
        raise ValueError('the spectrum holds no autospectrum to draw')
    own = list(dict.fromkeys(spectrum.response_dof[i] for i in autos))
    wanted = own if dofs is None else list(dofs)
    missing = [d for d in wanted if d not in own]
    if missing:
        raise ValueError(f'the spectrum holds no autospectrum at '
                         f'{", ".join(missing)}; it holds {", ".join(own)}')
    quantity = {spectrum.response_dof[i]: _quantity(spectrum.ordinate_dim[i])
                for i in autos}
    written = np.asarray(spectrum.abscissa)[
        np.any(np.isfinite(np.abs(spectrum.ordinate))
               & (np.abs(spectrum.ordinate) > 0), axis=0)]
    if written.size == 0:
        raise ValueError('the spectrum is zero everywhere: nothing to draw')
    nyquist = sample_rate / 2.0
    if written.max() >= nyquist:
        raise ValueError(
            f'the spectrum is written to {written.max():.6g} Hz, at or '
            f'above the Nyquist frequency of {nyquist:.6g} Hz; sample at '
            f'more than {2 * written.max():.6g} Hz')
    if frame_length is None:
        # lines at least as fine as a quarter of the lowest frequency
        # written, and no coarser than 1 Hz: a power of two
        finest = min(1.0, max(written.min(), 1e-3) / 4.0)
        frame_length = min(int(2 ** np.ceil(np.log2(sample_rate / finest))),
                           DEFAULT_FRAME_MAX)
    frame_length = int(frame_length)
    if frame_length < 4 or frame_length % 2:
        raise ValueError(f'a frame of {frame_length} samples: it has to be '
                         'even and at least 4')
    lines = np.arange(frame_length // 2 + 1) * sample_rate / frame_length
    block = _matrix(spectrum, wanted, lines, correlated)
    block[:, :, :][[0, -1]] = 0.0             # no DC, no Nyquist line
    return wanted, [quantity[d] for d in wanted], frame_length, \
        _factors(block, lines)


def _frames(factors: np.ndarray, frame_length: int, sample_rate: float,
            count: int, rng: np.random.Generator) -> np.ndarray:
    """`count` independent frames, shape (count, n, frame_length), each
    with the one-sided cross spectrum F Fᴴ at its lines."""
    lines, n, _ = factors.shape
    df = sample_rate / frame_length
    phases = np.exp(2j * np.pi * rng.random((count, lines, n)))
    amplitudes = np.einsum('kab,mkb->mka', factors, phases)
    amplitudes *= frame_length * np.sqrt(df / 2.0)
    return np.fft.irfft(np.swapaxes(amplitudes, 1, 2), n=frame_length,
                        axis=2)


def _history(rows: np.ndarray, sample_rate: float, dofs: list[str],
             quantities: list[str], comment: str, *,
             frame: int | None = None, overlap: float = 0.0,
             window: str = 'boxcar') -> TimeHistory:
    """The record, with the averaging it was made to be read with: frames
    of `frame` samples (the whole record when None), `overlap` and
    `window`, as many as fit. A signal made in frames is read in the same
    frames, so the analysis settings a user opens on it are the ones that
    match it and not a single frame of the whole record."""
    from ..units import SI
    from .averaging import Averaging

    time = np.arange(rows.shape[1]) / sample_rate
    history = TimeHistory(time, rows, response_dof=dofs,
                          ordinate_dim=quantities,
                          ordinate_unit=[SI.units.get(q) for q in quantities],
                          comment=[comment] * len(dofs))
    history.averaging = Averaging(
        frame_length=int(frame or rows.shape[1]), overlap=overlap,
        window=window).filled(rows.shape[1], sample_rate)
    return history


def _samples(duration: float, sample_rate: float) -> int:
    if not sample_rate > 0.0:
        raise ValueError('the sample rate has to be above zero')
    count = round(duration * sample_rate)
    if count < 2:
        raise ValueError(f'{duration:g} s at {sample_rate:g} Hz is fewer '
                         'than two samples')
    return count


def random_history(spectrum: Psd, *, duration: float, sample_rate: float,
                   frame_length: int | None = None,
                   dofs: Sequence[str] | None = None,
                   correlated: bool = True,
                   seed: int | np.random.Generator | None = None
                   ) -> TimeHistory:
    """Stationary Gaussian random at the reference DOFs, shaped to a PSD
    or a full CPSD.

    Parameters
    ----------
    spectrum : Psd or Specification
        The target, in declared units: an autospectrum per DOF, and the
        cross terms between them for correlated signals.
    duration : float
        Seconds.
    sample_rate : float
        Samples per second, above twice the highest frequency written.
    frame_length : int, optional
        Samples per frame, even, which sets the line spacing the target
        is drawn at; by default a power of two with lines no coarser
        than 1 Hz or a quarter of the lowest frequency written.
    dofs : sequence of str, optional
        Which of the spectrum's DOFs to draw, in this order; all of them
        by default.
    correlated : bool, default True
        Draw the cross terms the spectrum holds. False draws each DOF on
        its own, uncorrelated, whatever cross terms it holds; a spectrum
        of autospectra alone is drawn uncorrelated either way.
    seed : int or numpy.random.Generator, optional
        The same seed gives the same record.

    Returns
    -------
    TimeHistory
        One record per DOF, from time zero.

    Raises
    ------
    ValueError
        If the spectrum is not a density in declared units, holds no
        autospectrum at a DOF asked for, is zero everywhere, is written at
        or above the Nyquist frequency, or is a matrix no set of signals
        has at some line; or if the duration is shorter than two samples.
    """
    total = _samples(duration, sample_rate)
    wanted, quantities, n, factors = _plan(spectrum, sample_rate,
                                           frame_length, dofs, correlated)
    hop = n // 2
    count = int(np.ceil(total / hop)) + 1
    rng = _rng(seed)
    window = np.sin(np.pi * (np.arange(n) + 0.5) / n)
    out = np.zeros((len(wanted), (count + 1) * hop))
    for first in range(0, count, FRAMES_AT_ONCE):
        frames = _frames(factors, n, sample_rate,
                         min(FRAMES_AT_ONCE, count - first), rng)
        for m, frame in enumerate(frames, start=first):
            out[:, m * hop:m * hop + n] += window * frame
    # the first half frame has one window over it, not two; start where
    # every sample has its two
    # read as it was drawn: frames of its own length under a Hann window,
    # overlapped by half
    return _history(out[:, hop:hop + total], sample_rate, wanted, quantities,
                    'random', frame=n, overlap=0.5, window='hann')


def burst_random_history(spectrum: Psd, *, bursts: int, sample_rate: float,
                         frame_length: int, burst: float = 0.5,
                         taper: float = BURST_TAPER,
                         dofs: Sequence[str] | None = None,
                         correlated: bool = True,
                         seed: int | np.random.Generator | None = None
                         ) -> TimeHistory:
    """Burst random: `bursts` frames, each random for the first `burst`
    share of the frame and quiet for the rest.

    The level while a burst is on is the target's; averaged over the
    whole frame the PSD is the target times the share the burst is on
    (less its tapers). Each frame holds one burst, from the frame's
    start, so a capture of the frame's length holds the burst and the
    response's decay, and is read leakage-free with a rectangular window
    once the response has died away.

    Parameters
    ----------
    spectrum : Psd or Specification
        The target, as for `random_history`.
    bursts : int
        How many frames.
    sample_rate : float
        Samples per second.
    frame_length : int
        Samples per frame: the capture length the bursts are aligned to.
    burst : float, default 0.5
        The share of each frame the burst is on, above 0 and at most 1.
    taper : float, default `BURST_TAPER`
        The share of the burst each cosine ramp takes, at either end.
    dofs, correlated, seed
        As for `random_history`.

    Returns
    -------
    TimeHistory
        `bursts` frames end to end.

    Raises
    ------
    ValueError
        As `random_history`, and for a burst share or taper out of range
        or fewer than one burst.
    """
    if int(bursts) < 1:
        raise ValueError('at least one burst')
    if not 0.0 < burst <= 1.0:
        raise ValueError(f'a burst on for {burst:g} of its frame: it has to '
                         'be above 0 and at most 1')
    if not 0.0 <= taper < 0.5:
        raise ValueError(f'a taper of {taper:g} of the burst: it has to be '
                         'at least 0 and under one half')
    wanted, quantities, n, factors = _plan(spectrum, sample_rate,
                                           frame_length, dofs, correlated)
    on = round(burst * n)
    ramp = round(taper * on)
    gate = np.zeros(n)
    gate[:on] = 1.0
    if ramp > 0:
        rise = 0.5 - 0.5 * np.cos(np.pi * (np.arange(ramp) + 0.5) / ramp)
        gate[:ramp] = rise
        gate[on - ramp:on] = rise[::-1]
    rng = _rng(seed)
    rows = np.empty((len(wanted), int(bursts) * n))
    for first in range(0, int(bursts), FRAMES_AT_ONCE):
        frames = _frames(factors, n, sample_rate,
                         min(FRAMES_AT_ONCE, int(bursts) - first), rng)
        for m, frame in enumerate(frames, start=first):
            rows[:, m * n:(m + 1) * n] = frame * gate
    # one rectangular frame per burst: the burst and its quiet tail are
    # whole in each, so nothing leaks and each burst is one average
    return _history(rows, sample_rate, wanted, quantities,
                    f'burst random, {burst:.0%} on', frame=n)


def periodic_random_history(spectrum: Psd, *, periods: int,
                            sample_rate: float, frame_length: int,
                            repeats: int | None = None,
                            dofs: Sequence[str] | None = None,
                            correlated: bool = True,
                            seed: int | np.random.Generator | None = None
                            ) -> TimeHistory:
    """Periodic random: one random frame repeated, so that a frame of
    the same length read with a rectangular window has no leakage.

    With `repeats` omitted the one frame is repeated `periods` times
    (pseudo-random). With `repeats` r, a new frame is drawn every r
    periods (periodic random), so averages over many realizations are
    taken from a record whose every frame is still periodic; the first
    period of each realization carries the change from the last and is
    the one to discard once a response is computed from it.

    Parameters
    ----------
    spectrum : Psd or Specification
        The target, as for `random_history`.
    periods : int
        How many frames in all.
    sample_rate : float
        Samples per second.
    frame_length : int
        Samples per period.
    repeats : int, optional
        Periods per realization; all of them by default.
    dofs, correlated, seed
        As for `random_history`.

    Returns
    -------
    TimeHistory
        `periods` frames end to end.

    Raises
    ------
    ValueError
        As `random_history`, and for fewer than one period or repeat.
    """
    periods = int(periods)
    if periods < 1:
        raise ValueError('at least one period')
    repeats = periods if repeats is None else int(repeats)
    if repeats < 1:
        raise ValueError('at least one period per realization')
    wanted, quantities, n, factors = _plan(spectrum, sample_rate,
                                           frame_length, dofs, correlated)
    count = int(np.ceil(periods / repeats))
    frames = _frames(factors, n, sample_rate, count, _rng(seed))
    rows = np.repeat(frames, repeats, axis=0)[:periods]
    rows = rows.transpose(1, 0, 2).reshape(len(wanted), -1)
    kind = ('pseudo-random' if repeats >= periods
            else f'periodic random, {repeats} periods each')
    # one rectangular frame per period, leakage-free
    return _history(rows, sample_rate, wanted, quantities, kind, frame=n)


def impulse_history(dofs: str | Sequence[str], *, quantity: str,
                    duration: float, sample_rate: float, at: float = 0.0,
                    peak: float = 1.0, width: float = 1e-3,
                    pulse: ClassicalPulse | None = None) -> TimeHistory:
    """An impulse, or a classical pulse, at the same instant at every DOF.

    With no `pulse`, an impulse: a half-sine of `peak` and `width` from
    `at`, whose spectrum rolls off as `impulse_rolloff` gives. With a
    `pulse`, its shape, peak and duration from `at` (`ClassicalPulse.at`).

    Parameters
    ----------
    dofs : str or sequence of str
        The reference DOFs.
    quantity : str
        What the signal is, such as 'force' or 'acceleration'.
    duration : float
        Seconds of record.
    sample_rate : float
        Samples per second.
    at : float, default 0.0
        When the impulse or the pulse starts, in seconds.
    peak : float, default 1.0
        The impulse's peak, signed, in SI; not used with a pulse.
    width : float, default 1e-3
        The impulse's width, the half-sine's half cycle, in seconds; not
        used with a pulse. At least two samples.
    pulse : ClassicalPulse, optional
        A half-sine, sawtooth or trapezoid instead of the impulse.

    Returns
    -------
    TimeHistory
        One identical record per DOF.

    Raises
    ------
    ValueError
        If the start is outside the record or the impulse or pulse runs
        past its end, the peak is not a number other than zero, the
        impulse is narrower than two samples, or the duration is shorter
        than two samples.
    """
    dofs = [dofs] if isinstance(dofs, str) else list(dofs)
    if not dofs:
        raise ValueError('name at least one DOF')
    total = _samples(duration, sample_rate)
    start = round(at * sample_rate)
    if not 0 <= start < total:
        raise ValueError(f'an impulse at {at:g} s is outside the record of '
                         f'{total / sample_rate:g} s')
    if pulse is None:
        # a half-sine, not one sample (Brandon, 2026-10-25): its width
        # sets the band it excites, where a single sample's spectrum is
        # flat to whatever the Nyquist frequency happens to be
        if not width * sample_rate >= 2.0:
            raise ValueError(f'an impulse {1e3 * width:.4g} ms wide is under '
                             f'two samples at {sample_rate:g} Hz; the '
                             f'narrowest is {2e3 / sample_rate:.4g} ms')
        shape = ClassicalPulse('half-sine', peak, width)
        what = f'impulse, {1e3 * width:.4g} ms half-sine'
    else:
        shape, what = pulse, pulse.describe()
    if start + shape.extent * sample_rate > total:
        raise ValueError(f'the {what} from {at:g} s runs past the end of '
                         f'the record at {total / sample_rate:g} s')
    row = shape.at(np.arange(total) / sample_rate - start / sample_rate)
    return _history(np.tile(row, (len(dofs), 1)), sample_rate, dofs,
                    [quantity] * len(dofs), what)


def _half_sine_ratio(x: float) -> float:
    """|F(f)|/F(0) of a half-sine at f·width = x."""
    if abs(1.0 - 4.0 * x * x) < 1e-9:
        return np.pi / 4.0                    # the removable point, x = 1/2
    return abs(np.cos(np.pi * x) / (1.0 - 4.0 * x * x))


def impulse_rolloff(width: float) -> tuple[float, float, float]:
    """Where a half-sine impulse `width` seconds wide rolls off: the
    frequencies, in Hz, at which its spectrum is 3 dB and 10 dB below its
    level at low frequency, and its first zero, 1.5/`width`.

    The ratio |cos(πfT)/(1 - 4f²T²)| falls monotonically to its first
    zero, so each level is solved for once on that span. The rule of
    thumb that a hammer excites up to about one over its contact time is
    the 10 dB point, 1.02/T.
    """
    from scipy.optimize import brentq

    if not width > 0.0:
        raise ValueError(f'the width ({width:g} s) has to be above zero')
    points = [brentq(lambda x, db=db: _half_sine_ratio(x) - 10 ** (-db / 20),
                     1e-9, 1.5 - 1e-9) for db in (3.0, 10.0)]
    return points[0] / width, points[1] / width, 1.5 / width


#: the fewest lines an impulse's spectrum is drawn on
SPECTRUM_LINES = 2 ** 14


def impulse_spectrum(history: TimeHistory, record: int = 0) -> Spectrum:
    """The magnitude of a record's Fourier transform, |X(f)| = |Σ x_n
    e^(-i2πf n Δt)|·Δt, in the record's quantity times seconds (N·s for
    a force): what a transient puts into each frequency, flat at its
    integral at low frequency.

    The record is zero-padded to at least `SPECTRUM_LINES` lines, so a
    short impulse is drawn as a smooth curve and its rolloff and zeros
    can be read off it. Read from the samples, so it is what the record
    holds, sampling included.
    """
    dt = float(np.median(np.diff(history.abscissa)))
    row = np.asarray(history.ordinate[record], dtype=float)
    n = max(2 * SPECTRUM_LINES, 1 << int(np.ceil(np.log2(max(row.size, 2)))))
    magnitude = np.abs(np.fft.rfft(row, n)) * dt
    return Spectrum(np.fft.rfftfreq(n, dt), magnitude[np.newaxis, :],
                    response_dof=[history.response_dof[record]],
                    ordinate_dim=[f'{history.ordinate_dim[record]}*time'],
                    comment=['Fourier transform magnitude'])


def sine_history(specification: SineSweepSpecification, *,
                 sample_rate: float, tone: str | None = None) -> TimeHistory:
    """A sine specification's tone as the signal it describes, at each of
    its control DOFs.

    The phase is the tone's own (`SineTone.argument`), so the frequency
    follows its segments at their rates, dwells included. The amplitude
    and the phase of each DOF are interpolated linearly in time between
    the tone's breakpoints, which is the target the controller follows
    (`SineTone.target`). The record starts at the tone's start.

    Parameters
    ----------
    specification : SineSweepSpecification
        The tones and their control DOFs, in declared units.
    sample_rate : float
        Samples per second, above twice the tone's highest frequency.
    tone : str, optional
        Which tone; the first by default.

    Returns
    -------
    TimeHistory
        One record per control DOF.

    Raises
    ------
    ValueError
        If the tone is not one the specification holds, its units are not
        declared, or it reaches the Nyquist frequency.
    """
    if not specification.tones:
        raise ValueError('the specification holds no tone')
    names = [t.name for t in specification.tones]
    if tone is not None and tone not in names:
        raise ValueError(f'no tone {tone!r}; the tones are {", ".join(names)}')
    chosen = specification.tones[0 if tone is None else names.index(tone)]
    if specification.ordinate_dim == UNKNOWN:
        raise ValueError('the specification does not say what its tones '
                         'are; declare its units first')
    if not sample_rate > 0.0:
        raise ValueError('the sample rate has to be above zero')
    top = float(np.max(chosen.frequency))
    if top >= sample_rate / 2.0:
        raise ValueError(f'{chosen.name} reaches {top:.6g} Hz, at or above '
                         f'the Nyquist frequency of {sample_rate / 2:.6g} Hz')
    argument = chosen.argument(1.0 / sample_rate)
    t = np.arange(len(argument)) / sample_rate
    corners = np.concatenate([[0.0], np.cumsum(chosen.segment_seconds())])
    amplitude = np.asarray(chosen.amplitude, dtype=float)
    phase = (np.zeros_like(amplitude) if chosen.phase is None
             else np.unwrap(np.asarray(chosen.phase, dtype=float), axis=0))
    rows = np.empty((amplitude.shape[1], len(t)))
    for c in range(amplitude.shape[1]):
        size = np.interp(t, corners, amplitude[:, c])
        shift = np.interp(t, corners, phase[:, c])
        rows[c] = size * np.cos(argument + shift)
    quantity = specification.ordinate_dim
    history = _history(rows, sample_rate, list(specification.response_dof),
                       [quantity] * rows.shape[0], f'sine, {chosen.name}')
    # a sweep is read by tracking, not in frames: the record's own
    # default, as any other recording
    history.averaging = None
    return history


def flat_spectrum(dofs: str | Sequence[str], *, quantity: str,
                  band: tuple[float, float], level: float,
                  unit: str | None = None) -> Specification:
    """A flat target over a band at every DOF, uncorrelated, for a
    signal typed in with no specification to read.

    Parameters
    ----------
    dofs : str or sequence of str
        The reference DOFs.
    quantity : str
        What the signal is, such as 'force'; the target is its density.
    band : (float, float)
        Lowest and highest frequency, Hz.
    level : float
        The density across the band, in `unit`.
    unit : str, optional
        The density's unit, such as 'N**2/Hz'; SI when omitted.

    Returns
    -------
    Specification
        Two breakpoints per DOF.
    """
    dofs = [dofs] if isinstance(dofs, str) else list(dofs)
    low, high = float(band[0]), float(band[1])
    if not 0.0 < low < high:
        raise ValueError(f'a band from {low:g} to {high:g} Hz: the lowest '
                         'has to be above zero and below the highest')
    if not np.isfinite(level) or level <= 0.0:
        raise ValueError(f'a level of {level:g}: it has to be above zero')
    levels = np.full((len(dofs), 2), float(level))
    return Specification([low, high], levels, response_dof=dofs,
                         ordinate_dim=quantity + _DENSITY,
                         ordinate_unit=unit)


# ---- one call for every kind: the verb, the preview and the pane ------------

#: every kind of signal, in the order a pane offers them, with its name
KINDS: dict[str, str] = {
    'random': 'Random',
    'burst': 'Burst random',
    'periodic': 'Periodic random',
    'impulse': 'Impulse',
    'pulse': 'Classical pulse',
    'sine': 'Sine',
    'srs': 'SRS synthesis',
}

#: the kinds drawn from a power spectral density
RANDOM_KINDS = ('random', 'burst', 'periodic')

#: the most samples a preview draws per DOF, so that it answers while a
#: setting is being typed; the record made on Apply is the full one
PREVIEW_SAMPLES = 2 ** 17

#: the frame a burst or periodic signal is cut into when none is given
DEFAULT_FRAME = 8192


@dataclass(frozen=True)
class SignalSettings:
    """Everything a generated signal is made from, besides its source.

    One record for every kind; each reads the fields it needs and the
    rest are kept as they were, so switching kinds in a pane and back
    loses nothing. Stored with the object it made (`Project.generate_signal`
    records it, with the seed), so the same signal can be made again.
    """

    #: one of `KINDS`
    kind: str = 'random'
    #: seconds of record, for random, an impulse and a pulse
    duration: float = 10.0
    #: samples per second
    sample_rate: float = 8192.0
    #: samples per frame; random chooses its own when None, burst and
    #: periodic use `DEFAULT_FRAME`
    frame_length: int | None = None
    #: the random seed; None draws one, which the verb then records
    seed: int | None = None
    #: draw the cross terms a spectrum holds
    correlated: bool = True
    #: which DOFs, in order; all the source has when None
    dofs: tuple[str, ...] | None = None
    #: burst random: how many frames, the share on, the taper's share
    bursts: int = 10
    burst: float = 0.5
    taper: float = BURST_TAPER
    #: periodic random: how many periods, and periods per realization
    periods: int = 10
    repeats: int | None = None
    #: an impulse or a pulse: when it starts, in seconds
    at: float = 0.0
    #: an impulse's width, the half-sine's half cycle, in seconds
    width: float = 1e-3
    #: a classical pulse: its shape, its signed peak in SI (an impulse's
    #: peak too), its duration and its edge, in seconds
    shape: str = 'half-sine'
    peak: float = 1.0
    pulse_duration: float = 0.011
    edge: float | None = None
    #: a target typed rather than read: the band, Hz, and the density
    #: across it, in SI, for random from a channel table
    band: tuple[float, float] = (20.0, 2000.0)
    level: float = 1.0
    #: a sine specification's tone; the first when None
    tone: str | None = None
    #: an SRS synthesis: the fewest half cycles each wavelet has, odd
    #: and at least three; more are tried while the shaker cannot hold it
    half_cycles: int = HALF_CYCLES
    #: an SRS synthesis: the shaker's shock stroke peak to peak (m), its
    #: shock velocity (m/s) and force (N), the moving mass and the body's
    #: (kg), as `core.stroke.read_stroke` takes them; None, or zero, is
    #: not judged
    stroke: float | None = None
    velocity: float | None = None
    force: float | None = None
    moving_mass: float | None = None
    body_mass: float | None = None

    def rating(self) -> Any:
        """The shaker's shock ratings these settings give, as a
        `core.shaker.ShakerRating`; a rating at zero is none, as the
        stroke reading takes it."""
        from .shaker import ShakerRating
        from .stroke import _rating

        return ShakerRating(**{name: _rating(getattr(self, name))
                               for name in ('force', 'stroke', 'velocity',
                                            'moving_mass', 'body_mass')})

    def to_dict(self) -> dict[str, Any]:
        """The settings as plain values, for a recipe or a file."""
        out = asdict(self)
        out['band'] = list(self.band)
        if self.dofs is not None:
            out['dofs'] = list(self.dofs)
        return out

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> SignalSettings:
        """Settings from `to_dict`'s values. A field this version does not
        know is left out, and one it knows that is missing keeps its
        default, so a recipe written by a later or an earlier version
        still reads."""
        known = {f.name for f in fields(cls)}
        kept = {k: v for k, v in dict(values).items() if k in known}
        if 'band' in kept:
            kept['band'] = tuple(float(v) for v in kept['band'])
        if kept.get('dofs') is not None:
            kept['dofs'] = tuple(kept['dofs'])
        return cls(**kept)

    def describe(self) -> str:
        """The settings in words, as '10 s of random at 8192 Hz'."""
        what = KINDS.get(self.kind, self.kind).lower()
        if self.kind == 'burst':
            return (f'{self.bursts} bursts of random, {self.burst:.0%} on, '
                    f'at {self.sample_rate:g} Hz')
        if self.kind == 'periodic':
            return f'{self.periods} periods of {what} at {self.sample_rate:g} Hz'
        if self.kind == 'sine':
            return f'sine at {self.sample_rate:g} Hz'
        if self.kind == 'srs':
            return f'SRS synthesis at {self.sample_rate:g} Hz'
        return f'{self.duration:g} s of {what} at {self.sample_rate:g} Hz'


def kinds_for(source: Any) -> tuple[str, ...]:
    """The kinds of signal `source` can make, in `KINDS` order: random
    kinds from a spectrum, sine from a sine specification, an SRS
    synthesis from a shock specification, and from a channel table every
    kind but sine and the synthesis, at its reference channels."""
    from .channel_table import ChannelTable
    from .data import ShockSpecification

    if isinstance(source, SineSweepSpecification):
        return ('sine',)
    if isinstance(source, ShockSpecification):
        return ('srs',)
    if isinstance(source, Psd):
        return RANDOM_KINDS
    if isinstance(source, ChannelTable):
        return (*RANDOM_KINDS, 'impulse', 'pulse')
    return ()


def _references(table: Any, dofs: Sequence[str] | None
                ) -> tuple[list[str], str]:
    """A channel table's reference channels and the one quantity they
    measure, or a refusal naming what is missing."""
    rows = table.chosen('reference')
    if not rows:
        raise ValueError("no channel's role is reference: mark the channels "
                         'the signal drives')
    all_dofs, types = table.dof_strings(), table.types()
    own = [all_dofs[i] for i in rows]
    if '' in own:
        raise ValueError('a reference channel has no node or direction yet')
    wanted = own if dofs is None else list(dofs)
    missing = [d for d in wanted if d not in own]
    if missing:
        raise ValueError(f'{", ".join(missing)} is not a reference channel; '
                         f'the references are {", ".join(own)}')
    quantities = {types[rows[own.index(d)]] for d in wanted}
    if '' in quantities:
        raise ValueError('a reference channel does not say what it measures: '
                         'set its channel type')
    if len(quantities) > 1:
        raise ValueError(f'the references measure {", ".join(sorted(quantities))}; '
                         'a signal is one quantity: pick the DOFs of one')
    return wanted, quantities.pop()


def generate(source: Any, settings: SignalSettings, *,
             progress: Any = None) -> TimeHistory:
    """The signal `settings` describe, made from `source`.

    Parameters
    ----------
    source : Psd, Specification, SineSweepSpecification, ShockSpecification or ChannelTable
        A spectrum's DOFs and shape for random kinds; a sine
        specification's tone; a shock specification's spectrum for an
        SRS synthesis; a channel table's reference channels for an
        impulse, a pulse, or random to the typed `band` and `level`.
    settings : SignalSettings
        The kind and its settings.
    progress : callable, optional
        Called with ``(done, total)`` as an SRS synthesis iterates
        (`core.shock_synthesis.synthesize`); the other kinds are quick
        and do not call it.

    Returns
    -------
    TimeHistory
        One record per DOF.

    Raises
    ------
    ValueError
        If the kind is not one `source` can make (`kinds_for`), or as
        the generator for the kind refuses.
    """
    from .channel_table import ChannelTable

    s = settings
    offered = kinds_for(source)
    if s.kind not in offered:
        made = ', '.join(KINDS[k].lower() for k in offered) or 'no signal'
        raise ValueError(f'{KINDS.get(s.kind, s.kind)} is not made from '
                         f'this object; it makes {made}')
    if s.kind == 'sine':
        return sine_history(source, sample_rate=s.sample_rate, tone=s.tone)
    if s.kind == 'srs':
        from .shock_synthesis import synthesize

        return synthesize(source, sample_rate=s.sample_rate,
                          rating=s.rating(), half_cycles=s.half_cycles,
                          dofs=s.dofs, progress=progress).history
    spectrum, dofs = source, s.dofs
    if isinstance(source, ChannelTable):
        dofs, quantity = _references(source, s.dofs)
        if s.kind == 'impulse':
            return impulse_history(dofs, quantity=quantity,
                                   duration=s.duration,
                                   sample_rate=s.sample_rate, at=s.at,
                                   peak=s.peak, width=s.width)
        if s.kind == 'pulse':
            pulse = ClassicalPulse(s.shape, s.peak, s.pulse_duration, s.edge)
            return impulse_history(dofs, quantity=quantity,
                                   duration=s.duration,
                                   sample_rate=s.sample_rate, at=s.at,
                                   pulse=pulse)
        spectrum = flat_spectrum(dofs, quantity=quantity, band=s.band,
                                 level=s.level)
    common = {'dofs': dofs, 'correlated': s.correlated, 'seed': s.seed}
    if s.kind == 'random':
        return random_history(spectrum, duration=s.duration,
                              sample_rate=s.sample_rate,
                              frame_length=s.frame_length, **common)
    frame = s.frame_length or DEFAULT_FRAME
    if s.kind == 'burst':
        return burst_random_history(spectrum, bursts=s.bursts,
                                    sample_rate=s.sample_rate,
                                    frame_length=frame, burst=s.burst,
                                    taper=s.taper, **common)
    return periodic_random_history(spectrum, periods=s.periods,
                                   sample_rate=s.sample_rate,
                                   frame_length=frame, repeats=s.repeats,
                                   **common)


def target_of(source: Any, settings: SignalSettings) -> Psd | None:
    """The spectrum a random signal is drawn to, for drawing beside its
    own: the source itself, or the typed target at a channel table's
    references; None for the kinds with no spectrum to meet."""
    from .channel_table import ChannelTable

    if settings.kind not in RANDOM_KINDS:
        return None
    if isinstance(source, ChannelTable):
        dofs, quantity = _references(source, settings.dofs)
        return flat_spectrum(dofs, quantity=quantity, band=settings.band,
                             level=settings.level)
    return source


def preview(source: Any, settings: SignalSettings) -> TimeHistory:
    """The signal `settings` describe, cut short to `PREVIEW_SAMPLES` so
    that it answers while a setting is being typed: fewer seconds, bursts
    or periods, every other setting as given. A sine is drawn whole only
    when it fits; otherwise its first `PREVIEW_SAMPLES`. An SRS
    synthesis is iterated whole, as it is made: a shorter one would meet
    a different spectrum."""
    s = settings
    rate = s.sample_rate
    if not rate > 0.0:
        raise ValueError('the sample rate has to be above zero')
    if s.kind == 'srs':
        return generate(source, s)
    if s.kind == 'sine':
        full = generate(source, s)
        if full.ordinate.shape[1] <= PREVIEW_SAMPLES:
            return full
        return TimeHistory(full.abscissa[:PREVIEW_SAMPLES],
                           full.ordinate[:, :PREVIEW_SAMPLES],
                           response_dof=full.response_dof,
                           ordinate_dim=full.ordinate_dim,
                           comment=full.comment)
    frame = s.frame_length or DEFAULT_FRAME
    shorter = replace(
        s,
        duration=min(s.duration, PREVIEW_SAMPLES / rate),
        bursts=max(1, min(s.bursts, PREVIEW_SAMPLES // frame)),
        periods=max(1, min(s.periods, PREVIEW_SAMPLES // frame)))
    return generate(source, shorter)


def frame_used(source: Any, settings: SignalSettings) -> int | None:
    """The frame length the random kinds are drawn with: the setting's,
    or the one `random_history` chooses when it is None; None for the
    kinds that are not drawn in frames."""
    from .channel_table import ChannelTable

    if settings.kind not in RANDOM_KINDS:
        return None
    if settings.kind != 'random':
        return int(settings.frame_length or DEFAULT_FRAME)
    if settings.frame_length:
        return int(settings.frame_length)
    spectrum = target_of(source, settings) if isinstance(source, ChannelTable) \
        else source
    return _plan(spectrum, settings.sample_rate, None, settings.dofs,
                 settings.correlated)[2]


def suggested(source: Any) -> SignalSettings:
    """Settings to open a pane with: the first kind the source makes,
    and a sample rate 2.56 times the highest frequency it holds, as an
    analyzer's span is set, rounded up to a multiple of 1024; for an SRS
    synthesis, `core.shock_synthesis.suggested_rate`."""
    from .channel_table import ChannelTable
    from .data import ShockSpecification

    if isinstance(source, ShockSpecification):
        from .shock_synthesis import suggested_rate

        return SignalSettings(kind='srs', sample_rate=suggested_rate(source))
    kinds = kinds_for(source)
    top = None
    if isinstance(source, SineSweepSpecification):
        top = max(float(np.max(t.frequency)) for t in source.tones) \
            if source.tones else None
    elif isinstance(source, Psd):
        magnitude = np.abs(np.asarray(source.ordinate))
        written = np.asarray(source.abscissa)[np.any(
            np.isfinite(magnitude) & (magnitude > 0), axis=0)]
        top = float(written.max()) if written.size else None
    elif isinstance(source, ChannelTable):
        top = SignalSettings().band[1]
    rate = SignalSettings().sample_rate
    if top:
        rate = float(1024 * int(np.ceil(2.56 * top / 1024.0)))
    return SignalSettings(kind=kinds[0] if kinds else 'random',
                          sample_rate=rate)


def target_at(spectrum: Psd, lines: Any) -> Psd:
    """A spectrum's autospectra read at `lines` the way the random kinds
    draw them (between breakpoints log-log, a controller's lines per
    bin), NaN outside the band each is written on: the line the signal
    is drawn to, to plot beside it."""
    lines = np.asarray(lines, dtype=float)
    autos = [i for i in range(spectrum.num_records)
             if spectrum.is_autospectrum(i)]
    rows = []
    for i in autos:
        level = np.real(_level_at(spectrum, i, lines)).astype(float)
        level[level <= 0.0] = np.nan
        rows.append(level)
    return Psd(lines, np.array(rows), response_dof=[spectrum.response_dof[i]
                                                     for i in autos],
               ordinate_dim=[spectrum.ordinate_dim[i] for i in autos])
