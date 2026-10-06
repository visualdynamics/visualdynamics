"""A swept sine read the way a controller's tracking filter reads it.

`core.sine.extract_sine` is the best estimate of a tone's level there
is: every tone solved jointly, debiased, its smoothing chosen from the
data. That is the right number for a report and the wrong one for a
question a test floor asks often — *why does the controller's level
disagree with mine?* — because the controller did not read it that
way. It read the response through a tracking filter, a band-pass that
follows the drive frequency, of a fixed width in Hz or a width
proportional to the drive, and turned what came out into a level with
a detector: the peak, the RMS or the mean absolute value (the last two
reported as the peak of the sine they would be), or the filter's own
output. This module does that, so the readings can be laid side by
side and the difference seen rather than argued.

What the readings show, and why the module exists: a harmonic or
broadband noise moves the level by an amount that depends on the
detector and on the filter. Read unfiltered, a peak detector counts
the harmonic's peaks and an RMS detector counts its power; read
through the filter, neither is seen. The filter's price is time: it
settles in roughly one over its bandwidth, so a narrow band lags a
change in level that a wide one follows. Which reading a controller
used is a setting, and the same response record gives a different
level curve for each.

**Demodulation.** The record is multiplied by the complex exponential
of the drive's own phase, which moves the drive frequency to zero; a
low-pass of half the bandwidth there *is* the band-pass about the
drive, and its magnitude is the tone's amplitude. A band-pass run on
the raw record would have to be centered afresh at every frequency of
the sweep; the low-pass only changes its corner.

**Causal, on purpose.** `core.filters.filtered` runs forward and
backward for zero phase, which is right for a record and wrong here:
the point of the reading is the lag a controller's filter has, and a
zero-phase pass would spread it to both sides of a change and hide it.
The filter runs forward only, starting settled on the tone's first
cycle, the way a controller's filter has settled during the ramp-up
before the sweep proper begins.

**One filter, integrated through every sample.** A proportional
band's corner moves with the drive, so no fixed digital design fits
it. The first cut redesigned `core.filters.design`'s Butterworth each
time the corner had moved one percent and carried the state across,
and it measured wrong two ways (2026-10-05). Carried raw, the state of
a narrow section rang at each redesign: 5.7 % of error on a clean log
sweep read at 10 %. Carried as its settled part plus the transient in
flight, the redesigns still came at a steady rate along a log sweep,
and the demodulation's image (below) mixed down against that rate: at
50 % the image got through ten times stronger than the filter allows,
and a step a third the size moved the mixing frequency without curing
the leak. So the filter is the same Butterworth — scipy's analog
prototype, the one `design`'s `butter` maps to a digital filter — split
into its first-order modes, and each mode is integrated exactly across
each sample at the corner that sample has, the input held over the
sample. Nothing is redesigned; a constant input stays constant to
rounding whatever the corner does, because a mode's settled state does
not depend on its corner; and the image gets through at the filter's
own response, measured equal to it. A fixed band is the same code with
a corner that does not move. Against `design`'s bilinear filter at a
fixed corner well below Nyquist it differs by 0.3 % RMS on white
noise, the hold against the bilinear map, which no level reads.

**Fourth order by default, because of the image.** Demodulating a real
record moves the tone to zero and also leaves its mirror image at
twice the drive frequency, which the low-pass attenuates by its
response there, about (bandwidth / (4 x drive)) to the power of the
order; what gets through is a ripple at twice the drive. At second
order that is 1.6 % at 50 % proportional and 6.2 % at 100 %, a
demodulator's artifact rather than a reading; at fourth order, 0.02 %
and 0.4 %. A wide fixed band at the bottom of a sweep — half the
bandwidth near or past the drive frequency — is the case where the
band is no longer about the drive at all, and there the ripple is what
the reading shows.

**A peak is the peak of the samples**, as a sampled detector reads
it: it under-reads a sine by at most cos(pi x drive / sample rate),
0.3 % at a fortieth of the sample rate. RMS and mean read over the
same span, a whole number of the drive's cycles (`DETECTOR_CYCLES`
by default), where a pure sine's RMS times the square root of two and
its mean absolute value times pi/2 are its peak exactly.

They are exact only if the span really is a whole number of cycles
and the waveform between the samples is seen. Read off the samples
alone, neither holds when a cycle is a few samples long: the span
snapped to whole samples, and |x| has a corner at every zero crossing
that a handful of samples cannot place. On a clean 2 m/s**2 tone
sampled at 4096 Hz, one cycle, RMS read 3.6 % off at 400 Hz (10
samples a cycle) and mean 9 % at 750 Hz (5.5) (2026-10-05, found in
review; the tests here run at 51 samples a cycle and never saw it).
So the two are read from the band-limited waveform: the samples
around each span upsampled until a cycle has `FINE_POINTS` points
(`scipy.signal.resample_poly`), and averaged over exactly the span's
phase, its first point interpolated at the span's start. Measured
the same way, a clean sweep to 780 Hz: RMS within 0.005 % and mean
within 0.08 % sampled at 4096 Hz (5.3 samples a cycle at the top),
both within 0.09 % at 2048 Hz (2.6). The reconstruction leans on `MARGIN` samples
either side of a span, so a reading looks that far past its instant:
interpolation, not a lag. The peak stays the peak of the samples, by
design: that is how a sampled controller reads it.

**The waveform, the shape and the weight, from the same filter.** A
level is what a controller reports; the view that explains it draws
the record through the band, the band's magnitude about the drive and
the band's weight on the record behind any instant
(`plot.tracking_filter`). All three come from here, so the picture
cannot describe a different filter from the one that read the level:
`track_waveform` is `track_sine`'s own band run over every sample of
the span, and `SineTracking.response` and `SineTracking.weighting` are
the frequency and impulse responses of the same Butterworth modes the
band is integrated with — at the corner a given drive frequency has,
since a proportional band has a different one at every instant.

**Memory.** One channel's span of the tone is held at a time, with
its phase, the demodulated record, the filter's output and one mode's
working arrays beside it — about a hundred bytes a sample. The extraction streams where this does
not; a recording of tens of millions of samples per channel wants
the extraction's chunking, which this first cut does not have.

References
----------
The tracking filter here is complex demodulation followed by a
Butterworth low-pass, integrated with its input held over each sample;
these are where the three are set out, and the test this reading is
used to understand.

1. Bloomfield, P. (2000). *Fourier Analysis of Time Series: An
   Introduction*, 2nd ed. Wiley. Complex demodulation: a component
   near a known frequency moved to zero and low-passed to read its
   amplitude and phase as they vary.
2. Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal
   Processing*, 3rd ed. Prentice Hall. The Butterworth low-pass and
   its analog prototype.
3. Franklin, G. F., Powell, J. D., & Workman, M. L. (1998). *Digital
   Control of Dynamic Systems*, 3rd ed. Addison-Wesley. A continuous
   system integrated exactly across a sample with its input held: the
   zero-order-hold equivalent each mode is stepped with.
4. IEC 60068-2-6:2007, *Environmental testing - Part 2-6: Tests - Test
   Fc: Vibration (sinusoidal)*. The swept sine test whose drive level a
   tracking filter and detector measure.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from .sine import (
    SineLevel,
    SineSweepSpecification,
    _control_rows,
    _even_steps,
    find_environment,
)

#: the detectors, by the name a setting takes: the filter's own
#: output, and the three that read a waveform over a span of cycles
DETECTORS = ('filtered', 'peak', 'rms', 'mean')

#: what each detector's reading is multiplied by to report the peak of
#: the sine it would be: a sine's RMS is its peak over root two, its
#: mean absolute value its peak times 2/pi. Exact for a pure sine and
#: for nothing else — which is the point of reading a record that is
#: not one through each of them
TO_PEAK = {'filtered': 1.0, 'peak': 1.0, 'rms': float(np.sqrt(2.0)),
           'mean': float(np.pi / 2.0)}

#: each detector in words, for a legend and the levels' comments
DETECTOR_WORDS = {'filtered': 'filter output', 'peak': 'peak',
                  'rms': 'rms as peak', 'mean': 'mean as peak'}

#: the span the peak, RMS and mean detectors read over, in cycles of
#: the drive: one, the shortest span a sine's peak, RMS and mean are
#: exact over
DETECTOR_CYCLES = 1.0

#: how many points a cycle of the drive is given before the RMS and
#: mean detectors read it: their spans are upsampled to at least this
#: (see the module docstring, "exact only if")
FINE_POINTS = 64

#: how many samples either side of a span the band-limited
#: reconstruction reads, which covers `resample_poly`'s filter
MARGIN = 16

#: the reconstruction filter's Kaiser window. `resample_poly`'s own
#: (beta 5) ripples 0.17 % in its pass band, which every RMS and mean
#: read carried as a floor; beta 8 reconstructs a sine within 0.005 %
#: at 20 samples a cycle and 0.02 % at 2.7, where 10 and up give the
#: top of the band away (measured 2026-10-05)
RECONSTRUCTION = ('kaiser', 8.0)

#: the low-pass's design order. Four, because of the demodulation's
#: image — see the module docstring
ORDER = 4

#: how far a mode may decay through one piece of the recursion, in
#: nepers, before the piece is closed and its state carried into the
#: next: the piece's terms grow as e to this, and e**600 is well inside
#: what a double holds with room for the input's own scale
DECAY_LIMIT = 600.0

#: how many readings a tone's span is read at, evenly spaced in time
#: — log-spaced in frequency on a log sweep, linear on a linear one
LINES = 400


@dataclass(frozen=True, kw_only=True)
class SineTracking:
    """How a tone is read: through which band, by which detector.

    `proportional` is the band's width as a fraction of the drive
    frequency (0.5 is 50 %); `fixed` is a width in Hz. Neither is an
    unfiltered reading, where the detector reads the record as it
    stands. Both at once is refused — a band is one or the other — and
    so is the filter's own output with no filter. Which band this is
    follows from which width exists (`kind`), the way
    `core.filters.Filtering` derives low- or high-pass from its edges,
    so no setting can contradict its numbers.

    `detector` is 'filtered' (the band's own output: the demodulated
    amplitude at the instant of the reading), 'peak', 'rms' or 'mean'
    — the last two reported as the peak of the sine they would be
    (`TO_PEAK`). The three waveform detectors read over the last
    `cycles` cycles of the drive; the filter's output ignores it.
    `order` is the low-pass's design order (`ORDER`).

    A proportional band must sit under twice the drive frequency: past
    that, half of it reaches below zero frequency at every point of the
    sweep, and the band is not about the drive anywhere. Frozen like
    `Filtering`, so a setting cannot change under the levels read with
    it.
    """

    detector: str = 'filtered'
    proportional: float | None = None
    fixed: float | None = None
    order: int = ORDER
    cycles: float = DETECTOR_CYCLES

    def __post_init__(self) -> None:
        if self.detector not in DETECTORS:
            raise ValueError(
                f'{self.detector!r} is not a detector: '
                + ', '.join(repr(name) for name in DETECTORS))
        if self.proportional is not None and self.fixed is not None:
            raise ValueError('a tracking band is proportional or fixed, '
                             'not both')
        if self.proportional is not None:
            fraction = float(self.proportional)
            if not 0.0 < fraction < 2.0:
                raise ValueError(
                    f'a proportional band of {fraction:g} of the drive '
                    'frequency must sit between zero and two')
            object.__setattr__(self, 'proportional', fraction)
        if self.fixed is not None:
            width = float(self.fixed)
            if not width > 0.0:
                raise ValueError('a fixed band must be wider than zero')
            object.__setattr__(self, 'fixed', width)
        if self.detector == 'filtered' and self.kind == 'unfiltered':
            raise ValueError("the filter's own output needs a filter: "
                             'give proportional= or fixed=')
        if int(self.order) < 1:
            raise ValueError('the filter needs at least first order')
        object.__setattr__(self, 'order', int(self.order))
        if not float(self.cycles) > 0.0:
            raise ValueError('a detector reads over more than zero cycles')
        object.__setattr__(self, 'cycles', float(self.cycles))

    @property
    def kind(self) -> str:
        """'unfiltered', 'proportional' or 'fixed' — derived from which
        width exists, never stored beside it."""
        if self.proportional is not None:
            return 'proportional'
        if self.fixed is not None:
            return 'fixed'
        return 'unfiltered'

    def bandwidth(self, frequency: Any) -> np.ndarray:
        """The band's full width in Hz at drive frequency `frequency`.

        Parameters
        ----------
        frequency : float or array-like
            The drive frequency, Hz.

        Returns
        -------
        ndarray
            The width, Hz, the shape of `frequency`.
        """
        frequency = np.asarray(frequency, dtype=float)
        if self.proportional is not None:
            return self.proportional * frequency
        if self.fixed is not None:
            return np.full(frequency.shape, self.fixed)
        raise ValueError('an unfiltered reading has no band')

    def settling(self, frequency: Any) -> np.ndarray:
        """Roughly how long the band takes to follow a change of level,
        in seconds: one over its width. A rule of thumb rather than a
        bound — the fourth-order filter reaches half of a step in level
        at about 0.9 of it and nine tenths at about 1.3, the same at
        every width, measured on a step in a swept tone (2026-10-05).

        Parameters
        ----------
        frequency : float or array-like
            The drive frequency, Hz.

        Returns
        -------
        ndarray
            Seconds, the shape of `frequency`.
        """
        return 1.0 / self.bandwidth(frequency)

    def response(self, drive: float, frequencies: Any) -> np.ndarray:
        """The band's magnitude in dB at `frequencies` when the drive is
        at `drive` Hz: the shape the record is read through at that
        instant.

        The low-pass's response at the offset from the drive, which is
        what a band-pass about the drive is after demodulation; −3 dB
        at the drive plus and minus half the bandwidth, by the
        Butterworth's definition. What the image at twice the drive
        adds (the module docstring's ripple) is not in it: that is the
        demodulator's artifact, not the band's shape. The analog
        prototype's response, the one each mode is integrated from;
        the held input moves it by nothing a picture shows this far
        below the sample rate.

        Parameters
        ----------
        drive : float
            The drive frequency, Hz.
        frequencies : float or array-like
            Where to read the band, Hz.

        Returns
        -------
        ndarray
            dB, the shape of `frequencies`; 0 dB at the drive.
        """
        poles, residues = _modes(self.order)
        corner = float(self.bandwidth(float(drive))) / 2.0
        offset = (np.asarray(frequencies, dtype=float) - float(drive)) / corner
        gain = np.zeros(offset.shape, dtype=complex)
        for pole, residue in zip(poles, residues):
            gain += residue / (1j * offset - pole)
        return 20.0 * np.log10(np.maximum(np.abs(gain), 1e-300))

    def weighting(self, drive: float, lags: Any) -> np.ndarray:
        """How much the record `lags` seconds before an instant counts
        in the band's output at that instant, when the drive is at
        `drive` Hz: the low-pass's impulse response, per second.

        It integrates to one (a settled band reads a steady tone at its
        amplitude), rises from zero, peaks and has a small negative
        lobe — the overshoot a fourth order has on a step. Its running
        integral is the band's step response, half way at about 0.9 of
        one over the bandwidth (`settling`). Zero at negative lags: the
        band cannot see ahead.

        Parameters
        ----------
        drive : float
            The drive frequency, Hz, which sets a proportional band's
            corner.
        lags : float or array-like
            Seconds before the instant.

        Returns
        -------
        ndarray
            Per second, the shape of `lags`.
        """
        poles, residues = _modes(self.order)
        corner = np.pi * float(self.bandwidth(float(drive)))   # rad/s
        lags = np.asarray(lags, dtype=float)
        ahead = np.maximum(lags, 0.0)
        total = np.zeros(lags.shape, dtype=complex)
        for pole, residue in zip(poles, residues):
            total += residue * corner * np.exp(pole * corner * ahead)
        return np.where(lags >= 0.0, total.real, 0.0)

    def describe(self) -> str:
        """The reading in words — 'rms as peak, 10 % proportional'. One
        implementation: a legend, the levels' comments and the guide
        all say it this way."""
        if self.proportional is not None:
            band = f'{self.proportional * 100.0:g} % proportional'
        elif self.fixed is not None:
            band = f'{self.fixed:g} Hz fixed'
        else:
            band = 'unfiltered'
        return f'{DETECTOR_WORDS[self.detector]}, {band}'


def _rows(history: Any, specification: SineSweepSpecification,
          channels: Sequence[str] | None) -> list[int]:
    """The records read: the specification's control channels, or the
    named DOFs — any channel of the recording, since a response the
    controller never saw is read through the same filter."""
    if channels is None:
        return _control_rows(history, specification)
    dofs = list(history.response_dof)
    missing = [dof for dof in channels if dof not in dofs]
    if missing:
        raise ValueError(f'the recording carries no record for {missing}')
    return [dofs.index(dof) for dof in channels]


def _modes(order: int) -> tuple[np.ndarray, np.ndarray]:
    """The Butterworth low-pass of unit corner as a sum of first-order
    modes: (poles, residues), so that H(s) = sum r / (s - p).

    The poles are scipy's analog prototype (`buttap`), the same ones
    `core.filters.design` reaches through `butter`; Butterworth poles
    are distinct, so the expansion is plain partial fractions.
    """
    from scipy.signal import buttap

    _zeros, poles, gain = buttap(order)
    residues = np.array([
        gain / np.prod([pole - other for j, other in enumerate(poles)
                        if j != i])
        for i, pole in enumerate(poles)])
    return poles, residues


def _band_passed(demodulated: np.ndarray, corners: np.ndarray,
                 rate: float, order: int, settled: complex,
                 image: complex = 0j, image_hz: float = 0.0) -> np.ndarray:
    """The demodulated record through a causal Butterworth low-pass
    whose corner is `corners`, Hz, at each sample: the tracking band's
    output, as a complex amplitude at every sample.

    Each mode obeys dx/dt = w(t) (p x + r u), w the corner in rad/s;
    held over one sample at that sample's corner it steps exactly as
    x[n] = a x[n-1] + (r/p)(a - 1) u[n], with a = exp(w p dt). The
    first-order recursion is solved in closed form, a cumulative sum
    over the cumulative decay, in pieces short enough that the decay
    across a piece stays under `DECAY_LIMIT` — so it is vectorized
    whatever the corner does, and nothing is redesigned. A mode's
    settled state for a constant input u is -(r/p) u at any corner,
    and for a tone U at Omega rad/s it is w r U / (j Omega - w p); each
    mode starts on both, `settled` the constant and `image` the tone at
    `image_hz` — the opening's own image, so the start carries no
    transient for a clean tone. Started on the constant alone, the
    modes' image responses, which cancel in their sum when settled,
    were missing one by one and decayed at different rates: 1 % at the
    first line of a fast sweep read at 50 % (2026-10-05).
    """
    poles, residues = _modes(order)
    radians = 2.0 * np.pi * np.asarray(corners, dtype=float) / rate
    out = np.zeros(len(demodulated), dtype=complex)
    for pole, residue in zip(poles, residues):
        log_step = radians * pole                 # log of each sample's a
        drive = (residue / pole) * np.expm1(log_step) * demodulated
        w = radians[0] * rate                          # rad/s
        state = (-(residue / pole) * settled
                 + w * residue * image
                 / (2j * np.pi * image_hz - w * pole))
        decay = np.cumsum(-log_step.real)
        cuts = np.searchsorted(decay, np.arange(DECAY_LIMIT, decay[-1],
                                                DECAY_LIMIT))
        bounds = np.unique(np.concatenate(([0], cuts, [len(out)])))
        for lo, hi in pairwise(bounds):
            grown = np.cumsum(log_step[lo:hi])
            piece = np.exp(grown) * (
                state + np.cumsum(np.exp(-grown) * drive[lo:hi]))
            out[lo:hi] += piece
            state = piece[-1]
    return out


def _opening(demodulated: np.ndarray,
             phase: np.ndarray) -> tuple[complex, complex]:
    """Where the filter starts settled: the tone's demodulated amplitude
    over its first cycle, and its image at twice the drive, fitted
    together — (amplitude, image at the first sample).

    The plain mean of the cycle was the first choice and read 2.4 % high
    at the top of a fast descending sweep (2026-10-05): a cycle is
    27.3 samples there, the mean took 28, and the image's partial
    cycle stayed in it. Fitted, the image is a term of its own and a
    clean tone's opening is exact however the cycle falls on the
    samples — and the image is known, so the filter can start settled
    on it too (`_band_passed`).
    """
    basis = np.stack([np.ones(len(phase)), np.exp(-2j * phase)], axis=1)
    fitted, *_rest = np.linalg.lstsq(basis, demodulated, rcond=None)
    return complex(fitted[0]), complex(fitted[1] * basis[0, 1])


def _detect(signal: np.ndarray, phase: np.ndarray, reads: np.ndarray,
            detector: str, cycles: float) -> np.ndarray:
    """A waveform detector's reading at each sample in `reads`, over
    the `cycles` cycles of the drive before it, as the peak of the sine
    it would be. The peak reads the samples; RMS and mean read the
    band-limited waveform over exactly the span (`_span_average`)."""
    starts = np.searchsorted(phase, phase[reads] - 2.0 * np.pi * cycles)
    if detector == 'peak':
        magnitude = np.abs(signal)
        return np.array([magnitude[lo:hi + 1].max()
                         for lo, hi in zip(starts, reads)])
    means = np.array([
        _span_average(signal, phase, hi, phase[hi] - 2.0 * np.pi * cycles,
                      detector)
        for hi in reads])
    if detector == 'rms':
        means = np.sqrt(means)
    return means * TO_PEAK[detector]


def _span_average(signal: np.ndarray, phase: np.ndarray, end: int,
                  start_phase: float, detector: str) -> float:
    """The mean of x**2 ('rms') or |x| ('mean') over the drive's phase
    from `start_phase` to sample `end`, on the waveform between the
    samples as well as at them.

    The samples around the span, `MARGIN` either side, are upsampled
    so a cycle has at least `FINE_POINTS` points; the phase is laid on
    the fine points by interpolation (it is smooth, the drive's own);
    the span's first point is interpolated at `start_phase`; and the
    average is the trapezoid over phase, divided by the phase covered.
    Averaged over phase rather than time so a pure sine reads exactly
    on a sweep too, where a cycle's duration changes along it.
    """
    from scipy.signal import resample_poly

    first = int(np.searchsorted(phase, start_phase))
    lo = max(first - 1 - MARGIN, 0)
    hi = min(end + 1 + MARGIN, len(signal))
    steps = np.diff(phase[lo:hi])
    per_cycle = 2.0 * np.pi / float(np.mean(steps)) if len(steps) else 1.0
    up = max(1, int(np.ceil(FINE_POINTS / per_cycle)))
    if up > 1:
        fine = resample_poly(signal[lo:hi], up, 1,
                             window=RECONSTRUCTION)[:(hi - lo - 1) * up + 1]
        fine_phase = np.interp(np.arange(len(fine)) / up,
                               np.arange(hi - lo), phase[lo:hi])
    else:
        fine, fine_phase = signal[lo:hi], phase[lo:hi]
    stop = (end - lo) * up
    begin = int(np.searchsorted(fine_phase, start_phase))
    if begin == 0:
        xs, ps = fine[:stop + 1], fine_phase[:stop + 1]
    else:
        share = ((start_phase - fine_phase[begin - 1])
                 / (fine_phase[begin] - fine_phase[begin - 1]))
        edge = fine[begin - 1] + share * (fine[begin] - fine[begin - 1])
        xs = np.concatenate(([edge], fine[begin:stop + 1]))
        ps = np.concatenate(([start_phase], fine_phase[begin:stop + 1]))
    values = xs ** 2 if detector == 'rms' else np.abs(xs)
    span = ps[-1] - ps[0]
    return float(np.trapezoid(values, ps) / span) if span > 0 else float(values[-1])


def _tone_at(history: Any, specification: SineSweepSpecification,
             rows: list[int], tone: str | None, onset: float | None,
             dt: float) -> tuple[Any, float]:
    """The tone read, and where its sweep begins in the recording:
    `onset` seconds in, or found by matched filter on the rows read,
    as the extraction finds it."""
    chosen = (specification.tones[0] if tone is None
              else specification.tone(tone))
    if onset is None:
        signals = [history.ordinate[row] for row in rows]
        onset = (find_environment(signals, dt, [chosen])
                 + chosen.start_time)
    return chosen, float(onset)


def _laid(history: Any, chosen: Any, onset: float,
          dt: float) -> tuple[int, int, np.ndarray, np.ndarray]:
    """The span of the tone the recording holds, on the recording's
    samples: (first, last, drive frequency, phase), the law evaluated at
    each sample's own time from the sweep's start, so no grid is laid
    and then interpolated."""
    start = round(onset / dt)
    length = history.ordinate.shape[1]
    total = int(np.floor(chosen.duration() / dt)) + 1
    first, last = max(start, 0), min(start + total, length)
    if last - first < 2:
        raise ValueError(f'{chosen.name}: the recording holds none of the '
                         'tone')
    seconds = (np.arange(first, last) - start) * dt
    return first, last, chosen.frequency_at(seconds), chosen.phase_at(seconds)


class _Demodulated:
    """One channel's span of the tone moved to zero frequency, with
    where the band starts settled on it — worked out once per channel
    and read through as many bands as asked."""

    def __init__(self, x: np.ndarray, phase: np.ndarray) -> None:
        self.carrier: np.ndarray = np.exp(-1j * phase)
        self.record: np.ndarray = x * self.carrier
        opening = max(int(np.searchsorted(phase, phase[0] + 2.0 * np.pi)),
                      2)
        self.settled, self.image = _opening(self.record[:opening],
                                            phase[:opening])

    def through(self, setting: SineTracking, frequency: np.ndarray,
                rate: float) -> np.ndarray:
        """The band's output as a complex amplitude at every sample, in
        the record's units: its magnitude is the 'filtered' reading."""
        # held at the sample rate: a band wider than that passes
        # everything the record holds, and the cap keeps one sample's
        # decay inside what the recursion's pieces take
        corners = np.minimum(setting.bandwidth(frequency) / 2.0, rate)
        # the factor two: a real tone's demodulated amplitude is half
        # its peak, the other half being the image
        return 2.0 * _band_passed(self.record, corners, rate, setting.order,
                                  self.settled, self.image,
                                  -2.0 * float(frequency[0]))

    def waveform(self, amplitude: np.ndarray) -> np.ndarray:
        """The band's output moved back up to the drive: the record as
        the band passes it, the waveform the detectors read."""
        return np.real(amplitude / self.carrier)


def track_sine(history: Any, specification: SineSweepSpecification,
               settings: SineTracking | Sequence[SineTracking], *,
               tone: str | None = None, onset: float | None = None,
               channels: Sequence[str] | None = None,
               lines: int = LINES) -> list[SineLevel]:
    """Read one tone through a tracking filter and detector, one level
    per setting.

    The tone's sweep is rebuilt from the specification and laid on the
    recording where its sweep begins (found by matched filter, as the
    extraction finds it, or at `onset` seconds into the recording);
    each channel is demodulated against the sweep's phase once, and
    each setting reads it: through its band, if it has one, then by
    its detector. Every setting is read at the same `lines` instants,
    evenly spaced through the tone's span from the first instant every
    detector has a whole span behind it, so the levels overlay line for
    line.

    Parameters
    ----------
    history : TimeHistory
        The recording, evenly sampled.
    specification : SineSweepSpecification
        The sweep the drive followed.
    settings : SineTracking or sequence of SineTracking
        The readings to take. Several from one record is the point:
        the same response through different bands and detectors.
    tone : str, optional
        Which tone, by name; the specification's first by default.
    onset : float, optional
        Seconds into the recording where the tone's sweep begins.
        Found by matched filter when omitted.
    channels : sequence of str, optional
        The DOFs to read; the specification's control channels by
        default. Any channel of the recording may be named.
    lines : int
        How many readings along the sweep.

    Returns
    -------
    list of SineLevel
        One per setting, in order: the level against frequency on
        every channel read, ascending in frequency whichever way the
        tone swept, in the recording's own units, each line stamped
        with the second it was read (`SineLevel.seconds`). Each
        channel's comment names the tone and the reading
        (`SineTracking.describe`); the phase is not kept, since a peak
        or an RMS has none.
    """
    if isinstance(settings, SineTracking):
        settings = [settings]
    settings = list(settings)
    if not settings:
        raise ValueError('nothing to read: give at least one setting')
    dt = _even_steps(np.asarray(history.abscissa, dtype=float), None)
    rate = 1.0 / dt
    rows = _rows(history, specification, channels)
    chosen, onset = _tone_at(history, specification, rows, tone, onset, dt)
    first, last, frequency, phase = _laid(history, chosen, onset, dt)
    # phase runs forward whichever way the frequency sweeps, so the
    # detectors find their spans by searching it
    widest = max([setting.cycles for setting in settings
                  if setting.detector != 'filtered'] or [1.0])
    begin = int(np.searchsorted(phase, phase[0] + 2.0 * np.pi * widest))
    if begin >= len(phase) - 1:
        raise ValueError(f'{chosen.name}: the recording holds less than '
                         f'{widest:g} cycles of the tone; nothing to read')
    reads = np.unique(np.linspace(begin, len(phase) - 1,
                                  int(lines)).astype(np.int64))
    order = np.argsort(frequency[reads], kind='stable')

    readings = np.zeros((len(settings), len(rows), len(reads)))
    for c, row in enumerate(rows):
        x = np.real(np.asarray(history.ordinate[row, first:last],
                               dtype=float))
        demodulated = None
        if any(s.kind != 'unfiltered' for s in settings):
            demodulated = _Demodulated(x, phase)
        for k, setting in enumerate(settings):
            if setting.kind == 'unfiltered':
                readings[k, c] = _detect(x, phase, reads, setting.detector,
                                         setting.cycles)
                continue
            amplitude = demodulated.through(setting, frequency, rate)
            if setting.detector == 'filtered':
                readings[k, c] = np.abs(amplitude[reads])
            else:
                passed = demodulated.waveform(amplitude)
                readings[k, c] = _detect(passed, phase, reads,
                                         setting.detector, setting.cycles)

    dims = [history.ordinate_dim[row] for row in rows]
    units = [history.ordinate_unit[row] for row in rows]
    dofs = [history.response_dof[row] for row in rows]
    stamped = (first + reads) * dt
    return [SineLevel(
        abscissa=frequency[reads][order],
        ordinate=readings[k][:, order],
        response_dof=dofs, ordinate_dim=dims, ordinate_unit=units,
        comment=[f'{chosen.name} at {dof}, {setting.describe()}'
                 for dof in dofs],
        tone=chosen.name, onset=onset, seconds=stamped[order])
        for k, setting in enumerate(settings)]


@dataclass(frozen=True, kw_only=True, eq=False)
class TrackedWaveform:
    """One channel's span of a tone as a tracking band passes it,
    sample by sample: what `track_waveform` returns.

    `time` is the record's own clock over the span; `drive` the drive
    frequency at each of those samples; `passed` the record through
    the band, in the record's units — the waveform a detector after
    the band reads — and `level` the band's output amplitude, the
    'filtered' reading at every sample rather than at a reading's
    lines. Not compared by value (arrays do not have one), and frozen
    like the setting it was read with.
    """

    setting: SineTracking
    tone: str
    onset: float
    dof: str
    dimension: str
    unit: str
    time: np.ndarray
    drive: np.ndarray
    passed: np.ndarray
    level: np.ndarray

    def instant(self, frequency: float) -> float:
        """The first second, on the record's clock, at which the drive
        reaches `frequency` — where a cursor goes to look at the band
        as the tone passes a frequency.

        Parameters
        ----------
        frequency : float
            Hz.

        Returns
        -------
        float
            Seconds on the record's clock.
        """
        side = np.sign(self.drive - float(frequency))
        crossed = np.flatnonzero((side[:-1] != side[1:]) | (side[:-1] == 0))
        if not crossed.size:
            raise ValueError(f'the drive never passes {frequency:g} Hz; it '
                             f'runs {self.drive.min():g} to '
                             f'{self.drive.max():g} Hz')
        k = int(crossed[0])
        if side[k] == 0 or self.drive[k + 1] == self.drive[k]:
            return float(self.time[k])
        share = ((float(frequency) - self.drive[k])
                 / (self.drive[k + 1] - self.drive[k]))
        return float(self.time[k] + share * (self.time[k + 1]
                                             - self.time[k]))


def track_waveform(history: Any, specification: SineSweepSpecification,
                   setting: SineTracking, *, tone: str | None = None,
                   onset: float | None = None,
                   channel: str | None = None) -> TrackedWaveform:
    """One channel's span of a tone through a tracking band, at every
    sample: the waveform the band passes and its output amplitude.

    The same band `track_sine` reads its levels through, run the same
    way — the tone laid on the recording where its sweep begins,
    demodulated against its phase, low-passed causally from settled —
    and kept at every sample instead of read at lines, so the band can
    be drawn on the record it was applied to.

    Parameters
    ----------
    history : TimeHistory
        The recording, evenly sampled.
    specification : SineSweepSpecification
        The sweep the drive followed.
    setting : SineTracking
        The band. One with no band is refused: there is nothing to
        pass the record through.
    tone : str, optional
        Which tone; the specification's first by default.
    onset : float, optional
        Seconds into the recording where the tone's sweep begins;
        found by matched filter when omitted.
    channel : str, optional
        The DOF read; the specification's first control channel by
        default. Any channel of the recording may be named.

    Returns
    -------
    TrackedWaveform
        The span's clock, drive frequency, passed waveform and output
        amplitude, in the recording's own units.
    """
    if setting.kind == 'unfiltered':
        raise ValueError(f'{setting.describe()} has no band to pass the '
                         'record through')
    dt = _even_steps(np.asarray(history.abscissa, dtype=float), None)
    rate = 1.0 / dt
    rows = _rows(history, specification,
                 None if channel is None else [channel])[:1]
    chosen, onset = _tone_at(history, specification, rows, tone, onset, dt)
    first, last, frequency, phase = _laid(history, chosen, onset, dt)
    row = rows[0]
    x = np.real(np.asarray(history.ordinate[row, first:last], dtype=float))
    demodulated = _Demodulated(x, phase)
    amplitude = demodulated.through(setting, frequency, rate)
    return TrackedWaveform(
        setting=setting, tone=chosen.name, onset=onset,
        dof=history.response_dof[row],
        dimension=history.ordinate_dim[row],
        unit=history.ordinate_unit[row],
        time=np.asarray(history.abscissa, dtype=float)[first:last],
        drive=frequency, passed=demodulated.waveform(amplitude),
        level=np.abs(amplitude))
