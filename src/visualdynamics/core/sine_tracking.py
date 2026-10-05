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
    it would be."""
    starts = np.searchsorted(phase, phase[reads] - 2.0 * np.pi * cycles)
    if detector == 'peak':
        magnitude = np.abs(signal)
        return np.array([magnitude[lo:hi + 1].max()
                         for lo, hi in zip(starts, reads)])
    values = signal ** 2 if detector == 'rms' else np.abs(signal)
    sums = np.concatenate(([0.0], np.cumsum(values)))
    means = (sums[reads + 1] - sums[starts]) / (reads + 1 - starts)
    if detector == 'rms':
        means = np.sqrt(means)
    return means * TO_PEAK[detector]


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
    chosen = (specification.tones[0] if tone is None
              else specification.tone(tone))
    if onset is None:
        signals = [history.ordinate[row] for row in rows]
        onset = (find_environment(signals, dt, [chosen])
                 + chosen.start_time)
    onset = float(onset)

    # the span of the tone the recording holds, on the recording's
    # samples; the law is evaluated at each sample's own time from the
    # sweep's start, so no grid is laid and then interpolated
    start = round(onset / dt)
    length = history.ordinate.shape[1]
    total = int(np.floor(chosen.duration() / dt)) + 1
    first, last = max(start, 0), min(start + total, length)
    if last - first < 2:
        raise ValueError(f'{chosen.name}: the recording holds none of the '
                         'tone')
    seconds = (np.arange(first, last) - start) * dt
    frequency = chosen.frequency_at(seconds)
    phase = chosen.phase_at(seconds)
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
    opening = max(int(np.searchsorted(phase, phase[0] + 2.0 * np.pi)), 2)
    carrier = np.exp(-1j * phase)

    readings = np.zeros((len(settings), len(rows), len(reads)))
    for c, row in enumerate(rows):
        x = np.real(np.asarray(history.ordinate[row, first:last],
                               dtype=float))
        demodulated = settled = image = None
        if any(s.kind != 'unfiltered' for s in settings):
            demodulated = x * carrier
            settled, image = _opening(demodulated[:opening],
                                      phase[:opening])
        for k, setting in enumerate(settings):
            if setting.kind == 'unfiltered':
                readings[k, c] = _detect(x, phase, reads, setting.detector,
                                         setting.cycles)
                continue
            # held at the sample rate: a band wider than that passes
            # everything the record holds, and the cap keeps one
            # sample's decay inside what the recursion's pieces take
            corners = np.minimum(setting.bandwidth(frequency) / 2.0, rate)
            # the factor two: a real tone's demodulated amplitude is
            # half its peak, the other half being the image
            amplitude = 2.0 * _band_passed(
                demodulated, corners, rate, setting.order, settled,
                image, -2.0 * float(frequency[0]))
            if setting.detector == 'filtered':
                readings[k, c] = np.abs(amplitude[reads])
            else:
                passed = np.real(amplitude / carrier)
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
