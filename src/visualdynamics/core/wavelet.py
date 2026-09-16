"""Where a record's frequencies are, moment by moment.

A PSD says what frequencies a record contains and says nothing about
*when*. For a stationary run that is the whole truth and a scalogram
would add nothing. For everything that is not stationary — a rattle
that comes and goes, a shock's ring-down, a resonance that walks as a
structure heats, a sweep passing through a mode — the question is when,
and the spectrum has already averaged the answer away.

The continuous wavelet transform answers it by correlating the record
against a wave packet at many scales. Where a spectrum multiplies the
whole record by one sinusoid, this multiplies it by a short burst slid
along the record, and does that again for a burst of every duration.

**Scale and frequency are one axis, not two.** A wavelet at scale `s`
responds to a band around a frequency that follows from `s` — so a
scalogram has three things in it, time, frequency and magnitude, and
scale is the other name for the frequency axis rather than a third
dimension. This module works in Hz throughout and converts at the edge,
because every other frequency axis in this toolset is in Hz and a scale
number means nothing without the wavelet's centre frequency beside it.

The wavelet is **Morlet**, the standard choice for vibration work: a
Gaussian-windowed complex sinusoid, so it has a magnitude and a phase,
and its time-frequency trade is the best a wavelet can do. `omega0` is
the knob on that trade — how many cycles fit under the window. Low
resolves *when* and blurs *what*; high resolves *what* and blurs *when*.
Six is the conventional default and is very nearly the smallest value
for which the wavelet has no appreciable mean, which is the condition
that makes the transform admissible.

**The normalisation and the frequency mapping are one decision, and
pairing them wrongly biases every frequency this view reports.** Two
conventions are in use. Torrence and Compo (1998) normalise each
wavelet to unit *energy* and give the scale-to-period relation
`4*pi*s / (omega0 + sqrt(2 + omega0**2))`; the other normalises to
unit *amplitude* and pairs with `2*pi*s / omega0`. Both relations are
exact — for their own convention. At `omega0=6` they differ by 1.4%,
so borrowing T&C's relation (which is the one quoted everywhere,
including in this file's first draft) while normalising for amplitude
reports every frequency 1.4% **high** — a 100 Hz tone at 101.4. That is
not scatter; it is a bias, one way, and it is invisible on any
frequency grid coarser than about 1%.

This module normalises for **amplitude**, so it uses `2*pi/omega0`.
Amplitude because this is a units-aware toolset whose user reads g and
N off a picture: under the energy convention the same 2 g tone draws
four times taller at 50 Hz than at 800 Hz, which is right for asking
how variance is distributed and useless for reading a number.
`tests/test_wavelet.py` measures the pairing rather than trusting it —
a tone must be reported at its own frequency, on a grid fine enough
that 1.4% is two lines out.

## What is actually computed, and where it comes from

For a record `x`, one scale at a time and all of them at once in the
frequency domain:

    W(s) = IFFT[ 2 * exp(-(s*omega - omega0)**2 / 2) * H(omega) * X(omega) ]

`X` is the DFT of the record with its mean removed, `H` is the
Heaviside step that keeps only positive frequencies, and `s` is the
scale that tunes the wavelet to the frequency of the row.

Each piece has a source:

- **The Morlet wavelet** is standard, and its frequency-domain form is
  Torrence and Compo (1998), "A Practical Guide to Wavelet Analysis",
  *Bull. Amer. Meteor. Soc.* 79(1), table 1:
  `psi(s*omega) = pi**-0.25 * H(omega) * exp(-(s*omega - omega0)**2 / 2)`.
  Their equation 4 is the transform as a product in the frequency
  domain, which is what makes this affordable — one multiply and one
  inverse per scale instead of a convolution.
- **The `2`** in place of their `pi**-0.25` is the amplitude
  normalisation, and it is derived here rather than taken from
  anywhere. A real tone of amplitude `A` puts `A*N/2` in each half of
  its DFT; the analytic wavelet keeps one half; the tuned daughter's
  own peak is its constant; the inverse divides by `N`. So the constant
  that makes `|W|` read back `A` is exactly 2. Every step of that is
  checked numerically in `tests/test_wavelet.py` — a 2 g tone reads
  2.0 at 50, 200 and 800 Hz alike.
- **The padding** is T&C's too (their section 3f): a transform the
  length of the record wraps, and zeros are what stop the two ends
  meeting.
- **The cone of influence** is their table 1's e-folding time for the
  Morlet, `sqrt(2) * s`.

Written from those published definitions. **No code was read from
sdynpy, rattlesnake or forcefinder** — hard rule 1 in `AGENTS.md`,
which matters here because a structural dynamics library is exactly
the sort of place a wavelet transform lives. Nor from scipy's, which
no longer has one to read: `signal.cwt`, `morlet`, `morlet2` and
`ricker` were deprecated in 1.12 and removed in 1.15, so this is
written out because it has to be, not because rewriting it was
preferred. scipy is still what runs the transforms — `scipy.fft` takes
a `workers` argument that numpy's has nowhere to put, and the inverse
is one independent FFT per scale.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

#: cycles under the Gaussian window. Six is conventional, and is about
#: the smallest for which the Morlet wavelet's mean is negligible — the
#: admissibility condition, below which the transform stops being one.
OMEGA0 = 6.0

#: lines per octave when nothing says otherwise. A scalogram is read as
#: a picture rather than as numbers, and twelve is enough that a ridge
#: looks continuous without computing scales nobody can see apart.
PER_OCTAVE = 12

#: bytes one band of the transform may hold in each of its working
#: arrays. The one-shot form keeps three arrays of (scales, padded
#: length) complex values alive at once — the daughters, their product
#: with the spectrum, and the inverse — and on a five-minute record at
#: 16 kHz that is 121 rows of 4.9M samples, 27 GB, which is how the
#: desktop app died (Brandon, 2026-09-15). Cut into bands of rows that
#: fit this budget, the peak is three of these plus the answer,
#: whatever the record's length; the rows are independent, so the
#: cut costs nothing but the loop. 256 MB is 3.4M samples a row on
#: complex128: a minute at 16 kHz is still one band.
BAND_BYTES = 256 * 2**20

#: columns a picture-sized reading holds time to. A screen is a few
#: thousand pixels across and a surface of this width is half a million
#: vertices at the default grid — the 3-D stage's own budget is ten
#: million — so nothing a reader could see is lost, and a five-minute
#: record arrives as 60 KB rather than 4 GB.
COLUMNS = 4096


def fourier_factor(omega0: float = OMEGA0) -> float:
    """Seconds of Fourier period per second of wavelet scale.

    `2*pi/omega0`, which is the exact relation for the amplitude
    normalisation this module uses: a tone peaks where the wavelet is
    tuned to it, `s*omega == omega0`, because nothing reweights one
    scale against another.

    **Not** Torrence and Compo's `4*pi/(omega0 + sqrt(2 + omega0**2))`,
    which is exact for *their* normalisation — unit energy, whose
    `sqrt(s)` factor tilts the response across scales and moves the
    peak. Their form is the one usually quoted, and taking it while
    normalising for amplitude reports every frequency 1.4% low
    (measured, 2026-08-27). The two are a pair with the normalisation,
    not interchangeable constants.
    """
    return 2.0 * np.pi / omega0


def scale_for(frequency: ArrayLike, omega0: float = OMEGA0) -> np.ndarray:
    """The wavelet scale, in seconds, that listens at these frequencies."""
    return 1.0 / (np.asarray(frequency, dtype=float) * fourier_factor(omega0))


def frequency_for(scale: ArrayLike, omega0: float = OMEGA0) -> np.ndarray:
    """The frequency a wavelet of this scale listens at, in Hz.

    The inverse of `scale_for`, and the reason the view can label its
    axis in Hz: the scale axis and the frequency axis are the same axis
    under a change of variable.
    """
    return 1.0 / (np.asarray(scale, dtype=float) * fourier_factor(omega0))


def log_frequencies(low: float, high: float,
                    per_octave: int = PER_OCTAVE) -> np.ndarray:
    """Frequencies spaced evenly by octave, from `low` to `high`.

    Logarithmic because that is how a wavelet's resolution works: each
    scale has a constant *fractional* bandwidth, so evenly spaced lines
    would crowd together at the top of the range and leave gaps at the
    bottom. It is also how a structural dynamicist reads a frequency
    axis.
    """
    if not 0.0 < float(low) < float(high):
        raise ValueError('the frequency range runs from a positive low '
                         'to a higher high')
    if int(per_octave) < 1:
        raise ValueError('there is at least one line per octave')
    octaves = np.log2(float(high) / float(low))
    count = int(np.ceil(octaves * int(per_octave))) + 1
    return float(low) * 2.0 ** (np.arange(count) / float(per_octave))


def cone_of_influence(frequencies: ArrayLike, sample_rate: float,
                      omega0: float = OMEGA0) -> np.ndarray:
    """Seconds at each end of the record that the edges reach into.

    A wavelet near the start of a record hangs off the end of it, and
    what it correlates against there is the padding rather than the
    measurement. The transform still returns a number and the number
    still looks like data, which is why this is drawn rather than left
    to be remembered — inside the cone, a scalogram is an artefact of
    where the record was cut.

    The e-folding time of the Morlet's Gaussian envelope, `sqrt(2) * s`
    (Torrence and Compo table 1): the amplitude of an edge effect has
    fallen by 1/e² beyond it. Low frequencies use long wavelets, so the
    cone is wide at the bottom of the axis and narrow at the top, and
    the bottom of a short record may be nothing but cone.
    """
    scales = scale_for(frequencies, omega0)
    del sample_rate                      # the cone is in seconds, not samples
    return np.sqrt(2.0) * scales


def scalogram(values: ArrayLike, sample_rate: float,
              frequencies: ArrayLike,
              omega0: float = OMEGA0) -> np.ndarray:
    """The complex wavelet coefficients: one row per frequency.

    Computed in the frequency domain, which is what makes this
    affordable: the transform at every scale is one multiply against
    the record's FFT and one inverse, rather than a convolution per
    scale in time.

    Computed a band of scales at a time (`BAND_BYTES`), because every
    scale at once is three arrays of (scales, padded length) complex
    values and a long record turns that into tens of gigabytes. The
    answer here is still the whole thing — one complex row per
    frequency, as long as the record — which for a long record is
    itself gigabytes; a picture wants `scalogram_peaks`, which reduces
    each band as it lands and never holds the whole.

    Normalised so that **the magnitude is the record's own amplitude**:
    a 2 g tone reads 2 wherever it sits on the frequency axis, so the
    colour bar carries the record's units and a ridge's height means
    one thing everywhere.

    That is a choice, and the other one is defensible. Torrence and
    Compo normalise each wavelet to unit *energy*, which is right for
    asking how a record's variance is distributed and is what a
    geophysicist expects; under it a constant-amplitude tone reads
    proportional to the square root of its period, so the same 2 g at
    50 Hz and at 800 Hz differ by a factor of four on the picture. For
    a units-aware toolset whose user reads amplitudes in g and N, that
    is a scalogram you cannot read a number off. Amplitude it is —
    stated because the two conventions differ by sqrt(scale) and a
    scalogram never says which one it is in.

    Parameters
    ----------
    values : array_like
        One channel's record, evenly sampled. Complex input is refused:
        a measurement is real, and a complex array here is a mistake
        worth catching rather than transforming.
    sample_rate : float
        Samples per second.
    frequencies : array_like
        Where to listen, in Hz. Anything at or above Nyquist is a
        frequency the record cannot carry.
    omega0 : float, default OMEGA0
        Cycles under the wavelet's window: the time-against-frequency
        trade.

    Returns
    -------
    numpy.ndarray
        Complex, shape ``(len(frequencies), len(values))``. Magnitude
        is what a scalogram draws; the phase is kept because it costs
        nothing to keep and a ridge's phase is what an instantaneous
        frequency would be read from.
    """
    record, wanted, samples = _prepared(values, sample_rate, frequencies)
    out = np.empty((wanted.size, samples), dtype=complex)
    for rows, band in _bands(record, sample_rate, wanted, omega0):
        out[rows] = band
    return out


def scalogram_peaks(values: ArrayLike, sample_rate: float,
                    frequencies: ArrayLike, omega0: float = OMEGA0, *,
                    columns: int = COLUMNS) -> tuple[np.ndarray, np.ndarray]:
    """The scalogram's magnitude, held to at most `columns` of time.

    What a picture is drawn from: the view, the scripting plot and the
    report all want the magnitude at about a screen's worth of columns,
    and a million-sample record is not that. Time is cut into equal
    slices and each column is the **largest** magnitude in its slice —
    peak-hold, the same reading `decimate.peak_decimate` gives a curve
    — because a scalogram's story is its ridges and a stride would land
    between the very samples a transient's energy lives in, while a
    mean would halve it. Under the budget nothing is held: the reading
    is the magnitude itself, exactly.

    Computed band by band and reduced as each band lands, so the whole
    transform is never in memory at once: this is what a long record
    goes through, and it costs `BAND_BYTES` a working array whatever
    the length.

    Parameters
    ----------
    values, sample_rate, frequencies, omega0
        As for `scalogram`.
    columns : int
        The most columns to hand back.

    Returns
    -------
    times, magnitude
        `times` is each column's clock in seconds from the record's
        start — the centre of its slice, so the first and last columns
        sit half a slice inside the record's ends — and `magnitude` is
        ``(len(frequencies), len(times))``, real.
    """
    record, wanted, samples = _prepared(values, sample_rate, frequencies)
    step = max(1, -(-samples // int(columns)))
    kept = -(-samples // step)
    held = np.empty((wanted.size, kept), dtype=float)
    # the ragged end is a slice like any other, kept rather than dropped:
    # a burst in the last few samples is still a burst. Padding with
    # zero cannot win a maximum of magnitudes.
    pad = kept * step - samples
    for rows, band in _bands(record, sample_rate, wanted, omega0):
        magnitude = np.abs(band)
        if pad:
            magnitude = np.pad(magnitude, ((0, 0), (0, pad)))
        held[rows] = magnitude.reshape(len(magnitude), kept, step).max(axis=2)
    edges = np.arange(kept + 1) * step
    edges[-1] = samples
    times = (edges[:-1] + edges[1:] - 1) / 2.0 / float(sample_rate)
    return times, held


def _prepared(values, sample_rate, frequencies):
    """The record and the frequencies, checked once for both readings."""
    record = np.asarray(values)
    if np.iscomplexobj(record):
        raise ValueError('a scalogram is of a real record; this one is '
                         'complex')
    record = np.asarray(record, dtype=float)
    if record.ndim != 1:
        raise ValueError('one channel at a time')
    samples = record.size
    if samples < 2:
        raise ValueError('a scalogram needs at least two samples')
    wanted = np.atleast_1d(np.asarray(frequencies, dtype=float))
    nyquist = float(sample_rate) / 2.0
    if wanted.size and (wanted.max() >= nyquist or wanted.min() <= 0.0):
        raise ValueError(
            f'frequencies must lie above zero and below Nyquist '
            f'({nyquist:g} Hz); asked for '
            f'{wanted.min():g} to {wanted.max():g} Hz')
    return record, wanted, samples


def _bands(record, sample_rate, wanted, omega0):
    """The transform, a band of rows at a time: yields (row slice, coefficients).

    Each band's coefficients are already cut back to the record's
    length; the caller stacks them or reduces them as it likes.
    """
    from scipy.fft import fft, ifft, next_fast_len

    samples = record.size
    step = 1.0 / float(sample_rate)
    scales = scale_for(wanted, omega0)

    # **Padded, because a transform the length of the record is a
    # *circular* convolution**: the wavelet that runs off the end comes
    # back on at the beginning, so the loud start of a record is drawn
    # into its silent tail. Measured on a record loud for its first
    # second and silent after: 0.47 of full scale at the end, against
    # 0.004 padded, and 6% of it outside the cone of influence — so the
    # shading did not even cover the lie (2026-08-28).
    #
    # The pad is the cone's own width at the lowest frequency, both
    # ends, which ties the two together: whatever wraparound survives is
    # smaller than what the cone already marks as untrustworthy.
    # `next_fast_len` then rounds up to a length the transform likes,
    # which is free here and is worth having on its own — an awkward
    # record length is 3x slower to transform than the next good one.
    margin = int(np.ceil(np.sqrt(2.0) * scales.max() * sample_rate))
    length = next_fast_len(samples + 2 * margin)

    # the mean is removed because the Morlet is not quite admissible: a
    # DC offset would appear as a bright band across the bottom of every
    # scalogram, which is a picture of the offset and not of the record.
    # Removed before the padding, so the zeros are the same zero the
    # record is now centred on rather than a step down from its mean.
    spectrum = fft(record - record.mean(), n=length, workers=-1)
    # angular frequencies of the FFT bins, negative above Nyquist — the
    # Morlet is analytic, so the negative half is what gets discarded
    omega = 2.0 * np.pi * np.fft.fftfreq(length, step)
    # the analytic half, taken once: the Gaussian below is zero on the
    # negative frequencies, so they need not be raised to it at all
    positive = omega > 0.0
    omega_pos = omega[positive]

    rows_a_band = max(1, BAND_BYTES // (length * 16))
    for first in range(0, scales.size, rows_a_band):
        rows = slice(first, min(first + rows_a_band, scales.size))
        # (band, samples), one row per frequency
        scaled = scales[rows, None] * omega_pos[None, :]
        # The Gaussian, analytic (the negative half of the spectrum is
        # what makes a wavelet complex), times two. The two is the whole
        # normalisation: a real tone of amplitude A puts A/2 in each
        # half of its spectrum, the analytic wavelet keeps one of them,
        # and the peak of the Morlet's own transform is 1 where it is
        # tuned — so twice that reads back A. No sqrt(scale) anywhere,
        # which is exactly the difference from the unit-energy
        # convention.
        product = np.zeros((rows.stop - rows.start, length), dtype=complex)
        product[:, positive] = (
            2.0 * np.exp(-0.5 * (scaled - omega0) ** 2) * spectrum[positive])
        del scaled
        # workers=-1: the inverse is one transform per scale along axis
        # 1 and they do not depend on each other, so this is the one
        # place a thread each is free. Measured 2.7x on a 512 000-sample
        # record at 97 scales, and bit-identical — scipy.fft takes the
        # argument, numpy.fft has nowhere to put it, which is most of
        # why the transform is computed through scipy at all.
        # `overwrite_x`: the product is scaffolding, and letting the
        # inverse work in place is one array fewer a band.
        transformed = ifft(product, axis=1, workers=-1, overwrite_x=True)
        del product
        # back to the record: the pad was scaffolding, not signal
        yield rows, transformed[:, :samples]


def ridge(coefficients: ArrayLike, frequencies: ArrayLike) -> np.ndarray:
    """The strongest frequency at each instant, in Hz.

    What the eye follows across a scalogram, as a number a report can
    quote: for each column, the frequency of the largest magnitude.
    Instants with nothing in them still answer — a ridge is only
    meaningful where there is something to be the ridge of, so a
    caller with a quiet stretch should say so rather than trusting the
    line drawn across it.
    """
    magnitude = np.abs(np.asarray(coefficients))
    wanted = np.asarray(frequencies, dtype=float)
    return wanted[np.argmax(magnitude, axis=0)]


def default_range(sample_rate: float, duration: float) -> tuple[float, float]:
    """The frequency range a record can honestly carry, low to high.

    One implementation for the panel, the scripting plot and the
    report, because the three drifted within a day of each other:
    the view's default top moved from 0.4 of Nyquist to 0.98 when the
    0.4 was measured to be superstition (a tone at 0.95 of Nyquist
    reads its amplitude to 0.1%), and `plot_scalogram` kept the old
    number until this function existed.

    The top is just under Nyquist — under, because the topmost line
    has to lie below it, and the reading is true to a fraction of a
    percent right up against it. The bottom is where a handful of the
    longest wavelets still fit inside the record rather than hanging
    off both ends: a default whose bottom octave is all cone would be
    a default that draws an artefact.
    """
    nyquist = float(sample_rate) / 2.0
    high = nyquist * 0.98
    low = (8.0 / float(duration)) if duration else high / 100.0
    low = min(max(low, nyquist / 1000.0), high / 2.0)
    return low, high
