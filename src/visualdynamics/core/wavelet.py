"""The continuous wavelet transform: when each frequency occurs in a record.

A PSD gives the frequencies a record contains but not when they occur.
For a stationary record that is all there is to know. For a record that
is not stationary (a rattle that comes and goes, a shock's ring-down, a
resonance that shifts as a structure heats, a sweep passing through a
mode) the time is the question, and the spectrum has averaged it out.

The continuous wavelet transform correlates the record against a short
wave packet at many scales. A Fourier transform multiplies the whole
record by one sinusoid; the wavelet transform multiplies it by a short
burst moved along the record, and repeats that for bursts of many
durations.

Scale and frequency are the same axis. A wavelet at scale `s` responds
to a band around a frequency that follows from `s`, so a scalogram has
three quantities in it (time, frequency and magnitude), and scale is
another name for the frequency axis. This module works in Hz throughout
and converts to scale internally, because every other frequency axis in
Visual Dynamics is in Hz and a scale means nothing without the
wavelet's center frequency beside it.

The wavelet is the Morlet, the standard choice for vibration work: a
Gaussian-windowed complex sinusoid, so each coefficient has a magnitude
and a phase. `omega0`, the number of cycles under the window, sets the
trade between time and frequency resolution. A low value resolves when
something happens and blurs its frequency; a high value resolves the
frequency and blurs the time. Six is the conventional default and is
very nearly the smallest value for which the wavelet has a negligible
mean, which is the condition for the transform to be admissible.

The normalization and the scale-to-frequency relation have to be chosen
together. Two conventions are in use. Torrence and Compo (1998)
normalize each wavelet to unit energy and give the scale-to-period
relation `4*pi*s / (omega0 + sqrt(2 + omega0**2))`. The other
convention normalizes to unit amplitude and uses `2*pi*s / omega0`.
Each relation is exact for its own normalization. At `omega0=6` they
differ by 1.4%, so using Torrence and Compo's relation with the
amplitude normalization reports every frequency 1.4% high: a 100 Hz
tone at 101.4 Hz. The error is a bias in one direction, and it cannot
be seen on a frequency grid coarser than about 1%.

This module normalizes for amplitude and uses `2*pi*s / omega0`.
Amplitude, because the user reads g and N off the picture: under the
energy normalization the same 2 g tone is drawn four times taller at
50 Hz than at 800 Hz, which is correct for asking how the variance is
distributed and of no use for reading a value. `tests/test_wavelet.py`
checks the pairing by measuring it: a tone must be reported at its own
frequency, on a grid fine enough that 1.4% is two lines.

## What is computed

For a record `x`, at every scale, in the frequency domain:

    W(s) = IFFT[ 2 * exp(-(s*omega - omega0)**2 / 2) * H(omega) * X(omega) ]

`X` is the DFT of the zero-padded record with its mean removed, `H` is
the Heaviside step that keeps only positive frequencies, and `s` is the
scale that tunes the wavelet to the frequency of the row.

Where each piece comes from:

- The Morlet wavelet is standard, and its frequency-domain form is
  from Torrence and Compo (1998):
  `psi(s*omega) = pi**-0.25 * H(omega) * exp(-(s*omega - omega0)**2 / 2)`.
  They also write the transform as a product in the frequency domain,
  which keeps the cost down: one multiply and one inverse FFT per scale
  instead of a convolution.
- The factor `2` in place of their `pi**-0.25` is the amplitude
  normalization, derived here. A real tone of amplitude `A` puts
  `A*N/2` in each half of its DFT; the analytic wavelet keeps one half;
  the peak of the tuned wavelet is its constant; the inverse divides by
  `N`. So the constant that makes `|W|` equal `A` is exactly 2.
  `tests/test_wavelet.py` checks this numerically: a 2 g tone reads 2.0
  at 50, 200 and 800 Hz.
- The zero padding is also from Torrence and Compo. A transform the
  length of the record wraps around, and the zeros keep the two ends
  of the record apart.
- The cone of influence is Torrence and Compo's e-folding time for the
  Morlet, `sqrt(2) * s`.

The FFTs are computed with `scipy.fft`, which takes a `workers`
argument, and the inverse is one independent FFT per scale.

References
----------
The Morlet definition, the scale-to-frequency relation and the cone of
influence are Torrence and Compo's.

1. Torrence, C., & Compo, G. P. (1998). "A Practical Guide to Wavelet
   Analysis." *Bulletin of the American Meteorological Society*.
2. Morlet, J., Arens, G., Fourgeau, E., & Giard, D. (1982). "Wave
   propagation and sampling theory." *Geophysics*. The wavelet
   itself.
3. Mallat, S. (2009). *A Wavelet Tour of Signal Processing: The Sparse
   Way*, 3rd ed. Academic Press. Admissibility, and why the Morlet
   needs its correction term at low center frequency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

# Written from the published definitions above. No code was read from
# sdynpy, rattlesnake or forcefinder (hard rule 1 in AGENTS.md), which
# matters here because a structural dynamics library is exactly where a
# wavelet transform would be found. Nor from scipy's, which no longer
# has one to read: `signal.cwt`, `morlet`, `morlet2` and `ricker` were
# deprecated in 1.12 and removed in 1.15, so this is written out
# because it has to be. scipy still runs the transforms, because
# `scipy.fft` takes a `workers` argument that numpy's has nowhere to
# put.
#
# Torrence and Compo's scale-to-period relation is the one quoted
# everywhere, and this file's first draft used it with the amplitude
# normalization, which is the 1.4% bias the module docstring describes.
# Their paper is in Bull. Amer. Meteor. Soc. 79(1); the frequency-domain
# Morlet and the e-folding time are in its table 1, the transform as a
# product is its equation 4 and the padding is its section 3f.

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
    """Seconds of Fourier period per second of wavelet scale, `2*pi/omega0`.

    This is the exact relation for the amplitude normalization this
    module uses. A tone peaks where the wavelet is tuned to it,
    `s*omega == omega0`, because no factor weights one scale against
    another.

    Torrence and Compo's `4*pi/(omega0 + sqrt(2 + omega0**2))` is exact
    for their unit-energy normalization, whose `sqrt(s)` factor tilts
    the response across scales and moves the peak. Using it with the
    amplitude normalization reports every frequency 1.4% high: a 100 Hz
    tone at 101.37 Hz. Each relation belongs with its own normalization.
    """
    # The 1.4% was measured 2026-08-27, and again 2026-09-23, when this
    # docstring was found saying "low".
    return 2.0 * np.pi / omega0


def scale_for(frequency: ArrayLike, omega0: float = OMEGA0) -> np.ndarray:
    """The wavelet scale, in seconds, tuned to each frequency in Hz."""
    return 1.0 / (np.asarray(frequency, dtype=float) * fourier_factor(omega0))


def frequency_for(scale: ArrayLike, omega0: float = OMEGA0) -> np.ndarray:
    """The frequency, in Hz, that a wavelet of each scale is tuned to.

    This is the inverse of `scale_for`. The scale axis and the frequency
    axis are one axis under a change of variable, which is why the view
    can label its axis in Hz.
    """
    return 1.0 / (np.asarray(scale, dtype=float) * fourier_factor(omega0))


def log_frequencies(low: float, high: float,
                    per_octave: int = PER_OCTAVE) -> np.ndarray:
    """Frequencies spaced evenly in octaves, `per_octave` lines to the octave.

    The first line is `low` and the last is the first line at or above
    `high`, so the top line can be slightly above `high`. Raises
    `ValueError` unless `0 < low < high` and `per_octave` is at least
    one.

    The spacing is logarithmic because each scale has a constant
    fractional bandwidth: evenly spaced lines would crowd together at
    the top of the range and leave gaps at the bottom. It is also how a
    structural dynamics engineer reads a frequency axis.
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
    """The time, in seconds, at each end of the record that edge effects reach.

    One value per frequency. A wavelet near the start of a record
    extends past the start, and there it is correlated against the
    padding instead of the measurement. The transform still returns a
    value and the value still looks like data, so the cone is drawn on
    the picture; inside it, the scalogram is an artifact of where the
    record was cut.

    The width is the e-folding time of the Morlet's Gaussian envelope,
    `sqrt(2) * s` (Torrence and Compo): beyond it, the power of an edge
    effect has fallen by a factor of e² (its amplitude by e). Low
    frequencies use long wavelets, so the cone is wide at the bottom of
    the axis and narrow at the top, and at the bottom of a short record
    the cone can cover the whole record. `sample_rate` is accepted and
    not used, since the cone is in seconds.
    """
    scales = scale_for(frequencies, omega0)
    del sample_rate                      # the cone is in seconds, not samples
    return np.sqrt(2.0) * scales


def scalogram(values: ArrayLike, sample_rate: float,
              frequencies: ArrayLike,
              omega0: float = OMEGA0) -> np.ndarray:
    """The complex wavelet coefficients, one row per frequency.

    The transform is computed in the frequency domain: at each scale it
    is one multiply against the record's FFT and one inverse FFT,
    instead of a convolution in time.

    It is computed a band of scales at a time (`BAND_BYTES`), because
    every scale at once is three arrays of (scales, padded length)
    complex values, and for a long record that is tens of gigabytes.
    The result is still the whole transform, one complex row per
    frequency as long as the record, which for a long record is itself
    gigabytes. For a picture, use `scalogram_peaks`, which reduces each
    band as it is computed and never holds the whole transform.

    The magnitude is normalized to the record's own amplitude: a 2 g
    tone reads 2 at any frequency, so the color bar is in the record's
    units and a ridge's height means the same thing everywhere.
    Torrence and Compo normalize each wavelet to unit energy instead,
    which is correct for asking how a record's variance is distributed.
    Under that normalization a constant-amplitude tone reads in
    proportion to the square root of its period, so the same 2 g at
    50 Hz and at 800 Hz differ by a factor of four on the picture, and
    no amplitude in g or N can be read from it. The two conventions
    differ by sqrt(scale), and a scalogram does not show which one it
    is in, so it is stated here.

    Parameters
    ----------
    values : array_like
        One channel's record, evenly sampled. It must be real and
        one-dimensional, with at least two samples.
    sample_rate : float
        Samples per second.
    frequencies : array_like
        The frequency of each row, in Hz, each above zero and below
        Nyquist.
    omega0 : float, default OMEGA0
        Cycles under the wavelet's window, which sets the trade between
        time and frequency resolution.

    Returns
    -------
    numpy.ndarray
        Complex, shape ``(len(frequencies), len(values))``. The
        magnitude is what a scalogram draws. The phase is kept because
        it costs nothing, and an instantaneous frequency would be read
        from the phase along a ridge.

    Raises
    ------
    ValueError
        When `values` is complex, not one-dimensional or shorter than
        two samples, or when a frequency is at or below zero or at or
        above Nyquist.
    """
    record, wanted, samples = _prepared(values, sample_rate, frequencies)
    out = np.empty((wanted.size, samples), dtype=complex)
    for rows, band in _bands(record, sample_rate, wanted, omega0):
        out[rows] = band
    return out


def scalogram_peaks(values: ArrayLike, sample_rate: float,
                    frequencies: ArrayLike, omega0: float = OMEGA0, *,
                    columns: int = COLUMNS) -> tuple[np.ndarray, np.ndarray]:
    """The scalogram's magnitude, reduced to at most `columns` time columns.

    This is what a picture is drawn from. The view, the scripting plot
    and the report all draw about a screen's width of columns, and a
    million-sample record is far more than that. Time is cut into equal
    slices and each column is the largest magnitude in its slice (a
    peak hold, the same reduction `decimate.peak_decimate` applies to a
    curve). The peaks are kept because the ridges are what a scalogram
    is read for: taking every nth sample could skip the samples where a
    transient's energy is, and a mean would lower its peak. When the
    record has no more samples than `columns`, nothing is reduced and
    the result is the magnitude itself.

    The transform is computed band by band and each band is reduced as
    it is computed, so the whole transform is never in memory at once.
    A long record should go through this function, which uses
    `BAND_BYTES` per working array whatever the record's length.

    Parameters
    ----------
    values, sample_rate, frequencies, omega0
        As for `scalogram`.
    columns : int
        The largest number of time columns to return.

    Returns
    -------
    times, magnitude
        `times` is the time of each column in seconds from the
        record's start, at the center of its slice, so the first and
        last columns are half a slice inside the ends of the record.
        `magnitude` is real, shape ``(len(frequencies), len(times))``.

    Raises
    ------
    ValueError
        As for `scalogram`.
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
    from functools import partial

    from scipy.fft import fft as _fft
    from scipy.fft import ifft as _ifft
    from scipy.fft import next_fast_len

    # every core on the transforms (2026-10-01); the same numbers
    fft, ifft = partial(_fft, workers=-1), partial(_ifft, workers=-1)

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
    # record is now centered on rather than a step down from its mean.
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
        # normalization: a real tone of amplitude A puts A/2 in each
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
    """The frequency of the largest magnitude at each instant, in Hz.

    This is the line a reader follows across a scalogram, as numbers a
    report can quote: for each column of `coefficients`, the entry of
    `frequencies` at the row with the largest magnitude. A column with
    no signal in it still returns a frequency, so the ridge only means
    something where there is a ridge to follow; a caller with a quiet
    stretch should treat that part of the line as meaningless.
    """
    magnitude = np.abs(np.asarray(coefficients))
    wanted = np.asarray(frequencies, dtype=float)
    return wanted[np.argmax(magnitude, axis=0)]


def decade_values(low: float, high: float) -> list[float]:
    """The round frequencies to label a log axis with.

    1, 2 and 5 times each power of ten, every one inside
    ``[low, high]``, in ascending order. The transform's rows are
    twelfths of an octave and nobody reads 158.7 Hz off an axis, so the
    flat picture, the 3-D stage and the report figure all label these
    values instead, and all three call this function.
    """
    # One function rather than a loop in each caller, because the three
    # loops drifted apart once (2026-09-16: they were byte-for-byte the
    # same, and the fourth would not have been).
    low, high = float(low), float(high)
    values = []
    decade = 10.0 ** np.floor(np.log10(low))
    while decade <= high * 10:
        for step in (1.0, 2.0, 5.0):
            value = decade * step
            if low <= value <= high:
                values.append(float(value))
        decade *= 10.0
    return values


def default_range(sample_rate: float, duration: float) -> tuple[float, float]:
    """The default (low, high) frequency range for a record, in Hz.

    The panel, the scripting plot and the report all use this one
    range. The top is 0.98 of Nyquist: the top line has to be below
    Nyquist, and the amplitude is accurate to a fraction of a percent
    right up to it. The bottom is eight cycles over the record's
    `duration` in seconds, kept between a thousandth of Nyquist and half
    the top (with no duration it is a hundredth of the top). At that
    frequency the longest wavelets still fit inside the record, so the
    bottom octave is not all cone of influence.
    """
    # One implementation, because the three callers drifted within a day
    # of each other: the view's default top moved from 0.4 of Nyquist to
    # 0.98 when the 0.4 was measured to be unnecessary (a tone at 0.95 of
    # Nyquist reads its amplitude to 0.1%), and `plot_scalogram` kept the
    # old number until this function existed.
    nyquist = float(sample_rate) / 2.0
    high = nyquist * 0.98
    low = (8.0 / float(duration)) if duration else high / 100.0
    low = min(max(low, nyquist / 1000.0), high / 2.0)
    return low, high


def reading(history: Any, channel: int = 0, *, low: float | None = None,
            high: float | None = None, per_octave: int | None = None,
            omega0: float | None = None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """One channel's scalogram as the app draws it, flat or in 3-D.

    Returns (clock, magnitude, frequencies, omega0) for row `channel`
    of `history`. Settings left out take the defaults the panel opens
    with (`default_range`, `PER_OCTAVE`, `OMEGA0`), and lines at or
    above Nyquist are dropped. Time is reduced to at most `COLUMNS`
    columns by a peak hold (`scalogram_peaks`), so a long record is
    computed in bounded memory and a transient's ridge is kept. The
    clock is in seconds and starts at the record's first time value.

    The window, `plot.plot_scalogram` and
    `viz.scalogram.plot_scalogram_stage` all call this function.

    Raises
    ------
    ValueError
        When fewer than two lines of the range lie under Nyquist.
    """
    rate = float(history.sample_rate)
    values = np.real(np.asarray(history.ordinate)[channel])
    duration = len(history.abscissa) / rate
    default_low, default_high = default_range(rate, duration)
    frequencies = log_frequencies(
        default_low if low is None else low,
        default_high if high is None else high,
        PER_OCTAVE if per_octave is None else per_octave)
    frequencies = frequencies[frequencies < rate / 2.0]
    if frequencies.size < 2:
        raise ValueError('no frequencies this record can carry in that '
                         f'range; it reaches {rate / 2.0:g} Hz')
    width = OMEGA0 if omega0 is None else float(omega0)
    clock, magnitude = scalogram_peaks(values, rate, frequencies, width)
    return clock + float(history.abscissa[0]), magnitude, frequencies, width
