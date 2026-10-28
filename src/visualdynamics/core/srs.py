"""Shock response spectra.

An SRS is not a spectrum of the shock itself. A single-degree-of-freedom
oscillator is attached to the base, the base is driven with the
measured transient, and the largest response the oscillator reaches is
recorded. Repeating that for natural frequencies across the band of
interest gives the curve of peaks that is the shock response spectrum.

For that reason two shocks with the same SRS can look very different,
and an SRS cannot be inverted back to a time history: each point is one
number taken from a whole run of one filter, and the phase that
produced it is lost.

The filter is Smallwood's ramp-invariant recursion (Smallwood 1981),
the standard method for shock response spectra. Multiplying by the
continuous transfer function in the frequency domain is the other
common approach, and it is inaccurate near Nyquist: the sampled input
is a ramp between samples, not a train of impulses, and the
ramp-invariant filter is built on that assumption. The recursion below
is exact for that assumption rather than an approximation of it.

Damping is given either as a ratio or as Q = 1/(2*zeta). Q = 10 (5% of
critical) is the usual default, and a specification is assumed to be
written at that value unless it says otherwise.

References
----------
The recursion implemented here is Smallwood's; the others set out the
same analysis and where it goes wrong.

1. Smallwood, D. O. (1981). "An Improved Recursive Formula for
   Calculating Shock Response Spectra." *Shock and Vibration
   Bulletin*.
2. ISO 18431-4:2007, *Mechanical vibration and shock - Signal
   processing - Part 4: Shock-response spectrum analysis*.
3. Irvine, T. ["An Introduction to the Shock Response
   Spectrum"](https://www.vibrationdata.com/tutorials2/srs_intr.pdf),
   including why multiplying by the continuous transfer function in
   the frequency domain goes wrong near Nyquist.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

#: the amplification an SRS is worked out at unless told otherwise. The
#: convention is so settled that a curve quoted without a Q is read as
#: this one, but it is a choice and it belongs on the object.
DEFAULT_Q = 10.0

#: the peaks an SRS can keep: the oscillator's largest response in the
#: positive direction, its largest in the negative (reported as a
#: magnitude), and the larger of the two
KINDS = ('positive', 'negative', 'maximax')

#: octave fraction the natural frequencies are spaced at. A twelfth is
#: fine enough that a lightly damped peak is not stepped over and
#: coarse enough to stay quick.
PER_OCTAVE = 12


def damping_for(q: float = DEFAULT_Q) -> float:
    """The damping ratio an amplification factor means."""
    return 1.0 / (2.0 * float(q))


def q_for(damping: float) -> float:
    """The amplification factor a damping ratio means."""
    return 1.0 / (2.0 * float(damping))


def octave_frequencies(low: float, high: float,
                       per_octave: int = PER_OCTAVE) -> np.ndarray:
    """Natural frequencies from `low` to `high`, `per_octave` to an octave.

    The spacing is geometric, because an SRS is read on a log axis and a
    linear grid would put most of its points at the top of the band and
    few at the bottom. The last frequency is the nearest whole number
    of steps to `high`, so it can land slightly above or below it.
    Raises ValueError unless 0 < `low` < `high`.
    """
    low, high = float(low), float(high)
    if not (low > 0.0 and high > low):
        raise ValueError(f'octave band {low} to {high} is not a band')
    octaves = np.log2(high / low)
    return low * 2.0 ** (np.arange(round(octaves * per_octave) + 1)
                         / float(per_octave))


def ramp_invariant(frequencies: ArrayLike, sample_rate: float,
                   damping: float) -> tuple[np.ndarray, np.ndarray]:
    """Smallwood's filter coefficients, one row per natural frequency.

    Returns (b, a), each (F, 3), for the absolute acceleration of the
    oscillator given the acceleration of its base. `damping` is a ratio
    of critical and must be between 0 and 1, or it raises ValueError.

    The gain at DC is exactly one (b.sum() / a.sum() == 1), because an
    infinitely stiff oscillator moves with its base. That is the
    simplest check that the coefficients are correct, and `test_srs`
    makes it.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    dt = 1.0 / float(sample_rate)
    zeta = float(damping)
    if not 0.0 < zeta < 1.0:
        raise ValueError(f'damping {zeta} is not under critical')

    wn = 2.0 * np.pi * frequencies
    wd = wn * np.sqrt(1.0 - zeta ** 2)
    decay = np.exp(-zeta * wn * dt)
    turned = wd * dt
    cosine = decay * np.cos(turned)
    # sin(wd dt)/(wd dt) -> 1 as the frequency goes to nothing, which is
    # where the ramp-invariant form is otherwise 0/0
    shaped = decay * np.sinc(turned / np.pi)

    b = np.stack([1.0 - shaped,
                  2.0 * (shaped - cosine),
                  decay ** 2 - shaped], axis=-1)
    a = np.stack([np.ones_like(cosine),
                  -2.0 * cosine,
                  decay ** 2], axis=-1)
    return b, a


def peaks(signal: ArrayLike, frequencies: ArrayLike, sample_rate: float,
          damping: float | None = None,
          q: float = DEFAULT_Q) -> tuple[np.ndarray, np.ndarray]:
    """(highest, lowest) response the oscillators reach, one per
    frequency.

    `damping` is a ratio of critical; when it is None, `q` sets it.
    Each oscillator's response is run over the whole record in turn
    and only its maximum and minimum kept, so no array of size
    (frequencies x samples) is held, which is all an SRS needs. Both
    start at zero, so `highest` is never below zero and `lowest` never
    above it.
    """
    # The pass was a Python loop over the samples with the state a
    # vector across frequencies; on a five-minute record at 16 kHz that
    # was 17 s. scipy's lfilter runs the same recursion in C, one
    # frequency at a time: a fifth of the time, the same numbers to a
    # part in 10^12 (2026-10-28).
    from scipy.signal import lfilter

    x = np.asarray(signal, dtype=float).ravel()
    zeta = damping_for(q) if damping is None else float(damping)
    b, a = ramp_invariant(frequencies, sample_rate, zeta)
    highest = np.zeros(len(b))
    lowest = np.zeros(len(b))
    if x.size:
        for k in range(len(b)):
            y = lfilter(b[k], a[k], x)
            highest[k] = max(float(y.max()), 0.0)
            lowest[k] = min(float(y.min()), 0.0)
    return highest, lowest


def spectra(highest: ArrayLike, lowest: ArrayLike) -> dict[str, np.ndarray]:
    """The three spectra read off `peaks`: 'positive' is `highest`,
    'negative' the depth of `lowest` as a magnitude, and 'maximax' the
    larger of the two at each frequency.

    Parameters
    ----------
    highest, lowest : array_like
        What `peaks` returned.

    Returns
    -------
    dict of str to numpy.ndarray
        Keyed 'positive', 'negative' and 'maximax'.
    """
    highest = np.asarray(highest, dtype=float)
    lowest = np.asarray(lowest, dtype=float)
    return {'positive': highest, 'negative': -lowest,
            'maximax': np.maximum(highest, -lowest)}


def maximax(signal: ArrayLike, frequencies: ArrayLike, sample_rate: float,
            damping: float | None = None,
            q: float = DEFAULT_Q) -> np.ndarray:
    """The peak absolute response, in either direction, one per
    frequency.

    This is the value a shock specification is written against: a
    shock whose response is all in one direction is no less severe for
    it. `damping` and `q` are as in `peaks`.
    """
    return spectra(*peaks(signal, frequencies, sample_rate, damping, q))['maximax']
