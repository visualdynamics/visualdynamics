"""Shock response spectra.

An SRS is not a spectrum of the shock. It is the answer to a question
asked of it: hang a single-degree-of-freedom oscillator off the base,
shake the base with the measured transient, and record the largest
response the oscillator ever reaches. Sweep the oscillator's natural
frequency across the band of interest and the curve of those peaks is
the shock response spectrum.

That is why two shocks with the same SRS can look nothing alike, and
why an SRS cannot be inverted back to a time history: every peak is one
number taken from one whole run of one filter, and the phase that
produced it is gone.

The filter is Smallwood's ramp-invariant recursion (Smallwood 1981),
which is the method the shock community settled on. Multiplying by the
continuous transfer function in the frequency domain is the obvious
alternative and it is wrong near Nyquist: the sampled input is a ramp
between samples, not a train of impulses, and the ramp-invariant filter
is the one that gets that right. The recursion below is exact for that
assumption rather than an approximation of it.

Damping is quoted either as a ratio or as Q = 1/(2*zeta). Q = 10 —
5% of critical — is the near-universal default, and the value a
specification is assumed to be written at unless it says otherwise.

References
----------
1. Smallwood, D. O. (1981). "An Improved Recursive Formula for
   Calculating Shock Response Spectra." *Shock and Vibration
   Bulletin*, 51(2), 211-217. The ramp-invariant recursion the
   filter coefficients below are taken from.
2. ISO 18431-4:2007, *Mechanical vibration and shock - Signal
   processing - Part 4: Shock-response spectrum analysis*. The
   standard statement of the same digital filter, and of the maximax,
   primary and residual readings.
3. Irvine, T. ["An Introduction to the Shock Response
   Spectrum"](https://www.vibrationdata.com/tutorials2/srs_intr.pdf).
   A worked introduction, including why multiplying by the continuous
   transfer function in the frequency domain goes wrong near Nyquist.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

#: the amplification an SRS is worked out at unless told otherwise. The
#: convention is so settled that a curve quoted without a Q is read as
#: this one, but it is a choice and it belongs on the object.
DEFAULT_Q = 10.0

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

    Geometric, because an SRS is read on a log axis and a linear grid
    would crowd the top of the band and starve the bottom.
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
    oscillator given the acceleration of its base.

    The gain at DC is exactly one — b.sum() / a.sum() == 1 — which is
    the statement that an infinitely stiff oscillator goes wherever its
    base goes. It is the cheapest check that the coefficients are the
    right ones, and `test_srs` makes it.
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
    """(highest, lowest) the oscillators reach, one per frequency.

    The filter runs over the record once for every natural frequency at
    once — the state is a vector across frequencies and the loop is over
    time — so nothing of length (frequencies x samples) is ever held.
    Only the running extremes are kept, which is all an SRS is.
    """
    x = np.asarray(signal, dtype=float).ravel()
    zeta = damping_for(q) if damping is None else float(damping)
    b, a = ramp_invariant(frequencies, sample_rate, zeta)
    b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
    a1, a2 = a[:, 1], a[:, 2]

    y1 = np.zeros(b0.shape)
    y2 = np.zeros(b0.shape)
    highest = np.zeros(b0.shape)
    lowest = np.zeros(b0.shape)
    x1 = x2 = 0.0
    for sample in x:
        y = b0 * sample + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        np.maximum(highest, y, out=highest)
        np.minimum(lowest, y, out=lowest)
        y2, y1 = y1, y
        x2, x1 = x1, sample
    return highest, lowest


def maximax(signal: ArrayLike, frequencies: ArrayLike, sample_rate: float,
            damping: float | None = None,
            q: float = DEFAULT_Q) -> np.ndarray:
    """The peak absolute response, whichever way it went.

    The reading a shock specification is written against: a shock that
    only ever pushes one way is no gentler for it.
    """
    highest, lowest = peaks(signal, frequencies, sample_rate, damping, q)
    return np.maximum(highest, -lowest)
