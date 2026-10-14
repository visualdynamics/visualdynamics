"""Kurtosis, channel by channel: how close a record is to Gaussian.

A random vibration test is specified as a spectrum, and a spectrum says
nothing about the shape of the distribution that produced it. Two
records with the same PSD can be Gaussian noise or a series of rare,
large peaks, and the second does more fatigue damage to a test article
than the first. Kurtosis is the number that separates them, and
computing it per channel shows a rattling fixture or a clipping
amplifier while the spectrum still looks correct.

The kurtosis here is Pearson's, not Fisher's. The Pearson kurtosis of a
Gaussian is 3; Fisher's excess kurtosis subtracts 3, so a Gaussian is
0. Both are in use and they differ by exactly three, so the nominal
value is defined once as `NOMINAL`, and every plot and label of it says
which convention it is.

Within `TOLERANCE` of nominal a value is unremarkable, and beyond it in
either direction it is worth looking at. High values are the more
common concern, because they mean peaks the spectrum did not predict.
Low values matter too: a record well under three has usually been
clipped or is not random.

References
----------
The fourth standardized moment is standard; these are where it and its
use in vibration testing are set out.

1. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The moments of a
   distribution.
2. Joanes, D. N., & Gill, C. A. (1998). "Comparing measures of sample
   skewness and kurtosis." *Journal of the Royal Statistical Society:
   Series D*. The sample estimators and their bias; this reports the
   plain moment ratio, which is what a vibration specification means
   by the word.
3. Steinwolf, A. (2006). "Shaker simulation of random vibrations with
   a high kurtosis value." *Journal of the IEST*. Why a record reading
   above three fatigues an article that a Gaussian one of the same
   spectrum does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

# Pearson, not Fisher (Brandon, 2026-08-24). The two differ by exactly
# three, the kind of ambiguity that ends in a report quoting a value it
# did not compute, so the nominal is written down once as NOMINAL.

#: a Gaussian's Pearson kurtosis. Fisher's excess form is this minus
#: three; nothing here uses it, and the axis label says Pearson so a
#: reader never has to guess which convention a number is in.
NOMINAL = 3.0

#: how far from nominal is unremarkable. A finite record of a genuinely
#: Gaussian process scatters around three — the estimator's own
#: standard error is roughly sqrt(24/N), so even a clean 4096-sample
#: frame moves a few hundredths — and calling every such wobble a
#: finding would make the reading useless. Beyond one, something about
#: the record's shape is worth explaining.
TOLERANCE = 1.0

#: the band's edges, as the bar charts read them
LOW = NOMINAL - TOLERANCE
HIGH = NOMINAL + TOLERANCE


def kurtosis(values: ArrayLike) -> float:
    """The Pearson kurtosis of one record: 3 for a Gaussian.

    The population (biased) estimator, `m4 / m2**2`, which is the
    value vibration-test practice quotes and the value
    `scipy.stats.kurtosis(fisher=False, bias=True)` returns; the tests
    compare this function against scipy. Non-finite samples are
    dropped first.

    A record with no variance, or fewer than two finite samples, has no
    distribution shape to describe, so the result is NaN rather than a
    division by zero: a dead channel is not a Gaussian one.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float('nan')
    centered = x - x.mean()
    m2 = float(np.mean(centered ** 2))
    if m2 <= 0.0:
        return float('nan')
    return float(np.mean(centered ** 4) / m2 ** 2)


def analyzed_span(data: Any) -> tuple[list[tuple[int, int]], str]:
    """The part of a record the kurtosis is computed over, and its name.

    It is the same part the spectra shown with it are computed from.
    A record with a quiet lead-in and tail is, taken whole, a mixture
    of a loud distribution and a quiet one, and that mixture is not
    Gaussian even when the excitation is, so a kurtosis of the whole
    record would disagree with the PSD shown beside it.

    Returns ``([(first, last), ...], phrase)``: sample ranges, and the
    phrase a caption uses for them. A record with averaging frames set
    gives its analysis span, one with shocks found gives the shock
    windows (order does not matter to a distribution, so separate
    windows are read together), and one with neither is read whole.
    """
    # Found on a system-ID excitation (Brandon, 2026-08-24): read whole
    # it came out at 4.23, where every steady frame of it read 2.92.
    # The arithmetic was right; only a quarter of the record is
    # analyzed, and the silent lead-in and tail made the whole record a
    # mixture. A number that disagrees with the PSD printed beside it
    # is worse than no number, so this covers what the PSD covers.
    averaging = getattr(data, 'averaging', None)
    if averaging is not None:
        try:
            rate = data.sample_rate
        except ValueError:
            rate = None       # unevenly sampled: no frames to speak of
        if rate:
            first = int(averaging.start_sample(rate))
            last = round(averaging.stop(rate) * rate)
            if last > first:
                return ([(first, last)],
                        'over the frames the spectra are averaged over')
    shocks = getattr(data, 'shocks', None)
    if shocks:
        try:
            rate = data.sample_rate
        except ValueError:
            rate = None
        if rate:
            windows = [(int(shock.start * rate), round(shock.stop * rate))
                       for shock in shocks]
            windows = [(low, high) for low, high in windows if high > low]
            if windows:
                return windows, 'over the shock windows'
    return [(0, int(np.asarray(data.ordinate).shape[1]))], 'over the record'


def channel_kurtosis(data: Any, records: Sequence[int] | None = None
                     ) -> list[tuple[str, float]]:
    """[(channel label, Pearson kurtosis)] for a time history.

    Computed over `analyzed_span`, the part of the record the spectra
    shown with it are computed from, so the two describe the same data.

    Every channel goes on one chart, whatever it measures: kurtosis is
    dimensionless, so accelerations, forces and volts can share an
    axis. A channel whose record is dead gives NaN and is drawn as a
    gap, because a zero would look like an impossibly flat
    distribution.
    """
    # Every channel on one chart whatever it measures (Brandon,
    # 2026-08-24): this is the one value in the package where mixing
    # quantities on an axis is correct.
    wanted = (range(data.num_records) if records is None
              else [int(i) for i in records])
    ordinate = np.asarray(data.ordinate)
    spans, _phrase = analyzed_span(data)
    return [(data.record_label(i),
             kurtosis(np.concatenate(
                 [np.real(ordinate[i][low:high]) for low, high in spans])))
            for i in wanted]
