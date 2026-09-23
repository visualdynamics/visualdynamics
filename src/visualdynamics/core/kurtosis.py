"""How Gaussian a record is, channel by channel.

A random vibration test is specified as a spectrum, and a spectrum says
nothing about the *shape* of the distribution that produced it. Two
records with identical PSDs can be a smooth Gaussian hiss and a train
of rare hard peaks, and the second fatigues an article in a way the
first never will. Kurtosis is the number that tells them apart, and
reading it per channel is how a rattling fixture or a clipping
amplifier is caught while the spectrum still looks right.

**Pearson, not Fisher** (Brandon, 2026-08-24). The Pearson kurtosis of
a Gaussian is 3; Fisher's "excess" form subtracts that 3 so a Gaussian
reads 0. Both are in use and they differ by exactly three, which is
the sort of ambiguity that ends in a report claiming a value it did
not compute — so the nominal is written down here as `NOMINAL` and
every reading of it says which one it is.

The band: within `TOLERANCE` of nominal is unremarkable, and beyond it
in either direction is worth looking at. High is the interesting
direction — peaks the spectrum did not predict — but low matters too,
because a record that reads much *under* three has usually been
clipped or is not random at all.

References
----------
1. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. Chapter 3 on the moments of
   a distribution, and the fourth standardized moment reported here.
2. Joanes, D. N., & Gill, C. A. (1998). "Comparing measures of sample
   skewness and kurtosis." *Journal of the Royal Statistical Society:
   Series D*, 47(1), 183-189. The three sample estimators and their
   bias; this uses the plain moment ratio, which is what a vibration
   specification means by the word.
3. Steinwolf, A. (2006). "Shaker simulation of random vibrations with
   a high kurtosis value." *Journal of the IEST*, 49(1), 89-107. Why a
   record reading above three fatigues an article that a Gaussian one
   of the same spectrum does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

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

    The population (biased) estimator, `m4 / m2**2`, which is what
    every vibration-test convention quotes and what
    `scipy.stats.kurtosis(fisher=False, bias=True)` returns — the
    tests hold this against scipy rather than restating the algebra.

    A record with no variance has no shape to describe, so it answers
    NaN rather than dividing by zero: a dead channel is not a
    Gaussian one.
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
    """Which stretch of a record the analysis reads, and its name.

    The same stretch the spectra beside it are computed from, and for
    a reason found the hard way (Brandon, 2026-08-24): a system-ID
    excitation read whole came out at 4.23 where every steady frame of
    it read 2.92. Nothing was wrong with the arithmetic — the record
    is only a quarter analyzed, and the silent lead-in and tail make
    the *whole* record a mixture of a loud distribution and a quiet
    one, which genuinely is not Gaussian. A number that disagrees with
    the PSD printed beside it is worse than no number, so the reading
    covers what the PSD covers.

    Returns ``([(first, last), ...], phrase)`` — sample ranges and
    what to call them in a caption. A record with frames set answers
    with its analysis span, one with shocks found answers with those
    windows (order does not matter to a distribution, so separate
    windows read together), and one with neither is read whole.
    """
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

    Read over `analyzed_span` — the stretch the spectra beside it are
    computed from — so the two readings describe one thing.

    Every channel on one chart whatever it measures (Brandon,
    2026-08-24): kurtosis is dimensionless, so accelerations, forces
    and volts share an axis honestly — which is the one reading in
    this package where mixing quantities is not a lie. Channels whose
    record is dead answer NaN and draw as a gap rather than as zero,
    which would read as an impossibly flat distribution.
    """
    wanted = (range(data.num_records) if records is None
              else [int(i) for i in records])
    ordinate = np.asarray(data.ordinate)
    spans, _phrase = analyzed_span(data)
    return [(data.record_label(i),
             kurtosis(np.concatenate(
                 [np.real(ordinate[i][low:high]) for low, high in spans])))
            for i in wanted]
