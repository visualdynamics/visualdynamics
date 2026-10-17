"""The statistics of a record read off its time history.

A random test is specified as a spectrum, and the spectrum keeps only
how the mean square is spread across frequency. Squaring throws away
the sign, the frequency axis throws away when things happened, and the
average over frames dilutes anything that happened in a few of them.
Clipping, dropouts, spikes, offsets and a loosening mount live in what
was thrown away, and the time history still has them. So a record is
read in the time domain before its spectrum is trusted, and these are
the numbers that reading is made of:

- The mean, against the standard deviation. A vibration record sits on
  zero; a mean well away from it is a bias, a channel still settling,
  or a zero shift. The rms barely moves with it, which is why the mean
  is read on its own.
- The rms, the standard deviation, the skewness and the kurtosis: the
  second, third and fourth moments. A Gaussian record has a skewness of
  0 and a (Pearson) kurtosis of 3 (`core.kurtosis`).
- The crest factor, the largest excursion over the rms, with the
  largest peak and the time it happened.
- How many separate excursions reach the record's own peak, and, when
  the channel's range is given, how many samples sit on the rail:
  within a stated margin of the range. A Gaussian record reaches its
  peak once; a clipped one has a flat top on every excursion that
  reached the edge. Excursions are counted, not samples, because a
  record sampled well above its band puts several samples on the top
  of every peak.
- How many samples are not finite. They are left out of every other
  number, and counted so that leaving them out is never silent.
- The share of samples beyond 3, 4 and 5 standard deviations, beside
  the share a Gaussian puts there (0.27 %, 0.0063 % and 0.000057 %).
- The rate the record crosses its mean going upward, counted, and the
  same rate from its spectrum by Rice's formula, the square root of the
  spectrum's second moment over its area.
- The expected largest peak of a Gaussian process at that rate over a
  stated duration, in standard deviations: sqrt(2 ln(nu0 T)). It grows
  with the length of a run, slowly: at 1,000 crossings a second about
  4.7 for a minute and 5.5 for an hour.
- The kurtosis frame by frame. A short fault barely moves the kurtosis
  of a long record, and the frame it is in stands out. A frame that
  does not vary at all, a dropout, has no kurtosis and is outside its
  band.

`record_statistics` reads one channel of a `TimeHistory` (also
`TimeHistory.time_statistics`) over the same span `core.kurtosis` reads
it over, so the kurtosis here and the kurtosis bars agree to the last
digit. `time_statistics` reads a plain array. Both return a
`TimeStatistics`, which carries every number, the bands each was judged
against and which ones fell outside.

The bands. Each judged statistic has a band, and every band is an
argument, because a band is a choice a procedure makes and no number
here can make it for the reader:

- the kurtosis of the record, `KURTOSIS_BAND`, 2 to 4 by default, the
  kurtosis bars' own band. It is set for rattles and spikes, and it
  does not see clipping: a Gaussian record clipped at 2.8 standard
  deviations reads about 2.86 and one clipped at 2.5 about 2.76, both
  inside it, and a real record at 2.86 is ordinary. Clipping is found
  by the samples on the rail and the crest factor, which is why they
  are read beside it. A procedure that wants the kurtosis to see a clip
  sets the band tighter: a whole record scatters little about 3 (the
  estimator's standard error is about sqrt(24/N), 0.02 at 60,000
  samples), so 2.9 to 3.1 holds a clean record of ordinary length.
- the kurtosis of each frame, `FRAME_BAND`, 2 to 4, `core.kurtosis`'s
  own band. A frame's scatter is set by how many independent values it
  holds, which is its length times the record's statistical bandwidth
  (`statistical_bandwidth`), so the frame is sized from the bandwidth:
  at least `FRAME_SECONDS`, and long enough to hold `FRAME_PRODUCT`
  bandwidth-seconds, where its kurtosis scatters by about 0.15. A
  narrow-band record needs long frames, and a record shorter than one
  frame has none to judge.
- the skewness, `SKEWNESS_BAND`, and the mean as a share of the
  standard deviation, `OFFSET_SHARE`.
- the crest factor, `CREST_BAND`, 3 to 6. Below 3 the peaks were cut
  off somewhere; above 6 a Gaussian record practically never goes, and
  spikes do.
- the samples on the rail, `RAIL_SAMPLES`: none. A sample that reached
  the edge of the range is a sample whose true value is not known.
- the excursions at the record's own peak, `PEAK_RUNS`: two, since a
  second peak can come within the margin of the first.
- the samples that are not finite: none.
- the share beyond each sigma, judged against the Gaussian's share when
  a Gaussian would put at least `MIN_EXPECTED` independent values
  there, and flagged when the two differ by more than `TAIL_RATIO`
  either way. Independent values, not samples, because neighboring
  samples of a record are not independent: one excursion beyond the
  level puts several samples there, and a narrow-band record's
  excursions come in clumps, a few cycles of one swell of its
  envelope. A record holds about twice its statistical bandwidth times
  its length in independent values, however fast it was sampled.

References
----------
The moments, the Gaussian tails, the crossing rate and the expected
largest peak are standard results; these are where they are set out.

1. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. The moments of a record, the
   Gaussian probability density, Rice's zero-crossing rate from the
   spectrum's moments, the expected largest peak of a stationary
   Gaussian process over a duration, and the statistical bandwidth
   that sets how many independent values a stretch of record holds.
2. Newland, D. E. (1993). *An Introduction to Random Vibrations,
   Spectral and Wavelet Analysis*, 3rd ed. Longman. The crossing rate
   and the distribution of the largest peak in a run, and how slowly it
   grows with the run's length.
3. Joanes, D. N., & Gill, C. A. (1998). "Comparing measures of sample
   skewness and kurtosis." *Journal of the Royal Statistical Society:
   Series D*. The plain moment ratios reported here, and their sampling
   scatter.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .kurtosis import HIGH, LOW, NOMINAL, analyzed_span, kurtosis

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import TimeHistory

# The view was asked for on 2026-10-16, from a worked example: a
# Gaussian record of 8.88 grms through a 25 g range read 8.83 grms, a
# crest factor of 2.83, a kurtosis of 2.86 and 0.47 % of its samples on
# the rail, and the kurtosis bars' band of 2 to 4 passed it. The rail
# count and the crest factor are what say "clipped", and neither was
# read anywhere in the package.

#: the band the kurtosis of the whole record (or its analyzed span) is
#: judged against: the bars' own 2 to 4, settable tighter
KURTOSIS_BAND = (LOW, HIGH)
# Not tighter by default (2026-10-16, on review): a real record at 2.86
# is ordinary and is not to be flagged by default. The rail count and
# the crest factor are what catch a clip; the band is tightened where a
# procedure asks.

#: the band each frame's kurtosis is judged against: `core.kurtosis`'s
#: own, because a frame is short and scatters
FRAME_BAND = (LOW, HIGH)

#: the band the skewness is judged against. A Gaussian record's is 0,
#: and its standard error is about sqrt(6/N), 0.01 at 60,000 samples.
SKEWNESS_BAND = (-0.1, 0.1)

#: how far the mean may sit from zero, as a share of the standard
#: deviation, before it is flagged. A 0.5 g offset on 8.9 grms is 0.056.
OFFSET_SHARE = 0.02

#: the crest factor's band
CREST_BAND = (3.0, 6.0)

#: how close to the range a sample counts as on the rail, as a share of
#: the range, and how close to the record's own peak a sample counts as
#: at it, as a share of the peak. A converter's codes are not exactly
#: its range, so a clipped sample can sit a code or two inside it.
RAIL_MARGIN = 0.001

#: the most samples that may sit on the rail before it is flagged
RAIL_SAMPLES = 0

#: the most separate excursions that may reach the record's own peak:
#: a second peak can come within the margin of the first
PEAK_RUNS = 2
# Excursions, not samples (2026-10-16, on review): counted as samples, a
# clean 20 to 200 Hz record at 20,480 a second put three samples on the
# top of its one largest peak and was flagged in three seeds of five.

#: the multiples of the standard deviation the tails are counted beyond
SIGMAS = (3, 4, 5)

#: how far the share beyond a sigma may be from the Gaussian's, as a
#: ratio either way, before it is flagged
TAIL_RATIO = 2.0

#: the fewest independent values a Gaussian would put beyond a sigma
#: for that tail to be judged at all: fewer, and the count is mostly
#: chance
MIN_EXPECTED = 10.0
# Independent values, not samples (2026-10-16, on review). Judged by
# samples, a clean 100 Hz, 2 % damped response flagged its three-sigma
# tail in three seeds of ten and a clean 20 to 200 Hz record at 20,480
# a second its four-sigma tail in three of ten. Judged by the expected
# crossings into the tail, 2 nu0 T exp(-k^2/2), as the review proposed,
# the 100 Hz response (27 crossings beyond three sigma in 12 s) still
# flagged it in 21 seeds of 100: its log ratio scattered by 0.62 where
# 27 independent crossings would scatter by 0.2, because the crossings
# of a narrow-band record clump. Twice the statistical bandwidth times
# the length times the Gaussian's share (0.8 for that record, unjudged)
# is smaller than both counts for every record tried and was never
# looser than the scatter measured.

#: the shortest frame for the kurtosis against time, in seconds. A
#: time, not a count of samples, because a frame's scatter is set by how
#: many independent values it holds, which goes with its length in time
#: and the record's band, not with its sample count: a clean record of
#: the 20 to 2,000 Hz flat profile read in frames of 1,024 samples at
#: 20,480 a second (0.05 s) put one frame in 1,200 above 4.0, and in
#: frames of 0.2 s none reached 3.5 (measured 2026-10-16, three seeds).
FRAME_SECONDS = 0.2

#: the bandwidth-seconds a frame holds at least, by default: its length
#: times the record's statistical bandwidth. A 0.2 s frame of the flat
#: 20 to 2,000 Hz profile holds 396, so that record keeps 0.2 s frames,
#: and a frame's kurtosis scatters by about 0.15 whatever the band.
FRAME_PRODUCT = 350.0
# Sized from the bandwidth (2026-10-16, on review): at a fixed 0.2 s, a
# clean 100 Hz, 2 % damped response (a bandwidth of 12.6 Hz, two or
# three independent values to a frame) put frames outside 2 to 4 in
# every seed of ten, and a clean 20 to 200 Hz record in eight of ten.
# Sized this way, 100 seeds each of the flat profile (0.2 s frames), a
# 20 to 200 Hz band (1.9 s) and the 100 Hz mode over two minutes (28 s)
# scattered by 0.15, 0.14 and 0.12, and no frame of any reached 3.9.
# Keeping the frames out of the default verdict was the other way, and
# it would have left a dropout found by nothing: a whole second at zero
# moves the record's kurtosis only from 3 to 3.27.


def gaussian_share(sigmas: float) -> float:
    """The share of a Gaussian's values further than `sigmas` standard
    deviations from its mean, either side: 0.27 % beyond 3."""
    return math.erfc(float(sigmas) / math.sqrt(2.0))


def rice_rate(frequency: ArrayLike, density: ArrayLike) -> float:
    """Upward crossings of the mean per second of a Gaussian process
    with this one-sided spectrum: sqrt(m2 / m0), Rice's formula.

    `density` is any one-sided spectrum of the process on the
    frequencies `frequency` in Hz (an autospectrum, a specification's
    levels on a fine grid, a periodogram), evenly spaced or not: the
    moments are taken by the trapezoid rule, so the units and the line
    spacing cancel. A flat profile from 20 to 2,000 Hz crosses about
    1,160 times a second. NaN when the spectrum has no area.
    """
    f = np.asarray(frequency, dtype=float)
    g = np.abs(np.asarray(density, dtype=float))
    if f.size < 2:
        return float('nan')
    m0 = float(np.trapezoid(g, f))
    m2 = float(np.trapezoid(f ** 2 * g, f))
    if not m0 > 0.0:
        return float('nan')
    return math.sqrt(m2 / m0)


def statistical_bandwidth(frequency: ArrayLike, density: ArrayLike
                          ) -> float:
    """The statistical bandwidth of a process with this one-sided
    spectrum, in Hz: the square of its area over the area of its
    square, (integral G)^2 / integral G^2.

    It is the width of the flat band that holds as many independent
    values a second, and so it sets how much an estimate made over a
    stretch of the record scatters. A flat band is its own width; a
    single mode of natural frequency fn and damping ratio zeta is
    2 pi zeta fn, 12.6 Hz at 100 Hz and 2 %. NaN when the spectrum has
    no area.
    """
    f = np.asarray(frequency, dtype=float)
    g = np.abs(np.asarray(density, dtype=float))
    if f.size < 2:
        return float('nan')
    area = float(np.trapezoid(g, f))
    square = float(np.trapezoid(g ** 2, f))
    if not square > 0.0:
        return float('nan')
    return area ** 2 / square


def expected_largest_peak(rate: float, duration: float) -> float:
    """The expected largest peak of a Gaussian process, in standard
    deviations, over `duration` seconds at `rate` upward crossings a
    second: sqrt(2 ln(rate * duration)).

    The leading term of the standard result; a run typically reaches a
    little more. About 4.7 for a minute and 5.5 for an hour at 1,000
    crossings a second. NaN when the run holds one crossing or fewer,
    where the formula says nothing.
    """
    crossings = float(rate) * float(duration)
    if not crossings > 1.0:
        return float('nan')
    return math.sqrt(2.0 * math.log(crossings))


@dataclass(frozen=True)
class Tail:
    """The samples beyond one multiple of the standard deviation.

    `share` is the record's, `expected` the Gaussian's, both as
    fractions of the samples read. `independent` is how many
    independent values a Gaussian of the record's bandwidth and length
    would put there; `judged` says that is enough for the comparison
    to mean something (`MIN_EXPECTED`), and `flagged` that it was judged
    and the two shares differ by more than `ratio` either way.
    """

    sigmas: float
    count: int
    share: float
    expected: float
    judged: bool
    ratio: float = TAIL_RATIO
    independent: float = float('nan')

    @property
    def flagged(self) -> bool:
        """Whether the share beyond is outside the Gaussian's by more
        than `ratio` either way, where it was judged."""
        if not self.judged:
            return False
        return not (self.expected / self.ratio <= self.share
                    <= self.expected * self.ratio)


@dataclass(frozen=True, eq=False)
class TimeStatistics:
    """Every number the time-domain reading of one record gives, the
    bands they were judged against, and which fell outside.

    Values (`mean`, `rms`, `std`, `peak`, `full_scale`) are in the
    record's own units: SI for a channel of a `TimeHistory`, as it is
    stored. Shares are fractions, rates are per second, times are on
    the record's clock in seconds. `rail` is None when no range was
    given.
    """

    #: samples read, and the samples per second they were taken at
    samples: int
    sample_rate: float
    mean: float
    rms: float
    std: float
    skewness: float
    kurtosis: float
    #: the sample furthest from zero, signed, and when it was taken
    peak: float
    peak_time: float
    #: the largest excursion from the mean, in standard deviations
    peak_sigmas: float
    #: the separate excursions within `margin` of the record's own
    #: peak: each run of neighboring samples there counts once
    at_peak: int
    #: the samples within `margin` of `full_scale`, or None without one
    rail: int | None
    full_scale: float | None
    margin: float
    tails: tuple[Tail, ...]
    #: upward crossings of the mean per second, counted and from the
    #: record's own spectrum
    crossings: float
    spectral_crossings: float
    #: the duration the expected largest peak is for, in seconds
    duration: float
    #: the record's statistical bandwidth from its own spectrum, in Hz
    bandwidth: float
    #: samples to a frame, each frame's middle on the record's clock,
    #: and its kurtosis
    frame_length: int
    frame_times: np.ndarray = field(repr=False)
    frame_kurtosis: np.ndarray = field(repr=False)
    kurtosis_band: tuple[float, float] = KURTOSIS_BAND
    frame_band: tuple[float, float] = FRAME_BAND
    skewness_band: tuple[float, float] = SKEWNESS_BAND
    offset_share: float = OFFSET_SHARE
    crest_band: tuple[float, float] = CREST_BAND
    rail_samples: int = RAIL_SAMPLES
    peak_runs: int = PEAK_RUNS
    #: the samples in the span read that are not finite, left out of
    #: every other number
    nonfinite: int = 0

    # ---- what is read from the numbers ---------------------------------

    @property
    def record_length(self) -> float:
        """The seconds read: the samples over the sample rate."""
        return self.samples / self.sample_rate

    @property
    def crest(self) -> float:
        """The crest factor: the largest excursion over the rms."""
        return abs(self.peak) / self.rms if self.rms > 0.0 else float('nan')

    @property
    def rail_share(self) -> float | None:
        """The samples on the rail, as a share; None without a range."""
        if self.rail is None:
            return None
        return self.rail / self.samples if self.samples else 0.0

    @property
    def headroom(self) -> float | None:
        """The range over the largest excursion; None without a range.
        Below one, the range was reached."""
        if self.full_scale is None or self.peak == 0.0:
            return None
        return self.full_scale / abs(self.peak)

    @property
    def expected_peak(self) -> float:
        """The expected largest peak over `duration`, in standard
        deviations, at the spectrum's crossing rate."""
        return expected_largest_peak(self.spectral_crossings, self.duration)

    @property
    def expected_peak_here(self) -> float:
        """The same over the record's own length, to set beside
        `peak_sigmas`."""
        return expected_largest_peak(self.spectral_crossings,
                                     self.record_length)

    @property
    def worst_frame(self) -> tuple[float, float] | None:
        """(time, kurtosis) of the frame furthest from 3; None when the
        record is shorter than a frame. A frame with no kurtosis (one
        that does not vary, a dropout) is furthest of all, and is given
        with a kurtosis of NaN."""
        values = self.frame_kurtosis
        if not values.size:
            return None
        dead = np.flatnonzero(~np.isfinite(values))
        if dead.size:
            k = int(dead[0])
        else:
            k = int(np.argmax(np.abs(values - NOMINAL)))
        return float(self.frame_times[k]), float(values[k])

    @property
    def frames_outside_mask(self) -> np.ndarray:
        """Which frames are outside `frame_band`, a frame with no
        kurtosis among them."""
        low, high = self.frame_band
        values = self.frame_kurtosis
        with np.errstate(invalid='ignore'):
            return ~((values >= low) & (values <= high))

    @property
    def frames_outside(self) -> int:
        """How many frames read outside `frame_band`, a frame with no
        kurtosis counted as outside it."""
        return int(np.count_nonzero(self.frames_outside_mask))

    # ---- the judgement --------------------------------------------------

    def flags(self) -> dict[str, bool]:
        """Each judged statistic, by key, and whether it fell outside
        its band. The keys are `STATISTICS` keys; the rail is judged
        only when a range was given, the frames only when the record
        holds one, and a tail only where a Gaussian crosses into it
        often enough for the count to mean something. A moment
        that has no value (a dead channel's kurtosis, a crest factor
        with no rms) is outside its band: a record with no variance is
        not a Gaussian one."""
        def outside(value, band):
            return bool(not np.isfinite(value)
                        or not band[0] <= value <= band[1])

        found = {
            'nonfinite': self.nonfinite > 0,
            'mean': bool(abs(self.mean) > self.offset_share * self.std),
            'skewness': outside(self.skewness, self.skewness_band),
            'kurtosis': outside(self.kurtosis, self.kurtosis_band),
            'crest': outside(self.crest, self.crest_band),
            'at_peak': self.at_peak > self.peak_runs,
        }
        if self.frame_kurtosis.size:
            found['frames'] = self.frames_outside > 0
        if self.rail is not None:
            found['rail'] = self.rail > self.rail_samples
        for tail in self.tails:
            if tail.judged:
                found[f'beyond_{tail.sigmas:g}'] = tail.flagged
        return found

    @property
    def flagged(self) -> list[str]:
        """The keys of the statistics outside their bands, in
        `STATISTICS` order."""
        found = self.flags()
        return ([key for key in STATISTICS if found.get(key)]
                + [key for key, out in found.items()
                   if out and key not in STATISTICS])

    def describe(self) -> str:
        """The judgement in words: what fell outside its band, or that
        nothing did."""
        flagged = self.flagged
        if not flagged:
            return 'Every statistic is inside its band.'
        return ('Outside its band: '
                + ', '.join(statistic_name(key) for key in flagged) + '.')


#: every statistic that is judged, by key, and the name a table gives
#: it, in the order a table lists them
STATISTICS: dict[str, str] = {
    'nonfinite': 'Samples not finite',
    'mean': 'Mean',
    'skewness': 'Skewness',
    'kurtosis': 'Kurtosis',
    'crest': 'Crest factor',
    'at_peak': 'Excursions at the peak',
    'rail': 'Samples on the rail',
    'beyond_3': 'Beyond 3σ',
    'beyond_4': 'Beyond 4σ',
    'beyond_5': 'Beyond 5σ',
    'frames': 'Frame kurtosis',
}


def statistic_name(key: str) -> str:
    """The name a table gives a judged statistic, by its key: a tail
    beyond a sigma `SIGMAS` does not hold is named as the others are."""
    if key in STATISTICS:
        return STATISTICS[key]
    if key.startswith('beyond_'):
        return f'Beyond {key[len("beyond_"):]}σ'
    raise KeyError(key)


def _band(value: Sequence[float] | None, default: tuple[float, float],
          what: str) -> tuple[float, float]:
    band = default if value is None else (float(value[0]), float(value[1]))
    if not band[0] < band[1]:
        raise ValueError(f'the {what} band runs from its low edge to its '
                         f'high edge, and ({band[0]:g}, {band[1]:g}) does '
                         'not')
    return band


def _spans(n: int, spans: Sequence[tuple[int, int]] | None
           ) -> list[tuple[int, int]]:
    if spans is None:
        return [(0, n)]
    kept = [(max(int(low), 0), min(int(high), n)) for low, high in spans]
    kept = [(low, high) for low, high in kept if high > low]
    if not kept:
        raise ValueError('none of the spans holds a sample of the record')
    return kept


def _spectral(pieces: Sequence[np.ndarray], rate: float
              ) -> tuple[float, float]:
    """(Rice's rate, statistical bandwidth) from the pieces' own
    spectra, each piece's periodogram weighted by its length, so the
    moments are those of the samples read and not of the joins between
    them.

    A periodogram line of a Gaussian record scatters as much as it is
    large, so the mean of its square is twice the square of its mean,
    and the area of the spectrum's square is half the periodogram's.
    The bandwidth is the length-weighted mean of each piece's.
    """
    m0 = m2 = 0.0
    width = weight = 0.0
    for piece in pieces:
        if piece.size < 4:
            continue
        power = np.abs(np.fft.rfft(piece)) ** 2
        f = np.fft.rfftfreq(piece.size, 1.0 / rate)
        m0 += float(np.sum(power))
        m2 += float(np.sum(f ** 2 * power))
        square = float(np.sum(power ** 2))
        if square > 0.0:
            width += (piece.size * 2.0 * float(np.sum(power)) ** 2
                      / square * rate / piece.size)
            weight += piece.size
    return (math.sqrt(m2 / m0) if m0 > 0.0 else float('nan'),
            width / weight if weight else float('nan'))


def time_statistics(values: ArrayLike, sample_rate: float, *,
                    start: float = 0.0,
                    spans: Sequence[tuple[int, int]] | None = None,
                    full_scale: float | None = None,
                    margin: float = RAIL_MARGIN,
                    frame: float | None = None,
                    duration: float | None = None,
                    kurtosis_band: Sequence[float] | None = None,
                    frame_band: Sequence[float] | None = None,
                    skewness_band: Sequence[float] | None = None,
                    offset_share: float = OFFSET_SHARE,
                    crest_band: Sequence[float] | None = None,
                    rail_samples: int = RAIL_SAMPLES,
                    peak_runs: int = PEAK_RUNS,
                    sigmas: Sequence[float] = SIGMAS,
                    tail_ratio: float = TAIL_RATIO) -> TimeStatistics:
    """Read one record's statistics in the time domain.

    The moments, the crest factor and its peak, the excursions at the
    peak, the samples on the rail and those not finite, the tails beyond each sigma, the crossing rates,
    the expected largest peak and the kurtosis frame by frame, each
    judged against its band (the module docstring says what each is and
    why its band is where it is).

    Parameters
    ----------
    values : array_like
        The record, one channel, evenly sampled. Complex values are
        read by their real part, as `core.kurtosis` reads them.
        Samples that are not finite are left out and counted.
    sample_rate : float
        Samples per second.
    start : float, default 0.0
        The first sample's time, in seconds, so the peak and the frames
        are placed on the record's own clock.
    spans : sequence of (int, int), optional
        Sample ranges to read, first inclusive and last exclusive; the
        whole record by default. Several are read together, as
        `core.kurtosis` reads a record's shock windows: the statistics
        over all their samples, the crossings and frames within each.
    full_scale : float, optional
        The channel's range, in the units of `values`: the rail is
        counted only when it is given.
    margin : float, default `RAIL_MARGIN`
        How close to the range, and to the record's own peak, a sample
        counts as on it, as a share.
    frame : float, optional
        Seconds to a frame for the kurtosis against time; rounded to
        whole samples. By default it is sized from the record's
        statistical bandwidth: `FRAME_PRODUCT` over the bandwidth, and
        at least `FRAME_SECONDS`.
    duration : float, optional
        Seconds the expected largest peak is for; the record's length
        by default.
    kurtosis_band, frame_band, skewness_band, crest_band : (float, float), optional
        The bands, `KURTOSIS_BAND`, `FRAME_BAND`, `SKEWNESS_BAND` and
        `CREST_BAND` by default.
    offset_share : float, default `OFFSET_SHARE`
        How far the mean may sit from zero, as a share of the standard
        deviation.
    rail_samples : int, default `RAIL_SAMPLES`
        The most samples allowed on the rail.
    peak_runs : int, default `PEAK_RUNS`
        The most separate excursions allowed at the record's own peak.
    sigmas : sequence of float, default `SIGMAS`
        The multiples of the standard deviation the tails are counted
        beyond.
    tail_ratio : float, default `TAIL_RATIO`
        How far a tail's share may be from the Gaussian's, as a ratio
        either way.

    Returns
    -------
    TimeStatistics
        Every number, the bands and the flags.

    Raises
    ------
    ValueError
        If the record has fewer than two samples, the sample rate is
        not positive, a band runs backwards, the range is not positive,
        or the frame holds fewer than two samples.
    """
    x = np.real(np.asarray(values)).astype(float, copy=False).ravel()
    rate = float(sample_rate)
    if not rate > 0.0:
        raise ValueError('the sample rate has to be above zero')
    if frame is not None:
        length = round(float(frame) * rate)
        if length < 2:
            raise ValueError(f'a frame of {frame:g} s holds {length} '
                             f'sample{"s" * (length != 1)} at {rate:g} a '
                             'second, and a kurtosis needs at least two')
    if full_scale is not None and not float(full_scale) > 0.0:
        raise ValueError('the range is how far either way the channel '
                         'reads, and has to be above zero')
    if not 0.0 <= float(margin) < 1.0:
        raise ValueError('the margin is a share of the range, from 0 up '
                         'to 1')
    bands = {'kurtosis_band': _band(kurtosis_band, KURTOSIS_BAND,
                                    'kurtosis'),
             'frame_band': _band(frame_band, FRAME_BAND, 'frame kurtosis'),
             'skewness_band': _band(skewness_band, SKEWNESS_BAND,
                                    'skewness'),
             'crest_band': _band(crest_band, CREST_BAND, 'crest factor')}
    ranges = _spans(x.size, spans)
    pieces = [x[low:high] for low, high in ranges]
    data = np.concatenate(pieces)
    finite = np.isfinite(data)
    if np.count_nonzero(finite) < 2:
        raise ValueError('a record needs two finite samples to have a '
                         'distribution')
    nonfinite = int(data.size - np.count_nonzero(finite))
    data = data[finite]
    n = data.size

    mean = float(data.mean())
    centered = data - mean
    m2 = float(np.mean(centered ** 2))
    std = math.sqrt(m2)
    rms = math.sqrt(float(np.mean(data ** 2)))
    skewness = (float(np.mean(centered ** 3)) / std ** 3 if std > 0.0
                else float('nan'))
    # one estimator of kurtosis in the package, so this and the bars
    # can never disagree
    kurt = kurtosis(data)

    # the peak, signed, and its time on the record's clock: found in
    # the pieces so its sample index is the record's own
    best, where = -1.0, 0
    for (low, _high), piece in zip(ranges, pieces):
        size = np.where(np.isfinite(piece), np.abs(piece), -1.0)
        k = int(np.argmax(size))
        if size[k] > best:
            best, where = float(size[k]), low + k
    peak = float(x[where])
    peak_time = float(start) + where / rate
    # separate excursions at the peak: each run of neighboring samples
    # within the margin counts once, so a peak sampled several times
    # over is one peak and a clipped record's flat tops are many
    at_peak = 0
    for piece in pieces:
        with np.errstate(invalid='ignore'):
            near = (np.isfinite(piece)
                    & (np.abs(piece) >= (1.0 - margin) * best))
        at_peak += int(near[0]) + int(np.count_nonzero(near[1:] & ~near[:-1]))
    rail = None
    if full_scale is not None:
        rail = int(np.count_nonzero(
            np.abs(data) >= (1.0 - margin) * float(full_scale)))

    # upward crossings of the mean, counted within each piece so a join
    # between two windows is not read as a crossing
    crossings = 0
    for piece in pieces:
        d = piece[np.isfinite(piece)] - mean
        crossings += int(np.count_nonzero((d[:-1] < 0.0) & (d[1:] >= 0.0)))
    spectral, bandwidth = _spectral(
        [piece[np.isfinite(piece)] - mean for piece in pieces], rate)

    tails = []
    for k in sigmas:
        count = (int(np.count_nonzero(np.abs(centered) > k * std))
                 if std > 0.0 else 0)
        expected = gaussian_share(k)
        # a record holds about 2 B T independent values, never more
        # than its samples
        independent = expected * min(n, 2.0 * bandwidth * n / rate
                                     if bandwidth > 0.0 else n)
        tails.append(Tail(sigmas=float(k), count=count, share=count / n,
                          expected=expected,
                          judged=independent >= MIN_EXPECTED,
                          ratio=float(tail_ratio), independent=independent))

    if frame is None:
        seconds = FRAME_SECONDS
        if bandwidth > 0.0:
            seconds = max(seconds, FRAME_PRODUCT / bandwidth)
        length = max(round(seconds * rate), 2)

    # the kurtosis frame by frame, tiling each piece from its start; a
    # tail shorter than a frame is left out, as an averaged spectrum
    # leaves it out
    times, frames = [], []
    for (low, _high), piece in zip(ranges, pieces):
        for first in range(0, piece.size - length + 1, length):
            frames.append(kurtosis(piece[first:first + length]))
            times.append(float(start) + (low + first + length / 2.0) / rate)

    deviation = max(abs(float(data.max()) - mean),
                    abs(float(data.min()) - mean))
    return TimeStatistics(
        samples=n, sample_rate=rate, mean=mean, rms=rms, std=std,
        skewness=skewness, kurtosis=kurt, peak=peak, peak_time=peak_time,
        peak_sigmas=deviation / std if std > 0.0 else float('nan'),
        at_peak=at_peak, rail=rail,
        full_scale=None if full_scale is None else float(full_scale),
        margin=float(margin), tails=tuple(tails),
        crossings=crossings / (n / rate), spectral_crossings=spectral,
        duration=float(n / rate if duration is None else duration),
        bandwidth=bandwidth,
        frame_times=np.asarray(times, dtype=float),
        frame_kurtosis=np.asarray(frames, dtype=float),
        frame_length=length, offset_share=float(offset_share),
        rail_samples=int(rail_samples), peak_runs=int(peak_runs),
        nonfinite=nonfinite, **bands)


def record_statistics(history: TimeHistory, record: int = 0,
                      **options) -> TimeStatistics:
    """Read one channel of a time history in the time domain.

    Over the span `core.kurtosis.analyzed_span` names (the frames the
    spectra are averaged over, the shock windows, or the whole record),
    so the kurtosis here is the kurtosis bar's to the last digit. Values
    are in SI, as the record stores them, and so is `full_scale`: a
    25 g range on an acceleration channel is ``25 * 9.80665``.

    Parameters
    ----------
    history : TimeHistory
        The record.
    record : int, default 0
        Which channel, counting from 0.
    **options
        As `time_statistics` takes them: `full_scale`, `margin`,
        `frame`, `duration` and the bands.

    Returns
    -------
    TimeStatistics
        Every number, the bands and the flags.

    Raises
    ------
    ValueError
        If the record is not evenly sampled, or as `time_statistics`
        raises.
    """
    if not 0 <= int(record) < history.num_records:
        raise ValueError(f'the record has {history.num_records} channel'
                         f'{"s" * (history.num_records != 1)}; there is '
                         f'no channel {record}')
    spans, _phrase = analyzed_span(history)
    return time_statistics(
        np.asarray(history.ordinate)[int(record)], history.sample_rate,
        start=float(history.abscissa[0]), spans=spans, **options)
