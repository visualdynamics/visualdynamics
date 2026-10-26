"""Fatigue damage and extreme response spectra, and time compression.

Two environments are compared for what they do to an article, not for
how they look. A specification derived from field data, or a test
proposed to stand for a field record, is judged by running both
through the same set of single-degree-of-freedom resonators, one at
each natural frequency across the band, and reading two numbers off
each resonator's response:

- the **extreme response spectrum** (ERS): the largest response the
  resonator is expected to reach over the duration. It answers "is the
  test's peak response as high as the field's", the strength question.
- the **fatigue damage spectrum** (FDS): the fatigue damage the
  response does over the duration. It answers "does the test use up as
  much life as the field does", the fatigue question.

A test whose ERS and FDS lie at or above the field's at every natural
frequency is at least as severe, mode for mode, whatever the shapes of
the two spectra. That is what makes the pair the tool for deriving a
specification and for the question "is this test harder than that one".

**The resonator.** Each resonator is base-excited, with natural
frequency fn and amplification Q = 1 / (2 zeta). Its response is the
relative displacement z between the mass and the base, the motion that
strains a part, expressed as a pseudo-acceleration, (2 pi fn)^2 z, so
that the response is in the units of the input and reads like the
acceleration it is close to. Its gain from base acceleration is

    H(f) = 1 / (1 - r^2 + j r / Q),   r = f / fn,

which is one at low frequency (a stiff resonator follows its base), Q
at fn, and falls as 1 / r^2 above. The mean square of the response to
a one-sided acceleration density G is the integral of |H|^2 G. For a
flat G that integral is exactly (pi / 2) fn Q G, Miles' equation, since
the integral of |H|^2 over all frequencies is (pi / 2) fn Q; and the
integral of f^2 |H|^2 is fn^2 times the same, so the response crosses
zero upward exactly fn times a second.

**From a spectrum.** The response's spectral moments,
m_k = integral f^k |H|^2 G df for k = 0, 1, 2 and 4, are taken on a
grid of cells fine enough that every resonator's half-power band holds
`CELLS` of them. Each cell's share of G is the spectrum's own area over
the cell (`Psd.areas`), read the way the spectrum is read, per bin or
as a power law between breakpoints, so a spectrum whose lines are
coarser than a resonator's band is integrated as it is drawn and not
sampled at its lines. The response's rms is sqrt(m0), its rate of
upward crossings of zero sqrt(m2 / m0) (Rice's formula, as
`time_statistics.rice_rate`), its rate of peaks sqrt(m4 / m2), and its
irregularity m2 / sqrt(m0 m4), one for a pure tone and smaller the more
small peaks ride on the large ones.

The fatigue model is the S-N curve N S^b = C: a cycle of amplitude S
uses 1 / N(S) = S^b / C of the life, and the cycles add (the
Palmgren-Miner rule). The damage over a duration T is then the expected
number of cycles times the mean of S^b over their amplitudes. The
constant C and the stress per unit response are a material's and a
design's, so they are left at one, and the FDS is in the response unit
to the power b: a value to compare between environments read at the
same b and Q, not a life. Three readings of the amplitudes are offered,
`METHODS`:

- `'narrowband'` (the default): every upward crossing of zero is one
  cycle, of amplitude Rayleigh-distributed with the response's rms
  sigma, the distribution of a narrow-band Gaussian process's peaks.
  The mean of S^b over a Rayleigh distribution is
  (sqrt(2) sigma)^b Gamma(1 + b / 2), so

      FDS = nu0 T (sqrt(2) sigma)^b Gamma(1 + b / 2).

  A resonator's response is narrow-band, and this is the closed form
  that reads it. For a broader response it expects more damage than
  rainflow counting finds, so it errs on the safe side; on simulated
  records it reads a few percent over their rainflow counts, and over
  Dirlik's fit at every resonator.
- `'rice'`: every positive peak is a cycle of its own height, at the
  rate of peaks, the heights distributed as Rice's peak distribution
  for the response's irregularity alpha. In standard deviations u it
  is the mixture

      p(u) = sqrt(1 - alpha^2) / sqrt(2 pi) exp(-u^2 / (2 (1 - alpha^2)))
             + alpha u exp(-u^2 / 2) Phi(alpha u / sqrt(1 - alpha^2)),

  with Phi the standard normal distribution: Rayleigh at alpha = 1,
  where it is the narrow-band reading exactly, and Gaussian as alpha
  falls to zero. The mean of u^b over its positive part is taken by
  quadrature. Counting each peak as a whole cycle also counts high when
  small peaks ride on large ones.
- `'dirlik'`: the rainflow ranges' distribution as Dirlik's empirical
  fit gives it from m0, m1, m2 and m4: an exponential and two Rayleigh
  terms in Z = range / (2 sigma), with weights

      xm = (m1 / m0) sqrt(m2 / m4),   g = m2 / sqrt(m0 m4),
      D1 = 2 (xm - g^2) / (1 + g^2),
      R  = (g - xm - D1^2) / (1 - g - D1 + D1^2),
      D2 = (1 - g - D1 + D1^2) / (1 - R),   D3 = 1 - D1 - D2,
      Qd = 1.25 (g - D3 - D2 R) / D1,

  and density in Z of D1 / Qd exp(-Z / Qd) + D2 Z / R^2
  exp(-Z^2 / (2 R^2)) + D3 Z exp(-Z^2 / 2), counted at the rate of
  peaks. With amplitude half the range the mean of S^b is
  sigma^b (D1 Qd^b Gamma(1 + b) + sqrt(2)^b Gamma(1 + b / 2)
  (D2 |R|^b + D3)). It is a fit to simulated rainflow counts, not a
  derivation, and it is checked here against rainflow counts of
  simulated records. Where the response is so narrow that the weights
  are 0 / 0 (a single frequency, g = 1), the narrow-band reading stands in,
  which is the limit the fit approaches.

The ERS is sigma times the expected largest peak over the duration,
sqrt(2 ln(nu0 T)) standard deviations
(`time_statistics.expected_largest_peak`), whatever the method: the
leading term of the standard result for a Gaussian process whose
up-crossings of a high level are rare and independent. A run typically
reaches a little more. It says nothing where the run holds a single
crossing or fewer, and such resonators read NaN, said in the notes.

**From a time history.** The record, leveled to its mean as its
spectrum would be, is run through each resonator by a recursion exact
for an input that is a straight line between samples (the assumption
`core.srs` makes for the absolute acceleration, made here for the
relative displacement): the resonator's state equation is integrated
over one sample with the input a ramp, by the matrix exponential of the
state matrix augmented with the input and its slope, and the result is
the second-order filter `response_filter` returns. Its gain at zero
frequency is exactly one. Natural frequencies above a twentieth of
the sample rate are refused. A sampled tone is a chord between samples,
not a straight line, so a resonator near the top reads its input a
little low (0.8 % at resonance at a twentieth, 3.3 % at a tenth), and
its response's peaks fall between samples (read up to
1 - cos(pi / 20), 1.2 %, low at a twentieth, 4.9 % at a tenth).
Damage goes as the amplitude to the power b, so at b = 8 a 2 %
shortfall in amplitude is 15 % of damage. The result says how low its
top resonator reads.

The response is counted by rainflow (`rainflow`): reduced to its
turning points, then every range smaller than or equal to both ranges
beside it is a closed cycle, taken out with its two points, until none
is left; what remains is the residue, whose ranges are counted as half
cycles. The FDS is the sum over the cycles of (range / 2)^b, half
cycles at one half, and the ERS is the largest response in the record.
Both are over the record's own length. A duration given for a time
history scales the FDS linearly to it, which assumes the record is a
stationary sample of the whole, and leaves the ERS as the record's own
largest response, which the result says.

**Time compression.** At one fatigue exponent, a test at a level W1
for a time t1 does the damage of a service level W2 for a time t2 when

    t2 / t1 = (W1 / W2)^(b / 2) = (S1 / S2)^b,

with W a spectral density and S an rms or amplitude; the damage of the
narrow-band reading goes as T sigma^b, and sigma as sqrt(W). The ratio
W1 / W2 is the exaggeration factor on the ASD. A test document caps
it, examples of what test documents ask for being 4 on the ASD and 2 on
the rms (`CAP`), because a higher level can fail an article through a
mechanism the fatigue exponent does not describe; the exponent itself
is a convention of the document, commonly 7.5 for random, 5 to 8, and 6
for sine. The exponent is the m of that rule, which some documents
write as b; where a document uses b for a material's fatigue strength
exponent, a different quantity, its m is the number to give here.

Two environments read at one b and Q are set against each other by
`equivalence`: the test's FDS over the field's at each natural
frequency, the factor on the test's ASD that brings its FDS to the
field's at every frequency, the duration at its own level that does
the same, the exaggeration the test carries at each frequency in the
sense of the rule above, and the ERS of the two in decibels.

Refused, with the cause named: a record whose units are not declared;
a record of another quantity than acceleration; a cross spectrum; a
spectrum line or a sample that is not finite, or a line below zero,
with its frequency or time named; a spectrum with no area and a record
that does not vary; a Q or b outside `Q_RANGE` or `B_RANGE`; a
duration that is not given, or not positive, for a spectrum; and
natural frequencies that are not positive or are above a twentieth of
a time history's sample rate.

References
----------
The peak distributions, the crossing rate and the expected largest
peak are standard results of random data analysis, set out in the
first; the narrow-band damage follows from the Rayleigh distribution's
moments as above, and Dirlik's fit is checked against rainflow counts
of simulated records rather than cited. The time-compression rule and
its caps are the second's.

1. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. Rice's rate of zero
   crossings and of peaks from a spectrum's moments, the Rayleigh
   distribution of a narrow-band Gaussian process's peaks and Rice's
   distribution of the peaks of a broader one, and the expected largest
   peak of a stationary Gaussian process over a duration.
2. MIL-STD-810H with Change 1 (2022), *Environmental Engineering
   Considerations and Laboratory Tests*, Method 514.8 (Vibration),
   Annex A, paragraph 2.2. The time-compression rule on the rms and on
   the ASD, the exponent commonly used for random and for sine, and
   the caps on the exaggeration factor.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from .progress import Ticker
from .srs import DEFAULT_Q, PER_OCTAVE, damping_for, octave_frequencies
from .time_statistics import expected_largest_peak

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import Psd, TimeHistory

#: the dimension of the spectra read: a density of acceleration
DENSITY = 'acceleration**2/frequency'

#: the readings of a response's cycles a spectrum can be read with (see
#: the module's text); a time history is always counted by rainflow
METHODS = ('narrowband', 'rice', 'dirlik')

#: the reading used unless another is asked for: the closed form, which
#: errs on the safe side of rainflow counting
METHOD = 'narrowband'

#: the amplifications accepted. Below 1 a resonator barely amplifies at
#: all and is not a mode; above 500 (0.1 % of critical) it is lighter
#: than any structure a test is run on.
Q_RANGE = (1.0, 500.0)

#: the fatigue exponents accepted. Test documents use 4 to 8 for random
#: and 6 for sine; metals span about 3 to 25. Outside that a number is a
#: typing error, not a material.
B_RANGE = (2.0, 25.0)

#: the cells of the integration grid in each resonator's half-power
#: band, fn / Q wide. Forty keep the response's mean square within a
#: few parts in ten thousand of the exact integral at any Q.
CELLS = 40

#: the fewest samples per cycle a time history's resonator is read at:
#: natural frequencies above the sample rate over this are refused (see
#: the module's text for what a tenth would cost)
SAMPLES_PER_CYCLE = 20

#: the exaggeration factors test documents commonly cap a compressed
#: test at, on the ASD and on the rms (the square root of the first)
CAP = {'asd': 4.0, 'rms': 2.0}

#: the most natural frequencies integrated against the grid at once,
#: so a fine grid at a high Q is never held against every frequency
CHUNK = 16

#: the quadrature grid Rice's peak distribution is integrated on, in
#: standard deviations: u^b exp(-u^2 / 2) peaks at sqrt(b + 1), at most
#: 5.1 in `B_RANGE`, and is negligible long before 40
_U = np.linspace(0.0, 40.0, 16001)


def _checked(name: str, value: Any, bounds: tuple[float, float]) -> float:
    """`value` as a float within `bounds`, or a ValueError naming it."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} {value!r} is not a number') from None
    low, high = bounds
    if not (math.isfinite(number) and low <= number <= high):
        raise ValueError(f'{name} {value!r} is outside {low:g} to {high:g}')
    return number


def checked_q(q: Any) -> float:
    """`q` as a float, or a ValueError unless it is within `Q_RANGE`."""
    return _checked('Q', q, Q_RANGE)


def checked_b(b: Any) -> float:
    """`b` as a float, or a ValueError unless it is within `B_RANGE`:
    the fatigue exponent is never assumed, so None is refused too."""
    if b is None:
        raise ValueError('the fatigue exponent b is not given; it is the '
                         "governing document's convention, commonly 7.5 "
                         'for random and 6 for sine')
    return _checked('the fatigue exponent b', b, B_RANGE)


def checked_duration(duration: Any) -> float:
    """`duration` in seconds as a float, or a ValueError unless it is a
    positive finite number."""
    if duration is None:
        raise ValueError('the duration is not given; damage and the '
                         'largest peak both grow with it')
    try:
        seconds = float(duration)
    except (TypeError, ValueError):
        raise ValueError(f'the duration {duration!r} is not a number'
                         ) from None
    if not (math.isfinite(seconds) and seconds > 0.0):
        raise ValueError(f'the duration {duration!r} s is not positive')
    return seconds


def _natural(frequencies: ArrayLike) -> np.ndarray:
    """The natural frequencies as a sorted float array, or a ValueError
    unless every one is positive and finite."""
    fn = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if fn.size == 0:
        raise ValueError('no natural frequencies to read')
    if not (np.all(np.isfinite(fn)) and np.all(fn > 0.0)):
        raise ValueError('natural frequencies must be positive and finite')
    return np.sort(fn)


def pseudo_gain(frequency: ArrayLike, natural: ArrayLike,
                q: float = DEFAULT_Q) -> np.ndarray:
    """|H|^2 of the pseudo-acceleration of a base-excited resonator,
    1 / ((1 - r^2)^2 + (r / Q)^2) with r = f / fn.

    Broadcast over `frequency` and `natural`: one natural frequency per
    row when `natural` is a column. One at zero frequency and Q^2 at fn.
    """
    r = np.asarray(frequency, dtype=float) / np.asarray(natural, dtype=float)
    return 1.0 / ((1.0 - r * r) ** 2 + (r / float(q)) ** 2)


def _grid(low: float, high: float, natural: np.ndarray, q: float
          ) -> np.ndarray:
    """The edges of the integration cells over `low` to `high`:
    geometric, `CELLS` to a half-power band, from zero when the spectrum
    starts there (the first cell then runs from zero to a twentieth of
    the lowest natural frequency, where every resonator's gain is one to
    a fraction of a percent)."""
    start = low
    if start <= 0.0:
        start = min(float(natural.min()) / 20.0, high / 20.0)
    ratio = 1.0 + 1.0 / (CELLS * q)
    count = max(math.ceil(math.log(high / start) / math.log(ratio)), 1)
    edges = np.geomspace(start, high, count + 1)
    if low <= 0.0:
        edges = np.concatenate([[0.0], edges])
    return edges


def response_moments(psd: Psd, record: int, natural: ArrayLike,
                     q: float = DEFAULT_Q, *,
                     progress: Callable[[int, int], None] | None = None
                     ) -> np.ndarray:
    """The spectral moments m0, m1, m2 and m4 of each resonator's
    pseudo-acceleration response to record `record` of `psd`, shape
    (4, natural frequencies), in the record's units squared times Hz to
    the power k.

    The cells' shares of the spectrum are its own areas over them
    (`Psd.areas`), so a spectrum read per bin or between breakpoints is
    integrated as it is read, whatever its line spacing. The record is
    taken as given; `psd_spectra` checks it first.
    """
    fn = _natural(natural)
    q = checked_q(q)
    low, high = held_band(psd, record)
    edges = _grid(low, high, fn, q)
    areas = np.asarray(psd.areas(record, edges[:-1], edges[1:]), dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep = areas > 0.0
    areas, centers = areas[keep], centers[keep]
    powers = np.stack([np.ones_like(centers), centers, centers ** 2,
                       centers ** 4])
    moments = np.empty((4, fn.size))
    ticker = Ticker(progress)
    ticker.add(fn.size)
    for start in range(0, fn.size, CHUNK):
        rows = fn[start:start + CHUNK, None]
        weighted = pseudo_gain(centers, rows, q) * areas
        moments[:, start:start + CHUNK] = (weighted @ powers.T).T
        ticker.tick(rows.shape[0])
    return moments


def narrowband_damage(m0: ArrayLike, m2: ArrayLike, b: float,
                      duration: float) -> np.ndarray:
    """nu0 T (sqrt(2) sigma)^b Gamma(1 + b / 2): every upward crossing a
    cycle of Rayleigh amplitude (see the module's text)."""
    m0, m2 = np.asarray(m0, dtype=float), np.asarray(m2, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.sqrt(m2 / m0)
    return (rate * duration * (2.0 * m0) ** (b / 2.0)
            * math.gamma(1.0 + b / 2.0))


def rice_moment(alpha: ArrayLike, b: float) -> np.ndarray:
    """The mean over Rice's peak distribution, at irregularity `alpha`,
    of u^b on its positive part (u in standard deviations), taken by
    quadrature: 2^(b/2) Gamma(1 + b/2) at alpha = 1, the Rayleigh's."""
    from scipy.special import ndtr

    alpha = np.clip(np.atleast_1d(np.asarray(alpha, dtype=float)), 0.0, 1.0)
    u = _U
    rayleigh = u * np.exp(-u * u / 2.0)
    moments = np.empty(alpha.size)
    for k, a in enumerate(alpha):
        if a >= 1.0:
            density = rayleigh
        else:
            spread = math.sqrt(1.0 - a * a)
            density = (spread / math.sqrt(2.0 * math.pi)
                       * np.exp(-u * u / (2.0 * spread * spread))
                       + a * rayleigh * ndtr(a * u / spread))
        moments[k] = np.trapezoid(u ** b * density, u)
    return moments


def rice_damage(m0: ArrayLike, m2: ArrayLike, m4: ArrayLike, b: float,
                duration: float) -> np.ndarray:
    """The positive peaks counted as cycles at the rate of peaks, their
    heights distributed as Rice's peaks (see the module's text)."""
    m0, m2, m4 = (np.asarray(m, dtype=float) for m in (m0, m2, m4))
    with np.errstate(divide='ignore', invalid='ignore'):
        peaks = np.sqrt(m4 / m2)
        alpha = m2 / np.sqrt(m0 * m4)
    return peaks * duration * m0 ** (b / 2.0) * rice_moment(alpha, b)


def dirlik_damage(m0: ArrayLike, m1: ArrayLike, m2: ArrayLike, m4: ArrayLike,
                  b: float, duration: float) -> np.ndarray:
    """Dirlik's rainflow range distribution counted at the rate of
    peaks (see the module's text); the narrow-band reading where the
    weights are 0 / 0."""
    m0, m1, m2, m4 = (np.atleast_1d(np.asarray(m, dtype=float))
                      for m in (m0, m1, m2, m4))
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        peaks = np.sqrt(m4 / m2)
        g = m2 / np.sqrt(m0 * m4)
        xm = (m1 / m0) * np.sqrt(m2 / m4)
        d1 = 2.0 * (xm - g * g) / (1.0 + g * g)
        r = (g - xm - d1 * d1) / (1.0 - g - d1 + d1 * d1)
        d2 = (1.0 - g - d1 + d1 * d1) / (1.0 - r)
        d3 = 1.0 - d1 - d2
        qd = 1.25 * (g - d3 - d2 * r) / d1
        moment = (d1 * qd ** b * math.gamma(1.0 + b)
                  + 2.0 ** (b / 2.0) * math.gamma(1.0 + b / 2.0)
                  * (d2 * np.abs(r) ** b + d3))
        damage = peaks * duration * m0 ** (b / 2.0) * moment
    narrow = ~np.isfinite(damage)
    if narrow.any():
        damage = np.where(narrow, narrowband_damage(m0, m2, b, duration),
                          damage)
    return damage


@dataclass(frozen=True)
class ResponseSpectra:
    """The ERS and the FDS of one record, with what they rest on.

    `frequency` holds the natural frequencies in Hz, and the arrays
    beside it one value per natural frequency, in SI: the FDS in the
    record's unit to the power `b` (m/s² for an acceleration), with the
    S-N constant and the stress per unit response at one; the ERS and
    the rms in the record's unit.
    """

    #: the natural frequencies, Hz
    frequency: np.ndarray
    #: the fatigue damage over `duration`, K = C = 1
    fds: np.ndarray
    #: the largest response expected (a spectrum) or reached (a time
    #: history) over the duration
    ers: np.ndarray
    #: the response's rms
    rms: np.ndarray
    #: the response's upward crossings of zero per second
    crossings: np.ndarray
    #: the response's peaks (maxima) per second
    peaks: np.ndarray
    #: the amplification each resonator was read at
    q: float
    #: the fatigue exponent
    b: float
    #: the seconds the FDS is over
    duration: float
    #: how the cycles were read: one of `METHODS`, or 'rainflow'
    method: str
    #: 'spectrum' or 'time history'
    source: str
    #: the record's quantity, as `ordinate_dim` spells it
    dimension: str
    #: which record, in words
    label: str = ''
    #: the seconds the ERS is over: the duration for a spectrum, the
    #: record's own length for a time history
    ers_duration: float = float('nan')
    #: what the reader should know about the result, in words
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def damping(self) -> float:
        """The damping ratio Q stands for."""
        return damping_for(self.q)

    @property
    def irregularity(self) -> np.ndarray:
        """Upward crossings over peaks, one per natural frequency: one
        for a pure tone, smaller the more small peaks ride on large."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.crossings / self.peaks

    @property
    def rests_on(self) -> str:
        """What the result assumes, in words."""
        model = (f'an S-N curve N S^b = C at b = {self.b:g}, damage added '
                 'by the Palmgren-Miner rule, with C and the stress per '
                 'unit response at one')
        if self.source == 'time history':
            cycles = ('cycles counted by rainflow in the response, the '
                      'residue as half cycles')
            peak = f'the largest response in the record ({self.ers_duration:g} s)'
        else:
            cycles = {
                'narrowband': ('every upward crossing of zero a cycle of '
                               'Rayleigh amplitude, the narrow-band '
                               'reading'),
                'rice': ("every positive peak a cycle of its height, the "
                         "heights distributed as Rice's peaks"),
                'dirlik': ("rainflow ranges distributed as Dirlik's "
                           'empirical fit'),
            }[self.method]
            peak = ('sigma sqrt(2 ln(nu0 T)), the leading term of the '
                    'expected largest peak of a Gaussian response')
        return (f'pseudo-acceleration of resonators at Q = {self.q:g}; '
                f'{model}; {cycles}; the ERS as {peak}')

    def describe(self) -> str:
        """The result in one line: what was read and over how long."""
        band = (f'{self.frequency[0]:.4g} to {self.frequency[-1]:.4g} Hz'
                if self.frequency.size > 1
                else f'{self.frequency[0]:.4g} Hz')
        how = 'rainflow' if self.source == 'time history' else self.method
        return (f'{self.label + ": " if self.label else ""}ERS and FDS '
                f'from the {self.source}, {band}, Q = {self.q:g}, '
                f'b = {self.b:g}, {_seconds(self.duration)}, {how}')

    def at_duration(self, duration: float) -> ResponseSpectra:
        """The same result over another duration: the FDS scaled in
        proportion, the ERS of a spectrum read again at the new length,
        the ERS of a time history left as the record's own."""
        seconds = checked_duration(duration)
        fds = self.fds * (seconds / self.duration)
        notes = self.notes
        if self.source == 'time history':
            ers, ers_duration = self.ers, self.ers_duration
            notes = notes + (
                (f'the FDS is scaled from {_seconds(self.duration)} to '
                f'{_seconds(seconds)}, which takes the record as a '
                'stationary sample of the whole; the ERS stays the '
                "record's own largest response"),)
        else:
            ers = self.rms * np.array([expected_largest_peak(rate, seconds)
                                       for rate in self.crossings])
            ers_duration = seconds
        return replace(self, fds=fds, ers=ers, duration=seconds,
                       ers_duration=ers_duration, notes=notes)


def _seconds(value: float) -> str:
    """A duration in words, in the unit that reads best."""
    if value >= 3600.0:
        return f'{value / 3600.0:.4g} h'
    if value >= 60.0:
        return f'{value / 60.0:.4g} min'
    return f'{value:.4g} s'


def _band(low: float | None, high: float | None, top: float | None,
          default: tuple[float, float]) -> tuple[float, float]:
    """The natural-frequency band from what was given, or the default."""
    low = default[0] if low is None else float(low)
    high = default[1] if high is None else float(high)
    if top is not None:
        high = min(high, top)
    # an edge that is not finite has no octaves to count; the order of
    # the two is `srs.octave_frequencies`'s to refuse
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(f'natural-frequency band {low:g} to {high:g} Hz '
                         'is not a band')
    return low, high


def _octaves(low: float, high: float, per_octave: int) -> np.ndarray:
    """`per_octave` natural frequencies to an octave from `low`, none
    above `high` (`srs.octave_frequencies` rounds its last step either
    way)."""
    fn = octave_frequencies(low, high, per_octave)
    return fn[fn <= high * (1.0 + 1e-12)]


def held_band(psd: Psd, record: int = 0) -> tuple[float, float]:
    """(low, high), Hz, of what record `record` of `psd` holds above
    zero: the outer edges of its lines with a level above zero, read per
    bin or between breakpoints as the spectrum is. A measured PSD is
    stored on every line from 0 Hz, and its zeros below and above the
    band hold nothing a resonator could respond to."""
    values = np.real(np.asarray(psd.ordinate[record]))
    frequency = np.asarray(psd.abscissa, dtype=float)
    held = np.isfinite(values) & (values > 0.0)
    if psd.interpolation == 'log_log':
        held &= frequency > 0.0
        if not held.any():
            raise ValueError(f'record {record} holds no level above zero')
        return float(frequency[held].min()), float(frequency[held].max())
    left, right = psd.bin_bounds()
    held &= right > 0.0
    if not held.any():
        raise ValueError(f'record {record} holds no level above zero')
    return float(max(left[held].min(), 0.0)), float(right[held].max())


def spectrum_refusal(psd: Any, record: int = 0) -> str | None:
    """Why record `record` of `psd` cannot be read for its ERS and FDS,
    or None when it can: an acceleration autospectrum with its units
    declared, every line finite and none below zero, and some area."""
    from ..units import UNKNOWN
    from .data import Psd, channel_quantities, has_phase

    if not isinstance(psd, Psd):
        return 'the ERS and FDS are read from a PSD or a time history'
    if not 0 <= int(record) < psd.num_records:
        return f'there is no record {record}'
    response, reference = psd.record_pair(record)
    if not psd.is_autospectrum(record) or has_phase(psd.ordinate[record]):
        return (f'record {record} ({response}/{reference}) is a cross '
                'spectrum; the ERS and FDS are read from an autospectrum')
    dimension = psd.ordinate_dim[record]
    if dimension == UNKNOWN:
        return (f'the units of record {record} ({response}) are not '
                'declared, so its levels cannot be read; set its units '
                'first')
    if dimension != DENSITY:
        quantity = channel_quantities(dimension)[0]
        return (f'record {record} ({response}) is a density of '
                f'{quantity}; the ERS and FDS are read from an '
                'acceleration spectrum')
    values = np.real(np.asarray(psd.ordinate[record], dtype=complex))
    frequency = np.asarray(psd.abscissa, dtype=float)
    bad = ~np.isfinite(values)
    if bad.any():
        at = float(frequency[np.argmax(bad)])
        return (f'record {record} ({response}) has {int(bad.sum())} '
                f'line(s) that are not finite, the first at {at:g} Hz')
    below = values < 0.0
    if below.any():
        at = float(frequency[np.argmax(below)])
        return (f'record {record} ({response}) has a line below zero at '
                f'{at:g} Hz; a density cannot be negative')
    area = psd.area(record)
    if not (math.isfinite(area) and area > 0.0):
        return f'record {record} ({response}) has no area: every line is zero'
    return None


def psd_spectra(psd: Psd, record: int = 0, *, b: float | None = None,
                duration: float | None = None, q: float = DEFAULT_Q,
                method: str = METHOD,
                frequencies: ArrayLike | None = None,
                low: float | None = None, high: float | None = None,
                per_octave: int = PER_OCTAVE,
                progress: Callable[[int, int], None] | None = None
                ) -> ResponseSpectra:
    """The ERS and the FDS of one record of an acceleration spectrum.

    Each resonator's response moments are integrated against the
    spectrum as it is read (`response_moments`), and the damage is read
    by `method` (see the module's text). A `Specification` is read the
    same way as a measured PSD, at its target.

    Parameters
    ----------
    psd : Psd
        An acceleration autospectrum, a measured PSD or a specification.
    record : int, default 0
        Which record.
    b : float
        The fatigue exponent, within `B_RANGE`; never assumed.
    duration : float
        The seconds the environment lasts; never assumed.
    q : float, default `srs.DEFAULT_Q`
        The resonators' amplification, within `Q_RANGE`.
    method : str, default 'narrowband'
        How the cycles are read, one of `METHODS`.
    frequencies : array-like, optional
        The natural frequencies, Hz. When omitted, `per_octave` to an
        octave from `low` to `high`, which default to the spectrum's
        extent (a twentieth of its top where it starts at 0 Hz).
    low, high : float, optional
        The band of natural frequencies, Hz.
    per_octave : int, default `srs.PER_OCTAVE`
        Natural frequencies to an octave.
    progress : callable, optional
        Told (done, total) as the natural frequencies are read.

    Returns
    -------
    ResponseSpectra

    Raises
    ------
    ValueError
        When the record cannot be read (`spectrum_refusal`), the method
        is not one of `METHODS`, or Q, b, the duration or the natural
        frequencies are out of sense.
    """
    refused = spectrum_refusal(psd, record)
    if refused is not None:
        raise ValueError(refused)
    if method not in METHODS:
        raise ValueError(f'method {method!r} is not one of {", ".join(METHODS)}')
    q, b = checked_q(q), checked_b(b)
    seconds = checked_duration(duration)
    extent = held_band(psd, record)
    if frequencies is None:
        start = extent[0]
        if start <= 0.0:
            above = np.asarray(psd.abscissa, dtype=float)
            start = float(above[above > 0.0].min())
        fn = _octaves(*_band(low, high, None, (start, extent[1])),
                      per_octave)
    else:
        fn = _natural(frequencies)
    m0, m1, m2, m4 = response_moments(psd, record, fn, q, progress=progress)
    if method == 'narrowband':
        fds = narrowband_damage(m0, m2, b, seconds)
    elif method == 'rice':
        fds = rice_damage(m0, m2, m4, b, seconds)
    else:
        fds = dirlik_damage(m0, m1, m2, m4, b, seconds)
    rms = np.sqrt(m0)
    crossings = np.sqrt(m2 / m0)
    ers = rms * np.array([expected_largest_peak(rate, seconds)
                          for rate in crossings])
    notes: list[str] = []
    quiet = ~np.isfinite(ers)
    if quiet.any():
        notes.append(
            f'{int(quiet.sum())} resonator(s) cross zero once or less in '
            f'{_seconds(seconds)}, where the expected largest peak says '
            'nothing; their ERS is not read')
    outside = (fn < extent[0]) | (fn > extent[1])
    if outside.any():
        notes.append(
            f'{int(outside.sum())} natural frequenc'
            f'{"y lies" if outside.sum() == 1 else "ies lie"} outside the '
            f'spectrum ({extent[0]:.4g} to {extent[1]:.4g} Hz), where a '
            'resonator responds to the spectrum off its resonance')
    return ResponseSpectra(
        frequency=fn, fds=fds, ers=ers, rms=rms, crossings=crossings,
        peaks=np.sqrt(m4 / m2), q=q, b=b, duration=seconds,
        method=method, source='spectrum',
        dimension='acceleration', label=psd.record_label(record),
        ers_duration=seconds, notes=tuple(notes))


def response_filter(natural: ArrayLike, sample_rate: float,
                    q: float = DEFAULT_Q) -> tuple[np.ndarray, np.ndarray]:
    """The recursion for each resonator's pseudo-acceleration, given the
    base acceleration sampled at `sample_rate`: (b, a), each
    (frequencies, 3), for `scipy.signal.lfilter`.

    Exact for an input that is a straight line between samples. With
    w = -z, the state x = (w, dw/dt) obeys dx/dt = A x + B u, where
    A = [[0, 1], [-wn^2, -2 zeta wn]] and B = (0, 1), and the response
    is wn^2 w. Over one sample with u a ramp of slope s, the state
    (x, u, s) moves by the exponential of [[A, B, 0], [0, 0, 1],
    [0, 0, 0]] dt, which gives x[k+1] = Phi x[k] + P u[k] + R u[k+1];
    its transfer function to y = C x is a second-order filter whose
    denominator is the characteristic polynomial of Phi. The gain at
    zero frequency is exactly one.
    """
    from scipy.linalg import expm

    fn = _natural(natural)
    zeta = damping_for(checked_q(q))
    dt = 1.0 / float(sample_rate)
    b = np.empty((fn.size, 3))
    a = np.empty((fn.size, 3))
    for k, f in enumerate(fn):
        wn = 2.0 * math.pi * f
        m = np.zeros((4, 4))
        m[0, 1] = 1.0
        m[1, 0], m[1, 1], m[1, 2] = -wn * wn, -2.0 * zeta * wn, 1.0
        m[2, 3] = 1.0
        e = expm(m * dt)
        phi, g1, g2 = e[:2, :2], e[:2, 2], e[:2, 3]
        p, r = g1 - g2 / dt, g2 / dt
        # y = wn^2 w reads the first state alone, so of the adjugate of
        # (z I - Phi), z I + [[-Phi22, Phi12], [Phi21, -Phi11]], only the
        # first row enters the numerator
        row = np.array([-phi[1, 1], phi[0, 1]])
        b[k] = wn * wn * np.array([r[0], p[0] + row @ r, row @ p])
        a[k] = (1.0, -np.trace(phi), np.linalg.det(phi))
    return b, a


def turning_points(values: ArrayLike) -> np.ndarray:
    """The peaks and valleys of `values`, with its first and last
    samples: runs of equal samples are one, and a sample that continues
    its neighbors' direction is dropped."""
    x = np.asarray(values, dtype=float).ravel()
    if x.size == 0:
        return x
    x = x[np.concatenate([[True], np.diff(x) != 0.0])]
    if x.size < 3:
        return x
    direction = np.sign(np.diff(x))
    turn = np.concatenate([[True], direction[1:] != direction[:-1], [True]])
    return x[turn]


def rainflow(values: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """The rainflow count of `values`: (ranges, counts), each cycle's
    range and 1 for a closed cycle or 0.5 for a half cycle of the
    residue.

    Reduced to its turning points, a range that is no larger than
    either range beside it closes a cycle, which is taken out with its
    two points; the rule is applied to every such range at once, a pass
    at a time, leaving a neighbor of one taken in the same pass to the
    next, until none is left. The residue, a run whose ranges grow and
    then shrink, is counted as half cycles.
    """
    p = turning_points(values)
    ranges: list[np.ndarray] = []
    while p.size >= 4:
        d = np.abs(np.diff(p))
        inner = d[1:-1]
        closing = (inner <= d[:-2]) & (inner <= d[2:])
        if not closing.any():
            break
        # two neighbors close only when their ranges are equal; take the
        # first, and the next pass takes the other
        closing[1:] &= ~closing[:-1]
        at = np.nonzero(closing)[0] + 1
        ranges.append(d[at])
        gone = np.zeros(p.size, dtype=bool)
        gone[at] = True
        gone[at + 1] = True
        p = p[~gone]
    closed = np.concatenate(ranges) if ranges else np.empty(0)
    residue = np.abs(np.diff(p))
    return (np.concatenate([closed, residue]),
            np.concatenate([np.ones(closed.size),
                            np.full(residue.size, 0.5)]))


def rainflow_damage(values: ArrayLike, b: float) -> float:
    """The sum over the rainflow cycles of `values` of (range / 2)^b,
    half cycles at one half: the damage with C = 1."""
    ranges, counts = rainflow(values)
    return float(np.sum(counts * (0.5 * ranges) ** float(b)))


def _sampling_note(b: np.ndarray, a: np.ndarray, natural: float,
                   rate: float, q: float) -> str:
    """How low the top resonator reads, in words: its gain at
    resonance against Q, and its peaks between samples."""
    from scipy.signal import freqz

    _w, h = freqz(b, a, worN=[natural], fs=rate)
    gain = abs(complex(h[0])) / q
    peaks = 1.0 - math.cos(math.pi * natural / rate)
    return (f'the top resonator, {natural:.4g} Hz, reads its input '
            f'{100 * (1 - gain):.3g} % low at resonance and its peaks up to '
            f'{100 * peaks:.3g} % low between samples')


def history_refusal(history: Any, record: int = 0) -> str | None:
    """Why record `record` of `history` cannot be read for its ERS and
    FDS, or None when it can: an evenly sampled acceleration with its
    units declared, every sample finite, and some variation."""
    from ..units import UNKNOWN
    from .data import TimeHistory

    if not isinstance(history, TimeHistory):
        return 'the ERS and FDS are read from a PSD or a time history'
    if not 0 <= int(record) < history.num_records:
        return f'there is no record {record}'
    name = history.response_dof[record]
    dimension = history.ordinate_dim[record]
    if dimension == UNKNOWN:
        return (f'the units of record {record} ({name}) are not declared, '
                'so its levels cannot be read; set its units first')
    if dimension != 'acceleration':
        return (f'record {record} ({name}) is a record of {dimension}; the '
                'ERS and FDS are read from an acceleration')
    try:
        history.sample_rate  # noqa: B018 - raises on uneven sampling
    except ValueError as uneven:
        return str(uneven)
    values = np.asarray(history.ordinate[record], dtype=float)
    if values.size < 4:
        return f'record {record} ({name}) holds {values.size} samples'
    bad = ~np.isfinite(values)
    if bad.any():
        at = float(np.asarray(history.abscissa, dtype=float)[np.argmax(bad)])
        return (f'record {record} ({name}) has {int(bad.sum())} sample(s) '
                f'that are not finite, the first at {at:g} s')
    if np.ptp(values) == 0.0:
        return (f'record {record} ({name}) does not vary (every sample '
                f'{values[0]:g}): there is no vibration to read')
    return None


def history_spectra(history: TimeHistory, record: int = 0, *,
                    b: float | None = None, q: float = DEFAULT_Q,
                    duration: float | None = None,
                    frequencies: ArrayLike | None = None,
                    low: float | None = None, high: float | None = None,
                    per_octave: int = PER_OCTAVE,
                    progress: Callable[[int, int], None] | None = None
                    ) -> ResponseSpectra:
    """The ERS and the FDS of one channel of an acceleration time
    history, counted by rainflow in each resonator's response.

    The channel is leveled to its mean and run through each resonator
    (`response_filter`) from rest; the response's rainflow cycles give
    the FDS (`rainflow_damage`) and its largest excursion the ERS. Both
    are over the record's length; `duration`, when given, scales the
    FDS to it (`ResponseSpectra.at_duration`).

    Parameters
    ----------
    history : TimeHistory
        An evenly sampled acceleration record.
    record : int, default 0
        Which channel.
    b : float
        The fatigue exponent, within `B_RANGE`; never assumed.
    q : float, default `srs.DEFAULT_Q`
        The resonators' amplification, within `Q_RANGE`.
    duration : float, optional
        The seconds the FDS is scaled to; the record's length when
        omitted.
    frequencies : array-like, optional
        The natural frequencies, Hz, none above a twentieth of the
        rate. When omitted, `per_octave` to an octave from `low` to
        `high`, which default to ten cycles in the record and a twentieth of
        the sample rate.
    low, high : float, optional
        The band of natural frequencies, Hz.
    per_octave : int, default `srs.PER_OCTAVE`
        Natural frequencies to an octave.
    progress : callable, optional
        Told (done, total) as the natural frequencies are read.

    Returns
    -------
    ResponseSpectra

    Raises
    ------
    ValueError
        When the record cannot be read (`history_refusal`), or Q, b,
        the duration or the natural frequencies are out of sense.
    """
    from scipy.signal import lfilter

    refused = history_refusal(history, record)
    if refused is not None:
        raise ValueError(refused)
    q, b = checked_q(q), checked_b(b)
    rate = history.sample_rate
    values = np.asarray(history.ordinate[record], dtype=float)
    length = values.size / rate
    top = rate / SAMPLES_PER_CYCLE
    if frequencies is None:
        fn = _octaves(*_band(low, high, top, (10.0 / length, top)),
                      per_octave)
    else:
        fn = _natural(frequencies)
    if fn[-1] > top * (1.0 + 1e-9):
        raise ValueError(
            f'a natural frequency of {fn[-1]:g} Hz is above a twentieth of the '
            f'sample rate ({top:g} Hz), where the response\'s peaks fall '
            'between samples')
    leveled = values - values.mean()
    coefficients, denominators = response_filter(fn, rate, q)
    fds = np.empty(fn.size)
    ers = np.empty(fn.size)
    rms = np.empty(fn.size)
    crossings = np.empty(fn.size)
    peaks = np.empty(fn.size)
    ticker = Ticker(progress)
    ticker.add(fn.size)
    for k in range(fn.size):
        y = lfilter(coefficients[k], denominators[k], leveled)
        fds[k] = rainflow_damage(y, b)
        ers[k] = float(np.max(np.abs(y)))
        rms[k] = float(np.sqrt(np.mean(y * y)))
        # the record was leveled and the gain at zero frequency is one,
        # so the response sits on zero
        crossings[k] = np.count_nonzero((y[:-1] < 0.0) & (y[1:] >= 0.0)
                                        ) / length
        slope = np.diff(y)
        peaks[k] = np.count_nonzero(
            (slope[:-1] > 0.0) & (slope[1:] <= 0.0)) / length
        ticker.tick()
    notes = [(f'the record was leveled to its mean '
             f'({values.mean():.4g} in SI) first, as its spectrum would be')]
    notes.append(_sampling_note(coefficients[-1], denominators[-1],
                                fn[-1], rate, q))
    few = fn * length < 100.0
    if few.any():
        notes.append(
            f'{int(few.sum())} resonator(s) below {100.0 / length:.4g} Hz '
            'complete fewer than 100 cycles in the record, so their '
            'counts rest on few cycles')
    result = ResponseSpectra(
        frequency=fn, fds=fds, ers=ers, rms=rms, crossings=crossings,
        peaks=peaks, q=q, b=b, duration=length, method='rainflow',
        source='time history', dimension=history.ordinate_dim[record],
        label=history.record_label(record), ers_duration=length,
        notes=tuple(notes))
    if duration is not None:
        result = result.at_duration(duration)
    return result


def fatigue_spectra(data: Psd | TimeHistory, record: int = 0,
                    **settings: Any) -> ResponseSpectra:
    """The ERS and FDS of one record of a PSD (`psd_spectra`) or of a
    time history (`history_spectra`), whichever `data` is.

    Parameters
    ----------
    data : Psd or TimeHistory
        The record's object.
    record : int, default 0
        Which record.
    **settings
        As `psd_spectra` or `history_spectra` take them; a time history
        takes no `method`, and is refused one other than 'rainflow'.

    Returns
    -------
    ResponseSpectra

    Raises
    ------
    ValueError
        As the two raise, and for anything that is neither.
    """
    from .data import Psd, TimeHistory

    if isinstance(data, TimeHistory):
        method = settings.pop('method', None)
        if method not in (None, 'rainflow'):
            raise ValueError(f'a time history is counted by rainflow; '
                             f'method {method!r} is for a spectrum')
        return history_spectra(data, record, **settings)
    if isinstance(data, Psd):
        return psd_spectra(data, record, **settings)
    raise ValueError('the ERS and FDS are read from a PSD or a time history')


def compression_ratio(factor: float, b: float, on: str = 'asd') -> float:
    """t2 / t1, the service time a test time stands for, at exaggeration
    `factor` on the ASD (`on='asd'`, (W1/W2)^(b/2)) or on the rms
    (`on='rms'`, (S1/S2)^b). Raises ValueError for a factor that is not
    positive or an `on` that is neither."""
    b = checked_b(b)
    factor = float(factor)
    if not (math.isfinite(factor) and factor > 0.0):
        raise ValueError(f'the exaggeration factor {factor:g} is not positive')
    if on not in CAP:
        raise ValueError(f"on {on!r} is neither 'asd' nor 'rms'")
    return factor ** (b / 2.0 if on == 'asd' else b)


def exaggeration(time_ratio: float, b: float, on: str = 'asd') -> float:
    """The exaggeration factor on the ASD (or the rms) that makes a test
    `time_ratio` times shorter than the service do the same damage:
    the inverse of `compression_ratio`."""
    b = checked_b(b)
    ratio = float(time_ratio)
    if not (math.isfinite(ratio) and ratio > 0.0):
        raise ValueError(f'the time ratio {ratio:g} is not positive')
    if on not in CAP:
        raise ValueError(f"on {on!r} is neither 'asd' nor 'rms'")
    return ratio ** (2.0 / b if on == 'asd' else 1.0 / b)


@dataclass(frozen=True)
class TimeCompression:
    """A test shortened at a raised level with equal damage: the time
    ratio, the exaggeration factor on the ASD, and the cap it is judged
    against."""

    #: the fatigue exponent
    b: float
    #: t2 / t1: the service time over the test time
    time_ratio: float
    #: W1 / W2: the test's ASD over the service's
    asd_factor: float
    #: the most the ASD factor may be
    cap: float = CAP['asd']

    @property
    def rms_factor(self) -> float:
        """The exaggeration on the rms, the square root of the ASD's."""
        return math.sqrt(self.asd_factor)

    @property
    def decibels(self) -> float:
        """The exaggeration on the ASD in decibels."""
        return 10.0 * math.log10(self.asd_factor)

    @property
    def over_cap(self) -> bool:
        """Whether the ASD factor is over the cap."""
        return self.asd_factor > self.cap

    @property
    def at_cap(self) -> float:
        """The time ratio the cap allows at this exponent."""
        return self.cap ** (self.b / 2.0)

    @property
    def rests_on(self) -> str:
        """What the trade assumes, in words."""
        return (f'damage proportional to time and to the rms raised to '
                f'b = {self.b:g}: fatigue the only mechanism, the article '
                'linear between the two levels, and a Gaussian, stationary '
                'environment')

    def describe(self) -> str:
        """The trade in one line, with the cap's verdict."""
        verb = 'compresses' if self.time_ratio >= 1.0 else 'extends'
        ratio = (self.time_ratio if self.time_ratio >= 1.0
                 else 1.0 / self.time_ratio)
        said = (f'{verb} time {ratio:.4g} to 1 at {self.asd_factor:.4g} on '
                f'the ASD ({self.decibels:+.2f} dB, {self.rms_factor:.4g} '
                f'on the rms), b = {self.b:g}')
        if self.over_cap:
            return (said + f'; over the cap of {self.cap:g} on the ASD, '
                    f'which allows {self.at_cap:.4g} to 1')
        return said + f'; within the cap of {self.cap:g} on the ASD'


def time_compression(*, b: float | None = None,
                     time_ratio: float | None = None,
                     factor: float | None = None,
                     cap: float = CAP['asd']) -> TimeCompression:
    """The equal-damage trade of time for level at exponent `b`, from
    the time ratio (service over test) or from the exaggeration factor
    on the ASD, whichever is given.

    Parameters
    ----------
    b : float
        The fatigue exponent, within `B_RANGE`.
    time_ratio : float, optional
        t2 / t1, the service time over the test time; above one
        shortens the test.
    factor : float, optional
        W1 / W2, the test's ASD over the service's.
    cap : float, default 4
        The most the ASD factor may be, `CAP['asd']`.

    Returns
    -------
    TimeCompression

    Raises
    ------
    ValueError
        Unless exactly one of `time_ratio` and `factor` is given, or
        when b, the given one or the cap is out of sense.
    """
    if (time_ratio is None) == (factor is None):
        raise ValueError('give the time ratio or the exaggeration factor, '
                         'one of the two')
    cap = float(cap)
    if not (math.isfinite(cap) and cap >= 1.0):
        raise ValueError(f'the cap {cap:g} on the ASD is below one')
    if factor is None:
        ratio = float(time_ratio)
        return TimeCompression(checked_b(b), ratio, exaggeration(ratio, b),
                               cap)
    factor = float(factor)
    return TimeCompression(checked_b(b), compression_ratio(factor, b),
                           factor, cap)


@dataclass(frozen=True)
class Equivalence:
    """A test set against a field environment, resonator by resonator,
    at one Q and b."""

    #: the field environment's result
    field: ResponseSpectra
    #: the candidate test's result
    test: ResponseSpectra
    #: the cap on the ASD the exaggeration is judged against
    cap: float = CAP['asd']

    @property
    def frequency(self) -> np.ndarray:
        """The natural frequencies, Hz."""
        return self.field.frequency

    @property
    def damage_ratio(self) -> np.ndarray:
        """The test's FDS over the field's, one per natural frequency."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.test.fds / self.field.fds

    @property
    def damage_db(self) -> np.ndarray:
        """The same in decibels, 10 log10, the scale a density is read
        on: at one duration, 3 dB more on the ASD is 3 b / 2 dB more
        damage."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return 10.0 * np.log10(self.damage_ratio)

    @property
    def ers_db(self) -> np.ndarray:
        """The test's ERS over the field's in decibels, 20 log10."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return 20.0 * np.log10(self.test.ers / self.field.ers)

    @property
    def exaggeration(self) -> np.ndarray:
        """The factor on the ASD the test carries over the field at each
        natural frequency, in the sense of the time-compression rule:
        (damage ratio times field time over test time)^(2 / b)."""
        time = self.field.duration / self.test.duration
        return (self.damage_ratio * time) ** (2.0 / self.field.b)

    @property
    def time_ratio(self) -> float:
        """The field's duration over the test's."""
        return self.field.duration / self.test.duration

    @property
    def _judged(self) -> np.ndarray:
        # a field resonator with no damage gives no ratio to judge; a
        # test resonator with none where the field has some is judged,
        # at zero
        return np.isfinite(self.damage_ratio)

    @property
    def governs(self) -> float:
        """The natural frequency where the test's FDS falls furthest
        below the field's (or stands least above it), Hz."""
        ratio = np.where(self._judged, self.damage_ratio, np.inf)
        return float(self.frequency[int(np.argmin(ratio))])

    @property
    def least_ratio(self) -> float:
        """The smallest damage ratio over the natural frequencies."""
        return float(np.min(self.damage_ratio[self._judged]))

    @property
    def covers(self) -> bool:
        """Whether the test's FDS is at or above the field's at every
        natural frequency."""
        return self.least_ratio >= 1.0

    @property
    def ers_covers(self) -> bool:
        """Whether the test's ERS is at or above the field's at every
        natural frequency read."""
        db = self.ers_db
        db = db[np.isfinite(db)]
        return bool(db.size) and float(db.min()) >= 0.0

    @property
    def factor_needed(self) -> float:
        """The factor on the test's ASD, at its own duration, that
        brings its FDS to the field's at every natural frequency; below
        one when it has room to spare, infinite where the test does no
        damage at all."""
        if self.least_ratio == 0.0:
            return math.inf
        return self.least_ratio ** (-2.0 / self.field.b)

    @property
    def duration_needed(self) -> float:
        """The test's duration, at its own level, at which its FDS
        reaches the field's at every natural frequency, s; infinite
        where the test does no damage at all."""
        if self.least_ratio == 0.0:
            return math.inf
        return self.test.duration / self.least_ratio

    @property
    def over_cap(self) -> np.ndarray:
        """The natural frequencies where the test's exaggeration is over
        the cap, Hz."""
        judged = self._judged & (self.exaggeration > self.cap)
        return self.frequency[judged]

    def compression(self) -> TimeCompression:
        """The trade the test makes at the governing frequency: the
        field's time over the test's, at the exaggeration it carries
        there."""
        index = int(np.argmin(np.where(self._judged, self.damage_ratio,
                                       np.inf)))
        return TimeCompression(self.field.b, self.time_ratio,
                               float(self.exaggeration[index]), self.cap)

    @property
    def rests_on(self) -> str:
        """What the comparison assumes, in words."""
        return (f'both read at Q = {self.field.q:g} and b = '
                f'{self.field.b:g}; field: {self.field.rests_on}; test: '
                f'{self.test.rests_on}')

    def describe(self) -> str:
        """The comparison in one line."""
        if self.least_ratio == 0.0:
            said = (f'the test does no damage at {self.governs:.4g} Hz, '
                    'where the field does')
        elif self.covers:
            said = (f'the test covers the field\'s damage at every natural '
                    f'frequency, least {10 * math.log10(self.least_ratio):+.2f} '
                    f'dB at {self.governs:.4g} Hz')
        else:
            said = (f'the test falls short of the field\'s damage at '
                    f'{self.governs:.4g} Hz by '
                    f'{-10 * math.log10(self.least_ratio):.2f} dB: '
                    f'{self.factor_needed:.4g} on its ASD, or '
                    f'{_seconds(self.duration_needed)} at its level, covers it')
        if self.over_cap.size:
            said += (f'; exaggerated over the cap of {self.cap:g} at '
                     f'{self.over_cap.size} natural frequenc'
                     f'{"y" if self.over_cap.size == 1 else "ies"}')
        if not self.ers_covers:
            said += '; its ERS falls below the field\'s'
        return said


def equivalence(field: ResponseSpectra, test: ResponseSpectra, *,
                cap: float = CAP['asd']) -> Equivalence:
    """Set a candidate test against a field environment.

    Parameters
    ----------
    field, test : ResponseSpectra
        The two results, read at the same natural frequencies, Q and b,
        from records of one quantity.
    cap : float, default 4
        The most the exaggeration on the ASD may be, `CAP['asd']`.

    Returns
    -------
    Equivalence

    Raises
    ------
    ValueError
        When the two were read at different natural frequencies, Q or
        b, are of different quantities, or no natural frequency has a
        damage on both sides.
    """
    if field.dimension != test.dimension:
        raise ValueError(f'the field is of {field.dimension} and the test of '
                         f'{test.dimension}, which cannot be compared')
    if field.q != test.q or field.b != test.b:
        raise ValueError(f'the field is read at Q = {field.q:g}, b = '
                         f'{field.b:g} and the test at Q = {test.q:g}, b = '
                         f'{test.b:g}; read both at one Q and b')
    if (field.frequency.shape != test.frequency.shape
            or not np.allclose(field.frequency, test.frequency,
                               rtol=1e-9, atol=0.0)):
        raise ValueError('the field and the test are read at different '
                         'natural frequencies; read both on one set')
    result = Equivalence(field, test, float(cap))
    if not result._judged.any():
        raise ValueError('no natural frequency has a damage on both sides')
    return result


def compared(field: Any, test: Any, field_record: int = 0,
             test_record: int = 0, *,
             field_duration: float | None = None,
             test_duration: float | None = None,
             cap: float = CAP['asd'],
             frequencies: ArrayLike | None = None,
             low: float | None = None, high: float | None = None,
             per_octave: int = PER_OCTAVE,
             progress: Callable[[int, int], None] | None = None,
             **settings: Any) -> Equivalence:
    """Read a field record and a candidate test on one set of natural
    frequencies and set them against each other (`equivalence`).

    The natural frequencies are the field's band when none are given,
    and both are read on them. A spectrum's duration is its
    `field_duration` or `test_duration`; a time history's is its own
    length, scaled when one is given.

    Parameters
    ----------
    field, test : Psd or TimeHistory
        The two records' objects.
    field_record, test_record : int, default 0
        Which record of each.
    field_duration, test_duration : float, optional
        The seconds each lasts; required for a spectrum.
    cap : float, default 4
        As for `equivalence`.
    frequencies, low, high, per_octave
        As for `psd_spectra`, for the field.
    progress : callable, optional
        Told (done, total) as both are read.
    **settings
        `b`, `q` and `method`, as for `fatigue_spectra`; a time
        history's side ignores `method`.

    Returns
    -------
    Equivalence

    Raises
    ------
    ValueError
        As `fatigue_spectra` and `equivalence` raise.
    """
    from .data import TimeHistory

    # the test's count is told after the field's, which it starts at
    offset = [0]

    def told(done: int, total: int) -> None:
        if progress is not None:
            progress(offset[0] + done, offset[0] + total)

    def side(data: Any, record: int, duration: float | None,
             natural: ArrayLike | None) -> ResponseSpectra:
        given = dict(settings)
        if isinstance(data, TimeHistory):
            given.pop('method', None)
        return fatigue_spectra(data, record, duration=duration,
                               frequencies=natural, low=low, high=high,
                               per_octave=per_octave, progress=told,
                               **given)

    first = side(field, field_record, field_duration, frequencies)
    offset[0] = first.frequency.size
    second = side(test, test_record, test_duration, first.frequency)
    return equivalence(first, second, cap=cap)


def settings_of(spectra: ResponseSpectra) -> dict[str, Any]:
    """The settings `fatigue_spectra` takes that give `spectra` again
    from its record, for a journal line or a script."""
    said: dict[str, Any] = {'b': spectra.b, 'q': spectra.q}
    if spectra.source == 'spectrum':
        said['method'] = spectra.method
    said['duration'] = spectra.duration
    return said
