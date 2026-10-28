"""A shock a shaker can play, synthesized to meet a shock response
spectrum: a sum of wavelets, their amplitudes iterated until every point
of the spectrum is inside the requirement's band, then read against the
shaker's shock ratings.

An SRS requirement leaves the waveform free. Many time histories have
one spectrum, and which one is played decides the event's length, its
peak acceleration and the travel it asks of the shaker. So a synthesis
is two questions at once: does the waveform's spectrum meet the
requirement, and can the shaker play the waveform. Both are answered
here, with the package's own readings: the spectrum by `core.srs` and
judged by `compliance.srs_tolerance`, the travel, velocity and force by
`core.stroke.read_stroke` against a `core.shaker.ShakerRating`.

**The wavelet.** Each component is a tone of frequency f windowed by
one half cycle of a slower sine, N half cycles of the tone long, with N
odd and at least three:

    w(t) = A sin(πt/T) sin(2πft),   0 ≤ t ≤ T = N / (2f),

and zero outside. It starts and ends at zero acceleration, and both its
integrals come back to zero at its end: written as a difference of two
cosines of (N - 1)π t/T and (N + 1)π t/T, the velocity is a difference
of their sines, zero at T for any whole N, and the displacement a
difference of one minus their cosines, zero at T when N ± 1 are even,
which is N odd. So every wavelet leaves the table where it found it, at
rest and at center, and so does any sum of them: no compensation pulse
is needed, and none is added. On the samples the same holds exactly when
the wavelet spans a whole number of samples, so each wavelet's length
is rounded to whole samples and its tone set from that length, N times
the sample rate over twice the samples, a fraction of a percent off the
point it is for at the top of the band. Its samples then sum to zero
(the velocity's trapezoid rule ends at zero) and are symmetric about
the middle (so the displacement's ends at zero too).

**The iteration.** There is one wavelet per point of the requirement,
at that point's frequency, all of one polarity and centered on one
instant. Each pass computes the spectrum of the sum at the
requirement's points, of the requirement's type and at its Q, and
scales each wavelet's amplitude by the requirement over the spectrum at
its point. The closest pass whose every point lies inside the
requirement's inner band (its warning limits, read as `compliance.srs_tolerance`
reads them, or `compliance.SRS_INNER_DB` without them) is kept, and
the passes go on while they bring the spectrum closer, until it is
within `ACCURACY_DB` everywhere or a pass gains less than `STALL_DB`,
so the record has room inside the band for what the shaker adds to it.
Wavelets a twelfth of an octave apart overlap in frequency:
in opposite polarity they cancel at the oscillators between them, and
the iteration chases the cancellation rather than closing on the
requirement, so one polarity is kept. A requirement no pass meets in
`MAX_PASSES` is refused, naming its worst point.

**The shaker.** The record is read raw, from rest at its first sample,
by `read_stroke`, against the ratings given. A wavelet of more half
cycles builds its oscillator's response over more cycles, so it meets
the same spectrum with a smaller amplitude and asks for less velocity
and travel, and a longer event. The synthesis starts at the fewest half
cycles asked for and, while the shaker cannot hold the result, tries
two more at a time up to `MOST_HALF_CYCLES`, keeping the first that
fits: the shortest event the shaker can play. When none fits, the
requirement is refused, naming the limit that runs out at the closest
attempt and the band of the requirement whose wavelets alone take that
limit: the low end for the travel and the velocity, which are spent at
low frequency, the high end for the force. The spectrum is never split
into a low and a high pulse to fit, since the article's modes are then
loaded separately, and a test that splits it is a different test.

References
----------
The wavelet, its integrals and the iteration are worked out above; this
is where the rules the synthesis keeps are set out.

1. MIL-STD-810H (2019), *Environmental Engineering Considerations and
   Laboratory Tests*, Method 516.8, Shock. Paragraph 2.3.3.1: no
   synthesized transient is unique to a spectrum, a pulse that exceeds
   the exciter calls for a compromise, and splitting a spectrum into a
   low- and a high-frequency pulse is not acceptable. Paragraph
   4.2.2.2: the spectrum is judged point by point against a band on a
   share of the points. Annex A, paragraph 1.1: a record whose spectrum
   is computed is sampled at ten times its highest frequency or more.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .shaker import ShakerRating

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import ShockSpecification, Srs, TimeHistory
    from .stroke import StrokeReading

#: the fewest half cycles a wavelet is given unless asked otherwise. At
#: three a wavelet's displacement is one-sided (it rises from rest and
#: comes back without crossing it), at five it swings both ways, and
#: for the same spectrum at its frequency it asks for less than half
#: the travel
HALF_CYCLES = 5

#: the most half cycles the synthesis tries before it refuses. Each two
#: more lengthen the event by a cycle of the lowest frequency, and the
#: test plan holds the event's length to the requirement's duration
MOST_HALF_CYCLES = 15

#: passes of the iteration before a requirement is refused as one the
#: wavelets cannot meet
MAX_PASSES = 40

#: once every point is inside the band, the iteration stops when the
#: spectrum is this close to the requirement everywhere, in dB
ACCURACY_DB = 0.5

#: or when a pass brings its farthest point less than this much closer,
#: in dB: the wavelets' overlap holds it there
STALL_DB = 0.05

#: the sample rate is at least this many times the highest point: the
#: highest frequency an SRS is computed to here (`TimeHistory.srs_band`)
RATE_RATIO = 5.0

#: the sample rate suggested for a synthesis, as a multiple of its
#: highest point: the rate at which a spectrum is computed from a record
#: (the module's reference)
SUGGESTED_RATIO = 10.0

#: the demands, in the order `StrokeReading.shares` keys them, and the
#: words a refusal names each by
LIMITS = {'stroke': 'the stroke', 'velocity': 'the shock velocity',
          'force': 'the shock force'}


def wavelet(samples: int, half_cycles: int) -> np.ndarray:
    """One wavelet of unit amplitude on its own samples, both ends
    included: sin(πn/M) sin(πNn/M) for n from 0 to M, M = `samples`,
    N = `half_cycles`.

    Its tone is N half cycles over M samples, N times the sample rate
    over 2M in Hz. Its samples sum to zero and are symmetric about the
    middle when N is odd, so integrated twice by the trapezoid rule its
    velocity and its displacement both end at zero.

    Raises
    ------
    ValueError
        If `half_cycles` is not an odd whole number of at least three,
        or the wavelet is shorter than two samples a half cycle.
    """
    n = _half_cycles(half_cycles)
    m = int(samples)
    if m < 2 * n:
        raise ValueError(f'a wavelet of {n} half cycles over {m} samples '
                         'is under two samples a half cycle')
    k = np.arange(m + 1)
    return np.sin(np.pi * k / m) * np.sin(np.pi * n * k / m)


def _half_cycles(value: Any) -> int:
    """`value` as a count of half cycles, or a refusal naming it."""
    if (isinstance(value, bool) or not float(value).is_integer()
            or int(value) < 3 or int(value) % 2 == 0):
        raise ValueError(f'{value!r} half cycles: a wavelet ends at rest '
                         'only on an odd whole number, three or more')
    return int(value)


@dataclass(frozen=True, eq=False)
class SynthesizedChannel:
    """One channel's synthesis: its wavelets, how it was reached, and
    what it asks of the shaker.

    The wavelets are one per point of the requirement, in its order:
    each one's tone as played (`frequency`, Hz), its length and its
    start in samples, and its amplitude in m/s^2.
    """

    #: the channel, as the requirement names it
    dof: str
    #: the requirement's points this channel was synthesized to, Hz
    points: np.ndarray
    #: the requirement at each point, m/s^2
    target: np.ndarray
    #: each wavelet's tone as played, Hz
    frequency: np.ndarray
    #: each wavelet's length, in samples
    samples: np.ndarray
    #: each wavelet's first sample in the record
    start: np.ndarray
    #: each wavelet's amplitude, m/s^2
    amplitude: np.ndarray
    #: the half cycles every wavelet of the synthesis kept has
    half_cycles: int
    #: the passes the iteration took to put every point inside the band
    passes: int
    #: the half cycles tried before this one, each with the share of the
    #: rating it ran out of, as (half cycles, limit, share)
    tried: tuple[tuple[int, str, float], ...]
    #: the record's spectrum at the points, of the requirement's type
    spectrum: np.ndarray
    #: the record read against the ratings
    reading: StrokeReading

    @property
    def worst_db(self) -> float:
        """The spectrum's largest distance from the requirement, in dB
        either way."""
        return float(np.max(np.abs(20.0 * np.log10(self.spectrum
                                                   / self.target))))


@dataclass(frozen=True, eq=False)
class ShockSynthesis:
    """A synthesized shock and what it was synthesized to.

    `history` is the record, one acceleration channel per DOF of the
    requirement synthesized, in SI, from rest to rest with a quiet tail
    of a period of the lowest point so the spectrum's low oscillators
    reach their peaks inside it. `channels` holds each one's wavelets
    and its reading against the shaker.
    """

    #: the record a shaker plays
    history: TimeHistory
    #: the requirement it was synthesized to
    specification: ShockSpecification
    #: what it was checked against
    rating: ShakerRating
    #: one per channel, in the record's order
    channels: list[SynthesizedChannel] = field(default_factory=list)

    @property
    def srs(self) -> Srs:
        """The record's spectrum at the requirement's points, of its
        type and at its Q, one record per channel; NaN at a point
        outside the band a channel's requirement is written on."""
        from .data import Srs

        spec = self.specification
        points = np.asarray(spec.abscissa, dtype=float)
        rows = np.full((len(self.channels), points.size), np.nan)
        for row, c in zip(rows, self.channels, strict=True):
            first = int(np.searchsorted(points, c.points[0]))
            row[first:first + c.points.size] = c.spectrum
        return Srs(points, rows,
                   response_dof=[c.dof for c in self.channels],
                   ordinate_dim=['acceleration'] * len(rows),
                   q=spec.q, kind=spec.kind,
                   comment=[f'synthesized, {c.half_cycles} half cycles'
                            for c in self.channels])

    def tolerance(self) -> list[tuple[str, str | None, dict[str, Any]]]:
        """The spectrum judged against the requirement, every channel,
        as `compliance.srs_tolerance_all` judges a measured one: the
        requirement's own bands and the default share rule."""
        from .compliance import srs_tolerance_all

        return srs_tolerance_all(self.specification, self.srs)

    def describe(self, unit_system: Any = None) -> list[str]:
        """The synthesis in lines, a channel at a time: its wavelets and
        passes, its event length, the band it met, then its stroke
        reading's judgments (`StrokeReading.describe`)."""
        from ..units import DEFAULT_SYSTEM

        us = DEFAULT_SYSTEM if unit_system is None else unit_system
        judged = {dof: result for dof, _event, result in self.tolerance()}
        lines = []
        for c in self.channels:
            rate = self.history.sample_rate
            length = float(c.samples.max()) / rate
            lines.append(f'{c.dof}: {c.points.size} wavelets of '
                         f'{c.half_cycles} half cycles, {c.passes} '
                         f'pass{"es" * (c.passes != 1)}, event '
                         f'{1e3 * length:.4g} ms')
            result = judged.get(c.dof)
            if result is not None:
                lines.append(f'{result["inside_count"]}/{result["points"]} '
                             'points inside the band, within '
                             f'{c.worst_db:.2f} dB of the requirement')
            lines.extend(line for line in c.reading.describe(us)
                         if ' against ' in line)
        return lines


def _points(specification: Any, record: int) -> tuple[np.ndarray,
                                                      np.ndarray]:
    """The requirement's written points and its level at each, in SI;
    a refusal for a gap in the middle, or nothing written."""
    dof = specification.response_dof[record]
    f = np.asarray(specification.abscissa, dtype=float)
    level = np.asarray(np.real(np.atleast_2d(
        specification.ordinate)[record]), dtype=float)
    written = np.isfinite(level) & (level > 0.0) & (f > 0.0)
    if not written.any():
        raise ValueError(f'the requirement at {dof} is not above zero at '
                         'any point: there is nothing to meet')
    first, last = np.flatnonzero(written)[[0, -1]]
    # a point inside the band that is not a level cannot be met and
    # cannot be skipped without a word: the spectrum there would be
    # whatever its neighbors leave
    gap = np.flatnonzero(~written[first:last + 1])
    if gap.size:
        at = f[first + gap[0]]
        raise ValueError(f'the requirement at {dof} is not a level above '
                         f'zero at {at:.4g} Hz, inside the band it is '
                         'written on')
    return f[first:last + 1], level[first:last + 1]


def _layout(points: np.ndarray, rate: float, half_cycles: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(samples, start, tone) of each wavelet: whole samples long,
    centered on the longest within half a sample."""
    # at more than five samples a period of the highest point (the rate
    # is refused otherwise), every wavelet has over two a half cycle
    samples = np.round(half_cycles * rate / (2.0 * points)).astype(int)
    start = (samples.max() - samples) // 2
    tone = half_cycles * rate / (2.0 * samples)
    return samples, start, tone


def _record(amplitude: np.ndarray, samples: np.ndarray, start: np.ndarray,
            half_cycles: int, length: int) -> np.ndarray:
    """The sum of the wavelets on `length` samples."""
    out = np.zeros(length)
    for a, m, s in zip(amplitude, samples, start, strict=True):
        out[s:s + m + 1] += a * wavelet(m, half_cycles)
    return out


def _spectrum(record: np.ndarray, points: np.ndarray, rate: float,
              q: float, kind: str) -> np.ndarray:
    """The record's spectrum at `points`, of `kind`, from `core.srs`."""
    from .srs import peaks, spectra

    return spectra(*peaks(record, points, rate, q=q))[kind]


def _judged(specification: Any, record: int, dof: str, points: np.ndarray,
            spectrum: np.ndarray) -> dict[str, Any]:
    """Every point inside the requirement's inner band, judged by
    `compliance.srs_tolerance`: the share rule at one hundred percent."""
    from .compliance import srs_tolerance
    from .data import Srs

    srs = Srs(points, spectrum[None, :], response_dof=[dof],
              ordinate_dim=['acceleration'], q=specification.q,
              kind=specification.kind)
    return srs_tolerance(specification, srs, record, 0, percent=100.0)


def _band(contribution: np.ndarray, points: np.ndarray, limit: float,
          upward: bool) -> tuple[float, float]:
    """(lowest, highest) Hz of the requirement whose wavelets alone take
    a demand to `limit`: counted from the low end up for the travel and
    the velocity, from the high end down for the force, and the whole
    requirement when no end alone reaches it."""
    order = np.arange(points.size) if upward else np.arange(points.size)[::-1]
    reached = np.flatnonzero(np.cumsum(contribution[order]) >= limit)
    if not reached.size:
        return float(points[0]), float(points[-1])
    edge = points[order[reached[0]]]
    return ((float(points[0]), float(edge)) if upward
            else (float(edge), float(points[-1])))


def _refusal(dof: str, closest: dict[str, Any], tried: Sequence[int],
             rate: float) -> str:
    """Why no attempt fits: the limit the closest one runs out of, what
    the reading says of it, and the band whose wavelets take it there.

    Each wavelet's share of the demand is its own acceleration, or its
    velocity or displacement integrated raw as `read_stroke` integrates
    the record, at the sample where the record's demand peaks, times its
    amplitude and signed toward that peak."""
    from .data import TimeHistory
    from .filters import integrate

    reading = closest['reading']
    shares = reading.shares()
    limit = max((key for key in shares if shares[key] is not None),
                key=lambda key: shares[key])
    n, points = closest['half_cycles'], closest['points']
    length = reading.acceleration.size
    rating = reading.rating
    units = np.zeros((points.size, length))
    for k, (m, s) in enumerate(zip(closest['samples'], closest['start'],
                                   strict=True)):
        units[k, s:s + m + 1] = wavelet(m, n)
    order = {'force': 0, 'velocity': 1, 'stroke': 2}[limit]
    each = TimeHistory(np.arange(length) / rate, units,
                       response_dof=[f'{k}X+' for k in range(points.size)],
                       ordinate_dim=['acceleration'] * points.size)
    for _ in range(order):
        each = integrate(each, None)
    rows = np.asarray(each.ordinate.real, dtype=float)
    series = {'force': reading.acceleration, 'velocity': reading.velocity,
              'stroke': reading.displacement}[limit]
    at = int(np.argmax(np.abs(series)))
    contribution = np.sign(series[at]) * closest['amplitude'] * rows[:, at]
    if limit == 'force':
        allowed = rating.force / rating.moving_mass
        band = _band(contribution, points, allowed, upward=False)
        words = next(line for line in reading.describe()
                     if line.startswith('Force'))
    else:
        allowed = rating.room if limit == 'stroke' else rating.velocity
        band = _band(rating.relative(contribution), points, allowed,
                     upward=True)
        lines = reading.describe()
        words = (lines[0] if limit == 'stroke' else
                 next(line for line in lines
                      if line.startswith('Peak velocity')))
    span = (f'at {band[0]:.4g} Hz' if band[0] == band[1]
            else f'from {band[0]:.4g} to {band[1]:.4g} Hz')
    return (f'{dof}: no synthesis of {_counted(tried)} fits the shaker. '
            'Closest, at '
            f'{n} half cycles, {LIMITS[limit]} runs out ({words}), and the '
            f'requirement {span} alone asks for it')


def _channel(specification: Any, record: int, rate: float,
             rating: ShakerRating, half_cycles: int, ticker: Any
             ) -> tuple[np.ndarray, dict[str, Any]]:
    """One channel synthesized: the record and how it was reached, or a
    refusal."""
    from .data import TimeHistory
    from .stroke import read_stroke

    dof = str(specification.response_dof[record])
    points, target = _points(specification, record)
    if points[-1] * RATE_RATIO >= rate:
        raise ValueError(
            f'the requirement at {dof} runs to {points[-1]:.6g} Hz, and a '
            f'spectrum is computed here to a fifth of the sample rate; '
            f'sample at more than {RATE_RATIO * points[-1]:.6g} Hz')
    tail = int(np.ceil(rate / points[0]))
    tried: list[tuple[int, str, float]] = []
    attempts: list[dict[str, Any]] = []
    unmet: dict[str, Any] | None = None
    for n in range(half_cycles, MOST_HALF_CYCLES + 1, 2):
        samples, start, tone = _layout(points, rate, n)
        length = int(samples.max()) + 1 + tail
        # a lone wavelet of three half cycles takes its oscillator to
        # about two and a half times its amplitude, so half the
        # requirement is a first guess the passes put right
        amplitude = target / 2.0
        met, before = None, np.inf
        ticker.add(MAX_PASSES)
        for passes in range(1, MAX_PASSES + 1):
            row = _record(amplitude, samples, start, n, length)
            got = _spectrum(row, points, rate, specification.q,
                            specification.kind)
            ticker.tick()
            result = _judged(specification, record, dof, points, got)
            worst = float(np.max(np.abs(20.0 * np.log10(got / target))))
            if result['passed'] and (met is None or worst < met[3]):
                met = (passes, row, got, worst, amplitude)
            # inside the band, the passes go on while they bring the
            # spectrum closer, so the record has room inside the band
            # for what the shaker adds to it
            if met is not None and (worst <= ACCURACY_DB
                                    or before - worst < STALL_DB):
                ticker.tick(MAX_PASSES - passes)
                break
            before = worst
            amplitude = amplitude * target / got
        if met is None:
            if unmet is None or (result['worst_margin_db']
                                 < unmet['worst_margin_db']):
                unmet = dict(result, half_cycles=n)
            continue
        passes, row, got, _worst, amplitude = met
        history = TimeHistory(np.arange(length) / rate, row[None, :],
                              response_dof=[dof],
                              ordinate_dim=['acceleration'])
        reading = read_stroke(history, 0, stroke=rating.stroke,
                              velocity=rating.velocity, force=rating.force,
                              moving_mass=rating.moving_mass,
                              body_mass=rating.body_mass)
        attempt = {'half_cycles': n, 'passes': passes, 'row': row,
                   'spectrum': got, 'reading': reading, 'samples': samples,
                   'start': start, 'tone': tone, 'amplitude': amplitude,
                   'points': points, 'target': target,
                   'tried': tuple(tried)}
        if reading.fits is not False:
            return row, attempt
        shares = reading.shares()
        worst = max((key for key in shares if shares[key] is not None),
                    key=lambda key: shares[key])
        tried.append((n, worst, float(shares[worst])))
        attempts.append(attempt)
    counts = list(range(half_cycles, MOST_HALF_CYCLES + 1, 2))
    if attempts:
        closest = min(attempts, key=lambda a: max(
            v for v in a['reading'].shares().values() if v is not None))
        raise ValueError(_refusal(dof, closest, counts, rate))
    raise ValueError(
        f'{dof}: no synthesis of {_counted(counts)} meets the requirement '
        f'in {MAX_PASSES} passes; the closest ({unmet["half_cycles"]} half '
        f'cycles) is {unmet["worst_db"]:+.2f} dB from it at '
        f'{unmet["worst_frequency"]:.4g} Hz, outside its band')


def _counted(counts: Sequence[int]) -> str:
    """The half cycles tried, in words: '15 half cycles' or '5 to 15
    half cycles'."""
    return (f'{counts[0]} half cycles' if len(counts) == 1
            else f'{counts[0]} to {counts[-1]} half cycles')


def synthesize(specification: ShockSpecification, *, sample_rate: float,
               rating: ShakerRating | None = None,
               half_cycles: int = HALF_CYCLES,
               dofs: Sequence[str] | None = None,
               progress: Callable[[int, int], None] | None = None
               ) -> ShockSynthesis:
    """A shock a shaker can play, whose spectrum meets `specification`.

        made = synthesize(spec, sample_rate=20480.0,
                          rating=ShakerRating(stroke=0.0508, velocity=1.78,
                                              force=35600.0,
                                              moving_mass=45.0))
        made.history, made.tolerance(), made.channels[0].reading

    One wavelet per point of the requirement, their amplitudes iterated
    until every point of the record's spectrum lies inside the
    requirement's inner band; then the record is read against `rating`
    raw (`core.stroke.read_stroke`). While the shaker cannot hold it the
    wavelets are lengthened two half cycles at a time, up to
    `MOST_HALF_CYCLES`, and the first that fits is kept. The module
    docstring has the method.

    Parameters
    ----------
    specification : ShockSpecification
        The requirement, an acceleration SRS in declared units, with its
        Q and type; its warning limits are the band every point is put
        inside, `compliance.SRS_INNER_DB` without them.
    sample_rate : float
        Samples per second, above `RATE_RATIO` times the highest point.
    rating : ShakerRating, optional
        The shaker's shock ratings and the masses it carries; a rating
        left out is not judged, and with none the first synthesis that
        meets the requirement is kept.
    half_cycles : int, default `HALF_CYCLES`
        The fewest half cycles each wavelet has: odd, from three to
        `MOST_HALF_CYCLES`.
    dofs : sequence of str, optional
        Which of the requirement's DOFs, in this order; all by default.
    progress : callable, optional
        Called with ``(done, total)`` a pass at a time, the total raised
        as each attempt begins; the app's progress bar uses it.

    Returns
    -------
    ShockSynthesis
        The record and each channel's wavelets, spectrum and reading.

    Raises
    ------
    TypeError
        If `specification` is not a `ShockSpecification`.
    ValueError
        If the requirement is not an acceleration SRS in declared units,
        names no DOF asked for, is not a level above zero at a point
        inside its band, or runs to `RATE_RATIO` times the sample rate's
        fifth; if the half cycles are not odd and from three to
        `MOST_HALF_CYCLES`; if no pass meets the requirement (naming the
        worst point); or if no synthesis fits the shaker (naming the
        limit that runs out and the band that asks for it).
    """
    from ..units import UNKNOWN
    from .data import ShockSpecification
    from .progress import Ticker
    from .signals import _history

    if not isinstance(specification, ShockSpecification):
        raise TypeError('a shock is synthesized to a shock specification, '
                         f'not a {type(specification).__name__}')
    rate = float(sample_rate)
    if not (np.isfinite(rate) and rate > 0.0):
        raise ValueError('the sample rate has to be above zero')
    first = _half_cycles(half_cycles)
    if first > MOST_HALF_CYCLES:
        raise ValueError(f'{first} half cycles is past the most tried, '
                         f'{MOST_HALF_CYCLES}')
    rating = ShakerRating() if rating is None else rating
    own = [str(d) for d in specification.response_dof]
    wanted = (list(dict.fromkeys(own)) if dofs is None
              else [str(d) for d in dofs])
    if not wanted:
        raise ValueError('name at least one DOF to synthesize')
    twice = sorted({d for d in wanted if wanted.count(d) > 1})
    if twice:
        raise ValueError(f'{", ".join(twice)} is named more than once; a '
                         'record holds one channel a DOF')
    missing = [d for d in wanted if d not in own]
    if missing:
        raise ValueError(f'the requirement names no {", ".join(missing)}; '
                         f'it names {", ".join(own)}')
    ticker = Ticker(progress)
    rows, channels, records = [], [], []
    for dof in wanted:
        # the acceleration record at the DOF: one at a DOF can sit beside
        # another quantity's, and two of acceleration are two answers
        at = [i for i, d in enumerate(own) if d == dof]
        held = [i for i in at
                if specification.ordinate_dim[i] == 'acceleration']
        if len(held) > 1:
            raise ValueError(f'the requirement holds {len(held)} '
                             f'acceleration records at {dof}; keep one')
        if not held:
            dimension = specification.ordinate_dim[at[0]]
            if dimension == UNKNOWN:
                raise ValueError(f'the requirement at {dof} does not say '
                                 'what it is a spectrum of; declare its '
                                 'units first')
            raise ValueError(f'a shaker plays acceleration, and the '
                             f'requirement at {dof} is a spectrum of '
                             f'{dimension}')
        record = held[0]
        records.append(record)
        row, a = _channel(specification, record, rate, rating, first, ticker)
        rows.append(row)
        channels.append(SynthesizedChannel(
            dof=dof, points=a['points'], target=a['target'],
            frequency=a['tone'], samples=a['samples'], start=a['start'],
            amplitude=a['amplitude'], half_cycles=a['half_cycles'],
            passes=a['passes'], tried=a['tried'], spectrum=a['spectrum'],
            reading=a['reading']))
    # one clock for every channel: a shorter one keeps its quiet tail
    # longer, still at rest
    longest = max(row.size for row in rows)
    block = np.zeros((len(rows), longest))
    for i, row in enumerate(rows):
        block[i, :row.size] = row
    what = ', '.join(sorted({f'{c.half_cycles} half cycles'
                             for c in channels}))
    history = _history(block, rate, wanted, ['acceleration'] * len(rows),
                       f'SRS synthesis, wavelets of {what}')
    # shown in the requirement's units where it names them, held in SI
    # as every record is
    history.ordinate_unit = [specification.ordinate_unit[r] or shown
                             for r, shown in zip(records,
                                                 history.ordinate_unit,
                                                 strict=True)]
    return ShockSynthesis(history=history, specification=specification,
                          rating=rating, channels=channels)


def suggested_rate(specification: ShockSpecification) -> float:
    """A sample rate for a synthesis: `SUGGESTED_RATIO` times the
    requirement's highest written point, rounded up to a multiple of
    1024 Hz."""
    f = np.asarray(specification.abscissa, dtype=float)
    level = np.abs(np.asarray(specification.ordinate, dtype=float))
    written = f[np.any(np.isfinite(level) & (level > 0.0), axis=0)]
    top = float(written.max()) if written.size else 2000.0
    return float(1024 * int(np.ceil(SUGGESTED_RATIO * top / 1024.0)))
