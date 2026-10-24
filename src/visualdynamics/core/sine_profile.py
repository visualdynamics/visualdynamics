"""A sine profile written the way test documents write one: segments of
constant displacement, velocity or acceleration, with the crossovers
between them solved and checked.

A sine profile is a short table of segments, each holding one quantity
constant between two frequencies. At the low end the quantity is
usually displacement, since held at a constant acceleration a tone's
displacement grows as one over the frequency squared and the stroke runs
out; in the middle it is sometimes velocity; above that it is
acceleration. The three are tied together at every frequency by one
relation. For a tone of frequency f and peak displacement x,

    v = 2 pi f x,        a = (2 pi f)**2 x,

so a quantity of power n (0 for acceleration, 1 for velocity, 2 for
displacement) held at a peak level L is the acceleration
a = (2 pi f)**n L (`convert`). Where a segment of one quantity meets a
segment of another, the two give the same amplitude at one frequency,
the crossover:

    2 pi f_c = (L_2 / L_1)**(1 / (n_1 - n_2))      (`crossover`)

which for a displacement x below an acceleration a is the familiar
f_c = sqrt(a / x) / (2 pi). With x in inches peak to peak and a in g
peak it is the hand form f_c = sqrt(a / (0.0511 x)): 0.0511 is
(2 pi)**2 / 2 / 386.09, the half from peak to peak and the 386.09 in/s**2
of one g.

`SineProfile` takes the segments as the document writes them: each one's
quantity and level in its own unit and, for a displacement, its
convention (peak or peak to peak), which is asked for rather than
assumed because a double amplitude read as peak doubles the motion. A
boundary between two segments may be written or left to be solved.
Where it is written and the segments are of different quantities, the
crossover is solved anyway and checked against it: a written breakpoint
more than `CROSSOVER_TOLERANCE` from the solved crossover is a level or a
convention typed wrong, and the check names it with the step the profile
would take there. A double amplitude entered as peak on a displacement
of 0.20 in below a 2 g segment moves the crossover from 14.0 Hz to
9.9 Hz; 0.91 mm read as 0.91 in moves 33 Hz to 6.6 Hz. Two segments of
the same quantity meet in a step and have no crossover; their boundary
must be written, and the step is reported.

The acceleration the profile asks for is what a controller runs, and
`SineProfile.tone` writes it as a `SineTone`. A tone interpolates its
amplitude linearly in sweep time between breakpoints (`SineTone.target`),
which follows a constant acceleration exactly but not a constant
displacement or velocity, so such a segment is written as enough
breakpoints that the interpolated level stays within `ACCURACY` of it.
The tone is built on the solved crossovers, which the written figures
round, so it is continuous; a profile whose written breakpoints do not
check, or that steps, is refused there, since a sweep cannot jump.

`read_profile` goes the other way: a tone already written in breakpoints
(a controller's file, or one built here) is read back as segments, each
stretch classified by the slope of its acceleration on log axes (12 dB
per octave for constant displacement, 6 for velocity, flat for
acceleration), and its breakpoints checked as crossovers like any
written profile.

Everything is stored in SI: levels as peak values, displacement in m,
velocity in m/s and acceleration in m/s**2.

References
----------
The relation between the three quantities of a tone and the crossover
frequency between segments are set out in these.

1. Piersol, A. G., & Paez, T. L. (Eds.) (2010). *Harris' Shock and
   Vibration Handbook*, 6th ed. McGraw-Hill. Sinusoidal motion: peak
   displacement, velocity and acceleration of a tone as one amplitude
   times successive powers of its angular frequency.
2. Department of Defense (2019). *MIL-STD-810H, Environmental
   Engineering Considerations and Laboratory Tests*. Method 514.8,
   Annex E, Figure 514.8E-2: a sinusoidal profile written in segments of
   displacement and acceleration (0.20 in double amplitude from 5 to
   14 Hz, 2 g from 14 to 33 Hz, 0.036 in to 52.1 Hz, 5 g to 500 Hz),
   each crossover where the two levels meet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..units import dimension_of, si_factor
from .sine import LINEAR, LOG, SineSweepSpecification, SineTone

#: the three quantities a segment can hold constant, with the power of
#: the angular frequency that takes each to an acceleration
POWERS = {'displacement': 2, 'velocity': 1, 'acceleration': 0}

#: the two ways a displacement is written: half the swing, or all of it
#: (double amplitude)
CONVENTIONS = ('peak', 'peak to peak')

#: how far, as a share of the solved crossover, a written breakpoint may
#: stand from it and still check: one percent, room for a document's
#: rounding to three figures (52.1 Hz written for 52.13), and about a
#: twenty-ninth of the 29 % a double amplitude entered as peak moves a
#: crossover
CROSSOVER_TOLERANCE = 0.01

#: the most the interpolated level of a built tone may stand from the
#: profile, as a share of it: one percent, a tenth of the amplitude
#: tolerance test documents commonly ask for
ACCURACY = 0.01

#: how far the slope of a tone's stretch, in dB per octave, may stand
#: from flat, 6.02 or 12.04 and still be read as acceleration, velocity
#: or displacement held constant: 0.3 dB per octave, room for breakpoint
#: levels rounded to two figures (0.26 g written at 5 Hz for 0.2556)
SLOPE_TOLERANCE = 0.3

#: one power of frequency on log axes, in dB per octave: 20 log10 2
OCTAVE_DB = 20.0 * float(np.log10(2.0))

#: the most breakpoints one segment is written as; a segment that would
#: need more is asked of a tighter accuracy than the arithmetic can serve
MOST_PIECES = 1000

#: the sweep laws a built tone can follow, by the name `SineProfile.tone`
#: takes, with the segment type each writes
LAWS = {'log': LOG, 'linear': LINEAR}


def _a(word: str) -> str:
    """`word` with its article: 'an acceleration', 'a velocity'."""
    return f'{"an" if word[:1] in "aeiou" else "a"} {word}'


def _power(quantity: str) -> int:
    """The power of 2 pi f that takes `quantity` to an acceleration."""
    try:
        return POWERS[quantity]
    except KeyError:
        raise ValueError(f'{quantity!r} is not a quantity a sine segment '
                         f'holds; it is one of {", ".join(POWERS)}') from None


def convert(value: Any, frequency: Any, source: str,
            target: str = 'acceleration') -> np.ndarray:
    """A tone's peak level in one quantity as its peak level in another,
    at its frequency: `value` (2 pi f)**(n_source - n_target).

    The one place the relation between displacement, velocity and
    acceleration of a tone is written; every reading of a profile goes
    through it. A level that is not a number stays not a number.

    Raises ValueError for a quantity that is not one of `POWERS` or a
    frequency that is not more than zero.

    Parameters
    ----------
    value : float or array-like
        The peak level, SI.
    frequency : float or array-like
        Hz, more than zero; broadcast against `value`.
    source, target : str
        'displacement', 'velocity' or 'acceleration'.

    Returns
    -------
    ndarray
        The peak level in `target`, SI.
    """
    power = _power(source) - _power(target)
    frequency = np.asarray(frequency, dtype=float)
    if not np.all(frequency > 0.0):
        raise ValueError('a tone\'s level is converted at a frequency more '
                         'than zero')
    return np.asarray(value, dtype=float) * (2.0 * np.pi * frequency) ** power


def crossover(first: tuple[str, float], second: tuple[str, float]
              ) -> float | None:
    """The frequency at which two segments, each (quantity, peak level in
    SI), ask for the same amplitude; None when they hold the same
    quantity, which meet in a step rather than cross.

    Raises ValueError for a level that is not a finite number more than
    zero.

    Parameters
    ----------
    first, second : tuple of (str, float)
        Each segment's quantity and its peak level, SI.

    Returns
    -------
    float or None
        Hz.
    """
    (q1, level1), (q2, level2) = first, second
    n1, n2 = _power(q1), _power(q2)
    for level in (level1, level2):
        if not (np.isfinite(level) and level > 0.0):
            raise ValueError(f'a level of {level!r} meets nothing; a level '
                             'is a finite number more than zero')
    if n1 == n2:
        return None
    return float((level2 / level1) ** (1.0 / (n1 - n2)) / (2.0 * np.pi))


@dataclass(frozen=True, kw_only=True)
class ProfileSegment:
    """One segment of a sine profile: a quantity held at a peak level.

    `quantity` is 'displacement', 'velocity' or 'acceleration' and
    `level` its peak value in SI (m, m/s, m/s**2): a displacement written
    peak to peak is stored as half of it. `start` and `end` are the
    frequencies, Hz, the document writes it between, None where a
    boundary is left to be solved. `written` is the level as the document
    wrote it, unit and convention included, for messages and the table.
    Raises ValueError for an unknown quantity, a level that is not a
    finite number more than zero, or a frequency that is not, and when
    `start` is not below `end`. Build one with `displacement`,
    `velocity` or `acceleration`, which convert from the document's unit
    and refuse a unit of the wrong quantity.
    """

    quantity: str
    level: float
    start: float | None = None
    end: float | None = None
    written: str = ''

    def __post_init__(self) -> None:
        _power(self.quantity)
        if not (np.isfinite(self.level) and self.level > 0.0):
            raise ValueError(f'{_a(self.quantity)} of {self.level!r}: a level '
                             'is a finite number more than zero')
        for name in ('start', 'end'):
            value = getattr(self, name)
            if value is not None and not (np.isfinite(value) and value > 0.0):
                raise ValueError(f'{_a(self.quantity)} segment {name}ing at '
                                 f'{value!r} Hz: a frequency is a finite '
                                 'number more than zero')
        if (self.start is not None and self.end is not None
                and not self.start < self.end):
            raise ValueError(f'{_a(self.quantity)} segment from {self.start:g} '
                             f'to {self.end:g} Hz: it starts below where it '
                             'ends')

    def acceleration(self, frequency: Any) -> np.ndarray:
        """The peak acceleration this segment asks for at `frequency`,
        m/s**2 (`convert`)."""
        return convert(self.level, frequency, self.quantity)

    def describe(self) -> str:
        """The segment as the document wrote it, such as '0.2 in peak to
        peak displacement'."""
        return self.written or f'{self.level:.4g} SI {self.quantity}'


def _si_level(value: Any, unit: str, quantity: str) -> float:
    """`value` in `unit` as SI, refusing a unit of another quantity."""
    wanted = 'length' if quantity == 'displacement' else quantity
    found = dimension_of(str(unit))
    if found != wanted:
        said = 'no quantity it can read' if found is None else found
        raise ValueError(f'{unit!r} is {said}, not {_a(wanted)}: {_a(quantity)} '
                         f'segment is written in a unit of {wanted}')
    value = float(value)
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f'{_a(quantity)} of {value!r} {unit}: a level is a '
                         'finite number more than zero')
    return value * si_factor(str(unit), wanted)


def displacement(value: float, unit: str, convention: str, *,
                 start: float | None = None,
                 end: float | None = None) -> ProfileSegment:
    """A segment of constant displacement, as the document writes it.

    The convention is required, never assumed: test documents write
    displacement as double amplitude (peak to peak) as often as peak, and
    the one read as the other doubles or halves the motion and moves
    every crossover it takes part in. Raises ValueError for a unit that
    is not a length, a convention that is not one of `CONVENTIONS`, or a
    level or frequency that is not a finite number more than zero.

    Parameters
    ----------
    value : float
        The displacement as written.
    unit : str
        Its unit, a length ('in', 'mm', 'm').
    convention : str
        'peak' or 'peak to peak' (double amplitude).
    start, end : float, optional
        Hz, where the document writes the segment as starting and ending;
        left out where the boundary is to be solved.

    Returns
    -------
    ProfileSegment
        The segment, its level the peak displacement in m.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f'a displacement written {convention!r}: say '
                         'which it is, "peak" or "peak to peak" (double '
                         'amplitude)')
    level = _si_level(value, unit, 'displacement')
    if convention == 'peak to peak':
        level /= 2.0
    return ProfileSegment(quantity='displacement', level=level, start=start,
                          end=end, written=f'{float(value):g} {unit} '
                          f'{convention} displacement')


def velocity(value: float, unit: str, *, start: float | None = None,
             end: float | None = None) -> ProfileSegment:
    """A segment of constant peak velocity, as the document writes it.

    Raises ValueError for a unit that is not a velocity, or a level or
    frequency that is not a finite number more than zero.

    Parameters
    ----------
    value : float
        The peak velocity as written.
    unit : str
        Its unit, a velocity ('in/s', 'm/s').
    start, end : float, optional
        Hz, as for `displacement`.

    Returns
    -------
    ProfileSegment
        The segment, its level in m/s.
    """
    return ProfileSegment(quantity='velocity',
                          level=_si_level(value, unit, 'velocity'),
                          start=start, end=end,
                          written=f'{float(value):g} {unit} velocity')


def acceleration(value: float, unit: str, *, start: float | None = None,
                 end: float | None = None) -> ProfileSegment:
    """A segment of constant peak acceleration, as the document writes
    it.

    Raises ValueError for a unit that is not an acceleration, or a level
    or frequency that is not a finite number more than zero.

    Parameters
    ----------
    value : float
        The peak acceleration as written.
    unit : str
        Its unit, an acceleration ('g', 'm/s**2').
    start, end : float, optional
        Hz, as for `displacement`.

    Returns
    -------
    ProfileSegment
        The segment, its level in m/s**2.
    """
    return ProfileSegment(quantity='acceleration',
                          level=_si_level(value, unit, 'acceleration'),
                          start=start, end=end,
                          written=f'{float(value):g} {unit} acceleration')


@dataclass(frozen=True, kw_only=True)
class Crossing:
    """Where segment `index` meets segment `index + 1`.

    `written` is the boundary the document wrote, Hz, or None; `solved`
    the crossover the two levels give, or None for two segments of one
    quantity, which step rather than cross. `used` is where the profile
    changes segment: the solved crossover wherever there is one, so the
    profile is continuous, else the written boundary. `step` is the
    ratio of the upper segment's acceleration to the lower's at the
    written boundary (at `used` where nothing is written): 1 where they
    meet, more than 1 where the profile would jump up. `tolerance` is
    the share of the solved crossover a written boundary may stand from
    it.
    """

    index: int
    lower: ProfileSegment
    upper: ProfileSegment
    written: float | None
    solved: float | None
    used: float
    step: float
    tolerance: float

    @property
    def kind(self) -> str:
        """'crossover' where the two quantities differ, else 'step'."""
        return 'step' if self.solved is None else 'crossover'

    @property
    def off(self) -> float | None:
        """How far the written boundary stands from the solved crossover,
        as a share of it; None where either is missing."""
        if self.written is None or self.solved is None:
            return None
        return (self.written - self.solved) / self.solved

    @property
    def checks(self) -> bool:
        """Whether the written boundary is within `tolerance` of the
        solved crossover. A boundary left to be solved checks, and so
        does a step, which has no crossover to check against (it is
        reported as a step instead)."""
        off = self.off
        return off is None or abs(off) <= self.tolerance

    def describe(self) -> str:
        """The crossing in one line, such as 'crossover at 13.93 Hz,
        written 14 Hz: checks' or 'crossover at 9.889 Hz, written 14 Hz
        (+41.6 %): does not check, a step of 0.499 there'."""
        if self.solved is None:
            if abs(self.step - 1.0) <= 1e-9:
                return f'joined at {self.used:.4g} Hz'
            return (f'step at {self.used:.4g} Hz: {self.step:.3g} times '
                    'the acceleration below it')
        text = f'crossover at {self.solved:.4g} Hz'
        if self.written is None:
            return text + ', solved'
        text += f', written {self.written:.4g} Hz'
        if self.checks:
            return text + ': checks'
        return (text + f' ({100.0 * self.off:+.1f} %): does not check, a step '
                f'of {self.step:.3g} there')


class SineProfile:
    """A sine profile as segments of constant displacement, velocity or
    acceleration, with the crossovers solved and checked.

    The first segment's start and the last's end must be written; a
    boundary between two segments may be written on either side, both
    (when they agree), or neither, to be solved. Raises ValueError when
    there are no segments, an outer frequency is missing, the two sides
    of a boundary are written differently, two segments of one quantity
    meet with no boundary written, or the boundaries are not in order
    (two levels whose crossover falls outside the band between them do
    not meet there), and TypeError for an item that is not a
    `ProfileSegment`.

    Attributes:
        segments: The `ProfileSegment` list, low to high.
        crossings: One `Crossing` per boundary, low to high.
        tolerance: The share of a solved crossover a written boundary
            may stand from it (`CROSSOVER_TOLERANCE`).
    """

    def __init__(self, segments: Sequence[ProfileSegment], *,
                 tolerance: float = CROSSOVER_TOLERANCE) -> None:
        self.segments: list[ProfileSegment] = list(segments)
        tolerance = float(tolerance)
        if not (np.isfinite(tolerance) and tolerance >= 0.0):
            raise ValueError(f'a crossover tolerance of {tolerance!r}: it is '
                             'a share of the crossover, zero or more')
        self.tolerance: float = tolerance
        if not self.segments:
            raise ValueError('a sine profile has at least one segment')
        for segment in self.segments:
            if not isinstance(segment, ProfileSegment):
                raise TypeError(f'{segment!r} is not a ProfileSegment; '
                                 'build one with displacement, velocity or '
                                 'acceleration')
        first, last = self.segments[0], self.segments[-1]
        if first.start is None:
            raise ValueError('the first segment says where the profile '
                             f'starts: give {first.describe()} a start')
        if last.end is None:
            raise ValueError('the last segment says where the profile ends: '
                             f'give {last.describe()} an end')
        self.crossings: list[Crossing] = []
        bounds = [first.start]
        for i, (lower, upper) in enumerate(zip(self.segments[:-1],
                                               self.segments[1:])):
            written = self._written(i, lower, upper)
            solved = crossover((lower.quantity, lower.level),
                               (upper.quantity, upper.level))
            if written is None and solved is None:
                raise ValueError(
                    f'segments {i + 1} and {i + 2} are both '
                    f'{lower.quantity} and meet in a step, with no crossover '
                    'to solve: write the frequency where one ends')
            used = solved if solved is not None else written
            at = used if written is None else written
            step = float(upper.acceleration(at) / lower.acceleration(at))
            self.crossings.append(Crossing(
                index=i, lower=lower, upper=upper, written=written,
                solved=solved, used=float(used), step=step,
                tolerance=tolerance))
            bounds.append(float(used))
        bounds.append(last.end)
        for i in range(len(self.segments)):
            if not bounds[i] < bounds[i + 1]:
                raise ValueError(
                    f'segment {i + 1} ({self.segments[i].describe()}) would '
                    f'run from {bounds[i]:.4g} to {bounds[i + 1]:.4g} Hz: '
                    'the levels around it do not meet inside the band, so '
                    'one of them, its unit or its convention is not what '
                    'the document says')
        self._bounds = np.asarray(bounds, dtype=float)

    @staticmethod
    def _written(i: int, lower: ProfileSegment,
                 upper: ProfileSegment) -> float | None:
        """The boundary the document wrote between two segments, from
        either side; refused when the two sides disagree."""
        sides = [value for value in (lower.end, upper.start)
                 if value is not None]
        if len(sides) == 2 and not np.isclose(sides[0], sides[1],
                                              rtol=1e-9, atol=0.0):
            raise ValueError(
                f'segment {i + 1} ends at {sides[0]:g} Hz and segment '
                f'{i + 2} starts at {sides[1]:g} Hz: a boundary is one '
                'frequency')
        return float(sides[0]) if sides else None

    # ---- what it asks for --------------------------------------------------

    @property
    def bounds(self) -> np.ndarray:
        """The frequencies the segments change at, Hz, low to high, the
        band's two ends included: the solved crossovers where there are
        any, the written boundaries of steps."""
        return self._bounds.copy()

    @property
    def band(self) -> tuple[float, float]:
        """(low, high), Hz."""
        return float(self._bounds[0]), float(self._bounds[-1])

    @property
    def checks(self) -> bool:
        """Whether every written breakpoint checks against its crossover
        (`Crossing.checks`)."""
        return all(crossing.checks for crossing in self.crossings)

    @property
    def continuous(self) -> bool:
        """Whether the profile changes segment without a step anywhere:
        every boundary a crossover (not two segments of one quantity at
        different levels) and every written one checks."""
        return self.checks and all(
            crossing.solved is not None or abs(crossing.step - 1.0) <= 1e-9
            for crossing in self.crossings)

    def segment_of(self, frequencies: Any) -> np.ndarray:
        """The index of the segment each frequency falls in, -1 outside
        the band. A frequency on a boundary belongs to the segment above
        it, and the band's top to the last segment."""
        f = np.asarray(frequencies, dtype=float)
        index = np.searchsorted(self._bounds, f, side='right') - 1
        index = np.where(f == self._bounds[-1], len(self.segments) - 1, index)
        inside = (f >= self._bounds[0]) & (f <= self._bounds[-1])
        return np.where(inside, index, -1)

    def level(self, frequencies: Any, quantity: str = 'acceleration',
              convention: str = 'peak') -> np.ndarray:
        """What the profile asks for at each frequency in `quantity`, SI:
        each segment's own level carried to `quantity` by `convert`. Not
        a number outside the band, where the profile asks for nothing.

        Parameters
        ----------
        frequencies : array-like
            Hz.
        quantity : str, default 'acceleration'
            'displacement', 'velocity' or 'acceleration'.
        convention : str, default 'peak'
            For a displacement, 'peak' or 'peak to peak'; ignored for the
            other two.

        Returns
        -------
        ndarray
            The level, the shape of `frequencies`.
        """
        _power(quantity)
        if convention not in CONVENTIONS:
            raise ValueError(f'a convention of {convention!r}: it is "peak" '
                             'or "peak to peak"')
        f = np.asarray(frequencies, dtype=float)
        index = self.segment_of(f)
        out = np.full(f.shape, np.nan)
        for k, segment in enumerate(self.segments):
            here = index == k
            if here.any():
                out[here] = convert(segment.level, f[here], segment.quantity,
                                    quantity)
        if quantity == 'displacement' and convention == 'peak to peak':
            out *= 2.0
        return out

    def maxima(self) -> dict[str, tuple[float, float]]:
        """The most the profile asks for in each quantity and where:
        {'displacement': (Hz, m peak to peak), 'velocity': (Hz, m/s
        peak), 'acceleration': (Hz, m/s**2 peak)}. These are what go
        against a shaker's stroke, velocity and force ratings. Each
        quantity is a power law within a segment, so its largest value is
        at one of the segment's ends; both sides of a step are counted."""
        best: dict[str, tuple[float, float]] = {}
        for k, segment in enumerate(self.segments):
            for f in (self._bounds[k], self._bounds[k + 1]):
                for quantity in POWERS:
                    value = float(convert(segment.level, f, segment.quantity,
                                          quantity))
                    if quantity == 'displacement':
                        value *= 2.0
                    if quantity not in best or value > best[quantity][1]:
                        best[quantity] = (float(f), value)
        return best

    # ---- as a controller runs it -------------------------------------------

    def breakpoints(self, law: str = 'log', accuracy: float = ACCURACY
                    ) -> tuple[np.ndarray, np.ndarray]:
        """The profile as acceleration breakpoints a tone interpolates
        within `accuracy` of it: (Hz, m/s**2 peak), ascending.

        A tone interpolates its amplitude linearly in sweep time, which is
        linear in log frequency on a logarithmic sweep and in frequency on
        a linear one (`SineTone.target`). A constant acceleration needs
        only its two ends, and so does a constant velocity on a linear
        sweep; any other segment is cut into equal steps of the sweep's
        own frequency axis, as few as keep the interpolated level within
        `accuracy` of the profile everywhere along it. Raises ValueError
        for a law not in `LAWS`, an accuracy that is not more than zero,
        or a profile that is not continuous (`continuous`), since a
        sweep cannot jump.
        """
        if law not in LAWS:
            raise ValueError(f'a sweep law of {law!r}: it is one of '
                             f'{", ".join(LAWS)}')
        accuracy = float(accuracy)
        if not (np.isfinite(accuracy) and accuracy > 0.0):
            raise ValueError(f'an accuracy of {accuracy!r}: it is a share of '
                             'the level, more than zero')
        self._refuse_discontinuity()
        frequencies = [float(self._bounds[0])]
        for k, segment in enumerate(self.segments):
            low, high = float(self._bounds[k]), float(self._bounds[k + 1])
            nodes = _nodes(segment, low, high, law, accuracy)
            frequencies.extend(float(f) for f in nodes[1:])
        f = np.asarray(frequencies)
        # each breakpoint carries the segment it starts, and the last the
        # segment it ends; on a crossover the two agree
        a = self.level(f)
        return f, a

    def _refuse_discontinuity(self) -> None:
        bad = [c for c in self.crossings if not c.checks]
        if bad:
            raise ValueError('the profile does not check: '
                             + '; '.join(c.describe() for c in bad)
                             + '. Correct the level, its unit or its '
                             'convention before a tone is written from it')
        steps = [c for c in self.crossings
                 if c.solved is None and abs(c.step - 1.0) > 1e-9]
        if steps:
            raise ValueError('the profile steps ('
                             + '; '.join(c.describe() for c in steps)
                             + '), and a sweep cannot jump')

    def tone(self, name: str, rate: float, *, law: str = 'log',
             descending: bool = False, start_time: float = 0.0,
             channels: int = 1, accuracy: float = ACCURACY) -> SineTone:
        """The tone a controller runs for this profile: its acceleration
        as breakpoints (`breakpoints`) swept at one rate.

        Raises ValueError as `breakpoints` does, and for a rate that is
        not a finite number more than zero or a channel count under one.

        Parameters
        ----------
        name : str
            The tone's name.
        rate : float
            oct/min on a logarithmic sweep, Hz/s on a linear one; more
            than zero, whichever way it sweeps.
        law : str, default 'log'
            'log' or 'linear'.
        descending : bool, default False
            Sweep from the top of the band down.
        start_time : float, default 0.0
            Seconds after the environment starts that the tone turns on.
        channels : int, default 1
            How many control channels the level is asked of, each the
            same.
        accuracy : float, default `ACCURACY`
            As for `breakpoints`.

        Returns
        -------
        SineTone
            Acceleration breakpoints in m/s**2.
        """
        rate = float(rate)
        if not (np.isfinite(rate) and rate > 0.0):
            raise ValueError(f'a sweep rate of {rate!r}: it is a finite '
                             'number more than zero; say descending=True to '
                             'sweep down')
        channels = int(channels)
        if channels < 1:
            raise ValueError('a tone is asked of one control channel or more')
        f, a = self.breakpoints(law, accuracy)
        if descending:
            f, a, rate = f[::-1], a[::-1], -rate
        return SineTone(name, start_time, f, np.tile(a[:, None], (1, channels)),
                        [LAWS[law]] * (len(f) - 1), [rate] * (len(f) - 1))

    def specification(self, response_dof: Sequence[str], rate: float, *,
                      name: str = 'Sine Tone 1', law: str = 'log',
                      descending: bool = False, accuracy: float = ACCURACY,
                      comment: str = '') -> SineSweepSpecification:
        """The profile as a `SineSweepSpecification` of one tone
        (`tone`) asked of every control channel in `response_dof`, in
        m/s**2.

        Raises ValueError as `tone` does, and when `response_dof` is
        empty.
        """
        dofs = [str(dof) for dof in response_dof]
        if not dofs:
            raise ValueError('a specification controls at least one channel')
        tone = self.tone(name, rate, law=law, descending=descending,
                         channels=len(dofs), accuracy=accuracy)
        return SineSweepSpecification([tone], dofs,
                                      ordinate_dim='acceleration',
                                      ordinate_unit='m/s**2', comment=comment)

    def describe(self) -> list[str]:
        """One line per segment and per crossing, low to high, ending with
        the verdict: 'every crossover checks' or how many do not."""
        lines = []
        for k, segment in enumerate(self.segments):
            lines.append(f'{self._bounds[k]:.4g} to {self._bounds[k + 1]:.4g} '
                         f'Hz: {segment.describe()}')
            if k < len(self.crossings):
                lines.append(self.crossings[k].describe())
        bad = sum(not c.checks for c in self.crossings)
        lines.append('every crossover checks' if not bad else
                     f'{bad} crossover{"s" * (bad != 1)} do'
                     f'{"es" * (bad == 1)} not check')
        return lines

    def __repr__(self) -> str:
        low, high = self.band
        return (f'<SineProfile: {len(self.segments)} segments, '
                f'{low:g}-{high:g} Hz>')


def _nodes(segment: ProfileSegment, low: float, high: float, law: str,
           accuracy: float) -> np.ndarray:
    """The fewest equal steps of the sweep's frequency axis from `low` to
    `high` over which a tone's interpolation of the segment's
    acceleration stays within `accuracy` of it."""
    power = _power(segment.quantity)
    if power == 0 or (power == 1 and law == 'linear'):
        return np.array([low, high])
    probe = np.linspace(0.0, 1.0, 65)[1:-1]
    for pieces in range(1, MOST_PIECES + 1):
        if law == 'log':
            # a power law on equal steps of log frequency errs by the same
            # share on every step, so the first stands for them all
            nodes = np.geomspace(low, high, pieces + 1)
            f0, f1 = nodes[:1], nodes[1:2]
            f = f0[:, None] * (f1 / f0)[:, None] ** probe
        else:
            # on equal steps of frequency the share is largest at the low
            # end, where the curve bends most against its level; all are
            # read rather than lean on that
            nodes = np.linspace(low, high, pieces + 1)
            f0, f1 = nodes[:-1], nodes[1:]
            f = f0[:, None] + (f1 - f0)[:, None] * probe
        a0, a1 = f0[:, None] ** power, f1[:, None] ** power
        chord = a0 + (a1 - a0) * probe
        if float(np.max(np.abs(chord / f ** power - 1.0))) <= accuracy:
            return nodes
    raise ValueError(f'{segment.describe()} from {low:g} to {high:g} Hz needs '
                     f'more than {MOST_PIECES} breakpoints to hold within '
                     f'{100.0 * accuracy:g} %; ask for a looser accuracy')


def _tone_column(specification: SineSweepSpecification, tone: str | None,
                 channel: str | None) -> tuple[SineTone, int, str]:
    """The tone, the column of the control channel and the quantity a
    profile is read from, refusing a specification whose units are not
    declared or whose quantity is no displacement, velocity or
    acceleration, and a tone or channel it does not hold."""
    if specification.ordinate_unit is None:
        raise ValueError('the specification\'s units are not declared, so its '
                         'levels are no displacement, velocity or '
                         'acceleration yet; declare them first')
    quantity = specification.ordinate_dim
    if quantity not in POWERS:
        raise ValueError(f'the specification holds {quantity}, not a '
                         'displacement, velocity or acceleration')
    if tone is not None and tone not in [t.name for t in specification.tones]:
        raise ValueError(f'the specification holds no tone {tone!r}; it holds '
                         f'{[t.name for t in specification.tones]}')
    chosen = specification.tones[0] if tone is None else specification.tone(tone)
    dofs = [str(dof) for dof in specification.response_dof]
    if channel is not None and str(channel) not in dofs:
        raise ValueError(f'the specification controls no {channel}; it '
                         f'controls {dofs}')
    column = 0 if channel is None else dofs.index(str(channel))
    return chosen, column, quantity


def interpolated(specification: SineSweepSpecification,
                 tone: str | None = None, channel: str | None = None,
                 lines: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """A tone's own level as it runs it, carried to acceleration: its
    breakpoints interpolated linearly in sweep time (`SineTone.target`)
    at `lines` frequencies evenly in log across its band. Beside the
    profile `read_profile` reads from the same tone, it shows where the
    breakpoints are too far apart for the segment they write.

    Raises ValueError as `read_profile` does for the units, the tone and
    the channel, and for fewer than two lines.

    Parameters
    ----------
    specification : SineSweepSpecification
        Declared units.
    tone : str, optional
        The tone's name; the first by default.
    channel : str, optional
        The control channel whose column is read; the first by default.
    lines : int, default 400
        How many frequencies.

    Returns
    -------
    tuple of ndarray
        (Hz, m/s**2 peak).
    """
    chosen, column, quantity = _tone_column(specification, tone, channel)
    if int(lines) < 2:
        raise ValueError('a level is read at two lines or more')
    low, high = float(chosen.frequency.min()), float(chosen.frequency.max())
    f = np.geomspace(low, high, int(lines))
    return f, convert(chosen.target(f)[:, column], f, quantity)


def read_profile(specification: SineSweepSpecification,
                 tone: str | None = None, channel: str | None = None, *,
                 slope_tolerance: float = SLOPE_TOLERANCE,
                 tolerance: float = CROSSOVER_TOLERANCE) -> SineProfile:
    """Read a tone already written as breakpoints back as segments of
    constant displacement, velocity or acceleration, its breakpoints
    written where it changes from one to the next.

    Each stretch between two breakpoints is classified by the slope of
    its acceleration on log axes: 12.04 dB per octave holds the
    displacement constant, 6.02 the velocity and a flat stretch the
    acceleration, each within `slope_tolerance`. A stretch that holds none
    of them is refused by name with its slope: a breakpoint typed at the
    wrong frequency or level bends the stretches either side of it off
    every one of the three, so that refusal is the check a tone's own
    breakpoints get. The level a stretch holds is read at its geometric
    middle, so a slope a little off splits the difference between its
    ends, and the crossovers the levels solve to are checked against the
    breakpoints as for any profile.
    Consecutive stretches holding the same quantity at the same level,
    within `tolerance`, are one segment, as a built tone writes a
    displacement in several breakpoints. The breakpoints where one
    segment gives way to another are the profile's written boundaries.

    Raises ValueError when the specification's units are not declared or
    its quantity is not one of `POWERS`, for a tone or channel it does not
    hold, for a tone that turns back or holds a frequency (a dwell is no
    profile), for a level at a breakpoint that is not a finite number
    more than zero, for a slope tolerance that is not a finite number
    of zero or more, and for a stretch whose slope holds none of the
    three quantities, naming it in dB per octave.

    Parameters
    ----------
    specification : SineSweepSpecification
        Declared units.
    tone : str, optional
        The tone's name; the first by default.
    channel : str, optional
        The control channel whose column is read; the first by default.
    slope_tolerance : float, default `SLOPE_TOLERANCE`
        dB per octave.
    tolerance : float, default `CROSSOVER_TOLERANCE`
        As for `SineProfile`.

    Returns
    -------
    SineProfile
        Every boundary written, from the tone's breakpoints.
    """
    chosen, column, quantity = _tone_column(specification, tone, channel)
    slope_tolerance = float(slope_tolerance)
    if not (np.isfinite(slope_tolerance) and slope_tolerance >= 0.0):
        raise ValueError(f'a slope tolerance of {slope_tolerance!r} dB per '
                         'octave: it is zero or more')
    f = np.asarray(chosen.frequency, dtype=float)
    steps = np.diff(f)
    if chosen.dwell_segments():
        raise ValueError(f'{chosen.name} holds a frequency (a dwell), which '
                         'has no profile over frequency')
    if not (np.all(steps > 0.0) or np.all(steps < 0.0)):
        raise ValueError(f'{chosen.name} turns back on itself; read it one '
                         'direction at a time')
    values = np.asarray(chosen.amplitude[:, column], dtype=float)
    if not (np.all(np.isfinite(values)) and np.all(values > 0.0)):
        bad = int(np.flatnonzero(~(np.isfinite(values) & (values > 0.0)))[0])
        raise ValueError(f'{chosen.name}: a level of {values[bad]:g} at '
                         f'{f[bad]:g} Hz; every breakpoint of a profile is a '
                         'finite number more than zero')
    if steps[0] < 0.0:
        f, values = f[::-1], values[::-1]
    a = convert(values, f, quantity)
    by_power = {power: name for name, power in POWERS.items()}
    segments: list[tuple[str, float, float, float]] = []
    for i in range(len(f) - 1):
        slope = float(np.log(a[i + 1] / a[i]) / np.log(f[i + 1] / f[i]))
        power = round(slope)
        if (power not in by_power
                or abs(slope - power) * OCTAVE_DB > slope_tolerance):
            raise ValueError(
                f'{chosen.name}: from {f[i]:g} to {f[i + 1]:g} Hz the level '
                f'rises at {OCTAVE_DB * slope:.1f} dB per octave, '
                'which holds none of displacement, velocity or '
                'acceleration constant')
        name = by_power[power]
        middle = float(np.sqrt(f[i] * f[i + 1]))
        level = float(convert(np.sqrt(a[i] * a[i + 1]), middle,
                              'acceleration', name))
        if (segments and segments[-1][0] == name
                and abs(level / segments[-1][1] - 1.0) <= tolerance):
            kind, held, start, _end = segments[-1]
            segments[-1] = (kind, held, start, float(f[i + 1]))
        else:
            segments.append((name, level, float(f[i]), float(f[i + 1])))
    return SineProfile([
        ProfileSegment(quantity=name, level=level, start=start, end=end,
                       written=_si_words(name, level))
        for name, level, start, end in segments], tolerance=tolerance)


def _si_words(quantity: str, level: float) -> str:
    """A level read off a tone, in words: '0.00508 m peak to peak
    displacement', '19.6 m/s**2 acceleration'."""
    if quantity == 'displacement':
        return f'{2.0 * level:.4g} m peak to peak displacement'
    unit = 'm/s' if quantity == 'velocity' else 'm/s**2'
    return f'{level:.4g} {unit} {quantity}'


__all__ = ['ACCURACY', 'CONVENTIONS', 'CROSSOVER_TOLERANCE', 'OCTAVE_DB',
           'POWERS',
           'SLOPE_TOLERANCE', 'Crossing', 'ProfileSegment', 'SineProfile',
           'acceleration', 'convert', 'crossover', 'displacement',
           'interpolated', 'read_profile', 'velocity']
