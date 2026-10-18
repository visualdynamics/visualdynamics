"""Whether a shock pulse fits on the shaker: its stroke, velocity and
force against the shaker's shock ratings.

A shaker can deliver a pulse only if it can deliver three things at
once: the force for the peak, the velocity the pulse leaves the table
with, and the travel the whole event needs, compensation included. The
three are spent by different parts of the pulse. The peak acceleration
costs force (the moving mass times the peak). The duration costs
stroke, since displacement grows with the peak times the duration
squared. The velocity sits between them. So the check reads all three
off the pulse and sets each against its rating.

The pulse is a record or a nominal one. `compensated_half_sine` builds
the nominal pulse a stroke is usually first estimated on: a half-sine
with a half-sine pre-pulse before it and a post-pulse after it, each at
a share of the peak (a fifth by default, `COMPENSATION_SHARE`) and
opposite in sign, each lasting long enough to carry half the main
pulse's velocity change back. The table then starts and ends at rest:
the pre-pulse takes it to minus half the velocity change, the main
pulse to plus half, and the post-pulse back to zero, so its peak
velocity is half the velocity change. At a fifth of the peak the whole
event lasts six durations; at a tenth, eleven.

`read_stroke` integrates the acceleration twice with nothing removed,
the cumulative trapezoid rule of `filters.integrate` called with
``drift_corner=None``, from rest at the record's first sample. It is
the raw path on purpose. The displacement of a compensated pulse is
content at a few hertz, exactly where the default 2 Hz drift high-pass
stops being transparent, and with it the 50 g, 11 ms pulse's stroke
reads about 4 % high and no longer starts and ends at rest. A stroke
estimate is integrated raw, and a record whose velocity does not come
back to rest is a record to fix before its stroke is believed.

The reading is the peak acceleration, the peak velocity, the velocity
change, and the displacement each way from rest with its peak to peak.
The travel, the velocity and the force are judged by the rule in
`core.shaker`, the one the replication fit judges by, and three
corrections are carried with it:

- The shaker body. The ratings are the armature's motion relative to
  the body, and a body on isolation mounts moves the other way by the
  ratio of the moving mass to its own, so the travel that reaches the
  stops, and the velocity the coil moves at, are the table's times one
  plus that ratio. The compensation is low in frequency, where the body
  moves most. Given the body mass with the moving mass, both are
  counted that way.
- The stops each way. The stroke is peak to peak, and the armature
  rests at the middle of it, held there by its suspension, so the stops
  are half the stroke away on each side. The symmetric arrangement
  above travels one way only, to its full peak to peak and back, so it
  can be inside the stroke peak to peak and still reach a stop. The
  stroke share is therefore the larger travel each way from center
  over half the stroke, what the stops see, and the peak to peak is
  reported as a number without a verdict. When the peak to peak fits
  and the share is over, the reading says so: centering the travel is
  the controller's compensation, an asymmetric pre- and post-pulse.
- The first check. The peak to peak of the nominal arrangement is
  close to A T^2 / 2 for a half-sine of peak A and duration T, which in
  the units shock is specified in is about 193 A T^2 inches with A in g
  and T in seconds (`rule_stroke`). The reading reports it beside the
  integrated value, computed for the half-sine of the record's own peak
  and velocity change, so on the nominal pulse the rule reads 0.16 %
  high (the same on every pulse, since the ratio is the arrangement's)
  and on any other pulse the difference is what its shape and its
  compensation are worth.

A controller that shapes its compensation for least displacement does
better than the symmetric arrangement, sometimes by a useful margin,
so the nominal reading is a first check: a pulse it passes with room
fits, and one it puts near a rating sends the operator to the
controller's own prediction and to a measurement.

References
----------
The double integration and the compensated pulse are standard; these
are where the arrangement and the integration are set out.

1. Lalanne, C. (2009). *Mechanical Vibration and Shock Analysis,
   Volume 2: Mechanical Shock*, 2nd ed. ISTE and Wiley. Shocks on a
   shaker with pre- and post-pulses, the velocity and displacement
   they need, and why the table must end at rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .shaker import (
    ShakerRating,
    acceleration_channel,
    each_way,
    finite,
    sentence,
)

if TYPE_CHECKING:                                    # pragma: no cover
    from ..units import UnitSystem
    from .data import TimeHistory

# Asked for on 2026-10-16 from a worked example: the 2 Hz drift
# high-pass `filters.integrate` applies by default read the compensated
# 50 g, 11 ms half-sine's stroke 3.9 % high (1.212 in against 1.1661 in
# raw) and moved its displacement off rest at both ends. The integration
# here is the same raw path `core.shock_check` takes, not a second rule.
#
# On review (2026-10-16) the ratings became `core.shaker.ShakerRating`,
# shared with the replication fit, the mass ratio became the body mass
# that type takes, and the stroke share became the larger one-way travel
# over half the stroke: as peak to peak over the stroke it read
# "within, 58 %" on the line above one saying the pulse reached a stop
# at 117 %. The ISO 18431-2 reference went with it; that part is about
# time windows, not integration.

#: the compensation pulses' height as a share of the main pulse's peak,
#: the share test documents commonly allow. Each compensation pulse then
#: lasts 1 / (2 * share) durations: 2.5 at a fifth
COMPENSATION_SHARE = 0.2

#: the half-sine rule's constant in SI: the peak to peak of the
#: symmetric arrangement compensated at a fifth of its peak is about
#: this times A T^2, with A in m/s^2 and T in s (at another share the
#: arrangement travels further, 1.80 times at a tenth, and the constant
#: does not hold). Times standard gravity over an inch it is 193.04,
#: the "193 A T^2" of US customary units
RULE = 0.5


def compensated_half_sine(peak: float, duration: float,
                          sample_rate: float, *,
                          share: float = COMPENSATION_SHARE,
                          response_dof: str = '1Z+',
                          unit: str | None = None) -> TimeHistory:
    """A half-sine with symmetric half-sine compensation, as a record.

        pulse = compensated_half_sine(50.0, 0.011, 102400.0, unit='g')

    The main pulse, of peak `peak` and duration `duration`, sits between
    a pre-pulse and a post-pulse of the opposite sign, each at `share`
    times the peak and each lasting ``duration / (2 * share)``, so each
    carries back half the main pulse's area. The record opens at the
    pre-pulse's first sample and closes at the post-pulse's last, and
    the table starts and ends it at rest: its net velocity change and
    its net displacement are both zero.

    Parameters
    ----------
    peak : float
        The main pulse's peak, in `unit` (SI when omitted).
    duration : float
        The main pulse's duration, in seconds.
    sample_rate : float
        Samples per second.
    share : float, default `COMPENSATION_SHARE`
        The compensation pulses' height as a share of the peak, between
        zero and one.
    response_dof : str, default '1Z+'
        The channel the record is at.
    unit : str, optional
        The unit `peak` is written in, such as 'g'; the record keeps it
        as its declared unit.

    Returns
    -------
    TimeHistory
        One acceleration channel, stored in SI, on a clock that starts
        at zero.

    Raises
    ------
    ValueError
        If the peak, the duration or the sample rate is not positive,
        the share is not between zero and one, or the main pulse is
        shorter than four samples.
    """
    from ..units import si_transform
    from .data import TimeHistory

    peak, duration, rate, share = (float(peak), float(duration),
                                   float(sample_rate), float(share))
    if not (peak > 0.0 and duration > 0.0 and rate > 0.0):
        raise ValueError('a pulse needs a positive peak, duration and '
                         'sample rate')
    if not 0.0 < share < 1.0:
        raise ValueError(f'the compensation share ({share:g}) is a share '
                         'of the peak, between zero and one')
    if duration * rate < 4.0:
        raise ValueError(f'a {duration * 1e3:g} ms pulse at {rate:g} '
                         'samples a second is fewer than four samples; '
                         'its peak and its area would be guesses')
    if unit is not None:
        scale, offset = si_transform(unit, 'acceleration')
        peak = peak * scale + offset
    each = duration / (2.0 * share)               # 2.5 durations at 0.2
    total = 2.0 * each + duration
    t = np.arange(round(total * rate) + 1) / rate
    # three half-sines, each zero outside its own stretch; the sample
    # grid need not land on the joins, and every pulse is zero at both
    # of its ends, so a sample either side of a join is still exact
    pre = np.where(t < each, -share * peak * np.sin(np.pi * t / each), 0.0)
    inside = (t >= each) & (t <= each + duration)
    main = np.where(inside, peak * np.sin(np.pi * (t - each) / duration),
                    0.0)
    after = t > each + duration
    post = np.where(after, -share * peak
                    * np.sin(np.pi * np.minimum(t - each - duration, each)
                             / each), 0.0)
    return TimeHistory(t, (pre + main + post)[None, :],
                       response_dof=[response_dof],
                       ordinate_dim=['acceleration'],
                       ordinate_unit=None if unit is None else [unit],
                       comment=[(f'half-sine, {duration * 1e3:g} ms, '
                                 f'compensated at {share:g} of its peak')])


def rule_stroke(peak: float, duration: float) -> float:
    """The half-sine rule's peak to peak, in m: `RULE` times A T^2.

    `peak` is in m/s^2 and `duration` in seconds. In inches with the
    peak in g this is 193 A T^2. It reads the peak to peak of a
    half-sine compensated symmetrically at a fifth of its peak 0.16 %
    high, and it is the relation to keep in the head:
    double the peak and the stroke doubles, double the duration and it
    quadruples.
    """
    return RULE * float(peak) * float(duration) ** 2


@dataclass(frozen=True, eq=False)
class StrokeReading:
    """One channel's stroke, velocity and force against the ratings
    given.

    The arrays are the acceleration, and the velocity and displacement
    integrated from it raw, in SI (m/s^2, m/s and m), from rest at the
    record's first sample. `rating` holds the shaker's shock ratings and
    the masses, in SI, with None where none was given; a rating that is
    None is not judged. Its stroke is the shaker's shock stroke, peak to
    peak, the travel of the armature relative to the body.
    """

    #: the channel's DOF, as the record names it
    dof: str
    #: the samples per second the arrays are at
    sample_rate: float
    #: the first sample's time on the record's clock, in seconds
    start: float
    #: the acceleration, m/s^2
    acceleration: np.ndarray
    #: the velocity, m/s, from rest at the first sample
    velocity: np.ndarray
    #: the table's displacement, m, from rest at the first sample
    displacement: np.ndarray
    #: the shaker's shock ratings and the masses it carries
    rating: ShakerRating

    @property
    def time(self) -> np.ndarray:
        """Each sample's time on the record's clock, in seconds."""
        return self.start + np.arange(self.velocity.size) / self.sample_rate

    @property
    def duration(self) -> float:
        """How long the record lasts, first sample to last, in s."""
        return (self.velocity.size - 1) / self.sample_rate

    @property
    def peak_acceleration(self) -> float:
        """The largest acceleration either way, in m/s^2."""
        return float(np.abs(self.acceleration).max())

    @property
    def peak_velocity(self) -> float:
        """The table's largest velocity either way, in m/s."""
        return float(np.abs(self.velocity).max())

    @property
    def relative_velocity(self) -> float:
        """The armature's largest velocity against the body, in m/s:
        `peak_velocity` with the body counted, what the velocity rating
        is judged against."""
        return float(self.rating.relative(self.peak_velocity))

    @property
    def velocity_change(self) -> float:
        """The span from the lowest velocity to the highest, in m/s.

        For a one-signed pulse with compensation either side it is the
        main pulse's change, 2AT/π for a half-sine, where the record's
        net change is zero by design.
        """
        return float(self.velocity.max() - self.velocity.min())

    @property
    def lowest(self) -> float:
        """The table's furthest travel below where it started, in m:
        zero or negative."""
        return -each_way(self.displacement)[0]

    @property
    def highest(self) -> float:
        """The table's furthest travel above where it started, in m:
        zero or positive."""
        return each_way(self.displacement)[1]

    @property
    def peak_to_peak(self) -> float:
        """The table's displacement, highest less lowest, in m."""
        return float(np.ptp(self.displacement))

    @property
    def counts_body(self) -> bool:
        """Whether the body's motion is counted: the body mass and the
        moving mass were both given."""
        return self.rating.body_share > 0.0

    @property
    def travel(self) -> float:
        """The armature's peak-to-peak travel relative to the body, in
        m: the table's peak to peak with the body counted. A number, not
        judged: the stops see the travel each way."""
        return float(self.rating.relative(self.peak_to_peak))

    @property
    def travel_down(self) -> float:
        """The relative travel below center, where the armature rests,
        in m, positive."""
        return each_way(self.rating.relative(self.displacement))[0]

    @property
    def travel_up(self) -> float:
        """The relative travel above center, where the armature rests,
        in m, positive."""
        return each_way(self.rating.relative(self.displacement))[1]

    @property
    def equivalent_duration(self) -> float:
        """The duration of the half-sine with this record's peak and
        velocity change, in s: πΔV / (2A).

        On a half-sine it is the pulse's own duration; on any other
        shape it is the half-sine the rule beside the integration is
        computed for.
        """
        peak = self.peak_acceleration
        return np.pi * self.velocity_change / (2.0 * peak) if peak else 0.0

    @property
    def rule(self) -> float:
        """The half-sine rule's peak to peak for this record's peak and
        `equivalent_duration`, in m: the first check beside the
        integrated `peak_to_peak` (`rule_stroke`)."""
        return rule_stroke(self.peak_acceleration, self.equivalent_duration)

    @property
    def force(self) -> float | None:
        """The force the coil must make, the moving mass times the peak
        acceleration, in N; None without a moving mass."""
        return self.rating.force_needed(self.peak_acceleration)

    def shares(self) -> dict[str, float | None]:
        """Each demand as a share of its rating, by `core.shaker`'s rule:
        'stroke' (the larger travel each way from center over half the
        stroke, so one is a stop), 'velocity' (counted against the body)
        and 'force'. None where the rating, or for the force the moving
        mass, was not given."""
        return {'stroke': self.rating.stroke_share(self.travel_down,
                                                   self.travel_up),
                'velocity': self.rating.velocity_share(self.peak_velocity),
                'force': self.rating.force_share(self.peak_acceleration)}

    @property
    def fits(self) -> bool | None:
        """Whether every rating given holds the pulse; None when no
        rating was given."""
        judged = [value for value in self.shares().values()
                  if value is not None]
        if not judged:
            return None
        return all(value <= 1.0 for value in judged)

    @property
    def fits_stroke(self) -> bool | None:
        """Whether the travel stays inside half the stroke each way from
        center, where the armature rests and the stops are half the
        stroke away; None without a stroke."""
        share = self.shares()['stroke']
        return None if share is None else share <= 1.0

    def _travel_name(self) -> str:
        """'one-sided travel', 'travel down' or 'travel up', with the
        body when it is counted."""
        down, up = self.travel_down, self.travel_up
        # one-sided when the other way is under a hundredth of it: the
        # symmetric arrangement's other way is zero to the sampling
        if min(down, up) < 0.01 * max(down, up):
            name = 'one-sided travel'
        else:
            name = f'travel {"down" if down >= up else "up"}'
        return name + ' with the body' * self.counts_body

    def to_the_stop(self, unit_system: UnitSystem | None = None
                    ) -> str | None:
        """The larger travel from center against the half stroke to its
        stop, in words, such as 'one-sided travel 1.46 in against
        1.25 in to the stop'; None without a stroke."""
        from ..units import DEFAULT_SYSTEM

        if self.rating.room is None:
            return None
        us = DEFAULT_SYSTEM if unit_system is None else unit_system
        larger = max(self.travel_down, self.travel_up)
        return (f'{self._travel_name()} {_amount(larger, "length", us)} '
                f'against {_amount(self.rating.room, "length", us)} to the '
                'stop')

    def describe(self, unit_system: UnitSystem | None = None) -> list[str]:
        """The reading in lines, the numbers in `unit_system`
        (`units.DEFAULT_SYSTEM` when omitted): the travel to the stop,
        the velocity and the force each against its rating, the
        displacement each way, and the peak to peak beside the half-sine
        rule."""
        from ..units import DEFAULT_SYSTEM

        us = DEFAULT_SYSTEM if unit_system is None else unit_system
        shares = self.shares()

        def verdict(line, key):
            if shares[key] is None:
                return line
            word = 'within' if shares[key] <= 1.0 else 'OVER'
            return f'{line}: {word}, {100.0 * shares[key]:.0f} %'

        def against(name, demand, rating, key, dimension):
            line = f'{name} {_amount(demand, dimension, us)}'
            if shares[key] is not None:
                line += f' against {_amount(rating, dimension, us)}'
            return verdict(line, key)

        rating = self.rating
        if rating.room is None:
            travel = (f'{sentence(self._travel_name())} '
                      f'{_amount(max(self.travel_down, self.travel_up), "length", us)}')
        else:
            travel = verdict(sentence(self.to_the_stop(us)), 'stroke')
        lines = [travel]
        if self.fits_stroke is False and self.travel <= rating.stroke:
            lines.append(f'Centered, the travel would fit: '
                         f'{_amount(self.travel, "length", us)} peak to peak '
                         f'against {_amount(rating.stroke, "length", us)}. '
                         'Centering the travel is the controller\'s '
                         'compensation, an asymmetric pre- and post-pulse')
        lines.append(against('Peak velocity'
                             + ' with the body' * self.counts_body,
                             self.relative_velocity, rating.velocity,
                             'velocity', 'velocity'))
        if self.force is not None:
            lines.append(against('Force', self.force, rating.force,
                                 'force', 'force'))
        lines.append(f'Each way from center: '
                     f'{_amount(self.travel_down, "length", us)} down, '
                     f'{_amount(self.travel_up, "length", us)} up')
        lines.append(f'Peak to peak {_amount(self.peak_to_peak, "length", us)} '
                     f'integrated, {_amount(self.rule, "length", us)} by the '
                     'half-sine rule')
        return lines


def read_stroke(history: TimeHistory, channel: int | str | None = None, *,
                stroke: float | None = None, velocity: float | None = None,
                force: float | None = None,
                moving_mass: float | None = None,
                body_mass: float | None = None) -> StrokeReading:
    """Integrate one acceleration channel twice, raw, and read it
    against the shaker's shock ratings.

        reading = read_stroke(compensated_half_sine(50.0, 0.011, 102400.0,
                                                    unit='g'),
                              stroke=0.0635, velocity=1.778,
                              moving_mass=45.4, body_mass=181.4)

    The record is integrated from rest at its first sample with nothing
    removed (`filters.integrate` with ``drift_corner=None``; the module
    docstring says why raw), over its whole length, so a record of one
    event should open before the event and close after it. Every rating
    is optional, and one not given, or given as zero, is reported
    without a verdict. The ratings are judged by `core.shaker`'s rule.

    Parameters
    ----------
    history : TimeHistory
        The shock record, or a nominal pulse from
        `compensated_half_sine`.
    channel : int or str, optional
        The record's index or its DOF; its first acceleration channel
        when omitted. Only a channel whose unit is declared counts.
    stroke : float, optional
        The shaker's shock stroke, peak to peak, in m: the armature's
        travel relative to the body.
    velocity : float, optional
        The shaker's shock velocity, in m/s.
    force : float, optional
        The shaker's shock force, in N.
    moving_mass : float, optional
        Everything that moves, in kg; with `force`, the force is checked.
    body_mass : float, optional
        The shaker body's mass, in kg; with `moving_mass`, the body's
        motion is counted into the travel and the velocity. The table
        alone when omitted.

    Returns
    -------
    StrokeReading
        The arrays, the peaks, the travel each way, the half-sine rule
        and the shares of each rating.

    Raises
    ------
    ValueError
        If the channel is not an acceleration channel of the record, the
        record has none, the channel holds a sample that is not finite,
        or a rating or mass is negative or not finite.
    """
    from .data import TimeHistory
    from .filters import integrate

    rating = ShakerRating(force=_rating(force), stroke=_rating(stroke),
                          velocity=_rating(velocity),
                          moving_mass=_rating(moving_mass),
                          body_mass=_rating(body_mass))
    index = acceleration_channel(history, channel)
    dof = str(history.response_dof[index])
    rows = finite(history.ordinate[index], dof)[None]
    one = TimeHistory(history.abscissa, rows, response_dof=[dof],
                      ordinate_dim=['acceleration'])
    # twice through the one trapezoid rule the package has, raw
    moving = integrate(one, None)
    travel = integrate(moving, None)
    return StrokeReading(
        dof=dof, sample_rate=history.sample_rate,
        start=float(history.abscissa[0]), acceleration=rows[0],
        velocity=np.asarray(moving.ordinate[0].real, dtype=float),
        displacement=np.asarray(travel.ordinate[0].real, dtype=float),
        rating=rating)


def _number(value: float) -> str:
    """A reading's number as the panel and the plot write it: three
    significant figures, and whole numbers with a thousands separator
    from a thousand up, where three figures would read as 5e+03."""
    if abs(value) >= 1000.0:
        return f'{value:,.0f}'
    return f'{value:.3g}'


def _amount(value: float, dimension: str, unit_system: UnitSystem) -> str:
    """`value`, in SI, as a number and a unit in `unit_system`, as the
    panel and the plot write it."""
    return (f'{_number(float(unit_system.from_si(value, dimension)))} '
            f'{unit_system.label_text(dimension)}')


def _rating(value: Any) -> float | None:
    """A rating as the reading keeps it: None for none, and zero is
    none too, as the panel's empty box is (a stroke of zero would put
    both stops at rest and fail every pulse for a number nobody
    meant). Anything else, a negative or a NaN included, goes to
    `ShakerRating`, which refuses it by name."""
    return None if value is None or float(value) == 0.0 else float(value)
