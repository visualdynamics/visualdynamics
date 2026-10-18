"""Whether a shaker can play a reference record, and what the high-pass
that makes it playable costs.

A sine profile states its peak, velocity and displacement at every
frequency, and a random profile states its rms and, through the
Gaussian assumption, its peaks. A record to be replicated states
nothing until it is worked: its largest peak may come once, in one
second of a minute, and its largest excursion at another second
entirely. So the fit check is made on the record itself, sample by
sample, before the first pass.

The force is the moving mass (everything the shaker accelerates: the
armature, the head expander or slip table, the fixture and the article)
times the record's largest absolute acceleration, read from the record
and not estimated from its rms. A record with impacts has a crest
factor well past the three sigma a Gaussian test would suggest, and the
force at that one peak is what goes against the rating. Each cutoff
changes the largest peak a little, so each case judges the force at
its own.

The velocity and the displacement come from integrating the record
twice, and they live at the bottom of its band: a component of
acceleration a at frequency f moves the table a/(2πf)², so the lowest
content decides the stroke almost alone. That is also where a record
is least trustworthy. A logger's offset of a thousandth of a g, held
for a minute, integrates to hundreds of inches, and below the sensor's
own low-frequency limit what the record holds is the chain's response,
not the vehicle's. So the low end is edited before the integration: a
zero-phase high-pass at a cutoff below the band of interest.

Each candidate cutoff is worked the same way. The record, with its
offset taken out, has its ends tapered: a half-cosine from zero over
two periods of the cutoff, 2/fc seconds, at each end (half the record
each when it is shorter than that). It is then high-passed (a
second-order Butterworth run forward and backward, `filters.filtered`,
so nothing moves in time), integrated raw to velocity, the velocity
high-passed again at the same cutoff, and integrated raw to
displacement. The velocity is filtered too because an integration from
rest cannot know the record's velocity at its first sample, and that
unknown constant walks the displacement off as a ramp of its own. The
integration is `filters.integrate` with ``drift_corner=None``, the
trapezoid rule with nothing removed: the cutoff here is the edit being
priced, so the integrator adds no high-pass of its own.

The taper is there because a record cut out of a longer one does not
start or end at rest. A tone at full amplitude from its first sample
integrates to a velocity with a constant in it, and the high-pass's own
start-up transient turns that constant into a displacement offset of
about v0/(2π fc), which at a low cutoff is many times the tone's true
motion. Eased in and out over two periods of the cutoff, the same tone
moves the table A/ω² to a fifth of a percent.

Each cutoff has a price, paid by the article: the share of the record's
mean square the filter removed, and the change in its largest peak.
Both are measured against the same tapered record with its offset
taken out, so neither the offset nor the taper is counted as content
the filter removed. The record is also integrated as logged, with no
taper and no cutoff, so that a logger offset shows as the drift it is,
and the offset alone is integrated beside it.

The travel, the velocity and the force are judged by the rule in
`core.shaker`, which the stroke readout uses too: the armature's motion
against the body is the table's times one plus the body's share, the
travel is read each way from the middle against half the stroke, and
the force is the moving mass times the peak. A record that dips into a
hole and climbs out is not symmetric about zero, and its span alone
can fit a stroke its one-sided excursion does not.

References
----------
The check is a standard one; these set out what it asks for and why
integration needs the low end edited first.

1. MIL-STD-810H (2019), *Environmental Engineering Considerations and
   Laboratory Tests*, Method 525.2, Time Waveform Replication. Before
   the first pass the reference is band-limited, filtered of the low
   content that would over-travel the exciter, its force estimated from
   the peak acceleration and the total moving mass, and integrated to
   velocity and displacement.
2. Boore, D. M., & Bommer, J. J. (2005). "Processing of strong-motion
   accelerograms: needs, options and consequences." *Soil Dynamics
   and Earthquake Engineering*. The drift a double integration invents
   from an offset, and what a high-pass does and does not cure.
3. Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal
   Processing*, 3rd ed. Prentice Hall. Zero-phase forward-backward
   filtering.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .shaker import (
    ShakerRating,
    acceleration_channel,
    each_way,
    finite,
    sentence,
)

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import TimeHistory

# Asked for on 2026-10-16, from a worked example: a sixty-second road
# record of 2 grms with a 14 g pothole and a 0.001 g logger offset,
# whose displacement as logged runs to 663 in, with the offset taken
# out still spans 48.6 in, and high-passed at 2, 3, 5 and 8 Hz spans
# 1.40, 0.71, 0.43 and 0.20 in for 1.25, 2.02, 3.33 and 5.10 percent of
# its mean square. `filters.integrate` with its 2 Hz default already
# gave the 2 Hz case to 0.005 in; what was missing was the ratings and
# the sweep. The raw path is the one the shock check takes
# (`drift_corner=None`), for the opposite reason: there the drift is the
# fault looked for, here the cutoff is the edit priced.
#
# The taper was added on review (2026-10-16): untapered, a 1 g, 10 Hz
# tone of a minute, true travel 2.48 mm, read 18.2 mm at a 1 Hz cutoff
# and 8.7 mm at 2 Hz, and the road record cut to 10-40 s failed a 2 in
# stroke at 2 and 3 Hz on a start-up transient. The ratings type and the
# travel rule moved to `core.shaker` the same day, shared with the
# stroke readout.

#: the candidate high-pass cutoffs, in Hz, when none are given: a few
#: steps across the bottom of a shaker's band, where the stroke is
#: decided and the price of the filter is still small
CUTOFFS = (2.0, 3.0, 5.0, 8.0)

#: the design order of the high-pass, run forward and backward. Second
#: order, the same as the drift high-pass `filters.integrate` applies,
#: so the default integration and the 2 Hz candidate are one filter
ORDER = 2

#: how long the half-cosine taper at each end of the record is, in
#: periods of the cutoff
TAPER_PERIODS = 2.0

#: the fewest samples a record can have and be filtered at zero phase:
#: one more than `scipy.signal.sosfiltfilt` pads each end with, three
#: times one more than twice the second-order sections
SHORTEST = 3 * (2 * ((ORDER + 1) // 2) + 1) + 1


@dataclass(frozen=True, eq=False)
class FitCase:
    """The record worked at one cutoff, or as logged.

    The arrays are the table's velocity and displacement, in m/s and m,
    from rest at the record's first sample. `cutoff` is the high-pass in
    Hz, None for the record integrated as logged. `mean_square_removed`
    and `peak_change` are the price of the cutoff, as shares of the
    tapered record with its offset out; both are zero as logged, where
    nothing was edited. The travel, the velocity and the force are
    judged by `core.shaker.ShakerRating`'s rule.
    """

    #: the high-pass cutoff in Hz, None for the record as logged
    cutoff: float | None
    #: the ratings the travel and velocity are judged against
    rating: ShakerRating
    #: the record's first instant, in seconds
    start: float
    #: samples per second
    sample_rate: float
    #: the table's velocity, m/s, from rest at the first sample
    velocity: np.ndarray
    #: the table's displacement, m, from rest at the first sample
    displacement: np.ndarray
    #: the share of the record's mean square the cutoff removed
    mean_square_removed: float
    #: the change in the record's largest peak, as a share of it
    peak_change: float
    #: the largest absolute acceleration the shaker is asked for at this
    #: cutoff, m/s**2: the record's own as logged, the edited one's here
    peak_acceleration: float

    @property
    def time(self) -> np.ndarray:
        """Each sample's time on the record's clock, in seconds."""
        return self.start + np.arange(self.displacement.size) \
            / self.sample_rate

    @property
    def travel(self) -> np.ndarray:
        """The armature's travel against the body, in m: the table's
        displacement times one plus the body's share."""
        return self.rating.relative(self.displacement)

    @property
    def travel_up(self) -> float:
        """The largest travel above the middle, in m; zero if it never
        rises."""
        return each_way(self.travel)[1]

    @property
    def travel_down(self) -> float:
        """The largest travel below the middle, in m, as a positive
        number; zero if it never falls."""
        return each_way(self.travel)[0]

    @property
    def peak_to_peak(self) -> float:
        """The travel from its lowest to its highest, in m."""
        return float(np.ptp(self.travel))

    @property
    def excursion_time(self) -> float:
        """When the largest travel either way happens, in seconds."""
        return float(self.time[int(np.argmax(np.abs(self.travel)))])

    @property
    def peak_velocity(self) -> float:
        """The table's largest velocity either way, in m/s."""
        return float(np.abs(self.velocity).max())

    @property
    def relative_velocity(self) -> float:
        """The armature's largest velocity against the body, in m/s:
        `peak_velocity` with the body's share counted, what the velocity
        rating is judged against."""
        return float(self.rating.relative(self.peak_velocity))

    @property
    def up_share(self) -> float:
        """The travel up as a share of the room, half the stroke."""
        return self.travel_up / self.rating.room

    @property
    def down_share(self) -> float:
        """The travel down as a share of the room, half the stroke."""
        return self.travel_down / self.rating.room

    @property
    def stroke_share(self) -> float:
        """The larger of `up_share` and `down_share`: one is at the
        stop."""
        return self.rating.stroke_share(self.travel_down, self.travel_up)

    @property
    def velocity_share(self) -> float:
        """The peak velocity, counted against the body, as a share of
        the rating."""
        return self.rating.velocity_share(self.peak_velocity)

    @property
    def force(self) -> float:
        """The moving mass times `peak_acceleration`, in N."""
        return self.rating.force_needed(self.peak_acceleration)

    @property
    def force_share(self) -> float:
        """The force as a share of the rating."""
        return self.rating.force_share(self.peak_acceleration)

    @property
    def fits_stroke(self) -> bool:
        """Whether the travel stays inside the room both ways."""
        return bool(self.stroke_share <= 1.0)

    @property
    def fits_velocity(self) -> bool:
        """Whether the peak velocity is inside the rating."""
        return bool(self.velocity_share <= 1.0)

    @property
    def fits_force(self) -> bool:
        """Whether the force at this cutoff's largest peak is inside the
        rating."""
        return bool(self.force_share <= 1.0)

    @property
    def fits(self) -> bool:
        """Whether the record fits the stroke, the velocity and the
        force at this cutoff."""
        return self.fits_stroke and self.fits_velocity and self.fits_force

    def first_past_stroke(self) -> float | None:
        """When the travel first leaves the room either way, in seconds,
        or None when it never does."""
        out = np.flatnonzero(np.abs(self.travel) > self.rating.room)
        return None if out.size == 0 else float(self.time[out[0]])

    def name(self) -> str:
        """'as logged', or the cutoff such as '3 Hz'."""
        return 'as logged' if self.cutoff is None else f'{self.cutoff:g} Hz'

    def shortfalls(self) -> list[str]:
        """Each rating this case is over, in words, such as 'the
        travel up is 140 % of the room'; empty when it fits."""
        short = []
        if not self.fits_stroke:
            if self.up_share == self.down_share:
                side = 'each way'
            else:
                side = 'up' if self.up_share > self.down_share else 'down'
            short.append(f'the travel {side} is '
                         f'{_of(self.stroke_share, "the room")}')
        if not self.fits_velocity:
            short.append('the velocity is '
                         f'{_of(self.velocity_share, "the rating")}')
        if not self.fits_force:
            short.append('the force is '
                         f'{_of(self.force_share, "the rating")}')
        return short

    def describe(self) -> str:
        """The verdict in words, as shares of the ratings, as the view
        and the panel state it."""
        if self.fits:
            return (f'{sentence(self.name())}: fits, using '
                    f'{_of(self.stroke_share, "the room")}, '
                    f'{_of(self.velocity_share, "the velocity rating")} '
                    f'and {_of(self.force_share, "the force rating")}.')
        return (f'{sentence(self.name())}: does not fit; '
                + ' and '.join(self.shortfalls()) + '.')


@dataclass(frozen=True, eq=False)
class ReplicationFit:
    """One channel of a reference record checked against a shaker.

    `as_logged` is the record integrated with no cutoff, and `cases`
    the record worked at each candidate cutoff, lowest first. The force,
    the offset and its drift are read from the record as logged; each
    case also judges the force at its own largest peak.
    """

    #: the channel's DOF, as the record names it
    dof: str
    #: the ratings the record is judged against
    rating: ShakerRating
    #: the record's largest absolute acceleration, m/s**2
    peak_acceleration: float
    #: when it happens, in seconds
    peak_time: float
    #: the record's rms with its offset out, m/s**2
    rms: float
    #: the record's mean, m/s**2: a logger's offset, if it has one
    offset: float
    #: the record's length from first sample to last, in seconds
    duration: float
    #: the record integrated with no cutoff
    as_logged: FitCase
    #: the record worked at each candidate cutoff, lowest first
    cases: tuple[FitCase, ...]

    @property
    def force(self) -> float:
        """The moving mass times the largest peak, in N."""
        return self.rating.force_needed(self.peak_acceleration)

    @property
    def force_share(self) -> float:
        """The force at the largest peak as a share of the rating."""
        return self.rating.force_share(self.peak_acceleration)

    @property
    def fits_force(self) -> bool:
        """Whether the force at the largest peak is inside the rating."""
        return bool(self.force_share <= 1.0)

    @property
    def crest_factor(self) -> float:
        """The largest peak over the rms: well past three for a record
        with impacts, which is why the force is read from the peak."""
        return self.peak_acceleration / self.rms if self.rms > 0.0 else 0.0

    @property
    def offset_velocity(self) -> float:
        """What the offset alone integrates to by the end, in m/s."""
        return self.offset * self.duration

    @property
    def offset_displacement(self) -> float:
        """What the offset alone integrates to by the end, in m: half
        the offset times the square of the record's length."""
        return 0.5 * self.offset * self.duration ** 2

    def case(self, cutoff: float | None) -> FitCase:
        """The case at `cutoff` in Hz, or the record as logged for None.

        Raises
        ------
        KeyError
            If `cutoff` is not one of the candidates.
        """
        if cutoff is None:
            return self.as_logged
        for case in self.cases:
            if np.isclose(case.cutoff, cutoff):
                return case
        raise KeyError(f'{cutoff:g} Hz is not one of the candidate '
                       'cutoffs ('
                       + ', '.join(f'{c.cutoff:g}' for c in self.cases)
                       + ' Hz)')

    @property
    def lowest_fitting(self) -> FitCase | None:
        """The lowest cutoff at which the record fits the stroke, the
        velocity and the force, the one that takes least out of the
        record; None when none does."""
        return next((case for case in self.cases if case.fits), None)

    def describe(self) -> str:
        """The whole check in a few sentences: the force, the drift as
        logged, and the lowest cutoff that fits."""
        force = ('The force at the largest peak is '
                 f'{_of(self.force_share, "the rating")}'
                 + ('.' if self.fits_force else ', over it.'))
        logged = self.as_logged
        if logged.fits:
            drift = 'As logged, the record fits without a cutoff.'
        elif not logged.fits_stroke:
            drift = ('As logged, the travel runs to '
                     f'{_of(logged.stroke_share, "the room")} and leaves '
                     f'it {logged.first_past_stroke():.3g} s in.')
        else:
            drift = ('As logged, the travel stays inside the room and '
                     + ' and '.join(logged.shortfalls()) + '.')
        fitting = self.lowest_fitting
        if fitting is None:
            fit = 'No candidate cutoff fits.'
        else:
            fit = (f'The lowest cutoff that fits is {fitting.name()}, '
                   f'removing {_percent(fitting.mean_square_removed)} of '
                   'the mean square.')
        return f'{force} {drift} {fit}'


def _percent(share: float) -> str:
    """A share as a percentage, to two places below ten and none above;
    ten times or more is said in times, where a percentage of thousands
    reads as a misprint."""
    value = 100.0 * share
    if abs(share) >= 10.0:
        return f'{share:.0f} times'
    if abs(value) < 9.995:
        return f'{value:.2f} %'
    return f'{value:.0f} %'


def _of(share: float, what: str) -> str:
    """'71 % of the room', or '664 times the room'."""
    amount = _percent(share)
    return f'{amount} {what}' if amount.endswith('times') \
        else f'{amount} of {what}'


def replication_fit(history: TimeHistory, rating: ShakerRating, *,
                    cutoffs: Sequence[float] = CUTOFFS,
                    channel: str | int | None = None) -> ReplicationFit:
    """Check one channel of a reference record against a shaker.

    The force at the record's largest peak, the record integrated as
    logged, and the record worked at each candidate cutoff: its ends
    tapered over two periods of the cutoff, high-passed at zero phase,
    integrated raw to velocity, high-passed again, integrated raw to
    displacement (the module docstring says why each step). Each case
    carries the travel each way with the body counted, the peak
    velocity, the force at its own largest peak, the share of the mean
    square the cutoff removed, the change in the largest peak, and its
    verdict.

    Parameters
    ----------
    history : TimeHistory
        The reference record.
    rating : ShakerRating
        The shaker's ratings and what it carries; the force, stroke and
        velocity ratings and the moving mass are all needed.
    cutoffs : sequence of float, default `CUTOFFS`
        The candidate high-pass cutoffs, in Hz; worked lowest first.
    channel : str or int, optional
        The acceleration channel, by DOF or record index; the record's
        first acceleration channel by default. Only a channel whose unit
        is declared counts (`core.shaker.acceleration_channel`).

    Returns
    -------
    ReplicationFit
        The check, with a `FitCase` as logged and one per cutoff.

    Raises
    ------
    ValueError
        If the record holds no acceleration channel, `channel` is not
        one, the channel holds a sample that is not finite, the record
        is too short to filter, a rating is missing, no cutoff is given,
        or a cutoff is not between zero and the Nyquist frequency.
    """
    from .data import TimeHistory
    from .filters import Filtering, filtered, integrate

    rating.require('force', 'stroke', 'velocity', 'moving_mass',
                   reading='the replication fit')
    row = acceleration_channel(history, channel)
    rate = history.sample_rate
    cutoffs = sorted({float(cutoff) for cutoff in cutoffs})
    if not cutoffs:
        raise ValueError('give at least one candidate cutoff')
    for cutoff in cutoffs:
        if not 0.0 < cutoff < rate / 2.0:
            raise ValueError(f'the cutoff ({cutoff:g} Hz) must sit between '
                             'zero and the Nyquist frequency '
                             f'({rate / 2.0:g} Hz)')
    dof = str(history.response_dof[row])
    values = finite(history.ordinate[row], dof)
    if values.size < SHORTEST:
        raise ValueError(f'a record of {values.size} samples is too short '
                         'to high-pass at zero phase; it needs at least '
                         f'{SHORTEST}')
    centered = values - values.mean()
    start = float(history.abscissa[0])

    def alone(rows):
        # one channel on its own, so a force or a second accelerometer
        # beside it is neither filtered nor integrated for nothing
        return TimeHistory(history.abscissa, rows[None],
                           response_dof=[dof], ordinate_dim=['acceleration'])

    def worked(source, cutoff, reference, edited, peak):
        if cutoff is None:
            velocity = integrate(source, None)
        else:
            cut = Filtering(low=cutoff, order=ORDER)
            velocity = filtered(integrate(source, None), cut)
        displacement = integrate(velocity, None)
        mean_square = float(np.mean(reference ** 2))
        largest = float(np.abs(reference).max())
        return FitCase(
            cutoff=cutoff, rating=rating, start=start, sample_rate=rate,
            velocity=np.real(velocity.ordinate[0]).copy(),
            displacement=np.real(displacement.ordinate[0]).copy(),
            mean_square_removed=(
                _share(1.0 - np.mean(edited ** 2) / mean_square)
                if mean_square > 0.0 else 0.0),
            peak_change=(_share(np.abs(edited).max() / largest - 1.0)
                         if largest > 0.0 else 0.0),
            peak_acceleration=peak)

    as_logged = worked(alone(values), None, centered, centered,
                       float(np.abs(values).max()))
    cases = []
    for cutoff in cutoffs:
        tapered = centered * taper(values.size, TAPER_PERIODS / cutoff * rate)
        source = filtered(alone(tapered), Filtering(low=cutoff, order=ORDER))
        edited = np.real(source.ordinate[0])
        cases.append(worked(source, cutoff, tapered, edited,
                            float(np.abs(edited).max())))
    peak = int(np.argmax(np.abs(values)))
    return ReplicationFit(
        dof=dof, rating=rating,
        peak_acceleration=float(abs(values[peak])),
        peak_time=float(history.abscissa[peak]),
        rms=float(np.sqrt(np.mean(centered ** 2))),
        offset=float(values.mean()),
        duration=(values.size - 1) / rate,
        as_logged=as_logged, cases=tuple(cases))


def taper(samples: int, length: float) -> np.ndarray:
    """A window of `samples` that rises from zero over `length` samples
    by a half-cosine at each end and is one between; each end is at
    most half the window, where the two meet as a Hann window."""
    ramp = int(min(round(length), samples // 2))
    window = np.ones(samples)
    if ramp > 0:
        rise = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / ramp)
        window[:ramp] = rise
        window[samples - ramp:] = rise[::-1]
    return window


def _share(value: float) -> float:
    """A share, with the rounding of a subtraction near zero kept
    out of the sign."""
    return 0.0 if abs(value) < 1e-12 else float(value)
