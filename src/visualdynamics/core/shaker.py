"""What a shaker can do, and the one rule its travel, velocity and
force are judged by.

Two readings ask whether a shaker can play something: the replication
fit (`core.replication_fit`), a long reference record high-passed at a
few candidate cutoffs, and the stroke readout (`core.stroke`), one
shock pulse integrated raw. They read different records in different
ways, and they judge what they read by the same rule, kept here:

- The force is the moving mass (everything the shaker accelerates: the
  armature, the head expander or slip table, the fixture and the
  article) times the largest absolute acceleration.
- The shaker body moves opposite the armature. A body free on its
  isolation mounts moves by the ratio of the moving mass to its own
  mass, so the armature's motion against the body, which is what the
  ratings limit, is the table's times one plus that ratio
  (`ShakerRating.body_share`). The travel and the velocity both count
  it. The body moves most at low frequency, which is where the
  displacement of a replication record and of a compensated pulse
  lives.
- The armature rests at the middle of its stroke, held there by its
  suspension, so the stops are half the stroke away on each side
  (`ShakerRating.room`). The travel is read each way from the middle
  (`each_way`), and the stroke share is the larger of the two over half
  the stroke: one hundred percent puts the armature on a stop. A
  travel that runs one way only can fit the stroke peak to peak and
  still reach a stop, so the span is reported as a number and never
  judged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import TimeHistory

# Made on 2026-10-16 when the replication fit and the stroke readout
# arrived together, each with its own ratings type and its own body,
# room and share rules; the two disagreed on what a "stroke share" was
# (peak to peak over the stroke in one, the larger one-way travel over
# half of it in the other), so one of them could say "within, 58 %" of
# a pulse that ran 117 % of the way to a stop.

#: the ratings, in the order a message lists them
FIELDS = ('force', 'stroke', 'velocity', 'moving_mass', 'body_mass')


@dataclass(frozen=True, kw_only=True)
class ShakerRating:
    """What a shaker can do, and what it carries, in SI.

    `force` is the peak force rating in N, `stroke` the travel between
    the stops in m peak to peak, `velocity` the peak velocity rating in
    m/s, and `moving_mass` everything the shaker accelerates in kg: the
    armature, the head expander or slip table, the fixture and the
    article. `body_mass`, in kg, counts the body's motion opposite the
    armature when it is given with the moving mass.

    Each is optional, and a rating left out is not judged: a reading
    that needs one says so. One that is given must be a finite number
    above zero.
    """

    force: float | None = None
    stroke: float | None = None
    velocity: float | None = None
    moving_mass: float | None = None
    body_mass: float | None = None

    def __post_init__(self) -> None:
        for name in FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            value = float(value)
            if not (np.isfinite(value) and value > 0.0):
                hint = ('; leave it out to count no body motion'
                        if name == 'body_mass' else '')
                bound = 'above zero' if np.isfinite(value) \
                    else 'above zero and finite'
                raise ValueError(f'the {name.replace("_", " ")} has to be '
                                 f'{bound}, not {value:g}{hint}')
            object.__setattr__(self, name, value)

    def require(self, *names: str, reading: str) -> None:
        """Refuse a rating that `reading` judges and was not given.

        Raises
        ------
        ValueError
            Naming every one of `names` that is None.
        """
        missing = [name.replace('_', ' ') for name in names
                   if getattr(self, name) is None]
        if missing:
            listed = ' and the '.join((', the '.join(missing[:-1]),
                                       missing[-1])) if missing[:-1] \
                else missing[0]
            raise ValueError(f'{reading} needs the {listed} as well')

    @property
    def body_share(self) -> float:
        """The moving mass over the body's: the share of the table's
        motion the body adds to the armature's motion against it. Zero
        when the body mass or the moving mass is not given."""
        if self.body_mass is None or self.moving_mass is None:
            return 0.0
        return self.moving_mass / self.body_mass

    @property
    def room(self) -> float | None:
        """The travel available each way from the middle of the stroke,
        where the armature rests: half the stroke, in m; None without
        a stroke."""
        return None if self.stroke is None else self.stroke / 2.0

    def relative(self, motion: Any) -> Any:
        """The armature's motion against the body, from the table's: a
        displacement, a velocity, or an array of either, times one plus
        `body_share`."""
        return motion * (1.0 + self.body_share)

    def force_needed(self, peak_acceleration: float) -> float | None:
        """The moving mass times `peak_acceleration`, in N; None
        without a moving mass."""
        if self.moving_mass is None:
            return None
        return self.moving_mass * float(peak_acceleration)

    def stroke_share(self, travel_down: float,
                     travel_up: float) -> float | None:
        """The larger travel each way from the middle over the room, half
        the stroke; None without a stroke. At one the armature reaches a
        stop."""
        if self.room is None:
            return None
        return max(travel_down, travel_up) / self.room

    def velocity_share(self, table_velocity: float) -> float | None:
        """The table's peak velocity, counted against the body
        (`relative`), over the velocity rating; None without one."""
        if self.velocity is None:
            return None
        return self.relative(float(table_velocity)) / self.velocity

    def force_share(self, peak_acceleration: float) -> float | None:
        """The force at `peak_acceleration` over the force rating; None
        without the rating or the moving mass."""
        force = self.force_needed(peak_acceleration)
        if force is None or self.force is None:
            return None
        return force / self.force


def each_way(travel: Any) -> tuple[float, float]:
    """The largest travel below the middle and above it, in that order,
    both as positive numbers in the travel's unit; zero for a side it
    never reaches (never minus zero)."""
    travel = np.asarray(travel, dtype=float)
    # zero first: max keeps the first of equals, so a side at minus
    # zero reads as zero
    return max(0.0, -float(travel.min())), max(0.0, float(travel.max()))


def acceleration_channel(history: TimeHistory,
                         channel: str | int | None = None) -> int:
    """The record index of the acceleration channel asked for, or of
    the record's first one.

    A channel counts only when its unit is declared. One the source
    only named acceleration holds the file's raw numbers, and a travel
    integrated from them in SI would be read against the stroke in a
    scale nobody chose.

    Raises
    ------
    ValueError
        If the record holds no declared acceleration channel, or
        `channel` is not one.
    """
    accelerations = [i for i in range(history.num_records)
                     if history.ordinate_dim[i] == 'acceleration']
    if not accelerations:
        hinted = any(history.known_dim(i) == 'acceleration'
                     for i in range(history.num_records))
        raise ValueError(
            'the check integrates acceleration, and this record holds '
            + ('acceleration only in an undeclared unit; declare its '
               'unit first' if hinted else 'none'))
    if channel is None:
        return accelerations[0]
    if isinstance(channel, str):
        named = [i for i in accelerations
                 if str(history.response_dof[i]) == channel]
        if not named:
            raise ValueError(f'{channel} is not an acceleration channel of '
                             'this record')
        return named[0]
    index = int(channel)
    if index not in accelerations:
        raise ValueError(f'record {index} is not an acceleration channel '
                         'of this record')
    return index


def finite(values: Any, dof: str) -> np.ndarray:
    """`values` as real floats, refused when any is not finite: one NaN
    integrates to a travel of NaN from that sample on, and a NaN is
    neither inside a rating nor over it.

    Raises
    ------
    ValueError
        Naming the channel and where its first bad sample is.
    """
    values = np.real(np.asarray(values, dtype=complex)).astype(float)
    bad = np.flatnonzero(~np.isfinite(values))
    if bad.size:
        raise ValueError(f'{dof} holds {bad.size} '
                         + ('sample that is' if bad.size == 1
                            else 'samples that are')
                         + ' not finite, '
                         f'the first at index {bad[0]}; a travel '
                         'integrated through one is not a number')
    return values
