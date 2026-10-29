"""A random test's requirement, and its result criterion by criterion.

A test is run to a requirement the plan copied from its governing
document: the control band, each control channel's band when several
are averaged, the overall rms, the duration, the degrees of freedom and
the line spacing the control must have, the cross-axis limits, and the
rule a miss is decided by. "Passed" without those is not a result. A
`TestRequirement` holds them as the plan states them, each number with
where it came from (the laboratory, the customer, or derived from
another), and `evaluate` reads a run against it, one row per criterion,
each worded met, not met or not judged.

- **The control band.** Under a single control channel, that channel
  against the band; under a multi-point strategy, the composite the
  controller held (`core.multi_point`) against the composite band, and
  each control channel against its own wider band. A band is a pair of
  dB below and above the reference over a frequency range, so a band
  that steps (plus or minus 3 dB below 500 Hz and plus or minus 6 dB
  above) is two bands, each its own row. Every line is judged by
  `compliance.judge`, the comparison every other random reading makes,
  on a one-record specification holding the band as its abort limits.
- **The decision rule.** Either any line outside its band is a miss, or
  N adjacent lines are combined, their powers added and judged against
  the band's power over the same stretch at the group's degrees of
  freedom (`core.chance.compare_chance`), so the scatter of the estimate
  shrinks and a real excursion across neighboring lines remains. Some
  test documents allow the second; which one applies is the plan's
  choice, and the requirement names it.
- **Each miss** is given by its band edges in hertz, how far past the
  band it went at worst in dB, and how many lines (or groups) it holds,
  so the row can be checked against the figure.
- **The overall rms** of the composite (or the one channel) against the
  reference's over the band, in percent (`compliance.compare`).
- **The duration at level** against its nominal with a tolerance that
  may differ either side (minus 0, plus 10 percent in some documents).
  It is the controller's counter, typed for the run, since a PSD does
  not hold how long the run was at level.
- **The degrees of freedom** carried by the control channels' PSDs, the
  fewest of them, and the **line spacing**, each against the
  requirement.
- **Cross-axis** (`cross_axis_rows`): each cross-axis channel named,
  with the in-axis control channel it is read against, as a percentage
  of that channel's ASD line by line over each limit's band, at the
  same degrees of freedom, the worst percentage and its frequency
  against the limit.
- **Limit channels** (`limit_rows`): for each response channel held
  under a limit profile, the band and depth over which it held its
  limit while the control fell below the reference, the as-run notch,
  each notch against the notches the plan names. A notch the plan does
  not name is a departure from it, and its row is not met.

**The measurement's own uncertainty** is held with the requirement too,
since the tolerance it is read against is the requirement's: each
control channel's contributions (the accelerometer's calibration from
its certificate, the conditioner and converter, transverse sensitivity
times the cross-axis share, the mount and the chain's response), each
an expanded uncertainty in percent on the acceleration with its
coverage factor. `uncertainty_budget` divides each by its own coverage
factor, combines them as the root of the sum of their squares, expands
the result at `COVERAGE`, and states it in percent on the acceleration,
in percent on the ASD and in dB, against one third of the tolerance
(the tightest band edge in dB, and the overall rms tolerance in
percent), the share some test documents allow the measurement. It is
not a criterion of the run, and a channel with nothing entered reads
not judged, never met.

A row that judged nothing says so and why, and is never met: a band the
data do not reach, a channel with no finite line, a measurement that
does not carry its degrees of freedom. A row with some lines not judged
(a sample that is not a number) and none outside is not judged either,
and its words say how many lines were judged. The results are met only
when every row is.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .multi_point import STRATEGIES as COMPOSITE_STRATEGIES

if TYPE_CHECKING:                                    # pragma: no cover
    from .data import Psd, Specification

#: the control strategies: one channel, or one of the multi-point
#: composites
STRATEGIES = ('single', *COMPOSITE_STRATEGIES)

#: the decision rules: any line outside its band is a miss, or groups of
#: adjacent lines combined are judged
RULES = ('any_line', 'combined')

#: where a number of the requirement came from
ORIGINS = ('laboratory', 'customer', 'derived')

#: the numbers a source can be stated for, by the requirement's field
NUMBERS = ('composite_bands', 'individual_bands', 'grms_percent', 'duration',
           'min_dof', 'max_line_spacing', 'cross_axis', 'rule', 'uncertainty',
           'limit_channels')

#: a row's result, in the words the table prints
MET = 'met'
NOT_MET = 'not met'
NOT_JUDGED = 'not judged'

#: the sentence said of every number the customer supplied
CUSTOMER = ('supplied by the customer; the laboratory has not verified its '
            'derivation')

#: the most excursions a row names; the rest are counted
MOST_EXCURSIONS = 3

#: the coverage factor a channel's combined uncertainty is expanded
#: with: about 95 percent for a normal distribution, and the factor
#: calibration certificates commonly state
COVERAGE = 2.0

#: the share of the tolerance the measurement's uncertainty may take
SHARE = 1.0 / 3.0

#: how near its profile, in dB, a limit channel counts as holding its
#: limit, and how far under the reference the control must fall for a
#: line to count as notched: about twice the scatter of an estimate at
#: 120 degrees of freedom, so a line of scatter alone is rarely read as
#: either
HELD_DB = 1.0

#: how far apart two cells may be and still touch, as a fraction of the
#: frequency (the edges of neighboring bins are worked out separately)
_TOUCHING = 1e-9


def _number(value: Any, what: str) -> float:
    """`value` as a float, refused when it is not a finite number."""
    if isinstance(value, bool):
        raise TypeError(f'{what} is a number, not {value!r}')
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{what} is a number, not {value!r}') from None
    if math.isnan(out):
        raise ValueError(f'{what} is a number, not NaN')
    return out


def _hz(value: float) -> str:
    return f'{value:.6g}'


def _range_text(low: float | None, high: float | None) -> str:
    if low is None and high is None:
        return 'over the whole band'
    if low is None:
        return f'up to {_hz(high)} Hz'
    if high is None:
        return f'from {_hz(low)} Hz'
    return f'from {_hz(low)} to {_hz(high)} Hz'


def _edge(value: Any, what: str) -> float | None:
    if value is None:
        return None
    out = _number(value, what)
    if not 0.0 < out < math.inf:
        raise ValueError(f'{what} is a frequency above 0 Hz, not {value!r}')
    return out


def _reach(low: float | None, high: float | None, what: str) -> None:
    if low is not None and high is not None and not low < high:
        raise ValueError(f'{what} runs from {_hz(low)} to {_hz(high)} Hz; '
                         'its low edge is below its high edge')


def _overlap(ranges: Sequence[Any], what: str) -> None:
    """Refuse two ranges of one list that overlap: a line would then be
    judged against two bands."""
    spans = sorted((-math.inf if r.low is None else r.low,
                    math.inf if r.high is None else r.high) for r in ranges)
    for (_a, high), (low, _b) in itertools.pairwise(spans):
        if low < high:
            raise ValueError(f'two {what} overlap; each line is judged '
                             'against one band')


@dataclass(frozen=True)
class Band:
    """A band about the reference over a frequency range: `below_db` and
    `above_db` (ten log ten, on a density) from `low` to `high` Hz.

    `low` or `high` None reaches the specification's own end. An edge
    may be infinite (no limit on that side), but not both.
    """

    low: float | None
    high: float | None
    below_db: float
    above_db: float

    def __post_init__(self) -> None:
        low = _edge(self.low, "a band's low edge")
        high = _edge(self.high, "a band's high edge")
        _reach(low, high, 'a band')
        below = _number(self.below_db, "a band's lower edge in dB")
        above = _number(self.above_db, "a band's upper edge in dB")
        if below > 0.0 or above < 0.0:
            raise ValueError(f'a band lies about the reference: its lower '
                             f'edge at or below 0 dB and its upper edge at '
                             f'or above; {below:+g}/{above:+g} dB is not')
        if math.isinf(below) and math.isinf(above):
            raise ValueError('a band has no edge on either side')
        object.__setattr__(self, 'low', low)
        object.__setattr__(self, 'high', high)
        object.__setattr__(self, 'below_db', below)
        object.__setattr__(self, 'above_db', above)

    @property
    def db_text(self) -> str:
        """'−3/+3 dB', an open edge written 'down' or 'up'."""
        low = 'down' if math.isinf(self.below_db) else f'{self.below_db:+g}'
        high = 'up' if math.isinf(self.above_db) else f'{self.above_db:+g}'
        return f'{low}/{high} dB'.replace('-', '−')

    @property
    def text(self) -> str:
        """'−3/+3 dB from 20 to 500 Hz'."""
        return f'{self.db_text} {_range_text(self.low, self.high)}'

    def as_record(self) -> list[Any]:
        return [self.low, self.high, _json_db(self.below_db),
                _json_db(self.above_db)]


@dataclass(frozen=True)
class CrossAxisLimit:
    """A cross-axis limit: the cross-axis response at most `percent` of
    the in-axis response from `low` to `high` Hz."""

    low: float | None
    high: float | None
    percent: float

    def __post_init__(self) -> None:
        low = _edge(self.low, "a cross-axis limit's low edge")
        high = _edge(self.high, "a cross-axis limit's high edge")
        _reach(low, high, 'a cross-axis limit')
        percent = _number(self.percent, 'a cross-axis limit in percent')
        if not 0.0 < percent < math.inf:
            raise ValueError('a cross-axis limit is a percentage above 0, '
                             f'not {self.percent!r}')
        object.__setattr__(self, 'low', low)
        object.__setattr__(self, 'high', high)
        object.__setattr__(self, 'percent', percent)

    @property
    def text(self) -> str:
        return f'{self.percent:g} % of in-axis {_range_text(self.low, self.high)}'

    def as_record(self) -> list[Any]:
        return [self.low, self.high, self.percent]


@dataclass(frozen=True)
class Source:
    """Where a number of the requirement came from: `origin` (one of
    `ORIGINS`) and its `owner`, the person or organization answerable
    for it."""

    origin: str
    owner: str = ''

    def __post_init__(self) -> None:
        if self.origin not in ORIGINS:
            raise ValueError(f'a source is one of {", ".join(ORIGINS)}, not '
                             f'{self.origin!r}')
        if not isinstance(self.owner, str):
            raise TypeError(f"a source's owner is text, not {self.owner!r}")

    @property
    def text(self) -> str:
        said = {'laboratory': 'from the laboratory',
                'customer': 'supplied by the customer',
                'derived': 'derived'}[self.origin]
        if self.owner:
            said += f' ({self.owner})'
        if self.origin == 'customer':
            said += CUSTOMER.removeprefix('supplied by the customer')
        return said


@dataclass(frozen=True)
class Contribution:
    """One contribution to a control channel's measurement uncertainty:
    its `name`, its expanded uncertainty `percent` on the acceleration
    at coverage factor `coverage`, and where the number came from
    (`source`: a calibration certificate, a data sheet).

    A contribution is a percentage above 0 and under 100: a chain with
    no uncertainty, or one as large as the signal, is not a budget.
    """

    name: str
    percent: float
    coverage: float = COVERAGE
    source: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('an uncertainty contribution is named, not '
                             f'{self.name!r}')
        if not isinstance(self.source, str):
            raise TypeError("an uncertainty contribution's source is text, "
                            f'not {self.source!r}')
        what = f'the uncertainty contribution {self.name.strip()!r}'
        percent = _number(self.percent, f'{what} in percent')
        if not 0.0 < percent < 100.0:
            raise ValueError(f'{what} is a percentage above 0 and under 100, '
                             f'not {self.percent!r}')
        coverage = _number(self.coverage, f"{what}'s coverage factor")
        if not 0.0 < coverage < math.inf:
            raise ValueError(f"{what}'s coverage factor is a number above 0, "
                             f'not {self.coverage!r}')
        object.__setattr__(self, 'name', self.name.strip())
        object.__setattr__(self, 'percent', percent)
        object.__setattr__(self, 'coverage', coverage)

    @property
    def standard(self) -> float:
        """The standard uncertainty, percent: `percent` over `coverage`."""
        return self.percent / self.coverage

    @property
    def text(self) -> str:
        """'calibration 2 % (k = 2, certificate C-1)'."""
        said = f'k = {self.coverage:g}'
        if self.source:
            said += f', {self.source}'
        return f'{self.name} {self.percent:g} % ({said})'

    def as_record(self) -> list[Any]:
        return [self.name, self.percent, self.coverage, self.source]


def _contributions(value: Any, channels: tuple[str, ...]
                   ) -> dict[str, tuple[Contribution, ...]]:
    """The uncertainty contributions by control channel, refused when a
    channel is not one the requirement names or a contribution is named
    twice for one channel."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError('the uncertainty is a mapping of control channel to '
                        'its contributions, each (name, percent, coverage, '
                        'source)')
    out: dict[str, tuple[Contribution, ...]] = {}
    for channel, entered in value.items():
        channel = str(channel)
        if channels and channel not in channels:
            raise ValueError(f'uncertainty is entered for {channel}, which is '
                             'not a control channel of the requirement '
                             f'({", ".join(channels)})')
        if isinstance(entered, (str, Contribution)) or not isinstance(
                entered, Sequence):
            raise TypeError(f'the uncertainty of {channel} is a list of '
                            'contributions, each (name, percent, coverage, '
                            'source)')
        held = []
        for item in entered:
            if not isinstance(item, Contribution):
                if isinstance(item, str) or not isinstance(item, Sequence) \
                        or not 2 <= len(item) <= 4:
                    raise ValueError(f'each uncertainty contribution of '
                                     f'{channel} is (name, percent, coverage, '
                                     f'source), not {item!r}')
                item = Contribution(*item)
            held.append(item)
        names = [item.name for item in held]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(f'the uncertainty of {channel} names '
                             f'{", ".join(map(repr, twice))} twice; each '
                             'contribution counts once')
        out[channel] = tuple(held)
    return out


def _json_db(value: float) -> float | str:
    # JSON holds no infinity: an open edge is stored as its word
    if math.isinf(value):
        return 'down' if value < 0 else 'up'
    return value


def _db_from(value: Any) -> Any:
    if value == 'down':
        return -math.inf
    if value == 'up':
        return math.inf
    return value


def _bands(value: Any, what: str) -> tuple[Band, ...]:
    if value is None:
        return ()
    if isinstance(value, (Band, str)) or not isinstance(value, Sequence):
        raise TypeError(f'{what} is a list of bands, each (low, high, '
                         'below_db, above_db)')
    out = []
    for band in value:
        if isinstance(band, Band):
            out.append(band)
            continue
        try:
            low, high, below, above = band
        except (TypeError, ValueError):
            raise ValueError(f'each of {what} is (low, high, below_db, '
                             f'above_db), not {band!r}') from None
        out.append(Band(low, high, _db_from(below), _db_from(above)))
    _overlap(out, what)
    return tuple(out)


def _limits(value: Any) -> tuple[CrossAxisLimit, ...]:
    if value is None:
        return ()
    if isinstance(value, (CrossAxisLimit, str)) or not isinstance(value,
                                                                  Sequence):
        raise TypeError('the cross-axis limits are a list, each (low, high, '
                         'percent)')
    out = []
    for limit in value:
        if isinstance(limit, CrossAxisLimit):
            out.append(limit)
            continue
        try:
            low, high, percent = limit
        except (TypeError, ValueError):
            raise ValueError('each cross-axis limit is (low, high, percent), '
                             f'not {limit!r}') from None
        out.append(CrossAxisLimit(low, high, percent))
    _overlap(out, 'cross-axis limits')
    return tuple(out)


@dataclass(frozen=True)
class LimitChannel:
    """A limit channel: the response at `channel` (its response DOF)
    held under its limit profile, an autospectrum at that DOF in the
    object named `profile` ('' for the specification's own record
    there), and the notches the plan names for it (`notches`, each
    (low, high) Hz)."""

    channel: str
    profile: str = ''
    notches: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.channel, str) or not self.channel.strip():
            raise ValueError('a limit channel is named by its response DOF, '
                             f'not {self.channel!r}')
        if not isinstance(self.profile, str):
            raise TypeError("a limit channel's profile is named by text, "
                            f'not {self.profile!r}')
        what = f'a planned notch of {self.channel.strip()}'
        if isinstance(self.notches, str) or not isinstance(self.notches,
                                                           Sequence):
            raise TypeError(f'the planned notches of {self.channel.strip()} '
                            'are a list, each (low, high)')
        held = []
        for notch in self.notches:
            try:
                low, high = notch
            except (TypeError, ValueError):
                raise ValueError(f'{what} is (low, high), not '
                                 f'{notch!r}') from None
            low = _edge(low, f"{what}'s low edge")
            high = _edge(high, f"{what}'s high edge")
            if low is None or high is None:
                raise ValueError(f'{what} states both its edges')
            _reach(low, high, what)
            held.append(_Span(low, high))
        _overlap(held, f'planned notches of {self.channel.strip()}')
        object.__setattr__(self, 'channel', self.channel.strip())
        object.__setattr__(self, 'notches',
                           tuple((span.low, span.high) for span in held))

    def planned(self, notch: Excursion) -> bool:
        """Whether `notch` lies inside a notch the plan names."""
        return any(low * (1.0 - _TOUCHING) <= notch.low
                   and notch.high <= high * (1.0 + _TOUCHING)
                   for low, high in self.notches)

    @property
    def text(self) -> str:
        """What the row requires, in words."""
        profile = (f'its limit profile in {self.profile}' if self.profile
                   else "its limit profile in the specification")
        if self.notches:
            planned = ('notches planned ' + ', '.join(
                f'{_hz(low)} to {_hz(high)} Hz' for low, high in self.notches))
        else:
            planned = 'no notch planned'
        return f'held at or under {profile}; {planned}'

    def as_record(self) -> list[Any]:
        return [self.channel, self.profile,
                [list(notch) for notch in self.notches]]


@dataclass(frozen=True)
class _Span:
    low: float
    high: float


def _cross_channels(value: Any, channels: tuple[str, ...]
                    ) -> dict[str, str]:
    """The cross-axis channels, each with its in-axis control channel."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError('the cross-axis channels are a mapping of each '
                        'cross-axis DOF to its in-axis control channel')
    out = {}
    for cross, in_axis in value.items():
        cross, in_axis = str(cross).strip(), str(in_axis).strip()
        if not cross or not in_axis:
            raise ValueError('a cross-axis channel and its in-axis channel '
                             'are each named by a response DOF')
        if cross == in_axis:
            raise ValueError(f'{cross} is named as its own in-axis channel')
        if channels and in_axis not in channels:
            raise ValueError(f'the in-axis channel of {cross} is {in_axis}, '
                             'which is not a control channel of the '
                             f'requirement ({", ".join(channels)})')
        if cross in channels:
            raise ValueError(f'{cross} is a control channel, not a '
                             'cross-axis one')
        out[cross] = in_axis
    return out


def _limit_channels(value: Any, channels: tuple[str, ...]
                    ) -> tuple[LimitChannel, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, LimitChannel)) or not isinstance(value,
                                                                Sequence):
        raise TypeError('the limit channels are a list, each (channel, '
                        'profile, notches)')
    out = []
    for item in value:
        if not isinstance(item, LimitChannel):
            if isinstance(item, str):
                item = (item,)
            if not isinstance(item, Sequence) or not 1 <= len(item) <= 3:
                raise ValueError('each limit channel is (channel, profile, '
                                 f'notches), not {item!r}')
            item = LimitChannel(*item)
        if item.channel in channels:
            raise ValueError(f'{item.channel} is a control channel; a limit '
                             'channel is a response held under a profile')
        out.append(item)
    names = [item.channel for item in out]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise ValueError(f'the limit channel {", ".join(twice)} is named '
                         'twice')
    return tuple(out)


@dataclass(frozen=True)
class TestRequirement:
    """What a random test's run is accepted on, as the plan states it.

    Parameters
    ----------
    document : str
        The governing document, as the plan names it.
    composite_bands : sequence of Band or (low, high, below_db, above_db)
        The control band: under a single control channel, that
        channel's; under a multi-point strategy, the composite's. One
        band per frequency range; ranges may not overlap.
    revision : str
        The document's revision.
    clause : str
        The tolerance clause, as the plan copied it.
    strategy : str, default 'single'
        One of `STRATEGIES`.
    channels : sequence of str
        The control channels by response DOF. Empty takes them from the
        channel table's Control flags, or the records that pair with the
        specification (`core.multi_point.control_channels`).
    weights : sequence of float, optional
        One per channel, under 'weighted' only.
    individual_bands : sequence of Band or tuple
        Each control channel's band, under a multi-point strategy (and
        required there); none under 'single'.
    grms_percent : float, optional
        The overall rms tolerance, plus or minus, in percent of the
        reference's.
    duration : float, optional
        The nominal duration at level, seconds.
    duration_tolerance : (float, float), optional
        The duration's tolerance below and above its nominal, in
        percent; required with `duration`.
    min_dof : float, optional
        The fewest statistical degrees of freedom the control may have.
    max_line_spacing : float, optional
        The widest the control's line spacing may be, Hz.
    cross_axis : sequence of CrossAxisLimit or (low, high, percent)
        The cross-axis limits by band.
    cross_axis_channels : mapping of str to str
        Each cross-axis channel's response DOF, to the in-axis control
        channel it is read against.
    limit_channels : sequence of LimitChannel or (channel, profile, notches)
        The response channels held under a limit profile, each with the
        object its profile is in ('' for the specification) and the
        notches the plan names for it.
    rule : str, default 'any_line'
        One of `RULES`.
    combine : int, default 1
        How many adjacent lines are combined under 'combined' (two or
        more); 1 under 'any_line'.
    sources : mapping of str to Source or (origin, owner)
        Where each number came from, by field (`NUMBERS`).
    uncertainty : mapping of str to sequence of Contribution or tuple
        Each control channel's uncertainty contributions, by response
        DOF, each (name, percent, coverage, source): an expanded
        uncertainty in percent on the acceleration and its coverage
        factor (`uncertainty_budget`).

    Raises
    ------
    ValueError
        For anything the plan could not have meant: no document, no
        control band, overlapping bands, a band not about the
        reference, a strategy not in `STRATEGIES`, individual bands or
        channels a strategy cannot have, weights under a strategy that
        takes none, a tolerance that is not a positive number, a
        duration without its tolerance, a rule not in `RULES`, a
        combination the rule does not take, a source for a number the
        requirement does not hold, and an uncertainty contribution that
        is not a percentage above 0 and under 100, is named twice for a
        channel, or is entered for a channel the requirement does not
        name; cross-axis channels without cross-axis limits, or with an
        in-axis channel that is not a control channel; and a limit
        channel named twice, that is a control channel, or whose planned
        notches do not each run from a low edge to a higher one.
    """

    document: str
    composite_bands: tuple[Band, ...]
    revision: str = ''
    clause: str = ''
    strategy: str = 'single'
    channels: tuple[str, ...] = ()
    weights: tuple[float, ...] | None = None
    individual_bands: tuple[Band, ...] = ()
    grms_percent: float | None = None
    duration: float | None = None
    duration_tolerance: tuple[float, float] | None = None
    min_dof: float | None = None
    max_line_spacing: float | None = None
    cross_axis: tuple[CrossAxisLimit, ...] = ()
    rule: str = 'any_line'
    combine: int = 1
    sources: Mapping[str, Source] = field(default_factory=dict)
    uncertainty: Mapping[str, tuple[Contribution, ...]] = field(
        default_factory=dict)
    cross_axis_channels: Mapping[str, str] = field(default_factory=dict)
    limit_channels: tuple[LimitChannel, ...] = ()

    def __post_init__(self) -> None:
        put = object.__setattr__
        for name in ('document', 'revision', 'clause'):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f'the {name} is text, not '
                                 f'{getattr(self, name)!r}')
        if not self.document.strip():
            raise ValueError('a requirement names its governing document')
        put(self, 'composite_bands', _bands(self.composite_bands,
                                            'the composite bands'))
        if not self.composite_bands:
            raise ValueError('a requirement states its control band (the '
                             "composite band, or the one channel's)")
        put(self, 'individual_bands', _bands(self.individual_bands,
                                             'the individual bands'))
        if self.strategy not in STRATEGIES:
            raise ValueError(f'the strategy is one of {", ".join(STRATEGIES)}, '
                             f'not {self.strategy!r}')
        channels = self.channels
        if isinstance(channels, str) or not isinstance(channels, Sequence):
            raise TypeError('the channels are a list of response DOFs')
        channels = tuple(str(channel) for channel in channels)
        if len(set(channels)) != len(channels):
            raise ValueError('a control channel is named twice; each counts '
                             'once')
        put(self, 'channels', channels)
        if self.strategy == 'single':
            if len(channels) > 1:
                raise ValueError('a single-point requirement controls one '
                                 f'channel, not {len(channels)}')
            if self.individual_bands:
                raise ValueError('individual bands are for a multi-point '
                                 'strategy; a single channel is judged '
                                 'against the control band')
        else:
            if len(channels) == 1:
                raise ValueError(f'a {self.strategy} composite is of two or '
                                 'more control channels')
            if not self.individual_bands:
                raise ValueError('a multi-point requirement states each '
                                 "control channel's band as well as the "
                                 "composite's")
        if self.weights is not None:
            if self.strategy != 'weighted':
                raise ValueError(f'weights are for the weighted strategy; '
                                 f'{self.strategy} takes none')
            weights = tuple(_number(w, 'a weight') for w in self.weights)
            if channels and len(weights) != len(channels):
                raise ValueError(f'{len(weights)} weights for '
                                 f'{len(channels)} channels; give one per '
                                 'channel')
            put(self, 'weights', weights)
        elif self.strategy == 'weighted':
            raise ValueError('the weighted strategy needs a weight per '
                             'channel')
        for name, what in (('grms_percent', 'the rms tolerance'),
                           ('duration', 'the duration'),
                           ('min_dof', 'the fewest degrees of freedom'),
                           ('max_line_spacing', 'the widest line spacing')):
            value = getattr(self, name)
            if value is None:
                continue
            value = _number(value, what)
            if not 0.0 < value < math.inf:
                raise ValueError(f'{what} is a number above 0, not '
                                 f'{getattr(self, name)!r}')
            put(self, name, value)
        if self.duration is not None:
            if self.duration_tolerance is None:
                raise ValueError('a duration is stated with its tolerance, '
                                 'below and above, in percent')
            try:
                minus, plus = self.duration_tolerance
            except (TypeError, ValueError):
                raise ValueError('the duration tolerance is (below, above) '
                                 'in percent') from None
            minus = _number(minus, 'the duration tolerance below')
            plus = _number(plus, 'the duration tolerance above')
            if not 0.0 <= minus < 100.0 or not 0.0 <= plus < math.inf:
                raise ValueError('the duration tolerance is two percentages, '
                                 'at least 0, the one below under 100')
            put(self, 'duration_tolerance', (minus, plus))
        elif self.duration_tolerance is not None:
            raise ValueError('a duration tolerance needs the duration')
        put(self, 'cross_axis', _limits(self.cross_axis))
        put(self, 'cross_axis_channels',
            _cross_channels(self.cross_axis_channels, channels))
        if self.cross_axis_channels and not self.cross_axis:
            raise ValueError('cross-axis channels are read against the '
                             'cross-axis limits; state them')
        put(self, 'limit_channels',
            _limit_channels(self.limit_channels, channels))
        if self.rule not in RULES:
            raise ValueError(f'the decision rule is one of {", ".join(RULES)}, '
                             f'not {self.rule!r}')
        if isinstance(self.combine, bool) or not isinstance(
                self.combine, (int, np.integer)):
            raise TypeError(f'combine is a whole number, not '
                             f'{self.combine!r}')
        put(self, 'combine', int(self.combine))
        if self.rule == 'any_line' and self.combine != 1:
            raise ValueError('lines are combined only under the combined '
                             'rule')
        if self.rule == 'combined' and self.combine < 2:
            raise ValueError('the combined rule combines two or more '
                             'adjacent lines')
        sources = {}
        for name, source in dict(self.sources or {}).items():
            if name not in NUMBERS:
                raise ValueError(f'a source is stated for one of '
                                 f'{", ".join(NUMBERS)}, not {name!r}')
            if not isinstance(source, Source):
                if isinstance(source, str):
                    source = Source(source)
                else:
                    try:
                        source = Source(*source)
                    except TypeError:
                        raise ValueError(f'the source of {name} is (origin, '
                                         f'owner), not {source!r}') from None
            sources[name] = source
        put(self, 'sources', sources)
        put(self, 'uncertainty', _contributions(self.uncertainty, channels))

    @property
    def document_text(self) -> str:
        """'Test plan TP-12, revision B'."""
        return (f'{self.document}, revision {self.revision}'
                if self.revision else self.document)

    @property
    def rule_text(self) -> str:
        """The decision rule in words."""
        if self.rule == 'any_line':
            return 'any line outside its band is a miss'
        return (f'{self.combine} adjacent lines are combined, their powers '
                'added and judged against the band over the same stretch at '
                'the degrees of freedom of the sum; a group outside its band '
                'is a miss')

    def as_record(self) -> dict[str, Any]:
        """The requirement as plain values, for a file (`from_record`)."""
        return {
            'document': self.document, 'revision': self.revision,
            'clause': self.clause, 'strategy': self.strategy,
            'channels': list(self.channels),
            'weights': None if self.weights is None else list(self.weights),
            'composite_bands': [b.as_record() for b in self.composite_bands],
            'individual_bands': [b.as_record()
                                 for b in self.individual_bands],
            'grms_percent': self.grms_percent, 'duration': self.duration,
            'duration_tolerance': (None if self.duration_tolerance is None
                                   else list(self.duration_tolerance)),
            'min_dof': self.min_dof,
            'max_line_spacing': self.max_line_spacing,
            'cross_axis': [c.as_record() for c in self.cross_axis],
            'rule': self.rule, 'combine': self.combine,
            'sources': {name: [s.origin, s.owner]
                        for name, s in self.sources.items()},
            'uncertainty': {channel: [c.as_record() for c in held]
                            for channel, held in self.uncertainty.items()},
            'cross_axis_channels': dict(self.cross_axis_channels),
            'limit_channels': [c.as_record() for c in self.limit_channels]}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> TestRequirement:
        """A requirement from `as_record`'s values; refused as the
        constructor refuses, and for a field it does not take."""
        if not isinstance(record, Mapping):
            raise TypeError('a requirement is stored as a mapping of its '
                             'fields')
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(record) - known)
        if unknown:
            raise ValueError(f'a requirement has no field '
                             f'{", ".join(map(repr, unknown))}')
        if 'document' not in record or 'composite_bands' not in record:
            raise ValueError('a requirement names its document and its '
                             'control band')
        return cls(**record)

    def describe(self) -> list[str]:
        """The requirement in words, a line each: the document and its
        clause, the strategy, each band, each tolerance, the decision
        rule, and where each number came from, with the sentence for a
        number the customer supplied."""
        lines = [f'Governing document: {self.document_text}.']
        if self.clause.strip():
            lines.append(f'Tolerance clause: "{self.clause.strip()}"')
        named = ', '.join(self.channels)
        if self.strategy == 'single':
            lines.append('Control strategy: single control channel'
                         + (f', {named}.' if named else '.'))
        else:
            weights = ''
            if self.weights is not None:
                weights = ', weights ' + ', '.join(f'{w:g}'
                                                   for w in self.weights)
            lines.append(f'Control strategy: {self.strategy} of the control '
                         'channels'
                         + (f' {named}' if named else '') + weights + '.')
        noun = 'Composite band' if self.strategy != 'single' else \
            'Control band'
        lines.append(f'{noun}: '
                     + '; '.join(b.text for b in self.composite_bands)
                     + self._from('composite_bands') + '.')
        if self.individual_bands:
            lines.append('Individual band, each control channel: '
                         + '; '.join(b.text for b in self.individual_bands)
                         + self._from('individual_bands') + '.')
        if self.grms_percent is not None:
            lines.append(f'Overall rms: within ±{self.grms_percent:g} % of '
                         f'the reference{self._from("grms_percent")}.')
        if self.duration is not None:
            minus, plus = self.duration_tolerance
            lines.append(f'Duration at level: {duration_text(self.duration)}, '
                         f'−{minus:g}/+{plus:g} %{self._from("duration")}.')
        if self.min_dof is not None:
            lines.append(f'Degrees of freedom: at least {self.min_dof:g}'
                         f'{self._from("min_dof")}.')
        if self.max_line_spacing is not None:
            lines.append(f'Line spacing: at most {self.max_line_spacing:g} Hz'
                         f'{self._from("max_line_spacing")}.')
        if self.cross_axis:
            lines.append('Cross-axis: at most '
                         + '; '.join(c.text for c in self.cross_axis)
                         + self._from('cross_axis') + '.')
        if self.cross_axis_channels:
            lines.append('Cross-axis channels: ' + ', '.join(
                f'{cross} against {in_axis}'
                for cross, in_axis in self.cross_axis_channels.items()) + '.')
        if self.limit_channels:
            lines.append('Limit channels: ' + '; '.join(
                f'{c.channel}, {c.text}' for c in self.limit_channels)
                + self._from('limit_channels') + '.')
        lines.append(f'Decision rule: {self.rule_text}{self._from("rule")}.')
        if self.uncertainty:
            lines.append('Measurement uncertainty: entered for '
                         + ', '.join(self.uncertainty)
                         + ', held against one third of the tolerance'
                         + self._from('uncertainty') + '.')
        return lines

    def _from(self, name: str) -> str:
        source = self.sources.get(name)
        return f', {source.text}' if source is not None else ''


def duration_text(seconds: float) -> str:
    """'3 min 4 s', '45.5 s', '1 h 2 min'."""
    hours, rest = divmod(float(seconds), 3600.0)
    minutes, secs = divmod(rest, 60.0)
    parts = [f'{int(hours)} h'] if hours else []
    if minutes:
        parts.append(f'{int(minutes)} min')
    if secs or not parts:
        parts.append(f'{secs:.4g} s')
    return ' '.join(parts)


@dataclass(frozen=True)
class Excursion:
    """One stretch of adjacent lines outside a band on one side."""

    #: 'above' or 'below' the band
    side: str
    #: its reach, Hz
    low: float
    high: float
    #: how far past the band's edge it went at worst, dB; infinite for a
    #: stretch that reads zero below a band
    worst_db: float
    #: how many lines (or, combined, groups) it holds
    lines: int

    @property
    def text(self) -> str:
        """'above by up to 1.4 dB from 617.5 to 660 Hz (18 lines)'."""
        if math.isinf(self.worst_db):
            size = 'reading zero'
        else:
            # two figures under 10 dB, whole decibels above
            places = '.2g' if self.worst_db < 10 else '.0f'
            size = f'by up to {self.worst_db:{places}} dB'
        count = f'{self.lines} line{"s" * (self.lines != 1)}'
        return (f'{self.side} {size} from {_hz(self.low)} to '
                f'{_hz(self.high)} Hz ({count})')


@dataclass(frozen=True)
class Offer:
    """The fact a row of the result table offers a findings register:
    what was measured, in words, and the channel and band it concerns.

    A row offers one where the run departed from the plan: a notch the
    plan does not name, and a cross-axis channel over its limit. The
    report builds a fact draft from it, naming the record, and it
    reaches a finding only through the register's one entry point for
    drafts (`report.findings.offer`), which leaves it for the author to
    accept, edit or discard.
    """

    #: the fact, without the record it was read off
    text: str
    #: the channel it concerns
    channel: str
    #: the band it concerns, Hz
    band: tuple[float, float]


@dataclass(frozen=True)
class Criterion:
    """One row of the result table."""

    #: what the row judges: 'composite', 'channel', 'grms', 'duration',
    #: 'dof', 'line_spacing', 'cross_axis' or 'limit'
    key: str
    #: the row's name ('Composite (average of 1X+, 2X+)', 'Channel 2X+')
    name: str
    #: what was required, in words
    required: str
    #: what was measured, in words: a value, or each excursion
    measured: str
    #: `MET`, `NOT_MET` or `NOT_JUDGED`
    state: str
    #: the stretches outside the band, for a band row
    excursions: tuple[Excursion, ...] = ()
    #: the channel the row is of, '' for the composite and the run's rows
    channel: str = ''
    #: the frequency range the row judged, Hz, None for a row of no band
    band: tuple[float, float] | None = None
    #: the fact the row offers a findings register (`Offer`): on an
    #: unplanned notch and on a cross-axis channel over its limit, None
    #: on every other row
    finding: Offer | None = None


@dataclass(frozen=True)
class Results:
    """What `evaluate` read: one row per criterion, and the notes."""

    requirement: TestRequirement
    rows: tuple[Criterion, ...]
    #: the dB added to the measurement for every comparison
    scale_db: float
    #: the control channels, by response DOF
    channels: tuple[str, ...]
    #: what a reader of the rows should know: a scale detected, a dead
    #: channel, lines not judged, units undeclared
    notes: tuple[str, ...] = ()

    @property
    def state(self) -> str:
        """`NOT_MET` when any row is not met; `NOT_JUDGED` when none is
        not met and any was not judged; `MET` only when every row is."""
        states = {row.state for row in self.rows}
        if NOT_MET in states:
            return NOT_MET
        if NOT_JUDGED in states or not self.rows:
            return NOT_JUDGED
        return MET

    @property
    def met(self) -> int:
        """How many rows are met."""
        return sum(1 for row in self.rows if row.state == MET)


def evaluate(requirement: TestRequirement, specification: Specification,
             measured: Psd, *, table: Any = None,
             duration_run: float | None = None,
             spec_record: int | None = None,
             scale_db: float | None = None,
             profiles: Mapping[str, Any] | None = None) -> Results:
    """Read a run against its requirement, one row per criterion.

    The control channels are the requirement's, else the channel
    table's Control flags, else the autospectra that pair with the
    specification (`core.multi_point.control_channels`, and for a
    single channel the one record that pairs). The reference is one
    record of the specification, chosen as `core.multi_point` chooses
    it, and the scale is resolved once for the run
    (`compliance.comparison_scale_db`) and used for every row.

    Parameters
    ----------
    requirement : TestRequirement
        What the run is accepted on.
    specification : Specification
        The reference.
    measured : Psd
        The run's autospectra.
    table : ChannelTable, optional
        The channel table, whose Control flags name the control channels
        when the requirement names none.
    duration_run : float, optional
        How long the run was at level, seconds: the controller's
        counter. Without it the duration row is not judged.
    spec_record : int, optional
        The reference record.
    scale_db : float, optional
        The comparison's scale; resolved as above when omitted.
    profiles : mapping of str to Psd, optional
        The objects the limit channels name for their profiles, by name.

    Returns
    -------
    Results
        The rows and the notes.

    Raises
    ------
    TypeError
        When the reference is not a random specification or the run not
        a measured PSD.
    ValueError
        With the reason, for a run the requirement cannot be read
        against at all: control channels not found, ambiguous or a cross
        term, channels of different quantities, a weight per channel
        that does not fit them, or a duration that is not a positive
        number. A criterion that cannot be judged is a row saying why,
        not a refusal.
    """
    from .compliance import comparison_scale_db
    from .data import Psd, Specification
    from .multi_point import _reference, multi_point

    if not isinstance(requirement, TestRequirement):
        raise TypeError('the requirement is a TestRequirement')
    if not isinstance(specification, Specification):
        raise TypeError('the reference is a random specification')
    if not isinstance(measured, Psd) or isinstance(measured, Specification):
        raise TypeError('the run is a measured PSD')
    if duration_run is not None:
        duration_run = _number(duration_run, 'the duration at level')
        if not 0.0 < duration_run < math.inf:
            raise ValueError('the duration at level is a number of seconds '
                             f'above 0, not {duration_run!r}')
    records = _records(requirement, specification, measured, table)
    labels = [str(measured.response_dof[i]) for i in records]
    j = _reference(specification, measured, records, spec_record)
    notes: list[str] = []
    if scale_db is None:
        scale_db = comparison_scale_db(specification, measured, None,
                                       records, labels)
        if scale_db and getattr(measured, 'scale_db', None) is None:
            notes.append(f'the measurement was scaled by {scale_db:+g} dB, '
                         f'detected from the data as a run at '
                         f'{-scale_db:+g} dB')
    scale_db = float(scale_db)
    dofs = _dofs(measured, records)

    rows: list[Criterion] = []
    if requirement.strategy == 'single':
        target, target_record = measured, records[0]
        name = f'Channel {labels[0]}'
    else:
        first = requirement.individual_bands[0]
        reading = multi_point(
            specification, measured, records, requirement.strategy,
            requirement.weights,
            composite_band=(requirement.composite_bands[0].below_db,
                            requirement.composite_bands[0].above_db),
            individual_band=(first.below_db, first.above_db),
            spec_record=j, scale_db=scale_db)
        # what the reading noticed of the channels (a dead one, lines not
        # formed, a zero weight), but not its band's chance share, which
        # is of one band of several, nor its scale, said above
        notes.extend(note for note in reading.notes
                     if 'by chance alone' not in note
                     and not note.startswith('the measurement was scaled'))
        target, target_record = reading.composite, 0
        name = f'Composite ({requirement.strategy} of {", ".join(labels)})'
    for band in requirement.composite_bands:
        rows.append(_band_row('composite', name, '', band, specification, j,
                              target, target_record, scale_db, requirement,
                              dofs))
    for band in requirement.individual_bands:
        for record, label in zip(records, labels, strict=True):
            rows.append(_band_row('channel', f'Channel {label}', label, band,
                                  specification, j, measured, record,
                                  scale_db, requirement, _dofs(measured,
                                                               [record])))
    if requirement.grms_percent is not None:
        rows.append(_grms_row(requirement, specification, j, target,
                              target_record, scale_db,
                              'composite' if requirement.strategy != 'single'
                              else labels[0]))
    if requirement.duration is not None:
        rows.append(_duration_row(requirement, duration_run))
    if requirement.min_dof is not None:
        rows.append(_dof_row(requirement, dofs, labels))
    if requirement.max_line_spacing is not None:
        rows.append(_spacing_row(requirement, measured))
    rows.extend(cross_axis_rows(requirement, specification, measured, table,
                                scale_db, spec_record=j))
    rows.extend(limit_rows(requirement, specification, j, target,
                           target_record, measured, scale_db, profiles))
    if requirement.strategy == 'single':
        # a multi-point reading names a dead channel itself (above)
        zero = sum(e.lines for row in rows if row.key == 'composite'
                   for e in row.excursions if math.isinf(e.worst_db))
        if zero:
            notes.append(f'{labels[0]} reads zero at {zero} '
                         f'line{"s" * (zero != 1)}, as a dead or '
                         'disconnected sensor does')
    return Results(requirement, tuple(rows), scale_db, tuple(labels),
                   tuple(dict.fromkeys(notes)))


def cross_axis_rows(requirement: TestRequirement,
                    specification: Specification, measured: Psd,
                    table: Any = None, scale_db: float = 0.0, *,
                    spec_record: int = 0) -> list[Criterion]:
    """The rows of the requirement's cross-axis limits: one per limit
    and cross-axis channel.

    Each cross-axis channel's autospectrum is read as a percentage of
    its in-axis control channel's, line by line (a ratio of two
    densities, so the comparison's scale cancels), over the limit's
    band, which a limit left open at an end takes to the end the
    specification writes. The row gives the worst percentage and its
    frequency, how many lines were over the limit, and the degrees of
    freedom the two were read at, which must be the same for the ratio
    to be of like estimates. Met when no judged line is over. A line
    where either density is not a finite number, or the in-axis one is
    zero, is not judged; a row with such lines and none over is not
    judged, as a band row is, and so is a cross-axis channel that
    reads zero at every line, as a dead sensor does. A row with lines
    over offers their reach to a findings register (`Criterion.finding`).
    With no cross-axis channel named, each limit is one row saying so.
    This is the one place the rows are made.
    """
    rows = []
    if not requirement.cross_axis_channels:
        return [Criterion('cross_axis', 'Cross-axis', limit.text,
                          'not judged: no cross-axis channel is named',
                          NOT_JUDGED, band=_band_of(limit))
                for limit in requirement.cross_axis]
    for limit in requirement.cross_axis:
        for cross, in_axis in requirement.cross_axis_channels.items():
            rows.append(_cross_axis_row(limit, cross, in_axis,
                                        specification, spec_record,
                                        measured))
    return rows


def _autospectrum(measured: Psd, dof: str) -> int | None:
    """The one autospectrum of `measured` at `dof`, None for none;
    `ValueError` for two (of two quantities at one DOF), since which
    was meant cannot be told."""
    found = [i for i in range(measured.num_records)
             if str(measured.response_dof[i]) == dof
             and measured.is_autospectrum(i)]
    if len(found) > 1:
        raise ValueError(f'the measurement holds {len(found)} autospectra '
                         f'at {dof}; which is meant cannot be told')
    return found[0] if found else None


def _cross_axis_row(limit: CrossAxisLimit, cross: str, in_axis: str,
                    specification: Specification, j: int, measured: Psd
                    ) -> Criterion:
    from .compliance import coverage, units_check

    name = f'Cross-axis {cross} (in-axis {in_axis})'
    reach = _band_of(limit)

    def unjudged(why: str) -> Criterion:
        return Criterion('cross_axis', name, limit.text,
                         f'not judged: {why}', NOT_JUDGED, channel=cross,
                         band=reach)

    try:
        records = [_autospectrum(measured, dof) for dof in (cross, in_axis)]
    except ValueError as refusal:
        return unjudged(str(refusal))
    for dof, record in zip((cross, in_axis), records, strict=True):
        if record is None:
            return unjudged(f'the measurement has no autospectrum at {dof}')
    c, i = records
    refused, _raw = units_check(str(measured.ordinate_dim[i]),
                                str(measured.ordinate_dim[c]),
                                wanted_name=f'the in-axis record {in_axis}',
                                held_name=f'the cross-axis record {cross}')
    if refused is not None:
        return unjudged(refused)
    dofs = _dofs(measured, [c]), _dofs(measured, [i])
    if None not in dofs and abs(dofs[0] - dofs[1]) > 1e-9 * dofs[1]:
        return unjudged(f'the two are read at different degrees of freedom '
                        f'({dofs[0]:.4g} and {dofs[1]:.4g}), so their ratio '
                        'is not of like estimates')
    # a limit open at an end reaches the end the specification writes
    written = coverage(specification, j)
    low = (min(a for a, _b in written) if limit.low is None and written
           else limit.low or 0.0)
    high = (max(b for _a, b in written) if limit.high is None and written
            else limit.high or math.inf)
    x = np.asarray(measured.abscissa, dtype=float)
    inside = (x >= low) & (x <= high)
    if not inside.any():
        return unjudged(f'the measurement has no line {_range_text(low, high)}')
    held = np.real(np.asarray(measured.ordinate[c], dtype=float))[inside]
    under = np.real(np.asarray(measured.ordinate[i], dtype=float))[inside]
    with np.errstate(divide='ignore', invalid='ignore'):
        read = (np.isfinite(held) & (held >= 0.0) & np.isfinite(under)
                & (under > 0.0))
        percent = np.where(read, 100.0 * held / under, np.nan)
    count = int(read.sum())
    gaps = int(inside.sum()) - count
    if not count:
        return unjudged('no line of the band has both densities finite and '
                        'the in-axis one above zero')
    if not (held[read] > 0.0).any():
        return unjudged(f'{cross} reads zero over the band, as a dead or '
                        'disconnected sensor does')
    k = int(np.nanargmax(percent))
    worst, at = float(percent[k]), float(x[inside][k])
    over = int((percent[read] > limit.percent * (1.0 + 1e-12)).sum())
    words = f'worst {worst:.3g} % at {_hz(at)} Hz'
    if over:
        words += f', {over} of {count} line{"s" * (count != 1)} over'
    if None not in dofs:
        words += f', at {dofs[0]:.4g} degrees of freedom'
    if gaps:
        words += f'; {gaps} line{"s" * (gaps != 1)} not judged'
    state = NOT_MET if over else (NOT_JUDGED if gaps else MET)
    offer = None
    if over:
        # a cross-axis channel over its limit is a departure from the
        # plan, so the row offers it to a findings register (`finding`)
        lines = x[inside][read][percent[read] > limit.percent
                                * (1.0 + 1e-12)]
        first, last = float(lines[0]), float(lines[-1])
        offer = Offer(
            f'{cross} read up to {worst:.3g} % of {in_axis} at {_hz(at)} '
            f'Hz, over its {limit.percent:g} % cross-axis limit on {over} '
            f'of {count} line{"s" * (count != 1)} from {_hz(first)} to '
            f'{_hz(last)} Hz', cross, (first, last))
    return Criterion('cross_axis', name, limit.text, words, state,
                     channel=cross, band=(low, high), finding=offer)


def limit_rows(requirement: TestRequirement, specification: Specification,
               j: int, target: Psd, target_record: int, measured: Psd,
               scale_db: float, profiles: Mapping[str, Any] | None = None
               ) -> list[Criterion]:
    """The rows of the requirement's limit channels: for each, the
    notches it cut, the as-run notch.

    A controller holds its control to the reference except where that
    would put a limit channel over its profile; there it lowers the drive
    until the limit channel sits on its profile, and the control falls
    below the reference. So a line is notched where the limit channel is
    at its profile (no more than `HELD_DB` under it) while the control is
    more than `HELD_DB` under the reference, every curve read over the
    measurement's lines the way `core.limit_prediction` reads them
    (`limit_prediction._density`) and the measurement at the comparison's
    scale. A run of at least `limit_prediction.NARROW_LINES` adjacent
    notched lines is a notch, since a controller cuts no narrower and a
    line or two of scatter is not one; it is given by its band, how far
    under the reference the control went at worst, and its line count
    (an `Excursion` below). The row is not met when a notch lies outside
    every notch the plan names, and offers those notches to a findings
    register (`Criterion.finding`); its words say how close the limit
    channel came to its profile. A limit channel, profile or reference
    not read, and a limit channel reading zero, are a row saying so.

    `profiles` holds the objects a limit channel names for its profile,
    by name; a limit channel naming none takes the specification's own
    record at its DOF, as `core.limit_prediction.predict_limits` does.
    """
    from .limit_prediction import _density

    rows = []
    if not requirement.limit_channels:
        return rows
    left, right = target.bin_bounds()
    reference = _density(specification, j, left, right)
    factor = 10.0 ** (scale_db / 10.0)
    control = factor * np.real(np.asarray(target.ordinate[target_record],
                                          dtype=float))
    for channel in requirement.limit_channels:
        rows.append(_limit_row(channel, specification, measured, reference,
                               control, factor, left, right,
                               dict(profiles or {})))
    return rows


def _limit_row(channel: LimitChannel, specification: Specification,
               measured: Psd, reference: np.ndarray, control: np.ndarray,
               factor: float, left: np.ndarray, right: np.ndarray,
               profiles: Mapping[str, Any]) -> Criterion:
    from .compliance import units_check
    from .data import channel_quantities
    from .limit_prediction import NARROW_LINES, _density, _record_at, _runs

    dof = channel.channel
    name = f'Limit channel {dof}'
    required = channel.text

    def unjudged(why: str) -> Criterion:
        return Criterion('limit', name, required, f'not judged: {why}',
                         NOT_JUDGED, channel=dof)

    try:
        record = _autospectrum(measured, dof)
    except ValueError as refusal:
        return unjudged(str(refusal))
    if record is None:
        return unjudged(f'the measurement has no autospectrum at {dof}')
    quantity = channel_quantities(str(measured.ordinate_dim[record]))[0]
    source = specification if not channel.profile else \
        profiles.get(channel.profile)
    if source is None:
        return unjudged(f'its limit profile {channel.profile!r} is not '
                        'there to read')
    found = _record_at(source, dof, quantity)
    if found is None:
        return unjudged(f'no limit profile is written at {dof}')
    refused, _raw = units_check(
        channel_quantities(str(source.ordinate_dim[found]))[0], quantity,
        wanted_name=f'the limit profile at {dof}',
        held_name='the limit channel')
    if refused is not None:
        return unjudged(refused)
    profile = _density(source, found, left, right)
    limited = factor * np.real(np.asarray(measured.ordinate[record],
                                          dtype=float))
    read = (np.isfinite(reference) & np.isfinite(profile)
            & np.isfinite(limited) & (limited >= 0.0) & np.isfinite(control))
    if not read.any():
        return unjudged('no line has the limit channel, its profile, the '
                        'control and the reference all read')
    if not (limited[read] > 0.0).any():
        return unjudged(f'{dof} reads zero, as a dead or disconnected sensor '
                        'does')
    slack = 10.0 ** (-HELD_DB / 10.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        held = read & (limited >= profile * slack)
        below = read & (control < reference * slack)
        depth = 10.0 * np.log10(reference / control)
        closest = np.where(read, 10.0 * np.log10(limited / profile), np.nan)
    x = np.asarray(measured.abscissa, dtype=float)
    notches = []
    for first, last in _runs(held & below):
        if last - first + 1 < NARROW_LINES:
            continue
        notches.append(Excursion('below', float(x[first]), float(x[last]),
                                 float(np.max(depth[first:last + 1])),
                                 last - first + 1))
    unplanned = [notch for notch in notches if not channel.planned(notch)]
    k = int(np.nanargmax(closest))
    near = (f'{float(closest[k]):+.2g} dB from its limit profile at its '
            f'closest, {_hz(float(x[k]))} Hz')
    if not notches:
        return Criterion('limit', name, required, f'no notch; {near}', MET,
                         channel=dof)
    said = []
    for notch in notches:
        said.append(f'held its limit from {_hz(notch.low)} to '
                    f'{_hz(notch.high)} Hz ({notch.lines} lines) while the '
                    f'control fell up to {notch.worst_db:.2g} dB under the '
                    'reference'
                    + ('' if channel.planned(notch) else
                       ', a notch the plan does not name'))
    # an unplanned notch is a departure from the plan, so the row
    # offers it to a findings register (`finding`)
    return Criterion('limit', name, required, '; '.join(said) + f'; {near}',
                     NOT_MET if unplanned else MET, tuple(notches),
                     channel=dof,
                     band=(notches[0].low, notches[-1].high),
                     finding=_notch_offer(dof, unplanned))


def _depth(db: float) -> str:
    """How far a notch took the control under the reference, in words:
    two figures under 10 dB, whole decibels above, as `Excursion.text`."""
    if math.isinf(db):
        return 'to zero'
    return f'up to {db:.2g} dB' if db < 10 else f'up to {db:.0f} dB'


def _notch_offer(dof: str, unplanned: Sequence[Excursion]) -> Offer | None:
    """The fact the notches the plan does not name offer a findings
    register: the channel, each notch's band, line count and depth, and
    their reach as the band. None when every notch is planned."""
    if not unplanned:
        return None
    each = [f'from {_hz(notch.low)} to {_hz(notch.high)} Hz '
            f'({notch.lines} line{"s" * (notch.lines != 1)}), the control '
            f'{_depth(notch.worst_db)} under the reference'
            for notch in unplanned]
    count = ('a notch' if len(unplanned) == 1
             else f'{len(unplanned)} notches')
    return Offer(f'{dof} held its limit profile in {count} the plan does '
                 f'not name: ' + '; '.join(each), dof,
                 (unplanned[0].low, unplanned[-1].high))


def _band_of(item: Any) -> tuple[float, float] | None:
    if item.low is None and item.high is None:
        return None
    return (item.low if item.low is not None else 0.0,
            item.high if item.high is not None else math.inf)


def _records(requirement: TestRequirement, specification: Specification,
             measured: Psd, table: Any) -> list[int]:
    """The control records: under a multi-point strategy by the rule
    `core.multi_point.control_channels` uses, and for a single channel
    the one named, else the one the channel table flags, else the one
    autospectrum that pairs with the specification."""
    from .compliance import matched_records
    from .multi_point import control_channels

    named = list(requirement.channels) or None
    if requirement.strategy != 'single':
        return control_channels(specification, measured, named, table)
    # an autospectrum by the one rule (`core.data.autospectrum`): one
    # DOF and one quantity on both sides, so a drive point's force
    # against its acceleration is not read as a control channel
    autos = [i for i in range(measured.num_records)
             if measured.is_autospectrum(i)]
    if named:
        at = [i for i in autos if str(measured.response_dof[i]) == named[0]]
        if not at:
            raise ValueError(f'the measurement has no autospectrum at '
                             f'{named[0]}')
    else:
        flagged = (np.asarray(table.controls(), dtype=bool)
                   if table is not None else None)
        if flagged is not None and flagged.any():
            dofs = table.dof_strings()
            wanted = {dofs[k] for k in np.flatnonzero(flagged)}
            at = [i for i in autos if str(measured.response_dof[i]) in wanted]
            where = 'the channel table flags'
        else:
            at = list(dict.fromkeys(i for _l, _j, i in matched_records(
                specification, measured) if i in autos))
            where = 'pair with the specification'
        if not at:
            raise ValueError('no control channel is found: none of the '
                             f"measurement's autospectra {where}; name it")
    if len(at) > 1:
        said = ', '.join(str(measured.response_dof[i]) for i in at)
        raise ValueError(f'a single-point requirement controls one channel; '
                         f'{len(at)} autospectra are candidates ({said}), so '
                         'name the one meant')
    return at


def _dofs(measured: Psd, records: Sequence[int]) -> float | None:
    """The fewest statistical degrees of freedom among `records`, None
    when the PSD does not carry them for every one."""
    counts = getattr(measured, 'statistical_dof', None)
    if counts is None:
        return None
    counts = np.asarray(counts, dtype=float)[list(records)]
    if not counts.size or not np.isfinite(counts).all() \
            or not (counts > 0.0).all():
        return None
    return float(counts.min())


def within(specification: Specification, record: int, low: float | None,
           high: float | None) -> Specification:
    """One record of `specification` cut to `low` to `high` Hz.

    A specification of breakpoints gains a breakpoint at each edge,
    interpolated log-log as it is read (`compliance.log_interpolate`),
    so a line across an edge is judged in each range over its own part,
    as `compliance.cells` cuts a cell to what is written. One on lines
    or bands keeps the points from `low` to `high`. Raises `ValueError`
    when the range holds nothing the specification writes.
    """
    from .compliance import coverage, log_interpolate
    from .data import Specification

    x = np.asarray(specification.abscissa, dtype=float)
    values = np.real(np.asarray(specification.ordinate[record], dtype=float))
    written = coverage(specification, record)
    start = min(a for a, _b in written) if written else math.nan
    stop = max(b for _a, b in written) if written else math.nan
    lo = start if low is None else max(low, start)
    hi = stop if high is None else min(high, stop)
    if not written or not lo < hi:
        raise ValueError(f'the band {_range_text(low, high)} holds nothing '
                         'the specification writes (it runs from '
                         f'{_hz(start)} to {_hz(stop)} Hz)')
    bandwidth = getattr(specification, 'bandwidth', None)
    if getattr(specification, 'interpolation', 'bin') == 'log_log':
        inner = x[(x > lo) & (x < hi)]
        abscissa = np.concatenate([[lo], inner, [hi]])
        held = log_interpolate(abscissa, x, values)
        kept_bandwidth = None
    else:
        keep = (x >= lo) & (x <= hi)
        if not keep.any():
            raise ValueError(f'the band {_range_text(low, high)} holds no '
                             'line of the specification')
        abscissa, held = x[keep], values[keep]
        kept_bandwidth = (None if bandwidth is None
                          else np.asarray(bandwidth)[keep])
    out = Specification(
        abscissa, held[None],
        response_dof=[specification.response_dof[record]],
        ordinate_dim=[specification.ordinate_dim[record]],
        ordinate_unit=[specification.ordinate_unit[record]],
        dimension_hint=[specification.dimension_hint[record]],
        bandwidth=kept_bandwidth)
    out.interpolation = specification.interpolation
    return out


def _band_row(key: str, name: str, channel: str, band: Band,
              specification: Specification, j: int, target: Psd,
              record: int, scale_db: float, requirement: TestRequirement,
              dof: float | None) -> Criterion:
    """One band row: `target`'s record against `band` about reference
    record `j`, under the requirement's rule."""
    from .multi_point import _band

    reach = _band_of(band)
    try:
        cut = within(specification, j, band.low, band.high)
        held, _text = _band(cut, 0, (band.below_db, band.above_db), key)
    except ValueError as refusal:
        return Criterion(key, name, band.text, f'not judged: {refusal}',
                         NOT_JUDGED, channel=channel, band=reach)
    if requirement.rule == 'combined':
        state, words, found = _combined(held, target, record, scale_db,
                                        requirement.combine, dof)
    else:
        state, words, found = _lines(held, target, record, scale_db)
    written = coverage_reach(held)
    return Criterion(key, name, band.text, words, state, tuple(found),
                     channel=channel, band=reach or written)


def coverage_reach(specification: Specification) -> tuple[float, float]:
    """(low, high) Hz of what one-record `specification` writes."""
    from .compliance import coverage

    written = coverage(specification, 0)
    return (min(a for a, _b in written), max(b for _a, b in written))


def _lines(band: Specification, target: Psd, record: int,
           scale_db: float) -> tuple[str, str, list[Excursion]]:
    """(state, words, excursions) under 'any_line': every cell of
    `compliance.judge` against each edge of the band."""
    from .compliance import cell_lines, compare, judge

    result = compare(band, target, 0, record, scale_db=scale_db)
    if result.get('refused'):
        return NOT_JUDGED, f'not judged: {result["refused"]}', []
    # a line that is not a finite, non-negative density is no reading: a
    # cell holding one is counted not judged (`unread`), never a miss
    values = np.real(np.asarray(target.ordinate[record], dtype=float))
    with np.errstate(invalid='ignore'):
        bad = ~(np.isfinite(values) & (values >= 0.0))
    left, right = target.bin_bounds()
    found: list[tuple[str, float, float, float]] = []
    marks = []
    for edge, over, side in (('abort_upper', True, 'above'),
                             ('abort_lower', False, 'below')):
        if band.limits.get(edge) is None:
            continue
        read = judge(band, target, 0, record, edge, over, scale_db)
        marks.append(read['judged'])
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = (read['held'] / read['asked'] if over
                     else read['asked'] / read['held'])
            size = 10.0 * np.log10(ratio)
        for k in np.flatnonzero(read['out']):
            cell = read['cells'][k]
            reach = cell_lines(cell, left, right)
            if reach is not None and bad[reach].any():
                continue
            found.append((side, cell['low'], cell['high'], float(size[k])))
    count = int(np.logical_and.reduce(marks).sum())
    if not count:
        return (NOT_JUDGED,
                ('not judged: no line of the record in the band is a finite '
                 'density'), [])
    excursions = _grouped(found)
    gaps = unread(band, target, record)
    return _said(excursions, count, gaps, 'line')


def _combined(band: Specification, target: Psd, record: int,
              scale_db: float, combine: int, dof: float | None
              ) -> tuple[str, str, list[Excursion]]:
    """(state, words, excursions) under 'combined': the groups of
    `core.chance.compare_chance`."""
    from .chance import compare_chance

    if dof is None:
        return (NOT_JUDGED, ('not judged: combining lines needs the '
                             "measurement's degrees of freedom, which it "
                             'does not carry'), [])
    result = compare_chance(band, target, 0, record, dof=dof,
                            combine=combine, scale_db=scale_db)
    if result.get('refused'):
        return NOT_JUDGED, f'not judged: {result["refused"]}', []
    if not result.get('lines'):
        return (NOT_JUDGED, ('not judged: no group of lines in the band '
                             'holds a finite density'), [])
    groups = result['groups']
    found = []
    miss = np.asarray(groups['miss_db'], dtype=float)
    for side, flag, edge in (('above', 'abort_above', 'abort_upper_db'),
                             ('below', 'abort_below', 'abort_lower_db')):
        if flag not in groups:
            continue
        out = np.asarray(groups[flag], dtype=bool)
        edges = np.asarray(groups[edge], dtype=float)
        size = miss - edges if side == 'above' else edges - miss
        for k in np.flatnonzero(out):
            found.append((side, float(groups['low'][k]),
                          float(groups['high'][k]), float(size[k])))
    excursions = _grouped(found)
    state, words, _ = _said(excursions, int(result['lines']),
                            unread(band, target, record), 'group')
    expected = float(result.get('abort_expected', 0.0))
    words += (f'; {combine} lines a group at about '
              f'{float(result["judged_dof"]):.3g} degrees of freedom, '
              f'{expected:.2g} groups expected outside by chance')
    if result.get('note'):
        words += f'; {result["note"]}'
    return state, words, excursions


def unread(band: Specification, target: Psd, record: int) -> int:
    """How many lines of `target`'s record inside what `band` writes are
    not a finite density, and so were not judged: a sample that is not a
    number, or a line of a composite not formed."""
    x = np.asarray(target.abscissa, dtype=float)
    values = np.real(np.asarray(target.ordinate[record], dtype=float))
    low, high = coverage_reach(band)
    inside = (x >= low) & (x <= high)
    with np.errstate(invalid='ignore'):
        return int((inside & ~(np.isfinite(values) & (values >= 0.0))).sum())


def _grouped(found: list[tuple[str, float, float, float]]
             ) -> list[Excursion]:
    """Adjacent cells outside on one side joined into excursions, in
    frequency order."""
    out: list[Excursion] = []
    for side in ('above', 'below'):
        mine = sorted((low, high, size) for s, low, high, size in found
                      if s == side)
        run: list[tuple[float, float, float]] = []
        for cell in mine:
            if run and cell[0] > run[-1][1] * (1.0 + _TOUCHING):
                out.append(_excursion(side, run))
                run = []
            run.append(cell)
        if run:
            out.append(_excursion(side, run))
    return sorted(out, key=lambda e: e.low)


def _excursion(side: str, run: list[tuple[float, float, float]]
               ) -> Excursion:
    return Excursion(side, run[0][0], run[-1][1],
                     max(size for _l, _h, size in run), len(run))


def _said(excursions: list[Excursion], judged: int, gaps: int,
          unit: str) -> tuple[str, str, list[Excursion]]:
    """The state and words of a band row from its excursions."""
    units = f'{unit}{"s" * (judged != 1)}'
    if excursions:
        worst = sorted(excursions, key=lambda e: -e.worst_db)
        shown = [e.text for e in worst[:MOST_EXCURSIONS]]
        more = len(excursions) - MOST_EXCURSIONS
        if more > 0:
            outside = sum(e.lines for e in excursions)
            shown.insert(0, f'{len(excursions)} excursions, {outside} '
                         f'{unit}s outside; the worst:')
            shown[0] += f' {shown.pop(1)}'
            shown.append(f'and {more} more')
        if unit == 'group':
            shown = [text.replace(' line)', ' group)')
                     .replace(' lines)', ' groups)') for text in shown]
        words = '; '.join(shown)
        if gaps:
            words += f'; {gaps} {unit}{"s" * (gaps != 1)} not judged'
        return NOT_MET, words, excursions
    if gaps:
        return (NOT_JUDGED,
                (f'inside the band at {judged} {units}; {gaps} '
                 f'{unit}{"s" * (gaps != 1)} not judged (not a finite '
                 'density)'), [])
    return MET, f'inside the band at all {judged} {units}', []


def _grms_row(requirement: TestRequirement, specification: Specification,
              j: int, target: Psd, record: int, scale_db: float,
              which: str) -> Criterion:
    from .compliance import compare

    tolerance = requirement.grms_percent
    required = f'within ±{tolerance:g} % of the reference'
    result = compare(specification, target, j, record, scale_db=scale_db)
    name = 'Overall rms' + (' (composite)' if which == 'composite' else '')
    if result.get('refused'):
        return Criterion('grms', name, required,
                         f'not judged: {result["refused"]}', NOT_JUDGED)
    difference = float(result.get('difference_percent', math.nan))
    if not math.isfinite(difference):
        return Criterion('grms', name, required,
                         'not judged: the record has no finite rms over the '
                         'band', NOT_JUDGED)
    if not result.get('measured_rms'):
        return Criterion('grms', name, required,
                         'reads zero over the band, as a dead or '
                         'disconnected sensor does (−100 %)', NOT_MET)
    state = MET if abs(difference) <= tolerance * (1.0 + 1e-12) else NOT_MET
    return Criterion('grms', name, required,
                     f'{difference:+.1f} % of the reference rms', state,
                     band=tuple(result['band']))


def _duration_row(requirement: TestRequirement,
                  duration_run: float | None) -> Criterion:
    nominal = requirement.duration
    minus, plus = requirement.duration_tolerance
    required = f'{duration_text(nominal)}, −{minus:g}/+{plus:g} %'
    if duration_run is None:
        return Criterion('duration', 'Duration at level', required,
                         'not judged: no duration at level stated for the '
                         'run', NOT_JUDGED)
    low = nominal * (1.0 - minus / 100.0)
    high = nominal * (1.0 + plus / 100.0)
    state = MET if low * (1 - 1e-12) <= duration_run <= high * (1 + 1e-12) \
        else NOT_MET
    off = 100.0 * (duration_run - nominal) / nominal
    return Criterion('duration', 'Duration at level', required,
                     f'{duration_text(duration_run)} ({off:+.1f} %)', state)


def _dof_row(requirement: TestRequirement, dof: float | None,
             labels: Sequence[str]) -> Criterion:
    required = f'at least {requirement.min_dof:g}'
    if dof is None:
        return Criterion('dof', 'Degrees of freedom', required,
                         'not judged: the measurement does not carry its '
                         'degrees of freedom', NOT_JUDGED)
    state = MET if dof >= requirement.min_dof * (1.0 - 1e-12) else NOT_MET
    fewest = ', the fewest of the control channels' if len(labels) > 1 else ''
    return Criterion('dof', 'Degrees of freedom', required,
                     f'{dof:.4g}{fewest}', state)


def _spacing_row(requirement: TestRequirement, measured: Psd) -> Criterion:
    required = f'at most {requirement.max_line_spacing:g} Hz'
    x = np.asarray(measured.abscissa, dtype=float)
    steps = np.diff(x)
    if getattr(measured, 'bandwidth', None) is not None or not steps.size:
        return Criterion('line_spacing', 'Line spacing', required,
                         'not judged: the measurement is not on evenly '
                         'spaced lines', NOT_JUDGED)
    spacing = float(np.median(steps))
    if np.ptp(steps) > 1e-6 * spacing:
        return Criterion('line_spacing', 'Line spacing', required,
                         'not judged: the measurement is not on evenly '
                         'spaced lines', NOT_JUDGED)
    state = (MET if spacing <= requirement.max_line_spacing * (1.0 + 1e-9)
             else NOT_MET)
    return Criterion('line_spacing', 'Line spacing', required,
                     f'{spacing:.4g} Hz', state)


@dataclass(frozen=True)
class Budget:
    """One control channel's measurement uncertainty against one third
    of the tolerance (`uncertainty_budget`)."""

    #: the control channel, by response DOF
    channel: str
    #: what was entered for it
    contributions: tuple[Contribution, ...]
    #: the combined uncertainty expanded at `COVERAGE`, in percent on the
    #: acceleration, on the ASD, and in dB; NaN with nothing entered
    acceleration_percent: float
    asd_percent: float
    db: float
    #: one third of the tightest band edge, dB, and that edge
    band_limit_db: float
    band_db: float
    #: one third of the overall rms tolerance, percent, None without one
    grms_limit_percent: float | None
    #: `MET`, `NOT_MET` or `NOT_JUDGED`
    state: str
    #: why, when not judged; what exceeds its third, when not met
    words: str


def combined_percent(contributions: Sequence[Contribution]) -> float:
    """The contributions' combined uncertainty, expanded at `COVERAGE`,
    in percent on the acceleration: each divided by its own coverage
    factor, combined as the root of the sum of their squares, times
    `COVERAGE`. NaN for none."""
    if not contributions:
        return math.nan
    return COVERAGE * math.sqrt(sum(c.standard ** 2 for c in contributions))


def uncertainty_db(percent: float) -> float:
    """An uncertainty in percent on the acceleration in dB on the ASD,
    taken on the side it is larger, below: `−20 log10(1 − percent/100)`.
    Infinite at 100 percent or more."""
    if not percent < 100.0:
        return math.inf
    return -20.0 * math.log10(1.0 - percent / 100.0)


def uncertainty_budget(requirement: TestRequirement,
                       channels: Sequence[str] | None = None
                       ) -> tuple[Budget, ...]:
    """Each control channel's uncertainty against one third of the
    requirement's tolerance.

    The combined uncertainty is `combined_percent` of the channel's
    contributions, on the acceleration; on the ASD it is the same
    decibels (`uncertainty_db`) as a power ratio less one. It is held
    against one third of the tightest band edge of the requirement, in
    dB (composite and individual bands alike, since every control
    channel feeds the composite), and, with an overall rms tolerance,
    against one third of it in percent on the acceleration. Met when it
    is within both; a channel with nothing entered is not judged.

    Parameters
    ----------
    requirement : TestRequirement
        The requirement, with its contributions.
    channels : sequence of str, optional
        The control channels, by response DOF: those `evaluate` found
        for the run. Left out, those the requirement names, else those
        it holds contributions for.

    Returns
    -------
    tuple of Budget
        One per channel, in order.
    """
    if channels is None:
        channels = requirement.channels or tuple(requirement.uncertainty)
    edges = [abs(edge) for band in (*requirement.composite_bands,
                                    *requirement.individual_bands)
             for edge in (band.below_db, band.above_db)
             if math.isfinite(edge)]
    # a band has a finite edge on one side at least (`Band`)
    band_db = min(edges)
    band_limit = SHARE * band_db
    grms_limit = (None if requirement.grms_percent is None
                  else SHARE * requirement.grms_percent)
    out = []
    for channel in channels:
        channel = str(channel)
        held = requirement.uncertainty.get(channel, ())
        percent = combined_percent(held)
        db = uncertainty_db(percent) if held else math.nan
        asd = 100.0 * (10.0 ** (db / 10.0) - 1.0) if held else math.nan
        missed = []
        if not held:
            state, words = NOT_JUDGED, 'not judged: no contributions are entered'
        else:
            if not db <= band_limit * (1.0 + 1e-12):
                missed.append(f'{_db(db)} dB over one third of the '
                              f'{band_db:g} dB band edge, {band_limit:.2g} dB')
            if grms_limit is not None and \
                    not percent <= grms_limit * (1.0 + 1e-12):
                missed.append(f'{percent:.2g} % over one third of the '
                              f'{requirement.grms_percent:g} % rms '
                              f'tolerance, {grms_limit:.2g} %')
            state = NOT_MET if missed else MET
            words = '; '.join(missed)
        out.append(Budget(channel, held, percent, asd, db, band_limit,
                          band_db, grms_limit, state, words))
    return tuple(out)


def _db(value: float) -> str:
    return 'infinite' if math.isinf(value) else f'{value:.2g}'
