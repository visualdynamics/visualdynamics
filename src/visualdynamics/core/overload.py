"""Each channel of a record checked against its range.

Every input measures over a window, its range, and a signal larger than
the window is clipped: each sample beyond the edge is written as the
edge, and the tops of the waveform come out flat. The averaged spectrum
does not show it. Clipping takes a little energy off the few peaks that
cross the edge and spreads it across the band as harmonics, and over
dozens of frames the result is a spectrum a fraction of a decibel low
in band with a slightly raised floor above it. The kurtosis barely
moves either: a Gaussian record clipped at 2.8 standard deviations
reads about 2.86, inside the kurtosis bars' band of 2 to 4. The
overload is in the time history and nowhere else, so this reads it
there, channel by channel, against each channel's own range.

The reading of a record against a range is `core.time_statistics`'s,
and this does not read it again: the samples on the rail (within
`RAIL_MARGIN` of the range, judged against `RAIL_SAMPLES`), the crest
factor, the peak and the kurtosis are that reading's, over the same span
the kurtosis bars read (`core.kurtosis.analyzed_span`). What this adds
is what a range makes of them:

- The range itself, read from the channel table for every channel at
  once, where the Statistics reading takes one range typed for one
  channel.
- The samples on the rail as a rate per second, beside the count and
  the share.
- The headroom of the range over the largest excursion, in dB:
  20 log10(range / peak). At or below 0 dB the range was reached.
- The largest excursion scaled to the next level of the test, a
  stated step in dB up (6 dB, about twice the amplitude, by default),
  and its headroom there. A test is run up in steps, and a range that
  holds at one level is checked against the next level's peak before
  the step, not found wanting after it. The scaled peak is a starting
  point: structures at full level often do not behave as they did at
  the low level, so it is never a ceiling.
- Samples beyond the range: further out than the range and its margin.
  A converter cannot write a sample past its own range, so a record
  that holds one was not taken through the range the table states: the
  Range or the Sensitivity in the table is not the one the record was
  made with, and every other number here is against the wrong range.

The range comes from the channel table. Its Range column is the input's
range in volts and its Sensitivity column the chain's sensitivity in mV
per the unit beside it, so the range in that unit is
``range * 1000 / sensitivity``, and in SI it is that times the unit's
scale. `table_full_scales` reads it for each channel of a time history
by the same match the role pairing uses (`ChannelTable.identities`). A
channel the table does not name, or names with no sensitivity, or in a
unit of some other quantity, has no range, and says why rather than
being judged against a guess. A range is a size either way, so a unit
with an offset (a temperature) converts by its scale alone.

`overload_check` reads every channel of a `TimeHistory` (also
`TimeHistory.overload_check`), `channel_overload` one plain array; each
channel's result is a `ChannelOverload`, which carries its
`TimeStatistics`, and the whole an `OverloadCheck`.

What is judged: the samples beyond the range (none), the samples on the
rail (the time statistics' own judgement of them), and the peak scaled
to the next level against the range. No headroom is asked for beyond
that: how much a range should carry over its predicted peak is a choice
a test procedure makes, and the procedures put no single number on it,
so the headroom is reported and the step is set where the next level
is.

References
----------
Range, clipping and the way an average hides it are covered in the
texts on vibration data acquisition; these are where they are set out.

1. Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
   Measurement Procedures*, 4th ed. Wiley. Data acquisition and
   conversion, the clipping of a signal that exceeds the converter's
   range, and why a clipped record is detected in its time history and
   probability density rather than its spectrum.
2. Piersol, A. G., & Paez, T. L. (Eds.) (2010). *Harris' Shock and
   Vibration Handbook*, 6th ed. McGraw-Hill. The setting of each
   channel's range from its expected peak, the crest factor of random
   and sinusoidal signals, and the overload of a transducer ahead of
   the input.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .kurtosis import analyzed_span
from .time_statistics import (
    RAIL_MARGIN,
    RAIL_SAMPLES,
    TimeStatistics,
    record_statistics,
    time_statistics,
)

if TYPE_CHECKING:                                    # pragma: no cover
    from .channel_table import ChannelTable
    from .data import TimeHistory

# Asked for on 2026-10-23, from a worked example: a Gaussian record of
# 8.88 grms (a largest peak of 37 g) through a 25 g range reads 287
# samples on the rail, 0.47 %, 24 a second, a crest factor of 2.83 and a
# kurtosis of 2.86, which the kurtosis bars' band of 2 to 4 passes. The
# channel table's Range column was read by nothing. At minus 6 dB the
# same record peaks at 18.49 g, 2.6 dB under the range, and scaled to
# full level that is 37 g against 25: the check that would have caught
# it before the step.

#: the step to the next level, in dB: minus 6 dB to full level doubles
#: the amplitude
STEP_DB = 6.0

#: every judged item, by key, and the name a table or the words give it
OVERLOADS: dict[str, str] = {
    'beyond': 'Beyond the range',
    'rail': 'On the rail',
    'next': 'Reaches its range at the next level',
}


def range_in_unit(range_volts: float, sensitivity: float) -> float:
    """The range in the channel's own unit: an input range in volts
    over a sensitivity in mV per unit, ``range * 1000 / sensitivity``.

    A 10 V input on a 100 mV/g chain reads to 100 g. NaN when the
    sensitivity is not a positive number, since there is then no range
    to speak of.
    """
    sensitivity = float(sensitivity)
    if not (np.isfinite(sensitivity) and sensitivity > 0.0):
        return float('nan')
    return float(range_volts) * 1000.0 / sensitivity


def table_full_scales(table: ChannelTable, history: TimeHistory
                      ) -> tuple[np.ndarray, list[str]]:
    """Each channel's range in SI from a channel table, and why where
    there is none.

    The table's row for a channel is the one that names it
    (`ChannelTable.identities`): its DOF and its channel type, or its
    DOF alone when the type is blank and the history has one channel
    there. The range is the row's Range in volts over its Sensitivity
    in mV per the row's unit (`range_in_unit`), put into SI by the
    unit's scale.

    Parameters
    ----------
    table : ChannelTable
        The channel table describing the recording.
    history : TimeHistory
        The recording.

    Returns
    -------
    (numpy.ndarray, list of str)
        One range per channel of `history`, in SI, NaN where there is
        none; and one reason per channel, '' where the range was read.
    """
    from ..units import UNKNOWN, UnitError, si_transform

    rows = table.identities(history)
    ranges = table.ranges()
    sensitivities = table.sensitivities()
    units = [str(unit).strip() for unit in table.frame['unit']]
    count = history.num_records
    scales = np.full(count, np.nan)
    reasons = [''] * count
    for i in range(count):
        identity = (history.response_dof[i], history.ordinate_dim[i])
        row = next((k for k, named in enumerate(rows) if named == identity),
                   None)
        if row is None:
            reasons[i] = 'not in the channel table'
            continue
        in_unit = range_in_unit(ranges[row], sensitivities[row])
        if not np.isfinite(in_unit):
            reasons[i] = 'no sensitivity in the channel table'
            continue
        if not units[row]:
            reasons[i] = 'no unit in the channel table'
            continue
        dimension = history.ordinate_dim[i]
        if dimension == UNKNOWN:
            # raw numbers of no stated quantity: a range in a unit says
            # nothing about them
            reasons[i] = 'the channel states no quantity to hold a range in'
            continue
        try:
            scale, _offset = si_transform(units[row], dimension)
        except UnitError:
            reasons[i] = f'the table\'s unit {units[row]!r} is not {dimension}'
            continue
        # a range is a size either way: the unit's scale, no offset
        scales[i] = in_unit * scale
    return scales, reasons


@dataclass(frozen=True, eq=False)
class ChannelOverload:
    """One channel's record against its range.

    `full_scale` and the values read from `statistics` are in the
    record's own units (SI for a channel of a `TimeHistory`). Without a
    range, `full_scale` is NaN, `reason` says why, and nothing is
    judged; without a reading at all (a channel with fewer than two
    finite samples), `statistics` is None as well.
    """

    #: the channel's index in its record, and its label
    record: int
    label: str
    #: the range either way, NaN where there is none
    full_scale: float
    #: the time-domain reading the numbers come from, or None
    statistics: TimeStatistics | None
    #: the step to the next level, in dB
    step_db: float = STEP_DB
    #: why there is no range, or no reading; '' when there is both
    reason: str = ''
    #: the samples further out than the range and its margin: none
    #: unless the record was not taken through this range
    beyond: int = 0

    @property
    def has_range(self) -> bool:
        """Whether the channel was read against a range."""
        return self.statistics is not None and np.isfinite(self.full_scale)

    @property
    def peak(self) -> float:
        """The largest excursion either way, a size: NaN without a
        reading."""
        if self.statistics is None:
            return float('nan')
        return abs(self.statistics.peak)

    @property
    def rms(self) -> float:
        """The rms over the span read; NaN without a reading."""
        return (float('nan') if self.statistics is None
                else self.statistics.rms)

    @property
    def crest(self) -> float:
        """The crest factor: the largest excursion over the rms."""
        return (float('nan') if self.statistics is None
                else self.statistics.crest)

    @property
    def rail(self) -> int | None:
        """The samples on the rail; None without a range."""
        return self.statistics.rail if self.has_range else None

    @property
    def rail_share(self) -> float | None:
        """The samples on the rail as a share of those read; None
        without a range."""
        return self.statistics.rail_share if self.has_range else None

    @property
    def rail_rate(self) -> float | None:
        """The samples on the rail per second of the span read; None
        without a range."""
        if not self.has_range:
            return None
        return self.statistics.rail / self.statistics.record_length

    @property
    def headroom_db(self) -> float:
        """The range over the largest excursion in dB, 20 log10 of the
        time statistics' `headroom` ratio. At or below 0 the range was
        reached; infinite for a record that never leaves zero; NaN
        without a range."""
        if not self.has_range:
            return float('nan')
        ratio = self.statistics.headroom
        if ratio is None:
            return math.inf
        return 20.0 * math.log10(ratio)

    @property
    def next_peak(self) -> float:
        """The largest excursion scaled up by `step_db`, a size: what
        the next level would put on the channel if the structure
        behaves as it did at this one."""
        return self.peak * 10.0 ** (self.step_db / 20.0)

    @property
    def next_headroom_db(self) -> float:
        """The headroom at the next level, in dB: `headroom_db` less the
        step."""
        return self.headroom_db - self.step_db

    def flags(self) -> dict[str, bool]:
        """Each judged item, by `OVERLOADS` key, and whether it is
        flagged; empty for a channel with no range. The rail is the time
        statistics' own judgement (`TimeStatistics.flags`). The next
        level is judged only where the range has not already been
        reached, so a clipped channel is flagged for its clip and not
        again for a scaled peak that predicts nothing."""
        if not self.has_range:
            return {}
        stats = self.statistics
        reached = bool(stats.rail)
        # the scaled peak within the margin of the range is on the rail
        # at the next level
        edge = (1.0 - stats.margin) * self.full_scale
        return {
            'beyond': self.beyond > 0,
            'rail': stats.flags().get('rail', False),
            'next': not reached and self.next_peak >= edge,
        }

    @property
    def flagged(self) -> list[str]:
        """The keys flagged, in `OVERLOADS` order."""
        found = self.flags()
        return [key for key in OVERLOADS if found.get(key)]

    def describe(self, unit: str = '', scale: float = 1.0) -> str:
        """The channel's judgement in words.

        `scale` divides the values (SI for a channel of a time history)
        into `unit` for the words, as a unit system's transform scale
        does: 9.80665 and 'g' state an acceleration in g.
        """
        after = f' {unit}' if unit else ''
        if self.statistics is None or not np.isfinite(self.full_scale):
            return f'{self.label}: no range ({self.reason}).'
        stats = self.statistics
        if self.beyond:
            return (f'{self.label}: {self.beyond} sample'
                    f'{"s" * (self.beyond != 1)} beyond its range of '
                    f'{self.full_scale / scale:.4g}{after}, the largest '
                    f'{self.peak / scale:.4g}{after}: the range or the '
                    'sensitivity in the table is not the one the record '
                    'was taken with.')
        if 'rail' in self.flagged or stats.rail:
            return (f'{self.label}: {stats.rail} sample'
                    f'{"s" * (stats.rail != 1)} on the rail '
                    f'({100.0 * (self.rail_share or 0.0):.2g} %, '
                    f'{self.rail_rate:.3g} a second), crest factor '
                    f'{self.crest:.2f}.')
        if 'next' in self.flagged:
            return (f'{self.label}: {self.headroom_db:.1f} dB of headroom; '
                    f'+{self.step_db:g} dB puts its peak at '
                    f'{self.next_peak / scale:.4g}{after} against a range '
                    f'of {self.full_scale / scale:.4g}{after}.')
        return (f'{self.label}: {self.headroom_db:.1f} dB of headroom, '
                f'{self.next_headroom_db:.1f} dB at +{self.step_db:g} dB.')


@dataclass(frozen=True, eq=False)
class OverloadCheck:
    """Every channel read against its range, and the judgement."""

    #: one per channel read, in the order asked for
    channels: tuple[ChannelOverload, ...]
    step_db: float = STEP_DB
    margin: float = RAIL_MARGIN

    @property
    def checked(self) -> list[ChannelOverload]:
        """The channels read against a range."""
        return [channel for channel in self.channels if channel.has_range]

    @property
    def unranged(self) -> list[ChannelOverload]:
        """The channels with no range, each saying why."""
        return [channel for channel in self.channels
                if not channel.has_range]

    def with_flag(self, key: str) -> list[ChannelOverload]:
        """The channels flagged for `key` (an `OVERLOADS` key)."""
        if key not in OVERLOADS:
            raise KeyError(key)
        return [channel for channel in self.channels
                if key in channel.flagged]

    @property
    def flagged(self) -> list[ChannelOverload]:
        """The channels flagged for anything."""
        return [channel for channel in self.channels if channel.flagged]

    @property
    def worst(self) -> ChannelOverload | None:
        """The channel that most wants looking at: the most samples on
        the rail, then the least headroom at the next level. None when
        no channel has a range."""
        checked = self.checked
        if not checked:
            return None
        return max(checked, key=lambda channel: (channel.rail or 0,
                                                 -channel.next_headroom_db))

    def describe(self) -> str:
        """The judgement in words: which channels are on the rail,
        which reach their range at the next level, which hold samples
        beyond it, and how many had no range."""
        step = f'+{self.step_db:g} dB'
        phrases = {'beyond': ('beyond its range', 'beyond their ranges'),
                   'rail': ('on the rail', 'on the rail'),
                   'next': (f'reaches its range at {step}',
                            f'reach their ranges at {step}')}
        parts = []
        for key, (one, many) in phrases.items():
            found = self.with_flag(key)
            if found:
                names = ', '.join(channel.label for channel in found)
                parts.append(f'{len(found)} {one if len(found) == 1 else many}'
                             f' ({names})')
        checked = len(self.checked)
        if not parts:
            words = (f'All {checked} channel{"s" * (checked != 1)} inside '
                     f'their ranges, at {step} as well.'
                     if checked else 'No channel has a range.')
        else:
            words = (f'Of {checked} channel{"s" * (checked != 1)}: '
                     + '; '.join(parts) + '.')
        missing = len(self.unranged)
        if missing and checked:
            words += (f' {missing} channel{"s" * (missing != 1)} '
                      f'with no range.')
        return words


def _beyond(values: np.ndarray, spans: Sequence[tuple[int, int]] | None,
            full_scale: float, margin: float) -> int:
    """The finite samples of `values` within `spans` further out than
    the range and its margin."""
    if not np.isfinite(full_scale):
        return 0
    limit = (1.0 + margin) * full_scale
    pieces = ([values] if spans is None
              else [values[max(int(low), 0):int(high)] for low, high in spans])
    count = 0
    for piece in pieces:
        with np.errstate(invalid='ignore'):
            count += int(np.count_nonzero(np.abs(piece) > limit))
    return count


def channel_overload(values: ArrayLike, sample_rate: float,
                     full_scale: float | None, *,
                     step_db: float = STEP_DB,
                     margin: float = RAIL_MARGIN,
                     rail_samples: int = RAIL_SAMPLES,
                     label: str = '', record: int = 0,
                     start: float = 0.0,
                     spans: Sequence[tuple[int, int]] | None = None
                     ) -> ChannelOverload:
    """Read one record against its range.

    Parameters
    ----------
    values : array_like
        The record, one channel, evenly sampled.
    sample_rate : float
        Samples per second.
    full_scale : float or None
        The range either way, in the units of `values`. None or NaN
        reads the record with no range: nothing is judged.
    step_db : float, default `STEP_DB`
        The step to the next level, in dB.
    margin : float, default `core.time_statistics.RAIL_MARGIN`
        How close to the range a sample counts as on it, as a share.
    rail_samples : int, default `core.time_statistics.RAIL_SAMPLES`
        The most samples allowed on the rail.
    label : str, optional
        What the channel is called in the words.
    record : int, default 0
        The channel's index, carried for the caller.
    start, spans
        As `core.time_statistics.time_statistics` takes them.

    Returns
    -------
    ChannelOverload
        The numbers and the flags.

    Raises
    ------
    ValueError
        If the range is not above zero, or as
        `core.time_statistics.time_statistics` raises.
    """
    scale = (float('nan') if full_scale is None else float(full_scale))
    if np.isfinite(scale) and not scale > 0.0:
        raise ValueError('the range is how far either way the channel '
                         'reads, and has to be above zero')
    data = np.real(np.asarray(values)).astype(float, copy=False).ravel()
    stats = time_statistics(data, sample_rate, start=start, spans=spans,
                            full_scale=scale if np.isfinite(scale) else None,
                            margin=margin, rail_samples=rail_samples)
    return ChannelOverload(record=int(record), label=str(label),
                           full_scale=scale, statistics=stats,
                           step_db=float(step_db),
                           reason='' if np.isfinite(scale) else 'no range given',
                           beyond=_beyond(data, spans, scale, float(margin)))


def overload_check(history: TimeHistory,
                   ranges: ChannelTable | ArrayLike | float, *,
                   records: Sequence[int] | None = None,
                   step_db: float = STEP_DB,
                   margin: float = RAIL_MARGIN,
                   rail_samples: int = RAIL_SAMPLES) -> OverloadCheck:
    """Read each channel of a time history against its range.

    Over the span `core.kurtosis.analyzed_span` names, as the kurtosis
    bars and the Statistics reading read it, so a channel's rail count
    here is that reading's for the same range and margin.

    Parameters
    ----------
    history : TimeHistory
        The record.
    ranges : ChannelTable, array_like or float
        Where the ranges come from: the channel table describing the
        record (`table_full_scales` reads it), or the ranges themselves
        in SI, one per channel (NaN or None for a channel with none) or
        one for every channel. A 25 g range on an acceleration channel
        is ``25 * 9.80665``.
    records : sequence of int, optional
        Which channels, counting from 0; every channel by default.
    step_db : float, default `STEP_DB`
        The step to the next level, in dB.
    margin : float, default `core.time_statistics.RAIL_MARGIN`
        How close to the range a sample counts as on it, as a share.
    rail_samples : int, default `core.time_statistics.RAIL_SAMPLES`
        The most samples allowed on the rail.

    Returns
    -------
    OverloadCheck
        Every channel's numbers and flags, and the judgement.

    Raises
    ------
    ValueError
        If a channel named is not in the record, the ranges given are
        not one per channel, a range is zero or negative, the step is
        not finite, or the margin is not a share.
    """
    from .channel_table import ChannelTable

    count = history.num_records
    if not np.isfinite(float(step_db)):
        raise ValueError('the step to the next level is a number of dB')
    if not 0.0 <= float(margin) < 1.0:
        raise ValueError('the margin is a share of the range, from 0 up '
                         'to 1')
    if isinstance(ranges, ChannelTable):
        scales, reasons = table_full_scales(ranges, history)
    else:
        given = np.asarray(
            [np.nan if value is None else value
             for value in np.atleast_1d(np.asarray(ranges, dtype=object))],
            dtype=float)
        if given.size == 1:
            given = np.full(count, given[0])
        if given.size != count:
            raise ValueError(f'{given.size} ranges for {count} channel'
                             f'{"s" * (count != 1)}: one each, or one for '
                             'all')
        if np.any(np.isfinite(given) & ~(given > 0.0)):
            raise ValueError('a range is how far either way a channel '
                             'reads, and has to be above zero')
        scales = given
        reasons = ['' if np.isfinite(value) else 'no range given'
                   for value in given]
    wanted = range(count) if records is None else [int(i) for i in records]
    spans, _phrase = analyzed_span(history)
    ordinate = np.asarray(history.ordinate)
    channels = []
    for i in wanted:
        if not 0 <= i < count:
            raise ValueError(f'the record has {count} channel'
                             f'{"s" * (count != 1)}; there is no channel {i}')
        scale = float(scales[i])
        known = np.isfinite(scale)
        try:
            stats = record_statistics(
                history, i, full_scale=scale if known else None,
                margin=margin, rail_samples=rail_samples)
        except ValueError as error:
            # a dead or empty channel: said, and left out of the verdict
            channels.append(ChannelOverload(
                record=i, label=history.record_label(i), full_scale=scale,
                statistics=None, step_db=float(step_db),
                reason=str(error)))
            continue
        channels.append(ChannelOverload(
            record=i, label=history.record_label(i), full_scale=scale,
            statistics=stats, step_db=float(step_db), reason=reasons[i],
            beyond=_beyond(np.real(ordinate[i]), spans, scale,
                           float(margin))))
    return OverloadCheck(channels=tuple(channels), step_db=float(step_db),
                         margin=float(margin))


def headroom_rows(check: OverloadCheck) -> list[tuple[str, float]]:
    """[(label, headroom in dB)] for each channel, NaN where there is
    no range: the rows the headroom bars draw."""
    return [(channel.label, channel.headroom_db)
            for channel in check.channels]


def notes_for(check: OverloadCheck) -> dict[int, str]:
    """{row: why} for each channel with no range: the sentence the
    headroom bars write along its row in place of a bar."""
    return {row: f'no range: {channel.reason}'
            for row, channel in enumerate(check.channels)
            if not channel.has_range}

