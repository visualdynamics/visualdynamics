"""Whether each sensor was in calibration on the test date.

The channel table keeps each channel's calibration expiration (its
Expiration column, stored as an ISO date). This module compares that
date with the dates of the test and says, channel by channel, one of
four things:

- **in date**: the sensor's calibration runs past the last day of the
  test;
- **expired**: it ran out before the test began, named with how many
  days before;
- **lapses during the test**: it was in date when the test began and
  ran out before its last day;
- **no date given**: the table holds no expiration for it, so nothing
  was judged. A blank is never read as in date.

A sensor is taken as in calibration through its expiration date: a
sensor due on the 24th may be used on the 24th and not on the 25th.

The test's dates are a `DateSpan`, its first and last day (one day for a
test run on one day). `span_of` reads them from what was entered or,
where records carry dates of their own, from those; the two must agree.
No importer keeps an acquisition date on a record today, so the date is
entered (`Project.report_test_date`); `recorded` is where a later source
of dates is passed.

What cannot be read one way is refused with the value named: a test date
not written YYYY-MM-DD, a last day before the first, or a channel table
cell that is not a date (named by its row, channel and column).

`check` returns a `CalibrationCheck`: the span, one `ChannelCalibration`
per channel, and the words for it (`summary`). A check that judged
nothing (a table with no channels, or no expiration in it at all) says
so, and never reports the channels in date.
"""

from __future__ import annotations

import datetime as _datetime
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

#: the four answers, as `ChannelCalibration.status` holds them
IN_DATE = 'in date'
EXPIRED = 'expired'
LAPSES = 'lapses during the test'
NO_DATE = 'no date given'
STATUSES = (IN_DATE, EXPIRED, LAPSES, NO_DATE)

#: the one form a typed test date is read in: the American and European
#: orders of a slashed date read each other's dates, so neither is taken
DATE_FORM = 'YYYY-MM-DD'


def parse_date(value: Any, what: str = 'test date') -> _datetime.date:
    """`value` as a date, read one way only.

    A `datetime.date` is taken as it is (a `datetime.datetime` as its
    date). A string must be an ISO date, YYYY-MM-DD, with surrounding
    blanks ignored. `what` names the value in a refusal.

    Raises
    ------
    ValueError
        For a string that is not a YYYY-MM-DD date, naming `what` and
        the value.
    TypeError
        For anything that is neither a date nor a string.
    """
    if isinstance(value, _datetime.datetime):
        return value.date()
    if isinstance(value, _datetime.date):
        return value
    if not isinstance(value, str):
        raise TypeError(f'the {what} is a date or a {DATE_FORM} string, '
                        f'not {type(value).__name__}')
    text = value.strip()
    # fromisoformat also reads '20261024' and '2026-W43-6'; a date is
    # taken here only in the form the channel table stores
    if len(text) == 10 and text[4] == text[7] == '-':
        try:
            return _datetime.date.fromisoformat(text)
        except ValueError:
            pass
    raise ValueError(f'the {what} {value!r} cannot be read one way: write '
                     f'it {DATE_FORM}')


@dataclass(frozen=True)
class DateSpan:
    """The first and last day of a test (the same day for a test run on
    one day)."""

    start: _datetime.date
    end: _datetime.date

    @property
    def days(self) -> int:
        """How many calendar days the test covers, both ends counted."""
        return (self.end - self.start).days + 1

    def __str__(self) -> str:
        if self.start == self.end:
            return self.start.isoformat()
        return f'{self.start.isoformat()} to {self.end.isoformat()}'


def span_of(start: Any = None, end: Any = None,
            recorded: Iterable[Any] = ()) -> DateSpan | None:
    """The test's dates, or None when nothing gives them.

    `start` and `end` are what was entered: the first day and, for a
    test over several days, the last (the first when omitted).
    `recorded` is the dates the records themselves carry; without an
    entry, the span runs from the earliest of them to the latest.

    Refused, naming the value: a date that is not a date or a
    YYYY-MM-DD string, a last day without a first, a last day before
    the first, and an entered span that leaves out a recorded date.

    Raises
    ------
    ValueError
        For a refusal.
    TypeError
        For a value that is neither a date nor a string.
    """
    dates = sorted(parse_date(when, 'recorded test date')
                   for when in recorded)
    if start is None or start == '':
        if end is not None and end != '':
            raise ValueError('a last test day was given without a first')
        if not dates:
            return None
        return DateSpan(dates[0], dates[-1])
    first = parse_date(start, 'test date')
    last = (first if end is None or end == ''
            else parse_date(end, 'last test day'))
    if last < first:
        raise ValueError(f'the last test day, {last.isoformat()}, is before '
                         f'the first, {first.isoformat()}')
    span = DateSpan(first, last)
    outside = [when for when in dates if not first <= when <= last]
    if outside:
        raise ValueError(f'the test date entered, {span}, does not cover '
                         f'the date a record carries, '
                         f'{outside[0].isoformat()}')
    return span


@dataclass(frozen=True)
class ChannelCalibration:
    """One channel's calibration against the test's dates.

    `row` is its row in the channel table (from 0) and `channel` its
    channel number. `key` is its identity the way a record's
    `channel_key` gives it: the DOF, the quantity measured there
    ('unknown' when the table declares none) and no hint. `name` is how
    it is written for a reader ('channel 3 (101Z+ acceleration,
    control)'). `expiration` is the table's date, or None for a blank.
    `status` is one of `STATUSES`. `days` is, for an expired sensor,
    how many days before the first test day its calibration ran out;
    for one that lapses, how many test days it does not cover; None
    otherwise.
    """

    row: int
    channel: int
    key: tuple[str, str, None]
    name: str
    expiration: _datetime.date | None
    status: str
    days: int | None = None

    @property
    def words(self) -> str:
        """The status as the report's Calibration column writes it."""
        if self.status == EXPIRED:
            return (f'expired {self.days} '
                    f'day{"s" * (self.days != 1)} before the test')
        if self.status == LAPSES:
            return (f'lapses during the test, in date through '
                    f'{self.expiration.isoformat()}')
        if self.status == NO_DATE:
            return 'no date given, not checked'
        return IN_DATE


def _name(table: Any, row: int, key: tuple[str, str, None]) -> str:
    # what a reader recognizes the sensor by: its channel number, its
    # DOF and what it measures there (a drive point has a force and an
    # acceleration at one DOF), and whether it is a control channel
    number = int(table['channel'][row])
    dof, quantity, _hint = key
    said = ' '.join(part for part in (dof, quantity if quantity != 'unknown'
                                      else '') if part)
    if bool(table.controls()[row]):
        said = f'{said}, control' if said else 'control'
    return f'channel {number} ({said})' if said else f'channel {number}'


def _key(table: Any, row: int) -> tuple[str, str, None]:
    from ..units import dimension_of

    dof = table.dof_strings()[row]
    quantity = table.types()[row]
    if not quantity:
        unit = str(table['unit'][row]).strip()
        quantity = (dimension_of(unit) if unit else None) or ''
    return (dof, quantity or 'unknown', None)


def _expiration(table: Any, row: int) -> _datetime.date | None:
    text = str(table['expiration'][row]).strip()
    if not text:
        return None
    try:
        return parse_date(text)
    except ValueError:
        raise ValueError(
            f'channel table row {row + 1} (channel '
            f'{int(table["channel"][row])}), Expiration: {text!r} is not a '
            f'date ({DATE_FORM})') from None


def judge(expiration: _datetime.date | None,
          span: DateSpan) -> tuple[str, int | None]:
    """(status, days) for one expiration against the test's dates; see
    `ChannelCalibration`."""
    if expiration is None:
        return NO_DATE, None
    if expiration < span.start:
        return EXPIRED, (span.start - expiration).days
    if expiration < span.end:
        return LAPSES, (span.end - expiration).days
    return IN_DATE, None


@dataclass(frozen=True)
class CalibrationCheck:
    """Every channel of a table against the test's dates.

    `span` is the test's dates and `channels` one `ChannelCalibration`
    per channel, in the table's order.
    """

    span: DateSpan
    channels: tuple[ChannelCalibration, ...]

    def having(self, status: str) -> tuple[ChannelCalibration, ...]:
        """The channels with `status`, in order."""
        return tuple(c for c in self.channels if c.status == status)

    @property
    def judged(self) -> int:
        """How many channels had a date to judge."""
        return sum(c.status != NO_DATE for c in self.channels)

    @property
    def lapsed(self) -> tuple[ChannelCalibration, ...]:
        """The channels out of calibration on some day of the test."""
        return tuple(c for c in self.channels
                     if c.status in (EXPIRED, LAPSES))

    def of(self, key: Sequence[Any]) -> ChannelCalibration | None:
        """The channel whose identity is `key` (a record's
        `channel_key`), or None when no channel has it.

        Raises `ValueError` when two channels of the table share the
        key, naming both: the record cannot then be matched one way.
        """
        wanted = (str(key[0]), str(key[1]) if key[1] else 'unknown', None)
        found = [c for c in self.channels if c.key == wanted]
        if len(found) > 1:
            raise ValueError(f'{found[0].name} and {found[1].name} are both '
                             f'{wanted[0]} {wanted[1]}, so a record there '
                             'cannot be matched to one sensor')
        return found[0] if found else None

    def summary(self) -> list[str]:
        """The check in words: a headline, then one line for each
        channel out of calibration on a test day, and one naming the
        channels with no date. Never 'in date' for a channel that was
        not judged."""
        total = len(self.channels)
        when = (f'the test date, {self.span}' if self.span.days == 1
                else f'the test dates, {self.span}')
        if not total:
            return [(f'Calibration on {when}: not checked, the channel '
                    'table has no channels.')]
        if not self.judged:
            return [(f'Calibration on {when}: not checked, no channel has '
                    'an expiration date.')]
        fine = len(self.having(IN_DATE))
        if fine == total:
            head = f'all {total} channels in date'
        else:
            head = f'{fine} of {total} channels in date'
        lines = [f'Calibration on {when}: {head}.']
        for found in self.having(EXPIRED):
            lines.append(f'{_capital(found.name)} expired '
                         f'{found.expiration.isoformat()}, {found.days} '
                         f'day{"s" * (found.days != 1)} before the test.')
        for found in self.having(LAPSES):
            last = ('the last test day' if found.days == 1 else
                    f'the last {found.days} test days')
            lines.append(f'{_capital(found.name)} lapses during the test: '
                         f'in date through {found.expiration.isoformat()}, '
                         f'not on {last}.')
        missing = self.having(NO_DATE)
        if missing:
            names = ', '.join(found.name for found in missing)
            lines.append(f'No expiration date given, so not checked: '
                         f'{names}.')
        return lines


def _capital(text: str) -> str:
    # a channel's name opening a sentence
    return text[:1].upper() + text[1:]


def check(table: Any, span: DateSpan) -> CalibrationCheck:
    """Each channel of `table` (a `ChannelTable`) against `span`.

    Raises
    ------
    ValueError
        When an Expiration cell is not a date, naming its row, channel
        and column.
    TypeError
        When `table` is not a channel table or `span` not a `DateSpan`.
    """
    from .channel_table import ChannelTable

    if not isinstance(table, ChannelTable):
        raise TypeError(f'calibration is read from a channel table, not '
                        f'{type(table).__name__}')
    if not isinstance(span, DateSpan):
        raise TypeError(f'the test dates are a DateSpan, not '
                        f'{type(span).__name__}')
    out = []
    for row in range(table.num_channels):
        expiration = _expiration(table, row)
        status, days = judge(expiration, span)
        key = _key(table, row)
        out.append(ChannelCalibration(row, int(table['channel'][row]), key,
                                      _name(table, row, key), expiration,
                                      status, days))
    return CalibrationCheck(span, tuple(out))
