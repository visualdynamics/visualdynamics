"""Runs: which run of a test a time history is, and what its record shows.

A test is a dozen runs on three axes: a low level and a step-up, the
full-level run, a signature before and after, and the same again on
the next axis. A time history is one of those runs, and a report of the
test has to say which. A record does not say it by itself, so a time
history carries a run identity (`RunIdentity`): the test axis, the run
number, the level against full in decibels, the date and time the run
started, and a free comment.

Each field comes from somewhere, and says where (`RunIdentity.sources`):

- the name of the file the record was imported from, and a Rattlesnake
  run's environment names (`read_name`): `Z_run07_-6dB_2026-10-11T111105`
  is axis Z, run 7, -6 dB, started 11:11:05 on 11 October 2026;
- what a person entered (`entered`), which stands over anything read;
- the level the comparison with a specification already detects
  (`visualdynamics.report.runs` reads it through
  `compliance.read_scale_db`), when nothing else gives the level.

What cannot be read one way is refused, never guessed: a name with two
axes in it, a level written without a sign ('6dB' is 6 dB below full
or above it), two readings of the level that disagree. A refused field
is left unread, with the reason (`RunIdentity.refused`), until a
person enters it.

What a record shows about its run is read from the running RMS of its
control channels (`read_levels`): how long the run was at level, and
the interruptions in it, a stretch mid-run where the running RMS fell
well below the run's level (to zero, when the controller stopped). An
interruption is listed for the author to explain; nothing here says
why it happened. `order` puts runs in the order they were run, or
refuses when the identities cannot say it.

Every number here is read off the record as it stands. The outcome of
a run and the explanation of an interruption are the author's, or a run
log's.
"""

from __future__ import annotations

import datetime as _dt
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: the fields of a run identity, in the order they are shown
FIELDS = ('axis', 'run', 'level_db', 'start', 'comment')

#: what each field is called in a message or a column heading
FIELD_NAMES = {'axis': 'axis', 'run': 'run number', 'level_db': 'level',
               'start': 'start', 'comment': 'comment'}

#: where a field came from
FROM_NAME = 'file name'
FROM_ENVIRONMENT = 'environment name'
ENTERED = 'entered'

#: the running RMS is averaged over this long, in seconds. Long enough
#: that broadband random scatters by a few percent in mean square, short
#: enough to find a one-second stop
WINDOW_S = 0.5

#: an interruption reaches at least this far below the run's level
DROP_DB = 6.0

#: and lasts at least this long, in seconds, measured where the running
#: mean square crosses half the run's level
MIN_S = 1.0

#: a running RMS this far below the run's level is read as zero: a
#: stopped controller writes noise floor or exact zeros, and neither
#: is a level
ZERO_DB = -120.0


@dataclass(frozen=True)
class RunIdentity:
    """Which run of a test a time history is.

    `axis` is the test axis ('X', 'Y', 'Z', or a lab's own name for
    it), `run` the run's identifier ('7'), `level_db` the level against
    full in decibels (0 for full level, -6 for a quarter of the power),
    `start` the date, or the date and time, the run started, in ISO
    8601 ('2026-10-11T11:11:05'), and `comment` a free note. An empty
    field (None for the level) is not known.

    `sources` says where each known field came from: a file name, an
    environment name, an entry, or a comparison. `refused` names each
    field that was read two ways, or read in a way that does not settle
    it, with the reason; such a field stays unknown until it is entered.
    """

    axis: str = ''
    run: str = ''
    level_db: float | None = None
    start: str = ''
    comment: str = ''
    sources: tuple[tuple[str, str], ...] = ()
    refused: tuple[tuple[str, str], ...] = ()

    def value(self, name: str) -> Any:
        """The value of field `name`: '' or None when it is not known."""
        return getattr(self, name)

    def source(self, name: str) -> str:
        """Where field `name` came from, or '' when it is not known."""
        return dict(self.sources).get(name, '')

    def reason(self, name: str) -> str:
        """Why field `name` was refused, or '' when it was not."""
        return dict(self.refused).get(name, '')

    @property
    def known(self) -> bool:
        """Whether anything identifies the run: an axis, a run, a level
        or a start (a comment alone does not)."""
        return bool(self.axis or self.run or self.level_db is not None
                    or self.start)

    def label(self) -> str:
        """'run 7, Z axis, -6 dB, 2026-10-11 11:11:05', with the parts
        that are not known left out; '' when none is known."""
        parts = []
        if self.run:
            parts.append(f'run {self.run}')
        if self.axis:
            parts.append(f'{self.axis} axis')
        if self.level_db is not None:
            parts.append(level_text(self.level_db))
        if self.start:
            parts.append(self.start.replace('T', ' '))
        return ', '.join(parts)

    def when(self) -> _dt.datetime | None:
        """The start as a date and time (midnight for a date alone), or
        None when it is not known."""
        return parse_start(self.start) if self.start else None

    def to_json(self) -> dict[str, Any]:
        """The known fields, where each came from, and what was refused
        and why, for the project file. Empty when nothing is known or
        refused."""
        out: dict[str, Any] = {name: self.value(name) for name in FIELDS
                               if self.value(name) not in ('', None)}
        if self.sources:
            out['sources'] = {name: where for name, where in self.sources
                              if name in out}
        if self.refused:
            out['refused'] = dict(self.refused)
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any],
                  record: str = '') -> RunIdentity:
        """The identity `to_json` wrote. Raises `ValueError` or
        `TypeError`, naming `record`, when a stored field is not one a run
        identity can hold."""
        if not isinstance(data, Mapping):
            raise TypeError(f'{record}: a run identity is a mapping, not '
                             f'{type(data).__name__}')
        values = entered(record, **{name: data[name] for name in FIELDS
                                    if name in data})
        sources = data.get('sources', {}) or {}
        if not isinstance(sources, Mapping):
            raise TypeError(f'{record}: the sources of a run identity are a '
                             'mapping of field to where it came from')
        refused = data.get('refused', {}) or {}
        if not isinstance(refused, Mapping) or any(
                name not in FIELDS for name in refused):
            raise ValueError(f'{record}: the refusals of a run identity are a '
                             'mapping of field to why it was not read')
        return cls(**{name: value for name, value in values.items()
                      if value is not None},
                   sources=tuple((name, str(sources.get(name, ENTERED)))
                                 for name in FIELDS
                                 if values.get(name) not in (None, '')),
                   refused=tuple((name, str(refused[name]))
                                 for name in FIELDS if name in refused))

    def with_values(self, values: Mapping[str, Any],
                    where: str) -> RunIdentity:
        """A copy with `values` set, each from `where`. A value of None
        (or '' for a text field) clears the field and its source. A
        field set here is no longer refused."""
        current = {name: self.value(name) for name in FIELDS}
        sources = dict(self.sources)
        refused = dict(self.refused)
        for name, value in values.items():
            if name not in FIELDS:
                raise ValueError(f'{name!r} is not a field of a run identity; '
                                 f'the fields are: {", ".join(FIELDS)}')
            cleared = value is None or value == ''
            current[name] = (None if name == 'level_db' else '') \
                if cleared else value
            if cleared:
                sources.pop(name, None)
            else:
                sources[name] = where
                refused.pop(name, None)
        return RunIdentity(**current,
                           sources=tuple((name, sources[name])
                                         for name in FIELDS if name in sources),
                           refused=tuple((name, refused[name])
                                         for name in FIELDS if name in refused))


def level_text(level_db: float) -> str:
    """'0 dB' for full level, otherwise the signed level: '-6 dB',
    '+3 dB', '-4.5 dB'."""
    if level_db == 0:
        return '0 dB'
    return f'{level_db:+g} dB'


def parse_start(text: str) -> _dt.datetime:
    """`text` as a date and time, a date alone read as its midnight.

    ISO 8601, as `datetime.fromisoformat` reads it ('2026-10-11',
    '2026-10-11T11:11', '2026-10-11 11:11:05'). A time with a zone is
    turned to UTC and the zone dropped, so starts compare. Raises
    `ValueError` for anything else.
    """
    when = _dt.datetime.fromisoformat(str(text).strip())
    if when.tzinfo is not None:
        when = when.astimezone(_dt.UTC).replace(tzinfo=None)
    return when


# ---- reading a name -----------------------------------------------------

_EDGE = r'(?<![a-z0-9])'
_AXIS = re.compile(_EDGE + r'(?:axis[-_ ]?)?([xyz])(?:[-_ ]?axis)?(?![a-z0-9])',
                   re.IGNORECASE)
_RUN = re.compile(_EDGE + r'(?:run|r)[-_ ]?(\d+)(?![0-9])', re.IGNORECASE)
_LEVEL = re.compile(r'(?<![a-z0-9.])(minus|plus|m|p|[+-])?[-_ ]?'
                    r'(\d+(?:\.\d+)?)[-_ ]?db(?![a-z])', re.IGNORECASE)
_FULL = re.compile(_EDGE + r'(full[-_ ]?level|full|fl)(?![a-z0-9])',
                   re.IGNORECASE)
_START = re.compile(r'(?<!\d)(\d{4})-?(\d{2})-?(\d{2})'
                    r'(?:[T_ -]?(\d{2})[:.h-]?(\d{2})(?:[:.m-]?(\d{2}))?)?'
                    r'(?!\d)', re.IGNORECASE)

_WAYS = {2: 'two', 3: 'three', 4: 'four'}

#: a year a test could have been run in; an eight-digit serial number
#: outside it is not read as a date
_YEARS = (1980, 2200)


#: a file's extension: a letter and a few more, so the '.5dB' of a name
#: ending '-4.5dB' is not taken for one
_EXTENSION = re.compile(r'\.[a-z][a-z0-9]{0,5}$', re.IGNORECASE)


def _stem(name: str) -> str:
    base = os.path.basename(str(name).replace('\\', '/'))
    found = _EXTENSION.search(base)
    if found:
        return base[:found.start()]
    return base


def _levels(stem: str) -> tuple[set[float], list[str]]:
    found: set[float] = set()
    unsigned: list[str] = []
    for match in _LEVEL.finditer(stem):
        sign, number = match.group(1), float(match.group(2))
        if number == 0:
            found.add(0.0)
        elif sign is None:
            unsigned.append(match.group(0).strip('-_ '))
        elif sign.lower() in ('-', 'm', 'minus'):
            found.add(-number)
        else:
            found.add(number)
    if _FULL.search(stem):
        found.add(0.0)
    return found, unsigned


def _starts(stem: str) -> set[str]:
    found = set()
    for match in _START.finditer(stem):
        year, month, day, hour, minute, second = match.groups()
        if not _YEARS[0] <= int(year) <= _YEARS[1]:
            continue
        try:
            if hour is None:
                found.add(_dt.date(int(year), int(month), int(day))
                          .isoformat())
            else:
                found.add(_dt.datetime(
                    int(year), int(month), int(day), int(hour), int(minute),
                    int(second or 0)).isoformat())
        except ValueError:
            # not a calendar date, so not a date: a serial number
            continue
    # a date alone that a date and time also gives says nothing more
    dated = {when[:10] for when in found if 'T' in when}
    return {when for when in found if 'T' in when or when not in dated}


def read_name(*names: str) -> tuple[dict[str, Any], dict[str, str]]:
    """The run identity a file's name, or several names, give.

    Each name is read for an axis (a lone X, Y or Z, or 'Xaxis',
    'axis-Z'), a run ('run7', 'run_07', 'R7'), a level ('-6dB', 'm6dB',
    'minus6dB', '+3dB', '0dB', 'full', 'FL'), and a start (a date as
    '2026-10-11' or '20261011', with a time as 'T111105', '_1111' or
    '-11-11-05' after it). A directory and an extension are left off.

    A field read the same way twice is read once. Refused, with the
    reason, and so not read: a field read two different ways, in one
    name or across the names, and a level written without a sign other
    than 0 dB ('6dB' could be below full or above it).

    Parameters
    ----------
    *names : str
        The names to read, such as a file's name and a Rattlesnake
        run's environment names.

    Returns
    -------
    tuple of (dict, dict)
        The fields read, by name ('axis', 'run', 'level_db', 'start'),
        and the fields refused, each with its reason.
    """
    axes: set[str] = set()
    runs: set[str] = set()
    levels: set[float] = set()
    unsigned: list[str] = []
    starts: set[str] = set()
    for name in names:
        stem = _stem(name)
        axes |= {match.group(1).upper() for match in _AXIS.finditer(stem)}
        runs |= {str(int(match.group(1))) for match in _RUN.finditer(stem)}
        found, loose = _levels(stem)
        levels |= found
        unsigned += loose
        starts |= _starts(stem)
    values: dict[str, Any] = {}
    refused: dict[str, str] = {}

    def settle(field_name: str, found: set, shown) -> None:
        if len(found) == 1:
            values[field_name] = next(iter(found))
        elif found:
            readings = ', '.join(shown(value) for value in sorted(found))
            ways = _WAYS.get(len(found), str(len(found)))
            refused[field_name] = (f'the {FIELD_NAMES[field_name]} is read '
                                   f'{ways} ways ({readings})')

    settle('axis', axes, str)
    settle('run', runs, str)
    if unsigned:
        refused['level_db'] = (
            f'{unsigned[0]!r} has no sign: a level against full is below it '
            'or above it')
    else:
        settle('level_db', levels, level_text)
    settle('start', starts, lambda when: when.replace('T', ' '))
    return values, refused


def entered(record: str, **fields: Any) -> dict[str, Any]:
    """Entered fields, checked and put in the form a run identity holds.

    An empty string (or None) clears a field, which then reads as None
    in the result. `axis`, `run` and `comment` are text, and a run may
    be given as a whole number; `axis` and `run` are one line, and an
    axis named by one letter is upper-cased. `level_db` is a finite
    number of decibels. `start` is ISO 8601 text, a `datetime` or a
    `date`, and is kept as ISO text.

    Parameters
    ----------
    record : str
        The time history the fields are for, named in a refusal.
    **fields
        The fields, by name: 'axis', 'run', 'level_db', 'start',
        'comment'.

    Returns
    -------
    dict
        The fields given, each checked, None where cleared.

    Raises
    ------
    ValueError
        For a field that is not a run identity's, a level that is not a
        finite number, a start that is not a date, an axis or run on
        more than one line; the message names `record` and the field.
    TypeError
        For a value of the wrong type.
    """
    out: dict[str, Any] = {}
    for name, value in fields.items():
        if name not in FIELDS:
            raise ValueError(f'{record}: {name!r} is not a field of a run '
                             f'identity; the fields are: {", ".join(FIELDS)}')
        if value is None or (isinstance(value, str) and not value.strip()):
            out[name] = None
            continue
        if name == 'level_db':
            if isinstance(value, bool) or not isinstance(
                    value, (int, float, np.integer, np.floating)):
                raise TypeError(f'{record}: the level is a number of '
                                f'decibels, not {type(value).__name__}')
            if not math.isfinite(float(value)):
                raise ValueError(f'{record}: the level is a finite number of '
                                 f'decibels, not {value!r}')
            out[name] = float(value) + 0.0
        elif name == 'start':
            if isinstance(value, (_dt.date, _dt.datetime)):
                out[name] = value.isoformat()
                continue
            if not isinstance(value, str):
                raise TypeError(f'{record}: the start is a date and time, '
                                f'not {type(value).__name__}')
            try:
                parse_start(value)
            except ValueError:
                raise ValueError(
                    f'{record}: the start {value!r} is not a date or a date '
                    'and time (ISO 8601, such as 2026-10-11T11:11:05)'
                ) from None
            out[name] = value.strip()
        elif name == 'run' and isinstance(value, (int, np.integer)) \
                and not isinstance(value, bool):
            out[name] = str(int(value))
        else:
            if not isinstance(value, str):
                raise TypeError(f'{record}: the {FIELD_NAMES[name]} is text, '
                                f'not {type(value).__name__}')
            text = value.strip()
            if name in ('axis', 'run') and '\n' in text:
                raise ValueError(f'{record}: the {FIELD_NAMES[name]} is one '
                                 f'line, not {value!r}')
            if name == 'axis' and len(text) == 1:
                text = text.upper()
            out[name] = text
    return out


# ---- ordering -----------------------------------------------------------

def _run_key(run: str) -> tuple[int, Any]:
    return (0, int(run)) if run.isdigit() else (1, run)


def order(identities: Sequence[tuple[str, RunIdentity]]) -> list[str]:
    """The records in the order their runs were run.

    By start when every run has one, a tie broken by run number. When
    some do not, by run number, if every run has one. Otherwise the
    order cannot be read from the identities, and is refused; a lone
    run needs no order.

    Parameters
    ----------
    identities : sequence of (str, RunIdentity)
        Each record's name and its run identity.

    Returns
    -------
    list of str
        The record names, first run first.

    Raises
    ------
    ValueError
        Naming the records that have neither a start nor a run number.
    """
    if len(identities) < 2:
        return [name for name, _identity in identities]
    if all(identity.start for _name, identity in identities):
        return [name for name, _identity in sorted(
            identities, key=lambda pair: (pair[1].when(),
                                          _run_key(pair[1].run)))]
    if all(identity.run for _name, identity in identities):
        return [name for name, _identity in sorted(
            identities, key=lambda pair: (_run_key(pair[1].run),
                                          pair[1].axis))]
    lost = [name for name, identity in identities
            if not identity.start and not identity.run]
    raise ValueError(
        'the order of the runs cannot be read: '
        + ', '.join(repr(name) for name in lost)
        + f' {"has" if len(lost) == 1 else "have"} no start and no run '
        'number; enter one')


# ---- reading a record ---------------------------------------------------

@dataclass(frozen=True)
class Interruption:
    """A stretch mid-run where the running RMS fell below the run's level.

    `start_s` is where it began on the record's time axis, `seconds`
    how long it lasted, both measured where the running mean square
    crossed half the run's level, and `depth_db` how far below the
    level it reached at its lowest, or None when it reached zero.
    """

    start_s: float
    seconds: float
    depth_db: float | None

    def __str__(self) -> str:
        depth = ('to zero' if self.depth_db is None
                 else f'{-self.depth_db:.0f} dB down')
        return f'at {self.start_s:.1f} s for {self.seconds:.1f} s, {depth}'


@dataclass(frozen=True)
class RecordReading:
    """What a record shows about its run (`read_levels`).

    `start_s` and `seconds` are where the record starts on its own time
    axis and how long it is. `at_level_s` is how long the running RMS
    was within 3 dB of the run's level, None when nothing was read.
    `interruptions` are the interruptions found. `channels` are the
    channels the level was read on, and `left_out` each channel left
    out with why. `not_read` says why nothing was read, '' when the
    record was read.
    """

    start_s: float
    seconds: float
    at_level_s: float | None = None
    interruptions: tuple[Interruption, ...] = ()
    channels: tuple[str, ...] = ()
    left_out: tuple[str, ...] = ()
    not_read: str = ''
    settings: Mapping[str, float] = field(default_factory=dict)


def check_settings(window_s: float, drop_db: float, min_s: float) -> None:
    """Refuse, with `ValueError`, settings `read_levels` cannot read with:
    anything not a finite number, a window that is not positive, a drop
    not beyond the 3 dB that bounds the time at level, and a negative
    least duration."""
    for name, value in (('window_s', window_s), ('drop_db', drop_db),
                        ('min_s', min_s)):
        if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)) \
                or not math.isfinite(float(value)):
            raise ValueError(f'{name} is a finite number, not {value!r}')
    if not window_s > 0:
        raise ValueError(f'the running-RMS window is longer than zero, not '
                         f'{window_s!r} s')
    if not drop_db > 10.0 * math.log10(2.0):
        raise ValueError(
            f'an interruption reaches further below the level than the 3 dB '
            f'that bounds the time at level, not {drop_db!r} dB')
    if min_s < 0:
        raise ValueError(f'an interruption lasts zero seconds or more, not '
                         f'{min_s!r} s')


def running_mean_square(values: np.ndarray, width: int) -> np.ndarray:
    """The mean square of `values` (channels by samples, each channel's
    own mean taken off first), averaged over the channels and over
    `width` samples at every sample: entry k covers samples k to
    k + width - 1. Length samples - width + 1."""
    total = np.zeros(values.shape[1])
    for row in values:
        centered = np.asarray(row, dtype=np.float64) - float(np.mean(row))
        total += centered * centered
    total /= len(values)
    summed = np.concatenate([[0.0], np.cumsum(total)])
    out = (summed[width:] - summed[:-width]) / width
    # a difference of sums can go a hair below zero where the signal is
    # zero; a mean square cannot
    return np.maximum(out, 0.0)


def read_levels(values: np.ndarray, rate: float,
                labels: Sequence[str] | None = None, start_s: float = 0.0,
                window_s: float = WINDOW_S, drop_db: float = DROP_DB,
                min_s: float = MIN_S) -> RecordReading:
    """How long a continuous run was at level, and where it was interrupted.

    The control channels' running mean square, each channel's mean
    taken off and the channels averaged, is taken over `window_s` at
    every sample. The run's level is the median of the running mean
    square over the stretches within 3 dB of its loudest, so a ramp-up,
    the lower steps of a step-up, a lead-in and a tail do not pull it
    down. The run is at level where the running mean square is at least
    half the level (within 3 dB). Where the record starts or stops
    sharply, from or to silence, that half-way point falls on the edge
    itself, whatever the window, so the time at level is read to a
    sample; an edge between two levels that are not silence is placed
    within half a window of where it is.

    An interruption is a stretch out of level between the first and the
    last instant at level, at least `min_s` long, that reaches at least
    `drop_db` below the level. The lead-in before the run reaches level
    and the ramp-down after are not interruptions. A shallower dip, or
    a shorter one, counts against the time at level and is not listed.

    The window sets what is seen: a dip shorter than about half the
    window is smoothed away, and a record dominated by a tone at a few
    hertz ripples the running RMS by about 1/(4 pi f window) of its
    mean square. Broadband random at the default window scatters by a
    few percent.

    A channel holding a NaN sample, and a channel that is constant
    (all zero included), is left out with the reason. When none is
    left, or the record is shorter than the window, nothing is read
    and `not_read` says why.

    Parameters
    ----------
    values : numpy.ndarray
        The channels the level is read on, channels by samples.
    rate : float
        Samples per second.
    labels : sequence of str, optional
        Each channel's name, for `channels` and `left_out`.
    start_s : float
        The time of the first sample on the record's own axis.
    window_s, drop_db, min_s : float
        The running-RMS window in seconds, how far below the level an
        interruption reaches, and how long it lasts at least.

    Returns
    -------
    RecordReading
        The time at level, the interruptions, and what was read on.

    Raises
    ------
    ValueError
        For a window that is not positive, a drop not beyond 3 dB, a
        negative `min_s`, or a rate that is not positive.
    """
    check_settings(window_s, drop_db, min_s)
    if not (math.isfinite(float(rate)) and rate > 0):
        raise ValueError(f'the sample rate is positive, not {rate!r}')
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    count = values.shape[1]
    labels = [str(label) for label in (labels if labels is not None
                                       else range(len(values)))]
    seconds = count / float(rate)
    settings = {'window_s': float(window_s), 'drop_db': float(drop_db),
                'min_s': float(min_s)}
    base = {'start_s': float(start_s), 'seconds': seconds,
            'settings': settings}
    if not len(values):
        return RecordReading(**base, not_read='no channel to read the level '
                             'on')
    # an odd number of samples, so each window has a sample at its
    # center and a start or stop from silence falls on a sample
    half = round(float(window_s) * float(rate) / 2.0)
    width = 2 * half + 1
    if half < 1:
        return RecordReading(**base, not_read=(
            f'the {window_s:g} s running-RMS window holds fewer than three '
            f'samples at {float(rate):g} samples per second'))
    if count < width:
        return RecordReading(**base, not_read=(
            f'the record ({seconds:.3g} s) is shorter than the {window_s:g} s '
            'running-RMS window'))
    used, left_out = [], []
    for label, row in zip(labels, values, strict=True):
        if not np.all(np.isfinite(row)):
            left_out.append(f'{label}: holds a sample that is not a number')
        elif not np.any(row):
            left_out.append(f'{label}: all zero')
        elif np.ptp(row) == 0:
            left_out.append(f'{label}: constant')
        else:
            used.append(label)
    if not used:
        return RecordReading(**base, left_out=tuple(left_out),
                             not_read='no channel varies: '
                             + '; '.join(left_out))
    rows = values[[labels.index(label) for label in used]]
    mean_square = running_mean_square(rows, width)
    loudest = float(mean_square.max())
    level = float(np.median(mean_square[mean_square >= loudest / 2.0]))
    # each window stands for the sample at its center; the half window
    # at either end takes the nearest window's reading
    centered = np.empty(count)
    lead = half
    centered[:lead] = mean_square[0]
    centered[lead:lead + len(mean_square)] = mean_square
    centered[lead + len(mean_square):] = mean_square[-1]
    # a window in which every channel reads exactly zero is a stop, said
    # as zero: the running sums leave a rounding residue there, not 0
    sounding = np.concatenate([[0], np.cumsum(np.any(rows != 0.0, axis=0))])
    silent = np.zeros(count, dtype=bool)
    silent[lead:lead + len(mean_square)] = (
        (sounding[width:] - sounding[:-width]) == 0)
    at = centered >= level / 2.0
    at_level = float(np.count_nonzero(at)) / float(rate)
    found = []
    inside = np.flatnonzero(at)
    first, last = int(inside[0]), int(inside[-1])
    out_of_level = ~at[first:last + 1]
    edges = np.flatnonzero(np.diff(np.concatenate(
        [[0], out_of_level.astype(np.int8), [0]])))
    for begin, end in zip(edges[::2], edges[1::2], strict=True):
        begin, end = begin + first, end + first
        length = (end - begin) / float(rate)
        if length < min_s:
            continue
        lowest = float(centered[begin:end].min())
        depth = (-math.inf if silent[begin:end].any() or lowest <= 0
                 else 10.0 * math.log10(lowest / level))
        if depth > -drop_db:
            continue
        found.append(Interruption(
            float(start_s) + begin / float(rate), length,
            None if depth <= ZERO_DB else depth))
    return RecordReading(**base, at_level_s=at_level,
                         interruptions=tuple(found), channels=tuple(used),
                         left_out=tuple(left_out))
