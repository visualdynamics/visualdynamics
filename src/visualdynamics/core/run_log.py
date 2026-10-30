"""The run log: what happened on the floor, in order, as it happened.

A run log is kept while a test runs, by a log keeper who is not at the
controller. Each thing written down (a level reached, an abort, a
clock check, a person arriving, a note) is one **entry**. An entry is
never edited: a mistake is put right by a **correction**, a later entry
that names the one it corrects and carries the new values, so the
reader sees the corrected view and the record keeps both.

The file is JSON Lines (`.vdlog`): UTF-8 text, one JSON object per
line, each line written, flushed and synced to the disk before the
writer returns, so a crash loses at most the entry being written. The
first line is the header, with the format's stamp and version. A file
cut off in the middle of its last line is read up to that line, and the
torn line is named (`RunLog.problems`); any other line that cannot be
read is skipped and named the same way, and the rest is read.

**Versions.** `FORMAT_VERSION` is the newest version this build reads,
and the one it writes. Each version has its own reader, so a later
build reads every earlier version, and a file from a newer version is
refused with a message to update. While Visual Dynamics is an alpha
the format may still change; from the first release that is not an
alpha, an older file always opens.

**Integrity.** The entries of each test are chained by SHA-256: every
entry carries the hash of the entry before it in the same test and the
hash of its own canonical form, which includes that previous hash. The
entries between tests have a chain of their own. `RunLog.verify` checks
every chain and names the first broken link by test and entry. The
chain detects accidental damage, a file cut short and casual edits. It
is **not** proof against deliberate alteration: anyone who edits an
entry can compute every hash after it again, and the chain then checks.
A file cut off at the end of an entry also leaves a valid chain; it is
found only against a head hash recorded outside the file
(`RunLog.head`, which a report's data index can carry). Every entry
records the computer's login and machine name beside the author's
typed initials; those are recorded, not authenticated.

**One writer.** A log open for writing holds a lock file beside it
(`<log>.lock`); a second writer is refused with a message naming who
holds it, and readers open the file read-only (`RunLog.read`). The lock
is the operating system's, so a writer that crashes leaves no lock
behind. Each entry can also be copied to a second path (`mirror`, the
lab's share or the data folder); a mirror that cannot be written is
reported (`RunLog.mirror_problem`) and never costs the entry its place
in the log itself.

The entry kinds and the fields each may carry are one registry
(`KINDS`). Each kind requires as little as it can: an abort needs only
its time, the kind of stop and the author, because the first minute
after a stop is the worst moment to be refused. What is still owed of
an entry is a derived view, not a refusal.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import math
import os
import platform
import re
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, BinaryIO, Self

from . import data_index, findings

# One module for the file and the model of its entries. The tests,
# axes and runs (their states, holds and owed lists) are built on the
# plain references every entry carries here, so this module never has
# to know what a test is beyond its number.

#: the extension of a run log
SUFFIX = '.vdlog'

#: the stamp on a run log's first line, which says what the file is
FORMAT = 'visualdynamics-run-log'

#: the newest format version this build reads, and the version it
#: writes. Each version from 1 has a reader in `_READERS`, so a file of
#: any earlier version opens; a file of a newer one is refused.
FORMAT_VERSION = 1

#: an entry entered this long after its event time is a late entry,
#: marked as such where the log is shown or exported
LATE_AFTER = timedelta(minutes=10)

#: the kinds of stop an abort entry records, the one thing it needs
#: besides its time and author. A hard shutdown (power loss, an
#: amplifier trip) may have over-tested the article, which is why the
#: kind of stop is asked first.
STOPS = ('controller ramp', 'amplifier trip', 'power loss', 'stop word')

#: what a field of an entry holds: free text, a finite number, a whole
#: number from 1, a yes or no, a list of texts, any one of the scalars,
#: or the id of an earlier entry
FIELD_TYPES: dict[str, str] = {
    'title': 'text', 'test_kind': 'text', 'article': 'text',
    'serial': 'text', 'procedure': 'text', 'reason': 'text',
    'field': 'text', 'value': 'scalar', 'was': 'scalar',
    'offset_s': 'number', 'recorder': 'text', 'method': 'text',
    'person': 'text', 'role': 'text', 'attendees': 'texts',
    'topics': 'texts', 'item': 'text', 'reading': 'scalar',
    'taken_by': 'text', 'step': 'text', 'result': 'text',
    'level_db': 'number', 'counter': 'text', 'voltage': 'number',
    'current': 'number', 'drive_fraction': 'number', 'answer': 'text',
    'go': 'flag', 'direction': 'text', 'n': 'whole', 'of': 'whole',
    'polarity': 'text', 'stop': 'text', 'channel': 'text',
    'band': 'text', 'panel_fault': 'text', 'called_by': 'text',
    'heard': 'text', 'records_read': 'text', 'over_test': 'text',
    'cause': 'text', 'refers_to': 'entry', 'path': 'text',
    'counter_at_stop': 'text', 'counter_at_restart': 'text',
    'depth_db': 'number', 'planned': 'flag', 'number': 'text',
    'fact': 'text', 'agreed_by': 'text', 'owner': 'text',
    'disposition': 'text', 'by': 'text', 'what': 'text',
    'subject': 'text', 'setting': 'text', 'to': 'text',
    'opinion': 'text',
}

#: what a correction can change besides an entry's fields: its event
#: time, its free text and its test, axis and run references
CORRECTABLE = ('event_time', 'text', 'test', 'axis', 'run')


@dataclass(frozen=True)
class EntryKind:
    """One kind of entry: its name in the file, its label in words, the
    fields it needs and the fields it may carry.

    `needs_test` is set for the kinds that mean nothing without a test
    (a test opened or closed). `machine` marks the kinds a lab's own
    machine log keeps: the trips, the setting changes, the clock checks,
    people in and out, the restarts, and the tests opened and closed
    that say what was running.
    """

    name: str
    label: str
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    needs_test: bool = False
    machine: bool = False

    @property
    def fields(self) -> tuple[str, ...]:
        """Every field the kind may carry, the required first."""
        return self.required + self.optional


def _kinds(*kinds: EntryKind) -> dict[str, EntryKind]:
    return {kind.name: kind for kind in kinds}


#: every kind of entry, by its name in the file, in the order an entry
#: form offers them
KINDS: dict[str, EntryKind] = _kinds(
    EntryKind('test_opened', 'Test opened',
              optional=('title', 'test_kind', 'article', 'serial',
                        'procedure'),
              needs_test=True, machine=True),
    EntryKind('test_closed', 'Test closed', optional=('reason',),
              needs_test=True, machine=True),
    EntryKind('header_change', 'Header changed', ('field', 'value'),
              ('was',)),
    EntryKind('clock_check', 'Clock check', ('offset_s',),
              ('recorder', 'method'), machine=True),
    EntryKind('person_in', 'Person in', ('person',), ('role',),
              machine=True),
    EntryKind('person_out', 'Person out', ('person',), ('role',),
              machine=True),
    EntryKind('brief', 'Brief', optional=('attendees', 'topics')),
    EntryKind('checklist_reading', 'Checklist reading', ('item',),
              ('reading', 'taken_by')),
    EntryKind('setup_step', 'Setup step', ('step',), ('result',)),
    EntryKind('run_start', 'Run started', optional=('level_db', 'counter')),
    EntryKind('level_row', 'Level',
              optional=('level_db', 'voltage', 'current', 'drive_fraction',
                        'answer', 'go')),
    EntryKind('sweep', 'Sweep', optional=('direction', 'n', 'of')),
    EntryKind('shot', 'Shot', optional=('polarity', 'n', 'of', 'level_db')),
    EntryKind('run_stop', 'Run stopped', optional=('counter', 'reason')),
    EntryKind('abort', 'Abort or trip', ('stop',),
              ('channel', 'band', 'panel_fault', 'counter', 'called_by',
               'heard', 'records_read', 'over_test'),
              machine=True),
    EntryKind('cause_found', 'Cause found', ('cause',), ('refers_to',)),
    EntryKind('path_decided', 'Path decided', ('path',), ('refers_to',)),
    EntryKind('restart', 'Restart',
              optional=('counter_at_stop', 'counter_at_restart', 'cause'),
              machine=True),
    EntryKind('notch', 'Notch',
              optional=('channel', 'band', 'depth_db', 'planned')),
    EntryKind('deviation', 'Deviation',
              optional=('number', 'fact', 'agreed_by')),
    EntryKind('anomaly_opened', 'Anomaly opened',
              optional=('number', 'fact', 'owner')),
    EntryKind('anomaly_dispositioned', 'Anomaly dispositioned', ('number',),
              ('disposition', 'by')),
    EntryKind('adjustment', 'Adjustment', optional=('what',)),
    EntryKind('photo', 'Photo', optional=('subject',)),
    EntryKind('observation', 'Observation'),
    EntryKind('setting_change', 'Setting changed', ('setting',),
              ('value', 'was'), machine=True),
    EntryKind('opinion_given', 'Opinion given', optional=('to', 'opinion')),
    EntryKind('post_test_step', 'Post-test step', ('step',), ('result',)),
    EntryKind('correction', 'Correction'),
    EntryKind('note', 'Note'),
)

#: the finding each numbered kind opens, and so the letters of its
#: number (D-1, A-1), numbered across the whole log
NUMBERED = {'deviation': 'deviation', 'anomaly_opened': 'anomaly'}

_HEX = re.compile(r'[0-9a-f]{64}')


class RunLogError(ValueError):
    """A run log, or an entry for one, that cannot be read or written
    one way; the message names the cause."""


class RunLogLocked(RunLogError):
    """The log is already open for writing elsewhere; the message names
    who holds it."""


@dataclass(frozen=True)
class FileRef:
    """A file an entry names: its name, its SHA-256 as 64 lower-case hex
    digits, and its size in bytes when known.

    `format`, `holds` and `path` say, when known, the file's format,
    what it holds (its run and channels) and where it was when it was
    hashed, in the words a data index records them
    (`data_index.IndexEntry`), so the log's files go into a report's
    data index as they are. Each is left out of the file when empty.
    """

    name: str
    sha256: str
    size: int | None = None
    format: str = ''
    holds: str = ''
    path: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise RunLogError('a file an entry names needs its name')
        if not isinstance(self.sha256, str) or not _HEX.fullmatch(self.sha256):
            raise RunLogError(f'{self.name}: {self.sha256!r} is not a SHA-256 '
                              '(64 lower-case hex digits)')
        if self.size is not None and (
                isinstance(self.size, bool) or not isinstance(self.size, int)
                or self.size < 0):
            raise RunLogError(f'{self.name}: a size is a whole number of '
                              f'bytes, not {self.size!r}')
        for label in ('format', 'holds', 'path'):
            if not isinstance(getattr(self, label), str):
                raise RunLogError(f'{self.name}: its {label} is text, not '
                                  f'{getattr(self, label)!r}')

    def to_dict(self) -> dict[str, Any]:
        """The file as the log stores it."""
        stored: dict[str, Any] = {'name': self.name, 'sha256': self.sha256}
        if self.size is not None:
            stored['size'] = self.size
        for label in ('format', 'holds', 'path'):
            if getattr(self, label):
                stored[label] = getattr(self, label)
        return stored


def file_ref(path: str | os.PathLike, name: str | None = None) -> FileRef:
    """A `FileRef` for the file at `path`, hashed in chunks
    (`data_index.digest`); `name` is the file's own name unless given."""
    size, sha = data_index.digest(path)
    return FileRef(name or os.path.basename(os.fspath(path)), sha, size)


@dataclass(frozen=True)
class Header:
    """A log's first line: its format version, its id, when and where it
    was created, and its title."""

    version: int
    log_id: str
    created: datetime
    title: str = ''
    login: str = ''
    computer: str = ''


@dataclass(frozen=True)
class Entry:
    """One entry, as stored, or as corrected (`RunLog.view`).

    `event_time` is when the thing happened and `entered_at` when it
    was written down, each with its date and its UTC offset. `test`,
    `axis` and `run` are plain references (a test and run by number, an
    axis by name), any of them absent. `corrects` is the id of the
    entry a correction corrects. `prev` and `hash` are the entry's links
    in its test's chain. In the corrected view, `corrected_by` lists the
    corrections applied, in order. `line` is the entry's line in the
    file, the header being line 1.
    """

    id: int
    kind: str
    event_time: datetime
    entered_at: datetime
    author: str
    login: str = ''
    computer: str = ''
    test: int | None = None
    axis: str | None = None
    run: int | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)
    text: str = ''
    files: tuple[FileRef, ...] = ()
    corrects: int | None = None
    prev: str = ''
    hash: str = ''
    corrected_by: tuple[int, ...] = field(default=(), compare=False)
    line: int = field(default=0, compare=False)

    @property
    def label(self) -> str:
        """The kind in words, as `KINDS` names it."""
        kind = KINDS.get(self.kind)
        return kind.label if kind is not None else self.kind

    @property
    def chain(self) -> str:
        """The chain the entry belongs to: 'test N', or 'log' for an
        entry between tests."""
        return chain_name(self.test)

    def is_late(self, after: timedelta = LATE_AFTER) -> bool:
        """Whether the entry was written down more than `after` after
        the event it records. Both times carry their offsets, so a
        change of daylight saving time between them counts as the real
        interval, not the difference of the wall clocks."""
        return self.entered_at - self.event_time > after

    def to_dict(self) -> dict[str, Any]:
        """The entry as the log stores it, its hash included."""
        stored: dict[str, Any] = {
            'id': self.id, 'kind': self.kind,
            'event_time': _time_text(self.event_time),
            'entered_at': _time_text(self.entered_at),
            'author': self.author, 'login': self.login,
            'computer': self.computer, 'test': self.test,
            'axis': self.axis, 'run': self.run,
            'fields': dict(self.fields), 'text': self.text,
            'files': [ref.to_dict() for ref in self.files],
        }
        if self.corrects is not None:
            stored['corrects'] = self.corrects
        stored['prev'] = self.prev
        stored['hash'] = self.hash
        return stored


@dataclass(frozen=True)
class ReadProblem:
    """A line of the file that was not read as an entry: its number (the
    header is line 1) and why."""

    line: int
    reason: str

    def __str__(self) -> str:
        return f'line {self.line}: {self.reason}'


@dataclass(frozen=True)
class Break:
    """A broken link found by `RunLog.verify`: the chain ('test N', or
    'log' for the entries between tests), the test's number, the
    entry's id and line where they are known, and what was found."""

    chain: str
    test: int | None
    entry: int | None
    line: int | None
    reason: str

    def __str__(self) -> str:
        where = self.chain
        if self.entry is not None:
            where += f', entry #{self.entry}'
        if self.line is not None:
            where += f' (line {self.line})'
        return f'{where}: {self.reason}'


@dataclass(frozen=True)
class Verification:
    """What `RunLog.verify` found: every break, in the order of the file,
    and how many entries and chains were checked. `ok` only when
    nothing is broken; `first` is the first break."""

    breaks: tuple[Break, ...]
    entries: int
    chains: int

    @property
    def ok(self) -> bool:
        """Whether every chain checked."""
        return not self.breaks

    @property
    def first(self) -> Break | None:
        """The first broken link, or None."""
        return self.breaks[0] if self.breaks else None

    def __str__(self) -> str:
        if self.ok:
            return (f'every chain checks: {self.entries} '
                    f'entr{"y" if self.entries == 1 else "ies"} in '
                    f'{self.chains} chain{"" if self.chains == 1 else "s"}. '
                    'This detects accidental damage and casual edits; it is '
                    'not proof against deliberate alteration.')
        return f'broken at {self.first}'


def chain_name(test: Any) -> str:
    """The name of the chain an entry with test reference `test` belongs
    to: 'test N', or 'log' for none."""
    return 'log' if test is None else f'test {test}'


def identity() -> tuple[str, str]:
    """(login, machine name) of this computer, each '' where the system
    does not say."""
    try:
        login = getpass.getuser()
    except (OSError, KeyError, ImportError):         # pragma: no cover
        # getuser raises when no variable names the user and the
        # password database has no entry; the log still opens
        login = ''
    return login, platform.node()


def canonical(stored: Mapping[str, Any]) -> str:
    """The canonical text of a stored entry (or header) that its hash is
    taken over: keys sorted, no spaces, UTF-8 characters as themselves,
    and the entry's own `hash` left out. The previous hash is a key of
    the entry, so it is part of what is hashed."""
    body = {key: value for key, value in stored.items() if key != 'hash'}
    return json.dumps(body, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _hash_v1(stored: Mapping[str, Any]) -> str:
    return _sha(canonical(stored))


def _genesis(header: Mapping[str, Any], chain: str) -> str:
    # Each chain starts from its log's header and its own name, so a
    # chain cannot be spliced in from another log, or from another test
    # of the same log, and an edited header breaks every chain.
    return _sha(canonical(header) + '\n' + chain)


def _time_text(when: datetime) -> str:
    return when.isoformat(timespec='milliseconds')


def _parse_time(text: Any, what: str) -> datetime:
    if not isinstance(text, str):
        raise RunLogError(f'{what} is not a time')
    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        raise RunLogError(f'{what} {text!r} is not a time') from None
    if when.tzinfo is None:
        raise RunLogError(f'{what} {text!r} has no UTC offset')
    return when


def _aware(when: Any, what: str) -> datetime:
    if not isinstance(when, datetime):
        raise RunLogError(f'{what} is a datetime, not {type(when).__name__}')
    if when.utcoffset() is None:
        raise RunLogError(f'{what} {when.isoformat()} has no time zone: a '
                          'log stores every time with its UTC offset, so a '
                          'change of daylight saving time cannot make two '
                          'times ambiguous')
    return when


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _reference(value: Any, what: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RunLogError(f'a {what} is referred to by its number from 1, '
                          f'not {value!r}')
    return value


def _axis(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RunLogError(f'an axis is referred to by its name, not {value!r}')
    return value.strip()


def _value(kind: EntryKind, name: str, value: Any) -> Any:
    """`value` as field `name` of an entry of `kind` holds it, or
    `RunLogError` naming the kind, the field and what it holds."""
    held = FIELD_TYPES.get(name, 'text')
    where = f'{kind.label}: {name}'
    if held == 'text':
        if not isinstance(value, str):
            raise RunLogError(f'{where} is text, not {value!r}')
        return value
    if held == 'texts':
        if isinstance(value, str) or not isinstance(value, Sequence) or not all(
                isinstance(item, str) for item in value):
            raise RunLogError(f'{where} is a list of texts, not {value!r}')
        return list(value)
    if held == 'flag':
        if not isinstance(value, bool):
            raise RunLogError(f'{where} is yes or no (True or False), '
                              f'not {value!r}')
        return value
    if held in ('whole', 'entry'):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RunLogError(f'{where} is a whole number from 1, not '
                              f'{value!r}')
        return value
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    if held == 'number' and not numeric:
        raise RunLogError(f'{where} is a number, not {value!r}')
    if held == 'scalar' and not (numeric or isinstance(value, (str, bool))):
        raise RunLogError(f'{where} is text, a number or yes or no, not '
                          f'{value!r}')
    if numeric and not math.isfinite(value):
        raise RunLogError(f'{where} is not a finite number ({value!r})')
    return value


def check_fields(kind: str, fields: Mapping[str, Any] | None,
                 ) -> dict[str, Any]:
    """The fields of a new entry of `kind`, checked against `KINDS`.

    A field set to None or to blank text is left out, so a form can
    pass every box it has. Refused with `RunLogError`, each naming the
    kind and the field: a kind that is not in `KINDS`, a field the kind
    does not carry, a value of the wrong type or not finite, a required
    field missing, and an abort's kind of stop not one of `STOPS`.
    """
    entry_kind = _kind(kind)
    if fields is None:
        fields = {}
    if not isinstance(fields, Mapping):
        raise RunLogError(f'an entry\'s fields are a mapping of names to '
                          f'values, not {type(fields).__name__}')
    kept: dict[str, Any] = {}
    for name, value in fields.items():
        if name not in entry_kind.fields:
            carried = ', '.join(entry_kind.fields) or 'none'
            raise RunLogError(f'a {entry_kind.label} entry has no field '
                              f'{name!r}; its fields are: {carried}')
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        kept[name] = _value(entry_kind, name, value)
    _check_required(entry_kind, kept)
    return kept


def _kind(kind: Any) -> EntryKind:
    if kind not in KINDS:
        raise RunLogError(f'{kind!r} is not a kind of entry; the kinds are: '
                          + ', '.join(KINDS))
    return KINDS[kind]


def _check_required(kind: EntryKind, fields: Mapping[str, Any]) -> None:
    for name in kind.required:
        if name not in fields:
            if name == 'stop':
                raise RunLogError(
                    'an abort entry needs the kind of stop: one of '
                    + ', '.join(STOPS))
            raise RunLogError(f'a {kind.label} entry needs its {name}')
    if 'stop' in fields and fields['stop'] not in STOPS:
        raise RunLogError(f'{fields["stop"]!r} is not a kind of stop; the '
                          'kinds are: ' + ', '.join(STOPS))


def _files(files: Iterable[Any]) -> tuple[FileRef, ...]:
    refs = []
    for item in files:
        if isinstance(item, FileRef):
            refs.append(item)
        elif isinstance(item, (str, os.PathLike)):
            refs.append(file_ref(item))
        else:
            raise RunLogError(f'a file is given by its path or a FileRef, '
                              f'not {type(item).__name__}')
    return tuple(refs)


# ---- reading each version ---------------------------------------------------


def _entry_v1(stored: Mapping[str, Any], line: int) -> Entry:
    """A version 1 line as an `Entry`, or `RunLogError` saying why it is
    not one."""
    for key in ('id', 'kind', 'event_time', 'entered_at', 'author'):
        if key not in stored:
            raise RunLogError(f'it has no {key}')
    entry_id = stored['id']
    if isinstance(entry_id, bool) or not isinstance(entry_id, int) \
            or entry_id < 1:
        raise RunLogError(f'its id {entry_id!r} is not a number from 1')
    if stored['kind'] not in KINDS:
        raise RunLogError(f'its kind {stored["kind"]!r} is not one this '
                          'version knows')
    fields = stored.get('fields') or {}
    if not isinstance(fields, Mapping):
        raise RunLogError('its fields are not a mapping')
    files = []
    for item in stored.get('files') or []:
        if not isinstance(item, Mapping):
            raise RunLogError('a file it names is not a mapping')
        files.append(FileRef(item.get('name'), item.get('sha256'),
                             item.get('size'), item.get('format') or '',
                             item.get('holds') or '', item.get('path') or ''))
    corrects = stored.get('corrects')
    if corrects is not None:
        _reference(corrects, 'corrected entry')
    return Entry(
        id=entry_id, kind=stored['kind'],
        event_time=_parse_time(stored['event_time'], 'its event time'),
        entered_at=_parse_time(stored['entered_at'], 'its entered-at time'),
        author=str(stored['author']), login=str(stored.get('login') or ''),
        computer=str(stored.get('computer') or ''),
        test=_reference(stored.get('test'), 'test'),
        axis=_axis(stored.get('axis')),
        run=_reference(stored.get('run'), 'run'),
        fields=dict(fields), text=str(stored.get('text') or ''),
        files=tuple(files), corrects=corrects,
        prev=str(stored.get('prev') or ''),
        hash=str(stored.get('hash') or ''), line=line)


def _header_v1(stored: Mapping[str, Any]) -> Header:
    log_id = stored.get('log_id')
    if not isinstance(log_id, str) or not log_id:
        raise RunLogError('its header has no log id')
    return Header(version=1, log_id=log_id,
                  created=_parse_time(stored.get('created'),
                                      'its creation time'),
                  title=str(stored.get('title') or ''),
                  login=str(stored.get('login') or ''),
                  computer=str(stored.get('computer') or ''))


@dataclass(frozen=True)
class _Reader:
    """How one format version is read: its header, each entry, and the
    hash of an entry as that version wrote it."""

    header: Callable[[Mapping[str, Any]], Header]
    entry: Callable[[Mapping[str, Any], int], Entry]
    hash: Callable[[Mapping[str, Any]], str]


#: a reader for every format version from 1. A new version adds its
#: reader here and its frozen file under `testdata/vdlog_corpus/`, and
#: every earlier reader stays, so every earlier file still opens.
_READERS: dict[int, _Reader] = {1: _Reader(_header_v1, _entry_v1, _hash_v1)}


# ---- the lock -----------------------------------------------------------------

#: the byte the lock is taken on, far past anything the lock file holds,
#: so a refused writer can still read who holds it (Windows refuses to
#: read a locked byte)
_LOCK_BYTE = 1 << 20


def _try_lock(handle: Any) -> bool:
    if os.name == 'nt':
        import msvcrt
        handle.seek(_LOCK_BYTE)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl  # pragma: no cover
    try:                                             # pragma: no cover
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:                                  # pragma: no cover
        return False
    return True                                      # pragma: no cover


def _unlock(handle: Any) -> None:
    if os.name == 'nt':
        import msvcrt
        handle.seek(_LOCK_BYTE)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:                              # pragma: no cover
            pass
        return
    import fcntl  # pragma: no cover
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)      # pragma: no cover


class _Lock:
    """The one-writer lock on a log: an operating-system lock on a file
    beside it, which names its holder."""

    def __init__(self, log_path: str, login: str, computer: str) -> None:
        self.path: str = log_path + '.lock'
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        self.handle: BinaryIO = os.fdopen(fd, 'r+b')
        if not _try_lock(self.handle):
            holder = self._holder()
            self.handle.close()
            raise RunLogLocked(
                f'{log_path} is open for writing by {holder}. Open it '
                'read-only, or close it there first.')
        holder = {'login': login, 'computer': computer, 'pid': os.getpid(),
                  'since': _time_text(_local_now())}
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps(holder).encode('utf-8'))
        self.handle.flush()

    def _holder(self) -> str:
        try:
            with open(self.path, 'rb') as handle:
                held = json.loads(handle.read(4096).decode('utf-8'))
            return (f'{held.get("login") or "someone"} on '
                    f'{held.get("computer") or "another computer"} '
                    f'(process {held.get("pid")}) since {held.get("since")}')
        except (OSError, ValueError, AttributeError):
            return 'another writer (the lock file does not say who)'

    def release(self) -> None:
        _unlock(self.handle)
        self.handle.close()
        try:
            os.remove(self.path)
        except OSError:
            # another writer may have it open, waiting; it is harmless
            # left behind, since the lock is the operating system's
            pass


# ---- the log ------------------------------------------------------------------


def _durable_append(handle: Any, data: bytes) -> None:
    handle.write(data)
    handle.flush()
    os.fsync(handle.fileno())


class RunLog:
    """A run log: its header, its entries in the order written, and, when
    opened for writing, the way to add to it.

    Make one with `RunLog.create` (a new file), `RunLog.open` (an
    existing file, for writing) or `RunLog.read` (any file, read-only).
    A log open for writing holds the lock until `close`; use it in a
    ``with`` block.
    """

    def __init__(self, path: str, header: Header,
                 header_raw: Mapping[str, Any], *,
                 clock: Callable[[], datetime] | None = None,
                 who: tuple[str, str] | None = None) -> None:
        self.path: str = path
        self.header: Header = header
        self._header_raw = dict(header_raw)
        self._reader = _READERS[header.version]
        self._clock = clock or _local_now
        self._who = who if who is not None else identity()
        self._records: list[tuple[int, dict[str, Any]]] = []
        self._entries: list[Entry] = []
        self._by_id: dict[int, Entry] = {}
        self._corrections: dict[int, list[Entry]] = {}
        self._heads: dict[str, str] = {}
        self._problems: list[ReadProblem] = []
        self._lines = 1
        self._next_id = 1
        self._handle: Any = None
        self._lock: _Lock | None = None
        self._torn_tail = False
        self.mirror: str | None = None
        self.mirror_problem: str = ''
        self._mirror_in_step = False

    # ---- opening ----------------------------------------------------------------

    @classmethod
    def create(cls, path: str | os.PathLike, *, title: str = '',
               mirror: str | os.PathLike | None = None,
               clock: Callable[[], datetime] | None = None,
               who: tuple[str, str] | None = None) -> RunLog:
        """A new run log at `path`, open for writing.

        Parameters
        ----------
        path : str or os.PathLike
            Where the log is written. Nothing may be there yet.
        title : str, optional
            The log's title (the article or program it is kept for).
        mirror : str or os.PathLike, optional
            A second path every entry is copied to as well.
        clock : callable, optional
            Returns the time entries are entered at, with its zone; the
            computer's own clock unless given.
        who : (str, str), optional
            The (login, machine name) every entry records; read from the
            computer unless given.

        Returns
        -------
        RunLog
            The new log, holding only its header.

        Raises
        ------
        RunLogError
            When a file is already at `path`, or the title is not text.
        RunLogLocked
            When another writer holds the lock beside `path`.
        """
        path = os.fspath(path)
        if os.path.exists(path):
            raise RunLogError(f'a file is already at {path}; open it to add '
                              'to it, or choose another name')
        if not isinstance(title, str):
            raise RunLogError(f'a title is text, not {title!r}')
        login, computer = who if who is not None else identity()
        lock = _Lock(path, login, computer)
        try:
            raw = cls._new_header(title, login, computer, clock)
            with open(path, 'xb') as handle:
                _durable_append(handle, cls._line(raw))
            _sync_folder(path)
            log = cls(path, _READERS[FORMAT_VERSION].header(raw), raw,
                      clock=clock, who=(login, computer))
            log._take(lock, mirror)
        except BaseException:
            lock.release()
            raise
        return log

    @classmethod
    def open(cls, path: str | os.PathLike, *,
             mirror: str | os.PathLike | None = None,
             clock: Callable[[], datetime] | None = None,
             who: tuple[str, str] | None = None) -> RunLog:
        """The run log at `path`, open for writing.

        A torn last line (a crash in the middle of an entry) is named in
        `problems` and ended, so the next entry starts on a line of its
        own; nothing in the file is removed. An empty file (a crash
        while it was being created) is given its header.

        Parameters
        ----------
        path : str or os.PathLike
            The log.
        mirror : str or os.PathLike, optional
            A second path every entry is copied to as well; the whole
            log is copied there first.
        clock : callable, optional
            Returns the time entries are entered at, with its zone.
        who : (str, str), optional
            The (login, machine name) every entry records.

        Returns
        -------
        RunLog
            The log, with every entry read.

        Raises
        ------
        RunLogError
            When the file is not a run log, is of a newer format version,
            or is of an older version (which opens read-only, since a log
            is only ever written in one version).
        RunLogLocked
            When another writer holds the lock.
        FileNotFoundError
            When there is no file at `path`.
        """
        path = os.fspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'no run log at {path}')
        login, computer = who if who is not None else identity()
        lock = _Lock(path, login, computer)
        try:
            with open(path, 'rb') as handle:
                data = handle.read()
            if not data.strip():
                raw = cls._new_header('', login, computer, clock)
                with open(path, 'wb') as handle:
                    _durable_append(handle, cls._line(raw))
                log = cls(path, _READERS[FORMAT_VERSION].header(raw), raw,
                          clock=clock, who=(login, computer))
                log._problems.append(ReadProblem(
                    1, 'the file was empty, as a crash while it was being '
                    'created leaves it; its header was written'))
            else:
                log = cls._parse(path, data, clock=clock,
                                 who=(login, computer))
                if log.header.version != FORMAT_VERSION:
                    raise RunLogError(
                        f'{path} is in run log format version '
                        f'{log.header.version}; this build adds entries only '
                        f'to version {FORMAT_VERSION}. Read it with '
                        'RunLog.read, and start a new log to go on.')
            log._take(lock, mirror)
        except BaseException:
            lock.release()
            raise
        return log

    @classmethod
    def read(cls, path: str | os.PathLike) -> RunLog:
        """The run log at `path`, read-only, without the lock, so it can be
        read while another program writes it.

        Parameters
        ----------
        path : str or os.PathLike
            The log.

        Returns
        -------
        RunLog
            The log, every entry that could be read in it, and the lines
            that could not in `problems`.

        Raises
        ------
        RunLogError
            When the file is empty, is not a run log, or is of a newer
            format version than this build reads.
        FileNotFoundError
            When there is no file at `path`.
        """
        path = os.fspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'no run log at {path}')
        with open(path, 'rb') as handle:
            data = handle.read()
        if not data.strip():
            raise RunLogError(f'{path} is empty: it has no header line, so it '
                              'holds no log yet')
        return cls._parse(path, data)

    @staticmethod
    def _new_header(title: str, login: str, computer: str,
                    clock: Callable[[], datetime] | None) -> dict[str, Any]:
        import uuid
        created = _aware((clock or _local_now)(), 'the clock\'s time')
        return {'format': FORMAT, 'version': FORMAT_VERSION,
                'log_id': uuid.uuid4().hex, 'created': _time_text(created),
                'title': title, 'login': login, 'computer': computer}

    @staticmethod
    def _line(stored: Mapping[str, Any]) -> bytes:
        return (json.dumps(stored, ensure_ascii=False, allow_nan=False)
                + '\n').encode('utf-8')

    @classmethod
    def _parse(cls, path: str, data: bytes, *,
               clock: Callable[[], datetime] | None = None,
               who: tuple[str, str] | None = None) -> RunLog:
        lines = data.split(b'\n')
        ended = lines[-1] == b''
        if ended:
            lines.pop()
        try:
            header_raw = json.loads(lines[0].rstrip(b'\r').decode('utf-8'))
        except (UnicodeDecodeError, ValueError):
            if len(lines) == 1 and not ended:
                raise RunLogError(f'{path}: its first line is torn, as a crash '
                                  'while the log was being created leaves '
                                  'it') from None
            raise RunLogError(f'{path} is not a Visual Dynamics run log (its '
                              'first line is not its header)') from None
        if not isinstance(header_raw, dict) or header_raw.get('format') != FORMAT:
            raise RunLogError(f'{path} is not a Visual Dynamics run log (no '
                              f'{FORMAT!r} stamp on its first line)')
        version = header_raw.get('version')
        if isinstance(version, bool) or not isinstance(version, int):
            raise RunLogError(f'{path}: its header has no format version')
        if version > FORMAT_VERSION:
            raise RunLogError(
                f'{path} was written by a newer Visual Dynamics (run log '
                f'format {version}; this build reads up to {FORMAT_VERSION}). '
                'Update to open it.')
        if version not in _READERS:
            raise RunLogError(f'{path}: run log format {version} is not a '
                              'version this build knows')
        try:
            header = _READERS[version].header(header_raw)
        except RunLogError as error:
            raise RunLogError(f'{path}: {error}') from None
        log = cls(path, header, header_raw, clock=clock,
                  who=who)
        last = len(lines)
        for number, raw in enumerate(lines[1:], start=2):
            raw = raw.rstrip(b'\r')
            log._lines = number
            if not raw.strip():
                continue
            torn = number == last and not ended
            try:
                stored = json.loads(raw.decode('utf-8'))
            except (UnicodeDecodeError, ValueError):
                stored = None
            if not isinstance(stored, dict):
                if torn:
                    log._torn_tail = True
                    log._problems.append(ReadProblem(
                        number, 'the last line is torn: the file was cut off '
                        'in the middle of an entry, as a crash leaves it. '
                        'It was skipped.'))
                else:
                    log._problems.append(ReadProblem(
                        number, 'not an entry (not one JSON object); skipped'))
                continue
            if torn:
                log._torn_tail = True
                log._problems.append(ReadProblem(
                    number, 'the last line has no line end; it was read'))
            log._admit(number, stored)
        return log

    def _admit(self, line: int, stored: dict[str, Any]) -> None:
        """Take a stored line into the log: always onto its chain, and
        into the entries when it reads as one."""
        self._records.append((line, stored))
        entry_id = stored.get('id')
        if isinstance(entry_id, int) and not isinstance(entry_id, bool):
            self._next_id = max(self._next_id, entry_id + 1)
        if isinstance(stored.get('hash'), str):
            self._heads[chain_name(stored.get('test'))] = stored['hash']
        try:
            entry = self._reader.entry(stored, line)
        except (RunLogError, TypeError, ValueError) as error:
            self._problems.append(ReadProblem(line, f'not read as an entry: '
                                              f'{error}'))
            return
        if entry.id in self._by_id:
            self._problems.append(ReadProblem(
                line, f'a second entry #{entry.id}; the first was kept'))
            return
        self._entries.append(entry)
        self._by_id[entry.id] = entry
        if entry.kind == 'correction' and entry.corrects is not None:
            self._corrections.setdefault(entry.corrects, []).append(entry)

    def _take(self, lock: _Lock, mirror: str | os.PathLike | None) -> None:
        if mirror is not None:
            mirror = os.fspath(mirror)
            if _same_path(mirror, self.path):
                raise RunLogError('a mirror is a second path, not the log '
                                  'itself')
        # held open, and closed by close(), like the lock
        self._handle = open(self.path, 'ab')            # noqa: SIM115
        self._lock = lock
        if self._torn_tail:
            # end the torn line rather than remove it: the fragment is
            # evidence of the crash, and the next entry must start on a
            # line of its own
            _durable_append(self._handle, b'\n')
            self._torn_tail = False
        if mirror is not None:
            self.mirror = mirror
            self._copy_to_mirror()

    # ---- closing ----------------------------------------------------------------

    def close(self) -> None:
        """Release the lock and the file. A read-only log has neither;
        closing it does nothing."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    @property
    def writable(self) -> bool:
        """Whether entries can be added: opened for writing and not yet
        closed."""
        return self._handle is not None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        count = len(self._entries)
        return (f'RunLog({self.path!r}, {count} entr'
                f'{"y" if count == 1 else "ies"})')

    # ---- reading ----------------------------------------------------------------

    @property
    def version(self) -> int:
        """The format version the file is written in."""
        return self.header.version

    @property
    def entries(self) -> tuple[Entry, ...]:
        """Every entry as stored, corrections included, in the order
        written."""
        return tuple(self._entries)

    @property
    def problems(self) -> tuple[ReadProblem, ...]:
        """Every line that was not read as an entry, and why: a torn last
        line, a line that is not JSON, a line that is not an entry."""
        return tuple(self._problems)

    def entry(self, entry_id: int) -> Entry:
        """The entry with id `entry_id`, as stored."""
        if entry_id not in self._by_id:
            raise KeyError(f'the log has no entry #{entry_id}')
        return self._by_id[entry_id]

    def corrections_of(self, entry_id: int) -> tuple[Entry, ...]:
        """The corrections of entry `entry_id`, in the order written."""
        return tuple(self._corrections.get(entry_id, ()))

    def current(self, entry_id: int) -> Entry:
        """Entry `entry_id` with its corrections applied in order, each
        one's values replacing the entry's and a value of None striking
        a field; `corrected_by` lists them. The stored entry is
        unchanged (`entry`)."""
        entry = self.entry(entry_id)
        corrections = self.corrections_of(entry_id)
        if not corrections:
            return entry
        fields = dict(entry.fields)
        changed: dict[str, Any] = {}
        for correction in corrections:
            for name, value in correction.fields.items():
                if name in CORRECTABLE:
                    changed[name] = value
                elif value is None:
                    fields.pop(name, None)
                else:
                    fields[name] = value
        kept = {}
        for name, value in changed.items():
            try:
                if name == 'event_time':
                    kept[name] = _parse_time(value, 'a corrected event time')
                elif name == 'text':
                    kept[name] = '' if value is None else str(value)
                elif name == 'axis':
                    kept[name] = _axis(value)
                else:
                    kept[name] = _reference(value, name)
            except RunLogError:
                # an unreadable value in a hand-edited file: the stored
                # value stands, and the correction is still listed
                continue
        return replace(entry, fields=fields,
                       corrected_by=tuple(c.id for c in corrections), **kept)

    def view(self) -> list[Entry]:
        """The corrected view: every entry but the corrections, in the
        order written, each with its corrections applied
        (`current`)."""
        return [self.current(entry.id) for entry in self._entries
                if entry.kind != 'correction']

    def tests(self) -> list[int]:
        """The test numbers the log's entries refer to, in the order
        first referred to."""
        seen: list[int] = []
        for entry in self._entries:
            if entry.test is not None and entry.test not in seen:
                seen.append(entry.test)
        return seen

    def chains(self) -> list[str]:
        """The names of the log's chains, in the order first written:
        'log' for the entries between tests, 'test N' for each test."""
        seen: list[str] = []
        for _, stored in self._records:
            name = chain_name(stored.get('test'))
            if name not in seen:
                seen.append(name)
        return seen

    def head(self, test: int | None = None) -> str:
        """The head of a test's chain (`test` None for the entries
        between tests): the hash of its last entry, which the next entry
        will carry as its previous hash. A chain with no entries yet has
        its starting hash. A head recorded outside the file (printed at
        the test's close, in a report's data index) is what
        `verify(heads=...)` finds a file cut short against."""
        name = chain_name(test)
        return self._heads.get(name) or _genesis(self._header_raw, name)

    def next_number(self, kind: str) -> str:
        """The number the next finding of `kind` ('deviation' or
        'anomaly') takes, numbered across the whole log: D-1, D-2, A-1."""
        return findings.next_number(self._numbers(), kind)

    def _numbers(self, kind: str | None = None) -> list[dict[str, str]]:
        numbers = []
        for entry in self._entries:
            if entry.kind in NUMBERED and 'number' in entry.fields and (
                    kind is None or entry.kind == kind):
                try:
                    findings.number(entry.fields['number'])
                except (TypeError, ValueError):
                    continue
                numbers.append({'number': entry.fields['number']})
        return numbers

    # ---- the clock --------------------------------------------------------------

    def clock_check_for(self, at: Entry | datetime,
                        recorder: str | None = None) -> Entry | None:
        """The clock check in force at `at` (an entry, by its corrected
        event time, or a time with its zone): the latest clock check at
        or before it, of `recorder` when named. None when no clock check
        comes before it."""
        when = self._instant(at)
        best: Entry | None = None
        for stored in self._entries:
            if stored.kind != 'clock_check':
                continue
            entry = self.current(stored.id)
            if entry.event_time > when:
                continue
            if recorder is not None and entry.fields.get('recorder') != recorder:
                continue
            if best is None or entry.event_time >= best.event_time:
                best = entry
        return best

    def recorder_time(self, at: Entry | datetime,
                      recorder: str | None = None) -> datetime | None:
        """`at` on the recorders' clock: its time plus the offset of the
        clock check in force (`clock_check_for`), which records the
        recorders' clock less this computer's, in seconds. None when no
        clock check comes before it."""
        when = self._instant(at)
        check = self.clock_check_for(when, recorder)
        if check is None:
            return None
        return when + timedelta(seconds=float(check.fields['offset_s']))

    def _instant(self, at: Entry | datetime) -> datetime:
        if isinstance(at, Entry):
            return self.current(at.id).event_time if at.id in self._by_id \
                else at.event_time
        return _aware(at, 'a time')

    # ---- integrity --------------------------------------------------------------

    def verify(self, heads: Mapping[int | None, str] | None = None,
               tests: Iterable[int | None] | None = None) -> Verification:
        """Check every chain, and the heads recorded outside the file.

        Each entry's previous hash must be the hash of the entry before
        it in its chain (the chain's starting hash for the first), and
        its hash must be the hash of its own canonical form. Entry ids
        must run from 1 without a gap. A head in `heads` (by test number,
        None for the entries between tests) must still be in its chain;
        one that is not means the file was cut short or changed after
        the head was recorded.

        This detects accidental damage, a file cut short (mid-entry on
        reading; at an entry's end only against `heads`) and casual
        edits. It is not proof against deliberate alteration: an edit
        whose author computes every following hash again is not
        detected.

        Parameters
        ----------
        heads : mapping, optional
            Heads recorded outside the file, by test number.
        tests : iterable, optional
            Check only these tests' chains (None for the entries between
            tests), as a file holding one test's entries is checked;
            the ids then need not run without a gap.

        Returns
        -------
        Verification
            Every break in the order of the file, and the first.
        """
        breaks: list[tuple[float, Break]] = []
        last: dict[str, str] = {}
        seen: dict[str, set[str]] = {}
        only = None if tests is None else {chain_name(t) for t in tests}
        expected_id = 1
        for line, stored in self._records:
            test = stored.get('test')
            name = chain_name(test)
            if only is not None and name not in only:
                continue
            test = test if isinstance(test, int) else None
            entry_id = stored.get('id')
            entry_id = entry_id if isinstance(entry_id, int) else None
            previous = last.get(name) or _genesis(self._header_raw, name)
            if only is None and entry_id != expected_id:
                if entry_id is not None and entry_id > expected_id:
                    gap = (f'entry #{expected_id} is missing' if
                           entry_id == expected_id + 1 else
                           f'entries #{expected_id} to #{entry_id - 1} are '
                           'missing')
                    reason = f'the ids jump: {gap}'
                else:
                    reason = (f'the id is out of order (#{expected_id} was '
                              'next)')
                breaks.append((line, Break(name, test, entry_id, line,
                                           reason)))
            if entry_id is not None:
                expected_id = max(expected_id, entry_id + 1)
            if stored.get('prev') != previous:
                breaks.append((line, Break(
                    name, test, entry_id, line,
                    'its previous hash is not the hash of the entry before '
                    'it in the chain: an entry before it was removed, moved '
                    'or changed')))
            try:
                computed = self._reader.hash(stored)
            except (TypeError, ValueError):
                computed = ''
            if stored.get('hash') != computed:
                breaks.append((line, Break(
                    name, test, entry_id, line,
                    'its hash does not match its contents: it was changed '
                    'after it was written')))
            current = stored.get('hash') if isinstance(stored.get('hash'),
                                                       str) else computed
            last[name] = current
            seen.setdefault(name, set()).add(current)
        end = math.inf                               # after every line
        for test, recorded in (heads or {}).items():
            name = chain_name(test)
            if recorded != _genesis(self._header_raw, name) \
                    and recorded not in seen.get(name, set()):
                breaks.append((end, Break(
                    name, test, None, None,
                    f'the head recorded for it ({recorded[:12]}...) is not '
                    'in the log: the log was cut short, or changed, after '
                    'the head was recorded')))
        breaks.sort(key=lambda found: found[0])
        return Verification(tuple(found for _, found in breaks),
                            len(self._records), len(last))

    # ---- writing ----------------------------------------------------------------

    def append(self, kind: str, *, author: str,
               fields: Mapping[str, Any] | None = None, text: str = '',
               event_time: datetime | None = None, test: int | None = None,
               axis: str | None = None, run: int | None = None,
               files: Iterable[str | os.PathLike | FileRef] = (),
               ) -> Entry:
        """Write one entry, synced to the disk before this returns.

        A deviation or anomaly opened without a number takes the next of
        its kind across the log (`next_number`); a number given must be
        of its kind and not already taken in the log. A correction is written with
        `correct`, not here.

        Parameters
        ----------
        kind : str
            One of `KINDS`.
        author : str
            The author's typed initials.
        fields : mapping, optional
            The kind's fields (`check_fields`); a field set to None or
            blank is left out.
        text : str, optional
            Free text.
        event_time : datetime, optional
            When the thing happened, with its zone; now, if not given.
        test, run : int, optional
            The test and run it belongs to, by number from 1.
        axis : str, optional
            The axis it belongs to, by name.
        files : iterable, optional
            Files it names, each a path (hashed now) or a `FileRef`.

        Returns
        -------
        Entry
            The entry as written, with its id and hashes.

        Raises
        ------
        RunLogError
            For a log not open for writing, and for anything the entry
            cannot hold one way, the message naming the field.
        """
        if kind == 'correction':
            raise RunLogError('a correction is written with correct(), which '
                              'names the entry it corrects')
        entry_kind = _kind(kind)
        kept = check_fields(kind, fields)
        if entry_kind.needs_test and test is None:
            raise RunLogError(f'a {entry_kind.label} entry needs the test it '
                              'belongs to')
        self._check_entry_refs(entry_kind, kept)
        if kind in NUMBERED:
            kept = self._numbered(kind, kept)
        if kind == 'anomaly_dispositioned':
            opened = {item['number'] for item in
                      self._numbers('anomaly_opened')}
            if kept['number'] not in opened:
                raise RunLogError(f'{kept["number"]} is not an anomaly opened '
                                  'in this log')
        return self._write(kind, author=author, fields=kept,
                           text=text, event_time=event_time, test=test,
                           axis=axis, run=run, files=files, corrects=None)

    def _check_entry_refs(self, kind: EntryKind,
                          fields: Mapping[str, Any]) -> None:
        for name, value in fields.items():
            if FIELD_TYPES.get(name) == 'entry' and value not in self._by_id:
                raise RunLogError(f'{kind.label}: {name} names entry '
                                  f'#{value}, which the log does not hold')

    def _numbered(self, kind: str, kept: dict[str, Any]) -> dict[str, Any]:
        finding = NUMBERED[kind]
        if 'number' not in kept:
            return {'number': self.next_number(finding), **kept}
        try:
            numbered_kind, _ = findings.number(kept['number'])
        except ValueError as error:
            raise RunLogError(str(error)) from None
        if numbered_kind != finding:
            raise RunLogError(f'{kept["number"]} is not the number of '
                              f'{"a" if finding == "deviation" else "an"} '
                              f'{finding}')
        if kept['number'] in {item['number'] for item in self._numbers()}:
            raise RunLogError(f'{kept["number"]} is already taken in this log; '
                              f'the next is {self.next_number(finding)}')
        return kept

    def correct(self, target: int | Entry, *, author: str,
                fields: Mapping[str, Any] | None = None, text: str = '',
                event_time: datetime | None = None) -> Entry:
        """Write a correction of an earlier entry, which stays as it was.

        Parameters
        ----------
        target : int or Entry
            The entry corrected, by id. A correction is not corrected
            itself; correct the original entry again.
        author : str
            The author's typed initials.
        fields : mapping
            The new values: any of the corrected entry's fields (None
            strikes an optional one), and any of `CORRECTABLE` (its
            event time, free text, test, axis and run).
        text : str, optional
            Why it is corrected.
        event_time : datetime, optional
            When the correction was made; now, if not given.

        Returns
        -------
        Entry
            The correction as written. It belongs to the corrected
            entry's test, so it is in that test's chain.

        Raises
        ------
        RunLogError
            When the entry is not in the log or is a correction, when no
            value is given, when a value is one the entry cannot hold,
            when a required field would be struck, or when a finding's
            number would change.
        """
        entry_id = target.id if isinstance(target, Entry) else target
        if entry_id not in self._by_id:
            raise RunLogError(f'the log has no entry #{entry_id} to correct')
        original = self._by_id[entry_id]
        if original.kind == 'correction':
            raise RunLogError(f'entry #{entry_id} is a correction; correct '
                              f'the original, #{original.corrects}, again')
        if not fields:
            raise RunLogError('a correction needs the values it corrects')
        if not isinstance(fields, Mapping):
            raise RunLogError(f'a correction\'s values are a mapping, not '
                              f'{type(fields).__name__}')
        entry_kind = KINDS[original.kind]
        changes: dict[str, Any] = {}
        for name, value in fields.items():
            if name in CORRECTABLE:
                changes[name] = self._corrected_reference(name, value)
            elif name not in entry_kind.fields:
                carried = ', '.join(entry_kind.fields + CORRECTABLE)
                raise RunLogError(f'a {entry_kind.label} entry has no field '
                                  f'{name!r}; a correction may change: '
                                  f'{carried}')
            elif name == 'number' and original.kind in (
                    *NUMBERED, 'anomaly_dispositioned'):
                raise RunLogError('a finding\'s number is never changed; open '
                                  'a new finding instead')
            elif value is None:
                changes[name] = None
            else:
                changes[name] = _value(entry_kind, name, value)
                self._check_entry_refs(entry_kind, {name: changes[name]})
        corrected = dict(self.current(entry_id).fields)
        for name, value in changes.items():
            if name in CORRECTABLE:
                continue
            if value is None:
                corrected.pop(name, None)
            else:
                corrected[name] = value
        _check_required(entry_kind, corrected)
        if entry_kind.needs_test and 'test' in changes \
                and changes['test'] is None:
            raise RunLogError(f'a {entry_kind.label} entry needs the test it '
                              'belongs to')
        return self._write('correction', author=author,
                           fields=changes, text=text, event_time=event_time,
                           test=original.test, axis=original.axis,
                           run=original.run, files=(), corrects=entry_id)

    @staticmethod
    def _corrected_reference(name: str, value: Any) -> Any:
        if name == 'event_time':
            return _time_text(_aware(value, 'a corrected event time'))
        if name == 'text':
            if not isinstance(value, str):
                raise RunLogError(f'free text is text, not {value!r}')
            return value
        if name == 'axis':
            return _axis(value)
        return _reference(value, name)

    def _write(self, kind: str, *, author: str,
               fields: Mapping[str, Any], text: str,
               event_time: datetime | None, test: int | None,
               axis: str | None, run: int | None, files: Iterable[Any],
               corrects: int | None) -> Entry:
        if self._handle is None:
            raise RunLogError(f'{self.path} is not open for writing (it was '
                              'read, or it is closed); open it with '
                              'RunLog.open to add to it')
        if not isinstance(author, str) or not author.strip():
            raise RunLogError('an entry needs its author\'s initials')
        if not isinstance(text, str):
            raise RunLogError(f'free text is text, not {text!r}')
        entered_at = _aware(self._clock(), 'the clock\'s time')
        when = (entered_at if event_time is None
                else _aware(event_time, 'an event time'))
        login, computer = self._who
        entry = Entry(id=self._next_id, kind=kind, event_time=when,
                      entered_at=entered_at, author=author.strip(),
                      login=login, computer=computer,
                      test=_reference(test, 'test'), axis=_axis(axis),
                      run=_reference(run, 'run'), fields=dict(fields),
                      text=text, files=_files(files), corrects=corrects,
                      prev=self.head(test))
        stored = entry.to_dict()
        stored['hash'] = self._reader.hash(stored)
        entry = replace(entry, hash=stored['hash'], line=self._lines + 1)
        data = self._line(stored)
        size = os.path.getsize(self.path)
        try:
            _durable_append(self._handle, data)
        except BaseException:
            # a part of the line may be on the disk; whatever is there is
            # ended before the next entry, as a torn line found on
            # opening is, and the entry is not taken
            self._torn_tail = True
            self._end_torn()
            raise
        self._lines += 1
        self._admit(entry.line, stored)
        self._mirror(data, size)
        return self._by_id[entry.id]

    def _end_torn(self) -> None:
        try:
            # the header is always there, so the file is never empty
            if self._last_byte() != b'\n':
                _durable_append(self._handle, b'\n')
                self._lines += 1
            self._torn_tail = False
        except OSError:                              # pragma: no cover
            pass

    def _last_byte(self) -> bytes:
        with open(self.path, 'rb') as handle:
            handle.seek(-1, os.SEEK_END)
            return handle.read(1)

    # ---- the mirror -------------------------------------------------------------

    def _mirror(self, data: bytes, size_before: int) -> None:
        if self.mirror is None:
            return
        if self._mirror_in_step:
            try:
                if os.path.getsize(self.mirror) == size_before:
                    with open(self.mirror, 'ab') as handle:
                        _durable_append(handle, data)
                    return
            except OSError as error:
                self._mirror_failed(error)
                return
        self._copy_to_mirror()

    def _copy_to_mirror(self) -> None:
        # The whole log, written beside the mirror and moved into place,
        # so the mirror is never half a copy; this is also how a mirror
        # that missed entries (its share was away) catches up.
        assert self.mirror is not None
        staging = self.mirror + '.partial'
        try:
            with open(self.path, 'rb') as source, open(staging, 'wb') as copy:
                shutil.copyfileobj(source, copy)
                copy.flush()
                os.fsync(copy.fileno())
            os.replace(staging, self.mirror)
            self._mirror_in_step = True
            self.mirror_problem = ''
        except OSError as error:
            self._mirror_failed(error)

    def _mirror_failed(self, error: OSError) -> None:
        self._mirror_in_step = False
        self.mirror_problem = (f'the mirror at {self.mirror} was not written '
                               f'({error.strerror or error}); the entry is in '
                               f'the log at {self.path}')


def _same_path(a: str, b: str) -> bool:
    return (os.path.normcase(os.path.abspath(a))
            == os.path.normcase(os.path.abspath(b)))


def _sync_folder(path: str) -> None:
    # A new file's name is durable only once its folder is synced, on
    # the systems that allow it; Windows does not open a folder this way.
    if os.name == 'nt':
        return
    try:                                             # pragma: no cover
        fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    except OSError:                                  # pragma: no cover
        return
    try:                                             # pragma: no cover
        os.fsync(fd)
    except OSError:                                  # pragma: no cover
        pass
    finally:                                         # pragma: no cover
        os.close(fd)
