"""Views of a run log for other readers: plain text, CSV, and the
machine log.

The `.vdlog` file is its own JSON export, and it is the one format the
log is read back from. These views are written out only:

- **Plain text** (`to_text`): every entry in the order written, for
  printing or attaching to a report. A corrected entry is marked with
  the corrections that changed it, and each correction says what it
  changed and what the value was; a late entry is marked with when it
  was entered. The chains' heads and what `RunLog.verify` found close
  it.
- **CSV** (`to_csv`): one row per entry as stored, corrections included,
  with the corrected-by and corrects columns, a late column, the time
  on the recorders' clock, and the fields as JSON.
- **The machine log** (`machine_log_csv`): the entries a lab's own
  machine log keeps (trips and aborts, setting changes, clock checks,
  people in and out, restarts, and the tests opened and closed, which
  say what was running), corrected, one row each, in columns a
  spreadsheet or a machine log takes as they are.

Each takes `test`: 'all' (the whole log, the default), a test's number,
or None for the entries between tests.
"""

from __future__ import annotations

import csv
import io
import json
import os
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from .run_log import KINDS, LATE_AFTER, Entry, RunLog, RunLogError

#: the `test` that selects the whole log
ALL = 'all'

#: the views `export` writes, by name, with the extension each is given
#: when the path has none
VIEWS = {'text': '.txt', 'csv': '.csv', 'machine': '.csv'}

#: the entry CSV's columns, in order
CSV_COLUMNS = ('id', 'kind', 'label', 'event_time', 'recorder_time',
               'entered_at', 'late', 'author', 'login', 'computer', 'test',
               'axis', 'run', 'fields', 'text', 'files', 'corrects',
               'corrected_by', 'hash')

#: the machine log's columns, in order
MACHINE_COLUMNS = ('date', 'time', 'utc_offset', 'recorder_time', 'event',
                   'detail', 'test', 'test_title', 'axis', 'run', 'author',
                   'entry', 'corrected')


def _selected(entries: list[Entry], test: Any) -> list[Entry]:
    if test == ALL:
        return entries
    if test is not None and (isinstance(test, bool)
                             or not isinstance(test, int) or test < 1):
        raise RunLogError(f'a test is selected by its number from 1, by None '
                          f'for the entries between tests, or by {ALL!r}, '
                          f'not {test!r}')
    return [entry for entry in entries if entry.test == test]


def _value_text(value: Any) -> str:
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, list):
        return ', '.join(str(item) for item in value)
    return str(value)


def _fields_text(entry: Entry, fields: Mapping[str, Any]) -> str:
    kind = KINDS.get(entry.kind)
    order = list(kind.fields) if kind is not None else []
    order += sorted(name for name in fields if name not in order)
    return '; '.join(f'{name}: {_value_text(fields[name])}'
                     for name in order if name in fields)


def _clock(when: datetime) -> str:
    offset = when.strftime('%z')
    return (f'{when.strftime("%Y-%m-%d %H:%M:%S")} '
            f'{offset[:3]}:{offset[3:]}')


def _where(entry: Entry) -> str:
    parts = []
    if entry.test is not None:
        parts.append(f'test {entry.test}')
    if entry.axis is not None:
        parts.append(f'axis {entry.axis}')
    if entry.run is not None:
        parts.append(f'run {entry.run}')
    return ' / '.join(parts) or 'log'


def _correction_text(log: RunLog, correction: Entry) -> str:
    """What a correction changed, each value with the one it replaced."""
    target = correction.corrects
    try:
        before = log.entry(target)
    except KeyError:
        return f'Correction of #{target} (not in the log): ' + '; '.join(
            f'{name}: {_value_text(value)}'
            for name, value in correction.fields.items())
    # the entry as it stood just before this correction
    state: dict[str, Any] = dict(before.fields)
    for name in ('event_time', 'text', 'test', 'axis', 'run'):
        value = getattr(before, name)
        state[name] = (value.isoformat(timespec='milliseconds')
                       if isinstance(value, datetime) else value)
    for earlier in log.corrections_of(target):
        if earlier.id >= correction.id:
            break
        for name, value in earlier.fields.items():
            state[name] = value
    changes = []
    for name, value in correction.fields.items():
        was = state.get(name)
        was_text = 'not given' if was in (None, '') else _value_text(was)
        now_text = 'struck' if value is None else _value_text(value)
        changes.append(f'{name}: {now_text} (was {was_text})')
    return f'Correction of #{target}: ' + '; '.join(changes)


def to_text(log: RunLog, *, test: int | str | None = ALL,
            late_after: timedelta = LATE_AFTER) -> str:
    """The log as plain text, every entry in the order written.

    Parameters
    ----------
    log : RunLog
        The log.
    test : int, None or 'all', optional
        Which entries: the whole log, one test, or (None) those between
        tests.
    late_after : timedelta, optional
        How long after its event an entry is entered to be marked late.

    Returns
    -------
    str
        The text, a line or more per entry, then the lines that could
        not be read, the chains' heads, and what verifying found.
    """
    entries = _selected(list(log.entries), test)
    header = log.header
    lines = [f'Run log: {header.title or "(untitled)"}',
             (f'Log {header.log_id}, run log format {header.version}, created '
             f'{_clock(header.created)}'),
             f'{len(entries)} entr{"y" if len(entries) == 1 else "ies"}'
             + ('' if test == ALL else
                f', {"between tests" if test is None else f"test {test}"}'),
             '']
    for entry in entries:
        recorder = log.recorder_time(entry)
        lines.append(f'#{entry.id}  {_clock(entry.event_time)}  {entry.author}'
                     f' ({entry.login or "?"} on {entry.computer or "?"})  '
                     f'{_where(entry)}')
        if entry.kind == 'correction':
            body = _correction_text(log, entry)
        else:
            fields = _fields_text(entry, entry.fields)
            body = entry.label + (f': {fields}' if fields else '')
        lines.append(f'    {body}')
        if entry.text:
            lines.append(f'    "{entry.text}"')
        for ref in entry.files:
            size = '' if ref.size is None else f', {ref.size} bytes'
            lines.append(f'    file: {ref.name} (SHA-256 {ref.sha256}{size})')
        marks = []
        if recorder is not None:
            marks.append(f'recorders\' clock {_clock(recorder)}')
        if entry.is_late(late_after):
            marks.append(f'LATE: entered {_clock(entry.entered_at)}')
        corrections = log.corrections_of(entry.id)
        if corrections:
            marks.append('CORRECTED by '
                         + ', '.join(f'#{c.id}' for c in corrections))
        if marks:
            lines.append('    [' + '] ['.join(marks) + ']')
    if log.problems:
        lines += ['', 'Lines not read:']
        lines += [f'  {problem}' for problem in log.problems]
    chains = log.chains()
    if test != ALL:
        chains = [name for name in chains
                  if name == ('log' if test is None else f'test {test}')]
    if chains:
        lines += ['', 'Chain heads (SHA-256):']
        for name in chains:
            number = None if name == 'log' else int(name.split()[1])
            lines.append(f'  {name}: {log.head(number)}')
    lines += ['', f'Integrity: {log.verify()}']
    return '\n'.join(lines) + '\n'


def _csv(rows: list[list[Any]], columns: tuple[str, ...]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue()


def _optional(value: Any) -> Any:
    return '' if value is None else value


def to_csv(log: RunLog, *, test: int | str | None = ALL,
           late_after: timedelta = LATE_AFTER) -> str:
    """The log as CSV: a header row of `CSV_COLUMNS`, then one row per
    entry as stored, corrections included.

    Parameters
    ----------
    log : RunLog
        The log.
    test : int, None or 'all', optional
        Which entries: the whole log, one test, or (None) those between
        tests.
    late_after : timedelta, optional
        How long after its event an entry is entered to be marked late.

    Returns
    -------
    str
        The CSV text. Times are ISO 8601 with their UTC offsets; `late`
        is 'late' or blank; `fields` is the entry's fields as JSON;
        `files` lists each file as its name and SHA-256; a corrected
        entry lists its corrections in `corrected_by`, and a correction
        names its entry in `corrects`.
    """
    rows = []
    for entry in _selected(list(log.entries), test):
        recorder = log.recorder_time(entry)
        rows.append([
            entry.id, entry.kind, entry.label,
            entry.event_time.isoformat(timespec='milliseconds'),
            '' if recorder is None else recorder.isoformat(
                timespec='milliseconds'),
            entry.entered_at.isoformat(timespec='milliseconds'),
            'late' if entry.is_late(late_after) else '',
            entry.author, entry.login, entry.computer,
            _optional(entry.test), _optional(entry.axis),
            _optional(entry.run),
            json.dumps(dict(entry.fields), ensure_ascii=False,
                       sort_keys=True),
            entry.text,
            '; '.join(f'{ref.name} {ref.sha256}' for ref in entry.files),
            _optional(entry.corrects),
            ' '.join(str(c.id) for c in log.corrections_of(entry.id)),
            entry.hash])
    return _csv(rows, CSV_COLUMNS)


def machine_log_csv(log: RunLog, *, test: int | str | None = ALL) -> str:
    """The trips and machine events of the log, for the lab's own
    machine log: a header row of `MACHINE_COLUMNS`, then one row per
    entry of a kind `KINDS` marks `machine`, in the corrected view.

    Parameters
    ----------
    log : RunLog
        The log.
    test : int, None or 'all', optional
        Which entries: the whole log, one test, or (None) those between
        tests.

    Returns
    -------
    str
        The CSV text: the event's local date, time and UTC offset, its
        time on the recorders' clock where a clock check is in force,
        the event in words ('Abort or trip'), its details, the test
        that was running and its title, the axis and run, the author,
        the entry's id, and 'corrected' when a correction changed it.
    """
    view = log.view()
    titles: dict[int, str] = {}
    for entry in view:
        if entry.kind == 'test_opened' and entry.test is not None:
            titles[entry.test] = str(entry.fields.get('title', ''))
    rows = []
    for entry in _selected(view, test):
        kind = KINDS.get(entry.kind)
        if kind is None or not kind.machine:
            continue
        recorder = log.recorder_time(entry)
        offset = entry.event_time.strftime('%z')
        detail = _fields_text(entry, entry.fields)
        if entry.text:
            detail = f'{detail}; {entry.text}' if detail else entry.text
        rows.append([
            entry.event_time.strftime('%Y-%m-%d'),
            entry.event_time.strftime('%H:%M:%S'),
            f'{offset[:3]}:{offset[3:]}',
            '' if recorder is None else recorder.isoformat(
                timespec='milliseconds'),
            kind.label, detail, _optional(entry.test),
            titles.get(entry.test, '') if entry.test is not None else '',
            _optional(entry.axis), _optional(entry.run), entry.author,
            entry.id, 'corrected' if entry.corrected_by else ''])
    return _csv(rows, MACHINE_COLUMNS)


def export(log: RunLog, path: str | os.PathLike, view: str = 'text', *,
           test: int | str | None = ALL) -> str:
    """Write one view of the log to `path`, in UTF-8, and return the
    path written.

    Parameters
    ----------
    log : RunLog
        The log.
    path : str or os.PathLike
        Where to write; given the view's extension (`VIEWS`) when it has
        none.
    view : str, optional
        'text' (`to_text`), 'csv' (`to_csv`) or 'machine'
        (`machine_log_csv`).
    test : int, None or 'all', optional
        Which entries.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    RunLogError
        For a view that is not one of `VIEWS`, or a path that is the
        log itself.
    """
    if view not in VIEWS:
        raise RunLogError(f'{view!r} is not a view of a run log; the views '
                          'are: ' + ', '.join(VIEWS))
    path = os.fspath(path)
    if not os.path.splitext(path)[1]:
        path += VIEWS[view]
    if os.path.normcase(os.path.abspath(path)) == os.path.normcase(
            os.path.abspath(log.path)):
        raise RunLogError('an export is written beside the log, never over it')
    if view == 'text':
        text = to_text(log, test=test)
    elif view == 'csv':
        text = to_csv(log, test=test)
    else:
        text = machine_log_csv(log, test=test)
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        handle.write(text)
    return path
