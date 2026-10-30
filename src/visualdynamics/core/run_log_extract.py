"""A run log brought into a project: one test, frozen at a stated entry.

A report holds references to project objects, never data that can move
under it, and a run log keeps growing while the test runs. So a log is
not referenced from a project; it is brought in as a snapshot, the way
`Photos` brings image files in. `take` reads one test of a `.vdlog` up
to a stated entry and keeps the lines of the file that test's views
read, exactly as written: the log's header, every entry of the test's
chain and of the chain between tests up to that entry, and the few
entries written elsewhere that answer something in the test (the next
test's opening, which closes it to runs; a cause, path, restart or
deviation that names one of its entries; a disposition of one of its
anomalies), each with its corrections.

`RunLogExtract` reads that snapshot as `RunLogModel` reads a log, so
its views (runs, interruptions, credited duration, findings, files,
chronology, holds, owed lists) are what the model gave for the test at
that entry, and nothing the file gains afterwards changes them. The
extract keeps the test's head hash, which the report's data index can
carry beside the log's other files. `RunLogExtract.status` compares
the live file with the snapshot: when the file has grown it says how
many newer entries the log holds, and a missing or changed file is
said in words, never raised.

**Refusals.** `take` refuses a file that is not a run log, a test the
log does not hold at the stated entry, an entry the log does not hold,
a chain of the test (or between tests) that does not verify, an entry
before the cut that the log cannot read, and, when no entry is stated,
a last line still torn (an entry being written). A line that is not an
entry at all (the fragment a crash leaves, ended by the next writer) is
skipped as the log itself skips it, and named in `problems`, since the
chain checks across it.

**Drafts.** The text an extract offers a report's author sections
(`draft_chronology`, `draft_findings`) is a `Draft`, marked as drafted:
it is offered, never placed, and the section stays owed until the
author accepts or edits it. Only the fact of a deviation or anomaly is
offered; its cause, effect and approval or disposition are the
author's. A log with nothing of a kind offers nothing for it rather
than "none", since an empty log cannot tell "nothing happened" from
"nobody wrote it down".
"""

from __future__ import annotations

import json
import ntpath
import os
import posixpath
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .run_log import (
    Entry,
    RunLog,
    RunLogError,
    Verification,
    chain_name,
)
from .run_log_model import (
    ChronologyRow,
    Credit,
    FileRow,
    Finding,
    Hold,
    Interruption,
    OwedList,
    Run,
    RunLogModel,
    Test,
    TestHeader,
)
from .runs import level_text

#: what a draft offered to a report's author section is marked with
#: until the author accepts or edits it
DRAFTED = 'DRAFTED FROM THE RUN LOG, NOT YET ACCEPTED'

#: what `RunLogExtract.status` can say of the live file
STATES = ('current', 'newer', 'missing', 'changed', 'unreadable')

#: the short words a badge shows for each state but 'current'
BADGES = {'newer': 'newer entries in the log', 'missing': 'log not found',
          'changed': 'log changed', 'unreadable': 'log unreadable'}

#: the report skeleton's sections the drafts are offered to
CHRONOLOGY_SECTION = 'chronology'
FINDINGS_SECTION = 'deviations'


@dataclass(frozen=True)
class LogStatus:
    """What the live log file says against an extract: its state (one of
    `STATES`), how many entries it holds after the extract's last entry
    (`newer`) and how many of those are of the extract's test
    (`newer_in_test`), and the state in words."""

    state: str
    newer: int
    newer_in_test: int
    message: str

    @property
    def badge(self) -> str:
        """The short words a badge shows: '' when the extract is
        current."""
        return BADGES.get(self.state, '')

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class Draft:
    """Text the run log offers one of a report's author sections.

    `section` is the skeleton section it is offered to, `text` the draft
    in the report's Markdown ('' when the log has nothing to offer),
    `source` which log, test and entry it was drafted from, `notes` what
    the author should know that the text does not say, and `mark` what
    it is shown with until the author accepts or edits it. A draft never
    fills a section: it is offered beside it, and the section stays owed
    until it is accepted. False when there is no text.
    """

    section: str
    text: str
    source: str
    notes: tuple[str, ...] = ()
    mark: str = DRAFTED

    def __bool__(self) -> bool:
        return bool(self.text)


def _cell(text: Any) -> str:
    return str(text).replace('|', '\\|').replace('\n', ' ')


def _is_object(raw: bytes) -> bool:
    """Whether a line of the file is one JSON object: a record the log
    could not read as an entry, as against a fragment."""
    try:
        return isinstance(json.loads(raw.decode('utf-8')), dict)
    except (UnicodeDecodeError, ValueError):
        return False


def _joined(raw: Sequence[bytes], lines: Sequence[int]) -> bytes:
    """The file's header line and its `lines` (numbered from 1), as the
    JSON Lines of a log holding only those entries."""
    return b'\n'.join([raw[0].rstrip(b'\r'),
                       *(raw[line - 1].rstrip(b'\r') for line in lines)]
                      ) + b'\n'


def _whole_number(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RunLogError(f'{what} is a whole number from 1, not {value!r}')
    return value


def take(path: str | os.PathLike, test: int | None = None,
         upto: int | None = None, *,
         taken: datetime | None = None) -> RunLogExtract:
    """One test of the run log at `path`, frozen at entry `upto`.

    Parameters
    ----------
    path : str or os.PathLike
        The `.vdlog`.
    test : int, optional
        The test, by number. May be left out when the log holds one
        test.
    upto : int, optional
        The id of the last entry the extract holds, from anywhere in the
        log; the log's last entry when omitted.
    taken : datetime, optional
        When the extract was taken, with its zone; now when omitted.

    Returns
    -------
    RunLogExtract
        The snapshot.

    Raises
    ------
    FileNotFoundError
        When there is no file at `path`.
    RunLogError
        When the file is not a run log or holds no entries, the log has
        no entry `upto`, or no test `test` by then (or several tests and
        `test` is omitted), an entry before the cut cannot be read, a
        chain the extract rests on does not verify (the message names
        the chain, the entry and the line), or, with `upto` omitted, the
        last line is torn.
    """
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(f'no run log at {path}')
    with open(path, 'rb') as handle:
        data = handle.read()
    log = RunLog.parse(data, path)
    raw = data.split(b'\n')
    entries = log.entries
    if not entries:
        raise RunLogError(f'{path} holds no entries yet; there is no test to '
                          'bring in')
    by_id = {entry.id: entry for entry in entries}
    if upto is None:
        cut = max(by_id)
        tail = raw[-1]
        if tail.strip() and not _is_object(tail):
            raise RunLogError(
                f'{path}, line {len(raw)}: the last line is torn, as an entry '
                'still being written (or a writer that stopped in the middle '
                'of one) leaves it. Give the last whole entry as upto, or '
                'wait for the entry to be written.')
    else:
        cut = _whole_number(upto, 'the last entry (upto)')
        if cut not in by_id:
            raise RunLogError(f'{path} has no entry #{cut}; its entries run '
                              f'from #{min(by_id)} to #{max(by_id)}')
    cut_line = by_id[cut].line
    read = {entry.line for entry in entries}
    skipped: list[str] = []
    for problem in log.problems:
        # a note on a line that was read (a last line with no line end)
        # costs the extract nothing
        if problem.line > cut_line or problem.line in read:
            continue
        if _is_object(raw[problem.line - 1]):
            raise RunLogError(
                f'{path}, line {problem.line}: an entry the log cannot read '
                f'({problem.reason}). It may be of the test, so the test '
                'cannot be brought in whole; correct the line or give an '
                'earlier entry as upto.')
        skipped.append(str(problem))
    kept = [entry for entry in entries if entry.id <= cut]
    before = RunLog.parse(_joined(raw, [entry.line for entry in kept]), path)
    model = RunLogModel(before)
    numbers = [found.number for found in model.tests()]
    if test is None:
        if len(numbers) != 1:
            raise RunLogError(
                f'{path} holds {len(numbers)} tests by entry #{cut}'
                + (f' ({", ".join(map(str, numbers))})' if numbers else '')
                + '; say which test to bring in')
        test = numbers[0]
    test = _whole_number(test, 'a test')
    if test not in numbers:
        held = (f'its tests are {", ".join(map(str, numbers))}' if numbers
                else 'it holds no test yet')
        raise RunLogError(f'{path} has no test {test} by entry #{cut}; '
                          f'{held}')
    found = model.test(test)
    if found.opened is None:
        raise RunLogError(f'{path}: test {test} has entries but was never '
                          'opened, so it has no kind or header to bring in')
    # the test's chain and the chain between tests, whole up to the cut,
    # so each verifies alone in the snapshot as it does in the file
    chosen = {entry.id for entry in kept if entry.test in (test, None)}
    view = before.view()
    of_test = {entry.id for entry in view if entry.test == test}
    chosen |= of_test                    # moved into the test by correction
    if found.closed is not None:
        # what closed it to runs: its own close, or the next opening
        chosen.add(found.closed.id)
    anomalies = {entry.fields['number'] for entry in view
                 if entry.id in of_test and entry.kind == 'anomaly_opened'
                 and 'number' in entry.fields}
    for entry in view:
        if entry.fields.get('refers_to') in of_test or (
                entry.kind == 'anomaly_dispositioned'
                and entry.fields.get('number') in anomalies):
            chosen.add(entry.id)
    chosen |= {entry.id for entry in kept if entry.kind == 'correction'
               and entry.corrects in chosen}
    extras = {entry_id for entry_id in chosen
              if by_id[entry_id].test not in (test, None)}
    own = {chain_name(test), chain_name(None)}
    check = log.verify(tests={test, None, *(by_id[entry_id].test
                                            for entry_id in extras)})
    for broken in check.breaks:
        if broken.line is None or broken.line > cut_line:
            continue
        if broken.chain in own or broken.entry in extras:
            raise RunLogError(f'{path}: the chain does not verify at '
                              f'{broken}')
    snapshot = _joined(raw, [by_id[entry_id].line
                             for entry_id in sorted(chosen)])
    later = [entry for entry in entries if entry.id > cut]
    return RunLogExtract(
        snapshot.decode('utf-8'), path=path, test=test, upto=cut,
        taken=taken if taken is not None else datetime.now().astimezone(),
        log_size=len(data), after=len(later),
        after_in_test=sum(entry.test == test for entry in later),
        problems=skipped)


class RunLogExtract:
    """One test of a run log, frozen at a stated entry, held in a project.

    Made by `take` (or `Project.attach_run_log`), and saved in the
    project file whole, so it reads the same wherever the project is
    opened, with or without the log beside it. Its views are the
    `RunLogModel` views of its test, read from the snapshot.

    Parameters
    ----------
    text : str
        The snapshot: the log's header line and the lines of the entries
        kept, as JSON Lines.
    path : str
        Where the log was, absolute (or as stored, when the project file
        it was resolved against is not known).
    test : int
        The test.
    upto : int
        The id of the last entry of the log the extract was taken to.
    taken : datetime or str
        When it was taken, with its zone (ISO 8601 text as stored).
    log_size : int
        The log file's size in bytes when it was taken.
    after, after_in_test : int, optional
        How many entries the file held after `upto` when it was taken,
        and how many of those were of the test.
    problems : sequence of str, optional
        The lines of the log before `upto` that were skipped as not
        entries.

    Raises
    ------
    RunLogError
        When `text` is not a run log, does not hold the test, or holds
        an entry past `upto`.
    """

    def __init__(self, text: str, *, path: str, test: int, upto: int,
                 taken: datetime | str, log_size: int, after: int = 0,
                 after_in_test: int = 0,
                 problems: Sequence[str] = ()) -> None:
        if not isinstance(text, str):
            raise RunLogError(f'an extract\'s snapshot is text, not '
                              f'{type(text).__name__}')
        self.text: str = text
        self.path: str = str(path)
        self.test: int = _whole_number(test, 'a test')
        self.upto: int = _whole_number(upto, 'the last entry (upto)')
        if isinstance(taken, str):
            taken = datetime.fromisoformat(taken)
        if not isinstance(taken, datetime) or taken.tzinfo is None:
            raise RunLogError(f'when an extract was taken is a time with its '
                              f'zone, not {taken!r}')
        self.taken: datetime = taken
        self.log_size: int = int(log_size)
        self.after: int = int(after)
        self.after_in_test: int = int(after_in_test)
        self.problems: tuple[str, ...] = tuple(str(p) for p in problems)
        #: the snapshot read as a log, read-only
        self.log: RunLog = RunLog.parse(text.encode('utf-8'), self.path)
        #: the snapshot read as tests, axes and runs
        self.model: RunLogModel = RunLogModel(self.log)
        past = [entry.id for entry in self.log.entries if entry.id > self.upto]
        if past:
            raise RunLogError(f'an extract to entry #{self.upto} holds entry '
                              f'#{past[0]}')
        self.model.test(self.test)               # refuses a test not held

    # ---- what it is ---------------------------------------------------------

    def __repr__(self) -> str:
        return (f'RunLogExtract(test {self.test} of '
                f'{self.log.header.title or os.path.basename(self.path)!r}, '
                f'to entry #{self.upto})')

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RunLogExtract):
            return NotImplemented
        return self._key() == other._key()

    __hash__ = None                                      # type: ignore[assignment]

    def _key(self) -> tuple:
        return (self.text, self.path, self.test, self.upto, self.taken,
                self.log_size, self.after, self.after_in_test, self.problems)

    @property
    def log_id(self) -> str:
        """The log's id, from its header: the same log wherever it is."""
        return self.log.header.log_id

    @property
    def log_title(self) -> str:
        """The log's title (the article or program it is kept for)."""
        return self.log.header.title

    @property
    def head(self) -> str:
        """The head of the test's chain at the extract's last entry: the
        hash of the test's last entry by then. A report's data index
        carries it, so the record outside the log anchors the record
        inside it (`RunLog.verify(heads=...)`)."""
        return self.log.head(self.test)

    @property
    def info(self) -> Test:
        """The test as the snapshot holds it: kind, state, header, axes,
        runs and entries."""
        return self.model.test(self.test)

    @property
    def kind(self) -> str:
        """The test's kind (`run_log_model.TEST_KINDS`)."""
        return self.info.kind

    @property
    def title(self) -> str:
        """The test's title, from its header."""
        return self.info.title

    @property
    def state(self) -> str:
        """The test's state at the extract's last entry: 'open', 'closed
        to runs' or 'released'."""
        return self.info.state

    @property
    def header(self) -> TestHeader:
        """The test's header as it stood at the extract's last entry."""
        return self.info.header

    @property
    def entries(self) -> tuple[Entry, ...]:
        """The test's entries, corrected, in the order written."""
        return self.info.entries

    def verify(self) -> Verification:
        """Check the snapshot's chains (the test's and the one between
        tests), as `RunLog.verify` checks a log holding one test."""
        return self.log.verify(tests=[self.test, None])

    # ---- the views ------------------------------------------------------------

    def runs(self) -> list[Run]:
        """The test's runs (`RunLogModel.runs`)."""
        return self.model.runs(self.test)

    def interruptions(self) -> list[Interruption]:
        """The test's aborts, trips and scheduled stops, with the cause,
        path and restart answering each (`RunLogModel.interruptions`)."""
        return self.model.interruptions(self.test)

    def credited(self) -> list[Credit]:
        """The time credited on each axis against the duration and its
        tolerance (`RunLogModel.credited`)."""
        return self.model.credited(self.test)

    def findings(self) -> list[Finding]:
        """The test's deviations and anomalies by number, with the
        dispositions against each (`RunLogModel.findings`)."""
        return self.model.findings(self.test)

    def open_anomalies(self) -> list[Finding]:
        """The test's anomalies with no disposition, with their owners."""
        return self.model.open_anomalies(self.test)

    def files(self) -> list[FileRow]:
        """Every file the test's entries name, by SHA-256
        (`RunLogModel.files`)."""
        return self.model.files(self.test)

    def chronology(self) -> list[ChronologyRow]:
        """The test's entries in the order of their events, each with its
        time on the recorders' clock (`RunLogModel.chronology`)."""
        return self.model.chronology(self.test)

    def holds(self) -> list[Hold]:
        """The test's holds still unanswered at the extract's last
        entry."""
        return self.model.holds(self.test)

    def owed(self) -> list[OwedList]:
        """The test's owed lists with a line still owed at the extract's
        last entry."""
        return self.model.owed(self.test)

    def late(self) -> list[Entry]:
        """The test's entries written down late."""
        return self.model.late(self.test)

    # ---- the live file --------------------------------------------------------

    def status(self, path: str | os.PathLike | None = None) -> LogStatus:
        """The live log against the extract.

        The file's length is compared first; it is read only when its
        length differs from the length when the extract was taken. It is
        then 'newer' when it holds entries after the extract's last
        entry (and they are counted), 'changed' when it is another log
        or no longer holds the test's head (cut short or edited since),
        'unreadable' when it is no longer a run log this build reads,
        and 'missing' when there is no file. None of these is an error:
        the extract stands as it was taken.

        Parameters
        ----------
        path : str or os.PathLike, optional
            Where to look for the log; the extract's own path when
            omitted.

        Returns
        -------
        LogStatus
            The state, the newer entries counted, and the state in words.
        """
        where = self.path if path is None else os.fspath(path)
        frozen = f'the extract holds test {self.test} as it stood at entry ' \
                 f'#{self.upto}'
        try:
            size = os.path.getsize(where)
        except OSError:
            return LogStatus('missing', 0, 0,
                             f'the log is not at {where}; {frozen}')
        if size == self.log_size:
            return self._counted(self.after, self.after_in_test, frozen)
        try:
            live = RunLog.read(where)
        except (OSError, RunLogError) as error:
            return LogStatus('unreadable', 0, 0, f'{error}; {frozen}')
        if live.header.log_id != self.log_id:
            return LogStatus('changed', 0, 0,
                             f'{where} is another run log (its id is not this '
                             f'log\'s); {frozen}')
        if self.head not in {entry.hash for entry in live.entries
                             if entry.test == self.test}:
            return LogStatus(
                'changed', 0, 0,
                f'{where} no longer holds the last entry of test {self.test} '
                'the extract was taken to: it was cut short or changed since; '
                f'{frozen}')
        later = [entry for entry in live.entries if entry.id > self.upto]
        return self._counted(len(later),
                             sum(entry.test == self.test for entry in later),
                             frozen)

    def _counted(self, newer: int, in_test: int, frozen: str) -> LogStatus:
        if not newer:
            return LogStatus('current', 0, 0,
                             f'the extract is current: the log holds nothing '
                             f'after entry #{self.upto}')
        return LogStatus(
            'newer', newer, in_test,
            f'newer entries in the log: {newer} after entry #{self.upto}, '
            f'{in_test} of them of test {self.test}; {frozen}. Refresh to '
            'take them.')

    # ---- drafts for the author's sections ------------------------------------

    def _source(self) -> str:
        title = f' {self.log_title!r}' if self.log_title else ''
        return (f'the run log{title}, test {self.test}, to entry #{self.upto} '
                f'(test head {self.head[:12]})')

    def _time(self, entry: Entry) -> tuple[str, bool]:
        """An entry's time as the draft prints it, on the recorders' clock
        when a clock check comes before it, and whether it was."""
        recorder = self.log.recorder_time(entry)
        when = recorder if recorder is not None else entry.event_time
        return f'{when:%Y-%m-%d %H:%M:%S}', recorder is not None

    def draft_chronology(self) -> Draft:
        """The test's chronology from the log, as draft text for the
        report's Chronology section: its runs in order with their axis,
        start, what each held, the counter at the stop and the outcome;
        its interruptions with the cause, path and restart logged; and
        the time credited on each axis.

        Draft text only: it is offered beside the section and marked as
        drafted (`DRAFTED`), and the section stays owed until the author
        accepts or edits it. A test with no runs offers nothing.

        Returns
        -------
        Draft
            The draft for section 'chronology'.
        """
        runs = sorted(self.runs(), key=lambda run: run.entries[0].id)
        notes: list[str] = []
        if not runs:
            return Draft(CHRONOLOGY_SECTION, '', self._source(),
                         ((f'The log holds no runs of test {self.test} by '
                          f'entry #{self.upto}.'),))
        computer = False
        lines = ['| Run | Axis | Started | Held | Counter | Outcome |',
                 '|---|---|---|---|---|---|']
        for run in runs:
            started, on_recorders = self._time(run.entries[0])
            computer |= not on_recorders
            lines.append('| ' + ' | '.join(_cell(cell) for cell in (
                run.number, run.axis or '', started, _held(run),
                run.counter or '', run.outcome)) + ' |')
        stops = self.interruptions()
        if stops:
            lines += ['', '| Entry | Time | Stop | Cause | Path | Restart |',
                      '|---|---|---|---|---|---|']
            for stop in stops:
                when, on_recorders = self._time(stop.entry)
                computer |= not on_recorders
                lines.append('| ' + ' | '.join(_cell(cell) for cell in (
                    f'#{stop.entry.id}', when, _stop(stop),
                    stop.cause.fields.get('cause') if stop.cause
                    else 'not logged',
                    stop.path.fields.get('path') if stop.path
                    else 'not logged', _restart(stop))) + ' |')
        else:
            notes.append(f'The log holds no abort, trip or scheduled stop of '
                         f'test {self.test}; whether the test ran without '
                         'one is the author\'s to say.')
        credits = self.credited()
        if credits:
            lines.append('')
            lines += [f'- Axis {credit.axis}: {credit.message}'
                      if credit.axis else f'- {credit.message}'
                      for credit in credits]
        clock = ('Times are on the log computer\'s clock where no clock check '
                 'comes before them, and on the recorders\' clock otherwise.'
                 if computer else 'Times are on the recorders\' clock.')
        lines += ['', (f'Drafted from {self._source()}. {clock} The outcome '
                  'of each run and the cause of each stop are as the log '
                  'records them; what they mean for the test is the '
                  'author\'s to say.')]
        return Draft(CHRONOLOGY_SECTION, '\n'.join(lines), self._source(),
                     tuple(notes))

    def draft_findings(self) -> Draft:
        """The test's deviations and anomalies from the log, as draft text
        for the report's deviations section: each by number with its
        fact as typed at the console, an anomaly's owner, and whether a
        disposition is logged.

        Only the fact is offered. The cause, the effect and the approval
        or disposition are the author's, and a log with no findings
        offers nothing, never "none": the author sets the section to
        None in one step when that is so.

        Returns
        -------
        Draft
            The draft for section 'deviations'.
        """
        found = self.findings()
        if not found:
            return Draft(FINDINGS_SECTION, '', self._source(), (
                (f'The log holds no deviation or anomaly of test {self.test} '
                f'by entry #{self.upto}; whether there were none is the '
                'author\'s to say.'),))
        lines = []
        for finding in found:
            parts = [(f'**{finding.number}**, {finding.kind} '
                     f'(log entry #{finding.entry.id})')]
            if finding.kind == 'anomaly':
                if finding.owner:
                    parts.append(f'owner {finding.owner}')
                parts.append('open' if finding.open else
                             'a disposition is logged ('
                             + ', '.join(f'#{entry.id}'
                                         for entry in finding.dispositions)
                             + ')')
            fact = finding.fact or 'not logged'
            lines.append(f'- {", ".join(parts)}. Fact: {fact}')
        lines += ['', (f'Drafted from {self._source()}. The fact of each is as '
                  'typed at the console; its cause, effect and approval or '
                  'disposition are the author\'s.')]
        return Draft(FINDINGS_SECTION, '\n'.join(lines), self._source())


def _held(run: Run) -> str:
    """What a run held, in a few words: its levels, sweeps or shots, or a
    survey's result."""
    parts = []
    levels = [entry.fields['level_db'] for entry in run.levels
              if 'level_db' in entry.fields]
    if levels:
        parts.append('levels ' + ', '.join(level_text(level)
                                           for level in levels))
    for count, word in ((len(run.sweeps), 'sweep'), (len(run.shots), 'shot')):
        if count:
            parts.append(f'{count} {word}{"s" * (count != 1)}')
    if run.result:
        parts.append(f'result: {run.result}')
    return '; '.join(parts)


def _stop(stop: Interruption) -> str:
    if stop.kind == 'scheduled':
        return 'scheduled stop'
    return str(stop.entry.fields.get('stop', stop.kind))


def _restart(stop: Interruption) -> str:
    restart = stop.restart
    if restart is None:
        return 'none logged'
    resume = restart.fields.get('resume')
    counter = restart.fields.get('counter_at_restart')
    words = ('resumed' if resume is True else 'started again'
             if resume is False else 'restarted')
    return f'{words} at {counter}' if counter else words


def _absolute(path: str) -> bool:
    # absolute on either platform: a project saved on one and opened on
    # the other keeps its log's absolute path as an absolute path (not
    # found, and said so), never one joined onto the project's folder
    return ntpath.isabs(path) or posixpath.isabs(path)


def stored_path(path: str, project_file: str | None) -> str:
    """The log's path as a project file stores it: relative to the
    project file's folder, with forward slashes, when the log is in
    that folder or one beneath it, and absolute otherwise (or when the
    project file is not known).

    Parameters
    ----------
    path : str
        The log's path.
    project_file : str or None
        The project file the path is stored in.

    Returns
    -------
    str
        The path to store.
    """
    if not _absolute(path):
        # a relative path that was never resolved (the project file it
        # was stored against was not known): kept as it was stored
        return path.replace(os.sep, '/')
    absolute = os.path.abspath(path)
    if not project_file:
        return absolute
    folder = os.path.dirname(os.path.abspath(project_file))
    try:
        relative = os.path.relpath(absolute, folder)
    except ValueError:                       # another drive
        return absolute
    if relative == os.pardir or relative.startswith(os.pardir + os.sep) \
            or os.path.isabs(relative):
        return absolute
    return relative.replace(os.sep, '/')


def resolved_path(stored: str, project_file: str | None) -> str:
    """A stored log path made absolute against the project file it was
    stored in; an absolute path, or a relative one with no project file
    known, is returned as stored.

    Parameters
    ----------
    stored : str
        The path as the project file stores it.
    project_file : str or None
        The project file.

    Returns
    -------
    str
        The path to look for the log at.
    """
    if _absolute(stored) or not project_file:
        return stored
    folder = os.path.dirname(os.path.abspath(project_file))
    return os.path.normpath(os.path.join(folder, *stored.split('/')))
