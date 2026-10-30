"""The Run Log window: a dock in the main window where the log keeper
writes the run log while a test runs.

The log keeper is a named person who is not at the controller, so the
dock is built to be read across a room and typed into fast. It floats
to a second monitor and remembers where it was put, and
`visualdynamics --run-log [file]` opens the main window with the dock
filling it and the analysis panes hidden (`MainWindow.set_log_only`),
for a lab that analyzes its data elsewhere. Nothing here drives the
controller: every check is a reminder, never an interlock.

What it shows, in large type:

- **The status strip**: the current test, its axis, run and level,
  whether the run is running or stopped (in words), the time since the
  run started, and the holds, counted, with the list a click away. A
  log open read-only says who holds it, and a mirror that could not be
  written says so.
- **The tests** as a strip of tabs, the whole log first, so an earlier
  test is a click away.
- **The day** as a list in the order written, with a line for each
  date: a corrected entry is shown struck through with what corrected
  it beneath, and a late entry is marked with when it was entered.
- **The owed lists**, shown and never enforced.

What it takes:

- **The level row**: one entry per step, its level offered from the
  header's step schedule, then the voltage, current and drive fraction
  in tab order, the analyst's answer with answers by channel, and the
  go; Enter writes it. The projection to full level (`run_log_model.project`)
  is shown live as the row is typed, a check that cannot be made saying
  why.
- **The abort**, one keystroke (Ctrl+Shift+A) that takes the event time
  at the key and opens a form needing only the kind of stop and the
  author's initials; Enter writes the run stop and the abort together,
  and what the first minute still owes is listed.
- **Small forms** for the rest: a new or opened log, a test opened, the
  roll to the next test, a test closed or released, a clock check, a
  person in or out, a note, a run started or stopped, a cause found and
  a correction. Each takes its event time when it opens.

Every read and write of the file, and every rebuild of the model from
it, runs on one worker thread of its own (`LogWorker`), in order, and
never gates input: the keeper keeps typing while an entry is synced, a
mirror is copied or a file is hashed. The window only ever shows a
`Snapshot`, which the worker builds after each change.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any

from PySide6.QtCore import QByteArray, QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import run_log_model as model_module
from ..core.run_log import (
    ANSWERS,
    FIELD_TYPES,
    KINDS,
    STOPS,
    SUFFIX,
    Entry,
    RunLog,
    RunLogError,
    check_fields,
)
from ..core.run_log_export import ALL, _correction_text, _fields_text, _where
from ..core.run_log_model import (
    LEVEL_MATCH_DB,
    TEST_KINDS,
    Hold,
    OwedList,
    RunLogModel,
)

#: how much larger than the application's own type the dock is drawn:
#: the list and the entry, and the status strip larger again, read from
#: across the room where the keeper sits
TYPE_SCALE = 1.3
STRIP_SCALE = 1.6

#: the preference keys under which the dock's place is remembered
PLACE_FLOATING = 'run_log/floating'
PLACE_AREA = 'run_log/area'
PLACE_GEOMETRY = 'run_log/geometry'

#: the keys of the log's acts. Each has a modifier, so none is taken
#: from a text field being typed into (a bare letter would be)
KEYS = {
    'abort': 'Ctrl+Shift+A',
    'run start': 'Ctrl+Shift+R',
    'run stop': 'Ctrl+Shift+S',
    'note': 'Ctrl+Shift+N',
    'clock check': 'Ctrl+Shift+K',
    'person in': 'Ctrl+Shift+I',
    'person out': 'Ctrl+Shift+O',
    'cause found': 'Ctrl+Shift+C',
    'level row': 'Ctrl+Shift+L',
}


def _now() -> datetime:
    return datetime.now().astimezone()


def _clock(when: datetime) -> str:
    return when.strftime('%H:%M:%S')


def _span(seconds: float) -> str:
    return model_module.clock_text(max(seconds, 0.0))


# ---- what the window shows -----------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One entry of the day: as corrected, as written, what each of its
    corrections changed in words, and whether it was entered late."""

    entry: Entry
    stored: Entry
    corrections: tuple[str, ...]
    late: bool

    @property
    def corrected(self) -> bool:
        """Whether a correction changed the entry."""
        return bool(self.corrections)


@dataclass(frozen=True)
class TestTab:
    """A test as the strip of tests names it: its number, title, kind and
    state."""

    number: int
    title: str
    kind: str
    state: str


@dataclass(frozen=True)
class Status:
    """What the status strip reads: the current test (None when no test
    is open to runs), its title and kind, the axis, run and level of its
    latest run, that run's outcome ('running', 'aborted', 'stopped',
    'completed', or '' with no run yet), and when it started and
    stopped."""

    test: int | None = None
    title: str = ''
    kind: str = ''
    axis: str | None = None
    run: int | None = None
    level_db: float | None = None
    outcome: str = ''
    started: datetime | None = None
    stopped: datetime | None = None


@dataclass(frozen=True)
class Snapshot:
    """A log as the window shows it, built on the worker after each
    change (`snapshot`), so nothing on the window's thread reads the
    file or rebuilds the model.

    `writable` is False for a log opened read-only, `read_only` saying
    why (who holds it). `rows` is the corrected view with what each
    correction changed; `holds` and `owed` are the whole log's; `header`
    is the current test's header as it stands. `run` is the run a level
    row goes into (the latest, while it runs, otherwise the next) and
    `levels` the levels already written in it. `open_aborts` are the
    aborts with no cause, `unreleased` the tests closed to runs and not
    yet released.
    """

    path: str
    title: str
    writable: bool
    read_only: str
    mirror: str | None
    mirror_problem: str
    problems: tuple[str, ...]
    rows: tuple[Row, ...]
    tests: tuple[TestTab, ...]
    status: Status
    header: Mapping[str, Any]
    holds: tuple[Hold, ...]
    owed: tuple[OwedList, ...]
    run: int | None
    levels: tuple[float, ...]
    open_aborts: tuple[int, ...]
    unreleased: tuple[int, ...]


def snapshot(log: RunLog, *, read_only: str = '') -> Snapshot:
    """The window's view of `log`: its model read once (`RunLogModel`)
    and everything the dock shows taken from it, in plain values.

    Run on the log's worker, never on the window's thread: it reads the
    whole corrected view and every hold and owed rule.
    """
    model = RunLogModel(log)
    rows = tuple(
        Row(entry, log.entry(entry.id),
            tuple(f'{_correction_text(log, correction)} (#{correction.id}, '
                  f'{correction.author}, {_clock(correction.event_time)})'
                  for correction in log.corrections_of(entry.id)),
            entry.is_late())
        for entry in model.entries)
    tests = tuple(TestTab(test.number, test.title, test.kind, test.state)
                  for test in model.tests())
    problems = [str(problem) for problem in log.problems]
    for test in model.tests():
        problems.extend(test.problems)
    current = model.current
    status = Status()
    header: Mapping[str, Any] = {}
    run_number = None
    levels: tuple[float, ...] = ()
    if current is not None:
        header = dict(current.header.values)
        latest = max(current.runs, key=lambda run: run.number, default=None)
        if latest is None:
            status = Status(current.number, current.title, current.kind,
                            current.axes[-1] if current.axes else None)
            run_number = model.next_run(current.number)
        else:
            level = None
            for entry in latest.entries:
                if isinstance(entry.fields.get('level_db'), (int, float)):
                    level = float(entry.fields['level_db'])
            running = latest.outcome == 'running'
            stop = latest.stop
            status = Status(current.number, current.title, current.kind,
                            latest.axis, latest.number, level, latest.outcome,
                            latest.entries[0].event_time,
                            None if running or stop is None
                            else stop.event_time)
            run_number = (latest.number if running
                          else model.next_run(current.number))
            if running:
                levels = tuple(float(entry.fields['level_db'])
                               for entry in latest.levels
                               if 'level_db' in entry.fields)
    holds = tuple(model.holds())
    return Snapshot(
        path=log.path, title=log.header.title, writable=log.writable,
        read_only=read_only, mirror=log.mirror,
        mirror_problem=log.mirror_problem, problems=tuple(problems),
        rows=rows, tests=tests, status=status, header=header, holds=holds,
        owed=tuple(model.owed()), run=run_number, levels=levels,
        open_aborts=tuple(hold.entry.id for hold in holds
                          if hold.rule == 'abort-cause'),
        unreleased=tuple(test.number for test in model.tests()
                         if test.state == 'closed to runs'))


def entry_line(entry: Entry) -> str:
    """An entry on one line, as the day list shows it: its time, kind,
    place, fields as the plain text export writes them, free text and
    author."""
    parts = [_clock(entry.event_time), entry.label, _where(entry)]
    fields = _fields_text(entry, entry.fields)
    if fields:
        parts.append(fields)
    if entry.text:
        parts.append(f'"{entry.text}"')
    parts.append(entry.author)
    return '   '.join(parts)


def next_level(steps: Any, written: Sequence[float]) -> float | None:
    """The level a level row offers: the first step of the header's step
    schedule above the last level written in the run, the first step
    when none is, the last written when the schedule is used up or
    absent, and None with neither."""
    schedule = sorted(float(step) for step in steps or ()
                      if isinstance(step, (int, float))
                      and not isinstance(step, bool))
    last = written[-1] if written else None
    if not schedule:
        return last
    if last is None:
        return schedule[0]
    higher = [step for step in schedule if step > last + LEVEL_MATCH_DB]
    return higher[0] if higher else last


def projection_text(header: Mapping[str, Any], level: Any, *,
                    voltage: Any = None, current: Any = None,
                    drive_fraction: Any = None) -> str:
    """The projection of a level row's readings to full level, in words,
    against the test's header (`run_log_model.project`): the factor, the
    verdict and every check, a check that cannot be made saying why."""
    if level is None:
        return 'not judged: no level is given'
    try:
        projection = model_module.project(
            level, voltage=voltage, current=current,
            drive_fraction=drive_fraction, sigma=header.get('sigma'),
            rated_current_peak=header.get('rated_current_peak'),
            rated_current_rms=header.get('rated_current_rms'),
            rail_voltage=header.get('rail_voltage'),
            drive_window=header.get('drive_window'))
    except RunLogError as error:
        return f'not judged: {error}'
    return str(projection)


# ---- reading what is typed -----------------------------------------------------


def _number(text: str, what: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise RunLogError(f'{what}: {text!r} is not a number') from None


def _whole(text: str, what: str) -> int | None:
    text = text.strip().lstrip('#')
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        raise RunLogError(f'{what}: {text!r} is not a whole number') from None


def _numbers(text: str, what: str) -> list[float] | None:
    items = [item for item in text.replace(';', ',').split(',')
             if item.strip()]
    if not items:
        return None
    return [_number(item, what) for item in items]  # type: ignore[misc]


def _texts(text: str) -> list[str] | None:
    items = [item.strip() for item in text.replace(';', ',').split(',')]
    return [item for item in items if item] or None


def channel_answers(text: str) -> dict[str, str] | None:
    """Answers by channel as typed in a level row: 'channel: answer'
    items separated by commas or semicolons, such as '12: not
    explained, 4: as predicted'. None for nothing typed; `RunLogError`
    naming the item for one without a colon or with an answer that is
    not one of `run_log.ANSWERS`."""
    answers: dict[str, str] = {}
    for item in text.replace(';', ',').split(','):
        if not item.strip():
            continue
        name, colon, answer = item.partition(':')
        answer = answer.strip().lower()
        if not colon or not name.strip():
            raise RunLogError(f'channel answers: {item.strip()!r} is not '
                              '"channel: answer"')
        if answer not in ANSWERS:
            raise RunLogError(f'channel {name.strip()}: {answer!r} is not an '
                              'answer; it is one of: ' + ', '.join(ANSWERS))
        answers[name.strip()] = answer
    return answers or None


def event_time(text: str, taken: datetime) -> datetime:
    """The event time a form gives: `taken` (the moment the form was
    opened) while its box still reads that time, otherwise the time
    typed (HH:MM or HH:MM:SS) on `taken`'s date and zone, so a keeper
    entering a late event writes when it happened."""
    text = text.strip()
    if not text or text == _clock(taken):
        return taken
    for pattern in ('%H:%M:%S', '%H:%M'):
        try:
            typed = datetime.strptime(text, pattern).time()
            break
        except ValueError:
            continue
    else:
        raise RunLogError(f'event time: {text!r} is not a time of day '
                          '(HH:MM or HH:MM:SS)')
    return datetime.combine(taken.date(), time(typed.hour, typed.minute,
                                               typed.second),
                            tzinfo=taken.tzinfo)


# ---- the worker ----------------------------------------------------------------


class _Relay(QObject):
    """The worker's way back to the window's thread. Made with the panel
    as its parent, so it is deleted on the window's thread."""

    done = Signal(int, bool, object)


class LogWorker:
    """One worker thread for everything that touches a log's file: opening
    and reading it, each entry written and synced, the mirror copied,
    files hashed, and the model rebuilt after each change. Jobs run one
    at a time in the order given, so entries are written in the order
    the keeper wrote them.

    Unlike the analysis verbs' runner, it never gates input: the window
    keeps taking keys and clicks while a job runs, and each result comes
    back through a queued signal to the callback given with it. A job
    holds no Qt object and no callback, so nothing of Qt's is let go on
    the worker; the callbacks stay on the window's thread, by number.
    """

    def __init__(self, parent: QObject) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix='run-log')
        self._relay = _Relay(parent)
        self._relay.done.connect(self._deliver)
        self._waiting: dict[int, Callable[[bool, Any], None]] = {}
        self._next = 0
        self._stopped = False

    @property
    def pending(self) -> int:
        """How many jobs have not yet come back."""
        return len(self._waiting)

    def submit(self, work: Callable[[], Any],
               then: Callable[[bool, Any], None]) -> None:
        """Run `work` on the worker; `then(ok, value)` is called on the
        window's thread with its result, or with the exception it
        raised."""
        if self._stopped:
            then(False, RunLogError('the run log is closed'))
            return
        self._next += 1
        number = self._next
        self._waiting[number] = then
        relay = self._relay

        def job() -> None:
            try:
                outcome = (True, work())
            except Exception as error:  # noqa: BLE001 — handed back whole
                outcome = (False, error)
            relay.done.emit(number, *outcome)

        self._pool.submit(job)

    def _deliver(self, number: int, ok: bool, value: Any) -> None:
        then = self._waiting.pop(number, None)
        if then is not None:
            then(ok, value)

    def shut_down(self, last: Callable[[], Any] | None = None) -> None:
        """Run `last` after every job already given, wait for it, and take
        no more. What comes back after this is not delivered."""
        if self._stopped:
            return
        self._stopped = True
        if last is not None:
            self._pool.submit(last)
        self._pool.shutdown(wait=True)
        self._waiting.clear()


# ---- the parts of the panel ----------------------------------------------------


def _scaled(widget: QWidget, scale: float, bold: bool = False) -> None:
    font = QFont(widget.font())
    size = font.pointSizeF()
    if size > 0:
        font.setPointSizeF(size * scale)
    font.setBold(bold)
    widget.setFont(font)


class StatusStrip(QFrame):
    """The strip across the top: the current test, its axis, run and
    level, running or stopped in words, the time since the run started,
    the holds counted (the button shows their list), and a notice line
    for a log read-only or a mirror that could not be written."""

    def __init__(self) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.test_label: QLabel = QLabel('No log open')
        self.place_label: QLabel = QLabel('')
        self.state_label: QLabel = QLabel('')
        self.since_label: QLabel = QLabel('')
        self.holds_button: QToolButton = QToolButton()
        self.holds_button.setCheckable(True)
        self.holds_button.setText('Holds: 0')
        self.holds_button.setToolTip('Show the holds: what the log asks to be '
                                     'answered before the test goes on')
        self.notice_label: QLabel = QLabel('')
        self.notice_label.setWordWrap(True)
        self.notice_label.hide()
        for label in (self.test_label, self.place_label, self.state_label,
                      self.since_label, self.holds_button):
            _scaled(label, STRIP_SCALE, bold=label is self.state_label)
        _scaled(self.notice_label, TYPE_SCALE)
        for label in (self.test_label, self.place_label, self.state_label,
                      self.since_label, self.notice_label):
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.test_label, 3)
        row.addWidget(self.place_label, 3)
        row.addWidget(self.state_label, 2)
        row.addWidget(self.since_label, 1)
        row.addWidget(self.holds_button)
        column = QVBoxLayout(self)
        column.setContentsMargins(8, 6, 8, 6)
        column.addLayout(row)
        column.addWidget(self.notice_label)
        self._status = Status()

    def show_snapshot(self, seen: Snapshot | None, now: datetime) -> None:
        """Read the strip off `seen` (None: no log open) at time `now`."""
        if seen is None:
            self._status = Status()
            self.test_label.setText('No log open')
            for label in (self.place_label, self.state_label,
                          self.since_label):
                label.setText('')
            self.holds_button.setText('Holds: 0')
            self.notice_label.hide()
            return
        status = self._status = seen.status
        if status.test is None:
            self.test_label.setText('No test open')
            self.place_label.setText('')
        else:
            self.test_label.setText(f'Test {status.test}: {status.title} '
                                    f'({status.kind})')
            parts = [f'axis {status.axis}' if status.axis else 'no axis']
            if status.run is not None:
                parts.append(f'run {status.run}')
            if status.level_db is not None:
                parts.append('full level' if status.level_db == 0
                             else f'{status.level_db:+g} dB')
            self.place_label.setText(', '.join(parts))
        word = {'running': 'RUNNING', 'aborted': 'STOPPED: aborted',
                'stopped': 'STOPPED: scheduled stop',
                'completed': 'STOPPED: completed'}.get(status.outcome,
                                                      'STOPPED')
        self.state_label.setText(word if status.test is not None else '')
        count = len(seen.holds)
        self.holds_button.setText(f'Holds: {count}')
        notices = []
        if not seen.writable:
            notices.append('Read-only: ' + (seen.read_only or 'opened to read'))
        if seen.mirror_problem:
            notices.append('Mirror: ' + seen.mirror_problem)
        if seen.problems:
            notices.append(f'{len(seen.problems)} line'
                           f'{"s" * (len(seen.problems) != 1)} not read or '
                           'not fitting: ' + '; '.join(seen.problems[:3]))
        self.notice_label.setText('\n'.join(notices))
        self.notice_label.setVisible(bool(notices))
        self.tick(now)

    def tick(self, now: datetime) -> None:
        """The time since the run started, to `now` while it runs; once it
        has stopped, how long it ran."""
        status = self._status
        if status.started is None:
            self.since_label.setText('')
        elif status.stopped is None:
            self.since_label.setText(
                f'{_span((now - status.started).total_seconds())} since start')
        else:
            self.since_label.setText(
                f'stopped after '
                f'{_span((status.stopped - status.started).total_seconds())}')


class EntryForm(QWidget):
    """A small form for one act: its event time, taken when the form
    opened and editable for an event entered late, its fields, a line of
    free text, and Write and Cancel. Enter in any box writes; Esc
    cancels. What a box cannot hold is said on the form's message line,
    and nothing is written."""

    written = Signal()
    cancelled = Signal()

    #: what a field of a form is typed as
    BOX_KINDS = ('text', 'number', 'whole', 'numbers', 'texts', 'flag',
                 'choice', 'path', 'entry')

    def __init__(self, name: str, title: str,
                 fields: Sequence[tuple[str, str, str, Any]], taken: datetime,
                 *, required: Sequence[str] = (), text: str = 'Note',
                 note: str = '', defaults: Mapping[str, Any] | None = None,
                 with_time: bool = True) -> None:
        super().__init__()
        self.name: str = name
        self.taken: datetime = taken
        self.required: tuple[str, ...] = tuple(required)
        self.on_write: Callable[[dict[str, Any], str, datetime], None] | None \
            = None
        #: the entry a correction form corrects
        self.target: Entry | None = None
        self.boxes: dict[str, QWidget] = {}
        self._kinds: dict[str, str] = {}
        self._labels: dict[str, str] = {}
        column = QVBoxLayout(self)
        column.setContentsMargins(4, 4, 4, 4)
        self.title_label: QLabel = QLabel(title)
        _scaled(self.title_label, TYPE_SCALE, bold=True)
        column.addWidget(self.title_label)
        if note:
            said = QLabel(note)
            said.setWordWrap(True)
            column.addWidget(said)
        grid = QGridLayout()
        column.addLayout(grid)
        self.time_box: QLineEdit = QLineEdit(_clock(taken))
        self.time_box.setToolTip('When it happened: taken when this form '
                                 'opened; type another time for an event '
                                 'entered late')
        row = 0
        if with_time:
            grid.addWidget(QLabel('Event time'), row, 0)
            grid.addWidget(self.time_box, row, 1)
            row += 1
        defaults = dict(defaults or {})
        for field, label, kind, choices in fields:
            if kind not in self.BOX_KINDS:
                raise ValueError(f'{kind!r} is not a kind of box')
            box = self._box(kind, choices, defaults.get(field))
            marked = label + (' *' if field in self.required else '')
            grid.addWidget(QLabel(marked), row, 0)
            grid.addWidget(box, row, 1)
            self.boxes[field] = box
            self._kinds[field] = kind
            self._labels[field] = label
            row += 1
        self.text_box: QLineEdit | None = None
        if text:
            self.text_box = QLineEdit()
            grid.addWidget(QLabel(text), row, 0)
            grid.addWidget(self.text_box, row, 1)
        self.message: QLabel = QLabel('')
        self.message.setWordWrap(True)
        column.addWidget(self.message)
        buttons = QHBoxLayout()
        self.write_button: QPushButton = QPushButton('Write (Enter)')
        self.cancel_button: QPushButton = QPushButton('Cancel (Esc)')
        self.write_button.clicked.connect(self.write)
        self.cancel_button.clicked.connect(self.cancelled.emit)
        buttons.addWidget(self.write_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        column.addLayout(buttons)
        column.addStretch(1)
        for widget in self._inputs():
            widget.installEventFilter(self)
        _scaled(self, TYPE_SCALE)

    def _box(self, kind: str, choices: Any, default: Any) -> QWidget:
        if kind == 'choice':
            box = QComboBox()
            box.addItems([str(choice) for choice in choices])
            if default is not None:
                box.setCurrentText(str(default))
            return box
        if kind == 'flag':
            box = QComboBox()
            box.addItems(['', 'yes', 'no'])
            if default is not None:
                box.setCurrentText('yes' if default else 'no')
            return box
        if kind == 'path':
            holder = QWidget()
            line = QLineEdit('' if default is None else str(default))
            browse = QPushButton('Browse…')
            row = QHBoxLayout(holder)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(line, 1)
            row.addWidget(browse)
            browse.clicked.connect(lambda: self._browse(line))
            holder.line = line  # type: ignore[attr-defined]
            return holder
        line = QLineEdit()
        if default is not None:
            line.setText(', '.join(str(item) for item in default)
                         if isinstance(default, (list, tuple))
                         else str(default))
        return line

    def _browse(self, line: QLineEdit) -> None:
        chosen, _ = QFileDialog.getSaveFileName(
            self, 'Run log', line.text(), f'Run log (*{SUFFIX})')
        if chosen:
            line.setText(chosen)

    def _inputs(self) -> list[QWidget]:
        found: list[QWidget] = [self.time_box]
        for box in self.boxes.values():
            found.append(getattr(box, 'line', box))
        if self.text_box is not None:
            found.append(self.text_box)
        return found

    def box_text(self, field: str) -> str:
        """What box `field` holds, as text."""
        box = self.boxes[field]
        box = getattr(box, 'line', box)
        if isinstance(box, QComboBox):
            return box.currentText()
        assert isinstance(box, QLineEdit)
        return box.text()

    def set_box(self, field: str, value: str) -> None:
        """Put `value` in box `field`: typed into a text box, chosen in a
        list."""
        box = self.boxes[field]
        box = getattr(box, 'line', box)
        if isinstance(box, QComboBox):
            box.setCurrentText(value)
        else:
            assert isinstance(box, QLineEdit)
            box.setText(value)

    def values(self) -> dict[str, Any]:
        """Every box read as its field holds it, blank boxes left out.
        `RunLogError` naming the box for one that cannot be read, or for
        a required box left blank."""
        values: dict[str, Any] = {}
        for field, kind in self._kinds.items():
            text = self.box_text(field)
            label = self._labels[field]
            if kind in ('text', 'path', 'choice'):
                value: Any = text.strip() or None
            elif kind == 'number':
                value = _number(text, label)
            elif kind in ('whole', 'entry'):
                value = _whole(text, label)
            elif kind == 'numbers':
                value = _numbers(text, label)
            elif kind == 'texts':
                value = _texts(text)
            else:
                value = {'yes': True, 'no': False}.get(text)
            if value is None and field in self.required:
                raise RunLogError(f'{label} is needed')
            if value is not None:
                values[field] = value
        return values

    def write(self) -> None:
        """Read the form and hand it to `on_write`; say why on the form if
        it cannot be read."""
        try:
            values = self.values()
            when = event_time(self.time_box.text(), self.taken)
        except RunLogError as error:
            self.say(str(error))
            return
        text = self.text_box.text().strip() if self.text_box else ''
        if self.on_write is not None:
            self.on_write(values, text, when)

    def say(self, message: str) -> None:
        """Show `message` on the form's message line."""
        self.message.setText(message)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()                         # type: ignore[attr-defined]
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.write()
                return True
            if key == Qt.Key.Key_Escape:
                self.cancelled.emit()
                return True
        return super().eventFilter(watched, event)

    def focus_first(self) -> None:
        """Put the cursor in the first box after the time."""
        inputs = self._inputs()
        (inputs[1] if len(inputs) > 1 else inputs[0]).setFocus()


class LevelRow(QWidget):
    """The level row: one entry per step of the step-up. The level is
    offered from the header's step schedule; the voltage, current and
    drive fraction follow in tab order, then the analyst's answer, the
    answers by channel, the go and who agreed it at a hold point. Enter
    in any box writes the row. The projection to full level is redrawn
    on every keystroke, and the event time is taken at the row's first
    keystroke."""

    def __init__(self, clock: Callable[[], datetime]) -> None:
        super().__init__()
        self._clock = clock
        self.header: Mapping[str, Any] = {}
        self.started: datetime | None = None
        self._offered: float | None = None
        self.on_write: Callable[[dict[str, Any], datetime], None] | None = None
        self.level_box: QLineEdit = QLineEdit()
        self.voltage_box: QLineEdit = QLineEdit()
        self.current_box: QLineEdit = QLineEdit()
        self.drive_box: QLineEdit = QLineEdit()
        self.answer_box: QComboBox = QComboBox()
        self.answer_box.addItems(['', *ANSWERS])
        self.channels_box: QLineEdit = QLineEdit()
        self.channels_box.setPlaceholderText('12: not explained, 4: explained')
        self.go_box: QCheckBox = QCheckBox('Go')
        self.agreed_box: QLineEdit = QLineEdit()
        self.agreed_box.setPlaceholderText('at a hold point')
        grid = QGridLayout()
        boxes = (('Level (dB)', self.level_box), ('Voltage (V)',
                                                   self.voltage_box),
                 ('Current (A)', self.current_box),
                 ('Drive fraction', self.drive_box),
                 ('Answer', self.answer_box),
                 ('By channel', self.channels_box), ('', self.go_box),
                 ('Agreed by', self.agreed_box))
        for column, (label, box) in enumerate(boxes):
            grid.addWidget(QLabel(label), 0, column)
            grid.addWidget(box, 1, column)
        grid.setColumnStretch(5, 2)
        self.time_label: QLabel = QLabel('')
        self.projection_label: QLabel = QLabel('')
        self.projection_label.setWordWrap(True)
        self.projection_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.message: QLabel = QLabel('')
        self.message.setWordWrap(True)
        title = QLabel('Level row (Enter writes it)')
        _scaled(title, TYPE_SCALE, bold=True)
        column_layout = QVBoxLayout(self)
        column_layout.setContentsMargins(4, 4, 4, 4)
        heading = QHBoxLayout()
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(self.time_label)
        column_layout.addLayout(heading)
        column_layout.addLayout(grid)
        column_layout.addWidget(self.projection_label)
        column_layout.addWidget(self.message)
        column_layout.addStretch(1)
        for line in (self.level_box, self.voltage_box, self.current_box,
                     self.drive_box, self.channels_box, self.agreed_box):
            line.textEdited.connect(self._typed)
        self.answer_box.activated.connect(self._typed)
        self.go_box.clicked.connect(self._typed)
        for widget in self.inputs():
            widget.installEventFilter(self)
        for line in (self.level_box, self.voltage_box, self.current_box,
                     self.drive_box):
            line.textChanged.connect(self.redraw)
        _scaled(self, TYPE_SCALE)
        self.redraw()

    def inputs(self) -> list[QWidget]:
        """The boxes in tab order."""
        return [self.level_box, self.voltage_box, self.current_box,
                self.drive_box, self.answer_box, self.channels_box,
                self.go_box, self.agreed_box]

    def _typed(self, *_args: Any) -> None:
        if self.started is None:
            self.started = self._clock()
            self.time_label.setText(f'event time {_clock(self.started)}')

    def offer(self, header: Mapping[str, Any], level: float | None) -> None:
        """The test's header the projection is judged against, and the
        level offered for the next row (left alone once typed into)."""
        self.header = header
        self._offered = level
        if self.started is None:
            self._put_level()
        self.redraw()

    def _put_level(self) -> None:
        level = self._offered
        self.level_box.setText('' if level is None else f'{level:g}')

    def fields(self) -> dict[str, Any]:
        """The row as a level row entry's fields; `RunLogError` naming
        the box that cannot be read."""
        fields: dict[str, Any] = {
            'level_db': _number(self.level_box.text(), 'level'),
            'voltage': _number(self.voltage_box.text(), 'voltage'),
            'current': _number(self.current_box.text(), 'current'),
            'drive_fraction': _number(self.drive_box.text(),
                                      'drive fraction'),
            'answer': self.answer_box.currentText() or None,
            'channel_answers': channel_answers(self.channels_box.text()),
            'go': True if self.go_box.isChecked() else None,
            'agreed_by': self.agreed_box.text().strip() or None,
        }
        return {name: value for name, value in fields.items()
                if value is not None}

    def redraw(self) -> None:
        """The projection to full level of what the row holds now."""
        try:
            fields = self.fields()
        except RunLogError as error:
            self.projection_label.setText(f'not judged: {error}')
            return
        self.projection_label.setText(projection_text(
            self.header, fields.get('level_db'),
            voltage=fields.get('voltage'), current=fields.get('current'),
            drive_fraction=fields.get('drive_fraction')))

    def write(self) -> None:
        """Hand the row to `on_write` with its event time."""
        try:
            fields = self.fields()
        except RunLogError as error:
            self.message.setText(str(error))
            return
        if not fields:
            self.message.setText('the row is empty')
            return
        when = self.started or self._clock()
        if self.on_write is not None:
            self.on_write(fields, when)

    def clear(self) -> None:
        """Empty the row after it is written, ready for the next step."""
        self.started = None
        self.time_label.setText('')
        self._put_level()
        for line in (self.voltage_box, self.current_box, self.drive_box,
                     self.channels_box, self.agreed_box):
            line.clear()
        self.answer_box.setCurrentIndex(0)
        self.go_box.setChecked(False)
        self.message.setText('')
        self.voltage_box.setFocus()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.KeyPress and event.key() in (  # type: ignore[attr-defined]
                Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.write()
            return True
        return super().eventFilter(watched, event)


# ---- the panel -----------------------------------------------------------------


class RunLogPanel(QWidget):
    """Everything inside the Run Log dock: the status strip, the holds,
    the tests, the day, the owed lists and the entry.

    `clock` gives the event time an act takes; `log_clock` and `who` are
    handed to the log for its entered-at times and the login and machine
    every entry records (each the computer's own unless set).
    """

    #: a log was opened, created or closed; carries the path or ''
    log_changed = Signal(str)
    #: the layout asked for: True for the log alone, False for the whole
    #: window
    layout_wanted = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.clock: Callable[[], datetime] = _now
        self.log_clock: Callable[[], datetime] | None = None
        self.who: tuple[str, str] | None = None
        self.snapshot: Snapshot | None = None
        self.worker: LogWorker = LogWorker(self)
        self._log: RunLog | None = None
        self._shown_test: Any = ALL
        self._last_current: int | None = None
        self.form: EntryForm | None = None

        self.status: StatusStrip = StatusStrip()
        self.status.holds_button.toggled.connect(self._show_holds)
        self.holds_list: QListWidget = QListWidget()
        self.holds_list.setWordWrap(True)
        self.holds_list.hide()
        _scaled(self.holds_list, TYPE_SCALE)

        self.tests_bar: QTabBar = QTabBar()
        self.tests_bar.setExpanding(False)
        self.tests_bar.currentChanged.connect(self._test_picked)
        _scaled(self.tests_bar, TYPE_SCALE)

        self.day_list: QListWidget = QListWidget()
        self.day_list.setWordWrap(True)
        self.day_list.setAlternatingRowColors(True)
        _scaled(self.day_list, TYPE_SCALE)
        self.owed_tree: QTreeWidget = QTreeWidget()
        self.owed_tree.setHeaderLabels(['Owed (shown, not enforced)'])
        _scaled(self.owed_tree, TYPE_SCALE)

        self.initials_box: QLineEdit = QLineEdit()
        self.initials_box.setPlaceholderText('initials')
        self.initials_box.setMaximumWidth(120)
        self.axis_box: QLineEdit = QLineEdit()
        self.axis_box.setPlaceholderText('axis')
        self.axis_box.setMaximumWidth(120)
        self.run_label: QLabel = QLabel('')
        self.message: QLabel = QLabel('')
        self.message.setWordWrap(True)
        self.level_row: LevelRow = LevelRow(lambda: self.clock())
        self.level_row.on_write = self.write_level
        self.entry_stack: QStackedWidget = QStackedWidget()
        self.entry_stack.addWidget(self.level_row)

        self.acts: QToolBar = QToolBar('Run log')
        self.actions_by_name: dict[str, QAction] = {}
        self._build_acts()

        context = QHBoxLayout()
        said = [QLabel('Initials'), QLabel('Axis')]
        context.addWidget(said[0])
        context.addWidget(self.initials_box)
        context.addWidget(said[1])
        context.addWidget(self.axis_box)
        context.addWidget(self.run_label, 1)
        for widget in (*said, self.initials_box, self.axis_box,
                       self.run_label, self.message):
            _scaled(widget, TYPE_SCALE)

        entry = QWidget()
        entry_column = QVBoxLayout(entry)
        entry_column.setContentsMargins(0, 0, 0, 0)
        entry_column.addLayout(context)
        entry_column.addWidget(self.entry_stack, 1)
        entry_column.addWidget(self.message)

        day = QSplitter(Qt.Orientation.Horizontal)
        day.addWidget(self.day_list)
        day.addWidget(self.owed_tree)
        day.setStretchFactor(0, 3)
        day.setStretchFactor(1, 1)
        self.split: QSplitter = QSplitter(Qt.Orientation.Vertical)
        self.split.addWidget(day)
        self.split.addWidget(entry)
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 2)

        column = QVBoxLayout(self)
        column.setContentsMargins(4, 4, 4, 4)
        column.addWidget(self.acts)
        column.addWidget(self.status)
        column.addWidget(self.holds_list)
        column.addWidget(self.tests_bar)
        column.addWidget(self.split, 1)

        # the time since start moves once a second; nothing else is
        # read on the tick, so it costs a label
        self._ticker = QTimer(self)
        self._ticker.setInterval(1000)
        self._ticker.timeout.connect(lambda: self.status.tick(self.clock()))
        self._ticker.start()
        self._show(None)

    # ---- the acts -------------------------------------------------------------

    def _act(self, name: str, label: str, slot: Callable[[], Any],
             tip: str) -> QAction:
        action = QAction(label, self)
        key = KEYS.get(name)
        if key:
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(
                Qt.ShortcutContext.WidgetWithChildrenShortcut)
            tip += f' ({key})'
        action.setToolTip(tip)
        action.triggered.connect(lambda _checked=False: slot())
        self.addAction(action)
        self.actions_by_name[name] = action
        return action

    def _build_acts(self) -> None:
        bar = self.acts
        bar.setMovable(False)
        bar.addAction(self._act('abort', 'ABORT', self.abort,
                                'Write a stop: the time is taken at the key'))
        _scaled(bar.widgetForAction(self.actions_by_name['abort']), 1.2,
                bold=True)
        for name, label, tip in (
                ('run start', 'Run Start', 'Start the next run'),
                ('run stop', 'Run Stop', 'Stop the run, not an abort'),
                ('note', 'Note', 'Write a note'),
                ('clock check', 'Clock Check',
                 'The recorders\' clock less this computer\'s'),
                ('person in', 'Person In', 'Someone came into the room'),
                ('person out', 'Person Out', 'Someone left the room'),
                ('cause found', 'Cause Found', 'The cause of an abort'),
                ('correct', 'Correct', ('Correct the entry picked in the '
                                        'day; the entry stays as written'))):
            bar.addAction(self._act(name, label,
                                    lambda name=name: self.open_form(name),
                                    tip))
        bar.addSeparator()
        for name, label, tip in (
                ('open test', 'Open Test', 'Open the next test'),
                ('roll', 'Roll', ('Close this test to runs and open the '
                                  'next, its header carried')),
                ('close test', 'Close Test', 'Close this test to runs'),
                ('release', 'Release', 'Release a test closed to runs')):
            bar.addAction(self._act(name, label,
                                    lambda name=name: self.open_form(name),
                                    tip))
        bar.addSeparator()
        bar.addAction(self._act('new log', 'New Log',
                                lambda: self.open_form('new log'),
                                'Start a new run log file'))
        bar.addAction(self._act('open log', 'Open Log', self.ask_open_log,
                                'Open a run log; read-only when another '
                                'program is writing it'))
        bar.addAction(self._act('close log', 'Close Log', self.close_log,
                                'Close the log and release it for another '
                                'writer'))
        bar.addAction(self._act('reload', 'Reload', self.reload,
                                'Read the log again from its file'))
        bar.addSeparator()
        self.layout_action: QAction = self._act(
            'layout', 'Whole Window', self._toggle_layout,
            'Show the analysis panes beside the log, or the log alone')
        bar.addAction(self.layout_action)
        self._act('level row', 'Level Row', self.focus_level_row,
                  'Back to the level row')

    def _toggle_layout(self) -> None:
        self.layout_wanted.emit(self.layout_action.text() == 'Log Only')

    def show_layout(self, log_only: bool) -> None:
        """Name the layout the toggle would switch to."""
        self.layout_action.setText('Whole Window' if log_only else 'Log Only')

    # ---- opening and closing --------------------------------------------------

    def new_log(self, path: str, title: str = '',
                mirror: str | None = None) -> None:
        """Create a log at `path` and open it for writing, on the worker."""
        log_clock, who = self.log_clock, self.who
        self._swap(lambda: (RunLog.create(path, title=title, mirror=mirror,
                                          clock=log_clock, who=who), ''))

    def open_log(self, path: str, mirror: str | None = None) -> None:
        """Open the log at `path` for writing, on the worker. One that
        cannot be written here (another writer holds it, or it is of an
        older format version) opens read-only, saying why."""
        log_clock, who = self.log_clock, self.who

        def opened() -> tuple[RunLog, str]:
            try:
                return RunLog.open(path, mirror=mirror, clock=log_clock,
                                   who=who), ''
            except RunLogError as refused:
                # read where it cannot be written; a file that cannot be
                # read either raises what reading it says
                return RunLog.read(path), str(refused)

        self._swap(opened)

    def ask_open_log(self) -> None:
        """Ask for a log to open."""
        path, _ = QFileDialog.getOpenFileName(self, 'Open Run Log', '',
                                              f'Run log (*{SUFFIX})')
        if path:
            self.open_log(path)

    def _swap(self, opening: Callable[[], tuple[RunLog, str]]) -> None:
        old = self._log
        self._log = None

        def work() -> tuple[RunLog, Snapshot]:
            if old is not None:
                old.close()
            log, why = opening()
            return log, snapshot(log, read_only=why)

        def then(ok: bool, value: Any) -> None:
            if not ok:
                self._show(None)
                self.say(str(value))
                self.log_changed.emit('')
                return
            self._log, seen = value
            self.say('')
            self._show(seen)
            self.log_changed.emit(seen.path)

        self.worker.submit(work, then)

    def close_log(self) -> None:
        """Close the log and release its lock, on the worker."""
        log, self._log = self._log, None
        if log is None:
            return
        self.worker.submit(log.close, lambda ok, value: (
            self._show(None), self.log_changed.emit('')))

    def reload(self) -> None:
        """Read the log again: a log read-only shows what its writer has
        added since."""
        if self.snapshot is None:
            return
        if self._log is not None and self._log.writable:
            log = self._log
            self.worker.submit(lambda: snapshot(log), self._showing)
            return
        path, why = self.snapshot.path, self.snapshot.read_only

        def work() -> tuple[RunLog, Snapshot]:
            log = RunLog.read(path)
            return log, snapshot(log, read_only=why)

        def then(ok: bool, value: Any) -> None:
            if not ok:
                self.say(str(value))
                return
            self._log, seen = value
            self._show(seen)

        self.worker.submit(work, then)

    def shut_down(self) -> None:
        """Close the log, releasing its lock, after every entry already
        given is written; then stop the worker. The window calls this as
        it closes."""
        self._ticker.stop()
        log, self._log = self._log, None
        self.worker.shut_down(log.close if log is not None else None)

    def settle(self, timeout: float = 10.0) -> bool:
        """Turn the event loop until every job given to the worker has
        come back, or `timeout` seconds pass; whether it settled. For a
        script or a test driving the dock."""
        import time as clock

        from PySide6.QtWidgets import QApplication

        deadline = clock.monotonic() + timeout
        while self.worker.pending and clock.monotonic() < deadline:
            QApplication.processEvents()
            clock.sleep(0.005)
        QApplication.processEvents()
        return not self.worker.pending

    # ---- writing --------------------------------------------------------------

    def _author(self) -> str | None:
        author = self.initials_box.text().strip()
        if not author:
            self.say('Type your initials first: every entry records its '
                     'author.')
            self.initials_box.setFocus()
            return None
        return author

    def _axis(self) -> str | None:
        return self.axis_box.text().strip() or None

    def _writing(self, write: Callable[[RunLog], Any],
                 done: Callable[[Any], None] | None = None,
                 refused: Callable[[str], None] | None = None) -> bool:
        """Run `write(log)` on the worker, then show the log as it stands;
        `done` gets what `write` returned, `refused` the message when it
        raised. False, saying why, with no log open for writing."""
        log = self._log
        if log is None or self.snapshot is None:
            self.say('No log is open: start one with New Log, or open one.')
            return False
        if not self.snapshot.writable:
            self.say('This log is open read-only. '
                     + self.snapshot.read_only)
            return False

        def work() -> tuple[Any, Snapshot]:
            try:
                return write(log), snapshot(log)
            except RunLogError:
                raise
            except OSError as error:
                raise RunLogError(f'the log could not be written: {error}') \
                    from None

        def then(ok: bool, value: Any) -> None:
            if not ok:
                message = str(value)
                self.say(message)
                if refused is not None:
                    refused(message)
                return
            result, seen = value
            self.say('')
            self._show(seen)
            if done is not None:
                done(result)

        self.worker.submit(work, then)
        return True

    def _place(self) -> tuple[int | None, str | None, int | None]:
        seen = self.snapshot
        if seen is None or seen.status.test is None:
            return None, self._axis(), None
        return seen.status.test, self._axis(), seen.run

    def write_level(self, fields: dict[str, Any], when: datetime) -> None:
        """Write a level row to the current run of the current test."""
        author = self._author()
        if author is None:
            return
        test, axis, run = self._place()
        written = self._writing(
            lambda log: model_module.append(
                log, 'level_row', author=author, test=test, axis=axis,
                run=run, fields=fields, event_time=when),
            done=lambda _entry: self.level_row.clear(),
            refused=self.level_row.message.setText)
        if written:
            self.level_row.message.setText('')

    def abort(self) -> EntryForm | None:
        """The abort: the event time is taken now, at the key, and the
        form needs only the kind of stop and the author; the rest of the
        first minute is offered and, left blank, owed."""
        taken = self.clock()
        fields = [('stop', 'Kind of stop', 'choice', ('', *STOPS))]
        fields += [(name, name.replace('_', ' ').capitalize(),
                    'number' if FIELD_TYPES[name] == 'number' else 'text',
                    None)
                   for name in KINDS['abort'].optional]
        form = EntryForm('abort', f'ABORT at {_clock(taken)}', fields, taken,
                         required=('stop',), text='What was seen',
                         note='Only the kind of stop is needed now; what is '
                              'left blank is listed as owed.')
        form.on_write = self._write_abort
        self._open(form)
        form.boxes['stop'].setFocus()
        return form

    def _write_abort(self, values: dict[str, Any], text: str,
                     when: datetime) -> None:
        author = self._author()
        if author is None:
            return
        test, axis, run = self._place()
        stop_fields = {'reason': f'abort: {values.get("stop", "")}'}
        if 'counter' in values:
            stop_fields['counter'] = values['counter']

        def write(log: RunLog) -> Entry:
            # both checked before either is written, so a refusal never
            # leaves the stop written without its abort
            check_fields('run_stop', stop_fields)
            check_fields('abort', values)
            model = RunLogModel(log)
            model_module.admit(model, 'abort', test)
            log.append('run_stop', author=author, test=test, axis=axis,
                       run=run, fields=stop_fields, event_time=when)
            return log.append('abort', author=author, test=test, axis=axis,
                              run=run, fields=values, text=text,
                              event_time=when)

        self._writing(write, done=self._aborted, refused=self._form_refused)

    def _aborted(self, entry: Entry) -> None:
        self._close_form()
        owed = next((found for found in (self.snapshot.owed if self.snapshot
                                         else ())
                     if found.name == 'abort'
                     and found.title.endswith(f'#{entry.id}')), None)
        if owed is not None and owed.owed:
            self.say(f'Abort #{entry.id} written. Owed: '
                     + ', '.join(line.label for line in owed.owed))

    def _form_refused(self, message: str) -> None:
        if self.form is not None:
            self.form.say(message)

    # ---- the small forms ------------------------------------------------------

    def open_form(self, name: str) -> EntryForm | None:
        """Open the small form `name` ('note', 'clock check', 'person in',
        'person out', 'run start', 'run stop', 'cause found', 'correct',
        'open test', 'roll', 'close test', 'release' or 'new log'), its
        event time taken now."""
        taken = self.clock()
        seen = self.snapshot
        builder = {
            'note': lambda: EntryForm('note', 'Note', [], taken,
                                      text='Note'),
            'clock check': lambda: EntryForm(
                'clock check', 'Clock check',
                [('offset_s', 'Recorders less this computer (s)', 'number',
                  None), ('recorder', 'Recorder', 'text', None),
                 ('method', 'Method', 'text', None)], taken,
                required=('offset_s',)),
            'person in': lambda: EntryForm(
                'person in', 'Person in',
                [('person', 'Person', 'text', None),
                 ('role', 'Role', 'text', None)], taken,
                required=('person',)),
            'person out': lambda: EntryForm(
                'person out', 'Person out',
                [('person', 'Person', 'text', None),
                 ('role', 'Role', 'text', None)], taken,
                required=('person',)),
            'run start': lambda: EntryForm(
                'run start', 'Run start',
                [('level_db', 'Level (dB)', 'number', None),
                 ('counter', 'Counter', 'text', None)], taken),
            'run stop': lambda: EntryForm(
                'run stop', 'Run stop (not an abort)',
                [('counter', 'Counter', 'text', None),
                 ('scheduled', 'Scheduled stop', 'flag', None),
                 ('result', 'Result', 'text', None),
                 ('by', 'Run by', 'text', None),
                 ('reason', 'Reason', 'text', None)], taken),
            'cause found': lambda: EntryForm(
                'cause found', 'Cause found',
                [('cause', 'Cause', 'text', None),
                 ('refers_to', 'Of entry #', 'entry', None)], taken,
                required=('cause',),
                defaults={'refers_to': seen.open_aborts[-1]
                          if seen and seen.open_aborts else None}),
            'open test': lambda: self._test_form('open test', taken),
            'roll': lambda: self._test_form('roll', taken),
            'close test': lambda: EntryForm(
                'close test', 'Close the test to runs',
                [('reason', 'Reason', 'text', None)], taken, text=''),
            'release': lambda: EntryForm(
                'release', 'Release a test',
                [('test', 'Test', 'whole', None)], taken,
                required=('test',), text='Released by',
                defaults={'test': seen.unreleased[-1]
                          if seen and seen.unreleased else None}),
            'new log': lambda: EntryForm(
                'new log', 'New run log',
                [('path', 'File', 'path', None),
                 ('title', 'Article or program', 'text', None),
                 ('mirror', 'Second copy (optional)', 'path', None)], taken,
                required=('path',), text='', with_time=False),
            'correct': lambda: self._correct_form(taken),
        }.get(name)
        if builder is None:
            raise ValueError(f'{name!r} is not a form of the run log')
        form = builder()
        if form is None:
            return None
        form.on_write = (lambda values, text, when, name=name:
                         self._write_form(name, values, text, when))
        self._open(form)
        form.focus_first()
        return form

    def _test_form(self, name: str, taken: datetime) -> EntryForm:
        # The header fields the level row reads; the rest of a header is
        # written by script or a header change until the header has a
        # form of its own.
        rolling = name == 'roll'
        kinds = ('', *TEST_KINDS) if rolling else TEST_KINDS
        fields = [('test_kind', 'Kind' + (' (blank: carried)' if rolling
                                          else ''), 'choice', kinds),
                  ('title', 'Title', 'text', None),
                  ('article', 'Article', 'text', None),
                  ('serial', 'Serial', 'text', None),
                  ('configuration', 'Configuration', 'text', None),
                  ('steps_db', 'Step schedule (dB)', 'numbers', None),
                  ('hold_points_db', 'Hold points (dB)', 'numbers', None),
                  ('sigma', 'Sigma', 'number', None),
                  ('rated_current_peak', 'Peak current rating (A)', 'number',
                   None),
                  ('rated_current_rms', 'Continuous current rating (A)',
                   'number', None),
                  ('rail_voltage', 'Rail (V)', 'number', None)]
        title = ('Roll to the next test (blank fields are carried)'
                 if rolling else 'Open the next test')
        if rolling:
            fields.append(('reason', 'Why this test closes', 'text', None))
        return EntryForm(name, title, fields, taken,
                         required=() if rolling else ('test_kind',))

    def _correct_form(self, taken: datetime) -> EntryForm | None:
        item = self.day_list.currentItem()
        entry = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not isinstance(entry, Entry):
            self.say('Pick the entry to correct in the day list first.')
            return None
        if entry.kind == 'correction':
            self.say('A correction is not corrected; correct the original.')
            return None
        kind = KINDS[entry.kind]
        fields = []
        for name in kind.fields:
            held = FIELD_TYPES.get(name, 'text')
            box = {'number': 'number', 'whole': 'whole', 'entry': 'entry',
                   'flag': 'flag', 'numbers': 'numbers',
                   'texts': 'texts'}.get(held, 'text')
            if held in ('notches', 'answers', 'scalar') and name in \
                    entry.fields and not isinstance(entry.fields[name], str):
                continue          # not typed in a box; corrected by script
            choices = None
            if name == 'stop':
                box, choices = 'choice', ('', *STOPS)
            fields.append((name, name.replace('_', ' ').capitalize(), box,
                           choices))
        form = EntryForm('correct', f'Correct #{entry.id} ({entry.label})',
                         fields, taken, text='Why', defaults=dict(
                             entry.fields), with_time=False)
        form.target = entry
        return form

    def _write_form(self, name: str, values: dict[str, Any], text: str,
                    when: datetime) -> None:
        if name == 'new log':
            self._close_form()
            self.new_log(values['path'], values.get('title', ''),
                         values.get('mirror'))
            return
        author = self._author()
        if author is None:
            return
        test, axis, run = self._place()
        kind = {'note': 'note', 'clock check': 'clock_check',
                'person in': 'person_in', 'person out': 'person_out',
                'run start': 'run_start', 'run stop': 'run_stop',
                'cause found': 'cause_found'}.get(name)
        if kind is not None:
            # every entry written while a test is open belongs to it; a
            # run's entries also to the axis and the run
            in_run = kind in model_module.RUN_KINDS
            self._writing(
                lambda log: model_module.append(
                    log, kind, author=author, test=test,
                    axis=axis if in_run else None,
                    run=run if in_run else None, fields=values, text=text,
                    event_time=when),
                done=lambda _entry: self._close_form(),
                refused=self._form_refused)
            return
        if name == 'correct':
            form = self.form
            assert form is not None
            target = form.target
            assert target is not None
            changed = {field: value for field, value in values.items()
                       if target.fields.get(field) != value}
            for field in form.boxes:
                # a box emptied strikes an optional field it held
                if (field not in values and field in target.fields
                        and field in KINDS[target.kind].optional):
                    changed[field] = None
            if not changed:
                form.say('Nothing was changed.')
                return
            self._writing(lambda log: log.correct(target.id, author=author,
                                                  fields=changed, text=text),
                          done=lambda _entry: self._close_form(),
                          refused=self._form_refused)
            return
        if name in ('open test', 'roll'):
            header = dict(values)
            test_kind = header.pop('test_kind', None)
            reason = header.pop('reason', '')
            if name == 'open test':
                write: Callable[[RunLog], Any] = (
                    lambda log: model_module.open_test(
                        log, author=author, test_kind=test_kind,
                        header=header, text=text, event_time=when))
            else:
                write = (lambda log: model_module.roll(
                    log, author=author, test_kind=test_kind, header=header,
                    reason=reason, text=text, event_time=when))
            self._writing(write, done=lambda _entry: self._close_form(),
                          refused=self._form_refused)
            return
        if name == 'close test':
            if test is None:
                self.form.say('No test is open to runs.')  # type: ignore[union-attr]
                return
            self._writing(lambda log: model_module.close_test(
                log, test, author=author, reason=values.get('reason', ''),
                event_time=when), done=lambda _entry: self._close_form(),
                refused=self._form_refused)
            return
        if name == 'release':
            self._writing(lambda log: model_module.release(
                log, values['test'], author=author, text=text,
                event_time=when), done=lambda _entry: self._close_form(),
                refused=self._form_refused)

    def _open(self, form: EntryForm) -> None:
        self._close_form()
        self.form = form
        form.cancelled.connect(self._close_form)
        self.entry_stack.addWidget(form)
        self.entry_stack.setCurrentWidget(form)

    def _close_form(self) -> None:
        form, self.form = self.form, None
        if form is not None:
            self.entry_stack.removeWidget(form)
            form.deleteLater()
        self.entry_stack.setCurrentWidget(self.level_row)

    def focus_level_row(self) -> None:
        """Close any open form and put the cursor in the level row."""
        self._close_form()
        self.level_row.voltage_box.setFocus()

    def say(self, message: str) -> None:
        """Show `message` under the entry."""
        self.message.setText(message)

    # ---- showing --------------------------------------------------------------

    def _showing(self, ok: bool, value: Any) -> None:
        if ok:
            self._show(value)
        else:
            self.say(str(value))

    def _show(self, seen: Snapshot | None) -> None:
        self.snapshot = seen
        self.status.show_snapshot(seen, self.clock())
        writable = seen is not None and seen.writable
        for name, action in self.actions_by_name.items():
            if name not in ('new log', 'open log', 'layout', 'level row'):
                action.setEnabled(seen is not None and (
                    writable or name in ('close log', 'reload')))
        self.level_row.setEnabled(writable)
        self.initials_box.setEnabled(writable)
        self.axis_box.setEnabled(writable)
        if seen is None:
            self.run_label.setText('')
            self.tests_bar.blockSignals(True)
            while self.tests_bar.count():
                self.tests_bar.removeTab(0)
            self.tests_bar.blockSignals(False)
            self.day_list.clear()
            self.owed_tree.clear()
            self.holds_list.clear()
            return
        status = seen.status
        if status.axis and not self.axis_box.text().strip():
            self.axis_box.setText(status.axis)
        if status.test is None:
            self.run_label.setText('No test open: Open Test to begin')
        else:
            running = status.outcome == 'running' and seen.run == status.run
            self.run_label.setText(f'run {seen.run}'
                                   + (' (running)' if running else ' (next)'))
        self.level_row.offer(seen.header, next_level(
            seen.header.get('steps_db'), seen.levels))
        self._fill_tests(seen)
        self._fill_holds(seen)
        self._fill_day(seen)
        self._fill_owed(seen)

    def _fill_tests(self, seen: Snapshot) -> None:
        bar = self.tests_bar
        was = self._shown_test
        # a keeper looking at the current test follows it to the next
        # one when the log rolls; one looking at an earlier test, or at
        # the whole log, stays where they are
        follows = was != ALL and was == self._last_current
        wanted = (seen.status.test if follows and seen.status.test is not None
                  else was)
        bar.blockSignals(True)
        while bar.count():
            bar.removeTab(0)
        bar.addTab('Whole log')
        bar.setTabData(0, ALL)
        for tab in seen.tests:
            index = bar.addTab(f'Test {tab.number}: {tab.kind}'
                               + ('' if tab.state == 'open'
                                  else f' ({tab.state})'))
            bar.setTabData(index, tab.number)
            bar.setTabToolTip(index, tab.title)
        index = next((i for i in range(bar.count())
                      if bar.tabData(i) == wanted), 0)
        bar.setCurrentIndex(index)
        self._shown_test = bar.tabData(index)
        bar.blockSignals(False)
        self._last_current = seen.status.test

    def show_test(self, test: Any) -> None:
        """Show test `test` in the day and the owed lists (`ALL` for the
        whole log)."""
        for index in range(self.tests_bar.count()):
            if self.tests_bar.tabData(index) == test:
                self.tests_bar.setCurrentIndex(index)
                return
        raise KeyError(f'the log has no test {test}')

    def _test_picked(self, index: int) -> None:
        self._shown_test = self.tests_bar.tabData(index)
        if self.snapshot is not None:
            self._fill_day(self.snapshot)
            self._fill_owed(self.snapshot)

    def _fill_holds(self, seen: Snapshot) -> None:
        self.holds_list.clear()
        for hold in seen.holds:
            self.holds_list.addItem(str(hold))
        if not seen.holds:
            self.holds_list.addItem('No holds.')
        # as tall as its holds, up to five; the day below is what the
        # keeper reads most
        rows = min(self.holds_list.count(), 5)
        self.holds_list.setFixedHeight(
            self.holds_list.fontMetrics().lineSpacing() * rows
            + 4 * self.holds_list.frameWidth() + 8)

    def _show_holds(self, shown: bool) -> None:
        self.holds_list.setVisible(shown)

    def _fill_day(self, seen: Snapshot) -> None:
        shown = self._shown_test
        listing = self.day_list
        picked = listing.currentItem()
        picked_id = (picked.data(Qt.ItemDataRole.UserRole).id
                     if picked is not None and isinstance(
                         picked.data(Qt.ItemDataRole.UserRole), Entry)
                     else None)
        listing.clear()
        day = None
        for row in seen.rows:
            entry = row.entry
            if shown != ALL and entry.test != shown:
                continue
            if entry.event_time.date() != day:
                day = entry.event_time.date()
                heading = QListWidgetItem(day.isoformat())
                # enabled, so it reads at full contrast, but not pickable
                heading.setFlags(Qt.ItemFlag.ItemIsEnabled)
                font = QFont(listing.font())
                font.setBold(True)
                heading.setFont(font)
                listing.addItem(heading)
            line = entry_line(row.stored if row.corrected else entry)
            if row.late:
                line += f'   LATE: entered {_clock(entry.entered_at)}'
            item = QListWidgetItem(f'#{entry.id}  {line}')
            item.setData(Qt.ItemDataRole.UserRole, entry)
            if row.corrected:
                font = QFont(listing.font())
                font.setStrikeOut(True)
                item.setFont(font)
                item.setToolTip('Struck through: corrected, as listed '
                                'beneath; the entry stays as written')
            listing.addItem(item)
            if entry.id == picked_id:
                listing.setCurrentItem(item)
            for said in row.corrections:
                note = QListWidgetItem(f'      corrected: {said}')
                note.setData(Qt.ItemDataRole.UserRole, entry)
                listing.addItem(note)
        if picked_id is None:
            listing.scrollToBottom()

    def _fill_owed(self, seen: Snapshot) -> None:
        shown = self._shown_test
        tree = self.owed_tree
        tree.clear()
        for owed in seen.owed:
            if shown != ALL and owed.test not in (shown, None):
                continue
            said = (f'{owed.title}: {len(owed.owed)} of {len(owed.lines)} '
                    'owed')
            top = QTreeWidgetItem([said])
            for line in owed.owed:
                top.addChild(QTreeWidgetItem([line.label]))
            tree.addTopLevelItem(top)
            top.setExpanded(True)
        if not tree.topLevelItemCount():
            tree.addTopLevelItem(QTreeWidgetItem(['Nothing owed.']))

    def day_texts(self) -> list[str]:
        """The day list's lines as shown, for a script or a test."""
        return [self.day_list.item(i).text()
                for i in range(self.day_list.count())]

    def owed_texts(self) -> list[str]:
        """The owed lists' lines as shown, each list's title first."""
        texts: list[str] = []
        for i in range(self.owed_tree.topLevelItemCount()):
            top = self.owed_tree.topLevelItem(i)
            texts.append(top.text(0))
            texts.extend(top.child(j).text(0)
                         for j in range(top.childCount()))
        return texts


# ---- the dock ------------------------------------------------------------------


class RunLogDock(QDockWidget):
    """The Run Log dock: movable, floatable to a second monitor, and
    closable, holding a `RunLogPanel`. Its place is remembered between
    launches (`remember_place`, `restore_place`)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Run Log', parent)
        # the name a main window's saved state knows a dock by
        self.setObjectName('run_log_dock')
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                         | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.panel: RunLogPanel = RunLogPanel()
        self.setWidget(self.panel)
        #: set while `restore_place` moves the dock, so the moves it makes
        #: are not remembered as the keeper's
        self.placing: bool = False


def remember_place(dock: QDockWidget, area: Qt.DockWidgetArea) -> None:
    """Store where `dock` is: floating, with its geometry, or docked in
    `area`."""
    from .preferences import settings

    store = settings()
    store.setValue(PLACE_FLOATING, bool(dock.isFloating()))
    store.setValue(PLACE_AREA, int(area.value))
    if dock.isFloating():
        store.setValue(PLACE_GEOMETRY, dock.saveGeometry())
    store.sync()


def restore_place(dock: QDockWidget, window: Any) -> None:
    """Put `dock` where it was last remembered (`remember_place`): back to
    its floating geometry, or into its dock area of `window`; the right
    edge when nothing is remembered."""
    from .preferences import settings

    store = settings()
    floating = str(store.value(PLACE_FLOATING, 'false')).lower() == 'true'
    geometry = store.value(PLACE_GEOMETRY)
    try:
        area = Qt.DockWidgetArea(int(store.value(
            PLACE_AREA, Qt.DockWidgetArea.RightDockWidgetArea.value)))
    except (TypeError, ValueError):
        area = Qt.DockWidgetArea.RightDockWidgetArea
    if area not in (Qt.DockWidgetArea.LeftDockWidgetArea,
                    Qt.DockWidgetArea.RightDockWidgetArea,
                    Qt.DockWidgetArea.TopDockWidgetArea,
                    Qt.DockWidgetArea.BottomDockWidgetArea):
        area = Qt.DockWidgetArea.RightDockWidgetArea
    placing, dock.placing = getattr(dock, 'placing', False), True
    try:
        window.addDockWidget(area, dock)
        if floating:
            dock.setFloating(True)
            if isinstance(geometry, QByteArray) and not geometry.isEmpty():
                with contextlib.suppress(Exception):
                    dock.restoreGeometry(geometry)
    finally:
        dock.placing = placing


def log_argument(argv: Sequence[str]) -> tuple[list[str], bool, str | None]:
    """Take `--run-log [file]` out of the command line: the arguments
    left, whether the flag was given, and the log named after it (an
    argument ending in `.vdlog`), or None."""
    rest: list[str] = []
    wanted = False
    path = None
    skip = False
    for index, argument in enumerate(argv):
        if skip:
            skip = False
            continue
        if argument == '--run-log':
            wanted = True
            following = argv[index + 1] if index + 1 < len(argv) else None
            if following is not None and following.lower().endswith(SUFFIX):
                path = following
                skip = True
        elif argument.startswith('--run-log='):
            wanted = True
            path = argument.split('=', 1)[1] or None
        else:
            rest.append(argument)
    if path is not None:
        path = os.path.abspath(path)
    return rest, wanted, path
