"""The report editor's pieces for runs and chronology (report feature
`run_sections`).

Two panels, shown only while the feature is on and the editor renders
a project:

- `RunsPanel`, in the pane when no block is selected, under the Report
  Features panel: a row per run with its record, axis, run, level,
  start and comment. A cell's tooltip says where its value came from
  (the file name, the comparison, an entry) or why it was not read. A
  cell typed in enters that field; an empty cell clears it. Under the
  table, **Results for Every Run** repeats the report's Results for
  each run it does not draw on yet.
- `ChronologyPanel`, in the pane of the Chronology section: **Draft from
  the Runs** reads the records and offers a draft, which the page shows
  under the owed section marked as drafted. While a draft is on offer
  and the section is unwritten, the panel shows it and **Accept Draft**
  makes it the section's words, for the author to edit from there.

The panels decide nothing. Each change is emitted as `requested`, and
`operate` makes it through the project's verbs (`set_run_identity`,
`report_runs`, `report_chronology`, `accept_chronology`), so each is
journaled as a script would make it, and refusals come from the core.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

#: the table's columns: (heading, the run identity's field, or '' for
#: the record's name)
COLUMNS = (('Record', ''), ('Axis', 'axis'), ('Run', 'run'),
           ('Level (dB)', 'level_db'), ('Start', 'start'),
           ('Comment', 'comment'))

#: what a refused cell shows until something is entered
NOT_READ = 'not read'


def project_of(objects: Any) -> Any:
    """`objects` when it is a project (what the window hands the editor),
    otherwise None: the panels act through the project's verbs."""
    from ..project import Project

    return objects if isinstance(objects, Project) else None


def shown(objects: Any, report: Any) -> bool:
    """Whether the runs panels belong in the pane: the report has the
    feature on and the editor renders a project."""
    from ..report.features import enabled
    from ..report.runs import FEATURE

    return (report is not None and FEATURE in enabled(report)
            and project_of(objects) is not None)


class RunsPanel(QWidget):
    """The runs of the project, one row each, with their identities.

    `requested` carries one change: {'op': 'run', 'name': record,
    'field': field, 'value': text} for a cell typed in, or {'op':
    'repeat_runs'} for Results for Every Run.
    """

    requested = Signal(dict)

    def __init__(self, project: Any, message: str = '',
                 parent: QWidget | None = None) -> None:
        from ..core.runs import level_text
        from ..report.runs import histories, run_order

        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(QLabel('<b>Runs</b>'))
        intro = QLabel('Each time history is a run. What the file name or '
                       'the comparison could not give one way, enter here; '
                       'a cell left empty goes back to what the record '
                       'shows.')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        names, found = run_order(project, histories(project),
                                  project.object_groups)
        self.names: list[str] = names
        self.table: QTableWidget = QTableWidget(len(names), len(COLUMNS))
        self.table.setHorizontalHeaderLabels([title for title, _f
                                              in COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed)
        for row, name in enumerate(names):
            identity = found[name]
            for column, (_title, field) in enumerate(COLUMNS):
                if not field:
                    item = QTableWidgetItem(name)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setToolTip('the time history')
                else:
                    value = identity.value(field)
                    why = identity.reason(field)
                    if why:
                        text, tip = NOT_READ, f'not read: {why}; enter it'
                    elif value in ('', None):
                        text, tip = '', 'not known; enter it'
                    else:
                        text = (level_text(value).removesuffix(' dB')
                                if field == 'level_db' else str(value))
                        tip = f'from the {identity.source(field)}' \
                            if identity.source(field) != 'entered' \
                            else 'entered'
                    item = QTableWidgetItem(text)
                    item.setToolTip(tip)
                    if why:
                        item.setForeground(Qt.GlobalColor.gray)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self.table.itemChanged.connect(self._edited)
        layout.addWidget(self.table)
        self.repeat_button: QPushButton = QPushButton('Results for Every Run')
        self.repeat_button.setToolTip(
            "Adds the report's Results for its first run again for each "
            "run it does not draw on yet, bound to that run's own objects")
        self.repeat_button.clicked.connect(
            lambda: self.requested.emit({'op': 'repeat_runs'}))
        layout.addWidget(self.repeat_button)
        self.message: QLabel = QLabel(message)
        self.message.setWordWrap(True)
        self.message.setVisible(bool(message))
        layout.addWidget(self.message)

    def _edited(self, item: QTableWidgetItem) -> None:
        field = COLUMNS[item.column()][1]
        if not field:
            return
        text = item.text().strip()
        if text == NOT_READ:
            return
        self.requested.emit({'op': 'run', 'name': self.names[item.row()],
                             'field': field, 'value': text})


class ChronologyPanel(QWidget):
    """Draft from the Runs, and the draft on offer with Accept Draft.

    `requested` carries {'op': 'draft_chronology'} or {'op':
    'accept_chronology'}.
    """

    requested = Signal(dict)

    def __init__(self, offered: str = '', message: str = '',
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self.draft_button: QPushButton = QPushButton('Draft from the Runs')
        self.draft_button.setToolTip(
            'Reads every run and offers a chronology drafted from the '
            'records. The section stays owed until the draft is accepted '
            'or the section is written.')
        self.draft_button.clicked.connect(
            lambda: self.requested.emit({'op': 'draft_chronology'}))
        layout.addWidget(self.draft_button)
        self.draft_view: QPlainTextEdit | None = None
        self.accept_button: QPushButton | None = None
        if offered:
            told = QLabel('Drafted from the records, not yet accepted:')
            told.setWordWrap(True)
            layout.addWidget(told)
            self.draft_view = QPlainTextEdit(offered)
            self.draft_view.setReadOnly(True)
            layout.addWidget(self.draft_view)
            self.accept_button = QPushButton('Accept Draft')
            self.accept_button.setToolTip(
                "Makes the draft the section's words, to edit from there")
            self.accept_button.clicked.connect(
                lambda: self.requested.emit({'op': 'accept_chronology'}))
            layout.addWidget(self.accept_button)
        self.message: QLabel = QLabel(message)
        self.message.setWordWrap(True)
        self.message.setVisible(bool(message))
        layout.addWidget(self.message)


def _level(text: str) -> Any:
    """A typed level as a number, '' to clear; text that is not a number
    is passed on for the core to refuse by name."""
    if not text:
        return ''
    try:
        return float(text.removesuffix('dB').strip())
    except ValueError:
        return text


def operate(project: Any, report: Any, operation: dict,
            run_long: Callable[..., Any] | None = None) -> str:
    """Make one runs-panel change through the project's verbs, and return
    what the panel should say afterwards ('' for nothing).

    A refusal is returned as the message, and nothing changed. Drafting
    the chronology reads every record, so it runs through `run_long`
    (the window's worker) when one is given.
    """
    from ..core.progress import Cancelled

    name = project.name_of(report)
    op = operation.get('op')
    try:
        if op == 'run':
            field = str(operation['field'])
            value = operation.get('value', '')
            if field == 'level_db':
                value = _level(str(value))
            project.set_run_identity(operation['name'], **{field: value})
            return ''
        if op == 'repeat_runs':
            return '\n'.join(project.report_runs(name))
        if op == 'draft_chronology':
            if run_long is not None:
                run_long('Drafting the chronology from the runs',
                         project.report_chronology, name)
            else:
                project.report_chronology(name)
            return ('Drafted from the records. Read it, then accept it or '
                    'write the section yourself.')
        if op == 'accept_chronology':
            project.accept_chronology(name)
            return ''
    except (TypeError, ValueError) as refused:
        return str(refused)
    except Cancelled:
        return 'Drafting the chronology was cancelled; nothing was offered.'
    return ''
