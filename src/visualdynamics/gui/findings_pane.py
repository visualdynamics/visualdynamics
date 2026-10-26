"""The findings register in the report editor's pane (report feature
`findings`).

Shown under section 9's words when the deviations section is selected
and the report has the findings register on. A list of the findings
entered, by kind and number, each with the parts it still owes; a kind
and Add to enter one; and, for the finding chosen, its run, channel,
band and status, its four parts in the author's words, and a row that
reads a draft of its fact off a comparison the project holds, with the
draft shown beside Accept and Discard until the author decides.

The pane decides nothing. Every change is emitted as `requested`, and
the editor makes it through `report.findings`, which refuses what
cannot be entered; the pane is built again to show the outcome, with
the refusal's words when there was one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import findings as core

#: how long typing in a part pauses before it is entered, in ms
PART_DEBOUNCE_MS = 600


def _edge(text: str) -> Any:
    """A band edge as typed: a number when it reads as one, else the
    text, for the register to refuse by name."""
    try:
        return float(text)
    except ValueError:
        return text


class FindingsPane(QWidget):
    """The register's list, the chosen finding's fields, and its draft.

    `requested` carries one change as a dict with 'action': 'select',
    'add' (with 'kind'), 'remove', 'edit' (with 'fields'), 'draft' (with
    'measured', 'specification' and 'limits'), 'accept' or 'discard',
    each but 'add' with the finding's 'number'.
    """

    requested = Signal(dict)

    def __init__(self, entries: Sequence[Mapping[str, Any]],
                 selected: str | None, measured: Sequence[str] = (),
                 specifications: Sequence[str] = (), message: str = '',
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = True
        #: the register's findings, in its order, as `report.findings` reads them
        self.entries: list[Mapping[str, Any]] = list(entries)
        chosen = [entry for entry in self.entries
                  if entry['number'] == selected]
        #: the finding whose fields are shown, or None
        self.entry: Mapping[str, Any] | None = chosen[0] if chosen else None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(QLabel('<b>Findings register</b>'))
        intro = QLabel('Each finding in four parts. A blank part is owed '
                       'on the page; None above says a kind has none.')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.list: QListWidget = QListWidget()
        for entry in self.entries:
            owed = core.owed(entry)
            label = f'{entry["number"]}  {entry["kind"].capitalize()}'
            if owed:
                named = [core.FOURTH[entry['kind']].lower()
                         if part == 'approval' else part for part in owed]
                label += f'  (owes {", ".join(named)})'
            if entry['draft'] is not None:
                label += '  [draft waiting]'
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, entry['number'])
            self.list.addItem(item)
        if self.entry is not None:
            self.list.setCurrentRow(self.entries.index(self.entry))
        self.list.currentRowChanged.connect(self._chosen)
        self.list.setMaximumHeight(120)
        layout.addWidget(self.list)
        adding = QHBoxLayout()
        self.kind_box: QComboBox = QComboBox()
        for kind in core.KINDS:
            self.kind_box.addItem(kind.capitalize(), kind)
        self.add_button: QPushButton = QPushButton('Add')
        self.add_button.setToolTip('Enter a finding of this kind under the '
                                   'next number of its kind')
        self.add_button.clicked.connect(lambda: self.requested.emit(
            {'action': 'add', 'kind': self.kind_box.currentData()}))
        self.remove_button: QPushButton = QPushButton('Remove')
        self.remove_button.setEnabled(self.entry is not None)
        self.remove_button.clicked.connect(lambda: self._ask('remove'))
        adding.addWidget(self.kind_box, 1)
        adding.addWidget(self.add_button)
        adding.addWidget(self.remove_button)
        layout.addLayout(adding)
        #: the chosen finding's line fields and part editors, by field
        self.lines: dict[str, QLineEdit] = {}
        self.parts: dict[str, QPlainTextEdit] = {}
        self.status_box: QComboBox | None = None
        self.measured_box: QComboBox | None = None
        self.specification_box: QComboBox | None = None
        self.limits_box: QComboBox | None = None
        self.draft_button: QPushButton | None = None
        self.accept_button: QPushButton | None = None
        self.discard_button: QPushButton | None = None
        self._timer: QTimer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(PART_DEBOUNCE_MS)
        self._timer.timeout.connect(self.flush)
        if self.entry is not None:
            layout.addWidget(self._fields(self.entry, measured,
                                          specifications))
        self.note: QLabel = QLabel(message)
        self.note.setWordWrap(True)
        self.note.setStyleSheet('color: #c0392b')
        self.note.setVisible(bool(message))
        layout.addWidget(self.note)
        self._loading = False

    def _fields(self, entry: Mapping[str, Any], measured: Sequence[str],
                specifications: Sequence[str]) -> QWidget:
        body = QWidget()
        form = QFormLayout(body)
        form.setContentsMargins(0, 4, 0, 0)
        form.addRow(QLabel(f'<b>{entry["number"]}</b> '
                           f'{entry["kind"].capitalize()}'))
        for field, label, hint in (
                ('run', 'Run', 'the run it concerns'),
                ('channel', 'Channel', 'the channel, such as 2Y+')):
            edit = QLineEdit(entry[field])
            edit.setPlaceholderText(hint)
            edit.editingFinished.connect(
                lambda f=field, e=edit: self._line_edited(f, e.text()))
            self.lines[field] = edit
            form.addRow(label, edit)
        band = entry['band'] or ('', '')
        row = QHBoxLayout()
        for field, value in (('low', band[0]), ('high', band[1])):
            edit = QLineEdit('' if value == '' else f'{value:g}')
            edit.setPlaceholderText(f'{field}, Hz')
            edit.editingFinished.connect(self._band_edited)
            self.lines[field] = edit
            row.addWidget(edit)
        form.addRow('Band', row)
        self.status_box = QComboBox()
        for value in core.STATUSES:
            self.status_box.addItem(value.capitalize() or '(not stated)',
                                    value)
        self.status_box.setCurrentIndex(core.STATUSES.index(entry['status']))
        self.status_box.currentIndexChanged.connect(
            lambda _i: self._edit({'status': self.status_box.currentData()}))
        form.addRow('Status', self.status_box)
        for part in core.PARTS:
            label = (core.FOURTH[entry['kind']] if part == 'approval'
                     else part.capitalize())
            edit = QPlainTextEdit(entry[part])
            edit.setPlaceholderText(f'{label}, in your words; owed until '
                                    'written')
            edit.setMaximumHeight(70)
            edit.textChanged.connect(self._typed)
            self.parts[part] = edit
            form.addRow(label, edit)
        form.addRow(QLabel('Draft the fact from a measurement'))
        self.measured_box = QComboBox()
        for name in measured:
            self.measured_box.addItem(name, name)
        self.specification_box = QComboBox()
        for name in specifications:
            self.specification_box.addItem(name, name)
        self.limits_box = QComboBox()
        for pair in reversed(core.PAIRS):
            self.limits_box.addItem(f'{pair.capitalize()} limits', pair)
        form.addRow('Measured', self.measured_box)
        form.addRow('Against', self.specification_box)
        form.addRow('Limits', self.limits_box)
        self.draft_button = QPushButton('Draft the fact')
        self.draft_button.setToolTip(
            "Read the channel's excursion off the comparison, over the "
            'band if one is entered, as a draft to accept or discard')
        self.draft_button.setEnabled(bool(measured) and bool(specifications))
        self.draft_button.clicked.connect(self._draft)
        form.addRow(self.draft_button)
        draft = entry['draft']
        if draft is not None:
            # plain text: the draft quotes object names, which may hold
            # anything
            shown = QLabel(f'Draft ({draft["source"]}): {draft["text"]}')
            shown.setTextFormat(Qt.TextFormat.PlainText)
            shown.setWordWrap(True)
            form.addRow(shown)
            decide = QHBoxLayout()
            self.accept_button = QPushButton('Accept')
            self.accept_button.setToolTip('Make the draft the fact; edit it '
                                          'in the Fact box after')
            self.accept_button.clicked.connect(lambda: self._ask('accept'))
            self.discard_button = QPushButton('Discard')
            self.discard_button.clicked.connect(lambda: self._ask('discard'))
            decide.addWidget(self.accept_button)
            decide.addWidget(self.discard_button)
            form.addRow(decide)
        return body

    def _chosen(self, row: int) -> None:
        if self._loading or not 0 <= row < len(self.entries):
            return
        self.flush()
        self.requested.emit({'action': 'select',
                             'number': self.entries[row]['number']})

    def _ask(self, action: str) -> None:
        if self.entry is not None:
            self.flush()
            self.requested.emit({'action': action,
                                 'number': self.entry['number']})

    def _edit(self, fields: dict[str, Any]) -> None:
        if self._loading or self.entry is None or not fields:
            return
        # what is entered now, so an unchanged field is not entered
        # again; a refusal builds the pane again from the register
        self.entry = {**self.entry, **fields}
        self.requested.emit({'action': 'edit',
                             'number': self.entry['number'],
                             'fields': fields})

    def _line_edited(self, field: str, text: str) -> None:
        if self.entry is not None and text.strip() != self.entry[field]:
            self._edit({field: text.strip()})

    def _band_edited(self) -> None:
        if self.entry is None:
            return
        low = self.lines['low'].text().strip()
        high = self.lines['high'].text().strip()
        band = [] if not low and not high else [_edge(low), _edge(high)]
        held = self.entry['band'] or []
        if band != held:
            self._edit({'band': band})

    def _typed(self) -> None:
        if not self._loading:
            self._timer.start()

    def flush(self) -> None:
        """Enter whatever has been typed in the parts and not entered
        yet. The pane does this itself once typing pauses for
        `PART_DEBOUNCE_MS`, and the editor before anything else."""
        self._timer.stop()
        if self.entry is None:
            return
        changed = {part: edit.toPlainText()
                   for part, edit in self.parts.items()
                   if edit.toPlainText().strip() != self.entry[part].strip()}
        if changed:
            self._edit(changed)

    def _draft(self) -> None:
        if self.entry is None or self.measured_box is None:
            return
        self.flush()
        self.requested.emit({
            'action': 'draft', 'number': self.entry['number'],
            'measured': self.measured_box.currentData(),
            'specification': self.specification_box.currentData(),
            'limits': self.limits_box.currentData()})
