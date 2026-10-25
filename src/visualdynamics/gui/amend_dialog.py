"""The dialog that asks for an amendment to an issued report: its own
revision, its own number if it has one, and what it changes.

The answer goes to `report.identity.amend`, which refuses what is
missing; the dialog only collects it.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class AmendDialog(QDialog):
    """Asks for an amendment: the revision, the number, and a table of
    changes (where, was, now, reason), one row per change."""

    def __init__(self, report: Any, parent: Any = None) -> None:
        super().__init__(parent)
        from ..report.identity import CHANGE_PARTS, name

        self.setWindowTitle('Amend Report')
        layout = QVBoxLayout(self)
        told = QLabel(f'{name(report)} is issued. An amendment is issued in '
                      'its turn, under its own revision, with the changes '
                      'listed on its first page; the issued report is kept.')
        told.setWordWrap(True)
        layout.addWidget(told)
        form = QFormLayout()
        self.revision = QLineEdit()
        self.revision.setPlaceholderText('the amendment\'s revision')
        form.addRow('Revision', self.revision)
        self.number = QLineEdit()
        self.number.setPlaceholderText(
            f'{report.identity.get("number", "")} (the original\'s, unless '
            'the amendment has its own)')
        form.addRow('Report number', self.number)
        layout.addLayout(form)
        self.changes = QTableWidget(1, len(CHANGE_PARTS))
        self.changes.setHorizontalHeaderLabels(
            [part.capitalize() for part in CHANGE_PARTS])
        self.changes.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.changes)
        more = QPushButton('Add Change')
        more.clicked.connect(
            lambda: self.changes.insertRow(self.changes.rowCount()))
        layout.addWidget(more)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(640, 360)

    def amendment(self) -> dict[str, Any]:
        """What was entered: 'revision', 'number' (None when left blank)
        and 'changes', a mapping per row that holds anything, in order."""
        from ..report.identity import CHANGE_PARTS

        changes = []
        for row in range(self.changes.rowCount()):
            cells = {}
            for column, part in enumerate(CHANGE_PARTS):
                item = self.changes.item(row, column)
                cells[part] = item.text() if item is not None else ''
            if any(text.strip() for text in cells.values()):
                changes.append(cells)
        number = self.number.text().strip()
        return {'revision': self.revision.text(),
                'number': number or None, 'changes': changes}

    def fill(self, revision: str, changes: list[dict[str, str]],
             number: str = '') -> None:
        """Enter an amendment, as a person would type it."""
        from ..report.identity import CHANGE_PARTS

        self.revision.setText(revision)
        self.number.setText(number)
        self.changes.setRowCount(max(len(changes), 1))
        for row, change in enumerate(changes):
            for column, part in enumerate(CHANGE_PARTS):
                self.changes.setItem(row, column,
                                     QTableWidgetItem(change.get(part, '')))
