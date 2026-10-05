"""Reports from runs: the batch of one-call reports, asked in the window.

`visualdynamics.run_report` writes the report each controller run's
type calls for, one per run, and a script hands it the runs, a
geometry, a destination and a marking (Brandon, 2026-10-04: "We made
functions to batch process multi-selected nc4 files. Can we make a way
to call those from the application?"). This dialog asks for the same
four things, so File → Reports from Runs… is that call and nothing
else: the window runs `run_report` with the answers, on its long-verb
runner, with the bar and Cancel.

The runs are listed with the report each one will get, read from the
file the way `run_report` reads it (`report_kind`), so a campaign's
odd one out — a modal survey in a folder of random runs — is seen and
left out here rather than refusing the batch after the button.
"""

from __future__ import annotations

import os
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from .ask import GEOMETRY_FILTER

#: what each kind of one-call report is called where a person reads it
KIND_NAMES = {'random': 'Random Vibration', 'mixed': 'Random and Sine',
              'sine': 'Sine Sweep', 'sysid': 'System ID'}

#: the banner a report is stamped with unless it is changed here — the
#: default `Project.generate_report` uses
DEFAULT_MARKING = 'UNCLASSIFIED'


def run_kind(path: str) -> tuple[str | None, str]:
    """(kind, what the list says) for one run: the one-call report it
    gets, or None and why it gets none."""
    from ..io.rattlesnake import project_type
    from ..project import report_kind

    try:
        kind = report_kind(path)
    except ValueError:
        try:
            declared = project_type(path)
        except (OSError, ValueError):
            declared = None
        return None, (f'{declared}: no one-call report' if declared
                      else 'not a controller run')
    except OSError:
        return None, 'cannot be read'
    return kind, KIND_NAMES[kind]


class RunReportsDialog(QDialog):
    """The runs, the geometry, where the reports go, and their marking.

    `choices()` is the answer in `run_report`'s own keywords: `runs`
    (the reportable ones), `path` (a folder, or None for beside each
    run), `geometry` (a path or None) and `marking`.
    """

    def __init__(self, runs: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Reports from Runs')
        self.kinds: list[tuple[str, str | None, str]] = [
            (path, *run_kind(path)) for path in runs]
        self.geometry: str | None = None
        self.folder: str | None = None

        layout = QVBoxLayout(self)
        reportable = [path for path, kind, _said in self.kinds if kind]
        left_out = len(self.kinds) - len(reportable)
        heading = (f'{len(reportable)} of {len(self.kinds)} runs get a report'
                   if left_out else
                   f'{len(reportable)} run{"s" * (len(reportable) != 1)}, '
                   'one report each')
        layout.addWidget(QLabel(heading))
        self.list: QListWidget = QListWidget()
        for path, kind, said in self.kinds:
            item = QListWidgetItem(f'{os.path.basename(path)} — {said}')
            item.setToolTip(path)
            if kind is None:
                # seen and left out, not hidden: the reader should know
                # which file of the campaign was passed over, and why
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.list.addItem(item)
        layout.addWidget(self.list)

        form = QFormLayout()
        self.geometry_label: QLabel = QLabel('None')
        choose_geometry = QPushButton('Choose…')
        choose_geometry.clicked.connect(self._choose_geometry)
        clear_geometry = QPushButton('None')
        clear_geometry.clicked.connect(lambda: self._set_geometry(None))
        row = QHBoxLayout()
        row.addWidget(self.geometry_label, 1)
        row.addWidget(choose_geometry)
        row.addWidget(clear_geometry)
        form.addRow('Geometry', row)

        # beside each run is what `run_report` does with no path, so it
        # is the default here too; one folder is the other answer
        self.beside: QRadioButton = QRadioButton('Beside each run')
        self.beside.setChecked(True)
        self.in_folder: QRadioButton = QRadioButton('In one folder')
        self.folder_label: QLabel = QLabel('')
        choose_folder = QPushButton('Choose…')
        choose_folder.clicked.connect(self._choose_folder)
        self.in_folder.toggled.connect(
            lambda on: on and self.folder is None and self._choose_folder())
        where = QVBoxLayout()
        where.addWidget(self.beside)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.in_folder)
        folder_row.addWidget(self.folder_label, 1)
        folder_row.addWidget(choose_folder)
        where.addLayout(folder_row)
        form.addRow('Write', where)

        self.marking: QLineEdit = QLineEdit(DEFAULT_MARKING)
        self.marking.setToolTip('The banner across the top and bottom of '
                                'every page of every report')
        form.addRow('Marking', self.marking)
        layout.addLayout(form)

        buttons = QDialogButtonBox()
        self.write_button: QPushButton = buttons.addButton(
            f'Write {len(reportable)} Report{"s" * (len(reportable) != 1)}',
            QDialogButtonBox.ButtonRole.AcceptRole)
        self.write_button.setEnabled(bool(reportable))
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_geometry(self, path: str | None) -> None:
        self.geometry = path or None
        self.geometry_label.setText(
            os.path.basename(path) if path else 'None')
        self.geometry_label.setToolTip(path or '')

    def _choose_geometry(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select a geometry for every report', '', GEOMETRY_FILTER)
        if path:
            self._set_geometry(path)

    def _choose_folder(self) -> None:
        start = os.path.dirname(self.kinds[0][0]) if self.kinds else ''
        folder = QFileDialog.getExistingDirectory(
            self, 'Write the reports into', start)
        if folder:
            self.folder = folder
            self.folder_label.setText(folder)
            self.in_folder.setChecked(True)
        elif self.folder is None:
            self.beside.setChecked(True)

    def choices(self) -> dict[str, Any]:
        """The answer, in `run_report`'s keywords."""
        return {
            'runs': [path for path, kind, _said in self.kinds if kind],
            # a trailing separator: `run_report` reads a path ending in
            # one as a folder, existing or not
            'path': (os.path.join(self.folder, '')
                     if self.in_folder.isChecked() and self.folder else None),
            'geometry': self.geometry,
            'marking': self.marking.text().strip() or DEFAULT_MARKING,
        }


def ask_run_reports(parent: QWidget | None, runs: list[str]
                    ) -> dict[str, Any] | None:
    """The dialog, modal; its choices, or None when cancelled."""
    dialog = RunReportsDialog(runs, parent)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.choices()
    return None
