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
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .ask import GEOMETRY_FILTER

#: what each kind of one-call report is called where a person reads it
KIND_NAMES = {'random': 'Random Vibration',
              'mixed': 'Random and Sine: a random and a sine report',
              'sine': 'Sine Sweep', 'sysid': 'System ID'}

#: what a run's Report box offers, in order: every one-call report, then
#: leaving the run out (2026-10-06: the detected report is a default,
#: not a verdict — a restarted random run the system-ID guess took for
#: one, or a file that declares no type, is put right here)
CHOICES = [*((KIND_NAMES[kind], kind)
             for kind in ('random', 'mixed', 'sine', 'sysid')),
           ('Leave out', None)]

#: the reports the "every random report" box sets a Last for: a random
#: run's, and a random-and-sine run's random half (its sine report
#: reads the whole run whatever Last says)
RANDOM_KINDS = ('random', 'mixed')

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
    """The runs, the report each gets and how much of it to read, the
    geometry, where the reports go, and their marking.

    `choices()` is the answer in `run_report`'s own keywords: `runs`
    (the ones not left out), `kinds` and `last` (each by run), `path`
    (a folder, or None for beside each run), `geometry` (a path or
    None) and `marking`.
    """

    def __init__(self, runs: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Reports from Runs')
        self.kinds: list[tuple[str, str | None, str]] = [
            (path, *run_kind(path)) for path in runs]
        self.geometry: str | None = None
        self.folder: str | None = None

        layout = QVBoxLayout(self)
        self.heading: QLabel = QLabel()
        layout.addWidget(self.heading)

        # a run a row: its name, the report it gets (detected, or Leave
        # out when it has none) and how much of its end to read
        self.table: QTableWidget = QTableWidget(len(self.kinds), 3)
        self.table.setHorizontalHeaderLabels(['Run', 'Report', 'Last'])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.reports: list[QComboBox] = []
        self.lasts: list[QDoubleSpinBox] = []
        for row, (path, kind, said) in enumerate(self.kinds):
            name = QTableWidgetItem(os.path.basename(path) if kind
                                    else f'{os.path.basename(path)} — {said}')
            # the reason in full: a narrow column cuts it off
            name.setToolTip(path if kind else f'{path}\n{said}')
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, name)
            box = QComboBox()
            for label, value in CHOICES:
                box.addItem(label, value)
            # seen and left out, not hidden: the reader should know
            # which file of the campaign was passed over, and why
            box.setCurrentIndex(box.findData(kind) if kind
                                else len(CHOICES) - 1)
            box.currentIndexChanged.connect(self._restate)
            self.table.setCellWidget(row, 1, box)
            self.reports.append(box)
            span = _seconds_box()
            self.table.setCellWidget(row, 2, span)
            self.lasts.append(span)
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)

        # most random reports want only the end of the run: one number
        # for all of them, each row still its own to change
        self.every_random: QDoubleSpinBox = _seconds_box()
        self.every_random.valueChanged.connect(self._last_for_random)
        every = QHBoxLayout()
        every.addWidget(QLabel('Last, for every random report'))
        every.addWidget(self.every_random)
        every.addStretch(1)
        layout.addLayout(every)

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
            '', QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._restate()

    def _picked(self) -> list[tuple[str, str | None, float | None]]:
        """(run, report, last seconds or None) for every row."""
        return [(path, box.currentData(), span.value() or None)
                for (path, _kind, _said), box, span
                in zip(self.kinds, self.reports, self.lasts)]

    def _restate(self) -> None:
        """The heading and the button, from the rows as they stand. A
        random-and-sine run is two reports (2026-10-05), so the button
        counts reports and the heading counts runs."""
        picked = [kind for _path, kind, _last in self._picked() if kind]
        reports = sum(2 if kind == 'mixed' else 1 for kind in picked)
        self.heading.setText(
            f'{len(picked)} of {len(self.kinds)} runs, '
            f'{reports} report{"s" * (reports != 1)}')
        self.write_button.setText(
            f'Write {reports} Report{"s" * (reports != 1)}')
        self.write_button.setEnabled(bool(picked))

    def _last_for_random(self, seconds: float) -> None:
        for box, span in zip(self.reports, self.lasts):
            if box.currentData() in RANDOM_KINDS:
                span.setValue(seconds)

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
        picked = [row for row in self._picked() if row[1]]
        return {
            'runs': [path for path, _kind, _last in picked],
            'kinds': {path: kind for path, kind, _last in picked},
            'last': {path: last for path, _kind, last in picked},
            # a trailing separator: `run_report` reads a path ending in
            # one as a folder, existing or not
            'path': (os.path.join(self.folder, '')
                     if self.in_folder.isChecked() and self.folder else None),
            'geometry': self.geometry,
            'marking': self.marking.text().strip() or DEFAULT_MARKING,
        }


def _seconds_box() -> QDoubleSpinBox:
    """How much of a run's end to read, in seconds; zero reads as the
    whole run, which is what a blank `last` means."""
    box = QDoubleSpinBox()
    box.setRange(0.0, 1e6)
    box.setDecimals(1)
    box.setSuffix(' s')
    box.setSpecialValueText('whole run')
    return box


def ask_run_reports(parent: QWidget | None, runs: list[str]
                    ) -> dict[str, Any] | None:
    """The dialog, modal; its choices, or None when cancelled."""
    dialog = RunReportsDialog(runs, parent)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.choices()
    return None
