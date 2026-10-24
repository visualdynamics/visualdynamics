"""The dialog that reads a delimited text file: the layout found, set
where it is wrong, on a preview of the channels as they will be read.

A text export says nothing about itself that a reader can trust
without looking: which column is time, at what rate, in which unit each
channel is. So importing one asks, every time. The dialog opens on what
`io.delimited.text_layout` found: the delimiter and the decimal point,
the time column (or none, and a sample rate in its place), and a table
of the channels, each with a box to keep it, its name and its unit as
the header gave them. Anything can be changed, and the preview under
it is read again from the file's head (`PREVIEW_ROWS` rows) by the same
`read_delimited` call the import makes, drawn by
`plot.text_import.build_import_preview`: each channel in its row, its
missing samples crossed, a dead channel named. The notes that read
gives are listed under the preview, and a refusal is shown in their
place with Import disabled, since the import would refuse the same way.

Import then reads the whole file with exactly those arguments, holding
only what differs from what was found, so the journal replays the
import as it was made: `project.import_file(path, sample_rate=5120.0)`.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..io import delimited
from ..io.notes import ImportNote
from ..plot import background_brush
from ..plot.text_import import build_import_preview

# Asked for by the course audit (2026-10-23): a reader brings their own
# recorded data as text, and the import is where a wrong rate or unit
# is cheapest to catch.

#: how many data rows the preview reads from the head of the file
PREVIEW_ROWS = 20000

#: the delimiters offered, in the reader's order
DELIMITER_CHOICES = [(delimited.DELIMITER_NAMES[d].capitalize(), d)
                     for d in delimited.DELIMITERS]

#: the channel table's columns
KEEP, NAME, UNIT = 0, 1, 2


class TextImportDialog(QDialog):
    """How a delimited text file is read, chosen on a preview of its
    channels.

    `path` is the file and `colors` the theme's colors. The dialog opens
    on `io.delimited.text_layout`'s reading of the file.

    `options()` is the answer in `read_delimited`'s own keywords, holding
    only what differs from what was found: an untouched dialog answers
    {}, and the journal line is the plain import.
    """

    def __init__(self, path: str | os.PathLike, colors: Mapping[str, str],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        import pyqtgraph as pg

        self.path: str = str(path)
        self.colors: dict[str, str] = dict(colors)
        #: what the file was found to be, which `options` is measured from
        self.found: delimited.TextLayout = delimited.text_layout(self.path)
        #: the layout as the delimiter and decimal now chosen read it
        self.layout: delimited.TextLayout = self.found
        #: the last preview read, or None when it was refused
        self.history: Any = None
        #: what the last preview read noted, or the refusal
        self.notes: list[str] = []
        self.refusal: str | None = None
        self._building = False

        base = os.path.basename(self.path)
        self.setWindowTitle(f'Import {base}')
        self.header: QLabel = QLabel()
        self.header.setWordWrap(True)

        self.delimiter_box: QComboBox = QComboBox()
        for label, value in DELIMITER_CHOICES:
            self.delimiter_box.addItem(label, value)
        self.decimal_box: QComboBox = QComboBox()
        self.decimal_box.addItem('Point (0.25)', '.')
        self.decimal_box.addItem('Comma (0,25)', ',')
        self.time_box: QComboBox = QComboBox()
        self.time_box.setToolTip('The column that holds each row\'s time, or '
                                 'none when the rows are evenly sampled at '
                                 'the sample rate')
        self.rate_box: QDoubleSpinBox = QDoubleSpinBox()
        self.rate_box.setRange(0.0, 1e9)
        self.rate_box.setDecimals(3)
        self.rate_box.setSuffix(' /s')
        self.rate_box.setSpecialValueText('—')
        self.rate_box.setKeyboardTracking(False)
        self.rate_box.setToolTip('Samples per second. With a time column it '
                                 'must agree with the column\'s step; '
                                 'without one it is the rate the rows were '
                                 'sampled at')

        self.table: QTableWidget = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['Column', 'Name', 'Unit'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.setToolTip('Uncheck a column to leave it out. A name that '
                              'is a DOF (101X+) becomes the channel\'s DOF; '
                              'a unit is declared and converted to SI')

        self.plot: Any = pg.GraphicsLayoutWidget()
        self.plot.setBackground(background_brush(self.colors))
        self.plot.setMinimumSize(560, 360)
        self.message: QLabel = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)

        buttons = QDialogButtonBox()
        self.import_button: Any = buttons.addButton(
            'Import', QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow('Delimiter', self.delimiter_box)
        form.addRow('Decimal point', self.decimal_box)
        form.addRow('Time column', self.time_box)
        form.addRow('Sample rate', self.rate_box)
        side = QVBoxLayout()
        side.addLayout(form)
        side.addWidget(QLabel('Channels'))
        side.addWidget(self.table, 1)
        middle = QHBoxLayout()
        middle.addWidget(self.plot, 1)
        middle.addLayout(side)
        layout = QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addLayout(middle, 1)
        layout.addWidget(self.message)
        layout.addWidget(buttons)

        self._fill(self.found)
        if self.found.sample_rate is not None:
            self.rate_box.setValue(self.found.sample_rate)
        self.delimiter_box.currentIndexChanged.connect(lambda _i: self._relayout())
        self.decimal_box.currentIndexChanged.connect(lambda _i: self._relayout())
        self.time_box.currentIndexChanged.connect(lambda _i: self._time_chosen())
        self.rate_box.valueChanged.connect(lambda _v: self._restate())
        self.table.itemChanged.connect(lambda _item: self._restate())
        self._restate()

    # ---- the settings ---------------------------------------------------

    def _fill(self, layout: delimited.TextLayout) -> None:
        """Show `layout`: its delimiter, decimal, time column and the
        table of its channels."""
        self._building = True
        try:
            self.layout = layout
            self.delimiter_box.setCurrentIndex(
                self.delimiter_box.findData(layout.delimiter))
            self.decimal_box.setCurrentIndex(self.decimal_box.findData(layout.decimal))
            self.time_box.clear()
            self.time_box.addItem('None: rows evenly sampled', None)
            for k, column in enumerate(layout.columns):
                self.time_box.addItem(column, k)
            self.time_box.setCurrentIndex(
                0 if layout.time_column is None else layout.time_column + 1)
            self._fill_table()
        finally:
            self._building = False

    def _fill_table(self) -> None:
        """One row per column but the time column: kept, named and in the
        unit its header gave."""
        self._building, was = True, self._building
        try:
            time = self.time_column()
            self.table.setRowCount(0)
            for k, column in enumerate(self.layout.columns):
                if k == time:
                    continue
                row = self.table.rowCount()
                self.table.insertRow(row)
                keep = QTableWidgetItem(column)
                keep.setData(Qt.ItemDataRole.UserRole, k)
                keep.setFlags(Qt.ItemFlag.ItemIsEnabled
                              | Qt.ItemFlag.ItemIsUserCheckable)
                keep.setCheckState(Qt.CheckState.Checked)
                self.table.setItem(row, KEEP, keep)
                self.table.setItem(row, NAME, QTableWidgetItem(column))
                self.table.setItem(row, UNIT, QTableWidgetItem(
                    self.layout.units[k] or ''))
            self.table.resizeColumnsToContents()
        finally:
            self._building = was

    def _relayout(self) -> None:
        """A delimiter or decimal chosen: the file read again with it,
        which can change every column."""
        if self._building:
            return
        delimiter = self.delimiter_box.currentData()
        decimal = self.decimal_box.currentData()
        try:
            layout = delimited.text_layout(self.path, delimiter=delimiter,
                                           decimal=decimal)
        except ValueError as refusal:
            self._refuse(str(refusal))
            return
        self._fill(layout)
        # _fill puts the boxes back to what the layout read; the choice
        # just made is the person's and stays
        self._building = True
        self.delimiter_box.setCurrentIndex(self.delimiter_box.findData(delimiter))
        self.decimal_box.setCurrentIndex(self.decimal_box.findData(decimal))
        self._building = False
        self._restate()

    def _time_chosen(self) -> None:
        if self._building:
            return
        self._fill_table()
        self._restate()

    # ---- the answer -------------------------------------------------------

    def time_column(self) -> int | None:
        """The time column chosen, by index, or None."""
        return self.time_box.currentData()

    def _rows(self) -> list[tuple[int, bool, str, str]]:
        """(column index, kept, name, unit) per row of the table."""
        out = []
        for row in range(self.table.rowCount()):
            keep = self.table.item(row, KEEP)
            name = self.table.item(row, NAME)
            unit = self.table.item(row, UNIT)
            out.append((int(keep.data(Qt.ItemDataRole.UserRole)),
                        keep.checkState() == Qt.CheckState.Checked,
                        name.text().strip() if name else '',
                        unit.text().strip() if unit else ''))
        return out

    def options(self) -> dict[str, Any]:
        """The import's keyword arguments, holding only what differs from
        what was found: 'delimiter' and 'decimal', 'time_column' (a name,
        or False for none), 'sample_rate', 'columns' when some are left
        out, 'names' when one was changed and 'units' by column name
        where one differs from the header's."""
        out: dict[str, Any] = {}
        layout = self.layout
        if layout.delimiter != self.found.delimiter:
            out['delimiter'] = layout.delimiter
        if layout.decimal != self.found.decimal:
            out['decimal'] = layout.decimal
        # measured from the layout as now read: the import reads it the
        # same way from the delimiter and decimal stated
        time = self.time_column()
        if time != layout.time_column:
            out['time_column'] = False if time is None else layout.columns[time]
        rate = float(self.rate_box.value())
        if rate > 0 and rate != (layout.sample_rate or 0.0):
            out['sample_rate'] = rate
        rows = self._rows()
        kept = [(k, name, unit) for k, keep, name, unit in rows if keep]
        if len(kept) < len(rows):
            out['columns'] = [layout.columns[k] for k, _n, _u in kept]
        if any(name != layout.columns[k] for k, name, _u in kept):
            out['names'] = [name or layout.columns[k] for k, name, _u in kept]
        units = {layout.columns[k]: unit for k, _n, unit in kept
                 if unit and unit != (layout.units[k] or '')}
        if units:
            out['units'] = units
        return out

    # ---- the preview ------------------------------------------------------

    def _refuse(self, refusal: str) -> None:
        self.history, self.notes, self.refusal = None, [], refusal
        self.plot.clear()
        self.message.setText(f'Cannot import: {refusal}')
        self.message.setStyleSheet(f'color: {self.colors["limit_abort"]}')
        self.import_button.setEnabled(False)

    def _restate(self) -> None:
        """Read the file's head with the options as they stand and show
        it: the header's words, the preview, and the notes or the
        refusal."""
        if self._building:
            return
        self.header.setText(self.layout.describe(self.path)
                            + '. Set what is wrong; the preview is the head of '
                              f'the file, up to {PREVIEW_ROWS:,} rows.')
        options = self.options()
        if not any(keep for _k, keep, _n, _u in self._rows()):
            self._refuse('no column is checked')
            return
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                history = delimited.read_delimited(self.path, max_rows=PREVIEW_ROWS,
                                                   **options)
        except ValueError as refusal:
            self._refuse(str(refusal))
            return
        self.history, self.refusal = history, None
        self.notes = [str(w.message) for w in caught
                      if issubclass(w.category, ImportNote)]
        self.plot.clear()
        build_import_preview(self.plot, history, theme=self.colors)
        rate = f'{history.sample_rate:.6g}'
        said = (f'{history.num_records} channel'
                f'{"s" * (history.num_records != 1)} at {rate} samples per second.')
        self.message.setText(' '.join([said, *self.notes]))
        self.message.setStyleSheet(
            f'color: {self.colors["limit_abort"]}' if self.notes else '')
        self.import_button.setEnabled(True)


def ask_text_import(parent: QWidget | None, path: str | os.PathLike,
                    colors: Mapping[str, str]) -> dict[str, Any] | None:
    """Show the dialog and return the import options chosen, or None when
    it is cancelled and the file is skipped."""
    dialog = TextImportDialog(path, colors, parent)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.options()
    return None
