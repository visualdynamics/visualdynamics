"""The shaker's ratings beside a reference record checked against them.

The Replication Fit reading draws a record's travel at one high-pass
cutoff inside the stroke, over the sweep of every candidate cutoff
(`plot.replication_fit`). This panel holds what the record is checked
against, in the display units: the force, stroke and velocity ratings,
the moving mass, the body's mass when its share of the motion is to be
counted, and the candidate cutoffs. Under
them it reads out the record's largest peak and the force it asks for,
then a table with a column per candidate, the record as logged first:
the travel each way, the peak velocity (both with the body counted
when its mass is given), the share of the mean square the cutoff
removed, the change in the largest peak, and the verdict on the stroke,
the velocity and the force at that cutoff's largest peak.
Clicking a column draws that cutoff.

Every edit redraws at once, so the effect of a rating or a cutoff is
seen on the record while it is set. Nothing is made here: the check is
a reading of the record, as the kurtosis is. The edited record itself
is made with the Filter reading, a high-pass at the chosen cutoff of
order 2, the filter this check applies. Every number shown comes from
`core.replication_fit.replication_fit`, the call a script makes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from ..core.replication_fit import CUTOFFS
from ..units import DEFAULT_SYSTEM
from .editors import DoubleSpinBox
from .settings_panel import add_derived, add_keyed_settings, add_rule, panel_grid

# A starting point to be overwritten, so the reading draws the moment
# it is chosen rather than waiting on four fields: a mid-size shaker of
# 4000 lbf, 2 in of stroke and 70 in/s carrying 150 lb, in SI. Nothing
# is judged against these until the user has said what the shaker is;
# the panel's note says so while they stand.
DEFAULTS: dict[str, Any] = {
    'force': 17792.89, 'stroke': 0.0508, 'velocity': 1.778,
    'moving_mass': 68.04, 'body_mass': None,
    'cutoffs': CUTOFFS, 'cutoff': ...}

#: the editable ratings, as (key, label, dimension)
RATINGS = (('force', 'Force', 'force'),
           ('stroke', 'Stroke p-p', 'length'),
           ('velocity', 'Velocity', 'velocity'),
           ('moving_mass', 'Moving mass', 'mass'),
           ('body_mass', 'Body mass', 'mass'))

#: the table's rows, as (label, how the cell is written)
ROWS = ('Down', 'Up', 'Velocity', 'Mean sq. out', 'Peak change', 'Fits')

#: the panel's width: wider than a settings panel, for the table's
#: column per candidate
WIDTH = 340


def cutoffs_text(cutoffs: Sequence[float]) -> str:
    """The cutoffs as the field shows them, '2, 3, 5, 8'."""
    return ', '.join(f'{c:g}' for c in cutoffs)


def parse_cutoffs(text: str) -> tuple[float, ...] | None:
    """The cutoffs typed in the field, sorted and without repeats, or
    None when the text is not a list of numbers above zero."""
    parts = [part for part in text.replace(',', ' ').split() if part]
    try:
        values = sorted({float(part) for part in parts})
    except ValueError:
        return None
    if not values or values[0] <= 0.0:
        return None
    return tuple(values)


class ReplicationFitPanel(QWidget):
    """The ratings, the candidate cutoffs and the readout. Each edit is
    emitted as the whole settings dict, in SI: the `ShakerRating` fields,
    'cutoffs', and 'cutoff', the one drawn (None for the record as
    logged, ``...`` for the default)."""

    #: a setting moved, or a column was clicked; here is the whole set
    changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Replication Fit')
        grid = panel_grid(self, self.title)
        self.setFixedWidth(WIDTH)
        self._loading = False
        self._settings: dict[str, Any] = dict(DEFAULTS)
        self._unit_system = DEFAULT_SYSTEM
        #: the cutoff of each table column, None for the record as logged
        self._columns: list[float | None] = []

        self.boxes: dict[str, DoubleSpinBox] = {}
        rows = []
        for key, label, _dimension in RATINGS:
            box = DoubleSpinBox()
            box.setDecimals(4)
            box.setRange(0.0, 1e9)
            self.boxes[key] = box
            rows.append((key, label, box))
        self.boxes['force'].setToolTip("The shaker's force rating")
        self.boxes['stroke'].setToolTip("The shaker's stroke, peak to peak")
        self.boxes['velocity'].setToolTip("The shaker's velocity rating")
        self.boxes['body_mass'].setSpecialValueText('not counted')
        self.boxes['body_mass'].setToolTip(
            'The shaker body’s mass. Given, the body’s motion '
            'opposite the armature is counted in the travel; at its '
            'lowest the body is not counted')
        self.boxes['moving_mass'].setToolTip(
            'Everything the shaker accelerates: armature, head expander '
            'or slip table, fixture and article')
        self.cutoffs_edit: QLineEdit = QLineEdit(cutoffs_text(CUTOFFS))
        self.cutoffs_edit.setToolTip(
            'The candidate high-pass cutoffs, in Hz, separated by commas')
        rows.append(('cutoffs', 'Cutoffs [Hz]', self.cutoffs_edit))
        self.names, self.editors = add_keyed_settings(grid, rows)
        add_rule(grid, len(rows) + 1)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value) in add_derived(
                grid, (('peak', 'Largest peak'), ('force', 'Force needed'),
                       ('crest', 'Crest factor')),
                len(rows) + 2).items()}
        first = len(rows) + 5
        self.table: QTableWidget = QTableWidget(len(ROWS), 0)
        self.table.setVerticalHeaderLabels(list(ROWS))
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectColumns)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.table.setToolTip('Click a column to draw that cutoff, or the '
                              'record as logged')
        self.table.itemSelectionChanged.connect(self._column_picked)
        grid.addWidget(self.table, first, 0, 1, 2)
        self.note: QLabel = QLabel('')
        self.note.setWordWrap(True)
        grid.addWidget(self.note, first + 1, 0, 1, 2)
        grid.setRowStretch(first + 2, 1)

        for key, box in self.boxes.items():
            box.valueChanged.connect(
                lambda _value, key=key: self._edited(key))
        self.cutoffs_edit.editingFinished.connect(self._cutoffs_edited)
        self.show_settings(DEFAULTS, DEFAULT_SYSTEM)

    # ---- the settings -----------------------------------------------------

    def show_settings(self, settings: dict[str, Any] | None,
                      unit_system: Any = None) -> dict[str, Any]:
        """Show `settings` (SI; `DEFAULTS` for None) in `unit_system`'s
        units, without emitting, and return the settings now held."""
        us = DEFAULT_SYSTEM if unit_system is None else unit_system
        self._unit_system = us
        self._settings = dict(DEFAULTS if settings is None else settings)
        self._loading = True
        try:
            for key, label, dimension in RATINGS:
                box = self.boxes[key]
                self.names[key].setText(
                    f'{label} [{us.label_text(dimension)}]')
                value = self._settings[key]
                box.setValue(0.0 if value is None
                             else float(us.from_si(value, dimension)))
            self.cutoffs_edit.setText(cutoffs_text(self._settings['cutoffs']))
        finally:
            self._loading = False
        return dict(self._settings)

    def settings(self) -> dict[str, Any]:
        """The settings held, in SI."""
        return dict(self._settings)

    def rating(self) -> dict[str, Any]:
        """The `ShakerRating` keyword arguments, in SI."""
        return {key: self._settings[key] for key, _label, _dim in RATINGS}

    def _edited(self, key: str) -> None:
        """One rating moved: only that one is read back, so the others
        are not nudged by a round trip through their display units."""
        if self._loading:
            return
        dimension = next(dim for k, _label, dim in RATINGS if k == key)
        value = self.boxes[key].value()
        if key == 'body_mass' and value <= 0.0:
            self._settings[key] = None
        else:
            self._settings[key] = float(
                self._unit_system.to_si(value, dimension))
        self.changed.emit(self.settings())

    def _cutoffs_edited(self) -> None:
        """The cutoffs field finished: kept when it reads as a list of
        cutoffs, put back as it was when it does not."""
        if self._loading:
            return
        cutoffs = parse_cutoffs(self.cutoffs_edit.text())
        if cutoffs is None:
            self.cutoffs_edit.setText(cutoffs_text(self._settings['cutoffs']))
            return
        if cutoffs == tuple(self._settings['cutoffs']):
            return
        self._settings['cutoffs'] = cutoffs
        if self._settings['cutoff'] not in (None, ...) \
                and self._settings['cutoff'] not in cutoffs:
            self._settings['cutoff'] = ...
        self.changed.emit(self.settings())

    def _column_picked(self) -> None:
        if self._loading:
            return
        columns = {index.column() for index in self.table.selectedIndexes()}
        if len(columns) != 1:
            return
        cutoff = self._columns[columns.pop()]
        if cutoff == self._settings['cutoff']:
            return
        self._settings['cutoff'] = cutoff
        self.changed.emit(self.settings())

    # ---- the readout ------------------------------------------------------

    def show_fit(self, fit: Any, drawn: Any = None) -> None:
        """Write the readout for a `core.replication_fit.ReplicationFit`,
        with the column of the case `drawn` selected; None clears it."""
        us = self._unit_system
        self._loading = True
        try:
            self.table.clearSelection()
            if fit is None:
                for value in self.derived.values():
                    value.setText('—')
                self.table.setColumnCount(0)
                self._columns = []
                self.note.setText('')
                return
            g = us.label_text('acceleration')
            self.derived['peak'].setText(
                f'{float(us.from_si(fit.peak_acceleration, "acceleration")):.4g}'
                f' {g} at {fit.peak_time:.4g} s')
            force = float(us.from_si(fit.force, 'force'))
            self.derived['force'].setText(
                f'{force:.4g} {us.label_text("force")}, '
                f'{100.0 * fit.force_share:.0f} %')
            self.derived['crest'].setText(f'{fit.crest_factor:.3g}')
            cases = [fit.as_logged, *fit.cases]
            self._columns = [case.cutoff for case in cases]
            self.table.setColumnCount(len(cases))
            self.table.setHorizontalHeaderLabels(
                ['Logged' if case.cutoff is None else f'{case.cutoff:g} Hz'
                 for case in cases])
            length = us.label_text('length')
            for column, case in enumerate(cases):
                cells = (
                    f'{float(us.from_si(case.travel_down, "length")):.3g}',
                    f'{float(us.from_si(case.travel_up, "length")):.3g}',
                    f'{float(us.from_si(case.relative_velocity, "velocity")):.3g}',
                    f'{100.0 * case.mean_square_removed:.2f}',
                    f'{100.0 * case.peak_change:+.2f}',
                    'yes' if case.fits else 'no')
                for row, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                    self.table.setItem(row, column, item)
            self.table.setVerticalHeaderLabels(
                [f'Down [{length}]', f'Up [{length}]',
                 f'Velocity [{us.label_text("velocity")}]',
                 'Mean sq. out [%]', 'Peak change [%]', 'Fits'])
            if drawn is not None and drawn.cutoff in self._columns:
                self.table.selectColumn(self._columns.index(drawn.cutoff))
            words = fit.describe()
            if self._settings_are_defaults():
                words += (' The ratings are a starting point: set them to '
                          'the shaker’s own.')
            self.note.setText(words)
        finally:
            self._loading = False

    def _settings_are_defaults(self) -> bool:
        return all(self._settings[key] == DEFAULTS[key]
                   for key, _label, _dim in RATINGS)
