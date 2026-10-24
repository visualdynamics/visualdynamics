"""The notch a set of limit channels would cut, read beside its plot.

The Notch Prediction toggle, offered on the bar when a transmissibility
from a control channel is selected with the specification written at
that channel (and, if they are held apart from it, the limit profiles),
draws the control spectrum with every predicted notch hatched and each
response channel against its limit profile (`plot.limit_prediction`).
This panel holds the one choice the reading takes, which tolerance pair
a notch is read against, and, under it, what came of it: the verdict, a
row a notch (its band, its depth in dB with the ratio beside it, and
whether it leaves the tolerance), and a line a notch and a line a
channel in words, with the control left at each notch's deepest line.

Every choice redraws at once. Nothing is made: the prediction is of
objects that exist, so there is no Apply button. The same call a script
makes, `core.limit_prediction.predict_limits`, gives every number shown
here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from .settings_panel import add_keyed_settings, add_rule, panel_grid

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.limit_prediction import LimitPrediction
    from ..units import UnitSystem

#: the tolerance pairs, as the box names them, in
#: `core.limit_prediction.TOLERANCES`'s order
TOLERANCE_NAMES = ('Abort', 'Warning')

#: the table's columns: the key of `LimitPrediction.table` each shows
COLUMNS = (('band', 'Band [Hz]'), ('depth', 'Depth [dB]'),
           ('tolerance', 'Tolerance'))


def defaults() -> dict[str, Any]:
    """The settings a fresh panel holds: `predict_limits`'s own."""
    return {'tolerance': 'abort'}


class NotchPanel(QWidget):
    """The tolerance choice, the verdict, the notches in a table and the
    reading in words. A choice is emitted as `settings()`."""

    #: a setting moved; here are all of them now
    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from ..core.limit_prediction import TOLERANCES

        self.title: QLabel = QLabel('Notch Prediction')
        grid = panel_grid(self, self.title)
        self._loading = False
        self._tolerances = TOLERANCES
        self._settings: dict[str, Any] = defaults()
        self.tolerance_box: QComboBox = QComboBox()
        self.tolerance_box.addItems(TOLERANCE_NAMES)
        self.tolerance_box.setToolTip(
            'The tolerance a notch is read against: where the control '
            'left by a notch falls under the lower edge of this pair, the '
            'notch takes the control outside its tolerance')
        self.names, self.editors = add_keyed_settings(
            grid, [('tolerance', 'Tolerance', self.tolerance_box)])
        add_rule(grid, 2)
        self.verdict: QLabel = QLabel('')
        self.verdict.setWordWrap(True)
        grid.addWidget(self.verdict, 3, 0, 1, 2)
        self.table: QTableWidget = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([name for _k, name in COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(90)
        self.table.setToolTip('One row a predicted notch; the whole notch '
                              'in words on each row')
        grid.addWidget(self.table, 4, 0, 1, 2)
        self.readout: QLabel = QLabel('')
        self.readout.setWordWrap(True)
        grid.addWidget(self.readout, 5, 0, 1, 2)
        grid.setRowStretch(6, 1)
        self.tolerance_box.currentIndexChanged.connect(self._chosen)

    # ---- the settings ----------------------------------------------------

    def settings(self) -> dict[str, Any]:
        """The settings: 'tolerance', the pair a notch is read against."""
        return dict(self._settings)

    def set_settings(self, settings: dict[str, Any] | None) -> None:
        """Show these settings (missing ones at their defaults) without
        reporting a choice."""
        self._settings = {**defaults(), **(settings or {})}
        self._loading = True
        try:
            self.tolerance_box.setCurrentIndex(
                self._tolerances.index(self._settings['tolerance']))
        finally:
            self._loading = False

    def _chosen(self, index: int) -> None:
        if self._loading:
            return
        self._settings['tolerance'] = self._tolerances[index]
        self.changed.emit(self.settings())

    # ---- the reading -----------------------------------------------------

    def show_prediction(self, prediction: LimitPrediction | None,
                        refusal: str | None = None, *,
                        unit_system: UnitSystem | None = None) -> None:
        """Write the verdict, a row a notch and the reading in words, in
        `unit_system`'s units; the refusal in the verdict's place when
        the prediction could not be made; all empty for None."""
        self.table.setRowCount(0)
        if prediction is None:
            self.verdict.setText(refusal or '')
            self.verdict.setStyleSheet('font-weight: 600;' if refusal
                                       else '')
            self.readout.setText('')
            return
        lines = prediction.describe(unit_system)
        self.verdict.setText(lines[0][0].upper() + lines[0][1:] + '.')
        self.verdict.setStyleSheet(
            'font-weight: 600;' if prediction.within_tolerance is False
            else '')
        rows = prediction.table(unit_system)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, (key, _name) in enumerate(COLUMNS):
                item = QTableWidgetItem(row[key])
                item.setToolTip(row['describe'][0].upper()
                                + row['describe'][1:] + '.')
                self.table.setItem(r, c, item)
        self.readout.setText('\n'.join(line[0].upper() + line[1:] + '.'
                                       for line in lines[1:]))
