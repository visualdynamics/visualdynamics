"""The settings of a drive prediction, and what it came to, beside it.

The Drive Prediction toggle offered on the bar when an FRF and a
specification are selected together draws the drives the specification
asks for through the FRF, one channel's predicted response and every
quantity against its rating (`plot.drive_prediction`). This panel holds
what the prediction is made with (the inverse and its regularization,
the run's length for the expected peaks, the peak each drive may reach
and the shaker behind every drive, its ratings the ones the Stroke
reading takes) and, under them, the verdict: which quantity runs out
first, at what share of its rating and where, and each quantity in a
line.

Every edit redraws at once, so a rating is seen against the drives
while it is typed. Nothing is made: the prediction is of two objects
that exist, so there is no Apply button. A rating or a limit left at
zero is not given, and is reported without a verdict. The same call a
script makes, `core.drive_prediction.predict_drives`, gives every
number shown here.

The settings are kept in SI and shown in the window's units, so a
change of units restates them rather than reinterpreting them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QWidget

from .editors import DoubleSpinBox, commit_on_enter
from .settings_panel import add_keyed_settings, add_rule, panel_grid
from .stroke_panel import RATINGS as SHAKER_RATINGS

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.drive_prediction import DrivePrediction
    from ..units import UnitSystem

#: the inverses, as the box names them, in `core.drive_prediction`'s
#: order
METHOD_NAMES = ('Truncated SVD', 'Tikhonov')


def defaults() -> dict[str, Any]:
    """The settings a fresh panel holds: `predict_drives`'s own
    defaults, and no ratings."""
    from ..core.drive_prediction import DURATION, TIKHONOV, TRUNCATION

    return {'method': 'truncated', 'tolerance': TRUNCATION,
            'weight': TIKHONOV, 'duration': DURATION, 'drive_limit': None,
            **{key: None for key, *_rest in SHAKER_RATINGS}}


def prediction_arguments(settings: dict[str, Any]) -> dict[str, Any]:
    """The panel's settings as `predict_drives` takes them: the shaker
    ratings gathered into one `ShakerRating` for every drive, or none
    when no rating is given."""
    from ..core.shaker import ShakerRating

    ratings = {key: settings.get(key) for key, *_rest in SHAKER_RATINGS
               if settings.get(key) is not None}
    return {'method': settings['method'],
            'tolerance': settings['tolerance'],
            'weight': settings['weight'],
            'duration': settings['duration'],
            'drive_limit': settings.get('drive_limit'),
            'shakers': ShakerRating(**ratings) if ratings else None}


class DrivePredictionPanel(QWidget):
    """The settings and the verdict. Edits are emitted as
    `settings()`."""

    #: a setting moved; here are all of them now, in SI
    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from ..units import DEFAULT_SYSTEM

        self.title: QLabel = QLabel('Drive Prediction')
        grid = panel_grid(self, self.title)
        self._loading = False
        self.unit_system: UnitSystem = DEFAULT_SYSTEM
        #: the drives' quantity, which the drive limit is shown in
        self.drive_quantity: str = 'voltage'
        self._settings: dict[str, Any] = defaults()

        self.method_box: QComboBox = QComboBox()
        self.method_box.addItems(METHOD_NAMES)
        self.method_box.setToolTip(
            'How the FRF matrix is inverted at each line: the truncated '
            'singular value decomposition drops the directions weaker '
            'than the tolerance times the line\'s strongest; Tikhonov '
            'holds the inverse down where the plant barely responds')
        self.tolerance_box = self._box(
            0.0001, 0.5, 4, 'The share of each line\'s largest singular '
            'value under which a direction is dropped: 0.01 drops what '
            'a condition number over 100 makes')
        self.weight_box = self._box(
            1e-6, 1.0, 6, 'The Tikhonov weight, as a share of the largest '
            'squared singular value in the band')
        self.duration_box = self._box(
            1.0, 1e6, 1, 'How long the run lasts, for the expected '
            'largest peak of every quantity')
        self.limit_box = self._box(
            0.0, 1e9, 4, 'The peak each drive may reach: the volts the '
            'amplifier input accepts. Zero is no limit')
        self.rating_boxes: dict[str, DoubleSpinBox] = {}
        rows: list[tuple[str, str, QWidget]] = [
            ('method', 'Inverse', self.method_box),
            ('tolerance', 'Tolerance', self.tolerance_box),
            ('weight', 'Weight', self.weight_box),
            ('duration', 'Duration', self.duration_box),
            ('drive_limit', 'Drive limit', self.limit_box)]
        for key, name, _dimension, tip in SHAKER_RATINGS:
            box = self._box(0.0, 1e9, 4, tip.replace('shock ', ''))
            self.rating_boxes[key] = box
            rows.append((key, name, box))
        self.names, self.editors = add_keyed_settings(grid, rows)
        add_rule(grid, len(rows) + 1)
        # the verdict, then a line a quantity, wrapped to the panel
        self.verdict: QLabel = QLabel('')
        self.verdict.setWordWrap(True)
        grid.addWidget(self.verdict, len(rows) + 2, 0, 1, 2)
        self.readout: QLabel = QLabel('')
        self.readout.setWordWrap(True)
        grid.addWidget(self.readout, len(rows) + 3, 0, 1, 2)
        grid.setRowStretch(len(rows) + 4, 1)

        commit_on_enter(self.tolerance_box, self.weight_box,
                        self.duration_box, self.limit_box,
                        *self.rating_boxes.values())
        self.method_box.currentIndexChanged.connect(self._method_chosen)
        for key in ('tolerance', 'weight', 'duration', 'drive_limit',
                    *self.rating_boxes):
            self.editors[key].valueChanged.connect(
                lambda _value, key=key: self._edited(key))
        self.set_unit_system(DEFAULT_SYSTEM)

    @staticmethod
    def _box(low: float, high: float, decimals: int,
             tip: str) -> DoubleSpinBox:
        box = DoubleSpinBox()
        box.setRange(low, high)
        box.setDecimals(decimals)
        box.setToolTip(tip)
        return box

    # ---- the settings ----------------------------------------------------

    def settings(self) -> dict[str, Any]:
        """The settings in SI: 'method', 'tolerance', 'weight',
        'duration', 'drive_limit' and the shaker ratings, each rating
        and the limit None where not given (`prediction_arguments`
        turns them into `predict_drives`'s arguments)."""
        return dict(self._settings)

    def set_settings(self, settings: dict[str, Any] | None) -> None:
        """Show these settings (SI; missing ones at their defaults)
        without reporting an edit."""
        self._settings = {**defaults(), **(settings or {})}
        self._restate()

    def set_unit_system(self, unit_system: UnitSystem) -> None:
        """Show the settings in `unit_system`'s units."""
        self.unit_system = unit_system
        self._relabel()
        self._restate()

    def set_drive_quantity(self, quantity: str) -> None:
        """Show the drive limit in the drives' quantity: volts, or the
        force unit for drives measured in force."""
        if quantity != self.drive_quantity:
            self.drive_quantity = quantity
            self._relabel()
            self._restate()

    def _relabel(self) -> None:
        us = self.unit_system
        self.names['duration'].setText(f'Duration [{us.label_text("time")}]')
        self.names['drive_limit'].setText(
            f'Drive limit [{us.label_text(self.drive_quantity)}]')
        for key, name, dimension, _tip in SHAKER_RATINGS:
            self.names[key].setText(f'{name} [{us.label_text(dimension)}]')

    def _dimension(self, key: str) -> str | None:
        """The dimension a setting is shown in; None for a pure number."""
        if key == 'drive_limit':
            return self.drive_quantity
        if key == 'duration':
            return 'time'
        return next((dim for name, _n, dim, _t in SHAKER_RATINGS
                     if name == key), None)

    def _restate(self) -> None:
        """Write the SI settings into the boxes in the units shown, and
        show the one regularization setting the inverse uses."""
        self._loading = True
        try:
            method = self._settings['method']
            self.method_box.setCurrentIndex(
                0 if method == 'truncated' else 1)
            for key in ('tolerance', 'weight', 'duration', 'drive_limit',
                        *self.rating_boxes):
                value = self._settings.get(key)
                dimension = self._dimension(key)
                if value is not None and dimension is not None:
                    value = float(self.unit_system.from_si(value, dimension))
                self.editors[key].setValue(0.0 if value is None else value)
            for key, wanted in (('tolerance', method == 'truncated'),
                                ('weight', method == 'tikhonov')):
                self.names[key].setVisible(wanted)
                self.editors[key].setVisible(wanted)
        finally:
            self._loading = False

    def _method_chosen(self, index: int) -> None:
        if self._loading:
            return
        self._settings['method'] = 'truncated' if index == 0 else 'tikhonov'
        self._restate()
        self.changed.emit(self.settings())

    def _edited(self, key: str) -> None:
        """One box was edited: take its value into SI and report all
        the settings. A rating or limit at zero is none."""
        if self._loading:
            return
        value = float(self.editors[key].value())
        dimension = self._dimension(key)
        optional = key == 'drive_limit' or key in self.rating_boxes
        if optional and value <= 0.0:
            self._settings[key] = None
        else:
            self._settings[key] = (value if dimension is None else
                                   float(self.unit_system.to_si(value,
                                                                dimension)))
        self.changed.emit(self.settings())

    # ---- the reading -----------------------------------------------------

    def show_prediction(self, prediction: DrivePrediction | None,
                        refusal: str | None = None) -> None:
        """Write the verdict and a line a quantity, in the units shown;
        the refusal in its place when the prediction could not be made;
        both empty for None."""
        if prediction is None:
            self.verdict.setText(refusal or '')
            self.verdict.setStyleSheet('font-weight: 600;' if refusal
                                       else '')
            self.readout.setText('')
            return
        self.set_drive_quantity(prediction.drive_quantity)
        lines = prediction.describe(self.unit_system)
        self.verdict.setText(lines[0][0].upper() + lines[0][1:] + '.')
        self.verdict.setStyleSheet('font-weight: 600;'
                                   if prediction.fits is False else '')
        self.readout.setText('\n'.join(line[0].upper() + line[1:] + '.'
                                       for line in lines[1:]))
