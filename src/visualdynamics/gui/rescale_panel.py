"""A time history's records rescaled, beside the plot that previews it.

The Rescale reading draws the records picked (all of them when none
are) as recorded and as they would be rescaled, one over the other on
the same axes. This panel holds how the factor is given, above the rule:
typed as a factor, or worked out from the sensitivity the data was
recorded with and the correct one (`core.rescale.sensitivity_factor`).
Under the rule are the factor that applies, the records it applies to
and the largest value drawn before and after.

Every edit redraws at once. Apply Rescale makes the record
(`Project.rescale`, the call a script makes) beside the source and
linked to it, the records picked multiplied by the factor shown.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QPushButton, QWidget

from .editors import DoubleSpinBox, number_box
from .settings_panel import add_derived, add_keyed_settings, add_rule, panel_grid

#: how the factor is given, as (key, name in the box)
WAYS = (('factor', 'Factor'), ('sensitivity', 'Sensitivities'))

#: the read-only rows under the rule, as (key, name)
ROWS = (('factor', 'Factor applied'), ('records', 'Records'),
        ('before', 'Largest, recorded'), ('after', 'Largest, rescaled'))

#: what the panel opens with: nothing changed until something is typed
DEFAULTS = {'way': 'factor', 'factor': 1.0, 'recorded': 100.0,
            'correct': 100.0}


class RescalePanel(QWidget):
    """The settings and the readout. Edits are emitted as `settings()`."""

    #: a setting moved; here is the whole set now
    changed = Signal(dict)
    #: Apply Rescale was pressed: make the record at these settings
    apply_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Rescale')
        grid = panel_grid(self, self.title)
        self._loading = False

        self.way_box: QComboBox = QComboBox()
        for key, text in WAYS:
            self.way_box.addItem(text, key)
        self.way_box.setToolTip(
            'Type the factor, or the sensitivity the data was recorded '
            'with and the correct one')
        self.factor_box: DoubleSpinBox = number_box(
            -1e9, 1e9, value=DEFAULTS['factor'], decimals=6, step=0.1,
            tip='What the picked records are multiplied by; negative '
                'flips their sign')
        self.recorded_box: DoubleSpinBox = number_box(
            1e-9, 1e9, value=DEFAULTS['recorded'], decimals=6,
            tip='The sensitivity the acquisition was given, in mV per '
                'engineering unit', suffix=' mV/unit')
        self.correct_box: DoubleSpinBox = number_box(
            1e-9, 1e9, value=DEFAULTS['correct'], decimals=6,
            tip='The sensor\'s correct sensitivity, in the same units; '
                'the factor is recorded over correct', suffix=' mV/unit')
        self.names, self.editors = add_keyed_settings(grid, [
            ('way', 'Given as', self.way_box),
            ('factor', 'Factor', self.factor_box),
            ('recorded', 'Recorded with', self.recorded_box),
            ('correct', 'Correct', self.correct_box),
        ])
        add_rule(grid, 5)
        made = add_derived(grid, ROWS, 6)
        self.values: dict[str, QLabel] = {
            key: value for key, (_name, value) in made.items()}
        self.apply_button: QPushButton = QPushButton('Apply Rescale')
        self.apply_button.setToolTip(
            'Make the record with the picked records rescaled, beside '
            'this one')
        self.apply_button.clicked.connect(self.apply_asked.emit)
        grid.addWidget(self.apply_button, 6 + len(ROWS), 0, 1, 2)
        grid.setRowStretch(7 + len(ROWS), 1)

        self.way_box.currentIndexChanged.connect(self._way_edited)
        for box in (self.factor_box, self.recorded_box, self.correct_box):
            box.valueChanged.connect(self._edited)
        self._show_way()

    def _show_way(self) -> None:
        """Only the rows the chosen way reads."""
        by_factor = self.way_box.currentData() == 'factor'
        for key, shown in (('factor', by_factor), ('recorded', not by_factor),
                           ('correct', not by_factor)):
            self.names[key].setVisible(shown)
            self.editors[key].setVisible(shown)

    def _way_edited(self, *_args: Any) -> None:
        self._show_way()
        self._edited()

    def _edited(self, *_args: Any) -> None:
        if not self._loading:
            self.changed.emit(self.settings())

    def settings(self) -> dict[str, Any]:
        """The settings: 'way' ('factor' or 'sensitivity'), 'factor', and
        'recorded' and 'correct' in mV per engineering unit."""
        return {'way': self.way_box.currentData(),
                'factor': float(self.factor_box.value()),
                'recorded': float(self.recorded_box.value()),
                'correct': float(self.correct_box.value())}

    def set_settings(self, settings: dict[str, Any] | None) -> None:
        """Set the fields from a `settings()` dict without a redraw per
        field; emits nothing, since the caller is about to draw."""
        if not settings:
            return
        self._loading = True
        try:
            keys = [key for key, _text in WAYS]
            way = settings.get('way', DEFAULTS['way'])
            self.way_box.setCurrentIndex(keys.index(way) if way in keys else 0)
            for key, box in (('factor', self.factor_box),
                             ('recorded', self.recorded_box),
                             ('correct', self.correct_box)):
                box.setValue(float(settings.get(key, DEFAULTS[key])))
            self._show_way()
        finally:
            self._loading = False

    def show_reading(self, reading: dict[str, str] | None) -> None:
        """Fill the readout from `reading` (key to text, `ROWS`' keys);
        None clears it."""
        for value in self.values.values():
            value.setText('—')
        for key, text in (reading or {}).items():
            self.values[key].setText(text)
