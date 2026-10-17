"""The shaker's shock ratings, beside a pulse read against them.

The Stroke reading of a time history draws the record's acceleration,
its velocity and its displacement, integrated raw, on one clock
(`plot.stroke`). This panel holds what the pulse is judged against (the
shaker's shock stroke, velocity and force, the moving mass, and the
body's mass) and, under them, what the reading came to: the peak
acceleration and velocity, the table's peak to peak beside the
half-sine rule, the travel with the body counted, peak to peak and each
way from center, and the force, then each demand against its rating in
words.

Every edit redraws the plot at once, so a rating is seen against the
pulse while it is typed. Nothing is made: the reading is of a record
that already exists, as the kurtosis bars are, so there is no Apply
button to press. A rating left at zero is not given, and is reported
without a verdict. The same call a script makes,
`core.stroke.read_stroke`, gives every number shown here.

The ratings are kept in SI and shown in the window's units, so a
change of units restates them rather than reinterpreting them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QWidget

from .editors import DoubleSpinBox, commit_on_enter
from .settings_panel import add_derived, add_keyed_settings, add_rule, panel_grid

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.stroke import StrokeReading
    from ..units import UnitSystem

#: the ratings, as (key, name, the dimension each is shown in, tooltip),
#: in the order the panel lists them; the keys are `read_stroke`'s
RATINGS = (
    ('stroke', 'Stroke', 'length',
     ('The shaker\'s shock stroke, peak to peak: the armature\'s travel '
      'relative to the body. Zero is no rating')),
    ('velocity', 'Velocity', 'velocity',
     'The shaker\'s shock velocity. Zero is no rating'),
    ('force', 'Force', 'force',
     ('The shaker\'s shock force, judged against the moving mass times '
      'the peak. Zero is no rating')),
    ('moving_mass', 'Moving mass', 'mass',
     ('Everything that moves: armature, table or expander, fixture and '
      'article. Zero leaves the force and the body out')),
    ('body_mass', 'Body mass', 'mass',
     ('The shaker body\'s mass. Given with the moving mass, the body '
      'moves the other way by the moving mass over this mass, as a share '
      'of the table\'s motion, so the travel the stops see and the '
      'velocity the coil moves at are the table\'s times one plus that '
      'share. Zero counts the table alone')),
)

#: the readings under the rule, as (key, name)
READINGS = (('peak', 'Peak'), ('peak_velocity', 'Peak velocity'),
            ('peak_to_peak', 'Peak to peak'), ('rule', 'Half-sine rule'),
            ('travel', 'Travel p-p'), ('each_way', 'Down / up'),
            ('force', 'Force'))


class StrokePanel(QWidget):
    """The ratings and the readout. Edits are emitted as `settings()`."""

    #: a rating moved; here are all of them now, in SI
    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from ..units import DEFAULT_SYSTEM

        self.title: QLabel = QLabel('Stroke')
        grid = panel_grid(self, self.title)
        self._loading = False
        #: the units the boxes are shown in
        self.unit_system: UnitSystem = DEFAULT_SYSTEM
        #: the ratings in SI, None where not given: what `settings`
        #: returns, kept here so a change of units restates them
        self._ratings: dict[str, float | None] = {
            key: None for key, *_rest in RATINGS}

        rows = []
        self.boxes: dict[str, DoubleSpinBox] = {}
        for key, name, _dimension, tip in RATINGS:
            box = DoubleSpinBox()
            box.setRange(0.0, 1e9)
            box.setDecimals(4)
            box.setToolTip(tip)
            self.boxes[key] = box
            rows.append((key, name, box))
        self.names, _editors = add_keyed_settings(grid, rows)
        add_rule(grid, len(rows) + 1)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value) in add_derived(
                grid, READINGS, len(rows) + 2).items()}
        # the verdicts in words, a line a rating, wrapped to the panel
        self.readout: QLabel = QLabel('')
        self.readout.setWordWrap(True)
        grid.addWidget(self.readout, len(rows) + 2 + len(READINGS), 0, 1, 2)
        grid.setRowStretch(len(rows) + 3 + len(READINGS), 1)

        commit_on_enter(*self.boxes.values())
        for key, box in self.boxes.items():
            box.valueChanged.connect(
                lambda _value, key=key: self._edited(key))
        self.set_unit_system(DEFAULT_SYSTEM)

    # ---- the ratings -----------------------------------------------------

    def settings(self) -> dict[str, float | None]:
        """The ratings as `core.stroke.read_stroke` takes them, in SI:
        'stroke', 'velocity', 'force', 'moving_mass' and 'body_mass',
        each None where it is not given."""
        return dict(self._ratings)

    def set_settings(self, settings: dict[str, float | None] | None) -> None:
        """Show these ratings (SI, None or missing for none) without
        reporting an edit."""
        self._ratings = {key: (None if settings is None
                               else settings.get(key))
                         for key, *_rest in RATINGS}
        self._restate()

    def set_unit_system(self, unit_system: UnitSystem) -> None:
        """Show the ratings in `unit_system`'s units."""
        self.unit_system = unit_system
        for key, name, dimension, _tip in RATINGS:
            self.names[key].setText(
                f'{name} [{unit_system.label_text(dimension)}]')
        self._restate()

    def _restate(self) -> None:
        """Write the SI ratings into the boxes in the units shown."""
        self._loading = True
        try:
            for key, _name, dimension, _tip in RATINGS:
                value = self._ratings.get(key)
                if value is not None:
                    value = float(self.unit_system.from_si(value, dimension))
                self.boxes[key].setValue(0.0 if value is None else value)
        finally:
            self._loading = False

    def _edited(self, key: str) -> None:
        """One box was edited: take its value into SI and report all
        the ratings. Only that box is read back, so a rating the boxes
        show rounded to their decimals is not rounded by an edit to
        another one."""
        if self._loading:
            return
        dimension = next(dim for name, _n, dim, _t in RATINGS if name == key)
        value = self.boxes[key].value()
        if value <= 0.0:
            self._ratings[key] = None
        else:
            self._ratings[key] = float(self.unit_system.to_si(value, dimension))
        self.changed.emit(self.settings())

    # ---- the reading -----------------------------------------------------

    def show_reading(self, reading: StrokeReading | None) -> None:
        """Write what `reading` came to under the ratings, in the units
        shown; None clears it."""
        from ..core.stroke import _amount, _number

        if reading is None:
            for value in self.derived.values():
                value.setText('—')
            self.readout.setText('')
            self.readout.setStyleSheet('')
            return
        us = self.unit_system

        def amount(value, dimension):
            return _amount(value, dimension, us)

        self.derived['peak'].setText(
            amount(reading.peak_acceleration, 'acceleration'))
        self.derived['peak_velocity'].setText(
            amount(reading.peak_velocity, 'velocity'))
        self.derived['peak_to_peak'].setText(
            amount(reading.peak_to_peak, 'length'))
        self.derived['rule'].setText(amount(reading.rule, 'length'))
        self.derived['travel'].setText(amount(reading.travel, 'length'))
        length = us.label_text('length')
        self.derived['each_way'].setText(
            f'{_number(float(us.from_si(reading.travel_down, "length")))} / '
            f'{_number(float(us.from_si(reading.travel_up, "length")))} '
            f'{length}')
        self.derived['force'].setText(
            '—' if reading.force is None else amount(reading.force, 'force'))
        # the lines that judge something: each rating given, and the
        # travel to the stop when it reaches one; the numbers alone are
        # above
        lines = [line for line in reading.describe(us) if ' against ' in line]
        if not lines:
            lines = ['No ratings given: the reading stands on its own.']
        self.readout.setText('\n'.join(lines))
        failed = reading.fits is False
        self.readout.setStyleSheet('font-weight: 600;' if failed else '')
