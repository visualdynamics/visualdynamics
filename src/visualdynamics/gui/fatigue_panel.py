"""The extreme response and fatigue damage spectra, beside the plot that
draws them.

The Fatigue reading of an acceleration record (one record of a PSD or a
specification, or one channel of a time history) runs it through a
resonator at each natural frequency and reads two spectra off them
(`core.fatigue`): the largest response each is expected to reach, and
the fatigue damage each response does. This panel holds the settings
the two rest on and says what was found.

The settings: the candidate test the record is set against (none, or
any other acceleration record in the project); the fatigue exponent b,
the resonators' Q and, for a spectrum, how its cycles are read; and the
seconds each environment lasts, which a spectrum does not carry and a
time history does (its own length, which a duration given here scales
the damage to). Under the rule, what each result rests on and, with a
test, the least the test's damage stands above the field's and where,
the factor on its ASD or the duration at its level that covers the
field at every natural frequency, the exaggeration it carries against
the cap, and its extreme response against the field's. Nothing is made,
so there is no Apply button: the reading is of records that already
exist.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QWidget

from ..core.fatigue import B_RANGE, METHODS, Q_RANGE, _seconds
from ..core.srs import DEFAULT_Q
from .editors import DoubleSpinBox, commit_on_enter
from .settings_panel import add_derived, add_keyed_settings, add_rule, panel_grid

#: the exponent the panel starts at: the one test documents give for
#: random vibration when they give one; the reading always says which
#: was used, and the governing document's goes here
B_START = 7.5

#: the readings under the rule, in order: key and row name
ROWS = (('field', 'Field'), ('test', 'Test'), ('band', 'Natural frequencies'),
        ('least', 'Least damage margin'), ('factor', 'Factor that covers'),
        ('duration', 'Duration that covers'),
        ('exaggeration', 'Exaggeration'), ('ers', 'Least ERS margin'))


class FatiguePanel(QWidget):
    """The candidate test, b, Q, the reading and the durations, then the
    two spectra and their comparison. Edits are emitted as the dict
    `settings` returns."""

    #: a setting was changed; the settings now
    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Fatigue')
        grid = panel_grid(self, self.title)
        self._loading = False

        self.against_box: QComboBox = QComboBox()
        self.against_box.setToolTip(
            'The candidate test the record is set against: another '
            'acceleration record in the project, read on the same '
            'natural frequencies, Q and b')
        self.b_box: DoubleSpinBox = DoubleSpinBox()
        self.b_box.setDecimals(2)
        self.b_box.setRange(*B_RANGE)
        self.b_box.setSingleStep(0.5)
        self.b_box.setValue(B_START)
        self.b_box.setToolTip(
            "The fatigue exponent, the governing document's convention: "
            'commonly 7.5 for random (5 to 8 in use) and 6 for sine. The '
            'damage goes as the response to this power')
        self.q_box: DoubleSpinBox = DoubleSpinBox()
        self.q_box.setDecimals(1)
        self.q_box.setRange(*Q_RANGE)
        self.q_box.setValue(DEFAULT_Q)
        self.q_box.setToolTip('The resonators\' amplification, 1 / (2 zeta)')
        self.method_box: QComboBox = QComboBox()
        for method in METHODS:
            self.method_box.addItem(method, method)
        self.method_box.setToolTip(
            'How a spectrum\'s cycles are read: narrowband (Rayleigh '
            "amplitudes at the crossing rate), Rice's peaks, or Dirlik's "
            'fit to rainflow ranges. A time history is counted by rainflow')
        self.duration_box: DoubleSpinBox = DoubleSpinBox()
        self.test_duration_box: DoubleSpinBox = DoubleSpinBox()
        for box, whose in ((self.duration_box, 'the record'),
                           (self.test_duration_box, 'the test')):
            box.setDecimals(1)
            box.setRange(0.0, 1e9)
            box.setSuffix(' s')
            box.setSpecialValueText('not given')
            box.setToolTip(
                f'The seconds {whose} lasts. A spectrum does not carry '
                'one; a time history is its own length, and a duration '
                'here scales its damage to it')
            commit_on_enter(box)
        for box in (self.b_box, self.q_box):
            commit_on_enter(box)
        self.names, self.editors = add_keyed_settings(grid, [
            ('against', 'Against', self.against_box),
            ('b', 'Exponent b', self.b_box),
            ('q', 'Q', self.q_box),
            ('method', 'Reading', self.method_box),
            ('duration', 'Duration', self.duration_box),
            ('test_duration', 'Test duration', self.test_duration_box)])
        add_rule(grid, 7)
        self.derived: dict[str, tuple[QLabel, QLabel]] = add_derived(
            grid, ROWS, 8)
        self.rests: QLabel = QLabel('')
        self.rests.setWordWrap(True)
        grid.addWidget(self.rests, 8 + len(ROWS), 0, 1, 2)
        self.verdict: QLabel = QLabel('')
        self.verdict.setWordWrap(True)
        font = self.verdict.font()
        font.setBold(True)
        self.verdict.setFont(font)
        grid.addWidget(self.verdict, 9 + len(ROWS), 0, 1, 2)
        self.notes: QLabel = QLabel('')
        self.notes.setWordWrap(True)
        grid.addWidget(self.notes, 10 + len(ROWS), 0, 1, 2)
        grid.setRowStretch(11 + len(ROWS), 1)
        self.against_box.currentIndexChanged.connect(self._edited)
        self.method_box.currentIndexChanged.connect(self._edited)
        for box in (self.b_box, self.q_box, self.duration_box,
                    self.test_duration_box):
            box.valueChanged.connect(self._edited)

    def _edited(self, *_args: Any) -> None:
        if not self._loading:
            self.changed.emit(self.settings())

    def settings(self) -> dict[str, Any]:
        """{'against': (name, record) or None, 'b', 'q', 'method',
        'duration' and 'test_duration' (seconds, None when not
        given)}."""
        def seconds(box: DoubleSpinBox) -> float | None:
            value = float(box.value())
            return value if value > 0.0 else None

        against = self.against_box.currentData()
        return {'against': None if against is None else tuple(against),
                'b': float(self.b_box.value()),
                'q': float(self.q_box.value()),
                'method': str(self.method_box.currentData()),
                'duration': seconds(self.duration_box),
                'test_duration': seconds(self.test_duration_box)}

    def set_settings(self, settings: dict[str, Any],
                     choices: Sequence[tuple[str, tuple[str, int]]], *,
                     spectrum: bool, test_spectrum: bool) -> None:
        """Show `settings` without emitting: `choices` offered as the
        candidate tests by (words, (name, record)), the reading's row
        shown for a spectrum (`spectrum`) and the test's duration for a
        spectrum test (`test_spectrum`)."""
        self._loading = True
        try:
            self.against_box.clear()
            self.against_box.addItem('none', None)
            for words, key in choices:
                self.against_box.addItem(words, list(key))
            against = settings.get('against')
            index = (self.against_box.findData(list(against))
                     if against is not None else 0)
            self.against_box.setCurrentIndex(max(index, 0))
            self.b_box.setValue(float(settings.get('b', B_START)))
            self.q_box.setValue(float(settings.get('q', DEFAULT_Q)))
            index = self.method_box.findData(settings.get('method',
                                                          METHODS[0]))
            self.method_box.setCurrentIndex(max(index, 0))
            self.duration_box.setValue(float(settings.get('duration') or 0.0))
            self.test_duration_box.setValue(
                float(settings.get('test_duration') or 0.0))
        finally:
            self._loading = False
        tested = self.against_box.currentData() is not None
        for key, shown in (('method', spectrum or (tested and test_spectrum)),
                           ('test_duration', tested and test_spectrum)):
            self.names[key].setVisible(shown)
            self.editors[key].setVisible(shown)
        self.names['duration'].setText('Duration' if spectrum
                                       else 'Scale to')

    def show_reading(self, field: Any, pair: Any = None, *,
                     refusal: str | None = None) -> None:
        """Write `field`, a `core.fatigue.ResponseSpectra`, and `pair`,
        its `Equivalence` with a test, into the rows; None clears them,
        with `refusal` said in its place when there is one."""
        def db(value: float) -> str | None:
            return f'{value:+.2f} dB' if math.isfinite(value) else None

        values: dict[str, str | None] = dict.fromkeys(key for key, _ in ROWS)
        if field is not None:
            f = field.frequency
            values['field'] = (f'{field.source}, {_seconds(field.duration)}')
            values['band'] = (f'{f.size}, {f[0]:.4g} to {f[-1]:.4g} Hz'
                              if f.size > 1 else f'{f[0]:.4g} Hz')
        if pair is not None:
            test = pair.test
            values['test'] = f'{test.source}, {_seconds(test.duration)}'
            values['least'] = (f'{10 * math.log10(pair.least_ratio):+.2f} dB '
                               f'at {pair.governs:.4g} Hz')
            values['factor'] = f'{pair.factor_needed:.4g} on the ASD'
            values['duration'] = _seconds(pair.duration_needed)
            exaggeration = np.asarray(pair.exaggeration, dtype=float)
            exaggeration = exaggeration[np.isfinite(exaggeration)]
            if exaggeration.size:
                values['exaggeration'] = (
                    f'{exaggeration.min():.3g} to {exaggeration.max():.3g}, '
                    f'cap {pair.cap:g}')
            ers = np.asarray(pair.ers_db, dtype=float)
            ers = ers[np.isfinite(ers)]
            if ers.size:
                values['ers'] = db(float(ers.min()))
        for key, value in values.items():
            self.derived[key][1].setText('—' if value is None else value)
        if field is None:
            self.rests.setText('')
            self.verdict.setText('' if refusal is None
                                 else refusal[0].upper() + refusal[1:] + '.')
            self.notes.setText('')
            return
        rests = pair.rests_on if pair is not None else field.rests_on
        self.rests.setText('Rests on: ' + rests + '.')
        said = pair.describe() if pair is not None else field.describe()
        self.verdict.setText(said[0].upper() + said[1:] + '.')
        notes = list(field.notes) + ([] if pair is None else
                                     [f'test: {note}' for note in
                                      pair.test.notes])
        self.notes.setText('\n'.join(note[0].upper() + note[1:] + '.'
                                     for note in notes))
