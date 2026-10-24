"""The matrix comparison's tolerances and its table, beside the grid it
draws.

The Matrix reading of a specification and a cross-spectral matrix
selected together draws the measured matrix against the specified one
as a grid (`plot.matrix_comparison`). This panel holds what the
cross terms are judged under (the coherence tolerance, the phase
tolerance and the specified coherence below which a phase is not
judged) and, under them, what came of it: the verdict, then every cell
of the grid in words, the levels first and then each pair's coherence
and phase.

Every edit redraws the grid at once. Nothing is made: the comparison is
a reading of two objects that already exist, as the signature
comparison is, so there is no Apply button. The same call a script
makes, `core.matrix_comparison.compare_matrix`, gives every number
shown here.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QWidget

from ..core.matrix_comparison import (
    COHERENCE_FLOOR,
    COHERENCE_TOLERANCE,
    PHASE_TOLERANCE,
)
from .editors import DoubleSpinBox, number_box
from .settings_panel import add_derived, add_keyed_settings, add_rule, panel_grid


class MatrixComparisonPanel(QWidget):
    """The settings and the table. Edits are emitted as `settings()`."""

    #: a setting moved; here is the whole set now
    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Matrix')
        grid = panel_grid(self, self.title)
        self._loading = False

        self.coherence_box: DoubleSpinBox = number_box(
            0.01, 1.0, value=COHERENCE_TOLERANCE, decimals=2, step=0.01,
            tip='How far each cross term\'s measured coherence may be from '
                'the specified one, either way')
        self.phase_box: DoubleSpinBox = number_box(
            0.5, 180.0, value=PHASE_TOLERANCE, decimals=1, step=1.0,
            tip='How far each cross term\'s measured phase may be from the '
                'specified one, in degrees either way')
        self.floor_box: DoubleSpinBox = number_box(
            0.0, 0.99, value=COHERENCE_FLOOR, decimals=2, step=0.05,
            tip='The specified coherence a phase is judged from: below it '
                'the channels are asked to be nearly independent, and the '
                'phase between them is not held')
        self.names, self.editors = add_keyed_settings(grid, [
            ('coherence', 'Coherence [±]', self.coherence_box),
            ('phase', 'Phase [±°]', self.phase_box),
            ('floor', 'Phase from γ²', self.floor_box),
        ])
        add_rule(grid, 4)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value) in add_derived(
                grid, (('verdict', 'Verdict'),), 5).items()}
        # one line per cell of the grid, as many as there are channels
        # and pairs, so a label that wraps rather than a fixed set of rows
        self.readout: QLabel = QLabel('')
        self.readout.setWordWrap(True)
        grid.addWidget(self.readout, 6, 0, 1, 2)
        grid.setRowStretch(7, 1)

        for box in (self.coherence_box, self.phase_box, self.floor_box):
            box.valueChanged.connect(self._edited)

    def _edited(self, *_args: Any) -> None:
        if not self._loading:
            self.changed.emit(self.settings())

    def settings(self) -> dict[str, float]:
        """The comparison's arguments: 'coherence_tolerance',
        'phase_tolerance' and 'coherence_floor'."""
        return {'coherence_tolerance': self.coherence_box.value(),
                'phase_tolerance': self.phase_box.value(),
                'coherence_floor': self.floor_box.value()}

    def set_settings(self, coherence_tolerance: float | None = None,
                     phase_tolerance: float | None = None,
                     coherence_floor: float | None = None) -> None:
        """Set the fields without a redraw per field, then emit once."""
        self._loading = True
        try:
            if coherence_tolerance is not None:
                self.coherence_box.setValue(coherence_tolerance)
            if phase_tolerance is not None:
                self.phase_box.setValue(phase_tolerance)
            if coherence_floor is not None:
                self.floor_box.setValue(coherence_floor)
        finally:
            self._loading = False
        self._edited()

    def show_result(self, comparison: Any) -> None:
        """Write the verdict and every cell's words, one per line: the
        levels, then each pair's coherence and phase, then the notes.
        None clears the readout."""
        if comparison is None:
            self.derived['verdict'].setText('—')
            self.readout.setText('')
            return
        verdict = {True: 'passes', False: 'fails',
                   None: 'nothing judged'}[comparison.passed]
        self.derived['verdict'].setText(verdict)
        lines = [f'{row["label"]} {row["words"]}'
                 for row in comparison.table()]
        if comparison.scale_db:
            lines.append(f'measured levels scaled {comparison.scale_db:+g} dB')
        lines.extend(comparison.notes)
        for pair in comparison.pairs:
            lines.extend(f'{pair.label}: {note}' for note in pair.notes)
        self.readout.setText('\n'.join(lines))
