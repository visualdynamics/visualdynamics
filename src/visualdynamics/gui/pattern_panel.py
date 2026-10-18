"""Pattern, as a pane beside the 3-D view.

Pattern copies the solids picked in the tree in rows, columns and
layers, so a part with many identical holes or inserts does not need
each one added by hand. The pane takes how many along the geometry's
X, Y and Z (the originals included) and how far apart, and the copies
are drawn in the view as the numbers are typed. Pattern then runs the
project's `pattern_solids`, which makes one solid from the originals,
so changing an original changes every copy. One hole repeated 13 times
along X makes a row; that pattern picked and repeated twice along Y
makes two rows. A pattern's pencil in the tree opens the pane on it
again to change its counts and spacing.

The pane holds no rules of its own: the window draws the copies and
says what they are.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from .editors import DoubleSpinBox, SpinBox, commit_on_enter
from .settings_panel import panel_grid

# Brandon, 2026-10-11. The Frame-Wing has 41 insert holes, 41 inserts
# and 18 wing holes: a hundred cylinders, typed one at a time, is a
# workflow nobody finishes. One hole and 13 along X at the insert pitch
# is a rail's row; that pattern picked and 2 along Y is both rails'.


class PatternPanel(QWidget):
    """How many along each axis, how far apart, and the Pattern button."""

    #: a count or a spacing changed: the window draws the copies again
    changed = Signal()
    pattern_asked = Signal()
    close_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Pattern')
        grid = panel_grid(self, self.title)
        self.parts_label: QLabel = QLabel()
        self.parts_label.setWordWrap(True)
        self.parts_label.setEnabled(False)
        grid.addWidget(self.parts_label, 1, 0, 1, 3)
        self.count_label: QLabel = QLabel('How many')
        self.spacing_label: QLabel = QLabel('Spacing')
        for label in (self.count_label, self.spacing_label):
            label.setEnabled(False)
        grid.addWidget(self.count_label, 2, 1)
        grid.addWidget(self.spacing_label, 2, 2)
        self.counts: list[SpinBox] = []
        self.spacings: list[DoubleSpinBox] = []
        for row, axis in enumerate('XYZ', start=3):
            count = SpinBox()
            count.setRange(1, 10_000)
            count.setToolTip(f'How many along {axis}, the original included')
            spacing = DoubleSpinBox()
            spacing.setDecimals(6)
            spacing.setRange(-1e9, 1e9)
            spacing.setToolTip(f'How far apart along {axis}; negative goes '
                               'the other way')
            grid.addWidget(QLabel(axis), row, 0)
            grid.addWidget(count, row, 1)
            grid.addWidget(spacing, row, 2)
            self.counts.append(count)
            self.spacings.append(spacing)
        self.reading_label: QLabel = QLabel()
        self.reading_label.setWordWrap(True)
        grid.addWidget(self.reading_label, 6, 0, 1, 3)
        buttons = QHBoxLayout()
        self.pattern_button: QPushButton = QPushButton('Pattern')
        self.pattern_button.setToolTip('Make the copies the preview shows, '
                                       'as one solid')
        self.pattern_button.setDefault(True)
        self.close_button: QPushButton = QPushButton('Close')
        self.close_button.setToolTip('Put the pane away without making the '
                                     'pattern')
        buttons.addWidget(self.pattern_button)
        buttons.addWidget(self.close_button)
        grid.addLayout(buttons, 7, 0, 1, 3)
        grid.setRowStretch(8, 1)
        commit_on_enter(*self.counts, *self.spacings)
        for box in (*self.counts, *self.spacings):
            box.valueChanged.connect(lambda _value: self.changed.emit())
        self.pattern_button.clicked.connect(self.pattern_asked.emit)
        self.close_button.clicked.connect(self.close_asked.emit)

    def open_for(self, unit: str, parts: str, editing: bool = False) -> None:
        """Set the pane up with every count one and every spacing zero,
        the spacing in the display unit `unit` and `parts` naming the
        solids. `editing` opens it on a pattern already made, from its
        row's pencil: the title reads Edit Pattern and the button
        Apply."""
        self.title.setText('Edit Pattern' if editing else 'Pattern')
        self.pattern_button.setText('Apply' if editing else 'Pattern')
        self.spacing_label.setText(f'Spacing [{unit}]')
        self.set_values(counts=(1, 1, 1), spacing=(0.0, 0.0, 0.0), quiet=True)
        self.parts_label.setText(f'Solids: {parts}')

    def values(self) -> dict:
        return {'counts': tuple(box.value() for box in self.counts),
                'spacing': tuple(box.value() for box in self.spacings)}

    def set_values(self, quiet: bool = False, **values) -> None:
        """Fill the fields by name, counts=(x, y, z) and
        spacing=(x, y, z), as a test would type them. With `quiet`,
        `changed` is not emitted."""
        boxes = [*self.counts, *self.spacings]
        for box in boxes:
            box.blockSignals(True)
        for key, numbers in values.items():
            for box, number in zip(self.counts if key == 'counts'
                                   else self.spacings, numbers):
                box.setValue(number)
        for box in boxes:
            box.blockSignals(False)
        if not quiet:
            self.changed.emit()

    def show_reading(self, text: str, ok: bool) -> None:
        self.reading_label.setText(text)
        self.pattern_button.setEnabled(ok)
