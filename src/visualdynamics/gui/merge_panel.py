"""Merge Coincident Nodes, as a pane beside the 3-D view.

The merge was a question and an answer: a tolerance asked in a dialog,
and every coincidence within it made one node. Brandon (2026-10-11)
wanted to see what it would merge before it does, and to leave some
of it out — the way a joint is built from parts meshed apart, one
place at a time, instead of with rigid links — so the merge became
select, see, set, apply (principle 13): the element groups picked in
the tree, every place their nodes coincide marked in the view, the
tolerance set here, a click on a mark in the view to leave that place
out (and again to put it back), and **Merge**, the project's
`merge_coincident_nodes` with the places left out as its `exclude`.

The pane holds no rule of its own: the window asks the geometry
(`coincident_spots`, `merge_folds`) what the settings would merge,
marks it, and says it here.

**Align Meshes** is the same pane (Brandon, 2026-10-11): parts meshed
apart at sizes that do not match have no coincident nodes to merge, so
the step before the merge moves the coarser mesh's nodes onto the
finer's where they meet. A reach in place of the tolerance, a move per
mark (the node, and a line to where it goes), **Align** in place of
Merge (`align_moves`, `align_folds`, the project's `align_meshes`).
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from .editors import DoubleSpinBox, commit_on_enter
from .settings_panel import panel_grid

#: what the pane says for each kind: (title, the number's label, its
#: tooltip, the button)
KINDS = {
    'merge': ('Merge Coincident Nodes', 'Tolerance',
              'Nodes closer than this are one place, in the display unit',
              'Merge'),
    'align': ('Align Meshes', 'Reach',
              ('How far a node of the coarser mesh may move onto the '
               "finer mesh's nearest node, in the display unit"),
              'Align'),
}


class MergePanel(QWidget):
    """The tolerance, what it finds, and the Merge — or the reach, the
    moves, and the Align."""

    #: the tolerance was edited: the window finds the places again
    changed = Signal()
    #: every place marked to merge, or none
    all_asked = Signal()
    none_asked = Signal()
    merge_asked = Signal()
    close_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Merge Coincident Nodes')
        grid = panel_grid(self, self.title)
        self.tolerance_label: QLabel = QLabel('Tolerance')
        self.tolerance: DoubleSpinBox = DoubleSpinBox()
        self.tolerance.setDecimals(6)
        self.tolerance.setRange(0.0, 1e9)
        self.tolerance.setToolTip(
            'Nodes closer than this are one place, in the display unit')
        grid.addWidget(self.tolerance_label, 1, 0)
        grid.addWidget(self.tolerance, 1, 1)
        self.groups_label: QLabel = QLabel()
        self.groups_label.setWordWrap(True)
        self.groups_label.setEnabled(False)
        grid.addWidget(self.groups_label, 2, 0, 1, 2)
        self.reading_label: QLabel = QLabel()
        self.reading_label.setWordWrap(True)
        grid.addWidget(self.reading_label, 3, 0, 1, 2)
        picks = QHBoxLayout()
        self.all_button: QPushButton = QPushButton('All')
        self.all_button.setToolTip('Mark every place to merge')
        self.none_button: QPushButton = QPushButton('None')
        self.none_button.setToolTip('Leave every place out, to mark a few '
                                    'by clicking them in the view')
        picks.addWidget(self.all_button)
        picks.addWidget(self.none_button)
        grid.addLayout(picks, 4, 0, 1, 2)
        buttons = QHBoxLayout()
        self.merge_button: QPushButton = QPushButton('Merge')
        self.merge_button.setDefault(True)
        self.close_button: QPushButton = QPushButton('Close')
        buttons.addWidget(self.merge_button)
        buttons.addWidget(self.close_button)
        grid.addLayout(buttons, 5, 0, 1, 2)
        grid.setRowStretch(6, 1)
        commit_on_enter(self.tolerance)
        self.tolerance.valueChanged.connect(lambda _value: self.changed.emit())
        self.all_button.clicked.connect(self.all_asked.emit)
        self.none_button.clicked.connect(self.none_asked.emit)
        self.merge_button.clicked.connect(self.merge_asked.emit)
        self.close_button.clicked.connect(self.close_asked.emit)

    def open_for(self, unit: str, tolerance: float, groups: str,
                 kind: str = 'merge') -> None:
        """Set the pane up for `kind`, 'merge' or 'align': the tolerance
        (or the reach) in the display unit `unit`, and the element groups
        it works on, named."""
        title, label, tip, button = KINDS[kind]
        self.title.setText(title)
        self.tolerance_label.setText(f'{label} [{unit}]')
        self.tolerance.setToolTip(tip)
        self.merge_button.setText(button)
        self.all_button.setToolTip(f'Mark every place to {button.lower()}')
        self.tolerance.blockSignals(True)
        # a step a tenth of the starting tolerance's order, so the
        # arrows move it by something that can matter
        self.tolerance.setSingleStep(max(tolerance, 1e-6))
        self.tolerance.setValue(tolerance)
        self.tolerance.blockSignals(False)
        self.groups_label.setText(f'Element groups: {groups}')

    def value(self) -> float:
        """The tolerance as typed, in the display unit."""
        return self.tolerance.value()

    def set_value(self, tolerance: float) -> None:
        """Type a tolerance, as a person would: the places are found
        again."""
        self.tolerance.setValue(tolerance)

    def show_reading(self, text: str, ok: bool) -> None:
        self.reading_label.setText(text)
        self.merge_button.setEnabled(ok)
