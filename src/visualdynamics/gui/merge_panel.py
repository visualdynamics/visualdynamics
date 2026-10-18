"""Merge Coincident Nodes and Align Meshes, as a pane beside the 3-D view.

Merge follows select, see, set, apply (principle 13). The element
groups are picked in the tree, every place their nodes coincide is
marked in the view, and the tolerance is set here. A click on a mark
leaves that place out of the merge and a second click puts it back;
All and None mark every place or none. Merge then runs the project's
`merge_coincident_nodes` with the places left out as its `exclude`.
Leaving places out is how a joint is built from parts meshed
separately, one place at a time, instead of with rigid links.

Align Meshes uses the same pane. Parts meshed separately at element
sizes that do not match have no coincident nodes to merge, so this
step, done before the merge, moves the coarser mesh's nodes onto the
finer mesh's where they meet. The pane shows a reach in place of the
tolerance, each mark is a move (the node, and a line to where it goes),
and Align runs the project's `align_meshes`.

The pane holds no rules of its own. The window asks the geometry what
the settings would merge (`coincident_spots`, `merge_folds`) or move
(`align_moves`, `align_folds`), marks it in the view, and shows the
result here.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from .editors import DoubleSpinBox, commit_on_enter
from .settings_panel import panel_grid

# Brandon, 2026-10-11: the merge was a question and an answer, a
# tolerance asked in a dialog and every coincidence within it made one
# node. He wanted to see what it would merge before it does, and to
# leave some of it out, so it became this pane; Align Meshes joined it
# the same day.

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
    """The tolerance, what it finds, and the Merge button; or, for Align
    Meshes, the reach, the moves and the Align button."""

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
        self.merge_button.setToolTip('Make the coincident nodes picked one '
                                     'node each')
        self.merge_button.setDefault(True)
        self.close_button: QPushButton = QPushButton('Close')
        self.close_button.setToolTip('Put the pane away without merging')
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
        (or the reach) in the display unit `unit`, and `groups`, the
        names of the element groups it works on."""
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
        """The tolerance (or the reach) as typed, in the display unit."""
        return self.tolerance.value()

    def set_value(self, tolerance: float) -> None:
        """Set the tolerance (or the reach) as if it were typed.
        `changed` is emitted, so the window finds the places again."""
        self.tolerance.setValue(tolerance)

    def show_reading(self, text: str, ok: bool) -> None:
        self.reading_label.setText(text)
        self.merge_button.setEnabled(ok)
