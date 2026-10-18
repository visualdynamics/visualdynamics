"""Mesh Solids, as a pane beside the 3-D view.

It follows select, see, set, apply (principle 13). The parts are picked
in the tree, the bricks they would be meshed into are drawn in the view
as the element size is typed, the pane says what that size makes, and
Mesh runs the project's `mesh_solids`. Keep apart joins the bricks only
to nodes of their own element groups, as Add Block's option does, so a
part meshed beside another keeps its own nodes where they touch, to be
merged later at the places picked. With two or more parts, Keep the
parts apart from each other meshes them on one grid with each part
keeping its own nodes where the parts touch.

The pane holds no rules of its own: the window asks `core.solids` what
the size makes, draws it, and shows the result here.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QWidget

from .editors import LENGTH_DECIMALS, DoubleSpinBox, commit_on_enter
from .settings_panel import panel_grid

# The pane: Brandon, 2026-10-11.


class MeshSolidsPanel(QWidget):
    """The element size, what it makes, and the Mesh button."""

    #: the size or Keep apart changed: the window meshes the preview again
    changed = Signal()
    mesh_asked = Signal()
    close_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Mesh Solids')
        grid = panel_grid(self, self.title)
        self.size_label: QLabel = QLabel('Element size')
        self.size: DoubleSpinBox = DoubleSpinBox()
        self.size.setDecimals(LENGTH_DECIMALS)
        self.size.setRange(0.0, 1e9)
        self.size.setToolTip('Each span between flat faces is divided into '
                             'the whole number of bricks nearest this size')
        grid.addWidget(self.size_label, 1, 0)
        grid.addWidget(self.size, 1, 1)
        self.parts_label: QLabel = QLabel()
        self.parts_label.setWordWrap(True)
        self.parts_label.setEnabled(False)
        grid.addWidget(self.parts_label, 2, 0, 1, 2)
        self.apart: QCheckBox = QCheckBox('Keep apart from other groups')
        self.apart.setToolTip(
            'Join only nodes of its own element group: where it touches '
            'another part it keeps nodes of its own, to be merged at the '
            'places picked with Merge Coincident Nodes')
        grid.addWidget(self.apart, 3, 0, 1, 2)
        # parts bolted together, meshed on one grid each with its own
        # nodes (the BARC, 2026-10-11): shown with two parts or more
        self.parts_apart: QCheckBox = QCheckBox('Keep the parts apart from each other')
        self.parts_apart.setToolTip(
            'Mesh the parts on one grid, each keeping its own nodes where it '
            'touches another: they lie on each other\'s, to be merged at the '
            'places picked with Merge Coincident Nodes')
        grid.addWidget(self.parts_apart, 4, 0, 1, 2)
        self.reading_label: QLabel = QLabel()
        self.reading_label.setWordWrap(True)
        grid.addWidget(self.reading_label, 5, 0, 1, 2)
        buttons = QHBoxLayout()
        self.mesh_button: QPushButton = QPushButton('Mesh')
        self.mesh_button.setToolTip('Mesh the solids into elements at the '
                                    'size set above')
        self.mesh_button.setDefault(True)
        self.close_button: QPushButton = QPushButton('Close')
        self.close_button.setToolTip('Put the pane away without meshing')
        buttons.addWidget(self.mesh_button)
        buttons.addWidget(self.close_button)
        grid.addLayout(buttons, 6, 0, 1, 2)
        grid.setRowStretch(7, 1)
        commit_on_enter(self.size)
        self.size.valueChanged.connect(lambda _value: self.changed.emit())
        self.apart.toggled.connect(lambda _on: self.changed.emit())
        self.parts_apart.toggled.connect(lambda _on: self.changed.emit())
        self.mesh_button.clicked.connect(self.mesh_asked.emit)
        self.close_button.clicked.connect(self.close_asked.emit)

    def open_for(self, unit: str, size: float, parts: str, count: int = 1) -> None:
        """Set the pane up: the size in the display unit `unit`, `parts`
        naming the parts it meshes, and `count`, how many there are.
        Both Keep apart boxes are cleared, and the second is shown only
        for two or more parts."""
        self.size_label.setText(f'Element size [{unit}]')
        self.size.blockSignals(True)
        self.size.setSingleStep(max(size / 4.0, 1e-6))
        self.size.setValue(size)
        self.size.blockSignals(False)
        for box in (self.apart, self.parts_apart):
            box.blockSignals(True)
            box.setChecked(False)
            box.blockSignals(False)
        self.parts_apart.setVisible(count > 1)
        self.parts_label.setText(f'Solids: {parts}')

    def show_reading(self, text: str, ok: bool) -> None:
        self.reading_label.setText(text)
        self.mesh_button.setEnabled(ok)
