"""Mesh Solids, as a pane beside the 3-D view.

It follows select, see, set, apply (principle 13). The parts are picked
in the tree, the bricks they would be meshed into are drawn in the view
as the element size is typed, the pane says what that size makes, and
Mesh runs the project's `mesh_solids`. Keep apart joins the bricks only
to nodes of their own element groups (`mesh.join`), so a
part meshed beside another keeps its own nodes where they touch, to be
merged later at the places picked. With two or more parts, Keep the
parts apart from each other meshes them on one grid with each part
keeping its own nodes where the parts touch. A union that is no
extrusion is meshed in pieces, and then the size is one per piece, in
a table in place of the box: each piece's row takes its own, and
Detect Element Size fills every row.

The elements are Hexes, Tetras, or shells: Shells (quads) and Shells
(triangles), meshed on a plate's or a disk's plane and on the outer
surface of a solid with a volume. A plate or a disk among the parts
takes shells only, so only they are offered then.

The pane holds no rules of its own: the window asks `core.solids` what
the size makes, draws it, and shows the result here.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from .editors import LENGTH_DECIMALS, DoubleSpinBox, commit_on_enter
from .settings_panel import panel_grid

# The pane: Brandon, 2026-10-11.


#: the element kinds offered, as the tree names the families, and what
#: the box calls them; a flat part takes shells only (principle 3)
ELEMENT_LABELS = {'hexes': 'Hexes', 'tetras': 'Tetras', 'quads': 'Shells (quads)',
                  'triangles': 'Shells (triangles)'}
ELEMENT_KINDS = ('hexes', 'tetras', 'quads', 'triangles')
SHELL_KINDS = ('quads', 'triangles')


class MeshSolidsPanel(QWidget):
    """The element size, what it makes, and the Mesh button."""

    #: the size, the elements or Keep apart changed: the window meshes
    #: the preview again
    changed = Signal()
    mesh_asked = Signal()
    detect_asked = Signal()
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
        # a union meshed in pieces takes a size per piece (Brandon,
        # 2026-10-20: a 3 in block and 0.2 in flanges want two sizes,
        # which one fit cannot give)
        self.pieces_label: QLabel = QLabel('Element size per piece')
        self.pieces: QTableWidget = QTableWidget(0, 2)
        self.pieces.setHorizontalHeaderLabels(['Piece', 'Size'])
        self.pieces.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.pieces.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.pieces.verticalHeader().setVisible(False)
        self.pieces.setToolTip('Each piece of the union is meshed at its own size, swept '
                               'where it is an extrusion and fitted where it is not, and '
                               'the pieces are joined where their grids meet and tied '
                               'where they do not')
        grid.addWidget(self.pieces_label, 10, 0, 1, 2)
        grid.addWidget(self.pieces, 11, 0, 1, 2)
        self.pieces_label.hide()
        self.pieces.hide()
        self._piece_ids: list[int] = []
        # Brandon, 2026-10-19: a size the mesh is likely to hold at, from
        # the smallest feature, written into the box
        self.detect_button: QPushButton = QPushButton('Detect Element Size')
        self.detect_button.setToolTip(
            'Set the size from the smallest feature: five bricks across the '
            'smallest radius for bricks fitted to a shape, half the thinnest '
            'feature for an extrusion, three across for tetrahedra')
        grid.addWidget(self.detect_button, 2, 0, 1, 2)
        # the element type is the user's choice (Brandon, 2026-10-18):
        # bricks where the parts are an extrusion, tetrahedra for any
        # shape; a shape bricks cannot mesh says why in the reading
        self.elements_label: QLabel = QLabel('Elements')
        self.elements: QComboBox = QComboBox()
        # named as the tree names the families (Brandon, 2026-10-19);
        # shells, quads and triangles both (Brandon, 2026-10-26)
        self._kinds_shown: tuple[str, ...] = ()
        self._show_kinds(ELEMENT_KINDS)
        self.elements.setToolTip(
            'Hexes: eight-node bricks on a grid, for parts that are an '
            'extrusion along X, Y or Z, holes rounded, and fitted to any '
            'other shape. Tetras: ten-node tetrahedra for any shape, from '
            'the recipe alone. Shells: four-node plates or three-node '
            'triangles, on a plate\'s or a disk\'s plane and on the outer '
            'surface of a solid, their thickness set on the element group')
        grid.addWidget(self.elements_label, 3, 0)
        grid.addWidget(self.elements, 3, 1)
        self.parts_label: QLabel = QLabel()
        self.parts_label.setWordWrap(True)
        self.parts_label.setEnabled(False)
        grid.addWidget(self.parts_label, 4, 0, 1, 2)
        self.apart: QCheckBox = QCheckBox('Keep apart from other groups')
        self.apart.setToolTip(
            'Join only nodes of its own element group: where it touches '
            'another part it keeps nodes of its own, to be merged at the '
            'places picked with Merge Coincident Nodes')
        grid.addWidget(self.apart, 5, 0, 1, 2)
        # parts bolted together, meshed on one grid each with its own
        # nodes (the BARC, 2026-10-11): shown with two parts or more,
        # meshed into bricks
        self.parts_apart: QCheckBox = QCheckBox('Keep the parts apart from each other')
        self.parts_apart.setToolTip(
            'Mesh the parts on one grid, each keeping its own nodes where it '
            'touches another: they lie on each other\'s, to be merged at the '
            'places picked with Merge Coincident Nodes')
        grid.addWidget(self.parts_apart, 6, 0, 1, 2)
        self.reading_label: QLabel = QLabel()
        self.reading_label.setWordWrap(True)
        grid.addWidget(self.reading_label, 7, 0, 1, 2)
        buttons = QHBoxLayout()
        self.mesh_button: QPushButton = QPushButton('Mesh')
        self.mesh_button.setToolTip('Mesh the solids into elements at the '
                                    'size set above')
        self.mesh_button.setDefault(True)
        self.close_button: QPushButton = QPushButton('Close')
        self.close_button.setToolTip('Put the pane away without meshing')
        buttons.addWidget(self.mesh_button)
        buttons.addWidget(self.close_button)
        grid.addLayout(buttons, 8, 0, 1, 2)
        grid.setRowStretch(12, 1)
        commit_on_enter(self.size)
        self.size.valueChanged.connect(lambda _value: self.changed.emit())
        self.elements.currentIndexChanged.connect(self._elements_changed)
        self.apart.toggled.connect(lambda _on: self.changed.emit())
        self.parts_apart.toggled.connect(lambda _on: self.changed.emit())
        self.mesh_button.clicked.connect(self.mesh_asked.emit)
        self.detect_button.clicked.connect(self.detect_asked.emit)
        self.close_button.clicked.connect(self.close_asked.emit)

    def open_for(self, unit: str, size: float, parts: str, count: int = 1,
                 flat: bool = False) -> None:
        """Set the pane up: the size in the display unit `unit`, `parts`
        naming the parts it meshes, and `count`, how many there are.
        Both Keep apart boxes are cleared, and the second is shown only
        for two or more parts meshed into bricks; the elements are
        bricks again. With `flat`, a plate or a disk among the parts,
        only shells are offered, quads first."""
        self.size_label.setText(f'Element size [{unit}]')
        self.size.blockSignals(True)
        self.size.setSingleStep(max(size / 4.0, 1e-6))
        self.size.setValue(size)
        self.size.blockSignals(False)
        for box in (self.apart, self.parts_apart):
            box.blockSignals(True)
            box.setChecked(False)
            box.blockSignals(False)
        self.elements.blockSignals(True)
        self._show_kinds(SHELL_KINDS if flat else ELEMENT_KINDS)
        self.elements.setCurrentIndex(0)
        self.elements.blockSignals(False)
        self._count = count
        self.parts_apart.setVisible(count > 1 and not flat)
        self.parts_label.setText(f'Solids: {parts}')

    def _show_kinds(self, kinds) -> None:
        """Offer these element kinds, in this order."""
        if tuple(kinds) == self._kinds_shown:
            return
        self.elements.clear()
        for kind in kinds:
            self.elements.addItem(ELEMENT_LABELS[kind], kind)
        self._kinds_shown = tuple(kinds)

    @property
    def element_kind(self) -> str:
        """'hexes', 'tetras', 'quads' or 'triangles', as picked."""
        return str(self.elements.currentData())

    def set_pieces(self, unit: str, rows) -> None:
        """Show a size per piece, `rows` being (solid id, name, size)
        in the display unit, in place of the one box; None puts the box
        back."""
        rows = list(rows or [])
        self._piece_ids = [int(i) for i, _n, _s in rows]
        self.pieces.setRowCount(len(rows))
        for r, (_i, name, size) in enumerate(rows):
            item = QTableWidgetItem(str(name))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.pieces.setItem(r, 0, item)
            box = DoubleSpinBox()
            box.setDecimals(LENGTH_DECIMALS)
            box.setRange(0.0, 1e9)
            box.setSingleStep(max(size / 4.0, 1e-6))
            box.setValue(size)
            commit_on_enter(box)
            box.valueChanged.connect(lambda _value: self.changed.emit())
            self.pieces.setCellWidget(r, 1, box)
        self.pieces_label.setText(f'Element size per piece [{unit}]')
        showing = bool(rows)
        self.pieces_label.setVisible(showing)
        self.pieces.setVisible(showing)
        self.size_label.setVisible(not showing)
        self.size.setVisible(not showing)
        self.detect_button.setText('Detect Element Sizes' if showing else 'Detect Element Size')
        self.pieces.setFixedHeight(
            self.pieces.horizontalHeader().height() + 30 * len(rows) + 4)

    @property
    def piece_ids(self) -> list[int]:
        """The pieces shown, by solid id; empty with the one box."""
        return list(self._piece_ids)

    def piece_sizes(self) -> dict[int, float]:
        """Each piece's size as typed, by solid id."""
        return {int(i): float(self.pieces.cellWidget(r, 1).value())
                for r, i in enumerate(self._piece_ids)}

    def set_piece_size(self, solid_id: int, size: float) -> None:
        box = self.pieces.cellWidget(self._piece_ids.index(int(solid_id)), 1)
        box.blockSignals(True)
        box.setValue(size)
        box.blockSignals(False)

    def _elements_changed(self, _index: int) -> None:
        # the parts can be kept apart on one grid: a brick option
        self.parts_apart.setVisible(self.element_kind == 'hexes'
                                    and self.__dict__.get('_count', 1) > 1)
        self.changed.emit()

    def show_reading(self, text: str, ok: bool) -> None:
        self.reading_label.setText(text)
        self.mesh_button.setEnabled(ok)
