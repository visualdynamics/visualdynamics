"""Mate Surfaces, as a pane beside the 3-D view.

It follows select, see, set, apply (principle 13). Two element groups
or more are picked in the tree and Mate Surfaces opened on them: every
patch where their surfaces mate is marked in the view, and listed here,
one entry per connected patch. A patch is picked from the list, the side
to pick nodes for is chosen, and a click on a node in the view adds it
to that side or takes it away. Only that side's group's nodes are
drawn, so coincident nodes of the two sides are told apart. The normal
and the shear are each held rigidly, by a stiffness in the display
units, or left free, every node in a frame of its own on the surface it
lies on, and Keep adds the patch to
the geometry's Mated Surfaces, where it can be picked, edited or deleted
later. Opened from a set's row, the pane edits that set.

The pane holds no rules of its own: the window asks `core.mates` where
the surfaces mate, marks it, and shows the result here.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QWidget,
)

from .editors import LENGTH_DECIMALS, DoubleSpinBox, commit_on_enter
from .settings_panel import panel_grid

# Brandon, 2026-10-20: on element groups picked in the tree rather than
# the whole geometry, a patch at a time, its nodes picked by hand, kept
# as a sub-object of the geometry. Share nodes was dropped: Merge
# Coincident Nodes does that.

#: how the normal and the shear are each held, as the pane names them
#: and `core.mates` takes them
MODES = (('Rigid', 'rigid'), ('Stiffness', 'stiffness'), ('Free', 'free'))


class MatePanel(QWidget):
    """The patches, the side being picked, how the normal and the shear
    are held and their stiffness, a name, what the patch makes, and
    Keep."""

    #: the tolerance changed: the window finds the patches again
    tolerance_changed = Signal()
    #: the distance a node may be picked within changed: the window
    #: marks the nodes that can be picked again
    search_changed = Signal()
    #: the patch, the side or how it is held changed: the window marks
    #: the patch again and says what Keep would do
    changed = Signal()
    keep_asked = Signal()
    close_asked = Signal()
    #: every node of the side being picked back, or none
    all_asked = Signal()
    none_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Mate Surfaces')
        grid = panel_grid(self, self.title)
        self.groups_label: QLabel = QLabel()
        self.groups_label.setWordWrap(True)
        grid.addWidget(self.groups_label, 1, 0, 1, 2)
        self.tolerance_label: QLabel = QLabel('Tolerance')
        self.tolerance: DoubleSpinBox = DoubleSpinBox()
        self.tolerance.setDecimals(LENGTH_DECIMALS)
        self.tolerance.setRange(0.0, 1e9)
        self.tolerance.setToolTip('How far a node may be from the other surface '
                                  'and still lie on it, in the display unit')
        grid.addWidget(self.tolerance_label, 2, 0)
        grid.addWidget(self.tolerance, 2, 1)
        # how far from the mated surface a node can be picked (Brandon,
        # 2026-10-20: every node of the group was pickable)
        self.search_label: QLabel = QLabel('Pick within')
        self.search: DoubleSpinBox = DoubleSpinBox()
        self.search.setDecimals(LENGTH_DECIMALS)
        self.search.setRange(0.0, 1e9)
        self.search.setToolTip('How far from the other side\'s surface a node may be '
                               'and still be picked or unpicked, in the display unit')
        grid.addWidget(self.search_label, 3, 0)
        grid.addWidget(self.search, 3, 1)
        self.patches: QListWidget = QListWidget()
        self.patches.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.patches.setToolTip('Each patch where two of the element groups mate; '
                                'pick one to change its nodes, or several to keep '
                                'them together')
        self.patches.setMinimumHeight(90)
        grid.addWidget(self.patches, 4, 0, 1, 2)
        self.side_label: QLabel = QLabel('Pick nodes of')
        sides = QHBoxLayout()
        self.side_a: QPushButton = QPushButton()
        self.side_b: QPushButton = QPushButton()
        self.sides: QButtonGroup = QButtonGroup(self)
        for k, button in enumerate((self.side_a, self.side_b)):
            button.setCheckable(True)
            self.sides.addButton(button, k)
            sides.addWidget(button)
        self.side_a.setChecked(True)
        self.side_a.setToolTip('A click on a node adds it to the side that follows, '
                               'or takes it away')
        self.side_b.setToolTip('A click on a node adds it to the side followed, or '
                               'takes it away')
        grid.addWidget(self.side_label, 5, 0)
        grid.addLayout(sides, 5, 1)
        # every node of the side back, or none, to pick a few by clicking:
        # a beam's end on a channel is 81 nodes and its bolts clamp 16
        # (the BARC, 2026-10-20), as the merge pane's All and None
        picks = QHBoxLayout()
        self.all_button: QPushButton = QPushButton('All')
        self.all_button.setToolTip('Every node the side had when the patch was found')
        self.none_button: QPushButton = QPushButton('None')
        self.none_button.setToolTip('Take every node off the side, to click a few back')
        picks.addWidget(self.all_button)
        picks.addWidget(self.none_button)
        grid.addLayout(picks, 6, 1)
        # the normal and the shear, each held its own way (Brandon,
        # 2026-10-20): rigid, a stiffness in the display units, or free
        self.normal_mode_label: QLabel = QLabel('Normal')
        self.normal_mode: QComboBox = self._modes(
            'How each node is held along the normal of the surface it lies on')
        self.normal_label: QLabel = QLabel('Normal stiffness')
        self.normal: DoubleSpinBox = self._stiffness_box(
            "Each node's stiffness along the surface's normal")
        self.shear_mode_label: QLabel = QLabel('Shear')
        self.shear_mode: QComboBox = self._modes(
            "How each node is held in the plane of the joint, across the normal")
        self.shear_label: QLabel = QLabel('Shear stiffness')
        self.shear: DoubleSpinBox = self._stiffness_box(
            "Each node's stiffness in the plane of the joint, the same both ways")
        for row, (label, widget) in enumerate(
                ((self.normal_mode_label, self.normal_mode), (self.normal_label, self.normal),
                 (self.shear_mode_label, self.shear_mode), (self.shear_label, self.shear)),
                start=7):
            grid.addWidget(label, row, 0)
            grid.addWidget(widget, row, 1)
        self.name_label: QLabel = QLabel('Name')
        self.name: QLineEdit = QLineEdit()
        self.name.setToolTip('What the tree calls the set kept')
        grid.addWidget(self.name_label, 11, 0)
        grid.addWidget(self.name, 11, 1)
        self.reading_label: QLabel = QLabel()
        self.reading_label.setWordWrap(True)
        grid.addWidget(self.reading_label, 12, 0, 1, 2)
        buttons = QHBoxLayout()
        self.keep_button: QPushButton = QPushButton('Keep')
        self.keep_button.setToolTip('Add the picked patches to the geometry\'s Mated '
                                    'Surfaces, joined as set')
        self.keep_button.setDefault(True)
        self.close_button: QPushButton = QPushButton('Close')
        self.close_button.setToolTip('Put the pane away; what was kept stays')
        buttons.addWidget(self.keep_button)
        buttons.addWidget(self.close_button)
        grid.addLayout(buttons, 13, 0, 1, 2)
        grid.setRowStretch(14, 1)
        for box in (self.tolerance, self.search, self.normal, self.shear):
            commit_on_enter(box)
        self.search.valueChanged.connect(lambda _value: self.search_changed.emit())
        self.tolerance.valueChanged.connect(lambda _value: self.tolerance_changed.emit())
        self.patches.itemSelectionChanged.connect(self.changed.emit)
        self.sides.idToggled.connect(lambda _id, on: on and self.changed.emit())
        for box in (self.normal_mode, self.shear_mode):
            box.currentIndexChanged.connect(self._mode_changed)
        self.normal.valueChanged.connect(lambda _value: self.changed.emit())
        self.shear.valueChanged.connect(lambda _value: self.changed.emit())
        self.keep_button.clicked.connect(self.keep_asked.emit)
        self.all_button.clicked.connect(self.all_asked.emit)
        self.none_button.clicked.connect(self.none_asked.emit)
        self.close_button.clicked.connect(self.close_asked.emit)
        self._show_stiffness()

    def open_for(self, groups: str, unit: str, tolerance: float | None,
                 stiffness_unit: str, normal: float, shear: float,
                 normal_mode: str = 'rigid', shear_mode: str = 'rigid',
                 search: float = 0.0) -> None:
        """Set the pane up on the element groups named `groups`: the
        tolerance in the display unit `unit`, or None when a kept set is
        being edited (there is nothing to find again), the distance a
        node may be picked within, `search`, in the same unit, and the
        stiffness boxes in `stiffness_unit`."""
        editing = tolerance is None
        self.title.setText('Edit Mated Surfaces' if editing else 'Mate Surfaces')
        self.groups_label.setText(groups)
        self.tolerance_label.setVisible(not editing)
        self.tolerance.setVisible(not editing)
        self.keep_button.setText('Save' if editing else 'Keep')
        self.keep_button.setToolTip('Save the set as changed' if editing else
                                    'Add the picked patches to the geometry\'s Mated '
                                    'Surfaces, joined as set')
        if not editing:
            self.tolerance.blockSignals(True)
            self.tolerance.setSingleStep(max(tolerance, 1e-6))
            self.tolerance.setValue(tolerance)
            self.tolerance.blockSignals(False)
        self.search_label.setText(f'Pick within [{unit}]')
        self.search.blockSignals(True)
        self.search.setSingleStep(max(search, 1e-6))
        self.search.setValue(search)
        self.search.blockSignals(False)
        self.normal_label.setText(f'Normal stiffness [{stiffness_unit}]')
        self.shear_label.setText(f'Shear stiffness [{stiffness_unit}]')
        for box, value in ((self.normal, normal), (self.shear, shear)):
            box.blockSignals(True)
            box.setSingleStep(max(value / 10.0, 1e-6))
            box.setValue(value)
            box.blockSignals(False)
        for box, mode in ((self.normal_mode, normal_mode), (self.shear_mode, shear_mode)):
            box.blockSignals(True)
            box.setCurrentIndex(box.findData(mode))
            box.blockSignals(False)
        self._show_stiffness()
        self.side_a.setChecked(True)

    def set_patches(self, rows, current: int | None = 0) -> None:
        """The patches to pick from, each row's text, `current` picked."""
        self.patches.blockSignals(True)
        self.patches.clear()
        for text in rows:
            self.patches.addItem(QListWidgetItem(text))
        if current is not None and 0 <= current < self.patches.count():
            self.patches.item(current).setSelected(True)
            self.patches.setCurrentRow(current)
        self.patches.blockSignals(False)
        self.patches.setEnabled(self.patches.count() > 0)

    def set_patch_text(self, row: int, text: str) -> None:
        item = self.patches.item(row)
        if item is not None and item.text() != text:
            item.setText(text)

    def picked_patches(self) -> list[int]:
        """The patches picked, by row, in the list's order."""
        return sorted(self.patches.row(item) for item in self.patches.selectedItems())

    def current_patch(self) -> int | None:
        """The patch whose nodes a click changes: the one picked, or the
        list's current row among several picked; None with none."""
        picked = self.picked_patches()
        if not picked:
            return None
        row = self.patches.currentRow()
        return row if row in picked else picked[0]

    def set_sides(self, first: str, second: str) -> None:
        """The side buttons named for the current patch's groups."""
        self.side_a.setText(first)
        self.side_b.setText(second)

    @property
    def side(self) -> str:
        """'a' or 'b': the side a click on a node changes."""
        return 'b' if self.side_b.isChecked() else 'a'

    @staticmethod
    def _modes(tip: str) -> QComboBox:
        box = QComboBox()
        for label, key in MODES:
            box.addItem(label, key)
        box.setToolTip(tip)
        return box

    @staticmethod
    def _stiffness_box(tip: str) -> DoubleSpinBox:
        box = DoubleSpinBox()
        box.setDecimals(3)
        box.setRange(0.0, 1e30)
        box.setToolTip(tip)
        return box

    @property
    def normal_held(self) -> str:
        """'rigid', 'stiffness' or 'free', as picked for the normal."""
        return str(self.normal_mode.currentData())

    @property
    def shear_held(self) -> str:
        """'rigid', 'stiffness' or 'free', as picked for the shear."""
        return str(self.shear_mode.currentData())

    def _mode_changed(self, _index: int) -> None:
        self._show_stiffness()
        self.changed.emit()

    def _show_stiffness(self) -> None:
        # a stiffness box applies where its direction takes a stiffness
        # (principle 3)
        for held, label, box in ((self.normal_held, self.normal_label, self.normal),
                                 (self.shear_held, self.shear_label, self.shear)):
            label.setVisible(held == 'stiffness')
            box.setVisible(held == 'stiffness')

    def value(self) -> float:
        """The tolerance as typed, in the display unit."""
        return self.tolerance.value()

    def show_reading(self, text: str, ok: bool) -> None:
        self.reading_label.setText(text)
        self.keep_button.setEnabled(ok)
