"""Add Block and Add Plane, typed into a pane beside the 3-D view.

The pop-up windows these replaced (`plane_dialog`, 2026-09-26 and
2026-09-30) took a corner and two or three edges; Brandon wanted no
window at all, and the box described the way it is thought of — where
its center is and how wide it is along each axis — with the preview in
the view turned and slid by the same gizmo that turns a coordinate
system (2026-10-02). So the pane holds a center, three widths and an
element size, in the display unit, and the window holds the frame the
box is drawn in: the rings turn it about its center, the arrows slide
the center along its own axes onto the grid. The rotation is also three
fields, degrees about the geometry's X, Y and Z, for an odd angle the
rings' whole degrees do not reach (Brandon, 2026-10-02); the rings
write the angles back, so the two never disagree.

A block takes three widths. A plate lies in a plane, and the plane is
named by the one width left at zero — a width of zero along Z is a
plate in the frame's X-Y plane — so a plate and a block are typed the
same way.

The pane holds no rule of its own. The window turns the fields and its
frame into the corner and edges the project's `add_block` and
`add_plane` take, asks the core what they would make, and draws it;
**Add** is the project verb, so the journal line is the call a script
makes.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from .editors import DoubleSpinBox, commit_on_enter
from .settings_panel import panel_grid

#: what the fields open on, in the display unit: a box one unit on a side
#: in X and Y and a tenth deep, sitting on the origin's corner, a tenth
#: per element — the old dialog's defaults, said as a center
DEFAULTS = {
    'block': {'center': (0.5, 0.5, 0.05), 'widths': (1.0, 1.0, 0.1)},
    'plane': {'center': (0.5, 0.5, 0.0), 'widths': (1.0, 1.0, 0.0)},
}
DEFAULT_SIZE = 0.1


def _box(value: float, minimum: float = -1e9) -> DoubleSpinBox:
    box = DoubleSpinBox()
    box.setDecimals(4)
    box.setRange(minimum, 1e9)
    box.setValue(value)
    return box


class MeshPanel(QWidget):
    """The fields for one box or plate, and its Add."""

    #: a field was edited: the window reads `values()` and redraws
    changed = Signal()
    add_asked = Signal()
    close_asked = Signal()
    #: square the preview's frame up with the geometry's axes
    square_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind: str = 'block'
        self.title: QLabel = QLabel('Add Block')
        grid = panel_grid(self, self.title)
        self.group: QComboBox = QComboBox()
        self.group.setEditable(True)
        self.group.setToolTip('The element group the elements go in: one '
                              "of the geometry's, or a new name")
        grid.addWidget(QLabel('Element group'), 1, 0)
        grid.addWidget(self.group, 1, 1)
        self.center_label: QLabel = QLabel('Center')
        self.center_label.setEnabled(False)
        grid.addWidget(self.center_label, 2, 0, 1, 2)
        self.center: list[DoubleSpinBox] = []
        for row, axis in enumerate('XYZ', start=3):
            box = _box(0.0)
            box.setToolTip(f'{axis} of the center, in the display unit; the '
                           'arrows in the view slide it onto the grid')
            grid.addWidget(QLabel(axis), row, 0)
            grid.addWidget(box, row, 1)
            self.center.append(box)
        self.width_label: QLabel = QLabel('Width')
        self.width_label.setEnabled(False)
        grid.addWidget(self.width_label, 6, 0, 1, 2)
        self.widths: list[DoubleSpinBox] = []
        for row, axis in enumerate('XYZ', start=7):
            box = _box(0.0, minimum=0.0)
            box.setToolTip(f'The width along the frame\'s {axis}; a plate '
                           'leaves one width at zero, which names its plane')
            grid.addWidget(QLabel(axis), row, 0)
            grid.addWidget(box, row, 1)
            self.widths.append(box)
        self.angle_label: QLabel = QLabel('Rotation [\u00b0]')
        self.angle_label.setEnabled(False)
        grid.addWidget(self.angle_label, 10, 0, 1, 2)
        self.angles: list[DoubleSpinBox] = []
        for row, axis in enumerate('XYZ', start=11):
            box = _box(0.0, minimum=-360.0)
            box.setMaximum(360.0)
            box.setDecimals(2)
            box.setToolTip(f'Degrees about the geometry\'s {axis} axis, '
                           'applied X, then Y, then Z; the rings in the view '
                           'turn it too, a degree at a time')
            grid.addWidget(QLabel(axis), row, 0)
            grid.addWidget(box, row, 1)
            self.angles.append(box)
        self.size_label: QLabel = QLabel('Element size')
        self.size: DoubleSpinBox = _box(DEFAULT_SIZE, minimum=0.0)
        self.size.setToolTip('Each edge is divided into the whole number of '
                             'elements nearest this size')
        grid.addWidget(self.size_label, 14, 0)
        grid.addWidget(self.size, 14, 1)
        self.square_button: QPushButton = QPushButton('Square Up')
        self.square_button.setToolTip(
            "Turn the box back to the geometry's axes, keeping its center")
        grid.addWidget(self.square_button, 15, 0, 1, 2)
        self.reading_label: QLabel = QLabel()
        self.reading_label.setWordWrap(True)
        grid.addWidget(self.reading_label, 16, 0, 1, 2)
        buttons = QHBoxLayout()
        self.add_button: QPushButton = QPushButton('Add')
        self.add_button.setDefault(True)
        self.close_button: QPushButton = QPushButton('Close')
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.close_button)
        grid.addLayout(buttons, 17, 0, 1, 2)
        grid.setRowStretch(18, 1)
        commit_on_enter(*self.center, *self.widths, *self.angles, self.size)
        for box in (*self.center, *self.widths, *self.angles, self.size):
            box.valueChanged.connect(lambda _value: self.changed.emit())
        self.group.currentTextChanged.connect(lambda _text: self.changed.emit())
        self.add_button.clicked.connect(self.add_asked.emit)
        self.close_button.clicked.connect(self.close_asked.emit)
        self.square_button.clicked.connect(self.square_asked.emit)

    def open_for(self, kind: str, unit: str, groups: Sequence[str],
                 default: str | None = None) -> None:
        """Set the pane up for a block or a plane, in the display unit
        `unit`, with the geometry's element groups to choose from, opening on
        `default`; the fields open on the defaults."""
        self.kind = kind
        noun = 'Block' if kind == 'block' else 'Plane'
        self.title.setText(f'Add {noun}')
        self.center_label.setText(f'Center [{unit}]')
        self.width_label.setText(f'Width [{unit}]')
        self.size_label.setText(f'Element size [{unit}]')
        names = [name for name in groups if name]
        self.group.blockSignals(True)
        self.group.clear()
        self.group.addItems(names)
        self.group.setCurrentText(default or ('block' if kind == 'block'
                                              else 'plate'))
        self.group.blockSignals(False)
        defaults = DEFAULTS[kind]
        self.set_values(center=defaults['center'], widths=defaults['widths'],
                        angles=(0.0, 0.0, 0.0), size=DEFAULT_SIZE, quiet=True)

    def values(self) -> dict:
        """The fields as typed, in the display unit."""
        return {'center': tuple(box.value() for box in self.center),
                'widths': tuple(box.value() for box in self.widths),
                'angles': tuple(box.value() for box in self.angles),
                'size': self.size.value(),
                'group': self.group.currentText().strip()}

    def set_values(self, quiet: bool = False, **values) -> None:
        """Fill fields by name — center=(x, y, z), widths=(x, y, z),
        angles=(x, y, z) in degrees, size=..., group=...; how a test
        types, and how the gizmo writes the center and the angles back.
        `quiet` fills them without a `changed`."""
        boxes = [*self.center, *self.widths, *self.angles, self.size]
        if quiet:
            for box in boxes:
                box.blockSignals(True)
            self.group.blockSignals(True)
        try:
            for key, value in values.items():
                if key == 'group':
                    self.group.setCurrentText(value)
                elif key == 'size':
                    self.size.setValue(value)
                else:
                    for box, number in zip(getattr(self, key), value):
                        box.setValue(number)
        finally:
            if quiet:
                for box in boxes:
                    box.blockSignals(False)
                self.group.blockSignals(False)
        if not quiet:
            self.changed.emit()

    def show_reading(self, text: str, ok: bool) -> None:
        self.reading_label.setText(text)
        self.add_button.setEnabled(ok)
