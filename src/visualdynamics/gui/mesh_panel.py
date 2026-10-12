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

**Keep apart** (Brandon, 2026-10-11) lands the box only on its own
element group's nodes, so a part meshed against another — a channel's
foot on the box it is bolted to — keeps nodes of its own where the two
touch, for Merge Coincident Nodes to join at the bolts. Shown once the
geometry holds something to be kept apart from.

**Solids** (2026-10-11) are typed in the same pane: a solid block is a
center, three widths and three angles, a cylinder a center, a radius, a
length and three angles — its axis the frame's Z — and either a name.
No element size and no element group: a solid is a shape, meshed
later (`core.solids`). The same gizmo turns and slides it.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
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
    'solid_block': {'center': (0.5, 0.5, 0.5), 'widths': (1.0, 1.0, 1.0)},
    'solid_cylinder': {'center': (0.0, 0.0, 0.5), 'widths': (0.5, 0.5, 1.0)},
    'solid_part': {'center': (0.0, 0.0, 0.0), 'widths': (1.0, 1.0, 1.0)},
}
#: each kind's title
TITLES = {'block': 'Add Block', 'plane': 'Add Plane',
          'solid_block': 'Add Solid Block', 'solid_cylinder': 'Add Cylinder'}
#: and the title when a solid already made is opened to change it (its
#: row's pencil, 2026-10-11): a boolean has no size of its own, so its
#: pane moves and turns it whole
EDIT_TITLES = {'solid_block': 'Edit Solid Block', 'solid_cylinder': 'Edit Cylinder',
               'solid_part': 'Move Part'}
#: the kinds that are solids, typed without a mesh
SOLID_KINDS = ('solid_block', 'solid_cylinder', 'solid_part')
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
        self.group_label: QLabel = QLabel('Element group')
        grid.addWidget(self.group_label, 1, 0)
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
        self.width_axes: list[QLabel] = []
        for row, axis in enumerate('XYZ', start=7):
            box = _box(0.0, minimum=0.0)
            box.setToolTip(f'The width along the frame\'s {axis}; a plate '
                           'leaves one width at zero, which names its plane')
            label = QLabel(axis)
            grid.addWidget(label, row, 0)
            grid.addWidget(box, row, 1)
            self.widths.append(box)
            self.width_axes.append(label)
        # a cylinder's two numbers, in the widths' rows: one set or the
        # other is shown, never both
        self.radius_label: QLabel = QLabel('Radius')
        self.radius: DoubleSpinBox = _box(0.25, minimum=0.0)
        self.radius.setToolTip("Round the frame's Z axis")
        self.length_label: QLabel = QLabel('Length')
        self.length: DoubleSpinBox = _box(1.0, minimum=0.0)
        self.length.setToolTip("Along the frame's Z axis, centered")
        grid.addWidget(self.radius_label, 7, 0)
        grid.addWidget(self.radius, 7, 1)
        grid.addWidget(self.length_label, 8, 0)
        grid.addWidget(self.length, 8, 1)
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
        # a solid block's corner radius, in the size's row: a solid has
        # no element size, and only a block has corners
        self.corner_label: QLabel = QLabel('Corner radius')
        self.corner: DoubleSpinBox = _box(0.0, minimum=0.0)
        self.corner.setToolTip("Round the block's four edges along its own "
                               'Z: a window cut with them leaves fillets')
        grid.addWidget(self.corner_label, 14, 0)
        grid.addWidget(self.corner, 14, 1)
        self.square_button: QPushButton = QPushButton('Square Up')
        self.square_button.setToolTip(
            "Turn the box back to the geometry's axes, keeping its center")
        grid.addWidget(self.square_button, 15, 0, 1, 2)
        self.apart: QCheckBox = QCheckBox('Keep apart from other groups')
        self.apart.setToolTip(
            'Join only nodes of its own element group: where it touches '
            'another part it keeps nodes of its own, to be merged at the '
            'places picked with Merge Coincident Nodes')
        grid.addWidget(self.apart, 16, 0, 1, 2)
        self.reading_label: QLabel = QLabel()
        self.reading_label.setWordWrap(True)
        grid.addWidget(self.reading_label, 17, 0, 1, 2)
        buttons = QHBoxLayout()
        self.add_button: QPushButton = QPushButton('Add')
        self.add_button.setDefault(True)
        self.close_button: QPushButton = QPushButton('Close')
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.close_button)
        grid.addLayout(buttons, 18, 0, 1, 2)
        grid.setRowStretch(19, 1)
        commit_on_enter(*self.center, *self.widths, *self.angles, self.size,
                        self.radius, self.length, self.corner)
        for box in (*self.center, *self.widths, *self.angles, self.size,
                    self.radius, self.length, self.corner):
            box.valueChanged.connect(lambda _value: self.changed.emit())
        self.group.currentTextChanged.connect(lambda _text: self.changed.emit())
        self.apart.toggled.connect(lambda _on: self.changed.emit())
        self.add_button.clicked.connect(self.add_asked.emit)
        self.close_button.clicked.connect(self.close_asked.emit)
        self.square_button.clicked.connect(self.square_asked.emit)

    def open_for(self, kind: str, unit: str, groups: Sequence[str],
                 default: str | None = None, editing: bool = False) -> None:
        """Set the pane up for `kind` — a block or a plane of elements, or
        a solid block or cylinder — in the display unit `unit`, with the
        geometry's element groups to choose from, opening on `default`;
        the fields open on the defaults. Only the fields the kind takes
        are shown. `editing` opens it on a solid already made, to change
        it ('solid_part' a boolean, moved and turned whole): Apply, not
        Add."""
        self.kind = kind
        solid = kind in SOLID_KINDS
        cylinder = kind == 'solid_cylinder'
        part = kind == 'solid_part'
        self.title.setText((EDIT_TITLES if editing else TITLES)[kind])
        self.add_button.setText('Apply' if editing else 'Add')
        self.center_label.setText(f'Center [{unit}]')
        self.width_label.setText(f'Width [{unit}]')
        self.radius_label.setText(f'Radius [{unit}]')
        self.length_label.setText(f'Length [{unit}]')
        self.size_label.setText(f'Element size [{unit}]')
        self.group_label.setText('Name' if solid else 'Element group')
        for widget in (self.width_label, *self.widths, *self.width_axes):
            widget.setVisible(not cylinder and not part)
        for widget in (self.radius_label, self.radius, self.length_label,
                       self.length):
            widget.setVisible(cylinder)
        for widget in (self.size_label, self.size):
            widget.setVisible(not solid)
        for widget in (self.corner_label, self.corner):
            widget.setVisible(kind == 'solid_block')
        self.corner_label.setText(f'Corner radius [{unit}]')
        if solid:
            self.apart.setVisible(False)
        names = [] if solid else [name for name in groups if name]
        self.group.blockSignals(True)
        self.group.clear()
        self.group.addItems(names)
        self.group.setCurrentText('' if solid else
                                  default or ('block' if kind == 'block'
                                              else 'plate'))
        self.group.blockSignals(False)
        defaults = DEFAULTS[kind]
        self.set_values(center=defaults['center'], widths=defaults['widths'],
                        angles=(0.0, 0.0, 0.0), size=DEFAULT_SIZE,
                        radius=0.25, length=1.0, corner=0.0, apart=False,
                        quiet=True)

    def values(self) -> dict:
        """The fields as typed, in the display unit."""
        # a cylinder's widths are its box, two diameters and its length:
        # what the gizmo sizes itself by
        widths = (tuple(box.value() for box in self.widths)
                  if self.kind != 'solid_cylinder' else
                  (2 * self.radius.value(), 2 * self.radius.value(),
                   self.length.value()))
        return {'center': tuple(box.value() for box in self.center),
                'widths': widths,
                'radius': self.radius.value(), 'length': self.length.value(),
                'corner': self.corner.value(),
                'angles': tuple(box.value() for box in self.angles),
                'size': self.size.value(),
                'group': self.group.currentText().strip(),
                'apart': self.apart.isChecked()}

    def set_values(self, quiet: bool = False, **values) -> None:
        """Fill fields by name — center=(x, y, z), widths=(x, y, z),
        angles=(x, y, z) in degrees, size=..., radius=..., length=...,
        corner=..., group=..., apart=...; how a test
        types, and how the gizmo writes the center and the angles back.
        `quiet` fills them without a `changed`."""
        boxes = [*self.center, *self.widths, *self.angles, self.size,
                 self.radius, self.length, self.corner]
        if quiet:
            for box in boxes:
                box.blockSignals(True)
            self.group.blockSignals(True)
            self.apart.blockSignals(True)
        try:
            for key, value in values.items():
                if key == 'group':
                    self.group.setCurrentText(value)
                elif key == 'apart':
                    self.apart.setChecked(bool(value))
                elif key in ('size', 'radius', 'length', 'corner'):
                    getattr(self, key).setValue(value)
                else:
                    for box, number in zip(getattr(self, key), value):
                        box.setValue(number)
        finally:
            if quiet:
                for box in boxes:
                    box.blockSignals(False)
                self.group.blockSignals(False)
                self.apart.blockSignals(False)
        if not quiet:
            self.changed.emit()

    def show_reading(self, text: str, ok: bool) -> None:
        self.reading_label.setText(text)
        self.add_button.setEnabled(ok)
