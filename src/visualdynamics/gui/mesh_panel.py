"""The pane beside the 3-D view where Add Block, Add Plane and the
solids are typed.

A box is described by its center and its width along each axis, and its
preview in the view is turned and slid by the same gizmo that turns a
coordinate system. So the pane holds a center, three widths and an
element size in the display unit, and the window holds the frame the box
is drawn in: the rings turn it about its center, and the arrows slide
the center along the frame's own axes onto the grid. The rotation is
also three fields, degrees about the geometry's X, Y and Z applied in
that order, for an angle the rings' whole degrees do not reach. The
rings write the angles back, so the fields and the view always agree.
Square Up turns the box back to the geometry's axes and keeps its
center.

A block takes three widths. A plate lies in a plane, and the plane is
named by the one width left at zero (a width of zero along Z is a plate
in the frame's X-Y plane), so a plate and a block are typed the same
way.

The pane holds no rule of its own. The window turns the fields and its
frame into the corner and edges the project's `add_block` and
`add_plane` take, asks the core what they would make, and draws it. Add
runs the project verb, so the journal line is the call a script makes.

Keep apart from other groups joins the box only to its own element
group's nodes. A part meshed against another (a channel's foot on the
box it is bolted to) then keeps nodes of its own where the two touch,
for Merge Coincident Nodes to join at the bolts. It is shown once the
geometry has nodes.

Solids are typed in the same pane and turned and slid by the same
gizmo. A solid has a name in place of the element group and no element
size, because it is a shape that is meshed later (`core.solids`).

- A solid block is a center, three widths, three angles and a corner
  radius, which rounds its four edges along its own Z.
- A cylinder is a center, a radius, a length along the frame's Z, and
  three angles.
- A section is a shape (I-beam, channel, tee, angle, rectangular tube or
  round tube), its dimensions, named as `core.solids.SECTIONS` names
  them, and a length along the frame's Z.
- A hole is a diameter, a depth or Through, and a counterbore's diameter
  and depth when one is asked for. Its center is where it enters, and it
  runs down the frame's Z.

A solid already made can be opened in the pane to change it, with Apply
in place of Add. A boolean has no size of its own, so its pane (Move
Part) only moves and turns it.
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

from ..core.solids import HOLE_DIMENSIONS, SECTIONS
from .editors import LENGTH_DECIMALS, DoubleSpinBox, commit_on_enter
from .settings_panel import panel_grid

# The pop-up windows this pane replaced (`plane_dialog`, 2026-09-26 and
# 2026-09-30) took a corner and two or three edges. Brandon wanted no
# window at all, the box described by its center and widths, and the
# preview turned and slid by the coordinate-system gizmo, with the
# rotation as fields too (Brandon, 2026-10-02). Keep apart: Brandon,
# 2026-10-11. Solids: 2026-10-11. Sections and holes: 2026-10-12.

#: what the fields open on, in the display unit: a box one unit on a side
#: in X and Y and a tenth deep, sitting on the origin's corner, a tenth
#: per element — the old dialog's defaults, said as a center. A solid
#: opens centered on the origin (Brandon, 2026-10-12): it is a shape to
#: be placed, and its own middle is the point it is placed by
DEFAULTS = {
    'block': {'center': (0.5, 0.5, 0.05), 'widths': (1.0, 1.0, 0.1)},
    'plane': {'center': (0.5, 0.5, 0.0), 'widths': (1.0, 1.0, 0.0)},
    'solid_block': {'center': (0.0, 0.0, 0.0), 'widths': (1.0, 1.0, 1.0)},
    'solid_cylinder': {'center': (0.0, 0.0, 0.0), 'widths': (0.5, 0.5, 1.0)},
    'solid_part': {'center': (0.0, 0.0, 0.0), 'widths': (1.0, 1.0, 1.0)},
    'solid_section': {'center': (0.0, 0.0, 0.0), 'widths': (0.5, 1.0, 4.0)},
    'solid_hole': {'center': (0.0, 0.0, 0.0), 'widths': (0.25, 0.25, 0.5)},
    'solid_cone': {'center': (0.0, 0.0, 0.0), 'widths': (1.0, 1.0, 1.0)},
    'solid_sphere': {'center': (0.0, 0.0, 0.0), 'widths': (1.0, 1.0, 1.0)},
}
#: a section's dimensions as it opens, in the display unit, in the order
#: `SECTIONS` names them: a foot deep in inches, a beam's proportions
SECTION_DEFAULTS = {'I-beam': (1.0, 0.5, 0.1, 0.1), 'channel': (1.0, 0.5, 0.1, 0.1),
                    'tee': (1.0, 1.0, 0.1, 0.1), 'angle': (1.0, 1.0, 0.1),
                    'rectangular tube': (1.0, 0.5, 0.1), 'round tube': (1.0, 0.1)}
SECTION_LENGTH = 4.0
#: a hole's: a quarter across, through, a counterbore twice as wide an
#: eighth deep when one is asked for
HOLE_DEFAULTS = (0.25, 0.5, 0.5, 0.125)
#: each kind's title
TITLES = {'block': 'Add Block', 'plane': 'Add Plane',
          'solid_block': 'Add Solid Block', 'solid_cylinder': 'Add Cylinder',
          'solid_section': 'Add Section', 'solid_hole': 'Add Hole',
          'solid_cone': 'Add Cone', 'solid_sphere': 'Add Sphere'}
#: and the title when a solid already made is opened to change it (its
#: row's pencil, 2026-10-11): a boolean has no size of its own, so its
#: pane moves and turns it whole
EDIT_TITLES = {'solid_block': 'Edit Solid Block', 'solid_cylinder': 'Edit Cylinder',
               'solid_part': 'Move Part', 'solid_section': 'Edit Section',
               'solid_hole': 'Edit Hole', 'solid_cone': 'Edit Cone',
               'solid_sphere': 'Edit Sphere'}
#: the kinds that are solids, typed without a mesh
SOLID_KINDS = ('solid_block', 'solid_cylinder', 'solid_part', 'solid_section',
               'solid_hole', 'solid_cone', 'solid_sphere')
#: a cone as it opens: a radius of half at its base, a quarter at its top
CONE_TOP = 0.25
DEFAULT_SIZE = 0.1


def _box(value: float, minimum: float = -1e9) -> DoubleSpinBox:
    box = DoubleSpinBox()
    box.setDecimals(LENGTH_DECIMALS)
    box.setRange(minimum, 1e9)
    box.setValue(value)
    return box


class MeshPanel(QWidget):
    """The fields for one block, plate or solid, and its Add (or Apply)
    button.

    `changed` is emitted when a field is edited; the window then reads
    `values` and redraws the preview. `add_asked`, `close_asked` and
    `square_asked` are the Add, Close and Square Up buttons.
    """

    #: a field was edited: the window reads `values()` and redraws
    changed = Signal()
    add_asked = Signal()
    close_asked = Signal()
    #: square the preview's frame up with the geometry's axes
    square_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind: str = 'block'
        self.editing: bool = False
        self.unit: str = ''
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
        # the length in the last of the rows a section's dimensions take,
        # so it is a cylinder's second number and a section's last
        grid.addWidget(self.length_label, 12, 0)
        grid.addWidget(self.length, 12, 1)
        # a cone's radius at its top, in a row only a section's or a
        # hole's other numbers take, never shown with them (2026-10-20)
        self.top_label: QLabel = QLabel('Top radius')
        self.top: DoubleSpinBox = _box(CONE_TOP, minimum=0.0)
        self.top.setToolTip("At the frame's +Z end; zero for a point")
        grid.addWidget(self.top_label, 8, 0)
        grid.addWidget(self.top, 8, 1)
        # a section: its shape in the width label's row, its dimensions
        # in the widths' rows and below
        self.shape_label: QLabel = QLabel('Shape')
        self.shape: QComboBox = QComboBox()
        self.shape.addItems(list(SECTIONS))
        self.shape.setToolTip('The section run along the frame\'s Z')
        grid.addWidget(self.shape_label, 6, 0)
        grid.addWidget(self.shape, 6, 1)
        self.section_labels: list[QLabel] = []
        self.section: list[DoubleSpinBox] = []
        for row in range(7, 11):
            label, box = QLabel(), _box(0.0, minimum=0.0)
            # named for its dimension once a shape is picked (_shape_rows)
            box.setToolTip('A dimension of the section, in the display unit')
            grid.addWidget(label, row, 0)
            grid.addWidget(box, row, 1)
            self.section_labels.append(label)
            self.section.append(box)
        # a hole: its diameter, Through or a depth, and a counterbore
        # when one is asked for
        self.hole_labels: list[QLabel] = []
        self.hole: list[DoubleSpinBox] = []
        for row in (7, 9, 11, 12):
            label, box = QLabel(), _box(0.0, minimum=0.0)
            grid.addWidget(label, row, 0)
            grid.addWidget(box, row, 1)
            self.hole_labels.append(label)
            self.hole.append(box)
        self.hole[0].setToolTip('Across the hole')
        self.hole[1].setToolTip('From where it enters, down the frame\'s Z')
        self.hole[2].setToolTip('Across the counterbore at the entry; '
                                'zero for none')
        self.hole[3].setToolTip('How deep the counterbore goes from the '
                                'entry')
        self.through: QCheckBox = QCheckBox('Through')
        self.through.setToolTip('Through the part it is cut from, however '
                                'thick that part is made')
        grid.addWidget(self.through, 8, 0, 1, 2)
        self.counterbore: QCheckBox = QCheckBox('Counterbore')
        self.counterbore.setToolTip('Wider at the entry, for a screw\'s head')
        grid.addWidget(self.counterbore, 10, 0, 1, 2)
        self.angle_label: QLabel = QLabel('Rotation [\u00b0]')
        self.angle_label.setEnabled(False)
        grid.addWidget(self.angle_label, 13, 0, 1, 2)
        self.angles: list[DoubleSpinBox] = []
        self.angle_axes: list[QLabel] = []
        for row, axis in enumerate('XYZ', start=14):
            box = _box(0.0, minimum=-360.0)
            box.setMaximum(360.0)
            box.setDecimals(2)
            box.setToolTip(f'Degrees about the geometry\'s {axis} axis, '
                           'applied X, then Y, then Z; the rings in the view '
                           'turn it too, a degree at a time')
            axis_label = QLabel(axis)
            self.angle_axes.append(axis_label)
            grid.addWidget(axis_label, row, 0)
            grid.addWidget(box, row, 1)
            self.angles.append(box)
        self.size_label: QLabel = QLabel('Element size')
        self.size: DoubleSpinBox = _box(DEFAULT_SIZE, minimum=0.0)
        self.size.setToolTip('Each edge is divided into the whole number of '
                             'elements nearest this size')
        grid.addWidget(self.size_label, 17, 0)
        grid.addWidget(self.size, 17, 1)
        # a solid block's corner radius, in the size's row: a solid has
        # no element size, and only a block has corners
        self.corner_label: QLabel = QLabel('Corner radius')
        self.corner: DoubleSpinBox = _box(0.0, minimum=0.0)
        self.corner.setToolTip("Round the block's four edges along its own "
                               'Z: a window cut with them leaves fillets')
        grid.addWidget(self.corner_label, 17, 0)
        grid.addWidget(self.corner, 17, 1)
        self.square_button: QPushButton = QPushButton('Square Up')
        self.square_button.setToolTip(
            "Turn the box back to the geometry's axes, keeping its center")
        grid.addWidget(self.square_button, 18, 0, 1, 2)
        self.apart: QCheckBox = QCheckBox('Keep apart from other groups')
        self.apart.setToolTip(
            'Join only nodes of its own element group: where it touches '
            'another part it keeps nodes of its own, to be merged at the '
            'places picked with Merge Coincident Nodes')
        grid.addWidget(self.apart, 19, 0, 1, 2)
        self.reading_label: QLabel = QLabel()
        self.reading_label.setWordWrap(True)
        grid.addWidget(self.reading_label, 20, 0, 1, 2)
        buttons = QHBoxLayout()
        self.add_button: QPushButton = QPushButton('Add')
        self.add_button.setToolTip('Add what the pane describes to the '
                                   'geometry, as the preview shows it')
        self.add_button.setDefault(True)
        self.close_button: QPushButton = QPushButton('Close')
        self.close_button.setToolTip('Put the pane away without adding')
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.close_button)
        grid.addLayout(buttons, 21, 0, 1, 2)
        grid.setRowStretch(22, 1)
        commit_on_enter(*self.center, *self.widths, *self.angles, self.size,
                        self.radius, self.length, self.corner, self.top, *self.section,
                        *self.hole)
        for box in (*self.center, *self.widths, *self.angles, self.size,
                    self.radius, self.length, self.corner, self.top, *self.section,
                    *self.hole):
            box.valueChanged.connect(lambda _value: self.changed.emit())
        self.shape.currentTextChanged.connect(self._shape_picked)
        self.through.toggled.connect(self._hole_toggled)
        self.counterbore.toggled.connect(self._hole_toggled)
        self.group.currentTextChanged.connect(lambda _text: self.changed.emit())
        self.apart.toggled.connect(lambda _on: self.changed.emit())
        self.add_button.clicked.connect(self.add_asked.emit)
        self.close_button.clicked.connect(self.close_asked.emit)
        self.square_button.clicked.connect(self.square_asked.emit)

    def open_for(self, kind: str, unit: str, groups: Sequence[str],
                 default: str | None = None, editing: bool = False) -> None:
        """Set the pane up for `kind`, with every field on its default.

        `kind` is 'block' or 'plane' for elements, or a solid:
        'solid_block', 'solid_cylinder', 'solid_cone', 'solid_sphere',
        'solid_section' or 'solid_hole',
        or 'solid_part' for a boolean, which is only edited. `unit` is the
        display unit the fields are in. `groups` are the geometry's
        element groups to choose from, and the field opens on `default`,
        or on 'block' or 'plate' when it is None; a solid has a name field
        instead, which opens empty. Only the fields the kind takes are
        shown. `editing` opens the pane on a solid already made, to change
        it, with Apply in place of Add; a section's shape cannot be
        changed then.
        """
        self.kind = kind
        self.editing = editing
        self.unit = unit
        solid = kind in SOLID_KINDS
        cylinder = kind == 'solid_cylinder'
        part = kind == 'solid_part'
        section, hole = kind == 'solid_section', kind == 'solid_hole'
        cone, sphere = kind == 'solid_cone', kind == 'solid_sphere'
        self.title.setText((EDIT_TITLES if editing else TITLES)[kind])
        self.add_button.setText('Apply' if editing else 'Add')
        self.center_label.setText(f'Center [{unit}]')
        self.width_label.setText(f'Width [{unit}]')
        self.radius_label.setText(f'{"Base radius" if cone else "Radius"} [{unit}]')
        self.length_label.setText(f'{"Height" if cone else "Length"} [{unit}]')
        self.top_label.setText(f'Top radius [{unit}]')
        self.size_label.setText(f'Element size [{unit}]')
        self.group_label.setText('Name' if solid else 'Element group')
        for widget in (self.width_label, *self.widths, *self.width_axes):
            widget.setVisible(not (cylinder or part or section or hole or cone or sphere))
        for widget in (self.radius_label, self.radius):
            widget.setVisible(cylinder or cone or sphere)
        for widget in (self.length_label, self.length):
            widget.setVisible(cylinder or section or cone)
        for widget in (self.top_label, self.top):
            widget.setVisible(cone)
        # a sphere turned is the same sphere (principle 3)
        for widget in (self.angle_label, *self.angle_axes, *self.angles):
            widget.setVisible(not sphere)
        # a section's shape is what it is: picked when it is added
        for widget in (self.shape_label, self.shape):
            widget.setVisible(section and not editing)
        for widget in (*self.section_labels, *self.section):
            widget.setVisible(False)
        for widget in (*self.hole_labels, *self.hole, self.through,
                       self.counterbore):
            widget.setVisible(False)
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
                        radius=0.5 if (cone or sphere) else 0.25,
                        length=SECTION_LENGTH if section else 1.0,
                        corner=0.0, top=CONE_TOP, apart=False, quiet=True)
        if section:
            self.set_values(shape='I-beam', quiet=True)
        if hole:
            for label, named in zip(self.hole_labels, HOLE_DIMENSIONS):
                label.setText(f'{named[:1].upper()}{named[1:]} [{unit}]')
            self.set_values(dimensions=(HOLE_DEFAULTS[0], 0.0, 0.0, 0.0), quiet=True)
            for box, value in zip(self.hole[1:], HOLE_DEFAULTS[1:]):
                box.blockSignals(True)
                box.setValue(value)
                box.blockSignals(False)

    def _shape_rows(self) -> None:
        """A section's rows, labeled for its shape."""
        named = SECTIONS[self.shape.currentText()]
        for k, (label, box) in enumerate(zip(self.section_labels, self.section)):
            shown = k < len(named)
            label.setVisible(shown)
            box.setVisible(shown)
            if shown:
                label.setText(f'{named[k][:1].upper()}{named[k][1:]} [{self.unit}]')
                box.setToolTip(f'The section\'s {named[k]}, in the display '
                               'unit')
        if self.editing:
            shape = self.shape.currentText()
            self.title.setText(f'Edit {shape[:1].upper()}{shape[1:]}')

    def _shape_picked(self, shape: str) -> None:
        """A shape picked: its rows, opened on its proportions."""
        if self.kind != 'solid_section' or shape not in SECTIONS:
            return
        self.set_values(shape=shape, quiet=True)
        self.changed.emit()

    def _hole_toggled(self, _on: bool) -> None:
        self._hole_rows()
        self.changed.emit()

    def _hole_rows(self) -> None:
        """A hole's rows: its depth unless it goes through, its
        counterbore's when it has one."""
        if self.kind != 'solid_hole':
            return
        shown = [True, not self.through.isChecked(), self.counterbore.isChecked(),
                 self.counterbore.isChecked()]
        for label, box, on in zip(self.hole_labels, self.hole, shown):
            label.setVisible(on)
            box.setVisible(on)
        self.through.setVisible(True)
        self.counterbore.setVisible(True)

    def values(self) -> dict:
        """The fields as typed, in the display unit, as a dict: 'center',
        'widths', 'shape', 'dimensions', 'radius', 'length', 'corner',
        'top', 'angles' in degrees, 'size', 'group' and 'apart'.

        A cone's 'radius' is at its base, 'top' at its top and 'length'
        its height. A section's 'dimensions' are its shape's, in the order
        `SECTIONS` gives them. A hole's are its diameter, its depth (zero when it
        goes through) and its counterbore's diameter and depth (zeros
        when it has none), as `edit_solid` takes them. For a cylinder, a
        section and a hole, 'widths' is the size of the box around the
        solid, which the gizmo is sized by.
        """
        shape = self.shape.currentText()
        section = tuple(box.value() for box in self.section[:len(SECTIONS[shape])])
        through, bored = self.through.isChecked(), self.counterbore.isChecked()
        hole = (self.hole[0].value(), 0.0 if through else self.hole[1].value(),
                self.hole[2].value() if bored else 0.0,
                self.hole[3].value() if bored else 0.0)
        # what the gizmo sizes itself by: a cylinder's box, two diameters
        # and its length, a section's, a hole's as wide as its widest
        if self.kind == 'solid_cylinder':
            widths = (2 * self.radius.value(), 2 * self.radius.value(),
                      self.length.value())
        elif self.kind == 'solid_cone':
            wide = 2 * max(self.radius.value(), self.top.value())
            widths = (wide, wide, self.length.value())
        elif self.kind == 'solid_sphere':
            widths = (2 * self.radius.value(),) * 3
        elif self.kind == 'solid_section':
            across = {'round tube': (section[0], section[0]),
                      'angle': section[:2], 'rectangular tube': section[:2]}.get(
                shape, (section[1], section[0]))
            widths = (*across, self.length.value())
        elif self.kind == 'solid_hole':
            wide = max(hole[0], hole[2])
            widths = (wide, wide, hole[1] or hole[0])
        else:
            widths = tuple(box.value() for box in self.widths)
        return {'center': tuple(box.value() for box in self.center),
                'widths': widths, 'shape': shape,
                'dimensions': hole if self.kind == 'solid_hole' else section,
                'radius': self.radius.value(), 'length': self.length.value(),
                'corner': self.corner.value(), 'top': self.top.value(),
                'angles': tuple(box.value() for box in self.angles),
                'size': self.size.value(),
                'group': self.group.currentText().strip(),
                'apart': self.apart.isChecked()}

    def set_values(self, quiet: bool = False, **values) -> None:
        """Fill fields by name, in the display unit: center=(x, y, z),
        widths=(x, y, z), angles=(x, y, z) in degrees, size, radius,
        length, corner, shape, dimensions, group and apart. A hole's
        dimensions are given as `values` gives them. Picking a shape
        opens its dimensions on that shape's defaults, except when
        editing.

        Tests type with this, and the gizmo writes the center and the
        angles back with it. With `quiet` the fields are filled without
        emitting `changed`.
        """
        boxes = [*self.center, *self.widths, *self.angles, self.size,
                 self.radius, self.length, self.corner, self.top, *self.section, *self.hole]
        if quiet:
            for box in boxes:
                box.blockSignals(True)
            self.group.blockSignals(True)
            self.apart.blockSignals(True)
        try:
            for key, value in values.items():
                if key == 'shape':
                    self.shape.blockSignals(True)
                    self.shape.setCurrentText(value)
                    self.shape.blockSignals(False)
                    if not self.editing:
                        for box, number in zip(self.section, SECTION_DEFAULTS[value]):
                            box.setValue(number)
                    self._shape_rows()
                elif key == 'dimensions' and self.kind == 'solid_hole':
                    diameter, depth, wide, deep = value
                    self.hole[0].setValue(diameter)
                    if depth:
                        self.hole[1].setValue(depth)
                    if wide:
                        self.hole[2].setValue(wide)
                        self.hole[3].setValue(deep)
                    for check, on in ((self.through, not depth),
                                      (self.counterbore, bool(wide))):
                        check.blockSignals(True)
                        check.setChecked(on)
                        check.blockSignals(False)
                    self._hole_rows()
                elif key == 'dimensions':
                    for box, number in zip(self.section, value):
                        box.setValue(number)
                elif key == 'group':
                    self.group.setCurrentText(value)
                elif key == 'apart':
                    self.apart.setChecked(bool(value))
                elif key in ('size', 'radius', 'length', 'corner', 'top'):
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
        """Show what Add would make, as the window reads it from the core,
        and enable Add only when `ok`."""
        self.reading_label.setText(text)
        self.add_button.setEnabled(ok)
