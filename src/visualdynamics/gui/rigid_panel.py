"""The rigid-body settings, as a table beside the 3-D view.

The reference point the rotations pivot on, and — when the set is to
be mass-normalized — the mass and the six terms of the inertia tensor
about it. The view previews the six shapes on the geometry as they
are set, and the button at the bottom makes the shape set: select,
see, set, apply (principle 13), the truncate panel's shape moved from
the plot to the scene.

Nothing here computes anything. The panel says what the settings are
in the display units, hands them over in SI as a `MassProperties`,
and reports when they move; `generate_rigid_body_modes` is what makes
the set, with whatever is set at that moment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from ..core.rigid import INERTIA_TERMS, MassProperties, axis_name
from .editors import DoubleSpinBox, commit_on_enter
from .settings_panel import add_derived, panel_grid

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.geometry import Geometry
    from ..units import UnitSystem

#: the dimension the inertia terms carry, in the units module's grammar
INERTIA_DIMENSION = 'mass*length**2'


class RigidBodyPanel(QWidget):
    """The parameter table. Edits arrive as a whole `MassProperties`."""

    #: a parameter moved; here are the settings that describe it now
    changed = Signal(object)

    #: the user asked for the shape set this panel describes
    apply_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        #: the geometry the settings describe, for counting DOFs and
        #: for knowing whether its coordinates mean anything yet
        self._geometry = None
        self._unit_system = None
        #: set while the panel is writing to its own editors, so that
        #: restating a value does not read as a fresh edit
        self._loading = False

        self.title: QLabel = QLabel('Rigid Body Modes')
        grid = panel_grid(self, self.title)

        point = QLabel('Reference point')
        point.setEnabled(False)
        grid.addWidget(point, 1, 0, 1, 2)
        self.point_boxes: list[DoubleSpinBox] = []
        for row, axis in enumerate(('X', 'Y', 'Z'), start=2):
            box = DoubleSpinBox()
            box.setDecimals(4)
            box.setRange(-1e9, 1e9)
            box.setToolTip(
                f'{axis} of the point the rotations pivot on — the CG, '
                'or wherever the virtual point is wanted. Starts at the '
                'centroid of the nodes')
            grid.addWidget(QLabel(axis), row, 0)
            grid.addWidget(box, row, 1)
            self.point_boxes.append(box)

        # mass and inertia are one optional group: both or neither
        # (`core.rigid` says why), so one checkbox reveals them. Named
        # for what it reveals, not for the result: 'Mass-normalize' did
        # not read as the place to enter a mass (Brandon, 2026-09-04)
        self.scale_check: QCheckBox = QCheckBox('Mass properties')
        self.scale_check.setToolTip(
            'Scale the shapes by the mass and the inertia about the '
            'point, so the set is mass-normalized like a solved one. '
            'Off, the shapes are unit translations and unit rotations')
        grid.addWidget(self.scale_check, 5, 0, 1, 2)
        self.scale_note: QLabel = QLabel(
            'Declare the length unit to mass-normalize')
        self.scale_note.setWordWrap(True)
        self.scale_note.setEnabled(False)
        grid.addWidget(self.scale_note, 6, 0, 1, 2)

        self.scale_panel: QWidget = QWidget()
        scale = QGridLayout(self.scale_panel)
        scale.setContentsMargins(0, 0, 0, 0)
        scale.setHorizontalSpacing(8)
        scale.setVerticalSpacing(4)
        self.mass_box: DoubleSpinBox = DoubleSpinBox()
        self.mass_box.setDecimals(4)
        self.mass_box.setRange(1e-9, 1e12)
        self.mass_box.setToolTip('The mass of the body')
        scale.addWidget(QLabel('Mass'), 0, 0)
        scale.addWidget(self.mass_box, 0, 1)
        self.inertia_boxes: dict[str, DoubleSpinBox] = {}
        for row, term in enumerate(INERTIA_TERMS, start=1):
            box = DoubleSpinBox()
            box.setDecimals(4)
            box.setRange(-1e12, 1e12)
            box.setToolTip(
                f'{term} of the inertia tensor about the point, as a '
                'CAD mass-properties report states it')
            scale.addWidget(QLabel(term), row, 0)
            scale.addWidget(box, row, 1)
            self.inertia_boxes[term] = box
        grid.addWidget(self.scale_panel, 7, 0, 1, 2)
        self.scale_panel.hide()

        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setFrameShadow(QFrame.Shadow.Sunken)
        grid.addWidget(rule, 8, 0, 1, 2)

        # what follows from the settings, shown so the set is known
        # before it is made
        self.derived: dict[str, QLabel] = {}
        derived = (('modes', 'Modes'), ('dofs', 'DOFs'), ('axes', 'Axes'),
                   ('scaling', 'Scaling'))
        for key, (_name, value) in add_derived(grid, derived,
                                               9).items():
            self.derived[key] = value

        # the one refusal the boxes' ranges cannot state at entry: a
        # tensor no body could have, which only the six together say
        self.problem: QLabel = QLabel('')
        self.problem.setWordWrap(True)
        grid.addWidget(self.problem, 13, 0, 1, 2)

        # the act itself, where its settings are set
        self.apply_button: QPushButton = QPushButton(
            'Generate Rigid Body Mode Shapes')
        self.apply_button.setToolTip(
            'Make the six rigid-body shapes a shape set in the '
            "geometry's group")
        self.apply_button.clicked.connect(self.apply_asked.emit)
        grid.addWidget(self.apply_button, 14, 0, 1, 2)
        grid.setRowStretch(15, 1)

        boxes = (*self.point_boxes, self.mass_box, *self.inertia_boxes.values())
        commit_on_enter(*boxes)
        for box in boxes:
            box.valueChanged.connect(self._edited)
        self.scale_check.toggled.connect(self._scale_toggled)

    # ---- what the panel is describing -------------------------------------

    def show_geometry(self, geometry: Geometry, properties: MassProperties,
                      unit_system: UnitSystem) -> None:
        """Point the panel at a geometry and the settings on it, in the
        display units of `unit_system`.

        A geometry whose length unit is undeclared shows its raw
        coordinates with no unit, and cannot be mass-normalized — an
        inertia in kg·m² against coordinates in nothing is not a number
        — so the checkbox gives way to a note saying what to declare.
        """
        self._geometry = geometry
        self._unit_system = unit_system
        self._loading = True
        try:
            defined = geometry.units_defined
            length = f' {unit_system.label_text("length")}' if defined else ''
            for box in self.point_boxes:
                box.setSuffix(length)
            self.mass_box.setSuffix(f' {unit_system.label_text("mass")}')
            inertia = unit_system.label_text(INERTIA_DIMENSION)
            for box in self.inertia_boxes.values():
                box.setSuffix(f' {inertia}')
            self.scale_check.setVisible(defined)
            self.scale_note.setVisible(not defined)
        finally:
            self._loading = False
        self.set_properties(properties)

    def set_properties(self, properties: MassProperties) -> None:
        """Restate the panel from the settings, without re-emitting."""
        self._loading = True
        try:
            for box, value in zip(self.point_boxes,
                                  self._display(properties.point, 'length')):
                box.setValue(float(value))
            self.scale_check.setChecked(properties.scaled)
            self.scale_panel.setVisible(properties.scaled)
            if properties.scaled:
                self.mass_box.setValue(
                    float(self._display(properties.mass, 'mass')))
                terms = self._display(properties.inertia, INERTIA_DIMENSION)
                for box, value in zip(self.inertia_boxes.values(), terms):
                    box.setValue(float(value))
        finally:
            self._loading = False
        self._restate()

    def properties(self) -> MassProperties:
        """The settings the editors currently describe, in SI.

        Raises `ValueError` for a tensor no body could have — the one
        state the boxes cannot refuse one at a time.
        """
        point = self._si([box.value() for box in self.point_boxes], 'length')
        if not (self.scale_check.isVisible() and self.scale_check.isChecked()):
            return MassProperties(tuple(point))
        return MassProperties(
            tuple(point), mass=float(self._si(self.mass_box.value(), 'mass')),
            inertia=tuple(self._si([box.value() for box in
                                    self.inertia_boxes.values()],
                                   INERTIA_DIMENSION)))

    # ---- units ------------------------------------------------------------

    def _defined(self) -> bool:
        return (self._geometry is not None and self._geometry.units_defined
                and self._unit_system is not None)

    def _display(self, values, dimension):
        """SI to display units — or the raw numbers, for a geometry
        whose coordinates mean nothing yet."""
        import numpy as np

        values = np.asarray(values, dtype=float)
        if dimension == 'length' and not self._defined():
            return values
        return self._unit_system.from_si(values, dimension)

    def _si(self, values, dimension):
        import numpy as np

        values = np.asarray(values, dtype=float)
        if dimension == 'length' and not self._defined():
            return values
        return self._unit_system.to_si(values, dimension)

    # ---- editing ----------------------------------------------------------

    def _scale_toggled(self, checked: bool) -> None:
        """Reveal the mass and inertia boxes, seeded with a body that
        exists — unit mass, unit moments, no products — the first time,
        so the set is valid the moment the box is ticked."""
        self.scale_panel.setVisible(bool(checked))
        if checked and self.mass_box.value() == self.mass_box.minimum():
            self._loading = True
            try:
                self.mass_box.setValue(1.0)
                for term, box in self.inertia_boxes.items():
                    box.setValue(1.0 if term in ('Ixx', 'Iyy', 'Izz')
                                 else 0.0)
            finally:
                self._loading = False
        self._edited()

    def _edited(self, *_args):
        if self._loading:
            return
        settled = self._restate()
        if settled is not None:
            self.changed.emit(settled)

    # ---- what follows from the settings ----------------------------------

    def _restate(self):
        """Read the set's size and axes off the settings; say what is
        wrong with them instead when they refuse. Returns the settings,
        or None when they are not a body."""
        try:
            properties = self.properties()
        except ValueError as refusal:
            self.problem.setText(str(refusal))
            self.derived['axes'].setText('—')
            self.apply_button.setEnabled(False)
            return None
        self.problem.setText('')
        self.derived['modes'].setText('6')
        nodes = self._geometry.num_nodes if self._geometry is not None else 0
        self.derived['dofs'].setText(f'3 × {nodes}')
        _moments, axes = properties.principal_axes()
        self.derived['axes'].setText(
            ', '.join(axis_name(axis) for axis in axes))
        self.derived['scaling'].setText(
            'mass-normalized' if properties.scaled else 'unit shapes')
        self.apply_button.setEnabled(nodes > 0)
        return properties
