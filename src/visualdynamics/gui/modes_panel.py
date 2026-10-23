"""The Modes pane beside the 3-D view: a geometry's mode shapes, rigid
body or finite element.

The kind is chosen at the top. Kinematic is offered for every
geometry; finite element only once its element groups have properties
to build a model from, and with only one kind there is no choice shown.
Both kinds are named for how the shapes are made: a finite element
solve returns rigid-body modes too, at 0 Hz, so 'Rigid body' did not
tell the two apart.

Kinematic: the six rigid-body shapes from the geometry alone. The
settings are the reference point the rotations are
about and, when the shape set is to be mass-normalized, the mass and
the six terms of the inertia tensor about that point. The view previews
the six shapes on the geometry as they are set, and the button at the
bottom makes the shape set (`generate_rigid_body_modes`).

Finite element: the settings are the highest frequency to solve to and
the damping to give every mode. A solve is too slow to preview as the
numbers change, so the readings say what will be solved (the nodes,
elements, element groups and DOFs) and refuse a group with no
properties before the solve does; the button solves (`solve_modes`).

This is select, see, set, apply (principle 13), laid out as in the
truncate panel but with the preview in the 3-D view instead of on a
plot. The panel makes nothing itself: it shows the settings in the
display units, emits the rigid ones in SI as a `MassProperties` when
they change, and the window runs the verb when a button is pressed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from ..core.rigid import INERTIA_TERMS, MassProperties, axis_name
from .editors import LENGTH_DECIMALS, DoubleSpinBox, commit_on_enter
from .settings_panel import add_derived, add_rule, add_settings, panel_grid

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.geometry import Geometry
    from ..units import UnitSystem

#: the dimension the inertia terms carry, in the units module's grammar
INERTIA_DIMENSION = 'mass*length**2'

#: the solve's settings as the pane opens: what Solve Modes asked for
#: by default when it asked in two dialogs
DEFAULT_MAXIMUM_FREQUENCY = 2000.0
DEFAULT_DAMPING_PERCENT = 2.0


def _section_grid(section: QWidget) -> QGridLayout:
    """A grid for one kind's rows, flush with the pane's own."""
    grid = QGridLayout(section)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(4)
    return grid


class ModesPanel(QWidget):
    """The settings table. Rigid-body edits are emitted as a whole
    `MassProperties`; the solve's two numbers are read when it runs."""

    #: a rigid-body parameter moved; here are the settings now
    changed = Signal(object)

    #: the user asked for the rigid-body shape set this panel describes
    apply_asked = Signal()

    #: the user asked for the finite element solve this panel describes
    solve_asked = Signal()

    #: the kind shown changed: 'rigid' or 'fem'
    kind_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        #: the geometry the settings describe, for counting DOFs and
        #: for knowing whether its coordinates mean anything yet
        self._geometry = None
        self._unit_system = None
        #: set while the panel is writing to its own editors, so that
        #: restating a value does not read as a fresh edit
        self._loading = False

        # One pane for both kinds of modes (Brandon, 2026-10-22): the
        # rigid-body reading and Solve Modes were two acts on one
        # geometry, the solve's settings in two dialogs.
        self.title: QLabel = QLabel('Modes')
        panel = panel_grid(self, self.title)
        self._fem_offered = False
        self.kind_label: QLabel = QLabel('Kind')
        self.kind_box: QComboBox = QComboBox()
        # 'Kinematic' and not 'Rigid body' (Brandon, 2026-10-22): the
        # solve's six 0 Hz modes are rigid-body modes as well
        self.kind_box.addItem('Kinematic', 'rigid')
        self.kind_box.addItem('Finite element', 'fem')
        # a model whose groups have properties is there to be solved
        self.kind_box.setCurrentIndex(1)
        self.kind_box.setToolTip(
            'Kinematic: the six rigid-body shapes about a point, from the '
            'geometry alone. Finite element: the normal modes solved from '
            "the element groups' properties")
        self.kind_box.currentIndexChanged.connect(self._kind_chosen)
        panel.addWidget(self.kind_label, 1, 0)
        panel.addWidget(self.kind_box, 1, 1)
        self.rigid_section: QWidget = QWidget()
        self.fem_section: QWidget = self._build_fem_section()
        panel.addWidget(self.rigid_section, 2, 0, 1, 2)
        panel.addWidget(self.fem_section, 3, 0, 1, 2)
        panel.setRowStretch(4, 1)
        grid = _section_grid(self.rigid_section)

        point = QLabel('Reference point')
        point.setEnabled(False)
        grid.addWidget(point, 1, 0, 1, 2)
        self.point_boxes: list[DoubleSpinBox] = []
        for row, axis in enumerate(('X', 'Y', 'Z'), start=2):
            box = DoubleSpinBox()
            box.setDecimals(LENGTH_DECIMALS)
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

        add_rule(grid, 8)

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

        boxes = (*self.point_boxes, self.mass_box, *self.inertia_boxes.values())
        commit_on_enter(*boxes)
        for box in boxes:
            box.valueChanged.connect(self._edited)
        self.scale_check.toggled.connect(self._scale_toggled)
        self.offer_kinds(False)

    def _build_fem_section(self) -> QWidget:
        """The finite element rows: the two numbers the solve takes, what
        will be solved, and the button."""
        section = QWidget()
        grid = _section_grid(section)
        self.top_box: DoubleSpinBox = DoubleSpinBox()
        self.top_box.setDecimals(1)
        self.top_box.setRange(0.1, 1e9)
        self.top_box.setValue(DEFAULT_MAXIMUM_FREQUENCY)
        self.top_box.setSuffix(' Hz')
        self.top_box.setToolTip('Solve for every mode up to this frequency')
        self.damping_box: DoubleSpinBox = DoubleSpinBox()
        self.damping_box.setDecimals(2)
        self.damping_box.setRange(0.0, 100.0)
        self.damping_box.setValue(DEFAULT_DAMPING_PERCENT)
        self.damping_box.setSuffix(' %')
        self.damping_box.setToolTip(
            'The damping to give every mode, in percent of critical. A '
            'finite element model has none of its own')
        add_settings(grid, (('Highest frequency', self.top_box),
                            ('Damping', self.damping_box)), 0)
        commit_on_enter(self.top_box, self.damping_box)
        add_rule(grid, 2)
        self.fem_derived: dict[str, QLabel] = {}
        derived = (('nodes', 'Nodes'), ('elements', 'Elements'),
                   ('groups', 'Element groups'), ('dofs', 'DOFs'))
        for key, (_name, value) in add_derived(grid, derived, 3).items():
            self.fem_derived[key] = value
        self.fem_problem: QLabel = QLabel('')
        self.fem_problem.setWordWrap(True)
        grid.addWidget(self.fem_problem, 7, 0, 1, 2)
        self.solve_button: QPushButton = QPushButton('Solve Modes')
        self.solve_button.setToolTip(
            "Solve the normal modes to the highest frequency and make them "
            "a shape set in the geometry's group")
        self.solve_button.clicked.connect(self.solve_asked.emit)
        grid.addWidget(self.solve_button, 8, 0, 1, 2)
        return section

    # ---- which kind -------------------------------------------------------

    def offer_kinds(self, fem: bool) -> None:
        """Offer the finite element kind beside the kinematic one, or
        not. With kinematic the only kind, no choice is shown."""
        self._fem_offered = bool(fem)
        self.kind_label.setVisible(self._fem_offered)
        self.kind_box.setVisible(self._fem_offered)
        self._show_kind()

    @property
    def kind(self) -> str:
        """'fem' when the finite element kind is offered and chosen,
        else 'rigid'."""
        return self.kind_box.currentData() if self._fem_offered else 'rigid'

    def _kind_chosen(self, _index) -> None:
        self._show_kind()
        self.kind_changed.emit(self.kind)

    def _show_kind(self) -> None:
        self.rigid_section.setVisible(self.kind == 'rigid')
        self.fem_section.setVisible(self.kind == 'fem')

    # ---- the finite element kind ------------------------------------------

    def show_model(self, geometry: Geometry) -> None:
        """Say what a solve of `geometry` would build: its nodes,
        elements, element groups and DOFs. An element group holding
        elements but no properties is named, and the solve refused until
        it has them, as the model build would refuse it."""
        self._geometry = geometry
        groups = sorted({int(group) for group in geometry.elem_group})
        missing = [group for group in groups
                   if group not in geometry.group_properties]
        nodes = geometry.num_nodes
        self.fem_derived['nodes'].setText(f'{nodes:,}')
        self.fem_derived['elements'].setText(f'{len(geometry.elem_conn):,}')
        self.fem_derived['groups'].setText(
            f'{len(groups) - len(missing)} of {len(groups)} with properties'
            if missing else f'{len(groups)}')
        self.fem_derived['dofs'].setText(f'6 × {nodes:,}')
        if missing:
            ids = ', '.join(str(group) for group in missing)
            plural = len(missing) > 1
            self.fem_problem.setText(
                f'Element group{"s" if plural else ""} {ids} '
                f'{"have" if plural else "has"} no properties: give '
                f'{"each" if plural else "it"} a material in the Element '
                'Groups table to solve')
        else:
            self.fem_problem.setText('')
        self.solve_button.setEnabled(not missing and nodes > 0)

    def maximum_frequency(self) -> float:
        """The highest frequency to solve to, in Hz."""
        return float(self.top_box.value())

    def damping(self) -> float:
        """The damping to give every mode, as a fraction of critical."""
        return float(self.damping_box.value()) / 100.0

    # ---- what the panel is describing -------------------------------------

    def show_geometry(self, geometry: Geometry, properties: MassProperties,
                      unit_system: UnitSystem) -> None:
        """Point the panel at a geometry and the settings on it, in the
        display units of `unit_system`.

        A geometry whose length unit is not declared shows its raw
        coordinates with no unit and cannot be mass-normalized, since an
        inertia in kg·m² means nothing against coordinates with no unit.
        For such a geometry the checkbox is hidden and a note says to
        declare the length unit.
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
        """Show the settings in the panel without emitting `changed`."""
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

        Raises `ValueError` for an inertia tensor no body could have,
        the one problem the boxes cannot catch one value at a time.
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
