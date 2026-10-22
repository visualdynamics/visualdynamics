"""Plan Instrumentation, as a pane beside the 3-D view.

It follows select, see, set, apply (principle 13). A shape set, or some
of its modes picked in its grid, is selected and Plan Instrumentation
opened on it. The modes measured are the ones selected, and change as
the selection does while the pane is open; the pane names them. The
element groups sensors may go on are checked here, and the surfaces
of the model are drawn in the view, where a click on one leaves it out
or puts it back. The numbers of tri-axial and uni-axial accelerometers
are typed, and the sensors Effective Independence chooses are drawn on
the model as they change, each tri-axial one as a triad and each
uni-axial one as an arrow, with how well they tell the modes apart: the
worst off-diagonal term of the auto-MAC at the chosen channels and the
condition number of the information matrix. Keep adds the plan to the
project.

The pane holds no rules of its own: the window asks
`core.instrumentation` for the plan and shows it here.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QWidget,
)

from ..core.instrumentation import EDGE_ANGLE
from .editors import DoubleSpinBox, SpinBox, commit_on_enter, number_box
from .settings_panel import panel_grid

# Brandon, 2026-10-21: plan the accelerometers of a modal or vibration
# test from the finite element modes, tri-axial ones placed first.
# PLAN.md, "Plan Instrumentation".

#: the directions a uni-axial accelerometer may measure, as the pane
#: names them and `core.instrumentation` takes them
AXES = (('Surface normal (Z)', (2,)), ('Any direction', (0, 1, 2)))


class InstrumentationPanel(QWidget):
    """The modes, the element groups, the counts, how a uni-axial sensor
    may point, a name, what the plan reads, and Keep."""

    #: the counts or the uni-axial directions changed: the window chooses
    #: the sensors again
    changed = Signal()
    #: the element groups or the edge angle changed: the window finds the
    #: places and reads the modes again, then chooses
    places_changed = Signal()
    #: every surface put back, or every one left out
    all_asked = Signal()
    none_asked = Signal()
    keep_asked = Signal()
    close_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Plan Instrumentation')
        grid = panel_grid(self, self.title)
        self.source_label: QLabel = QLabel()
        self.source_label.setWordWrap(True)
        grid.addWidget(self.source_label, 1, 0, 1, 2)
        # the modes are the selection's (Brandon, 2026-10-21: the tree
        # already picks modes, so the pane does not pick them again)
        self.modes_label: QLabel = QLabel()
        self.modes_label.setWordWrap(True)
        self.modes_label.setToolTip('The modes the sensors are chosen to measure: the ones '
                                    'picked in the shape set, or every one above 0 Hz')
        grid.addWidget(self.modes_label, 2, 0, 1, 2)
        grid.addWidget(QLabel('Place on'), 3, 0, 1, 2)
        self.groups: QListWidget = QListWidget()
        self.groups.setToolTip('The element groups sensors may go on: only their outer '
                               'surfaces, never inside a part or where parts touch')
        self.groups.setMinimumHeight(60)
        grid.addWidget(self.groups, 4, 0, 1, 2)
        self.surfaces_label: QLabel = QLabel()
        self.surfaces_label.setWordWrap(True)
        grid.addWidget(self.surfaces_label, 5, 0, 1, 2)
        surface_buttons = QHBoxLayout()
        self.all_button: QPushButton = QPushButton('All')
        self.all_button.setToolTip('Put every surface back')
        self.none_button: QPushButton = QPushButton('None')
        self.none_button.setToolTip('Leave every surface out, to click back the few wanted')
        surface_buttons.addWidget(self.all_button)
        surface_buttons.addWidget(self.none_button)
        grid.addLayout(surface_buttons, 6, 0, 1, 2)
        self.edge_angle: DoubleSpinBox = number_box(
            1.0, 179.0, value=EDGE_ANGLE, decimals=0, step=5.0, suffix='°',
            tip='Two neighboring faces are on one surface while their normals are within '
                'this angle: a curved surface is one surface however far it turns. Raise it '
                'for a coarse curved mesh that splits into strips.')
        self.triaxial: SpinBox = SpinBox()
        self.triaxial.setRange(0, 100000)
        self.triaxial.setToolTip('How many tri-axial accelerometers: placed first')
        self.uniaxial: SpinBox = SpinBox()
        self.uniaxial.setRange(0, 100000)
        self.uniaxial.setToolTip('How many uni-axial accelerometers: placed after the '
                                 'tri-axial ones, at other places')
        self.axes: QComboBox = QComboBox()
        for label, axes in AXES:
            self.axes.addItem(label, axes)
        self.axes.setToolTip('Which way a uni-axial accelerometer may point, in its '
                             "place's own system: Z is out of the surface")
        for row, (label, widget) in enumerate(
                (('Surface edge', self.edge_angle), ('Tri-axial', self.triaxial),
                 ('Uni-axial', self.uniaxial), ('Uni-axial along', self.axes)), start=7):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)
        self.name_label: QLabel = QLabel('Name')
        self.name: QLineEdit = QLineEdit()
        self.name.setToolTip("The start of the three names Keep adds: '... Geometry', "
                             "'... Channels' and '... Shapes'")
        grid.addWidget(self.name_label, 11, 0)
        grid.addWidget(self.name, 11, 1)
        self.reading_label: QLabel = QLabel()
        self.reading_label.setWordWrap(True)
        self.reading_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        grid.addWidget(self.reading_label, 12, 0, 1, 2)
        buttons = QHBoxLayout()
        self.keep_button: QPushButton = QPushButton('Keep')
        self.keep_button.setToolTip('Add the experimental geometry, the channel table '
                                    'and the modes at those channels to the project')
        self.keep_button.setDefault(True)
        self.close_button: QPushButton = QPushButton('Close')
        self.close_button.setToolTip('Put the pane away; nothing is added')
        buttons.addWidget(self.keep_button)
        buttons.addWidget(self.close_button)
        grid.addLayout(buttons, 13, 0, 1, 2)
        grid.setRowStretch(14, 1)
        commit_on_enter(self.triaxial, self.uniaxial, self.edge_angle)
        self.triaxial.valueChanged.connect(lambda _value: self.changed.emit())
        self.uniaxial.valueChanged.connect(lambda _value: self.changed.emit())
        self.axes.currentIndexChanged.connect(lambda _index: self.changed.emit())
        self.edge_angle.valueChanged.connect(lambda _value: self.places_changed.emit())
        self.all_button.clicked.connect(self.all_asked.emit)
        self.none_button.clicked.connect(self.none_asked.emit)
        self.groups.itemChanged.connect(lambda _item: self.places_changed.emit())
        self.keep_button.clicked.connect(self.keep_asked.emit)
        self.close_button.clicked.connect(self.close_asked.emit)

    def open_for(self, source: str, groups, triaxial: int, uniaxial: int,
                 name: str) -> None:
        """Set the pane up on shape set `source`: `groups` the element
        groups as `(id, text)`, all checked; the counts and the name."""
        self.source_label.setText(source)
        self.groups.blockSignals(True)
        self.groups.clear()
        for key, text in groups:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, int(key))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.groups.addItem(item)
        self.groups.blockSignals(False)
        for box, value in ((self.triaxial, triaxial), (self.uniaxial, uniaxial),
                           (self.edge_angle, EDGE_ANGLE)):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        self.axes.blockSignals(True)
        self.axes.setCurrentIndex(0)
        self.axes.blockSignals(False)
        self.name.setText(name)
        self.reading_label.setText('')

    @property
    def picked_groups(self) -> list[int]:
        """The element groups checked, by id."""
        return [int(self.groups.item(row).data(Qt.ItemDataRole.UserRole))
                for row in range(self.groups.count())
                if self.groups.item(row).checkState() == Qt.CheckState.Checked]

    @property
    def all_groups(self) -> bool:
        """Whether every element group listed is checked."""
        return len(self.picked_groups) == self.groups.count()

    @property
    def uniaxial_axes(self) -> tuple[int, ...]:
        """The directions a uni-axial sensor may measure: 2, the normal,
        or 0, 1 and 2."""
        return tuple(self.axes.currentData())

    def show_modes(self, text: str) -> None:
        """Name the modes measured."""
        self.modes_label.setText(text)

    def show_surfaces(self, kept: int, total: int) -> None:
        """How many of the surfaces sensors may go on."""
        self.surfaces_label.setText(
            f'Sensors on {kept} of {total} surfaces. Click a surface in the view to '
            'leave it out or put it back.')

    def show_reading(self, text: str, ok: bool) -> None:
        """What the plan reads, or why there is none; Keep only with a
        plan."""
        self.reading_label.setText(text)
        self.keep_button.setEnabled(ok)
