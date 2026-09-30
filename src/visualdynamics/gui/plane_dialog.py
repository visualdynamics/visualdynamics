"""Add Plane and Add Block: a meshed rectangle of plates, or a meshed
box of bricks, typed into a geometry.

The app's side of `Project.add_plane` (Brandon, 2026-09-26: the BARC's
planes could only be built from a script, twelve lines the app had no
way to say). A corner, two edges and an element size, in the display
unit, and the block the plates go in — select, see, set, apply: the
plane is drawn in the 3-D view as it is typed, the reading says what it
will add and what it will share with the geometry, and **Add** adds it
and keeps the dialog open for the next one, since a model is several
planes and each is usually the last one moved.

The dialog holds no rule of its own. What a set of numbers makes, what
it shares and whether it is refused are asked of the window
(`reading`), which asks the core; **Add** hands the numbers to the
window (`add`), which calls the project verb, so a plane added here is
the same journaled call a script makes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

#: what the fields hold, in the display unit: a plane one unit on a side,
#: a tenth of that per element — a default that is visible in any unit;
#: a block's third edge a tenth deep, one element through it
DEFAULTS = {'corner': (0.0, 0.0, 0.0), 'edge_a': (1.0, 0.0, 0.0),
            'edge_b': (0.0, 1.0, 0.0), 'edge_c': (0.0, 0.0, 0.1),
            'size': 0.1}


def _spin(value: float) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(-1e6, 1e6)
    box.setDecimals(4)
    box.setValue(value)
    box.setKeyboardTracking(False)
    return box


class AddPlaneDialog(QDialog):
    """The fields, the reading and the buttons; the window does the rest.
    `AddBlockDialog` is the same dialog with a third edge.

    Parameters
    ----------
    parent : QWidget
        The window.
    geometry_name : str
        What the planes are added to, for the title.
    unit : str
        The display unit's label, for the fields.
    blocks : sequence of str
        The geometry's block names, offered in the block box; a name typed
        there that is not one of them makes a new block.
    reading : callable
        ``reading(values) -> (text, ok)``: what these numbers would add,
        or why they are refused. Also draws the preview.
    add : callable
        ``add(values)``: add the plane (and say so on the status bar).
    """

    #: the act's name, the edges typed and the sentence above the fields
    ACT = 'Add Plane'
    EDGES = (('edge_a', 'Edge A'), ('edge_b', 'Edge B'))
    NOTE = ('Edges run from the corner and must be perpendicular. Each is '
            'divided evenly into the number of elements nearest the size.')
    DEFAULT_BLOCK = 'plane'

    def __init__(self, parent: QWidget, geometry_name: str, unit: str,
                 blocks: Sequence[str],
                 reading: Callable[[dict], tuple[str, bool]],
                 add: Callable[[dict], None]) -> None:
        super().__init__(parent)
        self.setWindowTitle(f'{self.ACT} to {geometry_name}')
        self._reading, self._add = reading, add
        self.block: QComboBox = QComboBox()
        self.block.setEditable(True)
        self.block.addItems([name for name in blocks if name])
        self.block.setCurrentText(next((name for name in blocks if name),
                                       self.DEFAULT_BLOCK))
        self.fields: dict[str, list[QDoubleSpinBox]] = {}
        form = QFormLayout()
        form.addRow('Block', self.block)
        for key, label in (('corner', 'Corner'), *self.EDGES):
            row = QHBoxLayout()
            boxes = []
            for axis, value in zip('xyz', DEFAULTS[key]):
                box = _spin(value)
                row.addWidget(QLabel(axis))
                row.addWidget(box, 1)
                boxes.append(box)
            self.fields[key] = boxes
            form.addRow(f'{label} [{unit}]', row)
        self.size: QDoubleSpinBox = _spin(DEFAULTS['size'])
        self.size.setMinimum(0.0)
        form.addRow(f'Element size [{unit}]', self.size)
        self.reading_label: QLabel = QLabel()
        self.reading_label.setWordWrap(True)
        self.add_button: QPushButton = QPushButton('Add')
        self.add_button.setDefault(True)
        self.add_button.clicked.connect(self.add_plane)
        buttons = QDialogButtonBox()
        buttons.addButton(self.add_button,
                          QDialogButtonBox.ButtonRole.ApplyRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(self.NOTE))
        layout.addLayout(form)
        layout.addWidget(self.reading_label)
        layout.addWidget(buttons)
        for boxes in self.fields.values():
            for box in boxes:
                box.valueChanged.connect(self.refresh)
        self.size.valueChanged.connect(self.refresh)
        self.block.currentTextChanged.connect(self.refresh)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.refresh()

    def values(self) -> dict:
        """The fields as typed, in the display unit."""
        return {**{key: tuple(box.value() for box in boxes)
                   for key, boxes in self.fields.items()},
                'size': self.size.value(),
                'block': self.block.currentText().strip()}

    def set_values(self, **values) -> None:
        """Fill fields by name — corner=(x, y, z), size=..., block=...;
        how a test types, and how a caller seeds the next plane."""
        for key, value in values.items():
            if key == 'block':
                self.block.setCurrentText(value)
            elif key == 'size':
                self.size.setValue(value)
            else:
                for box, number in zip(self.fields[key], value):
                    box.setValue(number)

    def refresh(self, *_args) -> None:
        """Ask what the fields would add; say it, draw it, and allow Add
        only when it can be added."""
        text, ok = self._reading(self.values())
        self.reading_label.setText(text)
        self.add_button.setEnabled(ok)

    def add_plane(self) -> None:
        """Add the plane and stay open for the next: the fields keep
        their values, so the next plane is the last one moved. The
        reading after it counts the plane just added as shared, which is
        how a second Add of the same numbers shows itself."""
        self._add(self.values())
        self.refresh()


class AddBlockDialog(AddPlaneDialog):
    """Add Block: a corner and three edges, meshed into bricks
    (`Project.add_block`, 2026-09-30). Holes are the script's business;
    the dialog types the box."""

    ACT = 'Add Block'
    EDGES = (('edge_a', 'Edge A'), ('edge_b', 'Edge B'), ('edge_c', 'Edge C'))
    NOTE = ('Edges run from the corner and must be perpendicular. Each is '
            'divided evenly into the number of bricks nearest the size; '
            'blocks meeting over a face are tied there.')
    DEFAULT_BLOCK = 'block'

