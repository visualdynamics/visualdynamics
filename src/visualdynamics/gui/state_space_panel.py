"""The Export State-Space Model pane beside the plot: the frequencies
that shape the model, what it holds, a preview, and the button that
writes it.

It opens on a real shape set selected together with a channel table.
The table's reference rows are the inputs and every row an output, so
the file matches the table Rattlesnake runs with
(`core.state_space`). The maximum frequency leaves out the modes at
and above it, for a controller its Nyquist frequency, and opens blank:
it is not guessed. Modes below the rigid frequency are moved up to it.
The plot previews the model's drive-point FRFs over the set's own;
Export writes the `.npz` (principle 13: select, see, set, apply).
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from .editors import DoubleSpinBox
from .settings_panel import add_derived, add_rule, add_settings, panel_grid


class StateSpacePanel(QWidget):
    """The maximum and rigid frequencies, emitted in Hz when either
    changes and when Export is pressed; a maximum of 0 means none set."""

    #: a frequency changed: (maximum, rigid)
    changed = Signal(float, float)

    #: Export was pressed with these frequencies
    apply_asked = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = False
        self.title: QLabel = QLabel('State-Space Model')
        grid = panel_grid(self, self.title)
        self.maximum_box: DoubleSpinBox = DoubleSpinBox()
        self.maximum_box.setDecimals(1)
        self.maximum_box.setRange(0.0, 1e7)
        self.maximum_box.setSuffix(' Hz')
        # blank until set: the minimum stands for every mode kept
        self.maximum_box.setSpecialValueText('—')
        self.maximum_box.setToolTip(
            "Modes at and above this frequency are left out. For a "
            "controller, its Nyquist frequency, half its sample rate: a "
            'mode above it would alias into the band')
        self.rigid_box: DoubleSpinBox = DoubleSpinBox()
        self.rigid_box.setDecimals(2)
        self.rigid_box.setRange(0.01, 1e4)
        self.rigid_box.setSuffix(' Hz')
        self.rigid_box.setValue(1.0)
        self.rigid_box.setToolTip(
            'Modes below this frequency are moved up to it, keeping their '
            'damping, so a rigid-body mode does not drift through a long '
            'run')
        for box in (self.maximum_box, self.rigid_box):
            box.valueChanged.connect(self._edited)
        add_settings(grid, [('Maximum', self.maximum_box),
                            ('Rigid', self.rigid_box)])
        add_rule(grid, 3)
        self._derived = add_derived(grid, [('modes', 'Modes'),
                                           ('lifted', 'Moved up'),
                                           ('dropped', 'Left out'),
                                           ('inputs', 'Inputs'),
                                           ('channels', 'Channels'),
                                           ('units', 'Units')], 4)
        # a table in lbf, g, in/s and mm names them all
        self._derived['units'][1].setWordWrap(True)
        self.note: QLabel = QLabel('')
        self.note.setWordWrap(True)
        grid.addWidget(self.note, 10, 0, 1, 2)
        self.apply_button: QPushButton = QPushButton('Export…')
        self.apply_button.setToolTip(
            "Write the model as the .npz Rattlesnake's state-space "
            'hardware loads')
        self.apply_button.clicked.connect(
            lambda: self.apply_asked.emit(*self.frequencies()))
        grid.addWidget(self.apply_button, 11, 0, 1, 2)
        grid.setRowStretch(12, 1)

    def frequencies(self) -> tuple[float, float]:
        """(maximum, rigid) in Hz; a maximum of 0 when none is set."""
        return float(self.maximum_box.value()), float(self.rigid_box.value())

    def show_settings(self, maximum: float | None, rigid: float) -> None:
        """Put the frequencies up without emitting them."""
        self._loading = True
        try:
            self.maximum_box.setValue(maximum or 0.0)
            self.rigid_box.setValue(rigid)
        finally:
            self._loading = False

    def show_reading(self, *, modes: str = '—', lifted: str = '—',
                     dropped: str = '—', inputs: str = '—',
                     channels: str = '—', units: str = '—', note: str = '',
                     ready: bool = True) -> None:
        """The rows under the settings, the note under them, and whether
        Export can be pressed."""
        for key, value in (('modes', modes), ('lifted', lifted),
                           ('dropped', dropped), ('inputs', inputs),
                           ('channels', channels), ('units', units)):
            self._derived[key][1].setText(value)
        self.note.setText(note)
        self.note.setVisible(bool(note))
        self.apply_button.setEnabled(ready)

    def _edited(self, *_args) -> None:
        if not self._loading:
            self.changed.emit(*self.frequencies())
