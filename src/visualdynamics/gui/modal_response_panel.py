"""The Compute Modal Responses pane beside the plot: which response of
each mode to keep, what it is computed from, a preview, and the button
that computes it.

It opens on a time history with force channels, or a CPSD of forces,
selected together with a shape set: the whole set for every mode, or
the modes picked in its grid. The plot previews the modal responses
over the start of the record, or each mode's response autospectrum for
a CPSD; Compute runs the project's `compute_modal_responses`, which
puts them in a new object linked beside the forces (principle 13:
select, see, set, apply).
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QPushButton, QWidget

from ..core.modal_response import QUANTITIES
from .settings_panel import add_derived, add_rule, add_settings, panel_grid


class ModalResponsePanel(QWidget):
    """The response to keep, emitted as its name ('acceleration') when it
    changes and when Compute is pressed."""

    #: the quantity changed
    changed = Signal(str)

    #: Compute was pressed with this quantity
    apply_asked = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = False
        self.title: QLabel = QLabel('Modal Responses')
        grid = panel_grid(self, self.title)
        self.quantity_box: QComboBox = QComboBox()
        for quantity in QUANTITIES:
            self.quantity_box.addItem(quantity.capitalize(), quantity)
        self.quantity_box.setToolTip(
            "Which response of each modal coordinate to keep: its "
            "displacement q, velocity or acceleration. The physical "
            'response is the shapes times it')
        self.quantity_box.currentIndexChanged.connect(self._edited)
        add_settings(grid, [('Response', self.quantity_box)])
        add_rule(grid, 2)
        self._derived = add_derived(grid, [('modes', 'Modes'),
                                           ('forces', 'Force channels'),
                                           ('samples', 'Samples'),
                                           ('highest', 'Highest mode')], 3)
        self.note: QLabel = QLabel('')
        self.note.setWordWrap(True)
        grid.addWidget(self.note, 7, 0, 1, 2)
        self.apply_button: QPushButton = QPushButton('Compute')
        self.apply_button.setToolTip(
            'Compute the modal responses as an object of their own, '
            'beside the forces')
        self.apply_button.clicked.connect(
            lambda: self.apply_asked.emit(self.quantity()))
        grid.addWidget(self.apply_button, 8, 0, 1, 2)
        grid.setRowStretch(9, 1)

    def quantity(self) -> str:
        """The response chosen."""
        return self.quantity_box.currentData() or 'acceleration'

    def show_settings(self, quantity: str) -> None:
        """Put the quantity up without emitting it."""
        self._loading = True
        try:
            self.quantity_box.setCurrentIndex(
                max(0, self.quantity_box.findData(quantity)))
        finally:
            self._loading = False

    def show_reading(self, *, modes: str = '—', forces: str = '—',
                     samples: str = '—', highest: str = '—',
                     note: str = '', lines: bool = False) -> None:
        """The rows under the setting, and the note under them; `lines`
        when the forces are a density, whose length is in frequency
        lines rather than samples."""
        self._derived['samples'][0].setText('Lines' if lines else 'Samples')
        for key, value in (('modes', modes), ('forces', forces),
                           ('samples', samples), ('highest', highest)):
            self._derived[key][1].setText(value)
        self.note.setText(note)
        self.note.setVisible(bool(note))
        self.apply_button.setEnabled(not note.startswith('Cannot'))

    def _edited(self, *_args) -> None:
        if not self._loading:
            self.changed.emit(self.quantity())
