"""The Expand to Physical Responses pane for modal FRFs and a channel
table: how far a channel may be from the model, what the expansion
gives, a preview, and the button that makes it.

It opens on modal FRFs, whole or some of their records picked, selected
together with a channel table. Each response and monitor channel is
answered against each reference channel, in what the channel measures,
through the shape set the modal FRFs are of
(`Project.channel_frfs`). When the set and the table are on different
geometries the shapes are taken at the channels from the nearest node,
and the tolerance says how far that may be; otherwise the set's own DOFs
are read and the tolerance does not apply. The plot previews the
drive-point FRFs; Expand runs the project's `expand` (principle 13:
select, see, set, apply).
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from .editors import DoubleSpinBox, number_box
from .settings_panel import add_derived, add_rule, add_settings, panel_grid

#: as `Project.expand`: a fiftieth of the table's geometry's extent
TOLERANCE = 0.02


class ChannelFrfPanel(QWidget):
    """The tolerance, emitted as a fraction when it changes and when
    Expand is pressed."""

    #: the tolerance changed
    changed = Signal(float)

    #: Expand was pressed with this tolerance
    apply_asked = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = False
        self.title: QLabel = QLabel('FRFs at the Channels')
        grid = panel_grid(self, self.title)
        self.tolerance_box: DoubleSpinBox = number_box(
            0.1, 100.0, value=100 * TOLERANCE, decimals=1, suffix=' %',
            tip="How far a channel's nearest model node may be, as a share "
                "of the table's geometry's size; a channel further away is "
                'dropped. Only when the set and the table are on different '
                'geometries')
        self.tolerance_box.valueChanged.connect(self._edited)
        add_settings(grid, [('Tolerance', self.tolerance_box)])
        add_rule(grid, 2)
        self._derived = add_derived(grid, [('modes', 'Modes'),
                                           ('references', 'References'),
                                           ('responses', 'Responses'),
                                           ('pairs', 'FRFs'),
                                           ('shapes', 'Shapes'),
                                           ('dropped', 'Dropped')], 3)
        for key in ('shapes', 'dropped'):
            self._derived[key][1].setWordWrap(True)
        self.note: QLabel = QLabel('')
        self.note.setWordWrap(True)
        grid.addWidget(self.note, 9, 0, 1, 2)
        self.apply_button: QPushButton = QPushButton('Expand')
        self.apply_button.setToolTip(
            'Make the FRFs between the channels an object of their own, '
            'beside the table')
        self.apply_button.clicked.connect(
            lambda: self.apply_asked.emit(self.tolerance()))
        grid.addWidget(self.apply_button, 10, 0, 1, 2)
        grid.setRowStretch(11, 1)

    def tolerance(self) -> float:
        """The tolerance, as a fraction of the geometry's extent."""
        return float(self.tolerance_box.value()) / 100.0

    def show_settings(self, tolerance: float, projected: bool) -> None:
        """Put the tolerance up without emitting it, editable only when
        the shapes are projected."""
        self._loading = True
        try:
            self.tolerance_box.setValue(100.0 * tolerance)
            self.tolerance_box.setEnabled(projected)
        finally:
            self._loading = False

    def show_reading(self, *, modes: str = '—', references: str = '—',
                     responses: str = '—', pairs: str = '—',
                     shapes: str = '—', dropped: str = '—', note: str = '',
                     ready: bool = True) -> None:
        """The rows under the settings, the note under them, and whether
        Expand can be pressed."""
        for key, value in (('modes', modes), ('references', references),
                           ('responses', responses), ('pairs', pairs),
                           ('shapes', shapes), ('dropped', dropped)):
            self._derived[key][1].setText(value)
        self.note.setText(note)
        self.note.setVisible(bool(note))
        self.apply_button.setEnabled(ready)

    def _edited(self, *_args) -> None:
        if not self._loading:
            self.changed.emit(self.tolerance())
