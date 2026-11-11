"""The Synthesize Modal FRFs pane beside the plot: the lines to
synthesize on, which response of each mode, what that gives, a preview,
and the button that makes them.

It opens on a real shape set, whole for every mode or the modes picked
in its grid. Each mode's FRF is its modal coordinate per unit modal
force, from its frequency, damping and modal mass
(`core.modal_frf`). The lines are a test frame's: the sample rate and
the frame length set the spacing and the Nyquist frequency. The plot
previews them; Generate runs the project's `synthesize_modal_frfs`,
which puts them in a new object beside the set (principle 13: select,
see, set, apply).
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QPushButton, QWidget

from ..core.modal_response import QUANTITIES
from .editors import DoubleSpinBox, SpinBox, number_box
from .settings_panel import add_derived, add_rule, add_settings, panel_grid

#: what the pane opens at: Generate Signal's sample rate, with a frame
#: as long, so the lines are 1 Hz apart
SAMPLE_RATE = 8192.0
FRAME_LENGTH = 8192


class ModalFrfPanel(QWidget):
    """The sample rate, the frame length and the response, emitted as
    (rate in Hz, samples, quantity) when one changes and when Generate is
    pressed."""

    #: a setting changed: (sample rate, frame length, quantity)
    changed = Signal(float, int, str)

    #: Generate was pressed with these settings
    apply_asked = Signal(float, int, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = False
        self.title: QLabel = QLabel('Modal FRFs')
        grid = panel_grid(self, self.title)
        self.rate_box: DoubleSpinBox = number_box(
            1.0, 1e7, value=SAMPLE_RATE, decimals=1, suffix=' Hz',
            tip='Samples per second; the lines run to half of it, the '
                'Nyquist frequency')
        self.frame_box: SpinBox = SpinBox()
        self.frame_box.setRange(2, 2 ** 24)
        self.frame_box.setValue(FRAME_LENGTH)
        self.frame_box.setToolTip(
            'Samples per frame: the lines are the sample rate over this '
            'apart, as Compute FRFs gives for a record framed this way')
        self.quantity_box: QComboBox = QComboBox()
        for quantity in QUANTITIES:
            self.quantity_box.addItem(quantity.capitalize(), quantity)
        self.quantity_box.setCurrentIndex(
            self.quantity_box.findData('acceleration'))
        self.quantity_box.setToolTip(
            "Each mode's displacement, velocity or acceleration per unit "
            'modal force')
        self.rate_box.valueChanged.connect(self._edited)
        self.frame_box.valueChanged.connect(self._edited)
        self.quantity_box.currentIndexChanged.connect(self._edited)
        add_settings(grid, [('Sample rate', self.rate_box),
                            ('Frame', self.frame_box),
                            ('Response', self.quantity_box)])
        add_rule(grid, 4)
        self._derived = add_derived(grid, [('modes', 'Modes'),
                                           ('spacing', 'Spacing'),
                                           ('lines', 'Lines'),
                                           ('nyquist', 'Nyquist'),
                                           ('highest', 'Highest mode')], 5)
        self.note: QLabel = QLabel('')
        self.note.setWordWrap(True)
        grid.addWidget(self.note, 10, 0, 1, 2)
        self.apply_button: QPushButton = QPushButton('Generate')
        self.apply_button.setToolTip(
            'Synthesize the modal FRFs as an object of their own, beside '
            'the shape set')
        self.apply_button.clicked.connect(
            lambda: self.apply_asked.emit(*self.settings()))
        grid.addWidget(self.apply_button, 11, 0, 1, 2)
        grid.setRowStretch(12, 1)

    def settings(self) -> tuple[float, int, str]:
        """(sample rate in Hz, frame length, quantity)."""
        return (float(self.rate_box.value()), int(self.frame_box.value()),
                str(self.quantity_box.currentData()))

    def show_settings(self, sample_rate: float, frame_length: int,
                      quantity: str) -> None:
        """Put the settings up without emitting them."""
        self._loading = True
        try:
            self.rate_box.setValue(sample_rate)
            self.frame_box.setValue(frame_length)
            self.quantity_box.setCurrentIndex(
                self.quantity_box.findData(quantity))
        finally:
            self._loading = False

    def show_reading(self, *, modes: str = '—', spacing: str = '—',
                     lines: str = '—', nyquist: str = '—',
                     highest: str = '—', note: str = '',
                     ready: bool = True) -> None:
        """The rows under the settings, the note under them, and whether
        Generate can be pressed."""
        for key, value in (('modes', modes), ('spacing', spacing),
                           ('lines', lines), ('nyquist', nyquist),
                           ('highest', highest)):
            self._derived[key][1].setText(value)
        self.note.setText(note)
        self.note.setVisible(bool(note))
        self.apply_button.setEnabled(ready)

    def _edited(self, *_args) -> None:
        if not self._loading:
            self.changed.emit(*self.settings())
