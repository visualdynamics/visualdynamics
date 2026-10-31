"""The Convert to PSD/CPSD pane beside the plot, for a specification
written at breakpoints: the frequency spacing of the lines to read it
onto, what that gives, a preview, and the button that converts it.

A density to drive or predict with is read line by line, so a
specification at breakpoints is read onto evenly spaced lines when it
is converted (`Project.convert_to_psd`, with the sheet's own reading,
`core.author.read_onto_lines`). The spacing opens at the one its sheet
remembers, and blank when there is none: it is not guessed. The plot
previews the result; Generate converts it (principle 13: select, see,
set, apply). A specification already on lines converts with no pane.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from .editors import DoubleSpinBox, commit_on_enter
from .settings_panel import add_derived, add_rule, add_settings, panel_grid


class LinesPanel(QWidget):
    """The spacing, emitted in Hz when it changes and when Generate is
    pressed; 0 means none set."""

    #: the spacing changed
    changed = Signal(float)

    #: Generate was pressed with this spacing
    apply_asked = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = False
        self.title: QLabel = QLabel('Convert to PSD')
        grid = panel_grid(self, self.title)
        self.spacing_box: DoubleSpinBox = DoubleSpinBox()
        self.spacing_box.setDecimals(4)
        self.spacing_box.setRange(0.0, 1e6)
        self.spacing_box.setSuffix(' Hz')
        # blank until set: the minimum stands for no spacing
        self.spacing_box.setSpecialValueText('—')
        self.spacing_box.setToolTip(
            'The spacing of the frequency lines to read the breakpoints '
            "onto, as a controller's lines: multiples of it from 0 Hz")
        self.spacing_box.valueChanged.connect(self._edited)
        add_settings(grid, [('Spacing', self.spacing_box)])
        commit_on_enter(self.spacing_box)
        add_rule(grid, 2)
        self._derived = add_derived(grid, [('breakpoints', 'Breakpoints'),
                                           ('range', 'Range'),
                                           ('lines', 'Lines')], 3)
        self.note: QLabel = QLabel('')
        self.note.setWordWrap(True)
        grid.addWidget(self.note, 6, 0, 1, 2)
        self.apply_button: QPushButton = QPushButton('Generate')
        self.apply_button.setToolTip(
            'Convert the specification to a density on these lines, under '
            'its own name; its warning and abort limits are left out')
        self.apply_button.clicked.connect(
            lambda: self.apply_asked.emit(self.spacing()))
        grid.addWidget(self.apply_button, 7, 0, 1, 2)
        grid.setRowStretch(8, 1)

    def spacing(self) -> float:
        """The spacing set, in Hz; 0 when none is."""
        return float(self.spacing_box.value())

    def show_settings(self, title: str, spacing: float | None) -> None:
        """Put the title and the spacing up without emitting them."""
        self._loading = True
        try:
            self.title.setText(title)
            self.spacing_box.setValue(spacing or 0.0)
        finally:
            self._loading = False

    def show_reading(self, *, breakpoints: str = '—', span: str = '—',
                     lines: str = '—', note: str = '',
                     ready: bool = True) -> None:
        """The rows under the setting, the note under them, and whether
        Generate can be pressed."""
        for key, value in (('breakpoints', breakpoints), ('range', span),
                           ('lines', lines)):
            self._derived[key][1].setText(value)
        self.note.setText(note)
        self.note.setVisible(bool(note))
        self.apply_button.setEnabled(ready)

    def _edited(self, *_args) -> None:
        if not self._loading:
            self.changed.emit(self.spacing())
