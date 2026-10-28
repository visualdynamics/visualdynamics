"""A record's resampling, beside the spectrum that shows it.

The Resample reading of a time history draws one channel's
autospectrum as recorded and at a lower rate three ways, filtered first,
decimated bare and low-passed after decimating, with the span, the
Nyquist frequency, the dropped lines and the folds marked
(`plot.resample`). This panel holds what the reading is made with,
above the rule: the decimation factor (the new rate is the record's
over it, since keeping every q-th sample is the only resampling done),
the anti-alias filter's order, the frame the spectra are taken with,
and the frequencies above the new Nyquist frequency to fold, the
record's tallest line there when none is typed. Under the rule are the
rates, the span and its lines, and the filter's gain at the span's top
and at the lowest frequency that folds into it (`core.resample`), and
under them each fold in words.

Every edit redraws at once. Apply Resample makes the record
(`Project.resample`, the call a script makes), beside the source and
linked to it, at the factor and the filter order shown.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QLineEdit, QPushButton, QWidget

from ..core.resample import FRAME, ORDER, ResampleReading
from .editors import SpinBox
from .settings_panel import add_derived, add_keyed_settings, add_rule, panel_grid

#: the factor the reading opens with: a quarter of the record's rate
FACTOR = 4

#: the frames offered, in samples at the new rate: powers of two, as an
#: analyzer's are, each named with the lines it keeps
FRAMES = (256, 512, 1024, 2048, 4096, 8192, 16384)

#: the read-only rows under the rule, as (key, name)
ROWS = (('source', 'Record rate'), ('rate', 'New rate'),
        ('nyquist', 'Nyquist'), ('span', 'Span'),
        ('lines', 'Lines kept'), ('dropped', 'Lines dropped'),
        ('resolution', 'Resolution'), ('frame', 'Frame used'),
        ('at_span', 'At the span'), ('lowest', 'Lowest fold'),
        ('at_lowest', 'Gain there'))


def frequencies_text(frequencies: Sequence[float] | None) -> str:
    """The frequencies as the field shows them, '4600, 7000'; empty for
    none."""
    return ', '.join(f'{f:g}' for f in frequencies or ())


def parse_frequencies(text: str) -> tuple[float, ...] | None:
    """The frequencies typed in the field, sorted and without repeats:
    an empty tuple for an empty field, None when the text is not a list
    of numbers above zero."""
    parts = [part for part in text.replace(',', ' ').split() if part]
    try:
        values = sorted({float(part) for part in parts})
    except ValueError:
        return None
    if values and values[0] <= 0.0:
        return None
    return tuple(values)


class ResamplePanel(QWidget):
    """The settings and the readout. Edits are emitted as `settings()`."""

    #: a setting moved; here is the whole set now
    changed = Signal(dict)
    #: Apply Resample was pressed: make the record at these settings
    apply_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Resample')
        grid = panel_grid(self, self.title)
        self._loading = False
        self._frequencies: tuple[float, ...] = ()

        self.factor_box: SpinBox = SpinBox()
        self.factor_box.setRange(2, 4096)
        self.factor_box.setValue(FACTOR)
        self.factor_box.setPrefix('÷ ')
        self.factor_box.setToolTip(
            'Keep every n-th sample: the new rate is the record\'s over '
            'this')
        self.order_box: SpinBox = SpinBox()
        self.order_box.setRange(1, 20)
        self.order_box.setValue(ORDER)
        self.order_box.setToolTip(
            'The anti-alias filter\'s design order; run forward and '
            'backward, it rolls off twice as steeply')
        self.frame_box: QComboBox = QComboBox()
        for frame in FRAMES:
            self.frame_box.addItem(f'{frame}', frame)
        self.frame_box.setCurrentIndex(FRAMES.index(FRAME))
        self.frame_box.setToolTip(
            'The spectra\'s frame at the new rate, in samples; a frame '
            'longer than the record takes the longest that fits')
        self.frequencies_edit: QLineEdit = QLineEdit('')
        self.frequencies_edit.setPlaceholderText('tallest line')
        self.frequencies_edit.setToolTip(
            'Frequencies above the new Nyquist frequency to fold, in Hz, '
            'separated by commas. Empty, the record\'s tallest line above '
            'it is folded')
        self.names, self.editors = add_keyed_settings(grid, [
            ('factor', 'Factor', self.factor_box),
            ('order', 'Filter order', self.order_box),
            ('frame', 'Frame', self.frame_box),
            ('frequencies', 'Fold [Hz]', self.frequencies_edit),
        ])
        add_rule(grid, 5)
        made = add_derived(grid, ROWS, 6)
        self.values: dict[str, QLabel] = {
            key: value for key, (_name, value) in made.items()}
        # each fold in words: where it lands and what each order of
        # filtering left there, read rather than seen in a color
        self.verdict: QLabel = QLabel('')
        self.verdict.setWordWrap(True)
        grid.addWidget(self.verdict, 6 + len(ROWS), 0, 1, 2)
        # the reading had no apply (2026-10-28): processing is select,
        # see, set, apply, and a record a script made had no provenance
        self.apply_button: QPushButton = QPushButton('Apply Resample')
        self.apply_button.setToolTip(
            'Make the record at the new rate, filtered first, beside '
            'this one')
        self.apply_button.clicked.connect(self.apply_asked.emit)
        grid.addWidget(self.apply_button, 7 + len(ROWS), 0, 1, 2)
        grid.setRowStretch(8 + len(ROWS), 1)

        for box in (self.factor_box, self.order_box):
            box.valueChanged.connect(self._edited)
        self.frame_box.currentIndexChanged.connect(self._edited)
        self.frequencies_edit.editingFinished.connect(
            self._frequencies_edited)

    def _edited(self, *_args: Any) -> None:
        if not self._loading:
            self.changed.emit(self.settings())

    def _frequencies_edited(self) -> None:
        """The fold field finished: kept when it reads as a list of
        frequencies, put back as it was when it does not."""
        if self._loading:
            return
        found = parse_frequencies(self.frequencies_edit.text())
        if found is None:
            self.frequencies_edit.setText(
                frequencies_text(self._frequencies))
            return
        if found == self._frequencies:
            return
        self._frequencies = found
        self.changed.emit(self.settings())

    def settings(self) -> dict[str, Any]:
        """The reading's settings: 'factor', 'order', 'frame_length' in
        samples at the new rate, and 'frequencies' in Hz (None while the
        field is empty, for the record's tallest line)."""
        return {'factor': int(self.factor_box.value()),
                'order': int(self.order_box.value()),
                'frame_length': int(self.frame_box.currentData()),
                'frequencies': self._frequencies or None}

    def set_settings(self, settings: dict[str, Any] | None) -> None:
        """Set the fields from a `settings()` dict without a redraw per
        field; emits nothing, since the caller is about to draw."""
        if not settings:
            return
        self._loading = True
        try:
            self.factor_box.setValue(int(settings.get('factor', FACTOR)))
            self.order_box.setValue(int(settings.get('order', ORDER)))
            frame = int(settings.get('frame_length', FRAME))
            if frame in FRAMES:
                self.frame_box.setCurrentIndex(FRAMES.index(frame))
            self._frequencies = tuple(settings.get('frequencies') or ())
            self.frequencies_edit.setText(
                frequencies_text(self._frequencies))
        finally:
            self._loading = False

    def show_reading(self, reading: ResampleReading | None) -> None:
        """Fill the readout from `reading`; None clears it."""
        for value in self.values.values():
            value.setText('—')
        if reading is None:
            self.verdict.setText('')
            return
        text = {
            'source': f'{reading.source_rate:g} Hz',
            'rate': f'{reading.rate:g} Hz',
            'nyquist': f'{reading.nyquist:g} Hz',
            'span': f'{reading.span:g} Hz',
            'lines': f'{reading.kept} of {reading.lines}',
            'dropped': f'{reading.dropped}',
            'resolution': f'{reading.resolution:.4g} Hz',
            'frame': f'{reading.frame_length}',
            'at_span': f'{reading.gain_at_span:.1f} dB',
            'lowest': f'{reading.lowest_fold:g} Hz',
            'at_lowest': f'{reading.gain_at_fold:.1f} dB',
        }
        for key, value in text.items():
            self.values[key].setText(value)
        self.values['span'].setToolTip(
            'The alias-free span, the new rate over 2.56. The anti-alias '
            'filter\'s corner is 1.2 times it, so the span is flat')
        self.values['lowest'].setToolTip(
            'The lowest frequency that folds into the span: the new rate '
            'less the span')
        lines = [fold.describe() + '.' for fold in reading.folds]
        if not lines:
            lines = ['Nothing above the new Nyquist frequency to fold.']
        # a typed frequency that does not fold is said, not dropped
        lines += [note + '.' for note in reading.notes]
        self.verdict.setText('\n'.join(lines))
