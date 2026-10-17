"""The signature comparison's settings and its table, beside the two
signatures it compares.

The Signature reading of two FRFs, transmissibilities or tracked
sweeps selected together draws them over each other with a window on
each mode for its limits (`plot.signature`). This panel holds what the
comparison is made under (which of the two is the signature before the
test, the modes named and which of them count, and the two limits) and,
under them, what came of it: each mode's frequency and height before
and after, its two shifts, Q before and after, its verdict, and what
its readings cannot be trusted for.

Every edit redraws the plot at once. Nothing is made: the comparison is
a reading of two objects that already exist, as the signal-to-noise
reading of two densities is, so there is no Apply button. The same
call a script makes, `core.signature.compare_signatures`, gives every
number shown here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QLineEdit, QWidget

from ..core.signature import AMPLITUDE_LIMIT, FREQUENCY_LIMIT, SignatureMode
from .editors import DoubleSpinBox, SpinBox
from .settings_panel import add_derived, add_keyed_settings, add_rule, panel_grid

#: how many modes are found when none is named
FIND_COUNT = 2


def parse_modes(text: str) -> list[SignatureMode]:
    """The modes field read: frequencies in Hz separated by commas or
    spaces, a mode that does not count written in parentheses, as in
    '84, (412)'. Raises ValueError for anything else."""
    out = []
    for token in text.replace(',', ' ').split():
        counts = not (token.startswith('(') and token.endswith(')'))
        number = token.strip('()')
        try:
            frequency = float(number)
        except ValueError:
            raise ValueError(f'{token!r} is not a frequency') from None
        if not frequency > 0.0:
            raise ValueError(f'{token!r} is not a positive frequency')
        out.append(SignatureMode(frequency, counts))
    return out


def modes_text(modes: Sequence[SignatureMode]) -> str:
    """The modes as the field writes them, the inverse of
    `parse_modes`."""
    return ', '.join(f'{m.frequency:g}' if m.counts else f'({m.frequency:g})'
                     for m in modes)


def _percent(value: float, tip: str) -> DoubleSpinBox:
    box = DoubleSpinBox()
    box.setRange(0.0, 100.0)
    box.setDecimals(1)
    box.setSingleStep(1.0)
    box.setValue(value)
    box.setToolTip(tip)
    return box


class SignaturePanel(QWidget):
    """The settings and the table. Edits are emitted as `settings()`."""

    #: a setting moved; here is the whole set now
    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Signature')
        grid = panel_grid(self, self.title)
        self._loading = False

        self.before_box: QComboBox = QComboBox()
        self.before_box.setToolTip(
            'Which of the two selected is the signature before the test; '
            'the other is after it')
        self.modes_edit: QLineEdit = QLineEdit()
        self.modes_edit.setPlaceholderText('found')
        self.modes_edit.setToolTip(
            'The modes compared, in Hz, a frequency near each: 84, 412. '
            'A mode that does not count is written in parentheses, '
            '(412), and is reported but not judged.\nEmpty, the most '
            'prominent peaks of the signature before are found and all '
            'count; the test plan says which modes count')
        self.find_box: SpinBox = SpinBox()
        self.find_box.setRange(1, 20)
        self.find_box.setValue(FIND_COUNT)
        self.find_box.setToolTip('How many peaks to find when no mode '
                                 'is named')
        self.frequency_box: DoubleSpinBox = _percent(
            FREQUENCY_LIMIT, 'How far each mode that counts may move in '
            'frequency, in percent either way')
        self.amplitude_box: DoubleSpinBox = _percent(
            AMPLITUDE_LIMIT, 'How far each mode that counts may move in '
            'height, in percent either way')
        self.names, self.editors = add_keyed_settings(grid, [
            ('before', 'Before', self.before_box),
            ('modes', 'Modes [Hz]', self.modes_edit),
            ('find', 'Peaks found', self.find_box),
            ('frequency', 'Frequency [±%]', self.frequency_box),
            ('amplitude', 'Amplitude [±%]', self.amplitude_box),
        ])
        add_rule(grid, 6)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value) in add_derived(
                grid, (('verdict', 'Verdict'),), 7).items()}
        # one block per mode, as many as are named, so a label that
        # wraps rather than a fixed set of rows
        self.readout: QLabel = QLabel('')
        self.readout.setWordWrap(True)
        grid.addWidget(self.readout, 8, 0, 1, 2)
        grid.setRowStretch(9, 1)

        self.before_box.currentIndexChanged.connect(self._edited)
        self.modes_edit.editingFinished.connect(self._edited)
        for box in (self.find_box, self.frequency_box, self.amplitude_box):
            box.valueChanged.connect(self._edited)
        self._shown_rows()

    def _shown_rows(self) -> None:
        """The peak count applies only while no mode is named
        (principle 3)."""
        shown = not self.modes_edit.text().strip()
        self.names['find'].setVisible(shown)
        self.editors['find'].setVisible(shown)

    def _edited(self, *_args: Any) -> None:
        self._shown_rows()
        if not self._loading:
            self.changed.emit(self.settings())

    def set_pair(self, names: Sequence[str]) -> None:
        """Offer the two selected objects as the signature before,
        keeping the choice when the same two are selected again. A new
        pair clears the modes named, which were named for the last
        pair's article, so its own are found."""
        names = [str(name) for name in names]
        if [self.before_box.itemText(i)
                for i in range(self.before_box.count())] == names:
            return
        self._loading = True
        try:
            self.before_box.clear()
            self.before_box.addItems(names)
            self.modes_edit.clear()
        finally:
            self._loading = False
        self._shown_rows()

    def settings(self) -> dict[str, Any]:
        """The comparison's arguments: 'before' (the name chosen),
        'modes' (the field read, None when empty, a string when it
        cannot be read), 'find', 'frequency_limit' and
        'amplitude_limit'."""
        text = self.modes_edit.text().strip()
        modes: Any = None
        if text:
            try:
                modes = parse_modes(text)
            except ValueError as error:
                modes = str(error)
        return {'before': self.before_box.currentText(), 'modes': modes,
                'find': self.find_box.value(),
                'frequency_limit': self.frequency_box.value(),
                'amplitude_limit': self.amplitude_box.value()}

    def set_settings(self, before: str | None = None,
                     modes: Sequence[SignatureMode] | str | None = None,
                     find: int | None = None,
                     frequency_limit: float | None = None,
                     amplitude_limit: float | None = None) -> None:
        """Set the fields without a redraw per field, then emit once."""
        self._loading = True
        try:
            if before is not None:
                self.before_box.setCurrentText(before)
            if modes is not None:
                self.modes_edit.setText(
                    modes if isinstance(modes, str) else modes_text(modes))
            if find is not None:
                self.find_box.setValue(find)
            if frequency_limit is not None:
                self.frequency_box.setValue(frequency_limit)
            if amplitude_limit is not None:
                self.amplitude_box.setValue(amplitude_limit)
        finally:
            self._loading = False
        self._edited()

    def show_result(self, comparison: Any, scale: float = 1.0,
                    unit: str = '') -> None:
        """Write the verdict and each mode's block: its label and
        verdict, then the frequency, height and Q before and after with
        the two shifts, then the mode's notes. `scale` takes the heights
        from SI to the units the plot is drawn in, and `unit` names
        them. None clears the readout."""
        from ..plot.signature import percent

        if comparison is None:
            self.derived['verdict'].setText('—')
            self.readout.setText('')
            return
        verdict = {True: 'passes', False: 'fails',
                   None: 'none judged'}[comparison.passed]
        self.derived['verdict'].setText(verdict)
        blocks = []
        for shift in comparison.shifts:
            def q(reading):
                return '—' if reading.q is None else f'{reading.q:.3g}'

            blocks.append(
                f'{shift.mode.label()}: {shift.verdict}\n'
                f'  {shift.before.frequency:.1f} to '
                f'{shift.after.frequency:.1f} Hz, '
                f'{percent(shift.frequency_shift)}\n'
                f'  height {shift.before.height * scale:#.3g} to '
                f'{shift.after.height * scale:#.3g}'
                f'{" " + unit if unit else ""}, '
                f'{percent(shift.amplitude_shift)}\n'
                f'  Q {q(shift.before)} to {q(shift.after)}'
                + ''.join(f'\n  {note}' for note in shift.notes))
        self.readout.setText('\n'.join(blocks))
