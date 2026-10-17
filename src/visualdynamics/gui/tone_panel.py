"""The tone readout, beside the autospectrum it reads.

The Tones reading of a plain autospectrum puts a cursor on the plot.
Wherever it stands, the tone nearest it is read off the spectrum
(`core.tone_readout.read_tone`) and drawn on it (`plot.tone_readout`),
and this panel says what was read: the tone's amplitude and rms, the
spike's height beside it and the height it would have on a line, the
floor taken away, the line spacing, the window and its noise bandwidth,
and where the tone sits between lines. The spike and the amplitude are
shown together because a spike's height moves with the spacing and the
window while the amplitude does not, and the plot alone invites reading
one as the other.

One setting is made here, the window (with its alpha or beta for a
Tukey or Kaiser window), because a spectrum does not carry the one it
was averaged with: a PSD computed in the project has it in
its provenance (`Project.averaging_of`), and the panel starts from that
and says so; an imported one is read as Hann, the common choice, and the
panel says that too. Nothing is made, so there is no Apply button: the
reading is of a spectrum that already exists, as the octave reading's
preview is before its button is pressed.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QWidget

from ..core.averaging import WINDOW_PARAMETERS, WINDOWS
from ..core.tone_readout import density_label
from .editors import DoubleSpinBox
from .settings_panel import add_derived, add_rule, add_settings, panel_grid

#: the readings under the rule, in order: key and row name
ROWS = (('source', 'Window from'), ('channel', 'Channel'),
        ('frequency', 'Tone at'), ('amplitude', 'Tone'), ('rms', 'Tone rms'),
        ('spike', 'Spike'), ('on_line', 'On a line'), ('floor', 'Floor'),
        ('spacing', 'Line spacing'), ('bandwidth', 'Bandwidth'),
        ('offset', 'Off its line'))


class TonePanel(QWidget):
    """The window setting and the reading. Edits are emitted as the
    window's name; `parameter` gives its alpha or beta."""

    #: the window or its parameter was changed; here is its name now
    changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Tone Readout')
        grid = panel_grid(self, self.title)
        self._loading = False

        self.window_box: QComboBox = QComboBox()
        for name in WINDOWS:
            self.window_box.addItem(name, name)
        self.window_box.setToolTip(
            'The window the spectrum was averaged with. It sets how far '
            'a tone leaks and how tall its spike stands, so a reading '
            'under the wrong window reads the spike wrong, though the '
            'amplitude summed from the lines barely moves')
        # a Tukey's alpha or a Kaiser's beta, shown only while such a
        # window is chosen, as the averaging pane shows it
        self.parameter_box: DoubleSpinBox = DoubleSpinBox()
        self.parameter_box.setDecimals(2)
        self.parameter_box.setToolTip(
            "The window's own number: Tukey's alpha, the tapered "
            "fraction, or Kaiser's beta")
        names = add_settings(grid, [('Window', self.window_box),
                                    ('Alpha', self.parameter_box)])
        self._parameter_label: QLabel = names[1]
        add_rule(grid, 3)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value)
            in add_derived(grid, ROWS, 4).items()}
        # what the reading warns, in words, under the numbers: a tone
        # between lines, a floor that differs either side, no tone
        self.notes: QLabel = QLabel('')
        self.notes.setWordWrap(True)
        grid.addWidget(self.notes, 4 + len(ROWS), 0, 1, 2)
        grid.setRowStretch(5 + len(ROWS), 1)
        self._restate_parameter(None)
        self.window_box.currentIndexChanged.connect(self._window_edited)
        self.parameter_box.valueChanged.connect(self._edited)

    def _restate_parameter(self, value: float | None) -> None:
        """Fit the parameter row to the window: named for its number and
        set to `value`, or its default when None; hidden for a window
        that takes none."""
        settings = WINDOW_PARAMETERS.get(self.window())
        self._parameter_label.setVisible(settings is not None)
        self.parameter_box.setVisible(settings is not None)
        if settings is None:
            return
        what, default, (lowest, highest) = settings
        was, self._loading = self._loading, True
        try:
            self._parameter_label.setText(what.capitalize())
            self.parameter_box.setRange(lowest, highest)
            self.parameter_box.setSingleStep((highest - lowest) / 20.0)
            self.parameter_box.setValue(default if value is None else value)
        finally:
            self._loading = was

    def _window_edited(self, *_args: Any) -> None:
        if not self._loading:
            self._restate_parameter(None)
        self._edited()

    def _edited(self, *_args: Any) -> None:
        if not self._loading:
            self.derived['source'].setText('chosen')
            self.changed.emit(self.window())

    def window(self) -> str:
        """The window chosen, one of `core.averaging.WINDOWS`."""
        return str(self.window_box.currentData())

    def parameter(self) -> float | None:
        """The Tukey window's alpha or the Kaiser window's beta; None
        for a window that takes none."""
        if self.window() not in WINDOW_PARAMETERS:
            return None
        return float(self.parameter_box.value())

    def set_window(self, name: str, source: str,
                   parameter: float | None = None) -> None:
        """Show `name` as the window, with `parameter` for a Tukey or
        Kaiser window (its default when None), without emitting, and say
        where it came from ('the averaging', 'chosen', or 'assumed')."""
        self._loading = True
        try:
            self.window_box.setCurrentIndex(self.window_box.findData(name))
            self._restate_parameter(parameter)
        finally:
            self._loading = False
        self.derived['source'].setText(source)

    def show_reading(self, reading: Any, channel: str = '', *,
                     scale: float = 1.0, density_scale: float = 1.0,
                     unit: str = '') -> None:
        """Write `reading`, a `ToneReading`, into the rows: amplitudes
        converted by `scale` into `unit`, densities by `density_scale`.
        None clears them."""
        values = dict.fromkeys(key for key, _name in ROWS if key != 'source')
        if reading is not None:
            density = density_label(unit)
            bandwidth = f'{reading.noise_bandwidth:.3g}'
            amplitude = ('none' if not reading.amplitude else
                         f'{reading.amplitude * scale:.4g} {unit} peak')
            values.update(
                channel=channel or '—',
                frequency=f'{reading.frequency:.4g} Hz',
                amplitude=amplitude.replace('  ', ' '),
                rms=f'{reading.rms * scale:.4g} {unit}'.strip(),
                spike=f'{reading.spike * density_scale:.4g} {density}',
                on_line=(f'{(reading.on_line + reading.floor) * density_scale:.4g}'
                         f' {density}'),
                floor=f'{reading.floor * density_scale:.4g} {density}',
                spacing=f'{reading.spacing:.4g} Hz',
                bandwidth=(f'{bandwidth} line' if bandwidth == '1'
                           else f'{bandwidth} lines'),
                offset=f'{abs(reading.offset):.2f} of a line')
        for key, value in values.items():
            self.derived[key].setText('—' if value is None else value)
        # the notes' densities in the same units as the rows above them
        self.notes.setText('' if reading is None else '\n'.join(
            note[0].upper() + note[1:] + '.'
            for note in reading.notes_in(unit, scale)))
