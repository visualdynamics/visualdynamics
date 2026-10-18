"""The classical pulse a record is judged against, and what came of it.

The Pulse Check reading of a time history draws the record on the
tolerance envelope of a nominal classical pulse, the samples outside it
marked and the velocity window shaded (`plot.classical_pulse`). This
panel holds the pulse (its shape, peak, duration and edge) and the
tolerances it is judged under (the band, the duration's, the edges'
limit, the velocity change's and the quiet either side), and under the
rule, the numbers: the measured peak, duration and edges against
nominal, the velocity change against nominal, the samples outside, and
the verdict in words.

Every edit redraws the plot at once, so the effect of a tolerance is
seen on the record while it is set. Nothing is made: the check is a
reading of a record that already exists, so there is no Apply button.
`core.classical_pulse.check_pulse`, the call a script makes, gives every
number shown here.

The peak is entered in the window's units for acceleration and kept in
SI, so a change of unit system shows the same pulse in the new units
rather than reinterpreting the number in the box.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QWidget

from ..core import classical_pulse as pulses
from .editors import DoubleSpinBox, commit_on_enter, number_box
from .settings_panel import add_keyed_settings, add_rule, panel_grid

#: the shapes as the box names them, in `SHAPES` order
SHAPE_NAMES = (('half-sine', 'Half-sine'), ('sawtooth', 'Sawtooth'),
               ('trapezoid', 'Trapezoid'))

#: what the readout calls each of `plot.classical_pulse.readout`'s rows
READOUT_NAMES = {'peak': 'Peak', 'duration': 'Duration', 'rise': 'Rise',
                 'fall': 'Fall', 'velocity': 'Velocity change',
                 'outside': 'Outside'}

#: the edge a sawtooth or trapezoid opens with when none was set: a
#: twentieth of the duration, half the default limit
OPENING_EDGE = 0.05


def _box(low: float, high: float, decimals: int, step: float, tip: str,
         ) -> DoubleSpinBox:
    """A number field reporting only a finished edit, since each report
    redraws the check."""
    return number_box(low, high, decimals=decimals, step=step, tip=tip)


class PulsePanel(QWidget):
    """The pulse, its tolerances and the readout. Edits are emitted as
    `settings()`."""

    #: a setting moved; here is the whole set now
    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from ..units import DEFAULT_SYSTEM

        self.unit_system: Any = DEFAULT_SYSTEM
        self._loading = False
        self.title: QLabel = QLabel('Pulse Check')
        grid = panel_grid(self, self.title)

        self.shape_box: QComboBox = QComboBox()
        for key, name in SHAPE_NAMES:
            self.shape_box.addItem(name, key)
        self.shape_box.setToolTip(
            'The classical pulse the record was meant to be: a half '
            'cycle of a sine, a terminal-peak sawtooth (a straight rise '
            'over the duration, then a fast fall), or a trapezoid')
        self.peak_box: DoubleSpinBox = _box(
            -1e9, 1e9, 3, 1.0,
            'The nominal peak, signed: a pulse in the negative direction '
            'has a negative peak')
        self.duration_box: DoubleSpinBox = _box(
            0.001, 1e6, 3, 0.5,
            'The nominal duration: the half-sine\'s half cycle, the '
            'sawtooth\'s rise, or the trapezoid\'s rise, top and fall '
            'together')
        self.edge_box: DoubleSpinBox = _box(
            0.0, 1e6, 3, 0.1,
            'The nominal edge: the sawtooth\'s fall after its rise, or '
            'the trapezoid\'s rise and its fall, each')
        self.band_box: DoubleSpinBox = _box(
            0.1, 99.0, 1, 1.0,
            'The band either side of the ideal shape on the pulse, and '
            'either side of zero before and after it, in percent of the '
            'peak. It holds the measured peak too')
        self.duration_tolerance_box: DoubleSpinBox = _box(
            0.1, 99.0, 1, 1.0,
            'How far the measured duration may sit from nominal, in '
            'percent')
        self.edge_limit_box: DoubleSpinBox = _box(
            0.0, 49.0, 1, 1.0,
            'The longest the sawtooth\'s fall, or the trapezoid\'s rise '
            'or fall, may take, in percent of the duration')
        self.velocity_box: DoubleSpinBox = _box(
            0.1, 99.0, 1, 1.0,
            'How far the velocity change, integrated from 0.4 durations '
            'before the pulse to 0.1 after it, may sit from the nominal '
            'pulse\'s, in percent. The document the procedure cites '
            'gives the figure')
        self.quiet_box: DoubleSpinBox = _box(
            0.0, 20.0, 2, 0.5,
            'How long before and after the pulse the record is held to '
            'the band, in durations')
        self.names, self.editors = add_keyed_settings(grid, [
            ('shape', 'Shape', self.shape_box),
            ('peak', 'Peak', self.peak_box),
            ('duration', 'Duration [ms]', self.duration_box),
            ('edge', 'Fall [ms]', self.edge_box),
            ('band', 'Band [±%]', self.band_box),
            ('duration_tolerance', 'Duration [±%]',
             self.duration_tolerance_box),
            ('edge_limit', 'Edge limit [%]', self.edge_limit_box),
            ('velocity_tolerance', 'Velocity [±%]', self.velocity_box),
            ('quiet', 'Quiet [T]', self.quiet_box),
        ])
        add_rule(grid, 10)
        # the numbers, a line apiece, and the verdict under them: labels
        # that wrap, since a value with its distance from nominal is
        # wider than the panel's value column
        self.readout: QLabel = QLabel('')
        self.readout.setWordWrap(True)
        grid.addWidget(self.readout, 11, 0, 1, 2)
        self.verdict: QLabel = QLabel('')
        self.verdict.setWordWrap(True)
        grid.addWidget(self.verdict, 12, 0, 1, 2)
        grid.setRowStretch(13, 1)

        self.set_settings(self.defaults(), emit=False)
        commit_on_enter(self.peak_box, self.duration_box, self.edge_box,
                        self.band_box, self.duration_tolerance_box,
                        self.edge_limit_box, self.velocity_box,
                        self.quiet_box)
        self.shape_box.currentIndexChanged.connect(self._edited)
        for box in (self.peak_box, self.duration_box, self.edge_box,
                    self.band_box, self.duration_tolerance_box,
                    self.edge_limit_box, self.velocity_box,
                    self.quiet_box):
            box.valueChanged.connect(self._edited)

    @staticmethod
    def defaults() -> dict[str, Any]:
        """The settings a record opens on when it has none: a 50 g,
        11 ms half-sine (the peak in SI) and the module's tolerances."""
        return {'shape': 'half-sine', 'peak': 50.0 * 9.80665,
                'duration': 0.011, 'edge': None, 'band': pulses.BAND,
                'duration_tolerance': pulses.DURATION_TOLERANCE,
                'edge_limit': pulses.EDGE_LIMIT,
                'velocity_tolerance': pulses.VELOCITY_TOLERANCE,
                'quiet': pulses.QUIET}

    # ---- the settings -----------------------------------------------------

    def shape(self) -> str:
        """The shape chosen, one of `core.classical_pulse.SHAPES`."""
        return str(self.shape_box.currentData())

    def set_unit_system(self, unit_system: Any) -> None:
        """Show the peak in `unit_system`'s acceleration unit, keeping
        the pulse it describes."""
        kept = self.settings()
        self.unit_system = unit_system
        self.set_settings(kept, emit=False)

    def settings(self) -> dict[str, Any]:
        """The check's arguments, in SI and as shares: 'shape', 'peak'
        (m/s²), 'duration' and 'edge' (seconds; edge None for a
        half-sine), 'band', 'duration_tolerance', 'edge_limit',
        'velocity_tolerance' and 'quiet' (durations)."""
        shape = self.shape()
        return {
            'shape': shape,
            'peak': float(self.unit_system.to_si(self.peak_box.value(),
                                                 'acceleration')),
            'duration': self.duration_box.value() / 1e3,
            'edge': (None if shape == 'half-sine'
                     else self.edge_box.value() / 1e3),
            'band': self.band_box.value() / 100.0,
            'duration_tolerance': self.duration_tolerance_box.value() / 100.0,
            'edge_limit': self.edge_limit_box.value() / 100.0,
            'velocity_tolerance': self.velocity_box.value() / 100.0,
            'quiet': self.quiet_box.value()}

    def set_settings(self, settings: dict[str, Any],
                     emit: bool = True) -> None:
        """Set every field from `settings` (as `settings()` gives them)
        without a redraw per field; then emit once, unless told not to."""
        self._loading = True
        try:
            keys = [key for key, _name in SHAPE_NAMES]
            self.shape_box.setCurrentIndex(keys.index(settings['shape']))
            self.peak_box.setValue(float(self.unit_system.from_si(
                settings['peak'], 'acceleration')))
            self.duration_box.setValue(settings['duration'] * 1e3)
            edge = settings.get('edge')
            if edge is None:
                edge = OPENING_EDGE * settings['duration']
            self.edge_box.setValue(edge * 1e3)
            self.band_box.setValue(100.0 * settings['band'])
            self.duration_tolerance_box.setValue(
                100.0 * settings['duration_tolerance'])
            self.edge_limit_box.setValue(100.0 * settings['edge_limit'])
            self.velocity_box.setValue(
                100.0 * settings['velocity_tolerance'])
            self.quiet_box.setValue(settings['quiet'])
        finally:
            self._loading = False
        self._shown_rows()
        if emit:
            self.changed.emit(self.settings())

    def _shown_rows(self) -> None:
        """Show only the rows that apply (principle 3): a half-sine has
        no edge and no edge limit, and the edge is a fall on a sawtooth
        and a rise and fall on a trapezoid. The peak's unit follows the
        window's."""
        shape = self.shape()
        edged = shape != 'half-sine'
        for key in ('edge', 'edge_limit'):
            self.names[key].setVisible(edged)
            self.editors[key].setVisible(edged)
        self.names['edge'].setText('Fall [ms]' if shape == 'sawtooth'
                                   else 'Edges [ms]')
        self.names['peak'].setText(
            f'Peak [{self.unit_system.label_text("acceleration")}]')

    def _edited(self, *_args: Any) -> None:
        self._shown_rows()
        if not self._loading:
            self.changed.emit(self.settings())

    # ---- the readout ------------------------------------------------------

    def show_check(self, check: Any | None, reason: str = '') -> None:
        """Write a `core.classical_pulse.PulseCheck`'s numbers and its
        verdict, in bold when it is flagged; None clears the numbers and
        shows `reason` in their place."""
        from ..plot.classical_pulse import readout

        if check is None:
            self.readout.setText('')
            self.verdict.setText(reason)
            self.verdict.setStyleSheet('font-weight: 600;' if reason
                                       else '')
            return
        rows = readout(check, self.unit_system)
        self.readout.setText('\n'.join(
            f'{READOUT_NAMES[key]}: {value}' for key, value in rows.items()))
        self.verdict.setText(check.describe())
        self.verdict.setStyleSheet('' if check.passed
                                   else 'font-weight: 600;')
