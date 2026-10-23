"""The sine extraction's settings, as a table beside the time history.

The settings say how a sweep's levels are read out of the recording:
the smoothing, chosen automatically or set as a number of cycles, and
whether the sweep clock is fitted to the recording. Below them are the
values that follow: the cycles the automatic choice used, the smoothing
window in seconds at the start and end of the first tone, the median
scatter a reading is predicted to have, and the share of samples under
the noise floor. A preview plot shows the readings themselves, sampled
along each tone at the smoothing shown (`core.sine.sample_levels`, a
few short solves, so it follows every edit). The Extract Sine Levels
button runs the project verb with whatever is set at that moment, the
same split as in the filter panel, since these are also settings
carried by the record. Under the button, when the specification holds a
dwell, the dwell readout reads it against time (`gui.dwell_panel`); a
held tone is one point over frequency, and its story is in time. Under
that, when the project also holds a random specification of the floor,
the mixed-mode readout reads the record's data sheet, floor, tones and
total, each by its own rule (`gui.mixed_panel`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QPushButton,
    QWidget,
)

from ..core.sine import CYCLES_LADDER, SineExtraction
from .dwell_panel import DwellPanel
from .editors import DoubleSpinBox, commit_on_enter
from .mixed_panel import MixedPanel
from .settings_panel import (
    PreviewPlot,
    add_derived,
    add_keyed_settings,
    add_rule,
    panel_grid,
)

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.data import TimeHistory
    from ..core.sine import SineSweepSpecification

# Brandon, 2026-09-30: a 0.5 g sweep under a 2.3 g random environment
# read with a 17 dB spread at the one smoothing the code allowed, and
# there was no control for it. Hence the settings and this panel.

#: the two ways the smoothing is chosen, in the order the box lists them
MODES = ('automatic', 'set')
MODE_LABELS = ('Automatic', 'Set')


class SamplePlot(PreviewPlot):
    """The sampled readings against frequency: a dot per control
    channel at the smoothing shown, or a hollow ring at the noise floor
    where the tone is under it.

    It is small on purpose, like the filter panel's response plot: it
    shows at a glance how clean the readings are at this smoothing, and
    the extracted levels show the rest.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.items: list[Any] = []

    def show_samples(self, sampled: list[dict[str, Any]]) -> None:
        """Draw the sampled readings: one color per tone, a dot per
        channel and point, and a ring at the floor where the tone was
        under it."""
        import numpy as np
        import pyqtgraph as pg

        from ..plot import curve_color

        for item in self.items:
            self.plot.removeItem(item)
        self.items = []
        for k, reading in enumerate(sampled):
            color = curve_color(k, self.colors)
            amplitude = np.asarray(reading['amplitude'])
            floor = np.asarray(reading['floor'])
            frequency = np.asarray(reading['frequency'])
            under = amplitude <= floor
            shown = np.where(under, floor, amplitude)
            for row in range(amplitude.shape[0]):
                x = frequency[~under[row]]
                if x.size:
                    self.items.append(self.plot.plot(
                        x, shown[row][~under[row]], pen=None, symbol='o',
                        symbolSize=5, symbolPen=None,
                        symbolBrush=pg.mkBrush(color)))
                x = frequency[under[row]]
                if x.size:
                    self.items.append(self.plot.plot(
                        x, shown[row][under[row]], pen=None, symbol='o',
                        symbolSize=6, symbolBrush=None,
                        symbolPen=pg.mkPen(color, width=1)))


class SinePanel(QWidget):
    """The settings table. Edits are emitted as a whole `SineExtraction`."""

    #: a parameter moved; here is the setting that describes it now
    changed = Signal(object)
    #: the user asked for the levels this panel describes
    apply_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        #: the record and the sweep the parameters describe
        self._history: Any = None
        self._specification: Any = None
        #: what the automatic chose for this record, from the last
        #: sampling — carried on the setting as `chosen`
        self._chosen: float | None = None
        #: the last sampled readings, for the derived rows
        self._sampled: list[dict[str, Any]] = []
        #: set while the panel is writing to its own editors
        self._loading = False
        self.title: QLabel = QLabel('Sine Levels')
        grid = panel_grid(self, self.title)
        self.mode_box: QComboBox = QComboBox()
        self.mode_box.addItems(MODE_LABELS)
        self.mode_box.setToolTip(
            'Automatic: the smoothing climbs until a reading is '
            'predicted to scatter less than the target, measured from '
            'this recording. Set: the number of cycles you type')
        self.cycles_box: DoubleSpinBox = DoubleSpinBox()
        self.cycles_box.setRange(CYCLES_LADDER[0], CYCLES_LADDER[-1])
        self.cycles_box.setDecimals(0)
        self.cycles_box.setSingleStep(10.0)
        self.cycles_box.setValue(CYCLES_LADDER[0])
        self.cycles_box.setToolTip(
            'How many cycles of the tone’s own frequency each reading '
            'averages over. Fewer follow a resonance closely; more hold '
            'the random environment out of the reading')
        self.target_box: DoubleSpinBox = DoubleSpinBox()
        self.target_box.setRange(0.1, 6.0)
        self.target_box.setDecimals(1)
        self.target_box.setSingleStep(0.5)
        self.target_box.setSuffix(' dB')
        self.target_box.setValue(1.0)
        self.target_box.setToolTip(
            'The scatter the automatic smoothing aims to hold a '
            'reading under, one standard deviation')
        self.refine_box: QCheckBox = QCheckBox()
        self.refine_box.setChecked(True)
        self.refine_box.setToolTip(
            'Fit the sweep clock to the recording and solve again when '
            'the recorded sweep drifted from the one commanded — a long '
            'run whose rate was not quite the specification’s would '
            'otherwise read low toward its end')
        rows = (('mode', 'Smoothing', self.mode_box),
                ('cycles', 'Cycles', self.cycles_box),
                ('target', 'Scatter under', self.target_box),
                ('refine', 'Follow the clock', self.refine_box))
        self._labels, self._editors = add_keyed_settings(grid, rows)
        add_rule(grid, len(rows) + 1)
        # what follows from the settings, read where they are set
        self.derived: dict[str, QLabel] = {}
        self.derived_names: dict[str, QLabel] = {}
        derived = (('chosen', 'Cycles used'),
                   ('window', 'Window, start → end'),
                   ('scatter', 'Predicted scatter'),
                   ('under', 'Under the floor'))
        for key, (name, value) in add_derived(grid, derived,
                                              len(rows) + 2).items():
            self.derived[key] = value
            self.derived_names[key] = name
        self.preview: SamplePlot = SamplePlot()
        grid.addWidget(self.preview, len(rows) + 2 + len(derived), 0, 1, 2)
        self.extract_button: QPushButton = QPushButton('Extract Sine Levels')
        self.extract_button.setToolTip(
            'Read every tone’s level out of the recording with exactly '
            'the smoothing shown, one object for the set')
        self.extract_button.clicked.connect(self.apply_asked.emit)
        grid.addWidget(self.extract_button, len(rows) + 3 + len(derived),
                       0, 1, 2)
        # the dwell readout, shown only beside a specification that
        # holds a dwell; it lives in this panel so it comes and goes
        # with the sine view, and reads nothing the extraction sets
        self.dwell: DwellPanel = DwellPanel()
        self.dwell.hide()
        grid.addWidget(self.dwell, len(rows) + 4 + len(derived), 0, 1, 2)
        # and the mixed-mode data sheet, shown only when the project
        # holds the floor's random specification beside the tones
        self.mixed: MixedPanel = MixedPanel()
        self.mixed.hide()
        grid.addWidget(self.mixed, len(rows) + 5 + len(derived), 0, 1, 2)
        grid.setRowStretch(len(rows) + 6 + len(derived), 1)
        commit_on_enter(self.cycles_box, self.target_box)
        self.mode_box.currentIndexChanged.connect(self._edited)
        self.cycles_box.valueChanged.connect(self._edited)
        self.target_box.valueChanged.connect(self._edited)
        self.refine_box.toggled.connect(self._edited)
        self._show_mode('automatic')

    # ---- what the panel is describing -------------------------------------

    def show_history(self, history: TimeHistory,
                     specification: SineSweepSpecification,
                     setting: SineExtraction) -> None:
        """Point the panel at a record, the sweep it is read against
        and the setting on it, and sample the readings at that
        setting."""
        self._history = history
        self._specification = specification
        self._chosen = setting.chosen
        self.set_setting(setting)

    def set_setting(self, setting: SineExtraction) -> None:
        """Show a setting in the panel without emitting `changed`."""
        self._loading = True
        try:
            self.mode_box.setCurrentIndex(
                MODES.index('automatic' if setting.automatic else 'set'))
            if setting.cycles is not None:
                self.cycles_box.setValue(float(setting.cycles))
            elif setting.chosen is not None:
                self.cycles_box.setValue(float(setting.chosen))
            self.target_box.setValue(float(setting.target_db))
            self.refine_box.setChecked(bool(setting.refine))
        finally:
            self._loading = False
        self._show_mode('automatic' if setting.automatic else 'set')
        self._restate()

    def setting(self) -> SineExtraction:
        """The setting the editors currently describe, carrying the
        cycles the automatic choice picked for this record."""
        automatic = MODES[self.mode_box.currentIndex()] == 'automatic'
        return SineExtraction(
            cycles=None if automatic else float(self.cycles_box.value()),
            target_db=float(self.target_box.value()),
            refine=self.refine_box.isChecked(),
            chosen=self._chosen if automatic else None)

    def effective_cycles(self) -> float | None:
        return self.setting().effective_cycles()

    # ---- the editing grammar ------------------------------------------------

    def _show_mode(self, mode: str) -> None:
        """Only what applies (principle 3): the cycles box when they are
        set, the target when the automatic chooses them."""
        automatic = mode == 'automatic'
        for key, wanted in (('cycles', not automatic), ('target', automatic)):
            self._labels[key].setVisible(wanted)
            self._editors[key].setVisible(wanted)
        self.derived_names['chosen'].setVisible(automatic)
        self.derived['chosen'].setVisible(automatic)

    def _edited(self, *_args) -> None:
        if self._loading:
            return
        self._show_mode(MODES[self.mode_box.currentIndex()])
        self._restate()
        self.changed.emit(self.setting())

    def _restate(self) -> None:
        """Sample the readings at the smoothing shown and read the
        derived numbers off them. The automatic's choice is worked out
        here too, once per record and target, since it is what the
        preview must be drawn at."""
        import numpy as np

        from ..core.sine import sample_levels, suggest_cycles

        if self._history is None or self._specification is None:
            return
        setting = self.setting()
        try:
            if setting.automatic:
                self._chosen = suggest_cycles(
                    self._history, self._specification, setting.target_db)
                cycles = self._chosen
            else:
                cycles = float(setting.cycles)
            self._sampled = sample_levels(self._history, self._specification,
                                          cycles)
        except ValueError as refusal:
            self._sampled = []
            self.derived['scatter'].setText(str(refusal)[:40])
            return
        self.derived['chosen'].setText(f'{cycles:g}')
        rate = float(self._history.sample_rate)
        windows = []
        for tone in self._specification.tones:
            _t, f = tone.trajectory(1.0 / rate)
            windows.append((cycles / max(float(f[0]), 1e-9),
                            cycles / max(float(f[-1]), 1e-9)))
        first, last = windows[0]
        self.derived['window'].setText(f'{first:.2f} s → {last:.2f} s')
        scatter = np.concatenate([np.ravel(r['scatter_db'])
                                  for r in self._sampled])
        finite = scatter[np.isfinite(scatter)]
        self.derived['scatter'].setText(
            f'{np.median(finite):.2f} dB' if finite.size else 'under the floor')
        under = np.concatenate([np.ravel(r['amplitude'] <= r['floor'])
                                for r in self._sampled])
        self.derived['under'].setText(
            f'{100.0 * under.mean():.0f} % of samples' if under.size else '—')
        self.preview.show_samples(self._sampled)

    def apply_theme(self, colors: Mapping[str, str]) -> None:
        self.preview.apply_theme(colors)
        if self._sampled:
            self.preview.show_samples(self._sampled)
        self.dwell.apply_theme(colors)
        self.mixed.apply_theme(colors)


def with_chosen(setting: SineExtraction, chosen: float | None) -> SineExtraction:
    """A copy of the setting with `chosen`, the cycles the automatic
    choice picked, written on it."""
    return replace(setting, chosen=chosen)


__all__ = ['MODES', 'SamplePlot', 'SinePanel', 'with_chosen']
