"""The sine extraction's settings, as a table beside the time history.

How a sweep's levels are read out of the recording — the smoothing,
automatic or set, and whether the sweep clock is refined against the
recording — with what follows from it read beside the numbers: the
cycles the automatic chose, the smoothing window in seconds at the
sweep's ends, the scatter a reading is predicted to carry and how
much of the sweep sits under the noise floor, and a preview of the
readings themselves, sampled along each tone at the smoothing shown
(`core.sine.sample_levels`: a few short solves, so it follows every
edit). Beside it the Extract Sine Levels button, the project verb with
whatever is set at that moment — the same division of labor as the
filter panel, because it is the same kind of thing: parameters that
ride the record (Brandon, 2026-09-30: a 0.5 g sweep under a 2.3 g
random environment read with a 17 dB spread at the one smoothing the
code allowed, and the lever had no control).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.sine import CYCLES_LADDER, SineExtraction
from .editors import DoubleSpinBox, commit_on_enter
from .settings_panel import add_derived, panel_grid

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.data import TimeHistory
    from ..core.sine import SineSweepSpecification

#: the two ways the smoothing is chosen, in the order the box lists them
MODES = ('automatic', 'set')
MODE_LABELS = ('Automatic', 'Set')


class SamplePlot(QWidget):
    """The sampled readings against frequency: dots per control
    channel at the smoothing shown, the noise floor beside each as a
    hollow ring where the tone sits under it.

    Small on purpose, like the filter panel's response: it answers
    "how clean does this smoothing read" at a glance, and the
    extracted levels answer the rest.
    """

    HEIGHT = 150

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        import pyqtgraph as pg

        from ..theme import theme as resolve_theme

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.colors: Mapping[str, str] = resolve_theme(None)
        self.widget: Any = pg.PlotWidget()
        self.widget.setFixedHeight(self.HEIGHT)
        self.plot: Any = self.widget.getPlotItem()
        self.plot.setLogMode(x=True, y=True)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        for edge in ('left', 'bottom'):
            axis = self.plot.getAxis(edge)
            axis.setStyle(tickTextOffset=2, tickLength=-3)
            axis.setTextPen(self.colors['plot_foreground'])
            axis.setPen(self.colors['plot_foreground'])
        self.plot.getAxis('left').setWidth(34)
        self.items: list[Any] = []
        layout.addWidget(self.widget)

    def apply_theme(self, colors: Mapping[str, str]) -> None:
        self.colors = colors
        self.widget.setBackground(colors['plot_background'])
        for edge in ('left', 'bottom'):
            axis = self.plot.getAxis(edge)
            axis.setTextPen(colors['plot_foreground'])
            axis.setPen(colors['plot_foreground'])

    def show_samples(self, sampled: list[dict[str, Any]]) -> None:
        """Draw the sampled readings: one color per tone, a dot per
        channel and point, rings where the tone was under its floor."""
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
    """The parameter table. Edits arrive as a whole `SineExtraction`."""

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
        self._labels: dict[str, QLabel] = {}
        self._editors: dict[str, QWidget] = {}
        rows = (('mode', 'Smoothing', self.mode_box),
                ('cycles', 'Cycles', self.cycles_box),
                ('target', 'Scatter under', self.target_box),
                ('refine', 'Follow the clock', self.refine_box))
        for row, (key, label, editor) in enumerate(rows, start=1):
            name = QLabel(label)
            grid.addWidget(name, row, 0)
            grid.addWidget(editor, row, 1)
            self._labels[key] = name
            self._editors[key] = editor
        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setFrameShadow(QFrame.Shadow.Sunken)
        grid.addWidget(rule, len(rows) + 1, 0, 1, 2)
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
        grid.setRowStretch(len(rows) + 4 + len(derived), 1)
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
        and the setting on it; sample the readings at that setting."""
        self._history = history
        self._specification = specification
        self._chosen = setting.chosen
        self.set_setting(setting)

    def set_setting(self, setting: SineExtraction) -> None:
        """Restate the panel from a setting, without re-emitting."""
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
        """The setting the editors currently describe, carrying what
        the automatic chose for this record."""
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


def with_chosen(setting: SineExtraction, chosen: float | None) -> SineExtraction:
    """The setting with what the automatic chose written on it."""
    return replace(setting, chosen=chosen)


__all__ = ['MODES', 'SamplePlot', 'SinePanel', 'with_chosen']
