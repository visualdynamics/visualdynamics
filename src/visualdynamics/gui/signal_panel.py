"""The Generate Signal pane beside the plot: the kind of signal and its
settings, a preview as they change, and the button that makes it.

It opens on the object the signal is drawn from: a spectrum (random,
burst random, periodic random at its DOFs), a sine specification (its
tone), or a channel table (any of those but sine at its reference
channels, the random kinds to a typed flat band and level, or an impulse
or a classical pulse). Only the kinds the object makes are offered, and
only the rows the kind reads are shown (principle 3).

This is select, see, set, apply (principle 13): the plot previews a
short stretch of the signal, its PSD over the target for the random
kinds and its trace for the others; Generate runs the project's
`generate_signal` with the settings shown, the seed included, so the
record can be made again. The panel makes nothing itself: it shows the
settings in display units and emits them in SI as a `SignalSettings`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QPushButton,
    QWidget,
)

from ..core.classical_pulse import SHAPES
from ..core.signals import KINDS, RANDOM_KINDS, SignalSettings
from .editors import DoubleSpinBox, SpinBox, number_box
from .settings_panel import add_derived, add_keyed_settings, add_rule, panel_grid

if TYPE_CHECKING:                                    # pragma: no cover
    from ..units import UnitSystem

#: which rows each kind reads; the others are put away
ROWS: dict[str, tuple[str, ...]] = {
    'random': ('rate', 'duration', 'frame', 'seed', 'correlated'),
    'burst': ('rate', 'frame', 'bursts', 'burst', 'taper', 'seed',
              'correlated'),
    'periodic': ('rate', 'frame', 'periods', 'repeats', 'seed', 'correlated'),
    'impulse': ('rate', 'duration', 'at', 'area'),
    'pulse': ('rate', 'duration', 'at', 'shape', 'peak', 'pulse', 'edge'),
    'sine': ('rate', 'tone'),
}

#: the rows a typed target adds to the random kinds at a channel table
TYPED_ROWS = ('low', 'high', 'level')


def _int_box(low: int, high: int, tip: str, special: str = '') -> SpinBox:
    box = SpinBox()
    box.setRange(low, high)
    box.setToolTip(tip)
    if special:
        box.setSpecialValueText(special)
    return box


class SignalPanel(QWidget):
    """The settings of the signal to generate, emitted whole as a
    `SignalSettings` when one changes and when Generate is pressed."""

    #: a setting moved; here are the settings now
    changed = Signal(object)

    #: Generate was pressed with these settings
    apply_asked = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = False
        self._unit_system = None
        #: the quantity the signal is, for its units ('force')
        self._quantity = ''
        #: whether the target is typed (a channel table) or read
        self._typed = False
        #: the settings as last shown, so a kind switched back keeps
        #: what was typed for it
        self._settings = SignalSettings()
        self.title: QLabel = QLabel('Generate Signal')
        grid = panel_grid(self, self.title)
        self.kind_box: QComboBox = QComboBox()
        self.kind_box.setToolTip('What kind of signal to make from the '
                                 'object selected')
        self.shape_box: QComboBox = QComboBox()
        for shape in SHAPES:
            self.shape_box.addItem(shape.capitalize(), shape)
        self.shape_box.setToolTip('The classical pulse: a half-sine, a '
                                  'terminal-peak sawtooth or a trapezoid')
        self.tone_box: QComboBox = QComboBox()
        self.tone_box.setToolTip("Which of the specification's tones to make")
        self.correlated_box: QCheckBox = QCheckBox('Cross terms')
        self.correlated_box.setToolTip(
            "Draw the spectrum's cross terms, so the DOFs are as coherent "
            'as it says; unticked, each DOF is drawn on its own')
        editors = {
            'kind': self.kind_box,
            'rate': number_box(1.0, 1e7, value=8192.0, decimals=1,
                               suffix=' Hz',
                               tip='Samples per second: above twice the '
                                   'highest frequency the signal holds'),
            'duration': number_box(1e-3, 1e5, value=10.0, decimals=3,
                                   suffix=' s', tip='Seconds of record'),
            'frame': _int_box(0, 2 ** 22,
                              'Samples per frame, which sets the line '
                              'spacing; for random, Auto chooses lines no '
                              'coarser than 1 Hz or a quarter of the '
                              'lowest frequency', special='Auto'),
            'bursts': _int_box(1, 100000, 'How many bursts, one per frame'),
            'burst': number_box(1.0, 100.0, value=50.0, decimals=0,
                                suffix=' %',
                                tip='The share of each frame the burst is on'),
            'taper': number_box(0.0, 49.0, value=5.0, decimals=0, suffix=' %',
                                tip='The share of the burst each cosine '
                                    'ramp takes, at either end'),
            'periods': _int_box(1, 100000, 'How many periods in all'),
            'repeats': _int_box(0, 100000,
                                'Periods per realization: a new random '
                                'frame every this many; All repeats one '
                                'frame throughout (pseudo-random)',
                                special='All'),
            'seed': _int_box(-1, 2 ** 31 - 1,
                             'The random seed: the same seed makes the '
                             'same record. Drawn picks one on Generate, '
                             'and it is kept with the signal',
                             special='Drawn'),
            'correlated': self.correlated_box,
            'low': number_box(0.001, 1e7, value=20.0, decimals=2,
                              suffix=' Hz', tip='Lowest frequency of the band'),
            'high': number_box(0.001, 1e7, value=2000.0, decimals=2,
                               suffix=' Hz', tip='Highest frequency of the band'),
            'level': number_box(1e-12, 1e12, value=1.0, decimals=6,
                                tip='The density across the band'),
            'at': number_box(0.0, 1e5, value=0.0, decimals=4, suffix=' s',
                             tip='When the impulse or the pulse starts'),
            'area': number_box(-1e12, 1e12, value=1.0, decimals=6,
                               tip="The impulse's integral"),
            'shape': self.shape_box,
            'peak': number_box(-1e12, 1e12, value=1.0, decimals=4,
                               tip="The pulse's peak, signed"),
            'pulse': number_box(0.001, 1e5, value=11.0, decimals=3,
                                suffix=' ms', tip="The pulse's duration"),
            'edge': number_box(0.0, 1e5, value=0.0, decimals=3, suffix=' ms',
                               tip="A sawtooth's fall, or a trapezoid's rise "
                                   'and fall, each'),
            'tone': self.tone_box,
        }
        self._names, self._editors = add_keyed_settings(grid, [
            ('kind', 'Kind', self.kind_box),
            ('rate', 'Sample rate', editors['rate']),
            ('duration', 'Duration', editors['duration']),
            ('frame', 'Frame', editors['frame']),
            ('bursts', 'Bursts', editors['bursts']),
            ('burst', 'On', editors['burst']),
            ('taper', 'Taper', editors['taper']),
            ('periods', 'Periods', editors['periods']),
            ('repeats', 'Realization', editors['repeats']),
            ('low', 'From', editors['low']),
            ('high', 'To', editors['high']),
            ('level', 'Level', editors['level']),
            ('at', 'Starts', editors['at']),
            ('area', 'Area', editors['area']),
            ('shape', 'Shape', self.shape_box),
            ('peak', 'Peak', editors['peak']),
            ('pulse', 'Duration', editors['pulse']),
            ('edge', 'Edge', editors['edge']),
            ('tone', 'Tone', self.tone_box),
            ('seed', 'Seed', editors['seed']),
            ('correlated', '', self.correlated_box),
        ])
        rows = len(self._editors) + 1
        add_rule(grid, rows)
        self._derived = add_derived(grid, [('samples', 'Samples'),
                                           ('lines', 'Line spacing'),
                                           ('rms', 'Preview rms')], rows + 1)
        self.note: QLabel = QLabel('')
        self.note.setWordWrap(True)
        grid.addWidget(self.note, rows + 4, 0, 1, 2)
        self.apply_button: QPushButton = QPushButton('Generate')
        self.apply_button.setToolTip('Make the signal shown, as a time '
                                     'history beside its source')
        self.apply_button.clicked.connect(self._apply)
        grid.addWidget(self.apply_button, rows + 5, 0, 1, 2)
        grid.setRowStretch(rows + 6, 1)
        for editor in self._editors.values():
            if isinstance(editor, QComboBox):
                editor.currentIndexChanged.connect(self._edited)
            elif isinstance(editor, QCheckBox):
                editor.toggled.connect(self._edited)
            elif isinstance(editor, (DoubleSpinBox, SpinBox)):
                editor.valueChanged.connect(self._edited)

    # ---- shown ----------------------------------------------------------

    def show_source(self, kinds: tuple[str, ...], settings: SignalSettings,
                    unit_system: UnitSystem, *, quantity: str,
                    typed: bool, cross_terms: bool,
                    tones: tuple[str, ...] = ()) -> None:
        """Put the settings for a source up: the kinds it makes, the
        settings in display units, and the rows the kind reads."""
        self._loading = True
        try:
            self._unit_system, self._quantity = unit_system, quantity
            self._typed = typed
            self._settings = settings
            self.kind_box.clear()
            for kind in kinds:
                self.kind_box.addItem(KINDS[kind], kind)
            self.kind_box.setCurrentIndex(max(0, self.kind_box.findData(
                settings.kind)))
            self.tone_box.clear()
            for tone in tones:
                self.tone_box.addItem(tone, tone)
            if settings.tone in tones:
                self.tone_box.setCurrentIndex(tones.index(settings.tone))
            self._cross_terms = cross_terms
            self._write(settings)
            self._show_rows()
        finally:
            self._loading = False

    def _write(self, s: SignalSettings) -> None:
        e = self._editors
        e['rate'].setValue(s.sample_rate)
        e['duration'].setValue(s.duration)
        e['frame'].setValue(int(s.frame_length or 0))
        e['bursts'].setValue(int(s.bursts))
        e['burst'].setValue(100.0 * s.burst)
        e['taper'].setValue(100.0 * s.taper)
        e['periods'].setValue(int(s.periods))
        e['repeats'].setValue(int(s.repeats or 0))
        e['seed'].setValue(-1 if s.seed is None else int(s.seed))
        self.correlated_box.setChecked(bool(s.correlated))
        e['low'].setValue(float(s.band[0]))
        e['high'].setValue(float(s.band[1]))
        e['level'].setValue(self._shown(s.level, self._density()))
        e['at'].setValue(s.at)
        e['area'].setValue(self._shown(s.area, self._area_dim()))
        self.shape_box.setCurrentIndex(max(0, self.shape_box.findData(s.shape)))
        e['peak'].setValue(self._shown(s.peak, self._quantity))
        e['pulse'].setValue(1e3 * s.pulse_duration)
        e['edge'].setValue(1e3 * (s.edge or 0.0))
        for key, dim in (('level', self._density()),
                         ('area', self._area_dim()),
                         ('peak', self._quantity)):
            e[key].setSuffix(f' {self._label(dim)}' if dim else '')

    def _density(self) -> str:
        return f'{self._quantity}**2/frequency' if self._quantity else ''

    def _area_dim(self) -> str:
        return f'{self._quantity}*time' if self._quantity else ''

    def _label(self, dim: str) -> str:
        if self._unit_system is None or not dim:
            return ''
        try:
            return self._unit_system.label_text(dim)
        except (KeyError, ValueError):
            return ''

    def _shown(self, value: float, dim: str) -> float:
        if self._unit_system is None or not dim:
            return float(value)
        try:
            return float(self._unit_system.from_si(value, dim))
        except (KeyError, ValueError):
            return float(value)

    def _si(self, value: float, dim: str) -> float:
        if self._unit_system is None or not dim:
            return float(value)
        try:
            return float(self._unit_system.to_si(value, dim))
        except (KeyError, ValueError):
            return float(value)

    def _show_rows(self) -> None:
        kind = self.kind_box.currentData() or 'random'
        wanted = set(ROWS.get(kind, ()))
        if self._typed and kind in RANDOM_KINDS:
            wanted |= set(TYPED_ROWS)
        if not getattr(self, '_cross_terms', False):
            wanted.discard('correlated')
        if self.tone_box.count() < 2:
            wanted.discard('tone')
        wanted.add('kind')
        for key, editor in self._editors.items():
            shown = key in wanted and not (key == 'kind'
                                           and self.kind_box.count() < 2)
            editor.setVisible(shown)
            self._names[key].setVisible(shown)

    def show_reading(self, samples: int | None = None,
                     spacing: float | None = None,
                     rms: str | None = None, note: str = '') -> None:
        """The derived rows and the note under them, after a preview."""
        values = {'samples': '—' if samples is None else f'{samples:,}',
                  'lines': '—' if spacing is None else f'{spacing:.4g} Hz',
                  'rms': rms or '—'}
        for key, (name, value) in self._derived.items():
            value.setText(values[key])
            shown = values[key] != '—'
            name.setVisible(shown)
            value.setVisible(shown)
        self.note.setText(note)
        self.note.setVisible(bool(note))

    # ---- read back ------------------------------------------------------

    def settings(self) -> SignalSettings:
        """The settings as shown, in SI."""
        e = self._editors
        seed = int(e['seed'].value())
        repeats = int(e['repeats'].value())
        frame = int(e['frame'].value())
        edge = 1e-3 * float(e['edge'].value())
        return replace(
            self._settings,
            kind=self.kind_box.currentData() or self._settings.kind,
            sample_rate=float(e['rate'].value()),
            duration=float(e['duration'].value()),
            frame_length=frame or None,
            bursts=int(e['bursts'].value()),
            burst=float(e['burst'].value()) / 100.0,
            taper=float(e['taper'].value()) / 100.0,
            periods=int(e['periods'].value()),
            repeats=repeats or None,
            seed=None if seed < 0 else seed,
            correlated=bool(self.correlated_box.isChecked()),
            band=(float(e['low'].value()), float(e['high'].value())),
            level=self._si(float(e['level'].value()), self._density()),
            at=float(e['at'].value()),
            area=self._si(float(e['area'].value()), self._area_dim()),
            shape=self.shape_box.currentData() or 'half-sine',
            peak=self._si(float(e['peak'].value()), self._quantity),
            pulse_duration=1e-3 * float(e['pulse'].value()),
            edge=edge or None,
            tone=self.tone_box.currentData())

    def _edited(self, *_args) -> None:
        if self._loading:
            return
        self._show_rows()
        self._settings = self.settings()
        self.changed.emit(self._settings)

    def _apply(self) -> None:
        self._settings = self.settings()
        self.apply_asked.emit(self._settings)
