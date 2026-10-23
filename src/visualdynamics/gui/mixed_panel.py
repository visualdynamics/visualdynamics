"""The mixed-mode readout: a sine-on-random or random-on-random record's
data sheet, read in the sine view.

A random and sine project holds two requirements for one record, the
random specification of the floor and the sine specification of the
tones, and judges them in two reports. When the project beside the
record holds both, this readout sits under the extraction's settings
and reads them as one test (`core.mixed_mode.judge_mixed`): the tones
tracked and taken out, the floor judged on what is left with the lines
around each tone left out, each narrowband at its own degrees of
freedom, and the total reported unjudged. It shows the data sheet, one
row for the floor, the tones together and each tone in amplitude, each
narrowband and the total, each with its tolerance and verdict, the
settings that decide what the numbers mean, and a small spectrum of the
residual over the reference with the left-out lines hatched, the same
drawing `plot_mixed_mode` makes (`plot.mixed_mode.draw_mixed_mode`).

The channel, the width left out either side of a tone, the tracking
band and the tones' tolerance are set here; the narrowbands are the
floor's own (`Specification.narrowbands`). Reading a long record takes
seconds, so the first reading of a record waits for the button; once
the sheet is up, an edit reads again at once. It is a reading, not an
act: nothing is made, and every number shown is the reading's.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from ..core.mixed_mode import EXCLUSION, TONE_TOLERANCE, TRACKING_BAND
from .editors import DoubleSpinBox, commit_on_enter
from .settings_panel import PreviewPlot, add_keyed_settings


class MixedPlot(PreviewPlot):
    """The residual over the reference, small, the left-out lines
    hatched: log on both axes, as a density is read."""

    HEIGHT = 170

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.items: list[Any] = []

    def show_reading(self, reading: Any, unit_system: Any) -> None:
        """Draw `reading`, or clear the plot when it is None."""
        from ..plot.mixed_mode import draw_mixed_mode

        for item in self.items:
            self.plot.removeItem(item)
        self.items = []
        if reading is not None:
            self.items = draw_mixed_mode(self.plot, reading,
                                         unit_system=unit_system,
                                         colors=self.colors, legend=False,
                                         markers=False)


class MixedPanel(QWidget):
    """The settings, the button, then the data sheet and its spectrum."""

    #: the sheet was read: (history, floor, tones, reading, settings), so
    #: the window can journal the call that draws it
    read = Signal(object, object, object, object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setFrameShadow(QFrame.Shadow.Sunken)
        grid.addWidget(rule, 0, 0, 1, 2)
        self.title: QLabel = QLabel('Mixed Mode')
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        grid.addWidget(self.title, 1, 0, 1, 2)
        #: what is being read, and the reading last made of it
        self._history: Any = None
        self._floor: Any = None
        self._tones: Any = None
        self._unit_system: Any = None
        self._key: tuple | None = None
        self._colors: Mapping[str, str] | None = None
        self.reading: Any = None
        self._loading = False

        self.channel_box: QComboBox = QComboBox()
        self.channel_box.setToolTip(
            'The control channel read: one the floor is written at and '
            'the record holds')
        self.exclusion_box: DoubleSpinBox = DoubleSpinBox()
        self.exclusion_box.setRange(0.0, 100.0)
        self.exclusion_box.setDecimals(2)
        self.exclusion_box.setSingleStep(0.5)
        self.exclusion_box.setPrefix('±')
        self.exclusion_box.setSuffix(' Hz')
        self.exclusion_box.setValue(EXCLUSION)
        self.exclusion_box.setToolTip(
            'The width left out of the floor either side of each tone: '
            'wide enough to hold the window\'s spread of a tone at the '
            'line spacing. A floor feature inside it is not judged')
        self.band_box: DoubleSpinBox = DoubleSpinBox()
        self.band_box.setRange(0.01, 100.0)
        self.band_box.setDecimals(2)
        self.band_box.setSingleStep(0.1)
        self.band_box.setSuffix(' Hz')
        self.band_box.setValue(TRACKING_BAND)
        self.band_box.setToolTip(
            'The tracking band each tone is read through. A narrower band '
            'admits less of the floor into a reading and takes longer to '
            'settle, about one over its width')
        self.tolerance_box: DoubleSpinBox = DoubleSpinBox()
        self.tolerance_box.setRange(0.1, 100.0)
        self.tolerance_box.setDecimals(1)
        self.tolerance_box.setSingleStep(1.0)
        self.tolerance_box.setPrefix('±')
        self.tolerance_box.setSuffix(' %')
        self.tolerance_box.setValue(TONE_TOLERANCE)
        self.tolerance_box.setToolTip(
            'The tolerance on each tone\'s amplitude, an example of what '
            'test documents ask for')
        self.names, self.editors = add_keyed_settings(grid, [
            ('channel', 'Channel', self.channel_box),
            ('exclusion', 'Left out', self.exclusion_box),
            ('band', 'Tracking band', self.band_box),
            ('tone_tolerance', 'Tone tolerance', self.tolerance_box)],
            first_row=2)
        self.read_button: QPushButton = QPushButton('Read Data Sheet')
        self.read_button.setToolTip(
            'Track the tones and take them out, then judge the floor, '
            'each tone, each narrowband and the total, each by its own '
            'rule')
        self.read_button.clicked.connect(self._read)
        grid.addWidget(self.read_button, 6, 0, 1, 2)
        #: the data sheet, as rich text
        self.sheet: QLabel = QLabel('')
        self.sheet.setTextFormat(Qt.TextFormat.RichText)
        self.sheet.setWordWrap(True)
        self.sheet.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sheet.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        grid.addWidget(self.sheet, 7, 0, 1, 2)
        self.preview: MixedPlot = MixedPlot()
        self.preview.hide()
        grid.addWidget(self.preview, 8, 0, 1, 2)
        commit_on_enter(self.exclusion_box, self.band_box, self.tolerance_box)
        self.channel_box.currentIndexChanged.connect(self._edited)
        for box in (self.exclusion_box, self.band_box, self.tolerance_box):
            box.valueChanged.connect(self._edited)

    # ---- what is being read -------------------------------------------------

    def show_mixed(self, history: Any, floor: Any, tones: Any,
                   unit_system: Any = None) -> Any:
        """Point the readout at `history`, read against `floor` (a random
        specification) and `tones` (a sine specification, or None); hide,
        and return None, when there is no floor or it shares no channel
        with the record (principle 3). Shows the reading already held
        for the same record and settings; otherwise waits for the
        button. Returns the reading shown, or None."""
        dofs = [str(dof) for dof in history.response_dof]
        if floor is None:
            self.hide()
            self.reading = None
            return None
        channels = [str(dof) for dof in floor.response_dof if str(dof) in dofs]
        if not channels:
            self.hide()
            self.reading = None
            return None
        held = self.channel_box.currentText()
        same = (history is self._history and floor is self._floor
                and tones is self._tones)
        self._loading = True
        try:
            if [self.channel_box.itemText(i)
                    for i in range(self.channel_box.count())] != channels:
                self.channel_box.clear()
                self.channel_box.addItems(channels)
            self.channel_box.setCurrentIndex(channels.index(held)
                                             if held in channels else 0)
            if not same:
                width = getattr(floor, 'tone_exclusion', None)
                self.exclusion_box.setValue(EXCLUSION if width is None
                                            else float(width))
        finally:
            self._loading = False
        self._history, self._floor, self._tones = history, floor, tones
        self._unit_system = unit_system
        self.show()
        if self._key is not None and self._key == self._key_now():
            self._show(self.reading)
        else:
            self.reading = None
            self._key = None
            self._show(None, 'Press Read Data Sheet to judge the floor, the '
                       'tones and the narrowbands, each by its own rule.')
        return self.reading

    def settings(self) -> dict[str, Any]:
        """The reading's settings: 'channel', 'exclusion' (Hz either
        side), 'band' (the tracking band's width, Hz) and
        'tone_tolerance' (percent)."""
        return {'channel': self.channel_box.currentText() or None,
                'exclusion': round(self.exclusion_box.value(), 6),
                'band': round(self.band_box.value(), 6),
                'tone_tolerance': round(self.tolerance_box.value(), 6)}

    def _key_now(self) -> tuple:
        settings = self.settings()
        return (id(self._history), id(getattr(self._history, 'ordinate',
                                              None)),
                id(self._floor), id(self._tones),
                *sorted(settings.items()))

    # ---- the reading ------------------------------------------------------

    def _edited(self, *_args: Any) -> None:
        """A setting moved: read again at once if a sheet is up, since
        the reader is already looking at it; otherwise wait for the
        button."""
        if self._loading or self._history is None:
            return
        if self.reading is not None:
            self._read()

    def _read(self) -> None:
        """Read the sheet with the settings shown, unless the same
        reading is already held, and show it."""
        from ..core.mixed_mode import MixedSpecification, judge_mixed
        from ..core.sine_tracking import SineTracking

        if self._history is None:
            return
        settings = self.settings()
        key = self._key_now()
        if key != self._key or self.reading is None:
            try:
                spec = MixedSpecification(self._floor, self._tones,
                                          exclusion=settings['exclusion'])
                self.reading = judge_mixed(
                    self._history, spec, channel=settings['channel'],
                    band=SineTracking(fixed=settings['band']),
                    tone_tolerance=settings['tone_tolerance'])
            except ValueError as refusal:
                self.reading = None
                self._key = None
                self._show(None, str(refusal))
                return
            self._key = key
        self._show(self.reading)
        self.read.emit(self._history, self._floor, self._tones,
                       self.reading, settings)

    def _show(self, reading: Any, words: str = '') -> None:
        """The sheet and its spectrum, or `words` in their place."""
        from ..plot.mixed_mode import sheet_html
        from ..theme import theme as resolve_theme

        colors = self._colors or resolve_theme(None)
        if reading is None:
            self.sheet.setText(words)
            self.preview.show_reading(None, self._unit_system)
            self.preview.hide()
            return
        self.sheet.setText(sheet_html(reading, colors['plot_foreground'],
                                      self._unit_system))
        self.preview.show()
        self.preview.show_reading(reading, self._unit_system)

    def apply_theme(self, colors: Mapping[str, str]) -> None:
        """Redraw the small plot and the sheet in a new theme's colors."""
        self._colors = colors
        self.preview.apply_theme(colors)
        if self.reading is not None:
            self._show(self.reading)


__all__ = ['MixedPanel', 'MixedPlot']
