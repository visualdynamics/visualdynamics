"""The multi-point reading's settings, beside a random run's control
channels drawn with their composite.

The Composite reading of a random comparison combines the control
channels into the composite a controller holds and judges the
composite against the composite band and each channel against its own
(`plot.multi_point`). This panel holds what that reading is made
under (the strategy, the weights under a weighted average, and the two
bands, each a pair of the specification's limits or a pair of dB) and,
under the rule, what came of it: the composite's lines outside its
band, then each channel's lines outside its own and how many of those
the composite was inside at, then the notes.

Every edit redraws the plot at once, so the effect of a strategy or a
band is seen on the curves while it is set. Nothing is made: the
reading is of two objects that already exist, as the RMS bars are, so
there is no Apply button to press. The same call a script makes,
`core.multi_point.multi_point`, gives every number shown here.
"""

from __future__ import annotations

import re
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QLineEdit, QWidget

from ..core.multi_point import COMPOSITE_DB, INDIVIDUAL_DB
from .editors import DoubleSpinBox, number_box
from .settings_panel import add_derived, add_keyed_settings, add_rule, panel_grid

#: the strategies as the box names them, in `STRATEGIES` order
STRATEGY_NAMES = (('average', 'Average'), ('weighted', 'Weighted'),
                  ('maximum', 'Maximum'), ('minimum', 'Minimum'))

#: what a band can be, as its box names it
BAND_NAMES = (('warning', 'Spec warning limits'),
              ('abort', 'Spec abort limits'),
              ('db', 'dB band'))


def _spin(low: float, high: float, value: float, tip: str) -> DoubleSpinBox:
    """A dB field in half-decibel steps, reporting only a finished edit,
    since each report redraws the reading."""
    return number_box(low, high, value=value, step=0.5, tip=tip)


def parse_weights(text: str) -> tuple[float, ...] | None:
    """The weights typed, numbers separated by commas or spaces; None
    for an empty field.

    Raises
    ------
    ValueError
        For a field holding anything that is not a number.
    """
    parts = [part for part in re.split(r'[,\s;]+', text.strip()) if part]
    if not parts:
        return None
    try:
        return tuple(float(part) for part in parts)
    except ValueError:
        raise ValueError('the weights are numbers, one per channel, '
                         'separated by commas') from None


class MultiPointPanel(QWidget):
    """The settings and the readout. Edits are emitted as `settings()`."""

    #: a setting moved; here is the whole set now
    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Composite')
        grid = panel_grid(self, self.title)
        self._loading = False
        #: the specification the band boxes were filled for
        self._offered_for: Any = None
        #: how many channels the weights field was seeded for
        self._count = 0

        self.strategy_box: QComboBox = QComboBox()
        for key, name in STRATEGY_NAMES:
            self.strategy_box.addItem(name, key)
        self.strategy_box.setToolTip(
            'How the control channels are combined, line by line. '
            'Average: the mean of their densities. Weighted: the mean '
            'with the weights below. Maximum: the highest channel at '
            'each line. Minimum: the lowest. The test plan says which')
        self.weights_edit: QLineEdit = QLineEdit()
        self.weights_edit.setToolTip(
            'One weight per channel, in the order the channels are '
            'listed below, separated by commas; they are scaled to add '
            'up to one')
        self.composite_box: QComboBox = QComboBox()
        self.composite_box.setToolTip(
            'The band the composite is judged against: one of the '
            'specification\'s pairs of limits, or a band in dB')
        self.composite_low: DoubleSpinBox = _spin(
            -40.0, 0.0, COMPOSITE_DB[0],
            'The composite band\'s lower edge, dB below the reference')
        self.composite_high: DoubleSpinBox = _spin(
            0.0, 40.0, COMPOSITE_DB[1],
            'The composite band\'s upper edge, dB above the reference')
        self.individual_box: QComboBox = QComboBox()
        self.individual_box.setToolTip(
            'The band each control channel is judged against on its '
            'own, usually wider than the composite\'s')
        self.individual_low: DoubleSpinBox = _spin(
            -40.0, 0.0, INDIVIDUAL_DB[0],
            'The individual band\'s lower edge, dB below the reference')
        self.individual_high: DoubleSpinBox = _spin(
            0.0, 40.0, INDIVIDUAL_DB[1],
            'The individual band\'s upper edge, dB above the reference')
        self.names, self.editors = add_keyed_settings(grid, [
            ('strategy', 'Strategy', self.strategy_box),
            ('weights', 'Weights', self.weights_edit),
            ('composite', 'Composite band', self.composite_box),
            ('composite_low', 'Below [dB]', self.composite_low),
            ('composite_high', 'Above [dB]', self.composite_high),
            ('individual', 'Channel band', self.individual_box),
            ('individual_low', 'Below [dB]', self.individual_low),
            ('individual_high', 'Above [dB]', self.individual_high),
        ])
        add_rule(grid, 9)
        self.derived: dict[str, QLabel] = {
            key: value for key, (_name, value) in add_derived(
                grid, (('channels', 'Channels'),
                       ('lines', 'Lines judged')), 10).items()}
        # a line for the composite, one per channel and one per note,
        # as many as there are, so a label that wraps
        self.readout: QLabel = QLabel('')
        self.readout.setWordWrap(True)
        grid.addWidget(self.readout, 12, 0, 1, 2)
        grid.setRowStretch(13, 1)
        self._fill_bands(False, False)

        self.strategy_box.currentIndexChanged.connect(self._edited)
        self.composite_box.currentIndexChanged.connect(self._edited)
        self.individual_box.currentIndexChanged.connect(self._edited)
        self.weights_edit.editingFinished.connect(self._edited)
        for box in (self.composite_low, self.composite_high,
                    self.individual_low, self.individual_high):
            box.valueChanged.connect(self._edited)
        self._shown_rows()

    def _fill_bands(self, warning: bool, abort: bool) -> None:
        """Offer in each band box the pairs the specification carries,
        and the dB band always."""
        for box in (self.composite_box, self.individual_box):
            box.clear()
            for key, name in BAND_NAMES:
                if key == 'db' or (key == 'warning' and warning) \
                        or (key == 'abort' and abort):
                    box.addItem(name, key)

    @staticmethod
    def _choose(box: QComboBox, key: str) -> None:
        index = box.findData(key)
        if index >= 0:
            box.setCurrentIndex(index)

    def _shown_rows(self) -> None:
        """Show only the rows that apply (principle 3): the weights under
        the weighted strategy, a band's dB fields while it is a band in
        dB."""
        weighted = self.strategy() == 'weighted'
        composite_db = self.composite_box.currentData() == 'db'
        individual_db = self.individual_box.currentData() == 'db'
        for key, shown in (('weights', weighted),
                           ('composite_low', composite_db),
                           ('composite_high', composite_db),
                           ('individual_low', individual_db),
                           ('individual_high', individual_db)):
            self.names[key].setVisible(shown)
            self.editors[key].setVisible(shown)

    def _edited(self, *_args: Any) -> None:
        self._shown_rows()
        if not self._loading:
            self.changed.emit(self.settings())

    def strategy(self) -> str:
        """The strategy chosen, one of `STRATEGIES`."""
        return str(self.strategy_box.currentData())

    def _band(self, box: QComboBox, low: DoubleSpinBox,
              high: DoubleSpinBox) -> str | tuple[float, float]:
        key = box.currentData()
        return (low.value(), high.value()) if key == 'db' else str(key)

    def settings(self) -> dict[str, Any]:
        """The reading's arguments as `multi_point` takes them:
        'strategy', 'weights' (None unless weighted; the field's text
        when it holds something that is not a list of numbers, for the
        window to refuse), 'composite_band' and 'individual_band'."""
        weights: Any = None
        if self.strategy() == 'weighted':
            try:
                weights = parse_weights(self.weights_edit.text())
            except ValueError:
                weights = self.weights_edit.text()
        return {'strategy': self.strategy(), 'weights': weights,
                'composite_band': self._band(self.composite_box,
                                             self.composite_low,
                                             self.composite_high),
                'individual_band': self._band(self.individual_box,
                                              self.individual_low,
                                              self.individual_high)}

    def offer_bands(self, key: Any, warning: bool, abort: bool,
                    channels: int) -> None:
        """Offer a specification's own limits as the bands, without a
        redraw: `key` names the specification, `warning` and `abort`
        say which pairs it carries, and `channels` is how many control
        channels are combined. A specification new to the panel starts
        the composite on its warning limits (its abort limits, or a
        band in dB, where it lacks them) and the channels on a band in
        dB; the same key offered again keeps what was set. The weights
        field is seeded with equal weights whenever the count changes."""
        self._loading = True
        try:
            if key != self._offered_for:
                self._offered_for = key
                self._fill_bands(warning, abort)
                self._choose(self.composite_box,
                             'warning' if warning else
                             'abort' if abort else 'db')
                self._choose(self.individual_box, 'db')
            if channels != self._count:
                self._count = channels
                self.weights_edit.setText(', '.join(['1'] * channels))
        finally:
            self._loading = False
        self._shown_rows()

    def set_settings(self, strategy: str | None = None,
                     weights: tuple[float, ...] | None = None,
                     composite_band: Any = None,
                     individual_band: Any = None) -> None:
        """Set the fields without a redraw per field, then emit once. A
        band is a key of the specification's limits or a pair of dB."""
        self._loading = True
        try:
            if strategy is not None:
                self._choose(self.strategy_box, strategy)
            if weights is not None:
                self.weights_edit.setText(', '.join(f'{w:g}' for w in weights))
            for band, box, low, high in (
                    (composite_band, self.composite_box,
                     self.composite_low, self.composite_high),
                    (individual_band, self.individual_box,
                     self.individual_low, self.individual_high)):
                if band is None:
                    continue
                if isinstance(band, str):
                    self._choose(box, band)
                else:
                    self._choose(box, 'db')
                    low.setValue(band[0])
                    high.setValue(band[1])
        finally:
            self._loading = False
        self._edited()

    def show_results(self, reading: Any = None,
                     reason: str | None = None) -> None:
        """Write the reading: the composite's line
        (`plot.multi_point.describe_composite`), each channel's
        (`describe_channel`) and the notes. With no reading, `reason`
        says why there is none."""
        from ..plot.multi_point import describe_channel, describe_composite

        if reading is None:
            self.derived['channels'].setText('—')
            self.derived['lines'].setText('—')
            self.readout.setText(reason or '')
            return
        self.derived['channels'].setText(str(len(reading.channels)))
        self.derived['lines'].setText(str(reading.result.get('lines', 0)))
        lines = [describe_composite(reading)]
        lines += [describe_channel(channel, reading)
                  for channel in reading.channels]
        lines += list(reading.notes)
        self.readout.setText('\n'.join(lines))
