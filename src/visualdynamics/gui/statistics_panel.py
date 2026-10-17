"""The time statistics of a record, beside the three rows that draw them.

The Statistics reading of a time history draws one channel's trace,
its histogram against the Gaussian and its kurtosis frame by frame
(`plot.time_statistics`). This panel holds what the reading is made
with, above the rule: the channel's range when it is known, how close
to it a sample counts as on the rail, the frame the kurtosis is read
over (sized from the record's bandwidth at its lowest), the duration the expected largest peak is for, and the band the
record's kurtosis is judged against, 2 to 4 as the kurtosis bars use
and settable tighter. Under the rule is the table of every number the
reading gives (`core.time_statistics`), and under the table the
judgement in words.

A statistic outside its band is set in bold and named in the words
under the table, never told apart by color alone. Every edit redraws
the rows at once. Nothing is made: the reading is of a record that
already exists, as the kurtosis bars are, so there is no Apply button.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QLabel, QWidget

from ..core.time_statistics import (
    KURTOSIS_BAND,
    RAIL_MARGIN,
    TimeStatistics,
)
from ..plot.time_statistics import percent
from .editors import DoubleSpinBox
from .settings_panel import add_derived, add_keyed_settings, add_rule, panel_grid

#: the table under the rule, as (key, name), in the order it reads.
#: The keys that are also judged statistics (`core.time_statistics.
#: STATISTICS`) are set in bold when their statistic is flagged.
ROWS = (('nonfinite', 'Not finite'),
        ('mean', 'Mean'), ('rms', 'RMS'), ('std', 'Std deviation'),
        ('skewness', 'Skewness'), ('kurtosis', 'Kurtosis'),
        ('crest', 'Crest factor'), ('peak', 'Largest peak'),
        ('peak_time', 'At time'), ('peak_sigmas', 'Largest / σ'),
        ('at_peak', 'At the peak'), ('rail', 'On the rail'),
        ('beyond_3', 'Beyond 3σ'), ('beyond_4', 'Beyond 4σ'),
        ('beyond_5', 'Beyond 5σ'), ('crossings', 'Crossings /s'),
        ('spectral_crossings', 'From spectrum /s'),
        ('expected_peak', 'Expected peak'), ('frames', 'Frames out'))

#: how the range field starts the first time it is ticked: unset, so
#: the first value is the user's, never a guess about the channel
NO_RANGE = 0.0

#: the frame field at its lowest: the frame sized from the record's
#: bandwidth (`core.time_statistics.FRAME_PRODUCT`)
FROM_BAND = 0.0


def _spin(low: float, high: float, value: float, decimals: int,
          tip: str, suffix: str = '') -> DoubleSpinBox:
    """A field that reports only a finished edit, since each report
    redraws three plots."""
    box = DoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setValue(value)
    box.setToolTip(tip)
    if suffix:
        box.setSuffix(suffix)
    return box


class StatisticsPanel(QWidget):
    """The settings and the table. Edits are emitted as `settings()`."""

    #: a setting moved; here is the whole set now
    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Statistics')
        grid = panel_grid(self, self.title)
        self._loading = False

        self.range_box: QCheckBox = QCheckBox()
        self.range_box.setToolTip('Count the samples on the channel\'s '
                                  'range')
        self.range_value: DoubleSpinBox = _spin(
            0.0, 1e12, NO_RANGE, 4, 'The channel\'s range, either way')
        self.margin_box: DoubleSpinBox = _spin(
            0.0, 10.0, 100.0 * RAIL_MARGIN, 3,
            'How close to the range, or the peak, counts as on it', ' %')
        self.frame_box: DoubleSpinBox = _spin(
            FROM_BAND, 1e6, FROM_BAND, 3,
            'Frame for the kurtosis against time. At its lowest, sized '
            'from the record\'s bandwidth', ' s')
        self.frame_box.setSpecialValueText('from band')
        self.duration_box: DoubleSpinBox = _spin(
            0.0, 1e9, 0.0, 1,
            'The run the expected largest peak is for. At its lowest, '
            'the record\'s length', ' s')
        self.duration_box.setSpecialValueText('record')
        self.low_box: DoubleSpinBox = _spin(
            0.0, 3.0, KURTOSIS_BAND[0], 2,
            'Below this the record\'s kurtosis is flagged')
        self.high_box: DoubleSpinBox = _spin(
            3.0, 100.0, KURTOSIS_BAND[1], 2,
            'Above this the record\'s kurtosis is flagged')
        for box in (self.low_box, self.high_box):
            box.setSingleStep(0.05)
        self.names, self.editors = add_keyed_settings(grid, [
            ('range', 'Range given', self.range_box),
            ('range_value', 'Range', self.range_value),
            ('margin', 'Margin', self.margin_box),
            ('frame', 'Frame', self.frame_box),
            ('duration', 'Duration', self.duration_box),
            ('low', 'Kurtosis low', self.low_box),
            ('high', 'Kurtosis high', self.high_box),
        ])
        add_rule(grid, 8)
        made = add_derived(grid, ROWS, 9)
        self.row_names: dict[str, QLabel] = {
            key: name for key, (name, _value) in made.items()}
        self.values: dict[str, QLabel] = {
            key: value for key, (_name, value) in made.items()}
        # the judgement in words: what is flagged, by name, so a flag
        # is read rather than seen in a color
        self.verdict: QLabel = QLabel('')
        self.verdict.setWordWrap(True)
        grid.addWidget(self.verdict, 9 + len(ROWS), 0, 1, 2)
        grid.setRowStretch(10 + len(ROWS), 1)

        self.range_box.toggled.connect(self._edited)
        for box in (self.range_value, self.margin_box, self.frame_box,
                    self.duration_box, self.low_box, self.high_box):
            box.valueChanged.connect(self._edited)
        self._shown_rows()

    def _shown_rows(self) -> None:
        """Only the rows that apply (principle 3): the range's value
        while it is ticked."""
        given = self.range_box.isChecked()
        for key in ('range_value',):
            self.names[key].setVisible(given)
            self.editors[key].setVisible(given)

    def _edited(self, *_args: Any) -> None:
        self._shown_rows()
        if not self._loading:
            self.changed.emit(self.settings())

    def set_unit(self, unit: str) -> None:
        """Name the range in the units the trace is drawn in."""
        self.names['range_value'].setText(f'Range [{unit}]' if unit
                                          else 'Range')

    def settings(self) -> dict[str, Any]:
        """The reading's settings: 'full_scale' (in the units the trace
        is drawn in, None while the range is unticked or unset),
        'margin' as a share, 'frame' and 'duration' in seconds ('frame'
        None to size it from the bandwidth, 'duration' None for the
        record's own length), and 'kurtosis_band'."""
        given = self.range_box.isChecked() and self.range_value.value() > 0
        duration = self.duration_box.value()
        frame = self.frame_box.value()
        return {'full_scale': self.range_value.value() if given else None,
                'margin': self.margin_box.value() / 100.0,
                'frame': frame if frame > FROM_BAND else None,
                'duration': duration if duration > 0.0 else None,
                'kurtosis_band': (self.low_box.value(),
                                  self.high_box.value())}

    def set_settings(self, settings: dict[str, Any] | None) -> None:
        """Set the fields from a `settings()` dict without a redraw per
        field; emits nothing, since the caller is about to draw."""
        if not settings:
            return
        self._loading = True
        try:
            full_scale = settings.get('full_scale')
            self.range_box.setChecked(full_scale is not None)
            if full_scale is not None:
                self.range_value.setValue(full_scale)
            self.margin_box.setValue(100.0 * settings.get('margin',
                                                          RAIL_MARGIN))
            self.frame_box.setValue(settings.get('frame') or FROM_BAND)
            self.duration_box.setValue(settings.get('duration') or 0.0)
            low, high = settings.get('kurtosis_band', KURTOSIS_BAND)
            self.low_box.setValue(low)
            self.high_box.setValue(high)
        finally:
            self._loading = False
        self._shown_rows()

    def show_statistics(self, stats: TimeStatistics | None,
                        transform: tuple[float, float] = (1.0, 0.0),
                        unit: str = '',
                        time: tuple[float, str] = (1.0, 's')) -> None:
        """Fill the table from `stats`, its values put into the display
        units by `transform` (the unit system's (scale, offset), SI =
        value * scale + offset) and labeled `unit`, its times and rates
        by `time`, (scale, label); a flagged statistic is set in bold,
        and the words under the table say what is flagged. None clears
        the table."""
        for key in self.values:
            self.values[key].setText('—')
            self._bold(key, False)
        if stats is None:
            self.verdict.setText('')
            return

        scale, offset = transform
        tscale, tunit = time

        def convert(value):
            return (float(value) - offset) / scale

        def size(value):
            # an rms and a spread are magnitudes, so they convert by
            # the unit's scale alone, without the offset a temperature
            # would carry
            return abs(float(value) / scale)

        after = f' {unit}' if unit else ''
        self.row_names['crossings'].setText(f'Crossings /{tunit}')
        self.row_names['spectral_crossings'].setText(
            f'From spectrum /{tunit}')
        text = {
            'nonfinite': f'{stats.nonfinite}',
            'mean': f'{convert(stats.mean):.4g}{after}',
            'rms': f'{size(stats.rms):.4g}{after}',
            'std': f'{size(stats.std):.4g}{after}',
            'skewness': f'{stats.skewness:.3f}',
            'kurtosis': f'{stats.kurtosis:.3f}',
            'crest': f'{stats.crest:.3f}',
            'peak': f'{convert(stats.peak):.4g}{after}',
            'peak_time': f'{stats.peak_time / tscale:.4g} {tunit}',
            'peak_sigmas': f'{stats.peak_sigmas:.2f}',
            'at_peak': f'{stats.at_peak}',
            'crossings': f'{stats.crossings * tscale:.4g}',
            'spectral_crossings':
                f'{stats.spectral_crossings * tscale:.4g}',
            'expected_peak': f'{stats.expected_peak:.2f} σ',
            'frames': (f'{stats.frames_outside} of '
                       f'{stats.frame_kurtosis.size}'),
        }
        if stats.rail is not None:
            text['rail'] = (f'{stats.rail} '
                            f'({percent(stats.rail_share or 0.0)})')
        for key, value in text.items():
            if key in self.values:
                self.values[key].setText(value)
        # the record's share in the table and the Gaussian's beside it
        # in the tip: both in the row ran past the panel's width at five
        # sigma, and the histogram's title states the two together
        for tail in stats.tails:
            key = f'beyond_{tail.sigmas:g}'
            if key in self.values:
                self.values[key].setText(percent(tail.share))
                tip = (f'{tail.count} samples; a Gaussian puts '
                       f'{percent(tail.expected)}'
                       + ('' if tail.judged else ', too few to judge'))
                self.values[key].setToolTip(tip)
                self.row_names[key].setToolTip(tip)
        self.values['expected_peak'].setToolTip(
            f'Over {stats.duration / tscale:.4g} {tunit}; over the '
            f'record, {stats.expected_peak_here:.2f} σ')
        self.values['frames'].setToolTip(
            f'Frames of {stats.frame_length / stats.sample_rate / tscale:.3g}'
            f' {tunit}')
        found = stats.flags()
        for key in self.values:
            self._bold(key, bool(found.get(key)))
        self.verdict.setText(stats.describe())
        font = self.verdict.font()
        font.setBold(bool(stats.flagged))
        self.verdict.setFont(font)

    def _bold(self, key: str, bold: bool) -> None:
        for label in (self.row_names[key], self.values[key]):
            font = label.font()
            font.setBold(bold)
            label.setFont(font)
        # the name is grayed like every derived row's until it is
        # flagged, when it is read as loudly as its value
        self.row_names[key].setEnabled(bold)

