"""Each channel against its range, beside the trace and the bars that
draw it.

The Overload reading of a time history reads every channel against the
range its linked channel table gives it, the table's Range in volts over
its Sensitivity (`core.overload`), and draws the worst channel's trace
over its range with the headroom bars beside the kurtosis bars
(`plot.overload`). This panel says which table the ranges came from, or
that no linked table describes the record, and holds what the reading
is made with above the rule: the step to the next level in dB and how
close to the range a sample counts as on it. Under the rule is a row a
channel, its headroom in dB and its samples on the rail, then the
numbers for the channel the trace draws, and under them the judgment
in words.

A channel that is flagged is set in bold in the table and named in the
words, never told apart by color alone. Clicking a channel's row draws
its trace. Every edit redraws at once. Nothing is made: the reading is
of a record and a table that already exist, as the kurtosis bars are,
so there is no Apply button.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from ..core.overload import STEP_DB, ChannelOverload, OverloadCheck
from ..core.time_statistics import RAIL_MARGIN
from ..plot.overload import finding
from ..plot.time_statistics import percent
from .editors import DoubleSpinBox, number_box
from .settings_panel import add_derived, add_keyed_settings, add_rule, panel_grid

#: the numbers for the channel the trace draws, as (key, name)
ROWS = (('channel', 'Channel'), ('range', 'Range'), ('peak', 'Largest peak'),
        ('rms', 'RMS'), ('crest', 'Crest factor'),
        ('headroom', 'Headroom'), ('rail', 'On the rail'),
        ('rate', 'Rail per second'), ('next_peak', 'Peak at next level'),
        ('next_headroom', 'Headroom there'), ('kurtosis', 'Kurtosis'))

#: what the panel says when no linked channel table describes the record
NO_TABLE = ('No channel table linked with this record describes its '
            'channels. Link one: its Range and Sensitivity columns give '
            'each channel its range.')


class OverloadPanel(QWidget):
    """The source, the settings, the table and the numbers. Edits are
    emitted as `settings()`; a channel's row clicked as `picked`."""

    #: a setting moved; here is the whole set now
    changed = Signal(dict)
    #: a channel's row was clicked: its index in the record
    picked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Overload')
        grid = panel_grid(self, self.title)
        self._loading = False
        #: the channel each table row is, by its index in the record
        self._records: list[int] = []

        self.source: QLabel = QLabel('')
        self.source.setWordWrap(True)
        grid.addWidget(self.source, 1, 0, 1, 2)
        self.step_box: DoubleSpinBox = number_box(
            0.0, 40.0, value=STEP_DB, decimals=1, step=1.0,
            tip='The step to the next level of the test: each channel\'s '
                'largest peak is scaled up by it and set against the range',
            suffix=' dB')
        self.margin_box: DoubleSpinBox = number_box(
            0.0, 10.0, value=100.0 * RAIL_MARGIN, decimals=3,
            tip='How close to the range a sample counts as on the rail',
            suffix=' %')
        self.names, self.editors = add_keyed_settings(grid, [
            ('step', 'Next level', self.step_box),
            ('margin', 'Margin', self.margin_box),
        ], 2)
        add_rule(grid, 4)
        self.table: QTableWidget = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['Channel', 'Headroom',
                                              'On rail'])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(120)
        self.table.setToolTip(
            'A row a channel: the headroom of its range over its largest '
            'peak in dB, and its samples on the rail. A flagged channel '
            'is in bold. Click a row to draw its trace')
        grid.addWidget(self.table, 5, 0, 1, 2)
        made = add_derived(grid, ROWS, 6)
        self.values: dict[str, QLabel] = {
            key: value for key, (_name, value) in made.items()}
        self.row_names: dict[str, QLabel] = {
            key: name for key, (name, _value) in made.items()}
        # the judgment in words: which channels, for what
        self.verdict: QLabel = QLabel('')
        self.verdict.setWordWrap(True)
        grid.addWidget(self.verdict, 6 + len(ROWS), 0, 1, 2)
        grid.setRowStretch(7 + len(ROWS), 1)

        for box in (self.step_box, self.margin_box):
            box.valueChanged.connect(self._edited)
        self.table.cellClicked.connect(self._clicked)

    def _edited(self, *_args: Any) -> None:
        if not self._loading:
            self.changed.emit(self.settings())

    def _clicked(self, row: int, _column: int) -> None:
        if 0 <= row < len(self._records):
            self.picked.emit(self._records[row])

    def settings(self) -> dict[str, Any]:
        """The reading's settings: 'step_db', the step to the next
        level in dB, and 'margin' as a share."""
        return {'step_db': self.step_box.value(),
                'margin': self.margin_box.value() / 100.0}

    def set_settings(self, settings: dict[str, Any] | None) -> None:
        """Set the fields from a `settings()` dict without a redraw per
        field; emits nothing, since the caller is about to draw."""
        if not settings:
            return
        self._loading = True
        try:
            self.step_box.setValue(settings.get('step_db', STEP_DB))
            self.margin_box.setValue(100.0 * settings.get('margin',
                                                          RAIL_MARGIN))
        finally:
            self._loading = False

    def show_check(self, check: OverloadCheck | None,
                   channel: ChannelOverload | None = None,
                   transform: tuple[float, float] = (1.0, 0.0),
                   unit: str = '', source: str | None = None) -> None:
        """Fill the panel from `check`: the source named, a table row a
        channel, the numbers for `channel` (the one the trace draws)
        put into the display units by `transform`'s scale (SI = value
        * scale + offset; a range and a peak are sizes, so the offset
        does not apply) and labeled `unit`, and the judgment in words.
        None clears it and says no linked table describes the record."""
        self.table.setRowCount(0)
        self._records = []
        for value in self.values.values():
            value.setText('—')
        if check is None:
            self.source.setText(NO_TABLE)
            self.verdict.setText('')
            return
        self.source.setText(f'Ranges from {source}' if source
                            else 'Ranges as given')
        for row, item in enumerate(check.channels):
            self.table.insertRow(row)
            self._records.append(item.record)
            if item.has_range:
                headroom = f'{item.headroom_db:.1f} dB'
                rail = f'{item.rail}'
            else:
                headroom = 'not read' if item.statistics is None else 'no range'
                rail = '—'
            cells = (item.label, headroom, rail)
            loud = bool(item.flagged)
            missing = 'Not read' if item.statistics is None else 'No range'
            tip = (finding(item) if loud
                   else item.reason and f'{missing}: {item.reason}')
            if item.has_range and item.cautions():
                # what to doubt about the range itself, on its row
                tip = '\n'.join([tip] * bool(tip) + [
                    f'{c[0].upper()}{c[1:]}.' for c in item.cautions()])
            for column, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                if column:
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                font = cell.font()
                font.setBold(loud)
                cell.setFont(font)
                if tip:
                    cell.setToolTip(tip)
                self.table.setItem(row, column, cell)
            if channel is not None and item.record == channel.record:
                self.table.selectRow(row)
        self._show_channel(channel, transform[0], unit)
        self.verdict.setText(check.describe())
        font = self.verdict.font()
        font.setBold(bool(check.flagged))
        self.verdict.setFont(font)

    def _show_channel(self, channel: ChannelOverload | None, scale: float,
                      unit: str) -> None:
        """The numbers for the channel the trace draws."""
        if channel is None:
            return
        after = f' {unit}' if unit else ''
        text = {'channel': channel.label}
        stats = channel.statistics
        if stats is not None:
            text.update({
                'peak': f'{channel.peak / scale:.4g}{after}',
                'rms': f'{channel.rms / scale:.4g}{after}',
                'crest': f'{channel.crest:.3f}',
                'kurtosis': f'{stats.kurtosis:.3f}',
            })
        if channel.has_range:
            text.update({
                'range': (f'{channel.full_scale / scale:.4g}{after}'
                          + (', the default' if channel.on_default else '')),
                'headroom': f'{channel.headroom_db:.2f} dB',
                'rail': (f'{channel.rail} '
                         f'({percent(channel.rail_share or 0.0)})'),
                'rate': f'{channel.rail_rate:.3g}',
                'next_peak': f'{channel.next_peak / scale:.4g}{after}',
                'next_headroom': f'{channel.next_headroom_db:.2f} dB',
            })
        else:
            text['range'] = 'none' if channel.statistics is not None \
                else 'not read'
        self.row_names['next_peak'].setText(
            f'Peak at +{channel.step_db:g} dB')
        for key, value in text.items():
            self.values[key].setText(value)
        self.values['range'].setToolTip(
            '\n'.join(f'{c[0].upper()}{c[1:]}.' for c in channel.cautions())
            if channel.has_range else
            f'{"Not read" if channel.statistics is None else "No range"}: '
            f'{channel.reason}')
