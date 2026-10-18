"""The segment marks and their tolerances, beside a replicated run judged
segment by segment.

The Segments reading of a transient replication draws the run over its
reference with each marked segment shaded and judged by the rule for
its kind (`plot.replication_segments`). This panel holds the marks, a
row each (where the segment starts and ends on the reference's clock,
and whether it is stationary, transient or a shock), the tolerances
each kind is held to and, under them, what came of it: the
whole-record rms error, then each segment's verdict and the checks it
failed.

Every edit redraws the plot at once, so a mark or a tolerance is seen
on the record while it is set. Nothing is made: the judgment is a
reading of two objects that already exist, as the waveform error bars
are, so there is no Apply button to press. The same call a script
makes, `core.replication_segments.replication_segments`, gives every
number shown here.

The marks are kept on the reference as its `segments`, so they are
saved with the project and come back when the reference is shown again.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.replication_segments import (
    ASD_DB,
    ENVELOPE_DB,
    PEAK_PERCENT,
    SEGMENT_KINDS,
    SRS_DB,
    TAPER_PERCENT,
    VARIANCE_PERCENT,
    Segment,
)
from .editors import DoubleSpinBox, number_box
from .icons import control_icon
from .settings_panel import (
    PANEL_WIDTH,
    add_keyed_settings,
    add_rule,
    panel_grid,
)

#: the length a new segment is given, in seconds, when the record leaves
#: room for it
NEW_SECONDS = 1.0

#: the test band offered when it is first ticked, in Hz
BAND_HZ = (5.0, 400.0)

#: which settings rows apply to which kinds of segment; the taper and
#: the band apply to every kind and are always shown
ROWS_FOR = {'asd': ('stationary', 'transient'),
            'variance': ('stationary',),
            'envelope': ('transient',), 'window': ('transient',),
            'srs_low': ('shock',), 'srs_high': ('shock',),
            'peaks': ('shock',)}


def _spin(low: float, high: float, value: float, step: float,
          tip: str) -> DoubleSpinBox:
    """A field reporting only a finished edit, since each report
    redraws the judgment."""
    return number_box(low, high, value=value, step=step, tip=tip)


class SegmentsPanel(QWidget):
    """The marks, the settings and the readout. Edits are emitted as
    `settings()`."""

    #: a mark or a setting moved; here is the whole set of settings now
    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title: QLabel = QLabel('Segments')
        # the one settings panel that scrolls: a row per mark, a row per
        # kind's tolerance and a line per segment's verdict outgrow the
        # plot's height with three segments marked, and a panel cut off
        # at the bottom hides the verdicts it exists to show
        body = QWidget()
        grid = panel_grid(body, self.title)
        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.setFixedWidth(PANEL_WIDTH + self.style().pixelMetric(
            QStyle.PixelMetric.PM_ScrollBarExtent))
        self._loading = False
        #: the reference's length in seconds, where a new mark may reach
        self.duration: float | None = None

        self.table: QTableWidget = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['From [s]', 'To [s]', 'Kind'])
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        # the times are short numbers; the kind's name is what needs
        # the width, or it reads 'static' and 'shocl'
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(120)
        self.table.setToolTip(
            'One row a segment: where it starts and ends on the '
            'reference\'s clock, and the kind of motion it is, which '
            'sets the rule it is judged by')
        grid.addWidget(self.table, 1, 0, 1, 2)
        buttons = QHBoxLayout()
        self.add_button: QToolButton = QToolButton()
        self.add_button.setIcon(control_icon('add'))
        self.add_button.setToolTip('Mark a segment after the last')
        self.remove_button: QToolButton = QToolButton()
        self.remove_button.setIcon(control_icon('trash'))
        self.remove_button.setToolTip('Remove the selected segment')
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        grid.addLayout(buttons, 2, 0, 1, 2)

        self.taper_box: DoubleSpinBox = _spin(
            0.0, 100.0, TAPER_PERCENT, 5.0,
            'The share of each segment under its tapered cosine window\'s '
            'two ends, before its spectra and mean square are taken')
        self.band_box: QCheckBox = QCheckBox()
        self.band_box.setToolTip(
            'Judge the spectra over the test band only. Past it the '
            'reference holds the filters\' skirts and a ratio there is '
            'not a miss')
        self.band_low: DoubleSpinBox = _spin(
            0.01, 1e6, BAND_HZ[0], 1.0, 'The test band\'s lower edge, Hz')
        self.band_high: DoubleSpinBox = _spin(
            0.01, 1e6, BAND_HZ[1], 10.0, 'The test band\'s upper edge, Hz')
        # the tolerances start a little above zero: a band of ±0 is no
        # band, and the judgment would refuse it (2026-10-16 review)
        self.asd_box: DoubleSpinBox = _spin(
            0.1, 40.0, ASD_DB[1], 0.5,
            'A stationary segment\'s ASD and a transient segment\'s '
            'normalized ASD, dB either side of the reference\'s at every '
            'line')
        self.variance_box: DoubleSpinBox = _spin(
            0.0, 1000.0, VARIANCE_PERCENT[1], 1.0,
            'How far over the reference\'s a stationary segment\'s mean '
            'square may be, percent')
        self.envelope_box: DoubleSpinBox = _spin(
            0.1, 40.0, ENVELOPE_DB[1], 0.5,
            'A transient segment\'s short-time rms, dB either side of the '
            'reference\'s in every window')
        self.window_box: DoubleSpinBox = _spin(
            0.0, 1e4, 0.0, 0.1,
            'The envelope\'s window and step, s; at its lowest (auto), '
            'one percent of the record')
        self.window_box.setDecimals(3)
        self.window_box.setSpecialValueText('auto')
        self.srs_low: DoubleSpinBox = _spin(
            -40.0, 0.0, SRS_DB[0], 0.5,
            'A shock\'s SRS, dB below the reference\'s, each polarity')
        self.srs_high: DoubleSpinBox = _spin(
            0.1, 40.0, SRS_DB[1], 0.5,
            'A shock\'s SRS, dB above the reference\'s, each polarity')
        self.peaks_box: DoubleSpinBox = _spin(
            0.1, 100.0, PEAK_PERCENT, 5.0,
            'How far each of a shock\'s major peaks may be from the '
            'reference\'s, percent')
        self.names, self.editors = add_keyed_settings(grid, [
            ('taper', 'Taper [%]', self.taper_box),
            ('band', 'Test band', self.band_box),
            ('band_low', 'Band from [Hz]', self.band_low),
            ('band_high', 'Band to [Hz]', self.band_high),
            ('asd', 'ASD ± [dB]', self.asd_box),
            ('variance', 'Variance + [%]', self.variance_box),
            ('envelope', 'Envelope ± [dB]', self.envelope_box),
            ('window', 'Window [s]', self.window_box),
            ('srs_low', 'SRS below [dB]', self.srs_low),
            ('srs_high', 'SRS above [dB]', self.srs_high),
            ('peaks', 'Peaks ± [%]', self.peaks_box),
        ], 3)
        add_rule(grid, 14)
        # one line for the record and a few for each segment, so a label
        # that wraps rather than a fixed set of rows
        self.readout: QLabel = QLabel('')
        self.readout.setWordWrap(True)
        grid.addWidget(self.readout, 15, 0, 1, 2)
        grid.setRowStretch(16, 1)

        self.add_button.clicked.connect(self.add_segment)
        self.remove_button.clicked.connect(self.remove_segment)
        self.table.cellChanged.connect(self._edited)
        self.band_box.toggled.connect(self._edited)
        for box in (self.taper_box, self.band_low, self.band_high,
                    self.asd_box, self.variance_box, self.envelope_box,
                    self.window_box, self.srs_low, self.srs_high,
                    self.peaks_box):
            box.valueChanged.connect(self._edited)
        self._shown_rows()

    # ---- the marks ----------------------------------------------------------

    def _kind_box(self, kind: str) -> QComboBox:
        box = QComboBox()
        for name in SEGMENT_KINDS:
            box.addItem(name, name)
        box.setCurrentIndex(SEGMENT_KINDS.index(kind))
        box.currentIndexChanged.connect(self._edited)
        return box

    def _append(self, start: float, end: float, kind: str,
                name: str = '') -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, value in ((0, start), (1, end)):
            self.table.setItem(row, column, QTableWidgetItem(f'{value:g}'))
        # a name given in a script rides with its row, unseen, so the
        # plot and the stored marks keep it through an edit of the times
        self.table.item(row, 0).setData(Qt.ItemDataRole.UserRole, name)
        self.table.setCellWidget(row, 2, self._kind_box(kind))

    def add_segment(self) -> None:
        """Mark a stationary segment after the last one, `NEW_SECONDS`
        long or to the end of the reference."""
        marks = self.segments()
        start = marks[-1][1] if marks else 0.0
        end = start + NEW_SECONDS
        if self.duration is not None:
            if start >= self.duration:
                start = max(self.duration - NEW_SECONDS, 0.0)
            end = min(start + NEW_SECONDS, self.duration)
        self._loading = True
        try:
            self._append(start, end, 'stationary')
        finally:
            self._loading = False
        self._edited()

    def remove_segment(self) -> None:
        """Remove the selected segment, or the last when none is
        selected."""
        row = self.table.currentRow()
        if row < 0:
            row = self.table.rowCount() - 1
        if row >= 0:
            self.table.removeRow(row)
            self._edited()

    def segments(self) -> list[tuple[float, float, str]]:
        """The marks, (start, end, kind), in the table's order. A row
        whose times do not read as numbers is left out."""
        return [mark[:3] for mark in self.marks()]

    def marks(self) -> list[Segment]:
        """The marks as `Segment`s, with any name they were given, in
        the table's order. A row whose times do not read as numbers is
        left out."""
        out = []
        for row in range(self.table.rowCount()):
            try:
                start = float(self.table.item(row, 0).text())
                end = float(self.table.item(row, 1).text())
            except (AttributeError, ValueError):
                continue
            box = self.table.cellWidget(row, 2)
            if box is None:
                # a row being made: its times are in, its kind not yet
                continue
            name = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            out.append(Segment(start, end, str(box.currentData()),
                               str(name or '')))
        return out

    def set_segments(self, marks: Sequence[Sequence[Any]],
                     emit: bool = True) -> None:
        """Replace the marks, each (start, end, kind[, name]), then emit
        once, or not at all when `emit` is False."""
        self._loading = True
        try:
            self.table.setRowCount(0)
            for mark in marks:
                self._append(float(mark[0]), float(mark[1]), str(mark[2]),
                             str(mark[3]) if len(mark) > 3 else '')
        finally:
            self._loading = False
        if emit:
            self._edited()
        else:
            self._shown_rows()

    # ---- the settings -------------------------------------------------------

    def _shown_rows(self) -> None:
        """Show only the rows that apply (principle 3): each kind's
        tolerances while a segment of that kind is marked, the band's
        edges while it is ticked."""
        kinds = {kind for _start, _end, kind in self.segments()}
        shown = {key: bool(kinds & set(wanted))
                 for key, wanted in ROWS_FOR.items()}
        band = self.band_box.isChecked()
        shown['band_low'] = shown['band_high'] = band
        for key, visible in shown.items():
            self.names[key].setVisible(visible)
            self.editors[key].setVisible(visible)

    def _edited(self, *_args: Any) -> None:
        self._shown_rows()
        if not self._loading:
            self.changed.emit(self.settings())

    def settings(self) -> dict[str, Any]:
        """The judgment's tolerances as `judge_segments` takes them:
        'taper_percent', 'band' (None when unticked), 'asd_db',
        'variance_percent', 'envelope_db', 'window' (None at its
        lowest), 'srs_db' and 'peak_percent'."""
        asd = self.asd_box.value()
        envelope = self.envelope_box.value()
        window = self.window_box.value()
        return {
            'taper_percent': self.taper_box.value(),
            'band': ((self.band_low.value(), self.band_high.value())
                     if self.band_box.isChecked() else None),
            'asd_db': (-asd, asd),
            'variance_percent': (VARIANCE_PERCENT[0],
                                 self.variance_box.value()),
            'envelope_db': (-envelope, envelope),
            'window': window if window > self.window_box.minimum() else None,
            'srs_db': (self.srs_low.value(), self.srs_high.value()),
            'peak_percent': self.peaks_box.value()}

    def set_settings(self, **settings: Any) -> None:
        """Set any of the fields `settings()` names, without a redraw per
        field, then emit once."""
        self._loading = True
        try:
            if 'taper_percent' in settings:
                self.taper_box.setValue(settings['taper_percent'])
            if 'band' in settings:
                band = settings['band']
                self.band_box.setChecked(band is not None)
                if band is not None:
                    self.band_low.setValue(band[0])
                    self.band_high.setValue(band[1])
            if 'asd_db' in settings:
                self.asd_box.setValue(settings['asd_db'][1])
            if 'variance_percent' in settings:
                self.variance_box.setValue(settings['variance_percent'][1])
            if 'envelope_db' in settings:
                self.envelope_box.setValue(settings['envelope_db'][1])
            if 'window' in settings:
                window = settings['window']
                self.window_box.setValue(0.0 if window is None else window)
            if 'srs_db' in settings:
                self.srs_low.setValue(settings['srs_db'][0])
                self.srs_high.setValue(settings['srs_db'][1])
            if 'peak_percent' in settings:
                self.peaks_box.setValue(settings['peak_percent'])
        finally:
            self._loading = False
        self._edited()

    # ---- the readout --------------------------------------------------------

    def show_result(self, result: Any) -> None:
        """Write the record's line, then each segment's verdict with the
        checks it failed by name; each check's reading is in the table
        under the plot. None clears the readout."""
        if result is None:
            self.readout.setText('')
            return
        lines = [result.describe()]
        lines += [segment.describe() for segment in result.segments]
        self.readout.setText('\n'.join(lines))
