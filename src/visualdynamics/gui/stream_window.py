"""The import window for a long run, chosen on a preview.

A stream that would take a large share of the machine — a quarter of
its memory, `io.rattlesnake.LARGE_STREAM_SHARE` — is not imported
whole without asking (Brandon, 2026-09-18: a 22.5 GB run fits a 64 GB
machine and not a 32 GB one). This dialog is the question: one
channel's envelope over the whole run to read the run by, the
truncate reading's own span over it to choose the stretch, the
channels to keep, and what the choice costs in samples and gigabytes
against the machine's memory. Import reads exactly that — the same
`import_file(path, start=, stop=, channels=)` a script says, so the
journal replays the import as it was made.

The preview is one channel, and the window chosen on it applies to
every channel: right for a run, and a trust that the run is
stationary across channels, which the chooser is there to check.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.truncate import Truncation
from ..io import rattlesnake
from ..plot import background_brush
from ..plot.truncation import TruncationOverlay
from .truncate_panel import TruncatePanel

GB = 2 ** 30


class StreamWindowDialog(QDialog):
    """How much of a run to import, chosen on one channel's envelope.

    `options()` is the answer in the importer's own vocabulary: empty
    for the whole run, `start`/`stop` for a window, `channels` for a
    subset — only what differs from importing whole, so the journal
    line stays the plain one when nothing was cut.
    """

    def __init__(self, path: str | os.PathLike, summary: Mapping[str, Any],
                 colors: Mapping[str, str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        import pyqtgraph as pg

        self.path: str = str(path)
        self.summary: dict[str, Any] = dict(summary)
        self.colors: dict[str, str] = dict(colors)
        self.rate: float = float(summary['sample_rate'])
        streams = list(summary['streams'])
        #: the stream being previewed — the largest, since it is the one
        #: that raised the question
        self.stream: dict[str, Any] = max(streams, key=lambda s: s['bytes'])
        #: the last preview read, for whoever asks what was drawn
        self.preview: dict[str, Any] | None = None
        self.overlay: TruncationOverlay | None = None
        self._curves: list[Any] = []

        base = os.path.basename(self.path)
        self.setWindowTitle(f'Import a part of {base}')
        memory = int(summary.get('memory') or 0)
        total = sum(s['bytes'] for s in streams)
        words = (f'{base} holds {len(summary["channels"])} channels for '
                 f'{self.stream["seconds"]:.4g} s at {self.rate:g} Hz: '
                 f'{total / GB:.2f} GB of time data')
        if memory:
            words += f' on a machine with {memory / GB:.0f} GB'
        words += ('. Choose the stretch and the channels to import; the '
                  'preview is one channel over the whole run.')
        self.header: QLabel = QLabel(words)
        self.header.setWordWrap(True)

        self.channel_box: QComboBox = QComboBox()
        self.channel_box.addItems(list(summary['channels']))
        self.channel_box.setToolTip('Which channel the preview shows')
        self.stream_box: QComboBox = QComboBox()
        for stream in streams:
            self.stream_box.addItem(
                f'{stream["key"]} ({stream["seconds"]:.4g} s)', stream['key'])
        self.stream_box.setCurrentIndex(streams.index(self.stream))
        self.stream_box.setVisible(len(streams) > 1)

        self.plot: Any = pg.PlotWidget()
        self.plot.setBackground(background_brush(self.colors))
        foreground = pg.mkPen(self.colors['plot_foreground'])
        for edge in ('left', 'bottom'):
            axis = self.plot.getAxis(edge)
            axis.setPen(foreground)
            axis.setTextPen(foreground)
        self.plot.setLabel('bottom', 'time [s]')
        self.plot.setMinimumSize(560, 260)

        self.panel: TruncatePanel = TruncatePanel()
        self.panel.title.setText('Window')
        # the dialog's Import is the apply; a second button that made a
        # record from a run not yet imported would be a promise
        self.panel.apply_button.hide()

        self.channel_list: QListWidget = QListWidget()
        self.channel_list.setToolTip('Uncheck a channel to leave it out')
        for dof in summary['channels']:
            item = QListWidgetItem(dof)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.channel_list.addItem(item)
        self.cost: QLabel = QLabel()
        self.cost.setWordWrap(True)

        buttons = QDialogButtonBox()
        self.import_button: Any = buttons.addButton(
            'Import', QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        chooser = QHBoxLayout()
        chooser.addWidget(QLabel('Preview channel'))
        chooser.addWidget(self.channel_box)
        chooser.addWidget(self.stream_box)
        chooser.addStretch(1)
        side = QVBoxLayout()
        side.addWidget(self.panel)
        side.addWidget(QLabel('Channels'))
        side.addWidget(self.channel_list, 1)
        middle = QHBoxLayout()
        middle.addWidget(self.plot, 1)
        middle.addLayout(side)
        layout = QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addLayout(chooser)
        layout.addLayout(middle, 1)
        layout.addWidget(self.cost)
        layout.addWidget(buttons)

        self.channel_box.currentIndexChanged.connect(lambda _i: self._draw_preview())
        self.stream_box.currentIndexChanged.connect(self._stream_chosen)
        self.panel.changed.connect(self._panel_edited)
        self.channel_list.itemChanged.connect(lambda _item: self._restate())
        self._show_stream(self.stream)

    # ---- what is shown ------------------------------------------------------

    def _stream_chosen(self, index: int) -> None:
        key = self.stream_box.itemData(index)
        chosen = next(s for s in self.summary['streams'] if s['key'] == key)
        if chosen is not self.stream:
            self._show_stream(chosen)

    def _show_stream(self, stream: Mapping[str, Any]) -> None:
        """Put a stream up: its preview, and the whole of it as the span."""
        self.stream = dict(stream)
        first, last = 0.0, (self.stream['samples'] - 1) / self.rate
        whole = Truncation(first, last)
        if self.overlay is not None:
            self.overlay.remove()
            self.overlay = None
        self._draw_preview()
        self.overlay = TruncationOverlay(self.plot.getPlotItem(), whole,
                                         first, last, self.colors)
        self.overlay.changed.connect(self._dragged)
        self.panel.show_span(first, last, 1.0 / self.rate, whole)
        self._restate()

    def _draw_preview(self) -> None:
        import pyqtgraph as pg

        for curve in self._curves:
            self.plot.removeItem(curve)
        self._curves = []
        preview = rattlesnake.stream_preview(
            self.path, self.channel_box.currentIndex(),
            stream=self.stream['variable'])
        self.preview = preview
        # two curves and no fill between them: a bucket the stream has
        # no sample in is NaN, drawn as a gap (connect='finite'), and a
        # filled band would have to invent an edge across it
        pen = pg.mkPen(self.colors['response_curve'], width=1)
        low = self.plot.plot(preview['times'], preview['low'], pen=pen,
                             connect='finite')
        high = self.plot.plot(preview['times'], preview['high'], pen=pen,
                              connect='finite')
        self._curves = [low, high]
        unit = preview['unit'] or 'units undefined'
        self.plot.setLabel('left', f'{preview["dof"]} [{unit}]')
        self.plot.setXRange(0.0, (preview['samples'] - 1) / self.rate, padding=0)

    # ---- the span, from either side ----------------------------------------

    def _dragged(self, truncation: Truncation) -> None:
        self.panel.set_truncation(truncation)
        self._restate()

    def _panel_edited(self, truncation: Truncation) -> None:
        if self.overlay is not None:
            self.overlay.set_truncation(truncation)
        self._restate()

    def truncation(self) -> Truncation:
        return self.panel.truncation()

    def channels(self) -> list[str]:
        """The checked channels, in table order."""
        return [self.channel_list.item(i).text()
                for i in range(self.channel_list.count())
                if self.channel_list.item(i).checkState() == Qt.CheckState.Checked]

    def _window(self) -> tuple[int, int]:
        span = self.truncation()
        found = rattlesnake._sample_window(span.start, span.stop,
                                           int(self.stream['samples']), self.rate)
        return found if found is not None else (0, 0)

    def _restate(self) -> None:
        """What the choice costs, said where it is being made."""
        first, last = self._window()
        samples = last - first
        kept = len(self.channels())
        size = samples * kept * 8
        words = f'{samples:,} samples × {kept} channels = {size / GB:.2f} GB'
        memory = int(self.summary.get('memory') or 0)
        if memory:
            words += f' of the machine\'s {memory / GB:.0f} GB'
        if kept == 0:
            words += ' — no channel is checked'
        elif samples < 2:
            words += ' — the span holds fewer than two samples'
        self.cost.setText(words)
        self.import_button.setEnabled(kept > 0 and samples >= 2)

    def options(self) -> dict[str, Any]:
        """The import's keyword arguments: only what differs from whole."""
        out: dict[str, Any] = {}
        first, last = self._window()
        span = self.truncation()
        if (first, last) != (0, int(self.stream['samples'])):
            out['start'] = float(span.start)
            out['stop'] = float(span.stop)
        kept = self.channels()
        if len(kept) < len(self.summary['channels']):
            out['channels'] = kept
        return out


def ask_stream_window(parent: QWidget | None, path: str | os.PathLike,
                      summary: Mapping[str, Any],
                      colors: Mapping[str, str]) -> dict[str, Any] | None:
    """Put the dialog up; the import options chosen, or None to skip the file."""
    dialog = StreamWindowDialog(path, summary, colors, parent)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.options()
    return None
