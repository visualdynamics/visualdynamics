"""Drawing a tracking filter's band on the record it is applied to.

A level read through a tracking filter depends on a band that moves
with the drive frequency. This view draws the band where it is applied,
in the same way the averaging frames are drawn on the record they are
cut from and a filter preview is drawn over the time history it
filters. It shows one tone of one channel: the record at the top, its
scalogram below, and beside the scalogram the band's magnitude at one
instant.

The record is drawn gray and the band's output, the waveform it
passes, is drawn over it in color, as in the filter preview, because
the passed waveform is what is being judged. The color is the theme's
`filter_preview` (magenta) and not the channel's own color, which is
the one difference from the filter preview. The same color is used for
the band's corridor, its shape and its weight on the record, so that on
all three panels magenta means the filter and nothing else is drawn in
it.

The time-frequency picture is the scalogram of `core.wavelet`, drawn by
`plot.scalogram.scalogram_image` with time across, log frequency up and
the cone of influence shaded. Its rows are evenly spaced in log
frequency, the axis on which a proportional band has a constant width.
Two settings differ from the scalogram view. First, the color is the
magnitude in dB below the picture's largest value, over `COLOR_RANGE`.
On the scalogram's linear scale, a noise floor a fifth of the tone's
amplitude spread over the whole band is a few percent of the color bar
and is drawn black, and this picture has to show the noise in order to
show the band keeping it out. Second, the wavelet is narrower in
frequency: `OMEGA0` is twelve cycles, against the scalogram's six, so
the tone's ridge is narrower than the common bands. At six cycles the
ridge is 14 % either side of the tone at -3 dB, more than half of a
50 % band's 25 %; at twelve it is 7 %. A band narrower than that, such
as 10 % proportional or a few hertz fixed, is narrower than the picture
can resolve for a moving tone, and its corridor is drawn inside the
ridge. The cost is resolution in time, because a longer wavelet spreads
a sweep along the time axis. At one octave per second the drive moves
about 7 % within the wavelet's width at 20 Hz, as much as the wavelet
resolves, and a fifth of that at 100 Hz, so the ridge stays narrow from
the bottom of a sweep up.

The band is drawn over the picture as a corridor that follows the
sweep. Its edges are the drive plus and minus half the bandwidth, the
-3 dB points, drawn as dashed lines over a wider line in the background
color. The dashes let a reader who cannot tell magenta from the
colormap still tell the line from the image, and the wider line under
them keeps a thin line visible over every color of the colormap. A
proportional band has a constant width on the log axis and is drawn as
a pair of lines parallel to the ridge. A fixed band has a constant
width in hertz, so it is wide at the bottom of the axis and narrow at
the top. Showing that difference is the purpose of the picture.

The band's magnitude at the cursor (`SineTracking.response`) is drawn
beside the picture with frequency up, on an axis linked to the
picture's, so the -3 dB edges of the shape line up with the corridor's
edges, and the drive and its harmonics are marked at the same heights
on both. The drive is inside the band, where the shape is 0 dB; a
harmonic is outside it, where the shape gives its rejection. The drive
is marked with a circle and a harmonic with a triangle, each filled
with the background color and outlined in the foreground color, so
they can be seen over any color of the colormap.

The band's output at an instant is the record before that instant
weighted by the low-pass filter's impulse response
(`SineTracking.weighting`). That weight is shaded on the record,
reaching back from the cursor and fading as the weight decreases, in
the same way the averaging band's shading shows the weight each part of
the record carries in a frame. Its length is the settling time given in
the readout: a wide band's weight is short, a narrow band's reaches
back across a change in level, and the envelope of the passed waveform
follows that change over the same time. The shading is clipped at zero
where the impulse response is negative. For the default fourth-order
filter the negative part is about a sixth of the peak, and it is the
reason the band overshoots a step.

One setting is drawn per view. Several corridors over one picture,
each with its own shape beside it, would cover the image they are read
against, so two settings are two calls placed side by side. A strip of
the level under the record is left out for the same reason: the
envelope of the passed waveform is the 'filtered' reading and is
already drawn at every sample.

In the application the cursor can be dragged along either time axis,
and the shape, the marks, the weight and the readout follow it. Each is
computed in closed form, so the redraw is immediate. In a script the
cursor is set by the `cursor` argument or by
`TrackingFilterView.set_cursor`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..core import wavelet

# The record is gray and the passed waveform in color, the filter
# preview's way round (Brandon, 2026-08-25, for the filter's preview:
# with the colors the other way, the reader judged the thing being
# decided from the drabber line).

#: the shape panel's floor and ceiling, dB. The floor is the filter
#: panel's: below it a band has stopped doing anything a real record
#: shows
SHAPE_FLOOR = -80.0
SHAPE_CEILING = 5.0

#: the picture's color range, in dB below its largest magnitude: deep
#: enough for a noise floor tens of dB under the tone to have a color
COLOR_RANGE = 60.0

#: cycles under the wavelet — twelve, so the tone's ridge is narrower
#: than the bands it is drawn inside (module docstring)
OMEGA0 = 12.0

#: the harmonics marked by default: the third, a clipped drive's
HARMONICS = (3,)

#: points along each corridor edge. The drive is smooth, so this only
#: has to be finer than the eye at the picture's width
CORRIDOR_POINTS = 1200

#: how far back the weight is shaded: until it stays under this share
#: of its peak
WEIGHT_FLOOR = 0.01

#: how solid the weight's shading is where the weight peaks — the
#: averaging band's own peak alpha
WEIGHT_ALPHA = 64 / 255

#: color stops across the weight's gradient (the averaging's count)
WEIGHT_STOPS = 48

#: the marks' shapes: the drive and a harmonic, distinct in outline
DRIVE_SYMBOL = 'o'
HARMONIC_SYMBOL = 't'
MARK_SIZE = 11

#: the corridor's dashed line and the halo under it, pixels
CORRIDOR_WIDTH = 2
HALO_WIDTH = 5

#: widths held equal across the panels, pixels: the left axes, so the
#: record and the picture share one time-to-pixel mapping; the picture's
#: color bar, which the record matches with an empty column; the shape
#: panel
LEFT_AXIS = 64
BAR_WIDTH = 96
SHAPE_WIDTH = 240


def default_range(waveform: Any, harmonics: Sequence[int],
                  sample_rate: float) -> tuple[float, float]:
    """The picture's frequency range in Hz, as (low, high).

    From an octave below the lowest drive frequency up to 1.5 times the
    highest harmonic marked, or 1.5 times the band's top edge if that is
    higher, and no higher than 0.45 times `sample_rate`.
    """
    drive = np.asarray(waveform.drive, dtype=float)
    top_edge = float(np.max(drive + waveform.setting.bandwidth(drive) / 2.0))
    highest = float(drive.max()) * max([1, *harmonics])
    high = min(1.5 * max(highest, top_edge), 0.45 * float(sample_rate))
    return float(drive.min()) / 2.0, high


def _db(value: float) -> str:
    """Decibels as the readout prints them, with a true minus sign."""
    return f'{value:.0f} dB'.replace('-', '−')


def _seconds(value: float) -> str:
    return (f'{value * 1000.0:.3g} ms' if value < 1.0
            else f'{value:.3g} s')


class TrackingFilterView:
    """The items `build_tracking_filter` draws, and the cursor that moves them.

    The view owns every item it adds. `set_cursor` redraws the shape,
    the marks, the weight and the readout for a new instant. After each
    call, `cursor`, `drive_hz`, `bandwidth_hz`, `edges_hz`,
    `harmonic_db` and `text` hold what was read there.
    """

    def __init__(self, record_plot: Any, picture: Any, shape: Any,
                 readout: Any, waveform: Any, frequencies: np.ndarray,
                 harmonics: Sequence[int], colors: Mapping[str, str],
                 heights: tuple[float, float], interactive: bool) -> None:
        import pyqtgraph as pg
        from PySide6.QtCore import Qt

        self.record_plot: Any = record_plot
        self.picture: Any = picture
        self.shape: Any = shape
        self.readout: Any = readout
        self.waveform: Any = waveform
        self.setting: Any = waveform.setting
        self.harmonics: tuple[int, ...] = tuple(int(k) for k in harmonics)
        self.colors: Mapping[str, str] = colors
        self.frequencies: np.ndarray = np.asarray(frequencies, dtype=float)
        self._heights = heights
        self._restating = False

        #: where the cursor is, on the record's clock, and what was read
        #: there: the drive, the bandwidth, and each harmonic's rejection
        self.cursor: float = float(waveform.time[0])
        self.drive_hz: float = 0.0
        self.bandwidth_hz: float = 0.0
        self.edges_hz: tuple[float, float] = (0.0, 0.0)
        self.harmonic_db: dict[int, float] = {}
        self.text: str = ''

        foreground = colors['plot_foreground']
        background = colors['plot_background']
        lighter = max((foreground, background), key=_lightness)
        self.corridor_pen: Any = pg.mkPen(colors['filter_preview'],
                                     width=CORRIDOR_WIDTH,
                                     style=Qt.PenStyle.DashLine)
        bounds = (float(waveform.time[0]), float(waveform.time[-1]))

        # the cursor on both time axes: the foreground on the record,
        # and the lighter of the two theme colors on the picture,
        # whose colormap is dark over most of its field
        self.lines: list[Any] = []
        for plot, color in ((record_plot, foreground), (picture, lighter)):
            line = pg.InfiniteLine(
                angle=90, movable=interactive, bounds=bounds,
                pen=pg.mkPen(color, width=1),
                hoverPen=pg.mkPen(colors['filter_preview'], width=2))
            line.setZValue(30)
            line.sigPositionChanged.connect(self._dragged)
            plot.addItem(line, ignoreBounds=True)
            self.lines.append(line)

        marker_pen = pg.mkPen(foreground, width=1.5)
        marker_brush = pg.mkBrush(background)

        def mark(plot: Any, symbol: str) -> Any:
            item = pg.PlotDataItem([], [], pen=None, symbol=symbol,
                                   symbolSize=MARK_SIZE,
                                   symbolPen=marker_pen,
                                   symbolBrush=marker_brush)
            item.setZValue(31)
            plot.addItem(item, ignoreBounds=True)
            return item

        self.picture_drive: Any = mark(picture, DRIVE_SYMBOL)
        self.picture_harmonics: Any = mark(picture, HARMONIC_SYMBOL)
        self.shape_curve: Any = shape.plot(
            [], [], pen=pg.mkPen(colors['filter_preview'], width=2))
        self.shape_edges: list[Any] = []
        for _ in range(2):
            edge = pg.InfiniteLine(angle=0, movable=False,
                                   pen=self.corridor_pen)
            shape.addItem(edge, ignoreBounds=True)
            self.shape_edges.append(edge)
        self.shape_drive: Any = mark(shape, DRIVE_SYMBOL)
        self.shape_harmonics: Any = mark(shape, HARMONIC_SYMBOL)

        self.weight: Any = pg.PlotDataItem([], [], pen=pg.mkPen(None),
                                      fillLevel=heights[0])
        self.weight.setZValue(-20)       # under the trace it describes
        record_plot.addItem(self.weight, ignoreBounds=True)

    # ---- the cursor --------------------------------------------------------

    def set_cursor(self, seconds: float) -> float:
        """Move the cursor to `seconds` and redraw what is read there.

        `seconds` is on the record's clock, and the cursor is held
        inside the tone's span. Returns the time it was placed at, in
        seconds.
        """
        time = self.waveform.time
        at = float(np.clip(seconds, time[0], time[-1]))
        self.cursor = at
        self._restating = True
        try:
            for line in self.lines:
                if line.value() != at:
                    line.setValue(at)
        finally:
            self._restating = False
        drive = float(np.interp(at, time, self.waveform.drive))
        width = float(self.setting.bandwidth(drive))
        self.drive_hz, self.bandwidth_hz = drive, width
        self.edges_hz = (drive - width / 2.0, drive + width / 2.0)
        self._draw_marks(at, drive)
        self._draw_shape(drive)
        self._draw_weight(at, drive, width)
        self._write_readout(at, drive, width)
        return at

    def _dragged(self, line: Any) -> None:
        if not self._restating:
            self.set_cursor(float(line.value()))

    # ---- what moves with it ------------------------------------------------

    def _draw_marks(self, at: float, drive: float) -> None:
        self.picture_drive.setData([at], [np.log10(drive)])
        above = [k * drive for k in self.harmonics]
        self.picture_harmonics.setData([at] * len(above),
                                       list(np.log10(above)))

    def _draw_shape(self, drive: float) -> None:
        magnitude = self.setting.response(drive, self.frequencies)
        self.shape_curve.setData(np.maximum(magnitude, SHAPE_FLOOR),
                                 np.log10(self.frequencies))
        for edge, hz in zip(self.shape_edges, self.edges_hz):
            # a fixed band wider than twice the drive reaches past zero
            # frequency, where a log axis has nowhere to put its edge
            edge.setVisible(hz > 0.0)
            if hz > 0.0:
                edge.setValue(float(np.log10(hz)))
        self.shape_drive.setData(
            [float(self.setting.response(drive, drive))], [np.log10(drive)])
        self.harmonic_db = {
            k: float(self.setting.response(drive, k * drive))
            for k in self.harmonics}
        self.shape_harmonics.setData(
            [max(self.harmonic_db[k], SHAPE_FLOOR) for k in self.harmonics],
            [np.log10(k * drive) for k in self.harmonics])

    def _draw_weight(self, at: float, drive: float, width: float) -> None:
        """The band's impulse response shaded behind the cursor, faded
        across with the weight it gives each moment of the record — the
        averaging band's gradient, in data coordinates so it stretches
        with the view."""
        import pyqtgraph as pg
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QBrush, QColor, QLinearGradient

        lags = np.linspace(0.0, 12.0 / width, 4000)
        weight = self.setting.weighting(drive, lags)
        weight = weight / weight.max()
        reach = float(lags[np.flatnonzero(np.abs(weight) > WEIGHT_FLOOR)[-1]])
        start = at - reach
        low, high = self._heights
        self.weight.setData([start, at], [high, high])
        self.weight.setFillLevel(low)
        gradient = QLinearGradient(QPointF(start, 0.0), QPointF(at, 0.0))
        color = QColor(self.colors['filter_preview'])
        for share in np.linspace(0.0, 1.0, WEIGHT_STOPS):
            stop = QColor(color)
            value = float(np.interp(reach * (1.0 - share), lags, weight))
            stop.setAlpha(round(WEIGHT_ALPHA * 255 * min(max(value, 0.0),
                                                         1.0)))
            gradient.setColorAt(float(share), stop)
        self.weight.setFillBrush(pg.mkBrush(QBrush(gradient)))
        #: how far back the weight reaches, seconds
        self.reach: float = reach

    def _write_readout(self, at: float, drive: float, width: float) -> None:
        low, high = self.edges_hz
        lines = [f'at {at:.3f} s, drive {drive:.4g} Hz',
                 f'band {width:.3g} Hz wide: '
                 + (f'{low:.4g} to {high:.4g} Hz' if low > 0.0
                    else f'up to {high:.4g} Hz'),
                 f'settles in about {_seconds(1.0 / width)} (1 / band)']
        # under the panel's floor the number is a property of the
        # prototype, not of anything a record would show
        lines += [f'{k} \u00d7 drive, {k * drive:.4g} Hz: '
                  + (_db(self.harmonic_db[k])
                     if self.harmonic_db[k] >= SHAPE_FLOOR
                     else f'below {_db(SHAPE_FLOOR)}')
                  for k in self.harmonics]
        self.text = '\n'.join(lines)
        self.readout.setText('<br>'.join(lines))


def _translucent(color: str, alpha: float) -> Any:
    """A theme color at `alpha` (0 to 1), as a QColor."""
    from PySide6.QtGui import QColor

    out = QColor(color)
    out.setAlpha(round(alpha * 255))
    return out


def _lightness(color: str) -> float:
    """A theme color's lightness, 0 to 255: which of two reads lighter."""
    text = str(color).lstrip('#')
    red, green, blue = (int(text[i:i + 2], 16) for i in (0, 2, 4))
    return 0.299 * red + 0.587 * green + 0.114 * blue


def build_tracking_filter(layout: Any, history: Any, waveform: Any, *,
                          cursor: float | None = None,
                          span: tuple[float, float] | None = None,
                          harmonics: Sequence[int] = HARMONICS,
                          low: float | None = None,
                          high: float | None = None,
                          omega0: float = OMEGA0,
                          unit_system: Any = None, theme: Any = None,
                          interactive: bool = True) -> TrackingFilterView:
    """Draw one tone's tracking band on its record and return the view.

    `waveform` is the result of `core.sine_tracking.track_waveform` for
    channel `waveform.dof` of `history`. The record and the passed
    waveform are drawn in row 0, the picture and the corridor in row 2
    with the shape beside it in column 1, the readout above the shape,
    and the legend under the picture. `cursor` is a time in seconds on
    the record's clock, the middle of the tone's span by default; `span`
    is the (start, stop) the time axes show, the whole record by
    default. `harmonics` are the multiples of the drive frequency that
    are marked, the third by default. `low` and `high` are the
    picture's frequency range in Hz (`default_range` when omitted), and
    `omega0` is the wavelet's width parameter (`OMEGA0`). With
    `interactive` False the cursor cannot be dragged.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QGraphicsWidget

    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import axis_label, legend_below
    from .scalogram import _decade_ticks, scalogram_image

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    setting = waveform.setting
    row = list(history.response_dof).index(waveform.dof)
    clock = np.asarray(history.abscissa, dtype=float)
    rate = float(history.sample_rate)
    dimension = waveform.dimension
    raw = np.real(np.asarray(history.ordinate[row], dtype=float))
    harmonics = tuple(int(k) for k in harmonics)

    # ---- the record, and what the band passes -----------------------------
    record_plot = layout.addPlot(row=0, col=0)
    record_plot.showGrid(x=True, y=True, alpha=0.2)
    record_plot.setTitle(f'{waveform.tone} at {waveform.dof}: '
                         f'{setting.describe()}',
                         color=colors['plot_foreground'], size='9pt')
    shown = us.from_si(raw, dimension)
    record = record_plot.plot(clock, shown, pen=pg.mkPen(
        colors['specification_curve'], width=1))
    passed = record_plot.plot(
        waveform.time, us.from_si(waveform.passed, dimension),
        pen=pg.mkPen(colors['filter_preview'], width=1))
    passed.setZValue(15)                 # over the record: it is the subject
    for curve in (record, passed):
        curve.setDownsampling(auto=True, method='peak')
        curve.setClipToView(True)
    top = float(np.max(np.abs(shown))) * 1.08 or 1.0
    record_plot.setYRange(-top, top, padding=0)
    record_plot.setLabel('left', axis_label(dimension, us))
    record_plot.getAxis('left').setWidth(LEFT_AXIS)

    # ---- the picture --------------------------------------------------------
    if low is None or high is None:
        default_low, default_high = default_range(waveform, harmonics, rate)
        low = default_low if low is None else low
        high = default_high if high is None else high
    frequencies = wavelet.log_frequencies(low, high, wavelet.PER_OCTAVE)
    frequencies = frequencies[frequencies < rate / 2.0]
    times, magnitude = wavelet.scalogram_peaks(raw, rate, frequencies,
                                               omega0)
    times = times + float(clock[0])
    largest = float(np.max(magnitude)) or 1.0
    decibels = 20.0 * np.log10(np.maximum(magnitude / largest,
                                          10.0 ** (-COLOR_RANGE / 20.0)))
    picture = layout.addPlot(row=2, col=0)
    scalogram_image(picture, decibels, times, frequencies, colors,
                    label='level', units='dB re the largest',
                    omega0=omega0, levels=(-COLOR_RANGE, 0.0))
    picture.getAxis('left').setWidth(LEFT_AXIS)

    # the corridor, over the picture and over its veil: the band is
    # what the controller applied, whatever the transform could see
    stride = max(1, len(waveform.time) // CORRIDOR_POINTS)
    t = waveform.time[::stride]
    drive = waveform.drive[::stride]
    half = setting.bandwidth(drive) / 2.0
    corridor = []
    for edge in (drive - half, drive + half):
        rows = np.where(edge > 0.0, np.log10(np.maximum(edge, 1e-300)),
                        np.nan)
        for pen in (pg.mkPen(colors['plot_background'], width=HALO_WIDTH),
                    pg.mkPen(colors['filter_preview'], width=CORRIDOR_WIDTH,
                             style=Qt.PenStyle.DashLine)):
            item = pg.PlotDataItem(t, rows, pen=pen, connect='finite')
            item.setZValue(25)
            picture.addItem(item, ignoreBounds=True)
            corridor.append(item)
    picture.corridor = corridor[1::2]    # the dashed lines, edge by edge

    # ---- the shape, on the picture's frequency axis --------------------------
    shape = layout.addPlot(row=2, col=1)
    shape.setYLink(picture)
    # the picture's decades as grid lines, unlabeled: the numbers are
    # on the picture's axis beside it, at the same heights
    shape.getAxis('left').setTicks([_decade_ticks(frequencies)])
    shape.getAxis('left').setStyle(showValues=False)
    shape.getAxis('left').setWidth(4)
    shape.setXRange(SHAPE_FLOOR, SHAPE_CEILING, padding=0)
    shape.setMouseEnabled(x=False, y=False)
    shape.hideButtons()
    shape.setMenuEnabled(False)
    shape.showGrid(x=True, y=True, alpha=0.2)
    shape.setLabel('bottom', 'band [dB]')
    shape.setMinimumWidth(SHAPE_WIDTH)
    shape.setMaximumWidth(SHAPE_WIDTH)
    readout = pg.LabelItem(justify='left', color=colors['plot_foreground'],
                           size='9pt')
    readout.setMinimumWidth(SHAPE_WIDTH)
    readout.setMaximumWidth(SHAPE_WIDTH)
    layout.addItem(readout, row=0, col=1)

    # ---- one time axis -------------------------------------------------------
    # The picture's color bar sits inside its plot, after the right
    # axis; the record gets an empty column of the bar's width in the
    # same place, so the two views have one left edge and one right
    # edge and the cursor is at one pixel on both.
    bar = picture.layout.itemAt(2, 5)
    if bar is not None:
        bar.setMinimumWidth(BAR_WIDTH)
        bar.setMaximumWidth(BAR_WIDTH)
    spacer = QGraphicsWidget()
    spacer.setMinimumWidth(BAR_WIDTH)
    spacer.setMaximumWidth(BAR_WIDTH)
    record_plot.layout.addItem(spacer, 2, 5)
    record_plot.layout.setColumnFixedWidth(4, 5)
    record_plot.setXLink(picture)
    left, right = (float(times[0]), float(times[-1])) if span is None \
        else (float(span[0]), float(span[1]))
    picture.setXRange(left, right, padding=0)

    # ---- the legend ----------------------------------------------------------
    legend = legend_below(layout, picture, 1, colors)
    hollow = pg.mkBrush(colors['plot_background'])
    samples = [
        (pg.PlotDataItem([], [], pen=pg.mkPen(colors['specification_curve'],
                                              width=2)), 'record'),
        (pg.PlotDataItem([], [], pen=pg.mkPen(colors['filter_preview'],
                                              width=2)), 'through the band'),
        (pg.PlotDataItem([], [], pen=pg.mkPen(
            colors['filter_preview'], width=CORRIDOR_WIDTH,
            style=Qt.PenStyle.DashLine)),
         'band edges, −3 dB'),
        (pg.PlotDataItem([], [], pen=pg.mkPen(None), fillLevel=0.0,
                         fillBrush=pg.mkBrush(_translucent(
                             colors['filter_preview'], WEIGHT_ALPHA))),
         "the band's weight on the record"),
    ]
    # the marks' samples are scatter items: a curve's sample paints its
    # symbol from the scatter it has not built yet for empty data, and
    # the two-tone mark came out filled in the scatter's default
    mark_pen = pg.mkPen(colors['plot_foreground'], width=1.5)
    samples += [(pg.ScatterPlotItem(symbol=symbol, size=MARK_SIZE,
                                    pen=mark_pen, brush=hollow), name)
                for symbol, name in [(DRIVE_SYMBOL, 'drive')]
                + [(HARMONIC_SYMBOL, f'{k} \u00d7 drive')
                   for k in harmonics]]
    for sample, name in samples:
        legend.addItem(sample, name)

    view = TrackingFilterView(record_plot, picture, shape, readout, waveform,
                              np.geomspace(frequencies[0], frequencies[-1],
                                           800),
                              harmonics, colors, (-top, top), interactive)
    view.set_cursor(float(waveform.time[len(waveform.time) // 2])
                    if cursor is None else float(cursor))
    # the view lives as long as the picture it draws on: its handlers
    # are what the cursor's drag calls
    picture.tracking = view
    return view
