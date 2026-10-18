"""A record's time statistics as three rows: the trace, its histogram,
its kurtosis against time.

The trace at the top, over the span the statistics were read from,
with the channel's range drawn across it when one was given, every
sample on the rail marked with a cross and the largest peak with a
diamond. Under it the histogram of the samples as a density, beside the
Gaussian bell with the record's own mean and standard deviation, with
3, 4 and 5 sigma marked either side and labeled with the share of the
record beyond each against the Gaussian's. At the bottom the kurtosis
frame by frame, the frame band shaded behind it, the frames outside the
band drawn filled where the rest are hollow, and the record's own
kurtosis as a dashed line across. A frame that does not vary, a
dropout, has no kurtosis to draw; it is marked at the band's low edge
with an upward triangle of its own.

Each row is titled with what it found in words, and the top row with
the judgement of the whole reading (`TimeStatistics.describe`), so the
plot carries its verdict. Nothing is told apart by hue alone: the
Gaussian is dashed where the histogram is solid, the rail samples are
crosses, the frames outside their band are filled triangles where the
others are hollow circles, and every mark is named in its legend.
Levels and times are drawn in the unit system's units.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _reading_plot

#: the bars of the histogram. Odd, so the middle bar is centered on
#: the record's middle value; enough that the bell's shoulders and a
#: clipped record's spikes at each edge are separate bars
BINS = 81

#: how solid the frame band behind the kurtosis is
BAND_ALPHA = 40

#: marker sizes: the rail's crosses and the frames' shapes
MARKER_SIZE = 8


def histogram(values: np.ndarray, bins: int = BINS
              ) -> tuple[np.ndarray, np.ndarray]:
    """(edges, density) of `values`: the histogram scaled so its area
    is one, over the span the values cover.

    The edges run from the smallest value to the largest, so a clipped
    record's samples on the rail fall in the outermost bars and stand
    there as spikes.
    """
    values = values[np.isfinite(values)]
    low, high = float(values.min()), float(values.max())
    if not high > low:
        high = low + 1.0
    density, edges = np.histogram(values, bins=bins, range=(low, high),
                                  density=True)
    return edges, density


def build_time_statistics(layout: Any, history: Any, stats: Any,
                          record: int = 0, *, unit_system: Any = None,
                          theme: Any = None) -> None:
    """Draw one channel's time statistics into `layout`.

    `stats` is the `core.time_statistics.TimeStatistics` of channel
    `record` of `history` (`TimeHistory.time_statistics`), whose span
    the trace is drawn over. Values are drawn in `unit_system`'s units.
    """
    import pyqtgraph as pg
    from PySide6.QtGui import QColor

    from ..core.kurtosis import NOMINAL, analyzed_span
    from ..core.time_statistics import SIGMAS
    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM, UNKNOWN
    from . import axis_label, curve_color, legend_below

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    foreground = colors['plot_foreground']
    alarm = colors['limit_abort']
    dimension = history.ordinate_dim[record]
    # an undefined dimension transforms as (1, 0): the raw numbers
    scale, offset = us.transform(dimension)

    def shown(values):
        return (values - offset) / scale

    tscale = us.transform('time')[0]
    hint = (history.dimension_hint[record]
            if dimension == UNKNOWN and history.dimension_hint else None)
    level_label = axis_label(dimension, us, hint)
    spans, phrase = analyzed_span(history)
    ordinate = np.real(np.asarray(history.ordinate)[record])
    clock = np.asarray(history.abscissa, dtype=float) / tscale
    hollow = pg.mkBrush(0, 0, 0, 0)

    # ---- the trace, over the span read -----------------------------
    trace = _reading_plot(layout, 0)
    legend = legend_below(layout, trace, 0, colors)
    pen = pg.mkPen(colors['response_curve'], width=1)
    for k, (low, high) in enumerate(spans):
        curve = trace.plot(clock[low:high], shown(ordinate[low:high]),
                           pen=pen, name=None if k else 'record')
        curve.setDownsampling(auto=True, method='peak')
        curve.setClipToView(True)
    if stats.full_scale is not None:
        rail_pen = pg.mkPen(colors['limit_abort'], width=1.5,
                            style=pg.QtCore.Qt.PenStyle.DashLine)
        for at in (float(shown(stats.full_scale)),
                   float(shown(-stats.full_scale))):
            line = pg.InfiniteLine(pos=at, angle=0, pen=rail_pen,
                                   movable=False)
            trace.addItem(line, ignoreBounds=False)
        legend.addItem(pg.PlotDataItem([], [], pen=rail_pen),
                       'range, either way')
        # every sample on the rail, a cross apiece: the flat tops a
        # clipped record wears, where they are
        edge = (1.0 - stats.margin) * stats.full_scale
        times, levels = [], []
        for low, high in spans:
            hit = np.flatnonzero(np.abs(ordinate[low:high]) >= edge) + low
            times.append(clock[hit])
            levels.append(ordinate[hit])
        times = np.concatenate(times)
        if times.size:
            trace.plot(times, shown(np.concatenate(levels)), pen=None,
                       symbol='x', symbolSize=MARKER_SIZE,
                       symbolPen=pg.mkPen(alarm, width=1.5),
                       symbolBrush=pg.mkBrush(alarm),
                       name=f'on the rail ({stats.rail})')
    trace.plot([stats.peak_time / tscale], [float(shown(stats.peak))],
               pen=None,
               symbol='d', symbolSize=MARKER_SIZE + 2,
               symbolPen=pg.mkPen(foreground), symbolBrush=hollow,
               name='largest peak')
    trace.setLabel('left', level_label)
    trace.setLabel('bottom', f'time [{us.label_html("time")}]')
    flagged = bool(stats.flagged)
    trace.setTitle(f'{history.record_label(record)}, {phrase}. '
                   + stats.describe(),
                   color=alarm if flagged else foreground, size='9pt')

    # ---- the histogram against the Gaussian -------------------------
    bell = _reading_plot(layout, 2)
    legend = legend_below(layout, bell, 1, colors)
    values = np.concatenate([ordinate[low:high] for low, high in spans])
    edges, density = histogram(np.asarray(shown(values), dtype=float))
    # the density in display units: the area stays one, so it scales
    # inversely with the unit the values are shown in
    fill = QColor(colors['response_curve'])
    fill.setAlpha(70)
    bell.plot(edges, density, stepMode='center', fillLevel=0.0,
              pen=pg.mkPen(colors['response_curve'], width=1.5),
              brush=pg.mkBrush(fill), name='record')
    mean = float(shown(stats.mean))
    # a standard deviation is a difference, so it converts by the
    # unit's scale alone, without the offset a temperature would carry
    sigma = stats.std / scale
    if sigma > 0.0:
        reach = max(abs(edges[0] - mean), abs(edges[-1] - mean),
                    (max(SIGMAS) + 0.5) * sigma)
        grid = np.linspace(mean - reach, mean + reach, 801)
        gauss = (np.exp(-0.5 * ((grid - mean) / sigma) ** 2)
                 / (sigma * np.sqrt(2.0 * np.pi)))
        bell.plot(grid, gauss,
                  pen=pg.mkPen(colors['specification_curve'], width=2,
                               style=pg.QtCore.Qt.PenStyle.DashLine),
                  name='Gaussian, same mean and σ')
        mark = pg.mkPen(foreground, width=1,
                        style=pg.QtCore.Qt.PenStyle.DotLine)
        tails = {tail.sigmas: tail for tail in stats.tails}
        for k in SIGMAS:
            tail = tails.get(float(k))
            out = tail is not None and tail.flagged
            for side in (1.0, -1.0):
                line = pg.InfiniteLine(
                    pos=mean + side * k * sigma, angle=90, pen=mark,
                    movable=False,
                    label=f'{"" if side > 0 else "−"}{k}σ',
                    labelOpts={'position': 0.95, 'color': (
                        alarm if out else foreground)})
                bell.addItem(line, ignoreBounds=True)
        legend.addItem(pg.PlotDataItem([], [], pen=mark),
                       '3, 4 and 5 σ either side of the mean')
    bell.setLabel('left', 'probability density')
    bell.setLabel('bottom', level_label)
    beyond = ', '.join(f'{tail.sigmas:g}σ {percent(tail.share)}'
                       + (' (flagged)' if tail.flagged else '')
                       for tail in stats.tails)
    gaussian = ', '.join(percent(tail.expected) for tail in stats.tails)
    bell.setTitle(f'Histogram: skewness {stats.skewness:.3f}, kurtosis '
                  f'{stats.kurtosis:.3f} (Gaussian 0 and 3)<br>'
                  f'beyond {beyond} (Gaussian {gaussian})',
                  color=(alarm if any(tail.flagged for tail in stats.tails)
                         else foreground), size='9pt')

    # ---- the kurtosis against time ----------------------------------
    frames = _reading_plot(layout, 4)
    legend = legend_below(layout, frames, 2, colors)
    low, high = stats.frame_band
    shade = QColor(colors['averaging_band'])
    shade.setAlpha(BAND_ALPHA)
    band = pg.LinearRegionItem(values=(low, high), orientation='horizontal',
                               movable=False, brush=pg.mkBrush(shade),
                               pen=pg.mkPen(shade))
    band.setZValue(-15)
    frames.addItem(band, ignoreBounds=True)
    legend.addItem(pg.PlotDataItem([], [], pen=pg.mkPen(shade, width=8)),
                   f'frame band {low:g} to {high:g}')
    color = curve_color(0, colors)
    t, k_values = stats.frame_times / tscale, stats.frame_kurtosis
    time_unit = us.label_text('time')
    if t.size:
        seconds = stats.frame_length / stats.sample_rate / tscale
        frames.plot(t, k_values, pen=pg.mkPen(color, width=1.5),
                    symbol='o', symbolSize=MARKER_SIZE - 2,
                    symbolPen=pg.mkPen(color), symbolBrush=hollow,
                    name=f'each {seconds:.3g} {time_unit} frame')
        outside = stats.frames_outside_mask & np.isfinite(k_values)
        if outside.any():
            frames.plot(t[outside], k_values[outside], pen=None,
                        symbol='t', symbolSize=MARKER_SIZE + 2,
                        symbolPen=pg.mkPen(alarm),
                        symbolBrush=pg.mkBrush(alarm),
                        name='frame outside its band')
        # a frame that does not vary has no kurtosis; it is outside its
        # band all the same, and is marked where the band begins
        dead = ~np.isfinite(k_values)
        if dead.any():
            frames.plot(t[dead], np.full(int(dead.sum()), low), pen=None,
                        symbol='t1', symbolSize=MARKER_SIZE + 2,
                        symbolPen=pg.mkPen(alarm),
                        symbolBrush=pg.mkBrush(alarm),
                        name='frame with no variance')
    frames.addItem(pg.InfiniteLine(
        pos=NOMINAL, angle=0, movable=False,
        pen=pg.mkPen(foreground, width=1,
                     style=pg.QtCore.Qt.PenStyle.DotLine)))
    record_pen = pg.mkPen(colors['specification_curve'], width=2,
                          style=pg.QtCore.Qt.PenStyle.DashLine)
    frames.addItem(pg.InfiniteLine(pos=stats.kurtosis, angle=0,
                                   movable=False, pen=record_pen))
    legend.addItem(pg.PlotDataItem([], [], pen=record_pen),
                   f'record {stats.kurtosis:.3f}, band '
                   f'{stats.kurtosis_band[0]:g} to {stats.kurtosis_band[1]:g}')
    frames.setXLink(trace)
    frames.setLabel('left', 'Pearson kurtosis')
    frames.setLabel('bottom', f'time [{us.label_html("time")}]')
    worst = stats.worst_frame
    words = (f'{stats.frames_outside} of {t.size} frames outside '
             f'{low:g} to {high:g}' if t.size
             else 'the record is shorter than one frame')
    if worst is not None and not np.isfinite(worst[1]):
        words += (f'; no variance at {worst[0] / tscale:.4g} '
                  f'{time_unit}')
    elif worst is not None:
        words += (f'; furthest from 3 at {worst[0] / tscale:.4g} '
                  f'{time_unit}, {worst[1]:.3f}')
    frames.setTitle(f'Kurtosis frame by frame: {words}',
                    color=alarm if stats.frames_outside else foreground,
                    size='9pt')


def percent(share: float) -> str:
    """A share as a percentage to two significant figures, as the
    Gaussian's tails are quoted: 0.27 %, 0.0063 %."""
    if not np.isfinite(share):
        return '—'
    if share == 0.0:
        return '0 %'
    # positional, never 5.7e-05: a share is read beside its neighbors
    return np.format_float_positional(100.0 * share, precision=2,
                                      unique=False, fractional=False,
                                      trim='-') + ' %'


