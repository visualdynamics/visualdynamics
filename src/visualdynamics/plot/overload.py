"""Each channel against its range: the worst one's trace over its range,
and every channel's headroom beside its kurtosis.

At the top, one channel's trace over the span read, with its range drawn
across it either way as dashed lines, every sample on the rail marked
with a cross, the largest peak with a diamond, and, where the range has
held, that peak scaled to the next level drawn either way as dotted
lines, so a range that will not hold the next step is seen against the
peak that says so. The row is titled with the channel's judgement in
words.

Under it, two bar charts on one row, a bar per channel and the channels
in the same order on both, their rows locked together. On the left the
headroom of each channel's range over its largest peak, in dB, grown
from zero, with the step to the next level as a floor: a bar short of it
reaches its range at the next level, and a bar of zero is on the rail.
Each channel flagged has its finding written along its row in words
('on the rail, 287', 'reaches its range at +6 dB', and 'beyond its
range' where the table's range cannot be the one the record was taken
with), and a channel with no range says why in place of a bar.
On the right the kurtosis bars (`plot.bars.kurtosis_chart`), unchanged,
because the kurtosis is what a reader looks at for a clip and its band
of 2 to 4 passes one at 2.86: the two charts side by side show the
clipped channel with an ordinary kurtosis and no headroom.

Nothing is told apart by hue alone: the range is dashed and the scaled
peak dotted, the rail samples are crosses and the peak a diamond, every
mark is named in the legend, and every flagged bar carries its finding
in words. Levels and times are drawn in the unit system's units.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from . import _reading_plot

#: marker sizes: the rail's crosses and the peak's diamond
MARKER_SIZE = 8


def finding(channel: Any) -> str:
    """What a channel is flagged for, in the few words its bar
    carries; '' when it is not flagged."""
    flagged = channel.flagged
    if 'beyond' in flagged:
        return f'beyond its range, {channel.beyond}'
    if 'rail' in flagged:
        return f'on the rail, {channel.rail}'
    if 'next' in flagged:
        return f'reaches its range at +{channel.step_db:g} dB'
    return ''


def build_overload(layout: Any, history: Any, check: Any,
                   record: int | None = None, *,
                   kurtosis_band: tuple[float, float] | None = None,
                   unit_system: Any = None, theme: Any = None,
                   step_changed: Callable[..., None] | None = None,
                   kurtosis_changed: Callable[..., None] | None = None
                   ) -> tuple[Any, Any]:
    """Draw an overload check into `layout`.

    `check` is the `core.overload.OverloadCheck` of `history`
    (`TimeHistory.overload_check`). The trace is channel `record`, or
    the check's worst channel (`OverloadCheck.worst`) when `record` is
    None, or its first when no channel has a range. `kurtosis_band` is
    the kurtosis bars' band, `core.kurtosis`'s 2 to 4 by default.
    `step_changed` is called with the headroom floor when it is
    dragged, and `kurtosis_changed` with the kurtosis band, as
    `plot.bars.BarChart` calls them. Returns the two charts, headroom
    then kurtosis.
    """
    import pyqtgraph as pg

    from ..core.kurtosis import HIGH, LOW, analyzed_span, channel_kurtosis
    from ..core.overload import headroom_rows, notes_for
    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM, UNKNOWN
    from . import axis_label, legend_below
    from .bars import BarChart, kurtosis_chart

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    foreground = colors['plot_foreground']
    alarm = colors['limit_abort']
    channels = list(check.channels)
    if record is None:
        chosen = check.worst or (channels[0] if channels else None)
    else:
        chosen = next((channel for channel in channels
                       if channel.record == int(record)), None)
    hollow = pg.mkBrush(0, 0, 0, 0)

    # ---- the trace over its range -------------------------------------
    trace = _reading_plot(layout, 0, colspan=2)
    legend = legend_below(layout, trace, 0, colors)
    spans, phrase = analyzed_span(history)
    if chosen is not None:
        index = chosen.record
        dimension = history.ordinate_dim[index]
        # an undefined dimension transforms as (1, 0): the raw numbers
        scale, offset = us.transform(dimension)
        unit = '' if dimension == UNKNOWN else us.label_text(dimension)
        after = f' {unit}' if unit else ''

        def shown(values):
            return (values - offset) / scale

        tscale = us.transform('time')[0]
        ordinate = np.real(np.asarray(history.ordinate)[index])
        clock = np.asarray(history.abscissa, dtype=float) / tscale
        pen = pg.mkPen(colors['response_curve'], width=1)
        for k, (low, high) in enumerate(spans):
            curve = trace.plot(clock[low:high], shown(ordinate[low:high]),
                               pen=pen, name=None if k else 'record')
            curve.setDownsampling(auto=True, method='peak')
            curve.setClipToView(True)
        stats = chosen.statistics
        if chosen.has_range:
            # a range is a size, so it is drawn by the unit's scale
            # alone, either side of zero
            reach = chosen.full_scale / scale
            rail_pen = pg.mkPen(alarm, width=1.5,
                                style=pg.QtCore.Qt.PenStyle.DashLine)
            for at in (reach, -reach):
                trace.addItem(pg.InfiniteLine(pos=at, angle=0, pen=rail_pen,
                                              movable=False))
            legend.addItem(pg.PlotDataItem([], [], pen=rail_pen),
                           f'range, either way ({reach:.4g}{after})')
            # the peak the next level would bring, where the range has
            # held: a clipped record's peak is the range, and scaled up
            # it predicts nothing
            if not stats.rail and not chosen.beyond:
                ahead = chosen.next_peak / scale
                next_pen = pg.mkPen(foreground, width=1.5,
                                    style=pg.QtCore.Qt.PenStyle.DotLine)
                for at in (ahead, -ahead):
                    trace.addItem(pg.InfiniteLine(pos=at, angle=0,
                                                  pen=next_pen,
                                                  movable=False))
                legend.addItem(pg.PlotDataItem([], [], pen=next_pen),
                               f'largest peak at +{chosen.step_db:g} dB '
                               f'({ahead:.4g}{after})')
            # every sample on the rail, a cross apiece: the flat tops a
            # clipped record wears, where they are
            edge = (1.0 - stats.margin) * chosen.full_scale
            times, levels = [], []
            for low, high in spans:
                hit = np.flatnonzero(np.abs(ordinate[low:high]) >= edge) + low
                times.append(clock[hit])
                levels.append(ordinate[hit])
            times = np.concatenate(times) if times else np.empty(0)
            if times.size:
                trace.plot(times, shown(np.concatenate(levels)), pen=None,
                           symbol='x', symbolSize=MARKER_SIZE,
                           symbolPen=pg.mkPen(alarm, width=1.5),
                           symbolBrush=pg.mkBrush(alarm),
                           name=f'on the rail ({stats.rail})')
        if stats is not None:
            trace.plot([stats.peak_time / tscale], [float(shown(stats.peak))],
                       pen=None, symbol='d', symbolSize=MARKER_SIZE + 2,
                       symbolPen=pg.mkPen(foreground), symbolBrush=hollow,
                       name='largest peak')
        hint = (history.dimension_hint[index]
                if dimension == UNKNOWN and history.dimension_hint else None)
        trace.setLabel('left', axis_label(dimension, us, hint))
        trace.setLabel('bottom', f'time [{us.label_html("time")}]')
        trace.setTitle(f'{phrase.capitalize()}. '
                       + chosen.describe(unit, scale),
                       color=alarm if chosen.flagged else foreground,
                       size='9pt')

    # ---- the headroom and the kurtosis, a bar per channel ---------------
    # a channel with no range has a headroom of NaN, drawn as no bar,
    # its row carrying the reason instead; the count over the bars is
    # left off, since the title says it in the check's own terms
    notes = notes_for(check)
    left = _reading_plot(layout, 2, 0)
    headroom = BarChart(left, headroom_rows(check), colors, low=check.step_db, high=None,
                        changed=step_changed,
                        label='headroom of the range over the peak',
                        units='dB', neutral='specification_curve',
                        side='under', summary='none', notes=notes)
    flagged = check.flagged
    for row, channel in enumerate(channels):
        words = finding(channel)
        if not words:
            continue
        text = pg.TextItem(words, anchor=(0.0, 0.5), color=alarm)
        text.setPos(max(float(channel.headroom_db), 0.0), float(row))
        left.addItem(text, ignoreBounds=True)
    left.setTitle(f'Headroom: {len(flagged)} of {len(check.checked)} '
                  f'flagged, floor +{check.step_db:g} dB (the next level)',
                  color=alarm if flagged else foreground, size='9pt')

    right = _reading_plot(layout, 2, 1)
    low, high = kurtosis_band or (LOW, HIGH)
    shapes = channel_kurtosis(history,
                              [channel.record for channel in channels])
    kurtosis = kurtosis_chart(right, shapes, colors, low=low, high=high,
                              changed=kurtosis_changed)
    right.setYLink(left)
    # the two charts share the row's width: the headroom's words and
    # the kurtosis' names would otherwise take it in turn
    grid = getattr(layout, 'ci', layout).layout
    grid.setColumnStretchFactor(0, 1)
    grid.setColumnStretchFactor(1, 1)
    clipped = check.with_flag('rail')
    passed = [channel.label for channel in clipped
              if channel.statistics is not None
              and low <= channel.statistics.kurtosis <= high]
    right.setTitle('Kurtosis'
                   + (f': {", ".join(passed)} on the rail and inside '
                      f'{low:g} to {high:g}' if passed else ''),
                   color=foreground, size='9pt')
    return headroom, kurtosis
