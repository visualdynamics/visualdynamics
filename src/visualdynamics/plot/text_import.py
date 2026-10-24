"""The channels of an import as read: one row each, missing samples and
dead channels marked.

What a person checks before accepting an import of text is that each
column landed where it should: the time running the right way at the
right rate, each channel's level in its unit, and nothing missing or
flat that should not be. So each channel is drawn in a row of its own
against time, titled with its label, its unit and what
`core.channel_health` found in it, in words. A missing sample has no
value to draw, so it is marked with a cross on the row's zero line at
its time; a dead channel is drawn as the flat line it is, dashed, and
titled "constant at"; a channel with no values at all is titled so and left
empty. The rows share their time axis.

Nothing is told apart by hue alone: the finding is in the title's words
(drawn in the alarm color as well), a dead channel's line is dashed, and
the missing samples are crosses named in the legend under the last
row.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _long_trace, _reading_plot

#: the most channels drawn; beyond it the last title says how many more
MAX_ROWS = 8

#: the cross marking a missing sample
MARKER_SIZE = 8


def build_import_preview(layout: Any, history: Any, *,
                         unit_system: Any = None, theme: Any = None,
                         rows: int = MAX_ROWS) -> list:
    """Draw each channel of `history` in its own row of `layout`.

    Values are drawn in `unit_system`'s units; a channel with no units
    is drawn as its numbers. At most `rows` channels are drawn.

    Returns
    -------
    list
        The plots, one per channel drawn, top to bottom.
    """
    import pyqtgraph as pg

    from ..core.channel_health import channel_health
    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM, UNKNOWN
    from . import axis_label, curve_color, legend_below

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    foreground, alarm = colors['plot_foreground'], colors['limit_abort']
    tscale = us.transform('time')[0]
    clock = np.asarray(history.abscissa, dtype=float) / tscale
    ordinate = np.real(np.asarray(history.ordinate))
    health = {finding.record: finding for finding in channel_health(history)}
    shown = min(history.num_records, max(int(rows), 1))
    cross = pg.mkPen(alarm, width=1.5)
    plots, marked = [], False
    for i in range(shown):
        plot = _reading_plot(layout, 2 * i)
        dimension = history.ordinate_dim[i]
        scale, offset = us.transform(dimension)
        values = (ordinate[i] - offset) / scale
        finite = np.isfinite(values)
        finding = health.get(i)
        if finite.any():
            # drawn where it has values, a gap breaking the line as it
            # breaks the record; a dead channel dashed, so its flat line
            # is not read as a quiet live one
            pen = pg.mkPen(curve_color(i, colors), width=1)
            if finding is not None and finding.dead:
                pen = pg.mkPen(curve_color(i, colors), width=1.5,
                               style=pg.QtCore.Qt.PenStyle.DashLine)
            _long_trace(plot, clock, values, pen)
        words = ''
        if finding is not None and not finite.any():
            words = 'no values: every sample is missing'
        elif finding is not None:
            said = []
            if finding.missing:
                said.append(f'{finding.missing} missing (crosses), the first at '
                            f'{clock[finding.first_missing]:.4g} '
                            f'{us.label_text("time")}')
                plot.plot(clock[~finite], np.zeros(int((~finite).sum())),
                          pen=None, symbol='x', symbolSize=MARKER_SIZE,
                          symbolPen=cross, symbolBrush=pg.mkBrush(alarm))
                marked = True
            if finding.dead:
                said.append(f'constant at {float(values[finite][0]):g}: a dead '
                            'or unconnected channel')
            words = '; '.join(said)
        hint = (history.dimension_hint[i]
                if dimension == UNKNOWN and history.dimension_hint else None)
        plot.setLabel('left', axis_label(dimension, us, hint))
        label = history.record_label(i) or f'record {i + 1}'
        more = history.num_records - shown
        tail = (f' (and {more} more channel{"s" * (more != 1)} not drawn)'
                if i == shown - 1 and more else '')
        plot.setTitle(f'{label}' + (f': {words}' if words else '') + tail,
                      color=alarm if words else foreground, size='9pt')
        if plots:
            plot.setXLink(plots[0])
        plots.append(plot)
    plots[-1].setLabel('bottom', f'time [{us.label_html("time")}]')
    if marked:
        legend = legend_below(layout, plots[-1], shown - 1, colors)
        legend.addItem(pg.PlotDataItem([], [], pen=None, symbol='x',
                                       symbolPen=cross,
                                       symbolBrush=pg.mkBrush(alarm)),
                       'missing sample, at its time')
    return plots
