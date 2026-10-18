"""Whether a shaker can play a reference record: the travel against
time at one cutoff, and what every candidate cutoff saves and costs.

The top row is the armature's travel through the record at the cutoff
being read, with the body's share counted, inside the band the stroke
allows each way from the middle: the band's edges are the stops.
The largest excursion either way is marked and its second labeled. The
row is titled with the verdict in words and, under it, what the record
does integrated as logged, so a logger's offset shows as the drift it
is even while an edited record is drawn.

The bottom row is the sweep across the candidate cutoffs, three plots
on one cutoff axis: the travel each way against the room on each side,
the peak velocity against the rating (the armature's against the body,
as the travel is), and the price of the filter (the
share of the mean square it removed and what it took off the largest
peak). The cutoff drawn above is marked on each. A marker filled is a
cutoff that fits and a marker hollow one that does not, and each
series wears its own marker shape, so nothing is told by color alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _long_trace, _reading_plot

#: how solid the stroke band behind the travel is: light, since the
#: travel is what is read and the band only brackets it
BAND_ALPHA = 40

#: the marker of each series on the sweep, by what it shows
SYMBOLS = {'up': 't1', 'down': 't', 'velocity': 'o',
           'mean_square': 's', 'peak': 'd'}


def default_cutoff(fit: Any) -> float | None:
    """The cutoff drawn when none is asked for: the lowest that fits,
    the one that takes least out of the record, or the lowest candidate
    when none fits."""
    fitting = fit.lowest_fitting
    return (fitting if fitting is not None else fit.cases[0]).cutoff


def drift_words(fit: Any, unit_system: Any = None) -> str:
    """What the record does integrated as logged, in display units: the
    offset and what it alone integrates to, and when the travel leaves
    the room."""
    from ..units import DEFAULT_SYSTEM

    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    logged = fit.as_logged
    past = logged.first_past_stroke()
    offset = float(us.from_si(fit.offset, 'acceleration'))
    reach = float(us.from_si(fit.offset_displacement, 'length'))
    travel = float(us.from_si(logged.peak_to_peak, 'length'))
    return (f'As logged the travel spans {travel:.4g} '
            f'{us.label_text("length")}'
            + (f' and leaves the room {past:.3g} s in' if past is not None
               else ', inside the room')
            + f'; the record’s mean of {offset:.3g} '
            f'{us.label_text("acceleration")} alone integrates to '
            f'{reach:.4g} {us.label_text("length")} by its end.')


def build_replication_fit(layout: Any, fit: Any, cutoff: Any = ..., *,
                          unit_system: Any = None,
                          theme: Any = None) -> Any:
    """Draw a `core.replication_fit.ReplicationFit` into `layout`.

    `cutoff` is the candidate drawn on the top row, in Hz; None draws
    the record as logged, and omitted it is `default_cutoff`. Returns
    the case drawn.
    """
    import pyqtgraph as pg
    from PySide6.QtGui import QColor

    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from .sine_tracking import MARKER_SIZE

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    foreground = colors['plot_foreground']
    case = fit.case(default_cutoff(fit) if cutoff is ... else cutoff)
    rating = fit.rating
    length = us.label_text('length')

    def shown(values: Any, dimension: str) -> Any:
        return us.from_si(np.asarray(values, dtype=float), dimension)

    # the travel through the record, inside the stroke
    travel_plot = _reading_plot(layout, 0, 0, colspan=3)
    up = down = float(shown(rating.room, 'length'))
    fill = QColor(colors['averaging_band'])
    fill.setAlpha(BAND_ALPHA)
    band = pg.LinearRegionItem(values=(-down, up), orientation='horizontal',
                               movable=False, brush=pg.mkBrush(fill),
                               pen=pg.mkPen(fill))
    band.setZValue(-15)                   # under the trace it brackets
    travel_plot.addItem(band, ignoreBounds=True)
    stop = pg.mkPen(colors['limit_abort'], width=1.5,
                    style=pg.QtCore.Qt.PenStyle.DashLine)
    # the stops count toward the view, so the room is always on screen
    # and a travel well inside it reads as well inside it
    for edge in (up, -down):
        travel_plot.addItem(pg.InfiniteLine(pos=edge, angle=0, pen=stop))
    travel = shown(case.travel, 'length')
    _long_trace(travel_plot, case.time, travel,
                pg.mkPen(colors['response_curve'], width=1.2))
    at = int(np.argmax(np.abs(travel)))
    travel_plot.plot([case.time[at]], [travel[at]], pen=None, symbol='o',
                     symbolSize=MARKER_SIZE + 2,
                     symbolPen=pg.mkPen(colors['limit_abort'], width=2),
                     symbolBrush=pg.mkBrush(0, 0, 0, 0))
    # the label runs left of the mark near the record's end, where to
    # the right it would be cut off at the plot's edge
    late = case.time[at] > case.time[0] + 0.8 * (case.time[-1]
                                                  - case.time[0])
    mark = pg.TextItem(f'largest {travel[at]:+.3g} {length} at '
                       f'{case.time[at]:.3g} s', color=foreground,
                       anchor=(1 if late else 0, 1 if travel[at] >= 0 else 0))
    mark.setPos(float(case.time[at]), float(travel[at]))
    travel_plot.addItem(mark, ignoreBounds=True)
    counted = ('travel, body counted' if rating.body_share > 0.0
               else 'travel')
    travel_plot.setLabel('left', f'{counted} [{us.label_html("length")}]')
    travel_plot.setLabel('bottom', f'time [{us.label_html("time")}]')
    lines = [case.describe()]
    if case.cutoff is not None:
        lines.append(drift_words(fit, us))
    travel_plot.setTitle('<br>'.join(lines),
                         color=(foreground if case.fits
                                else colors['limit_abort']), size='9pt')

    # the sweep: what each cutoff saves and what it costs
    cutoffs = np.array([c.cutoff for c in fit.cases])
    stroke_plot = _reading_plot(layout, 1, 0)
    velocity_plot = _reading_plot(layout, 1, 1)
    price_plot = _reading_plot(layout, 1, 2)
    hollow = pg.mkBrush(0, 0, 0, 0)

    def series(plot, values, symbol, color, fits):
        plot.plot(cutoffs, values, pen=pg.mkPen(color, width=1.2))
        for x, y, ok in zip(cutoffs, values, fits):
            plot.plot([x], [y], pen=None, symbol=symbol,
                      symbolSize=MARKER_SIZE + 2,
                      symbolPen=pg.mkPen(color, width=1.5),
                      symbolBrush=pg.mkBrush(color) if ok else hollow)

    def limit(plot, value):
        plot.addItem(pg.InfiniteLine(pos=value, angle=0, pen=stop))

    stroke_fits = [c.fits_stroke for c in fit.cases]
    series(stroke_plot, shown([c.travel_up for c in fit.cases], 'length'),
           SYMBOLS['up'], colors['response_curve'], stroke_fits)
    series(stroke_plot, shown([c.travel_down for c in fit.cases], 'length'),
           SYMBOLS['down'], colors['specification_curve'], stroke_fits)
    limit(stroke_plot, up)
    stroke_plot.setTitle(f'travel [{length}]: ▲ up, ▼ down',
                         color=foreground, size='9pt')

    # the velocity the rating is judged against: the armature's against
    # the body, the table's when no body is counted
    series(velocity_plot,
           shown([c.relative_velocity for c in fit.cases], 'velocity'),
           SYMBOLS['velocity'], colors['response_curve'],
           [c.fits_velocity for c in fit.cases])
    limit(velocity_plot, float(shown(rating.velocity, 'velocity')))
    velocity_plot.setTitle(
        f'peak velocity{", body counted" * (rating.body_share > 0.0)} '
        f'[{us.label_text("velocity")}]', color=foreground, size='9pt')

    every = [True] * len(fit.cases)
    series(price_plot, 100.0 * np.array([c.mean_square_removed
                                         for c in fit.cases]),
           SYMBOLS['mean_square'], colors['response_curve'], every)
    series(price_plot, 100.0 * np.array([-c.peak_change
                                         for c in fit.cases]),
           SYMBOLS['peak'], colors['specification_curve'], every)
    price_plot.setTitle('price [%]: ■ mean square removed, '
                        '◆ largest peak lost', color=foreground,
                        size='9pt')

    chosen = pg.mkPen(foreground, width=1,
                      style=pg.QtCore.Qt.PenStyle.DotLine)
    for plot in (stroke_plot, velocity_plot, price_plot):
        plot.setLabel('bottom', f'cutoff [{us.label_html("frequency")}]')
        if case.cutoff is not None:
            plot.addItem(pg.InfiniteLine(pos=case.cutoff, angle=90,
                                         pen=chosen), ignoreBounds=True)
    layout.ci.layout.setRowStretchFactor(0, 3)
    layout.ci.layout.setRowStretchFactor(1, 2)
    return case


