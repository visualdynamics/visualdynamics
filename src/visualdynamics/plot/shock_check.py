"""A shock record checked before its spectrum is read: three rows.

The record at the top, its velocity integrated raw beneath it, and the
positive and negative shock response spectra at the bottom, all for
one channel over one window. The velocity row carries the verdict in
words as its title, and the band either side of zero that the velocity
has to end inside, so the claim and its evidence are on one plot: a
velocity that comes back into the band passes, and one that ends
outside it, held or still ramping, is flagged.

The two spectra are drawn together because they part where the shock
was one-sided, and a maximax curve, the larger of the two at every
frequency, shows only the upper one. They are told apart by a marker
shape as well as a color, as the overlaid readings of a tone are
(`plot.sine_tracking`), so the plot reads in gray and to a reader who
cannot separate the palette's colors.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: the marker each polarity wears, in `NAMES` order
SYMBOLS = ('o', 't')

#: the polarities drawn on the spectrum row, and what the legend calls
#: them
NAMES = (('positive', 'positive SRS'), ('negative', 'negative SRS'))

#: how solid the tolerance band behind the velocity is: light, since
#: the velocity is what is being read and the band only brackets it
BAND_ALPHA = 40


def build_shock_check(layout: Any, history: Any, check: Any,
                      spectra: dict, *, unit_system: Any = None,
                      theme: Any = None) -> None:
    """Draw one channel's check over one window into `layout`.

    `check` is a `core.shock_check.ShockCheck` of `history`, and
    `spectra` is `TimeHistory.compute_srs_polarities` over that one
    window, so it holds one curve per channel in the record's order and
    the check's channel is drawn by its index. The acceleration and the
    velocity share their time axis.
    """
    import pyqtgraph as pg
    from PySide6.QtGui import QColor

    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import axis_label, curve_color, legend_below
    from .scalogram import _decade_ticks
    from .sine_tracking import MARKER_SIZE, _marker_spots

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    foreground = colors['plot_foreground']
    # by index, not DOF: an acceleration and a velocity can share one
    # (2026-10-16, review of #141, which found the first of two taken)
    record = check.record
    rate = check.sample_rate
    first = round((check.start - float(history.abscissa[0])) * rate)
    time = check.time
    time_label = f'time [{us.label_html("time")}]'

    # the record over the window
    record_plot = _plot(layout, 0)
    acceleration = np.real(np.asarray(history.ordinate)[
        record, first:first + time.size])
    record_plot.plot(time, us.from_si(acceleration, 'acceleration'),
                     pen=pg.mkPen(colors['response_curve'], width=1))
    record_plot.setLabel('left', axis_label('acceleration', us))
    record_plot.setTitle(f'{check.dof}, the record', color=foreground,
                         size='9pt')

    # its velocity, raw, with the band it has to end inside
    velocity_plot = _plot(layout, 2)
    velocity_plot.setXLink(record_plot)
    reach = float(us.from_si(check.tolerance * check.peak_velocity,
                             'velocity'))
    fill = QColor(colors['averaging_band'])
    fill.setAlpha(BAND_ALPHA)
    band = pg.LinearRegionItem(values=(-reach, reach),
                               orientation='horizontal', movable=False,
                               brush=pg.mkBrush(fill),
                               pen=pg.mkPen(fill))
    band.setZValue(-15)                   # under the trace it brackets
    velocity_plot.addItem(band, ignoreBounds=True)
    velocity_plot.plot(time, us.from_si(check.velocity, 'velocity'),
                       pen=pg.mkPen(colors['response_curve'], width=1.5))
    velocity_plot.setLabel('left', axis_label('velocity', us))
    velocity_plot.setLabel('bottom', time_label)
    # a sentence to a line: the verdict is read, and a line wider than
    # the plot is cut off at its edge
    velocity_plot.setTitle('<br>'.join(_sentences(check.describe())),
                           color=(colors['limit_abort'] if check.flagged
                                  else foreground), size='9pt')

    # the two spectra, on log axes
    srs_plot = _plot(layout, 4)
    srs_plot.setLogMode(x=True, y=True)
    legend = legend_below(layout, srs_plot, 2, colors)
    first_curve = next(iter(spectra.values()))
    frequencies = np.asarray(first_curve.abscissa, dtype=float)
    hollow = pg.mkBrush(0, 0, 0, 0)
    reached = []
    for k, ((kind, name), symbol) in enumerate(zip(NAMES, SYMBOLS)):
        color = curve_color(k, colors)
        values = us.from_si(np.asarray(spectra[kind].ordinate[record].real,
                                       dtype=float), 'acceleration')
        # a log axis has no zero: a polarity that never responded is a
        # gap, not a line drawn at the bottom of the plot
        values = np.where(values > 0.0, values, np.nan)
        if np.isfinite(values).any():
            reached += [float(np.nanmin(values)), float(np.nanmax(values))]
        pen = pg.mkPen(color, width=1.5)
        srs_plot.plot(frequencies, values, pen=pen, connect='finite')
        spots = _marker_spots(frequencies, k, len(NAMES), True)
        spots = spots[np.isfinite(values[spots])]
        srs_plot.plot(frequencies[spots], values[spots], pen=None,
                      symbol=symbol, symbolSize=MARKER_SIZE,
                      symbolPen=pg.mkPen(color), symbolBrush=hollow)
        legend.addItem(pg.PlotDataItem(
            [], [], pen=pen, symbol=symbol, symbolSize=MARKER_SIZE,
            symbolPen=pg.mkPen(color), symbolBrush=hollow), name)
    srs_plot.setLabel('left', axis_label('acceleration', us))
    srs_plot.setLabel('bottom',
                      f'natural frequency [{us.label_html("frequency")}]')
    # 1, 2 and 5 a decade, as the scalogram labels its frequencies: the
    # lines are twelfths of an octave and pyqtgraph's own log ticks
    # crowd them into one another
    srs_plot.getAxis('bottom').setTicks(
        [_decade_ticks(frequencies[[0, -1]])])
    if reached:
        srs_plot.getAxis('left').setTicks(
            [_level_ticks(min(reached), max(reached))])
    srs_plot.setTitle(f'shock response spectra at Q = {first_curve.q:g}',
                      color=foreground, size='9pt')


def _plot(layout: Any, row: int) -> Any:
    """A plot in `row` of the layout, gridded, with no SI scaling."""
    plot = layout.addPlot(row=row, col=0)
    plot.showGrid(x=True, y=True, alpha=0.2)
    # the units are in the labels, and a scale of 'x0.001' beside an
    # axis already labeled in seconds says the same thing twice. Turned
    # off before anything is drawn: turned off after, an axis drawn at
    # the old scale kept its numbers and read 250 s for 0.25 s
    for side in ('left', 'bottom'):
        plot.getAxis(side).enableAutoSIPrefix(False)
    return plot


def _level_ticks(low: float, high: float) -> list[tuple[float, str]]:
    """(position, text) up the spectra's log level axis: 1, 2 and 5 a
    decade over three decades or less, powers of ten alone over more,
    where 1, 2 and 5 would stack into one another."""
    from .scalogram import _decade_ticks

    ticks = _decade_ticks(np.array([low, high]))
    if np.log10(high / low) > 3.0:
        ticks = [(at, text) for at, text in ticks
                 if np.isclose(at, round(at))]
    return ticks


def _sentences(text: str) -> list[str]:
    """`text` cut into its sentences, each keeping its full stop."""
    return [part if part.endswith('.') else part + '.'
            for part in text.split('. ') if part]

