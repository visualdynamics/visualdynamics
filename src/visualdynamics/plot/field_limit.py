"""A set of measured spectra drawn with the limit taken over them and
the straight-line envelope written over the limit.

One plot on log-log axes, in the unit system's units
(`core.field_limit`):

- every record of the set, thin and faint, named once in the legend
  with the count;
- the mean of the levels in decibels, dotted;
- the limit, solid and heavy, named by its statistic and count;
- the envelope, solid, each breakpoint a filled square numbered from
  the low end, as the breakpoint table reads;
- the gap between the envelope and the limit, hatched, with the area
  it adds (a PSD) or the most it stands above (an SRS) in the legend;
- with a margin, the envelope raised by it, dashed.

The title says what the limit is, what it rests on (the number of
records and the assumption it needs), the breakpoints, the overshoot
and the area added; a fit stopped by the breakpoints it was allowed
says so there. Nothing is told apart by hue alone: the curves differ by
weight and line style, the breakpoints by their markers, the gap by
its hatching, and every one is named.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _reading_plot

#: the breakpoint markers' size, pixels
MARKER_SIZE = 9

#: room above the tallest curve, in decades, for the numbers beside
#: the breakpoints
HEADROOM = 0.35


def summary(fit: Any) -> str:
    """The reading in one line, for the status bar: the limit and the
    envelope over it."""
    return f'{fit.limit.describe()}; {fit.describe()}'


def build_field_limit(layout: Any, fit: Any, *, margin_db: float = 0.0,
                      unit_system: Any = None, theme: Any = None) -> Any:
    """Draw `fit`, a `core.field_limit.EnvelopeFit`, into `layout` and
    return the plot.

    `margin_db`, when not zero, draws the envelope raised by it as well,
    on the set's convention (`EnvelopeFit.margined`).
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QBrush, QColor

    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM, UNKNOWN
    from . import axis_label, curve_color, legend_below

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    foreground = colors['plot_foreground']
    limit = fit.limit
    fieldset = limit.fieldset
    dim = fieldset.dim
    scale = 1.0 if dim == UNKNOWN else float(us.transform(dim)[0])
    f = fieldset.frequencies

    def shown(values):
        # a level is a size, so it converts by the unit's scale alone
        return np.asarray(values, dtype=float) / scale

    plot = _reading_plot(layout, 0)
    legend = legend_below(layout, plot, 0, colors)
    plot.setLogMode(x=True, y=True)

    faint = QColor(foreground)
    faint.setAlpha(70)
    for k, row in enumerate(fieldset.levels):
        plot.plot(f, shown(row), pen=pg.mkPen(faint, width=1),
                  name=(None if k else
                        f'the set, {fieldset.count} '
                        f'record{"s" * (fieldset.count != 1)}'))
    mean = 10.0 ** (limit.mean_db / fieldset.db_scale)
    plot.plot(f, shown(mean),
              pen=pg.mkPen(foreground, width=1.5, style=Qt.PenStyle.DotLine),
              name='mean in dB')
    plot.plot(f, shown(limit.levels),
              pen=pg.mkPen(curve_color(0, colors), width=3),
              name=limit.describe())

    # the gap between the envelope and the limit under it, hatched
    envelope = fit.at(f)
    hatch = QColor(foreground)
    hatch.setAlpha(120)
    upper = pg.PlotDataItem(f, shown(envelope), pen=pg.mkPen(None))
    lower = pg.PlotDataItem(f, shown(limit.levels), pen=pg.mkPen(None))
    # the two edges are put on the plot first, which is what puts them
    # on its log axes; a fill between curves the plot never held was
    # drawn on linear ones and stretched the frequency axis to 10^2000
    plot.addItem(upper)
    plot.addItem(lower)
    gap = pg.FillBetweenItem(upper, lower,
                             brush=QBrush(hatch, Qt.BrushStyle.BDiagPattern))
    gap.setZValue(-10)
    plot.addItem(gap)
    if limit.kind == 'psd':
        said = (f'area added, {fit.added_db:+.2f} dB' if fit.rule == 'exceed'
                else f'gap, {fit.below} line{"s" * (fit.below != 1)} under '
                f'the limit by up to {fit.deepest_db:.2f} dB')
    else:
        said = f'gap, up to {fit.overshoot_db:.2f} dB over the limit'
    legend.addItem(pg.PlotDataItem([0.0, 1.0], [1.0, 1.0], fillLevel=0.0,
                                   brush=QBrush(hatch,
                                                Qt.BrushStyle.BDiagPattern),
                                   pen=pg.mkPen(hatch)), said)

    line = colors['specification_curve']
    plot.plot(fit.frequencies, shown(fit.levels),
              pen=pg.mkPen(line, width=2.5), symbol='s',
              symbolSize=MARKER_SIZE, symbolPen=pg.mkPen(line),
              symbolBrush=pg.mkBrush(line),
              name=(f'envelope, {fit.breakpoints} breakpoints'
                    + ('' if fit.rule == 'exceed' else ', rms kept')))
    for k, (at, level) in enumerate(zip(fit.frequencies, fit.levels)):
        number = pg.TextItem(str(k + 1), color=foreground, anchor=(0.5, 1.2))
        number.setPos(float(np.log10(at)), float(np.log10(shown(level))))
        plot.addItem(number, ignoreBounds=True)
    top = float(np.nanmax(np.concatenate([shown(fieldset.levels).ravel(),
                                          shown(fit.levels)])))
    if margin_db:
        raised = fit.margined(margin_db)
        plot.plot(fit.frequencies, shown(raised),
                  pen=pg.mkPen(line, width=2, style=Qt.PenStyle.DashLine),
                  name=f'with {margin_db:+g} dB margin')
        top = max(top, float(np.nanmax(shown(raised))))
    low = float(np.nanmin(shown(fieldset.levels)))
    if np.isfinite(top) and np.isfinite(low) and top > 0.0 and low > 0.0:
        plot.setYRange(np.log10(low), np.log10(top) + HEADROOM, padding=0.02)
    plot.setXRange(np.log10(f[0]), np.log10(f[-1]), padding=0.02)
    plot.setLabel('left', axis_label(dim, us))
    plot.setLabel('bottom', f'frequency [{us.label_html("frequency")}]')
    title = f'{limit.describe()}<br>{limit.rests_on}<br>{fit.describe()}'
    plot.setTitle(title, color=foreground, size='9pt')
    return plot
