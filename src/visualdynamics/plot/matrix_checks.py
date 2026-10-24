"""The two matrix checks of a multi-input test, drawn against frequency.

A `Conditioning` (`core.matrix_checks`) holds the singular values of an
FRF or a CSD at every line, and a `Realizability` the eigenvalues of a
cross-spectral matrix's coherence matrix with its repair. Each is drawn
as a figure of its own, and every curve is told apart by line style and
named in words, never by hue alone:

- `build_conditioning`: above, on log axes, the singular values, the
  largest solid and each smaller one in a style of its own down the
  list, with the rank threshold (the largest over the limit) dotted;
  below, the condition number with the limit dashed and the lines over
  it hatched;
- `build_realizability`: above, the eigenvalues of the coherence
  matrix on a linear axis, so a negative one stands below the zero line
  drawn dashed, the smallest solid, the largest dashed and the rest
  dotted, the lines that are not realizable hatched; and, when the
  repair moved anything, the coherence and the phase of the pair it
  moved most, before solid and after dashed.

The hatching is a diagonal line pattern, which reads without color.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: the styles the singular values wear down the list, largest first;
#: past the last, the last repeats and the name says which it is
SINGULAR_STYLES = ('solid', 'dash', 'dashdot', 'dot')

#: how opaque the hatching's lines are, of 255
HATCH_ALPHA = 110


def _pen(color: Any, width: float, style: str) -> Any:
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    styles = {'solid': Qt.PenStyle.SolidLine,
              'dash': Qt.PenStyle.DashLine,
              'dashdot': Qt.PenStyle.DashDotLine,
              'dot': Qt.PenStyle.DotLine}
    return pg.mkPen(color, width=width, style=styles[style])


def singular_name(k: int) -> str:
    """What the `k`-th singular value (from 0) is called in the legend."""
    return 'σ1, largest' if k == 0 else f'σ{k + 1}'


def hatch_bands(plot: Any, frequency: Any, flags: Any, color: Any) -> list:
    """Hatch the runs of flagged lines on `plot`, each from halfway to
    the line before to halfway to the line after, and return the items.
    On a log frequency axis the edges are placed in decades."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QBrush

    from ..core.matrix_checks import flagged_runs

    f = np.asarray(frequency, dtype=float)
    log_x = plot.getViewBox().state['logMode'][0]
    hatch = pg.mkColor(color)
    hatch.setAlpha(HATCH_ALPHA)
    brush = QBrush(hatch, Qt.BrushStyle.BDiagPattern)
    items = []
    for low, high in flagged_runs(f, np.asarray(flags, dtype=bool)):
        i, j = int(np.searchsorted(f, low)), int(np.searchsorted(f, high))
        left = 0.5 * (f[i] + f[i - 1]) if i > 0 else f[i]
        right = 0.5 * (f[j] + f[j + 1]) if j + 1 < len(f) else f[j]
        if log_x:
            if right <= 0.0:
                continue
            left = max(left, right * 1e-3) if left <= 0.0 else left
            edges = (float(np.log10(left)), float(np.log10(right)))
        else:
            edges = (float(left), float(right))
        region = pg.LinearRegionItem(values=edges, movable=False,
                                     brush=brush, pen=pg.mkPen(None))
        region.setZValue(-10)
        plot.addItem(region, ignoreBounds=True)
        items.append(region)
    return items


def _title(reading: Any) -> str:
    title = reading.describe()
    if reading.notes:
        title += '<br>' + '<br>'.join(reading.notes)
    return title


def _fit_band(plot: Any, band: Any, log_x: bool) -> None:
    if band is None:
        return
    low, high = sorted(float(edge) for edge in band)
    if log_x:
        if not low > 0.0:
            raise ValueError('a band on a log frequency axis starts above '
                             'zero')
        plot.setXRange(np.log10(low), np.log10(high), padding=0.0)
    else:
        plot.setXRange(low, high, padding=0.0)


def build_conditioning(layout: Any, reading: Any, *,
                       band: tuple[float, float] | None = None,
                       unit_system: Any = None,
                       theme: Any = None) -> tuple[Any, Any]:
    """Draw a conditioning reading into a graphics layout and return its
    two plots, (singular values, condition number).

    Above, on log axes, the singular values in the matrix's units, the
    largest solid and the smaller ones dashed, dash-dotted and dotted in
    turn, each named, with the rank threshold (the largest over the
    limit) dotted thin; below, the condition number on a log axis with
    the limit dashed and the lines over it hatched, the two plots
    sharing the frequency axis. `band` is the (low, high) frequencies
    shown, in Hz; the whole reading when omitted. The verdict and the
    notes go in the title.
    """
    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM, UNKNOWN
    from . import _reading_plot, axis_label, legend_below

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    ink = colors['plot_foreground']
    scale = (1.0 if reading.dimension == UNKNOWN
             else float(us.from_si(1.0, reading.dimension)))
    f = np.asarray(reading.frequency, dtype=float)
    judged = reading.judged & (f > 0.0)

    top = _reading_plot(layout, 0)
    top.setLogMode(x=True, y=True)
    legend_below(layout, top, 0, colors)
    top.setTitle(_title(reading), color=ink, size='9pt')
    for k in range(reading.size):
        values = reading.singular[k] * scale
        shown = judged & (values > 0.0)
        style = SINGULAR_STYLES[min(k, len(SINGULAR_STYLES) - 1)]
        top.plot(f[shown], values[shown], pen=_pen(ink, 1.5, style),
                 name=singular_name(k))
    threshold = reading.singular[0] * scale / reading.limit
    top.plot(f[judged], threshold[judged], pen=_pen(ink, 1.0, 'dot'),
             name=f'σ1 / {reading.limit:g}, the rank threshold')
    top.setLabel('left', 'singular value, '
                 + axis_label(reading.dimension, us))

    bottom = _reading_plot(layout, 2)
    bottom.setLogMode(x=True, y=True)
    bottom.setXLink(top)
    condition = reading.condition
    finite = judged & np.isfinite(condition)
    bottom.plot(f[finite], condition[finite], pen=_pen(ink, 1.5, 'solid'))
    limit = bottom.plot(f[judged], np.full(int(judged.sum()), reading.limit),
                        pen=_pen(ink, 1.0, 'dash'))
    limit.setZValue(-1)
    hatch_bands(bottom, f, reading.ill & (f > 0.0), ink)
    bottom.setLabel('left', f'condition number (limit {reading.limit:g} '
                            'dashed)')
    bottom.setLabel('bottom', 'frequency [Hz]')
    grid = getattr(layout, 'ci', layout).layout
    grid.setRowStretchFactor(0, 3)
    grid.setRowStretchFactor(2, 2)
    _fit_band(top, band, True)
    return top, bottom


def build_realizability(layout: Any, reading: Any, *,
                        band: tuple[float, float] | None = None,
                        pair: tuple[int, int] | None = None,
                        theme: Any = None) -> tuple[Any, ...]:
    """Draw a realizability reading into a graphics layout and return its
    plots: (eigenvalues,), or (eigenvalues, coherence, phase) when the
    repair moved anything.

    Above, the eigenvalues of the coherence matrix on a linear axis
    against log frequency, the smallest solid, the largest dashed and
    any between dotted, with the zero line dashed thin and the lines
    that are not realizable hatched: a negative eigenvalue is a
    combination of the channels the specification gives negative power.
    Below it, for the pair the repair moved most, the coherence and the
    phase before (solid) and after (dashed); `pair`, (i, j) by channel
    index, draws that pair instead. `band` is the (low, high)
    frequencies shown, in Hz.
    """
    import pyqtgraph as pg

    from ..theme import theme as resolve_theme
    from . import _reading_plot, legend_below

    colors = resolve_theme(theme)
    ink = colors['plot_foreground']
    f = np.asarray(reading.frequency, dtype=float)
    judged = reading.judged & (f > 0.0)
    n = len(reading.channels)

    top = _reading_plot(layout, 0)
    top.setLogMode(x=True, y=False)
    legend_below(layout, top, 0, colors)
    top.setTitle(_title(reading), color=ink, size='9pt')
    zero = pg.InfiniteLine(pos=0.0, angle=0, movable=False,
                           pen=_pen(ink, 1.0, 'dash'))
    top.addItem(zero, ignoreBounds=True)
    for k in range(1, n - 1):
        top.plot(f[judged], reading.eigenvalues[k][judged],
                 pen=_pen(ink, 1.0, 'dot'),
                 name='the others' if k == 1 else None)
    top.plot(f[judged], reading.largest[judged], pen=_pen(ink, 1.5, 'dash'),
             name='largest eigenvalue')
    top.plot(f[judged], reading.smallest[judged],
             pen=_pen(ink, 2.0, 'solid'), name='smallest eigenvalue')
    hatch_bands(top, f, reading.failing & (f > 0.0), ink)
    top.setLabel('left', 'eigenvalue of the coherence matrix '
                         '(not realizable hatched)')
    plots: tuple[Any, ...] = (top,)
    worst = reading.worst()
    if worst is not None or pair is not None:
        i, j = pair if pair is not None else worst[:2]
        before, after, phase_before, phase_after = reading.pair(i, j)
        pair = f'{reading.channels[i]} and {reading.channels[j]}'
        middle = _reading_plot(layout, 2)
        middle.setLogMode(x=True, y=False)
        middle.setXLink(top)
        legend_below(layout, middle, 1, colors)
        middle.plot(f[judged], before[judged], pen=_pen(ink, 1.5, 'solid'),
                    name=f'{pair} as written', connect='finite')
        middle.plot(f[judged], after[judged], pen=_pen(ink, 1.5, 'dash'),
                    name=f'{pair} repaired', connect='finite')
        middle.setYRange(0.0, 1.05, padding=0.0)
        middle.setLabel('left', 'coherence')
        bottom = _reading_plot(layout, 4)
        bottom.setLogMode(x=True, y=False)
        bottom.setXLink(top)
        bottom.plot(f[judged], phase_before[judged],
                    pen=_pen(ink, 1.5, 'solid'), connect='finite')
        bottom.plot(f[judged], phase_after[judged],
                    pen=_pen(ink, 1.5, 'dash'), connect='finite')
        bottom.setYRange(-185.0, 185.0, padding=0.0)
        bottom.setLabel('left', 'phase [°]')
        bottom.setLabel('bottom', 'frequency [Hz]')
        grid = getattr(layout, 'ci', layout).layout
        grid.setRowStretchFactor(0, 3)
        grid.setRowStretchFactor(2, 2)
        grid.setRowStretchFactor(4, 2)
        plots = (top, middle, bottom)
    else:
        top.setLabel('bottom', 'frequency [Hz]')
    _fit_band(top, band, True)
    return plots
