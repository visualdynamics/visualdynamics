"""The partial, reference and virtual coherence of one response, drawn.

A `PartialCoherence` (`core.partial_coherence`) holds every response
against every reference. `build_partial_coherence` draws one response
as three plots on a shared frequency axis, each told apart by line
style, marker shape and words, never by hue alone:

- above, the multiple coherence heavy and solid, and under it the
  partial coherence with each reference, each in its own dash and with
  its own marker every few lines, named 'partial with 1Z+';
- in the middle, the coherence between every pair of references, the
  same styles, with the level over which two drives are hard to tell
  apart drawn dotted (`core.partial_coherence.COUPLED`);
- beneath, the principal references' autospectra as shares of the
  first, in decibels: how many independent drives the test had at each
  line, with the response's virtual coherences named in the legend.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: the dash of each reference's curve, in turn: told apart by the line,
#: not by its color
STYLES = ('dash', 'dot', 'dashdot', 'dashdotdot')

#: and the marker each wears every `MARK_EVERY` lines
SYMBOLS = ('o', 's', 't', 'd', '+', 'x')

#: how often a curve's marker is drawn, in lines, so a marker names its
#: curve without hiding it
MARK_EVERY = 8

#: the lowest share of the first principal reference drawn, in decibels
FLOOR_DB = -60.0


def _pen(color: Any, index: int, width: float = 1.5) -> Any:
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    style = {'dash': Qt.PenStyle.DashLine, 'dot': Qt.PenStyle.DotLine,
             'dashdot': Qt.PenStyle.DashDotLine,
             'dashdotdot': Qt.PenStyle.DashDotDotLine}[
        STYLES[index % len(STYLES)]]
    return pg.mkPen(color, width=width, style=style)


def _curve(plot: Any, x: np.ndarray, y: np.ndarray, color: Any, index: int,
           name: str) -> None:
    """One curve in its dash, its marker every few lines on a second
    item so the marker does not take every point, both named once."""
    import pyqtgraph as pg

    shown = np.isfinite(y)
    plot.plot(x, y, pen=_pen(color, index), connect='finite', name=name)
    marks = np.zeros(len(x), dtype=bool)
    marks[index % MARK_EVERY::MARK_EVERY] = True
    marks &= shown
    plot.plot(x[marks], y[marks], pen=None,
              symbol=SYMBOLS[index % len(SYMBOLS)], symbolSize=7,
              symbolPen=pg.mkPen(color, width=1.2), symbolBrush=None)


def build_partial_coherence(layout: Any, reading: Any,
                            response: int | str = 0, *,
                            theme: Any = None) -> tuple[Any, Any, Any]:
    """Draw one response of a partial coherence reading into a graphics
    layout and return its three plots, (coherence, references,
    principal).

    `response` is the response's position or name. The reading's notes
    and its one-line description go under the title.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..theme import theme as resolve_theme
    from . import _reading_plot, legend_below

    colors = resolve_theme(theme)
    ink = colors['plot_foreground']
    k = reading.index(response)
    x = np.asarray(reading.frequency, dtype=np.float64)

    top = _reading_plot(layout, 0)
    legend_below(layout, top, 0, colors)
    title = reading.describe(k)
    if reading.notes:
        title += '<br>' + '<br>'.join(reading.notes)
    top.setTitle(title, color=ink, size='9pt')
    top.plot(x, reading.multiple[k], pen=pg.mkPen(ink, width=2.5),
             connect='finite', name='multiple, every reference')
    for i, name in enumerate(reading.references):
        _curve(top, x, reading.partial[k, i], ink, i,
               f'partial with {name}, the others removed')
    top.setYRange(0.0, 1.05, padding=0.0)
    top.setLabel('left', f'{reading.responses[k]} coherence')

    middle = _reading_plot(layout, 2)
    legend_below(layout, middle, 1, colors)
    middle.setXLink(top)
    pair = 0
    for i in range(len(reading.references)):
        for j in range(i + 1, len(reading.references)):
            _curve(middle, x, reading.reference[i, j], ink, pair,
                   f'{reading.references[i]} with {reading.references[j]}')
            pair += 1
    from ..core.partial_coherence import COUPLED

    middle.addLine(y=COUPLED, pen=pg.mkPen(ink, width=1,
                                           style=Qt.PenStyle.DotLine))
    middle.setYRange(0.0, 1.05, padding=0.0)
    middle.setLabel('left', 'between references')

    bottom = _reading_plot(layout, 4)
    legend_below(layout, bottom, 2, colors)
    bottom.setXLink(top)
    first = reading.principal[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        shares = np.where(first > 0.0, reading.principal / first, np.nan)
        decibels = np.maximum(10.0 * np.log10(shares), FLOOR_DB)
    for i in range(len(reading.references)):
        virtual = reading.virtual[k, i]
        defined = virtual[np.isfinite(virtual)]
        median = (f', virtual coherence median {np.median(defined):.2f}'
                  if defined.size else '')
        if i == 0:
            bottom.plot(x, decibels[i], pen=pg.mkPen(ink, width=1.5),
                        connect='finite', name=f'principal 1{median}')
        else:
            _curve(bottom, x, decibels[i], ink, i - 1,
                   f'principal {i + 1}{median}')
    bottom.setYRange(FLOOR_DB, 3.0, padding=0.0)
    bottom.setLabel('left', 'share of principal 1 [dB]')
    bottom.setLabel('bottom', 'frequency [Hz]')
    grid = getattr(layout, 'ci', layout).layout
    grid.setRowStretchFactor(0, 3)
    grid.setRowStretchFactor(2, 2)
    grid.setRowStretchFactor(4, 2)
    return top, middle, bottom
