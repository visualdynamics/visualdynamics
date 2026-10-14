"""Levels of one sine tone read several ways, overlaid on one plot.

Each reading is the level against frequency through one tracking band
and detector, drawn over the specification. The plot shows how much
the level depends on how it was measured, so every reading is drawn on
the same axes, and the specification is drawn wide and gray behind them
as the reference, the same as a specification behind its response in
the other plots.

Each reading has a marker shape as well as a color. This is the only
plot where several curves of one quantity on one channel are meant to
be compared line by line, and a reader who cannot tell two of the
palette's colors apart could not read it by color alone. The markers
are placed at a dozen points along each curve (a marker at every point
would hide the curve) and are shown beside each name in the legend.
Shapes are used instead of dashes because dashes already mean a
synthesis drawn over a measurement.

The level axis is linear. Sine levels are drawn in decades elsewhere,
which suits a sweep through a resonance. Here the readings are within
tens of percent of each other, and on a log axis the 30 % a harmonic
adds to a peak reading would be a very small difference.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

#: the marker each reading wears, in order — shapes distinct in
#: outline, so they survive being printed small and in gray
SYMBOLS = ('o', 't', 's', 'd', 't1', 'x', '+', 'star')

#: how many markers along each curve
MARKERS = 12

#: the markers' size, pixels
MARKER_SIZE = 8


def _marker_spots(x: np.ndarray, k: int, curves: int,
                  log_x: bool) -> np.ndarray:
    """Where curve `k` of `curves` wears its markers: `MARKERS` lines
    evenly spaced along the axis as drawn — in decades on a log axis —
    rather than along the readings, which a log sweep crowds at its low
    end; and staggered curve to curve, so two readings that agree do
    not stack their shapes on the same points."""
    position = np.log10(np.maximum(x, 1e-300)) if log_x else x
    low, high = float(np.min(position)), float(np.max(position))
    count = min(MARKERS, len(x))
    wanted = low + (high - low) * (np.arange(count)
                                   + (k + 1) / (curves + 1)) / count
    order = np.argsort(position, kind='stable')
    found = np.searchsorted(position[order], wanted).clip(0, len(x) - 1)
    return np.unique(order[found])


def build_sine_tracking(layout: Any, levels: Sequence[Any],
                        settings: Sequence[Any], *, channel: int = 0,
                        specification: Any = None,
                        unit_system: Any = None, theme: Any = None) -> int:
    """Draw one channel's readings of a tone and return how many were drawn.

    `settings` is the list of tracking settings given to
    `core.sine_tracking.track_sine` and `levels` is the list it
    returned, in the same order; each reading is named in the legend by
    its setting. `channel` is the row of the levels that is drawn. With
    `specification`, the tone's target at that channel is drawn behind
    the readings in the specification's gray.

    Raises `ValueError` when `levels` is empty or when `levels` and
    `settings` differ in length.
    """
    import pyqtgraph as pg

    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import STOOD_BACK_WIDTH, axis_label, curve_color, legend_below

    if len(levels) != len(settings):
        raise ValueError(f'{len(levels)} levels for {len(settings)} '
                         'settings; they are read and named in pairs')
    if not levels:
        raise ValueError('no readings to draw')
    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    first = levels[0]
    dof = first.response_dof[channel]
    dimension = first.ordinate_dim[channel]

    plot = layout.addPlot(row=0, col=0)
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setLogMode(x=bool(first.log_abscissa), y=False)
    legend = legend_below(layout, plot, 0, colors)
    plot.setTitle(f'{first.tone} at {dof}', color=colors['plot_foreground'],
                  size='9pt')

    if specification is not None and dof in specification.response_dof:
        tone = specification.tone(first.tone)
        frequencies = np.sort(np.concatenate([level.abscissa
                                              for level in levels]))
        target = tone.target(frequencies)[:, specification.response_dof
                                          .index(dof)]
        plot.plot(frequencies, us.from_si(target, dimension),
                  connect='finite', name='specification',
                  pen=pg.mkPen(colors['specification_curve'],
                               width=STOOD_BACK_WIDTH))

    log_x = bool(first.log_abscissa)
    hollow = pg.mkBrush(0, 0, 0, 0)
    for k, (level, setting) in enumerate(zip(levels, settings)):
        color = curve_color(k, colors)
        x = np.asarray(level.abscissa, dtype=float)
        y = us.from_si(np.abs(level.ordinate[channel]), dimension)
        pen = pg.mkPen(color, width=1.5)
        symbol = SYMBOLS[k % len(SYMBOLS)]
        plot.plot(x, y, pen=pen)
        spots = _marker_spots(x, k, len(levels), log_x)
        plot.plot(x[spots], y[spots], pen=None,
                  symbol=symbol, symbolSize=MARKER_SIZE,
                  symbolPen=pg.mkPen(color), symbolBrush=hollow)
        # the legend's sample: the line and the shape together, drawn
        # nowhere but in the legend
        legend.addItem(pg.PlotDataItem(
            [], [], pen=pen, symbol=symbol, symbolSize=MARKER_SIZE,
            symbolPen=pg.mkPen(color), symbolBrush=hollow),
            setting.describe())

    plot.setLabel('bottom', 'frequency [Hz]')
    plot.setLabel('left', axis_label(dimension, us))
    return len(levels)
