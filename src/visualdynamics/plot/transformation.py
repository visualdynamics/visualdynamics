"""What a transformation explains of the measured motion, and what it
leaves, against frequency.

A `TransformationResidual` (`core.transformation`) splits the measured
power at a transformation's inputs, line by line, into the part its
outputs carry back and the residual. `build_transformation_residual`
draws it as two plots on a shared frequency axis, each line told apart
by its style and its name, never by hue alone:

- above, on log axes, the measured power summed over the inputs solid,
  the part the outputs explain dotted, and the residual dashed: where
  the dashed line rises to the solid one, the outputs no longer
  describe the motion;
- beneath, the residual's share of the measured power, the threshold
  dotted across it, and a dashed vertical line, named in the legend, at
  the frequency up to which the share stays under the threshold.

The transformation's conditioning and the reading's one-line
description go in the title.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MEASURED_NAME = 'measured, summed over the inputs'
EXPLAINED_NAME = 'explained by the outputs'
RESIDUAL_NAME = 'residual, what the outputs leave'


def display_scale(reading: Any, unit_system: Any = None) -> tuple[float, str]:
    """(scale, unit): the factor that converts the reading's SI spectra
    into `unit_system`'s units, and that unit's name."""
    from ..units import DEFAULT_SYSTEM

    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    return (float(us.from_si(1.0, reading.dimension)),
            us.label_text(reading.dimension))


def title_of(reading: Any) -> str:
    """The figure's title: the transformation, then the reading."""
    return f'{reading.transformation.describe()}<br>{reading.describe()}'


def build_transformation_residual(layout: Any, reading: Any, *,
                                  unit_system: Any = None,
                                  theme: Any = None) -> tuple[Any, Any]:
    """Draw a transformation's residual reading into a graphics layout
    and return its two plots, (spectra, share)."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import _reading_plot, axis_label, legend_below

    colors = resolve_theme(theme)
    ink = colors['plot_foreground']
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    scale, _unit = display_scale(reading, us)
    x = np.asarray(reading.frequency, dtype=np.float64)
    measured = reading.total * scale
    explained = np.sum(reading.explained, axis=0) * scale
    residual = np.sum(reading.residual, axis=0) * scale

    top = _reading_plot(layout, 0)
    top.setLogMode(x=True, y=True)
    legend_below(layout, top, 0, colors)
    title = title_of(reading)
    if reading.notes:
        title += '<br>' + '<br>'.join(reading.notes)
    top.setTitle(title, color=ink, size='9pt')
    for values, pen, name in (
            (measured, pg.mkPen(ink, width=2.0), MEASURED_NAME),
            (explained, pg.mkPen(ink, width=1.5, style=Qt.PenStyle.DotLine),
             EXPLAINED_NAME),
            (residual, pg.mkPen(ink, width=1.5, style=Qt.PenStyle.DashLine),
             RESIDUAL_NAME)):
        shown = np.isfinite(values) & (values > 0.0) & (x > 0.0)
        top.plot(x[shown], values[shown], pen=pen, name=name)
    top.setLabel('left', axis_label(reading.dimension, us))

    bottom = _reading_plot(layout, 2)
    legend_below(layout, bottom, 1, colors)
    bottom.setLogMode(x=True, y=False)
    bottom.setXLink(top)
    fraction = reading.fraction
    keep = x > 0.0
    bottom.plot(x[keep], fraction[keep], pen=pg.mkPen(ink, width=1.5),
                connect='finite', name='residual share of the measured power')
    dotted = pg.mkPen(ink, width=1, style=Qt.PenStyle.DotLine)
    bottom.plot(x[keep], np.full(int(keep.sum()), reading.threshold),
                pen=dotted, name=f'threshold, {reading.threshold:.0%}')
    holds = reading.holds_to
    if holds is not None and holds > 0.0 and not reading.holds_throughout:
        dashed = pg.mkPen(ink, width=1.5, style=Qt.PenStyle.DashLine)
        bottom.plot([holds, holds], [0.0, 1.0], pen=dashed,
                    name=f'holds to {holds:.4g} Hz')
    bottom.setYRange(0.0, 1.05, padding=0.0)
    bottom.setLabel('left', 'residual share')
    bottom.setLabel('bottom', 'frequency [Hz]')
    grid = getattr(layout, 'ci', layout).layout
    grid.setRowStretchFactor(0, 3)
    grid.setRowStretchFactor(2, 2)
    return top, bottom
