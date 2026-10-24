"""The notch a set of limit channels would cut, drawn from one
`core.limit_prediction.LimitPrediction`.

At the top, the control channel: the reference dashed, the control the
controller could hold solid, the lower edge of the tolerance dotted, and
every predicted notch hatched across its band with its depth written on
it. Under it, a row per response channel: the predicted response at full
level solid against its limit profile dashed, the bands where it is over
hatched. Each row is titled with its reading in words.

Nothing is carried by hue alone. The curves differ by line style and are
named in the legend; the notches and the exceedances are hatched as well
as shaded, and labeled; and the verdicts are in the titles in words, in
the abort color only as well.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: the pen styles (Qt.PenStyle values): the prediction solid, the curve
#: it is read against dashed, the tolerance dotted
SOLID, DASHED, DOTTED = 1, 2, 3

#: how solid the shading of a hatched band is, of 255
HATCH_ALPHA = 90


def _density(values: Any, dimension: str, us: Any) -> np.ndarray:
    """Autospectral values in SI restated in `us`'s units."""
    from ..units import UNKNOWN

    values = np.asarray(values, dtype=float)
    if dimension == UNKNOWN:
        return values
    return np.asarray(us.from_si(values, dimension), dtype=float)


def _curve(plot: Any, f: np.ndarray, values: np.ndarray, pen: Any,
           name: str) -> None:
    """A curve on log axes, broken where it holds no positive number."""
    good = np.isfinite(values) & (values > 0.0) & (f > 0.0)
    if not good.any():
        return
    y = np.where(good, values, np.nan)
    plot.plot(f, y, pen=pen, name=name, connect='finite')


def _hatch(plot: Any, f: np.ndarray, first: float, last: float,
           color: Any) -> Any:
    """A hatched band across the bins of the lines from `first` to
    `last` (Hz), full height; returns the region."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QBrush

    from ..core.octave import bin_bounds

    left, right = bin_bounds(f, None)
    i, j = int(np.searchsorted(f, first)), int(np.searchsorted(f, last))
    low = max(float(left[i]), float(f[i]) * 0.999, 1e-9)
    high = float(right[j])
    shade = pg.mkColor(color)
    shade.setAlpha(HATCH_ALPHA)
    region = pg.LinearRegionItem(
        values=(np.log10(low), np.log10(high)), movable=False,
        brush=QBrush(shade, Qt.BrushStyle.BDiagPattern),
        pen=pg.mkPen(None))
    region.setZValue(-10)
    plot.addItem(region, ignoreBounds=True)
    return region


def _words(plot: Any, words: str, frequency: float, level: float,
           color: Any, above: bool) -> Any:
    """`words` on a log-log plot at (`frequency`, `level`), centered,
    standing above the point or hanging below it."""
    import pyqtgraph as pg

    if not (np.isfinite(level) and level > 0.0 and frequency > 0.0):
        return None
    label = pg.TextItem(words, color=color,
                        anchor=(0.5, 1.0 if above else 0.0))
    label.setPos(np.log10(frequency), np.log10(level))
    plot.addItem(label, ignoreBounds=True)
    return label


def notch_words(notch: Any) -> str:
    """What a notch's band says on the plot: its depth, and its channel."""
    return f'notch {notch.depth_db:.1f} dB, {notch.channels[0]}'


def build_limit_prediction(layout: Any, prediction: Any, *,
                           unit_system: Any = None,
                           theme: Any = None) -> list[Any]:
    """Draw `prediction` into a graphics layout: the control row, then a
    row per response channel. Returns the plots, the control's first."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import _reading_plot, axis_label, legend_below

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    ink = colors['plot_foreground']
    abort = colors['limit_abort']
    f = np.asarray(prediction.frequency, dtype=float)

    def pen(style: int, color: Any = ink, width: float = 1.5) -> Any:
        return pg.mkPen(color, width=width, style=Qt.PenStyle(style))

    plots = []
    control = _reading_plot(layout, 0)
    control.setLogMode(x=True, y=True)
    legend_below(layout, control, 0, colors)
    dimension = prediction.dimension
    _curve(control, f, _density(prediction.reference, dimension, us),
           pen(DASHED), f'reference at {prediction.control_label}')
    _curve(control, f, _density(prediction.control, dimension, us),
           pen(SOLID), 'control, predicted')
    if prediction.lower is not None:
        _curve(control, f, _density(prediction.lower, dimension, us),
               pen(DOTTED, abort, 1.0),
               f'{prediction.tolerance} tolerance, lower edge')
    shown = _density(prediction.control, dimension, us)
    for notch in prediction.notches:
        _hatch(control, f, notch.low, notch.high, abort)
        k = int(np.argmin(np.abs(f - notch.frequency)))
        _words(control, notch_words(notch), notch.frequency, shown[k], ink,
               above=False)
    control.setLabel('left', axis_label(dimension, us))
    outside = prediction.within_tolerance is False
    control.setTitle(prediction.verdict(), color=abort if outside else ink,
                     size='9pt')
    plots.append(control)

    for k, channel in enumerate(prediction.channels, start=1):
        row = _reading_plot(layout, 2 * k)
        row.setLogMode(x=True, y=True)
        row.setXLink(control)
        legend_below(layout, row, k, colors)
        _curve(row, f, _density(channel.response, channel.dimension, us),
               pen(SOLID), f'{channel.label} predicted at full level')
        if channel.profile is not None:
            _curve(row, f, _density(channel.profile, channel.dimension, us),
                   pen(DASHED), 'limit profile')
            response = _density(channel.response, channel.dimension, us)
            for over in channel.exceedances:
                _hatch(row, f, over.low, over.high, abort)
                k = int(np.argmin(np.abs(f - over.frequency)))
                _words(row, f'over {over.over_db:.1f} dB', over.frequency,
                       response[k], ink, above=True)
        row.setLabel('left', axis_label(channel.dimension, us))
        row.setTitle(channel.describe(f, us),
                     color=abort if channel.exceedances else ink, size='9pt')
        plots.append(row)
    plots[-1].setLabel('bottom', 'frequency [Hz]')
    # the lines predicted on, not the whole transmissibility: outside the
    # reference nothing is predicted
    span = f[np.isfinite(prediction.reference) & (f > 0.0)]
    if span.size > 1:
        control.setXRange(np.log10(span[0]), np.log10(span[-1]),
                          padding=0.02)
    return plots


def summary(prediction: Any) -> str:
    """The prediction's verdict in a line, for a status bar."""
    return prediction.verdict()
