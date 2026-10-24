"""The drives a specification asks for, and what they ask of the rig:
three rows from one `core.drive_prediction.DrivePrediction`.

At the top, each drive's predicted autospectrum on log axes, titled
with the drive nearest its limit. In the middle, one channel's
predicted autospectrum: a control channel beside its target, with the
lines the inverse did not meet shaded, or any other channel the FRF
holds, titled with its own limit when it has one. At the bottom, a bar
for every quantity judged, its expected peak as a share of its rating,
with the rating at 100 % and the quantity that runs out first named in
the title.

Nothing is carried by hue alone. The drives are told apart by line
style as well as color and named in the legend; the prediction is solid
and the target dashed; the unmet lines are hatched as well as shaded;
and every bar is labeled with what it is and where, its verdict in the
title in words.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: the pen styles the drives cycle through (Qt.PenStyle values), so two
#: drives differ in more than their color
STYLES = (1, 2, 3, 4)                  # solid, dash, dot, dash-dot
SOLID, DASHED = 1, 2

#: how solid the shading of the unmet lines is
UNMET_ALPHA = 45


def _density(prediction: Any, values: Any, quantity: str, us: Any) -> Any:
    """Autospectrum values in SI restated in `us`'s units, and the
    dimension they are in."""
    dimension = f'{quantity}**2/frequency'
    return np.asarray(us.from_si(np.asarray(values, dtype=float), dimension),
                      dtype=float), dimension


def _positive(x: Any, y: Any) -> tuple[Any, Any]:
    """The points a log axis can draw: both coordinates finite and above
    zero."""
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    return x[keep], y[keep]


def channel_index(prediction: Any, channel: int | str | None) -> int:
    """The FRF row a channel names, by index or label; the first control
    channel when none is named.

    Raises
    ------
    ValueError
        If `channel` is not a row of the prediction.
    """
    if channel is None:
        return int(prediction.controls[0])
    if isinstance(channel, str):
        if channel not in prediction.rows:
            raise ValueError(f'{channel} is not a channel of the FRF '
                             f'({", ".join(prediction.rows)})')
        return prediction.rows.index(channel)
    index = int(channel)
    if not 0 <= index < len(prediction.rows):
        raise ValueError(f'channel {index} is not one of the FRF\'s '
                         f'{len(prediction.rows)}')
    return index


def channel_words(prediction: Any, row: int,
                  unit_system: Any = None) -> str:
    """The middle row's title: a control channel against its target, or
    another channel against its limit."""
    from ..core.drive_prediction import UNMET_DB

    label = prediction.rows[row]
    if row in prediction.controls:
        off = prediction.missed(prediction.controls.index(row))
        if not off.any():
            return (f'{label}: the predicted response meets the target '
                    f'within {UNMET_DB:g} dB')
        lines = int(off.sum())
        return (f'{label}: the predicted response misses the target by '
                f'more than {UNMET_DB:g} dB at {lines} '
                f'line{"s" * (lines != 1)}, hatched')
    limited = [d for d in prediction.demands
               if d.kind == 'response' and d.channel == label]
    if limited:
        return limited[0].describe(unit_system)
    return f'{label}: predicted response'


def bar_rows(prediction: Any) -> tuple[list[tuple[str, float]],
                                       dict[int, str]]:
    """The bars: (name, percent of its rating) for every demand, and a
    note for each one with no rating, which has no bar."""
    rows, notes = [], {}
    for k, demand in enumerate(prediction.demands):
        percent = (float('nan') if demand.share is None
                   else 100.0 * demand.share)
        rows.append((demand.name, percent))
        if demand.share is None:
            notes[k] = 'no rating given'
    return rows, notes


def build_drive_prediction(layout: Any, prediction: Any, *,
                           channel: int | str | None = None,
                           unit_system: Any = None,
                           theme: Any = None) -> tuple[Any, Any, Any]:
    """Draw `prediction` into a graphics layout as three rows and return
    their plots, (drives, channel, shares).

    `channel` picks the middle row's channel, by index or label, the
    first control channel by default.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import _reading_plot, axis_label, curve_color, legend_below
    from .bars import BarChart

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    ink = colors['plot_foreground']
    f = np.asarray(prediction.frequency, dtype=float)
    row = channel_index(prediction, channel)

    # the drives
    drives = _reading_plot(layout, 0)
    drives.setLogMode(x=True, y=True)
    legend_below(layout, drives, 0, colors)
    dimension = None
    for j, label in enumerate(prediction.drives):
        values, dimension = _density(prediction, prediction.drive_asd(j),
                                     prediction.drive_quantity, us)
        x, y = _positive(f, values)
        drives.plot(x, y, pen=pg.mkPen(
            curve_color(j, colors), width=1.5,
            style=Qt.PenStyle(STYLES[j % len(STYLES)])),
            name=f'drive at {label}')
    drives.setLabel('left', axis_label(dimension, us))
    worst = [d for d in prediction.demands if d.kind == 'drive']
    rated = [d for d in worst if d.share is not None]
    lead = (max(rated, key=lambda d: d.share) if rated
            else max(worst, key=lambda d: d.peak))
    drives.setTitle(lead.describe(us), color=_tone(lead.share, colors),
                    size='9pt')

    # one channel, against its target or its limit
    middle = _reading_plot(layout, 2)
    middle.setLogMode(x=True, y=True)
    middle.setXLink(drives)
    legend_below(layout, middle, 1, colors)
    quantity = prediction.row_quantities[row]
    values, dimension = _density(prediction, prediction.asd[:, row],
                                 quantity, us)
    x, y = _positive(f, values)
    middle.plot(x, y, pen=pg.mkPen(ink, width=1.5),
                name=f'{prediction.rows[row]} predicted')
    if row in prediction.controls:
        k = prediction.controls.index(row)
        asked, _dim = _density(prediction,
                               np.real(prediction.target[:, k, k]),
                               quantity, us)
        tx, ty = _positive(f, asked)
        middle.plot(tx, ty, pen=pg.mkPen(ink, width=1.5,
                                         style=Qt.PenStyle(DASHED)),
                    name='target')
        _hatch_unmet(middle, prediction, k, f, colors)
    middle.setLabel('left', axis_label(dimension, us))
    middle.setLabel('bottom', 'frequency [Hz]')
    limited = [d for d in prediction.demands
               if d.kind == 'response' and d.channel == prediction.rows[row]]
    middle.setTitle(channel_words(prediction, row, us),
                    color=_tone(limited[0].share if limited else None,
                                colors), size='9pt')

    # every quantity against its rating
    shares = _reading_plot(layout, 4)
    rows, notes = bar_rows(prediction)
    BarChart(shares, rows, colors, low=100.0, high=None,
             label='expected peak, share of its rating', units='%',
             summary='none', notes=notes)
    shares.setTitle(summary(prediction, us),
                    color=_tone(None if prediction.first_out is None
                                else prediction.first_out.share, colors),
                    size='9pt')
    grid = getattr(layout, 'ci', layout).layout
    grid.setRowStretchFactor(0, 3)
    grid.setRowStretchFactor(2, 3)
    grid.setRowStretchFactor(4, 2)
    return drives, middle, shares


def _hatch_unmet(plot: Any, prediction: Any, k: int, f: Any,
                 colors: Any) -> None:
    """Shade and hatch the lines where control channel `k` misses its
    target, full height, so the lines read without their color."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QBrush

    off = prediction.missed(k)
    if not off.any():
        return
    from ..core.drive_prediction import runs

    half = np.gradient(f) / 2.0
    color = pg.mkColor(colors['limit_abort'])
    color.setAlpha(UNMET_ALPHA)
    brush = QBrush(color, Qt.BrushStyle.BDiagPattern)
    for first, last in runs(off):
        # a line stands for its bin, so the shading covers the bins
        low = max(f[first] - half[first], f[first] * 0.999)
        high = f[last] + half[last]
        region = pg.LinearRegionItem(
            values=(np.log10(low), np.log10(high)), movable=False,
            brush=brush, pen=pg.mkPen(None))
        region.setZValue(-10)
        plot.addItem(region, ignoreBounds=True)


def _tone(share: float | None, colors: Any) -> str:
    """The title's color: the abort color over a rating, the foreground
    otherwise. The words carry the verdict as well."""
    if share is not None and share > 1.0:
        return colors['limit_abort']
    return colors['plot_foreground']


def summary(prediction: Any, unit_system: Any = None) -> str:
    """The prediction's verdict in a line, for a title or a status bar."""
    del unit_system                      # the verdict carries no amounts
    return prediction.describe()[0]
