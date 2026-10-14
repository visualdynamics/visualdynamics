"""Drawing a scalogram: time across, frequency up, magnitude as color.

One channel is drawn at a time, because the image is dense and a second
one beside it at the same size is too hard to read.

The frequency axis is logarithmic. A wavelet's bandwidth is a constant
fraction of its frequency, so the rows are evenly spaced in log
frequency, and drawing them evenly spaced in linear frequency would put
them in the wrong place. The image is drawn in log space and the axis
is labeled in Hz.

The cone of influence is shaded over the image. Near each end of the
record, over a time that is longer at lower frequencies, the values
depend on where the record was cut and not only on the record. They
look the same as the rest of the image, so the shading is the only way
to tell them apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..core import wavelet


def scalogram_image(plot: Any, magnitude: np.ndarray, times: np.ndarray,
                    frequencies: np.ndarray, colors: Mapping[str, str],
                    *, label: str = '', units: str = '',
                    omega0: float = wavelet.OMEGA0,
                    time_label: str = 'time [s]',
                    levels: tuple[float, float] | None = None) -> Any:
    """Draw one channel's scalogram onto `plot`, and return the image.

    Parameters
    ----------
    plot : pyqtgraph.PlotItem
        Where to draw. Nothing on it is cleared; the caller owns the
        layout.
    magnitude : numpy.ndarray
        ``(frequencies, times)``, already the magnitude. No transform
        is computed here.
    times : numpy.ndarray
        The record's clock, in seconds, one per column.
    frequencies : numpy.ndarray
        One per row, ascending, evenly spaced in log frequency.
    colors : mapping
        The theme.
    label, units : str
        The color bar's quantity and unit: the record's own, which the
        transform's amplitude normalization keeps.
    omega0 : float
        The wavelet's width parameter, used to compute the cone of
        influence.
    time_label : str
        The bottom axis's label, units included.
    levels : (float, float), optional
        The color scale's ends. The default is zero to the largest
        finite magnitude. A caller drawing the magnitude in decibels,
        which is never above its reference, passes the range it wants.

    Returns
    -------
    pyqtgraph.ImageItem
        The image, so the caller can change its levels.
    """
    import pyqtgraph as pg

    foreground = pg.mkPen(colors['plot_foreground'])
    for edge in ('left', 'bottom', 'top', 'right'):
        axis = plot.getAxis(edge)
        axis.setPen(foreground)
        axis.setTextPen(foreground)

    image = pg.ImageItem(np.asarray(magnitude).T)
    image.setColorMap(pg.colormap.get('viridis'))
    if levels is None:
        finite = np.isfinite(magnitude)
        top = float(np.max(magnitude[finite])) if finite.any() else 1.0
        levels = (0.0, top if top > 0.0 else 1.0)
    levels = (float(levels[0]), float(levels[1]))
    image.setLevels(levels)

    left, right = float(times[0]), float(times[-1])
    # drawn in log frequency, which is where the rows are actually
    # evenly spaced; the axis is labeled back in Hz below
    low, high = np.log10(frequencies[0]), np.log10(frequencies[-1])
    image.setRect(left, float(low), right - left, float(high - low))
    plot.addItem(image)

    plot.setXRange(left, right, padding=0)
    plot.setYRange(float(low), float(high), padding=0)
    plot.getViewBox().setLimits(xMin=left, xMax=right,
                                yMin=float(low), yMax=float(high))
    plot.setLabel('bottom', time_label)
    plot.setLabel('left', 'frequency [Hz]')
    plot.getAxis('left').setTicks([_decade_ticks(frequencies)])
    plot.showGrid(x=True, y=True, alpha=0.2)

    _draw_cone(plot, times, frequencies, colors, omega0)

    bar = pg.ColorBarItem(
        values=levels,
        colorMap=pg.colormap.get('viridis'),
        label=f'{label} [{units}]' if units else label, interactive=False)
    bar.setImageItem(image, insert_in=plot)
    return image


def _decade_ticks(frequencies: np.ndarray) -> list[tuple[float, str]]:
    """(position, text) up a log axis, at values a person would choose.

    1, 2, 5 per decade rather than the row positions: the rows are at
    twelfths of an octave and nobody reads 158.7 Hz off an axis.
    """
    ticks = []
    for value in wavelet.decade_values(frequencies[0], frequencies[-1]):
        text = f'{value:g}' if value >= 1.0 else f'{value:.3g}'
        ticks.append((float(np.log10(value)), text))
    return ticks


def _draw_cone(plot: Any, times: np.ndarray, frequencies: np.ndarray,
               colors: Mapping[str, str], omega0: float) -> None:
    """Shade where the record's ends reach into the picture.

    Drawn as two filled curves rather than as a line, because the
    question a reader has is "is this bit real", and a boundary answers
    it less directly than a veil over the part that is not.

    The cone is wide at the bottom of the axis and narrow at the top —
    low frequencies use long wavelets — so on a short record it can
    swallow the bottom octave entirely, which is exactly the case worth
    seeing.
    """
    import pyqtgraph as pg

    start, stop = float(times[0]), float(times[-1])
    reach = wavelet.cone_of_influence(frequencies, 1.0, omega0)
    rows = np.log10(frequencies)

    shade = pg.mkBrush(colors.get('plot_background', '#000000'))
    color = shade.color()
    color.setAlpha(150)
    shade = pg.mkBrush(color)
    edge = pg.mkPen(colors['plot_foreground'], width=1,
                    style=__import__('PySide6').QtCore.Qt.PenStyle.DotLine)

    for side in ('left', 'right'):
        # the cone's edge, as a time for each row
        if side == 'left':
            xs = np.minimum(start + reach, stop)
            wall = np.full_like(xs, start)
        else:
            xs = np.maximum(stop - reach, start)
            wall = np.full_like(xs, stop)
        curve = pg.PlotDataItem(xs, rows, pen=edge)
        against = pg.PlotDataItem(wall, rows)
        fill = pg.FillBetweenItem(curve, against, brush=shade)
        plot.addItem(curve)
        plot.addItem(fill)
