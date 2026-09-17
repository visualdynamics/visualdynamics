"""The scalogram, drawn: time across, frequency up, magnitude as color.

One channel at a time, because the picture is dense — a second one
beside it at this size reads as noise rather than as a second answer.

Two things here are not decoration and are the reason this is a module
rather than four lines beside the coherence map.

**The frequency axis is logarithmic**, because a wavelet's bandwidth is
a constant fraction of its frequency: the rows *are* evenly spaced in
log frequency, so drawing them evenly spaced in linear frequency would
put the image's own rows where they do not belong. The image is drawn
in log space and the axis is labeled back in Hz.

**The cone of influence is drawn over the picture**, shaded, because
inside it a scalogram is an artifact of where the record was cut rather
than a reading of the record. It looks exactly like data — that is the
whole problem — and a reading that cannot be told from an artifact is
worse than no reading.
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
                    time_label: str = 'time [s]') -> Any:
    """Draw one channel's scalogram onto `plot`, and return the image.

    Parameters
    ----------
    plot : pyqtgraph.PlotItem
        Where to draw. Cleared of nothing — the caller owns the layout.
    magnitude : numpy.ndarray
        ``(frequencies, times)``, already the magnitude: this draws, it
        does not transform.
    times : numpy.ndarray
        The record's clock, in seconds, one per column.
    frequencies : numpy.ndarray
        One per row, ascending, spaced evenly by octave.
    colors : mapping
        The theme.
    label, units : str
        What the color bar is of — the record's own quantity and unit,
        which the amplitude normalization is what makes meaningful.
    omega0 : float
        The wavelet's width, for working out the cone.
    time_label : str
        The bottom axis's label, units included.

    Returns
    -------
    pyqtgraph.ImageItem
        The image, so a caller can adjust its levels.
    """
    import pyqtgraph as pg

    foreground = pg.mkPen(colors['plot_foreground'])
    for edge in ('left', 'bottom', 'top', 'right'):
        axis = plot.getAxis(edge)
        axis.setPen(foreground)
        axis.setTextPen(foreground)

    image = pg.ImageItem(np.asarray(magnitude).T)
    image.setColorMap(pg.colormap.get('viridis'))
    finite = np.isfinite(magnitude)
    top = float(np.max(magnitude[finite])) if finite.any() else 1.0
    image.setLevels((0.0, top if top > 0.0 else 1.0))

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
        values=(0.0, top if top > 0.0 else 1.0),
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
