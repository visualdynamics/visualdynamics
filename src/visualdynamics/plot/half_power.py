"""A half-power reading drawn on the peak it was read from.

The reading (`core.half_power.read_half_power`) is three numbers a
reader would otherwise take off the plot with a ruler: the half-power
level, the two frequencies where the curve crosses it, and Q. Drawing
them on the curve is how a reader checks the numbers, because a
crossing placed on a shoulder or a level taken from the wrong peak is
obvious on the plot and invisible in a table.

The overlay is a dotted line at 0.707 of the peak, reaching half a width
past each crossing so it reads as a level and not a segment of the
curve; a hollow circle on each crossing; a filled diamond on the top as
read between lines; and the Q beside the lower crossing. Dotted, because
a dashed curve already means a synthesis drawn over a measurement. Shapes
and the dots, not a color, tell it from the curves under it: the overlay
takes one color, the plot's foreground unless the caller passes another
(the fitting screen passes its cursor's), so nothing in it is told apart
by hue alone.

The items go on the plot's view box, not the plot: they are a reading of
the data, not data, so the plot's data items (which its legend, its
copy and its filtering preview read) never include them. A plot converts
only its own data items to a log axis, so the overlay converts itself,
from the axes as they are when it is shown.

`HalfPowerOverlay` owns its items and moves them, so a display that
reads on every mouse move updates the same items instead of rebuilding
them; `add_half_power` draws one reading once, for a figure.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.half_power import _figures

#: the markers' size, pixels
MARKER_SIZE = 9

#: how far the level line reaches past each crossing, in half-power
#: widths: far enough to read as a level across the peak, short enough
#: not to run into the next one
REACH = 0.5


def label_text(reading: Any) -> str:
    """What the overlay writes beside the peak: the Q, and when the
    reading rests on too few lines or has no width, why it is not to be
    trusted, in as few words as the plot can carry."""
    if reading.q is None:
        return 'no half-power width'
    text = f'Q {_figures(reading.q, 3)}'
    if not reading.resolved:
        text += f'\n{reading.lines:.2g} lines: too few'
    return text


class HalfPowerOverlay:
    """The items that draw one half-power reading on a pyqtgraph plot.

    Made once per plot with `plot`, the PlotItem to draw on, and
    `color`, a pyqtgraph color (the plot's foreground by default); then
    `show(reading, scale)` places them and `clear()` hides them. `scale`
    multiplies the reading's height and level before drawing, for a
    curve drawn in other units than it was read in. The items ignore
    the plot's bounds, so drawing them never re-zooms the view.
    """

    def __init__(self, plot: Any, color: Any = None, *,
                 z: float = 42.0) -> None:
        import pyqtgraph as pg
        from PySide6.QtCore import Qt

        if color is None:
            color = plot.getAxis('left').pen().color()
        self.plot: Any = plot
        self.color: Any = pg.mkColor(color)
        pen = pg.mkPen(self.color, width=1.5, style=Qt.PenStyle.DotLine)
        #: the dotted 0.707 line, the crossings' circles, the top's
        #: diamond and the Q beside them
        self.level: Any = pg.PlotDataItem(pen=pen)
        self.crossings: Any = pg.PlotDataItem(
            pen=None, symbol='o', symbolSize=MARKER_SIZE,
            symbolPen=pg.mkPen(self.color, width=1.5),
            symbolBrush=pg.mkBrush(0, 0, 0, 0))
        self.top: Any = pg.PlotDataItem(
            pen=None, symbol='d', symbolSize=MARKER_SIZE,
            symbolPen=pg.mkPen(self.color), symbolBrush=pg.mkBrush(self.color))
        # to the left of the lower crossing, at the level: the fitting
        # screen hangs its damping figure to the right of the crown, and
        # the two must not sit on each other
        self.label: Any = pg.TextItem(color=self.color, anchor=(1.08, 0.5))
        box = plot.getViewBox()
        for k, item in enumerate((self.level, self.crossings, self.top,
                                  self.label)):
            item.setZValue(z + k * 0.1)
            box.addItem(item, ignoreBounds=True)
        self.reading: Any = None

    def show(self, reading: Any, scale: float = 1.0) -> None:
        """Place the overlay on `reading`, a `HalfPower`."""
        self.reading = reading
        height = reading.height * scale
        level = reading.level * scale
        found = [x for x in (reading.low, reading.high) if x is not None]
        # a missing side is drawn from the top to the found crossing only:
        # a level drawn out to where nothing was measured would claim one
        reach = REACH * (reading.width if reading.width is not None
                         else 2.0 * abs(found[0] - reading.frequency)
                         if found else 0.0)
        left = (reading.low - reach if reading.low is not None
                else reading.frequency)
        right = (reading.high + reach if reading.high is not None
                 else reading.frequency)
        log_x, log_y = self.plot.getViewBox().state['logMode']
        for item in (self.level, self.crossings, self.top):
            item.setLogMode(log_x, log_y)
        self.level.setData([left, right], [level, level])
        self.crossings.setData(found, [level] * len(found))
        self.top.setData([reading.frequency], [height])
        self.label.setText(label_text(reading))
        # a TextItem lives in view coordinates, which on a log axis are
        # the logarithm
        x = reading.low if reading.low is not None else reading.frequency
        self.label.setPos(float(np.log10(x)) if log_x else float(x),
                          float(np.log10(max(level, 1e-300))) if log_y
                          else float(level))
        for item in (self.level, self.crossings, self.top, self.label):
            item.setVisible(True)

    def clear(self) -> None:
        """Hide the overlay until the next `show`."""
        self.reading = None
        for item in (self.level, self.crossings, self.top, self.label):
            item.setVisible(False)


def add_half_power(plot: Any, reading: Any, *, color: Any = None,
                   scale: float = 1.0) -> HalfPowerOverlay:
    """Draw `reading`, a `HalfPower`, on `plot` and return the overlay.

    `color` and `scale` are as for `HalfPowerOverlay`.
    """
    overlay = HalfPowerOverlay(plot, color)
    overlay.show(reading, scale)
    return overlay
