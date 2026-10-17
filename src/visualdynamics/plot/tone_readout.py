"""A tone read off an autospectrum, drawn on the spike it was read from.

The reading (`core.tone_readout.read_tone`) puts two numbers side by
side that a plot of an ASD invites a reader to confuse: the height of
the spike, which moves with the line spacing and the window, and the
tone's amplitude, which does not. The overlay draws both at the tone:

- a stem from the floor up to the height this tone stands at for the
  spacing, window and placement in use, capped with a downward triangle.
  It is what a controller that draws its tone references on the ASD
  scales them to, and it lands on the spike's top when the reading is
  right, so a reading taken off the wrong spike, or a floor misjudged,
  shows as a triangle that misses;
- a dotted line at the floor, across the lines the tone was summed
  over, so what was taken away is seen;
- the amplitude in words beside the triangle, with the spike's height
  under it, and "between lines" when the tone sits far enough off a
  line that its spike reads short.

Shapes, line styles and words carry the meaning, not a color: the
overlay takes one color, the plot's foreground unless the caller passes
another. The items go on the plot's view box and are kept out of its
bounds, as `plot.half_power`'s are: they are a reading of the data, not
data, so the legend, the copy and the plot's own preview never include
them, and drawing them never re-zooms the view.

`ToneOverlay` owns its items and moves them, so a cursor that reads on
every move updates the same items; `add_tone_readout` draws one reading
once, for a figure.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: the triangle's size, pixels
MARKER_SIZE = 11


def label_text(reading: Any, unit: str = '', scale: float = 1.0) -> str:
    """What the overlay writes beside a tone: its amplitude, then the
    spike's height in the spectrum's units, then 'between lines' when
    the spike reads short for it, or 'uncertain' when a note on the
    reading bears on the amplitude. `scale` converts the amplitude into
    `unit`; the density takes its square."""
    from ..core.tone_readout import density_label

    if not reading.amplitude:
        return 'no tone'
    density = density_label(unit)
    text = (f'{reading.amplitude * scale:.3g} {unit} peak'.replace('  ', ' ')
            + f'\nspike {reading.spike * scale ** 2:.3g} {density}')
    if reading.uncertain:
        text += '\nuncertain'
    elif reading.between_lines:
        text += '\nbetween lines'
    return text


def _view(value: float, log: bool) -> float:
    """A coordinate in the view box's own terms: the logarithm on a log
    axis, where a TextItem lives."""
    return float(np.log10(max(value, 1e-300))) if log else float(value)


class ToneOverlay:
    """The items that draw one tone reading on a pyqtgraph plot.

    Made once per plot with `plot`, the PlotItem to draw on, and
    `color`, a pyqtgraph color (the plot's foreground by default); then
    `show(reading, ...)` places them and `clear()` hides them. The words
    go to the right of the triangle, or with `above=True` centered over
    it, which is how a figure of several tones keeps them apart.
    """

    def __init__(self, plot: Any, color: Any = None, *,
                 above: bool = False, z: float = 42.0) -> None:
        import pyqtgraph as pg
        from PySide6.QtCore import Qt

        if color is None:
            color = plot.getAxis('left').pen().color()
        self.plot: Any = plot
        self.color: Any = pg.mkColor(color)
        #: the stem, the triangle on it, the floor's dotted line and the
        #: words
        self.stem: Any = pg.PlotDataItem(pen=pg.mkPen(self.color, width=2))
        self.top: Any = pg.PlotDataItem(
            pen=None, symbol='t', symbolSize=MARKER_SIZE,
            symbolPen=pg.mkPen(self.color), symbolBrush=pg.mkBrush(self.color))
        self.floor: Any = pg.PlotDataItem(
            pen=pg.mkPen(self.color, width=1.5, style=Qt.PenStyle.DotLine))
        # to the right of the triangle, level with it, for a cursor:
        # above it the words run off the top of a view fitted to the
        # spikes. A figure makes the headroom and centers them over it
        self.label: Any = pg.TextItem(
            color=self.color, anchor=(0.5, 1.15) if above else (-0.06, 0.5))
        box = plot.getViewBox()
        for k, item in enumerate(self.items()):
            item.setZValue(z + k * 0.1)
            box.addItem(item, ignoreBounds=True)
        self.reading: Any = None

    def items(self) -> tuple[Any, ...]:
        """Every item the overlay draws with."""
        return (self.floor, self.stem, self.top, self.label)

    def show(self, reading: Any, frequencies: Any = None, *,
             scale: float = 1.0, density_scale: float | None = None,
             unit: str = '') -> None:
        """Place the overlay on `reading`, a `ToneReading`.

        `frequencies` are the spectrum's lines, which place the floor's
        line across the lines summed; without them it spans the main
        lobe about the tone. `scale` converts the amplitude for the
        words and `density_scale` the densities for the drawing (the
        square of `scale` when None), for a spectrum drawn in other
        units than it was read in; `unit` names the amplitude's.
        """
        self.reading = reading
        dense = scale ** 2 if density_scale is None else density_scale
        floor = max(reading.floor, 0.0) * dense
        top = floor + reading.height * dense
        if frequencies is not None:
            f = np.asarray(frequencies, dtype=np.float64)
            low, high = (float(f[reading.lines[0]]),
                         float(f[reading.lines[1]]))
        else:
            reach = 0.5 * (reading.lines[1] - reading.lines[0])
            low = reading.frequency - reach * reading.spacing
            high = reading.frequency + reach * reading.spacing
        log_x, log_y = self.plot.getViewBox().state['logMode']
        for item in (self.stem, self.top, self.floor):
            item.setLogMode(log_x, log_y)
        # a floor of zero has no place on a log axis: the stem starts
        # at the spike's own height there, and no floor line is drawn
        bottom = floor if floor > 0.0 or not log_y else top
        self.stem.setData([reading.frequency] * 2, [bottom, top])
        self.top.setData([reading.frequency], [top])
        if floor > 0.0 or not log_y:
            self.floor.setData([low, high], [floor, floor])
        else:
            self.floor.setData([], [])
        self.label.setText(label_text(reading, unit, scale))
        self.label.setPos(_view(reading.frequency, log_x),
                          _view(top, log_y))
        for item in self.items():
            item.setVisible(True)

    def clear(self) -> None:
        """Hide the overlay until the next `show`."""
        self.reading = None
        for item in self.items():
            item.setVisible(False)

    def remove(self) -> None:
        """Take the items off the plot for good."""
        box = self.plot.getViewBox()
        for item in self.items():
            box.removeItem(item)


def add_tone_readout(plot: Any, reading: Any, frequencies: Any = None, *,
                     color: Any = None, above: bool = False,
                     scale: float = 1.0,
                     density_scale: float | None = None,
                     unit: str = '') -> ToneOverlay:
    """Draw `reading`, a `ToneReading`, on `plot` and return the
    overlay. `color` and `above` are as for `ToneOverlay`, the others
    as for `ToneOverlay.show`."""
    overlay = ToneOverlay(plot, color, above=above)
    overlay.show(reading, frequencies, scale=scale,
                 density_scale=density_scale, unit=unit)
    return overlay


def display_units(data: Any, record: int, unit_system: Any
                  ) -> tuple[float, float, str]:
    """(amplitude scale, density scale, amplitude unit) for showing a
    tone read off record `record` of `data` in `unit_system`: g and
    g²/Hz for an acceleration's autospectrum shown in g, ones and an
    empty unit for a record whose units are not declared or that is not
    an autospectrum."""
    from ..core.data import channel_quantities
    from ..units import UNKNOWN

    dimension = str(data.ordinate_dim[record])
    amplitude, other = channel_quantities(dimension)
    if dimension == UNKNOWN or amplitude != other:
        return 1.0, 1.0, ''
    return (float(unit_system.from_si(1.0, amplitude)),
            float(unit_system.from_si(1.0, dimension)),
            unit_system.label_text(amplitude))
