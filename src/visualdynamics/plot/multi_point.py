"""A multi-point control run on one plot: the composite heavy over its
channels, both bands, and the lines each channel is outside its own.

The question the view answers is the one the composite cannot: is
every control channel inside its own band, and where one is not, was
the composite inside its band there? So the composite is the heavy
curve the controller held, each channel stands behind it as a light
one, and the two bands are shaded as a specification's zones are
everywhere else here. Between the composite band and the individual
band is yellow, the ground a channel may stand in and the composite
may not; past the individual band is red above and blue below, where
no channel may be.

Every line a channel is outside its own band is marked on that
channel with its marker filled, and every line the composite is
outside its band with a filled circle on the composite. The legend
says, for each, how many lines are outside and, for each channel, how
many of those the composite was inside at, because that number is the
reason a multi-point run has two judgements.

**Each channel wears a marker as well as a color**, for the reason the
sine tracking view gives: several curves of one quantity are compared
line for line here, and a reader who cannot tell two of the palette's
hues apart must still be able to tell one channel from another. The
composite is told from them by its width and its color, the plot's
own foreground, never a palette hue.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..core.multi_point import INDIVIDUAL_DB, MultiPointReading

#: the marker each channel wears, in the order the channels are given
SYMBOLS = ('o', 's', 'd', 't1', 'star', 't', '+', 'x')

#: how wide the composite is drawn, against the channels' one and a half
COMPOSITE_WIDTH = 3.5

#: the size of a filled mark at a line outside its band: a shade
#: smaller than the hollow markers that name a curve, since a run of
#: them along a dead channel would otherwise be a solid bar
OUTSIDE_MARK_SIZE = 6


def _shown(values: Any, dimension: str, unit_system: Any,
           scale_db: float = 0.0) -> np.ndarray:
    """Values in display units, times the comparison's scale; a record
    whose units are undeclared is drawn as it is."""
    from ..units import UNKNOWN

    values = np.asarray(np.real(values), dtype=float) * 10.0 ** (
        float(scale_db) / 10.0)
    if dimension == UNKNOWN:
        return values
    return np.asarray(unit_system.from_si(values, dimension), dtype=float)


def describe_composite(reading: MultiPointReading) -> str:
    """The composite in a line: its strategy, how many channels, and how
    many lines are outside the composite band with their share, as in
    'composite, average of 4: 0 lines outside −3/+3 dB (0.0 %)'."""
    count = reading.outside_lines
    return (f'composite, {reading.strategy} of {len(reading.channels)}: '
            f'{count} line{"s" * (count != 1)} outside '
            f'{reading.composite_text} '
            f'({reading.result.get("outside_percent", 0.0):.1f} %)')


def describe_channel(channel: Any, reading: MultiPointReading) -> str:
    """One channel in a line: how many lines are outside its own band
    and how many of those the composite was inside at, as in
    '2Z+: 37 lines outside −6/+6 dB, 37 with the composite inside'."""
    count = channel.outside_lines
    text = (f'{channel.label}: {count} line{"s" * (count != 1)} outside '
            f'{reading.individual_text}')
    if count:
        text += f', {channel.hidden_lines} with the composite inside'
    return text


def build_multi_point(layout: Any, specification: Any, measured: Any, *,
                      channels: Sequence[int | str] | None = None,
                      strategy: str = 'average',
                      weights: Sequence[float] | None = None,
                      composite_band: str | tuple[float, float] = 'warning',
                      individual_band: (str | tuple[float, float])
                      = INDIVIDUAL_DB,
                      spec_record: int | None = None,
                      scale_db: float | None = None,
                      unit_system: Any = None,
                      theme: Any = None) -> MultiPointReading:
    """Draw a multi-point reading on `layout` and return it
    (`core.multi_point.multi_point`, with every argument passed on).

    The reference in gray, the composite band and the individual band
    around it shaded, each channel light in its own color and marker
    with its lines outside its own band filled, and the composite
    heavy with its lines outside its band marked. Every curve is the
    measurement scaled as the comparison scaled it, so what is drawn
    against the bands is what was judged against them.
    """
    import pyqtgraph as pg

    from ..core.multi_point import multi_point
    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import (
        STOOD_BACK_WIDTH,
        _draw_shaped,
        _Extents,
        _reading_plot,
        _shade_limit_zones,
        axis_label,
        curve_color,
        drawing_shape,
        legend_below,
        zone_colors,
    )
    from .sine_tracking import MARKER_SIZE, _marker_spots

    reading = multi_point(specification, measured, channels, strategy,
                          weights, composite_band, individual_band,
                          spec_record, scale_db)
    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    dimension = specification.ordinate_dim[reading.spec_record]

    plot = _reading_plot(layout, 0)
    plot.setLogMode(x=True, y=True)
    legend = legend_below(layout, plot, 0, colors)
    extents = _Extents()

    # the bands: the composite's as the inner pair, the individual
    # channels' as the outer, each a specification of one record
    # holding its band as its abort limits
    x = np.asarray(specification.abscissa, dtype=float)
    shape = drawing_shape(reading.composite_band)
    widths = specification.bandwidth
    bands = {}
    for band, inner in ((reading.composite_band, True),
                        (reading.individual_band, False)):
        for edge in ('lower', 'upper'):
            values = band.limits.get(f'abort_{edge}')
            if values is None:
                continue
            name = f'{"warning" if inner else "abort"}_{edge}'
            bands[name] = _shown(values[0], dimension, us)
    zones = _shade_limit_zones(plot, x, bands, colors, shape, widths)
    for values in bands.values():
        extents.add(x, values, True, True)
    required = _shown(reading.composite_band.ordinate[0], dimension, us)
    reference, _x, _y = _draw_shaped(
        plot, x, required, shape,
        pg.mkPen(colors['specification_curve'], width=STOOD_BACK_WIDTH),
        widths=widths, log_abscissa=True, log_ordinate=True)
    reference.setZValue(-1)
    extents.add(x, required, True, True)

    fx = np.asarray(measured.abscissa, dtype=float)
    m_shape = drawing_shape(measured)
    hollow = pg.mkBrush(0, 0, 0, 0)
    count = len(reading.channels)
    for k, channel in enumerate(reading.channels):
        color = curve_color(k, colors)
        symbol = SYMBOLS[k % len(SYMBOLS)]
        fy = _shown(measured.ordinate[channel.record],
                    measured.ordinate_dim[channel.record], us,
                    reading.scale_db)
        pen = pg.mkPen(color, width=1.5)
        _draw_shaped(plot, fx, fy, m_shape, pen, widths=measured.bandwidth,
                     log_abscissa=True, log_ordinate=True)
        good = np.isfinite(fy) & (fy > 0.0)
        spots = _marker_spots(fx, k, count, True)
        spots = spots[good[spots]]
        plot.plot(fx[spots], fy[spots], pen=None, symbol=symbol,
                  symbolSize=MARKER_SIZE, symbolPen=pg.mkPen(color),
                  symbolBrush=hollow)
        out = channel.outside & good
        if out.any():
            marks = plot.plot(fx[out], fy[out], pen=None, symbol=symbol,
                              symbolSize=OUTSIDE_MARK_SIZE,
                              symbolPen=pg.mkPen(color),
                              symbolBrush=pg.mkBrush(color))
            marks.is_outside_mark = True
        extents.add(fx, fy, True, True)
        legend.addItem(pg.PlotDataItem(
            [], [], pen=pen, symbol=symbol, symbolSize=MARKER_SIZE,
            symbolPen=pg.mkPen(color), symbolBrush=hollow),
            describe_channel(channel, reading))

    ink = colors['plot_foreground']
    cy = _shown(reading.composite.ordinate[0], reading.composite.ordinate_dim[0],
                us, reading.scale_db)
    heavy = pg.mkPen(ink, width=COMPOSITE_WIDTH)
    drawn, _x, _y = _draw_shaped(plot, fx, cy, m_shape, heavy,
                                 widths=measured.bandwidth,
                                 log_abscissa=True, log_ordinate=True)
    drawn.setZValue(2)
    drawn.is_composite = True
    out = (reading.over | reading.under) & np.isfinite(cy) & (cy > 0.0)
    if out.any():
        marks = plot.plot(fx[out], cy[out], pen=None, symbol='o',
                          symbolSize=OUTSIDE_MARK_SIZE,
                          symbolPen=pg.mkPen(ink), symbolBrush=pg.mkBrush(ink))
        marks.setZValue(3)
        marks.is_outside_mark = True
    extents.add(fx, cy, True, True)
    legend.addItem(pg.PlotDataItem([], [], pen=heavy),
                   describe_composite(reading))

    shades = zone_colors(colors)
    for zone, name in band_key(zones, reading):
        swatch = pg.BarGraphItem(x=[0.0], height=[1.0], width=1.0,
                                 pen=None, brush=pg.mkBrush(shades[zone]))
        swatch.is_zone_edge = True
        legend.addItem(swatch, name)
    legend.addItem(pg.PlotDataItem(
        [], [], pen=None, symbol=SYMBOLS[0], symbolSize=OUTSIDE_MARK_SIZE,
        symbolPen=pg.mkPen(ink), symbolBrush=pg.mkBrush(ink)),
        'filled: a line outside its band')

    plot.setTitle(f'{reading.strategy.capitalize()} of {count} control '
                  f'channels against {reading.composite_text}; each '
                  f'against {reading.individual_text}',
                  color=ink, size='9pt')
    plot.setLabel('bottom', 'frequency [Hz]')
    plot.setLabel('left', axis_label(dimension, us))
    extents.frame_y(plot)
    extents.lock(plot)
    return reading


def band_key(zones: set, reading: MultiPointReading
             ) -> list[tuple[str, str]]:
    """The legend entries for the shading, (zone, name), each only when
    it was drawn: `zones` holds any of 'warning', 'above' and 'below',
    as `_shade_limit_zones` returns them."""
    entries = []
    if 'warning' in zones:
        entries.append(('warning', (f'outside the composite band, '
                                    f'{reading.composite_text}')))
    if 'above' in zones:
        entries.append(('above', (f'above the individual band, '
                                  f'{reading.individual_text}')))
    if 'below' in zones:
        entries.append(('below', (f'below the individual band, '
                                  f'{reading.individual_text}')))
    return entries
