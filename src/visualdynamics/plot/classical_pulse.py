"""A measured classical pulse on its tolerance envelope, and its velocity
across the window the velocity change is read over: two rows.

The question this view answers is the one a classical shock is accepted
on: did the pulse stay inside its envelope, from the quiet before it to
the quiet after, and did it change velocity by what the nominal pulse
does. So the envelope is shaded behind the record with its edges drawn,
the nominal shape stands back dashed in gray, every sample outside the
envelope is marked, and the velocity window is shaded across the
record's plot. Below, the velocity climbs across that window from rest
to the measured change, against the nominal change and its tolerance.

Each row's title carries its reading in words, and the record's
carries the verdict. A sample outside the envelope wears a filled
marker, named in the legend with the count, so being out is read from a
shape and a word, not from a hue alone; the two shadings are told apart
by their direction (the envelope follows the pulse, the window spans
the plot's height) and their legend names.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: how solid the envelope and the window are: light, since the record
#: is what is read and the shading only brackets it
SHADE_ALPHA = 45

#: the most samples outside the envelope that are each given a marker;
#: past it every k-th is, so a record sampled at a megahertz does not
#: bury its own trace. The count in the legend is always the whole.
MOST_MARKS = 400

#: points the envelope's edges are drawn at across the judged span, so
#: they keep their corners on a coarsely sampled record
ENVELOPE_POINTS = 4001


def build_pulse_check(layout: Any, check: Any, *, unit_system: Any = None,
                      theme: Any = None, name: str | None = None) -> None:
    """Draw a `core.classical_pulse.PulseCheck` into `layout`.

    The record over the judged span on its envelope, with the velocity
    window shaded across it, above; the velocity across the window
    beneath, sharing its time axis. `name` labels the record (its DOF)
    in the titles.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor

    from ..core.classical_pulse import pulse_envelope
    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import axis_label, legend_below
    from .sine_tracking import MARKER_SIZE

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    foreground = colors['plot_foreground']
    pulse = check.pulse
    measured = check.measured

    def shown(values: Any, dimension: str) -> np.ndarray:
        return np.asarray(us.from_si(np.asarray(values, dtype=float),
                                     dimension), dtype=float)

    def shade(key: str) -> Any:
        color = QColor(colors[key])
        color.setAlpha(SHADE_ALPHA)
        return color

    time = shown(check.time, 'time')
    time_label = f'time [{us.label_html("time")}]'

    # ---- the record on its envelope --------------------------------------
    record_plot = _plot(layout, 0)
    legend = legend_below(layout, record_plot, 0, colors)
    if check.time.size:
        fine = np.linspace(check.time[0], check.time[-1], ENVELOPE_POINTS)
        # the judged samples' own times as well, so a sample drawn on an
        # edge is drawn on the line it was judged against
        fine = np.union1d(fine, check.time)
        lower, upper = pulse_envelope(
            pulse, fine - measured.start, band=check.band,
            duration_tolerance=check.duration_tolerance,
            edge_limit=check.edge_limit)
        x = shown(fine, 'time')
        edge_pen = pg.mkPen(colors['specification_curve'], width=1.5)
        low_line = record_plot.plot(x, shown(lower, 'acceleration'),
                                    pen=edge_pen)
        high_line = record_plot.plot(x, shown(upper, 'acceleration'),
                                     pen=edge_pen)
        band = pg.FillBetweenItem(low_line, high_line,
                                  brush=pg.mkBrush(shade('averaging_band')))
        band.setZValue(-20)
        record_plot.addItem(band)
        legend.addItem(pg.PlotDataItem([], [], pen=edge_pen),
                       f'envelope: ±{100 * check.band:g} % of the peak')
        nominal_pen = pg.mkPen(colors['specification_curve'], width=1.5,
                               style=Qt.PenStyle.DashLine)
        record_plot.plot(x, shown(pulse.at(fine - measured.start),
                                  'acceleration'), pen=nominal_pen)
        legend.addItem(pg.PlotDataItem([], [], pen=nominal_pen),
                       'nominal ' + pulse.shape)

    # the window the velocity change is read over, across the plot's
    # height: an upright band, where the envelope follows the pulse
    opens, closes = shown(np.array(check.window), 'time')
    window = pg.LinearRegionItem(
        values=(opens, closes), orientation='vertical', movable=False,
        brush=pg.mkBrush(shade('averaging_window')),
        pen=pg.mkPen(shade('averaging_window')))
    window.setZValue(-25)
    record_plot.addItem(window, ignoreBounds=True)
    swatch = pg.BarGraphItem(x=[0.0], height=[1.0], width=1.0, pen=None,
                             brush=pg.mkBrush(shade('averaging_window')))
    legend.addItem(swatch, 'velocity window')

    trace_pen = pg.mkPen(colors['response_curve'], width=1.5)
    record_plot.plot(time, shown(check.acceleration, 'acceleration'),
                     pen=trace_pen)
    legend.addItem(pg.PlotDataItem([], [], pen=trace_pen),
                   name or 'measured')

    out = np.flatnonzero(check.outside)
    if out.size:
        step = max(1, -(-out.size // MOST_MARKS))
        drawn = out[::step]
        color = colors['limit_abort']
        marks = record_plot.plot(
            time[drawn], shown(check.acceleration[drawn], 'acceleration'),
            pen=None, symbol='o', symbolSize=MARKER_SIZE,
            symbolPen=pg.mkPen(color), symbolBrush=pg.mkBrush(color))
        marks.is_outside_mark = True
        legend.addItem(pg.PlotDataItem(
            [], [], pen=None, symbol='o', symbolSize=MARKER_SIZE,
            symbolPen=pg.mkPen(color), symbolBrush=pg.mkBrush(color)),
            f'outside the envelope: {check.outside_count} of '
            f'{check.points} points')

    record_plot.setLabel('left', axis_label('acceleration', us))
    unit = us.label_text('acceleration')
    title = (f'{name + ": " if name else ""}'
             f'{pulse.describe(unit, float(us.from_si(1.0, "acceleration")))}'
             f'<br>{check.describe()}')
    record_plot.setTitle(title, color=(foreground if check.passed
                                       else colors['limit_abort']),
                         size='9pt')

    # ---- the velocity across the window ----------------------------------
    velocity_plot = _plot(layout, 2)
    velocity_plot.setXLink(record_plot)
    nominal = check.nominal_velocity_change
    reach = check.velocity_tolerance * abs(nominal)
    tolerance = pg.LinearRegionItem(
        values=tuple(shown(np.array([nominal - reach, nominal + reach]),
                           'velocity')),
        orientation='horizontal', movable=False,
        brush=pg.mkBrush(shade('averaging_band')),
        pen=pg.mkPen(shade('averaging_band')))
    tolerance.setZValue(-20)
    velocity_plot.addItem(tolerance)
    velocity_plot.addItem(pg.InfiniteLine(
        pos=float(shown(nominal, 'velocity')), angle=0,
        pen=pg.mkPen(colors['specification_curve'], width=1.5,
                     style=Qt.PenStyle.DashLine)))
    velocity_plot.plot(shown(check.velocity_time, 'time'),
                       shown(check.velocity, 'velocity'),
                       pen=pg.mkPen(colors['response_curve'], width=1.5))
    velocity_plot.setLabel('left', axis_label('velocity', us))
    velocity_plot.setLabel('bottom', time_label)
    speed = us.label_text('velocity')
    velocity_plot.setTitle(
        f'velocity change {float(shown(check.velocity_change, "velocity")):.4g}'
        f' {speed} against the nominal '
        f'{float(shown(nominal, "velocity")):.4g} {speed} '
        f'± {100 * check.velocity_tolerance:g} % (dashed, shaded)'
        .replace('-', '−'),
        color=(foreground if check.velocity_ok else colors['limit_abort']),
        size='9pt')


def _plot(layout: Any, row: int) -> Any:
    """A plot in `row` of the layout, gridded, with no SI scaling: the
    units are in the labels."""
    plot = layout.addPlot(row=row, col=0)
    plot.showGrid(x=True, y=True, alpha=0.2)
    for side in ('left', 'bottom'):
        plot.getAxis(side).enableAutoSIPrefix(False)
    return plot


def readout(check: Any, unit_system: Any = None) -> dict[str, str]:
    """The numbers of a check, short enough for the app's panel, in
    display units: 'peak', 'duration' and 'velocity' as the measured
    value and its distance from nominal, 'rise' and 'fall' (where the
    shape has them) against the limit, and 'outside' as a count. A
    value out of its tolerance ends in 'out', so the panel says so in a
    word.
    """
    from ..units import DEFAULT_SYSTEM

    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    measured = check.measured

    def out(ok: bool) -> str:
        return '' if ok else ', out'

    level = (f'{float(us.from_si(measured.peak, "acceleration")):.4g} '
             f'{us.label_text("acceleration")}')
    speed = (f'{float(us.from_si(check.velocity_change, "velocity")):.4g} '
             f'{us.label_text("velocity")}')
    limit = f'{check.edge_allowance * 1e3:.3g}'
    rows = {'peak': f'{level}, {_percent(check.peak_ratio)}'
                    f'{out(check.peak_ok)}',
            'duration': f'{measured.duration * 1e3:.4g} ms, '
                        f'{_percent(check.duration_ratio)}'
                        f'{out(check.duration_ok)}'}
    if measured.rise is not None:
        rows['rise'] = (f'{measured.rise * 1e3:.3g} of {limit} ms'
                        f'{out(check.rise_ok)}')
    if measured.fall is not None:
        rows['fall'] = (f'{measured.fall * 1e3:.3g} of {limit} ms'
                        f'{out(check.fall_ok)}')
    rows['velocity'] = (f'{speed}, {_percent(check.velocity_ratio)}'
                        f'{out(check.velocity_ok)}')
    rows['outside'] = f'{check.outside_count} of {check.points}'
    return {key: value.replace('-', '\u2212')
            for key, value in rows.items()}


def _percent(ratio: float) -> str:
    return f'{100.0 * (ratio - 1.0):+.1f} %'
