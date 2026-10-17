"""Two signatures of one article overlaid, with each mode's limits drawn
as a window about it and the table of shifts beside.

The question this view answers is the one a pre- and post-test
comparison is decided on: did each mode that counts stay inside its
limits. So the limits are drawn where the eye already is, as a box
about each mode's peak before the test: as wide as the frequency limit
and as tall as the amplitude limit. The peak after the test passes when
it lands inside its box. The table beside gives the numbers, because a
reader of a printed figure should not have to measure a box.

**Nothing here is carried by hue alone.** The two curves wear different
markers as well as colors (hollow circles before, hollow squares
after), each mode's peak is marked with its curve's marker filled, and
the boxes are told apart by line and by word: a mode that counts has a
solid box and its verdict, 'pass' or 'flag', written over it; a mode
that does not count has a dotted box and 'not judged'.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.signature import AMPLITUDE_LIMIT, FREQUENCY_LIMIT, SEARCH

#: the marker each curve wears: before, after
SYMBOLS = ('o', 's')


def _shown(values: Any, dimension: str, unit_system: Any) -> np.ndarray:
    """Magnitudes in display units; a record whose units are undeclared
    is drawn as it is."""
    from ..units import UNKNOWN

    values = np.abs(np.asarray(values)).astype(np.float64)
    if dimension == UNKNOWN:
        return values
    return np.asarray(unit_system.from_si(values, dimension), dtype=float)


def display_scale(dimension: str, unit_system: Any = None) -> float:
    """The factor from SI to the units a record of `dimension` is drawn
    in, so a height read in SI lands on the curve as drawn; 1 for a
    record whose units are undeclared."""
    from ..units import DEFAULT_SYSTEM

    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    return float(_shown(np.array([1.0]), dimension, us)[0])


def display_unit(dimension: str, unit_system: Any = None) -> str:
    """The unit a record of `dimension` is drawn in, as plain text ('g/g',
    'g/lbf'); empty for a record whose units are undeclared."""
    from ..units import DEFAULT_SYSTEM, UNKNOWN

    if dimension == UNKNOWN:
        return ''
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    return us.label_text(dimension)


def percent(value: float) -> str:
    """A shift as the table writes it: '−3.3 %'."""
    return f'{value:+.1f} %'.replace('-', '−')


def table_html(comparison: Any, scale: float = 1.0,
               color: str | None = None, unit: str = '') -> str:
    """The comparison as an HTML table: one row per mode, with its
    frequency and height before and after, the two shifts, Q before and
    after, and the verdict, then the limits, the overall verdict and
    the readings' notes. `scale` takes the heights from SI to the units
    drawn, and `unit` names those units."""
    from html import escape

    style = f' style="color:{color}"' if color else ''
    height = f'height before, after [{escape(unit)}]' if unit \
        else 'height before, after'
    head = ('<tr><th align="left">mode</th><th>Hz before, after</th>'
            f'<th>shift</th><th>{height}</th><th>shift</th>'
            '<th>Q before, after</th><th align="left">verdict</th></tr>')
    rows = []
    for shift in comparison.shifts:
        def q(reading):
            return '—' if reading.q is None else f'{reading.q:.3g}'

        rows.append(
            f'<tr><td>{escape(shift.mode.label())}</td>'
            f'<td align="right">{shift.before.frequency:.1f}, '
            f'{shift.after.frequency:.1f}</td>'
            f'<td align="right">{percent(shift.frequency_shift)}</td>'
            f'<td align="right">{shift.before.height * scale:#.3g}, '
            f'{shift.after.height * scale:#.3g}</td>'
            f'<td align="right">{percent(shift.amplitude_shift)}</td>'
            f'<td align="right">{q(shift.before)}, {q(shift.after)}</td>'
            f'<td>{shift.verdict}</td></tr>')
    f_low, f_high = comparison.frequency_limit
    a_low, a_high = comparison.amplitude_limit
    limits = (f'frequency limit {percent(f_low)} to {percent(f_high)}<br/>'
              f'amplitude limit {percent(a_low)} to {percent(a_high)}<br/>'
              f'{comparison.describe()}')
    for note in comparison.notes:
        limits += f'<br/>{escape(note)}'
    return (f'<table cellspacing="0" cellpadding="2"{style}>{head}'
            + ''.join(rows) + f'</table><br/><span{style}>{limits}</span>')


def build_signature(layout: Any, before: Any, after: Any, modes: Any, *,
                    record: int = 0, after_record: int | None = None,
                    frequency_limit: Any = FREQUENCY_LIMIT,
                    amplitude_limit: Any = AMPLITUDE_LIMIT,
                    search: float = SEARCH,
                    names: tuple[str, str] = ('before', 'after'),
                    table: bool = True,
                    unit_system: Any = None, theme: Any = None) -> Any:
    """Draw the two signatures, each mode's window and (with `table`)
    the table beside them, and return the comparison
    (`core.signature.compare_signatures`).

    `names` labels the two curves in the legend. The application puts
    its own table in the panel beside the plot and passes
    ``table=False``.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..core.signature import compare_signatures, matching_record
    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import _Extents, axis_label, curve_color, legend_below
    from .sine_tracking import MARKER_SIZE, _marker_spots

    comparison = compare_signatures(
        before, after, modes, record=record, after_record=after_record,
        frequency_limit=frequency_limit, amplitude_limit=amplitude_limit,
        search=search)
    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    after_record = (matching_record(before, record, after)
                    if after_record is None else after_record)
    dimension = before.ordinate_dim[record]

    plot = layout.addPlot(row=0, col=0)
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setLogMode(x=False, y=True)
    legend = legend_below(layout, plot, 0, colors)
    extents = _Extents()

    hollow = pg.mkBrush(0, 0, 0, 0)
    for k, (data, i) in enumerate(((before, record), (after, after_record))):
        color = curve_color(k, colors)
        x = np.asarray(data.abscissa, dtype=float)
        y = _shown(np.atleast_2d(data.ordinate)[i], data.ordinate_dim[i], us)
        pen = pg.mkPen(color, width=1.5)
        plot.plot(x, y, pen=pen, connect='finite')
        spots = _marker_spots(x, k, 2, False)
        plot.plot(x[spots], y[spots], pen=None, symbol=SYMBOLS[k],
                  symbolSize=MARKER_SIZE, symbolPen=pg.mkPen(color),
                  symbolBrush=hollow)
        extents.add(x, y, True)
        legend.addItem(pg.PlotDataItem(
            [], [], pen=pen, symbol=SYMBOLS[k], symbolSize=MARKER_SIZE,
            symbolPen=pg.mkPen(color), symbolBrush=hollow), names[k])

    # the scale from SI to what is drawn, read off one value, so the
    # boxes and the table land on the curves whatever the unit system
    scale = display_scale(dimension, us)
    unit = display_unit(dimension, us)
    foreground = colors['plot_foreground']
    for shift in comparison.shifts:
        f_low, f_high, h_low, h_high = shift.box()
        h_low, h_high = h_low * scale, h_high * scale
        counts = shift.mode.counts
        edge = pg.mkPen(foreground if counts else colors['specification_curve'],
                        width=1.5 if counts else 1.2,
                        style=(Qt.PenStyle.SolidLine if counts
                               else Qt.PenStyle.DotLine))
        box = plot.plot([f_low, f_high, f_high, f_low, f_low],
                        [h_low, h_low, h_high, h_high, h_low], pen=edge)
        box.is_signature_window = True
        box.mode_counts = counts
        extents.add(np.array([f_low, f_high]), np.array([h_low, h_high]),
                    True)
        for k, reading in enumerate((shift.before, shift.after)):
            color = curve_color(k, colors)
            peak = plot.plot([reading.frequency], [reading.height * scale],
                             pen=None, symbol=SYMBOLS[k],
                             symbolSize=MARKER_SIZE + 3,
                             symbolPen=pg.mkPen(color),
                             symbolBrush=pg.mkBrush(color))
            peak.is_signature_peak = True
        label = pg.TextItem(f'{shift.mode.label()}: {shift.verdict}',
                            color=foreground, anchor=(0.5, 1.0))
        label.setPos(shift.before.frequency, np.log10(h_high))
        label.is_signature_label = True
        plot.addItem(label, ignoreBounds=True)

    legend.addItem(pg.PlotDataItem(
        [], [], pen=pg.mkPen(foreground, width=1.5)),
        'window: the limits about a mode that counts')
    if any(not s.mode.counts for s in comparison.shifts):
        legend.addItem(pg.PlotDataItem(
            [], [], pen=pg.mkPen(colors['specification_curve'], width=1.2,
                                 style=Qt.PenStyle.DotLine)),
            'dotted: a mode reported, not judged')

    where = comparison.dof
    plot.setTitle(f'{where}: {comparison.describe()}' if where
                  else comparison.describe(), color=foreground, size='9pt')
    plot.setLabel('bottom', 'frequency [Hz]')
    plot.setLabel('left', axis_label(dimension, us))
    extents.frame_y(plot)
    extents.lock(plot)
    if table:
        text = pg.LabelItem(table_html(comparison, scale, foreground, unit),
                            justify='left', size='8pt')
        text.is_signature_table = True
        layout.addItem(text, row=0, col=1)
        # the plot is what is read first; the table, set in small type,
        # needs about a third of the width and no more
        grid = getattr(layout, 'ci', layout).layout
        grid.setColumnStretchFactor(0, 2)
        grid.setColumnStretchFactor(1, 1)
    return comparison
