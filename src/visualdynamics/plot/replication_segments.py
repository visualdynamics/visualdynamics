"""A replicated record judged segment by segment: the two traces
overlaid with the segments shaded, their difference beneath, and the
verdict by segment in a table under both.

The question the view answers is the one the whole-record figure
cannot: which stretch of the record was reproduced and which was not.
So the record is drawn whole, the reference standing back in gray and
the control over it, each marked segment shaded across both plots, and
the table under them gives, for the record and then for each segment,
its share of the record's mean square, its own rms error, and each of
its checks against its tolerance. The whole-record rms error is the
table's first row, so the number that hides a segment's miss is read
beside the one that shows it.

**A verdict is never carried by hue alone.** Each segment's shading
is labeled in words with its name, its kind and its verdict, and its
edges are drawn in a line style of its kind (solid for stationary,
dashed for transient, dotted for shock); a failing segment's fill is
the exceedance red where a passing one is the analysis blue, which
adds to the words and replaces none of them. In the table a failing
verdict is set in bold.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from . import _long_trace, _reading_plot

#: each kind's edge style, as Qt pen styles by name
KIND_STYLES = {'stationary': 'SolidLine', 'transient': 'DashLine',
               'shock': 'DotLine'}

#: how solid a segment's fill is: light, since the traces are what is
#: read and the shading only brackets them
FILL_ALPHA = 40

#: the table's columns, in order
COLUMNS = ('Segment', 'Kind', 'Span [s]', 'Share', 'RMS error', 'Check',
           'Reading', 'Tolerance', 'Verdict')


def _verdict(passed: bool | None) -> str:
    return {True: 'passes', False: 'fails', None: 'not judged'}[passed]


def _number(value: float, places: int = 2) -> str:
    """A number for the table, with the minus sign; a dash for NaN."""
    if not np.isfinite(value):
        return '\u2014'
    return f'{value:.{places}f}'.replace('-', '\u2212')


def _reading(check: Any) -> str:
    """A check's reading as the table gives it: its range, or its one
    value, with its unit."""
    from ..core.replication_segments import _signed

    if not check.judged:
        return 'nothing to judge'
    if check.judged == 1:
        return _signed(check.low, check.unit)
    text = (f'{_signed(check.low, check.unit)} to '
            f'{_signed(check.high, check.unit)}')
    return text + f'; {check.outside} of {check.judged} outside'


def verdict_rows(result: Any) -> list[tuple[str, ...]]:
    """The table's rows, one per check, the whole record first, each a
    tuple of `COLUMNS` strings. A segment's name, kind, span, share and
    error are given on its first row only."""
    from ..core.replication_segments import band_text

    rows = [('whole record', '\u2014',
             f'0 to {result.samples / result.sample_rate:g}',
             '100.00 %', f'{_number(result.rms_error)} %', 'rms error',
             f'{_number(result.rms_error)} %',
             'none: the controller\'s figure', '\u2014')]
    for segment in result.segments:
        seg = segment.segment
        for k, check in enumerate(segment.checks):
            head = ((segment.label, seg.kind,
                     f'{seg.start:g} to {seg.end:g}',
                     f'{_number(segment.share)} %',
                     f'{_number(segment.rms_error)} %') if k == 0
                    else ('', '', '', '', ''))
            rows.append((*head, check.reading, _reading(check),
                         band_text(check.tolerance, check.unit),
                         _verdict(check.passed)))
    return rows


def table_html(result: Any, foreground: str) -> str:
    """The verdict table as rich text, for a `LabelItem`: a header row,
    then `verdict_rows`, with each failing verdict and each failing
    segment's name in bold."""
    def cell(text: str, bold: bool = False, tag: str = 'td') -> str:
        # a segment's name is the user's text, and a '<' in it would be
        # read as markup (2026-10-16 review)
        body = html.escape(text)
        body = f'<b>{body}</b>' if bold else body
        return (f'<{tag} style="padding: 1px 8px 1px 0px; '
                f'color: {foreground};">{body}</{tag}>')

    failed = {segment.label for segment in result.segments
              if segment.passed is False}
    lines = ['<table cellspacing="0">',
             '<tr>' + ''.join(cell(name, True, 'th') for name in COLUMNS)
             + '</tr>']
    for row in verdict_rows(result):
        lines.append('<tr>' + ''.join(
            cell(text, bold=(k == 0 and text in failed)
                 or (k == len(COLUMNS) - 1 and text == 'fails'))
            for k, text in enumerate(row)) + '</tr>')
    lines.append('</table>')
    return ''.join(lines)


def describe_record(result: Any) -> str:
    """The plot's title: the channel, how many segments fail, and the
    whole-record figure beside them."""
    where = f'{result.channel}: ' if result.channel else ''
    return where + result.describe()


def build_replication_segments(layout: Any, measured: Any,
                               specification: Any,
                               segments: Sequence[Any], *,
                               channel: str | None = None,
                               lag: int | None = None, playing: int = 0,
                               unit_system: Any = None, theme: Any = None,
                               **settings: Any) -> Any:
    """Draw one channel of a run against its reference, segments shaded,
    with the verdict table, and return the judgment
    (`core.replication_segments.replication_segments`).

    Three rows of `layout`: the two traces overlaid (the reference in
    gray, the control in the first curve color) with their legend, the
    difference, control minus reference, on the same time axis, and the
    table (`verdict_rows`). The settings are the judgment's.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor

    from ..core.replication_segments import aligned_traces, judge_segments
    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM, UNKNOWN
    from . import STOOD_BACK_WIDTH, axis_label, curve_color, legend_below

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    time, reference, control, dof, shift = aligned_traces(
        measured, specification, channel, lag, playing)
    judged = judge_segments(reference, control,
                            float(specification.sample_rate), segments,
                            **settings)
    result = replace(judged, channel=dof, lag=shift)
    dimension = specification.ordinate_dim[
        [str(d) for d in specification.response_dof].index(dof)]

    def shown(values):
        if dimension == UNKNOWN:
            return values
        return np.asarray(us.from_si(values, dimension), dtype=float)

    reference_shown, control_shown = shown(reference), shown(control)
    foreground = colors['plot_foreground']
    time_label = f'time [{us.label_html("time")}]'

    overlay = _reading_plot(layout, 0)
    # the legend under the plot takes every named curve added after it
    legend_below(layout, overlay, 0, colors)
    _long_trace(overlay, time, reference_shown,
           pg.mkPen(colors['specification_curve'], width=STOOD_BACK_WIDTH),
           'reference')
    _long_trace(overlay, time, control_shown,
           pg.mkPen(curve_color(0, colors), width=1), 'control')
    overlay.setLabel('left', axis_label(dimension, us))
    overlay.setTitle(describe_record(result), color=foreground, size='9pt')

    difference = _reading_plot(layout, 2)
    difference.setXLink(overlay)
    _long_trace(difference, time, control_shown - reference_shown,
           pg.mkPen(curve_color(0, colors), width=1),
           'control \u2212 reference')
    difference.setLabel('left', f'difference [{axis_label(dimension, us)}]')
    difference.setLabel('bottom', time_label)
    difference.setMaximumHeight(170)

    passing = QColor(colors['averaging_band'])
    passing.setAlpha(FILL_ALPHA)
    failing = QColor(colors['exceed_over'])
    failing.setAlpha(FILL_ALPHA)
    top = float(np.nanmax(np.abs(np.r_[reference_shown, control_shown,
                                       0.0])))
    overlay.setYRange(-1.15 * top, 1.3 * top, padding=0)
    for k, segment in enumerate(result.segments):
        seg = segment.segment
        fill = failing if segment.passed is False else passing
        edge = QColor(foreground)
        edge.setAlpha(160)
        pen = pg.mkPen(edge, width=1,
                       style=getattr(Qt.PenStyle, KIND_STYLES[seg.kind]))
        for plot in (overlay, difference):
            region = pg.LinearRegionItem(values=(seg.start, seg.end),
                                         brush=pg.mkBrush(fill), pen=pen,
                                         movable=False)
            region.setZValue(-15)          # under the traces it brackets
            for line in region.lines:
                line.setPen(pen)
                line.setHoverPen(pen)
            region.segment = segment
            plot.addItem(region, ignoreBounds=True)
        # an unnamed segment is already called by its kind
        kind = f'{seg.kind}, ' if seg.name else ''
        label = pg.TextItem(
            f'{segment.label}\n{kind}{_verdict(segment.passed)}',
            color=foreground, anchor=(0.0, 0.0))
        # alternate heights, so two short segments side by side do not
        # write over each other
        label.setPos(seg.start, top * (1.28 if k % 2 == 0 else 1.1))
        label.segment = segment
        overlay.addItem(label, ignoreBounds=True)

    table = pg.LabelItem(justify='left', color=foreground, size='8pt')
    table.setText(table_html(result, foreground))
    table.is_verdict_table = True
    layout.addItem(table, row=4, col=0)
    return result




