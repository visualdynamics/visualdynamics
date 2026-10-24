"""A measured cross-spectral matrix against its specification, drawn as
the matrix: one small plot per cell, its verdict over it.

The question this view answers is whether the run was the test that was
asked for, and a multi-input test's answer is a matrix. So the picture
is the matrix itself, a grid with one row and one column per control
channel:

- on the diagonal, each channel's autospectrum (the measurement as it
  is compared, with the comparison's scale) over its specification and
  abort limits, on log axes;
- above the diagonal, each pair's coherence, measured over specified,
  with the tolerance either side of the specified one;
- below the diagonal, the same pair's phase in degrees, with its
  tolerance, drawn where the phase is judged.

Under the grid two strips read the whole matrix at each line: the RMS
over the channels of their level errors in dB, and how many cross terms
are out on either reading.

**Nothing here is carried by hue alone.** Every cell's title says its
reading and its verdict in words ('coherence OUT 23 %'). A cell's frame
says it too: heavy and solid round a cell that is out, thin round one
that is in, dotted round one that was not judged or not specified. The
measurement is a solid line and the specification a dashed one, the
tolerance and abort lines dotted, and the lines judged out wear a cross
marker on the measured curve.

Every number is `core.matrix_comparison.compare_matrix`'s.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.matrix_comparison import (
    COHERENCE_FLOOR,
    COHERENCE_TOLERANCE,
    PHASE_TOLERANCE,
    level_text,
)

#: the marker the lines judged out wear on the measured curve
OUT_SYMBOL = 'x'

#: the frame round a cell, by its state: (width, dotted)
FRAMES = {'out': (3.0, False), 'in': (1.0, False)}
#: and round anything else (not judged, not specified, not measured,
#: refused)
OTHER_FRAME = (1.0, True)


def short_words(comparison: Any, row: int, column: int) -> str:
    """A cell's verdict as its title carries it, in a few words:
    'level in +0.4 dB', 'coherence OUT 23 % of lines', 'phase not
    judged'. Out is written in capitals, so it is read before it is
    seen."""
    cell = comparison.cell(row, column)
    state = cell.state.upper() if cell.state == 'out' else cell.state
    value = ''
    if cell.state in ('in', 'out'):
        if cell.reading == 'level':
            db = float(comparison.levels[row].result.get(
                'difference_db', float('nan')))
            value = f' {level_text(db)}' if np.isfinite(db) else ''
        else:
            pair = comparison.pair(row, column)
            value = f' {getattr(pair, f"{cell.reading}_percent"):.0f} % of lines'
    return f'{cell.reading} {state}{value}'


def table_html(comparison: Any, color: str | None = None) -> str:
    """The summary as an HTML table: one row per reading of the matrix
    (each level, then each pair's coherence and phase) with its value
    and verdict, then the tolerances, the verdict and the notes."""
    from html import escape

    style = f' style="color:{color}"' if color else ''
    head = ('<tr><th align="left">cell</th><th align="left">reading</th>'
            '<th>value</th><th align="left">verdict</th></tr>')
    rows = ''.join(
        f'<tr><td>{escape(row["label"])}</td><td>{row["reading"]}</td>'
        f'<td align="right">{escape(row["value"])}</td>'
        f'<td>{escape(row["words"].partition(": ")[0])}</td></tr>'
        for row in comparison.table())
    tail = (f'coherence ±{comparison.coherence_tolerance:.2f}, phase '
            f'±{comparison.phase_tolerance:g}° where the specified '
            f'coherence is at least {comparison.coherence_floor:.2f}<br/>'
            f'{escape(comparison.describe())}')
    if comparison.scale_db:
        tail += f'<br/>measured levels scaled {comparison.scale_db:+g} dB'
    for note in comparison.notes:
        tail += f'<br/>{escape(note)}'
    return (f'<table cellspacing="0" cellpadding="2"{style}>{head}{rows}'
            f'</table><br/><span{style}>{tail}</span>')


def _frame(plot: Any, state: str, color: str) -> None:
    """The cell's frame, by its state (`FRAMES`)."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    width, dotted = FRAMES.get(state, OTHER_FRAME)
    plot.getViewBox().setBorder(pg.mkPen(
        color, width=width,
        style=Qt.PenStyle.DotLine if dotted else Qt.PenStyle.SolidLine))
    plot.cell_state = state


def _shown(values: Any, dimension: str, unit_system: Any) -> np.ndarray:
    """A density in display units; undeclared units drawn as they are."""
    from ..units import UNKNOWN

    values = np.real(np.asarray(values)).astype(float)
    if dimension == UNKNOWN:
        return values
    return np.asarray(unit_system.from_si(values, dimension), dtype=float)


def _curve(plot: Any, x: np.ndarray, y: np.ndarray, pen: Any,
           log_y: bool = False, log_x: bool = False) -> Any:
    """A curve broken where it is not a number (or, on a log axis, not
    above zero: the 0 Hz line is off a log frequency axis)."""
    y = np.asarray(y, dtype=float).copy()
    if log_y:
        y[~(y > 0.0)] = np.nan
    if log_x:
        y[~(np.asarray(x, dtype=float) > 0.0)] = np.nan
    return plot.plot(x, y, pen=pen, connect='finite')


def _span(lines: np.ndarray, log_x: bool) -> tuple[float, float]:
    """The frequency span a cell is drawn over, in the axis's own
    coordinates (decades on a log axis)."""
    shown = lines[lines > 0.0] if log_x else lines
    if not shown.size:
        return 0.0, 1.0
    low, high = float(shown.min()), float(shown.max())
    return (np.log10(low), np.log10(high)) if log_x else (low, high)


def _marks(plot: Any, x: np.ndarray, y: np.ndarray, out: np.ndarray,
           color: str) -> None:
    """The lines judged out, marked on the measured curve."""
    import pyqtgraph as pg

    if out is None or not np.any(out):
        return
    item = plot.plot(x[out], np.asarray(y, dtype=float)[out], pen=None,
                     symbol=OUT_SYMBOL, symbolSize=5,
                     symbolPen=pg.mkPen(color), symbolBrush=None)
    item.is_out_mark = True


def build_matrix_comparison(layout: Any, specification: Any, measured: Any,
                            *, coherence_tolerance: float = COHERENCE_TOLERANCE,
                            phase_tolerance: float = PHASE_TOLERANCE,
                            coherence_floor: float = COHERENCE_FLOOR,
                            scale_db: float | None = None,
                            comparison: Any = None,
                            table: bool = True, unit_system: Any = None,
                            theme: Any = None) -> Any:
    """Draw the grid, the two summary strips under it and (with `table`)
    the summary table beside it, and return the comparison
    (`core.matrix_comparison.compare_matrix`).

    A `comparison` already made of the two is drawn as it is, and the
    settings are not read; the application holds the last one and
    judges it again at a new tolerance (`with_tolerances`). The
    application puts its own table in the panel beside the plot and
    passes ``table=False``.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..core.matrix_comparison import compare_matrix
    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import _reading_plot, axis_label

    if comparison is None:
        comparison = compare_matrix(
            specification, measured,
            coherence_tolerance=coherence_tolerance,
            phase_tolerance=phase_tolerance,
            coherence_floor=coherence_floor, scale_db=scale_db)
    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    foreground = colors['plot_foreground']
    response = pg.mkPen(colors['response_curve'], width=1.2)
    target = pg.mkPen(colors['specification_curve'], width=1.2,
                      style=Qt.PenStyle.DashLine)
    bound = pg.mkPen(colors['limit_abort'], width=1.0,
                     style=Qt.PenStyle.DotLine)
    out_color = colors['exceed_over']
    log_x = bool(measured.log_abscissa)
    lines = np.asarray(comparison.frequency, dtype=float)
    scale = 10.0 ** (comparison.scale_db / 10.0)
    n = comparison.size
    spec_lines = np.asarray(specification.abscissa, dtype=float)

    plots: dict[tuple[int, int], Any] = {}
    for r in range(n):
        for c in range(n):
            cell = comparison.cell(r, c)
            plot = _reading_plot(layout, r, c)
            plot.setLogMode(x=log_x, y=(r == c))
            plot.matrix_cell = (r, c)
            plot.setTitle(f'{cell.label}<br/>{short_words(comparison, r, c)}',
                          color=foreground, size='8pt')
            _frame(plot, cell.state, foreground)
            plots[(r, c)] = plot
            if r == c:
                level = comparison.levels[r]
                dim = specification.ordinate_dim[level.spec_record]
                _curve(plot, spec_lines,
                       _shown(np.asarray(specification.ordinate)
                              [level.spec_record], dim, us), target, True, log_x)
                for name in ('abort_lower', 'abort_upper'):
                    limit = specification.limits.get(name)
                    if limit is not None:
                        _curve(plot, spec_lines,
                               _shown(limit[level.spec_record], dim, us),
                               bound, True, log_x)
                if level.measured_record is not None:
                    held = _shown(np.asarray(measured.ordinate)
                                  [level.measured_record],
                                  measured.ordinate_dim[level.measured_record],
                                  us) * scale
                    _curve(plot, lines, held, response, True, log_x)
                    _marks(plot, lines, held,
                           comparison_lines_out(specification, measured,
                                                level, comparison.scale_db),
                           out_color)
                if r == 0:
                    plot.setLabel('left', axis_label(dim, us))
                continue
            pair = comparison.pair(r, c)
            if not pair.lines:
                # nothing to draw, and the frame and title say why; the
                # same frequency span as its neighbors all the same
                plot.setXRange(*_span(lines, log_x), padding=0.0)
                continue
            if r < c:
                specified, held = (pair.specified_coherence,
                                   pair.measured_coherence)
                limit = comparison.coherence_tolerance
                judged = pair.lines['coherence_judged']
                out = pair.lines['coherence_out']
                plot.setYRange(0.0, 1.0, padding=0.02)
                # drawn where the cross term is written, and nowhere a
                # coherence is not asked for
                held = np.where(np.isfinite(specified), held, np.nan)
                low = np.clip(specified - limit, 0.0, 1.0)
                high = np.clip(specified + limit, 0.0, 1.0)
            else:
                specified, held = pair.specified_phase, pair.measured_phase
                limit = comparison.phase_tolerance
                judged = pair.lines['phase_judged']
                out = pair.lines['phase_out']
                plot.setYRange(-180.0, 180.0, padding=0.02)
                # drawn only where it is judged: elsewhere the phase is
                # asked of channels meant to be nearly independent, and
                # is the estimate's scatter
                specified = np.where(judged, specified, np.nan)
                # about the specified phase the short way round: a
                # measurement read 350 degrees off is drawn 10 off
                held = specified + np.where(judged, pair.lines['phase_error'],
                                            np.nan)
                low, high = specified - limit, specified + limit
            _curve(plot, lines, held, response, log_x=log_x)
            _curve(plot, lines, specified, target, log_x=log_x)
            _curve(plot, lines, low, bound, log_x=log_x)
            _curve(plot, lines, high, bound, log_x=log_x)
            _marks(plot, lines, held, out, out_color)

    # the per-line summary, the whole matrix at each line
    half = max(n // 2, 1)
    rms = _reading_plot(layout, n, 0, colspan=half)
    rms.setLogMode(x=log_x, y=False)
    rms.setTitle('level error, RMS over channels [dB]', color=foreground,
                 size='8pt')
    _curve(rms, lines, comparison.level_rms_db, response, log_x=log_x)
    rms.is_level_summary = True
    count = _reading_plot(layout, n, half, colspan=max(n - half, 1))
    count.setLogMode(x=log_x, y=False)
    count.setTitle('cross terms out at each line', color=foreground,
                   size='8pt')
    count.plot(lines, comparison.cross_out.astype(float),
               pen=pg.mkPen(colors['response_curve'], width=1.2,
                            style=Qt.PenStyle.DashLine), connect='finite')
    count.is_cross_summary = True
    for summary in (rms, count):
        summary.setLabel('bottom', 'frequency [Hz]')
    if table:
        text = pg.LabelItem(table_html(comparison, foreground),
                            justify='left', size='8pt')
        text.is_matrix_table = True
        layout.addItem(text, row=0, col=n, rowspan=n + 1)
    return comparison


def comparison_lines_out(specification: Any, measured: Any, level: Any,
                         scale_db: float) -> np.ndarray | None:
    """The measured lines of one channel outside its abort limits, as
    the comparison marks them (`compliance.exceedances`); None for a
    channel that was not compared."""
    from ..core.compliance import exceedances

    if level.state not in ('in', 'out') or level.measured_record is None:
        return None
    over, under = exceedances(specification, measured, level.spec_record,
                              level.measured_record, 'abort', scale_db)
    return over | under
