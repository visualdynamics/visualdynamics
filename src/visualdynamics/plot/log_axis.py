"""Log-axis tick labels that fit the axis they are drawn on.

pyqtgraph's log axis labels every tick from 1 to 9 in each decade when
the axis spans few decades, and the first level of tick text is always
drawn, however little room there is: two decades on a plot 100 pixels
tall carried eighteen labels on top of each other. It also scales a log
axis by an SI prefix, so a spectrum in N·s read '10¹' for 1e-5 beside a
'×1e-06' on the axis label.

Here the labels are the densest set that fits: every mantissa, then 1,
2 and 5, then 1 and 3, then the decades alone, then every second, fifth
or tenth decade. The room a label needs is measured from the axis font:
a line's height on a vertical axis, the widest label's width on a
horizontal one. Every tick from 1 to 9 is still drawn, unlabeled, while
there are pixels for it, so the grid reads as a log grid. A span too
narrow to hold two of those labels (less than a factor of two, say) is
labeled at round values in between instead. A log axis takes no SI
prefix: its labels carry their own powers of ten.

This applies to every axis in the process. The package makes its plots
in more than thirty places, each through pyqtgraph's default axis, and
how a log axis is labeled is a property of the axis, not of a plot, so
the three methods are replaced on `pyqtgraph.AxisItem` itself when the
plotting package is imported (`install`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QFont, QFontMetricsF

#: the label sets tried, densest first, as mantissas within a decade
MANTISSAS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 5), (1, 3), (1,))

#: every how many decades a label goes when one a decade does not fit
DECADE_STEPS = (2, 5, 10, 20, 50, 100)

_SUPERSCRIPT = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')


#: the span labeled in plain digits; powers of ten outside it, so one
#: axis does not mix '0.0001' with '10⁻⁵'
PLAIN = (0.01, 1e4)


def label(value: float) -> str:
    """A log axis's text for `value`: plain digits from 0.01 up to
    10 000 ('0.5', '200'), powers of ten outside ('2·10⁻⁵', '10⁶')."""
    if PLAIN[0] <= abs(value) < PLAIN[1]:
        return f'{value:.4g}'
    mantissa, exponent = f'{value:.2e}'.split('e')
    mantissa = mantissa.rstrip('0').rstrip('.')
    power = '10' + str(int(exponent)).translate(_SUPERSCRIPT)
    return power if mantissa == '1' else f'{mantissa}·{power}'


def choose(low: float, high: float, pixels: float, room: float
           ) -> tuple[tuple[int, ...], int]:
    """The mantissas to label and every how many decades, for a log
    span from `low` to `high` (log10 values) drawn over `pixels`, each
    label needing `room` pixels from the next."""
    per_decade = pixels / max(high - low, 1e-12)
    for mantissas in MANTISSAS:
        marks = np.log10([*mantissas, 10])
        if np.diff(marks).min() * per_decade >= room:
            return mantissas, 1
    for step in DECADE_STEPS:
        if step * per_decade >= room:
            return (1,), step
    return (1,), DECADE_STEPS[-1]


def _in_span(low: float, high: float, mantissas: Sequence[int],
             step: int = 1) -> list[float]:
    """log10 of every `mantissas`·10ⁿ inside (`low`, `high`), n a
    multiple of `step` for the decades themselves."""
    found = []
    for decade in range(int(np.floor(low)), int(np.ceil(high)) + 1):
        for mantissa in mantissas:
            if mantissa == 1 and decade % step:
                continue
            value = decade + np.log10(mantissa)
            if low <= value <= high:
                found.append(float(value))
    return found


def _room(axis: pg.AxisItem, strings: Sequence[str]) -> float:
    """The pixels a label needs along `axis`: a line's height on a
    vertical axis, the widest of `strings` and a space on a horizontal
    one."""
    font = axis.style.get('tickFont') or QFont()
    metrics = QFontMetricsF(font)
    if axis.orientation in ('left', 'right'):
        return 1.2 * metrics.height()
    widest = max((metrics.horizontalAdvance(s) for s in strings), default=0.0)
    return widest + metrics.horizontalAdvance('  ')


def log_tick_values(axis: pg.AxisItem, minVal: float, maxVal: float,
                    size: float, stdTicks: list) -> list:
    """Replaces `AxisItem.logTickValues`: the labeled ticks first, the
    rest unlabeled after them."""
    low, high = sorted((float(minVal), float(maxVal)))
    if not (np.isfinite(low) and np.isfinite(high)) or size <= 0:
        return stdTicks
    candidates = [label(10 ** v) for v in _in_span(low, high, (1, 2, 5))]
    mantissas, step = choose(low, high, size, _room(axis, candidates))
    labeled = _in_span(low, high, mantissas, step)
    if len(labeled) < 2:
        # too narrow a span for round mantissas: round values between
        lo, hi = 10 ** low, 10 ** high
        spacing = axis.tickSpacing(lo, hi, size)[0][0]
        first = np.ceil(lo / spacing) * spacing
        values = np.arange(first, hi + 0.5 * spacing, spacing)
        labeled = [float(np.log10(v)) for v in values if lo <= v <= hi]
    axis._log_labels = {round(v, 9) for v in labeled}
    every = _in_span(low, high, range(1, 10))
    per_decade = size / max(high - low, 1e-12)
    if np.log10(10 / 9) * per_decade < 2.0:
        every = _in_span(low, high, (1,))          # no room for minors
    rest = [v for v in every if round(v, 9) not in axis._log_labels]
    return [(1.0, labeled), (None, rest)]


def log_tick_strings(axis: pg.AxisItem, values: Sequence[float],
                     scale: float, spacing: float) -> list[str]:
    """Replaces `AxisItem.logTickStrings`: the text of the labeled ticks,
    nothing for the rest."""
    chosen = getattr(axis, '_log_labels', None)
    return [label(10 ** v) if chosen is None or round(v, 9) in chosen
            else '' for v in values]


_stock_si_prefix = pg.AxisItem.updateAutoSIPrefix


def update_auto_si_prefix(axis: pg.AxisItem) -> None:
    """Replaces `AxisItem.updateAutoSIPrefix`: no prefix on a log axis,
    whose labels carry their own powers of ten."""
    if axis.logMode:
        axis.autoSIPrefixScale = 1.0
        axis.labelUnitPrefix = ''
        axis._updateLabel()
        return
    _stock_si_prefix(axis)


def install() -> None:
    """Put these on `pyqtgraph.AxisItem`, for every axis in the process."""
    pg.AxisItem.logTickValues = log_tick_values
    pg.AxisItem.logTickStrings = log_tick_strings
    pg.AxisItem.updateAutoSIPrefix = update_auto_si_prefix
