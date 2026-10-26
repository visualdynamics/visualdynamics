"""The extreme response and fatigue damage spectra of a record, against a
candidate test when there is one, and the time-compression rule they
are read through.

Three plots, one above the other (`core.fatigue`):

- the ERS on log axes, natural frequency across: the largest response
  each resonator is expected to reach (a spectrum) or reached (a time
  history), in the unit system's acceleration unit;
- the FDS on the same frequency axis, linked to the first: the damage
  each resonator's response does over the duration, in that unit to the
  power b, the S-N constant at one;
- the time-compression rule at the result's b: the time ratio a test
  stands for against its exaggeration factor on the ASD, a straight
  line on log axes, with the cap on the factor as a dashed wall and the
  time ratio the cap allows written on it. With a test, the
  exaggeration the test carries at each natural frequency is drawn as a
  bar at the field's time over the test's, from its least (a hollow
  square, the governing frequency) to its most (a filled square), each
  end labeled with its frequency. Where the bar lies right of the line
  the test does more damage than the field there; left of it, less;
  past the wall, the test is exaggerated beyond the cap.

The field is a solid line and the test a dashed one, named in the
legend, never told apart by color alone. What each result rests on is
in the plots' titles: Q and the duration on the ERS, b and the reading
of the cycles on the FDS, and on the third the comparison's verdict.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _reading_plot

#: the bar's end markers' size, pixels
MARKER_SIZE = 10

#: the exaggeration factors the rule's line is drawn over, on the ASD
FACTORS = (0.5, 8.0)


def _shown(values: Any, scale: float, power: float = 1.0) -> np.ndarray:
    """SI values in the display unit (to `power`), with what has no place
    on a log axis left out."""
    values = np.asarray(values, dtype=float) / scale ** power
    return np.where(np.isfinite(values) & (values > 0.0), values, np.nan)


def _seconds(value: float) -> str:
    from ..core.fatigue import _seconds as said

    return said(value)


def title_ers(spectra: Any) -> str:
    """The ERS plot's title: what the largest response is, at what Q,
    over how long."""
    if spectra.source == 'time history':
        how = f"the largest in the record's {_seconds(spectra.ers_duration)}"
    else:
        how = f'expected over {_seconds(spectra.ers_duration)}'
    return f'extreme response: Q = {spectra.q:g}, {how}'


def title_fds(spectra: Any) -> str:
    """The FDS plot's title: b, the reading of the cycles and the
    duration."""
    how = ('rainflow' if spectra.source == 'time history'
           else f'{spectra.method} reading')
    return (f'fatigue damage: b = {spectra.b:g}, {how}, over '
            f'{_seconds(spectra.duration)}, C = 1')


def build_fatigue(layout: Any, field: Any, test: Any = None, *,
                  pair: Any = None, unit_system: Any = None,
                  theme: Any = None, field_name: str | None = None,
                  test_name: str | None = None) -> list[Any]:
    """Draw `field`, a `core.fatigue.ResponseSpectra`, and `test` beside
    it when given, into `layout`; return the three plots.

    `pair` is their `core.fatigue.Equivalence`, made here when `test` is
    given and it is not. `field_name` and `test_name` are the legend's
    words, the records' labels when omitted.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..core.fatigue import CAP, compression_ratio, equivalence
    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import axis_label, curve_color, legend_below
    from .scalogram import _decade_ticks

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    ink = colors['plot_foreground']
    scale, _offset = us.transform(field.dimension)
    if test is not None and pair is None:
        pair = equivalence(field, test)
    sides = [(field, field_name or field.label or 'field',
              pg.mkPen(curve_color(0, colors), width=2.5))]
    if test is not None:
        sides.append((test, test_name or test.label or 'test',
                      pg.mkPen(curve_color(1, colors), width=2.0,
                               style=Qt.PenStyle.DashLine)))

    # plots in the even rows, the legend in the odd row under the FDS
    # (`legend_below` places it at 2 * row + 1)
    ers_plot = _reading_plot(layout, 0)
    fds_plot = _reading_plot(layout, 2)
    fds_plot.setXLink(ers_plot)
    legend_below(layout, fds_plot, 1, colors)
    rule_plot = _reading_plot(layout, 4)
    unit = us.label_html(field.dimension)
    for plot in (ers_plot, fds_plot, rule_plot):
        plot.setLogMode(x=True, y=True)
    for spectra, name, pen in sides:
        f = spectra.frequency
        ers_plot.plot(f, _shown(spectra.ers, scale), pen=pen)
        fds_plot.plot(f, _shown(spectra.fds, scale, spectra.b), pen=pen,
                      name=f'{name}: {spectra.source}')
    ticks = [_decade_ticks(np.array([field.frequency[0],
                                     field.frequency[-1]]))]
    for plot in (ers_plot, fds_plot):
        plot.getAxis('bottom').setTicks(ticks)
    ers_plot.setLabel('left', axis_label(field.dimension, us))
    fds_plot.setLabel('left', f'damage [({unit})<sup>{field.b:g}</sup>]')
    fds_plot.setLabel('bottom', 'natural frequency [Hz]')
    ers_plot.setTitle(title_ers(field), color=ink, size='9pt')
    fds_plot.setTitle(title_fds(field), color=ink, size='9pt')

    # ---- the rule, its cap, and where the test lies on it --------------
    cap = pair.cap if pair is not None else CAP['asd']
    factors = np.geomspace(*FACTORS, 64)
    ratios = np.array([compression_ratio(x, field.b) for x in factors])
    rule_plot.plot(factors, ratios, pen=pg.mkPen(ink, width=1.5),
                   name=None)
    allowed = compression_ratio(cap, field.b)
    wall = pg.mkPen(colors['limit_abort'], width=1.5,
                    style=Qt.PenStyle.DashLine)
    rule_plot.addItem(pg.InfiniteLine(
        pos=float(np.log10(cap)), angle=90, pen=wall, movable=False,
        label=f'cap {cap:g} on the ASD: {allowed:.4g} to 1',
        labelOpts={'position': 0.92, 'color': ink,
                   'anchors': [(1, 0), (1, 0)]}), ignoreBounds=True)
    # under the line's left end, clear of the cap's words at the top
    words = pg.TextItem(f'equal damage at b = {field.b:g}', color=ink,
                        anchor=(0.0, -0.1))
    words.setPos(float(np.log10(factors[2])), float(np.log10(ratios[2])))
    rule_plot.addItem(words, ignoreBounds=True)
    if pair is not None:
        exaggeration = np.asarray(pair.exaggeration, dtype=float)
        judged = np.isfinite(exaggeration) & (exaggeration > 0.0)
        if judged.any():
            low = int(np.argmin(np.where(judged, exaggeration, np.inf)))
            high = int(np.argmax(np.where(judged, exaggeration, -np.inf)))
            y = pair.time_ratio
            rule_plot.plot([exaggeration[low], exaggeration[high]], [y, y],
                           pen=pg.mkPen(curve_color(1, colors), width=3))
            for index, brush, side in ((low, pg.mkBrush(0, 0, 0, 0), 1.0),
                                       (high, pg.mkBrush(ink), 0.0)):
                rule_plot.plot([exaggeration[index]], [y], pen=None,
                               symbol='s', symbolSize=MARKER_SIZE,
                               symbolPen=pg.mkPen(ink), symbolBrush=brush)
                said = pg.TextItem(
                    f'{exaggeration[index]:.3g} at '
                    f'{pair.frequency[index]:.4g} Hz', color=ink,
                    anchor=(side, -0.2))
                said.setPos(float(np.log10(exaggeration[index])),
                            float(np.log10(y)))
                rule_plot.addItem(said, ignoreBounds=True)
        rule_plot.setTitle(pair.describe(), color=ink, size='9pt')
    else:
        rule_plot.setTitle(f'time compression at b = {field.b:g}',
                           color=ink, size='9pt')
    rule_plot.setLabel('left', 'service time over test time')
    rule_plot.setLabel('bottom', 'exaggeration factor on the ASD')
    return [ers_plot, fds_plot, rule_plot]


def summary(field: Any, pair: Any = None) -> str:
    """The reading in one line, for the status bar."""
    if pair is not None:
        return f'{field.label}: {pair.describe()}'
    return field.describe()
