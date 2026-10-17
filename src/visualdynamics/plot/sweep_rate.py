"""A sine sweep's path through the modes it passes, with the time and
the cycles it spends in each one's half-power band.

The sweep is drawn as what it is, frequency against time: time across,
frequency up on a log axis, so a logarithmic sweep is a straight line
and a linear one bends over. Each mode's half-power band is a strip
across the plot at its frequency, and where the sweep crosses the strip
its path is drawn heavy, with the time and the cycles it spent there
written beside it against the Q the mode needs to build, and the rate
as a multiple of the rule's (`core.sweep_rate.check_sweep_rate`).

The question is which modes the sweep rushes, and a log sweep answers
it at a glance: every crossing of a strip of the same relative width
takes the same time, so a rushed mode is one whose strip is narrow (a
high Q) or low (few cycles in that time).

**A rushed mode is told by line and marker as well as color.** Its
crossing is dashed and marked with a cross; a mode the sweep lets build
is solid and marked with a circle, and the label ends in the words
"too fast" or "builds". A reader who cannot tell the two colors apart
reads the same verdict from the shapes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..core.sweep_rate import MARGIN, SweepRateCheck

#: the markers on a crossing: a circle where the mode builds, a cross
#: where the sweep is too fast for it
SYMBOLS = {False: 'o', True: 'x'}

#: the crossing's line width, pixels: heavy against the sweep's 1.5
CROSSING_WIDTH = 4.0

#: the band strips' fill, as an alpha over the theme color
BAND_ALPHA = 45


def label(check: SweepRateCheck) -> str:
    """What is written beside a crossing: '22 Hz, Q 40: 0.54 s,
    12 cycles of 40; 3.4 × rule, too fast'."""
    return (f'{check.frequency:.4g} Hz, Q {check.q:.3g}: '
            f'{check.seconds:.2g} s, {check.cycles:.3g} cycles of '
            f'{check.q:.3g}; {check.ratio:.2g} × rule, {check.verdict}')


def _decade_ticks(low: float, high: float) -> list[tuple[float, str]]:
    """(position, text) up the log frequency axis, at 1, 2 and 5 per
    decade, as the scalogram labels its axis."""
    from ..core import wavelet

    return [(float(np.log10(value)),
             f'{value:g}' if value >= 1.0 else f'{value:.3g}')
            for value in wavelet.decade_values(low, high)]


def build_sweep_rate(layout: Any, specification: Any,
                     modes: Iterable[tuple[float, float]] | Any, *,
                     tone: str | None = None, margin: float = MARGIN,
                     theme: Any = None) -> list[SweepRateCheck]:
    """Draw one tone's sweep through the modes' half-power bands and
    return its checks (`core.sweep_rate.check_sweep_rate`).

    `tone` names the tone drawn, the specification's first by default;
    `modes` and `margin` are as `check_sweep_rate` takes them. A mode
    the tone never reaches gets no strip, since there is nothing of the
    sweep to mark on it.
    """
    import pyqtgraph as pg

    from ..core.sine import SineTone
    from ..core.sweep_rate import check_sweep_rate, half_power_band
    from ..theme import theme as resolve_theme
    from . import legend_below

    chosen = (specification if isinstance(specification, SineTone)
              else specification.tone(tone if tone is not None
                                      else specification.tones[0].name))
    checks = check_sweep_rate(chosen, modes, margin=margin)
    colors = resolve_theme(theme)

    plot = layout.addPlot(row=0, col=0)
    plot.showGrid(x=True, y=True, alpha=0.2)
    legend = legend_below(layout, plot, 0, colors)
    duration = chosen.duration()
    t, f = chosen.trajectory(duration / 2000.0)
    plot.plot(t, np.log10(f), pen=pg.mkPen(colors['response_curve'],
                                           width=1.5))
    low, high = float(np.min(f)), float(np.max(f))

    def tinted(key):
        color = pg.mkColor(colors[key])
        color.setAlpha(BAND_ALPHA)
        return color

    banded = set()
    # a strip is tinted for its mode's worst pass, not its first: a
    # sweep up slowly and back fast was tinted as building (2026-10-16)
    rushed = {(check.frequency, check.q) for check in checks
              if check.too_fast}
    for check in checks:
        fast = check.too_fast
        key = (check.frequency, check.q)
        if key not in banded:
            banded.add(key)
            lower, upper = half_power_band(check.frequency, check.q)
            strip = pg.LinearRegionItem(
                values=(np.log10(lower), np.log10(upper)),
                orientation='horizontal', movable=False,
                brush=pg.mkBrush(tinted('limit_warning' if key in rushed
                                        else 'specification_curve')),
                pen=pg.mkPen(None))
            strip.setZValue(-10)
            strip.is_band = True
            strip.too_fast = key in rushed
            plot.addItem(strip)
            low, high = min(low, lower), max(high, upper)
        color = colors['limit_abort'] if fast else colors['response_curve']
        style = (pg.QtCore.Qt.PenStyle.DashLine if fast
                 else pg.QtCore.Qt.PenStyle.SolidLine)
        inside = np.linspace(check.start, check.start + check.seconds, 60)
        crossing = plot.plot(
            inside, np.log10(chosen.frequency_at(inside)),
            pen=pg.mkPen(color, width=CROSSING_WIDTH, style=style))
        crossing.is_crossing = True
        crossing.too_fast = fast
        middle = check.start + check.seconds / 2.0
        mark = plot.plot([middle], [np.log10(chosen.frequency_at(middle))],
                         pen=None, symbol=SYMBOLS[fast], symbolSize=11,
                         symbolPen=pg.mkPen(color, width=2),
                         symbolBrush=pg.mkBrush(0, 0, 0, 0))
        mark.is_mode_mark = True

    # one label per mode, at the left over its strip, for the shortest
    # of its passes: a sweep up and back passes each mode twice, and two
    # labels at the two crossings ran into each other and off the plot
    by_mode: dict[tuple[float, float], list[SweepRateCheck]] = {}
    for check in checks:
        by_mode.setdefault((check.frequency, check.q), []).append(check)
    for passes in by_mode.values():
        worst = min(passes, key=lambda check: check.seconds)
        words = label(worst)
        if len(passes) > 1:
            words += f' (the shortest of {len(passes)} passes)'
        text = pg.TextItem(words, color=colors['plot_foreground'],
                           anchor=(0.0, 1.0))
        text.setPos(0.01 * duration,
                    float(np.log10(half_power_band(worst.frequency,
                                                   worst.q)[1])))
        text.is_check_label = True
        plot.addItem(text)

    for fast, name in ((False, 'builds: solid, circle'),
                       (True, 'too fast: dashed, cross')):
        if any(check.too_fast == fast for check in checks):
            color = colors['limit_abort'] if fast else colors['response_curve']
            style = (pg.QtCore.Qt.PenStyle.DashLine if fast
                     else pg.QtCore.Qt.PenStyle.SolidLine)
            legend.addItem(pg.PlotDataItem(
                [], [], pen=pg.mkPen(color, width=CROSSING_WIDTH, style=style),
                symbol=SYMBOLS[fast], symbolSize=11,
                symbolPen=pg.mkPen(color, width=2),
                symbolBrush=pg.mkBrush(0, 0, 0, 0)), name)
    if checks:
        swatch = pg.BarGraphItem(x=[0.0], height=[1.0], width=1.0, pen=None,
                                 brush=pg.mkBrush(tinted('specification_curve')))
        legend.addItem(swatch, 'half-power band, fn/Q wide')

    plot.getAxis('left').setTicks([_decade_ticks(low, high)])
    margin_decades = 0.04 * max(np.log10(high / low), 0.1)
    plot.setYRange(np.log10(low) - margin_decades,
                   np.log10(high) + margin_decades, padding=0)
    plot.setXRange(0.0, duration, padding=0.02)
    plot.setLabel('bottom', 'time from the tone’s start [s]')
    plot.setLabel('left', 'frequency [Hz]')
    flagged = sum(check.too_fast for check in checks)
    plot.setTitle(f'{chosen.name}: {len(checks)} pass'
                  f'{"es" * (len(checks) != 1)} through a mode, '
                  f'{flagged} too fast', color=colors['plot_foreground'],
                  size='9pt')
    return checks
