"""A mixed-mode test's data sheet beside the spectrum it was read from.

The reading (`core.mixed_mode.judge_mixed`) puts three kinds of figure
side by side that are misread when quoted alone: the floor's rms with
the tone lines left out, each tone in amplitude, and the total rms of
the whole drive. The view draws the averaged autospectrum on the left
and the sheet on the right, one row for the floor, a row for the tones
together and one for each tone, a row for each narrowband, and the
total, each with its written and measured figure, its tolerance and its
verdict in words.

On the spectrum, each meaning has a line style, a fill pattern or a
word, never a hue alone:

- the reference (the floor with each fixed narrowband on it), wide and
  gray behind the rest, as a requirement is drawn everywhere, with the
  floor's line tolerance dashed either side;
- the residual, the record with the tones taken out, solid: what the
  floor and the bands are judged on;
- the record as recorded, tones in, dotted, with a marker on each fixed
  tone (`plot.tone_readout.ToneOverlay`): a stem to the height its spike
  stands at for this spacing and window, and its amplitude in words;
- the lines left out of the floor for a tone, hatched with diagonal
  lines and labeled "left out", since a floor feature inside them is not
  judged;
- each narrowband's edges, dash-dotted verticals named with the band,
  and a swept band's travel cross-hatched and labeled "swept band".

A failing verdict is set in bold in the sheet and named in the title.
"""

from __future__ import annotations

import html
from typing import Any

import numpy as np

#: the sheet's columns, in order
COLUMNS = ('Part', 'Frequency [Hz]', 'Written', 'Measured', 'Error',
           'Tolerance', 'Verdict')

#: how strongly the hatching of a left-out region is drawn, 0 to 255
HATCH_ALPHA = 110


def _verdict(passed: bool | None) -> str:
    return {True: 'passes', False: 'fails', None: 'not judged'}[passed]


def _signed(value: float, unit: str, places: int = 1) -> str:
    """A signed figure with its unit and a true minus sign; a dash for
    NaN."""
    if not np.isfinite(value):
        return '—'
    # a figure that rounds to nothing has no sign, not a minus zero
    value = 0.0 if round(value, places) == 0.0 else value
    return f'{value:+.{places}f} {unit}'.replace('-', '−')


def scales(dimension: str, unit_system: Any = None
           ) -> tuple[float, float, str]:
    """(amplitude scale, density scale, amplitude unit) that take the
    reading's SI values into `unit_system`'s: g and g²/Hz for an
    acceleration shown in g; ones and no unit for a record whose units
    are not declared."""
    from ..units import DEFAULT_SYSTEM, UNKNOWN

    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    if dimension == UNKNOWN:
        return 1.0, 1.0, ''
    return (float(us.from_si(1.0, dimension)),
            float(us.from_si(1.0, f'{dimension}**2/frequency')),
            us.label_text(dimension))


def sheet_rows(reading: Any, unit_system: Any = None
               ) -> list[tuple[str, ...]]:
    """The data sheet's rows, each a tuple of `COLUMNS` strings, in the
    display units: the floor, the tones together, each tone, each
    narrowband and the total. The total's tolerance says it has none
    and why."""
    scale, _dense, unit = scales(reading.dimension, unit_system)
    rms = f'{unit} rms'.strip()
    peak = f'{unit} peak'.strip()
    floor = reading.floor
    rows = [('Floor, tone and band lines left out', '—',
             f'{floor.reference * scale:.3g} {rms}',
             f'{floor.measured * scale:.3g} {rms}',
             _signed(floor.error, '%'),
             (f'±{floor.rms_percent:g} %, lines ±{floor.db:g} dB '
             f'({floor.outside} of {floor.judged} outside)'),
             _verdict(floor.passed))]
    if reading.tones:
        rows.append((f'Tones, {len(reading.tones)} together', '—',
                     f'{reading.tones_reference * scale:.3g} {rms}',
                     f'{reading.tones_measured * scale:.3g} {rms}',
                     _signed(100.0 * (reading.tones_measured
                                      / reading.tones_reference - 1.0)
                             if reading.tones_reference else float('nan'),
                             '%'),
                     'none: each tone is judged', '—'))
    for tone in reading.tones:
        where = (f'{tone.low:.4g}' if tone.fixed
                 else f'{tone.low:.4g} to {tone.high:.4g}')
        rows.append((f'  {tone.name}', where,
                     f'{tone.reference * scale:.3g} {peak}',
                     f'{tone.measured * scale:.3g} {peak}',
                     _signed(tone.error, '%'),
                     f'±{tone.tolerance:g} %',
                     _verdict(tone.passed)))
    for band in reading.narrowbands:
        where = (f'{band.low:.4g} to {band.high:.4g}' if not band.swept
                 else f'{band.span[0]:.4g} to {band.span[1]:.4g}, swept')
        tolerance = (f'{_signed(band.low_db, "dB")} to '
                     f'{_signed(band.high_db, "dB")} at {band.dof:.0f} dof')
        if band.swept:
            tolerance += (f'; {band.slices} slices of {band.slice_dof:.0f} '
                          f'dof, {100 * band.inside:.0f} % inside')
        rows.append((f'Band {band.name}', where,
                     f'{band.reference * scale:.3g} {rms}',
                     f'{band.measured * scale:.3g} {rms}',
                     _signed(band.error_db, 'dB'), tolerance,
                     _verdict(band.passed if band.slices else None)))
    total = reading.total
    rows.append(('Total, the whole drive', '—',
                 f'{total.reference * scale:.3g} {rms}',
                 f'{total.measured * scale:.3g} {rms}',
                 _signed(100.0 * (total.measured / total.reference - 1.0)
                         if total.reference else float('nan'), '%'),
                 'none: the machine\'s figure', '—'))
    return rows


def sheet_settings(reading: Any, unit_system: Any = None) -> list[str]:
    """The settings that decide what the sheet's numbers mean, a clause
    each: the line spacing and degrees of freedom, the width left out
    and the share of the floor it took, the tracking band, when the
    span read starts, and the residual's rms over every line."""
    scale, _dense, unit = scales(reading.dimension, unit_system)
    floor = reading.floor
    rms = f'{unit} rms'.strip()
    tones = (f'tones read through {reading.tones[0].band}, from '
             f'{reading.start:g} s' if reading.tones
             else f'no tones; read from {reading.start:g} s')
    return [(f'{floor.spacing:.3g} Hz lines, {floor.dof:.0f} degrees of '
            'freedom a line'),
            (f'±{floor.exclusion:g} Hz left out about each tone; tone and '
             f'band lines {100 * floor.left_out:.1f} % of the floor\'s'),
            tones,
            (f'the residual over every line {floor.residual * scale:.4g} '
            f'{rms}')]


def sheet_html(reading: Any, foreground: str, unit_system: Any = None
               ) -> str:
    """The data sheet as rich text, for a `LabelItem` or a `QLabel`: a
    header row, then `sheet_rows`, each failing verdict in bold, and the
    settings under it."""
    def cell(text: str, bold: bool = False, tag: str = 'td') -> str:
        # a tone's name is the controller's text, and a '<' in it would
        # be read as markup
        body = html.escape(text).replace('  ', '&nbsp;&nbsp;')
        body = f'<b>{body}</b>' if bold else body
        return (f'<{tag} style="padding: 1px 8px 1px 0px; '
                f'color: {foreground};">{body}</{tag}>')

    lines = ['<table cellspacing="0">',
             '<tr>' + ''.join(cell(name, True, 'th') for name in COLUMNS)
             + '</tr>']
    for row in sheet_rows(reading, unit_system):
        lines.append('<tr>' + ''.join(
            cell(text, bold=(k == len(COLUMNS) - 1 and text == 'fails'))
            for k, text in enumerate(row)) + '</tr>')
    lines.append('</table>')
    settings = '<br>'.join(html.escape(clause) for clause
                           in sheet_settings(reading, unit_system))
    lines.append(f'<p style="color: {foreground};">{settings}</p>')
    for note in reading.notes:
        lines.append(f'<p style="color: {foreground};">'
                     f'{html.escape(note)}</p>')
    return ''.join(lines)


def _log(value: float) -> float:
    return float(np.log10(max(float(value), 1e-300)))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) index of each run of True in `mask`."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1),
                    np.flatnonzero(edges == -1) - 1))


def draw_mixed_mode(plot: Any, reading: Any, *, unit_system: Any = None,
                    colors: Any = None, legend: bool = True,
                    markers: bool = True) -> list[Any]:
    """Draw a `MixedReading`'s spectra onto a pyqtgraph plot item, on
    log axes, and return the items drawn.

    The reference wide and gray with the floor's tolerance dashed, the
    residual solid, the record dotted, the left-out lines hatched and
    labeled, each narrowband's edges dash-dotted and named, and, with
    `markers`, a tone marker on each fixed tone. With `legend` the
    curves are named for the plot's legend; the panel's small preview
    passes False.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QBrush, QColor

    from ..theme import theme as resolve_theme
    from . import STOOD_BACK_WIDTH, curve_color

    colors = resolve_theme(None) if colors is None else colors
    scale, dense, unit = scales(reading.dimension, unit_system)
    ink = colors['plot_foreground']
    gray = colors['specification_curve']
    f = np.asarray(reading.frequencies, dtype=np.float64)
    keep = f > 0.0
    f = f[keep]

    def named(text):
        return {'name': text} if legend else {}

    def positive(values):
        values = np.asarray(values, dtype=np.float64)[keep] * dense
        return np.where(values > 0.0, values, np.nan)

    plot.setLogMode(x=True, y=True)
    items = []
    reference = positive(reading.reference)
    items.append(plot.plot(f, reference, connect='finite',
                           pen=pg.mkPen(gray, width=STOOD_BACK_WIDTH),
                           **named('reference: floor and fixed bands')))
    factor = 10.0 ** (reading.floor.db / 10.0)
    for k, edge in enumerate((reference * factor, reference / factor)):
        items.append(plot.plot(
            f, edge, connect='finite',
            pen=pg.mkPen(gray, width=1, style=Qt.PenStyle.DashLine),
            **(named(f'±{reading.floor.db:g} dB on the floor')
               if k == 0 else {})))
    # the record as recorded, where tones were taken out of it; without
    # them it is the residual, drawn once
    if reading.tones:
        items.append(plot.plot(
            f, positive(reading.composite), connect='finite',
            pen=pg.mkPen(curve_color(1, colors), width=1,
                         style=Qt.PenStyle.DotLine),
            **named('record, tones in')))
    items.append(plot.plot(
        f, positive(reading.residual), connect='finite',
        pen=pg.mkPen(curve_color(0, colors), width=1.5),
        **named('residual, tones taken out')))
    shown = positive(reading.composite)
    shown = shown[np.isfinite(shown)]
    top = float(np.max(shown)) if shown.size else 1.0
    written = reference[np.isfinite(reference)]
    bottom = float(np.min(written)) / factor if written.size else top

    # the lines left out of the floor, hatched: a pattern, not a hue
    hatch = QColor(ink)
    hatch.setAlpha(HATCH_ALPHA)
    left = np.asarray(reading.left_out, dtype=bool)[keep]
    spacing = float(reading.floor.spacing)
    for first, last in _runs(left):
        low, high = f[first] - 0.5 * spacing, f[last] + 0.5 * spacing
        region = pg.LinearRegionItem(
            values=(_log(max(low, f[0])), _log(high)), movable=False,
            brush=QBrush(hatch, Qt.BrushStyle.BDiagPattern),
            pen=pg.mkPen(hatch, width=1))
        region.setZValue(-15)
        region.left_out = (low, high)
        plot.addItem(region, ignoreBounds=True)
        items.append(region)
        label = pg.TextItem('left out', color=ink, anchor=(0.0, 0.5),
                            angle=90)
        label.setPos(_log(0.5 * (low + high)), _log(bottom) - 0.15)
        plot.addItem(label, ignoreBounds=True)
        items.append(label)

    # each narrowband's edges, and a swept band's travel
    for band in reading.narrowbands:
        edges = (band.low, band.high) if not band.swept else band.span
        style = pg.mkPen(ink, width=1, style=Qt.PenStyle.DashDotLine)
        for k, edge in enumerate(edges):
            line = pg.InfiniteLine(
                pos=_log(edge), angle=90, pen=style,
                label=(f'{band.name} edges' if k == 0 else None),
                labelOpts={'position': 0.92, 'color': ink, 'rotateAxis':
                           (1, 0)})
            line.band_edge = edge
            plot.addItem(line, ignoreBounds=True)
            items.append(line)
        if band.swept:
            region = pg.LinearRegionItem(
                values=(_log(edges[0]), _log(edges[1])), movable=False,
                brush=QBrush(hatch, Qt.BrushStyle.DiagCrossPattern),
                pen=pg.mkPen(hatch, width=1))
            region.setZValue(-16)
            plot.addItem(region, ignoreBounds=True)
            items.append(region)
            label = pg.TextItem('swept band', color=ink, anchor=(0.0, 1.0))
            label.setPos(_log(edges[0]), _log(top) + 1.6)
            plot.addItem(label, ignoreBounds=True)
            items.append(label)

    if markers:
        items.extend(_tone_markers(plot, reading, ink, scale, dense, unit))
    return items


def _tone_markers(plot: Any, reading: Any, ink: str, scale: float,
                  dense: float, unit: str) -> list[Any]:
    """A `ToneOverlay` on the record's spectrum at each fixed tone: the
    stem to the height its spike stands at, and its amplitude read off
    the spectrum, beside the tracked reading in the sheet. A tone that
    cannot be read off the spectrum (at its end) has no marker."""
    from ..core.tone_readout import read_tone
    from .tone_readout import ToneOverlay

    items = []
    placed = 0
    for tone in reading.tones:
        if not tone.fixed:
            continue
        try:
            found = read_tone(reading.frequencies, reading.composite,
                              tone.low, window=reading.averaging,
                              snap=False, warn=False)
        except ValueError:
            continue
        overlay = ToneOverlay(plot, ink, above=True)
        overlay.show(found, reading.frequencies, scale=scale,
                     density_scale=dense, unit=unit)
        # every other tone's words a decade higher, so neighbors' do
        # not write over each other
        if placed % 2:
            where = overlay.label.pos()
            overlay.label.setPos(where.x(), where.y() + 1.0)
        placed += 1
        overlay.tone = tone.name
        items.extend(overlay.items())
    return items


def build_mixed_mode(layout: Any, reading: Any, *, unit_system: Any = None,
                     theme: Any = None) -> Any:
    """Draw a mixed reading into a graphics layout: the spectrum with its
    legend below on the left, the data sheet on the right; return the
    spectrum's plot item.

    The title is the reading in one line (`MixedReading.describe`), the
    floor, the tones and the total each labeled, and what failed.
    """
    import pyqtgraph as pg

    from ..theme import theme as resolve_theme
    from ..units import DEFAULT_SYSTEM
    from . import axis_label, legend_below

    colors = resolve_theme(theme)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    scale, dense, unit = scales(reading.dimension, us)
    plot = layout.addPlot(row=0, col=0)
    plot.showGrid(x=True, y=True, alpha=0.2)
    for side in ('left', 'bottom'):
        plot.getAxis(side).enableAutoSIPrefix(False)
    legend_below(layout, plot, 0, colors)
    plot.setTitle(html.escape(reading.describe(unit, scale)),
                  color=colors['plot_foreground'], size='9pt')
    draw_mixed_mode(plot, reading, unit_system=us, colors=colors)
    plot.setLabel('bottom', 'frequency [Hz]')
    plot.setLabel('left', axis_label(f'{reading.dimension}**2/frequency',
                                     us))
    f = np.asarray(reading.frequencies, dtype=np.float64)
    g = np.asarray(reading.composite, dtype=np.float64) * dense
    w = np.asarray(reading.reference, dtype=np.float64) * dense
    written = (f > 0.0) & np.isfinite(w) & (w > 0.0)
    if written.any():
        # the floor's own band and a margin, so a roll-off past it does
        # not take the picture; the words stand over the tallest spike
        low, high = f[written][0], f[written][-1]
        plot.setXRange(np.log10(low) - 0.3, np.log10(high) + 0.15,
                       padding=0.0)
        inside = written & np.isfinite(g) & (g > 0.0)
        peak = g[inside].max() if inside.any() else w[written].max()
        plot.setYRange(np.log10(w[written].min()) - 1.0,
                       np.log10(peak) + 2.4, padding=0.0)
    sheet = pg.LabelItem(justify='left', color=colors['plot_foreground'],
                         size='8pt')
    sheet.setText(sheet_html(reading, colors['plot_foreground'], us))
    sheet.is_data_sheet = True
    layout.addItem(sheet, row=0, col=1)
    return plot


__all__ = ['COLUMNS', 'build_mixed_mode', 'draw_mixed_mode', 'scales',
           'sheet_html', 'sheet_rows', 'sheet_settings']
