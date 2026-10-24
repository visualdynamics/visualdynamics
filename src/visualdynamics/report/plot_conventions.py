"""Plot conventions for a report's figures (report feature
`plot_conventions`, with its option `shared_decades`).

Three habits a figure in a test report is expected to keep:

* An ASD or an SRS opens on log-log axes. A specification's
  breakpoints are joined by straight lines, and its tolerance band has a
  constant width, only on log-log axes; a narrowband comparison
  otherwise opens on a linear frequency axis.
* Curves are told apart by line style as well as color, and a figure of
  up to `END_LABEL_CURVES` curves names each one at its end, so the
  figure survives a monochrome copy and a reader who does not see color.
* With `shared_decades`, every figure of a kind (the ASD figures of one
  quantity and unit, the SRS figures likewise) opens on the same whole
  decades, so two figures can be compared by eye.

Each acts on the figures already built for the page, so the figure
builders are unchanged and a report without the feature is drawn as it
always was.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..plot import END_LABEL_CURVES, LINE_PATTERNS

#: the figure kinds the log-log axes and the shared decades apply to
KINDS = ('asd', 'srs')


def figure_kind(block: Mapping[str, Any], objects: Mapping[str, Any],
                object_groups: Sequence[Mapping[str, Any]] | None = None
                ) -> str | None:
    """'asd' for a plot of a power spectral density (a measured PSD, a
    specification, or the two compared), 'srs' for a plot of a shock
    response spectrum (with its specification or alone), and None for
    every other figure."""
    from ..core.data import Psd, Srs
    from ..core.report import resolve_binding

    if block.get('kind') != 'plot' or block.get('mode', 'curves') not in (
            'curves', 'pair'):
        return None
    source = objects.get(resolve_binding(block.get('source'), objects,
                                         object_groups) or '')
    if isinstance(source, Srs):
        return 'srs'
    if isinstance(source, Psd):
        return 'asd'
    return None


def _cells(built: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The flat plots a built block draws: itself, or a grid's cells."""
    if built.get('kind') == 'plot':
        return [built]           # type: ignore[list-item]
    if built.get('kind') == 'grid':
        return [cell for row in built['rows'] for column in row['cells']
                for cell in column]
    return []


def apply(payload: dict[str, Any],
          figures: Sequence[tuple[Mapping[str, Any], str | None]],
          shared: bool = False) -> None:
    """Put the page's figures under the conventions, in place.

    `payload` is the page's data, which carries the line patterns and
    the end-label limit for the page to draw with. `figures` is each
    built block with its kind (`figure_kind`). An ASD or SRS figure on
    log ordinates goes to log frequency; a figure already there, or one
    on linear ordinates (a phase, a ratio in decibels), is left on its
    axes. With `shared`, the figures of each kind and unit open on the
    same whole decades (`shared_decades`).
    """
    from . import _into_decades

    payload['line_patterns'] = [list(pattern) for pattern in LINE_PATTERNS]
    payload['end_labels'] = END_LABEL_CURVES
    for built, kind in figures:
        if kind not in KINDS:
            continue
        for cell in _cells(built):
            if cell.get('logy') and not cell.get('logx'):
                _into_decades(cell)
                if cell.get('marks'):
                    # a mode bookmark is a frequency too
                    cell['marks'] = [math.log10(f) for f in cell['marks']
                                     if f > 0]
    if shared:
        share_decades(figures)


def _finite(values: Any) -> list[float]:
    out = []
    for value in values or ():
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            out.append(value)
    return out


def _opening(cell: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    """The window a figure opens on, in its own axis coordinates: its
    `home_x` where it has one (a comparison opens on its specification's
    band) and its data's extent otherwise, and the extent of every curve,
    target, response and limit inside that span. None when the figure
    has nothing finite to draw."""
    grid = cell.get('x') or []
    xs = _finite(grid)
    for curve in cell.get('curves', ()):
        xs += _finite(curve.get('x'))
    if cell.get('home_x'):
        xs = _finite(cell['home_x'])
    if not xs:
        return None
    x0, x1 = min(xs), max(xs)
    ys: list[float] = []

    def inside(values, on):
        if not values:
            return
        for at, value in zip(on, values, strict=False):
            if (at is not None and value is not None
                    and x0 <= at <= x1 and math.isfinite(value)):
                ys.append(float(value))

    for curve in cell.get('curves', ()):
        inside(curve.get('y'), curve.get('x') or grid)
    target_x = (cell.get('curves') or [{}])[0].get('x') or grid
    for channel in cell.get('channels', ()):
        inside(channel.get('y'), target_x)
        inside(channel.get('response'), grid)
        # a shock series' events, all on the measurement's grid
        for response in channel.get('responses', ()):
            inside(response.get('y'), grid)
        for zone in channel.get('zones', ()):
            inside(zone.get('lower'), target_x)
            inside(zone.get('upper'), target_x)
    if not ys:
        return None
    return x0, x1, min(ys), max(ys)


def share_decades(figures: Sequence[tuple[Mapping[str, Any], str | None]]
                  ) -> None:
    """Give every log-log figure of a kind the same opening window, in
    whole decades, in place.

    Figures are grouped by kind and by their axis labels, which carry
    the quantity and the unit: an acceleration ASD in g²/Hz and a force
    ASD are different scales and are never forced onto one. Within a
    group the window is the union of each figure's opening window
    (`_opening`), widened outward to whole decades, at least one decade
    on each axis. A figure with nothing finite takes no part.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for built, kind in figures:
        if kind not in KINDS:
            continue
        for cell in _cells(built):
            if not (cell.get('logx') and cell.get('logy')):
                continue
            key = (kind, str(cell.get('xlabel', '')),
                   str(cell.get('ylabel', '')))
            groups.setdefault(key, []).append(cell)
    for cells in groups.values():
        windows = [window for window in map(_opening, cells)
                   if window is not None]
        if not windows:
            continue
        x0 = math.floor(min(w[0] for w in windows))
        x1 = math.ceil(max(w[1] for w in windows))
        y0 = math.floor(min(w[2] for w in windows))
        y1 = math.ceil(max(w[3] for w in windows))
        home = {'x0': float(x0), 'x1': float(max(x1, x0 + 1)),
                'y0': float(y0), 'y1': float(max(y1, y0 + 1))}
        for cell in cells:
            cell['home'] = dict(home)
