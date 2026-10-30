"""Dragging a specification's bands on the plot.

A specification's warning and abort bands are drawn on the plot as
shaded zones. While its sheet is open they can also be set there. Each
of the four edges (warning lower and upper, abort lower and upper) has
one handle for each run of segments that share a band, for the channel
drawn. Dragging a handle moves that edge in steps of `STEP` decibels,
and the change is applied when the drag ends, to every selected channel
in that frequency range.

A handle knows nothing of channels or constraints. It reports
(kind, side, segments, decibels), the level the edge was released at,
and whatever opened the sheet applies it through the draft's
`with_band_edge`, which applies each channel's symmetric and uniform
settings.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..core.author import LIMIT_STEP_DB

# Brandon, 2026-09-06: the only place the bands are shown is the plot,
# the edit applies to every selected channel in that frequency range,
# and it lands when the drag stops.

#: how wide the grab is, in pixels, either side of an edge
GRAB = 6
def snap_db(value_db: float, dy: float) -> float:
    """The level in decibels that a drag of `dy` decades from `value_db`
    reaches: ten decibels per decade, rounded to the nearest multiple of
    `STEP`."""
    return float(np.round((float(value_db) + 10.0 * dy) / STEP) * STEP)


#: the grid a dragged limit lands on, in decibels — the sheet's own
#: (`core.author.LIMIT_STEP_DB`), so a drag and a typed step agree
STEP = LIMIT_STEP_DB


def band_edges(draft: Any, channel: int, unit_system: Any
               ) -> dict[tuple[str, str], list[tuple[list[int], np.ndarray, np.ndarray]]]:
    """The four band edges of one channel, in display units.

    Returns a dict `(kind, side) -> [(segments, frequencies, levels)]`
    with one piece per run of segments that share a band. Each piece
    covers its own span from breakpoint to breakpoint, so where the band
    steps at a breakpoint two pieces meet at one frequency at two
    levels, the same as the shading. Between breakpoints a piece follows
    the power law the sheet's specification is read by, filled in as
    the shading is (`plot.as_power_law`), so on a linear frequency axis
    the edge stays on its zone instead of cutting straight across.
    """
    # Straight between breakpoints, the handles drifted off their zones
    # on a linear frequency axis (Brandon, 2026-10-29); the sheet writes
    # a log-log specification, so the law is the one the shading draws
    from . import as_power_law

    # Each piece runs corner to corner (Brandon, 2026-09-06). Taken at
    # the breakpoints instead, a corner belonged to the section on its
    # right and the left section's line slanted to its neighbor's level.
    f = np.asarray(draft.frequencies, dtype=float)
    density = f'{draft.dims[channel]}**2/frequency'
    target = np.asarray(unit_system.from_si(
        np.asarray(draft.levels[channel], dtype=float), density), dtype=float)
    edges: dict[tuple[str, str], list] = {}
    for kind in ('warning', 'abort'):
        for run in draft.band_runs(channel, kind):
            first, last = run[0], run[-1] + 1
            below, above = draft.bands[channel][kind][run[0]]
            x, span = as_power_law(f[first:last + 1], target[first:last + 1])
            edges.setdefault((kind, 'lower'), []).append(
                (list(run), x, span * 10 ** (below / 10.0)))
            edges.setdefault((kind, 'upper'), []).append(
                (list(run), x, span * 10 ** (above / 10.0)))
    return edges


def add_band_handles(plot: Any, draft: Any, channel: int, unit_system: Any,
                     colors: Any,
                     on_release: Callable[[str, str, list[int], float], None],
                     on_menu: Callable[[Any], None] | None = None
                     ) -> list[Any]:
    """Add a draggable handle for each band edge of the drawn channel.

    There is one handle per edge for each run of segments that share a
    band, with a label giving its level in decibels. When a drag ends,
    `on_release(kind, side, segments, decibels)` is called with the
    level reached; a drag on a linear ordinate changes nothing. A
    right-click calls `on_menu` with the screen position, when given.

    Returns the handles added.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor

    box = plot.getViewBox()
    log_x, log_y = box.state['logMode']
    handles = []
    edges = band_edges(draft, channel, unit_system)
    for (kind, side), pieces in edges.items():
        for run, x, y in pieces:
            below, above = draft.bands[channel][kind][run[0]]
            color = QColor(colors['limit_warning' if kind == 'warning'
                                   else 'exceed_over'])
            color.setAlpha(160)
            handle = BandHandle(
                np.log10(x) if log_x else x, np.log10(y) if log_y else y,
                kind, side, list(run), log_y,
                below if side == 'lower' else above,
                on_release, on_menu,
                pen=pg.mkPen(color, width=3,
                             style=Qt.PenStyle.DashLine))
            handle.setZValue(30)
            plot.addItem(handle, ignoreBounds=True)
            # the value, said on the plot: at the run's start, riding
            # with the handle (Brandon, 2026-09-06: the dB levels were
            # shown nowhere)
            label = pg.TextItem(handle.said(), color=color,
                                anchor=(0.0, 1.0 if side == 'upper' else 0.0))
            label.setZValue(31)
            label.setPos(float(handle.xData[0]), float(handle.yData[0]))
            plot.addItem(label, ignoreBounds=True)
            handle.label = label
            handles.append(handle)
    return handles


def _handle_base():
    import pyqtgraph as pg

    class BandHandle(pg.PlotCurveItem):
        """One edge of one band over a run of segments; drag it up or
        down. Reports the drag in decibels when the button is let go,
        and moves nothing itself until then."""

        def __init__(self, x, y, kind, side, segments, log_y, value_db,
                     on_release, on_menu, **kwargs):
            super().__init__(x, y, clickable=True, **kwargs)
            self.kind: str = kind
            self.side: str = side
            self.segments: list[int] = segments
            self.log_y: bool = log_y
            #: the edge's decibels as drawn — what a drag moves from
            self.value_db: float = float(value_db)
            self.on_release: Callable[[str, str, list[int], float], None] = \
                on_release
            self.on_menu: Callable[[Any], None] | None = on_menu
            self.label: Any = None
            self.setAcceptHoverEvents(True)
            # the grab: a stroke this many pixels wide either side
            self.setClickable(True, width=GRAB * 2)

        def said(self, value: float | None = None) -> str:
            value = self.value_db if value is None else value
            return f'{value:+g} dB'.replace('-', '−')

        def snapped(self, dy: float) -> float:
            """The decibels the drag has reached, on the `STEP` grid the
            edge lands on: the edge's value plus the decades dragged
            times ten, rounded (`snap_db`)."""
            return snap_db(self.value_db, dy)

        def hoverEvent(self, ev):
            from PySide6.QtCore import Qt

            if ev.isExit():
                self.unsetCursor()
            elif self.mouseShape().contains(ev.pos()):
                self.setCursor(Qt.CursorShape.SizeVerCursor)

        def mouseClickEvent(self, ev):
            from PySide6.QtCore import Qt

            if ev.button() == Qt.MouseButton.RightButton and self.on_menu:
                ev.accept()
                self.on_menu(ev.screenPos())

        def mouseDragEvent(self, ev):
            from PySide6.QtCore import Qt

            if ev.button() != Qt.MouseButton.LeftButton:
                return
            if ev.isStart() and not self.mouseShape().contains(ev.buttonDownPos()):
                return
            ev.accept()
            dy = ev.pos().y() - ev.buttonDownPos().y()
            if ev.isFinish():
                self.setPos(0, 0)
                self.release(dy)
            else:
                # ride with the cursor, but only to the next `STEP`
                # of decibels, and say which
                reached = self.snapped(dy) if self.log_y else self.value_db
                self.setPos(0, (reached - self.value_db) / 10.0)
                if self.label is not None:
                    self.label.setText(self.said(reached))
                    self.label.setPos(float(self.xData[0]),
                                      float(self.yData[0])
                                      + (reached - self.value_db) / 10.0)

        def release(self, dy: float) -> None:
            """The drag landed `dy` up in the plot's own y units —
            decades on a log axis, so ten times that in decibels,
            snapped to the `STEP` grid; on a linear axis there is no
            honest decibel and nothing moves."""
            if not self.log_y:
                return
            reached = self.snapped(dy)
            if reached == self.value_db:
                return
            # the level reached, not the distance moved: every channel
            # under the sheet lands on it (Brandon, 2026-09-06 — a
            # channel at 3 dB and one at 4, dragged to 4, both at 4)
            self.on_release(self.kind, self.side, list(self.segments), reached)

    return BandHandle


BandHandle = _handle_base()
