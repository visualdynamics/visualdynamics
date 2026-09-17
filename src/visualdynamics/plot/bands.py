"""The bands of a specification, dragged on the plot.

A specification's warning and abort bands are read off the plot as
shaded zones (`plot._shade_limit_zones`); while its sheet is open they
are also *set* there. Each edge of each band — warning below and
above, abort below and above — carries one handle per run of segments
sharing a band, for the channel drawn; dragging a handle moves that
edge by the decibels dragged, and lands when the drag ends (Brandon,
2026-09-06: the only place the bands are shown is the plot, the edit
applies to every selected channel in that frequency range, and it
lands when the drag stops). The handle knows nothing of channels or
constraints: it reports (kind, side, segments, decibels) — the level
the edge was let go at — and whoever opened the sheet lands it
through the draft's `with_band_edge`, which puts every channel there
under its own symmetric and uniform rules.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

#: how wide the grab is, in pixels, either side of an edge
GRAB = 6
#: the decibel grid a drag lands on
STEP = 1.0


def band_edges(draft: Any, channel: int, unit_system: Any
               ) -> dict[tuple[str, str], list[tuple[list[int], np.ndarray, np.ndarray]]]:
    """The four band edges of one channel in display units, one piece
    per run of segments sharing a band: `(kind, side) -> [(segments,
    frequencies, levels)]`. Each piece is its own band over its own
    span, corner to corner — so where the band steps at a breakpoint
    two pieces meet at one frequency at two levels, exactly as the
    shading does. Taken at the breakpoints instead, a corner belongs
    to the section on its right and the left section's line slanted
    to its neighbor's level (Brandon, 2026-09-06)."""
    f = np.asarray(draft.frequencies, dtype=float)
    density = f'{draft.dims[channel]}**2/frequency'
    target = np.asarray(unit_system.from_si(
        np.asarray(draft.levels[channel], dtype=float), density), dtype=float)
    edges: dict[tuple[str, str], list] = {}
    for kind in ('warning', 'abort'):
        for run in draft.band_runs(channel, kind):
            first, last = run[0], run[-1] + 1
            below, above = draft.bands[channel][kind][run[0]]
            span = target[first:last + 1]
            edges.setdefault((kind, 'lower'), []).append(
                (list(run), f[first:last + 1], span * 10 ** (below / 10.0)))
            edges.setdefault((kind, 'upper'), []).append(
                (list(run), f[first:last + 1], span * 10 ** (above / 10.0)))
    return edges


def add_band_handles(plot: Any, draft: Any, channel: int, unit_system: Any,
                     colors: Any,
                     on_release: Callable[[str, str, list[int], float], None],
                     on_menu: Callable[[Any], None] | None = None
                     ) -> list[Any]:
    """One draggable handle per run of segments sharing a band, per
    edge, for the drawn channel. Returns the handles added."""
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
            self.band_handle: bool = True
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
            return f'{value:+.0f} dB'.replace('-', '−')

        def snapped(self, dy: float) -> float:
            """The decibels the drag has reached, on the whole-decibel
            grid the edge lands on: the edge's value plus the decades
            dragged times ten, rounded."""
            return float(np.round((self.value_db + 10.0 * dy) / STEP) * STEP)

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
                # ride with the cursor, but only to the next whole
                # decibel, and say which
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
            snapped to the whole decibel; on a linear axis there is no
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
