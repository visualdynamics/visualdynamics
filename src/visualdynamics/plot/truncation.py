"""A truncation drawn on the time history it will be applied to.

A start and a stop time alone do not show whether the kept span covers
the part of the record that was intended, so the span is drawn on the
plot. The discarded ends are shaded gray, and the kept span can be
dragged the same way as the averaging region: by an edge to move that
time, or by the middle to slide the whole span along the record.

During a drag only the shading moves. `TruncationOverlay.changed` is
emitted once, when the mouse is released.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor

from ..core.truncate import Truncation

# Gray because cut-away data is reference, not the subject. The drag
# settles before it is said (the filter corner's lesson, 2026-08-28):
# the shading follows the mouse move by move, which is cheap (four
# rectangle corners), and `changed` is emitted once, at release,
# because whoever listens stores the span, re-reads staleness and
# re-settles the report.

#: how solid the grayed-out ends are. Enough to read as "going away",
#: light enough that the trace under them can still be judged — the
#: decision of where to cut is made by looking at exactly that data
SHADE_ALPHA = 46

#: the least span a drag can leave, as a share of the record — the
#: region's own guard; the panel's boxes enforce the finer
#: one-sample rule when the number is typed
LEAST_SHARE = 1e-3


class TruncationOverlay(QObject):
    """A truncation drawn as marks on a time history plot.

    The overlay owns every item it adds and removes all of them in
    `remove`, so a redraw cannot leave shading behind. A drag that
    leaves a span shorter than a thousandth of the record is undone.
    Any other finished drag is reported through `changed` and not acted
    on here: storing the new span and updating the table are left to
    whatever created the overlay.
    """

    #: the user let go of a drag; here is the truncation it settled on
    changed = Signal(object)

    def __init__(self, plot: Any, truncation: Truncation,
                 first: float, last: float,
                 colors: Mapping[str, str], parent: Any = None) -> None:
        super().__init__(parent)
        import pyqtgraph as pg

        self.plot: Any = plot
        self.truncation: Truncation = truncation
        self.first: float = float(first)
        self.last: float = float(last)
        self.colors: dict[str, str] = dict(colors)
        #: set while this overlay writes its own region, so a
        #: programmatic restate does not read as a drag —
        #: `setRegion` raises both region signals, unlike an
        #: InfiniteLine's `setValue`
        self._restating = False

        shade = QColor(self.colors['specification_curve'])
        shade.setAlpha(SHADE_ALPHA)
        self.shades: list[Any] = []
        for _ in range(2):
            zone = pg.LinearRegionItem(
                values=(self.first, self.first), movable=False,
                brush=pg.mkBrush(shade), pen=pg.mkPen(None))
            zone.setZValue(15)          # over the trace: it dims it
            for line in zone.lines:
                line.setPen(pg.mkPen(None))
            self.plot.addItem(zone, ignoreBounds=True)
            self.shades.append(zone)
        # the span's edges are the handles, the foreground's dashed
        # line — the corner line's own dress, because it is the same
        # kind of thing: a draggable boundary that is not data
        self.region: Any = pg.LinearRegionItem(
            values=(truncation.start, truncation.stop), movable=True,
            brush=pg.mkBrush(QColor(0, 0, 0, 0)),
            pen=pg.mkPen(self.colors['plot_foreground'], width=2),
            hoverPen=pg.mkPen(self.colors['filter_preview'], width=2),
            bounds=(self.first, self.last))
        self.region.setZValue(20)
        self.region.sigRegionChanged.connect(self._dragging)
        self.region.sigRegionChangeFinished.connect(self._dragged)
        self.plot.addItem(self.region, ignoreBounds=True)
        self._shade_ends(truncation.start, truncation.stop)

    # ---- drawing ----------------------------------------------------------

    def _shade_ends(self, start: float, stop: float) -> None:
        self.shades[0].setRegion((self.first, max(start, self.first)))
        self.shades[1].setRegion((min(stop, self.last), self.last))

    def set_truncation(self, truncation: Truncation) -> None:
        """Move the marks to a span set elsewhere, without emitting `changed`."""
        self.truncation = truncation
        self._restating = True
        try:
            self.region.setRegion((truncation.start, truncation.stop))
        finally:
            self._restating = False
        self._shade_ends(truncation.start, truncation.stop)

    def remove(self) -> None:
        for item in (*self.shades, self.region):
            self.plot.removeItem(item)
        self.shades = []
        self.region = None

    # ---- dragging ---------------------------------------------------------

    def _dragging(self, *_args) -> None:
        """The region is moving under the mouse: the shading follows,
        and nothing else does."""
        if self._restating:
            return
        low, high = self.region.getRegion()
        self._shade_ends(float(low), float(high))

    def _dragged(self, *_args) -> None:
        """The mouse let go: settle the span and say it once."""
        if self._restating:
            return
        low, high = (float(v) for v in self.region.getRegion())
        least = (self.last - self.first) * LEAST_SHARE
        if high - low < least:
            # a span dragged shut means nothing to keep; the marks go
            # back to the stored answer rather than committing it
            self.set_truncation(self.truncation)
            return
        settled = Truncation(low, high)
        self.truncation = settled
        self._shade_ends(low, high)
        self.changed.emit(settled)
