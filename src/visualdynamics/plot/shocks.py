"""The shock windows, marked on the time history they were found in.

The averaging overlay has a hard job: frames overlap, so it has to say
which sample belongs to which average and how heavily. This one does
not. A shock window is a span, the spans do not overlap, and each is one
event — so the mark is a shaded region with a number on it, and the
number is the point. `shock 3` in a table of spectra means nothing until
you can see which of the four events on the trace it was.

The regions are draggable, because detection is a good guess and not an
oracle: a window that opened a little late, or closed before the
ringdown was done, is fixed by pulling its edge rather than by arguing
with a threshold. `locked` turns that off, for a report or a plot
rendered to a file — there is nobody on the other end of a drag there.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from PySide6.QtGui import QColor

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.shocks import Shock

#: how solid a window's shading is. Light: the trace under it is what
#: the user is actually reading, and the mark is there to bracket it.
BAND_ALPHA = 46

#: and how solid the edges are, which is what you grab to drag
EDGE_ALPHA = 150

#: where the number sits inside its own window, as a fraction of the
#: plot's height from the bottom
LABEL_HEIGHT = 0.93

class ShockOverlay:
    """The windows on one plot, and the dragging of them.

    Owns nothing but its own items: the shocks themselves live on the
    time history, and this is told about them and reports back when one
    is dragged. `changed` is a callback rather than a Qt signal because
    the overlay is not a QObject — it is a handful of items on somebody
    else's plot, and it goes away when they do.
    """

    def __init__(self, plot: Any, shocks: Sequence[Shock],
                 colors: Mapping[str, str],
                 changed: Callable[..., None] | None = None,
                 locked: bool = False, common: bool = False,
                 limit: float | None = None) -> None:
        self.plot: Any = plot
        self.shocks: tuple[Any, ...] = tuple(shocks)
        self.colors: dict[str, str] = colors
        self.changed: Callable[..., None] | None = changed
        self.locked: bool = bool(locked)
        #: whether the series is held at one length, so a drag on any
        #: window is a drag on all of them
        self.common: bool = bool(common)
        #: the end of the record in seconds, so a shared length cannot
        #: be widened past it
        self.limit: float | None = limit
        self.regions: list[Any] = []
        self.labels: list[Any] = []
        self._dragging = False
        self._draw()

    # ---- drawing ----------------------------------------------------------

    def _brushes(self):
        import pyqtgraph as pg
        from PySide6.QtGui import QBrush

        fill = QColor(self.colors.get('averaging_band',
                                      self.colors['plot_foreground']))
        fill.setAlpha(BAND_ALPHA)
        edge = QColor(self.colors.get('averaging_window',
                                      self.colors['plot_foreground']))
        edge.setAlpha(EDGE_ALPHA)
        return QBrush(fill), pg.mkPen(edge, width=1)

    def _draw(self):
        import pyqtgraph as pg

        brush, pen = self._brushes()
        for index, shock in enumerate(self.shocks):
            region = pg.LinearRegionItem(
                values=(shock.start, shock.stop), brush=brush, pen=pen,
                movable=not self.locked)
            region.setZValue(-15)      # under the trace it brackets
            region.is_zone_edge = True
            for line in region.lines:
                line.setPen(pen)
                line.setHoverPen(pen)
            region.sigRegionChangeFinished.connect(
                lambda _region, at=index: self._dragged(at))
            self.plot.addItem(region, ignoreBounds=True)
            self.regions.append(region)
            self.labels.append(self._label(index, shock))

    def _label(self, index, shock):
        import pyqtgraph as pg

        text = pg.TextItem(f'{index + 1}', anchor=(0.5, 0.5),
                           color=self.colors['plot_foreground'])
        text.setZValue(25)
        text.is_zone_edge = True
        self.plot.addItem(text, ignoreBounds=True)
        self._place(text, shock)
        return text

    def _place(self, text, shock):
        """Center the number over its own window, near the top."""
        low, high = self.plot.getViewBox().viewRange()[1]
        text.setPos((shock.start + shock.stop) / 2.0,
                    low + (high - low) * LABEL_HEIGHT)

    # ---- dragging ---------------------------------------------------------

    def _dragged(self, index):
        """One window's edge was moved: rebuild the list and report.

        Two questions, and they are separate. *Which* window moved is
        always just this one: a start belongs to the event it was
        measured from, so dragging a window along the record drags that
        window and nothing else, in either mode. What can be shared is
        the *length*, which is an analysis choice rather than a
        measurement — so when a drag resizes a window and the series is
        held at one length, the new length goes to all of them and they
        grow together until the closest pair runs out of room.

        Either way the result is stopped at its neighbors rather than
        allowed through them. `_windowed` promised no two windows share
        a sample and nothing but this was keeping that promise once the
        plot could edit them.

        Guarded against re-entry. Reporting the change makes the panel
        rewrite the table, which redraws the overlay, which would fire
        this again from inside itself.
        """
        if self._dragging or self.locked or index >= len(self.regions):
            return
        from ..core.shocks import drag_settled

        low, high = sorted(self.regions[index].getRegion())
        settled = drag_settled(self.shocks, index, low, high,
                               self.common, self.limit)
        if settled is None:
            # the drag asked for something the series had no room for;
            # put the region back rather than leaving it where the
            # mouse left it, which would show a window nothing holds.
            # Under the guard: setRegion reports finished, which is the
            # signal this method is answering.
            self._dragging = True
            try:
                self._restate()
            finally:
                self._dragging = False
            return
        self._dragging = True
        try:
            self.shocks = settled
            self._restate()
            if self.changed is not None:
                self.changed(self.shocks)
        finally:
            self._dragging = False

    def _restate(self):
        """Put every region and label back on what `self.shocks` says.

        The regions the user did *not* drag have to be moved by hand: a
        shared length changes all of them, and `set_shocks` will not do
        it afterwards because by then the list it is handed already
        matches the one held here.
        """
        for shock, region, label in zip(self.shocks, self.regions,
                                        self.labels):
            if sorted(region.getRegion()) != [shock.start, shock.stop]:
                region.setRegion((shock.start, shock.stop))
            self._place(label, shock)

    # ---- keeping up -------------------------------------------------------

    def set_shocks(self, shocks: Sequence[Shock]) -> None:
        """Show these instead, without going through a redraw of the plot."""
        shocks = tuple(shocks)
        if shocks == self.shocks:
            return
        self.remove()
        self.shocks = shocks
        self.regions, self.labels = [], []
        self._draw()

    def remove(self) -> None:
        for item in (*self.regions, *self.labels):
            try:
                self.plot.removeItem(item)
            except RuntimeError:
                # the plot was torn down first, taking its items with
                # it, which is the usual way a pane is replaced
                pass
        self.regions, self.labels = [], []
