"""The frames, drawn on the record they will be cut from.

A list of five numbers does not tell you whether the analysis sits on
the part of the record you meant. Drawn, it does.

Above the trace is the rail: one slot per frame carrying that frame's
window, shaded under the curve, so the window is read where it is set
rather than over the data it is applied to. Frames that overlap cannot
share a slot without running through each other, so each takes the
lowest slot whose last frame has already ended — one line at no
overlap, two at half, four at three quarters.

Under each frame's slot, its band falls to the foot of the plot. The
height is what says which slot a frame is in, so where two frames share
record you see two heights and the step between them: the overlap has a
shape rather than only a shade.

The span is one draggable region. Take it by the middle to slide the
analysis along the record; take it by an edge to add or drop frames —
and only whole frames, because half a frame is not an average, so the
edge snaps back to where the last whole frame ends.

A capture the controller already cut into frames has nothing to drag:
the frame is the record, and the region is drawn locked around it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping  # noqa: F401
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor

from ..core.averaging import from_span

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.averaging import Averaging

#: how many color stops the gradient carries. The window is smooth,
#: so this only has to be finer than the eye at plot width.
BAND_STOPS = 48

#: the marks sit above the trace: under it they were lost in the ink of
#: a busy record, which is the one thing they cannot be
MARK_Z = 20

#: The rail's proportions and its two alphas live on the averaging
#: (`Averaging.rail`), not here: the report draws this same rail in a
#: canvas, and one set of numbers is what keeps the two figures
#: describing one analysis. The shading under a frame follows that
#: frame's own window, so what is drawn is the weight each moment of
#: the record carries — a Hann fades in and out, a rectangle is flat,
#: and where two frames overlap the two shadings add exactly as the
#: averaging does.


class AveragingOverlay(QObject):
    """The averaging parameters as marks on a time history plot.

    Owns every item it adds and takes them all away again on `remove`,
    so a redraw cannot leave a band behind. It reports a drag and does
    not act on it: what a new averaging *means* — storing it, redrawing
    the table — belongs to whoever put the overlay up.
    """

    #: the user dragged; here is the averaging that describes where it is
    changed = Signal(object)

    def __init__(self, plot: Any, averaging: Averaging, sample_rate: float,
                 samples: int, colors: Mapping[str, str],
                 locked: bool = False, parent: Any = None,
                 origin: float = 0.0) -> None:
        super().__init__(parent)
        self.plot: Any = plot
        self.averaging: Averaging = averaging
        self.sample_rate: float = float(sample_rate)
        self.samples: int = int(samples)
        #: the record's first instant: the marks sit on the plot's
        #: clock, the averaging counts from the record's beginning
        self.origin: float = float(origin)
        self.colors: dict[str, str] = colors
        self.locked: bool = bool(locked)
        self.bands: list[Any] = []
        self.glyphs: list[Any] = []
        self.caps: list[Any] = []
        self.region: Any = None
        self._snapping = False
        #: the plot's own y limits, put back when the marks come off
        self._locked = None
        self._draw()
        # connected before the room is made, so that the marks follow the
        # range they are about to be given rather than the one they were
        # drawn against
        plot.getViewBox().sigYRangeChanged.connect(self._rescale)
        self._make_room()

    # ---- where the rail is ------------------------------------------------

    def _make_room(self):
        """Raise the view's top so the rail sits above the trace.

        Drawn into the height the trace already fills, the windows and
        their ticks land on the data they describe and both are harder
        to read. The marks are excluded from the extents — they have to
        be, or they would chase their own scaling — so the room has to
        be asked for rather than taken.

        From the data's own bounds and not the current range: at the
        moment an overlay is built the view may still be at its default
        and about to autorange, and a range set now would be the one
        that autorange never gets to replace.

        The ceiling has to be lifted before the range can reach it. A
        plot is locked to its data's extents so that zoom and pan cannot
        wander off into blank space, and a range set past that lock is
        quietly clamped back — which is what made the first attempt at
        this do nothing at all.
        """
        view = self.plot.getViewBox()
        bounds = view.childrenBounds()[1]
        if bounds is None or bounds[0] is None or bounds[1] == bounds[0]:
            return
        low, high = float(bounds[0]), float(bounds[1])
        foot = self.averaging.rail(self.sample_rate)['foot']
        # the data keeps the space it had; the rail gets its share above
        span = (high - low) / foot
        pad = (high - low) * 0.02
        floor, ceiling = low - pad, low - pad + span
        self._locked = view.state['limits']['yLimits'][:]
        # only ever outward: a plot whose ceiling is already above what
        # the rail needs keeps it, and a lock brought *down* to fit the
        # rail would take away room the user had to pan in
        was_low, was_high = self._locked
        view.setLimits(
            yMin=floor if was_low is None else min(was_low, floor),
            yMax=ceiling if was_high is None else max(was_high, ceiling))
        view.setYRange(floor, ceiling, padding=0)

    def _rail(self):
        """(level per frame, baseline y per frame, glyph height, foot).

        Everything the rail needs, in data coordinates, worked out from
        the visible height — so the marks keep their proportions as the
        view is zoomed and panned rather than shrinking to a smear.
        """
        low_view, high_view = self.plot.getViewBox().viewRange()[1]
        foot, span = float(low_view), float(high_view) - float(low_view)
        rail = self.averaging.rail(self.sample_rate)
        baselines = [foot + span * at for at in rail['baselines']]
        return (rail['levels'], baselines, span * rail['glyph'], foot)

    # ---- drawing ----------------------------------------------------------

    def _draw(self):
        import pyqtgraph as pg

        self._clear_items()
        self._make_marks()

        # the span's edges are the handles, so they take the band's own
        # color at full strength: those are the parts you grab
        edge = QColor(self.colors['averaging_band'])
        self.region = pg.LinearRegionItem(
            values=self._span(), movable=not self.locked,
            brush=pg.mkBrush(QColor(0, 0, 0, 0)), pen=pg.mkPen(edge, width=2))
        self.region.setZValue(-5)
        if not self.locked:
            self.region.sigRegionChanged.connect(self._dragging)
            self.region.sigRegionChangeFinished.connect(self._dragged)
        self.plot.addItem(self.region, ignoreBounds=True)

    def _span(self):
        """(first sample, last sample) of the analysis, in seconds."""
        first = self.averaging.start_sample(self.sample_rate)
        return (self.origin + first / self.sample_rate,
                self.averaging.stop(self.sample_rate, self.origin))

    def _make_marks(self):
        """A band and a window for every frame, and the ticks under them.

        Each is its own item because they overlap and have to layer; one
        item with lifts in it would fill as a single polygon and close
        across the gaps.
        """
        import pyqtgraph as pg

        _levels, baselines, height, foot = self._rail()
        bounds = self.averaging.frame_bounds(self.sample_rate, self.origin)
        n = self.averaging.frame_length
        shape = self.averaging.shape()
        offsets = np.arange(n) / self.sample_rate
        glyph_color = QColor(self.colors['averaging_window'])
        glyph_color.setAlpha(round(self.averaging.rail(
            self.sample_rate)['glyph_alpha'] * 255))

        for (low, high), baseline in zip(bounds, baselines):
            # the band is a column from the foot of the plot up to this
            # frame's own slot — the height says which level it is on —
            # and it fades across with the window, so the shading is the
            # weight each moment of the record actually carries
            band = pg.PlotDataItem(
                np.array([low, high]), np.array([baseline, baseline]),
                pen=pg.mkPen(None), fillLevel=foot,
                fillBrush=self._band_brush(low, high, shape))
            band.setZValue(-20)          # under the trace it describes
            band.is_zone_edge = True     # a mark, not a measurement
            # the marks describe the record rather than adding to it:
            # counted in the extents they would rescale the plot, and a
            # taller view would draw them taller still
            self.plot.addItem(band, ignoreBounds=True)
            self.bands.append(band)

            curve = pg.PlotDataItem(
                low + offsets, baseline + shape * height,
                pen=pg.mkPen(self.colors['averaging_window'], width=1),
                fillLevel=baseline,
                fillBrush=pg.mkBrush(QColor(glyph_color)))
            curve.setZValue(MARK_Z)
            curve.is_zone_edge = True
            self.plot.addItem(curve, ignoreBounds=True)
            self.glyphs.append(curve)

        for xs, ys in self._cap_segments():
            tick = pg.PlotDataItem(
                xs, ys,
                pen=pg.mkPen(self.colors['averaging_window'], width=1))
            tick.setZValue(MARK_Z)
            tick.is_zone_edge = True
            self.plot.addItem(tick, ignoreBounds=True)
            self.caps.append(tick)

    def _band_brush(self, low, high, shape):
        """A brush that fades across the frame with its window.

        A gradient in data coordinates, so it stretches with the view
        the way the band it fills does. Sampled at a fixed number of
        stops rather than once per sample: the window is smooth, and a
        thousand stops would say nothing a few dozen do not.

        Normalized by the peak rather than assumed to reach one: a flat
        top does not. Its shoulders dip slightly below zero — a real
        negative weight — but far too little to round to a visible
        alpha, so they come out clear and nothing has to clamp them.
        """
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QBrush, QLinearGradient

        color = QColor(self.colors['averaging_band'])
        peak_alpha = self.averaging.rail(
            self.sample_rate)['band_alpha'] * 255
        weights = shape / np.max(np.abs(shape))
        across = np.linspace(0.0, 1.0, len(weights))
        gradient = QLinearGradient(QPointF(low, 0.0), QPointF(high, 0.0))
        for at in np.linspace(0.0, 1.0, BAND_STOPS):
            stop = QColor(color)
            stop.setAlpha(round(peak_alpha
                                * float(np.interp(at, across, weights))))
            gradient.setColorAt(float(at), stop)
        return QBrush(gradient)

    def _cap_segments(self):
        """A tick under each end of every window, one curve each.

        A tapering window comes back down to its baseline at both ends,
        so without these there is nothing to say where a Hann frame
        stopped and the next began.

        One curve each rather than one curve with lifts in it. A gapped
        array is the shape pyqtgraph builds a path from by packing Qt's
        bytes itself rather than through Qt's own API — see
        `visualdynamics.plot.gapless`. Two points per tick is a third of them
        non-finite, which is as gapped as an array gets.
        """
        _levels, baselines, _height, foot = self._rail()
        span = self.plot.getViewBox().viewRange()[1][1] - foot
        cap = span * self.averaging.rail(self.sample_rate)['cap']
        segments = []
        for (low, high), baseline in zip(
                self.averaging.frame_bounds(self.sample_rate, self.origin),
                baselines):
            for at in (low, high):
                segments.append((np.array([at, at]),
                                 np.array([baseline - cap, baseline + cap])))
        return segments

    def _rescale(self, *_args):
        """Redraw the marks against the height that is now visible."""
        self._redraw_marks()

    # ---- dragging ---------------------------------------------------------

    def _dragging(self, *_args):
        """Follow the cursor with the marks while the drag is live."""
        if self._snapping:
            return
        moved = self._from_region(*self.region.getRegion())
        if moved != self.averaging:
            self.averaging = moved
            self._redraw_marks()

    def _dragged(self, *_args):
        """The drag ended: snap the region onto whole frames and report.

        Snapping moves the region, and moving it says the drag finished
        — so without the guard one mouse release reports itself twice.
        """
        if self._snapping:
            return
        self.averaging = self._from_region(*self.region.getRegion())
        self._snap()
        self._redraw_marks()
        self.changed.emit(self.averaging)

    def _snap(self):
        """Put the region back where the frames actually are."""
        self._snapping = True
        try:
            self.region.setRegion(self._span())
        finally:
            self._snapping = False

    def _from_region(self, low, high):
        """The averaging a dragged region means — `core.averaging.from_span`,
        which the stage's handles commit through too."""
        return from_span(self.averaging, low, high, self.sample_rate,
                         self.samples, origin=self.origin)

    def _redraw_marks(self):
        """The marks for the averaging as it now stands, leaving the
        region alone — it is what the cursor is holding."""
        self._clear_marks()
        self._make_marks()

    # ---- taking it away ---------------------------------------------------

    def _drop(self, item):
        try:
            self.plot.removeItem(item)
        except RuntimeError:
            # the plot was torn down first, taking its items with it,
            # which is the usual way a pane is replaced
            pass

    def _clear_marks(self):
        for item in [*self.bands, *self.glyphs, *self.caps]:
            if item is not None:
                self._drop(item)
        self.bands = []
        self.glyphs = []
        self.caps = []

    def _clear_items(self):
        self._clear_marks()
        if self.region is not None:
            self._drop(self.region)
        self.region = None

    def remove(self) -> None:
        """Take every mark off the plot. The overlay is finished after."""
        try:
            self.plot.getViewBox().sigYRangeChanged.disconnect(self._rescale)
        except (RuntimeError, TypeError):
            # the view went first, which is the usual way a pane is torn
            # down; there is nothing left to disconnect from
            pass
        self._clear_items()
        try:
            # the room was for the rail; with the rail gone the trace
            # has the height, and the lock its own ceiling again
            view = self.plot.getViewBox()
            if self._locked is not None:
                view.setLimits(yMin=self._locked[0], yMax=self._locked[1])
                self._locked = None
            view.enableAutoRange(axis='y')
        except RuntimeError:
            pass

    def set_averaging(self, averaging: Averaging) -> None:
        """Draw a different averaging — what the table's edits arrive as."""
        self.averaging = averaging
        self._redraw_marks()
        if self.region is not None:
            self._snap()
