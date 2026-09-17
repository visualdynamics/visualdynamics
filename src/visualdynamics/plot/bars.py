"""The comparison as bar charts: a bar per control channel.

A table of six channels is read; a table of sixty is scanned, and the
one channel that matters is somewhere in it. A bar chart is the same
numbers arranged so the answer is the shape rather than a row — which
channel is worst, how many are out, and by how much, before anything
is read at all.

Two charts, because a random vibration run is judged two ways and they
disagree often enough to be worth seeing apart. **RMS error** is the
level: how far the whole channel sits from what was asked for, in dB.
**Lines outside abort** is the shape: a channel can sit at exactly the
right level and still be out of tolerance across half its band, and one
that is 2 dB low everywhere may never cross an abort limit at all.

A threshold is drawn as shaded ground rather than as a line: what is
being said is "past here is out", and a filled region says it where a
line leaves it to be inferred — red past the upper, blue past the
lower, the same two colors the specification plot shades its abort
zones with. The shading is the handle, and dragging it snaps to a tenth
of a dB or of a percent. A tolerance is a judgment — ±3 dB and a tenth
of the band are where most specifications land, not where they all do —
so those are a starting point, and everything reads off wherever they
are put. In a report they are drawn and not moved: there is nobody on
the other end of a drag in a file.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ..core.compliance import ERROR_DB, LINES_PERCENT, outside_fraction
from ..core.kurtosis import HIGH as KURTOSIS_HIGH
from ..core.kurtosis import LOW as KURTOSIS_LOW
from ..core.kurtosis import NOMINAL

#: how wide a bar is, as a fraction of the space per channel
BAR_WIDTH = 0.68

#: how solid a bar is. Enough to read as a block, light enough that the
#: threshold line over it stays visible.
BAR_ALPHA = 210

#: where the summary sits, as a fraction of the plot's height
SUMMARY_HEIGHT = 0.97

#: how solid a threshold's shading is. The same alpha the specification
#: plot shades its abort zones with, and the same two colors, so "past
#: the limit" looks the same wherever it is said.
ZONE_ALPHA = 38

#: how far past the data a threshold's shading runs. It only has to
#: reach beyond anything that will ever be on screen; the view is set
#: from the bars, so the shading has no say in the scaling.
BEYOND = 1e6

#: what a dragged threshold snaps to. A tolerance is quoted to a tenth
#: of a dB or a tenth of a percent, and a threshold that lands on
#: -2.9473 dB is a number nobody wrote down.
SNAP = 0.1

#: room beyond the widest channel name on the left axis, in pixels:
#: the tick marks themselves, and a gap so the text is not against the
#: bars.
LABEL_MARGIN = 14

#: how much height one channel's row wants, in pixels — in a *report*,
#: whose figure is sized from this so a tall chart scrolls with the
#: page. The app does not use it: its plot surface has no scroll bars,
#: so the chart there fits the pane it is given and the y-axis zooms
#: in for detail.
ROW_HEIGHT = 18

#: how far past the furthest bar or threshold the view may be pulled,
#: as a share of that reach. Enough that the longest bar is not jammed
#: against the frame, and not so much that the chart reads as mostly
#: empty when it is fully zoomed out.
ZOOM_MARGIN = 0.15


class BarChart:
    """A bar per channel, a threshold that can be dragged, and a count.

    Owns its plot. `changed` is called with the thresholds whenever one
    is moved, so whoever put the chart up can remember where they were
    put — a threshold that reset on every redraw would be no threshold
    at all.
    """

    def __init__(self, plot: Any, rows: Sequence[tuple[str, float]], colors: Mapping[str, str],
                 low: float | None, high: float | None = None,
                 changed: Callable[..., None] | None = None,
                 label: str = '', units: str = '',
                 neutral: str = 'response_curve',
                 baseline: float = 0.0) -> None:
        self.plot: Any = plot
        self.rows: list[Any] = list(rows)
        self.colors: dict[str, str] = colors
        #: the thresholds; `low` None is a chart with none at all — a
        #: level chart, where nothing is out and nothing is colored
        self.low: float | None = None if low is None else float(low)
        self.high: float | None = None if high is None else float(high)
        self.changed: Callable[..., None] | None = changed
        self.label: str | None = label
        self.units: str | None = units
        #: what a bar inside the thresholds is painted. The ink by
        #: default — a comparison's in-tolerance channels are the
        #: subject of the chart — but a reading whose whole point is
        #: the *exception* stands its ordinary channels back in gray
        #: (the kurtosis chart, Brandon 2026-08-24), which is also
        #: what the report has always drawn there.
        self.neutral: str = neutral
        #: where a bar grows from. Zero for a reading whose zero means
        #: 'no error at all'; the *nominal* for one that has a nominal
        #: — a kurtosis bar drawn from zero is three units of
        #: agreement dressed up as a measurement, where drawn from
        #: three it is the departure from Gaussian and reads both ways
        #: (Brandon, 2026-08-24).
        self.baseline: float = float(baseline)
        #: the bars themselves, one item holding all of them
        self.bars: Any = None
        self.lines: list[Any] = []
        self.summary: Any = None
        self._moving = False
        self._draw()

    # ---- what is out ------------------------------------------------------

    def values(self) -> list[float]:
        return [value for _label, value in self.rows]

    def beyond(self, value: float) -> str | None:
        """Which way this bar is out, or None.

        A one-sided chart has a ceiling and no floor: no amount of
        staying inside the abort limits is a fault.
        """
        if not np.isfinite(value) or self.low is None:
            return None
        if self.high is None:
            return 'over' if value >= self.low else None
        if value > self.high:
            return 'over'
        if value < self.low:
            return 'under'
        return None

    def share(self) -> float:
        """The percentage of channels outside, wherever the lines are."""
        if self.low is None:
            return 0.0
        return outside_fraction(self.values(), self.low, self.high)

    # ---- drawing ----------------------------------------------------------

    def _over(self, which):
        """Is the ground past this threshold the *upper* fault?

        Named for the fault and not for the threshold, which is the
        distinction the shading got wrong: a one-sided chart's only
        threshold is called 'low' because it is the only one, and what
        is out is everything *above* it.
        """
        return which == 'high' or self.high is None

    def _span(self, which, position):
        """(from, to) the shading for this threshold covers.

        Out to `BEYOND`, which is past anything that will ever be on
        screen: the view is set from the bars, so the shading never gets
        a say in the scaling.
        """
        return ((position, BEYOND) if self._over(which)
                else (-BEYOND, position))

    def _zone_brush(self, which, hover=False):
        """Red past the upper threshold, blue past the lower.

        The same two colors, at the same alpha, that a specification
        plot shades its abort zones with — "past the limit" should look
        the same wherever it is said.

        A one-sided chart has only a ceiling, so its single threshold is
        the red one however it is named.
        """
        import pyqtgraph as pg
        from PySide6.QtGui import QColor

        color = QColor(
            self.colors['exceed_over' if self._over(which)
                        else 'exceed_under'])
        color.setAlpha(ZONE_ALPHA * (2 if hover else 1))
        return pg.mkBrush(color)

    def _brush(self, value):
        import pyqtgraph as pg
        from PySide6.QtGui import QColor

        which = self.beyond(value)
        key = {'over': 'exceed_over', 'under': 'exceed_under'}.get(
            which, self.neutral)
        color = QColor(self.colors[key])
        color.setAlpha(BAR_ALPHA)
        return pg.mkBrush(color)

    def _draw(self):
        """Bars along the y axis, all of them on one set of axes.

        Horizontal, because a channel is named '101Z+' and a column of
        names is legible where a row of them is a smear of overlapping
        ticks. One axis however many there are: with the names down the
        side, splitting buys nothing that height does not, and a tall
        chart scrolls perfectly well in both the app and the report.
        """
        import pyqtgraph as pg
        from PySide6.QtGui import QFontMetrics

        values = self.values()
        positions = np.arange(len(values), dtype=float)
        widths = np.array([self.baseline if not np.isfinite(v) else v
                           for v in values])
        self.bars = pg.BarGraphItem(
            x0=np.full(len(values), self.baseline), y=positions,
            height=BAR_WIDTH,
            width=widths - self.baseline,
            brushes=[self._brush(v) for v in values],
            pen=pg.mkPen(None))
        self.plot.addItem(self.bars)
        axis = self.plot.getAxis('left')
        axis.setTicks([[(i, label) for i, (label, _v) in enumerate(self.rows)]])
        # wide enough for the longest channel name, said outright.
        # Left to size itself the axis settles on the width a number
        # needs, and pyqtgraph draws no tick text that will not fit — so
        # '101Z+' simply vanished, and the chart lost the one thing
        # naming which bar is which.
        metrics = QFontMetrics(axis.font())
        widest = max((metrics.horizontalAdvance(label)
                      for label, _v in self.rows), default=0)
        axis.setWidth(widest + LABEL_MARGIN)
        self.plot.setLabel('bottom', self.label, units=self.units or None)
        self.plot.showGrid(x=True, y=False, alpha=0.25)
        self.plot.setYRange(-0.7, max(len(values) - 0.3, 0.3))
        # the chart fits whatever height it is given, however many
        # channels there are. It used to demand ROW_HEIGHT per row —
        # meant to make a tall chart scroll, but the plot surface in
        # the app has no scroll bars, so a 260-row reading was simply
        # clipped off the bottom of the pane and no zoom could reach
        # the missing bars (Brandon, 2026-08-28). All the bars on
        # screen is the opening shape; zooming in is how a name or a
        # cluster is read. The report still grows with the count — it
        # sizes its figure from ROW_HEIGHT itself, and a page scrolls.
        self._limit_zoom(values)

        # the thresholds as shaded ground rather than as lines: what is
        # being said is "past here is out", and a filled region says it
        # where a line leaves it to be inferred. Drag the shading to
        # move it — the whole region is the handle.
        for which, position in (('low', self.low), ('high', self.high)):
            if position is None:
                continue
            zone = pg.LinearRegionItem(
                values=self._span(which, position), orientation='vertical',
                brush=self._zone_brush(which), movable=True,
                pen=pg.mkPen(None), hoverBrush=self._zone_brush(which, hover=True))
            for edge in zone.lines:
                edge.setPen(pg.mkPen(None))
                edge.setHoverPen(pg.mkPen(None))
            zone.setZValue(-20)      # under the bars it is judging
            zone.sigRegionChangeFinished.connect(
                lambda moved, at=which: self._dragged(at, moved))
            self.plot.addItem(zone, ignoreBounds=True)
            self.lines.append((which, zone))

        self.summary = pg.TextItem(anchor=(0.5, 0.0),
                                   color=self.colors['plot_foreground'])
        self.summary.setZValue(25)
        self.plot.addItem(self.summary, ignoreBounds=True)
        self._restate()

    def _restate(self):
        """The bars in their colors and the count over them."""
        values = self.values()
        if self.bars is not None:
            self.bars.setOpts(brushes=[self._brush(v) for v in values])
        for which, zone in self.lines:
            wanted = self.low if which == 'low' else self.high
            if wanted is not None:
                zone.setRegion(self._span(which, wanted))
        if self.summary is None:
            return
        if self.low is None:
            # a level chart judges nothing: it says how many and which
            # is highest, the way a table's eye runs down its column
            finite = [(v, label) for label, v in self.rows if np.isfinite(v)]
            top = max(finite, default=None)
            self.summary.setText(
                f'{len(self.rows)} channel{"s" * (len(self.rows) != 1)}'
                + (f' — {top[1]} highest at {top[0]:.4g} {self.units}'
                   .rstrip() if top else ''))
        else:
            share = self.share()
            out = sum(1 for v in self.values() if self.beyond(v))
            where = (f'over {self.low:g}{self.units}' if self.high is None
                     else f'outside {self.low:g} to {self.high:g}{self.units}')
            self.summary.setText(
                f'{out} of {len(self.rows)} channels {where} — {share:.0f}%')
        view = self.plot.getViewBox().viewRange()
        self.summary.setPos(
            (view[0][0] + view[0][1]) / 2.0,
            view[1][0] + (view[1][1] - view[1][0]) * SUMMARY_HEIGHT)

    def _limit_zoom(self, values):
        """Fence the view so it cannot be zoomed out into empty ground.

        A bar chart has nothing to find by pulling back: every bar
        starts at zero and the longest one is the whole story, so the
        far side of the axis says only how much nothing there is. Left
        unfenced a scroll wheel takes the bars down to a row of specks
        against a decade of blank, and getting back means guessing.

        The fence is the data and the thresholds, whichever reaches
        further, with a margin. Zero is always inside it, because a bar
        is drawn from there and an axis that can exclude its own origin
        makes every bar the same length.

        Only zooming out. Zooming *in* stays free — reading a cluster
        of near-identical bars apart is a real thing to want, and
        `minXRange` is left unset so it can be done.
        """
        finite = [v for v in values if np.isfinite(v)]
        marks = [m for m in (self.low, self.high) if m is not None]
        # measured from the baseline, which is where the bars grow
        # from — not from zero. A kurtosis chart of bars around three
        # took its reach from the origin and fenced the axis at -5,
        # with every bar crowded into the right-hand quarter of a
        # chart that was mostly empty (Brandon, 2026-08-24). For every
        # reading whose baseline *is* zero this is the same number it
        # always was.
        base = self.baseline
        reach = max([abs(v - base) for v in finite + marks] or [1.0])
        margin = reach * ZOOM_MARGIN
        # a one-sided reading cannot go below zero, so the axis does not
        # offer room there; a two-sided one runs the *whole reach* both
        # ways. It used to offer only the margin below zero — the fence
        # was reach + margin on the right and 15% of it on the left, so
        # a chart of negative deviations clamped its own bars off the
        # axis: four events sitting 13 dB low showed as slivers against
        # the left edge, on the SRS deviation and the random RMS error
        # alike (Brandon, 2026-08-20).
        low = (base - (reach + margin)
               if self.high is not None or min(finite or [base]) < base
               else base - margin * 0.1)
        high = base + reach + margin
        self.plot.setLimits(
            xMin=low, xMax=high,
            yMin=-1.0, yMax=max(len(values), 1),
            maxXRange=high - low,
            maxYRange=max(len(values), 1) + 1.0)
        # and the opening view is the fence, not pyqtgraph's guess: the
        # autorange reads the bars alone, so thresholds past the data
        # (or data past a stale view) started out cropped
        self.plot.setXRange(low, high, padding=0)

    # ---- dragging ---------------------------------------------------------

    def set_threshold(self, which: str, value: float) -> None:
        """Move a threshold, snapped, and restate everything from it.

        What dragging the shading does, and what a caller does to set
        one outright — one path, so a dragged threshold and a stored one
        cannot land on different numbers.
        """
        if self._moving:
            return
        # rounded twice: the second one is only to keep binary
        # floating point from turning 2.9 into 2.9000000000000004
        position = round(round(float(value) / SNAP) * SNAP, 10)
        if which == 'low':
            self.low = position
        else:
            self.high = position
        self._moving = True
        try:
            self._restate()
            if self.changed is not None:
                self.changed(self.low, self.high)
        finally:
            self._moving = False

    def _dragged(self, which, moved=None):
        """The shading was dragged: take its inner edge as the threshold.

        The outer edge is out at `BEYOND` and means nothing — the
        threshold is the edge facing the data, so that is the one read,
        and `_restate` pins the far side back where it belongs.
        """
        if self._moving or moved is None:
            return
        low, high = sorted(moved.getRegion())
        # the edge facing the data is the threshold; the far one is out
        # at BEYOND and means nothing
        self.set_threshold(which, low if self._over(which) else high)


def error_chart(plot: Any, rows: Sequence[tuple[str, float, float]],
                colors: Mapping[str, str], low: float = -ERROR_DB,
                high: float = ERROR_DB,
                changed: Callable[..., None] | None = None) -> BarChart:
    """How far each channel's RMS sits from what was asked for, in dB.

    Two thresholds, moved independently: a specification is not always
    written symmetrically, and over-testing and under-testing are not
    the same fault.
    """
    return BarChart(plot, [(label, value) for label, value, _p in rows],
                    colors, low=low, high=high, changed=changed,
                    label='RMS error', units='dB')


def lines_chart(plot: Any, rows: Sequence[tuple[str, float, float]],
                colors: Mapping[str, str], low: float = LINES_PERCENT,
                changed: Callable[..., None] | None = None) -> BarChart:
    """How much of each channel's band fell outside its abort limits."""
    return BarChart(plot, [(label, percent) for label, _v, percent in rows],
                    colors, low=low, high=None, changed=changed,
                    label='lines outside abort', units='%')


def level_chart(plot: Any, rows: Sequence[tuple[str, float]],
                colors: Mapping[str, str], units: str = '') -> BarChart:
    """The RMS level each channel of a specification asks for, a bar
    apiece — the table's column as a picture, so which channel is
    loudest is seen before anything is read. No thresholds: a
    specification on its own has nothing to be out of (Brandon,
    2026-09-06: the RMS error reading without the ±3 dB coloring).
    """
    return BarChart(plot, list(rows), colors, low=None, high=None,
                    label='RMS level', units=units,
                    neutral='specification_curve')


def kurtosis_chart(plot: Any, rows: Sequence[tuple[str, float]],
                   colors: Mapping[str, str],
                   low: float = KURTOSIS_LOW, high: float = KURTOSIS_HIGH,
                   changed: Callable[..., None] | None = None) -> BarChart:
    """How Gaussian each channel is: Pearson kurtosis, a bar apiece.

    Two thresholds around the nominal three, because both directions
    say something — above, peaks the spectrum never predicted; below,
    a record that has been clipped or was never random. The ordinary
    channels stand back in gray so the exceptions are the chart.

    Every channel whatever it measures: kurtosis is dimensionless, so
    this is the one reading here where accelerations and forces share
    an axis honestly.
    """
    return BarChart(plot, list(rows), colors, low=low, high=high,
                    changed=changed, label='Pearson kurtosis', units='',
                    neutral='specification_curve', baseline=NOMINAL)


#: (axis label, units) for each reading of a transient replication
REPLICATION_LABELS = {
    'waveform': ('waveform error', '%'),
    'srs': ('worst SRS deviation', 'dB'),
    'level': ('level error', 'dB'),
    'srs_rms': ('SRS deviation, RMS across the band', 'dB'),
}


def replication_chart(plot: Any, rows: Sequence[tuple[str, float]], colors: Mapping[str, str],
                      which: str, low: float, high: float | None,
                      changed: Callable[..., None] | None = None
                      ) -> BarChart:
    """One reading of a transient replication, a bar per control channel.

    `rows` is [(label, value)] already reduced to the reading wanted —
    which repeat those values came from is the caller's business, and
    on screen it is the one the event box is pointing at.
    """
    label, units = REPLICATION_LABELS[which]
    return BarChart(plot, list(rows), colors, low=low, high=high,
                    changed=changed, label=label, units=units)
