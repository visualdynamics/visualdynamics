"""2D plotting, built on pyqtgraph.

One renderer serves both the GUI and scripting, so the rules for how data is
drawn — grouping by dimension, unit-aware labels, log magnitude in the
frequency domain, legends — live here only.

Records are grouped by ordinate dimension, one stacked plot per dimension,
so an object mixing accelerations, forces and voltages lands on comparable
axes. Curves are clipped to the view and, on an evenly spaced axis,
peak-downsampled — what keeps million-sample time histories interactive.
In decades the spacing is not even and the downsampling stays off (see
`_draw_shaped`).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ..core.data import DataArray, Frf, _CoherenceBase
    from ..core.photos import Photos
    from ..core.shapes import ShapeSet
    from ..units import UnitSystem

from ..theme import theme as resolve_theme
from ..units import DEFAULT_SYSTEM, UNKNOWN

MAX_RECORDS = 50
MAX_LEGEND = 12
#: legend entries per row to start from; the row refits itself to the
#: axes' width once it knows it
LEGEND_COLUMNS = 4
#: the legend's text size. 8 pt, the size a printed figure can still be
#: read at — it was 7, which printed at a third of a page wide came out
#: smaller than anything else on the figure (2026-09-26, for the print
#: figures of a paper). A script drawing for print may set it larger;
#: `legend_below` reads it at each call.
LEGEND_TEXT_SIZE = '8pt'


#: room beyond the widest category label on an axis, in pixels: the tick
#: marks themselves, and a gap so the text is not against the data
LABEL_MARGIN = 14


def fit_label_axis(axis: Any, labels: Iterable[str],
                   margin: int = LABEL_MARGIN) -> None:
    """Make a category axis wide enough for its widest label.

    pyqtgraph draws no tick text that will not fit its axis, and says
    nothing: left to size itself, a left axis settles on the width a
    number needs, and a channel name or a label like "band-average,
    update" simply vanishes while the shorter ones draw (the bar
    charts found it first; the band-average paper again, 2026-09-26).
    Measured with the axis's own font, the label that is there is the
    label that is drawn.
    """
    from PySide6.QtGui import QFontMetrics

    metrics = QFontMetrics(axis.font())
    widest = max((metrics.horizontalAdvance(str(label)) for label in labels),
                 default=0)
    axis.setWidth(widest + margin)


def legend_below(layout: Any, plot: Any, row: int,
                 colors: dict | None = None,
                 text_size: str | None = None) -> Any:
    """A horizontal legend in its own layout row under the plot.

    pyqtgraph's default legend floats inside the view, anchored to a
    corner, and on a plot with any real content it lands on top of the
    data — a specification's bands, an FRF's peaks — and neither can be
    read (Brandon, 2026-09-01). Outside is the honest place: the plot
    keeps its whole area, the names line up beneath it in columns, and
    the entries still register themselves through `plot.legend`, so
    every ``name=`` a curve was drawn with arrives as before.

    Plots occupy the even rows of a layout (``2 * row``) and their
    legends the odd ones — the window's row walker keys on
    ``series_key``, which a legend does not carry, so it steps past.

    `text_size` is the labels' size, `LEGEND_TEXT_SIZE` when omitted.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    class Row(pg.LegendItem):
        """A legend that fills the axes' width and no more.

        pyqtgraph's legend is built for a corner of the view: it sizes
        itself from its entries and never asks how wide its host is.
        Below the plot the width matters twice over — a grid cell
        re-sets the geometry to the full strip, so the maximum has to
        be pinned to the content for the cell's alignment to center
        it; and the names should wrap to the axes' width rather than
        run in one row past it (Brandon, 2026-09-01). The pin is
        lifted before each re-measure — kept, it froze the box at the
        first entry's width and every later name drew past its edge.
        """

        def __init__(self, view, **kwargs):
            super().__init__(**kwargs)
            # pyqtgraph's legend ignores transformations — built to float
            # over the data at a fixed screen size — so any drawing of the
            # plot at a scale passed it by: printed at three times, every
            # other item grew and the legend's text stayed a third of its
            # size (the paper's print figures, 2026-09-26). In a layout row
            # it is laid out like the axes, and scales with them.
            self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations,
                         False)
            self._view = view
            self._fitting = False
            view.sigResized.connect(self.refit)

        def addItem(self, item, name):
            super().addItem(item, name)
            self.refit()

        def refit(self):
            """As many entries per row as the axes are wide — measured,
            not predicted: the box carries padding the entry widths do
            not, so the count starts from the estimate and comes down
            until the box fits."""
            if self._fitting or not self.items:
                return
            available = self._view.width()
            widest = max(sample.width() + label.width() + 6
                         for sample, label in self.items)
            columns = max(1, min(len(self.items), int(available // widest)))
            self._fitting = True
            try:
                self.setColumnCount(columns)
                while columns > 1 and self.width() > available:
                    columns -= 1
                    self.setColumnCount(columns)
            finally:
                self._fitting = False
            self.updateGeometry()

        def setGeometry(self, *args):
            """Centered on the axes, not on the strip: the cell spans the
            left axis too, so its own center sits right of the data."""
            from PySide6.QtCore import QRectF

            rect = args[0] if len(args) == 1 else QRectF(*args)
            parent = self.parentItem()
            if parent is not None and self._view.scene() is not None:
                axes = parent.mapRectFromScene(
                    self._view.sceneBoundingRect())
                rect = QRectF(axes.center().x() - rect.width() / 2,
                              rect.y(), rect.width(), rect.height())
            super().setGeometry(rect)

        def updateSize(self):
            """Sized from what the entries ask for, not from what they
            were last given: the box's own grid stretches its labels
            to whatever height the box has, so measuring geometries
            fed the height back into itself — 44 px became 1060 in
            two refits. Minimum sizes are the entries' true extents
            (a label's text, a sample's fixed 20 px)."""
            width = height = 0
            for line in range(self.layout.rowCount()):
                line_height = line_width = 0
                for col in range(self.layout.columnCount()):
                    item = self.layout.itemAt(line, col)
                    if item:
                        line_width += item.minimumWidth() + 3
                        line_height = max(line_height, item.minimumHeight())
                width = max(width, line_width)
                height += line_height
            self.setMinimumSize(width, height)
            self.setMaximumSize(width, height)
            # the cell places the box only once told the size changed
            self.updateGeometry()

    style = ({'labelTextColor': colors['plot_foreground']}
             if colors else {})
    legend = Row(plot.getViewBox(), colCount=LEGEND_COLUMNS, offset=None,
                 labelTextSize=text_size or LEGEND_TEXT_SIZE, **style)
    layout.addItem(legend, row=2 * row + 1, col=0)
    grid = getattr(layout, 'ci', layout).layout
    grid.setAlignment(legend, Qt.AlignmentFlag.AlignHCenter)
    plot.legend = legend
    return legend

# How lopsided a MAC grid may be before it stops being drawn with square
# cells. A cross-MAC's shape is the two sets' mode counts, and 139 modes
# against 8 is a grid 17 times taller than it is wide: held to that shape
# it is a sliver too narrow to read a color out of, let alone click a
# cell in. Past the clamp the grid fills the frame instead and the cells
# stretch — the only alternative that keeps every column on screen.
MAC_ASPECT_CLAMP = 3.0

# pixels a mode's frequency label needs along its axis before the next
# one may be drawn. Upright on the bottom axis and horizontal on the
# left, a label is about a line of text across either way, so one number
# serves both.
MODE_LABEL_EXTENT = 18


def mac_frame_ratio(rows: int, columns: int) -> float:
    """Width over height for a MAC grid of this shape, clamped.

    The frame outside the plot is sized by this and the aspect lock
    inside it is decided by the same clamp, so the two cannot disagree
    about whether cells are square — a frame narrower than the grid's own
    shape plus a lock inside it scrolls columns off the screen.
    """
    ratio = float(columns) / float(rows) if rows else 1.0
    return min(max(ratio, 1.0 / MAC_ASPECT_CLAMP), MAC_ASPECT_CLAMP)

# how far past the outermost limit the abort fill runs. The view is
# locked to the data's own extents, so this only has to be beyond
# anything that will ever be on screen; a factor rather than a constant
# keeps it sane on a log axis, where the fill has to stay positive.
BEYOND = 1e6

#: a phase axis: every quarter turn labeled, and room past the half
#: turns for their labels. Pinned at ±185° the ±180° labels sat a few
#: pixels from the edge and pyqtgraph, which draws no label that does
#: not fit, dropped them — the axis read -90, 0 and 90 alone (Brandon,
#: 2026-09-27, through the band-average paper). The report's phase
#: figures take the same two.
PHASE_LIMITS = (-200.0, 200.0)
PHASE_TICKS = (-180, -90, 0, 90, 180)

#: every ordinary curve's line on screen, in the app, a standalone plot
#: and the report page (sent in its payload). One pixel: anything wider
#: leaves Qt's fast path, and two dozen dense FRFs at two repainted a
#: cursor's strip in 41 ms where one takes 2.5, the whole plot in 113
#: where one takes 31 (Retina, 2026-09-27). Brandon kept the speed; the
#: light theme's darker palette and `EXPORT_CURVE_WIDTH` answer "the
#: lines are hard to see" instead.
CURVE_WIDTH = 1
#: an image of a plot — a .png, a print figure, a copy to the clipboard
#: — draws every data curve at least this wide. An image is drawn once,
#: so the width costs nothing there, where on screen anything wider
#: than one device pixel repaints about ten times slower; printed, a
#: one-pixel curve is three quarters of a point (Brandon, 2026-09-27).
#: Wider pens keep their width, so a specification still shows around
#: the run over it.
EXPORT_CURVE_WIDTH = 2
#: a measurement's line when it is the one run read against its
#: specification: wider than the rest, so it stands out from the gray
#: specification behind it in a small printed legend (Brandon,
#: 2026-09-27: "maybe thicker line")
MEASURED_WIDTH = 2.5
#: and translucent, so where it lies exactly on its specification — a
#: band value sitting on the narrowband target it was taken from — the
#: gray still shows through it, not only at its edges. Dashes would
#: have separated them too; Brandon does not want dashes (2026-09-27)
MEASURED_ALPHA = 0.75
#: the specification standing back behind that measurement: wider still,
#: a pixel of gray either side of the line over it — at one pixel under
#: a dense measurement it vanished (Brandon, 2026-09-25), and a
#: half-pixel edge was lost in print where the two coincided
STOOD_BACK_WIDTH = 4.5
#: the ring drawn at a sine reading that sat under its noise floor
FLOOR_MARK_SIZE = 6

#: categorical curve colors on a dark background
CURVE_COLORS = [
    '#4c92d9', '#ff8c2b', '#3fb950', '#e5534b', '#a371f7',
    '#b07d62', '#e668c3', '#8b949e', '#d2c14e', '#39c5cf',
]
#: the same hues on a light one, darkened to 4.5:1 against white. The
#: dark set stood at 1.8:1 (yellow) to 3.7:1 there, and a one-pixel line
#: that faint was hard to see (Brandon, 2026-09-27). Wider lines would
#: have helped too and cost ten times the repaint; color costs nothing.
CURVE_COLORS_LIGHT = [
    '#2a77c6', '#bc5600', '#2e863a', '#df2d23', '#8c4df5',
    '#9b694e', '#d223a1', '#6b7580', '#827621', '#218086',
]


def curve_colors(colors: Mapping[str, str] | None = None) -> list[str]:
    """The curve palette for a theme's plot background: the light set
    on a light background, the dark set on a dark one or with no theme
    given."""
    if colors is None:
        return CURVE_COLORS
    background = str(colors['plot_background']).lstrip('#')
    red, green, blue = (int(background[i:i + 2], 16) for i in (0, 2, 4))
    light = 0.299 * red + 0.587 * green + 0.114 * blue > 127
    return CURVE_COLORS_LIGHT if light else CURVE_COLORS


def curve_color(index: int, colors: Mapping[str, str] | None = None) -> str:
    """The nth curve's color, wrapping. The same cycle everywhere a
    curve is drawn, so a record keeps its place in it between the app,
    a standalone plot and the report; `colors` (a resolved theme) picks
    the light or dark shade of it."""
    palette = curve_colors(colors)
    return palette[index % len(palette)]


def axis_label(dimension: str, unit_system: UnitSystem,
                hint: str | None = None) -> str:
    """Axis label for a dimension: a typeset unit, or a note when undefined.

    pyqtgraph renders label HTML, so a quotient dimension shows as a real
    stacked fraction rather than '(in/s**2)/lbf'.

    A `hint` is what the source said the quantity was without saying what
    scale it was on. The axis names it, since an unlabeled axis of
    accelerations is less use than one that at least says 'acceleration'.
    """
    if dimension == UNKNOWN:
        return f'{hint} [units undefined]' if hint else 'units undefined'
    return unit_system.label_html(dimension) or '-'


def build_plot(layout: Any, data: Any, unit_system: UnitSystem | None = None,
               theme: Any = None, max_records: int = MAX_RECORDS,
               records: Sequence[int] | None = None,
               component: str = 'magnitude') -> tuple[int, int]:
    """Draw one data array. See `build_plots` for several at once."""
    return build_plots(layout, [(None, data, records)],
                       unit_system=unit_system, theme=theme,
                       max_records=max_records, component=component)


def background_brush(colors: Mapping[str, str]) -> Any:
    """The plots' background: the same gradient the 3D view uses.

    Anchored to the device rather than the scene, so it stays put while the
    data is panned and zoomed instead of sliding around behind it.
    """
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QBrush, QColor, QLinearGradient

    gradient = QLinearGradient(QPointF(0, 0), QPointF(0, 1))
    gradient.setCoordinateMode(QLinearGradient.CoordinateMode.StretchToDeviceMode)
    gradient.setColorAt(0.0, QColor(colors['plot_background_top']))
    gradient.setColorAt(1.0, QColor(colors['plot_background']))
    return QBrush(gradient)


def display_limits(data: Any,
                   unit_system: UnitSystem) -> dict[str, np.ndarray]:
    """A specification's limit curves in display units, or nothing.

    Anything else is not a specification and has none, which is why this
    asks rather than testing the type: the plot does not need to know what a
    Specification is, only that some data brings bounds with it.
    """
    limits = getattr(data, 'limits', None)
    if not limits:
        return {}
    return {bound: data.display_limit(bound, unit_system) for bound in limits}


def curve_budget(limit: int, needs: Sequence[int]) -> list[int]:
    """How many curves each plot may draw, sharing one limit between them.

    First come, first served lets the first plot spend the whole budget and
    leaves the next one empty — which is what selecting a spectrum and a
    time history did: one plot drew 45 curves and the other drew five.

    Plots wanting less than an equal share release the rest to the others,
    so a two-channel plot beside a large one costs the large one almost
    nothing.
    """
    shares = [0] * len(needs)
    remaining = min(limit, sum(needs))
    while remaining > 0:
        wanting = [i for i, need in enumerate(needs) if shares[i] < need]
        if not wanting:
            break
        fair = remaining // len(wanting)
        if not fair:
            # fewer curves left than plots: one each, in order, and stop
            for i in wanting[:remaining]:
                shares[i] += 1
            break
        for i in wanting:
            take = min(fair, needs[i] - shares[i])
            shares[i] += take
            remaining -= take
    return shares


def _pair(data, index):
    """The (response, reference) a record is between — the data's own
    `record_pair`, named here for the plot's readers."""
    return data.record_pair(index)


def bounded_by_specification(series: Sequence[tuple[str | None, Any, Sequence[int] | None]],
                             scales: dict[str, int] | None = None
                             ) -> tuple[Sequence[tuple[str | None, Any, Sequence[int] | None]], int]:
    """Restrict PSDs to the records a selected specification actually bounds.

    A specification is normally written for autospectra only, while a measured
    CPSD carries every cross term too. Drawn together, the 30 cross terms of a
    6-channel CPSD are 30 curves with no limit anywhere near them — they are
    not wrong, they are just not the comparison being asked for, and they bury
    the six that are.

    So when a specification is plotted alongside other PSDs, both are cut to
    the pairs they have in common. Nothing else is touched: a specification
    says nothing about a time history, and restricting one against it would
    drop every record for no reason.

    The measured spectra are also drawn **scaled to the specification**
    when a scaling applies — the standard practice for a run captured
    below the 0 dB requirement (`compliance.comparison_scale_db`; held
    on the object, or detected in whole dB). The scaling exists only in
    the comparison: it is applied to a throwaway copy, the object's own
    values are never touched, and a scaled curve says so in its legend
    name. `scales` lets a caller who resolved the number with more
    context — the app, which knows an octave PSD and the PSD it was
    banded from share one scale — say what each named entry gets.

    Returns (series, dropped).
    """
    from ..core.compliance import comparison_scale_db
    from ..core.data import Psd, Specification

    def keyed(data, i):
        # the DOF pair *and* the quantity: a drive's force spectrum
        # shares a DOF name with the control channel measured beside
        # it, and a specification for accelerations says nothing about
        # newtons. 'unknown' matches anything — an undefined-units
        # record has made no claim to contradict.
        return (*_pair(data, i), data.known_dim(i))

    def compatible(a, b):
        return (a[:2] == b[:2]
                and (a[2] == b[2] or 'unknown' in (a[2], b[2])))

    spec_pairs, other_pairs, spec = set(), set(), None
    for _name, data, records in series:
        if not isinstance(data, Psd):
            continue
        wanted = range(data.num_records) if records is None else records
        pairs = {keyed(data, int(i)) for i in wanted}
        if isinstance(data, Specification):
            spec_pairs |= pairs
            spec = data if spec is None else spec
        else:
            other_pairs |= pairs
    if not spec_pairs or not other_pairs:
        return series, 0

    shared = {key for key in spec_pairs | other_pairs
              if any(compatible(key, other) for other in
                     (other_pairs if key in spec_pairs else spec_pairs))}
    if not shared:
        # nothing in common at all, so the restriction has nothing to say.
        # Cutting both to nothing would leave an empty plot, where the useful
        # answer is to draw them and let the mismatch be visible.
        return series, 0
    out, dropped = [], 0
    for name, data, records in series:
        if not isinstance(data, Psd):
            out.append((name, data, records))
            continue
        wanted = list(range(data.num_records) if records is None else records)
        keep = [int(i) for i in wanted if keyed(data, int(i)) in shared]
        dropped += len(wanted) - len(keep)
        if not keep:
            continue
        if not isinstance(data, Specification):
            db = (scales[name] if scales is not None and name in scales
                  else comparison_scale_db(spec, data))
            if db:
                data = scaled_for_comparison(data, db)
                if name is not None:
                    name = f'{name} ({db:+d} dB)'
        out.append((name, data, keep))
    return out, dropped


def scaled_for_comparison(data: Any, db: float) -> Any:
    """A throwaway copy of a spectrum with the comparison scale applied.

    A shallow copy sharing everything but the values, because the whole
    point is that the object itself is never changed — the scale exists
    on the drawn comparison and nowhere else. The copy holds its scale
    at 0 so nothing downstream scales it again.
    """
    import copy

    out = copy.copy(data)
    out.ordinate = np.asarray(data.ordinate) * 10.0 ** (db / 10.0)
    out.scale_db = 0
    return out


#: points per decade a breakpoint specification is filled in to, so the
#: polyline drawn for it follows the power law its breakpoints mean
POWER_LAW_POINTS = 400

def specification_pairs(series: Sequence[tuple[str | None, Any, Sequence[int] | None]]) -> list[tuple[str, str]]:
    """The DOF pairs the plot draws one at a time, in the
    specification's own order.

    Where there is a measurement, these are the pairs both sides have:
    a specification with its response is a comparison, and six of them
    on one axis is a thicket with no comparison visible in it.

    A specification on its own is the same story without the second
    curve. Six targets and their two dozen limit lines are exactly as
    unreadable stacked together, so the channels are still offered one
    at a time — every channel the specification carries, since there is
    nothing to intersect them with.
    """
    from ..core.data import Psd, Specification

    wanted, measured = [], set()
    for _name, data, records in series:
        if not isinstance(data, Psd):
            continue
        rows = range(data.num_records) if records is None else records
        pairs = [_pair(data, int(i)) for i in rows]
        if isinstance(data, Specification):
            for pair in pairs:
                if pair not in wanted:
                    wanted.append(pair)
        else:
            measured.update(pairs)
    if not measured:
        return wanted
    return [pair for pair in wanted if pair in measured]


def only_pairs(series: Sequence[tuple[str | None, Any, Sequence[int] | None]],
               pairs: Sequence[tuple[str, str]]) -> Sequence[tuple[str | None, Any, Sequence[int] | None]]:
    """The series cut to these DOF pairs, leaving anything else alone.

    The plural of `only_pair`, because a table under the plot can name
    several channels at once where a drop-down names one.
    """
    from ..core.data import Psd

    wanted = list(pairs)
    out = []
    for name, data, records in series:
        if not isinstance(data, Psd):
            out.append((name, data, records))
            continue
        rows = range(data.num_records) if records is None else records
        keep = [int(i) for i in rows if _pair(data, int(i)) in wanted]
        if keep:
            out.append((name, data, keep))
    return out


def only_pair(series: Sequence[tuple[str | None, Any, Sequence[int] | None]], pair: tuple[str, str]) -> Sequence[tuple[str | None, Any, Sequence[int] | None]]:
    """The series cut to one DOF pair, leaving anything else alone."""
    from ..core.data import Psd

    out = []
    for name, data, records in series:
        if not isinstance(data, Psd):
            out.append((name, data, records))
            continue
        rows = range(data.num_records) if records is None else records
        keep = [int(i) for i in rows if _pair(data, int(i)) == pair]
        if keep:
            out.append((name, data, keep))
    return out


def pair_label(pair: tuple[str, str]) -> str:
    """How a DOF pair reads: the DOF alone when it is its own reference."""
    return pair[0] if pair[0] == pair[1] else f'{pair[0]}/{pair[1]}'


# how many channel labels a map's y-axis can carry before they collide
MAP_LABELS = 24


def build_coherence_map(layout: Any, series: Sequence[tuple[str | None, Any, Sequence[int] | None]],
                        unit_system: UnitSystem | None = None,
                        theme: Any = None) -> tuple[int, int]:
    """Every channel's coherence at once: frequency across, channel down.

    A line plot answers "is this channel good"; there is no reading 339 of
    them at once, and the curve budget means most are not even drawn. As a
    map the same data answers "which channels are bad, and where" in one
    look — a poor channel is a dark row, a poor band is a dark column.

    Coherence is a bounded ratio, so the color scale is pinned to 0..1
    rather than fitted to the data: a map whose scale moved with the
    selection would make a good channel look bad next to a better one.

    Returns (rows_drawn, rows_requested); nothing is dropped, so they match.
    """
    import numpy as np
    import pyqtgraph as pg

    us = unit_system or DEFAULT_SYSTEM
    colors = resolve_theme(theme)
    layout.clear()
    layout.setBackground(background_brush(colors))

    rows, labels, abscissa = [], [], None
    multiple = len({name for name, _, _ in series if name}) > 1
    for name, data, records in series:
        wanted = (range(data.num_records) if records is None
                  else [int(i) for i in records])
        values = data.display_ordinate(us, wanted)
        if abscissa is None:
            abscissa = data.display_abscissa(us)
        for position, record in enumerate(wanted):
            rows.append(values[position].real)
            label = data.record_label(record)
            labels.append(f'{name}: {label}' if multiple and name else label)
    if not rows:
        return 0, 0

    plot = layout.addPlot(row=0, col=0)
    # the same reason as build_plots: an axis reads the global foreground when
    # it is built, so a theme switch would leave old axes in the old color
    foreground = pg.mkPen(colors['plot_foreground'])
    for edge in ('left', 'bottom', 'top', 'right'):
        axis = plot.getAxis(edge)
        axis.setPen(foreground)
        axis.setTextPen(foreground)
    image = pg.ImageItem(np.asarray(rows).T)
    image.setColorMap(pg.colormap.get('viridis'))
    # 0..1 always, not the selection's own range
    image.setLevels((0.0, 1.0))
    left, right = float(abscissa[0]), float(abscissa[-1])
    image.setRect(left, 0.0, right - left, float(len(rows)))
    plot.addItem(image)
    plot.setXRange(left, right, padding=0)
    plot.setYRange(0, len(rows), padding=0)
    plot.getViewBox().setLimits(xMin=left, xMax=right,
                                yMin=0, yMax=len(rows))
    # a channel list reads downward. Image coordinates run the other way, so
    # left alone the first channel sits at the bottom and the axis reads
    # backwards against the tree it came from.
    plot.getViewBox().invertY(True)
    plot.showGrid(x=True, y=False, alpha=0.25)
    plot.setLabel('bottom', f'frequency [{us.label_html("frequency")}]')
    plot.setLabel('left', 'channel')
    plot.getAxis('left').setTicks([_map_ticks(labels)])
    fit_label_axis(plot.getAxis('left'),
                   [label for _position, label in _map_ticks(labels)])
    # the bar is what makes the colors readable as numbers
    bar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=pg.colormap.get('viridis'),
                          label='coherence', interactive=False)
    bar.setImageItem(image, insert_in=plot)
    return len(rows), len(rows)


def _map_ticks(labels):
    """(position, text) for a channel axis, thinned so they do not collide.

    Every label at 339 channels is a gray smear; one in every nth is a scale.
    Each sits at the middle of its row, which is where its data is.
    """
    stride = max(1, -(-len(labels) // MAP_LABELS))
    return [(index + 0.5, label) for index, label in enumerate(labels)
            if index % stride == 0]


def cmif_curves(data: Any, records: Sequence[int] | None = None,
                unit_system: UnitSystem | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """(singular values (k, freqs), display abscissa) — the CMIF.

    The records assemble into the response x reference matrix they are —
    an absent pair is zero, the standard practical treatment of an
    incomplete matrix — and every frequency line gets a singular value
    decomposition. The largest singular value peaks at every mode; the
    second one peaking too is how a repeated root shows itself.
    """
    us = unit_system or DEFAULT_SYSTEM
    wanted = (list(range(data.num_records)) if records is None
              else [int(i) for i in records])
    values = data.display_ordinate(us, wanted)
    responses = list(dict.fromkeys(data.response_dof[i] for i in wanted))
    references = list(dict.fromkeys(data.reference_dof[i] for i in wanted))
    row = {dof: k for k, dof in enumerate(responses)}
    column = {dof: k for k, dof in enumerate(references)}
    matrix = np.zeros((len(data.abscissa), len(responses), len(references)),
                      dtype=np.complex128)
    for position, i in enumerate(wanted):
        matrix[:, row[data.response_dof[i]],
               column[data.reference_dof[i]]] = values[position]
    singular = np.linalg.svd(matrix, compute_uv=False)
    return singular.T, data.display_abscissa(us)


def _mode_axis(orientation, frequencies, height=None):
    """An axis ticked one mode at a time, labeled with its frequency.

    The ticks are the grid's cells rather than round numbers, so they
    cannot be computed the way a frequency axis's are: tick *i* sits at
    the center of cell *i* and reads mode *i*'s own frequency. Handing
    pyqtgraph the whole list through `setTicks` is what drew all 139 of
    the drone's modes on top of one another — it drops a tick *level*
    that will not fit, never the individual labels inside one — so the
    thinning belongs here, computed against the axis's real length. Doing
    it here also makes it follow the zoom: pull in on a cluster of modes
    and every one of them is named.

    A bottom axis additionally stands its labels on end, since one
    frequency per mode overlaps horizontally by a dozen modes and
    pyqtgraph has no rotated-tick option — it redraws the text of the
    stock `drawPicture` rotated 90° about each label's center, and
    reserves `height` for them to stand in.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import QRectF

    values = np.asarray(frequencies, dtype=float)

    class ModeAxis(pg.AxisItem):
        """Cell-centered ticks, thinned to the labels that fit."""

        def tickValues(self, minVal: float, maxVal: float,
                       size: float) -> list[tuple[float, list[float]]]:
            first = max(0, int(np.floor(min(minVal, maxVal))))
            last = min(len(values), int(np.ceil(max(minVal, maxVal))))
            if last <= first:
                return []
            fits = max(int(size // MODE_LABEL_EXTENT), 1)
            step = max(1, -(-(last - first) // fits))
            # anchored on multiples of the step, so labels stay put while
            # panning instead of stepping along with the near edge
            start = first - first % step
            return [(step, [i + 0.5 for i in range(start, last, step)
                            if i >= first])]

        def tickStrings(self, positions: Sequence[float], scale: float,
                        spacing: float) -> list[str]:
            return [f'{values[int(at)]:.1f}'
                    if 0 <= int(at) < len(values) else ''
                    for at in positions]

    class UprightAxis(ModeAxis):
        """An axis whose tick labels stay horizontal.

        pyqtgraph rotates a left axis's text to run up the side, which
        is unreadable for a channel name and merely awkward for a
        number. Everything else is pyqtgraph's own drawing.
        """

        def drawPicture(self, painter: Any, axisSpec: Any, tickSpecs: Any,
                        textSpecs: Any) -> None:
            painter.setRenderHint(painter.RenderHint.Antialiasing, False)
            painter.setRenderHint(painter.RenderHint.TextAntialiasing, True)
            pen, start, end = axisSpec
            painter.setPen(pen)
            painter.drawLine(start, end)
            for pen, tick_start, tick_end in tickSpecs:
                painter.setPen(pen)
                painter.drawLine(tick_start, tick_end)
            if self.style['tickFont'] is not None:
                painter.setFont(self.style['tickFont'])
            painter.setPen(self.textPen())
            for rect, flags, text in textSpecs:
                painter.save()
                # rotate about a point far enough down that the label
                # hangs entirely below the axis line — spinning about
                # the stock rect's center pokes long labels up into
                # the plot area
                painter.translate(rect.center().x(),
                                  rect.top() + rect.width() / 2)
                painter.rotate(-90)
                painter.drawText(
                    QRectF(-rect.width() / 2, -rect.height() / 2,
                           rect.width(), rect.height()), int(flags), text)
                painter.restore()

    axis = (UprightAxis if orientation == 'bottom' else ModeAxis)(
        orientation=orientation)
    if height is not None:
        # the attribute, not setFixedHeight: the axis recomputes its
        # height from the (unrotated) text metrics and clobbers a Qt
        # fixed height
        axis.fixedHeight = height
    return axis


def build_mac(view: Any, frequencies: ArrayLike, matrix: ArrayLike,
              theme: Any = None,
              column_frequencies: ArrayLike | None = None
              ) -> tuple[int, int]:
    """A MAC matrix as a viridis grid.

    Rows and columns are modes, labeled by their frequencies; with
    `column_frequencies` the grid is a cross-MAC — rows one set, columns
    the other — and rectangular when the counts differ. The scale is
    pinned 0..1 like the coherence map, because a MAC is a bounded
    ratio. The color is the reading — per-cell numbers were tried and
    made the grid too busy to read at a glance.
    """
    import pyqtgraph as pg

    colors = resolve_theme(theme)
    view.clear()
    view.setBackground(background_brush(colors))
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    rows, columns = matrix.shape
    across = (frequencies if column_frequencies is None
              else column_frequencies)
    plot = view.addPlot(
        row=0, col=0,
        axisItems={'left': _mode_axis('left', frequencies),
                   'bottom': _mode_axis('bottom', across, height=48)})
    foreground = pg.mkPen(colors['plot_foreground'])
    for edge in ('left', 'bottom', 'top', 'right'):
        axis = plot.getAxis(edge)
        axis.setPen(foreground)
        axis.setTextPen(foreground)
    image = pg.ImageItem(matrix.T)
    image.setColorMap(pg.colormap.get('viridis'))
    image.setLevels((0.0, 1.0))
    image.setRect(0.0, 0.0, float(columns), float(rows))
    plot.addItem(image)
    # the table reads downward: first mode top-left, like the mode list
    plot.invertY(True)
    # zoomable like every other plot, but never past the grid itself
    plot.getViewBox().setLimits(xMin=0, xMax=float(columns),
                                yMin=0, yMax=float(rows))
    # Cells stay square whatever the widget's shape: the frame outside
    # sizes the whole widget, axes included, and its margins would
    # otherwise stretch the grid — the aspect lock holds it exactly.
    # Only while the grid is near enough square to be given its own
    # shape, though: a 139x8 cross-MAC clamps to 1:3, and locking cells
    # square inside a frame wider than that shows 1.3 of the 8 columns
    # and scrolls the other seven out of sight. A lopsided grid fills
    # what it is given.
    plot.getViewBox().setAspectLocked(
        1.0 / MAC_ASPECT_CLAMP <= columns / rows <= MAC_ASPECT_CLAMP)
    plot.setMenuEnabled(False)
    plot.setLabel('left', 'mode frequency [Hz]')
    plot.setLabel('bottom', 'mode frequency [Hz]')
    plot.setXRange(0, columns, padding=0)
    plot.setYRange(0, rows, padding=0)
    return matrix.shape


class _Extents:
    """The data's own bounding box, in view coordinates, and the lock
    that keeps zoom and pan inside it.

    View coordinates matter: a log-ordinate plot's ViewBox works in
    log10 of the data, so the extents collect log10 there too.
    """

    def __init__(self) -> None:
        self.x_lo: float | None = None
        self.x_hi: float | None = None
        self.y_lo: float | None = None
        self.y_hi: float | None = None

    def add(self, x: ArrayLike, y: ArrayLike, log_ordinate: bool,
            log_abscissa: bool = False) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        finite_x = x[np.isfinite(x)]
        finite_y = y[np.isfinite(y)]
        if log_ordinate:
            finite_y = np.log10(finite_y[finite_y > 0])
        if log_abscissa:
            # same reason as the ordinate: a log-x ViewBox works in
            # log10 of the data, so the extents collect log10 there too
            finite_x = np.log10(finite_x[finite_x > 0])
        if len(finite_x):
            self.x_lo = min(self.x_lo, float(finite_x.min())) \
                if self.x_lo is not None else float(finite_x.min())
            self.x_hi = max(self.x_hi, float(finite_x.max())) \
                if self.x_hi is not None else float(finite_x.max())
        if len(finite_y):
            self.y_lo = min(self.y_lo, float(finite_y.min())) \
                if self.y_lo is not None else float(finite_y.min())
            self.y_hi = max(self.y_hi, float(finite_y.max())) \
                if self.y_hi is not None else float(finite_y.max())

    def frame_y(self, plot: Any) -> None:
        """Show the y range these extents cover, autorange off.

        Separate from `lock`, which only says how far the view may be
        pulled: a plot left autoranging still frames whatever was
        drawn, however little of it is the thing being judged.
        """
        if self.y_lo is None:
            return
        pad = 0.05 * (self.y_hi - self.y_lo) or 1.0
        plot.getViewBox().disableAutoRange(axis='y')
        plot.setYRange(self.y_lo - pad, self.y_hi + pad, padding=0)

    def lock(self, plot: Any,
             y_bounds: tuple[float, float] | None = None) -> None:
        if self.x_lo is None or self.y_lo is None:
            return
        pad_x = 0.02 * (self.x_hi - self.x_lo) or 1.0
        pad_y = 0.05 * (self.y_hi - self.y_lo) or 1.0
        y_lo, y_hi = ((self.y_lo - pad_y, self.y_hi + pad_y)
                      if y_bounds is None else y_bounds)
        plot.getViewBox().setLimits(
            xMin=self.x_lo - pad_x, xMax=self.x_hi + pad_x,
            yMin=y_lo, yMax=y_hi)


def add_mode_markers(plot: Any, frequencies: ArrayLike,
                     foreground: str | None = None) -> None:
    """Dashed bookmarks at each mode's frequency, labels staggered so
    clusters stay readable — the fitting screen's markers, wherever a
    modal synthesis is drawn.

    The fitting screen keeps its subtle gray (the default); the
    resynthesis overlays pass the theme foreground instead — white on
    the dark theme, black on the light — so the markers read apart
    from the grid."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    color = pg.mkColor(foreground if foreground is not None
                       else (139, 148, 158))
    rgb = (color.red(), color.green(), color.blue())
    for index, frequency in enumerate(frequencies):
        marker = pg.InfiniteLine(
            pos=float(frequency), angle=90, movable=False,
            # heavier and more opaque than the grid it sits on: at one
            # pixel and half alpha a bookmark reads as another grid
            # line, which is the one thing it must not
            pen=pg.mkPen(rgb + (200,), width=2,
                         style=Qt.PenStyle.DashLine),
            label=f'{float(frequency):.2f}',
            labelOpts={'position': 0.95 - 0.05 * (index % 3),
                       'color': rgb + (170,),
                       'movable': False})
        plot.addItem(marker)


def _synthesis_marks(series):
    """The mode frequencies any synthesized overlay in `series` carries,
    deduped in order."""
    marks = []
    for _name, data, _records in series:
        for frequency in getattr(data, 'mode_frequencies', ()) or ():
            if frequency not in marks:
                marks.append(frequency)
    return marks


def build_cmif(layout: Any, series: Sequence[tuple[str | None, Any, Sequence[int] | None]],
               unit_system: UnitSystem | None = None,
               theme: Any = None) -> int:
    """The Complex Mode Indicator Function of each selected FRF set.

    One curve per singular value, log magnitude over frequency. A
    synthesized overlay's CMIF draws dashed in the measured set's colors,
    singular value for singular value — the modal model's indicator
    against the measurement's. Singular values that are numerically zero
    (a synthesis truncated below the reference count is rank-deficient)
    are dropped rather than drawn twenty decades down.

    Returns the number of curves drawn.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    us = unit_system or DEFAULT_SYSTEM
    colors = resolve_theme(theme)
    layout.clear()
    layout.setBackground(background_brush(colors))
    plot = layout.addPlot(row=0, col=0)
    foreground = pg.mkPen(colors['plot_foreground'])
    for edge in ('left', 'bottom', 'top', 'right'):
        axis = plot.getAxis(edge)
        axis.setPen(foreground)
        axis.setTextPen(foreground)
    plot.showGrid(x=True, y=True, alpha=0.3)
    plot.setLogMode(x=False, y=True)
    first = series[0][1]
    plot.setLabel('left', axis_label(first.ordinate_dim[0], us,
                                     first.dimension_hint[0]))
    plot.setLabel('bottom', f'frequency [{us.label_html("frequency")}]')
    legend_below(layout, plot, 0, colors)

    drawn, index, pair_colors = 0, 0, {}
    synthesized_curves = []
    extents = _Extents()
    several = len({name for name, _d, _r in series}) > 1
    for name, data, records in series:
        singular, x = cmif_curves(data, records, us)
        synthesized = bool(getattr(data, 'synthesized', False))
        base = name.removesuffix(' synthesized')
        floor = singular.max() * 1e-10
        for k in range(singular.shape[0]):
            if singular[k].max() <= floor:
                continue
            key = (base, k)
            paired = synthesized and key in pair_colors
            if paired:
                color = pair_colors[key]
            else:
                color = curve_color(index, colors)
                index += 1
                if not synthesized:
                    pair_colors[key] = color
            style = (Qt.PenStyle.DashLine if synthesized
                     else Qt.PenStyle.SolidLine)
            label = f'{name}: CMIF {k + 1}' if several else f'CMIF {k + 1}'
            # connect='finite' everywhere curves are drawn: zeros in
            # log mode are -inf coordinates, and Qt's raster drawLines
            # fast path (used only for connect='all') intermittently
            # SIGBUSes on them. Finite is also simply right — a gap, not
            # a line to infinity.
            curve = plot.plot(x, singular[k], connect='finite',
                              pen=pg.mkPen(color, width=CURVE_WIDTH, style=style),
                              **({} if paired else {'name': label}))
            # same treatment as every other line curve (see build_curve):
            # peak-keeping, so no singular-value peak is thinned away.
            # Without it the fit screen repainted 24 curves of 8193
            # points on every cursor tick — 34 ms of a 16.7 ms frame
            # spent rastering points no pixel could show.
            curve.setDownsampling(auto=True, method='peak')
            curve.setClipToView(True)
            if synthesized:
                # kept so a caller can move the synthesis without
                # rebuilding the plot: while a fit cursor is dragged the
                # measurement does not change and only these do, and
                # tearing the whole CMIF down for them costs 34 ms
                # against a frame's 16.7
                synthesized_curves.append(curve)
            if not synthesized:
                # the axes are the measurement's. A synthesis reaches
                # wherever the model puts it — a lightly damped mode
                # ten decades below anything measured, an anti-resonance
                # the model has and the data does not — and letting it
                # set the limits zooms the measurement into a band at
                # the top of the plot to make room for a curve nobody
                # is judging. It is drawn where it falls; it does not
                # get a say in the scaling.
                extents.add(x, singular[k], True)
            drawn += 1
    plot.synthesized_curves = synthesized_curves
    add_mode_markers(plot, _synthesis_marks(series),
                     colors['plot_foreground'])
    extents.lock(plot)
    # and the *visible* range too, not only how far it can be pulled.
    # `lock` sets pan limits; without this the plot still autoranges to
    # everything drawn, so a synthesis dipping decades below anything
    # measured squeezed the measurement into a band at the top.
    extents.frame_y(plot)
    return drawn


def build_plots(layout: Any, series: Sequence[tuple[str | None, Any, Sequence[int] | None]],
                unit_system: UnitSystem | None = None, theme: Any = None,
                max_records: int = MAX_RECORDS,
                component: str = 'magnitude') -> tuple[int, int]:
    """Draw several data arrays into one pyqtgraph GraphicsLayout.

    `series` is a list of (name, data, records); `records` may be None for
    every record. Curves are grouped into a plot per (abscissa dimension,
    ordinate dimension) pair, so selecting an FRF and a time history at once
    stacks them on separate axes instead of nonsense-sharing one. Names
    prefix the legend when more than one object is drawn.

    `component` picks how complex frequency data is read: 'magnitude'
    (log ordinate, the default), 'real', 'imag', or 'phase' (degrees,
    axis pinned to ±180). Real data ignores it.

    Returns (curves_drawn, curves_requested).
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..core.data import Specification

    series, _dropped = bounded_by_specification(series)
    us = unit_system or DEFAULT_SYSTEM
    colors = resolve_theme(theme)
    layout.clear()
    layout.setBackground(background_brush(colors))

    multiple = len({name for name, _, _ in series if name}) > 1
    groups, group_marks, requested = {}, {}, 0
    for name, data, records in series:
        wanted = (range(data.num_records) if records is None
                  else [int(i) for i in records])
        x = data.display_abscissa(us)
        # Nothing is converted here. The rows are converted below, once
        # the budget says which curves are drawn, and in blocks: only
        # what is being drawn — converting the whole object to show one
        # curve is what made a 1356-record FRF take half a second a
        # render — and never the whole selection at once, since the
        # gathered copy and its scaled copy beside a long run's own
        # data is what took a 22 GB import to the machine's ceiling
        # (2026-09-18). The entry carries (data, record) in the row's
        # place until then.
        # the object's own reading of its axis — shared with the waterfall
        log_ordinate = data.log_scaled()
        # and of the abscissa: an SRS reads in decades of natural
        # frequency (Brandon, 2026-08-25) — shared with the waterfall
        # and the report, one flag on the class for all three
        log_abscissa = bool(data.log_abscissa)
        # a complex FRF reads four ways; anything but the magnitude is
        # signed, so the log axis gives way to a linear one
        tag = (component if component != 'magnitude'
               and np.iscomplexobj(data.ordinate)
               and data.abscissa_dim == 'frequency' else None)
        if tag:
            log_ordinate = False
        # once per object, not once per record: the conversion is the same
        limits = display_limits(data, us)
        synthesized = bool(getattr(data, 'synthesized', False))
        # a specification follows the measurement it bounds the same way a
        # synthesized FRF follows the one it predicts: same pair, same color
        from ..core.sine import SineTarget
        follower = synthesized or isinstance(data,
                                             (Specification, SineTarget))
        # How this one means its own area. A specification is a
        # continuous requirement whose points are breakpoints of a power
        # law; a PSD is a density per bin, flat across each. Anything
        # else — an FRF, a coherence — is a value at a frequency and is
        # drawn as the line it is. Only for a magnitude on a frequency
        # axis: a phase, or a real part, is not an area under anything.
        # the object's own reading of itself, which is the same field
        # its area is worked out from — so the picture is of the number
        # that would be computed from it and cannot drift from it. No
        # class is consulted: a specification computed from a record is
        # a density like any other PSD, and a type test said otherwise.
        # One rule for every reading: the waterfall consults it too.
        shape = drawing_shape(data, bool(tag))
        for position, i in enumerate(wanted):
            requested += 1
            # hinted records join the axis for their own quantity, not one
            # pooled 'units undefined' axis holding forces and volts alike
            # the axis preference joins the key: a coherence and a
            # dimensionless spectrum must not share one axis, since one
            # wants 0..1 linear and the other decades of log
            key = (data.abscissa_dim, data.ordinate_dim[i],
                   data.dimension_hint[i], log_ordinate, log_abscissa,
                   PHASE_LIMITS if tag == 'phase'
                   else data.ordinate_limits, tag)
            label = data.record_label(i)
            # a specification's limits ride along on the record they bound
            bands = {bound: values[i] for bound, values in limits.items()}
            # which measured curve a follower belongs to, by DOF pair
            # *and* quantity — the channel identity everywhere now
            pair = (*data.record_pair(i), data.known_dim(i))
            # a spectrum that knows its own bin widths says so, and a
            # step plot then lands on the bands the standard defines
            # rather than on the midpoints between their centers
            own = getattr(data, 'bin_edges', None)
            widths = own() if own is not None and getattr(
                data, 'bandwidth', None) is not None else None
            groups.setdefault(key, []).append(
                (f'{name}: {label}' if multiple and name else label, x,
                 (data, i), data.abscissa_dim == 'frequency', bands,
                 pair, follower, synthesized, shape, widths,
                 # the object and its record, for the judgment the
                 # exceedance boxes are drawn from
                 (data, i)))
            if synthesized:
                marks = group_marks.setdefault(key, [])
                for f in getattr(data, 'mode_frequencies', ()) or ():
                    if f not in marks:
                        marks.append(f)

    drawn = 0
    first_by_abscissa = {}
    keys = list(groups)
    budget = curve_budget(max_records, [len(groups[key]) for key in keys])
    for row, key in enumerate(keys):
        (abscissa_dim, ordinate_dim, hint, log_ordinate, log_abscissa,
         limits, tag) = key
        plot = layout.addPlot(row=2 * row, col=0)   # legends take the odd rows
        # what this plot draws, readable back by whoever adds overlays:
        # the envelope's cursor has to ride the plot of the quantity it
        # deflects, not whichever row happens to be first
        plot.series_key = key
        # set the axis colors here rather than leaning on pyqtgraph's
        # global foreground option: that is read when an axis is built, so
        # a theme switch would leave old axes in the old theme's color
        foreground = pg.mkPen(colors['plot_foreground'])
        for edge in ('left', 'bottom', 'top', 'right'):
            axis = plot.getAxis(edge)
            axis.setPen(foreground)
            axis.setTextPen(foreground)
        plot.showGrid(x=True, y=True, alpha=0.3)
        if tag == 'phase':
            left = 'phase [deg]'
            plot.getAxis('left').setTicks(
                [[(t, f'{t:g}') for t in PHASE_TICKS], []])
        else:
            left = axis_label(ordinate_dim, us, hint)
            if tag:
                left += ' (real)' if tag == 'real' else ' (imaginary)'
        plot.setLabel('left', left)
        plot.setLabel('bottom',
                      f'{abscissa_dim} [{us.label_html(abscissa_dim)}]')
        if log_ordinate or log_abscissa:
            # pyqtgraph transforms every PlotDataItem's data itself, so
            # the zone fills — built from plot.plot curves for exactly
            # this reason — land in the same log space
            plot.setLogMode(x=log_abscissa, y=log_ordinate)
        if limits is not None:
            # a bounded ratio reads against its own bounds, not against
            # whatever range this particular measurement happened to cover
            plot.setYRange(*limits, padding=0)
        # link only axes that share an abscissa, so zoom stays meaningful
        if (abscissa_dim, log_abscissa) in first_by_abscissa:
            plot.setXLink(first_by_abscissa[abscissa_dim, log_abscissa])
        else:
            # keyed by scale as well as dimension: a log-x view and a
            # linear one share coordinates in name only, and linking
            # them slaves real frequencies to their own logarithms
            first_by_abscissa[abscissa_dim, log_abscissa] = plot

        curves = _converted(groups[key][:budget[row]], us, tag)
        # paired predictions stay out of the legend, so they don't count
        # against it either
        if sum(1 for c in curves if not c[7]) <= MAX_LEGEND:
            legend_below(layout, plot, row, colors)
        # measured first, followers after: a follower takes its pair's
        # color — a synthesized FRF goes dashed on top and out of the
        # legend (the solid line already names the pair); a specification
        # keeps its own solid line and legend entry, in the color of the
        # measurement it bounds
        ordered = ([c for c in curves if not c[6]]
                   + [c for c in curves if c[6]])
        pair_colors, index = {}, 0
        extents = _Extents()
        # a specification's abort limits mark the measurement they bound,
        # which is drawn separately and found again by its DOF pair
        measured_by_pair, bounded = {}, []
        # a pair is 'bounded' when the group holds both a specification
        # for it and a measurement of it.
        #
        # A specification is a follower that is not a synthesis — those
        # are the only two kinds of follower there are. What it is *not*
        # is "a follower carrying limit curves", which is what this asked
        # before: that made the colors depend on whether bounds happened
        # to be written, so one selection drew two ways. A random run's
        # target came out gray behind its response, and a transient run's
        # target — same objects, same gesture, but no limits, because a
        # tolerance on a waveform is not a settled convention — fell
        # through to the ordinary color cycle and took its response's
        # color as an undistinguished follower.
        with_spec = {c[5] for c in ordered if c[6] and not c[7]}
        with_measurement = {c[5] for c in ordered if not c[6]}
        bounded_pairs = with_spec & with_measurement
        # how many measurements each pair has: one is read against its
        # specification drawn wider, several — five runs of one channel
        # against its target — at the ordinary width. Either way each
        # takes the color cycle and only the specification stands back
        # in gray. All in one color, five runs could not be told apart
        # (2026-09-26)
        measurements = {}
        for c in ordered:
            if not c[6]:
                measurements[c[5]] = measurements.get(c[5], 0) + 1
        zones, exceeded, zone_db = set(), set(), {}
        for (label, x, values, is_frequency, bands, pair, follower,
             dashed, shape, widths, owner) in ordered:
            magnitude = (np.abs(values) if is_frequency
                         and np.iscomplexobj(values) else values.real)
            # A specification and the response it bounds are a pair: the
            # specification stands back in gray because it is the
            # reference, and the response takes a curve color. It used
            # to take the foreground, and printed, a near-black line
            # over a gray one could not be told from it (Brandon,
            # 2026-09-27, the band-average paper's figures).
            #
            # Alone, the specification *is* what is being looked at, so
            # it takes the foreground itself. Gray would be saying "this
            # is the reference for something", and there is nothing here
            # for it to be the reference for.
            paired = follower and pair in pair_colors
            if bounded_pairs and pair in bounded_pairs and follower:
                color = colors['specification_curve']
            elif bounded_pairs and pair in bounded_pairs:
                color = curve_color(index, colors)
                index += 1
            elif follower and pair in with_spec and not with_measurement:
                color = colors['response_curve']
            elif paired:
                color = pair_colors[pair]
            else:
                color = curve_color(index, colors)
                index += 1
                if not follower:
                    pair_colors.setdefault(pair, color)
            style = (Qt.PenStyle.DashLine if dashed
                     else Qt.PenStyle.SolidLine)
            # The reference that stands back is drawn twice as wide as
            # the measurement over it, as the 3-D stage has always drawn
            # its stood-back target. At one pixel a gray requirement
            # under a dense level trace vanished entirely — Brandon
            # could not find the specification under the sine levels
            # (2026-09-25) — and at two it shows through as the band the
            # measurement is being read against, still behind the data.
            stands_back = bool(follower and bounded_pairs
                               and pair in bounded_pairs)
            read_against = bool(not follower and pair in bounded_pairs
                                and measurements.get(pair, 0) == 1)
            width = (STOOD_BACK_WIDTH if stands_back
                     else MEASURED_WIDTH if read_against else CURVE_WIDTH)
            pen = pg.mkPen(color, width=width, style=style)
            if read_against:
                shade = pen.color()
                shade.setAlphaF(MEASURED_ALPHA)
                pen.setColor(shade)
            named = {} if paired and dashed else {'name': label}
            curve, drawn_x, drawn_y = _draw_shaped(plot, x, magnitude, shape,
                                                   pen, widths=widths,
                                                   log_abscissa=log_abscissa,
                                                   **named)
            if stands_back:
                # beneath the measurements whatever order the series
                # came in, and still over the zone fills (-5, -10): drawn
                # after its response at the same depth, the opaque gray
                # covered it and a compliant run showed no color at all
                # (the band-average paper, 2026-09-27)
                curve.setZValue(-1)
            # a sine level's readings where the tone was under the noise
            # sit at the floor and are marked: hollow rings, so the
            # curve says "not seen above this" rather than drawing a
            # hole or a level that was not there (2026-09-30)
            flagged = getattr(owner[0], 'below_floor', None)
            if flagged is not None and np.asarray(flagged)[owner[1]].any():
                mask = np.asarray(flagged)[owner[1]]
                plot.plot(np.asarray(x)[mask], np.asarray(magnitude)[mask],
                          pen=None, symbol='o', symbolSize=FLOOR_MARK_SIZE,
                          symbolPen=pg.mkPen(color, width=1),
                          symbolBrush=None)
            extents.add(drawn_x, drawn_y, log_ordinate, log_abscissa)
            if shape == 'steps' and not follower:
                measured_by_pair.setdefault(pair, (x, magnitude, owner))
            if bands and follower:
                bounded.append((pair, x, bands, owner))
            # The zones say where the limits are. Drawn as lines too,
            # four more curves per record crowd the two that are being
            # compared and the shading behind them says the same thing.
            zones |= _shade_limit_zones(plot, x, bands, colors, shape,
                                        widths)
            if bands and follower:
                _zone_decibels(zone_db, bands, magnitude, ordinate_dim)
            for values in bands.values():
                # they still set how far the view reaches, so a limit
                # above everything measured is not cropped off
                extents.add(x, values.real, log_ordinate, log_abscissa)
            del curve
            drawn += 1
        for pair, spec_x, bands, (spec_obj, si) in bounded:
            found = measured_by_pair.get(pair)
            if found is None:
                continue
            lines, measured, (meas_obj, mi) = found
            written = {bound: bands.get(bound)
                       for bound in ('abort_lower', 'abort_upper')}
            # the one judgment, `compliance.judge` — an area against
            # an area over each cell, the same call the table and the
            # report's page make — with each mark's level brought into
            # the plot's units by the limit's own conversion
            from ..core.compliance import comparison_scale_db, judge

            scale = comparison_scale_db(spec_obj, meas_obj)
            verdicts = {}
            for bound, over in (('abort_upper', True), ('abort_lower', False)):
                if written.get(bound) is None:
                    continue
                verdict = judge(spec_obj, meas_obj, si, mi, bound, over, scale)
                raw = np.atleast_2d(spec_obj.limits[bound])[si]
                verdict['level'] = verdict['level'] * _display_factor(
                    written[bound], raw)
                verdicts[bound] = verdict
            # the extents are known by now: every curve is drawn, and
            # the view is about to be locked to them
            reach = _reach(extents, log_ordinate)
            exceeded |= _shade_exceedances(plot, lines, measured, verdicts,
                                            colors, reach)
        if getattr(plot, 'legend', None) is not None:
            _legend_zones(plot.legend, zones, exceeded, zone_db, colors)
        add_mode_markers(plot, group_marks.get(key, []),
                         colors['plot_foreground'])
        # zoom and pan stay inside what the data covers — the opening
        # view is also the outer limit
        extents.lock(plot, y_bounds=limits)
    return drawn, requested


def _converted(curves, us, tag):
    """The drawn curves with their rows converted, a block at a time.

    Each entry arrives with (data, record) where its row will go; the
    rows come out of `DataArray.display_blocks` for exactly the records
    that made the budget, object by object, so the plot never holds a
    converted copy of the whole selection — only of what it draws,
    beside one block in flight. A signed component of complex data is
    taken here, per row, the same reading the waterfall takes.
    """
    by_object = {}
    for n, entry in enumerate(curves):
        data, record = entry[2]
        by_object.setdefault(id(data), (data, []))[1].append((n, record))
    out = list(curves)
    for data, wanted in by_object.values():
        for start, _stop, block in data.display_blocks(
                us, [record for _n, record in wanted]):
            for offset, row in enumerate(block):
                n = wanted[start + offset][0]
                out[n] = (*out[n][:2], _component_row(row, tag), *out[n][3:])
    return out


def _component_row(row, tag):
    if tag == 'real':
        return np.asarray(row).real
    if tag == 'imag':
        return np.asarray(row).imag
    if tag == 'phase':
        return np.degrees(np.angle(row))
    return row


#: how solid an exceedance box is. Stronger than the zone shading it
#: sits inside — a line over the abort limit is inside the red the zone
#: already paints, and has to be visible against it.
EXCEED_ALPHA = 150


def _display_factor(shown, raw):
    """What one curve's SI values are multiplied by to be the shown
    ones — a unit conversion is one factor, read off any written
    point. One when nothing is written."""
    shown = np.asarray(np.real(shown), dtype=float)
    raw = np.asarray(np.real(raw), dtype=float)
    with np.errstate(invalid='ignore'):
        good = np.isfinite(shown) & np.isfinite(raw) & (raw > 0.0)
    if not good.any():
        return 1.0
    k = int(np.flatnonzero(good)[0])
    return float(shown[k] / raw[k])


def _shade_exceedances(plot, x, y, verdicts, colors, reach):
    """Box every measured line that went outside an abort limit.

    `verdicts` is `compliance.judge`'s answer per limit — which lines
    are out, the limit's mean density over each one's cell (in the
    plot's units by now) and the stretch of each bin that was judged
    — so what is shaded is exactly what the table counted and what
    the report's page marks (Brandon, 2026-09-19: one judgment, an
    area against an area over each cell).

    Over its own bin, from the limit to the edge of the plot: red above
    the upper abort limit, blue below the lower. Which way it went is
    the first thing to know, and a stripe running off the plot is seen
    at a glance where a box a few pixels tall is not. A bin at the end
    of the specification is drawn over the part of it that was judged
    and no further.

    `reach` is (floor, ceiling) in data units — where the plot's own
    edges will be, which is what the stripe runs to. Not some multiple
    of the limit: a polygon reaching a million times past everything
    else on the plot is a thing Qt has to rasterize, and it crashed
    doing it about one run in three.

    Returns which were drawn, of 'exceed_over' and 'exceed_under', for
    the legend.

    One fill for each direction rather than one per violation. The fill
    runs the judged width, hugging the line wherever it is inside the
    limit — zero height there, so nothing shows — and opening out to
    the plot's edge only where it is not.
    """
    import pyqtgraph as pg

    drawn = set()
    all_edges = bin_edges(x)
    if all_edges.size != np.asarray(y).size + 1:
        return drawn
    y = np.asarray(y, dtype=float)
    for bound, key, over in (('abort_upper', 'exceed_over', True),
                             ('abort_lower', 'exceed_under', False)):
        verdict = verdicts.get(bound)
        if verdict is None or not verdict['lines'].any():
            continue
        out = verdict['lines']
        # only the judged stretch is drawn over: the bins the cells
        # reach, each cut to the part of it that was judged, so the
        # box at an end stops where the specification does
        marked = np.flatnonzero(out)
        first, last = int(marked[0]), int(marked[-1])
        edges = all_edges[first:last + 2].copy()
        edges[0] = max(edges[0], float(verdict['start'][first]))
        edges[-1] = min(edges[-1], float(verdict['stop'][last]))
        keep = slice(first, last + 1)
        base = np.where(out, verdict['level'], y)[keep]
        edge = reach[1] if over else reach[0]
        far = np.where(out[keep], edge, base)
        color = zone_colors(colors)[key]
        curves = []
        for values in (base, far):
            curve = plot.plot(edges, values, stepMode='center',
                              pen=pg.mkPen(None))
            curve.is_zone_edge = True     # an edge, not a measurement
            curves.append(curve)
        fill = pg.FillBetweenItem(*curves, brush=pg.mkBrush(color))
        fill.setZValue(-5)     # over the zone shading, under the curves
        fill.is_exceedance = True
        plot.addItem(fill)
        drawn.add(key)
    return drawn


def _reach(extents, log_ordinate):
    """(floor, ceiling) in data units, a little past what will be drawn.

    The view is locked to the data's own extents, so a mark reaching
    just past them fills the plot to its edges however it is zoomed.
    """
    if extents.y_lo is None:
        return 0.0, 1.0
    pad = 0.05 * (extents.y_hi - extents.y_lo) or 1.0
    low, high = extents.y_lo - pad, extents.y_hi + pad
    return (10.0 ** low, 10.0 ** high) if log_ordinate else (low, high)


def _draw_shaped(plot, x, y, shape, pen, widths=None, log_abscissa=False,
                 **kwargs):
    """Draw one curve the way its own points mean, and say what it drew.

    'steps' for a density per bin — flat across each, which is what is
    summed when a PSD is integrated, so the area drawn is the area
    counted. 'law' for a specification, filled in so the polyline
    follows the power law between its breakpoints. 'line' otherwise.

    `log_abscissa` says the plot draws x in decades, where pyqtgraph's
    automatic downsampling must stay off (below).
    """
    if shape == 'steps' and np.asarray(x).size > 1:
        edges = bin_edges(x, widths)
        # Neither clipping nor downsampling here. Both re-slice x and y
        # to the same length, and a step curve wants one more x than y
        # — pyqtgraph raises from inside the paint, where Qt eats it and
        # the plot simply comes back empty. A spectrum is thousands of
        # lines rather than the hundreds of thousands a record is, so
        # there is nothing here that needed them.
        curve = plot.plot(edges, y, stepMode='center', pen=pen, **kwargs)
        return curve, edges, y
    if shape == 'law':
        x, y = as_power_law(x, y)
    # connect='finite': see build_cmif — gaps for non-finite points, and
    # never Qt's crash-prone drawLines fast path. `gapless` is the other
    # half of that: keep the gaps down to what pyqtgraph builds a path
    # from with Qt's own API, rather than by hand-packing its bytes.
    x, y = gapless(x, y)
    curve = plot.plot(x, y, connect='finite', pen=pen, **kwargs)
    # Peak downsampling is what keeps 500k-sample records interactive.
    # pyqtgraph buckets points in index space with one bucket size taken
    # from the mean x spacing, then draws each bucket's max and min at
    # one x — which presumes evenly spaced x, as a record's or a linear
    # frequency axis is. In decades the low end has far fewer lines per
    # pixel than the mean, so each bucket there spans several pixels and
    # its max-then-min pair is drawn as a tooth: the white specification
    # line came out jagged on its left skirt and smooth on its right
    # (Brandon, 2026-09-06). A spectrum is thousands of lines rather
    # than the hundreds of thousands of a record, so on the log axis it
    # is drawn whole. Clipping does not bucket and stays.
    curve.setDownsampling(auto=not log_abscissa, method='peak')
    curve.setClipToView(True)
    return curve, x, y


def drawing_shape(data: Any, tagged: bool = False) -> str:
    """'steps' | 'law' | 'line': how this object's curve is drawn.

    **The hard rule, for every reading of the plot — 2-D and the 3-D
    waterfall alike**: a density is drawn as the energy beneath it, so
    the RMS is always the plain area under the curve on screen. A PSD,
    a CPSD's terms and an octave-band PSD (`interpolation == 'bin'`)
    draw flat across each bin — what is summed when the object is
    integrated — and a specification (`'log_log'`) follows the power
    law its breakpoints mean. Anything that is a value at a frequency
    rather than an area — an FRF, a coherence, a linear spectrum, any
    signed component — draws as the line it is; except a value that
    belongs to a whole band, which draws flat across it, whatever
    component is read: a banded coherence, or the phase of a banded
    cross term, is one number for its band, not a sample at its center
    (Brandon, 2026-09-27).

    One implementation, consulted by `build_plots` and by
    `viz.waterfall`, because a picture that disagrees with
    `Psd.area()` is a picture of a number nobody computed. `tagged`
    says a signed component (real/imag/phase) is being read, which is
    never a density — but of a banded array, still a band's value.
    """
    if data.abscissa_dim != 'frequency':
        return 'line'
    banded = getattr(data, 'bandwidth', None) is not None
    if tagged:
        return 'steps' if banded else 'line'
    shape = {'bin': 'steps', 'log_log': 'law'}.get(
        getattr(data, 'interpolation', None))
    if shape is None and banded:
        return 'steps'
    return shape or 'line'


def step_outline(centers: ArrayLike, values: ArrayLike,
                 widths: ArrayLike | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """The flat-across-each-bin trace as explicit points.

    What `stepMode='center'` has pyqtgraph draw for the 2-D plot, made
    concrete for a renderer that only takes polylines — the 3-D
    waterfall. Two points per bin on the bin's own edges (the same
    `bin_edges`), values duplicated across, so the trapezoid under the
    trace is exactly `sum(value * width)`: the RMS is the area drawn.
    `values` may be (records, lines); the outline is per row.
    """
    centers = np.asarray(centers, dtype=float)
    values = np.atleast_2d(np.asarray(values))
    if centers.size < 2:
        return centers, values[0] if values.shape[0] == 1 else values
    edges = bin_edges(centers, widths)
    x = np.repeat(edges, 2)[1:-1]
    expanded = np.repeat(values, 2, axis=1)
    return x, expanded[0] if expanded.shape[0] == 1 else expanded


def bin_edges(centers: ArrayLike,
              widths: ArrayLike | None = None) -> np.ndarray:
    """The boundaries of the bins a set of line centers stands for.

    A line of a discrete spectrum is a density over its own bin, so
    without more to go on the bin runs to the midpoint of the gap
    either side. Arithmetic midpoints, because FFT lines are evenly
    spaced in frequency.

    `widths` is for a spectrum that knows better. An octave band's
    center is the *geometric* mean of its own edges, so neither the
    midpoints between neighbors nor the center plus and minus half a
    width lands on them. Given the center and the width both, the edges
    follow exactly: with `c = sqrt(l u)` and `w = u - l`,

        u = (w + sqrt(w^2 + 4 c^2)) / 2,   l = u - w

    which is the positive root and needs no assumption about the bands
    tiling — though these do, so one edge array serves them all.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        return centers
    if widths is not None:
        widths = np.asarray(widths, dtype=float)
        if widths.size == centers.size + 1:
            # the object's own drawn edges (`Psd.bin_edges`), handed
            # over whole: a banded spectrum's end band is drawn only
            # where its source covered it (2026-09-19), which no
            # center-and-width can say
            return widths
        from ..core.octave import bin_bounds

        lower, upper = bin_bounds(centers, widths)
        return np.concatenate([lower, [upper[-1]]])
    mids = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate([[2 * centers[0] - mids[0]], mids,
                           [2 * centers[-1] - mids[-1]]])


def as_power_law(frequencies: ArrayLike, values: ArrayLike,
                 per_decade: int = POWER_LAW_POINTS
                 ) -> tuple[np.ndarray, np.ndarray]:
    """A breakpoint curve resampled onto enough points to draw as one.

    A specification's segments are power laws, and the plot's frequency
    axis is linear — so a straight line drawn between two breakpoints is
    not the curve the breakpoints mean, and for a decade-wide segment it
    runs well above it through the middle. Filled in, the polyline
    follows the law closely enough that the difference is under a pixel.

    Left alone when the points are already dense: a controller writes
    its specification on the same line grid as everything else, and
    there is nothing between two adjacent lines to fill in.

    Only over the band the specification is written for. A controller
    writes zero outside it, and zero is a value the curve cannot be
    drawn at on a log axis — it maps to -inf — while filling in across
    it gives NaN. Either way the curve comes back full of holes, and a
    gapped array is the one thing pyqtgraph cannot build a path from
    safely (see `gapless`).
    """
    from ..core.compliance import log_interpolate

    frequencies = np.asarray(frequencies, dtype=float)
    values = np.asarray(values, dtype=float)
    good = (np.isfinite(frequencies) & (frequencies > 0.0)
            & np.isfinite(values) & (values > 0.0))
    if good.sum() < 2:
        return frequencies, values
    said = np.flatnonzero(good)
    low, high = frequencies[said[0]], frequencies[said[-1]]
    decades = np.log10(high / low)
    wanted = int(decades * per_decade)
    if wanted <= good.sum():
        span = slice(said[0], said[-1] + 1)
        return frequencies[span], values[span]
    filled = np.geomspace(low, high, wanted)
    return filled, log_interpolate(filled, frequencies, values)


def gapless(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One curve's points cut to the stretch it can be drawn over.

    pyqtgraph builds a `QPainterPath` two ways. With no gaps it fills a
    `QPolygonF`, which is Qt's own API. With gaps — more than 2% of the
    points non-finite — it packs a byte buffer by hand and has Qt
    deserialize it, against a private binary layout its own docstring
    warns "may change in future versions of Qt". That is the path the
    intermittent Bus error was raised from, and the only curve on the
    plot taking it was a specification: 404 non-finite points in 1083,
    because it is written to zero outside its band.

    So a curve is cut to the run of it that says something rather than
    handed over with holes in. Nothing is bridged: the cut is the first
    and last point that can be drawn, so a curve that stops is drawn
    stopping.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        return x, y
    good = np.isfinite(x) & np.isfinite(y)
    if good.all() or good.sum() < 2:
        return x, y
    said = np.flatnonzero(good)
    span = slice(said[0], said[-1] + 1)
    return x[span], y[span]


def data_curves(plot: Any) -> list[Any]:
    """The curves on a plot that are data, not the invisible edges the
    limit shading is built from."""
    return [item for item in plot.listDataItems()
            if not getattr(item, 'is_zone_edge', False)]


def _shade_limit_zones(plot, x, bands, colors, shape='line', widths=None):
    """Fill the zones a response must not be in.

    Yellow between the warning limit and the abort limit, red beyond
    abort — above the upper pair and below the lower pair alike. With no
    abort limit the yellow runs on to the edge of the plot, and with no
    warning limit the red starts at abort; each side is read on its own,
    since a specification may carry any of the four.

    Fills are drawn behind everything and are not counted in the
    extents: the zone beyond abort has no top, so letting it vote on the
    view would zoom the data into a line. Their key is added to the
    legend after them (`_legend_zones`), from what this returns: which
    of 'warning', 'above' and 'below' were drawn.
    """
    import pyqtgraph as pg

    if shape == 'steps' and np.asarray(x).size > 1:
        # the zones step with the target they bound: a limit is a
        # density per bin like the level it is written around, and a
        # polygon through the bin centers drew a banded
        # specification's warning and abort limits as slopes under a
        # stepped target (Brandon, 2026-09-19)
        stepped = {}
        for bound, values in bands.items():
            _grid, rows = step_outline(x, np.asarray(values).real, widths)
            stepped[bound] = np.atleast_2d(rows)[0]
        bands = stepped
        x = step_outline(x, np.zeros(np.asarray(x).size), widths)[0]
    elif shape == 'law':
        # the zone a limit fills is bounded by the limit, and a limit
        # means the power law its breakpoints make just as the curve it
        # bounds does. Filled between straight lines instead, the shaded
        # region and the specification drawn through it disagree.
        dense, filled = None, {}
        for bound, values in bands.items():
            grid, on_grid = as_power_law(x, np.asarray(values).real)
            if dense is None or len(grid) > len(dense):
                dense = grid
            filled[bound] = (grid, on_grid)
        if dense is not None:
            from ..core.compliance import log_interpolate

            bands = {bound: (on_grid if len(grid) == len(dense)
                             else log_interpolate(dense, grid, on_grid))
                     for bound, (grid, on_grid) in filled.items()}
            x = dense

    shades = zone_colors(colors)
    drawn = set()

    def band(lower: np.ndarray, upper: np.ndarray, zone: str) -> None:
        """Fill between two curves, where both say something."""
        if lower is None or upper is None:
            return
        finite = (np.isfinite(lower) & np.isfinite(upper)
                  & (lower > 0.0) & (upper > 0.0))
        if finite.sum() < 2:
            # limits are per control channel, so a cross-spectral record
            # has none at all — nothing to shade rather than a fill of NaN
            return
        drawn.add(zone)
        # the fill spans the band the limits are written over, and no
        # further: a zone edge with holes in it is a gapped path, which
        # is the shape `gapless` exists to keep away from pyqtgraph
        said = np.flatnonzero(finite)
        span = slice(said[0], said[-1] + 1)
        low = plot.plot(x[span], lower[span], connect='finite')
        high = plot.plot(x[span], upper[span], connect='finite')
        for curve in (low, high):
            curve.setPen(pg.mkPen(None))
            # they are in the plot because that is what puts them in the
            # axis's log space along with everything else — a fill built
            # from curves outside it would be drawn in linear values on a
            # log axis. Marked so that anything counting the curves on a
            # plot can tell an edge from a measurement.
            curve.is_zone_edge = True
        fill = pg.FillBetweenItem(low, high, brush=pg.mkBrush(shades[zone]))
        fill.setZValue(-10)      # under the curves it describes
        plot.addItem(fill)

    def values(bound: str) -> np.ndarray | None:
        found = bands.get(bound)
        return None if found is None else np.asarray(found).real

    warning_upper, abort_upper = values('warning_upper'), values('abort_upper')
    warning_lower, abort_lower = values('warning_lower'), values('abort_lower')

    # upward: warning to abort in yellow, abort to the sky in red
    band(warning_upper, abort_upper if abort_upper is not None
         else _beyond(warning_upper, up=True), 'warning')
    band(abort_upper, _beyond(abort_upper, up=True), 'above')
    # downward: the same read the other way
    band(abort_lower if abort_lower is not None
         else _beyond(warning_lower, up=False), warning_lower, 'warning')
    band(_beyond(abort_lower, up=False), abort_lower, 'below')
    return drawn


#: how solid a limit zone's shading is: faint, since it lies behind
#: everything it describes
ZONE_ALPHA = 38


def zone_colors(colors: Mapping[str, str]) -> dict:
    """The fills of a specification's zones and of its exceedances, as
    QColors — one place, so the legend's swatch is the plot's fill.

    Yellow between warning and abort; past abort, which way: red above,
    blue below. Both were red once, which said "out of tolerance" and
    left the direction to be read off the geometry — where every other
    mark on the plot already says over in red and under in blue. The
    exceedance boxes are the same hues, stronger (`EXCEED_ALPHA`).
    """
    from PySide6.QtGui import QColor

    def shade(name, alpha):
        color = QColor(colors[name])
        color.setAlpha(alpha)
        return color

    return {'warning': shade('limit_warning', ZONE_ALPHA),
            'above': shade('exceed_over', ZONE_ALPHA),
            'below': shade('exceed_under', ZONE_ALPHA),
            'exceed_over': shade('exceed_over', EXCEED_ALPHA),
            'exceed_under': shade('exceed_under', EXCEED_ALPHA)}


def _zone_decibels(found: dict, bands: Mapping[str, Any], target: Any,
                   ordinate_dim: str) -> None:
    """Each limit's distance from the target it bounds, in decibels,
    gathered into `found` per limit — for the legend to state when it
    is one number everywhere. Ten log ten of a power quantity (a PSD's
    dimension is a square), twenty of an amplitude, as `compliance`
    reads them."""
    per = 10.0 if '**2' in str(ordinate_dim) else 20.0
    target = np.asarray(np.real(target), dtype=float)
    for bound, values in bands.items():
        values = np.asarray(np.real(values), dtype=float)
        if values.shape != target.shape:
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            db = per * np.log10(values / target)
        db = db[np.isfinite(db)]
        if db.size:
            found.setdefault(bound, []).append(db)


def _uniform_db(found: dict, bound: str) -> float | None:
    """The one value a limit sits at in decibels, or None when it moves."""
    values = found.get(bound)
    if not values:
        return None
    every = np.concatenate(values)
    if np.ptp(every) > 0.05:
        return None
    return float(np.round(np.mean(every), 1))


def _signed_db(value: float) -> str:
    return f'{value:+g} dB'.replace('-', '\u2212')


def zone_key(zones: set, exceeded: set, found: Mapping[str, Any]
             ) -> list[tuple[str, str]]:
    """What a specification's shading is called in a legend: the warning
    band, the zones above and below abort and the exceedance marks, each
    only when drawn (`zones` of 'warning', 'above', 'below'; `exceeded`
    of 'exceed_over', 'exceed_under'), with each band's decibels when the
    specification holds it at one value (`found`, from `_zone_decibels`).

    One rule for every plot that shades them — the app's (`build_plots`)
    and the report's, which must always match (Brandon, 2026-09-27).

    Returns
    -------
    list of (str, str)
        (zone, name), in the order a legend lists them.
    """
    def db_note(*bounds):
        values = [_uniform_db(found, bound) for bound in bounds]
        if any(v is None for v in values):
            return ''
        if len(values) == 2 and np.isclose(values[0], -values[1]):
            return f' (\u00b1{abs(values[0]):g} dB)'
        return ' (' + '/'.join(_signed_db(v) for v in values) + ')'

    entries = []
    if 'warning' in zones:
        entries.append(('warning', 'warning band'
                        + db_note('warning_upper', 'warning_lower')))
    if 'above' in zones:
        entries.append(('above', 'above abort' + db_note('abort_upper')))
    if 'below' in zones:
        entries.append(('below', 'below abort' + db_note('abort_lower')))
    if 'exceed_over' in exceeded:
        entries.append(('exceed_over', 'line over abort'))
    if 'exceed_under' in exceeded:
        entries.append(('exceed_under', 'line under abort'))
    return entries


def _legend_zones(legend: Any, zones: set, exceeded: set, found: dict,
                  colors: Mapping[str, str]) -> None:
    """Name the shading in the legend (`zone_key`), each swatch the
    plot's own fill. The measured curves and the specification name
    themselves as they are drawn; the fills had no entries, and a reader
    of a printed figure could not tell the yellow, red and blue apart
    without the caption (2026-09-26)."""
    import pyqtgraph as pg

    shades = zone_colors(colors)
    for zone, name in zone_key(zones, exceeded, found):
        # a swatch, not a mark on the plot: never added to the view,
        # only drawn by the legend's sample. A bar's sample is a filled
        # square, an area's key; a filled curve's is a triangle
        swatch = pg.BarGraphItem(x=[0.0], height=[1.0], width=1.0,
                                 pen=None, brush=pg.mkBrush(shades[zone]))
        swatch.is_zone_edge = True
        legend.addItem(swatch, name)


def _beyond(values, up):
    """A curve far outside `values`, for a zone with no far side."""
    if values is None:
        return None
    return values * BEYOND if up else values / BEYOND


def _application():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _mark(layout, data, marks, theme=None):
    """The frames or the shocks over every plot in `layout`.

    Locked, because a mark you can drag is a mark that reports the drag
    to somebody, and in a standalone plot there is nobody. Returns the
    overlays so the caller can keep them alive.
    """
    import pyqtgraph as pg

    from ..core.averaging import Averaging
    from ..core.data import TimeHistory

    if marks is None:
        return []
    if marks not in ('averaging', 'shocks'):
        raise ValueError(f"{marks!r} is not a reading of a time history: "
                         "'averaging' or 'shocks'")
    if not isinstance(data, TimeHistory):
        raise TypeError(f'{marks} marks belong on a time history, not on '
                        f'{type(data).__name__}')
    colors = resolve_theme(theme)
    plots = [item for item in layout.ci.items if isinstance(item, pg.PlotItem)]
    if marks == 'shocks':
        from ..core.shocks import suggest
        from .shocks import ShockOverlay

        found = tuple(data.shocks or suggest(data))
        return [ShockOverlay(plot, found, colors, locked=True)
                for plot in plots]
    from .averaging import AveragingOverlay

    samples = len(data.abscissa)
    averaging = data.averaging or Averaging.for_records(samples)
    return [AveragingOverlay(plot, averaging, data.sample_rate, samples,
                             colors, locked=True)
            for plot in plots]


def plot_data(data: DataArray, unit_system: UnitSystem | None = None,
              theme: Any = None, show: bool = True,
              title: str | None = None, size: tuple[int, int] = (1000, 700),
              path: str | os.PathLike | None = None,
              marks: str | None = None,
              **kwargs: Any) -> Any:
    """Plot a data array in a window, or to a file with `path`.

    Returns the DataPane (or the path). Complex data gets the component
    box over it — the same control the app offers — so which part is
    drawn is a choice rather than an argument. With `show=True` and no
    Qt event loop already running, this blocks until the window is
    closed.

    `marks` puts a time history's own reading over the trace, which is
    what the app's two toggles do: `'averaging'` brackets the frames a
    spectrum is averaged over, `'shocks'` brackets the events an SRS is
    computed from. Both are read from the history — whatever it carries,
    or what the detector would suggest — and neither can be dragged
    here, since there is nothing on the other end of a drag in a file.
    """
    # the pane owns which part of a complex ordinate is drawn, so the
    # draw reads it back rather than being told once at the call
    held = kwargs.pop('component', None)
    pane: list[Any] = [None]
    # the overlays are QObjects on somebody else's plot; dropped, they
    # take their items with them
    overlays: list[Any] = []

    def build(widget: Any) -> Any:
        component = (pane[0].component() if pane[0] is not None
                     else held or 'magnitude')
        build_plot(widget, data, unit_system=unit_system, theme=theme,
                   component=component, **kwargs)
        overlays.extend(_mark(widget, data, marks, theme))

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or repr(data), size=size,
                        complex_data=np.iscomplexobj(data.ordinate),
                        on_pane=lambda p: pane.__setitem__(0, p))


def save_plot(data: DataArray, path: str | os.PathLike,
              unit_system: UnitSystem | None = None, theme: Any = None,
              size: tuple[int, int] = (1200, 800),
              **kwargs: Any) -> Any:
    """Render a data array to an image file (.png or .svg), no window."""
    return plot_data(data, unit_system=unit_system, theme=theme,
                     size=size, path=path, **kwargs)


#: the resolution a standalone plot is written to a .png at, in dots per
#: inch, or None for one pixel per logical pixel (the screen's 96). For
#: print: lay a figure out at its printed size — `size` in logical pixels
#: is inches times 96 — and set this to 300; text, pens and layout scale
#: together and every font prints at its set size (2026-09-26).
EXPORT_DPI: float | None = None
#: the logical pixels in an inch, which a device ratio is counted from
LOGICAL_DPI = 96.0


def render_image(widget: Any, ratio: float = 1.0,
                 background: Any = None) -> Any:
    """A widget painted into an image at `ratio` device pixels per
    logical pixel.

    The one way a plot becomes pixels. pyqtgraph's ImageExporter scales
    the scene to a requested width, and not evenly: exported at three
    times a plot's width, legend labels came out at a third of their set
    size and QFont tick labels half again too large, while HTML titles
    were right (the band-average paper's print figures, 2026-09-26).
    Painted through a painter whose device carries the ratio, the scene
    is laid out once, at its logical size, and every item — text, pens,
    fills — is drawn scaled by the same factor, as a high-density screen
    draws it.

    Parameters
    ----------
    widget : QWidget
        Laid out already, at its logical size.
    ratio : float, default 1.0
        Device pixels per logical pixel.
    background : color, optional
        What the image is filled with first; transparent when omitted.

    Returns
    -------
    QImage
    """
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtWidgets import QGraphicsView

    image = QImage(max(1, round(widget.width() * ratio)),
                   max(1, round(widget.height() * ratio)),
                   QImage.Format.Format_ARGB32)
    image.fill(QColor(background) if background is not None
               else Qt.GlobalColor.transparent)
    if isinstance(widget, QGraphicsView):
        # The image carries the ratio and the view renders into its
        # logical rectangle, so the ratio is applied once; a target of
        # the image's full device size as well applied it twice and the
        # image held a ninth of the plot. Through the ratio, not a
        # scaled target, because pyqtgraph's point markers are pixmaps
        # drawn at the device ratio: through a scaled target a 12 px
        # marker printed at 12 device pixels, a third of its size at
        # 288 dpi (the band-average paper, 2026-09-27).
        image.setDevicePixelRatio(ratio)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    if isinstance(widget, QGraphicsView):
        restore = _widen_pens(widget.scene(), ratio)
        try:
            widget.render(painter, QRectF(0, 0, widget.width(),
                                          widget.height()),
                          widget.viewport().rect())
        finally:
            for item, pen in restore:
                item.setPen(pen)
    else:
        painter.scale(ratio, ratio)
        widget.render(painter)
    painter.end()
    return image


def _widen_pens(scene: Any, ratio: float) -> list:
    """Every cosmetic line pen in a scene widened by `ratio`, and every
    data curve to at least `EXPORT_CURVE_WIDTH` first; returns the
    (item, pen) pairs to put back.

    pyqtgraph's pens are cosmetic, and Qt draws a cosmetic pen's width
    in device pixels whatever the painter's scale or the device's ratio:
    at 288 dpi a 3 px curve printed a third as heavy as the text and
    legend around it (the band-average paper, 2026-09-27). Widened for
    the render, a line prints at its width in logical pixels, as
    everything else in the figure does.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPen

    saved, owned = [], set()
    if scene is None:
        return saved
    for item in scene.items():
        floor = 0.0
        if isinstance(item, pg.PlotDataItem):
            # through the data item, so its curve and its legend sample
            # both take the wider pen
            owned.add(id(item.curve))
            pen = item.opts.get('pen')
            if not getattr(item, 'is_zone_edge', False):
                floor = EXPORT_CURVE_WIDTH
        elif id(item) in owned:
            continue
        elif isinstance(item, pg.PlotCurveItem):
            pen = item.opts.get('pen')
        elif isinstance(item, pg.InfiniteLine):
            pen = item.pen
        elif isinstance(item, pg.AxisItem):
            pen = item.pen()
        else:
            continue
        if not isinstance(pen, QPen) or not pen.isCosmetic() or \
                pen.style() == Qt.PenStyle.NoPen:
            continue
        # width 0 is Qt's one-device-pixel hairline
        width = max(pen.widthF(), 1.0, floor) * ratio
        if width == pen.widthF():
            continue
        wider = QPen(pen)
        wider.setWidthF(width)
        saved.append((item, pen))
        item.setPen(wider)
    return saved


def save_image(widget: Any, path: str | os.PathLike, *,
               dpi: float | None = None, theme: Any = None) -> str:
    """A laid-out plot widget written to a .png, at `dpi` for print.

    The widget is drawn at its logical size scaled by ``dpi / 96``
    (`render_image`), on the theme's plot background, and the file says
    its resolution, so a figure laid out at 3.42 in (328 logical pixels)
    and saved at 300 dpi is 1026 pixels that print at 3.42 in, every
    font at its set size.

    Parameters
    ----------
    widget : QWidget
        The plot, laid out at its printed size in logical pixels.
    path : str or path-like
        Where the .png goes.
    dpi : float, optional
        Dots per inch; one pixel per logical pixel when omitted.
    theme : optional
        Whose plot background to fill with.

    Returns
    -------
    str
        The path written.
    """
    colors = resolve_theme(theme)
    ratio = 1.0 if dpi is None else float(dpi) / LOGICAL_DPI
    image = render_image(widget, ratio, colors['plot_background'])
    if dpi is not None:
        dots = round(float(dpi) / 0.0254)
        image.setDotsPerMeterX(dots)
        image.setDotsPerMeterY(dots)
    path = str(path)
    if not image.save(path):
        raise OSError(f'could not write {path}')
    return path


def _plot_window(build: Callable[[Any], Any], theme: Any = None,
                 path: str | os.PathLike | None = None,
                 show: bool = True, title: str | None = None,
                 size: tuple[int, int] = (1000, 700),
                 complex_data: bool = False,
                 on_pane: Any = None) -> Any:
    """Every standalone plot, drawn the one way: build into a widget,
    then either show it or export it.

    `build(widget)` draws. With `path` the plot is rendered to a file
    (.png or .svg) and a bare layout widget is used, since a toolbar is
    not in the picture. Shown, it comes up in the app's own data pane,
    so a plot opened from a script wears the same bar as the one in the
    window. Returns the pane, or the path. A .png is written by
    `save_image`, at `EXPORT_DPI` when that is set.
    """
    import pyqtgraph as pg

    app = _application()
    colors = resolve_theme(theme)
    pg.setConfigOption('background', colors['plot_background'])
    pg.setConfigOption('foreground', colors['plot_foreground'])
    if path is None:
        from ..gui.windows import data_window

        return data_window(build, theme=theme, title=title, size=size,
                           complex_data=complex_data, on_pane=on_pane,
                           show=show)
    import pyqtgraph.exporters

    widget = pg.GraphicsLayoutWidget(title=title or '', size=size)
    widget.resize(*size)
    build(widget)
    # the widget must be shown and events flushed, or it never gets
    # a resize and the export captures an unlaid-out scene
    widget.show()
    app.processEvents()
    path = str(path)
    if path.lower().endswith('.svg'):
        pg.exporters.SVGExporter(widget.scene()).export(path)
    else:
        # the theme's background, painted first: pyqtgraph's exporter
        # painted its own and defaulted to black, so a light-themed
        # export once came out on black in a report page that is white
        save_image(widget, path, dpi=EXPORT_DPI, theme=theme)
    widget.close()
    return path


def plot_mac(shapes: ShapeSet, other: ShapeSet | None = None, *,
             theme: Any = None, path: str | os.PathLike | None = None,
             show: bool = True, title: str | None = None,
             size: tuple[int, int] | None = None, bars: bool = False,
             screenshot: str | None = None) -> Any:
    """The MAC grid the GUI draws: `shapes` against itself, or the
    cross-MAC against `other`.

    The picture defaults to 700 tall by the grid's own shape, clamped the
    way the app clamps it, so a 139-against-8 cross-MAC renders as the
    same tall panel it is on screen rather than being stretched into a
    square. Pass `size` to say otherwise.

    `bars=True` is the plot bar's 3-D reading: the matrix as bars,
    height and color both the value. A scene renders through its own
    plotter, so it takes `screenshot=` rather than `path=`, like every
    other 3-D view.
    """
    from ..core.shapes import cross_mac

    matrix = shapes.auto_mac() if other is None else cross_mac(shapes, other)
    columns = None if other is None else other.frequency
    return plot_mac_matrix(shapes.frequency, matrix,
                           column_frequencies=columns, theme=theme,
                           path=path, show=show, title=title, size=size,
                           bars=bars, screenshot=screenshot)


def plot_mac_matrix(frequencies: ArrayLike, matrix: ArrayLike, *,
                    column_frequencies: ArrayLike | None = None,
                    theme: Any = None, path: str | os.PathLike | None = None,
                    show: bool = True, title: str | None = None,
                    size: tuple[int, int] | None = None, bars: bool = False,
                    screenshot: str | None = None) -> Any:
    """Draw a MAC matrix somebody else computed — `plot_mac` for two
    sets that share DOF names, `Project.plot_mac` for the projected
    comparison across geometries. One drawing, whoever did the sum.
    `frequencies` label the rows; `column_frequencies` the columns
    when they are another set's."""
    if bars:
        from ..viz.mac_bars import mac_bars_scene

        if screenshot is not None:
            plotter = mac_bars_scene(frequencies, matrix,
                                     column_frequencies=column_frequencies,
                                     off_screen=True, theme=theme)
            img = plotter.screenshot(screenshot)
            plotter.close()
            return img
        from ..gui.windows import scene_window

        return scene_window(
            lambda plotter: mac_bars_scene(
                frequencies, matrix, column_frequencies=column_frequencies,
                plotter=plotter, theme=theme),
            theme=theme, axis_unit='', title=title or 'MAC', show=show)
    if size is None:
        height = 700
        size = (round(height * mac_frame_ratio(*np.shape(
            np.atleast_2d(matrix)))), height)

    def build(widget: Any) -> Any:
        build_mac(widget, frequencies, matrix, theme=theme,
                  column_frequencies=column_frequencies)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or 'MAC', size=size)


def plot_cmif(frf: Frf, shapes: ShapeSet | None = None, *,
              modes: Iterable[int] | None = None,
              unit_system: UnitSystem | None = None, theme: Any = None,
              path: str | os.PathLike | None = None, show: bool = True,
              title: str | None = None,
              size: tuple[int, int] = (1000, 700)) -> Any:
    """The CMIF the fitting screen draws: the measured indicator, and
    with `shapes` the modal model's synthesis dashed over it."""
    from ..core.shapes import synthesize_overlay

    series = [('FRF', frf, None)]
    if shapes is not None:
        overlay = synthesize_overlay(shapes, frf, modes=modes)
        if overlay is not None:
            series.append(('FRF synthesized', overlay, None))

    def build(widget: Any) -> Any:
        build_cmif(widget, series, unit_system=unit_system, theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or 'CMIF', size=size)


def plot_coherence_map(coherence: _CoherenceBase, *,
                       unit_system: UnitSystem | None = None,
                       theme: Any = None,
                       path: str | os.PathLike | None = None,
                       show: bool = True, title: str | None = None,
                       size: tuple[int, int] = (1000, 700)) -> Any:
    """The coherence map: frequency across, channel down, pinned 0..1."""
    def build(widget: Any) -> Any:
        build_coherence_map(widget, [(None, coherence, None)],
                            unit_system=unit_system, theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or 'Coherence', size=size)


def build_photos(layout: Any, photos: Photos,
                 picks: Sequence[int] | None = None,
                 start_row: int = 0) -> int:
    """Photos stacked down a layout, each at its own aspect ratio with
    its name as the title.

    Drawing starts at `start_row` and the layout is not cleared, so
    several sets can stack into one layout — which is what selecting
    two Photos objects in the app does. Returns how many were drawn.
    """
    import pyqtgraph as pg
    from PySide6.QtGui import QImage

    shown = 0
    wanted = (sorted(picks) if picks is not None
              else range(photos.num_photos))
    for i in wanted:
        image = QImage.fromData(photos.images[i])
        if image.isNull():
            continue
        image = image.convertToFormat(QImage.Format.Format_RGBA8888)
        height, width = image.height(), image.width()
        rows = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
            height, image.bytesPerLine())
        array = rows[:, :width * 4].reshape(height, width, 4).copy()
        plot = layout.addPlot(row=start_row + shown, col=0)
        plot.hideAxis('left')
        plot.hideAxis('bottom')
        plot.setAspectLocked(True)
        plot.invertY(True)
        plot.setTitle(photos.names[i])
        plot.addItem(pg.ImageItem(array, axisOrder='row-major'))
        shown += 1
    return shown


def plot_photos(photos: Photos, picks: Iterable[int] | None = None, *,
                theme: Any = None, path: str | os.PathLike | None = None,
                show: bool = True, title: str | None = None,
                size: tuple[int, int] = (900, 700)) -> Any:
    """The photos, as the app shows them."""
    def build(widget: Any) -> Any:
        widget.clear()
        build_photos(widget, photos, picks)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or 'Photos', size=size)


def _as_series(items):
    """Whatever was handed in, as the (name, data, records) the builders
    take: a bare data array, or a (name, data) or (name, data, records)
    tuple, in any mixture."""
    series = []
    for item in items:
        if isinstance(item, tuple):
            name, data, *rest = item
            series.append((name, data, rest[0] if rest else None))
        else:
            series.append((None, item, None))
    return series


def plot_series(items: Sequence[tuple[str | None, Any, Sequence[int] | None]], unit_system: UnitSystem | None = None,
                theme: Any = None,
                path: str | os.PathLike | None = None,
                show: bool = True, title: str | None = None,
                size: tuple[int, int] = (1000, 700),
                component: str = 'magnitude', **kwargs: Any) -> Any:
    """Several data arrays on one figure — what selecting more than one
    object in the tree draws.

        visualdynamics.plot.plot_series([frf, synthesized], path='both.png')

    Each item is a data array, or `(name, data)`, or
    `(name, data, records)` to draw only some of its records. Curves
    group onto an axis per quantity, so an FRF and a time history stack
    rather than sharing one, and names prefix the legend.
    """
    series = _as_series(items)

    def build(widget: Any) -> Any:
        build_plots(widget, series, unit_system=unit_system, theme=theme,
                    component=component, **kwargs)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or 'Data', size=size)


def build_ratio(layout: Any, signal: Any, floor: Any,
                records: Sequence[int] | None = None,
                theme: Any = None) -> int:
    """The signal-to-noise of the louder density over the quieter, in
    decibels, one curve per shared channel — the reading of two
    selected PSDs (Brandon, 2026-08-25). Returns how many channels
    drew.

    The signal's power is the louder density less the quieter, over
    the quieter (`snr.signal_to_noise`, 2026-10-03), so a line where
    the two are equal is not 0 dB but no signal at all, and is left
    undrawn. Linear in dB on purpose, not a log axis of the linear
    ratio: the number being read *is* the decibel. A dashed line at
    0 dB marks where the signal's power equals the noise's.
    `records` restricts to the named rows of the louder side, the
    grid's own picks.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..core.snr import signal_to_noise

    colors = resolve_theme(theme)
    abscissa, rows, dofs, _dims = signal_to_noise(signal, floor)
    if records is not None:
        keep = {str(signal.response_dof[k]) for k in records}
        picked = [(row, dof) for row, dof in zip(rows, dofs)
                  if dof in keep]
        if picked:
            rows = np.asarray([row for row, _dof in picked])
            dofs = [dof for _row, dof in picked]
    plot = layout.addPlot(row=0, col=0)
    # linear frequency, like nearly every plot in this GUI — the first
    # cut logged it out of habit and read as a different instrument
    # (Brandon, 2026-08-25)
    plot.showGrid(x=True, y=True, alpha=0.2)
    legend_below(layout, plot, 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        decibels = 10.0 * np.log10(rows)
    for k, (row, dof) in enumerate(zip(decibels, dofs)):
        finite = np.isfinite(row) & (abscissa > 0)
        if not finite.any():
            continue
        plot.plot(abscissa[finite], row[finite],
                  pen=pg.mkPen(curve_color(k, colors), width=1.5),
                  name=str(dof))
    zero = pg.InfiniteLine(
        pos=0.0, angle=0,
        pen=pg.mkPen(colors['plot_foreground'], width=1,
                     style=Qt.PenStyle.DashLine))
    zero.is_zone_edge = True
    plot.addItem(zero, ignoreBounds=True)
    plot.setLabel('bottom', 'frequency [Hz]')
    plot.setLabel('left', 'signal to noise [dB]')
    return len(dofs)


def plot_ratio(signal: Any, floor: Any, *,
               records: Sequence[int] | None = None,
               unit_system: UnitSystem | None = None,
               theme: Any = None,
               path: str | os.PathLike | None = None,
               show: bool = True, title: str | None = None,
               size: tuple[int, int] = (1000, 700)) -> Any:
    """The signal-to-noise of a driven density over its ambient one,
    line by line, in decibels — the headless call for the toolbar's
    Signal to noise reading of two selected PSDs.

        visualdynamics.plot.plot_ratio(driven_psds, noise_psds,
                                       path='snr.png')

    The signal's power is the driven less the ambient, over the
    ambient (`visualdynamics.core.snr`); a line with no signal above
    the floor is not drawn.
    """
    del unit_system     # a ratio of densities converts to itself

    def build(widget: Any) -> Any:
        build_ratio(widget, signal, floor, records=records, theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or 'Signal to noise', size=size)


def plot_comparison(measured: Any, specification: Any, *,
                    pair: tuple[str, str] | None = None,
                    unit_system: UnitSystem | None = None,
                    theme: Any = None,
                    path: str | os.PathLike | None = None,
                    show: bool = True, title: str | None = None,
                    size: tuple[int, int] = (1000, 700)) -> Any:
    """One control channel against its specification, shaded.

        visualdynamics.plot.plot_comparison(psds, spec, path='control.png')

    The comparison the app draws when a specification and the PSDs it
    bounds are selected together: the measurement stepped flat across
    each bin, the requirement as the power law its breakpoints mean, and
    the ground outside the abort limits shaded — red above the upper,
    blue below the lower.

    One channel at a time, because six of them and their two dozen limit
    lines on one axis is a thicket. `pair` names which, as the DOF pair
    the dropdown lists (`('101Z+', '101Z+')`); the first the two have in
    common by default. Band it first — `psds.to_octave(6)` — to compare
    on proportional bands instead.
    """
    series = _as_series([(None, measured), (None, specification)])
    series, _dropped = bounded_by_specification(series)
    pairs = specification_pairs(series)
    if not pairs:
        raise ValueError('the specification and the measurement have no '
                         'channel in common to compare')
    if pair is not None and pair not in pairs:
        raise ValueError(f'{pair_label(pair)} is not one of the channels '
                         'they have in common: '
                         + ', '.join(pair_label(p) for p in pairs))
    drawn = only_pair(series, pairs[0] if pair is None else pair)

    def build(widget: Any) -> Any:
        build_plots(widget, drawn, unit_system=unit_system, theme=theme)

    from ..core.compliance import comparison_scale_db
    db = comparison_scale_db(specification, measured)
    note = f' — measured scaled {db:+d} dB' if db else ''
    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'Control against specification{note}',
                        size=size)


def plot_kurtosis(history: Any, *, records: Sequence[int] | None = None,
                  low: float | None = None, high: float | None = None,
                  theme: Any = None,
                  path: str | os.PathLike | None = None,
                  show: bool = True, title: str | None = None,
                  size: tuple[int, int] | None = None) -> Any:
    """How Gaussian each channel of a record is, a bar apiece.

        visualdynamics.plot.plot_kurtosis(history, path='kurtosis.png')

    Pearson kurtosis, where 3 is Gaussian: above the band the record
    carries peaks the spectrum does not predict, below it the record
    is clipped or was never random. Every channel whatever it
    measures — kurtosis is dimensionless — and the thresholds default
    to one either side of nominal.
    """
    from ..core.kurtosis import HIGH, LOW, channel_kurtosis
    from .bars import ROW_HEIGHT, kurtosis_chart

    rows = [(label, value)
            for label, value in channel_kurtosis(history, records)
            if np.isfinite(value)]
    if not rows:
        raise ValueError('no channel of this record has a shape to '
                         'describe')
    colors = resolve_theme(theme)
    if size is None:
        size = (900, ROW_HEIGHT * len(rows) + 120)

    def build(widget: Any) -> Any:
        widget.clear()
        kurtosis_chart(widget.addPlot(row=0, col=0), rows, colors,
                       low=LOW if low is None else low,
                       high=HIGH if high is None else high)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or 'Kurtosis by channel', size=size)


def plot_snr(signal: Any, floor: Any, *,
             records: Sequence[int] | None = None,
             low: float | None = None, theme: Any = None,
             path: str | os.PathLike | None = None,
             show: bool = True, title: str | None = None,
             size: tuple[int, int] | None = None) -> Any:
    """How far each channel's signal stands above its noise, a bar
    apiece: RMS signal-to-noise in dB.

        visualdynamics.plot.plot_snr(driven_psds, noise_psds,
                                     path='snr_bars.png')

    The headless call for the toolbar's RMS signal to noise reading of
    two selected PSDs. Each channel's power is the area under its
    density over the band the two share, and the reading is
    10 log10((P_driven - P_ambient) / P_ambient)
    (`visualdynamics.core.snr`). `low` is the floor a channel is
    judged against, `core.snr.THRESHOLD_DB` by default; a channel
    with no signal above its noise has no bar and is named as at the
    noise floor, and one whose ambient recorded nothing is left out.
    """
    from ..core.snr import rms_signal_to_noise
    from .bars import ROW_HEIGHT, snr_chart

    rows, _silent = rms_signal_to_noise(signal, floor, records)
    if not rows:
        raise ValueError('no shared channel recorded any ambient noise '
                         'to divide by')
    colors = resolve_theme(theme)
    if size is None:
        size = (900, ROW_HEIGHT * len(rows) + 120)

    def build(widget: Any) -> Any:
        widget.clear()
        snr_chart(widget.addPlot(row=0, col=0), rows, colors, low=low)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or 'RMS signal to noise by channel',
                        size=size)


def plot_scalogram(history: Any, channel: int = 0, *,
                   low: float | None = None, high: float | None = None,
                   per_octave: int | None = None,
                   omega0: float | None = None,
                   theme: Any = None,
                   path: str | os.PathLike | None = None,
                   show: bool = True, title: str | None = None,
                   size: tuple[int, int] | None = None) -> Any:
    """Where one channel's frequencies are, moment by moment.

        from visualdynamics.plot import plot_scalogram
        plot_scalogram(history, path='scalogram.png')

    The flat form of the app's wavelet reading: time across, frequency
    up a logarithmic axis, magnitude as color in the record's own
    units. The cone of influence is shaded, because inside it the
    picture is an artifact of where the record was cut and looks
    exactly like data. Time is held to `core.wavelet.COLUMNS` columns
    by peak-hold — each column its slice's largest magnitude, the
    reading `core.wavelet.scalogram_peaks` gives the app's own view —
    so a long record draws in bounded memory; a script that wants the
    transform itself calls `core.wavelet.scalogram`.

    Parameters
    ----------
    history : TimeHistory
        The record to read.
    channel : int, default 0
        Which channel. One at a time: a scalogram is dense enough that
        two side by side read as noise rather than as two answers.
    low, high : float, optional
        The frequency range, in Hz. Defaults span from where a handful
        of the longest wavelets still fit inside the record up to just
        under Nyquist — a range the record can actually carry
        (`core.wavelet.default_range`).
    per_octave : int, optional
        Lines per octave. Defaults to `core.wavelet.PER_OCTAVE`.
    omega0 : float, optional
        Cycles under the wavelet: the time-against-frequency trade.
        Defaults to `core.wavelet.OMEGA0`.
    theme, path, show, title, size
        As every other plot here.

    Returns
    -------
    The pane, or the path written.
    """
    from ..core import wavelet as wavelet_math
    from .scalogram import scalogram_image

    rate = history.sample_rate
    values = np.real(np.asarray(history.ordinate)[channel])
    duration = len(history.abscissa) / rate
    default_low, default_high = wavelet_math.default_range(rate, duration)
    top = high if high is not None else default_high
    bottom = low if low is not None else default_low
    frequencies = wavelet_math.log_frequencies(
        bottom, top,
        wavelet_math.PER_OCTAVE if per_octave is None else per_octave)
    frequencies = frequencies[frequencies < rate / 2.0]
    if frequencies.size < 2:
        raise ValueError('no frequencies this record can carry in that '
                         f'range; it reaches {rate / 2.0:g} Hz')
    width = wavelet_math.OMEGA0 if omega0 is None else omega0
    # held to a picture's width, each column its slice's peak — the
    # app's own reading, and the one a long record survives
    clock, magnitude = wavelet_math.scalogram_peaks(
        values, rate, frequencies, width)
    clock = clock + float(history.abscissa[0])
    colors = resolve_theme(theme)

    def build(widget: Any) -> Any:
        widget.clear()
        scalogram_image(
            widget.addPlot(row=0, col=0), magnitude, clock, frequencies,
            colors,
            label=history.ordinate_dim[channel],
            units=history.ordinate_unit[channel] or '', omega0=width)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or 'Scalogram', size=size or (900, 520))


def plot_bars(measured: Any, specification: Any, mode: str = 'error', *,
              low: float | None = None, high: float | None = None,
              theme: Any = None,
              path: str | os.PathLike | None = None,
              show: bool = True, title: str | None = None,
              size: tuple[int, int] | None = None,
              summary: str = 'over') -> Any:
    """The comparison as a bar per control channel.

        visualdynamics.plot.plot_bars(psds, spec, 'error', path='error.png')

    `mode` picks the reading, which is the choice the app's toolbar
    offers: `'error'` is the level — how far each channel's RMS sits
    from what was asked for, in dB, over the band they share — and
    `'lines'` is the shape — how much of each channel's band fell
    outside an abort limit. A channel can sit at exactly the right level
    and still be out of tolerance across half its band, which is why
    both exist. `'srs'` is the shock reading: each channel's SRS
    against its target, as a signed RMS deviation in dB.

    `low` and `high` are the thresholds, defaulting to +/-3 dB and 10%.
    The chart grows with the channel count rather than squeezing them
    in, so `size` defaults to whatever fits the channels there are.
    Below it, a key to the colors. `summary` places the count: 'over'
    the bars, in the plot's 'title' (what a narrow printed figure
    wants), or 'none'.
    """
    from ..core.compliance import (
        ERROR_DB,
        LINES_PERCENT,
        channel_errors,
        compare_all,
    )
    from .bars import (
        ROW_HEIGHT,
        error_chart,
        lines_chart,
        replication_chart,
    )

    if mode not in ('error', 'lines', 'srs'):
        raise ValueError(
            f"{mode!r} is not a reading: 'error', 'lines' or 'srs'")
    if mode == 'srs':
        # a shock spectrum is not a density, so there is no area to
        # compare: the reading is how far the curve sits from the one
        # it had to meet, RMS across the band in dB
        from ..core.compliance import srs_errors
        errors = [(f'{dof} {block}' if block else dof, value, 0.0)
                  for dof, block, value in srs_errors(specification, measured)]
    else:
        errors = channel_errors(compare_all(specification, measured))
    if not errors:
        raise ValueError('the specification and the measurement have no '
                         'channel in common to compare')
    colors = resolve_theme(theme)
    if size is None:
        size = (900, ROW_HEIGHT * len(errors) + 120)

    def build(widget: Any) -> Any:
        widget.clear()
        plot = widget.addPlot(row=0, col=0)
        if mode == 'error':
            chart = error_chart(plot, errors, colors,
                                low=-ERROR_DB if low is None else low,
                                high=ERROR_DB if high is None else high,
                                summary=summary)
        elif mode == 'srs':
            from ..core.compliance import SRS_ERROR_DB
            # both sides: the deviation is signed, and under-testing
            # is a different fault from over-testing, not a lesser one
            chart = replication_chart(
                plot, [(label, value) for label, value, _p in errors],
                colors, 'srs_rms',
                low=-SRS_ERROR_DB if low is None else low,
                high=SRS_ERROR_DB if high is None else high, summary=summary)
        else:
            chart = lines_chart(plot, errors, colors,
                                low=LINES_PERCENT if low is None else low,
                                summary=summary)
        chart.key(legend_below(widget, plot, 0, colors))

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or {
                            'error': 'RMS error by channel',
                            'srs': 'SRS deviation by channel',
                        }.get(mode, 'Band outside abort'),
                        size=size)


def plot_replication(measured: Any, specification: Any,
                     mode: str = 'waveform', *,
                     event: int | None = None, channel: str | None = None,
                     low: float | None = None, high: float | None = None,
                     theme: Any = None,
                     path: str | os.PathLike | None = None,
                     show: bool = True, title: str | None = None,
                     size: tuple[int, int] | None = None) -> Any:
    """How closely a transient replicated the waveform it was asked for.

        visualdynamics.plot.plot_replication(record, target, 'srs', path='srs.png')

    `mode` picks the reading, the same four the app's bar offers:
    `'overlay'` draws one repeat over the target it was aiming at, and
    `'waveform'`, `'srs'` and `'level'` are the three numbers — the
    difference between the waveforms as a share of the target, the
    worst deviation of the shock response spectra, and the scale alone.

    `event` is which repeat, counting from zero; left out, the bars
    report every repeat and the overlay draws the first. No repeat is
    singled out as the worst one — that is a judgment about what the
    article is for, not a measurement, and the numbers are all returned
    for the caller to make it with.

    `channel` is which control DOF the overlay draws, defaulting to the
    first. One at a time and not all of them: six measured curves over
    six targets share one set of colors, and nothing on the plot then
    says which curve is the target.
    """
    from ..core.compliance import ERROR_DB
    from ..core.replication import WAVEFORM_PERCENT, compare, event_slice
    from .bars import ROW_HEIGHT, replication_chart

    if mode not in ('overlay', 'waveform', 'srs', 'level'):
        raise ValueError(f'{mode!r} is not a reading of a replication: '
                         "'overlay', 'waveform', 'srs' or 'level'")
    if mode == 'overlay':
        chosen = 0 if event is None else int(event)
        window = event_slice(measured, specification, chosen)
        if window is None:
            raise ValueError(f'repeat {chosen + 1} is not in the record')
        dofs = list(window.response_dof)
        if channel is None:
            channel = dofs[0]
        if channel not in dofs:
            raise ValueError(f'{channel!r} is not a control channel: '
                             + ', '.join(dofs))
        wanted = list(specification.response_dof)
        return plot_series(
            [(f'Event {chosen + 1}', window, [dofs.index(channel)]),
             ('Specification', specification, [wanted.index(channel)])],
            theme=theme, path=path, show=show,
            # its own default, not None: plot_series sizes a figure and
            # passing the unset value through overrides that with nothing
            **({} if size is None else {'size': size}),
            title=title or (f'{channel}: event {chosen + 1} against its '
                            'specification'))
    rows = compare(measured, specification, frame=event, metrics=(mode,))
    shown = [(row['label'], float(row[mode])) for row in rows
             if np.isfinite(row[mode])]
    if not shown:
        raise ValueError('no control channel here can be scored against '
                         'its target')
    colors = resolve_theme(theme)
    if size is None:
        size = (900, ROW_HEIGHT * len(shown) + 120)
    if mode == 'waveform':
        bounds = (WAVEFORM_PERCENT if low is None else low, high)
    else:
        bounds = (-ERROR_DB if low is None else low,
                  ERROR_DB if high is None else high)

    def build(widget: Any) -> Any:
        widget.clear()
        plot = widget.addPlot(row=0, col=0)
        replication_chart(plot, shown, colors, mode,
                          low=bounds[0], high=bounds[1])

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'{mode} by control channel',
                        size=size)


__all__ = ['MAX_RECORDS', 'axis_label', 'build_photos', 'build_plot',
           'build_plots',
           'plot_bars', 'plot_cmif', 'plot_coherence_map', 'plot_comparison',
           'plot_data', 'plot_mac', 'plot_photos', 'plot_replication',
           'plot_series', 'save_plot']
