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


def legend_below(layout: Any, plot: Any, row: int,
                 colors: dict | None = None) -> Any:
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
                 labelTextSize='7pt', **style)
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

# categorical curve colors: readable on both light and dark backgrounds
CURVE_COLORS = [
    '#4c92d9', '#ff8c2b', '#3fb950', '#e5534b', '#a371f7',
    '#b07d62', '#e668c3', '#8b949e', '#d2c14e', '#39c5cf',
]


def curve_color(index: int) -> str:
    """The nth curve's color, wrapping. The same cycle everywhere a
    curve is drawn, so a record keeps its color between the app, a
    standalone plot and the report."""
    return CURVE_COLORS[index % len(CURVE_COLORS)]


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
                color = curve_color(index)
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
                              pen=pg.mkPen(color, width=1, style=style),
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
                   (-185.0, 185.0) if tag == 'phase'
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
            own = getattr(data, 'bin_widths', None)
            widths = own() if own is not None and getattr(
                data, 'bandwidth', None) is not None else None
            groups.setdefault(key, []).append(
                (f'{name}: {label}' if multiple and name else label, x,
                 (data, i), data.abscissa_dim == 'frequency', bands,
                 pair, follower, synthesized, shape, widths))
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
        for (label, x, values, is_frequency, bands, pair, follower,
             dashed, shape, widths) in ordered:
            magnitude = (np.abs(values) if is_frequency
                         and np.iscomplexobj(values) else values.real)
            # A specification and the response it bounds are a pair, and
            # only one pair is ever drawn, so there is nothing to tell
            # apart by color: the response takes the foreground because
            # it is what is being looked at, the specification gray
            # behind it because it is the reference.
            #
            # Alone, the specification *is* what is being looked at, so
            # it takes the foreground itself. Gray would be saying "this
            # is the reference for something", and there is nothing here
            # for it to be the reference for.
            paired = follower and pair in pair_colors
            if bounded_pairs and pair in bounded_pairs:
                color = colors['specification_curve' if follower
                               else 'response_curve']
            elif follower and pair in with_spec and not with_measurement:
                color = colors['response_curve']
            elif paired:
                color = pair_colors[pair]
            else:
                color = curve_color(index)
                index += 1
                if not follower:
                    pair_colors.setdefault(pair, color)
            style = (Qt.PenStyle.DashLine if dashed
                     else Qt.PenStyle.SolidLine)
            pen = pg.mkPen(color, width=1, style=style)
            named = {} if paired and dashed else {'name': label}
            curve, drawn_x, drawn_y = _draw_shaped(plot, x, magnitude, shape,
                                                   pen, widths=widths,
                                                   log_abscissa=log_abscissa,
                                                   **named)
            extents.add(drawn_x, drawn_y, log_ordinate, log_abscissa)
            if shape == 'steps' and not follower:
                measured_by_pair.setdefault(pair, (x, magnitude))
            if bands and follower:
                bounded.append((pair, x, bands))
            # The zones say where the limits are. Drawn as lines too,
            # four more curves per record crowd the two that are being
            # compared and the shading behind them says the same thing.
            _shade_limit_zones(plot, x, bands, colors, shape, widths)
            for values in bands.values():
                # they still set how far the view reaches, so a limit
                # above everything measured is not cropped off
                extents.add(x, values.real, log_ordinate, log_abscissa)
            del curve
            drawn += 1
        for pair, spec_x, bands in bounded:
            found = measured_by_pair.get(pair)
            if found is None:
                continue
            lines, measured = found
            written = {bound: bands.get(bound)
                       for bound in ('abort_lower', 'abort_upper')}
            # the extents are known by now: every curve is drawn, and
            # the view is about to be locked to them
            reach = _reach(extents, log_ordinate)
            _shade_exceedances(plot, lines, measured, spec_x, written,
                               colors, reach)
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


def _shade_exceedances(plot, x, y, spec_x, written, colors, reach):
    """Box every measured line that went outside an abort limit.

    Over its own bin, from the limit to the edge of the plot: red above
    the upper abort limit, blue below the lower. Which way it went is
    the first thing to know, and a stripe running off the plot is seen
    at a glance where a box a few pixels tall is not.

    A bin at the end of the specification is drawn over the part of it
    the specification covers and no further, because that is the part
    that was judged — `compliance.outside` decides which bins are out,
    so what is shaded and what is counted in the table cannot disagree.

    `reach` is (floor, ceiling) in data units — where the plot's own
    edges will be, which is what the stripe runs to. Not some multiple
    of the limit: a polygon reaching a million times past everything
    else on the plot is a thing Qt has to rasterize, and it crashed
    doing it about one run in three.

    One fill for each direction rather than one per violation. The fill
    runs the whole width, hugging the limit wherever the line is inside
    it — zero height there, so nothing shows — and opening out to the
    line only where it is not. Where the specification says nothing the
    limit is taken as the line itself, which closes the fill for the
    same reason.
    """
    import pyqtgraph as pg
    from PySide6.QtGui import QColor

    from ..core.compliance import covered, log_interpolate, outside, written_band

    all_edges = bin_edges(x)
    if all_edges.size != np.asarray(y).size + 1:
        return
    for bound, key, over in (('abort_upper', 'exceed_over', True),
                             ('abort_lower', 'exceed_under', False)):
        written_values = written.get(bound)
        if written_values is None:
            continue
        written_values = np.real(written_values)
        limit = log_interpolate(x, spec_x, written_values)
        base = np.where(np.isfinite(limit), limit, y)
        out = outside(x, y, spec_x, written_values, over)
        if not out.any():
            continue
        # Only the band is drawn over. Nothing outside it can be an
        # exceedance, so carrying those bins along adds thousands of
        # flat segments to a polygon Qt has to rasterize for nothing —
        # and the two bins the edge cuts are clipped to the part the
        # specification covers, which is the part that was judged.
        band = written_band(spec_x, written_values)
        first, last = 0, len(x) - 1
        edges = all_edges
        if band is not None:
            reached = np.flatnonzero(covered(x, *band)[2] > 0.0)
            if reached.size == 0:
                continue
            first, last = int(reached[0]), int(reached[-1])
            edges = np.clip(all_edges[first:last + 2], *band)
        keep = slice(first, last + 1)
        # to the edge of the plot where the line is out, and nowhere at
        # all where it is not: the fill runs the whole width, closed on
        # itself except over the bins that went outside
        edge = reach[1] if over else reach[0]
        far = np.where(out, edge, base)[keep]
        color = QColor(colors[key])
        color.setAlpha(EXCEED_ALPHA)
        curves = []
        for values in (base[keep], far):
            curve = plot.plot(edges, values, stepMode='center',
                              pen=pg.mkPen(None))
            curve.is_zone_edge = True     # an edge, not a measurement
            curves.append(curve)
        fill = pg.FillBetweenItem(*curves, brush=pg.mkBrush(color))
        fill.setZValue(-5)     # over the zone shading, under the curves
        fill.is_exceedance = True
        plot.addItem(fill)


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
    signed component — draws as the line it is.

    One implementation, consulted by `build_plots` and by
    `viz.waterfall`, because a picture that disagrees with
    `Psd.area()` is a picture of a number nobody computed. `tagged`
    says a signed component (real/imag/phase) is being read, which is
    never a density.
    """
    if tagged or data.abscissa_dim != 'frequency':
        return 'line'
    return {'bin': 'steps', 'log_log': 'law'}.get(
        getattr(data, 'interpolation', None), 'line')


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

    Fills are drawn behind everything, are not in the legend, and are not
    counted in the extents: the zone beyond abort has no top, so letting
    it vote on the view would zoom the data into a line.
    """
    import pyqtgraph as pg
    from PySide6.QtGui import QColor

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

    warm = QColor(colors['limit_warning'])
    warm.setAlpha(38)
    # past abort, and which way: red above, blue below. Both were red,
    # which said "out of tolerance" and left the direction to be read
    # off the geometry — where every other mark on this plot already
    # says over in red and under in blue.
    hot = QColor(colors['exceed_over'])
    hot.setAlpha(38)
    cold = QColor(colors['exceed_under'])
    cold.setAlpha(38)

    def band(lower: np.ndarray, upper: np.ndarray, brush: Any) -> None:
        """Fill between two curves, where both say something."""
        if lower is None or upper is None:
            return
        finite = (np.isfinite(lower) & np.isfinite(upper)
                  & (lower > 0.0) & (upper > 0.0))
        if finite.sum() < 2:
            # limits are per control channel, so a cross-spectral record
            # has none at all — nothing to shade rather than a fill of NaN
            return
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
        fill = pg.FillBetweenItem(low, high, brush=pg.mkBrush(brush))
        fill.setZValue(-10)      # under the curves it describes
        plot.addItem(fill)

    def values(bound: str) -> np.ndarray | None:
        found = bands.get(bound)
        return None if found is None else np.asarray(found).real

    warning_upper, abort_upper = values('warning_upper'), values('abort_upper')
    warning_lower, abort_lower = values('warning_lower'), values('abort_lower')

    # upward: warning to abort in yellow, abort to the sky in red
    band(warning_upper, abort_upper if abort_upper is not None
         else _beyond(warning_upper, up=True), warm)
    band(abort_upper, _beyond(abort_upper, up=True), hot)
    # downward: the same read the other way
    band(abort_lower if abort_lower is not None
         else _beyond(warning_lower, up=False), warning_lower, warm)
    band(_beyond(abort_lower, up=False), abort_lower, cold)


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
    window. Returns the pane, or the path.
    """
    import pyqtgraph as pg
    from PySide6.QtGui import QColor

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
        exporter = pg.exporters.SVGExporter(widget.scene())
    else:
        exporter = pg.exporters.ImageExporter(widget.scene())
        exporter.parameters()['width'] = size[0]
        # the exporter paints its own background and defaults to black,
        # whatever the widget is set to — so a light-themed export came
        # out on black, in a report page that is white
        exporter.parameters()['background'] = QColor(
            colors['plot_background'])
    exporter.export(path)
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
    """The louder density over the quieter, in decibels, one curve per
    shared channel — the signal-to-noise reading of two selected PSDs
    (Brandon, 2026-08-25). Returns how many channels drew.

    Linear in dB on purpose, not a log axis of the linear ratio: the
    number being read *is* the decibel. A dashed line at 0 dB marks
    where the two densities are equal — for a noise floor, where the
    measurement is the room. `records` restricts to the named rows of
    the louder side, the grid's own picks.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..core.data import density_ratio

    colors = resolve_theme(theme)
    abscissa, rows, dofs, _dims = density_ratio(signal, floor)
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
                  pen=pg.mkPen(curve_color(k), width=1.5),
                  name=str(dof))
    zero = pg.InfiniteLine(
        pos=0.0, angle=0,
        pen=pg.mkPen(colors['plot_foreground'], width=1,
                     style=Qt.PenStyle.DashLine))
    zero.is_zone_edge = True
    plot.addItem(zero, ignoreBounds=True)
    plot.setLabel('bottom', 'frequency [Hz]')
    plot.setLabel('left', 'ratio [dB]')
    return len(dofs)


def plot_ratio(signal: Any, floor: Any, *,
               records: Sequence[int] | None = None,
               unit_system: UnitSystem | None = None,
               theme: Any = None,
               path: str | os.PathLike | None = None,
               show: bool = True, title: str | None = None,
               size: tuple[int, int] = (1000, 700)) -> Any:
    """The ratio of two densities in decibels — the headless call for
    the toolbar's Ratio reading of two selected PSDs.

        visualdynamics.plot.plot_ratio(driven_psds, noise_psds,
                                       path='snr.png')
    """
    del unit_system     # a ratio of densities converts to itself

    def build(widget: Any) -> Any:
        build_ratio(widget, signal, floor, records=records, theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or 'Ratio', size=size)


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
              size: tuple[int, int] | None = None) -> Any:
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
            error_chart(plot, errors, colors,
                        low=-ERROR_DB if low is None else low,
                        high=ERROR_DB if high is None else high)
        elif mode == 'srs':
            from ..core.compliance import SRS_ERROR_DB
            # both sides: the deviation is signed, and under-testing
            # is a different fault from over-testing, not a lesser one
            replication_chart(plot, [(label, value)
                                     for label, value, _p in errors],
                              colors, 'srs_rms',
                              low=-SRS_ERROR_DB if low is None else low,
                              high=SRS_ERROR_DB if high is None else high)
        else:
            lines_chart(plot, errors, colors,
                        low=LINES_PERCENT if low is None else low)

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
