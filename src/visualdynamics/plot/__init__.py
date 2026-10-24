"""2D plotting with pyqtgraph.

The application and scripts draw plots with the same functions, so the
rules for how data is drawn (grouping by dimension, unit-aware labels,
log magnitude in the frequency domain, legends) are written only here.

Records are grouped by their abscissa and ordinate dimensions, one
stacked plot per group, so an object holding accelerations, forces and
voltages is drawn on axes that can be compared. Curves are clipped to
the view and, on an evenly spaced axis, peak-downsampled, which keeps
time histories of a million samples interactive. On a log frequency axis
the spacing is not even, so the downsampling is off there (see
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

    pyqtgraph does not draw a tick label that does not fit its axis, and
    gives no warning. Left to size itself, a left axis takes the width a
    number needs, and a channel name or a label such as "band-average,
    update" is not drawn while the shorter ones are. Here the width is
    measured with the axis's own font, so every label is drawn. `margin`
    is the room beyond the widest label in pixels, `LABEL_MARGIN` by
    default.
    """
    # Found first on the bar charts, and again in the band-average
    # paper's figures (2026-09-26).
    from PySide6.QtGui import QFontMetrics

    metrics = QFontMetrics(axis.font())
    widest = max((metrics.horizontalAdvance(str(label)) for label in labels),
                 default=0)
    axis.setWidth(widest + margin)


def _reading_plot(layout: Any, row: int, col: int = 0,
                  colspan: int = 1) -> Any:
    """A plot at (`row`, `col`) of a reading's layout, gridded, with no
    SI scaling.

    The units are in the axis labels, so pyqtgraph's own 'x0.001' beside
    an axis labeled in seconds would say the same thing twice. It is
    turned off before anything is drawn: turned off after, an axis drawn
    at the old scale kept its numbers and read 250 s for 0.25 s.
    """
    plot = layout.addPlot(row=row, col=col, colspan=colspan)
    plot.showGrid(x=True, y=True, alpha=0.2)
    for side in ('left', 'bottom'):
        plot.getAxis(side).enableAutoSIPrefix(False)
    return plot


def _long_trace(plot: Any, x: Any, y: Any, pen: Any,
                name: str | None = None) -> Any:
    """A record-length trace on `plot`, drawn at a picture's width:
    decimated to its peaks and clipped to the view, as every time
    history here is. Drawn whole, a million-sample trace cost over a
    second a repaint, on every hover and resize."""
    curve = plot.plot(x, y, pen=pen, name=name)
    curve.setDownsampling(auto=True, method='peak')
    curve.setClipToView(True)
    return curve


def legend_below(layout: Any, plot: Any, row: int,
                 colors: dict | None = None,
                 text_size: str | None = None) -> Any:
    """Add a legend in its own layout row under the plot; return it.

    pyqtgraph's default legend is placed inside the view, in a corner,
    and on a plot with much content it covers the data (a
    specification's bands, an FRF's peaks) so that neither can be read.
    Under the plot, the plot keeps its whole area, and the names are
    laid out in rows no wider than the axes. The legend is set as
    `plot.legend`, so every curve drawn with ``name=`` is added to it as
    before.

    Plots are placed in the even rows of a layout (``2 * row``) and
    their legends in the odd rows. The window's row walker looks for
    ``series_key``, which a legend does not have, so it skips them.

    `colors` is a resolved theme, used for the label color, and
    `text_size` is the labels' size, `LEGEND_TEXT_SIZE` when omitted.
    """
    # Brandon, 2026-09-01: the legend floating inside the view landed on
    # the data, and neither could be read.
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

        def __init__(self, view, host, **kwargs):
            super().__init__(**kwargs)
            # the layout the row is placed in, held rather than asked
            # for: `parentItem()` called while a nested panel was being
            # torn down wrapped the half-destroyed layout and corrupted
            # the heap when the figure closed (2026-10-24)
            self._host = host
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
            import shiboken6
            from PySide6.QtCore import QRectF

            rect = args[0] if len(args) == 1 else QRectF(*args)
            parent = self._host
            # A figure closing deletes its layouts first, and Qt lays the
            # row out once more on the way down. Mapped through a layout
            # already deleted, a legend in a nested panel crashed the
            # interpreter (2026-10-24); a row being torn down has no
            # axes to center on.
            if shiboken6.isValid(parent) and shiboken6.isValid(self._view) \
                    and self._view.scene() is not None:
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
    host = getattr(layout, 'ci', layout)
    legend = Row(plot.getViewBox(), host, colCount=LEGEND_COLUMNS,
                 offset=None, labelTextSize=text_size or LEGEND_TEXT_SIZE,
                 **style)
    layout.addItem(legend, row=2 * row + 1, col=0)
    host.layout.setAlignment(legend, Qt.AlignmentFlag.AlignHCenter)
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

    The ratio is held between 1 / `MAC_ASPECT_CLAMP` and
    `MAC_ASPECT_CLAMP`. The frame outside the plot is sized from this
    value, and the aspect lock inside it uses the same clamp, so the two
    agree on whether the cells are square. A frame narrower than the
    grid's shape with an aspect lock inside it would push columns off
    the screen.
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
    """The curve palette for a theme's plot background:
    `CURVE_COLORS_LIGHT` on a light background, `CURVE_COLORS` on a dark
    one or when no theme is given.
    """
    if colors is None:
        return CURVE_COLORS
    background = str(colors['plot_background']).lstrip('#')
    red, green, blue = (int(background[i:i + 2], 16) for i in (0, 2, 4))
    light = 0.299 * red + 0.587 * green + 0.114 * blue > 127
    return CURVE_COLORS_LIGHT if light else CURVE_COLORS


def curve_color(index: int, colors: Mapping[str, str] | None = None) -> str:
    """The color of the nth curve, wrapping around the palette.

    The same cycle is used everywhere a curve is drawn, so a record has
    the same color in the app, in a standalone plot and in the report.
    `colors` (a resolved theme) picks the light or the dark palette.
    """
    palette = curve_colors(colors)
    return palette[index % len(palette)]


def axis_label(dimension: str, unit_system: UnitSystem,
                hint: str | None = None) -> str:
    """Axis label for a dimension, or a note when units are undefined.

    The unit is written as HTML, which pyqtgraph renders, so a quotient
    dimension is shown as a stacked fraction and not as '(in/s**2)/lbf'.

    A `hint` is the quantity the source named without giving its units.
    The label includes it, as in 'acceleration [units undefined]',
    because that is more useful than an axis with no name at all.
    """
    if dimension == UNKNOWN:
        return f'{hint} [units undefined]' if hint else 'units undefined'
    return unit_system.label_html(dimension) or '-'


def build_plot(layout: Any, data: Any, unit_system: UnitSystem | None = None,
               theme: Any = None, max_records: int = MAX_RECORDS,
               records: Sequence[int] | None = None,
               component: str = 'magnitude') -> tuple[int, int]:
    """Draw one data array (or its `records`) into `layout`.

    This is `build_plots` with a single entry, and its other arguments
    and its return value are the same.
    """
    return build_plots(layout, [(None, data, records)],
                       unit_system=unit_system, theme=theme,
                       max_records=max_records, component=component)


def background_brush(colors: Mapping[str, str]) -> Any:
    """The plot background: the same gradient as the 3D view.

    The gradient is anchored to the device and not to the scene, so it
    does not move while the data is panned and zoomed.
    """
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QBrush, QColor, QLinearGradient

    gradient = QLinearGradient(QPointF(0, 0), QPointF(0, 1))
    gradient.setCoordinateMode(QLinearGradient.CoordinateMode.StretchToDeviceMode)
    gradient.setColorAt(0.0, QColor(colors['plot_background_top']))
    gradient.setColorAt(1.0, QColor(colors['plot_background']))
    return QBrush(gradient)


def _ground(layout: Any, colors: Mapping[str, str]) -> None:
    """Paint the plot background on a layout that owns one.

    A panel nested in a figure (``layout.addLayout(row, col)``) has no
    background of its own; the figure it sits in paints the ground
    under it, so the builders draw into a panel as they draw into a
    whole figure.
    """
    # the builders called setBackground unconditionally, which only the
    # layout widget has, so a figure of two panels side by side could
    # not be drawn at all and had to be stitched from two images
    if hasattr(layout, 'setBackground'):
        layout.setBackground(background_brush(colors))


def display_limits(data: Any,
                   unit_system: UnitSystem) -> dict[str, np.ndarray]:
    """A specification's limit curves in display units, or an empty dict.

    Any object with `limits` brings bounds with it, and anything else
    has none. The check is on the attribute and not on the type, so the
    plot does not need to know about the Specification class.
    """
    limits = getattr(data, 'limits', None)
    if not limits:
        return {}
    return {bound: data.display_limit(bound, unit_system) for bound in limits}


def curve_budget(limit: int, needs: Sequence[int]) -> list[int]:
    """How many curves each plot may draw, sharing one limit among them.

    The limit is divided evenly. A plot that needs less than an equal
    share gives the rest to the others, so a two-channel plot beside a
    large one costs the large one almost nothing. When fewer curves are
    left than plots that want them, they are given one each, in order.
    Returns one count per entry of `needs`.
    """
    # First come, first served let the first plot spend the whole budget
    # and left the next one empty: selecting a spectrum and a time
    # history drew 45 curves in one plot and five in the other.
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
    """Cut PSDs to the records a specification bounds, scaled to it.

    A specification is usually written for autospectra only, while a
    measured CPSD has every cross term as well. Drawn together, the 30
    cross terms of a 6-channel CPSD are 30 curves with no limit near
    them. They are not wrong, but they are not the comparison asked for,
    and they hide the six that are.

    So when a specification is plotted with other PSDs, both are cut to
    the records they have in common: the same DOF pair and the same
    quantity, where a record with undefined units matches any quantity.
    Anything that is not a PSD is left alone, since a specification says
    nothing about a time history. When the two have nothing in common,
    the series is returned unchanged, so the mismatch can be seen.

    The measured spectra are also drawn scaled to the specification when
    a scaling applies, as is standard for a run captured below the 0 dB
    level (`compliance.comparison_scale_db`, held on the object or
    detected in whole dB). The scaling is applied only to a copy used
    for the drawing. The object's own values are never changed, and a
    scaled curve's legend name gives the scale, as in 'run (-6 dB)'.
    `scales` maps an entry's name to its scale in dB, for a caller that
    has worked the scale out with more information: the app knows that
    an octave PSD and the PSD it was banded from share one scale.

    Returns (series, dropped): the restricted series, and the number of
    records left out.
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
    """A copy of a spectrum with a comparison scale of `db` dB applied.

    The copy is shallow and shares everything except the values, because
    the object itself must not change: the scale is applied to the drawn
    comparison and nowhere else. The copy's `scale_db` is 0, so nothing
    later scales it again.
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
    """The DOF pairs drawn one at a time, in the specification's order.

    With a measurement, these are the pairs both have. A specification
    with its response is a comparison, and six such comparisons on one
    axis are too crowded to read.

    A specification on its own is offered one channel at a time as well,
    because six targets and their two dozen limit curves are just as
    hard to read together. Then every pair the specification has is
    returned.
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
    """The series cut to these DOF pairs; other data is left alone.

    Anything that is not a PSD is kept as it is. This is the plural of
    `only_pair`, for a table under the plot that can select several
    channels where a drop-down selects one.
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
    """The series cut to one DOF pair; other data is left alone.

    Anything that is not a PSD is kept as it is.
    """
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
    """How a DOF pair is written in a label.

    The DOF alone when it is its own reference, otherwise
    'response/reference'.
    """
    return pair[0] if pair[0] == pair[1] else f'{pair[0]}/{pair[1]}'


# how many channel labels a map's y-axis can carry before they collide
MAP_LABELS = 24


def build_coherence_map(layout: Any, series: Sequence[tuple[str | None, Any, Sequence[int] | None]],
                        unit_system: UnitSystem | None = None,
                        theme: Any = None) -> tuple[int, int]:
    """Draw every channel's coherence as a map, frequency by channel.

    Frequency runs across and channels run down.

    A line plot shows whether one channel is good, but 339 of them
    cannot be read at once, and the curve budget means most would not be
    drawn. As a map the same data shows which channels are poor and
    where: a poor channel is a dark row, and a poor frequency band is a
    dark column.

    Coherence is a bounded ratio, so the color scale is fixed at 0 to 1
    and not fitted to the data. A scale that changed with the selection
    would make a good channel look poor beside a better one.

    Returns (rows_drawn, rows_requested). Nothing is dropped, so the two
    are equal.
    """
    import numpy as np
    import pyqtgraph as pg

    us = unit_system or DEFAULT_SYSTEM
    colors = resolve_theme(theme)
    layout.clear()
    _ground(layout, colors)

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
    """The CMIF's singular values (k, frequencies) and display abscissa.

    The records are assembled into the response-by-reference matrix they
    belong to, with a missing pair set to zero (the usual practical
    treatment of an incomplete matrix), and a singular value
    decomposition is taken at every frequency line. The largest singular
    value peaks at every mode. A second singular value peaking at the
    same frequency indicates a repeated root.
    """
    from ..core.matrix_checks import frf_matrix

    us = unit_system or DEFAULT_SYSTEM
    wanted = (list(range(data.num_records)) if records is None
              else [int(i) for i in records])
    # assembled where the conditioning reading assembles its matrix; a
    # missing pair stays zero here, where the reading refuses it
    _rows, _columns, matrix, _filled = frf_matrix(
        data, wanted, data.display_ordinate(us, wanted))
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
    """Draw a MAC matrix as a viridis grid; return the matrix's shape.

    Rows and columns are modes, labeled by their frequencies. With
    `column_frequencies` the grid is a cross-MAC, rows from one set and
    columns from the other, and it is rectangular when the counts
    differ. The color scale is fixed at 0 to 1, as in the coherence map,
    because a MAC is a bounded ratio. The values are shown by color
    only.
    """
    # Numbers in each cell were tried and made the grid too busy to read
    # at a glance.
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
        # the abscissa where something is drawn, never every line a
        # curve is carried on: a specification written as a controller
        # writes one — every line to Nyquist, zero outside its band —
        # drew 0 to 2048 Hz around a 50 to 1500 Hz band, the zeros
        # invisible on the log axis but counted (Brandon, 2026-10-12;
        # the waterfall's `drawn_span`, 2026-10-09). A step curve's x is
        # its bins' edges, one more than its levels
        drawable = np.isfinite(y)
        if log_ordinate:
            drawable &= np.where(drawable, y, 0.0) > 0
        if x.size == y.size + 1:
            shown = np.concatenate([x[:-1][drawable], x[1:][drawable]])
        elif x.size == y.size:
            shown = x[drawable]
        else:
            shown = x
        finite_x = shown[np.isfinite(shown)]
        finite_y = y[drawable]
        if log_ordinate:
            finite_y = np.log10(finite_y)
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
    """Draw a dashed marker, with a staggered label, at each mode.

    Each marker is at a mode's frequency, and the labels are at
    staggered heights so clusters can be read. These are the fitting
    screen's markers, drawn wherever a modal synthesis is plotted. With
    no `foreground` they are gray, as on the fitting screen. The
    resynthesis overlays pass the theme's foreground instead (white on
    the dark theme, black on the light), so the markers can be told from
    the grid.
    """
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
    """Draw the Complex Mode Indicator Function of each FRF set.

    `series` is a list of (name, data, records), as for `build_plots`.

    There is one curve per singular value, log magnitude against
    frequency. A synthesized overlay's CMIF is drawn dashed in the
    measured set's colors, singular value by singular value, so the
    modal model's indicator can be compared with the measurement's.
    Singular values that are numerically zero (a synthesis truncated
    below the reference count is rank-deficient) are left out instead of
    being drawn twenty decades down. The axes are set from the measured
    curves only, and the mode frequencies of a synthesis are marked.

    Returns the number of curves drawn.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    us = unit_system or DEFAULT_SYSTEM
    colors = resolve_theme(theme)
    layout.clear()
    _ground(layout, colors)
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

    `series` is a list of (name, data, records), with `records` None for
    every record. Curves are grouped into one plot for each combination
    of abscissa dimension, ordinate dimension and axis scale, so
    selecting an FRF and a time history together stacks them on separate
    axes instead of sharing one. Names prefix the legend entries when
    more than one object is drawn. PSDs drawn with a specification are
    first restricted and scaled by `bounded_by_specification`. At most
    `max_records` curves are drawn in all, shared among the plots by
    `curve_budget`.

    `component` picks how complex frequency data is drawn: 'magnitude'
    (log ordinate, the default), 'real', 'imag', or 'phase' (degrees, on
    an axis from -200 to 200 labeled every 90 degrees from -180 to 180).
    Real data ignores it.

    Returns (curves_drawn, curves_requested).
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..core.data import Specification

    series, _dropped = bounded_by_specification(series)
    us = unit_system or DEFAULT_SYSTEM
    colors = resolve_theme(theme)
    layout.clear()
    _ground(layout, colors)

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
                                                   log_ordinate=log_ordinate,
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
                verdict['level'] = verdict['level'] * display_factor(
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


def display_factor(shown, raw):
    """The factor that converts one curve's SI values to displayed ones.

    A unit conversion is one factor, so it is read from the first point
    where both values are finite and the SI value is positive. Returns 1
    when there is no such point. The plot and the report page both scale
    a compliance verdict's level by it.
    """
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
                 log_ordinate=False, **kwargs):
    """Draw one curve the way its own points mean, and say what it drew.

    'steps' for a density per bin — flat across each, which is what is
    summed when a PSD is integrated, so the area drawn is the area
    counted. 'law' for a specification, filled in so the polyline
    follows the power law between its breakpoints. 'line' otherwise.

    `log_abscissa` says the plot draws x in decades, where pyqtgraph's
    automatic downsampling must stay off (below); `log_ordinate` that y
    is drawn in decades, where a zero is no value (`gapless`).
    """
    if shape == 'steps' and np.asarray(x).size > 1:
        # cut to the bins that have values, as `gapless` cuts a line
        # below. This branch returned before that cut, and a
        # controller's specification is drawn stepped: NaN outside its
        # band, 299 empty bins of 1,025, so pyqtgraph hand-packed the
        # stepped path, the Bus error's route (2026-10-19, a worker
        # crashed in `test_the_bounds_are_shaded_rather_than_drawn`
        # once in four gates; the probe of test_path_safety found the
        # eight paths)
        x, y, widths = _finite_bins(x, y, widths, log_ordinate)
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
    x, y = gapless(x, y, log_ordinate)
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
    """How this object's curve is drawn: 'steps', 'law' or 'line'.

    The rule, for the 2-D plot and the 3-D waterfall alike, is that a
    density is drawn so that its RMS is the area under the curve on
    screen. A PSD, a CPSD's terms and an octave-band PSD (`interpolation
    == 'bin'`) are drawn flat across each bin ('steps'), which is what
    is summed when the object is integrated, and a specification
    (`interpolation == 'log_log'`) follows the power law between its
    breakpoints ('law'). Anything that is a value at a frequency and not
    an area, such as an FRF, a coherence, a linear spectrum or any
    signed component, is drawn as a line ('line'). The exception is a
    value that belongs to a whole band, which is drawn flat across the
    band whatever component is read: a banded coherence, or the phase of
    a banded cross term, is one number for its band and not a sample at
    its center. Data that is not on a frequency axis is always a line.

    There is one implementation, used by `build_plots` and by
    `viz.waterfall`, because a curve that disagrees with `Psd.area()`
    shows a number nobody computed. `tagged` says a signed component
    (real, imag or phase) is being drawn, which is never a density,
    though for a banded array it is still a band's value.
    """
    # Brandon, 2026-09-27: a banded value is one number for its band and
    # is drawn flat across it, whatever component is read.
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

    This is what `stepMode='center'` has pyqtgraph draw in the 2-D plot,
    written out for a renderer that only takes polylines, the 3-D
    waterfall. There are two points per bin, on the bin's edges (from
    `bin_edges`), with the value repeated, so the area under the trace
    is exactly `sum(value * width)` and the RMS is the area drawn.
    `values` may be (records, lines), in which case each row is
    outlined. Returns (x, values).
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
    """The edges of the bins that a set of line centers represents.

    A line of a discrete spectrum is a density over its own bin, so with
    no other information each bin runs to the midpoint of the gap on
    either side, and the outer edges are as far beyond the end lines as
    the nearest midpoints are inside them. Arithmetic midpoints are used
    because FFT lines are evenly spaced in frequency.

    `widths` is for a spectrum that knows its bins. One more value than
    there are centers is taken as the edges themselves
    (`Psd.bin_edges`). Otherwise `widths` is each band's width. An
    octave band's center is the geometric mean of its edges, so neither
    the midpoints between neighbors nor the center plus and minus half a
    width gives them. From the center and the width the edges follow
    exactly: with `c = sqrt(l u)` and `w = u - l`,

        u = (w + sqrt(w^2 + 4 c^2)) / 2,   l = u - w

    which is the positive root and does not assume the bands tile. These
    bands do tile, so one array of edges serves them all.
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
    """A breakpoint curve resampled densely enough to draw as a curve.

    A specification's segments are power laws, and the plot's frequency
    axis is linear, so a straight line drawn between two breakpoints is
    not the curve the breakpoints define. For a decade-wide segment it
    runs well above that curve through the middle. Resampled at
    `per_decade` points per decade (`POWER_LAW_POINTS` by default), the
    polyline is within a pixel of the power law.

    Points already that dense are not resampled: a controller writes its
    specification on the same line grid as everything else, and there is
    nothing between two adjacent lines to fill in.

    Only the band the specification is written for is returned. A
    controller writes zero outside it, and zero cannot be drawn on a log
    axis (it maps to -inf), while interpolating across it gives NaN.
    Either way the curve would have gaps, and pyqtgraph cannot safely
    build a path from an array with gaps (see `gapless`). With fewer
    than two points that can be drawn, the input is returned unchanged.
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


def _drawable(x, y, log_ordinate):
    """Which points can be drawn: finite, and positive where the
    ordinate is in decades, since a zero there is minus infinity once
    pyqtgraph has taken its logarithm, and counts as a hole in the
    path. A controller writes its specification to zero outside its
    band."""
    good = np.isfinite(x) & np.isfinite(y)
    if log_ordinate:
        good &= y > 0.0
    return good


def _finite_bins(x, y, widths, log_ordinate=False):
    """A stepped curve's bins cut to the run that has values, first
    drawable to last (`_drawable`), with the widths cut to match:
    `widths` is one per bin, or the bin edges, one more than the bins
    (`bin_edges`)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        return x, y, widths
    good = _drawable(x, y, log_ordinate)
    if good.all() or good.sum() < 2:
        return x, y, widths
    said = np.flatnonzero(good)
    lo, hi = int(said[0]), int(said[-1]) + 1
    if widths is not None:
        widths = np.asarray(widths, dtype=float)
        if widths.size == x.size + 1:
            widths = widths[lo:hi + 1]
        elif widths.size == x.size:
            widths = widths[lo:hi]
    return x[lo:hi], y[lo:hi], widths


def gapless(x: np.ndarray, y: np.ndarray,
            log_ordinate: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """One curve's points cut to the span of its finite points.

    The span runs from the first finite point to the last; with
    `log_ordinate`, a point at or below zero is not finite either, as
    it is not once its logarithm is taken.

    pyqtgraph builds the path for a curve with more than 2 % of its
    points non-finite by packing a byte buffer itself against a private
    Qt binary layout, and that way of building a path is not reliable.
    With no gaps it uses Qt's own API. A specification is written as
    zero outside its band, so it is the curve most likely to have gaps.

    So a curve is cut to the part of it that has values instead of being
    passed with gaps. Nothing is bridged: the cut is at the first and
    last points that can be drawn, so a curve that stops is drawn
    stopping. Non-finite points between them are kept. Arrays of
    different lengths, or with fewer than two finite points, are
    returned unchanged.
    """
    # With gaps pyqtgraph packs a byte buffer by hand and has Qt
    # deserialize it, against a private binary layout its own docstring
    # warns "may change in future versions of Qt". That is the path the
    # intermittent Bus error was raised from, and the only curve on the
    # plot taking it was a specification: 404 non-finite points in 1083,
    # because it is written to zero outside its band.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        return x, y
    good = _drawable(x, y, log_ordinate)
    if good.all() or good.sum() < 2:
        return x, y
    said = np.flatnonzero(good)
    span = slice(said[0], said[-1] + 1)
    return x[span], y[span]


def data_curves(plot: Any) -> list[Any]:
    """The data curves on a plot, without the limit shading's edges.

    This leaves out the invisible curves the limit shading is built
    from, and anything else marked `is_zone_edge`.
    """
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
    """The fills of a specification's zones and exceedances, as QColors.

    Yellow between the warning and abort limits, and past the abort
    limits red above and blue below. The exceedance boxes are the same
    colors, more opaque (`EXCEED_ALPHA`). The plot and its legend both
    take their colors from here, so a legend's swatch matches the fill.
    Returns a dict with the keys 'warning', 'above', 'below',
    'exceed_over' and 'exceed_under'.
    """
    # Both sides past abort were red once, which said "out of tolerance"
    # and left the direction to be read off the geometry, where every
    # other mark on the plot already says over in red and under in blue.
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
    """The legend entries for a specification's shading.

    The entries are the warning band, the zones above and below abort,
    and the exceedance marks, each only when it was drawn: `zones` holds
    any of 'warning', 'above' and 'below', and `exceeded` any of
    'exceed_over' and 'exceed_under'. When the specification holds a
    band at one value, its level in decibels is added to the name, as in
    'warning band (±3 dB)'. `found` is the decibel values gathered by
    `_zone_decibels`.

    The app's plots (`build_plots`) and the report's both use this, so
    their legends match.

    Returns
    -------
    list of (str, str)
        (zone, name), in the order a legend lists them.
    """
    # Brandon, 2026-09-27: the app's legend and the report's must always
    # match.
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
    """A Qt application, made if a script has none — a plot rendered to a
    file, and a movie being encoded, both need one.

    A widgets application, not a QGuiApplication: Qt keeps the first
    one made for the life of the process, and a script that later opens
    a window would abort on its first widget.
    """
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def octave_preview(items: Iterable[Any], steps: tuple[Any, Any, Any],
                   colors: Mapping[str, str],
                   records: Sequence[int] | None = None
                   ) -> list[tuple[Any, Any]]:
    """Draw an octave-band conversion as steps over its narrowband.

    The steps are drawn over the narrowband spectrum they integrate, and
    the (plot, curve) pairs drawn are returned. Steps are drawn only on
    plots whose quantity matches, so a mixed object's force bands are
    not drawn across its acceleration axes, and only for `records` when
    given, so the steps match what the plot shows. `steps` is the result
    of `viz.marks.octave_steps`, banded once by the caller, which reads
    the band count from the same banding. The window's octave view and
    `plot_data(marks='octave')` both use this.
    """
    import pyqtgraph as pg

    banded, x, rows = steps
    wanted = (range(rows.shape[0]) if records is None
              else [int(i) for i in records])
    pen = pg.mkPen(colors['filter_preview'], width=2)
    drawn = []
    for item in items:
        if not isinstance(item, pg.PlotItem):
            continue
        key = getattr(item, 'series_key', None)
        for k in wanted:
            if key is not None and (banded.ordinate_dim[k],
                                    banded.dimension_hint[k]) \
                    != (key[1], key[2]):
                continue
            curve = pg.PlotDataItem(x, rows[k], pen=pen)
            curve.setZValue(20)
            curve.is_zone_edge = True     # a preview, not data
            item.addItem(curve, ignoreBounds=True)
            drawn.append((item, curve))
    return drawn


#: how each confidence level's edges are dashed, in the order the levels
#: are given: told apart by the dash as well as by the depth of their
#: shading, since both bands are one color
BAND_STYLES = ('dash', 'dot', 'dashdot')


def _finite_runs(finite: np.ndarray) -> list[slice]:
    """The stretches of two or more points where `finite` holds."""
    padded = np.concatenate(([False], finite, [False]))
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [slice(start, stop) for start, stop
            in zip(changes[::2], changes[1::2]) if stop - start >= 2]


def confidence_preview(items: Iterable[Any], psd: Any,
                       colors: Mapping[str, str],
                       records: Sequence[int] | None = None,
                       levels: Sequence[float] | None = None,
                       unit_system: UnitSystem | None = None,
                       dof: float | Sequence[float] | None = None
                       ) -> list[tuple[Any, Any]]:
    """Draw the chi-square confidence band of an averaged PSD over it.

    For each record and each level (`core.averaging.CONFIDENCE_LEVELS`,
    90 and 95 percent, by default), the band from `Psd.confidence_band`
    is shaded between its two edges, and the edges are drawn as steps
    the way the PSD itself is, so each line's band sits on its own bin.
    The bands share one color, the specification's gray, and the
    narrower one is shaded over the wider, so the inner band reads
    darker; the edges of each level are dashed their own way
    (`BAND_STYLES`) and named in the legend with the confidence and the
    degrees of freedom the band was worked out at, '90 % confidence,
    225.9 DOF'. The band is drawn behind the data and widens the
    plot's limits to take it in, so its top edge is not cut off at the
    highest peak.

    Bands are drawn only on plots whose quantity matches, and only for
    `records` when given, as `octave_preview` draws its steps. Returns
    the (plot, item) pairs added, edges and fills, so the caller can
    take them away again. The spectrum's own degrees of freedom are
    used (`Psd.statistical_dof`) unless `dof` is given, one number for
    every record or one per record, as for a spectrum read from a file
    whose count is known from the system that averaged it; a spectrum
    with neither is refused with the `ValueError` `Psd.confidence_band`
    raises.
    """
    import pyqtgraph as pg
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor

    from ..core.averaging import CONFIDENCE_LEVELS, dof_text

    levels = list(CONFIDENCE_LEVELS if levels is None else levels)
    us = unit_system or DEFAULT_SYSTEM
    wanted = (range(psd.num_records) if records is None
              else [int(i) for i in records])
    widths = (psd.bin_edges() if getattr(psd, 'bandwidth', None)
              is not None else None)
    dashes = {'dash': Qt.PenStyle.DashLine, 'dot': Qt.PenStyle.DotLine,
              'dashdot': Qt.PenStyle.DashDotLine}
    # each record's count, as the reading states it, for the legend
    counts = [band.dof for band in psd.confidence_reading(levels[0], dof)]
    bands = []
    for n, level in enumerate(levels):
        lower, upper = psd.confidence_band(level, dof)
        bands.append((level, BAND_STYLES[n % len(BAND_STYLES)],
                      lower.display_ordinate(us),
                      upper.display_ordinate(us)))
    # the widest first, so the narrower band's shading lies over it
    bands.sort(key=lambda band: band[0], reverse=True)
    # the specification's gray, not a curve color: the band is a
    # statement about the estimate rather than another measurement,
    # and gray is the one hue no curve in the cycle takes
    color = colors['specification_curve']
    shade = QColor(color)
    shade.setAlpha(ZONE_ALPHA)
    drawn = []
    for item in items:
        if not isinstance(item, pg.PlotItem):
            continue
        key = getattr(item, 'series_key', None)
        log_y = bool(key[3]) if key is not None else False
        reach = []
        banded = []
        for k in wanted:
            if key is not None and (psd.ordinate_dim[k],
                                    psd.dimension_hint[k]) \
                    != (key[1], key[2]):
                continue
            for level, style, lows, highs in bands:
                pen = pg.mkPen(color, width=1, style=dashes[style])
                x, low = step_outline(psd.abscissa, np.real(lows[k]), widths)
                _x, high = step_outline(psd.abscissa, np.real(highs[k]),
                                        widths)
                # a line at zero has no place on a log axis, and its band
                # is zero wide anyway
                low = np.where(low > 0.0, low, np.nan)
                high = np.where(high > 0.0, high, np.nan)
                finite = np.isfinite(low) & np.isfinite(high)
                if not finite.any():
                    # a record that is zero throughout has no band to
                    # draw, and nanmin of it would only warn
                    continue
                # drawn run by run between the lines that say something:
                # one zero line once took the whole record's shading
                # with it (2026-10-16), and a fill across the hole would
                # be the gapped path the limit zones keep away from
                for run in _finite_runs(finite):
                    edges = []
                    for values in (low, high):
                        edge = pg.PlotDataItem(x[run], values[run],
                                               pen=pen)
                        edge.setZValue(-2)    # behind the data it bounds
                        edge.is_zone_edge = True  # a band, not a curve
                        item.addItem(edge)
                        edges.append(edge)
                        drawn.append((item, edge))
                    fill = pg.FillBetweenItem(edges[0], edges[1],
                                              brush=pg.mkBrush(shade))
                    fill.setZValue(-10)
                    item.addItem(fill)
                    drawn.append((item, fill))
                reach.extend((float(np.min(low[finite])),
                              float(np.max(high[finite]))))
                banded.append(counts[k])
        if not reach:
            continue
        legend = getattr(item, 'legend', None)
        if legend is not None:
            # the count beside the confidence: a band's width means
            # nothing without it, and records averaged differently are
            # given as the range they span
            fewest, most = min(banded), max(banded)
            worth = (dof_text(fewest) if dof_text(fewest) == dof_text(most)
                     else f'{dof_text(fewest)} to {dof_text(most)}')
            for level, style, _lows, _highs in sorted(bands):
                legend.addItem(pg.PlotDataItem(
                    [], [], pen=pg.mkPen(color, width=1,
                                         style=dashes[style])),
                    f'{round(level * 100)} % confidence, {worth} DOF')
        # widen the limits the plot was locked to, so the band's top
        # edge over the highest peak can be seen (`_Extents.lock`)
        top, bottom = max(reach), min(reach)
        if log_y:
            top, bottom = np.log10(top), np.log10(bottom)
        view = item.getViewBox()
        y_lo, y_hi = view.state['limits']['yLimits']
        pad = 0.05 * (top - bottom) or 1.0
        view.setLimits(yMin=(bottom - pad if y_lo is None
                             else min(y_lo, bottom - pad)),
                       yMax=(top + pad if y_hi is None
                             else max(y_hi, top + pad)))
    return drawn


def _mark(layout, data, marks, theme=None, per_octave=None,
          unit_system=None, levels=None, records=None, dof=None):
    """A reading over every plot in `layout` — the frames, the shocks,
    the cut, the filter's preview or the octave steps — as the window's
    toggles draw it (`viz.marks.held_reading`), or a PSD's confidence
    band (`confidence_preview`, at `levels` and `dof`), which is drawn
    flat only.

    Locked, because a mark you can drag is a mark that reports the drag
    to somebody, and in a standalone plot there is nobody. Returns the
    overlays so the caller can keep them alive.
    """
    from ..viz.marks import held_reading, octave_steps

    if levels is not None and marks != 'confidence':
        # a level given to a mark that has none would be dropped without
        # a word (2026-10-16)
        raise ValueError("levels are the confidence of marks='confidence'"
                         f', not of marks={marks!r}')
    if dof is not None and marks != 'confidence':
        # the same for a count, which only a band is worked out from
        raise ValueError("dof is the count of marks='confidence', not "
                         f'of marks={marks!r}')
    if marks is None:
        return []
    colors = resolve_theme(theme)
    plots = figure_plots(layout)
    if marks == 'confidence':
        # a property of the spectrum rather than a setting held on it,
        # so it has no `held_reading` and no stage counterpart
        from ..core.data import Psd

        if not isinstance(data, Psd):
            raise TypeError('a confidence band is drawn on a PSD, not '
                            f'on {type(data).__name__}')
        return confidence_preview(plots, data, colors, records=records,
                                  levels=levels, unit_system=unit_system,
                                  dof=dof)
    held = held_reading(data, marks)
    if marks == 'octave':
        return octave_preview(
            plots, octave_steps(data, per_octave, unit_system), colors)
    if marks == 'shocks':
        from .shocks import ShockOverlay

        return [ShockOverlay(plot, held, colors, locked=True)
                for plot in plots]
    if marks == 'filter':
        from .filtering import FilterOverlay

        return [FilterOverlay(plot, held, data.sample_rate, colors)
                for plot in plots]
    if marks == 'truncation':
        from .truncation import TruncationOverlay

        abscissa = np.asarray(data.abscissa, dtype=float)
        return [TruncationOverlay(plot, held, float(abscissa[0]),
                                  float(abscissa[-1]), colors, locked=True)
                for plot in plots]
    from .averaging import AveragingOverlay

    return [AveragingOverlay(plot, held, data.sample_rate,
                             len(data.abscissa), colors, locked=True)
            for plot in plots]


def plot_data(data: DataArray, unit_system: UnitSystem | None = None,
              theme: Any = None, show: bool = True,
              title: str | None = None, size: tuple[int, int] = (1000, 700),
              path: str | os.PathLike | None = None,
              marks: str | None = None,
              per_octave: int | None = None,
              levels: Sequence[float] | None = None,
              dof: float | Sequence[float] | None = None,
              **kwargs: Any) -> Any:
    """Plot a data array in a window, or save it to a file with `path`.

    Returns the data pane, or the path written. Complex data is shown
    with the component box above it, the same control as in the app, so
    the part drawn is chosen there; a `component` keyword sets the one
    drawn first. With `show=True` and no Qt event loop already running,
    this blocks until the window is closed. Other keyword arguments go
    to `build_plot`.

    `marks` draws a reading over the plot, as the app's toggles do. On a
    time history, 'averaging' shows the frames a spectrum is averaged
    over, 'shocks' the events an SRS is computed from, 'truncation'
    shades the ends a truncation would remove, and 'filter' draws the
    filtered record over the original. On a PSD, 'octave' draws the
    bands it would integrate to as steps, with `per_octave` bands per
    octave. Each is read from the object: the setting it holds, or else
    what its detector suggests. The frames and the shock windows are
    drawn locked, since nothing here would receive a drag. On an
    averaged PSD, 'confidence' shades the chi-square band the true
    spectrum lies in (`confidence_preview`), at the confidence `levels`
    given, 90 and 95 percent by default, from the degrees of freedom
    the PSD was averaged to, or from `dof` when given (one count, or
    one per record), and the legend states both. `levels` or `dof` with
    any other mark is refused.
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
        overlays.extend(_mark(widget, data, marks, theme, per_octave,
                              unit_system, levels, kwargs.get('records'),
                              dof))

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or repr(data), size=size,
                        complex_data=np.iscomplexobj(data.ordinate),
                        on_pane=lambda p: pane.__setitem__(0, p))


def save_plot(data: DataArray, path: str | os.PathLike,
              unit_system: UnitSystem | None = None, theme: Any = None,
              size: tuple[int, int] = (1200, 800),
              **kwargs: Any) -> Any:
    """Save a data array's plot to a .png or .svg without a window.

    Returns the path. Other keyword arguments go to `plot_data`.
    """
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
    """Paint a widget into an image at a device pixel ratio of `ratio`.

    This is the one way a plot is turned into pixels. The scene is laid
    out once, at its logical size, and painted through a device that
    carries the ratio, so every item (text, pens, fills) is drawn scaled
    by the same factor, as a high-density screen draws it. pyqtgraph's
    ImageExporter is not used, because it scales the scene to a
    requested width unevenly and text comes out at the wrong size. For a
    graphics view, cosmetic line pens are widened by `ratio` for the
    render, and every data curve to at least `EXPORT_CURVE_WIDTH`, and
    then put back.

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
    # pyqtgraph's ImageExporter, at three times a plot's width: legend
    # labels came out at a third of their set size and QFont tick labels
    # half again too large, while HTML titles were right (the
    # band-average paper's print figures, 2026-09-26).
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
    """Write a laid-out plot widget to a .png, at `dpi` for print.

    The widget is drawn at its logical size scaled by ``dpi / 96``
    (`render_image`), on the theme's plot background, and the file
    records its resolution. A figure laid out at 3.42 in (328 logical
    pixels) and saved at 300 dpi is 1026 pixels wide and prints at 3.42
    in, with every font at its set size.

    Parameters
    ----------
    widget : QWidget
        The plot, laid out at its printed size in logical pixels.
    path : str or path-like
        Where the .png goes.
    dpi : float, optional
        Dots per inch; one pixel per logical pixel when omitted.
    theme : optional
        The theme whose plot background the image is filled with.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    OSError
        When the file cannot be written.
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
    `save_image`, at `EXPORT_DPI` when that is set. A size that is not
    two positive numbers, or a file type no image writer here knows, is
    refused before anything is drawn (`_check_target`).
    """
    import pyqtgraph as pg

    _check_target(path, size)
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
    # closed whatever the build does: a script's own build that raised
    # left a hidden top-level widget behind for every failed attempt
    try:
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
    finally:
        widget.close()
    return path


def _check_target(path: str | os.PathLike | None,
                  size: Any) -> None:
    """Refuse a figure that cannot be made, before it is drawn.

    The size is two positive, finite numbers of logical pixels. The
    file's extension picks its writer: .svg is drawn as vectors, and
    anything else must be a raster type Qt's image writer knows (.png,
    .jpg and so on). Asked for a .pdf, the export used to build the
    whole figure and then report only that the file could not be
    written.
    """
    try:
        width, height = (float(value) for value in size)
    except (TypeError, ValueError):
        raise ValueError(f'size is (width, height) in pixels, not {size!r}'
                         ) from None
    if not (np.isfinite(width) and np.isfinite(height)
            and width > 0 and height > 0):
        raise ValueError('size is two positive numbers of pixels, not '
                         f'{size!r}')
    if path is None:
        return
    suffix = os.path.splitext(str(path))[1].lower().lstrip('.')
    if suffix == 'svg':
        return
    from PySide6.QtGui import QImageWriter

    known = {bytes(name).decode().lower()
             for name in QImageWriter.supportedImageFormats()}
    if suffix not in known:
        said = f'.{suffix}' if suffix else 'no extension'
        raise ValueError(f'a figure is written as .png or .svg (or another '
                         f'image type), and {str(path)!r} has {said}')


# ---- composing a figure of your own ------------------------------------
#
# The calls below are what a script needs to draw a figure the package
# has no single view for (several readings on one axis, marks and words
# over them, panels side by side) without reaching into the private
# machinery every standalone plot goes through. They add no drawing of
# their own: `compose` is `_plot_window` with the arguments a script may
# set, `as_series` is how `plot_series` reads its items, and the rest
# read what `build_plots` already records (`series_key`) and what
# pyqtgraph already holds (each axis's log mode). Written for a book's
# figure scripts, which called `_plot_window` and `_as_series` directly
# and repeated the log10 and the range pinning by hand in every chapter
# (2026-10-24).


def compose(build: Callable[[Any], Any], *, theme: Any = None,
            path: str | os.PathLike | None = None, show: bool = True,
            title: str | None = None, size: tuple[int, int] = (1000, 700),
            frequency_axis: str | None = None) -> Any:
    """Draw a figure of your own, in the frame every plot here uses.

        def build(layout):
            build_plots(layout, as_series([psd, specification]))
            plot = figure_plots(layout, 'acceleration**2/frequency')[0]
            pin_range(plot, x=(20.0, 2000.0))

        compose(build, path='figure.png', theme='light', size=(900, 460))

    `build(layout)` draws into a pyqtgraph ``GraphicsLayoutWidget``, with
    the package's builders (`build_plots`, `legend_below`, the views'
    own ``build_*`` functions) and with pyqtgraph's own items for marks
    and words. Panels side by side are nested layouts,
    ``layout.addLayout(row=0, col=1)``, and `build_plots` draws into one
    as it draws into the whole figure.

    Written to a file, the figure is laid out off screen at `size` and
    exported, with no window and no event loop of your own. Shown, it
    comes up in the application's data pane. This is the one path every
    standalone plot in the package takes, so a composed figure is drawn,
    themed and exported exactly as theirs are.

    Parameters
    ----------
    build : callable
        Called once with the layout to draw into.
    theme : str or dict, optional
        'light' or 'dark', or a resolved theme; light when omitted.
    path : str or path-like, optional
        Where to write the figure. The extension picks the format: .svg,
        or .png (at `EXPORT_DPI` when that is set) or another image
        type.
    show : bool, default True
        Without `path`: open the pane, blocking until it is closed when
        no event loop is running. False returns it unopened.
    title : str, optional
        The window's title.
    size : (int, int), default (1000, 700)
        Width and height in logical pixels.
    frequency_axis : {'log', 'linear', 'default'}, optional
        How frequency axes are drawn while this figure is built, put
        back afterwards (`visualdynamics.core.data.frequency_axis`).
        Only with `path`: a shown pane may redraw after the setting has
        been put back.

    Returns
    -------
    str or DataPane
        The path written, or the pane.

    Raises
    ------
    TypeError
        If `build` is not callable.
    ValueError
        If `size` is not two positive numbers, the extension names no
        image type, `frequency_axis` is not a setting, or it is given
        without `path`.
    OSError
        If the file cannot be written.
    """
    from ..core.data import frequency_axis as axis_setting

    if not callable(build):
        raise TypeError('build is a function that draws into the layout, '
                        f'not a {type(build).__name__}')
    if frequency_axis is None:
        return _plot_window(build, theme=theme, path=path, show=show,
                            title=title, size=size)
    if path is None:
        raise ValueError('frequency_axis applies to a figure written to '
                         'path: a shown pane may redraw after the setting '
                         'is put back')
    _check_target(path, size)
    before = axis_setting()
    # set through the setting itself, which refuses a mode it does not
    # know before anything is drawn
    axis_setting(frequency_axis)
    try:
        return _plot_window(build, theme=theme, path=path, show=show,
                            title=title, size=size)
    finally:
        axis_setting(before)


def figure_plots(layout: Any, dimension: str | None = None) -> list[Any]:
    """The plots in a figure, in reading order; or those of one quantity.

    Reading order is row by row, left to right within a row, and a
    nested panel's plots are read where the panel sits. A legend is not
    a plot and is left out.

    `build_plots` stacks one plot per quantity, so the plot to draw a
    mark on is found by what it draws, not by its row. With `dimension`
    (a dimension expression as a record's ``ordinate_dim`` holds it:
    'length', 'acceleration**2/frequency'), only the plots drawing that
    quantity are returned. A plot a script drew itself records no
    quantity, and only the unfiltered list holds it.

    Parameters
    ----------
    layout : GraphicsLayoutWidget or GraphicsLayout
        The figure, or one panel of it.
    dimension : str, optional
        The quantity to find.

    Returns
    -------
    list of pyqtgraph.PlotItem
        Every plot, or every plot drawing `dimension`.

    Raises
    ------
    UnitError
        If `dimension` names a dimension the units do not track.
    ValueError
        If no plot in the figure draws `dimension`; the message names
        the quantities it does draw.
    """
    import pyqtgraph as pg

    from ..units import parse_dimension

    wanted = (None if dimension is None
              else sorted(parse_dimension(dimension)))
    found: list[Any] = []

    def walk(container: Any) -> None:
        # pyqtgraph keeps {item: [(row, col), ...]} in the order items
        # were added, which is not the order they are read in
        cells = getattr(container, 'ci', container).items
        for item, _places in sorted(cells.items(),
                                    key=lambda pair: min(pair[1])):
            if isinstance(item, pg.PlotItem):
                found.append(item)
            elif isinstance(item, pg.GraphicsLayout):
                walk(item)

    walk(layout)
    if wanted is None:
        return found
    keys = [getattr(plot, 'series_key', None) for plot in found]
    matched = [plot for plot, key in zip(found, keys)
               if key is not None
               and sorted(parse_dimension(key[1])) == wanted]
    if not matched:
        drawn = sorted({key[1] for key in keys if key is not None})
        held = (f'it draws {", ".join(drawn)}' if drawn
                else 'it holds no plot drawn from data')
        raise ValueError(f'no plot in this figure draws {dimension}; {held}')
    return matched


def to_view(plot: Any, x: ArrayLike | None = None,
            y: ArrayLike | None = None) -> tuple[Any, Any]:
    """Data coordinates as the plot's view holds them.

    A plot drawn on a log axis works in decades: its view holds log10 of
    the data. A curve added with ``plot.plot`` is converted by pyqtgraph,
    but a line (``InfiniteLine``), a label (``TextItem``), a set of
    markers and a range are placed in view coordinates, and a value
    placed without the conversion lands at its own logarithm. This
    converts each axis as that axis is drawn.

    Parameters
    ----------
    plot : pyqtgraph.PlotItem
        The plot the coordinates are for.
    x, y : float or array_like, optional
        Values in the plot's display units; either may be omitted.

    Returns
    -------
    (x, y)
        Each converted, a float for a number and an array for an array,
        or None where it was omitted.

    Raises
    ------
    ValueError
        If a value is not finite, or is zero or negative on a log axis;
        neither has a place on the plot.
    """
    log_x, log_y = plot.getViewBox().state['logMode']
    return (_view_axis(x, bool(log_x), 'x'), _view_axis(y, bool(log_y), 'y'))


def _view_axis(values: ArrayLike | None, log: bool, name: str) -> Any:
    """One axis's values in view coordinates (`to_view`)."""
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    bad = ~np.isfinite(array)
    if bad.any():
        raise ValueError(f'{name} holds {int(bad.sum())} value(s) that are '
                         'not finite, which have no place on a plot')
    if log:
        low = array[array <= 0.0]
        if low.size:
            raise ValueError(f'the {name} axis is drawn in decades, and '
                             f'{name} = {float(low.flat[0]):g} has no '
                             'logarithm')
        array = np.log10(array)
    return float(array) if array.ndim == 0 else array


def pin_range(plot: Any, x: tuple[float, float] | None = None,
              y: tuple[float, float] | None = None) -> None:
    """Fix a plot's range, in data units, so it opens where it is told.

    Panels meant to be compared need one scale. `build_plots` frames
    each plot on its own data and limits zoom and pan to that frame, so
    a range set straight on the plot is clamped back inside it; and on a
    log axis the range is in decades. This sets the range from data
    units (`to_view`), turns autorange off on that axis, and makes the
    pinned range the plot's outer limit, as the data's own frame was.

    Parameters
    ----------
    plot : pyqtgraph.PlotItem
        The plot to pin.
    x, y : (float, float), optional
        The low and high ends, in the plot's display units; either may
        be omitted.

    Raises
    ------
    ValueError
        If a range is not two values, its low end is not below its high
        end, or an end has no place on the axis (`to_view`).
    """
    view = plot.getViewBox()
    for axis, span in (('x', x), ('y', y)):
        if span is None:
            continue
        try:
            low, high = (float(end) for end in span)
        except (TypeError, ValueError):
            raise ValueError(f'{axis} is a (low, high) range, not {span!r}'
                             ) from None
        ends = to_view(plot, **{axis: [low, high]})[0 if axis == 'x' else 1]
        if not low < high:
            raise ValueError(f'{axis} runs from low to high, and {low:g} is '
                             f'not below {high:g}')
        lo, hi = float(ends[0]), float(ends[1])
        view.setLimits(**{f'{axis}Min': lo, f'{axis}Max': hi})
        if axis == 'x':
            plot.setXRange(lo, hi, padding=0)
        else:
            plot.setYRange(lo, hi, padding=0)


def plot_mac(shapes: ShapeSet, other: ShapeSet | None = None, *,
             theme: Any = None, path: str | os.PathLike | None = None,
             show: bool = True, title: str | None = None,
             size: tuple[int, int] | None = None, bars: bool = False,
             screenshot: str | None = None) -> Any:
    """Draw the MAC of `shapes` against itself, or against `other`.

    This is the grid the app draws; with `other` it is the cross-MAC.
    The image is 700 pixels tall by default, and as wide as the grid's
    shape gives, clamped as the app clamps it (`mac_frame_ratio`), so a
    139-by-8 cross-MAC is drawn as the same tall panel as on screen and
    not stretched into a square. `size` overrides this.

    `bars=True` draws the app's 3-D form: the matrix as bars, with
    height and color both giving the value. A 3-D scene is rendered by
    its own plotter, so it is saved with `screenshot=` instead of
    `path=`, as every other 3-D view is.

    Returns the pane or the path written, or with `bars` and
    `screenshot` the screenshot image.
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
    """Draw a MAC matrix that was computed elsewhere.

    `plot_mac` uses this for two sets that share DOF names, and
    `Project.plot_mac` for the comparison projected across geometries,
    so the drawing is the same whoever computed the matrix.
    `frequencies` label the rows, and `column_frequencies` the columns
    when they belong to another set. The other arguments and the return
    value are as for `plot_mac`.
    """
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
    """Draw the CMIF as the fitting screen draws it.

    The plot has the measured indicator and, with `shapes`, the modal
    model's synthesis dashed over it. `modes` selects the modes used in
    the synthesis, all of them by default. Returns the pane, or the path
    written.
    """
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
    """Draw the coherence map: frequency across, channel down.

    The color scale is fixed at 0 to 1 (`build_coherence_map`). Returns
    the pane, or the path written.
    """
    def build(widget: Any) -> Any:
        build_coherence_map(widget, [(None, coherence, None)],
                            unit_system=unit_system, theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or 'Coherence', size=size)


def build_photos(layout: Any, photos: Photos,
                 picks: Sequence[int] | None = None,
                 start_row: int = 0) -> int:
    """Draw photos down a layout, each with its name as the title.

    Each photo keeps its own aspect ratio. Drawing starts at `start_row`
    and the layout is not cleared, so several sets can be stacked in one
    layout, as when two Photos objects are selected in the app. `picks`
    selects photos by index, drawn in ascending order, and an image that
    cannot be decoded is skipped. Returns how many were drawn.
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
    """Show the photos as the app shows them, or save them with `path`.

    `picks` selects photos by index. Returns the pane, or the path
    written.
    """
    def build(widget: Any) -> Any:
        widget.clear()
        build_photos(widget, photos, picks)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or 'Photos', size=size)


def as_series(items: Iterable[Any]
              ) -> list[tuple[str | None, Any, list[int] | None]]:
    """Data arrays as the (name, data, records) list the builders take.

        build_plots(layout, as_series([psd, ('Spec', specification)]))

    Each item is a data array, ``(name, data)``, or ``(name, data,
    records)`` to draw only some of its records, in any mixture. The
    name prefixes the record's legend entries when more than one object
    is named. This is how `plot_series` reads its items, so a figure
    built with `compose` takes them the same way.

    Parameters
    ----------
    items : iterable
        Data arrays and tuples, as above.

    Returns
    -------
    list of (str or None, DataArray, list of int or None)
        One entry per item, `records` None for every record.

    Raises
    ------
    TypeError
        If an item holds something that is not a data array, a name that
        is not a string, or records that are not indices.
    ValueError
        If nothing is given, a tuple is not two or three long, or
        `records` is empty or names a record the array does not have.
    """
    series = []
    for position, item in enumerate(items):
        if isinstance(item, tuple):
            if len(item) not in (2, 3):
                raise ValueError(f'item {position} is a tuple of {len(item)}; '
                                 'an item is (name, data) or (name, data, '
                                 'records)')
            name, data = item[0], item[1]
            records = item[2] if len(item) == 3 else None
        else:
            name, data, records = None, item, None
        if not (hasattr(data, 'num_records') and hasattr(data, 'ordinate')):
            raise TypeError(f'item {position} is a {type(data).__name__}, '
                            'not a data array')
        if name is not None and not isinstance(name, str):
            raise TypeError(f'item {position} is named by a '
                            f'{type(name).__name__}, not a string')
        if records is not None:
            # a bare index is one record, as a list of one would be
            picked = np.atleast_1d(np.asarray(records))
            if picked.size == 0:
                raise ValueError(f'item {position} picks no records')
            if not np.issubdtype(picked.dtype, np.integer):
                raise TypeError(f'item {position} picks records by '
                                f'{picked.dtype} values, not indices')
            records = [int(i) for i in picked]
            # a negative index is Python's count from the end, which no
            # record list means: -1 drew the last record without a word
            outside = [i for i in records if not 0 <= i < data.num_records]
            if outside:
                raise ValueError(f'item {position} has {data.num_records} '
                                 f'record(s), so record(s) {outside} are not '
                                 'in it')
        series.append((name, data, records))
    if not series:
        raise ValueError('nothing to draw: no data arrays were given')
    return series


def plot_series(items: Sequence[tuple[str | None, Any, Sequence[int] | None]], unit_system: UnitSystem | None = None,
                theme: Any = None,
                path: str | os.PathLike | None = None,
                show: bool = True, title: str | None = None,
                size: tuple[int, int] = (1000, 700),
                component: str = 'magnitude', **kwargs: Any) -> Any:
    """Draw several data arrays on one figure.

        visualdynamics.plot.plot_series([frf, synthesized], path='both.png')

    This is what selecting more than one object in the tree draws. Each
    item is a data array, `(name, data)`, or `(name, data, records)` to
    draw only some of its records. Curves are grouped onto one axis per
    quantity, so an FRF and a time history are stacked instead of
    sharing one, and the names prefix the legend entries. `component`
    and the other keyword arguments are as for `build_plots`. Returns
    the pane, or the path written.
    """
    series = as_series(items)

    def build(widget: Any) -> Any:
        build_plots(widget, series, unit_system=unit_system, theme=theme,
                    component=component, **kwargs)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or 'Data', size=size)


def plot_half_power(data: Any, near: float, record: int = 0, *,
                    within: tuple[float, float] | None = None,
                    unit_system: UnitSystem | None = None,
                    theme: Any = None,
                    path: str | os.PathLike | None = None,
                    show: bool = True, title: str | None = None,
                    size: tuple[int, int] = (1000, 700)) -> Any:
    """Draw one record with the half-power reading of a peak on it.

        visualdynamics.plot.plot_half_power(frf, 180.0, path='q.png')

    The record is drawn as `plot_series` draws it, and the peak nearest
    `near` (or the tallest inside `within`) is read by
    `core.half_power.read_half_power` and drawn over it by
    `plot.half_power.add_half_power`: the 0.707 level, the two
    crossings, the top as read between lines, and the Q beside the
    peak. Works on an FRF (a transmissibility among them) and a
    spectrum (a tracked sine level among them): a magnitude against
    frequency, whose half-power level is 0.707 of the peak. A PSD is
    refused, since it is a power, whose half-power level is half the
    peak, and read at 0.707 its Q comes out wrong; so are an SRS and a
    coherence, which are not a response's magnitude. A reading with a
    missing crossing or too few lines warns as `read_half_power` does,
    and the label on the plot says so too.

    Parameters
    ----------
    data : Frf or Spectrum
        The curve, in the frequency domain.
    near : float
        A frequency on the peak to read, in Hz.
    record : int, default 0
        Which record of `data`.
    within : tuple of float, optional
        A (low, high) band in Hz whose tallest line is the peak.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.

    Raises
    ------
    TypeError
        When `data` is not an FRF or a spectrum.
    """
    from ..core.data import Frf, Spectrum
    from ..core.half_power import read_half_power
    from .half_power import add_half_power

    # 2026-10-16: the guard was abscissa_dim == 'frequency', which let a
    # PSD through; a PSD of the Q 20 accelerance's |H|^2 read Q 31
    if not isinstance(data, (Frf, Spectrum)):
        raise TypeError('a half-power width is read off an FRF or a '
                        'spectrum, a magnitude against frequency; this '
                        f'is a {type(data).__name__}')
    reading = read_half_power(data.abscissa, data.ordinate[record], near,
                              within=within, stacklevel=2)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    dimension = data.ordinate_dim[record]
    # drawn in display units, read in SI: the conversion of a magnitude
    # is one factor, so the height scales by it and Q does not move
    scale = (1.0 if dimension == UNKNOWN or not reading.height
             else float(us.from_si(reading.height, dimension))
             / reading.height)

    def build(widget: Any) -> Any:
        build_plots(widget, [(None, data, [record])], unit_system=us,
                    theme=theme)
        plot = figure_plots(widget)[0]
        add_half_power(plot, reading,
                       color=resolve_theme(theme)['plot_foreground'],
                       scale=scale)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or (f'{data.response_dof[record]}: '
                                        f'{reading.describe()}'),
                        size=size)


def plot_tone_readout(data: Any, tones: float | Sequence[float],
                      record: int = 0, *, window: Any = 'hann',
                      parameter: float | None = None, snap: bool = True,
                      band: tuple[float, float] | None = None,
                      unit_system: UnitSystem | None = None,
                      theme: Any = None,
                      path: str | os.PathLike | None = None,
                      show: bool = True, title: str | None = None,
                      size: tuple[int, int] = (1000, 700)) -> Any:
    """Draw one record of an autospectrum with each tone read off it.

        visualdynamics.plot.plot_tone_readout(asd, [19.44, 38.88],
                                              path='tones.png')

    The record is drawn as `plot_series` draws it, and each tone in
    `tones` is read by `Psd.read_tone` and drawn over it by
    `plot.tone_readout.add_tone_readout`: a stem to the height the
    spike stands at for this spacing and window, a triangle on it, the
    floor taken away, and the tone's amplitude beside it with the
    spike's height under it. A reading that warns (a tone between
    lines, an uneven floor, no tone) warns as `read_tone` does.

    Parameters
    ----------
    data : Psd
        The averaged autospectrum.
    tones : float or sequence of float
        The frequencies to read, in Hz.
    record : int, default 0
        Which record of `data`.
    window : str or Averaging, default 'hann'
        The window the spectrum was averaged with, or its `Averaging`
        (`Project.averaging_of` gives a computed one's).
    parameter : float, optional
        The Tukey window's alpha or the Kaiser window's beta, when
        `window` is a name.
    snap : bool, default True
        Find each tone's own spike near its frequency; False reads at
        the frequencies as given.
    band : tuple of float, optional
        The (low, high) frequencies the view shows, in Hz; the whole
        record when omitted. Tones crowd the left of a wide band.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from ..core.tone_readout import leakage, line_spacing
    from .tone_readout import add_tone_readout, display_units

    refusal = data.tone_refusal(record)
    if refusal is not None:
        raise ValueError(refusal)
    frequencies = ([float(tones)] if np.ndim(tones) == 0
                   else [float(f) for f in tones])
    readings = [data.read_tone(f, record, window=window,
                               parameter=parameter, snap=snap)
                for f in frequencies]
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    scale, dense, unit = display_units(data, record, us)

    def build(widget: Any) -> Any:
        build_plots(widget, [(None, data, [record])], unit_system=us,
                    theme=theme)
        plot = figure_plots(widget)[0]
        for reading in readings:
            add_tone_readout(plot, reading, data.abscissa,
                             color=resolve_theme(theme)['plot_foreground'],
                             above=True, scale=scale, density_scale=dense,
                             unit=unit)
        f = np.asarray(data.abscissa, dtype=np.float64)
        low, high = ((float(f[0]), float(f[-1])) if band is None
                     else sorted(float(edge) for edge in band))
        log_x, log_y = plot.getViewBox().state['logMode']
        shown = np.abs(np.asarray(data.ordinate[record], dtype=np.float64)
                       [(f >= low) & (f <= high)]) * dense
        shown = shown[np.isfinite(shown) & (shown > 0.0 if log_y else True)]
        if band is not None:
            plot.setXRange(*((np.log10(max(low, 1e-12)), np.log10(high))
                             if log_x else (low, high)), padding=0.0)
        if len(shown) and log_y:
            # the band's own decades, and two more over the tallest
            # spike for the words stacked over each triangle
            plot.setYRange(np.log10(shown.min()) - 0.3,
                           np.log10(shown.max()) + 2.0, padding=0.0)
        elif len(shown):
            plot.setYRange(0.0, 1.6 * shown.max(), padding=0.0)

    # from the spectrum and the window, not the first reading: a call
    # with no tones has none to take them from
    spacing = line_spacing(data.abscissa)
    window_name = leakage(window, parameter=parameter).window
    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or (f'{data.response_dof[record]}: '
                                        f'tones at {spacing:.3g} Hz lines, '
                                        f'{window_name} window'),
                        size=size)


def plot_transmissibility(reading: Any, *, spread: float = 1.0,
                          band: tuple[float, float] | None = None,
                          unit_system: UnitSystem | None = None,
                          theme: Any = None,
                          path: str | os.PathLike | None = None,
                          show: bool = True, title: str | None = None,
                          size: tuple[int, int] = (1000, 760)) -> Any:
    """Draw the two transmissibilities of a reading together, with the
    band H1's random error puts about it and the coherence beneath.

        reading = history.read_transmissibility('101Z+', '1Z+')
        visualdynamics.plot.plot_transmissibility(reading, path='t.png')

    Drawn by `plot.transmissibility.build_transmissibility`: above, on
    log axes, H1 solid, the ASD ratio dashed and the band dotted with a
    faint fill, each named in the legend, so the reader sees where the
    two agree and where they part; beneath, the coherence the
    band was read from. The window's FRF view draws the same reading
    over an FRF with its Transmissibility toggle.

    Parameters
    ----------
    reading : TransmissibilityReading
        From `TimeHistory.read_transmissibility`,
        `Project.read_transmissibility` or
        `core.transmissibility.read_transmissibility`.
    spread : float, default 1.0
        How many random errors of H1 the band reaches either side.
    band : tuple of float, optional
        The (low, high) frequencies the view shows, in Hz; the whole
        reading when omitted.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from .transmissibility import build_transmissibility

    def build(widget: Any) -> Any:
        widget.clear()
        return build_transmissibility(widget, reading, spread=spread,
                                      band=band, unit_system=unit_system,
                                      theme=theme)

    pair = (f'{reading.response} over {reading.reference}: '
            if reading.response or reading.reference else '')
    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'{pair}H1 and the ASD ratio',
                        size=size)


def plot_rms_check(data: Any, history: Any, record: int = 0, *,
                   averaging: Any = None,
                   threshold_db: float | None = None,
                   unit_system: UnitSystem | None = None,
                   theme: Any = None,
                   path: str | os.PathLike | None = None,
                   show: bool = True, title: str | None = None,
                   size: tuple[int, int] = (1000, 700)) -> Any:
    """Draw one record of an autospectrum with its rms check written on
    it: the rms from the spectrum beside the rms of the time history it
    came from, the gap in dB and the verdict in words.

        visualdynamics.plot.plot_rms_check(asd, history, path='rms.png')

    The record is drawn as `plot_series` draws it, the check is made by
    `Psd.rms_check` with the same arguments, and
    `plot.rms_check.add_rms_check` writes it in the plot's top-right
    corner, where the RMS Check reading in the window writes it too.

    Parameters
    ----------
    data : Psd
        The autospectrum.
    history : TimeHistory
        The record it was computed from.
    record : int, default 0
        Which record of `data`.
    averaging : Averaging, optional
        How `data` was averaged; `history`'s own when omitted
        (`Project.averaging_of` gives a computed spectrum's).
    threshold_db : float, optional
        The gap, in dB either way, beyond which the two disagree;
        `core.rms_check.GAP_DB` when omitted.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from .rms_check import add_rms_check
    from .tone_readout import display_units

    check = data.rms_check(history, record, averaging=averaging,
                           threshold_db=threshold_db)
    us = DEFAULT_SYSTEM if unit_system is None else unit_system
    scale, _dense, unit = display_units(data, record, us)

    def build(widget: Any) -> Any:
        build_plots(widget, [(None, data, [record])], unit_system=us,
                    theme=theme)
        plot = next(item for item in widget.ci.items
                    if hasattr(item, 'addItem'))
        add_rms_check(plot, check,
                      color=resolve_theme(theme)['plot_foreground'],
                      unit=unit, scale=scale)
        # headroom over the trace for the words in the corner, so they
        # write over no line; the plot is locked to its data's extents,
        # so the lock is lifted to the new top first
        box = plot.getViewBox()
        log_y = box.state['logMode'][1]
        shown = np.abs(np.asarray(data.ordinate[record], dtype=np.float64)
                       ) * _dense
        shown = shown[np.isfinite(shown) & (shown > 0.0 if log_y else True)]
        if len(shown) and log_y:
            low, high = np.log10(shown.min()), np.log10(shown.max())
            reach = max(high - low, 0.5)
            low, high = low - 0.05 * reach, high + 0.75 * reach
        elif len(shown):
            low, high = 0.0, 1.9 * shown.max()
        if len(shown):
            box.setLimits(yMin=low, yMax=high)
            plot.setYRange(low, high, padding=0.0)

    gap = check.gap_db
    said = f'{gap:+.3f} dB' if np.isfinite(gap) else 'no gap'
    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or (f'{data.response_dof[record]}: '
                                        f'rms check, {said}'),
                        size=size)


def plot_conditioning(reading: Any, *,
                      band: tuple[float, float] | None = None,
                      unit_system: UnitSystem | None = None,
                      theme: Any = None,
                      path: str | os.PathLike | None = None,
                      show: bool = True, title: str | None = None,
                      size: tuple[int, int] = (1000, 760)) -> Any:
    """Draw the singular values of an FRF or CSD matrix against
    frequency, with its condition number beneath and the lines over the
    limit hatched.

        reading = project['Run H1 FRFs'].conditioning()
        visualdynamics.plot.plot_conditioning(reading, path='c.png')

    Drawn by `plot.matrix_checks.build_conditioning`: above, on log
    axes, the singular values, each in its own line style and named,
    with the rank threshold dotted; beneath, the condition number with
    the limit dashed. The window's Conditioning toggle draws the same
    reading for a selected FRF or CSD.

    Parameters
    ----------
    reading : Conditioning
        From `Frf.conditioning`, `Psd.conditioning` or
        `core.matrix_checks.read_conditioning`.
    band : tuple of float, optional
        The (low, high) frequencies the view shows, in Hz; the whole
        reading when omitted.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from .matrix_checks import build_conditioning

    def build(widget: Any) -> Any:
        widget.clear()
        return build_conditioning(widget, reading, band=band,
                                  unit_system=unit_system, theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'Conditioning: {reading.title}',
                        size=size)


def plot_realizability(reading: Any, *,
                       band: tuple[float, float] | None = None,
                       pair: tuple[int, int] | None = None,
                       theme: Any = None,
                       path: str | os.PathLike | None = None,
                       show: bool = True, title: str | None = None,
                       size: tuple[int, int] = (1000, 860)) -> Any:
    """Draw the eigenvalues of a cross-spectral matrix's coherence matrix
    against frequency, the lines that are not realizable hatched, and
    what the repair does to the pair it moves most.

        reading = project['Specification'].realizability()
        visualdynamics.plot.plot_realizability(reading, path='r.png')

    Drawn by `plot.matrix_checks.build_realizability`: above, the
    eigenvalues on a linear axis with the zero line dashed, so a
    negative one stands below it; beneath, when the repair moved
    anything, the coherence and phase of that pair as written (solid)
    and repaired (dashed). The window's Realizability toggle draws the
    same reading for a selected CSD or specification.

    Parameters
    ----------
    reading : Realizability
        From `Psd.realizability` or
        `core.matrix_checks.read_realizability`.
    band : tuple of float, optional
        The (low, high) frequencies the view shows, in Hz; the whole
        reading when omitted.
    pair : tuple of int, optional
        The two channels, by index, whose coherence and phase are drawn
        before and after the repair; the pair it moves most when
        omitted.
    theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from .matrix_checks import build_realizability

    def build(widget: Any) -> Any:
        widget.clear()
        return build_realizability(widget, reading, band=band, pair=pair,
                                   theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or 'Realizability of '
                                       f'{len(reading.channels)} channels',
                        size=size)


def build_ratio(layout: Any, signal: Any, floor: Any,
                records: Sequence[int] | None = None,
                theme: Any = None) -> int:
    """Draw the signal-to-noise ratio of two sets of densities, in dB.

    The ratio is `signal` over `floor`, one curve per shared channel.
    This is the reading of two selected PSDs. The signal's power is the
    louder density minus the quieter, divided by the quieter
    (`snr.signal_to_noise`), so a line where the two are equal has no
    signal at all and is not drawn, instead of being drawn at 0 dB. The
    ordinate is decibels on a linear axis, not a log axis of the linear
    ratio, because the decibel value is what is read. A dashed line at 0
    dB marks where the signal's power equals the noise's. `records`
    restricts the curves to those rows of `signal`, the rows picked in
    the grid; when none of them is a shared channel, every channel is
    drawn.

    Returns the number of shared channels. A channel with no finite
    value has no curve but is still counted.
    """
    # The reading of two selected PSDs is Brandon's (2026-08-25); the
    # signal's power is the louder less the quieter, over the quieter,
    # since `snr.signal_to_noise` (2026-10-03).
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from ..core.snr import picked, signal_to_noise

    colors = resolve_theme(theme)
    abscissa, rows, dofs, _dims = signal_to_noise(signal, floor)
    kept = picked(signal, records, dofs)
    rows, dofs = np.asarray(rows)[kept], [dofs[k] for k in kept]
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
    """Draw the signal-to-noise ratio of driven over ambient densities.

        visualdynamics.plot.plot_ratio(driven_psds, noise_psds,
                                       path='snr.png')

    The ratio is drawn line by line, in decibels. This is the script
    form of the toolbar's Signal to noise reading of two selected PSDs.
    The signal's power is the driven density minus the ambient, divided
    by the ambient (`visualdynamics.core.snr`), and a line with no
    signal above the noise is not drawn. `records` restricts the curves
    to those rows of `signal`. `unit_system` is accepted and ignored,
    since a ratio of densities has no units. Returns the pane, or the
    path written.
    """
    del unit_system     # a ratio of densities converts to itself

    def build(widget: Any) -> Any:
        build_ratio(widget, signal, floor, records=records, theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or 'Signal to noise', size=size)



def plot_signature(before: Any, after: Any, modes: Any, *,
                   record: int = 0, after_record: int | None = None,
                   frequency_limit: Any = None,
                   amplitude_limit: Any = None,
                   search: float | None = None,
                   names: tuple[str, str] = ('before', 'after'),
                   unit_system: UnitSystem | None = None,
                   theme: Any = None,
                   path: str | os.PathLike | None = None,
                   show: bool = True, title: str | None = None,
                   size: tuple[int, int] = (1300, 700)) -> Any:
    """Two signatures overlaid, a window drawn on each mode for its
    frequency and amplitude limits, and the table of shifts beside.

        visualdynamics.plot.plot_signature(
            before, after, [84.0, SignatureMode(412.0, counts=False)],
            path='signature.png')

    The comparison is `core.signature.compare_signatures`'s: each mode
    read before and after, its frequency and amplitude shift in
    percent, and a verdict on each mode that counts. The window about a
    mode's peak before the test is as wide as the frequency limit and
    as tall as the amplitude limit, so the peak after passes when it
    lands inside. A mode that counts has a solid window and its verdict
    written over it; one that does not count has a dotted window and
    'not judged'.

    Parameters
    ----------
    before, after : DataArray
        The two signatures: FRFs, transmissibilities or tracked sweep
        levels.
    modes : float, SignatureMode, (float, bool), sequence, or shape set
        The modes compared (`core.signature.as_modes`).
    record, after_record : int
        Which record of each; `after_record` by DOF when omitted.
    frequency_limit, amplitude_limit : float or (float, float), optional
        The limits in percent; `FREQUENCY_LIMIT` and `AMPLITUDE_LIMIT`
        when omitted.
    search : float, optional
        How far either side a mode's window reaches, as a share of its
        frequency; `SEARCH` when omitted.
    names : (str, str), default ('before', 'after')
        The two curves' names in the legend.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from ..core.signature import AMPLITUDE_LIMIT, FREQUENCY_LIMIT, SEARCH
    from .signature import build_signature

    settings = {
        'frequency_limit': (FREQUENCY_LIMIT if frequency_limit is None
                            else frequency_limit),
        'amplitude_limit': (AMPLITUDE_LIMIT if amplitude_limit is None
                            else amplitude_limit),
        'search': SEARCH if search is None else search}

    def build(widget: Any) -> Any:
        widget.clear()
        build_signature(widget, before, after, modes, record=record,
                        after_record=after_record, names=names,
                        unit_system=unit_system, theme=theme, **settings)

    dof = before.response_dof[record]
    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'{dof} before and after',
                        size=size)

def plot_comparison(measured: Any, specification: Any, *,
                    pair: tuple[str, str] | None = None,
                    unit_system: UnitSystem | None = None,
                    theme: Any = None,
                    path: str | os.PathLike | None = None,
                    show: bool = True, title: str | None = None,
                    size: tuple[int, int] = (1000, 700)) -> Any:
    """Draw one control channel against its specification.

        visualdynamics.plot.plot_comparison(psds, spec, path='control.png')

    The limit zones are shaded. This is the comparison the app draws
    when a specification and the PSDs it bounds are selected together:
    the measurement drawn flat across each bin, the specification as the
    power law between its breakpoints, the warning band shaded yellow,
    and the regions outside the abort limits shaded red above the upper
    limit and blue below the lower. A measurement captured below the
    specification's level is drawn scaled to it, and the title gives the
    scale.

    One channel is drawn at a time, because six channels and their two
    dozen limit curves on one axis are too crowded to read. `pair` names
    the channel as the DOF pair the drop-down lists, such as `('101Z+',
    '101Z+')`; by default it is the first pair the two have in common.
    To compare on proportional bands, band the measurement first with
    `psds.to_octave(6)`.

    Raises `ValueError` when the two have no channel in common, or when
    `pair` is not one of the shared channels. Returns the pane, or the
    path written.
    """
    series = as_series([(None, measured), (None, specification)])
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


def plot_srs_tolerance(measured: Any, specification: Any, *,
                       channel: str | None = None,
                       event: str | None = None,
                       inner: tuple[float, float] | None = None,
                       outer: tuple[float, float] | None = None,
                       rule: str = 'share',
                       percent: float | None = None,
                       limits: bool = True,
                       unit_system: UnitSystem | None = None,
                       theme: Any = None,
                       path: str | os.PathLike | None = None,
                       show: bool = True, title: str | None = None,
                       size: tuple[int, int] = (1000, 700)) -> Any:
    """One channel's shock response spectra against the requirement's
    tolerance band, point by point, with the points outside marked and
    the count inside stated.

        positive = history.compute_srs(kind='positive')
        negative = history.compute_srs(kind='negative')
        visualdynamics.plot.plot_srs_tolerance(
            [positive, negative], spec, outer=(-3.0, 6.0),
            path='tolerance.png')

    The judgement is `core.compliance.srs_tolerance`'s, one per
    spectrum: the requirement in gray, the band around it shaded as a
    specification's zones are (yellow outside the inner band, red and
    blue past the outer one), each spectrum in its own color and
    marker, and every point the rule counts against it marked with
    its marker filled. Each spectrum's legend entry gives its
    polarity, how many of its points are inside the band, the share,
    and whether it passed.

    Parameters
    ----------
    measured : Srs or sequence of Srs
        The spectra, typically one per polarity.
    specification : ShockSpecification or Srs
        The requirement.
    channel : str, optional
        The DOF drawn; the specification's first by default.
    event : str, optional
        The event drawn, by block label; each spectrum's first record
        at the channel by default.
    inner, outer, rule, percent, limits
        The bands and the rule, as for `core.compliance.srs_tolerance`:
        a band not given is the requirement's own limits when it has
        them, and `percent` defaults to `SRS_POINTS_PERCENT`.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from ..core.compliance import SRS_POINTS_PERCENT
    from .srs_tolerance import build_srs_tolerance

    settings = {'inner': inner, 'outer': outer, 'rule': rule,
                'percent': SRS_POINTS_PERCENT if percent is None else percent,
                'limits': limits}

    def build(widget: Any) -> Any:
        widget.clear()
        build_srs_tolerance(widget, specification, measured,
                            channel=channel, event=event,
                            unit_system=unit_system, theme=theme, **settings)

    dof = specification.response_dof[0] if channel is None else channel
    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'{dof} against its SRS tolerance',
                        size=size)


def plot_breakpoints(specification: Any, *, channel: str | None = None,
                     unit_system: UnitSystem | None = None,
                     theme: Any = None,
                     path: str | os.PathLike | None = None,
                     show: bool = True, title: str | None = None,
                     size: tuple[int, int] = (1000, 700)) -> Any:
    """One channel of a random specification read by its breakpoints:
    each segment's slope and area written on it, and the rms
    accumulated from the low end beneath.

        draft = SpecificationDraft.at_dofs(['1Z+'])   # then shaped
        visualdynamics.plot.plot_breakpoints(draft, path='areas.png')

    The reading is `core.breakpoints.read_breakpoints`'s. The
    specification is drawn solid on log-log axes with its breakpoints
    circled, each segment labeled with its slope in decibels per octave
    and its area; under it the running rms is dashed, sharing the
    frequency axis, and its title gives where it ends, the
    specification's rms. Areas are in the square of the channel's
    quantity (g² for a density in g²/Hz) and the rms in the quantity.

    Parameters
    ----------
    specification : SpecificationDraft or Specification
        The specification, as a sheet on its breakpoints or as an
        object written at breakpoints (`interpolation` 'log_log'). An
        object is read at the lines where its channel's autospectrum is
        positive.
    channel : str, optional
        The channel drawn; the first by default.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.

    Raises
    ------
    ValueError
        For a specification read as a density per bin (`interpolation`
        'bin', a controller's target on its lines) or a sheet on its
        lines: the power law between its lines is not the curve its rms
        is the area of, so the areas would add to an rms the object
        does not have. Open it as a sheet and read `to_breakpoints`.
    """
    from ..core.author import SpecificationDraft
    from ..core.breakpoints import read_breakpoints
    from .breakpoints import build_breakpoints

    if isinstance(specification, SpecificationDraft):
        draft = specification
    else:
        # a density per bin integrates as bins: 6.799 g for a profile
        # of 6.784 g put on 20 Hz lines, where the power law between
        # those lines reads 6.784, a picture of a number nobody computed
        if getattr(specification, 'interpolation', 'bin') != 'log_log':
            raise ValueError(
                'the specification is a density per line, integrated as '
                'bins, not a breakpoint table; open it as a sheet and read '
                'its breakpoints (SpecificationDraft.to_breakpoints)')
        draft = SpecificationDraft.from_specification(
            specification, None if channel is None else [channel])
    if channel is not None and channel not in draft.channels:
        raise ValueError(f'no channel {channel!r} in the specification')
    k = 0 if channel is None else draft.channels.index(channel)
    # an object written at breakpoints on an even grid opens on the
    # lines form, and is still a power law between its points
    reading = (draft.reading(k) if draft is specification
               else read_breakpoints(draft.frequencies, draft.levels[k]))
    name = draft.channels[k]

    def build(widget: Any) -> Any:
        widget.clear()
        build_breakpoints(widget, reading, draft.dims[k], name=name,
                          unit_system=unit_system, theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'{name}: segment areas and running rms',
                        size=size)


def plot_chance(measured: Any, specification: Any, *,
                channel: str | None = None, combine: int = 1,
                dof: float | None = None, scale_db: float | None = None,
                theme: Any = None,
                path: str | os.PathLike | None = None,
                show: bool = True, title: str | None = None,
                size: tuple[int, int] = (1000, 700)) -> Any:
    """One channel of a random comparison as its miss from the target in
    dB, line by line, with the lines outside each band counted beside
    the number chance alone puts there at the comparison's degrees of
    freedom.

        asd = history.compute_psds()
        visualdynamics.plot.plot_chance(asd, spec, combine=8,
                                        path='chance.png')

    The reading is `core.chance.compare_chance`'s: the target at zero,
    the warning edges dashed and the abort edges solid about it, the
    measurement over the target as a step across each line (or group of
    `combine` lines, drawn over the single lines in gray), and each line
    outside marked with a triangle pointing the way it is out, hollow
    past the warning band and filled past abort. The title and the
    legend give each count and its expectation.

    Parameters
    ----------
    measured : Psd
        The averaged autospectrum, on narrowband lines.
    specification : Specification
        The specification it was controlled to, with its bands.
    channel : str, optional
        The channel drawn; the first the two have in common by default.
    combine, dof, scale_db
        As for `core.chance.compare_chance`: lines judged together, one
        line's degrees of freedom (read from the PSD when omitted), and
        the comparison's scale.
    theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from .chance import _record_pair, build_chance

    def build(widget: Any) -> Any:
        widget.clear()
        build_chance(widget, specification, measured, channel=channel,
                     combine=combine, dof=dof, scale_db=scale_db,
                     theme=theme)

    label = _record_pair(specification, measured, channel)[0]
    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'{label} against chance',
                        size=size)


def plot_kurtosis(history: Any, *, records: Sequence[int] | None = None,
                  low: float | None = None, high: float | None = None,
                  theme: Any = None,
                  path: str | os.PathLike | None = None,
                  show: bool = True, title: str | None = None,
                  size: tuple[int, int] | None = None) -> Any:
    """Draw each channel's Pearson kurtosis as a bar chart.

        visualdynamics.plot.plot_kurtosis(history, path='kurtosis.png')

    The kurtosis is 3 for a Gaussian record. Above the band between the
    thresholds, the record has peaks the spectrum does not predict;
    below it, the record has been clipped or was never random. Every
    channel is drawn on one axis whatever it measures, since kurtosis is
    dimensionless. The thresholds `low` and `high` default to one either
    side of three (`core.kurtosis.LOW` and `HIGH`), and `records`
    selects the channels. A channel with no finite kurtosis is left out,
    and a `ValueError` is raised when none is left. `size` defaults to a
    height that fits the channels. Returns the pane, or the path
    written.
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


def plot_time_statistics(history: Any, record: int = 0, *,
                         full_scale: float | None = None,
                         unit_system: UnitSystem | None = None,
                         theme: Any = None,
                         path: str | os.PathLike | None = None,
                         show: bool = True, title: str | None = None,
                         size: tuple[int, int] = (1000, 980),
                         **options: Any) -> Any:
    """Draw one channel's time statistics: the trace, its histogram
    against the Gaussian, and its kurtosis frame by frame.

        visualdynamics.plot.plot_time_statistics(
            history, 0, full_scale=25 * 9.80665, path='statistics.png')

    Three rows (`plot.time_statistics`): the trace with the range drawn
    across it and every sample on the rail marked; the histogram as a
    density beside the bell of the record's own mean and standard
    deviation, with 3, 4 and 5 sigma marked and the share beyond each
    against the Gaussian's; the kurtosis against time inside its frame
    band, with the record's own kurtosis across it. The top row is
    titled with the judgement (`TimeStatistics.describe`). This is the
    script form of the Statistics reading on a time history's bar, and
    the numbers are `TimeHistory.time_statistics`.

    Parameters
    ----------
    history : TimeHistory
        The record.
    record : int, default 0
        Which channel.
    full_scale : float, optional
        The channel's range, in SI: the rail is drawn and counted only
        when it is given.
    unit_system, theme, path, show, title, size
        As for every standalone plot.
    **options
        `margin`, `frame`, `duration` and the bands, as
        `core.time_statistics.time_statistics` takes them.

    Returns
    -------
    object
        The pane, or the path written.

    Raises
    ------
    ValueError
        If `record` names no channel, or as
        `core.time_statistics.time_statistics` raises.
    """
    from .time_statistics import build_time_statistics

    stats = history.time_statistics(record, full_scale=full_scale,
                                    **options)

    def build(widget: Any) -> Any:
        widget.clear()
        build_time_statistics(widget, history, stats, record,
                              unit_system=unit_system, theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or (f'{history.record_label(record)}: '
                                        'time statistics'),
                        size=size)



def plot_amplitude_spectrum(history: Any, record: int = 0, *,
                            averaging: Any = None, window: str = 'hann',
                            parameter: float | None = None,
                            log: bool = True,
                            band: tuple[float, float] | None = None,
                            unit_system: UnitSystem | None = None,
                            theme: Any = None,
                            path: str | os.PathLike | None = None,
                            show: bool = True, title: str | None = None,
                            size: tuple[int, int] = (1000, 620)) -> Any:
    """Draw one channel's amplitude spectrum, its window and correction
    named on the plot and its tallest line read.

        visualdynamics.plot.plot_amplitude_spectrum(
            history, 0, window='flattop', path='amplitude.png')

    Each line is scaled so a sine of amplitude A on it reads A, in peak
    units, and the frames' magnitudes are averaged
    (`TimeHistory.amplitude_spectrum`, `plot.amplitude_spectrum`). The
    title names the window, its amplitude correction, the line spacing,
    the frames averaged and how far low the window reads a sine midway
    between lines; the tallest line above DC is marked with a triangle
    and its amplitude written beside it. This is the script form of the
    Amplitude reading on a time history's bar.

    Parameters
    ----------
    history : TimeHistory
        The record.
    record : int, default 0
        Which record of `history`, as every plot of a history counts
        them; a channel saved as captures is drawn as its one row
        (`core.amplitude_spectrum.channel_row`), so any of its captures
        names it.
    averaging : Averaging, optional
        The frames, as `TimeHistory.amplitude_spectrum` takes them.
    window : str, default 'hann'
        The window; 'flattop' for a tone's amplitude wherever it falls.
    parameter : float, optional
        The Tukey window's alpha or the Kaiser window's beta.
    log : bool, default True
        Draw the amplitude on a logarithmic axis.
    band : (float, float), optional
        The band, in Hz, the tallest line is looked for in.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.

    Raises
    ------
    ValueError
        As `TimeHistory.amplitude_spectrum` raises, or for a record the
        history does not hold.
    """
    from ..core.amplitude_spectrum import channel_row
    from .amplitude_spectrum import build_amplitude_spectrum

    count = len(history.response_dof)
    if not 0 <= int(record) < count:
        raise ValueError(f'record {record} is not one of the {count} '
                         'records of this history')
    spectrum = history.amplitude_spectrum(averaging, window=window,
                                          parameter=parameter)
    record = channel_row(history, int(record))

    def build(widget: Any) -> Any:
        widget.clear()
        build_amplitude_spectrum(widget, spectrum, record,
                                 unit_system=unit_system, theme=theme,
                                 log=log, band=band)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or (f'{spectrum.label(record)}: '
                                        'amplitude spectrum'),
                        size=size)


def plot_pulse_check(history: Any, shape: str, peak: float,
                     duration: float, edge: float | None = None, *,
                     channel: int | str | None = None,
                     band: float | None = None,
                     duration_tolerance: float | None = None,
                     edge_limit: float | None = None,
                     velocity_tolerance: float | None = None,
                     quiet: float | None = None,
                     unit_system: UnitSystem | None = None,
                     theme: Any = None,
                     path: str | os.PathLike | None = None,
                     show: bool = True, title: str | None = None,
                     size: tuple[int, int] = (1000, 850)) -> Any:
    """Draw a measured classical pulse on its tolerance envelope.

        visualdynamics.plot.plot_pulse_check(history, 'half-sine',
                                             50 * 9.80665, 0.011,
                                             path='pulse.png')

    Two rows for one acceleration channel: the record on the envelope
    of the nominal pulse, the samples outside it marked and the
    velocity window shaded, with the verdict as its title; and the
    velocity across that window against the nominal change and its
    tolerance (`core.classical_pulse.check_pulse`). The pulse is given
    as `core.classical_pulse.ClassicalPulse` takes it, the peak in SI.

    Parameters
    ----------
    history : TimeHistory
        The shock record.
    shape : str
        'half-sine', 'sawtooth' or 'trapezoid'.
    peak : float
        The nominal peak, signed, in m/s².
    duration : float
        The nominal duration, in seconds.
    edge : float, optional
        The sawtooth's fall or the trapezoid's rise and fall, in
        seconds.
    channel : int or str, optional
        The record's row or DOF; its first acceleration channel by
        default.
    band, duration_tolerance, edge_limit, velocity_tolerance, quiet : float, optional
        The tolerances, as shares (the quiet in durations); the
        `core.classical_pulse` defaults when omitted.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.

    Raises
    ------
    ValueError
        If the pulse is not a classical pulse, the channel is not an
        acceleration, or the pulse cannot be found in the record.
    """
    from ..core import classical_pulse as pulses
    from .classical_pulse import build_pulse_check

    pulse = pulses.ClassicalPulse(shape, peak, duration, edge)
    given = {'band': band, 'duration_tolerance': duration_tolerance,
             'edge_limit': edge_limit,
             'velocity_tolerance': velocity_tolerance, 'quiet': quiet}
    check = pulses.check_pulse(
        history, pulse, channel,
        **{key: value for key, value in given.items() if value is not None})
    dof = str(history.response_dof[pulses.pulse_channel(history, channel)])

    def build(widget: Any) -> Any:
        widget.clear()
        build_pulse_check(widget, check, unit_system=unit_system,
                          theme=theme, name=dof)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'{dof}: classical pulse check',
                        size=size)

def plot_replication_fit(history: Any,
                         rating: Any,
                         channel: str | int | None = None, *,
                         cutoffs: Sequence[float] | None = None,
                         cutoff: Any = ...,
                         unit_system: UnitSystem | None = None,
                         theme: Any = None,
                         path: str | os.PathLike | None = None,
                         show: bool = True, title: str | None = None,
                         size: tuple[int, int] = (1100, 800)) -> Any:
    """Draw whether a shaker can play a reference record.

        rating = {'force': 17800.0, 'stroke': 0.0508, 'velocity': 1.78,
                  'moving_mass': 68.0, 'body_mass': 1360.0}
        visualdynamics.plot.plot_replication_fit(history, rating,
                                                 path='fit.png')

    The check is `core.replication_fit.replication_fit`. The top row is
    the armature's travel through the record at one cutoff, the body's
    share counted, inside the band the stroke allows, with the verdict
    and what the record does integrated as logged written above it. The
    bottom row is the sweep across the candidate cutoffs: the travel
    each way against the room, the peak velocity against the rating
    (the body counted, as in the travel),
    and the share of the mean square removed with the change in the
    largest peak. A filled marker fits and a hollow one does not.

    Parameters
    ----------
    history : TimeHistory
        The reference record.
    rating : ShakerRating or mapping
        The shaker's ratings in SI, as a `core.shaker.ShakerRating` or
        the keyword arguments that make one.
    channel : str or int, optional
        The acceleration channel, by DOF or index; the first by default.
    cutoffs : sequence of float, optional
        The candidate high-pass cutoffs in Hz;
        `core.replication_fit.CUTOFFS` by default.
    cutoff : float or None, optional
        The candidate drawn on the top row; None draws the record as
        logged. Omitted, the lowest cutoff that fits, or the lowest
        candidate when none does.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.

    Raises
    ------
    ValueError
        If the record has no acceleration channel, `channel` is not
        one, or a rating or cutoff is out of range.
    KeyError
        If `cutoff` is not one of the candidates.
    """
    from ..core.replication_fit import CUTOFFS, replication_fit
    from ..core.shaker import ShakerRating
    from .replication_fit import build_replication_fit, default_cutoff

    if not isinstance(rating, ShakerRating):
        rating = ShakerRating(**dict(rating))
    fit = replication_fit(history, rating,
                          cutoffs=CUTOFFS if cutoffs is None else cutoffs,
                          channel=channel)
    drawn = fit.case(default_cutoff(fit) if cutoff is ... else cutoff)

    def build(widget: Any) -> Any:
        widget.clear()
        build_replication_fit(widget, fit, drawn.cutoff,
                              unit_system=unit_system, theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or (f'{fit.dof}: replication fit, '
                                        f'{drawn.name()}'),
                        size=size)


def plot_stroke(history: Any, channel: int | str | None = None, *,
                stroke: float | None = None, velocity: float | None = None,
                force: float | None = None,
                moving_mass: float | None = None,
                body_mass: float | None = None,
                unit_system: UnitSystem | None = None, theme: Any = None,
                path: str | os.PathLike | None = None,
                show: bool = True, title: str | None = None,
                size: tuple[int, int] = (1000, 900)) -> Any:
    """Draw a shock pulse's acceleration, velocity and displacement
    against the shaker's shock ratings.

        visualdynamics.plot.plot_stroke(pulse, stroke=0.0635,
                                        velocity=1.778, moving_mass=45.4,
                                        body_mass=181.4, path='stroke.png')

    Three rows on one clock for one channel, read by
    `core.stroke.read_stroke`: the acceleration titled with its peak
    and the force it asks for, the velocity integrated raw with the
    shock velocity drawn either side of zero, and the displacement with
    the stops drawn half the stroke either side of center, where the
    armature rests. With `body_mass` and `moving_mass` the motion
    relative to the body is drawn beside the table's, dashed. Each row's title is its
    reading in words, against its rating where one was given.

    Parameters
    ----------
    history : TimeHistory
        The shock record, or a pulse from
        `core.stroke.compensated_half_sine`.
    channel : int or str, optional
        The record's index or DOF; its first acceleration channel by
        default.
    stroke, velocity, force, moving_mass, body_mass : float, optional
        The shaker's shock ratings and the masses, in SI, as
        `core.stroke.read_stroke` takes them.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.

    Raises
    ------
    ValueError
        If the record has no acceleration channel, `channel` is not
        one, or a rating is negative.
    """
    from ..core.stroke import read_stroke
    from .stroke import build_stroke

    reading = read_stroke(history, channel, stroke=stroke,
                          velocity=velocity, force=force,
                          moving_mass=moving_mass, body_mass=body_mass)

    def build(widget: Any) -> Any:
        widget.clear()
        build_stroke(widget, reading, unit_system=unit_system, theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'{reading.dof}: stroke, velocity '
                                       'and force',
                        size=size)


def plot_resample(history: Any, rate: float, record: int = 0, *,
                  frequencies: float | Sequence[float] | None = None,
                  order: int | None = None,
                  frame_length: int | None = None,
                  unit_system: UnitSystem | None = None, theme: Any = None,
                  path: str | os.PathLike | None = None,
                  show: bool = True, title: str | None = None,
                  size: tuple[int, int] = (1000, 700)) -> Any:
    """Draw what taking a record to a lower sample rate does, on its
    autospectrum.

        visualdynamics.plot.plot_resample(history, 5120.0,
                                          frequencies=[4600.0],
                                          path='resample.png')

    One channel read by `core.resample.read_resample` and drawn by
    `plot.resample`: the record as recorded, filtered first and then
    decimated, decimated with no filter, and decimated then low-passed
    at the span, on one line spacing; the new Nyquist frequency and the
    span's top marked, the dropped lines between them hatched, and each
    frequency above the new Nyquist frequency joined to where it folds,
    with what each order of filtering left there in words. Where the
    bare and the low-passed-after curves lie on each other at an alias,
    the low-pass came too late. This is the script form of the
    Resample reading on a time history's bar.

    Parameters
    ----------
    history : TimeHistory
        The record, evenly sampled.
    rate : float
        The new sample rate, in Hz, a whole division of the record's.
    record : int, default 0
        Which channel.
    frequencies : float or sequence of float, optional
        Frequencies above the new Nyquist frequency to fold, in Hz; the
        record's tallest line above it when omitted.
    order : int, optional
        The anti-alias filter's design order, `core.resample.ORDER`
        when omitted.
    frame_length : int, optional
        The spectra's frame at the new rate, in samples.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.

    Raises
    ------
    ValueError
        As `core.resample.read_resample` raises.
    """
    from ..core.resample import ORDER, read_resample
    from .resample import build_resample

    reading = read_resample(history, rate, record, frequencies=frequencies,
                            order=ORDER if order is None else order,
                            frame_length=frame_length)

    def build(widget: Any) -> Any:
        widget.clear()
        build_resample(widget, reading, unit_system=unit_system,
                       theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'{reading.label}: resampled to '
                                       f'{reading.rate:g} Hz',
                        size=size)


def plot_snr(signal: Any, floor: Any, *,
             records: Sequence[int] | None = None,
             low: float | None = None, theme: Any = None,
             path: str | os.PathLike | None = None,
             show: bool = True, title: str | None = None,
             size: tuple[int, int] | None = None) -> Any:
    """Draw each channel's RMS signal-to-noise ratio as a bar chart.

        visualdynamics.plot.plot_snr(driven_psds, noise_psds,
                                     path='snr_bars.png')

    The ratio is in dB. This is the script form of the toolbar's RMS
    signal to noise reading of two selected PSDs. Each channel's power
    is the area under its density over the band the two share, and the
    reading is 10 log10((P_driven - P_ambient) / P_ambient)
    (`visualdynamics.core.snr`). `low` is the floor a channel is judged
    against, `core.snr.THRESHOLD_DB` (10 dB) by default. A channel with
    no signal above its noise has no bar and is labeled as at the noise
    floor, and one whose ambient recorded nothing is left out. When no
    channel is left, a `ValueError` is raised. Returns the pane, or the
    path written.
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


def plot_shock_check(history: Any, channel: str | None = None, *,
                     shock: int = 0, shocks: Sequence[Any] | None = None,
                     tolerance: float | None = None,
                     q: float | None = None, per_octave: int | None = None,
                     low: float | None = None, high: float | None = None,
                     unit_system: UnitSystem | None = None,
                     theme: Any = None,
                     path: str | os.PathLike | None = None,
                     show: bool = True, title: str | None = None,
                     size: tuple[int, int] = (1000, 900)) -> Any:
    """Draw a shock record's velocity check beside its spectra.

        visualdynamics.plot.plot_shock_check(history, '101Z+',
                                             path='check.png')

    Three rows for one channel over one window: the acceleration, its
    velocity integrated raw (`core.shock_check`), and the positive and
    negative shock response spectra drawn together. The velocity row
    is titled with the verdict in words and shades the band either side
    of zero that the velocity has to end inside. A record whose
    velocity does not come back to rest is not one to report a
    spectrum from, and this is where that is seen before the spectrum
    is read.

    Parameters
    ----------
    history : TimeHistory
        The shock record.
    channel : str, optional
        The DOF drawn; the record's first acceleration channel by
        default.
    shock : int, default 0
        Which window, counting from 0, of `TimeHistory.srs_windows`.
    shocks : sequence of Shock, optional
        Windows to use in place of the record's own.
    tolerance : float, optional
        How far from rest the velocity may end, as a share of its peak;
        `core.shock_check.RETURN_SHARE` by default.
    q : float, optional
        The oscillator amplification; `core.srs.DEFAULT_Q` by default.
    per_octave : int, optional
        Natural-frequency lines per octave; `core.srs.PER_OCTAVE` (12)
        by default.
    low, high : float, optional
        The natural-frequency band, in Hz; `TimeHistory.srs_band` of
        the window by default.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.

    Raises
    ------
    ValueError
        If the record has no acceleration channel, `channel` is not
        one or names more than one, or `shock` names no window.
    """
    from dataclasses import replace

    from ..core.shock_check import RETURN_SHARE, check_shock
    from .shock_check import build_shock_check

    windows = history.srs_windows(shocks)
    if not 0 <= shock < len(windows):
        raise ValueError(f'the record has {len(windows)} window'
                         f'{"s" * (len(windows) != 1)}; there is no '
                         f'window {shock}')
    window = [windows[shock]]
    checks = check_shock(history, shocks=window,
                         tolerance=(RETURN_SHARE if tolerance is None
                                    else tolerance))
    dof = checks[0].dof if channel is None else channel
    found = [check for check in checks if check.dof == dof]
    if not found:
        raise ValueError(f'{dof} is not an acceleration channel of this '
                         'record, and only acceleration is checked')
    if len(found) > 1:
        raise ValueError(f'{len(found)} acceleration channels of this '
                         f'record are named {dof}, so which to draw is '
                         'not clear')
    check, = found
    spectra = history.compute_srs_polarities(window, low, high,
                                             per_octave, q)
    # the check counts windows within the one it was given; named back
    # to the record's own count so the picture says which event it is
    check = replace(check, window=shock)

    def build(widget: Any) -> Any:
        widget.clear()
        build_shock_check(widget, history, check, spectra,
                          unit_system=unit_system, theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or (f'{dof}, shock {shock + 1}: '
                                        'velocity check and spectra'),
                        size=size)


def plot_scalogram(history: Any, channel: int = 0, *,
                   low: float | None = None, high: float | None = None,
                   per_octave: int | None = None,
                   omega0: float | None = None,
                   theme: Any = None,
                   path: str | os.PathLike | None = None,
                   show: bool = True, title: str | None = None,
                   size: tuple[int, int] | None = None) -> Any:
    """Draw one channel's scalogram, its frequency content over time.

        from visualdynamics.plot import plot_scalogram
        plot_scalogram(history, path='scalogram.png')

    This is the flat form of the app's wavelet reading: time across,
    frequency up a logarithmic axis, and magnitude as color in the
    record's own units. The cone of influence is shaded, because there
    the values depend on where the record was cut and look the same as
    the rest of the data. Time is reduced to `core.wavelet.COLUMNS`
    columns by peak-hold, each column the largest magnitude in its slice
    (the reading `core.wavelet.scalogram_peaks` gives the app's own
    view), so a long record is drawn in bounded memory. A script that
    needs the transform itself calls `core.wavelet.scalogram`.

    Parameters
    ----------
    history : TimeHistory
        The record to read.
    channel : int, default 0
        Which channel. One is drawn at a time, because a scalogram is
        dense enough that two side by side are hard to read.
    low, high : float, optional
        The frequency range, in Hz. By default it runs from where a few
        of the longest wavelets still fit inside the record up to just
        under Nyquist (`core.wavelet.default_range`).
    per_octave : int, optional
        Lines per octave. Defaults to `core.wavelet.PER_OCTAVE`.
    omega0 : float, optional
        Cycles under the wavelet, which trades resolution in time
        against resolution in frequency. Defaults to
        `core.wavelet.OMEGA0`.
    theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from ..core import wavelet as wavelet_math
    from .scalogram import scalogram_image

    # the app's own reading, flat here and a surface on the stage
    clock, magnitude, frequencies, width = wavelet_math.reading(
        history, channel, low=low, high=high, per_octave=per_octave,
        omega0=omega0)
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


def plot_sine_tracking(history: Any, specification: Any, settings: Any, *,
                       tone: str | None = None, channel: str | None = None,
                       onset: float | None = None, lines: int | None = None,
                       unit_system: UnitSystem | None = None,
                       theme: Any = None,
                       path: str | os.PathLike | None = None,
                       show: bool = True, title: str | None = None,
                       size: tuple[int, int] = (1000, 700)) -> Any:
    """Draw one tone's levels read with several tracking settings.

        from visualdynamics.core.sine_tracking import SineTracking
        visualdynamics.plot.plot_sine_tracking(
            history, spec,
            [SineTracking(detector='peak'),
             SineTracking(proportional=0.5),
             SineTracking(proportional=0.1)],
            path='tracking.png')

    The tone is read through several tracking bands and detectors, and
    the levels are overlaid against frequency over the specification.
    The readings come from `core.sine_tracking.track_sine`, one per
    setting, on one channel. Each is drawn in its own color and with its
    own marker shape and is named by `SineTracking.describe`, and the
    specification's target for the tone is drawn in gray behind them.

    Parameters
    ----------
    history : TimeHistory
        The recording.
    specification : SineSweepSpecification
        The sweep the drive followed.
    settings : SineTracking or sequence of SineTracking
        The tracking settings, one reading each.
    tone : str, optional
        Which tone; the specification's first by default.
    channel : str, optional
        The DOF drawn; the specification's first control channel by
        default.
    onset : float, optional
        Seconds into the recording where the tone's sweep begins; found
        by matched filter when omitted.
    lines : int, optional
        Readings along the sweep, `core.sine_tracking.LINES` by default.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from ..core.sine_tracking import LINES, SineTracking, track_sine
    from .sine_tracking import build_sine_tracking

    if isinstance(settings, SineTracking):
        settings = [settings]
    settings = list(settings)
    dof = specification.response_dof[0] if channel is None else channel
    levels = track_sine(history, specification, settings, tone=tone,
                        onset=onset, channels=[dof],
                        lines=LINES if lines is None else lines)

    def build(widget: Any) -> Any:
        widget.clear()
        build_sine_tracking(widget, levels, settings,
                            specification=specification,
                            unit_system=unit_system, theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or (f'{levels[0].tone} at {dof}, '
                                        'read through tracking filters'),
                        size=size)


def plot_tracking_filter(history: Any, specification: Any, setting: Any, *,
                         tone: str | None = None,
                         channel: str | None = None,
                         onset: float | None = None,
                         cursor: float | None = None,
                         cursor_hz: float | None = None,
                         span: tuple[float, float] | None = None,
                         harmonics: Sequence[int] | None = None,
                         low: float | None = None,
                         high: float | None = None,
                         omega0: float | None = None,
                         unit_system: UnitSystem | None = None,
                         theme: Any = None,
                         path: str | os.PathLike | None = None,
                         show: bool = True, title: str | None = None,
                         size: tuple[int, int] = (1100, 760)) -> Any:
    """Draw one tone's tracking band on the record it is applied to.

        from visualdynamics.core.sine_tracking import SineTracking
        visualdynamics.plot.plot_tracking_filter(
            history, spec, SineTracking(proportional=0.5),
            cursor_hz=100.0, path='band.png')

    The plot shows the record and the band's output, the record's
    time-frequency picture with the band's corridor following the sweep,
    and the band's shape at a cursor. The passed waveform comes from
    `core.sine_tracking.track_waveform`, through the same band
    `track_sine` reads its levels with. The picture is the wavelet
    scalogram in dB below its largest value, with the cone of influence
    shaded. The corridor's edges are the drive frequency plus and minus
    half the bandwidth, so a proportional band runs parallel to the tone
    up the log axis and a fixed band narrows toward the top. Beside the
    picture is the band's magnitude at the cursor on the same frequency
    axis, with the drive marked inside it and each harmonic outside. On
    the record, behind the cursor, the band's weight on the earlier
    record is shaded over the time the band takes to settle. In a window
    the cursor can be dragged; in a file it is fixed. One setting is
    drawn per call; `plot_sine_tracking` overlays the levels read with
    several settings.

    Parameters
    ----------
    history : TimeHistory
        The recording.
    specification : SineSweepSpecification
        The sweep the drive followed.
    setting : SineTracking
        The tracking setting, which must have a band (proportional or
        fixed).
    tone : str, optional
        Which tone; the specification's first by default.
    channel : str, optional
        The DOF drawn; the first control channel by default.
    onset : float, optional
        Seconds into the recording where the tone's sweep begins; found
        by matched filter when omitted.
    cursor : float, optional
        Where the shape is read, in seconds on the record's clock; the
        middle of the tone's span by default.
    cursor_hz : float, optional
        Instead of `cursor`, a frequency in Hz: the cursor is placed
        where the drive first reaches it. Giving both raises
        `ValueError`.
    span : (float, float), optional
        The stretch of record the time axes show; the whole record by
        default.
    harmonics : sequence of int, optional
        The multiples of the drive marked, `(3,)` by default; empty to
        mark none.
    low, high : float, optional
        The picture's frequency range, Hz. By default from an octave
        below the sweep up to 1.5 times the highest harmonic marked, or
        the band's top edge if that is higher
        (`plot.tracking_filter.default_range`).
    omega0 : float, optional
        Cycles under the picture's wavelet,
        `plot.tracking_filter.OMEGA0` by default.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from ..core.sine_tracking import track_waveform
    from .tracking_filter import HARMONICS, OMEGA0, build_tracking_filter

    if cursor is not None and cursor_hz is not None:
        raise ValueError('the cursor is placed in seconds or in hertz, '
                         'not both')
    waveform = track_waveform(history, specification, setting, tone=tone,
                              onset=onset, channel=channel)
    if cursor_hz is not None:
        cursor = waveform.instant(cursor_hz)

    def build(widget: Any) -> Any:
        widget.clear()
        return build_tracking_filter(
            widget, history, waveform, cursor=cursor, span=span,
            harmonics=HARMONICS if harmonics is None else harmonics,
            low=low, high=high, omega0=OMEGA0 if omega0 is None else omega0,
            unit_system=unit_system, theme=theme,
            interactive=path is None)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or (f'{waveform.tone} at '
                                        f'{waveform.dof}, '
                                        f'{setting.describe()}'),
                        size=size)


def plot_sweep_rate(specification: Any, modes: Any, *,
                    tone: str | None = None,
                    margin: float | None = None,
                    theme: Any = None,
                    path: str | os.PathLike | None = None,
                    show: bool = True, title: str | None = None,
                    size: tuple[int, int] = (1000, 700)) -> Any:
    """One sine tone's sweep through the modes it passes: frequency
    against time, each mode's half-power band a strip, and the time and
    cycles the sweep spends in each written beside its crossing.

        visualdynamics.plot.plot_sweep_rate(
            spec, [(22.0, 40.0), (180.0, 20.0)], path='sweep_rate.png')

    The checks are `core.sweep_rate.check_sweep_rate`'s. A crossing the
    sweep makes slowly enough for the mode to build is solid and marked
    with a circle; one above the rule's rate (about 87 fn / Q**2 oct/min
    on a log sweep) is dashed, marked with a cross, and labeled too
    fast. Each label gives the seconds and cycles in the band, the Q
    the mode needs, and the rate as a multiple of the rule's.

    Parameters
    ----------
    specification : SineSweepSpecification or SineTone
        The sweep.
    modes : iterable of (frequency, Q), or ShapeSet
        The modes, natural frequency in Hz with its Q.
    tone : str, optional
        The tone drawn; the specification's first by default.
    margin : float, optional
        The multiple of the rule's rate a crossing may reach before it
        is flagged; `core.sweep_rate.MARGIN` (1) by default.
    theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from ..core.sweep_rate import MARGIN
    from .sweep_rate import build_sweep_rate

    modes = list(modes) if not hasattr(modes, 'damping') else modes

    def build(widget: Any) -> Any:
        widget.clear()
        build_sweep_rate(widget, specification, modes, tone=tone,
                         margin=MARGIN if margin is None else margin,
                         theme=theme)

    name = tone or getattr(specification, 'name', None) or \
        specification.tones[0].name
    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'{name}: sweep rate against the modes',
                        size=size)


def plot_dwell(history: Any, specification: Any, *, tone: str | None = None,
               segment: int | None = None, channel: str | None = None,
               setting: Any = None, share: float | None = None,
               onset: float | None = None, after: float | None = None,
               unit_system: UnitSystem | None = None, theme: Any = None,
               path: str | os.PathLike | None = None, show: bool = True,
               title: str | None = None,
               size: tuple[int, int] = (1000, 620)) -> Any:
    """Draw a dwell's level against time, the shares it is read by
    marked.

        visualdynamics.plot.plot_dwell(history, spec, channel='101Z+',
                                       path='dwell.png')

    The dwell is read by `core.dwell.read_dwell`, with the same
    arguments, and drawn by `plot.dwell.build_dwell`: the tracked level
    of every cycle where it stands, the steady level in gray, the share
    dashed and its complement dotted, a filled circle where the level
    first reaches the share and a filled triangle where the ring-down
    falls to its complement, each named with its count of cycles in
    the legend. The window's sine view reads the same dwell in its
    Dwell readout.

    Parameters
    ----------
    history : TimeHistory
        The recording.
    specification : SineSweepSpecification
        What the drive followed; one of its tones holds the dwell.
    tone, segment, channel, setting, onset, after
        As for `core.dwell.read_dwell`.
    share : float, optional
        The share of the steady level the build-up is read to,
        `core.dwell.DWELL_SHARE` by default.
    unit_system, theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from ..core.dwell import DWELL_SHARE, read_dwell
    from .dwell import build_dwell

    reading = read_dwell(history, specification, tone=tone, segment=segment,
                         channel=channel, setting=setting,
                         share=DWELL_SHARE if share is None else share,
                         onset=onset, after=after)

    def build(widget: Any) -> Any:
        widget.clear()
        return build_dwell(widget, reading, unit_system=unit_system,
                           theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or (f'{reading.tone} at {reading.dof}, '
                                        'level against time'),
                        size=size)


def plot_sine_distortion(history: Any, specification: Any,
                         levels: Any = None, *, tone: str | None = None,
                         channel: str | None = None,
                         excess: float | None = None,
                         cycles: float | None = None, theme: Any = None,
                         path: str | os.PathLike | None = None,
                         show: bool = True, title: str | None = None,
                         size: tuple[int, int] = (1000, 620)) -> Any:
    """Draw a sine tone's distortion against frequency, the limit
    across it.

        visualdynamics.plot.plot_sine_distortion(history, spec, levels,
                                                 path='distortion.png')

    The check is `core.sine_distortion.check_distortion`, with the same
    arguments: the RMS of the record less the tones the extraction
    tracked against the RMS of the tones, at every line of `levels`
    (extracted here when omitted). It is drawn by `plot.sine_distortion.build_distortion`:
    the limit dashed, each control channel (or only `channel`) as a solid
    curve with its own hollow marker, the same marker filled on a line
    over the limit, and each named in the legend with its worst reading
    and its verdict in words. The window's sine view reads the same check
    in its Distortion readout.

    Parameters
    ----------
    history : TimeHistory
        The recording.
    specification : SineSweepSpecification
        The sweep the drive followed.
    levels : SineLevelSet, optional
        The levels extracted from `history`; extracted when omitted.
    tone, cycles
        As for `core.sine_distortion.check_distortion`.
    channel : str, optional
        The one control channel drawn; every channel the levels hold by
        default.
    excess : float, optional
        The share the unfiltered RMS may stand above the tone's,
        `core.sine_distortion.DISTORTION_EXCESS` by default.
    theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from ..core.sine_distortion import DISTORTION_EXCESS, check_distortion
    from .sine_distortion import build_distortion

    check = check_distortion(
        history, specification, levels, tone=tone,
        channels=None if channel is None else [channel],
        excess=DISTORTION_EXCESS if excess is None else excess,
        cycles=cycles)

    def build(widget: Any) -> Any:
        widget.clear()
        return build_distortion(widget, check, theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'{check.tone}, distortion',
                        size=size)


def plot_sine_frequency(history: Any, specification: Any,
                        levels: Any = None, *, tone: str | None = None,
                        channel: str | None = None,
                        tolerance: float | None = None, floor: float = 0.0,
                        cycles: float | None = None, theme: Any = None,
                        path: str | os.PathLike | None = None,
                        show: bool = True, title: str | None = None,
                        size: tuple[int, int] = (1000, 620)) -> Any:
    """Draw a sine tone's frequency error against frequency, the band
    either side of zero.

        visualdynamics.plot.plot_sine_frequency(history, spec, levels,
                                                path='frequency.png')

    The check is `core.sine_frequency.check_frequency`, with the same
    arguments: the tone's own frequency on the channel against the
    specification's, at every line of `levels` (extracted here when
    omitted). It is drawn by `plot.sine_frequency.build_frequency`: the
    band dashed, each control channel (or only `channel`) as a solid
    curve with its own hollow marker, the same marker filled on a line
    outside the band, and each named in the legend with its worst reading
    and its verdict in words. The window's sine view reads the same check
    in its Frequency readout.

    Parameters
    ----------
    history : TimeHistory
        The recording.
    specification : SineSweepSpecification
        The sweep the drive followed.
    levels : SineLevelSet, optional
        The levels extracted from `history`; extracted when omitted.
    tone, floor, cycles
        As for `core.sine_frequency.check_frequency`.
    channel : str, optional
        The one control channel drawn; every channel the levels hold by
        default.
    tolerance : float, optional
        The band as a share of the frequency,
        `core.sine_frequency.FREQUENCY_TOLERANCE` by default.
    theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from ..core.sine_frequency import FREQUENCY_TOLERANCE, check_frequency
    from .sine_frequency import build_frequency

    check = check_frequency(
        history, specification, levels, tone=tone,
        channels=None if channel is None else [channel],
        tolerance=FREQUENCY_TOLERANCE if tolerance is None else tolerance,
        floor=floor, cycles=cycles)

    def build(widget: Any) -> Any:
        widget.clear()
        return build_frequency(widget, check, theme=theme)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'{check.tone}, frequency',
                        size=size)


def plot_sine_profile(source: Any, *, tone: str | None = None,
                      channel: str | None = None,
                      tolerance: float | None = None,
                      slope_tolerance: float | None = None,
                      interpolated: bool = True,
                      unit_system: UnitSystem | None = None,
                      theme: Any = None,
                      path: str | os.PathLike | None = None,
                      show: bool = True, title: str | None = None,
                      size: tuple[int, int] = (1000, 900)) -> Any:
    """Draw a sine profile three ways, displacement, velocity and
    acceleration against frequency, with its crossovers marked.

        visualdynamics.plot.plot_sine_profile(profile, path='profile.png')
        visualdynamics.plot.plot_sine_profile(spec, path='profile.png')

    `source` is a `core.sine_profile.SineProfile`, written segment by
    segment, or a `SineSweepSpecification`, whose tone (`tone`, the first
    by default) is read back as a profile at control channel `channel`
    (the first by default) by `core.sine_profile.read_profile`; with
    `interpolated`, the tone's own level as it runs it is drawn dashed
    over the profile read from it. It is drawn by
    `plot.sine_profile.build_profile`: three rows on one log frequency
    axis, each titled with the most the profile asks for in its
    quantity, each crossover a line across the rows with its check in
    words in the legend. The window's Sine Profile reading on a sine
    specification's bar shows the same.

    Parameters
    ----------
    source : SineProfile or SineSweepSpecification
        What is drawn.
    tone, channel : str, optional
        For a specification, which tone and which control channel.
    tolerance : float, optional
        How far a breakpoint may stand from its solved crossover and
        still check, a share of it;
        `core.sine_profile.CROSSOVER_TOLERANCE` for a specification and
        the profile's own for a profile, by default.
    slope_tolerance : float, optional
        For a specification, how far a stretch's slope may stand from
        a quantity held constant, dB per octave;
        `core.sine_profile.SLOPE_TOLERANCE` by default.
    interpolated : bool, default True
        For a specification, draw the tone's interpolated level too.
    unit_system : UnitSystem, optional
        The units the rows are read in.
    theme, path, show, title, size
        As for every standalone plot.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from ..core.sine import SineSweepSpecification
    from ..core.sine_profile import (
        CROSSOVER_TOLERANCE,
        SLOPE_TOLERANCE,
        SineProfile,
        read_profile,
    )
    from ..core.sine_profile import interpolated as tone_level
    from .sine_profile import build_profile

    if isinstance(source, SineProfile):
        profile = (source if tolerance is None
                   else SineProfile(source.segments, tolerance=tolerance))
        level, name = None, 'profile'
    elif isinstance(source, SineSweepSpecification):
        profile = read_profile(
            source, tone, channel,
            tolerance=CROSSOVER_TOLERANCE if tolerance is None else tolerance,
            slope_tolerance=(SLOPE_TOLERANCE if slope_tolerance is None
                             else slope_tolerance))
        level = tone_level(source, tone, channel) if interpolated else None
        name = (tone or source.tones[0].name) + ', read as a profile'
    else:
        raise TypeError('plot_sine_profile draws a SineProfile or a '
                        f'SineSweepSpecification, not {type(source).__name__}')

    def build(widget: Any) -> Any:
        widget.clear()
        return build_profile(widget, profile, interpolated=level,
                             unit_system=unit_system, theme=theme, name=name)

    low, high = profile.band
    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'sine profile, {low:g} to {high:g} Hz',
                        size=size)


def plot_bars(measured: Any, specification: Any, mode: str = 'error', *,
              low: float | None = None, high: float | None = None,
              theme: Any = None,
              path: str | os.PathLike | None = None,
              show: bool = True, title: str | None = None,
              size: tuple[int, int] | None = None,
              summary: str = 'over') -> Any:
    """Draw a comparison as one bar per control channel.

        visualdynamics.plot.plot_bars(psds, spec, 'error', path='error.png')

    `mode` picks the reading, as the app's toolbar does. 'error' is the
    level: how far each channel's RMS is from the specification, in dB,
    over the band they share. 'lines' is the shape: the percentage of
    each channel's band outside an abort limit. A channel can be at
    exactly the right level and still be out of tolerance across half
    its band, which is why there are both. 'srs' is the shock reading:
    each channel's SRS against its target, as a signed RMS deviation in
    dB. A pair that cannot be compared (another quantity, only one
    side's units declared, or for an SRS another Q) is drawn as a row
    that says why, and a comparison of two sets of undeclared units
    says so in the count.

    `low` and `high` are the thresholds: -3 and +3 dB for 'error', 10 %
    for 'lines' (which has only `low`), and -3 and +3 dB
    (`core.compliance.SRS_ERROR_DB`) for 'srs'. The chart grows with the
    number of channels, so `size` defaults to a height that fits them. A
    key to the colors is drawn below the chart. `summary` places the
    count: 'over' the bars, in the plot's 'title' (better for a narrow
    printed figure), or 'none'.

    Raises `ValueError` for an unknown `mode`, or when the two have no
    channel in common. Returns the pane, or the path written.
    """
    from ..core.compliance import (
        ERROR_DB,
        LINES_PERCENT,
        channel_errors,
        channel_refusals,
        compare_all,
        comparison_note,
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
        scored = srs_errors(specification, measured)
        errors = [(f'{row.dof} {row.event}' if row.event else row.dof,
                   row.db, 0.0) for row in scored]
        # a pair that cannot be compared keeps its row and says why
        notes = {i: f'not scored: {row.refused}'
                 for i, row in enumerate(scored) if row.refused}
    else:
        scored = compare_all(specification, measured)
        errors = channel_errors(scored)
        notes = channel_refusals(scored)
    # two sets of raw numbers are compared on trust, and the figure
    # says so where a reader of it will look
    raw = comparison_note(scored)
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
                                summary=summary, notes=notes, note=raw)
        elif mode == 'srs':
            from ..core.compliance import SRS_ERROR_DB
            # both sides: the deviation is signed, and under-testing
            # is a different fault from over-testing, not a lesser one
            chart = replication_chart(
                plot, [(label, value) for label, value, _p in errors],
                colors, 'srs_rms',
                low=-SRS_ERROR_DB if low is None else low,
                high=SRS_ERROR_DB if high is None else high, summary=summary,
                notes=notes, note=raw)
        else:
            chart = lines_chart(plot, errors, colors,
                                low=LINES_PERCENT if low is None else low,
                                summary=summary, notes=notes, note=raw)
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
    """Draw how closely a transient test reproduced its waveform.

        visualdynamics.plot.plot_replication(record, target, 'srs', path='srs.png')

    `mode` picks the reading, the same four as the app's bar. 'overlay'
    draws one repeat over its target. 'waveform', 'srs' and 'level' are
    bar charts of three numbers: the difference between the waveforms as
    a percentage of the target, the worst deviation of the shock
    response spectra in dB, and the level error alone in dB.

    `event` is the repeat, counting from zero. When it is left out, the
    bar charts have a bar for every repeat of every channel, and the
    overlay draws the first repeat. No repeat is picked out as the
    worst, because that is a judgment about what the test article is for
    and not a measurement; every repeat is drawn, and
    `core.replication.compare` returns the numbers for the caller to
    decide.

    `channel` is the control DOF the overlay draws, the first by
    default. One channel is drawn at a time because six measured curves
    over six targets share one set of colors, and nothing on the plot
    would then show which curve is the target.

    `low` and `high` are the bar chart's thresholds:
    `core.replication.WAVEFORM_PERCENT` (20 %) and no upper threshold
    for 'waveform', and -3 and +3 dB otherwise. Raises `ValueError` for
    an unknown `mode`, a repeat that is not in the record, a channel
    that is not a control channel, or when no channel can be scored.
    Returns the pane, or the path written.
    """
    from ..core.compliance import ERROR_DB, comparison_note
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
    # a refused channel keeps its row, with why written along it
    kept = [row for row in rows
            if np.isfinite(row[mode]) or row.get('refused')]
    shown = [(row['label'], float(row[mode])) for row in kept]
    notes = {i: f"not scored: {row['refused']}"
             for i, row in enumerate(kept) if row.get('refused')}
    raw = comparison_note(kept)
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
                          low=bounds[0], high=bounds[1], notes=notes,
                          note=raw)

    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'{mode} by control channel',
                        size=size)


def plot_replication_segments(measured: Any, specification: Any,
                              segments: Sequence[Any], *,
                              channel: str | None = None,
                              lag: int | None = None, playing: int = 0,
                              unit_system: UnitSystem | None = None,
                              theme: Any = None,
                              path: str | os.PathLike | None = None,
                              show: bool = True, title: str | None = None,
                              size: tuple[int, int] = (1100, 820),
                              **settings: Any) -> Any:
    """A replicated run against its reference, judged segment by
    segment: the two traces overlaid with the segments shaded, their
    difference beneath, and the verdict by segment in a table.

        visualdynamics.plot.plot_replication_segments(
            run, reference, [(6.0, 30.0, 'stationary', 'rough road'),
                             (31.9, 32.7, 'shock', 'pothole'),
                             (41.0, 51.0, 'transient', 'braking')],
            band=(5.0, 400.0), path='segments.png')

    The judgment is `core.replication_segments.replication_segments`'s,
    and every setting it takes is passed through (`taper_percent`,
    `band`, `asd_db`, `variance_percent`, `envelope_db`, `window`,
    `srs_db`, `peak_percent` and the rest). The reference stands back
    in gray with the control over it; each segment is shaded across
    both plots and labeled with its name, kind and verdict, its edges
    in its kind's line style; the table gives the whole-record rms
    error first, then each segment's share of the record's mean square,
    its own rms error and each check against its tolerance.

    Parameters
    ----------
    measured : TimeHistory
        The run.
    specification : TransientSpecification or TimeHistory
        The reference it was controlled to.
    segments : sequence
        The marks, each a `Segment` or (start, end, kind[, name]).
    channel, lag, playing
        As for `core.replication_segments.aligned_traces`.
    unit_system, theme, path, show, title, size
        As for every standalone plot.
    **settings
        The tolerances, as `core.replication_segments.judge_segments`
        takes them.

    Returns
    -------
    object
        The pane, or the path written.
    """
    from .replication_segments import build_replication_segments

    def build(widget: Any) -> Any:
        widget.clear()
        build_replication_segments(widget, measured, specification,
                                   segments, channel=channel, lag=lag,
                                   playing=playing, unit_system=unit_system,
                                   theme=theme, **settings)

    dof = specification.response_dof[0] if channel is None else channel
    return _plot_window(build, theme=theme, path=path, show=show,
                        title=title or f'{dof} judged by segment',
                        size=size)


__all__ = ['MAX_RECORDS', 'as_series', 'axis_label', 'build_photos',
           'build_plot', 'build_plots', 'compose', 'figure_plots',
           'pin_range', 'plot_bars', 'plot_cmif', 'plot_coherence_map',
           'plot_comparison', 'plot_data', 'plot_mac', 'plot_photos',
           'plot_replication', 'plot_replication_segments',
           'plot_series', 'save_plot', 'to_view']
