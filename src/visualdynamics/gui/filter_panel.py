"""The filter parameters, as a table beside the time history.

The pass band is set here — its kind, its edges, the order — and the
filter they make is drawn underneath them: its magnitude against
frequency, so the shape is read where it is set rather than inferred
from the numbers (Brandon, 2026-08-25). Beside it the figures that
follow, all measured off that same curve.

One kind at a time, chosen outright: a low-pass shows one corner, a
high-pass one, a band-pass its two edges — only what applies is on
screen (principle 3), and the kind cannot disagree with the numbers
because `core.filters.Filtering` derives it from which edges exist.

The panel says what the settings are, reports when they move, and
its Apply Filter button turns the history into a filtered one with
whatever is set at that moment — the project verb a script calls,
the same division of labor as the averaging panel, because it is the
same kind of thing: parameters that ride the record.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.filters import Filtering
from .editors import DoubleSpinBox, SpinBox, commit_on_enter
from .settings_panel import add_derived, panel_grid

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.data import TimeHistory

#: the kinds on offer, in the order the Type box lists them. The keys
#: are `Filtering.kind`'s own words, so the two cannot drift.
KINDS = ('low-pass', 'high-pass', 'band-pass')
KIND_LABELS = ('Low-pass', 'High-pass', 'Band-pass')

#: the least daylight between a band-pass's edges, in Hz — the corner
#: boxes' own resolution, so the constraint and the display agree
EDGE_GAP = 0.1


class FilterResponsePlot(QWidget):
    """The filter's magnitude against frequency: a small log-log plot,
    whose corner lines are the corner controls.

    **What the data sees**, which is `core.filters.response` — the
    squared magnitude, because the filter runs forward and backward.
    Drawn here rather than computed here, so the report could draw the
    same curve from the same numbers.

    Deliberately bare: no legend, no axis titles, decade grid lines
    and a dashed mark at each corner — one for a low- or high-pass,
    two for a band-pass. It sits in a 210-pixel column and answers one
    question — what shape is this — where anything more would be a
    second plot competing with the record beside it.
    """

    #: an edge was dragged; which one ('low' or 'high'), and where in
    #: hertz
    edge_moved = Signal(str, float)

    #: the drag ended; which edge, and where it settled, in hertz
    edge_settled = Signal(str, float)

    #: how tall, in pixels. Enough for the knee to be a knee
    HEIGHT = 130
    #: the floor of the vertical axis, in dB. Below this a filter has
    #: stopped doing anything anyone can measure on real data
    FLOOR = -80.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        import pyqtgraph as pg

        from ..theme import theme as resolve_theme

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.colors: Mapping[str, str] = resolve_theme(None)
        self.widget: Any = pg.PlotWidget()
        self.widget.setFixedHeight(self.HEIGHT)
        self.plot: Any = self.widget.getPlotItem()
        self.plot.setLogMode(x=True, y=False)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        self.plot.setYRange(self.FLOOR, 5.0, padding=0)
        for edge in ('left', 'bottom'):
            axis = self.plot.getAxis(edge)
            axis.setStyle(tickTextOffset=2, tickLength=-3)
            axis.setTextPen(self.colors['plot_foreground'])
            axis.setPen(self.colors['plot_foreground'])
        self.plot.getAxis('left').setWidth(30)
        self.curve: Any = self.plot.plot(
            [], [], pen=pg.mkPen(self.colors['filter_preview'], width=2))
        # The lines *are* the corner controls (Brandon, 2026-08-25) —
        # better than sliders beside them, which would be a second
        # thing to keep in step with the first. `sigDragged` fires
        # only for a real drag, never for the `setValue` that restates
        # a line, so the two cannot chase each other round.
        self.lines: dict[str, Any] = {}
        for which in ('low', 'high'):
            line = pg.InfiniteLine(
                angle=90, movable=True,
                pen=pg.mkPen(self.colors['plot_foreground'],
                             width=1, style=Qt.PenStyle.DashLine),
                hoverPen=pg.mkPen(self.colors['filter_preview'], width=2))
            line.setCursor(Qt.CursorShape.SizeHorCursor)
            line.sigDragged.connect(
                lambda item, which=which: self._dragged(which, item))
            line.sigPositionChangeFinished.connect(
                lambda item, which=which: self._drag_finished(which, item))
            self.plot.addItem(line)
            self.lines[which] = line
        layout.addWidget(self.widget)

    def _dragged(self, which: str, line: Any) -> None:
        """A line moved under the mouse: say which and where, in hertz.

        The axis is in log mode, so a line's position is the exponent
        — the one place in this widget that is true of, and the
        reason the value is raised here rather than by whoever
        listens.
        """
        self.edge_moved.emit(which, float(10.0 ** line.value()))

    def _drag_finished(self, which: str, line: Any) -> None:
        """The mouse let go: say where the edge settled, once.

        `sigPositionChangeFinished` fires only for a real drag ending,
        never for `setValue` — the same guarantee `sigDragged` gives —
        so a programmatic restate cannot masquerade as a release.
        """
        self.edge_settled.emit(which, float(10.0 ** line.value()))

    def apply_theme(self, colors: Mapping[str, str]) -> None:
        """Follow a theme change, like every other drawing here."""
        import pyqtgraph as pg

        self.colors = colors
        self.widget.setBackground(colors['plot_background'])
        self.curve.setPen(pg.mkPen(colors['filter_preview'], width=2))
        for line in self.lines.values():
            line.setPen(pg.mkPen(colors['plot_foreground'], width=1,
                                 style=Qt.PenStyle.DashLine))
            line.setHoverPen(pg.mkPen(colors['filter_preview'], width=2))
        for edge in ('left', 'bottom'):
            axis = self.plot.getAxis(edge)
            axis.setTextPen(colors['plot_foreground'])
            axis.setPen(colors['plot_foreground'])

    def show_response(self, frequencies: Any, magnitude: Any,
                      low: float | None, high: float | None) -> None:
        """Draw one response. `frequencies` is in Hz, `magnitude` dB;
        `low` and `high` are the pass-band edges, each shown as a
        dashed line where it exists and hidden where it does not.

        **The curve takes hertz and everything else takes log10**, and
        the split is pyqtgraph's rather than ours: a log-mode plot
        transforms its *data items* itself, but an `InfiniteLine` and
        `setXRange` are positioned in view coordinates, which on that
        axis already are the logarithms. Handing the curve log10 too
        drew the decades twice over — the axis ran 0.01 to 3 Hz for a
        record whose Nyquist is 1024 (Brandon, 2026-08-25, found by
        pinning the passband).
        """
        import numpy as np

        frequencies = np.asarray(frequencies, dtype=float)
        magnitude = np.asarray(magnitude, dtype=float)
        keep = frequencies > 0
        # clipped at the floor rather than dropped: a curve that ends
        # mid-air reads as missing data, where a filter that has run
        # out of dB has simply stopped
        self.curve.setData(frequencies[keep],
                           np.maximum(magnitude[keep], self.FLOOR))
        left = float(np.log10(frequencies[keep][0]))
        right = float(np.log10(frequencies[keep][-1]))
        self.plot.setXRange(left, right, padding=0)
        # a drag cannot leave the axis, cannot reach Nyquist — where
        # `filtered` refuses — and a band-pass's edges cannot cross:
        # each line stops one box-step short of the other. Clamped
        # here rather than caught later, which is this interface's
        # rule: refuse invalid state at entry rather than reporting
        # it afterwards
        floor = 10.0 ** left
        self.lines['low'].setVisible(low is not None)
        self.lines['high'].setVisible(high is not None)
        if low is not None:
            top = right - 1e-3 if high is None else float(
                np.log10(max(high - EDGE_GAP, floor)))
            self.lines['low'].setBounds([left, top])
            self.lines['low'].setValue(float(np.log10(max(low, 1e-9))))
        if high is not None:
            bottom = left if low is None else float(
                np.log10(max(low + EDGE_GAP, floor)))
            self.lines['high'].setBounds([bottom, right - 1e-3])
            self.lines['high'].setValue(float(np.log10(max(high, 1e-9))))


class FilterPanel(QWidget):
    """The parameter table. Edits arrive as a whole `Filtering`."""

    #: a parameter moved; here is the filtering that describes it now
    changed = Signal(object)

    #: the user asked for the filtered record this panel describes
    apply_asked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.sample_rate: float = 1.0
        #: the record the parameters describe, for suggesting again
        self._history = None
        #: set while the panel is writing to its own editors, so that
        #: restating a clamped value does not read as a fresh edit
        self._loading = False
        #: set while a corner line is under the mouse. A drag emits
        #: `changed` once, at release: every listener refilters every
        #: visible curve, and doing that per pixel of drag froze the
        #: whole panel — response plot included — on a 260-record
        #: history (Brandon, 2026-08-28). The cheap local restate
        #: still runs per move, so the box and the curve track the
        #: hand; only the expensive work waits.
        self._dragging = False
        #: the kind on show before the last Type switch, so a switch
        #: can carry the right number across — leaving a band-pass
        #: toward a low-pass takes the top edge, toward a high-pass
        #: the bottom one
        self._was = 'low-pass'

        self.title: QLabel = QLabel('Filter')
        grid = panel_grid(self, self.title)

        self.kind_box: QComboBox = QComboBox()
        self.kind_box.addItems(KIND_LABELS)
        self.kind_box.setToolTip(
            'Which band passes: everything below the corner, everything '
            'above it, or the stretch between two corners. Zero phase '
            'in every case')

        self.corner_box: DoubleSpinBox = DoubleSpinBox()
        self.corner_box.setToolTip(
            'The corner. Zero phase (applied forward and backward), so '
            'the peaks stay at their measured instants — and so the '
            'filter is −6 dB here, not −3. Drag the dashed line on the '
            'response below to set it by hand')

        self.from_box: DoubleSpinBox = DoubleSpinBox()
        self.from_box.setToolTip(
            'The pass band’s lower edge — the filter is −6 dB here. '
            'Drag the left dashed line on the response to set it by '
            'hand')

        self.to_box: DoubleSpinBox = DoubleSpinBox()
        self.to_box.setToolTip(
            'The pass band’s upper edge — the filter is −6 dB here. '
            'Drag the right dashed line on the response to set it by '
            'hand')

        for box in (self.corner_box, self.from_box, self.to_box):
            box.setDecimals(1)
            box.setSuffix(' Hz')

        self.order_box: SpinBox = SpinBox()
        self.order_box.setRange(1, 12)
        self.order_box.setToolTip(
            'The Butterworth design order. The forward-and-backward '
            'pass doubles the effective roll-off; a band-pass carries '
            'this many poles per edge')

        # every row is built and the kind decides which are on screen
        # — show only what applies (principle 3): a low-pass with a
        # grayed pair of band edges would be a question, not a control
        self._labels: dict[str, QLabel] = {}
        self._editors: dict[str, QWidget] = {}
        rows = (('kind', 'Type', self.kind_box),
                ('corner', 'Corner', self.corner_box),
                ('from', 'From', self.from_box),
                ('to', 'To', self.to_box),
                ('order', 'Order', self.order_box))
        for row, (key, label, editor) in enumerate(rows, start=1):
            name = QLabel(label)
            grid.addWidget(name, row, 0)
            grid.addWidget(editor, row, 1)
            self._labels[key] = name
            self._editors[key] = editor

        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setFrameShadow(QFrame.Shadow.Sunken)
        grid.addWidget(rule, len(rows) + 1, 0, 1, 2)

        # what follows from the settings, shown so the trade is
        # visible while it is being made rather than after the
        # filtered record comes out
        self.derived: dict[str, QLabel] = {}
        self.derived_names: dict[str, QLabel] = {}
        derived = (('nyquist', 'Nyquist'), ('octave', '@ 2× corner'),
                   ('half_power', '−3 dB at'))
        for key, (name, value) in add_derived(grid, derived, len(rows) + 2).items():
            self.derived[key] = value
            self.derived_names[key] = name

        # the shape of the filter being made, drawn where it is set
        # (Brandon, 2026-08-25). Small on purpose: it answers "what
        # does this do to my data" at a glance, and the stage beside
        # it answers "what did it do to *this* record".
        self.response: FilterResponsePlot = FilterResponsePlot()
        grid.addWidget(self.response, len(rows) + 2 + len(derived),
                       0, 1, 2)

        # No Suggest button beside these (Brandon, 2026-08-30): the
        # panel already *opens* on the suggestion when the record has
        # no filtering set, so the button could only ever undo edits
        # back to the starting default — a reset button in a smarter
        # name, and the truncate panel's rule holds here too: the
        # boxes and the drag lines reach every value.

        # the act itself, where its settings are set — this used to
        # be a note apologizing that the record was made elsewhere,
        # and a note explaining that the act lives somewhere else is
        # the interface admitting the act is in the wrong place
        # (Brandon, 2026-08-28).
        self.apply_button: QPushButton = QPushButton('Apply Filter')
        self.apply_button.setToolTip(
            'Make the filtered record — exactly the filter drawn '
            'above — beside this one')
        self.apply_button.clicked.connect(self.apply_asked.emit)
        grid.addWidget(self.apply_button, len(rows) + 3 + len(derived),
                       0, 1, 2)
        grid.setRowStretch(len(rows) + 4 + len(derived), 1)

        commit_on_enter(self.corner_box, self.from_box, self.to_box,
                        self.order_box)
        self.kind_box.currentIndexChanged.connect(self._kind_switched)
        for box in (self.corner_box, self.from_box, self.to_box,
                    self.order_box):
            box.valueChanged.connect(self._edited)
        # dragging a line and typing in its box are two views of one
        # number: the drag writes the box, and the box's own signal
        # carries it on from there, so there is one edit path
        self.response.edge_moved.connect(self._edge_dragged)
        self.response.edge_settled.connect(self._edge_settled)
        self._show_kind('low-pass')

    # ---- what the panel is describing -------------------------------------

    def show_history(self, history: TimeHistory,
                     filtering: Filtering) -> None:
        """Point the panel at a time history and the filtering set on it.

        The Nyquist clamp comes from the history rather than being
        guessed from the numbers already in the boxes.
        """
        self._history = history
        self.sample_rate = history.sample_rate
        self.set_filtering(filtering)

    def set_filtering(self, filtering: Filtering) -> None:
        """Restate the panel from a filtering, without re-emitting."""
        kind = filtering.kind
        self._loading = True
        try:
            # the limit first, for the same reason the averaging panel
            # sets its limits first: a spin box clamps to the range it
            # has at the moment it is written
            nyquist = self.sample_rate / 2.0
            top = max(nyquist * 0.999, 0.1)
            for box in (self.corner_box, self.from_box, self.to_box):
                box.setRange(0.1, top)
                box.setSingleStep(max(nyquist / 100.0, 0.1))
            self.kind_box.setCurrentIndex(KINDS.index(kind))
            if kind == 'low-pass':
                self.corner_box.setValue(filtering.high)
            elif kind == 'high-pass':
                self.corner_box.setValue(filtering.low)
            else:
                self.corner_box.setValue(filtering.high)
            # the hidden boxes still hold something coherent, so
            # switching Type proposes a filter rather than a blank:
            # the top edge mirrors the corner, the bottom sits a
            # decade under it unless the filtering says otherwise
            self.to_box.setValue(filtering.high if filtering.high
                                 is not None else top)
            self.from_box.setValue(
                filtering.low if filtering.low is not None
                else round(max(self.to_box.value() / 10.0, 0.1), 1))
            if kind == 'band-pass':
                self._constrain()
            self.order_box.setValue(filtering.order)
        finally:
            self._loading = False
        self._was = kind
        self._show_kind(kind)
        self._restate(self.filtering())

    def filtering(self) -> Filtering:
        """The filtering the editors currently describe."""
        kind = KINDS[self.kind_box.currentIndex()]
        order = self.order_box.value()
        if kind == 'low-pass':
            return Filtering(high=self.corner_box.value(), order=order)
        if kind == 'high-pass':
            return Filtering(low=self.corner_box.value(), order=order)
        return Filtering(low=self.from_box.value(),
                         high=self.to_box.value(), order=order)

    # ---- editing ----------------------------------------------------------

    def _show_kind(self, kind: str) -> None:
        """Put the rows that apply on screen and take the rest off."""
        single = kind != 'band-pass'
        for key, wanted in (('corner', single), ('from', not single),
                            ('to', not single)):
            self._labels[key].setVisible(wanted)
            self._editors[key].setVisible(wanted)
        # the octave row reads off a single corner; a band-pass has
        # two skirts and one number would be about neither
        self.derived_names['octave'].setVisible(single)
        self.derived['octave'].setVisible(single)
        if single:
            self.derived_names['octave'].setText(
                '@ 2× corner' if kind == 'low-pass' else '@ ½× corner')

    def _constrain(self) -> None:
        """Keep a band-pass's edges apart, by the boxes' own ranges.

        The ranges are the enforcement — a value typed past the other
        edge clamps to one step short of it — so an impossible
        band-pass cannot be stated, rather than being caught after.
        Only ever called with `_loading` held, because a clamp writes
        the box and the write must not read as a fresh edit.
        """
        top = max(self.sample_rate / 2.0 * 0.999, 0.2)
        self.from_box.setRange(
            0.1, max(self.to_box.value() - EDGE_GAP, 0.1))
        self.to_box.setRange(
            min(self.from_box.value() + EDGE_GAP, top), top)

    def _kind_switched(self, *_args) -> None:
        """The Type box moved: recut the panel and carry the numbers.

        The corner travels with its meaning where one exists — toward
        a band-pass it becomes the top edge; leaving one, the box
        takes the edge nearest in meaning (top for a low-pass, bottom
        for a high-pass). Between low- and high-pass the number stays
        put and only its meaning flips, which is what a reader
        switching kinds expects the box to do.
        """
        if self._loading:
            return
        kind = KINDS[self.kind_box.currentIndex()]
        self._loading = True
        try:
            nyquist = self.sample_rate / 2.0
            top = max(nyquist * 0.999, 0.1)
            for box in (self.corner_box, self.from_box, self.to_box):
                box.setRange(0.1, top)
            if kind == 'band-pass':
                self.to_box.setValue(self.corner_box.value())
                if not self.from_box.value() < self.to_box.value():
                    self.from_box.setValue(
                        round(max(self.to_box.value() / 10.0, 0.1), 1))
                self._constrain()
            elif self._was == 'band-pass':
                self.corner_box.setValue(
                    self.to_box.value() if kind == 'low-pass'
                    else self.from_box.value())
        finally:
            self._loading = False
        self._was = kind
        self._show_kind(kind)
        self._edited()

    def _edited(self, *_args):
        if self._loading:
            return
        if KINDS[self.kind_box.currentIndex()] == 'band-pass':
            # the other edge's range follows, before the value is read
            self._loading = True
            try:
                self._constrain()
            finally:
                self._loading = False
        settled = self.filtering()
        self._restate(settled)
        if self._dragging:
            return          # the release will say it, once
        self.changed.emit(settled)

    def _edge_dragged(self, which: str, hz: float) -> None:
        """A line was dragged: write it into its box.

        Rounded to the box's own precision first, so the number shown
        and the number filtered with are the same one — a box showing
        204.8 while the filter ran at 204.83 is two answers to one
        question.
        """
        self._dragging = True
        kind = KINDS[self.kind_box.currentIndex()]
        if kind == 'band-pass':
            box = self.from_box if which == 'low' else self.to_box
        else:
            box = self.corner_box
        box.setValue(round(float(hz), 1))

    def _edge_settled(self, _which: str, _hz: float) -> None:
        """The drag ended: now the edit is said, once.

        The box already holds the edge, rounded — the drag wrote it
        move by move — so the filtering emitted is the one on show.
        """
        if not self._dragging:
            return
        self._dragging = False
        self.changed.emit(self.filtering())

    # ---- what follows from the settings -----------------------------------

    def _restate(self, filtering: Filtering) -> None:
        """Redraw the response and read the numbers off it.

        Measured from the curve rather than from the textbook
        asymptote (Brandon, 2026-08-25): 6 dB per octave per pole,
        doubled for the zero-phase pass, predicted −48 dB an octave
        above an order-4 corner where the real filter is at −55.9 —
        the digital response steepens as it approaches Nyquist, and a
        panel that states a number the filter does not deliver is
        worse than one that states none.
        """
        import numpy as np

        from ..core.filters import response

        self.derived['nyquist'].setText(f'{self.sample_rate / 2.0:.4g} Hz')
        frequencies, magnitude = response(filtering, self.sample_rate)
        self.response.show_response(frequencies, magnitude,
                                    filtering.low, filtering.high)
        if filtering.kind == 'low-pass':
            octave = float(np.interp(filtering.high * 2.0,
                                     frequencies, magnitude))
            self.derived['octave'].setText(f'{octave:.0f} dB')
        elif filtering.kind == 'high-pass':
            octave = float(np.interp(filtering.low / 2.0,
                                     frequencies, magnitude))
            self.derived['octave'].setText(f'{octave:.0f} dB')
        # where the *effective* filter is actually half power. Not the
        # corners: a Butterworth is −3 dB at its corner by definition,
        # so filtering forward and backward is −6 dB there and the
        # half-power points sit inside the pass band. Read as
        # crossings of the drawn curve — one for a low- or high-pass,
        # two for a band-pass
        above = magnitude > -3.0
        crossings = []
        for i in np.flatnonzero(above[1:] != above[:-1]):
            f0, f1 = frequencies[i], frequencies[i + 1]
            m0, m1 = magnitude[i], magnitude[i + 1]
            crossings.append(f0 + (-3.0 - m0) * (f1 - f0) / (m1 - m0))
        self.derived['half_power'].setText(
            ' – '.join(f'{point:.4g}' for point in crossings) + ' Hz'
            if crossings else '—')
