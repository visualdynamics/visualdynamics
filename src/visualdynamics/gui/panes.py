"""Self-contained view panes.

A pane holds a view, the bar of controls above it, and the state those
controls carry, and nothing else. It does not read the project, the tree
or the selection. When a choice on its bar changes, it emits a signal,
and whoever placed the pane decides what the change means. As a result
the same pane can be used in the main window, in a dock, or in a window
of its own opened from a script.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any, ClassVar, NamedTuple

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QImage,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..plot import background_brush
from ..theme import theme as resolve_theme
from ..viz.geometry import annotate_scene
from .amplitude_panel import AmplitudePanel
from .author_panel import AuthorPanel
from .averaging_panel import AveragingPanel
from .chance_panel import ChancePanel
from .confidence_panel import ConfidencePanel
from .editors import SpinBox
from .filter_panel import FilterPanel
from .icons import child_icon, control_icon
from .instrumentation_panel import InstrumentationPanel
from .mate_panel import MatePanel
from .matrix_check_panel import ConditioningPanel, RealizabilityPanel
from .merge_panel import MergePanel
from .mesh_panel import MeshPanel
from .mesh_solids_panel import MeshSolidsPanel
from .modes_panel import ModesPanel
from .octave_panel import OctavePanel
from .pattern_panel import PatternPanel
from .profile_panel import ProfilePanel
from .pulse_panel import PulsePanel
from .replication_fit_panel import ReplicationFitPanel
from .resample_panel import ResamplePanel
from .rms_check_panel import RmsCheckPanel
from .segments_panel import SegmentsPanel
from .settings_panel import PANEL_WIDTH
from .shock_panel import ShockPanel
from .signature_panel import SignaturePanel
from .sine_panel import SinePanel
from .srs_tolerance_panel import SrsTolerancePanel
from .statistics_panel import StatisticsPanel
from .stroke_panel import StrokePanel
from .sweep_rate_panel import SweepRatePanel
from .tone_panel import TonePanel
from .toolbars import fence, shows_anything, tidy
from .transmissibility_panel import TransmissibilityPanel
from .truncate_panel import TruncatePanel
from .wavelet_panel import WaveletPanel


def copy_act(pane: Any, what: str) -> tuple:
    """The Copy act for a pane's bar, as an `offer_acts` entry.

    Pressed, it copies what `pane` shows to the clipboard as an image;
    `what` names that in the tooltip ("plot", "3-D view"). It is an act
    rather than a control that is always there, so it is on the bar only
    while something is drawn. Both panes use the same icon and tooltip
    form, so it reads as the same act on each."""
    # Brandon, 2026-09-24: every plot, table and figure should copy like
    # a chat's code blocks do.
    return ('copy', 'Copy', 'copy', pane.copy_view,
            f'Copy the {what} to the clipboard as an image')


def image_of(pixels: Any) -> QImage:
    """A rendered `(rows, columns, 3 or 4)` array of bytes as a QImage
    with its own copy of the pixels.

    The array a screenshot returns is freed with it, and a QImage that
    only pointed at that memory would read garbage."""
    import numpy as np

    pixels = np.ascontiguousarray(pixels)
    rows, columns, depth = pixels.shape
    layout = (QImage.Format.Format_RGBA8888 if depth == 4
              else QImage.Format.Format_RGB888)
    return QImage(pixels.data, columns, rows, depth * columns, layout).copy()


def offer_acts(toolbar: Any, state: dict, acts: Sequence[tuple]) -> None:
    """Put the acts a selection can take on a bar, in their own group
    at its end, after a separator.

    Each act is an icon, like every other button on a bar, with the
    label and what it does in the tooltip. Every act is on the bar;
    none is behind a menu.

    `acts` is a list of `(verb, label, icon name, callback, tooltip)`,
    in the order the bar shows them. `state` is a dict the caller keeps
    for this bar between calls. The group is rebuilt on every call,
    because a toolbar cannot reorder its actions and each verb should
    be in the same place whatever else the selection allows.
    """
    # Icons only (Brandon, 2026-09-12: "none should [carry text] and the
    # user should rely on tool tips to learn icons they don't
    # recognize"); this reverses the labeled buttons of 2026-09-04.
    from PySide6.QtWidgets import QToolButton

    actions = state.setdefault('actions', {})
    for action in actions.values():
        toolbar.removeAction(action)
    if 'fence' not in state:
        state['fence'] = toolbar.addSeparator()
    state['fence'].setVisible(bool(acts))
    state['callbacks'] = {verb: callback for verb, _l, _i, callback, _t
                          in acts}
    for verb, label, icon, _callback, tooltip in acts:
        action = actions.get(verb)
        if action is None:
            action = QAction(control_icon(icon), label, toolbar)
            # through the state, so a later offer's callback is the
            # one a press runs
            action.triggered.connect(
                lambda _checked=False, v=verb: state['callbacks'][v]())
            actions[verb] = action
        action.setText(label)
        # the label leads the tooltip: the icon is what shows, and the
        # words are where a person learns what it does
        action.setToolTip(f'{label} — {tooltip}' if tooltip else label)
        action.setVisible(True)
        toolbar.addAction(action)
        button = toolbar.widgetForAction(action)
        if isinstance(button, QToolButton):
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)


# how a complex ordinate reads, in the order the box offers it
COMPONENTS: tuple[tuple[str, str], ...] = (
    ('Magnitude', 'magnitude'),
    ('Real', 'real'),
    ('Imaginary', 'imag'),
    ('Phase', 'phase'),
)


class Reading(NamedTuple):
    """One of the data bar's readings of a time history, by the names
    its parts have on `DataPane`."""

    #: the pane holds `<key>_action` and `<key>_wanted`
    key: str
    #: the attribute holding its settings panel, or None for none
    panel: str | None
    #: which `show_controls` offer puts it on the bar: 'averaging',
    #: 'sine', 'shocks', 'fit', or 'record' for either kind of record
    offered_by: str
    #: drawn in place of the trace with no 3-D form, so the 2-D/3-D
    #: toggle is not offered while it is up
    flat: bool = False
    #: its panel's settings are kept on the pane as `<key>_settings`
    sticky: bool = False


class DataPane(QWidget):
    """The data view: a plot, or its 3-D form, under a bar of choices
    for how to read the data.

    The bar has the readings of the data shown (averaging, filter,
    truncate, kurtosis, shocks, scalogram, replication fit, octave
    bands, the CMIF and others), the choices of a comparison against a
    specification, the record filters and the acts the selection can
    take. It shows only
    the controls the data can use. A control that does not apply is
    hidden, not grayed out.

    The pane keeps each choice and emits a signal when it changes. What
    the choice means is decided outside: filtering to the diagonal
    changes the tree's selection, and Edit Fit opens a fitting session,
    and neither is the plot's job.
    """

    #: a reading choice moved; draw the same data again
    reread = Signal()
    #: what was copied to the clipboard, for the status line
    copied = Signal(str)
    #: the diagonal filter went down (True) or came back up (False)
    drive_points_toggled = Signal(bool)
    #: an FRF beside a shape set should read this way: 'fit' | 'overlay'
    pair_mode_chosen = Signal(str)
    #: the averaging view was asked for (True) or put away (False)
    averaging_toggled = Signal(bool)

    #: the filter view was asked for (True) or put away (False)
    filter_toggled = Signal(bool)
    #: the sine view was asked for (True) or put away (False)
    sine_toggled = Signal(bool)
    #: the truncate view was asked for (True) or put away (False)
    truncate_toggled = Signal(bool)
    #: the octave-band view was asked for (True) or put away (False)
    octave_toggled = Signal(bool)
    #: the tone readout was asked for (True) or put away (False)
    tone_toggled = Signal(bool)
    #: the transmissibility reading was asked for (True) or put away
    transmissibility_toggled = Signal(bool)
    #: the confidence band was asked for (True) or put away (False)
    confidence_toggled = Signal(bool)
    #: the rms check was asked for (True) or put away (False)
    rms_check_toggled = Signal(bool)
    #: the conditioning reading of an FRF or CSD matrix was asked for
    #: (True) or put away
    conditioning_toggled = Signal(bool)
    #: the realizability reading of a CSD was asked for (True) or put
    #: away
    realizability_toggled = Signal(bool)
    #: a sine specification's sweep-rate reading was asked for (True)
    #: or put away (False)
    sweep_rate_toggled = Signal(bool)
    #: a sine specification's profile reading was asked for (True) or
    #: put away (False)
    sine_profile_toggled = Signal(bool)
    #: the fit's residual CMIF was toggled
    residual_toggled = Signal(bool)
    shocks_toggled = Signal(bool)
    #: which reading of a comparison the top plot shows
    comparison_chosen = Signal(str)
    #: a different specification-and-response pair was picked
    pair_chosen = Signal()
    #: the comparison scaling was edited; the text as typed
    scaling_edited = Signal(str)
    #: which reading of a transient replication the plot shows
    replication_chosen = Signal(str)
    #: a different repeat of the transient was picked
    event_chosen = Signal()
    #: which reading of a shock-spectrum comparison the plot shows
    srs_chosen = Signal(str)
    #: which reading of two selected densities the plot shows
    spectra_view_chosen = Signal(str)

    def __init__(self, theme_name: str, offscreen: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        import pyqtgraph as pg

        self.theme_name: str = theme_name
        #: offscreen replaces the embedded 3-D data view with an
        #: off-screen plotter, exactly as ScenePane does for the scene
        self.offscreen: bool = offscreen
        #: the 3-D data surface, built on first use — None until the
        #: waterfall is first asked for
        self.waterfall_plotter: Any = None
        self._waterfall_page: QWidget | None = None
        #: None lets the data choose how it reads; a click outranks it
        self.plot_mode: str | None = None
        #: set while the pair, event or quantity box is being refilled,
        #: so restating the same list does not read as picking from it
        self._loading_pairs = False
        self._loading_events = False
        self._loading_quantities = False
        self._loading_page = False
        #: and the same for the scaling field, which is restated on
        #: every drawing of a comparison
        self._loading_scaling = False
        #: the averaging button is asked for once and stays asked for.
        #: The bar is reset before every drawing, so the button's own
        #: checked state is not somewhere the choice can be kept.
        self.averaging_wanted: bool = False
        self.shocks_wanted: bool = False
        self.filter_wanted: bool = False
        self.sine_wanted: bool = False
        self.truncate_wanted: bool = False
        self.octave_wanted: bool = False
        self.tone_wanted: bool = False
        self.transmissibility_wanted: bool = False
        self.confidence_wanted: bool = False
        self.rms_check_wanted: bool = False
        self.conditioning_wanted: bool = False
        self.realizability_wanted: bool = False
        self.sweep_rate_wanted: bool = False
        self.sine_profile_wanted: bool = False
        #: the kurtosis reading, sticky like every other view choice
        self.kurtosis_wanted: bool = False
        #: and the time statistics, the same way
        self.statistics_wanted: bool = False
        #: what they are read with, kept across redraws so a range typed
        #: for one channel survives clicking away and back
        self.statistics_settings: dict[str, Any] | None = None
        #: and the amplitude spectrum, the same way, with the window it
        #: is read under kept across redraws and records
        self.amplitude_wanted: bool = False
        self.amplitude_settings: dict[str, Any] | None = None
        #: and the scalogram, the same way
        self.wavelet_wanted: bool = False
        #: what it is computed with, kept across redraws so a tuned
        #: range survives clicking away and back
        self.wavelet_settings: dict[str, Any] | None = None
        #: the classical pulse check, the same way; its pulse and
        #: tolerances kept across redraws, None until one is set
        self.pulse_wanted: bool = False
        self.pulse_settings: dict[str, Any] | None = None
        #: the replication fit, the same way, and the ratings and
        #: cutoffs it was read with (SI), kept across redraws and
        #: objects: a shaker's ratings do not change with the record
        self.fit_wanted: bool = False
        self.fit_settings: dict[str, Any] | None = None
        #: and the stroke reading, the same way: a record's pulse
        #: integrated raw against the shaker's shock ratings
        self.stroke_wanted: bool = False
        #: the ratings it is read against, in SI, kept across redraws
        #: so they survive clicking away and back (a shaker's ratings
        #: do not change with the record)
        self.stroke_settings: dict[str, Any] | None = None
        #: and the resampling, the same way: a record taken to a lower
        #: rate, its spectrum read for what the filter does and where
        #: content above the new Nyquist frequency folds
        self.resample_wanted: bool = False
        #: the factor, filter order, frame and folded frequencies, kept
        #: across redraws so they survive clicking away and back
        self.resample_settings: dict[str, Any] | None = None
        #: where its band has been dragged to, if it has
        self.kurtosis_bounds: tuple[float, float] | None = None
        #: 'curves', 'error', 'lines' or 'chance' — sticky, like the
        #: other view choices, so it survives a redraw
        self.comparison_view: str = 'curves'
        #: where the bar charts' thresholds have been dragged to
        self.error_bounds: tuple[float, float] | None = None
        self.lines_bound: float | None = None
        #: 'overlay', 'waveform' or 'segments' — the transient's own
        #: readings, kept apart from the random ones above because they
        #: are different questions and a single sticky choice shared
        #: between them would carry a random answer into a transient
        self.replication_view: str = 'overlay'
        #: 'curves', 'error' or 'points' — the same for a pair of
        #: shock spectra ('points' is theirs alone, not the sine's)
        self.srs_view: str = 'curves'
        #: 'overlay', 'ratio' or 'snr' — how two selected densities
        #: read together; sticky like every view choice
        self.spectra_view: str = 'overlay'
        #: where the RMS signal-to-noise bars' floor has been dragged
        self.snr_bound: float | None = None
        self.srs_bounds: tuple[float, float] | None = None
        #: where the replication bars' thresholds have been dragged
        self.waveform_bound: float | None = None

        self.graphics: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget()
        # pyqtgraph's GraphicsView ignores every drag it is offered — its own
        # comment says the class "likes to consume drag events" — but the
        # widget still advertises that it takes drops. So Qt routes a drag
        # across the plot to it, gets nothing back to record as an entry, and
        # then complains on the way out: "drag leave received before drag
        # enter". Dropping files is the tree's job; the plot saying it takes
        # them was never true.
        self.graphics.setAcceptDrops(False)
        self.graphics.viewport().setAcceptDrops(False)

        # the points rule's band and rule, beside a shock spectrum
        # judged against them; the Points reading shows it. Made before
        # the bar, whose SRS readings put it away
        self.srs_tolerance_panel: SrsTolerancePanel = SrsTolerancePanel()
        self.srs_tolerance_panel.hide()
        # the lines outside a random comparison's bands beside what
        # chance puts there, and how many lines are judged together;
        # the Chance reading shows it. Made before the bar, whose
        # comparison readings put it away
        self.chance_panel: ChancePanel = ChancePanel()
        self.chance_panel.hide()
        # the segment marks and their tolerances, beside a run judged
        # segment by segment; the Segments reading shows it. Made before
        # the bar, whose replication readings put it away
        self.segments_panel: SegmentsPanel = SegmentsPanel()
        self.segments_panel.hide()
        # the signature comparison's modes and limits, beside two
        # signatures read against each other; the Signature reading
        # shows it. Made before the bar, whose readings put it away
        self.signature_panel: SignaturePanel = SignaturePanel()
        self.signature_panel.hide()
        self.toolbar: QToolBar = self._build_toolbar()
        # The comparison, replication and SRS readings are built
        # visible and were only ever put away by `reset_controls` —
        # harmless while the bar computed its own visibility from the
        # arguments it was handed, and wrong the moment the bar is
        # derived from what is actually on it. A control that is not a
        # live option is absent, and that has to be true from the
        # first frame, not from the first reset.
        self.show_comparison_views(False)
        self.show_spectra_views(False)
        self.show_replication_views(False)
        self.show_srs_views(False)
        # the averaging parameters sit beside the plot rather than under
        # it: they describe a span along the time axis, and reading them
        # against the shading means seeing both at once
        self.averaging_panel: AveragingPanel = AveragingPanel()
        self.averaging_panel.hide()
        self.shock_panel: ShockPanel = ShockPanel()
        self.shock_panel.hide()
        self.filter_panel: FilterPanel = FilterPanel()
        self.filter_panel.hide()
        self.sine_panel: SinePanel = SinePanel()
        self.sine_panel.hide()
        self.truncate_panel: TruncatePanel = TruncatePanel()
        self.truncate_panel.hide()
        self.octave_panel: OctavePanel = OctavePanel()
        self.octave_panel.hide()
        self.tone_panel: TonePanel = TonePanel()
        self.tone_panel.hide()
        self.transmissibility_panel: TransmissibilityPanel = \
            TransmissibilityPanel()
        self.transmissibility_panel.hide()
        # what an averaged autospectrum is worth, beside the band
        # shaded on it; read only, nothing to apply
        self.confidence_panel: ConfidencePanel = ConfidencePanel()
        self.confidence_panel.hide()
        # the rms from a spectrum beside the rms of its time history,
        # read beside the spectrum; nothing to apply
        self.rms_check_panel: RmsCheckPanel = RmsCheckPanel()
        self.rms_check_panel.hide()
        # the two checks on a multi-input test's matrices, each read at
        # a cursor on its own figure
        self.conditioning_panel: ConditioningPanel = ConditioningPanel()
        self.conditioning_panel.hide()
        self.realizability_panel: RealizabilityPanel = RealizabilityPanel()
        self.realizability_panel.hide()
        # the modes a sine sweep is checked against, beside the sweep
        # drawn through their bands; the Sweep Rate reading shows it
        self.sweep_rate_panel: SweepRatePanel = SweepRatePanel()
        self.sweep_rate_panel.hide()
        # what a sine specification's profile is read with, beside the
        # profile drawn three ways; the Sine Profile reading shows it
        self.sine_profile_panel: ProfilePanel = ProfilePanel()
        self.sine_profile_panel.hide()
        self.wavelet_panel: WaveletPanel = WaveletPanel()
        self.wavelet_panel.hide()
        # the time statistics' settings and their table, beside the
        # three rows that draw them; read only, nothing to apply
        self.statistics_panel: StatisticsPanel = StatisticsPanel()
        self.statistics_panel.hide()
        # the amplitude spectrum's window and its reading, beside the
        # spectrum; read only, nothing to apply
        self.amplitude_panel: AmplitudePanel = AmplitudePanel()
        self.amplitude_panel.hide()
        # the pulse a record is judged against, beside the judgment
        self.pulse_panel: PulsePanel = PulsePanel()
        self.pulse_panel.hide()
        # the shaker's ratings beside a record checked against them
        self.replication_fit_panel: ReplicationFitPanel = \
            ReplicationFitPanel()
        self.replication_fit_panel.hide()
        # the shaker's shock ratings beside the pulse read against them
        self.stroke_panel: StrokePanel = StrokePanel()
        self.stroke_panel.hide()
        # the decimation and its anti-alias filter beside the spectrum
        # that shows what they do; read only, nothing to apply
        self.resample_panel: ResamplePanel = ResamplePanel()
        self.resample_panel.hide()
        for reading in self._READINGS:
            if reading.sticky:
                getattr(self, reading.panel).changed.connect(
                    partial(self._settings_edited, reading.key))
        # the specification being written at a shape set's modal
        # coordinates, beside the plot that previews it; the toggle
        # is on the table bar, where a lone shape set's readings are
        self.author_panel: AuthorPanel = AuthorPanel()
        self.author_panel.hide()
        plot_row = QHBoxLayout()
        plot_row.setContentsMargins(0, 0, 0, 0)
        plot_row.setSpacing(0)
        # the specification sheet is the one side panel a user works
        # *in* — a grid of levels, a grid of pairs — so its width is
        # theirs to set (Brandon, 2026-09-06): a splitter between the
        # plot and the sheet, where the settings panels keep their
        # fixed width. The 3-D page joins the splitter beside the
        # graphics on first use, so the sheet sits beside either view
        self.author_split: QSplitter = QSplitter(Qt.Orientation.Horizontal)
        self.author_split.setChildrenCollapsible(False)
        self.author_split.setHandleWidth(6)
        self.author_split.addWidget(self.graphics)
        self.author_split.addWidget(self.author_panel)
        self.author_split.setStretchFactor(0, 1)
        self.author_split.setStretchFactor(1, 0)
        plot_row.addWidget(self.author_split, 1)
        plot_row.addWidget(self.averaging_panel)
        plot_row.addWidget(self.shock_panel)
        plot_row.addWidget(self.filter_panel)
        plot_row.addWidget(self.sine_panel)
        plot_row.addWidget(self.truncate_panel)
        plot_row.addWidget(self.octave_panel)
        plot_row.addWidget(self.srs_tolerance_panel)
        plot_row.addWidget(self.chance_panel)
        plot_row.addWidget(self.tone_panel)
        plot_row.addWidget(self.transmissibility_panel)
        plot_row.addWidget(self.confidence_panel)
        plot_row.addWidget(self.rms_check_panel)
        plot_row.addWidget(self.conditioning_panel)
        plot_row.addWidget(self.realizability_panel)
        plot_row.addWidget(self.segments_panel)
        plot_row.addWidget(self.sweep_rate_panel)
        plot_row.addWidget(self.sine_profile_panel)
        plot_row.addWidget(self.signature_panel)
        plot_row.addWidget(self.wavelet_panel)
        plot_row.addWidget(self.statistics_panel)
        plot_row.addWidget(self.amplitude_panel)
        plot_row.addWidget(self.pulse_panel)
        plot_row.addWidget(self.replication_fit_panel)
        plot_row.addWidget(self.stroke_panel)
        plot_row.addWidget(self.resample_panel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addLayout(plot_row, 1)

    # ---- the bar ----------------------------------------------------------

    def _build_toolbar(self) -> QToolBar:
        toolbar = QToolBar('Plot')
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))
        # First on the bar, always (Brandon, 2026-08-28): the 2D/3D
        # toggle is how *any* reading is drawn, not one of the readings,
        # and a setting that outranks every choice beside it belongs at
        # a fixed place the hand can find — leftmost, whatever the
        # selection has hidden or shown. Checked state is the choice and
        # it sticks; hiding the action when a selection cannot use it
        # leaves the choice armed for the next one that can. Checked
        # from the start: the 3-D reading is the default (Brandon's
        # call), and unchecking is how the flat plot is asked for.
        self.waterfall_action: QAction = QAction(
            control_icon('waterfall'), '3D', self)
        self.waterfall_action.setCheckable(True)
        self.waterfall_action.setToolTip(
            'Spread the channels along a depth axis, colored by level')
        self.waterfall_action.setChecked(True)
        self.waterfall_action.triggered.connect(
            lambda _checked: self.reread.emit())
        toolbar.addAction(self.waterfall_action)
        # the frequency axis, decades or hertz, for every plot at once:
        # some readers of a specification want one and some the other
        # (Brandon, 2026-09-05). A view control, so it sits with 3D;
        # the choice is the process's (`core.data.frequency_axis`),
        # read by the stage and the report too, and the window sets it
        self.log_frequency_action: QAction = QAction(
            control_icon('log_axis'), 'Log f', self)
        self.log_frequency_action.setCheckable(True)
        self.log_frequency_action.setToolTip(
            'Frequency in decades rather than hertz, on every plot, '
            'the 3-D stage and report figures')
        toolbar.addAction(self.log_frequency_action)
        self.log_frequency_action.setVisible(False)
        self.plot_mode_group: QActionGroup = QActionGroup(self)
        self.plot_mode_group.setExclusive(True)
        self.curves_action: QAction = QAction(control_icon('curves'), 'Curves', self)
        self.curves_action.setToolTip(
            'One line per channel, coherence against frequency')
        self.map_action: QAction = QAction(control_icon('map'), 'Map', self)
        self.map_action.setToolTip(
            'Every channel at once: frequency across, channel down, '
            'coherence as color')
        for action, mode in ((self.curves_action, 'curves'),
                             (self.map_action, 'map')):
            action.setCheckable(True)
            self.plot_mode_group.addAction(action)
            toolbar.addAction(action)
            action.triggered.connect(
                lambda _checked=False, mode=mode: self._choose_plot_mode(mode))
        # one floor, one quantity: which of a mixed object's quantities
        # stands on the waterfall's stage. The 2-D plot stacks an axis
        # per quantity and needs no such choice, so the box shows only
        # in 3-D and only when the object actually mixes
        self.quantity_box: QComboBox = QComboBox()
        self.quantity_box.setToolTip(
            'Which quantity the waterfall shows — one vertical axis '
            'holds one; the flat plot stacks an axis per quantity')
        self.quantity_box.currentIndexChanged.connect(
            lambda _index: None if self._loading_quantities
            else self.reread.emit())
        self.quantity_action: QAction = toolbar.addWidget(self.quantity_box)
        # A scene holds so many records and no more (SCENE_BUDGET), and
        # what does not fit goes on the next page rather than being
        # thinned — thinning is what broke the treads. `1 / 3` with its
        # own arrows, the same control the scene bar's mode and scale
        # wear (Brandon, 2026-08-28): the arrows belong to the number,
        # and two toolbar buttons either side of a box only look as
        # though they do.
        self.page_box: SpinBox = SpinBox()
        self.page_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_box.setMinimum(1)
        self.page_box.setMaximum(1)
        self.page_box.setToolTip(
            'Which page of channels is on the stage. A scene holds as '
            'many as it can draw exactly; the rest are a page away')
        self.page_box.valueChanged.connect(
            lambda _v: None if self._loading_page else self.reread.emit())
        self.page_action: QAction = toolbar.addWidget(self.page_box)
        # complex data reads four ways: |H|, real, imaginary, phase
        self.component_box: QComboBox = QComboBox()
        self.component_box.setToolTip(
            'Which part of the complex data to plot')
        for label, component in COMPONENTS:
            self.component_box.addItem(label, component)
        self.component_box.currentIndexChanged.connect(
            lambda _index: self.reread.emit())
        self.component_action: QAction = toolbar.addWidget(self.component_box)
        # a specification with its response is one comparison, and six of
        # them on one axis is a thicket with no comparison visible in it.
        # So one at a time, and this is how the others are reached.
        self.pair_box: QComboBox = QComboBox()
        self.pair_box.setToolTip(
            'Which channel to compare against its specification.\n'
            'Up and Down step through them while the plot has focus; '
            'Alt+Up and Alt+Down work anywhere.')
        self.pair_box.currentIndexChanged.connect(
            lambda _index: None if self._loading_pairs
            else self.pair_chosen.emit())
        self.pair_action: QAction = toolbar.addWidget(self.pair_box)
        # four readings of one comparison: the spectra themselves, the
        # level each channel came out at, how much of each channel's
        # band fell outside, and how many of one channel's lines fell
        # outside beside how many chance alone puts there. One at a
        # time, because they answer different questions and stacking
        # them answers none.
        self.comparison_group: QActionGroup = QActionGroup(self)
        self.comparison_actions: dict[str, QAction] = {}
        for key, icon, text, tip in (
                ('curves', 'curves', 'Spectra',
                 'The control spectra against their specification'),
                ('error', 'bars', 'RMS Error',
                 'How far each channel sits from the level asked for'),
                ('lines', 'percent_bars', 'Lines Out',
                 'How much of each channel fell outside its abort limits'),
                ('chance', 'chance', 'Chance',
                 ("How many of the channel's lines fell outside each band, "
                  'beside how many chance alone puts there at the '
                  "comparison's degrees of freedom"))):
            action = QAction(control_icon(icon), text, self)
            action.setCheckable(True)
            action.setToolTip(tip)
            action.triggered.connect(
                lambda _checked=False, which=key: self._choose_comparison(which))
            self.comparison_group.addAction(action)
            toolbar.addAction(action)
            self.comparison_actions[key] = action
        self.comparison_actions['curves'].setChecked(True)
        # a specification on its own: its spectra, or the RMS level each
        # channel asks for as a bar apiece with the table beneath —
        # the comparison's RMS reading without the coloring (Brandon,
        # 2026-09-06). A toggle rather than a third group: it is one
        # other reading, and unchecking is the way back to the spectra
        self.rms_action: QAction = QAction(control_icon('bars'), 'RMS', self)
        self.rms_action.setCheckable(True)
        self.rms_action.setToolTip(
            'The RMS level each channel asks for, a bar apiece, with '
            'the table of levels beneath')
        self.rms_action.triggered.connect(
            lambda _checked: self.reread.emit())
        toolbar.addAction(self.rms_action)
        self.rms_action.setVisible(False)
        # two signatures of one article selected together, before and
        # after a test: overlaid, or compared mode by mode against
        # frequency and amplitude limits. A toggle like the RMS one: it
        # is one other reading of the pair, and unchecking is the way
        # back to the overlay
        self.signature_action: QAction = QAction(
            control_icon('overlay'), 'Signature', self)
        self.signature_action.setCheckable(True)
        self.signature_action.setToolTip(
            'The two signatures compared mode by mode: each mode\'s '
            'frequency and amplitude shift against its limits, with a '
            'window drawn on each mode and the table beside')
        self.signature_action.triggered.connect(
            lambda _checked: self.reread.emit())
        toolbar.addAction(self.signature_action)
        self.signature_action.setVisible(False)
        # two densities selected together: drawn over each other, or
        # read as signal over noise — line by line in dB (Brandon,
        # 2026-08-25), or one RMS number per channel as bars
        # (2026-10-03). One at a time, like the comparison's readings.
        self.spectra_group: QActionGroup = QActionGroup(self)
        self.spectra_actions: dict[str, QAction] = {}
        for key, icon, text, tip in (
                ('overlay', 'overlay', 'Overlaid',
                 'The two spectra over each other'),
                ('ratio', 'ratio', 'Signal to noise (dB)',
                 ('The signal-to-noise of the louder spectrum over the '
                  'quieter, line by line, in decibels: the louder less '
                  'the quieter, over the quieter')),
                ('snr', 'bars', 'RMS signal to noise',
                 ('The signal-to-noise of each channel over the whole '
                  'band, a bar apiece, from the RMS of the two '
                  'spectra'))):
            action = QAction(control_icon(icon), text, self)
            action.setCheckable(True)
            action.setToolTip(tip)
            action.triggered.connect(
                lambda _checked=False, which=key:
                self._choose_spectra_view(which))
            self.spectra_group.addAction(action)
            toolbar.addAction(action)
            self.spectra_actions[key] = action
        self.spectra_actions['overlay'].setChecked(True)
        # a run captured below its requirement is compared scaled up to
        # it — standard practice, and the number belongs on the bar
        # because every reading on this bar is of the scaled data
        self.scaling_edit: QLineEdit = QLineEdit()
        self.scaling_edit.setFixedWidth(64)
        self.scaling_edit.setToolTip(
            'Decibels added to the measured spectra when compared '
            'against the specification.\nDetected from the data; type '
            'a number to hold it (0 holds it unscaled), clear the '
            'field to detect again.\nThe data itself is never changed.')
        self.scaling_edit.editingFinished.connect(
            lambda: None if self._loading_scaling
            else self.scaling_edited.emit(self.scaling_edit.text()))
        scaling = QWidget()
        row = QHBoxLayout(scaling)
        row.setContentsMargins(6, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(QLabel('Scaling'))
        row.addWidget(self.scaling_edit)
        # the toolbar hands the widget whatever width is spare, and the
        # layout would spend it between the label and the field —
        # putting the name at one end of the bar and the number at the
        # other. The stretch takes the spare width instead.
        row.addStretch(1)
        self.scaling_action: QAction = toolbar.addWidget(scaling)
        self.scaling_action.setVisible(False)
        # a transient is played over and over, so its axis is the repeat
        # rather than the channel: one event on screen, and this is how
        # the others are reached. Left and Right step through them —
        # not Up and Down, which the channel box above already owns.
        self.event_box: QComboBox = QComboBox()
        self.event_box.setToolTip(
            'Which repeat of the transient to compare against the '
            'specification.\nLeft and Right step through them while the '
            'plot has focus.')
        self.event_box.currentIndexChanged.connect(
            lambda _index: None if self._loading_events
            else self.event_chosen.emit())
        self.event_action: QAction = toolbar.addWidget(self.event_box)
        self.event_action.setVisible(False)
        # two readings of the whole run, and only two. The waveform
        # error is the one thing the time data alone can answer; a level
        # error wants two PSDs and an SRS deviation wants two spectra,
        # and those are reached by computing them and selecting the
        # pair — where the specification's own PSD is a Specification
        # and its SRS a ShockSpecification, so the app still knows which
        # is which. Offering them here as well would be two ways to the
        # same number, computed differently, agreeing only by luck. The
        # third reading is not another number for the whole run: it
        # judges each marked segment by the rule for its kind, which is
        # what a whole-record figure hides
        self.replication_group: QActionGroup = QActionGroup(self)
        self.replication_actions: dict[str, QAction] = {}
        for key, icon, text, tip in (
                ('overlay', 'overlay', 'Waveforms',
                 'The measured event against the waveform asked for'),
                ('waveform', 'bars', 'Waveform Error',
                 ('How far each channel is from the target waveform, '
                  'as a share of the target')),
                ('segments', 'segments', 'Segments',
                 ('Each marked segment judged by the rule for its kind: '
                  'a stationary stretch by its ASD and variance, a long '
                  'transient by its envelope, a shock by its SRS and '
                  'peaks')),
                ):
            action = QAction(control_icon(icon), text, self)
            action.setCheckable(True)
            action.setToolTip(tip)
            action.triggered.connect(
                lambda _checked=False, which=key:
                self._choose_replication(which))
            self.replication_group.addAction(action)
            toolbar.addAction(action)
            self.replication_actions[key] = action
        self.replication_actions['overlay'].setChecked(True)
        # the same two readings for a pair of shock spectra: the curves,
        # and how far each channel sits from what it had to meet; and a
        # third that is the shock's own, the points rule
        self.srs_group: QActionGroup = QActionGroup(self)
        self.srs_actions: dict[str, QAction] = {}
        for key, icon in (('curves', 'curves'), ('error', 'bars'),
                          ('points', 'percent_bars')):
            text, tip = self.SRS_LABELS['srs'][key]
            action = QAction(control_icon(icon), text, self)
            action.setCheckable(True)
            action.setToolTip(tip)
            action.triggered.connect(
                lambda _checked=False, which=key: self._choose_srs(which))
            self.srs_group.addAction(action)
            toolbar.addAction(action)
            self.srs_actions[key] = action
        self.srs_actions['curves'].setChecked(True)
        for key, step in ((Qt.Key.Key_Left, -1), (Qt.Key.Key_Right, 1)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(
                Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(
                lambda step=step: self.step_event(step))
        # Up and Down step through the comparisons while the plot has
        # the keyboard. They belong to the pane rather than to the
        # window: the tree uses the same keys to move its selection, and
        # a window-wide shortcut would take them from it. Click the plot
        # once and the arrows are the channel's; click the tree and they
        # are the tree's again.
        for key, step in ((Qt.Key.Key_Up, -1), (Qt.Key.Key_Down, 1)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(
                Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(
                lambda step=step: self.step_pair(step))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # FRFs get a filter rather than a mode: the diagonal of the grid
        self.drive_point_action: QAction = QAction(control_icon('drive_point'),
                                          'Drive points', self)
        self.drive_point_action.setCheckable(True)
        self.drive_point_action.setToolTip(
            'Only the drive-point FRFs: response and reference at the '
            'same DOF')
        self.drive_point_action.triggered.connect(
            lambda checked: self.drive_points_toggled.emit(checked))
        toolbar.addAction(self.drive_point_action)
        # ...and a way to read the whole matrix at once: its singular
        # values per frequency line
        self.cmif_action: QAction = QAction(control_icon('cmif'), 'CMIF', self)
        self.cmif_action.setCheckable(True)
        self.cmif_action.setToolTip(
            'Complex Mode Indicator Function — the singular values of the '
            'FRF matrix at every frequency line; peaks mark the modes')
        self.cmif_action.triggered.connect(lambda _checked: self.reread.emit())
        toolbar.addAction(self.cmif_action)
        # ---- the readings of a record: one at a time, or none -------
        #
        # Fenced by separators and held in an exclusive group, because
        # that is what they are: four ways of reading one record, and a
        # record is read one way at a time. The exclusion used to be
        # hand-rolled — every button stood its siblings down by name,
        # which is four lists to keep in step and was already wrong once
        # (shocks up with the filter still on). Qt has the rule;
        # `_readings_settled` only has to follow it (Brandon,
        # 2026-08-27).
        #
        # ExclusiveOptional, not Exclusive: none of them is a legitimate
        # state and the common one — the plain trace, unmarked — so
        # clicking the checked button turns it off rather than being
        # refused. The separators say "pick one of these"; they do not
        # say "you must".
        #
        # The 2-D/3-D toggle is deliberately outside the fence. It is
        # not a fifth reading; it is how any of them is drawn.
        self.reading_group: QActionGroup = QActionGroup(self)
        self.reading_group.setExclusionPolicy(
            QActionGroup.ExclusionPolicy.ExclusiveOptional)
        # a time history gets the averaging: the parameters beside the
        # plot and the frames shaded on it, or neither
        self.averaging_action: QAction = QAction(control_icon('averaging'),
                                        'Averaging', self)
        self.averaging_action.setCheckable(True)
        self.averaging_action.setToolTip(
            'Show the frames a PSD would be averaged over, and the '
            'parameters that set them')
        self.averaging_action.triggered.connect(self._choose_averaging)
        toolbar.addAction(self.averaging_action)
        # the filter a record would be read through: the pass-band
        # edges and order beside the plot, the filtered trace
        # previewed over the raw one (Brandon, 2026-08-24)
        self.filter_action: QAction = QAction(control_icon('filter'),
                                              'Filter', self)
        self.filter_action.setCheckable(True)
        self.filter_action.setToolTip(
            'Show the filter a record would be read through — low-, '
            'high- or band-pass — previewed over the raw trace')
        self.filter_action.triggered.connect(self._choose_filter)
        toolbar.addAction(self.filter_action)
        # how a sweep's levels are read out of the record: the
        # smoothing, automatic or set, with the readings it gives
        # sampled beside the plot (Brandon, 2026-09-30)
        self.sine_action: QAction = QAction(control_icon('sine_levels'),
                                            'Sine Levels', self)
        self.sine_action.setCheckable(True)
        self.sine_action.setToolTip(
            'Show how the sweep’s levels are read out of this record — '
            'the smoothing and the clock refinement — with the readings '
            'sampled at that setting')
        self.sine_action.triggered.connect(self._choose_sine)
        toolbar.addAction(self.sine_action)
        # the stretch a record would be cut to: start and stop beside
        # the plot, the discarded ends grayed over on it — the
        # averaging span's editing grammar with nothing computed
        # (Brandon, 2026-08-28)
        self.truncate_action: QAction = QAction(control_icon('truncate'),
                                                'Truncate', self)
        self.truncate_action.setCheckable(True)
        self.truncate_action.setToolTip(
            'Show the stretch a record would be cut to — drag the '
            'span; Apply Truncation in the pane makes the cut record')
        self.truncate_action.triggered.connect(self._choose_truncate)
        toolbar.addAction(self.truncate_action)
        # how Gaussian each channel of a record is, as a bar apiece:
        # a spectrum says nothing about the shape of the distribution
        # that made it, and two records with the same PSD fatigue an
        # article differently (Brandon, 2026-08-24)
        self.kurtosis_action: QAction = QAction(control_icon('kurtosis'),
                                                'Kurtosis', self)
        self.kurtosis_action.setCheckable(True)
        self.kurtosis_action.setToolTip(
            'Pearson kurtosis of every channel — 3 is Gaussian, higher '
            'means peaks the spectrum did not predict, lower means a '
            'clipped or non-random record')
        self.kurtosis_action.triggered.connect(self._choose_reading)
        toolbar.addAction(self.kurtosis_action)
        # the five-minute look at one channel, in numbers: the moments,
        # the crest factor, the rail, the tails against the Gaussian's
        # and the kurtosis frame by frame, each against its band. Beside
        # the kurtosis bars because it reads the same span, and replaces
        # the trace for the same reason they do
        self.statistics_action: QAction = QAction(
            control_icon('statistics'), 'Statistics', self)
        self.statistics_action.setCheckable(True)
        self.statistics_action.setToolTip(
            'One channel read in the time domain, each statistic judged '
            'against its band')
        self.statistics_action.triggered.connect(self._choose_reading)
        toolbar.addAction(self.statistics_action)
        # the record's lines scaled so a sine reads its amplitude, in
        # peak units, under a window chosen for it: where a tone on the
        # record is read, which its ASD cannot give. A reading of the
        # record, so it replaces the trace as the statistics do
        self.amplitude_action: QAction = QAction(
            control_icon('amplitude'), 'Amplitude', self)
        self.amplitude_action.setCheckable(True)
        self.amplitude_action.setToolTip(
            "One channel's amplitude spectrum in peak units: the window's "
            "amplitude correction applied and named, the frames' "
            'magnitudes averaged, the tallest line read')
        self.amplitude_action.triggered.connect(self._choose_reading)
        toolbar.addAction(self.amplitude_action)
        # and the same for the events in a transient record: the windows
        # bracketed on the plot and listed beside it, or neither
        self.shocks_action: QAction = QAction(control_icon('shocks'), 'Shocks', self)
        self.shocks_action.setCheckable(True)
        self.shocks_action.setToolTip(
            'Show the shocks found in this record, and the windows each '
            'spectrum would be computed over')
        self.shocks_action.triggered.connect(self._choose_shocks)
        toolbar.addAction(self.shocks_action)
        # where the record's frequencies are moment by moment: the one
        # reading here that answers *when*, which a spectrum has
        # already averaged away (Brandon, 2026-08-27)
        self.wavelet_action: QAction = QAction(control_icon('wavelet'),
                                               'Wavelet', self)
        self.wavelet_action.setCheckable(True)
        self.wavelet_action.setToolTip(
            'The scalogram: time across, frequency up, magnitude as '
            'color — where a rattle, a ring-down or a sweep actually '
            'is in the record')
        self.wavelet_action.triggered.connect(self._choose_reading)
        toolbar.addAction(self.wavelet_action)
        # a classical shock pulse judged against its envelope and its
        # velocity change: a reading of the record, not a mark on it,
        # so it replaces the trace as the wavelet does
        self.pulse_action: QAction = QAction(control_icon('pulse'),
                                             'Pulse Check', self)
        self.pulse_action.setCheckable(True)
        self.pulse_action.setToolTip(
            'Judge a classical pulse — half-sine, sawtooth or trapezoid '
            '— against its tolerance envelope and its velocity change')
        self.pulse_action.triggered.connect(self._choose_reading)
        toolbar.addAction(self.pulse_action)
        # whether a shaker can play this record as a replication
        # reference: its travel and velocity at a few high-pass cutoffs
        # against the ratings, and the force at its largest peak
        self.fit_action: QAction = QAction(control_icon('replication_fit'),
                                           'Replication Fit', self)
        self.fit_action.setCheckable(True)
        self.fit_action.setToolTip(
            'Whether a shaker can play this record: the force at its '
            'largest peak, and its travel each way and peak velocity at '
            'each candidate high-pass cutoff, against the ratings')
        self.fit_action.triggered.connect(self._choose_reading)
        toolbar.addAction(self.fit_action)
        # whether a pulse fits on the shaker: the record integrated
        # raw to velocity and displacement, against the shock ratings
        # typed beside it
        self.stroke_action: QAction = QAction(control_icon('stroke'),
                                              'Stroke', self)
        self.stroke_action.setCheckable(True)
        self.stroke_action.setToolTip(
            'Whether a pulse fits on the shaker: the acceleration '
            'integrated raw to velocity and displacement, against the '
            'shock stroke, velocity and force typed beside it')
        self.stroke_action.triggered.connect(self._choose_reading)
        toolbar.addAction(self.stroke_action)
        # what taking the record to a lower rate does: the anti-alias
        # filter, the span and the dropped lines on its spectrum, and
        # where content above the new Nyquist frequency folds
        self.resample_action: QAction = QAction(control_icon('resample'),
                                                'Resample', self)
        self.resample_action.setCheckable(True)
        self.resample_action.setToolTip(
            'The record at a lower sample rate: the anti-alias filter, '
            'the span and the dropped lines on its spectrum, and where '
            'content above the new Nyquist frequency folds')
        self.resample_action.triggered.connect(self._choose_reading)
        toolbar.addAction(self.resample_action)
        for reading in self._READINGS:
            self.reading_group.addAction(
                getattr(self, f'{reading.key}_action'))
        # the banded reading of a spectrum: the octave conversion
        # previewed as steps over the narrowband, its spacing in the
        # pane, the act on the pane's own button (Brandon,
        # 2026-08-29) — principle 13's four parts for a PSD
        self.octave_action: QAction = QAction(control_icon('octave'),
                                              'Octave Bands', self)
        self.octave_action.setCheckable(True)
        self.octave_action.setToolTip(
            'Preview the spectrum integrated onto proportional bands, '
            'over the narrowband it comes from')
        # a lambda, not a partial: PySide reads a slot's signature to
        # decide whether `triggered` passes `checked`, and called a
        # partial with no `checked` at all
        self.octave_action.triggered.connect(
            lambda checked=False: self._choose_toggle('octave', checked))
        toolbar.addAction(self.octave_action)
        # and the tone read off it: a cursor on the spectrum, the tone
        # under it read as an amplitude beside the spike it stands at,
        # in the pane. A reading of the spectrum as it is, so a toggle
        # with no act, like the octave preview before its button
        self.tone_action: QAction = QAction(control_icon('tone'), 'Tones',
                                            self)
        self.tone_action.setCheckable(True)
        self.tone_action.setToolTip(
            'Read the tone under a cursor on the spectrum: its amplitude '
            'summed from the lines it leaks into, beside the height of '
            'its spike, which moves with the line spacing and the window')
        self.tone_action.triggered.connect(
            lambda checked=False: self._choose_toggle('tone', checked))
        toolbar.addAction(self.tone_action)
        # an FRF read behind: the ASD-ratio transmissibility beside H1
        # and the band H1's coherence and averages give, with a cursor
        # read in the pane. A reading of an FRF that exists, so a toggle
        # with no act, like the tones
        self.transmissibility_action: QAction = QAction(
            control_icon('transmissibility'), 'Transmissibility', self)
        self.transmissibility_action.setCheckable(True)
        self.transmissibility_action.setToolTip(
            'Draw the ASD-ratio transmissibility, the root of the '
            'response ASD over the input ASD, dashed beside H1, with the '
            "band H1's random error puts about it from its coherence "
            'and averages: the two part wherever the coherence falls, '
            'the ratio always the higher; which is nearer the truth '
            'depends on which channel the noise is on')
        self.transmissibility_action.triggered.connect(
            lambda checked=False: self._choose_toggle('transmissibility',
                                                      checked))
        # offered by the render that draws an FRF computed from a time
        # history, and by nothing else, so it starts put away
        self.transmissibility_action.setVisible(False)
        toolbar.addAction(self.transmissibility_action)
        # and what the average is worth: the chi-square band the true
        # spectrum lies in, shaded on it, the confidence and the count
        # in the pane. A statement about the spectrum as it is, so a
        # toggle with no act, like the tones
        self.confidence_action: QAction = QAction(
            control_icon('confidence'), 'Confidence', self)
        self.confidence_action.setCheckable(True)
        self.confidence_action.setToolTip(
            'Shade the band the true spectrum lies in, at 90 or 95 % '
            'confidence, from the degrees of freedom it was averaged to')
        self.confidence_action.triggered.connect(
            lambda checked=False: self._choose_toggle('confidence', checked))
        toolbar.addAction(self.confidence_action)
        # and the check on the spectrum as a whole: the rms from the
        # area under it beside the rms of the time history it came
        # from, the gap in dB and the verdict in the pane. A reading of
        # the spectrum as it is, so a toggle with no act, like the tones
        self.rms_check_action: QAction = QAction(
            control_icon('rms_check'), 'RMS Check', self)
        self.rms_check_action.setCheckable(True)
        self.rms_check_action.setToolTip(
            'Check the spectrum against its time history: the rms from '
            'the area under it beside the rms of the record, the gap in '
            'dB, and whether it is within the threshold')
        self.rms_check_action.triggered.connect(
            lambda checked=False: self._choose_toggle('rms_check', checked))
        toolbar.addAction(self.rms_check_action)
        # a multi-input test's two matrices, each checked line by line:
        # the plant (an FRF from several references, or the drives' CSD)
        # for how near singular it is, and a CSD for whether any signals
        # could have it. Readings of objects that exist, so toggles, one
        # up at a time; each draws a figure of its own
        self.conditioning_action: QAction = QAction(
            control_icon('conditioning'), 'Conditioning', self)
        self.conditioning_action.setCheckable(True)
        self.conditioning_action.setToolTip(
            'Draw the singular values of the matrix and its condition '
            'number at every line, the lines over the limit hatched: '
            'there the references reach the responses in nearly the same '
            'proportions, and an inverse of the matrix is mostly noise')
        self.conditioning_action.triggered.connect(
            lambda checked=False: self._choose_matrix_check('conditioning',
                                                            checked))
        self.conditioning_action.setVisible(False)
        toolbar.addAction(self.conditioning_action)
        self.realizability_action: QAction = QAction(
            control_icon('realizability'), 'Realizability', self)
        self.realizability_action.setCheckable(True)
        self.realizability_action.setToolTip(
            'Draw the eigenvalues of the coherence matrix at every line, '
            'the lines where one is negative hatched: there no set of '
            'signals has these coherences and phases together, and no '
            'controller can play them')
        self.realizability_action.triggered.connect(
            lambda checked=False: self._choose_matrix_check('realizability',
                                                            checked))
        self.realizability_action.setVisible(False)
        toolbar.addAction(self.realizability_action)
        # a sine specification read against the modes it sweeps
        # through: the time and cycles in each half-power band, and
        # whether the rate lets the mode build. A reading of the
        # specification alone, so it stands beside the stage toggle
        # rather than among a record's readings
        self.sweep_rate_action: QAction = QAction(
            control_icon('sweep_rate'), 'Sweep Rate', self)
        self.sweep_rate_action.setCheckable(True)
        self.sweep_rate_action.setToolTip(
            'Check the sweep rate against the modes it passes: the '
            'time and cycles in each half-power band, and whether the '
            'rate is under 87 fn / Q\u00b2 oct/min on a log sweep, '
            'fn\u00b2 / Q\u00b2 Hz/s on a linear one')
        self.sweep_rate_action.triggered.connect(
            lambda checked=False: self._choose_spec_reading('sweep_rate',
                                                            checked))
        # offered by the render that draws a sine specification, and by
        # nothing else, so it starts put away
        self.sweep_rate_action.setVisible(False)
        toolbar.addAction(self.sweep_rate_action)
        # the same specification read as segments of constant
        # displacement, velocity and acceleration, drawn three ways with
        # the crossovers checked; the other reading of the specification
        # alone, so it shares the sweep rate's fence and takes turns
        # with it
        self.sine_profile_action: QAction = QAction(
            control_icon('sine_profile'), 'Sine Profile', self)
        self.sine_profile_action.setCheckable(True)
        self.sine_profile_action.setToolTip(
            'Read the tone as segments of constant displacement, velocity '
            'and acceleration: the profile drawn three ways, each '
            'crossover checked against its breakpoint, and the most it '
            'asks for in each')
        self.sine_profile_action.triggered.connect(
            lambda checked=False: self._choose_spec_reading('sine_profile',
                                                            checked))
        self.sine_profile_action.setVisible(False)
        toolbar.addAction(self.sine_profile_action)
        # ---- and out of the fence again -----------------------------
        # the fit's residual CMIF: what is left after the confirmed
        # modes are taken out — the curve Find Mode hunts. A plot
        # overlay, so it lives here with the other show/hide toggles
        # rather than among the fit's verbs.
        self.residual_action: QAction = QAction(control_icon('residual'),
                                       'Residual', self)
        self.residual_action.setCheckable(True)
        self.residual_action.setToolTip(
            'Show what is left of the CMIF after the confirmed modes '
            'are taken out — the curve Find Mode hunts and the parabola '
            'rides')
        self.residual_action.toggled.connect(self.residual_toggled.emit)
        toolbar.addAction(self.residual_action)
        # an FRF beside a shape set reads two ways: edit the fit, or see
        # the resynthesis over the measurement. One sticky choice.
        self.pair_group: QActionGroup = QActionGroup(self)
        self.pair_edit_action: QAction = QAction(control_icon('edit'),
                                        'Edit Fit', self)
        self.pair_edit_action.setToolTip(
            'Open the modal fit, seeded with this shape set')
        self.pair_synthesis_action: QAction = QAction(control_icon('curves'),
                                             'Resynthesis', self)
        self.pair_synthesis_action.setToolTip(
            "Overlay the modal model's resynthesis on the measurement")
        for action, mode in ((self.pair_edit_action, 'fit'),
                             (self.pair_synthesis_action, 'overlay')):
            action.setCheckable(True)
            self.pair_group.addAction(action)
            toolbar.addAction(action)
            action.triggered.connect(
                lambda _checked=False, mode=mode:
                self.pair_mode_chosen.emit(mode))

        # Every set of choices gets a fence, and so does the 2-D/3-D
        # toggle — which is fenced *because* it is not a choice among
        # the others but a setting applying to whichever is in force
        # (Brandon, 2026-08-27). Done here in one place rather than by
        # hand as each group is built: a bar with a group somebody
        # forgot to fence is exactly the undivided row this is meant to
        # stop being.
        fence(toolbar, [
            self.plot_mode_group.actions(),
            [self.waterfall_action],
            [self.octave_action, self.tone_action, self.confidence_action,
             self.rms_check_action],
            [self.sweep_rate_action, self.sine_profile_action],
            [self.transmissibility_action],
            [self.conditioning_action, self.realizability_action],
            self.comparison_group.actions(),
            self.spectra_group.actions(),
            self.replication_group.actions(),
            self.srs_group.actions(),
            self.reading_group.actions(),
            self.pair_group.actions(),
        ])
        return toolbar

    def copy_view(self) -> bool:
        """Copy what the pane shows to the clipboard as an image: the
        3-D view when it is up, the flat plot otherwise.

        The plot is drawn the way the headless plots are
        (`plot.render_image`), on the theme's background and at the
        screen's pixel density, so a Retina display copies a Retina
        image with every font at its set size. Emits `copied` with a
        line for the status bar, and returns False when nothing was
        drawn."""
        from PySide6.QtWidgets import QApplication

        if (self.waterfall_plotter is not None
                and not self.graphics.isVisibleTo(self)):
            image = image_of(self.waterfall_plotter.screenshot(return_img=True))
            what = 'stage'
        else:
            from ..plot import render_image

            # at the screen's density, drawn the one way a plot becomes
            # pixels: pyqtgraph's exporter, asked for a retina width,
            # shrank the legend's text and grew the ticks'
            colors = resolve_theme(self.theme_name)
            image = render_image(self.graphics,
                                 self.graphics.devicePixelRatioF(),
                                 colors['plot_background'])
            what = 'plot'
        if image is None or image.isNull():
            self.copied.emit('Nothing drawn to copy')
            return False
        QApplication.clipboard().setImage(image)
        self.copied.emit(f'{what.capitalize()} copied to the clipboard '
                         f'({image.width()} × {image.height()})')
        return True

    #: the readings of a time history, one entry each: what the pane
    #: offers, checks, puts away and keeps is derived from this list,
    #: so a new reading is a row here rather than an edit to every
    #: other reading's handler. The order is the bar's.
    _READINGS: ClassVar[tuple[Reading, ...]] = (
        Reading('averaging', 'averaging_panel', 'averaging'),
        Reading('filter', 'filter_panel', 'averaging'),
        # offered for a record that has a sweep specification to be
        # read against, and nowhere else (principle 3)
        Reading('sine', 'sine_panel', 'sine'),
        # the cut is offered wherever the averaging is: both are
        # readings of a record's stretch of time
        Reading('truncate', 'truncate_panel', 'averaging'),
        # offered wherever a record is, shock or random: a record is
        # what has channels to be Gaussian
        Reading('kurtosis', None, 'record', flat=True),
        # offered where the averaging is, as the scalogram is: its
        # bands are for a record meant to be Gaussian random, which a
        # shock record is not (2026-10-16, on review)
        Reading('statistics', 'statistics_panel', 'averaging',
                flat=True, sticky=True),
        # offered where the averaging is: its frames are the
        # averaging's, and only its window is its own
        Reading('amplitude', 'amplitude_panel', 'averaging',
                flat=True, sticky=True),
        Reading('shocks', 'shock_panel', 'shocks'),
        Reading('wavelet', 'wavelet_panel', 'averaging', sticky=True),
        # offered wherever the shocks are: a record that may hold a
        # pulse
        Reading('pulse', 'pulse_panel', 'shocks', sticky=True),
        # both integrate acceleration, so they are offered for a record
        # that holds some in a declared unit and nowhere else
        Reading('fit', 'replication_fit_panel', 'fit',
                flat=True, sticky=True),
        Reading('stroke', 'stroke_panel', 'fit', flat=True, sticky=True),
        # offered where the averaging is: a record with a sample rate
        # to lower, read on its spectrum, which has no 3-D form here
        Reading('resample', 'resample_panel', 'averaging', flat=True,
                sticky=True),
    )

    def show_acts(self, acts: Sequence[tuple]) -> None:
        """Put the acts the selection can take on this bar
        (see `offer_acts`)."""
        offer_acts(self.toolbar, self.__dict__.setdefault('_acts', {}), acts)
        self.sync_toolbar()

    def _readings_settled(self) -> None:
        """Bring the flags and the panels into line with the buttons.

        The action group has already done the unchecking — that is what
        it is for — so this reads the buttons rather than deciding
        anything: whatever is not checked has its flag cleared and its
        panel put away.

        One implementation, run after every reading toggle. The version
        this replaces had each button naming its siblings, which is the
        same rule written four times; the time it disagreed with itself
        left shocks up with the filter still on.
        """
        for reading in self._READINGS:
            action = getattr(self, f'{reading.key}_action')
            setattr(self, f'{reading.key}_wanted', action.isChecked())
            if not action.isChecked() and reading.panel is not None:
                getattr(self, reading.panel).hide()

    def _choose_averaging(self, checked: bool) -> None:
        """The averaging button: remember it, then say so.

        Turning one reading on turns the others off. They mark the same
        record differently — frames to average against events to
        analyze — and a record is being read one way or the other, never
        both. Overlaid they are just two sets of shading on one trace.
        """
        self._readings_settled()
        self.averaging_toggled.emit(checked)

    def _choose_filter(self, checked: bool) -> None:
        """The filter button: remember it, then say so."""
        self._readings_settled()
        self.filter_toggled.emit(checked)

    def _choose_sine(self, checked: bool) -> None:
        """The sine button: remember it, then say so."""
        self._readings_settled()
        self.sine_toggled.emit(checked)

    def _choose_truncate(self, checked: bool) -> None:
        """The truncate button: remember it, then say so."""
        self._readings_settled()
        self.truncate_toggled.emit(checked)

    def _choose_toggle(self, key: str, checked: bool) -> None:
        """The octave, tones, transmissibility, confidence, rms check
        or sweep-rate button, each a toggle of its own: remember it,
        put its panel away when it is unchecked, then say so."""
        setattr(self, f'{key}_wanted', bool(checked))
        if not checked:
            getattr(self, f'{key}_panel').hide()
        getattr(self, f'{key}_toggled').emit(checked)

    def showing(self, key: str) -> bool:
        """Whether the reading or toggle `key` (its action is
        `<key>_action`) is offered and checked."""
        action = getattr(self, f'{key}_action')
        return action.isVisible() and action.isChecked()

    def _choose_matrix_check(self, key: str, checked: bool) -> None:
        """The Conditioning or Realizability button: a toggle like the
        tones', and one of the two at a time, since each draws a figure
        of its own in place of the plot."""
        if checked:
            other = ('realizability' if key == 'conditioning'
                     else 'conditioning')
            setattr(self, f'{other}_wanted', False)
            getattr(self, f'{other}_action').setChecked(False)
            getattr(self, f'{other}_panel').hide()
        self._choose_toggle(key, checked)

    def _offer(self, key: str, panel: str | None, offered: bool) -> None:
        """Put `<key>_action` on the bar or take it off, checked as it
        was last left; its panel is put away unless it is up."""
        wanted = offered and getattr(self, f'{key}_wanted')
        getattr(self, f'{key}_action').setVisible(offered)
        getattr(self, f'{key}_action').setChecked(wanted)
        if not wanted and panel is not None:
            getattr(self, panel).hide()

    def offer_sweep_rate(self, offered: bool) -> None:
        """Show or hide the Sweep Rate and Sine Profile toggles, the two
        readings of a sine specification alone; each checked as it was
        last left. A panel goes with its toggle when it is not offered or
        not checked."""
        self._offer('sweep_rate', 'sweep_rate_panel', offered)
        self._offer('sine_profile', 'sine_profile_panel', offered)
        if offered:
            self.toolbar.setVisible(True)

    def _choose_spec_reading(self, key: str, checked: bool) -> None:
        """The Sweep Rate or Sine Profile button: one reading of the
        specification at a time, so checking one puts the other away
        before the toggle is said."""
        if checked:
            other = 'sine_profile' if key == 'sweep_rate' else 'sweep_rate'
            if getattr(self, f'{other}_wanted'):
                getattr(self, f'{other}_action').setChecked(False)
                setattr(self, f'{other}_wanted', False)
                getattr(self, f'{other}_panel').hide()
        self._choose_toggle(key, checked)

    def _choose_reading(self, checked: bool) -> None:
        """The kurtosis, statistics, wavelet or replication fit button: a
        different *reading* of the record rather than a mark on it, so it
        replaces the trace outright — and the mark-the-trace toggles stand down, having
        nothing to mark."""
        del checked                      # the buttons are the state now
        self._readings_settled()
        self.reread.emit()

    def _settings_edited(self, key: str, settings) -> None:
        """A reading's setting moved: keep it, then draw again.

        Kept on the pane as `<key>_settings` rather than in the panel,
        so a tuned range, a pulse or a shaker's ratings survive clicking
        away to another object and back: the same stickiness every
        other view choice on this bar has.
        """
        setattr(self, f'{key}_settings', dict(settings))
        self.reread.emit()

    @property
    def rms_wanted(self) -> bool:
        """Whether the RMS reading of a specification is checked.

        Only the checked state is read, not whether the toggle is
        offered, so a render can lay out its plots before the bar is
        set up."""
        return self.rms_action.isChecked()

    @property
    def signature_wanted(self) -> bool:
        """Whether the Signature reading of two signatures is
        checked; like `rms_wanted`, only the checked state is read."""
        return self.signature_action.isChecked()

    def offer_signature(self, offered: bool) -> None:
        """Show or hide the Signature toggle for two signatures
        selected together. Put away, the reading's panel goes with it."""
        self.signature_action.setVisible(offered)
        if offered:
            self.toolbar.setVisible(True)
        else:
            self.signature_panel.hide()

    def offer_rms(self, offered: bool) -> None:
        """Show or hide the RMS toggle for a specification shown on
        its own."""
        self.rms_action.setVisible(offered)
        if offered:
            self.toolbar.setVisible(True)

    def _choose_comparison(self, which: str) -> None:
        """Which reading of the comparison the top plot shows."""
        self.comparison_view = which
        self.comparison_chosen.emit(which)

    def _choose_spectra_view(self, which: str) -> None:
        self.spectra_view = which
        self.spectra_view_chosen.emit(which)

    def show_spectra_views(self, offered: bool) -> None:
        """Show or hide the readings of two selected densities
        (overlay, ratio, signal-to-noise)."""
        for action in self.spectra_actions.values():
            action.setVisible(offered)
        if offered:
            self.toolbar.setVisible(True)
            self.spectra_actions[self.spectra_view].setChecked(True)

    def show_comparison_views(self, offered: bool) -> None:
        """Show or hide the four readings of a comparison (curves,
        error, lines, chance).

        Only a specification with a measurement against it has four
        readings. Anything else has one, and a choice of one is not
        offered. The Chance reading's panel is put away with them, and
        whenever another reading is chosen; the drawing puts it up.
        """
        for action in self.comparison_actions.values():
            action.setVisible(offered)
        if not offered or self.comparison_view != 'chance':
            self.chance_panel.hide()
        if offered:
            self.toolbar.setVisible(True)
            self.comparison_actions[self.comparison_view].setChecked(True)

    def show_scaling(self, text: str | None) -> None:
        """Show the comparison scaling field with `text`, or hide it
        with None.

        The field is set again on every drawing, so setting it here does
        not count as an edit and does not emit `scaling_edited`.
        """
        self.scaling_action.setVisible(text is not None)
        if text is None:
            return
        self._loading_scaling = True
        try:
            self.scaling_edit.setText(text)
        finally:
            self._loading_scaling = False
        self.toolbar.setVisible(True)

    def _choose_srs(self, which: str) -> None:
        """Which reading of a shock-spectrum comparison the plot shows."""
        self.srs_view = which
        self.srs_chosen.emit(which)

    #: the two readings' names, by comparison: a shock spectrum against
    #: its requirement, or sine levels against the tones they were
    #: controlled to — the same pair of readings, said in each one's words
    #: (the sine comparison read "SRS Error" until 2026-10-13)
    SRS_LABELS: ClassVar[dict] = {
        'srs': {'curves': ('Spectra',
                           'The measured shock spectra against the one required'),
                'error': ('SRS Error',
                          ('How far each channel sits from the spectrum it had '
                           'to meet, RMS across the band in dB')),
                'points': ('Points Rule',
                           ('Each spectrum against the tolerance band point by '
                            'point: how many points are inside, under the '
                            'rule the test plan names'))},
        'sine': {'curves': ('Levels',
                            'The extracted levels against the levels required'),
                 'error': ('Level Error',
                           ('How far each channel sits from the level it had to '
                            'reach, RMS across the sweep in dB'))},
    }

    def show_srs_views(self, offered: bool, kind: str = 'srs',
                       readings: Sequence[str] | None = None) -> None:
        """Show or hide the readings of a shock-spectrum
        comparison, or, with `kind` 'sine', of sine levels against their
        specification. The points rule is a shock's alone, so the sine
        comparison has two. `readings` narrows the ones offered, as
        several spectra of different polarities are read by the points
        rule alone. While shown, the three comparison readings are
        hidden."""
        labels = self.SRS_LABELS[kind]
        for key, action in self.srs_actions.items():
            text, tip = labels.get(key, (action.text(), action.toolTip()))
            action.setText(text)
            action.setToolTip(tip)
            action.setVisible(offered and key in labels
                              and (readings is None or key in readings))
        if not offered or 'points' not in labels:
            self.srs_tolerance_panel.hide()
        if offered:
            for action in self.comparison_actions.values():
                action.setVisible(False)
            self.toolbar.setVisible(True)
            # a reading this comparison does not have stands for its
            # first, as `srs_reading` says
            self.srs_actions[self.srs_reading(kind, readings)].setChecked(True)

    def srs_reading(self, kind: str = 'srs',
                    readings: Sequence[str] | None = None) -> str:
        """The reading a comparison of `kind` shows: the sticky choice,
        or the first one offered when the choice is one this comparison
        does not have (the points rule, carried from a shock to a sine
        comparison, reads as its curves). `readings` narrows the ones
        offered, as for `show_srs_views`. The sticky choice is kept, so
        the comparison it belongs to shows it again."""
        offered = [key for key in self.SRS_LABELS[kind]
                   if readings is None or key in readings]
        return self.srs_view if self.srs_view in offered else offered[0]

    def _choose_replication(self, which: str) -> None:
        """Which reading of a transient replication the plot shows."""
        self.replication_view = which
        self.replication_chosen.emit(which)

    def show_replication_views(self, offered: bool) -> None:
        """Show or hide the readings of a transient replication
        (overlay, waveform error, segments). Hidden, the segments'
        panel goes with them.

        Only a transient specification with a record against it has
        them. While they are shown, the three readings of a random
        comparison are hidden: both sets answer how the measurement
        compared, and offering all five together would invite reading a
        waveform error against an abort band that a transient does not
        have.
        """
        for action in self.replication_actions.values():
            action.setVisible(offered)
        if not offered:
            self.segments_panel.hide()
        if offered:
            for action in self.comparison_actions.values():
                action.setVisible(False)
            self.toolbar.setVisible(True)
            self.replication_actions[self.replication_view].setChecked(True)

    def show_events(self, labels: Sequence[tuple[str, int]]) -> None:
        """Fill the event box with these repeats, keeping the one already
        chosen.

        `labels` is one label per repeat, in order. The choice is kept
        by index, unlike the channel box beside it, because a repeat is
        its index (the third playing of the waveform is the third
        whatever else the record holds), while a channel is a DOF that
        may come and go. The box is hidden for fewer than two repeats.

        No repeat is marked as the worst. Which one is worst depends on
        the reading and on what the test article is for, so the box
        lists the repeats and the numbers beside the plot are what to
        decide with.
        """
        wanted = list(labels)
        self.event_action.setVisible(len(wanted) > 1)
        if len(wanted) < 2:
            return
        self.toolbar.setVisible(True)
        held = self.chosen_event()
        self._loading_events = True
        try:
            self.event_box.clear()
            for index, label in enumerate(wanted):
                self.event_box.addItem(label, index)
            if held is not None and 0 <= held < len(wanted):
                self.event_box.setCurrentIndex(held)
        finally:
            self._loading_events = False

    def chosen_event(self) -> int | None:
        """The index of the repeat being shown, or None."""
        return self.event_box.currentData()

    def step_event(self, step: int) -> None:
        """Move the event box by `step`, stopping at either end.

        It stops rather than wrapping because the repeats are in time
        order, and going from the last one back to the first would look
        like a step forward.
        """
        count = self.event_box.count()
        if count < 2:
            return
        index = self.event_box.currentIndex() + step
        if 0 <= index < count:
            self.event_box.setCurrentIndex(index)

    def _choose_shocks(self, checked: bool) -> None:
        """The shocks button, and the same exclusion the other way.

        Shocks up with the filter still on was exactly the both-at-once
        state the exclusion exists to refuse (Brandon, 2026-08-24), and
        it happened because the rule lived in four places. It lives in
        the action group now.
        """
        self._readings_settled()
        self.shocks_toggled.emit(checked)

    # A time history opens plain — the record on the stage, no reading
    # pre-selected — whatever the project type (Brandon, 2026-08-30).
    # `read_record_as` used to pre-select a reading from the type:
    # random and transient opened with the averaging view up (marks
    # before the data had been looked at), shock opened on detection.
    # The toggles are sticky once touched, and that is the whole of
    # the mechanism now.

    def _choose_plot_mode(self, mode: str) -> None:
        """An explicit choice, which outranks the default until the
        selection moves on."""
        self.plot_mode = mode
        self.reread.emit()

    def reset_plot_mode(self) -> None:
        """Clear the plot mode chosen on the bar, so the data chooses
        again. Called when the selection changes, so a choice applies to
        the object it was made for and not to every later one."""
        self.plot_mode = None

    # ---- what the pane is showing ----------------------------------------

    @property
    def showing_averaging(self) -> bool:
        """Whether the averaging view is checked. It applies only to
        a time history, so the caller checks that first."""
        return self.averaging_action.isChecked()

    @property
    def showing_filter(self) -> bool:
        """Whether the filter view is checked. It applies only to a
        time history, so the caller checks that first."""
        return self.filter_action.isChecked()

    @property
    def showing_sine(self) -> bool:
        """Whether the sine view is checked. It applies only to a
        time history beside a sweep specification, so the caller checks
        that first."""
        return self.sine_action.isChecked()

    @property
    def showing_octave(self) -> bool:
        """Whether the octave-band view is checked. It applies only to
        a plain PSD or CPSD, so the caller checks that first."""
        return self.octave_action.isChecked()

    @property
    def showing_tone(self) -> bool:
        """Whether the tone readout is checked. It applies only to a
        plain autospectrum on evenly spaced lines, so the caller checks
        that first."""
        return self.tone_action.isChecked()

    @property
    def showing_confidence(self) -> bool:
        """Whether the confidence band is checked. It applies only to an
        autospectrum, so the caller checks that first."""
        return self.confidence_action.isChecked()

    @property
    def showing_truncate(self) -> bool:
        """Whether the truncate view is checked. It applies only to a
        time history, so the caller checks that first."""
        return self.truncate_action.isChecked()

    @property
    def showing_shocks(self) -> bool:
        """Whether the shock view is checked. It applies only to a
        time history, so the caller checks that first."""
        return self.shocks_action.isChecked()

    @property
    def showing_cmif(self) -> bool:
        """Whether the singular-value reading (CMIF) is checked. It
        applies only to FRFs, so the caller checks that first."""
        return self.cmif_action.isChecked()

    @property
    def showing_waterfall(self) -> bool:
        """Whether the 3-D form is checked. It applies only where the
        window offered it, so the caller checks that first.

        The filter view does not change this; its preview is drawn in
        3-D too. While the specification sheet is open, `flat_only`
        keeps the drawing flat whatever the toggle says, because the
        sheet's handles are on the flat plot.
        """
        # The filter view used to outrank this and force the drawing
        # flat: my judgment that two overlaid ribbons would occlude each
        # other, made without drawing them. Wrong twice over (Brandon,
        # 2026-08-25): the stage already draws paired data at one
        # station, and a view choice that silently overrides another
        # view choice is exactly what this interface does not do. The
        # preview is stage geometry now, so both readings stand.
        # `flat_only` for the specification sheet: Brandon, 2026-09-06.
        return self.waterfall_action.isChecked() and not self.flat_only

    #: the flat plot only, whatever the 3D toggle says — set by the
    #: window while the specification sheet is open, since the sheet's
    #: handles live on the flat plot (Brandon, 2026-09-06)
    flat_only: bool = False

    def offer_waterfall(self, offered: bool) -> None:
        """Show or hide the 2D/3D toggle.

        Called by the code that draws plain curves, like `show_pairs`,
        because `show_controls` runs before the window knows whether
        the drawing will be plain curves. The toggle is offered whenever
        the current reading has both forms, and hidden only where one
        does not exist: the kurtosis bars and the statistics' rows are
        flat, with nothing to put along a depth axis, and while `flat_only` is set. The scalogram
        has both forms, and in 3-D it is drawn as a surface.
        """
        # Offering the toggle wherever both forms exist is the *setting*
        # half of principle 3, and the half that survives: it is the
        # readings that stay offered whichever one is chosen (Brandon,
        # 2026-08-27).
        # by whether a reading is up, not wanted: the choice is sticky
        # across a selection it is not offered for, such as a spectrum,
        # which keeps its 3-D form
        flat = self.flat_only or any(
            reading.flat and self.showing(reading.key)
            for reading in self._READINGS)
        self.waterfall_action.setVisible(offered and not flat)
        if offered and not flat:
            self.toolbar.setVisible(True)

    def offer_frequency_axis(self, offered: bool, log: bool) -> None:
        """Show or hide the decades/hertz toggle for data drawn over
        frequency. `log` is the axis as it is now drawn: the data
        class's default until the viewer chooses
        (`core.data.frequency_axis`)."""
        self.log_frequency_action.setVisible(offered)
        self.log_frequency_action.setChecked(offered and log)
        if offered:
            self.toolbar.setVisible(True)

    def show_quantities(self, entries: Sequence[tuple[str, Any]]) -> None:
        """Fill the quantity box with a mixed object's quantity groups,
        keeping the one already chosen.

        `entries` is a list of `(label, key)`, largest group first, and
        the first is the one the 3-D view draws by default. The choice
        is kept by key, not by position, like the pair box: the list is
        rebuilt on every drawing, and a choice kept by index would jump
        to another quantity whenever picking records changed the
        counts. The box is hidden for fewer than two entries.
        """
        self._loading_quantities = True
        try:
            previous = self.quantity_box.currentData()
            self.quantity_box.clear()
            for label, key in entries:
                self.quantity_box.addItem(label, key)
            if previous is not None:
                at = next((k for k in range(self.quantity_box.count())
                           if self.quantity_box.itemData(k) == previous), -1)
                if at >= 0:
                    self.quantity_box.setCurrentIndex(at)
        finally:
            self._loading_quantities = False
        self.quantity_action.setVisible(len(entries) > 1)
        if len(entries) > 1:
            self.toolbar.setVisible(True)

    def show_pages(self, page: int, pages: int) -> None:
        """Show the page stepper at `page` of `pages`, or hide it for a
        single page.

        `page` is zero-based, as the scene counts pages. The box shows
        it one-based with the total after it, so it reads
        `[<] 2 / 3 [>]`. It is set again on every drawing, and setting
        it here does not count as turning a page.
        """
        self._loading_page = True
        try:
            self.page_box.setMaximum(max(1, pages))
            self.page_box.setSuffix(f' / {max(1, pages)}')
            self.page_box.setValue(min(page, pages - 1) + 1)
        finally:
            self._loading_page = False
        for action in (self.page_action,):
            action.setVisible(pages > 1)
        if pages > 1:
            self.toolbar.setVisible(True)

    def chosen_page(self) -> int:
        """The page shown in the 3-D view, zero-based."""
        return self.page_box.value() - 1

    def chosen_quantity(self) -> Any:
        """The quantity box's (dimension, hint) key, or None."""
        return self.quantity_box.currentData()

    def create_waterfall_plotter(self) -> Any:
        """Build the 3-D data view the first time it is needed.

        It is built late for the same macOS reason as
        `ScenePane.create_plotter`. No callback is needed here, because
        the only thing that asks for it is a toggle on this pane's own
        bar, which cannot be clicked before the pane is on screen.
        Returns the plotter, or None when it was already built.
        """
        if self.waterfall_plotter is not None:
            return None
        if self.offscreen:
            import pyvista as pv

            self.waterfall_plotter = pv.Plotter(off_screen=True)
            self._waterfall_page = QLabel('3D view disabled (offscreen mode)')
        else:
            from pyvistaqt import QtInteractor

            from ..viz import undeferred

            self.waterfall_plotter = undeferred(QtInteractor(self))
            # same refusal as the scene pane: pyvistaqt answers drops
            # with pyvista.read, which cannot read a .vdyn — the window
            # can, and gets the drop by this widget refusing it
            self.waterfall_plotter.setAcceptDrops(False)
            self.waterfall_plotter.enable_anti_aliasing('fxaa')
            self._waterfall_page = self.waterfall_plotter
        self._waterfall_page.hide()
        # beside the graphics, same stretch, so the swap keeps the size
        # — and inside the sheet's splitter, so the sheet sits beside
        # the stage exactly as it sits beside the flat plot
        self.author_split.insertWidget(1, self._waterfall_page)
        self.author_split.setStretchFactor(1, 1)
        self.author_split.setStretchFactor(2, 0)
        return self.waterfall_plotter

    def reserve_bottom(self, pixels: int) -> None:
        """Keep the bottom `pixels` of the plot area empty.

        The console tab sits over the bottom edge of the views. The
        space is a layout margin, so every plot, legend and axis ends
        above it, and the margin is kept through `clear()`, so one call
        when the window is built is enough.
        """
        # The console tab landed on the legend row and the bottom axis
        # (Brandon, 2026-09-01).
        left, top, right, _ = self.graphics.ci.layout.getContentsMargins()
        self.graphics.ci.layout.setContentsMargins(left, top, right, pixels)

    def show_waterfall(self, wanted: bool) -> None:
        """Show the 3-D view (`wanted` True) or the 2-D plot.

        Both stay in the pane and only their visibility changes, so the
        switch needs no layout work and the hidden one keeps its state.
        The 3-D view is built the first time it is wanted.
        """
        if wanted:
            self.create_waterfall_plotter()
        if self._waterfall_page is None:
            return   # never built, and not wanted: the graphics stand
        self.graphics.setVisible(not wanted)
        self._waterfall_page.setVisible(wanted)

    def component(self) -> str:
        """Which part of a complex ordinate to draw: 'magnitude',
        'real', 'imag' or 'phase'."""
        return self.component_box.currentData()

    def show_pairs(self, pairs: Sequence[tuple[str, str]]) -> None:
        """Fill the comparison box with these DOF pairs, keeping the one
        already chosen.

        The choice is kept by DOF pair, not by position: the list is
        rebuilt on every drawing, and a choice kept by index would jump
        to another channel whenever one came or went. The box is hidden
        for fewer than two pairs.

        This shows the bar itself, because `show_controls` has already
        run and set the bar from the controls it knows about. A QAction's
        `isVisible` is its own flag and stays True inside a hidden
        toolbar, so a check of that alone can report the box as shown
        while nothing is on screen.
        """
        # A drop-down put up after `show_controls` landed in a bar that
        # had already been hidden. It went unnoticed because a
        # specification used to be complex, which offered the component
        # box, which kept the bar up for its own reasons; storing PSDs
        # real took that away and the drop-down went with it.
        from ..plot import pair_label

        wanted = list(pairs)
        self.pair_action.setVisible(len(wanted) > 1)
        if len(wanted) < 2:
            return
        self.toolbar.setVisible(True)
        held = self.chosen_pair()
        self._loading_pairs = True
        try:
            self.pair_box.clear()
            for pair in wanted:
                self.pair_box.addItem(pair_label(pair), pair)
            if held in wanted:
                self.pair_box.setCurrentIndex(wanted.index(held))
        finally:
            self._loading_pairs = False

    def chosen_pair(self) -> tuple[str, str] | None:
        """The DOF pair of the comparison being shown, or None."""
        return self.pair_box.currentData()

    def select_pair_label(self, label: str) -> bool:
        """Choose the comparison whose label is `label`. Returns True if
        the choice changed.

        It matches by label rather than by DOF pair because the table
        labels its rows the way the box does, so the two agree without
        either one reading the other's data. The change does not emit
        `pair_chosen`.
        """
        from ..plot import pair_label

        for index in range(self.pair_box.count()):
            pair = self.pair_box.itemData(index)
            if pair is not None and pair_label(pair) == label:
                if index == self.pair_box.currentIndex():
                    return False
                self._loading_pairs = True
                try:
                    self.pair_box.setCurrentIndex(index)
                finally:
                    self._loading_pairs = False
                return True
        return False

    def step_pair(self, step: int) -> None:
        """Move the comparison box by `step`, stopping at either end.

        It stops rather than wrapping because going from the bottom of a
        channel list back to the top looks as if nothing happened.
        """
        if not self.pair_action.isVisible():
            return
        wanted = self.pair_box.currentIndex() + step
        if 0 <= wanted < self.pair_box.count():
            self.pair_box.setCurrentIndex(wanted)

    def sync_toolbar(self) -> None:
        """Show the bar exactly when something on it is shown.

        The bar's visibility is read from its contents, never set by a
        caller. Several methods put controls on this bar and each knows
        only its own, so any one of them setting the bar's visibility
        would hide the others' controls. An action's `isVisible` is its
        own flag and does not follow the bar's, which is what makes the
        contents readable. Separators are tidied first and then not
        counted, because a separator is visible whether or not anything
        is around it.
        """
        # What happened: the fitting screen's `show_pair_controls` hid
        # the whole bar whenever no shape set was co-selected, and once
        # Residual moved onto that bar, a fit started from the FRF's own
        # act lost it. And a bar of nothing but dividers passes a plain
        # `any(isVisible)` and stays up, which is what the first fenced
        # group did.
        tidy(self.toolbar)
        self.toolbar.setVisible(shows_anything(self.toolbar))

    def show_controls(self, *, map_wanted: bool | None,
                      diagonal: tuple[str, str, bool] | None, cmif: bool,
                      complex_data: bool, pair: bool,
                      averaging: bool = False, shocks: bool = False,
                      residual: bool = False,
                      octave: bool = False, sine: bool = False,
                      tone: bool = False, fit: bool = False,
                      transmissibility: bool = False,
                      confidence: bool = False,
                      rms_check: bool = False,
                      conditioning: bool = False,
                      realizability: bool = False) -> None:
        """Show exactly the controls this data can use, and hide the bar
        when there are none.

        `map_wanted` is True for a map, False for curves and None when
        the data has no such choice. `diagonal` is None when there is no
        diagonal to filter to, else `(label, tooltip, already_filtered)`.
        `cmif` offers the singular-value reading, `complex_data` the
        component box, and `pair` Edit Fit and Resynthesis. `averaging`
        offers the readings of a time history (averaging, filter,
        truncate, kurtosis, scalogram), `sine` adds the sine reading to
        them for a record with a sweep specification, and `shocks` offers
        the shock view and kurtosis. `fit` offers the replication fit and
        the stroke readout, for a record with acceleration in a declared
        unit to integrate. `octave` offers the
        octave bands of a plain density, `tone` the tone readout of a plain
        autospectrum, `transmissibility` the ASD ratio and H1's band
        behind an FRF computed from a time history, `confidence` the
        chi-square band of an autospectrum, `rms_check` the rms check
        of one with a time history to check against, `conditioning` the
        singular values of an FRF from several references or of a CSD,
        `realizability` the eigenvalues of a CSD's coherence matrix, and
        `residual` the fit's residual CMIF. The pair,
        3-D, frequency-axis, quantity and page controls are hidden here;
        the code that draws offers them afterwards.
        """
        # visibility is settled by `sync_toolbar` at the end, from what
        # is actually on the bar — see the note there
        for action in (self.curves_action, self.map_action):
            action.setVisible(map_wanted is not None)
        self.drive_point_action.setVisible(diagonal is not None)
        self.cmif_action.setVisible(cmif)
        # Every reading this record can carry stays offered, whichever
        # one is currently up.
        #
        # It used to be otherwise: kurtosis replaced the trace, so the
        # marks-on-the-trace buttons were hidden while it was up as
        # having nothing to mark (2026-08-24). That reasoning was about
        # *marks* and stopped being right when the four became a group
        # of readings — the way out of a reading is to pick another
        # one, and hiding the alternatives left the group collapsed to
        # whichever button had been pressed, with no way back except
        # unpicking it first (Brandon, 2026-08-27). A fenced set of
        # five that shows one is worse than the undivided row was.
        offers = {'averaging': averaging, 'sine': averaging and sine,
                  'shocks': shocks, 'fit': fit,
                  'record': averaging or shocks}
        for reading in self._READINGS:
            self._offer(reading.key, reading.panel,
                        offers[reading.offered_by])
        # only show_pairs puts it up, and only when there is a choice
        self.pair_action.setVisible(False)
        # and only offer_waterfall puts this one up, where plain curves
        # actually draw — registered here so no other drawing inherits it
        self.waterfall_action.setVisible(False)
        # likewise the frequency axis: offered by the series renderer
        # for data over frequency, and by nothing else
        self.log_frequency_action.setVisible(False)
        # the quantity box is the waterfall path's too, and the pager.
        # Hidden, never *reset*: the page is a view choice like every
        # other on this bar, and `show_pages` clamps the box through
        # setMaximum — so resetting here wiped the user's page a moment
        # before the drawing that was about to read it back, and the
        # arrows appeared to do nothing at all.
        self.quantity_action.setVisible(False)
        for action in (self.page_action,):
            action.setVisible(False)
        # the octave reading belongs to a plain density, and the tone
        # readout to an autospectrum on even lines, which has a line
        # spacing to undo; offered there, absent everywhere else
        # (principle 3)
        self._offer('octave', 'octave_panel', octave)
        self._offer('tone', 'tone_panel', tone)
        # and the transmissibility behind an FRF whose history is in the
        # project, read on its magnitude
        self._offer('transmissibility', 'transmissibility_panel',
                    transmissibility)
        # and the band to an autospectrum, the one estimate whose
        # error is chi-square
        self._offer('confidence', 'confidence_panel', confidence)
        self._offer('rms_check', 'rms_check_panel', rms_check)
        # and the two matrix checks, on the matrices they read
        self._offer('conditioning', 'conditioning_panel', conditioning)
        self._offer('realizability', 'realizability_panel', realizability)
        self.residual_action.setVisible(residual)
        # the CMIF is singular values — there is no real part to pick
        self.component_action.setVisible(
            complex_data and not (cmif and self.showing_cmif))
        self.pair_edit_action.setVisible(pair)
        self.pair_synthesis_action.setVisible(pair)
        if pair:
            self.pair_synthesis_action.setChecked(True)
        if diagonal is not None:
            # same button, same symbol, named for what the diagonal is here
            label, tooltip, filtered = diagonal
            self.drive_point_action.setText(label)
            self.drive_point_action.setToolTip(tooltip)
            # checked is derived, never stored: the button is down exactly
            # when the selection is the diagonal, so picking any other cell
            # visibly releases it
            self.drive_point_action.setChecked(filtered)
        if map_wanted is not None:
            (self.map_action if map_wanted
             else self.curves_action).setChecked(True)
        self.sync_toolbar()

    def reset_controls(self) -> None:
        """Hide the controls and acts on the bar.

        This hides everything `show_controls` sets, and the event box,
        the scaling field, the RMS and Signature toggles, the
        comparison, two-density,
        replication and shock-spectrum readings, and the acts.

        Called before every render, so a drawing that has no use for the
        bar, such as photos or a report, does not have to put away the
        controls the last one left up. A new control on the bar has to
        be hidden here too.
        """
        # The event box was added later and was not hidden here, so once
        # a transient comparison showed it, it stayed up over the record
        # on its own (where every event is on screen at once and there is
        # nothing to choose between) and over a channel table, which has
        # no events at all.
        self.show_controls(map_wanted=None, diagonal=None, cmif=False,
                           complex_data=False, pair=False, averaging=False,
                           shocks=False)
        self.event_action.setVisible(False)
        self.scaling_action.setVisible(False)
        self.rms_action.setVisible(False)
        self.offer_sweep_rate(False)
        self.offer_signature(False)
        self.show_replication_views(False)
        self.show_srs_views(False)
        # the comparison's and two densities' readings too: they were
        # left up under a hidden bar when a selection moved off them,
        # for the next bar shown by another path to carry (2026-10-13,
        # found by the docstring pass)
        self.show_comparison_views(False)
        self.show_spectra_views(False)
        # the acts too: the render that follows puts back the ones
        # the new selection can take
        self.show_acts([])

    def show_pair_controls(self, offered: bool) -> None:
        """Set the bar up for the fitting screen: hide the reading and
        view controls, and show Edit Fit and Resynthesis when
        `offered`."""
        # the readings from their table: the list written out here
        # missed the scalogram, the statistics and the pulse check
        readings = [getattr(self, f'{reading.key}_action')
                    for reading in self._READINGS]
        for action in (*readings, self.curves_action, self.map_action,
                       self.drive_point_action, self.cmif_action,
                       self.octave_action, self.tone_action,
                       self.transmissibility_action,
                       self.confidence_action, self.rms_check_action,
                       self.conditioning_action, self.realizability_action,
                       self.sweep_rate_action, self.sine_profile_action,
                       self.scaling_action, self.waterfall_action,
                       self.log_frequency_action,
                       self.quantity_action, self.page_action):
            action.setVisible(False)
        self.pair_edit_action.setVisible(offered)
        self.pair_synthesis_action.setVisible(offered)
        if offered:
            self.pair_edit_action.setChecked(True)
        # NOT `setVisible(offered)`: this used to hide the whole bar
        # whenever no pair was offered, which was right while the
        # fitting screen's bar held nothing else — and wrong the day
        # the Residual toggle moved onto it. A fit started from the
        # FRF's own calculator has no co-selected shape set, so the
        # bar vanished and took Residual with it.
        self.sync_toolbar()

    # ---- drawing ----------------------------------------------------------

    def clear(self) -> None:
        self.graphics.clear()
        if self.waterfall_plotter is not None:
            self.waterfall_plotter.clear()

    def apply_theme(self, name: str, colors: dict) -> None:
        self.theme_name = name
        self.graphics.setBackground(background_brush(colors))
        # the same theme dict carries the scene colors, so the 3-D
        # surface follows the switch without a second lookup
        if self.waterfall_plotter is not None:
            self.waterfall_plotter.set_background(
                colors['scene_background'],
                top=colors['scene_background_top'])
        # and the filter panel's own little plot, which draws itself
        # rather than going through `build_plots` and so would have
        # kept the old theme's ink
        self.filter_panel.response.apply_theme(colors)
        self.sine_panel.apply_theme(colors)


#: the panel column's starting width beside the 3-D view, in pixels
SIDE_WIDTH = 340


class ScenePane(QWidget):
    """The 3-D view: the scene, its camera, and the annotations drawn
    around it.

    What is in the scene is decided outside the pane: the geometry, the
    deflection and the picking that edits them all need the project.
    The pane holds the render window, the annotations (the labeled axes
    and the orientation triad), the bar with their toggles, and the
    settings panels shown beside the view (rigid-body modes, meshing,
    merging nodes, patterns). Whoever places the pane can add their own
    actions to the bar.
    """

    #: what was copied to the clipboard, for the status line
    copied = Signal(str)

    #: a view choice moved; draw the same scene again
    reread = Signal()

    #: the rigid-body reading was asked for, or put away
    modes_toggled = Signal(bool)

    def __init__(self, theme_name: str, offscreen: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.theme_name: str = theme_name
        #: the wish survives the selection, the data pane's rule: pick
        #: another geometry and the reading comes back up
        self.modes_wanted: bool = False
        # the orientation triad is useful at a glance; the labeled box
        # around the geometry is clutter until asked for
        self.bounds_visible: bool = False
        self.orientation_visible: bool = True
        self.axis_unit: str = ''
        #: the view Reset View returns to: the shown geometry's own
        #: (`Geometry.opening_view`), None while no geometry is shown
        self.home_view: Any = None
        self._home_owner: Any = None      # the geometry it belongs to
        self._framed_owner: Any = None    # the last one turned to it

        self.toolbar: QToolBar = self._build_toolbar()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        if offscreen:
            import pyvista as pv
            self.plotter: Any = pv.Plotter(off_screen=True)
            self._page = QLabel('3D view disabled (offscreen mode)')
        else:
            # On macOS, building the VTK widget before the window is mapped
            # leaves the whole window 0x0 and never shown; it is created in
            # create_plotter() instead, once the pane is on screen.
            self.plotter = None
            self._page = QLabel('Loading 3D view...')
        self._creating_plotter: bool = False
        # the settings panels sit beside the view rather than under
        # it, the data pane's own arrangement: the rigid-body table is
        # read against the model it is moving, and both want the height
        self.modes_panel: ModesPanel = ModesPanel()
        self.modes_panel.hide()
        #: Add Block's and Add Plane's fields, beside the view rather than in
        #: a window of their own (2026-10-02)
        self.mesh_panel: MeshPanel = MeshPanel()
        self.mesh_panel.hide()
        #: Merge Coincident Nodes' tolerance and its Merge (2026-10-11)
        self.merge_panel: MergePanel = MergePanel()
        self.merge_panel.hide()
        self.mesh_solids_panel: MeshSolidsPanel = MeshSolidsPanel()
        self.mesh_solids_panel.hide()
        self.mate_panel: MatePanel = MatePanel()
        self.mate_panel.hide()
        self.pattern_panel: PatternPanel = PatternPanel()
        self.pattern_panel.hide()
        #: Plan Instrumentation's modes, counts and reading (2026-10-21)
        self.instrumentation_panel: InstrumentationPanel = InstrumentationPanel()
        self.instrumentation_panel.hide()
        # the panels in a column beside the view, the two in a splitter
        # the user can drag (Brandon, 2026-10-20: the pane's width was
        # the panels' own and could not be changed); the column shows
        # while any panel does and keeps the width it was dragged to
        self.side_panels: tuple = (self.modes_panel, self.mesh_panel, self.merge_panel,
                                   self.mesh_solids_panel, self.mate_panel,
                                   self.pattern_panel, self.instrumentation_panel)
        self.side: QWidget = QWidget()
        column = QVBoxLayout(self.side)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        for panel in self.side_panels:
            # `panel_grid` fixes a panel's width for the data pane's
            # settings column; beside the view the panel fills the
            # column the user drags, so its contents grow with it
            # (Brandon, 2026-10-20: the column widened and the panel
            # did not)
            panel.setMinimumWidth(PANEL_WIDTH)
            panel.setMaximumWidth(16777215)      # Qt's QWIDGETSIZE_MAX
            panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            column.addWidget(panel)
            panel.installEventFilter(self)
        column.addStretch(1)
        self.side.hide()
        self.view_split: QSplitter = QSplitter(Qt.Orientation.Horizontal)
        self.view_split.setChildrenCollapsible(False)
        self.view_split.addWidget(self._page)
        self.view_split.addWidget(self.side)
        self.view_split.setStretchFactor(0, 1)
        self.view_split.setStretchFactor(1, 0)
        self._side_width: int = SIDE_WIDTH
        layout.addWidget(self.view_split, 1)

    def eventFilter(self, watched, event) -> bool:
        """The panel column follows its panels: shown while any is,
        at the width it was last dragged to, and hidden with the last."""
        from PySide6.QtCore import QEvent

        if watched in self.side_panels and event.type() in (QEvent.Type.Show, QEvent.Type.Hide,
                                                             QEvent.Type.ShowToParent,
                                                             QEvent.Type.HideToParent):
            wanted = any(panel.isVisibleTo(self.side) for panel in self.side_panels)
            if wanted and not self.side.isVisibleTo(self):
                self.side.show()
                total = sum(self.view_split.sizes()) or self.width()
                self.view_split.setSizes([max(total - self._side_width, 1), self._side_width])
            elif not wanted and self.side.isVisibleTo(self):
                shown = self.view_split.sizes()[1]
                if shown > 0:
                    self._side_width = shown
                self.side.hide()
        return super().eventFilter(watched, event)

    def _build_toolbar(self) -> QToolBar:
        toolbar = QToolBar('3D view')
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))
        # there was no way back to where a geometry opened: a turned
        # model stayed turned until another object was shown (2026-09-27)
        self.reset_view_action: QAction = QAction(
            control_icon('reset_view'), 'Reset View', self)
        self.reset_view_action.setToolTip(
            "Turn the view back to the geometry's default view and fit it")
        self.reset_view_action.triggered.connect(self.reset_view)
        self.reset_view_action.setVisible(False)
        toolbar.addAction(self.reset_view_action)

        self.bounds_action: QAction = QAction(child_icon('bounds', 'Test'),
                                     'Plot axes', self)
        self.bounds_action.setCheckable(True)
        self.bounds_action.setChecked(False)
        self.bounds_action.setToolTip(
            'Show the labeled axes drawn around the geometry')
        self.bounds_action.toggled.connect(self.set_bounds_visible)
        toolbar.addAction(self.bounds_action)

        self.orientation_action: QAction = QAction(
            child_icon('coordinate_systems', 'Test'), 'Orientation marker',
            self)
        self.orientation_action.setCheckable(True)
        self.orientation_action.setChecked(True)
        self.orientation_action.setToolTip(
            'Show the orientation triad in the corner')
        self.orientation_action.toggled.connect(self.set_orientation_visible)
        toolbar.addAction(self.orientation_action)

        # a reading of the geometry, fenced from the view choices the
        # way every group on the data pane's bar is; shown only when
        # one whole geometry is selected (`offer_modes`)
        self._modes_fence = toolbar.addSeparator()
        # one reading for both kinds of modes (Brandon, 2026-10-22): it
        # was the rigid-body preview alone, with Solve Modes an act of
        # its own beside it
        self.modes_action: QAction = QAction(control_icon('solve_modes'),
                                             'Modes', self)
        self.modes_action.setCheckable(True)
        self.modes_action.setToolTip(
            "Make this geometry's mode shapes: kinematic, the six "
            'rigid-body shapes about a point, previewed as they are set; '
            'or finite element, once its element groups have properties')
        self.modes_action.triggered.connect(self._choose_modes)
        toolbar.addAction(self.modes_action)
        self.modes_action.setVisible(False)
        self._modes_fence.setVisible(False)
        return toolbar

    def copy_view(self) -> bool:
        """Copy the 3-D view, as it is now, to the clipboard as an image.

        Emits `copied` with a line for the status bar, and returns False
        when the view is not built yet."""
        from PySide6.QtWidgets import QApplication

        if self.plotter is None:
            self.copied.emit('The 3-D view is not built yet')
            return False
        image = image_of(self.plotter.screenshot(return_img=True))
        QApplication.clipboard().setImage(image)
        self.copied.emit(f'3-D view copied to the clipboard '
                         f'({image.width()} × {image.height()})')
        return True

    def show_acts(self, acts: Sequence[tuple]) -> None:
        """Put the acts the selection can take on this bar
        (see `offer_acts`)."""
        offer_acts(self.toolbar, self.__dict__.setdefault('_acts', {}), acts)

    # ---- the modes reading ------------------------------------------------

    def _choose_modes(self, checked: bool) -> None:
        """The Modes button: remember it, then say so."""
        self.modes_wanted = bool(checked)
        if not checked:
            self.modes_panel.hide()
        self.modes_toggled.emit(bool(checked))

    def offer_modes(self, offered: bool) -> None:
        """Show or hide the Modes toggle. The window offers it
        only when one whole geometry is selected with nothing drawn on
        it, and it comes back checked if it was checked last time."""
        offered = bool(offered)
        self.modes_action.setVisible(offered)
        self._modes_fence.setVisible(offered)
        self.modes_action.setChecked(offered and self.modes_wanted)
        if not (offered and self.modes_wanted):
            self.modes_panel.hide()

    @property
    def showing_modes(self) -> bool:
        """Whether the Modes toggle is offered and checked."""
        return self.modes_action.isVisible() and self.modes_action.isChecked()

    # ---- the render window ------------------------------------------------

    def create_plotter(self) -> Any:
        """Build the embedded VTK view, once the pane is on screen.

        On macOS, building it before the window is shown leaves the
        window with no size. Returns the plotter, or None if it was
        already built (or is being built), so the caller knows whether
        it has a new view to draw into.
        """
        if self.plotter is not None or self._creating_plotter:
            return None
        from pyvistaqt import QtInteractor

        from ..viz import undeferred

        placeholder = self._page
        # Re-entrant on Windows: making the VTK view's native window
        # re-shows the main window, whose showEvent lands here again
        # while the first QtInteractor is half-built — and built a
        # second, then read `renderers` off the first before it had
        # any. The first Windows launch of 0.1.0a1 died in that loop
        # (the release smoke test, 2026-09-14). One construction at a
        # time; the nested call answers None and the outer one draws.
        self._creating_plotter = True
        try:
            self.plotter = undeferred(QtInteractor(self))
        finally:
            self._creating_plotter = False
        # pyvistaqt takes drops and answers them with `pyvista.read`, so a
        # .vdyn dropped on the 3D view — aimed at the window, landing
        # here — comes back as "not able to be automatically read by
        # pyvista" and is never imported. Qt routes drags only to widgets
        # that say they take them, so refusing here sends the drop up to
        # the window, which knows what a .vdyn is. Adding a mesh to a
        # scene by dropping a file on it was never a thing this offers.
        self.plotter.setAcceptDrops(False)
        # cheap post-process pass: smooths the model's edges without the
        # supersampling alternative, which shrinks points and lines because
        # their sizes are in pixels
        self.plotter.enable_anti_aliasing('fxaa')
        self._page = self.plotter
        self.apply_background()
        self.view_split.replaceWidget(0, self.plotter)
        self.view_split.setStretchFactor(0, 1)
        placeholder.deleteLater()
        return self.plotter

    def set_home_view(self, view: Any, owner: Any = None) -> None:
        """Set the view Reset View returns to, and `owner`, the geometry
        it belongs to. None hides Reset View, because a scene with no
        geometry in it has no view of its own to return to."""
        self.home_view = view
        self._home_owner = owner
        self.reset_view_action.setVisible(view is not None)

    def frame(self) -> None:
        """Fit the camera to new content.

        When the home view's geometry is new to the pane, the camera
        turns to the home view. Otherwise it is only refitted, so a turn
        the user made is kept when stepping from a geometry to the
        shapes drawn on it."""
        if self.plotter is None:
            return
        if self.home_view is not None and \
                self._home_owner is not self._framed_owner:
            self._framed_owner = self._home_owner
            self.reset_view()
        else:
            self.plotter.reset_camera()

    def reset_view(self) -> None:
        """Turn the camera to the home view and fit the scene."""
        if self.plotter is None or self.home_view is None:
            return
        from ..viz.geometry import place_view

        place_view(self.plotter, self.home_view)

    def clear(self) -> None:
        self.set_home_view(None)
        if self.plotter is not None:
            self.plotter.clear()
            self.apply_background()
            self.plotter.render()

    # ---- how the scene is dressed ----------------------------------------

    def set_bounds_visible(self, visible: bool) -> None:
        """Show or hide the labeled axes drawn around the geometry."""
        self.bounds_visible = bool(visible)
        self.apply_annotations()

    def set_orientation_visible(self, visible: bool) -> None:
        """Show or hide the orientation triad in the corner."""
        self.orientation_visible = bool(visible)
        self.apply_annotations()

    def apply_annotations(self) -> None:
        """Draw the annotations again in place, so the camera keeps its
        position."""
        if self.plotter is None:
            return
        annotate_scene(self.plotter, self.axis_unit,
                       resolve_theme(self.theme_name), self.bounds_visible,
                       self.orientation_visible)
        self.plotter.render()

    def apply_background(self) -> None:
        """Set the view's background to the theme's, so an empty 3-D
        view matches the theme instead of VTK's white default."""
        if self.plotter is not None:
            colors = resolve_theme(self.theme_name)
            self.plotter.set_background(colors['scene_background'],
                                        top=colors['scene_background_top'])

    def apply_theme(self, name: str) -> None:
        self.theme_name = name
        self.apply_background()
