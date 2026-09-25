"""Self-contained view panes.

A pane owns a view, the bar of controls that belongs to it, and the
state those controls carry — and nothing else. It never reaches for the
project, the tree or the selection: it reports that a choice moved and
lets whoever placed it decide what that means. That is what lets the
same pane sit in the main window, in a dock, or in a window of its own
opened from a script.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QImage,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..plot import background_brush
from ..theme import theme as resolve_theme
from ..viz.geometry import annotate_scene
from .author_panel import AuthorPanel
from .averaging_panel import AveragingPanel
from .editors import SpinBox
from .filter_panel import FilterPanel
from .icons import child_icon, control_icon
from .octave_panel import OctavePanel
from .rigid_panel import RigidBodyPanel
from .shock_panel import ShockPanel
from .toolbars import fence, shows_anything, tidy
from .truncate_panel import TruncatePanel
from .wavelet_panel import WaveletPanel


def copy_act(pane: Any, what: str) -> tuple:
    """The copy act a pane's bar carries beside the selection's acts:
    what the pane shows, onto the clipboard as an image (Brandon,
    2026-09-24: every plot, table and figure should copy like a chat's
    code blocks do). An act rather than a standing control, because the
    bar shows exactly when something on it applies, and a copy applies
    exactly when something is drawn. One glyph and one tooltip shape on
    both bars, so it reads as the same act."""
    return ('copy', 'Copy', 'copy', pane.copy_view,
            f'Copy the {what} to the clipboard as an image')


def image_of(pixels: Any) -> QImage:
    """A rendered (rows, columns, 3 or 4) array of bytes as a QImage
    that owns its memory — the array a screenshot hands back is freed
    with it, and a QImage over it would read garbage."""
    import numpy as np

    pixels = np.ascontiguousarray(pixels)
    rows, columns, depth = pixels.shape
    layout = (QImage.Format.Format_RGBA8888 if depth == 4
              else QImage.Format.Format_RGB888)
    return QImage(pixels.data, columns, rows, depth * columns, layout).copy()


def offer_acts(toolbar: Any, state: dict, acts: Sequence[tuple]) -> None:
    """Put the acts a selection can take on a bar, in their own fenced
    group at its end — an icon each, like every other button on a
    bar, the verb and its meaning in the tooltip (Brandon, 2026-09-12:
    "none should [carry text] and the user should rely on tool tips to
    learn icons they don't recognize"; this reverses the labeled
    buttons of 2026-09-04). Every act on the bar, none behind a menu.

    `acts` is [(verb, label, icon name, callback, tooltip)], in the
    order the bar shows them. Rebuilt on every call: a toolbar cannot
    reorder its actions, and the order is part of the reading — the
    same verb in the same place whatever else the selection allows.
    """
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


class DataPane(QWidget):
    """The 2-D data view: a plot under a bar saying how to read it.

    The bar speaks three vocabularies at once — how coherence reads,
    which records an FRF is filtered to, and how an FRF beside a shape
    set is used — and shows only what the data in front of it can
    actually use, because a control that is not a live option is absent
    rather than grayed.

    The pane keeps the choice and announces it. What a choice *means*
    stays outside: filtering to the diagonal moves the tree selection,
    and choosing Edit Fit opens a fitting session, neither of which is
    a plot's business.
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
    #: the truncate view was asked for (True) or put away (False)
    truncate_toggled = Signal(bool)
    #: the octave-band view was asked for (True) or put away (False)
    octave_toggled = Signal(bool)
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
        self.truncate_wanted: bool = False
        self.octave_wanted: bool = False
        #: the kurtosis reading, sticky like every other view choice
        self.kurtosis_wanted: bool = False
        #: and the scalogram, the same way
        self.wavelet_wanted: bool = False
        #: what it is computed with, kept across redraws so a tuned
        #: range survives clicking away and back
        self.wavelet_settings: dict[str, Any] | None = None
        #: where its band has been dragged to, if it has
        self.kurtosis_bounds: tuple[float, float] | None = None
        #: 'curves', 'error' or 'lines' — sticky, like the other view
        #: choices, so it survives a redraw
        self.comparison_view: str = 'curves'
        #: where the bar charts' thresholds have been dragged to
        self.error_bounds: tuple[float, float] | None = None
        self.lines_bound: float | None = None
        #: 'overlay' or 'waveform' — the transient's own
        #: readings, kept apart from the random ones above because they
        #: are different questions and a single sticky choice shared
        #: between them would carry a random answer into a transient
        self.replication_view: str = 'overlay'
        #: 'curves' or 'error' — the same for a pair of shock spectra
        self.srs_view: str = 'curves'
        #: 'overlay' or 'ratio' — how two selected densities read
        #: together; sticky like every view choice
        self.spectra_view: str = 'overlay'
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
        self.truncate_panel: TruncatePanel = TruncatePanel()
        self.truncate_panel.hide()
        self.octave_panel: OctavePanel = OctavePanel()
        self.octave_panel.hide()
        self.wavelet_panel: WaveletPanel = WaveletPanel()
        self.wavelet_panel.hide()
        self.wavelet_panel.changed.connect(self._wavelet_edited)
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
        plot_row.addWidget(self.truncate_panel)
        plot_row.addWidget(self.octave_panel)
        plot_row.addWidget(self.wavelet_panel)
        self._plot_row = plot_row

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
        # three readings of one comparison: the spectra themselves, the
        # level each channel came out at, and how much of each channel's
        # band fell outside. One at a time, because they answer
        # different questions and stacking them answers none.
        self.comparison_group: QActionGroup = QActionGroup(self)
        self.comparison_actions: dict[str, QAction] = {}
        for key, icon, text, tip in (
                ('curves', 'curves', 'Spectra',
                 'The control spectra against their specification'),
                ('error', 'bars', 'RMS Error',
                 'How far each channel sits from the level asked for'),
                ('lines', 'percent_bars', 'Lines Out',
                 'How much of each channel fell outside its abort limits')):
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
        # two densities selected together: drawn over each other, or
        # divided — the signal-to-noise reading, in dB (Brandon,
        # 2026-08-25). One at a time, like the comparison's readings.
        self.spectra_group: QActionGroup = QActionGroup(self)
        self.spectra_actions: dict[str, QAction] = {}
        for key, icon, text, tip in (
                ('overlay', 'overlay', 'Overlaid',
                 'The two spectra over each other'),
                ('ratio', 'ratio', 'Ratio (dB)',
                 ('The louder spectrum over the quieter, in decibels '
                  '— the signal-to-noise reading'))):
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
        # two readings, and only two. The waveform error is the one
        # thing the time data alone can answer; a level error wants two
        # PSDs and an SRS deviation wants two spectra, and those are
        # reached by computing them and selecting the pair — where the
        # specification's own PSD is a Specification and its SRS a
        # ShockSpecification, so the app still knows which is which.
        # Offering them here as well would be two ways to the same
        # number, computed differently, agreeing only by luck.
        self.replication_group: QActionGroup = QActionGroup(self)
        self.replication_actions: dict[str, QAction] = {}
        for key, icon, text, tip in (
                ('overlay', 'overlay', 'Waveforms',
                 'The measured event against the waveform asked for'),
                ('waveform', 'bars', 'Waveform Error',
                 ('How far each channel is from the target waveform, '
                  'as a share of the target')),
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
        # and how far each channel sits from what it had to meet
        self.srs_group: QActionGroup = QActionGroup(self)
        self.srs_actions: dict[str, QAction] = {}
        for key, icon, text, tip in (
                ('curves', 'curves', 'Spectra',
                 'The measured shock spectra against the one required'),
                ('error', 'bars', 'SRS Error',
                 ('How far each channel sits from the spectrum it had '
                  'to meet, RMS across the band in dB'))):
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
        for action in (self.averaging_action, self.filter_action,
                       self.truncate_action, self.kurtosis_action,
                       self.shocks_action, self.wavelet_action):
            self.reading_group.addAction(action)
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
        self.octave_action.triggered.connect(self._choose_octave)
        toolbar.addAction(self.octave_action)
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
            [self.octave_action],
            self.comparison_group.actions(),
            self.spectra_group.actions(),
            self.replication_group.actions(),
            self.srs_group.actions(),
            self.reading_group.actions(),
            self.pair_group.actions(),
        ])
        return toolbar

    def copy_view(self) -> bool:
        """What the pane shows, onto the clipboard as an image: the 3-D
        stage when it is up, the flat plot otherwise. The plot is drawn
        by the same exporter the report figures and the headless plots
        use, on the theme's own background — the exporter paints one of
        its own and defaults to black — at the screen's pixel density, so
        a retina display copies a retina image."""
        from PySide6.QtWidgets import QApplication

        if (self.waterfall_plotter is not None
                and not self.graphics.isVisibleTo(self)):
            image = image_of(self.waterfall_plotter.screenshot(return_img=True))
            what = 'stage'
        else:
            from pyqtgraph.exporters import ImageExporter

            exporter = ImageExporter(self.graphics.scene())
            colors = resolve_theme(self.theme_name)
            exporter.parameters()['background'] = QColor(
                colors['plot_background'])
            exporter.parameters()['width'] = int(
                self.graphics.width() * self.graphics.devicePixelRatioF())
            image = exporter.export(toBytes=True)
            what = 'plot'
        if image is None or image.isNull():
            self.copied.emit('Nothing drawn to copy')
            return False
        QApplication.clipboard().setImage(image)
        self.copied.emit(f'{what.capitalize()} copied to the clipboard '
                         f'({image.width()} × {image.height()})')
        return True

    #: the readings, as (action attribute, wanted flag, panel attribute).
    #: One list, so that adding a fifth reading is one entry rather than
    #: an edit to every other reading's handler.
    _READINGS = (('averaging_action', 'averaging_wanted', 'averaging_panel'),
                 ('filter_action', 'filter_wanted', 'filter_panel'),
                 ('truncate_action', 'truncate_wanted', 'truncate_panel'),
                 ('kurtosis_action', 'kurtosis_wanted', None),
                 ('shocks_action', 'shocks_wanted', 'shock_panel'),
                 ('wavelet_action', 'wavelet_wanted', 'wavelet_panel'))

    def show_acts(self, acts: Sequence[tuple]) -> None:
        """The acts the selection can take, on this bar (`offer_acts`)."""
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
        for action_name, wanted, panel_name in self._READINGS:
            action = getattr(self, action_name)
            setattr(self, wanted, action.isChecked())
            if not action.isChecked() and panel_name is not None:
                getattr(self, panel_name).hide()

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

    def _choose_truncate(self, checked: bool) -> None:
        """The truncate button: remember it, then say so."""
        self._readings_settled()
        self.truncate_toggled.emit(checked)

    def _choose_octave(self, checked: bool) -> None:
        """The octave button: remember it, then say so."""
        self.octave_wanted = bool(checked)
        if not checked:
            self.octave_panel.hide()
        self.octave_toggled.emit(checked)

    def _choose_reading(self, checked: bool) -> None:
        """The kurtosis or the wavelet button: a different *reading* of
        the record rather than a mark on it, so it replaces the trace
        outright — and the mark-the-trace toggles stand down, having
        nothing to mark."""
        del checked                      # the buttons are the state now
        self._readings_settled()
        self.reread.emit()

    def _wavelet_edited(self, settings) -> None:
        """A scalogram parameter moved: keep it, then draw again.

        Kept on the pane rather than in the panel, so a tuned range
        survives clicking away to another object and back — the same
        stickiness every other view choice on this bar has.
        """
        self.wavelet_settings = dict(settings)
        self.reread.emit()

    @property
    def showing_wavelet(self) -> bool:
        """Whether the scalogram is up — asked by whoever draws."""
        return (self.wavelet_action.isVisible()
                and self.wavelet_action.isChecked())

    @property
    def showing_kurtosis(self) -> bool:
        """Whether the bar reading is up — asked by whoever draws."""
        return (self.kurtosis_action.isVisible()
                and self.kurtosis_action.isChecked())

    @property
    def rms_wanted(self) -> bool:
        """Whether the RMS reading of a specification is asked for —
        the checked state alone, so a render can decide its panes
        before the bar is offered."""
        return self.rms_action.isChecked()

    def offer_rms(self, offered: bool) -> None:
        """Show the RMS toggle for a specification on its own."""
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
        """Offer the overlay/ratio pair, or take it away."""
        for action in self.spectra_actions.values():
            action.setVisible(offered)
        if offered:
            self.toolbar.setVisible(True)
            self.spectra_actions[self.spectra_view].setChecked(True)

    def show_comparison_views(self, offered: bool) -> None:
        """Offer the three readings, or none of them.

        Only a specification with a measurement against it has three
        readings; anything else has one, and a chooser with one choice
        is a question with one answer.
        """
        for action in self.comparison_actions.values():
            action.setVisible(offered)
        if offered:
            self.toolbar.setVisible(True)
            self.comparison_actions[self.comparison_view].setChecked(True)

    def show_scaling(self, text: str | None) -> None:
        """Show the comparison scaling, or take it away with None.

        The field is restated on every drawing, so setting the text
        must not read as the user typing it — hence the guard, the same
        one the pair and event boxes need for the same reason.
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

    def show_srs_views(self, offered: bool) -> None:
        """Offer the two readings of a shock-spectrum comparison."""
        for action in self.srs_actions.values():
            action.setVisible(offered)
        if offered:
            for action in self.comparison_actions.values():
                action.setVisible(False)
            self.toolbar.setVisible(True)
            self.srs_actions[self.srs_view].setChecked(True)

    def _choose_replication(self, which: str) -> None:
        """Which reading of a transient replication the plot shows."""
        self.replication_view = which
        self.replication_chosen.emit(which)

    def show_replication_views(self, offered: bool) -> None:
        """Offer the two readings of a replication, or neither.

        Only a transient specification with a record against it has
        them. They hide the random three while they are up: both sets
        are readings of "how did this compare", and a bar offering
        seven of those invites reading a waveform error against an
        abort band that does not exist.
        """
        for action in self.replication_actions.values():
            action.setVisible(offered)
        if offered:
            for action in self.comparison_actions.values():
                action.setVisible(False)
            self.toolbar.setVisible(True)
            self.replication_actions[self.replication_view].setChecked(True)

    def show_events(self, labels: Sequence[tuple[str, int]]) -> None:
        """Offer these repeats, keeping the one already chosen.

        Sticky by index, unlike the channel box beside it, because a
        repeat *is* its index — the third playing of the waveform is
        the third whatever else the record holds — where a channel is a
        DOF that may arrive or leave.

        No repeat is marked out. Which one is the bad one depends on
        which reading you care about and on what the article is for,
        and naming one here would be putting a judgment in a list of
        facts — the numbers beside the plot are what to decide with.
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
        """The repeat being looked at, or None."""
        return self.event_box.currentData()

    def step_event(self, step: int) -> None:
        """Move the event box by `step`, stopping at either end.

        Stopping rather than wrapping: the repeats are in time order,
        and running off the last one back to the first would read as
        having gone forwards.
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
        """Back to letting the data choose. Called when the selection
        changes, so an override applies to the thing being looked at and
        not for ever."""
        self.plot_mode = None

    # ---- what the pane is showing ----------------------------------------

    @property
    def showing_averaging(self) -> bool:
        """Is the averaging view asked for? Only meaningful for a time
        history, which is why the caller checks that first."""
        return self.averaging_action.isChecked()

    @property
    def showing_filter(self) -> bool:
        """Is the filter view asked for? Only meaningful for a time
        history, which is why the caller checks that first."""
        return self.filter_action.isChecked()

    @property
    def showing_octave(self) -> bool:
        """Is the octave-band view asked for? Only meaningful for a
        plain PSD or CPSD, which is why the caller checks that
        first."""
        return self.octave_action.isChecked()

    @property
    def showing_truncate(self) -> bool:
        """Is the truncate view asked for? Only meaningful for a time
        history, which is why the caller checks that first."""
        return self.truncate_action.isChecked()

    @property
    def showing_shocks(self) -> bool:
        """Is the shock view asked for? Only meaningful for a time
        history, which is why the caller checks that first."""
        return self.shocks_action.isChecked()

    @property
    def showing_cmif(self) -> bool:
        """Is the singular-value reading asked for? Only meaningful for
        FRFs, which is why the caller checks that first."""
        return self.cmif_action.isChecked()

    @property
    def showing_waterfall(self) -> bool:
        """Is the 3-D reading asked for? Only meaningful where the
        window offered it, which is why the caller checks that first.

        The filter view used to outrank this and force the drawing
        flat — my judgment that two overlaid ribbons would occlude
        each other, made without drawing them. Wrong twice over
        (Brandon, 2026-08-25): the stage already draws paired data at
        one station, and a view choice that silently overrides another
        view choice is exactly what this interface does not do. The
        preview is stage geometry now, so both readings stand.

        Except while the specification sheet is open: its handles
        live on the flat plot, so `flat_only` holds the drawing flat
        whatever the toggle says (Brandon, 2026-09-06).
        """
        return self.waterfall_action.isChecked() and not self.flat_only

    #: the flat plot only, whatever the 3D toggle says — set by the
    #: window while the specification sheet is open, since the sheet's
    #: handles live on the flat plot (Brandon, 2026-09-06)
    flat_only: bool = False

    def offer_waterfall(self, offered: bool) -> None:
        """Show the 2D/3D toggle, raised by the plain-curves path
        itself, like `show_pairs`: `show_controls` runs before the
        window knows whether this drawing ends as plain curves.

        Offered whenever the current reading has both forms, and
        withheld only where one of them does not exist — the kurtosis
        bars are a flat picture and a depth axis would have nothing to
        put along it. That is the *setting* half of principle 3, and
        the half that survives: it is the readings that stay offered
        whichever one is chosen (Brandon, 2026-08-27).

        The scalogram has both, and opens in 3-D: it is already a
        function of two variables, so a surface is its natural form.
        """
        flat = self.kurtosis_wanted or self.flat_only
        self.waterfall_action.setVisible(offered and not flat)
        if offered and not flat:
            self.toolbar.setVisible(True)

    def offer_frequency_axis(self, offered: bool, log: bool) -> None:
        """Show the decades/hertz toggle for data drawn over frequency,
        reading `log`, the axis as it stands — the class's convention
        until the viewer chooses (`core.data.frequency_axis`)."""
        self.log_frequency_action.setVisible(offered)
        self.log_frequency_action.setChecked(offered and log)
        if offered:
            self.toolbar.setVisible(True)

    def show_quantities(self, entries: Sequence[tuple[str, Any]]) -> None:
        """Offer a mixed object's quantity groups, keeping the choice.

        `entries` is [(label, key)], largest group first — index 0 is
        the default the waterfall draws unasked. Sticky by key, not by
        position, like the pair box: the list is rebuilt on every
        drawing, and a choice that moved with the index would jump to
        another quantity whenever a record pick changed the counts.
        Hidden below two entries — one quantity is not a choice.
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
        """Offer the page stepper, or put it away for a single page.

        `page` is zero-based, as the scene counts them; the box shows
        one-based with the total as its suffix, so it reads
        `[<] 2 / 3 [>]`. Restated on every drawing, so the guard keeps
        that from reading as the user turning a page.
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
        """The page on the stage, zero-based."""
        return self.page_box.value() - 1

    def chosen_quantity(self) -> Any:
        """The quantity box's (dimension, hint) key, or None."""
        return self.quantity_box.currentData()

    def create_waterfall_plotter(self) -> Any:
        """Build the 3-D data surface, on first use.

        The same deferral as `ScenePane.create_plotter` and for the
        same macOS reason — and safe here without a callback, because
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

        The console tab floats over the views' bottom edge rather than
        claiming a strip, and it landed on the legend row and the
        bottom axis (Brandon, 2026-09-01). A layout margin is the
        clearance: every plot, legend and axis ends above it, and the
        margin survives `clear()`, so one call at construction holds.
        """
        left, top, right, _ = self.graphics.ci.layout.getContentsMargins()
        self.graphics.ci.layout.setContentsMargins(left, top, right, pixels)

    def show_waterfall(self, wanted: bool) -> None:
        """Swap the plot surface: the 2-D graphics or the 3-D view.

        Both stay children of the pane; only visibility moves, so the
        toggle costs no layout work and the hidden one keeps its state.
        """
        if wanted:
            self.create_waterfall_plotter()
        if self._waterfall_page is None:
            return   # never built, and not wanted: the graphics stand
        self.graphics.setVisible(not wanted)
        self._waterfall_page.setVisible(wanted)

    def component(self) -> str:
        """Which part of a complex ordinate to draw."""
        return self.component_box.currentData()

    def show_pairs(self, pairs: Sequence[tuple[str, str]]) -> None:
        """Offer these comparisons, keeping the one already chosen.

        Sticky by DOF and not by position: the list is rebuilt on every
        drawing, and a selection that moved with the index would jump to
        another channel whenever one arrived or left.

        This raises the bar itself, and has to. `show_controls` decides
        whether there is a bar from the controls *it* knows about, and
        it runs before this does — so a drop-down put up afterwards
        landed in a bar that had already been hidden. It went unnoticed
        because a specification used to be complex, which offered the
        component box, which kept the bar up for its own reasons;
        storing PSDs real took that away and the drop-down went with it.

        A QAction's `isVisible` is its own flag and stays true inside a
        hidden toolbar, so anything checking that alone will agree the
        box is up while the screen shows nothing.
        """
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
        """The comparison being looked at, or None."""
        return self.pair_box.currentData()

    def select_pair_label(self, label: str) -> bool:
        """Show the comparison this label names. True if it moved.

        By label rather than by DOF pair, because the table names its
        rows the way the box does and matching the text is what keeps
        the two agreeing without either learning the other's model.
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
        """Move to the next comparison, or the previous one.

        Stops at the ends rather than wrapping: running off the bottom
        of a channel list and arriving back at the top reads as nothing
        having happened.
        """
        if not self.pair_action.isVisible():
            return
        wanted = self.pair_box.currentIndex() + step
        if 0 <= wanted < self.pair_box.count():
            self.pair_box.setCurrentIndex(wanted)

    def sync_toolbar(self) -> None:
        """The bar shows exactly when something on it does.

        Derived, never decided by a caller. Five methods put controls
        on this bar and each knows only its own slice, so any one of
        them setting the bar's visibility from its own slice hides
        another's controls — which is exactly what happened: the
        fitting screen's `show_pair_controls` hid the whole bar
        whenever no shape set was co-selected, and once Residual moved
        onto that bar, a fit started from the FRF's own act lost
        it. An action's `isVisible` is its own property and does not
        follow the bar's, so the bar can be read back off its contents.

        The dividers settle first and are then discounted: a separator
        is visible whether or not anything is around it, so a bar of
        nothing but dividers passes a plain `any(isVisible)` and stays
        up — which is what the first fenced group did.
        """
        tidy(self.toolbar)
        self.toolbar.setVisible(shows_anything(self.toolbar))

    def show_controls(self, *, map_wanted: bool | None,
                      diagonal: tuple[str, str, bool] | None, cmif: bool,
                      complex_data: bool, pair: bool,
                      averaging: bool = False, shocks: bool = False,
                      residual: bool = False,
                      octave: bool = False) -> None:
        """Show exactly the controls this data can use, and hide the bar
        when that is none of them.

        `map_wanted` is True for a map, False for curves and None when
        the data has no such choice; `diagonal` is None when there is no
        diagonal to filter to, else `(label, tooltip, already_filtered)`;
        `cmif` offers the singular-value reading; `complex_data` offers
        the component box; `pair` offers Edit Fit / Resynthesis;
        `averaging` offers the frames a time history would be cut into.
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
        reading = averaging or shocks
        instead = reading and (self.kurtosis_wanted or self.wavelet_wanted)
        self.averaging_action.setVisible(averaging)
        self.filter_action.setVisible(averaging)
        self.filter_action.setChecked(averaging and self.filter_wanted)
        if not averaging or instead:
            self.filter_panel.hide()
        # the cut is offered wherever the averaging is: both are
        # readings of a record's stretch of time
        self.truncate_action.setVisible(averaging)
        self.truncate_action.setChecked(averaging and self.truncate_wanted)
        if not averaging or instead:
            self.truncate_panel.hide()
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
        self.averaging_action.setChecked(
            averaging and not instead and self.averaging_wanted)
        if not averaging or instead:
            # the panel belongs to the button: nothing else may show it,
            # so a selection with no averaging — or one reading its
            # kurtosis instead — cannot leave one up
            self.averaging_panel.hide()
        # offered wherever the averaging is: both are readings of a
        # record, and a record is what has channels to be Gaussian
        self.kurtosis_action.setVisible(reading)
        self.kurtosis_action.setChecked(reading and self.kurtosis_wanted)
        # the scalogram is offered wherever the averaging is, and for
        # the same reason: both are readings of a record
        self.wavelet_action.setVisible(averaging)
        self.wavelet_action.setChecked(averaging and self.wavelet_wanted)
        self.shocks_action.setVisible(shocks)
        # the octave reading belongs to a plain density; offered there,
        # absent everywhere else (principle 3)
        self.octave_action.setVisible(octave)
        self.octave_action.setChecked(octave and self.octave_wanted)
        if not octave or not self.octave_wanted:
            self.octave_panel.hide()
        self.residual_action.setVisible(residual)
        self.shocks_action.setChecked(shocks and self.shocks_wanted)
        if not shocks or instead:
            self.shock_panel.hide()
        if not averaging or not self.wavelet_wanted:
            self.wavelet_panel.hide()
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
        """Nothing applies until something says otherwise.

        Called before every render, so a drawing that has no use for the
        bar — photos, a report — does not have to remember to put away
        the controls the last one left up.

        Which is a promise each control has to be registered here to
        keep. The event box was added later and never was, so once a
        transient comparison raised it, it stayed up over the record on
        its own — where every event is on screen at once and there is
        nothing to choose between — and over a channel table, which has
        no events at all.
        """
        self.show_controls(map_wanted=None, diagonal=None, cmif=False,
                           complex_data=False, pair=False, averaging=False,
                           shocks=False)
        self.event_action.setVisible(False)
        self.scaling_action.setVisible(False)
        self.rms_action.setVisible(False)
        self.show_replication_views(False)
        self.show_srs_views(False)
        # the acts too: the render that follows puts back the ones
        # the new selection can take
        self.show_acts([])

    def show_pair_controls(self, offered: bool) -> None:
        """The bar as the fitting screen wants it: the pair choice alone."""
        for action in (self.curves_action, self.map_action,
                       self.drive_point_action, self.cmif_action,
                       self.averaging_action, self.kurtosis_action,
                       self.shocks_action, self.filter_action,
                       self.truncate_action, self.octave_action,
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


class ScenePane(QWidget):
    """The 3-D view: the scene, its camera, and how the scene is annotated.

    What is *in* the scene stays outside — geometry, deflection and the
    picking that edits them all need the project, and none of them is a
    view's business. The pane owns the render window, the annotations
    drawn around whatever is in it, and the bar those annotations hang
    from. Anyone placing the pane can add their own actions to that bar.
    """

    #: what was copied to the clipboard, for the status line
    copied = Signal(str)

    #: a view choice moved; draw the same scene again
    reread = Signal()

    #: the rigid-body reading was asked for, or put away
    rigid_toggled = Signal(bool)

    def __init__(self, theme_name: str, offscreen: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.theme_name: str = theme_name
        #: the wish survives the selection, the data pane's rule: pick
        #: another geometry and the reading comes back up
        self.rigid_wanted: bool = False
        # the orientation triad is useful at a glance; the labeled box
        # around the geometry is clutter until asked for
        self.bounds_visible: bool = False
        self.orientation_visible: bool = True
        self.axis_unit: str = ''

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
        self.rigid_panel: RigidBodyPanel = RigidBodyPanel()
        self.rigid_panel.hide()
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self._page, 1)
        row.addWidget(self.rigid_panel)
        layout.addLayout(row)
        self._layout = layout
        self._view_row = row

    def _build_toolbar(self) -> QToolBar:
        toolbar = QToolBar('3D view')
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))
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
        # one whole geometry is selected (`offer_rigid`)
        self._rigid_fence = toolbar.addSeparator()
        self.rigid_action: QAction = QAction(control_icon('rigid'),
                                             'Rigid body modes', self)
        self.rigid_action.setCheckable(True)
        self.rigid_action.setToolTip(
            'Preview the six rigid-body mode shapes of this geometry '
            'about a reference point, and make them a shape set')
        self.rigid_action.triggered.connect(self._choose_rigid)
        toolbar.addAction(self.rigid_action)
        self.rigid_action.setVisible(False)
        self._rigid_fence.setVisible(False)
        return toolbar

    def copy_view(self) -> bool:
        """The 3-D view as it stands, onto the clipboard as an image."""
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
        """The acts the selection can take, on this bar (`offer_acts`)."""
        offer_acts(self.toolbar, self.__dict__.setdefault('_acts', {}), acts)

    # ---- the rigid-body reading -------------------------------------------

    def _choose_rigid(self, checked: bool) -> None:
        """The rigid button: remember it, then say so."""
        self.rigid_wanted = bool(checked)
        if not checked:
            self.rigid_panel.hide()
        self.rigid_toggled.emit(bool(checked))

    def offer_rigid(self, offered: bool) -> None:
        """Show the toggle only while it applies — one whole geometry,
        nothing riding it — checked if it was wanted last time."""
        offered = bool(offered)
        self.rigid_action.setVisible(offered)
        self._rigid_fence.setVisible(offered)
        self.rigid_action.setChecked(offered and self.rigid_wanted)
        if not (offered and self.rigid_wanted):
            self.rigid_panel.hide()

    @property
    def showing_rigid(self) -> bool:
        """Whether the rigid-body reading is up."""
        return self.rigid_action.isVisible() and self.rigid_action.isChecked()

    # ---- the render window ------------------------------------------------

    def create_plotter(self) -> Any:
        """Build the embedded VTK view, once the pane is on screen.

        Returns the plotter, or None if it was already built — the caller
        usually wants to draw into a view that has just appeared.
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
        self._view_row.replaceWidget(placeholder, self.plotter)
        self._view_row.setStretchFactor(self.plotter, 1)
        placeholder.deleteLater()
        return self.plotter

    def clear(self) -> None:
        if self.plotter is not None:
            self.plotter.clear()
            self.apply_background()
            self.plotter.render()

    # ---- how the scene is dressed ----------------------------------------

    def set_bounds_visible(self, visible: bool) -> None:
        """Toggle the labeled axes drawn around the geometry."""
        self.bounds_visible = bool(visible)
        self.apply_annotations()

    def set_orientation_visible(self, visible: bool) -> None:
        """Toggle the orientation triad in the corner."""
        self.orientation_visible = bool(visible)
        self.apply_annotations()

    def apply_annotations(self) -> None:
        """Re-apply annotations in place, so the camera keeps its position."""
        if self.plotter is None:
            return
        annotate_scene(self.plotter, self.axis_unit,
                       resolve_theme(self.theme_name), self.bounds_visible,
                       self.orientation_visible)
        self.plotter.render()

    def apply_background(self) -> None:
        """An empty 3D view must match the theme, not VTK's white default."""
        if self.plotter is not None:
            colors = resolve_theme(self.theme_name)
            self.plotter.set_background(colors['scene_background'],
                                        top=colors['scene_background_top'])

    def apply_theme(self, name: str) -> None:
        self.theme_name = name
        self.apply_background()
