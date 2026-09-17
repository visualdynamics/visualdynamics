"""Main application window: project tree, 3D/plot/table views, unit selector."""

from __future__ import annotations

import contextlib
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PySide6.QtCore import (
    QEvent,
    QEventLoop,
    QItemSelectionModel,
    QObject,
    QSignalBlocker,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QBrush,
    QCloseEvent,
    QColor,
    QCursor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QIcon,
    QKeySequence,
    QPalette,
    QResizeEvent,
    QShortcut,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import io
from ..compatibility import check_compatibility
from ..core.averaging import Averaging
from ..core.channel_table import ChannelTable
from ..core.data import (
    Bounded,
    DataArray,
    Frf,
    Psd,
    Specification,
    Spectrum,
    TimeHistory,
    TransientSpecification,
    has_phase,
)
from ..core.entities import LABELS as ENTITY_LABELS
from ..core.geometry import CS_TYPES, ELEMENT_TYPES, Geometry
from ..core.matches import MatchedModes
from ..core.modal_fit import ModalFitSession
from ..core.photos import FORMATS as PHOTO_FORMATS
from ..core.photos import Photos
from ..core.report import (
    PROJECT_TEMPLATES,
    PROJECT_TYPES,
    Report,
    missing_expectations,
)
from ..core.shapes import (
    ShapeSet,
    aligned_mode,
    cross_mac,
    mac_matrix,
    synthesize_overlay,
)
from ..core.sine import (
    SineLevel,
    SineLevelSet,
    SineSweepSpecification,
)
from ..deform import (
    EnvelopeDeflection,
    OdsDeflection,
    ShapeDeflection,
    TimeDeflection,
    animation_records,
)
from ..plot import (
    MAX_RECORDS,
    add_mode_markers,
    bounded_by_specification,
    build_cmif,
    build_coherence_map,
    build_mac,
    build_plots,
    curve_color,
    mac_frame_ratio,
    only_pairs,
    pair_label,
    specification_pairs,
)
from ..rotate import (
    angle_in_plane,
    identity_frame,
    plane_hit,
    ring_points,
    ring_under_cursor,
    rotate_frame,
    wrapped,
)
from ..theme import OVERLAY_ALPHA
from ..theme import theme as resolve_theme
from ..units import DEFAULT_SYSTEM, SYSTEMS, UnitSystem
from ..viz.geometry import (
    AXIS_COLORS,
    ENTITY_LABEL_LIMIT,
    _display_length,
    add_coordinate_system,
    add_geometry,
    annotate_scene,
    display_points,
    labels_fit,
)
from .editors import DoubleSpinBox, SpinBox, commit_on_enter
from .icons import (
    child_icon,
    control_icon,
    object_icon,
    placeholder_icon,
    type_icon,
)

#: how much imaginary part, against the real, counts as a complex
#: measurement rather than the residue of computing one

# what the toolbar's element-type buttons build: type code and node count
ELEMENT_ADD_TYPES = {'beam': (21, 2), 'tri': (41, 3), 'quad': (44, 4)}

# how the user adds to a selection, in the words of their platform
EXTEND_KEYS = 'Shift or Cmd' if sys.platform == 'darwin' else 'Shift or Ctrl'
from ..core.report import OTHER_SIDE
from ..names import display_name
from ..project import TYPE_ORDER, Project, describe, type_rank
from .object_tables import (
    ENTITY_TABLES,
    channel_table_model,
    frf_units_models,
    matched_modes_model,
    modal_fit_table_model,
    shape_table_model,
    units_table_model,
)
from .panes import DataPane, ScenePane
from .preferences import (
    chosen_scheme,
    refresh_palettes,
    remember_appearance,
    remembered_appearance,
    wear_appearance,
)
from .project_tree import (
    PROJECT_ROW,
    ROLE_ACTIVE,
    ROLE_DRAGGABLE,
    ROLE_DROP_SLOT,
    ROLE_WHOLE_PROJECT,
    ProjectTree,
)
from .record_grid import RecordGrid, channel_quantities
from .tables import CopyPasteTableView

ROLE_ORIGINAL_NAME = Qt.ItemDataRole.UserRole       # top-level, for renames
ROLE_REFERENCE = Qt.ItemDataRole.UserRole + 1       # what an item points at
ROLE_POPULATED = Qt.ItemDataRole.UserRole + 2       # lazy children built yet?

# room a unit cell needs beyond its text: the drop-down arrow and padding
COMBO_CELL_PADDING = 44

FRAMES_PER_SECOND = 30
PHASE_STEPS = 120            # frames in one mode-shape cycle
SECONDS_PER_CYCLE = 2.0      # every mode animates at this rate, whatever its Hz
SECONDS_PER_RECORD = 10.0    # a time record plays in about this long

# What Faster and Slower step through. Doublings, because that is how a
# speed reads — half as fast, twice as fast — and a ladder rather than a
# free number because nobody wants to type 1.7. The default is 1.0, which
# is the rate above, and the two ends are far enough apart to be worth
# having: an eighth crawls a mode through its cycle, eight times runs a
# long record past in seconds.
SPEEDS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
NORMAL_SPEED = SPEEDS.index(1.0)

MAX_LISTED_ENTITIES = 2000  # a big FE model must not build 100k tree items

ENTITY_COMPONENT = {
    'node': 'nodes',
    'coordinate_system': 'coordinate_systems',
    'traceline': 'tracelines',
    'element': 'elements',
    'block': 'blocks',
}

# geometry categories: (label, attribute holding the collection, component)
# Blocks come after the elements they group, and are the one category with
# nothing of their own in the view: picking one shows its elements.
GEOMETRY_PARTS = [
    ('Nodes', 'node_id', 'nodes'),
    ('Coordinate systems', 'cs_id', 'coordinate_systems'),
    ('Tracelines', 'traceline_conn', 'tracelines'),
    ('Elements', 'elem_conn', 'elements'),
    ('Blocks', 'block_id', 'blocks'),
]

# what is drawn for a category, where that is not the category itself
DRAWN_AS = {'blocks': 'elements'}


def _row_for_entity(geometry, component, entity):
    """Table row for a picked entity — id for nodes and coordinate systems,
    index for tracelines and elements."""
    if geometry is None or entity is None:
        return None
    if component == 'nodes':
        rows = np.flatnonzero(geometry.node_id == entity)
        return int(rows[0]) if len(rows) else None
    if component == 'coordinate_systems':
        rows = np.flatnonzero(geometry.cs_id == entity)
        return int(rows[0]) if len(rows) else None
    if component == 'blocks':
        rows = np.flatnonzero(geometry.block_id == entity)
        return int(rows[0]) if len(rows) else None
    total = len(geometry.traceline_conn if component == 'tracelines'
                else geometry.elem_conn)
    return int(entity) if 0 <= int(entity) < total else None


def _entity_key(geometry, component, row):
    """What identifies the entity in a given table row: its id, for all
    five groups — every one of them is named the same way."""
    return int({'nodes': geometry.node_id,
                'coordinate_systems': geometry.cs_id,
                'tracelines': geometry.traceline_id,
                'elements': geometry.elem_id,
                'blocks': geometry.block_id}[component][row])


def _delete_from_geometry(geometry, component, keys):
    return {
        'nodes': geometry.delete_nodes,
        'coordinate_systems': geometry.delete_coordinate_systems,
        'tracelines': geometry.delete_tracelines,
        'elements': geometry.delete_elements,
        'blocks': geometry.delete_blocks,
    }[component](keys)


def _block_element_rows(geometry, block_ids):
    """Which element rows the given blocks hold — what a block *is* in the
    view, since a block has no geometry of its own."""
    wanted = [int(b) for b in block_ids]
    return [int(row) for row in np.flatnonzero(
        np.isin(geometry.elem_block, wanted))]


def _drawable(geometry, components, entities):
    """The same selection, in what the renderer can draw.

    A block is a label on elements, so picking one draws the elements it
    holds and picking the category draws all of them. Translating here
    rather than in the renderer keeps `add_geometry` knowing only about
    things that have coordinates — and means an empty block correctly
    highlights nothing rather than the whole model.
    """
    if components and 'blocks' in components:
        components = {DRAWN_AS.get(c, c) for c in components}
    if entities and entities.get('blocks'):
        rows = _block_element_rows(geometry, entities['blocks'])
        entities = {kind: values for kind, values in entities.items()
                    if kind != 'blocks'}
        entities['elements'] = sorted({*entities.get('elements', []), *rows})
    return components, entities


def _label_kinds(geometry: Geometry, components: Any,
                 entities: Any) -> dict[str, Any]:
    """Which kinds the selection wants captioned with their ids.

    Picking a category captions all of it; picking entities captions
    those. Worked out **before** `_drawable` translates blocks into the
    elements they hold, because a block's caption is its own name at the
    middle of its elements and the elements' captions are their ids —
    after the translation there is nothing left to tell those apart.

    A kind with more of it than `ENTITY_LABEL_LIMIT` is left out rather
    than drawn: see `_label_note`, which is what tells the user so.
    """
    wanted: dict[str, Any] = {}
    for kind in components or ():
        wanted[kind] = None
    for kind, chosen in (entities or {}).items():
        if chosen:
            wanted[kind] = list(chosen)
    return {kind: chosen for kind, chosen in wanted.items()
            if labels_fit(geometry, kind, chosen)}


def _label_note(geometry: Geometry, components: Any, entities: Any) -> str:
    """'; too many to number' when a selection is past the caption limit.

    Said rather than left to be noticed: ids appearing for a small
    geometry and silently not for a large one reads as a bug in the
    large one.
    """
    asked = set(components or ()) | {kind for kind, chosen
                                     in (entities or {}).items() if chosen}
    dropped = [kind for kind in asked
               if not labels_fit(geometry, kind,
                                 (entities or {}).get(kind) or None)]
    if not dropped:
        return ''
    return (f'; too many {", ".join(sorted(k.replace("_", " ") for k in dropped))}'
            f' to number (over {ENTITY_LABEL_LIMIT})')


def _summarize(obj: Any) -> str:
    """A few words about an object for the status bar — the project's
    own description, which the tree's own repr uses too."""
    return describe(obj)


def _keep_selection_vivid(view):
    """Draw selected rows the same whether or not the view holds focus.

    Rows also get selected by picking in the 3D view, which keeps the focus;
    Qt would gray those out as an inactive selection and make a live
    selection look disabled.
    """
    palette = view.palette()
    roles = (QPalette.ColorRole.Highlight, QPalette.ColorRole.HighlightedText)
    if all(palette.color(QPalette.ColorGroup.Inactive, role)
           == palette.color(QPalette.ColorGroup.Active, role)
           for role in roles):
        return
    for role in roles:
        palette.setColor(QPalette.ColorGroup.Inactive, role,
                         palette.color(QPalette.ColorGroup.Active, role))
    view.setPalette(palette)


def _hover_cells(geometry, component, entity):
    """VTK cell arrays lighting up one entity, over the scene's own points."""
    empty = {'verts': np.empty(0, dtype=np.int64),
             'lines': np.empty(0, dtype=np.int64)}
    row = _row_for_entity(geometry, component, entity)
    if row is None:
        return empty
    if component == 'nodes':
        return {**empty, 'verts': np.array([1, row], dtype=np.int64)}
    if component == 'coordinate_systems':
        return empty          # triads are drawn separately
    nodes = (geometry.traceline_conn[row] if component == 'tracelines'
             else geometry.elem_conn[row])
    lookup = {int(node): index for index, node in enumerate(geometry.node_id)}
    indices = [lookup[int(node)] for node in nodes if int(node) in lookup]
    if len(indices) < 2:
        return ({**empty, 'verts': np.array([1, indices[0]], dtype=np.int64)}
                if indices else empty)
    if component == 'elements' and len(indices) > 2:
        indices = indices + indices[:1]
    return {**empty,
            'lines': np.array([len(indices), *indices], dtype=np.int64)}


def _units_defined(obj) -> bool:
    """Channel tables carry unit text per channel and are always 'defined'."""
    return getattr(obj, 'units_defined', True)


class _RatioFrame(QWidget):
    """Keeps its one child at the matrix's own aspect, centered.

    An auto-MAC is square; a cross-MAC between sets of different sizes is
    as rectangular as they are, cells staying square either way — up to
    `MAC_ASPECT_CLAMP`, past which the shape is clamped and the cells
    stretch instead. 139 modes against 8 asked for a widget 52 pixels
    wide, of which the left axis alone wants 66.
    """

    def __init__(self, child: QWidget,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.child: QWidget = child
        #: width over height, clamped by `mac_frame_ratio`
        self.ratio: float = 1.0
        child.setParent(self)

    def set_ratio(self, ratio: float) -> None:
        self.ratio = max(float(ratio), 1e-6)
        self._place()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._place()

    def _place(self):
        width = min(self.width(), int(self.height() * self.ratio))
        height = int(width / self.ratio)
        if height > self.height():
            height = self.height()
            width = int(height * self.ratio)
        self.child.setGeometry((self.width() - width) // 2,
                               (self.height() - height) // 2, width, height)


class _WindowDragWatch(QObject):
    """Logs drag events at the QWindow, before widget dispatch."""

    _KINDS: ClassVar[dict] = {
        QEvent.Type.DragEnter: 'enter', QEvent.Type.DragMove: 'move',
        QEvent.Type.DragLeave: 'leave', QEvent.Type.Drop: 'DROP'}

    def __init__(self, window) -> None:
        super().__init__(window)
        self._window = window
        self._watched = None                # the QWindow filtered so far
        window.installEventFilter(self)     # for winId changes
        self._watch_handle()

    def _watch_handle(self) -> None:
        """Filter the widget's QWindow, once per QWindow: a Show can
        arrive more than once (Windows re-shows the main window when a
        native child is made), and stacking a filter per Show is what
        put this object on the stack six deep in the 0.1.0a1 launch
        crash (2026-09-14)."""
        handle = self._window.windowHandle()
        if handle is not None and handle is not self._watched:
            handle.installEventFilter(self)
            self._watched = handle

    def eventFilter(self, target, event):
        from .project_tree import trace_drag

        if target is self._window and event.type() == QEvent.Type.Show:
            self._watch_handle()
        kind = self._KINDS.get(event.type())
        if kind is not None and target is self._window.windowHandle():
            mime = getattr(event, 'mimeData', lambda: None)()
            trace_drag('qwindow', kind, mime, throttle=kind == 'move')
        return False                        # observe, never consume


#: built on first use — pyqtgraph is imported lazily everywhere in this
#: module, and a class statement is an import
_DAMPING_CURSOR: type | None = None


def _damping_cursor_class() -> type:
    """The fit cursor: horizontal for frequency, vertical for damping.

    `InfiniteLine` reads only the motion across itself; the vertical
    component of the same drag is picked up here and handed to
    `dragged_vertically` as pixels from the grab, with the damping the
    drag started from — the window maps pixels to damping, so the
    mapping lives with the other fit policy rather than in a plot item.
    """
    global _DAMPING_CURSOR
    if _DAMPING_CURSOR is not None:
        return _DAMPING_CURSOR
    import pyqtgraph as pg

    class _DampingCursor(pg.InfiniteLine):
        #: set by the window: called with the scene y, once armed
        dragged_vertically = None
        #: vertical pixels before a drag means damping at all — a
        #: horizontal drag wobbles, and a wobble must not silently take
        #: the damping over. Crossing it once arms the axis for the
        #: rest of that drag, so dragging back near the grab line does
        #: not disarm what the user has started steering.
        dead_band = 14.0

        def begin_vertical(self, y):
            self._grab_y = y
            self._armed = False

        def follow_vertical(self, y):
            if not self._armed and abs(y - self._grab_y) > self.dead_band:
                self._armed = True
            return self._armed

        def mouseDragEvent(self, ev):
            super().mouseDragEvent(ev)   # the frequency half, unchanged
            if self.dragged_vertically is None or not self.movable:
                return
            if ev.isStart():
                self.begin_vertical(ev.scenePos().y())
                return
            if getattr(self, '_grab_y', None) is None or ev.isFinish():
                return
            y = ev.scenePos().y()
            if self.follow_vertical(y):
                self.dragged_vertically(y)

    _DAMPING_CURSOR = _DampingCursor
    return _DampingCursor


class _FileDropMixin:
    """Accepts a drag of files and imports it; leaves all else alone.

    Mixed into the project dock and the holder inside it, so the dock's
    chrome answers a file drag the same way the tree and the window do.
    Everything funnels into the window's importer through
    `files_dropped`; the queued import is the window's business, not
    repeated here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        from .project_tree import dropped_files, trace_drag

        trace_drag(type(self).__name__, 'enter', event.mimeData())
        if dropped_files(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        from .project_tree import dropped_files

        if dropped_files(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        from .project_tree import dropped_files, trace_drag

        paths = dropped_files(event.mimeData())
        trace_drag(type(self).__name__, f'drop paths={len(paths)}',
                   event.mimeData())
        if not paths:
            super().dropEvent(event)
            return
        event.setDropAction(Qt.DropAction.CopyAction)
        event.acceptProposedAction()
        self.files_dropped.emit(paths)


class _TakesFileDrops(_FileDropMixin, QWidget):
    files_dropped = Signal(list)


class _FileDropDock(_FileDropMixin, QDockWidget):
    files_dropped = Signal(list)


#: how long after the last edit of a dragged setting the open report
#: is rebuilt. Long enough that a drag never pays for it (a rebuild is
#: 254 ms where the filtering it reports is 3), short enough that
#: letting go and looking at the report shows the new numbers.
REPORT_SETTLE_MS = 250


class MainWindow(QMainWindow):
    """The app: a project tree, and panes that read whatever is picked.

    The window holds a `Project` and nothing else — the same object a
    script builds, with the same verbs on it — so a project built by
    clicking and one built by calling are the same project and open in
    each other. Every button here is one of those verbs plus a sentence
    in the status bar; nothing is computed in this file that a script
    cannot compute without it.

    What it adds is the reading: which pane suits what is selected,
    which of several ways to read it is wanted, and the editing of the
    things that are edited rather than computed — units, channel
    tables, geometry, the report.
    """

    def __init__(self, offscreen_3d: bool = False) -> None:
        """offscreen_3d replaces the embedded VTK view with an off-screen
        plotter — VTK's Qt widget cannot run under Qt's offscreen platform,
        so headless tests use this to exercise the full render path."""
        super().__init__()
        from .project_tree import start_trace

        start_trace()               # the drop log begins with the window
        #: where keyboard focus waits out the window's inactive spells —
        #: see `event` for the macOS drop bug this dodges. Zero-size on
        #: purpose: the bug poisons the focused widget's rect, and this
        #: rect cannot be hit.
        self._focus_park: QWidget = QWidget(self)
        self._focus_park.setFixedSize(0, 0)
        self._focus_park.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._parked_focus: QWidget | None = None
        # The QWindow sees platform drag events before any widget
        # dispatch, so its line splits the world in two: present here
        # and absent from a widget is Qt losing it; absent here is the
        # platform never delivering it. Installed on our own QWindow
        # alone — an application-wide filter touched Chromium's
        # machinery and segfaulted.
        self._drag_watch = _WindowDragWatch(self)
        self.setWindowTitle('Visual Dynamics')
        self.resize(1280, 800)
        self.setAcceptDrops(True)
        self.unit_system: UnitSystem = DEFAULT_SYSTEM
        # the window shows a Project and owns nothing the Project owns:
        # objects, links, type and active geometry are its fields,
        # reached through the properties below, so clicking and
        # scripting act on one structure rather than two that agree
        self.project: Project = Project()
        #: name -> RecordGrid, while that object's row is expanded
        self.record_grids: dict[str, Any] = {}
        self._rows_wired = False  # is rows_deleted connected right now
        #: name -> the file it was read from
        self.object_sources: dict[str, str] = {}
        #: name -> what that file called it
        self.object_keys: dict[str, str] = {}
        #: the last compatibility check over the project
        self.report: Any = None
        self._status_text = ''

        from . import check_qt_binding

        check_qt_binding()   # a mismatch is unreadable once it reaches a layout
        import pyqtgraph as pg

        # the appearance chosen for this launch, remembered, or the
        # platform's — in that order (gui/preferences.py) — worn by the
        # whole application first, so the chrome and the drawn parts
        # are built to the same scheme
        wear_appearance()
        self.theme_name: str = chosen_scheme()
        colors = resolve_theme(self.theme_name)
        pg.setConfigOption('background', colors['plot_background'])
        pg.setConfigOption('foreground', colors['plot_foreground'])

        # panes stack top to bottom: geometry above its data
        self.views: QSplitter = QSplitter(Qt.Orientation.Vertical)
        #: the live deflection, while data is shown on a geometry
        self.animator: Any = None
        self._playing = False
        self._rendering = False       # drop frames rather than queue them
        self._shape_mode = False      # phase sweep, versus stepping samples
        #: (geometry name, entity kind) while an edit table is open
        self.editing: tuple[str, str] | None = None
        #: clicking in the view creates things
        self.add_mode: bool = False
        self._picked_nodes = []       # nodes gathered for a traceline/element
        self._shape_source = None     # (geometry, shape set) while a mode plays
        #: (geometry name, the proposed set) while the rigid-body
        #: reading previews on the scene
        self._rigid_preview = None
        self._picker = None           # entity under the cursor, while editing
        self._projector = None
        self._hovered = None
        self._hover_mesh = None
        self._picked_mesh = None
        self._framed = ()          # what the camera was last framed for
        self._rotating = None      # the ring gesture in progress
        self._rotate_observers = []
        self._last_axis = 2        # the ring a typed angle turns about
        self._pick_observers = []
        self._cursor = None
        self._cursor_dimension = None
        self._cursor_abscissa = None
        self._frame_index = 0
        # the same position, unrounded: see `_advance`
        self._frame_position = 0.0
        self._phase_position = 0.0
        # the 3-D view is a pane of its own: it owns the render window, the
        # annotations drawn around whatever is in it, and the bar they hang
        # from. What goes *into* the scene needs the project, so that stays
        # here and is added to the same bar.
        #: kept for the 3-D surfaces built later than the panes — the
        #: MAC bars' plotter is made on first use, long after __init__
        self._offscreen_3d: bool = offscreen_3d
        self.scene: ScenePane = ScenePane(self.theme_name,
                                          offscreen=offscreen_3d)
        self._build_view_toolbar(self.scene.toolbar)
        # the 2-D data view is a pane of its own: it owns the plot, the bar
        # that says how to read it, and that choice — and tells us when the
        # choice moves rather than acting on it
        self.data_pane: DataPane = DataPane(self.theme_name,
                                            offscreen=offscreen_3d)
        #: which object the *pager* is counting pages for — shared by
        #: both readings, since it is one control on one bar. Separate
        #: from `_waterfall_object` below, which answers a different
        #: question: where the 3-D camera belongs.
        self._paged_object: str | None = None
        #: which object the waterfall camera was placed for. The camera
        #: is placed when this changes and at no other time — a reread
        #: (units switch, record pick, component change) redraws the
        #: scene under the view the user set
        self._waterfall_object: str | None = None
        #: drags the stage's averaging span and shock windows; built
        #: with the stage plotter on first use
        self._stage_dragger = None
        #: coalesces report rebuilds during a drag — see
        #: `_report_content_changed`
        self._report_settle: QTimer = QTimer(self)
        self._report_settle.setSingleShot(True)
        self._report_settle.timeout.connect(self._report_content_changed)
        #: how to restate the stage's filter twin when the panel
        #: moves — set by `_stage_filter_preview`, which is the
        #: only thing that knows this render's extents
        self._stage_filter_redraw: Any = None
        #: {name: why} for derived objects whose source's settings have
        #: moved — read by the badges, refreshed at every settings edit
        self._stale: dict[str, str] = {}
        self.data_pane.reread.connect(self.render_current)
        self.data_pane.log_frequency_action.triggered.connect(
            self._frequency_axis_toggled)
        self.data_pane.drive_points_toggled.connect(self._toggle_drive_points)
        self.data_pane.residual_toggled.connect(self._toggle_fit_residual)
        self.data_pane.pair_mode_chosen.connect(self._set_pair_mode)
        self.data_pane.averaging_toggled.connect(
            lambda _wanted: self.render_current())
        self.data_pane.averaging_panel.changed.connect(self._averaging_edited)
        self.data_pane.shocks_toggled.connect(
            lambda _wanted: self.render_current())
        self.data_pane.filter_toggled.connect(
            lambda _wanted: self.render_current())
        self.data_pane.filter_panel.changed.connect(self._filtering_edited)
        self.data_pane.truncate_toggled.connect(
            lambda _wanted: self.render_current())
        self.data_pane.truncate_panel.changed.connect(
            self._truncation_edited)
        self.data_pane.octave_toggled.connect(
            lambda _wanted: self.render_current())
        self.data_pane.octave_panel.changed.connect(
            lambda _per: self.render_current())
        self.data_pane.octave_panel.apply_asked.connect(
            self.compute_octave)
        # the acts, asked for where their settings are set (Brandon,
        # 2026-08-28): each button is a second entrance to the same
        # verb the bar's act offers, so provenance, badges and the
        # scripting API are untouched
        self.data_pane.filter_panel.apply_asked.connect(self.filter_data)
        self.data_pane.truncate_panel.apply_asked.connect(
            self.truncate_data)
        # the specification draft: every edit re-previews, the button
        # makes the object. A specification opened to edit is drawn on
        # the plot, so its toggle sits on the plot's own bar
        self.data_pane.author_panel.changed.connect(self._author_edited)
        self.data_pane.author_panel.form_asked.connect(self._author_form_asked)
        self.data_pane.author_panel.constraint_asked.connect(
            self._author_constraint_asked)
        self.author_data_action: QAction = QAction(control_icon('edit'),
                                                   'Edit', self)
        self.author_data_action.setCheckable(True)
        self.author_data_action.setToolTip(
            'Open this specification as a sheet — breakpoints, cross '
            'terms, bands — and replace it')
        self.author_data_action.triggered.connect(self._author_toggled)
        self.data_pane.toolbar.addAction(self.author_data_action)
        self.author_data_action.setVisible(False)
        # the scene's own reading: the rigid-body preview and its act
        self.scene.rigid_toggled.connect(
            lambda _wanted: self.render_current())
        self.scene.rigid_panel.changed.connect(self._rigid_edited)
        self.scene.rigid_panel.apply_asked.connect(
            self.generate_rigid_body_modes)
        self.data_pane.shock_panel.srs_asked.connect(self.compute_srs)
        self.data_pane.averaging_panel.psds_asked.connect(
            self.compute_psds)
        self.data_pane.averaging_panel.cpsds_asked.connect(
            self.compute_cpsds)
        self.data_pane.averaging_panel.spectra_asked.connect(
            self.compute_spectra)
        self.data_pane.averaging_panel.frfs_asked.connect(
            self.compute_frfs)
        self.data_pane.averaging_panel.coherence_asked.connect(
            self.compute_multiple_coherence)
        self.data_pane.shock_panel.changed.connect(self._shocks_edited)
        self.data_pane.shock_panel.length_mode_changed.connect(
            self._shock_length_mode)
        self.data_pane.shock_panel.detect_asked.connect(self._detect_shocks)
        self.data_pane.shock_panel.add_asked.connect(self._add_shock)
        self.data_pane.spectra_view_chosen.connect(
            lambda _which: self.render_current())
        self.data_pane.comparison_chosen.connect(
            lambda _which: self.render_current())
        self.data_pane.scaling_edited.connect(self._comparison_scale_edited)
        # the channel box and the channel table ask the same question,
        # so whichever was touched last is the answer. Connected before
        # the redraw below, because it has to have taken effect by the
        # time the drawing reads it
        self.data_pane.pair_chosen.connect(
            lambda: setattr(self, '_replication_pairs', []))
        self.data_pane.pair_chosen.connect(
            lambda: setattr(self, '_compliance_channels', []))
        self.data_pane.pair_chosen.connect(self.render_current)
        self.data_pane.replication_chosen.connect(
            lambda _which: self.render_current())
        self.data_pane.event_chosen.connect(self._event_chosen)
        self.data_pane.srs_chosen.connect(
            lambda _which: self.render_current())
        # ...and from anywhere, for when the tree has the keyboard and
        # plain arrows are its own
        for keys, step in (('Alt+Up', -1), ('Alt+Down', 1)):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.activated.connect(
                lambda step=step: self.data_pane.step_pair(step))
        #: the frames drawn on the time history, one overlay per plot the
        #: history was split across — force and acceleration get a row each
        self.averaging_overlays: list[Any] = []
        self.filter_overlays: list[Any] = []
        self.truncation_overlays: list[Any] = []
        self.octave_previews: list[Any] = []
        #: and the shock windows, the same way
        self.shock_overlays: list[Any] = []
        #: which comparison the plot is drawing, so the compliance table
        #: can mark its row; and the guard that stops the two of them
        #: from chasing each other
        self._plotted_pair = None
        self._syncing_compliance = False
        #: the bar chart currently up, when the comparison is being read
        #: as one rather than as spectra
        self.bar_chart: Any = None
        self.table: CopyPasteTableView = CopyPasteTableView()
        self.table.edits_applied.connect(self._report_edits)
        # the table with, in fit mode, its Confirm bar underneath
        self.table_pane: QWidget = QWidget()
        table_layout = QVBoxLayout(self.table_pane)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(0)
        # the auto-MAC sits to the right of a mode table — how distinct
        # the modes are belongs beside the list of them. Square, because
        # the matrix is; the frame keeps it that way whatever the pane does
        self.mac_view: Any = pg.GraphicsLayoutWidget()
        self.mac_frame: _RatioFrame = _RatioFrame(self.mac_view)
        self.mac_frame.hide()
        self.mac_view.scene().sigMouseClicked.connect(self._mac_clicked)
        # comparing shape sets is often about the MAC alone: a bar over
        # the pane toggles the mode table on the left away
        self.table_bar: QToolBar = QToolBar()
        self.table_bar.setIconSize(QSize(20, 20))
        self.mode_table_action: QAction = QAction(control_icon('table'),
                                         'Mode Table', self)
        self.mode_table_action.setCheckable(True)
        self.mode_table_action.setChecked(True)
        self.mode_table_action.setToolTip(
            'Show the table beside the MAC — the mode list for one '
            'set, the matched-modes table while comparing two')
        self.mode_table_action.triggered.connect(
            lambda checked: self.table.setVisible(checked))
        # the MAC's own 3-D reading: the matrix as bars, height and
        # color both the value, and the default here too (Brandon's
        # call, made once the bars could be picked like the grid)
        self.mac_bars_action: QAction = QAction(control_icon('waterfall'),
                                       '3D', self)
        self.mac_bars_action.setCheckable(True)
        self.mac_bars_action.setChecked(True)
        self.mac_bars_action.setToolTip(
            'The MAC as 3-D bars, height and color the value — '
            'click picks a pair, Shift adds it, exactly as on the grid')
        self.mac_bars_action.triggered.connect(self._rerender_mac)
        # the 2D/3D toggle leads this bar, as it leads the data pane's
        # (Brandon, 2026-08-30): the same control sits in the same
        # place whichever view is up
        self.table_bar.addAction(self.mac_bars_action)
        self.table_bar.addAction(self.mode_table_action)
        self.mac_bars_action.setVisible(False)
        # writing a specification at the set's modal coordinates: the
        # reading of a lone shape set that makes an object (PLAN.md,
        # "The virtual point arc"). Its pane sits beside the plot,
        # which previews the autospectra as they are typed
        self.author_action: QAction = QAction(control_icon('edit'),
                                              'Specification', self)
        self.author_action.setCheckable(True)
        self.author_action.setToolTip(
            'Write a specification at this set\'s modal coordinates — '
            'breakpoints, every cross term, the bands — and make it')
        self.author_action.triggered.connect(self._author_toggled)
        self.table_bar.addAction(self.author_action)
        self.author_action.setVisible(False)
        self._author_wanted: bool = False
        #: the draft per object it was opened on, kept while the
        #: window lives
        self._drafts: dict[str, Any] = {}
        #: which objects each cached draft was opened on — so a write
        #: through one door (the whole specification) drops the drafts
        #: cached under its other doors (one picked channel), which
        #: otherwise kept showing the object as it was (Brandon,
        #: 2026-09-06: the sub-items had not updated back to ±3 dB)
        self._draft_names: dict[str, tuple[str, ...]] = {}
        #: what the reading is open on: ('shapes' | 'spec' | 'table', name)
        self._author_door: tuple[str, str] | None = None
        #: the specification a shape-set or channel-table door made, by
        #: door key — a render between the making and the deferred
        #: switch of selection must not make a second one
        self._author_made: dict[str, str] = {}
        #: the 3-D MAC surface, built on first use, and which comparison
        #: its camera was placed for — placed on a change, kept on a
        #: reread, the same standing rule as every scene
        self._mac_bars_page: QWidget | None = None
        self._mac_bars_plotter_obj: Any = None
        self._mac_bars_shown: Any = None
        #: (rows, columns) of the bars last drawn — what a picked mesh
        #: cell is decoded against
        self._mac_bars_grid: tuple[int, int] | None = None
        # comparing two sets that live on a geometry: the picked pair
        # can animate overlaid, phase-aligned — the toggle remembers
        self.overlay_action: QAction = QAction(control_icon('overlay'),
                                      'Overlay Animation', self)
        self.overlay_action.setCheckable(True)
        self.overlay_action.setChecked(True)
        self.overlay_action.setToolTip(
            'Animate the picked pair of modes over each other, the '
            'second phase-aligned to the first')
        self.overlay_action.triggered.connect(
            lambda _checked: self.render_current())
        self.table_bar.addAction(self.overlay_action)
        self.overlay_action.setVisible(False)
        # committing picked MAC squares builds the matched-modes
        # object, the way fitting builds the shape set
        self.add_matches_action: QAction = QAction(control_icon('add'),
                                          'Add Matches', self)
        self.add_matches_action.setToolTip(
            'Add the selected MAC squares to the matched modes for '
            'this pair of sets — created on first use')
        self.add_matches_action.triggered.connect(self.add_matches)
        self.table_bar.addAction(self.add_matches_action)
        self.add_matches_action.setVisible(False)
        self.table_bar.hide()
        table_layout.addWidget(self.table_bar)
        # a splitter, not a row: how much of the pane the mode list wants
        # against how big the MAC should be is a judgment about the data
        # in front of you, so it is a divider to drag rather than a ratio
        # we picked
        self.tables_row: QSplitter = QSplitter(Qt.Orientation.Horizontal)
        self.tables_row.setChildrenCollapsible(False)
        self.tables_row.addWidget(self.table)
        self.tables_row.addWidget(self.mac_frame)
        self.tables_row.setSizes([500, 500])
        # the report editor borrows this pane: the rendered report
        # itself, editable — created on first use, Chromium is heavy
        self.report_editor: Any = None
        self._table_layout = table_layout
        # the splitter takes the height going spare; without the
        # stretch the button bar below it claimed an even share and
        # stood a quarter of the window tall
        table_layout.addWidget(self.tables_row, 1)
        self.fit_bar: QWidget = QWidget()
        fit_layout = QHBoxLayout(self.fit_bar)
        fit_layout.setContentsMargins(6, 4, 6, 4)
        fit_layout.addStretch(1)
        # Left of Find Mode, and first into the layout, because the
        # bar is right-aligned by the stretch above: the group grows
        # and shrinks at its *left* edge, so Refine All appearing after
        # the second mode does not shift Find Mode and Confirm Mode out
        # from under a cursor already on them. Added last, it pushed
        # both of them left the moment it showed up.
        #
        # Sequential fitting never goes back: the first of a close pair
        # was fit on data still containing the second. This goes back —
        # poles held, every mode's residues re-fit together. Offered
        # only once there are two modes to influence each other.
        self.refine_modes_button: QPushButton = QPushButton('Refine All')
        self.refine_modes_button.setToolTip(
            'Re-fit every confirmed mode\'s shape together, holding the '
            'frequencies and dampings — close modes stop borrowing from '
            'each other')
        self.refine_modes_button.clicked.connect(self.refine_all_modes)
        fit_layout.addWidget(self.refine_modes_button)
        self.find_mode_button: QPushButton = QPushButton('Find Mode')
        self.find_mode_button.setToolTip(
            'Move the cursor to the largest CMIF peak left in the '
            'residual, within the frequencies on screen — the next '
            'mode to fit. Zoom to narrow the search.')
        self.find_mode_button.clicked.connect(self.find_next_mode)
        fit_layout.addWidget(self.find_mode_button)
        self.confirm_mode_button: QPushButton = QPushButton('Confirm Mode')
        self.confirm_mode_button.setToolTip(
            'Fit a real normal mode at the cursor and move it to the next '
            'largest residual peak')
        self.confirm_mode_button.clicked.connect(self.confirm_fit_mode)
        fit_layout.addWidget(self.confirm_mode_button)
        table_layout.addWidget(self.fit_bar)
        self.fit_bar.hide()
        for pane in (self.scene, self.data_pane, self.table_pane):
            self.views.addWidget(pane)
        self.data_pane.hide()
        self.table_pane.hide()
        # declaring units sits beside the views rather than over them: the
        # plot is how you tell what a channel is, so it has to stay visible
        #: (name, object) while the imported-units pane is up
        self.units_target: tuple[str, Any] | None = None
        #: (channel, playing) pairs the replication grid is pointing
        #: the plot at, kept here rather than in the table because the
        #: table is rebuilt on every drawing and the choice is not
        self._replication_pairs = []
        #: DOF pairs the compliance table is pointing the plot at — the
        #: same arrangement the transient grid has, for a comparison
        #: with no repeats to spread across
        self._compliance_channels = []
        #: set while a redraw is already queued, so a drag across the
        #: grid draws once rather than once per cell it crosses
        self._replication_pending = False
        #: the grid on screen and the model showing it, so a drawing
        #: that changes nothing about them leaves the user's selection
        #: alone rather than rebuilding it from this end's idea of it
        self._replication_holder = None
        self._replication_model = None
        self._compliance_holder = None
        self._compliance_model = None
        self._replication_units = '%'
        #: the ModalFitSession while a modal fit is open
        self.fit: Any = None
        self.fit_name: str | None = None
        self.fit_object_name: str | None = None
        self._fit_cursor = None
        self._fit_parabola = None
        self._fit_damping_label = None
        self._fit_coherence_name = None
        #: what the plot's top and bottom edges mean as damping, kept
        #: across fits — an article's plausible range rarely changes
        #: between two of its own FRF sets
        self._fit_damping_range: list[float] = [self.FIT_DAMPING_TOP,
                                                self.FIT_DAMPING_BOTTOM]
        self._fit_range_edits: list | None = None
        #: the dashed synthesis curves currently on the CMIF,
        #: for updating in place while the cursor is dragged
        self._fit_synthesis = []
        #: fires when the cursor has been still a moment
        self._fit_settle = QTimer(self)
        self._fit_settle.setSingleShot(True)
        self._fit_settle.timeout.connect(self._fit_settled)
        #: set while a drag tick is running, so the next is
        #: dropped rather than queued
        self._fit_dragging = False
        #: what the dear tier — the fit and the dashed synthesis —
        #: cost the last time it ran, in seconds. The gate on a drag
        #: tick reads this *before* paying: the old scheme ran the
        #: dear tier first and checked the budget after, which on the
        #: hard drone survey meant every drag opened with the full
        #: bill — seconds of freeze — before the guard could notice,
        #: and paid it again after every pause, because the settle
        #: re-armed it. Remembered cost, primed by the settle that
        #: `start_modal_fit` schedules, means a set too big to follow
        #: live never stalls a single tick.
        self._fit_dear_cost = 0.0
        #: whether the dashed synthesis has fallen behind the cursor —
        #: a tick that skipped the dear tier sets it, and the settle
        #: catches up
        self._fit_synthesis_stale = True
        self._fit_model = None
        self._fit_plot = None
        self._compare = None           # {'pair', 'cell'} while comparing
        #: (spec name, measured name) the Scaling field is editing
        self._scaling_pair = None
        #: an import is running; a second one queued behind it waits
        self._importing = False
        #: what MAC the mac_view currently shows, so redrawing the same
        #: comparison keeps the zoom (picking cells re-renders it)
        self._mac_shown = None
        #: re-entrancy guard: showing a pair rebuilds the table that
        #: asked for it, and its selection would ask again
        self._matched_row_moving = False
        self._compare_active = False
        #: how an FRF and a shape set together read: 'fit' | 'overlay'
        self.pair_mode: str = 'fit'
        self._pair_selected = False
        self._pair_with_picks = False
        self._units_restate = QTimer(self)
        self._units_restate.setSingleShot(True)
        self._units_restate.timeout.connect(self._restate_after_units)
        self.main_split: QSplitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_split.addWidget(self.views)
        self.main_split.addWidget(self._build_units_panel())
        self.main_split.setStretchFactor(0, 3)
        self.main_split.setStretchFactor(1, 1)
        self.units_panel.hide()
        # the console rides the bottom edge as a tab (Brandon,
        # 2026-08-30): collapsed by default, one click to expand the
        # session's journal, one to put it away — no menu to find
        from .console import ConsolePanel

        center = QWidget()
        column = QVBoxLayout(center)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.main_split, 1)
        self.console: ConsolePanel = ConsolePanel()
        column.addWidget(self.console)
        # the tab is not a layout row: floated over the views' bottom
        # edge, so collapsed the console claims no strip at all
        self.console.float_over(center)
        # what the tab covers, the 2-D plots leave empty: legend row
        # and bottom axis end above it, whichever pane it lands on
        self.data_pane.reserve_bottom(
            self.console.tab.sizeHint().height() + 4)
        self.setCentralWidget(center)

        # Dragging the divider between the tree and the views has to
        # track the cursor. AnimatedDocks — on by default — eases every
        # dock geometry change, so mid-drag the divider trails the mouse
        # rather than following it. And the separator is 4 px by default,
        # a target you aim at rather than a thing you grab.
        self.setDockOptions(QMainWindow.DockOption.AllowNestedDocks
                            | QMainWindow.DockOption.AllowTabbedDocks)
        self.setStyleSheet(self.styleSheet()
                           + '\nQMainWindow::separator { width: 7px; '
                             'height: 7px; }\n')

        self.tree: ProjectTree = ProjectTree()
        # a drop on the tree is the same import as a drop anywhere else,
        # queued the same way — see _import_after_drop for why it is
        # queued at all
        self.tree.files_dropped.connect(self._import_after_drop)
        self.tree.objects_moved.connect(self._move_objects)
        self.tree.landing_for = self._landing_of
        self.tree.write_for_drag = self._write_for_drag
        # column 1 carries the refresh badge of a stale object and the
        # edit pencil of a sub-row; the acts are on the bar
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(['Project', ''])
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.tree.setColumnWidth(1, 26)
        self.tree.itemClicked.connect(self._tree_item_clicked)
        self.tree.setIconSize(QSize(20, 20))
        self.tree.currentItemChanged.connect(self._selection_changed)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_tree_menu)
        self.tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        self.tree.setEditTriggers(
            QTreeWidget.EditTrigger.DoubleClicked
            | QTreeWidget.EditTrigger.EditKeyPressed)
        self.test_item: QTreeWidgetItem = QTreeWidgetItem(['Project'])
        self.test_item.setIcon(0, type_icon('Test'))
        self.test_item.setData(0, ROLE_ORIGINAL_NAME, 'Project')
        self.test_item.setData(0, ROLE_REFERENCE, ('test', None, None))
        # draggable *out* of the window, where it saves the whole
        # project; not ROLE_DRAGGABLE, which means movable between
        # link groups and the project belongs to none
        self.test_item.setData(0, ROLE_WHOLE_PROJECT, True)
        self.test_item.setFlags(self.test_item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.tree.addTopLevelItem(self.test_item)
        self.test_item.setExpanded(True)
        self.tree.setCurrentItem(self.test_item)

        # linking lives on a small bar above the tree, beside the same
        # actions on the context menu
        tree_bar = QToolBar('Links')
        tree_bar.setMovable(False)
        tree_bar.setIconSize(QSize(16, 16))
        self.link_action: QAction = QAction(control_icon('link'), '&Link', self)
        self.link_action.setToolTip(
            'Link the selected objects so they are known to belong '
            'together — shapes to their geometry')
        self.link_action.triggered.connect(self.link_selected)
        tree_bar.addAction(self.link_action)
        self.link_action.setVisible(False)
        self.unlink_action: QAction = QAction(control_icon('unlink'), '&Unlink',
                                     self)
        self.unlink_action.setToolTip(
            'Take the selected objects out of their link')
        self.unlink_action.triggered.connect(self.unlink_selected)
        tree_bar.addAction(self.unlink_action)
        self.unlink_action.setVisible(False)
        # Everything in the dock takes a file drag the way the tree
        # does. The dock's chrome — the title bar, the toolbar, the
        # holder around the tree — accepted nothing, and a drag refused
        # there is left to Qt walking up to the window, which through a
        # dock's widget stack is exactly the walk that misses (the
        # tree's module docstring records it). On a fresh window the
        # tree is a strip about 90 px wide, so most of the left side
        # *was* chrome: importing by drop worked on the right side of
        # the window and only sometimes on the left, which reads as a
        # broken feature aimed at the one place that names the project.
        tree_holder = _TakesFileDrops()
        tree_layout = QVBoxLayout(tree_holder)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_layout.setSpacing(0)
        tree_layout.addWidget(tree_bar)
        tree_layout.addWidget(self.tree)
        tree_holder.files_dropped.connect(self._import_after_drop)
        # The tree is not a view of the project, it is how you say what
        # the views are of: every pane answers to a selection made here,
        # and there is only ever the one window, so floating it out would
        # leave an empty one behind. It stays — movable between edges,
        # never floating, never closed.
        self.project_dock: QDockWidget = _FileDropDock('Project')
        self.project_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.project_dock.files_dropped.connect(self._import_after_drop)
        self.project_dock.setWidget(tree_holder)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea,
                           self.project_dock)
        # the dock follows its content: expanding a 20-average grid widens
        # it instead of showing one column of the grid through a slot

        self.active_geometry_action: QAction = QAction('Set as &Active Geometry', self)
        # never `connect(self.set_active_geometry)`: `triggered` passes
        # its checked bool, which landed in `name` as False — not None —
        # so the verb skipped the current-item lookup, looked up the
        # object called False, and refused. The menu entry did nothing,
        # ever, and said "Select a geometry" to a selected geometry.
        self.active_geometry_action.triggered.connect(
            lambda _checked=False: self.set_active_geometry())
        self.edit_action: QAction = QAction('&Edit', self)
        self.edit_action.triggered.connect(self.edit_entities)
        # Save As lives on the object's context menu and acts on that
        # one object; the File menu saves only the whole project. It
        # asks for the format in the dialog's own file-type list, which
        # is where a save dialog has always asked, so there is no second
        # verb for the foreign formats.
        self.save_action: QAction = QAction('&Save As...', self)
        self.save_action.setToolTip(
            'Write this object — as a .vdyn, or in a format another '
            'tool can read')
        self.save_action.triggered.connect(self.save_selected)
        self.export_report_action: QAction = QAction('Export &Report...', self)
        self.export_report_action.setToolTip(
            'Write this report as one self-contained HTML file — opens '
            'in any browser, nothing to install')
        self.export_report_action.triggered.connect(self.export_report)
        self.save_test_action: QAction = QAction('Save &Project As...', self)
        self.save_test_action.setToolTip(
            'Save the whole project — every object, under its name — as '
            'one .vdyn file')
        self.save_test_action.triggered.connect(self.save_test)
        self.delete_action: QAction = QAction('&Delete Selected', self)
        # fires while focus is in the tree, the table or the 3D view — picking
        # in the 3D view selects table rows, so delete has to reach it too.
        # WithChildren: the panes hold focus in a child (a viewport, or VTK's
        # render widget), never themselves.
        self.delete_action.setShortcuts([QKeySequence.StandardKey.Delete,
                                         QKeySequence(Qt.Key.Key_Backspace)])
        self.delete_action.setShortcutContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.delete_action.triggered.connect(self.delete_selected)
        self.tree.addAction(self.delete_action)
        self.scene.addAction(self.delete_action)
        # deliberately not on the table: there, Delete clears the selected
        # cells like it does in every table. It is added back only while
        # editing geometry, where the rows *are* the entities and removing
        # them is what Delete has always meant. See _set_table_model.
        # Escape and Return belong to a geometry editing session: one
        # abandons it, the other makes a traceline or an element from
        # the nodes picked so far. They were **window** shortcuts, and a
        # window shortcut is answered before the focused widget sees the
        # key — so Return typed into any editor anywhere in the window
        # was swallowed by a verb that was usually a no-op. Renaming a
        # row could not be finished with the Return key; it had to be
        # clicked away from, which reads as an editor that will not
        # close.
        #
        # Scoped to the two places a picking session actually lives, so
        # an editor keeps its own keys. The tree is not one of them.
        for action, keys, verb in (
                ('escape_action', [QKeySequence(Qt.Key.Key_Escape)],
                 self.stop_editing),
                ('commit_action', [QKeySequence(Qt.Key.Key_Return),
                                   QKeySequence(Qt.Key.Key_Enter)],
                 self._commit_picked_nodes)):
            made = QAction('Stop editing' if verb is self.stop_editing
                           else 'Create from picked nodes', self)
            made.setShortcuts(keys)
            made.setShortcutContext(
                Qt.ShortcutContext.WidgetWithChildrenShortcut)
            made.triggered.connect(verb)
            # the render window holds focus in a child of its own, which
            # is why it is WithChildren rather than WidgetShortcut
            self.scene.addAction(made)
            self.table.addAction(made)
            setattr(self, action, made)
        # Copy and Paste on the tree alone: the tables keep their own
        # spreadsheet clipboard, and a copy from the tree is the
        # selection as objects (paste back: duplicates) and as .vdyn
        # files (paste into a folder: an export) — Brandon, 2026-09-03
        self.copy_action: QAction = QAction('&Copy', self)
        self.copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        self.copy_action.setShortcutContext(
            Qt.ShortcutContext.WidgetShortcut)
        self.copy_action.triggered.connect(self.copy_selected)
        self.tree.addAction(self.copy_action)
        self.paste_action: QAction = QAction('&Paste', self)
        self.paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        self.paste_action.setShortcutContext(
            Qt.ShortcutContext.WidgetShortcut)
        self.paste_action.triggered.connect(self.paste_objects)
        self.tree.addAction(self.paste_action)
        self.tree.objects_pasted.connect(self._paste_named_objects)
        self.rename_action: QAction = QAction('&Rename', self)
        self.rename_action.setShortcut(QKeySequence(Qt.Key.Key_F2))
        self.rename_action.setShortcutContext(
            Qt.ShortcutContext.WidgetShortcut)
        self.rename_action.triggered.connect(self.rename_selected)
        self.tree.addAction(self.rename_action)
        self.tree.itemChanged.connect(self._item_renamed)
        self.tree.itemExpanded.connect(self._populate_entities)
        self.tree.setMouseTracking(True)
        self.tree.itemEntered.connect(self._hover_item)

        self.unit_combo: QComboBox = QComboBox()
        self.unit_combo.addItems(list(SYSTEMS))
        self.unit_combo.setCurrentText(DEFAULT_SYSTEM.name)
        self.unit_combo.currentTextChanged.connect(self._units_changed)
        # right-justified in the status bar, keeping the chrome minimal
        self.statusBar().addPermanentWidget(QLabel('Display units:'))
        self.statusBar().addPermanentWidget(self.unit_combo)

        # deliberately spare: import in, project out. Per-object save,
        # export and delete live on the object's own context menu.
        file_menu = self.menuBar().addMenu('&File')
        file_menu.addAction('&Import...', self.import_files)
        file_menu.addAction(self.save_test_action)
        report_menu = file_menu.addMenu('Generate &Report')
        report_menu.addAction('&Modal Test',
                              lambda: self.generate_report('modal'))
        report_menu.addAction('&Random Vibration',
                              lambda: self.generate_report('random'))
        report_menu.addAction('&Transient',
                              lambda: self.generate_report('transient'))
        report_menu.addAction('&Shock',
                              lambda: self.generate_report('shock'))
        report_menu.addAction('&Empty',
                              lambda: self.generate_report('empty'))
        file_menu.addSeparator()
        # light, dark, or the platform's choice, remembered between
        # launches: a Linux desktop Qt could not read left a friend of
        # Brandon's with a light window and no way to change it
        # (2026-09-14). One menu, still — File stays the only one.
        appearance = file_menu.addMenu('&Appearance')
        group = QActionGroup(self)
        group.setExclusive(True)
        self.appearance_actions: dict[str, QAction] = {}
        remembered = remembered_appearance()
        for choice, label in (('system', '&System'), ('light', '&Light'),
                              ('dark', '&Dark')):
            action = appearance.addAction(label)
            action.setCheckable(True)
            action.setChecked(choice == remembered)
            action.triggered.connect(
                lambda _checked=False, choice=choice:
                self.choose_appearance(choice))
            group.addAction(action)
            self.appearance_actions[choice] = action
        file_menu.addAction('Check for &Updates...', self.check_for_updates)
        # the About role: macOS moves it into the application menu,
        # where a Mac user looks for a version; Windows and Linux keep
        # it here (Brandon, 2026-09-12: no easy way to see the version)
        about = file_menu.addAction('&About Visual Dynamics...', self.about)
        about.setMenuRole(QAction.MenuRole.AboutRole)
        file_menu.addSeparator()
        file_menu.addAction('&Quit', self.close)

        self._show_status('Import a file to get started')
        # How far along an import is, beside the 'Importing …' text —
        # *beside* it, on the left, which takes more than a layout
        # choice: a temporary status message obscures the status bar's
        # normal (left) widgets, so a permanent widget on the right was
        # the only place a bar survived showMessage. Instead the import
        # brings its own strip — the text and the bar in one left-hand
        # widget — and while it runs, status lines go onto the strip's
        # label rather than through showMessage (see _show_status).
        # The bar itself is hidden except while there is something
        # honest to show: a project file reports per object and a
        # multi-file drop per file, but a single foreign file is one
        # unreported read, and a busy-bar over a blocked event loop
        # cannot animate — a frozen busy-bar reads as a hang, which is
        # worse than no bar.
        self._import_progress: QProgressBar = QProgressBar()
        self._import_progress.setMaximumWidth(200)
        self._import_progress.setTextVisible(False)
        self._import_label: QLabel = QLabel()
        strip = QWidget()
        strip_row = QHBoxLayout(strip)
        strip_row.setContentsMargins(6, 0, 0, 0)
        strip_row.setSpacing(8)
        # bar first, words after: the leftmost thing on the screen is
        # the one moving, and the label reads as its caption
        strip_row.addWidget(self._import_progress)
        strip_row.addWidget(self._import_label)
        self.statusBar().insertWidget(0, strip)
        self._import_strip: QWidget = strip
        strip.hide()
        self._import_progress.hide()

        hints = QApplication.instance().styleHints()
        if hasattr(hints, 'colorSchemeChanged'):
            # a bound method, never a lambda: the style hints live as
            # long as the application, and a lambda capturing self
            # would keep every closed window alive with its project —
            # the 15 MB-a-window leak that grew CI's workers to 2.7 GB
            # (2026-09-13). PySide drops a bound-method connection when
            # its receiver is destroyed.
            hints.colorSchemeChanged.connect(self._scheme_changed)
        # and once now: the panes colored themselves at construction,
        # but the tree's palette only exists in apply_theme — without
        # this it wore the platform's gray until the OS switched theme
        self.apply_theme(self.theme_name)

    def _stepper(self, toolbar, box, what):
        """A value box on the bar, wearing its own up/down arrows.

        It used to be flanked by two toolbar arrows instead, with Qt's
        stacked spinner turned off, on the argument that the stacked
        pair is a pixel-hunt beside a number reached for this often.
        Brandon put the wavelet panel's plain boxes beside these and
        preferred the plain ones (2026-08-28): the arrows belong to the
        number, and two toolbar buttons that happen to sit either side
        of a box only look like they do.

        Fewer parts, too. The flanking pair had to be shown and hidden
        with the box, which is three things to keep in step where the
        animation controls appear and disappear — and `_mode_actions`
        existed only to hold them together.
        """
        del what                     # the box's own arrows need no words
        box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return toolbar.addWidget(box)

    def _drive_points_for(self, series):
        """{name: diagonal record indices} when the button applies, else None.

        One filter, two names: the diagonal of an FRF grid is its drive
        points, and the diagonal of a CPSD grid is its autospectra. It
        applies when everything selected is one of those kinds and the
        diagonal is a proper subset — a filter that selects nothing, or
        everything, is not a legitimate option, and controls that do not
        apply are absent.

        Computed over the whole object, not the current selection, so the
        button works from any starting subset.
        """
        if not series or not (
                all(isinstance(data, Frf) for _n, data, _r in series)
                or all(isinstance(data, Psd) for _n, data, _r in series)):
            return None
        diagonals = {}
        off_diagonal = False
        for name, data, _records in series:
            if name not in self.objects:
                continue        # a synthesis overlay follows its original
            if data.reference_dof is None:      # a plain PSD is all ASDs
                diagonals[name] = list(range(data.num_records))
            else:
                diagonals[name] = [
                    i for i in range(data.num_records)
                    if data.response_dof[i] == data.reference_dof[i]]
            off_diagonal |= len(diagonals[name]) < data.num_records
        if not diagonals or not any(diagonals.values()) or not off_diagonal:
            return None
        return diagonals

    def _toggle_drive_points(self, checked):
        """Select the diagonal of each FRF grid, or give the selection back.

        The grid's cell selection *is* the record selection, so filtering by
        selecting is what keeps the tree, the grid and the plot telling one
        story — the diagonal lights up blue in the grid when the button goes
        down, exactly as if it had been picked by hand.
        """
        _kinds, series = self._current_series()
        diagonals = self._drive_points_for(series)
        if diagonals is None:
            return
        for name, records in diagonals.items():
            self._select_records(name, records if checked else [])
        self.render_current()

    def _current_series(self):
        """(kinds, series) for what is selected right now, as render_current
        would build it — one entry per data object, records coalesced."""
        series = []
        kinds = set()
        for kind, name, obj, detail in self.selected_references():
            kinds.add(kind)
            if not isinstance(obj, DataArray):
                continue
            if kind == 'record':
                for entry_name, _obj, records in series:
                    if entry_name == name and records is not None:
                        records.append(detail)
                        break
                else:
                    series.append((name, obj, [detail]))
            else:
                series.append((name, obj, None))
        return kinds, series

    def _select_records(self, name, records):
        """Make `records` the selection for `name`, grid or flat list alike.

        An empty list means the whole object — the same convention as an
        empty grid selection."""
        item = self._item_for_object(name)
        if item is None:
            return
        grid = self.record_grids.get(name)
        self.tree.blockSignals(True)
        if not item.isSelected():
            item.setSelected(True)
        if grid is not None:
            grid.select_records(records)
        else:
            for i in range(item.childCount()):
                child = item.child(i)
                reference = child.data(0, ROLE_REFERENCE)
                child.setSelected(bool(records) and reference is not None
                                  and reference[0] == 'record'
                                  and reference[2] in records)
        self.tree.blockSignals(False)

    def _map_wanted(self, series):
        """True for a map, False for curves, None when there is no choice.

        A whole coherence object is hundreds of channels, which the flat
        plot cannot show as lines — they overlay into a band and most are
        not even drawn. The map was the answer, and it still is when the
        flat reading is chosen; but the waterfall shows every channel at
        once too, and it is the resting default, so a checked 3D toggle
        takes the whole-object case (Brandon's call). An explicit click
        on Map outranks everything, as every explicit choice does.
        """
        from ..core.data import COHERENCE_TYPES

        if not series or not all(isinstance(data, COHERENCE_TYPES)
                                 for _name, data, _records in series):
            return None
        if self.data_pane.plot_mode is not None:
            return self.data_pane.plot_mode == 'map'
        # only for a selection the waterfall can actually take — two
        # coherence objects at once still read best as a map
        if self.data_pane.showing_waterfall and len(series) == 1:
            return False
        return any(records is None for _name, _data, records in series)

    # ---- the project the window shows -------------------------------------

    @property
    def objects(self) -> Project:
        """The project's objects: a mapping of name to object."""
        return self.project

    @property
    def links(self) -> list:
        return self.project.links

    @links.setter
    def links(self, groups: list[dict[str, Any]]) -> None:
        self.project.links = list(groups)

    @property
    def project_type(self) -> str | None:
        return self.project.project_type

    @project_type.setter
    def project_type(self, value: str | None) -> None:
        self.project.project_type = value

    @property
    def active_geometry(self) -> str | None:
        return self.project.active_geometry

    @active_geometry.setter
    def active_geometry(self, value: str | None) -> None:
        self.project.active_geometry = value

    def _build_view_toolbar(self, toolbar):
        """Everything on the 3D view's bar that needs the project.

        The pane's own bar arrives carrying the annotations it owns; what
        is added here — which DOFs to mark, what a click builds, what
        plays — all reads the project or the selection, so it belongs to
        the window and not to a view.
        """
        # labeled arrows at the DOFs the project measures; the
        # drop-down picks which quantity's DOFs are marked
        self.dofs_action: QAction = QAction(control_icon('dofs'), 'DOF arrows', self)
        self.dofs_action.setCheckable(True)
        self.dofs_action.setToolTip(
            "Mark measurement DOFs with labeled arrows; the drop-down "
            "picks which quantity's DOFs to show")
        self.dofs_action.toggled.connect(self._dofs_toggled)
        toolbar.addAction(self.dofs_action)
        self.dofs_action.setVisible(False)
        self._dof_series = []
        self.dofs_combo: QComboBox = QComboBox()
        self.dofs_combo.setToolTip("Which quantity's DOFs to mark")
        self.dofs_combo.currentIndexChanged.connect(
            lambda _index: self.render_current())
        self.dofs_combo_action: QAction = toolbar.addWidget(self.dofs_combo)
        self.dofs_combo_action.setVisible(False)

        # across geometries the comparison's second set animates on its
        # own mesh by default; this switches it to its projection on
        # the basis mesh — it lives here with the animation it changes
        self.project_action: QAction = QAction(control_icon('project'),
                                      'Animate the Projection', self)
        self.project_action.setCheckable(True)
        self.project_action.setToolTip(
            'Animate the other set projected onto the basis geometry, '
            'instead of on its own geometry')
        self.project_action.triggered.connect(
            lambda _checked: self.render_current())
        toolbar.addAction(self.project_action)
        self.project_action.setVisible(False)

        # Every action below appears only when it is a real option — see
        # the interface note in PLAN.md. A toolbar of grayed-out buttons is
        # clutter; one that only offers what applies is a menu of the
        # possible.
        # Editing lives on the tree: each geometry category row wears a
        # pencil in the icon column, and clicking it toggles the edit
        # table. This action carries the toggle state (and the keyboard
        # of habits wired to it) without sitting on any toolbar.
        self.edit_toggle_action: QAction = QAction(control_icon('edit'), '', self)
        self.edit_toggle_action.setCheckable(True)
        self.edit_toggle_action.toggled.connect(self._edit_toggled)

        self.add_action: QAction = QAction(control_icon('add'), '', self)
        self.add_action.setCheckable(True)
        self.add_action.setToolTip('Add items by clicking in the view')
        self.add_action.toggled.connect(self.set_add_mode)
        self._add_action_handle = toolbar.addAction(self.add_action)
        self.add_action.setVisible(False)

        # what a click builds while adding elements; only on screen when
        # adding elements, since nothing else is built from a node count
        self.element_type_group: QActionGroup = QActionGroup(self)
        self.element_type_group.setExclusive(True)
        self.element_type_actions: dict[str, QAction] = {}
        for kind, label in (('beam', 'Beam (2 nodes)'),
                            ('tri', 'Triangle (3 nodes)'),
                            ('quad', 'Quadrilateral (4 nodes)')):
            action = QAction(control_icon(kind), '', self)
            action.setCheckable(True)
            action.setToolTip(f'Add elements as: {label}')
            action.setChecked(kind == 'tri')
            self.element_type_group.addAction(action)
            toolbar.addAction(action)
            action.setVisible(False)
            action.triggered.connect(
                lambda _checked, k=kind: self._element_type_chosen(k))
            self.element_type_actions[kind] = action

        self.rotate_action: QAction = QAction(control_icon('rotate'), '', self)
        self.rotate_action.setCheckable(True)
        self.rotate_action.setToolTip(
            'Turn the selected coordinate system by dragging a ring')
        self.rotate_action.toggled.connect(self.set_rotate_mode)
        toolbar.addAction(self.rotate_action)
        self.rotate_action.setVisible(False)

        self.reset_rotation_action: QAction = QAction(control_icon('reset'), '', self)
        self.reset_rotation_action.setToolTip(
            'Put the coordinate system back to no rotation')
        self.reset_rotation_action.triggered.connect(self.reset_rotation)
        toolbar.addAction(self.reset_rotation_action)
        self.reset_rotation_action.setVisible(False)

        self.angle_box: DoubleSpinBox = DoubleSpinBox()
        self.angle_box.setRange(-360.0, 360.0)
        self.angle_box.setDecimals(2)
        self.angle_box.setSingleStep(5.0)
        self.angle_box.setSuffix('\u00b0')
        self.angle_box.setToolTip('Angle to turn to, in degrees')
        commit_on_enter(self.angle_box)
        self.angle_box.valueChanged.connect(self._angle_typed)
        self._angle_actions = (self._stepper(toolbar, self.angle_box,
                                             'angle'),)
        for action in self._angle_actions:
            action.setVisible(False)

        self._animation_actions = [toolbar.addSeparator()]
        self.colormap_action: QAction = QAction(control_icon('colormap'), '', self)
        self.colormap_action.setCheckable(True)
        self.colormap_action.setChecked(True)   # deflection is colored by
        self.colormap_action.setToolTip(        # default; the toggle is off
            'Color the model by how far each node moves')
        self.colormap_action.toggled.connect(self._colormap_toggled)
        toolbar.addAction(self.colormap_action)
        self._animation_actions.append(self.colormap_action)
        # slower, play, pause, faster — the transport, in the order a
        # media player puts it, with the pair that starts and stops kept
        # together and the two that change the rate flanking them
        self.slower_action: QAction = QAction(control_icon('slower'), '', self)
        self.slower_action.triggered.connect(lambda: self._change_speed(-1))
        toolbar.addAction(self.slower_action)
        self._animation_actions.append(self.slower_action)

        self.play_action: QAction = QAction(control_icon('play'), '', self)
        self.play_action.setToolTip('Play the animation')
        self.play_action.setEnabled(False)
        self.play_action.triggered.connect(lambda: self.set_playing(True))
        toolbar.addAction(self.play_action)
        self._animation_actions.append(self.play_action)

        self.pause_action: QAction = QAction(control_icon('pause'), '', self)
        self.pause_action.setToolTip('Pause the animation')
        self.pause_action.setEnabled(False)
        self.pause_action.triggered.connect(lambda: self.set_playing(False))
        toolbar.addAction(self.pause_action)
        self._animation_actions.append(self.pause_action)

        self.faster_action: QAction = QAction(control_icon('faster'), '', self)
        self.faster_action.triggered.connect(lambda: self._change_speed(1))
        toolbar.addAction(self.faster_action)
        self._animation_actions.append(self.faster_action)
        self._speed_index = NORMAL_SPEED
        self._restate_speed()

        self._mode_label_action = toolbar.addWidget(QLabel(' Mode '))
        self._animation_actions.append(self._mode_label_action)
        self.mode_box: SpinBox = SpinBox()
        self.mode_box.setRange(1, 1)
        self.mode_box.setToolTip('Which mode shape to animate')
        commit_on_enter(self.mode_box)
        self.mode_box.valueChanged.connect(self._mode_changed)
        self._mode_action = self._stepper(toolbar, self.mode_box, 'mode')
        self._mode_actions = (self._mode_action,)
        self._animation_actions.extend(self._mode_actions)

        self._animation_actions.append(toolbar.addWidget(QLabel(' Scale ')))
        self.scale_box: DoubleSpinBox = DoubleSpinBox()
        self.scale_box.setRange(0.01, 1000.0)
        self.scale_box.setValue(1.0)
        self.scale_box.setSingleStep(0.5)
        self.scale_box.setDecimals(2)
        self.scale_box.setToolTip('Multiplies the automatic deflection scale')
        commit_on_enter(self.scale_box)
        self.scale_box.valueChanged.connect(self._scale_changed)
        self._scale_actions = (self._stepper(toolbar, self.scale_box,
                                             'scale'),)
        self._animation_actions.extend(self._scale_actions)

        self.phase_slider: QSlider = QSlider(Qt.Orientation.Horizontal)
        self.phase_slider.setRange(0, PHASE_STEPS - 1)
        self.phase_slider.setMaximumWidth(180)
        self.phase_slider.setToolTip('Phase through the mode shape')
        self.phase_slider.valueChanged.connect(self._phase_changed)
        self._phase_action = toolbar.addWidget(self.phase_slider)
        self._animation_actions.append(self._phase_action)
        self._show_animation_controls(False)

        self._timer = QTimer(self)
        self._timer.setInterval(1000 // FRAMES_PER_SECOND)
        self._timer.timeout.connect(self._advance)

    # ---- animation ----------------------------------------------------------

    def _show_animation_controls(self, visible, phase=False):
        """Play and Scale only exist while something can be animated."""
        for action in self._animation_actions:
            action.setVisible(visible)
        self._phase_action.setVisible(visible and phase)
        # the mode chooser is only meaningful for a set with several modes
        modes = (self._shape_source[1].num_shapes
                 if visible and phase and self._shape_source else 0)
        for action in (self._mode_label_action, *self._mode_actions):
            action.setVisible(bool(modes > 1))

    def _caption_scene(self, text):
        """Name what is being animated, in the corner of the 3D view.

        Replaced by name rather than added, so stepping through modes does
        not stack captions on top of each other.
        """
        if self.scene.plotter is None:
            return
        self.scene.plotter.remove_actor('scene-caption', render=False)
        if text:
            self.scene.plotter.add_text(
                text, position='upper_left', font_size=11,
                color=resolve_theme(self.theme_name)['scene_text'],
                name='scene-caption')

    def _colormap_toggled(self, _checked):
        """Coloring is built into the scene, so switching rebuilds it."""
        if self.animator is not None:
            self.render_current()

    def _build_animator(self, geometry, deflection, caption='', showing=None,
                        mirrored=False):
        """Replace the static scene with one whose nodes can move.

        `caption` names what is being animated, in the corner of the view.
        It is an argument rather than a separate call because building the
        scene clears it: every rebuild has to say what the caption is now.

        `showing` is the name of the object being animated, and follows the
        same rule as the static renderer: framing is for new content. Left
        out, the camera is kept — stepping to another mode of the same
        geometry must not throw away the view the user set up.

        `mirrored` builds the ± pair instead — two translucent copies of
        the geometry sharing one deflection, the envelope's picture.
        """
        from ..viz.animate import (
            EnvelopeAnimator,
            GeometryAnimator,
            PairedAnimator,
        )

        colors = resolve_theme(self.theme_name)
        reframe = showing is not None and (showing,) != self._framed
        if showing is not None:
            self._framed = (showing,)
        camera = self.scene.plotter.camera_position
        self.scene.plotter.clear()
        self.scene.plotter.set_background(colors['scene_background'],
                                    top=colors['scene_background_top'])
        if mirrored:
            from ..viz.geometry import add_geometry

            # the undeflected geometry first, faint, as the zero
            # reference the two extremes are read against
            self._envelope_reference = []
            add_geometry(self.scene.plotter, geometry,
                         unit_system=self.unit_system, opacity=0.25,
                         color_override=colors['scene_muted'],
                         node_size=4.0, line_width=1.0,
                         text_color=colors['scene_text'],
                         meshes=self._envelope_reference)
            copies = [EnvelopeAnimator(
                self.scene.plotter, geometry, deflection, sign=sign,
                unit_system=self.unit_system,
                text_color=colors['scene_text'],
                colormap=self.colormap_action.isChecked(),
                opacity=0.6) for sign in (1.0, -1.0)]
            self.animator = PairedAnimator(*copies)
        else:
            self.animator = GeometryAnimator(
                self.scene.plotter, geometry, deflection,
                unit_system=self.unit_system, text_color=colors['scene_text'],
                colormap=self.colormap_action.isChecked())
        self.animator.set_scale(self.scale_box.value())
        self.scene.axis_unit = self.animator.axis_unit
        annotate_scene(self.scene.plotter, self.scene.axis_unit, colors,
                       self.scene.bounds_visible, self.scene.orientation_visible)
        self._caption_scene(caption)
        self._decorate_rigid()
        if reframe:
            self.scene.plotter.reset_camera()
        else:
            self.scene.plotter.camera_position = camera
        self.scene.plotter.render()

    def _clear_animator(self):
        self.set_playing(False)
        self._cursor = None
        self._cursor_dimension = None
        self._cursor_abscissa = None
        self._frame_index = 0
        # the same position, unrounded: see `_advance`
        self._frame_position = 0.0
        self._phase_position = 0.0
        self.play_action.setEnabled(False)
        self.pause_action.setEnabled(False)
        self._show_animation_controls(False)
        self.animator = None
        self._shape_source = None
        self._rigid_preview = None
        self._envelope_reference = []

    def set_playing(self, playing: bool) -> None:
        """Start or stop the animation."""
        playing = bool(playing) and self.animator is not None
        self._playing = playing
        if playing:
            self._timer.start()
        else:
            self._timer.stop()
        self.play_action.setEnabled(self.animator is not None and not playing)
        self.pause_action.setEnabled(playing)

    @property
    def speed(self) -> float:
        """How fast the animation runs, as a multiple of the normal rate."""
        return SPEEDS[self._speed_index]

    def _change_speed(self, direction):
        """One rung up or down the ladder, and say where it landed.

        Said out loud because the buttons cannot show it: a speed is a
        number, and two arrow glyphs are not a readout. The tooltips carry
        it too, for anyone who arrives at the button rather than at the
        status line.
        """
        wanted = self._speed_index + direction
        if not 0 <= wanted < len(SPEEDS):
            self._show_status(
                f'Already at {"the fastest" if direction > 0 else "the slowest"}'
                f' — {self.speed:g}x')
            return
        self._speed_index = wanted
        self._restate_speed()
        self._show_status(f'Animation speed {self.speed:g}x')

    def _restate_speed(self):
        """The buttons' tooltips, and whether either is at its end."""
        self.slower_action.setToolTip(
            f'Slower — currently {self.speed:g}x')
        self.faster_action.setToolTip(
            f'Faster — currently {self.speed:g}x')
        self.slower_action.setEnabled(self._speed_index > 0)
        self.faster_action.setEnabled(self._speed_index < len(SPEEDS) - 1)

    def _advance(self):
        """One frame; skipped if the previous render is still going.

        The position is carried as a float and only rounded when it is
        used. Stepping an integer instead cannot go slower than one step
        a frame — `max(1, ...)` sees to that — so every speed under about
        a half came out the same, which is precisely the range anyone
        reaches for Slower to get to.
        """
        if self.animator is None or self._rendering:
            return
        if self._shape_mode:
            step = PHASE_STEPS / (SECONDS_PER_CYCLE * FRAMES_PER_SECOND)
            self._phase_position = (
                (self._phase_position + step * self.speed) % PHASE_STEPS)
            # wraps, so playback loops
            self.phase_slider.setValue(int(self._phase_position))
        else:
            self._advance_cursor()

    def _phase_changed(self, step):
        if self.animator is None:
            return
        # dragging the slider is the same statement about position, so the
        # accumulator follows it — unless this *is* the accumulator's own
        # write, where rounding back would lose the fraction it carries
        if int(self._phase_position) != step:
            self._phase_position = float(step)
        self._draw_frame(step / PHASE_STEPS * 2 * np.pi)

    def _mode_changed(self, number):
        """Animate a different mode without disturbing phase or scale."""
        if self._shape_source is None:
            return
        geometry, shape_set = self._shape_source
        index = max(0, min(int(number) - 1, shape_set.num_shapes - 1))
        phase = self.phase_slider.value()
        self._build_animator(
            geometry, ShapeDeflection(geometry, shape_set.coordinate,
                                      shape_set.shape_matrix[index]),
            caption=self._shape_caption(shape_set, index))
        self._draw_frame(phase / PHASE_STEPS * 2 * np.pi)
        self._show_status(self._shape_caption(shape_set, index))

    def _shape_caption(self, shape_set, index):
        """What the corner of the view calls the mode being played:
        the set's own label, or — while the rigid-body reading
        previews a set that does not exist yet — which of the six it
        is and that it is a preview."""
        if self._rigid_preview is None:
            return shape_set.mode_label(index)
        return (f'Rigid body mode {index + 1} of {shape_set.num_shapes} — '
                f'{shape_set.description[index]} (preview)')

    def _scale_changed(self, value):
        if self.animator is not None:
            self.animator.set_scale(value)
            self._render_animation()

    def _draw_frame(self, parameter):
        self.animator.set_parameter(parameter)
        self._render_animation()

    def _render_animation(self):
        self._rendering = True
        try:
            self.animator.render()
        finally:
            self._rendering = False

    def _add_plot_cursor(self, data):
        """A draggable line marking the abscissa the geometry is showing.

        Mixed quantities stack one plot per row, and the cursor rides
        the plot of the quantity being deflected — an envelope showing
        force must not leave its cursor on the acceleration axes.
        """
        import pyqtgraph as pg

        from ..core.report import base_quantity

        plot = None
        if self._cursor_dimension:
            row = 0
            while (candidate := self.data_pane.graphics.getItem(
                    row, 0)) is not None:
                key = getattr(candidate, 'series_key', None)
                if key and base_quantity(key[1]) == self._cursor_dimension:
                    plot = candidate
                    break
                row += 1
        if plot is None:
            plot = self.data_pane.graphics.getItem(0, 0)
        if plot is None:
            return
        self._cursor_abscissa = np.asarray(
            data.display_abscissa(self.unit_system))
        # a line-picking cursor (ODS, envelope) starts where its
        # deflection already is — the strongest line — not at the left
        # edge of the plot
        deflection = (self.animator.deflection
                      if self.animator is not None else None)
        start = (deflection.line
                 if isinstance(deflection,
                               (OdsDeflection, EnvelopeDeflection))
                 else 0)
        self._cursor = pg.InfiniteLine(
            pos=float(self._cursor_abscissa[start]), angle=90, movable=True,
            pen=pg.mkPen(resolve_theme(self.theme_name)['scene_highlight'],
                         width=1))
        plot.addItem(self._cursor)
        self._cursor.sigPositionChanged.connect(self._cursor_moved)

    def _cursor_moved(self, line=None):
        if self.animator is None or self._cursor_abscissa is None:
            return
        index = int(np.clip(
            np.searchsorted(self._cursor_abscissa, self._cursor.value()),
            0, len(self._cursor_abscissa) - 1))
        deflection = self.animator.deflection
        if self._shape_mode and isinstance(deflection, OdsDeflection):
            # the cursor picks which frequency deflects; the phase the
            # sweep is at stays where it was
            deflection.line = index
            self._caption_scene(
                f'ODS — {self._cursor_abscissa[index]:.5g} Hz')
            self._draw_frame(
                self.phase_slider.value() / PHASE_STEPS * 2 * np.pi)
            return
        if isinstance(deflection, EnvelopeDeflection):
            self._caption_scene(
                f'Envelope — {self._cursor_abscissa[index]:.5g} Hz')
        self._frame_index = index
        # Dragging the cursor moves the playback position with it — but
        # only when it is a *drag*. Setting it unconditionally rounds the
        # accumulator to a whole sample on every frame, which throws away
        # the fraction it exists to carry: every speed then ran slow, and
        # an eighth ran at a fifth of what it asked for.
        if int(self._frame_position) != index:
            self._frame_position = float(index)
        self._draw_frame(index)

    def _advance_cursor(self):
        """Step the cursor so a record plays in about SECONDS_PER_RECORD
        at normal speed, and that over `speed` at any other."""
        if self._cursor is None or self._cursor_abscissa is None:
            return
        total = len(self._cursor_abscissa)
        step = total / (SECONDS_PER_RECORD * FRAMES_PER_SECOND)
        self._frame_position = (
            (self._frame_position + step * self.speed) % total)
        self._cursor.setValue(
            float(self._cursor_abscissa[int(self._frame_position)]))

    def _dofs_toggled(self, _checked):
        self.render_current()

    def _update_dof_controls(self, geometries, series):
        """The DOF-arrows toggle is an option only when a geometry and
        the data to read DOFs from are selected together; the quantity
        drop-down offers what that selection actually measures."""
        from ..core.report import series_dof_quantities

        self._dof_series = list(series) if geometries and series else []
        # a geometry on its own can mark its DOFs too — every node's
        # own X, Y and Z. There is no measurement to read a quantity
        # from, so the drop-down has nothing to offer and stays away;
        # what is drawn is the geometry's own axes rather than what
        # something happened to measure along them.
        offered = bool(self._dof_series) or bool(geometries)
        self.dofs_action.setVisible(offered)
        # an envelope in view chooses a quantity even with the arrows
        # off — the deflection itself is per-quantity, and one combo
        # serves both rather than two saying the same thing
        envelope = (len(geometries) == 1 and len(self._dof_series) == 1
                    and isinstance(self._dof_series[0][1], Psd))
        quantities = (series_dof_quantities(self._dof_series)
                      if self._dof_series
                      and (self.dofs_action.isChecked() or envelope)
                      else [])
        previous = self.dofs_combo.currentData()
        self.dofs_combo.blockSignals(True)
        self.dofs_combo.clear()
        for quantity in quantities:
            self.dofs_combo.addItem(quantity.title(), quantity)
        if previous in quantities:
            self.dofs_combo.setCurrentIndex(quantities.index(previous))
        self.dofs_combo.blockSignals(False)
        self.dofs_combo_action.setVisible(bool(quantities))

    def _scheme_changed(self, _scheme) -> None:
        """The platform switched light and dark: restate the theme —
        which follows it only while the appearance is System — and
        make every widget re-read the palette, which they do not do
        on their own (gui/preferences.py, `refresh_palettes`)."""
        refresh_palettes()
        self.apply_theme()

    def choose_appearance(self, choice: str) -> None:
        """File → Appearance: remember the choice and wear it now.
        'system' means follow the platform again."""
        remember_appearance(choice)
        for name, action in self.appearance_actions.items():
            action.setChecked(name == choice)
        # the application first — chrome, menus, native widgets — then
        # the parts drawn here; Qt announces the switch through
        # colorSchemeChanged too, which lands on apply_theme as well
        wear_appearance(choice=choice)
        self.apply_theme()
        self._show_status(
            'Following the system appearance' if choice == 'system'
            else f'{choice.capitalize()} appearance, remembered')

    def apply_theme(self, name: str | None = None) -> None:
        """Adopt a light/dark theme (default: the one chosen — this
        launch's flag, the remembered appearance, or the OS)."""
        import pyqtgraph as pg

        self.theme_name = name or chosen_scheme()
        colors = resolve_theme(self.theme_name)
        _keep_selection_vivid(self.table)
        pg.setConfigOption('foreground', colors['plot_foreground'])
        self.data_pane.apply_theme(self.theme_name, colors)
        # the scene's own theme name, not just its background: told
        # only to repaint, it repainted in the theme it was built with,
        # and a window opened light stayed white after Dark (Brandon's
        # screenshot, 2026-09-14)
        self.scene.apply_theme(self.theme_name)
        # the tree wears the scene's own ground — one theme for every
        # surface, black or white, never the platform's gray (Brandon,
        # 2026-08-23)
        palette = self.tree.palette()
        palette.setColor(QPalette.ColorRole.Base,
                         QColor(colors['scene_background']))
        palette.setColor(QPalette.ColorRole.Text,
                         QColor(colors['scene_text']))
        self.tree.setPalette(palette)
        _keep_selection_vivid(self.tree)
        if self.report is not None:
            self._paint_compatibility()
        self.render_current()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        # after super(): the style polishes the widget on first show and
        # assigns it a palette, discarding anything set in __init__
        _keep_selection_vivid(self.table)
        if self.scene.create_plotter() is not None:
            self.render_current()
        # once, so it opens fitting its content; after that the width is
        # the user's and nothing here touches it again
        self._size_tree_once()
        # With the project tree floating, an interactive resize on macOS
        # can land on the native window without reaching the widget tree,
        # stranding the status bar mid-window with dead space below it.
        # The QWindow still hears about it, so listen there and restate.
        handle = self.windowHandle()
        if handle is not None and not getattr(self, '_window_synced', False):
            self._window_synced = True
            handle.widthChanged.connect(self._sync_to_window)
            handle.heightChanged.connect(self._sync_to_window)

    def _sync_to_window(self, *_args):
        """Restate the widget tree from the native window's size when the
        two disagree — cheap and idempotent when they already agree."""
        handle = self.windowHandle()
        if handle is None:
            return
        if handle.size() != self.size():
            self.resize(handle.size())
        if (self.statusBar().isVisible()
                and self.statusBar().geometry().bottom() < self.height() - 1
                and self.layout() is not None):
            self.layout().invalidate()
            self.layout().activate()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Shut every VTK view down before Qt destroys the widgets.

        Without this, closing the window after a 3D view has rendered
        segfaults: the render window is torn down with its GL context
        still live. All three of them, not just the scene — the stage
        and the MAC bars have the same interactor, the same GL context
        and the same timers, and the scene-only version left them to
        take their chances (Brandon's crash report, 2026-08-30).
        """
        self.set_playing(False)
        plotters = [self.scene.plotter, self.data_pane.waterfall_plotter,
                    self._mac_bars_plotter_obj]
        self.scene.plotter = None
        self.data_pane.waterfall_plotter = None
        self._mac_bars_plotter_obj = None
        for plotter in plotters:
            if plotter is not None:
                with contextlib.suppress(Exception):  # never block closing
                    plotter.close()
        super().closeEvent(event)

    # ---- project management -------------------------------------------------

    def add_object(self, name: str, obj: Any, alternate: str | None = None,
                   source: str | None = None,
                   key: Any = None) -> str:
        """Add an object, keeping names unique.

        `alternate` is tried first on a clash. `source` is the file it was
        read from and `key` what that file's reader called it; both go on the
        tooltip, because the name carries neither.
        """
        if name in self.objects and alternate:
            name = alternate
        return self.show_object(self.project.add(name, obj),
                                source=source, key=key)

    def show_object(self, name: str, source: str | None = None,
                    key: Any = None, select: bool = True) -> str:
        """Give an object already in the project its row in the tree.

        The store and the showing are separate: a project verb — a
        computation, a fit, a projection — puts the result in the
        project, and this is how the window catches up with it.
        """
        obj = self.objects[name]
        item = QTreeWidgetItem([name])
        item.setIcon(0, object_icon(obj, units_defined=_units_defined(obj)))
        self.object_sources[name] = source
        self.object_keys[name] = key
        item.setToolTip(0, self._object_tooltip(name, obj))
        self._mark_computable(item, obj)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        # what the tree reads as "this row is an object", and so is
        # draggable between link groups: sub-items are parts of one
        item.setData(0, ROLE_DRAGGABLE, True)
        item.setData(0, ROLE_ORIGINAL_NAME, name)  # name before an edit
        item.setData(0, ROLE_REFERENCE, ('object', name, None))
        self._build_children(item, obj, name)
        self.test_item.addChild(item)
        self.test_item.setExpanded(True)
        self.refresh_compatibility()
        self._reorder_tree()    # canonical type order, then placeholders
        self._apply_type_links([name])
        self._paint_links()
        # a new object may be what a report figure was waiting for
        self._report_content_changed(settling=True)
        if select:
            self.tree.setCurrentItem(item)
            self.tree.clearSelection()
            item.setSelected(True)
        return name

    def _build_children(self, item, obj, name):
        """List what an object contains, so it can be expanded and picked."""
        self.record_grids.pop(name, None)   # the old one goes with the old rows
        item.takeChildren()
        if isinstance(obj, Geometry):
            for label, attribute, component in GEOMETRY_PARTS:
                # every category is listed, empty or not, so anything can be
                # selected and added to
                count = len(getattr(obj, attribute))
                child = QTreeWidgetItem([f'{label} ({count})'])
                child.setIcon(0, child_icon(component, 'Geometry',
                                            obj.units_defined, not count))
                child.setData(0, ROLE_REFERENCE, ('component', name, component))
                child.setData(0, ROLE_POPULATED, not count)
                child.setFlags(child.flags() & ~Qt.ItemFlag.ItemIsEditable)
                # the pencil: click it to open (or close) the edit table
                child.setIcon(1, control_icon('edit'))
                child.setToolTip(1, f'Edit {label.lower()} in a table')
                if count:
                    # entities are listed only when the branch is opened
                    child.setChildIndicatorPolicy(
                        QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
                item.addChild(child)
        elif isinstance(obj, (DataArray, ShapeSet, ChannelTable, Photos,
                              MatchedModes, SineSweepSpecification,
                              SineLevelSet)):
            # one format for every object that expands into sub-items: the
            # grid, even one column wide. One set of habits — the same
            # selection, the same deletion, the same icons — and no flat
            # list of 6859 rows to freeze the tree.
            self._build_record_grid(item, obj, name)

    def _populate_entities(self, item):
        """List a category's individual entities the first time it opens."""
        reference = item.data(0, ROLE_REFERENCE)
        if reference is None or item.data(0, ROLE_POPULATED):
            return
        kind, name, component = reference
        if kind != 'component':
            return
        geometry = self.objects.get(name)
        if geometry is None:
            return
        item.setData(0, ROLE_POPULATED, True)
        labels = list(self._entity_labels(geometry, component))
        self.tree.blockSignals(True)
        for entity_kind, detail, label in labels[:MAX_LISTED_ENTITIES]:
            child = QTreeWidgetItem([label])
            child.setIcon(0, child_icon(component, 'Geometry',
                                        geometry.units_defined))
            child.setData(0, ROLE_REFERENCE, (entity_kind, name, detail))
            child.setFlags(child.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.addChild(child)
        if len(labels) > MAX_LISTED_ENTITIES:
            note = QTreeWidgetItem(
                [f'… {len(labels) - MAX_LISTED_ENTITIES} more not listed'])
            note.setFlags(Qt.ItemFlag.NoItemFlags)
            item.addChild(note)
        self.tree.blockSignals(False)
        self._paint_compatibility()
        if not labels:
            item.setChildIndicatorPolicy(
                QTreeWidgetItem.ChildIndicatorPolicy.DontShowIndicator)

    @staticmethod
    def _entity_labels(geometry, component):
        """(reference kind, detail, label) for each entity in a category."""
        if component == 'nodes':
            for node_id in geometry.node_id:
                yield 'node', int(node_id), f'Node {int(node_id)}'
        elif component == 'coordinate_systems':
            for index, cs_id in enumerate(geometry.cs_id):
                cs_name = geometry.cs_name[index].strip()
                kind = CS_TYPES.get(int(geometry.cs_type[index]), 'cartesian')
                yield 'coordinate_system', int(cs_id), (
                    f'CS {int(cs_id)}' + (f' — {cs_name}' if cs_name else '')
                    + f' ({kind})')
        elif component == 'tracelines':
            for index, conn in enumerate(geometry.traceline_conn):
                label = f'Traceline {int(geometry.traceline_id[index])}'
                description = geometry.traceline_desc[index].strip()
                yield 'traceline', index, (
                    f'{label} ({len(conn)} nodes)'
                    + (f' — {description}' if description else ''))
        elif component == 'elements':
            for index, conn in enumerate(geometry.elem_conn):
                type_name = ELEMENT_TYPES[int(geometry.elem_type[index])][0]
                yield 'element', index, (
                    f'Element {int(geometry.elem_id[index])} '
                    f'({type_name}, {len(conn)} nodes)')
        elif component == 'blocks':
            for index, block_id in enumerate(geometry.block_id):
                name = geometry.block_name[index].strip()
                held = int((geometry.elem_block == int(block_id)).sum())
                yield 'block', int(block_id), (
                    f'Block {int(block_id)}' + (f' — {name}' if name else '')
                    + f' ({held} elements)')

    def _build_record_grid(self, item, obj, name):
        """Expand a matrix of records into a grid instead of a list.

        The grid goes in a single spanned child row, so the tree keeps its
        one column and nothing else in it has to know about grids.
        """
        holder = QTreeWidgetItem()
        holder.setData(0, ROLE_REFERENCE, ('grid', name, None))
        holder.setFlags(Qt.ItemFlag.ItemIsEnabled)
        item.addChild(holder)
        holder.setFirstColumnSpanned(True)
        grid = RecordGrid(obj, self.tree)
        _keep_selection_vivid(grid)
        grid.owner = name       # kept in step by rename, so no captured name
        grid.selection_changed.connect(self._grid_selected)
        grid.row_renamed.connect(
            lambda row, text, grid=grid: self._grid_row_renamed(
                grid, row, text))
        grid.column_renamed.connect(
            lambda column, text, grid=grid: self._apply_dof_rename(
                grid.owner, *grid.column_keys[column], text))
        # the grid is a widget, so the tree's context menu never sees a
        # right-click inside it; without this, records have no route to
        # the units pane
        grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        grid.customContextMenuRequested.connect(
            lambda position, grid=grid: self._show_grid_menu(grid, position))
        holder.setSizeHint(0, grid.preferred_size())
        self.tree.setItemWidget(holder, 0, grid)
        self.record_grids[name] = grid

    def _adding_to_selection(self):
        """Is the multi-select modifier down as this selection is made?

        What a tree reads for itself and a grid cannot: the pick arrives
        as a selection-changed signal with no event behind it, so the
        keyboard is the only place left to ask. Control covers Command
        on macOS, which Qt maps to it.
        """
        held = QApplication.keyboardModifiers()
        return bool(held & (Qt.KeyboardModifier.ControlModifier
                            | Qt.KeyboardModifier.ShiftModifier
                            | Qt.KeyboardModifier.MetaModifier))

    def _emptied_beside_others(self, grid, item):
        """Did this grid just go empty while another data object is up?

        The two halves matter equally. Empty, because a grid with rows
        in it is asking for those rows. Beside others, because the last
        object on the plot going empty means "show all of it" and not
        "show nothing" — that is what selecting an object has always
        done, and there would be nothing left to look at.
        """
        if grid.selected_records():
            return False
        if not item.isSelected():
            return False
        for other in self.tree.selectedItems():
            if other is item:
                continue
            reference = other.data(0, ROLE_REFERENCE)
            if reference is None:
                continue
            if isinstance(self.objects.get(reference[1]), DataArray):
                return True
        return False

    def _grid_selected(self):
        """Picking cells in a grid is picking that object's records.

        The tree only has to point at the owner; which records are wanted is
        already in the grid, and `selected_references` reads it from there.
        """
        self.data_pane.reset_plot_mode()
        grid = self.sender()
        item = self._item_for_object(grid.owner)
        if item is not None and self._emptied_beside_others(grid, item):
            # Taking the last row off a grid means take mine off the
            # plot — but only while something else is on it. An empty
            # grid otherwise means the whole object, which is what a
            # click on the object itself has always shown, and what a
            # single selection has to keep meaning.
            #
            # Without this the deselection reads as "all of my records",
            # and a specification then cuts the PSD beside it back to
            # the pairs it bounds — the same curves as before, so the
            # plot answers a deselection by not changing.
            self.tree.blockSignals(True)
            item.setSelected(False)
            self.tree.blockSignals(False)
            self.render_current()
            return
        adding = self._adding_to_selection()
        if not adding:
            # A plain pick reads alone (Brandon, 2026-08-30): every
            # other grid's sub-items give way, whatever object holds
            # them — multi-selection is the modifier's job. This runs
            # whether or not this grid's own item was already selected;
            # the old guard skipped everything when it was, and two
            # objects browsed together then kept both their picks
            # through plain clicks.
            for other_grid in self.record_grids.values():
                if other_grid is not grid and other_grid.selected_records():
                    other_grid.select_records([])
        if item is not None:
            self.tree.blockSignals(True)     # one render, not three
            # Other *data* objects give way — grids compete for the plot —
            # but a geometry or a shape set stays selected: records picked
            # in a grid animate on a selected geometry, and clearing the
            # whole selection here is what made that impossible. Mode picks
            # give way to nothing: modes beside FRFs synthesize a
            # prediction of them, which needs both selected at once.
            #
            # And nothing gives way while the multi-select modifier is
            # held. Each grid is its own widget with its own selection,
            # so Qt never tells one that a pick in another was meant to
            # add rather than replace; without reading the modifier here
            # a specification and the PSD it bounds — the comparison a
            # random vibration test exists to make — could not both be
            # put on the plot.
            owner_is_shapes = isinstance(
                self.objects.get(grid.owner), ShapeSet)
            for other in list(self.tree.selectedItems()):
                reference = other.data(0, ROLE_REFERENCE)
                if reference is None or other is item:
                    continue
                # everything but the animation pair gives way: a report
                # or a photo set lingering in the selection used to be
                # cleared by the focus bug this rule replaced, so the
                # give-way owns that job now — not only data arrays
                lingering = self.objects.get(reference[1])
                if (not owner_is_shapes and not adding
                        and not isinstance(lingering, (Geometry, ShapeSet))):
                    other.setSelected(False)
            if not item.isSelected():
                item.setSelected(True)
                self.tree.setCurrentItem(
                    item, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
            self.tree.blockSignals(False)
        self.render_current()

    def current_reference(self) -> tuple[str | None, Any, Any]:
        """(kind, object, index-or-component) for the selected tree item."""
        item = self.tree.currentItem()
        if item is None:
            return None, None, None
        reference = item.data(0, ROLE_REFERENCE)
        if reference is None:
            return None, None, None
        kind, name, detail = reference
        return kind, self.objects.get(name), detail

    def current_object(self) -> Any:
        """The object owning the selection (a child resolves to its parent)."""
        return self.current_reference()[1]

    def selected_references(self) -> list[tuple[str, str, Any, Any]]:
        """[(kind, name, object, detail)] for every selected tree item."""
        out = []
        for item in self.tree.selectedItems():
            reference = item.data(0, ROLE_REFERENCE)
            if reference is None:
                continue
            kind, name, detail = reference
            if kind == 'placeholder':   # a gray slot holds nothing yet
                continue
            if kind == 'test':  # the whole test: everything it holds
                out.extend(('object', key, value, None)
                           for key, value in self.objects.items())
                continue
            obj = self.objects.get(name)
            if obj is None:
                continue
            grid = self.record_grids.get(name) if kind == 'object' else None
            records = grid.selected_records() if grid is not None else []
            if records:
                # the grid's own selection is the sub-item selection — there
                # is nowhere else for the two to disagree — and its kind
                # keeps the tree's vocabulary: record, channel or mode
                out.extend((grid.kind, name, obj, i) for i in records)
            else:
                out.append((kind, name, obj, detail))
        return out

    def object_item(self,
                    item: QTreeWidgetItem | None = None
                    ) -> QTreeWidgetItem | None:
        """The object-level item for a selection (walks up to the test)."""
        item = item or self.tree.currentItem()
        if item is self.test_item:
            return None
        while item is not None and item.parent() not in (None, self.test_item):
            item = item.parent()
        return item

    # kept for callers that predate the test root
    top_level_item = object_item

    def delete_selected(self) -> None:
        """Remove what is selected: entities while editing or when entity
        rows are picked in the tree, whole objects otherwise."""
        if self.editing is not None:
            return self.delete_entity_rows()
        entities = self._selected_entities()
        if entities:
            return self._delete_entities(*entities)
        emptied = self._selected_components()
        if emptied:
            return self._empty_components(*emptied)
        if any(item is self.test_item for item in self.tree.selectedItems()):
            self._show_status('Select an object inside the test to delete it')
            return
        # Sub-items go before objects: a record, mode or channel picked in
        # the tree or in a grid deletes that record, not — as it used to —
        # the whole object it lives in. An object-level pick still deletes
        # the object, and outranks sub-item picks of the same object.
        doomed, subitems = {}, {}
        for kind, name, obj, detail in self.selected_references():
            if kind == 'object':
                doomed[name] = obj
            elif kind in ('record', 'mode', 'channel', 'photo'):
                subitems.setdefault(name, (obj, kind, set()))[2].add(detail)
        if subitems:
            # Sub-item picks make the whole keystroke about sub-items. A
            # co-selected object is context, not a target — a geometry is
            # selected so records can animate on it, and deleting one grid
            # record must not take the geometry with it. That happened: one
            # record and the whole geometry went in a single keystroke.
            doomed = {}
        if not doomed and not subitems:
            return
        removed = []
        for name, (obj, kind, picks) in subitems.items():
            deleter = {'record': 'delete_records', 'mode': 'delete_modes',
                       'channel': 'delete_channels',
                       'photo': 'delete_photos'}[kind]
            line = self._deletion_line(name, obj, deleter, picks)
            try:
                getattr(obj, deleter)(picks)
            except ValueError as refusal:
                self._show_status(f'{name}: {refusal}')
                return
            self.project.journal.append(line)
            self._refresh_item(self._item_for_object(name), obj)
            removed.append(f'{len(picks)} {kind}'
                           + ('s' if len(picks) != 1 else '')
                           + f' from {name}')
        if doomed:
            # One selection change for the whole batch, not one per
            # removal: taking rows out one at a time promotes a
            # neighbor to current each time, and the handler
            # re-rendered each survivor's view only for the next
            # removal to kill it — the specification plot blinked once
            # per object on a delete-all (Brandon, 2026-08-31). The
            # links and placeholders settle once too, after the loop.
            with QSignalBlocker(self.tree):
                self.tree.clearSelection()
                self.tree.setCurrentItem(None)
                for name in doomed:
                    self._remove_object(name, settle=False)
            self._paint_links()
            self._refresh_placeholders()
            removed.extend(doomed)
        self.refresh_compatibility()
        self.render_current()
        self._show_status('Removed ' + ', '.join(removed))

    def _journal_geometry_delete(self, geometry, component, keys):
        """A geometry deletion, journaled by the ids a script names —
        stable across a replay where row numbers are not."""
        method = {'nodes': 'delete_nodes',
                  'coordinate_systems': 'delete_coordinate_systems',
                  'tracelines': 'delete_tracelines',
                  'elements': 'delete_elements',
                  'blocks': 'delete_blocks'}[component]
        self.project.record_call(geometry, method,
                                 sorted(int(k) for k in keys))

    def _journal_view(self, prefix, line):
        """A parameterized view, journaled as its headless plot call.

        Brandon's call (2026-08-30): the console speaks views as well
        as acts — every reading with parameters records the
        `visualdynamics.plot` line that renders it headless, settling
        in place per reading and object, so a session of nudging reads
        as the reading you ended on and a replay writes the figures
        the session looked at.
        """
        journal = self.project.journal
        at = next((i for i in range(len(journal) - 1, -1, -1)
                   if journal[i].startswith(prefix)), None)
        if at is not None:
            journal[at] = line
        else:
            journal.append(line)

    def _journal_report_edit(self):
        """A report edit, journaled as the state that stands.

        The editor's ops — a caption typed, a block inserted, moved or
        removed, the title or the marking changed — all land on the
        Report object, and the honest replay is its post-state: four
        settling assignments, each replacing its own last line, so an
        editing session reads as what the report became rather than a
        keystroke log.
        """
        editor = self.report_editor
        report = editor.report if editor is not None else None
        if report is None:
            return
        try:
            name = self.project.name_of(report)
        except (KeyError, ValueError):
            return
        journal = self.project.journal
        for attribute, value in (('blocks', list(report.blocks)),
                                 ('title', report.title),
                                 ('marking', report.marking),
                                 ('marking_color', report.marking_color)):
            prefix = f'project[{name!r}].{attribute} = '
            line = prefix + repr(value)
            at = next((i for i in range(len(journal) - 1, -1, -1)
                       if journal[i].startswith(prefix)), None)
            if at is not None:
                journal[at] = line
            else:
                journal.append(line)

    def _journal_table_edit(self, model, suffix):
        """Prefix a table edit's suffix with the object it edited — and
        for a channel table, redraw its grid: a row's coordinate *is*
        its node and direction, so a cell typed in the table view moves
        the coordinate in the tree the same instant (Brandon,
        2026-09-06: the two must always remain in step)."""
        try:
            name = self.project.name_of(model.obj)
        except (KeyError, ValueError):
            return          # a working table over no project object
        self.project.journal.append(f'project[{name!r}]{suffix}')
        if isinstance(model.obj, ChannelTable):
            item = self._item_for_object(name)
            grid = self.record_grids.get(name)
            picked = grid.selected_records() if grid is not None else []
            if item is not None:
                self._refresh_item(item, model.obj)
            grid = self.record_grids.get(name)
            if picked and grid is not None:
                grid.select_records(picked)

    def _deletion_line(self, name, obj, deleter, picks):
        """The journal line a deletion will deserve, computed BEFORE
        it runs — the semantic check reads the object as it stands,
        and after the delete the indices mean different records.

        Row indices as they stood at that moment are exactly what a
        sequential replay re-creates (Brandon, 2026-08-30: deleting
        two modes said nothing in the console). Where the picks are
        exactly a channel or a capture, the line says so instead —
        thirteen indices where one capture number carries the meaning
        read as the machine talking to itself (Brandon, the same day).
        """
        picks = sorted(int(p) for p in picks)
        if deleter == 'delete_records':
            said = self._semantic_deletion(obj, picks)
            if said is not None:
                return f'project[{name!r}]{said}'
        return f'project[{name!r}].{deleter}({picks!r})'

    @staticmethod
    def _semantic_deletion(obj, picks):
        """'.delete_records(dof=…)' or '(capture=…)' when the picks
        are exactly that — else None, and the indices stand."""
        if obj is None:
            return None
        chosen = set(picks)
        dofs = list(dict.fromkeys(obj.response_dof[i] for i in picks))
        if {i for i in range(obj.num_records)
                if obj.response_dof[i] in set(dofs)} == chosen:
            said = dofs[0] if len(dofs) == 1 else dofs
            return f'.delete_records(dof={said!r})'
        references = getattr(obj, 'reference_dof', None)
        if references is not None:
            # a column of the matrix — every record at these
            # reference DOFs (Brandon, 2026-08-30: an FRF column
            # deleted by twenty-two indices)
            refs = list(dict.fromkeys(references[i] for i in picks))
            if {i for i in range(obj.num_records)
                    if references[i] in set(refs)} == chosen:
                said = refs[0] if len(refs) == 1 else refs
                return f'.delete_records(reference={said!r})'
            # one cell, or a sub-block: rows and columns together
            if {i for i in range(obj.num_records)
                    if obj.response_dof[i] in set(dofs)
                    and references[i] in set(refs)} == chosen:
                dof_said = dofs[0] if len(dofs) == 1 else dofs
                ref_said = refs[0] if len(refs) == 1 else refs
                return (f'.delete_records(dof={dof_said!r}, '
                        f'reference={ref_said!r})')
        # a drive point carries two records at one DOF; when the DOF
        # alone would say too much, DOF plus quantity may say exactly
        # this (Brandon's question, 2026-08-30)
        dims = list(dict.fromkeys(obj.ordinate_dim[i] for i in picks))
        if {i for i in range(obj.num_records)
                if obj.response_dof[i] in set(dofs)
                and obj.ordinate_dim[i] in set(dims)} == chosen:
            dof_said = dofs[0] if len(dofs) == 1 else dofs
            dim_said = dims[0] if len(dims) == 1 else dims
            return (f'.delete_records(dof={dof_said!r}, '
                    f'dim={dim_said!r})')
        ordinals = getattr(obj, 'capture_indices', None)
        if ordinals is not None:
            ordinals = ordinals()
            captures = sorted({ordinals[i] for i in picks})
            if {i for i in range(obj.num_records)
                    if ordinals[i] in set(captures)} == chosen:
                said = captures[0] if len(captures) == 1 else captures
                return f'.delete_records(capture={said!r})'
        return None

    def _remove_object(self, name, settle=True):
        """Delete an object from the project and the tree.

        `settle=False` defers the links and placeholder repaint to the
        caller — a delete-all repaints once after the loop, not once
        per object."""
        self._forget_object_row(name)
        self.project.remove(name)      # prunes the links with it
        # a report being edited goes off the screen with its object. Asked
        # of the project *after* the removal rather than of the name being
        # removed: what matters is whether the thing on screen is still in
        # the project, and that is a question about the object.
        if (self.report_editor is not None
                and self.report_editor.report is not None
                and not any(obj is self.report_editor.report
                            for obj in self.objects.values())):
            self._forget_report_editor()
        if settle:
            self._paint_links()
            # ...and a figure bound to what just left goes with it
            self._report_content_changed(settling=True)
            self._refresh_placeholders()

    def _forget_object_row(self, name):
        """Take an object's row and widgets out of the tree, leaving
        the project alone — what merging needs, since the object is
        not going away so much as being replaced."""
        item = self._item_for_object(name)
        self.record_grids.pop(name, None)
        self.object_sources.pop(name, None)
        self.object_keys.pop(name, None)
        if item is not None:
            self.test_item.removeChild(item)

    # ---- links: objects explicitly declared to belong together ----------

    # ---- the acts: what a selection can be processed into ----------------

    #: the acts with no settings, as the bar shows them: the verb, its
    #: label, its glyph and the handler. Every act lives on the bar and
    #: nowhere else (Brandon, 2026-09-04): these used to be split
    #: between the tree's calculator column, the right-click menu and
    #: the bar, and a user could not predict which. What *applies* is
    #: the project's own answer — `Project.selection_verbs` holds the
    #: one applicability table, so a verb the API lists is the verb
    #: the bar offers and neither can drift. Everything a view
    #: parameterizes stays on that view's pane, principle 13.
    ACTS: ClassVar[tuple] = (
        ('integrate', 'Integrate', 'integrate', 'integrate_history'),
        ('differentiate', 'Differentiate', 'differentiate',
         'differentiate_history'),
        ('extract_sine', 'Extract Sine Levels', 'sine',
         'extract_sine_levels'),
        ('fit_modes', 'Fit Modal Model', 'edit', 'start_modal_fit'),
        ('transform', 'Transform to Modal Responses', 'transform',
         'transform_selection'),
        ('expand', 'Expand to Physical Responses', 'expand',
         'transform_selection'),
        ('project_onto_basis', 'Project onto Basis DOFs', 'project',
         'project_onto_basis'),
        ('merge', 'Merge into One', 'merge', 'merge_selected'),
    )

    def acts_for(self, names=None):
        """[(verb, label, icon, handler, tooltip)] the selection can
        take, in the bar's order.

        `names` is the selection; left out, it is read from the tree,
        a record pick counting as its object. The project row selected
        offers the one act the project itself has, its report; a stale
        object offers Recompute first, ahead of anything else it can do.
        """
        if names is None:
            if self.test_item.isSelected():
                return [('generate_report', 'Generate Report', 'report',
                         self.generate_report_act,
                         ('Generate the report this project\'s type '
                          'calls for'))]
            names = list(dict.fromkeys(
                name for kind, name, _obj, _detail
                in self.selected_references()
                if kind in ('object', 'record')))
        names = [name for name in names if name in self.objects]
        if not names:
            return []
        applicable = dict(self.project.selection_verbs(*names))
        acts = [(verb, label, icon, getattr(self, handler),
                 applicable[verb])
                for verb, label, icon, handler in self.ACTS
                if verb in applicable]
        if len(names) == 1 and names[0] in self._stale:
            acts.insert(0, ('refresh', 'Recompute', 'refresh',
                            self.refresh_selected,
                            (f'Settings changed: {self._stale[names[0]]}. '
                             'Recompute this object from its source')))
        return acts

    def _offer_acts(self):
        """Put the selection's acts on the bar that is up — the plot's
        when data is drawn, the 3-D view's otherwise — and take them
        off the other. Nothing during a fit, which owns the bar."""
        acts = [] if self.fit is not None else self.acts_for()
        plot_up = self.data_pane.isVisibleTo(self)
        self.data_pane.show_acts(acts if plot_up else [])
        self.scene.show_acts([] if plot_up else acts)

    def refresh_selected(self) -> None:
        """Recompute the selected stale object — the bar's Recompute,
        the same act as the badge's click."""
        item = self.object_item()
        if item is not None:
            self.refresh_object(item.text(0))

    def generate_report_act(self) -> None:
        """The bar's Generate Report: the typed report at once, or the
        choice of templates for an untyped project — and the templates
        the user has saved, beside the built-in ones, whenever there
        are any (2026-09-08)."""
        from ..io.report_template import saved_templates

        saved = saved_templates()
        if self.project_type and not saved:
            # the type already says which report this project is for,
            # so there is nothing to choose — one click, one report
            self.generate_typed_report()
            return
        menu = QMenu(self)
        if self.project_type:
            menu.addAction(f'Generate {self.project_type} Report',
                           self.generate_typed_report)
        else:
            for label, template in (('&Modal Test', 'modal'),
                                    ('&Random Vibration', 'random'),
                                    ('&Transient', 'transient'),
                                    ('&Shock', 'shock'),
                                    ('S&ine Sweep', 'sine'),
                                    ('S&ystem ID', 'sysid'),
                                    ('&Empty', 'empty')):
                menu.addAction(f'Generate {label} Report',
                               lambda _checked=False, t=template:
                               self.generate_report(t))
        if saved:
            menu.addSeparator()
            for label, path in saved:
                menu.addAction(f'Generate {label} Report (saved template)',
                               lambda _checked=False, t=str(path):
                               self.generate_report(t))
        self._pop_menu(menu)

    def _mark_computable(self, item, obj):
        """The refresh badge on an object whose source's settings have
        moved, with the mismatch in its tooltip; clicking it recomputes
        (Brandon, 2026-08-23 — recomputing is an act, never automatic;
        a report must not rewrite itself). The column carries nothing
        else: the acts an object can take are on the bar (Brandon,
        2026-09-04), and the badge stays here because it is status,
        and the tree is where the objects are."""
        name = item.data(0, ROLE_ORIGINAL_NAME)
        reason = self._stale.get(name)
        if reason is not None:
            item.setIcon(1, control_icon('refresh'))
            item.setToolTip(1, f'Settings changed: {reason}.\n'
                            'Click to recompute.')
        else:
            item.setIcon(1, QIcon())
            item.setToolTip(1, '')

    def _refresh_stale_badges(self):
        """Re-read what is stale and restate every object's badge."""
        self._stale = self.project.stale()
        for name, obj in self.objects.items():
            item = self._item_for_object(name)
            if item is not None:
                self._mark_computable(item, obj)
        # every settings write refreshes the badges, so this is where
        # the console keeps up with the journal's tail
        self.console.refresh(self.project.journal)

    def refresh_object(self, name: str) -> None:
        """Recompute a stale object in place — the badge's click."""
        reason = self._stale.get(name, '')
        try:
            self.project.refresh(name)
        except ValueError as refusal:
            self._show_status(f'{name}: {refusal}')
            return
        obj = self.objects[name]
        item = self._item_for_object(name)
        if item is not None:
            item.setToolTip(0, self._object_tooltip(name, obj))
            self._build_children(item, obj, name)
        self._refresh_stale_badges()
        self._report_content_changed()
        self.render_current()
        followed = [other for other in self._stale
                    if self.provenance_source(other) == name]
        told = (f' — {", ".join(followed)} now stale in turn'
                if followed else '')
        self._show_status(
            f'{name} recomputed'
            + (f' ({reason})' if reason else '') + told)

    def provenance_source(self, name: str) -> str | None:
        record = self.project.provenance.get(name)
        return record.get('source') if record else None

    def _report_content_changed(self, settling: bool = False) -> None:
        """Re-render the open report: something it draws from moved.

        The editor's own guard re-renders only when the Report object
        or the unit system changes — right for selection churn, wrong
        after a recompute or a settings edit, where the same Report
        must draw the new numbers (Brandon, 2026-08-23: Figure 5 kept
        the old averages through a refresh).

        `settling` says the change came from a gesture still under the
        mouse — a dragged corner, a dragged averaging span — and the
        rebuild waits for it to stop. **Measured, and it was the whole
        of the problem** (Brandon, 2026-08-25, on a corner that
        dragged slowly): a rebuild costs 254 ms on shock.vdyn, against
        3 ms to filter the record and 9 ms to restage it, so a drag
        was spending 96% of itself redrawing a document nobody was
        looking at yet. The report is right the moment the hand stops,
        which is the first moment anyone can read it.
        """
        if self.report_editor is None:
            return
        if self.report_editor.isHidden():
            # off screen (the editor's own hidden flag — isVisible would
            # also answer False for a window not yet shown, as in the
            # tests): remember, and rebuild when shown. The fit
            # that confirmed five modes into the same set one at a
            # time, and the matched set added beside it, both left the
            # page at its first render — one mark, two figures gone
            # (Brandon, 2026-09-03)
            self._report_settle.stop()
            self.report_editor.stale = True
            return
        if settling:
            self._report_settle.start(REPORT_SETTLE_MS)
            return
        self._report_settle.stop()
        self.report_editor.rebuild()

    def _tree_item_clicked(self, item, column):
        """The icon column: the pencil on a geometry's categories, and
        the refresh badge on a stale object. Nothing else lives here —
        the acts are on the bar (Brandon, 2026-09-04)."""
        if column != 1:
            return
        reference = item.data(0, ROLE_REFERENCE)
        if reference is not None and reference[0] == 'component':
            # the pencil is a toggle: editing this category closes it,
            # anything else opens (or switches to) its table
            if self.editing == (reference[1], reference[2]):
                self.stop_editing()
                return
            self.tree.setCurrentItem(item)
            self.tree.clearSelection()
            item.setSelected(True)
            if self.editing is not None:
                self.stop_editing()
            self.edit_entities()
            return
        if reference is None or reference[0] != 'object':
            return
        if self._stale.get(reference[1]) is None:
            return
        # the badge is one verb, so the click is that verb — a menu
        # between them was a question with one answer (Brandon,
        # 2026-08-23). The recompute reads the selection: point it at
        # this object
        self.tree.clearSelection()
        item.setSelected(True)
        self.tree.setCurrentItem(item)
        self.refresh_object(reference[1])

    def _pop_menu(self, menu) -> None:
        """Open a context menu at the cursor.

        Its own method because tests must intercept it: QMenu.exec is
        a C++ slot that no monkeypatch reaches, and under offscreen a
        real popup never returns — a test that would open one hangs
        the suite instead of failing."""
        menu.exec(QCursor.pos())

    def _selected_object_names(self):
        """Names of whole objects in the selection, in tree order."""
        names = []
        for kind, name, _obj, _detail in self.selected_references():
            if kind == 'object' and name not in names:
                names.append(name)
        return names

    def _group_of(self, name):
        """The link group dict holding `name`, or None."""
        return next((group for group in self.links
                     if name in group['members']), None)

    def linked_group(self, name: str) -> list[str] | None:
        """The linked member names beside `name`, or None."""
        group = self._group_of(name)
        return None if group is None else group['members']

    def link_role(self, name: str) -> str | None:
        """The role of `name`'s link group — 'Basis', or None for an
        unroled or unlinked object. The Basis group defines the DOF
        space comparisons happen in: other sets project onto its DOFs,
        its modes are the MAC rows and the frequency-error baseline."""
        group = self._group_of(name)
        return None if group is None else group['role']

    def set_link_role(self, name: str, role: str | None) -> None:
        """Declare a link group the Basis of comparisons — explicit,
        so nothing downstream has to guess which group is which."""
        group = self._group_of(name)
        if group is None:
            return
        # the role means one thing: taking it takes it from any other
        if role is not None:
            for other in self.links:
                if other is not group and other['role'] == role:
                    other['role'] = None
        group['role'] = role
        self._reorder_tree()      # the Basis group reads first
        self._paint_links()
        self._show_status(
            f'{", ".join(group["members"])}: '
            + ('marked as the Basis — comparisons happen in this '
               'group\'s DOFs' if role else 'Basis cleared'))

    def linked_geometry(self,
                        name: str) -> tuple[str, Geometry] | None:
        """(geometry name, geometry) linked with `name`, or None —
        explicit knowledge, never a guess."""
        members = self.linked_group(name)
        if members is None:
            return None
        return next(((member, self.objects[member])
                     for member in members
                     if isinstance(self.objects.get(member), Geometry)),
                    None)

    def link_selected(self) -> None:
        self._link_names(self._selected_object_names())

    def _link_names(self, names, announce=True):
        """Link the named objects. The project merges the groups and
        refuses what cannot be true — a second geometry, or DOFs the
        group's geometry lacks — and its refusal is what the status
        bar says. Returns whether the link happened."""
        if len(names) < 2:
            return False
        try:
            merged = self.project.link(*names)
        except (ValueError, KeyError) as refusal:
            # an exception message starts lower case; the status bar
            # is a sentence
            message = str(refusal).strip("'")
            self._show_status(message[:1].upper() + message[1:])
            return False
        self._links_changed()
        if announce:
            self._show_status('Linked ' + ', '.join(merged))
        return True

    def unlink_selected(self) -> None:
        """Take the selected objects out of their groups; a group left
        with one member dissolves."""
        names = self._selected_object_names()
        if not names:
            return
        self.project.unlink(*names)
        self._links_changed()
        self._show_status('Unlinked ' + ', '.join(names))

    def _trial_move(self, names, target):
        """Do the move, hand back what the links became, and put them
        back — so a question about a move is answered by the move.

        Asked twice: once per mouse tick while a drag is over the tree,
        to decide whether the drop is offered at all, and once when it
        lands. Both go through the same code, so the drag cannot promise
        something the drop then refuses. It is a few list copies against
        a handful of groups, not a computation.
        """
        def snapshot() -> list[dict[str, Any]]:
            return [dict(group) for group in self.links]

        before = snapshot()
        try:
            # quiet: this is a question, asked once per mouse tick
            # while a drag is over the tree, and it answers itself by
            # doing the move and rolling it back. Journaled, one drag
            # wrote eighty relink lines (Brandon, 2026-08-30); the
            # act's record is _apply_move's settling links echo.
            with self.project.journal_as(None):
                if isinstance(target, tuple):
                    for name in names:
                        self.project.place(name, self._slot_role(target[1]))
                else:
                    for name in names:
                        self.project.relink(name, target)
            return snapshot(), before, None
        except (ValueError, KeyError) as refusal:
            return None, before, refusal
        finally:
            self.project.links = before

    def _allowed_move(self, names, target):
        """The links after dropping these objects there, or None when
        the drop would be refused or would change nothing. One rule
        under the drag's landing line and the drop itself: a drop that
        does nothing is not offered, so dragging within one group
        shows no target."""
        names = [name for name in names if name in self.objects]
        if not names or target in names:
            return None
        if not isinstance(target, tuple) and target not in (None,
                                                            *self.objects):
            return None
        after, before, refusal = self._trial_move(names, target)
        if refusal is not None or after == before:
            return None
        return after

    def _can_move_objects(self, names, target):
        """Whether dropping these objects there would be allowed and
        would change anything."""
        return self._allowed_move(names, target) is not None

    def _move_objects(self, names, target):
        """Queue a dragged move for after the drag has unwound.

        Never straight to the move, for the same reason a dropped file
        is never imported straight: the drop is delivered from inside
        macOS's own drag run loop, and this one reparents tree items and
        rebuilds their record grids. Reparenting a widget from inside a
        nested loop is how the docking experiment segfaulted.
        """
        QTimer.singleShot(0, lambda: self._apply_move(list(names), target))

    def _apply_move(self, names, target):
        """Move dragged objects into the group holding `target`, into
        one of the type's named groups, or out of their groups when
        they were dropped on nothing.

        Every ending says something. A drag that would do nothing is
        never offered, so a drop that lands and changes nothing means
        the aim missed — and a tree that simply does not move is the
        one report there is no way to act on.
        """
        names = [name for name in names if name in self.objects]
        if not names:
            return
        # read before the move: afterwards a slot resolves to the group
        # the move has just put there, so it would report joining what
        # it created
        where = self._move_reads_as(names, target)
        is_are = 'is' if len(names) == 1 else 'are'
        listed = ', '.join(names)
        after, before, refusal = self._trial_move(names, target)
        if refusal is not None:
            # an exception message starts lower case; the status bar is
            # a sentence
            message = str(refusal).strip("'")
            self._show_status(message[:1].upper() + message[1:])
            return
        if after == before:
            self._show_status(f'{listed} {is_are} already {where}')
            return
        # commit by running the verbs the trial ran, unsuppressed this
        # time, so the journal records the move as the command a script
        # would write — place/relink/unlink — rather than restating the
        # whole links list (Brandon, 2026-08-30: the wholesale echo made
        # a one-object move read as if it needed every object listed)
        if isinstance(target, tuple):
            for name in names:
                self.project.place(name, self._slot_role(target[1]))
        elif target is not None:
            for name in names:
                self.project.relink(name, target)
        else:
            self.project.unlink(*names)
        self._links_changed()
        self._show_status(f'Moved {listed} {where}')

    def _move_reads_as(self, names, target):
        """How a move reads in the status bar, said in its own terms:
        a group by a member of it, a role by what that role is."""
        if isinstance(target, tuple):
            return f'into the {self._role_word(self._slot_role(target[1]))} group'
        if target is not None:
            return f'in with {target}'
        return ('out of its group' if len(names) == 1
                else 'out of their groups')

    #: how the type's named groups read in a sentence
    ROLE_WORDS: ClassVar[dict] = {'Basis': 'Basis', None: 'other',
                                  OTHER_SIDE: 'other'}

    def _role_word(self, role):
        return self.ROLE_WORDS.get(role, str(role))

    @staticmethod
    def _slot_role(tag):
        """A slot's tag as the project's role: the other side's tag
        is a word for the tree's sake, and the project calls it None."""
        return None if tag == OTHER_SIDE else tag

    def _links_changed(self):
        self._reorder_tree()
        self._paint_links()
        self._update_link_actions()
        # linking changes which geometry an object answers to
        self.refresh_compatibility()
        # and which objects the report's '@basis:' bindings mean. A
        # project file's Report renders the moment its row arrives —
        # selection follows each object as it is added — which is
        # *before* the file's own link groups are absorbed, so that
        # first page resolved every symbolic binding with no Basis and
        # showed the FEM set in every figure that meant the test's.
        # Nothing rebuilt it afterwards, because the editor was already
        # showing that report. The moment the links change is the
        # moment every symbolic binding may mean something else.
        if (self.report_editor is not None
                and self.report_editor.report in self.objects.values()):
            self.report_editor.rebuild()

    # the tree's canonical type order lives with the project, so the
    # tree and a Project's own repr cannot disagree about it
    TYPE_ORDER: ClassVar[tuple] = TYPE_ORDER

    def _type_rank(self, name):
        return type_rank(self.objects.get(name))

    def _object_order(self, links):
        """Every object's tree position under `links` — the one
        ordering rule, asked with the standing links by `_reorder_tree`
        and with a trial's outcome by the drag's landing line."""
        # groups read in the order the type lists their roles, so the
        # the Basis group is above the model one whether or not either
        # holds anything; a group with no role follows them
        linked = [name
                  for group in sorted(
                      links,
                      key=lambda g: (self._role_rank(g['role']),
                                     g['role'] != 'Basis'))
                  for name in sorted(
                      (n for n in group['members'] if n in self.objects),
                      key=self._type_rank)]
        # Once each, whatever the links say. An object listed in two
        # groups — which they must not be, and were — put its name in
        # this list twice, so the positions ran past the end of the
        # tree, and `insertChild` past the end **silently drops a child
        # that has already been taken out**. Seven rows disappeared from
        # a project that still held all forty-five objects: invisible in
        # the tree, present in the file, and saved again on the next
        # save. The invariant is enforced in the Project now; this stays
        # because the failure is silent data loss and the guard is one
        # `dict.fromkeys`.
        linked = list(dict.fromkeys(linked))
        return linked + sorted(
            (name for name in self.objects if name not in linked),
            key=self._type_rank)

    def _landing_of(self, names, target):
        """Where dropping these objects there would land them: the tree
        items each would sit *above* (None for the end of the objects),
        or None when the move would not happen at all.

        The outline this replaced wrapped the target and read as the
        object going *into* it (Brandon, 2026-08-30); the honest mark
        is a line at the row where the canonical order will actually
        put the object, asked of the same trial that answers whether
        the move is allowed."""
        after = self._allowed_move(names, target)
        if after is None:
            return None
        order, moved = self._object_order(after), set(names)
        landings = []
        for name in names:
            below = next((other for other in
                          order[order.index(name) + 1:]
                          if other not in moved), None)
            item = None if below is None else self._item_for_object(below)
            if item not in landings:
                landings.append(item)
        return landings

    def _reorder_tree(self):
        """Linked groups first, members adjacent; unlinked objects
        after; the gray placeholders keep the bottom. Everywhere —
        inside a group and out — types keep the canonical order, and
        objects of one type keep their arrival order."""
        wanted = self._object_order(self.links)
        selected = self.tree.selectedItems()
        current = self.tree.currentItem()
        moved = []
        for position, name in enumerate(wanted):
            item = self._item_for_object(name)
            if item is None:
                continue
            if self.test_item.indexOfChild(item) != position:
                self.test_item.takeChild(
                    self.test_item.indexOfChild(item))
                self.test_item.insertChild(
                    min(position, self.test_item.childCount()), item)
                moved.append((item, name))
        for item, name in moved:
            # reparenting drops item widgets: the record grid rebuilds
            self._build_children(item, self.objects[name], name)
        if moved and selected:
            # reparenting also drops the selection — put it back
            # (current first: setCurrentItem clears multi-selection)
            if current is not None:
                self.tree.setCurrentItem(current)
            self.tree.clearSelection()
            for item in selected:
                item.setSelected(True)
        self._refresh_placeholders()

    # the Basis keeps the comparison overlay's baseline blue; other
    # groups cycle the curve palette
    LINK_ROLE_COLORS: ClassVar[dict] = {'Basis': '#4c92d9'}

    def _role_holder(self, role):
        """The real link group holding the type's `role`.

        A group that was *declared* to hold this role answers first;
        only where nobody has said is the role worked out from the
        objects that satisfy its expectations, which is a guess — the
        first geometry imported could belong to either group.
        """
        if not self.project_type:
            return None
        if role is None:
            return None
        if role == OTHER_SIDE:
            # the other side has no name: it is the one group that is
            # not the Basis, when there is exactly one
            others = [g for g in self.links if g['role'] != 'Basis']
            return others[0] if len(others) == 1 else None
        from ..core.report import expectation_satisfiers

        declared = self.project.role_group(role)
        if declared is not None:
            return declared
        # sides, not roles: an object in a group on the other side is
        # not a candidate for the Basis, even when it is the project's
        # first geometry (the pair dragged across used to be guessed
        # straight back, once the other side stopped carrying a name)
        members = expectation_satisfiers(
            self.project_type, self.objects, self._sides()).get(role, [])
        return next((group for name in members
                     if (group := self._group_of(name)) is not None),
                    None)

    def _group_colors(self):
        """{id(group): color}, exactly as the brackets paint them —
        the Basis blue, other groups the rest of the curve palette."""
        colors, others = {}, 0
        for group in self.links:
            if group['role'] == 'Basis':
                colors[id(group)] = self.LINK_ROLE_COLORS['Basis']
            else:
                others += 1               # never the Basis blue
                colors[id(group)] = curve_color(others)
        return colors

    def _link_color(self, name):
        """The bracket color of `name`'s link group, or None."""
        group = self._group_of(name)
        return (None if group is None
                else self._group_colors()[id(group)])

    def _paint_links(self):
        """Real link groups get their brackets; the type's placeholder
        slots join their role's bracket too, so a fresh Modal project
        shows both groups before anything exists. The Basis is told
        apart by its blue, bold bracket alone — no text — and other
        groups take the rest of the curve palette."""
        placeholders = getattr(self, '_placeholder_items', {})
        tag_groups = {tag: self._role_holder(tag) for tag in placeholders}
        colors = self._group_colors()
        spans = []
        for group in self.links:
            items = [self._item_for_object(name)
                     for name in group['members'] if name in self.objects]
            for tag, owner in tag_groups.items():
                if owner is group:
                    items += placeholders[tag]
            spans.append((colors[id(group)], items,
                          group['role'] == 'Basis'))
        for tag, items in placeholders.items():
            # a side with no group yet is still a side: the Basis, or
            # the other one, which has no name — both bracketed, so a
            # fresh typed project shows its shape before anything
            # exists; slots on neither side (the report) get none
            if tag is not None and tag_groups.get(tag) is None:
                basis = tag == 'Basis'
                spans.append((self.LINK_ROLE_COLORS['Basis'] if basis
                              else '#6a6a72', items, basis))
        # top-down, because the painter offsets neighboring brackets by
        # their position in this list: out of order, two brackets that
        # meet on screen are drawn in the same column and read as one
        self.tree.link_spans = sorted(spans, key=self._span_top)

    def _span_top(self, span):
        rows = [self.test_item.indexOfChild(item) for item in span[1]
                if item is not None]
        rows = [row for row in rows if row >= 0]
        return min(rows) if rows else self.test_item.childCount()

    def _apply_type_links(self, arrivals=None):
        """The project type's structure includes the links: the objects
        that satisfy the Basis slots go in the Basis group, applied as
        they arrive. The other side has no name and is never guessed:
        a second geometry and shape set link by hand, or by their file. `arrivals` limits
        the pass to roles a just-added object belongs to, so an explicit
        Unlink or a drag is not fought until something new turns up.

        One object is placed as readily as several. It used to take two
        before a group existed at all, which left a lone FRF sitting
        in no group while the gray slot beside it still said an FRF was
        wanted — the object was in the tree and the skeleton denied it.
        """
        if not self.project_type:
            return
        from ..core.report import expectation_satisfiers

        placed, changed = self._sides(), False
        for role, members in expectation_satisfiers(
                self.project_type, self.objects, placed).items():
            if role != 'Basis':
                continue      # the other side is read, never placed into
            if arrivals is not None and not set(arrivals) & set(members):
                continue
            # An object already placed elsewhere is not a candidate for
            # this role. Without this the rules undo the drag: put the
            # model pair in the FEM group, import the measured FRF, and
            # they are pulled back into the Basis, because they are
            # still the project's first geometry and first shape set.
            # ...and 'elsewhere' is any other group, named or not: the
            # other side has no name since 2026-09-02, and the pair
            # dragged into it must count as placed all the same
            elsewhere = {name for g in self.links if g['role'] != role
                         for name in g['members']}
            members = [name for name in members if name not in elsewhere]
            here = set(placed.get(role, ()))
            members = [name for name in members if name not in here]
            if members:
                changed = self._place_members(members, role) or changed
        if changed:
            # the slots are computed from the placement, so they are
            # stale until the placing is done: an FRF placed here after
            # the tree was built left its own gray slot showing beside it
            self._reorder_tree()
        self._paint_links()

    def _place_members(self, names, role):
        """Put objects into a named group of the project's structure,
        saying once in the status bar if the project refuses — a second
        geometry, or DOFs that group's geometry lacks."""
        for name in names:
            try:
                self.project.place(name, role)
            except (ValueError, KeyError) as refusal:
                # an exception message starts lower case; the status bar
                # is a sentence
                message = str(refusal).strip("'")
                self._show_status(message[:1].upper() + message[1:])
                return False
        return True

    def _update_link_actions(self):
        names = self._selected_object_names()
        self.link_action.setVisible(len(names) >= 2)
        self.unlink_action.setVisible(
            any(self.linked_group(name) is not None for name in names))

    def set_project_type(self, project_type: str | None) -> None:
        """Declare what kind of test this project is — the tree then
        shows a gray slot for everything that kind of report expects.

        The type never pre-selects a reading: a time history opens
        plain whatever the project is (Brandon, 2026-08-30).
        """
        self.project_type = project_type or None
        # an act like any other, journaled as the assignment a script
        # makes — settling in place, since imports may restate it
        journal = self.project.journal
        line = f'project.project_type = {self.project_type!r}'
        if journal and journal[-1].startswith('project.project_type = '):
            journal[-1] = line
        elif self.project_type is not None:
            journal.append(line)
        self._refresh_placeholders()
        self._apply_type_links()
        if self.project_type:
            missing = missing_expectations(self.project_type, self.objects,
                                           self._sides())
            self._show_status(
                f'{self.project_type} project — '
                + (f'{len(missing)} expected '
                   f'object{"s" * (len(missing) != 1)} still missing'
                   if missing else 'everything a report needs is here'))
        else:
            self._show_status('Project type cleared')

    def _sides(self):
        """{side: [object names]} for the skeleton: the Basis by its
        role, and under OTHER_SIDE every member of every group that is
        not the Basis — the side has no name, so it is read off the
        groups rather than declared."""
        sides = dict(self.project.placed())
        others = [name for g in self.links if g['role'] != 'Basis'
                  for name in g['members']]
        if others:
            sides[OTHER_SIDE] = others
        return sides

    def _refresh_placeholders(self):
        """Gray slots under the project for whatever its type still
        expects, removed as the real objects arrive."""
        for i in reversed(range(self.test_item.childCount())):
            child = self.test_item.child(i)
            reference = child.data(0, ROLE_REFERENCE)
            if reference is not None and reference[0] == 'placeholder':
                self.test_item.removeChild(child)
        self._placeholder_items = {}
        if not self.project_type:
            self._paint_links()
            return
        gray = QBrush(QColor('#6a6a72'))
        for name, _cls, icon, _count, optional, group in \
                missing_expectations(self.project_type, self.objects,
                                     self._sides()):
            item = QTreeWidgetItem([name])
            item.setIcon(0, placeholder_icon(icon))
            item.setForeground(0, gray)
            item.setToolTip(
                0, (f'Optional for a {self.project_type} report: a '
                    'second geometry and shape set — a model, or '
                    'another test — enable correlation' if optional else
                    f'A {name} is expected for a {self.project_type} '
                    'report — right-click for options'))
            item.setData(0, ROLE_REFERENCE, ('placeholder', name, icon))
            # a slot is a drop target: dragging onto it is how an
            # object is placed where the type's rules guessed wrong —
            # onto the Basis, or onto the other side, which has no name
            item.setData(0, ROLE_DROP_SLOT, group)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._placeholder_items.setdefault(group, []).append(item)
        # Each role's slots stay adjacent to that group's real members,
        # so its bracket covers both — and a group holding *no* members
        # keeps its place in the run rather than falling to the bottom.
        # It used to fall: drag the model pair across and the measured
        # group, now empty, dropped below them, so the model group and
        # its bracket jumped to the top of the tree and read as the
        # Basis. Roles keep the order the type lists them in, full or
        # empty. Slots belonging to no role keep the bottom.
        position = 0
        for role in self._role_order():
            after = self._placeholder_position(role)
            if after is not None:
                position = after
            for item in self._placeholder_items.get(role, []):
                self.test_item.insertChild(position, item)
                position += 1
        for item in self._placeholder_items.get(None, []):
            self.test_item.addChild(item)
        self.test_item.setExpanded(True)
        self._paint_links()

    def _role_order(self):
        """The type's roles, in the order the type itself lists them.

        Read off the expectations rather than written down again here,
        so a new project type gets its reading order from the one place
        that says what it is made of.
        """
        if not self.project_type:
            return ()
        from ..core.report import project_expectations

        order = []
        for expectation in project_expectations(self.project_type):
            role = expectation[5]
            if role and role not in order:
                order.append(role)
        return tuple(order)

    def _role_rank(self, role):
        """Where a group sits in the tree: by its role, so a named
        group keeps its place whether or not it holds anything."""
        order = self._role_order()
        # a real group that is not the Basis sits where the other side
        # sits: the project calls that role None, the tree calls it other
        role = OTHER_SIDE if role is None else role
        return order.index(role) if role in order else len(order)

    def _placeholder_position(self, role):
        """Where a role's placeholder slots go: right after the last
        real member of the link group holding that role, else None —
        meaning wherever the run of roles has got to, which is that
        role's own place."""
        group = self._role_holder(role)
        if group is None:
            return None
        last = max((self.test_item.indexOfChild(self._item_for_object(n))
                    for n in group['members'] if n in self.objects),
                   default=-1)
        return None if last < 0 else last + 1

    def _selected_components(self):
        """(geometry name, components) when whole categories are selected."""
        picks = {}
        for kind, name, obj, detail in self.selected_references():
            if kind != 'component' or not isinstance(obj, Geometry):
                return None
            picks.setdefault(name, []).append(detail)
        if len(picks) != 1:
            return None
        return next(iter(picks.items()))

    def _empty_components(self, name, components):
        """Delete everything in the selected categories, leaving them empty."""
        geometry = self.objects.get(name)
        if geometry is None:
            return
        # ids, for all five — the same thing a row's delete names
        counts = {
            'nodes': lambda g: g.node_id.tolist(),
            'coordinate_systems': lambda g: g.cs_id.tolist(),
            'tracelines': lambda g: g.traceline_id.tolist(),
            'elements': lambda g: g.elem_id.tolist(),
            'blocks': lambda g: g.block_id.tolist(),
        }
        removed, refused = {}, []
        for component in components:
            keys = counts[component](geometry)
            if not keys:
                continue
            try:
                report = _delete_from_geometry(geometry, component, keys)
            except ValueError as e:
                refused.append(str(e))
                continue
            self._journal_geometry_delete(geometry, component, keys)
            for kind, count in report.items():
                removed[kind] = removed.get(kind, 0) + count
        item = self._item_for_object(name)
        if item is not None:
            self._refresh_item(item, geometry)
        self.refresh_compatibility()
        self.render_current()
        done = ', '.join(f'{count} {kind.replace("_", " ")}'
                         for kind, count in removed.items() if count)
        self._show_status(' — '.join(
            part for part in (f'Removed {done}' if done else '',
                              '; '.join(refused)) if part)
            or 'Nothing to remove')

    def _selected_entities(self):
        """(geometry name, component, keys) when the tree selection is
        entities of one geometry, else None."""
        picks = {}
        for kind, name, obj, detail in self.selected_references():
            component = ENTITY_COMPONENT.get(kind)
            if component is None or not isinstance(obj, Geometry):
                return None
            picks.setdefault((name, component), []).append(detail)
        if len(picks) != 1:
            return None
        (name, component), keys = next(iter(picks.items()))
        return name, component, keys

    def _delete_entities(self, name, component, keys):
        """Delete entities and report what went with them."""
        geometry = self.objects.get(name)
        if geometry is None:
            return
        try:
            report = _delete_from_geometry(geometry, component, keys)
        except ValueError as e:
            self._show_status(str(e))
            return
        self._journal_geometry_delete(geometry, component, keys)
        item = self._item_for_object(name)
        if item is not None:
            self._refresh_item(item, geometry)
        self.refresh_compatibility()
        if self.editing is not None:
            self._reload_edit_table()
        else:
            self.render_current()
        self._show_status('Removed ' + ', '.join(
            f'{count} {kind.replace("_", " ")}'
            for kind, count in report.items() if count) or 'Nothing to remove')

    def delete_entity_rows(self) -> None:
        """Delete the rows selected in the edit table."""
        if self.editing is None:
            return
        name, component = self.editing
        geometry = self.objects.get(name)
        rows = sorted({index.row() for index
                       in self.table.selectionModel().selectedIndexes()})
        if not rows or geometry is None:
            return
        keys = [_entity_key(geometry, component, row) for row in rows]
        self._delete_entities(name, component, keys)

    def _reload_edit_table(self):
        """Rebuild the edit table and view after the geometry changed."""
        name, component = self.editing
        geometry = self.objects.get(name)
        if geometry is None:
            self.stop_editing()
            return
        model = self._set_table_model(
            ENTITY_TABLES[component](geometry, self.unit_system, self))
        model.dataChanged.connect(self._edited_geometry)
        self.table.selectionModel().selectionChanged.connect(
            self._edit_selection_changed)
        self._draw_edit_selection()
        self._begin_picking(geometry, self._picking_component())

    def define_units(self) -> None:
        """Declare units for what is selected.

        With individual records selected, only those get units; selecting an
        object (or its category) covers all of its records.
        """
        targets = {}
        for kind, name, obj, detail in self.selected_references():
            if isinstance(obj, ChannelTable):
                continue
            if isinstance(obj, ShapeSet):
                targets.setdefault(name, {'object': obj, 'records': set(),
                                          'whole': True})['whole'] = True
                continue
            entry = targets.setdefault(name, {'object': obj, 'records': set(),
                                              'whole': False})
            if kind == 'record':
                entry['records'].add(detail)
            else:
                entry['whole'] = True
        if not targets:
            self._show_status(
                'Select an object or channels inside the test to define units')
            return

        # every kind of object declares its units the same way, in the pane
        name = next(iter(targets))
        entry = targets[name]
        records = (None if entry['whole'] or not entry['records']
                   else sorted(entry['records']))
        self.show_units_panel(name, entry['object'], records)
        if len(targets) > 1:
            # the pane shows one object; it stays open, so the rest are a
            # matter of selecting them next rather than a queue of dialogs
            self._show_status(
                f'Defining imported units for {name}; select the other '
                f'{len(targets) - 1} separately')
        self.render_current()

    def set_active_geometry(self, name: str | None = None) -> None:
        """Choose the geometry every other object is checked against."""
        if name is None:
            item = self.object_item()
            name = item.text(0) if item is not None else None
        if not isinstance(self.objects.get(name), Geometry):
            self._show_status('Select a geometry to make it active')
            return
        self.active_geometry = name
        journal = self.project.journal
        line = f'project.active_geometry = {name!r}'
        if journal and journal[-1].startswith('project.active_geometry = '):
            journal[-1] = line
        else:
            journal.append(line)
        self.refresh_compatibility()
        self._show_status(f'{name} is now the active geometry')

    def refresh_compatibility(self) -> None:
        """Re-check the test and repaint the tree.

        Runs when the project changes — import, delete, rename, or a new
        active geometry — not on selection or render.
        """
        if self.active_geometry not in self.objects:
            self.active_geometry = next(
                (name for name, obj in self.objects.items()
                 if isinstance(obj, Geometry)), None)
        self.report = check_compatibility(self.objects,
                                          self.active_geometry,
                                          links=self.links)
        self._paint_compatibility()

    def _paint_compatibility(self):
        """Red text for what does not fit; nothing at all for what does."""
        error = QBrush(QColor(resolve_theme(self.theme_name)['row_error']))
        for i in range(self.test_item.childCount()):
            item = self.test_item.child(i)
            name = item.text(0)
            issue = self.report.issue_for(name) if self.report else None
            self._set_row_error(item, error if issue else None)
            item.setToolTip(0, issue.message if issue
                            else self._object_tooltip(
                                name, self.objects.get(name, '')))
            self._mark_active_geometry(item, name)
            for j in range(item.childCount()):
                child = item.child(j)
                reference = child.data(0, ROLE_REFERENCE)
                flagged = (issue is not None and reference is not None
                           and reference[0] in ('record', 'mode', 'channel')
                           and reference[2] in issue.sub_items)
                self._set_row_error(child, error if flagged else None)
                if flagged:
                    child.setToolTip(0, issue.message)

    @staticmethod
    def _set_row_error(item, brush):
        """Foreground only, so selection highlighting stays intact."""
        if brush is None:
            item.setData(0, Qt.ItemDataRole.ForegroundRole, None)
        else:
            item.setForeground(0, brush)

    def _mark_active_geometry(self, item, name):
        """Bold, and a bullet beside the icon.

        Bold alone said nothing: the real objects in this tree are
        already darker than the gray slots around them, so a bolder
        weight among them is not a mark anyone can find. The bullet is
        painted by the tree — see `ROLE_ACTIVE` — because the row's text
        is the object's name and anything prepended to it would be
        carried into a rename.
        """
        active = name == self.active_geometry
        font = QFont(item.font(0))
        font.setBold(active)
        item.setFont(0, font)
        item.setData(0, ROLE_ACTIVE, active or None)

    def _hover_item(self, item, column=0):
        """Explain an incompatible row while the pointer is over it."""
        reference = item.data(0, ROLE_REFERENCE) if item else None
        if reference is None or self.report is None:
            return
        owner = self.object_item(item)
        issue = (self.report.issue_for(owner.text(0))
                 if owner is not None else None)
        if issue is not None:
            self.statusBar().showMessage(issue.message)
        else:
            self.statusBar().showMessage(self._status_text)

    # ---- declaring units ----------------------------------------------------

    def _build_units_panel(self):
        """The pane beside the views where a data object's units are named."""
        self.units_panel: QWidget = QWidget()
        layout = QVBoxLayout(self.units_panel)
        layout.setContentsMargins(6, 6, 6, 6)
        self.units_title: QLabel = QLabel()
        font = self.units_title.font()
        font.setBold(True)
        self.units_title.setFont(font)
        layout.addWidget(self.units_title)
        self.units_note: QLabel = QLabel()
        self.units_note.setWordWrap(True)
        layout.addWidget(self.units_note)
        # Two tables, because an FRF has two kinds of channel: responses
        # and references, declared side by side rather than repeated per
        # record. Everything else uses the first table alone and the
        # captions stay hidden.
        self.units_table: CopyPasteTableView = CopyPasteTableView()
        self.units_reference_table: CopyPasteTableView = CopyPasteTableView()
        self.units_response_caption: QLabel = QLabel('Response channels')
        self.units_reference_caption: QLabel = QLabel('Reference channels')
        tables = QHBoxLayout()
        for caption, table in (
                (self.units_response_caption, self.units_table),
                (self.units_reference_caption, self.units_reference_table)):
            table.whole_table_copy = False
            table.edits_applied.connect(self._report_edits)
            side = QVBoxLayout()
            side.addWidget(caption)
            side.addWidget(table)
            tables.addLayout(side)
        tables.setStretch(0, 1)
        layout.addLayout(tables)
        # No OK button: every cell applies as it is set, so there is nothing
        # held back to confirm. Escape puts the pane away, and so does
        # selecting something else — see render_current.
        dismiss = QAction('Close the units pane', self.units_panel)
        dismiss.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        dismiss.setShortcutContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut)
        dismiss.triggered.connect(self.close_units_panel)
        self.units_panel.addAction(dismiss)
        return self.units_panel

    def show_units_panel(self, name: str, obj: Any,
                         records: Sequence[int] | None = None) -> None:
        """Open the units pane on one data object, leaving the plot up.

        An FRF gets two tables — its response channels and its reference
        channels — because a matrix of N×M records has only N+M channel
        units to name. Everything else fills the first table alone.
        """
        two_sided = isinstance(obj, DataArray) and obj.needs_reference
        if two_sided:
            models = frf_units_models(obj, records, self)
        else:
            models = (units_table_model(obj, records, self),)
            self.units_reference_table.setModel(None)
        wanted = 40
        for table, model in zip(
                (self.units_table, self.units_reference_table), models):
            table.setModel(model)
            model.edit_rejected.connect(self._show_status)
            model.dataChanged.connect(self._units_declared)
            wanted += self._size_units_columns(table, model)
        for widget in (self.units_response_caption,
                       self.units_reference_caption,
                       self.units_reference_table):
            widget.setVisible(two_sided)
        total = sum(self.main_split.sizes()) or self.width()
        self.main_split.setSizes([max(total - wanted, 320), wanted])
        # "Units" alone would read as the units things are shown and written
        # in, which is the selector in the status bar. These are the units the
        # values arrived in, and naming them converts once, to SI.
        self.units_title.setText(f'Imported Units — {name}')
        self.units_note.setText(
            'What the values were recorded in — naming a unit converts them '
            'once. Copy, paste, Delete to clear, and right-click to set '
            'several at a time. Drag the corner handle to fill down — or '
            'double-click it to fill to the end.')
        self.units_target = (name, obj)
        self.units_panel.show()

    def _size_units_columns(self, table, model):
        """Open unit columns wide enough for what they offer, and say how
        wide the table wants to be.

        A unit column starts empty, so sizing it to its content leaves the
        only columns worth clicking the narrowest on screen. Size them to
        the units they offer instead.
        """
        table.resizeColumnsToContents()
        metrics = table.fontMetrics()
        for i, column in enumerate(model.columns):
            if not column.editable:
                continue
            widest = max((metrics.horizontalAdvance(str(choice))
                          for choice in column.choices or []), default=0)
            table.setColumnWidth(
                i, max(table.columnWidth(i), widest + COMBO_CELL_PADDING))
        return (sum(table.columnWidth(i)
                    for i in range(model.columnCount()))
                + table.verticalHeader().width())

    def close_units_panel(self) -> None:
        self.units_target = None
        self.units_table.setModel(None)
        self.units_reference_table.setModel(None)
        self.units_panel.hide()

    def _units_declared(self, *_args):
        """A unit landed: restate the plot and the tree, which both show it.

        Deferred, and collapsed to one pass however many cells changed. It
        runs while the cell's editor is still open — committing is what
        emits this — and rebuilding the plot underneath an open drop-down
        closes it, so setting a column of units would fight the user.
        """
        self._units_restate.start(0)

    def _restate_after_units(self):
        if self.units_target is None:
            return
        name, obj = self.units_target
        self._journal_units(name, obj)
        item = self._item_for_object(name)
        if item is not None:
            # icon and tooltip only: rebuilding the children would drop the
            # tree selection, and the selection is what the pane follows
            item.setIcon(0, object_icon(obj, units_defined=_units_defined(obj)))
            item.setToolTip(0, self._object_tooltip(item.text(0), obj))
            self._refresh_child_badges(item, obj)
        self.render_current()

    def _journal_units(self, name, obj):
        """The object's declared units, as the one call that replays
        them.

        The pane writes unit by unit, and each write already converted
        its record — but replayed against the raw import, one
        `define_units` with everything currently declared lands the
        same values, so that is the line recorded, settling in place
        as the declarations grow (Brandon, 2026-08-30: naming units
        said nothing in the console).
        """
        if isinstance(obj, (Geometry, ShapeSet)):
            # object-wide units: one unit, one call — the same line the
            # workflow guides teach
            if not obj.units_defined:
                return
            unit = (obj.length_unit if isinstance(obj, Geometry)
                    else obj.mass_unit)
            line = f'project[{name!r}].define_units({unit!r})'
            prefix = f'project[{name!r}].define_units('
            journal = self.project.journal
            if journal and journal[-1].startswith(prefix):
                journal[-1] = line
            else:
                journal.append(line)
            return
        units = getattr(obj, 'ordinate_unit', None)
        if units is None:
            return
        declared = {i: unit for i, unit in enumerate(units) if unit}
        line = f'project[{name!r}].define_units({declared!r}'
        references = getattr(obj, 'reference_unit', None)
        referenced = {i: unit for i, unit in enumerate(references or [])
                      if unit}
        if referenced:
            line += f', {referenced!r}'
        line += ')'
        journal = self.project.journal
        prefix = f'project[{name!r}].define_units('
        if journal and journal[-1].startswith(prefix):
            journal[-1] = line
        elif declared or referenced:
            journal.append(line)

    def _object_tooltip(self, name, obj):
        """What the object is, and where it came from.

        The name says what it is; the file it was read from used to be in the
        name too and is here instead, so nothing was lost by taking it out.
        """
        lines = [repr(obj)]
        key = self.object_keys.get(name)
        if key and display_name(key) != name:
            # the reader was more specific than the type: 'response cpsd' and
            # 'drive cpsd' are both a PSD, and that distinction lives here now
            lines.append(display_name(key))
        source = self.object_sources.get(name)
        if source:
            lines.append(source)
        return '\n'.join(lines)

    def _refresh_child_badges(self, item, obj):
        """Restate the units badge on already-built children, in place."""
        grid = self.record_grids.get(item.text(0))
        if grid is not None:
            grid.refresh_icons()
            return
        if isinstance(obj, Geometry):
            for i, (_label, attribute, component) in enumerate(GEOMETRY_PARTS):
                if i < item.childCount():
                    item.child(i).setIcon(0, child_icon(
                        component, 'Geometry', obj.units_defined,
                        not len(getattr(obj, attribute))))

    def _report_edits(self, applied, rejected, reason=''):
        """Say how a multi-cell edit landed, including what it could not do."""
        parts = []
        if applied:
            parts.append(f'Set {applied} cell{"s" * (applied != 1)}')
        if rejected:
            refused = f'{rejected} rejected'
            parts.append(f'{refused}: {reason}' if reason else refused)
        self._show_status(' — '.join(parts) or 'Nothing to set')

    def _show_status(self, text):
        """Set the status line, remembering it so hovering can restore it.

        While an import runs the line goes onto the import strip's own
        label instead: a temporary message obscures the status bar's
        left widgets, and the strip — text and progress bar together —
        is exactly what is being painted there.
        """
        self._status_text = text
        # nearly every act says something here, which makes this the
        # cheap place for the console to keep up with the journal
        console = getattr(self, 'console', None)
        if console is not None:
            console.refresh(self.project.journal)
        if self._importing:
            self._import_label.setText(text)
        else:
            self.statusBar().showMessage(text)

    def _item_for_object(self, name):
        """The row holding the named object.

        Matched on being an object *and* on the name, never on the name
        alone: the type's gray slots carry names too, and a project
        whose model geometry is called "FEM Geometry" has a slot of that
        exact name sitting beside it. Answering with the slot moved the
        slot instead of the object when the tree reordered, and drew the
        group's bracket around a row that was not in the group — which
        is what made the brackets look like they wandered.
        """
        for i in range(self.test_item.childCount()):
            child = self.test_item.child(i)
            if child.text(0) == name and child.data(0, ROLE_DRAGGABLE):
                return child
        return None

    def _refresh_item(self, item, obj):
        """Update an item's icon, tooltip and children after a change."""
        item.setIcon(0, object_icon(obj, units_defined=_units_defined(obj)))
        item.setToolTip(0, self._object_tooltip(item.text(0), obj))
        self._mark_computable(item, obj)
        expanded = item.isExpanded()
        self.tree.blockSignals(True)
        self._build_children(item, obj, item.text(0))
        self.tree.blockSignals(False)
        item.setExpanded(expanded)

    def rename_selected(self) -> None:
        """Start inline editing of the selected name.

        The project's own row included. `top_level_item` walks *up* to
        the project and answers None when it is already there — which is
        right for finding an object's row and wrong here, and left
        Rename silently doing nothing on the one row that carries the
        project's name.
        """
        current = self.tree.currentItem()
        item = (self.test_item if current is self.test_item
                else self.top_level_item())
        if item is None:
            return
        # A photo picked in the grid is the thing being renamed, not the
        # object holding it — the grid's cell selection *is* the sub-item
        # selection. Renaming the object instead is not a refusal, it is
        # the wrong rename carried out silently, which is how this read
        # as "renaming a photo does not work".
        name = item.text(0)
        grid = self.record_grids.get(name)
        if isinstance(self.objects.get(name), Photos) and grid is not None:
            picked = grid.selected_records()
            if len(picked) == 1:
                self._rename_photo(name, picked[0])
                return
        self.tree.setCurrentItem(item)
        self.tree.editItem(item, 0)

    def _item_renamed(self, item, column):
        if item is self.test_item:
            name = item.text(0).strip()
            if not name:
                self.tree.blockSignals(True)
                item.setText(0, item.data(0, ROLE_ORIGINAL_NAME))
                self.tree.blockSignals(False)
                return
            item.setData(0, ROLE_ORIGINAL_NAME, name)
            # and the project itself, which is the part that was missing:
            # the row said one thing and `project.name` went on saying
            # another, so Save wrote the old name and so did a drag
            self.project.name = name
            self._show_status(f'Project renamed to {name}')
            return
        if item.parent() is not self.test_item:
            return  # sub-items are labels, not names
        old = item.data(0, ROLE_ORIGINAL_NAME)
        new = item.text(0).strip()
        if old is None or new == old:
            return
        if not new or new in self.objects:
            self.tree.blockSignals(True)
            item.setText(0, old)
            self.tree.blockSignals(False)
            self._show_status(
                f'Name {new!r} is already in use' if new
                else 'Name cannot be empty')
            return
        self._carry_rename(item, old, new)
        self._show_status(f'Renamed {old} to {new}')

    def _carry_rename(self, item, old, new):
        """Rename in the project and every window-side ledger; the row's
        stored references follow. The row's *text* is the caller's — the
        tree edit already changed it, a programmatic rename sets it."""
        self.project.rename(old, new)
        if old in self.record_grids:
            self.record_grids[new] = self.record_grids.pop(old)
            self.record_grids[new].owner = new
        if old in self.object_sources:
            self.object_sources[new] = self.object_sources.pop(old)
        if old in self.object_keys:
            self.object_keys[new] = self.object_keys.pop(old)
        # the active geometry, the links, and every reference inside
        # the objects themselves followed in Project.rename above
        if (self.report_editor is not None
                and self.report_editor.report in self.objects.values()):
            self.report_editor.rebuild()
        self._paint_links()
        item.setData(0, ROLE_ORIGINAL_NAME, new)
        item.setData(0, ROLE_REFERENCE, ('object', new, None))
        for i in range(item.childCount()):
            kind, _, detail = item.child(i).data(0, ROLE_REFERENCE)
            item.child(i).setData(0, ROLE_REFERENCE, (kind, new, detail))
        self.refresh_compatibility()

    def rename_object(self, old: str, new: str) -> str:
        """Rename an object from code, numbering a taken name the way
        `Project.add` does; returns the name used."""
        unique, n = new, 1
        while unique in self.objects:
            n += 1
            unique = f'{new} ({n})'
        item = self._item_for_object(old)
        self.tree.blockSignals(True)
        try:
            item.setText(0, unique)
            self._carry_rename(item, old, unique)
        finally:
            self.tree.blockSignals(False)
        return unique

    def _context_selection(self, item):
        """Set the current item for a context menu without losing a
        multi-selection: right-clicking inside one must keep it, or the menu
        would act on the single row under the cursor."""
        if item.isSelected():
            self.tree.setCurrentItem(
                item, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
        else:
            self.tree.setCurrentItem(item)

    def edit_entities(self) -> None:
        """Edit a geometry's nodes, coordinate systems, tracelines,
        elements or blocks in a table beside the model."""
        kind, obj, detail = self.current_reference()
        component = detail if kind == 'component' else ENTITY_COMPONENT.get(kind)
        if not isinstance(obj, Geometry) or component not in ENTITY_TABLES:
            self._show_status(
                'Select nodes, coordinate systems, tracelines, elements or '
                'blocks to edit them')
            return
        item = self.object_item()
        self.editing = (item.text(0), component)
        self.views.setOrientation(Qt.Orientation.Horizontal)
        model = self._set_table_model(
            ENTITY_TABLES[component](obj, self.unit_system, self))
        model.dataChanged.connect(self._edited_geometry)
        self.table.selectionModel().selectionChanged.connect(
            self._edit_selection_changed)
        self._show_views(three_d=True, table=True)
        self._draw_edit_selection()
        self.add_action.setVisible(True)
        # a block is not placed in space, so its + adds a row outright
        # rather than arming a mode that waits for a click in the view
        self.add_action.setToolTip('Add an empty block' if component == 'blocks'
                                   else 'Add items by clicking in the view')
        self._begin_picking(obj, self._picking_component())
        self._update_toolbar_actions()
        self._show_status(
            f'Editing {component.replace("_", " ")} of {self.editing[0]} — '
            'select rows to highlight them; Escape or reselecting the tree '
            'leaves editing')

    # ---- add mode -----------------------------------------------------------

    def set_add_mode(self, enabled: bool) -> None:
        """Toggle creating things by clicking in the 3D view."""
        enabled = bool(enabled) and self.editing is not None
        if enabled and self.editing[1] == 'blocks':
            # Nothing to click: a block is a name over elements, not a
            # place. The button adds one and comes straight back up —
            # arming a mode that could never be satisfied would be a
            # control that looks live and does nothing. The button comes
            # back up first, because that re-enters here and would
            # otherwise overwrite what was just said in the status bar.
            self.add_action.setChecked(False)
            self._add_block()
            return
        if not enabled and self._picked_nodes:
            self._commit_picked_nodes()
        self.add_mode = enabled
        self._picked_nodes = []
        self._draw_picked()
        if self.add_action.isChecked() != enabled:
            self.add_action.setChecked(enabled)
        self._update_element_type_actions()
        if self.editing is None:
            return
        geometry_name, component = self.editing
        geometry = self.objects.get(geometry_name)
        if geometry is not None:
            # tracelines and elements are built from nodes, so pick nodes
            self._begin_picking(geometry, self._picking_component())
        self._show_status(self._add_mode_hint() if enabled
                          else f'Editing {component.replace("_", " ")}')

    def _picking_component(self):
        """What the cursor selects: nodes while building a line or element,
        and nothing at all while editing blocks, which are not in the view
        to be clicked on."""
        _name, component = self.editing
        if self.add_mode and component in ('tracelines', 'elements'):
            return 'nodes'
        if component == 'blocks':
            return None
        return component

    def _add_block(self):
        """Add an empty block and put the cursor on its row to be named."""
        name, _component = self.editing
        geometry = self.objects.get(name)
        if geometry is None:
            return
        block_id = geometry.add_block()
        self.project.record_call(geometry, 'add_block')
        self._after_geometry_change(name, geometry, f'Added block {block_id}')
        model = self.table.model()
        if model is not None:
            row = model.rowCount() - 1
            self.table.selectRow(row)
            self.table.scrollTo(model.index(row, 1))

    def _edit_toggled(self, checked):
        """The toolbar's edit button, kept in step with the edit state."""
        if checked and self.editing is None:
            self.edit_entities()
        elif not checked and self.editing is not None:
            self.stop_editing()

    def _update_toolbar_actions(self):
        """Show only what the current selection can actually do."""
        if not hasattr(self, 'edit_toggle_action'):
            return                      # the toolbar is still being built
        editing = self.editing is not None
        if self.edit_toggle_action.isChecked() != editing:
            self.edit_toggle_action.setChecked(editing)
        self._update_rotate_actions()

    # ---- turning a coordinate system ---------------------------------------

    def set_rotate_mode(self, enabled: bool) -> None:
        """Show the rings, and take over the mouse while one is dragged."""
        enabled = bool(enabled) and self._turnable_row() is not None
        self._rotating = None
        if enabled:
            self._begin_rotating()
        else:
            self._end_rotating()
        self._update_rotate_actions()

    def _rotating_frame(self):
        """(row, the coordinate system's 4x3 matrix) being turned."""
        row = self._turnable_row()
        if row is None:
            return None, None
        geometry = self.objects.get(self.editing[0])
        return row, geometry.cs_matrix[row]

    def _begin_rotating(self):
        """Draw the rings and watch the mouse ahead of the camera."""
        interactor = getattr(self.scene.plotter, 'iren', None)
        self._draw_rings()
        if interactor is None or self._rotate_observers:
            return
        raw = interactor.interactor
        # ahead of the interactor style, so a drag on a ring turns the
        # frame instead of orbiting the camera. Off a ring the event is
        # left alone and the camera behaves as usual.
        self._rotate_observers = [
            raw.AddObserver('LeftButtonPressEvent', self._on_rotate_press,
                            10.0),
            raw.AddObserver('MouseMoveEvent', self._on_rotate_move, 10.0),
            raw.AddObserver('LeftButtonReleaseEvent', self._on_rotate_release,
                            10.0)]

    def _end_rotating(self):
        interactor = getattr(self.scene.plotter, 'iren', None)
        if interactor is not None:
            for observer in self._rotate_observers:
                with contextlib.suppress(Exception):
                    interactor.interactor.RemoveObserver(observer)
        self._rotate_observers = []
        self._rotating = None
        if self.scene.plotter is not None:
            for axis in range(3):
                self.scene.plotter.remove_actor(f'rotate-ring-{axis}', render=False)
            self.scene.plotter.render()

    def _ring_radius(self):
        _points, _unit = display_points(self.objects[self.editing[0]],
                                        self.unit_system)
        span = np.ptp(_points, axis=0).max() if len(_points) > 1 else 1.0
        return 0.12 * (span or 1.0) * 1.15

    def _draw_rings(self):
        """Three rings, one about each of the frame's own axes."""
        import pyvista as pv

        _row, matrix = self._rotating_frame()
        if matrix is None or self.scene.plotter is None:
            return
        matrix = self._display_frame(matrix)
        radius = self._ring_radius()
        for axis in range(3):
            points = ring_points(matrix, axis, radius)
            loop = np.vstack([points, points[:1]])
            self.scene.plotter.add_mesh(
                pv.lines_from_points(loop).tube(radius=radius * 0.03,
                                                n_sides=12),
                color=AXIS_COLORS[axis], name=f'rotate-ring-{axis}')
        self.scene.plotter.render()

    def _display_frame(self, matrix):
        """The frame in the units the scene is drawn in."""
        shown = np.array(matrix, dtype=np.float64)
        shown[3] = _display_length(matrix[3], self.objects[self.editing[0]],
                                   self.unit_system)
        return shown

    def _ring_screen_points(self):
        """Each ring's axis and its pixels, for hit testing."""
        from ..viz.pick import ScreenProjector

        _row, matrix = self._rotating_frame()
        if matrix is None:
            return []
        matrix = self._display_frame(matrix)
        radius = self._ring_radius()
        rings = []
        for axis in range(3):
            projector = ScreenProjector(self.scene.plotter.renderer,
                                        ring_points(matrix, axis, radius))
            rings.append((axis, projector.screen()[0]))
        return rings

    def _abort(self, caller, observer):
        """Stop VTK passing this event on to the camera."""
        with contextlib.suppress(Exception):
            caller.GetCommand(observer).SetAbortFlag(1)

    def _on_rotate_press(self, caller, _event):
        position = self._cursor_position()
        if position is None:
            return
        axis = ring_under_cursor(self._ring_screen_points(), position)
        if axis is None:
            return                      # not on a ring: let the camera have it
        row, matrix = self._rotating_frame()
        world = self._cursor_on_ring(matrix, axis, position)
        if world is None:
            return
        self._rotating = {
            'row': row, 'axis': axis,
            'start': np.array(matrix, dtype=np.float64),
            'from': angle_in_plane(self._display_frame(matrix), axis, world)}
        self.angle_box.blockSignals(True)
        self.angle_box.setValue(0.0)
        self.angle_box.blockSignals(False)
        self._abort(caller, self._rotate_observers[0])

    def _on_rotate_move(self, caller, _event):
        if self._rotating is None:
            return
        position = self._cursor_position()
        if position is None:
            return
        axis = self._rotating['axis']
        start = self._rotating['start']
        world = self._cursor_on_ring(start, axis, position)
        if world is None:
            return
        swept = wrapped(angle_in_plane(self._display_frame(start), axis, world)
                        - self._rotating['from'])
        self._apply_rotation(swept)
        self.angle_box.blockSignals(True)
        self.angle_box.setValue(np.degrees(swept))
        self.angle_box.blockSignals(False)
        self._abort(caller, self._rotate_observers[1])

    def _on_rotate_release(self, caller, _event):
        if self._rotating is None:
            return
        self._commit_rotation()
        self._abort(caller, self._rotate_observers[2])

    def _cursor_on_ring(self, matrix, axis, position):
        """Where the cursor's ray meets the plane the ring lies in."""
        projector = self._projector
        if projector is None:
            return None
        near = projector.unproject(position[0], position[1], -1.0)
        far = projector.unproject(position[0], position[1], 1.0)
        shown = self._display_frame(matrix)
        return plane_hit(shown[3], shown[axis], near, far - near)

    def _apply_rotation(self, radians):
        """Turn the working copy and redraw only the gizmo."""
        if self._rotating is None:
            return
        turned = rotate_frame(self._rotating['start'],
                              self._rotating['axis'], radians)
        geometry = self.objects[self.editing[0]]
        geometry.cs_matrix[self._rotating['row']] = turned
        self._draw_rings()
        self._draw_live_triad(self._rotating['row'], turned)

    def _draw_live_triad(self, row, matrix):
        """Redraw just this frame's arrows, not the model around it."""
        geometry = self.objects[self.editing[0]]
        shown = self._display_frame(matrix)
        add_coordinate_system(
            self.scene.plotter, shown[3], shown, geometry.cs_type[row],
            self._ring_radius() / 1.15,
            text_color=resolve_theme(self.theme_name)['scene_text'],
            label=int(geometry.cs_id[row]), name='rotate-frame')
        self.scene.plotter.render()

    def _commit_rotation(self):
        """End the gesture: the table and the scene catch up."""
        if self._rotating is None:
            return
        name = self.editing[0]
        row = self._rotating['row']
        self._rotating = None
        # one row, not a whole model reset: a reset would drop the table
        # selection, and the selection is what the rings belong to
        self.table.model().refresh_row(row)
        self._show_status(f'Turned coordinate system in {name}')

    def _angle_typed(self, degrees):
        """An exact angle, applied from where the drag began."""
        if self._rotating is not None:
            self._apply_rotation(np.radians(float(degrees)))
            return
        row, matrix = self._rotating_frame()
        if matrix is None:
            return
        self._rotating = {'row': row, 'axis': self._last_axis,
                          'start': np.array(matrix, dtype=np.float64),
                          'from': 0.0}
        self._apply_rotation(np.radians(float(degrees)))
        self._rotating = None
        self.table.model().refresh_row(row)

    def reset_rotation(self) -> None:
        """Put the frame back to no rotation, keeping where it sits."""
        row, matrix = self._rotating_frame()
        if matrix is None:
            return
        geometry = self.objects[self.editing[0]]
        geometry.cs_matrix[row] = identity_frame(matrix)
        self.angle_box.blockSignals(True)
        self.angle_box.setValue(0.0)
        self.angle_box.blockSignals(False)
        self.table.model().refresh_row(row)
        self._draw_rings()
        self._draw_live_triad(row, geometry.cs_matrix[row])
        self._show_status('Rotation reset')

    def _turnable_row(self):
        """The one coordinate system row selected, or None.

        Turning is a single-object gesture: with two selected there is no
        answer to which one the rings belong to.
        """
        if self.editing is None or self.editing[1] != 'coordinate_systems':
            return None
        model = self.table.selectionModel()
        if model is None:
            return None
        rows = {index.row() for index in model.selectedIndexes()}
        return next(iter(rows)) if len(rows) == 1 else None

    def _update_rotate_actions(self):
        """The rings are offered only when there is one system to turn."""
        turnable = self._turnable_row() is not None
        self.rotate_action.setVisible(turnable)
        if not turnable and self.rotate_action.isChecked():
            self.rotate_action.setChecked(False)
        turning = turnable and self.rotate_action.isChecked()
        self.reset_rotation_action.setVisible(turning)
        for action in self._angle_actions:
            action.setVisible(turning)

    def _update_element_type_actions(self):
        """Show the element-type buttons only while adding elements."""
        adding = (self.add_mode and self.editing is not None
                  and self.editing[1] == 'elements')
        for action in self.element_type_actions.values():
            action.setVisible(adding)

    def _element_type_chosen(self, kind):
        """A different element type: drop a part-built element and say so."""
        self._picked_nodes = []
        self._draw_picked()
        if self.add_mode and self.editing is not None:
            self._show_status(self._add_mode_hint())

    @property
    def element_type(self) -> tuple[int, int]:
        """(type code, node count) for the element type now selected."""
        checked = self.element_type_group.checkedAction()
        for kind, action in self.element_type_actions.items():
            if action is checked:
                return ELEMENT_ADD_TYPES[kind]
        return ELEMENT_ADD_TYPES['tri']

    def _add_mode_hint(self):
        _name, component = self.editing
        if component == 'elements':
            code, count = self.element_type
            return (f'Add mode: {EXTEND_KEYS}-click {count} nodes to make a '
                    f'{ELEMENT_TYPES[code][0]}')
        return {
            'nodes': 'Add mode: click in the view to place a node',
            'coordinate_systems': 'Add mode: click to place a coordinate system',
            'tracelines': f'Add mode: {EXTEND_KEYS}-click nodes in order, then '
                          'press Enter (or switch off +) for the traceline',
        }[component]

    def _add_at(self, x, y, extend=False):
        """Handle a click in the view while adding.

        `extend` is the multi-select modifier: without it a click starts a
        fresh selection, with it the node joins the ones already picked, or
        leaves them if it was picked already.
        """
        name, component = self.editing
        geometry = self.objects.get(name)
        if geometry is None:
            return
        if component in ('nodes', 'coordinate_systems'):
            point = self._projector.place_point(x, y)
            if geometry.units_defined:      # the view is in display units
                point = self.unit_system.to_si(np.asarray(point), 'length')
            if component == 'nodes':
                new_id = geometry.add_node(point)
                self.project.record_call(
                    geometry, 'add_node', [float(v) for v in point])
                message = f'Added node {new_id}'
            else:
                new_id = geometry.add_coordinate_system(origin=point)
                self.project.record_call(
                    geometry, 'add_coordinate_system',
                    origin=[float(v) for v in point])
                message = f'Added coordinate system {new_id}'
            self._after_geometry_change(name, geometry, message)
            return

        node = self._hovered
        if node is None:
            return
        node = int(node)
        if not extend:
            self._picked_nodes = [node]
        elif node in self._picked_nodes:
            self._picked_nodes.remove(node)     # picked twice: unpick it
        else:
            self._picked_nodes.append(node)
        self._draw_picked()
        limit = self.element_type[1] if component == 'elements' else None
        picked = ', '.join(str(n) for n in self._picked_nodes)
        self._show_status(f'{self._add_mode_hint()} — picked {picked}'
                          if picked else self._add_mode_hint())
        if limit and len(self._picked_nodes) == limit:
            self._commit_picked_nodes()

    def _commit_picked_nodes(self):
        """Create the traceline or element from the nodes picked so far."""
        nodes, self._picked_nodes = self._picked_nodes, []
        self._draw_picked()
        if self.editing is None or len(nodes) < 2:
            return
        name, component = self.editing
        geometry = self.objects.get(name)
        if geometry is None:
            return
        try:
            if component == 'tracelines':
                geometry.add_traceline(nodes)
                self.project.record_call(
                    geometry, 'add_traceline', [int(n) for n in nodes])
                message = f'Added traceline through {len(nodes)} nodes'
            else:
                code, count = self.element_type
                if len(nodes) != count:
                    code = None      # Enter with fewer picks: fit the count
                geometry.add_element(nodes, elem_type=code)
                self.project.record_call(
                    geometry, 'add_element', [int(n) for n in nodes],
                    elem_type=code)
                kind = ELEMENT_TYPES[int(geometry.elem_type[-1])][0]
                message = f'Added {kind} element'
        except ValueError as e:
            self._show_status(str(e))
            return
        self._after_geometry_change(name, geometry, message)

    def _after_geometry_change(self, name, geometry, message):
        item = self._item_for_object(name)
        if item is not None:
            self._refresh_item(item, geometry)
        self.refresh_compatibility()
        self._reload_edit_table()
        self._show_status(message)

    def _begin_picking(self, geometry, component):
        """Track what the cursor is over, so clicking selects what is lit.

        `component` is None where there is nothing in the view to pick —
        editing blocks — and then nothing is tracked at all.
        """
        from ..viz.geometry import display_points
        from ..viz.pick import EntityPicker, ScreenProjector

        if component is None:
            self._end_picking()
            return

        points, _ = display_points(geometry, self.unit_system)
        self._projector = ScreenProjector(self.scene.plotter.renderer, points)
        self._picker = EntityPicker(geometry, component, self._projector)
        self._hovered = None
        if self._pick_observers:
            return
        interactor = getattr(self.scene.plotter, 'iren', None)
        if interactor is None:
            return          # offscreen plotter in tests: picking is exercised
        self._pick_observers = [                     # directly, not by mouse
            interactor.add_observer('MouseMoveEvent', self._on_hover),
            interactor.add_observer('LeftButtonPressEvent', self._on_click)]

    def _end_picking(self):
        interactor = getattr(self.scene.plotter, 'iren', None)
        for observer in self._pick_observers:
            with contextlib.suppress(Exception):
                interactor.remove_observer(observer)
        self._pick_observers = []
        self._picker = self._projector = self._hovered = None
        self._hover_mesh = self._picked_mesh = None

    def _cursor_position(self):
        interactor = getattr(self.scene.plotter, 'iren', None)
        if interactor is None:
            return None
        return interactor.interactor.GetEventPosition()

    def _on_hover(self, *_):
        position = self._cursor_position()
        if self.editing is None or self._picker is None or position is None:
            return
        self.hover_at(*position)

    def hover_at(self, x: float, y: float) -> int | None:
        """Light up whatever is under this pixel; returns the entity."""
        entity = self._picker.pick(x, y)
        if entity == self._hovered:
            return entity        # nothing changed, so nothing to redraw
        self._hovered = entity
        self._draw_hover()
        return entity

    def _draw_picked(self, render=True):
        """Keep every node picked so far lit, and the line they trace.

        Picking the second node of a four-node element should not look the
        same as picking the first, so the ones already taken stay visible
        instead of going dark as the cursor moves on.
        """
        if self._picked_mesh is None:
            return
        name, _component = self.editing or (None, None)
        geometry = self.objects.get(name)
        colors = resolve_theme(self.theme_name)
        nodes = self._picked_nodes
        rows = []
        if geometry is None or not nodes:
            self._picked_mesh.verts = np.empty(0, dtype=np.int64)
            self._picked_mesh.lines = np.empty(0, dtype=np.int64)
        else:
            rows = geometry.node_index(nodes)
            self._picked_mesh.verts = np.column_stack(
                [np.ones(len(rows), dtype=np.int64), rows]).ravel()
            self._picked_mesh.lines = (
                np.array([len(rows), *rows], dtype=np.int64) if len(rows) > 1
                else np.empty(0, dtype=np.int64))
        self._label_nodes('picked-labels', rows if nodes else [], nodes,
                          colors['scene_picked'])
        if render:
            self.scene.plotter.render()

    def _label_nodes(self, name, rows, labels, color):
        """Caption these node rows in the scene, or clear the captions.

        Coordinates come from the scene's own point array, so labeling a
        handful of nodes costs a handful of points, not a copy of the model.
        """
        self.scene.plotter.remove_actor(name, render=False)
        if not len(rows):
            return
        self.scene.plotter.add_point_labels(
            self._picked_mesh.points[list(rows)],
            [str(label) for label in labels], name=name, font_size=14,
            always_visible=True, text_color=color, shape=None,
            fill_shape=False, show_points=False, render=False)

    def _draw_hover(self, render=True):
        """Update the hover mesh in place — no scene rebuild per mouse move."""
        if self._hover_mesh is None:
            return
        name, _component = self.editing
        geometry = self.objects.get(name)
        # what the cursor is over, not what is being edited: building a
        # traceline or element lights up the nodes the picker is returning
        component = self._picking_component()
        if component is None:
            return                       # blocks: nothing in the view to hover
        cells = _hover_cells(geometry, component, self._hovered)
        self._hover_mesh.verts = cells['verts']
        self._hover_mesh.lines = cells['lines']
        rows = ([] if component != 'nodes' or self._hovered is None
                else geometry.node_index([self._hovered]))
        self._label_nodes('hover-label', rows, [self._hovered],
                          resolve_theme(self.theme_name)['scene_highlight'])
        if render:
            self.scene.plotter.render()

    def _extend_pressed(self):
        """Is the multi-select modifier down: Shift, or Cmd on a Mac and
        Ctrl elsewhere?

        Both layers get a say. The click arrives through VTK, which carries
        Shift and Ctrl but knows nothing of the Command key; Qt knows
        Command but only as of the last event it handled itself.
        """
        modifiers = QApplication.keyboardModifiers()
        if modifiers & (Qt.KeyboardModifier.ShiftModifier
                        | Qt.KeyboardModifier.ControlModifier
                        | Qt.KeyboardModifier.MetaModifier):
            return True
        interactor = getattr(self.scene.plotter, 'iren', None)
        if interactor is None:
            return False
        vtk_interactor = interactor.interactor
        return bool(vtk_interactor.GetShiftKey()
                    or vtk_interactor.GetControlKey())

    def _on_click(self, *_):
        if self.editing is None:
            return
        if self.add_mode:
            position = self._cursor_position()
            if position is not None:
                self._add_at(*position, extend=self._extend_pressed())
            return
        if self._hovered is None:
            return
        self.select_entity(self._hovered, extend=self._extend_pressed())

    def select_entity(self, entity: int | None,
                      extend: bool = False) -> int | None:
        """Select the table row for an entity picked in the 3D view.

        The same rule for every kind of entity: a plain click selects only
        what was clicked, the modifier adds to the selection, and clicking
        something already selected takes it back out.
        """
        if self.editing is None or entity is None:
            return None
        name, component = self.editing
        geometry = self.objects.get(name)
        row = _row_for_entity(geometry, component, entity)
        if row is None:
            return None
        if not extend:
            mode = QItemSelectionModel.SelectionFlag.ClearAndSelect
        elif row in {index.row() for index
                     in self.table.selectionModel().selectedIndexes()}:
            mode = QItemSelectionModel.SelectionFlag.Deselect
        else:
            mode = QItemSelectionModel.SelectionFlag.Select
        model = self.table.model()
        self.table.selectionModel().select(
            model.index(row, 0),
            mode | QItemSelectionModel.SelectionFlag.Rows)
        self.table.scrollTo(model.index(row, 0))
        return row

    def stop_editing(self) -> None:
        if self.editing is None:
            return
        self.set_add_mode(False)
        self.add_action.setVisible(False)
        self._update_element_type_actions()
        self._end_picking()
        self.editing = None
        self.views.setOrientation(Qt.Orientation.Vertical)
        self.render_current()
        self._update_toolbar_actions()

    def _edited_geometry(self, *_):
        """An edit changed the model: redraw and re-check compatibility."""
        if self.editing is None:
            return
        name, _component = self.editing
        geometry = self.objects.get(name)
        if geometry is None:
            return
        try:
            geometry.validate()
        except ValueError as e:
            self._show_status(f'{e}')
        self.refresh_compatibility()
        self._draw_edit_selection()

    def _edit_selection_changed(self, *_):
        self._update_toolbar_actions()
        self._draw_edit_selection()

    def _draw_edit_selection(self):
        """Highlight the selected rows' entities in the 3D view."""
        if self.editing is None or self.scene.plotter is None:
            return
        name, component = self.editing
        geometry = self.objects.get(name)
        if geometry is None:
            return
        rows = sorted({index.row()
                       for index in self.table.selectionModel().selectedIndexes()})
        entities = {}
        if rows:
            entities[component] = self._entity_keys(geometry, component, rows)
        colors = resolve_theme(self.theme_name)
        camera = self.scene.plotter.camera_position
        self.scene.plotter.clear()
        self.scene.plotter.set_background(colors['scene_background'],
                                    top=colors['scene_background_top'])
        if entities:
            axis_unit = self._add_with_context(geometry, None, entities, colors)
        else:
            axis_unit = add_geometry(self.scene.plotter, geometry,
                                     unit_system=self.unit_system,
                                     text_color=colors['scene_text'])
        self.scene.axis_unit = axis_unit
        self._add_hover_mesh(geometry, colors)
        annotate_scene(self.scene.plotter, axis_unit, colors, self.scene.bounds_visible,
                       self.scene.orientation_visible)
        # adding an element must not move the camera: an edit redraws the
        # scene, and a rebuilt scene would otherwise re-frame itself
        self.scene.plotter.camera_position = camera
        self.scene.plotter.render()

    def _add_hover_mesh(self, geometry, colors):
        """Two meshes over the scene's points: one follows the cursor, one
        holds the nodes picked so far.

        Both share the scene's own point array, so lighting something up
        moves no coordinates — only which cells are drawn.
        """
        import pyvista as pv

        from ..viz.geometry import display_points, shared_points

        points, _ = display_points(geometry, self.unit_system)
        source, _view = shared_points(points)
        mesh = pv.PolyData()
        mesh.SetPoints(source)
        self._hover_mesh = mesh
        self.scene.plotter.add_mesh(mesh, color=colors['scene_highlight'],
                              point_size=18.0, line_width=6.0,
                              render_points_as_spheres=True)

        picked = pv.PolyData()
        picked.SetPoints(source)
        self._picked_mesh = picked
        self.scene.plotter.add_mesh(picked, color=colors['scene_picked'],
                              point_size=20.0, line_width=6.0,
                              render_points_as_spheres=True)
        self._draw_picked(render=False)
        self._draw_hover(render=False)

    @staticmethod
    def _entity_keys(geometry, component, rows):
        """What add_geometry wants: node, cs or block ids, or row indices."""
        if component == 'nodes':
            return [int(geometry.node_id[row]) for row in rows]
        if component == 'coordinate_systems':
            return [int(geometry.cs_id[row]) for row in rows]
        if component == 'blocks':
            return [int(geometry.block_id[row]) for row in rows]
        return list(rows)

    @staticmethod
    def _units_declarable(obj) -> bool:
        """Whether *Define Imported Units* is a legitimate option here.

        Asked of the object rather than listed here, because the list
        was wrong twice over — it named a channel table and a photo set
        and missed a report and a matched-modes set, both of which were
        being offered a verb with nothing to apply it to.

        `define_units` is the verb itself, so an object that has it can
        be asked and an object that has not, cannot. What that comes to:
        a geometry, a data array and a shape set have units. A channel
        table declares its own in its `unit` column, edited in the table
        like every other cell — a second place to say the same thing
        could disagree with the first. A photograph has no units,
        nothing about a JPEG being a quantity, and a report is a
        document.
        """
        return hasattr(obj, 'define_units')

    def _grid_menu(self, grid, position):
        """The record half of the tree menu, built for a grid cell.

        `position` is in viewport coordinates. Returns None off the cells.
        Mirrors `_context_selection`: right-clicking inside a selection
        keeps it, right-clicking outside moves it.
        """
        cell = grid.itemAt(position)
        if cell is None or not (cell.flags() & Qt.ItemFlag.ItemIsSelectable):
            return None
        if not cell.isSelected():
            grid.setCurrentItem(cell)
        menu = QMenu(grid)
        if self._units_declarable(self.objects.get(grid.owner)):
            records = sum(1 for kind, *_ in self.selected_references()
                          if kind == 'record')
            units_action = menu.addAction(
                f'Define &Imported Units for {records} channels...'
                if records > 1 else 'Define &Imported Units...')
            units_action.triggered.connect(self.define_units)
        elif isinstance(self.objects.get(grid.owner), ChannelTable):
            # a channel table has no imported units — its Unit column
            # is the data — but a dead right-click read as a missing
            # feature (Brandon, 2026-08-30): the row's answer is the
            # edit table, so the menu takes you there
            edit_action = menu.addAction('&Edit Channel Table...')
            edit_action.triggered.connect(self.edit_entities)
        # Delete is not here. It is the Delete and Backspace keys, on
        # the tree, the grids and the 3-D view alike, and a menu entry
        # for it only crowds the verbs that have nowhere else to live.
        return menu if not menu.isEmpty() else None

    def _show_grid_menu(self, grid, position):
        position = grid.viewport().mapFrom(grid, position)
        menu = self._grid_menu(grid, position)
        if menu is not None:
            menu.exec(grid.viewport().mapToGlobal(position))

    def _show_tree_menu(self, position):
        # a right-click on a link bracket is about the group, not the
        # object beside it
        span = self.tree.span_at(position)
        if span is not None:
            self._show_bracket_menu(span, position)
            return
        item = self.tree.itemAt(position)
        if item is None:
            return
        self._context_selection(item)
        menu = QMenu(self.tree)
        kind = (item.data(0, ROLE_REFERENCE) or (None,))[0]
        if kind == 'test':
            # generating the report lives on the bar, with the project
            # row selected — not in this menu
            types = menu.addMenu('Set Project &Type')
            for option in (*PROJECT_TYPES, None):
                action = types.addAction(option or 'None')
                action.setCheckable(True)
                action.setChecked(self.project_type == option)
                action.triggered.connect(
                    lambda _checked=False, option=option:
                    self.set_project_type(option))
            menu.addAction(self.save_action)
            menu.exec(self.tree.viewport().mapToGlobal(position))
            return
        if kind == 'placeholder':
            self._show_placeholder_menu(item, position)
            return
        sub_item = kind == 'component' or kind in ENTITY_COMPONENT
        # a geometry is in one length unit throughout — units are declared
        # for the whole thing, never part of it. Data records are different:
        # channels genuinely differ, and keep their own entry.
        if not sub_item and self._units_declarable(self.current_object()):
            records = sum(1 for kind, *_ in self.selected_references()
                          if kind == 'record')
            units_action = menu.addAction(
                f'Define &Imported Units for {records} channels...'
                if records > 1 else 'Define &Imported Units...')
            units_action.triggered.connect(self.define_units)
        if sub_item:
            menu.addAction(self.edit_action)
        # the rest act on the whole object, which a sub-item is not: its
        # name is fixed, and there is nothing there to save or make active
        if not sub_item:
            if isinstance(self.current_object(), Geometry):
                menu.addAction(self.active_geometry_action)
            if isinstance(self.current_object(), Report):
                menu.addAction(self.export_report_action)
            # no acts here: merging, projecting and transforming live
            # on the bar with every other act (Brandon, 2026-09-04),
            # and linking on the tree's own toolbar — which group an
            # object is in is said by dragging it there
            menu.addAction(self.save_action)
        if not menu.isEmpty():
            menu.exec(self.tree.viewport().mapToGlobal(position))

    # ---- import / save ------------------------------------------------------

    #: the update check's answer arrives from its own thread
    _update_answer = Signal(object)

    def about(self) -> None:
        """The version and what it is: an alpha, to be checked, with
        the contact address. The one place a user can read the version
        without the network or a relaunch."""
        from .. import __version__
        from .disclaimer import TEXT

        box = QMessageBox(self)
        box.setWindowTitle('About Visual Dynamics')
        box.setIconPixmap(self.windowIcon().pixmap(64, 64))
        box.setText(f'<b>Visual Dynamics {__version__}</b><br>'
                    'A units-aware toolset for structural dynamics test '
                    'work.<br>© 2026 Brandon Zwink — see LICENSE for the '
                    'terms.')
        box.setInformativeText(TEXT)
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.setModal(False)
        box.show()
        self.about_box: QMessageBox | None = box

    def check_for_updates(self) -> None:
        """File → Check for Updates: ask visualdynamics.org for the
        latest version and say what it found.

        A check, never an update — `update.py` explains the line: an
        updater that runs what it fetched without verifying a
        signature is remote code execution with a friendly name, and
        there is no signing identity yet. So a newer version is
        offered as a page to open, in the browser, where the release
        is (and where a private release wants the user's own login).
        The fetch runs off the main thread: the check is the least
        important thing the application does and must never be why it
        is slow.
        """
        import threading

        from .. import update

        self._show_status('Checking for updates…')
        if not getattr(self, '_update_wired', False):
            self._update_answer.connect(self._report_update)
            self._update_wired = True
        threading.Thread(target=lambda: self._update_answer.emit(
            update.fetch()), daemon=True).start()

    def _report_update(self, manifest) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QMessageBox

        from .. import __version__, update

        if manifest is None:
            self._show_status('Could not reach visualdynamics.org to '
                              'check for updates')
            return
        version = str(manifest.get('version', ''))
        if not version or not update.newer(version):
            self._show_status(f'Visual Dynamics {__version__} is up to date')
            return
        self._show_status(f'Visual Dynamics {version} is available')
        box = QMessageBox(self)
        box.setWindowTitle('Update available')
        box.setText(f'Visual Dynamics {version} is available '
                    f'(this is {__version__}).')
        notes = str(manifest.get('notes', '')).strip()
        if notes:
            box.setInformativeText(notes)
        url = str(manifest.get('url', '')).strip()
        opener = box.addButton('Open download page',
                               QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.setDefaultButton(opener)
        if url:
            opener.clicked.connect(
                lambda: QDesktopServices.openUrl(QUrl(url)))
        else:
            opener.setEnabled(False)
        self._update_box = box               # kept alive while open
        box.open()

    def import_files(self) -> None:
        filters = ('Importable files (*.vdyn *.vdreport *.npz *.exo *.e *.exo2 *.g *.gen '
                   '*.unv *.uff *.uf *.nc4 *.nc *.afu *.ati *.ash '
                   '*.bdf *.dat *.nas *.pch *.neu *.3mf *.stl '
                   '*.step *.stp *.iges *.igs'
                   ');;All files (*)')
        paths, _ = QFileDialog.getOpenFileNames(self, 'Import', '', filters)
        self.import_paths(paths)

    @staticmethod
    def _dropped_files(mime):
        from .project_tree import dropped_files

        return dropped_files(mime)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Anywhere on the window will do.

        Dropping on the tree alone made a target of whatever the tree
        happened to be that moment — and on a fresh window it is sized to
        no content at all, a strip about 90 px wide. Missing it does
        nothing at all, which reads as the drop being ignored, so the
        file gets dragged over again. The window is the thing being
        aimed at; let it take the file.
        """
        from .project_tree import trace_drag

        trace_drag('window', 'enter', event.mimeData())
        if event.mimeData().hasUrls() and self._dropped_files(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def event(self, found: QEvent) -> bool:
        """Keyboard focus is parked while the window is inactive.

        macOS 26 with Qt 6.11 discards a Finder drop released over the
        rect of the widget that held focus when the app deactivated:
        draggingEntered and draggingUpdated are answered Copy, and the
        release still arrives as draggingExited — no drop, no error,
        nothing. The tree is the first focusable widget, so on a fresh
        window it held focus and the one dead spot in the window was
        the very widget whose status line says "drag files onto the
        tree". Proven by swizzling the NSView's dragging methods and
        toggling nothing but focus; the full hunt is in the log for
        commit that introduced this.

        Three shapes of this fix failed first, each for a reason worth
        keeping:

        - Parking when the drag *enters* is too late. The drag's source
          deactivated the app before the first enter, and the platform
          state that kills the drop is set by then.
        - `clearFocus()` at deactivation is in time but does nothing:
          focus-to-nobody leaves the platform's input state armed. Only
          a genuine handoff to another widget runs the full teardown.
        - Disabling input methods on the tree changes nothing — the
          poison is the focus itself, not the input context.

        So focus is *handed* to a zero-size widget whose rect nothing
        can hit, and handed back on activation. An inactive window has
        no keyboard, so nothing is lost in between.
        """
        if found.type() == QEvent.Type.WindowDeactivate:
            if QApplication.activePopupWidget() is not None:
                # deactivated by one of our own popups — a combo's
                # drop-down list is its own window on macOS 26 — and
                # parking here hands focus away from the editor, which
                # closes the popup the instant it opens (Brandon,
                # 2026-08-30, the units drop-down). The park exists
                # for drags arriving from *another app*; while our
                # popup is up, the app never lost the stage.
                return super().event(found)
            focused = self.focusWidget()
            if focused is not None and focused is not self._focus_park:
                self._parked_focus = focused
                self._focus_park.setFocus(
                    Qt.FocusReason.OtherFocusReason)
        elif found.type() == QEvent.Type.WindowActivate:
            parked, self._parked_focus = self._parked_focus, None
            if parked is not None and self.focusWidget() is self._focus_park:
                try:
                    if parked.isVisible():
                        parked.setFocus(Qt.FocusReason.OtherFocusReason)
                except RuntimeError:
                    pass            # it died while the window was away
        return super().event(found)

    def dropEvent(self, event: QDropEvent) -> None:
        from .project_tree import trace_drag

        paths = self._dropped_files(event.mimeData())
        trace_drag('window', f'drop paths={len(paths)}', event.mimeData())
        if paths:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.acceptProposedAction()
            self._import_after_drop(paths)
        else:
            super().dropEvent(event)

    def _import_after_drop(self, paths):
        """Queue a dropped import for after the drag has unwound.

        Never straight to import_paths. A drop is delivered from inside
        macOS's own drag run loop — `NSCoreDragReceiveMessageProc` spins
        one and pumps it — so anything done here runs with a nested loop
        live and free to deliver paint events. Importing and rebuilding
        the plot in that window means Qt can paint a scene we are halfway
        through replacing; a crash report showed exactly that, a
        pyqtgraph curve item painting from the drag's run loop.

        The answer is immediate even though the work is queued. A
        project file runs to hundreds of megabytes and half a minute,
        and for all of it the window used to say nothing — a drop that
        answers with thirty silent seconds is indistinguishable from a
        drop that was ignored, so the file got dragged again, somewhere
        else, and the second spot got credit for the first one's work.
        """
        from .project_tree import trace_drag

        trace_drag('import', f'queued {len(paths)} file(s)')
        told = ', '.join(os.path.basename(path) for path in paths[:3])
        if len(paths) > 3:
            told += f' and {len(paths) - 3} more'
        self._show_status(f'Importing {told}…')
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        # painted *now*, synchronously: the import runs on the very next
        # event-loop turn and blocks it for as long as the file is big,
        # and a status message that never reached the screen said
        # nothing — the drop looked ignored for thirty seconds, which is
        # the exact confusion the message exists to prevent
        self.statusBar().repaint()

        def run():
            try:
                self.import_paths(paths)
            finally:
                QApplication.restoreOverrideCursor()

        QTimer.singleShot(0, run)

    def _exodus_import_choices(self, path):
        """How an exodus file's results should be read, asked as needed.

        {} for a file with no results (or any non-exodus file) — no
        dialog, nothing to ask. Otherwise the step axis is declared by
        the user, because the file cannot say what it means: a modal
        run, a transient and a spectral convention all ride time_whole.
        None means cancel — skip this file. Choosing a time or
        frequency reading on a file with node sets asks which nodes,
        since a big mesh's every-node reading is a channel per node per
        variable and the sets are the format's own way of naming the
        instrumented few.
        """
        from ..io import exodus

        if not exodus.sniff(path):
            return {}
        summary = exodus.result_summary(path)
        if not summary['nodal'] and not summary['global']:
            return {}
        base = os.path.basename(path)
        found = len(summary['nodal']) + len(summary['global'])
        readings = ['Mode shapes — one mode per step',
                    'Time data — the steps are time',
                    'Spectra — the steps are frequency lines']
        chosen, ok = QInputDialog.getItem(
            self, 'Reading the results',
            f'{base} carries {found} result variable'
            f'{"s" * (found != 1)} over {summary["steps"]} steps.\n'
            'What does its step axis mean?', readings, 0, False)
        if not ok:
            return None
        steps = ('modes', 'time', 'frequency')[readings.index(chosen)]
        if steps == 'modes':
            return {}
        options = {'steps': steps}
        if summary['node_sets'] and summary['nodal']:
            entries = [f'Every node ({summary["nodes"]})']
            for set_id, name, count in summary['node_sets']:
                label = name or f'set {set_id}'
                entries.append(f'{label} ({count} nodes)')
            chosen, ok = QInputDialog.getItem(
                self, 'Which nodes',
                'Each node is one channel per variable:', entries, 0,
                False)
            if not ok:
                return None
            at = entries.index(chosen)
            if at:
                set_id = summary['node_sets'][at - 1][0]
                options['nodes'] = exodus.node_set_nodes(path, set_id)
        return options

    def import_paths(self, paths: Sequence[str]) -> list[str]:
        """Import several files, reporting any failures once at the end.

        Dropped images are not their own objects: they land together in
        the project's Photos object, created on first use.
        """
        if self._importing:
            # the progress ticks flush paints by pumping the queue, and
            # a drop landing mid-import queues a second import that the
            # pump would start *inside* the first — the very
            # re-entrancy all the deferral exists to prevent. It waits
            # its turn instead.
            QTimer.singleShot(100, lambda: self.import_paths(list(paths)))
            return []
        self._importing = True
        # the strip takes over from the temporary message: same words,
        # left side, immune to nothing painting while the loop blocks
        self.statusBar().clearMessage()
        self._import_label.setText(self._status_text)
        self._import_strip.show()
        try:
            return self._import_paths(paths)
        finally:
            self._importing = False
            self._import_strip.hide()
            # the one render the suppressed selection changes add up to.
            # It narrates what it drew, which must not shout down what
            # the import said — a type announcement outranks "8
            # channels on the stage" — so the import's last word is
            # restated after it.
            told = self._status_text
            self.render_current()
            # whatever was said last mid-import — a type announcement,
            # a refusal — went to the strip's label; restate it the
            # normal way so hiding the strip does not eat it
            self._show_status(told)

    def _import_paths(self, paths: Sequence[str]) -> list[str]:
        pictures = [path for path in paths
                    if os.path.splitext(path)[1].lower() in PHOTO_FORMATS]
        imported, failures, announced = [], [], False
        if pictures:
            imported.append(self._add_photos(pictures, failures))
        remaining = [path for path in paths if path not in pictures]
        bar = self._import_progress

        def tick(done: int, total: int) -> None:
            bar.setRange(0, total)
            bar.setValue(done)
            if bar.isHidden():
                bar.show()
                # show() only *posts* the layout request, and the loop
                # that would deliver it is blocked — the bar stayed
                # zero-sized, and every synchronous repaint painted
                # nothing. Activating the status bar's layout here gives
                # it geometry to paint, which is the difference between
                # a progress bar and a progress variable.
                layout = self.statusBar().layout()
                if layout is not None:
                    layout.activate()
            bar.repaint()
            # repaint() alone was the first version, on the rule that
            # the import loop stays blocked. On macOS it painted into
            # the backing store and no further: the views are
            # layer-backed and only the event loop flushes them to the
            # screen, so the bar moved and nobody could ever have seen
            # it — reported twice as "no loading bar". Excluding user
            # input keeps out the second drop and the mid-import click,
            # which are the re-entrancies the blocking exists for.
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

        many = len(remaining) > 1
        if many:
            tick(0, len(remaining))
        try:
            for n, path in enumerate(remaining):
                try:
                    # an exodus file with results cannot say what its
                    # own step axis means; the reading is asked, not
                    # guessed — and canceling skips the file
                    options = self._exodus_import_choices(path)
                    if options is None:
                        continue
                    # a lone project file reports per object; in a
                    # multi-file drop the files themselves are the steps
                    result = io.import_file(
                        path, progress=None if many else tick, **options)
                except Exception as e:  # noqa: BLE001 — see below
                    # Any exception, not only ValueError and OSError. A
                    # file refused *by the objects* raises ValueError,
                    # but a file that is simply malformed raises
                    # whatever its reader trips over first — an
                    # IndexError off the end of a short UNV record, a
                    # KeyError for a missing netCDF variable — and those
                    # used to escape this loop into the Qt event loop,
                    # where a traceback goes nowhere and the user sees
                    # the import silently do nothing. It is not
                    # swallowed: the file is named, with what went
                    # wrong, in the dialog below.
                    failures.append(f'{os.path.basename(path)}: '
                                    f'{type(e).__name__}: {e}')
                    continue
                finally:
                    if many:
                        tick(n + 1, len(remaining))
                # a lone project spends most of a big import *after* the
                # read, building each object's rows — measured at 4 s of
                # tree against 0.5 s of file — so the bar walks that too
                added = self._add_result(
                    path, result, tick=None if many else tick)
                imported.append(added)
                announced = self._adopt_project_type(
                    path, added if isinstance(added, list) else [added]) \
                    or announced
                if (isinstance(result, dict)
                        and not isinstance(result, io.TestContents)):
                    # a controller run, whose objects are one
                    # measurement — a .vdyn is a dict too, but it
                    # brings its author's own arrangement
                    self._absorb_run_kin(
                        [name for name in
                         (added if isinstance(added, list) else [added])
                         if name])
        finally:
            bar.hide()
        if failures:
            QMessageBox.warning(
                self, 'Some files could not be imported', '\n\n'.join(failures))
        # a .vdyn can arrive already stale — saved settings moved after
        # its derived objects were computed
        self._refresh_stale_badges()
        flat = [name for part in imported
                for name in (part if isinstance(part, list) else [part])
                if name]
        # the roll call yields to a project-type announcement: 'this is
        # a Random Vibration run' matters more than the list of names,
        # and both compete for the same one line
        if flat and not announced:
            told = ', '.join(flat[:3]) + (f' and {len(flat) - 3} more'
                                          if len(flat) > 3 else '')
            self._show_status(f'Imported {told}')
        return imported

    def _adopt_project_type(self, path, added=()) -> bool:
        """Take the project's type from a file that knows what it is.
        Returns whether it put an announcement on the status bar.

        A controller's own save says which environment drove the run, so
        importing one settles a question the user would otherwise answer
        from the Project menu — and settles it right, since the file was
        written by the thing that ran the test.

        A switch is announced rather than silent: the type decides which
        report is generated and which slots the tree expects, so quietly
        turning a modal project into a random one would be a surprise
        the next Generate Report delivered.
        """
        try:
            wanted = io.project_type_of(path)
        except Exception as refusal:  # noqa: BLE001 - never fail an import
            # the objects are already in the project; a file that cannot
            # say what kind of test it was does not undo that
            self._show_status(f'Could not read the test type: {refusal}')
            return True
        # a streamed system-ID save is indistinguishable from a run of
        # its environment, so a file shaped like one — two streams,
        # quiet then loud — is a question for the person, not a guess
        # (Brandon, 2026-08-23)
        from ..io.rattlesnake import streamed_sysid_candidate
        candidate = streamed_sysid_candidate(path)
        if candidate and self.project_type == 'System ID':
            # nothing left to ask — the project already says what these
            # two streams are, so they take their phase names directly
            self._name_sysid_streams(added)
        elif candidate:
            answer = QMessageBox.question(
                self, 'System identification?',
                f'{os.path.basename(path)} holds two streams, a quiet '
                'one then a loud one — the shape a system ID save '
                'leaves, which the file cannot tell apart from a run '
                'of its environment.\n\n'
                'Treat this project as a System ID test?')
            if answer == QMessageBox.StandardButton.Yes:
                previous = self.project_type
                self.set_project_type('System ID')
                self._name_sysid_streams(added)
                self._show_status(
                    f'{os.path.basename(path)} taken as a System ID run'
                    + (f' — project type changed from {previous}'
                       if previous else ''))
                return True
        if wanted is None or wanted == self.project_type:
            return False
        previous = self.project_type
        self.set_project_type(wanted)
        if previous:
            self._show_status(f'{os.path.basename(path)} is a {wanted} run — '
                              f'project type changed from {previous}')
        return bool(previous)

    def _name_sysid_streams(self, added):
        """Call a system ID's two recordings what they are.

        'Time History' and 'Time History (2)' ask the person to
        remember which phase streamed first; once they have said the
        file is a system ID, the quiet one *is* the noise measurement
        and the loud one the excitation, so the names can say so
        (Brandon, 2026-08-23). Told apart by level, the same reading
        the import question and the signal-to-noise default use — not
        by stream order, which would trust a convention the check can
        measure instead.
        """
        histories = [name for name in added
                     if isinstance(self.objects.get(name), TimeHistory)]
        if len(histories) != 2:
            return
        quiet, loud = sorted(
            histories,
            key=lambda name: float(self.objects[name].ordinate.std()))
        for old, new in ((quiet, 'Noise Time History'),
                         (loud, 'Excitation Time History')):
            used = self.rename_object(old, new)
            # the import loop goes on using its list of added names —
            # the run's own linking walks it — so the renames have to
            # land there too, or the link asks for a name that no
            # longer exists and the whole group quietly refuses
            if isinstance(added, list) and old in added:
                added[added.index(old)] = used

    def _add_photos(self, paths, failures):
        name = next((n for n, obj in self.objects.items()
                     if isinstance(obj, Photos)), None)
        if name is None:
            # quiet through the raw add; the creation line below is
            # the replayable record (Photos() has an eval-able source,
            # unlike a data object built from arrays)
            with self.project.journal_as(None):
                name = self.add_object('Photos', Photos())
            self.project.journal.append(
                f'project.add({name!r}, Photos())')
        target = self.objects[name]
        added = []
        for path in paths:
            try:
                added.append(target.add_file(path))
            except (ValueError, OSError) as e:
                failures.append(f'{os.path.basename(path)}: {e}')
            else:
                self.project.journal.append(
                    f'project[{name!r}].add_file({str(path)!r})')
        item = self._item_for_object(name)
        item.setToolTip(0, self._object_tooltip(name, target))
        if added:
            self._refresh_item(item, target)    # the grid gains its rows
            self.tree.setCurrentItem(item)
            self._show_status(
                f'{len(added)} photo{"s" * (len(added) != 1)} added to '
                f'{name} ({target.num_photos} total)')
        return name

    def import_path(self, path: str, **units: str) -> str | None:
        try:
            result = io.import_file(path, **units)
        except (ValueError, OSError) as e:
            QMessageBox.critical(self, 'Import failed', str(e))
            return None
        return self._add_result(path, result)

    def _absorb_run_kin(self, names):
        """One controller file is one measurement: whatever a run
        brought that the type's structure has no slot for still belongs
        beside the rest of it, so after the expectations have placed
        what they place, the file's leftovers join the group its
        members landed in — a random run's system-ID FRF sits inside
        the basis bracket rather than outside it.

        Runs only (dict results): a .vdyn brings its *own* links, and
        an object its author left unlinked stays that way. Runs at
        import only, so an explicit Unlink later is not fought.
        """
        grouped = {member for group in self.links
                   for member in group['members']}
        placed = next((name for name in names if name in grouped), None)
        loose = [name for name in names if name not in grouped]
        if placed is None or not loose:
            return
        try:
            self.project.link(placed, *loose)
        except (ValueError, KeyError) as refusal:
            # incompatibility warns, it does not block the import
            self._show_status(str(refusal))
            return
        self._links_changed()
        told = ', '.join(loose)
        self._show_status(f'{told} joined the run’s group')

    def _add_result(self, path, result, tick=None):
        """Name each imported object for its type. Nothing else.

        Every importer but one returned a single object and got its class
        name; the Rattlesnake reader returned a dict and got hand-written
        keys, so the tree spoke two vocabularies — 'Modal FRF' beside
        'Geometry', qualified by an environment name that came from the file
        and means nothing to the rest of the model. One rule instead: an
        object is named for what it is, and a second of the same kind is
        '(2)'.

        The reader's own key is more specific than the type — 'response cpsd'
        and 'drive cpsd' are both a PSD — so it goes on the tooltip beside
        the file, rather than into the name.
        """
        # journaled as the one line a script would say — the adds
        # inside are this front end doing import_file's job by hand,
        # and a pile of not-replayable comments was the journal being
        # honest about a hole this wrapper closes (Brandon,
        # 2026-08-30). A .vdyn opened into a fresh window IS the
        # session's genesis, so the journal restarts on that line the
        # way Project.open restarts it.
        fresh_open = isinstance(result, io.TestContents) and not self.objects
        with self.project.journal_as(
                None if fresh_open
                else f'project.import_file({str(path)!r})'):

            if isinstance(result, io.TestContents):
                # a saved project: the names are the user's own, verbatim,
                # and its structure comes with it. An empty untouched
                # project adopts the saved one's identity too.
                from ..project import remap_links, retarget

                fresh = not self.objects
                # The roles the project held *before* this file, read
                # before its objects arrive: the type rules place each
                # arrival into the Basis as it comes, and a Basis those
                # rules built out of the file's own objects is not one
                # the file must yield to. Read after the adds, that
                # guess counted as taken, the file's Basis group was
                # demoted to no role, and later-wins carried every
                # member into it: a project emptied and re-imported
                # came back with no Basis at all (Brandon, 2026-09-03).
                taken = {group['role'] for group in self.links
                         if group['role']}
                mapping, contents = {}, list(result.items())
                for n, (name, obj) in enumerate(contents):
                    mapping[name] = self.add_object(name, obj, source=path)
                    if tick is not None:
                        tick(n + 1, len(contents))
                if any(old != new for old, new in mapping.items()):
                    for obj in result.values():
                        retarget(obj, mapping)
                self.project.absorb_links(
                    remap_links(result.links, mapping, taken))
                # the provenance travels like the links, through the same
                # renames — dropped here, a freshly regenerated project
                # opened with no staleness bookkeeping at all (Brandon,
                # 2026-08-23: 'still no refresh button')
                for name, record in result.provenance.items():
                    if name in mapping and record.get('source') in mapping:
                        self.project.provenance[mapping[name]] = {
                            **record, 'source': mapping[record['source']]}
                if fresh:
                    if result.name:
                        self.test_item.setText(0, result.name)
                        self.test_item.setData(0, ROLE_ORIGINAL_NAME, result.name)
                    if result.active_geometry in self.objects:
                        self.set_active_geometry(result.active_geometry)
                    if getattr(result, 'project_type', None):
                        self.set_project_type(result.project_type)
                if self.links:
                    self._links_changed()
                imported = list(mapping.values())
            elif isinstance(result, dict):
                imported = [
                    self.add_object(display_name(type(obj).__name__), obj,
                                    source=path, key=key)
                    for key, obj in result.items()]
            else:
                imported = self.add_object(
                    display_name(type(result).__name__), result, source=path)
        if fresh_open:
            self.project.journal = [
                f'project = visualdynamics.Project.open({str(path)!r})']
        return imported

    def save_test(self) -> None:
        """Save the whole project — every object, under its name — one file."""
        name = self.test_item.text(0)
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Project', f'{name}.vdyn', 'Visual Dynamics files (*.vdyn)')
        if not path:
            return
        # through the verb, not io directly: the verb carries the
        # provenance records (the direct call dropped them, and a
        # GUI-saved project reopened with no staleness bookkeeping)
        # and journals the save like any other act (Brandon,
        # 2026-08-30)
        self.project.name = name
        self.project.save(path)
        count = len(self.objects)
        self._show_status(
            f'Saved {name} ({count} object{"s" * (count != 1)}) to '
            f'{os.path.basename(path)}')

    def _geometry_for_export(self):
        """(name, geometry) a shapes export rides on: the one *linked*
        to the shapes first — explicit beats implicit — else the active
        one, else the only one there is, else nothing."""
        exporting = self.object_item()
        if exporting is not None:
            linked = self.linked_geometry(exporting.text(0))
            if linked is not None:
                return linked
        active = self.objects.get(self.active_geometry)
        if isinstance(active, Geometry):
            return self.active_geometry, active
        geometries = [(name, obj) for name, obj in self.objects.items()
                      if isinstance(obj, Geometry)]
        return geometries[0] if len(geometries) == 1 else None

    #: characters a file name cannot carry on some platform or other,
    #: plus the separators. An object may be called anything at all —
    #: 'FRF 1/2' is a perfectly good name and not a path.
    _UNSAFE_IN_A_NAME = '/\\:*?"<>|'

    def _drag_file_name(self, name: str) -> str:
        safe = ''.join('-' if c in self._UNSAFE_IN_A_NAME or ord(c) < 32
                       else c for c in name).strip(' .') or 'Untitled'
        return f'{safe}.vdyn'

    def copy_selected(self) -> None:
        """Cmd/Ctrl+C in the tree: the selected objects onto the
        clipboard, as objects and as files."""
        names = self.tree.copy_selected()
        if names == [PROJECT_ROW]:
            self._show_status('Copied the project — paste into a folder '
                              'to save it as a .vdyn')
        elif names:
            self._show_status(
                f'Copied {len(names)} object{"s" * (len(names) != 1)} — '
                'paste here to duplicate, or into a folder to export')
        else:
            self._show_status('Select objects in the tree to copy them')

    def paste_objects(self) -> None:
        """Cmd/Ctrl+V in the tree: whatever the clipboard holds that
        this tree can take."""
        if not self.tree.paste():
            self._show_status('Nothing on the clipboard to paste here')

    def _paste_named_objects(self, names):
        """Object names on the clipboard: ours are duplicated through
        the project verb; another window's are imported from the
        files its clipboard payload writes on request."""
        ours = [name for name in names if name in self.objects]
        if ours:
            added = self.project.duplicate(*ours)
            for k, name in enumerate(added):
                self.show_object(name, source=ours[k], select=k == 0)
            self._show_status(
                f'Pasted {len(added)} cop{"y" if len(added) == 1 else "ies"}')
            return
        mime = QApplication.clipboard().mimeData()
        paths = [url.toLocalFile() for url in mime.urls()
                 if url.isLocalFile()] if mime is not None else []
        if paths:
            self._import_after_drop(paths)
        else:
            self._show_status('The copied objects are not in this project')

    def _write_for_drag(self, names, folder):
        """Save what is being dragged out, into `folder`; return the paths.

        `names is None` is the project row: the whole project, exactly
        what Save As writes. Otherwise one `.vdyn` per object, named for
        the object.

        Called by the tree only when a drop target has asked for files,
        so a drag that stays inside the window never reaches here.
        """
        written = []
        if names is None:
            path = os.path.join(
                folder, self._drag_file_name(self.project.name or 'Project'))
            self.project.save(path)
            written.append(path)
        else:
            for name in names:
                obj = self.objects.get(name)
                if obj is None:
                    continue
                written.extend(self._drag_one(name, obj, folder))
        if written:
            self._show_status(
                f'Dragged out {len(written)} '
                f'file{"s" * (len(written) != 1)} as .vdyn')
        return written

    def _drag_one(self, name, obj, folder):
        """One object, in the form it is most useful outside the window.

        A `.vdyn` is the lossless form and, for most objects, the only
        one — nothing else holds a geometry or an FRF whole. Two are
        different:

        - **A channel table is a spreadsheet.** `.xlsx` carries every
          column, every value, and the leading zero on a serial number,
          and it is what a colleague or a calibration lab can actually
          open. It reads back in, so nothing is given up.
        - **Photos are pictures.** A `.vdyn` full of them is useless to
          anything but this application; the images are what somebody
          dragging them to the desktop wants, and their bytes go out
          exactly as they came in.

        Both were one-way doors until the Excel importer and the photo
        exporter existed, which is why the drag only started producing
        them now. Everything else is a `.vdyn`, because for everything
        else that is the only form that comes back.

        A photo set becomes a *folder* of images: one object, several
        pictures, and no picture format holds more than one.
        """
        if isinstance(obj, ChannelTable):
            path = os.path.join(folder, f'{self._drag_file_name(name)[:-5]}.xlsx')
            io.export_file(obj, path, format='excel')
            return [path]
        if isinstance(obj, Photos):
            out = os.path.join(folder, self._drag_file_name(name)[:-5])
            # the folder, not its contents: dropping it on the desktop
            # should land one thing called what the object is called
            io.export_file(obj, out, format='photos')
            return [out]
        path = os.path.join(folder, self._drag_file_name(name))
        io.save(obj, path)
        return [path]

    def save_selected(self) -> None:
        """Write the selected object, in whichever form is asked for.

        One verb, because *Save As* and *Export* were the same act with
        two names — pick an object, pick a file, write it — and a user
        who wanted a UNV had to know that the second menu entry existed
        and that the first would not offer it.

        The format is the dialog's file-type list, which is where a save
        dialog has always asked. `.vdyn` is first because it is the only
        form that comes back whole; the rest are what this particular
        object can be written as, so a mode shape is never offered a
        format that cannot hold one — said in the list rather than in an
        error afterwards.
        """
        if self.tree.currentItem() is self.test_item:
            self.save_test()
            return
        obj = self.current_object()
        if obj is None:
            QMessageBox.information(self, 'Save', 'Select an object to save.')
            return
        native = 'Visual Dynamics object (*.vdyn)'
        available = io.exporters(obj)
        filters = ';;'.join([native, *(f'{e.description} (*{e.suffix})'
                                       for e in available)])
        path, chosen = QFileDialog.getSaveFileName(self, 'Save As', '', filters)
        if not path:
            return
        exporter = next((e for e in available
                         if f'{e.description} (*{e.suffix})' == chosen), None)
        if exporter is None:
            if not path.endswith('.vdyn'):
                path += '.vdyn'
            io.save(obj, path)
            self._show_status(f'Saved {path}')
            return
        self._export_object(obj, path, exporter)

    def _export_object(self, obj, path, exporter):
        """Write one object in a foreign format.

        Values go out in the display unit system: what is on screen is
        what lands in the file. An object with no units declared is
        written as it stands, there being nothing to convert it by.
        """
        if not path.endswith(exporter.suffix):
            path += exporter.suffix
        extras = {}
        rider = ''
        if (isinstance(obj, (ShapeSet, TimeHistory, Spectrum))
                and exporter.name == 'exodus'):
            # exodus is a mesh with results: shapes and data both ride
            # on the active geometry (or the only one), never out
            # alone — and the status line says which, since nothing
            # else would
            what = 'mode shapes' if isinstance(obj, ShapeSet) else 'data'
            found = self._geometry_for_export()
            if found is None:
                self._show_status(
                    f'Writing {what} to exodus needs a geometry in '
                    'the project — the file is a mesh with results')
                return
            geo_name, geometry = found
            dofs = (obj.coordinate if isinstance(obj, ShapeSet)
                    else obj.response_dof)
            # DOF-less records are global variables and need no node,
            # so a data object with any of them always has a fit
            if (not any(not d.strip() for d in dofs)
                    and self._fit_note(geometry, dofs) is None):
                self._show_status(
                    f'{geo_name} has none of the {what} DOFs — set the '
                    'right geometry active (right-click it) and save '
                    'again')
                return
            extras['geometry'] = geometry
            rider = f' on {geo_name}'
        try:
            io.export_file(obj, path, format=exporter.name,
                           unit_system=self.unit_system, **extras)
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, 'Could not save', str(e))
            return
        # name the units: two of the three formats cannot record them, so
        # this line is the only place they are ever stated.
        # The coherent system, not the display one: reading in g does not
        # put g in the file, and the status line must say what did
        written = (f' in {self.unit_system.coherent.name}'
                   if getattr(obj, 'units_defined', False)
                   else ' — units undefined, written as they stand')
        self._show_status(
            f'Saved {os.path.basename(path)}{rider}{written}')

    # ---- rendering ----------------------------------------------------------

    def _selection_changed(self, *_):
        self.data_pane.reset_plot_mode()
        # Making the owner row current means "the whole object", so its
        # grid picks are released — a cell selected minutes ago must not
        # silently turn a later delete-the-object into delete-that-record.
        # Grid-driven changes block tree signals, so reaching here with an
        # owner current item is always the user's own click; and only the
        # *current* item is cleared, so Cmd-clicking a geometry to animate
        # against keeps the records picked in another object's grid.
        current = self.tree.currentItem()
        reference = (current.data(0, ROLE_REFERENCE)
                     if current is not None else None)
        if reference is not None and reference[0] == 'object':
            grid = self.record_grids.get(reference[1])
            if grid is not None and grid.selected_records():
                grid.select_records([])
        # A grid whose owner is no longer selected does not keep its
        # rows lit. The rows only count while the owner is selected
        # (`selected_references`), so a highlight without the owner is
        # a selection the plot will never honor — and it lied twice:
        # a record giving way to a plain click in another grid kept
        # its blue row, so the tree showed two selected sub-items
        # whose plot held one, and the record still drawn read as its
        # own specification missing (Brandon, 2026-08-20); and the
        # remembered rows resurfaced as a phantom restriction the
        # next time the object was clicked. One sweep here covers the
        # grid-click give-way too — the selection model's signal is
        # not the tree's own, so it arrives even from that path.
        still = {ref[1] for it in self.tree.selectedItems()
                 if (ref := it.data(0, ROLE_REFERENCE)) is not None}
        for name, grid in self.record_grids.items():
            if name not in still and grid.selected_records():
                grid.select_records([])
        if self.editing is not None:
            if self._selection_is_being_edited():
                return      # the table drives the view while editing
            self.stop_editing()   # anything else in the tree leaves editing
            return
        if self._importing:
            # an import moves the selection once per object it adds and
            # once more per tree reshuffle a type switch causes — 48
            # selection changes for one four-object file, 31 of them
            # re-building the specification's 3D stage in front of the
            # user. One render, after the loop, shows the same final
            # state (Brandon watched the storm, 2026-08-23).
            return
        self.render_current()

    def _selection_is_being_edited(self):
        """Is the tree selection still the category (or its entities) that
        editing is showing?"""
        if self.editing is None:
            return False
        name, component = self.editing
        for kind, other, _obj, detail in self.selected_references():
            if other != name:
                return False
            if kind == 'component' and detail == component:
                continue
            if ENTITY_COMPONENT.get(kind) == component:
                continue
            return False
        return True

    def _units_changed(self, name):
        self.unit_system = SYSTEMS[name]
        self.render_current()

    def _status_for(self, obj):
        """How big the object is, and anything wrong with it."""
        summary = _summarize(obj)
        if _units_defined(obj):
            return summary
        undefined = getattr(obj, 'undefined_records', None)
        missing = (f'{len(undefined)} of {obj.num_records} channels need units'
                   if undefined else 'no units defined')
        return f'{summary} — {missing}' if summary else missing

    def _show_views(self, three_d=False, plots=False, table=False):
        """Show every pane the selection needs, side by side.

        Selecting a geometry and a time history shows both rather than one
        winning; with nothing selected the 3D view holds the space.
        """
        if not (three_d or plots or table):
            three_d = True
        # whatever is about to be drawn declares which controls it can
        # use; anything left up from the last drawing is not one of them
        self.data_pane.reset_controls()
        panes = (self.scene, self.data_pane, self.table_pane)
        wanted = (three_d, plots, table)
        for pane, visible in zip(panes, wanted):
            pane.setVisible(visible)
        # a pane revealed after being hidden can come back with no size.
        # Splitter positions, not tuple positions: comparing reorders them
        sizes = self.views.sizes()
        showing = [self.views.indexOf(pane)
                   for pane, visible in zip(panes, wanted) if visible]
        if any(sizes[i] == 0 for i in showing):
            total = sum(sizes) or self.views.height() or 800
            share = total // len(showing)
            self.views.setSizes([share if i in showing else 0
                                 for i in range(len(panes))])

    def _clear_views(self):
        """Show nothing, because there is nothing selected to show.

        The report editor goes too. It is the one pane that is not
        rebuilt on every render — it holds a Chromium view of one report
        and is only ever *shown*, by the renderer that has a report to
        put in it — so nothing was taking it down. Deleting every object
        in a project left its report on screen, still editable, backed by
        an object no longer in the project.
        """
        self.scene.clear()
        self.scene.offer_rigid(False)
        self.data_pane.clear()
        self.table.setModel(None)
        if self.report_editor is not None:
            # hidden, not forgotten. `_clear_views` runs whenever nothing
            # is selected, which is often and is not the same as the
            # report having gone — dropping what the editor holds there
            # makes the next render reload the page, and an edit issued
            # against the old one lands on a document being replaced.
            self.report_editor.hide()

    def _forget_report_editor(self):
        """Take the report editor down and let go of what it was showing.

        Letting go matters as much as hiding, and only when the report is
        actually gone: `_render_report_builder` skips the rebuild when it
        is asked for the report it already holds, so an editor still
        pointing at a deleted one would come back showing it if a new
        report were later selected into the same pane.
        """
        if self.report_editor is None:
            return
        self.report_editor.hide()
        self.report_editor.report = None

    def render_current(self) -> None:
        """Render everything selected: geometries overlay, curves share axes."""
        self._update_toolbar_actions()
        # the frames and the shock windows belong to the plot that was
        # up; whatever is drawn next puts its own back if it wants them
        self._clear_averaging()
        self._clear_shocks()
        self._clear_filtering()
        self._clear_truncation()
        self._clear_octave()
        # and so does the 3-D surface: the waterfall path raises it
        # again itself, so a photo, a report, a fit or an empty
        # selection never inherits the last selection's 3-D view. Here
        # rather than in _show_views because the empty-selection path
        # exits before _show_views runs
        self.data_pane.show_waterfall(False)
        references = self.selected_references()
        # the fit owns the panes while its FRF stays selected; looking at
        # anything else ends the fit (the fitted modes are already in the
        # project — nothing is lost by leaving)
        if self.fit is not None:
            selected = {name for _kind, name, _obj, _detail in references}
            if selected and selected <= {self.fit_name, self.fit_object_name}:
                return
            self.stop_fitting()
        # one FRF and one shape set, both whole, read two ways — the
        # Edit Fit / Resynthesis buttons choose, and the choice sticks.
        # Sub-item picks keep their own meanings (a mode pick beside an
        # FRF is always the synthesis overlay), so they do not enter here.
        chosen = {name: obj for _kind, name, obj, _detail in references}
        is_pair = (len(chosen) == 2
                   and sum(isinstance(obj, Frf)
                           for obj in chosen.values()) == 1
                   and sum(isinstance(obj, ShapeSet)
                           for obj in chosen.values()) == 1)
        # mode or record picks keep the overlay showing whatever the
        # toggle last said — browsing a truncated synthesis must not keep
        # snapping into the fit — but the Edit button stays offered
        self._pair_selected = is_pair and all(
            kind == 'object' for kind, *_rest in references)
        self._pair_with_picks = is_pair and all(
            kind in ('object', 'mode', 'record')
            for kind, *_rest in references)
        if (self._pair_selected and self.pair_mode == 'fit'
                and self.fit is None):
            self.start_modal_fit()
            return
        # the pane edits what is selected; looking at something else puts it
        # away, and so does deleting the object out from under it
        if self.units_target is not None and (
                self.objects.get(self.units_target[0])
                is not self.units_target[1]
                or self.units_target[0] not in {
                    name for _kind, name, _obj, _detail in references}):
            self.close_units_panel()
        # The project row alone shows nothing on the right. Drawn as
        # "everything it holds" it was every geometry overlaid, fifty
        # curves on one axis and the channel table, and the act a person
        # came to it for — Generate Report — sat a third of the way down
        # the window on the time data's bar (Brandon, 2026-09-09: "I'd
        # almost rather see nothing in the right screen when the top
        # level project is selected but still see the generate report
        # option in the toolbar"). So the panes clear, the 3-D view
        # holds the space with the project's own acts on its bar, and
        # the status line says what the project holds. The row still
        # *means* everything for copy and rename — that is the tree's
        # vocabulary, not the render's.
        if (references and self.test_item.isSelected()
                and all(item is self.test_item
                        for item in self.tree.selectedItems())):
            self._clear_views()
            self._show_views()
            self._offer_acts()
            n = len(self.objects)
            self._show_status(
                f'{self.test_item.text(0)}: {n} object{"s" * (n != 1)} — '
                'Generate Report is on the bar; select an object to see it')
            return
        if not references:
            self._clear_views()
            self._offer_acts()
            if self.tree.currentItem() is self.test_item:
                self._show_status(
                    f'{self.test_item.text(0)} is empty — import a file, '
                    'or drag files onto the tree')
            else:
                self._show_status(
                    'Import a file to get started — or drag files onto the '
                    'project tree')
            return

        geometries, series, channels, shapes = {}, [], [], []
        reports, picture_sets, matches, sine_specs = [], [], [], []
        sine_levels = []
        for kind, name, obj, detail in references:
            if isinstance(obj, Geometry):
                entry = geometries.setdefault(
                    name, {'object': obj, 'components': set(), 'entities': {}})
                if kind == 'component' and detail:
                    entry['components'].add(detail)
                elif kind in ENTITY_COMPONENT:
                    entry['entities'].setdefault(
                        ENTITY_COMPONENT[kind], []).append(detail)
            elif isinstance(obj, DataArray):
                # one entry per object, not per record: ten cells picked in
                # a grid are one time history restricted to ten records,
                # and the animator refuses a selection of ten objects
                if kind == 'record':
                    for entry_name, _obj, records in series:
                        if entry_name == name and records is not None:
                            records.append(detail)
                            break
                    else:
                        series.append((name, obj, [detail]))
                else:
                    series.append((name, obj, None))
            elif isinstance(obj, ShapeSet):
                shapes.append((name, obj, detail if kind == 'mode' else None))
            elif isinstance(obj, ChannelTable):
                channels.append((name, obj, detail if kind == 'channel' else None))
            elif isinstance(obj, Report):
                reports.append((name, obj))
            elif isinstance(obj, MatchedModes):
                matches.append((name, obj))
            elif isinstance(obj, (SineSweepSpecification, SineLevelSet)):
                # picked tones restrict either object, exactly as
                # picked records restrict a data object
                into = (sine_specs
                        if isinstance(obj, SineSweepSpecification)
                        else sine_levels)
                if kind == 'tone':
                    for entry_name, _obj, picks in into:
                        if entry_name == name and picks is not None:
                            picks.append(detail)
                            break
                    else:
                        into.append((name, obj, [detail]))
                else:
                    into.append((name, obj, None))
            elif isinstance(obj, Photos):
                # picked photos restrict the object, exactly as records do
                if kind == 'photo':
                    for entry_name, _obj, picks in picture_sets:
                        if entry_name == name and picks is not None:
                            picks.append(detail)
                            break
                    else:
                        picture_sets.append((name, obj, [detail]))
                else:
                    picture_sets.append((name, obj, None))

        self._update_dof_controls(geometries, series)
        self._update_link_actions()
        shape_table = bool(shapes) and not geometries and not series
        shape_sets = list(dict.fromkeys(
            (name, id(obj)) for name, obj, _detail in shapes))
        comparing = (len(shape_sets) == 2 and len(geometries) <= 1
                     and not series and not any(
                         entry['components'] or entry['entities']
                         for entry in geometries.values()))
        self._set_compare_layout(comparing)
        # a comparison if there is one to make, otherwise what the
        # specification says on its own — either way a table per channel
        # that the plot follows
        # what a specification says on its own, a row per channel. A
        # specification *with* a measurement has no table: the bar
        # charts carry those numbers, and carry them better.
        specifications = (self._specification_rows(series)
                          if series and not channels else None)
        # the levels are a reading asked for, not the default: the
        # spectra alone by default, and the RMS toggle brings the bars
        # up with the table beneath (Brandon, 2026-09-06)
        levels = (specifications
                  if specifications is not None and self.data_pane.rms_wanted
                  else None)
        # a transient record beside its target splits the space: the
        # waveform on top, and under it every control channel with how
        # far it is from what was asked. The plot draws one channel or a
        # few, so the table is where the rest are still accounted for
        replicating = bool(series) and self._replication_found(series)
        authoring = self._authoring(shapes, geometries, series, channels)
        # no 3-D view in edit mode (Brandon, 2026-09-06): every editing
        # gesture is on the flat plot, so the stage stands down while
        # the sheet is open and comes back when it closes
        self.data_pane.flat_only = authoring is not None
        self._show_views(three_d=bool(geometries),
                         plots=bool(series) or bool(picture_sets)
                         or bool(sine_specs) or bool(sine_levels)
                         or authoring is not None,
                         table=bool(channels) or shape_table or comparing
                         or bool(reports) or bool(matches)
                         or levels is not None
                         or bool(replicating)
                         or bool(series and self._srs_found(series))
                         or (bool(series)
                             and self._compliance_rows(series) is not None))
        if not (geometries or series or channels or picture_sets
                or matches or sine_specs or sine_levels):
            self._clear_views()


        # the rigid-body reading is a reading of one whole geometry with
        # nothing riding it: no shapes or data to animate, no part picked
        lone = (len(geometries) == 1 and not shapes and not series
                and not channels and self.editing is None
                and not any(entry['components'] or entry['entities']
                            for entry in geometries.values()))
        self.scene.offer_rigid(lone)
        deflection = self._deflection_for(geometries, series, shapes)
        parts = []
        synthesis_note = ''
        if deflection is not None:
            parts.append(deflection)
            if series:
                parts.append(self._render_series(series, cursor=True))
        else:
            if geometries:
                parts.append(self._render_geometries(list(geometries.items())))
            if sine_specs or sine_levels:
                parts.append(self._render_sine(
                    sine_specs[0] if sine_specs else None,
                    sine_levels, series))
            elif series:
                overlay, synthesis_note = self._synthesis_series(shapes,
                                                                 series)
                parts.append(self._render_series(series + overlay))
                if synthesis_note:
                    parts.append(synthesis_note)
        if picture_sets and not series:
            parts.append(self._render_photos(picture_sets))
        if reports:
            parts.append(self._render_report_builder(*reports[0]))
        if channels:
            name, table, _detail = channels[0]
            # picked cells in the tree's grid are the rows shown here;
            # selecting the object (no picks) shows the whole table
            rows = sorted({detail for n, _table, detail in channels
                           if n == name and detail is not None})
            summary = self._render_table(table, rows=rows or None, name=name)
            tables = {n for n, _table, _detail in channels}
            if len(tables) > 1:
                summary += f' (1 of {len(tables)} tables selected)'
            parts.append(summary)
        if levels is not None:
            parts.append(self._render_specifications(levels))
        if matches and not comparing and not channels:
            parts.append(self._render_matches(*matches[0]))
        if shapes and deflection is None and not synthesis_note:
            if shape_table:
                sets = list(dict.fromkeys(
                    (name, id(obj)) for name, obj, _detail in shapes))
                if len(sets) == 2:
                    # the same interactive pairing as the animated
                    # comparison — no geometry required to match modes
                    summary = self._compare_flat(shapes)
                else:
                    summary = self._render_shape_table(
                        shapes[0][1], mode=shapes[0][2],
                        name=shapes[0][0])
                parts.append(summary)
            else:
                parts.append(self._shape_summary(shapes, geometries))
        if authoring is not None:
            parts.append(self._render_author(authoring))
            if authoring[0] == 'table':
                # the table's bar is put away by its own renderer; the
                # sheet's toggle lives there, so it comes back alone
                self.mode_table_action.setVisible(False)
                self.mac_bars_action.setVisible(False)
                self.table_bar.show()
        elif self.author_action.isVisible() and channels and not shapes:
            self.mode_table_action.setVisible(False)
            self.mac_bars_action.setVisible(False)
            self.table_bar.show()
        self._offer_acts()
        self._show_status('  |  '.join(part for part in parts if part))

    def _empty_category(self, geometries):
        """The name of a selected category holding nothing, else None.

        Selecting one only says so: opening the table is what Edit is for,
        and an empty category is a common thing to click past.
        """
        if self.editing is not None or len(geometries) != 1:
            return None
        _name, entry = next(iter(geometries.items()))
        components = entry['components']
        if len(components) != 1 or entry['entities']:
            return None
        component = next(iter(components))
        counts = {'nodes': entry['object'].num_nodes,
                  'coordinate_systems': len(entry['object'].cs_id),
                  'tracelines': len(entry['object'].traceline_conn),
                  'elements': len(entry['object'].elem_conn),
                  'blocks': len(entry['object'].block_id)}
        return component if not counts.get(component, 1) else None

    def _deflection_for(self, geometries, series, shapes):
        """Show data on the geometry: one shape set animates, two compare.

        Returns a status line, or None when there is nothing to deflect and
        the static renderers should take over.
        """
        self._clear_animator()
        if len(geometries) != 1:
            return None
        name, entry = next(iter(geometries.items()))
        geometry = entry['object']
        if entry['components'] or entry['entities']:
            return None          # a sub-selection means 'show me this part'

        if not shapes and not series and self.scene.showing_rigid:
            return self._deflect_rigid(name, geometry, showing=name)
        if shapes and not series:
            sets = list(dict.fromkeys(n for n, _obj, _detail in shapes))
            if len(sets) == 2:
                return self._deflect_comparison(name, geometry, shapes)
            return self._deflect_shape(geometry, shapes, showing=name)
        if series and not shapes:
            return self._deflect_series(geometry, series, showing=name)
        return None

    def _size_tree_once(self):
        """Give the project dock a sensible width the first time the
        window is shown, and never again.

        It used to refit to its content on every import, expand, collapse
        and rename. That reads well in a demo and badly in use: moving
        between a geometry and a photo set stepped the dock in and out
        under the cursor, because the two names are different lengths.
        How wide it is is the user's.

        Deferred a turn so Qt has finished the first layout.
        """
        QTimer.singleShot(0, self._apply_tree_width)

    def _apply_tree_width(self):
        # a floating panel is sized by the user, and resizeDocks on a dock
        # that is not in the layout has nothing legitimate to act on — on
        # macOS it disturbed the main window's own layout instead
        if self.project_dock.isFloating():
            return
        content = self._visible_content_width()
        chrome = (self.tree.frameWidth() * 2
                  + self.tree.verticalScrollBar().sizeHint().width())
        # a fifth of the window, or more if the content needs it: fitted
        # to an empty tree this opened as a strip too narrow to read a
        # name in, let alone drop a file on
        wanted = max(content + chrome + 8, self.width() // 5)
        low = self.tree.header().sectionSizeHint(0) + chrome
        high = max(low, self.width() // 2)
        self.resizeDocks([self.project_dock],
                         [max(low, min(wanted, high))], Qt.Orientation.Horizontal)

    def _visible_content_width(self):
        """The widest row currently on screen, in content coordinates.

        `sizeHintForColumn` walks every row, and a demoted 6859-record list
        made every deletion pay for all of them. The dock fits what is
        *showing* — which is also the honest reading of fitting to content —
        so only the viewport's worth of rows is measured.
        """
        viewport = self.tree.viewport()
        widest, y = 0, 0
        while y < viewport.height():
            item = self.tree.itemAt(0, y)
            if item is None:
                break
            rect = self.tree.visualItemRect(item)
            # the declared hint, not the current width: a grid squeezed into
            # the old dock width reports what it *has*, and fitting to that
            # would never widen anything
            hint = item.sizeHint(0)
            width = (rect.x() + hint.width() if hint.isValid()
                     and hint.width() > 0
                     else rect.x() + self.tree.fontMetrics().horizontalAdvance(
                         item.text(0)) + self.tree.iconSize().width() + 12)
            widest = max(widest, width)
            y = rect.bottom() + 1
        return widest

    @staticmethod
    def _fit_note(geometry, dofs):
        """How well these DOFs fit the geometry, as a status-line suffix.

        None when nothing lands on it at all — there is no animation to
        show. Otherwise '' for a clean fit, or a note naming what is
        missing: data that covers more nodes than the geometry still
        animates the nodes that do exist, rather than refusing outright.
        """
        missing = geometry.missing_dofs(dofs)
        if len(missing) >= len(dofs):
            return None
        if not len(missing):
            return ''
        return (f' — {len(missing)} of {len(dofs)} DOFs are not on this '
                'geometry and are not shown')

    def _set_compare_layout(self, comparing):
        """The MAC pane sits left of the 3D view while comparing — read
        the MAC, look right at the animation — and goes back after."""
        if comparing == self._compare_active:
            if not comparing:
                self._compare = None
            return
        self._compare_active = comparing
        if comparing:
            self.views.insertWidget(0, self.table_pane)
            # side by side, not stacked: the MAC reads at the animation's
            # left, not over its head
            self.views.setOrientation(Qt.Orientation.Horizontal)
            # an even split, set outright: leftover splitter positions
            # from the stacked layout squeezed the animation to a sliver
            total = (sum(self.views.sizes()) or self.views.width()
                     or 1200)
            sizes = [0] * self.views.count()
            sizes[self.views.indexOf(self.table_pane)] = total // 2
            sizes[self.views.indexOf(self.scene)] = total - total // 2
            self.views.setSizes(sizes)
        else:
            self.views.insertWidget(2, self.table_pane)
            self.views.setOrientation(Qt.Orientation.Vertical)
            self._compare = None
            self.add_matches_action.setVisible(False)
            self.overlay_action.setVisible(False)
            self.project_action.setVisible(False)

    def _show_bracket_menu(self, span, position):
        """Right-clicking a link bracket: the group's own options.
        Placeholder-only brackets hold no group yet and offer none."""
        if span >= len(self.links):
            return
        group = self.links[span]
        menu = QMenu(self.tree)
        basis = menu.addAction('&Basis of Comparisons')
        basis.setCheckable(True)
        basis.setChecked(group['role'] == 'Basis')
        basis.triggered.connect(
            lambda _checked=False, name=group['members'][0]:
            self.set_link_role(
                name, None if self.link_role(name) == 'Basis'
                else 'Basis'))
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _render_matches(self, name, matched):
        """The matched-modes object on its own: the table, rows
        deletable — deleting a row removes the match."""
        self._set_table_model(matched_modes_model(matched, self.objects,
                                                  self))
        self._arm_row_deletion(name, 'match')
        self.table.show()
        gone = [set_name for set_name in (matched.first, matched.second)
                if set_name not in self.objects]
        note = (f' — {", ".join(gone)} no longer in the project'
                if gone else '')
        return (f'{name}: {matched.num_matches} matched '
                f'pair{"s" * (matched.num_matches != 1)} between '
                f'{matched.first} and {matched.second}{note}')

    def _update_pairs_action(self):
        self.add_matches_action.setVisible(
            self._compare is not None
            and bool(self._compare.get('pairs')))

    def _matched_for(self, a_name, b_name):
        """(name, MatchedModes) between these two sets, or None."""
        return next(
            ((name, obj) for name, obj in self.objects.items()
             if isinstance(obj, MatchedModes)
             and obj.first == a_name and obj.second == b_name),
            None)

    def add_matches(self) -> None:
        """Commit the selected MAC squares to the matched modes for
        this pair of sets — the project creates or extends the object,
        links it to the Basis group, and keeps the MAC values that were
        displayed (a name-matched recompute would not reproduce a
        projected comparison)."""
        if self._compare is None or not self._compare.get('pairs'):
            return
        a_name, b_name = self._compare['pair']
        matrix = self._compare['matrix']
        pairs = [list(pair) for pair in self._compare['pairs']]
        macs = [float(matrix[r, c]) for r, c in self._compare['pairs']]
        found = self._matched_for(a_name, b_name)
        if found is None:
            selected = self.tree.selectedItems()
            current = self.tree.currentItem()
            name = self.project.match_modes(a_name, b_name, pairs=pairs,
                                            macs=macs)
            self.show_object(name, select=False)
            # the comparison is the user's context: put it back
            if current is not None:
                self.tree.setCurrentItem(current)
            self.tree.clearSelection()
            for item in selected:
                item.setSelected(True)
            matched = self.objects[name]
        else:
            name, matched = found
            matched.add(pairs, macs)
            # growing an existing set is an act like creating one: the
            # first commit journals through the match_modes verb, and
            # this is the same statement made again (Brandon,
            # 2026-08-30, the audit's second pass)
            self.project.record_call(
                matched, 'add',
                [(int(a), int(b)) for a, b in pairs],
                [float(m) for m in macs])
            a_home = self.project.geometry_for(a_name)
            b_home = self.project.geometry_for(b_name)
            was = (matched.first_geometry, matched.second_geometry)
            matched.first_geometry = (matched.first_geometry
                                      or (a_home[0] if a_home else None))
            matched.second_geometry = (matched.second_geometry
                                       or (b_home[0] if b_home else None))
            if (matched.first_geometry, matched.second_geometry) != was:
                self.project.record_setting(
                    matched, 'first_geometry', matched.first_geometry)
                self.project.record_setting(
                    matched, 'second_geometry', matched.second_geometry)
            self._refresh_item(self._item_for_object(name), matched)
        self.render_current()
        self._show_status(
            f'{name}: {len(pairs)} '
            f'match{"es" * (len(pairs) != 1)} added '
            f'({matched.num_matches} total)')

    def _comparison_basis(self, names):
        """Which of two selected sets leads a comparison: the declared
        Basis, else — when the two live on different geometries — the
        sparser set, whose DOFs are what the other can be sampled at,
        else the first selected."""
        a_name, b_name = names
        roles = [self.link_role(name) for name in names]
        if 'Basis' in roles and roles[0] != roles[1]:
            return b_name if roles[1] == 'Basis' else a_name
        homes = [self.linked_geometry(name) for name in names]
        if (None not in homes and homes[0][1] is not homes[1][1]
                and len(self.objects[b_name].coordinate)
                < len(self.objects[a_name].coordinate)):
            return b_name
        return a_name

    def _comparison_matrix(self, a_name, a, b_name, b):
        """The cross-MAC, in `a`'s DOF space. When the two sets live
        on different linked geometries, `b` is projected onto `a`'s
        DOFs first — matching DOF names across geometries would trust
        the names to mean the same physical directions. Returns
        (matrix, note, projected): the note says what the projection
        did, and the projected set (or None) is what an overlay
        animation of `b` should deflect."""
        from ..core.correlate import project_shapes

        a_home = self.linked_geometry(a_name)
        b_home = self.linked_geometry(b_name)
        if a_home is None or b_home is None or a_home[1] is b_home[1]:
            return cross_mac(a, b), None, None
        try:
            projected, report = project_shapes(b, b_home[1], a,
                                               a_home[1])
        except ValueError as refusal:
            raise ValueError(
                f'{refusal} (Project onto Basis DOFs offers a looser '
                'tolerance)') from refusal
        dropped = (f', {len(report["dropped"])} DOFs dropped'
                   if report['dropped'] else '')
        note = (f'{b_name} projected onto {a_name} DOFs — '
                f'{report["matched"]} of {report["total"]} nodes '
                f'matched{dropped}')
        return cross_mac(a, projected), note, projected

    def _compare_mac_view(self, shapes):
        """The interactive cross-MAC both comparison screens share:
        the matrix, the picked pairs outlined (the animated one
        bolder), and the pairing table. The basis set leads — rows,
        table, frequency-error baseline — and a set on a different
        geometry is projected onto it before the MAC is taken.
        Returns the view's pieces, or an error string."""
        names = list(dict.fromkeys(n for n, _obj, _detail in shapes))
        if self._comparison_basis(names) != names[0]:
            names.reverse()
        a_name, b_name = names
        a, b = self.objects[a_name], self.objects[b_name]
        try:
            matrix, note, projected = self._comparison_matrix(
                a_name, a, b_name, b)
        except ValueError as why:
            self._clear_animator()
            self._hide_mac()
            return f'{a_name} × {b_name}: {why}'
        picked = {n: d for n, _obj, d in shapes if d is not None}
        if self._compare is None or self._compare['pair'] != (a_name,
                                                              b_name):
            row = min(picked.get(a_name, 0), a.num_shapes - 1)
            column = picked.get(b_name)
            if column is None:
                column = int(np.argmax(matrix[row]))
            cell = (row, int(min(column, b.num_shapes - 1)))
            self._compare = {'pair': (a_name, b_name), 'cell': cell,
                             'pairs': [cell]}
        row, column = self._compare['cell']
        pairs = self._compare.setdefault('pairs', [(row, column)])
        self._compare['matrix'] = matrix
        self._compare['note'] = note
        self._compare['projected'] = projected
        # the grid reads tall: whichever set has more modes makes the
        # rows, and clicks map back through the same orientation
        tall = b.num_shapes > a.num_shapes
        self._compare['tall'] = tall
        # the table shows the *committed* matches for this pair of
        # sets — the object the + button builds — not the transient
        # square selection
        found = self._matched_for(a_name, b_name)
        committed = (found[1] if found is not None
                     else MatchedModes(a_name, b_name))
        self._set_table_model(matched_modes_model(
            committed, self.objects, self))
        self._arm_matched_rows(committed)
        if found is not None:
            self._arm_row_deletion(found[0], 'match')
        self.table.setVisible(self.mode_table_action.isChecked())
        if [row, column] in committed.pairs:
            self.table.selectRow(committed.pairs.index([row, column]))
        self._render_cross_mac((a_name, a), (b_name, b), matrix)
        if self.mac_bars_action.isChecked():
            # the bars drew their own marks; the overlay below belongs
            # to the flat grid, which was not rebuilt
            self._update_pairs_action()
            return a_name, a, b_name, b, matrix, row, column
        import pyqtgraph as pg
        plot = next(item for item in self.mac_view.ci.items
                    if hasattr(item, 'getViewBox'))
        def outline(r: int, c: int, pen: Any) -> None:
            x, y = (r, c) if tall else (c, r)
            plot.addItem(pg.PlotCurveItem(
                [x, x + 1, x + 1, x, x], [y, y, y + 1, y + 1, y],
                pen=pen))
        if committed.pairs:
            # committed matches wear a red checker: already in the
            # table (outlines alone vanished against the viridis field,
            # and Brandon preferred the checker to diagonal stripes).
            # One path item for all of them, filled, no pen
            from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
            from PySide6.QtWidgets import QGraphicsPathItem

            from ..viz.mac_bars import CHECKER, MARK_RED

            path = QPainterPath()
            step = 1.0 / CHECKER
            for r, c in committed.pairs:
                x, y = (r, c) if tall else (c, r)
                for i in range(CHECKER):
                    for j in range(CHECKER):
                        if (i + j) % 2 == 0:
                            path.addRect(x + i * step, y + j * step,
                                         step, step)
            checker = QGraphicsPathItem(path)
            checker.setBrush(QBrush(QColor(MARK_RED)))
            checker.setPen(QPen(Qt.PenStyle.NoPen))
            #: how the test tells this overlay from every other item
            checker.mac_checker = True
            plot.addItem(checker)
        for r, c in pairs:
            # the selection outlines blue — apart from the committed
            # red, which an outline of the same red disappeared
            # against — and the animated square boldest of all
            from ..viz.mac_bars import PICK_BLUE

            active = (r, c) == (row, column)
            outline(r, c, pg.mkPen(PICK_BLUE,
                                   width=5 if active else 3))
        self._update_pairs_action()
        return a_name, a, b_name, b, matrix, row, column

    def _compare_flat(self, shapes):
        """Two shape sets selected alone: the interactive MAC, the
        matched-modes table beside it, and — when a linked geometry
        can host it and the toggle says so — the overlaid animation."""
        view = self._compare_mac_view(shapes)
        if isinstance(view, str):
            self._clear_animator()
            return view
        a_name, a, b_name, b, matrix, row, column = view
        self.table_bar.show()
        home = (self.linked_geometry(a_name)
                or self.linked_geometry(b_name))
        self.overlay_action.setVisible(home is not None)
        self.project_action.setVisible(
            home is not None
            and self._compare.get('projected') is not None)
        problem = None
        animated = False
        if home is not None and self.overlay_action.isChecked():
            problem = self._animate_pair(a_name, a, row, b_name, b,
                                         column, fallback=home)
            animated = problem is None
        if not animated:
            self._clear_animator()
        self._show_views(three_d=animated, table=True)
        note = self._compare.get('note')
        return (f'{a_name} {a.mode_label(row)} vs {b_name} '
                f'{b.mode_label(column)} — MAC {matrix[row, column]:.2f}; '
                'click picks a pair, Shift adds it to the matched table'
                + (f' ({note})' if note else '')
                + (f' ({problem})' if problem else ''))

    def _deflect_comparison(self, geo_name, geometry, shapes):
        """Two shape sets and a geometry selected: the same comparison,
        animating when the toggle says so — each set on its own linked
        geometry, the selected one standing in for unlinked sets."""
        view = self._compare_mac_view(shapes)
        if isinstance(view, str):
            return view
        a_name, a, b_name, b, matrix, row, column = view
        self.table_bar.show()
        self.overlay_action.setVisible(True)
        self.project_action.setVisible(
            self._compare.get('projected') is not None)
        summary = (f'{a_name} {a.mode_label(row)} vs {b_name} '
                   f'{b.mode_label(column)} — MAC '
                   f'{matrix[row, column]:.2f}; '
                   'click a MAC cell to compare another pair')
        if self.overlay_action.isChecked():
            problem = self._animate_pair(a_name, a, row, b_name, b,
                                         column,
                                         fallback=(geo_name, geometry))
            if problem is not None:
                self._clear_animator()
                return f'{summary} ({problem})'
            return summary
        self._clear_animator()
        self._render_geometries([(geo_name, {'object': geometry,
                                             'components': set(),
                                             'entities': {}})])
        return summary

    def _animate_pair(self, a_name, a, row, b_name, b, column,
                      fallback):
        """The picked pair of modes overlaid, each set deflecting its
        own linked geometry — basis shapes on the basis mesh, the
        other set on its own — with the second phase-aligned to the
        first so they move together, drawn in one flat color each to
        stay tellable apart. When the geometries differ the alignment
        is measured through the projection (the sets themselves share
        no DOFs) and applied to the raw second set. `fallback` hosts a
        set with no linked geometry of its own. Returns None, or why
        the animation cannot happen."""
        from ..core.shapes import alignment_factor, apply_alignment

        a_home = self.linked_geometry(a_name) or fallback
        b_home = self.linked_geometry(b_name) or a_home
        a_geo_name, a_geometry = a_home
        b_geo_name, b_geometry = b_home
        if self._fit_note(a_geometry, a.coordinate) is None:
            return 'no animation: DOFs are not on the geometry'
        projected = self._compare.get('projected')
        if a_geometry is b_geometry or projected is None:
            if self._fit_note(b_geometry, b.coordinate) is None:
                return 'no animation: DOFs are not on the geometry'
            aligned = aligned_mode(a, row, b, column)
            second = (b_geometry,
                      ShapeDeflection(b_geometry, b.coordinate,
                                      aligned))
            showing = (a_geo_name, b_geo_name)
        elif self.project_action.isChecked():
            # the projection lives at the basis DOFs: both sets on
            # the basis mesh, sampled apples to apples
            aligned = aligned_mode(a, row, projected, column)
            second = (a_geometry,
                      ShapeDeflection(a_geometry, projected.coordinate,
                                      aligned))
            showing = (a_geo_name, 'projected')
        else:
            aligned = apply_alignment(
                b.shape_matrix[column],
                alignment_factor(a, row, projected, column))
            second = (b_geometry,
                      ShapeDeflection(b_geometry, b.coordinate,
                                      aligned))
            showing = (a_geo_name, b_geo_name)
        # each mesh wears its own bracket's color, so the scene reads
        # straight off the project tree; unlinked sets keep the old
        # natural-vs-orange contrast
        color_a = self._link_color(a_name)
        color_b = self._link_color(b_name)
        if color_b is None or color_b == color_a:
            color_b = next(c for c in ('#ff8c2b', '#3fb950')
                           if c != color_a)
        self._compare['colors'] = (color_a, color_b)
        # The basis is the one being looked *at*; the other is drawn
        # through it. Which of the pair that is depends on the project,
        # not on the order they were selected in — so ask the links.
        basis_is_second = (self.link_role(b_name) == 'Basis'
                           and self.link_role(a_name) != 'Basis')
        alphas = ((OVERLAY_ALPHA, 1.0) if basis_is_second
                  else (1.0, OVERLAY_ALPHA))
        self._compare['alphas'] = alphas
        self._shape_mode = True
        self._shape_source = None

        def described(name: str, shape_set: Any, mode: int) -> str:
            description = str(shape_set.description[mode]).strip()
            return (f'{name}  mode {mode + 1}'
                    + (f'  {description}' if description else '')
                    + f'  {float(shape_set.frequency[mode]):.4f} Hz'
                    f'  {float(shape_set.damping[mode]) * 100:.3f} %')

        # each copy is drawn to its own peak, so the scene compares
        # shape and says nothing about level — and the one number that
        # says what the scaling did goes on its own line
        caption = (described(a_name, a, row) + '\n'
                   + described(b_name, b, column))
        scaling = self._scale_note(a_name, a, b_name, b, row, column)
        if scaling:
            caption += '\n' + scaling
        self._build_comparison_animator(
            (a_geometry,
             ShapeDeflection(a_geometry, a.coordinate,
                             a.shape_matrix[row])),
            second,
            caption=caption,
            showing=showing, colors=(color_a, color_b), alphas=alphas)
        # the phase carries over: stepping through MAC cells mid-swing
        # must not snap the model straight every time
        self.animator.set_parameter(
            self.phase_slider.value() / PHASE_STEPS * 2 * np.pi)
        self._show_animation_controls(True, phase=True)
        self.mode_box.setVisible(False)   # the MAC cell is the mode picker
        self.play_action.setEnabled(True)
        return None

    def _scale_note(self, a_name, a, b_name, b, row, column):
        """One line: how big this pair's second shape is against its first.

        Overlaying scales each shape to its own peak, which is what
        makes a sparse test set and a dense model comparable — and also
        what hides one set being thirty times the other. So the number
        the overlay cannot show is written above it.

        **For the pair on screen, not for the sets.** A constant factor
        across every mode is a unit or a normalization convention; a
        factor that is 1.00 everywhere and 3.4 on one mode is that mode
        fitted badly, and a figure averaged over the set hides exactly
        that. The MAC cell picks one pair, so this answers about the one
        pair. `ScaleComparison.message` is the other reading — over a
        whole set, at length — and the report still uses it.

        Named by the objects rather than by their roles: 'basis' and
        'other' are a property of the link groups, and a reader looking
        at two animations wants to know which of the two things in front
        of them is the bigger.
        """
        from ..core.shapes import compare_scaling

        try:
            ratios = compare_scaling(a, b, [(row, column)]).ratios
        except (ValueError, IndexError):
            return None
        if not ratios.size or not np.isfinite(ratios[0]):
            return None
        # an unscaled fit has no drive point pinning its size, so the
        # ratio is arithmetic rather than physics — said in a word,
        # because the number is about to be read as if it meant one
        unscaled = [name for name, shapes in ((a_name, a), (b_name, b))
                    if getattr(shapes, 'unscaled', False)]
        note = f'{b_name}/{a_name} = {float(ratios[0]):.2f}'
        return note + (f'  ({", ".join(unscaled)} unscaled)'
                       if unscaled else '')

    def _build_comparison_animator(self, first_pair, second_pair,
                                   caption, showing,
                                   colors=(None, '#ff8c2b'),
                                   alphas=(1.0, OVERLAY_ALPHA)):
        """Like _build_animator, twice: two moving copies in one
        scene, each (geometry, deflection) pair on its own mesh, in
        its own color and at its own opacity."""
        from ..viz.animate import GeometryAnimator, PairedAnimator

        theme = resolve_theme(self.theme_name)
        reframe = showing is not None and tuple(showing) != self._framed
        if showing is not None:
            self._framed = tuple(showing)
        camera = self.scene.plotter.camera_position
        self.scene.plotter.clear()
        self.scene.plotter.set_background(theme['scene_background'],
                                    top=theme['scene_background_top'])
        first = GeometryAnimator(
            self.scene.plotter, first_pair[0], first_pair[1],
            unit_system=self.unit_system, text_color=theme['scene_text'],
            color_override=colors[0], opacity=alphas[0])
        second = GeometryAnimator(
            self.scene.plotter, second_pair[0], second_pair[1],
            unit_system=self.unit_system, text_color=theme['scene_text'],
            color_override=colors[1], opacity=alphas[1])
        self.animator = PairedAnimator(first, second)
        self.animator.set_scale(self.scale_box.value())
        self.scene.axis_unit = self.animator.axis_unit
        annotate_scene(self.scene.plotter, self.scene.axis_unit, theme,
                       self.scene.bounds_visible, self.scene.orientation_visible)
        self._caption_scene(caption)
        if reframe:
            self.scene.plotter.reset_camera()
        else:
            self.scene.plotter.camera_position = camera
        self.scene.plotter.render()

    def _mac_clicked(self, event):
        """Comparing two sets: a click on the MAC grid picks the pair."""
        if self._compare is None:
            return
        plot = next((item for item in self.mac_view.ci.items
                     if hasattr(item, 'getViewBox')), None)
        if plot is None:
            return
        position = plot.getViewBox().mapSceneToView(event.scenePos())
        down = int(np.floor(position.y()))
        across = int(np.floor(position.x()))
        # a tall grid drew the second set as the rows: map the click
        # back into (first set, second set) space
        row, column = ((across, down) if self._compare.get('tall')
                       else (down, across))
        self._pick_mac_pair(row, column,
                            bool(event.modifiers()
                                 & (Qt.KeyboardModifier.ShiftModifier
                                    | Qt.KeyboardModifier.ControlModifier)))

    def _pick_mac_pair(self, row, column, extend):
        """The selection idiom, on MAC cells — one implementation for
        both readings, so the bars cannot pick differently from the
        grid: a plain click picks one pair, a modifier extends, and a
        modifier on a selected pair removes it. Pairs accumulate into
        the comparison table."""
        if self._compare is None:
            return
        a = self.objects.get(self._compare['pair'][0])
        b = self.objects.get(self._compare['pair'][1])
        if a is None or b is None:
            return
        if not (0 <= row < a.num_shapes and 0 <= column < b.num_shapes):
            return
        cell = (row, column)
        pairs = self._compare.setdefault('pairs',
                                         [self._compare['cell']])
        if extend and cell in pairs:
            pairs.remove(cell)
            if not pairs:
                self._compare = None    # empty: the default pair returns
            else:
                self._compare['cell'] = pairs[-1]
        elif extend:
            pairs.append(cell)
            self._compare['cell'] = cell
        else:
            self._compare['pairs'] = [cell]
            self._compare['cell'] = cell
        playing = self._playing
        self.render_current()
        if playing:      # rebuilding the scene must not stop the show
            self.set_playing(True)

    def _mac_bars_clicked(self, position):
        """A click (not a drag) landed on the 3-D MAC: pick the bar.

        The bars draw rows as the first set directly — no tall
        transpose — so a picked cell maps straight to (first, second)
        and the shared idiom does the rest.
        """
        if self._compare is None:
            return
        from vtkmodules.vtkRenderingCore import vtkCellPicker

        from ..viz.mac_bars import cell_to_pair

        plotter = self._mac_bars_plotter_obj
        picker = vtkCellPicker()
        picker.SetTolerance(0.002)
        picker.Pick(position[0], position[1], 0, plotter.renderer)
        actors = plotter.renderer.actors
        if picker.GetActor() is not actors.get('mac-bars'):
            return      # the floor, a label, an axis: not a bar
        cell = picker.GetCellId()
        if cell < 0 or self._mac_bars_grid is None:
            return
        _rows, columns = self._mac_bars_grid
        row, column = cell_to_pair(cell, columns)
        interactor = plotter.iren.interactor
        extend = bool(interactor.GetShiftKey()
                      or interactor.GetControlKey())
        self._pick_mac_pair(row, column, extend)

    def _arm_mac_bars_picking(self):
        """Click-picking on the bars, told apart from an orbit.

        Every press starts a possible orbit, so picking on press would
        re-pick whenever the user only meant to turn the scene. A press
        remembered and a release within a few pixels of it is a click;
        anything farther is the camera's.
        """
        iren = getattr(self._mac_bars_plotter_obj, 'iren', None)
        if iren is None:
            return      # off-screen: no interactor, nothing to arm
        state = {'press': None}

        def pressed(_obj, _event):
            state['press'] = iren.interactor.GetEventPosition()

        def released(_obj, _event):
            start, state['press'] = state['press'], None
            if start is None:
                return
            end = iren.interactor.GetEventPosition()
            if abs(end[0] - start[0]) <= 3 and abs(end[1] - start[1]) <= 3:
                self._mac_bars_clicked(end)

        iren.add_observer('LeftButtonPressEvent', pressed)
        iren.add_observer('LeftButtonReleaseEvent', released)

    def _arm_matched_rows(self, committed):
        """A row of the matched table is a pair, so picking one picks it.

        The MAC and the table are two views of the same list: clicking a
        square already selects the row, and this is the other direction —
        the square moves, the scene animates that pair, and stepping down
        the table walks the comparison.
        """
        model = self.table.selectionModel()
        if model is None:
            return
        model.selectionChanged.connect(
            lambda *_args, pairs=list(committed.pairs):
            self._matched_row_chosen(pairs))

    def _matched_row_chosen(self, pairs):
        """Show the pair whose row was picked, once Qt has finished.

        Queued for the reason the compliance table is: replacing the
        model from inside its own selectionChanged swaps the selection
        model under Qt while it is mid-way through a keypress, and the
        arrow keys go dead.
        """
        if self._compare is None or self._matched_row_moving:
            return
        rows = {index.row() for index
                in self.table.selectionModel().selectedIndexes()}
        if len(rows) != 1:
            return                      # a scatter of rows is not one pair
        row = next(iter(rows))
        if not 0 <= row < len(pairs):
            return
        cell = tuple(pairs[row])
        if tuple(self._compare.get('cell') or ()) == cell:
            return                      # already showing it
        self._matched_row_moving = True
        QTimer.singleShot(0, lambda: self._show_matched_pair(cell))

    def _show_matched_pair(self, cell):
        if self._compare is None:
            self._matched_row_moving = False
            return
        self._compare['pairs'] = [cell]
        self._compare['cell'] = cell
        playing = self._playing
        try:
            self.render_current()
        finally:
            self._matched_row_moving = False
        if playing:      # stepping through the table must not stop the show
            self.set_playing(True)

    def _deflect_shape(self, geometry, shapes, showing=None):
        shape_name, shape_set, mode = shapes[0]
        chosen = 0 if mode is None else mode   # the set itself: its first mode
        if not shape_set.num_shapes:
            return None
        fit = self._fit_note(geometry, shape_set.coordinate)
        if fit is None:
            issue = self.report.issue_for(shape_name) if self.report else None
            return (f'{shape_name}: {issue.message}' if issue else
                    f'{shape_name}: none of its DOFs are on this geometry')
        self._shape_mode = True
        self._shape_source = (geometry, shape_set)
        self._build_animator(
            geometry, ShapeDeflection(geometry, shape_set.coordinate,
                                      shape_set.shape_matrix[chosen]),
            caption=shape_set.mode_label(chosen), showing=showing)
        self.mode_box.blockSignals(True)
        self.mode_box.setRange(1, max(1, shape_set.num_shapes))
        self.mode_box.setValue(chosen + 1)
        self.mode_box.blockSignals(False)
        self.phase_slider.blockSignals(True)
        self.phase_slider.setValue(0)
        self.phase_slider.blockSignals(False)
        self._show_animation_controls(True, phase=True)
        self.play_action.setEnabled(True)
        note = ('' if mode is not None else
                f' (first of {shape_set.num_shapes}; expand to pick another)')
        return (f'{shape_name}: {shape_set.mode_label(chosen)}{note} '
                f'— press Play{fit}')

    # ---- writing a specification at a set's modal coordinates ------------

    def _author_toggled(self, checked):
        self._author_wanted = bool(checked)
        if not checked:
            self.data_pane.author_panel.hide()
        self.render_current()

    def _authoring(self, shapes, geometries, series, channels):
        """The door a specification sheet can open on, while the
        reading is wanted; None otherwise, standing the reading down.

        ('shapes', key, name, modes) for one shape set — whole, or the
        shapes picked in it, `modes` their indices — ('table', name) for
        one whole channel table, a geometry allowed alongside either;
        ('spec', key, specs) for one or more specifications and
        nothing else, `specs` a list of (name, channels) with the
        channels the picked records touch, or None for all of them
        (Brandon, 2026-09-04: a pick of records, in one specification
        or a few, is an edit of those channels)."""
        from .object_tables import _pair_of

        parts_picked = any(entry['components'] or entry['entities']
                           for entry in geometries.values())
        door = None
        sets = {name for name, _obj, _detail in shapes}
        if len(sets) == 1 and not series and not channels and not parts_picked:
            # one set, whole or as the shapes picked in it: a virtual
            # point's target is its three translations, the rotations
            # left out rather than written as zero (Brandon, 2026-09-06)
            (name,) = sets
            picked = sorted({d for _n, _o, d in shapes if d is not None})
            whole = any(d is None for _n, _o, d in shapes) or not picked
            modes = None if whole else picked
            key = name if modes is None else \
                f'{name}[{",".join(f"M{k + 1}" for k in modes)}]'
            door = ('shapes', key, name, modes)
        elif (series and all(isinstance(obj, Specification)
                             for _n, obj, _r in series)
              and not shapes and not channels and not parts_picked):
            specs = []
            for name, obj, records in series:
                picked = None
                if records is not None:
                    picked = list(dict.fromkeys(
                        dof for k in records for dof in _pair_of(obj, k)))
                specs.append((name, picked))
            key = ' | '.join(f'{name}[{",".join(picked)}]' if picked else name
                             for name, picked in specs)
            door = ('spec', key, specs)
        elif (len(channels) == 1 and channels[0][2] is None and not shapes
              and not series and not parts_picked):
            door = ('table', channels[0][0])
        on_plot = door is not None and door[0] == 'spec'
        self.author_action.setVisible(door is not None and not on_plot)
        self.author_data_action.setVisible(on_plot)
        wanted = door is not None and self._author_wanted
        self.author_action.setChecked(wanted and not on_plot)
        self.author_data_action.setChecked(wanted and on_plot)
        if not wanted:
            self.data_pane.author_panel.hide()
            self._author_door = None
            return None
        self._author_door = door
        return door

    def _draft_for(self, door):
        """The sheet open on `door`, started the way its kind starts."""
        from ..core.author import SpecificationDraft

        kind, key = door[0], door[1]
        draft = self._drafts.get(key)
        if draft is None:
            if kind == 'shapes':
                draft = SpecificationDraft.at_modal_coordinates(
                    self.objects[door[2]], modes=door[3])
            elif kind == 'spec':
                specs = door[2]
                if len(specs) == 1:
                    name, picked = specs[0]
                    draft = SpecificationDraft.from_specification(
                        self.objects[name], picked)
                else:
                    draft = SpecificationDraft.across({
                        name: SpecificationDraft.from_specification(
                            self.objects[name], picked, source=name)
                        for name, picked in specs})
            else:
                draft = SpecificationDraft.at_control_channels(
                    self.objects[key])
            self._drafts[key] = draft
            self._draft_names[key] = tuple(self._door_names(door))
        return draft

    def _forget_drafts_on(self, names, keep):
        """Drop every cached draft opened on any of `names` but the one
        under `keep` — the door that just wrote the object."""
        touched = set(names)
        for key, held in list(self._draft_names.items()):
            if key != keep and touched & set(held):
                self._drafts.pop(key, None)
                self._draft_names.pop(key, None)

    @staticmethod
    def _door_names(door):
        """The objects a door is open on, for a message."""
        if door[0] == 'spec':
            return [name for name, _picked in door[2]]
        return [door[2]] if door[0] == 'shapes' else [door[1]]

    def _render_author(self, door):
        """The draft's autospectra and bands on the plot, the draft
        itself beside it."""
        from ..core.author import band_notes

        kind = door[0]
        names = self._door_names(door)
        said_names = ', '.join(names)
        panel = self.data_pane.author_panel
        try:
            draft = self._draft_for(door)
        except ValueError as refusal:
            panel.hide()
            self.data_pane.clear()
            return f'{said_names}: {refusal}'
        if kind == 'spec':
            picked = [f'{len(chs)} of {name}' for name, chs in door[2] if chs]
            origin = (f'opened from {said_names}'
                      + (f' — {", ".join(picked)} channels picked'
                         if picked else ''))
        else:
            origin = (f'at the control channels of {said_names}'
                      if kind == 'table' else
                      f'at the modal coordinates of {said_names}'
                      if door[3] is None else
                      f'at {", ".join(draft.channels)} of {said_names}')
        if kind != 'spec':
            # there has to be an object for the sheet to edit, so the
            # starter is made at once beside the set or the table and
            # the sheet carries on there (Brandon, 2026-09-06: every
            # edit lands on the selected object). Deferred a tick: the
            # selection change re-renders, and this is a render.
            # Remembered under the source's name as well as the door's
            # key: adding the object rebuilds the tree and drops a pick
            # of shapes, so the render in between arrives on the whole
            # set's door — and made a second object from it (2026-09-06)
            added = (self._author_made.get(door[1])
                     or self._author_made.get(names[0]))
            if added not in self.objects:
                try:
                    added = self.project.author_specification(names[0], draft)
                except ValueError as refusal:
                    panel.hide()
                    self.data_pane.clear()
                    return f'{said_names}: {refusal}'
                self._author_made[door[1]] = added
                self._author_made[names[0]] = added
                self._drafts.pop(door[1], None)
                self._draft_names.pop(door[1], None)
                self._drafts[added] = draft
                self._draft_names[added] = (added,)
            QTimer.singleShot(0, lambda: self.show_object(added))
            return (f'{added}: made at {origin.split(" of ")[0]}'
                    f' of {said_names} — the sheet edits it from here')
        # bands the object does not have yet are said here, from the
        # object: after the first edit writes them the saying stops
        missing = [note for name in names
                   for note in band_notes(self.objects[name])]
        if missing:
            origin += ' — ' + '; '.join(missing)
        panel.show_draft(draft, self.unit_system, origin,
                         spacing_hint=self._spacing_hint(names))
        panel.show()
        if len(set(draft.sources)) > 1:
            previews = [(f'{name} (draft)', draft.for_source(name).preview(),
                         None) for name in names]
        else:
            previews = [(f'{names[0]} (draft)', draft.preview(), None)]
        status = self._render_series(previews)
        self._add_band_handles(draft)
        unset = draft.unset_pairs()
        said = ('every cross term stated' if not unset else
                f'{len(unset)} pair{"s" * (len(unset) != 1)} unstated')
        # which channels the sheet is on, said outright: a witness for
        # an edit that seems to land on fewer channels than were picked
        # (Brandon, 2026-09-06 — not reproduced headless in either view)
        on = (f'{draft.num_channels} channel{"s" * (draft.num_channels != 1)}'
              f' ({", ".join(draft.channels[:4])}'
              f'{", …" if draft.num_channels > 4 else ""})')
        return f'{said_names}: specification sheet on {on}, {said}' + (
            f' | {status}' if status else '')

    def _spacing_hint(self, names):
        """A frequency spacing for the sheet's interpolated form when
        the sheet remembers none: the averaging of a time history
        linked with the object the sheet is open on, else of any time
        history in the project — sample rate over frame length, the
        lines a PSD of it would have. None when there is no history;
        the sheet then asks rather than guesses."""
        linked = []
        for name in names:
            linked.extend(self.project.group_of(name) or [])
        candidates = [n for n in linked if n in self.objects] + [
            n for n in self.objects if n not in linked]
        for name in candidates:
            obj = self.objects[name]
            averaging = getattr(obj, 'averaging', None)
            if isinstance(obj, TimeHistory) and averaging is not None:
                return float(obj.sample_rate) / int(averaging.frame_length)
        return None

    def _add_band_handles(self, draft):
        """The drawn channel's bands as drag handles on the flat plot,
        one per run of segments sharing a band — the place the bands
        are set (Brandon, 2026-09-06). The stage draws them but has no
        drag; the flat plot is the editor."""
        import pyqtgraph as pg

        from ..plot.bands import add_band_handles

        self._band_handles = []
        if self.data_pane.showing_waterfall or self._plotted_pair is None:
            return
        channel = self._plotted_pair[0]
        if channel not in draft.channels:
            return
        plot = next((item for item in self.data_pane.graphics.ci.items
                     if isinstance(item, pg.PlotItem)), None)
        if plot is None:
            return
        self._band_handles = add_band_handles(
            plot, draft, draft.channels.index(channel), self.unit_system,
            resolve_theme(self.theme_name), self._band_dragged)

    def _band_dragged(self, kind, side, segments, decibels):
        """A band's edge let go on the plot at `decibels`: every
        channel the sheet holds lands there, in the segments under the
        drag, under each channel's own constraints."""
        door = self._author_door
        if door is None:
            return
        try:
            draft = self._draft_for(door).with_band_edge(kind, side, segments,
                                                         decibels)
            self._author_edited(draft)
        except Exception as exc:  # noqa: BLE001 - a slot: said and logged, never swallowed
            self._sheet_failed(f'{kind} band drag', exc)
            return
        self._show_status(
            f'{kind} band {"above" if side == "upper" else "below"} set to '
            f'{decibels:+.0f} dB on {draft.num_channels} channel'
            f'{"s" * (draft.num_channels != 1)}')

    #: where a failure inside the sheet's slots is written in full — a
    #: Qt slot that raises is swallowed with a line on stderr nobody
    #: running the app sees; the status line carries the last line and
    #: this file the traceback (Brandon, 2026-09-06: Breakpoints spun
    #: and left the old rows standing, and nothing said why)
    SHEET_LOG: ClassVar[Path] = (Path.home() / 'Library' / 'Logs'
                                 / 'Visual Dynamics' / 'sheet.log')

    def _sheet_failed(self, where, exc):
        """Say that a sheet gesture failed, after the render that
        would otherwise overwrite the saying, and keep the traceback."""
        import traceback

        try:
            self.SHEET_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(self.SHEET_LOG, 'a', encoding='utf-8') as log:
                log.write(f'--- {where}\n{traceback.format_exc()}\n')
        except OSError:
            pass
        self.render_current()
        self._show_status(f'{where} failed: {exc} (traceback in '
                          f'{self.SHEET_LOG})')

    def _author_constraint_asked(self, which, on):
        """Symmetric or Uniform, for every channel of the whole
        specification the sheet is open on — as the form is (Brandon,
        2026-09-06) — then the sheet reopened on the same picks."""
        from ..core.author import SpecificationDraft

        door = self._author_door
        if door is None or door[0] != 'spec':
            return
        try:
            for name, _picked in door[2]:
                whole = SpecificationDraft.from_specification(
                    self.objects[name]).constrained(**{which: on})
                self.project.author_specification(name, whole, replace=True)
            remembered = self._draft_for(door).spacing
            self._drafts.pop(door[1], None)
            draft = self._draft_for(door)
            if draft.spacing is None and remembered is not None:
                draft = draft._copy(spacing=remembered)
                self._drafts[door[1]] = draft
            self._author_edited(draft)
        except Exception as exc:  # noqa: BLE001 - a slot: said and logged, never swallowed
            self._sheet_failed(f'{which} {"on" if on else "off"}', exc)
            return
        self._show_status(f'{which} {"on" if on else "off"} for every '
                          f'channel of {", ".join(n for n, _p in door[2])}')

    def _author_form_asked(self, form, spacing):
        """Breakpoints or Interpolated, for the whole specification the
        sheet is open on — not the channels it happens to hold
        (Brandon, 2026-09-06) — then the sheet reopened on the same
        picks from the converted object."""
        from ..core.author import SpecificationDraft

        door = self._author_door
        if door is None or door[0] != 'spec':
            return
        said = []
        try:
            for name, _picked in door[2]:
                whole = SpecificationDraft.from_specification(self.objects[name])
                whole = (whole.to_breakpoints() if form == 'breakpoints'
                         else whole.to_lines(spacing))
                self.project.author_specification(name, whole, replace=True)
                said.append(f'{name}: {whole.notes[-1]}' if whole.notes
                            else f'{name}: {len(whole.frequencies)} '
                                 f'{"breakpoints" if form == "breakpoints" else "lines"}')
            remembered = self._draft_for(door).spacing
            self._drafts.pop(door[1], None)
            draft = self._draft_for(door)
            if draft.spacing is None and remembered is not None:
                # the breakpoints form has no spacing of its own; the
                # sheet keeps the one the lines had, for the way back
                draft = draft._copy(spacing=remembered)
                self._drafts[door[1]] = draft
            self._author_edited(draft)
        except Exception as exc:  # noqa: BLE001 - a slot: said and logged, never swallowed
            # said *after* the render: said before it, the render's own
            # status line overwrote the refusal and the sheet looked
            # merely stale
            self._sheet_failed(form, exc)
            return
        self._show_status(' | '.join(said))

    def _author_edited(self, draft):
        """An edit on the sheet, landed on the specification at once —
        the verb a script calls, with `replace`. The sheet's draft
        stays the source of truth for the sheet (its form and spacing
        are the sheet's, not the object's); the object is what the
        draft says."""
        from .object_tables import _pair_of

        door = self._author_door
        if door is None:
            return
        self._drafts[door[1]] = draft
        names = self._door_names(door)
        self._draft_names[door[1]] = tuple(names)
        # the objects as they were, to carry each grid's picks across
        # the rewrite by pair rather than by index
        self._edited_before = {name: self.objects[name] for name in names
                               if name in self.objects}
        try:
            self.project.author_specification(
                names if len(names) > 1 else names[0], draft, replace=True)
        except ValueError as refusal:
            self._show_status(f'{", ".join(names)}: {refusal}')
            return
        self._forget_drafts_on(names, keep=door[1])
        for name in names:
            item = self._item_for_object(name)
            if item is None:
                continue
            obj = self.objects[name]
            item.setToolTip(0, self._object_tooltip(name, obj))
            # the record picks are the door — a sheet on two picked
            # channels must stay on them — and rebuilding the children
            # for the rewritten object would drop them, widening the
            # sheet to every channel under the user's hands
            grid = self.record_grids.get(name)
            picked = ([_pair_of(self._edited_before[name], i)
                       for i in grid.selected_records()]
                      if grid is not None and name in self._edited_before
                      else [])
            self._build_children(item, obj, name)
            grid = self.record_grids.get(name)
            if picked and grid is not None:
                pairs = {_pair_of(obj, i): i for i in range(obj.num_records)}
                grid.select_records([pairs[p] for p in picked if p in pairs])
        self._report_content_changed()
        self.render_current()

    # ---- the rigid-body reading: six shapes previewed before they exist --

    def _rigid_properties(self, geometry):
        return (geometry.mass_properties
                or geometry.suggest_mass_properties())

    def _describe_rigid(self, geometry, properties):
        """The settings in the display units — the status line's
        wording, one implementation with the journal's SI one."""
        return properties.describe(self.unit_system,
                                   defined=geometry.units_defined)

    def _deflect_rigid(self, name, geometry, showing=None):
        """Animate the six rigid-body shapes the pane describes, on the
        geometry they would be made from — the preview half of
        select, see, set, apply. The set is rebuilt on every edit
        (six shapes over 3N DOFs, microseconds) and never stored:
        Generate Rigid Body Mode Shapes is what makes it."""
        from ..core.rigid import rigid_body_shapes

        if not geometry.num_nodes:
            return f'{name}: no nodes to move'
        properties = self._rigid_properties(geometry)
        panel = self.scene.rigid_panel
        panel.show_geometry(geometry, properties, self.unit_system)
        panel.show()
        shapes = rigid_body_shapes(geometry, properties)
        self._rigid_preview = (name, shapes)
        self._shape_mode = True
        self._shape_source = (geometry, shapes)
        self._build_animator(
            geometry, ShapeDeflection(geometry, shapes.coordinate,
                                      shapes.shape_matrix[0]),
            caption=self._shape_caption(shapes, 0), showing=showing)
        self.mode_box.blockSignals(True)
        self.mode_box.setRange(1, shapes.num_shapes)
        self.mode_box.setValue(1)
        self.mode_box.blockSignals(False)
        self.phase_slider.blockSignals(True)
        self.phase_slider.setValue(0)
        self.phase_slider.blockSignals(False)
        self._show_animation_controls(True, phase=True)
        self.play_action.setEnabled(True)
        return (f'{name}: rigid body modes '
                f'{self._describe_rigid(geometry, properties)} — press '
                'Play; Generate Rigid Body Mode Shapes makes the set')

    def _decorate_rigid(self):
        """The reference point on the scene, while the reading is up.

        Called by every animator build, so stepping to another mode —
        which clears the scene — puts the marker back where the pane
        says it is."""
        if self._rigid_preview is None or self.animator is None:
            return
        from ..viz.rigid import add_reference_point

        name, _shapes = self._rigid_preview
        geometry = self.objects.get(name)
        if geometry is None:
            return
        point = np.asarray(self._rigid_properties(geometry).point)
        if geometry.units_defined:
            point = self.unit_system.from_si(point, 'length')
        add_reference_point(self.scene.plotter, point,
                            self.animator.model_size,
                            resolve_theme(self.theme_name)['scene_highlight'])

    def _rigid_edited(self, properties):
        """The panel moved: store it, restate the preview, and say so.

        Storing is what makes the badge work — a generated set's
        provenance fingerprints the geometry's `mass_properties`, so
        the edit is the moment its refresh badge appears."""
        from ..core.rigid import rigid_body_shapes

        if self._rigid_preview is None:
            return
        name, _old = self._rigid_preview
        geometry = self.objects.get(name)
        if geometry is None:
            return
        geometry.mass_properties = properties
        self.project.record_setting(geometry, 'mass_properties', properties)
        self._refresh_stale_badges()
        self._report_content_changed(settling=True)
        shapes = rigid_body_shapes(geometry, properties)
        self._rigid_preview = (name, shapes)
        self._shape_source = (geometry, shapes)
        # the same mode, the same phase, the camera kept
        self._mode_changed(self.mode_box.value())
        self._show_status(f'{name}: rigid body modes '
                          f'{self._describe_rigid(geometry, properties)}')

    def _deflect_series(self, geometry, series, showing=None):
        name, data, records = series[0]
        if len(series) > 1:
            return None          # one object on the geometry at a time
        # a PSD is checked first: a CPSD's ordinate is complex, but its
        # auto rows are phaseless and must not fall into the ODS path
        # wearing a phase they do not have. A *picked reference column*
        # is the exception — magnitude and phase relative to that
        # reference is the operating deflection shape from operating
        # data, the same shape the FRF ODS shows.
        if isinstance(data, Psd):
            column = self._cpsd_column(data, records)
            if column is not None:
                return self._deflect_ods(geometry, name, data, column,
                                         showing=showing)
            if records is None:
                # a whole CPSD carries phase and answers with its
                # principal shape; only an autos-only set falls through
                # to the envelope
                principal = self._deflect_principal(geometry, name, data,
                                                    showing=showing)
                if principal is not None:
                    return principal
            return self._deflect_envelope(geometry, name, data, records,
                                          showing=showing)
        if (data.abscissa_dim == 'frequency'
                and np.iscomplexobj(data.ordinate)):
            return self._deflect_ods(geometry, name, data, records,
                                     showing=showing)
        if data.abscissa_dim != 'time':
            return None
        indices, notes = animation_records(data, records)
        dofs = [data.response_dof[i] for i in indices]
        fit = self._fit_note(geometry, dofs)
        if fit is None:
            issue = self.report.issue_for(name) if self.report else None
            return (f'{name}: {issue.message}' if issue else
                    f'{name}: none of its DOFs are on this geometry')
        ordinate = data.ordinate[list(indices)]
        deflection = TimeDeflection(geometry, dofs, ordinate)
        if not len(deflection.rows):
            return None
        self._shape_mode = False
        self._build_animator(geometry, deflection, showing=showing)
        self._add_dof_arrows([geometry])
        self._show_animation_controls(True)
        self.play_action.setEnabled(True)
        units = ', '.join(sorted({data.ordinate_unit[i] or 'unknown units'
                                  for i in indices}))
        return (f'{name} — deflection proportional to {units}; '
                f'drag the cursor or press Play{fit}{notes}')

    def _cpsd_column(self, data, records):
        """The picked records as one CPSD reference column, or None.

        A column against a single reference channel carries each
        response's magnitude and phase *relative to that reference* —
        an honest complex shape. Only an explicit pick reads this way:
        a whole CPSD stays an envelope, because no reference was
        chosen, and the diagonal is the autos. Responses of other
        quantities than the column's commonest are left to the
        envelope's rule and dropped here the same way.
        """
        if records is None or len(records) < 2:
            return None
        if (data.reference_dof is None
                or not np.iscomplexobj(data.ordinate)):
            return None
        quantities = [channel_quantities(data.known_dim(i))
                      for i in records]
        references = {(data.reference_dof[i], quantities[k][1])
                      for k, i in enumerate(records)}
        if len(references) != 1:
            return None
        if all(data.response_dof[i] == data.reference_dof[i]
               for i in records):
            return None
        kinds = [response for response, _reference in quantities]
        wanted = max(set(kinds), key=kinds.count)
        return [i for i, kind in zip(records, kinds) if kind == wanted]

    def _deflect_envelope(self, geometry, name, data, records,
                          showing=None):
        """A PSD's envelope: two copies of the geometry deflected
        ±sqrt(PSD) at the cursor line, colored dB below the loudest
        node at any line.

        No phase is claimed — the pair shows every extreme every DOF
        reaches and nothing about when. Play walks the cursor up the
        spectrum, the same reading a time history's Play gives its
        samples.
        """
        from ..deform import envelope_records

        # which records the envelope may show is `envelope_records` —
        # one rule with the headless call in viz.animate. The combo's
        # choice may be stale for this object (it rides the toolbar
        # across selections), so a choice no record answers falls back
        # to the commonest kind rather than an empty scene.
        quantity = self.dofs_combo.currentData()
        of_kind, quantity, crossed, left_out = envelope_records(
            data, records, quantity)
        if not of_kind and quantity is not None:
            of_kind, quantity, crossed, left_out = envelope_records(
                data, records, None)
        self._cursor_dimension = quantity   # the cursor rides this plot
        chosen, notes = animation_records(data, of_kind)
        if crossed:
            notes += f' — {crossed} cross records not shown'
        if left_out:
            notes += (f' — {left_out} records of other quantities not '
                      'shown (the quantity box picks)')
        dofs = [data.response_dof[i] for i in chosen]
        fit = self._fit_note(geometry, dofs) if chosen else None
        if fit is None:
            issue = self.report.issue_for(name) if self.report else None
            return (f'{name}: {issue.message}' if issue else
                    f'{name}: none of its DOFs are on this geometry')
        deflection = EnvelopeDeflection(
            geometry, dofs, np.asarray(data.ordinate)[chosen].real)
        if not len(deflection.rows):
            return None
        line = deflection.strongest_line
        frequency = float(np.asarray(
            data.display_abscissa(self.unit_system))[line])
        self._shape_mode = False      # Play walks the cursor, like time
        self._build_animator(geometry, deflection,
                             caption=f'Envelope — {frequency:.5g} Hz',
                             showing=showing, mirrored=True)
        self.animator.set_parameter(line)
        self._add_dof_arrows([geometry])
        self._show_animation_controls(True)
        self.play_action.setEnabled(True)
        return (f'{name} — the {quantity} envelope, both extremes of '
                f'every channel at the cursor frequency, color dB '
                f'below the loudest; drag the cursor or press '
                f'Play{fit}{notes}')

    def _deflect_ods(self, geometry, name, data, records, showing=None):
        """Complex spectra deflect the geometry at one frequency line —
        the operating deflection shape, swept in phase like a complex
        mode. The plot cursor picks the line; Play sweeps the phase."""
        indices, notes = animation_records(data, records)
        dofs = [data.response_dof[i] for i in indices]
        units = ', '.join(sorted(
            {(f'{data.ordinate_unit[i]}/{data.reference_unit[i]}'
              if data.reference_unit[i] else data.ordinate_unit[i])
             if data.ordinate_unit[i] else 'unknown units'
             for i in indices}))
        return self._arm_ods(
            geometry, name, data, dofs,
            np.asarray(data.ordinate)[list(indices)],
            f'operating deflection proportional to {units}',
            notes, showing)

    def _deflect_principal(self, geometry, name, data, showing=None):
        """The whole CPSD's reading: the principal operating deflection
        shape — the dominant eigenvector of the cross-spectral matrix
        per line, each channel's phase relative to the others, no
        reference to choose. A grid-picked column still reads against
        one channel; this is the matrix's own answer."""
        try:
            dofs, shapes, used = data.principal_shapes(
                self.dofs_combo.currentData())
        except ValueError:
            return None
        self._cursor_dimension = used
        return self._arm_ods(
            geometry, name, data, dofs, shapes,
            f'the principal {used} operating deflection — the '
            'cross-spectral matrix’s dominant shape (a grid column '
            'reads against one reference instead)', '', showing)

    def _arm_ods(self, geometry, name, data, dofs, ordinate, flavor,
                 notes, showing):
        fit = self._fit_note(geometry, dofs)
        if fit is None:
            issue = self.report.issue_for(name) if self.report else None
            return (f'{name}: {issue.message}' if issue else
                    f'{name}: none of its DOFs are on this geometry')
        deflection = OdsDeflection(geometry, dofs, ordinate)
        if not len(deflection.rows):
            return None
        # a cursor born on line 0 would deflect the flattest corner of
        # the spectrum; the strongest line is where the shape is
        deflection.line = deflection.strongest_line
        frequency = float(np.asarray(
            data.display_abscissa(self.unit_system))[deflection.line])
        self._shape_mode = True      # Play sweeps phase, the line holds
        self._build_animator(geometry, deflection,
                             caption=f'ODS — {frequency:.5g} Hz',
                             showing=showing)
        self._add_dof_arrows([geometry])
        self.phase_slider.blockSignals(True)
        self.phase_slider.setValue(0)
        self.phase_slider.blockSignals(False)
        self._show_animation_controls(True, phase=True)
        self.play_action.setEnabled(True)
        return (f'{name} — {flavor}; drag the cursor to pick the '
                f'frequency, Play sweeps the phase{fit}{notes}')

    def _add_dof_arrows(self, geometries):
        """The DOF arrows, on whichever scene is being built.

        There are two, and only one of them ever drew these. A geometry
        on its own goes through `_render_geometries`; a geometry with a
        time history on it goes through the deflection path instead and
        builds its scene through the animator — so turning the arrows
        on beside a time history, which is the one selection that makes
        the button appear at all, did nothing.

        Drawn after the animator, because it builds the scene it is
        going to animate and anything added before is not in it.
        """
        if not self.dofs_action.isChecked():
            return
        from ..core.report import (
            EXCITATION_QUANTITIES,
            series_quantity_dofs,
        )
        from ..viz.geometry import add_dof_arrows

        quantity = self.dofs_combo.currentData() if self._dof_series else None
        for geometry in geometries:
            if quantity:
                dofs = series_quantity_dofs(self._dof_series, quantity)
            else:
                # nothing measured to ask about, so every node's own
                # three translations. Written as plain DOF strings, so
                # a node whose displacement coordinate system is turned
                # gets its arrows turned with it — X+ means that node's
                # X, not the global one.
                dofs = [f'{node}{axis}' for node in geometry.node_id
                        for axis in ('X+', 'Y+', 'Z+')]
            if not dofs:
                continue
            add_dof_arrows(self.scene.plotter, geometry, dofs,
                           unit_system=self.unit_system,
                           incoming=quantity in EXCITATION_QUANTITIES)

    def _shape_summary(self, shapes, geometries=None):
        """Mode shapes have no view of their own; say what is selected, and
        why nothing is moving when there is no geometry to move."""
        _name, shape_set, mode = shapes[0]
        if mode is not None:
            what = shape_set.mode_label(mode)
        elif len(shapes) > 1:
            what = f'{len(shapes)} shape sets'
        else:
            what = self._status_for(shape_set)
        if geometries:
            return what
        return f'{what} — select a geometry to see it deflected'

    def _render_geometries(self, geometries, ignored=0):
        if self.scene.plotter is None:
            return ''  # not on screen yet; showEvent will build and render
        colors = resolve_theme(self.theme_name)
        # framing is for new content: redrawing what is already on screen
        # must not throw away a view the user set up
        showing = tuple(sorted(name for name, _ in geometries))
        reframe = showing != self._framed
        self._framed = showing
        camera = self.scene.plotter.camera_position
        self.scene.plotter.clear()
        self.scene.plotter.set_background(colors['scene_background'],
                                    top=colors['scene_background_top'])
        axis_unit = '[units undefined]'
        for i, (_name, entry) in enumerate(geometries):
            geometry = entry['object']
            components = entry['components'] or None
            entities = entry['entities']
            # one color per geometry so overlaid ones stay tellable apart
            override = curve_color(i) if len(geometries) > 1 else None
            if components or entities:
                axis_unit = self._add_with_context(
                    geometry, components, entities, colors)
            else:
                axis_unit = add_geometry(
                    self.scene.plotter, geometry, unit_system=self.unit_system,
                    color_override=override, text_color=colors['scene_text'])
        self._add_dof_arrows(entry['object'] for _name, entry in geometries)
        self.scene.axis_unit = axis_unit
        annotate_scene(self.scene.plotter, axis_unit, colors, self.scene.bounds_visible,
                       self.scene.orientation_visible)
        if reframe:
            self.scene.plotter.reset_camera()
        else:
            # explicit: clearing a scene lets the next mesh reset the camera
            self.scene.plotter.camera_position = camera
        self.scene.plotter.render()
        return self._geometry_status(geometries, ignored)

    def _add_with_context(self, geometry, components, entities, colors):
        """Draw the whole geometry muted, with the selection standing out.

        Used for any partial pick — a category or an individual entity — so
        the selection always keeps the rest of the model as context.
        """
        wanted = _label_kinds(geometry, components, entities)
        components, entities = _drawable(geometry, components, entities)
        axis_unit = add_geometry(
            self.scene.plotter, geometry, unit_system=self.unit_system,
            color_override=colors['scene_muted'], node_size=5.0,
            line_width=1.0, opacity=0.35, text_color=colors['scene_text'])
        if components:
            axis_unit = add_geometry(
                self.scene.plotter, geometry, unit_system=self.unit_system,
                components=components, labels=wanted,
                color_override=colors['scene_highlight'], node_size=11.0,
                line_width=4.0, text_color=colors['scene_text'])
        if entities:
            axis_unit = add_geometry(
                self.scene.plotter, geometry, unit_system=self.unit_system,
                entities=entities, labels=wanted,
                color_override=colors['scene_highlight'],
                node_size=16.0, line_width=5.0,
                text_color=colors['scene_text'])
        return axis_unit

    def _geometry_status(self, geometries, ignored):
        if len(geometries) == 1:
            _name, entry = geometries[0]
            geometry, entities = entry['object'], entry['entities']
            if entities:
                message = (self._entity_summary(geometry, entities)
                           + _label_note(geometry, None, entities))
            elif entry['components']:
                shown = ', '.join(sorted(c.replace('_', ' ')
                                         for c in entry['components']))
                empty = self._empty_category(dict(geometries))
                message = (f'No {shown} in this geometry — Edit to add some'
                           if empty else f'Highlighting {shown}'
                           + _label_note(geometry, entry['components'], None))
            else:
                message = self._status_for(geometry)
        else:
            undefined = sum(1 for _, e in geometries
                            if not e['object'].units_defined)
            message = f'{len(geometries)} geometries overlaid'
            if undefined:
                message += f' ({undefined} with no units defined)'
        if ignored:
            message += f'; {ignored} non-geometry selection(s) not shown'
        return message

    def _entity_summary(self, geometry, entities):
        """Describe the picked entities, with coordinates for a single node."""
        nodes = entities.get('nodes') or []
        if len(nodes) == 1 and not any(
                v for k, v in entities.items() if k != 'nodes'):
            row = geometry.node_index([nodes[0]])[0]
            xyz = geometry.node_xyz[row]
            if geometry.units_defined:
                xyz = self.unit_system.from_si(xyz, 'length')
                unit = self.unit_system.label_text('length')
            else:
                unit = 'units undefined'
            return (f'Node {nodes[0]} at '
                    f'({xyz[0]:.4g}, {xyz[1]:.4g}, {xyz[2]:.4g}) {unit}')
        parts = [f'{len(values)} {ENTITY_LABELS[kind]}'
                 f'{"s" * (len(values) != 1)}'
                 for kind, values in entities.items() if values]
        return 'Highlighting ' + ', '.join(parts)

    # ---- the averaging: the frames a PSD would be built from ------------

    def _render_pair_stage(self, series):
        """Two densities on the stage — overlaid, or divided in dB.

        The louder is the numerator and wears the level coloring;
        the quieter stands back in gray, exactly as a compared
        reference does on the banded stage. Returns None when the
        pair cannot pair (no shared channels, or for the ratio no
        shared lines); the caller falls back to the flat readings
        with the reason on the status line.
        """
        (name_a, a, records_a), (name_b, b, records_b) = series
        louder_first = float(np.nanmean(np.real(np.asarray(
            a.ordinate)))) >= float(np.nanmean(np.real(np.asarray(
                b.ordinate))))
        loud, quiet = ((name_a, a, records_a), (name_b, b, records_b)) \
            if louder_first else ((name_b, b, records_b),
                                  (name_a, a, records_a))
        mode = self.data_pane.spectra_view
        info = self._draw_pair_stage(loud, quiet, mode)
        if info is None:
            return None
        told = ('divided, in decibels' if mode == 'ratio'
                else f'overlaid, {quiet[0]} in gray')
        return (f'{loud[0]} over {quiet[0]}: {info["drawn"]} shared '
                f'channel{"s" * (info["drawn"] != 1)}, {told}')

    def _draw_pair_stage(self, loud, quiet, mode):
        """Two objects paired on one stage. Returns the stage's own
        account, or None when the pair cannot pair — the status line
        already says why.

        Shared by the density pair and the resynthesis overlay, which
        are the same picture of two different things: objects holding
        the same channels, one drawn level-colored and the other stood
        back. Two copies of this would be two stages that could drift
        apart (PRINCIPLES.md, 9).
        """
        from ..viz.paired import add_paired_stage, place_camera

        pane = self.data_pane
        pane.show_waterfall(True)
        pane.graphics.clear()
        plotter = pane.waterfall_plotter
        plotter.clear()
        colors = resolve_theme(self.theme_name)
        plotter.set_background(colors['scene_background'],
                               top=colors['scene_background_top'])
        # one floor, one quantity, exactly as the single-object stage:
        # the box offers the quantities *both* objects hold, and the
        # readings honor the choice — the first cut showed only the
        # largest group in overlay while the ratio divided everything,
        # a silent cap and a disagreement in one (Brandon, 2026-08-25)
        from ..viz.waterfall import quantity_label, waterfall_groups

        shared = [key for key, _members in waterfall_groups(loud[1],
                                                            loud[2])
                  if any(key == other
                         for other, _m in waterfall_groups(quiet[1],
                                                           quiet[2]))]
        entries = []
        if len(shared) > 1:
            words = [quantity_label(key) for key in shared]
            for k, key in enumerate(shared):
                word = words[k]
                if words.count(word) > 1:
                    word = f'{word} ({key[0]})'
                entries.append((word, key))
        pane.show_quantities(entries)
        chosen = pane.chosen_quantity() if entries else None
        try:
            info = add_paired_stage(
                plotter, loud[1], quiet[1], loud_records=loud[2],
                quiet_records=quiet[2], mode=mode, quantity=chosen,
                unit_system=self.unit_system, theme=self.theme_name)
        except ValueError as refusal:
            self._show_status(f'No {mode}: {refusal}')
            pane.show_waterfall(False)
            return None
        if not info.get('drawn'):
            self._show_status(f'{loud[0]} and {quiet[0]} share no '
                              'channel; nothing to pair')
            pane.show_waterfall(False)
            return None
        pair_key = f'{loud[0]}|{quiet[0]}|{mode}'
        if self._waterfall_object != pair_key:
            place_camera(plotter)
            self._waterfall_object = pair_key
        plotter.render()
        return info

    def _render_ratio(self, series):
        """Two densities divided, in decibels — the Ratio reading.

        The louder is the numerator, the same physical rule the sysid
        naming and the import question use: for a noise floor the
        ratio then reads *up* as signal-to-noise. Returns None when
        the pair cannot divide (mismatched lines, no shared channel);
        the caller falls back to the overlay with the reason on the
        status line, because a blank pane explains nothing.
        """
        from ..core.data import density_ratio
        from ..plot import build_ratio

        (name_a, a, records_a), (name_b, b, records_b) = series
        louder_first = float(np.nanmean(np.real(np.asarray(
            a.ordinate)))) >= float(np.nanmean(np.real(np.asarray(
                b.ordinate))))
        loud, quiet = ((name_a, a, records_a), (name_b, b, records_b)) \
            if louder_first else ((name_b, b, records_b),
                                  (name_a, a, records_a))
        try:
            _x, rows, _dofs, _dims = density_ratio(loud[1], quiet[1])
        except ValueError as refusal:
            self._show_status(f'No ratio: {refusal}')
            return None
        pane = self.data_pane
        pane.show_waterfall(False)
        pane.graphics.clear()
        drawn = build_ratio(pane.graphics, loud[1], quiet[1],
                            records=loud[2], theme=self.theme_name)
        finite = np.real(rows)[np.isfinite(np.real(rows))]
        reading = (f'median {10 * np.log10(np.median(finite)):+.1f} dB'
                   if finite.size and np.median(finite) > 0
                   else 'no finite lines')
        return (f'{loud[0]} over {quiet[0]}: {drawn} '
                f'channel{"s" * (drawn != 1)}, {reading} — equal at 0 dB')

    def _sole_history(self, series):
        """The one time history being looked at, if that is what this is.

        The averaging describes a span along one record; two histories
        side by side have two, and there is no one answer to show.
        """
        if len(series) != 1:
            return None
        data = series[0][1]
        return data if isinstance(data, TimeHistory) else None

    def _draw_averaging(self, history):
        """Put the frames on the plot and the parameters beside it.

        One overlay per plot: a history holding force and acceleration
        is drawn as a row each, and the shading belongs on both.

        Draggable even when the file already cut the record into one
        frame per average. What the controller wrote is what it averaged
        and it is the right default, but it is not the only reading of
        the record: a burst-random capture is half excitation and half
        ringdown, and analyzing the burst alone is an ordinary thing to
        want.
        """
        import pyqtgraph as pg

        from ..plot.averaging import AveragingOverlay

        self._clear_averaging()
        try:
            rate = history.sample_rate
        except ValueError as refusal:
            # unevenly sampled, so there are no frames to speak of
            self._show_status(f'No averaging: {refusal}')
            self.data_pane.averaging_panel.hide()
            return
        samples = len(history.abscissa)
        averaging = history.averaging or Averaging.for_records(samples)
        colors = resolve_theme(self.theme_name)
        for item in self.data_pane.graphics.ci.items:
            if not isinstance(item, pg.PlotItem):
                continue
            overlay = AveragingOverlay(item, averaging, rate, samples, colors)
            overlay.changed.connect(self._averaging_dragged)
            self.averaging_overlays.append(overlay)
        panel = self.data_pane.averaging_panel
        panel.show_history(history, averaging)
        panel.show()

    def _clear_averaging(self):
        """Take the marks off. Called before every drawing, so a plot
        cannot come back with the last selection's frames still on it."""
        self._clear_overlays('averaging_overlays')

    def _draw_filtering(self, history):
        """Preview the filter over the raw trace, with its edges
        and order beside the plot.

        The same shape as the averaging view, and for the same reason:
        the parameters live on the history, the view shows and edits
        them, and Apply Filter uses whatever is there when it runs.
        """
        import pyqtgraph as pg

        from ..plot.filtering import FilterOverlay

        self._clear_filtering()
        try:
            rate = history.sample_rate
        except ValueError as refusal:
            # unevenly sampled, so there is nothing to filter through
            self._show_status(f'No filter: {refusal}')
            self.data_pane.filter_panel.hide()
            return
        filtering = history.filtering or history.suggest_filtering()
        colors = resolve_theme(self.theme_name)
        for item in self.data_pane.graphics.ci.items:
            if not isinstance(item, pg.PlotItem):
                continue
            self.filter_overlays.append(
                FilterOverlay(item, filtering, rate, colors))
        panel = self.data_pane.filter_panel
        panel.show_history(history, filtering)
        panel.show()

    def _clear_filtering(self):
        """Take the previews off, before every drawing, like the
        frames and the windows."""
        self._clear_overlays('filter_overlays')

    def _octave_source(self, series):
        """The one plain density the octave reading would band — or
        None, which is also what hides the button."""
        if len(series) != 1:
            return None
        _name, data, _records = series[0]
        if isinstance(data, Psd) and not isinstance(data, Specification):
            return data
        return None

    def _draw_octave(self, psd, records=None):
        """The banded conversion previewed as steps over the
        narrowband it integrates.

        Recomputed whole on every draw — banding is a resample and an
        integration, cheap at any spacing — and drawn in the preview
        color on the plot rows whose quantity matches, so a mixed
        object's forces never step across its accelerations' axes.
        `records` restricts to what is on the plot: a sub-item picked
        in the tree draws one curve, and the preview stepping every
        other channel over it said the selection one thing and the
        plot another (Brandon, 2026-08-30).
        """
        import pyqtgraph as pg

        from ..plot import step_outline

        self._clear_octave()
        panel = self.data_pane.octave_panel
        try:
            banded = psd.to_octave(panel.per_octave())
        except ValueError as refusal:
            panel.show_bands(None)
            panel.show()
            self._show_status(f'No octave preview: {refusal}')
            return
        colors = resolve_theme(self.theme_name)
        x, rows = step_outline(banded.abscissa,
                               banded.display_ordinate(self.unit_system),
                               banded.bin_widths())
        rows = np.atleast_2d(rows)
        wanted = (range(rows.shape[0]) if records is None
                  else [int(i) for i in records])
        pen = pg.mkPen(colors['filter_preview'], width=2)
        for item in self.data_pane.graphics.ci.items:
            if not isinstance(item, pg.PlotItem):
                continue
            key = getattr(item, 'series_key', None)
            for k in wanted:
                if key is not None and (banded.ordinate_dim[k],
                                        banded.dimension_hint[k]) \
                        != (key[1], key[2]):
                    continue
                curve = pg.PlotDataItem(x, np.abs(rows[k]), pen=pen)
                curve.setZValue(20)
                curve.is_zone_edge = True     # a preview, not data
                item.addItem(curve, ignoreBounds=True)
                self.octave_previews.append((item, curve))
        panel.show_bands(len(banded.abscissa))
        panel.show()
        self._journal_octave_view(psd, panel.per_octave())

    def _journal_octave_view(self, psd, per_octave):
        try:
            name = self.project.name_of(psd)
        except (KeyError, ValueError):
            return
        self._journal_view(
            f"project[{name!r}].to_octave(",
            f"project[{name!r}].to_octave({per_octave})"
            f".plot(path='octave.png', show=False)")

    def _clear_octave(self):
        """Take the steps off, before every drawing, like every other
        preview."""
        for item, curve in self.octave_previews:
            item.removeItem(curve)
        self.octave_previews = []

    def _draw_truncation(self, history):
        """Gray the ends being cut away and put the span beside it.

        One overlay per plot, like the averaging marks: a history
        holding force and acceleration is drawn as a row each, and
        the cut lands on both.
        """
        import numpy as np
        import pyqtgraph as pg

        from ..plot.truncation import TruncationOverlay

        self._clear_truncation()
        abscissa = np.asarray(history.abscissa, dtype=float)
        first, last = float(abscissa[0]), float(abscissa[-1])
        truncation = history.truncation or history.suggest_truncation()
        colors = resolve_theme(self.theme_name)
        for item in self.data_pane.graphics.ci.items:
            if not isinstance(item, pg.PlotItem):
                continue
            overlay = TruncationOverlay(item, truncation, first, last,
                                        colors)
            overlay.changed.connect(self._truncation_dragged)
            self.truncation_overlays.append(overlay)
        panel = self.data_pane.truncate_panel
        panel.show_history(history, truncation)
        panel.show()

    def _clear_truncation(self):
        """Take the shades off, before every drawing, like the frames
        and the previews."""
        self._clear_overlays('truncation_overlays')

    def _truncation_edited(self, truncation):
        """The panel moved: store it and restate the shades.

        Storing is what makes the badge work — a truncated record's
        provenance fingerprints the source's `truncation`, so the
        edit is the moment its refresh badge appears.
        """
        self._span_settled('truncation', truncation,
                           self.truncation_overlays)

    def _truncation_dragged(self, truncation):
        """The span moved: store it and restate the panel — and the
        other plots, which are showing the same record."""
        self._span_settled('truncation', truncation,
                           self.truncation_overlays,
                           self.data_pane.truncate_panel.set_truncation)

    def _filtering_edited(self, filtering):
        """The panel moved: store it and restate the previews.

        Storing is what makes the badge work — a filtered record's
        provenance fingerprints the source's `filtering`, so the edit
        is the moment its refresh badge appears.
        """
        history = self._averaging_history()
        if history is None:
            return
        history.filtering = filtering
        self.project.record_setting(history, 'filtering', filtering)
        for overlay in self.filter_overlays:
            overlay.set_filtering(filtering)
        # and the stage's twin, when that is the reading up: the flat
        # overlays restate themselves in place, so this is the same
        # loop-safe path the averaging edit takes
        redraw = self._stage_filter_redraw
        if redraw is not None and self.data_pane.showing_waterfall \
                and self.data_pane.showing_filter:
            redraw(filtering)
        self._refresh_stale_badges()
        self._report_content_changed(settling=True)

    def _draw_shocks(self, history):
        """Bracket the events on the plot and list them beside it.

        The same shape as the averaging view, and for the same reason:
        the parameters live on the history, the view shows and edits
        them, and the calculator uses whatever is there when it runs.
        """
        import pyqtgraph as pg

        from ..plot.shocks import ShockOverlay

        self._clear_shocks()
        try:
            _rate = history.sample_rate
        except ValueError as refusal:
            # unevenly sampled, so there are no windows to speak of
            self._show_status(f'No shocks: {refusal}')
            self.data_pane.shock_panel.hide()
            return
        from ..core.shocks import uniform

        found = tuple(history.shocks or ())
        locked = bool(history.split_into_frames)
        colors = resolve_theme(self.theme_name)
        for item in self.data_pane.graphics.ci.items:
            if not isinstance(item, pg.PlotItem):
                continue
            self.shock_overlays.append(ShockOverlay(
                item, found, colors, changed=self._shocks_dragged,
                locked=locked, common=uniform(found),
                limit=self._record_end(history)))
        panel = self.data_pane.shock_panel
        panel.set_shocks(found, locked=locked)
        panel.show_band(*history.srs_band())
        panel.show()

    def _clear_shocks(self):
        self._clear_overlays('shock_overlays')

    def _clear_overlays(self, name):
        for overlay in getattr(self, name):
            overlay.remove()
        setattr(self, name, [])

    def _record_end(self, history):
        """Where the record stops, in seconds."""
        return (np.asarray(history.ordinate).shape[-1]
                / float(history.sample_rate))

    def _store_shocks(self, found):
        """Put the events on the history and restate the marks.

        Whatever route an edit arrived by, it is held inside the record
        and off its neighbors here — the last place every path passes
        through, so the invariant is stated once rather than in each of
        the three callers that can break it.

        The panel and every overlay are told, but only the ones that do
        not already agree — set_shocks is a no-op on a match, which is
        what keeps a drag from redrawing the region being dragged.
        """
        from ..core.shocks import held_apart, uniform

        history = self._averaging_history()
        if history is None:
            return None
        samples = int(np.asarray(history.ordinate).shape[-1])
        rate = float(history.sample_rate)
        # clipped only where it has to be: `clipped` round-trips through
        # sample indices, so calling it on a window that already fits
        # would snap a dragged edge to a sample and move a number the
        # user just typed. Out-of-record is what is being fixed here,
        # not precision.
        history.shocks = held_apart(tuple(
            s if s.fits(samples, rate) else s.clipped(samples, rate)
            for s in found))
        self.project.record_setting(history, 'shocks', history.shocks)
        shared = uniform(history.shocks)
        for overlay in self.shock_overlays:
            overlay.set_shocks(history.shocks)
            # restated with the list, not only at construction: Detect
            # replaces the windows without rebuilding the overlays, and
            # a stale mode here is a drag that does the other thing
            overlay.common = shared
        self.data_pane.shock_panel.set_shocks(
            history.shocks, locked=bool(history.split_into_frames))
        self._refresh_stale_badges()
        self._report_content_changed(settling=True)
        # the stage's slabs are rebuilt with the scene, so an edit
        # while the 3-D reading is up re-renders it — same reason and
        # same loop-safety as the averaging panel's edits
        pane = self.data_pane
        if pane._waterfall_page is not None and \
                pane._waterfall_page.isVisible():
            self.render_current()
        return history

    def _shocks_edited(self, found):
        """The table was typed into."""
        self._store_shocks(found)

    def _add_shock(self):
        """The Add button: a window the detector missed.

        Placed by `core.shocks.with_added` — the series' own length,
        in the largest open stretch — and committed through the same
        store as every other edit, so it lands held apart, clipped,
        on the panel, the plot and the stage alike.
        """
        from ..core.shocks import with_added

        history = self._averaging_history()
        if history is None:
            return
        was = tuple(history.shocks or ())
        grown = with_added(was, self._record_end(history))
        if len(grown) == len(was):
            self._show_status('No room for another window — the record '
                              'is windowed wall to wall')
            return
        self._store_shocks(grown)

    def _shocks_dragged(self, found):
        """A region on the plot was pulled."""
        self._store_shocks(found)

    def _shock_length_mode(self, common):
        """The panel's "same length" box was ticked or cleared.

        Only the overlays need telling: the box changes what a drag
        does, and the list itself the panel has already settled.
        """
        for overlay in self.shock_overlays:
            overlay.common = bool(common)

    def _detect_shocks(self):
        """The Detect button: find the events and replace the list."""
        from ..core.shocks import find

        history = self._averaging_history()
        if history is None:
            return
        try:
            found = find(history)
        except ValueError as refusal:
            self._show_status(f'No shocks: {refusal}')
            return
        self._store_shocks(found)
        self._show_status(
            f'{len(found)} shock{"s" * (len(found) != 1)} found'
            if found else
            'No shocks found — nothing in this record rises clear of its '
            'own quiet')

    def _averaging_history(self):
        """The history the averaging view is currently describing."""
        _kinds, series = self._current_series()
        return self._sole_history(series)

    def _averaging_edited(self, averaging):
        """The panel moved: store it and restate the marks."""
        self._span_settled('averaging', averaging, self.averaging_overlays)

    def _averaging_dragged(self, averaging):
        """The region moved: store it and restate the panel — and the
        other plots, which are showing the same record."""
        self._span_settled('averaging', averaging, self.averaging_overlays,
                           self.data_pane.averaging_panel.set_averaging)

    def _span_settled(self, setting, value, overlays, restate_panel=None):
        """A record's span setting (its averaging or its truncation)
        moved, on the panel or on a plot: store it and restate
        everything showing it. One rule for the four routes, the way
        `_store_shocks` is one for the shocks'.

        Storing is what makes the badge work — a derived record's
        provenance fingerprints the source's setting, so the edit is
        the moment its refresh badge appears. The flat overlays
        restate themselves in place, skipping the one already there
        (the one that was dragged); the stage's marks are rebuilt with
        the scene, so a move while the 3-D reading is up re-renders it
        — a window-type change has to reshape the glyphs there too,
        and the re-render is what re-arms the dragger with the stored
        span: without it the next drag anchored on the *old* span, and
        dragging the start put the stop back where it began (Brandon,
        2026-08-24). Loop-safe: the re-render restates the panel
        through its setter, whose `_loading` guard keeps it from
        re-emitting.
        """
        history = self._averaging_history()
        if history is None:
            return
        setattr(history, setting, value)
        self.project.record_setting(history, setting, value)
        if restate_panel is not None:
            restate_panel(value)
        for overlay in overlays:
            if getattr(overlay, setting) != value:
                getattr(overlay, f'set_{setting}')(value)
        self._refresh_stale_badges()
        self._report_content_changed(settling=True)
        pane = self.data_pane
        if pane._waterfall_page is not None and \
                pane._waterfall_page.isVisible():
            self.render_current()

    @staticmethod
    def _has_imaginary_part(data, records):
        """Is there anything for the component box to choose between?

        A CPSD is stored complex because its cross terms are, but the
        diagonal of that same array is not, and the box is offered for
        what is on the plot rather than for what the object holds. So
        ask the selected records: pick the autospectra out of a CPSD
        and there is no phase to choose a component of.
        """
        values = data.ordinate
        if records is not None:
            values = values[list(records)]
        return has_phase(values)

    def _frequency_axis_toggled(self, checked):
        """The viewer's choice of decades or hertz, for every frequency
        plot from here on; the stage and the report read the same
        switch."""
        from ..core.data import frequency_axis

        frequency_axis('log' if checked else 'linear')
        self.render_current()

    def _offer_frequency_axis(self, series):
        """The decades/hertz toggle, for whatever is over frequency.
        Returns the note for a 0 Hz line that a log axis cannot draw,
        or ''."""
        over_frequency = [data for _n, data, _r in series
                          if data.abscissa_dim == 'frequency']
        log = bool(over_frequency) and bool(over_frequency[0].log_abscissa)
        self.data_pane.offer_frequency_axis(bool(over_frequency), log)
        if log and any(np.any(np.asarray(data.abscissa) <= 0)
                       for data in over_frequency):
            return ' — the 0 Hz line is off the log axis'
        return ''

    def _render_series(self, series, cursor=False):
        wants_map = self._map_wanted(series)
        diagonals = self._drive_points_for(series)
        all_frf = bool(series) and all(isinstance(data, Frf)
                                       for _name, data, _records in series)
        complex_series = bool(series) and all(
            self._has_imaginary_part(data, records)
            for _name, data, records in series)
        # one bar, three vocabularies: coherence picks how to read the
        # data, an FRF filters which records are read or trades its curves
        # for the matrix's singular values. Each control shows only for
        # the data it applies to.
        offered = self._pair_selected or self._pair_with_picks
        diagonal = None
        if diagonals is not None:
            asd = all(isinstance(data, Psd) for _n, data, _r in series)
            # checked is derived, never stored: the button is down exactly
            # when the selection is the diagonal, so picking any other cell
            # visibly releases it
            filtered = all(
                records is not None
                and sorted(records) == sorted(diagonals[name])
                for name, _data, records in series if name in diagonals)
            diagonal = (
                'Autospectra' if asd else 'Drive points',
                'Only the autospectra (ASDs): response and reference at '
                'the same DOF' if asd else
                'Only the drive-point FRFs: response and reference at the '
                'same DOF',
                filtered)
        history = self._sole_history(series)
        self.data_pane.show_controls(map_wanted=wants_map, diagonal=diagonal,
                                     cmif=all_frf,
                                     complex_data=complex_series,
                                     pair=offered,
                                     averaging=history is not None,
                                     shocks=history is not None,
                                     octave=self._octave_source(series)
                                     is not None)
        if history is not None and self.data_pane.showing_wavelet:
            return self._render_wavelet(history, series[0][0],
                                        series[0][2])
        if history is not None and self.data_pane.showing_kurtosis:
            return self._render_kurtosis(history, series)
        if all_frf and self.data_pane.showing_cmif:
            drawn = build_cmif(self.data_pane.graphics, series,
                               unit_system=self.unit_system,
                               theme=self.theme_name)
            return (f'CMIF: {drawn} singular value '
                    f'curve{"s" * (drawn != 1)}')
        if wants_map:
            drawn, requested = build_coherence_map(
                self.data_pane.graphics, series,
                unit_system=self.unit_system, theme=self.theme_name)
            return (f'{drawn} channels' if drawn != 1 else '1 channel')
        # a transient record beside the waveform it was controlled to is
        # its own comparison, and none of the machinery below fits it:
        # there is no band to fall outside and no spectrum to integrate,
        # there is one target played over and over. It goes first
        # because it settles which set of readings the bar offers.
        replication = self._replication_found(series)
        self.data_pane.show_replication_views(replication is not None)
        if replication is not None:
            return self._render_replication(*replication)
        # a measured shock spectrum beside the one it had to meet. Same
        # shape as the transient comparison — several playings of one
        # target — so it gets the same grid and the same two readings
        spectra = self._srs_found(series)
        self.data_pane.show_srs_views(spectra is not None)
        if spectra is not None:
            return self._render_srs(*spectra)
        # a specification on its own reads two ways: its spectra, or
        # the level each channel asks for as bars over the table
        levels = self._specification_rows(series)
        self.data_pane.offer_rms(levels is not None)
        if levels is not None and self.data_pane.rms_wanted:
            return self._render_levels(levels)
        # three readings of one comparison, and the bar offers them:
        # the spectra themselves, the level each channel came out at,
        # and how much of each channel's band fell outside
        comparison = self._compliance_rows(series)
        self.data_pane.show_comparison_views(comparison is not None)
        # two plain densities selected together read two ways: drawn
        # over each other, or divided in decibels — the signal-to-
        # noise reading (Brandon, 2026-08-25). A specification is not
        # a density pair; it has the comparison machinery above.
        densities = (len(series) == 2 and all(
            isinstance(data, Psd) and not isinstance(data, Specification)
            for _name, data, _records in series))
        self.data_pane.show_spectra_views(densities)
        if densities:
            # the pair's readings are three-dimensional by default —
            # shared channels receding, overlaid or divided — and the
            # 2D/3D toggle stands down to the flat pair (Brandon,
            # 2026-08-25); the tail's offer keeps the toggle up
            self.data_pane.offer_waterfall(True)
            if self.data_pane.showing_waterfall:
                drawn = self._render_pair_stage(series)
                if drawn is not None:
                    return drawn
            elif self.data_pane.spectra_view == 'ratio':
                drawn = self._render_ratio(series)
                if drawn is not None:
                    return drawn
        self._show_comparison_scaling(series if comparison is not None
                                      else None)
        if comparison is not None:
            # the account of every channel, under whichever reading is
            # on top of it — the same arrangement the transient
            # comparison has, and for the same reason: the plot draws
            # the channels that were picked and is never the whole list
            self._show_compliance_table(comparison)
        if comparison is not None and self.data_pane.comparison_view != 'curves':
            return self._render_bars(comparison)
        # a density pair upstream has already claimed the toggle; the
        # comparison machinery below must not un-offer it
        banded_offered = densities
        if comparison is not None:
            # the spectra reading of a comparison is three-dimensional
            # by default — every channel at once, channels receding —
            # and the 2D/3D toggle stands down to the paged flat view
            banded_offered = True
            self.data_pane.offer_waterfall(True)
            if self.data_pane.showing_waterfall:
                specs = [(name, data, records)
                         for name, data, records in series
                         if isinstance(data, Specification)]
                others = [(name, data, records)
                          for name, data, records in series
                          if isinstance(data, Psd)
                          and not isinstance(data, Specification)]
                spec_name, spec, spec_records = specs[0]
                m_name, m_data, m_records = others[0]
                return self._render_banded(
                    spec_name, spec, spec_records, m_data, m_name,
                    m_records,
                    scale_db=self._family_scale_db(spec, m_name))
        # a specification bounds autospectra; drawn beside a full CPSD its 30
        # cross terms have no limit near them and bury the six that do —
        # and the window resolves each measured entry's comparison scale
        # itself, because it knows what the plot function cannot: an
        # octave PSD and the PSD it was banded from share one scale
        if len(series) == 1:
            lone_name, lone, lone_records = series[0]
            if (isinstance(lone, Bounded) and lone.has_limits
                    and not isinstance(lone, TransientSpecification)):
                # a specification alone is its channels and their
                # bands, and the stage shows all of them at once
                banded_offered = True
                self.data_pane.offer_waterfall(True)
                if self.data_pane.showing_waterfall:
                    return self._render_banded(lone_name, lone,
                                               lone_records)
        series, unbounded = bounded_by_specification(
            series, scales=self._series_scales(series))
        # and one comparison at a time: a specification with its response
        # is the thing being read, and six of them on one axis is a
        # thicket. The bar offers the rest.
        pairs = specification_pairs(series)
        # a transient specification alone reads one channel at a time
        # too: its waveforms stacked are a thicket with no comparison
        # in them, so the same drop-down reaches the rest
        lone_fed = False
        if not pairs and len(series) == 1:
            lone_name, lone, lone_records = series[0]
            if isinstance(lone, TransientSpecification):
                rows = (lone_records if lone_records is not None
                        else range(lone.num_records))
                dofs = list(dict.fromkeys(lone.response_dof[i]
                                          for i in rows))
                if len(dofs) > 1:
                    self.data_pane.show_pairs([(d, d) for d in dofs])
                    chosen = self.data_pane.chosen_pair()
                    dof = (chosen[0] if chosen is not None
                           and chosen[0] in dofs else dofs[0])
                    series = [(lone_name, lone,
                               [i for i in rows
                                if lone.response_dof[i] == dof])]
                    lone_fed = True
        if not lone_fed:
            self.data_pane.show_pairs(pairs)
        # which comparison the plot is actually drawing, so the table
        # beside it can say which of its rows is the one on screen
        self._plotted_pair = pairs[0] if pairs else None
        if len(pairs) > 1:
            # the table under the plot outranks the box beside it, the
            # same way the transient's grid does — and it can name
            # several channels where a drop-down names one
            # matched by label, not by tuple: the table names its rows
            # the way the drop-down does — `channel_errors` hands back
            # '121Z+' where `specification_pairs` hands back
            # ('121Z+', '121Z+') — and matching the text is what keeps
            # the two agreeing without either learning the other's
            # model. Compared as tuples this never matched, so every
            # pick silently fell back to the first channel.
            wanted = set(self._compliance_channels)
            picked = [pair for pair in pairs if pair_label(pair) in wanted]
            if not picked:
                chosen = self.data_pane.chosen_pair()
                picked = [chosen if chosen in pairs else pairs[0]]
            self._compliance_channels = [pair_label(p) for p in picked]
            self._plotted_pair = picked[0]
            series = only_pairs(series, picked)
        # the plain-curves path is the one place the 3-D reading applies:
        # one object, no comparison machinery holding the plot — a single
        # record spreads to a single line, and the color still reads.
        # The toggle is offered here rather than in show_controls because
        # only this point knows the render came all the way down — the
        # same reason show_pairs raises the bar itself. Anything that is
        # a *mark on the flat plot* outranks it while it is up: the
        # averaging and shock frames, and the animation cursor that
        # drives an ODS or an envelope beside a geometry — each needs
        # the plot it marks, and the checked choice waits underneath
        # A measured set beside its resynthesis is a pair like any
        # other — two objects holding the same channels — so it reads
        # on the stage the way a density pair does, station by station,
        # rather than as a hundred curves interfering on one axis
        # (Brandon, 2026-08-26). The synthesis is the stood-back one:
        # the flat plot draws it dashed *under* the measurement, and
        # the emphasis must not flip when the toggle does.
        resynthesis = (len(series) == 2
                       and getattr(series[1][1], 'synthesized', False)
                       and not getattr(series[0][1], 'synthesized', False))
        if resynthesis:
            self.data_pane.offer_waterfall(True)
            if self.data_pane.showing_waterfall:
                measured, synthesized = series
                info = self._draw_pair_stage(measured, synthesized,
                                             'overlay')
                if info is not None:
                    missed = info.get('unmatched', 0)
                    note = (f' ({missed} unmatched)' if missed else '')
                    return (f'{measured[0]} with {synthesized[0]} '
                            f'over it: {info["drawn"]} '
                            f'channel{"s" * (info["drawn"] != 1)} on the '
                            f'stage, the synthesis in gray{note}')
        entry = series[0] if len(series) == 1 else None
        if entry is None and series and not pairs and not lone_fed \
                and not any(getattr(data, 'synthesized', False)
                            for _name, data, _records in series):
            # a selection spanning several objects can still read on
            # one stage (Brandon, 2026-08-30: an FRF from each of two
            # surveys had no 3-D reading at all). A synthesis stays
            # out: measured-beside-synthesized has its own paired
            # readings, flat overlay and pair stage both, and the
            # stack must not swallow them.
            entry = self._stacked_entry(series)
        # the averaging and shock readings work on the stage too now —
        # the marks draw as stage geometry and the side panel edits in
        # either view (Brandon, 2026-08-23) — so only the animation
        # cursor still outranks the toggle: an ODS is driven from the
        # flat plot's cursor and has no 3-D reading
        offered = (entry is not None and not pairs and not lone_fed
                   and not cursor
                   and self._waterfall_count(entry) >= 1)
        # a stage offered upstream stays offered: this flat pass is the
        # toggle standing down, and recomputing visibility here hid the
        # very button that brings the stage back (Brandon, 2026-08-22)
        self.data_pane.offer_waterfall(offered or banded_offered
                                       or resynthesis)
        if offered and self.data_pane.showing_waterfall:
            return self._render_waterfall(entry, complex_series)
        self.data_pane.show_waterfall(False)
        # The flat plot pages too, and by the same control. Its limit is
        # `MAX_RECORDS` rather than a vertex budget, because what stops
        # a 2-D plot is legibility, not memory: past fifty curves they
        # overlay into a band and most are not even drawn. It used to
        # say '(first 50 shown)' and leave the rest reachable only by
        # picking records in the grid — true, and a dead end. Paging by
        # slicing the record list leaves `build_plots` and its
        # across-axes `curve_budget` untouched.
        whole = series          # before paging: what the user selected
        series, page, pages = self._paged_series(series, entry, pairs,
                                                 lone_fed)
        self.data_pane.show_pages(page, pages)
        drawn, requested = build_plots(
            self.data_pane.graphics, series, unit_system=self.unit_system,
            theme=self.theme_name,
            component=self.data_pane.component()
            if complex_series else 'magnitude')
        axis_note = self._offer_frequency_axis(series)
        if len(series) == 1:
            # the *unpaged* entry: paging swaps records=None for a slice,
            # and reading that back would report a whole object as
            # '50 selected records' — a selection the user never made
            _name, data, records = whole[0]
            capture_paged = (pages > 1 and self.data_pane.showing_averaging
                             and isinstance(data, TimeHistory)
                             and any(data.capture_indices()))
            if capture_paged:
                # the pager is stepping playings, one at a time
                # (Brandon, 2026-08-28) — the tail says which, and
                # the panel's row says what they pool into
                message = (f'{drawn} channel{"s" * (drawn != 1)} — '
                           f'capture {page + 1} of {pages}')
            elif records is not None and len(records) == 1:
                message = data.record_label(records[0])
            elif records is not None:
                message = f'{drawn} curves from {len(records)} selected records'
            else:
                message = self._status_for(data)
                if pages > 1:
                    # a page, not a truncation: '(first 50 shown)' was
                    # true and read as a dead end, which it no longer is
                    message += (f' — page {page + 1} of {pages}, '
                                f'{data.num_records} in all')
                elif drawn < requested:
                    message += f' (first {drawn} shown)'
        else:
            message = f'{drawn} curves from {len(series)} selections'
            if drawn < requested:
                message += f' (of {requested} requested)'
        if unbounded:
            message += (f'; {unbounded} records hidden with no matching '
                        'specification')
        if cursor and series:
            self._add_plot_cursor(series[0][1])
        # the frames go on last: they are marks on the plot that was just
        # built, and there is nothing to mark until it exists
        if history is not None and self.data_pane.showing_averaging:
            self._draw_averaging(history)
        if history is not None and self.data_pane.showing_shocks:
            self._draw_shocks(history)
        if history is not None and self.data_pane.showing_filter:
            self._draw_filtering(history)
        if history is not None and self.data_pane.showing_truncate:
            self._draw_truncation(history)
        banded_source = self._octave_source(series)
        if banded_source is not None and self.data_pane.showing_octave:
            self._draw_octave(banded_source, series[0][2])
        return message + axis_note

    def _paged_series(self, series, entry, pairs, lone_fed):
        """(series, page, pages) with the flat plot's records paged.

        Only for the plain single-object drawing — the same case the
        3-D reading pages, and for the same reason: a comparison, a
        specification pairing or several objects at once are each
        already the small, deliberate selection the pager exists to
        rescue you from.
        """
        if entry is None or pairs or lone_fed:
            return series, 0, 1
        name, data, records = entry
        wanted = (list(range(data.num_records)) if records is None
                  else [int(i) for i in records])
        capture = self._capture_page(name, data, wanted)
        if capture is not None:
            page, pages, picked = capture
            self._paged_object = name
            return [(name, data, picked)], page, pages
        if len(wanted) <= MAX_RECORDS:
            return series, 0, 1
        pages = -(-len(wanted) // MAX_RECORDS)
        page = (self.data_pane.chosen_page()
                if self._paged_object == name else 0)
        page = max(0, min(page, pages - 1))
        self._paged_object = name
        return ([(name, data, wanted[page * MAX_RECORDS:
                                     (page + 1) * MAX_RECORDS])],
                page, pages)

    def _capture_page(self, name, data, wanted):
        """(page, pages, record indices) of the playing on show — or
        None when the averaging view is not reading in playings.

        Under the averaging view a multi-capture record draws one
        playing at a time, in both views (Brandon, 2026-08-28, after
        driving two concatenated layouts and loving neither): the
        pager steps playings, the marks sit on the record's own
        clock, and the panel's '20 × 3 = 60' row is what tells the
        pooling. One implementation — the flat pager and the stage
        call this same method.
        """
        if not (self.data_pane.showing_averaging
                and isinstance(data, TimeHistory)):
            return None
        ordinals = data.capture_indices()
        if not any(ordinals):
            return None
        present = sorted({ordinals[i] for i in wanted})
        if len(present) < 2:
            return None
        page = (self.data_pane.chosen_page()
                if self._paged_object == name else 0)
        page = max(0, min(page, len(present) - 1))
        chosen = present[page]
        return (page, len(present),
                [i for i in wanted if ordinals[i] == chosen])

    #: how many records a cross-object stage will stack before the
    #: stage stands down — stacking copies the selected rows, and two
    #: whole surveys would be a multi-gigabyte copy per redraw
    STACK_LIMIT = 500

    def _stacked_entry(self, series):
        """One stage entry for a selection spanning several objects.

        Same concrete type and the same abscissa, or None: a record
        from each of two surveys reads station by station on one stage
        exactly as two records of one object do. Display only —
        nothing is merged and nothing is stored, so the merge rules'
        refusals (duplicate pairs, mixed capture counts) rightly do
        not apply: they protect stored objects from telling false
        stories, and a drawing tells none.
        """
        first = series[0][1]
        if not isinstance(first, DataArray):
            return None
        if any(type(data) is not type(first) for _n, data, _r in series):
            return None
        if any(not np.array_equal(data.abscissa, first.abscissa)
               for _n, data, _r in series[1:]):
            return None
        chosen = [(name, data,
                   list(range(data.num_records)) if records is None
                   else [int(i) for i in records])
                  for name, data, records in series]
        if sum(len(rows) for _n, _d, rows in chosen) > self.STACK_LIMIT:
            return None
        referenced = first.reference_dof is not None
        rows, resp, ref = [], [], []
        dims, units, runits, hints = [], [], [], []
        for _name, data, wanted in chosen:
            ordinate = np.asarray(data.ordinate)
            for i in wanted:
                rows.append(ordinate[i])
                resp.append(data.response_dof[i])
                if referenced:
                    ref.append(data.reference_dof[i])
                dims.append(data.ordinate_dim[i])
                units.append((data.ordinate_unit or
                              [None] * data.num_records)[i])
                runits.append((data.reference_unit or
                               [None] * data.num_records)[i])
                hints.append((data.dimension_hint or
                              [None] * data.num_records)[i])
        stacked = type(first)(
            first.abscissa, np.asarray(rows), response_dof=resp,
            reference_dof=ref if referenced else None,
            ordinate_dim=dims, ordinate_unit=units,
            reference_unit=runits, dimension_hint=hints)
        name = ' + '.join(dict.fromkeys(n for n, _d, _r in chosen))
        return (name, stacked, None)

    @staticmethod
    def _waterfall_count(entry) -> int:
        """How many records this drawing would put on the stage. A
        single record is a legitimate waterfall — one line, color by
        level — so this guards only the empty case."""
        _name, data, records = entry
        return len(records) if records is not None else data.num_records

    def _render_waterfall(self, entry, complex_series: bool) -> str:
        """The 3-D reading: the records spread along a depth axis.

        Redrawn from scratch on every reread — the scene is one mesh
        and costs nothing to rebuild — but the camera is placed only
        when the displayed object changes, so a units switch, a record
        pick or a component change happens under the view the user set.
        """
        from ..viz.waterfall import (
            add_waterfall,
            place_camera,
            quantity_label,
            waterfall_groups,
        )

        name, data, records = entry
        pane = self.data_pane
        pane.show_waterfall(True)
        # the flat plot holds nothing while the 3-D reading is up —
        # build_plots clears it on its way in, and this path must not
        # leave the previous drawing's curves standing behind the swap
        pane.graphics.clear()
        # one floor, one quantity — and the quantity box is how a mixed
        # object's other floors are reached, where the flat plot would
        # have stacked an axis per quantity. Largest first, which is
        # also what draws when nothing has been chosen
        groups = sorted(waterfall_groups(data, records),
                        key=lambda pair: -len(pair[1]))
        entries = []
        if len(groups) > 1:
            words = [quantity_label(key) for key, _members in groups]
            for k, (key, _members) in enumerate(groups):
                word = words[k]
                # two groups can share a word — an FRF in acceleration
                # over force and one over voltage both read
                # 'acceleration' — and a box that repeats itself is
                # undecidable, so the full dimension tells them apart
                if words.count(word) > 1:
                    word = f'{word} ({key[0]})'
                entries.append((word, key))
        pane.show_quantities(entries)
        plotter = pane.waterfall_plotter
        plotter.clear()
        colors = resolve_theme(self.theme_name)
        plotter.set_background(colors['scene_background'],
                               top=colors['scene_background_top'])
        # the page the stepper is on — reset to the first whenever the
        # object changes, since page 3 of something else is not a place
        page = pane.chosen_page() if self._paged_object == name else 0
        self._paged_object = name
        # under the filter view the raw record is the reference, not
        # the reading: it stands back in gray and the filtered twin
        # drawn over it carries the color (Brandon, 2026-08-25)
        stood_back = (colors['specification_curve']
                      if pane.showing_filter and isinstance(data, TimeHistory)
                      else None)
        # the same rule as the flat plot, through the same method:
        # under the averaging view a multi-capture record stages one
        # playing at a time, and the stepper steps playings
        capture = (self._capture_page(
            name, data,
            list(range(data.num_records)) if records is None
            else [int(i) for i in records])
            if isinstance(data, TimeHistory) else None)
        if capture is not None:
            page, pages, records = capture
        info = add_waterfall(plotter, data, records,
                             unit_system=self.unit_system,
                             theme=self.theme_name,
                             component=pane.component()
                             if complex_series else 'magnitude',
                             quantity=pane.chosen_quantity()
                             if entries else None,
                             page=0 if capture is not None else page,
                             color=stood_back)
        if capture is not None:
            info['page'], info['pages'] = page, pages
        pane.show_pages(info['page'], info['pages'])
        # the marks go on last here too: frames and shock windows are
        # stage geometry over the stage that was just built
        self._stage_marks(plotter, data, info)
        if pane.showing_octave and \
                self._octave_source([(name, data, records)]) is not None:
            self._stage_octave_preview(plotter, data, info)
        #: which records the stage is actually showing — the status
        #: line says how many, this says which
        self._waterfall_drawn: list[int] = list(info['drawn'])
        if self._waterfall_object != name:
            place_camera(plotter)
            self._waterfall_object = name
        plotter.render()
        drawn = len(info['drawn'])
        message = (f'{drawn} channel{"s" * (drawn != 1)} as a waterfall'
                   + self._offer_frequency_axis([(name, data, records)]))
        if capture is not None:
            message = (f'{drawn} channel{"s" * (drawn != 1)} — '
                       f"capture {info['page'] + 1} of {info['pages']}")
        elif info['pages'] > 1:
            # never a silent cap: what a scene can draw exactly is what
            # it draws, and the rest is a page away rather than thinned
            message += (f" — page {info['page'] + 1} of {info['pages']}"
                        f", {info['of_quantity']} in all")
        if info['left_out']:
            message += (f"; {len(info['left_out'])} of other quantities "
                        '— the quantity box beside 3D chooses')
        return message

    def _stage_filter_preview(self, plotter, data, info) -> None:
        """The filter previewed on the stage, and its panel beside it.

        The filtered record is staged a second time and drawn against
        the *raw* stage's extents, so the twin sits where it belongs
        against what it is being compared to — see
        `viz.marks.add_filter_preview`. Same records, same page, same
        budget, so twin and original stand at the same stations.
        """
        from ..core.filters import filtered as apply_filter
        from ..viz.marks import add_filter_preview
        from ..viz.waterfall import waterfall_arrays

        pane = self.data_pane
        filtering = data.filtering or data.suggest_filtering()
        panel = pane.filter_panel
        panel.show_history(data, filtering)
        panel.show()

        def draw(proposed):
            try:
                filtered = apply_filter(data, proposed)
            except ValueError as refusal:
                # a corner that has wandered past Nyquist mid-edit:
                # nothing to draw rather than a scene that throws
                self._show_status(f'No preview: {refusal}')
                return
            twin = waterfall_arrays(
                filtered, info['drawn'], self.unit_system,
                page=0)
            add_filter_preview(plotter, twin, info['extents'],
                               info['stations'], theme=self.theme_name)
            plotter.render()

        draw(filtering)
        self._stage_filter_redraw = draw

    def _stage_octave_preview(self, plotter, psd, info) -> None:
        """The banded conversion previewed on the stage, panel beside.

        The same steps the flat preview draws, at each drawn record's
        own station — through `step_outline` for the geometry and the
        raw stage's extents for the scale, the filter preview's rule.
        The reading transforms once here, exactly as `waterfall_arrays`
        transforms the stage it lands on: log10 of the frequency axis
        and of the level where those axes are logarithmic.
        """
        from ..plot import step_outline
        from ..viz.marks import add_octave_preview

        panel = self.data_pane.octave_panel
        try:
            banded = psd.to_octave(panel.per_octave())
        except ValueError as refusal:
            panel.show_bands(None)
            panel.show()
            self._show_status(f'No octave preview: {refusal}')
            return
        x, rows = step_outline(banded.abscissa,
                               banded.display_ordinate(self.unit_system),
                               banded.bin_widths())
        rows = np.abs(np.atleast_2d(rows))
        if getattr(banded, 'log_abscissa', False):
            x = np.log10(np.where(x > 0, x, np.nan))
        if banded.log_scaled():
            rows = np.where(rows > 0, rows, np.nan)
            rows = np.log10(rows)
        curves = [(x, rows[k]) for k in info['drawn']]
        add_octave_preview(plotter, curves, info['extents'],
                           info['stations'], theme=self.theme_name)
        panel.show_bands(len(banded.abscissa))
        panel.show()
        self._journal_octave_view(psd, panel.per_octave())

    def _stage_marks(self, plotter, data, info) -> None:
        """The averaging, shock or filter reading, drawn on the stage.

        The same numbers the flat overlays draw, without the drag: the
        side panel is the editor in this view, by design — read-only
        marks first, handles only if they are ever missed (Brandon,
        2026-08-23).
        """
        pane = self.data_pane
        if self._stage_dragger is not None:
            # whatever the marks meant last render, this one decides
            self._stage_dragger.disarm()
        if not isinstance(data, TimeHistory):
            return
        if not (pane.showing_averaging or pane.showing_shocks
                or pane.showing_filter or pane.showing_truncate):
            return
        try:
            rate = data.sample_rate
        except ValueError as refusal:
            # unevenly sampled: no frames or windows to speak of,
            # exactly as the flat overlays refuse
            self._show_status(f'No marks: {refusal}')
            pane.averaging_panel.hide()
            pane.shock_panel.hide()
            pane.filter_panel.hide()
            return
        if pane.showing_filter:
            self._stage_filter_preview(plotter, data, info)
            return
        from ..gui.stage_drag import StageMarksDragger

        if self._stage_dragger is None:
            self._stage_dragger = StageMarksDragger(plotter)
        extents = info['extents']
        if pane.showing_truncate:
            import numpy as np

            from ..viz.marks import add_truncation_marks

            abscissa = np.asarray(data.abscissa, dtype=float)
            first, last = float(abscissa[0]), float(abscissa[-1])
            truncation = data.truncation or data.suggest_truncation()
            add_truncation_marks(plotter, truncation, first, last,
                                 extents, theme=self.theme_name)
            panel = pane.truncate_panel
            panel.show_history(data, truncation)
            panel.show()

            def preview_truncation(proposed):
                add_truncation_marks(plotter, proposed, first, last,
                                     extents, theme=self.theme_name)
                plotter.render()

            self._stage_dragger.arm_truncation(
                truncation, first, last, extents,
                preview_truncation, self._truncation_dragged)
            return
        if pane.showing_averaging:
            from ..viz.marks import add_averaging_marks

            averaging = data.averaging or Averaging.for_records(
                len(data.abscissa))
            add_averaging_marks(plotter, averaging, rate,
                                extents, theme=self.theme_name)
            panel = pane.averaging_panel
            panel.show_history(data, averaging)
            panel.show()

            def preview_averaging(proposed):
                add_averaging_marks(plotter, proposed, rate, extents,
                                    theme=self.theme_name)
                plotter.render()

            self._stage_dragger.arm_averaging(
                averaging, rate, len(data.abscissa), extents,
                preview_averaging, self._averaging_dragged)
        else:
            from ..core.shocks import uniform
            from ..viz.marks import add_shock_marks

            found = tuple(data.shocks or ())
            locked = bool(data.split_into_frames)
            add_shock_marks(plotter, found, extents,
                            theme=self.theme_name, locked=locked)
            panel = pane.shock_panel
            panel.set_shocks(found, locked=locked)
            panel.show_band(*data.srs_band())
            panel.show()
            if not locked:
                def preview_shocks(proposed):
                    add_shock_marks(plotter, proposed, extents,
                                    theme=self.theme_name)
                    plotter.render()

                self._stage_dragger.arm_shocks(
                    found, uniform(found), self._record_end(data),
                    extents, preview_shocks, self._shocks_dragged)

    def _act_on(self, kind, refusal, verb, /, *args, **kwargs):
        """The head every act on the bar shares: the current object has
        to be a `kind` (a class, or a predicate on the object) or the
        refusal is said; the project `verb` runs on the object's name
        with whatever else the act adds; and the verb's own refusal
        lands on the status line under the name. Returns `(name, obj,
        added)`, or None when the act said something and stopped. The
        act's success line stays its own — that sentence is what
        differs between them.
        """
        obj = self.current_object()
        fits = kind(obj) if callable(kind) and not isinstance(kind, type) \
            else isinstance(obj, kind)
        if not fits:
            self._show_status(refusal)
            return None
        name = self.object_item().text(0)
        try:
            added = verb(name, *args, **kwargs)
        except ValueError as why:
            self._show_status(f'{name}: {why}')
            return None
        return name, obj, added

    def compute_spectra(self) -> None:
        """Averaged spectra from the selected time history, one record
        per channel, added to the project beside it."""
        acted = self._act_on(
            TimeHistory,
            'Select a time history to compute spectra',
            self.project.compute_spectra)
        if acted is None:
            return
        name, obj, added = acted
        spectra = self.objects[added]
        frames = obj.num_records // max(spectra.num_records, 1)
        self.show_object(added)
        self._show_status(
            f'{added}: {spectra.num_records} channel '
            f'spectr{"a" if spectra.num_records != 1 else "um"} averaged '
            f'over {frames} frame{"s" * (frames != 1)} — linked to '
            f'{name}')

    def _correlation_candidates(self):
        """{'basis'/'other': (shapes name, shapes, geometry name,
        geometry)} when two whole shape sets are selected, each
        *linked* to its own geometry — explicit association, never a
        guess. The declared Basis defines the DOF space; without one,
        the sparser set stands in — projection is sampling a dense
        field at sparse points, never the reverse. None otherwise."""
        names = self._selected_object_names()
        shape_sets = [(name, self.objects[name]) for name in names
                      if isinstance(self.objects.get(name), ShapeSet)]
        others = [name for name in names
                  if not isinstance(self.objects.get(name),
                                    (ShapeSet, Geometry))]
        if len(shape_sets) != 2 or others:
            return None
        paired = []
        for shapes_name, shapes in shape_sets:
            home = self.linked_geometry(shapes_name)
            if home is None:
                return None      # unlinked: Link it to its geometry
            paired.append((shapes_name, shapes, *home))
        if paired[0][3] is paired[1][3]:
            return None          # both sets linked to one geometry
        roles = [self.link_role(entry[0]) for entry in paired]
        if 'Basis' in roles:
            if roles[1] == 'Basis':
                paired.reverse()
        else:
            paired.sort(key=lambda entry: len(entry[1].coordinate))
            if len(paired[0][1].coordinate) == \
                    len(paired[1][1].coordinate):
                return None      # no Basis declared and no sparser set
        return {'basis': paired[0], 'other': paired[1]}

    def project_onto_basis(self) -> None:
        """The other set sampled at the Basis set's DOFs, added as a
        new object — the project does the projection and the linking."""
        candidates = self._correlation_candidates()
        if candidates is None:
            return
        basis_name = candidates['basis'][0]
        other_name = candidates['other'][0]
        percent, ok = QInputDialog.getDouble(
            self, 'Project onto Basis DOFs',
            'Node match tolerance (% of the basis model size):',
            2.0, 0.01, 100.0, 2)
        if not ok:
            return
        try:
            added = self.project.project_onto_basis(
                other_name, onto=basis_name, tolerance=percent / 100.0)
        except ValueError as refusal:
            self._show_status(f'{other_name}: {refusal}')
            return
        self.show_object(added)
        report = self.objects[added].projection_report
        home = self.project.geometry_for(basis_name)
        if home is not None and home[1].units_defined:
            worst = self.unit_system.from_si(report['worst'], 'length')
            unit = self.unit_system.label_text('length')
        else:
            worst, unit = report['worst'], 'model units'
        note = (f'; {len(report["dropped"])} basis DOFs dropped'
                if report['dropped'] else '')
        self._show_status(
            f'{added}: {report["matched"]} of {report["total"]} basis '
            f'nodes matched (worst {worst:.4g} {unit}){note} — select '
            f'it beside {basis_name} to compare')

    # ---- the transform: a record through a shape set --------------------

    def _transform_candidates(self):
        """(history name, shapes name, direction, records) when the
        selection is one time history — whole, or some of its records
        picked — and one whole shape set, plus at most a geometry, and
        the record reads through the set one way or the other. None
        otherwise. `records` is the pick, or None for the whole
        object: a pick of modal records expands those modes alone
        (Brandon, 2026-09-04), a pick of channels transforms those."""
        from ..core.transform import reads_as

        histories, sets, picks = [], [], {}
        for kind, name, obj, detail in self.selected_references():
            if isinstance(obj, DataArray) and kind in ('object', 'record'):
                if name not in histories:
                    histories.append(name)
                if kind == 'record':
                    picks.setdefault(name, []).append(detail)
                else:
                    picks[name] = None
            elif isinstance(obj, ShapeSet) and kind == 'object':
                if name not in sets:
                    sets.append(name)
            elif isinstance(obj, Geometry) and kind == 'object':
                continue
            else:
                return None
        if len(histories) != 1 or len(sets) != 1:
            return None
        history = histories[0]
        direction = reads_as(self.objects[history], self.objects[sets[0]])
        if direction is None:
            return None
        records = picks.get(history)
        return history, sets[0], direction, (
            None if records is None else sorted(set(records)))

    def transform_selection(self) -> None:
        """The selected record through the selected shape set — modal
        responses from physical, or physical from modal — added to the
        project by the verb a script calls. No pane and no preview: a
        transform has no settings (Brandon, 2026-09-04), so it is an
        act like Integrate, and the account it leaves is on the status
        line and on the object's `transform_report`."""
        candidates = self._transform_candidates()
        if candidates is None:
            self._show_status('Select a data object and a shape set '
                              'together to transform one through the '
                              'other')
            return
        history, shapes, direction, records = candidates
        # the pick only when there is one, so the journal's line is the
        # one a script would write for the whole object
        picked = {} if records is None else {'records': records}
        try:
            if direction == 'physical':
                added = self.project.transform(history, shapes, **picked)
            else:
                added = self.project.expand(history, shapes, **picked)
        except ValueError as refusal:
            self._show_status(f'{history}: {refusal}')
            return
        report = self.objects[added].transform_report
        self.show_object(added)
        self._show_status(f'{added}: {report.describe()}, through {shapes}')

    def generate_typed_report(self) -> None:
        """Generate the report the project's declared type calls for."""
        self.generate_report(PROJECT_TEMPLATES[self.project_type])

    def _show_placeholder_menu(self, item, position):
        """A gray slot's options: compute it from what the project has,
        when that is possible; otherwise say how to fill it."""
        label = item.data(0, ROLE_REFERENCE)[1]
        menu = QMenu(self.tree)
        time_name = next((name for name, obj in self.objects.items()
                          if isinstance(obj, TimeHistory)), None)
        if label == 'PSD' and time_name is not None:
            menu.addAction(
                'Compute &PSDs',
                lambda: self._compute_from_time(time_name, 'psds'))
        spec_present = any(isinstance(obj, SineSweepSpecification)
                           for obj in self.objects.values())
        if label == 'Sine Level Set' and time_name is not None and spec_present:
            menu.addAction(
                'Extract &Sine Levels',
                lambda: self.extract_sine_levels(time_name))
        if label == 'Report' and self.project_type:
            # the type already says which report this is, so the slot
            # fills itself — the same one click the project's own icon
            # gives, offered where the missing thing is named
            menu.addAction('Generate &Report', self.generate_typed_report)
        if not menu.actions():
            hints = {
                'Photos': 'Drop png or jpeg files on the project tree '
                          'to add photos',
                'PSD': 'Import time data first — PSDs can then be '
                       'computed from it',
                ('Geometry', OTHER_SIDE):
                    'Import the geometry to compare against — a model, '
                    'or another test — to enable correlation',
                ('Shape Set', OTHER_SIDE):
                    'Import the shape set to compare against — a model, '
                    'or another test — to enable correlation',
                'Matched Modes': 'Select the two shape sets, pick MAC '
                                 'squares, and press + to commit '
                                 'matched pairs',
            }
            self._show_status(hints.get(
                (label, item.data(0, ROLE_DROP_SLOT)), hints.get(
                    label, f'Import a file holding a {label} to fill '
                           'this slot')))
            return
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _compute_from_time(self, time_name, what):
        """Fill a placeholder from the named time history."""
        item = self._item_for_object(time_name)
        self.tree.setCurrentItem(item)
        if what == 'psds':
            self.compute_psds()
        else:
            self.compute_spectra()

    def extract_sine_levels(self, time_name=None) -> None:
        """Each specification tone's level, read out of the recording
        and added beside it — the project verb, with the tree kept in
        step and the result selected so the comparison is one click."""
        if time_name is None:
            obj = self.current_object()
            if not isinstance(obj, TimeHistory):
                self._show_status(
                    'Select a time history to extract sine levels')
                return
            time_name = self.object_item().text(0)
        try:
            added = self.project.extract_sine(time_name)
        except (ValueError, AttributeError) as refusal:
            self._show_status(f'{time_name}: {refusal}')
            return
        for k, name in enumerate(added):
            self.show_object(name, source=time_name, select=k == 0)
        self._show_status(
            f'{len(added)} tone level{"s" * (len(added) != 1)} '
            f'extracted from {time_name}')

    def compute_psds(self) -> None:
        """Averaged auto-power spectral densities from the selected
        time history, one record per channel, added beside it."""
        acted = self._act_on(
            TimeHistory,
            'Select a time history to compute PSDs',
            self.project.compute_psds)
        if acted is None:
            return
        name, obj, added = acted
        psds = self.objects[added]
        frames, where = self._averaged_frames(obj, psds.num_records)
        self.show_object(added)
        self._show_status(
            f'{added}: {psds.num_records} channel '
            f'PSD{"s" * (psds.num_records != 1)} averaged over '
            f'{frames} frame{"s" * (frames != 1)}{where} — linked to {name}')

    def _averaged_frames(self, history, channels):
        """(frames each channel's average pooled, ' (start–end s)').

        Not the record-count ratio: that reads a stack of captures
        (one frame per record) but calls a continuous recording cut by
        the averaging window '1 frame' — which hid a window that had
        not moved, because the status said nothing either way. The
        span says *which part of the run* was averaged, which is the
        thing the window is for.
        """
        per = history.num_records // max(channels, 1)
        averaging = history.averaging
        if averaging is None:
            return per, ''
        rate = history.sample_rate
        low = averaging.start
        high = low + averaging.span / rate
        return per * averaging.frames, f' ({low:.1f}–{high:.1f} s)'

    def compute_octave(self) -> None:
        """A spectrum on proportional bands, added beside it — the
        Octave Bands panel's own act, making exactly the steps its
        preview draws with the spacing set beside them."""
        # the spacing is the panel's — the very steps being previewed
        # are what the button makes, which is the whole contract
        per_octave = self.data_pane.octave_panel.per_octave()
        acted = self._act_on(
            lambda o: isinstance(o, Psd) and not isinstance(o, Specification),
            'Select a PSD or CPSD to band',
            self.project.compute_octave, per_octave)
        if acted is None:
            return
        name, obj, added = acted
        banded = self.objects[added]
        self.show_object(added)
        self._show_status(
            f'{added}: {len(banded.abscissa)} bands from '
            f'{len(obj.abscissa)} lines, {banded.num_records} '
            f'record{"s" * (banded.num_records != 1)} — linked to {name}')

    def compute_frfs(self) -> None:
        """FRFs from the selected time history, added beside it.

        Asks which estimator first. The three differ in where they
        assume the noise is — on the response, on the reference, or on
        both — and that is not something the data can say; it is a
        statement about the instrumentation, which the person who ran
        the test is the one holding.

        The drives are the history's own excitation channels and the
        frames are the ones a PSD would use, so the FRFs, the PSDs and
        the multiple coherence beside them all describe one measurement.
        """
        obj = self.current_object()
        if not isinstance(obj, TimeHistory):
            self._show_status('Select a time history to compute FRFs')
            return
        name = self.object_item().text(0)
        methods = list(TimeHistory.FRF_METHODS)
        labels = [f'{method} — {note}' for method, note in zip(methods, (
            'noise on both (total least squares)',
            'noise on the response',
            'noise on the reference (single reference only)'))]
        chosen, ok = QInputDialog.getItem(
            self, 'Compute FRFs', 'Estimator:', labels, 0, False)
        if not ok:
            return
        method = methods[labels.index(chosen)]
        seeded = obj.averaging is None
        try:
            added = self.project.compute_frfs(name, method)
        except ValueError as refusal:
            self._show_status(f'{name}: {refusal}')
            return
        frfs = self.objects[added]
        drives = len(set(frfs.reference_dof))
        self.show_object(added)
        self._show_status(
            f'{added}: {frfs.num_records} {method} FRFs, '
            f'{frfs.num_records // max(drives, 1)} responses against '
            f'{drives} reference{"s" * (drives != 1)}, '
            f'{obj.averaging.frames * max(obj.records_per_channel.values())} '
            f'averages{" (detected)" if seeded else ""} — linked to {name}')

    def compute_multiple_coherence(self) -> None:
        """Multiple coherence from the selected time history.

        Over the same frames a PSD would be averaged from, so the two
        describe one measurement — the averaging view sets both.

        With no averaging set at all, `Project.compute_multiple_coherence`
        works one out and *stores it on the history*, the way the shock
        calculator stores the events it detects. Storing it is the
        point: a coherence that quietly averaged differently from the
        PSD beside it would describe a different measurement, and the
        whole reason it follows the averaging is so that it does not.
        This says which of the two happened.
        """
        obj = self.current_object()
        if not isinstance(obj, TimeHistory):
            self._show_status(
                'Select a time history to compute multiple coherence')
            return
        name = self.object_item().text(0)
        seeded = obj.averaging is None
        try:
            added = self.project.compute_multiple_coherence(name)
        except ValueError as refusal:
            self._show_status(f'{name}: {refusal}')
            return
        coherence = self.objects[added]
        drives = len(obj.drive_dofs())
        self.show_object(added)
        self._show_status(
            f'{added}: {coherence.num_records} '
            f'response{"s" * (coherence.num_records != 1)} against '
            f'{drives} reference{"s" * (drives != 1)}, '
            f'{obj.averaging.frames} averages'
            f'{" (detected)" if seeded else ""} — linked to {name}')

    def compute_srs(self) -> None:
        """Shock response spectra from the selected time history — one
        curve per channel per shock, added beside it.

        The shocks come from the history itself, the way averaging does:
        whatever the shock view has on it at the time. With none,
        `Project.compute_srs` runs the detector first, so the calculator
        answers rather than asking the user to go and find the events by
        hand.
        """
        settings = self.data_pane.shock_panel.srs_settings()
        acted = self._act_on(
            TimeHistory,
            'Select a time history to compute an SRS',
            self.project.compute_srs, **settings)
        if acted is None:
            return
        name, obj, added = acted
        spectra = self.objects[added]
        # what the windows turned out to be, said in the terms they came
        # from: detected events, the frames the record is read as, or the
        # single playing a target is. `shocks` is None where nothing was
        # detected, which is now the ordinary case rather than an error
        events = len(obj.shocks or ())
        if events:
            over = f'{events} shock{"s" * (events != 1)}'
        elif obj.averaging is not None:
            over = (f'{obj.averaging.frames} '
                    f'frame{"s" * (obj.averaging.frames != 1)}')
        else:
            over = 'the whole record'
        self.show_object(added)
        self._show_status(
            f'{added}: {spectra.num_records} spectra over {over} '
            f'at Q={spectra.q:g} — linked to {name}')

    def filter_data(self) -> None:
        """The selected time history through its filter, every
        channel, added beside it. The settings are the history's own;
        with none set, the suggestion is adopted — same contract as
        `Project.filter_data`."""
        acted = self._act_on(
            TimeHistory,
            'Select a time history to filter',
            self.project.filter_data)
        if acted is None:
            return
        name, obj, added = acted
        filtering = obj.filtering
        self.show_object(added)
        self._show_status(
            f'{added}: {filtering.describe()}, order '
            f'{filtering.order}, zero phase — linked to {name}')

    def truncate_data(self) -> None:
        """The selected time history cut to its span, every channel,
        added beside it. The span is the history's own; with none
        set, this refuses and says where to set one — keeping the
        whole record is not an act, so there is no suggestion to
        adopt the way Filter Data adopts one."""
        acted = self._act_on(
            TimeHistory,
            'Select a time history to truncate',
            self.project.truncate_data)
        if acted is None:
            return
        name, obj, added = acted
        truncation = obj.truncation
        cut = self.objects[added]
        self.show_object(added)
        self._show_status(
            f'{added}: {truncation.describe()} — '
            f'{cut.ordinate.shape[-1]} of {obj.ordinate.shape[-1]} '
            f'samples, linked to {name}')

    def generate_rigid_body_modes(self) -> None:
        """The six rigid-body mode shapes of the selected geometry,
        about the point the rigid-body pane shows, added to its group
        — the same verb a script calls, so provenance and the badge
        follow."""
        acted = self._act_on(
            Geometry,
            'Select a geometry to make rigid body ' 'modes of',
            self.project.generate_rigid_body_modes)
        if acted is None:
            return
        name, obj, added = acted
        properties = obj.mass_properties
        self.show_object(added)
        self._show_status(
            f'{added}: 6 modes {self._describe_rigid(obj, properties)}, '
            f'linked to {name}')

    def integrate_history(self) -> None:
        """One integration of the selected time history — acceleration
        to velocity, velocity to displacement — added beside it."""
        acted = self._act_on(
            TimeHistory,
            'Select a time history to integrate',
            self.project.integrate)
        if acted is None:
            return
        name, obj, added = acted
        result = self.objects[added]
        left_out = obj.num_records - result.num_records
        aside = (f' — {left_out} record{"s" * (left_out != 1)} of other '
                 'quantities left out' if left_out else '')
        self._show_status(
            f'{added}: {result.num_records} channel'
            f'{"s" * (result.num_records != 1)} integrated, drift '
            f'high-passed{aside} — linked to {name}')
        self.show_object(added)

    def differentiate_history(self) -> None:
        """One differentiation of the selected time history —
        displacement to velocity, velocity to acceleration — added
        beside it."""
        acted = self._act_on(
            TimeHistory,
            'Select a time history to differentiate',
            self.project.differentiate)
        if acted is None:
            return
        name, _obj, added = acted
        result = self.objects[added]
        self._show_status(
            f'{added}: {result.num_records} channel'
            f'{"s" * (result.num_records != 1)} differentiated — '
            f'linked to {name}')
        self.show_object(added)

    def compute_cpsds(self) -> None:
        """The full cross-spectral matrix from the selected time
        history — every channel against every channel — added beside
        it."""
        acted = self._act_on(
            TimeHistory,
            'Select a time history to compute CPSDs',
            self.project.compute_cpsds)
        if acted is None:
            return
        name, obj, added = acted
        cpsds = self.objects[added]
        channels = round(cpsds.num_records ** 0.5)
        frames, where = self._averaged_frames(obj, channels)
        # shown first: selecting it renders, and the render restates the
        # status line over anything said before it
        self.show_object(added)
        self._show_status(
            f'{added}: {channels}x{channels} cross-spectral matrix '
            f'averaged over {frames} frame{"s" * (frames != 1)}{where} — '
            f'linked to {name}')

    def _rename_photo(self, name, index):
        """Ask for a photo's new name, then set it.

        The dialog is what F2 opens, and it is the *second* route: the
        first is double-clicking the name in the grid, which edits it in
        place. Both end in `_apply_photo_name`, so they cannot come to
        different conclusions about what a name may be.
        """
        photos = self.objects.get(name)
        if not isinstance(photos, Photos) \
                or not 0 <= index < photos.num_photos:
            return
        new, ok = QInputDialog.getText(self, 'Rename Photo', 'Photo name:',
                                       text=photos.names[index])
        if ok:
            self._apply_photo_name(name, index, (new or '').strip())

    def _grid_row_renamed(self, grid, row, text):
        """A grid's row label typed over: a photo's name, or the
        coordinate of a record's or a channel's row — that channel, the
        row's coordinate and quantity, not every channel at the point."""
        if grid.kind == 'photo':
            self._apply_photo_name(grid.owner, row, text)
        else:
            key = grid.row_keys[row]
            self._apply_dof_rename(grid.owner, key.dof, key.quantity, text)

    def _apply_dof_rename(self, name, old, quantity, new):
        """Correct a channel's coordinate on `name` and on what was
        derived from it, through the project's verb, and redraw
        whatever shows them."""
        if name not in self.objects or not new or new == old:
            return
        try:
            changed = self.project.rename_dof(name, old, new, quantity)
        except (ValueError, TypeError) as refusal:
            self._show_status(f'{name}: {refusal}')
            return
        for each in changed:
            item = self._item_for_object(each)
            if item is None:
                continue
            grid = self.record_grids.get(each)
            picked = grid.selected_records() if grid is not None else []
            self._refresh_item(item, self.objects[each])
            grid = self.record_grids.get(each)
            if picked and grid is not None:
                grid.select_records(picked)
        self.render_current()
        self._show_status(
            f'{old} ({quantity}) is now {new} on {", ".join(changed)}')

    def _apply_photo_name(self, name, index, new):
        """Rename one photo; report blocks bound to it follow along."""
        photos = self.objects.get(name)
        if not isinstance(photos, Photos) \
                or not 0 <= index < photos.num_photos:
            return
        old = photos.names[index]
        if not new or new == old:
            return
        try:
            photos.rename(index, new)
        except ValueError as refusal:
            self._show_status(str(refusal))
            return
        self.project.record_call(photos, 'rename', int(index), new)
        for obj in self.objects.values():
            if isinstance(obj, Report):
                for block in obj.blocks:
                    if (block.get('kind') == 'photo'
                            and block.get('source') == name
                            and block.get('photo') == old):
                        block['photo'] = new
        picked = self.record_grids[name].selected_records()
        self._refresh_item(self._item_for_object(name), photos)
        self.record_grids[name].select_records(picked)
        if self.report_editor is not None and self.report_editor.isVisible():
            self.report_editor.rebuild()
        self._show_status(f"Renamed photo '{old}' to '{new}'")

    def _render_photos(self, picture_sets):
        """Every photo of the selected Photos objects, stacked — the
        same drawing a script gets from `photos.plot()`."""
        from ..plot import build_photos

        self.data_pane.clear()
        shown = 0
        for _name, photos, picks in picture_sets:
            shown += build_photos(self.data_pane.graphics, photos, picks,
                                  start_row=shown)
        return f'{shown} photo{"s" * (shown != 1)}'

    def _synthesis_series(self, shapes, series):
        """(overlay series, status note): the modal model's prediction.

        A shape set selected beside measured FRFs re-synthesizes them,
        record for record — `synthesize_overlay` does the work, and the
        fitting screen's own resynthesis calls it too.
        """
        if not shapes or not series or not all(
                isinstance(data, Frf) for _name, data, _records in series):
            return [], ''
        name, shape_set = shapes[0][0], shapes[0][1]
        modes = sorted({detail for n, _s, detail in shapes
                        if n == name and detail is not None}) or None
        out, skipped = [], 0
        for series_name, data, records in series:
            rows = (list(records) if records is not None
                    else list(range(data.num_records)))
            overlay = synthesize_overlay(shape_set, data, records=rows,
                                         modes=modes)
            if overlay is None:
                skipped += len(rows)
                continue
            skipped += len(rows) - overlay.num_records
            out.append((f'{series_name} synthesized', overlay, None))
        if not out:
            return [], ''
        count = len(modes) if modes is not None else shape_set.num_shapes
        note = (f'synthesized from {count} mode{"s" * (count != 1)} '
                f'of {name}')
        if skipped:
            note += f' ({skipped} records outside the shape DOFs)'
        return out, note

    def _set_pair_mode(self, mode):
        self.pair_mode = mode
        if mode == 'overlay':
            self.stop_fitting()
            self.render_current()
        elif self.fit is None:
            # entered from the button, whatever the selection's sub-picks:
            # Edit means Edit
            self.start_modal_fit()

    def _render_report_builder(self, name, report):
        """The report itself, editable: a browser view of exactly what
        the export will be, its acts on the bar above and the selected
        block's settings in the pane beside (2026-09-08)."""
        from .report_editor import ReportEditor

        if self.report_editor is None:
            self.report_editor = ReportEditor()
            self.report_editor.export_requested.connect(self.export_report)
            self.report_editor.template_requested.connect(
                self.export_report_template)
            self.report_editor.edited.connect(self._journal_report_edit)
            self._table_layout.insertWidget(1, self.report_editor, 1)
        # the report has the pane to itself. Hiding the table and the
        # MAC is not enough: the splitter holding them keeps its share of
        # the height, which left the report in the top half, and the bar
        # above them carries a mode-table toggle that means nothing here
        self.tables_row.hide()
        self.table.setVisible(False)
        self._hide_mac()
        self.table_bar.hide()
        if self.report_editor.report is not report:
            self.report_editor.show_report(report, lambda: self.objects,
                                           self.unit_system,
                                           links=lambda: self.links)
        elif self.report_editor.unit_system is not self.unit_system:
            # the display system changed underneath the open report —
            # it shows display units, so it re-renders in the new ones
            self.report_editor.unit_system = self.unit_system
            self.report_editor.rebuild()
        elif self.report_editor.stale:
            # the project changed while the page was off screen
            self.report_editor.rebuild()
        self.report_editor.show()
        unbound = len(report.unbound(self.objects, self.links))
        note = (f'; {unbound} block{"s" * (unbound != 1)} unbound — pick '
                'a source' if unbound else '')
        return (f'{name}: {report.num_blocks} '
                f'block{"s" * (report.num_blocks != 1)}{note}')

    def generate_report(self, template: str = 'empty') -> str:
        """A new Report object from a starter template, opened to edit.

        Templates bind symbolically — '@basis:Frf' and friends,
        resolved against the link groups at render time — so the
        report depends on the project's structure, never on what
        anyone named their objects.
        """
        name = self.show_object(self.project.generate_report(
            template, name=display_name('Report')))
        self._show_status(
            f'{name}: click a block in the page, edit it in the pane '
            'beside; Insert, Move, Delete and Export are on the bar')
        return name

    def export_report_template(self) -> None:
        """Write the selected report on its own, as a template another
        project can load — into the templates folder by default, which
        is where Generate Report looks."""
        import pathlib

        from ..io.report_template import SUFFIX, templates_folder

        obj = self.current_object()
        if not isinstance(obj, Report):
            self._show_status('Select a report to save as a template')
            return
        name = self.object_item().text(0)
        folder = templates_folder()
        folder.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Report Template', str(folder / f'{name}{SUFFIX}'),
            f'Report template (*{SUFFIX})')
        if not path:
            return
        if not path.endswith(SUFFIX):
            path += SUFFIX
        self.project.export(name, path)
        where = (' — Generate Report offers it'
                 if pathlib.Path(path).parent == folder else '')
        self._show_status(f'Saved template {os.path.basename(path)}{where}')

    def export_report(self) -> None:
        """Write the selected report as one self-contained HTML file."""
        from ..report import render_html

        obj = self.current_object()
        if not isinstance(obj, Report):
            self._show_status('Select a report to export')
            return
        name = self.object_item().text(0)
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Report', f'{name}.html', 'Report (*.html)')
        if not path:
            return
        if not path.endswith('.html'):
            path += '.html'
        skipped = len(obj.unbound(self.objects))
        with open(path, 'w', encoding='utf-8') as out:
            out.write(render_html(obj, self.objects, self.unit_system,
                                  links=self.links))
        note = (f' ({skipped} unbound block'
                f'{"s" * (skipped != 1)} left out)') if skipped else ''
        self._show_status(
            f'Exported {os.path.basename(path)}{note} — opens in any '
            'browser, nothing to install')

    def _merge_candidates(self):
        """[(name, object)] when the selection is a legitimate merge —
        two or more whole objects of one type that `mergeable` accepts —
        else None, and the menu simply does not offer it."""
        from ..core.merge import mergeable

        references = self.selected_references()
        if not references or not all(
                kind == 'object' for kind, *_rest in references):
            return None
        chosen = list({name: obj for _kind, name, obj, _detail
                       in references}.items())
        if len(chosen) < 2:
            return None
        if mergeable([obj for _name, obj in chosen]) is not None:
            return None
        return chosen

    def merge_selected(self) -> None:
        """Replace the selected objects with their combination."""
        chosen = self._merge_candidates()
        if chosen is None:
            return
        names = [name for name, _obj in chosen]
        self.tree.blockSignals(True)
        try:
            for name in names:
                self._forget_object_row(name)
            merged_name = self.show_object(self.project.merge(
                *names, name=display_name(
                    type(self.objects[names[0]]).__name__)))
        finally:
            self.tree.blockSignals(False)
        self.refresh_compatibility()
        self.render_current()
        what = ', '.join(name for name, _obj in chosen)
        self._show_status(f'Merged {what} into {merged_name}')

    def start_modal_fit(self) -> None:
        """Fit real normal modes to the selected FRFs, one mode at a time.

        The measured CMIF stays solid while the fit's synthesis climbs
        onto it dashed. A cursor sits on the largest residual peak, and
        the frequencies on screen are what narrow the search — zoom to
        say where to look, and the cursor rides the zoom rather than
        being left off-screen; the table beside the plot lists the
        fitted modes plus the pending one; Find Mode previews a fit at the
        cursor, Confirm Mode adopts it and moves to the next residual
        peak — the natural next mode, since a confirmed peak collapses.
        A co-selected shape set seeds the session and receives every
        change in place.
        """
        # a fresh session, and the previous one's drag budget is no part
        # of it
        self._fit_dear_cost = 0.0
        self._fit_synthesis_stale = True
        # the fitting screen is the flat plot — the CMIF, the cursor and
        # the corner range fields all live on it. Said here as well as
        # in render_current because the toolbar's Fit button reaches
        # this directly, and an FRF showing as a waterfall would
        # otherwise host the whole fit on a hidden surface
        self.data_pane.show_waterfall(False)
        selected = self.selected_references()
        frfs = {name: obj for _kind, name, obj, _detail in selected
                    if isinstance(obj, Frf)}
        shape_sets = {name: obj for _kind, name, obj, _detail
                          in selected if isinstance(obj, ShapeSet)}
        if len(frfs) != 1:
            self._show_status('Select one FRF object to fit modes to')
            return
        name, obj = next(iter(frfs.items()))
        records = sorted({d for k, n, _o, d in selected
                          if n == name and k == 'record'}) or None
        self.stop_editing()
        self.close_units_panel()
        self.fit = ModalFitSession(obj, records)
        # the measurement's own statement of which channel to believe
        # where: with a coherence in the project covering these
        # responses, Refine All weights its solve and its judgment by
        # it — a channel the coherence distrusts cannot vote noise into
        # every mode's residues
        self._fit_coherence_name = None
        coherence = self._fit_coherence(name, self.fit.responses)
        if coherence is not None:
            self.fit.weights = self.fit._coherence_weights(
                self.objects[coherence])
            if self.fit.weights is not None:
                self._fit_coherence_name = coherence
        self.fit_name = name
        self.fit_object_name = None
        if len(shape_sets) == 1:
            # a co-selected shape set is the fit to edit: its modes seed
            # the session, and every change publishes back to it in place
            seed_name, seed = next(iter(shape_sets.items()))
            self.fit.adopt(seed)
            self.fit_object_name = seed_name
        # stacked: the CMIF wants the full width up top, with the mode
        # table bottom-left and the MAC bottom-right under it — side by
        # side it was tall and skinny
        self.views.setOrientation(Qt.Orientation.Vertical)
        self._show_views(plots=True, table=True)
        self.fit_bar.show()
        # the plot bar over the CMIF offers the one overlay the fit
        # has: the residual. Show/hide toggles live there, with their
        # kin, rather than among the fit's verbs.
        self.data_pane.show_controls(
            map_wanted=None, diagonal=None, cmif=False,
            complex_data=False, pair=False, residual=True)
        self._fit_model = modal_fit_table_model(self.fit, self)
        self._set_table_model(self._fit_model)
        self._fit_model.dataChanged.connect(self._fit_edited)
        # a fitted mode can be taken back: select its row, hit Delete
        self.table.rows_deletable = True
        self._rows_wired = True
        self.table.rows_deleted.connect(self._delete_fit_modes)
        self._render_fit()

    def _render_fit(self):
        """The measured CMIF with the fit's synthesis dashed over it.

        The measurement is the solid, stable reference; Find Mode and
        Confirm redraw the dashed synthesis climbing onto its peaks. A
        redraw keeps the axis ranges the user has set — rebuilding the
        plot must not reset their zoom."""
        session = self.fit

        source_frf = self._fit_source_frf

        import pyqtgraph as pg
        previous = None
        if self._fit_cursor is not None:
            old_plot = next((item for item in self.data_pane.graphics.ci.items
                             if hasattr(item, 'viewRange')), None)
            if old_plot is not None:
                previous = old_plot.viewRange()
        measured = source_frf(np.stack([session.frf.ordinate[i]
                                        for i in session.rows]))
        series = [(self.fit_name, measured, None)]
        if session.modes or session.preview is not None:
            synthesis = source_frf(session.synthesis_records())
            synthesis.synthesized = True
            series.append((f'{self.fit_name} synthesized', synthesis, None))
        build_cmif(self.data_pane.graphics, series,
                   unit_system=self.unit_system, theme=self.theme_name)
        self.data_pane.show_pair_controls(self._pair_selected)
        plot = next(item for item in self.data_pane.graphics.ci.items
                    if hasattr(item, 'addItem'))
        # the dashed synthesis curves, so a dragged cursor can move them
        # without rebuilding the plot around them
        self._fit_synthesis = list(getattr(plot, 'synthesized_curves', ()))
        # kept, because what is *on screen* narrows the mode search —
        # see `_fit_window`
        self._fit_plot = plot
        if previous is not None:
            plot.setRange(xRange=previous[0], yRange=previous[1], padding=0)
        # The cursor may go anywhere in the record; what narrows the
        # search is the view, and the cursor rides it (`_fit_view_moved`).
        # There used to be a draggable band here as well, shading the
        # rest of the plot out, and it earned nothing — the view already
        # said where to look, and the band made you say it twice.
        #
        # Three pixels wide: at one it was indistinguishable from a grid
        # line, and this is a thing to grab.
        if self._fit_range_edits:
            for edit in self._fit_range_edits:
                edit.deleteLater()
            self._fit_range_edits = None
        cursor = _damping_cursor_class()(
            pos=session.pending['frequency'], angle=90, movable=True,
            pen=pg.mkPen('#e5534b', width=3), bounds=session.span)
        cursor.dragged_vertically = self._fit_cursor_dragged
        plot.addItem(cursor)
        cursor.sigPositionChanged.connect(self._fit_cursor_moved)
        self._fit_cursor = cursor
        # The SDOF curve the pending (frequency, damping) claims — the
        # damping made visible, so a vertical drag can be aimed. In the
        # cursor's own red, dashed, and *on top of* the CMIF curves: it
        # hugs the residual near the peak, and drawn beneath them it
        # was simply not visible — reported as absent.
        self._fit_parabola = pg.PlotDataItem(
            pen=pg.mkPen('#e5534b', width=2.0,
                         style=Qt.PenStyle.DashLine))
        self._fit_parabola.setZValue(40)
        plot.addItem(self._fit_parabola, ignoreBounds=True)
        # the number the drag is setting, riding the crown — hung
        # *below* it, because the crown touches the CMIF at the cursor
        # and the cursor lives on peaks: above the crown is off the top
        # of the plot exactly when the label matters
        self._fit_damping_label = pg.TextItem(
            color='#e5534b', anchor=(-0.15, 0.0))
        self._fit_damping_label.setZValue(41)
        plot.addItem(self._fit_damping_label, ignoreBounds=True)
        self._move_fit_parabola()
        self._build_fit_range_edits(plot)
        plot.getViewBox().sigXRangeChanged.connect(self._fit_view_moved)
        # each confirmed mode leaves a labeled marker where it was
        # fitted — quiet gray, half-transparent: bookmarks under the
        # curves, not more curves.
        add_mode_markers(plot,
                         [mode['frequency'] for mode in session.modes])
        self._draw_fit_residual(plot)
        self._render_fit_mac()
        self._fit_status()

    def _fit_status(self):
        session = self.fit
        fitted = len(session.modes)
        # two modes are the first that can influence each other
        self.refine_modes_button.setVisible(fitted >= 2)
        state = ('fitted, Confirm Mode to add'
                 if session.preview is not None else 'drag to fit')
        # the damping is on the line because the cursor now sets it:
        # drag up for sharper, down for wider — and 'held' says the
        # automatic estimate has let go until the next Find Mode
        damping = (f"{session.pending['damping'] * 100.0:.3g} % damping"
                   + (' (held — Find Mode releases)'
                      if session.pending['overridden'] else ''))
        weighted = (f' (weighted by {self._fit_coherence_name})'
                    if self._fit_coherence_name else '')
        self._show_status(
            f'Fitting {self.fit_name}{weighted}: cursor at '
            f"{session.pending['frequency']:.2f} Hz, {damping} — {state} "
            f'({fitted} mode{"s" * (fitted != 1)} confirmed)')

    def _fit_view_moved(self):
        """Zooming carries the cursor with it, rather than leaving it
        behind off-screen.

        The view is what narrows the mode search, so a cursor outside it
        is a cursor pointing at somewhere the search will not go — and
        the way that used to be fixed was to zoom out, drag the cursor
        across, and zoom back in. Now the zoom does it.

        Only when it is actually outside: a zoom that still contains the
        cursor leaves it exactly where the user put it, which matters
        because most zooms are made *around* the mode being worked on.
        Held to the edge rather than re-suggested, for the same reason —
        moving it to the tallest peak in the new view would be deciding
        something the user has a button for.
        """
        cursor, window = self._fit_cursor, self._fit_window()
        if self.fit is None or cursor is None or window is None:
            return
        low, high = self.fit.searched(window)
        clamped = min(max(cursor.value(), low), high)
        if clamped != cursor.value():
            cursor.setValue(clamped)   # emits, so the fit follows

    #: how long a drag tick may take before the damping and the dashed
    #: synthesis stop following the cursor. A drag that stutters is the
    #: worst kind of regression, because it is felt rather than
    #: measured — so this watches itself and gives up rather than
    #: dropping frames on a set of FRFs bigger than any measured here.
    #:
    #: Twelve milliseconds against a frame's sixteen-point-seven. The
    #: airplane's 1356 FRFs cost about ten and a half, three quarters
    #: of which is the synthesis CMIF — an SVD at every frequency — so
    #: the margin here is thin by measurement rather than by guess, and
    #: a set much larger will fall back rather than stutter.
    FIT_DRAG_BUDGET = 0.012

    #: how long the cursor must be still before the MAC is rebuilt
    #: against it. Long enough that crossing a peak on the way
    #: somewhere else does not redraw it, short enough to feel like an
    #: answer rather than a wait.
    FIT_SETTLE_MS = 220

    #: the damping the plot's top and bottom edges mean by default: the
    #: mapping is absolute — the mouse's height *is* the damping — so
    #: every value a hand might want is inside the window, log-spaced
    #: because damping is log-natured. 0.01 % brushes the sharpest
    #: structure worth fitting; past 10 % the half-power band is most of
    #: an octave and peak-fitting has stopped meaning much. The corner
    #: fields on the fit plot re-tune the span for the article at hand.
    FIT_DAMPING_TOP = 1e-4
    FIT_DAMPING_BOTTOM = 0.10

    def _fit_cursor_dragged(self, scene_y: float) -> None:
        """The cursor was dragged vertically: the damping is in hand.

        Up is sharper. The plot height is the whole range, absolutely:
        the top edge is 0.01 % and the bottom edge 10 %, log-linear in
        between, so where the mouse sits says what the damping is —
        no relative creep, and nothing a hand might want is out of
        reach of the window.

        This is the second axis of the search put in the user's hand:
        where two solutions sit at nearly the same frequency they are
        told apart by damping, and holding it says which neighborhood
        the fit is meant to land in. Find Mode lets go again.
        """
        if self.fit is None or self._fit_plot is None:
            return
        rect = self._fit_plot.getViewBox().sceneBoundingRect()
        if rect.height() <= 0:
            return
        fraction = float(np.clip(
            (scene_y - rect.top()) / rect.height(), 0.0, 1.0))
        top, bottom = self._fit_damping_range
        self.fit.override_damping(top * (bottom / top) ** fraction)
        self._move_fit_parabola()
        self._fit_synthesis_stale = True
        self._fit_settle.start(self.FIT_SETTLE_MS)
        row = len(self.fit.modes)
        self._fit_updating = True
        try:
            self._fit_model.dataChanged.emit(
                self._fit_model.index(row, 1), self._fit_model.index(row, 2))
        finally:
            self._fit_updating = False
        self._fit_status()

    def _move_fit_parabola(self):
        """Redraw the SDOF curve under the cursor — closed form, no SVD,
        cheap enough for every tick of a drag — and the damping figure
        at its crown, since the number is what the drag is setting."""
        if self.fit is None or self._fit_parabola is None:
            return
        frequencies, values = self.fit.pending_curve(self.unit_system)
        self._fit_parabola.setData(frequencies, values)
        if self._fit_damping_label is not None:
            crown = int(np.argmax(values))
            self._fit_damping_label.setText(
                f"{self.fit.pending['damping'] * 100.0:.3g} %")
            # the label lives in view coordinates, and the axis is log:
            # position it in log units or it leaves the screen
            self._fit_damping_label.setPos(
                float(frequencies[crown]),
                float(np.log10(max(values[crown], 1e-300))))

    def _fit_cursor_moved(self):
        """The pending row and the status line follow the cursor, and
        so does the damping.

        It used not to. The damping estimate meant a full SVD of the
        residual every tick, so dragging deliberately did nothing and
        you learned the damping only on pressing Fit Mode — dragging
        blind, and finding out afterwards. The residual does not depend
        on the cursor; cached against the modes instead, a tick is
        about a millisecond of a sixteen-millisecond frame.

        Still guarded two ways. Qt sends moves faster than they can be
        served, so a tick that arrives while one is running is dropped
        rather than queued — the cursor's position is read when the
        work starts, so the one that runs is always the latest. And if
        a tick ever overruns `FIT_DRAG_BUDGET`, the damping stops
        following until the drag ends.
        """
        if self.fit is None or self._fit_cursor is None:
            return
        if self._fit_dragging:
            return
        self._fit_dragging = True
        try:
            # Two tiers, so the cheap half never waits behind the dear
            # half. The parabola is closed form — no SVD, no search —
            # and follows *every* tick whatever the data's size; it and
            # the cursor are the part of the view that must feel wired
            # to the hand. The damping estimate, the fit and the dashed
            # synthesis are the dear tier: budget-guarded, given up for
            # the drag the first time they overrun, and caught up by
            # the settle when the cursor rests. On a small set nobody
            # can tell the tiers apart; on a huge one the parabola
            # stays instant and the synthesis arrives when it can.
            self.fit.move_to(self._fit_cursor.value(), damping=False)
            self._move_fit_parabola()
            # the gate reads the remembered cost *before* paying — the
            # dear tier's last measured price, primed by the settle at
            # screen open — so a set too big to follow live never
            # stalls a tick to find that out
            live = self._fit_dear_cost <= self.FIT_DRAG_BUDGET
            if live:
                start = time.perf_counter()
                # the automatic damping estimate, unless a vertical
                # drag is holding it — then the parabola above already
                # told the whole truth
                self.fit.move_to(self._fit_cursor.value())
                self._move_fit_parabola()
                # the shape too, and the dashed curve that shows it —
                # so the synthesis follows the cursor and snaps onto a
                # peak as it crosses one, which is the thing the whole
                # search was for
                self.fit.fit_pending()
                self._move_fit_synthesis()
                self._fit_dear_cost = time.perf_counter() - start
            self._fit_synthesis_stale = not live
        finally:
            self._fit_dragging = False
        # and once the cursor has been still a moment, the rest
        self._fit_settle.start(self.FIT_SETTLE_MS)
        row = len(self.fit.modes)
        self._fit_updating = True
        try:
            self._fit_model.dataChanged.emit(
                self._fit_model.index(row, 1), self._fit_model.index(row, 2))
        finally:
            self._fit_updating = False
        self._fit_status()

    def _build_fit_range_edits(self, plot) -> None:
        """The damping the plot's edges mean, editable where they live.

        Two small fields pinned to the plot's right corners: the top
        one is the damping at the top edge, the bottom one at the
        bottom edge — each sits at the height it governs, which is the
        whole reason they are on the plot rather than on the fit bar.
        Children of the graphics widget, placed by hand from the
        viewbox's rect, because pyqtgraph has no editable text item.
        """
        surface = self.data_pane.graphics
        edits = []
        tips = (('Damping at the top of the plot, in % — drag the '
                 'cursor up to reach it'),
                ('Damping at the bottom of the plot, in % — drag the '
                 'cursor down to reach it'))
        for which, tip in enumerate(tips):
            edit = QLineEdit(surface)
            edit.setFixedWidth(64)
            edit.setAlignment(Qt.AlignmentFlag.AlignRight)
            edit.setToolTip(tip)
            edit.setText(f'{self._fit_damping_range[which] * 100.0:g}')
            edit.editingFinished.connect(
                lambda which=which: self._fit_range_edited(which))
            edit.show()
            edits.append(edit)
        self._fit_range_edits = edits
        viewbox = plot.getViewBox()
        viewbox.sigResized.connect(self._place_fit_range_edits)
        self._place_fit_range_edits()

    def _place_fit_range_edits(self, *_):
        """Pin the fields to the viewbox's right corners, inside it."""
        if not self._fit_range_edits or self._fit_plot is None:
            return
        surface = self.data_pane.graphics
        rect = self._fit_plot.getViewBox().sceneBoundingRect()
        corner = surface.mapFromScene(rect.topRight())
        top, bottom = self._fit_range_edits
        top.move(int(corner.x()) - top.width() - 6, int(corner.y()) + 4)
        corner = surface.mapFromScene(rect.bottomRight())
        bottom.move(int(corner.x()) - bottom.width() - 6,
                    int(corner.y()) - bottom.height() - 4)

    def _fit_range_edited(self, which: int) -> None:
        """One of the corner fields changed: adopt it, or refuse it.

        Refused, not repaired: an entry that inverts the span (top
        wetter than bottom) or leaves the believable range is put back
        to what it was, with the reason in the status bar — never
        silently rewritten to something the user did not type.
        """
        if not self._fit_range_edits:
            return
        edit = self._fit_range_edits[which]
        try:
            value = float(edit.text()) / 100.0
        except ValueError:
            value = None
        proposed = list(self._fit_damping_range)
        if value is not None:
            proposed[which] = value
        if (value is None or not 1e-6 <= value <= 0.5
                or proposed[0] >= proposed[1]):
            edit.setText(f'{self._fit_damping_range[which] * 100.0:g}')
            self._show_status(
                'Damping span must run sharper at the top than the '
                'bottom, between 0.0001 % and 50 %')
            return
        self._fit_damping_range = proposed
        self._show_status(
            f'Cursor damping span: {proposed[0] * 100.0:g} % at the top '
            f'of the plot to {proposed[1] * 100.0:g} % at the bottom')

    def _fit_coherence(self, frf_name, responses):
        """The name of the coherence to weight this fit by, or None —
        the one covering the most of the fit's response DOFs, a linked
        one winning ties."""
        from ..core.data import _CoherenceBase

        linked = set(self.linked_group(frf_name) or ())
        wanted = set(responses)
        best, best_key = None, (0, False)
        for name, obj in self.objects.items():
            if not isinstance(obj, _CoherenceBase):
                continue
            covered = len(set(obj.response_dof) & wanted)
            key = (covered, name in linked)
            if covered and key > best_key:
                best, best_key = name, key
        return best

    def refine_all_modes(self) -> None:
        """Re-fit every confirmed mode's residues together, poles held —
        see ModalFitSession.refine_residues for what this repairs."""
        if self.fit is None or len(self.fit.modes) < 2:
            return
        before = [mode['shape'].copy() for mode in self.fit.modes]
        count = self.fit.refine_residues()
        changed = any(not np.array_equal(early, mode['shape'])
                      for early, mode in zip(before, self.fit.modes))
        self._publish_fit()
        self._render_fit()
        # the CMIF may have chosen the sequential answer — see
        # refine_residues — and saying "re-fit" about an unchanged fit
        # would send the eye hunting for a difference that is not there
        self._show_status(
            f'Re-fit {count} modes\' shapes together — frequencies and '
            'dampings kept' if changed else
            'Refine changed nothing — the fits are already as '
            'consistent as the measured CMIF supports')

    def _toggle_fit_residual(self, _checked: bool = False) -> None:
        """Draw or drop the residual CMIF — a full re-render, which
        keeps the zoom and rebuilds every curve the toggle touches."""
        if self.fit is None:
            return
        self._render_fit()

    def _draw_fit_residual(self, plot):
        """The residual's own CMIF, gray and dashed under the rest.

        What is *left* after the confirmed modes are taken out — the
        curve Find Mode hunts and the parabola's crown rides. Usually
        it is legible from the synthesis climbing onto the measurement,
        which is why it is a toggle and not always on.
        """
        import pyqtgraph as pg

        from ..plot import cmif_curves

        session = self.fit
        if not self.data_pane.residual_action.isChecked():
            return
        records = session.residual_records()
        singular, x = cmif_curves(self._fit_source_frf(records), None,
                                  self.unit_system)
        floor = singular.max() * 1e-10
        pen = pg.mkPen('#8a8a92', width=1.0, style=Qt.PenStyle.DashLine)
        for k in range(singular.shape[0]):
            if singular[k].max() <= floor:
                continue
            curve = pg.PlotDataItem(x, singular[k], pen=pen,
                                    connect='finite')
            curve.setZValue(-5)         # context, beneath the evidence
            plot.addItem(curve, ignoreBounds=True)

    def _rerender_mac(self):
        """The 3D toggle redraws whichever MAC is up. During a fit,
        `render_current` returns early — the fit owns the panes — so the
        fit's own MAC renderer is called directly."""
        if self.fit is not None:
            self._render_fit_mac()
        else:
            self.render_current()

    def _mac_bars_plotter(self):
        """The 3-D MAC surface, built on first use.

        The same lazy deferral as the data pane's, and safe for the same
        reason: the only thing that asks for it is the toggle on the
        table bar, which cannot be clicked before the window is mapped.
        """
        if self._mac_bars_plotter_obj is None:
            if self._offscreen_3d:
                import pyvista as pv

                self._mac_bars_plotter_obj = pv.Plotter(off_screen=True)
                self._mac_bars_page = QLabel(
                    '3D view disabled (offscreen mode)')
            else:
                from pyvistaqt import QtInteractor

                from ..viz import undeferred

                self._mac_bars_plotter_obj = undeferred(QtInteractor(self))
                self._mac_bars_plotter_obj.setAcceptDrops(False)
                self._mac_bars_plotter_obj.enable_anti_aliasing('fxaa')
                self._mac_bars_page = self._mac_bars_plotter_obj
            self._mac_bars_page.hide()
            # a third splitter child beside the flat MAC's frame; the
            # two are never up together
            self.tables_row.addWidget(self._mac_bars_page)
            self._arm_mac_bars_picking()
        return self._mac_bars_plotter_obj

    def _mac_surface(self, bars):
        """Which MAC surface stands: the flat grid or the bars."""
        if bars:
            self.mac_frame.hide()
            self._mac_bars_page.show()
            self._share_tables_row(self._mac_bars_page)
        else:
            if self._mac_bars_page is not None:
                self._mac_bars_page.hide()
            self.mac_frame.show()
            self._share_tables_row(self.mac_frame)

    def _share_tables_row(self, widget):
        """Give a re-shown MAC surface a real share of the splitter.

        A splitter child keeps the width it had when it was hidden, and
        the two surfaces trade places while hidden — so the flat MAC
        could come back with the zero it was stored at, sit there
        `isVisible()` and 0 px wide, and read as "the 2-D toggle shows
        only the table" (Brandon, 2026-08-30). The threshold leaves a
        deliberately narrow drag alone; below it nothing can be read
        anyway — the MAC's left axis alone wants 66 px.
        """
        sizes = self.tables_row.sizes()
        at = self.tables_row.indexOf(widget)
        if at < 0 or sizes[at] >= 40:
            return
        total = sum(sizes) or self.tables_row.width()
        wanted = [0] * len(sizes)
        wanted[at] = total // 2
        others = [i for i in range(len(sizes)) if i != at
                  and not self.tables_row.widget(i).isHidden()]
        for i in others:
            wanted[i] = (total - wanted[at]) // len(others)
        self.tables_row.setSizes(wanted)

    def _hide_mac(self):
        """Both MAC surfaces away — one call for every path that used
        to hide the frame alone, so the bars cannot outlive the grid."""
        self.mac_frame.hide()
        if self._mac_bars_page is not None:
            self._mac_bars_page.hide()

    def _draw_mac_bars(self, frequencies, matrix, key,
                       column_frequencies=None, selected=(), active=None,
                       matched=()):
        """The MAC as bars, camera placed only when `key` changes —
        confirming a mode mid-fit or picking a cell redraws the scene
        under the view the user set. `selected`, `active` and `matched`
        are the comparison's marks, in the grid's own red."""
        from ..viz.mac_bars import add_mac_bars, place_camera

        plotter = self._mac_bars_plotter()
        plotter.clear()
        colors = resolve_theme(self.theme_name)
        plotter.set_background(colors['scene_background'],
                               top=colors['scene_background_top'])
        info = add_mac_bars(plotter, frequencies, matrix,
                            column_frequencies=column_frequencies,
                            theme=self.theme_name, selected=selected,
                            active=active, matched=matched)
        self._mac_bars_grid = (info['rows'], info['columns'])
        if self._mac_bars_shown != key:
            place_camera(plotter, info['rows'], info['columns'])
            self._mac_bars_shown = key
        plotter.render()
        self._mac_surface(bars=True)

    def _render_fit_mac(self):
        """The pending mode in the MAC against the ones already
        confirmed — is this a new mode, or one I already have?

        On its own, because it is the one part of the fit view worth
        rebuilding when the cursor stops rather than while it moves. A
        MAC redrawn sixty times a second is a flicker nobody can read,
        and it is the dearest thing here besides.
        """
        from ..plot import build_mac

        session = self.fit
        if session is None:
            self._hide_mac()
            return
        shown = list(session.modes) + (
            [session.preview] if session.preview is not None else [])
        if not shown:
            self._hide_mac()
            return
        self._mac_shown = None      # not a comparison; its zoom is gone
        frequencies = [mode['frequency'] for mode in shown]
        matrix = mac_matrix(np.array([mode['shape'] for mode in shown]))
        # the fit's MAC offers the 3-D reading too; the bar's comparison
        # controls stay away — this is one set against itself, and the
        # mode list is the fit's own table, not the toggle's business
        self.table_bar.show()
        for action in (self.mode_table_action, self.overlay_action,
                       self.add_matches_action):
            action.setVisible(False)
        self.mac_bars_action.setVisible(True)
        if self.mac_bars_action.isChecked():
            # the key carries the grid size: a cursor-stop redraw (same
            # size) keeps whatever view the user set, and confirming a
            # mode — which grows the grid — reframes for the new
            # extent. Keyed per session alone, the camera was placed
            # for the opening 1x1 and never again, so a grown MAC came
            # up out of frame with the view centered on its first corner
            # (Brandon, 2026-08-30).
            self._draw_mac_bars(frequencies, matrix,
                                key=('fit', self.fit_name, len(shown)))
            return
        build_mac(self.mac_view, frequencies, matrix,
                  theme=self.theme_name)
        self.mac_frame.set_ratio(1.0)
        self._mac_surface(bars=False)

    def _fit_settled(self):
        """The cursor stopped moving: catch up on what a drag skips.

        The MAC, too dear and too flickery to redraw while the cursor
        moves — and the fit itself when the drag gave it up, which is
        what keeps a set of FRFs too large to follow live still usable:
        it arrives when you stop rather than as you go.

        This is what removed the Fit Mode button. By the time anyone
        pressed it the mode was already fitted and drawn; what they
        were waiting for was this.
        """
        if self.fit is None:
            return
        if self._fit_synthesis_stale:
            start = time.perf_counter()
            self.fit.fit_pending()
            self._move_fit_synthesis()
            # the price just paid is the next drag's gate: a machine
            # that got faster — modes deleted, a smaller band — earns
            # the live path back by measurement, not by hope
            self._fit_dear_cost = time.perf_counter() - start
            self._fit_synthesis_stale = False
        self._render_fit_mac()
        self._fit_status()

    def _fit_source_frf(self, ordinate):
        """An FRF over the fit's own rows — what both the full redraw
        and the in-place update build their CMIF from."""
        session = self.fit
        return Frf(
            session.frf.abscissa, ordinate,
            response_dof=[session.frf.response_dof[i]
                          for i in session.rows],
            reference_dof=[session.frf.reference_dof[i]
                           for i in session.rows],
            ordinate_dim=[session.frf.ordinate_dim[i]
                          for i in session.rows],
            ordinate_unit=[session.frf.ordinate_unit[i]
                           for i in session.rows],
            reference_unit=[session.frf.reference_unit[i]
                            for i in session.rows],
            dimension_hint=[session.frf.dimension_hint[i]
                            for i in session.rows])

    def _move_fit_synthesis(self):
        """Move the dashed synthesis to the pending mode, in place.

        The measurement does not change while the cursor is dragged and
        the axes do not either, so rebuilding the CMIF for this — which
        is what pressing Fit Mode does — spends 34 ms of a 16.7 ms
        frame on redrawing 1356 records that did not move.

        Falls back to the full render whenever the curves it was given
        no longer match what is being drawn: a confirmed mode changes
        how many singular values are above the floor, and a curve count
        that has drifted is how a plot ends up showing a mode that is
        not there.
        """
        from ..core.data import UNKNOWN
        from ..plot import cmif_curves

        if self.fit is None or not self._fit_synthesis:
            return False
        if self.fit.preview is None and not self.fit.modes:
            return False
        session = self.fit
        dims = {session.frf.ordinate_dim[i] for i in session.rows}
        dim = dims.pop() if len(dims) == 1 else None
        if dim is not None and dim != UNKNOWN:
            # the factored path: same singular values from the QR of
            # the shape factors, no record array — 6.5 s to
            # milliseconds on the hard drone survey. One dimension
            # means one display factor, and a positive scalar moves
            # through an SVD untouched.
            singular = session.synthesis_singular_values()
            if singular is None:
                return False
            singular = singular * float(
                self.unit_system.from_si(1.0, dim))
            x = session.frf.display_abscissa(self.unit_system)
        else:
            # mixed dimensions have per-record factors, which do not
            # move through an SVD — the materialized path stands
            records = session.synthesis_records()
            synthesis = self._fit_source_frf(records)
            singular, x = cmif_curves(synthesis, None, self.unit_system)
        floor = singular.max() * 1e-10
        rows = [k for k in range(singular.shape[0])
                if singular[k].max() > floor]
        if len(rows) != len(self._fit_synthesis):
            return False
        for curve, k in zip(self._fit_synthesis, rows):
            curve.setData(x, singular[k], connect='finite')
        return True

    def _fit_window(self):
        """The frequency range on screen, or None before there is a plot.

        Zooming to a region is the plainest way to say where the next
        mode is, and having then to drag the band bars to match would be
        saying it twice. The bars stay the outer word — `searched`
        intersects the two — so a band dragged tighter than the view
        still wins where they disagree.
        """
        if self._fit_plot is None:
            return None
        return tuple(self._fit_plot.viewRange()[0])

    def find_next_mode(self) -> None:
        """Put the cursor on the next mode worth fitting.

        The largest CMIF peak left in the residual, inside what the plot
        is showing — which is what Confirm already moves to once a mode
        is taken. This is that on its own, for hunting before committing
        to anything.
        """
        if self.fit is None:
            return
        # A settle left armed by the gesture before this one would fire
        # a quarter-second from now and put the resting message back
        # over the one below — and the gesture before this one is
        # almost always the zoom that said where to look, since the
        # zoom carries the cursor. Canceled rather than raced: this
        # has already done the settle's work by rendering, and Find
        # Mode is a look rather than a drag catching up.
        self._fit_settle.stop()
        window = self._fit_window()
        low, high = self.fit.searched(window)
        frequency, damping = self.fit.suggest(window)
        self._fit_updating = True
        try:
            row = len(self.fit.modes)
            self._fit_model.dataChanged.emit(
                self._fit_model.index(row, 1), self._fit_model.index(row, 2))
        finally:
            self._fit_updating = False
        self._render_fit()
        self._show_status(
            f'Largest residual peak at {frequency:.2f} Hz, '
            f'{damping * 100:.2f}% — Confirm Mode to adopt it '
            f'(searched {low:.4g} to {high:.4g} Hz)')

    def fit_pending_mode(self) -> None:
        """Fit at the cursor without confirming, and restate everything.

        There is no button for this any more. Dragging fits as it goes
        and the MAC follows a moment after the cursor stops, so by the
        time anyone reached for Fit Mode the work was already done —
        the button was asking for something it had.

        It stays as a method because it is the whole view restated at
        once, which a caller that changed something out of band still
        wants: a damping typed into the table, or a script driving the
        fit with no cursor to drag.
        """
        if self.fit is None:
            return
        self.fit.fit_pending()
        self._fit_updating = True
        try:
            row = len(self.fit.modes)
            self._fit_model.dataChanged.emit(
                self._fit_model.index(row, 1), self._fit_model.index(row, 2))
        finally:
            self._fit_updating = False
        self._render_fit()

    def _fit_edited(self, *_args):
        """A damping or description typed by hand: restate the view,
        whose synthesis a damping change genuinely moves. Programmatic
        restatements of the pending row do not come through here."""
        if self.fit is not None and not getattr(self, '_fit_updating', False):
            self._render_fit()

    def confirm_fit_mode(self) -> None:
        if self.fit is None:
            return
        window = self._fit_window()
        mode = self.fit.confirm()
        self._publish_fit()
        # the same window Find Mode would have used: confirming a mode
        # and pressing Find Mode are the same hunt, and they should not
        # look in different places
        self.fit.suggest(window)
        self._fit_model.layoutChanged.emit()
        self._render_fit()
        note = ('' if mode.get('scaled', True) else
                ' (unscaled — no drive point in the FRFs)')
        self._show_status(
            f"Fitted mode at {mode['frequency']:.2f} Hz{note} — next "
            f"peak at {self.fit.pending['frequency']:.2f} Hz")

    def _delete_fit_modes(self, rows):
        """Take fitted modes back: the synthesis, the MAC and the
        published shapes all restate themselves. The pending row is not a
        mode and stays."""
        session = self.fit
        if session is None:
            return
        picked = sorted({int(row) for row in rows
                         if row < len(session.modes)}, reverse=True)
        if not picked:
            return
        for row in picked:
            pick = (float(session.modes[row]['frequency']),
                    float(session.modes[row]['damping']))
            if pick in session.confirmed:
                # the replay list follows: the journal's fit_modes call
                # holds what stands, not what was taken back
                session.confirmed.remove(pick)
            del session.modes[row]
        session.preview = None    # fitted against a residual that is gone
        self._publish_fit()
        self._fit_model.layoutChanged.emit()
        self._render_fit()
        self._show_status(
            f'Removed {len(picked)} mode{"s" * (len(picked) != 1)} '
            'from the fit')

    def _publish_fit(self):
        """The fitted modes live in the project from the first Confirm on.

        An empty fit publishes nothing: deleting the last mode takes the
        published object back too, since an empty shape set is not a
        thing anything else here can show."""
        prefix = f'project.fit_modes({self.fit_name!r}, at='
        if not self.fit.modes:
            if self.fit_object_name:
                self._remove_object(self.fit_object_name)
                self.refresh_compatibility()
                if self.project.journal and \
                        self.project.journal[-1].startswith(prefix):
                    # the fit this line replayed is gone whole
                    self.project.journal.pop()
            self.fit_object_name = None
            return
        shapes = self.fit.shape_set()
        if self.fit_object_name and self.fit_object_name in self.objects:
            self.objects[self.fit_object_name] = shapes
            self._refresh_item(self._item_for_object(self.fit_object_name),
                               shapes)
            # the same set, one more mode: the report draws the new one
            self._report_content_changed(settling=True)
        else:
            # add_object moves the selection to the new item, which would
            # read as leaving the fit; keep the FRF selected throughout.
            # The journal stays quiet through it — the fit_modes line
            # below is this stretch's honest record, and _derive links
            # the replay the way relink links this.
            self.tree.blockSignals(True)
            try:
                with self.project.journal_as(None):
                    self.fit_object_name = self.add_object(
                        f'{self.fit_name} Modes', shapes)
                    # Modes fitted from an FRF belong with it — not a
                    # guess. relink, not link: adding the object just ran
                    # the type's placement, which reads the project's
                    # *second* shape set as the FEM slot and parks the
                    # new fit in the model group. link would try to merge
                    # that whole group into the FRF's and be refused (two
                    # geometries); relink moves only the shapes.
                    try:
                        self.project.relink(self.fit_object_name,
                                            self.fit_name)
                    except (ValueError, KeyError):
                        pass    # an unlinkable fit is shown, not refused
                self._links_changed()
                item = self._item_for_object(self.fit_name)
                self.tree.clearSelection()
                item.setSelected(True)
                self.tree.setCurrentItem(item)
            finally:
                self.tree.blockSignals(False)
        # the session journals as the fit_modes call that replays it:
        # explicit picks in confirm order (the residual peels
        # sequentially, so order is part of the fit), the refine count,
        # and the published name — settling in place as the fit grows
        # (Brandon, 2026-08-30: nothing in the console may be
        # not-replayable)
        described = {(float(m['frequency']), float(m['damping'])):
                     m['description'] for m in self.fit.modes}
        picks = [(f, z, described[(f, z)])
                 if described.get((f, z)) else (f, z)
                 for f, z in self.fit.confirmed]
        line = (f'project.fit_modes({self.fit_name!r}, '
                f'at={picks!r}'
                + (f', refine={self.fit.refined}' if self.fit.refined
                   else '')
                + f', name={self.fit_object_name!r})')
        journal = self.project.journal
        if journal and journal[-1].startswith(prefix):
            journal[-1] = line
        else:
            journal.append(line)

    def stop_fitting(self) -> None:
        if self.fit is None:
            return
        self.fit = None
        self.fit_name = None
        self._fit_cursor = None
        self._fit_parabola = None
        self._fit_damping_label = None
        if self._fit_range_edits:
            for edit in self._fit_range_edits:
                edit.deleteLater()
        self._fit_range_edits = None
        self._fit_model = None
        self._fit_plot = None
        self.fit_bar.hide()
        self.views.setOrientation(Qt.Orientation.Vertical)

    def _series_scales(self, series):
        """{name: dB} for the measured PSDs in a drawn series, resolved
        through each one's family — or None when nothing is a
        comparison, leaving the plot to its own object-local reading."""
        from ..core.data import Psd, Specification

        specs = [data for _n, data, _r in series
                 if isinstance(data, Specification)]
        if len(specs) != 1:
            return None
        return {name: self._family_scale_db(specs[0], name)
                for name, data, _r in series
                if isinstance(data, Psd)
                and not isinstance(data, Specification)
                and name in self.objects}

    def _scale_family(self, name):
        """The measured PSDs that share one comparison scale: the
        object and every non-specification Psd linked with it — its
        source, or the octave bands made from it. One family, one
        number, because an octave PSD is the same measurement on
        another grid and two scales for one run is a contradiction."""
        from ..core.data import Psd, Specification

        group = self.project.group_of(name) or [name]
        if name not in group:
            group = [name, *group]
        return [n for n in group
                if isinstance(self.objects.get(n), Psd)
                and not isinstance(self.objects.get(n), Specification)]

    def _family_scale_db(self, spec, name):
        """The comparison scale for `name`, resolved the way the report
        resolves it: a value held anywhere in the family wins (edits
        keep the family equal), and detection happens on the narrowband
        member — the same data banded onto octaves must not round to a
        different decibel than the lines it came from."""
        from ..core.compliance import comparison_scale_db

        family = self._scale_family(name)
        held = [self.objects[n].scale_db for n in family
                if self.objects[n].scale_db is not None]
        if held:
            return int(held[0])
        roots = [n for n in family
                 if getattr(self.objects[n], 'bandwidth', None) is None]
        return comparison_scale_db(
            spec, self.objects[roots[0]] if roots else self.objects[name])

    def _show_comparison_scaling(self, series):
        """Put the comparison's scaling on the bar, or take it away.

        The number shown is the one every reading of the comparison
        uses — the held value, or the detected one — so the bar states
        what the curves, the error bars and the table were computed
        from. Editing it goes to `_comparison_scale_edited`.
        """
        from ..core.data import Psd, Specification

        self._scaling_pair = None
        if series is None:
            self.data_pane.show_scaling(None)
            return
        specs = [(name, data) for name, data, _r in series
                 if isinstance(data, Specification)]
        measured = [(name, data) for name, data, _r in series
                    if isinstance(data, Psd)
                    and not isinstance(data, Specification)]
        if len(specs) != 1 or len(measured) != 1:
            self.data_pane.show_scaling(None)
            return
        self._scaling_pair = (specs[0][0], measured[0][0])
        db = self._family_scale_db(specs[0][1], measured[0][0])
        self.data_pane.show_scaling(f'{db:+d} dB' if db else '0 dB')

    def _comparison_scale_edited(self, text):
        """The Scaling field was edited: hold the value on the measured
        spectra, or return it to automatic detection.

        Held on the object rather than the view, like the averaging —
        so the report compares exactly what the screen compares, and a
        saved project remembers the judgment. A refused entry changes
        nothing and says why; the redraw restates what still holds.
        """
        if self._scaling_pair is None:
            return
        spec_name, measured_name = self._scaling_pair
        spec = self.objects.get(spec_name)
        psd = self.objects.get(measured_name)
        if spec is None or psd is None:
            return
        raw = text.strip().lower().removesuffix('db').strip()
        if not raw:
            value = None
        else:
            try:
                value = int(raw)
            except ValueError:
                # redraw first: the drawing restates the status line,
                # and the refusal has to be what is left standing
                self.render_current()
                self._show_status(
                    'Scaling is whole decibels — "+6", "-3", 0, or blank '
                    'to detect it from the data')
                return
        # the whole family: a PSD and the octave bands made from it are
        # one measurement and get one scale, so editing either sets
        # both — the report reads the narrowband, and a value held on
        # only the octave object never reached it
        family = self._scale_family(measured_name)
        for other in family:
            self.objects[other].scale_db = value
        self.render_current()
        held = (measured_name if len(family) == 1
                else f'{measured_name} and {len(family) - 1} linked '
                     f'PSD{"s" * (len(family) > 2)}')
        if value is None:
            detected = self._family_scale_db(spec, measured_name)
            self._show_status('Scaling returned to automatic — detected '
                              f'{detected:+d} dB from the data')
        else:
            self._show_status(
                f'{held} held at {value:+d} dB against '
                f'{spec_name} — the data itself is unchanged')

    def _compliance_rows(self, series):
        """(rows, unit) for a specification and the measurement it
        bounds, or None when the selection is not that.

        What the bar charts are drawn from. There was a table too, and
        it was removed: six channels of it are read, sixty are scanned,
        and the same numbers as bars answer "which channel, and how
        many" before anything is read at all.
        """
        from ..core.compliance import compare_all
        from ..core.data import Psd, Specification

        specs = [(data, records) for _n, data, records in series
                 if isinstance(data, Specification)]
        measured = [(name, data, records) for name, data, records in series
                    if isinstance(data, Psd)
                    and not isinstance(data, Specification)]
        if len(specs) != 1 or len(measured) != 1:
            return None
        name, data, records = measured[0]
        rows = compare_all(specs[0][0], data, specs[0][1], records,
                           scale_db=self._family_scale_db(specs[0][0], name))
        if not rows:
            return None
        return rows, self.unit_system.label(
            data.ordinate_dim[0].replace('**2/frequency', ''))

    def _specification_rows(self, series):
        """(rows, unit) for a specification on its own — nothing else
        drawn beside it — or None when the selection is not that.

        The same shape `_compliance_rows` returns, so the table under
        the levels is the same machinery the comparison's is. On its
        own means on its own: a specification drawn with a time
        history, or the whole project, is not a specification being
        read, and offered the levels reading there (2026-09-06) the
        toggle turned up on the project row.
        """
        from ..core.compliance import specification_rms
        from ..core.data import Specification
        from ..plot import pair_label
        from .object_tables import _pair_of

        if len(series) != 1 or not isinstance(series[0][1], Specification):
            return None
        spec, records = series[0][1], series[0][2]
        wanted = (range(spec.num_records) if records is None
                  else sorted(int(i) for i in records))
        # a level is a channel's: a target that holds its cross terms
        # (a virtual point's, every pair stated) listed nine rows for
        # three channels, six of them NaN, and counted nine (2026-09-07)
        references = spec.reference_dof
        wanted = [i for i in wanted
                  if references is None
                  or str(references[i]) == str(spec.response_dof[i])]
        rows = []
        for i in wanted:
            rows.append((pair_label(_pair_of(spec, i)),
                         {'specification_rms': specification_rms(spec, i)}))
        if not rows:
            return None
        return rows, self.unit_system.label(
            spec.ordinate_dim[wanted[0]].replace('**2/frequency', ''))

    def _render_specifications(self, found):
        """The specification's own levels, a row per channel."""
        from ..plot import pair_label
        from .object_tables import ComplianceRows, specification_model

        rows, unit = found
        plotted = (pair_label(self._plotted_pair)
                   if self._plotted_pair else None)
        holder = ComplianceRows(rows, unit, plotted)
        self._set_table_model(specification_model(holder, self))
        self._follow_compliance(holder)
        return (f'{len(rows)} specification '
                f'channel{"s" * (len(rows) != 1)}')

    def _render_levels(self, found):
        """A specification's own levels as a bar per channel, the
        table of the same numbers beneath — the comparison's RMS
        reading without anything to be out of."""
        from ..plot.bars import level_chart

        rows, unit = found
        colors = resolve_theme(self.theme_name)
        self.data_pane.graphics.clear()
        plot = self.data_pane.graphics.addPlot(row=0, col=0)
        self.bar_chart = level_chart(
            plot, [(label, values['specification_rms'])
                   for label, values in rows], colors, units=unit)
        return self.bar_chart.summary.toPlainText()

    def _render_bars(self, found):
        """The comparison as a bar per channel, one reading at a time.

        A table of six channels is read; a table of sixty is scanned.
        The same numbers as bars answer which channel is worst and how
        many are out before anything is read at all — which is why the
        table these replaced is gone rather than sitting beside them.
        """
        from ..core.compliance import ERROR_DB, LINES_PERCENT, channel_errors
        from ..plot.bars import error_chart, lines_chart

        rows, _unit = found
        errors = channel_errors(rows)
        if not errors:
            return 'nothing in common to compare'
        colors = resolve_theme(self.theme_name)
        self.data_pane.graphics.clear()
        plot = self.data_pane.graphics.addPlot(row=0, col=0)
        which = self.data_pane.comparison_view
        if which == 'error':
            low, high = self.data_pane.error_bounds or (-ERROR_DB, ERROR_DB)
            self.bar_chart = error_chart(
                plot, errors, colors, low=low, high=high,
                changed=self._error_bounds_moved)
        else:
            low = (self.data_pane.lines_bound
                   if self.data_pane.lines_bound is not None
                   else LINES_PERCENT)
            self.bar_chart = lines_chart(
                plot, errors, colors, low=low,
                changed=self._lines_bound_moved)
        return self.bar_chart.summary.toPlainText()

    def _render_kurtosis(self, history, series):
        """How Gaussian each channel of this record is, a bar apiece.

        A different reading of the record rather than a mark on it, so
        it replaces the trace — the toggle stands the averaging and
        shock views down for the same reason.

        Every channel on one chart whatever it measures: kurtosis is
        dimensionless, so accelerations and forces share this axis
        honestly (Brandon, 2026-08-24).
        """
        from ..core.kurtosis import (
            HIGH,
            LOW,
            analyzed_span,
            channel_kurtosis,
        )
        from ..plot.bars import kurtosis_chart

        records = next((picked for _name, data, picked in series
                        if data is history and picked), None)
        rows = channel_kurtosis(history, records)
        if not rows:
            return 'no records to read'
        colors = resolve_theme(self.theme_name)
        self.data_pane.graphics.clear()
        plot = self.data_pane.graphics.addPlot(row=0, col=0)
        low, high = self.data_pane.kurtosis_bounds or (LOW, HIGH)
        try:
            name = self.project.name_of(history)
        except (KeyError, ValueError):
            pass
        else:
            self._journal_view(
                f"visualdynamics.plot.plot_kurtosis(project[{name!r}]",
                f"visualdynamics.plot.plot_kurtosis(project[{name!r}]"
                + (f", records={list(records)!r}" if records else '')
                + f", low={low!r}, high={high!r}, "
                "path='kurtosis.png', show=False)")
        self.bar_chart = kurtosis_chart(
            plot, rows, colors, low=low, high=high,
            changed=self._kurtosis_bounds_moved)
        # which stretch it read: the same record answers 4.23 whole
        # and 2.92 over its analyzed frames, so the status bar says
        # which one this is rather than leaving it to be guessed
        _spans, phrase = analyzed_span(history)
        return f'{self.bar_chart.summary.toPlainText()}, {phrase}'

    def _render_wavelet(self, history, name=None, records=None):
        """Where this record's frequencies are, moment by moment.

        One record at a time — a scalogram is dense enough that two
        side by side read as noise — and the project tree is where it
        is chosen, the same selection every other view reads (Brandon,
        2026-08-29; it was a combo box on the panel for two days, and
        that was the same choice in two places). The first selected
        record is the one drawn, and the selection is settled onto it
        so the tree always says what the picture is of.

        The transform is over what the record actually holds, and the
        panel says what of that is cone of influence — inside the cone
        the picture is an artifact of where the record was cut, and it
        looks exactly like data.
        """
        from ..core import wavelet as wavelet_math
        from ..plot.scalogram import scalogram_image

        panel = self.data_pane.wavelet_panel
        # the panel clamps what it is handed to what *this* record can
        # carry, so the settings come back off it rather than out of the
        # sticky dict: a range tuned on a 4 kHz run and carried to a
        # 512 Hz one arrives above Nyquist, and reading the dict
        # instead asked for a transform of frequencies that are not
        # there
        panel.show_history(history, self.data_pane.wavelet_settings)
        settings = panel.settings()
        self.data_pane.wavelet_settings = settings
        panel.show()

        # the first selected record, or the record's own first row when
        # the object is selected whole — and the tree is settled onto
        # that one record, so the selection and the picture cannot
        # disagree. Signals stay blocked through `_select_records`, so
        # this never re-renders.
        channel = records[0] if records else 0
        if name is not None and len(history.response_dof) > 1 \
                and records != [channel]:
            item = self._item_for_object(name)
            if item is not None:
                item.setExpanded(True)
            self._select_records(name, [channel])
        rate = history.sample_rate
        values = np.real(np.asarray(history.ordinate)[channel])
        if not 0.0 < settings['low'] < settings['high']:
            return ('no frequencies this record can carry in that range '
                    f'— it reaches {rate / 2.0:.4g} Hz')
        frequencies = wavelet_math.log_frequencies(
            settings['low'], settings['high'], settings['per_octave'])
        # the panel clamps to Nyquist, but a record swapped underneath a
        # sticky setting has its own; refuse to draw rather than to
        # transform something the record cannot carry
        frequencies = frequencies[frequencies < rate / 2.0]
        if frequencies.size < 2:
            return 'no frequencies this record can carry in that range'
        # the picture-sized reading, never the whole transform: a long
        # record's coefficients are gigabytes and its surface was what
        # crashed the app (Brandon, 2026-09-15). Each column is its
        # slice's peak, so a transient's ridge is not strided past.
        clock, magnitude = wavelet_math.scalogram_peaks(
            values, rate, frequencies, settings['omega0'])
        clock = clock + float(history.abscissa[0])

        colors = resolve_theme(self.theme_name)
        us = self.unit_system
        units = history.ordinate_unit[channel] or ''
        # 3-D is what this reading opens in (Brandon, 2026-08-27): a
        # scalogram is a function of two variables, so a surface is its
        # natural form and the flat picture is the fallback — which is
        # the opposite of the toggle's default elsewhere and is why the
        # bar offers it here at all.
        self.data_pane.offer_waterfall(True)
        if self.data_pane.showing_waterfall:
            from ..viz.scalogram import add_scalogram
            from ..viz.waterfall import place_camera

            self.data_pane.create_waterfall_plotter()
            self.data_pane.show_waterfall(True)
            self.data_pane.graphics.clear()
            plotter = self.data_pane.waterfall_plotter
            plotter.clear()
            plotter.set_background(colors['scene_background'],
                                   top=colors['scene_background_top'])
            add_scalogram(
                plotter, magnitude, clock, frequencies, theme=self.theme_name,
                omega0=settings['omega0'],
                time_label=f'time [{us.label_html("time")}]',
                level_label=f'{history.ordinate_dim[channel]}'
                            + (f' [{units}]' if units else ''))
            if self._waterfall_object != f'wavelet:{history.record_label(channel)}':
                place_camera(plotter)
                self._waterfall_object = \
                    f'wavelet:{history.record_label(channel)}'
            plotter.render()
        else:
            self.data_pane.show_waterfall(False)
            self.data_pane.graphics.clear()
            plot = self.data_pane.graphics.addPlot(row=0, col=0)
            scalogram_image(
                plot, magnitude, clock, frequencies, colors, label=history.ordinate_dim[channel],
                units=units, omega0=settings['omega0'],
                time_label=f'time [{us.label_html("time")}]')

        if name is not None:
            self._journal_view(
                f"visualdynamics.plot.plot_scalogram(project[{name!r}]",
                f"visualdynamics.plot.plot_scalogram(project[{name!r}], "
                f"channel={channel}, low={settings['low']!r}, "
                f"high={settings['high']!r}, "
                f"per_octave={settings['per_octave']}, "
                f"omega0={settings['omega0']!r}, path='scalogram.png')")
        cone = float(wavelet_math.cone_of_influence(
            [frequencies[0]], rate, settings['omega0'])[0])
        return (f'{history.record_label(channel)}: '
                f'{frequencies[0]:.3g} to {frequencies[-1]:.3g} Hz in '
                f'{len(frequencies)} lines, '
                f'cone {cone:.3g} s at the bottom')

    def _kurtosis_bounds_moved(self, low, high):
        """Remember a dragged band, so a redraw does not put it back."""
        self.data_pane.kurtosis_bounds = (low, high)

    def _replication_found(self, series):
        """(record, specification, channels) when the selection is a
        transient run and the waveform it was controlled to, else None.

        By type, not by name and not by the order they were clicked. A
        `TransientSpecification` *is* a `TimeHistory`, so the record has
        to be the one that is not the target — which is also why the
        user never has to say which is which.

        `channels` is the DOFs picked in either grid, and it is what
        the plot draws. Picking channels in the tree and then being
        shown every channel anyway is the grid saying one thing and the
        plot another; the two grids are pooled because a channel chosen
        on the target and the same channel chosen on the record are the
        same request. Empty when nothing was picked, which is every
        channel — the same as asking for the object whole.
        """
        from ..core.data import TimeHistory, TransientSpecification

        specs = [(data, records) for _n, data, records in series
                 if isinstance(data, TransientSpecification)]
        measured = [(data, records) for _n, data, records in series
                    if isinstance(data, TimeHistory)
                    and not isinstance(data, TransientSpecification)]
        if len(specs) != 1 or len(measured) != 1:
            return None
        channels = self._comparison_channels(specs[0], measured[0])
        return measured[0][0], specs[0][0], channels

    def _render_replication(self, measured, specification, channels):
        """A transient record against its target: the event, or the
        waveform error of it.

        Both readings describe the same repeat — the one the event box
        points at. Nothing here names a repeat the worst: which one that
        is depends on the reading you care about and on what the article
        is for, and the numbers are all on screen to decide with.

        Two readings and only two. A level error wants two PSDs and an
        SRS deviation wants two spectra, and those are reached by
        computing them and selecting the pair, where the
        specification's own PSD is a `Specification` and its SRS a
        `ShockSpecification` — so the app still knows which is which,
        and there is one way to each number rather than two that agree
        only by luck.
        """
        from ..core import replication as rep

        if channels is None:
            self.data_pane.show_events([])
            self.data_pane.graphics.clear()
            return ('the records picked share no channel — pick the '
                    'same DOF on both, or pick on one side only')
        try:
            event = self.project.name_of(measured)
            target = self.project.name_of(specification)
        except (KeyError, ValueError):
            pass
        else:
            mode = self.data_pane.replication_view
            self._journal_view(
                'visualdynamics.plot.plot_replication('
                f'project[{event!r}], project[{target!r}]',
                'visualdynamics.plot.plot_replication('
                f'project[{event!r}], project[{target!r}], {mode!r}, '
                "path='replication.png', show=False)")
        averaging = rep.averaging_for(measured, specification)
        if averaging is None:
            self.data_pane.show_events([])
            return ('the record is shorter than one playing of the '
                    'specification')
        lag = rep.lag_of(measured, specification, averaging)
        found = rep.playings(measured, specification, averaging, lag)
        self.data_pane.show_events(
            [f'Event {p["frame"] + 1} of {len(found)}' for p in found])
        bounds = found
        # what the record holds past the last whole playing. Not
        # analyzed — a waveform error over part of a window is taken
        # against a different stretch of the target — but said, because
        # a run that recorded two seconds of a further event should not
        # have it vanish between a frame count and a file size
        spare = rep.leftover(measured, specification, averaging, lag)
        trailing = ('' if spare is None else
                    f'; {spare[0] / measured.sample_rate:.2f} s of a '
                    f'further event ({spare[1]:.0%}) was recorded and is '
                    'too short to analyze')
        held = self.data_pane.chosen_event()
        event = held if held is not None and held < len(bounds) else 0
        which = self.data_pane.replication_view
        delay = (f'; aligned {lag:+d} sample{"s" * (abs(lag) != 1)}'
                 if lag else '') + trailing
        # only the waveform error: it is the one reading on screen,
        # and asking for the SRS as well costs a ramp-invariant
        # filter per band per cell on every selection change
        rows = rep.compare(measured, specification, averaging, lag,
                           metrics=('waveform',))
        every = [p['frame'] for p in found]
        # what is drawn, as (channel, playing) pairs. The tree outranks
        # the table, which outranks the event box: a narrower, more
        # deliberate pick wins over a wider one
        dofs = list(specification.response_dof)
        if channels:
            pairs = [(dof, event) for dof in channels if dof in dofs]
        else:
            pairs = [pair for pair in self._replication_pairs
                     if pair[0] in dofs and pair[1] in every]
        if not pairs:
            held = self.data_pane.chosen_pair()
            first = held[0] if held and held[0] in dofs else dofs[0]
            pairs = [(first, event)]
        self._replication_pairs = pairs
        if which == 'overlay':
            message = self._render_events(measured, specification, pairs,
                                          averaging, lag, len(bounds))
        else:
            wanted = set(pairs)
            message = self._render_replication_bars(
                [row for row in rows
                 if (row['label'], row['frame']) in wanted],
                len(bounds))
        # the table goes up whichever reading is on top of it: it is the
        # account of every channel at every playing, and neither the
        # plot nor the bar chart above it is ever that
        self._show_replication_table(rows, dofs, every)
        return message + delay

    def _event_chosen(self):
        """The event box moved: same channels, the playing it names.

        The grid pins both a channel and a playing, so left alone it
        would outrank the box and the box would do nothing — which is
        the trap the channel box fell into first. Moving the playing
        and taking the channels along keeps both meaningful: the box
        answers "which repeat", the grid answers "which of these", and
        neither has to give up its half.
        """
        event = self.data_pane.chosen_event()
        if event is not None and self._replication_pairs:
            drawn = []
            for channel, _playing in self._replication_pairs:
                if channel not in drawn:
                    drawn.append(channel)
            self._replication_pairs = [(channel, event) for channel in drawn]
        self.render_current()

    def _show_compliance_table(self, found):
        """Every control channel, how far its level is out and how much
        of its band is, under the plot.

        Picking rows points the plot at those channels, several at
        once. Rebuilt only when the numbers change, for the reason the
        transient grid is: a drawing is what a pick *causes*, so
        rewriting the selection on every drawing means arguing with the
        user about what they just clicked.
        """
        from ..core.compliance import channel_errors
        from .object_tables import ComplianceGrid, compliance_grid_model

        rows, _unit = found
        errors = channel_errors(rows)
        if not errors:
            return
        held = self._compliance_holder
        if (held is not None and self.table.model() is self._compliance_model
                and held.rows == errors):
            held.plotted = set(self._compliance_channels)
            return
        holder = ComplianceGrid(errors, self._compliance_channels)
        model = compliance_grid_model(holder, self)
        self._compliance_holder = holder
        self._compliance_model = model
        self._set_table_model(model)
        selection = self.table.selectionModel()
        if selection is None:
            return
        self._syncing_compliance = True
        try:
            selection.clearSelection()
            for row, (label, _db, _pct) in enumerate(errors):
                if label in holder.plotted:
                    self.table.selectRow(row)
        finally:
            self._syncing_compliance = False
        selection.selectionChanged.connect(
            lambda *_args, h=holder: self._compliance_channels_picked(h))

    def _compliance_channels_picked(self, holder):
        """Rows were picked: draw those channels."""
        if self._syncing_compliance:
            return
        picked = []
        for index in self.table.selectedIndexes():
            pair = holder.pair_at(index.row())
            if pair is not None and pair not in picked:
                picked.append(pair)
        if not picked or set(picked) == set(holder.plotted):
            return
        self._compliance_channels = picked
        if self._replication_pending:
            return
        self._replication_pending = True
        QTimer.singleShot(0, self._draw_replication)

    @staticmethod
    def _comparison_channels(spec_entry, measured_entry):
        """The DOFs a two-object comparison draws, from the records
        picked in either grid — or None when the picks cannot meet.

        One side picking names the channels for both: a channel chosen
        on the target and the same channel chosen on the record are
        the same request. But when *both* grids pick, the picks have
        to agree — the intersection, in the specification's order. A
        record on one side and a target on the other that share no
        DOF is not a comparison, and pooling them drew each with the
        other's channel as though it were (Brandon, 2026-08-20:
        104Z+ measured against the 101Z+ target put four curves up
        and answered a question nobody asked). None says so, and the
        caller shows nothing but the reason.
        """
        sides = []
        for data, records in (spec_entry, measured_entry):
            dofs = []
            for i in records or []:
                # the DOF, which every data array has; `channel_key`
                # is a time history's alone, and asking a shock
                # spectrum for it crashed the render the moment an
                # event was picked beside the target (Brandon,
                # 2026-09-11)
                dof = str(data.response_dof[i])
                if dof not in dofs:
                    dofs.append(dof)
            sides.append(dofs)
        picked_spec, picked_measured = sides
        if picked_spec and picked_measured:
            shared = [dof for dof in picked_spec if dof in picked_measured]
            return shared or None
        return picked_spec or picked_measured

    def _srs_found(self, series):
        """(measured, specification, channels) for a pair of shock
        spectra, else None. By type, as everywhere else — a
        `ShockSpecification` is an `Srs`, so the measurement is the one
        that is not the target."""
        from ..core.data import ShockSpecification, Srs

        specs = [(data, records) for _n, data, records in series
                 if isinstance(data, ShockSpecification)]
        measured = [(data, records) for _n, data, records in series
                    if isinstance(data, Srs)
                    and not isinstance(data, ShockSpecification)]
        if len(specs) != 1 or len(measured) != 1:
            return None
        channels = self._comparison_channels(specs[0], measured[0])
        # the events picked on the measured side, by their block: a
        # pick of one shock's records beside the target is that shock
        # against the target, not every shock (Brandon, 2026-09-11 —
        # "I want multi-selection to work for that even if I select a
        # single SRS event to plot against the specification")
        data, records = measured[0]
        events = None
        if records and data.block is not None:
            events = sorted({str(data.block[i]) for i in records})
        return data, specs[0][0], channels, events

    def _render_srs(self, measured, specification, channels, picked_events=None):
        """Measured shock spectra against the one they had to meet.

        Two readings, composed differently on purpose (Brandon,
        2026-08-20). The curves show **every event of one control
        channel** — the events are what a shock series compares, and
        the box picks whose spectra are up, because six channels of
        four events each is a plot with no reading in it. The bars
        show **everything**: every channel of every event, one signed
        number each, because the bar chart exists to say at a glance
        where the whole test stands and a view that pages is a view
        that hides.
        """
        import numpy as np

        from ..core.compliance import SRS_ERROR_DB, srs_errors
        from ..plot import build_plots
        from ..plot.bars import replication_chart

        if channels is None:
            self.data_pane.show_events([])
            self.data_pane.graphics.clear()
            return ('the records picked share no channel — pick the '
                    'same DOF on both, or pick on one side only')
        rows = srs_errors(specification, measured)
        # narrowed to what was picked: the channels either side named,
        # and the events the measured side named — the curves, the
        # bars and the table all read the same rows
        if channels:
            rows = [row for row in rows if row[0] in channels]
        if picked_events is not None:
            rows = [row for row in rows if row[1] in picked_events]
        if not rows:
            return 'no channel in common to compare'
        blocks = sorted({block for _d, block, _v in rows if block},
                        key=lambda b: (len(b), b))
        events = list(range(max(len(blocks), 1)))
        dofs = []
        for dof, _block, _value in rows:
            if dof not in dofs:
                dofs.append(dof)
        index = {block: i for i, block in enumerate(blocks)}
        values = {(dof, index.get(block, 0)): value
                  for dof, block, value in rows}
        if self.data_pane.srs_view == 'curves':
            # every channel at once is the three-dimensional default:
            # the stage shows all events of all channels with their
            # bands, and the 2D/3D toggle stands down to the flat
            # one-channel-at-a-time reading with its drop-down
            self.data_pane.offer_waterfall(True)
            if self.data_pane.showing_waterfall:
                self.data_pane.show_events([])
                # the stage draws the picked records and channels only
                # `picked_events`, never `events`: that name is the
                # table's event *indices* below, and the first cut
                # read block names against them and kept nothing
                narrowed = bool(channels) or picked_events is not None
                picked = [i for i in range(measured.num_records)
                          if str(measured.response_dof[i]) in dofs
                          and (picked_events is None or measured.block is None
                               or str(measured.block[i]) in picked_events)]
                targets = [j for j in range(specification.num_records)
                           if str(specification.response_dof[j]) in dofs]
                return self._render_banded(
                    'the required SRS', specification,
                    targets if narrowed else None,
                    measured, 'measured shock spectra',
                    picked if narrowed else None)
            # the box holds the control channels here, not the events:
            # all of one channel's events share the plot, and sticking
            # by index keeps the same position across redraws the way
            # the event box does everywhere else
            self.data_pane.show_events(list(dofs))
            chosen = self.data_pane.chosen_event() or 0
            dof = dofs[min(chosen, len(dofs) - 1)]
            keep = [i for i in range(measured.num_records)
                    if str(measured.response_dof[i]) == dof
                    and (picked_events is None or measured.block is None
                         or str(measured.block[i]) in picked_events)]
            target = [j for j in range(specification.num_records)
                      if str(specification.response_dof[j]) == dof]
            build_plots(self.data_pane.graphics,
                        [('Measured', measured, keep),
                         ('Specification', specification, target)],
                        unit_system=self.unit_system, theme=self.theme_name)
            message = (f'{dof}: {"all " if picked_events is None else ""}'
                       f'{len(keep)} event{"s" * (len(keep) != 1)} '
                       'against the one required')
        else:
            self.data_pane.show_events([])
            shown = [(f'{dof} e{e + 1}' if len(events) > 1 else dof,
                      values[(dof, e)])
                     for dof in dofs for e in events
                     if (dof, e) in values
                     and np.isfinite(values[(dof, e)])]
            if not shown:
                return 'nothing here can be scored'
            self.data_pane.show_pairs([])
            self.data_pane.graphics.clear()
            plot = self.data_pane.graphics.addPlot(row=0, col=0)
            low, high = (self.data_pane.srs_bounds
                         or (-SRS_ERROR_DB, SRS_ERROR_DB))
            # signed and bounded both sides: red past the ceiling is a
            # shock that hit too hard, blue past the floor one that
            # under-hit — the same reading, colors and unit the random
            # project's level error gives
            self.bar_chart = replication_chart(
                plot, shown, resolve_theme(self.theme_name), 'srs_rms',
                low=low, high=high, changed=self._srs_bounds_moved)
            message = self.bar_chart.summary.toPlainText()
        self._show_replication_table(
            [{'label': dof, 'frame': e, 'zero_target': False,
              'waveform': values[(dof, e)]}
             for dof in dofs for e in events if (dof, e) in values],
            dofs, events, units='dB')
        return message

    def _render_banded(self, spec_name, spec, spec_records,
                       measured=None, measured_name=None,
                       measured_records=None, scale_db=None):
        """Any banded spectrum on the stage: frequency across,
        channels receding, level up — every channel's requirement
        with its zones, and every channel's measurement with the
        lines outside an abort limit boxed over their own bins, at
        once. The scriptable face is `viz.banded.plot_banded_stage`.
        """
        from ..plot import scaled_for_comparison
        from ..viz.banded import add_banded_stage
        from ..viz.waterfall import place_camera

        pane = self.data_pane
        pane.show_waterfall(True)
        pane.graphics.clear()
        pane.show_quantities([])
        plotter = pane.waterfall_plotter
        plotter.clear()
        colors = resolve_theme(self.theme_name)
        plotter.set_background(colors['scene_background'],
                               top=colors['scene_background_top'])
        scaled_note = ''
        if measured is not None and scale_db:
            # the same scale the 2-D comparison applies, applied the
            # same way, so the two readings never disagree by a level
            measured = scaled_for_comparison(measured, scale_db)
            scaled_note = f' ({scale_db:+d} dB)'
        arrays = add_banded_stage(
            plotter, spec, measured, measured_records,
            unit_system=self.unit_system, theme=self.theme_name,
            specification_records=spec_records)
        place_camera(plotter)
        plotter.render()
        stations = arrays['stations']
        n = sum(not s['cross'] for s in stations)
        axis_note = self._offer_frequency_axis(
            [(spec_name, spec, spec_records)])
        if measured is None:
            crosses = [s for s in stations if s['cross']]
            blank = sum(not np.isfinite(s['target']).any() for s in crosses)
            said = (f'{spec_name}: {n} channel{"s" * (n != 1)} with '
                    'their bands')
            if crosses:
                said += (f' and {len(crosses)} cross term'
                         f'{"s" * (len(crosses) != 1)}')
                if blank:
                    # a pair stated as no coherence is zero at every
                    # line, and zero has no place on a log axis
                    said += (f' ({blank} of them zero at every line — '
                             'nothing to draw on a log axis)')
            return said + f' on the stage{axis_note}'
        boxed = sum(len(s['exceed']) for s in arrays['stations'])
        out = (f', {boxed} record{"s" * (boxed != 1)} outside an '
               'abort limit' if boxed else '')
        return (f'{measured_name}{scaled_note} against {spec_name}: '
                f'{n} channel{"s" * (n != 1)} on the stage{out}{axis_note}')

    def _render_sine(self, spec_entry, level_entries, series):
        """A sine specification, extracted levels, or both together.

        Three readings, each defaulting to the dimension the data
        lives in. The **stage** (frequency across, time receding,
        amplitude up) is the default for the specification, the
        levels, and their comparison alike — a sweep is a path
        through that space, and the 3D toggle stands down to the
        flat per-channel reading as everywhere. The **bars** read
        the comparison as one signed number per tone per channel.
        Tone picks on either object's rows restrict every reading.
        """
        import numpy as np

        from ..core.compliance import SINE_ERROR_DB, sine_errors
        from ..plot import build_plots
        from ..plot.bars import replication_chart
        from ..viz.sinespec import add_sine_stage
        from ..viz.waterfall import place_camera

        spec_name, spec, spec_picks = (spec_entry if spec_entry
                                       else (None, None, None))
        picked_names = None
        if spec is not None and spec_picks:
            picked_names = [spec.tones[i].name
                            for i in sorted(set(spec_picks))]

        measured = []
        if level_entries:
            _set_name, level_set, level_picks = level_entries[0]
            rows = (sorted(set(level_picks)) if level_picks
                    else range(len(level_set.levels)))
            measured += [level_set.levels[i] for i in rows]
        measured += [obj for _n, obj, _r in series
                     if isinstance(obj, SineLevel)]
        stray = [name for name, obj, _records in series
                 if not isinstance(obj, SineLevel)]
        note = ('' if not stray else
                f' — {", ".join(stray)} not drawn: deselect the sine '
                'objects to plot other data with them')

        def stage(spec_shown, levels_shown, dof, tones, what):
            pane = self.data_pane
            pane.show_waterfall(True)
            pane.graphics.clear()
            pane.show_quantities([])
            plotter = pane.waterfall_plotter
            plotter.clear()
            colors = resolve_theme(self.theme_name)
            plotter.set_background(colors['scene_background'],
                                   top=colors['scene_background_top'])
            add_sine_stage(plotter, specification=spec_shown,
                           levels=levels_shown, dof=dof,
                           unit_system=self.unit_system,
                           theme=self.theme_name, tones=tones)
            place_camera(plotter)
            plotter.render()
            return (f'{dof}: {what} on the stage — frequency across, '
                    'time receding, amplitude up' + note)

        if spec is None:
            # levels alone
            if not measured:
                return 'nothing to draw' + note
            dofs = list(measured[0].response_dof)
            self.data_pane.show_events(dofs)
            self.data_pane.show_srs_views(False)
            pick = self.data_pane.chosen_event() or 0
            dof = dofs[min(pick, len(dofs) - 1)]
            names = [level.tone for level in measured]
            if self.data_pane.showing_waterfall:
                return stage(None, measured, dof,
                             None, f'{len(names)} extracted '
                             f'level{"s" * (len(names) != 1)}')
            drawn = []
            for level in measured:
                keep = [i for i in range(level.num_records)
                        if str(level.response_dof[i]) == dof]
                if keep:
                    drawn.append((f'{level.tone} level', level, keep))
            build_plots(self.data_pane.graphics, drawn,
                        unit_system=self.unit_system,
                        theme=self.theme_name)
            return (f'{dof}: {len(drawn)} extracted '
                    f'level{"s" * (len(drawn) != 1)}' + note)

        known = {tone.name for tone in spec.tones}
        if picked_names is not None:
            known &= set(picked_names)
        matched = [level for level in measured if level.tone in known]
        tones = ([level.tone for level in matched] if matched
                 else [name for name in (picked_names
                                         or [tone.name
                                             for tone in spec.tones])
                       if name in {tone.name for tone in spec.tones}])

        self.data_pane.show_srs_views(bool(matched))
        showing_bars = matched and self.data_pane.srs_view != 'curves'
        if not showing_bars:
            dofs = list(spec.response_dof)
            self.data_pane.show_events(dofs)
            pick = self.data_pane.chosen_event() or 0
            dof = dofs[min(pick, len(dofs) - 1)]
            row = [dofs.index(dof)]
            if self.data_pane.showing_waterfall:
                what = (f'{len(matched)} level'
                        f'{"s" * (len(matched) != 1)} against '
                        if matched else '')
                return stage(spec, matched, dof, tones,
                             f'{what}{len(tones)} '
                             f'tone{"s" * (len(tones) != 1)} of '
                             f'{spec_name}')
            drawn = []
            for level in matched:
                keep = [i for i in range(level.num_records)
                        if str(level.response_dof[i]) == dof]
                if keep:
                    drawn.append((f'{level.tone} level', level, keep))
            for tone in tones:
                drawn.append((f'{tone} requirement',
                              spec.tone_curve(tone), row))
            build_plots(self.data_pane.graphics, drawn,
                        unit_system=self.unit_system,
                        theme=self.theme_name)
            what = (f'{len(matched)} level{"s" * (len(matched) != 1)} '
                    f'against ' if matched else '')
            return (f'{dof}: {what}{len(tones)} '
                    f'tone{"s" * (len(tones) != 1)} of {spec_name}'
                    + note)
        self.data_pane.show_events([])
        rows = sine_errors(spec, matched)
        if not rows:
            return 'nothing here can be scored' + note
        dofs = []
        for dof, _tone, _value in rows:
            if dof not in dofs:
                dofs.append(dof)
        values = {(dof, tone): value for dof, tone, value in rows}
        shown = [(f'{dof} {tone}' if len(tones) > 1 else dof,
                  values[(dof, tone)])
                 for dof in dofs for tone in tones
                 if (dof, tone) in values
                 and np.isfinite(values[(dof, tone)])]
        self.data_pane.show_pairs([])
        self.data_pane.graphics.clear()
        plot = self.data_pane.graphics.addPlot(row=0, col=0)
        low, high = (self.data_pane.srs_bounds
                     or (-SINE_ERROR_DB, SINE_ERROR_DB))
        self.bar_chart = replication_chart(
            plot, shown, resolve_theme(self.theme_name), 'srs_rms',
            low=low, high=high, changed=self._srs_bounds_moved)
        self._show_replication_table(
            [{'label': dof, 'frame': k, 'zero_target': False,
              'waveform': values[(dof, tone)]}
             for dof in dofs for k, tone in enumerate(tones)
             if (dof, tone) in values],
            dofs, list(range(len(tones))), units='dB')
        return self.bar_chart.summary.toPlainText() + note

    def _srs_bounds_moved(self, low, high):
        self.data_pane.srs_bounds = (low, high)
        self._show_status(self.bar_chart.summary.toPlainText())

    def _show_replication_table(self, rows, channels, events,
                                units='%'):
        """The waveform-error grid: channels down, playings across.

        Picking cells points the plot at them, and a cell names both a
        channel and a repeat — which is the reason for the grid. A row
        is that channel everywhere, a column is that playing across
        every channel, and any scatter of cells is exactly what it
        looks like.

        What is already drawn comes back selected, so a grid of a
        hundred cells says which of them is on screen — but only when
        the grid is being *built*. When the numbers in it have not
        changed, the model and the selection are left strictly alone.

        That last part is the whole of a bug worth remembering. Every
        drawing used to rebuild the model, clear the selection and put
        it back from `_replication_pairs`; and since a drawing is what a
        pick *causes*, command-clicking a cell tore down the selection
        that had just been made and rebuilt it from this end's idea of
        it. Anything that idea could not express — a column-zero pick,
        which stands for a whole row — vanished, so cells appeared to
        deselect themselves as more were added. The selection is the
        input here, not the output, and rebuilding it is the app
        arguing with the user about what they just clicked.
        """
        import numpy as np

        from .object_tables import ReplicationRows, replication_model

        values = {(row['label'], row['frame']):
                  (None if not np.isfinite(row['waveform'])
                   else row['waveform'])
                  for row in rows}
        held = self._replication_holder
        if (held is not None and self.table.model() is self._replication_model
                and held.channels == list(channels)
                and held.events == list(events)
                and held.values == values
                and self._replication_units == units):
            # same grid: keep what is on screen and say what is drawn
            held.plotted = set(self._replication_pairs)
            return
        self._replication_units = units
        holder = ReplicationRows(channels, events, values,
                                 self._replication_pairs)
        model = replication_model(holder, self, units)
        self._replication_holder = holder
        self._replication_model = model
        self._set_table_model(model)
        selection = self.table.selectionModel()
        if selection is None:
            return
        self._syncing_compliance = True
        try:
            selection.clearSelection()
            for channel, event in holder.plotted:
                if channel in holder.channels and event in holder.events:
                    selection.select(
                        model.index(holder.channels.index(channel),
                                    holder.events.index(event) + 1),
                        QItemSelectionModel.SelectionFlag.Select)
        finally:
            self._syncing_compliance = False
        selection.selectionChanged.connect(
            lambda *_args, h=holder: self._replication_cells_picked(h))

    def _replication_cells_picked(self, holder):
        """Cells were picked: draw what they name.

        Queued rather than done here, for `_compliance_row_picked`'s
        reason exactly — this runs from inside the table's own selection
        change, and rebuilding the view there replaces the model under
        Qt's feet halfway through an arrow key.
        """
        if self._syncing_compliance:
            return
        pairs = []
        for index in self.table.selectedIndexes():
            for pair in holder.pairs_at(index.row(), index.column()):
                if pair not in pairs:
                    pairs.append(pair)
        if not pairs or set(pairs) == set(holder.plotted):
            return
        self._replication_pairs = pairs
        # one redraw however many times the selection changes on the
        # way to settling. Qt emits selectionChanged per cell as a drag
        # crosses them, and drawing thirty-six times to arrive at the
        # thirty-sixth is thirty-five plots nobody sees
        if self._replication_pending:
            return
        self._replication_pending = True
        QTimer.singleShot(0, self._draw_replication)

    def _draw_replication(self):
        self._replication_pending = False
        self.render_current()

    def _render_events(self, measured, specification, pairs, averaging,
                       lag, count):
        """The picked repeats drawn over the waveform they aimed at.

        One curve per (channel, playing) asked for, and the target
        drawn once per channel however many playings are up — the
        target is the same waveform every time, and repeating it would
        put one curve under another and call them two.
        """
        from ..core import replication as rep
        from ..plot import build_plots

        dofs = list(specification.response_dof)
        self.data_pane.show_pairs([(dof, dof) for dof in dofs])
        by_event = {}
        for channel, event in pairs:
            by_event.setdefault(event, []).append(channel)
        series = []
        for event in sorted(by_event):
            slice_ = rep.event_slice(measured, specification, event,
                                     averaging, lag)
            if slice_ is None:
                continue
            held = list(slice_.response_dof)
            wanted = [held.index(c) for c in by_event[event] if c in held]
            if wanted:
                series.append((f'Event {event + 1}', slice_, wanted))
        if not series:
            return 'none of those repeats is in the record'
        drawn = []
        for channel, _event in pairs:
            if channel not in drawn and channel in dofs:
                drawn.append(channel)
        series.append(('Specification', specification,
                       [dofs.index(c) for c in drawn]))
        build_plots(self.data_pane.graphics, series,
                    unit_system=self.unit_system, theme=self.theme_name)
        playings = sorted({event for _c, event in pairs})
        return (f'{", ".join(drawn)} over '
                f'event{"s" * (len(playings) != 1)} '
                f'{", ".join(str(e + 1) for e in playings)} of {count}, '
                'against the specification')

    def _render_replication_bars(self, rows, count):
        """The picked cells as bars, one per (channel, playing).

        The repeat only earns a place in a label when more than one is
        up; with a single playing selected the channel names are the
        whole story and `101Z+ e3` on every bar is noise.
        """
        import numpy as np

        from ..core.replication import WAVEFORM_PERCENT
        from ..plot.bars import replication_chart

        usable = [row for row in rows if np.isfinite(row['waveform'])]
        silent = sorted({row['label'] for row in rows
                         if row.get('zero_target')})
        if not usable:
            return 'no control channel here can be scored against its target'
        many = len({row['frame'] for row in usable}) > 1
        shown = [(f"{row['label']} e{row['frame'] + 1}" if many
                  else row['label'], float(row['waveform']))
                 for row in usable]
        # every picked cell is a bar here, so there is nothing to choose
        self.data_pane.show_pairs([])
        colors = resolve_theme(self.theme_name)
        self.data_pane.graphics.clear()
        plot = self.data_pane.graphics.addPlot(row=0, col=0)
        low = (self.data_pane.waveform_bound
               if self.data_pane.waveform_bound is not None
               else WAVEFORM_PERCENT)
        self.bar_chart = replication_chart(
            plot, shown, colors, 'waveform', low=low, high=None,
            changed=self._waveform_bound_moved)
        playings = sorted({row['frame'] for row in usable})
        message = (f'event{"s" * (len(playings) != 1)} '
                   f'{", ".join(str(e + 1) for e in playings)} of {count}: '
                   f'{self.bar_chart.summary.toPlainText()}')
        if silent:
            # a channel with no bar is a channel that cannot be scored,
            # not one that scored zero, and the difference matters
            message += (f'; {", ".join(silent)} asked for nothing and '
                        'cannot be scored against it')
        return message

    def _waveform_bound_moved(self, low, _high):
        self.data_pane.waveform_bound = low
        self._show_status(self.bar_chart.summary.toPlainText())

    def _error_bounds_moved(self, low, high):
        """Remember where the thresholds were dragged to.

        Stored on the pane beside the other sticky view choices: a
        threshold that went back to its default on the next redraw
        would be no threshold at all.
        """
        self.data_pane.error_bounds = (low, high)
        self._show_status(self.bar_chart.summary.toPlainText())

    def _lines_bound_moved(self, low, _high):
        self.data_pane.lines_bound = low
        self._show_status(self.bar_chart.summary.toPlainText())

    def _follow_compliance(self, holder):
        """Tie the table and the plot together, both ways.

        The row the plot is drawing is selected, so a table of twenty
        channels says which one is on screen; and picking a row switches
        the plot to it, which is the same question asked from the other
        side.

        Guarded, because each direction triggers the other: selecting
        the row fires the selection signal, which would switch the pair,
        which redraws, which selects the row again.
        """
        selection = self.table.selectionModel()
        if selection is None:
            return
        if holder.plotted is not None:
            row = holder.row_of(holder.plotted)
            if row is not None:
                self._syncing_compliance = True
                try:
                    self.table.selectRow(row)
                finally:
                    self._syncing_compliance = False
        selection.selectionChanged.connect(
            lambda *_args, h=holder: self._compliance_row_picked(h))

    def _compliance_row_picked(self, holder):
        """A row was picked: draw that channel's comparison.

        Queued, never done here. This runs from inside the table's own
        selection change — which for an arrow key is inside
        `QAbstractItemView.keyPressEvent`, halfway through Qt moving the
        current index. Rebuilding the view there replaces the model and
        the selection model under Qt's feet, and it finishes the move
        against the new ones: the row went to 1, the plot switched, and
        then the selection landed back on 0 and switched it straight
        back. The arrow keys looked dead.

        So the label is taken now and acted on once Qt has finished.
        The same reasoning as `_import_after_drop`, and the same fix.

        Only when it is a different channel from the one already up —
        the plot is rebuilt to switch, and rebuilding it to show what it
        is already showing would fight the user's scroll position.
        """
        if self._syncing_compliance:
            return
        rows = {index.row() for index in self.table.selectedIndexes()}
        if len(rows) != 1:
            return
        label = holder.rows[rows.pop()][0]
        if label == holder.plotted:
            return
        QTimer.singleShot(0, lambda: self._show_compliance_pair(label))

    def _show_compliance_pair(self, label):
        """Draw the comparison the table is pointing at, if it moved.

        The channel list is cleared first. This arrives through
        `select_pair_label`, which moves the box *quietly* — it sets
        `_loading_pairs` so the box does not report a choice it was
        told to make — and that report is what `pair_chosen` normally
        clears on. Without this the previous pick outranked the box and
        the arrow keys in the table stopped moving the plot.
        """
        if self.data_pane.select_pair_label(label):
            self._compliance_channels = []
            self.render_current()

    def _render_table(self, table, rows=None, name=None):
        self._set_table_model(channel_table_model(table, self, rows=rows))
        self._arm_row_deletion(name, 'channel', rows)
        if rows is not None:
            return f'{len(rows)} of {table.num_channels} channels'
        return self._status_for(table)

    def _set_table_model(self, model):
        # every table edit that changes an object reaches the journal
        # through the model's own suffix (Brandon, 2026-08-30: adding a
        # traceline said nothing, and the sweep found the whole edit
        # surface silent)
        model.edit_journaled.connect(
            lambda suffix, m=model: self._journal_table_edit(m, suffix))
        editing = self.editing is not None
        # while editing, the table is a working surface, not something to
        # lift wholesale into a spreadsheet
        self.table.whole_table_copy = not editing
        # ...and its rows are the entities, so Delete removes them. Anywhere
        # else Delete clears cells, which is what it does in every table
        if editing:
            self.table.addAction(self.delete_action)
        else:
            self.table.removeAction(self.delete_action)
        # each model starts from clean row-deletion state; the renderer that
        # knows its rows are items arms it again afterwards
        self.table.rows_deletable = False
        self._hide_mac()
        self.table_bar.hide()
        if self.report_editor is not None:
            self.report_editor.hide()
        self.tables_row.show()
        self.table.setVisible(True)
        if self._rows_wired:
            self.table.rows_deleted.disconnect()
            self._rows_wired = False
        self.table.setModel(model)
        model.edit_rejected.connect(self._show_status)
        self.table.resizeColumnsToContents()
        return model

    def _arm_row_deletion(self, name, kind, rows=None):
        """Rows of this table are sub-items of `name`; Delete removes them.

        Only when the owner is known — a table shown without one has nowhere
        to send the deletion. With `rows` the table is a filtered view, and
        view row i is really item rows[i].
        """
        if name is None:
            return
        self.table.rows_deletable = True
        self._rows_wired = True
        self.table.rows_deleted.connect(
            lambda picked, name=name, kind=kind, rows=rows:
            self._delete_table_rows(
                name, kind,
                picked if rows is None else [rows[i] for i in picked]))

    def _delete_table_rows(self, name, kind, rows):
        obj = self.objects.get(name)
        if obj is None or not rows:
            return
        deleter = {'channel': 'delete_channels',
                   'mode': 'delete_modes',
                   'match': 'delete_matches'}[kind]
        line = self._deletion_line(name, obj, deleter, rows)
        try:
            getattr(obj, deleter)(rows)
        except ValueError as refusal:
            self._show_status(f'{name}: {refusal}')
            return
        self.project.journal.append(line)
        self._refresh_item(self._item_for_object(name), obj)
        self.refresh_compatibility()
        self.render_current()
        self._show_status(
            f'Removed {len(rows)} {kind}{"s" if len(rows) != 1 else ""} '
            f'from {name}')

    def _render_cross_mac(self, first, second, matrix):
        """Two shape sets selected: the MAC between them, each axis
        ticked by its own frequencies. The grid reads tall — whichever
        set has more modes makes the rows — and the matrix arrives
        computed, projected onto the basis DOFs when the geometries
        demanded it."""
        _name_a, a = first
        _name_b, b = second
        # Picking a cell re-renders this whole view; redrawing the same
        # comparison must keep the zoom the user set. A *different*
        # comparison starts framed whole — the view belongs to the user
        # only while it is their view.
        shown = (_name_a, _name_b, matrix.shape)
        # the comparison offers both readings, and the comparison
        # controls besides — restored here because the fit hides them
        self.mode_table_action.setVisible(True)
        self.mac_bars_action.setVisible(True)
        if self.mac_bars_action.isChecked():
            # the comparison's marks, gathered the way the flat grid's
            # overlay block gathers them: the picked pairs, the animated
            # one, and the pairs already committed to the matched table
            compare = self._compare or {}
            pairs = list(compare.get('pairs')
                         or ([compare['cell']] if compare.get('cell')
                             else []))
            found = self._matched_for(_name_a, _name_b)
            committed = ([tuple(pair) for pair in found[1].pairs]
                         if found is not None else [])
            self._mac_shown = shown
            self._draw_mac_bars(a.frequency, matrix, key=shown,
                                column_frequencies=b.frequency,
                                selected=pairs,
                                active=compare.get('cell'),
                                matched=committed)
            return
        previous = None
        if self._mac_shown == shown:
            old_plot = next((item for item in self.mac_view.ci.items
                             if hasattr(item, 'viewRange')), None)
            if old_plot is not None:
                previous = old_plot.viewRange()
        self._mac_shown = shown
        if b.num_shapes > a.num_shapes:
            build_mac(self.mac_view, b.frequency, matrix.T,
                      theme=self.theme_name,
                      column_frequencies=a.frequency)
            self.mac_frame.set_ratio(
                mac_frame_ratio(b.num_shapes, a.num_shapes))
        else:
            build_mac(self.mac_view, a.frequency, matrix,
                      theme=self.theme_name,
                      column_frequencies=b.frequency)
            self.mac_frame.set_ratio(
                mac_frame_ratio(a.num_shapes, b.num_shapes))
        if previous is not None:
            plot = next((item for item in self.mac_view.ci.items
                         if hasattr(item, 'viewRange')), None)
            if plot is not None:
                plot.setRange(xRange=previous[0], yRange=previous[1],
                              padding=0)
        self._mac_surface(bars=False)

    def _render_shape_table(self, shapes, mode=None, name=None):
        """Mode list: frequency, damping, and a description to fill in —
        with the auto-MAC beside it, since how distinct the modes are is
        part of reading the list."""
        model = self._set_table_model(shape_table_model(shapes, self))
        self._arm_row_deletion(name, 'mode')
        self._mac_shown = None      # not a comparison; its zoom is gone
        self.mode_table_action.setVisible(True)
        self.mac_bars_action.setVisible(True)
        if self.mac_bars_action.isChecked():
            self._draw_mac_bars(shapes.frequency, shapes.auto_mac(),
                                key=('auto', name, shapes.num_shapes))
        else:
            build_mac(self.mac_view, shapes.frequency, shapes.auto_mac(),
                      theme=self.theme_name)
            self.mac_frame.set_ratio(1.0)
            self._mac_surface(bars=False)
        self.table_bar.show()
        self.table.setVisible(self.mode_table_action.isChecked())
        if mode is not None:
            self.table.selectRow(mode)
            self.table.scrollTo(model.index(mode, 0))
            return shapes.mode_label(mode)
        return self._status_for(shapes)
