"""The report editor: the exported page as the preview, with the actions
on a bar above it.

The page shown is the same HTML the export writes, so it is exactly what
the reader will get, with the figures drawn by the same JavaScript the
reader's browser will run. It does two things the exported page does
not: it outlines the selected block, and it reports which block was
clicked. Everything else is in Qt. Above the page is a bar of actions
(Insert, Reference, Move Up, Move Down, Delete, and Export as HTML or as
a report template), and beside it is a settings pane for the selection:
a figure block's sources and caption, a text block's Markdown in a Qt
editor, or the report's title, marking and Report Features panel when
nothing is selected (`report_features_panel`). Each action is one
operation on the `Report` model, which the window records in the
journal, followed by a new render of the page. The exported file
carries none of the editor's controls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.report import Report

import contextlib
import html as html_escape
import json
import logging
import os
import tempfile
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import (
    QByteArray,
    QFile,
    QMimeData,
    QObject,
    QSize,
    Qt,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QImage
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .icons import control_icon
from .report_features_panel import ReportFeaturesPanel

# The bar and the pane are Qt, as for every other object (Brandon,
# 2026-09-08: "the toolbar should be the same as the task bar at the top
# of the report screen how we have for every other GUI object").

# the insert menu: Markdown, plus the Result drop-down's options. Each
# template's '_bind' fields are pointed at the first compatible object
# in the project on insert, so a result appears immediately and the
# drop-downs are there to repoint it.
BLOCK_TEMPLATES = {
    'markdown': {'kind': 'text', 'text': ''},
    'geometry': {'kind': 'scene', 'geometry': '', 'shapes': '',
                 'caption': '', '_bind': ('geometry',)},
    'modes': {'kind': 'scene', 'geometry': '', 'shapes': '',
              'caption': '', '_bind': ('geometry', 'shapes')},
    'cmif': {'kind': 'plot', 'source': '', 'mode': 'cmif', 'shapes': '',
             'caption': '', '_bind': ('source', 'shapes')},
    'coherence': {'kind': 'plot', 'source': '', 'mode': 'map',
                  'caption': '', '_bind': ('source',)},
    # the app's 3-D reading of a data object, in the document: every
    # record on the stage, receding, colored by level
    'stage': {'kind': 'plot', 'source': '', 'mode': 'stage',
              'caption': '', '_bind': ('source',)},
    'mac': {'kind': 'plot', 'source': '', 'mode': 'mac', 'caption': '',
            '_bind': ('source',)},
    'photos': {'kind': 'photo', 'source': '', 'photo': '', 'caption': '',
               '_bind': ('source', 'photo')},
    # a table block draws from a shape set or a channel table; this
    # insert means the channel table specifically, so it binds to one
    'channel_table': {'kind': 'table', 'source': '', 'caption': '',
                      '_bind': ('source',), '_only': 'ChannelTable'},
    'matches': {'kind': 'pairs', 'source': '', 'caption': '',
                '_bind': ('source',), '_only': 'MatchedModes'},
    'overlay': {'kind': 'overlay', 'source': '', 'caption': '',
                '_bind': ('source',), '_only': 'MatchedModes'},
    # two sources, and neither is optional: a bar chart of the
    # comparison is a comparison, so it binds a specification and the
    # measurement that answered it
    'bars': {'kind': 'bars', 'mode': 'error', 'source': '', 'measured': '',
             'caption': '', '_bind': ('source', 'measured'),
             '_only': 'Specification'},
    # the pass/fail box: the octave-band specification and the
    # octave-band PSDs judged against it
    'verdict': {'kind': 'verdict', 'source': '', 'measured': '',
                '_bind': ('source', 'measured'), '_only': 'Specification'},
    # responses against prediction: the predictions and the response
    # read against them; the level run is typed in the pane
    'prediction': {'kind': 'prediction', 'source': '', 'measured': '',
                   'level': None, 'caption': '',
                   '_bind': ('source', 'measured')},
}

#: how long typing in the text editor rests before the page re-renders
#: — a rebuild costs a quarter of a second, and paying it per keystroke
#: would make the editor stutter
TEXT_DEBOUNCE_MS = 400


_log = logging.getLogger(__name__)

class _Bridge(QObject):
    """What the page talks to: one JSON operation per `apply`, and a
    figure's image for the clipboard."""

    operated = Signal(dict)

    @Slot(str)
    def apply(self, payload: str) -> None:
        self.operated.emit(json.loads(payload))

    @Slot(str, result=bool)
    def copy_image(self, data_url: str) -> bool:
        """A figure's PNG, as the page's copy button sends it, onto the
        system clipboard. The page cannot reach the clipboard on its own
        from inside a `QWebEngineView` without a permission grant that
        differs by Qt version, and this process owns the clipboard
        anyway. Returns whether it copied, so the button can say so."""
        # the answer goes back for the button's own message (Brandon,
        # 2026-09-24)
        _header, _, encoded = data_url.partition(',')
        image = QImage.fromData(QByteArray.fromBase64(encoded.encode('ascii')))
        if image.isNull():
            return False
        QApplication.clipboard().setImage(image)
        return True

    @Slot(str, str, result=bool)
    def copy_table(self, text: str, html: str) -> bool:
        """A table from the page, as tab-separated text and as HTML in
        one clipboard entry, so a spreadsheet pastes cells and a
        document or a mail pastes a table."""
        if not text.strip():
            return False
        mime = QMimeData()
        mime.setText(text)
        mime.setHtml(html)
        QApplication.clipboard().setMimeData(mime)
        return True


class ReportEditor(QWidget):
    """The report editor: the web view, the bar and the settings pane.

    It applies each operation to the report and renders the page again.
    `edited` is emitted after each change to the report, and
    `export_requested` and `template_requested` when Export → HTML or
    Export → Report template is chosen.
    """

    export_requested = Signal()
    #: the report on its own, as a template another project can load
    template_requested = Signal()
    #: a file to add to the project's data index, asked for by the
    #: window (Insert, Attached File, with the data index on)
    attach_requested = Signal()
    #: a file to take out of the data index, by its name
    unindex_requested = Signal(str)
    edited = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.report: Any = None
        #: callables the window sets, so the editor reads the project
        #: as it stands rather than a copy taken when it opened
        self.objects: Callable[[], dict[str, Any]] | None = None
        self.object_groups: Callable[[], list[Any]] | None = None
        self.unit_system: Any = None
        #: the block the bar and the pane act on, or None for the report
        self.selected: int | None = None
        #: (label, caption) for every numbered figure and table, from
        #: the last render — the Reference menu's offer
        self._labels: list[tuple[str, str]] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.toolbar: QToolBar = self._build_toolbar()
        layout.addWidget(self.toolbar)
        self.split: QSplitter = QSplitter(Qt.Orientation.Horizontal)
        self.view: QWebEngineView = QWebEngineView()
        #: the page, made here and owned here rather than by the view:
        #: a view deletes a page it made itself in the same call that
        #: detaches it (`QWebEngineViewPrivate::bindPageAndView`), so
        #: its live web contents were always torn down with the widget
        #: they drew into still standing — the hang `stand_down` exists
        #: for. A page the view does not own is only detached by
        #: `setPage(None)`, and its teardown can be sequenced.
        self.page: QWebEnginePage = QWebEnginePage(self)
        self.view.setPage(self.page)
        #: the loadFinished slot of the navigation in flight, if any
        self._restore = None
        #: the one timer that scrolls a freshly loaded page back to
        #: where it was — held, so standing down can stop it: a bare
        #: single-shot could still fire its script into a page in the
        #: middle of being discarded (the gate's stall, 2026-09-19)
        self._scroll_timer: QTimer = QTimer(self.view)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.setInterval(50)
        self._scroll_timer.timeout.connect(self._scroll_back)
        self._scroll_to: int = 0
        self.split.addWidget(self.view)
        self.pane: QScrollArea = QScrollArea()
        self.pane.setWidgetResizable(True)
        self.pane.setMinimumWidth(240)
        self.split.addWidget(self.pane)
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 1)
        self.split.setSizes([900, 300])
        layout.addWidget(self.split, 1)
        self.bridge: _Bridge = _Bridge(self)
        self.bridge.operated.connect(self._operate)
        self.channel: QWebChannel = QWebChannel(self)
        self.channel.registerObject('bridge', self.bridge)
        self.page.setWebChannel(self.channel)
        self._scroll = 0
        self._page_path = None
        self._channel_js = None
        #: the project changed while this page was not on screen; the
        #: window rebuilds before showing it again rather than paying
        #: 254 ms per change for a document nobody is looking at
        self.stale: bool = False
        #: what stopped the last build, or None: a page that could not
        #: be built says so instead of staying white
        self.failure: str | None = None
        # the pane's widgets, rebuilt when the selection changes; None
        # while the selection has no such field
        self.text_editor: QPlainTextEdit | None = None
        self.caption_edit: QLineEdit | None = None
        # a plot block of a measured spectrum: whether its figure draws
        # the record's caption block, and that caption read out
        self.record_caption_check: QCheckBox | None = None
        self.record_caption_label: QLabel | None = None
        self.title_edit: QLineEdit | None = None
        self.marking_edit: QLineEdit | None = None
        self.color_box: QComboBox | None = None
        #: the Report Features panel, while the report's own settings
        #: are in the pane
        self.features_panel: ReportFeaturesPanel | None = None
        #: how a change of report features is made: the window sets the
        #: project's verb, so the change is journaled as a script would
        #: make it; on its own the editor calls the registry directly.
        #: Called as `features_changer(report, enable=, disable=,
        #: preset=)` and returns a `FeatureSelection`.
        self.features_changer: Callable[..., Any] | None = None
        #: how the readings of the report's check blocks are made before
        #: the page is drawn (report feature `check_blocks`): the window
        #: sets a call that makes them on its worker, so a slow reading
        #: never holds the GUI thread. Called as `checks_preparer(report)`;
        #: False means the worker is busy with another task and the page
        #: waits for the next rebuild. On its own the editor reads them as it renders.
        self.checks_preparer: Callable[[Report], bool] | None = None
        #: the Identity and Issue fields, by field key, and the line
        #: that says what was refused, while the report's own settings
        #: are in the pane and the feature is on
        self.identity_edits: dict[str, Any] = {}
        self.identity_notes: QLabel | None = None
        self.amend_button: QPushButton | None = None
        self.identity_box: QWidget | None = None
        #: how identification is entered and an amendment made: the
        #: window sets the project's verbs, so each is journaled as a
        #: script would make it; on its own the editor calls
        #: `report.identity` directly. Called as `identity_changer(report,
        #: **fields)` and `amender(report, changes, revision=, number=)`.
        self.identity_changer: Callable[..., Any] | None = None
        self.amender: Callable[..., Any] | None = None
        #: a skeleton section's None button and its Interpretation, by
        #: field, while such a block is in the pane
        self.none_button: QPushButton | None = None
        self.by_edit: QLineEdit | None = None
        #: the test dates on a channel table's pane, the Setup and
        #: Method pane's Insert Calibration Status button, and the line
        #: under them that says what was refused, while calibration in
        #: date is on (`report.calibration`)
        self.date_edit: QLineEdit | None = None
        self.end_edit: QLineEdit | None = None
        self.calibration_button: QPushButton | None = None
        self.calibration_note: QLabel | None = None
        #: how the test dates are set: the window sets the project's
        #: verb, so the change is journaled; on its own the editor
        #: calls `report.calibration.set_dates`. Called as
        #: `calibration_dater(report, start, end)`.
        self.calibration_dater: Callable[..., Any] | None = None
        #: the level run, while a responses-against-prediction block is
        #: in the pane
        self.level_edit: QLineEdit | None = None
        #: the runs panels (report feature `run_sections`), while shown,
        #: and what the last of their changes had to say
        self.runs_panel: Any = None
        self.chronology_panel: Any = None
        self._runs_message: str = ''
        #: the window's worker for a verb that takes a while, called as
        #: `run_long(status, verb, *args)`; None runs it here
        self.run_long: Callable[..., Any] | None = None
        self.field_boxes: dict[str, QComboBox] = {}
        self._loading_pane = False
        self._text_timer: QTimer = QTimer(self)
        self._text_timer.setSingleShot(True)
        self._text_timer.setInterval(TEXT_DEBOUNCE_MS)
        self._text_timer.timeout.connect(self.flush_text)
        self._show_selection()

    # ---- the bar ------------------------------------------------------------

    def _build_toolbar(self) -> QToolBar:
        toolbar = QToolBar('Report')
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))

        def menu_button(label, icon, tooltip):
            # an icon, the words in the tooltip — every bar button is
            # (Brandon, 2026-09-12); the text stays on the button for
            # the tests and for anything that reads the bar by name
            button = QToolButton()
            button.setText(label)
            button.setIcon(control_icon(icon))
            button.setToolTip(f'{label} — {tooltip}')
            button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            menu = QMenu(button)
            button.setMenu(menu)
            return button, menu

        self.insert_button, self.insert_menu = menu_button(
            'Insert', 'add',
            'Insert a block after the selected one — Markdown text, a '
            'result the project can give, a photo, a table')
        toolbar.addWidget(self.insert_button)
        self.reference_button, self.reference_menu = menu_button(
            'Reference', 'link',
            'Insert a reference to a figure or table at the cursor — '
            'renumbered with the document')
        toolbar.addWidget(self.reference_button)
        self.up_action: QAction = QAction(control_icon('up'), 'Move Up', self)
        self.up_action.setToolTip('Move Up — the selected block, one place')
        self.up_action.triggered.connect(lambda: self._move(-1))
        self.down_action: QAction = QAction(control_icon('down'), 'Move Down',
                                            self)
        self.down_action.setToolTip('Move Down — the selected block, one place')
        self.down_action.triggered.connect(lambda: self._move(+1))
        self.delete_action: QAction = QAction(control_icon('trash'), 'Delete',
                                              self)
        self.delete_action.setToolTip('Delete — remove the selected block')
        self.delete_action.triggered.connect(self._remove)
        for action in (self.up_action, self.down_action, self.delete_action):
            toolbar.addAction(action)
            button = toolbar.widgetForAction(action)
            if isinstance(button, QToolButton):
                button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        toolbar.addSeparator()
        self.export_button, self.export_menu = menu_button(
            'Export', 'report',
            'Write the report as a file a reader opens without the '
            'application')
        self.export_menu.addAction(
            'HTML…', lambda: self._operate({'op': 'export'}))
        self.export_menu.addAction(
            'Report template…', self.template_requested.emit)
        self.export_action: QAction = toolbar.addWidget(self.export_button)
        return toolbar

    def _fill_insert_menu(self) -> None:
        from ..core.report import insert_options

        self.insert_menu.clear()
        self.insert_menu.addAction(
            'Markdown', lambda: self.insert('markdown'))
        self.insert_menu.addSeparator()
        objects = self.objects() if self.objects else {}
        from ..report.features import enabled

        active = enabled(self.report) if self.report is not None else ()
        for kind, label in insert_options(objects):
            if kind == 'prediction' and 'prediction' not in active:
                continue        # drawn only while its feature is on
            self.insert_menu.addAction(
                label, lambda _checked=False, k=kind: self.insert(k))
        self._fill_index_actions(objects)

    def _fill_index_actions(self, objects) -> None:
        """With the data index on, Insert also adds a file to the index
        and takes out one added by hand; otherwise neither is shown."""
        from ..report.data_index import FEATURE
        from ..report.features import enabled

        if self.report is None or FEATURE not in enabled(self.report):
            return
        self.insert_menu.addSeparator()
        self.insert_menu.addAction('Attached File…',
                                   self.attach_requested.emit)
        index = getattr(objects, 'data_index', None)
        mine = [entry.name for entry in (index or ())
                if entry.source != 'import']
        if mine:
            remove = self.insert_menu.addMenu('Remove from Index')
            for name in mine:
                remove.addAction(name, lambda _checked=False, n=name:
                                 self.unindex_requested.emit(n))

    def _fill_reference_menu(self) -> None:
        self.reference_menu.clear()
        for label, caption in self._labels:
            action = self.reference_menu.addAction(
                f'{label} — {caption}' if caption else f'{label} (no caption)')
            if not caption:
                # a reference finds its figure by caption, so a figure
                # with none cannot be referred to
                action.setEnabled(False)
                action.setToolTip('Give the figure a caption to reference it')
                continue
            kind = 'table' if label.lower().startswith('table') else 'figure'
            action.triggered.connect(
                lambda _checked=False, k=kind, c=caption:
                self.insert_reference(k, c))
        self.reference_menu.setEnabled(bool(self._labels))

    def _offer(self) -> None:
        """The bar follows the selection: block acts with a block,
        Reference while a text block's editor is up."""
        has_block = (self.report is not None and self.selected is not None
                     and 0 <= self.selected < self.report.num_blocks)
        self.up_action.setEnabled(has_block and self.selected > 0)
        self.down_action.setEnabled(
            has_block and self.selected < self.report.num_blocks - 1)
        self.delete_action.setEnabled(has_block)
        self.reference_button.setEnabled(self.text_editor is not None
                                         and bool(self._labels))
        self.insert_button.setEnabled(self.report is not None)
        self.export_button.setEnabled(self.report is not None)

    # ---- the acts -------------------------------------------------------------

    def insert(self, kind: str) -> None:
        """Insert a block of `kind`, one of the Insert menu's kinds, after
        the selected block or at the end, and select it."""
        if self.report is None:
            return
        at = (self.selected + 1 if self.selected is not None
              else self.report.num_blocks)
        self._operate({'op': 'insert', 'at': at, 'kind': kind})

    def insert_reference(self, kind: str, caption: str) -> None:
        """Insert `{{figure:caption}}` or `{{table:caption}}` at the text
        editor's cursor. This is the token the report stores, numbered by
        the same code that numbers the page. Nothing happens unless a
        text block's editor is open."""
        if self.text_editor is None:
            return
        self.text_editor.textCursor().insertText(f'{{{{{kind}:{caption}}}}}')
        self.text_editor.setFocus()

    def _move(self, step: int) -> None:
        if self.report is None or self.selected is None:
            return
        to = self.selected + step
        if not 0 <= to < self.report.num_blocks:
            return
        # `move` drops the lifted block so the first lands at `to` after
        # the lift, which for a step down means two past its own index
        self._operate({'op': 'move', 'from': self.selected,
                       'to': to if step < 0 else to + 1})

    def _remove(self) -> None:
        if self.report is None or self.selected is None:
            return
        self._operate({'op': 'remove', 'at': self.selected})

    def flush_text(self) -> None:
        """Apply the text editor's Markdown to the selected block now. The
        editor does this itself once typing has paused for
        `TEXT_DEBOUNCE_MS`, and tests call it directly."""
        self._text_timer.stop()
        if (self.text_editor is None or self.report is None
                or self.selected is None
                or not 0 <= self.selected < self.report.num_blocks):
            return
        text = self.text_editor.toPlainText()
        if text == self.report.blocks[self.selected].get('text', ''):
            return
        self._operate({'op': 'field', 'at': self.selected, 'field': 'text',
                       'value': text})

    # ---- the pane -----------------------------------------------------------

    @staticmethod
    def _failure_page(text: str) -> str:
        return ('<!DOCTYPE html><html><body style="font-family: sans-serif; '
                'margin: 2em"><h2>This report could not be built</h2>'
                '<p>The error below is the whole story; the report and its '
                'blocks are unchanged.</p><pre style="white-space: pre-wrap">'
                f'{html_escape.escape(text)}</pre></body></html>')

    def _show_selection(self) -> None:
        """Rebuild the pane for what is selected: a text block's editor,
        a figure block's sources and caption, or the report's title and
        marking. Rebuilt on a selection change only — a field edit
        re-renders the page and leaves the pane, and the editor's
        cursor, where they were."""
        from ..core.report import (
            dofs_source_options,
            photo_options,
            shape_options,
            source_options,
            symbolic_options,
            takes_shapes,
        )
        from ..report import scalogram_channel_options

        self._text_timer.stop()
        self.text_editor = self.caption_edit = None
        self.record_caption_check = self.record_caption_label = None
        self.title_edit = self.marking_edit = self.color_box = None
        self.features_panel = None
        self.none_button = self.by_edit = self.level_edit = None
        self.date_edit = self.end_edit = None
        self.calibration_button = self.calibration_note = None
        self.identity_edits, self.identity_notes = {}, None
        self.amend_button = self.identity_box = None
        self.runs_panel = self.chronology_panel = None
        self.field_boxes = {}
        body = QWidget()
        form = QFormLayout(body)
        form.setContentsMargins(8, 8, 8, 8)
        self._loading_pane = True
        try:
            report = self.report
            block = (report.blocks[self.selected]
                     if report is not None and self.selected is not None
                     and 0 <= self.selected < report.num_blocks else None)
            if report is None:
                form.addRow(QLabel('No report'))
            elif self.failure:
                told = QLabel(f'The report could not be built:\n{self.failure}')
                told.setWordWrap(True)
                form.addRow(told)
            elif block is None:
                form.addRow(QLabel('<b>Report</b>'))
                self.title_edit = QLineEdit(report.title)
                self.title_edit.editingFinished.connect(
                    lambda: self._field_edited(
                        {'op': 'title', 'value': self.title_edit.text()},
                        report.title != self.title_edit.text()))
                form.addRow('Title', self.title_edit)
                # one value behind both banners: editing it is editing
                # the marking, and the rebuild redraws the pair in step
                self.marking_edit = QLineEdit(report.marking)
                self.marking_edit.setPlaceholderText('no marking')
                self.marking_edit.editingFinished.connect(
                    lambda: self._field_edited(
                        {'op': 'marking', 'value': self.marking_edit.text()},
                        report.marking != self.marking_edit.text().strip()))
                form.addRow('Marking', self.marking_edit)
                self.color_box = QComboBox()
                self.color_box.addItem('Black/White', 'ink')
                self.color_box.addItem('Red', 'red')
                self.color_box.setCurrentIndex(
                    1 if report.marking_color == 'red' else 0)
                self.color_box.currentIndexChanged.connect(
                    lambda _i: self._field_edited(
                        {'op': 'marking_color',
                         'value': self.color_box.currentData()}, True))
                form.addRow('Marking color', self.color_box)
                form.addRow(QLabel(
                    'Click a block in the page to edit it; Insert adds '
                    'one after the selection, or at the end.'))
                self.features_panel = ReportFeaturesPanel(report.features)
                self.features_panel.requested.connect(
                    lambda change: self._operate({'op': 'features',
                                                  **change}))
                form.addRow(self.features_panel)
                self._identity_pane(form, report)
                self._runs_panel(form)
            elif block.get('kind') == 'text' and 'section' in block:
                self._section_pane(form, block)
            elif block.get('kind') == 'text' and 'check' in block:
                self._check_pane(form, block)
            elif block.get('kind') == 'text':
                form.addRow(QLabel(f'<b>Block {self.selected + 1}</b> — text'))
                self.text_editor = QPlainTextEdit(block.get('text', ''))
                self.text_editor.setPlaceholderText(
                    'Markdown: # heading, **bold**, - list; '
                    'Reference on the bar inserts a figure number')
                self.text_editor.textChanged.connect(self._text_changed)
                form.addRow(self.text_editor)
            else:
                objects = self.objects() if self.objects else {}
                object_groups = self.object_groups() if self.object_groups else None
                form.addRow(QLabel(
                    f'<b>Block {self.selected + 1}</b> — {block.get("kind")}'
                    + (f' ({block.get("mode")})' if block.get('mode') else '')))
                source_key = ('geometry' if block.get('kind') == 'scene'
                              else 'source')
                # option lists are (value, label) pairs: symbolic
                # selectors lead (a binding that survives any renaming),
                # concrete names follow
                self._combo(form, 'source', 'Source', block.get(source_key, ''),
                            [('', '(source)')] + list(symbolic_options(block))
                            + [(name, name)
                               for name in source_options(block, objects)])
                if takes_shapes(block):
                    self._combo(form, 'shapes', 'Shapes', block.get('shapes', ''),
                                [('', '(no shapes)'),
                                 ('@basis:ShapeSet', 'Basis Shape Set (auto)'),
                                 ('@other:ShapeSet', 'Other Shape Set (auto)')]
                                + [(name, name) for name in shape_options(objects)])
                if block.get('mode') == 'scalogram':
                    # the scalogram shows one channel; the reader
                    # chooses which (Brandon, 2026-08-29)
                    self._combo(form, 'channel', 'Channel', block.get('channel', ''),
                                [(name, name) for name in
                                 scalogram_channel_options(block, objects, object_groups)])
                if block.get('dofs'):
                    # which data object the DOF arrows read from
                    self._combo(form, 'dofs_source', 'DOFs from',
                                block.get('dofs_source', ''),
                                [('', '(DOFs from)')]
                                + [(name, name) for name in
                                   dofs_source_options(block, objects)])
                if block.get('kind') == 'photo':
                    self._combo(form, 'photo', 'Photo', block.get('photo', ''),
                                [('', '(photo)')]
                                + [(name, name) for name in
                                   photo_options(block, objects)])
                if block.get('kind') == 'prediction':
                    self._prediction_pane(form, block, objects)
                self.caption_edit = QLineEdit(block.get('caption', ''))
                self.caption_edit.setPlaceholderText('caption')
                self.caption_edit.editingFinished.connect(
                    lambda: self._field_edited(
                        {'op': 'field', 'at': self.selected, 'field': 'caption',
                         'value': self.caption_edit.text()},
                        block.get('caption', '') != self.caption_edit.text()))
                form.addRow('Caption', self.caption_edit)
                self._record_caption_rows(form, block, objects, object_groups)
                self._dates_pane(form, block, objects, object_groups)
        finally:
            self._loading_pane = False
        self.pane.setWidget(body)
        self._offer()

    def _record_caption_rows(self, form, block, objects, object_groups):
        """The record's caption block, for a plot block of a measured
        spectrum: a check to draw it under the figure, and the caption
        itself read out beneath, as the page will show it
        (`report.block_caption`). Nothing for any other block."""
        from ..report import block_caption

        caption = block_caption(block, objects, object_groups,
                                self.unit_system)
        if caption is None:
            return
        self.record_caption_check = QCheckBox('Caption from the record')
        self.record_caption_check.setChecked(bool(block.get('record_caption')))
        self.record_caption_check.setToolTip(
            'Draw the caption block under the figure: the channel, its '
            'location and axis, the run, the estimate, the resolution, the '
            'window and overlap, the DOF and the requirement, each read '
            'from the record')
        self.record_caption_check.toggled.connect(
            lambda checked: self._field_edited(
                {'op': 'field', 'at': self.selected,
                 'field': 'record_caption', 'value': bool(checked)},
                bool(block.get('record_caption')) != bool(checked)))
        form.addRow(self.record_caption_check)
        self.record_caption_label = QLabel('<br>'.join(
            f'<b>{html_escape.escape(label)}:</b> {html_escape.escape(said)}'
            for label, said in caption.rows()))
        self.record_caption_label.setWordWrap(True)
        self.record_caption_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.record_caption_label)

    def _check_pane(self, form, block) -> None:
        """The pane for a check's own block: the check, what it reads and
        its settings, as `Project.report_check_block` wrote them. They
        are read here and set from a script; Remove takes the block."""
        from ..report.check_blocks import CHECKS

        found = CHECKS.get(str(block.get('check')))
        title = found.title if found is not None else str(block.get('check'))
        form.addRow(QLabel(f'<b>{html_escape.escape(title)}</b> — check'))
        if found is not None:
            told = QLabel(found.description)
            told.setWordWrap(True)
            form.addRow(told)
        fields = [(name, str(block[name])) for name in
                  ('source', 'measured', 'specification', 'shapes')
                  if block.get(name)]
        settings = block.get('settings') or {}
        for name, value in fields + [(name, repr(value))
                                     for name, value in settings.items()]:
            shown = QLabel(value)
            shown.setWordWrap(True)
            form.addRow(name, shown)

    def _section_pane(self, form, block) -> None:
        """The pane for a skeleton section's own block: its heading and
        what belongs in it, its words in the Markdown editor, None for a
        section that can have nothing to report, and who interpreted
        for the discussion."""
        from ..report.skeleton import ATTRIBUTED, BY_KEY

        found = BY_KEY.get(str(block.get('section')))
        if found is None:
            # a section this version does not know: its words, as text
            form.addRow(QLabel(f'<b>Block {self.selected + 1}</b> — text'))
        else:
            form.addRow(QLabel(f'<b>{html_escape.escape(found.heading)}</b>'))
            told = QLabel(found.description)
            told.setWordWrap(True)
            form.addRow(told)
        self.text_editor = QPlainTextEdit(block.get('text', ''))
        self.text_editor.setPlaceholderText(
            'Owed until written. Markdown: **bold**, - list, '
            '| a | b | tables')
        self.text_editor.textChanged.connect(self._text_changed)
        form.addRow(self.text_editor)
        if found is not None and found.none:
            self.none_button = QPushButton('None')
            self.none_button.setToolTip(
                f'Writes "{found.none}" in place of the section\'s text')
            self.none_button.clicked.connect(
                lambda: self._operate({'op': 'section', 'at': self.selected,
                                       'none': True}))
            form.addRow(self.none_button)
        if found is not None and found.key == ATTRIBUTED:
            self.by_edit = QLineEdit(str(block.get('by', '')))
            self.by_edit.setPlaceholderText('who made the interpretation')
            self.by_edit.editingFinished.connect(
                lambda: self._field_edited(
                    {'op': 'section', 'at': self.selected,
                     'by': self.by_edit.text()},
                    str(block.get('by', '')) != self.by_edit.text().strip()))
            form.addRow('Interpretation, by', self.by_edit)
        if (found is not None and found.key == 'setup'
                and self._calibrating()):
            # the section is the author's: the program offers what it
            # can prove, and the author keeps it, changes it or not
            self.calibration_button = QPushButton('Insert Calibration Status')
            self.calibration_button.setToolTip(
                'Inserts the sensors and their calibration status on the '
                'test date, from the channel table, at the cursor')
            self.calibration_button.clicked.connect(self._offer_calibration)
            form.addRow(self.calibration_button)
            self.calibration_note = QLabel('')
            self.calibration_note.setWordWrap(True)
            form.addRow(self.calibration_note)
        if found is not None and found.key == 'chronology':
            self._chronology_panel(form, block)

    def _calibrating(self) -> bool:
        from ..report.calibration import FEATURE
        from ..report.features import enabled

        return self.report is not None and FEATURE in enabled(self.report)

    def _dates_pane(self, form, block, objects, object_groups) -> None:
        """With calibration in date on, a channel table's pane carries
        the test's first and last day, which each sensor's calibration
        is judged against."""
        from ..report.calibration import END, START, channel_table_of

        if (not self._calibrating()
                or channel_table_of(block, objects, object_groups) is None):
            return
        self.date_edit = QLineEdit(str(block.get(START, '')))
        self.date_edit.setPlaceholderText('YYYY-MM-DD')
        self.end_edit = QLineEdit(str(block.get(END, '')))
        self.end_edit.setPlaceholderText('the first day (YYYY-MM-DD)')
        for edit in (self.date_edit, self.end_edit):
            edit.editingFinished.connect(self._dates_edited)
        form.addRow('Test date', self.date_edit)
        form.addRow('Last test day', self.end_edit)
        self.calibration_note = QLabel('')
        self.calibration_note.setWordWrap(True)
        form.addRow(self.calibration_note)

    def _dates_edited(self) -> None:
        from ..report.calibration import END, START

        if (self._loading_pane or self.date_edit is None
                or self.selected is None):
            return
        block = self.report.blocks[self.selected]
        start = self.date_edit.text().strip()
        end = self.end_edit.text().strip()
        if (start, end) == (str(block.get(START, '')),
                            str(block.get(END, ''))):
            return
        self._operate({'op': 'test_date', 'start': start, 'end': end})

    def _offer_calibration(self) -> None:
        """Insert the equipment list with its calibration status at the
        cursor of the Setup and Method editor, for the author to keep,
        change or delete; or say why there is none."""
        from ..report.calibration import draft

        if self.text_editor is None or self.report is None:
            return
        objects = self.objects() if self.objects else {}
        object_groups = self.object_groups() if self.object_groups else None
        try:
            text = draft(self.report, objects, object_groups)
        except (TypeError, ValueError) as refused:
            if self.calibration_note is not None:
                self.calibration_note.setText(str(refused))
            return
        if self.calibration_note is not None:
            self.calibration_note.setText('')
        self.text_editor.insertPlainText(text)
        self.flush_text()

    def _prediction_pane(self, form, block, objects) -> None:
        """A responses-against-prediction block's measurement and the
        level it was run at, beside the Source combo that names the
        predictions."""
        from ..core.report import measured_options

        self._combo(form, 'measured', 'Measured', block.get('measured', ''),
                    [('', '(measured)')]
                    + [(name, name) for name in measured_options(objects)])
        level = block.get('level')
        self.level_edit = QLineEdit('' if level is None else f'{level:g}')
        self.level_edit.setPlaceholderText(
            'dB relative to full level, as -12; blank for a '
            'transmissibility')
        self.level_edit.editingFinished.connect(
            lambda: self._field_edited(
                {'op': 'field', 'at': self.selected, 'field': 'level',
                 'value': self.level_edit.text()},
                self.level_edit.text().strip() != (
                    '' if block.get('level') is None
                    else f'{block["level"]:g}')))
        form.addRow('Level run', self.level_edit)

    def _identity_pane(self, outer, report) -> None:
        """The Identity and Issue fields: each identification field the
        author enters, the status, and Amend for an issued report. The
        number and revision of an issued report are read-only: they
        change only by an amendment. Shown while the feature is on
        (`_show_identity`)."""
        from ..report import identity

        self.identity_box = QWidget()
        form = QFormLayout(self.identity_box)
        form.setContentsMargins(0, 8, 0, 0)
        outer.addRow(self.identity_box)
        self._show_identity()
        form.addRow(QLabel('<b>Identity and Issue</b>'))
        held = report.identity
        issued = bool(held.get('record'))
        status = QLabel(f'Status: {identity.status_words(report)}')
        status.setObjectName('identity_status')
        form.addRow(status)
        for field in identity.FIELDS:
            given = identity.value(report, field.key)
            text = '; '.join(given) if field.key == 'authorized' else given
            edit = QLineEdit(text)
            edit.setPlaceholderText(
                'name, role; name, role' if field.key == 'authorized'
                else identity.NOT_ENTERED if field.required else 'optional')
            if issued and field.key in ('number', 'revision'):
                edit.setReadOnly(True)
                edit.setToolTip('Issued: changed only by an amendment')
            edit.editingFinished.connect(
                lambda key=field.key, box=edit: self._identity_edited(
                    key, box.text()))
            form.addRow(field.label, edit)
            self.identity_edits[field.key] = edit
        full = QCheckBox('Not reproduced except in full')
        full.setChecked(identity.value(report, 'full_only'))
        full.setToolTip(identity.FULL_ONLY)
        full.toggled.connect(
            lambda on: self._operate({'op': 'identity', 'field': 'full_only',
                                      'value': bool(on)}))
        form.addRow(full)
        self.identity_edits['full_only'] = full
        if held.get('status', 'draft') != 'draft':
            self.amend_button = QPushButton('Amend...')
            self.amend_button.setToolTip(
                'A change after issue: an amendment with its own revision '
                'and a table of what changed')
            self.amend_button.clicked.connect(self._ask_amendment)
            form.addRow(self.amend_button)
        self.identity_notes = QLabel('')
        self.identity_notes.setWordWrap(True)
        self.identity_notes.setStyleSheet('color: #c0392b;')
        form.addRow(self.identity_notes)

    def _show_identity(self) -> None:
        """Show the Identity and Issue fields while the report has the
        feature on, and hide them otherwise."""
        from ..report.features import enabled
        from ..report.identity import FEATURE

        if self.identity_box is not None and self.report is not None:
            self.identity_box.setVisible(FEATURE in enabled(self.report))

    def _identity_edited(self, key: str, text: str) -> None:
        if self._loading_pane or self.report is None:
            return
        from ..report import identity

        now = identity.value(self.report, key)
        if key == 'authorized':
            value: Any = [part.strip() for part in text.split(';')
                          if part.strip()]
        else:
            value = text.strip()
        if value != now:
            self._operate({'op': 'identity', 'field': key, 'value': value})

    def _change_identity(self, operation) -> bool:
        """Enter one identification field through the verb a script
        calls; a refusal is shown under the fields and changes nothing.
        Returns whether the report changed."""
        from ..report.identity import identify

        changer = self.identity_changer or identify
        before = dict(self.report.identity)
        try:
            changer(self.report, **{str(operation.get('field')):
                                    operation.get('value')})
        except (TypeError, ValueError) as refused:
            if self.identity_notes is not None:
                self.identity_notes.setText(str(refused))
            return False
        return self.report.identity != before

    def _ask_amendment(self) -> None:
        """Ask for the amendment's revision, number and changes, and make
        it (`_operate` 'amend')."""
        from .amend_dialog import AmendDialog

        dialog = AmendDialog(self.report, self)
        if dialog.exec():
            self._operate({'op': 'amend', **dialog.amendment()})

    def _amend(self, operation) -> bool:
        from ..report.identity import amend

        amender = self.amender or amend
        before = dict(self.report.identity)
        try:
            amender(self.report, operation.get('changes', []),
                    revision=operation.get('revision', ''),
                    number=operation.get('number'))
        except (TypeError, ValueError) as refused:
            if self.identity_notes is not None:
                self.identity_notes.setText(str(refused))
            return False
        return self.report.identity != before

    def _runs_panel(self, form) -> None:
        """The runs and their identities, under the Report Features
        panel, while the report has Runs and chronology on. With `form`
        None, the panel already in the pane is added or taken away to
        follow the feature, and nothing else in the pane moves."""
        from .report_runs_pane import RunsPanel, project_of, shown

        objects = self.objects() if self.objects else {}
        wanted = shown(objects, self.report)
        if form is None:
            if self.features_panel is None:
                return
            form = self.features_panel.parentWidget().layout()
            if self.runs_panel is not None and not wanted:
                form.removeRow(self.runs_panel)
                self.runs_panel = None
            if self.runs_panel is not None or not wanted:
                return
        if not wanted:
            return
        self.runs_panel = RunsPanel(project_of(objects), self._runs_message)
        self._runs_message = ''
        self.runs_panel.requested.connect(self._operate)
        form.addRow(self.runs_panel)

    def _chronology_panel(self, form, block) -> None:
        """Draft from the Runs, and the draft on offer while the section
        is unwritten, in the Chronology section's pane."""
        from ..report.runs import drafted
        from .report_runs_pane import ChronologyPanel, shown

        objects = self.objects() if self.objects else {}
        if not shown(objects, self.report):
            return
        offered = ('' if str(block.get('text', '')).strip()
                   else drafted(self.report))
        self.chronology_panel = ChronologyPanel(offered, self._runs_message)
        self._runs_message = ''
        self.chronology_panel.requested.connect(self._operate)
        form.addRow(self.chronology_panel)

    def _combo(self, form, field, label, current, options):
        box = QComboBox()
        for value, text in options:
            box.addItem(text, value)
        index = box.findData(current)
        box.setCurrentIndex(max(index, 0))
        box.currentIndexChanged.connect(
            lambda _i, f=field, b=box: self._field_edited(
                {'op': 'field', 'at': self.selected, 'field': f,
                 'value': b.currentData()}, True))
        form.addRow(label, box)
        self.field_boxes[field] = box
        return box

    def _field_edited(self, operation, changed):
        if self._loading_pane or not changed:
            return
        self._operate(operation)

    def _text_changed(self) -> None:
        if not self._loading_pane:
            self._text_timer.start()

    # ---- the page -----------------------------------------------------------

    def show_report(self, report: Report,
                    objects: Callable[[], dict[str, Any]],
                    unit_system: Any,
                    object_groups: Callable[[], list[Any]] | None = None) -> None:
        """Open `report` in the editor, with nothing selected.

        `objects` returns the project's objects by name and
        `object_groups`, when given, the project's object groups. Both
        are called on every render, so the editor reads the project as it
        stands. `unit_system` is the display units.
        """
        self.report = report
        self.objects = objects
        self.object_groups = object_groups
        self.unit_system = unit_system
        self._scroll = 0
        self.selected = None
        self.rebuild()
        self._show_selection()

    def rebuild(self) -> None:
        """Render the page again from the report and load it, keeping the
        scroll position. When the render raises, the page shows the
        traceback and `failure` holds a one-line summary of it."""
        from ..report import render_html

        if self.report is None:
            return
        if (self.checks_preparer is not None
                and not self.checks_preparer(self.report)):
            # the check readings wait on the worker; the page is drawn
            # by the rebuild that follows them
            self.stale = True
            return
        self.stale = False
        if self._channel_js is None:
            # Qt's own qwebchannel.js, inlined into the page: the page
            # loads from a file (setHtml silently blanks past
            # Chromium's 2 MB limit — a real coherence map is bigger)
            # and file: pages cannot reach qrc:
            resource = QFile(':/qtwebchannel/qwebchannel.js')
            resource.open(QFile.OpenModeFlag.ReadOnly)
            self._channel_js = bytes(resource.readAll()).decode('utf-8')
            resource.close()
        labels: list[tuple[str, str]] = []
        try:
            html = render_html(self.report, self.objects(), self.unit_system,
                               edit=True, channel_js=self._channel_js,
                               object_groups=self.object_groups() if self.object_groups else None,
                               selected=self.selected, labels=labels)
        except Exception as failure:  # noqa: BLE001 — the page reports it
            # A build that raised used to leave the view white and the
            # pane reading "No report", the exception gone to a console
            # nobody was looking at (Brandon, 2026-09-18: "just getting a
            # white screen and no report"). The page carries the
            # traceback now, the pane and the status line the one line
            # that names it, and the report object is still there to
            # be looked at.
            self.failure = f'{type(failure).__name__}: {failure}'
            html = self._failure_page(traceback.format_exc())
            labels = []
        else:
            self.failure = None
        self._labels = labels
        self._fill_insert_menu()
        self._fill_reference_menu()
        if self._page_path is None:
            handle, self._page_path = tempfile.mkstemp(
                prefix='visualdynamics_report_', suffix='.html')
            os.close(handle)
        with open(self._page_path, 'w', encoding='utf-8') as out:
            out.write(html)
        self.view.load(QUrl.fromLocalFile(self._page_path))
        scroll = self._scroll

        def restore(_ok: bool) -> None:
            self._restore = None
            self.view.loadFinished.disconnect(restore)
            # the view owns the timer: a window closed inside those
            # 50 ms takes the view with it, and a bare singleShot then
            # fired into a deleted QWebEngineView (2026-09-13, once
            # windows really died at teardown); and `stand_down` stops
            # it, so no script fires into a page being discarded
            self._scroll_to = int(scroll)
            self._scroll_timer.start()

        # one slot at a time: a rebuild before the last page finished
        # would otherwise leave the earlier slot connected, to fire on
        # this load and disconnect itself twice
        if self._restore is not None:
            with contextlib.suppress(RuntimeError, TypeError):
                self.view.loadFinished.disconnect(self._restore)
        self._restore = restore
        self.view.loadFinished.connect(restore)
        self._offer()

    def _change_features(self, operation) -> bool:
        """Make the Report Features panel's change, and show the panel
        the outcome. Returns whether the report changed: a refused
        change leaves it as it was, says why on the panel, and renders
        nothing again."""
        from ..report.features import configure

        change = {key: operation[key]
                  for key in ('enable', 'disable', 'preset')
                  if operation.get(key) is not None}
        before = list(self.report.features)
        changer = self.features_changer or configure
        try:
            selection = changer(self.report, **change)
        except (TypeError, ValueError) as refused:
            notes = [str(refused)]
        else:
            notes = list(selection.notes)
        if self.features_panel is not None:
            self.features_panel.show_state(self.report.features, notes)
        return self.report.features != before

    def _scroll_back(self) -> None:
        self.page.runJavaScript(f'window.scrollTo(0, {self._scroll_to});')

    def stand_down(self) -> None:
        """Leave the page with nothing for the view's destructor to wait on.

        Call this before the editor is destroyed. A `QWebEngineView`
        destroyed while its page is live can block the whole process,
        because Chromium's teardown waits on the render process and does
        not return. A hidden page can be discarded instead (its render
        process shut down and the page unloaded), and a discarded page is
        destroyed in about a millisecond.

        So the navigation in flight and its scroll-restoring slot are
        stopped, the view is hidden, the page is detached from the view,
        the events that remove the widget are processed, and only then is
        the page discarded, with nothing left to wait on but the render
        process it is shutting down. Each step is written to the debug
        log.
        """
        # History. A destroyed view blocked in a synchronous `mach_msg`
        # call that never returned: the gate's "stall at 98 %", sampled on
        # 2026-09-18 with a worker's main thread parked inside
        # QtWebEngineCore, then named by a faulthandler dump (the window
        # fixture delivering the deferred delete to a window whose report
        # page had just finished loading). Stopping a navigation in flight,
        # the first cut, was not enough: a loaded, rendering page hangs the
        # destructor the same way. Discarding took the destructor from a
        # hang to 0.001 s (measured).
        # Then the discard itself hung (2026-09-28, a stack this time:
        # `setLifecycleState(Discarded)` never returning from
        # `WebContentsAdapter::discard()`). Qt's source says why: discard
        # asks the render process to shut down and then destroys the live
        # web contents synchronously, the same teardown the destructor did,
        # while the page is still bound to the widget it drew into, with
        # that widget's compositor and surfaces alive to be waited on. The
        # view's destructor deletes an owned page in the same call that
        # detaches it. Hence detach first, pump, then discard. That is a
        # mechanism, not a witnessed cure (the hang needs a starved machine
        # and never reproduced alone), so the log still records each step.
        from PySide6.QtWidgets import QApplication

        page = self.page
        # the witness the stall's record asks for (PLAN.md "The 98 %
        # stall, named"): the page's state at the moment of the
        # discard, on the debug log — the faulthandler dump names the
        # line and not the page
        _log.debug('stand_down: loading=%s state=%s restore=%s scroll_timer=%s',
                   page.isLoading(), page.lifecycleState(),
                   self._restore is not None, self._scroll_timer.isActive())
        if self._restore is not None:
            with contextlib.suppress(RuntimeError, TypeError):
                self.view.loadFinished.disconnect(self._restore)
            self._restore = None
        self._scroll_timer.stop()
        self.view.stop()
        self.view.hide()
        # detached before discarded: the view forgets a page it does
        # not own without deleting it, and the widget the page drew
        # into goes away on the events pumped here
        self.view.setPage(None)
        for _ in range(20):
            QApplication.processEvents()
        _log.debug('stand_down: detached')
        with contextlib.suppress(RuntimeError):
            page.setLifecycleState(QWebEnginePage.LifecycleState.Discarded)
        _log.debug('stand_down: discarded, state=%s', page.lifecycleState())
        for _ in range(20):
            QApplication.processEvents()

    def _operate(self, operation):
        report = self.report
        if report is None:
            return
        self._scroll = operation.get('scroll', self._scroll)
        op = operation.get('op')
        if op == 'export':
            self.export_requested.emit()
            return
        if op == 'select':
            at = operation.get('at')
            if isinstance(at, str) and at.startswith('section:'):
                # an owed skeleton section with no block of its own yet:
                # the click makes it, so there is a block to write in
                from ..report.skeleton import write

                try:
                    at = write(report, at.partition(':')[2])
                except (TypeError, ValueError):
                    return
                self.flush_text()
                self.selected = at
                self.edited.emit()
                self.rebuild()
                self._show_selection()
                return
            at = int(at) if at is not None else None
            if at is not None and not 0 <= at < report.num_blocks:
                at = None
            if at != self.selected:
                self.flush_text()
                self.selected = at
                self._show_selection()
            return
        reselect = False
        if op == 'insert':
            kind = str(operation.get('kind', ''))
            if kind.startswith(('time:', 'spectrum:', 'psd:')):
                # Time (Acceleration), Spectra (Force), PSD (…) — one
                # insert per quantity that kind of data actually holds
                prefix, _, dimension = kind.partition(':')
                template = {'kind': 'plot', 'source': '',
                            'mode': 'curves',
                            'select': f'dim:{dimension}',
                            'caption': '', '_bind': ('source',),
                            '_only': {'time': 'TimeHistory',
                                      'spectrum': 'Spectrum',
                                      'psd': 'Psd'}[prefix]}
            elif kind.startswith('dofs:'):
                # DOFs (Force), DOFs (Acceleration), … — the geometry
                # with labeled arrows where one data object measures
                # that quantity
                template = {'kind': 'scene', 'geometry': '', 'shapes': '',
                            'dofs': kind.partition(':')[2],
                            'dofs_source': '', 'caption': '',
                            '_bind': ('geometry', 'dofs_source')}
            elif kind.startswith('frf_drive'):
                # FRF - Drive Point - Magnitude / Real / Imaginary /
                # Phase, with the named set's mode frequencies marked
                template = {'kind': 'plot', 'source': '',
                            'mode': 'curves', 'select': 'drive',
                            'component': kind.partition(':')[2]
                            or 'magnitude', 'shapes': '',
                            'caption': '', '_bind': ('source', 'shapes')}
            else:
                template = BLOCK_TEMPLATES.get(kind)
            if not template:
                return
            from ..core.report import (
                dofs_source_options,
                measured_options,
                photo_options,
                record_dimensions,
                shape_options,
                source_options,
            )

            block = {key: value for key, value in template.items()
                     if not key.startswith('_')}
            objects = self.objects() if self.objects else {}
            for field in template.get('_bind', ()):
                if field == 'shapes':
                    options = shape_options(objects)
                elif field == 'photo':
                    options = photo_options(block, objects)
                elif field == 'dofs_source':
                    options = dofs_source_options(block, objects)
                elif field == 'measured' and block.get('kind') == 'prediction':
                    options = measured_options(objects)
                else:
                    options = source_options(block, objects)
                only = template.get('_only')
                if field == 'source' and only:
                    options = [name for name in options
                               if type(objects[name]).__name__ == only]
                select = block.get('select', '')
                if field == 'source' and select.startswith('dim:'):
                    # bind to data that actually holds the
                    # quantity, not just any data of the right kind
                    options = [name for name in options
                               if select[4:] in record_dimensions(
                                   objects[name])]
                if options:
                    block[field] = options[0]
            at = operation.get('at')
            report.add(block, at=at)
            # the new block is what the bar and the pane act on next
            self.selected = (report.num_blocks - 1 if at is None
                             else int(at))
            reselect = True
        elif op == 'remove':
            report.remove([operation.get('at')])
            self.selected = None
            reselect = True
        elif op == 'move':
            source, to = int(operation.get('from')), int(operation.get('to'))
            report.move([source], to)
            # the selection follows the block it was on
            if self.selected == source:
                self.selected = to - 1 if to > source else to
            reselect = True
        elif op == 'title':
            report.title = str(operation.get('value', report.title))
        elif op == 'marking':
            # one value behind both banners: editing either is editing
            # the marking, and the rebuild redraws the pair in step
            report.marking = str(operation.get('value', '')).strip()
        elif op == 'marking_color':
            wanted = str(operation.get('value', 'ink'))
            report.marking_color = wanted if wanted in ('ink', 'red') \
                else 'ink'
        elif op == 'features':
            if not self._change_features(operation):
                return
            self._show_identity()
            # the runs panel comes or goes with its feature, in place,
            # so the Report Features panel and what it said stay
            self._runs_panel(None)
        elif op == 'identity':
            if not self._change_identity(operation):
                return
            if self.identity_notes is not None:
                self.identity_notes.setText('')
        elif op == 'amend':
            if not self._amend(operation):
                return
            reselect = True
        elif op in ('run', 'repeat_runs', 'draft_chronology',
                    'accept_chronology'):
            # runs and chronology: through the project's verbs, which
            # refuse; the pane says what they had to say
            from .report_runs_pane import operate, project_of

            project = project_of(self.objects() if self.objects else {})
            if project is None:
                return
            self.flush_text()
            self._runs_message = operate(project, report, operation,
                                         self.run_long)
            reselect = True
        elif op == 'section':
            # a skeleton section's one-step None, or who interpreted:
            # through the one function a script calls, which refuses
            index = int(operation.get('at', -1))
            if (not 0 <= index < report.num_blocks
                    or 'section' not in report.blocks[index]):
                return
            from ..report.skeleton import write

            try:
                write(report, report.blocks[index]['section'],
                      none=bool(operation.get('none')),
                      by=operation.get('by'))
            except (TypeError, ValueError):
                return
            # the pane shows the words None wrote
            reselect = bool(operation.get('none'))
        elif op == 'test_date':
            # through the one function a script calls, which refuses
            from ..report.calibration import set_dates

            def dater(report, start, end):
                objects = self.objects() if self.objects else {}
                groups = self.object_groups() if self.object_groups else None
                return set_dates(report, objects, groups, start, end)

            try:
                (self.calibration_dater or dater)(
                    report, operation.get('start') or None,
                    operation.get('end') or None)
            except (TypeError, ValueError) as refused:
                if self.calibration_note is not None:
                    self.calibration_note.setText(str(refused))
                return
            if self.calibration_note is not None:
                self.calibration_note.setText('')
        elif op == 'field':
            index = int(operation.get('at', -1))
            if 0 <= index < report.num_blocks:
                block = report.blocks[index]
                field = operation.get('field')
                value = str(operation.get('value', ''))
                if field == 'record_caption':
                    # a flag, stored only while it is set
                    if operation.get('value'):
                        block['record_caption'] = True
                    else:
                        block.pop('record_caption', None)
                elif field == 'source':
                    key = ('geometry' if block.get('kind') == 'scene'
                           else 'source')
                    block[key] = value
                elif field in ('text', 'caption', 'shapes', 'photo',
                               'dofs_source', 'channel', 'measured'):
                    block[field] = value
                elif field == 'level':
                    # dB relative to full level; blank for none. Text
                    # that is not a number changes nothing
                    try:
                        block['level'] = (float(value) if value.strip()
                                          else None)
                    except ValueError:
                        if self.level_edit is not None:
                            level = block.get('level')
                            self.level_edit.setText(
                                '' if level is None else f'{level:g}')
                        return
        else:
            return
        self.edited.emit()
        self.rebuild()
        if reselect:
            self._show_selection()
