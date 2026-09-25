"""The report editor: the exported page as the preview, the acts on the bar.

The page shown is the very HTML the export writes — a browser view of
exactly what the reader will get, figures drawn by the same JavaScript
their browser will run — and it does two things the export's page does
not: it frames the selected block, and it says which block was clicked.
Everything else lives in Qt (Brandon, 2026-09-08: "the toolbar should be
the same as the task bar at the top of the report screen how we have for
every other GUI object"): a bar of acts above the page — Insert,
Reference, Move Up, Move Down, Delete, Export — and a settings pane
beside it carrying what the selected block has to say: a figure block's
sources and caption, a text block's Markdown in a Qt editor, the
report's title and marking when nothing is selected. Every act is one
operation on the `Report` model, journaled by the window, followed by a
re-render of the page; the exported file never carries any chrome.
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
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .icons import control_icon

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
        anyway; the answer goes back so the button can say whether it
        copied (Brandon, 2026-09-24)."""
        _header, _, encoded = data_url.partition(',')
        image = QImage.fromData(QByteArray.fromBase64(encoded.encode('ascii')))
        if image.isNull():
            return False
        QApplication.clipboard().setImage(image)
        return True


class ReportEditor(QWidget):
    """Owns the web view, the bar and the pane; mutates the report the
    operations describe."""

    export_requested = Signal()
    #: the report on its own, as a template another project can load
    template_requested = Signal()
    edited = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.report: Any = None
        #: callables the window sets, so the editor reads the project
        #: as it stands rather than a copy taken when it opened
        self.objects: Callable[[], dict[str, Any]] | None = None
        self.links: Callable[[], list[Any]] | None = None
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
        self.view.page().setWebChannel(self.channel)
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
        self.title_edit: QLineEdit | None = None
        self.marking_edit: QLineEdit | None = None
        self.color_box: QComboBox | None = None
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
        self.insert_action: QAction = toolbar.addWidget(self.insert_button)
        self.reference_button, self.reference_menu = menu_button(
            'Reference', 'link',
            'Insert a reference to a figure or table at the cursor — '
            'renumbered with the document')
        self.reference_action: QAction = toolbar.addWidget(
            self.reference_button)
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
        for kind, label in insert_options(objects):
            self.insert_menu.addAction(
                label, lambda _checked=False, k=kind: self.insert(k))

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
        """Insert a block of `kind` after the selected block, or at the
        end, and select it."""
        if self.report is None:
            return
        at = (self.selected + 1 if self.selected is not None
              else self.report.num_blocks)
        self._operate({'op': 'insert', 'at': at, 'kind': kind})

    def insert_reference(self, kind: str, caption: str) -> None:
        """`{{figure:caption}}` or `{{table:caption}}` at the text
        editor's cursor — the token the model stores, renumbered by
        the same code that numbers the page."""
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
        """Land the text editor's Markdown on the block now — what the
        debounce does after typing rests, and what a test calls."""
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
        self.title_edit = self.marking_edit = self.color_box = None
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
                links = self.links() if self.links else None
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
                                 scalogram_channel_options(block, objects, links)])
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
                self.caption_edit = QLineEdit(block.get('caption', ''))
                self.caption_edit.setPlaceholderText('caption')
                self.caption_edit.editingFinished.connect(
                    lambda: self._field_edited(
                        {'op': 'field', 'at': self.selected, 'field': 'caption',
                         'value': self.caption_edit.text()},
                        block.get('caption', '') != self.caption_edit.text()))
                form.addRow('Caption', self.caption_edit)
        finally:
            self._loading_pane = False
        self.pane.setWidget(body)
        self._offer()

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
                    links: Callable[[], list[Any]] | None = None) -> None:
        self.report = report
        self.objects = objects
        self.links = links
        self.unit_system = unit_system
        self._scroll = 0
        self.selected = None
        self.rebuild()
        self._show_selection()

    def rebuild(self) -> None:
        from ..report import render_html

        if self.report is None:
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
                               links=self.links() if self.links else None,
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

    def _scroll_back(self) -> None:
        self.view.page().runJavaScript(f'window.scrollTo(0, {self._scroll_to});')

    def stand_down(self) -> None:
        """Leave the page with nothing for the view's destructor to wait on.

        A `QWebEngineView` destroyed while its page is live blocked the
        whole process: Chromium's teardown waits on the render process
        in a synchronous `mach_msg` call that never returned. That was
        the gate's "stall at 98 %" — sampled on 2026-09-18 with a
        worker's main thread parked inside QtWebEngineCore, and then
        named exactly by a faulthandler dump: the window fixture
        delivering the deferred delete to a window whose report page
        had *just finished* loading. Stopping a navigation in flight
        (the first cut of this) was not enough; a loaded, rendering
        page hangs the destructor just the same.

        The cure is Qt's own: a hidden page can be *discarded* — its
        render process shut down gracefully, the page unloaded — and
        a discarded page is destroyed in a millisecond (measured: the
        destructor went from a hang to 0.001 s). So the view is
        hidden, the page discarded, and the events that carry the
        change are pumped before the caller goes on to destroy
        anything. The scroll-restoring slot of a navigation in flight
        is dropped too, since it would fire into the discarded page.
        """
        from PySide6.QtWidgets import QApplication

        page = self.view.page()
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
        # the enum through the page rather than a QtWebEngineCore
        # import: the rulebook's sanctioned Qt modules are the ones
        # already imported, and the page carries its own states
        with contextlib.suppress(RuntimeError):
            page.setLifecycleState(type(page).LifecycleState.Discarded)
        _log.debug('stand_down: discarded')
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
        elif op == 'field':
            index = int(operation.get('at', -1))
            if 0 <= index < report.num_blocks:
                block = report.blocks[index]
                field = operation.get('field')
                value = str(operation.get('value', ''))
                if field == 'source':
                    key = ('geometry' if block.get('kind') == 'scene'
                           else 'source')
                    block[key] = value
                elif field in ('text', 'caption', 'shapes', 'photo',
                               'dofs_source', 'channel'):
                    block[field] = value
        else:
            return
        self.edited.emit()
        self.rebuild()
        if reselect:
            self._show_selection()
