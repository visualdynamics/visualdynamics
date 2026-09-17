"""The project tree.

Dropping files on the tree works, and it takes some doing.

The tree used to import what landed on it through `QAbstractItemView`'s
own drop machinery, which decides what a drop means from the item under
the cursor and the drag-drop mode — so the same file dragged onto the
same tree would import on the third go. That was replaced by having the
window answer drags for every part of itself at once, and the tree
refuse them, on the understanding that Qt walks up from a widget that
refuses to one that accepts.

It does walk up. But the tree lives in a dock widget inside a scroll
area, the walk starts at whichever child is under the cursor, and
whether the window is reached depends on where in that stack the drag
was resolved. Dropping on the tree still missed sometimes, while the
plot pane on the other side of the window never did.

So the tree accepts drags again — but it does not *handle* them. It
takes a drag carrying files, refuses everything else outright so
nothing else is swallowed, and hands the paths straight to the window's
own importer. `setDragDropMode(NoDragDrop)` keeps the view's item
machinery out of it entirely: these three overrides never call up to
it. One importer, reached two ways, neither of which depends on Qt
resolving a target through a dock.

Objects drag *within* the tree too, to move one between link groups,
and that rides on the same hand-rolled handling rather than on the item
machinery: a drag of our own carries a mime type of our own, and a drop
is read as the object row it lands on.

Every one of our drags is accepted, including ones the project will go
on to refuse, and this is the part that was wrong first. Refusing the
drag reads well — the move is simply not offered — but Qt takes an
ignored `dragMoveEvent` to mean the widget is not a drop target *there*,
so releasing the mouse produces no drop at all: no move, no message,
nothing. A geometry dragged onto a side that already had one behaved
exactly like a broken feature, because from the outside there is no
difference. So the drop always lands and the window always answers it,
and `landing_for` decides only whether the landing *line* is drawn —
the part that is genuinely about showing what applies.
"""

from __future__ import annotations

import atexit
import datetime
import pathlib
import shutil
import tempfile
from collections.abc import Callable, Sequence
from typing import Any

from PySide6.QtCore import QEvent, QMimeData, QPoint, Qt, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDrag,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

#: a drag of tree objects, told apart from anything else that lands here
OBJECT_MIME = 'application/x-visualdynamics-objects'
#: what `copy_selected` returns for the project row: the whole project
#: went on the clipboard as one file, and no object name would say so
PROJECT_ROW = '\x00project'

#: On every drag the tree starts, whether or not it names any objects.
#: A whole-project drag names none — the project row is not an object —
#: so it reaches a drop as nothing but a `.vdyn` file, indistinguishable
#: from the same file dragged in from Finder. Dropped back on the window
#: it imported itself, and every object in the project arrived a second
#: time. This marker is how a drag of ours is told from a real one; it
#: rides on the drag, never on the file, so that same `.vdyn` dragged in
#: from the desktop tomorrow is an ordinary import.
SELF_MIME = 'application/x-visualdynamics-from-this-tree'

#: marks a row as one of the project's objects — what a link group holds.
#: Sub-items are parts of an object and placeholders are not objects at
#: all, and no flag tells them apart: Qt gives every item
#: ItemIsDragEnabled by default, so reading that made a geometry's Nodes
#: draggable too.
ROLE_DRAGGABLE = Qt.ItemDataRole.UserRole + 9

#: on a project type's gray placeholder row, which side of the skeleton
#: it is a slot for. A slot is a drop target like an object row is: the
#: side a group belongs to is the one thing no type rule can derive, and
#: dropping onto the side is how it gets said.
ROLE_DROP_SLOT = Qt.ItemDataRole.UserRole + 10

#: the project's own row. It is not `ROLE_DRAGGABLE` — it cannot be
#: moved between link groups, which is what that role means — but it can
#: be dragged *out*, and doing so saves the whole project.
ROLE_WHOLE_PROJECT = Qt.ItemDataRole.UserRole + 11

#: The geometry everything unlinked is drawn on and checked against.
#: Bold was the only mark it wore, and against a tree where the real
#: objects are already darker than the gray slots, bold is not a mark at
#: all — nobody could tell which geometry was active by looking.
ROLE_ACTIVE = Qt.ItemDataRole.UserRole + 12

#: The bullet wears the geometry icon's own blue, from `icons.COLORS`.
#: Imported by value rather than by reference to keep this module free
#: of the icon drawing, which pulls in the whole palette.
ACTIVE_MARK = '#1f77b4'


TRACE = pathlib.Path.home() / 'Library' / 'Logs' / 'VisualDynamics-drops.log'


_LAST_TRACE: dict[str, float] = {}
#: the last accepted/refused answer each surface gave, so a flip is
#: always logged even mid-throttle
_LAST_ANSWER: dict[str, bool] = {}


def trace_drag(who: str, what: str, mime: QMimeData | None = None,
               throttle: bool = False) -> None:
    """One line per drag event, into a small log truncated at launch.

    Here because a drop that dies silently is undiagnosable from a
    description: the badge shows, nothing lands, and every stage of the
    chain — routing, accepting, delivering, importing — looks the same
    from the outside. A drop on a real window from a real Finder cannot
    be reproduced by tests, so when one misbehaves this log is the only
    witness. The cost is a few lines per drag, in a file that never
    outlives the session.
    """
    try:
        if throttle:
            import time
            now = time.monotonic()
            key = f'{who}:{what}'
            if now - _LAST_TRACE.get(key, 0.0) < 0.5:
                return
            _LAST_TRACE[key] = now
        detail = ''
        if mime is not None:
            formats = ','.join(mime.formats()[:6])
            detail = (f' urls={len(mime.urls())}'
                      f' formats=[{formats}]')
        stamp = datetime.datetime.now(  # noqa: DTZ005 — local
            ).strftime('%H:%M:%S.%f')[:-3]  # wall-clock is the point
        with open(TRACE, 'a', encoding='utf-8') as log:
            log.write(f'{stamp} {who}: {what}{detail}\n')
    except OSError:
        pass                    # a full disk must not break a drop


def start_trace() -> None:
    """Truncate the log; called once, when the window comes up.

    Deliberately *not* an application-wide event filter, though one
    would also catch a drop delivered to a widget nobody instrumented:
    a Python filter running for every event in a process that hosts
    QtWebEngine segfaulted inside Chromium's own machinery. The
    per-surface lines are enough — an accepted enter and accepted moves
    followed by no drop is a platform non-delivery, whoever it was not
    delivered to.
    """
    try:
        TRACE.parent.mkdir(parents=True, exist_ok=True)
        TRACE.write_text('')
    except OSError:
        pass


def dropped_files(mime: QMimeData) -> list[str]:
    """The local paths a drag is carrying, if it carries any.

    **Our own drags carry none**, however real the file they are
    offering. This is the one place that can say so: the tree ignoring
    the drop was not enough, because an ignored drop propagates to the
    window, which takes a file dropped anywhere on it — deliberately,
    since a fresh window's tree is a 90 px strip and missing it reads as
    the drop being ignored. Both ask here, so both get the same answer.
    """
    if not mime.hasUrls() or mime.hasFormat(SELF_MIME):
        return []
    return [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]


def dragged_objects(mime: QMimeData) -> list[str]:
    """The object names a drag of our own is carrying."""
    if not mime.hasFormat(OBJECT_MIME):
        return []
    raw = bytes(mime.data(OBJECT_MIME)).decode('utf-8')
    return [name for name in raw.split('\n') if name]


class _DraggedObjects(QMimeData):
    """The tree's drag payload: object names now, files only if asked.

    A project can be a hundred megabytes. Writing that at the *start* of
    every drag — including the ones that only reorder rows inside the
    tree — would stall the window every time an object is touched. So
    the `.vdyn` is written when a drop target actually asks for
    `text/uri-list`, which the tree itself never does: `_offer` and
    `dropEvent` ask for the object names first and stop there. Drag
    inside the tree and nothing is written at all; drop on the desktop
    and it is written once, at that moment.

    `formats` and `hasFormat` answer for the URL list without producing
    it, so hovering over anything — including this tree — stays free.
    """

    def __init__(self, names: Sequence[str],
                 write: Callable[[str], Sequence[str]] | None = None) -> None:
        super().__init__()
        self.setData(SELF_MIME, b'1')
        if names:
            self.setData(OBJECT_MIME, '\n'.join(names).encode('utf-8'))
        self._write = write
        self._written: list[str] | None = None

    def formats(self) -> list[str]:
        found = list(super().formats())
        if self._write is not None and 'text/uri-list' not in found:
            found.append('text/uri-list')
        return found

    def hasFormat(self, mimetype: str) -> bool:
        if mimetype == 'text/uri-list':
            return self._write is not None
        return super().hasFormat(mimetype)

    def _paths(self) -> list[str]:
        """Write the files, once, however many times we are asked."""
        if self._written is None:
            try:
                self._written = [str(p) for p in self._write(_drag_folder())]
            except Exception:  # noqa: BLE001 — a drag that cannot save
                # offers no file, and must not take the window down in
                # the middle of a gesture. Qt is calling this from
                # inside the drag loop, where a traceback goes nowhere.
                self._written = []
        return self._written

    def retrieveData(self, mimetype: str, preferred: Any) -> Any:
        """The URLs as **QUrl objects**, not as the wire format.

        Returning the encoded `text/uri-list` bytes here looks right and
        is not: `QMimeData.urls()` asks for a variant list, gets bytes,
        cannot convert, and answers empty — while `data()` answers
        perfectly, which is what makes it look like it works. macOS
        reads `urls()`, found nothing, and fell back to writing a
        *location* file named after the URL's host. Dragging an object
        to the desktop produced `localhost.fileloc`.

        Qt converts a list of QUrl into the wire format for any caller
        that wants bytes, so handing it the list serves both.
        """
        if mimetype == 'text/uri-list' and self._write is not None:
            return [QUrl.fromLocalFile(path) for path in self._paths()]
        return super().retrieveData(mimetype, preferred)


_DRAG_FOLDER: str | None = None


def _drag_folder() -> str:
    """Where dragged-out files are staged, made on first use.

    They have to survive the drag — the receiving application copies
    from this path after the drop — so they cannot be deleted when the
    gesture ends. The directory goes at exit instead.
    """
    global _DRAG_FOLDER
    if _DRAG_FOLDER is None:
        _DRAG_FOLDER = tempfile.mkdtemp(prefix='visualdynamics-drag-')
        atexit.register(shutil.rmtree, _DRAG_FOLDER, ignore_errors=True)
    return _DRAG_FOLDER


class ProjectTree(QTreeWidget):
    """Tree of imported objects.

    Linked objects are joined by a bracket painted down the left edge —
    `link_spans` is [(color, [items], bold)], maintained by the
    window; the bold flag marks the Basis group's bracket."""

    #: files were dropped on the tree; the window imports them
    files_dropped = Signal(list)
    #: a paste whose clipboard carries our own object names — the
    #: window decides what they mean here (a duplicate, or an import
    #: from the files the same clipboard can write)
    objects_pasted = Signal(list)
    #: objects were dragged onto another object (or onto nothing, which
    #: is None): the window moves them between link groups
    objects_moved = Signal(list, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # The mode first, then the flags, and the order is the whole
        # trick: setDragDropMode clears acceptDrops on the view and on
        # its viewport, so accepting before setting the mode accepts
        # nothing. The mode keeps the view's own item drop machinery out
        # of it; the flags are what get the events delivered here at
        # all. The viewport needs its own — that is the child actually
        # under the cursor.
        self.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        # NoDragDrop cleared this too, and starting a drag is the half of
        # it the item machinery may still do: what it must not do is
        # decide what a drop *means*.
        self.setDragEnabled(True)
        self.link_spans: list[Any] = []
        #: the window's answer to "where would this move land?", asked
        #: while the drag is still in the air: the items the dragged
        #: objects would sit above (None meaning the end), or None when
        #: the move would not happen. It decides the landing line, never
        #: whether the drop lands. Standalone, nothing is marked.
        self.landing_for: Callable[[list[str], Any], list | None] = (
            lambda names, target: None)
        self._drop_target = None
        self._drop_lines: list = []
        #: set by the window: (names or None for the whole project,
        #: folder) -> the paths written. None means dragging out of the
        #: window offers nothing, which is what a bare ProjectTree does.
        self.write_for_drag: Callable[
            [list[str] | None, str], Sequence[str]] | None = None

    def eventFilter(self, watched: Any, event: Any) -> bool:
        """Focus entering a record grid must not move the tree's current.

        `setItemWidget` makes this view an event filter on the widget,
        and QAbstractItemView's filter answers the widget's FocusIn by
        moving the current index to the widget's own row — the grid's
        holder, which is deliberately unselectable, so the move is a
        ClearAndSelect that selects nothing and clears everything.
        Any focus into a grid that a cell pick did not follow — a
        click on its header, its margins, the space right of its last
        column — silently dropped every other object's picks: a record
        chosen in one FRF's grid vanished when a click aimed at
        another grid's first row landed a few pixels high (Brandon,
        2026-08-30; reproduced with setFocus() alone). The grid keeps
        the focus; the tree keeps the selection.
        """
        from .record_grid import RecordGrid

        if (event.type() == QEvent.Type.FocusIn
                and isinstance(watched, RecordGrid)):
            return False
        return super().eventFilter(watched, event)

    # ---- dragging objects between groups -----------------------------------

    def draggable(self, item: QTreeWidgetItem | None) -> bool:
        """Whether an item is one of the project's objects. Sub-items —
        a geometry's Nodes, a record inside a data array — are parts of
        an object, and an object is what a link group holds."""
        return item is not None and bool(item.data(0, ROLE_DRAGGABLE))

    def _target_at(self, position):
        """What a drop here would land on: an object name, `('slot',
        tag)` for one of the project type's empty sides, or None.

        A drop on a sub-item means its object: the row a link bracket
        reaches is the top of that subtree, and asking the user to hit it
        exactly would be a precision game with no purpose. A drop on a
        bracket means the group it draws — an object of it where it has
        one, and the side it stands for where it is only gray slots.
        """
        span = self.span_at(position)
        if span is not None:
            return self._target_of(self.link_spans[span][1])
        item = self.itemAt(position)
        while item is not None and not (self.draggable(item)
                                        or item.data(0, ROLE_DROP_SLOT)):
            item = item.parent()
        return self._target_of([item])

    def _target_of(self, items):
        """An object among these rows, else the side their slots are
        for. An object wins: a side that already holds a group is joined
        by joining the group, and there is then only one thing to say."""
        items = [item for item in items if item is not None]
        named = next((item for item in items if self.draggable(item)), None)
        if named is not None:
            return named.text(0)
        tags = [item.data(0, ROLE_DROP_SLOT) for item in items]
        tag = next((tag for tag in tags if tag), None)
        return None if tag is None else ('slot', tag)

    def startDrag(self, actions: Qt.DropAction) -> None:
        """One drag, two readers.

        Inside the tree it carries `OBJECT_MIME` and moves objects
        between link groups, exactly as before. Outside — Finder,
        Explorer, a file manager — it carries `text/uri-list` and
        becomes a `.vdyn` saved wherever it lands. `_offer` and
        `dropEvent` both ask for the object names first, so an internal
        drop never sees the file half.

        Copy *and* Move are offered, defaulting to Move: Finder refuses
        a move-only drag outright, and Move is what a drop inside the
        tree means.
        """
        payload = self._selection_payload()
        if payload is None:
            return
        mime, _names, _whole = payload
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
                  Qt.DropAction.MoveAction)

    # ---- the clipboard ------------------------------------------------------

    def copy_selected(self) -> list[str]:
        """Copy (Cmd/Ctrl+C): the selection onto the clipboard, as the
        same two-faced payload a drag carries — object names for a
        paste back into a tree, and `.vdyn` files for a paste into
        Finder or Explorer, written only when a file manager actually
        asks (Brandon, 2026-09-03). Returns the names copied — or
        `[PROJECT_ROW]` for the project row, which copies the whole
        project as one file — and an empty list when nothing was
        selected, in which case the clipboard is left as it was."""
        payload = self._selection_payload()
        if payload is None:
            return []
        mime, names, _whole = payload
        QApplication.clipboard().setMimeData(mime)
        return names or [PROJECT_ROW]

    def _selection_payload(self):
        """The selection as the two-faced payload a drag and a copy
        both carry — `(mime, names, whole)`, or None when nothing
        draggable is selected. `whole` is the project row, which
        writes the whole project as one file."""
        current = self.currentItem()
        whole = bool(current is not None
                     and current.data(0, ROLE_WHOLE_PROJECT)
                     and current.isSelected())
        names = [item.text(0) for item in self.selectedItems()
                 if self.draggable(item)]
        if not names and not whole:
            return None
        mime = _DraggedObjects(
            names, None if self.write_for_drag is None
            else (lambda folder: self.write_for_drag(
                None if whole else names, folder)))
        return mime, names, whole

    def paste(self) -> bool:
        """Paste (Cmd/Ctrl+V): our own object names go to the window as
        `objects_pasted`; files from anywhere — a file manager, another
        window — arrive the way a drop of files does. Returns whether
        the clipboard held anything we read."""
        mime = QApplication.clipboard().mimeData()
        if mime is None:
            return False
        names = dragged_objects(mime)
        if names:
            self.objects_pasted.emit(names)
            return True
        paths = [url.toLocalFile() for url in mime.urls()
                 if url.isLocalFile()] if mime.hasUrls() else []
        if paths:
            self.files_dropped.emit(paths)
            return True
        return False

    # ---- drops ------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        self._offer(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        self._offer(event)

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        trace_drag('tree', 'leave')
        self._aim(None)
        super().dragLeaveEvent(event)

    def _offer(self, event):
        """Take a drag of files or of our own objects; leave every other
        drag alone.

        Ignoring rather than accepting is what lets anything else — a
        drag between two of visualdynamics's own widgets, or one from another
        application — carry on to whoever it was meant for.
        """
        names = dragged_objects(event.mimeData())
        if names:
            target = self._target_at(event.position().toPoint())
            # marked only where the move would happen; accepted either
            # way, or the drop never arrives to say why it would not
            landings = self.landing_for(names, target)
            self._aim(target if landings is not None else None,
                      landings or [])
            event.setDropAction(Qt.DropAction.MoveAction)
            event.acceptProposedAction()
        elif dropped_files(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.acceptProposedAction()
        else:
            event.ignore()
        kind = 'enter' if isinstance(event, QDragEnterEvent) else 'move'
        # Moves are throttled, EXCEPT when the answer changes — Cocoa
        # decides the drop from the *last* move before release, which is
        # exactly the move a plain throttle swallows. A drag that was
        # accepted all the way down and then refused once at the end
        # delivers no drop, with no other symptom at all; the transition
        # line is the only trace such a refusal leaves.
        accepted = event.isAccepted()
        flipped = accepted != _LAST_ANSWER.get('tree')
        _LAST_ANSWER['tree'] = accepted
        trace_drag('tree', f'{kind} accepted={accepted}',
                   event.mimeData(),
                   throttle=kind == 'move' and not flipped)

    def _aim(self, target, landings=()):
        """Mark where a drop would land, and repaint if that moved."""
        if target != self._drop_target or list(landings) != self._drop_lines:
            self._drop_target = target
            self._drop_lines = list(landings)
            self.viewport().update()

    def dropEvent(self, event: QDropEvent) -> None:
        names = dragged_objects(event.mimeData())
        if names:
            target = self._target_at(event.position().toPoint())
            self._aim(None)
            event.setDropAction(Qt.DropAction.MoveAction)
            event.acceptProposedAction()
            self.objects_moved.emit(names, target)
            return
        paths = dropped_files(event.mimeData())
        trace_drag('tree', f'drop paths={len(paths)}', event.mimeData())
        if not paths:
            event.ignore()
            return
        event.setDropAction(Qt.DropAction.CopyAction)
        event.acceptProposedAction()
        self.files_dropped.emit(paths)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_active(painter)
        self._paint_drop_target(painter)
        for offset, (color, items, bold) in enumerate(self.link_spans):
            rects = [self.visualItemRect(item) for item in items
                     if item is not None]
            rects = [rect for rect in rects if rect.height() > 0]
            if len(rects) < 2:
                continue
            x = 3 + 3 * (offset % 2)    # neighboring groups interleave
            # the Basis bracket is bold as well as blue, so it still
            # stands out without relying on color alone
            painter.setPen(QPen(QColor(color), 4 if bold else 2))
            top = min(rect.center().y() for rect in rects)
            bottom = max(rect.center().y() for rect in rects)
            painter.drawLine(x, top, x, bottom)
            for rect in rects:
                painter.drawLine(x, rect.center().y(),
                                 x + 5, rect.center().y())
        painter.end()

    def _paint_active(self, painter):
        """A bullet marking the active geometry, in a column of its own.

        Drawn rather than put in the row, because the row's text is the
        object's name and anything prepended to it would be carried
        into a rename, and because a 16 px icon has nothing spare
        inside it.

        It sits in the gap between the branch arrow and the icon,
        which is narrow: a full-sized dot there touched the collapsed
        triangle and the two read as one glyph, a little flag. Hence a
        smaller dot, centered in what room there is. Both edges are
        measured rather than written down, because the indentation and
        the arrow are a style's to choose and neither is the same width
        on every platform.

        The color is the geometry icon's own blue — the mark and the
        thing it marks are one statement, and a second color in a tree
        that already says a great deal with color would be a third
        vocabulary to learn.

        Further left is not available. That is the link brackets'
        gutter, and `span_at` reads a click there as a click on a group.
        """
        marked = [item for item in self._every_item()
                  if item.data(0, ROLE_ACTIVE)]
        if not marked:
            return
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(ACTIVE_MARK))
        for item in marked:
            rect = self.visualItemRect(item)
            if rect.height() <= 0:
                continue
            # hard against the icon, which starts a few px inside the
            # rect: the arrow's tip reaches further right than its
            # bounding box suggests, and a dot near it read as attached
            painter.drawEllipse(QPoint(rect.left() + 1, rect.center().y()),
                                2, 2)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_drop_target(self, painter):
        """Where the drag would land, if it would land anywhere: a line
        in the gap the canonical order will actually put the object.

        This replaced an outline around the target — a box around a row
        read as the object going *into* it, when a drop only ever makes
        the object a sibling above or below (Brandon, 2026-08-30). One
        line per landing: an entry names the row the object lands
        above; None means after the last object row.
        """
        if self._drop_target is None or not self._drop_lines:
            return
        painter.setPen(QPen(QColor('#4c8dff'), 2))
        right = self.viewport().width() - 2
        for below in self._drop_lines:
            if below is not None:
                rect = self.visualItemRect(below)
                if rect.height() <= 0:
                    continue
                left, y = rect.left(), rect.top()
            else:
                rects = [self.visualItemRect(item)
                         for item in self._every_item()
                         if self.draggable(item)]
                rects = [rect for rect in rects if rect.height() > 0]
                if not rects:
                    continue
                left = min(rect.left() for rect in rects)
                y = max(rect.bottom() for rect in rects) + 1
            painter.drawLine(left, y, right, y)

    def _every_item(self):
        root = self.invisibleRootItem()
        stack = [root.child(i) for i in range(root.childCount())]
        while stack:
            item = stack.pop()
            yield item
            stack += [item.child(i) for i in range(item.childCount())]

    def span_at(self, position: QPoint) -> int | None:
        """Index of the link bracket under a viewport position, or
        None — how a right-click lands on a group rather than on the
        object beside it."""
        if position.x() > 14:
            return None
        for index, (_color, items, _bold) in enumerate(self.link_spans):
            rects = [self.visualItemRect(item) for item in items
                     if item is not None]
            rects = [rect for rect in rects if rect.height() > 0]
            if len(rects) < 2:
                continue
            top = min(rect.center().y() for rect in rects)
            bottom = max(rect.center().y() for rect in rects)
            if top - 4 <= position.y() <= bottom + 4:
                return index
        return None
