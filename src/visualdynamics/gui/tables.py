"""Tables that behave like a spreadsheet.

The tables are model/view, not a widget per cell, so a table costs
nothing until it is scrolled, an edit is validated and written straight
back to the visualdynamics object under it, and switching display units
restates the values without rebuilding anything.

`CopyPasteTableView` uses Excel's clipboard format (tab-separated cells,
one row per line), so a selection copies straight into a spreadsheet and
a block of spreadsheet cells pastes back in. It also has Batch Edit,
Clear, a fill handle, a copy button for the whole table, and Delete for
whole rows where its owner allows it.
"""

from __future__ import annotations

import html
import pathlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import (
    QAbstractTableModel,
    QItemSelectionModel,
    QMimeData,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QRect,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QCursor,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QInputDialog,
    QLabel,
    QMenu,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionComboBox,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .icons import control_icon

NO_PARENT = QModelIndex()      # Qt's default parent for a flat table

FILL_HANDLE = 9                # the square you drag to fill, in pixels
FILL_GRAB = 4                  # slack around it, so it is not a pixel hunt


@dataclass
class Column:
    """One column of a `TableModel`: how to read it, how to show it, and
    whether it can be set."""

    title: str
    get: Callable                      # (obj, row) -> value
    set: Callable | None = None        # (obj, row, text) -> None, raises on bad
    format: Callable | None = None     # (value) -> str
    decoration: Callable | None = None  # (obj, row) -> QColor shown in the cell
    background: Callable | None = None  # (obj, row) -> QColor behind the cell
    choices: list | None = None        # the only accepted values, if limited
    choice_icon: Callable | None = None  # (choice) -> QIcon for the drop-down
    row_choices: Callable | None = None  # (obj, row) -> list, per cell
    choices_editable: bool = False     # the list suggests rather than limits
    #: drawn as a check box: `get` answers truthiness, clicking toggles,
    #: and the text side ('True'/'False') still rides copy and paste
    checkbox: bool = False
    #: edited with a calendar. The cell still holds ISO text, so typing
    #: and pasting are unaffected — the picker is an easier way in, not
    #: a different value.
    date: bool = False
    affects_row: bool = False          # setting it can restate its neighbors
    #: titles of columns that must fill along when this one is filled
    #: down — a unit dragged over rows carries its Type, because a unit
    #: without its quantity is half a statement (Brandon, 2026-08-30:
    #: filling V down a table of force channels left them force-in-Volts)
    carries: tuple = ()
    #: (obj, row, text) -> the replay line's suffix after `project[name]`
    #: — '.set_cell(...)', '.node_xyz[3, 0] = 0.1' — run AFTER a
    #: successful set, so a post-state echo reads the converted value.
    #: None on a read-only column; a column journaled elsewhere (the
    #: units pane) declares a closure returning None. The meta-test in
    #: test_table_conventions holds every editable column to this.
    journal: Callable | None = None
    alignment: int = int(Qt.AlignmentFlag.AlignRight
                         | Qt.AlignmentFlag.AlignVCenter)

    @property
    def editable(self) -> bool:
        """Whether the column can be set, which is whether it has a
        `set`."""
        return self.set is not None

    def choices_for(self, obj: Any, row: int) -> list | None:
        """What one cell offers, which can be narrower than the column's
        list.

        A channel the file said was an acceleration offers the
        acceleration units, not every unit, while the channel below it
        offers its own.
        """
        if self.row_choices is not None:
            return self.row_choices(obj, row)
        return self.choices

    def text(self, obj: Any, row: int) -> str:
        """The cell's text: `format` applied to the value when the column
        has one, otherwise a float to six significant figures and '' for
        None."""
        value = self.get(obj, row)
        if self.format is not None:
            return self.format(value)
        if isinstance(value, float):
            return f'{value:.6g}'
        return '' if value is None else str(value)


class TableModel(QAbstractTableModel):
    """A table over a visualdynamics object, described by a list of
    `Column` objects.

    `edit_rejected` carries the reason a cell refused a value. For each
    successful edit of a journaled column, `edit_journaled` carries the
    replay line's suffix; the window puts `project[name]` in front of it
    and appends it to the journal.
    """

    edit_rejected = Signal(str)         # why a cell refused a value
    #: a successful edit's replay-line suffix — the window prepends
    #: `project[name]` and appends it to the session journal
    edit_journaled = Signal(str)

    def __init__(self, obj: Any, columns: Sequence[Column],
                 row_count: Callable[[Any], int],
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.obj: Any = obj
        self.columns: list[Column] = list(columns)
        self._row_count = row_count

    # ---- Qt interface -------------------------------------------------------

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = NO_PARENT) -> int:
        return 0 if parent.isValid() else self._row_count(self.obj)

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = NO_PARENT) -> int:
        return 0 if parent.isValid() else len(self.columns)

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.columns[section].title
        return section + 1

    def data(self, index: QModelIndex | QPersistentModelIndex,
             role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        column = self.columns[index.column()]
        if column.checkbox:
            # the box is the display; the text side stays on EditRole so
            # copy and paste still speak 'True'/'False'
            if role == Qt.ItemDataRole.CheckStateRole:
                return (Qt.CheckState.Checked
                        if column.get(self.obj, index.row())
                        else Qt.CheckState.Unchecked)
            if role == Qt.ItemDataRole.DisplayRole:
                return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return column.text(self.obj, index.row())
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return column.alignment
        if (role == Qt.ItemDataRole.DecorationRole
                and column.decoration is not None):
            return column.decoration(self.obj, index.row())
        if (role == Qt.ItemDataRole.BackgroundRole
                and column.background is not None):
            return column.background(self.obj, index.row())
        return None

    def flags(self, index: QModelIndex | QPersistentModelIndex) -> Qt.ItemFlag:
        flags = (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        if index.isValid() and self.columns[index.column()].editable:
            flags |= Qt.ItemFlag.ItemIsEditable
            if self.columns[index.column()].checkbox:
                flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def setData(self, index: QModelIndex | QPersistentModelIndex, value: Any,
                role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid():
            return False
        if (role == Qt.ItemDataRole.CheckStateRole
                and self.columns[index.column()].checkbox):
            # a click on the box is the same statement as typing the word
            value = ('True' if Qt.CheckState(value) == Qt.CheckState.Checked
                     else 'False')
            role = Qt.ItemDataRole.EditRole
        if role != Qt.ItemDataRole.EditRole:
            return False
        column = self.columns[index.column()]
        if not column.editable:
            return False
        try:
            column.set(self.obj, index.row(), str(value))
        except (ValueError, KeyError, IndexError) as e:
            # the cell reverts; say why, or the refusal looks like a bug
            self.edit_rejected.emit(str(e))
            return False
        if column.affects_row:
            # changing a channel's quantity can clear the unit beside it
            self.dataChanged.emit(self.index(index.row(), 0),
                                  self.index(index.row(), self.columnCount() - 1))
        else:
            self.dataChanged.emit(index, index)
        self._journal_edit(column, index.row(), str(value))
        return True

    # ---- helpers ------------------------------------------------------------

    def refresh_row(self, row: int) -> None:
        """Redraw one row, leaving the selection alone.

        A full reset clears the selection, which matters when the
        selection is what the edit applies to: turning a coordinate system
        would lose the system being turned.
        """
        self.dataChanged.emit(self.index(row, 0),
                              self.index(row, self.columnCount() - 1))

    def _journal_edit(self, column, row, text):
        if column.journal is None:
            return
        suffix = column.journal(self.obj, row, text)
        if suffix:
            self.edit_journaled.emit(suffix)

    def set_cells(self, cells: Sequence[tuple[int, int, str]]
                  ) -> tuple[int, int, str]:
        """Write a different value into each of many cells, as one change,
        and return (applied, rejected, reason).

        `cells` holds (row, column, text) for each cell. `set_many` is
        this with one value for every cell. One `dataChanged` covers all
        of them: setData cell by cell would emit once per cell, and each
        emission redraws the 3-D view, so setting a column of a few
        thousand nodes costs one redraw and not a few thousand.

        Cells that refuse their value are left as they are and counted,
        so a batch that only partly lands is not mistaken for a clean
        one, and the reason for the first refusal is returned with the
        counts.
        """
        applied = rejected = 0
        reason = ''
        rows, columns = [], []
        for row, column_index, text in cells:
            column = self.columns[column_index]
            if not column.editable:
                rejected += 1
                continue
            try:
                column.set(self.obj, row, text)
            except (ValueError, KeyError, IndexError) as e:
                rejected += 1
                reason = reason or str(e)
                continue
            self._journal_edit(column, row, text)
            applied += 1
            rows.append(row)
            columns.append(column_index)
        if applied:
            self.dataChanged.emit(self.index(min(rows), min(columns)),
                                  self.index(max(rows), max(columns)))
        return applied, rejected, reason

    def set_many(self, indexes: Sequence[QModelIndex],
                 text: str) -> tuple[int, int, str]:
        """Write one value into many cells as one change, and return
        (applied, rejected, reason): `set_cells` with the same text for
        every cell."""
        return self.set_cells((index.row(), index.column(), text)
                              for index in indexes)


class ChoiceDialog(QDialog):
    """Pick one of a column's choices, shown the way its cells show them.

    Batch editing a column of colors offers the same list of colors as
    editing one cell of it, not a box to type a color name into.
    """

    def __init__(self, parent: QWidget | None, prompt: str,
                 column: Column) -> None:
        super().__init__(parent)
        self.setWindowTitle('Batch Edit')
        self.combo: QComboBox = QComboBox()
        for choice in column.choices:
            if column.choice_icon is not None:
                self.combo.addItem(column.choice_icon(choice), str(choice))
            else:
                self.combo.addItem(str(choice))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(prompt))
        layout.addWidget(self.combo)
        layout.addWidget(buttons)

    def value(self) -> str:
        """The choice picked, as text."""
        return self.combo.currentText()


import time as _time

_UNITS_TRACE = (pathlib.Path.home() / 'Library' / 'Logs'
                / 'VisualDynamics-units.log')


def trace_units(what: str) -> None:
    """Write one line to the units-editor log for an event in a choice
    cell's drop-down, with the mouse buttons, the open popup and the
    focused widget at that moment.

    The log is ~/Library/Logs/VisualDynamics-units.log, overwritten by
    the first line each run writes. A failure to write is ignored.
    """
    # A witness log, like the drop log: the drop-down that closed the
    # moment it opened could not be reproduced by synthetic clicks, so
    # the real click writes its own record.
    try:
        from PySide6.QtWidgets import QApplication

        mode = 'a' if getattr(trace_units, '_open', False) else 'w'
        trace_units._open = True
        with open(_UNITS_TRACE, mode, encoding='utf-8') as log:
            focus = QApplication.focusWidget()
            log.write(f'{_time.strftime("%H:%M:%S")} {what} '
                      f'[buttons={QApplication.mouseButtons()!r} '
                      f'popup={type(QApplication.activePopupWidget()).__name__} '
                      f'focus={type(focus).__name__}]\n')
    except Exception:  # noqa: BLE001, S110 — a witness line must never
        pass               # be the thing that breaks the editor


def _open_list(editor: QComboBox) -> None:
    """Open a choice cell's list, and disown the choice that opening
    it reports.

    On macOS `QComboBox.showPopup()` emits `activated` *from inside
    itself*, naming whatever entry the list opened on — no click, no
    key, nothing the person did. Caught by stack trace (2026-09-01)
    with not one mouse event in the process: showPopup -> activated ->
    the delegate commits and closes the editor, and the drop-down is
    gone in the same breath it appeared. The list itself is innocent
    and stays up; it was this delegate that tore it down.

    So the editor is marked while it opens, and an activation arriving
    in that window is ignored. A person's pick lands afterwards and
    commits as it always did.
    """
    try:
        alive = editor.isVisible()
    except RuntimeError:
        return                        # the editor closed while we waited
    if not alive:
        return
    trace_units('showPopup')
    editor.setProperty('vd_opening', True)
    try:
        editor.showPopup()
    finally:
        editor.setProperty('vd_opening', False)
    trace_units('list open: ' + str(_list_showing(editor)))


def _list_showing(editor: QComboBox) -> bool:
    try:
        return editor.view().isVisible()
    except RuntimeError:
        return False


#: room beside the longest entry in an opened list: the icon column and
#: the scroll bar
LIST_PADDING = 48


class ChoiceDelegate(QStyledItemDelegate):
    """Edit a column that declares `choices` with a drop-down, and a date
    column with a calendar.

    Opening the cell opens the list, because a closed combo box would
    take a second click to show the choices. Picking one applies it and
    closes the editor, without waiting for the focus to move. Typing
    still works, since the model accepts the same values either way, so
    pasting a column of them from a spreadsheet is unaffected.
    """

    def _date_editor(self, parent: QWidget,
                     index: QModelIndex | QPersistentModelIndex) -> QWidget:
        """A calendar, seeded from whatever the cell already holds.

        A blank cell opens on today rather than on Qt's year 2000, which
        is a long way to scroll from a calibration date.
        """
        from PySide6.QtCore import QDate
        from PySide6.QtWidgets import QDateEdit

        editor = QDateEdit(parent)
        editor.setCalendarPopup(True)
        editor.setDisplayFormat('yyyy-MM-dd')
        text = str(index.model().data(index, Qt.ItemDataRole.EditRole) or '')
        stated = QDate.fromString(text, 'yyyy-MM-dd')
        editor.setDate(stated if stated.isValid() else QDate.currentDate())
        return editor

    def _column(self, index: QModelIndex | QPersistentModelIndex) -> Column | None:
        columns = getattr(index.model(), 'columns', None)
        return columns[index.column()] if columns else None

    def _choices(self, index: QModelIndex | QPersistentModelIndex
                 ) -> tuple[Column | None, list | None]:
        column = self._column(index)
        if column is None:
            return None, None
        return column, column.choices_for(getattr(index.model(), 'obj', None),
                                          index.row())

    def createEditor(self, parent: QWidget, option: Any,
                     index: QModelIndex | QPersistentModelIndex) -> QWidget:
        column, choices = self._choices(index)
        if column is not None and column.date:
            return self._date_editor(parent, index)
        if not choices:
            return super().createEditor(parent, option, index)
        editor = QComboBox(parent)
        # a narrowed list is a shortlist, not a rule: any unit visualdynamics can
        # parse is still allowed, and pasting one in must keep working
        editor.setEditable(column.choices_editable)
        for choice in choices:
            if column.choice_icon is not None:
                editor.addItem(column.choice_icon(choice), str(choice))
            else:
                editor.addItem(str(choice))
        # the list opens as wide as its longest entry, not as wide as the
        # cell: a material name cut off in a narrow column could not be
        # read in the drop-down either (Brandon, 2026-10-02)
        metrics = editor.fontMetrics()
        widest = max((metrics.horizontalAdvance(editor.itemText(i))
                      for i in range(editor.count())), default=0)
        editor.view().setMinimumWidth(widest + LIST_PADDING)
        editor.activated.connect(lambda _index: self._finish(editor))
        trace_units(f'createEditor: combo, {editor.count()} choices')
        editor.destroyed.connect(lambda: trace_units('editor destroyed'))
        return editor

    def _finish(self, editor: QWidget) -> None:
        if editor.property('vd_opening'):
            # showPopup's own doing, not the person's — see _open_list
            trace_units('activation while opening: ignored')
            return
        trace_units('activated -> commit+close')
        self.commitData.emit(editor)
        self.closeEditor.emit(editor)

    def setEditorData(self, editor: QWidget, index: QModelIndex | QPersistentModelIndex) -> None:
        if not isinstance(editor, QComboBox):
            return super().setEditorData(editor, index)
        text = '' if (text := index.data(Qt.ItemDataRole.EditRole)) is None \
            else str(text)
        if editor.isEditable():
            # findText would miss a value outside the shortlist and fall back
            # to the first entry, quietly changing the cell just by opening it
            editor.setCurrentText(text)
        else:
            editor.setCurrentIndex(max(0, editor.findText(text)))
        # After the *click*, not merely after this event. A zero timer
        # fires on the next event pass, which can land between the
        # opening click's press and its release — the release then
        # arrives on the fresh combo and toggles the popup straight
        # back closed. Whether it does is a timing race: light sessions
        # won, and a 27k-element CAD geometry slowed the loop enough to
        # lose it (Brandon, 2026-09-01, the motor's units drop-down).
        # So the popup waits for the button to actually be up.
        QTimer.singleShot(0, lambda: _open_list(editor))

    def setModelData(self, editor: QWidget, model: Any,
                     index: QModelIndex | QPersistentModelIndex) -> None:
        from PySide6.QtWidgets import QDateEdit

        if isinstance(editor, QDateEdit):
            # the cell holds ISO text either way; the calendar is a way
            # in, not a different value
            model.setData(index, editor.date().toString('yyyy-MM-dd'),
                          Qt.ItemDataRole.EditRole)
            return None
        if not isinstance(editor, QComboBox):
            return super().setModelData(editor, model, index)
        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)
        return None

    def paint(self, painter: QPainter, option: Any,
              index: QModelIndex | QPersistentModelIndex) -> None:
        """Mark the selected cell of a choice column with a drop-down arrow.

        Otherwise nothing shows that a cell has a list behind it until
        one has been opened. Only the selected cell is marked: an arrow in
        every cell of a 200k-row column is noise, and a spreadsheet marks
        the active cell the same way.

        The arrow is the platform's own, drawn by the style as a combo
        box's arrow sub-control, so it is what a real drop-down would draw
        here.
        """
        super().paint(painter, option, index)
        column = self._column(index)
        if column is None or not column.choices_for(
                getattr(index.model(), 'obj', None), index.row()):
            return
        if not option.state & QStyle.StateFlag.State_Selected:
            return
        combo = QStyleOptionComboBox()
        # the whole cell, so the style lays the arrow out where it would put
        # one; a cropped rect gets a squashed arrow on macOS
        combo.rect = option.rect
        combo.state = (QStyle.StateFlag.State_Enabled
                       | QStyle.StateFlag.State_Active)
        combo.subControls = QStyle.SubControl.SC_ComboBoxArrow
        combo.editable = column.choices_editable
        style = option.widget.style() if option.widget else QApplication.style()
        painter.save()
        painter.setClipRect(option.rect)
        style.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, combo,
                                 painter, option.widget)
        painter.restore()


class CopyPasteTableView(QTableView):
    """A table view that behaves like a spreadsheet: copy and paste as
    tab-separated text, Batch Edit, Clear, a fill handle, and a copy
    button for the whole table.

    `edits_applied` reports (cells written, cells refused, the first
    refusal's reason) after a batch edit, a clear or a fill.
    `rows_deleted` carries the whole rows a person asked to remove, and
    is emitted only when the owner sets `rows_deletable`, because only
    the owner knows whether a row is an item (a channel, a mode) or just
    cells. `whole_table_copy` is False where the table is a working
    surface, as in the geometry editor, and hides the whole-table copy.
    """

    # cells written, cells refused, why the first refusal happened
    edits_applied = Signal(int, int, str)
    # whole rows the user asked to remove; only emitted when `rows_deletable`
    # was set by whoever owns the table, because only the owner knows whether
    # a row *is* something — a channel, a mode — or just cells
    rows_deleted = Signal(list)
    rows_deletable = False

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # off while editing geometry, where the table is a working surface
        # rather than something to lift wholesale into a spreadsheet
        self.whole_table_copy: bool = True
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectItems)
        self.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)
        # SelectedClicked: clicking a cell that is already current opens it,
        # so a drop-down is one click away once its arrow is showing
        self.setEditTriggers(QTableView.EditTrigger.DoubleClicked
                             | QTableView.EditTrigger.SelectedClicked
                             | QTableView.EditTrigger.EditKeyPressed
                             | QTableView.EditTrigger.AnyKeyPressed)
        self.setItemDelegate(ChoiceDelegate(self))
        self._filling = None       # (rows, columns) being dragged down from
        self._fill_to = None       # the row the cursor is over, while dragging
        self.viewport().setMouseTracking(True)   # to show the drag cursor
        # the copy button in the corner, shown while the pointer is over
        # the table — the same button every figure and report table
        # carries (Brandon, 2026-09-24)
        self._copy_button: QToolButton = QToolButton(self)
        self._copy_button.setIcon(control_icon('copy'))
        self._copy_button.setToolTip(
            'Copy the whole table — as cells for a spreadsheet, as a '
            'table for a document or a mail')
        self._copy_button.setAutoRaise(True)
        self._copy_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_button.clicked.connect(self.copy_table)
        self._copy_button.hide()

    # ---- a subset of the rows -----------------------------------------------

    def show_rows(self, rows: Iterable[int] | None) -> None:
        """Show only these rows of the model; None shows them all.

        The other rows are out of the table, not only out of sight. They
        cannot be selected (`selectionChanged` deselects them, so Select
        All and a drag across the gap reach only what is shown, and the
        owner's actions on a selection, such as a highlight or Delete,
        never act on a row nobody can see), and copy, paste and the fill
        handle skip them. Rows keep their model indices, so code that
        reads the selection still reads the model's own rows.
        """
        # 2026-10-10: the pencil on an element group opened every element
        # group.
        self._shown_rows = None if rows is None else {int(r) for r in rows}
        model = self.model()
        count = model.rowCount() if model is not None else 0
        for row in range(count):
            self.setRowHidden(row, not self.row_shown(row))
        selection = self.selectionModel()
        if selection is not None and self._shown_rows is not None:
            self._drop_hidden(selection.selection())

    def row_shown(self, row: int) -> bool:
        """Whether `row` is shown (see `show_rows`)."""
        shown = getattr(self, '_shown_rows', None)
        return shown is None or row in shown

    def setModel(self, model: Any) -> None:
        # a new model is a new table: every row of it is shown until its
        # owner says otherwise
        self._shown_rows = None
        super().setModel(model)
        if model is not None:
            for row in range(model.rowCount()):
                self.setRowHidden(row, False)

    def selectionChanged(self, selected: Any, deselected: Any) -> None:
        super().selectionChanged(selected, deselected)
        if getattr(self, '_shown_rows', None) is not None:
            self._drop_hidden(selected)

    def _drop_hidden(self, selection: Any) -> None:
        hidden = sorted({index.row() for index in selection.indexes()
                         if not self.row_shown(index.row())})
        if not hidden:
            return
        model, chosen = self.model(), self.selectionModel()
        for row in hidden:
            chosen.select(model.index(row, 0),
                          QItemSelectionModel.SelectionFlag.Deselect
                          | QItemSelectionModel.SelectionFlag.Rows)

    def _shown(self, rows: Iterable[int]) -> list[int]:
        return [row for row in rows if self.row_shown(row)]

    # ---- clipboard ----------------------------------------------------------

    def selected_block(self) -> tuple[list[int], list[int]]:
        """(rows, columns) covered by the selection, as sorted lists."""
        indexes = self.selectedIndexes()
        if not indexes:
            return [], []
        rows = sorted({index.row() for index in indexes})
        columns = sorted({index.column() for index in indexes})
        return rows, columns

    def to_tsv(self, whole_table: bool = False) -> str:
        """The selection, or every shown row with `whole_table`, as
        tab-separated text without headers."""
        model = self.model()
        if model is None:
            return ''
        if whole_table:
            rows = self._shown(range(model.rowCount()))
            columns = range(model.columnCount())
        else:
            rows, columns = self.selected_block()
            if not rows:
                return ''
        lines = []
        for row in rows:
            # EditRole, not Display: the two agree everywhere except a
            # checkbox cell, whose display is the box and whose text —
            # what a spreadsheet paste wants — lives on the edit side
            lines.append('\t'.join(
                str(model.data(model.index(row, column),
                               Qt.ItemDataRole.EditRole) or '')
                for column in columns))
        return '\n'.join(lines)

    def copy(self, whole_table: bool = False) -> str:
        """Put `to_tsv` on the clipboard, and return it."""
        text = self.to_tsv(whole_table)
        if text:
            QApplication.clipboard().setText(text)
        return text

    def table_cells(self) -> tuple[list[str], list[list[str]]]:
        """(headers, rows) of the whole table's shown rows, as text."""
        model = self.model()
        if model is None:
            return [], []
        columns = range(model.columnCount())
        headers = [str(model.headerData(c, Qt.Orientation.Horizontal,
                                        Qt.ItemDataRole.DisplayRole) or '')
                   for c in columns]
        rows = [[str(model.data(model.index(r, c), Qt.ItemDataRole.EditRole)
                     or '') for c in columns]
                for r in self._shown(range(model.rowCount()))]
        return headers, rows

    def copy_table(self) -> bool:
        """Copy the whole table with its headers, as tab-separated text
        and as HTML in one clipboard entry, so a spreadsheet pastes cells
        and a document or an email pastes a table. Returns False when
        there are no rows to copy.

        This is the copy button in the corner. A copy of a selection
        stays bare cells, which is what pasting back into a sheet needs.
        """
        # The copy button: Brandon, 2026-09-24.
        headers, rows = self.table_cells()
        if not rows:
            return False
        lines = [headers] + rows
        text = '\n'.join('\t'.join(v.replace('\t', ' ').replace('\n', ' ')
                                  for v in line) for line in lines)

        def cell(tag, value):
            return f'<{tag}>{html.escape(value)}</{tag}>'

        body = ''.join('<tr>' + ''.join(cell('td', v) for v in row) + '</tr>'
                       for row in rows)
        markup = ('<table><thead><tr>'
                  + ''.join(cell('th', h) for h in headers)
                  + '</tr></thead><tbody>' + body + '</tbody></table>')
        mime = QMimeData()
        mime.setText(text)
        mime.setHtml(markup)
        QApplication.clipboard().setMimeData(mime)
        return True

    def _place_copy_button(self) -> None:
        """The copy button sits in the top-right corner of the cells,
        clear of the scroll bar, wherever the view is resized to."""
        button = self._copy_button
        viewport = self.viewport()
        corner = viewport.mapTo(self, viewport.rect().topRight())
        button.move(corner.x() - button.width() - 6, corner.y() + 6)
        button.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_copy_button()

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self._copy_button.setVisible(self.whole_table_copy
                                     and self.model() is not None
                                     and self.model().rowCount() > 0)
        self._place_copy_button()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        if not self._copy_button.underMouse():
            self._copy_button.hide()

    def paste(self) -> tuple[int, int]:
        """Paste tab-separated text from the clipboard over the selection's
        top-left corner, onto the shown rows, and return (applied,
        rejected).

        Cells that will not accept a value are left as they are and
        counted, so a paste that partly lands is not silently half
        applied. A paste past the table's edge stops there and does not
        wrap.
        """
        model = self.model()
        text = QApplication.clipboard().text()
        if model is None or not text:
            return 0, 0
        rows, columns = self.selected_block()
        top = rows[0] if rows else 0
        left = columns[0] if columns else 0
        # line by line onto the rows that are shown, from the top one down
        landing = self._shown(range(top, model.rowCount()))

        applied = rejected = 0
        for row_offset, line in enumerate(text.split('\n')):
            if not line.strip() and row_offset == len(text.split('\n')) - 1:
                continue          # trailing newline from the spreadsheet
            for column_offset, cell in enumerate(line.split('\t')):
                column = left + column_offset
                if row_offset >= len(landing) or column >= model.columnCount():
                    continue      # pasting past the edge stops, not wraps
                row = landing[row_offset]
                index = model.index(row, column)
                if not (model.flags(index) & Qt.ItemFlag.ItemIsEditable):
                    rejected += 1
                    continue
                if model.setData(index, cell, Qt.ItemDataRole.EditRole):
                    applied += 1
                else:
                    rejected += 1
        return applied, rejected

    # ---- batch edit ---------------------------------------------------------

    def editable_selection(self) -> list[QModelIndex]:
        """The selected cells that will accept a value."""
        model = self.model()
        if model is None:
            return []
        return [index for index in self.selectedIndexes()
                if model.flags(index) & Qt.ItemFlag.ItemIsEditable]

    def shared_choices(self,
                       indexes: Sequence[QModelIndex]) -> Column | None:
        """The Column the selection shares, when all of its columns offer
        the same list, or None.

        A selection spanning a color column and a coordinate column has
        no common list, so a batch edit of it falls back to typing a
        value.
        """
        model = self.model()
        columns = [model.columns[column] for column
                   in {index.column() for index in indexes}]
        first = columns[0]
        if first.choices and all(column.choices == first.choices
                                 for column in columns):
            return first
        return None

    def batch_edit(self, text: str | None = None) -> tuple[int, int]:
        """Set every selected editable cell to one value, across columns
        as well, and return (applied, rejected).

        The prompt offers the columns' choices when the selection shares
        a list, and otherwise asks for text. Pass `text` to skip the
        prompt.
        """
        indexes = self.editable_selection()
        if not indexes:
            return 0, 0
        if text is None:
            prompt = f'Set {len(indexes)} selected cells to:'
            column = self.shared_choices(indexes)
            if column is not None:
                dialog = ChoiceDialog(self, prompt, column)
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return 0, 0
                text = dialog.value()
            else:
                text, confirmed = QInputDialog.getText(
                    self, 'Batch Edit', prompt)
                if not confirmed:
                    return 0, 0
        applied, rejected, reason = self.model().set_many(indexes, text)
        self.edits_applied.emit(applied, rejected, reason)
        return applied, rejected

    def whole_rows(self) -> list[int]:
        """Rows whose every column is selected (a row-header click, or a
        drag across the full width). Delete removes whole rows, where the
        table allows it, and clears anything less."""
        selection = self.selectionModel()
        if selection is None:
            return []
        return sorted({index.row() for index in selection.selectedRows()})

    def clear_cells(self) -> tuple[int, int]:
        """Empty the selected cells, for the columns that accept empty, and
        return (applied, rejected).

        A column that cannot be blank (a node id, a connectivity list)
        refuses and is counted, as it would refuse the same value typed
        in. The column decides what empty means.
        """
        indexes = self.editable_selection()
        if not indexes:
            return 0, 0
        applied, rejected, reason = self.model().set_many(indexes, '')
        self.edits_applied.emit(applied, rejected, reason)
        return applied, rejected

    # ---- fill handle --------------------------------------------------------

    def fill_handle_rect(self) -> QRect | None:
        """The small square at the bottom-right of the selection, or None.

        It is the spreadsheet fill handle, the fastest way to carry one
        value down a column without selecting the whole column first.
        """
        rows, columns = self.selected_block()
        if not rows or self.model() is None:
            return None
        corner = self.visualRect(self.model().index(rows[-1], columns[-1]))
        if not corner.isValid():
            return None
        # wholly inside the cell, not straddling its corner: an overhang
        # lies outside the rect Qt invalidates when the selection moves, and
        # stayed painted on the cell you clicked away from
        return QRect(corner.right() - FILL_HANDLE + 1,
                     corner.bottom() - FILL_HANDLE + 1,
                     FILL_HANDLE, FILL_HANDLE)

    def _on_handle(self, position: Any) -> bool:
        handle = self.fill_handle_rect()
        return handle is not None and handle.adjusted(
            -FILL_GRAB, -FILL_GRAB, FILL_GRAB, FILL_GRAB).contains(position)

    def fill_from_selection(self, through: int) -> tuple[int, int]:
        """Repeat the selected block over the rows out to `through`, and
        return (applied, rejected).

        The whole block repeats, not only its last row, so filling from
        two alternating rows carries the alternation, as a spreadsheet
        does. Columns named in a filled column's `carries` are filled
        along with it, and hidden rows are skipped.
        """
        rows, columns = self.selected_block()
        model = self.model()
        if not rows or model is None:
            return 0, 0
        # a column can declare companions that fill along with it —
        # sorted, so a carried Type lands before the Unit that carried it
        specs = getattr(model, 'columns', None)
        if specs is not None:
            wanted = set(columns)
            for c in columns:
                wanted.update(j for j, spec in enumerate(specs)
                              if spec.title in specs[c].carries)
            columns = sorted(wanted)
        if through > rows[-1]:
            targets = range(rows[-1] + 1, min(through, model.rowCount() - 1) + 1)
        elif through < rows[0]:
            targets = range(max(through, 0), rows[0])
        else:
            return 0, 0
        targets = self._shown(targets)
        cells = []
        for target in targets:
            # step through the block in order, wrapping, measured from its top
            source = rows[(target - rows[0]) % len(rows)]
            for column in columns:
                cells.append((target, column, model.data(
                    model.index(source, column), Qt.ItemDataRole.EditRole)))
        applied, rejected, reason = model.set_cells(cells)
        self.edits_applied.emit(applied, rejected, reason)
        return applied, rejected

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (event.button() == Qt.MouseButton.LeftButton
                and self._on_handle(event.position().toPoint())):
            rows, _columns = self.selected_block()
            self._filling = rows
            self._fill_to = rows[-1]
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        position = event.position().toPoint()
        if self._filling is not None:
            index = self.indexAt(position)
            if index.isValid():
                self._fill_to = index.row()
                self.viewport().update()
            event.accept()
            return
        # the cursor is the only hint that the square is grabbable
        self.viewport().setCursor(
            QCursor(Qt.CursorShape.CrossCursor) if self._on_handle(position)
            else QCursor(Qt.CursorShape.ArrowCursor))
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Double-clicking the handle fills all the way down, as Excel does.

        The common case is one unit for every channel, and dragging over
        45 rows to say so is slow.
        """
        if (event.button() == Qt.MouseButton.LeftButton
                and self._on_handle(event.position().toPoint())):
            self._filling = self._fill_to = None
            self.fill_from_selection(self.model().rowCount() - 1)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._filling is not None:
            through, self._filling, self._fill_to = self._fill_to, None, None
            self.viewport().update()
            if through is not None:
                self.fill_from_selection(through)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        model = self.model()
        if model is None:
            return
        painter = QPainter(self.viewport())
        rows, columns = self.selected_block()
        if self._filling is not None and rows:
            # outline what the drag would cover, before it is committed
            top = min(self._fill_to, rows[0])
            bottom = max(self._fill_to, rows[-1])
            area = (self.visualRect(model.index(top, columns[0]))
                    .united(self.visualRect(model.index(bottom, columns[-1]))))
            painter.setPen(QPen(self.palette().highlight().color(), 1,
                                Qt.PenStyle.DashLine))
            painter.drawRect(area.adjusted(0, 0, -1, -1))
        elif (handle := self.fill_handle_rect()) is not None:
            # It straddles the corner of the selection, so it has to read
            # against the selection color on one side and the table
            # background on the other. The palette's highlighted-text color
            # is the one guaranteed to contrast with the selection, and the
            # outline is what keeps it from vanishing into a light
            # background where those two are both near-white.
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            painter.setPen(QPen(self.palette().shadow().color(), 1))
            painter.setBrush(self.palette().highlightedText().color())
            painter.drawRect(handle.adjusted(0, 0, -1, -1))
        painter.end()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.matches(QKeySequence.StandardKey.Copy):
            event.accept()
            self.copy()
        elif event.matches(QKeySequence.StandardKey.Paste):
            event.accept()
            self.paste()
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            # a table with a Delete action of its own (the geometry editor,
            # where Delete removes entities) never reaches this: the action's
            # shortcut takes the key first
            event.accept()
            rows = self.whole_rows()
            if self.rows_deletable and rows:
                # the rows are items and whole rows are selected, so Delete
                # means remove them — same as the geometry editor. Cells
                # selected short of a full row still just clear.
                self.rows_deleted.emit(rows)
            else:
                self.clear_cells()
        else:
            super().keyPressEvent(event)

    # ---- menu ---------------------------------------------------------------

    def _show_menu(self, position):
        menu = QMenu(self)
        copy_selection = QAction('Copy', self)
        copy_selection.setShortcut(QKeySequence.StandardKey.Copy)
        copy_selection.triggered.connect(lambda: self.copy())
        menu.addAction(copy_selection)

        if self.whole_table_copy:
            copy_all = QAction('Copy Whole Table', self)
            copy_all.triggered.connect(self.copy_table)
            menu.addAction(copy_all)

        paste = QAction('Paste', self)
        paste.setShortcut(QKeySequence.StandardKey.Paste)
        paste.triggered.connect(self.paste)
        paste.setEnabled(bool(QApplication.clipboard().text()))
        menu.addAction(paste)

        batch = QAction('Batch Edit...', self)
        batch.triggered.connect(lambda: self.batch_edit())
        batch.setEnabled(len(self.editable_selection()) > 1)
        menu.addAction(batch)

        clear = QAction('Clear', self)
        clear.setShortcut(QKeySequence.StandardKey.Delete)
        clear.triggered.connect(self.clear_cells)
        clear.setEnabled(bool(self.editable_selection()))
        menu.addAction(clear)

        if self.rows_deletable:
            rows = self.whole_rows()
            label = (f'Delete {len(rows)} Rows' if len(rows) > 1
                     else 'Delete Row')
            remove = QAction(label, self)
            remove.triggered.connect(
                lambda: self.rows_deleted.emit(self.whole_rows()))
            remove.setEnabled(bool(rows))
            menu.addAction(remove)
        menu.exec(self.viewport().mapToGlobal(position))
