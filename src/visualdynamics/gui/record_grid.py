"""Sub-items shown as a grid, inside the object's own tree expansion.

Every object that expands into records, channels or modes expands into a
grid — even one column wide. Rows are DOFs (modes for a shape set), columns
are whatever tells records apart besides their row: reference DOFs for a
matrix of measurements, the capture for repeated averages, or a single
unlabeled column when the row alone is the identity. One format for every
object means one set of habits: the same selection, the same deletion, the
same icons.

A set that does not fill its rectangle — a record deleted out of a full
matrix, an FRF measured for some pairs only — stays a grid with disabled
holes where the missing records would be. The holes say precisely what is
absent, which neither a flat list nor a refusal did.

Geometry is the exception, on request: its categories keep the list.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
)

from ..core.data import channel_quantities
from ..core.unit_choices import shown_dimension
from .icons import record_icon

if TYPE_CHECKING:                                    # pragma: no cover
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QWidget

    from ..core.data import DataArray

ROW_HEIGHT = 24
MAX_HEIGHT = 320         # past this the grid scrolls rather than the tree
MAX_WIDTH = 460
# Cells hold a 16 px icon and nothing else, so a column only has to be wide
# enough to click. Qt's own minimum is wider than that, and with 'avg 20' as
# a header every column came out 45 px — a 20-average grid showed one of them
# in the project dock.
COLUMN_WIDTH = 24


class RowKey(NamedTuple):
    """What identifies a row: a DOF, what it measures, and — only if those
    two are not enough — which of the channels sharing them it is."""

    dof: str
    quantity: str
    occurrence: int = 0


def row_keys(data: DataArray) -> list[RowKey]:
    """One RowKey per record.

    A row is a DOF *and* a data type. Two channels can sit at one point — a
    shaker's load cell and the accelerometer beside it share a node and a
    direction — so a DOF may repeat, but never within a quantity, and a volt,
    a newton and a meter per second squared never belong on one row whatever
    their DOF says.

    The quantity comes from `known_dim`, not `ordinate_dim`: a source can name
    a quantity without sizing it, and that claim is enough to tell channels
    apart. It is the *response factor* of the dimension
    (`channel_quantities`), because the row is the response channel: keyed
    on the compound, a CPSD split every accelerometer into a row per thing
    it was measured against. The record's icon reads the hint too, and the
    two must agree or the axis and the cells describe different rows.

    When even that is not enough — an import that says nothing about units, so
    every channel at a DOF reads 'unknown' — `occurrence` breaks the tie by
    channel order. That is deliberately the last resort. A channel index is an
    artifact of how a file was written rather than a property of a
    measurement, and reordering the table would rename the row; but the
    alternative is refusing the grid and losing an arrangement we *do* know,
    which is worse. Declare the units and the quantities separate, every
    occurrence falls back to 0, and the artifact stops being used.
    """
    factors = [channel_quantities(data.known_dim(i))
               for i in range(data.num_records)]
    quantities = [(dof, factors[i][0])
                  for i, dof in enumerate(data.response_dof)]
    if data.reference_dof is not None:
        # the occurrence tiebreaker must see the reference as a
        # *channel* too: keyed on the DOF alone, a CPSD's records
        # against a drive point's accelerometer and its load cell
        # looked like one column measured twice, and every response
        # grew a phantom second row
        columns = [(dof, factors[i][1])
                   for i, dof in enumerate(data.reference_dof)]
    else:
        columns = data.column_keys() or [''] * data.num_records
    seen = Counter()
    keys = []
    for pair, column in zip(quantities, columns):
        keys.append(RowKey(*pair, seen[(pair, column)]))
        seen[(pair, column)] += 1
    return keys


def row_labels(keys: Sequence[RowKey]) -> list[str]:
    """Headers for `row_keys`: the DOF, and a marker only where it must have one.

    Normally the DOF alone — what a row measures is on the icon in each of its
    cells, which is read faster than a word. But two channels telling apart
    only by `occurrence` have the *same* icon, because neither says what it
    measures, so there the DOF is not enough and the channel's position has to
    show. It disappears again the moment units are declared.
    """
    ambiguous = {key.dof for key in keys if key.occurrence}
    return [f'{key.dof} #{key.occurrence + 1}' if key.dof in ambiguous
            else key.dof for key in keys]


def short_labels(columns: Sequence[str]) -> list[str]:
    """Column headers narrow enough that a 20-average grid is usable.

    A block reads as 'avg 7' in a record label, where it is prose. As a
    column header it only has to be told from its neighbors, and the word
    is the same on all twenty — so where every column shares one prefix, the
    prefix goes. Reference DOFs have no common prefix and are left alone.
    """
    if len(columns) < 2:
        return list(columns)
    heads = {column.rsplit(' ', 1)[0] for column in columns if ' ' in column}
    if len(heads) != 1 or any(' ' not in column for column in columns):
        return list(columns)
    return [column.rsplit(' ', 1)[1] for column in columns]


def grid_axes(data: DataArray
              ) -> tuple[list[RowKey], list[str]] | None:
    """(rows, columns) for any grid-able object; None only when it has none.

    The columns are whatever tells records apart besides their row — the
    reference DOF for a matrix of measurements, the capture for repeated
    averages — and a single unlabeled column when the row alone is the
    identity: multiple coherence, a plain time history, a channel table, a
    shape set. A specification whose every record is a channel against
    itself collapses its reference column too, because a diagonal spelled
    out across six columns says nothing the rows do not.
    """
    plan = grid_plan(data)
    if plan is None:
        return None
    return plan.rows, plan.columns


def _icon_source(obj, kind):
    """How a cell of this kind gets its icon, closed over the live object so
    a refresh after declaring units sees the change."""
    from .icons import child_icon, quantity_icon, quantity_of

    if kind == 'record':
        return lambda i: record_icon(obj, i)
    if kind == 'photo':
        # each cell is the photo itself, in miniature
        def photo_icon(i: int) -> QIcon:
            from PySide6.QtGui import QIcon, QPixmap

            pixmap = QPixmap()
            pixmap.loadFromData(obj.images[i])
            return (QIcon(pixmap) if not pixmap.isNull()
                    else child_icon('record', 'Photos'))
        return photo_icon
    if kind == 'channel':
        def channel_icon(i: int) -> QIcon:
            from ..units import dimension_of

            unit = str(obj['unit'][i]).strip()
            quantity = quantity_of(dimension_of(unit) or '') if unit else None
            if quantity is not None:
                return quantity_icon(quantity)
            return child_icon('channel', 'ChannelTable', bool(unit))
        return channel_icon
    if kind == 'match':
        # a pair of modes, and neither of them is this object's to
        # declare units for — the shape sets own that
        return lambda _i: child_icon('mode', 'ShapeSet', True)
    if kind == 'dof':
        declared = getattr(obj, 'ordinate_unit', True) is not None
        return lambda _i: child_icon(
            'record', 'SineSweepSpecification', declared)
    return lambda _i: child_icon('mode', 'ShapeSet', obj.units_defined)


class GridPlan(NamedTuple):
    """Everything a grid needs, independent of what kind of object it maps.

    `kind` is the reference vocabulary — 'record', 'channel' or 'mode' — so
    a selection in the grid speaks the same language as the tree always has.
    `cells` maps (row, column) to the item index; positions absent from it
    are holes.
    """

    kind: str
    rows: list
    row_labels: list
    columns: list
    cells: dict
    #: one quantity per column where the column's DOF is ambiguous —
    #: drawn as the quantity's icon on the header, the same mark the
    #: rows wear in their cells — and None where the DOF alone tells
    column_marks: list | None = None
    #: whether the columns are reference coordinates — editable, as the
    #: rows are — rather than captures or the one unlabeled column
    column_dofs: bool = False
    #: the reference channel each column is, (dof, quantity), where the
    #: columns are coordinates — a rename is the channel's, not the
    #: point's, so the column's quantity goes with its coordinate
    column_keys: list | None = None
    #: one flag per row where the rows are channels that can be the
    #: references of an FRF — a time history's — drawn as a leading
    #: column of check boxes headed Ref (Brandon, 2026-09-25: see which
    #: channels are the references, and tick or untick them there)
    checks: list[bool] | None = None


def grid_plan(obj: Any) -> GridPlan | None:
    from ..core.channel_table import ChannelTable
    from ..core.matches import MatchedModes
    from ..core.photos import Photos
    from ..core.shapes import ShapeSet

    if hasattr(obj, 'column_keys'):          # a DataArray
        return _data_plan(obj)
    if isinstance(obj, ChannelTable):
        return _channel_plan(obj)
    if isinstance(obj, ShapeSet):
        return _mode_plan(obj)
    if isinstance(obj, MatchedModes):
        return _match_plan(obj)
    if isinstance(obj, Photos):
        return _photo_plan(obj)
    from ..core.sine import SineLevelSet, SineSweepSpecification
    if isinstance(obj, (SineSweepSpecification, SineLevelSet)):
        return _dof_plan(obj)
    return None


def _data_plan(data):
    keys = row_keys(data)
    column_dofs = False
    if data.reference_dof is not None:
        column_dofs = True
        # a reference is a channel, and channel identity is DOF plus
        # quantity: a drive point's accelerometer and load cell share a
        # DOF, and columns keyed on the DOF alone collapsed a CPSD's 13
        # reference channels into 11 and shadowed 26 records behind
        # their cells. The header still says the DOF and nothing else;
        # where the DOF is ambiguous the column is *marked* with its
        # quantity, drawn as the quantity's icon — the same mark the
        # rows wear in their cells, and the same restraint as their
        # '#2'.
        quantities = [channel_quantities(data.known_dim(i))[1]
                      for i in range(data.num_records)]
        identities: list = list(zip(data.reference_dof, quantities))
        by_dof: dict[str, set[str]] = {}
        for dof, quantity in set(identities):
            by_dof.setdefault(dof, set()).add(quantity)
        display = {identity: (identity[0],
                              identity[1] if len(by_dof[identity[0]]) > 1
                              else None)
                   for identity in set(identities)}
        if all(identity == (key.dof, key.quantity)
               for identity, key in zip(identities, keys)):
            # the column would only repeat the row, as a
            # specification's reference does
            identities = [''] * data.num_records
            display = {'': ('', None)}
            column_dofs = False
    else:
        identities = data.column_keys() or [''] * data.num_records
        display = {identity: (identity, None) for identity in identities}
    rows, columns, marks, cells, column_keys = [], [], [], {}, []
    row_index, column_index = {}, {}
    for i, (key, identity) in enumerate(zip(keys, identities)):
        if key not in row_index:
            row_index[key] = len(rows)
            rows.append(key)
        if identity not in column_index:
            column_index[identity] = len(columns)
            label, mark = display[identity]
            columns.append(label)
            marks.append(mark)
            column_keys.append(tuple(identity) if column_dofs else None)
        cells[(row_index[key], column_index[identity])] = i
    # labels come from the unique rows, not the per-record keys: a 12 x 2
    # FRF has 24 keys but 12 rows, and labeling rows by the first 12 keys
    # silently shifted every header after the first repeat
    checks = None
    if hasattr(data, 'reference_channels'):
        # a row is a reference when any record in it is one: the
        # identity is the channel's, the same in every capture
        chosen = set(data.reference_channels())
        by_row: dict[int, bool] = {}
        for (row, _column), index in cells.items():
            identity = (data.response_dof[index], data.ordinate_dim[index])
            by_row[row] = by_row.get(row, False) or identity in chosen
        checks = [by_row.get(row, False) for row in range(len(rows))]
    return GridPlan('record', rows, row_labels(rows), columns, cells,
                    column_marks=marks, column_dofs=column_dofs,
                    column_keys=column_keys, checks=checks)


def _channel_plan(table):
    """One row per channel, named by its DOF.

    A channel is a DOF and a quantity, here as everywhere: a drive
    point's load cell and its accelerometer share a DOF and are told
    apart by their icons, so the '#2' marker appears only where the
    icons cannot help — two channels at one DOF whose units name the
    same quantity, or none.
    """
    from ..units import dimension_of
    from .icons import quantity_of

    dofs = table.dof_strings()
    units = table['unit']
    seen = Counter()
    rows, cells = [], {}
    for i, dof in enumerate(dofs):
        quantity = (quantity_of(dimension_of(str(units[i]).strip()) or '')
                    if str(units[i]).strip() else None) or 'unknown'
        rows.append(RowKey(dof, quantity, seen[(dof, quantity)]))
        seen[(dof, quantity)] += 1
        cells[(i, 0)] = i
    return GridPlan('channel', rows, row_labels(rows), [''], cells)


def _photo_plan(photos):
    """One row per photo, named by the photo's own name."""
    rows = [RowKey(name, 'photo', 0) for name in photos.names]
    cells = {(i, 0): i for i in range(photos.num_photos)}
    return GridPlan('photo', rows, [key.dof for key in rows], [''], cells)


def _mode_plan(shapes):
    """One row per mode. Not DOFs — a mode has none of its own — but the
    same format, so the habits transfer."""
    rows = [RowKey(shapes.mode_label(i), 'mode', 0)
            for i in range(shapes.num_shapes)]
    cells = {(i, 0): i for i in range(shapes.num_shapes)}
    return GridPlan('mode', rows, [key.dof for key in rows], [''], cells)


def sine_dofs(grouped) -> list[str]:
    """The control DOFs a sine specification or a level set covers, in
    order: the specification's own, or the union of its levels' — a
    tone the instructions windowed may have been read at fewer."""
    if hasattr(grouped, 'tones'):
        return [str(dof) for dof in grouped.response_dof]
    out: list[str] = []
    for level in grouped.levels:
        for dof in level.response_dof:
            if str(dof) not in out:
                out.append(str(dof))
    return out


def _dof_plan(grouped):
    """One row per control DOF, as every other object's rows are its
    channels. A sine specification is a requirement at each control
    DOF and an extraction is the level read at each; picking rows
    plots those DOFs — the records-and-modes habits — and the tone is
    the pick on the bar. The rows were the tones until 2026-09-25,
    with the DOF on the bar, which inverted the rest of the tree:
    Brandon expanded a three-DOF specification expecting three rows
    and found one."""
    rows = [RowKey(dof, 'dof', 0) for dof in sine_dofs(grouped)]
    cells = {(i, 0): i for i in range(len(rows))}
    return GridPlan('dof', rows, [key.dof for key in rows], [''], cells)


def _match_plan(matched):
    """One row per matched pair, labeled by the two mode numbers it
    joins — the same 1-based numbering the matched-modes table shows.

    Mode labels would need both shape sets, and a MatchedModes holds
    only their names; the numbers are what it knows on its own.
    """
    rows = [RowKey(f'{row + 1} \u2194 {column + 1}', 'match', 0)
            for row, column in matched.pairs]
    cells = {(i, 0): i for i in range(len(rows))}
    return GridPlan('match', rows, [key.dof for key in rows], [''], cells)


class _LabelEditor(QLineEdit):
    """A line edit that can be walked away from.

    `editingFinished` fires on Enter and on focus-out and has no reading
    for Escape, so abandoning an edit would otherwise commit it.
    """

    abandoned = Signal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.abandoned.emit()
            return
        super().keyPressEvent(event)


class RecordGrid(QTableWidget):
    """Sub-items laid out as rows against columns, whatever their kind.

    Selecting cells is how sub-items are chosen; the selection is the only
    state, so nothing can disagree with it. `kind` says what a cell *is* —
    'record', 'channel' or 'mode' — in the same vocabulary the tree has
    always used, so everything downstream of a selection is unchanged.
    """

    selection_changed = Signal()
    #: (row, new text) — a row's label typed over: a photograph's name,
    #: or the coordinate of a record's or a channel's row (Brandon,
    #: 2026-09-06: the rows of every data object are coordinates, and a
    #: user should be able to correct them)
    row_renamed = Signal(int, str)
    #: (column, new text) — a reference channel's coordinate typed over
    column_renamed = Signal(int, str)
    #: (row, checked) — a row's Ref box ticked or unticked
    reference_toggled = Signal(int, bool)

    owner = None        # the object name, set and maintained by the window

    def __init__(self, data: Any, parent: QWidget | None = None) -> None:
        plan = grid_plan(data)
        # The Ref column, where the rows are channels that can be
        # references. It is the *last* logical column, moved to the front
        # of the header's visual order: the record cells keep the
        # column numbers everything else counts on (`item(row, 0)` is
        # the first record, in the window and in every test), and the
        # boxes still read first, where a person looks for them.
        self.check_column: int | None = (len(plan.columns)
                                         if plan.checks is not None else None)
        super().__init__(len(plan.rows),
                         len(plan.columns) + (self.check_column is not None),
                         parent)
        self.kind: str = plan.kind
        self.row_keys: list[Any] = plan.rows
        # The header says the DOF and nothing else. What a row *measures* is
        # on the icon in every one of its cells — a force reads as a force at
        # a glance, where '[force]' spelled out in the label cost twice the
        # width and had to be read.
        self.responses: list[str] = plan.row_labels
        self.references: list[str] = plan.columns
        self.column_keys: list = plan.column_keys or [None] * len(plan.columns)
        self._icon_for = _icon_source(data, plan.kind)
        self.setHorizontalHeaderLabels(
            short_labels(plan.columns)
            + (['Ref'] if self.check_column is not None else []))
        if self.check_column is not None:
            self.column_keys.append(None)
            self.horizontalHeaderItem(self.check_column).setToolTip(
                'The reference channels: what FRFs and multiple coherence '
                'are computed against. Tick a channel to make it one.')
        # a marked column wears its quantity as the icon the cells
        # already use — '101Z+ (force)' spelled out is the width lesson
        # the row headers learned long ago
        marks = plan.column_marks or []
        for column, mark in enumerate(marks):
            if not mark:
                continue
            from .icons import quantity_icon

            item = self.horizontalHeaderItem(column)
            try:
                item.setIcon(quantity_icon(mark))
                item.setToolTip(f'{plan.columns[column]} — {mark}')
            except KeyError:     # a quantity with no icon: say the word
                item.setText(f'{plan.columns[column]} ({mark})')
        # a single unlabeled column has no header worth a strip of pixels
        # — unless the Ref column is there to be named
        self.horizontalHeader().setVisible(plan.columns != ['']
                                           or self.check_column is not None)
        self.setVerticalHeaderLabels(self.responses)
        # A qualified row label — '101Z+ [acceleration]' — is twice the width
        # of a DOF, and left to size the header it took 130 px of a 250 px
        # dock and left room for two columns. Cap it at a comfortable DOF and
        # let the long ones elide; the full text stays on the tooltip.
        header_rows = self.verticalHeader()
        # the cap is sized for the kind of label: a DOF for records and
        # channels, a 'Mode 12 — 45.6 Hz' for modes, which a DOF-wide cap
        # truncated to 'Mode 1 —'
        widest = ('Mode 00 \u2014 000.0 Hz' if plan.kind in ('mode', 'photo')
                  else '000RX+')
        header_rows.setMaximumWidth(
            self.fontMetrics().horizontalAdvance(widest) + 16)
        for row, key in enumerate(plan.rows):
            self.verticalHeaderItem(row).setToolTip(
                f'{key.dof} \u2014 {shown_dimension(key.quantity)}')
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        header = self.horizontalHeader()
        # sized to content either way, so a reference DOF header still fits
        # and a bare average number takes only what it needs — the minimum is
        # lowered because the style's default is wider than a 16 px icon
        header.setMinimumSectionSize(COLUMN_WIDTH)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.setIconSize(QSize(16, 16))

        self._records = {}
        self._building = True
        for row, key in enumerate(plan.rows):
            if self.check_column is not None:
                box = QTableWidgetItem()
                # checkable and enabled, never selectable: a tick is a
                # setting, not a pick of the row's records
                box.setFlags(Qt.ItemFlag.ItemIsEnabled
                             | Qt.ItemFlag.ItemIsUserCheckable)
                box.setCheckState(Qt.CheckState.Checked if plan.checks[row]
                                  else Qt.CheckState.Unchecked)
                box.setToolTip(f'{key.dof} \u2014 '
                               f'{shown_dimension(key.quantity)} as a '
                               'reference')
                self.setItem(row, self.check_column, box)
            for column, reference in enumerate(plan.columns):
                cell = QTableWidgetItem()
                cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                index = plan.cells.get((row, column))
                if index is None:
                    # a hole: this pair was never measured, or its record
                    # was deleted. Disabled says "nothing here" precisely,
                    # where a flat list said nothing at all.
                    cell.setFlags(Qt.ItemFlag.NoItemFlags)
                else:
                    mark = marks[column] if column < len(marks) else None
                    cell.setIcon(self._icon_for(index))
                    cell.setToolTip(
                        f'{key.dof} \u2014 '
                        f'{shown_dimension(key.quantity)}'
                        + (f' / {reference}' if reference else '')
                        + (f' ({mark})' if reference and mark else ''))
                    self._records[(row, column)] = index
                self.setItem(row, column, cell)
        self._building = False
        self.itemSelectionChanged.connect(self.selection_changed)
        if self.check_column is not None:
            self.itemChanged.connect(self._box_changed)
            header.moveSection(self.check_column, 0)
        # A row label is typed over where it is the user's to say: a
        # photograph's name, and the coordinate of a record or a
        # channel — where the wrong assignment was made at the
        # instrument and is corrected here. A mode, a matched pair and a
        # tone are labeled by what they are and stay so. The columns of
        # a matrix are coordinates too, and edit the same way.
        # Double-clicking a label to change it is what the tree already
        # does for objects; this is the same gesture on a grid's edges.
        self.editable_rows: bool = plan.kind in ('photo', 'record', 'channel')
        self.editable_columns: bool = bool(plan.column_dofs)
        if self.editable_rows:
            self.verticalHeader().setSectionsClickable(True)
            self.verticalHeader().sectionDoubleClicked.connect(
                self.edit_row_label)
        if self.editable_columns:
            self.horizontalHeader().setSectionsClickable(True)
            self.horizontalHeader().sectionDoubleClicked.connect(
                self.edit_column_label)

    def _box_changed(self, item: QTableWidgetItem) -> None:
        if self._building or item.column() != self.check_column:
            return
        self.reference_toggled.emit(
            item.row(), item.checkState() == Qt.CheckState.Checked)

    def reference_rows(self) -> list[int]:
        """The rows whose Ref box is ticked."""
        if self.check_column is None:
            return []
        return [row for row in range(self.rowCount())
                if self.item(row, self.check_column).checkState()
                == Qt.CheckState.Checked]

    def set_reference_rows(self, rows: Sequence[int]) -> None:
        """Restate the boxes without announcing a change."""
        if self.check_column is None:
            return
        wanted = set(rows)
        self._building = True
        try:
            for row in range(self.rowCount()):
                self.item(row, self.check_column).setCheckState(
                    Qt.CheckState.Checked if row in wanted
                    else Qt.CheckState.Unchecked)
        finally:
            self._building = False

    def row_records(self, row: int) -> list[int]:
        """The record indices in a row, in column order."""
        return [index for (r, _c), index in sorted(self._records.items())
                if r == row]

    def edit_row_label(self, row: int) -> None:
        """Open an editor over the row's header, in place.

        A header is not a cell and Qt will not edit one, so the editor
        is a line edit laid over the section. It commits on Enter or on
        losing focus and abandons on Escape, which is what editing a
        name anywhere else in the window does.

        The commit is *queued*: acting on it rebuilds this grid, and
        rebuilding the widget an editor is sitting in — from inside that
        editor's own signal — is the shape of crash that took the docks
        out.
        """
        header = self.verticalHeader()
        # the coordinate itself, not the label: a row told apart only
        # by its position reads '101Z+ #2', and the '#2' is the grid's
        current = (self.row_keys[row].dof if self.kind != 'photo'
                   else self.responses[row])
        return self._edit_label(
            header, current,
            QRect(0, header.sectionViewportPosition(row),
                  header.width(), header.sectionSize(row)),
            lambda text: self.row_renamed.emit(row, text))

    def edit_column_label(self, column: int) -> None:
        """Open an editor over a reference column's header, in place —
        the row gesture on the other edge of the grid."""
        header = self.horizontalHeader()
        return self._edit_label(
            header, self.references[column],
            QRect(header.sectionViewportPosition(column), 0,
                  header.sectionSize(column), header.height()),
            lambda text: self.column_renamed.emit(column, text))

    @staticmethod
    def _edit_label(header, current, geometry, commit):
        from PySide6.QtCore import QTimer

        editor = _LabelEditor(header)
        editor.setText(current)
        editor.setGeometry(geometry)
        editor.selectAll()
        editor.show()
        editor.setFocus()

        done = False

        def finish(keep):
            # Escape hides the editor, hiding loses the focus, and
            # losing focus is `editingFinished` — so without this the
            # abandoned edit committed itself on the way out.
            nonlocal done
            if done:
                return
            done = True
            text = editor.text().strip()
            editor.hide()
            editor.deleteLater()
            if keep and text and text != current:
                QTimer.singleShot(0, lambda: commit(text))

        editor.editingFinished.connect(lambda: finish(True))
        editor.abandoned.connect(lambda: finish(False))
        return editor

    def refresh_icons(self) -> None:
        """Restate every cell icon in place — after units are declared, the
        badge goes and a quantity may appear — without rebuilding the grid
        and losing the selection."""
        for (row, column), index in self._records.items():
            self.item(row, column).setIcon(self._icon_for(index))

    def selected_records(self) -> list[int]:
        """Record indices for the selected cells, in row-major order."""
        return sorted(self._records[(index.row(), index.column())]
                      for index in self.selectedIndexes())

    def select_records(self, records: Sequence[int]) -> None:
        """Show `records` as selected, without echoing back a change."""
        wanted = set(records)
        cells = [cell for cell, record in self._records.items()
                 if record in wanted]
        blocked = self.blockSignals(True)
        self.clearSelection()
        for row, column in cells:
            self.item(row, column).setSelected(True)
        self.blockSignals(blocked)

    def sizeHint(self) -> QSize:
        # QTableWidget's stock hint is 256x192 whatever the content;
        # the tree row takes the widget's word for it, so a one-photo
        # grid stood 192 px tall
        return self.preferred_size()

    def minimumSizeHint(self) -> QSize:
        return self.preferred_size()

    def preferred_size(self) -> QSize:
        """How big the tree should make room for.

        Capped: a 100-channel CPSD would otherwise push everything else in
        the tree off the bottom, so past the cap the grid scrolls itself.
        """
        # the header's own width is not settled until the widget is laid out,
        # and this is asked for before that; its hint capped is what it will be
        rows = self.verticalHeader()
        width = (min(rows.sizeHint().width(), rows.maximumWidth())
                 + sum(self.columnWidth(c) for c in range(self.columnCount()))
                 + 4)
        height = (ROW_HEIGHT * self.rowCount()
                  + self.horizontalHeader().height() + 4)
        if width > MAX_WIDTH:      # room for the scrollbar that will appear,
            height += self.horizontalScrollBar().sizeHint().height()
        return QSize(min(width, MAX_WIDTH), min(height, MAX_HEIGHT))
