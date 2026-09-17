"""Excel for channel tables, both ways.

A channel table is the one visualdynamics object that *is* a spreadsheet — rows of
per-channel text — so it gets a spreadsheet format. Nothing is lost in the
writing: every column goes out in order, 'channel' as the integer it is
and the rest as the strings they are. Units here are metadata, not values,
so there is nothing to convert and `unit_system` is accepted and unused.

Reading it back matters as much as writing it. A channel table arrives
as a spreadsheet far more often than as anything else — a calibration
lab sends one, a controller writes one — and until this existed the
export was a one-way door: a table dragged out of the window could not
be dragged back in.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.channel_table import ChannelTable
    from ..units import UnitSystem


def _cell(table: ChannelTable, name: str, row: int) -> Any:
    """One cell's value, as the spreadsheet should carry it."""
    from ..core.channel_table import COLUMNS_KIND
    from ..core.unit_choices import shown_dimension

    value = str(table[name][row])
    if COLUMNS_KIND(name) in ('int', 'index'):
        # a number, not text: Excel flags "number stored as text" with a
        # green triangle on every cell otherwise, and a node is always a
        # number. 'index' is blank-able, and a blank stays blank.
        return int(value) if value else None
    # the sheet says the word the interface says, and canonical import
    # maps it back — a spreadsheet reading 'length' beside an app
    # reading 'displacement' is the kind of difference nobody forgives
    return shown_dimension(value) if name == 'channel_type' else value


def _unit_lists(sheet: Any, table: ChannelTable, names: list[str],
                last: int) -> None:
    """Make the unit drop-down follow the channel's declared type.

    Excel cannot hold a different inline list per row, but it can look
    one up: the units for each quantity go on a hidden sheet as named
    ranges, and the validation is `INDIRECT` of the Channel Type cell
    beside it. Declare a channel a voltage and its unit cell offers V
    and mV; leave the type blank and it offers all of them, which is
    what the `IF` is for — `INDIRECT("")` is an error, not an empty
    list, and Excel would refuse the cell rather than shrug.
    """
    from openpyxl.utils import get_column_letter
    from openpyxl.workbook.defined_name import DefinedName
    from openpyxl.worksheet.datavalidation import DataValidation

    from ..core.unit_choices import (
        ALL_ORDINATE_UNITS,
        ORDINATE_UNITS,
        shown_dimension,
    )

    if 'unit' not in names or 'channel_type' not in names:
        return
    book = sheet.parent
    lookup = book.create_sheet('Units')
    lookup.sheet_state = 'hidden'
    groups = {shown_dimension(quantity): units
              for quantity, units in ORDINATE_UNITS.items()}
    groups['AllUnits'] = ALL_ORDINATE_UNITS
    for column, (quantity, units) in enumerate(groups.items(), start=1):
        letter = get_column_letter(column)
        for row, unit in enumerate(units, start=1):
            lookup.cell(row, column, unit)
        book.defined_names.add(DefinedName(
            quantity,
            attr_text=f"Units!${letter}$1:${letter}${len(units)}"))

    unit_at = get_column_letter(names.index('unit') + 1)
    type_at = get_column_letter(names.index('channel_type') + 1)
    rule = DataValidation(
        type='list', allow_blank=True, showDropDown=False,
        formula1=f'=INDIRECT(IF(${type_at}2="","AllUnits",${type_at}2))')
    rule.error = 'not a unit of this channel\'s type'
    rule.errorTitle = 'Not one of the choices'
    sheet.add_data_validation(rule)
    rule.add(f'{unit_at}2:{unit_at}{last}')


def _add_validation(sheet: Any, table: ChannelTable,
                    names: list[str]) -> None:
    """Excel's own drop-downs, from the same lists the window offers.

    A channel table is very often filled in by somebody else — a
    calibration lab, a test engineer with the spreadsheet and no copy of
    this — so the constraints have to travel with the file rather than
    live only in the window. Excel refuses a bad direction there for the
    same reason `set_cell` refuses one here.

    The unit list is every unit rather than the ones a channel's type
    allows: a validation applies to a column, and narrowing it per row
    would need one rule per row for a constraint the importer re-checks
    anyway.
    """
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    from ..core.channel_table import CHOICES_FOR
    from ..core.unit_choices import shown_dimension

    last = table.num_channels + 1          # row 1 is the header
    if last < 2:
        return
    _unit_lists(sheet, table, names, last)
    for index, name in enumerate(names, start=1):
        if name == 'unit':
            continue                       # handled above, per row
        choices = CHOICES_FOR(name)
        if name == 'channel_type':
            choices = [shown_dimension(c) for c in choices]
        if not choices:
            continue
        joined = ','.join(str(c) for c in choices)
        if len(joined) > 250:              # Excel's inline-list ceiling
            continue
        # showDropDown=False *shows* the arrow: the OOXML attribute
        # means "suppress the in-cell list", so the sense is inverted
        # from its name and True hides the very thing this is for
        rule = DataValidation(type='list', formula1=f'"{joined}"',
                              allow_blank=True, showDropDown=False)
        rule.error = f'{name} must be one of: {joined}'
        rule.errorTitle = 'Not one of the choices'
        letter = get_column_letter(index)
        sheet.add_data_validation(rule)
        rule.add(f'{letter}2:{letter}{last}')


def save(table: ChannelTable, path: str | os.PathLike, unit_system: UnitSystem | None = None,
         **_kwargs: Any) -> None:
    from openpyxl import Workbook
    from openpyxl.utils import get_column_letter

    book = Workbook()
    sheet = book.active
    sheet.title = 'Channel Table'
    names = list(table.column_names)
    # the same headers the window and the report use — one spelling for
    # one table, wherever it is read. The parenthetical units come back
    # through `canonical_name`, which strips them.
    from ..core.channel_table import title_of

    sheet.append([title_of(name) for name in names])
    for cell in sheet[1]:
        cell.font = cell.font.copy(bold=True)
    for row in range(table.num_channels):
        sheet.append([_cell(table, name, row) for name in names])
    _add_validation(sheet, table, names)
    for i, name in enumerate(names, start=1):
        widest = max([len(name)] + [len(str(table[name][r]))
                                    for r in range(table.num_channels)])
        sheet.column_dimensions[get_column_letter(i)].width = widest + 3
    book.save(str(path))


#: How many rows down to look for the header. Our own export puts it
#: first, but a controller's does not: Rattlesnake writes merged group
#: headings across row 1 and the column names on row 2. Rather than
#: teach this two layouts, it finds the row that *names columns* — a
#: spreadsheet's header is the first row that says 'channel'.
HEADER_SEARCH_ROWS = 10

#: what a header cell has to say for a row to be the header row. 'node'
#: and 'direction' are not required: a table whose channels have not
#: been assigned DOFs yet is a real thing, and importing it is how the
#: user gets to fix that.
HEADER_MARKERS = ('channel', 'channel_number', 'ch', 'ch.')

#: What other people call the four columns this object needs by name.
#: A calibration lab writes 'Node Number', a controller writes 'Point',
#: and both mean `node` — left alone they arrive as *extra* columns
#: while `node` itself is created empty beside them, which is the same
#: table with its DOFs thrown away.
#:
def _key(cell):
    """A header cell as a column name: 'Serial Number' -> 'serial_number'.

    The schema's own `canonical_name`, not a second table beside it.
    This file used to alias the four core columns only, on the reasoning
    that guessing at the rest would invent meaning the file did not
    carry. With a fixed schema an unrecognized column is *dropped*
    rather than kept under its own name, so the two spellings had to
    agree — and they did not: this end turned the exported header
    'Sensitivity (mV/Unit)' into a column nobody had heard of, and a
    table written here would not read back in.
    """
    from ..core.channel_table import canonical_name

    return canonical_name('' if cell is None else str(cell))


def sniff(path: str | os.PathLike) -> bool:
    """Is this a spreadsheet holding a channel table?

    The extension is not enough — an `.xlsx` may hold anything — so this
    opens it and looks for a header row naming a channel column. Wrong
    guesses here are expensive: this importer is asked about every file
    dropped on the window.
    """
    if not str(path).lower().endswith(('.xlsx', '.xlsm')):
        return False
    try:
        from openpyxl import load_workbook

        book = load_workbook(str(path), read_only=True, data_only=True)
        try:
            return _header_row(book.active) is not None
        finally:
            book.close()
    except Exception:      # noqa: BLE001 — an unreadable file is not ours
        return False


def _header_row(sheet: Any) -> tuple[int, list[str]] | None:
    """(row number, column names) of the header, or None if there is none."""
    for number, row in enumerate(
            sheet.iter_rows(max_row=HEADER_SEARCH_ROWS, values_only=True),
            start=1):
        keys = [_key(cell) for cell in row]
        if any(key in HEADER_MARKERS for key in keys):
            return number, keys
    return None


def load(path: str | os.PathLike, **_kwargs: Any) -> ChannelTable:
    """The spreadsheet as a ChannelTable.

    Every column is kept, in the file's own order, under the name its
    header spells — the same rule the object itself follows, so whatever
    a lab or a controller put in the file survives the trip. Values are
    read as **text**, because that is what the object stores: a serial
    number that happens to be all digits is not a number, and one with a
    leading zero would come back a different string if it were.

    Blank trailing rows are dropped; a blank *column* header is not a
    column and its cells go with it.
    """
    from openpyxl import load_workbook

    from ..core.channel_table import ChannelTable

    book = load_workbook(str(path), read_only=True, data_only=True)
    try:
        sheet = book.active
        found = _header_row(sheet)
        if found is None:
            raise ValueError(
                f'{os.path.basename(str(path))} has no channel table in it: '
                f'no row in the first {HEADER_SEARCH_ROWS} names a channel '
                f'column')
        number, keys = found
        # a column named twice — 'Node' beside 'Node Number', say — keeps
        # the first, since the second would silently overwrite it
        seen: set[str] = set()
        wanted = []
        for i, key in enumerate(keys):
            if key and key not in seen:
                seen.add(key)
                wanted.append((i, key))
        columns: dict[str, list[str]] = {key: [] for _i, key in wanted}
        for row in sheet.iter_rows(min_row=number + 1, values_only=True):
            values = [row[i] if i < len(row) else None for i, _k in wanted]
            # a row of nothing is the end of the table, not a channel
            if all(v is None or str(v).strip() == '' for v in values):
                continue
            for (_i, key), value in zip(wanted, values):
                columns[key].append('' if value is None else str(value).strip())
    finally:
        book.close()

    if not columns.get('channel'):
        raise ValueError(f'{os.path.basename(str(path))} has a channel '
                         f'header but no channels under it')
    # the object needs its four; anything the file did not say is blank,
    # which is a channel waiting to be told rather than a refusal
    for core in ('channel', 'node', 'direction', 'unit'):
        columns.setdefault(core, [''] * len(columns['channel']))
    return ChannelTable(columns)
