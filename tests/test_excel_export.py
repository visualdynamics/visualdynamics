"""Channel table -> Excel: the one visualdynamics object that *is* a spreadsheet.

Nothing is lost in the writing — every column in order, channel numbers as
integers, everything else as the strings they are. Units here are metadata
rather than values, so no unit system is involved.
"""

from __future__ import annotations

from conftest import fixture_path
from openpyxl import load_workbook

import visualdynamics
from visualdynamics import io


def test_the_channel_table_exports_to_excel(tmp_path):
    table = io.load(fixture_path('plate', 'channel_table.vdyn'))
    path = tmp_path / 'channels.xlsx'
    io.export_file(table, str(path), format='excel')
    sheet = load_workbook(path).active
    assert sheet.title == 'Channel Table'
    header = [c.value for c in sheet[1]]
    # the whole schema, in its order and under its own headers — the
    # report and the window show the same table, so they share these
    from visualdynamics.core.channel_table import title_of
    assert header == [title_of(name) for name in table.SCHEMA]
    assert sheet.max_row == table.num_channels + 1
    assert sheet.cell(2, 1).value == int(table['channel'][0])
    assert sheet.cell(2, 2).value == int(table['node'][0])
    columns = len(table.column_names)
    # a blank cell reads back as None; the table's blank is ''. The
    # channel type is written as the word the interface uses, and read
    # back through the same mapping — see test_channel_table_excel.
    from visualdynamics.core.unit_choices import shown_dimension

    def expected(name: str) -> object:
        value = table[name][0]
        if table.COLUMNS[name].kind in ('int', 'index'):
            return int(value) if str(value) else ''
        return (shown_dimension(str(value)) if name == 'channel_type'
                else str(value))

    assert [('' if c.value is None else c.value)
            for c in sheet[2]][:columns] == [
        expected(name) for name in table.column_names]


def test_excel_is_offered_for_channel_tables_alone():
    table = io.load(fixture_path('plate', 'channel_table.vdyn'))
    geometry = visualdynamics.import_file(fixture_path('plate', 'geometry.unv'))
    # matlab takes every object the project file holds; excel is
    # the one *spreadsheet* form, and a table's alone
    assert [e.name for e in io.exporters(table)] == ['excel', 'matlab']
    assert 'excel' not in [e.name for e in io.exporters(geometry)]
