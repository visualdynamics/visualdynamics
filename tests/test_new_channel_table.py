"""An empty channel table, filled in by hand.

A geometry could be started empty and built in the app; a channel table
could only be imported (Brandon, 2026-10-22: the user should be able to
create an empty channel table just as they can an empty geometry). New
Channel Table on the project's bar makes one, and Add Channel on the
table's bar gives it rows to type into.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QSize, Qt

import visualdynamics
from visualdynamics import io
from visualdynamics.core.channel_table import ChannelTable


def _empty():
    return ChannelTable({name: [] for name in ChannelTable.CORE})


def test_channels_are_numbered_on_from_the_highest():
    table = _empty()
    assert table.add_channels() == [1], 'an empty table starts at 1'
    assert table.add_channels(2) == [2, 3]
    table = ChannelTable({'channel': [3, 7], 'node': [101, 102],
                          'direction': ['X+', 'Y+'], 'unit': ['g', 'g']})
    assert table.add_channels() == [8], 'after the highest, not the count'
    assert table.dof_strings()[:2] == ['101X+', '102Y+'], 'the rows kept'


def test_a_new_channel_is_blank_but_its_number():
    table = _empty()
    table.add_channels()
    for name, spec in ChannelTable.COLUMNS.items():
        if name == 'channel':
            continue
        assert str(table[name][0]) == spec.default, name
    assert table['daq_range'][0] == '' and table['sensor_range'][0] == '', \
        'unstated until someone says'


def test_no_channels_cannot_be_added():
    with pytest.raises(ValueError, match='at least one'):
        _empty().add_channels(0)


def test_an_empty_table_has_a_filled_tables_column_types():
    """The type column was a bare list, which pandas read as floats when
    there were no channels in it, and a typed value would not go in."""
    empty, filled = _empty(), ChannelTable(
        {'channel': [1], 'node': [1], 'direction': ['X+'], 'unit': ['g']})
    assert dict(empty.frame.dtypes) == dict(filled.frame.dtypes)
    empty.add_channels()
    empty.set_cell('channel_type', 0, 'acceleration')
    assert empty.types() == ['acceleration']


def test_the_project_makes_one_and_adds_to_it():
    project = visualdynamics.Project('p')
    name = project.new_channel_table()
    assert name == 'Channel Table' and project[name].num_channels == 0
    assert 'add_channels' in dict(project.verbs(name))
    assert project.add_channels(name, 2) == [1, 2]
    assert project[name].num_channels == 2
    with pytest.raises(TypeError, match='not a channel table'):
        project.add_channels(project.new_geometry())


@pytest.mark.parametrize('suffix, fmt', [('xlsx', 'excel'), ('h5', 'escdf')])
def test_an_empty_table_reads_back(tmp_path, suffix, fmt):
    """What can be written can be read: Excel used to refuse a header
    with nothing under it."""
    path = tmp_path / f'empty.{suffix}'
    io.export_file(_empty(), str(path), format=fmt)
    back = io.import_file(str(path))
    back = back if isinstance(back, ChannelTable) else next(
        obj for obj in back.values() if isinstance(obj, ChannelTable))
    assert back.num_channels == 0


def test_an_empty_table_saves_and_opens(tmp_path):
    project = visualdynamics.Project('p')
    project.new_channel_table()
    project.save(tmp_path / 'p.vdyn')
    opened = visualdynamics.Project.open(tmp_path / 'p.vdyn')
    assert opened['Channel Table'] == project['Channel Table']
    assert opened['Channel Table'].num_channels == 0


def _acts(window):
    return {verb: (icon, call) for verb, _label, icon, call, _tip
            in window.acts_for()}


def test_the_app_makes_one_and_fills_it_in(window, pump):
    """New Channel Table on the project's bar, Add Channel on the table's,
    a cell typed into the new row, and the session replays to the same
    table."""
    window.tree.clearSelection()
    window.test_item.setSelected(True)
    pump()
    icon, make = _acts(window)['new_channel_table']
    assert icon == 'new_channel_table'
    make()
    pump()
    table = window.objects['Channel Table']
    assert table.num_channels == 0
    assert window.current_object() is table, 'the new table is selected'
    acts = _acts(window)
    assert acts['add_channels'][0] == 'add_channel'
    acts['add_channels'][1]()
    pump()
    assert table.num_channels == 1
    model = window.table.model()
    # the view's own count: the model's reads the table live, so it
    # would say 1 whether or not the view was told of the row
    assert window.table.verticalHeader().count() == 1, 'the new row is on screen'
    assert window.table.selectionModel().isRowSelected(0)
    assert 'added channel 1' in window.statusBar().currentMessage()
    acts = _acts(window)
    acts['add_channels'][1]()
    pump()
    assert window.table.verticalHeader().count() == 2
    assert window.table.selectionModel().isRowSelected(1), 'the cursor follows'
    model = window.table.model()
    node = next(c for c in range(model.columnCount())
                if model.headerData(c, Qt.Orientation.Horizontal) == 'Node')
    assert model.setData(model.index(1, node), '101', Qt.ItemDataRole.EditRole)
    assert table.dof_strings() == ['', '101']

    room: dict = {}
    exec(window.project.session_script(), room)          # noqa: S102
    replayed = room['project']['Channel Table']
    assert replayed == table


def test_the_new_icons_say_what_they_make(qt_app):
    from visualdynamics.gui.icons import control_icon

    def ink(name):
        image = control_icon(name).pixmap(QSize(64, 64)).toImage()
        return {(x, y) for x in range(64) for y in range(64)
                if image.pixelColor(x, y).alpha() > 60}

    image = control_icon('new_channel_table').pixmap(QSize(64, 64)).toImage()
    green = image.pixelColor(42, 42)          # on the disc, off the bars
    assert green.green() > 120 and green.red() < 100, 'the plus is drawn'
    assert ink('new_channel_table') - ink('table'), 'more than the table'
    assert ink('add_channel') != ink('new_channel_table')
    image = control_icon('add_channel').pixmap(QSize(64, 64)).toImage()
    green = image.pixelColor(26, 44)          # the disc under the row
    assert green.green() > 120 and green.red() < 100, 'the plus is drawn'


# ---- red until placed, and named while it is --------------------------------


def test_a_channel_not_yet_placed_is_named_by_its_number():
    """A blank channel listed as '', and the message ended at its colon."""
    from visualdynamics.compatibility import check_object

    geometry = visualdynamics.Geometry(node_id=[1, 2],
                                       node_xyz=[[0, 0, 0], [1, 0, 0]])
    table = ChannelTable({'channel': [1, 2, 3, 4, 5],
                          'node': ['', 2, '', 9, 9],
                          'direction': ['', '', 'X+', 'X+', ''],
                          'unit': [''] * 5})
    issue = check_object('Channels', table, geometry, 'Plate')
    # channel 2 is at node 2, on the geometry: no direction yet is not
    # off the geometry
    assert issue.message == (
        "4 of 5 channels are not in geometry 'Plate': "
        'channel 1 (no node or direction yet), channel 3 (no node yet), '
        '9X+, channel 5 at node 9 (no direction yet)')
    assert issue.missing_dofs == ['9X+'], 'only what is a DOF'
    assert issue.sub_items == [0, 2, 3, 4]


def test_the_row_stops_being_red_once_its_channels_are_placed(window, pump):
    """Typing a node and a direction re-runs the check: the red followed
    only the next unrelated check (Brandon, 2026-10-22, Project.vdyn)."""
    window.add_object('Plate', visualdynamics.Geometry(
        node_id=[1, 2], node_xyz=[[0, 0, 0], [1, 0, 0]], length_unit='in'))
    name = window.project.new_channel_table()
    window.show_object(name)
    pump()
    window.add_channel_act()
    pump()

    def red():
        return window._item_for_object(name).data(
            0, Qt.ItemDataRole.ForegroundRole) is not None

    assert red(), 'a channel with no node is not on the geometry'
    assert 'channel 1 (no node or direction yet)' in \
        window._item_for_object(name).toolTip(0)
    model = window.table.model()
    column = {model.headerData(c, Qt.Orientation.Horizontal): c
              for c in range(model.columnCount())}
    assert model.setData(model.index(0, column['Node']), '2',
                         Qt.ItemDataRole.EditRole)
    pump()
    assert not red(), 'at a node of the geometry now'
    assert window.table.model() is model, 'the table being typed in stays'
    assert model.setData(model.index(0, column['Direction']), 'Y+',
                         Qt.ItemDataRole.EditRole)
    pump()
    assert not red()
    assert model.setData(model.index(0, column['Node']), '7',
                         Qt.ItemDataRole.EditRole)
    pump()
    assert red(), 'and off it again, at once'
    assert '7Y+' in window._item_for_object(name).toolTip(0)
