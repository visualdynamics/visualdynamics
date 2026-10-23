"""A lone shape set's three views: mode table, MAC and shape values.

Brandon, 2026-10-22: a shape set showed its mode table, which could be
toggled, and its MAC, which could not. Now each of the three views has
a toggle on the table bar: the mode table, the MAC, and the shape
values (a row per DOF, headed by it, and a column per mode, M1, M2,
...). One always stays up, and the values are offered only when there
are few enough of them for a table.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path
from PySide6.QtCore import Qt

import visualdynamics
from visualdynamics.core.shapes import ShapeSet
from visualdynamics.gui import main_window as window_module
from visualdynamics.gui.object_tables import shape_values_model


def _shapes():
    return ShapeSet([10.0, 25.0, 40.0], [0.01, 0.02, 0.03],
                    ['101X+', '101Y+', '102Z+', '103Z+'],
                    [[1.0, 2.0, 3.0, 4.0],
                     [0.5, -0.5, 0.25, 0.0],
                     [1e-7, 2.0, 0.0, -3.5]])


def test_rows_are_dofs_and_columns_are_modes(qt_app):
    shapes = _shapes()
    model = shape_values_model(shapes)
    assert (model.rowCount(), model.columnCount()) == (4, 3)
    vertical, horizontal = Qt.Orientation.Vertical, Qt.Orientation.Horizontal
    assert [model.headerData(r, vertical) for r in range(4)] == \
        ['101X+', '101Y+', '102Z+', '103Z+']
    assert [model.headerData(c, horizontal) for c in range(3)] == \
        ['M1', 'M2', 'M3']
    assert model.headerData(1, horizontal, Qt.ItemDataRole.ToolTipRole) \
        == '25.0000 Hz'
    # row 1 (101Y+) of mode 3 is shape_matrix[2, 1], not [1, 2]
    assert model.data(model.index(1, 2)) == '2'
    assert model.data(model.index(0, 2)) == '1e-07'
    assert not model.flags(model.index(0, 0)) & Qt.ItemFlag.ItemIsEditable


def test_a_complex_value_shows_both_parts(qt_app):
    shapes = ShapeSet([10.0], [0.01], ['1X+'], [[0.5 - 0.25j]])
    model = shape_values_model(shapes)
    assert model.data(model.index(0, 0)) == '0.5-0.25j'


def _show(window, pump, shapes, name='Shapes'):
    window.add_object(name, shapes)
    item = window._item_for_object(name)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()


def _toggle(window, pump, action):
    action.trigger()
    pump()


def test_each_view_has_its_toggle(window, pump):
    _show(window, pump, _shapes())
    table, mac, values = (window.mode_table_action, window.mac_action,
                          window.shape_values_action)
    assert table.isVisible() and mac.isVisible() and values.isVisible()
    assert table.isChecked() and mac.isChecked() and not values.isChecked(), \
        'as it was: the mode table and the MAC'
    assert not window.shape_values_table.isVisibleTo(window)

    _toggle(window, pump, values)
    shown = window.shape_values_table
    assert shown.isVisibleTo(window)
    assert shown.model().rowCount() == 4
    assert shown.model().headerData(0, Qt.Orientation.Vertical) == '101X+'
    assert window.tables_row.sizes()[window.tables_row.indexOf(shown)] > 0

    _toggle(window, pump, mac)
    assert not window.mac_frame.isVisibleTo(window)
    assert window._mac_bars_page is None or \
        not window._mac_bars_page.isVisibleTo(window)
    assert not window.mac_bars_action.isVisible(), 'no 3D with no MAC'
    _toggle(window, pump, table)
    assert not window.table.isVisibleTo(window)
    assert shown.isVisibleTo(window), 'the values alone'
    _toggle(window, pump, mac)
    assert window.mac_bars_action.isVisible()


def test_the_values_come_back_wide_enough_to_read(window, pump):
    """A splitter child keeps the width it had while hidden, and the
    MAC once came back zero wide that way. The values table, asked for
    with its stored width at zero, opens readable: Qt gives a child
    shown again its room, so it needs none of the MAC's sharing."""
    _show(window, pump, _shapes())
    row, values = window.tables_row, window.shape_values_table
    row.setSizes([500, 0, 500])
    pump()
    _toggle(window, pump, window.shape_values_action)
    assert values.isVisibleTo(window)
    assert row.sizes()[row.indexOf(values)] >= 40


def test_the_last_view_up_cannot_be_turned_off(window, pump):
    _show(window, pump, _shapes())
    table, mac, values = (window.mode_table_action, window.mac_action,
                          window.shape_values_action)
    assert table.isEnabled() and mac.isEnabled() and values.isEnabled()
    _toggle(window, pump, mac)
    assert not table.isEnabled(), 'the mode table is the one left'
    assert mac.isEnabled() and values.isEnabled()
    _toggle(window, pump, values)
    assert table.isEnabled() and values.isEnabled()
    _toggle(window, pump, table)
    assert not values.isEnabled() and table.isEnabled()
    assert window.shape_values_table.isVisibleTo(window)


def test_too_many_values_are_not_offered(window, pump, monkeypatch):
    """Past the limit the toggle goes, and a set whose only view was the
    values opens on its mode table instead, the status line saying why."""
    _show(window, pump, _shapes())
    _toggle(window, pump, window.shape_values_action)
    _toggle(window, pump, window.mac_action)
    _toggle(window, pump, window.mode_table_action)
    assert window.shape_values_table.isVisibleTo(window)
    monkeypatch.setattr(window_module, 'SHAPE_VALUES_CELL_LIMIT', 11)
    _show(window, pump, _shapes(), name='Again')
    assert not window.shape_values_action.isVisible()
    assert not window.shape_values_table.isVisibleTo(window)
    assert window.table.isVisibleTo(window), 'the mode table came back'
    assert not window.mode_table_action.isEnabled()
    assert '12 shape values, too many for a table' in \
        window.statusBar().currentMessage()


def test_a_picked_mode_is_its_column(window, pump):
    _show(window, pump, _shapes())
    _toggle(window, pump, window.shape_values_action)
    window._item_for_object('Shapes').setExpanded(True)
    pump()
    window.record_grids['Shapes'].select_records([1])
    window.render_current()     # what a click on the cell runs
    pump()
    selection = window.shape_values_table.selectionModel()
    assert selection.isColumnSelected(1) and not selection.isColumnSelected(0)


def test_comparing_two_sets_keeps_its_own_bar(window, pump):
    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    _show(window, pump, shapes)
    _toggle(window, pump, window.mac_action)
    window.add_object('Other', shapes.copy() if hasattr(shapes, 'copy')
                      else visualdynamics.import_file(
                          fixture_path('plate', 'shapes.npy')))
    window.tree.clearSelection()
    for name in ('Shapes', 'Other'):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    assert not window.mac_action.isVisible()
    assert not window.shape_values_action.isVisible()
    assert window.mode_table_action.isEnabled()
    assert not window.shape_values_table.isVisibleTo(window)
    assert window.mac_frame.isVisibleTo(window) or \
        window._mac_bars_page.isVisibleTo(window), 'a comparison is its MAC'
    assert np.isfinite(window._compare['matrix']).all()


def test_a_script_reads_the_same_values():
    """Anything the window shows a script can have: the same headings and
    the same text, from one rule (`core.tables.shape_values`)."""
    project = visualdynamics.Project('p')
    project.add('Shapes', _shapes())
    headers, rows = project.table('Shapes', values=True)
    assert headers == ['DOF', 'M1', 'M2', 'M3']
    assert rows[1] == ['101Y+', '2', '-0.5', '2']
    model = shape_values_model(project['Shapes'])
    assert [model.data(model.index(1, c)) for c in range(3)] == rows[1][1:]
    assert project.table('Shapes')[0][0] == 'Mode', 'the mode list as before'
    project.add('Channels', visualdynamics.ChannelTable(
        {'channel': [1], 'node': [1], 'direction': ['X+'], 'unit': ['g']}))
    with pytest.raises(ValueError, match='only a shape set'):
        project.table('Channels', values=True)
