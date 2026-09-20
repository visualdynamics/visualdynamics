"""The channel table's direction columns, derived from a geometry.

Each channel's measured direction as a unit vector in the geometry's
global system, the global axis it is nearest and the angle to it
(Brandon, 2026-09-20) — read from the geometry, never stored, so they
follow an edit to the node or the direction at once and can never
disagree with the geometry. In the window's table, in the report's,
and in a script's.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.channel_table import DERIVED_COLUMNS, ChannelTable, title_of


def _framed_geometry():
    """Node 101 measured in a frame turned 30° about Z, 102 global."""
    c, s = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))
    turned = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    identity = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 0, 0]]
    return visualdynamics.Geometry(
        node_id=[101, 102], node_xyz=np.zeros((2, 3)), node_disp_cs=[2, 1],
        cs_id=[1, 2], cs_name=['', ''], cs_type=[0, 0],
        cs_matrix=[identity, turned], length_unit='m')


def _table():
    return ChannelTable({
        'channel': [1, 2, 3, 4],
        'node': [101, 101, 102, 999],
        'direction': ['X+', 'Y-', 'Z+', 'X+'],
        'channel_type': ['acceleration'] * 4,
        'unit': ['g'] * 4,
    })


def test_a_channel_reads_its_direction_through_its_nodes_frame():
    table, geometry = _table(), _framed_geometry()
    c, s = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))
    vector, axis, angle = table.orientation(0, geometry)
    assert np.allclose(vector, [c, s, 0.0])
    assert axis == 'X+' and angle == pytest.approx(30.0)
    vector, axis, angle = table.orientation(1, geometry)
    assert np.allclose(vector, [s, -c, 0.0])
    assert axis == 'Y-' and angle == pytest.approx(30.0)
    vector, axis, angle = table.orientation(2, geometry)
    assert np.allclose(vector, [0.0, 0.0, 1.0])
    assert axis == 'Z+' and angle == pytest.approx(0.0)
    assert table.orientation(3, geometry) == (None, None, None), 'no such node'
    assert table.orientation(0, None) == (None, None, None), 'no geometry'
    assert table.derived_cells(0, geometry) == ['+0.866', '+0.500', '+0.000', 'X+', '30.0']
    assert table.derived_cells(3, geometry) == [''] * 5
    assert [title_of(n) for n in DERIVED_COLUMNS] == [
        'Unit X', 'Unit Y', 'Unit Z', 'Nearest Axis', 'Angle to Axis (deg)']


def test_the_windows_table_shows_them_only_beside_a_geometry(qt_app):
    from PySide6.QtCore import Qt

    from visualdynamics.gui.object_tables import channel_table_model

    table, geometry = _table(), _framed_geometry()
    bare = channel_table_model(table)
    assert bare.columnCount() == len(table.SCHEMA), 'nothing to derive from'
    model = channel_table_model(table, geometry=geometry)
    assert model.columnCount() == len(table.SCHEMA) + len(DERIVED_COLUMNS)
    headers = [model.headerData(c, Qt.Orientation.Horizontal)
               for c in range(model.columnCount())]
    assert headers[-5:] == [title_of(n) for n in DERIVED_COLUMNS]
    first = len(table.SCHEMA)
    cells = [model.data(model.index(0, first + k)) for k in range(5)]
    assert cells == ['+0.866', '+0.500', '+0.000', 'X+', '30.0']
    assert not model.flags(model.index(0, first)) & Qt.ItemFlag.ItemIsEditable, \
        'derived, not the table\'s to edit'


def test_the_derived_cells_follow_an_edit_at_once(qt_app):
    """Change the direction, or the node, and the row restates its
    derived cells in the same breath — the model announces the whole
    row changed, and the cells read the new direction."""
    from PySide6.QtCore import Qt

    from visualdynamics.gui.object_tables import channel_table_model

    table, geometry = _table(), _framed_geometry()
    model = channel_table_model(table, geometry=geometry)
    first = len(table.SCHEMA)
    changed = []
    model.dataChanged.connect(lambda a, b, *_: changed.append((a.column(), b.column())))
    direction = list(table.SCHEMA).index('direction')
    assert model.setData(model.index(0, direction), 'Z+', Qt.ItemDataRole.EditRole)
    assert changed[-1] == (0, model.columnCount() - 1), 'the whole row'
    assert [model.data(model.index(0, first + k)) for k in range(5)] == \
        ['+0.000', '+0.000', '+1.000', 'Z+', '0.0']
    node = list(table.SCHEMA).index('node')
    assert model.setData(model.index(0, node), '102', Qt.ItemDataRole.EditRole)
    assert changed[-1] == (0, model.columnCount() - 1)
    assert model.data(model.index(0, first + 3)) == 'Z+'
    assert model.setData(model.index(0, node), '555', Qt.ItemDataRole.EditRole)
    assert [model.data(model.index(0, first + k)) for k in range(5)] == [''] * 5, \
        'a node the geometry lacks: nothing claimed'


def test_the_window_reads_the_tables_own_geometry(window, pump):
    """The table shown in the window carries the columns of the geometry
    its group holds, and none without one."""
    from PySide6.QtCore import Qt

    geometry = _framed_geometry()
    table = ChannelTable({          # only nodes the geometry has: a link is refused otherwise
        'channel': [1, 2, 3], 'node': [101, 101, 102],
        'direction': ['X+', 'Y-', 'Z+'], 'channel_type': ['acceleration'] * 3,
        'unit': ['g'] * 3})
    window.add_object('Channels', table)
    window.show_object('Channels')
    pump()
    model = window.table.model()
    assert model.columnCount() == len(table.SCHEMA)
    window.add_object('Geometry', geometry)
    window.project.link('Channels', 'Geometry')
    window.show_object('Channels')
    pump()
    model = window.table.model()
    assert model.columnCount() == len(table.SCHEMA) + len(DERIVED_COLUMNS)
    assert model.headerData(model.columnCount() - 2, Qt.Orientation.Horizontal) == 'Nearest Axis'


def test_a_script_reads_the_same_table():
    from visualdynamics.core.tables import table_of

    table, geometry = _table(), _framed_geometry()
    headers, rows = table_of(table)
    assert headers == [title_of(n) for n in table.SCHEMA]
    headers, rows = table_of(table, geometry)
    assert headers[-5:] == [title_of(n) for n in DERIVED_COLUMNS]
    assert rows[0][-5:] == ['+0.866', '+0.500', '+0.000', 'X+', '30.0']
    assert rows[3][-5:] == [''] * 5
