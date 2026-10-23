"""A channel table selected with a geometry draws the table's DOFs.

Brandon, 2026-10-22: with an empty channel table to fill in, selecting
it beside its geometry should show where the channels are, and a
drop-down on the 3-D view's bar should pick the references, the
responses or the monitors. The arrows turn on for the pair (and back
to as they were after it), and the drop-down opens on every channel.
"""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv
from conftest import fixture_path
from PySide6.QtCore import Qt

import visualdynamics
from visualdynamics.core.channel_table import ChannelTable
from visualdynamics.viz.geometry import add_dof_arrows, channel_dofs

ARROW_CELLS = pv.Arrow().n_cells


def _table():
    return ChannelTable({
        'channel': [1, 2, 3, 4],
        'node': [101, 102, 103, ''],
        'direction': ['Z+', 'X+', 'Y+', ''],
        'unit': ['g', 'lbf', 'g', ''],
        'role': ['response', 'reference', 'monitor', 'response'],
        'control': ['False', 'True', 'True', 'False']})


def test_the_roles_pick_the_channels():
    table = _table()
    assert channel_dofs(table) == (['101Z+', '102X+', '103Y+', ''],
                                   [False, True, False, False])
    assert channel_dofs(table, 'reference') == (['102X+'], [True]), \
        'a force points in, as a force record\'s arrows do'
    assert channel_dofs(table, 'response') == (['101Z+', ''], [False, False])
    assert channel_dofs(table, 'monitor') == (['103Y+'], [False])
    assert channel_dofs(table, 'control') == (['102X+', '103Y+'],
                                              [True, False]), \
        'by the flag, whatever the role'
    with pytest.raises(ValueError, match='not a role'):
        channel_dofs(table, 'drive')


def test_arrows_in_and_out_share_a_plot():
    """One call, one arrow length, each arrow its own way: out from node
    1 at x = 0, in to node 2 at x = 10, so nothing reaches past either."""
    geometry = visualdynamics.Geometry(node_id=[1, 2],
                                       node_xyz=[[0, 0, 0], [10, 0, 0]])
    plotter = pv.Plotter(off_screen=True)
    try:
        assert add_dof_arrows(plotter, geometry, ['1X+', '2X+'],
                              incoming=[False, True], name='t') == 2
        low, high = plotter.actors['t-arrows0'].GetBounds()[:2]
    finally:
        plotter.close()
    assert np.isclose(low, 0.0, atol=1e-9) and np.isclose(high, 10.0, atol=1e-9)


def test_a_script_draws_them_too(tmp_path):
    geometry = visualdynamics.Geometry(
        node_id=[101, 102, 103],
        node_xyz=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    path = tmp_path / 'channels.png'
    reference = geometry.plot_channels(_table(), 'reference',
                                       screenshot=str(path))
    assert path.exists()
    monitor = geometry.plot_channels(_table(), 'monitor',
                                     screenshot=str(tmp_path / 'm.png'))
    # the reference's red X arrow at node 102, the monitor's green Y at 103
    assert not np.array_equal(np.asarray(reference), np.asarray(monitor))


def _arrows(window):
    """How many channel arrows the 3-D view holds, by axis mesh."""
    return sum(actor.GetMapper().GetInput().GetNumberOfCells()
               for name, actor in window.scene.plotter.actors.items()
               if str(name).startswith('channel-dofs-0-arrows')) // ARROW_CELLS


def _pair(window, pump):
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4'),
                         fixture_path('plate', 'test_geometry.npz')])
    pump()
    window.tree.clearSelection()
    window.tree.setCurrentItem(window._item_for_object('Geometry'))
    window._item_for_object('Channel Table').setSelected(True)
    pump()


def test_the_pair_shows_every_channel_and_picks_by_role(window, pump):
    assert not window.dofs_action.isChecked()
    _pair(window, pump)
    assert window.dofs_action.isChecked(), 'on for the pair, unasked'
    combo = window.dofs_combo
    assert window.dofs_combo_action.isVisible()
    assert [combo.itemText(i) for i in range(combo.count())] == \
        ['All channels', 'References', 'Responses'], 'no monitors to offer'
    assert combo.currentData() == 'all'
    assert _arrows(window) == 13, 'every channel, two at node 101 and 1310'
    combo.setCurrentIndex(combo.findData('reference'))
    pump()
    assert _arrows(window) == 2
    combo.setCurrentIndex(combo.findData('response'))
    pump()
    assert _arrows(window) == 11


def _geometry_alone(window, pump):
    window.tree.clearSelection()
    window.tree.setCurrentItem(window._item_for_object('Geometry'))
    pump()


def test_the_arrows_go_back_to_as_they_were(window, pump):
    _pair(window, pump)
    _geometry_alone(window, pump)
    assert not window.dofs_action.isChecked(), \
        'a geometry alone keeps its own, off: not every node marked'
    _pair(window, pump)
    window.dofs_action.setChecked(False)
    pump()
    assert _arrows(window) == 0 and not window.dofs_combo_action.isVisible()
    _geometry_alone(window, pump)
    assert not window.dofs_action.isChecked()
    _pair(window, pump)
    assert window.dofs_action.isChecked(), 'on again for the pair'


def test_a_cell_typed_moves_the_arrows(window, pump):
    """A new table on its geometry is a layout being typed in: the arrow
    appears as the row says where it is, with no redraw of the table."""
    window.import_paths([fixture_path('plate', 'test_geometry.npz')])
    pump()
    name = window.project.new_channel_table()
    window.project.add_channels(name)
    window.show_object(name, select=False)
    window.tree.clearSelection()
    window.tree.setCurrentItem(window._item_for_object('Geometry'))
    window._item_for_object(name).setSelected(True)
    pump()
    assert _arrows(window) == 0
    assert '1 of 1 channels have no arrow' in window.statusBar().currentMessage()
    model = window.table.model()
    column = {model.headerData(c, Qt.Orientation.Horizontal): c
              for c in range(model.columnCount())}
    assert model.setData(model.index(0, column['Node']), '707',
                         Qt.ItemDataRole.EditRole)
    assert model.setData(model.index(0, column['Direction']), 'Y+',
                         Qt.ItemDataRole.EditRole)
    pump()
    assert window.table.model() is model, 'the table was not rebuilt'
    assert _arrows(window) == 1
    assert 'channel-dofs-0-arrows1' in window.scene.plotter.actors, 'along Y'
    assert model.setData(model.index(0, column['Direction']), 'Z+',
                         Qt.ItemDataRole.EditRole)
    pump()
    actors = window.scene.plotter.actors
    assert 'channel-dofs-0-arrows2' in actors
    assert 'channel-dofs-0-arrows1' not in actors, 'the Y arrow went with it'
    assert _arrows(window) == 1


def test_the_control_channels_are_a_choice(window, pump):
    """Brandon, 2026-10-22: the control DOFs, the channels with Control
    checked, offered only by a table that flags some."""
    window.import_paths([fixture_path('plate', 'random.nc4'),
                         fixture_path('plate', 'test_geometry.npz')])
    pump()
    window.tree.clearSelection()
    window.tree.setCurrentItem(window._item_for_object('Geometry'))
    window._item_for_object('Channel Table').setSelected(True)
    pump()
    combo = window.dofs_combo
    assert combo.itemText(1) == 'Control channels'
    table = window.objects['Channel Table']
    geometry = window.objects['Geometry']
    nodes = {int(node) for node in geometry.node_id}
    flagged = [dof for dof, on in zip(table.dof_strings(), table.controls())
               if on and int(dof[:-2]) in nodes]
    assert 0 < len(flagged) < table.num_channels
    combo.setCurrentIndex(combo.findData('control'))
    pump()
    assert _arrows(window) == len(flagged)


def test_a_table_flagging_no_control_offers_none(window, pump):
    _pair(window, pump)
    combo = window.dofs_combo
    assert combo.findData('control') == -1, 'the plate run flags none'
