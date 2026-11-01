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


def _two_models(window, pump):
    """A test geometry and its table in one group, and an FE geometry
    that shares the test's node numbers at other points — the FE mesh's
    node 101 is not the test's node 101."""
    test = visualdynamics.Geometry(
        node_id=[101, 102, 103],
        node_xyz=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], length_unit='m')
    fe = visualdynamics.Geometry(
        node_id=[101, 102, 103, 104],
        node_xyz=[[10, 0, 0], [11, 0, 0], [10, 1, 0], [11, 1, 0]],
        length_unit='m')
    window.add_object('Test', test)
    window.add_object('FE', fe)
    # a type and a model on each, so the sensors place as well
    window.add_object('Channels', ChannelTable({
        'channel': [1, 2, 3], 'node': [101, 102, 103],
        'direction': ['Z+', 'X+', 'Y+'], 'unit': ['g', 'lbf', 'g'],
        'channel_type': ['acceleration', 'force', 'acceleration'],
        'model': ['352A24', '208C02', '352A24'],
        'role': ['response', 'reference', 'response']}))
    window.project.link('Test', 'Channels')
    pump()
    return test, fe


def _select(window, pump, first, *others):
    window.tree.clearSelection()
    window.tree.setCurrentItem(window._item_for_object(first))
    for name in others:
        window._item_for_object(name).setSelected(True)
    pump()


def _arrows_drawn_on(window, pump, monkeypatch, first, *others):
    """The geometries the DOF arrows were drawn on once the whole
    selection is made — not on the way there, where the first geometry
    is drawn alone."""
    import visualdynamics.viz.geometry as viz_geometry

    _select(window, pump, first, *others)
    original = viz_geometry.add_dof_arrows
    drawn = []

    def spy(plotter, geometry, *args, **kwargs):
        drawn.append(geometry)
        return original(plotter, geometry, *args, **kwargs)

    monkeypatch.setattr(viz_geometry, 'add_dof_arrows', spy)
    window.render_current()
    pump()
    monkeypatch.setattr(viz_geometry, 'add_dof_arrows', original)
    return drawn


def test_a_linked_tables_arrows_go_on_its_own_geometry(window, pump,
                                                       monkeypatch):
    """Brandon, 2026-10-31: a channel table's arrows go only on the
    geometry it is linked with. Drawn on every geometry by node number,
    an FE mesh that reuses the test's node numbers got arrows at points
    the table says nothing about."""
    test, fe = _two_models(window, pump)
    drawn = _arrows_drawn_on(window, pump, monkeypatch,
                             'Test', 'FE', 'Channels')
    assert drawn and all(geometry is test for geometry in drawn), \
        'on the table\'s own geometry, not the FE mesh'
    assert _arrows(window) == 3

    drawn = _arrows_drawn_on(window, pump, monkeypatch, 'FE', 'Channels')
    assert drawn == [], 'its geometry is not drawn, so no arrows'
    message = window.statusBar().currentMessage()
    assert "'Channels' is linked with 'Test'" in message, message

    window.project.unlink('Channels')
    pump()
    drawn = _arrows_drawn_on(window, pump, monkeypatch,
                             'Test', 'FE', 'Channels')
    assert any(g is test for g in drawn) and any(g is fe for g in drawn), \
        'a table in no group can be held up against any geometry'


def test_a_linked_histories_arrows_go_on_its_own_geometry(window, pump,
                                                          monkeypatch):
    test, fe = _two_models(window, pump)
    t = np.arange(256) / 256.0
    history = visualdynamics.TimeHistory(
        t, np.random.default_rng(2).standard_normal((2, len(t))),
        response_dof=['101Z+', '103Y+'], ordinate_dim='acceleration')
    window.add_object('Run', history)
    window.project.link('Test', 'Run')
    pump()
    window.dofs_action.setChecked(True)
    drawn = _arrows_drawn_on(window, pump, monkeypatch, 'FE', 'Run')
    assert drawn == [], 'the run is linked with the test geometry'
    assert window._sensor_pairs == [], 'and so is its table'
    drawn = _arrows_drawn_on(window, pump, monkeypatch, 'Test', 'Run')
    assert drawn and all(geometry is test for geometry in drawn)
    assert [geometry for geometry, _t in window._sensor_pairs] == [test]

    window.project.unlink('Run')
    pump()
    drawn = _arrows_drawn_on(window, pump, monkeypatch, 'FE', 'Run')
    assert drawn and all(geometry is fe for geometry in drawn), \
        'a run in no group goes on the geometry it is animated on'


def test_a_linked_tables_sensors_go_on_its_own_geometry(window, pump):
    from visualdynamics.core.sensors import place

    test, fe = _two_models(window, pump)
    table = window.objects['Channels']
    assert place(table, fe)[0], 'the FE mesh would place them by number'
    _select(window, pump, 'FE', 'Channels')
    assert window._sensor_pairs == [], 'its own geometry is not drawn'
    _select(window, pump, 'Test', 'FE', 'Channels')
    assert [geometry for geometry, _t in window._sensor_pairs] == [test]
    window.project.unlink('Channels')
    _select(window, pump, 'FE', 'Channels')
    assert [geometry for geometry, _t in window._sensor_pairs] == [fe]
