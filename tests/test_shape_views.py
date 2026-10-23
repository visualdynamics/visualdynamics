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


# ---- at a channel table's DOFs, in three forms ------------------------------------
#
# Brandon, 2026-10-23: the kinematic transformation at a channel table's
# references or responses, beside the shapes, read as the transform reads
# them. One matrix (`transform.matrix_at`) for the window and a script.


def _rigid_project():
    """Four nodes of a square and their rigid shapes, with a channel
    table of twelve references (two of them written negative), four Z
    responses, and a monitor the set lacks; the responses and the
    monitor have Control checked."""
    from visualdynamics.core.rigid import MassProperties, rigid_body_shapes

    geometry = visualdynamics.Geometry(
        [1, 2, 3, 4], [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        length_unit='m')
    references = ['1X+', '1Y+', '1Z+', '2X-', '2Y+', '2Z+', '3X+', '3Y-',
                  '3Z+', '4X+', '4Y+', '4Z+']
    responses = ['1Z+', '2Z+', '3Z+', '4Z+']
    dofs = references + responses + ['9Z+']
    table = visualdynamics.ChannelTable({
        'channel': list(range(1, len(dofs) + 1)),
        'node': [dof[:-2] for dof in dofs],
        'direction': [dof[-2:] for dof in dofs],
        'unit': ['lbf'] * 12 + ['g'] * 5,
        'role': ['reference'] * 12 + ['response'] * 4 + ['monitor'],
        # the monitor is a control channel too, so the control choice is
        # not the responses under another name
        'control': ['False'] * 12 + ['True'] * 5})
    project = visualdynamics.Project('p')
    project.add('Square', geometry)
    project.add('Channels', table)
    project.add('Rigid', rigid_body_shapes(geometry, MassProperties((0.5, 0.5, 0))))
    project.link('Square', 'Channels')
    project.link('Square', 'Rigid')
    return project, references, responses


def test_the_shapes_at_a_choice_of_dofs_honor_the_sign():
    from visualdynamics.core.transform import matrix_at

    project, _references, _responses = _rigid_project()
    shapes = project['Rigid']
    view = matrix_at(shapes, ['2X+', '2X-', '9Z+'])
    assert view.rows == ['2X+', '2X-', '9Z+'] and view.columns[0] == 'M1'
    assert np.allclose(view.values[1], -view.values[0]), '2X- is minus 2X+'
    assert np.isnan(view.values[2]).all() and view.missing == ['9Z+']
    whole = matrix_at(shapes)
    assert np.array_equal(whole.values, np.asarray(shapes.shape_matrix).T)


def test_the_transformations_are_what_the_transform_applies():
    """Φ⁺ times a motion is the modal response `to_modal` makes, and Φᵀ
    times a force is the modal force, at the same DOFs and signs."""
    from visualdynamics.core.transform import matrix_at, to_modal

    project, references, _responses = _rigid_project()
    shapes = project['Rigid']
    rng = np.random.default_rng(4)
    t = np.arange(64) / 64.0
    values = rng.standard_normal((12, 64))
    for form, quantity in (('response', 'acceleration'), ('force', 'force')):
        view = matrix_at(shapes, references, form)
        assert view.rows == [f'M{k}' for k in range(1, 7)]
        assert view.columns == references
        history = visualdynamics.TimeHistory(
            t, values, response_dof=references, ordinate_dim=quantity)
        modal, _report = to_modal(history, shapes)
        assert np.allclose(view.values @ values, modal.ordinate), form
    response = matrix_at(shapes, references, 'response').values
    phi = matrix_at(shapes, references).values
    assert np.allclose(response @ phi, np.eye(6)), 'Φ⁺Φ = I'


def test_too_few_dofs_refuse_the_response_transformation():
    from visualdynamics.core.transform import matrix_at

    project, _references, responses = _rigid_project()
    with pytest.raises(ValueError, match='resolve only 3 of 6 modes'):
        matrix_at(project['Rigid'], responses, 'response')
    assert matrix_at(project['Rigid'], responses, 'force').values.shape == (6, 4)
    with pytest.raises(ValueError, match='not a form'):
        matrix_at(project['Rigid'], responses, 'inverse')
    with pytest.raises(ValueError, match='none of these'):
        matrix_at(project['Rigid'], ['9Z+'], 'force')


def test_a_tables_channels_are_chosen_by_one_rule():
    project, references, responses = _rigid_project()
    table = project['Channels']
    assert table.choices() == ['all', 'control', 'reference', 'response',
                               'monitor']
    assert [table.dof_strings()[i] for i in table.chosen('reference')] == references
    assert [table.dof_strings()[i] for i in table.chosen('control')] == \
        responses + ['9Z+']
    with pytest.raises(ValueError, match='not a choice'):
        table.chosen('drives')


def test_a_script_reads_a_choice_of_channels_in_a_form():
    project, references, _responses = _rigid_project()
    headers, rows = project.table('Rigid', values=True, dofs='reference',
                                  form='force')
    assert headers == ['Mode'] + references
    assert rows[0][0] == 'M1' and rows[0][4] == '-1', "M1 at 2X- is -1"
    headers, rows = project.table('Rigid', values=True, dofs='monitor')
    assert rows == [['9Z+ (not in the set)', '', '', '', '', '', '']]
    project.unlink('Channels')
    with pytest.raises(ValueError, match='no channel table'):
        project.table('Rigid', values=True, dofs='reference')


def test_the_window_offers_the_choices_and_the_forms(window, pump):
    project, references, _responses = _rigid_project()
    for name in project.names:
        window.add_object(name, project[name])
    for other in ('Channels', 'Rigid'):
        window.project.link('Square', other)
    window.tree.clearSelection()
    item = window._item_for_object('Rigid')
    window.tree.setCurrentItem(item)
    item.setSelected(True)
    pump()
    _toggle(window, pump, window.shape_values_action)
    dofs, form = window.values_dofs_combo, window.values_form_combo
    assert window.values_dofs_action.isVisible()
    assert window.values_form_action.isVisible()
    assert [dofs.itemText(i) for i in range(dofs.count())] == [
        'All DOFs', 'Control channels', 'References', 'Responses', 'Monitors']
    dofs.setCurrentIndex(dofs.findData('reference'))
    pump()
    model = window.shape_values_table.model()
    labels = [model.headerData(r, Qt.Orientation.Vertical)
              for r in range(model.rowCount())]
    assert labels == references
    assert model.data(model.index(3, 0)) == '-1', '2X- moves against M1'
    form.setCurrentIndex(form.findData('force'))
    pump()
    model = window.shape_values_table.model()
    assert (model.rowCount(), model.columnCount()) == (6, 12)
    assert model.headerData(0, Qt.Orientation.Vertical) == 'M1'
    dofs.setCurrentIndex(dofs.findData('response'))
    form.setCurrentIndex(form.findData('response'))
    pump()
    assert window.shape_values_table.model() is None
    assert 'resolve only 3 of 6 modes' in window.statusBar().currentMessage()
    dofs.setCurrentIndex(dofs.findData('monitor'))
    form.setCurrentIndex(form.findData('shapes'))
    pump()
    assert '1 monitors, 1 not in the set' in window.statusBar().currentMessage()
    window.shape_values_action.trigger()
    pump()
    assert not window.values_form_action.isVisible(), 'away with the values'
    assert not window.values_dofs_action.isVisible()


def test_a_picked_mode_is_a_row_of_a_transformation(window, pump):
    """A mode is a column of the shapes and a row of a transformation."""
    project, _references, _responses = _rigid_project()
    window.add_object('Rigid', project['Rigid'])
    _show(window, pump, project['Rigid'], name='Again')
    _toggle(window, pump, window.shape_values_action)
    form = window.values_form_combo
    form.setCurrentIndex(form.findData('force'))
    pump()
    window._item_for_object('Again').setExpanded(True)
    pump()
    window.record_grids['Again'].select_records([2])
    window.render_current()     # what a click on the cell runs
    pump()
    selection = window.shape_values_table.selectionModel()
    assert selection.isRowSelected(2) and not selection.isColumnSelected(2)
