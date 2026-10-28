"""A channel table carries the run's control loops.

A combined Rattlesnake run has one channel table and one environment
group per loop, and the loops may control different channels through
different matrices. Brandon (2026-10-28) opened such a file and found
the table marked for the random environment and not the sine: the
Control column was marked by matching each specification's DOFs
against the table, and a transformed sine's specification is over its
matrix's rows, '1', '2', '3', which match nothing. One table now holds
the instrumentation once and an `Environment` per loop — its control
channels and drives by number, and its response and reference
transformations — and the Control column is their union. The window
picks the environment and the view on the table's bar.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest
from conftest import fixture_path
from PySide6.QtCore import Qt

import visualdynamics
from visualdynamics.core.channel_table import ChannelTable, Environment

EDIT = Qt.ItemDataRole.EditRole
VERTICAL, HORIZONTAL = Qt.Orientation.Vertical, Qt.Orientation.Horizontal


def _table():
    return ChannelTable({
        'channel': [1, 2, 3, 4, 5],
        'node': ['101', '104', '110', '101', '110'],
        'direction': ['Z+'] * 5,
        'unit': ['m/s^2'] * 3 + ['N'] * 2,
        'role': ['response'] * 3 + ['reference'] * 2})


SINE = Environment(controls=[1, 2, 3], drives=[4, 5],
                   response_transformation=[[1 / 3, 1 / 3, 1 / 3],
                                            [0.5, 0, -0.5]],
                   reference_transformation=[[1, 1]])
RANDOM = Environment(controls=[2, 3], drives=[4, 5])


def _loops():
    table = _table()
    table.set_environment('Sine', SINE)
    table.set_environment('Random', RANDOM)
    return table


# --- the record -------------------------------------------------------


def test_the_control_column_is_the_union_of_the_loops():
    table = _loops()
    assert list(table.controls()) == [True, True, True, False, False]
    assert list(table.controls('Sine')) == [True, True, True, False, False]
    assert list(table.controls('Random')) == [False, True, True, False, False]
    assert list(table.drives('Random')) == [False, False, False, True, True]
    assert table.chosen('control') == [0, 1, 2], 'one rule, the column'


def test_a_loop_names_channels_the_table_has_once_each():
    table = _table()
    with pytest.raises(ValueError, match='no such channels: \\[9\\]'):
        table.set_environment('Sine', Environment(controls=[1, 9]))
    with pytest.raises(ValueError, match='listed twice'):
        table.set_environment('Sine', Environment(controls=[1, 1]))
    with pytest.raises(ValueError, match='needs a name'):
        table.set_environment(' ', Environment(controls=[1]))
    assert not table.environments, 'nothing recorded on a refusal'


def test_a_transformation_has_a_column_per_channel():
    table = _table()
    with pytest.raises(ValueError, match=r'response transformation is \(2, 2\) '
                                         r'but the environment has 3 controls'):
        table.set_environment('Sine', Environment(
            controls=[1, 2, 3], response_transformation=np.eye(2)))
    with pytest.raises(ValueError, match='reference transformation'):
        table.set_environment('Sine', Environment(
            drives=[4], reference_transformation=[[1, 1]]))
    with pytest.raises(ValueError, match='is not a transformation'):
        SINE.matrix('shapes')


def test_the_column_is_set_through_the_loop_while_there_is_one():
    table = _loops()
    with pytest.raises(ValueError, match='per environment'):
        table.set_cell('control', 0, 'True')
    table.set_control('Random', 0, 'True')
    assert table.environments['Random'].controls == [2, 3, 1], \
        'joins after the ones it has, in the controller\'s order'
    table.set_control('Random', 1, False)
    assert table.environments['Random'].controls == [3, 1]
    assert list(table.controls()) == [True, True, True, False, False], \
        'the union still has channel 2 from the sine'
    table.set_control('Random', 3, 'yes')
    assert list(table.controls()) == [True, True, True, True, False], \
        'the column follows the loop at once'
    table.set_control('Random', 3, 'no')
    assert list(table.controls()) == [True, True, True, False, False]
    table.remove_environment('Sine')
    assert list(table.controls()) == [True, False, True, False, False]
    with pytest.raises(ValueError, match="no environment 'Sine'; the table "
                                         "has Random"):
        table.controls('Sine')
    table.remove_environment('Random')
    with pytest.raises(ValueError, match='the table has none'):
        table.controls('Random')


def test_a_transformed_loops_controls_are_its_matrixs_columns():
    table = _loops()
    with pytest.raises(ValueError, match='transforms its control channels'):
        table.set_control('Sine', 4, True)
    assert table.environments['Sine'].controls == [1, 2, 3]


def test_a_deleted_channel_leaves_the_loop_and_its_column_leaves_the_matrix():
    table = _loops()
    table.delete_channels([1, 4])
    sine = table.environments['Sine']
    assert sine.controls == [1, 3] and sine.drives == [4]
    assert np.allclose(sine.response_transformation, [[1 / 3, 1 / 3],
                                                      [0.5, -0.5]])
    assert np.allclose(sine.reference_transformation, [[1]])
    assert table.environments['Random'].controls == [3]
    assert list(table.controls()) == [True, True, False]


def test_the_transformation_reads_as_a_labeled_matrix():
    table = _loops()
    view = table.transformation('Sine')
    assert (view.form, view.rows) == ('response', ['1', '2'])
    assert view.columns == ['101Z+', '104Z+', '110Z+']
    assert np.allclose(view.values, SINE.response_transformation)
    view = table.transformation('Sine', 'reference')
    assert view.columns == ['101Z+', '110Z+'] and view.rows == ['1']
    with pytest.raises(ValueError, match='holds no response transformation'):
        table.transformation('Random')


def test_loops_count_in_equality_and_copy():
    assert _loops() == _loops()
    assert _loops() != _table()
    other = _loops()
    other.set_control('Random', 0, True)
    assert _loops() != other
    twin = copy.deepcopy(_loops())
    assert twin == _loops()
    twin.environments['Sine'].response_transformation[0, 0] = 9.0
    assert twin != _loops(), 'the copy has its own matrices'


def test_the_environments_pass_through_the_constructor_and_ride_the_file(tmp_path):
    table = ChannelTable(_table().frame, environments={'Sine': SINE,
                                                       'Random': RANDOM})
    assert list(table.environments) == ['Sine', 'Random']
    path = tmp_path / 'loops.vdyn'
    table.save(path)
    back = visualdynamics.load(path)
    assert back == table
    assert back.environments['Sine'] == SINE
    assert back.environments['Random'].response_transformation is None
    plain = _table()
    plain.save(tmp_path / 'plain.vdyn')
    assert not visualdynamics.load(tmp_path / 'plain.vdyn').environments


def test_a_merged_table_carries_no_loops():
    project = visualdynamics.Project('p')
    project.add('A', _loops())
    project.add('B', _loops())
    merged = project.merge('A', 'B')
    assert not project[merged].environments
    assert list(project[merged].controls()) == [True, True, True, False, False]


# --- the import -------------------------------------------------------


def _write_mixed(path, *, reference=None, sine_matrix=True):
    """A combined run: the sine controls channels 1–4 through a
    matrix, the random controls 3–6 as they are, both drive the two
    force channels; the random carries an FRF so a spectrum exists."""
    import netCDF4

    channels, drives = 8, 2
    with netCDF4.Dataset(str(path), 'w', format='NETCDF4') as ds:
        ds.sample_rate = 256.0
        ds.createDimension('response_channels', channels)
        ds.createDimension('time_samples', 64)
        ds.createVariable('time_data', 'f8', ('response_channels',
                                              'time_samples'))[...] = 0.0
        group = ds.createGroup('channels')
        for name, values in (
                ('node_number', [str(101 + i) for i in range(channels)]),
                ('node_direction', ['Z+'] * channels),
                ('unit', ['m/s^2'] * 6 + ['N'] * drives),
                ('feedback_device', [''] * 6 + ['Input'] * drives)):
            group.createVariable(name, str, ('response_channels',))[:] = \
                np.array(values, dtype=object)
        ds.createDimension('num_environments', 2)
        ds.createVariable('environment_names', str,
                          ('num_environments',))[:] = np.array(
                              ['Sine', 'Random'], dtype=object)
        ds.createVariable('environment_types', int,
                          ('num_environments',))[:] = [3, 0]
        sine = ds.createGroup('Sine')
        sine.createDimension('control_channels', 4)
        sine.createVariable('control_channel_indices', 'i4',
                            ('control_channels',))[...] = [0, 1, 2, 3]
        if sine_matrix:
            sine.createDimension('response_transformation_rows', 2)
            sine.createDimension('response_transformation_cols', 4)
            sine.createVariable('response_transformation_matrix', 'f8',
                                ('response_transformation_rows',
                                 'response_transformation_cols'))[...] = \
                [[0.25] * 4, [0.5, 0.5, -0.5, -0.5]]
        random = ds.createGroup('Random')
        random.samples_per_frame = 32
        random.frames_in_cpsd = 2
        random.createDimension('control_channels', 4)
        random.createVariable('control_channel_indices', 'i4',
                              ('control_channels',))[...] = [2, 3, 4, 5]
        if reference is not None:
            reference = np.asarray(reference, dtype=float)
            random.createDimension('reference_transformation_rows',
                                   reference.shape[0])
            random.createDimension('reference_transformation_cols',
                                   reference.shape[1])
            random.createVariable('reference_transformation_matrix', 'f8',
                                  ('reference_transformation_rows',
                                   'reference_transformation_cols'))[...] = \
                reference
        random.createDimension('fft_lines', 3)
        random.createDimension('specification_channels', 4)
        random.createDimension('drive_channels', drives)
        for name in ('frf_data_real', 'frf_data_imag'):
            random.createVariable(name, 'f8', ('fft_lines',
                                               'specification_channels',
                                               'drive_channels'))[...] = 1.0
    return str(path)


def test_each_environments_controls_are_marked_from_its_indices(tmp_path):
    """The bug: the sine's controls went unmarked because its
    specification names the matrix's rows, not the channels."""
    out = visualdynamics.import_file(_write_mixed(tmp_path / 'mixed.nc4',
                                                  reference=[[1, -1]]))
    table = out['channel_table']
    assert list(table.environments) == ['Sine', 'Random']
    sine, random = table.environments['Sine'], table.environments['Random']
    assert sine.controls == [1, 2, 3, 4] and random.controls == [3, 4, 5, 6]
    assert sine.drives == [7, 8] == random.drives
    assert np.allclose(sine.response_transformation[1], [0.5, 0.5, -0.5, -0.5])
    assert np.allclose(random.reference_transformation, [[1, -1]])
    assert sine.reference_transformation is None
    assert list(table.controls()) == [True] * 6 + [False] * 2, 'the union'
    assert list(table.controls('Sine')) == [True] * 4 + [False] * 4
    view = table.transformation('Random', 'reference')
    assert view.columns == ['107Z+', '108Z+']


def test_a_reference_transformation_that_does_not_fit_the_drives_is_left_out(
        tmp_path):
    path = _write_mixed(tmp_path / 'odd.nc4', reference=[[1, 1, 1]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = visualdynamics.import_file(path)
    assert any('reference transformation is (1, 3) but the run drives 2'
               in str(w.message) for w in caught)
    assert out['channel_table'].environments['Random'].reference_transformation \
        is None


def test_the_real_mixed_run_marks_both_loops():
    """Brandon's layout, in the repository's own fixtures: a random
    over eight channels beside a sine transformed to three rows."""
    import os
    path = os.path.join('stressdata', 'plate', 'mixed_transformed.nc4')
    if not os.path.exists(path):
        pytest.skip('the stress data is not on this machine')
    table = visualdynamics.import_file(path, stop=0.1)['channel_table']
    assert table.environments['Sine'].response_transformation.shape == (3, 8)
    assert table.controls().sum() == 8
    assert table.controls('Sine').sum() == 8


# --- the script's view ----------------------------------------------------


def test_a_script_reads_the_loops_control_column_and_its_matrix():
    project = visualdynamics.Project('p')
    project.add('Channels', _loops())
    headers, rows = project.table('Channels', environment='Random')
    control = headers.index('Control')
    assert [row[control] for row in rows] == ['False', 'True', 'True',
                                              'False', 'False']
    headers, rows = project.table('Channels', environment='Sine',
                                  form='response')
    assert headers == ['Row', '101Z+', '104Z+', '110Z+']
    assert rows[1][0] == '2' and rows[1][2] == '0'
    assert float(rows[0][1]) == pytest.approx(1 / 3, abs=1e-4)
    with pytest.raises(ValueError, match="no environment 'Shock'"):
        project.table('Channels', environment='Shock')
    project.add('Shapes', visualdynamics.import_file(
        fixture_path('plate', 'shapes.npy')))
    with pytest.raises(ValueError, match='only a channel table'):
        project.table('Shapes', environment='Sine')


def test_the_virtual_rows_of_a_spectrum_are_tied_to_their_matrix(tmp_path):
    out = visualdynamics.import_file(_write_mixed(tmp_path / 'mixed.nc4'))
    project = visualdynamics.Project('p')
    for name, obj in out.items():
        project.add(name, obj)
    project.link('channel_table', 'Random_frf')
    # the random's FRF is over its own channels, so nothing is virtual
    assert project.transformed_rows('Random_frf') == []
    # a sine spectrum over the matrix's rows, '1' and '2'
    frf = out['Random_frf']
    virtual = type(frf)(abscissa=frf.abscissa, ordinate=frf.ordinate[:2],
                        response_dof=['1', '2'],
                        reference_dof=list(frf.reference_dof[:2]))
    project.add('Sine_frf', virtual)
    project.link('channel_table', 'Sine_frf')
    assert project.transformed_rows('Sine_frf') == [
        ('channel_table', 'Sine', ['1', '2'])]


# --- the window -----------------------------------------------------------


def _show(window, pump, name):
    window.tree.clearSelection()
    item = window._item_for_object(name)
    window.tree.setCurrentItem(item)
    item.setSelected(True)
    pump()


def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def _checked(model, column):
    """The Control column's boxes, top to bottom."""
    return [model.data(model.index(r, column), Qt.ItemDataRole.CheckStateRole)
            == Qt.CheckState.Checked for r in range(model.rowCount())]


def test_the_bar_offers_the_environment_and_the_view(window, pump):
    window.add_object('Channels', _loops())
    _show(window, pump, 'Channels')
    assert window.environment_action.isVisible()
    assert window.table_view_action.isVisible()
    assert window.table_bar.isVisible()
    assert _items(window.environment_combo) == ['Sine', 'Random']
    assert _items(window.table_view_combo) == [
        'Channels', 'Response transformation', 'Reference transformation']
    model = window.table.model()
    control = next(c for c in range(model.columnCount())
                   if model.headerData(c, HORIZONTAL) == 'Control')
    assert _checked(model, control) == [True, True, True, False, False]

    window.environment_combo.setCurrentIndex(1)
    pump()
    assert _items(window.table_view_combo) == ['Channels'], \
        'the random holds no transformation'
    assert not window.table_view_action.isVisible()
    model = window.table.model()
    assert _checked(model, control) == [False, True, True, False, False]

    # an edit of the Control column is the picked loop's, and journaled so
    model.setData(model.index(0, control), 'True', EDIT)
    pump()
    assert window.project['Channels'].environments['Random'].controls == [2, 3, 1]
    assert window.project.journal[-1] == \
        "project['Channels'].set_control('Random', 0, 'True')"


def test_the_view_shows_the_matrix_in_the_tables_place(window, pump):
    window.add_object('Channels', _loops())
    _show(window, pump, 'Channels')
    view = window.table_view_combo
    view.setCurrentIndex(view.findData('response'))
    pump()
    model = window.table.model()
    assert [model.headerData(c, HORIZONTAL) for c in range(model.columnCount())] \
        == ['101Z+', '104Z+', '110Z+']
    assert [model.headerData(r, VERTICAL) for r in range(model.rowCount())] \
        == ['1', '2']
    assert model.data(model.index(1, 0)) == '0.5'
    assert not (model.flags(model.index(0, 0)) & Qt.ItemFlag.ItemIsEditable)
    assert 'Sine response transformation: 2 rows over 3 control channels' \
        in window.statusBar().currentMessage()
    view.setCurrentIndex(view.findData('reference'))
    pump()
    model = window.table.model()
    assert [model.headerData(c, HORIZONTAL) for c in range(model.columnCount())] \
        == ['101Z+', '110Z+']
    assert 'over 2 drive channels' in window.statusBar().currentMessage()
    view.setCurrentIndex(view.findData('channels'))
    pump()
    assert window.table.model().columnCount() > 3, 'the channels again'


def test_a_table_without_loops_shows_neither_picker(window, pump):
    window.add_object('Channels', _table())
    _show(window, pump, 'Channels')
    assert not window.environment_action.isVisible()
    assert not window.table_view_action.isVisible()
    one = _table()
    one.set_environment('Random', RANDOM)
    window.add_object('One', one)
    _show(window, pump, 'One')
    assert not window.environment_action.isVisible(), 'no choice to make'
    assert not window.table_view_action.isVisible(), 'nothing but channels'
    model = window.table.model()
    control = next(c for c in range(model.columnCount())
                   if model.headerData(c, HORIZONTAL) == 'Control')
    assert _checked(model, control) == [False, True, True, False, False]


def test_a_transformed_spectrum_says_whose_rows_it_is_over(window, pump, tmp_path):
    out = visualdynamics.import_file(_write_mixed(tmp_path / 'mixed.nc4'))
    frf = out['Random_frf']
    virtual = type(frf)(abscissa=frf.abscissa, ordinate=frf.ordinate[:2],
                        response_dof=['1', '2'],
                        reference_dof=list(frf.reference_dof[:2]))
    window.add_object('Channels', out['channel_table'])
    window.add_object('Sine_frf', virtual)
    window.project.link('Channels', 'Sine_frf')
    _show(window, pump, 'Sine_frf')
    assert ('Sine_frf: rows 1, 2 are the Sine response transformation '
            '(Channels)') in window.statusBar().currentMessage()
