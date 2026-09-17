"""Coordinates are the user's to correct (Brandon, 2026-09-06).

The rows of every data object are coordinates, and the columns of a
matrix are coordinates too. A channel assigned to the wrong point at the
instrument is corrected here: double-click the label, type the
coordinate, and the object — and everything derived from it — carries the
correction. The channel moves, not the point: a force labeled at the
wrong node goes without the accelerometer at that node, which is changed
explicitly if it should be. `Project.rename_dof` is the verb; the grid
is its front.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.channel_table import ChannelTable
from visualdynamics.core.data import DataArray, Frf, Psd, TimeHistory


def _history():
    t = np.arange(0.0, 4.0, 0.01)
    return TimeHistory(t, np.random.default_rng(0).normal(size=(3, len(t))),
                       response_dof=['101Z+', '102Z+', '103Z+'],
                       ordinate_dim=['acceleration'] * 3)


# ---- the object -------------------------------------------------------------


def _frf():
    f = np.arange(0.0, 10.0)
    return Frf(f, np.ones((4, 10), dtype=complex),
               response_dof=['101Z+', '102Z+', '101Z+', '102Z+'],
               reference_dof=['101Z+', '101Z+', '105X+', '105X+'],
               ordinate_dim=['acceleration/force'] * 4)


def test_the_channel_moves_not_the_point():
    """Brandon, 2026-09-06: a force labeled at the wrong node moves
    without taking the accelerometer at that node with it. The drive
    point's accelerometer (a response) and its load cell (a reference)
    share 101Z+ and are two channels."""
    frf = _frf()
    assert frf.rename_dof('101Z+', '201Z+', 'acceleration') == 2, \
        'the two response records of the accelerometer'
    assert frf.response_dof == ['201Z+', '102Z+', '201Z+', '102Z+']
    assert frf.reference_dof == ['101Z+', '101Z+', '105X+', '105X+'], \
        'the load cell at 101Z+ is another channel and stays'
    assert frf.rename_dof('101Z+', '301Z+', 'force') == 2, \
        'the load cell, moved on its own'
    assert frf.reference_dof == ['301Z+', '301Z+', '105X+', '105X+']
    assert frf.response_dof == ['201Z+', '102Z+', '201Z+', '102Z+']
    with pytest.raises(ValueError, match="no force record at '102Z\\+'"):
        frf.rename_dof('102Z+', '1Z+', 'force')
    with pytest.raises(ValueError, match="no record at '999Z\\+'"):
        frf.rename_dof('999Z+', '1Z+')
    with pytest.raises(ValueError):
        frf.rename_dof('102Z+', 'not a dof')
    assert frf.rename_dof('102Z+', '102Z+', 'acceleration') == 0, \
        'the same name changes nothing'


def test_the_point_moves_when_no_quantity_is_named():
    """The explicit way to move every channel at a coordinate: the API
    with no quantity, which the grid never sends."""
    frf = _frf()
    assert frf.rename_dof('101Z+', '201Z+') == 4, 'two responses, two references'
    assert frf.response_dof == ['201Z+', '102Z+', '201Z+', '102Z+']
    assert frf.reference_dof == ['201Z+', '201Z+', '105X+', '105X+']
    assert frf.rename_dof('105X+', '106X') == 2, \
        'an unsigned direction typed is the positive one'
    assert frf.reference_dof[2:] == ['106X+', '106X+']


def test_one_sensor_on_both_sides_of_a_cross_term_moves_together():
    f = np.arange(0.0, 10.0)
    cpsd = Psd(f, np.ones((4, 10), dtype=complex),
               response_dof=['101Z+', '101Z+', '102Z+', '102Z+'],
               reference_dof=['101Z+', '102Z+', '101Z+', '102Z+'],
               ordinate_dim=['acceleration**2/frequency'] * 4)
    assert cpsd.rename_dof('102Z+', '302Z+', 'acceleration') == 4
    assert cpsd.response_dof == ['101Z+', '101Z+', '302Z+', '302Z+']
    assert cpsd.reference_dof == ['101Z+', '302Z+', '101Z+', '302Z+'], \
        'the accelerometer is one channel whichever side of the record'


def test_two_channels_cannot_be_given_one_identity():
    """A channel is a coordinate and a quantity: a drive point's load
    cell and accelerometer share a DOF and stay two rows; two
    accelerometers at one DOF would be one row hiding a record."""
    t = np.arange(0.0, 1.0, 0.1)
    data = TimeHistory(t, np.ones((3, 10)),
                       response_dof=['101Z+', '102Z+', '101Z+'],
                       ordinate_dim=['acceleration', 'acceleration', 'force'])
    with pytest.raises(ValueError, match='already has a acceleration record'):
        data.rename_dof('102Z+', '101Z+', 'acceleration')
    assert data.response_dof == ['101Z+', '102Z+', '101Z+'], 'refused, unchanged'
    # the force may go where an accelerometer already is: two channels
    assert data.rename_dof('101Z+', '102Z+', 'force') == 1
    assert data.response_dof == ['101Z+', '102Z+', '102Z+']
    # moving the whole point is refused where any channel of it collides
    with pytest.raises(ValueError, match='already has a'):
        data.rename_dof('102Z+', '101Z+')


def test_a_channel_table_row_renames_its_node_and_direction():
    table = ChannelTable({'channel': [1, 2, 3], 'node': [101, 102, 101],
                         'direction': ['Z+', 'Z+', 'X+'],
                         'unit': ['g', 'g', 'lbf']})
    assert table.rename_dof('101Z+', '201Y', 'acceleration') == 1
    assert table.dof_strings() == ['201Y+', '102Z+', '101X+']
    assert int(table.frame['node'][0]) == 201 and table.frame['direction'][0] == 'Y+'
    with pytest.raises(ValueError, match='node'):
        table.rename_dof('102Z+', 'Z+')
    with pytest.raises(ValueError, match="no channel at '9Z\\+'"):
        table.rename_dof('9Z+', '10Z+')
    with pytest.raises(ValueError, match="no force channel at '102Z\\+'"):
        table.rename_dof('102Z+', '10Z+', 'force')
    assert table.rename_dof('102Z+', '102Z+') == 0
    # two channels at one point: the one named by its quantity moves
    both = ChannelTable({'channel': [1, 2], 'node': [101, 101],
                         'direction': ['Z+', 'Z+'], 'unit': ['g', 'lbf']})
    assert both.rename_dof('101Z+', '201Z+', 'force') == 1
    assert both.dof_strings() == ['101Z+', '201Z+']
    assert both.rename_dof('101Z+', '301Z+') == 1, 'and every channel when unnamed'


# ---- the verb ---------------------------------------------------------------


def test_the_verb_carries_the_correction_down_the_derivation_chain():
    project = visualdynamics.Project()
    project.add('Run', _history())
    psds = project.compute_psds('Run')
    octave = project.compute_octave(psds)
    project.add('Other', _history())          # not derived: left alone
    changed = project.rename_dof('Run', '102Z+', '202Z+', 'acceleration')
    assert changed == ['Run', psds, octave]
    for name in changed:
        assert '202Z+' in project[name].response_dof, name
        assert '102Z+' not in project[name].response_dof, name
    assert '102Z+' in project['Other'].response_dof
    assert project.journal[-1] == \
        "project.rename_dof('Run', '102Z+', '202Z+', 'acceleration')", \
        'one line for the whole chain'
    # a derived object that never carried the coordinate is skipped, not
    # a refusal: a modal transformation has none of the source's
    geometry = visualdynamics.Geometry(np.array([101, 102, 103]),
                                       np.eye(3), length_unit='m')
    project.add('Plate', geometry)
    with pytest.raises(TypeError, match='no coordinates to rename'):
        project.rename_dof('Plate', '101Z+', '1Z+')


def test_the_verb_takes_the_object_as_well_as_the_name():
    project = visualdynamics.Project()
    project.add('Run', _history())
    assert project.rename_dof(project['Run'], '103Z+', '303Z+') == ['Run']
    assert project['Run'].response_dof[2] == '303Z+'


# ---- the grid ---------------------------------------------------------------


def _open_editor(header, section):
    from PySide6.QtWidgets import QLineEdit

    header.sectionDoubleClicked.emit(section)
    return header.findChild(QLineEdit)


def test_a_row_coordinate_is_typed_over_in_the_grid(window, pump):
    window.add_object('Run', _history())
    pump()
    grid = window.record_grids['Run']
    assert grid.editable_rows and not grid.editable_columns, \
        'rows are coordinates; the one unlabeled column is not'
    editor = _open_editor(grid.verticalHeader(), 1)
    assert editor is not None and editor.text() == '102Z+'
    editor.setText('202Z+')
    editor.editingFinished.emit()
    pump()
    pump()          # the commit is queued: it rebuilds this very grid
    assert window.objects['Run'].response_dof == ['101Z+', '202Z+', '103Z+']
    assert window.record_grids['Run'].responses == ['101Z+', '202Z+', '103Z+']
    assert window.project.journal[-1] == \
        "project.rename_dof('Run', '102Z+', '202Z+', 'acceleration')", \
        "the row's channel, named by its quantity"
    assert '102Z+ (acceleration) is now 202Z+ on Run' in \
        window.statusBar().currentMessage()


def test_a_reference_column_is_typed_over_too(window, pump):
    f = np.arange(0.0, 10.0)
    cpsd = Psd(f, np.ones((4, 10), dtype=complex),
               response_dof=['101Z+', '101Z+', '102Z+', '102Z+'],
               reference_dof=['101Z+', '102Z+', '101Z+', '102Z+'],
               ordinate_dim=['acceleration**2/frequency'] * 4)
    window.add_object('CPSD', cpsd)
    pump()
    grid = window.record_grids['CPSD']
    assert grid.editable_columns and grid.references == ['101Z+', '102Z+']
    editor = _open_editor(grid.horizontalHeader(), 1)
    assert editor is not None and editor.text() == '102Z+'
    editor.setText('302Z+')
    editor.editingFinished.emit()
    pump()
    pump()
    renamed = window.objects['CPSD']
    assert renamed.reference_dof == ['101Z+', '302Z+', '101Z+', '302Z+']
    assert renamed.response_dof == ['101Z+', '101Z+', '302Z+', '302Z+'], \
        'the coordinate moved on both sides: it is one point'
    grid = window.record_grids['CPSD']
    assert grid.references == ['101Z+', '302Z+'] and grid.responses == ['101Z+', '302Z+']


def test_a_drive_points_row_and_column_are_two_channels_in_the_grid(window, pump):
    """The FRF's row 101Z+ is the accelerometer and its column 101Z+ the
    load cell: typing over the row leaves the column, and the column's
    editor carries the force with it."""
    window.add_object('FRF', _frf())
    pump()
    grid = window.record_grids['FRF']
    assert grid.responses == ['101Z+', '102Z+']
    assert grid.references == ['101Z+', '105X+']
    assert grid.column_keys == [('101Z+', 'force'), ('105X+', 'force')]
    editor = _open_editor(grid.verticalHeader(), 0)
    editor.setText('201Z+')
    editor.editingFinished.emit()
    pump()
    pump()
    frf = window.objects['FRF']
    assert frf.response_dof == ['201Z+', '102Z+', '201Z+', '102Z+']
    assert frf.reference_dof == ['101Z+', '101Z+', '105X+', '105X+'], \
        'the load cell stays where it was labeled'
    grid = window.record_grids['FRF']
    editor = _open_editor(grid.horizontalHeader(), 0)
    assert editor.text() == '101Z+'
    editor.setText('301Z+')
    editor.editingFinished.emit()
    pump()
    pump()
    assert window.objects['FRF'].reference_dof == ['301Z+', '301Z+', '105X+', '105X+']
    assert window.project.journal[-1] == \
        "project.rename_dof('FRF', '101Z+', '301Z+', 'force')"


def test_a_specifications_collapsed_reference_column_is_not_an_editor(window, pump):
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    grid = window.record_grids['Specification']
    assert grid.editable_rows and not grid.editable_columns, \
        'the diagonal is spelled out on no column'


def test_a_channel_table_row_is_typed_over_in_the_grid(window, pump):
    window.add_object('Channels', ChannelTable(
        {'channel': [1, 2], 'node': [101, 102], 'direction': ['Z+', 'Z+'],
         'unit': ['g', 'g']}))
    pump()
    grid = window.record_grids['Channels']
    assert grid.kind == 'channel' and grid.editable_rows
    editor = _open_editor(grid.verticalHeader(), 0)
    editor.setText('501X+')
    editor.editingFinished.emit()
    pump()
    pump()
    assert window.objects['Channels'].dof_strings() == ['501X+', '102Z+']


def test_a_refused_rename_is_said_and_changes_nothing(window, pump):
    window.add_object('Run', _history())
    pump()
    grid = window.record_grids['Run']
    editor = _open_editor(grid.verticalHeader(), 0)
    editor.setText('102Z+')            # another accelerometer is there
    editor.editingFinished.emit()
    pump()
    pump()
    assert window.objects['Run'].response_dof == ['101Z+', '102Z+', '103Z+']
    assert 'already has a acceleration record' in window.statusBar().currentMessage()
    editor = _open_editor(window.record_grids['Run'].verticalHeader(), 0)
    editor.setText('never mind')
    editor.abandoned.emit()
    editor.editingFinished.emit()
    pump()
    pump()
    assert window.objects['Run'].response_dof == ['101Z+', '102Z+', '103Z+']


def test_the_ambiguous_row_edits_its_coordinate_not_its_marker(window, pump):
    """A row told apart only by its position reads '101Z+ #2'; the
    editor opens on the coordinate, since the marker is the grid's.

    The grid is the window's child and the editor's deferred delete is
    flushed here, on purpose: as first written this built a parentless
    grid, opened an editor, and let Python collect the grid at the
    test's end with the editor's `deleteLater` still posted — which
    ran in the next real event loop on the worker, a WebEngine test's
    `app.exec()`, and aborted it (2026-09-06, twice in four gates).
    """
    from PySide6.QtCore import QCoreApplication, QEvent

    from visualdynamics.gui.record_grid import RecordGrid

    data = DataArray(abscissa=np.arange(4.0), ordinate=np.ones((2, 4)),
                     response_dof=['101Z+', '101Z+'])
    grid = RecordGrid(data, window)
    assert grid.responses == ['101Z+ #1', '101Z+ #2']
    editor = grid.edit_row_label(1)
    assert editor.text() == '101Z+'
    editor.abandoned.emit()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    pump()
    grid.setParent(None)
    grid.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    pump()


# ---- the channel table's coordinates are its nodes and directions ----------


def test_a_runs_channel_table_and_data_are_linked_and_independent():
    """Brandon, 2026-09-06: imported together from one file they are
    linked; otherwise two independent objects — a coordinate corrected
    on the time history is not corrected on the table, or the reverse."""
    project = visualdynamics.Project()
    added = project.import_file(fixture_path('plate', 'random.nc4'))
    assert set(added) == {'Channel Table', 'Time History', 'Specification'}
    assert set(project.group_of('Channel Table')) == set(added), 'one file, one group'
    project.rename_dof('Time History', '101Z+', '201Z+', 'acceleration')
    assert project['Time History'].response_dof[0] == '201Z+'
    assert project['Channel Table'].dof_strings()[0] == '101Z+', \
        'the table is its own object'
    assert project['Specification'].response_dof[0] == '101Z+'
    project.rename_dof('Channel Table', '104Z+', '204Z+', 'acceleration')
    assert project['Channel Table'].dof_strings()[1] == '204Z+'
    assert project['Time History'].response_dof[1] == '104Z+'
    # a file holding one object links nothing
    alone = visualdynamics.Project()
    alone.import_file(fixture_path('plate', 'shapes.npy'))
    assert alone.links == []


def test_a_node_typed_in_the_table_view_moves_the_rows_coordinate(window, pump):
    """A channel table row's coordinate is derived from its node and
    direction and nothing else: typed in the table view, the grid's
    label follows at once; typed over in the grid, the cells follow
    (the test above) — one fact, edited from either side."""
    from PySide6.QtCore import Qt

    window.add_object('Channels', ChannelTable(
        {'channel': [1, 2], 'node': [101, 102], 'direction': ['Z+', 'Z+'],
         'unit': ['g', 'g']}))
    pump()
    window.show_object('Channels')
    pump()
    model = window.table.model()
    headers = [model.headerData(c, Qt.Orientation.Horizontal,
                                Qt.ItemDataRole.DisplayRole)
               for c in range(model.columnCount())]
    node = next(c for c, h in enumerate(headers) if str(h).lower().startswith('node'))
    direction = next(c for c, h in enumerate(headers)
                     if str(h).lower().startswith('direction'))
    assert model.setData(model.index(0, node), '501', Qt.ItemDataRole.EditRole)
    pump()
    assert window.objects['Channels'].dof_strings() == ['501Z+', '102Z+']
    assert window.record_grids['Channels'].responses == ['501Z+', '102Z+'], \
        'the grid follows the table'
    assert model.setData(model.index(1, direction), 'X+', Qt.ItemDataRole.EditRole)
    pump()
    assert window.record_grids['Channels'].responses == ['501Z+', '102X+']
    assert window.project.journal[-1] == "project['Channels'].set_cell('direction', 1, 'X+')"
