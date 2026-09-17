"""Deleting the thing that was selected, not the thing that contains it.

Sub-items — records, modes, channels — used to be undeletable: picking one
and pressing Delete walked up to the owner and deleted the whole object. Now
the selection is what goes: records from the tree or a grid, channels and
modes as whole rows of their tables, whole objects only when the object
itself is picked.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.data import TimeHistory

# ---- the model ---------------------------------------------------------------

def test_delete_records_takes_everything_a_record_owns():
    data = TimeHistory(abscissa=np.arange(4.0),
                       ordinate=np.arange(12.0).reshape(3, 4),
                       response_dof=['1Z+', '2Z+', '3Z+'],
                       block=['a', 'b', 'c'],
                       ordinate_dim='acceleration', ordinate_unit='m/s**2',
                       comment=['one', 'two', 'three'])
    data.delete_records([1])
    assert data.response_dof == ['1Z+', '3Z+']
    assert data.block == ['a', 'c']
    assert data.comment == ['one', 'three']
    assert np.allclose(data.ordinate[:, 0].real, [0.0, 8.0])


def test_a_specification_keeps_its_limits_aligned():
    """The limit curves ride per record; deleting a record without its rows
    would bound the wrong channels."""
    spec = visualdynamics.import_file(fixture_path('plate',
                                         'random_spectra.nc4'))['Random_specification']
    # the fixture's limit curves are identical across channels, so value
    # comparison alone proves nothing — tag each row before deleting
    for name in spec.limits:
        spec.limits[name] = spec.limits[name] * np.arange(
            1.0, spec.num_records + 1)[:, None]
    kept_dof = spec.response_dof[2]
    kept_limit = spec.limits['abort_upper'][2].copy()
    spec.delete_records([0, 1])
    assert spec.response_dof[0] == kept_dof
    for name, values in spec.limits.items():
        assert values.shape[0] == spec.num_records, name
    assert np.allclose(spec.limits['abort_upper'][0], kept_limit,
                       equal_nan=True)


@pytest.mark.parametrize('path,attribute,method', [
    ('shapes.npy', 'num_shapes', 'delete_modes'),
    ('modal_spectra.nc4', 'num_channels', 'delete_channels'),
])
def test_the_last_sub_item_is_refused(path, attribute, method):
    """An empty object is not a state anything else can show; delete the
    object instead."""
    out = visualdynamics.import_file(fixture_path('plate', path))
    obj = out['channel_table'] if isinstance(out, dict) else out
    with pytest.raises(ValueError, match='delete the'):
        getattr(obj, method)(range(getattr(obj, attribute)))


# ---- the tree and the grid ---------------------------------------------------

def loaded(window):
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4')])
    return window


def test_grid_records_delete_records_not_the_object(window, pump):
    loaded(window)
    data = window.objects['Time History']
    before = data.num_records
    grid = window.record_grids['Time History']
    grid.item(0, 0).setSelected(True)
    grid.item(0, 1).setSelected(True)
    pump()
    window.delete_selected()
    pump()
    assert 'Time History' in window.objects, 'the object survives'
    assert data.num_records == before - 2
    assert 'Removed 2 records from Time History' in \
        window.statusBar().currentMessage()


def test_an_object_pick_still_deletes_the_object(window, pump):
    loaded(window)
    window.tree.clearSelection()
    window._item_for_object('FRF').setSelected(True)
    pump()
    window.delete_selected()
    assert 'FRF' not in window.objects


def test_deleting_every_record_is_refused_with_a_reason(window, pump):
    loaded(window)
    grid = window.record_grids['FRF']
    grid.selectAll()
    pump()
    before = window.objects['FRF'].num_records
    window.delete_selected()
    assert window.objects['FRF'].num_records == before
    assert 'delete the object instead' in window.statusBar().currentMessage()


# ---- the tables --------------------------------------------------------------

def test_whole_rows_of_the_channel_table_delete_channels(window, pump):
    loaded(window)
    window.tree.clearSelection()
    window._item_for_object('Channel Table').setSelected(True)
    pump()
    table = window.objects['Channel Table']
    before = table.num_channels
    doomed = table.dof_strings()[1]
    window.table.selectRow(1)
    assert window.table.whole_rows() == [1]
    # through the key, not the signal: the gate that decides between
    # clearing cells and removing rows is exactly what is under test.
    # Focus first — clicking a row focuses the table for a real user, and
    # without it the tree's Delete shortcut wins and deletes the object.
    window.table.setFocus()
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    QTest.keyClick(window.table, Qt.Key.Key_Delete)
    pump()
    assert 'Channel Table' in window.objects, 'the object must survive'
    assert table.num_channels == before - 1
    assert doomed not in table.dof_strings()


def test_cells_short_of_a_row_still_just_clear(window, pump):
    """The table convention survives: Delete on a cell selection clears; only
    a whole-row selection removes."""
    loaded(window)
    window.tree.clearSelection()
    window._item_for_object('Channel Table').setSelected(True)
    pump()
    table = window.objects['Channel Table']
    before = table.num_channels
    window.table.clearSelection()
    index = window.table.model().index(0, 0)
    window.table.selectionModel().select(
        index, window.table.selectionModel().SelectionFlag.Select)
    assert window.table.whole_rows() == []
    assert table.num_channels == before


def test_the_units_pane_rows_are_not_deletable(window, pump):
    """Its rows are records, but the pane exists to declare units; deletion
    lives in the tree, the grid and the object tables."""
    loaded(window)
    window.tree.clearSelection()
    item = window._item_for_object('Time History')
    window.tree.setCurrentItem(item)
    pump()
    window.define_units()
    pump()
    assert not window.units_table.rows_deletable


# ---- sub-items never take co-selected objects with them ----------------------

def test_a_record_pick_never_deletes_a_co_selected_object(window, pump):
    """A geometry is selected so records can animate on it — context, not a
    target. One keystroke deleted one record *and* the whole geometry once,
    which on screen also tore down the live 3D scene mid-delete."""
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4'),
                         fixture_path('plate', 'geometry.npz')])
    window._item_for_object('Geometry').setSelected(True)
    pump()
    grid = window.record_grids['Time History']
    grid.item(0, 0).setSelected(True)
    pump()
    window.delete_selected()
    pump()
    assert 'Geometry' in window.objects, 'context must survive'
    assert 'Time History' in window.objects
    message = window.statusBar().currentMessage()
    assert 'Geometry' not in message
    assert '1 record from Time History' in message


def test_a_delete_all_renders_once_not_once_per_object(window, pump,
                                                       monkeypatch):
    """Deleting N selected objects used to promote a neighbor to
    current on every row removal, and the handler re-rendered each
    survivor's view only for the next removal to kill it — the
    specification plot blinked once per object and a delete-all read
    as hung (Brandon, 2026-08-31). The batch makes one selection
    change and settles the tree once."""
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4'),
                         fixture_path('plate', 'geometry.npz'),
                         fixture_path('plate', 'shapes.npy')])
    pump()
    names = list(window.objects)
    assert len(names) >= 3
    window.tree.clearSelection()
    for name in names:
        window._item_for_object(name).setSelected(True)
    pump()

    fired = []
    handler = window._selection_changed
    monkeypatch.setattr(window, '_selection_changed',
                        lambda *a: (fired.append(1), handler(*a)))
    settles = []
    refresher = window._refresh_placeholders
    monkeypatch.setattr(window, '_refresh_placeholders',
                        lambda: (settles.append(1), refresher()))
    window.delete_selected()
    assert not window.objects, 'everything selected went'
    assert len(fired) <= 1, (
        f'{len(fired)} selection rounds for one keystroke — the blink')
    assert len(settles) == 1, 'the placeholders settle once, after the loop'


def test_object_picks_alone_still_delete_objects(window, pump):
    """The narrowing only happens when sub-items are in the keystroke."""
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4'),
                         fixture_path('plate', 'geometry.npz')])
    window.tree.clearSelection()
    window._item_for_object('Geometry').setSelected(True)
    pump()
    window.delete_selected()
    assert 'Geometry' not in window.objects
