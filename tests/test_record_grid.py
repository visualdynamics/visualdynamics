"""Matrix data expands into a grid, and the grid is what gets plotted.

Two things are worth guarding. First that only real matrices get a grid — a
partial set would come out as a grid of holes, and the rule that decides is
easy to loosen by accident. Second that the grid's selection is the record
selection: it is the only copy of that state, so if the plot ever stopped
reading it the two would silently disagree.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.data import TimeHistory
from visualdynamics.gui.record_grid import grid_axes


def dofs_of(rows):
    """grid_axes returns RowKeys; often only the DOF matters."""
    return [key.dof for key in rows]


@pytest.fixture(scope='module')
def spectra():
    return (visualdynamics.import_file(fixture_path('plate', 'modal_spectra.nc4'))
            | visualdynamics.import_file(fixture_path('plate',
                                            'random_spectra.nc4')))


# --- what deserves a grid -------------------------------------------------

def test_an_frf_is_a_matrix(spectra):
    responses, references = grid_axes(spectra['Modal_frf'])
    assert (len(responses), len(references)) == (11, 2)
    assert len(responses) * len(references) == spectra['Modal_frf'].num_records


def test_a_cpsd_is_a_square_matrix(spectra):
    responses, references = grid_axes(spectra['Random_response_cpsd'])
    assert dofs_of(responses) == references
    assert len(responses) == 8


def test_coherence_grids_one_column_wide(spectra):
    """Nothing but the row tells its records apart, so the grid is a single
    unlabeled column — the same format as everything else, one habit."""
    rows, columns = grid_axes(spectra['Modal_coherence'])
    assert columns == ['']
    assert len(rows) == spectra['Modal_coherence'].num_records


def test_a_plain_time_history_grids_one_column_wide():
    time = TimeHistory(abscissa=np.arange(4.0),
                       ordinate=np.zeros((3, 4)),
                       response_dof=['1X+', '2X+', '3X+'])
    rows, columns = grid_axes(time)
    assert columns == ['']
    assert [key.dof for key in rows] == ['1X+', '2X+', '3X+']


def test_a_specification_collapses_its_redundant_reference_column():
    """Every record is a channel against itself, so a reference column would
    only repeat the rows — six diagonal cells across six columns say nothing
    the rows do not. One column instead."""
    spec = visualdynamics.import_file(
        fixture_path('plate', 'random.nc4'))['Random_specification']
    assert spec.reference_dof, 'the premise: it does carry references'
    rows, columns = grid_axes(spec)
    assert columns == ['']
    assert len(rows) == spec.num_records


# --- the grid itself ------------------------------------------------------

def loaded(window, path='modal_spectra.nc4'):
    window.import_paths([fixture_path('plate', path)])
    return window


def test_expanding_an_frf_builds_a_grid_not_child_rows(window):
    loaded(window)
    name = 'FRF'
    item = window._item_for_object(name)
    item.setExpanded(True)
    grid = window.record_grids[name]
    assert grid.rowCount() == 11 and grid.columnCount() == 2
    # one spanned holder row, not one row per record
    assert item.childCount() == 1
    assert window.tree.itemWidget(item.child(0), 0) is grid


def test_the_headers_say_which_dof_is_which(window):
    loaded(window)
    data = window.objects['FRF']
    grid = window.record_grids['FRF']
    assert grid.verticalHeaderItem(0).text() == data.response_dof[0]
    assert grid.row_keys[0].dof == data.response_dof[0]
    # the row is the response *channel*, so its quantity is the
    # response factor of 'acceleration/force' — keyed on the compound,
    # a CPSD split every accelerometer into a row per thing it was
    # measured against
    assert grid.row_keys[0].quantity == 'acceleration'
    assert grid.horizontalHeaderItem(0).text() == data.reference_dof[0]


def test_a_cell_maps_to_the_record_with_those_two_dofs(window):
    loaded(window)
    data = window.objects['FRF']
    grid = window.record_grids['FRF']
    for row, column in ((0, 0), (3, 1), (10, 1)):
        grid.clearSelection()
        grid.item(row, column).setSelected(True)
        record, = grid.selected_records()
        assert data.response_dof[record] == grid.responses[row]
        assert data.reference_dof[record] == grid.references[column]


def test_every_cell_carries_the_icon_for_what_it_measures(window):
    """An FRF record is an acceleration per force, and its cell says so:
    both quantities as a fraction, response over reference."""
    from visualdynamics.gui.icons import ratio_icon
    loaded(window)
    grid = window.record_grids['FRF']
    wanted = ratio_icon('acceleration', 'force').pixmap(16, 16).toImage()
    for row in range(grid.rowCount()):
        for column in range(grid.columnCount()):
            icon = grid.item(row, column).icon()
            assert not icon.isNull()
            assert icon.pixmap(16, 16).toImage() == wanted


# --- the grid drives the plot --------------------------------------------

def plotted(window):
    """The records the window would actually draw right now."""
    return [(name, detail) for kind, name, _obj, detail
            in window.selected_references() if kind == 'record']


def test_picking_cells_plots_exactly_those_records(window, pump):
    loaded(window)
    grid = window.record_grids['FRF']
    grid.item(3, 1).setSelected(True)
    grid.item(5, 0).setSelected(True)
    pump()
    assert plotted(window) == [('FRF', 7),
                               ('FRF', 10)]


def test_picking_in_the_grid_points_the_tree_at_the_owner(window, pump):
    """Nothing is drawn unless the object is selected, and clicking inside an
    embedded widget does not select the tree row on its own."""
    loaded(window)
    window.tree.clearSelection()
    grid = window.record_grids['FRF']
    grid.item(2, 0).setSelected(True)
    pump()
    assert [item.text(0) for item in window.tree.selectedItems()] == \
        ['FRF']


def test_an_empty_grid_selection_means_the_whole_object(window, pump):
    loaded(window)
    name = 'FRF'
    grid = window.record_grids[name]
    grid.item(1, 1).setSelected(True)
    pump()
    grid.clearSelection()
    window._item_for_object(name).setSelected(True)
    pump()
    assert plotted(window) == []
    assert [(kind, other) for kind, other, _obj, _d
            in window.selected_references()] == [('object', name)]


def test_a_whole_cpsd_including_its_cross_terms_can_be_picked(window, pump):
    loaded(window, 'random_spectra.nc4')
    grid = window.record_grids['PSD']
    grid.selectAll()
    pump()
    assert len(plotted(window)) == 64


def test_renaming_the_object_leaves_the_grid_working(window, pump):
    loaded(window)
    item = window._item_for_object('FRF')
    item.setText(0, 'renamed')
    window._item_renamed(item, 0)
    pump()
    grid = window.record_grids['renamed']
    window.tree.clearSelection()
    grid.item(4, 1).setSelected(True)
    pump()
    assert plotted(window) == [('renamed', 9)]


# --- the import that makes the matrix a matrix ---------------------------

def test_a_cpsd_keeps_its_cross_terms(spectra):
    """Dropping to the diagonal threw away most of what a CPSD is for."""
    cpsd = spectra['Random_response_cpsd']
    assert cpsd.num_records == 64
    responses, references = grid_axes(cpsd)
    pairs = set(zip(cpsd.response_dof, cpsd.reference_dof))
    assert pairs == {(dof, c) for dof in dofs_of(responses)
                     for c in references}


def test_a_cross_term_is_complex_and_its_transpose_is_the_conjugate(spectra):
    """A real check that these are cross spectra and not repeated ASDs."""
    cpsd = spectra['Random_response_cpsd']
    index = {pair: i for i, pair in
             enumerate(zip(cpsd.response_dof, cpsd.reference_dof))}
    responses, _ = grid_axes(cpsd)
    a, b = dofs_of(responses)[0], dofs_of(responses)[3]
    upper = cpsd.ordinate[index[(a, b)]]
    lower = cpsd.ordinate[index[(b, a)]]
    assert np.any(upper.imag != 0)
    assert np.allclose(upper, lower.conj())
    assert np.allclose(cpsd.ordinate[index[(a, a)]].imag, 0)


# --- two channels at one point ----------------------------------------------

def two_at_one_point():
    """A drive and the accelerometer beside it: same node, same direction,
    different quantity. A modal survey that measures its own drive points
    has one such pair per shaker."""
    from visualdynamics.core.data import TimeHistory

    dofs, dims, blocks = [], [], []
    for dof, dim in (('1Z+', 'acceleration'), ('2Z+', 'acceleration'),
                     ('2Z+', 'force')):
        for average in range(3):
            dofs.append(dof)
            dims.append(dim)
            blocks.append(f'avg {average + 1}')
    return TimeHistory(abscissa=np.arange(4.0),
                       ordinate=np.arange(9 * 4.0).reshape(9, 4),
                       response_dof=dofs, block=blocks, ordinate_dim=dims,
                       ordinate_unit=['m/s**2' if d == 'acceleration' else 'N'
                                      for d in dims])


def test_a_shared_dof_still_makes_a_grid():
    """Keyed by DOF alone these collapse to 2 rows, 2 x 3 != 9 records, and
    the whole object falls back to a list."""
    from visualdynamics.gui.record_grid import row_keys

    data = two_at_one_point()
    assert len(set(data.response_dof)) == 2, 'the premise: DOFs repeat'
    responses, columns = grid_axes(data)
    assert (len(responses), len(columns)) == (3, 3)
    assert len(set(row_keys(data))) == 3


def test_the_label_is_the_dof_and_the_quantity_is_the_icon():
    """Two rows can now read '2Z+'. What tells them apart is the icon in
    every one of their cells, which is a shape rather than a word."""
    responses, _columns = grid_axes(two_at_one_point())
    assert dofs_of(responses) == ['1Z+', '2Z+', '2Z+']
    assert [(k.dof, k.quantity) for k in responses] == [
        ('1Z+', 'acceleration'), ('2Z+', 'acceleration'), ('2Z+', 'force')]
    assert all(k.occurrence == 0 for k in responses), 'quantity was enough'


def test_the_cells_still_point_at_the_right_records(qt_app):
    from visualdynamics.gui.record_grid import RecordGrid

    data = two_at_one_point()
    grid = RecordGrid(data)
    for row, (dof, quantity, _occurrence) in enumerate(grid.row_keys):
        for column, block in enumerate(grid.references):
            grid.clearSelection()
            grid.item(row, column).setSelected(True)
            record, = grid.selected_records()
            assert data.block[record] == block
            assert data.response_dof[record] == dof
            assert data.ordinate_dim[record] == quantity


def three_types_at_one_dof():
    """A volt, a newton and a meter per second squared, all at 5Z+."""
    from visualdynamics.core.data import TimeHistory

    dofs, dims, units, blocks = [], [], [], []
    for dim, unit in (('acceleration', 'm/s**2'), ('force', 'N'),
                      ('voltage', 'V')):
        for average in range(2):
            dofs.append('5Z+')
            dims.append(dim)
            units.append(unit)
            blocks.append(f'avg {average + 1}')
    return TimeHistory(abscissa=np.arange(4.0),
                       ordinate=np.arange(6 * 4.0).reshape(6, 4),
                       response_dof=dofs, block=blocks,
                       ordinate_dim=dims, ordinate_unit=units)


def test_a_volt_a_newton_and_an_acceleration_get_their_own_rows():
    """One DOF, three quantities, three rows. Nothing about the rule is
    specific to forces and accelerations."""
    responses, columns = grid_axes(three_types_at_one_dof())
    assert columns == ['avg 1', 'avg 2']
    assert [(k.dof, k.quantity) for k in responses] == [
        ('5Z+', 'acceleration'), ('5Z+', 'force'), ('5Z+', 'voltage')]
    assert dofs_of(responses) == ['5Z+'] * 3, 'the label repeats; the icon does not'


def test_the_row_identity_is_the_dof_and_the_type():
    """Stated once, in the keys, rather than implied by when a label happens
    to get qualified."""
    from visualdynamics.gui.record_grid import row_keys

    data = three_types_at_one_dof()
    keys = row_keys(data)
    assert [(k.dof, k.quantity) for k in keys[:2]] == [
        ('5Z+', 'acceleration'), ('5Z+', 'acceleration')]
    assert len(set(keys)) == 3, 'three rows, two averages each'
    assert all(k.occurrence == 0 for k in keys), 'the quantity was enough'


def test_two_channels_at_one_dof_and_one_type_fall_back_to_channel_order():
    """Same DOF, same quantity — nothing in the measurement separates them, so
    the channel's position does. It is an artifact of the file rather than a
    property of the test, and using it is worse than using the quantity but
    better than refusing to lay the data out at all."""
    from visualdynamics.core.data import TimeHistory
    from visualdynamics.gui.record_grid import row_labels

    data = TimeHistory(abscissa=np.arange(4.0), ordinate=np.arange(16.0).reshape(4, 4),
                       response_dof=['7Z+'] * 4,
                       block=['avg 1', 'avg 2'] * 2,
                       ordinate_dim='acceleration', ordinate_unit='m/s**2')
    rows, columns = grid_axes(data)
    assert len(rows) == 2 and columns == ['avg 1', 'avg 2']
    assert [key.occurrence for key in rows] == [0, 1]
    assert row_labels(rows) == ['7Z+ #1', '7Z+ #2']


# --- column headers narrow enough to use -------------------------------------

def test_average_columns_lose_the_word_that_is_on_all_of_them():
    """'avg 7' is prose in a record label and noise in a column header. At
    45 px a column, a 20-average grid showed one column in the dock."""
    from visualdynamics.gui.record_grid import short_labels

    assert short_labels([f'avg {i + 1}' for i in range(20)])[:3] == ['1', '2', '3']


def test_reference_dofs_keep_their_full_names():
    """They share no prefix and every character of them means something."""
    from visualdynamics.gui.record_grid import short_labels

    dofs = ['210Z+', '250Z+', '101Z+', '241Y+']
    assert short_labels(dofs) == dofs


def test_a_mixed_set_is_left_alone():
    from visualdynamics.gui.record_grid import short_labels

    assert short_labels(['avg 1', '210Z+']) == ['avg 1', '210Z+']


def test_the_columns_themselves_are_untouched(window):
    """Only the header text is shortened; the keys the cells index by are
    the real block labels, and grid_axes still reports them."""
    loaded(window, 'modal_spectra.nc4')
    name = next(k for k in window.objects if k.endswith('Time History'))
    _rows, columns = grid_axes(window.objects[name])
    assert columns[0] == 'avg 1'
    grid = window.record_grids[name]
    assert grid.references[0] == 'avg 1'
    assert grid.horizontalHeaderItem(0).text() == '1'


def test_an_average_column_is_narrower_than_a_dof_column(window):
    """The whole point: 20 icons across should fit where 20 DOFs would not."""
    loaded(window, 'modal_spectra.nc4')
    frames = window.record_grids[
        next(k for k in window.objects if k.endswith('Time History'))]
    frf = window.record_grids[
        next(k for k in window.objects if k.endswith('FRF'))]
    assert frames.columnWidth(0) < frf.columnWidth(0)


# --- repeated DOFs with nothing to tell them apart ---------------------------

def repeated(**kwargs):
    """Four records, one DOF, two averages — so two channels at one point."""
    from visualdynamics.core.data import TimeHistory

    return TimeHistory(abscissa=np.arange(4.0), ordinate=np.arange(16.0).reshape(4, 4),
                       response_dof=['1Z+'] * 4,
                       block=['avg 1', 'avg 2'] * 2, **kwargs)


def test_with_no_units_the_channel_order_arranges_it_anyway(window, pump):
    """An import that says nothing about units still says how many channels
    sit at that DOF, and that is enough to lay the data out. Refusing the grid
    threw away an arrangement we knew — the units can be declared afterwards.
    """
    data = repeated()
    assert not data.units_defined
    rows, columns = grid_axes(data)
    assert len(rows) == 2 and columns == ['avg 1', 'avg 2']
    assert [key.occurrence for key in rows] == [0, 1], 'told apart by channel'
    name = window.add_object('unitless', data)
    grid = window.record_grids[name]
    assert grid.responses == ['1Z+ #1', '1Z+ #2'], 'the DOF alone would lie'
    grid.item(1, 0).setSelected(True)
    pump()
    assert grid.selected_records() == [2], 'the second channel, first average'


def test_the_channel_marker_goes_away_once_units_are_declared():
    """It is a last resort, and it stops being used the moment it is not
    needed — a channel index is an artifact of the file, not of the test."""
    from visualdynamics.gui.record_grid import row_labels

    vague = repeated()
    assert row_labels(grid_axes(vague)[0]) == ['1Z+ #1', '1Z+ #2']
    told = repeated(ordinate_dim=['acceleration'] * 2 + ['force'] * 2,
                    ordinate_unit=['m/s**2'] * 2 + ['N'] * 2)
    assert row_labels(grid_axes(told)[0]) == ['1Z+', '1Z+']
    assert all(key.occurrence == 0 for key in grid_axes(told)[0])


def test_a_hint_is_enough_to_separate_them():
    """A source can name a quantity without sizing it. That claim already
    picks the record's icon, so it has to key the row too — keying on the bare
    dimension left the two disagreeing: separate icons, one merged row."""
    data = repeated(dimension_hint=['acceleration', 'acceleration',
                                    'force', 'force'])
    assert not data.units_defined, 'still nothing to scale by'
    rows, columns = grid_axes(data)
    assert [(k.dof, k.quantity) for k in rows] == [
        ('1Z+', 'acceleration'), ('1Z+', 'force')]
    assert columns == ['avg 1', 'avg 2']


def test_declaring_the_units_separates_them_too():
    data = repeated(ordinate_dim=['acceleration'] * 2 + ['force'] * 2,
                    ordinate_unit=['m/s**2'] * 2 + ['N'] * 2)
    rows, _columns = grid_axes(data)
    assert [(k.dof, k.quantity) for k in rows] == [
        ('1Z+', 'acceleration'), ('1Z+', 'force')]


def test_the_row_key_and_the_icon_read_the_same_field(qt_app):
    """They disagreed once. (`qt_app`: painting an icon needs the
    application to exist — without it this test only passed when a
    neighbor in the same worker had already made one.) If one of them ever stops reading `known_dim` the
    grid and its cells describe different rows."""
    from visualdynamics.gui.icons import quantity_of, record_icon
    from visualdynamics.gui.record_grid import row_keys

    data = repeated(dimension_hint=['acceleration', 'acceleration',
                                    'force', 'force'])
    for i in range(data.num_records):
        quantity = row_keys(data)[i].quantity
        assert quantity_of(quantity) is not None
        assert record_icon(data, i).cacheKey() == \
            record_icon(data, i).cacheKey()
        assert quantity == data.known_dim(i)


def test_ambiguous_reference_headers_wear_quantity_icons(window):
    """A drive point's accelerometer and load cell share a DOF, so its
    two reference columns are told apart the way the rows' cells are:
    by the quantity's icon, not by '(force)' spelled out — the width
    lesson the row headers learned long ago."""
    from visualdynamics.core.data import Psd
    from visualdynamics.gui.record_grid import RecordGrid

    channels = [('1Z+', 'acceleration'), ('1Z+', 'force'),
                ('2Z+', 'acceleration')]
    records, rdofs, fdofs, dims = [], [], [], []
    for rd, rq in channels:
        for fd, fq in channels:
            records.append(np.full(4, 1.0 + 0.5j))
            rdofs.append(rd)
            fdofs.append(fd)
            dims.append(f'{rq}**2/frequency' if rq == fq
                        else f'{rq}*{fq}/frequency')
    cpsd = Psd(np.arange(1.0, 5.0), np.array(records),
               response_dof=rdofs, reference_dof=fdofs,
               ordinate_dim=dims)
    grid = RecordGrid(cpsd)
    assert [grid.horizontalHeaderItem(i).text() for i in range(3)] \
        == ['1Z+', '1Z+', '2Z+']
    icons = [grid.horizontalHeaderItem(i).icon() for i in range(3)]
    assert not icons[0].isNull() and not icons[1].isNull()
    assert icons[2].isNull(), 'an unambiguous DOF needs no mark'
    assert (icons[0].pixmap(16, 16).toImage()
            != icons[1].pixmap(16, 16).toImage()), (
        'the two marks are two different quantities')
