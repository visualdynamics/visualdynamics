"""The drive-point button: filter an FRF plot to the grid's diagonal.

A drive-point FRF has its response and reference at one DOF. The button does
not filter the plot behind the selection's back — it *is* a selection: the
diagonal cells of the FRF grid light up as if picked by hand, so the tree,
the grid and the plot keep telling one story, and the checked state is
derived from whether the selection equals the diagonal rather than stored.
"""

from __future__ import annotations

import numpy as np
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.data import Frf


def with_drive_points():
    """3 responses x 2 references, complete, with 2 drive points."""
    return Frf(abscissa=np.linspace(0, 128, 64),
               ordinate=np.ones((6, 64), dtype=complex),
               response_dof=['1Z+', '1Z+', '2Z+', '2Z+', '3Z+', '3Z+'],
               reference_dof=['1Z+', '2Z+'] * 3,
               ordinate_dim='acceleration/force',
               ordinate_unit='m/s**2', reference_unit='N')


def shown(window, obj, name='FRF'):
    # the filtering is asserted through the flat plot's curves; the
    # waterfall is the default reading now, so the flat plot is asked for
    window.data_pane.waterfall_action.setChecked(False)
    added = window.add_object(name, obj)
    item = window._item_for_object(added)
    item.setExpanded(True)
    window.tree.setCurrentItem(item)
    return added


def test_the_button_appears_for_an_frf_with_drive_points(window, pump):
    shown(window, with_drive_points())
    pump()
    assert window.data_pane.toolbar.isVisible()
    assert window.data_pane.drive_point_action.isVisible()
    assert not window.data_pane.drive_point_action.isChecked(), 'whole object showing'


def test_toggling_selects_the_diagonal_of_the_grid(window, pump):
    """The filter is the selection — that is what makes the tree reflect it."""
    name = shown(window, with_drive_points())
    pump()
    window.data_pane.drive_point_action.trigger()
    pump()
    grid = window.record_grids[name]
    data = window.objects[name]
    picked = [(data.response_dof[i], data.reference_dof[i])
              for i in grid.selected_records()]
    assert picked == [('1Z+', '1Z+'), ('2Z+', '2Z+')]
    assert window.data_pane.drive_point_action.isChecked()
    assert '2 curves' in window.statusBar().currentMessage()


def test_toggling_off_restores_the_whole_object(window, pump):
    name = shown(window, with_drive_points())
    pump()
    window.data_pane.drive_point_action.trigger()
    pump()
    window.data_pane.drive_point_action.trigger()
    pump()
    assert window.record_grids[name].selected_records() == []
    assert not window.data_pane.drive_point_action.isChecked()
    assert '6 records' in window.statusBar().currentMessage()


def test_picking_another_cell_releases_the_button(window, pump):
    """Checked is derived, never stored: the button is down exactly when the
    selection is the diagonal."""
    name = shown(window, with_drive_points())
    pump()
    window.data_pane.drive_point_action.trigger()
    pump()
    window.record_grids[name].item(0, 1).setSelected(True)
    pump()
    assert not window.data_pane.drive_point_action.isChecked()


def test_an_frf_without_drive_points_offers_no_button(window, pump):
    """A survey that measures none of its drive DOFs: a filter to
    nothing is not a legitimate option and the control is absent. The
    bar itself stays: CMIF applies to any FRF matrix."""
    frf = Frf(abscissa=np.linspace(0, 128, 64),
              ordinate=np.ones((4, 64), dtype=complex),
              response_dof=['1Z+', '2Z+'] * 2,
              reference_dof=['9Z+'] * 2 + ['8Z+'] * 2,
              ordinate_dim='acceleration/force',
              ordinate_unit='m/s**2', reference_unit='N')
    shown(window, frf, name='Plain')
    pump()
    assert not window.data_pane.drive_point_action.isVisible()
    assert window.data_pane.toolbar.isVisible()
    assert window.data_pane.cmif_action.isVisible()


def test_the_modal_survey_now_measures_its_drive_points(window, pump):
    """The committed fixture carries both drive-point FRFs, so the
    filter is a real option there."""
    out = visualdynamics.import_file(fixture_path('plate',
                                        'modal_spectra.nc4'))
    shown(window, out['Modal_frf'], name='Measured')
    pump()
    assert window.data_pane.drive_point_action.isVisible()


def test_coherence_keeps_its_own_controls(window, pump):
    """One bar, two vocabularies; each control shows only for its data."""
    out = visualdynamics.import_file(fixture_path('plate', 'modal_spectra.nc4'))
    shown(window, out['Modal_coherence'], name='Coh')
    pump()
    assert window.data_pane.toolbar.isVisible()
    assert window.data_pane.map_action.isVisible()
    assert not window.data_pane.drive_point_action.isVisible()


def test_a_cpsd_gets_the_same_button_named_autospectra(window, pump):
    """The diagonal of a CPSD grid is its ASDs — same filter, same symbol,
    named for what the diagonal is here."""
    out = visualdynamics.import_file(fixture_path('plate', 'random_spectra.nc4'))
    name = shown(window, out['Random_response_cpsd'], name='CPSD')
    pump()
    assert window.data_pane.drive_point_action.isVisible()
    assert window.data_pane.drive_point_action.text() == 'Autospectra'
    window.data_pane.drive_point_action.trigger()
    pump()
    grid = window.record_grids[name]
    data = window.objects[name]
    picked = grid.selected_records()
    assert picked and all(data.response_dof[i] == data.reference_dof[i]
                          for i in picked)
    assert len(picked) == len(set(data.response_dof))
    assert window.data_pane.drive_point_action.isChecked()


def test_a_diagonal_only_psd_offers_no_filter(window, pump):
    """Every record of a plain PSD is already an ASD; a filter that changes
    nothing is not a legitimate option."""
    psd = visualdynamics.import_file(fixture_path('plate', 'psd.npz'))
    shown(window, psd, name='PSD')
    pump()
    assert not window.data_pane.drive_point_action.isVisible()


def test_an_frf_keeps_its_own_name_on_the_button(window, pump):
    shown(window, with_drive_points())
    pump()
    assert window.data_pane.drive_point_action.text() == 'Drive points'


def test_the_button_works_from_a_subset_too(window, pump):
    """Computed over the whole object, not the current picks, so it filters
    from any starting selection."""
    name = shown(window, with_drive_points())
    pump()
    grid = window.record_grids[name]
    grid.item(0, 1).setSelected(True)      # an off-diagonal record
    pump()
    window.data_pane.drive_point_action.trigger()
    pump()
    data = window.objects[name]
    assert all(data.response_dof[i] == data.reference_dof[i]
               for i in grid.selected_records())
    assert len(grid.selected_records()) == 2


def test_a_drive_point_with_the_other_sign_is_on_the_diagonal(window, pump):
    """An accelerometer mounted the other way round from the force gives
    206X- against 206X+: still one node along one axis, so a drive point."""
    data = Frf(abscissa=np.linspace(0, 128, 64),
               ordinate=np.ones((4, 64), dtype=complex),
               response_dof=['206X-', '206X-', '207Y+', '207Y+'],
               reference_dof=['206X+', '207Y-'] * 2,
               ordinate_dim='acceleration/force',
               ordinate_unit='m/s**2', reference_unit='N')
    name = shown(window, data)
    pump()
    assert window.data_pane.drive_point_action.isVisible()
    window.data_pane.drive_point_action.trigger()
    pump()
    picked = [(data.response_dof[i], data.reference_dof[i])
              for i in window.record_grids[name].selected_records()]
    assert picked == [('206X-', '206X+'), ('207Y+', '207Y-')]
    assert window.data_pane.drive_point_action.isChecked()


def test_same_axis_reads_node_and_axis_not_sign():
    from visualdynamics.core.data import same_axis

    assert same_axis('206X+', '206X-')
    assert same_axis('206RZ-', '206RZ+')
    assert same_axis('206X', '206X-'), 'unsigned is the positive direction'
    assert not same_axis('206X+', '206Y+')
    assert not same_axis('206X+', '207X+')
    assert not same_axis('206X+', '206RX+'), 'a rotation is another axis'
    assert same_axis('ref', 'ref') and not same_axis('ref', 'other')
