"""Which records a deflection animation shows.

An animation gives every node one trajectory, so it can use at most one
record per DOF. Before this, a whole 20-average time object animated the
*sum* of its averages — a waveform nobody measured — a grid subset could
never animate at all (each picked cell became its own series entry and the
animator refuses more than one), and picking cells deselected the geometry.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from conftest import fixture_path

from visualdynamics.core.data import TimeHistory
from visualdynamics.gui.main_window import animation_records


def blocked(dofs=('1Z+', '2Z+'), averages=3):
    dof_list, blocks = [], []
    for dof in dofs:
        for average in range(averages):
            dof_list.append(dof)
            blocks.append(f'avg {average + 1}')
    n = len(dof_list)
    return TimeHistory(abscissa=np.arange(4.0),
                       ordinate=np.arange(float(n * 4)).reshape(n, 4),
                       response_dof=dof_list, block=blocks,
                       ordinate_dim='acceleration', ordinate_unit='m/s**2')


def test_a_whole_blocked_object_animates_its_first_average():
    """Not the sum of all twenty, which is not a measurement anyone took."""
    data = blocked()
    indices, note = animation_records(data, None)
    assert [data.block[i] for i in indices] == ['avg 1', 'avg 1']
    assert [data.response_dof[i] for i in indices] == ['1Z+', '2Z+']
    assert 'avg 1 of 3 averages' in note


def test_an_explicit_selection_is_taken_as_given():
    data = blocked()
    indices, note = animation_records(data, [1, 4])   # avg 2 of each channel
    assert indices == [1, 4]
    assert note == ''


def test_an_unblocked_object_is_untouched():
    plain = TimeHistory(abscissa=np.arange(4.0), ordinate=np.zeros((2, 4)),
                        response_dof=['1Z+', '2Z+'])
    indices, note = animation_records(plain, None)
    assert indices == [0, 1] and note == ''


def test_a_repeated_dof_is_dropped_with_a_note():
    """A drive point carries a load cell beside its accelerometer; meters per
    second squared plus newtons is not a deflection."""
    data = TimeHistory(abscissa=np.arange(4.0),
                       ordinate=np.arange(8.0).reshape(2, 4),
                       response_dof=['1Z+', '1Z+'],
                       ordinate_dim=['acceleration', 'force'],
                       ordinate_unit=['m/s**2', 'N'])
    indices, note = animation_records(data, [0, 1])
    assert indices == [0], 'the first record per DOF wins'
    assert '1 records repeat an animated DOF' in note


def test_the_two_rules_compose():
    """First average, and dedupe inside it."""
    data = TimeHistory(abscissa=np.arange(4.0),
                       ordinate=np.arange(16.0).reshape(4, 4),
                       response_dof=['1Z+', '1Z+', '1Z+', '1Z+'],
                       block=['avg 1', 'avg 2'] * 2,
                       ordinate_dim=['acceleration'] * 2 + ['force'] * 2,
                       ordinate_unit=['m/s**2'] * 2 + ['N'] * 2)
    indices, note = animation_records(data, None)
    assert indices == [0]
    assert 'avg 1' in note and 'repeat an animated DOF' in note


# --- the window: the user's actual scenario ----------------------------------

def loaded(window):
    window.import_paths([fixture_path('plate', 'geometry.npz'),
                         fixture_path('plate', 'modal_spectra.nc4')])
    return window


def test_geometry_plus_whole_time_object_arms_the_animator(window, pump):
    loaded(window)
    window.tree.clearSelection()
    for name in ('Geometry', 'Time History'):
        window._item_for_object(name).setSelected(True)
    pump()
    assert window.play_action.isEnabled()
    message = window.statusBar().currentMessage()
    assert 'animating avg 1 of 20 averages' in message, message


def test_picking_grid_cells_keeps_the_geometry_selected(window, pump):
    """This is what made subset animation impossible: the grid pick cleared
    the whole tree selection, geometry included."""
    loaded(window)
    window.tree.clearSelection()
    window._item_for_object('Geometry').setSelected(True)
    pump()
    grid = window.record_grids['Time History']
    grid.item(0, 0).setSelected(True)
    pump()
    selected = {item.text(0) for item in window.tree.selectedItems()}
    assert selected == {'Geometry', 'Time History'}


def test_a_grid_subset_animates_just_those_records(window, pump):
    """Accelerations from the first average, say."""
    loaded(window)
    window.tree.clearSelection()
    window._item_for_object('Geometry').setSelected(True)
    pump()
    grid = window.record_grids['Time History']
    for row in range(4):                    # four channels, first average
        grid.item(row, 0).setSelected(True)
    pump()
    assert window.play_action.isEnabled()
    assert window.animator is not None


def test_another_data_object_still_gives_way_to_a_grid_pick(window, pump):
    """Grids compete with each other; only non-data selections survive."""
    loaded(window)
    window.tree.clearSelection()
    window._item_for_object('FRF').setSelected(True)
    pump()
    grid = window.record_grids['Time History']
    grid.item(0, 0).setSelected(True)
    pump()
    selected = {item.text(0) for item in window.tree.selectedItems()}
    assert selected == {'Time History'}


# --- the operating deflection shape ------------------------------------------


def test_geometry_plus_frf_animates_the_ods(window, pump):
    """Complex spectra deflect the geometry at one frequency line, and
    the cursor starts on the strongest line, not the plot's left edge."""
    from visualdynamics.deform import OdsDeflection

    loaded(window)
    window.tree.clearSelection()
    for name in ('Geometry', 'FRF'):
        window._item_for_object(name).setSelected(True)
    pump()
    assert window.animator is not None
    assert window.play_action.isEnabled()
    deflection = window.animator.deflection
    assert isinstance(deflection, OdsDeflection)
    assert deflection.line == deflection.strongest_line > 0
    message = window.statusBar().currentMessage()
    assert 'operating deflection' in message, message
    assert 'repeat an animated DOF' in message, (
        'the second reference column gives way')


def test_the_ods_cursor_moves_the_line(window, pump):
    """Dragging the cursor re-deflects at that frequency; the phase and
    the scene scale stay put."""
    loaded(window)
    window.tree.clearSelection()
    for name in ('Geometry', 'FRF'):
        window._item_for_object(name).setSelected(True)
    pump()
    deflection = window.animator.deflection
    scale = window.animator.auto
    started = deflection.line
    before = deflection.offsets(0.0).copy()
    window._cursor.setValue(float(window._cursor_abscissa[10]))
    pump()
    assert deflection.line == 10 != started
    assert not np.allclose(deflection.offsets(0.0), before), (
        'the deflection followed the cursor')
    assert window.animator.auto == scale, (
        'moving the line must not rescale the scene')


# --- the PSD envelope --------------------------------------------------------


def test_geometry_plus_psd_shows_the_envelope(window, pump):
    """Two mirrored copies at the cursor line, color absolute; the
    cursor starts on the strongest line."""
    from visualdynamics.deform import EnvelopeDeflection
    from visualdynamics.viz.animate import PairedAnimator

    loaded(window)
    window.import_paths([fixture_path('plate', 'psd.npz')])
    pump()
    psd = next(n for n, o in window.objects.items()
               if type(o).__name__ == 'Psd')
    window.tree.clearSelection()
    for name in ('Geometry', psd):
        window._item_for_object(name).setSelected(True)
    pump()
    assert isinstance(window.animator, PairedAnimator)
    deflection = window.animator.deflection
    assert isinstance(deflection, EnvelopeDeflection)
    assert deflection.line == deflection.strongest_line > 0
    first, second = window.animator.first, window.animator.second
    assert first.scale == -second.scale != 0, 'the pair mirrors'
    assert not window._phase_action.isVisible(), 'no phase to sweep'
    assert 'envelope' in window.statusBar().currentMessage()
    # the cursor picks the line
    window._cursor.setValue(float(window._cursor_abscissa[5]))
    pump()
    assert deflection.line == 5


def test_a_whole_cpsd_animates_its_principal_shape(window, pump):
    """A CPSD carries phase, so selecting the whole object beside a
    geometry animates the principal operating deflection — the
    cross-spectral matrix's dominant eigenvector per line — not the
    phaseless envelope, and not a shape against any one reference."""
    from visualdynamics.core.data import Psd
    from visualdynamics.deform import OdsDeflection

    loaded(window)
    channels = ['101Z+', '707Z+']
    lines = 8
    # a rank-one matrix built from a known complex shape: the principal
    # shape must recover it
    shape = np.array([1.0, 2.0j])
    records, rdofs, fdofs = [], [], []
    for i, ri in enumerate(channels):
        for j, rj in enumerate(channels):
            records.append(np.full(lines, shape[i] * np.conj(shape[j])))
            rdofs.append(ri)
            fdofs.append(rj)
    cpsd = Psd(np.arange(1.0, lines + 1.0), np.array(records),
               response_dof=rdofs, reference_dof=fdofs,
               ordinate_dim='acceleration**2/frequency',
               ordinate_unit='m/s**2')
    window.add_object('CPSD', cpsd)
    window.tree.clearSelection()
    for name in ('Geometry', 'CPSD'):
        window._item_for_object(name).setSelected(True)
    pump()
    deflection = window.animator.deflection
    assert isinstance(deflection, OdsDeflection), (
        'phase information in, an operating shape out')
    assert 'principal' in window.statusBar().currentMessage()
    # the recovered pattern is the planted shape: 90 degrees between
    # the channels, twice the amplitude on the second
    # offsets reuses one buffer: copy before reading twice
    real = deflection.offsets(0.0).copy()
    imag = -deflection.offsets(np.pi / 2).copy()
    values = (real + 1j * imag)[:, 2]
    assert np.isclose(abs(values[1]) / abs(values[0]), 2.0, rtol=1e-6)
    assert np.isclose(abs(np.angle(values[1] / values[0])), np.pi / 2,
                      rtol=1e-6)


def test_a_picked_mix_of_autos_and_cross_still_envelopes(window, pump):
    """An explicit pick that is not one reference column falls to the
    envelope, and its cross rows are filtered before the
    one-record-per-DOF rule: the cross row comes FIRST here, and
    filtered after the dedupe it would shadow 101Z+'s genuine auto."""
    from visualdynamics.core.data import Psd
    from visualdynamics.deform import EnvelopeDeflection

    loaded(window)
    frequencies = np.arange(1.0, 9.0)
    lines = len(frequencies)
    cpsd = Psd(frequencies,
               np.array([np.full(lines, 1.0 + 1.0j),
                         np.full(lines, 4.0),
                         np.full(lines, 9.0)]),
               response_dof=['101Z+', '101Z+', '707Z+'],
               reference_dof=['707Z+', '101Z+', '707Z+'],
               ordinate_dim='acceleration', ordinate_unit='m/s**2')
    window.add_object('CPSD', cpsd)
    window.tree.clearSelection()
    window._item_for_object('Geometry').setSelected(True)
    pump()
    grid = window.record_grids['CPSD']
    for row, column in ((0, 0), (0, 1), (1, 0)):
        grid.item(row, column).setSelected(True)
    pump()
    deflection = window.animator.deflection
    assert isinstance(deflection, EnvelopeDeflection)
    assert '1 cross records not shown' in (
        window.statusBar().currentMessage())
    # the autos' amplitudes, sqrt(4)/sqrt(9) normalized — not the cross
    offsets = np.abs(deflection.offsets(0.0)).max(axis=1)
    assert np.allclose(sorted(offsets), [2.0 / 3.0, 1.0])


def test_a_computed_psd_with_no_reference_list_still_envelopes(window,
                                                               pump):
    """compute_psds makes a Psd whose reference_dof is None — not a
    list — and the first release of the envelope crashed on it inside
    the selection handler, which reads as the feature silently not
    existing."""
    from visualdynamics.deform import EnvelopeDeflection

    loaded(window)
    window.tree.setCurrentItem(window._item_for_object('Time History'))
    window.tree.clearSelection()
    window._item_for_object('Time History').setSelected(True)
    pump()
    window.compute_psds()
    pump()
    psd = next(n for n, o in window.objects.items()
               if type(o).__name__ == 'Psd')
    assert window.objects[psd].reference_dof is None, (
        'the fixture must exercise the None case or this test is idle')
    window.tree.clearSelection()
    for name in ('Geometry', psd):
        window._item_for_object(name).setSelected(True)
    pump()
    assert isinstance(window.animator.deflection, EnvelopeDeflection)
    assert window._cursor is not None


def test_a_drive_points_force_psd_does_not_eat_its_accelerometer(window,
                                                                 pump):
    """The quantity filter runs before the one-record-per-DOF rule: a
    drive point's force PSD listed first would otherwise shadow its own
    DOF's accelerometer out of the envelope, and picking Force in the
    combo found its records already deduped away."""
    from visualdynamics.core.data import Psd
    from visualdynamics.deform import EnvelopeDeflection

    loaded(window)
    frequencies = np.arange(1.0, 9.0)
    lines = len(frequencies)
    psd = Psd(frequencies,
              np.array([np.full(lines, 25.0),      # force at the drive
                        np.full(lines, 4.0),       # accel, same DOF
                        np.full(lines, 16.0)]),
              response_dof=['101Z+', '101Z+', '707Z+'],
              ordinate_dim=['force', 'acceleration', 'acceleration'],
              ordinate_unit=['N', 'm/s**2', 'm/s**2'])
    window.add_object('Run PSDs', psd)
    window.tree.clearSelection()
    for name in ('Geometry', 'Run PSDs'):
        window._item_for_object(name).setSelected(True)
    pump()
    deflection = window.animator.deflection
    assert isinstance(deflection, EnvelopeDeflection)
    # both accel DOFs deflect, sqrt(4)/sqrt(16) normalized — the force
    # record neither shadows 101Z+ nor joins the pattern
    offsets = np.abs(deflection.offsets(0.0)).max(axis=1)
    assert np.allclose(sorted(offsets), [0.5, 1.0])
    assert 'other quantities not shown' in (
        window.statusBar().currentMessage())


def test_the_cursor_rides_the_quantity_it_deflects(window, pump):
    """Mixed quantities stack one plot per row; choosing Force must put
    the cursor on the force plot, not leave it on the acceleration one."""
    from visualdynamics.core.data import Psd

    loaded(window)
    frequencies = np.arange(1.0, 9.0)
    lines = len(frequencies)
    psd = Psd(frequencies,
              np.array([np.full(lines, 4.0),
                        np.full(lines, 16.0),
                        np.full(lines, 25.0)]),
              response_dof=['101Z+', '707Z+', '1310Z+'],
              ordinate_dim=['acceleration', 'acceleration', 'force'],
              ordinate_unit=['m/s**2', 'm/s**2', 'N'])
    window.add_object('Run PSDs', psd)
    window.tree.clearSelection()
    for name in ('Geometry', 'Run PSDs'):
        window._item_for_object(name).setSelected(True)
    pump()

    def cursor_plot_dimension():
        row = 0
        while (plot := window.data_pane.graphics.getItem(row, 0)) \
                is not None:
            if window._cursor in plot.items:
                return plot.series_key[1]
            row += 1
        return None

    assert cursor_plot_dimension() == 'acceleration'
    force_row = next(i for i in range(window.dofs_combo.count())
                     if window.dofs_combo.itemData(i) == 'force')
    window.dofs_combo.setCurrentIndex(force_row)
    pump()
    assert cursor_plot_dimension() == 'force'
    # and the envelope deflects the force record's node
    assert len(window.animator.deflection.rows) == 1


def test_the_envelope_scene_keeps_a_zero_reference(window, pump):
    """The two extremes are read against the undeflected geometry — a
    faint copy that never moves."""
    loaded(window)
    window.import_paths([fixture_path('plate', 'psd.npz')])
    pump()
    psd = next(n for n, o in window.objects.items()
               if type(o).__name__ == 'Psd')
    window.tree.clearSelection()
    for name in ('Geometry', psd):
        window._item_for_object(name).setSelected(True)
    pump()
    reference = window._envelope_reference
    assert reference, 'the zero-displacement copy is in the scene'
    assert np.allclose(reference[0].points, window.animator.first.base), (
        'and it sits at the undeflected positions')


def test_a_picked_cpsd_reference_column_animates_the_ods(window, pump):
    """A column against one reference channel carries each response's
    phase relative to that reference — the operating deflection shape
    from operating data. The whole CPSD stays an envelope, because no
    reference was chosen."""
    from visualdynamics.core.data import Psd
    from visualdynamics.deform import OdsDeflection

    loaded(window)
    channels = [('101Z+', 'acceleration'), ('101Z+', 'force'),
                ('707Z+', 'acceleration')]
    records, rdofs, fdofs, dims = [], [], [], []
    for rd, rq in channels:
        for fd, fq in channels:
            same = (rd, rq) == (fd, fq)
            records.append(np.full(4, 4.0 + 0j if same else 1.0 + 0.5j))
            rdofs.append(rd)
            fdofs.append(fd)
            dims.append(f'{rq}**2/frequency' if rq == fq
                        else f'{rq}*{fq}/frequency')
    cpsd = Psd(np.arange(1.0, 5.0), np.array(records),
               response_dof=rdofs, reference_dof=fdofs,
               ordinate_dim=dims)
    window.add_object('CPSD', cpsd)
    window.tree.clearSelection()
    for name in ('Geometry', 'CPSD'):
        window._item_for_object(name).setSelected(True)
    pump()
    assert isinstance(window.animator.deflection, OdsDeflection), (
        'a whole CPSD answers with its principal shape')
    assert 'principal' in window.statusBar().currentMessage()
    # pick the acceleration-reference column: a shape against ONE
    # reference channel instead of the matrix's own
    grid = window.record_grids['CPSD']
    for row in range(3):
        grid.item(row, 0).setSelected(True)
    pump()
    deflection = window.animator.deflection
    assert isinstance(deflection, OdsDeflection), (
        'a reference column is an honest complex shape')
    assert len(deflection.rows) == 2, (
        'the two acceleration responses; the force response waits for '
        'its own quantity')


# ---- a grid that gives way, gives way visibly ---------------------------


def _transient_pair(window, pump):
    from visualdynamics.core.data import TimeHistory, TransientSpecification

    path = fixture_path('..', 'stressdata', 'plate_projects', 'transient.vdyn')
    if not os.path.exists(path):
        # gitignored and regenerable, so absent on every machine but
        # this one — and an import that fails raises a warning box,
        # which under a headless run is a hang: CI stalled here for 25
        # minutes and was killed (2026-09-13)
        pytest.skip('plate stressdata not on this machine')
    window.import_paths([path])
    pump()
    history = next(n for n, o in window.objects.items()
                   if isinstance(o, TimeHistory)
                   and not isinstance(o, TransientSpecification))
    spec = next(n for n, o in window.objects.items()
                if isinstance(o, TransientSpecification))
    for name in (history, spec):
        window._item_for_object(name).setExpanded(True)
    pump()
    return history, spec


def _click_record(window, name, row, modifier=None):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    grid = window.record_grids[name]
    rect = grid.visualRect(grid.model().index(row, 0))
    QTest.mouseClick(grid.viewport(), Qt.MouseButton.LeftButton,
                     modifier or Qt.KeyboardModifier.NoModifier,
                     rect.center())


def test_a_grid_that_gives_way_clears_its_rows(window, pump):
    """Brandon, on the transient project: picked a record in the time
    history, then a record in the specification, and the plot showed
    'the time history without the specification'. What actually
    happened: the plain click made the history give way — but only its
    *tree item* went gray, its grid rows stayed lit, so the tree
    showed two selected sub-items whose effective selection was one,
    and the spec's lone waveform read as the record. A selection the
    plot will never honor must not stay highlighted."""
    history, spec = _transient_pair(window, pump)
    _click_record(window, history, 0)
    pump()
    _click_record(window, spec, 0)     # plain: the history gives way
    pump()
    assert not window._item_for_object(history).isSelected()
    assert window.record_grids[history].selected_records() == [], (
        'the grid gave way visibly, with its owner')
    picked = [(k, n, d) for k, n, _o, d in window.selected_references()]
    assert picked == [('record', spec, 0)], 'what is lit is what is meant'


def test_the_modifier_still_combines_the_two(window, pump):
    """The comparison Brandon was after: hold the multi-select modifier
    and both records stay — the replication overlay draws the event
    over its specification."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    history, spec = _transient_pair(window, pump)
    _click_record(window, history, 0)
    pump()
    _click_record(window, spec, 0, Qt.KeyboardModifier.ControlModifier)
    pump()
    names = {n for _k, n, _o, _d in window.selected_references()}
    assert names == {history, spec}
    plots = [it for it in window.data_pane.graphics.ci.items
             if isinstance(it, pg.PlotItem)]
    drawn = [c.name() for p in plots for c in p.listDataItems()]
    assert any(n and 'Specification' in n for n in drawn), \
        'the specification is on the plot'
    assert any(n and 'Event' in n for n in drawn), \
        'and the measured event is over it'


def test_a_tree_click_clears_an_abandoned_grid(window, pump):
    """The same lie by the other route: records picked in a grid, then
    a plain click on another tree object — Qt deselects the owner, and
    the rows must go with it or they resurface as a phantom
    restriction the next time the object is clicked."""
    history, spec = _transient_pair(window, pump)
    _click_record(window, history, 0)
    pump()
    item = window._item_for_object(spec)
    window.tree.setCurrentItem(item)
    item.setSelected(True)
    window._item_for_object(history).setSelected(False)
    pump()
    assert window.record_grids[history].selected_records() == [], (
        'no rows stay lit on an unselected owner')


# ---- cross-object picks pair by DOF -------------------------------------


def test_mismatched_picks_draw_nothing_and_say_why(window, pump):
    """Brandon: the 104Z+ record picked against the 101Z+ target drew
    both channels with both targets — pooling read two disagreeing
    picks as one request for two channels. Picks on both sides must
    agree; a record and a target sharing no DOF is not a comparison,
    and the honest plot of it is empty with the reason said."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    history, spec = _transient_pair(window, pump)
    hobj, sobj = window.objects[history], window.objects[spec]
    hrow = [str(d) for d in hobj.response_dof].index('104Z+')
    srow = [str(d) for d in sobj.response_dof].index('101Z+')
    _click_record(window, history, hrow)
    pump()
    _click_record(window, spec, srow, Qt.KeyboardModifier.ControlModifier)
    pump()
    plots = [it for it in window.data_pane.graphics.ci.items
             if isinstance(it, pg.PlotItem)]
    drawn = [c for p in plots for c in p.listDataItems()]
    assert not drawn, 'nothing to draw when the picks cannot meet'
    assert 'share no channel' in window.statusBar().currentMessage()


def test_matching_picks_draw_that_channel_alone(window, pump):
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    history, spec = _transient_pair(window, pump)
    hobj, sobj = window.objects[history], window.objects[spec]
    hrow = [str(d) for d in hobj.response_dof].index('104Z+')
    srow = [str(d) for d in sobj.response_dof].index('104Z+')
    _click_record(window, history, hrow)
    pump()
    _click_record(window, spec, srow, Qt.KeyboardModifier.ControlModifier)
    pump()
    plots = [it for it in window.data_pane.graphics.ci.items
             if isinstance(it, pg.PlotItem)]
    drawn = [c.name() for p in plots for c in p.listDataItems()]
    assert drawn == ['Event 1: 104Z+', 'Specification: 104Z+'], (
        f'the one agreed channel, target under event, got {drawn}')


def test_a_one_sided_pick_still_names_the_channel_for_both(window, pump):
    """A channel chosen on the record alone is the same request as
    choosing it on the target — the pooling that is right stays."""

    import pyqtgraph as pg

    history, spec = _transient_pair(window, pump)
    hobj = window.objects[history]
    hrow = [str(d) for d in hobj.response_dof].index('104Z+')
    _click_record(window, history, hrow)
    pump()
    item = window._item_for_object(spec)     # the whole target object
    window.tree.blockSignals(True)
    item.setSelected(True)
    window.tree.blockSignals(False)
    window.render_current()
    pump()
    plots = [it for it in window.data_pane.graphics.ci.items
             if isinstance(it, pg.PlotItem)]
    drawn = [c.name() for p in plots for c in p.listDataItems()]
    assert drawn == ['Event 1: 104Z+', 'Specification: 104Z+'], drawn
