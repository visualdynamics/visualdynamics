"""Which channels are the references — chosen, shown and stored.

An FRF and a multiple coherence are computed against reference channels,
and until this the references were a guess from the quantities: every
force or voltage channel, nothing else, no way to say otherwise short of
a script. Now the choice rides the time history as `references`, the
way its averaging does — set in the grid's Ref column (Brandon,
2026-09-25), saved with the object, journaled, read by both
computations so the coherence beside an FRF is that FRF's coherence,
and fingerprinted so a change badges what was derived.

A reference is a channel, not a point: a drive point's load cell and
the accelerometer beside it can each be one, or both — an accelerometer
as a reference gives acceleration-over-acceleration FRFs, which is a
transmissibility, and a real thing to want.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path
from PySide6.QtCore import Qt
from test_compute_frfs import shaken

import visualdynamics
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import TimeHistory
from visualdynamics.project import Project


def _drive_point(samples=8192, seed=1):
    """A load cell and an accelerometer at one point, and two more
    accelerometers: the impedance-head layout."""
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / 2048.0
    force = rng.normal(0.0, 1.0, samples)
    rows = [force, 2.0 * force, 3.0 * force, 0.5 * force]
    history = TimeHistory(t, np.array(rows),
                          response_dof=['1Z+', '1Z+', '2Z+', '3Z+'],
                          ordinate_dim=['force', 'acceleration',
                                        'acceleration', 'acceleration'])
    history.averaging = Averaging(frame_length=512, overlap=0.5,
                                  window='hann', frames=20)
    return history


# ---- the resolution -----------------------------------------------------

def test_with_nothing_said_the_references_are_the_excitation_channels():
    history, _gains = shaken(drives=2)
    assert history.references is None
    assert history.reference_channels() == [('200Z+', 'force'),
                                            ('210Z+', 'force')]
    # the same channels the FRFs and the coherence use
    frfs = history.compute_frfs()
    assert set(frfs.reference_dof) == {'200Z+', '210Z+'}
    assert set(frfs.response_dof) == {'100Z+', '101Z+', '102Z+'}


def test_a_bare_dof_picks_the_force_there_and_a_pair_picks_the_channel():
    history = _drive_point()
    assert history.reference_channels() == [('1Z+', 'force')]
    assert history.reference_channels(['1Z+']) == [('1Z+', 'force')], (
        'a DOF names the excitation channel at the point')
    assert history.reference_channels([('1Z+', 'acceleration')]) == [
        ('1Z+', 'acceleration')], 'a pair names exactly that channel'
    assert history.reference_channels(['2Z+']) == [('2Z+', 'acceleration')], (
        'a DOF with no force there is still a reference')
    assert history.reference_channels(['9Z+', ('1Z+', 'strain')]) == [], (
        'a name matching nothing is left out, for the caller to say so')


def test_an_accelerometer_as_the_reference_gives_transmissibilities():
    history = _drive_point()
    history.references = [('1Z+', 'acceleration')]
    frfs = history.compute_frfs(method='H1')
    assert list(frfs.reference_dof) == ['1Z+'] * 3
    assert set(frfs.response_dof) == {'1Z+', '2Z+', '3Z+'}, (
        'the load cell is a response now, and the accelerometer is not')
    assert set(frfs.ordinate_dim) == {'acceleration/acceleration',
                                      'force/acceleration'}
    # 2Z+ moves 3/2 as much as 1Z+ does: a flat transmissibility of 1.5
    at_2 = frfs.ordinate[list(frfs.response_dof).index('2Z+')]
    assert np.allclose(np.abs(at_2[1:-1]), 1.5, rtol=1e-6)


def test_a_load_cell_and_the_accelerometer_beside_it_can_both_be_references():
    history = _drive_point()
    history.references = [('1Z+', 'force'), ('1Z+', 'acceleration')]
    frfs = history.compute_frfs(method='Hv')
    assert frfs.num_records == 4, 'two responses against two references'
    assert set(frfs.response_dof) == {'2Z+', '3Z+'}
    assert sorted(set(frfs.ordinate_dim)) == ['acceleration/acceleration',
                                              'acceleration/force']
    coherence = history.compute_multiple_coherence()
    assert set(coherence.response_dof) == {'2Z+', '3Z+'}, (
        'the coherence reads the same two references')


def test_the_setting_is_read_by_both_computations_and_can_be_overridden():
    history, _gains = shaken(drives=2)
    history.references = [('200Z+', 'force')]
    frfs = history.compute_frfs()
    assert set(frfs.reference_dof) == {'200Z+'}
    coherence = history.compute_multiple_coherence()
    assert '210Z+' in coherence.response_dof, (
        'the unchosen drive is a response to the coherence too')
    both = history.compute_frfs(references=['200Z+', '210Z+'])
    assert set(both.reference_dof) == {'200Z+', '210Z+'}, (
        'a call may still say otherwise, as a script does')


def test_a_choice_that_names_no_channel_is_refused_by_name():
    history, _gains = shaken(drives=2)
    history.references = [('999Z+', 'force')]
    with pytest.raises(ValueError, match=r"none of \[\('999Z\+', 'force'\)\]"):
        history.compute_frfs()
    history.references = []
    with pytest.raises(ValueError, match=r'none of \[\] is in this history'):
        history.compute_multiple_coherence()


# ---- the file and the fingerprint ---------------------------------------

def test_the_references_ride_the_native_file(tmp_path):
    history = _drive_point()
    history.references = [('1Z+', 'acceleration'), ('3Z+', 'acceleration')]
    visualdynamics.save(history, tmp_path / 'run.vdyn')
    back = visualdynamics.load(tmp_path / 'run.vdyn')
    assert back.references == [('1Z+', 'acceleration'), ('3Z+', 'acceleration')]
    history.references = []
    visualdynamics.save(history, tmp_path / 'none.vdyn')
    assert visualdynamics.load(tmp_path / 'none.vdyn').references == [], (
        'every box unticked is a choice, and comes back as one')
    history.references = None
    visualdynamics.save(history, tmp_path / 'guess.vdyn')
    assert visualdynamics.load(tmp_path / 'guess.vdyn').references is None


def test_changing_the_references_badges_the_frfs_and_the_coherence():
    project = Project()
    history, _gains = shaken(drives=2)
    project.add('Run', history)
    frfs = project.compute_frfs('Run')
    coherence = project.compute_multiple_coherence('Run')
    psds = project.compute_psds('Run')
    assert project.stale() == {}
    # the fingerprint without a choice is the plain averaging one, so a
    # project saved before the setting existed opens with nothing stale
    from dataclasses import asdict

    assert project.provenance[frfs]['state'] == ('averaging',
                                                 asdict(history.averaging))
    history.references = [('200Z+', 'force')]
    stale = project.stale()
    assert set(stale) == {frfs, coherence}, 'the PSDs read no references'
    assert 'references 200Z+ force' in stale[frfs], stale[frfs]
    project.refresh(frfs)
    assert set(project[frfs].reference_dof) == {'200Z+'}
    assert frfs not in project.stale()
    assert psds not in project.stale()


# ---- the grid ------------------------------------------------------------

def _check_column(grid):
    return grid.check_column


def test_a_time_historys_grid_has_a_ref_column_shown_first(qt_app):
    from visualdynamics.gui.record_grid import RecordGrid

    history = _drive_point()
    grid = RecordGrid(history)
    column = _check_column(grid)
    assert column == grid.columnCount() - 1, (
        'the last logical column, so record cells keep their numbers')
    assert grid.horizontalHeader().visualIndex(column) == 0, 'shown first'
    assert grid.horizontalHeaderItem(column).text() == 'Ref'
    assert grid.reference_rows() == [0], 'the load cell, ticked'
    assert grid.row_keys[0].quantity == 'force'
    box = grid.item(0, column)
    assert box.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert not box.flags() & Qt.ItemFlag.ItemIsSelectable, (
        'a tick is a setting, not a pick of the row')
    # the record cells are where they always were
    assert grid.item(0, 0).icon() is not None and (0, 0) in grid._records
    grid.item(2, 0).setSelected(True)
    assert grid.selected_records() == [2]


def test_other_grids_have_no_ref_column(qt_app):
    from visualdynamics.gui.record_grid import RecordGrid

    history = _drive_point()
    grid = RecordGrid(history.compute_psds())
    assert _check_column(grid) is None and grid.reference_rows() == []
    headers = [grid.horizontalHeaderItem(c).text()
               for c in range(grid.columnCount())]
    assert 'Ref' not in headers


def test_a_grid_with_a_choice_shows_the_choice(qt_app):
    from visualdynamics.gui.record_grid import RecordGrid

    history = _drive_point()
    history.references = [('1Z+', 'acceleration'), ('3Z+', 'acceleration')]
    grid = RecordGrid(history)
    assert grid.reference_rows() == [1, 3]


def test_ticking_a_box_in_the_window_stores_journals_and_badges(window, pump):
    from test_workflow_journals import _replay

    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    name = next(n for n, o in window.objects.items()
                if isinstance(o, TimeHistory))
    history = window.objects[name]
    frfs = window.project.compute_frfs(name)
    pump()
    item = window._item_for_object(name)
    item.setExpanded(True)
    pump()
    grid = window.record_grids[name]
    column = grid.check_column
    forces = [row for row, key in enumerate(grid.row_keys)
              if key.quantity == 'force']
    assert grid.reference_rows() == forces, 'the guess, ticked'
    # tick the first accelerometer: it joins the references
    accel = next(row for row, key in enumerate(grid.row_keys)
                 if key.quantity == 'acceleration')
    grid.item(accel, column).setCheckState(Qt.CheckState.Checked)
    pump()
    key = grid.row_keys[accel]
    assert history.references[-1] == (key.dof, 'acceleration')
    assert len(history.references) == len(forces) + 1
    assert f"project[{name!r}].references = " in window.project.journal[-1]
    assert 'references:' in window.statusBar().currentMessage()
    assert frfs in window._stale, 'the FRFs wear the badge from this moment'
    # untick a load cell: it leaves, and the repeated write is one line
    lines_before = len(window.project.journal)
    grid.item(forces[0], column).setCheckState(Qt.CheckState.Unchecked)
    pump()
    assert (grid.row_keys[forces[0]].dof, 'force') not in history.references
    assert len(window.project.journal) == lines_before, (
        'a second write to the same slot replaces its line')
    window.project.refresh(frfs)
    pump()
    assert set(window.project[frfs].reference_dof) == {
        dof for dof, _q in history.references}
    _replay(window)
