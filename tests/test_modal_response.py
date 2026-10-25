"""Modal responses: each mode answering a record's forces, appended to the
record.

Brandon, 2026-10-25: with an excitation and a model of the system, the
response is computed and kept as modal responses, one record per mode,
compact where a large model has many more physical DOFs than modes. The
modes are whatever is selected: a whole set for every mode, or the modes
picked in its grid. Checked against a direct simulation of the physical
equations, the package's own FRF synthesis and closed forms.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import bar_acts, select_objects
from scipy.linalg import eigh
from scipy.signal import lsim

import visualdynamics
from visualdynamics.core import modal_response as mr
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.shapes import ShapeSet
from visualdynamics.core.transform import expands, modal_records, reads_as, to_physical

FS = 4096.0
M = np.diag([2.0, 1.0])
K = np.array([[3e5, -1e5], [-1e5, 1e5]])
ZETA = np.array([0.02, 0.03])


def _model():
    lam, phi = eigh(K, M)                       # mass-normalized
    shapes = ShapeSet(np.sqrt(lam) / (2 * np.pi), ZETA, ['1X+', '2X+'],
                      phi.T, mass_unit='kg')
    return shapes, phi, np.sqrt(lam)


def _record(seconds=4.0, seed=1, force=None):
    t = np.arange(int(seconds * FS)) / FS
    f = (np.random.default_rng(seed).standard_normal(t.size) * 10.0
         if force is None else force(t))
    return TimeHistory(t, np.vstack([f, np.zeros_like(t)]),
                       response_dof=['1X+', '2X+'],
                       ordinate_dim=['force', 'acceleration'])


def _direct(force, t, phi, w, rows):
    """The physical equations simulated directly, the force at 1X+
    linear between samples, from rest: (displacement, velocity,
    acceleration) at both DOFs."""
    damping = M @ phi @ np.diag(2 * ZETA * w) @ phi.T @ M
    a = np.block([[np.zeros((2, 2)), np.eye(2)],
                  [-np.linalg.solve(M, K), -np.linalg.solve(M, damping)]])
    b = np.vstack([np.zeros((2, 1)),
                   np.linalg.solve(M, np.array([[1.0], [0.0]]))])
    c = {'displacement': (np.hstack([np.eye(2), np.zeros((2, 2))]),
                          np.zeros((2, 1))),
         'velocity': (np.hstack([np.zeros((2, 2)), np.eye(2)]),
                      np.zeros((2, 1))),
         'acceleration': (a[2:, :], b[2:, :])}[rows]
    return lsim((a, b, *c), force, t, interp=True)[1].T


# ---- exact, against the physical equations ----------------------------------

@pytest.mark.parametrize('quantity', ['displacement', 'velocity',
                                      'acceleration'])
def test_the_modal_responses_expand_to_the_physical_response(quantity):
    shapes, phi, w = _model()
    record = _record()
    q = mr.modal_responses(record, shapes, quantity=quantity)
    direct = _direct(record.ordinate[0], record.abscissa, phi, w, quantity)
    error = np.abs(phi @ q.ordinate - direct).max() / np.abs(direct).max()
    assert error < 1e-7


def test_a_steady_sine_is_the_synthesized_frf():
    shapes, phi, _ = _model()
    record = _record(force=lambda t: np.sin(2 * np.pi * 37.0 * t))
    acceleration = phi @ mr.modal_responses(record, shapes).ordinate
    tail = acceleration[0, -int(FS):]
    frf = shapes.synthesize_frf([37.0], ['1X+'], ['1X+'], power=2)
    assert np.sqrt(2.0) * tail.std() == pytest.approx(abs(frf).ravel()[0],
                                                      rel=1e-3)


def test_a_rigid_mode_under_a_constant_force_is_ft_squared_over_2m():
    """No head start: from rest at the record's first sample, exactly."""
    rigid = ShapeSet([0.0], [0.0], ['1X+'], [[1.0]], modal_mass=[4.0],
                     mass_unit='kg')
    t = np.arange(int(2 * FS)) / FS
    record = TimeHistory(t, np.full((1, t.size), 3.0), response_dof='1X+',
                         ordinate_dim='force')
    q = mr.modal_responses(record, rigid, quantity='displacement').ordinate[0]
    assert np.abs(q - 3.0 * t ** 2 / (2 * 4.0)).max() < 1e-10
    v = mr.modal_responses(record, rigid, quantity='velocity').ordinate[0]
    assert np.abs(v - 3.0 * t / 4.0).max() < 1e-12
    a = mr.modal_responses(record, rigid).ordinate[0]
    assert np.allclose(a, 3.0 / 4.0)


# ---- what is computed from what ---------------------------------------------

def test_only_the_modes_picked_are_answered_named_by_their_index():
    shapes, _, _ = _model()
    q = mr.modal_responses(_record(), shapes, modes=[1], label='Modes')
    assert q.response_dof == ['M2']
    assert q.ordinate_dim == ['modal_acceleration']
    assert q.comment == [(f'Modes: modal acceleration, mode 2 at '
                          f'{shapes.frequency[1]:.4g} Hz')]


def test_a_record_of_modal_forces_is_used_as_it_is():
    shapes, _, _ = _model()
    physical = _record()
    modal = mr.modal_forces(physical, shapes)
    assert modal.ordinate_dim == ['modal_force', 'modal_force']
    from_modal = mr.modal_responses(modal, shapes)
    from_physical = mr.modal_responses(physical, shapes)
    assert np.allclose(from_modal.ordinate, from_physical.ordinate)


def test_modal_forces_used_as_they_are_keep_their_dimension():
    """Through either kind of set: declared through a mass-normalized
    one, undeclared and hinted through any other, and the same when a
    record of them is read again."""
    shapes, phi, _ = _model()
    unscaled = ShapeSet(shapes.frequency, ZETA, ['1X+', '2X+'], phi.T)
    for modes, dim, hint in ((shapes, 'modal_force', None),
                             (unscaled, 'unknown', 'modal_force')):
        once = mr.modal_forces(_record(seconds=0.1), modes)
        again = mr.modal_forces(once, modes)
        for forces in (once, again):
            assert forces.ordinate_dim == [dim, dim]
            assert forces.dimension_hint == [hint, hint]
        assert np.array_equal(again.ordinate, once.ordinate)


def test_the_preview_is_the_start_of_the_full_response():
    shapes, _, _ = _model()
    record = _record(seconds=(mr.PREVIEW_SAMPLES + 100) / FS)
    full = mr.modal_responses(record, shapes)
    preview = mr.preview_responses(record, shapes)
    assert preview.ordinate.shape == (2, 2 ** 16)
    assert np.array_equal(preview.ordinate, full.ordinate[:, :2 ** 16])


def test_two_samples_are_a_record():
    shapes, _, _ = _model()
    q = mr.modal_responses(_record(seconds=2 / FS), shapes)
    assert q.ordinate.shape == (2, 2)


def test_a_set_with_no_mass_unit_gives_undeclared_responses():
    lam, phi = eigh(K, M)
    unscaled = ShapeSet(np.sqrt(lam) / (2 * np.pi), ZETA, ['1X+', '2X+'],
                        phi.T)
    q = mr.modal_responses(_record(seconds=0.5), unscaled)
    assert q.ordinate_dim == ['unknown', 'unknown']
    assert q.dimension_hint == ['modal_acceleration'] * 2


def test_what_is_refused():
    shapes, _, _ = _model()
    record = _record(seconds=0.5)
    with pytest.raises(ValueError, match='not a response'):
        mr.modal_responses(record, shapes, quantity='jerk')
    with pytest.raises(ValueError, match='no mode 3'):
        mr.modal_responses(record, shapes, modes=[2])
    with pytest.raises(ValueError, match='no force channel'):
        mr.modal_responses(record, shapes, records=[1])
    uneven = TimeHistory(np.array([0.0, 0.1, 0.3]), np.ones((1, 3)),
                         response_dof='1X+', ordinate_dim='force')
    with pytest.raises(ValueError, match='not evenly sampled'):
        mr.modal_responses(uneven, shapes)


# ---- appended, replacing an earlier run -------------------------------------

def test_appended_keeps_the_record_and_replaces_an_earlier_run():
    shapes, _, _ = _model()
    record = _record(seconds=1.0)
    record.averaging = Averaging(frame_length=1024).filled(4096, FS)
    first = mr.appended(record, mr.modal_responses(record, shapes,
                                                   label='Modes'), 'Modes')
    assert first.response_dof == ['1X+', '2X+', 'M1', 'M2']
    assert first.averaging == record.averaging
    assert record.num_records == 2, 'the source is not changed'
    other = mr.appended(first, mr.modal_responses(first, shapes, modes=[0],
                                                  label='Other'), 'Other')
    again = mr.appended(other, mr.modal_responses(
        other, shapes, modes=[1], quantity='displacement', label='Modes'),
        'Modes')
    assert again.response_dof == ['1X+', '2X+', 'M1', 'M2']
    assert again.comment[2].startswith('Other: ')
    assert again.comment[3].startswith('Modes: modal displacement')
    assert again.ordinate_dim[3] == 'modal_length'


def test_the_verb_appends_journals_and_is_offered_for_the_pair():
    shapes, _, _ = _model()
    project = visualdynamics.Project()
    project.add('Modes', shapes)
    project.add('Run', _record(seconds=1.0))
    assert 'compute_modal_responses' in dict(
        project.selection_verbs('Run', 'Modes'))
    assert project.compute_modal_responses('Run', 'Modes', modes=[1]) == 'Run'
    assert project['Run'].response_dof == ['1X+', '2X+', 'M2']
    project.compute_modal_responses('Run', 'Modes', quantity='velocity')
    assert project['Run'].response_dof == ['1X+', '2X+', 'M1', 'M2']
    assert "project.compute_modal_responses('Run', 'Modes', modes=[1])" in \
        project.journal
    responses_only = TimeHistory(np.arange(8.0) / FS, np.zeros((1, 8)),
                                 response_dof='1X+',
                                 ordinate_dim='acceleration')
    project.add('Quiet', responses_only)
    assert 'compute_modal_responses' not in dict(
        project.selection_verbs('Quiet', 'Modes')), 'no force to answer'


# ---- the pane ---------------------------------------------------------------

def _populate(window, pump):
    shapes, _, _ = _model()
    window.add_object('Shapes', shapes)
    window.add_object('Run', _record(seconds=1.0))
    pump()


def test_the_pair_offers_the_act_and_the_pane_previews(window, pump):
    _populate(window, pump)
    select_objects(window, pump, 'Run', 'Shapes')
    assert 'Compute Modal Responses' in bar_acts(window.data_pane)
    window.modal_response_act()
    pump()
    panel = window.data_pane.modal_response_panel
    assert panel.isVisible()
    assert panel._derived['modes'][1].text() == '2 of 2'
    assert panel._derived['forces'][1].text() == '1'
    assert 'modal acceleration of 2 modes from 1 force channel' in \
        window.statusBar().currentMessage()
    panel.quantity_box.setCurrentIndex(panel.quantity_box.findData(
        'displacement'))
    pump()
    assert 'modal displacement' in window.statusBar().currentMessage()
    panel.apply_button.click()
    pump()
    run = window.objects['Run']
    assert run.response_dof == ['1X+', '2X+', 'M1', 'M2']
    assert run.ordinate_dim[2] == 'modal_length'
    assert not panel.isVisible(), 'done: the history is shown'
    assert 'appended' in window.statusBar().currentMessage()
    assert any("compute_modal_responses('Run', 'Shapes', "
               "quantity='displacement')" in line
               for line in window.project.journal)


def test_modes_picked_in_the_grid_are_the_modes_answered(window, pump):
    _populate(window, pump)
    window.tree.clearSelection()
    item = window._item_for_object('Shapes')
    item.setExpanded(True)
    pump()
    window.record_grids['Shapes'].select_records([1])
    item.setSelected(True)
    window._item_for_object('Run').setSelected(True)
    window.render_current()
    pump()
    assert 'Compute Modal Responses' in bar_acts(window.data_pane)
    window.modal_response_act()
    pump()
    panel = window.data_pane.modal_response_panel
    assert panel._derived['modes'][1].text() == '1 of 2'
    panel.apply_button.click()
    pump()
    assert window.objects['Run'].response_dof == ['1X+', '2X+', 'M2']


def test_the_pane_stands_down_when_the_selection_moves(window, pump):
    _populate(window, pump)
    select_objects(window, pump, 'Run', 'Shapes')
    window.modal_response_act()
    pump()
    panel = window.data_pane.modal_response_panel
    assert panel.isVisible()
    select_objects(window, pump, 'Run')
    assert not panel.isVisible() and window._modaling is None


# ---- expanded where they were appended --------------------------------------

def _forces_and_responses():
    shapes, phi, _ = _model()
    t = np.arange(4096) / FS
    force = np.random.default_rng(3).standard_normal(t.size)
    run = TimeHistory(t, force[None, :], response_dof='1X+',
                      ordinate_dim='force')
    run = mr.appended(run, mr.modal_responses(run, shapes, label='Modes'),
                      'Modes')
    return shapes, phi, run


def test_the_appended_responses_expand_to_the_physical_ones():
    """The history reads physical as a whole (its force is at 1X+); what
    there is to expand is its modal records, and they expand to Φq."""
    shapes, phi, run = _forces_and_responses()
    assert reads_as(run, shapes) == 'physical'
    assert reads_as(run, shapes, [1, 2]) == 'modal'
    assert modal_records(run, shapes) == [1, 2]
    assert expands(run, shapes)
    physical, report = to_physical(run, shapes)
    assert physical.response_dof == ['1X+', '2X+']
    assert physical.ordinate_dim == ['acceleration', 'acceleration']
    assert np.allclose(physical.ordinate, phi @ run.ordinate[1:],
                       rtol=0, atol=1e-12)
    assert report.shared == {'acceleration': ['M1', 'M2']}
    assert report.skipped == {}, 'the force is not an input, not skipped'
    one, _report = to_physical(run, shapes, records=[2])
    assert np.allclose(one.ordinate, np.outer(phi[:, 1], run.ordinate[2]))
    with pytest.raises(ValueError, match='records picked are not all'):
        to_physical(run, shapes, records=[0, 2])
    forces_only = TimeHistory(run.abscissa, run.ordinate[:1],
                              response_dof='1X+', ordinate_dim='force')
    assert not expands(forces_only, shapes)


def test_the_verbs_offered_for_forces_with_responses():
    shapes, phi, run = _forces_and_responses()
    project = visualdynamics.Project()
    project.add('Modes', shapes)
    project.add('Run', run)
    assert [v for v, _ in project.selection_verbs('Run', 'Modes')] == [
        'compute_modal_responses', 'transform', 'expand']
    made = project.expand('Run', 'Modes')
    assert made == 'Run Physical Responses'
    assert np.allclose(project[made].ordinate, phi @ run.ordinate[1:])
    assert project.journal[-1] == "project.expand('Run', 'Modes')"
    assert project.transform('Run', 'Modes') == 'Run Modal Coordinates'
    assert project['Run Modal Coordinates'].ordinate_dim == ['modal_force'] * 2


def test_the_bar_expands_or_transforms_as_asked(window, pump):
    shapes, phi, run = _forces_and_responses()
    window.add_object('Shapes', shapes)
    window.add_object('Run', run)
    pump()
    select_objects(window, pump, 'Run', 'Shapes')
    acts = bar_acts(window.data_pane)
    assert 'Expand to Physical Responses' in acts
    assert 'Transform to Modal Coordinates' in acts
    window.expand_act()
    pump()
    made = window.project['Run Physical Responses']
    assert np.allclose(made.ordinate, phi @ run.ordinate[1:])
    select_objects(window, pump, 'Run', 'Shapes')
    window.transform_act()
    pump()
    assert window.project['Run Modal Coordinates'].ordinate_dim == \
        ['modal_force'] * 2


def test_modal_records_picked_in_the_history_expand(window, pump):
    shapes, phi, run = _forces_and_responses()
    window.add_object('Shapes', shapes)
    window.add_object('Run', run)
    pump()
    window.tree.clearSelection()
    window._item_for_object('Shapes').setSelected(True)
    item = window._item_for_object('Run')
    item.setExpanded(True)
    pump()
    grid = window.record_grids['Run']
    for cell, record in grid._records.items():
        if record == 2:
            grid.item(*cell).setSelected(True)
    pump()
    assert ('record', 2) in [(kind, detail) for kind, _name, _obj, detail
                             in window.selected_references()]
    window.transform_selection()
    pump()
    made = window.project['Run Physical Responses (M2)']
    assert np.allclose(made.ordinate, np.outer(phi[:, 1], run.ordinate[2]))
