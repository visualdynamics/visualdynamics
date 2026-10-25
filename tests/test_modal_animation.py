"""Modal coordinates played on a geometry through their shape set.

Brandon, 2026-10-25: a modal response selected with a geometry should
animate without forming the physical response at every DOF, which for a
large finite element model is far larger than the modal one. Each frame
is the modes' shapes scaled by their values at that sample and summed.
Checked against the same record expanded and played the ordinary way.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import plate_geometry_and_shapes, select_objects

import visualdynamics
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.shapes import ShapeSet
from visualdynamics.core.transform import modal_dofs
from visualdynamics.core.truncate import Truncation
from visualdynamics.deform import (
    ModalTimeDeflection,
    TimeDeflection,
    modal_deflection,
)

FS = 1024.0


def _modal_record(shapes, samples=300, seed=0, extra=()):
    """Every mode's coordinate as a record, with `extra` (dof, dim)
    records ahead of them."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((shapes.num_shapes, samples))
    rows = [rng.standard_normal(samples) for _ in extra] + list(q)
    return TimeHistory(np.arange(samples) / FS, np.asarray(rows),
                       response_dof=[d for d, _ in extra] + modal_dofs(shapes),
                       ordinate_dim=[k for _, k in extra]
                       + ['modal_acceleration'] * shapes.num_shapes), q


def _expanded(geometry, shapes, q):
    return TimeDeflection(geometry, shapes.coordinate,
                          np.real(shapes.shape_matrix).T @ q)


# ---- the deflection ---------------------------------------------------------

def test_every_frame_is_the_expanded_response():
    geometry, shapes = plate_geometry_and_shapes()
    _record, q = _modal_record(shapes)
    modal = ModalTimeDeflection(geometry, shapes.coordinate,
                                shapes.shape_matrix, q)
    physical = _expanded(geometry, shapes, q)
    assert np.array_equal(modal.rows, physical.rows)
    assert modal.num_frames == physical.num_frames == 300
    for k in (0, 17, 299):
        assert np.allclose(modal.offsets(k).copy(), physical.offsets(k),
                           rtol=0, atol=1e-12)
    assert modal.peak == pytest.approx(physical.peak, rel=1e-12)
    assert modal.peak_magnitude == pytest.approx(physical.peak_magnitude,
                                                 rel=1e-12)


def test_past_the_budget_the_loudest_frames_find_the_peak():
    """A frame where every mode is moderate but all push one node the
    same way travels further than any mode's own largest sample: the
    bound finds it, where each mode's largest sample alone would not."""
    geometry, shapes = plate_geometry_and_shapes()
    rng = np.random.default_rng(4)
    q = 0.05 * rng.standard_normal((shapes.num_shapes, 2000))
    for r in range(shapes.num_shapes):
        q[r, 100 + 50 * r] = 1.0           # each mode's own largest
    probe = ModalTimeDeflection(geometry, shapes.coordinate,
                                shapes.shape_matrix, q)
    column = probe._fields[:, 3 * 40 + 2]  # one node's Z, every mode
    q[:, 1777] = 0.6 * np.sign(column)
    modal = ModalTimeDeflection(geometry, shapes.coordinate,
                                shapes.shape_matrix, q)
    modal.EXACT_BUDGET = 0                 # the estimate, not the sweep
    physical = _expanded(geometry, shapes, q)
    travel = [np.linalg.norm(physical.offsets(k), axis=1).max()
              for k in range(q.shape[1])]
    assert int(np.argmax(travel)) == 1777, 'the test holds its premise'
    assert modal.peak_magnitude == pytest.approx(physical.peak_magnitude,
                                                 rel=1e-12)
    assert modal.peak == pytest.approx(physical.peak, rel=1e-12)


def test_the_records_played_are_the_sets_modal_coordinates():
    geometry, shapes = plate_geometry_and_shapes()
    record, q = _modal_record(shapes, extra=[('101Z+', 'force'),
                                             ('M30', 'modal_acceleration')])
    played = modal_deflection(geometry, shapes, record,
                              range(record.num_records))
    assert np.allclose(played.offsets(5).copy(),
                       _expanded(geometry, shapes, q).offsets(5))
    one = modal_deflection(geometry, shapes, record, [2 + 7])
    alone = TimeDeflection(geometry, shapes.coordinate,
                           np.outer(np.real(shapes.shape_matrix[7]), q[7]))
    assert np.allclose(one.offsets(5).copy(), alone.offsets(5)), \
        "a pick of one mode is that mode's contribution"
    with pytest.raises(ValueError, match='no record is a modal coordinate'):
        modal_deflection(geometry, shapes, record, [0, 1])
    complex_set = ShapeSet(shapes.frequency, shapes.damping,
                           shapes.coordinate, shapes.shape_matrix * 1j)
    with pytest.raises(ValueError, match='complex'):
        modal_deflection(geometry, complex_set, record, [2])


# ---- which set ---------------------------------------------------------------

def _plate_project():
    geometry, shapes = plate_geometry_and_shapes()
    project = visualdynamics.Project()
    project.add('Plate', geometry)
    project.add('Modes', shapes)
    project.link('Plate', 'Modes')
    t = np.arange(2048) / FS
    force = np.random.default_rng(2).standard_normal(t.size)
    project.add('Run', TimeHistory(t, force[None, :], response_dof='101Z+',
                                   ordinate_dim='force'))
    return project, geometry, shapes


def test_the_set_behind_computed_responses_is_read_from_them():
    project, _geometry, shapes = _plate_project()
    project.compute_modal_responses('Run', 'Modes', modes=range(6, 22))
    assert project.modal_basis('Run') == 'Modes'
    assert project.modal_basis('Run', records=[0]) is None, 'the force'
    project.add('Other', shapes)
    project.compute_modal_responses('Run', 'Other', modes=[6])
    assert project.modal_basis('Run') is None, 'two sets: none says'
    assert project.modal_basis('Run', records=[17]) == 'Other'
    del project['Other']
    assert project.modal_basis('Run', records=[17]) is None, 'set gone'


def test_the_set_behind_a_transform_is_read_from_its_provenance():
    project, _geometry, shapes = _plate_project()
    q = np.random.default_rng(5).standard_normal((shapes.num_shapes, 512))
    translational = [i for i, dof in enumerate(shapes.coordinate)
                     if 'R' not in dof]
    motion = np.real(shapes.shape_matrix).T[translational] @ q
    project.add('Motion', TimeHistory(
        np.arange(512) / FS, motion,
        response_dof=[shapes.coordinate[i] for i in translational],
        ordinate_dim='acceleration'))
    made = project.transform('Motion', 'Modes')
    assert project.modal_basis(made) == 'Modes'
    project[made].truncation = Truncation(0.1, 0.3)
    cut = project.truncate_data(made)
    assert project.modal_basis(cut) == 'Modes', \
        'a record derived from one keeps its set'
    assert project.modal_basis('Motion') is None


def test_a_script_plays_them_on_the_sets_geometry(tmp_path):
    project, _geometry, _shapes = _plate_project()
    project.compute_modal_responses('Run', 'Modes', modes=range(6, 22))
    path = tmp_path / 'modal.png'
    project.animate('Run', screenshot=str(path))
    assert path.stat().st_size > 1000


# ---- the window --------------------------------------------------------------

def test_appended_responses_add_no_capture_labels():
    """A history with no capture labels keeps none: blank ones read as a
    capture named '' and the playback said 'animating  of 1 averages'."""
    project, _geometry, _shapes = _plate_project()
    project.compute_modal_responses('Run', 'Modes', modes=[6, 7])
    assert project['Run'].block is None


def test_a_history_and_a_geometry_play_its_modes(window, pump):
    project, geometry, shapes = _plate_project()
    project.compute_modal_responses('Run', 'Modes', modes=range(6, 22))
    for name in ('Plate', 'Modes', 'Run'):
        window.add_object(name, project[name])
    pump()
    select_objects(window, pump, 'Run', 'Plate')
    deflection = window.animator.deflection
    assert isinstance(deflection, ModalTimeDeflection)
    run = window.objects['Run']
    q = np.zeros((shapes.num_shapes, run.ordinate.shape[1]))
    q[6:] = run.ordinate[1:]
    assert np.allclose(deflection.offsets(300).copy(),
                       _expanded(geometry, shapes, q).offsets(300))
    said = window.statusBar().currentMessage()
    assert '16 mode shapes scaled by' in said
    assert 'averages' not in said
    assert window.dofs_combo.currentData() == 'modal_acceleration', \
        'the box opens on the most common quantity, not the first by name'
    assert window.dofs_combo.currentText() == 'Modal Acceleration'


def test_with_no_set_behind_them_the_set_is_selected_too(window, pump):
    geometry, shapes = plate_geometry_and_shapes()
    record, q = _modal_record(shapes)
    window.add_object('Plate', geometry)
    window.add_object('Modes', shapes)
    window.add_object('Coordinates', record)
    pump()
    select_objects(window, pump, 'Coordinates', 'Plate')
    assert 'select their shape set too' in window.statusBar().currentMessage()
    select_objects(window, pump, 'Coordinates', 'Plate', 'Modes')
    deflection = window.animator.deflection
    assert isinstance(deflection, ModalTimeDeflection)
    assert np.allclose(deflection.offsets(9).copy(),
                       _expanded(geometry, shapes, q).offsets(9))


def test_modes_picked_in_the_history_play_their_contribution(window, pump):
    project, geometry, shapes = _plate_project()
    project.compute_modal_responses('Run', 'Modes', modes=range(6, 22))
    for name in ('Plate', 'Modes', 'Run'):
        window.add_object(name, project[name])
    pump()
    select_objects(window, pump, 'Plate')
    item = window._item_for_object('Run')
    item.setExpanded(True)
    pump()
    grid = window.record_grids['Run']
    for cell, record in grid._records.items():
        if record in (1, 3):                    # M7 and M9
            grid.item(*cell).setSelected(True)
    pump()
    deflection = window.animator.deflection
    run = window.objects['Run']
    q = np.zeros((shapes.num_shapes, run.ordinate.shape[1]))
    q[6], q[8] = run.ordinate[1], run.ordinate[3]
    assert np.allclose(deflection.offsets(200).copy(),
                       _expanded(geometry, shapes, q).offsets(200))
    assert '2 mode shapes scaled by' in window.statusBar().currentMessage()


def test_an_frf_beside_its_set_and_geometry_is_not_animated(window, pump):
    """The modal route is for a time history of modal coordinates: an FRF
    selected with its set and a geometry kept the static view and the
    synthesis overlay until that route took it into an operating
    deflection shape (2026-10-25)."""
    from conftest import fixture_path

    window.add_object('Plate', visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz'), length_unit='m'))
    window.add_object('Modes', visualdynamics.import_file(
        fixture_path('plate', 'shapes.npy')))
    window.add_object('FRF', visualdynamics.import_file(
        fixture_path('plate', 'frfs.npz')))
    pump()
    select_objects(window, pump, 'FRF', 'Modes', 'Plate')
    assert window.animator is None
    assert 'synthesized from 22 modes of Modes' in \
        window.statusBar().currentMessage()
