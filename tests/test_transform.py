"""Physical responses through a shape set to modal responses, and back.

The rules pinned here are the ones PLAN.md's phase 2 states: records
match columns by (DOF, quantity) with the sign honored; motions fit
(`Φ⁺u`) and forces project (`Φᵀf`); everything else is left out and
said; the unit rule `[q] = [u]/[Φ]` gives a virtual point's rotations
in rad/s² and lbf·in through unit rigid shapes, half-power mass units
through mass-normalized ones, and no unit through a set that has none;
a rank-deficient measurement is refused by name; and the expansion is
the exact inverse. The units module's new tags are pinned beside.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.rigid import MassProperties, rigid_body_shapes
from visualdynamics.core.transform import (
    modal_dofs,
    reads_as,
    to_modal,
    to_physical,
)
from visualdynamics.core.validate import modal_coordinate
from visualdynamics.units import SYSTEMS, dimension_of, plain_unit, pretty_unit

T = np.arange(1024) / 1024.0


def _geometry(seed=1, n=5):
    rng = np.random.default_rng(seed)
    return visualdynamics.Geometry(np.arange(101, 101 + n),
                                   rng.uniform(-1.0, 1.0, (n, 3)),
                                   length_unit='m')


def _rigid_motion():
    """Six modal coordinates, two of them alive."""
    q = np.zeros((6, len(T)))
    q[0] = np.sin(2 * np.pi * 5 * T)
    q[5] = 0.3 * np.cos(2 * np.pi * 3 * T)
    return q


def _history(shapes, q, dims='acceleration', dofs=None):
    u = np.real(shapes.shape_matrix).T @ q
    return visualdynamics.TimeHistory(T, u, response_dof=dofs or list(
        shapes.coordinate), ordinate_dim=dims)


# ---- through a unit rigid set: the virtual point ---------------------------


def test_rigid_motion_comes_back_as_the_six_coordinates():
    rigid = rigid_body_shapes(_geometry(), MassProperties((0.1, 0.0, -0.2)))
    q = _rigid_motion()
    modal, report = to_modal(_history(rigid, q), rigid)
    assert modal.response_dof == ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'], \
        'a modal coordinate is the letter and the mode index, every set'
    assert modal.ordinate_dim == ['acceleration'] * 3 \
        + ['angular_acceleration'] * 3, \
        'translations keep the quantity; rotations are per radian'
    assert np.allclose(modal.ordinate, q, atol=1e-10)
    assert report.worst_residual == pytest.approx(0.0, abs=1e-9)
    assert report.rank['acceleration'] == (6, 6)
    assert modal.comment[5].startswith('acceleration of mode 6: Rigid')
    assert modal.units_defined


def test_a_negated_channel_is_read_with_its_sign():
    rigid = rigid_body_shapes(_geometry(), MassProperties((0, 0, 0)))
    q = _rigid_motion()
    history = _history(rigid, q)
    # a channel the motion actually moves — 101X+ under the X
    # translation — so an ignored sign cannot hide in a zero row
    assert history.response_dof[0] == '101X+'
    assert np.abs(history.ordinate[0]).max() > 0.5
    history.ordinate[0] *= -1.0
    history.response_dof[0] = '101X-'
    modal, report = to_modal(history, rigid)
    assert np.allclose(modal.ordinate, q, atol=1e-10)
    assert report.worst_residual == pytest.approx(0.0, abs=1e-9)


def test_forces_project_to_the_points_force_and_moment():
    """`Φᵀf`: a force at one node becomes the same force at the point
    plus the moment of its offset, in lbf·in-type units."""
    geometry = _geometry()
    point = np.array([0.1, 0.0, -0.2])
    rigid = rigid_body_shapes(geometry, MassProperties(tuple(point)))
    force = np.vstack([np.ones(len(T)), 2 * np.ones(len(T))])
    history = visualdynamics.TimeHistory(
        T, force, response_dof=['103Z+', '104X+'], ordinate_dim='force')
    modal, report = to_modal(history, rigid)
    assert modal.ordinate_dim == ['force'] * 3 + ['moment'] * 3
    assert 'force' in report.shared and report.residual == {}, \
        'projection, not a fit: nothing is unexplained by definition'
    r3 = geometry.node_xyz[2] - point
    r4 = geometry.node_xyz[3] - point
    expected = (np.cross(r3, [0, 0, 1.0]) * 1.0
                + np.cross(r4, [1.0, 0, 0]) * 2.0)
    assert np.allclose(modal.ordinate[:3, 0], [2.0, 0.0, 1.0])
    assert np.allclose(modal.ordinate[3:, 0], expected)


def test_other_quantities_are_left_out_and_said():
    rigid = rigid_body_shapes(_geometry(), MassProperties((0, 0, 0)))
    history = _history(rigid, _rigid_motion())
    extra = visualdynamics.TimeHistory(
        T, np.vstack([history.ordinate, 20 * np.ones((2, len(T))),
                      np.ones((1, len(T)))]),
        response_dof=[*history.response_dof, '101X+', '102X+', '9001X+'],
        ordinate_dim=[*history.ordinate_dim, 'temperature', 'temperature',
                      'force'])
    modal, report = to_modal(extra, rigid)
    assert report.skipped == {'temperature': 2}
    assert report.dropped == ['9001X+'], \
        'a force at a synthetic drive DOF has no shape row'
    assert 'not transformed: 2 temperature' in report.describe()
    assert '1 not in the shapes' in report.describe()
    assert not any(dim == 'temperature' for dim in modal.ordinate_dim)


def test_too_few_directions_are_refused_by_name():
    """Z channels alone cannot resolve X, Y or the yaw."""
    rigid = rigid_body_shapes(_geometry(), MassProperties((0, 0, 0)))
    zs = [i for i, dof in enumerate(rigid.coordinate) if dof.endswith('Z+')]
    history = visualdynamics.TimeHistory(
        T, (rigid.shape_matrix.T @ _rigid_motion())[zs],
        response_dof=[rigid.coordinate[i] for i in zs],
        ordinate_dim='acceleration')
    with pytest.raises(ValueError, match='resolve only 3 of 6 modes'):
        to_modal(history, rigid)


def test_a_flexible_motion_leaves_a_residual():
    rigid = rigid_body_shapes(_geometry(), MassProperties((0, 0, 0)))
    history = _history(rigid, _rigid_motion())
    history.ordinate[0] += 0.5 * np.sin(2 * np.pi * 40 * T)
    _modal, report = to_modal(history, rigid)
    assert report.worst_residual > 0.05
    assert 'residual' in report.describe()


def test_undefined_units_are_not_transformed():
    rigid = rigid_body_shapes(_geometry(), MassProperties((0, 0, 0)))
    history = _history(rigid, _rigid_motion())
    history.undefine_units()
    with pytest.raises(ValueError, match='nothing to transform'):
        to_modal(history, rigid)


def test_a_complex_set_is_refused():
    rigid = rigid_body_shapes(_geometry(), MassProperties((0, 0, 0)))
    complex_set = visualdynamics.ShapeSet(
        rigid.frequency, rigid.damping, rigid.coordinate,
        rigid.shape_matrix * (1 + 0.1j))
    with pytest.raises(ValueError, match='complex'):
        to_modal(_history(rigid, _rigid_motion()), complex_set)


# ---- the unit rule ----------------------------------------------------------


def test_mass_normalized_shapes_give_half_power_mass_units():
    geometry = _geometry()
    unit = rigid_body_shapes(geometry, MassProperties((0, 0, 0)))
    scaled = rigid_body_shapes(geometry, MassProperties(
        (0, 0, 0), mass=4.0, inertia=(1.0, 2.0, 3.0, 0, 0, 0)))
    q = _rigid_motion()
    history = _history(unit, q)
    modal, _report = to_modal(history, scaled)
    assert modal.ordinate_dim == ['modal_acceleration'] * 6
    assert modal.ordinate_unit[0] == 'm/s**2*kg**0.5'
    assert np.allclose(modal.ordinate[0], q[0] * math.sqrt(4.0)), \
        'the translation coordinate is scaled by √mass'
    assert np.allclose(modal.ordinate[5], q[5] * math.sqrt(3.0))
    back, _report = to_physical(modal, scaled)
    assert np.allclose(back.ordinate, history.ordinate)
    assert back.ordinate_dim[0] == 'acceleration'


def test_a_set_with_no_mass_unit_gives_hinted_responses():
    """The shapes are numbers with no unit, so the response has none
    — badged, hinted with the modal tag, Define Units to declare."""
    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    assert shapes.mass_unit is None
    q = np.zeros((shapes.num_shapes, len(T)))
    q[0] = np.sin(2 * np.pi * 5 * T)
    q[3] = np.cos(2 * np.pi * 9 * T)
    history = _history(shapes, q)
    modal, _report = to_modal(history, shapes)
    assert modal.response_dof[:3] == ['M1', 'M2', 'M3']
    assert modal.response_dof[-1] == f'M{shapes.num_shapes}'
    assert not modal.units_defined
    assert set(modal.dimension_hint) == {'modal_acceleration'}
    assert np.allclose(modal.ordinate, q, atol=1e-8)
    modal.define_units('in/s**2*slinch**0.5')
    assert modal.ordinate_dim[0] == 'modal_acceleration'


def test_the_units_module_knows_the_new_quantities():
    coherent = SYSTEMS['in-slinch-lbf-s']
    assert coherent.unit('angular_acceleration') == 'rad/s**2'
    assert coherent.unit('moment') == 'lbf*in'
    assert coherent.unit('modal_acceleration') == 'in/s**2*slinch**0.5'
    assert coherent.label_text('modal_acceleration') == 'in/s²·slinch½'
    assert coherent.label_ascii('modal_acceleration') == 'in/s^2*slinch^0.5'
    assert coherent.factor('modal_acceleration') == pytest.approx(
        coherent.factor('acceleration') * math.sqrt(coherent.factor('mass')))
    assert SYSTEMS['m-kg-N-s'].unit('modal_force') == 'N*kg**0.5'
    assert SYSTEMS['in-slinch-lbf-s (g)'].unit('acceleration') == 'g'
    assert SYSTEMS['in-slinch-lbf-s (g)'].unit('modal_acceleration') == \
        'in/s**2*slinch**0.5', 'the modal unit stays coherent'
    assert dimension_of('rad/s') == 'angular_velocity'
    assert dimension_of('Hz') == 'frequency'
    assert dimension_of('deg/s**2') == 'angular_acceleration'
    assert dimension_of('lbf*in') == 'moment'
    assert dimension_of('rad') == 'angle'
    assert dimension_of('m/s**2*kg**0.5') == 'modal_acceleration'
    assert plain_unit(pretty_unit('m/s**2*kg**0.5')) == 'm/s**2*kg**0.5'


# ---- naming and direction ---------------------------------------------------


def test_modal_coordinates_are_spelled_m_and_the_index():
    rigid = rigid_body_shapes(_geometry(), MassProperties((0, 0, 0)))
    assert modal_dofs(rigid) == ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    assert modal_coordinate('M4') == 4
    assert modal_coordinate('m12') == 12, 'normalized like any DOF'
    assert modal_coordinate('101X+') is None
    assert modal_coordinate('M') is None and modal_coordinate('M0') is None
    assert modal_coordinate('9001') is None, 'a bare node id is not modal'
    from visualdynamics.core.validate import dofs

    assert dofs(['m3', '101z']) == ['M3', '101Z+']
    with pytest.raises(ValueError, match='node id'):
        dofs(['M3X+'])          # a modal coordinate takes no direction


def test_reads_as_says_which_way_a_transform_goes():
    rigid = rigid_body_shapes(_geometry(), MassProperties((0, 0, 0)))
    history = _history(rigid, _rigid_motion())
    modal, _report = to_modal(history, rigid)
    assert reads_as(history, rigid) == 'physical'
    assert reads_as(modal, rigid) == 'modal'
    other = visualdynamics.TimeHistory(T, np.zeros((1, len(T))),
                                       response_dof='7001Z+',
                                       ordinate_dim='acceleration')
    assert reads_as(other, rigid) is None
    beyond = visualdynamics.TimeHistory(T, np.zeros((1, len(T))),
                                        response_dof='M7',
                                        ordinate_dim='acceleration')
    assert reads_as(beyond, rigid) is None, \
        'a seventh coordinate is not one of six modes'
    first = visualdynamics.TimeHistory(T, np.zeros((1, len(T))),
                                       response_dof='M1',
                                       ordinate_dim='acceleration')
    assert reads_as(first, rigid) == 'modal'


# ---- the expansion ----------------------------------------------------------


def test_expansion_is_the_exact_inverse_and_skips_forces():
    rigid = rigid_body_shapes(_geometry(), MassProperties((0.1, 0, 0)))
    q = _rigid_motion()
    history = _history(rigid, q)
    modal, _report = to_modal(history, rigid)
    forces = visualdynamics.TimeHistory(
        T, np.vstack([modal.ordinate, np.ones((1, len(T)))]),
        response_dof=[*modal.response_dof, '9001X+'],
        ordinate_dim=[*modal.ordinate_dim, 'force'])
    back, report = to_physical(forces, rigid)
    assert back.response_dof == list(rigid.coordinate)
    assert set(back.ordinate_dim) == {'acceleration'}
    assert np.allclose(back.ordinate, history.ordinate, atol=1e-10)
    assert report.skipped == {'force': 1}


def test_a_pick_of_modes_expands_their_contribution():
    """One modal coordinate expanded is that mode's contribution to the
    physical motion, `u = φₖ qₖ` — a modal contribution plot — so a
    partial set expands rather than being refused, and says which
    modes it carries."""
    rigid = rigid_body_shapes(_geometry(), MassProperties((0, 0, 0)))
    q = _rigid_motion()
    modal, _report = to_modal(_history(rigid, q), rigid)
    one, report = to_physical(modal, rigid, records=[5])
    assert report.shared == {'acceleration': ['M6']}
    expected = np.outer(rigid.shape_matrix[5], q[5])
    assert np.allclose(one.ordinate, expected, atol=1e-10)
    assert one.comment[0] == 'acceleration from modes M6'
    assert one.response_dof == list(rigid.coordinate)
    # the same by leaving records out of the object
    four = visualdynamics.TimeHistory(
        T, modal.ordinate[:4], response_dof=modal.response_dof[:4],
        ordinate_dim=modal.ordinate_dim[:4])
    partial, report = to_physical(four, rigid)
    assert report.shared == {'acceleration': ['M1', 'M2', 'M3', 'M4']}
    assert partial.comment[0] == 'acceleration from modes M1, M2, M3, M4'
    assert np.allclose(partial.ordinate,
                       rigid.shape_matrix[:4].T @ q[:4], atol=1e-10)
    whole, _report = to_physical(modal, rigid)
    assert whole.comment[0] == 'acceleration from all modes'
    with pytest.raises(ValueError, match='not modal coordinates'):
        to_physical(_history(rigid, _rigid_motion()), rigid)
    with pytest.raises(ValueError, match='no record 9'):
        to_physical(modal, rigid, records=[9])
    # a coordinate past the set's last mode is not one of its modes
    beyond = visualdynamics.TimeHistory(
        T, np.vstack([modal.ordinate, modal.ordinate[:1]]),
        response_dof=[*modal.response_dof, 'M7'],
        ordinate_dim=[*modal.ordinate_dim, 'acceleration'])
    back, report = to_physical(beyond, rigid)
    assert report.dropped == ['M7']
    assert back.num_records == rigid.num_dofs


def test_a_pick_of_channels_transforms_those_alone():
    rigid = rigid_body_shapes(_geometry(), MassProperties((0, 0, 0)))
    history = _history(rigid, _rigid_motion())
    # every channel but one: still six independent directions
    picked = list(range(1, history.num_records))
    modal, report = to_modal(history, rigid, records=picked)
    assert len(report.shared['acceleration']) == history.num_records - 1
    assert history.response_dof[0] not in report.shared['acceleration']
    assert np.allclose(modal.ordinate, _rigid_motion(), atol=1e-10)


# ---- through the project ---------------------------------------------------


def _project():
    geometry = _geometry()
    rigid = rigid_body_shapes(geometry, MassProperties((0, 0, 0)))
    project = visualdynamics.Project('T')
    project.add('Plate', geometry)
    project.add('Modes', rigid)
    project.link('Plate', 'Modes')
    project.add('Run', _history(rigid, _rigid_motion()))
    project.link('Plate', 'Run')
    return project


def test_the_verbs_name_link_and_journal():
    project = _project()
    name = project.transform('Run', 'Modes')
    assert name == 'Run Modal Responses'
    assert project.provenance[name] == {
        'verb': 'transform', 'source': 'Run',
        'params': {'shapes': 'Modes'},
        'state': project.provenance[name]['state']}
    assert name in project.links[0]['members'], \
        'the modal responses belong with the shapes and the record'
    assert project[name].transform_report.worst_residual == pytest.approx(0.0)
    assert project.journal[-1] == "project.transform('Run', 'Modes')"
    assert 'transform' in dict(project.verbs('Run'))
    assert 'expand' not in dict(project.verbs('Run'))
    assert 'expand' in dict(project.verbs(name))
    assert 'transform' not in dict(project.verbs(name))

    back = project.expand(name, 'Modes')
    assert back == 'Run Physical Responses'
    assert back in project.links[0]['members'], \
        'the expansion lands back in the geometry\'s group'
    assert np.allclose(project[back].ordinate, project['Run'].ordinate)
    assert project.journal[-1] == \
        "project.expand('Run Modal Responses', 'Modes')"
    with pytest.raises(TypeError, match='not a shape set'):
        project.transform('Run', 'Plate')


def test_a_changed_set_or_record_badges_the_modal_responses():
    project = _project()
    name = project.transform('Run', 'Modes')
    assert project[name].response_dof[0] == 'M1'
    assert project.stale() == {}
    project['Modes'].shape_matrix[0, 0] *= 1.01
    assert project.stale() == {name: 'its source was recomputed'}
    project.refresh(name)
    assert project.stale() == {}
    project['Run'].ordinate[0] *= 2.0
    assert name in project.stale()


def test_the_session_script_replays(tmp_path):
    rigid = rigid_body_shapes(_geometry(), MassProperties((0, 0, 0)))
    saved = tmp_path / 'run.vdyn'
    _history(rigid, _rigid_motion()).save(saved)

    project = visualdynamics.Project('Replay')
    project.import_file(fixture_path('plate', 'geometry.unv'))
    # the rigid set of this five-node cloud, not the plate's: the
    # record was made from it, and a script can rebuild it exactly
    cloud = tmp_path / 'cloud.vdyn'
    _geometry().save(cloud)
    project.import_file(cloud)
    geometry = next(n for n in project.names
                    if isinstance(project[n], visualdynamics.Geometry)
                    and project[n].num_nodes == 5)
    modes = project.generate_rigid_body_modes(geometry)
    project.import_file(saved)
    run = next(n for n in project.names
               if isinstance(project[n], visualdynamics.TimeHistory))
    modal = project.transform(run, modes)
    back = project.expand(modal, modes)
    script = project.session_script()
    room: dict = {}
    exec(script, room)                                  # noqa: S102
    replayed = room['project']
    assert replayed[modal] == project[modal]
    assert replayed[back] == project[back]
    assert replayed[modal].response_dof[0] == 'M1'


# ---- what a modal object can and cannot be written as --------------------


def test_node_numbered_formats_refuse_modal_coordinates(tmp_path):
    """A universal file writes a DOF as a node number and a direction
    code; a modal coordinate has neither, and a made-up node would be
    a lie in a file another tool trusts."""
    rigid = rigid_body_shapes(_geometry(), MassProperties((0, 0, 0)))
    modal, _report = to_modal(_history(rigid, _rigid_motion()), rigid)
    for suffix in ('.unv', '.npz', '.ati'):
        with pytest.raises(ValueError, match='modal coordinates'):
            visualdynamics.export_file(modal, str(tmp_path / f'm{suffix}'))
    # the native file keeps them, and the physical expansion writes
    back, _report = to_physical(modal, rigid)
    visualdynamics.export_file(back, str(tmp_path / 'back.unv'))
    modal.save(tmp_path / 'modal.vdyn')
    assert visualdynamics.import_file(
        tmp_path / 'modal.vdyn').response_dof == modal.response_dof


# ---- the modal object belongs with its set --------------------------------------


def test_a_modal_object_answers_to_the_shape_set_it_came_through():
    """Its DOFs are on no geometry, so the link rule judges it against
    the group's shape set instead: it fits where a set has every mode
    it names (Brandon, 2026-09-04)."""
    from visualdynamics.compatibility import check_compatibility

    project = _project()
    name = project.transform('Run', 'Modes')
    report = check_compatibility(dict(project.items()), 'Plate', project.links)
    assert report.is_compatible(name)
    assert project.links[0]['members'] == ['Plate', 'Modes', 'Run', name]

    # nine coordinates against a six-mode set: refused by name, and
    # the report marks the coordinates past the set
    big = visualdynamics.TimeHistory(
        T, np.zeros((9, len(T))), response_dof=[f'M{k}' for k in range(1, 10)],
        ordinate_dim='acceleration')
    project.add('Big', big)
    with pytest.raises(ValueError, match="'Modes' has 6 modes"):
        project.link('Modes', 'Big')
    report = check_compatibility(dict(project.items()), 'Plate', project.links)
    issue = report.issue_for('Big')
    assert issue is not None and issue.kind == 'modes-not-in-set'
    assert issue.sub_items == [6, 7, 8]
    assert issue.missing_dofs == ['M7', 'M8', 'M9']
    # unlinked, it is judged against every set in the project
    project.remove('Big')
    loose = visualdynamics.TimeHistory(
        T, np.zeros((2, len(T))), response_dof=['M1', 'M2'],
        ordinate_dim='acceleration')
    project.add('Loose', loose)
    report = check_compatibility(dict(project.items()), 'Plate', project.links)
    assert report.is_compatible('Loose')
    # and with no set anywhere, it says what to link
    alone = visualdynamics.Project('alone')
    alone.add('Loose', loose)
    report = check_compatibility(dict(alone.items()), None, alone.links)
    assert 'no shape set to answer to' in report.issue_for('Loose').message


def test_the_marks_travel_through_the_transform_and_back():
    """A transform combines samples and moves none, so the averaging
    frames and shock windows land on the same instants — carried both
    ways, the filter's own rule (Brandon, 2026-09-04)."""
    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.shocks import Shock

    rigid = rigid_body_shapes(_geometry(), MassProperties((0, 0, 0)))
    history = _history(rigid, _rigid_motion())
    history.averaging = Averaging(frame_length=256, frames=3)
    history.shocks = (Shock(0.25, 0.1),)
    modal, _report = to_modal(history, rigid)
    assert modal.averaging == history.averaging
    assert modal.shocks == history.shocks
    back, _report = to_physical(modal, rigid)
    assert back.averaging == history.averaging
    assert back.shocks == history.shocks
    assert back.filtering is None, 'a filter design is not a mark'


# ---- every kind of data, strictly ------------------------------------------------


def _rigid_and_run(seed=1, samples=8192):
    rng = np.random.default_rng(seed)
    geometry = visualdynamics.Geometry(np.arange(101, 106),
                                       rng.uniform(-1.0, 1.0, (5, 3)),
                                       length_unit='m')
    rigid = rigid_body_shapes(geometry, MassProperties((0, 0, 0)))
    t = np.arange(samples) / 2048.0
    q = rng.standard_normal((6, samples)) * np.array(
        [1.0, 0.5, 0.2, 0.1, 0.3, 0.4])[:, None]
    run = visualdynamics.TimeHistory(t, rigid.shape_matrix.T @ q,
                                     response_dof=list(rigid.coordinate),
                                     ordinate_dim='acceleration')
    from visualdynamics.core.averaging import Averaging

    run.averaging = Averaging(frame_length=1024, frames=7)
    return rigid, run


def _by_pair(data):
    return {(r, f): i for i, (r, f) in enumerate(
        zip(data.response_dof, data.reference_dof))}


def _matched(ours, theirs):
    """The largest relative difference over records matched by (DOF,
    reference), for two objects that should hold the same matrix."""
    index = _by_pair(ours)
    worst = 0.0
    for i, pair in enumerate(_by_pair(theirs)):
        scale = np.abs(theirs.ordinate[i]).max() or 1.0
        worst = max(worst, np.abs(ours.ordinate[index[pair]]
                                  - theirs.ordinate[i]).max() / scale)
    return worst


def test_a_cpsd_transforms_as_the_cpsd_of_the_transformed_record():
    """Linear both ways, so the two orders agree to round-off — the
    matrix rule S_qq = Φ⁺ S_uu Φ⁺ᴴ checked against the record rule."""
    rigid, run = _rigid_and_run()
    modal_run, _report = to_modal(run, rigid)
    modal_cpsd, report = to_modal(run.compute_cpsds(), rigid)
    assert modal_cpsd.num_records == 36
    assert _matched(modal_cpsd, modal_run.compute_cpsds()) < 1e-12
    assert report.worst_residual == pytest.approx(0.0, abs=1e-9)
    pairs = _by_pair(modal_cpsd)
    assert modal_cpsd.ordinate_dim[pairs[('M1', 'M1')]] == \
        'acceleration**2/frequency'
    assert modal_cpsd.ordinate_dim[pairs[('M1', 'M6')]] == \
        'acceleration*angular_acceleration/frequency'
    assert modal_cpsd.ordinate_dim[pairs[('M6', 'M6')]] == \
        'angular_acceleration**2/frequency'
    back, _report = to_physical(modal_cpsd, rigid)
    assert back.num_records == 225
    assert _matched(back, run.compute_cpsds()) < 1e-12


def test_autospectra_alone_are_refused_by_name():
    rigid, run = _rigid_and_run()
    with pytest.raises(ValueError, match='no cross terms'):
        to_modal(run.compute_psds(), rigid)
    # and a matrix with one pair missing both ways
    cpsd = run.compute_cpsds()
    pairs = _by_pair(cpsd)
    keep = [i for pair, i in pairs.items()
            if {pair[0], pair[1]} != {'101X+', '102Y+'}]
    thinned = visualdynamics.Psd(
        cpsd.abscissa, cpsd.ordinate[keep],
        response_dof=[cpsd.response_dof[i] for i in keep],
        reference_dof=[cpsd.reference_dof[i] for i in keep],
        ordinate_dim=[cpsd.ordinate_dim[i] for i in keep])
    with pytest.raises(ValueError, match='no cross terms between'):
        to_modal(thinned, rigid)
    # one half of a pair is enough: a CPSD is Hermitian by definition
    keep = [i for (r, f), i in pairs.items()
            if not (r == '102Y+' and f == '101X+')]
    halved = visualdynamics.Psd(
        cpsd.abscissa, cpsd.ordinate[keep],
        response_dof=[cpsd.response_dof[i] for i in keep],
        reference_dof=[cpsd.reference_dof[i] for i in keep],
        ordinate_dim=[cpsd.ordinate_dim[i] for i in keep])
    whole, _report = to_modal(cpsd, rigid)
    filled, _report = to_modal(halved, rigid)
    assert _matched(filled, whole) < 1e-12


def test_a_spectrum_transforms_as_rows():
    rigid, run = _rigid_and_run()
    modal_run, _report = to_modal(run, rigid)
    modal_spectrum, _report = to_modal(run.compute_spectra(), rigid)
    assert isinstance(modal_spectrum, visualdynamics.Spectrum)
    expected = modal_run.compute_spectra()
    assert np.abs(modal_spectrum.ordinate - expected.ordinate).max() < \
        1e-12 * np.abs(expected.ordinate).max()
    assert modal_spectrum.response_dof == ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']


def test_an_srs_and_a_coherence_do_not_transform():
    from visualdynamics.core.shocks import Shock

    rigid, run = _rigid_and_run()
    run.shocks = (Shock(0.1, 0.5),)
    with pytest.raises(ValueError, match='transform the time history'):
        to_modal(run.compute_srs(), rigid)
    assert reads_as(run.compute_srs(), rigid) is None
    forced = visualdynamics.TimeHistory(
        run.abscissa, np.vstack([run.ordinate, run.ordinate[:1]]),
        response_dof=[*run.response_dof, '9001X+'],
        ordinate_dim=['acceleration'] * run.num_records + ['force'])
    forced.averaging = run.averaging
    with pytest.raises(ValueError, match='ratio'):
        to_modal(forced.compute_multiple_coherence(), rigid)


def test_an_frf_transforms_in_both_halves_and_round_trips():
    rigid, _run = _rigid_and_run(2)
    rng = np.random.default_rng(5)
    f = np.linspace(1.0, 100.0, 50)
    modal_frf = (rng.standard_normal((len(f), 6, 6))
                 + 1j * rng.standard_normal((len(f), 6, 6)))
    phi = np.real(rigid.shape_matrix).T
    physical = phi @ modal_frf @ phi.T
    rows, resp, refs = [], [], []
    for a, da in enumerate(rigid.coordinate):
        for b, db in enumerate(rigid.coordinate):
            rows.append(physical[:, a, b])
            resp.append(da)
            refs.append(db)
    frf = visualdynamics.Frf(f, np.asarray(rows), response_dof=resp,
                             reference_dof=refs,
                             ordinate_dim='acceleration/force')
    modal, report = to_modal(frf, rigid)
    assert modal.num_records == 36 and report.notes == []
    pairs = _by_pair(modal)
    got = np.stack([[modal.ordinate[pairs[(f'M{k + 1}', f'M{l + 1}')]]
                     for l in range(6)] for k in range(6)], axis=0)
    assert np.abs(np.moveaxis(got, 2, 0) - modal_frf).max() < 1e-12
    assert modal.ordinate_dim[pairs[('M1', 'M1')]] == 'acceleration/force'
    assert modal.ordinate_dim[pairs[('M6', 'M6')]] == \
        'angular_acceleration/moment', 'the point\'s rotation per moment'
    back, _report = to_physical(modal, rigid)
    assert back.num_records == 225
    assert _matched(back, frf) < 1e-12


def test_an_frf_keeps_a_drive_the_shapes_do_not_cover():
    """A shaker at a synthetic DOF has no shape row: the response rows
    transform, the reference stays physical, and the report says so."""
    rigid, _run = _rigid_and_run(2)
    rng = np.random.default_rng(6)
    f = np.linspace(1.0, 100.0, 50)
    modal = rng.standard_normal((len(f), 6)) + 1j * rng.standard_normal(
        (len(f), 6))
    phi = np.real(rigid.shape_matrix).T
    physical = np.einsum('ak,lk->la', phi, modal)
    frf = visualdynamics.Frf(f, physical.T, response_dof=list(rigid.coordinate),
                             reference_dof=['9001X+'] * rigid.num_dofs,
                             ordinate_dim='acceleration/force')
    result, report = to_modal(frf, rigid)
    assert result.num_records == 6
    assert set(result.reference_dof) == {'9001X+'}
    assert report.notes == [
        'references kept physical: 9001X+ not in the shapes']
    assert np.abs(result.ordinate.T - modal).max() < 1e-12
    assert result.ordinate_dim[5] == 'angular_acceleration/force'
    back, _report = to_physical(result, rigid)
    assert back.reference_dof[:1] == ['9001X+']
    assert np.abs(back.ordinate - physical.T).max() < 1e-12


def _specification(rigid, upper=2.0, lower=0.5, skew=None):
    f = np.linspace(10.0, 200.0, 20)
    target = np.zeros((len(f), 6, 6), dtype=complex)
    for k in range(6):
        target[:, k, k] = (k + 1) * 1e-3 * (f / 100.0) ** -0.5
    target[:, 0, 5] = 0.3e-3 * np.exp(0.4j)
    target[:, 5, 0] = np.conj(target[:, 0, 5])
    phi = np.real(rigid.shape_matrix).T
    physical = phi @ target @ phi.T
    rows, resp, refs, up, lo = [], [], [], [], []
    for a, da in enumerate(rigid.coordinate):
        for b, db in enumerate(rigid.coordinate):
            rows.append(physical[:, a, b])
            resp.append(da)
            refs.append(db)
            factor = skew if (skew is not None and a == 0) else upper
            auto = np.real(physical[:, a, a])
            up.append(auto * factor if a == b else np.full(len(f), np.nan))
            lo.append(auto * lower if a == b else np.full(len(f), np.nan))
    return target, visualdynamics.Specification(
        f, np.asarray(rows), response_dof=resp, reference_dof=refs,
        ordinate_dim='acceleration**2/frequency',
        abort_upper=np.asarray(up), abort_lower=np.asarray(lo))


def test_a_specification_transforms_exactly_with_its_uniform_bands():
    rigid, _run = _rigid_and_run(3)
    target, spec = _specification(rigid)
    modal, _report = to_modal(spec, rigid)
    assert isinstance(modal, visualdynamics.Specification)
    assert modal.num_records == 36 and modal.has_limits
    pairs = _by_pair(modal)
    i = pairs[('M6', 'M6')]
    assert np.allclose(np.real(modal.ordinate[i]), np.real(target[:, 5, 5]))
    assert np.allclose(modal.limits['abort_upper'][i],
                       2.0 * np.real(target[:, 5, 5]))
    assert np.allclose(modal.limits['abort_lower'][i],
                       0.5 * np.real(target[:, 5, 5]))
    assert np.isnan(modal.limits['abort_upper'][pairs[('M1', 'M6')]]).all(), \
        'a cross term has no band of its own'
    assert modal.interpolation == spec.interpolation
    back, _report = to_physical(modal, rigid)
    assert isinstance(back, visualdynamics.Specification)
    assert _matched(back, spec) < 1e-12
    j = _by_pair(back)[('101X+', '101X+')]
    assert np.allclose(back.limits['abort_upper'][j],
                       spec.limits['abort_upper'][_by_pair(spec)[
                           ('101X+', '101X+')]])


def test_bands_that_differ_between_channels_are_refused():
    rigid, _run = _rigid_and_run(3)
    _target, spec = _specification(rigid, skew=3.0)
    with pytest.raises(ValueError, match='abort upper band differs between '
                       r'101X\+ \(\+4.77 dB\) and 101Y\+ \(\+3.01 dB\)'):
        to_modal(spec, rigid)


def test_the_importer_reads_a_specifications_cross_terms_when_real(tmp_path):
    """The controller writes the target as a full matrix; the plate's
    off-diagonal is all zero — placeholders — so it imports as autos.
    Written with real cross terms, the same file imports the matrix."""
    import shutil

    import netCDF4

    from visualdynamics.io.rattlesnake import load

    plain = next(v for k, v in load(fixture_path('plate', 'random.nc4'))
                 .items() if k.endswith('_specification'))
    assert plain.num_records == 8
    copy = tmp_path / 'random.nc4'
    shutil.copy(fixture_path('plate', 'random.nc4'), copy)
    with netCDF4.Dataset(copy, 'a') as ds:
        group = next(g for g in ds.groups.values()
                     if 'specification_cpsd_matrix_real' in g.variables)
        real = group.variables['specification_cpsd_matrix_real']
        matrix = np.asarray(real[()])
        matrix[:, 0, 1] = matrix[:, 1, 0] = 0.5 * matrix[:, 0, 0]
        real[:] = matrix
    full = next(v for k, v in load(copy).items()
                if k.endswith('_specification'))
    assert full.num_records == 64
    pairs = _by_pair(full)
    assert np.allclose(np.real(full.ordinate[pairs[('101Z+', '104Z+')]]),
                       0.5 * np.real(full.ordinate[pairs[('101Z+', '101Z+')]]))
