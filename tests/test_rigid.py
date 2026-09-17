"""Rigid-body mode shapes of a geometry, from first principles.

The shapes are kinematic and checked kinematically: every node of a
translation mode moves the same way, a rotation mode moves a node by
the cross product of the axis with its offset from the point, and the
node *at* the point does not turn at all. Mass normalization is
checked the way the definition says — `φ M φᵀ = I` against a lumped
mass matrix whose mass, centroid and inertia tensor were computed from
the same masses — so a coupled tensor proves the principal-axis
handling rather than taking it on trust. The FEM's own rigid vectors,
written independently, agree on a beam.

Through the project, the verb behaves like every other derivation:
linked into the geometry's group, journaled, fingerprinted, refreshed,
and saved with its settings.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core import fem
from visualdynamics.core.rigid import (
    INERTIA_TERMS,
    MassProperties,
    axis_name,
    rigid_body_shapes,
)
from visualdynamics.deform import node_displacements


def _cloud(seed=3, n=7, unit='m'):
    """A few nodes scattered about, none on an axis of symmetry."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-1.0, 1.0, size=(n, 3))
    return visualdynamics.Geometry(node_id=np.arange(101, 101 + n),
                                   node_xyz=xyz, length_unit=unit)


def _lumped(geometry, masses):
    """Mass, CG and inertia tensor of point masses at the nodes, and the
    diagonal 3N mass matrix over the geometry's translational DOFs."""
    masses = np.asarray(masses, dtype=float)
    xyz = geometry.node_xyz
    mass = masses.sum()
    cg = (masses[:, None] * xyz).sum(axis=0) / mass
    d = xyz - cg
    tensor = sum(m * (np.dot(v, v) * np.eye(3) - np.outer(v, v))
                 for m, v in zip(masses, d))
    lumped = np.diag(np.repeat(masses, 3))
    return mass, cg, tensor, lumped


def _terms(tensor):
    return (tensor[0, 0], tensor[1, 1], tensor[2, 2],
            tensor[0, 1], tensor[0, 2], tensor[1, 2])


# ---- the shapes ------------------------------------------------------------


def test_six_unit_shapes_over_every_translational_dof():
    geometry = _cloud()
    shapes = rigid_body_shapes(geometry, geometry.suggest_mass_properties())
    assert shapes.num_shapes == 6
    assert shapes.num_dofs == 3 * geometry.num_nodes
    assert shapes.coordinate[:3] == ['101X+', '101Y+', '101Z+']
    assert np.all(shapes.frequency == 0.0), \
        'exactly zero — the FRF synthesis tests for that value'
    assert np.all(shapes.damping == 0.0)
    assert shapes.unscaled and shapes.mass_unit is None
    assert not shapes.is_complex
    assert shapes.description == [
        'Rigid translation X', 'Rigid translation Y',
        'Rigid translation Z', 'Rigid rotation about X',
        'Rigid rotation about Y', 'Rigid rotation about Z']


def test_the_shapes_move_the_nodes_rigidly():
    """Read back through the animation's own displacement, so the
    convention checked is the one the view draws."""
    geometry = _cloud()
    point = np.array([0.2, -0.1, 0.4])
    shapes = rigid_body_shapes(geometry, MassProperties(point))
    offsets = geometry.node_xyz - point
    for k in range(3):
        unit = np.zeros(3)
        unit[k] = 1.0
        moved = node_displacements(geometry, shapes.coordinate,
                                   shapes.shape_matrix[k])
        assert np.allclose(moved, unit), 'every node translates alike'
        turned = node_displacements(geometry, shapes.coordinate,
                                    shapes.shape_matrix[3 + k])
        assert np.allclose(turned, np.cross(unit, offsets)), \
            'a rotation of one radian moves a node by axis × offset'


def test_a_node_at_the_point_does_not_turn():
    geometry = visualdynamics.Geometry(
        [1, 2, 3], [[0.5, 0.5, 0.5], [1, 0, 0], [0, 2, 0]], length_unit='m')
    shapes = rigid_body_shapes(geometry, MassProperties((0.5, 0.5, 0.5)))
    assert np.allclose(shapes.shape_matrix[3:, :3], 0.0), \
        'the rotations pivot on the point; a node there stays put'
    assert not np.allclose(shapes.shape_matrix[3:, 3:], 0.0)


def test_the_fem_writes_the_same_vectors():
    """`fem.Model.rigid_body_vectors` is the other from-scratch
    derivation of these motions, about the node centroid; the two
    agree on every translational DOF of a beam."""
    steel = fem.Material('steel', youngs_modulus=200e9, density=7850,
                         poissons_ratio=0.3)
    model = fem.Model('beam')
    for i in range(5):
        model.add_node(100 + i, 0.3 * i, 0.1 * i * i, 0.05 * i)
    model.add_chain(range(100, 105), steel, fem.Section.rod('rod', 0.02))
    theirs = model.rigid_body_vectors()
    labels = model.dof_strings()

    geometry = visualdynamics.Geometry(
        model.node_ids, [model._nodes[k] for k in model.node_ids],
        length_unit='m')
    ours = rigid_body_shapes(geometry, geometry.suggest_mass_properties())
    for j, dof in enumerate(ours.coordinate):
        row = labels.index(dof)
        assert np.allclose(ours.shape_matrix[:, j], theirs[row, :]), dof


# ---- mass normalization ------------------------------------------------------


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_mass_normalized_against_the_lumped_masses(seed):
    """The definition itself: point masses at the nodes give a mass,
    a centroid and a tensor; the shapes built from those three are
    orthonormal in the lumped mass matrix. A random cloud's tensor is
    fully coupled, so this is what proves the principal axes."""
    geometry = _cloud(seed)
    masses = np.random.default_rng(seed + 10).uniform(0.5, 2.0,
                                                      geometry.num_nodes)
    mass, cg, tensor, lumped = _lumped(geometry, masses)
    assert abs(tensor[0, 1]) > 1e-3, 'the cloud must couple the axes'
    properties = MassProperties(tuple(cg), mass=mass,
                                inertia=_terms(tensor))
    shapes = rigid_body_shapes(geometry, properties)
    gram = shapes.shape_matrix @ lumped @ shapes.shape_matrix.T
    assert np.allclose(gram, np.eye(6), atol=1e-12)
    assert not shapes.unscaled
    assert shapes.mass_unit == 'kg'
    assert np.all(shapes.modal_mass == 1.0)


def test_the_diagonal_shortcut_would_fail():
    """What the principal axes guard against: rotations about the
    global axes scaled by the diagonal are not orthonormal under a
    coupled tensor. Written down so the previous test is known to be
    asking a question with a wrong answer available."""
    geometry = _cloud(5)
    masses = np.random.default_rng(15).uniform(0.5, 2.0, geometry.num_nodes)
    mass, cg, tensor, lumped = _lumped(geometry, masses)
    unit = rigid_body_shapes(geometry, MassProperties(tuple(cg)))
    naive = unit.shape_matrix.copy()
    naive[:3] /= np.sqrt(mass)
    naive[3:] /= np.sqrt(np.diag(tensor))[:, None]
    gram = naive @ lumped @ naive.T
    assert not np.allclose(gram, np.eye(6), atol=1e-6)


def test_a_diagonal_tensor_keeps_the_global_axes():
    properties = MassProperties((0, 0, 0), mass=2.0,
                                inertia=(3.0, 1.0, 2.0, 0, 0, 0))
    moments, axes = properties.principal_axes()
    assert np.allclose(axes, np.eye(3)), \
        'ordered X, Y, Z — not by ascending moment, which eigh would give'
    assert np.allclose(moments, [3.0, 1.0, 2.0])
    shapes = rigid_body_shapes(_cloud(), properties)
    assert shapes.description[3:] == [
        'Rigid rotation about X', 'Rigid rotation about Y',
        'Rigid rotation about Z']


def test_a_coupled_tensor_names_its_axes():
    """Lightly coupled: each principal axis is nearest one global axis
    and is signed along it, so mode 4 still reads as 'about X-ish'."""
    tensor = np.array([[3.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 2.0]])
    properties = MassProperties((0, 0, 0), mass=1.0, inertia=_terms(tensor))
    moments, axes = properties.principal_axes()
    assert np.allclose(axes @ axes.T, np.eye(3))
    for k in range(3):
        assert axes[k, k] > 0.9, 'nearest its own global axis, pointed +'
        assert np.allclose(tensor @ axes[k], moments[k] * axes[k])
    shapes = rigid_body_shapes(_cloud(), properties)
    assert shapes.description[3].startswith('Rigid rotation about (')
    assert shapes.description[5] == 'Rigid rotation about Z', \
        'the uncoupled axis keeps its name'


def test_axis_name():
    assert axis_name(np.array([0, 1, 0])) == 'Y'
    assert axis_name(np.array([0.6, 0.8, 0])) == '(0.60, 0.80, 0.00)'


def test_a_rotated_displacement_frame_gets_its_own_coefficients():
    """A node measured in a frame turned 90° about Z: its local X is
    global Y, so the global-X translation reads zero there and the
    global-Y translation reads one."""
    turned = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1], [0, 0, 0]],
                      dtype=float)
    geometry = visualdynamics.Geometry(
        [1, 2], [[0, 0, 0], [1, 0, 0]], node_disp_cs=[1, 2],
        cs_id=[1, 2], cs_matrix=[np.vstack([np.eye(3), np.zeros(3)]),
                                 turned], length_unit='m')
    shapes = rigid_body_shapes(geometry, MassProperties((0, 0, 0)))
    x, y = shapes.shape_matrix[0], shapes.shape_matrix[1]
    assert x[shapes.coordinate.index('2X+')] == pytest.approx(0.0)
    assert x[shapes.coordinate.index('2Y+')] == pytest.approx(-1.0)
    assert y[shapes.coordinate.index('2X+')] == pytest.approx(1.0)
    # and the picture is still a rigid translation
    assert np.allclose(node_displacements(geometry, shapes.coordinate, x),
                       [[1, 0, 0], [1, 0, 0]])


# ---- the settings ------------------------------------------------------------


def test_the_settings_validate_at_entry():
    with pytest.raises(ValueError, match='both'):
        MassProperties((0, 0, 0), mass=1.0)
    with pytest.raises(ValueError, match='both'):
        MassProperties((0, 0, 0), inertia=(1, 1, 1, 0, 0, 0))
    with pytest.raises(ValueError, match='positive'):
        MassProperties((0, 0, 0), mass=0.0, inertia=(1, 1, 1, 0, 0, 0))
    with pytest.raises(ValueError, match='positive definite'):
        MassProperties((0, 0, 0), mass=1.0, inertia=(1, 1, 1, 5, 0, 0))
    with pytest.raises(ValueError, match='six'):
        MassProperties((0, 0, 0), mass=1.0, inertia=(1, 1, 1))
    with pytest.raises(ValueError, match='finite'):
        MassProperties((0, float('nan'), 0))
    assert MassProperties([0, 1, 2]).point == (0.0, 1.0, 2.0), \
        'a list arrives as a tuple, so a JSON round trip compares equal'
    assert INERTIA_TERMS == ('Ixx', 'Iyy', 'Izz', 'Ixy', 'Ixz', 'Iyz')


def test_describe_is_the_one_wording():
    assert MassProperties((0.1, 0, 0.05)).describe() == \
        'about (0.1, 0, 0.05) m, unit shapes'
    scaled = MassProperties((0.1, 0, 0.05), mass=2.4,
                            inertia=(1, 1, 1, 0, 0, 0))
    assert scaled.describe() == \
        'about (0.1, 0, 0.05) m, mass-normalized (2.4 kg)'


def test_refusals_by_name():
    with pytest.raises(ValueError, match='no nodes'):
        rigid_body_shapes(visualdynamics.Geometry([], np.empty((0, 3)),
                                                  length_unit='m'),
                          MassProperties((0, 0, 0)))
    undeclared = _cloud(unit=None)
    with pytest.raises(ValueError, match='define_units'):
        rigid_body_shapes(undeclared, MassProperties(
            (0, 0, 0), mass=1.0, inertia=(1, 1, 1, 0, 0, 0)))
    shapes = rigid_body_shapes(undeclared, MassProperties((0, 0, 0)))
    assert shapes.num_shapes == 6, \
        'unit shapes are kinematic and need no unit'


def test_the_centroid_is_the_suggestion():
    geometry = _cloud()
    suggested = geometry.suggest_mass_properties()
    assert np.allclose(suggested.point, geometry.node_xyz.mean(axis=0))
    assert not suggested.scaled


# ---- through the project ---------------------------------------------------


def _project():
    project = visualdynamics.Project('Rigid')
    project.add('Plate', _cloud())
    return project


def test_the_verb_adds_the_set_to_the_geometry_group():
    project = _project()
    name = project.generate_rigid_body_modes('Plate')
    assert name == 'Plate Rigid Body Modes'
    assert project.links == [{'members': ['Plate', name], 'role': None}]
    assert project['Plate'].mass_properties == \
        project['Plate'].suggest_mass_properties(), \
        'the adopted suggestion is stored, so the fingerprint is honest'
    assert project.provenance[name]['verb'] == 'generate_rigid_body_modes'
    assert project.journal[-1] == "project.generate_rigid_body_modes('Plate')"


def test_the_verb_applies_to_geometries_only():
    project = _project()
    assert 'generate_rigid_body_modes' in dict(project.verbs('Plate'))
    project.import_file(fixture_path('plate', 'modal.nc4'))
    assert 'generate_rigid_body_modes' not in dict(
        project.verbs('Time History'))
    with pytest.raises(TypeError, match='not a geometry'):
        project.generate_rigid_body_modes('Time History')


def test_the_set_goes_stale_when_the_point_or_the_nodes_move():
    project = _project()
    geometry = project['Plate']
    name = project.generate_rigid_body_modes('Plate')
    assert project.stale() == {}
    geometry.mass_properties = MassProperties((0.5, 0, 0))
    story = project.stale()[name]
    assert 'about (0.5, 0, 0) m, unit shapes' in story
    assert 'the geometry now says' in story
    project.refresh(name)
    assert project.stale() == {}
    geometry.node_xyz[0] += 0.1
    assert project.stale()[name] == 'computed on nodes that have since moved'
    project.refresh(name)
    assert np.allclose(
        node_displacements(geometry, project[name].coordinate,
                           project[name].shape_matrix[5])[0],
        np.cross([0, 0, 1], geometry.node_xyz[0] - [0.5, 0, 0])), \
        'refresh rebuilt the shapes from the moved node and the set point'


def test_the_settings_survive_a_save_and_a_load(tmp_path):
    project = _project()
    scaled = MassProperties((0.1, 0.2, 0.3), mass=2.0,
                            inertia=(1.0, 2.0, 3.0, 0.1, 0.0, -0.2))
    project['Plate'].mass_properties = scaled
    name = project.generate_rigid_body_modes('Plate')
    back = visualdynamics.Project.open(project.save(tmp_path / 'r.vdyn'))
    assert back['Plate'].mass_properties == scaled, \
        'the point, the mass and all six inertia terms read back'
    assert back.stale() == {}, 'and compare equal to the recorded ones'
    assert back[name] == project[name]
    back['Plate'].mass_properties = MassProperties((0.1, 0.2, 0.3))
    assert name in back.stale(), \
        'the fingerprint survived the file, so the badge still works'
    again = visualdynamics.Project.open(back.save(tmp_path / 'u.vdyn'))
    assert again['Plate'].mass_properties == MassProperties((0.1, 0.2, 0.3)), \
        'an unscaled setting saves without mass attrs and reads back unscaled'


def test_the_session_script_replays(tmp_path):
    project = visualdynamics.Project('Replay')
    project.import_file(fixture_path('plate', 'geometry.unv'))
    geometry_name = next(n for n in project.names
                         if isinstance(project[n], visualdynamics.Geometry))
    # the UNV declares no unit, and mass properties need one
    project[geometry_name].define_units('in')
    project.record_call(geometry_name, 'define_units', 'in')
    properties = MassProperties((0.1, 0.0, 0.0), mass=3.0,
                                inertia=(1.0, 1.0, 1.0, 0, 0, 0))
    project[geometry_name].mass_properties = properties
    project.record_setting(geometry_name, 'mass_properties', properties)
    name = project.generate_rigid_body_modes(geometry_name)
    script = project.session_script()
    assert 'from visualdynamics.core.rigid import MassProperties' in script
    room: dict = {}
    exec(script, room)                                  # noqa: S102
    replayed = room['project']
    assert replayed[name] == project[name]
    assert replayed[geometry_name].mass_properties == properties


# ---- the sdynpy oracle ---------------------------------------------------------


def test_sdynpy_agrees_on_the_shapes():
    """Frozen numbers from `generate_rigid_oracle.py` in the generators
    repository — an agreement oracle, not a truth oracle. The scaled
    case uses a diagonal tensor on purpose: the reference takes the
    diagonal of a coupled one unless asked for principal axes, and the
    two would disagree there by design (this file's first tests say
    which is right)."""
    path = fixture_path('sdynpy_oracle', 'rigid.npz')
    try:
        oracle = np.load(path)
    except FileNotFoundError:
        pytest.skip('rigid oracle not generated')
    geometry = visualdynamics.Geometry(
        oracle['node_id'], oracle['node_xyz'],
        node_disp_cs=oracle['node_disp_cs'], cs_id=oracle['cs_id'],
        cs_matrix=oracle['cs_matrix'], length_unit='m')
    point = tuple(oracle['point'])
    dofs = [str(d) for d in oracle['dofs']]
    for case, properties in (
            ('unit', MassProperties(point)),
            ('scaled', MassProperties(point, mass=float(oracle['mass']),
                                      inertia=tuple(oracle['inertia'])))):
        ours = rigid_body_shapes(geometry, properties)
        columns = [ours.coordinate.index(dof) for dof in dofs]
        assert np.allclose(ours.shape_matrix[:, columns],
                           oracle[f'shapes_{case}'], atol=1e-12), case
