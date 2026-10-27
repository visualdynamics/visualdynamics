"""Planning instrumentation: where a sensor can sit, and which places
Effective Independence chooses (`core.instrumentation`).

The candidates are checked on shapes whose surfaces are known by eye:
a block's faces, edges and corners, two parts touching, a cylinder and
a plate. Effective Independence is checked against a plain version of
the method written out here, which recomputes the information matrix
from scratch at every removal: on problems small enough that nothing
is removed in batches, the two must choose the same sensors.
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core import instrumentation as ins
from visualdynamics.core import mesh
from visualdynamics.core.fem import MATERIALS, GroupProperties, Model
from visualdynamics.core.geometry import Geometry
from visualdynamics.core.shapes import ShapeSet

INCH = 0.0254


def _block(widths=(2, 1, 1), size=0.5):
    project = visualdynamics.Project('p')
    name = project.new_geometry('Part', unit='in')
    solid = project.add_solid_block(name, (0, 0, 0), widths, name='block', unit='in')
    project.mesh_solids(name, [solid], size, unit='in')
    return project[name]


def _at(geometry, point):
    """The id of the node at `point`, inches."""
    hits = np.flatnonzero(np.all(np.isclose(geometry.node_xyz / INCH, point), axis=1))
    assert len(hits) == 1, point
    return int(geometry.node_id[hits[0]])


def _frames_at(plan, node):
    return [plan.frames[k] for k in np.flatnonzero(plan.source == node)]


def test_a_block_offers_its_surface_once_per_face_and_never_its_inside():
    """A 2 x 1 x 1 in block at half an inch is 5 x 3 x 3 nodes. Its 3
    interior nodes are no place for a sensor; each face node is one
    place, each edge node two (one per face, so the sensor sits flat)
    and each corner three."""
    geometry = _block()
    plan = ins.candidates(geometry)
    inside = [_at(geometry, (x, 0, 0)) for x in (-0.5, 0, 0.5)]
    assert not set(inside) & set(plan.source.tolist()), 'an interior node'
    per_node = Counter(plan.source.tolist())
    assert sorted(Counter(per_node.values()).items()) == [(1, 14), (2, 20), (3, 8)]
    assert len(plan.source) == 14 + 2 * 20 + 3 * 8
    # the first place at a node keeps its id; the others are numbered
    # above every node of the model
    ids = plan.geometry.node_id
    assert len(set(ids.tolist())) == len(ids)
    first = {int(n) for n in plan.source}
    assert first <= set(ids.tolist())
    assert all(i > geometry.node_id.max() for i in ids if i not in first)
    assert plan.kind == ['solid'] * len(ids)
    # every place is measured in a system of its own, at the node
    assert len(set(plan.geometry.node_disp_cs.tolist())) == len(ids)
    rows = {int(n): k for k, n in enumerate(geometry.node_id)}
    assert np.allclose(plan.geometry.node_xyz, geometry.node_xyz[[rows[n] for n in plan.source]])


def test_z_is_the_outward_normal_and_x_lies_along_a_system_axis():
    """The standard convention: +Z out of the surface. +X is the global
    axis least aligned with the normal, laid into the surface: on the
    top face the global X, on the +X face the global Y, and Y completes
    a right-handed set."""
    geometry = _block()
    plan = ins.candidates(geometry)
    top, = _frames_at(plan, _at(geometry, (0, 0, 0.5)))
    assert np.allclose(top, np.eye(3))
    side, = _frames_at(plan, _at(geometry, (1, 0, 0)))
    assert np.allclose(side, [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    bottom, = _frames_at(plan, _at(geometry, (0, 0, -0.5)))
    assert np.allclose(bottom[2], [0, 0, -1]), 'out of the block, not up'
    corner = _frames_at(plan, _at(geometry, (1, 0.5, 0.5)))
    normals = sorted(tuple(np.round(f[2]).astype(int)) for f in corner)
    assert normals == [(0, 0, 1), (0, 1, 0), (1, 0, 0)], 'one place on each face'
    for frame in plan.frames:
        assert np.allclose(frame @ frame.T, np.eye(3))
        assert np.isclose(np.linalg.det(frame), 1.0)


def test_surface_frame_for_any_normal():
    rng = np.random.default_rng(3)
    for normal in rng.normal(size=(20, 3)):
        frame = ins.surface_frame(normal)
        assert np.allclose(frame[2], normal / np.linalg.norm(normal))
        assert np.allclose(frame @ frame.T, np.eye(3)) and np.isclose(np.linalg.det(frame), 1)
        least = np.eye(3)[np.argmin(np.abs(normal))]
        # X lies in the plane of the normal and the least-aligned axis
        assert abs(np.dot(np.cross(frame[2], least), frame[0])) < 1e-12


def test_where_two_parts_touch_is_no_place_for_a_sensor():
    """A post standing on a base, meshed apart at different sizes, so
    their nodes do not meet: the post's underside and the patch of the
    base it covers are inside the assembly. The rest of the base's top
    is still a place."""
    project = visualdynamics.Project('p')
    name = project.new_geometry('Part', unit='in')
    base = project.add_solid_block(name, (0, 0, 0), (4, 2, 1), name='base', unit='in')
    post = project.add_solid_block(name, (1.5, 0, 1), (1, 2, 1), name='post', unit='in')
    project.mesh_solids(name, [base], 0.5, unit='in')
    project.mesh_solids(name, [post], 0.25, unit='in', apart=True)
    geometry = project[name]
    plan = ins.candidates(geometry)
    xyz = plan.geometry.node_xyz / INCH
    normals = plan.frames[:, 2]
    up_at_seam = (np.isclose(xyz[:, 2], 0.5) & np.isclose(normals[:, 2], 1.0)
                  & (xyz[:, 0] > 1.0 + 1e-9))
    down_at_seam = np.isclose(xyz[:, 2], 0.5) & np.isclose(normals[:, 2], -1.0)
    assert not up_at_seam.any(), 'the base under the post'
    assert not down_at_seam.any(), "the post's underside"
    open_top = np.isclose(xyz[:, 2], 0.5) & np.isclose(normals[:, 2], 1.0) & (xyz[:, 0] < 0.9)
    # x from -2 to 0.5 by half inches, six rows of five
    assert open_top.sum() == 6 * 5, 'the base top beside the post is still a place'
    # the post's own top and sides are places
    assert (np.isclose(xyz[:, 2], 1.5) & np.isclose(normals[:, 2], 1.0)).sum() == 5 * 9


def test_a_cylinder_side_is_one_place_per_node_facing_out():
    """On a smooth curve the facets' normals are averaged: a node on a
    cylinder's side is one place, its normal radial."""
    project = visualdynamics.Project('p')
    name = project.new_geometry('Part', unit='in')
    cylinder = project.add_solid_cylinder(name, (0, 0, 0), 1.0, 2.0, name='c', unit='in')
    project.mesh_solids(name, [cylinder], 0.25, unit='in')
    plan = ins.candidates(project[name])
    xyz = plan.geometry.node_xyz / INCH
    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    side = np.isclose(radius, 1.0, atol=1e-6) & (np.abs(xyz[:, 2]) < 1 - 1e-6)
    assert side.sum() > 20
    nodes = plan.source[side]
    assert len(set(nodes.tolist())) == len(nodes), 'one place each'
    radial = np.column_stack([xyz[side, 0], xyz[side, 1], np.zeros(side.sum())])
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    cosine = np.einsum('ij,ij->i', plan.frames[side, 2], radial)
    assert cosine.min() > np.cos(np.radians(5)), 'radial, outward'


def test_a_plate_offers_every_node_once():
    project = visualdynamics.Project('p')
    name = project.new_geometry('Plate', unit='in')
    mesh.join(project[name], mesh.plane((0, 0, 0), (4, 0, 0), (0, 2, 0), 0.5, 'plate',
                                        unit='in'))
    geometry = project[name]
    plan = ins.candidates(geometry)
    assert sorted(plan.source.tolist()) == sorted(geometry.node_id.tolist())
    assert set(plan.kind) == {'plate'}
    assert np.allclose(np.abs(plan.frames[:, 2, 2]), 1.0), 'normal to the plate'
    assert len({tuple(np.round(f[2], 9)) for f in plan.frames}) == 1, 'one side for all'


def _bar():
    geometry = _block((8, 2, 1), 0.5)
    for group in geometry.group_id:
        geometry.group_properties[int(group)] = GroupProperties(MATERIALS['1018 steel'])
    return geometry, Model.from_geometry(geometry).eigensolution(num_modes=14)


def test_the_modes_are_read_in_each_places_own_directions():
    geometry, shapes = _bar()
    plan = ins.candidates(geometry)
    modes = list(range(6, 14))
    blocks, measured = ins.local_shapes(shapes, modes, plan, geometry)
    assert measured.all() and blocks.shape == (len(plan.source), 3, len(modes))
    node = _at(geometry, (4, 0, 0))                  # the +X end face
    k = int(np.flatnonzero(plan.source == node)[0])
    column = {dof: c for c, dof in enumerate(shapes.coordinate)}
    for local, dof in ((2, f'{node}X+'), (0, f'{node}Y+'), (1, f'{node}Z+')):
        assert np.allclose(blocks[k, local], shapes.shape_matrix[modes, column[dof]])


def _naive(blocks, triaxial, uniaxial, axes=(2,)):
    """Effective Independence the long way: every step recomputes the
    information matrix and removes the candidate whose loss leaves the
    largest determinant."""
    count, _three, modes = blocks.shape
    kept = list(range(count)) if triaxial else []
    while len(kept) > triaxial:
        def without(k, kept=kept):
            rest = [j for j in kept if j != k]
            return np.linalg.slogdet(sum(blocks[j].T @ blocks[j] for j in rest))[1]
        kept.remove(max(kept, key=without))
    tri = kept
    base = sum((blocks[k].T @ blocks[k] for k in tri), np.zeros((modes, modes)))
    options = [(k, a) for k in range(count) if k not in tri for a in axes]
    kept = list(range(len(options)))
    while len(kept) > uniaxial:
        def without(i, kept=kept):
            rest = [options[j] for j in kept if j != i]
            matrix = base + sum(np.outer(blocks[k, a], blocks[k, a]) for k, a in rest)
            return np.linalg.slogdet(matrix)[1]
        kept.remove(max(kept, key=without))
    return sorted(tri), sorted(options[i] for i in kept)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_effective_independence_chooses_as_the_long_way_does(seed):
    rng = np.random.default_rng(seed)
    blocks = rng.normal(size=(14, 3, 5))
    plan = ins.effective_independence(blocks, 4, 3)
    assert (sorted(plan.triaxial), sorted(plan.uniaxial)) == _naive(blocks, 4, 3)
    assert np.isclose(plan.independence.sum(), 5), 'the shares sum to the modes'
    assert len(plan.independence) == 4 + 3


def test_the_bar_is_measured_well_by_a_few_sensors():
    geometry, shapes = _bar()
    plan = ins.candidates(geometry)
    blocks, measured = ins.local_shapes(shapes, list(range(6, 14)), plan, geometry)
    placed = ins.effective_independence(blocks, 3, 4, allowed=measured, sources=plan.source)
    assert len(placed.triaxial) == 3 and len(placed.uniaxial) == 4
    assert all(axis == 2 for _k, axis in placed.uniaxial), 'along the normal by default'
    assert placed.condition < 20
    ends = np.abs(plan.geometry.node_xyz[placed.triaxial, 0] / INCH)
    assert np.allclose(ends, 4.0), 'a free bar is best measured at its ends'


def test_one_node_is_one_tri_axial_place_whichever_face_it_is_on():
    """A tri-axial accelerometer measures the same translations on
    either face of an edge, so the places at one node are offered once
    for them: two tri-axial sensors never land on one node."""
    geometry, shapes = _bar()
    plan = ins.candidates(geometry)
    blocks, measured = ins.local_shapes(shapes, list(range(6, 14)), plan, geometry)
    placed = ins.effective_independence(blocks, 12, 0, allowed=measured, sources=plan.source)
    nodes = plan.source[placed.triaxial]
    assert len(set(nodes.tolist())) == len(nodes)


def test_no_sensors_or_too_few_places_is_refused():
    blocks = np.random.default_rng(0).normal(size=(10, 3, 8))
    with pytest.raises(ValueError, match='no sensors to place'):
        ins.effective_independence(blocks, 0, 0)
    with pytest.raises(ValueError, match='but 10 places'):
        ins.effective_independence(blocks, 11, 0)
    with pytest.raises(ValueError, match='but 0 directions'):
        ins.effective_independence(blocks, 10, 1)


def test_many_candidates_are_thinned_in_batches_and_finished_one_at_a_time():
    rng = np.random.default_rng(5)
    blocks = rng.normal(size=(3000, 3, 6))
    placed = ins.effective_independence(blocks, 5, 4)
    assert len(placed.triaxial) == 5 and len(placed.uniaxial) == 4
    assert np.isclose(placed.independence.sum(), 6)
    # a tenth at a time while more than four times the wanted 20 remain,
    # never taking the pool below 80; from there, one at a time
    def left(count):
        return len(ins._remove(np.arange(count), np.arange(float(count)), 20))

    assert left(1000) == 900 and left(100) == 90
    assert left(85) == 80 and left(83) == 80
    assert left(80) == 79 and left(21) == 20


def test_normals_point_out_however_an_element_lists_its_nodes():
    """A brick whose nodes are listed the other way round (bottom face
    clockwise from above, as some files write them) has every face's
    corner order inward; the normal is turned to point away from the
    brick's own middle all the same."""
    from visualdynamics.core.geometry import Geometry

    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    for order in ([1, 2, 3, 4, 5, 6, 7, 8], [1, 4, 3, 2, 5, 8, 7, 6]):
        brick = Geometry(node_id=list(range(1, 9)), node_xyz=corners,
                         elem_conn=[order], elem_type=[115], elem_group=[1],
                         group_id=[1], group_name=['brick'])
        plan = ins.candidates(brick)
        assert len(plan.source) == 24, 'three faces at each of eight corners'
        middle = np.full(3, 0.5)
        outward = np.einsum('ij,ij->i', plan.frames[:, 2], plan.geometry.node_xyz - middle)
        assert (outward > 0).all(), order


def test_a_place_the_shapes_do_not_fully_measure_is_not_offered():
    """A node whose shapes lack a translation cannot carry a tri-axial
    accelerometer's three directions in its own frame; it is left out
    rather than read as zero."""
    from visualdynamics.core.shapes import ShapeSet

    geometry, shapes = _bar()
    node = _at(geometry, (4, 0, 0))
    keep = [c for c, dof in enumerate(shapes.coordinate) if dof != f'{node}Z+']
    thinned = ShapeSet(shapes.frequency, shapes.damping,
                       [shapes.coordinate[c] for c in keep], shapes.shape_matrix[:, keep])
    plan = ins.candidates(geometry)
    _blocks, measured = ins.local_shapes(thinned, list(range(6, 14)), plan, geometry)
    assert not measured[plan.source == node].any()
    assert measured[plan.source != node].all()


def test_the_experimental_geometry_is_a_whole_geometry():
    """Its system table holds the global system and one cartesian
    system per place, each with its origin at the place."""
    geometry = _block()
    plan = ins.candidates(geometry)
    experimental = plan.geometry
    count = len(plan.source)
    assert len(experimental.cs_id) == len(experimental.cs_name) == len(experimental.cs_type) == count + 1
    assert (experimental.cs_type == 0).all()
    assert np.allclose(experimental.cs_matrix[0], np.vstack([np.eye(3), np.zeros(3)]))
    rows = {int(c): k for k, c in enumerate(experimental.cs_id)}
    origins = np.array([experimental.cs_matrix[rows[int(c)], 3] for c in experimental.node_disp_cs])
    assert np.allclose(origins, experimental.node_xyz)
    assert (experimental.node_def_cs == experimental.cs_id[0]).all(), 'placed globally'
    extras = sorted(i for i in experimental.node_id if i > geometry.node_id.max())
    assert extras == list(range(int(geometry.node_id.max()) + 1,
                                int(geometry.node_id.max()) + 1 + len(extras))), 'in turn'
    assert experimental.length_unit == geometry.length_unit
    nothing = ins.candidates(geometry, groups=[])
    assert len(nothing.source) == 0 and len(nothing.geometry.node_id) == 0


def test_plates_point_away_from_the_middle_and_a_flat_one_up_its_axis():
    """A closed box of plates points out of the box. A single flat plate
    has the middle in its own plane, and points along the positive axis
    it is nearest."""
    project = visualdynamics.Project('p')
    name = project.new_geometry('Box', unit='in')
    for corner, a, b in [((0, 0, 0), (2, 0, 0), (0, 2, 0)), ((0, 0, 2), (2, 0, 0), (0, 2, 0)),
                         ((0, 0, 0), (2, 0, 0), (0, 0, 2)), ((0, 2, 0), (2, 0, 0), (0, 0, 2)),
                         ((0, 0, 0), (0, 2, 0), (0, 0, 2)), ((2, 0, 0), (0, 2, 0), (0, 0, 2))]:
        mesh.join(project[name], mesh.plane(corner, a, b, 1.0, 'skin', unit='in'))
    plan = ins.candidates(project[name])
    middle = np.full(3, 1.0 * INCH)
    outward = np.einsum('ij,ij->i', plan.frames[:, 2], plan.geometry.node_xyz - middle)
    assert (outward > 0).all()
    flat = visualdynamics.Project('q')
    name = flat.new_geometry('Plate', unit='in')
    mesh.join(flat[name], mesh.plane((0, 0, 0), (0, 2, 0), (2, 0, 0), 1.0, 'plate',
                                     unit='in'))
    normals = ins.candidates(flat[name]).frames[:, 2]
    assert np.allclose(normals, [0, 0, 1]), 'up, whichever way its corners run'


def test_beams_offer_every_node_measured_globally():
    from visualdynamics.core.geometry import Geometry

    frame = Geometry(node_id=[1, 2, 3, 4], node_xyz=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
                     elem_conn=[[1, 2], [2, 3], [3, 4]], elem_type=[21, 21, 21],
                     elem_group=[1, 1, 2], group_id=[1, 2], group_name=['a', 'b'])
    plan = ins.candidates(frame)
    assert plan.source.tolist() == [1, 2, 3, 4] and plan.kind == ['line'] * 4
    assert np.allclose(plan.frames, np.eye(3))
    assert ins.candidates(frame, groups=[2]).source.tolist() == [3, 4]


def test_a_model_measured_in_turned_systems_reads_the_same():
    """The finite element shapes are taken in each node's own measuring
    system (`node_disp_cs`); measured in turned ones, the same motion
    reads the same at every place."""
    from visualdynamics.core.shapes import ShapeSet
    from visualdynamics.rotate import frame_from_angles

    geometry, shapes = _bar()
    plan = ins.candidates(geometry)
    modes = list(range(6, 14))
    plain, _ = ins.local_shapes(shapes, modes, plan, geometry)
    turned = geometry.copy() if hasattr(geometry, 'copy') else __import__('copy').deepcopy(geometry)
    turn = frame_from_angles((30, -20, 75))[:3]
    system = turned.add_coordinate_system(rotation=turn)
    turned.node_disp_cs[:] = system
    columns = {dof: c for c, dof in enumerate(shapes.coordinate)}
    coordinate, matrix = [], []
    for node in geometry.node_id:
        global_motion = np.stack([shapes.shape_matrix[:, columns[f'{node}{a}+']] for a in 'XYZ'])
        local = turn @ global_motion                    # (3, modes): along the turned axes
        for a, axis in enumerate('XYZ'):
            coordinate.append(f'{node}{axis}+')
            matrix.append(local[a])
    reread = ShapeSet(shapes.frequency, shapes.damping, coordinate, np.array(matrix).T)
    seen, measured = ins.local_shapes(reread, modes, plan, turned)
    assert measured.all() and np.allclose(seen, plain)


def test_the_counts_at_their_limits():
    rng = np.random.default_rng(7)
    blocks = rng.normal(size=(10, 3, 8))
    exactly = ins.effective_independence(blocks, 2, 2)      # 3 x 2 + 2 = 8 directions
    assert np.isclose(exactly.independence.sum(), 8)
    assert exactly.planned.tolist() == list(range(8))
    short = ins.effective_independence(blocks, 2, 1)        # 7 directions for 8 modes
    assert short.planned.tolist() == list(range(7)) and np.isclose(short.independence.sum(), 7)
    assert short.information.shape == (7, 7)
    assert ins.shortfall(short, 8) == ('7 directions for 8 modes: planned for the lowest 7; '
                                       'the other 1 cannot be told apart by these sensors')
    assert ins.shortfall(exactly, 8) == ''
    every = ins.effective_independence(blocks, 10)            # uni-axial 0 by default
    assert sorted(every.triaxial) == list(range(10)) and every.uniaxial == []
    spare = ins.effective_independence(blocks, 7, 3)          # all three Z left
    assert sorted(k for k, _a in spare.uniaxial) == sorted(set(range(10)) - set(spare.triaxial))


@pytest.mark.parametrize('seed', [3, 4])
def test_uni_axial_sensors_alone(seed):
    rng = np.random.default_rng(seed)
    blocks = rng.normal(size=(12, 3, 4))
    plan = ins.effective_independence(blocks, 0, 6, axes=(0, 1, 2))
    assert plan.triaxial == []
    assert sorted(plan.uniaxial) == _naive(blocks, 0, 6, axes=(0, 1, 2))[1]
    assert np.isclose(plan.independence.sum(), 4) and len(plan.independence) == 6


def _bar_project():
    project = visualdynamics.Project('p')
    name = project.new_geometry('Bar', unit='in')
    solid = project.add_solid_block(name, (0, 0, 0), (8, 2, 1), name='bar', unit='in')
    project.mesh_solids(name, [solid], 0.5, unit='in')
    geometry = project[name]
    for group in geometry.group_id:
        geometry.group_properties[int(group)] = GroupProperties(MATERIALS['1018 steel'])
    return project, name, project.solve_modes(name, num_modes=14)


def test_a_plan_is_a_geometry_a_channel_table_and_the_modes_at_its_channels():
    project, name, modes = _bar_project()
    added = project.plan_instrumentation(modes, 3, 4)
    assert added == {'geometry': 'Bar Modes Plan Geometry',
                     'channels': 'Bar Modes Plan Channels',
                     'shapes': 'Bar Modes Plan Shapes'}
    assert set(project.object_group_of(added['geometry'])) == set(added.values())
    geometry, table, shapes = (project[added[k]] for k in ('geometry', 'channels', 'shapes'))
    # seven places: three tri-axial, four uni-axial, each its own system
    assert geometry.num_nodes == 7 and len(set(geometry.node_disp_cs.tolist())) == 7
    assert table.num_channels == 3 * 3 + 4
    dofs = table.dof_strings()
    assert shapes.coordinate == dofs, 'the shapes are at exactly the channels'
    assert {int(n) for n in table.frame['node']} == set(geometry.node_id.tolist())
    legs = list(table.frame['triax_dof'])
    assert legs[:9] == ['X', 'Y', 'Z'] * 3 and legs[9:] == [''] * 4
    assert list(table.frame['comment'][:3]) == ['tri-axial 1'] * 3
    assert (np.asarray(table.frame['direction'][9:]) == 'Z+').all(), 'along the normal'
    assert set(table.frame['role']) == {'response'} and set(table.frame['unit']) == {'g'}
    # the elastic modes, the rigid ones left out
    fe = project[modes]
    assert np.allclose(shapes.frequency, fe.frequency[fe.frequency > 0])
    # a channel reads the finite element motion along its place's axis
    fe_geometry = project[name]
    rows = {int(n): k for k, n in enumerate(fe_geometry.node_id)}
    columns = {dof: c for c, dof in enumerate(fe.coordinate)}
    systems = {int(c): k for k, c in enumerate(geometry.cs_id)}
    for c, dof in enumerate(dofs):
        node = int(table.frame['node'][c])
        k = list(geometry.node_id).index(node)
        axis = 'XYZ'.index(dof[-2])
        along = geometry.cs_matrix[systems[int(geometry.node_disp_cs[k])], axis]
        source = int(fe_geometry.node_id[np.argmin(np.linalg.norm(
            fe_geometry.node_xyz - geometry.node_xyz[k], axis=1))])
        assert source in rows
        motion = np.stack([fe.shape_matrix[fe.frequency > 0, columns[f'{source}{a}+']]
                           for a in 'XYZ'], axis=1)
        assert np.allclose(shapes.shape_matrix[:, c], motion @ along)
    off = shapes.auto_mac() - np.diag(np.diag(shapes.auto_mac()))
    assert off.max() < 0.5, 'the planned channels tell the modes apart'


def test_a_plan_replays_from_its_journal_line():
    import copy

    project, _name, modes = _bar_project()
    before = copy.deepcopy(project)
    added = project.plan_instrumentation(modes, 2, 3, modes=[6, 7, 8, 9], groups=['bar'])
    line = project.journal[-1]
    assert line == ("project.plan_instrumentation('Bar Modes', 2, 3, modes=[6, 7, 8, 9], "
                    "groups=['bar'])")
    exec(line, {'project': before})                  # noqa: S102
    assert np.allclose(before[added['shapes']].shape_matrix, project[added['shapes']].shape_matrix)
    assert list(before[added['geometry']].node_id) == list(project[added['geometry']].node_id)


def test_a_plan_needs_shapes_on_a_geometry_and_modes_to_measure():
    project, name, modes = _bar_project()
    with pytest.raises(TypeError, match='not a shape set'):
        project.plan_instrumentation(name, 2)
    with pytest.raises(ValueError, match='no sensors to place'):
        project.plan_instrumentation(modes, 0, 0)
    with pytest.raises(ValueError, match='no modes to measure'):
        project.plan_instrumentation(modes, 2, modes=[])
    loose = visualdynamics.Project('loose')
    loose.add('Modes', copy_of(project[modes]))
    with pytest.raises(ValueError, match='no geometry to plan'):
        loose.plan_instrumentation('Modes', 3)


def copy_of(obj):
    import copy

    return copy.deepcopy(obj)


def test_picked_element_groups_are_where_the_sensors_go():
    """A post meshed onto a base, sharing their seam: with only the post
    picked, every planned place is on the post."""
    project = visualdynamics.Project('p')
    name = project.new_geometry('Part', unit='in')
    base = project.add_solid_block(name, (0, 0, 0), (4, 2, 1), name='base', unit='in')
    post = project.add_solid_block(name, (1.5, 0, 1), (1, 2, 1), name='post', unit='in')
    project.mesh_solids(name, [base, post], 0.5, unit='in')
    geometry = project[name]
    for group in geometry.group_id:
        geometry.group_properties[int(group)] = GroupProperties(MATERIALS['1018 steel'])
    modes = project.solve_modes(name, num_modes=12)
    added = project.plan_instrumentation(modes, 3, 2, groups=['post'])
    placed = project[added['geometry']].node_xyz / INCH
    assert (placed[:, 2] >= 0.5 - 1e-9).all() and (placed[:, 0] >= 1.0 - 1e-9).all()
    everywhere = project.plan_instrumentation(modes, 3, 2, name='Anywhere')
    anywhere = project[everywhere['geometry']].node_xyz / INCH
    assert not ((anywhere[:, 2] >= 0.5 - 1e-9) & (anywhere[:, 0] >= 1.0 - 1e-9)).all(), (
        'left free, some go on the base: the narrowing is the pick, not the model')


def _normal_of(surface):
    """A flat surface's one normal."""
    assert np.allclose(surface.normals, surface.normals[0], atol=1e-9)
    return np.round(surface.normals[0], 9)


def test_a_block_is_six_surfaces_split_at_its_edges():
    geometry = _block()
    found = ins.skin(geometry)
    assert [s.id for s in found] == list(range(1, 7))
    normals = sorted(tuple(_normal_of(s)) for s in found)
    assert normals == sorted(tuple(row) for row in np.vstack([np.eye(3), -np.eye(3)]))
    # 2 x 1 x 1 at half an inch: faces of 4 x 2, 4 x 2 and 2 x 2
    assert sorted(len(s.faces) for s in found) == [4, 4, 8, 8, 8, 8]
    assert {s.kind for s in found} == {'solid'}
    assert sum(s.area for s in found) == pytest.approx(10 * INCH ** 2)


def test_a_curved_surface_is_one_surface_however_far_it_turns():
    """A cylinder is its side, its top and its bottom. Faces join across
    an edge while their normals are within the edge angle, so a coarse
    mesh of eight facets round (45 degrees apart) splits at 40 degrees
    and is one side again at 50."""
    project = visualdynamics.Project('p')
    name = project.new_geometry('Part', unit='in')
    for size in (0.25, 1.0):
        part = project.add_solid_cylinder(name, (4 * size, 0, 0), 1.0, 2.0,
                                          name=f'c{size}', unit='in')
        project.mesh_solids(name, [part], size, unit='in')
    geometry = project[name]
    group_of = dict(zip(geometry.group_name, (int(g) for g in geometry.group_id), strict=True))

    def faces(angle, group):
        return sorted(len(s.faces) for s in ins.skin(geometry, edge_angle=angle)
                      if s.group == group_of[group])

    assert faces(40, 'c0.25') == [128, 128, 256], 'the fine one: one side'
    assert faces(40, 'c1.0') == [2] * 8 + [12, 12], 'the coarse one: a strip per facet'
    assert faces(50, 'c1.0') == [12, 12, 16]


def test_coincident_nodes_never_merged_still_make_one_surface():
    """A block whose two halves have their own nodes at the seam, in one
    element group: the faces either side of the seam share no node,
    only places, and are still one surface."""
    block = _block()
    seam = np.flatnonzero(np.isclose(block.node_xyz[:, 0], 0.0))
    fresh = {int(block.node_id[r]): 1000 + k for k, r in enumerate(seam)}
    conn = []
    for nodes in block.elem_conn:
        rows = block.node_index([int(n) for n in nodes])
        right = block.node_xyz[rows, 0].mean() > 0
        conn.append([fresh.get(int(n), int(n)) if right else int(n) for n in nodes])
    split = Geometry(
        node_id=[*block.node_id.tolist(), *fresh.values()],
        node_xyz=np.vstack([block.node_xyz, block.node_xyz[seam]]),
        elem_type=block.elem_type, elem_conn=conn, elem_group=block.elem_group,
        group_id=block.group_id, group_name=block.group_name, length_unit=block.length_unit)
    assert ins.skin(split, edge_angle=40)[0].faces, 'it has a skin'
    assert sorted(len(s.faces) for s in ins.skin(split)) == [4, 4, 8, 8, 8, 8]


def test_sensors_go_only_on_the_surfaces_kept():
    geometry, shapes = _bar()
    found = ins.skin(geometry)
    top = next(s.id for s in found if _normal_of(s)[2] == 1)
    places = ins.candidates(geometry, surfaces=[top], found=found)
    assert set(places.surface.tolist()) == {top}
    assert np.allclose(places.frames[:, 2], [0, 0, 1]), 'all on the top, normal up'
    assert len(places.source) == 17 * 5, 'one place per node of the top'
    every = ins.candidates(geometry)
    assert sorted(set(every.surface.tolist())) == [s.id for s in found]
    kept = ins.plan(shapes, geometry, 4, 2, surfaces=[top])
    assert np.allclose(kept.geometry.node_xyz[:, 2] / INCH, 0.5)


def test_a_surface_stays_in_its_element_group():
    """Two blocks side by side, meshed together so they share the seam's
    nodes: their tops are flush and share an edge, and are still two
    surfaces, one per group, so that a group left unchecked takes its
    surfaces with it."""
    project = visualdynamics.Project('p')
    name = project.new_geometry('Part', unit='in')
    left = project.add_solid_block(name, (0, 0, 0), (1, 1, 1), name='left', unit='in')
    right = project.add_solid_block(name, (1, 0, 0), (1, 1, 1), name='right', unit='in')
    project.mesh_solids(name, [left, right], 0.5, unit='in')
    geometry = project[name]
    found = ins.skin(geometry)
    tops = [s for s in found if np.allclose(s.normals, [0, 0, 1])]
    assert len(tops) == 2 and len({s.group for s in tops}) == 2
    assert len(found) == 10, 'five outer faces each; the seam is inside'


def test_a_minus_label_reads_as_the_plus_one_turned_round():
    """A shape set labelled '101X-' measures along -X: the same block of
    numbers as '101X+' with its values negated."""
    geometry, shapes = _bar()
    places = ins.candidates(geometry)
    modes = ins.elastic(shapes)[:4]
    plus, _ = ins.local_shapes(shapes, modes, places, geometry)
    labels = [c[:-1] + '-' if c.endswith('X+') else c for c in shapes.coordinate]
    sign = np.array([-1.0 if c.endswith('X+') else 1.0 for c in shapes.coordinate])
    turned = ShapeSet(shapes.frequency, shapes.damping, labels, shapes.shape_matrix * sign,
                      modal_mass=shapes.modal_mass, mass_unit=shapes.mass_unit)
    minus, _ = ins.local_shapes(turned, modes, places, geometry)
    assert np.any(sign < 0)
    assert np.allclose(minus, plus)


def test_at_an_edge_the_larger_surface_keeps_the_nodes_own_id():
    """A 2 x 1 x 0.75 in block meshed at half an inch is one element
    thick, so a top edge node has 0.375 sq in of side face against
    0.25 of top: its place on the side keeps the node's id, and the one
    on the top is numbered above the model."""
    geometry = _block((2, 1, 0.75), 0.5)
    places = ins.candidates(geometry)
    xyz = places.geometry.node_xyz / INCH
    top_edge = (np.isclose(xyz[:, 2], xyz[:, 2].max()) & np.isclose(np.abs(xyz[:, 1]), 0.5)
                & (np.abs(xyz[:, 0]) < 1 - 1e-9))
    own = places.geometry.node_id == places.source
    normals = places.frames[:, 2]
    assert top_edge.sum() == 2 * 2 * 3
    assert np.allclose(np.abs(normals[top_edge & own, 1]), 1.0), 'the side keeps the id'
    assert np.allclose(normals[top_edge & ~own], [0, 0, 1]), 'the top is numbered above'


def test_fewer_directions_than_modes_plan_for_the_lowest():
    """Seven directions for eight modes resolve seven: the plan is made
    for the lowest seven by frequency, whatever order the modes come
    in, and is the plan those seven alone would get."""
    blocks = np.random.default_rng(11).normal(size=(12, 3, 8))
    lowest = np.array([3, 0, 7, 1, 5, 2, 6, 4])
    short = ins.effective_independence(blocks, 2, 1, lowest=lowest)
    assert short.planned.tolist() == sorted(lowest[:7].tolist())
    # the same seven, lowest first: the tri-axials chosen for the lowest six
    within = [short.planned.tolist().index(m) for m in lowest[:7]]
    alone = ins.effective_independence(blocks[:, :, short.planned], 2, 1, lowest=within)
    assert (short.triaxial, short.uniaxial) == (alone.triaxial, alone.uniaxial)


def test_the_tri_axial_pass_chooses_for_the_modes_it_can_resolve():
    """Two tri-axials measure six directions: with eight modes they are
    chosen for the lowest six, as the long way chooses them on those six
    alone, and the uni-axials then for all eight with them fixed. Chosen
    for all eight, the information matrix is singular at the end of the
    pass and rounding decides."""
    for seed in (0, 1, 2):
        blocks = np.random.default_rng(seed).normal(size=(9, 3, 8))
        placed = ins.effective_independence(blocks, 2, 3)
        assert sorted(placed.triaxial) == _naive(blocks[:, :, :6], 2, 0)[0]
        assert placed.planned.tolist() == list(range(8))
        base = sum(blocks[k].T @ blocks[k] for k in placed.triaxial)
        options = [(k, 2) for k in range(9) if k not in placed.triaxial]
        best = max(itertools.combinations(options, 3), key=lambda chosen: np.linalg.slogdet(
            base + sum(np.outer(blocks[k, a], blocks[k, a]) for k, a in chosen))[1])
        assert np.linalg.slogdet(placed.information)[1] >= np.linalg.slogdet(
            base + sum(np.outer(blocks[k, a], blocks[k, a]) for k, a in best))[1] - 1.0, (
            'the uni-axials are chosen for all eight modes')


def test_a_plan_short_of_directions_warns_and_keeps_every_mode():
    project, _name, modes = _bar_project()
    # at the edges too: a quarter of an inch in from them, the half-inch
    # bar's 1 in faces keep only their center lines
    with pytest.warns(ins.FewerDirectionsWarning,
                      match='Bar Modes: 7 directions for 8 modes: planned for the lowest 7'):
        added = project.plan_instrumentation(modes, 2, 1, edge_distance=0)
    kept = project[added['shapes']]
    assert kept.num_shapes == 8, 'every mode asked for, so the auto-MAC shows the cost'
    assert project[added['channels']].num_channels == 7


def test_the_lowest_are_by_frequency_however_the_modes_are_listed():
    geometry, shapes = _bar()
    backward = ins.elastic(shapes)[::-1]
    made = ins.plan(shapes, geometry, 2, 1, modes=backward, edge_distance=0.0)
    left_out = sorted(set(range(8)) - set(made.placement.planned.tolist()))
    assert [backward[k] for k in left_out] == [max(backward, key=lambda m: shapes.frequency[m])]


def _gap_to_edge(point):
    """How far a point on the 8 x 2 x 1 in bar's outside is from the
    edge of the face it is on, in inches, worked out by hand: the face
    is the one the point is on, and its edge the rectangle around it."""
    half = np.array([4.0, 1.0, 0.5])
    on = int(np.argmax(np.isclose(np.abs(point), half)))
    return min(half[a] - abs(point[a]) for a in range(3) if a != on)


def test_places_keep_the_distance_from_their_faces_edges():
    """At a quarter of an inch every place is at least that far from
    its face's edge, and every node that far in is still a place: none
    lost that should stay. Each face's edge is its own, so a node on a
    block edge is near both faces' edges and is no place on either."""
    geometry, _shapes = _bar()
    near = ins.candidates(geometry)
    away = ins.candidates(geometry, edge_distance=ins.EDGE_DISTANCE)
    gaps = [_gap_to_edge(p) for p in near.geometry.node_xyz / INCH]
    wanted = {(int(n), int(s)) for n, s, gap in zip(near.source, near.surface, gaps, strict=True)
              if gap >= 0.25 - 1e-9}
    assert {(int(n), int(s)) for n, s in zip(away.source, away.surface, strict=True)} == wanted
    assert min(_gap_to_edge(p) for p in away.geometry.node_xyz / INCH) >= 0.25 - 1e-9
    # the 8 x 2 faces keep 15 x 3 nodes, the 8 x 1 faces 15 x 1, the ends 3 x 1
    assert sorted(Counter(away.surface.tolist()).values()) == [3, 3, 15, 15, 45, 45]
    farther = ins.candidates(geometry, edge_distance=0.6 * INCH)
    assert sorted(Counter(farther.surface.tolist()).values()) == [13, 13], (
        'only the 8 x 2 faces have a node 0.6 in from every edge')


def test_a_surface_with_no_edge_keeps_every_place():
    """With the edge angle past 90 degrees the bar is one surface, a
    closed one: it has no edge to keep away from."""
    geometry, _shapes = _bar()
    whole = ins.candidates(geometry, edge_angle=120.0)
    away = ins.candidates(geometry, edge_angle=120.0, edge_distance=ins.EDGE_DISTANCE)
    assert len(whole.source) == len(away.source) > 0


def test_the_plan_keeps_a_quarter_inch_from_the_edges_unless_told():
    """The distance is typed in a unit like the project's other
    lengths, kept in meters, and a geometry whose units are not defined
    keeps none by default."""
    project, _name, modes = _bar_project()
    added = project.plan_instrumentation(modes, 2, 3, modes=[6, 7, 8, 9])
    placed = project[added['geometry']].node_xyz / INCH
    assert min(_gap_to_edge(p) for p in placed) >= 0.25 - 1e-9
    added = project.plan_instrumentation(modes, 2, 3, modes=[6, 7, 8, 9], name='Far',
                                         edge_distance=0.6, unit='in')
    assert project.journal[-1] == ("project.plan_instrumentation('Bar Modes', 2, 3, "
                                   "modes=[6, 7, 8, 9], name='Far', edge_distance=0.6, unit='in')")
    placed = project[added['geometry']].node_xyz / INCH
    assert min(_gap_to_edge(p) for p in placed) >= 0.6 - 1e-9
    assert np.isclose(project.provenance[added['geometry']]['params']['edge_distance'],
                      0.6 * INCH), 'kept in meters'
    geometry, shapes = _bar()
    geometry.length_unit = None                  # the same numbers, now in meters, read raw
    on_edges = ins.plan(shapes, geometry, 2, 3, modes=[6, 7, 8, 9]).geometry.node_xyz / INCH
    assert min(_gap_to_edge(p) for p in on_edges) < 0.25, 'no distance kept without units'


def _surface(number, area):
    return ins.Surface(number, 1, 'solid', [(1, 2, 3)], np.array([[0.0, 0.0, 1.0]]),
                       np.array([area]))


def test_small_surfaces_are_left_out_unless_picked():
    """A surface under 1 % of the outside's area starts out left out: a
    sliver, a fillet or a bore, where nobody mounts a sensor. When none
    is that large, all of them stay."""
    found = [_surface(1, 50.0), _surface(2, 30.0), _surface(3, 19.0),
             _surface(4, 0.99), _surface(5, 0.01)]
    assert ins.default_surfaces(found) == [1, 2, 3]
    assert ins.SMALL_SURFACE == 0.01
    even = [_surface(k, 1.0) for k in range(1, 201)]      # each 0.5 %
    assert ins.default_surfaces(even) == list(range(1, 201)), 'nothing left to plan on'


def test_a_small_part_starts_out_left_out():
    """A half-inch cube sitting on an 8 x 2 x 1 in bar, meshed apart: each
    of its five open faces is 0.25 of 52.75 sq in, under 1 %, so by
    default no place is on it; asked for, its places are there."""
    project = visualdynamics.Project('p')
    name = project.new_geometry('Part', unit='in')
    bar = project.add_solid_block(name, (0, 0, 0), (8, 2, 1), name='bar', unit='in')
    cube = project.add_solid_block(name, (0, 0, 0.75), (0.5, 0.5, 0.5), name='cube', unit='in')
    project.mesh_solids(name, [bar], 0.5, unit='in')
    project.mesh_solids(name, [cube], 0.5, unit='in', apart=True)
    geometry = project[name]
    found = ins.skin(geometry)
    group_of = dict(zip(geometry.group_name, (int(g) for g in geometry.group_id), strict=True))
    small = [s.id for s in found if s.group == group_of['cube']]
    assert len(small) == 5
    assert set(ins.default_surfaces(found)) == {s.id for s in found} - set(small)
    places = ins.candidates(geometry, found=found)
    assert not set(places.surface.tolist()) & set(small)
    every = ins.candidates(geometry, surfaces=[s.id for s in found], found=found)
    assert set(every.surface.tolist()) >= set(small)
