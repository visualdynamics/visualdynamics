"""The structured mesher's holes, pinned by value where a mutation pass
found them checked only for being there (2026-10-11).

`tools/mutate.py` over mesh.py found the round rim of a hole — the
reason a hole's size stops wandering with where its center falls — and
the slide of a rim node along the block's face untested: the hole
tests counted bricks and looked for the nearest center, which a
stair-stepped hole passes too. The blocks here are whole numbers of
unit bricks in no unit, so every brick and every rim node is known.
Since the same day a hole with room is a ring mesh, Mesh Solids' own
(`solids._mesh_grid`): centered on a square of unit bricks its circle's
nodes are a regular sixteen-sided polygon, and a hole's volume is that
polygon's, exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core import mesh


def _block(dims, holes=(), **kw):
    a, b, c = dims
    return mesh.block((0, 0, 0), (a, 0, 0), (0, b, 0), (0, 0, c), 1.0, 'part',
                      unit=None, holes=holes, **kw)


def _volume(geometry):
    """The bricks' volume: each split into six tetrahedra about its
    first diagonal, exact for bricks with flat faces."""
    conn = np.array(geometry.elem_conn)
    points = geometry.node_xyz[geometry.node_index(conn.ravel())].reshape(len(conn), 8, 3)
    total = 0.0
    for a, b, c, d in ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6),
                       (0, 4, 5, 6), (0, 5, 1, 6)):
        total += np.abs(np.einsum('mi,mi->m', points[:, b] - points[:, a], np.cross(
            points[:, c] - points[:, a], points[:, d] - points[:, a]))).sum() / 6
    return total


def _polygon(radius, sides=16):
    """A regular polygon's area, its corners on the circle."""
    return sides / 2.0 * radius ** 2 * np.sin(2.0 * np.pi / sides)


def _centers(geometry):
    return np.array([geometry.node_xyz[geometry.node_index(c)].mean(axis=0)
                     for c in geometry.elem_conn])


def test_a_hole_inside_the_block_is_a_ring():
    """Through a 6 x 6 x 1 block at (3, 3), radius 1.5: the ring takes
    the 4 x 4 bricks about the axis (out to the grid line past a quarter
    radius or half a brick beyond the circle), and puts as many nodes on
    the circle as that square's edge has, 16 in each of the two layers,
    none inside. Nothing outside the square moves, and the hole is the
    sixteen-sided polygon exactly."""
    holed = _block((6, 6, 1), [((3.0, 3.0, 0.0), 1.5, 2)])
    xy = holed.node_xyz[:, :2] - [3.0, 3.0]
    radial = np.linalg.norm(xy, axis=1)
    assert np.isclose(radial, 1.5).sum() == 32 and radial.min() == pytest.approx(1.5)
    outside = np.abs(xy).max(axis=1) > 2.0 + 1e-9
    assert outside.sum() == 2 * (49 - 25)
    assert np.allclose(holed.node_xyz[outside], np.round(holed.node_xyz[outside]))
    assert _volume(holed) == pytest.approx(36 - _polygon(1.5))


def test_a_rim_node_on_the_blocks_face_slides_along_it_to_the_circle():
    """A hole at (3, 0.5) breaks out of the face y = 0: six bricks go
    (four along the face, two above), and the rim's two nodes on that
    face stay on it, where the circle crosses it — x = 3 -+ sqrt(2.25
    - 0.25) — rather than moving off the block radially."""
    holed = _block((6, 6, 1), [((3.0, 0.5, 0.0), 1.5, 2)])
    assert len(holed.elem_conn) == 36 - 6
    xyz = holed.node_xyz
    radial = np.linalg.norm(xyz[:, :2] - [3.0, 0.5], axis=1)
    face = np.isclose(xyz[:, 1], 0.0) & np.isclose(radial, 1.5)
    assert sorted(set(np.round(xyz[face, 0], 12))) == pytest.approx(
        [3.0 - np.sqrt(2.0), 3.0 + np.sqrt(2.0)])
    assert face.sum() == 4, 'two on each of the two layers'
    assert xyz[:, 1].min() == 0.0, 'nothing pulled off the face'


def test_a_hole_on_the_blocks_edge_is_a_round_over_and_keeps_its_steps():
    stepped = _block((6, 6, 1), [((0.0, 0.0, 0.0), 1.5, 2)])
    assert len(stepped.elem_conn) == 36 - 1, 'the corner brick alone'
    assert np.allclose(stepped.node_xyz, np.round(stepped.node_xyz))


@pytest.mark.parametrize('center, kept_below', [
    ((2.0, 2.0, 0.0), False),        # from the corner's face: the bottom two layers
    ((2.0, 2.0, 4.0), True),         # from the far face: the top two
])
def test_a_blind_hole_goes_its_depth_from_the_face_it_enters(center, kept_below):
    blind = _block((4, 4, 4), [(center, 0.8, 2, 2.0)])
    assert _volume(blind) == pytest.approx(64 - 2 * _polygon(0.8))
    centers = _centers(blind)
    column = np.linalg.norm(centers[:, :2] - [2.0, 2.0], axis=1) < 0.8
    assert sorted(set(centers[column, 2])) == ([0.5, 1.5] if kept_below else [2.5, 3.5])
    floor = blind.node_xyz[np.isclose(blind.node_xyz[:, 0], 2.0)
                           & np.isclose(blind.node_xyz[:, 1], 2.0)]
    assert sorted(floor[:, 2]) == ([0.0, 1.0, 2.0] if kept_below else [2.0, 3.0, 4.0]), (
        'the floor of the hole stays where it is, inside the circle')


def test_a_hole_names_the_group_its_bricks_go_to():
    named = _block((6, 6, 1), [((3.0, 3.0, 0.0), 1.5, 2, None, 'insert')],
                   hole_name='other')
    assert list(named.group_name) == ['part', 'insert']
    insert = named.elements_in('insert')
    assert _volume(named) == pytest.approx(36)
    rows = np.isin(named.elem_id, insert)
    inside = _centers(named)[rows, :2] - [3.0, 3.0]
    assert np.linalg.norm(inside, axis=1).max() < 1.5, 'the insert fills the circle'
    void = _block((6, 6, 1), [((3.0, 3.0, 0.0), 1.5, 2, None, None)],
                  hole_name='other')
    assert list(void.group_name) == ['part']
    assert _volume(void) == pytest.approx(36 - _polygon(1.5))
    # a hole that cuts nothing adds no group: less than a brick across,
    # it gets no ring, and no brick's center falls in it
    missed = _block((6, 6, 1), [((3.0, 3.0, 0.0), 0.1, 2)], hole_name='other')
    assert list(missed.group_name) == ['part'] and len(missed.elem_conn) == 36


def test_the_element_size_and_the_edges_are_checked_by_name():
    for build, what in ((_rebuild_block, 'a block'), (_rebuild_plane, 'a plane')):
        with pytest.raises(ValueError, match=f'{what}: the element size must be positive'):
            build(size=0.0)
        with pytest.raises(ValueError, match=f'{what}: the element size must be positive'):
            build(size=-1.0)
        with pytest.raises(ValueError, match=f'{what}: an edge has no length'):
            build(edge=(0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match=f'{what}: the edges are not perpendicular'):
            build(edge=(0.001, 1.0, 0.0))
        build(edge=(1e-12, 1.0, 0.0))                  # square to round-off


def _rebuild_block(size=1.0, edge=(0.0, 1.0, 0.0)):
    return mesh.block((0, 0, 0), (2, 0, 0), edge, (0, 0, 1), size, unit=None)


def _rebuild_plane(size=1.0, edge=(0.0, 1.0, 0.0)):
    return mesh.plane((0, 0, 0), (2, 0, 0), edge, size, unit=None)


def test_a_plane_starts_at_its_corner_and_runs_along_its_edges():
    sheet = mesh.plane((1.0, 2.0, 3.0), (2.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0,
                       unit=None)
    assert sheet.node_xyz.tolist()[:4] == [[1, 2, 3], [2, 2, 3], [3, 2, 3], [1, 2, 4]]
    assert set(sheet.elem_type.tolist()) == {44}


def _square(x0, unit=None, name='sheet'):
    return mesh.plane((x0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0,
                      name, unit=unit)


def test_parts_are_one_node_within_a_millionth_of_the_assemblys_size():
    """Two unit squares side by side span sqrt(5): nodes a millionth of
    that apart are one, three millionths apart are two."""
    near = mesh.assemble(_square(0.0), _square(1.0 + 1e-6))
    assert near.num_nodes == 6
    far = mesh.assemble(_square(0.0), _square(1.0 + 3e-6))
    assert far.num_nodes == 8
    lands, _rows = mesh.landing(_square(0.0), _square(1.0 + 1e-6))
    assert lands.tolist() == [True, False, True, False]
    lands, _rows = mesh.landing(_square(0.0), _square(1.0), tolerance=0.0)
    assert lands.sum() == 2, 'a node exactly on another lands at no tolerance'


def test_parts_in_different_units_are_refused_by_naming_them():
    with pytest.raises(ValueError, match='different length units: in, m'):
        mesh.assemble(_square(0.0, 'm'), _square(1.0, 'in'))
    assert mesh.assemble(_square(0.0, 'in')).length_unit == 'in'


def test_a_geometry_with_no_unit_takes_its_first_parts():
    from visualdynamics import Geometry

    empty = Geometry(node_id=[], node_xyz=np.empty((0, 3)), length_unit=None)
    mesh.join(empty, _square(0.0, 'mm'))
    assert empty.length_unit == 'mm'


def test_a_tie_names_the_element_that_is_not_there_and_links_rigidly():
    sheet = mesh.assemble(_square(0.0, name='a'),
                          mesh.plane((0, 0, 0.5), (1, 0, 0), (0, 1, 0), 1.0, 'b',
                                     unit=None))
    with pytest.raises(ValueError, match='no element 99'):
        mesh.tie(sheet, [1, 99], 'a')
    sheet.add_group('bolts')                     # a group with no properties yet
    found = mesh.tie(sheet, sheet.elements_in('b'), 'a', group='bolts')
    rows = np.flatnonzero(np.isin(sheet.elem_id, sheet.elements_in('bolts')))
    assert found['links'] == 4 and set(sheet.elem_type[rows].tolist()) == {21}


def test_holes_along_two_edges_are_refused():
    """The bricks are a grid swept along the holes' edge (one mesher
    with Mesh Solids, 2026-10-11), so a block's holes all run along one;
    a hole along another is refused by name rather than left square."""
    with pytest.raises(ValueError, match='part: every hole must run along the same edge'):
        _block((6, 6, 6), [((3.0, 3.0, 0.0), 1.0, 2), ((3.0, 0.0, 3.0), 1.0, 1)])
    along_x = _block((6, 6, 6), [((0.0, 3.0, 3.0), 1.5, 0)])
    assert _volume(along_x) == pytest.approx(216 - 6 * _polygon(1.5))
