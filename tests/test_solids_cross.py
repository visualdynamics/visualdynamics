"""A hole drilled across the extrusion: Brandon's block with a peanut
cut along Z and a 2 in hole in from the side along Y (2026-10-18). The
extrusion runs along the axis most of the round features run along;
the hole across it is cut from the grid and pulled round in its own
plane, the peanut keeps its ring, and where the two walls meet the
nodes sit on both.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_solids_mesh import INCH, _corner_volumes, _points, _project

from visualdynamics.core import solids


def _cross_drilled(project, name, radius=1.0):
    """The block, the peanut (3 in and 2 in cylinders along Z, 1.5 in
    off center each way) and a hole of `radius` along Y from the
    peanut out through the block's side, all cut from the block."""
    block = project.add_solid_block(name, (0, 0, 0), (10, 10, 3), unit='in')
    left = project.add_solid_cylinder(name, (-1.5, 0, 0), 3.0, 3.0, unit='in')
    right = project.add_solid_cylinder(name, (1.5, 0, 0), 2.0, 3.0, unit='in')
    across = project.add_solid_cylinder(name, (0, 3, 0), radius, 5.0,
                                        angles=(-90, 0, 0), unit='in')
    holes = project.combine_solids(name, 'union', [left, right, across])
    return project.combine_solids(name, 'subtract', [block, holes]), holes


def _nodes(geometry):
    return np.asarray(geometry.node_xyz) / INCH


def _on_peanut(xyz, tol=1e-9):
    inner = np.abs(np.hypot(xyz[:, 0] + 1.5, xyz[:, 1]) - 3.0)
    outer = np.abs(np.hypot(xyz[:, 0] - 1.5, xyz[:, 1]) - 2.0)
    return np.minimum(inner, outer) < tol


def test_a_hole_across_the_extrusion_is_meshed_and_the_peanut_keeps_its_ring():
    project, name, geometry = _project()
    cut, holes = _cross_drilled(project, name)
    made = project.mesh_solids(name, [cut], 0.5, unit='in')
    assert made['axis'] == 2, 'two cylinders along Z, one along Y'
    assert made['rings'] == 1, 'the peanut is ringed with the hole through its rectangle'
    # the hole across is cut: no brick's center inside any of the holes
    centers = _points(geometry).mean(axis=1) * INCH
    assert not solids.contains(geometry, holes, centers).any()
    assert (_corner_volumes(geometry) > 0).all(), 'every brick the right way out'
    # clear of the peanut, the hole along Y is round: eight nodes on its
    # circle in every section, and the 45-degree nodes, which the pull
    # cannot bring the whole way without folding a brick (four bricks
    # across), are within a tenth of the radius of it
    xyz = _nodes(geometry)
    clear = xyz[xyz[:, 1] > 3.5]
    r = np.hypot(clear[:, 0], clear[:, 2])
    assert int(np.isclose(r, 1.0, atol=1e-6).sum()) >= 8 * 4
    assert r.min() > 0.9
    # the two walls meet on a curve, and nodes sit on both
    both = _on_peanut(xyz) & np.isclose(np.hypot(xyz[:, 0], xyz[:, 2]), 1.0, atol=1e-6)
    assert int(both.sum()) >= 4


def test_the_pull_across_leaves_no_node_off_the_peanuts_wall(monkeypatch):
    """The same mesh with the hole across cut but not pulled round, node
    for node: the pull moves nodes, and every node on the peanut's arc
    before it is on the arc after it (`_rejoin`)."""
    project, name, geometry = _project()
    cut, _holes = _cross_drilled(project, name)
    project.mesh_solids(name, [cut], 0.5, unit='in')
    pulled = _nodes(geometry)
    ids = list(geometry.node_id)
    project2, name2, geometry2 = _project()
    cut2, _holes2 = _cross_drilled(project2, name2)
    monkeypatch.setattr(solids, '_cross_circles', lambda *_args: {})
    project2.mesh_solids(name2, [cut2], 0.5, unit='in')
    plain = _nodes(geometry2)
    assert list(geometry2.node_id) == ids
    moved = np.any(pulled != plain, axis=1)
    assert moved.sum() > 50, 'the pull across moved nodes'
    assert (_on_peanut(plain) == _on_peanut(pulled)).all()


def test_the_extrusion_runs_along_the_axis_most_round_features_do():
    project, name, geometry = _project()
    z1 = project.add_solid_cylinder(name, (0, 0, 0), 0.2, 1.0, unit='in')
    z2 = project.add_solid_cylinder(name, (2, 0, 0), 0.2, 1.0, unit='in')
    y1 = project.add_solid_cylinder(name, (4, 0, 0), 0.5, 1.0, angles=(90, 0, 0),
                                    unit='in')
    assert solids.extrusion_axis(geometry, [z1, z2, y1]) == 2
    y2 = project.add_solid_cylinder(name, (6, 0, 0), 0.2, 1.0, angles=(90, 0, 0),
                                    unit='in')
    assert solids.extrusion_axis(geometry, [z1, y1, y2]) == 1
    # a tie goes to the widest
    assert solids.extrusion_axis(geometry, [z1, y1]) == 1
    assert solids.extrusion_axis(geometry, [z2, y2]) == 1, 'equal every way: the lower axis'
    turned = project.add_solid_cylinder(name, (8, 0, 0), 0.2, 1.0, angles=(30, 0, 0),
                                        unit='in')
    with pytest.raises(ValueError, match=f'solid {turned} is turned off'):
        solids.extrusion_axis(geometry, [z1, turned])


def test_contains_can_read_a_recipe_without_a_cylinder():
    """What the rings are laid from: the part with the cylinders across
    the extrusion read as holding nothing."""
    project, name, geometry = _project()
    cut, _holes = _cross_drilled(project, name)
    across = [leaf.id for leaf in solids._leaves(geometry, cut)
              if leaf.kind == 'cylinder' and abs(solids._frame(leaf)[2][1]) > 0.5]
    assert len(across) == 1
    at = np.array([[0.0, 4.0, 0.0]]) * INCH         # in the hole along Y
    assert not solids.contains(geometry, cut, at)[0]
    assert solids.contains(geometry, cut, at, ignore=across)[0]
    inside_peanut = np.array([[-1.5, 0.0, 0.0]]) * INCH
    assert not solids.contains(geometry, cut, inside_peanut, ignore=across)[0]
