"""Cones and spheres, the solids added beside blocks, cylinders and
sections (Brandon, 2026-10-20: a few simple shapes to build parts
from, Add Hole off the bar). Their distances are exact, bricks are
fitted to them since no sweep makes either, and the pane adds and
edits them."""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core import solids, tetmesh
from visualdynamics.core.fem import MATERIALS, GroupProperties, Model
from visualdynamics.core.geometry import Geometry

INCH = 0.0254


def _empty():
    return Geometry(node_id=[], node_xyz=np.empty((0, 3)), length_unit='m')


def _volume(part):
    for group in part.group_id:
        part.group_properties[int(group)] = GroupProperties(MATERIALS['1018 steel'])
    return Model.from_geometry(part).structural_mass / MATERIALS['1018 steel'].density


def test_a_cones_distance_is_exact_and_its_sign_is_its_inside():
    """A frustum turned off the axes: the signed distance matches the
    distance to its surface sampled densely, within the sampling's own
    spacing; it is negative exactly where `contains` says inside; and
    its gradient is the distance's, by finite differences."""
    from scipy.spatial import cKDTree

    g = _empty()
    cone = solids.add_cone(g, (0.1, -0.2, 0.3), 1.0, 0.25, 2.0, angles=(20, 35, 0))
    item = solids.solid(g, cone)
    points = np.random.default_rng(1).uniform(-2.5, 2.5, (3000, 3)) + item.center
    d, grad = solids.distance_gradient(g, cone, points)
    zs, ts = np.linspace(-1, 1, 801), np.linspace(0, 2 * np.pi, 361)
    r = 1.0 - 0.75 * (zs + 1.0) / 2.0
    side = np.stack([np.outer(r, np.cos(ts)).ravel(), np.outer(r, np.sin(ts)).ravel(),
                     np.repeat(zs, len(ts))], axis=1)
    caps = []
    for z, rc in ((-1.0, 1.0), (1.0, 0.25)):
        rr, tt = np.meshgrid(np.linspace(0, rc, 121), ts)
        caps.append(np.stack([(rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel(),
                              np.full(rr.size, z)], axis=1))
    local = (points - np.asarray(item.center)) @ solids._frame(item).T
    brute = cKDTree(np.vstack([side, *caps])).query(local)[0]
    assert np.abs(np.abs(d) - brute).max() < 0.003
    assert np.array_equal(d <= 1e-9, solids.contains(g, cone, points))
    step = 1e-6
    fd = np.stack([(solids.distance(g, cone, points + step * e)
                    - solids.distance(g, cone, points - step * e)) / (2 * step)
                   for e in np.eye(3)], axis=1)
    assert np.linalg.norm(fd - grad, axis=1).max() < 1e-6


def test_a_spheres_distance_is_its_reach_less_its_radius():
    g = _empty()
    ball = solids.add_sphere(g, (1, 2, 3), 0.7)
    points = np.random.default_rng(2).uniform(-1, 1, (500, 3)) + (1, 2, 3)
    d, grad = solids.distance_gradient(g, ball, points)
    reach = np.linalg.norm(points - (1, 2, 3), axis=1)
    assert np.allclose(d, reach - 0.7)
    assert np.allclose(grad, (points - (1, 2, 3)) / reach[:, None])
    assert np.array_equal(d <= 1e-9, solids.contains(g, ball, points))
    low, high = solids._box(g, ball)
    assert np.allclose(low, (0.3, 1.3, 2.3)) and np.allclose(high, (1.7, 2.7, 3.7))


@pytest.mark.parametrize('numbers, said', [
    ((0.5, 0.5, 1.0), 'is a cylinder'),
    ((0.0, 0.0, 1.0), 'a radius at its base or its top'),
    ((-0.1, 0.5, 1.0), 'neither negative'),
    ((0.5, 0.2, 0.0), 'needs a height')])
def test_a_cone_that_is_no_cone_is_refused(numbers, said):
    with pytest.raises(ValueError, match=said):
        solids.add_cone(_empty(), (0, 0, 0), *numbers)


def test_a_sphere_of_no_radius_is_refused():
    with pytest.raises(ValueError, match='needs a radius'):
        solids.add_sphere(_empty(), (0, 0, 0), 0.0)


@pytest.mark.parametrize('kind', ['cone', 'sphere'])
def test_bricks_are_fitted_and_tetrahedra_fill_it(kind):
    """No sweep makes a cone or a sphere: bricks are fitted, within a
    percent and a half of its volume, and tetrahedra fill it to a
    hundredth of a percent."""
    g = _empty()
    if kind == 'cone':
        made = solids.add_cone(g, (0, 0, 0), 1.0, 0.4, 2.0, angles=(15, 0, 0), name='cone')
        exact = np.pi * 2.0 / 3.0 * (1.0 + 0.4 + 0.16)
    else:
        made = solids.add_sphere(g, (0, 0, 0), 1.0, name='ball')
        exact = 4.0 / 3.0 * np.pi
    with pytest.raises(solids.NotAnExtrusion, match=f'is a {kind}, which no sweep makes'):
        solids.extrusion_axis(g, [made])
    part, built = solids.brick_mesh(g, [made], 0.15)
    assert built['method'] == 'overlay' and built['quality'] > 0.1
    assert abs(_volume(part) - exact) / exact < 0.015
    tets, _made = tetmesh.tet_mesh(g, [made], 0.15)
    assert abs(_volume(tets) - exact) / exact < 1e-4


def test_a_ball_half_sunk_in_a_block_is_one_structure():
    """A sphere half in a block's top, meshed in pieces: the block swept,
    the ball fitted less the block, its seam tied, one structure."""
    import visualdynamics

    project = visualdynamics.Project('p')
    name = project.new_geometry('Part', unit='in')
    block = project.add_solid_block(name, (0, 0, 0), (4, 4, 1), name='block', unit='in')
    ball = project.add_solid_sphere(name, (0, 0, 0.5), 1.0, name='ball', unit='in')
    both = project.combine_solids(name, 'union', [block, ball], 'both')
    made = project.mesh_solids(name, [both], 0.25, unit='in', sizes={ball: 0.2})
    assert [m['method'] for m in made['pieces']] == ['extrusion', 'overlay']
    geometry = project[name]
    assert geometry.mates and made['ties'] + made['seam_shared'] > 20
    for group in geometry.group_id:
        geometry.group_properties[int(group)] = GroupProperties(MATERIALS['1018 steel'])
    assert len(Model.from_geometry(geometry).pieces()) == 1


def test_cones_and_spheres_edit_and_save(tmp_path):
    import visualdynamics

    project = visualdynamics.Project('p')
    name = project.new_geometry('Part', unit='in')
    cone = project.add_solid_cone(name, (0, 0, 0), 1.0, 0.5, 2.0, name='taper', unit='in')
    ball = project.add_solid_sphere(name, (3, 0, 0), 0.5, unit='in')
    project.edit_solid(name, cone, top_radius=0.0, unit='in')
    project.edit_solid(name, ball, radius=0.75, unit='in')
    geometry = project[name]
    assert solids._top(solids.solid(geometry, cone)) == 0.0
    assert solids.solid(geometry, ball).radius == pytest.approx(0.75 * INCH)
    with pytest.raises(ValueError, match='a sphere has no angles'):
        project.edit_solid(name, ball, angles=(10, 0, 0))
    path = tmp_path / 'g.vdyn'
    geometry.save(path)
    back = Geometry.load(path)
    assert back.solids == geometry.solids
    assert project.journal[-2] == (
        f"project.edit_solid('Part', {cone}, top_radius=0.0, unit='in')")


# ---- the pane ----

def test_add_cone_and_add_sphere_in_the_pane(window, pump):
    """Add Cone shows a base radius, a top radius and a height; Add
    Sphere a radius and no rotation, a sphere turned being the same
    sphere; each Add runs the project's verb, and a sphere, which has no
    sharp edge, draws."""
    pytest.importorskip('OCP')
    from test_solids_pane import _acts, _add, _part, _pick, _rows

    geometry = _part(window, pump)
    panel = _add(window, pump, 'add_solid_cone', radius=0.5, top=0.0, length=2.0)
    assert panel.title.text() == 'Add Cone'
    assert panel.radius_label.text() == 'Base radius [in]'
    assert panel.top_label.text() == 'Top radius [in]' and not panel.top.isHidden()
    assert panel.length_label.text() == 'Height [in]'
    assert panel.reading_label.text().startswith('A cone 2 in high, 1 in across at its base '
                                                 'and 0 in at its top')
    panel.add_button.click()
    pump()
    assert window.project.journal[-1] == (
        "project.add_solid_cone('Part', (0.0, 0.0, 0.0), 0.5, 0.0, 2.0, (0.0, 0.0, 0.0), "
        "'', unit='in')")
    panel = _add(window, pump, 'add_solid_cone', radius=0.5, top=0.5)
    assert 'is a cylinder' in panel.reading_label.text()
    assert not panel.add_button.isEnabled()
    window.close_mesh_pane()
    pump()
    panel = _add(window, pump, 'add_solid_sphere', radius=0.75)
    assert panel.title.text() == 'Add Sphere'
    assert all(box.isHidden() for box in panel.angles) and panel.angle_label.isHidden()
    assert panel.top.isHidden() and panel.length.isHidden()
    panel.add_button.click()
    pump()
    assert window.project.journal[-1] == (
        "project.add_solid_sphere('Part', (0.0, 0.0, 0.0), 0.75, '', unit='in')")
    assert [s.kind for s in geometry.solids] == ['cone', 'sphere']
    window.close_mesh_pane()
    pump()
    # picked, the sphere draws: its surface, and no edge lines, having none
    rows = _rows(window, 'solid')
    assert len(rows) == 2
    _pick(window, pump, [rows[1]])
    sphere = geometry.solids[1].id
    assert f'solid-{sphere}' in window.scene.plotter.actors
    assert f'solid-edges-{sphere}' not in window.scene.plotter.actors
    _pick(window, pump, [rows[0]])
    cone = geometry.solids[0].id
    assert f'solid-edges-{cone}' in window.scene.plotter.actors, 'a cone has its rims'
    _pick(window, pump, _rows(window, 'solids'))
    assert 'add_solid_hole' not in _acts(window)


def test_a_cones_pencil_changes_its_top_radius(window, pump):
    pytest.importorskip('OCP')
    from test_solids_pane import _part, _pencil, _rows

    geometry = _part(window, pump)
    window.project.add_solid_cone('Part', (0, 0, 0), 1.0, 0.5, 2.0, unit='in')
    window._solids_changed('Part')
    pump()
    _pencil(window, pump, _rows(window, 'solid')[0])
    panel = window.scene.mesh_panel
    assert panel.title.text() == 'Edit Cone' and panel.top.value() == pytest.approx(0.5)
    panel.set_values(top=0.25)
    pump()
    panel.add_button.click()
    pump()
    assert window.project.journal[-1] == (
        "project.edit_solid('Part', 1, top_radius=0.25, unit='in')")
    assert solids._top(geometry.solids[0]) == pytest.approx(0.25 * INCH)
