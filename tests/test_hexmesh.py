"""Bricks for any shape: a grid fitted to the recipe (`core.hexmesh`),
which Bricks falls through to when the parts are not an extrusion
(`solids.brick_mesh`). Brandon, 2026-10-19: *give the better / more
generic hex meshing a solid effort.*
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_solids_mesh import INCH, _corner_volumes, _points, _project, _volume
from test_tetmesh import _angled

from visualdynamics.core import hexmesh, solids


def _scaled_jacobians(geometry):
    xyz = np.asarray(geometry.node_xyz)
    conn = np.asarray(geometry.elem_conn) - 1
    return hexmesh._scaled_jacobians(xyz, conn)


def test_a_block_is_the_grid_itself_and_its_pillow():
    """Nothing to fit: the bricks stay cubes, the pillow layer under
    the skin is the only change, every corner is exactly where the
    recipe puts it, and the volume is exact."""
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (2, 1, 1), name='bar', unit='in')
    part, made = hexmesh.overlay_mesh(geometry, [block], 0.25 * INCH)
    xyz = np.asarray(part.node_xyz) / INCH
    for corner in [[sx, sy, sz] for sx in (-1, 1) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)]:
        assert np.linalg.norm(xyz - corner, axis=1).min() < 1e-9
    assert (_corner_volumes(part) > 0).all()
    assert _volume(part) == pytest.approx(2.0, abs=1e-9)
    assert made['quality'] > 0.5 and made['poor'] == 0 and made['uncaptured'] == 0
    assert list(part.group_name) == ['bar'] and set(part.elem_type) == {115}
    assert np.abs(xyz[:, 0]).max() <= 1.0 + 1e-9, 'no node outside the block'


def test_a_block_with_an_angled_hole_is_fitted_round_and_sharp():
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (4, 2, 1), name='slab', unit='in')
    tilted = project.add_solid_cylinder(name, (0, 0, 0), 0.4, 3.0, angles=(30, 0, 0),
                                        unit='in')
    cut = project.combine_solids(name, 'subtract', [block, tilted], 'slab')
    with pytest.raises(solids.NotAnExtrusion):
        solids.extrusion_mesh(geometry, [cut], 0.125 * INCH)
    part, made = solids.brick_mesh(geometry, [cut], 0.125 * INCH)
    assert made['method'] == 'overlay'
    sj = _scaled_jacobians(part)
    assert (sj > 0).all(), 'every brick the right way out'
    assert made['quality'] == pytest.approx(sj.min()) and sj.min() > 0.1
    assert made['poor'] < 0.02 * made['bricks']
    xyz = np.asarray(part.node_xyz) / INCH
    # the block's corners are captured exactly
    for corner in [[sx * 2, sy, sz * 0.5] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]:
        assert np.linalg.norm(xyz - corner, axis=1).min() < 1e-9
    # the hole is round: no node inside it, and nodes on its wall
    axis = solids._frame(solids.solid(geometry, tilted))[2]
    radial = np.linalg.norm(xyz - np.outer(xyz @ axis, axis), axis=1)
    assert not (radial < 0.4 - 1e-6).any()
    assert np.isclose(radial, 0.4, atol=1e-6).sum() >= 100
    # the volume is the recipe's: the block less the hole's slanted bore
    volume = _volume(part)
    # every brick's center in the material, a chord's sag allowed: a
    # brick lying flat against a curved wall has its center a few
    # thousandths of a size past the chord
    centers = _points(part).mean(axis=1) * INCH
    assert (solids.distance(geometry, cut, centers) < 0.02 * 0.125 * INCH).all()
    expected = 8.0 - math.pi * 0.4 ** 2 * 1.0 / math.cos(math.radians(30))
    assert volume == pytest.approx(expected, rel=0.01)


def test_brandons_angled_block_is_bricks_at_a_quarter_inch_and_refused_at_a_half():
    project, name, geometry = _project()
    cut, _union = _angled(project, name)
    with pytest.raises(ValueError, match='folded fitting the shape at that size'):
        hexmesh.overlay_mesh(geometry, [cut], 0.5 * INCH)
    part, made = hexmesh.overlay_mesh(geometry, [cut], 0.3 * INCH)
    # 0.18: the projection in two stages gives 0.21 here, in one 0.16
    assert (_scaled_jacobians(part) > 0).all() and made['quality'] > 0.18
    centers = _points(part).mean(axis=1) * INCH
    assert (solids.distance(geometry, cut, centers) < 0.02 * 0.3 * INCH).all()
    volume = _volume(part)
    assert 176.0 < volume < 180.0


def test_the_fitting_tells_its_progress_and_refuses_what_the_others_refuse():
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (2, 1, 1), unit='in')
    seen = []
    hexmesh.overlay_mesh(geometry, [block], 0.3 * INCH, progress=lambda d, t: seen.append((d, t)))
    assert seen[-1][0] == seen[-1][1] and [d for d, _t in seen] == sorted(d for d, _t in seen)
    with pytest.raises(ValueError, match='above zero'):
        hexmesh.overlay_mesh(geometry, [block], 0.0)
    with pytest.raises(ValueError, match='use a larger one'):
        hexmesh.overlay_mesh(geometry, [block], 0.001 * INCH)
    assert hexmesh.grid_cells(geometry, [block], 0.5 * INCH) == 4 * 2 * 2


def test_an_extrusion_keeps_its_rings_through_brick_mesh():
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, 0), (4, 2, 0.5), name='plate', unit='in')
    hole = project.add_solid_cylinder(name, (0, 0, 0), 0.3, 1.0, unit='in')
    cut = project.combine_solids(name, 'subtract', [plate, hole], 'plate')
    _part, made = solids.brick_mesh(geometry, [cut], 0.1 * INCH)
    assert made['method'] == 'extrusion' and made['rings'] == 1


def test_the_project_and_the_pane_mesh_a_tilted_hole_into_bricks(window, pump):
    pytest.importorskip('OCP')
    from conftest import select_objects
    from test_solids_mesh import _settled
    from test_solids_pane import _acts, _pick, _rows

    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    project = window.project
    name = project.new_geometry('Part', unit='in')
    block = project.add_solid_block(name, (0, 0, 0), (2, 1, 1), name='slab', unit='in')
    tilted = project.add_solid_cylinder(name, (0, 0, 0), 0.4, 3.0, angles=(30, 0, 0),
                                        unit='in')
    project.combine_solids(name, 'subtract', [block, tilted], 'slab')
    window.show_object(name)
    select_objects(window, pump, name)
    window._solids_changed(name)
    pump()
    _pick(window, pump, _rows(window, 'solid')[:1])
    _acts(window)['mesh_solids'][3]()
    pump()
    panel = window.scene.mesh_solids_panel
    panel.size.setValue(0.125)
    _settled(window, pump)
    reading = panel.reading_label.text()
    assert 'on a grid fitted to the shape (not an extrusion)' in reading
    assert 'The worst scaled Jacobian is' in reading and panel.mesh_button.isEnabled()
    panel.mesh_button.click()
    pump()
    geometry = window.objects[name]
    assert set(geometry.elem_type) == {115} and list(geometry.group_name) == ['slab']
    assert window.statusBar().currentMessage().startswith(
        f'{name}: meshed into {len(geometry.elem_conn):,} bricks')
    # too coarse to fit: the reading says so and offers tetrahedra
    panel2 = window.scene.mesh_solids_panel
    _pick(window, pump, _rows(window, 'solid')[:1])
    _acts(window)['mesh_solids'][3]()
    pump()
    panel2.size.setValue(0.5)
    _settled(window, pump)
    assert 'folded fitting the shape at that size' in panel2.reading_label.text()
    assert panel2.reading_label.text().endswith('Pick Tets to mesh it.')
    assert not panel2.mesh_button.isEnabled()
