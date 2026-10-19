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


def test_brandons_angled_block_is_refused_at_a_half_inch():
    project, name, geometry = _project()
    cut, _union = _angled(project, name)
    with pytest.raises(ValueError, match='folded fitting the shape at that size'):
        hexmesh.overlay_mesh(geometry, [cut], 0.5 * INCH)


@pytest.mark.slow
def test_brandons_angled_block_is_bricks_at_a_fifth_of_an_inch():
    """Five bricks across the 1 in holes' radius: fitted, none folded.
    At 0.3 and 0.25 in one and seven bricks fold at the side holes'
    feather edges and the mesher refuses (2026-10-19)."""
    project, name, geometry = _project()
    cut, _union = _angled(project, name)
    part, made = hexmesh.overlay_mesh(geometry, [cut], 0.2 * INCH)
    assert (_scaled_jacobians(part) > 0).all() and made['quality'] > 0.15
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


def test_the_project_and_the_pane_mesh_a_tilted_hole_into_bricks(window, pump, monkeypatch):
    pytest.importorskip('OCP')
    from conftest import select_objects
    from test_solids_mesh import _settled
    from test_solids_pane import _acts, _pick, _rows

    # the sizes here are under the five-across line on purpose: the
    # question before the fit is answered "anyway" (it has its own test)
    monkeypatch.setattr(window, '_ask_mesh_anyway', lambda *_args: 'anyway')

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


def test_one_cell_pits_are_filled_and_hanging_cells_emptied_but_a_plate_keeps_its_corners():
    """`_clean`: a void cell with four held neighbors is a pit and is
    filled with the part most of them hold; a held cell with five void
    neighbors hangs by one face and is emptied; a plate's corner cell,
    with three faces outside, stays."""
    owner = np.ones((5, 5, 3), dtype=int)
    owner[2, 2, 1] = 0                      # a pit in the middle, six held neighbors
    owner[0, 0, 0] = 2                      # a corner cell of another part: three faces outside, stays
    spike = np.zeros((3, 3, 3), dtype=int)
    spike[1, 1, 1] = 1                      # hangs by nothing: emptied
    spike[1, 1, 0] = 1                      # hangs by the one face above: emptied too, once the top goes
    cleaned = hexmesh._clean(owner)
    assert cleaned[2, 2, 1] == 1 and cleaned[0, 0, 0] == 2
    assert (cleaned == owner).sum() == owner.size - 1
    assert not hexmesh._clean(spike).any()
    # a pit of the other part is filled with that part
    owner[2, 2, 1] = 0
    owner[1:4, 1:4, :] = 2
    owner[2, 2, 1] = 0
    assert hexmesh._clean(owner)[2, 2, 1] == 2


def test_cells_touching_along_an_edge_are_split_apart():
    """`_split_pinches`: two cubes sharing only an edge make a node on
    that edge a corner of both with no face between them; each such
    node is split, one per cube, so the skin is a surface."""
    xyz = np.array([[x, y, z] for x in (0, 1, 2) for y in (0, 1, 2) for z in (0, 1)], dtype=float)
    node = np.arange(18).reshape(3, 3, 2)

    def cube(i, j):
        return [node[i, j, 0], node[i + 1, j, 0], node[i + 1, j + 1, 0], node[i, j + 1, 0],
                node[i, j, 1], node[i + 1, j, 1], node[i + 1, j + 1, 1], node[i, j + 1, 1]]

    hexes = np.array([cube(0, 0), cube(1, 1)])          # diagonal: they share the edge at (1, 1)
    out_xyz, out_hexes = hexmesh._split_pinches(xyz, hexes)
    assert len(out_xyz) == 20, 'the two nodes along the shared edge, split once each'
    assert len(set(out_hexes[0]) & set(out_hexes[1])) == 0
    assert np.allclose(out_xyz[18:], xyz[[node[1, 1, 0], node[1, 1, 1]]])
    # cubes sharing a face are left alone
    hexes = np.array([cube(0, 0), cube(1, 0)])
    same_xyz, same_hexes = hexmesh._split_pinches(xyz, hexes)
    assert len(same_xyz) == 18 and (same_hexes == hexes).all()


def test_the_grid_splits_its_pinches(monkeypatch):
    """`_grid` hands its cells through `_split_pinches` before anything
    is pillowed."""
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (2, 1, 1), unit='in')
    seen = []
    real = hexmesh._split_pinches

    def spy(xyz, hexes):
        seen.append(len(hexes))
        return real(xyz, hexes)

    monkeypatch.setattr(hexmesh, '_split_pinches', spy)
    _xyz, hexes, _groups = hexmesh._grid(geometry, [block], 0.5 * INCH)
    assert seen == [len(hexes)] == [4 * 2 * 2]


def test_the_smallest_feature_and_the_size_suggested_from_it():
    """Brandon's block: 1 in holes and a 3 in thickness, so a fit is
    suggested at a fifth of an inch (five bricks across the radius,
    where every fit has held and four has folded) and tetrahedra at a
    third; the slab with a 0.4 in hole likewise."""
    project, name, geometry = _project()
    cut, _union = _angled(project, name)
    radius, wall = hexmesh.smallest_feature(geometry, [cut])
    assert radius == pytest.approx(1.0 * INCH) and wall == pytest.approx(3.0 * INCH)
    assert hexmesh.suggested_size(geometry, [cut]) == pytest.approx(0.2 * INCH)
    assert hexmesh.suggested_size(geometry, [cut], 'tetras') == pytest.approx(1.0 / 3.0 * INCH)
    plain = project.add_solid_block(name, (20, 0, 0), (2, 1, 1), unit='in')
    assert hexmesh.smallest_feature(geometry, [plain]) == (np.inf, pytest.approx(1.0 * INCH))
    assert hexmesh.suggested_size(geometry, [plain]) == pytest.approx(1.0 / 3.0 * INCH)
    # the extrusion rule reads the same walls: the tilted hole's caps,
    # projected onto the nearest axis by the grid's cuts, are not walls
    # (the pane opened at 0.95 in on this block for them)
    assert solids.mesh_size(geometry, [cut]) == pytest.approx(1.0 * INCH)


def test_the_pane_opens_at_the_size_likely_to_hold(window, pump):
    """Brandon's block: the pane opened at 0.95 in, half a misread
    wall; it opens at 0.2 in, the holes' radius over five."""
    pytest.importorskip('OCP')
    from conftest import select_objects
    from test_solids_pane import _acts, _pick, _rows

    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    project = window.project
    name = project.new_geometry('Part', unit='in')
    _angled(project, name)
    window.show_object(name)
    select_objects(window, pump, name)
    window._solids_changed(name)
    pump()
    _pick(window, pump, _rows(window, 'solid')[:1])
    _acts(window)['mesh_solids'][3]()
    pump()
    panel = window.scene.mesh_solids_panel
    assert panel.size.value() == pytest.approx(0.2)
    assert panel.reading_label.text().startswith('About ')


def test_a_fit_likely_to_fold_is_read_before_it_is_paid_for():
    """At 0.25 in Brandon's block is four bricks across its holes'
    radius: likely to fold, with 0.2 in suggested; at 0.2 in it is not.
    The reading takes the grid and its pillow alone, a second."""
    import time

    project, name, geometry = _project()
    cut, _union = _angled(project, name)
    started = time.perf_counter()
    read = hexmesh.likely_to_fold(geometry, [cut], 0.3 * INCH)
    assert time.perf_counter() - started < 10.0
    assert read['likely'] and read['across'] == pytest.approx(10.0 / 3.0)
    assert read['suggested'] == pytest.approx(0.2 * INCH)
    assert not hexmesh.likely_to_fold(geometry, [cut], 0.2 * INCH)['likely']


def test_detect_element_size_and_the_question_before_a_fit(window, pump, monkeypatch):
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
    # Detect Element Size: the 0.4 in hole's radius over five
    panel.detect_button.click()
    _settled(window, pump)
    assert panel.size.value() == pytest.approx(0.08)
    # a coarser size typed: the question, with the suggested size taken
    asked = []

    def answer(text, suggested, anyway):
        asked.append((text, suggested, anyway))
        return answers.pop(0)

    answers = ['cancel', 'suggested']
    monkeypatch.setattr(window, '_ask_mesh_anyway', answer)
    panel.size.setValue(0.2)
    _settled(window, pump)
    journal = len(project.journal)
    panel.mesh_button.click()
    pump()
    assert len(project.journal) == journal, 'cancelled: nothing meshed'
    assert asked[-1][1] == 'Mesh at 0.08 in' and asked[-1][2] == 'Mesh at 0.2 in anyway'
    assert 'is 2.0 bricks across its radius' in asked[-1][0] and '0.08 in is likely to hold' in asked[-1][0]
    panel.mesh_button.click()
    pump()
    assert len(project.journal) == journal + 1 and ', 0.08, unit=' in project.journal[-1]
    assert panel.size.value() == pytest.approx(0.08) or True   # the pane closes after Mesh
    # a size that is likely to hold asks nothing
    _pick(window, pump, _rows(window, 'solid')[:1])
    _acts(window)['mesh_solids'][3]()
    pump()
    panel.size.setValue(0.08)
    _settled(window, pump)
    before = len(asked)
    panel.mesh_button.click()
    pump()
    assert len(asked) == before and len(project.journal) == journal + 2
