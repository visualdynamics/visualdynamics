"""Sections, tubes and holes as solids (Brandon, 2026-10-12: "go ahead
with the sections, tube and hole features").

Each is one row typed by its own dimensions and made of the blocks and
cylinders meshing already reads (`core.solids._pieces`), so a frame's
I-beam is changed by its flange width rather than three blocks one by
one, and meshes as the blocks would. Pinned: every shape's volume, from
the kernel and from its bricks, against the area times the length; the
way each is laid across its own axes; the beam element's refusals; a
hole through whatever its part is made, blind, counterbored, patterned;
the files and units; the project's verbs.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core import solids
from visualdynamics.core.geometry import Geometry

INCH = 0.0254

#: a size of each, and its area across: written out, not computed the
#: way the pieces are
SHAPES = {
    'I-beam': ((4.0, 2.0, 0.25, 0.2), 2 * 2.0 * 0.25 + 0.2 * 3.5),
    'channel': ((4.0, 2.0, 0.25, 0.2), 2 * 2.0 * 0.25 + 0.2 * 3.5),
    'tee': ((3.0, 2.0, 0.25, 0.2), 2.0 * 0.25 + 0.2 * 2.75),
    'angle': ((2.0, 1.5, 0.25), 2.0 * 0.25 + 0.25 * 1.25),
    'rectangular tube': ((3.0, 2.0, 0.25), 3.0 * 2.0 - 2.5 * 1.5),
    'round tube': ((2.0, 0.25), np.pi * (1.0 - 0.75 ** 2)),
}


def _empty():
    return Geometry(node_id=[], node_xyz=np.empty((0, 3)))


def _bricks_volume(mesh):
    conn = np.array(mesh.elem_conn)
    p = mesh.node_xyz[mesh.node_index(conn.ravel())].reshape(len(conn), 8, 3)
    tets = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6),
            (0, 5, 1, 6)]
    return sum(np.einsum('mi,mi->m', p[:, b] - p[:, a],
                         np.cross(p[:, c] - p[:, a], p[:, d] - p[:, a])).sum() / 6
               for a, b, c, d in tets)


@pytest.mark.parametrize('shape', list(SHAPES))
def test_a_sections_volume_is_its_area_along_its_length(shape):
    """From the kernel and from its bricks: the area across times 5 long,
    the round tube's bricks short by their chords."""
    dims, area = SHAPES[shape]
    geometry = _empty()
    made = solids.add_section(geometry, shape, (1, 2, 3), dims, 5.0)
    assert solids.volume(geometry, made) == pytest.approx(5 * area, rel=1e-9)
    mesh, found = solids.extrusion_mesh(geometry, [made], 0.05)
    exact = 1e-9 if shape != 'round tube' else 2e-3
    assert _bricks_volume(mesh) == pytest.approx(5 * area, rel=exact)
    assert found['rings'] == (1 if shape == 'round tube' else 0)


@pytest.mark.parametrize('shape, low, high', [
    # the flange along X, the depth along Y, the length along Z
    ('I-beam', (-1, -2, -2.5), (1, 2, 2.5)),
    ('tee', (-1, -1.5, -2.5), (1, 1.5, 2.5)),
    ('angle', (-1, -0.75, -2.5), (1, 0.75, 2.5)),
    ('rectangular tube', (-1.5, -1, -2.5), (1.5, 1, 2.5)),
])
def test_a_section_fills_the_box_its_dimensions_say(shape, low, high):
    geometry = _empty()
    made = solids.add_section(geometry, shape, (0, 0, 0), SHAPES[shape][0], 5.0)
    box = solids._box(geometry, made)
    assert np.allclose(box[0], low) and np.allclose(box[1], high)


def test_each_shape_is_laid_the_way_its_docstring_says():
    """A channel's web at -X, its flanges running to +X; a tee's flange
    at +Y; an angle's heel at -X, -Y, its long leg along X."""
    geometry = _empty()
    channel = solids.add_section(geometry, 'channel', (0, 0, 0), (4, 2, 0.25, 0.2), 1.0)
    tee = solids.add_section(geometry, 'tee', (0, 0, 0), (3, 2, 0.25, 0.2), 1.0)
    angle = solids.add_section(geometry, 'angle', (0, 0, 0), (2, 1.5, 0.25), 1.0)

    def inside(solid_id, *point):
        return bool(solids.contains(geometry, solid_id, [point])[0])

    assert inside(channel, -0.9, 0, 0) and not inside(channel, 0.9, 0, 0)
    assert inside(tee, 0.9, 1.4, 0) and not inside(tee, 0.9, -1.4, 0)
    assert inside(angle, 0.9, -0.7, 0) and not inside(angle, 0.9, 0.7, 0)
    assert inside(angle, -0.9, 0.7, 0)


@pytest.mark.parametrize('shape, dims, said', [
    ('I-beam', (4, 2, 2.0, 0.2), 'the flanges and web do not make an I'),
    ('channel', (4, 2, 0.25, 2.5), 'the flanges and web do not make a channel'),
    ('tee', (3, 2, 3.0, 0.2), 'the flanges and web do not make a tee'),
    ('angle', (1.5, 2, 0.25), 'a short leg no longer'),
    ('rectangular tube', (3, 2, 1.0), 'closes the section'),
    ('round tube', (2, 1.0), 'a wall under half its diameter'),
    ('round tube', (2, 1.5), 'a wall between 0 and the radius'),
    ('angle', (2, 1.5), 'takes 3 finite numbers'),
])
def test_a_shape_that_does_not_make_its_section_is_refused(shape, dims, said):
    with pytest.raises(ValueError, match=said):
        solids.add_section(_empty(), shape, (0, 0, 0), dims, 5.0)


@pytest.mark.parametrize('shape, dims', [
    ('tee', (3, 2, 2.0, 0.2)),             # a tee's stem is under one flange
    ('I-beam', (4, 2, 0.25, 2.0)),         # a web as wide as its flange
    ('I-beam', (4, 2, 1.5, 0.2)),          # a web of a quarter its depth
    ('angle', (2, 2, 0.25)),               # equal legs
])
def test_a_shape_at_the_edge_of_its_rules_is_made(shape, dims):
    solids.add_section(_empty(), shape, (0, 0, 0), dims, 5.0)


@pytest.mark.parametrize('shape, dims', [
    ('angle', (2, 1.5, 0.0)), ('angle', (2, 1.0, 1.0)),
])
def test_a_shape_with_no_thickness_or_all_thickness_is_refused(shape, dims):
    with pytest.raises(ValueError):
        solids.add_section(_empty(), shape, (0, 0, 0), dims, 5.0)


def test_a_section_needs_a_length_and_a_known_shape():
    with pytest.raises(ValueError, match='a length'):
        solids.add_section(_empty(), 'channel', (0, 0, 0), (4, 2, 0.25, 0.2), 0.0)
    with pytest.raises(ValueError, match='is not one of'):
        solids.add_section(_empty(), 'zee', (0, 0, 0), (4, 2, 0.25, 0.2), 1.0)


def test_a_section_is_changed_by_its_dimensions():
    geometry = _empty()
    made = solids.add_section(geometry, 'I-beam', (0, 0, 0), (4, 2, 0.25, 0.2), 5.0)
    solids.edit(geometry, made, dimensions=(4, 3, 0.25, 0.2), length=10.0)
    assert solids.volume(geometry, made) == pytest.approx(10 * (2 * 3 * 0.25 + 0.2 * 3.5))
    with pytest.raises(ValueError, match='an I-beam has no widths'):
        solids.edit(geometry, made, widths=(1, 1, 1))
    with pytest.raises(ValueError, match='do not make an I'):
        solids.edit(geometry, made, dimensions=(4, 3, 0.25, 4.0))
    assert solids.solid(geometry, made).dimensions == (4, 3, 0.25, 0.2), 'refused, kept'


def test_a_section_moves_and_turns_as_a_block_does():
    geometry = _empty()
    made = solids.add_section(geometry, 'channel', (0, 0, 0), (4, 2, 0.25, 0.2), 5.0)
    assert solids.move(geometry, [made], (1, 0, 0), (90, 0, 0)) == 1
    item = solids.solid(geometry, made)
    assert np.allclose(item.center, (1, 0, 0)) and item.angles[0] == pytest.approx(90)
    low, high = solids._box(geometry, made)
    assert np.allclose(high - low, (2, 5, 4)), 'its length along Y now'


def test_shapes_survive_a_file_and_are_held_in_meters(tmp_path):
    """The dimensions are lengths: written and read back, and held in
    meters as the center is, typed in inches."""
    project = visualdynamics.Project('p')
    name = project.new_geometry('Frame', unit='in')
    beam = project.add_solid_section(name, 'I-beam', (0, 0, 0), (4, 2, 0.25, 0.2), 5.0,
                                     unit='in')
    hole = project.add_solid_hole(name, (0, 0, 2), 0.5, 1.0, counterbore=(1.0, 0.25),
                                  unit='in')
    path = tmp_path / 'frame.vdyn'
    project.save(path)
    again = visualdynamics.Project.open(path)
    assert again[name].solids == project[name].solids
    doubled = solids.solid(project[name], beam).scaled(2.0)
    assert doubled.dimensions == pytest.approx(
        tuple(np.array([8, 4, 0.5, 0.4]) * INCH)), 'a change of unit scales them'
    assert solids.solid(project[name], beam).dimensions == pytest.approx(
        tuple(np.array([4, 2, 0.25, 0.2]) * INCH))
    assert solids.solid(project[name], hole).dimensions == pytest.approx(
        tuple(np.array([0.5, 1.0, 1.0, 0.25]) * INCH))


def _plate(geometry, thickness=1.0):
    return solids.add_block(geometry, (0, 0, 0), (4, 4, thickness))


def test_a_hole_with_no_depth_goes_through_however_thick_its_part_is_made():
    """Entered at the plate's top, it runs out past the bottom: made
    twice as thick, the plate keeps its hole through."""
    geometry = _empty()
    plate = _plate(geometry)
    hole = solids.add_hole(geometry, (0, 0, 0.5), 1.0)
    part = solids.combine(geometry, 'subtract', [plate, hole])
    assert solids.volume(geometry, part) == pytest.approx(16 - np.pi * 0.25)
    # out a hundredth of the part's diagonal past it, so the kernel meets
    # no face on a face
    (_cut, piece), = solids._pieces(geometry, solids.solid(geometry, hole))
    assert piece.length == pytest.approx(1 + 0.01 * np.sqrt(4 ** 2 + 4 ** 2 + 1))
    solids.edit(geometry, plate, center=(0, 0, -0.5), widths=(4, 4, 2))
    assert solids.volume(geometry, part) == pytest.approx(2 * (16 - np.pi * 0.25))
    mesh, made = solids.extrusion_mesh(geometry, [part], 0.1)
    assert made['rings'] == 1
    r = np.hypot(mesh.node_xyz[:, 0], mesh.node_xyz[:, 1])
    assert (r < 0.5 - 1e-9).sum() == 0, 'no node inside it, top to bottom'


def test_a_hole_alone_is_drawn_a_diameter_deep():
    geometry = _empty()
    hole = solids.add_hole(geometry, (0, 0, 0), 0.5)
    low, high = solids._box(geometry, hole)
    assert np.allclose(low, (-0.25, -0.25, -0.5)) and np.allclose(high, (0.25, 0.25, 0))


def test_a_hole_joined_rather_than_cut_goes_through_nothing():
    """Through is through the part it is cut from; unioned with one, it
    is cut from none, and stays a diameter deep."""
    geometry = _empty()
    plate = _plate(geometry)
    hole = solids.add_hole(geometry, (0, 0, 0.5), 0.5)
    solids.combine(geometry, 'union', [plate, hole])
    (_cut, piece), = solids._pieces(geometry, solids.solid(geometry, hole))
    assert piece.length == pytest.approx(0.5)


def test_a_patterned_hole_goes_through_too():
    """Its part is found past the pattern it is copied by."""
    geometry = _empty()
    plate = _plate(geometry)
    hole = solids.add_hole(geometry, (-1, 0, 0.5), 0.5)
    row = solids.pattern(geometry, [hole], (3, 1, 1), (1, 0, 0))
    part = solids.combine(geometry, 'subtract', [plate, row])
    assert solids.volume(geometry, part) == pytest.approx(16 - 3 * np.pi * 0.0625)


def test_a_blind_counterbored_hole_goes_its_depths_from_where_it_enters():
    geometry = _empty()
    plate = _plate(geometry)
    hole = solids.add_hole(geometry, (0, 0, 0.5), 0.5, 0.75, counterbore=(1.0, 0.25))
    part = solids.combine(geometry, 'subtract', [plate, hole])
    taken = np.pi * 0.0625 * 0.75 + np.pi * (0.25 - 0.0625) * 0.25
    assert solids.volume(geometry, part) == pytest.approx(16 - taken)
    mesh, made = solids.extrusion_mesh(geometry, [part], 0.05)
    assert made['rings'] == 1
    assert _bricks_volume(mesh) == pytest.approx(16 - taken, rel=2e-3)
    xyz = mesh.node_xyz
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    assert ((xyz[:, 2] > 0.25 + 1e-9) & (r < 0.5 - 1e-9)).sum() == 0, 'the counterbore'
    assert ((xyz[:, 2] > -0.25 + 1e-9) & (r < 0.25 - 1e-9)).sum() == 0, 'the hole'
    assert ((xyz[:, 2] < -0.25 - 1e-9) & (r < 0.25)).sum() > 0, 'solid below it'


@pytest.mark.parametrize('counterbore, said', [
    ((0.4, 0.25), 'a counterbore wider than the hole'),
    ((0.5, 0.25), 'a counterbore wider than the hole'),
    ((1.0, 0.75), 'shallower than it'),
    ((1.0, 0.0), 'shallower than it'),
])
def test_a_counterbore_narrower_or_deeper_than_its_hole_is_refused(counterbore, said):
    with pytest.raises(ValueError, match=said):
        solids.add_hole(_empty(), (0, 0, 0), 0.5, 0.75, counterbore=counterbore)


def test_the_project_journals_sections_and_holes_in_the_unit_typed():
    project = visualdynamics.Project('p')
    name = project.new_geometry('Frame', unit='in')
    beam = project.add_solid_section(name, 'channel', (0, 0, 0), (4, 2, 0.25, 0.2), 5.0,
                                     unit='in')
    hole = project.add_solid_hole(name, (0, 0, 2), 0.5, unit='in')
    project.edit_solid(name, beam, dimensions=(5, 2, 0.25, 0.2), unit='in')
    assert project.journal[-3:] == [
        ("project.add_solid_section('Frame', 'channel', (0, 0, 0), (4, 2, 0.25, 0.2), "
         "5.0, unit='in')"),
        "project.add_solid_hole('Frame', (0, 0, 2), 0.5, unit='in')",
        "project.edit_solid('Frame', 1, dimensions=(5, 2, 0.25, 0.2), unit='in')"]
    geometry = project[name]
    assert solids.solid(geometry, beam).dimensions[0] == pytest.approx(5 * INCH)
    assert solids.solid(geometry, beam).length == pytest.approx(5 * INCH)
    assert solids.solid(geometry, hole).dimensions == pytest.approx((0.5 * INCH, 0, 0, 0))


@pytest.mark.parametrize('diameter, depth', [(0.0, 1.0), (0.5, -1.0)])
def test_a_hole_needs_a_diameter_and_no_depth_below_zero(diameter, depth):
    with pytest.raises(ValueError, match='a hole needs a diameter'):
        solids.add_hole(_empty(), (0, 0, 0), diameter, depth)

