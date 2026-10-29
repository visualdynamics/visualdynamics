"""Solids on a geometry: blocks, cylinders and booleans of them, the
shapes a part is made of before it is meshed (Brandon, 2026-10-11;
`visualdynamics.core.solids`).

The table holds recipes — the numbers typed — and OpenCASCADE evaluates
them. What is pinned: each primitive's volume and placement against
hand arithmetic, each boolean's, the refusals, the lengths following a
change of unit, the project's verbs in the journal, the file round
trip, and the Frame-Wing's frame built as a plate less its windows and
holes, to the cubic inch.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core import solids

INCH = 0.0254
IN3 = INCH ** 3


def _project():
    project = visualdynamics.Project('p')
    name = project.new_geometry('Part', unit='in')
    return project, name, project[name]


def test_the_primitives_are_where_and_what_they_were_typed():
    project, name, geometry = _project()
    block = project.add_solid_block(name, (1, 2, 3), (4, 2, 1), unit='in')
    cylinder = project.add_solid_cylinder(name, (0, 0, 0), 0.5, 2.0, unit='in')
    assert solids.volume(geometry, block) / IN3 == pytest.approx(8.0)
    assert solids.volume(geometry, cylinder) / IN3 == pytest.approx(np.pi * 0.5)
    corners = solids.shape_triangles(solids.shape(geometry, block)).reshape(-1, 3)
    assert np.allclose(corners.min(axis=0) / INCH, [-1, 1, 2.5])
    assert np.allclose(corners.max(axis=0) / INCH, [3, 3, 3.5])
    # a cylinder with no angles stands along Z, its middle on its center
    corners = solids.shape_triangles(solids.shape(geometry, cylinder)).reshape(-1, 3)
    assert np.allclose(corners[:, 2].min() / INCH, -1.0)
    assert np.allclose(corners[:, 2].max() / INCH, 1.0)


def test_angles_turn_a_solid_about_its_center_as_the_block_pane_does():
    """Degrees about X, then Y, then Z (`rotate.frame_from_angles`): a
    quarter turn about Y lays a cylinder along X; thirty degrees about Z
    leaves a block's volume and widens its footprint."""
    project, name, geometry = _project()
    laid = project.add_solid_cylinder(name, (0, 0, 0), 0.25, 4.0,
                                      angles=(0, 90, 0), unit='in')
    corners = solids.shape_triangles(solids.shape(geometry, laid)).reshape(-1, 3)
    assert np.ptp(corners[:, 0]) / INCH == pytest.approx(4.0)
    assert np.ptp(corners[:, 2]) / INCH == pytest.approx(0.5, rel=0.01)
    turned = project.add_solid_block(name, (0, 0, 0), (2, 1, 1),
                                     angles=(0, 0, 30), unit='in')
    assert solids.volume(geometry, turned) / IN3 == pytest.approx(2.0)
    corners = solids.shape_triangles(solids.shape(geometry, turned)).reshape(-1, 3)
    half = np.cos(np.pi / 6) + 0.5 * np.sin(np.pi / 6)
    assert corners[:, 0].max() / INCH == pytest.approx(half)


def test_the_booleans_are_the_arithmetic_of_their_volumes():
    project, name, geometry = _project()
    a = project.add_solid_block(name, (0, 0, 0), (2, 2, 2), unit='in')
    b = project.add_solid_block(name, (1, 0, 0), (2, 2, 2), unit='in')
    union = project.combine_solids(name, 'union', [a, b])
    assert solids.volume(geometry, union) / IN3 == pytest.approx(12.0)
    c = project.add_solid_block(name, (0, 0, 0), (2, 2, 2), unit='in')
    d = project.add_solid_block(name, (1, 0, 0), (2, 2, 2), unit='in')
    assert solids.volume(geometry, project.combine_solids(
        name, 'intersect', [c, d])) / IN3 == pytest.approx(4.0)
    plate = project.add_solid_block(name, (0, 0, 0), (4, 4, 0.5), unit='in')
    holes = [project.add_solid_cylinder(name, (x, 0, 0), 0.25, 1.0, unit='in')
             for x in (-1, 1)]
    holed = project.combine_solids(name, 'subtract', [plate, *holes], 'plate')
    assert solids.volume(geometry, holed) / IN3 == pytest.approx(
        8.0 - 2 * np.pi * 0.25 ** 2 * 0.5)
    assert [s.name for s in solids.top_level(geometry)] == ['', '', 'plate']
    assert solids.used_by(geometry)[holes[1]] == holed


def test_a_boolean_takes_two_solids_there_each_once_and_unused():
    project, name, _geometry = _project()
    a = project.add_solid_block(name, (0, 0, 0), (1, 1, 1), unit='in')
    b = project.add_solid_block(name, (0, 0, 0), (1, 1, 1), unit='in')
    with pytest.raises(ValueError, match='two solids or more, each once'):
        project.combine_solids(name, 'union', [a])
    with pytest.raises(ValueError, match='two solids or more, each once'):
        project.combine_solids(name, 'union', [a, a])
    with pytest.raises(ValueError, match='no solid 9'):
        project.combine_solids(name, 'union', [a, 9])
    with pytest.raises(ValueError, match="'glue' is not one of"):
        project.combine_solids(name, 'glue', [a, b])
    joined = project.combine_solids(name, 'union', [a, b])
    c = project.add_solid_block(name, (0, 0, 0), (1, 1, 1), unit='in')
    with pytest.raises(ValueError, match=f'solid {a} is already part of solid {joined}'):
        project.combine_solids(name, 'subtract', [c, a])
    with pytest.raises(ValueError, match='a width along all three axes'):
        project.add_solid_block(name, (0, 0, 0), (1, 0, 1), unit='in')
    with pytest.raises(ValueError, match='a radius and a length'):
        project.add_solid_cylinder(name, (0, 0, 0), 0.0, 1.0, unit='in')
    with pytest.raises(ValueError, match='a center takes 3 finite numbers'):
        project.add_solid_block(name, (0, 0), (1, 1, 1), unit='in')


def test_lengths_are_held_as_the_coordinates_are_and_follow_the_unit():
    """Typed in inches, held in meters; the unit undefined, back to the
    numbers as typed; defined as millimeters, read as such."""
    project, name, geometry = _project()
    project.add_solid_block(name, (1, 2, 3), (4, 5, 6), unit='in')
    project.add_solid_cylinder(name, (0, 0, 1), 0.5, 2.0, unit='in')
    block, cylinder = geometry.solids
    assert np.allclose(block.center, np.array([1, 2, 3]) * INCH)
    assert np.allclose(block.widths, np.array([4, 5, 6]) * INCH)
    assert cylinder.radius == pytest.approx(0.5 * INCH)
    geometry.undefine_units()
    block, cylinder = geometry.solids
    assert np.allclose(block.center, [1, 2, 3]) and cylinder.length == pytest.approx(2.0)
    assert block.angles == (0.0, 0.0, 0.0)
    geometry.define_units('mm')
    block, _cylinder = geometry.solids
    assert np.allclose(block.widths, np.array([4, 5, 6]) * 0.001)


def test_the_solids_ride_the_project_file_and_its_journal(tmp_path):
    project, name, _geometry = _project()
    a = project.add_solid_block(name, (0, 0, 0), (2, 2, 0.5), angles=(0, 0, 15),
                                name='plate', unit='in')
    b = project.add_solid_cylinder(name, (0, 0, 0), 0.25, 1.0, name='hole',
                                   unit='in')
    project.combine_solids(name, 'subtract', [a, b], 'plate with a hole')
    assert project.journal[-1] == (
        "project.combine_solids('Part', 'subtract', [1, 2], 'plate with a hole')")
    project.save(tmp_path / 'p.vdyn')
    back = visualdynamics.load(tmp_path / 'p.vdyn')[name]
    assert back.solids == project[name].solids
    # a geometry with none writes none, and reads back with none
    plain = visualdynamics.Project('q')
    plain.new_geometry('Empty', unit='in')
    plain.save(tmp_path / 'q.vdyn')
    assert visualdynamics.load(tmp_path / 'q.vdyn')['Empty'].solids == []


def test_the_frame_wings_frame_is_a_plate_less_its_windows_and_holes():
    """The frame of the Frame-Wing as a person would build it from
    solids: a 16 by 6 by 0.5 in plate, its four windows and 41 insert
    holes subtracted — the volume to the arithmetic (the fillets not
    yet), the reason solids were wanted (Brandon, 2026-10-11)."""
    from visualdynamics.demo import frame_wing as fw

    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, -0.25), (16, 6, 0.5), unit='in')
    inner = fw.WIDTH / 2 - fw.MEMBER
    cuts = [project.add_solid_block(name, ((a + b) / 2, 0, -0.25),
                                    (b - a - fw.MEMBER, 2 * inner, 1.0), unit='in')
            for a, b in zip(fw.UPRIGHTS[:-1], fw.UPRIGHTS[1:])]
    cuts += [project.add_solid_cylinder(name, (x, y, -0.25), fw.HOLE_RADIUS, 1.0,
                                        unit='in') for x, y in fw.INSERTS]
    frame = project.combine_solids(name, 'subtract', [plate, *cuts], 'frame')
    windows = 4 * (3.875 - 0.5) * 5.0 * 0.5
    holes = 41 * np.pi * fw.HOLE_RADIUS ** 2 * 0.5
    assert solids.volume(geometry, frame) / IN3 == pytest.approx(48 - windows - holes)
    assert [s.name for s in solids.top_level(geometry)] == ['frame']


def test_delete_takes_a_part_whole_and_take_apart_undoes_a_boolean():
    """Delete goes deep — a plate with its holes cut goes with its
    cylinders — and refuses a solid inside another; Take Apart is the
    boolean's undo, its operands parts again, and a primitive has
    nothing to take apart."""
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, 0), (4, 4, 0.5), unit='in')
    hole = project.add_solid_cylinder(name, (0, 0, 0), 0.25, 1.0, unit='in')
    holed = project.combine_solids(name, 'subtract', [plate, hole])
    spare = project.add_solid_block(name, (5, 0, 0), (1, 1, 1), unit='in')
    with pytest.raises(ValueError, match=f'solid {hole} is part of solid {holed}; '
                                         'take that apart first'):
        project.delete_solids(name, [hole])
    with pytest.raises(ValueError, match='is a block, with nothing to take apart'):
        project.separate_solids(name, [spare])
    assert project.separate_solids(name, [holed]) == 2
    assert [s.id for s in solids.top_level(geometry)] == [plate, hole, spare]
    holed = project.combine_solids(name, 'subtract', [plate, hole])
    assert project.delete_solids(name, [holed]) == 3
    assert [s.id for s in geometry.solids] == [spare]


def test_a_pattern_is_one_solid_made_of_its_original():
    """One hole and 13 along X at the insert pitch is a rail's row; that
    row and 2 along Y is both rails' — the Frame-Wing's 26 rail holes
    from one typed, a pattern of a pattern. One solid, not a copy each
    (Brandon, 2026-10-11): it takes its original's name, its copies
    stand where they should, the hole made larger is every copy larger,
    and its counts changed are more holes."""
    project, name, geometry = _project()
    hole = project.add_solid_cylinder(name, (-7.75, -2.75, -0.25), 0.1451, 1.0,
                                      name='hole', unit='in')
    row = project.pattern_solids(name, [hole], (13, 1, 1), (15.5 / 12, 0, 0),
                                 unit='in')
    rails = project.pattern_solids(name, [row], (1, 2, 1), (0, 5.5, 0), unit='in')
    assert project.journal[-1] == ("project.pattern_solids('Part', [2], (1, 2, 1), "
                                   "(0, 5.5, 0), unit='in')")
    assert [s.id for s in solids.top_level(geometry)] == [rails]
    made = solids.solid(geometry, rails)
    assert made.kind == 'pattern' and made.name == 'hole' and made.operands == (row,)
    centers = np.array([leaf.center for leaf in solids._leaves(geometry, rails)]) / INCH
    assert len(centers) == 26
    assert np.allclose(centers[[0, 12, 13, 25]],
                       [[-7.75, -2.75, -0.25], [7.75, -2.75, -0.25],
                        [-7.75, 2.75, -0.25], [7.75, 2.75, -0.25]])
    plate = project.add_solid_block(name, (0, 0, -0.25), (16, 6, 0.5), unit='in')
    holed = project.combine_solids(name, 'subtract', [plate, rails])
    cut = 26 * np.pi * 0.1451 ** 2 * 0.5
    assert solids.volume(geometry, holed) / IN3 == pytest.approx(48 - cut)
    # the original changed: every copy
    project.edit_solid(name, hole, radius=0.2, unit='in')
    assert solids.volume(geometry, holed) / IN3 == pytest.approx(
        48 - 26 * np.pi * 0.04 * 0.5)
    # the pattern changed: three rows, not two
    project.edit_solid(name, rails, counts=(1, 3, 1), spacing=(0, 2.75, 0), unit='in')
    assert solids.volume(geometry, holed) / IN3 == pytest.approx(
        48 - 39 * np.pi * 0.04 * 0.5)
    middle = (-7.75 + 0.6 * 15.5 / 12, 0.0, -0.25)
    assert solids.contains(geometry, holed, [np.array(middle) * INCH]).tolist() == [True]
    on_hole = (-7.75 + 15.5 / 12, 0.0, -0.25)
    assert solids.contains(geometry, holed, [np.array(on_hole) * INCH]).tolist() == [False]
    with pytest.raises(ValueError, match='more than one along Y needs a spacing'):
        project.edit_solid(name, rails, spacing=(0, 0, 0), unit='in')


def test_a_pattern_turned_turns_its_rows():
    """A row of three holes along X in a plate, the part turned a
    quarter about Z: the row runs along Y, the pattern's own axes
    turned with its original."""
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, 0), (4, 4, 0.5), unit='in')
    hole = project.add_solid_cylinder(name, (-1, 0, 0), 0.2, 1.0, unit='in')
    row = project.pattern_solids(name, [hole], (3, 1, 1), (1, 0, 0), unit='in')
    part = project.combine_solids(name, 'subtract', [plate, row])
    volume = solids.volume(geometry, part)
    assert project.move_solids(name, [part], turn=(0, 0, 90), unit='in') == 2
    centers = np.array([leaf.center for leaf in solids._leaves(geometry, part)]) / INCH
    assert np.allclose(centers[1:], [(0, -1, 0), (0, 0, 0), (0, 1, 0)], atol=1e-9)
    assert solids.volume(geometry, part) == pytest.approx(volume)


def test_a_pattern_refuses_copies_on_top_of_each_other():
    project, name, geometry = _project()
    a = project.add_solid_block(name, (0, 0, 0), (1, 1, 1), unit='in')
    with pytest.raises(ValueError, match='more than one along Y needs a spacing'):
        project.pattern_solids(name, [a], (1, 3, 1), (1, 0, 0), unit='in')
    with pytest.raises(ValueError, match='one or more along each of X, Y and Z'):
        project.pattern_solids(name, [a], (0, 1, 1), (1, 1, 1), unit='in')
    with pytest.raises(ValueError, match='copies: more than 10,000'):
        project.pattern_solids(name, [a], (101, 101, 1), (1, 1, 1), unit='in')
    b = project.add_solid_block(name, (5, 0, 0), (1, 1, 1), unit='in')
    both = project.combine_solids(name, 'union', [a, b])
    with pytest.raises(ValueError, match=f'solid {a} is part of solid {both}'):
        project.pattern_solids(name, [a], (2, 1, 1), (2, 0, 0), unit='in')
    assert len(geometry.solids) == 3, 'nothing made'


def test_a_rounded_block_is_a_block_less_its_corners():
    """A corner radius rounds the four edges along the block's own Z:
    its volume the box's less (4 - pi) c^2 per unit depth, a point in a
    corner's square inside only within the radius of the arc's center,
    and the radius at most half the narrower width across."""
    project, name, geometry = _project()
    window = project.add_solid_block(name, (0, 0, 0), (3.375, 5, 0.5), corner=0.138,
                                     unit='in')
    exact = (3.375 * 5 - (4 - np.pi) * 0.138 ** 2) * 0.5
    assert solids.volume(geometry, window) / IN3 == pytest.approx(exact)
    corner = np.array([[3.375 / 2 - 0.01, 2.5 - 0.01, 0.0],     # cut away
                       [3.375 / 2 - 0.138, 2.5 - 0.01, 0.0],    # on the arc's line
                       [0.0, 0.0, 0.0]]) * INCH
    assert solids.contains(geometry, window, corner).tolist() == [False, True, True]
    with pytest.raises(ValueError, match='a corner radius from zero to half'):
        project.add_solid_block(name, (0, 0, 0), (1, 2, 1), corner=0.6, unit='in')
    assert geometry.solids[0].corner == pytest.approx(0.138 * INCH)
    assert project.journal[-1].endswith("corner=0.138, unit='in')"), \
        'the refused call is not journaled'


def test_a_rounded_corner_rides_the_file(tmp_path):
    project, name, _geometry = _project()
    project.add_solid_block(name, (0, 0, 0), (2, 2, 1), corner=0.25, unit='in')
    project.save(tmp_path / 'r.vdyn')
    (back,) = visualdynamics.load(tmp_path / 'r.vdyn')[name].solids
    assert back.corner == pytest.approx(0.25 * INCH)


def test_a_boolean_takes_its_first_solids_name_unless_given_one():
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, 0), (4, 4, 0.5), name='frame',
                                    unit='in')
    hole = project.add_solid_cylinder(name, (0, 0, 0), 0.25, 1.0, unit='in')
    holed = project.combine_solids(name, 'subtract', [plate, hole])
    assert solids.solid(geometry, holed).name == 'frame'
    other = project.add_solid_block(name, (9, 0, 0), (1, 1, 1), unit='in')
    named = project.combine_solids(name, 'union', [other, holed], 'both')
    assert solids.solid(geometry, named).name == 'both'


def test_a_solid_is_changed_after_it_is_made():
    """Brandon (2026-10-11): move a solid, size it, reshape it once it is
    there. A hole made larger and moved is a larger hole elsewhere in
    the plate it was subtracted from — the boolean keeps its solid by id
    — and the journal reads what was typed; only what is given changes."""
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, 0), (4, 4, 0.5), name='plate',
                                    unit='in')
    hole = project.add_solid_cylinder(name, (0, 0, 0), 0.25, 1.0, unit='in')
    part = project.combine_solids(name, 'subtract', [plate, hole])
    project.edit_solid(name, hole, center=(1, 0.5, 0), radius=0.5, unit='in')
    assert project.journal[-1] == ("project.edit_solid('Part', 2, center=(1, 0.5, 0), "
                                   "radius=0.5, unit='in')")
    moved = solids.solid(geometry, hole)
    assert np.allclose(np.asarray(moved.center) / INCH, (1, 0.5, 0))
    assert moved.length / INCH == pytest.approx(1.0), 'kept'
    assert solids.volume(geometry, part) / IN3 == pytest.approx(8 - np.pi * 0.25 * 0.5)
    project.edit_solid(name, plate, widths=(5, 4, 0.5), corner=0.25, name='base',
                       unit='in')
    assert solids.solid(geometry, plate).name == 'base'
    assert solids.solid(geometry, plate).corner / INCH == pytest.approx(0.25)
    project.edit_solid(name, part, name='holed')
    assert solids.solid(geometry, part).name == 'holed'


def test_a_change_an_add_would_refuse_is_refused_and_changes_nothing():
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (2, 2, 1), unit='in')
    cylinder = project.add_solid_cylinder(name, (0, 0, 0), 0.25, 1.0, unit='in')
    part = project.combine_solids(name, 'union', [block, cylinder])
    before = list(geometry.solids)
    with pytest.raises(ValueError, match='corner radius from zero'):
        project.edit_solid(name, block, corner=1.5, unit='in')
    with pytest.raises(ValueError, match='a cylinder has no widths'):
        project.edit_solid(name, cylinder, widths=(1, 1, 1), unit='in')
    with pytest.raises(ValueError, match='a union: change the solids it is made of'):
        project.edit_solid(name, part, center=(1, 0, 0), unit='in')
    with pytest.raises(ValueError, match='needs a radius and a length'):
        project.edit_solid(name, cylinder, length=0.0, unit='in')
    assert geometry.solids == before


def test_a_part_moves_and_turns_whole():
    """A plate with a hole off its middle, turned a quarter about Z and
    slid: the hole goes round with it, the volume is kept, and the
    extent lands where the turned and slid box does. Turned about the
    middle of its extent, so a part turned in place stays put."""
    project, name, geometry = _project()
    plate = project.add_solid_block(name, (0, 0, 0), (4, 2, 0.5), unit='in')
    hole = project.add_solid_cylinder(name, (1, 0, 0), 0.25, 1.0, unit='in')
    part = project.combine_solids(name, 'subtract', [plate, hole])
    volume = solids.volume(geometry, part)
    moved = project.move_solids(name, [part], (10, 0, 0), turn=(0, 0, 90), unit='in')
    assert moved == 2
    assert project.journal[-1] == ("project.move_solids('Part', [3], (10, 0, 0), "
                                   "turn=(0, 0, 90), unit='in')")
    assert solids.volume(geometry, part) == pytest.approx(volume)
    hole_now = solids.solid(geometry, hole)
    assert np.allclose(np.asarray(hole_now.center) / INCH, (10, 1, 0), atol=1e-9)
    low, high = solids._box(geometry, part)
    assert np.allclose(low / INCH, (9, -2, -0.25), atol=1e-9)
    assert np.allclose(high / INCH, (11, 2, 0.25), atol=1e-9)
    corners = solids.shape_triangles(solids.shape(geometry, part)).reshape(-1, 3) / INCH
    assert np.allclose(corners.min(axis=0), (9, -2, -0.25), atol=1e-6)


def test_a_pattern_rides_the_file(tmp_path):
    """Its counts and spacing, its original under it; a file from before
    patterns (no counts in its records) reads each solid one of each."""
    project, name, geometry = _project()
    hole = project.add_solid_cylinder(name, (0, 0, 0), 0.1, 1.0, unit='in')
    made = project.pattern_solids(name, [hole], (3, 2, 1), (0.5, 0.75, 0), unit='in')
    project.save(tmp_path / 'p.vdyn')
    back = visualdynamics.load(tmp_path / 'p.vdyn')[name]
    assert back.solids == geometry.solids
    assert solids.solid(back, made).counts == (3, 2, 1)
    record = solids.solid(geometry, hole).as_record()
    del record['counts']
    assert solids.Solid.from_record(record).counts == (1, 1, 1)


def test_without_the_kernel_the_volume_says_how_to_install_it(monkeypatch):
    """Every kernel call says how to get the kernel back; the volume
    raised a bare ImportError instead (2026-10-13)."""
    import sys

    from visualdynamics.core import solids
    from visualdynamics.core.geometry import Geometry

    geometry = Geometry(node_id=[], node_xyz=np.empty((0, 3)))
    block = solids.add_block(geometry, (0, 0, 0), (1, 1, 1))
    monkeypatch.setitem(sys.modules, 'OCP', None)
    with pytest.raises(ValueError,
                       match='pip install --force-reinstall cadquery-ocp'):
        solids.volume(geometry, block)
