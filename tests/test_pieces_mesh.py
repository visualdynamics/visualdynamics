"""A union that is no extrusion meshed in pieces (`solids.brick_mesh`):
each operand swept where it is an extrusion and fitted where it is
not, at its own size, joined where the grids meet and tied at the
seams where they do not. The fits are stood in for by a plain block's
(`hexmesh.overlay_mesh` fits on any machine for that), so nothing
here depends on a fit's outcome; the real thing is in the slow test.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core import hexmesh, mates, solids
from visualdynamics.core.geometry import Geometry

INCH = 0.0254


def _project():
    import visualdynamics

    project = visualdynamics.Project('p')
    name = project.new_geometry('Part', unit='in')
    return project, name, project[name]


def _slab_and_bar(project, name, bar_at=1.0):
    """A slab with a hole tilted through it, which is no extrusion and
    is fitted, and a bar standing on its top (at `bar_at`, the bar's
    center height: 1.0 touches, 1.5 floats), which is swept."""
    block = project.add_solid_block(name, (0, 0, 0), (4, 2, 1), name='slab', unit='in')
    tilted = project.add_solid_cylinder(name, (0, 0, 0), 0.6, 3.0, angles=(30, 0, 0),
                                        unit='in')
    cut = project.combine_solids(name, 'subtract', [block, tilted], 'slab')
    bar = project.add_solid_block(name, (1.5, 0, bar_at), (1, 2, 1), name='bar', unit='in')
    return cut, bar, project.combine_solids(name, 'union', [cut, bar], 'both')


def _group(geometry, name):
    return int(geometry.group_id[list(geometry.group_name).index(name)])


def _stand_in(monkeypatch):
    """The fit stood in for by the plain block's mesh, the hole left
    out: the piece's grid, on any machine."""
    real = hexmesh.overlay_mesh

    def fit(geometry, ids, size, progress=None, piece=False, cuts_from=(), discard=None):
        plain = Geometry(node_id=[], node_xyz=np.zeros((0, 3)), elem_id=[],
                         elem_type=[], elem_conn=[], length_unit='in')
        box = solids.add_block(plain, (0, 0, 0), (4 * INCH, 2 * INCH, 1 * INCH),
                               name=solids.piece_name(geometry, ids[0]))
        return real(plain, [box], size, progress=progress, discard=discard)

    monkeypatch.setattr(hexmesh, 'overlay_mesh', fit)


def test_a_lone_unions_operands_are_its_pieces():
    project, name, geometry = _project()
    cut, bar, both = _slab_and_bar(project, name)
    assert solids.pieces(geometry, [both]) == [cut, bar]
    assert solids.pieces(geometry, [cut, bar]) == [cut, bar]
    assert solids.pieces(geometry, [bar]) == [bar]


def test_the_plan_names_each_pieces_route_and_size():
    project, name, geometry = _project()
    cut, bar, both = _slab_and_bar(project, name)
    plan = solids.piece_plan(geometry, [both], 0.25 * INCH, sizes={bar: 0.125 * INCH})
    assert [(p['id'], p['name'], p['method']) for p in plan] == [
        (cut, 'slab', 'overlay'), (bar, 'bar', 'extrusion')]
    assert [p['size'] for p in plan] == [0.25 * INCH, 0.125 * INCH]
    assert plan[1]['cells'] == 8 * 16 * 8
    # an extrusion is one entry, the parts together
    free = project.add_solid_block(name, (0, 0, 4), (1, 1, 1), name='free', unit='in')
    alone = solids.piece_plan(geometry, [free], 0.25 * INCH)
    assert len(alone) == 1 and alone[0]['method'] == 'extrusion'
    # an operand on its own is refused, as any mesh of one is
    with pytest.raises(ValueError, match='part of solid'):
        solids.grid_cells(geometry, [bar], 0.25 * INCH)
    with pytest.raises(ValueError, match='part of solid'):
        solids.piece_plan(geometry, [bar], 0.25 * INCH)


def test_the_pieces_are_swept_and_fitted_each_and_joined(monkeypatch):
    _stand_in(monkeypatch)
    project, name, geometry = _project()
    _cut, _bar, both = _slab_and_bar(project, name)
    part, made = solids.brick_mesh(geometry, [both], 0.25 * INCH)
    assert made['method'] == 'pieces'
    assert [(m['name'], m['method']) for m in made['pieces']] == [
        ('slab', 'overlay'), ('bar', 'extrusion')]
    slab, swept = made['pieces']
    assert swept['bricks'] == 4 * 8 * 4 and swept['nodes'] == 5 * 9 * 5
    assert slab['bricks'] > 16 * 8 * 4, 'the fit pillows its surface'
    assert made['bricks'] == slab['bricks'] + swept['bricks']
    assert list(part.group_name) == ['slab', 'bar']
    # the bar's bottom lands on the slab's top grid, node for node: the
    # seam is shared and nothing is tied
    assert made['seam_shared'] == 5 * 9 and made['ties'] == 0
    assert part.num_nodes == slab['nodes'] + swept['nodes'] - 5 * 9


def test_a_finer_piece_is_tied_to_the_coarser_at_the_seam(monkeypatch):
    _stand_in(monkeypatch)
    project, name, geometry = _project()
    _cut, bar, both = _slab_and_bar(project, name)
    part, made = solids.brick_mesh(geometry, [both], 0.25 * INCH, sizes={bar: 0.125 * INCH})
    assert [m['size'] for m in made['pieces']] == [0.25 * INCH, 0.125 * INCH]
    # the bar's 9 x 17 seam nodes: every other one on the slab's grid,
    # the rest joined to it rigidly, as one set of mated surfaces
    assert made['seam_shared'] == 5 * 9
    assert made['ties'] == 9 * 17 - 5 * 9
    assert list(part.group_name) == ['slab', 'bar'] and made['seams'] == 1
    seam, = part.mates
    assert (seam.normal_mode, seam.shear_mode) == ('rigid', 'rigid')
    assert seam.name == 'bar on slab' and len(seam.nodes_a) == made['ties']
    # links are made when a model is built, not kept as elements
    assert not (np.asarray(part.elem_type) == 21).any()
    assert len(mates.links(part, seam)) == made['ties']


def test_pieces_apart_share_nothing_and_tie_nothing(monkeypatch):
    _stand_in(monkeypatch)
    project, name, geometry = _project()
    _cut, _bar, both = _slab_and_bar(project, name, bar_at=1.5)
    part, made = solids.brick_mesh(geometry, [both], 0.25 * INCH)
    assert made['seam_shared'] == 0 and made['ties'] == 0
    assert not part.mates


def test_a_piece_running_into_an_earlier_one_loses_the_overlap(monkeypatch):
    """The union is held: a bar sunk half its height into the slab
    keeps only its bricks above the slab's top, its grid cut at that
    face, and the seam there is the slab's top (Brandon, 2026-10-20:
    legs run up into the table top, and meshed whole they doubled the
    block's bricks there)."""
    _stand_in(monkeypatch)
    project, name, geometry = _project()
    _cut, _bar, both = _slab_and_bar(project, name, bar_at=0.5)
    part, made = solids.brick_mesh(geometry, [both], 0.25 * INCH)
    slab, swept = made['pieces']
    assert swept['dropped'] == 4 * 8 * 2 and swept['bricks'] == 4 * 8 * 2
    assert slab['dropped'] == 0
    xyz = np.asarray(part.node_xyz) / INCH
    groups = np.asarray(part.elem_group)
    bar_rows = np.flatnonzero(groups == part.group_id[list(part.group_name).index('bar')])
    lowest = min(xyz[part.node_index(part.elem_conn[r]), 2].min() for r in bar_rows)
    assert lowest == pytest.approx(0.5)
    assert made['seam_shared'] == 5 * 9 and made['ties'] == 0
    # wholly inside, there is nothing of it to mesh
    slab2 = project.add_solid_block(name, (0, 0, 3), (4, 2, 1), name='slab2', unit='in')
    tilted2 = project.add_solid_cylinder(name, (0, 0, 3), 0.6, 3.0, angles=(30, 0, 0),
                                         unit='in')
    cut2 = project.combine_solids(name, 'subtract', [slab2, tilted2], 'slab2')
    hidden = project.add_solid_block(name, (-1, 0, 3), (1, 1, 0.5), name='hidden', unit='in')
    whole2 = project.combine_solids(name, 'union', [cut2, hidden], 'both2')
    with pytest.raises(ValueError, match='hidden lies wholly inside'):
        solids.brick_mesh(geometry, [whole2], 0.25 * INCH)


def test_a_fitted_piece_leaves_out_its_bricks_inside_an_earlier_one(monkeypatch):
    """With the bar first and the slab after it, sunk half into the bar,
    the fit itself leaves out the slab's bricks inside the bar, so a
    fold among them would be no refusal (the rod of Brandon's volume
    folded four bricks at its end inside the block, 2026-10-20)."""
    _stand_in(monkeypatch)
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (4, 2, 1), name='slab', unit='in')
    tilted = project.add_solid_cylinder(name, (0, 0, 0), 0.6, 3.0, angles=(30, 0, 0),
                                        unit='in')
    cut = project.combine_solids(name, 'subtract', [block, tilted], 'slab')
    bar = project.add_solid_block(name, (1.5, 0, 0.5), (1, 2, 1), name='bar', unit='in')
    both = project.combine_solids(name, 'union', [bar, cut], 'both')
    part, made = solids.brick_mesh(geometry, [both], 0.25 * INCH)
    swept, slab = made['pieces']
    # the stand-in's pillow doubles the bricks under its skin, so the
    # count is the fit's own: everything inside the bar went in the fit
    assert swept['dropped'] == 0
    assert slab['discarded'] > 4 * 8 * 2 and slab['dropped'] == slab['discarded']
    centers = np.array([part.node_xyz[part.node_index(conn)].mean(axis=0)
                        for conn in part.elem_conn])
    assert not solids.contains(geometry, bar, centers[
        np.asarray(part.elem_group) == part.group_id[list(part.group_name).index('slab')]]).any()


def test_a_later_piece_is_fitted_as_itself_less_the_earlier_pieces(monkeypatch):
    """The fit of a piece after others is of a subtraction that lives
    for the fit alone, named as the piece, so the face where the piece
    enters an earlier one is a surface of what is fitted and its nodes
    are held there to be mated (fitted whole and cut after, they
    drifted off and the piece was tied to nothing, 2026-10-20)."""
    fitted = []
    real = hexmesh.overlay_mesh

    def spy(geometry, ids, size, progress=None, piece=False, cuts_from=(), discard=None):
        item = solids.solid(geometry, ids[0])
        fitted.append((item.kind, item.name, item.operands, len(geometry.solids)))
        plain = Geometry(node_id=[], node_xyz=np.zeros((0, 3)), elem_id=[],
                         elem_type=[], elem_conn=[], length_unit='in')
        box = solids.add_block(plain, (0, 0, 0), (4 * INCH, 2 * INCH, 1 * INCH), name=item.name)
        return real(plain, [box], size, progress=progress, discard=discard)

    monkeypatch.setattr(hexmesh, 'overlay_mesh', spy)
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (4, 2, 1), name='slab', unit='in')
    tilted = project.add_solid_cylinder(name, (0, 0, 0), 0.6, 3.0, angles=(30, 0, 0),
                                        unit='in')
    cut = project.combine_solids(name, 'subtract', [block, tilted], 'slab')
    bar = project.add_solid_block(name, (1.5, 0, 0.5), (1, 2, 1), name='bar', unit='in')
    both = project.combine_solids(name, 'union', [bar, cut], 'both')
    count = len(geometry.solids)
    part, _made = solids.brick_mesh(geometry, [both], 0.25 * INCH)
    assert fitted == [('subtract', 'slab', (cut, bar), count + 1)]
    assert len(geometry.solids) == count, 'the subtraction lived for the fit alone'
    assert list(part.group_name)[:2] == ['bar', 'slab']
    # first of the pieces, a fit is of the piece itself
    fitted.clear()
    project2, name2, geometry2 = _project()
    _cut2, _bar2, both2 = _slab_and_bar(project2, name2)
    solids.brick_mesh(geometry2, [both2], 0.25 * INCH)
    assert fitted[0][0] == 'subtract' and fitted[0][1] == 'slab' and len(fitted[0][2]) == 2


def test_every_seam_keeps_its_ties(monkeypatch):
    """Two bars standing on the slab, each finer than it: both seams
    are kept, 108 nodes each, a set of mated surfaces each (the second
    seam's links had replaced the first's, 2026-10-20)."""
    _stand_in(monkeypatch)
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (4, 2, 1), name='slab', unit='in')
    tilted = project.add_solid_cylinder(name, (0, 0, 0), 0.6, 3.0, angles=(30, 0, 0),
                                        unit='in')
    cut = project.combine_solids(name, 'subtract', [block, tilted], 'slab')
    left = project.add_solid_block(name, (-1.5, 0, 1.0), (1, 2, 1), name='left', unit='in')
    right = project.add_solid_block(name, (1.5, 0, 1.0), (1, 2, 1), name='right', unit='in')
    both = project.combine_solids(name, 'union', [cut, left, right], 'both')
    part, made = solids.brick_mesh(geometry, [both], 0.25 * INCH,
                                   sizes={left: 0.125 * INCH, right: 0.125 * INCH})
    assert made['ties'] == 2 * 108 and made['seams'] == 2
    assert sorted(m.name for m in part.mates) == ['left on slab', 'right on slab']
    assert [len(mates.links(part, m)) for m in part.mates] == [108, 108]


def test_a_refused_fit_names_its_piece(monkeypatch):
    def refuse(geometry, ids, size, progress=None, piece=False, cuts_from=(), discard=None):
        raise ValueError('4 bricks folded fitting the shape at that size: '
                         'use a smaller one, or tetrahedra')

    monkeypatch.setattr(hexmesh, 'overlay_mesh', refuse)
    project, name, geometry = _project()
    _cut, _bar, both = _slab_and_bar(project, name)
    with pytest.raises(ValueError, match='^slab: 4 bricks folded'):
        solids.brick_mesh(geometry, [both], 0.25 * INCH)


def test_progress_runs_once_over_all_the_pieces(monkeypatch):
    _stand_in(monkeypatch)
    project, name, geometry = _project()
    _cut, _bar, both = _slab_and_bar(project, name)
    told = []
    solids.brick_mesh(geometry, [both], 0.25 * INCH, progress=lambda d, t: told.append((d, t)))
    assert told and all(t == 200 for _d, t in told)
    assert told[0][0] < 100 <= told[-1][0] <= 200
    assert [d for d, _t in told] == sorted(d for d, _t in told)


def test_the_project_meshes_a_union_in_pieces_with_a_size_each(monkeypatch):
    """`mesh_solids` takes a size per piece in the typed unit, puts the
    pieces' groups and the seam's mated surfaces into the geometry, and
    meshing again replaces the bricks and the old seam with them."""
    _stand_in(monkeypatch)
    project, name, geometry = _project()
    _cut, bar, both = _slab_and_bar(project, name)
    made = project.mesh_solids(name, [both], 0.25, unit='in', sizes={bar: 0.125})
    assert made['method'] == 'pieces' and made['ties'] == 108
    assert [m['size'] for m in made['pieces']] == pytest.approx([0.25 * INCH, 0.125 * INCH])
    assert set(geometry.group_name) == {'slab', 'bar'}
    seam, = geometry.mates
    assert seam.groups == (_group(geometry, 'bar'), _group(geometry, 'slab'))
    assert len(mates.links(geometry, seam)) == 108
    bricks = int((np.asarray(geometry.elem_type) == 115).sum())
    # meshed again at one size: the old bricks and their seam go, and
    # the seam shares its nodes outright
    again = project.mesh_solids(name, [both], 0.25, unit='in')
    assert again['replaced'] == bricks and again['ties'] == 0
    assert again['mates_dropped'] == 1 and not geometry.mates
    assert int((np.asarray(geometry.elem_type) == 115).sum()) == again['bricks']


def test_the_pane_takes_a_size_per_piece(window, pump, monkeypatch):
    """Mesh Solids on a union that is no extrusion shows a row per
    piece with the size each is likely to hold at, in place of the one
    box; Tetras puts the box back; Detect fills every row; and Mesh
    hands the sizes on, the bar's finer one tying it at the seam."""
    pytest.importorskip('OCP')
    from conftest import select_objects
    from test_solids_mesh import _settled
    from test_solids_pane import _acts, _pick, _rows

    _stand_in(monkeypatch)
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    project = window.project
    name = project.new_geometry('Part', unit='in')
    cut, bar, _both = _slab_and_bar(project, name)
    window.show_object(name)
    select_objects(window, pump, name)
    window._solids_changed(name)
    pump()
    _pick(window, pump, _rows(window, 'solid')[:1])
    _acts(window)['mesh_solids'][3]()
    pump()
    panel = window.scene.mesh_solids_panel
    assert panel.piece_ids == [cut, bar]
    assert panel.pieces.isVisible() and not panel.size.isVisible()
    assert panel.piece_sizes()[bar] == pytest.approx(0.5)       # half the bar's thinnest
    assert panel.piece_sizes()[cut] == pytest.approx(0.12)      # the 0.6 in hole over five
    # coarser, the preview runs and reads the pieces
    panel.set_piece_size(cut, 0.25)
    panel.set_piece_size(bar, 0.25)
    panel.changed.emit()
    _settled(window, pump)
    reading = panel.reading_label.text()
    assert 'bricks in 2 pieces' in reading and 'slab fitted at 0.25 in' in reading
    assert 'bar swept along Z at 0.25 in' in reading and '45 nodes shared' in reading
    panel.elements.setCurrentIndex(panel.elements.findData('tetras'))
    _settled(window, pump)
    assert not panel.pieces.isVisible() and panel.size.isVisible() and not panel.piece_ids
    panel.elements.setCurrentIndex(panel.elements.findData('hexes'))
    _settled(window, pump)
    assert panel.piece_ids == [cut, bar]
    panel.set_piece_size(cut, 0.25)
    panel.set_piece_size(bar, 0.3)
    panel.detect_button.click()
    pump()
    assert panel.piece_sizes()[bar] == pytest.approx(0.5)
    panel.set_piece_size(cut, 0.25)
    panel.set_piece_size(bar, 0.125)
    monkeypatch.setattr(window, '_ask_mesh_anyway', lambda *args: 'anyway')
    panel.mesh_button.click()
    pump()
    geometry = project[name]
    seam, = geometry.mates
    assert len(mates.links(geometry, seam)) == 108
    assert set(geometry.group_name) == {'slab', 'bar'}
    assert 'bricks' in window.statusBar().currentMessage()


@pytest.mark.slow
def test_the_real_fit_and_the_sweep_meet_at_the_seam():
    """The slab fitted for real (4.8 bricks across its hole at 0.125 in,
    which fits here and may not elsewhere: slow, and asserting only
    what every machine's fit gives) and the bar swept finer on it."""
    project, name, geometry = _project()
    _cut, bar, both = _slab_and_bar(project, name)
    part, made = solids.brick_mesh(geometry, [both], 0.125 * INCH, sizes={bar: 0.0625 * INCH})
    assert made['method'] == 'pieces' and made['pieces'][0]['method'] == 'overlay'
    assert made['ties'] > 0 and made['seam_shared'] > 0
    assert set(part.group_name) == {'slab', 'bar'} and part.mates


@pytest.mark.slow
def test_a_rod_running_into_a_block_is_one_structure():
    """A tilted rod into a block, the block swept first and the rod
    fitted as itself less the block: the rod's cut face lies on the
    block's top and is tied there, so the mesh is one connected piece
    (Brandon's project, 2026-10-20: the rod was tied to nothing and
    the modes could not be solved)."""
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (4, 3, 1), name='block', unit='in')
    rod = project.add_solid_cylinder(name, (1.0, 0, 1.4), 0.4, 3.0, angles=(0, -50, 0),
                                     name='rod', unit='in')
    both = project.combine_solids(name, 'union', [block, rod], 'both')
    made = project.mesh_solids(name, [both], 0.25, unit='in', sizes={rod: 0.08})
    assert [m['method'] for m in made['pieces']] == ['extrusion', 'overlay']
    # fitted less the block, the rod has nothing inside it to drop
    assert made['pieces'][1]['dropped'] == 0
    assert made['ties'] + made['seam_shared'] > 20
    # one structure once the seams' links are made
    from visualdynamics.core.fem import MATERIALS, GroupProperties, Model

    for group in geometry.group_id:
        geometry.group_properties[int(group)] = GroupProperties(MATERIALS['1018 steel'])
    Model.from_geometry(geometry)


# ---- a piece turned off the axes, swept along its own (2026-10-20) ----

def _local(geometry, solid_id, xyz):
    """Points in a solid's own frame: centered on it, along its axes."""
    item = solids.solid(geometry, solid_id)
    return (np.asarray(xyz) - np.asarray(item.center)) @ solids._frame(item).T


def test_a_turned_cylinder_alone_is_swept_along_its_own_axis():
    """A rod turned 50 degrees is swept along its own axis, as the same
    rod square to the axes is, rings and all, and put back: the same
    bricks and nodes, every node within its radius and length (Brandon,
    2026-10-20: the tilted rod's mesh did not line up with it)."""
    project, name, geometry = _project()
    rod = project.add_solid_cylinder(name, (0.3, 0.2, 0.1), 0.4, 3.0, angles=(0, -50, 0),
                                     name='rod', unit='in')
    part, made = solids.brick_mesh(geometry, [rod], 0.125 * INCH)
    assert made['method'] == 'extrusion' and made['turned'] and made['cut'] == 0
    _p, name2, square = _project()
    level = _p.add_solid_cylinder(name2, (0, 0, 0), 0.4, 3.0, name='rod', unit='in')
    flat, flat_made = solids.brick_mesh(square, [level], 0.125 * INCH)
    assert (made['bricks'], part.num_nodes) == (flat_made['bricks'], flat.num_nodes)
    local = _local(geometry, rod, part.node_xyz) / INCH
    assert np.hypot(local[:, 0], local[:, 1]).max() == pytest.approx(0.4)
    assert np.abs(local[:, 2]).max() == pytest.approx(1.5)
    assert made['quality'] == pytest.approx(flat_made.get('quality', made['quality']))
    assert list(part.group_name) == ['rod']


def test_a_turned_rod_sunk_into_a_block_ends_on_its_face_and_is_tied():
    """A rod entering a block's top at 50 degrees, its lower end inside:
    swept along its own axis, each line of nodes cut where it leaves the
    block and spread to the free end, so the last layer lies on the
    block's top, no node of the rod inside it, every end node tied, the
    structure one piece with six free modes, and the sheared end's worst
    brick said."""
    from visualdynamics.core.fem import MATERIALS, GroupProperties, Model

    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (4, 3, 1), name='block', unit='in')
    rod = project.add_solid_cylinder(name, (0.0, 0, 1.5), 0.4, 4.0, angles=(0, -50, 0),
                                     name='rod', unit='in')
    both = project.combine_solids(name, 'union', [block, rod], 'both')
    assert [e['method'] for e in solids.piece_plan(geometry, [both], 0.25 * INCH,
                                                   {rod: 0.1 * INCH})] == ['extrusion'] * 2
    made = project.mesh_solids(name, [both], 0.25, unit='in', sizes={rod: 0.1})
    swept = made['pieces'][1]
    assert swept['name'] == 'rod' and swept['method'] == 'extrusion' and swept['turned']
    assert swept['cut'] > 100 and 0.2 < swept['quality'] < 1.0
    rows = np.flatnonzero(geometry.elem_group == _group(geometry, 'rod'))
    nodes = np.unique(np.concatenate([geometry.elem_conn[r] for r in rows]))
    xyz = geometry.node_xyz[geometry.node_index(nodes)]
    on_top = np.isclose(xyz[:, 2], 0.5 * INCH, atol=1e-9 * INCH)
    assert on_top.sum() == swept['cut']
    assert not solids.contains(geometry, block, xyz[~on_top]).any(), 'none inside the block'
    seam, = geometry.mates
    assert seam.groups == (_group(geometry, 'rod'), _group(geometry, 'block'))
    assert made['ties'] + made['seam_shared'] == swept['cut']
    for group in geometry.group_id:
        geometry.group_properties[int(group)] = GroupProperties(MATERIALS['1018 steel'])
    model = Model.from_geometry(geometry)
    frequencies = np.asarray(model.eigensolution(num_modes=8).frequency)
    assert len(model.pieces()) == 1 and int(np.sum(frequencies < 1.0)) == 6


def test_a_turned_strut_is_swept_along_the_axis_that_meets_the_block():
    """A strut turned 30 degrees about Z, running into a block's side at
    mid-height: lines along its thin Z lie wholly inside the block or
    wholly out, so the sweep runs along its length, the axis whose lines
    end on the block's side."""
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (2, 2, 1), name='block', unit='in')
    strut = project.add_solid_block(name, (1.8, 0.6, 0), (2.4, 0.4, 0.4), angles=(0, 0, 30),
                                    name='strut', unit='in')
    both = project.combine_solids(name, 'union', [block, strut], 'both')
    made = project.mesh_solids(name, [both], 0.2, unit='in')
    swept = made['pieces'][1]
    assert swept['method'] == 'extrusion' and swept['turned'] and swept['cut'] > 0
    rows = np.flatnonzero(geometry.elem_group == _group(geometry, 'strut'))
    nodes = np.unique(np.concatenate([geometry.elem_conn[r] for r in rows]))
    xyz = geometry.node_xyz[geometry.node_index(nodes)]
    at_side = (np.isclose(xyz[:, 0], 1.0 * INCH, atol=1e-9 * INCH)
               | np.isclose(xyz[:, 1], 1.0 * INCH, atol=1e-9 * INCH))
    assert at_side.sum() == swept['cut']
    assert not solids.contains(geometry, block, xyz[~at_side]).any(), 'none inside the block'


def test_a_turned_rod_through_the_side_of_a_block_is_fitted():
    """A rod whose lower end pokes out of the block's side: its lines
    run out, in and out again, which no one sweep holds; it is fitted,
    the reason said by the sweep."""
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (4, 3, 1), name='block', unit='in')
    rod = project.add_solid_cylinder(name, (1.0, 0, 1.4), 0.4, 3.0, angles=(0, -50, 0),
                                     name='rod', unit='in')
    with pytest.raises(solids.NotAnExtrusion, match='runs through the pieces before it'):
        solids.turned_sweep(geometry, rod, 0.1 * INCH, [block])


# ---- pieces as deep as unions go; bricks counted by volume (2026-10-20) ----

def test_pieces_go_down_through_unions_and_cuts_through_a_union():
    """A union inside a union gives up its operands; a subtraction whose
    first operand is a union is a cut piece per member, (A + B) - C is
    A - C and B - C, and a cut of that again adds its cut to each; a
    subtraction of one solid, an intersection and a pattern stay one
    piece (Brandon, 2026-10-20: the mesh listed only the first level)."""
    project, name, geometry = _project()
    a = project.add_solid_block(name, (0, 0, 0), (2, 2, 1), name='a', unit='in')
    b = project.add_solid_block(name, (2, 0, 0), (2, 2, 1), name='b', unit='in')
    c = project.add_solid_cylinder(name, (1, 0, 0), 0.3, 2.0, name='c', unit='in')
    d = project.add_solid_cylinder(name, (3, 0, 0), 0.3, 2.0, name='d', unit='in')
    e = project.add_solid_sphere(name, (6, 0, 0), 1.0, name='e', unit='in')
    ab = project.combine_solids(name, 'union', [a, b])
    cut = project.combine_solids(name, 'subtract', [ab, c])
    cut_again = project.combine_solids(name, 'subtract', [cut, d])
    whole = project.combine_solids(name, 'union', [cut_again, e])
    assert solids.piece_cuts(geometry, [whole]) == [(a, (d, c)), (b, (d, c)), (e, ())]
    assert solids.pieces(geometry, [whole]) == [a, b, e]
    _p2, name2, other = _project()
    f = _p2.add_solid_block(name2, (0, 0, 0), (2, 2, 1), unit='in')
    g = _p2.add_solid_cylinder(name2, (0, 0, 0), 0.3, 2.0, unit='in')
    h = _p2.add_solid_sphere(name2, (3, 0, 0), 1.0, unit='in')
    lone_cut = _p2.combine_solids(name2, 'subtract', [f, g])
    common = _p2.combine_solids(name2, 'intersect', [h, _p2.add_solid_block(
        name2, (3, 0, 0), (1, 1, 1), unit='in')])
    for one in (lone_cut, common):
        assert solids.piece_cuts(other, [one]) == [(one, ())]
    both = _p2.combine_solids(name2, 'union', [lone_cut, common])
    assert solids.piece_cuts(other, [both]) == [(lone_cut, ()), (common, ())]


def test_a_cut_piece_is_meshed_less_its_cut_and_planned_by_its_own_id(monkeypatch):
    """(slab + bar) - hole meshed in pieces: the fitter is handed the
    slab less the hole, then the bar less the hole less the slab's
    piece, each at the size given under its own id; the plan reads each
    piece as meshed, its cuts named; and the solids made for the mesh
    are gone after. The fit itself is stood in for: whether one holds
    differs by machine."""
    project, name, geometry = _project()
    slab = project.add_solid_block(name, (0, 0, 0), (4, 2, 1), name='slab', unit='in')
    bar = project.add_solid_block(name, (1.5, 0, 1), (1, 2, 1), name='bar', unit='in')
    # tilted, so the part is no extrusion and is meshed in pieces
    hole = project.add_solid_cylinder(name, (1.5, 0, 0.5), 0.3, 4.0, angles=(30, 0, 0),
                                      name='hole', unit='in')
    joined = project.combine_solids(name, 'union', [slab, bar])
    cut = project.combine_solids(name, 'subtract', [joined, hole], 'part')
    count = len(geometry.solids)
    plan = solids.piece_plan(geometry, [cut], 0.25 * INCH, {bar: 0.125 * INCH})
    assert [(e['id'], e['name'], e['cuts']) for e in plan] == [
        (slab, 'slab', (hole,)), (bar, 'bar', (hole,))]
    assert [e['size'] for e in plan] == [0.25 * INCH, 0.125 * INCH]
    assert len(geometry.solids) == count, 'the plan made and took off its cuts'
    with solids.meshed_as(geometry, slab, (hole,)) as read_as:
        assert np.isfinite(hexmesh.likely_to_fold(geometry, [read_as], 0.25 * INCH)['across'])
    assert len(geometry.solids) == count
    handed = []
    _stand_in(monkeypatch)
    standing = hexmesh.overlay_mesh

    def record(geometry_, ids, size, **kwargs):
        item = solids.solid(geometry_, ids[0])
        inner = solids.solid(geometry_, item.operands[0])
        handed.append((item.kind, inner.kind, inner.operands, size))
        return standing(geometry_, ids, size, **kwargs)

    monkeypatch.setattr(hexmesh, 'overlay_mesh', record)
    made = project.mesh_solids(name, [cut], 0.25, unit='in', sizes={bar: 0.125})
    assert [(m['id'], m['name']) for m in made['pieces']] == [(slab, 'slab'), (bar, 'bar')]
    # the slab less the hole; then the bar less the hole, less the slab's piece
    assert handed[0][:3] == ('subtract', 'block', ())
    assert handed[1][:3] == ('subtract', 'subtract', (bar, hole))
    assert [h[3] for h in handed] == pytest.approx([0.25 * INCH, 0.125 * INCH])
    assert len(geometry.solids) == count, 'and so did the mesh'


def test_bricks_are_counted_from_the_volume():
    """A block's count is its volume over a brick's; a hollow box's is
    its walls', not its box's cells, within a few percent."""
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (4, 2, 1), unit='in')
    assert solids.brick_count(geometry, [block], 0.25 * INCH) == 16 * 8 * 4
    outer = project.add_solid_block(name, (10, 0, 0), (4, 4, 4), unit='in')
    inner = project.add_solid_block(name, (10, 0, 0), (3.5, 3.5, 3.5), unit='in')
    box = project.combine_solids(name, 'subtract', [outer, inner])
    walls = (64 - 3.5 ** 3) / 0.25 ** 3
    assert solids.brick_count(geometry, [box], 0.25 * INCH) == pytest.approx(walls, rel=0.05)
    assert hexmesh.grid_cells(geometry, [box], 0.25 * INCH) == 16 ** 3


def test_a_mostly_empty_box_is_held_to_its_bricks_and_its_grid_to_memory(monkeypatch):
    """Two small blocks far apart: their grid is mostly empty, its cells
    past the brick limit and its bricks under it, and it meshes, the
    limit read from the bricks; the grid's own cells are held to
    `GRID_LIMIT`, refused by name past it."""
    project, name, geometry = _project()
    a = project.add_solid_block(name, (0, 0, 0), (1, 1, 1), name='a', unit='in')
    b = project.add_solid_block(name, (9, 9, 9), (1, 1, 1), name='b', unit='in')
    size = 0.25 * INCH
    cells = solids.grid_cells(geometry, [a, b], size)
    bricks = solids.brick_count(geometry, [a, b], size)
    assert bricks == 2 * 4 ** 3 and cells > 100 * bricks
    monkeypatch.setattr(solids, 'MESH_LIMIT', 10 * bricks)
    _part, made = solids.brick_mesh(geometry, [a, b], size)
    assert made['bricks'] == bricks and cells > solids.MESH_LIMIT
    monkeypatch.setattr(solids, 'GRID_LIMIT', cells - 1)
    with pytest.raises(ValueError, match=f'the grid over these parts at that size is {cells:,} '
                                         'cells, over the'):
        solids.brick_mesh(geometry, [a, b], size)
    monkeypatch.setattr(solids, 'GRID_LIMIT', cells)
    monkeypatch.setattr(solids, 'MESH_LIMIT', bricks - 1)
    with pytest.raises(ValueError, match=f'about {bricks:,} bricks at that size, over the limit'):
        solids.brick_mesh(geometry, [a, b], size)


def test_a_fit_counts_the_bricks_it_keeps_again(monkeypatch):
    """A part thinner than the count's samples can be undercounted: the
    fit counts the bricks its grid keeps and refuses past the limit."""
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (4, 2, 1), unit='in')
    hole = project.add_solid_cylinder(name, (0, 0, 0), 0.3, 2.0, angles=(30, 0, 0), unit='in')
    cut = project.combine_solids(name, 'subtract', [block, hole])
    monkeypatch.setattr(hexmesh, 'brick_count', lambda *args: 0)
    monkeypatch.setattr(solids, 'brick_count', lambda *args: 0)
    monkeypatch.setattr(hexmesh, 'MESH_LIMIT', 100)
    with pytest.raises(ValueError, match='bricks at that size, over the limit of 100'):
        hexmesh.overlay_mesh(geometry, [cut], 0.25 * INCH)


def test_a_sweep_counts_the_bricks_it_keeps_again(monkeypatch):
    """The sweep, too, counts what it keeps and refuses past the limit,
    the count from the volume aside."""
    project, name, geometry = _project()
    block = project.add_solid_block(name, (0, 0, 0), (4, 2, 1), unit='in')
    monkeypatch.setattr(solids, 'brick_count', lambda *args: 0)
    monkeypatch.setattr(solids, 'MESH_LIMIT', 100)
    with pytest.raises(ValueError, match='512 bricks at that size, over the limit of 100'):
        solids.extrusion_mesh(geometry, [block], 0.25 * INCH)
