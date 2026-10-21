"""The BARC, built from planes (Brandon, 2026-09-26;
`visualdynamics.demo.barc`, from https://wiki.sem.org/wiki/BARC).

What is pinned: the geometry is the solid model's — its mass within a
percent of the solids' volumes, its parts where the drawing puts them,
one connected structure; the bolts are the patches the documentation
tells a person to pick; and the modes stay where they were when the
model was checked against the finite element models shared on the wiki
(2026-09-27: the check was kept, and the wiki's modes left out of the
project — Brandon).
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.fem import RIGID
from visualdynamics.demo import barc

INCH3 = 0.0254 ** 3


@pytest.fixture(scope='module')
def solved():
    model = barc.build()
    return model, model.eigensolution(maximum_frequency=2000.0)


def test_the_geometry_is_the_solid_model():
    geometry = barc.geometry()
    assert list(geometry.group_name) == ['box', 'right channel',
                                         'left channel', 'beam', 'bolts']
    bolts = int(geometry.group_id[list(geometry.group_name).index('bolts')])
    assert geometry.group_properties[bolts].material is RIGID
    box = int(geometry.group_id[0])
    assert geometry.group_properties[box].thickness == pytest.approx(0.25 * 0.0254)
    inch = geometry.node_xyz / 0.0254
    assert inch[:, 0].min() == pytest.approx(-2.875)            # mid-surface
    assert inch[:, 1].max() == pytest.approx(barc.BEAM_Y)
    # the slot: nothing on the top wall between -0.25 and 0.25
    top = np.abs(inch[:, 1] - barc.BOX_MID) < 1e-6
    assert not np.any(top & (np.abs(inch[:, 0]) < barc.SLOT_HALF - 1e-6))


def test_it_weighs_what_the_solid_model_does(solved):
    """The STEP's solids: box 16.83 in³, channels 0.453 in³ each, beam
    0.615 in³, of 6061-T6. The plates carry no bolt holes, and weigh
    half a percent more."""
    model, _shapes = solved
    solid = (16.8299 + 2 * 0.4527 + 0.6154) * INCH3 * barc.MATERIAL.density
    assert model.structural_mass == pytest.approx(solid, rel=0.01)
    assert len(model.pieces()) == 1


#: the first ten elastic modes of the model as built at `barc.SIZE`, Hz —
#: four bolts at each end of the beam since 2026-10-11, each tying the
#: element it passes through, within 0.2 % (MAC 0.999 and up) of the
#: build with one bolt there that was checked against the wiki's models
#: (194.5, 212.5, 265.3, 461.8, 494.9, 581.0, 593.5, 681.4, 1128.1,
#: 1170.9)
CHECKED = (194.4, 212.4, 264.8, 461.6, 494.4, 580.3, 593.4, 681.1,
           1125.6, 1168.8)


def test_the_modes_stay_where_they_were_checked(solved):
    """Free-free, and the first ten elastic modes within half a percent
    of the build that was checked against the wiki's models: a change to
    the planes, the joints or the solver that moves them is a change to
    look at, not a drift to accept."""
    _model, shapes = solved
    assert int(np.sum(shapes.frequency == 0.0)) == 6, 'free-free'
    elastic = [f for f in shapes.frequency if f > 1.0][:10]
    assert np.allclose(elastic, CHECKED, rtol=0.005), np.round(elastic, 1)


def test_each_bolt_ties_the_elements_it_clamps():
    """The rule the docs give a person to click: under a foot bolt the
    element it passes through and the eight around it, the washer's
    area, and through the beam the element it passes through alone (four
    bolts at each end since 2026-10-11) — each tied to the part below,
    every patch node that part does not share linked once."""
    from visualdynamics import mesh

    planes = mesh.assemble(*barc._planes(barc.SIZE))
    patches = [barc.bolt_patch(planes, bolt) for bolt in barc.BOLTS]
    assert [len(patch) for patch, _below in patches] == [9] * 8 + [1] * 8
    assert [below for _patch, below in patches] == (
        ['box'] * 8 + ['left channel'] * 4 + ['right channel'] * 4)
    # the beam bolt's element is the one it passes through
    xyz = planes.node_xyz / 0.0254
    for (patch, _below), bolt in zip(patches[8:], barc.BOLTS[8:]):
        row = int(np.flatnonzero(planes.elem_id == patch[0])[0])
        corners = xyz[planes.node_index(planes.elem_conn[row])]
        assert np.allclose(corners.mean(axis=0)[[0, 2]], bolt[0])
    geometry = barc.geometry()
    links = geometry.elements_in('bolts')
    assert len(links) == 8 * 16 + 8 * 4


def test_the_project_is_the_app_workflow(tmp_path):
    """What the docs page walks through, as the project verbs the app
    calls: the saved project opened, and Solve Modes on it."""
    import visualdynamics

    project = barc.project()
    assert list(project.keys()) == ['BARC'], 'the model alone'
    project.save(tmp_path / 'barc.vdyn')
    project = visualdynamics.load(tmp_path / 'barc.vdyn')
    solved = project.solve_modes('BARC', maximum_frequency=1200.0)
    elastic = [f for f in project[solved].frequency if f > 1.0][:3]
    assert np.allclose(elastic, CHECKED[:3], rtol=0.005)


def test_the_docs_page_builds_the_demos_mesh():
    """The workflow page writes the planes out for a reader; run its
    code, and it must build the demo's mesh node for node — a page that
    drifted from the model would teach a different one."""
    import pathlib
    import re

    page = (pathlib.Path(__file__).resolve().parents[1] / 'docs' / 'guide'
            / 'workflows' / 'fem-workflow.md').read_text(encoding='utf-8')
    planes = next(block for block in
                  re.findall(r'```python\n(.*?)```', page, re.DOTALL)
                  if 'mesh.assemble' in block)
    room: dict = {}
    exec(planes, room)  # noqa: S102 — the page's own example
    built = room['barc']
    demo = barc.geometry()
    planes_only = demo.num_nodes                  # the links add no node
    assert built.num_nodes == planes_only
    assert np.allclose(np.sort(built.node_xyz, axis=0),
                       np.sort(demo.node_xyz, axis=0))
    assert list(built.group_name) == ['box', 'right channel', 'left channel',
                                      'beam']


def test_the_docs_table_typed_into_add_plane_is_the_demos_mesh():
    """The page's app steps: a new geometry in inches and the planes table
    typed row by row into Add Plane (the project verb each Add records)
    must give the demo's planes node for node — same ids, same places,
    same plates — which is what lets the page say a model built by hand
    solves to the same modes. The table is centers and widths since the
    pane replaced the dialog (2026-10-02), turned into the corner and
    edges the verb takes the way the pane turns them: each nonzero width
    along its axis, in X, Y, Z order, the corner half their sum below
    the center."""
    import pathlib

    import visualdynamics
    from visualdynamics import mesh

    page = (pathlib.Path(__file__).resolve().parents[1] / 'docs' / 'guide'
            / 'workflows' / 'fem-workflow.md').read_text(encoding='utf-8')
    table = page.split('| Plane | Element group | Center | Widths |')[1]
    rows = [[cell.strip() for cell in line.strip('|').split('|')]
            for line in table.split('\n\n')[0].splitlines()[2:]]

    def vector(cell):
        return tuple(float(v) for v in cell.replace('\u2212', '-')
                     .strip('()').split(','))

    project = visualdynamics.Project('p')
    name = project.new_geometry(unit='in')
    for _plane, block, center, widths in rows:
        widths = np.array(vector(widths))
        assert int(np.sum(widths == 0.0)) == 1, 'a plate names its plane'
        edges = [tuple(widths[i] * np.eye(3)[i]) for i in range(3)
                 if widths[i] > 0.0]
        corner = tuple(np.array(vector(center)) - 0.5 * np.sum(edges, axis=0))
        project.add_plane(name, corner, *edges, barc.SIZE, block, unit='in')
    typed = project[name]
    demo = mesh.assemble(*barc._planes(barc.SIZE))
    assert len(rows) == 12
    assert np.array_equal(typed.node_id, demo.node_id)
    assert np.allclose(typed.node_xyz, demo.node_xyz)
    assert all(np.array_equal(a, b) for a, b in zip(typed.elem_conn,
                                                    demo.elem_conn,
                                                    strict=True))
    assert list(typed.group_name) == list(demo.group_name)


def test_the_barc_opens_upright_without_being_turned():
    """The model is built y-up, as its solid model is, and every 3-D view
    used to open z-up and draw it lying on its side; the website figure
    turned its nodes to stand it up. It carries its own view now, and a
    view is only how it is looked at (2026-09-27)."""
    import pyvista as pv

    from visualdynamics import View
    from visualdynamics.viz.geometry import add_geometry, place_view

    geometry = barc.geometry(0.5)
    assert geometry.view == View(eye=(1, 1, -1), up=(0, 1, 0))
    plotter = pv.Plotter(off_screen=True)
    try:
        add_geometry(plotter, geometry)
        place_view(plotter, geometry.opening_view, render=False)
        _position, _focus, up = plotter.camera_position
    finally:
        plotter.close()
    # up on screen leans on y, the Bench's way up, not on z
    assert np.argmax(np.abs(up)) == 1 and up[1] > 0


@pytest.mark.slow
def test_the_downloadable_projects_open_solved(tmp_path):
    """The downloads page's BARC is cut by tools/cut_examples.py from
    `project(solved=True)` and `solid_project(solved=True)`: the models
    and their modes, solved. Slow since the bricks joined it: the
    assembly of bricks solves in about ten seconds."""
    import importlib.util
    import pathlib

    import visualdynamics

    path = (pathlib.Path(__file__).resolve().parents[1] / 'tools'
            / 'cut_examples.py')
    spec = importlib.util.spec_from_file_location('cut_examples', path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    source, projects, _sentence, _about = tool.SETS[
        'VisualDynamics-examples-barc.zip']
    source(str(tmp_path))
    assert projects == ('barc', 'barc-bricks')
    project = visualdynamics.Project.open(tmp_path / f'{projects[0]}.vdyn')
    assert sorted(project.keys()) == ['BARC', 'BARC Modes']
    assert project['BARC Modes'].frequency.max() <= barc.SOLVE_TO
    project = visualdynamics.Project.open(tmp_path / f'{projects[1]}.vdyn')
    assert sorted(project.keys()) == ['BARC', 'BARC Modes',
                                      'Removable Component',
                                      'Removable Component Modes']
    elastic = [f for f in project['Removable Component Modes'].frequency
               if f > 1.0]
    assert np.allclose(elastic[:7], PART_CHECKED, rtol=0.005)
    assert 'top bolts' in list(project['Removable Component'].group_name), \
        'the bolt masses ride the saved project'


# ---- the BARC from bricks (2026-10-07) -----------------------------------

#: the first ten elastic modes of the assembly of bricks at `barc.SIZE`,
#: Hz — its bolts merged nodes since 2026-10-11, 0.3 to 3.4 % under the
#: build checked against the wiki's model and test (MAC 0.97 and up, in
#: order), which joined each foot over its whole footprint and read 1 to
#: 4 % above the measured modes: (191.8, 213.1, 271.6, 457.9, 490.7,
#: 581.5, 603.1, 696.7, 1150.0, 1226.0)
SOLID_CHECKED = (191.2, 210.7, 263.3, 442.4, 485.8, 570.1, 598.5, 689.6,
                 1124.3, 1197.6)
#: the removable component's first seven, 0.3 to 1.2 % under the build
#: checked against the shared model of it (MAC 0.999 and up), which was
#: 6 % under it for the first two and within 2 % above after: (280.0,
#: 435.6, 596.3, 1320.8, 1633.8, 1954.8, 2025.9)
PART_CHECKED = (278.6, 432.9, 592.2, 1305.0, 1627.8, 1949.3, 2014.7)


def test_the_bricks_are_the_solid_model_and_the_bolts_weigh_in():
    """The bricks' volume is the solid model's without its bolt holes
    (the box 16.9 in³ as a tube with its slot, the channels and beam
    as drawn); the bolts are point masses at their heads, eight #8 at
    the feet and eight at the beam carrying the shared model's two 1/4
    in screws between them, and nothing is rigid: the parts are joined
    by the nodes their bolts merged."""
    from visualdynamics.core.fem import Model

    geometry = barc.solid_geometry()
    assert list(geometry.group_name) == ['box', 'right channel',
                                         'left channel', 'beam',
                                         'foot bolts', 'top bolts']
    assert len(geometry.elements_in('foot bolts')) == 8
    assert len(geometry.elements_in('top bolts')) == 8
    model = Model.from_geometry(geometry)
    assert not model.rigid_links
    lb = 0.45359237
    assert sum(m.mass for m in model.masses) == pytest.approx(
        (8 * 0.00744 + 2 * 0.0125) * lb)
    bricks = (6 * 6 - 5.5 * 5.5 - 0.5 * 0.25) * 3.0 + 2 * (
        2 * 0.125 + 1.75 * 0.125) + 5 * 0.125
    assert model.structural_mass == pytest.approx(
        bricks * INCH3 * barc.MATERIAL.density)
    assert len(model.pieces()) == 1


def _shared(geometry, a, b):
    ids = dict(zip(geometry.group_name, geometry.group_id.tolist()))
    nodes = [set(np.concatenate([geometry.elem_conn[i] for i in np.flatnonzero(
        geometry.elem_group == ids[name])]).tolist()) for name in (a, b)]
    return nodes[0] & nodes[1]


def test_each_joint_ties_the_places_its_bolts_clamp_and_no_other():
    """Meshed apart, each channel's foot is 9 x 9 places coincident with
    the box and each channel's top 9 x 9 with the beam, none shared. Each
    joint is a set of mated surfaces, rigid (Brandon, 2026-10-20: the
    examples remade with the new joints): a foot's four bolts tie the
    washer's 4 x 4 places each, which share their middle rows into one
    7 x 7 square; a beam end's four bolts tie the four corners of the
    brick each passes through. The patch is as Mate Surfaces finds it,
    the part below the first side (both sides hold 81, and the first
    group picked follows), kept at the bolts' places and clicked off
    everywhere else; the second side stays the patch's whole 81."""
    geometry = barc.solid_geometry()
    ids = dict(zip(geometry.group_name, geometry.group_id.tolist()))
    sets = {m.name: m for m in geometry.mates}
    assert sorted(sets) == ['left foot bolts', 'left top bolts', 'right foot bolts',
                            'right top bolts']
    assert all(m.describe() == 'rigid' for m in sets.values())
    for side in ('right', 'left'):
        channel = f'{side} channel'
        for kind, below, above, places in (('foot', 'box', channel, 49),
                                           ('top', channel, 'beam', 16)):
            mated = sets[f'{side} {kind} bolts']
            assert mated.groups == (ids[below], ids[above])
            assert (len(mated.nodes_a), len(mated.nodes_b)) == (places, 81), (side, kind)
            assert not _shared(geometry, below, above), 'joined by ties, not merged'
    inch = geometry.node_xyz / 0.0254
    for (bx, bz), kind in barc.SOLID_BOLTS[8:]:
        side = 'right' if bx > 0 else 'left'
        places = barc.bolt_places(geometry, (bx, bz), kind)
        at = inch[geometry.node_index(sorted(places))]
        assert np.allclose(at.mean(axis=0)[[0, 2]], (bx, bz))
        assert np.allclose(np.ptp(at, axis=0), [0.125, 0.0, 0.125])
        # the joint's first side, the channel, is tied at the places'
        # partners: the four corners of the brick the bolt passes through
        tied = inch[geometry.node_index(list(sets[f'{side} top bolts'].nodes_a))]
        near = tied[np.hypot(tied[:, 0] - bx, tied[:, 2] - bz) < 0.1]
        assert len(near) == 4


def test_the_docs_solids_meshed_apart_and_mated_are_the_demos_bricks():
    """The page's bricks steps, as the project verbs the app records:
    the blocks table typed into Add Solid Block, each part's blocks
    subtracted from its first, the four parts meshed with Keep the parts
    apart, then the four joints the page makes in Mate Surfaces — at each
    foot the patch with its outer ring of 32 clicked out, at each end of
    the beam None and the 16 under its bolts clicked back — must give
    the demo's bricks, node for node and brick for brick, tied at the
    same places (2026-10-20; merged nodes before)."""
    import pathlib

    import visualdynamics

    page = (pathlib.Path(__file__).resolve().parents[1] / 'docs' / 'guide'
            / 'workflows' / 'fem-workflow.md').read_text(encoding='utf-8')
    table = page.split('| Block | Center | Widths | Name |')[1]
    rows = [[cell.strip() for cell in line.strip('|').split('|')]
            for line in table.split('\n\n')[0].strip().splitlines()[1:]]

    def numbers(cell):
        return [float(v) for v in cell.replace('\u2212', '-').replace('×', ',')
                .strip('()').split(',')]

    project = visualdynamics.Project('p')
    name = project.new_geometry(unit='in')
    parts, blocks = [], []
    for _block, center, widths, label in rows:
        if label and blocks:
            parts.append(blocks)
            blocks = []
        blocks.append(project.add_solid_block(name, numbers(center), numbers(widths),
                                              name=label, unit='in'))
    parts.append(blocks)
    made = [project.combine_solids(name, 'subtract', part) if len(part) > 1
            else part[0] for part in parts]
    project.mesh_solids(name, made, barc.SIZE, unit='in', parts_apart=True)
    typed = project[name]
    ids = dict(zip(typed.group_name, typed.group_id.tolist()))
    assert list(ids) == ['box', 'right channel', 'left channel', 'beam']
    inch = 0.0254
    for side in ('right channel', 'left channel'):
        for below, above in (('box', side), (side, 'beam')):
            groups = [ids[below], ids[above]]
            patch, = project.detect_mated_surfaces(
                name, groups, 0.25 * typed.node_spacing(groups))
            assert len(patch['nodes_a']) == len(patch['nodes_b']) == 81
            at = typed.node_xyz[typed.node_index(list(patch['nodes_a']))] / inch
            if below == 'box':
                # the page: click out the 32 around the foot's edge
                x, z = at[:, 0], at[:, 2]
                out = ((np.isclose(x, x.min()) | np.isclose(x, x.max()))
                       | (np.isclose(z, z.min()) | np.isclose(z, z.max())))
            else:
                # None, then the four corners of the brick under each bolt
                axes = [a for a, k in barc.SOLID_BOLTS if k == 'top'
                        and (a[0] > 0) == (side == 'right channel')]
                out = ~np.array([any(abs(p[0] - a[0]) < 0.07 and abs(p[2] - a[1]) < 0.07
                                     for a in axes) for p in at])
            kept = [n for n, o in zip(patch['nodes_a'], out) if not o]
            assert len(kept) == (49 if below == 'box' else 16)
            joint = 'foot' if below == 'box' else 'top'
            project.add_mated_surfaces(name, patch['groups'], kept, patch['nodes_b'],
                                       name=f'{side.split()[0]} {joint} bolts')
    demo = barc.solid_geometry()
    assert typed.num_nodes == demo.num_nodes
    assert np.allclose(np.sort(typed.node_xyz, axis=0),
                       np.sort(demo.node_xyz, axis=0))
    for group in ('box', 'right channel', 'left channel', 'beam'):
        assert len(typed.elements_in(group)) == len(demo.elements_in(group))
    sets = {m.name: m for m in demo.mates}
    for mated in typed.mates:
        assert set(mated.nodes_a) == set(sets[mated.name].nodes_a), mated.name
        assert set(mated.nodes_b) == set(sets[mated.name].nodes_b), mated.name


def test_the_removable_component_stays_where_it_was_checked():
    geometry = barc.solid_geometry('removable component')
    assert 'box' not in list(geometry.group_name)
    assert 'foot bolts' not in list(geometry.group_name), \
        'its foot bolts stay with the box'
    shapes = barc.solid_build('removable component').eigensolution(
        maximum_frequency=barc.PART_SOLVE_TO)
    assert int(np.sum(shapes.frequency == 0.0)) == 6, 'free-free'
    elastic = [f for f in shapes.frequency if f > 1.0][:7]
    assert np.allclose(elastic, PART_CHECKED, rtol=0.005), np.round(elastic, 1)
    with pytest.raises(ValueError, match='not one of'):
        barc.solid_geometry('bench')


@pytest.mark.slow
def test_the_bricks_stay_where_they_were_checked():
    shapes = barc.solid_build().eigensolution(maximum_frequency=barc.SOLVE_TO)
    assert int(np.sum(shapes.frequency == 0.0)) == 6, 'free-free'
    elastic = [f for f in shapes.frequency if f > 1.0][:10]
    assert np.allclose(elastic, SOLID_CHECKED, rtol=0.005), np.round(elastic, 1)


def test_a_coarser_mesh_of_bricks_still_holds_together():
    """Every box is cut at every other's faces before it is meshed: at
    a quarter inch a channel's eighth-inch web met a flange with no line
    of nodes there, hung from one edge, and the BARC read 76 Hz for
    192 (2026-10-07). Coarse now reads 1 to 5 % above the checked mesh:
    each foot bolt clamps three bricks across, which at a quarter inch
    is the whole of a one-inch foot (the website's figure is drawn at
    it)."""
    shapes = barc.solid_build('BARC', 0.25).eigensolution(
        maximum_frequency=1300.0)
    elastic = [f for f in shapes.frequency if f > 1.0][:8]
    assert np.allclose(elastic, SOLID_CHECKED[:8], rtol=0.06), \
        np.round(elastic, 1)


def test_the_guides_bricks_are_the_demos_steps():
    """The finite element workflow page's BARC of bricks is built from
    solids a student adds (2026-10-11): each block's row there is held
    to the call the demonstration journals for it — its center, its
    widths, its name — so the page cannot drift from the model the
    tests check."""
    import ast
    import pathlib
    import re

    from visualdynamics.project import Project

    project = Project('steps')
    name = project.new_geometry('BARC', unit='in')
    barc._parts(project, name, 'BARC')
    calls = [ast.parse(line).body[0].value for line in project.journal
             if line.startswith('project.add_solid_block(')]
    page = (pathlib.Path(__file__).resolve().parents[1] / 'docs' / 'guide'
            / 'workflows' / 'fem-workflow.md').read_text(encoding='utf-8')
    table = page.split('| Block | Center | Widths | Name |')[1].split('\n\n')[0]
    rows = [[cell.strip().replace('−', '-') for cell in line.strip('|').split('|')]
            for line in table.strip().splitlines()[1:]]
    assert len(rows) == len(calls) == 8

    def numbers(cell):
        return [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', cell)]

    for (block, center, widths, label), call in zip(rows, calls):
        args = [ast.literal_eval(a) for a in call.args]
        keywords = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
        assert np.allclose(numbers(center), args[1]), block
        assert np.allclose(numbers(widths), args[2]), block
        assert label == keywords.get('name', ''), block
