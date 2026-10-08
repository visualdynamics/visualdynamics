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
#: the build that was checked against the wiki's models
CHECKED = (194.5, 212.5, 265.3, 461.8, 494.9, 581.0, 593.5, 681.4,
           1128.1, 1170.9)


def test_the_modes_stay_where_they_were_checked(solved):
    """Free-free, and the first ten elastic modes within half a percent
    of the build that was checked against the wiki's models: a change to
    the planes, the joints or the solver that moves them is a change to
    look at, not a drift to accept."""
    _model, shapes = solved
    assert int(np.sum(shapes.frequency == 0.0)) == 6, 'free-free'
    elastic = [f for f in shapes.frequency if f > 1.0][:10]
    assert np.allclose(elastic, CHECKED, rtol=0.005), np.round(elastic, 1)


def test_each_bolt_ties_the_elements_its_washer_covers():
    """The rule the docs give a person to click: under a foot bolt the
    element it passes through and the eight around it, under a beam bolt
    the 4 x 4 around its node — each tied to the part below, every
    patch node that part does not share linked once."""
    from visualdynamics import mesh

    planes = mesh.assemble(*barc._planes(barc.SIZE))
    patches = [barc.washer_patch(planes, bolt) for bolt in barc.BOLTS]
    assert [len(patch) for patch, _below in patches] == [9] * 8 + [16] * 2
    assert [below for _patch, below in patches] == (
        ['box'] * 8 + ['left channel', 'right channel'])
    geometry = barc.geometry()
    links = geometry.elements_in('bolts')
    assert len(links) == 8 * 16 + 2 * 25


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
#: Hz — the build checked against the wiki's model and test: 1 to 4 %
#: above the measured modes, the shapes in the shared model's order
SOLID_CHECKED = (191.8, 213.1, 271.6, 457.9, 490.7, 581.5, 603.1, 696.7,
                 1150.0, 1226.0)
#: the removable component's first seven, checked against the shared
#: model of it: 6 % under for the first two, within 2 % above after
PART_CHECKED = (280.0, 435.6, 596.3, 1320.8, 1633.8, 1954.8, 2025.9)


def test_the_bricks_are_the_solid_model_and_the_bolts_weigh_in():
    """The bricks' volume is the solid model's without its bolt holes
    (the box 16.9 in³ as a tube with its slot, the channels and beam
    as drawn); the bolts are point masses at their heads, eight #8 at
    the feet and two 1/4 in at the beam, and the beam rests over the
    channels, tied at its bolts only."""
    from visualdynamics.core.fem import RIGID, Model

    geometry = barc.solid_geometry()
    assert list(geometry.group_name) == ['box', 'right channel',
                                         'left channel', 'beam', 'bolt ties',
                                         'foot bolts', 'top bolts']
    ids = dict(zip(geometry.group_name, geometry.group_id.tolist()))
    assert geometry.group_properties[ids['bolt ties']].material is RIGID
    assert len(geometry.elements_in('foot bolts')) == 8
    assert len(geometry.elements_in('top bolts')) == 2
    model = Model.from_geometry(geometry)
    lb = 0.45359237
    assert sum(m.mass for m in model.masses) == pytest.approx(
        (8 * 0.00744 + 2 * 0.0125) * lb)
    bricks = (6 * 6 - 5.5 * 5.5 - 0.5 * 0.25) * 3.0 + 2 * (
        2 * 0.125 + 1.75 * 0.125) + 5 * 0.125
    assert model.structural_mass == pytest.approx(
        bricks * INCH3 * barc.MATERIAL.density)
    assert len(model.pieces()) == 1
    inch = geometry.node_xyz / 0.0254
    beam = geometry.node_index(np.unique(np.concatenate(
        [geometry.elem_conn[i] for i in np.flatnonzero(
            geometry.elem_group == ids['beam'])])))
    assert inch[beam, 1].min() == pytest.approx(5.0 + barc.BEAM_GAP), \
        'the beam is not fused to the channels'


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
    192 (2026-10-07). Coarse now reads within a percent and a half of
    the checked mesh."""
    shapes = barc.solid_build('BARC', 0.25).eigensolution(
        maximum_frequency=1300.0)
    elastic = [f for f in shapes.frequency if f > 1.0][:8]
    assert np.allclose(elastic, SOLID_CHECKED[:8], rtol=0.015), \
        np.round(elastic, 1)
