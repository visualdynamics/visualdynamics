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
    assert list(geometry.block_name) == ['box', 'right channel',
                                         'left channel', 'beam', 'bolts']
    bolts = int(geometry.block_id[list(geometry.block_name).index('bolts')])
    assert geometry.block_properties[bolts].material is RIGID
    box = int(geometry.block_id[0])
    assert geometry.block_properties[box].thickness == pytest.approx(0.25 * 0.0254)
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
    assert list(built.block_name) == ['box', 'right channel', 'left channel',
                                      'beam']


def test_the_docs_table_typed_into_add_plane_is_the_demos_mesh():
    """The page's app steps: a new geometry in inches and the planes table
    typed row by row into Add Plane (the project verb each Add records)
    must give the demo's planes node for node — same ids, same places,
    same plates — which is what lets the page say a model built by hand
    solves to the same modes."""
    import pathlib

    import visualdynamics
    from visualdynamics import mesh

    page = (pathlib.Path(__file__).resolve().parents[1] / 'docs' / 'guide'
            / 'workflows' / 'fem-workflow.md').read_text(encoding='utf-8')
    table = page.split('| Plane | Block | Corner | Edge A | Edge B |')[1]
    rows = [[cell.strip() for cell in line.strip('|').split('|')]
            for line in table.split('\n\n')[0].splitlines()[2:]]

    def vector(cell):
        return tuple(float(v) for v in cell.replace('\u2212', '-')
                     .strip('()').split(','))

    project = visualdynamics.Project('p')
    name = project.new_geometry(unit='in')
    for _plane, block, corner, edge_a, edge_b in rows:
        project.add_plane(name, vector(corner), vector(edge_a),
                          vector(edge_b), barc.SIZE, block, unit='in')
    typed = project[name]
    demo = mesh.assemble(*barc._planes(barc.SIZE))
    assert len(rows) == 12
    assert np.array_equal(typed.node_id, demo.node_id)
    assert np.allclose(typed.node_xyz, demo.node_xyz)
    assert all(np.array_equal(a, b) for a, b in zip(typed.elem_conn,
                                                    demo.elem_conn,
                                                    strict=True))
    assert list(typed.block_name) == list(demo.block_name)


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


def test_the_downloadable_project_opens_solved(tmp_path):
    """The downloads page's BARC is cut by tools/cut_examples.py from
    `project(solved=True)`: the model and its modes, solved."""
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
    project = visualdynamics.Project.open(tmp_path / f'{projects[0]}.vdyn')
    assert sorted(project.keys()) == ['BARC', 'BARC Modes']
    assert project['BARC Modes'].frequency.max() <= barc.SOLVE_TO
