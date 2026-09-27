"""The BARC, built from planes, against the published model of it
(Brandon, 2026-09-26; `visualdynamics.demo.barc`, the reference frozen
in `testdata/barc/` from https://wiki.sem.org/wiki/BARC).

What is pinned: the geometry is the solid model's — its mass within a
percent of the solids' volumes, its parts where the drawing puts them,
one connected structure; and the modes answer to the reference — the
first ten paired one-to-one and in order by MAC, each within five
percent. The joint model is what makes the last hold: tie each bolt at
a point instead of over its washer and the check fails.
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


def test_the_modes_answer_to_the_reference(solved):
    model, shapes = solved
    assert int(np.sum(shapes.frequency == 0.0)) == 6, 'free-free'
    rows = barc.compare(shapes, model)
    first = rows[:10]
    assert [row['mode'] for row in first] == list(range(1, 11)), (
        'the first ten pair one-to-one, in order')
    assert all(row['mac'] > 0.9 for row in first), [r['mac'] for r in first]
    # a few percent stiff, consistently: the washer patches (0 to +6%)
    assert all(-1.0 < row['error'] < 7.0 for row in first), [
        round(r['error'], 1) for r in first]


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


def test_the_reference_is_the_shared_data():
    ref = barc.reference()
    assert len(ref['frequency']) == 30 and np.all(ref['frequency'][:6] < 1.0)
    assert ref['frequency'][6] == pytest.approx(185.7, abs=0.1)
    assert len(ref['dof']) == 118 and ref['shape'].shape == (118, 30)
    # moved into the solid model's frame: the Bench top is the beam's
    # upper face, 5.125 in
    assert ref['coordinates'][:, 1].max() / 0.0254 == pytest.approx(5.128,
                                                                   abs=0.01)


def test_the_apps_correlation_finds_every_reference_point():
    """The reference's sensors sit on the parts' surfaces, the model's
    nodes on their mid-surfaces: the box's are half a wall, 0.125 in,
    away, and the correlation's tolerance dropped all 39 of them until
    it allowed half the thickest plate (2026-09-26). Through the same
    projection the comparison screen uses, every point is found, and the
    MAC it reads agrees with the demo's own."""
    from visualdynamics.core.correlate import project_shapes
    from visualdynamics.core.shapes import cross_mac

    model = barc.build()
    shapes = model.eigensolution(maximum_frequency=1200.0)
    reference_shapes, reference_geometry = barc.reference_shapes()
    projected, report = project_shapes(shapes, barc.geometry(),
                                       reference_shapes, reference_geometry)
    assert report['matched'] == report['total'] == 59
    assert not report['dropped']
    mac = cross_mac(reference_shapes, projected)
    rows = barc.compare(shapes, model)
    for i, row in enumerate(rows[:8]):
        assert mac[6 + i, 5 + row['mode']] == pytest.approx(row['mac'],
                                                            abs=0.02)


def test_the_project_is_the_app_workflow(tmp_path):
    """What the docs page walks through, as the project verbs the app
    calls: solve the geometry, compare with the reference, match."""
    import visualdynamics

    project = barc.project()
    project.save(tmp_path / 'barc.vdyn')
    project = visualdynamics.load(tmp_path / 'barc.vdyn')
    solved = project.solve_modes('BARC', maximum_frequency=1200.0)
    mac = project.comparison_mac('Reference Modes', solved)
    assert mac.shape[0] == 30
    first_elastic = [int(np.argmax(mac[6 + i])) for i in range(3)]
    assert first_elastic == [6, 7, 8], 'the first three pair in order'
    assert min(mac[6 + i, j] for i, j in enumerate(first_elastic)) > 0.98


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


def test_the_websites_table_is_the_models(solved):
    """The examples page states the comparison as numbers; they must be
    the model's, rounded as printed."""
    import pathlib
    import re

    page = (pathlib.Path(__file__).resolve().parents[1] / 'web' / 'launch'
            / 'examples.html').read_text(encoding='utf-8')
    table = page.split('id="barc-comparison"')[1].split('</table>')[0]
    printed = [[cell.replace('&minus;', '-') for cell in
                re.findall(r'<td>(.*?)</td>', row)]
               for row in re.findall(r'<tr>(.*?)</tr>', table)
               if '<td>' in row]
    model, shapes = solved
    rows = barc.compare(shapes, model)[:len(printed)]
    assert len(printed) == 8
    for cells, row in zip(printed, rows, strict=True):
        assert cells == [f"{row['reference']:.1f}", f"{row['model']:.1f}",
                         f"{row['error']:+.1f}%", f"{row['mac']:.3f}"]


def test_a_picture_of_the_barc_stands_it_up(solved):
    """The model is built y-up, as its solid model is; a 3-D scene is
    z-up, and the website's figure first drew the BARC lying on its
    side. `upright` turns it a quarter turn about x for the figure and
    the downloads tile — nodes and every shape DOF with them."""
    _model, shapes = solved
    geometry = barc.geometry()
    turned, moved = barc.upright(geometry, shapes)
    height = turned.node_xyz.max(axis=0) - turned.node_xyz.min(axis=0)
    assert np.argmax(height) == 2               # the Bench is on top
    assert np.allclose(turned.node_xyz[:, 2], geometry.node_xyz[:, 1])
    assert list(moved.coordinate[:6]) == ['1X+', '1Z+', '1Y-',
                                          '1RX+', '1RZ+', '1RY-']
    assert np.array_equal(moved.shape_matrix, shapes.shape_matrix)
    assert np.array_equal(barc.upright(geometry).node_xyz, turned.node_xyz)
    assert np.array_equal(geometry.node_xyz, barc.geometry().node_xyz), \
        'the model itself is not turned'


def test_the_downloadable_project_opens_on_the_answer(tmp_path):
    """The downloads page's BARC is cut by tools/cut_examples.py from
    `project(solved=True)`: the modes solved and the first ten elastic
    reference modes matched, one-to-one and in order, as the docs'
    table has them."""
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
    assert project['BARC Modes'].frequency.max() <= barc.SOLVE_TO
    matched = project['Matched Modes']
    assert (matched.first, matched.second) == ('Reference Modes',
                                               'BARC Modes')
    assert [tuple(p) for p in matched.pairs] == [(i, i) for i in range(6, 16)]
    assert min(matched.macs) > 0.8


def test_the_reference_is_credited_wherever_it_travels():
    """The reference modes are another group's finite element model
    (Brandon, 2026-09-26: credit them, and link the wiki). The credit
    rides with them — the project's own About the Reference report, the
    shape set, the download's README — and stands on the pages that show
    them: the workflow guide, the examples page, the downloads page."""
    import importlib.util
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    authors = 'R. Schultz, T. Schoenherr and B. Owens'
    assert barc.CITATION.startswith(authors) and barc.WIKI in barc.CITATION
    project = barc.project()
    about = project['About the Reference'].blocks[0]['text']
    assert barc.CITATION in about
    assert set(project['Reference Modes'].comment) == {barc.CITATION}, \
        'every mode carries it'
    spec = importlib.util.spec_from_file_location(
        'cut_examples', root / 'tools' / 'cut_examples.py')
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    readme = ' '.join(tool._barc_credit().split())
    assert barc.CITATION in readme
    for page in ('docs/guide/workflows/fem-workflow.md',
                 'web/launch/examples.html', 'web/launch/downloads.html',
                 'testdata/barc/README.md'):
        text = (root / page).read_text(encoding='utf-8')
        assert barc.SOURCE in text, f'{page} does not credit the source'
        assert 'Schoenherr' in text and 'IMAC 2021' in text, page
