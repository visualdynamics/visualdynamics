"""The Frame-Wing, built from blocks of bricks (2026-09-30;
`visualdynamics.demo.frame_wing`, from
https://wiki.sem.org/wiki/Round_Robin_Frame_Structure).

The fast tests pin the geometry — the dimensions read off the shared
models, the 41 inserts and the 18 wing holes, the blocks and what they
are made of, the project and its verbs — on a coarse mesh. The `slow`
ones solve the models as the example is cut and hold them to the
frequencies that were measured: the wiki's table for frame SN003, and
the wings' modes fitted from the shared test FRFs with `fit_modes`
(2026-09-30). The measured numbers are frozen here; none of the wiki's
files are in the repository.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.demo import frame_wing

INCH = 0.0254
#: a coarse mesh for the fast tests: a quarter inch, two bricks across a
#: member, no fillets
COARSE = 0.25

#: the wiki's measured frequencies of frame SN003, Hz, laser vibrometer
MEASURED_FRAME = [234.18, 291.71, 624.38, 652.81, 715.31, 751.99,
                  1118.75, 1155.31, 1241.27]
#: the wings' modes, fitted from the shared test FRFs of units 003A and
#: 003B (the thin wing's fit missed its torsion modes: the excitation
#: sat on their node line, so they are matched by nearest and the rest
#: pass by)
MEASURED_THIN = [144.7, 285.8, 472.0, 500.4, 700.8, 709.3, 934.5, 992.7, 1197.7]
MEASURED_THICK = [108.6, 302.7, 310.3, 596.3, 644.2, 985.1, 1003.8, 1409.8, 1477.0]


def test_the_frame_is_the_shared_models_frame():
    geometry = frame_wing.geometry()            # meshing is fast; solving is not
    inch = geometry.node_xyz / INCH
    assert np.allclose(inch.min(axis=0), [-8, -3, -0.5], atol=1e-9)
    assert np.allclose(inch.max(axis=0), [8, 3, 0], atol=1e-9)
    assert sorted(geometry.group_name) == ['frame', 'inserts']
    assert set(geometry.elem_type.tolist()) == {115}, 'bricks only'
    # 41 inserts: a hole through and the insert in its top, each a
    # column of bricks
    inserts = geometry.elements_in('inserts')
    centers = np.array([inch[geometry.node_index(geometry.elem_conn[
        int(np.flatnonzero(geometry.elem_id == e)[0])])].mean(axis=0)
        for e in inserts])
    holes = np.array(frame_wing.INSERTS)
    nearest = np.argmin(np.linalg.norm(centers[:, None, :2] - holes[None], axis=2),
                        axis=1)
    assert len(set(nearest.tolist())) == len(frame_wing.INSERTS) == 41
    assert np.linalg.norm(centers[:, :2] - holes[nearest], axis=1).max() < frame_wing.HOLE_RADIUS
    assert centers[:, 2].min() > -frame_wing.INSERT_DEPTH - frame_wing.SIZE, 'the top only'
    assert centers[:, 2].max() < 0.0
    for block in geometry.group_id:
        assert geometry.group_properties[int(block)].kind == 'solid'
    assert geometry.view == frame_wing.VIEW


def test_a_wing_alone_is_a_plate_with_eighteen_holes():
    thin = frame_wing.geometry('thin wing alone', wing_size=COARSE)
    inch = thin.node_xyz / INCH
    assert np.allclose(inch.min(axis=0), [frame_wing.WING_X, -11, frame_wing.WING_GAP], atol=1e-9)
    assert inch[:, 2].max() == pytest.approx(frame_wing.WING_GAP + frame_wing.WING_THICKNESS['thin wing'])
    assert list(thin.group_name) == ['thin wing']
    # every hole is a void: no brick center within its radius
    centers = np.array([inch[thin.node_index(c)].mean(axis=0) for c in thin.elem_conn])
    for hx, hy in frame_wing.WING_HOLES:
        assert np.hypot(centers[:, 0] - hx, centers[:, 1] - hy).min() > frame_wing.WING_HOLE_RADIUS
    assert len(frame_wing.WING_HOLES) == 18
    with pytest.raises(ValueError, match='the frame'):
        frame_wing.geometry('tail')


def test_the_assembly_bolts_the_wing_on_through_four_washers():
    """The wing and its washers meshed apart from the frame and joined
    to it by Mate Surfaces (Brandon, 2026-10-20: the examples remade with
    the new joints): two rigid sets at each washer, one where its
    underside rests on the frame and one where it rests on the screw,
    eight in all; no node shared between the wing and the frame, no
    rigid link, one structure, a screw head's grams on the wing's top at
    each screw. The meshes need not line up: Align Meshes is gone."""
    from visualdynamics.core.fem import Model

    both = frame_wing.geometry('thick wing')
    names = set(both.group_name)
    assert {'frame', 'inserts', 'thick wing', 'washers', 'screw heads'} <= names
    ids = dict(zip(both.group_name, both.group_id.tolist()))
    frame = set(both.nodes_in([ids[n] for n in ('frame', 'inserts', 'screws')
                               if n in ids]))
    assert not frame & set(both.nodes_in([ids['washers']])), 'joined by ties alone'
    assert not frame & set(both.nodes_in([ids['thick wing']]))
    sets = both.mates
    assert len(sets) == 8 and all(m.describe() == 'rigid' for m in sets)
    assert sorted({tuple(both.group_name[list(both.group_id).index(g)] for g in m.groups)
                   for m in sets}) == [('frame', 'washers'), ('screws', 'washers')]
    inch = both.node_xyz / INCH
    for x, y in frame_wing.SCREWS:
        # each washer has a set on the frame and one on the screw, near it
        near = [m for m in sets if np.hypot(*(inch[both.node_index(list(m.nodes_a))]
                                               .mean(axis=0)[:2] - (x, y))) < 0.05]
        assert len(near) == 2, (x, y)
    model = Model.from_geometry(both)
    assert len(model.pieces()) == 1 and not model.rigid_links and model.ties
    assert len(model.masses) == 4
    assert sum(m.mass for m in model.masses) == pytest.approx(4 * frame_wing.SCREW_MASS)


def test_the_project_is_the_app_workflow(tmp_path):
    project = frame_wing.project(size=COARSE, wing_size=COARSE)
    assert list(project.keys()) == ['Frame', 'Thin Wing', 'Thick Wing',
                                    'Thin Wing on Frame', 'Thick Wing on Frame']
    project.save(tmp_path / 'frame-wing.vdyn')
    project = visualdynamics.load(tmp_path / 'frame-wing.vdyn')
    assert project['Frame'].group_properties[1].kind == 'solid'
    assert 'solve_modes' in [verb for verb, _ in project.verbs('Frame')]
    solved = project.solve_modes('Frame', num_modes=8)
    assert int(np.sum(project[solved].frequency == 0.0)) == 6


@pytest.mark.slow
def test_the_frame_lands_on_the_measured_frame():
    """Frame SN003's first nine modes: within 1.5 % on six of them and
    within 4 % on the three in-plane modes, soft (its holes ring-meshed,
    2026-10-11: 0.4, 0.0, 0.7, 0.0, -2.7, -3.6, 0.5, 0.5, -2.1 %; pulled
    round from the grid they read -3.1, -4.0 and -2.4 on those three,
    and the blocks of 2026-09-30 -2.5, -3.5 and -2.1, their rim bricks
    folded into darts). Mass within 2 % of the frames weighed."""
    model = frame_wing.build()
    assert model.total_mass == pytest.approx(0.6242, rel=0.02)
    shapes = model.eigensolution(maximum_frequency=1400.0)
    assert int(np.sum(shapes.frequency == 0.0)) == 6
    elastic = shapes.frequency[6:15]
    error = elastic / np.array(MEASURED_FRAME) - 1.0
    assert np.abs(error[[0, 1, 2, 3, 6, 7]]).max() < 0.015, error
    assert np.abs(error[[4, 5, 8]]).max() < 0.04, error


def _matched(found, measured):
    """Each measured mode against the nearest found one."""
    found = np.asarray(found)
    return np.array([found[np.argmin(np.abs(found - m))] for m in measured])


@pytest.mark.slow
def test_the_wings_land_on_their_measured_modes():
    """The thin wing within 1.5 % of every measured mode (its torsion
    modes were not fitted; they pass by), the thick wing within 3 %
    (measured 2026-09-30: -0.9 to +2.8 %; ring-meshed holes, 2026-10-11:
    -0.3 to +0.9 % thin, -1.3 to +2.7 % thick)."""
    thin = frame_wing.build('thin wing alone').eigensolution(maximum_frequency=1300.0)
    error = _matched(thin.frequency[6:], MEASURED_THIN) / np.array(MEASURED_THIN) - 1.0
    assert np.abs(error).max() < 0.015, error
    thick = frame_wing.build('thick wing alone').eigensolution(maximum_frequency=1600.0)
    error = thick.frequency[6:15] / np.array(MEASURED_THICK) - 1.0
    assert np.abs(error).max() < 0.03, error


#: the first nine elastic modes of the finite element models of the frame
#: with each wing shared on the wiki (Linderholt, 2024; "Linderholt_Files_v2"),
#: tuned there to test, Hz — frozen here, none of the files shipped
SHARED_THICK_ON_FRAME = [126.0, 233.4, 242.7, 312.4, 439.3, 454.3, 460.8, 645.1,
                         676.5]
SHARED_THIN_ON_FRAME = [63.8, 113.5, 224.4, 224.8, 268.6, 306.2, 373.0, 417.2,
                        497.2]


@pytest.mark.slow
@pytest.mark.parametrize('wing, shared', [('thick wing', SHARED_THICK_ON_FRAME),
                                          ('thin wing', SHARED_THIN_ON_FRAME)])
def test_the_assemblies_land_on_the_shared_models(wing, shared):
    """The frame with each wing, bolted by merged washers and steel screws
    threaded into the inserts, against the shared models of the same:
    every one of the first nine modes within 3 % (holes and washers
    ring-meshed, 2026-10-11: -1.1 to +2.9 % thick, -2.2 to +1.0 % thin;
    pulled round from the grid, -3.4 to +1.9 and -3.8 to +0.2). A washer
    merged onto the insert's smeared material read the first mode 7 %
    soft; the steel the hardware has there is what closed it."""
    shapes = frame_wing.build(wing).eigensolution(maximum_frequency=1000.0)
    assert int(np.sum(shapes.frequency == 0.0)) == 6, 'one structure'
    elastic = shapes.frequency[6:15]
    error = elastic / np.array(shared) - 1.0
    assert np.abs(error).max() < 0.03, np.round(100 * error, 1)


@pytest.mark.slow
def test_the_downloadable_project_opens_solved(tmp_path):
    """The downloads page's frame is cut by tools/cut_examples.py from
    `frame_wing.project(solved=True)`: five models, every one solved."""
    import importlib.util
    import pathlib

    path = (pathlib.Path(__file__).resolve().parents[1] / 'tools'
            / 'cut_examples.py')
    spec = importlib.util.spec_from_file_location('cut_examples', path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    source, projects, _sentence, _about = tool.SETS[
        'VisualDynamics-examples-frame-wing.zip']
    source(str(tmp_path))
    project = visualdynamics.Project.open(tmp_path / f'{projects[0]}.vdyn')
    assert sorted(project.keys()) == sorted(
        ['Frame', 'Thin Wing', 'Thick Wing', 'Thin Wing on Frame',
         'Thick Wing on Frame'] + [f'{n} Modes' for n in (
             'Frame', 'Thin Wing', 'Thick Wing', 'Thin Wing on Frame',
             'Thick Wing on Frame')])
    assert project['Frame Modes'].frequency.max() <= frame_wing.SOLVE_TO


def _page_tables(header):
    """Every table on the solid workflow page under this header, its
    rows in page order, each a list of cells."""
    import pathlib

    page = (pathlib.Path(__file__).resolve().parents[1] / 'docs' / 'guide'
            / 'workflows' / 'fem-solids.md').read_text(encoding='utf-8')
    rows = []
    for table in page.split(header)[1:]:
        lines = table.split('\n\n')[0].strip().splitlines()[1:]
        rows += [[cell.strip().replace('−', '-') for cell in line.strip('|').split('|')]
                 for line in lines]
    return rows


def _calls(project):
    """(model, verb, args, kwargs) of each journaled call, in order."""
    import ast

    out = []
    for line in project.journal:
        if line.startswith('project.') and '(' in line:
            call = ast.parse(line).body[0].value
            if call.args:
                args = [ast.literal_eval(a) for a in call.args]
                out.append((args[0], call.func.attr, args,
                            {k.arg: ast.literal_eval(k.value) for k in call.keywords}))
    return out


def _numbers(cell):
    import re

    return [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', cell)]


def test_the_page_is_the_demos_steps():
    """The solid workflow page tells a student every shape and every
    pattern to type (Brandon, 2026-10-11: students should be able to
    follow it and build the model themselves). Each row is held to the
    call the demonstration journals for the same step — its center, its
    size, its name; its counts and spacing — so the page cannot drift
    from the model the tests check against the measurements."""
    from visualdynamics.project import Project

    project = Project('steps')
    for label, choice in (('Frame', 'frame'), ('Thick Wing', 'thick wing alone'),
                          ('Assembly', 'thick wing')):
        frame_wing._model(project, label, choice, COARSE, frame_wing.WING_SIZE)
    calls = _calls(project)
    adds = [c for c in calls if c[1] in ('add_solid_block', 'add_solid_cylinder')]
    patterns = [c for c in calls if c[1] == 'pattern_solids']

    def of(model, rows):
        return [c for c in rows if c[0] == model]

    # the page's shapes: the frame and its inserts, the wing, then the
    # assembly's own two — the screw and the washer
    expected = (of('Frame', adds) + of('Thick Wing', adds)
                + [c for c in of('Assembly', adds)
                   if c[3].get('name') in ('screws', 'washers')])
    shapes = _page_tables('| Shape | Add | Center | Size | Name |')
    assert len(shapes) == len(expected) == 12
    for (shape, add, center, size, name), (_m, verb, args, kw) in zip(shapes, expected):
        block = verb == 'add_solid_block'
        assert (add == 'Block') == block, shape
        assert np.allclose(_numbers(center), args[1]), shape
        numbers = _numbers(size)
        if block:
            assert np.allclose(numbers[:3], args[2]), shape
            assert numbers[3:] == ([kw['corner']] if 'corner' in kw else []), shape
        else:
            assert np.allclose(numbers, args[2:4]), shape
        assert name == kw.get('name', ''), shape
    # the page's patterns: the frame's holes and window (the inserts are
    # patterned the same way, as the page says), the wing's, the screws'
    # and the washers'
    frame = [(c[2][2], c[2][3]) for c in of('Frame', patterns)]
    assert frame[3:] == frame[1:3], 'the inserts patterned as the holes are'
    # the assembly patterns the frame's five (the window's one, the
    # holes' two, the inserts' two), then the screw's one, the wing's
    # three and the washer's one — each a whole grid, the fewest steps
    # (Brandon, 2026-10-11)
    assembly = [(c[2][2], c[2][3]) for c in of('Assembly', patterns)]
    assert len(frame) == 5 and len(assembly) == 10
    expected = (frame[:3] + [(c[2][2], c[2][3]) for c in of('Thick Wing', patterns)]
                + assembly[5:6] + assembly[9:10])
    pages = _page_tables('| Pick | How many along X, Y, Z | Spacing along X, Y, Z |')
    assert len(pages) == len(expected) == 8
    for (pick, counts, spacing), (made, apart) in zip(pages, expected):
        assert _numbers(counts) == list(made), pick
        assert np.allclose(_numbers(spacing), apart, atol=5e-7), pick
