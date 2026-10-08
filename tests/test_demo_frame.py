"""The four-unit frame, built from blocks of bricks (2026-09-30;
`visualdynamics.demo.frame`, from
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
from visualdynamics.demo import frame

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
    geometry = frame.geometry()            # meshing is fast; solving is not
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
    holes = np.array(frame.INSERTS)
    nearest = np.argmin(np.linalg.norm(centers[:, None, :2] - holes[None], axis=2),
                        axis=1)
    assert len(set(nearest.tolist())) == len(frame.INSERTS) == 41
    assert np.linalg.norm(centers[:, :2] - holes[nearest], axis=1).max() < frame.HOLE_RADIUS
    assert centers[:, 2].min() > -frame.INSERT_DEPTH - frame.SIZE, 'the top only'
    assert centers[:, 2].max() < 0.0
    for block in geometry.group_id:
        assert geometry.group_properties[int(block)].kind == 'solid'
    assert geometry.view == frame.VIEW


def test_a_wing_alone_is_a_plate_with_eighteen_holes():
    thin = frame.geometry('thin wing alone', wing_size=COARSE)
    inch = thin.node_xyz / INCH
    assert np.allclose(inch.min(axis=0), [frame.WING_X, -11, frame.WING_GAP], atol=1e-9)
    assert inch[:, 2].max() == pytest.approx(frame.WING_GAP + frame.WING_THICKNESS['thin wing'])
    assert list(thin.group_name) == ['thin wing']
    # every hole is a void: no brick center within its radius
    centers = np.array([inch[thin.node_index(c)].mean(axis=0) for c in thin.elem_conn])
    for hx, hy in frame.WING_HOLES:
        assert np.hypot(centers[:, 0] - hx, centers[:, 1] - hy).min() > frame.WING_HOLE_RADIUS
    assert len(frame.WING_HOLES) == 18
    with pytest.raises(ValueError, match='the frame'):
        frame.geometry('tail')


def test_the_assembly_ties_the_wing_through_four_washers():
    both = frame.geometry('thick wing', size=COARSE, wing_size=COARSE)
    assert sorted(both.group_name) == ['frame', 'inserts', 'screws', 'thick wing']
    links = both.elements_in('screws')
    assert len(links) >= 4 * 4, 'each washer covers bricks whose nodes are tied'
    for screw in frame.SCREWS:
        assert frame.washer_patch(both, 'thick wing', screw), 'bricks under it'
    screws = both.group_properties[int(both.group_id[list(both.group_name).index('screws')])]
    assert screws.material.is_rigid
    model = frame.build('thick wing', size=COARSE, wing_size=COARSE)
    assert len(model.masses) == 4 and all(m.name == 'screw' for m in model.masses)
    assert model.total_mass == pytest.approx(
        frame.build(size=COARSE).total_mass
        + frame.build('thick wing alone', wing_size=COARSE).total_mass
        + 4 * frame.SCREW_MASS)


def test_the_project_is_the_app_workflow(tmp_path):
    project = frame.project(size=COARSE, wing_size=COARSE)
    assert list(project.keys()) == ['Frame', 'Thin Wing', 'Thick Wing',
                                    'Thin Wing on Frame', 'Thick Wing on Frame']
    project.save(tmp_path / 'frame.vdyn')
    project = visualdynamics.load(tmp_path / 'frame.vdyn')
    assert project['Frame'].group_properties[1].kind == 'solid'
    assert 'solve_modes' in [verb for verb, _ in project.verbs('Frame')]
    solved = project.solve_modes('Frame', num_modes=8)
    assert int(np.sum(project[solved].frequency == 0.0)) == 6


@pytest.mark.slow
def test_the_frame_lands_on_the_measured_frame():
    """Frame SN003's first nine modes: within 1.5 % on six of them and
    within 4 % on the three in-plane modes the stair-stepped fillets
    leave soft (measured 2026-09-30: 0.2, 1.0, 0.6, 0.7, -2.5, -3.5,
    0.5, 0.9, -2.1 %). Mass within 2 % of the frames weighed."""
    model = frame.build()
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
    (measured 2026-09-30: -0.9 to +2.8 %)."""
    thin = frame.build('thin wing alone').eigensolution(maximum_frequency=1300.0)
    error = _matched(thin.frequency[6:], MEASURED_THIN) / np.array(MEASURED_THIN) - 1.0
    assert np.abs(error).max() < 0.015, error
    thick = frame.build('thick wing alone').eigensolution(maximum_frequency=1600.0)
    error = thick.frequency[6:15] / np.array(MEASURED_THICK) - 1.0
    assert np.abs(error).max() < 0.03, error


@pytest.mark.slow
def test_the_downloadable_project_opens_solved(tmp_path):
    """The downloads page's frame is cut by tools/cut_examples.py from
    `frame.project(solved=True)`: five models, every one solved."""
    import importlib.util
    import pathlib

    path = (pathlib.Path(__file__).resolve().parents[1] / 'tools'
            / 'cut_examples.py')
    spec = importlib.util.spec_from_file_location('cut_examples', path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    source, projects, _sentence, _about = tool.SETS[
        'VisualDynamics-examples-frame.zip']
    source(str(tmp_path))
    project = visualdynamics.Project.open(tmp_path / f'{projects[0]}.vdyn')
    assert sorted(project.keys()) == sorted(
        ['Frame', 'Thin Wing', 'Thick Wing', 'Thin Wing on Frame',
         'Thick Wing on Frame'] + [f'{n} Modes' for n in (
             'Frame', 'Thin Wing', 'Thick Wing', 'Thin Wing on Frame',
             'Thick Wing on Frame')])
    assert project['Frame Modes'].frequency.max() <= frame.SOLVE_TO
