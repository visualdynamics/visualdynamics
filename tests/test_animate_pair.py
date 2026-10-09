"""Two mode shapes overlaid, from a script: `ShapeSet.animate_pair` and
`Project.animate_pair`, the window's picked MAC cell without the window.

The pairing — the second mode phase-aligned to the first, through the
projection when the two sets live on different geometries — the caption
and the opacities are one implementation each, shared by the window,
the report's overlay block and these calls (PLAN.md, "Saving an
animation").
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path, plate_geometry_and_shapes

import visualdynamics
from visualdynamics.core.shapes import ShapeSet
from visualdynamics.gui.movie import mp4_reading
from visualdynamics.viz import animate


@pytest.fixture(scope='module')
def plate():
    """The plate's dense geometry and truth shapes, a sign-flipped copy
    of them (the same modes, each the other way up), and a sparse test
    set: the truth at the ten test nodes, on the test geometry."""
    geometry, truth = plate_geometry_and_shapes()
    flipped = ShapeSet(truth.frequency, truth.damping, truth.coordinate,
                       -truth.shape_matrix)
    sparse = visualdynamics.import_file(fixture_path('plate',
                                                     'test_geometry.npz'))
    sparse.define_units('m')
    kept = [k for k, dof in enumerate(truth.coordinate)
            if dof not in set(sparse.missing_dofs(truth.coordinate))]
    test = ShapeSet(truth.frequency, truth.damping,
                    [truth.coordinate[k] for k in kept],
                    truth.shape_matrix[:, kept])
    return geometry, truth, flipped, sparse, test


def _captured(monkeypatch):
    """What `animate_pair` was asked to draw, without drawing it."""
    seen = {}

    def capture(first, second, **kwargs):
        seen.update(first=first, second=second, **kwargs)
        return 'drawn'

    monkeypatch.setattr(animate, 'animate_pair', capture)
    return seen


def test_the_second_mode_is_turned_to_swing_with_the_first(plate,
                                                          monkeypatch):
    geometry, truth, flipped, _sparse, _test = plate
    seen = _captured(monkeypatch)
    assert truth.animate_pair(geometry, 8, flipped, 8) == 'drawn'
    first, second = seen['first'][1], seen['second'][1]
    assert np.allclose(first.offsets(0.0), second.offsets(0.0)), \
        'the flipped copy is aligned back onto the original'


def test_the_caption_names_both_modes_and_their_sizes(plate, monkeypatch):
    geometry, truth, flipped, _sparse, _test = plate
    seen = _captured(monkeypatch)
    truth.animate_pair(geometry, 8, flipped, 8, names=('FEM', 'Copy'))
    lines = seen['caption'].split('\n')
    assert lines[0].startswith('FEM  mode 9') and 'Hz' in lines[0]
    assert lines[1].startswith('Copy  mode 9')
    assert lines[2] == 'Copy/FEM = 1.00'


def test_two_geometries_align_through_the_projection(plate, monkeypatch):
    """A sparse test set on its own ten nodes and the dense truth share
    no geometry; the alignment is measured through the projection and
    the dense set is drawn on its own mesh — or, `on_basis`, the
    projection on the test mesh."""
    geometry, truth, _flipped, sparse, test = plate
    seen = _captured(monkeypatch)
    flipped = ShapeSet(truth.frequency, truth.damping, truth.coordinate,
                       -truth.shape_matrix)
    test.animate_pair(sparse, 8, flipped, 8, geometry)
    assert seen['first'][0] is sparse and seen['second'][0] is geometry
    # aligned: the flipped dense set is turned back, so at a node both
    # meshes have it moves the way the test set does, not against it
    sparse_side, dense_side = seen['first'][1], seen['second'][1]
    node = sparse.node_id[sparse_side.rows[0]]
    at = list(dense_side.rows).index(list(geometry.node_id).index(node))
    assert np.dot(sparse_side.offsets(0.0)[0],
                  dense_side.offsets(0.0)[at]) > 0
    test.animate_pair(sparse, 8, flipped, 8, geometry, on_basis=True)
    assert seen['second'][0] is sparse


def test_a_set_off_its_geometry_is_refused(plate):
    geometry, truth, _flipped, _sparse, _test = plate
    far = ShapeSet(truth.frequency, truth.damping, ['9999X+'],
                   np.ones((truth.num_shapes, 1)))
    with pytest.raises(ValueError, match='no DOFs on its geometry'):
        truth.animate_pair(geometry, 0, far, 0)


def test_the_project_draws_the_basis_solid_whichever_is_named_first(
        plate, monkeypatch):
    geometry, truth, flipped, _sparse, _test = plate
    project = visualdynamics.Project('Pair')
    project.add('FEM Geometry', geometry)
    project.add('FEM', truth)
    copy_geometry, _truth = plate_geometry_and_shapes()
    project.add('Copy Geometry', copy_geometry)
    project.add('Copy', flipped)
    project.link('FEM Geometry', 'FEM')
    project.link('Copy Geometry', 'Copy')
    seen = _captured(monkeypatch)
    project.animate_pair('FEM', 8, 'Copy', 8)
    assert seen['alphas'] == (1.0, 0.25), 'no Basis: the first leads'
    project.set_role('Copy', 'Basis')
    project.animate_pair('FEM', 8, 'Copy', 8)
    assert seen['alphas'] == (0.25, 1.0), 'the Basis is looked at'
    assert seen['caption'].startswith('FEM  mode 9')
    assert seen['first'][0] is geometry
    assert seen['second'][0] is copy_geometry, 'each on its own group\'s'


def test_a_pair_saves_a_still_and_a_movie(h264, plate, tmp_path):
    geometry, truth, flipped, _sparse, _test = plate
    still = tmp_path / 'pair.png'
    truth.animate_pair(geometry, 8, flipped, 8, screenshot=str(still))
    assert still.stat().st_size > 2000
    movie = tmp_path / 'pair.mp4'
    assert truth.animate_pair(geometry, 8, flipped, 8,
                              movie=str(movie)) == str(movie)
    assert mp4_reading(movie) == (b'avc1', 180)


def test_sets_sharing_no_dofs_align_only_through_the_projection():
    """A test set and an FE model rarely share a DOF label; the
    projection — the model carried onto the test DOFs — is what the
    alignment is measured on, and it is applied to the model itself."""
    from visualdynamics.core.shapes import paired_mode

    up = np.array([[1.0, 2.0]])
    test = ShapeSet([10.0], [0.01], ['1X+', '2X+'], up)
    model = ShapeSet([10.0], [0.01], ['101X+', '102X+'], -up)
    projected = ShapeSet([10.0], [0.01], ['1X+', '2X+'], -up)
    assert np.allclose(paired_mode(test, 0, model, 0, projected), up[0])
    assert np.allclose(paired_mode(test, 0, model, 0), -up[0]), \
        'without it there is nothing shared to turn by'
