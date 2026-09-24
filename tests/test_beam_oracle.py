"""The beam finite element, against sdynpy's own, matrix for matrix.

`tests/test_fem.py` checks the element against closed forms — the
cantilever series in bending, extension and torsion, free-free beams,
rigid-body modes. This is the second opinion: a small free 3D frame
assembled by both tools, frozen by `generate_beam_oracle.py` in the
private generators repository and read here as numbers, never imported
(`AGENTS.md` hard rule 1).

Six members along X, Y, Z and two skew diagonals, two materials, a
rectangle and a tube, and an orientation vector per member — two of
them not perpendicular to their member, so the orthogonalization is
compared too. The whole 30x30 mass and stiffness matrices must agree,
entry by entry in the same degree-of-freedom order.

Two conventions are mapped by the generator rather than compared, and
its docstring has both: which bending stiffness sdynpy calls `ei1`, and
the section numbers, which are handed to sdynpy rather than derived by
it, because its rectangle helper takes the shear modulus with the wrong
sign on Poisson's ratio and the polar moment for the torsion constant.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

from visualdynamics.core.fem import Material, Model, Section

ORACLE = fixture_path('sdynpy_oracle', 'beam.npz')


@pytest.fixture(scope='module')
def oracle():
    with np.load(ORACLE) as frozen:
        return {key: frozen[key] for key in frozen.files}


def _model(oracle):
    model = Model()
    for i, xyz in enumerate(oracle['nodes']):
        model.add_node(i + 1, *xyz)
    materials = [Material(f'material {i}', e, rho, nu)
                 for i, (e, rho, nu) in enumerate(oracle['materials'])]
    sections = [Section(f'section {i}', *numbers)
                for i, numbers in enumerate(oracle['sections'])]
    for (a, b), orientation, mat, sec in zip(
            oracle['connectivity'], oracle['orientation'],
            oracle['material_of'], oracle['section_of']):
        model.add_beam(int(a) + 1, int(b) + 1, materials[mat], sections[sec],
                       orientation=tuple(orientation))
    return model


@pytest.mark.parametrize('which', ['stiffness', 'mass'])
def test_the_assembled_matrices_agree_with_sdynpys(oracle, which):
    mass, stiffness = _model(oracle).matrices()
    ours = {'mass': mass, 'stiffness': stiffness}[which]
    theirs = oracle[which]
    assert ours.shape == theirs.shape
    worst = np.abs(ours - theirs).max() / np.abs(theirs).max()
    assert worst < 1e-12, f'{which} departs from sdynpy by {worst:.2e} of peak'


def test_so_do_the_natural_frequencies(oracle):
    """What a user reads off the model, and a check that no entry small
    enough to hide under the peak-relative tolerance above matters."""
    from scipy.linalg import eigh
    mass, stiffness = _model(oracle).matrices()
    ours = eigh(stiffness, mass, eigvals_only=True)[6:]
    theirs = eigh(oracle['stiffness'], oracle['mass'], eigvals_only=True)[6:]
    assert np.allclose(ours, theirs, rtol=1e-9)


def test_the_frozen_frame_exercises_what_it_claims_to(oracle):
    """So the agreement means something: members in every direction,
    orientation vectors not perpendicular to their member, a section
    that bends differently about its two axes, and two materials."""
    nodes, conn = oracle['nodes'], oracle['connectivity']
    axes = nodes[conn[:, 1]] - nodes[conn[:, 0]]
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    assert (np.abs(axes) > 0.99).any(axis=0).all(), 'a member along each axis'
    assert (np.abs(axes).max(axis=1) < 0.99).sum() >= 2, 'skew members'
    along = np.abs(np.einsum('ij,ij->i', axes, oracle['orientation']))
    assert (along > 0.1).sum() >= 2, 'orientation vectors to orthogonalize'
    _area, iy, iz, j = oracle['sections'].T
    assert (iy != iz).any() and (j < iy + iz).any(), 'a non-round section'
    assert len(set(oracle['material_of'])) == 2
