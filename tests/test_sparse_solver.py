"""The sparse solver: the same modes as the dense one, from matrices
stored as their nonzeros (Brandon, 2026-09-26 — a plate model of a
small real structure outgrew the dense ceiling: the BARC at a quarter
inch, ~1,500 nodes, is ~13 GB dense).

Held against the dense solver on the models it can still solve — the
demonstration plate free-free, plates joined by rigid links, a grounded
model — frequency by frequency and shape by shape, and on the parts only
the sparse one has: the search up to a frequency, and the refusals.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_rigid_links import _strip

from visualdynamics.core import fem
from visualdynamics.core.fem import SPARSE_ABOVE, Model, polish_modes

#: the relative residual a polished mode is allowed, for the shape
#: comparison below: an order above the floor the conditioning sets on
#: this machine (3e-7 on the rigid-link model) and above what one
#: runner's build reached (2026-09-30, about 5e-6 by the MAC it gave)
RESIDUAL_ALLOWANCE = 1e-5


def _compare(model, **kwargs):
    dense = model.eigensolution(solver='dense', **kwargs)
    sparse = model.eigensolution(solver='sparse', **kwargs)
    assert len(dense.frequency) == len(sparse.frequency)
    rigid = dense.frequency == 0.0
    assert np.array_equal(rigid, sparse.frequency == 0.0), 'the same rigid modes'
    assert np.allclose(sparse.frequency, dense.frequency, rtol=1e-8, atol=1e-6)
    mass, _stiffness = model.matrices()
    eigenvalue = (2.0 * np.pi * dense.frequency) ** 2
    for k in np.flatnonzero(~rigid):
        a, b = dense.shape_matrix[k], sparse.shape_matrix[k]
        # distinct elastic modes: the same shape, up to its sign — to the
        # accuracy a residual buys against the nearest neighbor, since a
        # residual r leaves a mode mixed with its neighbor by about
        # r * lambda / (lambda_neighbor - lambda), and 1 - MAC is that
        # squared. A fixed 1e-8 was asked before and failed on a runner
        # for a pair 1.45 % apart (2026-09-30).
        gap = min(abs(eigenvalue[k] - eigenvalue[j])
                  for j in range(len(rigid)) if j != k)
        if gap > 1e-6 * eigenvalue[k]:
            mac = (a @ mass @ b) ** 2 / ((a @ mass @ a) * (b @ mass @ b))
            allowed = (RESIDUAL_ALLOWANCE * eigenvalue[k] / gap) ** 2
            assert mac > 1 - max(allowed, 1e-12), (k, mac, allowed)
        assert b @ mass @ b == pytest.approx(1.0, rel=1e-9), 'mass-normalized'
    return dense, sparse


def _rigid_link_model():
    model = Model()
    lower = _strip(model, 1, nx=8, ny=3)
    upper = _strip(model, 101, nx=8, ny=3, z=0.02)
    for key in ((0, 0), (8, 0), (0, 3), (8, 3), (4, 1)):
        model.add_rigid_link(lower[key], upper[key])
    return model, lower


def test_the_polish_cleans_a_corrupted_start(monkeypatch):
    """What one runner's build did once — hand the polish a start it
    could not clean in the three steps that used to be fixed — done on
    purpose: Lanczos' vectors corrupted by a part in a thousand. The
    residual-driven polish reaches the conditioning's floor; held to
    three steps it stops three orders above it, which is the failure."""
    from scipy.sparse.linalg import eigsh

    model, _lower = _rigid_link_model()
    stiffness, mass, _t, _s, sigma, _k = model.scaled_system()
    _values, vectors = eigsh(stiffness, k=22, M=mass, sigma=sigma, which='LM')
    rng = np.random.default_rng(0)
    corrupted = vectors + 1e-3 * (rng.standard_normal(vectors.shape)
                                  * np.linalg.norm(vectors, axis=0))

    _e, _v, residuals = polish_modes(stiffness, mass, sigma, corrupted.copy(), 14)
    assert residuals[:14].max() < 1e-6, 'reaches the floor'

    monkeypatch.setattr(fem, 'POLISH_STEPS', 3)
    _e, _v, residuals = polish_modes(stiffness, mass, sigma, corrupted.copy(), 14)
    assert residuals[:14].max() > 1e-5, 'three fixed steps do not'


def test_the_polish_stops_at_the_floor():
    """A clean start reaches the floor in a few steps; the polish
    notices a step that no longer halves the worst residual and stops,
    rather than spending every step it is allowed at the floor.

    The start is seeded. eigsh's own start is random, and over 200 of
    them the polish took 3 to 6 steps (56, 115, 27 and 2 of them); the
    bound here was "under 6", which the 2 in 200 broke on a public CI
    run of 2026-10-05. The claim is the budget, so that is the bound."""
    from scipy.sparse.linalg import eigsh

    model, _lower = _rigid_link_model()
    stiffness, mass, _t, _s, sigma, _k = model.scaled_system()
    start = np.random.default_rng(0).standard_normal(stiffness.shape[0])
    _values, vectors = eigsh(stiffness, k=22, M=mass, sigma=sigma,
                             which='LM', v0=start)
    solves = []
    real = fem.polish_modes.__globals__['np'].linalg.norm

    class Counting:
        """Count the polish's steps by its residual evaluations."""

        def __call__(self, *args, **kwargs):
            solves.append(1)
            return real(*args, **kwargs)

    np_linalg = fem.np.linalg
    original = np_linalg.norm
    np_linalg.norm = Counting()
    try:
        polish_modes(stiffness, mass, sigma, vectors, 14)
    finally:
        np_linalg.norm = original
    # two norms per step, and the polish stopped short of its budget
    assert len(solves) // 2 < fem.POLISH_STEPS, len(solves)


def test_the_demo_plate_solves_the_same_either_way():
    from visualdynamics.demo import plate

    model = plate.build()
    dense, _sparse = _compare(model, num_modes=16)
    assert int(np.sum(dense.frequency == 0.0)) == 6


def test_rigid_links_and_a_ground_solve_the_same_either_way():
    model, lower = _rigid_link_model()
    _compare(model, num_modes=14)
    _compare(model, num_modes=10, fixed=[str(lower[(0, 0)]), str(lower[(0, 3)])])


def test_the_sparse_search_reaches_a_frequency():
    """Asked for every mode up to a frequency, the sparse solver widens
    its search until it has passed it — the same set the dense one keeps."""
    from visualdynamics.demo import plate

    model = plate.build()
    top = float(model.eigensolution(solver='dense', num_modes=60).frequency[-1])
    _dense, sparse = _compare(model, maximum_frequency=top * 0.99)
    assert len(sparse.frequency) > 24, 'more than the first search held'


def test_a_large_model_is_solved_sparse_and_must_say_how_many():
    model = Model()
    _strip(model, 1, nx=30, ny=18, length=0.6, width=0.36)
    assert model.num_dof > SPARSE_ABOVE
    with pytest.raises(ValueError, match='say how many'):
        model.eigensolution()
    shapes = model.eigensolution(num_modes=10)
    assert int(np.sum(shapes.frequency == 0.0)) == 6
    with pytest.raises(ValueError, match='is not a solver'):
        model.eigensolution(num_modes=4, solver='fast')


def test_the_sparse_matrices_are_the_dense_ones():
    model = Model()
    lower = _strip(model, 1)
    upper = _strip(model, 101, z=0.02)
    model.add_rigid_link(lower[(0, 0)], upper[(0, 0)])
    model.add_mass(lower[(3, 1)], 0.2, (1e-4, 2e-4, 3e-4))
    dense_m, dense_k = model.matrices()
    sparse_m, sparse_k = model.sparse_matrices()
    # the same entries, summed in a different order: equal to rounding
    assert np.allclose(sparse_m.toarray(), dense_m, rtol=0, atol=1e-15 * abs(dense_m).max())
    assert np.allclose(sparse_k.toarray(), dense_k, rtol=0, atol=1e-15 * abs(dense_k).max())
    transform = model.constraint_transform()
    assert np.array_equal(model.constraint_transform(sparse=True).toarray(),
                          transform)
