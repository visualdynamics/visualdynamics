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

from visualdynamics.core.fem import SPARSE_ABOVE, Model


def _compare(model, **kwargs):
    dense = model.eigensolution(solver='dense', **kwargs)
    sparse = model.eigensolution(solver='sparse', **kwargs)
    assert len(dense.frequency) == len(sparse.frequency)
    rigid = dense.frequency == 0.0
    assert np.array_equal(rigid, sparse.frequency == 0.0), 'the same rigid modes'
    assert np.allclose(sparse.frequency, dense.frequency, rtol=1e-8, atol=1e-6)
    mass, _stiffness = model.matrices()
    for k in np.flatnonzero(~rigid):
        a, b = dense.shape_matrix[k], sparse.shape_matrix[k]
        # distinct elastic modes: the same shape, up to its sign
        if min(abs(dense.frequency[k] - dense.frequency[j])
               for j in range(len(rigid)) if j != k) > 1e-6 * dense.frequency[k]:
            mac = (a @ mass @ b) ** 2 / ((a @ mass @ a) * (b @ mass @ b))
            assert mac > 1 - 1e-8, (k, mac)
        assert b @ mass @ b == pytest.approx(1.0, rel=1e-9), 'mass-normalized'
    return dense, sparse


def test_the_demo_plate_solves_the_same_either_way():
    from visualdynamics.demo import plate

    model = plate.build()
    dense, _sparse = _compare(model, num_modes=16)
    assert int(np.sum(dense.frequency == 0.0)) == 6


def test_rigid_links_and_a_ground_solve_the_same_either_way():
    model = Model()
    lower = _strip(model, 1, nx=8, ny=3)
    upper = _strip(model, 101, nx=8, ny=3, z=0.02)
    for key in ((0, 0), (8, 0), (0, 3), (8, 3), (4, 1)):
        model.add_rigid_link(lower[key], upper[key])
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
