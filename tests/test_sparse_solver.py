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

import warnings

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


def _solid_bar():
    """A free-free bar of bricks, a wedge-free solid model."""
    from test_fem_solids import _bar_of_bricks

    return _bar_of_bricks(along=12, deep=2)


def _springs_model():
    """Two strips joined by springs, from the springs tests."""
    model = Model()
    lower = _strip(model, 1, nx=6, ny=3)
    upper = _strip(model, 101, nx=6, ny=3, z=0.03)
    for key in ((0, 0), (6, 0), (0, 3), (6, 3)):
        model.add_spring(f'{lower[key]}Z', f'{upper[key]}Z', 2e5)
    return model


def _demo_plate():
    from visualdynamics.demo import plate

    return plate.build()


@pytest.mark.parametrize('build, kwargs', [
    (_demo_plate, {'num_modes': 16}),
    (lambda: _rigid_link_model()[0], {'num_modes': 14}),
    (lambda: _rigid_link_model()[0], {'num_modes': 10, 'fixed': ['1', '4']}),
    (_solid_bar, {'num_modes': 12}),
    (_springs_model, {'num_modes': 20}),
    (lambda: _rigid_link_model()[0], {'num_modes': 1}),
], ids=['free plate', 'rigid links', 'grounded', 'bricks', 'springs', 'one mode'])
def test_the_block_lanczos_and_arpack_give_one_answer(build, kwargs, monkeypatch):
    """The sparse solve runs on the block Lanczos written here
    (`block_lanczos_modes`), ARPACK behind it: on every kind of model,
    free-free ones with their six rigid modes included, the two give
    every frequency the same to 1e-9, the rigid ones exactly zero on
    both, and the block solver is the one that ran."""
    model = build()
    called = []
    real = fem.block_lanczos_modes
    monkeypatch.setattr(fem, 'block_lanczos_modes',
                        lambda *a, **k: called.append(1) or real(*a, **k))
    found = {}
    for solver in ('block', 'arpack'):
        monkeypatch.setattr(fem, 'SPARSE_EIGENSOLVER', solver)
        # and the block solver converged on its own: a fallback to
        # ARPACK would give the right answer for the wrong reason
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            found[solver] = model.eigensolution(solver='sparse', **kwargs).frequency
    assert len(called) >= 1 and len(found['block']) == len(found['arpack'])
    rigid = found['arpack'] == 0.0
    assert np.array_equal(rigid, found['block'] == 0.0)
    assert np.allclose(found['block'][~rigid], found['arpack'][~rigid],
                       rtol=1e-9, atol=0.0), found['block'] / found['arpack'] - 1


def _tiny_beam():
    """A free-free beam of two elements: 18 degrees of freedom, fewer
    than two blocks, so the second block has room for six directions
    of its twelve and the rest are dependent."""
    from test_fem import straight_beam

    return straight_beam(1.0, fem.Section.rod('rod', 0.01), 2)


@pytest.mark.parametrize('build, backend', [
    (_demo_plate, ''), (_demo_plate, 'superlu'), (lambda: _rigid_link_model()[0], ''),
], ids=['free plate', 'free plate, SuperLU', 'rigid links'])
def test_the_block_lanczos_claims_each_pair_what_a_solve_measures(
        build, backend, monkeypatch):
    """Each pair's claimed residual is the one a solve with the same
    factor measures, to 2 %, down to round-off: eps per entry of a
    vector scaled to the operator's largest eigenvalue, summed over the
    entries, sqrt(n) eps theta_max / theta. The solves are given M
    times the basis, taken afresh; the image the orthonormalization
    carried as combinations was 6e-10 off it in the first applied
    block, the shift-inverse magnified that rigid error by
    theta_rigid / theta, and the free plate's pairs measured 0.3 to 6.8
    times their claims, a spread that moved with Accelerate's thread
    count and failed a full gate a third of the time (SuperLU, the same
    every run, 1.49). LU's solves on the rigid-link model measure 0.6
    to 1.9 times the claim pair by pair with or without that, so the
    default backend's cases need Accelerate's Cholesky and are skipped
    where it is not (the public runner's Linux)."""
    from visualdynamics.core.sparse_solve import BACKEND_VARIABLE, factor_symmetric

    monkeypatch.setenv(BACKEND_VARIABLE, backend)
    stiffness, mass, _t, _s, sigma, _k = build().scaled_system()
    factor = factor_symmetric((stiffness - sigma * mass).tocsc())
    if not backend and getattr(factor, 'kind', '') != 'Cholesky':
        pytest.skip('needs Accelerate\'s Cholesky; the SuperLU case is its own')
    eigenvalues, vectors, claimed = fem.block_lanczos_modes(factor, mass, sigma, 12)
    theta = 1.0 / (eigenvalues - sigma)
    residual = factor.solve(mass @ vectors) - vectors * theta
    measured = np.sqrt(np.sum(residual * (mass @ residual), axis=0)) / theta
    # four times the round-off: the solver branch's pairs reach half
    # of it once converged to 2e-8, where claim and measurement still
    # agree to 1e-3 (measured 2026-10-20); the defect was 1,300 to
    # 4,300 times it
    floor = 4.0 * np.sqrt(mass.shape[0]) * np.finfo(float).eps * theta.max() / theta
    assert np.all(np.abs(measured - claimed) <= 0.02 * claimed + floor), \
        (measured / claimed, claimed)


@pytest.mark.parametrize('backend', ['', 'superlu'])
@pytest.mark.parametrize('build', [lambda: _rigid_link_model()[0], _demo_plate, _tiny_beam],
                         ids=['rigid links', 'free plate', 'tiny beam'])
def test_the_block_lanczos_starts_the_same_twice_and_keeps_its_word(
        build, backend, monkeypatch):
    """Its random start is seeded, so a model solved twice gives one
    answer to the bit (ARPACK's start is random and did not); and the
    pairs it hands back meet the residual it claims for them: measured
    whole with one more solve, |A x - theta x|_M / theta, the worst is
    within a factor of four of the worst claimed, either way (1.1 the
    most seen; under SuperLU the claim is mostly the computed
    operator's asymmetry, 7e-4 on the rigid-link model, and the last
    block's share alone said 9e-8: the public runner's failure of
    2026-10-19). The free plate's first block has a direction the Gram
    matrix cannot see, and a claim that dropped it overstated five
    times. Run through both factorizations where a Mac has both.
    `test_the_block_lanczos_claims_each_pair_what_a_solve_measures`
    holds the claim to the measurement pair by pair."""
    from visualdynamics.core.sparse_solve import BACKEND_VARIABLE, factor_symmetric

    monkeypatch.setenv(BACKEND_VARIABLE, backend)
    model = build()
    stiffness, mass, _t, _s, sigma, _k = model.scaled_system()
    factor = factor_symmetric((stiffness - sigma * mass).tocsc())
    first = fem.block_lanczos_modes(factor, mass, sigma, 12)
    second = fem.block_lanczos_modes(factor, mass, sigma, 12)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    eigenvalues, vectors, claimed = first
    theta = 1.0 / (eigenvalues - sigma)
    # M-orthonormal to the basis's own orthogonality, about 2e-10
    assert vectors.T @ (mass @ vectors) == pytest.approx(np.eye(12), abs=1e-8)
    residual = factor.solve(mass @ vectors) - vectors * theta
    measured = np.sqrt(np.sum(residual * (mass @ residual), axis=0)) / theta
    # within a factor of four either way, above the measurement's own
    # floor (1e-9 on the tiny beam, where both are that small)
    assert measured.max() <= 4.0 * claimed.max() + 1e-8, (measured, claimed)
    assert claimed.max() <= 4.0 * measured.max() + 1e-8, (measured, claimed)
    assert claimed.max() < 1e-3, 'and the claim is a converged one'


def test_a_basis_that_runs_out_goes_to_arpack_with_a_warning(monkeypatch):
    """A model the block basis cannot converge within its limit is not
    left unsolved: `_sparse_modes` warns and solves it by ARPACK, and
    the answer is the one ARPACK gives on its own."""
    model, _lower = _rigid_link_model()
    monkeypatch.setattr(fem, 'SPARSE_EIGENSOLVER', 'arpack')
    expected = model.eigensolution(num_modes=14, solver='sparse').frequency
    monkeypatch.setattr(fem, 'SPARSE_EIGENSOLVER', 'block')
    # two blocks of basis for 22 modes: the Ritz values cannot have
    # settled, and the solver must say so rather than hand them over
    monkeypatch.setattr(fem, 'LANCZOS_BASIS', 2)
    stiffness, mass, _t, _s, sigma, _k = model.scaled_system()
    from visualdynamics.core.sparse_solve import factor_symmetric

    factor = factor_symmetric((stiffness - sigma * mass).tocsc())
    with pytest.raises(fem.NotConverged, match='residual'):
        fem.block_lanczos_modes(factor, mass, sigma, 22)
    with pytest.warns(UserWarning, match='did not converge.*ARPACK'):
        found = model.eigensolution(num_modes=14, solver='sparse').frequency
    assert np.allclose(found, expected, rtol=1e-9, atol=0.0)


def test_the_m_orthonormalization_survives_an_ill_conditioned_block():
    """A block of columns that share a direction and differ by parts in
    a million has a Gram matrix of condition 1e12, and one pass through
    its eigenvectors leaves Q^T M Q off the identity by 4e-6 (measured);
    two passes leave it at round-off, Q R is still the block, and a
    dependent column is never divided by nothing."""
    from scipy.sparse import diags

    rng = np.random.default_rng(5)
    n = 200
    mass = diags(rng.uniform(0.5, 2.0, n)).tocsc()
    shared = rng.standard_normal(n)
    block = shared[:, None] + (rng.standard_normal((n, 6))
                               * np.array([1.0, 1e-3, 1e-6, 1e-2, 1e-4, 1e-5]))
    q, mq, r, keep = fem._m_orthonormal(block, mass @ block)
    assert keep.all()
    assert np.abs(q.T @ (mass @ q) - np.eye(6)).max() < 1e-11
    assert np.allclose(q @ r, block, rtol=0, atol=1e-12 * np.abs(block).max())
    assert np.allclose(mq, mass @ q)
    q1, _mq1, _r1, _keep1 = fem._gram_pass(block, mass @ block)
    assert np.abs(q1.T @ (mass @ q1) - np.eye(6)).max() > 1e-7, 'one pass does not'
    # an empty column is dropped, a zero column of Q and row of R; a
    # duplicate column's null direction may come back as real content
    # (the Gram eigenvector's own error lets the 1e-6 columns through),
    # which is in the block's span and harmless: finite, orthonormal
    # with the rest, and Q R is still the block
    for name, column in (('empty', np.zeros(n)), ('duplicate', block[:, 0] * 2.0)):
        dependent = block.copy()
        dependent[:, 5] = column
        q, mq, r, keep = fem._m_orthonormal(dependent, mass @ dependent)
        assert np.isfinite(q).all(), name
        if name == 'empty':
            assert keep.sum() == 5 and not q[:, ~keep].any() and not r[~keep].any()
        survivors = q[:, keep]
        assert np.abs(survivors.T @ (mass @ survivors) - np.eye(keep.sum())).max() < 1e-11, name
        assert np.allclose(q @ r, dependent, rtol=0, atol=1e-12 * np.abs(dependent).max()), name


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
