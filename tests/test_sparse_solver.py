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
    # and a converged one: under SuperLU the elastic pairs stop at the
    # LU's asymmetry, which no more basis lowers and the polish then
    # cleans (the solve matches the dense one to 2e-11): 6.9e-4 at
    # worst here, 1.14e-3 on the public runner (0.1.0a63's sync), the
    # rigid modes 4e-9 on both. A tenth of a percent pinned that floor
    # by the machine's rounding; an order above it still tells a claim
    # that stopped short
    assert claimed.max() < 1e-2, 'and the claim is a converged one'


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


def test_the_m_orthonormalization_keeps_a_share_of_1e_minus_10():
    """`_m_orthonormal` is Gram-Schmidt column by column, twice, with
    the product M Q taken afresh after each pass: on a block of columns
    that share a direction and differ by parts in 1e-10 (what one solve
    with the shifted factor leaves of the elastic modes of bricks tied
    across a seam), Q^T M Q is the identity to round-off and Q R is
    still the block; a column that is a multiple of another is dropped,
    a zero column of Q and row of R; and a single pass, or the product
    carried along as combinations, is not enough (measured: 5e-3)."""
    from scipy.sparse import diags

    rng = np.random.default_rng(5)
    n = 200
    mass = diags(rng.uniform(0.5, 2.0, n)).tocsc()
    shared = rng.standard_normal(n)
    block = shared[:, None] + (rng.standard_normal((n, 6))
                               * np.array([1.0, 1e-3, 1e-10, 1e-2, 1e-6, 1e-10]))
    q, mq, r, keep, share = fem._m_orthonormal(block, mass)
    assert keep.all()
    assert np.abs(q.T @ (mass @ q) - np.eye(6)).max() < 1e-13
    assert np.allclose(mq, mass @ q, rtol=0, atol=1e-15 * np.abs(mq).max())
    assert np.allclose(q @ r, block, rtol=0, atol=1e-13 * np.abs(block).max())
    assert np.allclose(r, np.triu(r)), 'R is a triangle'
    assert share.min() < 1e-9 and share.max() <= 1.0 + 1e-12, 'the shares read the 1e-10 columns'
    dependent = block.copy()
    dependent[:, 5] = dependent[:, 0] * 2.0
    q, mq, r, keep, _share = fem._m_orthonormal(dependent, mass)
    assert keep.sum() == 5 and not keep[5] and not q[:, 5].any() and not r[5].any()
    survivors = q[:, keep]
    assert np.abs(survivors.T @ (mass @ survivors) - np.eye(5)).max() < 1e-13
    assert np.allclose(q @ r, dependent, rtol=0, atol=1e-13 * np.abs(dependent).max())


def _brick_bar(model, first, nx, ny, nz, dx, z0):
    """A bar of bricks, `nx` by `ny` by `nz` of side `dx`, its bottom
    at `z0`; the node ids by (i, j, k)."""
    from test_fem import ALUMINUM

    ids = {}
    nid = first
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                ids[i, j, k] = model.add_node(nid, i * dx, j * dx, z0 + k * dx)
                nid += 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                model.add_solid([ids[i, j, k], ids[i + 1, j, k], ids[i + 1, j + 1, k],
                                 ids[i, j + 1, k], ids[i, j, k + 1], ids[i + 1, j, k + 1],
                                 ids[i + 1, j + 1, k + 1], ids[i, j + 1, k + 1]], ALUMINUM)
    return ids


def _tied_bars(ratio=2, slanted=False):
    """Two bars of bricks stacked, the upper `ratio` times finer, each
    upper bottom node tied to the nearest lower top node; `slanted`
    ties only the fine nodes a half step off the coarse ones, one to a
    lead, so every tie is one follower on a diagonal arm."""
    model = Model()
    nx, ny, nz, dx = 6, 2, 2, 0.01
    lower = _brick_bar(model, 1, nx, ny, nz, dx, 0.0)
    upper = _brick_bar(model, 10001, nx * ratio, ny * ratio, nz, dx / ratio, nz * dx)
    tops = {(i, j): lower[i, j, nz] for i in range(nx + 1) for j in range(ny + 1)}
    if slanted:
        # the twelve fine nodes diagonally off a coarse one, (2a+1, 2b+1)
        # one layer up, each tied to one of its four coarse corners, no
        # corner twice, so every body is one follower on a diagonal arm.
        # A rigid link whose lead only solids touch holds the follower
        # only along the arm (the lead's rotation moves it freely
        # across), so the corners cycle through all four diagonals, or
        # the upper bar would slide as a mechanism
        corner = {0: [(0, 0), (1, 1), (1, 0), (0, 1)],
                  1: [(1, 1), (0, 0), (0, 1), (1, 0)]}
        for b in range(ny):
            for a in range(nx):
                da, db = corner[b][a % 4]
                model.add_rigid_link(tops[a + da, b + db],
                                     upper[2 * a + 1, 2 * b + 1, 1])
        return model
    for i in range(nx * ratio + 1):
        for j in range(ny * ratio + 1):
            lead = tops[round(i / ratio), round(j / ratio)]
            model.add_rigid_link(lead, upper[i, j, 0])
    return model


@pytest.mark.parametrize('ratio', [1, 2], ids=['coincident', 'nearest node'])
def test_bricks_tied_across_a_seam_solve_by_the_block_method(ratio, monkeypatch):
    """Two brick bars tied node to node across a seam, their elastic
    modes at 20 kHz: the shift-inverted spectrum spreads over 4e8, so
    one solve leaves an elastic direction at 1e-9 of a block, which the
    Gram-matrix orthonormalization lost (2026-10-19: the block method
    handed the polish dependent vectors on the coincident ties and ran
    out of basis on the nearest-node ones, while ARPACK solved both).
    Now both solve by the block method, with no fallback, to the dense
    answer and to ARPACK's."""
    model = _tied_bars(ratio)
    dense = model.eigensolution(num_modes=8, solver='dense').frequency
    found = {}
    for solver in ('block', 'arpack'):
        monkeypatch.setattr(fem, 'SPARSE_EIGENSOLVER', solver)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            found[solver] = model.eigensolution(num_modes=8, solver='sparse').frequency
    assert int(np.sum(dense == 0.0)) == 6
    for solver, frequency in found.items():
        assert np.array_equal(frequency == 0.0, dense == 0.0), solver
        assert np.allclose(frequency[6:], dense[6:], rtol=1e-9, atol=0.0), solver


def test_a_tie_on_a_slanted_arm_holds_the_rotation_about_it(monkeypatch):
    """A fine mesh tied to a coarse one node to nearest node puts most
    ties on slanted arms, one follower to a lead: the lead's rotation
    about the arm moves nothing and nothing resists it, and the rule
    that holds such a rotation knew only axis-aligned lines (a desktop
    model of 2026-10-19 had 82 of them; Accelerate refused the matrix
    and SuperLU factored it into garbage). The lead's rotation is now
    written in a basis turned to the line, without the component about
    it: the model solves, dense and sparse alike, and without that the
    sparse solve is refused as unfactorable."""
    model = _tied_bars(ratio=2, slanted=True)
    oblique = model.oblique_lead_lines()
    assert len(oblique) == len(model.rigid_bodies()) == 12
    assert all(np.abs(line).max() < 1 - 1e-9 for _lead, line in oblique)
    # a line has no sign: (1, 1, 1) and (-1, -1, -1) are one direction
    lines = {tuple(np.round(line * np.sign(line[2]), 6)) for _lead, line in oblique}
    assert len(lines) == 4, 'all four diagonals'
    assert all(abs(np.linalg.norm(line) - 1.0) < 1e-12 for _lead, line in oblique)
    # one rotation fewer per slanted line
    transform = model.constraint_transform(sparse=True)
    followers = sum(len(body) - 1 for body in model.rigid_bodies())
    assert transform.shape[1] == len(model._free_dofs(())) - 6 * followers - len(oblique)
    dense = model.eigensolution(num_modes=8, solver='dense').frequency
    sparse = model.eigensolution(num_modes=8, solver='sparse').frequency
    assert int(np.sum(dense == 0.0)) == 6 and np.array_equal(sparse == 0.0, dense == 0.0)
    assert np.allclose(sparse[6:], dense[6:], rtol=1e-9, atol=0.0)
    monkeypatch.setattr(Model, 'oblique_lead_lines', lambda self: [])
    with pytest.raises(ValueError, match='could not be factored'):
        model.eigensolution(num_modes=8, solver='sparse')


def test_slanted_arms_and_a_springy_tie_share_one_transform():
    """A tie held by a spring across its normal gives the transform
    columns of its own past the kept ones, and a lead on a slanted arm
    turns its rotation columns by a second transform; the second must
    carry the tie's columns through. Both in one model (the branch that
    turned the rotations was cut before ties had columns of their own,
    2026-10-20): one column fewer per slanted line, one per tie (three
    translations, two of them back as their own columns), and the
    sparse solve the dense one's."""
    model = _tied_bars(ratio=2, slanted=True)
    nx, ny, nz = 6, 2, 2
    lower = {(i, j): 1 + nz * (nx + 1) * (ny + 1) + j * (nx + 1) + i
             for i in range(nx + 1) for j in range(ny + 1)}
    upper = {(i, j): 10001 + j * (2 * nx + 1) + i
             for i in range(2 * nx + 1) for j in range(2 * ny + 1)}
    for a, b in ((0, 0), (2, 0), (4, 2), (6, 2)):
        assert np.allclose(model._nodes[upper[2 * a, 2 * b]], model._nodes[lower[a, b]])
        model.add_tie(upper[2 * a, 2 * b], lower[a, b], (0, 0, 1),
                      shear_mode='stiffness', shear_stiffness=1e6)
    oblique = model.oblique_lead_lines()
    assert len(oblique) == 12
    followers = sum(len(body) - 1 for body in model.rigid_bodies())
    transform = model.constraint_transform(sparse=True)
    assert transform.shape[1] == (len(model._free_dofs(())) - 6 * followers
                                  - len(oblique) - 4)
    assert np.allclose(transform.toarray(), model.constraint_transform())
    dense = model.eigensolution(num_modes=8, solver='dense').frequency
    sparse = model.eigensolution(num_modes=8, solver='sparse').frequency
    assert int(np.sum(dense == 0.0)) == 6 and np.array_equal(sparse == 0.0, dense == 0.0)
    assert np.allclose(sparse[6:], dense[6:], rtol=1e-9, atol=0.0)


def test_superlu_refuses_a_factor_whose_solves_are_garbage(monkeypatch):
    """SuperLU reports success on a matrix singular to working precision
    and its solves come back garbage (a backward error of 11 on the
    tied desktop model, which two eigensolvers then ran on): a probe
    solve after the factorization refuses it. The matrix is the one
    that model makes, in small: bricks tied on slanted arms with the
    rotation about each arm left unheld."""
    from visualdynamics.core import sparse_solve
    from visualdynamics.core.sparse_solve import BACKEND_VARIABLE, SuperLUFactor

    model = _tied_bars(ratio=2, slanted=True)
    stiffness, mass, _t, _s, sigma, _k = model.scaled_system()
    sound = (stiffness - sigma * mass).tocsc()
    assert SuperLUFactor(sound).backend == 'superlu', 'the held matrix is taken'
    monkeypatch.setattr(Model, 'oblique_lead_lines', lambda self: [])
    stiffness, mass, _t, _s, sigma, _k = model.scaled_system()
    singular = (stiffness - sigma * mass).tocsc()
    with pytest.raises(RuntimeError, match='singular to working precision'):
        SuperLUFactor(singular)
    monkeypatch.setenv(BACKEND_VARIABLE, 'superlu')
    with pytest.raises(RuntimeError):
        sparse_solve.factor_symmetric(singular)


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
