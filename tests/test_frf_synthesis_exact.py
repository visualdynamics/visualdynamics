"""FRF synthesis checked against physics, not against itself.

`ShapeSet.synthesize_frf` sums the residue form over real modes,

    H_jk(w) = sum_r (iw)^p phi_jr phi_kr / (m_r (w_r^2 - w^2 + 2i z_r w_r w)).

Until 2026-09-23 the only test of that sum compared it with FRFs that
`testdata/generate_plate.py` had produced *with this same function*, so
a convention error shared by the two would have passed at machine
precision (`tests/test_frf_synthesis.py` says as much). This file
breaks the circle with answers that owe nothing to the code under test:

- a single-degree-of-freedom oscillator, whose receptance is
  1 / (k - w^2 m + i w c) by definition;
- a two-degree-of-freedom system with proportional damping, whose FRF
  matrix is the exact inverse of the dynamic stiffness K - w^2 M + i w C.
  For proportional damping the real-mode sum is not an approximation of
  that inverse — it *is* that inverse — so the two must agree to
  rounding. The modes come from `scipy.linalg.eigh`, which returns them
  mass-normalized; the damping ratio of each follows from
  C = alpha M + beta K as (alpha / w_r + beta w_r) / 2.

What these prove is the residue form, the mass normalization, the
damping convention and the powers of iw, each against a definition.

The second half does the same for complex shapes, which take the
pole-plus-conjugate form: a three-degree-of-freedom system damped at one
mass only, so no alpha M + beta K describes it and its modes are
complex. Its FRF is still the exact inverse of the dynamic stiffness,
and the complex-mode sum must equal it — with the shapes at the
eigenproblem's own scale and complex modal A, and again scaled to unit
modal A. Before 2026-09-23 complex shapes went through the real-mode
form, which no scaling can make right.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from visualdynamics.core.shapes import ShapeSet

#: lines to compare on: off zero, through both resonances below
FREQUENCIES = np.linspace(0.5, 60.0, 600)


def _exact(k, m, c, omega):
    """Receptance matrix, the inverse of the dynamic stiffness."""
    return np.array([np.linalg.inv(k - w * w * m + 1j * w * c)
                     for w in omega])


def test_a_single_oscillator_is_its_textbook_receptance():
    m, k, zeta = 2.5, 8.0e4, 0.03
    omega_n = np.sqrt(k / m)
    c = 2.0 * zeta * np.sqrt(k * m)
    omega = 2.0 * np.pi * FREQUENCIES
    exact = 1.0 / (k - omega ** 2 * m + 1j * omega * c)

    # one mode with a unit shape; the modal mass is the physical mass
    shapes = ShapeSet([omega_n / (2.0 * np.pi)], [zeta], ['1X+'],
                      [[1.0]], modal_mass=[m])
    for power in (0, 1, 2):
        got = shapes.synthesize_frf(FREQUENCIES, ['1X+'], ['1X+'],
                                    power=power)[0]
        want = (1j * omega) ** power * exact
        assert np.abs(got - want).max() < 1e-12 * np.abs(want).max(), (
            f'power {power}: the residue form is the oscillator it models'
        )


def _two_dof():
    """Masses, stiffnesses and a proportional damping matrix."""
    m = np.diag([1.2, 0.8])
    k = np.array([[3.0e4 + 1.5e4, -1.5e4],
                  [-1.5e4, 1.5e4]])
    alpha, beta = 0.6, 2.0e-5
    return m, k, alpha * m + beta * k, alpha, beta


def _modes(m, k, alpha, beta):
    """Mass-normalized modes of (K, M) and their damping ratios."""
    eigenvalues, vectors = scipy.linalg.eigh(k, m)
    omega_r = np.sqrt(eigenvalues)
    assert np.allclose(vectors.T @ m @ vectors, np.eye(2)), (
        'eigh returns shapes with unit modal mass'
    )
    return omega_r, (alpha / omega_r + beta * omega_r) / 2.0, vectors


def test_proportional_damping_is_the_exact_inverse_of_the_dynamic_stiffness():
    m, k, c, alpha, beta = _two_dof()
    omega_r, zeta, vectors = _modes(m, k, alpha, beta)
    omega = 2.0 * np.pi * FREQUENCIES
    exact = _exact(k, m, c, omega)

    shapes = ShapeSet(omega_r / (2.0 * np.pi), zeta, ['1X+', '2X+'],
                      vectors.T)                 # modes by DOFs
    pairs = [('1X+', '1X+'), ('1X+', '2X+'), ('2X+', '1X+'), ('2X+', '2X+')]
    responses = [p[0] for p in pairs]
    references = [p[1] for p in pairs]
    index = {'1X+': 0, '2X+': 1}
    for power in (0, 1, 2):
        got = shapes.synthesize_frf(FREQUENCIES, responses, references,
                                    power=power)
        for row, (j, r) in enumerate(pairs):
            want = (1j * omega) ** power * exact[:, index[j], index[r]]
            assert np.abs(got[row] - want).max() < 1e-10 * np.abs(want).max(), (
                f'H{j}/{r}, power {power}: the modal sum is the inverse'
            )


def test_the_modal_mass_carries_any_scaling_of_the_shapes():
    """Shapes scaled by s with modal mass s^2 describe the same system:
    the residue is phi phi / m, and both factors of s cancel against
    the mass. Unit-mass shapes are one scaling of many; the synthesis
    must not care which one a set arrived in."""
    m, k, _c, alpha, beta = _two_dof()
    omega_r, zeta, vectors = _modes(m, k, alpha, beta)
    unit = ShapeSet(omega_r / (2.0 * np.pi), zeta, ['1X+', '2X+'], vectors.T)
    s = np.array([3.7, 0.21])                    # a different scale per mode
    scaled = ShapeSet(omega_r / (2.0 * np.pi), zeta, ['1X+', '2X+'],
                      vectors.T * s[:, None], modal_mass=s ** 2)
    a = unit.synthesize_frf(FREQUENCIES, ['1X+', '2X+'], ['2X+', '2X+'])
    b = scaled.synthesize_frf(FREQUENCIES, ['1X+', '2X+'], ['2X+', '2X+'])
    assert np.abs(a - b).max() < 1e-12 * np.abs(a).max()


def test_reciprocity_holds_because_the_physics_does():
    """H_jk = H_kj for a structure with a symmetric mass and stiffness,
    which is every structure these shapes come from. A residue form
    that ever transposed a pair would break it."""
    m, k, _c, alpha, beta = _two_dof()
    omega_r, zeta, vectors = _modes(m, k, alpha, beta)
    shapes = ShapeSet(omega_r / (2.0 * np.pi), zeta, ['1X+', '2X+'],
                      vectors.T)
    forward, back = shapes.synthesize_frf(FREQUENCIES, ['1X+', '2X+'],
                                          ['2X+', '1X+'])
    assert np.abs(forward - back).max() < 1e-12 * np.abs(forward).max()


@pytest.mark.parametrize('zeta', [0.0005, 0.2])
def test_light_and_heavy_damping_both_hold(zeta):
    """The residue form is exact for any modal damping, not only the
    few-percent a structure usually has."""
    m, k = 1.0, 4.0e4
    omega_n = np.sqrt(k / m)
    c = 2.0 * zeta * np.sqrt(k * m)
    omega = 2.0 * np.pi * FREQUENCIES
    exact = 1.0 / (k - omega ** 2 * m + 1j * omega * c)
    shapes = ShapeSet([omega_n / (2.0 * np.pi)], [zeta], ['1X+'], [[1.0]],
                      modal_mass=[m])
    got = shapes.synthesize_frf(FREQUENCIES, ['1X+'], ['1X+'])[0]
    assert np.abs(got - exact).max() < 1e-12 * np.abs(exact).max()


# ---- complex modes: damping that is not proportional ---------------------
#
# A dashpot on one mass only cannot be written as alpha M + beta K, so the
# modes are complex: each DOF passes its peak at its own phase. The
# structure's FRF is still the inverse of K - w^2 M + i w C, and the
# first-order (state-space) eigenproblem gives the poles and shapes whose
# pole-plus-conjugate sum *is* that inverse. Modal A is psi^T (2 l M + C)
# psi, the norm of the state-space eigenvector [psi; l psi] against
# [[C, M], [M, 0]].


def _non_proportional():
    m = np.diag([1.0, 1.5, 0.8])
    k = np.array([[4.0e5, -2.0e5, 0.0],
                  [-2.0e5, 5.0e5, -3.0e5],
                  [0.0, -3.0e5, 3.0e5]])
    c = np.diag([40.0, 0.0, 0.0])
    return m, k, c


def _complex_modes(m, k, c):
    """Poles in the upper half plane, their shapes, and each modal A."""
    n = len(m)
    state = np.block([[np.zeros((n, n)), np.eye(n)],
                      [-np.linalg.solve(m, k), -np.linalg.solve(m, c)]])
    poles, vectors = np.linalg.eig(state)
    upper = np.flatnonzero(poles.imag > 0)
    upper = upper[np.argsort(poles[upper].imag)]
    poles, psi = poles[upper], vectors[:n, upper]
    modal_a = np.array([psi[:, r] @ (2.0 * poles[r] * m + c) @ psi[:, r]
                        for r in range(len(poles))])
    return poles, psi, modal_a


def _complex_set(poles, shapes, modal_mass=None):
    return ShapeSet(np.abs(poles) / (2.0 * np.pi), -poles.real / np.abs(poles),
                    ['1X+', '2X+', '3X+'], shapes.T, modal_mass=modal_mass)


DOFS3 = ['1X+', '2X+', '3X+']
PAIRS3 = [(j, r) for j in DOFS3 for r in DOFS3]


def _check_against_exact(shapes, m, k, c, lines):
    omega = 2.0 * np.pi * lines
    exact = _exact(k, m, c, omega)
    for power in (0, 1, 2):
        got = shapes.synthesize_frf(lines, [p[0] for p in PAIRS3],
                                    [p[1] for p in PAIRS3], power=power)
        want = np.array([(1j * omega) ** power
                         * exact[:, DOFS3.index(j), DOFS3.index(r)]
                         for j, r in PAIRS3])
        worst = np.abs(got - want).max() / np.abs(want).max()
        assert worst < 1e-10, (
            f'power {power}: complex-mode sum departs from the inverse '
            f'by {worst:.1e} of peak')


def test_the_modes_of_this_system_really_are_complex():
    """So the tests below mean something: the shapes' phases spread well
    away from 0 and 180 degrees, and modal A is itself complex."""
    _poles, psi, modal_a = _complex_modes(*_non_proportional())
    reference = psi[np.argmax(np.abs(psi), axis=0), range(psi.shape[1])]
    phase = np.degrees(np.angle(psi / reference))
    off_axis = np.abs(((phase + 90.0) % 180.0) - 90.0)
    assert off_axis.max() > 30.0
    assert np.abs(modal_a.imag).min() > 0.1 * np.abs(modal_a).min()


def test_complex_modes_with_their_modal_a_are_the_exact_inverse():
    """The shapes as the eigenproblem gives them, arbitrary complex
    scale and all, with that scale carried in modal A — which is
    complex here, so the conjugate term must divide by conj(A)."""
    m, k, c = _non_proportional()
    poles, psi, modal_a = _complex_modes(m, k, c)
    lines = np.linspace(0.5, 1.3 * np.abs(poles).max() / (2 * np.pi), 1500)
    _check_against_exact(_complex_set(poles, psi, modal_a), m, k, c, lines)


def test_unity_modal_a_shapes_need_no_modal_mass():
    """The same modes scaled to unit modal A, the usual way a curve
    fitter hands complex shapes over: no `modal_mass` at all, and the
    same exact answer."""
    m, k, c = _non_proportional()
    poles, psi, modal_a = _complex_modes(m, k, c)
    lines = np.linspace(0.5, 1.3 * np.abs(poles).max() / (2 * np.pi), 1500)
    unit = psi / np.sqrt(modal_a)
    _check_against_exact(_complex_set(poles, unit), m, k, c, lines)


def test_a_real_mode_written_as_complex_is_the_same_frf():
    """The two forms connect: a real mode is a complex one whose modal A
    is 2i w_d m. Written that way it must synthesize exactly what the
    real form does, which ties the first-order branch to the
    second-order one already pinned above."""
    m, k, _c, alpha, beta = _two_dof()
    omega_r, zeta, vectors = _modes(m, k, alpha, beta)
    real = ShapeSet(omega_r / (2.0 * np.pi), zeta, ['1X+', '2X+'], vectors.T)
    damped = omega_r * np.sqrt(1.0 - zeta ** 2)
    as_complex = ShapeSet(omega_r / (2.0 * np.pi), zeta, ['1X+', '2X+'],
                          vectors.T.astype(complex),
                          modal_mass=2j * damped)
    for power in (0, 1, 2):
        a = real.synthesize_frf(FREQUENCIES, ['1X+', '2X+'], ['2X+', '2X+'],
                                power=power)
        b = as_complex.synthesize_frf(FREQUENCIES, ['1X+', '2X+'],
                                      ['2X+', '2X+'], power=power)
        assert np.abs(a - b).max() < 1e-12 * np.abs(a).max()
