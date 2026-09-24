"""FRF synthesis from a modal model, against sdynpy's own.

The second, independent opinion on `ShapeSet.synthesize_frf`. The first
is exact physics (`tests/test_frf_synthesis_exact.py`); this is the tool
the audience already trusts, synthesizing the same modal model, frozen
by `generate_modal_frf_oracle.py` in the private generators repository
and read here as numbers — never an import (`AGENTS.md` hard rule 1).

Real shapes in the second-order residue form, and complex shapes in the
first-order pole-plus-conjugate form with the modal mass read as modal
A — the branch Visual Dynamics gained on 2026-09-23. sdynpy holds modal
A as a real number and divides the conjugate term by it rather than its
conjugate, which is the same thing for a real modal A, so that is what
is frozen; a complex modal A is checked against exact physics in
`tests/test_frf_synthesis_exact.py`.

The model has unequal modal masses and damping from 1.2% to 8%, and is
compared as displacement, velocity and acceleration. Each cell of the
frozen array carries its own (response, reference) labels, so the
orientation is read from the file rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

from visualdynamics.core.shapes import ShapeSet

ORACLE = fixture_path('sdynpy_oracle', 'modal_frf.npz')


@pytest.fixture(scope='module')
def oracle():
    with np.load(ORACLE) as frozen:
        return {key: frozen[key] for key in frozen.files}


def _shapes(oracle):
    return ShapeSet(oracle['frequency'], oracle['damping'],
                    [str(d) for d in oracle['dofs']],
                    oracle['shape_matrix'], modal_mass=oracle['modal_mass'])


@pytest.mark.parametrize('power,key', [(0, 'frf_displacement'),
                                       (1, 'frf_velocity'),
                                       (2, 'frf_acceleration')])
def test_the_synthesis_agrees_with_sdynpys(oracle, power, key):
    shapes = _shapes(oracle)
    pairs = oracle['coordinate_pairs'].reshape(-1, 2)
    responses = [str(p[0]) for p in pairs]
    references = [str(p[1]) for p in pairs]
    got = shapes.synthesize_frf(oracle['lines'], responses, references,
                                power=power)
    want = oracle[key].reshape(-1, len(oracle['lines']))
    assert got.shape == want.shape
    worst = np.abs(got - want).max() / np.abs(want).max()
    assert worst < 1e-12, (
        f'{key}: synthesis departs from sdynpy by {worst:.2e} of peak'
    )


@pytest.mark.parametrize('power,key', [(0, 'complex_frf_displacement'),
                                       (1, 'complex_frf_velocity'),
                                       (2, 'complex_frf_acceleration')])
def test_complex_shapes_agree_with_sdynpys_too(oracle, power, key):
    shapes = ShapeSet(oracle['frequency'], oracle['damping'],
                      [str(d) for d in oracle['dofs']],
                      oracle['complex_shape_matrix'],
                      modal_mass=oracle['complex_modal_a'])
    pairs = oracle['coordinate_pairs'].reshape(-1, 2)
    got = shapes.synthesize_frf(oracle['lines'], [str(p[0]) for p in pairs],
                                [str(p[1]) for p in pairs], power=power)
    want = oracle[key].reshape(-1, len(oracle['lines']))
    worst = np.abs(got - want).max() / np.abs(want).max()
    assert worst < 1e-12, (
        f'{key}: synthesis departs from sdynpy by {worst:.2e} of peak'
    )


def test_the_frozen_model_exercises_what_it_claims_to(oracle):
    """So the agreement above means something: unequal modal masses,
    a real spread of damping, and shapes that are not all one sign."""
    assert np.ptp(oracle['modal_mass']) > 1.0
    assert oracle['damping'].max() / oracle['damping'].min() > 5
    assert (oracle['shape_matrix'] < 0).any()
    assert not np.iscomplexobj(oracle['shape_matrix']), 'a real set'
    phases = np.angle(oracle['complex_shape_matrix'])
    assert np.ptp(phases) > np.pi, 'and a complex one, phases spread'
    assert not np.allclose(oracle['complex_modal_a'], 1.0)
