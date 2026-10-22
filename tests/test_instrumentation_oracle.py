"""Effective Independence against sdynpy's, frozen as data.

`testdata/sdynpy_oracle/instrumentation.npz` holds sdynpy's choices for
three problems (`generate_instrumentation_oracle.py` in the private
generators repository; README.md beside the data says why the numbers
travel and the code never does). sdynpy removes one candidate at a time
by the same measure `core.instrumentation.effective_independence` uses,
so where Visual Dynamics also removes one at a time the two must choose
the same sensors. Where it thins in batches, the choice may differ, and
the information kept is what is compared.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from visualdynamics.core import instrumentation as ins

ORACLE = pathlib.Path(__file__).resolve().parent.parent / 'testdata' / 'sdynpy_oracle'


@pytest.fixture(scope='module')
def frozen():
    return np.load(ORACLE / 'instrumentation.npz')


def _out_of_plane(shapes):
    """A plate's out-of-plane shapes as blocks, measured along Z."""
    blocks = np.zeros((len(shapes), 3, shapes.shape[1]))
    blocks[:, 2] = shapes
    return blocks


def test_tri_axial_blocks_are_chosen_as_sdynpy_chooses_them(frozen):
    placed = ins.effective_independence(frozen['tri_blocks'], 8)
    assert sorted(placed.triaxial) == frozen['tri_kept'].tolist()


def test_uni_axial_points_on_a_plate_are_chosen_as_sdynpy_chooses_them(frozen):
    placed = ins.effective_independence(_out_of_plane(frozen['uni_shapes']), 0, 16)
    assert sorted(k for k, _axis in placed.uniaxial) == frozen['uni_kept'].tolist()
    assert {axis for _k, axis in placed.uniaxial} == {2}


def test_thinned_in_batches_the_information_kept_is_no_less(frozen):
    """315 points for 12 sensors: Visual Dynamics removes a tenth at a time
    until 48 remain, sdynpy one at a time throughout. The sets differ;
    the determinant of the information they keep is within 1% of
    sdynpy's or above it (12.857 against 12.847 as frozen)."""
    placed = ins.effective_independence(_out_of_plane(frozen['batched_shapes']), 0, 12)
    ours = np.linalg.slogdet(placed.information)[1]
    assert ours >= float(frozen['batched_logdet']) - np.log(1.01)
