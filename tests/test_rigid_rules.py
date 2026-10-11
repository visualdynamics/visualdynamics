"""The rigid set's recognition and settings, pinned by value where a
mutation pass found them unguarded (2026-10-11).

`tools/mutate.py` over rigid.py caught 56 of 65 against test_rigid.py;
six of the nine were `is_rigid_set`, which nothing asked to say no.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.rigid import MassProperties, is_rigid_set, rigid_body_shapes
from visualdynamics.units import SYSTEMS


def _rigid():
    geometry = visualdynamics.Geometry([1, 2, 3], np.eye(3), length_unit='m')
    return rigid_body_shapes(geometry, MassProperties((0.0, 0.0, 0.0)))


def test_a_rigid_set_is_read_off_the_set_itself():
    rigid = _rigid()
    assert is_rigid_set(rigid)
    assert is_rigid_set(copy.deepcopy(rigid)), 'a copy is still one'


def _altered(change):
    rigid = _rigid()
    change(rigid)
    return rigid


@pytest.mark.parametrize('change', [
    lambda s: s.delete_modes([5]),                                   # five modes
    lambda s: s.frequency.__setitem__(2, 0.1),                       # one not at 0 Hz
    lambda s: s.description.__setitem__(0, 'Mode 1'),                # not described so
    lambda s: s.description.__setitem__(5, 'Rigid translation Z'),   # a translation last
    lambda s: s.description.__setitem__(2, 'Rigid rotation about X'),  # a rotation third
])
def test_anything_else_is_not_one(change):
    assert not is_rigid_set(_altered(change))


def test_a_seventh_mode_makes_it_not_one():
    rigid = _rigid()
    seven = visualdynamics.ShapeSet(
        np.zeros(7), np.zeros(7), rigid.coordinate,
        np.vstack([rigid.shape_matrix, rigid.shape_matrix[:1]]),
        description=list(rigid.description) + ['Rigid rotation about X'])
    assert not is_rigid_set(seven)


def test_an_inertia_tensor_with_a_zero_moment_is_refused():
    MassProperties((0.0, 0.0, 0.0), 1.0, (1.0, 1.0, 1e-9, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match='not positive definite'):
        MassProperties((0.0, 0.0, 0.0), 1.0, (1.0, 1.0, 0.0, 0.0, 0.0, 0.0))


def test_unit_shapes_are_described_in_a_unit_system_without_a_mass():
    millimeters = SYSTEMS['mm-kg-N-s']
    assert MassProperties((0.0254, 0.0, 0.0)).describe(millimeters) == \
        'about (25.4, 0, 0) mm, unit shapes'
    inches = SYSTEMS['in-slinch-lbf-s']
    scaled = MassProperties((0.0254, 0.0, 0.0), 175.126835, (1, 1, 1, 0, 0, 0))
    assert scaled.describe(inches) == 'about (1, 0, 0) in, mass-normalized (1 slinch)'
