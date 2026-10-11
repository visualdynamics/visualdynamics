"""What value an ESCDF property may hold, pinned case by case where a
mutation pass found it unguarded (2026-10-11).

`tools/mutate.py` over escdf.py caught 118 of 152 against its two test
files. `_fits` decides which of a property's alternatives a value is
written as, and the tests wrote values that fit; here each refusal is
asked for by name, beside the value one step inside it.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.io.escdf import Property, _fits


@pytest.mark.parametrize('dtype, value, fits', [
    ('f8', 1.5, True),
    ('f8', 1.5 + 0.5j, False),          # complex into a real property
    ('c16', 1.5 + 0.5j, True),
    ('c16', 1.5, True),                 # a real is a complex with no imaginary part
    ('i8', 3, True),
    ('i8', 3.0, False),                 # a float into an integer
    ('u8', 0, True),
    ('u8', -1, False),                  # a negative into an unsigned
    ('i8', -1, True),
    ('f8', 'text', False),
    ('f8', True, False),
])
def test_a_scalar_fits_a_type_it_casts_to_safely(dtype, value, fits):
    assert _fits(Property('x', dtype), value) is fits


def test_an_unsigned_array_refuses_any_negative_and_an_empty_one_fits():
    prop = Property('x', 'u4', shape=('n',))
    assert _fits(prop, np.array([0, 1, 2]))
    assert not _fits(prop, np.array([0, -1, 2]))
    assert _fits(prop, np.array([], dtype=np.int64))


def test_a_literal_size_must_match_and_a_named_one_need_not():
    assert _fits(Property('x', 'f8', shape=(3,)), [1.0, 2.0, 3.0])
    assert not _fits(Property('x', 'f8', shape=(3,)), [1.0, 2.0])
    assert _fits(Property('x', 'f8', shape=('num_nodes', 3)), np.zeros((7, 3)))
    assert not _fits(Property('x', 'f8', shape=('num_nodes', 3)), np.zeros(3))
