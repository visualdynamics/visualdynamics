"""The validators' refusals, pinned by the values they quote back
(2026-10-11).

`tools/mutate.py` over validate.py caught 44 of 61 against the tests
that reach it through geometry, data and transforms, which check that
a bad id is refused but not what the refusal says. A refusal is read
by someone fixing a file, so which entries it names is the point.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from visualdynamics.core import validate


def test_ids_quote_the_first_three_that_are_not_whole():
    with pytest.raises(ValueError, match=r'node ids must be whole numbers; got 1\.5, 2\.5, 3\.5$'):
        validate.ids([1.0, 1.5, 2.5, 3.5, 4.5], 'node ids')
    assert validate.ids([1.0, 2.0], 'node ids').tolist() == [1, 2]


def test_ids_quote_the_first_three_that_are_negative():
    with pytest.raises(ValueError, match=r'node ids cannot be negative; got -1, -2, -3$'):
        validate.ids([0, -1, 5, -2, -3, -4], 'node ids')
    assert validate.ids([0, 3], 'node ids').tolist() == [0, 3], 'zero is an id'


def test_duplicates_are_named_once_each_five_at_most():
    with pytest.raises(ValueError, match=r'duplicate node ids: 2, 3$'):
        validate.ids([1, 2, 2, 3, 2, 3], 'node ids', unique=True)
    with pytest.raises(ValueError, match=r'duplicate node ids: 1, 2, 3, 4, 5$'):
        validate.ids(list(range(1, 7)) * 2, 'node ids', unique=True)
    assert validate.ids([1, 1], 'element ids').tolist() == [1, 1], 'unique on request'


def test_a_negative_frequency_or_damping_is_quoted_three_at_most():
    with pytest.raises(ValueError,
                       match=r'frequency cannot be negative; got -1, -2, -3$'):
        validate.frequency([1.0, -1.0, -2.0, 0.0, -3.0, -4.0])
    assert validate.damping([0.0]).tolist() == [0.0]


def test_a_dof_with_no_direction_lists_the_ones_it_could_have():
    with pytest.raises(ValueError, match=r"DOF '101Q' names no direction: expected "
                                         r"one of X\+, X-, Y\+"):
        validate.dofs(['101Q'])
    with pytest.raises(ValueError, match=r"DOF '101' names no direction: expected one of X\+"):
        validate.dofs(['101'])


def test_direction_codes_count_the_axes_from_one_negated_for_minus():
    assert validate.DIRECTION_CODES == {
        'X+': 1, 'X-': -1, 'Y+': 2, 'Y-': -2, 'Z+': 3, 'Z-': -3,
        'RX+': 4, 'RX-': -4, 'RY+': 5, 'RY-': -5, 'RZ+': 6, 'RZ-': -6}


@pytest.mark.parametrize('text, index', [
    ('M1', 1), ('m12', 12), (' M3 ', 3), ('M', None), ('M0', None),
    ('MX', None), ('1', None), ('', None)])
def test_a_modal_coordinate_is_the_letter_and_a_positive_index(text, index):
    assert validate.modal_coordinate(text) == index


#: strings `dofs` rewrites or refuses, which its fast path must hand
#: to the per-string rule rather than pass through: each is one step
#: off the normalized spelling the fast path recognizes
IRREGULAR = ['101z', '0rx-', '007Y+', 'm03', ' 12Z- ', '١X+', '1١X+', 'M3X+',
             '101Q', '1X+\n2Y+', '', None, 1, 'Z+', '7', 'M0']


def test_the_fast_path_agrees_with_the_per_string_rule_on_a_mixed_list():
    mixed = ['101X+', '101z', '0rx-', '007Y+', 'M3', 'm03', ' 12Z- ',
             '١X+', None, 'Z+', '7']
    expected = ['101X+', '101Z+', '0RX-', '7Y+', 'M3', 'M3', '12Z-',
                '1X+', '', 'Z+', '7']
    assert validate.dofs(mixed, allow_unknown=True) == expected
    assert validate._dofs_each(mixed, 'DOF', True) == expected
    with pytest.raises(ValueError, match=r'^DOF is empty$'):
        validate.dofs(mixed)
    with pytest.raises(ValueError, match=r"^DOF 'Z\+' does not start with a node id$"):
        validate.dofs([v for v in mixed if v is not None])


def test_a_normalized_list_takes_the_fast_path_and_anything_else_does_not(
        monkeypatch):
    normalized = ['1X+', '0RZ-', '45087RX+', 'M12', '7Y-']

    def refuse(*args):
        raise AssertionError('the per-string rule was called')
    monkeypatch.setattr(validate, '_dofs_each', refuse)
    out = validate.dofs(normalized)
    assert out == normalized and out is not normalized
    assert all(type(s) is str for s in validate.dofs(np.array(normalized)))
    for item in IRREGULAR:
        with pytest.raises(AssertionError, match='per-string rule'):
            validate.dofs(['1X+', item, 'M2'], allow_unknown=True)


@pytest.mark.parametrize('item', IRREGULAR)
def test_an_irregular_string_among_normalized_ones_gets_the_rule(item):
    values = ['1X+', item, 'M2']
    try:
        expected = validate._dofs_each(values, 'DOF', False)
    except ValueError as refusal:
        with pytest.raises(ValueError, match='^' + re.escape(str(refusal)) + '$'):
            validate.dofs(values)
    else:
        assert validate.dofs(values) == expected
    try:
        expected = validate._dofs_each(values, 'DOF', True)
    except ValueError as refusal:
        with pytest.raises(ValueError, match='^' + re.escape(str(refusal)) + '$'):
            validate.dofs(values, allow_unknown=True)
    else:
        assert validate.dofs(values, allow_unknown=True) == expected
