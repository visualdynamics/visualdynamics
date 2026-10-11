"""The validators' refusals, pinned by the values they quote back
(2026-10-11).

`tools/mutate.py` over validate.py caught 44 of 61 against the tests
that reach it through geometry, data and transforms, which check that
a bad id is refused but not what the refusal says. A refusal is read
by someone fixing a file, so which entries it names is the point.
"""

from __future__ import annotations

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
