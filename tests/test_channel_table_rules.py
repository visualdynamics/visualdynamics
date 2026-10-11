"""The channel table's refusals and edits, pinned by value where a
mutation pass found them unguarded by the schema tests (2026-10-11)."""

from __future__ import annotations

import pytest

from visualdynamics.core.channel_table import ChannelTable
from visualdynamics.core.unit_choices import (
    ALL_ORDINATE_UNITS,
    ORDINATE_UNITS,
    base_dimension,
    engineering_unit,
    squared_per_hz,
    units_for,
)


def _table():
    return ChannelTable({'channel': [1, 2, 3], 'node': [101, 102, 103],
                         'direction': ['Z+', 'Z+', 'X+'], 'unit': ['g', 'g', 'lbf']})


def test_a_delete_names_rows_that_exist_and_leaves_one():
    table = _table()
    with pytest.raises(IndexError, match=r'no such channels: \[3\]'):
        table.delete_channels([3])
    with pytest.raises(IndexError, match=r'no such channels: \[-1\]'):
        table.delete_channels([-1])
    with pytest.raises(ValueError, match='cannot delete every channel'):
        table.delete_channels([0, 1, 2])
    table.delete_channels([0])
    assert table.dof_strings() == ['102Z+', '103X+']


def test_columns_of_different_lengths_name_the_odd_one():
    with pytest.raises(ValueError, match="column 'node' has 2 entries, expected 3"):
        ChannelTable({'channel': [1, 2, 3], 'node': [101, 102],
                      'direction': ['Z+', 'Z+', 'X+']})


def test_a_sensitivity_is_more_than_zero():
    table = _table()
    table.set_cell('sensitivity', 0, '0.001')
    with pytest.raises(ValueError, match='sensitivity must be greater than 0'):
        table.set_cell('sensitivity', 0, '0')


def test_a_new_type_withdraws_a_unit_of_another_kind_and_keeps_its_own():
    table = _table()
    table.set_cell('channel_type', 2, 'acceleration')
    assert table.frame['unit'][2] == '', 'lbf is not an acceleration'
    table.set_cell('channel_type', 0, 'acceleration')
    assert table.frame['unit'][0] == 'g'


def test_the_quantity_a_unit_is_picked_for_is_the_leading_term():
    assert base_dimension('acceleration**2/frequency') == 'acceleration'
    assert base_dimension('velocity/force') == 'velocity'
    assert units_for('acceleration/force') == ORDINATE_UNITS['acceleration']
    assert units_for(None) == ALL_ORDINATE_UNITS
    assert units_for('unheard of', ['x']) == ['x']


@pytest.mark.parametrize('unit, label', [
    ('g', 'g²/Hz'), ('m/s**2', '(m/s²)²/Hz'), ('', ''), (None, '')])
def test_a_psd_unit_reads_as_its_square_per_hertz_and_back(unit, label):
    assert squared_per_hz(unit) == label
    if unit:
        assert engineering_unit(label) == unit
        assert engineering_unit(unit) == unit, 'the suffix is optional'
