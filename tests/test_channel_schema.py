"""The channel table's schema: the columns visualdynamics interprets.

Identity (channel, node, direction), purpose (role, control),
calibration (unit, sensitivity, range) — typed, validated at entry,
defaulted where a source never said, with everything else a file
carries preserved behind them.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.channel_table import ChannelTable


def bare(**extra):
    columns = {'channel': [1, 2, 3], 'node': ['101', '102', '103'],
               'direction': ['Z+', 'Z+', 'X+'], 'unit': ['g', 'g', 'N'],
               **extra}
    return ChannelTable(columns)


def test_every_schema_column_exists_with_its_default():
    """A source that never said grows the columns, visibly defaulted —
    which is also the whole migration story for older saved tables."""
    table = bare()
    assert table.column_names[:len(table.SCHEMA)] == list(table.SCHEMA)
    assert list(table['role']) == [''] * 3, 'undeclared, like a unit'
    assert list(table['control']) == ['False'] * 3
    assert list(table['range']) == [5] * 3, 'the +/-5 V most hardware has'
    assert list(table['sensitivity']) == [''] * 3


def test_foreign_columns_are_dropped():
    """The schema is the whole table now. A Rattlesnake save carries its
    coupling, excitation and feedback wiring — how the controller ran,
    not what was measured — and carrying it made this object a place
    data went to be ignored."""
    table = bare(warning_level=['a', 'b', 'c'], coupling=['AC'] * 3)
    assert table.column_names == list(table.SCHEMA)
    assert 'warning_level' not in table.column_names


def test_a_column_spelled_differently_lands_in_its_own():
    """Dropping the extras only works because what belongs is
    recognized: 'Serial Number', 'serial-number' and 'SN' are one
    column, and 'Cal Due' is the expiration."""
    from visualdynamics.core.channel_table import canonical_name

    for spelling in ('Serial Number', 'serial-number', 'SN', 'serial_no'):
        assert canonical_name(spelling) == 'serial_number', spelling
    assert canonical_name('Cal Due') == 'expiration'
    assert canonical_name('Point') == 'node'
    assert canonical_name('Units:') == 'unit'
    table = bare(**{'Cal Due': ['2027-01-04', '', '']})
    assert next(iter(table['expiration'])) == '2027-01-04'


def test_role_vocabulary_is_enforced():
    table = bare()
    table.set_cell('role', 0, 'Reference')     # any case
    assert table.roles()[0] == 'reference'
    table.set_cell('role', 0, '')              # undeclared is reachable
    with pytest.raises(ValueError, match='reference, response, monitor'):
        table.set_cell('role', 0, 'drive')


def test_control_implies_response_both_ways():
    """The one nonsense pair is unrepresentable from either door."""
    table = bare()
    with pytest.raises(ValueError, match='control channel is a response'):
        table.set_cell('control', 0, 'True')
    table.set_cell('role', 0, 'response')
    table.set_cell('control', 0, 'True')
    assert bool(table.controls()[0])
    with pytest.raises(ValueError, match='uncheck Control first'):
        table.set_cell('role', 0, 'reference')
    table.set_cell('control', 0, 'False')
    table.set_cell('role', 0, 'reference')     # now it may drive


def test_channel_numbers_stay_unique():
    """The channel number is the join key for everything linking will
    mean; two rows with one number is two claims on one wire."""
    table = bare()
    with pytest.raises(ValueError, match='already exists'):
        table.set_cell('channel', 1, '1')
    table.set_cell('channel', 1, '9')
    assert list(table['channel']) == [1, 9, 3]


def test_sensitivity_and_range_are_positive_numbers_or_blank():
    table = bare()
    table.set_cell('sensitivity', 0, '10.2')
    table.set_cell('range', 0, '2')
    with pytest.raises(ValueError, match='whole number'):
        table.set_cell('range', 0, '2.5')       # volts, as an integer
    with pytest.raises(ValueError):
        table.set_cell('sensitivity', 0, 'ten')
    with pytest.raises(ValueError, match='positive'):
        table.set_cell('range', 0, '-5')
    table.set_cell('sensitivity', 1, '')
    assert np.isnan(table.sensitivities()[1])
    assert table.sensitivities()[0] == 10.2
    assert table.ranges()[0] == 2


def test_a_foreign_role_word_reads_as_undeclared():
    """A file may carry a 'role' column in its own vocabulary; reading
    it must not poison the table."""
    table = bare(role=['DRIVE', 'response', ''])
    assert table.roles() == ['', 'response', '']


def test_a_rattlesnake_run_states_its_roles():
    """A channel with a feedback device is a drive — rattlesnake's own
    rule — and every other enabled channel is measured as a response.
    Mapping that is translation, not guessing."""
    run = visualdynamics.import_file(fixture_path('plate', 'modal.nc4'))
    table = run['channel_table']
    roles = table.roles()
    assert roles.count('reference') == 2, 'the two shaker drives'
    assert roles.count('response') == len(roles) - 2
    dofs = np.array(table.dof_strings())
    assert set(dofs[[r == 'reference' for r in roles]]) == {'101Z+',
                                                            '1310Z+'}


def test_a_random_run_states_its_control_channels():
    """The requirement is written per control channel, so a response
    channel whose DOF the specification names is a control channel —
    and a drive sharing a controlled DOF stays a reference."""
    run = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    table = run['channel_table']
    controls = table.controls()
    spec = next(obj for name, obj in run.items()
                if name.endswith('_specification'))
    dofs = np.array(table.dof_strings())
    assert set(dofs[controls]) == set(spec.response_dof)
    assert controls.sum() == len(set(spec.response_dof))
    roles = np.array(table.roles())
    assert all(roles[controls] == 'response')


# --- every column says what it holds -----------------------------------


def test_direction_is_one_of_the_twelve():
    """Six translations and six rotations, and nothing else — the same
    vocabulary a DOF string is parsed against, so a table and a record
    cannot disagree about what a direction is."""
    table = bare()
    for good in ('Z+', 'z+', 'RX-', 'Y-'):
        table.set_cell('direction', 0, good)
    assert table['direction'][0] == 'Y-'
    for bad in ('Q+', 'up', 'ZZ', '+Z'):
        with pytest.raises(ValueError, match='direction is one of'):
            table.set_cell('direction', 0, bad)
    table.set_cell('direction', 0, '')      # undeclared stays reachable


def test_channel_type_is_a_quantity_and_narrows_the_units():
    """The type is the dimension vocabulary the rest of the package
    uses, so the unit shortlist follows from it rather than being a
    second list to keep in step."""
    table = bare()
    table.set_cell('channel_type', 0, 'Acceleration')   # any case
    assert table.types()[0] == 'acceleration'
    assert 'g' in table.units_for(0) and 'N' not in table.units_for(0)
    with pytest.raises(ValueError, match='channel_type is one of'):
        table.set_cell('channel_type', 0, 'wobble')


def test_a_unit_and_a_type_have_to_describe_one_channel():
    table = bare()
    table.set_cell('channel_type', 0, 'force')
    with pytest.raises(ValueError, match='declared force'):
        table.set_cell('unit', 0, 'm/s**2')
    table.set_cell('unit', 0, 'N')
    # changing the type withdraws a unit that no longer answers
    table.set_cell('channel_type', 0, 'acceleration')
    assert table['unit'][0] == ''


def test_triax_dof_is_one_leg_of_three():
    table = bare()
    table.set_cell('triax_dof', 0, 'y')
    assert table['triax_dof'][0] == 'Y'
    with pytest.raises(ValueError, match='triax_dof is one of'):
        table.set_cell('triax_dof', 0, 'Z+')   # a direction, not a leg


def test_expiration_is_a_date_however_it_was_written():
    table = bare()
    for written, iso in (('2027-01-04', '2027-01-04'),
                         ('01/04/2027', '2027-01-04'),
                         ('2027-01-04 00:00:00', '2027-01-04')):
        table.set_cell('expiration', 0, written)
        assert table['expiration'][0] == iso, written
    with pytest.raises(ValueError, match='not a date'):
        table.set_cell('expiration', 0, 'next Tuesday')


def test_a_node_is_a_non_negative_whole_number_or_nothing():
    table = bare()
    table.set_cell('node', 0, '0')            # a legal node id
    assert table['node'][0] == '0'
    table.set_cell('node', 0, '')             # and blank is 'not recorded'
    assert table['node'][0] == ''
    with pytest.raises(ValueError, match='node must be 0 or more'):
        table.set_cell('node', 0, '-3')


def test_a_channel_number_is_positive_whole_and_never_blank():
    """It is the join key: a channel without a number is not a channel,
    which is what separates it from a node nobody wrote down."""
    table = bare()
    with pytest.raises(ValueError, match='cannot be blank'):
        table.set_cell('channel', 0, '')
    with pytest.raises(ValueError, match='must be positive'):
        table.set_cell('channel', 0, '0')
    with pytest.raises(ValueError, match='whole number'):
        table.set_cell('channel', 0, '1.5')
    table.set_cell('channel', 0, '7.0')       # a spreadsheet's float
    assert table['channel'][0] == 7


def test_a_file_is_read_leniently_where_an_edit_is_refused():
    """The two strictnesses. A source full of rubbish still imports, so
    there is something to fix; typing the same rubbish is refused, with
    the reason."""
    table = ChannelTable({'channel': [1, 2], 'node': ['101', 'nonsense'],
                          'direction': ['Z+', 'sideways'],
                          'unit': ['g', 'wibble'],
                          'expiration': ['2027-01-04', 'someday']})
    assert table.num_channels == 2
    assert list(table['node']) == ['101', '']
    assert list(table['direction']) == ['Z+', '']
    assert list(table['unit']) == ['g', '']
    assert list(table['expiration']) == ['2027-01-04', '']
    with pytest.raises(ValueError):
        table.set_cell('direction', 1, 'sideways')
