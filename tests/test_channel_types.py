"""A channel's type, when the file spells it in its own words.

A source states each channel's type however it likes, and the import
is lenient: a spelling the schema does not know blanks rather than
failing. So a controller writing 'Accel' — or nothing — lost the
type while its 'Voltage' channels kept theirs, and a table of
accelerometers in G came in typeless (Brandon, 2026-09-20). The unit
beside it is the same claim said another way.
"""

from __future__ import annotations

import pytest

from visualdynamics.core.channel_table import CHANNEL_TYPES, ChannelTable


def _one(channel_type, unit):
    table = ChannelTable({'channel': [1], 'node': ['101'], 'direction': ['Z+'],
                          'channel_type': [channel_type], 'unit': [unit]})
    return str(table['channel_type'][0]), str(table['unit'][0])


@pytest.mark.parametrize('unit,expected', [
    ('G', 'acceleration'), ('g', 'acceleration'), ('m/s^2', 'acceleration'),
    ('in/s**2', 'acceleration'), ('V', 'voltage'), ('mV', 'voltage'),
    ('N', 'force'), ('lbf', 'force'), ('m/s', 'velocity'), ('m', 'length'),
    ('Pa', 'pressure'), ('degC', 'temperature'),
])
def test_a_blank_type_takes_the_unit_s_answer(unit, expected):
    assert _one('', unit) == (expected, unit)
    assert expected in CHANNEL_TYPES


@pytest.mark.parametrize('spelling', ['Accel', 'ACC', 'accelerometer',
                                      'Acceleration (g)', 'ACCEL'])
def test_a_spelling_the_schema_does_not_know_falls_to_the_unit(spelling):
    """The lenient coercion blanks such a value; the unit answers for
    it rather than leaving the channel typeless."""
    assert _one(spelling, 'G')[0] == 'acceleration'


def test_a_stated_type_is_never_overruled():
    """A disagreement between the two is the owner's to resolve, and
    `set_cell` refuses to create one — so the file's own word stands."""
    assert _one('Force', 'G')[0] == 'force'
    assert _one('Voltage', 'V')[0] == 'voltage'
    assert _one('Acceleration', 'G')[0] == 'acceleration'


def test_a_unit_that_names_no_channel_type_answers_nothing():
    """Strain and a bare ratio share a dimensionality, which is the
    whole reason they are two names: a unitless unit cannot say which,
    so it says nothing."""
    assert _one('', 'strain')[0] == ''
    assert _one('', '')[0] == ''
    assert _one('', 'Hz')[0] == '', 'a frequency is not a channel type'


def test_a_controllers_run_keeps_the_types_it_states():
    """The fixture states 'Acceleration' and 'Force' and is unaffected."""
    from conftest import fixture_path

    import visualdynamics

    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    table = next(o for o in loaded.values() if isinstance(o, ChannelTable))
    assert set(table['channel_type']) == {'acceleration', 'force'}


def test_the_editable_table_reads_the_same_rows(qt_app):
    """What the window shows is what the object holds."""
    from visualdynamics.gui.object_tables import channel_table_model

    table = ChannelTable({'channel': [1, 2], 'node': ['101', '102'],
                          'direction': ['Z+', 'Z+'],
                          'channel_type': ['Accel', 'Voltage'],
                          'unit': ['G', 'V']})
    model = channel_table_model(table)
    column = list(table.SCHEMA).index('channel_type')
    assert [model.data(model.index(r, column)) for r in range(2)] == [
        'acceleration', 'voltage']
