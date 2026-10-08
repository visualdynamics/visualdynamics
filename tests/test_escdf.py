"""The Engineering Sciences Common Data Format's generic layer
(`visualdynamics.io.escdf`, 2026-09-30; PLAN.md "ESCDF: import and
export first"), held to the reference implementation both ways.

The reference package (BSD-3-Clause, a dev dependency, never imported
by the package itself) is the agreement oracle: a file it writes reads
here with every value intact and no problem found; a file written here
loads there and validates, dataset by dataset. The specification files
it ships are the ones vendored here, and the parser is held to its view
of every type's properties.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from visualdynamics.io import escdf

reference = pytest.importorskip('escdf')


@pytest.fixture(autouse=True)
def _no_prompt(monkeypatch):
    """The reference asks a name at the terminal the first time; a test
    is not the first time."""
    monkeypatch.setattr(reference.ESCDF, 'get_or_prompt_attribution_name',
                        staticmethod(lambda **kwargs: 'test'))


def test_every_vendored_specification_parses_to_the_references_properties():
    specs = escdf.specifications()
    assert len(specs) >= 23
    for name, spec in specs.items():
        theirs = reference.Dataset('x', name, 'x')
        assert {p.name for p in spec.all_properties()} == set(theirs._valid_properties), name
        assert spec.version == tuple(theirs.version_numbers), name


def test_the_grammar_reads_shapes_options_and_choices():
    spec = escdf.parse_specification(
        'thing - v1.2.3\n--------------\nextends: parameter_set\nSome prose.\n\n'
        'properties\n----------\n'
        'a - f8\nb - u8 - num_rows,3 - optional\nc - str - scalar - enum:kinds\n'
        'd - f8 - num_rows - variable_length,optional\n'
        'e - f4 - n - or:e:single\ne - f8 - n - or:e:double\n'
        'stamp - str - scalar - optional, regex:^(\\d{4}),(\\d{2})$\n'
        '\nenumerations\n------------\nkinds - one, two, three\n\nnotes\n-----\nx - y\n')
    assert (spec.name, spec.version, spec.extends) == ('thing', (1, 2, 3), 'parameter_set')
    by = {p.name: p for p in spec.properties}
    assert by['a'].shape == () and not by['a'].optional
    assert by['b'].shape == ('num_rows', 3) and by['b'].optional
    assert by['c'].enum == 'kinds' and spec.enumerations['kinds'] == ['one', 'two', 'three']
    assert by['d'].variable_length and by['d'].optional
    choices = [p.choice for p in spec.properties if p.name == 'e']
    assert choices == [('e', 'single'), ('e', 'double')]
    assert by['stamp'].regex == '^(\\d{4}),(\\d{2})$', 'a comma inside a regex is the regex'
    assert spec.doc == 'Some prose.'


def _reference_file(path):
    """A file the reference writes, one of every kind that maps here."""
    f = reference.ESCDF()
    g = reference.Dataset('geom', 'geometry', 'A geometry')
    g.node_id = np.array([1, 2, 3, 4], dtype=np.uint64)
    g.node_position = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.]])
    for axis, v in (('x', [1, 0, 0]), ('y', [0, 1, 0]), ('z', [0, 0, 1])):
        setattr(g, f'node_{axis}_direction', np.tile(np.array(v, float), (4, 1)))
    g.line_connection = [np.array([1, 2, 3], dtype=np.uint64), np.array([3, 4], dtype=np.uint64)]
    g.element_connection = [np.array([1, 2, 3, 4], dtype=np.uint64),
                            np.array([1, 2, 3], dtype=np.uint64)]
    g.element_type = ['quad4', 'tri3']
    g.element_color = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64)
    g.position_units = 'm'
    g.notes = ['first note', 'second']
    g.attachments = [np.frombuffer(b'hello', dtype='uint8')]
    g.attachment_names = ['hello.txt']
    f.add_metadata(g)
    f.add_activity('act1', 'Modal test', dt.datetime(2026, 9, 30, tzinfo=dt.UTC),
                   metadata_links=['geom'])
    th = reference.Dataset('time', 'data', 'Time histories')
    th.data_type = 'time response'
    th.channel = np.array([['1X+'], ['2Y-']])
    th.ordinate_unit = 'm/s^2'
    th.abscissa_unit = 's'
    th.ordinate = np.arange(16.).reshape(2, 8)
    th.abscissa_start = 0.0
    th.abscissa_step = 0.01
    f.add_data_to_activity('act1', th)
    m = reference.Dataset('modes', 'mode', 'Modes')
    m.frequency = np.array([10., 20.])
    m.shape = np.array([[1, 2], [3, 4], [5, 6.]]) + 1j
    m.dof_name = ['1X+', '2Y-', '3Z+']
    f.add_data_to_activity('act1', m)
    f.write_to_disk(str(path), clobber=True).close()


def test_a_file_the_reference_wrote_reads_whole(tmp_path):
    path = tmp_path / 'ref.escdf'
    _reference_file(path)
    assert escdf.sniff(path)
    file = escdf.read(path)
    assert file.created_by == 'test' and file.created_date.tzinfo is not None
    assert file.problems() == []
    geom = file.metadata['geom']
    assert (geom.kind, geom.descriptive_name, geom.version) == ('geometry', 'A geometry', (0, 1, 0))
    assert geom.values['position_units'] == 'm'
    assert [c.tolist() for c in geom.values['element_connection']] == [[1, 2, 3, 4], [1, 2, 3]]
    assert list(geom.values['element_type']) == ['quad4', 'tri3']
    assert list(geom.values['notes']) == ['first note', 'second']
    assert bytes(geom.values['attachments'][0]) == b'hello'
    act = file.activities['act1']
    assert act.descriptive_name == 'Modal test' and act.links == ['geom']
    assert act.date == dt.datetime(2026, 9, 30, tzinfo=dt.UTC)
    time = act.data['time']
    assert time.values['channel'].tolist() == [['1X+'], ['2Y-']]
    assert time.values['abscissa_step'] == 0.01 and time.values['ordinate_unit'] == 'm/s^2'
    assert act.data['modes'].values['shape'].dtype == np.complex128


def test_a_file_written_here_loads_and_validates_in_the_reference(tmp_path):
    original = tmp_path / 'ref.escdf'
    _reference_file(original)
    file = escdf.read(original)
    ours = tmp_path / 'ours.escdf'
    escdf.write(file, ours)
    loaded = reference.ESCDF.load(str(ours))
    assert sorted(loaded.metadata.names) == ['geom']
    assert loaded.activities.names == ['act1']
    for dataset in list(loaded.metadata) + list(loaded.get_activity_data('act1')):
        assert dataset.validate(), dataset.name
    geom = loaded.metadata['geom']
    assert [c.tolist() for c in geom.element_connection[...]] == [[1, 2, 3, 4], [1, 2, 3]]
    assert bytes(geom.attachments[...][0]) == b'hello'
    assert list(loaded.activities['act1'].metadata_links) == ['geom']
    modes = loaded.get_activity_data('act1', 'modes')
    np.testing.assert_array_equal(modes.shape[...], np.array([[1, 2], [3, 4], [5, 6.]]) + 1j)
    # and back here, byte for byte in what matters
    again = escdf.read(ours)
    assert again.problems() == []
    np.testing.assert_array_equal(again.activities['act1'].data['time'].values['ordinate'],
                                  np.arange(16.).reshape(2, 8))


def test_what_the_specification_does_not_define_survives(tmp_path):
    """A dataset of a type nobody defined, and a property the type does
    not name, are kept as read and written back: the reference does the
    same (it warns and keeps them), so neither side drops the other's
    extras."""
    file = escdf.File(created_by='x')
    mystery = escdf.Dataset('m', 'geometry', 'with extras')
    mystery.values = {'node_id': np.array([1], dtype=np.uint64),
                      'node_position': np.zeros((1, 3)),
                      'node_x_direction': np.array([[1., 0, 0]]),
                      'node_y_direction': np.array([[0, 1., 0]]),
                      'node_z_direction': np.array([[0, 0, 1.]]),
                      'position_units': 'in'}
    mystery.extras = {'group_name': ('str', np.array(['frame'], dtype=object)),
                      'group_id': ('u8', np.array([7], dtype=np.uint64))}
    file.metadata['m'] = mystery
    unknown = escdf.Dataset('u', 'visualdynamics_report', 'a report')
    file.activities['a'] = escdf.Activity('a', 'An activity', links=['m'],
                                          data={'u': unknown})
    unknown.extras = {'html': ('str', '<p>hi</p>')}
    assert file.problems() == []
    unknown.values = {'notes': ['x']}
    assert file.problems() == [
        ("a/u: 'visualdynamics_report' is not a type the specifications define, "
         'so it can carry nothing but extras')]
    unknown.values = {}
    # a known type that is not a result is refused inside an activity, as
    # the reference refuses it
    file.activities['a'].data['p'] = escdf.Dataset('p', 'parameter_set', 'p')
    assert file.problems() == [('a/p: parameter_set is not an activity result; it '
                                'belongs in the metadata, linked')]
    del file.activities['a'].data['p']
    path = tmp_path / 'extras.escdf'
    escdf.write(file, path)
    again = escdf.read(path)
    assert again.metadata['m'].extras['group_name'][1].tolist() == ['frame']
    assert again.activities['a'].data['u'].extras['html'] == ('str', '<p>hi</p>')
    with pytest.warns(UserWarning):
        theirs = reference.ESCDF.load(str(path))
    assert theirs.metadata['m'].group_name[...].tolist() == ['frame']
    report = theirs.get_activity_data('a', 'u')
    assert report.dataset_type == 'unknown' and report.html[...] == '<p>hi</p>'


def test_validation_says_what_the_reference_would_refuse():
    data = escdf.Dataset('d', 'data', 'x')
    data.values = {'data_type': 'time response', 'channel': np.array([['1X+']]),
                   'ordinate_unit': 'g', 'abscissa_unit': 's',
                   'ordinate': np.zeros((1, 4)), 'abscissa_start': 0.0}
    assert data.problems() == ['choice abscissa/even_spacing also needs abscissa_step']
    data.values['abscissa_step'] = 0.5
    assert data.problems() == []
    data.values['abscissa'] = np.zeros(4)
    assert 'choice abscissa: both' in data.problems()[0]
    del data.values['abscissa']
    data.values['ordinate'] = np.zeros((2, 4))
    assert data.problems() == ['ordinate: num_data is 2 here and 1 elsewhere']
    data.values['ordinate'] = np.zeros((1, 4))
    data.values['data_type'] = 'weather'
    assert data.problems()[0].startswith("data_type: 'weather' is not one of")
    data.values['data_type'] = 'time response'
    data.values['channel'] = np.array([['one']])
    assert 'does not match' in data.problems()[0]
    data.values['channel'] = np.array([['1X+']])
    del data.values['ordinate']
    assert 'choice ordinate: none of' in data.problems()[0]
    data.values['ordinate'] = np.array([['a', 'b', 'c', 'd']], dtype=object)
    assert 'does not fit' in data.problems()[0]


def test_names_are_repaired_as_the_reference_repairs_them():
    for raw in ('Plate Modal Test', '3 sensors', 'a/b:c', '', 'ok_name'):
        assert escdf.valid_identifier(raw, 'item_') == \
            reference.make_valid_identifier(raw, 'item_'), raw


def test_a_file_that_is_not_the_format_is_not_sniffed(tmp_path):
    import h5py

    other = tmp_path / 'other.h5'
    with h5py.File(other, 'w') as h5:
        h5.create_group('geometry')
    assert not escdf.sniff(other)
    assert not escdf.sniff(tmp_path / 'missing.h5')
