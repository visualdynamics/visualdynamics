"""Objects are named for what they are, not for the file they arrived in.

An importer's keys are for code — `time_data`, `Modal_frf`, `ChannelTable` —
and three ways of spelling the same idea. The tree shows one way, written for
reading.
"""

from __future__ import annotations

import pytest
from conftest import fixture_path

from visualdynamics.names import display_name


@pytest.mark.parametrize('key,expected', [
    ('time_data', 'Time Data'),
    ('channel_table', 'Channel Table'),
    ('Modal_coherence', 'Modal Coherence'),
    ('Random_specification', 'Random Specification'),
    # class names are the same words with the spaces left out
    ('ShapeSet', 'Shape Set'),
    ('TimeHistory', 'Time History'),
    ('MultipleCoherence', 'Multiple Coherence'),
    ('Geometry', 'Geometry'),
    # already a name: never reformatted underneath a user's own rename
    ('Modal FRF', 'Modal FRF'),
])
def test_a_key_reads_as_a_name(key, expected):
    assert display_name(key) == expected


@pytest.mark.parametrize('key,expected', [
    ('Frf', 'FRF'),
    ('Modal_frf', 'Modal FRF'),
    ('Random_drive_cpsd', 'Random Drive CPSD'),
    ('Random_response_cpsd', 'Random Response CPSD'),
    ('Psd', 'PSD'),
])
def test_the_acronyms_this_field_uses_stay_acronyms(key, expected):
    """'Frf' is not a word and reads as one. Capitalizing the first letter of
    each word is the rule; these are the exceptions."""
    assert display_name(key) == expected


def test_an_import_is_named_for_its_types(window):
    """One vocabulary. The reader's own keys — 'Modal_frf', 'time_data' —
    invented names that exist nowhere else in the model, and qualified them
    with an environment name that came from the file."""
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4')])
    assert set(window.objects) == {'Channel Table', 'Time History',
                                   'FRF', 'Multiple Coherence'}


def test_two_of_a_type_from_one_file_are_numbered(window):
    """A random spectral save holds two PSDs: the response CPSD and the drive
    CPSD. Both are a PSD, so the second is '(2)'."""
    window.import_paths([fixture_path('plate', 'random_spectra.nc4')])
    assert 'PSD' in window.objects and 'PSD (2)' in window.objects
    from visualdynamics.core.data import Psd
    assert all(isinstance(window.objects[name], Psd)
               for name in ('PSD', 'PSD (2)'))


def test_what_the_reader_called_it_is_on_the_tooltip(window):
    """The type cannot say which PSD is which; the reader's key can, so it is
    kept where it does not have to be in every name."""
    window.import_paths([fixture_path('plate', 'random_spectra.nc4')])
    first = window._item_for_object('PSD').toolTip(0)
    second = window._item_for_object('PSD (2)').toolTip(0)
    assert 'Random Response CPSD' in first
    assert 'Random Drive CPSD' in second


def test_a_name_that_already_matches_its_type_is_not_repeated(window):
    """'channel_table' formats to exactly 'Channel Table', so saying it twice
    on the tooltip would be noise."""
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4')])
    tooltip = window._item_for_object('Channel Table').toolTip(0)
    assert tooltip.count('Channel Table') == 0, tooltip


def test_a_single_object_import_is_named_for_its_type(window):
    window.import_paths([fixture_path('plate', 'geometry.npz')])
    assert list(window.objects) == ['Geometry']


def test_a_second_import_of_the_same_kind_is_numbered(window):
    """The file name used to keep these apart, at the cost of being in front
    of every name."""
    window.import_paths([fixture_path('plate', 'geometry.npz'),
                         fixture_path('plate', 'geometry.npz')])
    assert list(window.objects) == ['Geometry', 'Geometry (2)']


def test_the_file_it_came_from_is_on_the_tooltip(window):
    """Taking the file out of the name must not lose it."""
    path = fixture_path('plate', 'geometry.npz')
    window.import_paths([path])
    item = window._item_for_object('Geometry')
    assert path in item.toolTip(0)
    assert repr(window.objects['Geometry']) in item.toolTip(0)


def test_the_source_follows_a_rename(window):
    path = fixture_path('plate', 'geometry.npz')
    window.import_paths([path])
    item = window._item_for_object('Geometry')
    item.setText(0, 'wing')
    window._item_renamed(item, 0)
    assert window.object_sources['wing'] == path
    assert path in item.toolTip(0)


def test_deleting_an_object_forgets_where_it_came_from(window):
    window.import_paths([fixture_path('plate', 'geometry.npz')])
    window.tree.setCurrentItem(window._item_for_object('Geometry'))
    window.delete_selected()
    assert 'Geometry' not in window.object_sources


def test_a_name_already_in_use_is_refused_and_put_back(window):
    """Two objects with one name is a project that cannot be saved or
    linked, so the rename is refused at entry — the row goes back to what
    it said, with the reason in the status bar."""
    window.import_paths([fixture_path('plate', 'geometry.npz'),
                         fixture_path('plate', 'geometry.npz')])
    first = window._item_for_object('Geometry')
    first.setText(0, 'Geometry (2)')
    window._item_renamed(first, 0)
    assert first.text(0) == 'Geometry', 'the row says what it said'
    assert set(window.objects) >= {'Geometry', 'Geometry (2)'}
    assert 'already in use' in window.statusBar().currentMessage()


def test_a_name_cannot_be_emptied(window):
    window.import_paths([fixture_path('plate', 'geometry.npz')])
    item = window._item_for_object('Geometry')
    item.setText(0, '   ')
    window._item_renamed(item, 0)
    assert item.text(0) == 'Geometry'
    assert 'cannot be empty' in window.statusBar().currentMessage()
