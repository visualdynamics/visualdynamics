"""Data index (report feature `data_index`, `core.data_index`).

The checksum against hand values, the index's entries and their
refusals, the check against the files, what an import records and how
the entries follow the objects, persistence in the project file, the
page with the feature on (alone and under the skeleton) and off, the
attachments printed, and the app's import, Attached File and Export.
"""

from __future__ import annotations

import hashlib
import html
import json
import os

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics import io
from visualdynamics.core import data_index as di
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.report import Report
from visualdynamics.project import Project, holds
from visualdynamics.report import data_index as rdi
from visualdynamics.report import features as rf
from visualdynamics.report import page, render_html
from visualdynamics.report import skeleton as sk

#: SHA-256 of b'abc' and of no bytes at all (FIPS 180-2's examples)
ABC = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
NOTHING = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


def _payload(html):
    data = html.split('<script id="data" type="application/json">')[1]
    return json.loads(data.split('</script>')[0].replace('<\\/', '</'))


def _file(folder, name, data=b'abc'):
    path = folder / name
    path.write_bytes(data)
    return str(path)


def _history(values=None, rate=1024.0):
    rows = (np.vstack([np.sin(np.arange(2048) / 7.0),
                       np.cos(np.arange(2048) / 5.0)])
            if values is None else np.atleast_2d(values))
    return TimeHistory(np.arange(rows.shape[1]) / rate, rows,
                       response_dof=[f'{k + 1}X+' for k in range(len(rows))],
                       ordinate_dim=['acceleration'] * len(rows),
                       ordinate_unit=['g'] * len(rows))


def _imported(tmp_path, values=None, name='run1.vdyn'):
    """A project with one time history imported from a file of its own."""
    path = str(tmp_path / name)
    _history(values).save(path)
    project = Project('P')
    project.import_file(path)
    return project, path


def _render(project, name, **kwargs):
    return render_html(project[name], project,
                       unit_system=visualdynamics.SI,
                       object_groups=project.object_groups, **kwargs)


# ---- the checksum -------------------------------------------------------------


def test_the_checksum_is_sha256_against_hand_values(tmp_path, monkeypatch):
    assert di.digest(_file(tmp_path, 'a.txt')) == (3, ABC)
    assert di.digest(_file(tmp_path, 'empty.txt', b'')) == (0, NOTHING)
    # a file read in many chunks hashes as one read, and progress walks
    # from nothing to the whole in order
    data = bytes(range(256)) * 41
    monkeypatch.setattr(di, 'CHUNK', 1000)
    seen = []
    size, sha = di.digest(_file(tmp_path, 'b.bin', data),
                          lambda done, total: seen.append((done, total)))
    assert (size, sha) == (len(data), hashlib.sha256(data).hexdigest())
    assert seen[0] == (0, len(data)) and seen[-1] == (len(data), len(data))
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)
    assert len(seen) == 1 + -(-len(data) // 1000)


def test_the_checksum_refuses_what_is_not_a_file(tmp_path):
    with pytest.raises(FileNotFoundError, match='no file at'):
        di.digest(str(tmp_path / 'gone.csv'))
    with pytest.raises(IsADirectoryError, match='is a folder'):
        di.digest(str(tmp_path))
    with pytest.raises(TypeError, match='not int'):
        di.digest(3)


# ---- an entry -------------------------------------------------------------------


def _entry(**change):
    fields = {'name': 'a.csv', 'path': '/x/a.csv', 'size': 3,
              'sha256': ABC, 'holds': 'settings list'}
    fields.update(change)
    return di.IndexEntry(**fields)


@pytest.mark.parametrize(('change', 'match'), [
    ({'holds': '  '}, 'say what the file holds'),
    ({'name': ''}, "needs the file's name"),
    ({'source': ''}, 'say what added'),
    ({'role': 'enclosed'}, "role 'enclosed' is not one of"),
    ({'size': -1}, 'whole number of bytes'),
    ({'size': True}, 'whole number of bytes'),
    ({'sha256': ABC.upper()}, 'not a SHA-256'),
    ({'sha256': ABC[:-1]}, 'not a SHA-256'),
    ({'status': 'fine'}, "status 'fine' is not one of"),
])
def test_an_entry_refuses_what_it_cannot_stand_behind(change, match):
    with pytest.raises(ValueError, match=match):
        _entry(**change)


def test_an_entry_round_trips_and_reads_a_later_versions_record():
    entry = _entry(objects=['Time History'], status=di.MATCHES,
                   checked='2026-10-24T14:02:00+00:00')
    record = entry.record()
    assert 'text' not in record and 'found' not in record, 'empty is left out'
    assert record['size'] == 3 and record['objects'] == ['Time History']
    assert di.IndexEntry.from_record(record) == entry
    assert di.IndexEntry.from_record({**record, 'later': 1}) == entry
    with pytest.raises(ValueError, match='missing sha256, holds'):
        di.IndexEntry.from_record({'name': 'a', 'path': 'a', 'size': 1})
    with pytest.raises(TypeError, match='not list'):
        di.IndexEntry.from_record([])
    assert _entry(size=0).record()['size'] == 0, 'an empty file says 0 bytes'


# ---- the index ------------------------------------------------------------------


def test_add_hashes_and_one_file_is_one_entry(tmp_path):
    index = di.DataIndex()
    path = _file(tmp_path, 'run.csv')
    first = index.add(path, 'run 1', format='CSV', objects=['A'])
    assert (first.name, first.size, first.sha256) == ('run.csv', 3, ABC)
    assert first.path == os.path.abspath(path) and first.indexed
    again = index.add(path, 'run 1, again', objects=['B', 'A'])
    assert len(index) == 1 and again.objects == ('A', 'B')
    assert again.holds == 'run 1', 'the first description stands'
    _file(tmp_path, 'run.csv', b'abd')
    index.add(path, 'run 1, rewritten')
    assert len(index) == 2, 'the same path with new contents is a new file'
    assert index[1].sha256 == hashlib.sha256(b'abd').hexdigest()


def test_add_refuses_before_it_reads(tmp_path, monkeypatch):
    index = di.DataIndex()
    path = _file(tmp_path, 'run.csv')

    def never(*_args, **_kwargs):
        raise AssertionError('a refused file is not read')

    monkeypatch.setattr(di, 'digest', never)
    with pytest.raises(ValueError, match='say what the file holds'):
        index.add(path, '')
    with pytest.raises(ValueError, match="role 'kept' is not one of"):
        index.add(path, 'run', role='kept')
    with pytest.raises(TypeError, match='holds is a str'):
        index.add(path, None)
    with pytest.raises(FileNotFoundError, match='no file at'):
        index.add(str(tmp_path / 'gone.csv'), 'run')
    with pytest.raises(IsADirectoryError, match='is a folder'):
        index.add(str(tmp_path), 'run')
    with pytest.raises(FileNotFoundError, match='no file at'):
        index.add(str(tmp_path / 'gone.csv'), 'run', checksum=(3, ABC))
    assert len(index) == 0
    # a checksum handed over (from a worker) is used, not recomputed
    made = index.add(path, 'run', checksum=(3, ABC))
    assert made.sha256 == ABC


def test_an_attached_text_file_keeps_its_words(tmp_path, monkeypatch):
    index = di.DataIndex()
    text = 'Setting,Value\nWindow,Hann\n'
    kept = index.add(_file(tmp_path, 'settings.csv', text.encode()),
                     'settings list', role='attached')
    assert kept.text == text
    referenced = index.add(_file(tmp_path, 'log.txt', b'log'), 'log')
    assert referenced.text == '', 'a referenced file is named, not carried'
    binary = index.add(_file(tmp_path, 'scan.pdf', b'%PDF\x00\x01'), 'scan',
                       role='attached')
    assert binary.text == '', 'not text: it travels, unprinted'
    latin = index.add(_file(tmp_path, 'old.txt', 'caf\xe9'.encode('latin-1')),
                      'old', role='attached')
    assert latin.text == ''
    # 64 KiB is kept, a byte more is not
    edge = b'x' * (64 * 1024)
    assert index.add(_file(tmp_path, 'edge.txt', edge), 'edge',
                     role='attached').text == edge.decode()
    assert index.add(_file(tmp_path, 'over.txt', edge + b'x'), 'over',
                     role='attached').text == ''
    monkeypatch.setattr(di, 'TEXT_LIMIT', 4)
    large = index.add(_file(tmp_path, 'big.txt', b'12345'), 'big',
                      role='attached')
    assert large.text == ''
    # the words kept are the bytes hashed, or nothing is kept
    path = _file(tmp_path, 'moved.txt', b'abcd')
    with pytest.raises(ValueError, match='changed while it was being indexed'):
        index.add(path, 'moved', role='attached', checksum=(3, ABC))


def test_verify_finds_matches_changes_and_absences(tmp_path):
    index = di.DataIndex()
    same = _file(tmp_path, 'same.csv')
    edited = _file(tmp_path, 'edited.csv')
    gone = _file(tmp_path, 'gone.csv')
    for path in (same, edited, gone):
        index.add(path, 'run')
    assert all(entry.status == '' for entry in index), 'not checked yet'
    _file(tmp_path, 'edited.csv', b'abcd')
    os.remove(gone)
    seen = []
    after = index.verify(progress=lambda done, total: seen.append((done,
                                                                   total)))
    assert [entry.status for entry in after] == [di.MATCHES, di.CHANGED,
                                                 di.NOT_FOUND]
    assert after == tuple(index) and all(entry.checked for entry in after)
    assert len({entry.checked for entry in after}) == 1, 'one check, one time'
    # bytes across the index: 3 and 3 as indexed, the edited file read
    # at its new 4, the one not found counted without a read
    assert seen == [(0, 9), (3, 9), (3, 9), (7, 9)]
    # an index of one empty file still reports a whole to count against
    empty = di.DataIndex()
    empty.add(_file(tmp_path, 'empty.csv', b''), 'nothing yet')
    seen.clear()
    empty.verify(progress=lambda done, total: seen.append((done, total)))
    assert seen == [(0, 1)] and empty[0].status == di.MATCHES


def test_verify_looks_in_the_folders_it_is_given(tmp_path):
    index = di.DataIndex()
    path = _file(tmp_path, 'run.csv')
    index.add(path, 'run')
    archive = tmp_path / 'archive'
    archive.mkdir()
    os.replace(path, archive / 'run.csv')
    assert index.verify()[0].status == di.NOT_FOUND
    found = index.verify([str(archive)])[0]
    assert found.status == di.MATCHES
    assert found.found == str(archive / 'run.csv')
    (archive / 'run.csv').write_bytes(b'xyz')
    assert index.verify([str(archive)])[0].status == di.CHANGED


def test_a_check_stopped_part_way_changes_nothing(tmp_path):
    index = di.DataIndex()
    for name in ('a.csv', 'b.csv'):
        index.add(_file(tmp_path, name), 'run')
    before = list(index)

    class Stop(Exception):
        pass

    def stop(done, _total):
        if done:
            raise Stop

    with pytest.raises(Stop):
        index.verify(progress=stop)
    assert list(index) == before
    assert di.DataIndex().verify() == ()


def test_entries_follow_the_objects(tmp_path):
    index = di.DataIndex()
    index.add(_file(tmp_path, 'a.csv'), 'a', objects=['A', 'B'])
    index.add(_file(tmp_path, 'b.csv', b'b'), 'b', objects=['C'])
    index.add(_file(tmp_path, 'c.csv', b'c'), 'c', source='author')
    index.rename_object('A', 'A2')
    assert index[0].objects == ('A2', 'B')
    assert index.forget_objects(['B']) == [] and index[0].objects == ('A2',)
    dropped = index.forget_objects(['C'])
    assert [entry.name for entry in dropped] == ['b.csv']
    kept = index.forget_objects(['A2'], keep=lambda _entry: True)
    assert kept == [] and index[0].objects == ()
    assert [entry.name for entry in index] == ['a.csv', 'c.csv']


def test_remove_by_name_or_path(tmp_path):
    index = di.DataIndex()
    first = index.add(_file(tmp_path, 'a.csv'), 'a')
    (tmp_path / 'other').mkdir()
    index.add(_file(tmp_path / 'other', 'a.csv', b'z'), 'a too')
    with pytest.raises(ValueError, match="2 entries are named 'a.csv'"):
        index.remove('a.csv')
    with pytest.raises(ValueError, match="no file named 'b.csv'"):
        index.remove('b.csv')
    assert index.remove(first.path) == first and len(index) == 1


def test_adopt_renames_through_the_mapping_and_merges(tmp_path):
    mine, theirs = di.DataIndex(), di.DataIndex()
    shared = _file(tmp_path, 'a.csv')
    mine.add(shared, 'a', objects=['X'])
    theirs.add(shared, 'a', objects=['T'])
    theirs.add(_file(tmp_path, 'b.csv', b'b'), 'b', objects=['T', 'U'])
    mine.adopt(theirs, {'T': 'T (2)', 'U': 'U'})
    assert [entry.objects for entry in mine] == [('X', 'T (2)'),
                                                 ('T (2)', 'U')]
    mine.adopt(None)
    assert len(mine) == 2
    plain = di.DataIndex()
    plain.adopt(theirs)
    assert [entry.objects for entry in plain] == [('T',), ('T', 'U')], \
        'without a mapping the names stand'


def test_an_index_holds_entries_only():
    with pytest.raises(TypeError, match='holds IndexEntry, not dict'):
        di.DataIndex([{'name': 'a'}])


# ---- formats and holdings ---------------------------------------------------------


def test_the_format_is_named_from_the_file(tmp_path, monkeypatch):
    assert io.format_of(fixture_path('plate', 'random.nc4')) == \
        'Rattlesnake controller output (.nc4)'
    assert io.format_of(str(tmp_path / 'p.vdyn')) == \
        'Visual Dynamics file (.vdyn)'
    assert io.format_of(_file(tmp_path, 'Settings.CSV')) == \
        'comma-separated text (.csv)'
    assert io.format_of(_file(tmp_path, 'x.qqq')) == 'QQQ file (.qqq)'
    assert io.format_of(_file(tmp_path, 'bare')) == 'file'

    def trips(_path):
        raise RuntimeError('a reader that cannot read this')

    monkeypatch.setattr(io, '_IMPORTERS', [io.Importer(
        'tripping', 'never', trips, trips)] + io.importers())
    assert io.format_of(fixture_path('plate', 'random.nc4')) == \
        'Rattlesnake controller output (.nc4)', 'a sniffer that trips says no'


def test_holdings_name_the_channels_and_rate():
    said = holds({'run': _history()})
    assert said == ('Time History: 2 records, 2048 samples, channels 1X+ to '
                    '2X+, 1024 Hz')
    assert holds({'one': _history(np.zeros(16))}).endswith(
        'channel 1X+, 1024 Hz')
    uneven = _history(np.zeros(4))
    uneven.abscissa = np.array([0.0, 0.1, 0.3, 0.35])
    assert holds({'u': uneven}).endswith('samples not evenly spaced')
    assert holds({}) == ''
    assert holds({'r': Report('R')}) == 'Report', 'nothing to count'


# ---- what an import records --------------------------------------------------------


def test_an_import_records_the_file_it_read(tmp_path):
    project, path = _imported(tmp_path)
    (entry,) = project.data_index
    with open(path, 'rb') as handle:
        data = handle.read()
    assert (entry.name, entry.size) == ('run1.vdyn', len(data))
    assert entry.sha256 == hashlib.sha256(data).hexdigest()
    assert entry.objects == ('Time History',)
    assert entry.format == 'Visual Dynamics file (.vdyn)'
    assert entry.holds.startswith('Time History: 2 records, 2048 samples')
    assert (entry.role, entry.source) == ('referenced', 'import')
    assert project.journal[-1] == f'project.import_file({path!r})'


def test_dead_nan_and_empty_inputs(tmp_path):
    """A file of zeros or with one NaN is still a file: indexed with what
    it holds. An entry that says nothing is refused, naming the cause."""
    dead, _ = _imported(tmp_path, np.zeros((2, 64)), 'dead.vdyn')
    assert dead.data_index[0].holds == ('Time History: 2 records, 64 '
                                        'samples, channels 1X+ to 2X+, '
                                        '1024 Hz')
    constant, _ = _imported(tmp_path, np.full(64, 3.0), 'constant.vdyn')
    assert constant.data_index[0].holds == ('Time History: 1 record, 64 '
                                            'samples, channel 1X+, 1024 Hz')
    spiked = np.ones(64)
    spiked[10] = np.nan
    nan, _ = _imported(tmp_path, spiked, 'nan.vdyn')
    assert nan.data_index[0].holds.startswith('Time History: 1 record')
    with pytest.raises(ValueError, match='say what the file holds: give '
                                         'holds, or the objects'):
        nan.index_file(_file(tmp_path, 'log.txt'))
    assert len(nan.data_index) == 1


def test_an_import_that_cannot_be_hashed_adds_nothing(tmp_path, monkeypatch):
    import visualdynamics.project as module

    def gone(_path):
        raise FileNotFoundError('no file at it any more')

    monkeypatch.setattr(module, 'digest_file', gone)
    path = str(tmp_path / 'run1.vdyn')
    _history().save(path)
    project = Project('P')
    with pytest.raises(FileNotFoundError, match='any more'):
        project.import_file(path)
    assert len(project) == 0 and len(project.data_index) == 0


def test_a_project_file_brings_its_index(tmp_path):
    first, data_path = _imported(tmp_path)
    saved = str(tmp_path / 'first.vdyn')
    first.save(saved)
    opened = Project('Empty')
    opened.import_file(saved)
    assert [entry.name for entry in opened.data_index] == ['run1.vdyn'], \
        'opening a project is not importing a file it rests on'
    second, _ = _imported(tmp_path, name='run2.vdyn')
    second.import_file(saved)
    assert [entry.name for entry in second.data_index] == [
        'run2.vdyn', 'run1.vdyn', 'first.vdyn']
    # the clash renamed the imported time history, and its entry follows
    assert second.data_index[1].objects == ('Time History (2)',)
    assert second.data_index[2].holds == 'Visual Dynamics project: 1 objects'
    assert data_path == second.data_index[1].path


def test_rename_and_remove_follow(tmp_path):
    project, _path = _imported(tmp_path)
    project.rename('Time History', 'Run 1')
    assert project.data_index[0].objects == ('Run 1',)
    psd = project.compute_psds('Run 1')
    project.remove('Run 1')
    assert len(project.data_index) == 1, f'{psd} still rests on the file'
    assert project.data_index[0].objects == ()
    project.remove(psd)
    assert len(project.data_index) == 1, 'nothing named it any more'
    other, _ = _imported(tmp_path, name='run3.vdyn')
    other.remove('Time History')
    assert len(other.data_index) == 0, 'the last object went, and the file'


def test_a_merge_rests_on_every_file_of_its_parts(tmp_path):
    project = Project('P')
    paths = []
    for k in (1, 2):
        history = _history()
        history.response_dof = [f'{k}{axis}+' for axis in 'XY']
        paths.append(str(tmp_path / f'part{k}.vdyn'))
        history.save(paths[-1])
        project.import_file(paths[-1])
    merged = project.merge('Time History', 'Time History (2)', name='Run')
    assert [(entry.name, entry.objects) for entry in project.data_index] == [
        ('part1.vdyn', (merged,)), ('part2.vdyn', (merged,))]


def test_index_file_and_unindex_file_are_journaled_and_refuse(tmp_path):
    project, _path = _imported(tmp_path)
    path = _file(tmp_path, 'settings.csv', b'a,b\n1,2\n')
    entry = project.index_file(path, 'settings list', role='attached')
    assert (entry.source, entry.role, entry.text) == ('author', 'attached',
                                                      'a,b\n1,2\n')
    assert project.journal[-1] == (f'project.index_file({path!r}, '
                                   "'settings list', role='attached')")
    by_object = project.index_file(_file(tmp_path, 'copy.bin', b'q'),
                                   objects=['Time History'])
    assert by_object.holds.startswith('Time History: 2 records')
    before = list(project.journal)
    with pytest.raises(ValueError, match="role 'sent' is not one of"):
        project.index_file(path, 'settings list', role='sent')
    with pytest.raises(KeyError):
        project.index_file(path, objects=['Nothing'])
    assert project.journal == before, 'a refusal is not journaled'
    project.unindex_file('settings.csv')
    assert project.journal[-1] == "project.unindex_file('settings.csv')"
    with pytest.raises(ValueError, match='no file named'):
        project.unindex_file('settings.csv')


def test_verify_data_index_reports_and_states_the_project_file(tmp_path):
    project, path = _imported(tmp_path)
    assert project.verify_data_index() == ['run1.vdyn: matches']
    assert project.journal[-1] == 'project.verify_data_index()'
    assert project.data_index.project_file is None, 'not saved yet'
    saved = str(tmp_path / 'p')
    project.save(saved)
    assert project.file == str(tmp_path / 'p.vdyn')
    # moved beside the project file: found by its name there
    os.replace(path, tmp_path / 'moved.vdyn')
    os.makedirs(tmp_path / 'kept')
    os.replace(tmp_path / 'moved.vdyn', tmp_path / 'kept' / 'run1.vdyn')
    assert project.verify_data_index() == ['run1.vdyn: not found']
    assert project.verify_data_index(search=[str(tmp_path / 'kept')]) == [
        'run1.vdyn: matches']
    name, size, sha, _when = project.data_index.project_file
    with open(project.file, 'rb') as handle:
        data = handle.read()
    assert (name, size, sha) == ('p.vdyn', len(data),
                                 hashlib.sha256(data).hexdigest())
    assert Project.open(project.file).file == project.file
    # moved beside the project file: found there with no folder named
    os.replace(tmp_path / 'kept' / 'run1.vdyn', tmp_path / 'run1.vdyn')
    project.data_index.entries[0] = di.IndexEntry.from_record(
        {**project.data_index[0].record(),
         'path': str(tmp_path / 'gone' / 'run1.vdyn')})
    assert project.verify_data_index() == ['run1.vdyn: matches']
    assert project.data_index[0].found == str(tmp_path / 'run1.vdyn')


# ---- persistence --------------------------------------------------------------------


def test_the_index_rides_in_the_project_file(tmp_path):
    import h5py

    project, _path = _imported(tmp_path)
    project.index_file(_file(tmp_path, 'settings.csv', b'a,b\n'),
                       'settings list', role='attached')
    project.verify_data_index()
    saved = str(tmp_path / 'p.vdyn')
    project.save(saved)
    back = Project.open(saved)
    assert list(back.data_index) == list(project.data_index)
    assert back.data_index[1].text == 'a,b\n'
    empty = Project('E')
    empty.save(str(tmp_path / 'e.vdyn'))
    with h5py.File(tmp_path / 'e.vdyn') as handle:
        assert 'data_index' not in handle, 'none: stored as it always was'
    assert len(Project.open(str(tmp_path / 'e.vdyn')).data_index) == 0


# ---- the page -----------------------------------------------------------------------


@pytest.fixture(scope='module')
def run():
    """A random run worked up, with its report and its file indexed."""
    project = visualdynamics.random_vibration_run(
        fixture_path('plate', 'random.nc4'))
    name = project.generate_report('random')
    return project, name


def test_off_the_page_is_the_page_it_was_whatever_is_indexed(run, tmp_path):
    project, name = run
    report = project[name]
    assert len(project.data_index) == 1, 'the run is indexed on import'
    plain = _render(project, name)
    held = project.data_index
    try:
        project.data_index = di.DataIndex()
        assert _render(project, name) == plain, 'the index is not read'
        project.data_index = held
        report.features = ['data_index']
        assert _render(project, name) != plain
    finally:
        project.data_index = held
        report.features = []
    assert _render(project, name) == plain
    assert 'mark-hash' not in plain


def test_on_the_index_closes_the_report(run):
    project, name = run
    report = project[name]
    try:
        project.verify_data_index()
        report.features = ['data_index']
        labels = []
        data = _payload(_render(project, name, labels=labels))
    finally:
        report.features = []
    blocks = data['blocks']
    heading = next(at for at, block in enumerate(blocks)
                   if block.get('html') == '<h2>Record and Data Package</h2>')
    table, check, analysis, recipe = blocks[heading + 1:heading + 5]
    assert heading + 5 == len(blocks), 'the index closes the report'
    entry = project.data_index[0]
    assert table['headers'] == list(rdi.HEADERS)
    assert table['rows'] == [['random.nc4', entry.holds,
                              'Rattlesnake controller output (.nc4)',
                              '17,735,629', entry.sha256, 'referenced',
                              'matches']]
    assert table['marks'] == [['', '', '', '', 'hash', '', '']]
    assert table['caption'] == rdi.CAPTION
    assert check['html'].startswith('<p class="index-check">Checked ')
    assert check['html'].endswith('1 of 1 file matches the checksum '
                                  'indexed.</p>')
    assert analysis['html'] == (
        f'<p class="index-analysis">Analysis: Visual Dynamics '
        f'{visualdynamics.__version__}. The project has not been saved to '
        'a file, so the analysis is not yet part of the record.</p>')
    assert recipe['rows'][0] == ['Time History PSDs', 'compute_psds',
                                 'Time History']
    tables = [label for label, _caption in labels if label.startswith('Table')]
    assert table['label'] == tables[-2] and recipe['label'] == tables[-1]
    assert int(tables[-1].split()[1]) == len(tables), 'numbered in order'
    css, _js = page.page_parts(('data_index',))
    assert 'td.mark-hash' in css and css.startswith(page._CSS)


def test_under_the_skeleton_the_index_is_in_the_record_section(run):
    project, name = run
    report = project[name]
    blocks = [dict(block) for block in report.blocks]
    try:
        project.report_features(name, enable=['skeleton_sections',
                                              'data_index'])
        page_blocks = _payload(_render(project, name))['blocks']
        heads = [at for at, block in enumerate(page_blocks)
                 if block.get('html', '').startswith('<h2 class="skeleton">')]
        record = heads[-1]
        assert page_blocks[record]['html'].startswith(
            '<h2 class="skeleton">12. Record and Data Package</h2>'
            '<div class="owed">'), 'still the author\'s to write'
        assert page_blocks[record + 1]['kind'] == 'table'
        assert page_blocks[record + 1]['caption'] == rdi.CAPTION
        assert page_blocks[-1]['html'] == '<p class="endmark">End of report.</p>'
        assert not any(block.get('html') == '<h2>Record and Data Package</h2>'
                       for block in page_blocks), 'the skeleton heads it'
        owed = project.report_check(name)
        assert owed[-1] == '12. Record and Data Package: not yet written'
        project.report_section(name, 'record',
                               'Archive: the lab, test 37; ask the engineer.')
        page_blocks = _payload(_render(project, name))['blocks']
        record = next(at for at, block in enumerate(page_blocks)
                      if block.get('html', '').startswith(
                          '<h2 class="skeleton">12.'))
        assert 'Archive: the lab' in page_blocks[record]['html']
        assert page_blocks[record + 1]['caption'] == rdi.CAPTION, \
            "the index follows the author's words"
        assert not any(line.startswith('12.')
                       for line in project.report_check(name))
    finally:
        report.blocks = blocks
        report.features = []


def test_under_refuses_what_is_not_on_the_page():
    with pytest.raises(ValueError, match='unknown skeleton section'):
        sk.under([], 'archive', [])
    with pytest.raises(ValueError, match='is not on this page'):
        sk.under([{'kind': 'text'}], 'record', [])
    blocks = [{'skeleton': {'key': 'record'}}, {'kind': 'plot'},
              {'skeleton': {'key': 'end'}}]
    assert sk.under(blocks, 'record', [{'x': 1}]) == [
        blocks[0], blocks[1], {'x': 1}, blocks[2]]


def _project_with_report(tmp_path):
    project, _path = _imported(tmp_path)
    name = project.add('Report', Report('Report', [
        {'kind': 'text', 'text': '## Results\n\nThe run.'}],
        features=['data_index']))
    return project, name


def test_an_empty_index_says_so_and_is_never_a_full_one(tmp_path):
    project = Project('P')
    name = project.add('Report', Report('Report', [], features=['data_index']))
    data = _payload(_render(project, name))
    assert [block['kind'] for block in data['blocks']] == ['text'] * 3
    assert data['blocks'][1]['html'] == (f'<p class="index-empty">'
                                         f'{rdi.EMPTY}</p>')


def test_a_page_drawn_without_the_project_says_it_has_no_index(tmp_path):
    project, name = _project_with_report(tmp_path)
    data = _payload(render_html(project[name], dict(project.items()),
                                unit_system=visualdynamics.SI))
    assert data['blocks'][-1]['html'] == (
        f'<p class="index-empty">{html.escape(rdi.UNKNOWN)}</p>')
    assert not any(block['kind'] == 'table' for block in data['blocks'])


def test_the_project_file_export_hands_the_project_to_the_page(tmp_path,
                                                              monkeypatch):
    import visualdynamics.report as report_module

    project, _name = _project_with_report(tmp_path)
    seen = []
    real = report_module.render_html
    monkeypatch.setattr(report_module, 'render_html', lambda report, objects,
                        *args, **kwargs: seen.append(objects)
                        or real(report, objects, *args, **kwargs))
    project.save(str(tmp_path / 'p.h5'))
    assert seen and seen[0] is project


def test_nothing_is_hashed_while_a_page_is_drawn(tmp_path, monkeypatch):
    project, name = _project_with_report(tmp_path)

    def never(*_args, **_kwargs):
        raise AssertionError('hashed while drawing')

    monkeypatch.setattr(di, 'digest', never)
    monkeypatch.setattr(di.hashlib, 'sha256', never)
    assert 'mark-hash' in _render(project, name)


def test_what_the_check_found_is_in_capitals_and_counted(tmp_path):
    project, name = _project_with_report(tmp_path)
    project.index_file(_file(tmp_path, 'log.txt'), 'log')
    project.index_file(_file(tmp_path, 'gone.txt', b'g'), 'gone')
    unchecked = _payload(_render(project, name))['blocks']
    table = next(block for block in unchecked if block['kind'] == 'table')
    assert [row[-1] for row in table['rows']] == ['not checked'] * 3
    assert [marks[-1] for marks in table['marks']] == ['unchecked'] * 3
    assert rdi.checked(project.data_index).startswith('Not checked: ')
    _file(tmp_path, 'log.txt', b'edited')
    os.remove(tmp_path / 'gone.txt')
    project.verify_data_index()
    project.index_file(_file(tmp_path, 'late.txt', b'l'), 'late')
    table = next(block for block in _payload(_render(project, name))['blocks']
                 if block['kind'] == 'table')
    assert [row[-1] for row in table['rows']] == [
        'matches', 'CHANGED', 'NOT FOUND', 'not checked']
    assert [marks[-1] for marks in table['marks']] == [
        '', 'changed', 'changed', 'unchecked']
    said = rdi.checked(project.data_index)
    assert said.endswith(': 1 of 4 files match the checksum indexed; 1 '
                         'changed; 1 not found; 1 indexed since, not '
                         'checked.')
    assert said.startswith('Checked 20') and ' UTC: ' in said


def test_an_attachment_is_printed_whole(tmp_path):
    project, name = _project_with_report(tmp_path)
    project.index_file(_file(tmp_path, 'Y_run11_settings.csv',
                             b'Setting,Value\nWindow,Hann,extra\nLines\n'),
                       'settings list', role='attached')
    project.index_file(_file(tmp_path, 'notes.txt',
                             b'<script>alert(1)</script> & so on'),
                       'console notes', role='attached')
    project.index_file(_file(tmp_path, 'kept.txt', b'kept'), 'kept')
    project.index_file(_file(tmp_path, 'ragged.csv', b'a,b\n1\n'),
                       'short row', role='attached')
    project.index_file(_file(tmp_path, 'levels.tsv',
                             b'Run\tLevel\n1\t-6 dB\n'),
                       'levels', role='attached')
    blocks = _payload(_render(project, name))['blocks']
    settings = next(block for block in blocks if 'Y_run11_settings.csv'
                    in block.get('html', ''))
    assert '<pre class="attachment">Setting,Value\nWindow,Hann,extra\n' \
        in settings['html'], 'ragged: printed as written, not guessed'
    ragged = next(block for block in blocks if 'ragged.csv, short row'
                  in block.get('html', ''))
    assert '<pre class="attachment">a,b\n1\n</pre>' in ragged['html']
    levels = next(block for block in blocks if block.get('caption') ==
                  'Attachment: levels.tsv, levels.')
    assert levels['headers'] == ['Run', 'Level']
    assert levels['rows'] == [['1', '-6 dB']]
    notes = next(block for block in blocks
                 if 'class="attachment">Attachment: notes.txt' in
                 block.get('html', ''))
    assert '<pre class="attachment">&lt;script&gt;alert(1)&lt;/script&gt; '\
           '&amp; so on</pre>' in notes['html']
    index = next(block for block in blocks if block.get('caption') ==
                 rdi.CAPTION)
    assert index['rows'][1][0] == 'Y_run11_settings.csv', \
        'a file name is not read as Markdown'
    assert not any('Attachment: kept.txt' in block.get('html', '') +
                   block.get('caption', '') for block in blocks)


def test_the_check_column_and_the_attachment_edges():
    moved = _entry(status=di.MATCHES, found='/archive/a.csv')
    assert rdi._check(moved) == 'matches, found by name in another folder'
    assert rdi._check(_entry(status=di.NOT_FOUND)) == 'NOT FOUND'
    header = _entry(name='h.csv', role='attached', text='Run,Level\n')
    assert rdi.attachment(header) == {
        'kind': 'table', 'caption': 'Attachment: h.csv, settings list.',
        'headers': ['Run', 'Level'], 'rows': []}
    assert rdi.attachment(_entry(text='a,b\n')) is None, \
        'a referenced file is not printed'
    assert rdi.attachment(_entry(role='attached')) is None, 'no text kept'


def test_the_analysis_line_names_the_project_file(tmp_path):
    index = di.DataIndex()
    assert rdi.analysis(index, True).endswith(
        'The project file has not been checked.')
    index.project_file = ('p.vdyn', 1234, ABC, '2026-10-24T14:02:00+00:00')
    assert rdi.analysis(index, True).endswith(
        f'Project file: p.vdyn, 1,234 bytes, SHA-256 {ABC}, as it stood at '
        'the check; changes made since it was saved are not in it.')
    assert rdi.recipe({}, {'A': 1}) is None
    assert rdi.recipe({'B': {'verb': 'compute_psds', 'source': 'A'},
                       'Gone': {'verb': 'compute_psds', 'source': 'A'}},
                      {'A': 1, 'B': 2})['rows'] == [
        ['B', 'compute_psds', 'A']]
    assert rdi._when('not a time') == 'not a time'


def test_export_report_checks_the_index_first(tmp_path):
    project, name = _project_with_report(tmp_path)
    project.index_file(_file(tmp_path, 'log.txt'), 'log')
    _file(tmp_path, 'log.txt', b'edited')
    out = project.export_report(name, str(tmp_path / 'out.html'))
    with open(out, encoding='utf-8') as handle:
        data = _payload(handle.read())
    table = next(block for block in data['blocks'] if block['kind'] == 'table')
    assert [row[-1] for row in table['rows']] == ['matches', 'CHANGED']
    assert project.journal[-1].startswith('project.export_report('), \
        'the check inside is not a line of its own'


def test_the_registry_entry():
    feature = rf.FEATURES[rdi.FEATURE]
    assert feature.available and feature.requires == () and not feature.default
    assert feature.group == 'Records'
    assert rdi.FEATURE in rf.PRESETS['test_report'].features


# ---- the app ------------------------------------------------------------------------


def test_the_app_indexes_what_it_imports(window, tmp_path, monkeypatch):
    path = str(tmp_path / 'run1.vdyn')
    _history().save(path)
    calls = []
    real = window._run_long
    monkeypatch.setattr(window, '_run_long',
                        lambda *args, **kwargs: calls.append(args[0])
                        or real(*args, **kwargs))
    window.import_path(path)
    (entry,) = window.project.data_index
    assert entry.name == 'run1.vdyn' and entry.objects == ('Time History',)
    with open(path, 'rb') as handle:
        assert entry.sha256 == hashlib.sha256(handle.read()).hexdigest()
    assert calls == [], 'a small file is hashed in place'
    assert not any('index_file' in line for line in window.project.journal)
    # a large one is hashed on the worker, with the bar
    assert type(window).INLINE_HASH_BYTES == 32 * 1024 * 1024
    monkeypatch.setattr(type(window), 'INLINE_HASH_BYTES', 0)
    second = str(tmp_path / 'run2.vdyn')
    _history().save(second)
    window.import_path(second)
    assert calls == ['Checksumming run2.vdyn…']
    assert window.project.data_index[1].name == 'run2.vdyn'
    # canceled: the file stays imported, out of the index, and says so
    from visualdynamics.core.progress import Cancelled

    def cancel(*_args, **_kwargs):
        raise Cancelled('Checksumming')

    monkeypatch.setattr(window, '_run_long', cancel)
    said = []
    monkeypatch.setattr(window, '_show_status', said.append)
    third = str(tmp_path / 'run3.vdyn')
    _history().save(third)
    window.import_path(third)
    assert len(window.project.data_index) == 2
    assert 'Time History (3)' in window.objects
    assert any(line.startswith('run3.vdyn is imported but not in the data '
                               'index') for line in said)


def test_a_drop_indexes_each_file(window, tmp_path):
    paths = []
    for k in (1, 2):
        paths.append(str(tmp_path / f'drop{k}.vdyn'))
        _history().save(paths[-1])
    window._import_paths(paths)
    assert [entry.name for entry in window.project.data_index] == [
        'drop1.vdyn', 'drop2.vdyn']


def test_the_app_opens_a_project_with_its_index(window, tmp_path):
    project, _path = _imported(tmp_path)
    saved = str(tmp_path / 'p.vdyn')
    project.save(saved)
    window.import_path(saved)
    assert [entry.name for entry in window.project.data_index] == [
        'run1.vdyn'], 'opened, not imported'
    assert window.project.file == saved


@pytest.fixture
def editing(window, pump, tmp_path):
    path = str(tmp_path / 'run1.vdyn')
    _history().save(path)
    window.import_path(path)
    name = window.add_object('Report', Report('Report', [
        {'kind': 'text', 'text': '## Results\n\nThe run.'}]))
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    return window, window.report_editor, window.objects[name], name


def _menu(editor):
    out = []
    for action in editor.insert_menu.actions():
        out.append(action.text())
        if action.menu() is not None:
            out += [f'> {sub.text()}' for sub in action.menu().actions()]
    return out


def test_attached_file_in_the_editor(editing, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QInputDialog

    window, editor, report, _name = editing
    assert 'Attached File…' not in _menu(editor), 'off: not offered'
    editor._operate({'op': 'features', 'enable': ['data_index']})
    assert 'Attached File…' in _menu(editor)
    assert 'Remove from Index' not in _menu(editor), 'nothing added by hand'
    path = _file(tmp_path, 'settings.csv', b'a,b\n1,2\n')
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', staticmethod(
        lambda *_args: (path, '')))
    monkeypatch.setattr(QInputDialog, 'getText', staticmethod(
        lambda *_args: ('settings list', True)))
    monkeypatch.setattr(QInputDialog, 'getItem', staticmethod(
        lambda _parent, _title, _label, items, *_rest: (items[0], True)))
    next(action for action in editor.insert_menu.actions()
         if action.text() == 'Attached File…').trigger()
    entry = window.project.data_index[1]
    assert (entry.name, entry.role, entry.holds) == ('settings.csv',
                                                     'attached',
                                                     'settings list')
    assert window.project.journal[-1] == (
        f"project.index_file({path!r}, 'settings list', role='attached')")
    with open(editor._page_path, encoding='utf-8') as page_file:
        shown = _payload(page_file.read())
    assert any(block.get('caption') == 'Attachment: settings.csv, settings '
               'list.' for block in shown['blocks'])
    assert '> settings.csv' in _menu(editor)
    remove = next(action for action in editor.insert_menu.actions()
                  if action.text() == 'Remove from Index')
    remove.menu().actions()[0].trigger()
    assert [entry.name for entry in window.project.data_index] == [
        'run1.vdyn']
    assert window.project.journal[-1] == (
        "project.unindex_file('settings.csv')")
    # a refusal shows and changes nothing
    monkeypatch.setattr(QInputDialog, 'getText', staticmethod(
        lambda *_args: ('', True)))
    next(action for action in editor.insert_menu.actions()
         if action.text() == 'Attached File…').trigger()
    assert len(window.project.data_index) == 1
    assert report.features == ['data_index']


def test_export_checks_the_files_and_asks(editing, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    window, editor, _report, name = editing
    asked, saved, checked = [], [], []
    answer = [QMessageBox.StandardButton.No]
    monkeypatch.setattr(QMessageBox, 'question', staticmethod(
        lambda _parent, _title, text, *_rest: asked.append(text)
        or answer[0]))
    target = tmp_path / 'out.html'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', staticmethod(
        lambda *_args: saved.append(1) or (str(target), '')))
    real = window._run_long
    monkeypatch.setattr(window, '_run_long',
                        lambda *args, **kwargs: checked.append(args[0])
                        or real(*args, **kwargs))
    editor._operate({'op': 'export'})
    assert asked == [] and checked == [] and saved == [1], \
        'off: nothing read, nothing asked'
    editor._operate({'op': 'features', 'enable': ['data_index']})
    editor._operate({'op': 'export'})
    assert checked == ['Checking the data index…'] and asked == []
    assert saved == [1, 1], 'every file matches: nothing to ask'
    with open(window.project.data_index[0].path, 'ab') as handle:
        handle.write(b'!')
    editor._operate({'op': 'export'})
    assert asked == [(f'1 file in the data index of {name} does not match '
                      'the checksum indexed:\n\nrun1.vdyn: CHANGED\n\n'
                      'Export anyway?')]
    assert saved == [1, 1], 'No stops before the file dialog'
    answer[0] = QMessageBox.StandardButton.Yes
    said = []
    monkeypatch.setattr(window, '_show_status', said.append)
    editor._operate({'op': 'export'})
    assert saved == [1, 1, 1]
    assert '"CHANGED"' in target.read_text(encoding='utf-8')
    assert said[-1].endswith(': 0 of 1 file matches the checksum indexed; 1 '
                             'changed.')
    # a canceled check exports nothing
    from visualdynamics.core.progress import Cancelled

    def cancel(*_args, **_kwargs):
        raise Cancelled('Checking')

    monkeypatch.setattr(window, '_run_long', cancel)
    editor._operate({'op': 'export'})
    assert saved == [1, 1, 1]
    assert said[-1] == 'Export canceled: the data index was not checked'
