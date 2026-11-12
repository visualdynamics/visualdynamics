"""Opening, dropping or sniffing a file never runs code from it.

A NumPy file can hold a pickle, and a pickle is a program. Every file
here carries one that would write a marker file when unpickled, and the
marker must never appear, whichever way the file is reached.

Geometry files really do hold object arrays, so they are read through a
restricted reader; the tests at the end hold that it keeps reading them.

Brought over from Hertzforge with its restricted reader (Brandon,
2026-11-11: "Yes, port it"); the service tests became the window's.
"""

from __future__ import annotations

import io
import os
import pickle
import zipfile

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.io import (
    format_of,
    import_file,
    known_format,
    rattlesnake_sine,
    rattlesnake_specification,
    sdynpy_data,
    sdynpy_npz,
    sdynpy_shapes,
)
from visualdynamics.io.sniffing import describe, npz_has, npz_summary

MODULES = (sdynpy_npz, sdynpy_data, sdynpy_shapes, rattlesnake_specification,
           rattlesnake_sine)
RUNS_CODE = 'could run code'


class Payload:
    """Writes `marker` when unpickled, as a harmful pickle would do worse."""

    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return (open, (self.marker, 'w'))


def object_array(marker):
    array = np.empty(1, dtype=object)
    array[0] = Payload(marker)
    return array


def geometry_arrays():
    node = np.zeros(1, dtype=sdynpy_npz.NODE_DTYPE)
    cs = np.zeros(1, dtype=sdynpy_npz.CS_DTYPE)
    cs['matrix'][0] = np.vstack([np.eye(3), np.zeros(3)])
    node['id'], cs['id'] = 1, 1
    node['def_cs'], node['disp_cs'] = 1, 1
    return node, cs


def hostile_line(marker):
    line = np.zeros(1, dtype=sdynpy_npz.TRACELINE_DTYPE)
    line['connectivity'][0] = Payload(marker)
    return line


def bare_pickle(path, marker):
    path.write_bytes(pickle.dumps(Payload(marker)))


def write(kind, path, marker):
    """A hostile file of `kind`, its pickle where that format's reader or
    sniffer would meet it."""
    if kind == 'npy of objects':
        np.save(path, object_array(marker), allow_pickle=True)
    elif kind == 'npy shaped like mode shapes':
        fields = [('frequency', '<f8'), ('damping', '<f8'),
                  ('coordinate', 'O'), ('shape_matrix', 'O')]
        array = np.zeros(1, dtype=fields)
        array['shape_matrix'][0] = Payload(marker)
        np.save(path, array, allow_pickle=True)
    elif kind == 'pickle named npy' or kind == 'pickle named npz':
        bare_pickle(path, marker)
    elif kind == 'npz of objects':
        np.savez(path, evil=object_array(marker))
    elif kind == 'npz geometry, objects for nodes':
        np.savez(path, node=object_array(marker),
                 coordinate_system=geometry_arrays()[1])
    elif kind == 'npz geometry, payload in a traceline':
        node, cs = geometry_arrays()
        np.savez(path, node=node, coordinate_system=cs,
                 traceline=hostile_line(marker))
    elif kind == 'npz geometry, payload in an element':
        node, cs = geometry_arrays()
        element = np.zeros(1, dtype=sdynpy_npz.ELEMENT_DTYPE)
        element['connectivity'][0] = Payload(marker)
        np.savez(path, node=node, coordinate_system=cs, element=element)
    elif kind == 'npz data':
        np.savez(path, data=object_array(marker), function_type=1)
    elif kind == 'npz random specification':
        np.savez(path, f=object_array(marker), cpsd=object_array(marker))
    elif kind == 'npz sine specification':
        np.savez(path, frequency=object_array(marker), amplitude=np.zeros(1),
                 sweep_type=np.zeros(1), sweep_rate=np.zeros(1))
    else:
        raise AssertionError(kind)


# every kind, with the suffix it arrives under and the words its import
# has to give
KINDS = {
    'npy of objects': ('evil.npy', RUNS_CODE),
    'npy shaped like mode shapes': ('evil.npy', RUNS_CODE),
    'pickle named npy': ('evil.npy', 'not a NumPy file'),
    'pickle named npz': ('evil.npz', 'could not be read'),
    'npz of objects': ('evil.npz', 'No importer recognizes'),
    'npz geometry, objects for nodes': ('evil.npz', RUNS_CODE),
    'npz geometry, payload in a traceline': ('evil.npz', RUNS_CODE),
    'npz geometry, payload in an element': ('evil.npz', RUNS_CODE),
    'npz data': ('evil.npz', RUNS_CODE),
    'npz random specification': ('evil.npz', RUNS_CODE),
    'npz sine specification': ('evil.npz', RUNS_CODE),
}


@pytest.fixture(params=sorted(KINDS))
def hostile(request, tmp_path):
    """(the file, the marker its pickle would write, the words to expect)"""
    name, words = KINDS[request.param]
    path = tmp_path / name
    marker = tmp_path / 'RAN'
    write(request.param, path, marker)
    assert not marker.exists(), 'writing the file did not run it'
    return path, marker, words


def test_format_of_and_every_sniffer_do_not_run_it(hostile):
    path, marker, _words = hostile
    assert isinstance(format_of(path), str)
    known_format(path)
    describe(path)
    npz_summary(path)
    npz_has(path, {'node'})
    for module in MODULES:
        assert isinstance(module.sniff(path), bool)
    assert not marker.exists()


def test_every_reader_refuses_it_in_words(hostile):
    path, marker, _words = hostile
    for module in MODULES:
        with pytest.raises((ValueError, KeyError)) as refused:
            module.load(path)
        assert str(refused.value), f'{module.__name__} said nothing'
    assert not marker.exists()


def test_importing_it_says_why(hostile):
    path, marker, words = hostile
    with pytest.raises(ValueError, match=words):
        import_file(path)
    assert not marker.exists()


def test_dropping_it_on_the_window_runs_nothing(hostile, window, pump,
                                               monkeypatch):
    """The window says why, as for any file it cannot read."""
    from visualdynamics.gui import main_window as window_module

    path, marker, words = hostile
    shown = []
    monkeypatch.setattr(window_module.QMessageBox, 'warning',
                        lambda parent, title, text: shown.append((title, text)))
    assert window.import_paths([str(path)]) == []
    pump()
    assert not marker.exists()
    assert len(shown) == 1 and words in shown[0][1]


def test_opening_it_as_a_project_runs_nothing(hostile):
    path, marker, _words = hostile
    with pytest.raises(Exception):  # noqa: B017 - any refusal; the marker is the test
        visualdynamics.Project.open(path)
    assert not marker.exists()


def test_a_project_import_runs_nothing(hostile):
    path, marker, words = hostile
    with pytest.raises(ValueError, match=words):
        visualdynamics.Project().import_file(str(path))
    assert not marker.exists()


class Reduces:
    """Pickles as the call it is given, for what numpy itself would never
    write."""

    def __init__(self, *call):
        self.call = call

    def __reduce__(self):
        return self.call


def _reconstruct():
    return np.empty(0).__reduce__()[0]


def geometry_holding(path, connectivity):
    node, cs = geometry_arrays()
    line = np.zeros(1, dtype=sdynpy_npz.TRACELINE_DTYPE)
    line['connectivity'][0] = connectivity
    np.savez(path, node=node, coordinate_system=cs, traceline=line)


@pytest.mark.parametrize('call', [
    # the names a restricted unpickler is usually caught out by
    lambda m: Reduces(os.system, (f'touch {m}',)),
    lambda m: Reduces(eval, (f"open({str(m)!r}, 'w')",)),
    lambda m: Reduces(exec, (f"open({str(m)!r}, 'w')",)),
    lambda m: Reduces(np.save, (str(m), np.zeros(1))),
    lambda m: Reduces(np.load, (str(m),)),
    lambda m: Reduces(getattr, ('text', 'upper')),
    # names numpy would accept, asked to make a thing it would not
    lambda m: Reduces(_reconstruct(), (Reduces, (1,), b'b')),
    lambda m: Reduces(_reconstruct(), (np.ndarray, (1,), b'b'),
                      (1, (1,), np.dtype('O'), False, b'\xef' * 8)),
    lambda m: Reduces(_reconstruct(), (np.ndarray, (1,), b'b'),
                      (1, (1,), np.dtype('O'), False, [Payload(m)])),
    lambda m: Reduces(_reconstruct(), (np.recarray, (1,), b'b')),
    lambda m: Reduces(np.dtype, ('O', False, True)),
], ids=['system', 'eval', 'exec', 'numpy save', 'numpy load', 'getattr',
        'a class of its own', 'objects from raw bytes', 'an array holding a payload',
        'an ndarray subclass', 'a dtype on its own'])
def test_the_restricted_reader_refuses_what_is_not_plain_data(tmp_path, call):
    path = tmp_path / 'evil.npz'
    marker = tmp_path / 'RAN'
    geometry_holding(path, call(marker))
    assert sdynpy_npz.sniff(path), 'claimed by its names, and refused on reading'
    with pytest.raises(ValueError, match=RUNS_CODE):
        sdynpy_npz.load(path)
    assert not marker.exists()


def test_a_dtype_state_cannot_hide_an_object_field(tmp_path):
    """numpy takes a dtype's flags and fields from its pickled state, so a
    file could say a field holds no objects and give raw bytes for it."""
    state = (3, '|', None, ('a',), {'a': (np.dtype('O'), 0)}, 8, 1, 0)
    dtype = Reduces(np.dtype, ('V8', False, True), state)
    path = tmp_path / 'evil.npz'
    geometry_holding(path, Reduces(_reconstruct(), (np.ndarray, (1,), b'b'),
                                   (1, (1,), dtype, False, b'\xef' * 8)))
    with pytest.raises(ValueError, match=RUNS_CODE):
        sdynpy_npz.load(path)


def test_the_refusal_names_what_the_file_asked_for(tmp_path):
    path = tmp_path / 'evil.npz'
    geometry_holding(path, Reduces(os.system, ('true',)))
    with pytest.raises(ValueError, match=r'traceline.*(posix|nt)\.system'):
        sdynpy_npz.load(path)


def test_a_broken_archive_is_refused_in_words(tmp_path):
    path = tmp_path / 'broken.npz'
    path.write_bytes(b'PK\x03\x04 and then nothing an archive would hold')
    assert not sdynpy_npz.sniff(path)
    with pytest.raises(ValueError, match='could not be read'):
        sdynpy_npz.load(path)


def test_a_truncated_pickle_is_refused_too(tmp_path):
    path = tmp_path / 'short.npz'
    node, cs = geometry_arrays()
    line = np.zeros(1, dtype=sdynpy_npz.TRACELINE_DTYPE)
    line['connectivity'][0] = np.array([1, 2], dtype=np.uint32)
    np.savez(path, node=node, coordinate_system=cs, traceline=line)
    with zipfile.ZipFile(path) as zipped:
        members = {name: zipped.read(name) for name in zipped.namelist()}
    members['traceline.npy'] = members['traceline.npy'][:-20]
    with zipfile.ZipFile(path, 'w') as zipped:
        for name, data in members.items():
            zipped.writestr(name, data)
    with pytest.raises(ValueError, match=RUNS_CODE):
        sdynpy_npz.load(path)


@pytest.mark.parametrize('name', [
    'test_geometry.npz', 'geometry.npz', 'frfs.npz', 'time.npz', 'psd.npz',
    'spectrum.npz', 'shapes.npy'])
def test_the_plate_files_still_read(name):
    path = fixture_path('plate', name)
    assert known_format(path) is not None
    assert import_file(path) is not None


def test_the_drawn_lines_of_a_real_geometry_survive():
    """`test_geometry.npz` holds object arrays, which is why the geometry
    reader has to read them at all."""
    with np.load(fixture_path('plate', 'test_geometry.npz'),
                 allow_pickle=True) as raw:
        expected = [[int(n) for n in row] for row in raw['traceline']['connectivity']]
    assert expected, 'the fixture holds tracelines'
    geometry = sdynpy_npz.load(fixture_path('plate', 'test_geometry.npz'))
    drawn = [[int(n) for n in chain] for line in geometry.drawn_lines()
             for chain in line['chains']]
    assert drawn == expected


def legitimate_geometry(connectivity):
    """A geometry whose element connectivity holds `connectivity`, one
    entry per element, as sdynpy writes it."""
    node = np.zeros(4, dtype=sdynpy_npz.NODE_DTYPE)
    node['id'] = [1, 2, 3, 4]
    node['coordinate'] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    node['def_cs'], node['disp_cs'] = 1, 1
    cs = geometry_arrays()[1]
    element = np.zeros(len(connectivity), dtype=sdynpy_npz.ELEMENT_DTYPE)
    element['id'] = np.arange(1, len(connectivity) + 1)
    element['type'] = 41
    for i, row in enumerate(connectivity):
        element['connectivity'][i] = row
    return node, cs, element


@pytest.mark.parametrize('connectivity', [
    [np.array([1, 2, 3], dtype=np.uint32), np.array([2, 3, 4], dtype=np.uint32)],
    [np.array([1, 2, 3], dtype=np.uint64), np.array([2, 3, 4], dtype=np.int64)],
    [np.array([1, 2, 3], dtype='>u4'), np.array([2, 3, 4], dtype='>i8')],
    [[1, 2, 3], (2, 3, 4)],
], ids=['uint32, as sdynpy writes it', 'other widths', 'big-endian',
        'lists and tuples'])
def test_connectivity_reads_back_as_written(tmp_path, connectivity):
    path = tmp_path / 'good.npz'
    node, cs, element = legitimate_geometry(connectivity)
    np.savez(path, node=node, coordinate_system=cs, element=element)
    geometry = sdynpy_npz.load(path)
    assert [list(map(int, c)) for c in geometry.elem_conn] == [[1, 2, 3], [2, 3, 4]]


def test_a_geometry_with_no_rows_reads(tmp_path):
    path = tmp_path / 'cloud.npz'
    node, cs, _element = legitimate_geometry([])
    np.savez(path, node=node, coordinate_system=cs,
             traceline=np.zeros(0, dtype=sdynpy_npz.TRACELINE_DTYPE),
             element=np.zeros(0, dtype=sdynpy_npz.ELEMENT_DTYPE))
    assert sdynpy_npz.load(path).num_nodes == 4


@pytest.mark.parametrize('protocol', [3, 4])
def test_a_file_from_an_older_numpy_reads(tmp_path, protocol):
    """numpy before 2.0 pickled under `numpy.core.multiarray`, and a saved
    file keeps the name for good."""
    node, cs, element = legitimate_geometry([np.array([1, 2, 3], dtype=np.uint32)])
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer, np.lib.format.header_data_from_array_1_0(element))
    head = buffer.getvalue()
    pickled = _renamed_pickle(element, protocol)
    assert b'numpy.core.multiarray' in pickled

    path = tmp_path / 'older.npz'
    np.savez(path, node=node, coordinate_system=cs)
    with zipfile.ZipFile(path, 'a') as zipped:
        zipped.writestr('element.npy', head + pickled)
    assert [int(n) for n in sdynpy_npz.load(path).elem_conn[0]] == [1, 2, 3]


def _renamed_pickle(array, protocol):
    """`array` pickled, naming numpy's reconstruct the older way. Protocol
    3 names it in text; protocol 4 counts the length of the string, which
    is written out here by hand."""
    pickled = pickle.dumps(array, protocol=protocol)
    new, old = 'numpy._core.multiarray', 'numpy.core.multiarray'
    if protocol == 3:
        return pickled.replace(new.encode(), old.encode())
    counted = bytes([0x8c, len(new)]) + new.encode()
    assert counted in pickled
    # a framed pickle holds its length up front, so drop the frame
    framed = pickled.index(b'\x95')
    body = pickled[framed + 9:]
    return pickled[:framed] + body.replace(
        counted, bytes([0x8c, len(old)]) + old.encode())


def test_geometry_written_by_visual_dynamics_round_trips(tmp_path):
    path = tmp_path / 'again.npz'
    geometry = sdynpy_npz.load(fixture_path('plate', 'test_geometry.npz'))
    sdynpy_npz.save(geometry, path)
    again = sdynpy_npz.load(path)
    assert again.num_nodes == geometry.num_nodes
    assert again.drawn_lines()[0]['chains'] == geometry.drawn_lines()[0]['chains']


def test_a_plain_array_that_is_no_format_is_still_named_plainly(tmp_path):
    """A `.npy` of numbers is unrecognized, and nothing suggests it is
    dangerous."""
    path = tmp_path / 'numbers.npy'
    np.save(path, np.arange(3))
    assert describe(path) is None
    assert format_of(path) == 'NPY file (.npy)'
    with pytest.raises(ValueError, match='No importer recognizes') as refused:
        import_file(path)
    assert RUNS_CODE not in str(refused.value)
