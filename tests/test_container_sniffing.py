"""An npz archive is claimed for what is in it.

An sdynpy geometry archive was claimed only when it carried all four
of its arrays, though tracelines and elements are each optional — a
point cloud has neither (Brandon, 2026-09-20). And when nothing
claims an archive, the refusal says what is inside rather than only
the file's name.

Geometry is not read from netCDF: exodus keeps its own extensions,
and a `.nc4` is a controller's file (Brandon, 2026-09-20).
"""

from __future__ import annotations

import re

import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.geometry import Geometry
from visualdynamics.io.sniffing import npz_summary


def test_a_sniffer_never_raises_on_anything(tmp_path):
    """Whatever it is handed: a text file, a directory, a name that is
    not there."""
    from visualdynamics.io import sdynpy_npz

    text = tmp_path / 'notes.txt'
    text.write_text('a specification, in words')
    missing = tmp_path / 'absent.npz'
    for path in (text, tmp_path, missing,
                 fixture_path('plate', 'geometry.unv')):
        assert sdynpy_npz.sniff(path) is False
    for path in (text, tmp_path, fixture_path('plate', 'geometry.unv')):
        assert npz_summary(path) is None, 'not an archive at all'
    assert npz_summary(missing) == 'an npz archive that could not be read', (
        'named like one and unreadable, which is worth saying')


def test_a_geometry_is_not_read_from_netcdf(tmp_path):
    """A `.nc4` is a controller's file; exodus keeps its own
    extensions (Brandon, 2026-09-20)."""
    import shutil

    from visualdynamics.io import exodus

    named = tmp_path / 'geometry.nc4'
    shutil.copy(fixture_path('plate', 'geometry.exo'), named)
    assert not exodus.sniff(named)
    with pytest.raises(ValueError, match='No importer recognizes'):
        visualdynamics.import_file(named)
    assert exodus.sniff(fixture_path('plate', 'geometry.exo'))


def _geometry_archive(path, *, tracelines=True, elements=True):
    """An sdynpy geometry archive, written from this package's own
    record of the format rather than by sdynpy: one node, one
    coordinate system, and the two optional arrays on request."""
    import numpy as np

    from visualdynamics.io.sdynpy_npz import (
        CS_DTYPE,
        ELEMENT_DTYPE,
        NODE_DTYPE,
        TRACELINE_DTYPE,
    )

    node = np.zeros(3, NODE_DTYPE)
    node['id'] = [1, 2, 3]
    node['coordinate'] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    node['def_cs'] = node['disp_cs'] = 1
    cs = np.zeros(1, CS_DTYPE)
    cs['id'] = 1
    cs['matrix'] = [[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]]
    arrays = {'node': node, 'coordinate_system': cs}
    if elements:
        element = np.zeros(1, ELEMENT_DTYPE)
        element['id'] = 1
        element['type'] = 41
        conn = np.empty(1, dtype=object)
        conn[0] = np.array([1, 2, 3])
        element['connectivity'] = conn
        arrays['element'] = element
    if tracelines:
        arrays['traceline'] = np.zeros(0, TRACELINE_DTYPE)
    np.savez(path, **arrays)
    return path


def test_a_geometry_archive_needs_only_its_nodes_and_frames(tmp_path):
    """Tracelines and elements are each optional — a point cloud has
    neither — and asking for all four turned a geometry that lacked
    one into a file no reader would claim."""
    from visualdynamics.io import sdynpy_npz

    whole = _geometry_archive(tmp_path / 'whole.npz')
    bare = _geometry_archive(tmp_path / 'points.npz',
                             tracelines=False, elements=False)
    no_lines = _geometry_archive(tmp_path / 'no_lines.npz', tracelines=False)
    for path in (whole, bare, no_lines):
        assert sdynpy_npz.sniff(path), path.name
        geometry = visualdynamics.import_file(path)
        assert isinstance(geometry, Geometry) and geometry.num_nodes == 3
    assert len(visualdynamics.import_file(bare).elem_conn) == 0
    assert len(visualdynamics.import_file(whole).elem_conn) == 1


def test_the_other_sdynpy_archives_are_left_alone(tmp_path):
    """Two names a geometry cannot be without, and no other sdynpy
    save carries both."""
    import numpy as np

    from visualdynamics.io import sdynpy_data, sdynpy_npz

    data = tmp_path / 'data.npz'
    np.savez(data, data=np.zeros(3), function_type=np.array(1))
    assert not sdynpy_npz.sniff(data)
    assert sdynpy_data.sniff(data)


def test_an_archive_nobody_claims_says_what_it_holds(tmp_path):
    import numpy as np

    path = tmp_path / 'strange.npz'
    np.savez(path, shapes=np.zeros(3), frequency=np.zeros(3))
    with pytest.raises(ValueError, match='No importer recognizes') as refusal:
        visualdynamics.import_file(path)
    said = str(refusal.value)
    assert 'npz archive holding 2 arrays' in said
    assert 'frequency' in said and 'shapes' in said


def test_a_path_that_is_not_there_says_so(tmp_path):
    """The window hands over a path that exists; a person types one.
    A missing file, and a `~` the shell never expanded, both came back
    as "No importer recognizes", which reads as "your file is the
    wrong kind" (Brandon, 2026-09-20)."""
    with pytest.raises(FileNotFoundError, match='no file at'):
        visualdynamics.import_file(tmp_path / 'absent.npz')
    with pytest.raises(IsADirectoryError, match='is a folder'):
        visualdynamics.import_file(tmp_path)


def _home_is(monkeypatch, folder):
    """Point `~` at `folder` on every platform.

    `HOME` on its own is a POSIX answer. `os.path.expanduser` reads
    `USERPROFILE` on Windows and never consults `HOME`, so setting only
    `HOME` left `~` expanding to the real account: the first of these
    two tests looked for its archive in the developer's home folder,
    and the second wrote `written.unv` into it, which is the thing a
    test must never do (Kevin Cross, 2026-09-21).
    """
    monkeypatch.setenv('HOME', str(folder))
    monkeypatch.setenv('USERPROFILE', str(folder))


def test_a_typed_home_path_is_expanded(tmp_path, monkeypatch):
    """`~/article.npz` is what a person writes, and Python does not
    expand it — the file was reported unrecognized rather than sought
    in the home folder."""
    _home_is(monkeypatch, tmp_path)
    _geometry_archive(tmp_path / 'article.npz')
    geometry = visualdynamics.import_file('~/article.npz')
    assert isinstance(geometry, Geometry) and geometry.num_nodes == 3
    # re.escape: `match` is a regular expression, and a Windows tmp_path
    # is full of backslashes that read as escapes
    with pytest.raises(FileNotFoundError, match=re.escape(str(tmp_path))):
        visualdynamics.import_file('~/absent.npz')


def test_a_written_path_is_expanded_too(tmp_path, monkeypatch):
    """Or the file lands in a folder named for a tilde."""
    _home_is(monkeypatch, tmp_path)
    geometry = visualdynamics.import_file(
        _geometry_archive(tmp_path / 'source.npz'))
    visualdynamics.export_file(geometry, '~/written.unv', 'unv')
    assert (tmp_path / 'written.unv').exists()
    assert not (tmp_path / '~').exists()
