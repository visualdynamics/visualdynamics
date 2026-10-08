"""Dragging an object out of the window writes it as `.vdyn`.

The tree's drag already carried its own mime type so objects could be
moved between object groups. It now carries a second reading of the same
gesture — `text/uri-list` — so the same drag, released on the desktop
instead of on another row, saves a file there. Finder, Explorer and
Linux file managers all read that type; the tree ignores it.

The half worth testing hardest is not that a file appears. It is *when*
it appears: a project can be a hundred megabytes, and writing that at
the start of every drag — including the ones that only reorder rows —
would stall the window on every touch.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
from PySide6.QtCore import QUrl

import visualdynamics
from visualdynamics.gui.project_tree import (
    OBJECT_MIME,
    ROLE_WHOLE_PROJECT,
    _DraggedObjects,
    dragged_objects,
)


@pytest.fixture
def loaded(window, pump):
    """Two objects and a named project."""
    from visualdynamics.core.data import Psd

    f = np.linspace(10.0, 2000.0, 64)
    window.add_object('PSD', Psd(f, np.ones((1, 64)), response_dof=['1Z+'],
                                ordinate_dim='acceleration**2/frequency',
                                ordinate_unit='m/s**2'))
    window.add_object('Geometry', visualdynamics.Geometry(
        node_id=[1, 2], node_xyz=[[0, 0, 0], [1, 0, 0]], length_unit='m'))
    window.project.name = 'Drag Test'
    pump()
    return window


def urls_from(mime):
    """The paths a drop target receives — through `urls()`, which is
    what the platform actually calls.

    Not `data('text/uri-list')`. Both are supposed to answer, and for a
    while only the second one did: `urls()` came back empty, macOS
    treated the drag as a *location* rather than a file, and dropping an
    object on the desktop produced `localhost.fileloc`. The tests all
    passed, because they read the half that worked.
    """
    return [url.toLocalFile() for url in mime.urls()]


# ---- the laziness, which is the whole design ------------------------------

def test_nothing_is_written_until_a_target_asks(loaded, tmp_path):
    """The cost of a drag must not depend on the size of what is
    dragged. Hovering, outlining, dropping on another row — none of it
    asks for files, so none of it writes any."""
    wrote = []

    def write(names, folder):
        wrote.append(names)
        return []

    mime = _DraggedObjects(['PSD'], lambda folder: write(['PSD'], folder))
    # everything a drag over the tree itself does
    assert mime.hasFormat(OBJECT_MIME)
    assert mime.hasFormat('text/uri-list')
    assert 'text/uri-list' in mime.formats()
    assert dragged_objects(mime) == ['PSD']
    assert mime.hasUrls()
    assert wrote == [], 'asking what the drag *offers* wrote a file'


def test_both_readings_of_the_url_list_agree(loaded, tmp_path):
    """`urls()` and `data('text/uri-list')` must both answer, because
    different platforms ask differently — and an empty `urls()` is
    invisible from the other one."""
    window = loaded
    mime = _DraggedObjects(
        ['PSD'], lambda folder: window._write_for_drag(['PSD'], folder))

    from_urls = [u.toLocalFile() for u in mime.urls()]
    raw = bytes(mime.data('text/uri-list')).decode('utf-8')
    from_bytes = [QUrl(line).toLocalFile()
                  for line in raw.split('\r\n') if line.strip()]

    assert from_urls, 'urls() answered nothing — the platform reads this one'
    assert from_urls == from_bytes
    assert os.path.basename(from_urls[0]) == 'PSD.vdyn'


def test_asking_for_the_urls_writes_once(loaded, tmp_path):
    calls = []

    def write(folder):
        calls.append(folder)
        path = os.path.join(folder, 'x.vdyn')
        pathlib.Path(path).write_bytes(b'x')
        return [path]

    mime = _DraggedObjects(['PSD'], write)
    first = urls_from(mime)
    second = urls_from(mime)
    assert len(calls) == 1, 'a second reader must not write it again'
    assert first == second


# ---- what actually lands on the desktop -----------------------------------

def test_an_object_lands_as_a_readable_vdyn(loaded, tmp_path):
    window = loaded
    paths = window._write_for_drag(['PSD'], str(tmp_path))
    assert [os.path.basename(p) for p in paths] == ['PSD.vdyn']
    back = visualdynamics.load(paths[0])
    assert back.num_records == 1, 'and it reads back as the object it was'


def test_the_project_row_saves_the_whole_project(loaded, tmp_path):
    """`names is None` is the project row, and it must match Save As —
    every object, the links, the type, not just the selection."""
    window = loaded
    paths = window._write_for_drag(None, str(tmp_path))
    assert [os.path.basename(p) for p in paths] == ['Drag Test.vdyn']
    back = visualdynamics.Project.open(paths[0])
    assert set(back) == {'PSD', 'Geometry'}


def test_several_objects_land_as_several_files(loaded, tmp_path):
    window = loaded
    paths = window._write_for_drag(['PSD', 'Geometry'], str(tmp_path))
    assert sorted(os.path.basename(p) for p in paths) == [
        'Geometry.vdyn', 'PSD.vdyn']


def test_a_name_that_is_not_a_filename_still_lands(loaded, tmp_path):
    """'FRF 1/2' is a perfectly good object name and a terrible path."""
    window = loaded
    window.project.rename('PSD', 'FRF 1/2: "control"')
    paths = window._write_for_drag(['FRF 1/2: "control"'], str(tmp_path))
    name = os.path.basename(paths[0])
    assert not set(name) & set('/\\:*?"<>|'), name
    assert name.endswith('.vdyn')
    assert os.path.exists(paths[0])


def test_an_object_that_has_gone_is_skipped_rather_than_raising(loaded,
                                                               tmp_path):
    """The drag outlives the selection it started from."""
    window = loaded
    assert window._write_for_drag(['Not Here'], str(tmp_path)) == []


# ---- and the tree still moves objects the way it did ----------------------

def test_the_project_row_is_draggable_out_but_not_between_groups(loaded):
    """`ROLE_WHOLE_PROJECT` is not `ROLE_DRAGGABLE`: the project can be
    dragged to the desktop, and cannot be dropped into an object group."""
    window = loaded
    assert window.test_item.data(0, ROLE_WHOLE_PROJECT)
    assert not window.tree.draggable(window.test_item)


def test_a_drag_with_no_writer_offers_no_files():
    """A bare ProjectTree — one built without a window behind it — must
    not claim to offer files it cannot produce."""
    mime = _DraggedObjects(['PSD'], None)
    assert mime.hasFormat(OBJECT_MIME)
    assert not mime.hasFormat('text/uri-list')
    assert 'text/uri-list' not in mime.formats()


# ---- the form each object takes on the way out ----------------------------

def test_a_channel_table_drags_out_as_a_spreadsheet(loaded, tmp_path):
    """`.vdyn` is the lossless form for most objects and the wrong one
    here: a channel table *is* a spreadsheet, `.xlsx` keeps every column
    and every value, and it is what a colleague can open.

    Only true since the Excel importer existed. Before that it was a
    one-way door, and a one-way door is not equivalent to a `.vdyn`.
    """
    from visualdynamics.core.channel_table import ChannelTable

    window = loaded
    window.add_object('Channels', ChannelTable({
        'channel': [1], 'node': ['101'], 'direction': ['Z+'],
        'unit': ['g'], 'serial_number': ['007']}))
    paths = window._write_for_drag(['Channels'], str(tmp_path))
    assert [os.path.basename(p) for p in paths] == ['Channels.xlsx']

    back = visualdynamics.import_file(paths[0])
    assert isinstance(back, ChannelTable)
    assert list(back['serial_number']) == ['007'], 'and it comes back whole'


def test_photos_drag_out_as_a_folder_of_pictures(loaded, tmp_path, qt_app):
    """A `.vdyn` full of photographs is useless to anything but this
    application. One folder lands, named for the object, because no
    picture format holds more than one picture."""
    from PySide6.QtGui import QImage

    from visualdynamics.core.photos import Photos

    window = loaded
    album = Photos()
    for name in ('Setup', 'Fixture'):
        made = tmp_path / f'{name}.png'
        QImage(8, 8, QImage.Format.Format_RGB32).save(str(made))
        album.add_file(str(made))
    window.add_object('Photos', album)

    out = tmp_path / 'dropped'
    out.mkdir()
    paths = window._write_for_drag(['Photos'], str(out))
    assert len(paths) == 1, 'one thing lands, not two loose files'
    assert os.path.isdir(paths[0])
    assert os.path.basename(paths[0]) == 'Photos'
    assert sorted(os.listdir(paths[0])) == ['Fixture.png', 'Setup.png']


def test_everything_else_is_still_a_vdyn(loaded, tmp_path):
    """Because for everything else it is the only form that comes back."""
    window = loaded
    paths = window._write_for_drag(['Geometry', 'PSD'], str(tmp_path))
    assert sorted(os.path.basename(p) for p in paths) == [
        'Geometry.vdyn', 'PSD.vdyn']
