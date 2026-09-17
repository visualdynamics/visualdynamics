"""Photos: dropped images, one object per project, and photo figures.

Dropping png/jpeg files on the tree lands them together in the
project's Photos object; a report's photo block embeds one of them as
a numbered figure, chosen by name from a drop-down.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from conftest import fixture_path

from visualdynamics import io
from visualdynamics.core.photos import Photos
from visualdynamics.core.report import Report, insert_options
from visualdynamics.report import render_html


def _image(path, color):
    from PySide6.QtGui import QColor, QImage

    image = QImage(4, 3, QImage.Format.Format_RGB32)
    image.fill(QColor(color))
    assert image.save(str(path))
    return str(path)


def _payload(html):
    return json.loads(
        html.split('type="application/json">')[1].split('</script>')[0])


def test_photos_hold_the_file_bytes_verbatim(qt_app, tmp_path):
    path = _image(tmp_path / 'front.png', 'red')
    photos = Photos()
    assert photos.add_file(path) == 'front'
    assert photos.add_file(path) == 'front (2)', 'names dedupe'
    assert photos.formats == ['png', 'png']
    assert photos.images[0] == pathlib.Path(path).read_bytes()
    with pytest.raises(ValueError):
        photos.add_file(str(tmp_path / 'notes.txt'))


def test_photos_round_trip_through_vibe(qt_app, tmp_path):
    photos = Photos()
    photos.add_file(_image(tmp_path / 'front.png', 'red'))
    photos.add_file(_image(tmp_path / 'side.jpg', 'blue'))
    io.save(photos, str(tmp_path / 'photos.vdyn'))
    back = io.load(str(tmp_path / 'photos.vdyn'))
    assert isinstance(back, Photos)
    assert back.names == photos.names
    assert back.formats == ['png', 'jpeg']
    assert back.images == photos.images, 'bytes verbatim, no re-encode'


def test_dropped_images_land_in_one_photos_object(window, pump, tmp_path):
    front = _image(tmp_path / 'front.png', 'red')
    side = _image(tmp_path / 'side.jpg', 'blue')
    window.import_paths([front, side])
    holders = {name: obj for name, obj in window.objects.items()
               if isinstance(obj, Photos)}
    assert list(holders) == ['Photos']
    photos = holders['Photos']
    assert photos.names == ['front', 'side']
    window.import_paths([front])
    assert len([obj for obj in window.objects.values()
                if isinstance(obj, Photos)]) == 1, 'one object, always'
    assert photos.names == ['front', 'side', 'front (2)']


def test_selecting_photos_shows_the_pictures(window, pump, tmp_path):
    import pyqtgraph as pg

    window.import_paths([_image(tmp_path / 'front.png', 'red'),
                         _image(tmp_path / 'side.png', 'blue')])
    window.render_current()
    pump()
    images = [item
              for plot in window.data_pane.graphics.ci.items
              for item in getattr(plot, 'items', [])
              if isinstance(item, pg.ImageItem)]
    assert len(images) == 2, 'every photo on screen, stacked'


def test_photos_expand_into_a_grid_and_picks_restrict_the_view(
        window, pump, tmp_path):
    """The same grid habits as every other object: expand, pick, see
    exactly what is picked."""
    import pyqtgraph as pg

    window.import_paths([_image(tmp_path / 'front.png', 'red'),
                         _image(tmp_path / 'side.png', 'blue'),
                         _image(tmp_path / 'top.png', 'green')])
    grid = window.record_grids['Photos']
    assert grid.kind == 'photo'
    assert grid.responses == ['front', 'side', 'top']
    item = window._item_for_object('Photos')
    item.setExpanded(True)
    pump()
    grid.select_records([1, 2])
    item.setSelected(True)
    window.render_current()
    pump()
    images = [image
              for plot in window.data_pane.graphics.ci.items
              for image in getattr(plot, 'items', [])
              if isinstance(image, pg.ImageItem)]
    assert len(images) == 2, 'only the picked photos are shown'


def test_picked_photos_delete_like_records(window, pump, tmp_path):
    window.import_paths([_image(tmp_path / 'front.png', 'red'),
                         _image(tmp_path / 'side.png', 'blue')])
    item = window._item_for_object('Photos')
    item.setExpanded(True)
    pump()
    window.record_grids['Photos'].select_records([0])
    item.setSelected(True)
    window.delete_selected()
    photos = window.objects['Photos']
    assert photos.names == ['side']
    assert window.record_grids['Photos'].responses == ['side'], (
        'the grid rebuilt with the survivor')
    assert 'Photos' in window.objects, (
        'a photo pick deletes the photo, never the object')


def test_a_photo_renames_and_report_blocks_follow(window, pump, tmp_path,
                                                  monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    window.import_paths([_image(tmp_path / 'front.png', 'red'),
                         _image(tmp_path / 'side.png', 'blue')])
    name = window.generate_report('empty')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    window.report_editor._operate({'op': 'insert', 'at': 0,
                                   'kind': 'photos'})
    report = window.objects[name]
    assert report.blocks[0]['photo'] == 'front'
    monkeypatch.setattr(QInputDialog, 'getText',
                        staticmethod(lambda *a, **k: ('front view', True)))
    window._rename_photo('Photos', 0)
    photos = window.objects['Photos']
    assert photos.names == ['front view', 'side']
    assert report.blocks[0]['photo'] == 'front view', (
        'a bound report block follows the rename')
    assert window.record_grids['Photos'].responses == ['front view', 'side']
    # a rename that would collide is refused, nothing changes
    monkeypatch.setattr(QInputDialog, 'getText',
                        staticmethod(lambda *a, **k: ('side', True)))
    window._rename_photo('Photos', 0)
    assert photos.names == ['front view', 'side']


def test_a_photo_block_is_a_numbered_figure(qt_app, tmp_path):
    photos = Photos()
    photos.add_file(_image(tmp_path / 'front.png', 'red'))
    report = Report('R', [
        {'kind': 'photo', 'source': 'Photos', 'photo': 'front',
         'caption': 'Test setup'},
        {'kind': 'photo', 'source': 'Photos', 'photo': 'gone',
         'caption': ''},
    ])
    blocks = _payload(render_html(report, {'Photos': photos}))['blocks']
    assert len(blocks) == 1, 'a missing photo renders nothing'
    assert blocks[0]['kind'] == 'image'
    assert blocks[0]['label'] == 'Figure 1'
    assert blocks[0]['src'].startswith('data:image/png;base64,')
    assert ('photos', 'Photos') in insert_options({'Photos': photos})


def test_the_template_adds_every_photo_as_a_figure(qt_app, tmp_path):
    from visualdynamics.core.report import modal_template

    photos = Photos()
    photos.add_file(_image(tmp_path / 'front.png', 'red'))
    photos.add_file(_image(tmp_path / 'side.png', 'blue'))
    template = modal_template({'Photos': photos})
    blocks = [b for b in template.blocks if b['kind'] == 'photo']
    assert [b['photo'] for b in blocks] == ['front', 'side']
    assert all(b['caption'].startswith('Test setup') for b in blocks)
    # with no photos yet, one slot keeps the outline
    empty = [b for b in modal_template({}).blocks if b['kind'] == 'photo']
    assert len(empty) == 1 and empty[0]['photo'] == ''


def test_the_photos_insert_binds_the_first_photo(window, pump, tmp_path):
    window.import_paths([_image(tmp_path / 'setup.png', 'red')])
    name = window.generate_report('empty')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    editor = window.report_editor
    editor._operate({'op': 'insert', 'at': 0, 'kind': 'photos'})
    report = window.objects[name]
    assert report.blocks[0] == {'kind': 'photo', 'source': 'Photos',
                                'photo': 'setup', 'caption': ''}
    editor._operate({'op': 'field', 'at': 0, 'field': 'photo',
                     'value': ''})
    assert report.blocks[0]['photo'] == ''


def test_a_small_grid_takes_a_small_row(window, pump, tmp_path):
    """QTableWidget's stock size hint is 192 px tall whatever the
    content; the tree row must take the grid's actual size instead —
    one photo is one 24 px row, not a half-empty panel."""
    photos = Photos()
    photos.add_file(_image(tmp_path / 'one.png', 'red'))
    window.add_object('Photos', photos)
    item = window._item_for_object('Photos')
    item.setExpanded(True)
    pump()
    holder = item.child(0)
    assert window.tree.visualItemRect(holder).height() < 60


def test_a_photo_name_is_edited_in_place(window, pump, tmp_path):
    """A photograph's row label *is* its name. Double-clicking a name
    to change it is what the tree does for an object; this is the same
    gesture on a grid's edge, since photos expand into a grid (and since
    2026-09-06 the coordinates of records and channels edit the same
    way — test_rename_dof).
    """
    from PySide6.QtWidgets import QLineEdit

    window.import_paths([_image(tmp_path / 'front.png', 'red'),
                         _image(tmp_path / 'side.png', 'blue')])
    pump()
    grid = window.record_grids['Photos']
    grid.verticalHeader().sectionDoubleClicked.emit(0)   # the gesture itself
    editor = grid.verticalHeader().findChild(QLineEdit)
    assert editor is not None, 'double-clicking the name opens an editor'
    assert editor.text() == 'front', 'opens on the name it will replace'
    editor.setText('front quarter')
    editor.editingFinished.emit()
    pump()
    pump()          # the commit is queued: it rebuilds this very grid
    assert window.objects['Photos'].names == ['front quarter', 'side']
    assert window.record_grids['Photos'].responses == ['front quarter', 'side']


def test_an_abandoned_edit_changes_nothing(window, pump, tmp_path):
    """Escape hides the editor, hiding loses the focus, and losing focus
    is `editingFinished` — so without a guard the abandoned edit
    committed itself on the way out."""
    window.import_paths([_image(tmp_path / 'front.png', 'red')])
    pump()
    grid = window.record_grids['Photos']
    editor = grid.edit_row_label(0)
    editor.setText('never mind')
    editor.abandoned.emit()
    editor.editingFinished.emit()       # what the focus loss then sends
    pump()
    pump()
    assert window.objects['Photos'].names == ['front']


def test_a_mode_grid_does_not_offer_to_rename_a_row(window, pump):
    """A mode is labeled by what it is. Names and coordinates are the
    user's to type over (test_rename_dof); a mode's label is not."""
    from PySide6.QtWidgets import QLineEdit

    window.import_paths([fixture_path('plate', 'shapes.npy')])
    pump()
    name = next(n for n, o in window.objects.items()
                if type(o).__name__ == 'ShapeSet')
    grid = window.record_grids[name]
    assert grid.kind == 'mode' and not grid.editable_rows
    grid.verticalHeader().sectionDoubleClicked.emit(0)
    assert grid.verticalHeader().findChild(QLineEdit) is None
