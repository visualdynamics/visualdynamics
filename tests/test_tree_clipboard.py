"""Copy and paste in the project tree (Brandon, 2026-09-03).

Cmd/Ctrl+C puts the selection on the clipboard two ways at once: as
object names, so a paste back into the tree duplicates them, and as
`.vdyn` files, so a paste into Finder or Explorer exports them — the
same two-faced payload a drag out of the tree already carried, with
the files written only when a file manager asks. Cmd/Ctrl+V takes
either: our names become copies through the project verb, and files
from anywhere import the way a drop does.

The actions are triggered rather than typed: `QTest.keySequence` with
a Cmd/Ctrl chord leaves the modifier held on the offscreen platform,
and every mouse drag in the tests that follow reads as a Ctrl-drag
(eleven truncate-view failures that never reproduced alone,
2026-09-03). The bindings themselves are pinned in their own test.
"""

from __future__ import annotations

import numpy as np
from conftest import fixture_path
from PySide6.QtCore import QMimeData, QUrl
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication

import visualdynamics
from visualdynamics.gui.project_tree import OBJECT_MIME, dragged_objects


def _select(window, *names):
    # current first: setCurrentItem clears a multi-selection
    window.tree.setCurrentItem(window._item_for_object(names[0]))
    window.tree.clearSelection()
    for name in names:
        window._item_for_object(name).setSelected(True)
    _focus(window)


def _focus(window):
    # a widget shortcut answers only in the focused widget of the
    # active window, which is what a person's Cmd+C in the tree is
    window.activateWindow()
    window.tree.setFocus()


def _two_objects(window, survey):
    shapes, frfs = survey
    window.add_object('Geometry', visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz')))
    window.add_object('FRF', frfs)
    return shapes


def test_copy_puts_names_and_files_on_the_clipboard(window, pump, survey,
                                                    tmp_path):
    _two_objects(window, survey)
    _select(window, 'Geometry', 'FRF')
    window.copy_action.trigger()
    pump()
    mime = QApplication.clipboard().mimeData()
    assert dragged_objects(mime) == ['Geometry', 'FRF'], 'the names, for a tree'
    assert mime.hasFormat('text/uri-list'), 'and files, for a folder'
    # the files exist only once somebody asks for them — a file manager
    # pasting — and then they are real .vdyn files that load back
    paths = [url.toLocalFile() for url in mime.urls()]
    assert [p.rsplit('/', 1)[1] for p in paths] == ['Geometry.vdyn', 'FRF.vdyn']
    back = visualdynamics.import_file(paths[1])     # a one-object .vdyn
    assert np.allclose(back.ordinate, window.objects['FRF'].ordinate)


def test_paste_duplicates_the_copied_objects(window, pump, survey):
    """A paste back into the tree is a duplicate: its own arrays, its
    own name, journaled as the verb a script would call."""
    _two_objects(window, survey)
    _select(window, 'Geometry', 'FRF')
    window.copy_action.trigger()
    pump()
    window.paste_action.trigger()
    pump()
    assert 'Geometry copy' in window.objects and 'FRF copy' in window.objects
    copy = window.objects['FRF copy']
    assert np.allclose(copy.ordinate, window.objects['FRF'].ordinate)
    copy.ordinate[...] = 0.0
    assert not np.allclose(window.objects['FRF'].ordinate, 0.0), (
        'a copy is its own object, not a view of the original')
    assert any('duplicate(' in line and "'Geometry'" in line
               for line in window.project.journal), window.project.journal
    # pasting again numbers the next copies rather than refusing
    window.paste_action.trigger()
    pump()
    assert 'FRF copy (2)' in window.objects


def test_pasting_files_imports_them(window, pump, survey, tmp_path):
    """Files on the clipboard — copied in a file manager, or from
    another window — come in the way a drop of files does."""
    shapes = _two_objects(window, survey)
    path = tmp_path / 'shapes.vdyn'
    visualdynamics.Project('t', {'Shapes': shapes}).save(path)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    QApplication.clipboard().setMimeData(mime)
    _select(window, 'Geometry')
    window.paste_action.trigger()
    pump(30)
    assert 'Shapes' in window.objects, 'imported from the pasted file'


def test_copying_the_project_row_offers_the_whole_project(window, pump,
                                                          survey):
    _two_objects(window, survey)
    window.tree.setCurrentItem(window.test_item)     # current first
    window.tree.clearSelection()
    window.test_item.setSelected(True)
    _focus(window)
    QApplication.clipboard().clear()
    window.copy_action.trigger()
    pump()
    mime = QApplication.clipboard().mimeData()
    assert not mime.hasFormat(OBJECT_MIME), 'no single objects: the project'
    paths = [url.toLocalFile() for url in mime.urls()]
    assert len(paths) == 1 and paths[0].endswith('.vdyn')
    assert set(visualdynamics.Project.open(paths[0]).names) >= {'Geometry', 'FRF'}


def test_the_shortcuts_are_the_platform_copy_and_paste(window):
    """Cmd+C / Cmd+V on the Mac, Ctrl+C / Ctrl+V elsewhere — Qt's
    standard keys, bound to the tree alone so the tables keep their
    own spreadsheet clipboard."""
    from PySide6.QtCore import Qt

    for action, standard in ((window.copy_action, QKeySequence.StandardKey.Copy),
                             (window.paste_action, QKeySequence.StandardKey.Paste)):
        assert action.shortcut() == QKeySequence(standard)
        assert action.shortcutContext() == Qt.ShortcutContext.WidgetShortcut
        assert action in window.tree.actions()
        assert action not in window.table.actions()
