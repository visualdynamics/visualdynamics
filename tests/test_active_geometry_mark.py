"""Which geometry is active, said where it can be seen.

The active geometry is what an object *not* linked to a geometry of its
own is drawn on, checked against, and projected onto — so in a project
with two of them it decides what a great deal of the window is showing.
Bold was the only mark it wore, and in a tree where the real objects are
already darker than the gray slots around them, a bolder weight among
them is not a mark anyone can find.

The bullet is painted rather than put in the row: the row's text is the
object's name, and anything prepended to it would be carried into a
rename.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.shapes import ShapeSet
from visualdynamics.gui.project_tree import ACTIVE_MARK, ROLE_ACTIVE


def _geometry():
    return visualdynamics.Geometry(
        node_id=[1, 2, 3], node_xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        length_unit='m')


def _shapes():
    return ShapeSet(np.array([10.0, 20.0]), np.array([0.01, 0.01]),
                    ['1Z+', '2Z+', '3Z+'], np.ones((2, 3)))


@pytest.fixture
def two_geometries(window, pump):
    window.add_object('Experimental Geometry', _geometry())
    window.add_object('Experimental Modes', _shapes())
    window.add_object('FEM Geometry', _geometry())
    window.show()
    window.tree.collapseAll()
    window.test_item.setExpanded(True)
    pump()
    return window


def _marked(window):
    return [item.text(0) for item in window.tree._every_item()
            if item.data(0, ROLE_ACTIVE)]


def test_exactly_one_row_is_marked_and_it_is_the_active_one(two_geometries):
    window = two_geometries
    assert window.active_geometry == 'Experimental Geometry', (
        'the first geometry imported is active by default')
    assert _marked(window) == ['Experimental Geometry']


def test_the_mark_follows_the_active_geometry(two_geometries, pump):
    window = two_geometries
    window.set_active_geometry('FEM Geometry')
    pump()
    assert _marked(window) == ['FEM Geometry'], 'and the old one gives it up'


def _bullet_drawn(window, name):
    """Whether the bullet's blue is on the painted tree beside this row.

    The **viewport**, not the tree: `visualItemRect` is in viewport
    coordinates and grabbing the tree includes its header and frame, so
    a pixel read at face value lands twenty rows off — on the branch
    arrow, as it happens, which is gray and reads as "nothing was
    drawn".

    The band stops just past the bullet's center, short of where the
    icon's ink begins — the geometry icon is this very blue, so a scan
    that reached it would pass whether or not anything was painted.
    (The icon starts a few pixels inside the rect; the bullet sits at
    rect.left()+1 and the icon's leftmost ink at about +4.)
    """
    item = window._item_for_object(name)
    window.tree.scrollToItem(item)
    rect = window.tree.visualItemRect(item)
    image = window.tree.viewport().grab().toImage()
    return any(
        image.pixelColor(x, y).name() == ACTIVE_MARK
        for x in range(rect.left() - 4, rect.left() + 2)
        for y in range(rect.top(), rect.bottom() + 1))


def test_the_bullet_is_actually_drawn_and_only_on_that_row(two_geometries,
                                                           pump):
    """The role alone proves nothing — it is a number on an item. This
    reads the pixel where the bullet belongs, in the gap between the
    branch arrow and the icon."""
    window = two_geometries
    assert _bullet_drawn(window, 'Experimental Geometry')
    assert not _bullet_drawn(window, 'FEM Geometry')

    window.set_active_geometry('FEM Geometry')
    pump()
    assert _bullet_drawn(window, 'FEM Geometry')
    assert not _bullet_drawn(window, 'Experimental Geometry')


def test_it_wears_the_geometry_icons_own_blue(two_geometries):
    """One statement, not two: a second color would be a third
    vocabulary in a tree that already says a lot with color."""
    from visualdynamics.gui.icons import COLORS

    assert ACTIVE_MARK == COLORS['Geometry']


def test_the_name_is_still_only_the_name(two_geometries, pump):
    """Painted, not prepended — or the bullet would be renamed with it."""
    window = two_geometries
    item = window._item_for_object('Experimental Geometry')
    assert item.text(0) == 'Experimental Geometry'
    item.setText(0, 'Wing Test Article')
    pump()
    assert 'Wing Test Article' in window.objects
    assert _marked(window) == ['Wing Test Article']


# ---- finishing a rename ---------------------------------------------------

def test_return_finishes_a_rename(window, pump):
    """Escape and Return belong to a geometry editing session — one
    abandons it, the other makes a traceline from the nodes picked so
    far — and they were **window** shortcuts. A window shortcut is
    answered before the focused widget sees the key, so Return typed
    into any editor anywhere in the window was swallowed by a verb that
    was usually a no-op: a rename could not be finished with the Return
    key, only by clicking away from it, which reads as an editor that
    will not close.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLineEdit

    window.add_object('Geometry', _geometry())
    window.show()
    pump()
    item = window._item_for_object('Geometry')
    window.tree.setCurrentItem(item)
    window.tree.editItem(item, 0)
    pump()
    editor = window.tree.findChild(QLineEdit)
    assert editor is not None, 'the editor opened'
    editor.setText('Wing Test Article')
    QTest.keyClick(editor, Qt.Key.Key_Return)
    pump()
    assert 'Wing Test Article' in window.objects
    assert item.text(0) == 'Wing Test Article'


def test_escape_abandons_one(window, pump):
    """The other half of the same swallow."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLineEdit

    window.add_object('Geometry', _geometry())
    window.show()
    pump()
    item = window._item_for_object('Geometry')
    window.tree.setCurrentItem(item)
    window.tree.editItem(item, 0)
    pump()
    editor = window.tree.findChild(QLineEdit)
    editor.setText('Never Mind')
    QTest.keyClick(editor, Qt.Key.Key_Escape)
    pump()
    assert sorted(window.objects) == ['Geometry']
    assert item.text(0) == 'Geometry'


def test_the_session_keys_still_reach_the_places_that_use_them(window):
    """Scoped, not removed: the render window holds focus in a child of
    its own, hence WithChildren rather than WidgetShortcut."""
    from PySide6.QtCore import Qt

    for action in (window.commit_action, window.escape_action):
        assert action.shortcutContext() == \
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        assert action in window.scene.actions()
        assert action in window.table.actions()
        assert action not in window.actions(), 'no longer window-wide'


def test_the_menu_verb_actually_changes_the_active_geometry(two_geometries,
                                                            pump):
    """It never had. `QAction.triggered` passes its checked bool, and
    connected straight to `set_active_geometry` that bool landed in
    `name` as False — not None — so the verb skipped the current-item
    lookup, looked up the object called False, and refused. The menu
    entry did nothing, ever, and told a user with a geometry selected
    to select a geometry.
    """
    window = two_geometries
    window.tree.setCurrentItem(window._item_for_object('FEM Geometry'))
    window.active_geometry_action.trigger()
    pump()
    assert window.active_geometry == 'FEM Geometry'
    assert _marked(window) == ['FEM Geometry'], 'and the bullet moves'
