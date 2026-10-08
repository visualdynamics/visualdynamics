"""The tree's context menu, and what a right-click is offered on.

Show only what applies, in the one place where a wrong entry does real
damage: a sub-item is a *part* of an object, not an object, so Save,
Rename and Active Geometry belong to the object row and must be absent
on the category under it — Save on a category would ask what file a list
of nodes is, and Rename would offer to rename a label that is not a name.

A geometry's Imported Units are the same story from the other side: one
length unit belongs to the whole geometry, so it is offered on the
object and not on the category.

The menu is built by `_show_tree_menu`, which pops it up modally; it is
read here by grabbing the visible QMenu from a zero-timer and closing it,
the same trick the end-to-end script used.
"""

from __future__ import annotations

import pytest
from conftest import fixture_path, menu_entries


@pytest.fixture
def geometry_item(window, pump, qt_app):
    window.import_paths([fixture_path('plate', 'geometry.exo')])
    pump()
    window.tree.collapseAll()
    window.test_item.setExpanded(True)
    item = window._item_for_object('Geometry')
    item.setExpanded(True)
    pump()
    return item


def test_an_object_row_offers_the_whole_objects_verbs(geometry_item, window,
                                                      qt_app):
    entries = menu_entries(window, qt_app, geometry_item)
    assert any('Save' in e for e in entries), entries
    assert any('Active Geometry' in e for e in entries), entries
    assert any('Units' in e for e in entries), (
        'a geometry has one length unit, declared on the geometry')


def test_a_category_row_offers_only_what_a_part_can_do(geometry_item, window,
                                                       qt_app):
    entries = menu_entries(window, qt_app, geometry_item.child(0))
    assert any('Edit' in e for e in entries), entries
    for absent in ('Save', 'Rename', 'Active Geometry', 'Units'):
        assert not any(absent in e for e in entries), (absent, entries)


def test_a_category_label_cannot_be_renamed_by_anything(geometry_item):
    """Labels are not names: nothing may start editing one."""
    from PySide6.QtCore import Qt

    for i in range(geometry_item.childCount()):
        child = geometry_item.child(i)
        assert not (child.flags() & Qt.ItemFlag.ItemIsEditable), child.text(0)


def test_deleting_a_category_empties_it_and_leaves_it_there(geometry_item,
                                                            window, pump):
    """The category is what the geometry *has*, not what is in it: the
    elements go, the nodes stay, and the row remains to add to."""
    geometry = window.objects['Geometry']
    window.tree.clearSelection()
    elements = next(geometry_item.child(i)
                    for i in range(geometry_item.childCount())
                    if geometry_item.child(i).text(0).startswith('Quads'))
    window.tree.setCurrentItem(elements)
    elements.setSelected(True)
    pump()
    assert len(geometry.elem_conn) > 0, 'the fixture has some'
    window.delete_selected()
    pump()
    assert len(geometry.elem_conn) == 0, 'every element gone'
    assert geometry.num_nodes > 0, 'the nodes are not the elements'
    assert window.objects.get('Geometry') is geometry, 'the geometry stays'
    assert geometry_item.childCount() == 9, 'and so do all its categories'
    assert any(geometry_item.child(i).text(0).startswith('Quads (0)')
               for i in range(geometry_item.childCount()))
    assert 'Removed' in window.statusBar().currentMessage()
    geometry.validate()


# ---- what the menu deliberately does not offer ----------------------------

def test_rename_is_a_double_click_and_not_a_menu_entry(geometry_item, window,
                                                       qt_app):
    """Double-clicking a row edits its name in place — that is the tree's
    edit trigger, on every object row — so the menu entry was a second
    way to reach a thing that was already one gesture away. F2 still
    does it too, for anyone who reaches for a key."""
    from PySide6.QtCore import Qt

    assert geometry_item.flags() & Qt.ItemFlag.ItemIsEditable
    entries = menu_entries(window, qt_app, geometry_item)
    assert not any('Rename' in e for e in entries), entries


def test_delete_is_a_key_and_not_a_menu_entry(geometry_item, window, qt_app):
    """Delete and Backspace do it, on the tree, the grids and the 3-D
    view alike. A menu entry for a verb that has a key only crowds the
    ones that have nowhere else to live."""
    for item in (geometry_item, geometry_item.child(0)):
        entries = menu_entries(window, qt_app, item)
        assert not any('Delete' in e for e in entries), (item.text(0), entries)


def test_a_channel_table_is_not_asked_for_imported_units(window, qt_app, pump):
    """It declares them in its own `unit` column, which is edited in the
    table like every other cell. A second place to say the same thing
    could disagree with the first."""
    from visualdynamics.core.channel_table import ChannelTable

    window.add_object('Channels', ChannelTable({
        'channel': [1], 'node': ['101'], 'direction': ['Z+'], 'unit': ['g']}))
    pump()
    entries = menu_entries(window, qt_app, window._item_for_object('Channels'))
    assert entries, 'the row still has a menu'
    assert not any('Units' in e for e in entries), entries


def test_photographs_are_not_asked_for_imported_units(window, qt_app, pump,
                                                      tmp_path):
    """Nothing about a JPEG is a quantity."""
    from PySide6.QtGui import QImage

    made = tmp_path / 'Setup.png'
    QImage(8, 8, QImage.Format.Format_RGB32).save(str(made))
    window.import_paths([str(made)])
    pump()
    entries = menu_entries(window, qt_app, window._item_for_object('Photos'))
    assert entries, 'the row still has a menu'
    assert not any('Units' in e for e in entries), entries


def test_a_report_is_not_asked_for_imported_units(window, qt_app, pump):
    """A report is a document. It was being offered the verb because the
    rule was a list of what to leave out rather than a question asked of
    the object."""
    from visualdynamics.core.report import Report

    window.add_object('Report', Report())
    pump()
    entries = menu_entries(window, qt_app, window._item_for_object('Report'))
    assert entries, 'the row still has a menu'
    assert not any('Units' in e for e in entries), entries
