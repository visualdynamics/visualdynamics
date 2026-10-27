"""Solids put in order by dragging their rows (Brandon, 2026-10-26:
the user needs to drag and rearrange the volumes, since the second is
always subtracted from the first).

The order is what a boolean made from a pick reads first to last, and a
subtraction cuts every solid after its first from the first. A drag
moves solid rows among their siblings only: the parts among the parts,
a boolean's own solids among themselves.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt
from PySide6.QtGui import QDropEvent

import visualdynamics
from visualdynamics.core import solids
from visualdynamics.gui.project_tree import ROLE_REFERENCE, SOLIDS_MIME

pytest.importorskip('OCP')


def _project():
    project = visualdynamics.Project()
    geometry = visualdynamics.Geometry([], [])
    geometry.define_units('m')
    project.add('G', geometry)
    cube = project.add_solid_block('G', (0, 0, 0), (1, 1, 1), name='cube')
    rod = project.add_solid_cylinder('G', (0, 0, 0), 0.2, 2.0, name='rod')
    cut = project.combine_solids('G', 'subtract', [cube, rod])
    ball = project.add_solid_sphere('G', (3, 0, 0), 0.5, name='ball')
    return project, geometry, cube, rod, cut, ball


# ---- the verb ----------------------------------------------------------------

def test_the_order_of_a_subtraction_decides_what_is_cut():
    project, geometry, cube, rod, cut, _ball = _project()
    assert solids.volume(geometry, cut) == pytest.approx(1 - 0.04 * 3.14159265,
                                                         rel=1e-6)
    project.reorder_solids('G', [rod, cube], within=cut)
    assert solids.solid(geometry, cut).operands == (rod, cube)
    assert solids.volume(geometry, cut) == pytest.approx(0.04 * 3.14159265,
                                                         rel=1e-6), \
        'the rod with the cube cut away: the part of it outside the cube'
    assert project.journal[-1] == \
        f"project.reorder_solids('G', [{rod}, {cube}], within={cut})"


def test_the_parts_reorder_and_the_solids_a_boolean_uses_stay():
    project, geometry, cube, rod, cut, ball = _project()
    before = [s.id for s in geometry.solids]
    project.reorder_solids('G', [ball, cut])
    assert [s.id for s in solids.top_level(geometry)] == [ball, cut]
    assert [s.id for s in geometry.solids if s.id in (cube, rod)] == \
        [i for i in before if i in (cube, rod)]
    assert project.journal[-1] == f"project.reorder_solids('G', [{ball}, {cut}])"


def test_what_is_refused():
    project, _geometry, cube, rod, cut, ball = _project()
    with pytest.raises(ValueError, match='give each of them once'):
        project.reorder_solids('G', [ball])
    with pytest.raises(ValueError, match='give each of them once'):
        project.reorder_solids('G', [rod, rod], within=cut)
    with pytest.raises(ValueError, match='is not made of other solids'):
        project.reorder_solids('G', [cube], within=ball)


# ---- the tree ----------------------------------------------------------------

def _rows(window):
    """{solid id: its row}, every solid row in the tree."""
    found = {}

    def walk(item):
        for i in range(item.childCount()):
            child = item.child(i)
            reference = child.data(0, ROLE_REFERENCE)
            if reference is not None and reference[0] == 'solid':
                found[int(reference[2])] = child
            walk(child)

    walk(window.tree.invisibleRootItem())
    return found


def _open(window, pump):
    _, geometry, cube, rod, cut, ball = _project()
    window.add_object('G', geometry)
    pump()
    item = window._item_for_object('G')
    item.setExpanded(True)
    pump()
    for i in range(item.childCount()):
        item.child(i).setExpanded(True)
    rows = _rows(window)
    rows[cut].setExpanded(True)
    pump()
    return geometry, cube, rod, cut, ball


def _drag(window, pump, ids, onto, lower):
    """Select `ids`' rows, take their drag payload, and drop it on the
    upper or lower half of `onto`'s row."""
    rows = _rows(window)
    window.tree.clearSelection()
    for i in ids:
        rows[i].setSelected(True)
    payload = window.tree._solid_payload()
    if payload is None:
        return None
    mime = QMimeData()
    import json

    mime.setData(SOLIDS_MIME, json.dumps(payload).encode('utf-8'))
    window.tree.scrollToItem(rows[onto])
    pump()
    rect = window.tree.visualItemRect(rows[onto])
    y = rect.bottom() - 2 if lower else rect.top() + 2
    event = QDropEvent(QPointF(rect.center().x(), y),
                       Qt.DropAction.MoveAction, mime,
                       Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)
    window.tree.dropEvent(event)
    pump()
    return payload


def test_dragging_a_subtractions_second_solid_first_reverses_the_cut(window,
                                                                     pump):
    _geometry, cube, rod, cut, _ball = _open(window, pump)
    _drag(window, pump, [rod], onto=cube, lower=False)
    assert solids.solid(window.objects['G'], cut).operands == (rod, cube)
    assert window.project.journal[-1] == \
        f"project.reorder_solids('G', [{rod}, {cube}], within={cut})"
    assert 'rod first, the rest cut from it' in \
        window.statusBar().currentMessage()


def test_dragging_a_part_below_another_reorders_the_parts(window, pump):
    _geometry, _cube, _rod, cut, ball = _open(window, pump)
    _drag(window, pump, [cut], onto=ball, lower=True)
    assert [s.id for s in solids.top_level(window.objects['G'])] == [ball, cut]
    assert 'solids in a new order' in window.statusBar().currentMessage()


def test_a_drop_outside_its_siblings_does_nothing(window, pump):
    _geometry, cube, rod, cut, ball = _open(window, pump)
    journal = list(window.project.journal)
    _drag(window, pump, [rod], onto=ball, lower=False)
    assert solids.solid(window.objects['G'], cut).operands == (cube, rod)
    assert [s.id for s in solids.top_level(window.objects['G'])] == [cut, ball]
    assert window.project.journal == journal, 'nothing happened'
    assert 'give each of them once' not in \
        window.statusBar().currentMessage(), \
        'not offered at all, rather than tried and refused'
    rows = _rows(window)
    window.tree.clearSelection()
    rows[rod].setSelected(True)
    import json

    mime = QMimeData()
    mime.setData(SOLIDS_MIME,
                 json.dumps(window.tree._solid_payload()).encode('utf-8'))
    rect = window.tree.visualItemRect(rows[ball])
    assert window.tree._solid_landing(mime, rect.center()) is None, \
        'no landing line among rows it cannot join'


def test_a_pick_across_two_parents_is_not_a_drag(window, pump):
    _geometry, _cube, rod, _cut, ball = _open(window, pump)
    rows = _rows(window)
    window.tree.clearSelection()
    rows[rod].setSelected(True)
    rows[ball].setSelected(True)
    assert window.tree._solid_payload() is None
