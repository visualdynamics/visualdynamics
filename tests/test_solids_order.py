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


def _cone_shell():
    """Brandon's cone (2026-10-27): an outer cone with an inner one
    subtracted, the same height, so the inner lies wholly inside."""
    project = visualdynamics.Project()
    geometry = visualdynamics.Geometry([], [])
    geometry.define_units('m')
    project.add('G', geometry)
    outer = project.add_solid_cone('G', (0, 0, 0), 0.6096, 0.1016, 2.032,
                                   name='outer')
    inner = project.add_solid_cone('G', (0, 0, 0), 0.5842, 0.0762, 2.032,
                                   name='inner')
    cone = project.combine_solids('G', 'subtract', [outer, inner], name='Cone')
    return project, geometry, outer, inner, cone


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


def test_an_order_that_leaves_nothing_is_refused():
    """The inner cone first would have the outer cut from it, which leaves
    nothing: an empty solid with nothing to draw or mesh, and every
    redraw raised."""
    project, geometry, outer, inner, cone = _cone_shell()
    journal = list(project.journal)
    with pytest.raises(ValueError, match=(
            f'^solid {inner} cannot come first: taking solid {outer} from '
            f'solid {inner} leaves nothing$')):
        project.reorder_solids('G', [inner, outer], within=cone)
    assert solids.solid(geometry, cone).operands == (outer, inner)
    assert project.journal == journal
    assert len(solids.shape_triangles(solids.shape(geometry, cone)))


def test_a_boolean_that_leaves_nothing_is_not_made():
    project, geometry, outer, inner, cone = _cone_shell()
    project.separate_solids('G', [cone])
    before = list(geometry.solids)
    with pytest.raises(ValueError, match=(
            f'^the subtract cannot be made: taking solid {outer} from solid '
            f'{inner} leaves nothing$')):
        project.combine_solids('G', 'subtract', [inner, outer])
    assert geometry.solids == before
    apart = project.add_solid_sphere('G', (5, 0, 0), 0.5)
    with pytest.raises(ValueError, match=f'solids {outer}, {apart} share nothing'):
        project.combine_solids('G', 'intersect', [outer, apart])


def test_an_empty_shape_has_no_triangles():
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    empty = TopoDS_Compound()
    BRep_Builder().MakeCompound(empty)
    assert solids.shape_triangles(empty).shape == (0, 3, 3)


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


def test_a_drag_that_would_leave_nothing_is_refused_and_said(window, pump):
    _shell, geometry, outer, inner, cone = _cone_shell()
    window.add_object('G', geometry)
    pump()
    item = window._item_for_object('G')
    item.setExpanded(True)
    for i in range(item.childCount()):
        item.child(i).setExpanded(True)
    pump()
    _rows(window)[cone].setExpanded(True)
    pump()
    _drag(window, pump, [inner], onto=outer, lower=False)
    assert solids.solid(window.objects['G'], cone).operands == (outer, inner)
    assert window.statusBar().currentMessage() == (
        f'G: solid {inner} cannot come first: taking solid {outer} from '
        f'solid {inner} leaves nothing')
    select = _rows(window)
    assert inner in select and outer in select, 'the rows are still there'


def test_a_pick_across_two_parents_is_not_a_drag(window, pump):
    _geometry, _cube, rod, _cut, ball = _open(window, pump)
    rows = _rows(window)
    window.tree.clearSelection()
    rows[rod].setSelected(True)
    rows[ball].setSelected(True)
    assert window.tree._solid_payload() is None
