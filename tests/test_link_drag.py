"""Dragging an object from one link group into another.

A typed project sorts arriving objects into its groups by itself, and it
guesses: import an FEM geometry and its mode shapes into a Modal Test and
both land in the Basis, because from the outside they satisfy the same
slots the measured pair does. Until this there was no way to say
otherwise — Unlink took an object out of every group and Link merges the
two groups it names, which is the opposite of moving one thing between
them.

The gesture is the obvious one, so what is worth pinning is the parts
that are not: that a refused move is never offered rather than being
taken and complained about, that the file drop the tree already handled
is untouched, and that the project's own rule decides both.
"""

from __future__ import annotations

from conftest import fixture_path
from PySide6.QtCore import QMimeData, QPoint, Qt
from PySide6.QtGui import (
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
)

import visualdynamics
from visualdynamics.core.report import OTHER_SIDE
from visualdynamics.core.shapes import ShapeSet
from visualdynamics.gui.project_tree import OBJECT_MIME


def _drag(*names):
    mime = QMimeData()
    mime.setData(OBJECT_MIME, '\n'.join(names).encode('utf-8'))
    return mime


def _row(window, name):
    """The middle of an object's row, in viewport coordinates."""
    item = window._item_for_object(name)
    rect = window.tree.visualItemRect(item)
    return QPoint(rect.center().x(), rect.center().y())


def _drop(window, names, onto):
    """Deliver a drag of objects the way Qt delivers one."""
    mime = _drag(*names)
    where = onto if isinstance(onto, QPoint) else _row(window, onto)
    window.tree.dragEnterEvent(QDragEnterEvent(
        where, Qt.DropAction.MoveAction, mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    event = QDropEvent(
        where, Qt.DropAction.MoveAction, mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    window.tree.dropEvent(event)
    return event


def _two_groups(window, survey):
    """Two shape sets on one geometry each, linked into two groups."""
    shapes, frfs = survey
    window.add_object('Geometry', visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz')))
    window.add_object('FRF', frfs)
    window.add_object('Shapes', shapes)
    window.add_object('Other', ShapeSet(
        frequency=shapes.frequency.copy(), damping=shapes.damping.copy(),
        shape_matrix=shapes.shape_matrix.copy(),
        coordinate=shapes.coordinate.copy()))
    window.project.link('Geometry', 'Shapes', role='Basis')
    window.project.link('FRF', 'Other')
    window._links_changed()


# ---- the move itself -------------------------------------------------------


def test_the_project_moves_one_object_without_merging_the_groups(
        window, survey):
    """`link` merges, which is right for declaring a relation and wrong
    for moving: linking Shapes to FRF would drag Geometry across with
    it. The move takes the object out first, so only it travels."""
    _two_groups(window, survey)
    window.project.relink('Other', 'Shapes')
    assert window.project.group_of('Other') == ['Geometry', 'Shapes', 'Other']
    assert window.project.group_of('FRF') is None, 'a group of one dissolves'


def test_the_group_it_lands_in_keeps_its_role(window, survey):
    """Dropping something into the Basis joins the Basis. Rebuilding the
    group without carrying the role would quietly demote it, and nothing
    on screen says 'Basis' in words — only the bracket's blue."""
    _two_groups(window, survey)
    window.project.relink('Other', 'Shapes')
    assert window.project.role_of('Other') == 'Basis'
    assert window.project.role_of('Shapes') == 'Basis'


def test_dropping_on_nothing_takes_it_out_of_its_group(window, survey):
    _two_groups(window, survey)
    window.project.relink('Other', None)
    assert window.project.group_of('Other') is None


def test_a_refused_move_changes_nothing(window, survey):
    """A group holds one geometry. The object has already left where it
    was by the time the link is refused, so a refusal that is not undone
    loses the group it came from as well as failing to make the new one."""
    _two_groups(window, survey)
    second = visualdynamics.import_file(fixture_path('plate',
                                                     'geometry.npz'))
    window.add_object('Geometry 2', second)
    window.project.link('Geometry 2', 'Other')
    before = [dict(group) for group in window.project.links]
    try:
        window.project.relink('Geometry 2', 'Shapes')
    except ValueError:
        pass
    else:
        raise AssertionError('two geometries cannot share a group')
    assert window.project.links == before


# ---- the drag, through the tree --------------------------------------------


def test_a_dragged_object_moves_groups(window, pump, survey):
    _two_groups(window, survey)
    _drop(window, ['Other'], 'Shapes')
    pump()
    assert window.linked_group('Other') == ['Geometry', 'Shapes', 'Other']


def test_the_drag_marks_the_gap_the_object_will_land_in(window, pump,
                                                        survey):
    """The mark while dragging is a line in the gap where the canonical
    order will actually put the object — it used to be a box around the
    target, which read as the object going *into* it (Brandon,
    2026-08-30) when a drop only ever makes it a sibling."""
    _two_groups(window, survey)
    # FRF ranks between Geometry and Shapes, so joining their group
    # lands it above Shapes — inside the box the old outline drew
    assert window._landing_of(['FRF'], 'Shapes') == [
        window._item_for_object('Shapes')]
    # out of its group, Other is the last object: the line sits at the
    # end, which the item None stands for
    assert window._landing_of(['Other'], None) == [None]
    # a move that would change nothing is not marked at all
    assert window._landing_of(['Other'], 'FRF') is None

    # and the tree carries the answer while the drag is in the air
    mime = _drag('FRF')
    hover = QDragMoveEvent(
        _row(window, 'Shapes'), Qt.DropAction.MoveAction, mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    window.tree.dragMoveEvent(hover)
    assert window.tree._drop_lines == [window._item_for_object('Shapes')]


def test_the_move_happens_after_the_drop_unwinds(window, pump, survey):
    """Same rule as a dropped file, for the same reason: the drop is
    delivered from inside macOS's drag run loop, and the move reparents
    tree rows and rebuilds their record grids."""
    _two_groups(window, survey)
    _drop(window, ['Other'], 'Shapes')
    assert window.linked_group('Other') == ['FRF', 'Other'], \
        'moved inside the drop'
    pump()
    assert window.linked_group('Other') == ['Geometry', 'Shapes', 'Other']


def test_a_drop_on_blank_space_takes_it_out_of_every_group(
        window, pump, survey):
    """The rest of the tree is empty space, and dropping something on
    nothing is what unlinking looks like as a gesture."""
    _two_groups(window, survey)
    blank = QPoint(200, window.tree.viewport().height() - 3)
    assert window.tree.itemAt(blank) is None, 'the test aimed at a row'
    _drop(window, ['Other'], blank)
    pump()
    assert window.linked_group('Other') is None
    assert window.linked_group('Shapes') == ['Geometry', 'Shapes'], \
        'the group it was not in is untouched'


def test_a_drop_on_a_sub_item_means_its_object(window, pump, survey):
    """The bracket reaches the object's row; asking the user to hit that
    row rather than the Nodes under it is a precision game with no
    purpose."""
    _two_groups(window, survey)
    item = window._item_for_object('Geometry')
    item.setExpanded(True)
    rect = window.tree.visualItemRect(item.child(0))
    _drop(window, ['Other'], QPoint(rect.center().x(), rect.center().y()))
    pump()
    assert 'Other' in window.linked_group('Geometry')


def test_a_refused_move_is_not_outlined_but_still_lands(window, pump,
                                                       survey):
    """The bug that made this look broken for a whole session.

    Refusing the drag reads well — the move is simply not offered — but
    Qt takes an ignored dragMoveEvent to mean the widget is not a drop
    target *there*, so releasing produces no drop at all. No move, no
    message, nothing: from the outside, identical to a feature that does
    not work. A geometry dragged into a group that already had one hit
    exactly this.

    So the target is not outlined, and the drop still lands and answers.
    """
    _two_groups(window, survey)
    window.add_object('Geometry 2', visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz')))
    assert not window._can_move_objects(['Geometry 2'], 'Shapes')

    mime = _drag('Geometry 2')
    where = _row(window, 'Shapes')
    args = (Qt.DropAction.MoveAction, mime, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier)
    hover = QDragMoveEvent(where, *args)
    window.tree.dragMoveEvent(hover)
    assert hover.isAccepted(), 'ignored, so no drop would ever arrive'
    assert window.tree._drop_target is None, 'and so not outlined'

    drop = QDropEvent(where, *args)
    window.tree.dropEvent(drop)
    assert drop.isAccepted()
    pump()
    assert 'one geometry' in window.statusBar().currentMessage()
    assert window.linked_group('Geometry 2') is None, 'and nothing moved'


def test_a_move_that_would_change_nothing_is_not_offered(window, survey):
    """Dragging within one group, or onto itself."""
    _two_groups(window, survey)
    assert not window._can_move_objects(['Shapes'], 'Geometry')
    assert not window._can_move_objects(['Shapes'], 'Shapes')
    assert window._can_move_objects(['Shapes'], 'Other')


def test_asking_whether_a_move_is_allowed_leaves_the_links_alone(
        window, survey):
    """It is answered by trying the move, and it is asked on every mouse
    tick of a drag. Putting the links back is the whole of it."""
    _two_groups(window, survey)
    before = [dict(group) for group in window.project.links]
    for _ in range(5):
        window._can_move_objects(['Other'], 'Shapes')
    assert window.project.links == before


# ---- what must not have broken ---------------------------------------------


def test_only_objects_drag_not_their_parts(window, survey):
    """A link group holds objects. A geometry's Nodes is part of one."""
    _two_groups(window, survey)
    item = window._item_for_object('Geometry')
    item.setExpanded(True)
    assert window.tree.draggable(item)
    assert not window.tree.draggable(item.child(0))


def test_the_tree_still_leaves_a_foreign_drag_alone(window):
    """Everything that is neither a file nor one of our objects still
    carries on to whoever it was meant for."""
    mime = QMimeData()
    mime.setText('not for us')
    event = QDragEnterEvent(
        QPoint(10, 10), Qt.DropAction.CopyAction, mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    window.tree.dragEnterEvent(event)
    assert not event.isAccepted()


def test_the_view_still_never_decides_what_a_drop_means(window):
    """Starting a drag is the half of the item machinery we let it do.
    Deciding what a drop *means* is not — that is what put a file import
    on the third attempt.

    `dragDropMode()` is computed from dragEnabled and acceptDrops rather
    than stored, so enabling drags makes it read DragDrop however it was
    set. What actually keeps the view out is that all three handlers are
    overridden and none of them calls up.
    """
    from PySide6.QtWidgets import QAbstractItemView

    from visualdynamics.gui.project_tree import ProjectTree

    assert window.tree.dragDropMode() != \
        QAbstractItemView.DragDropMode.InternalMove
    for handler in ('dragEnterEvent', 'dragMoveEvent', 'dropEvent'):
        assert handler in vars(ProjectTree)
        assert 'super()' not in _body(getattr(ProjectTree, handler))
    assert window.tree.dragEnabled()
    assert window.tree.acceptDrops()
    assert window.tree.viewport().acceptDrops()


def _body(function):
    import inspect

    return inspect.getsource(function)


# ---- the project type's own named groups -----------------------------------


def _modal_fem_pair(window, pump, survey):
    """The reported case: a model's geometry and its mode shapes
    imported into a Modal Test, which sorts both into the Basis — from
    the outside they satisfy the Basis slots exactly as well as
    measured objects would. Returns the other side's first gray slot:
    that side has no name (Brandon, 2026-09-02), so its slots are keyed
    by OTHER_SIDE, which is also the tag a drop reads."""
    shapes, _frfs = survey
    window.set_project_type('Modal Test')
    window.add_object('FEM Geometry', visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz')))
    window.add_object('FEM Shapes', shapes)
    pump()
    assert window.link_role('FEM Geometry') == 'Basis', 'the type guessed'
    return window._placeholder_items[OTHER_SIDE][0]


def _slot_point(window, item):
    rect = window.tree.visualItemRect(item)
    return QPoint(rect.center().x(), rect.center().y())


def test_a_gray_slot_is_a_drop_target(window, pump, survey):
    """The type's empty group is drawn as gray slots and nothing else,
    so the slots are what there is to drop on. Reading them as no target
    was the first version, and dropping on the skeleton did nothing."""
    slot = _modal_fem_pair(window, pump, survey)
    assert window.tree._target_at(_slot_point(window, slot)) == ('slot', OTHER_SIDE)


def test_dropping_a_pair_on_an_empty_group_makes_them_that_group(
        window, pump, survey):
    """An empty role has no group to join, so the objects dropped on it
    become one."""
    slot = _modal_fem_pair(window, pump, survey)
    _drop(window, ['FEM Geometry', 'FEM Shapes'],
          _slot_point(window, slot))
    pump()
    assert window.linked_group('FEM Geometry') == ['FEM Geometry',
                                                   'FEM Shapes']
    assert window.link_role('FEM Geometry') is None, 'no longer the Basis'


def test_the_type_does_not_undo_the_drag(window, pump, survey):
    """The rules run again on every arrival, and they are the rules that
    guessed wrong. Without a declared role, importing the measured FRF
    pulls the model pair straight back into the Basis: they are still
    the project's first geometry and first shape set, which is all the
    type has to go on."""
    _shapes, frfs = survey
    slot = _modal_fem_pair(window, pump, survey)
    _drop(window, ['FEM Geometry', 'FEM Shapes'], _slot_point(window, slot))
    pump()
    window.add_object('FRF', frfs)
    pump()
    assert window.link_role('FEM Geometry') is None, 'not pulled back'
    assert window.linked_group('FEM Geometry') == ['FEM Geometry',
                                                   'FEM Shapes']
    assert 'FRF' not in (window.linked_group('FEM Geometry') or [])


def test_a_role_that_holds_a_group_is_joined_by_joining_it(
        window, pump, survey):
    """Once a named group has real members its slots sit under the same
    bracket, and there is only one thing a drop there can mean."""
    slot = _modal_fem_pair(window, pump, survey)
    _drop(window, ['FEM Geometry', 'FEM Shapes'], _slot_point(window, slot))
    pump()
    window.add_object('Loose', visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz')))
    pump()
    # a second geometry cannot join a group that has one
    assert not window._can_move_objects(['Loose'], ('slot', OTHER_SIDE))


def test_a_declared_role_rides_the_project_file(window, pump, survey,
                                                tmp_path):
    """It is a decision the user made, so it outlives the session — and
    reloading into rules that would guess again is exactly when it
    matters. The other side carries no name, so what rides the file is
    the group itself, and the Basis it is not."""
    from visualdynamics.io import load

    slot = _modal_fem_pair(window, pump, survey)
    _drop(window, ['FEM Geometry', 'FEM Shapes'], _slot_point(window, slot))
    pump()
    path = tmp_path / 'groups.vdyn'
    window.project.save(path)
    back = load(path)
    assert back.role_of('FEM Geometry') is None
    assert back.group_of('FEM Geometry') == ['FEM Geometry', 'FEM Shapes']


# ---- the skeleton is per group, not per project ----------------------------
#
# All three of these were one bug reported as three: the slots and the
# auto-sorting both counted objects by class across the *whole project*,
# so which group anything was in made no difference to either.


def test_a_slot_is_filled_by_an_object_in_its_own_group(window, pump,
                                                        survey):
    """Reported as "I unlinked them and now I don't see the skeleton's
    basis geometry or mode shapes". They were never redrawn: one
    geometry existed somewhere in the project, so the Basis group's
    Geometry slot read as filled while the Basis group was empty."""
    slot = _modal_fem_pair(window, pump, survey)
    _drop(window, ['FEM Geometry', 'FEM Shapes'], _slot_point(window, slot))
    pump()
    assert window.project.placed() == {}, 'the other side has no name'
    assert window.linked_group('FEM Geometry') == ['FEM Geometry',
                                                   'FEM Shapes']
    measured = [item.text(0) for item in window._placeholder_items[
        'Basis']]
    assert 'Geometry' in measured, 'no measured geometry, so the slot stands'
    assert 'Shape Set' in measured


def test_a_group_is_built_up_one_drag_at_a_time(window, pump, survey):
    """Reported as "I was never able to get them into the other group".
    A group used to need two members before it could exist at all, so
    dropping the first object only unlinked it, and dropping the second
    unlinked that — they never met."""
    slot = _modal_fem_pair(window, pump, survey)
    _drop(window, ['FEM Geometry'], _slot_point(window, slot))
    pump()
    assert window.linked_group('FEM Geometry') == ['FEM Geometry']
    assert window.link_role('FEM Geometry') is None
    _drop(window, ['FEM Shapes'],
          _slot_point(window, window._placeholder_items[OTHER_SIDE][0]))
    pump()
    assert window.linked_group('FEM Shapes') == ['FEM Geometry', 'FEM Shapes']


def test_a_lone_object_still_lands_in_a_group(window, pump, survey):
    """Counting per group is only right if one object is placed as
    readily as two: an FRF that sits in no group leaves its own gray
    slot showing beside it, which is the skeleton denying an object
    that is plainly in the tree."""
    _shapes, frfs = survey
    slot = _modal_fem_pair(window, pump, survey)
    _drop(window, ['FEM Geometry', 'FEM Shapes'], _slot_point(window, slot))
    pump()
    window.add_object('FRF', frfs)
    pump()
    assert window.project.placed()['Basis'] == ['FRF']
    assert 'FRF' not in [item.text(0) for item in
                         window._placeholder_items['Basis']]


def test_a_legacy_side_key_reads_as_the_role_it_meant(
        window, pump, survey):
    """Files written while 'side' existed carry it in their links, and
    files written while the FEM role existed carry that. A measured
    side said nothing the Basis role does not, and the model side and
    the FEM role both named what is now simply the other group — so
    only the Basis survives, and everything else reads as nothing
    (Brandon, 2026-08-30; the role retired 2026-09-02)."""
    from visualdynamics import Project

    project = Project('t', {}, links=[
        {'members': ['a', 'b'], 'role': None, 'side': 'fem'},
        {'members': ['c', 'd'], 'role': 'Basis', 'side': 'experimental'},
        {'members': ['e', 'f'], 'role': 'FEM'},
        {'members': ['g', 'h'], 'side': 'experimental'}])
    assert project.links == [{'members': ['a', 'b'], 'role': None},
                             {'members': ['c', 'd'], 'role': 'Basis'},
                             {'members': ['e', 'f'], 'role': None},
                             {'members': ['g', 'h'], 'role': 'Basis'}]
    assert project.placed() == {'Basis': ['c', 'd', 'g', 'h']}


def test_a_lone_geometry_drags_into_the_model_group(window, pump, survey):
    """The smallest project there is: one geometry, typed Modal Test.
    The type calls it the measured geometry — it has nothing else to go
    on — and dragging it into the model group has to be enough to say
    otherwise, with no second object involved anywhere.
    """
    del survey
    window.add_object('FEM Geometry', visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz')))
    window.set_project_type('Modal Test')
    pump()
    assert window.project.placed() == {'Basis': ['FEM Geometry']}
    _drop(window, ['FEM Geometry'],
          _slot_point(window, window._placeholder_items[OTHER_SIDE][0]))
    pump()
    assert window.project.placed() == {}, 'out of the Basis, into the other'
    assert window.linked_group('FEM Geometry') == ['FEM Geometry']
    assert 'Geometry' in [item.text(0) for item
                          in window._placeholder_items['Basis']]


def test_the_drag_survives_qt_delivering_it(window, pump, survey):
    """Every other test here calls the handlers. Qt does not: it sends
    the events to the *viewport*, which sits inset from the tree by its
    frame and header, and a handler reading those positions as tree
    coordinates would aim a row or two off — right in the tests and
    wrong in the application, which is the one gap the rest cannot see.

    Aimed at an object row with different rows above and below it, so an
    offset of the frame's own size lands somewhere that answers
    differently. Aiming at a slot could not fail: its neighbor is
    another slot of the same group, and both give the same target.
    """
    from PySide6.QtWidgets import QApplication

    _two_groups(window, survey)
    window.show()
    pump()
    viewport = window.tree.viewport()
    assert viewport.mapTo(window.tree, QPoint(0, 0)) != QPoint(0, 0), \
        'the offset this is about'

    window.tree.collapseAll()
    window.test_item.setExpanded(True)
    point = _row(window, 'FRF')
    offset = viewport.mapTo(window.tree, QPoint(0, 0))
    missed = QPoint(point.x() + offset.x(), point.y() + offset.y())
    assert window.tree._target_at(missed) != 'FRF', \
        'the test aimed where reading the offset wrongly is undetectable'

    mime = _drag('Shapes')
    args = (Qt.DropAction.MoveAction, mime, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier)
    for event in (QDragEnterEvent(point, *args), QDragMoveEvent(point, *args)):
        QApplication.sendEvent(viewport, event)
        assert event.isAccepted(), 'Qt-delivered drag was refused'
    assert window.tree._drop_target == 'FRF', 'aimed at the wrong row'
    drop = QDropEvent(point, *args)
    QApplication.sendEvent(viewport, drop)
    assert drop.isAccepted()
    pump()
    assert 'Shapes' in window.linked_group('FRF')


def test_a_drop_that_changes_nothing_says_so(window, pump, survey):
    """Reported twice as "it doesn't move". A tree that simply does not
    move is the one outcome there is no way to act on: the status bar
    now separates a miss from a refusal from a move."""
    _two_groups(window, survey)
    _drop(window, ['Shapes'], 'Geometry')     # already in that group
    pump()
    assert 'already' in window.statusBar().currentMessage()


def test_a_drop_the_project_refuses_says_why(window, pump, survey):
    _two_groups(window, survey)
    window.add_object('Geometry 2', visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz')))
    pump()
    window._apply_move(['Geometry 2'], 'Shapes')
    assert 'one geometry' in window.statusBar().currentMessage()


# ---- the brackets say what is linked ---------------------------------------


def _rows_of_spans(window):
    """[(rows, bold)] — which tree rows each bracket actually covers."""
    index = {id(window.test_item.child(i)): i
             for i in range(window.test_item.childCount())}
    return [(sorted(index[id(item)] for item in items
                    if item is not None and id(item) in index), bold)
            for _color, items, bold in window.tree.link_spans]


def _kinds(window):
    from visualdynamics.gui.main_window import ROLE_REFERENCE

    return [(window.test_item.child(i).data(0, ROLE_REFERENCE) or ('?',))[0]
            for i in range(window.test_item.childCount())]


def test_a_placeholder_of_the_same_name_is_not_the_object(window, pump,
                                                          survey):
    """The root of the wandering brackets. It needs two things that are
    both what a person actually does: a model geometry called what the
    model slot is called, and the objects imported *before* the type is
    set.

    Rows were found by their text alone. Setting the type lays the gray
    slots down before anything is linked, so the slot named "Geometry"
    came to sit above the object of that name — and `_item_for_object`
    then answered with the slot. The reorder moved the slot instead of
    the object, leaving the geometry below its own shape set, and
    `_paint_links` drew the group's bracket around a row that was not
    in the group. Every slot is named for its type now (2026-09-02), so
    the collision is the ordinary case rather than a coincidence.
    """
    from visualdynamics.gui.main_window import ROLE_REFERENCE

    shapes, _frfs = survey
    window.add_object('Geometry', visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz')))
    window.add_object('Shapes', shapes)
    window.set_project_type('Modal Test')       # after, as a person does
    pump()

    slots = [item.text(0) for item in window._placeholder_items[OTHER_SIDE]]
    assert 'Geometry' in slots, 'the name collision this is about'
    assert window._item_for_object('Geometry').data(
        0, ROLE_REFERENCE)[0] == 'object'
    rows = [window.test_item.child(i).text(0)
            for i in range(window.test_item.childCount())
            if (window.test_item.child(i).data(0, ROLE_REFERENCE)
                or ('?',))[0] == 'object']
    assert rows == ['Geometry', 'Shapes'], \
        'a geometry reads above the shapes on it'


def test_every_bracket_covers_a_run_of_rows(window, pump, survey):
    """A bracket is drawn from its topmost member to its lowest, so a
    group whose rows are not consecutive reaches across rows that belong
    to somebody else — which is what "the bracket shifted up into the
    Basis bracket" looks like."""
    slot = _modal_fem_pair(window, pump, survey)
    for spans in (_rows_of_spans(window),):
        for rows, _bold in spans:
            assert rows == list(range(rows[0], rows[-1] + 1)), rows
    _drop(window, ['FEM Geometry', 'FEM Shapes'], _slot_point(window, slot))
    pump()
    spans = _rows_of_spans(window)
    for rows, _bold in spans:
        assert rows == list(range(rows[0], rows[-1] + 1)), rows
    tops = [rows[0] for rows, _bold in spans]
    assert tops == sorted(tops), 'brackets are listed top-down'
    assert len({row for rows, _bold in spans for row in rows}) == \
        sum(len(rows) for rows, _bold in spans), 'brackets overlap'


def test_a_group_keeps_its_place_when_it_empties(window, pump, survey):
    """Drag the model pair across and the Basis group holds nothing but
    slots. It still reads first, because that is the order the type
    lists its roles in — the model group used to leapfrog to the top of
    the tree, taking its bracket with it, and read as the Basis."""
    slot = _modal_fem_pair(window, pump, survey)
    _drop(window, ['FEM Geometry', 'FEM Shapes'], _slot_point(window, slot))
    pump()
    spans = _rows_of_spans(window)
    measured = next(rows for rows, bold in spans if bold)
    model = next(rows for rows, bold in spans if not bold)
    assert measured[-1] < model[0], 'the measured group reads first'
    assert _kinds(window)[model[0]] == 'object'
    assert all(_kinds(window)[row] == 'placeholder' for row in measured)
