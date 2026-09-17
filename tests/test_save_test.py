"""Saving the whole test as one .vdyn file, and getting it back.

A test file holds every object under the user's own name, in order, plus
what the test itself carried: its name and the active geometry. Loading
one into an untouched window puts everything back exactly; loading into a
project already underway just adds the objects, suffixing collisions.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics import io


@pytest.fixture
def project(window):
    """A window holding a geometry and unit-defined time data."""
    geometry = visualdynamics.import_file(fixture_path('plate', 'geometry.unv'))
    geometry.define_units('m')
    time = visualdynamics.import_file(fixture_path('plate', 'time.npz'))
    time.define_units({0: 'g'})
    window.add_object('Geometry', geometry)
    window.add_object('My Time Data', time)
    window.set_active_geometry('Geometry')
    window.test_item.setText(0, 'Wing Survey')
    return window


def test_a_test_file_round_trips_names_order_and_units(project, tmp_path):
    path = tmp_path / 'survey.vdyn'
    io.save_test(path, 'Wing Survey', dict(project.objects),
                 active_geometry='Geometry')
    back = io.load(str(path))
    assert isinstance(back, io.TestContents)
    assert back.name == 'Wing Survey'
    assert back.active_geometry == 'Geometry'
    assert list(back) == ['Geometry', 'My Time Data']
    original = project.objects['My Time Data']
    assert back['My Time Data'].ordinate_unit[0] == 'g'
    assert np.allclose(back['My Time Data'].ordinate, original.ordinate)


def test_an_untouched_window_adopts_the_saved_test(
        project, window_factory, tmp_path, pump):
    path = tmp_path / 'survey.vdyn'
    io.save_test(path, 'Wing Survey', dict(project.objects),
                 active_geometry='Geometry')
    fresh = window_factory()
    fresh.import_path(str(path))
    pump()
    assert list(fresh.objects) == ['Geometry', 'My Time Data']
    assert fresh.test_item.text(0) == 'Wing Survey'
    assert fresh.active_geometry == 'Geometry'


def test_a_project_underway_keeps_its_own_identity(project, tmp_path, pump):
    """Importing a test into a non-empty project adds its objects — it does
    not silently replace the test's name or the active geometry."""
    path = tmp_path / 'survey.vdyn'
    io.save_test(path, 'Someone Elses Survey', dict(project.objects))
    project.test_item.setText(0, 'Mine')
    project.import_path(str(path))
    pump()
    assert project.test_item.text(0) == 'Mine'
    assert 'Geometry (2)' in project.objects, 'collisions suffix as ever'


def test_saving_via_the_test_item_saves_the_test(
        project, tmp_path, pump, monkeypatch):
    """Save Selected with the test root current is saving the test — the
    old placeholder dialog is gone."""
    from PySide6.QtWidgets import QFileDialog

    path = tmp_path / 'via_menu.vdyn'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: (str(path), '')))
    project.tree.setCurrentItem(project.test_item)
    project.save_selected()
    pump()
    back = io.load(str(path))
    assert isinstance(back, io.TestContents)
    assert back.name == 'Wing Survey'
    assert list(back) == ['Geometry', 'My Time Data']


def test_a_single_object_file_still_loads_as_itself(tmp_path):
    """The container is additive: one-object .vdyn files are untouched."""
    from visualdynamics.core.channel_table import ChannelTable

    table = io.load(fixture_path('plate', 'channel_table.vdyn'))
    assert isinstance(table, ChannelTable)
    assert not isinstance(table, io.TestContents)


def test_a_project_file_dropped_on_the_tree_loads(
        project, window_factory, tmp_path, pump):
    """Dragging a saved .vdyn project from Finder onto the tree is the
    shortest path back into it — same route as any dropped file."""
    path = tmp_path / 'Project.vdyn'
    io.save_test(path, 'Wing Survey', dict(project.objects),
                 active_geometry='Geometry')
    fresh = window_factory()
    fresh._import_after_drop([str(path)])
    pump(12)
    assert list(fresh.objects) == ['Geometry', 'My Time Data']
    assert fresh.test_item.text(0) == 'Wing Survey'
    assert fresh.active_geometry == 'Geometry'


# ---- a project imported into itself ---------------------------------------
#
# Dragging a project out and dropping it back on the window imported it
# again, so a real project came to hold four copies of itself. That route
# is closed (`SELF_MIME`), but the damage it did on the way through was
# in code that any repeated import reaches, and this is that code.


def _modal_project(window, pump):
    """A Modal Test project with both sides populated and linked."""
    from visualdynamics.core.shapes import ShapeSet

    def geometry():
        return visualdynamics.Geometry(
            node_id=[1, 2, 3], node_xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            length_unit='m')

    def shapes():
        return ShapeSet(np.array([10.0, 20.0]), np.array([0.01, 0.01]),
                        ['1Z+', '2Z+', '3Z+'], np.ones((2, 3)))

    window.set_project_type('Modal Test')
    window.add_object('FEM Geometry', geometry())
    window.add_object('FEM Modes', shapes())
    window.add_object('Experimental Geometry', geometry())
    window.add_object('Experimental Modes', shapes())
    pump()
    return window


def test_an_object_ends_up_in_exactly_one_link_group(window, pump):
    """The invariant everything downstream assumes: `group_of` answers
    with the first match, and a bracket is painted over a run of rows
    that has to be contiguous.

    **The later group wins.** The order is the order things were
    declared in, and the last word is the most informed: an imported
    file's own structure arrives after the guess the type rules made
    about its objects one at a time, as each was added.
    """
    project = _modal_project(window, pump).project
    project.links = [
        {'members': ['FEM Geometry', 'FEM Modes'], 'role': 'FEM'},
        {'members': ['FEM Modes', 'Experimental Modes'], 'role': 'Basis'}]
    project.absorb_links([])            # any mutation prunes
    holders = [group['members'] for group in project.links]
    assert [m for m in holders[0] if m == 'FEM Modes'] == []
    assert 'FEM Modes' in holders[1], 'the later group keeps it'
    counted = Counter(name for group in project.links
                      for name in group['members'])
    assert [name for name, n in counted.items() if n > 1] == []


def test_a_group_emptied_by_that_goes_unless_it_has_a_role(window, pump):
    """An emptied named group keeps its place in the tree; a group
    with no role and no members is nothing at all."""
    project = _modal_project(window, pump).project
    project.links = [
        {'members': ['FEM Geometry', 'FEM Modes'], 'role': None},
        {'members': ['FEM Geometry', 'FEM Modes'], 'role': 'FEM'}]
    project.absorb_links([])
    assert [group['role'] for group in project.links] == ['FEM']


def test_no_row_is_lost_when_the_links_overlap(window, pump):
    """The bug this is here for lost rows silently.

    The tree orders its rows by walking the link groups, so an object
    named in two of them put its row in that order twice — the
    positions then ran past the end of the tree, and Qt's `insertChild`
    past the end **drops a child that has already been taken out**,
    without a word. Seven rows went missing from a real project that
    still held all forty-five of its objects, in the tree and in every
    save made afterwards.

    The links are set behind the project's back on purpose: the
    invariant above is what stops this arising, and this is the second
    line, for the failure that is silent data loss.
    """
    _modal_project(window, pump)
    # A *complete* project, which the real one was: with gray slots
    # still showing there are spare rows for the doubled positions to
    # land on, and the loss does not show. Forty-five objects and
    # nothing missing is exactly the case with no slack in it.
    window.set_project_type(None)
    pump()
    window.project.links = [
        {'members': ['FEM Geometry', 'FEM Modes'], 'role': 'FEM'},
        {'members': ['FEM Modes', 'Experimental Modes'], 'role': None},
        {'members': ['Experimental Modes', 'Experimental Geometry'],
         'role': 'Basis'}]
    window._reorder_tree()
    pump()
    rows = [window.test_item.child(i).text(0)
            for i in range(window.test_item.childCount())]
    assert [name for name in window.objects if name not in rows] == [], (
        f'{len(window.objects)} objects, {len(rows)} rows')


def test_a_project_imported_into_itself_keeps_every_row(window, pump,
                                                        tmp_path):
    """End to end, three times over, which is what the real project had
    been through."""
    _modal_project(window, pump)
    path = tmp_path / 'modal.vdyn'
    io.save_test(path, 'Modal', dict(window.objects),
                 active_geometry=window.active_geometry,
                 project_type=window.project_type, links=window.links)
    for _ in range(3):
        window.import_paths([str(path)])
        pump()
        rows = [window.test_item.child(i).text(0)
                for i in range(window.test_item.childCount())]
        assert [name for name in window.objects if name not in rows] == []
        counted = Counter(name for group in window.links
                          for name in group['members'])
        assert [name for name, n in counted.items() if n > 1] == []


def test_an_emptied_project_reimported_gets_its_basis_back(window, pump,
                                                          tmp_path):
    """Delete everything, import the same file again: the file's Basis
    is the Basis. The type rules had placed the arrivals into a Basis
    of their own before the file's groups were read, and that guess,
    counted as a Basis already held, demoted the file's group to no
    role (Brandon, 2026-09-03)."""
    _modal_project(window, pump)
    basis = list(window.project.role_group('Basis')['members'])
    path = tmp_path / 'modal.vdyn'
    io.save_test(path, 'Modal', dict(window.objects),
                 active_geometry=window.active_geometry,
                 project_type=window.project_type, links=window.links)
    window.tree.clearSelection()
    for name in list(window.objects):
        window._item_for_object(name).setSelected(True)
    window.delete_selected()
    pump()
    assert not window.objects and window.links == []
    window.import_paths([str(path)])
    pump()
    assert window.project.role_group('Basis') is not None, 'no Basis'
    assert set(window.project.role_group('Basis')['members']) == set(basis)
    assert [g['role'] for g in window.links].count('Basis') == 1


def test_the_copy_links_to_its_own_geometry(window, pump, tmp_path):
    """The type's rules place an arriving object the moment it arrives,
    with nothing but its type to go on; the file says where it actually
    belonged."""
    _modal_project(window, pump)
    path = tmp_path / 'modal.vdyn'
    io.save_test(path, 'Modal', dict(window.objects),
                 active_geometry=window.active_geometry,
                 project_type=window.project_type, links=window.links)
    window.import_paths([str(path)])
    pump()
    arrived = window.linked_group('FEM Modes (2)')
    assert arrived is not None
    assert 'FEM Geometry (2)' in arrived
    assert 'FEM Geometry' not in arrived, 'not to the original'
