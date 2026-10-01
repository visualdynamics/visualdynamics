"""Tie: a patch of elements tied rigidly to the part under it.

A bolted joint had been two clicks per rigid link in add mode — 148 for
the BARC. Brandon (2026-09-26): select the elements under a washer, one
at a time or several, and tie them to the part below in one step.
`mesh.tie` is the rule; `Project.tie_elements` records it; the Elements
edit bar's Tie button offers it for a selection.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import edit_category, select_objects

import visualdynamics
from visualdynamics import fem, mesh
from visualdynamics.core.fem import RIGID


def _floor_and_foot():
    """A floor plate and, 0.25 above it, a smaller foot plate: nothing
    shared, as a bolted foot's mid-surface is not the floor's."""
    geometry = mesh.assemble(
        mesh.plane((0, 0, 0), (4, 0, 0), (0, 4, 0), 1, 'floor'),
        mesh.plane((1, 1, 0.25), (2, 0, 0), (0, 2, 0), 1, 'foot'))
    aluminum = fem.material('6061-T6')
    geometry.block_properties = {1: fem.BlockProperties(aluminum, 0.01),
                                 2: fem.BlockProperties(aluminum, 0.01)}
    return geometry


def test_a_patch_ties_to_the_nearest_nodes_of_a_block():
    geometry = _floor_and_foot()
    patch = geometry.elements_in('foot')[:1]             # one element
    found = mesh.tie(geometry, patch, 'floor')
    assert found['links'] == 4 and found['shared'] == 0
    assert geometry.block_name[-1] == 'ties'
    assert geometry.block_properties[found['block']].material is RIGID
    links = geometry.elements_in('ties')
    rows = np.flatnonzero(np.isin(geometry.elem_id, links))
    for row in rows:
        leader, follower = geometry.elem_conn[row]
        a, b = geometry.node_xyz[geometry.node_index([leader, follower])]
        assert np.allclose(a[:2], b[:2]) and np.isclose(b[2] - a[2], 0.25), \
            'each foot node to the floor node straight under it'


def test_a_second_tie_joins_the_rigid_block_and_shared_nodes_are_left():
    geometry = _floor_and_foot()
    foot = geometry.elements_in('foot')
    first = mesh.tie(geometry, foot[:1], 'floor')
    second = mesh.tie(geometry, foot[1:2], 'floor')
    assert second['block'] == first['block'], 'one block of ties'
    assert second['links'] == 4
    floor = geometry.elements_in('floor')
    with pytest.raises(ValueError, match='every node is shared'):
        mesh.tie(geometry, floor[:1], floor[:2])


def test_ties_join_the_rigid_block_already_there_whatever_its_name():
    """The demo's links are in 'bolts'; a tie made in the app on its
    project joins them rather than starting a block of its own."""
    geometry = _floor_and_foot()
    foot = geometry.elements_in('foot')
    mesh.tie(geometry, foot[:1], 'floor', block='bolts')
    found = mesh.tie(geometry, foot[1:2], 'floor')
    assert geometry.block_name[list(geometry.block_id).index(
        found['block'])] == 'bolts'
    assert 'ties' not in geometry.block_name


def test_a_patch_ties_to_a_second_patch():
    geometry = _floor_and_foot()
    floor = geometry.elements_in('floor')
    found = mesh.tie(geometry, geometry.elements_in('foot')[:1], floor[5:6])
    assert found['links'] == 4
    targets = {int(geometry.elem_conn[row][0]) for row in np.flatnonzero(
        np.isin(geometry.elem_id, geometry.elements_in('ties')))}
    corners = set(geometry.elem_conn[int(np.flatnonzero(
        geometry.elem_id == floor[5])[0])].tolist())
    assert targets <= corners, 'only the second patch leads'


def test_ties_refuse_a_block_that_is_not_rigid():
    geometry = _floor_and_foot()
    with pytest.raises(ValueError, match='not rigid'):
        mesh.tie(geometry, geometry.elements_in('foot')[:1], 'floor',
                 block='floor')


def test_the_tied_model_solves_as_one_piece():
    geometry = _floor_and_foot()
    with pytest.raises(ValueError, match='2 disconnected pieces'):
        fem.Model.from_geometry(geometry)
    mesh.tie(geometry, geometry.elements_in('foot'), 'floor')
    assert len(fem.Model.from_geometry(geometry).pieces()) == 1


def test_the_project_records_a_tie():
    project = visualdynamics.Project('p')
    project.add('Plate', _floor_and_foot())
    patch = project['Plate'].elements_in('foot')[:2]
    found = project.tie_elements('Plate', patch, 'floor')
    assert found['links'] == 6
    assert project.journal[-1] == (
        f"project.tie_elements('Plate', {patch}, 'floor')")


def _edit_elements(window, pump):
    window.add_object('Geometry', _floor_and_foot())
    select_objects(window, pump, 'Geometry')
    edit_category(window, pump, 'Quads')
    return window.objects['Geometry']


def _select_rows(window, pump, rows):
    from PySide6.QtCore import QItemSelectionModel

    model = window.table.model()
    window.table.clearSelection()
    for row in rows:
        window.table.selectionModel().select(
            model.index(row, 0),
            QItemSelectionModel.SelectionFlag.Select
            | QItemSelectionModel.SelectionFlag.Rows)
    pump()


def test_tie_is_on_the_edit_bar_only_for_a_selection(window, pump):
    plate = _edit_elements(window, pump)
    assert not window.tie_action.isVisible(), 'nothing selected'
    _select_rows(window, pump, [16])
    assert window.tie_action.isVisible()
    window.set_add_mode(True)
    pump()
    assert not window.tie_action.isVisible(), 'add mode builds, not selects'
    window.set_add_mode(False)
    assert plate.num_nodes == 34


def test_tie_to_a_block_from_its_menu(window, pump, monkeypatch):
    plate = _edit_elements(window, pump)
    foot_rows = [int(np.flatnonzero(plate.elem_id == e)[0])
                 for e in plate.elements_in('foot')]
    _select_rows(window, pump, foot_rows[:2])
    popped = []
    monkeypatch.setattr(window, '_pop_menu', popped.append)
    window.tie_action.trigger()
    labels = [action.text() for action in popped[0].actions()]
    assert labels == ['Tie to the nearest nodes of', 'floor', 'foot', '',
                      'A second selection…']
    next(a for a in popped[0].actions() if a.text() == 'floor').trigger()
    pump()
    assert len(plate.elements_in('ties')) == 6
    assert 'tied 2 elements to floor — 6 rigid links in ties' in \
        window.statusBar().currentMessage()
    popped.clear()
    _select_rows(window, pump, foot_rows[2:3])
    window.tie_action.trigger()
    assert 'ties' not in [a.text() for a in popped[0].actions()], \
        'a rigid block is not something to tie to'


def test_tie_to_a_second_selection(window, pump, monkeypatch):
    plate = _edit_elements(window, pump)
    foot = plate.elements_in('foot')
    foot_row = int(np.flatnonzero(plate.elem_id == foot[0])[0])
    _select_rows(window, pump, [foot_row])
    popped = []
    monkeypatch.setattr(window, '_pop_menu', popped.append)
    window.tie_action.trigger()
    popped[0].actions()[-1].trigger()
    pump()
    assert window.tie_action.isVisible() and window.tie_action.isChecked(), \
        'held, waiting for the second patch'
    assert 'select the elements to tie them to' in \
        window.statusBar().currentMessage()
    _select_rows(window, pump, [5])
    window.tie_action.trigger()
    pump()
    assert not window.tie_action.isChecked()
    assert len(plate.elements_in('ties')) == 4
    assert window.project.journal[-1] == (
        f"project.tie_elements('Geometry', [{foot[0]}], "
        f'[{int(plate.elem_id[5])}])')
