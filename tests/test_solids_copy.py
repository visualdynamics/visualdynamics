"""Copy and paste of a geometry's solids: Cmd/Ctrl+C on solid rows puts
them on the clipboard, Cmd/Ctrl+V with a geometry picked copies them
into it, recipe and all, and a script does the same with
`Project.copy_solids`.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core import solids
from visualdynamics.core.geometry import Geometry

IN = 0.0254


def _peanut(name='peanut'):
    """A block with two overlapping cylinders unioned and cut from it:
    five solids, one part."""
    geometry = Geometry(node_id=[], node_xyz=np.zeros((0, 3)), elem_id=[],
                        elem_type=[], elem_conn=[], length_unit='in')
    block = solids.add_block(geometry, (0, 0, 0), (10 * IN, 10 * IN, 3 * IN),
                             name='slab')
    left = solids.add_cylinder(geometry, (-1.5 * IN, 0, 0), 3 * IN, 3 * IN)
    right = solids.add_cylinder(geometry, (1.5 * IN, 0, 0), 3 * IN, 3 * IN)
    hole = solids.combine(geometry, 'union', [left, right])
    cut = solids.combine(geometry, 'subtract', [block, hole], name)
    return geometry, cut


def test_a_copied_solid_brings_its_whole_recipe_with_fresh_ids():
    geometry, cut = _peanut()
    before = {s.id for s in geometry.solids}
    (copied,) = solids.copy(geometry, [cut])
    assert copied not in before
    assert [s.id for s in solids.top_level(geometry)] == [cut, copied]
    made = [s for s in geometry.solids if s.id not in before]
    assert len(made) == 5, 'the block, both cylinders, the union, the cut'
    assert {s.kind for s in made} == {'block', 'cylinder', 'union', 'subtract'}
    assert solids.solid(geometry, copied).name == 'peanut copy'
    assert solids.solid(geometry, cut).name == 'peanut'
    # the recipe is the copy's own
    new_cylinder = next(s for s in made if s.kind == 'cylinder')
    solids.edit(geometry, new_cylinder.id, radius=1 * IN)
    old = [s for s in geometry.solids if s.id in before and s.kind == 'cylinder']
    assert all(s.radius == pytest.approx(3 * IN) for s in old)
    assert solids.solid(geometry, new_cylinder.id).radius == pytest.approx(1 * IN)


def test_an_unnamed_solid_copies_unnamed_and_a_pattern_copies_whole():
    geometry = Geometry(node_id=[], node_xyz=np.zeros((0, 3)), elem_id=[],
                        elem_type=[], elem_conn=[], length_unit='in')
    post = solids.add_cylinder(geometry, (0, 0, 0), 0.5 * IN, 4 * IN)
    posts = solids.pattern(geometry, [post], (2, 3, 1), (2 * IN, 2 * IN, 0))
    (copied,) = solids.copy(geometry, [posts])
    assert solids.solid(geometry, copied).name == ''
    assert solids.solid(geometry, copied).kind == 'pattern'
    assert solids.solid(geometry, copied).counts == (2, 3, 1)
    inner = solids.solid(geometry, copied).operands
    assert len(inner) == 1 and inner[0] != post


def test_solids_copy_into_another_geometry_of_the_same_units():
    source, cut = _peanut()
    target = Geometry(node_id=[], node_xyz=np.zeros((0, 3)), elem_id=[],
                      elem_type=[], elem_conn=[], length_unit='in')
    (copied,) = solids.copy(source, [cut], into=target)
    assert [s.id for s in solids.top_level(target)] == [copied]
    assert len(target.solids) == 5
    assert [s.id for s in solids.top_level(source)] == [cut]
    # the same recipe, id for id in order, under new ids
    same = [(s.kind, s.center, s.widths, s.radius, s.length, len(s.operands))
            for s in source.solids]
    assert [(s.kind, s.center, s.widths, s.radius, s.length, len(s.operands))
            for s in target.solids] == same


def test_a_paste_between_geometries_of_different_units_is_refused():
    source, cut = _peanut()
    raw = Geometry(node_id=[], node_xyz=np.zeros((0, 3)), elem_id=[],
                   elem_type=[], elem_conn=[], length_unit=None)
    with pytest.raises(ValueError, match='units'):
        solids.copy(source, [cut], into=raw)


def test_the_project_copies_solids_and_journals_it():
    project = visualdynamics.Project()
    source, cut = _peanut()
    project.add('Source', source)
    project.add('Target', Geometry(node_id=[], node_xyz=np.zeros((0, 3)),
                                   elem_id=[], elem_type=[], elem_conn=[],
                                   length_unit='in'))
    here = project.copy_solids('Source', [cut])
    there = project.copy_solids('Source', [cut], into='Target')
    assert len(here) == len(there) == 1
    assert len(solids.top_level(project['Source'])) == 2
    assert len(solids.top_level(project['Target'])) == 1
    assert project.journal[-2:] == [
        f"project.copy_solids('Source', [{cut}])",
        f"project.copy_solids('Source', [{cut}], into='Target')"]
    with pytest.raises(ValueError, match='pick'):
        project.copy_solids('Source', [])


def test_copy_and_paste_in_the_tree_moves_solids_between_geometries(window, pump):
    """Solid rows copied, a geometry picked, pasted: the copies are in
    it and selected, and the status says so each way."""
    from conftest import select_objects
    from test_solids_pane import _pick, _rows

    source, cut = _peanut()
    window.add_object('Source', source)
    window.add_object('Target', Geometry(node_id=[], node_xyz=np.zeros((0, 3)),
                                         elem_id=[], elem_type=[], elem_conn=[],
                                         length_unit='in'))
    window.show_object('Source')
    select_objects(window, pump, 'Source')
    window._solids_changed('Source')
    pump()
    _pick(window, pump, _rows(window, 'solid')[:1])
    window.copy_action.trigger()
    pump()
    assert window.statusBar().currentMessage() == (
        'Copied 1 solid — select a geometry and paste')
    select_objects(window, pump, 'Target')
    window.paste_action.trigger()
    pump()
    target = window.objects['Target']
    assert len(solids.top_level(target)) == 1 and len(target.solids) == 5
    assert window.statusBar().currentMessage() == 'Pasted 1 solid into Target'
    picked = window._selected_solids()
    assert picked is not None and picked[0] == 'Target'
    assert picked[1] == [solids.top_level(target)[0].id]
    # pasted again into the source it came from: a second part there
    select_objects(window, pump, 'Source')
    window.paste_action.trigger()
    pump()
    assert len(solids.top_level(window.objects['Source'])) == 2
    assert window.project.journal[-1] == (
        f"project.copy_solids('Source', [{cut}])")


def test_a_paste_of_solids_needs_a_geometry_picked(window, pump):
    from conftest import select_objects
    from test_solids_pane import _pick, _rows

    source, _cut = _peanut()
    window.add_object('Source', source)
    window.add_object('PSD', visualdynamics.Psd(
        np.linspace(1.0, 10.0, 10), np.ones((1, 10)), response_dof=['1X+'],
        ordinate_dim='acceleration**2/frequency'))
    window.show_object('Source')
    select_objects(window, pump, 'Source')
    window._solids_changed('Source')
    pump()
    _pick(window, pump, _rows(window, 'solid')[:1])
    window.copy_action.trigger()
    pump()
    select_objects(window, pump, 'PSD')
    window.paste_action.trigger()
    pump()
    assert window.statusBar().currentMessage() == (
        'Select a geometry to paste the copied solids into')
    assert len(solids.top_level(window.objects['Source'])) == 1
