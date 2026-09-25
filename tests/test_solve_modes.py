"""Solve Modes: a geometry whose blocks carry their properties is a
finite element model, and the app solves it (Brandon, 2026-09-25).

The second slice of the block model. The Blocks table grew the property
columns — a material for any block, a thickness for plates, a section
for beams — writing into `geometry.block_properties` with journal lines
a session script replays; and `Project.solve_modes` builds the model
from the blocks and adds its normal modes in the geometry's group, with
a Solve Modes act on the bar that asks the two numbers the solution
takes.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import edit_category, select_objects
from PySide6.QtCore import Qt
from test_acts import _bar
from test_fem import ALUMINUM, square_plate

import visualdynamics
from visualdynamics.core.fem import BlockProperties, Model, Section
from visualdynamics.core.shapes import ShapeSet
from visualdynamics.gui.object_tables import block_table_model
from visualdynamics.project import Project

EDIT = Qt.ItemDataRole.EditRole


def _plate_geometry(mesh=4, properties=True):
    geometry = square_plate(mesh).geometry()
    if properties:
        geometry.block_properties = {
            int(geometry.block_id[0]): BlockProperties(ALUMINUM, 0.01)}
    return geometry


def _verbs(project, name):
    return [verb for verb, _reading in project.verbs(name)]


def test_solve_modes_applies_only_once_the_blocks_carry_properties():
    project = Project()
    project.add('Bare', _plate_geometry(properties=False))
    project.add('Skin', _plate_geometry())
    assert 'solve_modes' not in _verbs(project, 'Bare')
    assert 'generate_rigid_body_modes' in _verbs(project, 'Bare')
    assert 'solve_modes' in _verbs(project, 'Skin')


def test_solve_modes_adds_the_models_modes_in_the_geometrys_group():
    project = Project()
    geometry = _plate_geometry()
    project.add('Skin', geometry)
    expected = Model.from_geometry(geometry).eigensolution(
        maximum_frequency=2000.0, damping=0.02)
    name = project.solve_modes('Skin', maximum_frequency=2000.0, damping=0.02)
    assert name == 'Skin Modes'
    shapes = project[name]
    assert isinstance(shapes, ShapeSet)
    assert np.allclose(shapes.frequency, expected.frequency, rtol=1e-9,
                       atol=1e-6)
    assert int(np.sum(shapes.frequency == 0.0)) == 6, 'free-free'
    assert np.allclose(shapes.damping, 0.02)
    assert any({'Skin', 'Skin Modes'} <= set(group['members'])
               for group in project.links), 'linked to its geometry'
    assert project.provenance[name]['verb'] == 'solve_modes'


def test_solve_modes_refuses_a_bare_geometry_and_a_non_geometry():
    project = Project()
    project.add('Bare', _plate_geometry(properties=False))
    with pytest.raises(ValueError, match='no block properties'):
        project.solve_modes('Bare')
    project.add('Shapes', ShapeSet([10.0], [0.01], ['1Z+'], [[1.0]]))
    with pytest.raises(TypeError, match='not a geometry'):
        project.solve_modes('Shapes')


def _column(model, title):
    titles = [model.headerData(c, Qt.Orientation.Horizontal)
              for c in range(model.columnCount())]
    return titles.index(title)


def _set(model, title, text, row=0):
    index = model.index(row, _column(model, title))
    assert model.setData(index, text, EDIT), (title, text)
    return model.data(index, Qt.ItemDataRole.DisplayRole)


def test_the_blocks_table_builds_a_property_set_cell_by_cell(qt_app):
    geometry = _plate_geometry(properties=False)
    model = block_table_model(geometry)
    titles = [model.headerData(c, Qt.Orientation.Horizontal)
              for c in range(model.columnCount())]
    assert titles == ['Block', 'Name', 'Elements', 'Material', 'E [Pa]', 'ν',
                      'ρ [kg/m³]', 'Thickness [m]', 'Section', 'A [m²]',
                      'Iy [m⁴]', 'Iz [m⁴]', 'J [m⁴]', 'Orientation']
    assert model.data(model.index(0, _column(model, 'E [Pa]'))) == '', \
        'blank until set'
    _set(model, 'Material', '6061-T6')
    _set(model, 'E [Pa]', '68.9e9')
    _set(model, 'ρ [kg/m³]', '2700')
    _set(model, 'ν', '0.33')
    _set(model, 'Thickness [m]', '0.003')
    props = geometry.block_properties[int(geometry.block_id[0])]
    assert props.material.name == '6061-T6'
    assert props.material.youngs_modulus == 68.9e9
    assert props.material.density == 2700.0
    assert props.material.poissons_ratio == 0.33
    assert props.thickness == 0.003 and props.section is None
    assert props.kind == 'plate'
    assert Model.from_geometry(geometry).plates, 'the table wrote a model'


def test_a_section_fills_the_same_way_and_an_orientation_is_three_numbers(
        qt_app):
    geometry = _plate_geometry(properties=False)
    model = block_table_model(geometry)
    _set(model, 'Section', 'rib')
    _set(model, 'A [m²]', '3e-4')
    _set(model, 'Iy [m⁴]', '2.25e-9')
    _set(model, 'Iz [m⁴]', '2.5e-10')
    _set(model, 'J [m⁴]', '7e-10')
    _set(model, 'Orientation', '0, 0, 1')
    props = geometry.block_properties[int(geometry.block_id[0])]
    assert props.section == Section('rib', 3e-4, 2.25e-9, 2.5e-10, 7e-10)
    assert props.orientation == (0.0, 0.0, 1.0)
    assert props.kind == 'beam'
    assert model.data(model.index(0, _column(model, 'Orientation'))) == '0, 0, 1'
    said = []
    model.edit_rejected.connect(said.append)
    assert not model.setData(model.index(0, _column(model, 'Orientation')),
                             '0, 1', EDIT)
    assert said and 'three numbers' in said[0]
    assert props.orientation == (0.0, 0.0, 1.0), 'a refusal changes nothing'


def test_property_edits_journal_the_whole_set_and_replay(window, pump,
                                                        tmp_path):
    """Whichever cell was touched last, the journal line rebuilds the
    block's whole property set, so the session script lands on the same
    object — and `solve_modes` replays after it with the same modes."""
    from test_workflow_journals import _replay

    path = tmp_path / 'Geometry.vdyn'
    visualdynamics.save(_plate_geometry(properties=False), path)
    window.import_paths([str(path)])
    pump()
    assert 'Geometry' in window.project
    edit_category(window, pump, 'Blocks')
    model = window.table.model()
    _set(model, 'Material', 'Al')
    _set(model, 'E [Pa]', '70e9')
    _set(model, 'ρ [kg/m³]', '2700')
    _set(model, 'Thickness [m]', '0.01')
    pump()
    window.project.solve_modes('Geometry', maximum_frequency=500.0)
    script = window.project.session_script()
    assert ('from visualdynamics.core.fem import BlockProperties, Material, '
            'Section') in script
    assert script.count('.block_properties[') == 4, 'one line per edit'
    assert ("BlockProperties(Material('Al', 70000000000.0, 2700.0, 0.3), "
            'thickness=0.01)') in script
    assert "solve_modes('Geometry', maximum_frequency=500.0" in script
    _replay(window)


def test_the_act_is_on_the_bar_for_a_geometry_with_properties(window, pump):
    window.add_object('Bare', _plate_geometry(properties=False))
    window.add_object('Skin', _plate_geometry())
    select_objects(window, pump, 'Bare')
    assert 'Solve Modes' not in _bar(window.data_pane) + _bar(window.scene)
    select_objects(window, pump, 'Skin')
    assert 'Solve Modes' in _bar(window.data_pane) + _bar(window.scene)


def test_the_act_asks_two_numbers_and_shows_the_modes(window, pump,
                                                     monkeypatch):
    from visualdynamics.gui import main_window as window_module

    window.add_object('Skin', _plate_geometry())
    select_objects(window, pump, 'Skin')
    asked = []
    answers = iter([(800.0, True), (1.5, True)])

    def get_double(parent, title, label, *rest):
        asked.append(label)
        return next(answers)

    monkeypatch.setattr(window_module.QInputDialog, 'getDouble', get_double)
    window.solve_modes_act()
    pump()
    assert [a.split(' [')[0] for a in asked] == [
        'Highest frequency to solve for', 'Damping to give every mode']
    shapes = window.project['Skin Modes']
    assert shapes.frequency.max() <= 800.0
    assert np.allclose(shapes.damping, 0.015), 'percent in, fraction stored'
    assert window.current_object() is shapes
    message = window.statusBar().currentMessage()
    assert message.startswith('Skin Modes: ') and '6 rigid' in message


def test_declining_the_dialog_solves_nothing(window, pump, monkeypatch):
    from visualdynamics.gui import main_window as window_module

    window.add_object('Skin', _plate_geometry())
    select_objects(window, pump, 'Skin')
    monkeypatch.setattr(window_module.QInputDialog, 'getDouble',
                        lambda *args, **kwargs: (0.0, False))
    window.solve_modes_act()
    pump()
    assert 'Skin Modes' not in window.project


def test_a_saved_geometry_solves_to_the_same_modes(tmp_path):
    project = Project()
    project.add('Skin', _plate_geometry())
    project.solve_modes('Skin', num_modes=8)
    visualdynamics.save(project['Skin'], tmp_path / 'skin.vdyn')
    again = Project()
    again.add('Skin', visualdynamics.load(tmp_path / 'skin.vdyn'))
    again.solve_modes('Skin', num_modes=8)
    assert np.allclose(again['Skin Modes'].frequency,
                       project['Skin Modes'].frequency)
