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

import math

import numpy as np
import pytest
from conftest import edit_block, edit_category, select_objects
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
                      'ρ [kg/m³]', 'Thickness [m]', 'Section', 'Shape',
                      'Dimensions [m]', 'A [m²]', 'Iy [m⁴]', 'Iz [m⁴]',
                      'J [m⁴]', 'Orientation']
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
    # the table shows the display system (2026-09-26); this test is
    # about the journal, so it types SI, as a person would after picking
    # it from the unit menu
    window.unit_combo.setCurrentText('m-kg-N-s')
    pump()
    edit_block(window, pump, int(window.objects['Geometry'].block_id[0]))
    model = window.table.model()
    _set(model, 'Material', 'Al')
    _set(model, 'E [Pa]', '70e9')
    _set(model, 'ρ [kg/m³]', '2700')
    _set(model, 'Thickness [m]', '0.01')
    pump()
    window.project.solve_modes('Geometry', maximum_frequency=500.0)
    script = window.project.session_script()
    assert ('from visualdynamics.core.fem import RIGID, BlockProperties, '
            'Material, Section') in script
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


# ---- the material library (the third slice, 2026-09-25) ----------------

def test_the_library_is_typical_handbook_values_each_with_its_note():
    from visualdynamics.core.fem import MATERIAL_LIBRARY, MATERIALS, LibraryMaterial, material

    names = [entry.material.name for entry in MATERIAL_LIBRARY]
    assert len(names) == len(set(names)) >= 18
    # the lookup is the library and the rigid link, which is picked where
    # a material is but is not one (2026-09-26)
    assert set(MATERIALS) == set(names) | {'rigid (massless)'}
    for entry in MATERIAL_LIBRARY:
        assert isinstance(entry, LibraryMaterial) and entry.note, entry
        m = entry.material
        assert 1e9 < m.youngs_modulus < 500e9, m
        assert 900 < m.density < 20000, m
        assert 0.2 <= m.poissons_ratio < 0.5, m
        assert m.modulus_of_rigidity is None, 'isotropic, derived'
        assert material(m.name) is m
    with pytest.raises(KeyError, match="'unobtainium' is not in the "
                                        'material library; it has 6061-T6'):
        material('unobtainium')


def test_the_demo_plate_is_made_of_the_librarys_6061():
    from visualdynamics.core.fem import PSI, material
    from visualdynamics.demo import plate

    assert plate.ALUMINUM is material('6061-T6')
    assert plate.ALUMINUM.youngs_modulus == 10.0e6 * PSI, (
        'the handbook number, converted exactly, not a rounding')


def test_picking_a_library_material_fills_the_row(qt_app):
    from visualdynamics.core.fem import MATERIALS, material

    geometry = _plate_geometry(properties=False)
    model = block_table_model(geometry)
    column = model.columns[_column(model, 'Material')]
    assert column.choices == list(MATERIALS) and column.choices_editable, (
        'the library as a drop-down that still takes a typed name')
    _set(model, 'Material', 'Ti-6Al-4V')
    props = geometry.block_properties[int(geometry.block_id[0])]
    assert props.material is material('Ti-6Al-4V')
    shown = float(model.data(model.index(0, _column(model, 'E [Pa]'))))
    assert shown == pytest.approx(16.5e6 * 6894.757293168361, rel=1e-5), (
        'the row shows the library number')
    assert float(model.data(model.index(0, _column(model, 'ν')))) == 0.342
    # the thickness the row had stays; a second pick swaps the material
    _set(model, 'Thickness [m]', '0.005')
    _set(model, 'Material', '304 stainless')
    props = geometry.block_properties[int(geometry.block_id[0])]
    assert props.material is material('304 stainless')
    assert props.thickness == 0.005
    # a name outside the library is a name, and the numbers are the user's
    _set(model, 'Material', 'my alloy')
    props = geometry.block_properties[int(geometry.block_id[0])]
    assert props.material.name == 'my alloy'
    assert props.material.density == material('304 stainless').density, (
        'renaming keeps the numbers already in the row')



# ---- the Blocks table in the display unit system (2026-09-26) ----------

def test_the_blocks_table_shows_and_takes_the_display_units(qt_app):
    """Brandon: the Blocks table should show, and expect, the current
    display unit system. Held in SI on the geometry and in the journal,
    as the model is inside."""
    from visualdynamics.core.fem import material
    from visualdynamics.units import SYSTEMS

    inch = SYSTEMS['in-slinch-lbf-s']
    psi, inches = 6894.757293168361, 0.0254
    geometry = _plate_geometry(properties=False)
    model = block_table_model(geometry, inch)
    titles = [model.headerData(c, Qt.Orientation.Horizontal)
              for c in range(model.columnCount())]
    for title in ('E [psi]', 'ρ [slinch/in³]', 'Thickness [in]', 'A [in²]',
                  'Iy [in⁴]', 'Iz [in⁴]', 'J [in⁴]', 'ν', 'Material'):
        assert title in titles, (title, titles)
    # a library pick reads in psi and slinch/in³
    _set(model, 'Material', '6061-T6')
    assert float(model.data(model.index(0, _column(model, 'E [psi]')))) == \
        pytest.approx(10.0e6)
    assert float(model.data(model.index(0, _column(model, 'ρ [slinch/in³]')))) \
        == pytest.approx(0.098 / 386.08858, rel=1e-6)
    # typed in inches, stored in meters
    _set(model, 'Thickness [in]', '0.5')
    props = geometry.block_properties[int(geometry.block_id[0])]
    assert props.thickness == pytest.approx(0.5 * inches)
    assert props.material is material('6061-T6')
    # typed in psi, stored in pascals; ν has no unit and is not scaled
    _set(model, 'E [psi]', '1e7')
    _set(model, 'ν', '0.33')
    props = geometry.block_properties[int(geometry.block_id[0])]
    assert props.material.youngs_modulus == pytest.approx(1e7 * psi)
    assert props.material.poissons_ratio == 0.33
    # a section's second moments in in⁴
    _set(model, 'Iy [in⁴]', '2')
    props = geometry.block_properties[int(geometry.block_id[0])]
    assert props.section.iy == pytest.approx(2 * inches ** 4)


def test_the_blocks_table_journals_in_si_whatever_it_shows(qt_app):
    from visualdynamics.units import SYSTEMS

    geometry = _plate_geometry(properties=False)
    model = block_table_model(geometry, SYSTEMS['mm-kg-N-s'])
    lines = []
    model.edit_journaled.connect(lines.append)
    _set(model, 'Thickness [mm]', '5')
    assert lines[-1].endswith('thickness=0.005)'), lines[-1]


def test_switching_the_unit_system_restates_the_blocks_table(window, pump):
    """The table follows the unit menu: the same thickness reads in
    inches, then in millimeters."""
    geometry = _plate_geometry()           # 0.01 m thick, SI inside
    path_name = 'Geometry'
    window.add_object(path_name, geometry)
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    pump()
    edit_block(window, pump, int(window.objects['Geometry'].block_id[0]))
    model = window.table.model()
    assert float(model.data(model.index(0, _column(model, 'Thickness [in]')))) \
        == pytest.approx(0.01 / 0.0254)
    window.unit_combo.setCurrentText('mm-kg-N-s')
    pump()
    model = window.table.model()
    assert float(model.data(model.index(0, _column(model, 'Thickness [mm]')))) \
        == pytest.approx(10.0)


def test_switching_the_unit_system_restates_the_node_table(window, pump):
    """The same gap, found through the Blocks table: an open node table
    kept the coordinates' unit of the moment it was opened."""

    window.add_object('Geometry', _plate_geometry())
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    pump()
    edit_category(window, pump, 'Nodes')
    model = window.table.model()
    headers = [model.headerData(c, Qt.Orientation.Horizontal)
               for c in range(model.columnCount())]
    assert 'X [in]' in headers
    window.unit_combo.setCurrentText('mm-kg-N-s')
    pump()
    model = window.table.model()
    headers = [model.headerData(c, Qt.Orientation.Horizontal)
               for c in range(model.columnCount())]
    assert 'X [mm]' in headers



# ---- sections from their shapes (2026-09-26) -----------------------------

def _section_of(geometry):
    return geometry.block_properties[int(geometry.block_id[0])].section


def test_a_shape_asks_for_its_dimensions_and_computes_the_rest(qt_app):
    """Pick a shape, the Dimensions cell says what it wants; type them in
    the display units, and A, Iy, Iz and J are the shape's."""
    from visualdynamics.units import SYSTEMS

    inch = 0.0254
    geometry = _plate_geometry(properties=False)
    model = block_table_model(geometry, SYSTEMS['in-slinch-lbf-s'])
    shape_column = model.columns[_column(model, 'Shape')]
    assert shape_column.choices == ['round tube', 'rod', 'rectangle',
                                    'rectangular tube', 'I-beam', 'channel',
                                    'angle', 'custom']
    _set(model, 'Section', 'strut')
    assert model.data(model.index(0, _column(model, 'Shape'))) == 'custom'
    _set(model, 'Shape', 'round tube')
    dims = model.index(0, _column(model, 'Dimensions [in]'))
    assert model.data(dims) == 'D=?, t=?', 'the shape says what it wants'
    _set(model, 'Dimensions [in]', '1, 0.065')
    section = _section_of(geometry)
    assert section == Section.round_tube('strut', 1 * inch, 0.065 * inch)
    assert model.data(dims) == 'D=1, t=0.065'
    j = float(model.data(model.index(0, _column(model, 'J [in⁴]'))))
    assert j == pytest.approx(math.pi * (1 - 0.87 ** 4) / 32)
    # the labels are accepted as shown, in any order they were written
    _set(model, 'Dimensions [in]', 'D=1.25, t=0.049')
    assert _section_of(geometry).dimensions == pytest.approx(
        (1.25 * inch, 0.049 * inch))


def test_a_shaped_sections_numbers_are_computed_not_typed(qt_app):
    geometry = _plate_geometry(properties=False)
    model = block_table_model(geometry)
    _set(model, 'Shape', 'rectangle')
    _set(model, 'Dimensions [m]', 'b=0.01, h=0.03')
    said = []
    model.edit_rejected.connect(said.append)
    assert not model.setData(model.index(0, _column(model, 'Iy [m⁴]')),
                             '1e-6', EDIT)
    assert said and 'choose Custom in Shape' in said[0]
    # Custom keeps the numbers and lets them be typed
    _set(model, 'Shape', 'custom')
    assert _section_of(geometry).shape == ''
    _set(model, 'Iy [m⁴]', '1e-6')
    assert _section_of(geometry).iy == 1e-6
    assert _section_of(geometry).area == pytest.approx(3e-4), 'kept'


def test_wrong_dimensions_are_refused_by_name(qt_app):
    geometry = _plate_geometry(properties=False)
    model = block_table_model(geometry)
    said = []
    model.edit_rejected.connect(said.append)
    index = model.index(0, _column(model, 'Dimensions [m]'))
    assert not model.setData(index, '0.1, 0.2', EDIT)
    assert 'choose a shape first' in said[-1]
    _set(model, 'Shape', 'I-beam')
    assert not model.setData(index, '0.2, 0.1', EDIT)
    assert 'an I-beam takes 4 dimensions' in said[-1], said[-1]
    assert 'd (depth), bf (flange width)' in said[-1]
    assert not model.setData(index, '0.02, 0.05, 0.011, 0.005', EDIT)
    assert 'do not make an I' in said[-1]


def test_an_angle_says_where_to_point_it(qt_app):
    geometry = _plate_geometry(properties=False)
    model = block_table_model(geometry)
    _set(model, 'Shape', 'angle')
    shown = _set(model, 'Dimensions [m]', '0.1, 0.1, 0.01')
    assert shown.endswith('; orient 45.0° from the long leg'), shown
    # and what it shows is taken back as typed: the note is ignored
    _set(model, 'Dimensions [m]', shown)
    assert _section_of(geometry).dimensions == pytest.approx((0.1, 0.1, 0.01))


def test_an_unfinished_section_is_refused_when_the_model_is_built():
    frame = visualdynamics.Geometry(
        node_id=[1, 2, 3], node_xyz=[[0, 0, 0], [0, 0, 1], [1, 0, 1]],
        elem_id=[1, 2], elem_type=[21, 21], elem_conn=[[1, 2], [2, 3]],
        elem_block=[1, 1], block_id=[1], block_name=['frame'],
        length_unit='m',
        block_properties={1: BlockProperties(
            ALUMINUM, section=Section('tube', 0, 0, 0, 0, 'round tube', ()),
            orientation=(0.0, 1.0, 0.0))})
    with pytest.raises(ValueError, match=r'block 1 \(frame\): the section is '
                                          r'not finished — its round tube has '
                                          r'no dimensions yet'):
        Model.from_geometry(frame)


def test_a_shaped_section_journals_and_saves_by_its_dimensions(qt_app,
                                                              tmp_path):
    geometry = _plate_geometry(properties=False)
    model = block_table_model(geometry)
    lines = []
    model.edit_journaled.connect(lines.append)
    _set(model, 'Section', 'beam')
    _set(model, 'Shape', 'channel')
    _set(model, 'Dimensions [m]', '0.15, 0.05, 0.009, 0.006')
    assert ("section=Section.of_shape('beam', 'channel', "
            '[0.15, 0.05, 0.009, 0.006])') in lines[-1]
    visualdynamics.save(geometry, tmp_path / 'g.vdyn')
    back = visualdynamics.load(tmp_path / 'g.vdyn')
    assert _section_of(back) == _section_of(geometry)
    assert _section_of(back).shape == 'channel'



def test_a_rigid_link_is_picked_like_a_material_and_journals_by_name(qt_app):
    """The Material drop-down offers the rigid link; its modulus, density
    and ratio read blank and refuse typing, and its journal line names
    it — its infinite modulus is not a number a script can replay."""
    from visualdynamics.core.fem import RIGID

    geometry = _plate_geometry(properties=False)
    model = block_table_model(geometry)
    assert 'rigid (massless)' in model.columns[_column(model, 'Material')].choices
    lines = []
    model.edit_journaled.connect(lines.append)
    _set(model, 'Material', 'rigid (massless)')
    props = geometry.block_properties[int(geometry.block_id[0])]
    assert props.material is RIGID and props.kind == 'rigid'
    assert model.data(model.index(0, _column(model, 'E [Pa]'))) == ''
    assert lines[-1].endswith('BlockProperties(RIGID)'), lines[-1]
    said = []
    model.edit_rejected.connect(said.append)
    assert not model.setData(model.index(0, _column(model, 'E [Pa]')), '1e9',
                             EDIT)
    assert 'has no modulus, density or ratio' in said[-1]
    namespace = {}
    replay = ('from visualdynamics.core.fem import RIGID, BlockProperties\n'
              'props = ' + lines[-1].split(' = ', 1)[1])
    exec(replay, namespace)  # noqa: S102 — the table's own journal line
    assert namespace['props'].material is RIGID
