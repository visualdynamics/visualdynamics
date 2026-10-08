"""Point masses in a geometry (Brandon, 2026-10-07: "add point masses
to geometries"). An element group of point elements given a mass alone is a set
of lumped masses, the way a finite element deck states one — a CONM2,
a point-mass element group — and `Model.from_geometry` puts that mass at each
element's node. The BARC's bolts are the first use: too small to mesh,
heavy enough to move its modes by several percent."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import edit_category, fixture_path, select_objects
from PySide6.QtCore import Qt
from test_fem import ALUMINUM
from test_fem_solids import _bar_of_bricks
from test_solve_modes import _column, _set

import visualdynamics
from visualdynamics.core.fem import GroupProperties, Model
from visualdynamics.gui.object_tables import element_group_table_model
from visualdynamics.io import exodus, nastran

EDIT = Qt.ItemDataRole.EditRole
POINT = 161


def _bar_with_masses(mass=0.05, ends=(1, 26)):
    """The six-brick bar as a geometry, and an element group of point elements
    at `ends` given `mass` each."""
    bar = _bar_of_bricks(along=25)
    geometry = bar.geometry()
    geometry.group_properties = {int(geometry.group_id[0]):
                                 GroupProperties(ALUMINUM)}
    group = geometry.add_group('bolts')
    geometry.add_elements([[n] for n in ends], [POINT] * len(ends),
                          [group] * len(ends))
    geometry.group_properties[group] = GroupProperties(mass=mass)
    return bar, geometry, group


def test_a_mass_block_is_the_lumped_masses_add_mass_makes():
    """The same model to the last digit as the masses added by hand,
    and the mass is really there: the total grows by exactly it."""
    bar, geometry, _block = _bar_with_masses()
    built = Model.from_geometry(geometry)
    assert [(m.node, m.mass) for m in built.masses] == [(1, 0.05), (26, 0.05)]
    for node in (1, 26):
        bar.add_mass(node, 0.05)
    assert built.eigensolution(num_modes=10).frequency == pytest.approx(
        bar.eigensolution(num_modes=10).frequency, rel=1e-12, abs=1e-9)
    bare = _bar_of_bricks(along=25).eigensolution(num_modes=10).frequency
    assert built.eigensolution(num_modes=10).frequency[6] < 0.99 * bare[6], \
        'the masses lower the first bending mode'


def test_a_mass_block_takes_a_mass_alone():
    assert GroupProperties(mass=1.0).kind == 'mass'
    assert GroupProperties(ALUMINUM, mass=1.0).kind == 'mass', \
        'a mass makes an element group of point masses whatever else is set'
    assert GroupProperties().kind == 'no material'
    _bar, geometry, group = _bar_with_masses()
    # point elements given a material are refused by what they take
    geometry.group_properties[group] = GroupProperties(ALUMINUM)
    with pytest.raises(ValueError, match='mass elements, which take a mass'):
        Model.from_geometry(geometry)
    geometry.group_properties[group] = GroupProperties()
    with pytest.raises(ValueError, match='has no material'):
        Model.from_geometry(geometry)
    # and bricks given a mass by what a mass element group holds
    geometry.group_properties[group] = GroupProperties(mass=1.0)
    geometry.group_properties[int(geometry.group_id[0])] = \
        GroupProperties(mass=1.0)
    with pytest.raises(ValueError, match='holds point elements, one mass'):
        Model.from_geometry(geometry)


def test_a_mass_on_a_node_nothing_holds_is_refused_by_node():
    """A loose node is grounded whole at solve time; a mass on one is
    not a reference point but a mass free to fly off, three spurious
    modes at 0 Hz, so it is refused where the model is built."""
    _bar, geometry, group = _bar_with_masses()
    loose = geometry.add_node([0.0, 0.0, 1.0])
    geometry.add_element([loose], POINT, group)
    with pytest.raises(ValueError, match=f'point mass sits on node {loose}'):
        Model.from_geometry(geometry)


def test_a_mass_block_rides_the_native_file(tmp_path):
    _bar, geometry, group = _bar_with_masses(mass=0.0125)
    visualdynamics.save(geometry, tmp_path / 'bar.vdyn')
    back = visualdynamics.load(tmp_path / 'bar.vdyn')
    assert back.group_properties[group] == GroupProperties(mass=0.0125)
    assert back.group_properties[int(back.group_id[0])].material == ALUMINUM
    assert [m.mass for m in Model.from_geometry(back).masses] == [0.0125] * 2


def test_conm2_masses_read_into_blocks_and_write_back(tmp_path):
    """A plain CONM2 is a point mass: one element group per distinct mass, after
    the deck's own property ids, carrying it. One carried off its grid
    is not a point mass and its element group is left for the analyst."""
    deck = tmp_path / 'masses.bdf'
    deck.write_text(
        'BEGIN BULK\n'
        'GRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\nGRID,3,,1.,1.,0.\n'
        'GRID,4,,0.,1.,0.\n'
        'CQUAD4,10,7,1,2,3,4\n'
        'CONM2,21,1,0,0.25\nCONM2,22,2,0,0.25\nCONM2,23,3,0,0.5\n'
        'CONM2,24,4,0,0.5,0.,0.,0.1\n'
        'ENDDATA\n')
    geometry = nastran.load(str(deck))
    names = dict(zip(geometry.group_id.tolist(), geometry.group_name))
    of = {int(e): int(b) for e, b in zip(geometry.elem_id, geometry.elem_group)}
    assert of[21] == of[22] != of[23] and min(of[21], of[23]) > 7
    assert names[of[21]] == 'CONM2 0.25' and names[of[23]] == 'CONM2 0.5'
    assert geometry.group_properties[of[21]] == GroupProperties(mass=0.25)
    assert geometry.group_properties[of[23]] == GroupProperties(mass=0.5)
    assert of[24] not in geometry.group_properties, 'an offset mass is not one'
    out = tmp_path / 'out.bdf'
    nastran.save(geometry, str(out))
    back = nastran.load(str(out))
    masses = {int(e): back.group_properties.get(int(b))
              for e, b in zip(back.elem_id, back.elem_group)}
    assert masses[21].mass == masses[22].mass == 0.25
    assert masses[23].mass == 0.5
    assert masses[24] is None


def test_the_blocks_table_takes_a_mass_in_the_display_units(qt_app):
    """Typed in the display system, held in kilograms, journaled as the
    line that rebuilds it; a mass element group refuses a material, and clearing
    its mass leaves the element group with nothing to say."""
    from visualdynamics.units import SYSTEMS

    inch = SYSTEMS['in-slinch-lbf-s']
    _bar, geometry, group = _bar_with_masses()
    row = list(geometry.group_id).index(group)
    model = element_group_table_model(geometry, inch)
    lines = []
    model.edit_journaled.connect(lines.append)
    slinch = 1.0 / inch.from_si(1.0, 'mass')
    _set(model, 'Mass [slinch]', '2', row=row)
    assert geometry.group_properties[group] == GroupProperties(
        mass=pytest.approx(2 * slinch))
    assert lines[-1] == (f'.group_properties[{group}] = '
                         f'GroupProperties(mass={2 * slinch!r})')
    assert float(model.data(model.index(row, _column(model, 'Mass [slinch]')))) \
        == pytest.approx(2.0)
    assert model.data(model.index(row, _column(model, 'Material'))) == ''
    said = []
    model.edit_rejected.connect(said.append)
    assert not model.setData(model.index(row, _column(model, 'Material')),
                             '6061-T6', EDIT)
    assert said and 'takes a mass alone' in said[0]
    _set(model, 'Mass [slinch]', '', row=row)
    assert group not in geometry.group_properties


def test_a_point_element_is_added_with_one_click(window, pump, tmp_path):
    """A point is one node, and its one pick commits it; a floor of two
    picks had kept a point mass from ever being clicked into place."""
    geometry = exodus.load(fixture_path('plate', 'geometry.exo'))
    group = geometry.add_group('masses')
    geometry.add_element([int(geometry.node_id[0])], POINT, group)
    visualdynamics.save(geometry, tmp_path / 'Geometry.vdyn')
    window.import_paths([str(tmp_path / 'Geometry.vdyn')])
    pump()
    plate = window.objects['Geometry']
    edit_category(window, pump, 'Points')
    window.set_add_mode(True)
    assert window.element_type == (POINT, 1)
    before = len(plate.elem_conn)
    screen = window._projector.screen()[0]
    window.hover_at(*screen[5])
    window._add_at(*screen[5])
    assert len(plate.elem_conn) == before + 1
    assert int(plate.elem_type[-1]) == POINT
    assert int(plate.elem_group[-1]) == group
    assert np.asarray(plate.elem_conn[-1]).tolist() == \
        [int(plate.node_id[5])]


def _floor_foot_and_weights():
    """The tie tests' floor and foot, and an element group of point masses on
    the floor."""
    from test_tie import _floor_and_foot

    geometry = _floor_and_foot()
    group = geometry.add_group('weights')
    geometry.add_element([int(geometry.node_id[0])], POINT, group)
    geometry.group_properties[group] = GroupProperties(mass=0.1)
    return geometry


def test_ties_refuse_a_mass_block_by_what_it_is():
    from visualdynamics.core import mesh

    geometry = _floor_foot_and_weights()
    with pytest.raises(ValueError, match="'weights' is point masses, not rigid"):
        mesh.tie(geometry, geometry.elements_in('foot')[:1], 'floor',
                 group='weights')


def test_the_tie_menu_offers_no_mass_block(window, pump, monkeypatch):
    """Nothing to tie to in a lumped mass, as in a link: the menu lists
    the plates and leaves the weights out, where reading a mass element group's
    material had raised before it could list anything."""
    from test_tie import _select_rows

    window.add_object('Geometry', _floor_foot_and_weights())
    select_objects(window, pump, 'Geometry')
    edit_category(window, pump, 'Quads')
    plate = window.objects['Geometry']
    rows = [int(np.flatnonzero(plate.elem_id == e)[0])
            for e in plate.elements_in('foot')]
    _select_rows(window, pump, rows[:1])
    popped = []
    monkeypatch.setattr(window, '_pop_menu', popped.append)
    window.tie_action.trigger()
    labels = [action.text() for action in popped[0].actions()]
    assert labels == ['Tie to the nearest nodes of', 'floor', 'foot', '',
                      'A second selection…']
