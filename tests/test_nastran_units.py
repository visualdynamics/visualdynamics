"""A Nastran deck's property values read in its own unit system.

Brandon, 2026-10-26: importing a deck, the material properties did not
convert. The reader converted the coordinates from the declared length
and took every property value as SI, so a deck in millimetres read 2 mm
shells as 2 m and steel at 200 kPa. A deck's units are its length, a
mass and seconds, and every property converts with the coordinates.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.geometry import Geometry, deck_to_si

#: a 100 x 50 steel plate of four quads, a point mass, a spring along Z
#: and one about X, in the deck's {length} with its {E}, {rho}, {t}, {m}
#: and {k} written as that system's numbers
DECK = """BEGIN BULK
GRID,1,,0.,0.,0.
GRID,2,,{half},0.,0.
GRID,3,,{length},0.,0.
GRID,4,,0.,{width},0.
GRID,5,,{half},{width},0.
GRID,6,,{length},{width},0.
CQUAD4,1,10,1,2,5,4
CQUAD4,2,10,2,3,6,5
PSHELL,10,20,{t}
MAT1,20,{E},,0.3,{rho}
CONM2,9,3,,{m}
CELAS2,11,{k},6,3
CELAS2,12,{kr},6,4
ENDDATA
"""


def _deck(tmp_path, name, scale_length, E, rho, t, m, k, kr):
    path = tmp_path / name
    path.write_text(DECK.format(
        half=0.05 * scale_length, length=0.1 * scale_length,
        width=0.05 * scale_length, E=E, rho=rho, t=t, m=m, k=k, kr=kr))
    return path


def _mm(tmp_path):
    """mm-tonne-s: E in MPa, density in t/mm³, mass in t, a spring in
    N/mm and about an axis in N·mm/rad."""
    return _deck(tmp_path, 'plate_mm.nas', 1000.0, '200000.', '7.85-9',
                 '2.0', '0.0005', '1000.', '1000.')


def _si(tmp_path):
    return _deck(tmp_path, 'plate_m.nas', 1.0, '2.+11', '7850.', '0.002',
                 '0.5', '1.+6', '1.')


def _props(geometry):
    shell = geometry.group_properties[10]
    springs = [p.stiffness for p in geometry.group_properties.values()
               if p.stiffness is not None]
    mass = next(p.mass for p in geometry.group_properties.values()
                if p.mass is not None)
    return shell, springs, mass


def test_every_property_converts_with_the_declared_units(tmp_path):
    geometry = visualdynamics.import_file(_mm(tmp_path), length_unit='mm')
    shell, springs, mass = _props(geometry)
    assert geometry.mass_unit == 'tonne', 'the usual mass for millimetres'
    assert geometry.node_xyz[2, 0] == pytest.approx(0.1)
    assert shell.thickness == pytest.approx(0.002)
    assert shell.material.youngs_modulus == pytest.approx(2e11)
    assert shell.material.density == pytest.approx(7850.0)
    assert mass == pytest.approx(0.5)
    along, about = (next(k for k in springs if k[2] is not None),
                    next(k for k in springs if k[3] is not None))
    assert along[2] == pytest.approx(1e6), 'N/mm to N/m'
    assert about[3] == pytest.approx(1.0), 'N·mm/rad to N·m/rad'


def test_an_inch_deck_reads_in_si(tmp_path):
    path = _deck(tmp_path, 'plate_in.nas', 39.37007874, '3.+7',
                 '7.33-4', '0.1', '0.002', '100.', '100.')
    geometry = visualdynamics.import_file(path, length_unit='in')
    shell, springs, mass = _props(geometry)
    assert geometry.mass_unit == 'slinch'
    assert shell.thickness == pytest.approx(0.00254)
    assert shell.material.youngs_modulus == pytest.approx(2.0684e11, rel=1e-4)
    assert shell.material.density == pytest.approx(7835.0, rel=1e-3)
    assert mass == pytest.approx(0.002 * 175.1268, rel=1e-6)
    along, about = (next(k for k in springs if k[2] is not None),
                    next(k for k in springs if k[3] is not None))
    assert along[2] == pytest.approx(100.0 * 4.448222 / 0.0254, rel=1e-6)
    assert about[3] == pytest.approx(100.0 * 4.448222 * 0.0254, rel=1e-6)


def test_the_same_plate_in_mm_and_in_m_has_the_same_modes(tmp_path):
    """What the conversion is for: a deck's modes do not depend on the
    units it was written in."""
    frequencies = []
    for path, unit in ((_mm(tmp_path), 'mm'), (_si(tmp_path), 'm')):
        project = visualdynamics.Project()
        project.add('Plate', visualdynamics.import_file(path,
                                                        length_unit=unit))
        made = project.solve_modes('Plate', num_modes=12)
        frequencies.append(np.asarray(project[made].frequency))
    assert np.allclose(frequencies[0], frequencies[1], rtol=1e-9, atol=1e-6)
    assert frequencies[0].max() > 100.0, 'elastic modes were solved'


def test_a_stated_mass_unit_is_the_one_used(tmp_path):
    geometry = visualdynamics.import_file(_mm(tmp_path), length_unit='mm',
                                          mass_unit='kg')
    shell, _springs, _mass = _props(geometry)
    assert geometry.mass_unit == 'kg'
    assert shell.material.youngs_modulus == pytest.approx(2e8), \
        'mm-kg-s: a force of kg·mm/s² is a millinewton'
    with pytest.raises(ValueError, match='mass unit is declared with a '
                                         'length unit'):
        visualdynamics.import_file(_mm(tmp_path), mass_unit='kg')


def test_units_declared_later_convert_once_and_undo(tmp_path):
    geometry = visualdynamics.import_file(_mm(tmp_path))
    assert geometry.from_deck and geometry.length_unit is None
    shell, _springs, mass = _props(geometry)
    assert (shell.thickness, shell.material.youngs_modulus, mass) == \
        (2.0, 200000.0, 0.0005), 'the deck as written'
    geometry.define_units('mm')
    geometry.define_units('in')
    geometry.define_units('mm')
    shell, _springs, mass = _props(geometry)
    assert shell.thickness == pytest.approx(0.002), 'reinterpreted, not compounded'
    assert mass == pytest.approx(0.5)
    geometry.undefine_units()
    shell, _springs, mass = _props(geometry)
    assert shell.thickness == pytest.approx(2.0)
    assert shell.material.youngs_modulus == pytest.approx(200000.0)
    assert geometry.mass_unit is None


def test_a_wrong_system_is_warned_about(tmp_path):
    geometry = visualdynamics.import_file(_mm(tmp_path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        geometry.define_units('m')
    said = ' '.join(str(w.message) for w in caught)
    # 200 kPa is a soft foam's modulus, so the density is what says it
    assert 'MAT1 20' in said and 'density 7.85e-09 kg/m³' in said
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        geometry.define_units('mm')
    assert not caught, 'the right system says nothing'


def test_a_geometry_not_from_a_deck_keeps_its_properties(tmp_path):
    deck = visualdynamics.import_file(_si(tmp_path), length_unit='m')
    built = Geometry(deck.node_id, deck.node_xyz, elem_id=deck.elem_id,
                     elem_type=deck.elem_type, elem_conn=deck.elem_conn,
                     elem_group=deck.elem_group, group_id=deck.group_id,
                     group_properties=deck.group_properties,
                     length_unit='m')
    before = dict(built.group_properties)
    built.define_units('mm')
    assert built.group_properties == before, \
        'properties entered in SI are not a deck\'s numbers'
    with pytest.raises(ValueError, match='only a geometry read from a '
                                         'finite element deck'):
        built.define_units('m', 'kg')


def test_a_saved_deck_reopens_with_its_system(tmp_path):
    project = visualdynamics.Project()
    project.add('Plate', visualdynamics.import_file(_mm(tmp_path),
                                                    length_unit='mm',
                                                    mass_unit='kg'))
    project.save(tmp_path / 'deck.vdyn')
    reopened = visualdynamics.Project.open(tmp_path / 'deck.vdyn')['Plate']
    assert reopened.from_deck and reopened.mass_unit == 'kg'
    reopened.define_units('mm', 'tonne')
    shell, _springs, _mass = _props(reopened)
    assert shell.material.youngs_modulus == pytest.approx(2e11), \
        'reinterpreted from the deck numbers the file kept'


def test_the_inverse_undoes_the_conversion():
    from visualdynamics.core.fem import GroupProperties, Material, Section

    props = {1: GroupProperties(Material('m', 3e7, 7.3e-4, 0.3, 1.1e7),
                                thickness=0.25,
                                section=Section('s', 2.0, 3.0, 4.0, 5.0,
                                                'rectangle', (1.0, 2.0)),
                                mass=0.01,
                                stiffness=(10.0, None, 30.0, 40.0, None,
                                           60.0))}
    there = deck_to_si(props, 'in', 'slinch')
    back = deck_to_si(there, 'in', 'slinch', inverse=True)
    one, two = props[1], back[1]
    assert two.material.youngs_modulus == pytest.approx(
        one.material.youngs_modulus)
    assert two.material.modulus_of_rigidity == pytest.approx(1.1e7)
    assert two.section.iy == pytest.approx(3.0)
    assert two.section.dimensions == pytest.approx((1.0, 2.0))
    assert there[1].section.iy == pytest.approx(3.0 * 0.0254 ** 4)
    assert two.stiffness == pytest.approx((10.0, None, 30.0, 40.0, None,
                                           60.0))


# ---- the Units pane ----------------------------------------------------------


def _open_units(window, pump, name):
    item = window._item_for_object(name)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    pump()
    window.define_units()
    pump()
    return window.units_table.model()


def test_the_pane_offers_a_decks_mass_unit_and_journals_both(window, pump,
                                                             tmp_path):
    from PySide6.QtCore import Qt

    window.add_object('Plate', visualdynamics.import_file(_mm(tmp_path)))
    pump()
    model = _open_units(window, pump, 'Plate')
    unit = model.columnCount() - 1
    assert model.rowCount() == 2
    assert model.index(1, 0).data() == 'Deck properties'
    refused = []
    model.edit_rejected.connect(refused.append)
    model.setData(model.index(1, unit), 'kg', Qt.ItemDataRole.EditRole)
    assert refused and 'length unit first' in refused[0]
    model.setData(model.index(0, unit), 'mm', Qt.ItemDataRole.EditRole)
    window._restate_after_units()
    plate = window.objects['Plate']
    assert plate.mass_unit == 'tonne'
    assert model.index(1, unit).data() == 'tonne'
    assert plate.group_properties[10].material.youngs_modulus == \
        pytest.approx(2e11)
    assert window.project.journal[-1] == \
        "project['Plate'].define_units('mm', 'tonne')"
    model.setData(model.index(1, unit), 'kg', Qt.ItemDataRole.EditRole)
    window._restate_after_units()
    assert plate.group_properties[10].material.youngs_modulus == \
        pytest.approx(2e8)
    assert window.project.journal[-1] == \
        "project['Plate'].define_units('mm', 'kg')"
    model.setData(model.index(0, unit), 'in', Qt.ItemDataRole.EditRole)
    assert plate.mass_unit == 'kg', 'a mass stated apart from the usual is kept'


def test_the_pane_says_when_the_system_looks_wrong(window, pump, tmp_path):
    from PySide6.QtCore import Qt

    window.add_object('Plate', visualdynamics.import_file(_mm(tmp_path)))
    pump()
    model = _open_units(window, pump, 'Plate')
    model.setData(model.index(0, model.columnCount() - 1), 'm',
                  Qt.ItemDataRole.EditRole)
    window._restate_after_units()
    said = window.statusBar().currentMessage()
    assert 'MAT1 20: density 7.85e-09 kg/m³' in said
    assert 'another unit system than m and kg' in said


def test_a_built_geometry_keeps_its_one_row(window, pump):
    window.add_object('Frame', Geometry([1, 2], [[0, 0, 0], [1, 0, 0]]))
    pump()
    assert _open_units(window, pump, 'Frame').rowCount() == 1
