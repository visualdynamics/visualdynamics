
import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics

SHAPES = fixture_path('plate', 'shapes.npy')


def test_import_sdynpy_shapes():
    shapes = visualdynamics.import_file(SHAPES)
    assert isinstance(shapes, visualdynamics.ShapeSet)
    assert shapes.num_shapes == 22
    assert shapes.num_dofs == 1014
    assert not shapes.is_complex
    assert shapes.coordinate[:3] == ['101X+', '101Y+', '101Z+']
    assert shapes.frequency[0] >= 0
    assert np.allclose(shapes.damping, 0.02)


def test_shapes_import_unit_less():
    shapes = visualdynamics.import_file(SHAPES)
    assert not shapes.units_defined
    assert shapes.mass_unit is None


def test_define_units_scales_by_inverse_root_mass():
    """Coefficients are 1/sqrt(mass): declaring slinch divides by sqrt(kg
    per slinch)."""
    shapes = visualdynamics.import_file(SHAPES)
    raw = shapes.shape_matrix.copy()
    shapes.define_units('slinch')
    assert shapes.mass_unit == 'slinch'
    assert np.allclose(shapes.shape_matrix,
                       raw / np.sqrt(175.126835), rtol=1e-6)


def test_define_units_reinterprets():
    a = visualdynamics.import_file(SHAPES)
    raw = a.shape_matrix.copy()
    a.define_units('slinch')      # wrong guess
    a.define_units('kg')          # corrected: these were SI all along
    assert np.allclose(a.shape_matrix, raw)


def test_define_units_rejects_a_non_mass_unit():
    shapes = visualdynamics.import_file(SHAPES)
    with pytest.raises(visualdynamics.units.UnitError, match='is not a mass unit'):
        shapes.define_units('m')


def test_display_shapes_round_trips_the_declared_unit():
    """Shapes normalized in slinch, shown in the slinch system, are the
    file's own numbers again."""
    raw = visualdynamics.import_file(SHAPES).shape_matrix.copy()
    shapes = visualdynamics.import_file(SHAPES, mass_unit='slinch')
    assert np.allclose(shapes.display_shapes(visualdynamics.IN_LBF_S), raw)
    # and in SI they are smaller by sqrt(kg per slinch)
    assert np.allclose(shapes.display_shapes(visualdynamics.SI),
                       raw / np.sqrt(175.126835), rtol=1e-6)


def test_unit_label():
    shapes = visualdynamics.import_file(SHAPES, mass_unit='kg')
    assert shapes.unit_label() == '1/√kg'
    assert shapes.unit_label(visualdynamics.IN_LBF_S) == '1/√slinch'
    assert visualdynamics.import_file(SHAPES).unit_label() == ''


def test_display_shapes_passes_through_when_undefined():
    shapes = visualdynamics.import_file(SHAPES)
    assert np.array_equal(shapes.display_shapes(visualdynamics.IN_LBF_S),
                          shapes.shape_matrix)


def test_mode_label():
    shapes = visualdynamics.import_file(SHAPES)
    label = shapes.mode_label(6)
    assert label.startswith('Mode 7 — ')
    assert 'Hz' in label and 'damping' in label


def test_save_load_round_trip(tmp_path):
    shapes = visualdynamics.import_file(SHAPES, mass_unit='kg')
    path = str(tmp_path / 'modes.vdyn')
    shapes.save(path)
    assert visualdynamics.load(path) == shapes


def test_round_trip_keeps_unit_less_state(tmp_path):
    shapes = visualdynamics.import_file(SHAPES)
    path = str(tmp_path / 'modes.vdyn')
    shapes.save(path)
    again = visualdynamics.load(path)
    assert not again.units_defined
    assert again == shapes


def test_mismatched_shape_matrix_rejected():
    with pytest.raises(ValueError, match='shape_matrix has 1 modes but frequency/damping have 2/2'):
        visualdynamics.ShapeSet(frequency=[1.0, 2.0], damping=[0.0, 0.0],
                      coordinate=['1Z+', '2Z+'],
                      shape_matrix=[[1.0, 2.0]])   # one mode, two declared


def test_complex_shapes_supported():
    shapes = visualdynamics.ShapeSet(frequency=[10.0], damping=[0.01],
                           coordinate=['1Z+', '2Z+'],
                           shape_matrix=[[1 + 1j, 2 - 1j]])
    assert shapes.is_complex
    assert 'complex' in repr(shapes)


def test_all_plate_object_types_import():
    """One model, every type visualdynamics handles — the plate is a
    complete test on its own."""
    expected = {
        'test_geometry.npz': visualdynamics.Geometry,
        'geometry.npz': visualdynamics.Geometry,
        'geometry.unv': visualdynamics.Geometry,
        'geometry.exo': visualdynamics.Geometry,
        'shapes.npy': visualdynamics.ShapeSet,
        'frfs.npz': visualdynamics.Frf,
        'frfs.unv': visualdynamics.Frf,
        'time.npz': visualdynamics.TimeHistory,
        'psd.npz': visualdynamics.Psd,
        'spectrum.npz': visualdynamics.Spectrum,
        'channel_table.vdyn': visualdynamics.ChannelTable,
    }
    for name, kind in expected.items():
        path = fixture_path('plate', name)
        assert isinstance(visualdynamics.import_file(path), kind), name


def test_plate_objects_fit_the_plate_geometry():
    objects = {
        'geometry': visualdynamics.import_file(
            fixture_path('plate', 'geometry.npz')),
        'shapes': visualdynamics.import_file(
            fixture_path('plate', 'shapes.npy')),
        'table': visualdynamics.import_file(
            fixture_path('plate', 'channel_table.vdyn')),
    }
    assert visualdynamics.check_compatibility(objects, 'geometry').incompatible_names == []


def test_modes_are_stored_in_frequency_order():
    """A set is read as a list — mode 1, mode 2 — and everything that
    steps through one reads it in stored order. Fitting is what breaks
    it: peaks are confirmed in whatever order they are found, and the
    largest is rarely the lowest."""
    from visualdynamics.core.shapes import ShapeSet

    shapes = ShapeSet(frequency=[90.0, 12.0, 45.0],
                      damping=[0.03, 0.01, 0.02],
                      coordinate=['1X+', '2X+'],
                      shape_matrix=[[9.0, 9.5], [1.0, 1.5], [4.0, 4.5]],
                      modal_mass=[3.0, 1.0, 2.0],
                      description=['third', 'first', 'second'])
    assert list(shapes.frequency) == [12.0, 45.0, 90.0]
    assert list(shapes.damping) == [0.01, 0.02, 0.03], 'damping came along'
    assert list(shapes.modal_mass) == [1.0, 2.0, 3.0]
    assert shapes.description == ['first', 'second', 'third']
    assert list(shapes.shape_matrix[0]) == [1.0, 1.5], (
        'and the shapes themselves, or a mode would wear another mode')


def test_coincident_modes_keep_the_order_they_arrived_in():
    """A pair of repeated roots is not two modes anyone has ordered, and
    six rigid-body modes at exactly zero are not five and a spare."""
    from visualdynamics.core.shapes import ShapeSet

    shapes = ShapeSet(frequency=[0.0, 0.0, 0.0, 30.0],
                      damping=[0.0] * 4,
                      coordinate=['1X+'],
                      shape_matrix=[[1.0], [2.0], [3.0], [4.0]])
    assert list(shapes.shape_matrix[:, 0]) == [1.0, 2.0, 3.0, 4.0]


def test_an_imported_set_is_sorted_even_when_the_file_is_not(tmp_path):
    """The fitting screen already keeps its own list in order, so the
    case that reaches a ShapeSet unsorted is a *file*: a UNV orders its
    modes by mode number, and nothing says a writer numbered them by
    frequency."""
    from visualdynamics.core.shapes import ShapeSet
    from visualdynamics.io import load, save

    jumbled = ShapeSet(frequency=[70.0, 20.0], damping=[0.02, 0.01],
                       coordinate=['1X+'], shape_matrix=[[7.0], [2.0]])
    assert list(jumbled.frequency) == [20.0, 70.0], 'sorted on the way in'
    path = tmp_path / 'shapes.vdyn'
    save(jumbled, path)
    back = load(path)
    assert list(back.frequency) == [20.0, 70.0], 'and stays sorted'
    assert list(back.shape_matrix[:, 0]) == [2.0, 7.0]


def test_damping_may_be_overdamped_but_never_negative():
    """Above 1.0 is a real measurement — an overdamped oscillator — so
    the rule is a floor, not a range. Below zero is a fit that has gone
    wrong, and it synthesizes an FRF that grows without bound."""
    from visualdynamics.core.shapes import ShapeSet

    overdamped = ShapeSet([10.0], [1.4], ['1X+'], [[1.0]])
    assert overdamped.damping[0] == 1.4
    assert ShapeSet([10.0], [0.0], ['1X+'], [[1.0]]).damping[0] == 0.0
    with pytest.raises(ValueError, match='cannot be negative'):
        ShapeSet([10.0], [-0.01], ['1X+'], [[1.0]])


def test_a_frequency_may_be_zero_but_never_negative():
    """Zero is a rigid-body mode and load-bearing — the FRF synthesis
    cancels its 0/0 by testing for exactly that."""
    from visualdynamics.core.shapes import ShapeSet

    assert ShapeSet([0.0], [0.0], ['1X+'], [[1.0]]).frequency[0] == 0.0
    with pytest.raises(ValueError, match='cannot be negative'):
        ShapeSet([-10.0], [0.01], ['1X+'], [[1.0]])


def test_a_shape_coordinate_is_a_dof_like_any_other():
    from visualdynamics.core.shapes import ShapeSet

    with pytest.raises(ValueError, match='coordinate'):
        ShapeSet([10.0], [0.01], ['gibberish'], [[1.0]])
