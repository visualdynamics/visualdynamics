import pytest

import visualdynamics


def test_with_units_override():
    system = visualdynamics.IN_LBF_S.with_units(acceleration='g')
    assert system.unit('acceleration') == 'g'
    assert system.unit('length') == 'in'  # everything else unchanged
    assert system.from_si(9.80665, 'acceleration') == pytest.approx(1.0)


def test_with_units_validates_dimension():
    with pytest.raises(visualdynamics.units.UnitError, match='is not an acceleration unit'):
        visualdynamics.SI.with_units(acceleration='in')  # not an acceleration unit


def test_with_units_does_not_mutate_original():
    visualdynamics.SI.with_units(length='km')
    assert visualdynamics.SI.unit('length') == 'm'


def test_with_units_name():
    assert visualdynamics.SI.with_units('SI-g', acceleration='g').name == 'SI-g'
