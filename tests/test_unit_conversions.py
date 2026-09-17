"""Every unit the interface offers, against hand-checked constants.

The factors here are written from authority (NIST exact definitions),
not computed by the code under test — a wrong registry entry, a unit
pint reads differently than a test engineer does ('g', 'mil', 'lbm'),
or a future dead shortlist entry all fail loudly here. The second half
drives conversions through the data objects, because storage is SI and
every declare/undeclare/redeclare must scale exactly once.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from visualdynamics import units
from visualdynamics.core.data import Frf, Psd, Specification, TimeHistory
from visualdynamics.core.shapes import ShapeSet

IN = 0.0254                       # exact, by definition
LBF = 4.4482216152605             # exact, by definition
G = 9.80665                       # standard gravity, exact

# (unit, si scale, si offset) — scales from the definitions, not from pint
HAND_CHECKED = [
    # lengths
    ('m', 1.0, 0.0), ('mm', 1e-3, 0.0), ('cm', 1e-2, 0.0),
    ('in', IN, 0.0), ('ft', 0.3048, 0.0), ('mil', IN * 1e-3, 0.0),
    # accelerations
    ('m/s**2', 1.0, 0.0), ('mm/s**2', 1e-3, 0.0), ('in/s**2', IN, 0.0),
    ('g', G, 0.0),
    # velocities
    ('m/s', 1.0, 0.0), ('mm/s', 1e-3, 0.0), ('in/s', IN, 0.0),
    # forces
    ('N', 1.0, 0.0), ('kN', 1e3, 0.0), ('lbf', LBF, 0.0),
    # pressures
    ('Pa', 1.0, 0.0), ('kPa', 1e3, 0.0), ('MPa', 1e6, 0.0),
    ('psi', LBF / IN ** 2, 0.0),
    # voltages
    ('V', 1.0, 0.0), ('mV', 1e-3, 0.0),
    # strains, which are ratios: one meter per meter is one
    ('strain', 1.0, 0.0), ('microstrain', 1e-6, 0.0), ('m/m', 1.0, 0.0),
    # temperatures, the affine ones with their offsets
    ('K', 1.0, 0.0), ('degC', 1.0, 273.15),
    ('degF', 5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
    ('degR', 5.0 / 9.0, 0.0),
    # masses
    ('kg', 1.0, 0.0), ('tonne', 1e3, 0.0), ('lbm', 0.45359237, 0.0),
    ('slug', LBF / 0.3048, 0.0), ('slinch', LBF / IN, 0.0),
    ('dimensionless', 1.0, 0.0),
    # rotations at a virtual point, per radian, and the moments
    # conjugate to them
    ('rad', 1.0, 0.0), ('deg', math.pi / 180.0, 0.0),
    ('rad/s', 1.0, 0.0), ('deg/s', math.pi / 180.0, 0.0),
    ('rad/s**2', 1.0, 0.0), ('deg/s**2', math.pi / 180.0, 0.0),
    ('N*m', 1.0, 0.0), ('lbf*in', LBF * IN, 0.0), ('lbf*ft', LBF * 0.3048, 0.0),
    # modal responses through mass-normalized shapes: the response's
    # unit times the square root of the mass unit (a slinch is
    # lbf·s²/in, so √slinch is √(LBF/IN) kg½)
    ('m/s**2*kg**0.5', 1.0, 0.0),
    ('in/s**2*slinch**0.5', IN * math.sqrt(LBF / IN), 0.0),
    ('g*slinch**0.5', G * math.sqrt(LBF / IN), 0.0),
    ('m/s*kg**0.5', 1.0, 0.0), ('in/s*slinch**0.5', IN * math.sqrt(LBF / IN), 0.0),
    ('m*kg**0.5', 1.0, 0.0), ('in*slinch**0.5', IN * math.sqrt(LBF / IN), 0.0),
    ('N*kg**0.5', 1.0, 0.0), ('lbf*slinch**0.5', LBF * math.sqrt(LBF / IN), 0.0),
]


@pytest.mark.parametrize(('unit', 'scale', 'offset'),
                         HAND_CHECKED, ids=[u for u, _s, _o in HAND_CHECKED])
def test_the_unit_means_what_a_test_engineer_means(unit, scale, offset):
    got_scale, got_offset = units.si_transform(unit)
    assert got_scale == pytest.approx(scale, rel=1e-9)
    assert got_offset == pytest.approx(offset, abs=1e-9)


def test_every_offered_unit_is_hand_checked():
    """The shortlists and this file's table cover each other exactly, so
    a unit added to the interface without an authority check fails."""
    from visualdynamics.core.unit_choices import (
        ALL_ORDINATE_UNITS,
        LENGTH_UNITS,
        MASS_UNITS,
        REFERENCE_UNITS,
    )

    offered = set(ALL_ORDINATE_UNITS) | set(REFERENCE_UNITS) \
        | set(LENGTH_UNITS) | set(MASS_UNITS)
    assert offered <= {unit for unit, _s, _o in HAND_CHECKED}


def test_g_is_gravity_and_grams_are_still_reachable():
    assert units.si_factor('g') == pytest.approx(G)
    assert units.si_factor('gram', 'mass') == pytest.approx(1e-3)
    assert units.si_factor('kg', 'mass') == 1.0
    assert units.si_factor('mg', 'mass') == pytest.approx(1e-6), (
        'milligrams, not milli-gravities')


def test_mil_is_a_thousandth_of_an_inch_not_an_angle():
    assert units.si_factor('mil', 'length') == pytest.approx(IN * 1e-3)
    assert units.si_factor('mils', 'length') == pytest.approx(IN * 1e-3)


def test_affine_units_refuse_a_bare_factor():
    with pytest.raises(units.UnitError, match='offset'):
        units.si_factor('degC')


def test_affine_conversions_hit_the_known_points():
    assert units.to_si(0.0, 'degC') == pytest.approx(273.15)
    assert units.to_si(32.0, 'degF') == pytest.approx(273.15)
    assert units.to_si(212.0, 'degF') == pytest.approx(373.15)
    assert units.from_si(273.15, 'degF') == pytest.approx(32.0)
    assert units.convert(100.0, 'degC', 'degF') == pytest.approx(212.0)


# ---- conversions through the data objects -----------------------------------

def waveform():
    return TimeHistory(np.linspace(0.0, 1.0, 8),
                       np.arange(1.0, 9.0)[None, :],
                       response_dof=['1X+'])


def test_defining_scales_once_redefining_reinterprets():
    data = waveform()
    raw = data.ordinate.copy()
    data.define_units('g')
    assert np.allclose(data.ordinate, raw * G), 'raw times the factor, once'
    data.define_units('in/s**2')
    assert np.allclose(data.ordinate, raw * IN), (
        'a correction reinterprets the raw values; it never rescales '
        'what the wrong guess produced')
    data.undefine_units()
    assert np.allclose(data.ordinate, raw), 'withdrawn means raw again'
    assert data.ordinate_dim[0] == 'unknown'


def test_records_convert_independently():
    data = TimeHistory(np.linspace(0.0, 1.0, 4), np.ones((2, 4)),
                       response_dof=['1X+', '2X+'])
    data.define_units({0: 'g'})
    assert np.allclose(data.ordinate[0], G)
    assert np.allclose(data.ordinate[1], 1.0), 'the other stayed raw'
    assert data.ordinate_dim == ['acceleration', 'unknown']


def test_an_frf_scales_by_the_ratio_of_its_halves():
    frf = Frf(np.linspace(0.0, 10.0, 4), np.ones((1, 4), dtype=complex),
              response_dof=['1X+'], reference_dof=['2Z+'])
    frf.define_units('g', 'lbf')
    assert np.allclose(frf.ordinate, G / LBF)
    assert frf.ordinate_dim[0] == 'acceleration/force'
    frf.undefine_units()
    assert np.allclose(frf.ordinate, 1.0)


def test_a_psd_scales_by_the_square():
    psd = Psd(np.linspace(0.0, 10.0, 4), np.ones((1, 4), dtype=complex),
              response_dof=['1X+'])
    psd.define_units('g')
    assert np.allclose(psd.ordinate, G ** 2)
    assert psd.ordinate_dim[0] == 'acceleration**2/frequency'


def test_a_cross_psd_scales_by_the_product():
    psd = Psd(np.linspace(0.0, 10.0, 4), np.ones((1, 4), dtype=complex),
              response_dof=['1X+'], reference_dof=['2Z+'])
    psd.define_units({0: 'g'}, {0: 'lbf'})
    assert np.allclose(psd.ordinate, G * LBF)
    assert psd.ordinate_dim[0] == 'acceleration*force/frequency'


def test_specification_limits_ride_every_conversion():
    freq = np.linspace(10.0, 100.0, 5)
    spec = Specification(freq, np.ones((1, 5), dtype=complex),
                         response_dof=['1X+'],
                         abort_upper=np.full((1, 5), 4.0))
    spec.define_units('g')
    assert np.allclose(spec.limits['abort_upper'], 4.0 * G ** 2)
    spec.define_units('m/s**2')
    assert np.allclose(spec.limits['abort_upper'], 4.0), (
        'reinterpreted with the ordinate, exactly once')


def test_mass_normalized_shapes_convert_by_the_root():
    shapes = ShapeSet([10.0], [0.01], ['1X+'], [[6.0]])
    shapes.define_units('slinch')
    assert np.allclose(shapes.shape_matrix,
                       6.0 / np.sqrt(LBF / IN)), (
        '1/sqrt(slinch) into 1/sqrt(kg) divides by sqrt(kg per slinch)')
    shapes.define_units('kg')
    assert np.allclose(shapes.shape_matrix, 6.0), 'reinterpreted, not piled'


def test_display_ordinate_is_the_stored_si_over_the_display_factor():
    data = waveform()
    data.define_units('m/s**2')
    stored = data.ordinate.copy()
    shown = data.display_ordinate(units.IN_LBF_S)
    assert np.allclose(shown, stored / IN)
    in_g = units.IN_LBF_S.with_units(acceleration='g')
    assert np.allclose(data.display_ordinate(in_g), stored / G)
    assert np.allclose(data.display_ordinate(units.SI), stored)


def test_a_compound_display_factor_is_the_product_of_powers():
    system = units.IN_LBF_S
    assert system.factor('acceleration/force') == pytest.approx(IN / LBF)
    assert system.factor('acceleration**2/frequency') == pytest.approx(
        IN ** 2)


def test_geometry_lengths_convert_and_withdraw():
    from visualdynamics.core.geometry import Geometry

    geometry = Geometry(node_id=[1], node_xyz=[[1.0, 2.0, 3.0]])
    geometry.define_units('in')
    assert np.allclose(geometry.node_xyz, np.array([[1.0, 2.0, 3.0]]) * IN)
    geometry.undefine_units()
    assert np.allclose(geometry.node_xyz, [[1.0, 2.0, 3.0]])


def test_display_systems_agree_with_each_other():
    """in-lbf-s and mm-kg-N-s must describe the same physical value."""
    data = waveform()
    data.define_units('g')
    inches = data.display_ordinate(units.IN_LBF_S)
    millimeters = data.display_ordinate(units.MMKS)
    assert np.allclose(np.asarray(inches) * IN,
                       np.asarray(millimeters) * 1e-3)


# ---- the refusals -----------------------------------------------------------

def test_nonsense_units_raise_not_guess():
    with pytest.raises(units.UnitError, match='zorkmids'):
        units.si_transform('zorkmids')


def test_unknown_dimension_names_raise():
    with pytest.raises(units.UnitError, match='sideways'):
        units.parse_dimension('sideways/force')
    with pytest.raises(units.UnitError, match='sideways'):
        units.si_transform('m', 'sideways')


def test_dimension_of_unparseable_is_none():
    assert units.dimension_of('zorkmids') is None
    assert units.dimension_of('m*kg') is None, 'no visualdynamics dimension matches'
    assert units.dimension_of('') == 'dimensionless'


def test_a_system_missing_a_dimension_says_so():
    bare = units.UnitSystem('bare', {'length': 'm'})
    with pytest.raises(units.UnitError, match='force'):
        bare.unit('force')
    with pytest.raises(units.UnitError, match='has no unit'):
        bare.label_text('acceleration/force')


def test_an_offset_unit_cannot_join_a_compound():
    weird = units.SI.with_units(temperature='degC')
    with pytest.raises(units.UnitError, match='offset'):
        weird.factor('temperature/force')


def test_mismatched_record_metadata_is_refused():
    from visualdynamics.core.data import TimeHistory

    with pytest.raises(ValueError, match='response_dof'):
        TimeHistory(np.linspace(0, 1, 4), np.ones((2, 4)),
                    response_dof=['1X+'])
    with pytest.raises(ValueError, match='ordinate shape'):
        TimeHistory(np.linspace(0, 1, 4), np.ones((1, 5)),
                    response_dof=['1X+'])
    with pytest.raises(units.UnitError, match='sideways'):
        TimeHistory(np.linspace(0, 1, 4), np.ones((1, 4)),
                    response_dof=['1X+'], ordinate_dim='sideways')


def test_an_frf_needs_both_units():
    frf = Frf(np.linspace(0, 10, 4), np.ones((1, 4), dtype=complex),
              response_dof=['1X+'], reference_dof=['2Z+'])
    with pytest.raises(units.UnitError, match='both'):
        frf.define_units('g')


def test_deleting_out_of_range_records_is_refused():
    data = waveform()
    with pytest.raises(IndexError, match='no such records'):
        data.delete_records([5])
    with pytest.raises(ValueError, match='every record'):
        data.delete_records([0])
