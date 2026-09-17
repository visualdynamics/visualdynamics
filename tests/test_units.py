import numpy as np
import pytest

from visualdynamics import units


def test_si_factor_length():
    assert units.si_factor('m') == 1.0
    assert units.si_factor('in') == pytest.approx(0.0254)
    assert units.si_factor('mm') == pytest.approx(0.001)


def test_si_factor_dimension_check():
    assert units.si_factor('in', 'length') == pytest.approx(0.0254)
    with pytest.raises(units.UnitError):
        units.si_factor('kg', 'length')


def test_slinch():
    # 1 slinch = 12 slugs = 175.126835 kg
    assert units.si_factor('slinch', 'mass') == pytest.approx(175.126835, rel=1e-6)


def test_convert():
    assert units.convert(1.0, 'm', 'in') == pytest.approx(39.3700787, rel=1e-6)


def test_unit_system_round_trip():
    values = np.array([1.0, 2.5, -3.0])
    for system in units.SYSTEMS.values():
        out = system.to_si(system.from_si(values, 'length'), 'length')
        assert np.allclose(out, values)


def test_default_system_is_in_slinch_lbf_s_with_g():
    """The default a fresh install opens in (Brandon, 2026-08-28): the
    units his tests are specified and read in, with accelerations
    displayed in g. Display only — the coherent base is what exports
    write, since no file format can record 'g' as a unit."""
    assert units.DEFAULT_SYSTEM is units.SYSTEMS['in-slinch-lbf-s (g)']
    assert units.DEFAULT_SYSTEM.coherent is units.IN_LBF_S
    assert units.DEFAULT_SYSTEM.unit('length') == 'in'
    assert units.DEFAULT_SYSTEM.unit('force') == 'lbf'


def test_from_si_in_lbf_s():
    assert units.IN_LBF_S.from_si(0.0254, 'length') == pytest.approx(1.0)


def test_label_text_uses_real_exponents():
    assert units.IN_LBF_S.label_text('acceleration') == 'in/s²'
    assert units.SI.label_text('acceleration') == 'm/s²'
    assert units.IN_LBF_S.label_text('length') == 'in'


def test_label_text_brackets_a_compound_numerator():
    assert units.IN_LBF_S.label_text('acceleration/force') == '(in/s²)/lbf'
    assert units.IN_LBF_S.label_text('acceleration**2/frequency') == '(in/s²)²/Hz'


def test_label_html_is_a_stacked_fraction():
    html = units.IN_LBF_S.label_html('acceleration/force')
    assert '<table' in html and 'border-bottom' in html
    assert 'in/s<sup>2</sup>' in html
    assert 'lbf' in html


def test_label_html_without_a_denominator_is_plain():
    html = units.IN_LBF_S.label_html('acceleration')
    assert '<table' not in html
    assert html == 'in/s<sup>2</sup>'


def test_labels_empty_when_units_undefined():
    assert units.IN_LBF_S.label_text(units.UNKNOWN) == ''
    assert units.IN_LBF_S.label_html(units.UNKNOWN) == ''


# ---- 'g' is standard gravity in visualdynamics, never grams --------------------------

def test_g_is_an_acceleration():
    """Accelerometers are read in g; pint's 0.001 kg is both wrong and
    plausible enough to pass unnoticed."""
    assert units.dimension_of('g') == 'acceleration'
    assert units.si_factor('g') == pytest.approx(9.80665)


def test_g_survives_being_declared_on_data():
    import os

    import visualdynamics

    path = os.path.join(os.path.dirname(__file__), '..', 'testdata',
                        'plate', 'time.npz')
    data = visualdynamics.import_file(path, ordinate_unit='g')
    assert data.ordinate_dim[0] == 'acceleration'
    assert np.allclose(data.ordinate,
                       visualdynamics.import_file(path).ordinate * 9.80665)


def test_the_other_g_units_are_untouched():
    """Only the bare one-letter shorthand is reserved."""
    assert units.dimension_of('kg') == 'mass'
    assert units.dimension_of('mg') == 'mass'
    assert units.dimension_of('gram') == 'mass'
    assert units.si_factor('kg') == pytest.approx(1.0)
    assert units.si_factor('gram') == pytest.approx(0.001)


def test_gn_still_reads_as_gravity():
    assert units.si_factor('gn') == pytest.approx(9.80665)


def test_g_is_rejected_where_a_mass_is_wanted():
    with pytest.raises(units.UnitError):
        units.si_transform('g', 'mass')


def test_grams_are_not_offered_as_a_mass():
    from visualdynamics.core.unit_choices import MASS_UNITS

    assert 'g' not in MASS_UNITS
    assert 'kg' in MASS_UNITS


def test_acceleration_is_offered_as_g():
    from visualdynamics.core.unit_choices import ALL_ORDINATE_UNITS, ORDINATE_UNITS

    assert 'g' in ORDINATE_UNITS['acceleration']
    assert 'gn' not in ALL_ORDINATE_UNITS


def test_every_offered_unit_is_filed_under_its_own_dimension():
    """A unit in the wrong group would be offered for the wrong quantity."""
    from visualdynamics.core.unit_choices import ORDINATE_UNITS
    from visualdynamics.units import dimension_of

    for dimension, offered in ORDINATE_UNITS.items():
        for unit in offered:
            found = dimension_of(unit)
            if dimension in ('strain', 'dimensionless'):
                # the one pair a unit cannot tell apart: both have empty
                # dimensionality, which is exactly why they are two
                # names, and `dimension_of` answers 'dimensionless' for
                # either
                assert found == 'dimensionless', f'{unit!r} has units'
                continue
            assert found == dimension, f'{unit!r} is not {dimension}'


def test_a_display_system_can_show_accelerations_in_g():
    import visualdynamics

    system = visualdynamics.IN_LBF_S.with_units('in-lbf-s (g)', acceleration='g')
    assert system.from_si(9.80665, 'acceleration') == pytest.approx(1.0)
    assert system.unit('length') == 'in', 'everything else unchanged'


def test_every_system_has_a_g_variant():
    from visualdynamics.units import SYSTEMS

    coherent = [n for n in SYSTEMS if not n.endswith('(g)')]
    in_g = [n for n in SYSTEMS if n.endswith('(g)')]
    assert len(in_g) == len(coherent) == 4
    for name in coherent:
        assert f'{name} (g)' in SYSTEMS


def test_a_g_variant_changes_only_the_acceleration():
    from visualdynamics.units import SYSTEMS

    plain, in_g = SYSTEMS['in-slinch-lbf-s'], SYSTEMS['in-slinch-lbf-s (g)']
    assert in_g.unit('acceleration') == 'g'
    assert plain.unit('acceleration') == 'in/s**2'
    for dimension in ('length', 'force', 'mass', 'time', 'frequency'):
        assert in_g.unit(dimension) == plain.unit(dimension)


def test_a_g_variant_knows_the_system_it_came_from():
    """Exports use it: no file format here can record 'g' as a unit."""
    from visualdynamics.units import SYSTEMS

    assert SYSTEMS['in-slinch-lbf-s (g)'].coherent is SYSTEMS['in-slinch-lbf-s']
    assert SYSTEMS['m-kg-N-s'].coherent is SYSTEMS['m-kg-N-s']


def test_a_mil_is_a_thousandth_of_an_inch():
    """pint's 'mil' is the angular one — 1/6400 of a turn, and dimensionless,
    so it would neither convert nor complain. Vibration displacement is
    quoted in mils constantly; this is the 'g' trap one unit over."""
    from visualdynamics.units import dimension_of, si_factor

    assert dimension_of('mil') == 'length'
    assert si_factor('mil') == pytest.approx(0.0254e-3)
    assert si_factor('mils') == si_factor('mil')
    assert si_factor('mil/s') == pytest.approx(0.0254e-3), 'and in compounds'


def test_the_mil_rule_leaves_other_units_alone():
    """The substitution is on whole words, so units that merely contain the
    letters are untouched."""
    from visualdynamics.units import normalize_unit

    assert normalize_unit('mm') == 'mm'
    assert normalize_unit('milli') == 'milli'
    assert normalize_unit('military') == 'military'


def test_a_dimensionless_value_can_be_shown():
    """pint spells 'no dimension' as the empty expression; asking it about
    the word 'dimensionless' raises KeyError(''), which reached the user as
    a crash while rendering a strain or ratio channel."""
    assert units.si_transform('', 'dimensionless') == (1.0, 0.0)
    assert units.si_transform('dimensionless', 'dimensionless') == (1.0, 0.0)
    assert units.IN_LBF_S.from_si(2.5, 'dimensionless') == pytest.approx(2.5)
    assert units.si_transform('', 'strain') == (1.0, 0.0), 'strain too'
    with pytest.raises(units.UnitError):
        units.si_transform('m', 'dimensionless')


# ---- the case of a unit is the typist's, not pint's -------------------------

def test_a_units_case_is_read_the_way_it_was_typed():
    """A controller's channel table holds what was typed into it, and `G`
    is typed for g all day. pint reads `G` as gauss and refuses `LBF` and
    `Volts` outright, which imported a real run's accelerometers unit-less
    while its `V` channels came through (2026-09-17)."""
    assert units.dimension_of('G') == 'acceleration'
    assert units.si_factor('G') == pytest.approx(9.80665)
    assert units.dimension_of('LBF') == 'force'
    assert units.si_factor('LBF') == pytest.approx(4.4482216152605)
    assert units.dimension_of('Volts') == 'voltage'
    assert units.dimension_of('IN/S**2') == 'acceleration'
    assert units.si_factor('IN/S**2') == pytest.approx(0.0254)
    assert units.fold_unit_case('G') == 'g'
    assert units.fold_unit_case('LBF/IN') == 'lbf/in'


def test_a_spelling_that_already_means_something_is_left_alone():
    """The fold is a rescue, not a rewrite: `mV` and `MV` are different
    voltages, and a string as written wins whenever it names a quantity
    this toolset tracks. A string nothing rescues stays unknown, so a
    caller can still tell the two apart."""
    assert units.si_factor('mV') == pytest.approx(1e-3)
    assert units.si_factor('MV') == pytest.approx(1e6)
    assert units.fold_unit_case('mV') == 'mV'
    assert units.fold_unit_case('MV') == 'MV'
    assert units.fold_unit_case('gauss') == 'gauss'
    assert units.dimension_of('gauss') is None
    assert units.fold_unit_case('') == ''


def test_every_offered_unit_folds_back_to_itself_from_upper_case():
    """The spellings the fold knows must cover the shortlists the
    interface offers, or a typed `KPA` would land as unknown while `kPa`
    is one click away. Two upper-case spellings are units in their own
    right and stay as written, which is the rule the test above pins:
    `MV` is megavolts and `M/M` is molar over molar — the same dimension
    either way, so nothing downstream reads them differently."""
    from visualdynamics.core.unit_choices import ALL_ORDINATE_UNITS, LENGTH_UNITS, MASS_UNITS

    their_own = {'mV': 'MV', 'm/m': 'M/M'}
    for unit in [*ALL_ORDINATE_UNITS, *LENGTH_UNITS, *MASS_UNITS]:
        upper = unit.upper()
        assert units.dimension_of(upper) == units.dimension_of(unit), unit
        assert units.fold_unit_case(upper) == their_own.get(unit, unit), unit
