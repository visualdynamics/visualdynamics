"""A record's icon says what it measures.

One object routinely mixes quantities — a test's time data holds
accelerations, forces, voltages and temperatures side by side — so the
record, not the object, is what has to carry the answer. Putting it in the
row label instead cost twice the width and had to be read; a shape is read at
a glance.

Displacement, velocity and acceleration share one trace and are told apart by
Newton's dots, which is the notation a dynamicist already has.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QSize

import visualdynamics
from visualdynamics.core.data import Frf, Psd, TimeHistory
from visualdynamics.gui.icons import (
    QUANTITY_COLORS,
    quantity_icon,
    quantity_of,
    record_icon,
    type_icon,
)

QUANTITIES = list(QUANTITY_COLORS)


def pixels(icon, size=64):
    """What the icon actually draws.

    Not `cacheKey`: `type_icon` is memoized on its argument tuple, so
    `type_icon('Frf')` and `type_icon('Frf', True)` are two cache entries with
    different keys and identical pictures. Comparing the pictures is the
    question worth asking anyway.
    """
    return bytes(icon.pixmap(QSize(size, size)).toImage().constBits())


def test_every_quantity_asked_for_has_an_icon():
    assert set(QUANTITIES) == {'length', 'velocity', 'acceleration',
                               'force', 'temperature', 'voltage',
                               'strain', 'pressure',
                               # the rotations at a virtual point
                               # (Brandon, 2026-09-04)
                               'angle', 'angular_velocity',
                               'angular_acceleration', 'moment',
                               'modal_length', 'modal_velocity',
                               'modal_acceleration', 'modal_force'}


def test_a_modal_response_through_mass_normalized_shapes_wears_the_m(qt_app):
    from visualdynamics.core.rigid import MassProperties, rigid_body_shapes
    from visualdynamics.core.transform import to_modal
    from visualdynamics.gui.icons import record_icon

    rng = np.random.default_rng(2)
    geometry = visualdynamics.Geometry(np.arange(101, 106),
                                       rng.uniform(-1.0, 1.0, (5, 3)),
                                       length_unit='m')
    scaled = rigid_body_shapes(geometry, MassProperties(
        (0, 0, 0), mass=2.0, inertia=(1.0, 1.0, 1.0, 0, 0, 0)))
    t = np.arange(256) / 256.0
    history = visualdynamics.TimeHistory(
        t, np.ones((scaled.num_dofs, len(t))),
        response_dof=list(scaled.coordinate), ordinate_dim='acceleration')
    modal, _report = to_modal(history, scaled)
    assert modal.ordinate_dim[0] == 'modal_acceleration'
    assert pixels(record_icon(modal, 0)) == \
        pixels(quantity_icon('modal_acceleration'))
    assert pixels(record_icon(modal, 0)) != \
        pixels(quantity_icon('acceleration'))


def test_a_virtual_points_rotations_wear_their_own_glyphs(qt_app):
    """The rotational records of a modal response used to fall back to
    the object's icon; each now has a glyph, the linear counterpart's
    color with an arc for the trace and a twist for the force."""
    from visualdynamics.core.rigid import MassProperties, rigid_body_shapes
    from visualdynamics.core.transform import to_modal
    from visualdynamics.gui.icons import record_icon

    rng = np.random.default_rng(1)
    geometry = visualdynamics.Geometry(np.arange(101, 106),
                                       rng.uniform(-1.0, 1.0, (5, 3)),
                                       length_unit='m')
    rigid = rigid_body_shapes(geometry, MassProperties((0, 0, 0)))
    t = np.arange(256) / 256.0
    q = np.zeros((6, len(t)))
    q[5] = np.sin(2 * np.pi * 3 * t)
    ordinate = np.vstack([rigid.shape_matrix.T @ q, np.ones((1, len(t)))])
    history = visualdynamics.TimeHistory(
        t, ordinate, response_dof=[*rigid.coordinate, '103X+'],
        ordinate_dim=['acceleration'] * rigid.num_dofs + ['force'])
    modal, _report = to_modal(history, rigid)
    assert modal.ordinate_dim[5] == 'angular_acceleration'
    assert modal.ordinate_dim[11] == 'moment'
    assert pixels(record_icon(modal, 5)) == \
        pixels(quantity_icon('angular_acceleration'))
    assert pixels(record_icon(modal, 11)) == pixels(quantity_icon('moment'))
    assert pixels(record_icon(modal, 5)) != pixels(quantity_icon('acceleration'))
    assert pixels(record_icon(modal, 0)) == pixels(quantity_icon('acceleration'))


def test_every_icon_is_keyed_by_a_dimension_a_record_can_carry():
    """The bug this pins: the displacement glyph was keyed 'displacement',
    which is the *word* for it — the dimension is 'length', and
    `parse_dimension('displacement')` raises. So `quantity_of` answered
    None for every displacement channel ever measured and the blue trace
    was unreachable on real data, drawn only by tests calling for it by
    name."""
    from visualdynamics.units import parse_dimension

    for quantity in QUANTITIES:
        parse_dimension(quantity)     # raises if no record could carry it


@pytest.mark.parametrize('quantity', QUANTITIES)
def test_each_one_draws_something_of_its_own(qt_app, quantity):
    icon = quantity_icon(quantity)
    assert not icon.isNull()
    assert pixels(icon) != pixels(type_icon('NoSuchType'))


@pytest.mark.parametrize('size', [64, 16])
def test_no_two_quantities_look_the_same(qt_app, size):
    """16 px is where the tree uses them. The derivative dots were a single
    pixel at first, which lost velocity against acceleration at exactly the
    size that matters."""
    seen = {}
    for quantity in QUANTITIES:
        key = pixels(quantity_icon(quantity), size)
        assert key not in seen, \
            f'{quantity} and {seen.get(key)} are identical at {size} px'
        seen[key] = quantity


def test_none_of_them_collide_with_an_object_type_icon(qt_app):
    types = ['TimeHistory', 'Frf', 'Psd', 'Spectrum', 'Coherence',
             'MultipleCoherence', 'Specification', 'Geometry', 'ShapeSet']
    theirs = {pixels(type_icon(name), 16) for name in types}
    for quantity in QUANTITIES:
        assert pixels(quantity_icon(quantity), 16) not in theirs, \
            f'{quantity} looks like an object type'


# --- which records get one ----------------------------------------------------

@pytest.mark.parametrize('dimension,expected', [
    ('acceleration', 'acceleration'),
    ('force', 'force'),
    ('voltage', 'voltage'),
    ('temperature', 'temperature'),
    ('velocity', 'velocity'),
    ('length', 'length'),          # displacement, as a dimension
    # derived quantities belong to their object, whose own shape says more
    ('acceleration/force', None),
    ('acceleration**2/frequency', None),
    ('dimensionless', None),
    ('unknown', None),
    ('', None),
])
def test_only_a_simple_quantity_resolves(dimension, expected):
    assert quantity_of(dimension) == expected


def mixed():
    """What a real test's time data looks like."""
    dims = ['acceleration', 'force', 'voltage', 'temperature']
    units = ['m/s**2', 'N', 'V', 'K']
    return TimeHistory(abscissa=np.arange(4.0), ordinate=np.zeros((4, 4)),
                       response_dof=['1Z+', '1Z+', '2Z+', '3Z+'],
                       ordinate_dim=dims, ordinate_unit=units)


def test_records_of_one_object_get_different_icons(qt_app):
    """The whole point: four records, four quantities, four icons."""
    data = mixed()
    keys = {pixels(record_icon(data, i), 16)
            for i in range(data.num_records)}
    assert len(keys) == data.num_records


def test_an_frf_record_wears_its_fraction(qt_app):
    """An acceleration per force draws both quantities, response over
    reference — and the orientation is load-bearing: acceleration-over-
    force and force-over-acceleration are different measurements."""
    from visualdynamics.gui.icons import ratio_icon

    frf = Frf(abscissa=np.arange(4.0), ordinate=np.ones((1, 4)),
              response_dof=['1Z+'], reference_dof=['2Z+'],
              ordinate_dim='acceleration/force',
              ordinate_unit='m/s**2', reference_unit='N')
    icon = pixels(record_icon(frf, 0))
    assert icon == pixels(ratio_icon('acceleration', 'force'))
    assert icon != pixels(type_icon('Frf'))
    assert icon != pixels(ratio_icon('force', 'acceleration')), (
        'the fraction must not read upside down')
    assert icon != pixels(quantity_icon('acceleration'))


def test_a_psd_record_wears_its_base_quantity(qt_app):
    """A PSD record is accelerations, squared per hertz — the square and
    the density do not change what the channel measured."""
    psd = Psd(abscissa=np.arange(4.0), ordinate=np.ones((1, 4)),
              response_dof=['1Z+'], ordinate_dim='acceleration**2/frequency',
              ordinate_unit='m/s**2')
    assert pixels(record_icon(psd, 0)) == pixels(
        quantity_icon('acceleration'))


def test_a_cpsd_cross_term_wears_the_fraction_and_the_diagonal_does_not(
        qt_app):
    """A cross term is one channel against another: its two quantities
    read as a fraction where they differ, and collapse to the single
    glyph where they are the same."""
    from visualdynamics.gui.icons import ratio_icon

    cpsd = Psd(abscissa=np.arange(4.0),
               ordinate=np.ones((2, 4), dtype=complex),
               response_dof=['1Z+', '1Z+'], reference_dof=['2Z+', '2Z+'],
               ordinate_dim=['acceleration*force/frequency',
                             'acceleration**2/frequency'],
               ordinate_unit='m/s**2')
    assert pixels(record_icon(cpsd, 0)) == pixels(
        ratio_icon('acceleration', 'force'))
    assert pixels(record_icon(cpsd, 1)) == pixels(
        quantity_icon('acceleration'))


def test_an_undeclared_record_still_shows_it_is_undeclared(qt_app):
    """The units badge survives the change of icon source."""
    raw = TimeHistory(abscissa=np.arange(4.0), ordinate=np.zeros((1, 4)),
                      response_dof=['1Z+'])
    assert pixels(record_icon(raw, 0)) == pixels(type_icon('TimeHistory', False))
    assert pixels(record_icon(raw, 0)) != pixels(type_icon('TimeHistory', True))


def test_a_hint_is_enough_to_pick_the_icon(qt_app):
    """A source can say a channel is a force without saying its scale. The
    hint still answers what the record measures, which is what this shows."""
    hinted = TimeHistory(abscissa=np.arange(4.0), ordinate=np.zeros((1, 4)),
                         response_dof=['1Z+'], dimension_hint=['force'])
    assert not hinted.units_defined
    assert pixels(record_icon(hinted, 0)) == pixels(quantity_icon('force', False))
    assert pixels(record_icon(hinted, 0)) != pixels(quantity_icon('force', True)), \
        'and still badged'


def ink_above(icon, size=16, fraction=0.42):
    """How much is drawn in the top band of the icon, where the dots live."""
    image = icon.pixmap(QSize(size, size)).toImage()
    rows = int(size * fraction)
    return sum(image.pixelColor(x, y).alpha() > 60
               for y in range(rows) for x in range(size))


def test_the_derivative_dots_survive_being_shrunk_to_the_tree(qt_app):
    """Not just 'the images differ' — differing by one antialiased pixel is
    not a distinction anybody can see. Each derivative has to put real ink
    above the trace at 16 px, which is where the tree draws it.

    Byte-inequality alone passed with the dots at a quarter of a pixel, so
    this counts them instead.
    """
    counts = {q: ink_above(quantity_icon(q))
              for q in ('length', 'velocity', 'acceleration')}
    assert counts['velocity'] - counts['length'] >= 3, counts
    assert counts['acceleration'] - counts['velocity'] >= 3, counts


def test_length_is_shown_as_displacement_but_stored_as_length(qt_app):
    """The dimension is derived from the unit, and meters cannot say
    whether they are a coordinate or a motion — a geometry's nodes, a
    plate's thickness and a proximity probe's output are all `length`,
    so that is what a record stores. But a channel measured in meters is
    a *displacement*, and 'length' in a Type cell reads as a mistake to
    anyone who has run a survey. The dimension keeps its name; the
    interface uses the word."""
    import numpy as np

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.core.unit_choices import (
        shown_dimension,
        stored_dimension,
    )
    from visualdynamics.gui.object_tables import units_table_model
    from visualdynamics.gui.record_grid import RecordGrid

    assert shown_dimension('length') == 'displacement'
    assert stored_dimension('displacement') == 'length'
    # every other dimension is its own word, unchanged
    for dimension in ('acceleration', 'velocity', 'force', 'strain',
                      'pressure', 'voltage', 'temperature'):
        assert shown_dimension(dimension) == dimension
        assert stored_dimension(dimension) == dimension

    data = TimeHistory(abscissa=np.arange(3.0), ordinate=np.zeros((1, 3)),
                       response_dof=['101Z+'], ordinate_dim='length',
                       ordinate_unit='m')
    assert data.ordinate_dim[0] == 'length', 'stored as the dimension'
    grid = RecordGrid(data)
    assert grid.item(0, 0).toolTip() == '101Z+ — displacement'
    model = units_table_model(data)
    types = [column.title for column in model.columns].index('Type')
    assert model.data(model.index(0, types)) == 'displacement'
    assert 'displacement' in model.columns[types].choices_for(data, 0)
    assert 'length' not in model.columns[types].choices_for(data, 0)
