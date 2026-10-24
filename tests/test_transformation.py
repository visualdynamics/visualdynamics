"""The transformation as an object: a matrix with named inputs and
outputs and the units its entries carry, built from geometry or typed,
applied to records, densities and specifications, inverted with its
conditioning checked, and read for the residual it leaves.

The hand cases are a beam measured at two and three points along X, in
Z: the bounce of the point is the mean, its pitch the difference over
the span, and a bending shape `[1, -2, 1]` is what no rigid motion
explains. Built from geometry the matrix is the one `to_modal` applies
through the same rigid set, so the two are pinned against each other.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.rigid import MassProperties, rigid_body_shapes
from visualdynamics.core.transform import matrix_at, to_modal
from visualdynamics.core.transformation import (
    MAX_CONDITION,
    Transformation,
    read_residual,
)

INCH = 0.0254
T = np.arange(2048) / 1024.0


def _geometry(seed=1, n=5):
    rng = np.random.default_rng(seed)
    return visualdynamics.Geometry(np.arange(101, 101 + n),
                                   rng.uniform(-1.0, 1.0, (n, 3)),
                                   length_unit='m')


def _beam(a=10.0, unit='in'):
    """Bounce and pitch of a point midway between two Z accelerometers
    `a` either side of it along X, typed in `unit`."""
    return Transformation.typed(
        [[0.5, 0.5], [-0.5 / a, 0.5 / a]], ['1Z+', '2Z+'],
        ['1000Z+', '1000RY+'], length_unit=unit)


def _rigid_run(seed=1, samples=16384, flex=None):
    rng = np.random.default_rng(seed)
    geometry = _geometry(seed)
    rigid = rigid_body_shapes(geometry, MassProperties((0.0, 0.0, 0.0)))
    t = np.arange(samples) / 2048.0
    q = rng.standard_normal((6, samples)) * np.array(
        [1.0, 0.5, 0.2, 0.1, 0.3, 0.4])[:, None]
    u = rigid.shape_matrix.T @ q
    if flex is not None:
        u = u + flex(rng, t, u.shape)
    run = visualdynamics.TimeHistory(t, u,
                                     response_dof=list(rigid.coordinate),
                                     ordinate_dim='acceleration')
    run.averaging = Averaging(frame_length=1024, frames=15)
    return geometry, rigid, run


# ---- typed: the hand case ----------------------------------------------------


def test_a_typed_matrix_reads_its_kinds_and_units_from_the_names():
    t = _beam()
    assert t.input_kinds == ('translation', 'translation')
    assert t.output_kinds == ('translation', 'rotation')
    # typed per inch, stored per meter: -0.05/in is -1.9685/m
    assert t.matrix[1, 0] == pytest.approx(-0.05 / INCH)
    assert t.matrix[0, 0] == 0.5, 'a plain number is not scaled'
    assert t.entry_unit(0, 0) == ''
    assert t.entry_unit(1, 0, 'in') == '1/in'
    assert t.length_scale == pytest.approx(INCH)


def test_the_conditioning_of_the_beam_is_the_span_over_the_length_scale():
    """Scaled by one inch, the pitch row is ±0.05 and the bounce row
    0.5: orthogonal rows, singular values √0.5 and √0.005, conditioning
    ten, the half-span in inches."""
    t = _beam(a=10.0)
    assert np.allclose(t.singular_values, [np.sqrt(0.5), np.sqrt(0.005)])
    assert t.condition == pytest.approx(10.0)
    assert 'conditioning 10' in t.describe()


def test_the_beam_recovers_bounce_and_pitch_in_si():
    a = 10.0 * INCH
    z = np.sin(2 * np.pi * 5 * T)
    theta = 0.3 * np.cos(2 * np.pi * 3 * T)
    record = visualdynamics.TimeHistory(
        T, np.vstack([z - a * theta, z + a * theta]),
        response_dof=['1Z+', '2Z+'], ordinate_dim='acceleration')
    out, report = _beam().apply(record)
    assert out.response_dof == ['1000Z+', '1000RY+']
    assert out.ordinate_dim == ['acceleration', 'angular_acceleration'], \
        'a rotation comes out in rad/s², whatever length it was typed in'
    assert np.allclose(out.ordinate[0], z)
    assert np.allclose(out.ordinate[1], theta)
    assert report.shared == {'acceleration': ['1Z+', '2Z+']}
    assert report.worst_residual == pytest.approx(0.0, abs=1e-12)


def test_the_beam_inverts_by_hand():
    """[[1/2, 1/2], [-1/2a, 1/2a]] inverts to [[1, -a], [1, a]]; square,
    so nothing is left over."""
    a = 10.0 * INCH
    back = _beam().inverse()
    assert back.inputs == ('1000Z+', '1000RY+')
    assert back.outputs == ('1Z+', '2Z+')
    assert back.input_kinds == ('translation', 'rotation')
    assert np.allclose(back.matrix, [[1.0, -a], [1.0, a]])
    assert back.entry_unit(0, 1, 'm') == 'm'
    # scaled by the same inch, a rotation column over it: [[1, -10],
    # [1, 10]], the forward matrix's reciprocal singular values
    assert back.condition == pytest.approx(10.0)
    assert np.allclose(_beam().residual_projector, 0.0, atol=1e-12)


def test_the_inverse_carries_the_angular_record_back_to_motions():
    a = 10.0 * INCH
    z, theta = np.sin(2 * np.pi * 5 * T), 0.3 * np.cos(2 * np.pi * 3 * T)
    record = visualdynamics.TimeHistory(
        T, np.vstack([z, theta]), response_dof=['1000Z+', '1000RY+'],
        ordinate_dim=['acceleration', 'angular_acceleration'])
    back, _report = _beam().inverse().apply(record)
    assert back.ordinate_dim == ['acceleration', 'acceleration']
    assert np.allclose(back.ordinate, [z - a * theta, z + a * theta])


def test_a_negated_channel_is_read_with_its_sign():
    a = 10.0 * INCH
    z = np.sin(2 * np.pi * 5 * T)
    record = visualdynamics.TimeHistory(
        T, np.vstack([-z, z]), response_dof=['1Z-', '2Z+'],
        ordinate_dim='acceleration')
    out, _report = _beam().apply(record)
    assert np.allclose(out.ordinate[0], z)
    assert np.allclose(out.ordinate[1], 0.0, atol=1e-12)
    del a


def test_the_table_writes_each_row_in_the_unit_asked_for():
    headings, rows = _beam().table('in')
    assert headings == ['Output', '1Z+', '2Z+', 'Unit']
    assert rows[0] == ['1000Z+', '0.5', '0.5', '1']
    assert rows[1] == ['1000RY+', '-0.05', '0.05', '1/in']
    _headings, metric = _beam().table('m')
    assert float(metric[1][2]) == pytest.approx(0.05 / INCH, rel=1e-5)
    # the pseudo-inverse's round-off is written as the zero it is
    _headings, rows = Transformation.typed(
        [[1.0, 1e-20], [0.0, 1.0]], ['1Z+', '2Z+'], ['3Z+', '4Z+']).table()
    assert rows[0][1:3] == ['1', '0']
    # judged against the row's largest, not against one
    _headings, rows = Transformation.typed(
        [[1e6, 1e-7]], ['1Z+', '2Z+'], ['3Z+']).table()
    assert rows[0][1:3] == ['1e+06', '0']


# ---- built from geometry: the matrix to_modal applies ------------------------


def test_from_geometry_is_the_response_transformation_of_the_rigid_set():
    geometry, rigid, _run = _rigid_run()
    t = Transformation.from_geometry(geometry, (0.0, 0.0, 0.0),
                                     rigid.coordinate)
    view = matrix_at(rigid, rigid.coordinate, 'response')
    assert np.allclose(t.matrix, view.values)
    assert t.outputs == ('M1', 'M2', 'M3', 'M4', 'M5', 'M6')
    assert t.output_kinds[3:] == ('rotation',) * 3
    assert 'rigid body about (0, 0, 0) m' in t.origin
    assert t.matrix_at().values.shape == (6, 15)
    # the length scale is the RMS lever arm: √(mean |r|² − (r·d)²)
    xyz = geometry.node_xyz
    arms = [np.sum(np.cross(np.eye(3), r)[:, k] ** 2)
            for r in xyz for k in range(3)]
    assert t.length_scale == pytest.approx(np.sqrt(np.mean(arms)))


def test_applied_it_agrees_with_to_modal_on_rows_and_densities():
    geometry, rigid, run = _rigid_run()
    t = Transformation.from_geometry(geometry, (0.0, 0.0, 0.0),
                                     rigid.coordinate)
    ours, _r = t.apply(run)
    theirs, _r = to_modal(run, rigid)
    assert np.allclose(ours.ordinate, theirs.ordinate, atol=1e-12)
    assert ours.ordinate_dim == theirs.ordinate_dim
    cpsd = run.compute_cpsds()
    ours, report = t.apply(cpsd)
    theirs, _r = to_modal(cpsd, rigid)
    assert ours.num_records == 36
    order = {(r, c): i for i, (r, c) in enumerate(
        zip(theirs.response_dof, theirs.reference_dof))}
    for i, pair in enumerate(zip(ours.response_dof, ours.reference_dof)):
        j = order[pair]
        assert ours.ordinate_dim[i] == theirs.ordinate_dim[j]
        assert np.allclose(ours.ordinate[i], theirs.ordinate[j],
                           rtol=1e-10, atol=1e-18)
    assert report.worst_residual == pytest.approx(0.0, abs=1e-9)


def test_a_specification_keeps_its_bands_through_the_matrix():
    geometry, rigid, run = _rigid_run(3)
    cpsd = run.compute_cpsds()
    autos = [i for i, (r, c) in enumerate(zip(cpsd.response_dof,
                                              cpsd.reference_dof)) if r == c]
    upper = np.full(cpsd.ordinate.shape, np.nan)
    upper[autos] = 2.0 * np.real(cpsd.ordinate[autos])
    spec = visualdynamics.Specification(
        cpsd.abscissa, cpsd.ordinate, response_dof=list(cpsd.response_dof),
        reference_dof=list(cpsd.reference_dof),
        ordinate_dim=list(cpsd.ordinate_dim), abort_upper=upper)
    t = Transformation.from_geometry(geometry, (0.0, 0.0, 0.0),
                                     rigid.coordinate)
    out, _report = t.apply(spec)
    assert isinstance(out, visualdynamics.Specification)
    i = list(zip(out.response_dof, out.reference_dof)).index(('M5', 'M5'))
    assert np.allclose(out.limits['abort_upper'][i],
                       2.0 * np.real(out.ordinate[i]))


# ---- refusals in the core ----------------------------------------------------


def test_too_few_directions_are_refused_by_rank():
    _unused, rigid, _run = _rigid_run()
    with pytest.raises(ValueError, match='rank 5 of 6, so 1 direction '
                                         'cannot be resolved'):
        Transformation.from_shapes(rigid, rigid.coordinate[:5])
    with pytest.raises(ValueError, match='rank 4 of 6, so 2 directions'):
        Transformation.from_shapes(rigid, rigid.coordinate[:4])


def test_a_short_rank_conditions_infinitely_and_does_not_invert():
    flat = Transformation.typed([[1.0, 2.0], [2.0, 4.0]], ['1Z+', '2Z+'],
                                ['3Z+', '4Z+'])
    assert flat.condition == float('inf')
    assert 'conditioning infinite (short of full rank)' in flat.describe()
    with pytest.raises(ValueError, match='rank 1 of 2'):
        flat.inverse()
    zero = Transformation.typed([[0.0, 0.0]], ['1Z+', '2Z+'], ['3Z+'])
    assert zero.condition == float('inf')
    with pytest.raises(ValueError, match='rank 0 of 1'):
        zero.inverse()


def test_every_input_on_the_point_is_refused():
    geometry = visualdynamics.Geometry(np.arange(1, 4), np.zeros((3, 3)),
                                       length_unit='m')
    rigid = rigid_body_shapes(geometry, MassProperties((0.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match='every input is on the point'):
        Transformation.from_shapes(rigid, rigid.coordinate)


def test_a_poor_layout_is_refused_by_its_conditioning():
    """Six sensors nearly on one line about the point: rank six, but
    the rotation about that line is barely seen."""
    xyz = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                    [0.5, 1e-3, 0.0], [-0.5, 0.0, 1e-3]])
    geometry = visualdynamics.Geometry(np.arange(1, 5), xyz, length_unit='m')
    rigid = rigid_body_shapes(geometry, MassProperties((0.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match=r'conditioning is [0-9.e+]+, past '
                                         r'the limit of 100'):
        Transformation.from_shapes(rigid, rigid.coordinate)
    loose = Transformation.from_shapes(rigid, rigid.coordinate,
                                       max_condition=np.inf)
    assert loose.condition > MAX_CONDITION
    values = loose.singular_values
    # the shapes inverted have the reciprocals of the result's
    with pytest.raises(ValueError, match=rf'has singular values '
                                         rf'{1 / values[-1]:.4g} to '
                                         rf'{1 / values[0]:.4g}\)'):
        Transformation.from_shapes(rigid, rigid.coordinate)
    with pytest.raises(ValueError, match='the inverse: the conditioning'):
        loose.inverse()


def test_an_input_the_set_lacks_is_refused_by_name():
    _unused, rigid, _run = _rigid_run()
    with pytest.raises(ValueError, match=r'999X\+ is not a DOF of the set'
                                         r': the shapes'):
        Transformation.from_shapes(rigid, [*rigid.coordinate, '999X+'])
    with pytest.raises(ValueError, match=r'999X\+ is not a DOF of the set '
                                         r'\(nor 1 more\)'):
        Transformation.from_shapes(rigid, [*rigid.coordinate, '999X+',
                                           '998X+'])


def test_a_mass_normalized_set_is_refused():
    geometry = _geometry()
    scaled = rigid_body_shapes(geometry, MassProperties(
        (0.0, 0.0, 0.0), 2.0, (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match='mass-normalized'):
        Transformation.from_shapes(scaled, scaled.coordinate)


def test_geometry_without_units_is_refused():
    geometry = visualdynamics.Geometry(np.arange(1, 4), np.eye(3))
    with pytest.raises(ValueError, match='length unit is not declared'):
        Transformation.from_geometry(geometry, (0, 0, 0), ['1X+'])


@pytest.mark.parametrize('matrix,inputs,outputs,kinds,match', [
    ([[1.0j, 0.0]], ['1Z+', '2Z+'], ['3Z+'], None, 'real'),
    ([[1.0, 0.0, 0.0]], ['1Z+', '2Z+'], ['3Z+'], None, 'is 1 by 3'),
    ([[1.0, np.nan]], ['1Z+', '2Z+'], ['3Z+'], None, 'not a finite'),
    ([[1.0, 1.0]], ['1Z+', '1Z-'], ['3Z+'], None, 'named twice'),
    ([[1.0, 1.0]], ['1Z+', '2Z+'], ['M1'], None, 'state the output kinds'),
    ([[1.0, 1.0]], ['1Z+', '2Z+'], ['M1'], ['spin'], 'not a kind'),
    ([1.0, 1.0], ['1Z+', '2Z+'], ['3Z+'], None, 'a matrix of outputs by'),
    (np.zeros((0, 2)), ['1Z+', '2Z+'], [], None, 'a matrix of outputs by'),
])
def test_a_typed_matrix_that_does_not_agree_is_refused(matrix, inputs,
                                                      outputs, kinds, match):
    with pytest.raises(ValueError, match=match):
        Transformation.typed(matrix, inputs, outputs, output_kinds=kinds)


def test_kinds_and_a_length_scale_that_do_not_agree_are_refused():
    with pytest.raises(ValueError, match='each input is one of'):
        Transformation(np.eye(2), ('1Z+', '2Z+'), ('3Z+', '4Z+'),
                       ('translation',), ('translation', 'translation'))
    with pytest.raises(ValueError, match='positive number of meters'):
        Transformation(np.eye(1), ('1Z+',), ('3Z+',), ('translation',),
                       ('translation',), length_scale=0.0)


def test_one_input_to_one_output_is_said_in_the_singular():
    one = Transformation.typed([[2.0]], ['1Z+'], ['3Z+'])
    assert one.describe().startswith('1 input to 1 output, conditioning 1,')


def test_a_length_unit_that_is_not_a_length_is_refused():
    with pytest.raises(ValueError, match="'kg' is not a length unit"):
        _beam(unit='kg')


def _beam_record(dims='acceleration', dofs=('1Z+', '2Z+'), values=None):
    values = np.vstack([np.sin(T), np.cos(T)]) if values is None else values
    return visualdynamics.TimeHistory(T, values, response_dof=list(dofs),
                                      ordinate_dim=dims)


def test_records_at_other_dofs_are_left_out_and_listed():
    values = np.vstack([np.sin(T), np.cos(T), np.zeros_like(T)])
    record = _beam_record(dofs=('1Z+', '2Z+', '9Z+'), values=values)
    _out, report = _beam().apply(record)
    assert report.dropped == ['9Z+']


def test_no_motion_at_the_inputs_says_where_it_looked():
    record = _beam_record(dims='temperature')
    with pytest.raises(ValueError, match=r"transformation's inputs \(1Z\+, "
                                         r"2Z\+\)$"):
        _beam().apply(record)
    four = Transformation.typed(np.ones((1, 4)), ['1Z+', '2Z+', '3Z+', '4Z+'],
                                ['9Z+'])
    with pytest.raises(ValueError, match=r'\(1Z\+, 2Z\+, 3Z\+ …\)$'):
        four.apply(record)


def test_the_rows_residual_is_each_channels_share_of_its_rms():
    """Twice the bending shape at three points: no rigid motion has
    it, so every channel's residual is all of its RMS, and a channel
    that holds nothing has none."""
    t, _psd, _rigid, v = _three_point(lambda f: np.zeros_like(f))
    record = visualdynamics.TimeHistory(
        T, 2.0 * np.outer(v, np.sin(2 * np.pi * 5 * T)),
        response_dof=['1Z+', '2Z+', '3Z+'], ordinate_dim='acceleration')
    _out, report = t.apply(record)
    assert report.residual == pytest.approx({'1Z+': 1.0, '2Z+': 1.0,
                                             '3Z+': 1.0})
    small = visualdynamics.TimeHistory(
        T, 0.1 * np.outer(v, np.sin(2 * np.pi * 5 * T)),
        response_dof=['1Z+', '2Z+', '3Z+'], ordinate_dim='acceleration')
    _out, report = t.apply(small)
    assert report.residual['1Z+'] == pytest.approx(1.0), \
        'a share of the RMS, whatever the RMS is'
    quiet = visualdynamics.TimeHistory(
        T, np.vstack([np.sin(T), np.zeros_like(T)]),
        response_dof=['1Z+', '2Z+'], ordinate_dim='acceleration')
    _out, report = _beam().apply(quiet)
    assert report.residual['2Z+'] == 0.0


def test_a_missing_input_is_refused_by_name():
    record = _beam_record(dofs=('1Z+', '3Z+'))
    with pytest.raises(ValueError, match=r'no record at 2Z\+: the '
                                         'transformation needs every one'):
        _beam().apply(record)


def test_undefined_units_are_refused_by_name():
    record = visualdynamics.TimeHistory(T, np.vstack([np.sin(T), np.cos(T)]),
                                        response_dof=['1Z+', '2Z+'],
                                        dimension_hint='acceleration')
    with pytest.raises(ValueError, match=r'1Z\+ has no declared units'):
        _beam().apply(record)


def test_a_rotation_where_a_translation_belongs_is_refused():
    record = _beam_record(dims=['acceleration', 'angular_acceleration'])
    with pytest.raises(ValueError, match=r'2Z\+ holds angular_acceleration, '
                                         'but the transformation takes '
                                         'acceleration'):
        _beam().apply(record)


def test_two_motions_at_the_inputs_ask_which():
    values = np.vstack([np.sin(T), np.cos(T), np.sin(T), np.cos(T)])
    record = _beam_record(
        dims=['acceleration', 'acceleration', 'velocity', 'velocity'],
        dofs=('1Z+', '2Z+', '1Z+', '2Z+'), values=values)
    with pytest.raises(ValueError, match='name the quantity'):
        _beam().apply(record)
    out, report = _beam().apply(record, quantity='velocity')
    assert out.ordinate_dim == ['velocity', 'angular_velocity']
    assert report.skipped == {'acceleration': 2}, \
        'the other motion at the inputs is left out and counted'
    assert report.dropped == [], 'no DOF is outside the transformation'


def test_a_sample_that_is_not_a_number_is_refused_by_name():
    values = np.vstack([np.sin(T), np.cos(T)])
    values[1, 7] = np.nan
    with pytest.raises(ValueError, match=r'2Z\+ holds a value that is not '
                                         'a finite number'):
        _beam().apply(_beam_record(values=values))


def test_an_empty_pick_is_refused():
    with pytest.raises(ValueError, match='no records picked'):
        _beam().apply(_beam_record(), records=[])


def test_autospectra_alone_and_an_frf_are_refused():
    psd = visualdynamics.Psd(np.linspace(1, 10, 10), np.ones((2, 10)),
                             response_dof=['1Z+', '2Z+'],
                             ordinate_dim='acceleration**2/frequency')
    with pytest.raises(ValueError, match='autospectra only'):
        _beam().apply(psd)
    frf = visualdynamics.Frf(np.linspace(1, 10, 10), np.ones((2, 10)),
                             response_dof=['1Z+', '2Z+'],
                             reference_dof=['9Z+', '9Z+'],
                             ordinate_dim='acceleration/force')
    with pytest.raises(TypeError, match='both halves'):
        _beam().apply(frf)


# ---- the residual: three points on a beam ------------------------------------


def _three_point(bend, f=None):
    """The density at Z sensors at x = -a, 0, a: unit bounce and a
    tenth of a unit of pitch, uncorrelated, plus `bend(f)` of the
    bending shape [1, -2, 1]."""
    a = 0.2
    f = np.linspace(10.0, 500.0, 50) if f is None else f
    phi = np.array([[1.0, -a], [1.0, 0.0], [1.0, a]])
    rigid = phi @ np.diag([1.0, 0.1]) @ phi.T
    v = np.array([1.0, -2.0, 1.0])
    matrix = rigid[None] + bend(f)[:, None, None] * np.outer(v, v)[None]
    dofs = ['1Z+', '2Z+', '3Z+']
    rows, resp, refs = [], [], []
    for i, di in enumerate(dofs):
        for j, dj in enumerate(dofs):
            rows.append(matrix[:, i, j])
            resp.append(di)
            refs.append(dj)
    psd = visualdynamics.Psd(f, np.asarray(rows, dtype=complex),
                             response_dof=resp, reference_dof=refs,
                             ordinate_dim='acceleration**2/frequency')
    t = Transformation.typed(np.linalg.pinv(phi), dofs, ['9Z+', '9RY+'])
    return t, psd, rigid, v


def test_a_rigid_density_leaves_no_residual():
    t, psd, _rigid, _v = _three_point(lambda f: np.zeros_like(f))
    reading = t.read_residual(psd)
    assert reading.overall == pytest.approx(0.0, abs=1e-12)
    assert reading.holds_throughout
    assert reading.holds_to == psd.abscissa[-1], 'held to the last line'
    assert np.allclose(reading.measured, reading.explained)


def test_the_residual_is_the_bending_share_by_hand():
    """The bending shape is orthogonal to bounce and pitch, so all of
    it is residual: 6b² of the trace (|v|² = 6) against the rigid
    trace 3 + 0.1·2a², at every line."""
    b = 0.05
    t, psd, rigid, _v = _three_point(lambda f: np.full_like(f, b))
    reading = read_residual(t, psd)
    share = 6 * b / (np.trace(rigid) + 6 * b)
    assert np.allclose(reading.fraction, share)
    assert np.allclose(reading.residual[1], 4 * b), 'the middle point, 2²b'
    assert np.allclose(reading.measured, reading.explained + reading.residual)
    assert reading.worst_channel == '2Z+'
    assert reading.overall == pytest.approx(share)
    assert reading.channel_fractions[1] == pytest.approx(
        4 * b / (rigid[1, 1] + 4 * b))
    _out, report = t.apply(psd)
    assert report.residual['2Z+'] == pytest.approx(
        4 * b / (rigid[1, 1] + 4 * b))


def test_a_missing_density_input_is_refused_by_name():
    t, psd, _rigid, _v = _three_point(lambda f: np.zeros_like(f))
    keep = [i for i, (r, c) in enumerate(zip(psd.response_dof,
                                             psd.reference_dof))
            if '3Z+' not in (r, c)]
    with pytest.raises(ValueError, match=r'no acceleration density at 3Z\+'):
        t.apply(psd, records=keep)
    extra = visualdynamics.Psd(
        psd.abscissa, np.vstack([psd.ordinate, psd.ordinate[:1]]),
        response_dof=[*psd.response_dof, '9Z+'],
        reference_dof=[*psd.reference_dof, '9Z+'],
        ordinate_dim='acceleration**2/frequency')
    _out, report = t.apply(extra)
    assert report.dropped == ['9Z+']


def test_holds_to_is_the_last_line_under_the_threshold():
    f = np.linspace(10.0, 500.0, 50)
    t, psd, _rigid, _v = _three_point(
        lambda f: np.where(f >= 300.0, 1.0, 1e-4), f)
    reading = read_residual(t, psd, threshold=0.1)
    assert reading.holds_to == pytest.approx(f[f < 300.0][-1])
    assert not reading.holds_throughout
    assert 'under 10% up to 290 Hz' in reading.describe()
    for threshold in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match='above 0 and under 1'):
            read_residual(t, psd, threshold=threshold)


def test_a_dead_channel_and_an_empty_line_are_said():
    f = np.linspace(10.0, 500.0, 50)
    t, psd, _rigid, _v = _three_point(lambda f: np.zeros_like(f), f)
    ordinate = np.array(psd.ordinate)
    ordinate[:, :2] = 0.0                # nothing at the first two lines
    pairs = list(zip(psd.response_dof, psd.reference_dof))
    for i, (r, c) in enumerate(pairs):
        if '3Z+' in (r, c):
            ordinate[i] = 0.0            # and nothing ever at 3Z+
    dead = visualdynamics.Psd(f, ordinate, response_dof=psd.response_dof,
                              reference_dof=psd.reference_dof,
                              ordinate_dim='acceleration**2/frequency')
    reading = read_residual(t, dead)
    assert np.isnan(reading.fraction[0]), 'no power, no share'
    assert np.isnan(reading.channel_fractions[2])
    assert any('3Z+ measured nothing, so it says' in note
               for note in reading.notes)
    assert any('2 lines with no measured power' in note
               for note in reading.notes)
    for i, (r, c) in enumerate(pairs):
        if '2Z+' in (r, c):
            ordinate[i] = 0.0
    reading = read_residual(t, visualdynamics.Psd(
        f, ordinate, response_dof=psd.response_dof,
        reference_dof=psd.reference_dof,
        ordinate_dim='acceleration**2/frequency'))
    assert any('2Z+, 3Z+ measured nothing, so they say' in note
               for note in reading.notes)
    silent = read_residual(t, visualdynamics.Psd(
        f, np.zeros_like(ordinate), response_dof=psd.response_dof,
        reference_dof=psd.reference_dof,
        ordinate_dim='acceleration**2/frequency'))
    assert silent.describe() == 'nothing measured at the inputs'
    assert not any('with no measured power' in note for note in silent.notes)
    assert silent.holds_to is None and silent.worst_channel is None


def test_a_square_transformation_says_its_residual_is_nothing_by_design():
    t = _beam()
    psd = visualdynamics.Psd(
        np.linspace(1, 10, 10), np.ones((4, 10), dtype=complex),
        response_dof=['1Z+', '1Z+', '2Z+', '2Z+'],
        reference_dof=['1Z+', '2Z+', '1Z+', '2Z+'],
        ordinate_dim='acceleration**2/frequency')
    reading = read_residual(t, psd)
    assert any('square and invertible' in note for note in reading.notes)


def test_a_rotation_input_has_no_summed_residual():
    with pytest.raises(ValueError, match='must all be translations'):
        read_residual(_beam().inverse(), visualdynamics.Psd(
            np.ones(2), np.ones((1, 2)), response_dof=['1000Z+'],
            reference_dof=['1000Z+'],
            ordinate_dim='acceleration**2/frequency'))


# ---- through the project ----------------------------------------------------


def _project(flexible=True):
    def flex(rng, t, shape):
        # broadband noise above 500 Hz on every channel, independent:
        # the article flexing where the rigid motion stops explaining it
        noise = rng.standard_normal(shape)
        spectrum = np.fft.rfft(noise, axis=1)
        spectrum[:, np.fft.rfftfreq(shape[1], t[1] - t[0]) < 500.0] = 0.0
        return 2.0 * np.fft.irfft(spectrum, shape[1], axis=1)

    _unused, rigid, run = _rigid_run(7, flex=flex if flexible else None)
    project = visualdynamics.Project()
    project.add('Run', run)
    project.add('Rigid', rigid)
    cpsds = project.compute_cpsds('Run')
    modal = project.transform(cpsds, 'Rigid')
    return project, modal


def test_the_project_reads_where_the_rigid_body_holds():
    project, modal = _project()
    reading = project.read_transformation(modal)
    assert 480.0 <= reading.holds_to <= 500.0
    assert reading.transformation.inputs == tuple(
        project['Rigid'].coordinate)
    loose = project.read_transformation(modal, threshold=0.99)
    assert loose.holds_throughout


def test_the_project_refuses_what_was_not_transformed_from_a_density():
    project, _modal = _project(False)
    other = project.transform('Run', 'Rigid')
    with pytest.raises(ValueError, match='not transformed in this project '
                                         'from a cross-spectral density'):
        project.read_transformation(other)


def test_the_project_builds_a_sets_transformation_at_chosen_dofs():
    project, _modal = _project(False)
    chosen = list(project['Rigid'].coordinate)[:9]
    t = project.transformation_of('Rigid', chosen)
    assert t.inputs == tuple(chosen)
    with pytest.raises(TypeError, match='not a shape set'):
        project.transformation_of('Run')


def test_a_density_with_other_channels_is_split_to_its_inputs():
    """A cross-spectral matrix of the two inputs' accelerations, their
    velocities and an accelerometer elsewhere: the terms touching the
    other DOF are dropped, the velocity terms are skipped and counted,
    and what is left transforms as the inputs' matrix alone does."""
    from visualdynamics.core.averaging import Averaging

    rng = np.random.default_rng(7)
    a = rng.standard_normal((3, len(T)))
    v = rng.standard_normal((2, len(T)))
    whole = visualdynamics.TimeHistory(
        T, np.vstack([a, v]),
        response_dof=['1Z+', '2Z+', '9Z+', '1Z+', '2Z+'],
        ordinate_dim=['acceleration'] * 3 + ['velocity'] * 2)
    alone = visualdynamics.TimeHistory(
        T, a[:2], response_dof=['1Z+', '2Z+'],
        ordinate_dim=['acceleration'] * 2)
    for history in (whole, alone):
        history.averaging = Averaging(frame_length=256, overlap=0.5).filled(
            len(T), 1024.0)
    out, report = _beam().apply(whole.compute_cpsds(),
                                quantity='acceleration')
    expected, _report = _beam().apply(alone.compute_cpsds())
    assert report.dropped == ['9Z+']
    assert sum(report.skipped.values()) == 12
    assert set(report.skipped) == {'velocity**2/frequency',
                                   'acceleration*velocity/frequency',
                                   'velocity*acceleration/frequency'}
    np.testing.assert_allclose(out.ordinate, expected.ordinate)


def test_three_inputs_are_all_named_and_no_more_implied():
    three = Transformation.typed(np.ones((1, 3)), ['1Z+', '2Z+', '3Z+'],
                                 ['9Z+'])
    with pytest.raises(ValueError, match=r'\(1Z\+, 2Z\+, 3Z\+\)$'):
        three.apply(_beam_record(dims='temperature'))
