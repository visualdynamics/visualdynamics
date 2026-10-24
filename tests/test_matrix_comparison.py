"""The matrix comparison: a measured cross-spectral matrix judged against
its specification, levels on the diagonal and each cross term's
coherence and phase off it.

The hand matrices here are two and three channels on a few lines, with
every entry written out, so each coherence and phase the comparison
reads can be checked by hand: |S12|^2 / (S11 S22) and arg S12.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from visualdynamics.core.compliance import (
    CROSS_TERM_REFUSAL,
    compare,
    compare_all,
    detect_scale_db,
)
from visualdynamics.core.data import Psd, Specification, autospectrum
from visualdynamics.core.matrix_comparison import (
    COHERENCE_FLOOR,
    COHERENCE_TOLERANCE,
    PHASE_TOLERANCE,
    channel_labels,
    compare_matrix,
    cross_term_lines,
    record_grid,
    wrapped_degrees,
)

F = np.arange(20.0, 420.0, 10.0)            # 40 lines, 10 Hz apart
DOFS = ['1X+', '2X+', '3X+']


def term(coherence, phase, a, b):
    """A cross term from its coherence, phase in degrees and the two
    autospectra: sqrt(coherence a b) e^(i phase)."""
    return (np.sqrt(np.asarray(coherence) * a * b)
            * np.exp(1j * np.radians(phase)))


def matrix(autos, crosses, cls=Psd, f=F, dims=None, limits=True):
    """A full matrix, records row by row, from its autospectra (one per
    channel) and its cross terms {(i, j): values, i < j}; the term the
    other way is the conjugate. A cross term not given is left out. A
    specification is read per line and carries abort limits 6 dB
    either side of each autospectrum."""
    n = len(autos)
    rows, responses, references, record_dims = [], [], [], []
    lows, highs = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                values = np.broadcast_to(autos[i], f.shape).astype(complex)
            elif (min(i, j), max(i, j)) in crosses:
                values = np.broadcast_to(crosses[(min(i, j), max(i, j))],
                                         f.shape).astype(complex)
                values = values if i < j else np.conj(values)
            else:
                continue
            rows.append(values)
            responses.append(DOFS[i])
            references.append(DOFS[j])
            record_dims.append('acceleration**2/frequency' if dims is None
                               else dims[(i, j)])
            blank = np.full(f.shape, np.nan)
            lows.append(np.real(values) * 10 ** -0.6 if i == j else blank)
            highs.append(np.real(values) * 10 ** 0.6 if i == j else blank)
    kwargs = {}
    if cls is Specification and limits:
        kwargs = {'abort_lower': np.array(lows), 'abort_upper': np.array(highs)}
    out = cls(f, np.array(rows), response_dof=responses,
              reference_dof=references, ordinate_dim=record_dims, **kwargs)
    if cls is Specification:
        out.interpolation = 'bin'
    return out


#: the specification most tests judge against: two channels at 1 and 4,
#: coherent at 0.64 with a 30 degree phase
SPEC = matrix([1.0, 4.0], {(0, 1): term(0.64, 30.0, 1.0, 4.0)}, Specification)


def measured(coherence=0.64, phase=30.0, autos=(1.0, 4.0)):
    return matrix(list(autos),
                  {(0, 1): term(coherence, phase, autos[0], autos[1])})


# ---- the per-line judgement, by hand -----------------------------------------


def test_a_cross_term_is_judged_line_by_line_against_its_tolerances():
    """Coherence errors -0.05, -0.15, 0 and -0.2; phase errors 5, 0,
    15 (170 against -175, the short way) and not judged where the
    specified coherence, 0.3, is under the floor."""
    lines = cross_term_lines([0.8, 0.8, 0.8, 0.3], [30, 30, 170, 0],
                             [0.75, 0.65, 0.8, 0.1], [35, 30, -175, 90])
    np.testing.assert_allclose(lines['coherence_error'],
                               [-0.05, -0.15, 0.0, -0.2], atol=1e-12)
    assert lines['coherence_out'].tolist() == [False, True, False, True]
    np.testing.assert_allclose(lines['phase_error'][:3], [5.0, 0.0, 15.0])
    assert lines['phase_judged'].tolist() == [True, True, True, False]
    assert lines['phase_out'].tolist() == [False, False, True, False]


def test_a_reading_exactly_on_its_tolerance_is_inside():
    lines = cross_term_lines([0.5, 0.5], [0.0, 0.0], [0.6, 0.4],
                             [PHASE_TOLERANCE, -PHASE_TOLERANCE])
    assert not lines['coherence_out'].any()
    assert not lines['phase_out'].any()
    lines = cross_term_lines([0.5], [0.0], [0.6 + 1e-6], [10.001])
    assert lines['coherence_out'].all() and lines['phase_out'].all()


def test_a_phase_is_taken_the_short_way_round():
    np.testing.assert_allclose(wrapped_degrees([350.0, 180.0, -190.0, 10.0]),
                               [-10.0, -180.0, 170.0, 10.0])


def test_a_line_that_is_not_a_number_is_not_judged():
    lines = cross_term_lines([0.8, np.nan, 0.8], [0, 0, 0],
                             [np.nan, 0.8, 0.8], [0, 0, np.nan])
    assert lines['coherence_judged'].tolist() == [False, False, True]
    assert lines['phase_judged'].tolist() == [False, False, False]


@pytest.mark.parametrize('kwargs, message', [
    ({'coherence_tolerance': 0.0}, 'coherence tolerance'),
    ({'coherence_tolerance': 1.5}, 'coherence tolerance'),
    ({'phase_tolerance': -1.0}, 'phase tolerance'),
    ({'phase_tolerance': 200.0}, 'phase tolerance'),
    ({'coherence_floor': 1.0}, 'coherence floor'),
    ({'coherence_floor': np.nan}, 'coherence floor'),
    ({'coherence_tolerance': 'wide'}, 'are numbers'),
])
def test_a_tolerance_out_of_range_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        cross_term_lines([0.5], [0.0], [0.5], [0.0], **kwargs)
    with pytest.raises(ValueError, match=message):
        compare_matrix(SPEC, measured(), **kwargs)


def test_arrays_of_different_lengths_are_refused():
    with pytest.raises(ValueError, match='the same lines'):
        cross_term_lines([0.5, 0.5], [0.0], [0.5], [0.0])


# ---- the matrix, by hand -------------------------------------------------------


def test_the_records_are_laid_out_as_the_matrix():
    channels, grid = record_grid(SPEC)
    assert channels == [('1X+', 'acceleration'), ('2X+', 'acceleration')]
    assert grid == [[0, 1], [2, 3]]
    assert channel_labels([('1X+', 'acceleration'), ('1X+', 'force'),
                           ('2X+', 'acceleration')]) == [
        '1X+ acceleration', '1X+ force', '2X+']


def test_a_measurement_that_is_its_specification_passes_everywhere():
    result = compare_matrix(SPEC, measured(), scale_db=0)
    assert result.passed is True
    pair = result.pair(0, 1)
    np.testing.assert_allclose(pair.specified_coherence, 0.64)
    np.testing.assert_allclose(pair.measured_coherence, 0.64)
    np.testing.assert_allclose(pair.specified_phase, 30.0)
    assert pair.coherence_lines == pair.phase_lines == F.size
    assert result.describe() == ('passes: levels pass on 2 channels; '
                                 '0 of 1 cross term out (coherence 0, '
                                 'phase 0)')
    assert result.cell(0, 0).words.startswith('in: ')
    assert result.cell(0, 1).words == (
        'coherence in: 0 of 40 lines past ±0.10')
    assert result.cell(1, 0).words == 'phase in: 0 of 40 lines past ±10°'


def test_a_coherence_off_by_more_than_its_tolerance_is_out():
    """0.49 against 0.64: |S12|^2 / (S11 S22) = 1.96 / 4."""
    result = compare_matrix(SPEC, measured(coherence=0.49), scale_db=0)
    pair = result.pair(0, 1)
    np.testing.assert_allclose(pair.measured_coherence, 0.49)
    assert (pair.coherence_state, pair.phase_state) == ('out', 'in')
    assert result.passed is False
    assert result.cell(0, 1).state == 'out'
    assert result.cell(0, 1).words == (
        'coherence out: 100.0 % of 40 lines past ±0.10')
    assert 'levels pass' in result.describe()
    assert '1 of 1 cross term out (coherence 1, phase 0)' in result.describe()


def test_a_phase_off_by_more_than_its_tolerance_is_out():
    result = compare_matrix(SPEC, measured(phase=45.0), scale_db=0)
    pair = result.pair(0, 1)
    np.testing.assert_allclose(pair.lines['phase_error'], 15.0)
    assert (pair.coherence_state, pair.phase_state) == ('in', 'out')
    assert result.cell(1, 0).reading == 'phase'
    assert result.cell(1, 0).label == '2X+/1X+'
    assert result.describe() == ('fails: levels pass on 2 channels; 1 of 1 '
                                 'cross term out (coherence 0, phase 1)')
    rows = result.table()
    assert [(row['reading'], row['value'], row['state']) for row in rows] == [
        ('level', '+0.0 dB', 'in'), ('level', '+0.0 dB', 'in'),
        ('coherence', '0.0 % out', 'in'), ('phase', '100.0 % out', 'out')]
    assert rows[2]['words'] == 'coherence in: 0 of 40 lines past ±0.10'
    assert rows[3]['words'] == 'phase out: 100.0 % of 40 lines past ±10°'
    assert result.levels[0].words() == (
        'in: 0.0 % of lines outside abort, +0.0 dB rms')


def test_a_cross_term_is_out_past_the_share_a_level_is_allowed():
    """Four lines of forty out is ten percent, which a level is allowed
    outside its abort limits; five is over it."""
    for count, state in ((4, 'in'), (5, 'out')):
        coherence = np.full(F.shape, 0.64)
        coherence[:count] = 0.3
        result = compare_matrix(SPEC, measured(coherence=coherence),
                                scale_db=0)
        pair = result.pair(0, 1)
        assert pair.coherence_state == state
        assert pair.coherence_percent == pytest.approx(100.0 * count / 40)


def test_the_phase_is_not_judged_under_the_coherence_floor():
    spec = matrix([1.0, 4.0], {(0, 1): term(0.3, 30.0, 1.0, 4.0)},
                  Specification)
    result = compare_matrix(spec, measured(coherence=0.3, phase=120.0),
                            scale_db=0)
    pair = result.pair(0, 1)
    assert (pair.coherence_state, pair.phase_state) == ('in', 'not judged')
    assert pair.reason == ''
    assert result.cell(1, 0).words == (
        'phase not judged: specified coherence under 0.50')
    assert result.passed is True
    lowered = compare_matrix(spec, measured(coherence=0.3, phase=120.0),
                             scale_db=0, coherence_floor=0.25)
    assert lowered.pair(0, 1).phase_state == 'out'


def test_two_channels_stated_independent_are_judged_at_zero():
    """A coherence of zero is a statement and a value, not a gap: the
    cross term is written as zero, and a correlated measurement fails
    it."""
    spec = matrix([1.0, 4.0], {(0, 1): np.zeros(F.shape)}, Specification)
    good = compare_matrix(spec, measured(coherence=0.02), scale_db=0)
    assert good.pair(0, 1).coherence_state == 'in'
    assert good.pair(0, 1).phase_state == 'not judged'
    bad = compare_matrix(spec, measured(coherence=0.5), scale_db=0)
    assert bad.pair(0, 1).coherence_state == 'out'


def test_a_pair_with_no_cross_term_specified_is_not_judged():
    spec = matrix([1.0, 4.0, 2.0], {(0, 1): term(0.64, 30.0, 1.0, 4.0)},
                  Specification)
    run = matrix([1.0, 4.0, 2.0], {(0, 1): term(0.64, 30.0, 1.0, 4.0),
                                   (0, 2): term(0.9, 0.0, 1.0, 2.0),
                                   (1, 2): term(0.9, 0.0, 4.0, 2.0)})
    result = compare_matrix(spec, run, scale_db=0)
    assert [p.state for p in result.pairs] == ['in', 'not specified',
                                               'not specified']
    assert result.cell(0, 2).words == (
        'coherence not specified: the specification writes no cross term '
        'for the pair')
    assert result.passed is True
    assert result.describe().endswith('; 2 pairs not specified')


def test_a_cross_term_the_measurement_does_not_hold_is_said_to_be_missing():
    run = matrix([1.0, 4.0], {})
    result = compare_matrix(SPEC, run, scale_db=0)
    pair = result.pair(0, 1)
    assert pair.state == 'not measured'
    assert result.passed is True, 'the levels pass and no cross term judged'


def test_a_term_held_the_other_way_round_is_read_as_its_conjugate():
    """The specification writes only 2X+ against 1X+ (phase -30), the
    measurement only 1X+ against 2X+ (phase +30): the same pair."""
    spec_full = SPEC
    keep = [0, 2, 3]
    spec = Specification(F, spec_full.ordinate[keep],
                         response_dof=[spec_full.response_dof[i] for i in keep],
                         reference_dof=[spec_full.reference_dof[i] for i in keep],
                         ordinate_dim=[spec_full.ordinate_dim[i] for i in keep],
                         **{k: v[keep] for k, v in spec_full.limits.items()})
    spec.interpolation = 'bin'
    run = measured()
    keep = [0, 1, 3]
    run = Psd(F, run.ordinate[keep],
              response_dof=[run.response_dof[i] for i in keep],
              reference_dof=[run.reference_dof[i] for i in keep],
              ordinate_dim=[run.ordinate_dim[i] for i in keep])
    result = compare_matrix(spec, run, scale_db=0)
    pair = result.pair(0, 1)
    np.testing.assert_allclose(pair.specified_phase, 30.0)
    np.testing.assert_allclose(pair.measured_phase, 30.0)
    assert pair.state == 'in'


def test_a_breakpoint_specification_keeps_its_coherence_between_points():
    """Written at 20 and 2000 Hz with sloped levels: the cross term read
    log-log in magnitude and linearly in phase keeps the coherence and
    the phase the sheet entered at every line between."""
    from visualdynamics.core.author import SpecificationDraft

    draft = SpecificationDraft.at_dofs(DOFS[:2], 'acceleration')
    draft = dataclasses.replace(
        draft, levels=[[1.0, 0.01], [4.0, 0.4]],
        pairs={(0, 1): (0.8, -40.0)})
    spec = draft.make()
    assert spec.interpolation == 'log_log'
    lines = np.linspace(25.0, 1975.0, 80)
    s11 = 10 ** np.interp(np.log10(lines), np.log10([20, 2000]), [0, -2])
    s22 = 4.0 * 10 ** np.interp(np.log10(lines), np.log10([20, 2000]), [0, -1])
    run = matrix([s11, s22], {(0, 1): term(0.8, -40.0, s11, s22)}, f=lines)
    result = compare_matrix(spec, run, scale_db=0)
    pair = result.pair(0, 1)
    np.testing.assert_allclose(pair.specified_coherence, 0.8, rtol=1e-9)
    np.testing.assert_allclose(pair.specified_phase, -40.0, atol=1e-9)
    assert result.passed is True
    assert all(level.state == 'in' for level in result.levels)


def test_a_specification_on_coarser_lines_is_read_per_bin():
    """A specification on 20 Hz lines read at 10 Hz lines takes each
    bin's value: its coherence steps from 0.64 to 0.25 at 220 Hz, and a
    measurement holding 0.64 throughout is out above it."""
    coarse = np.arange(20.0, 440.0, 20.0)
    coherence = np.where(coarse < 220.0, 0.64, 0.25)
    spec = matrix([1.0, 4.0], {(0, 1): term(coherence, 30.0, 1.0, 4.0)},
                  Specification, f=coarse)
    result = compare_matrix(spec, measured(), scale_db=0)
    pair = result.pair(0, 1)
    judged = np.isfinite(pair.specified_coherence)
    expected = np.where(F[judged] < 210.0, 0.64, 0.25)
    np.testing.assert_allclose(pair.specified_coherence[judged], expected)
    assert pair.coherence_state == 'out'


def test_the_levels_are_the_single_channel_comparison():
    run = measured(autos=(2.0, 4.0))
    result = compare_matrix(SPEC, run, scale_db=0)
    rows = dict(compare_all(SPEC, run, scale_db=0))
    assert result.levels[0].result == rows['1X+']
    assert result.levels[0].result['difference_db'] == pytest.approx(
        10 * np.log10(2.0))
    # three decibels is the tolerance, and 3.01 is over it
    assert result.levels[0].state == 'out'
    assert result.levels[1].state == 'in'
    assert result.passed is False
    # the coherence and phase do not move with a level
    assert result.pair(0, 1).state == 'in'


def test_the_per_line_summary_is_the_rms_of_the_level_errors():
    """1X+ held at twice its specification, 3.01 dB at every line, and
    2X+ on it: sqrt((3.0103^2 + 0^2) / 2) = 2.1286 dB."""
    result = compare_matrix(SPEC, measured(autos=(2.0, 4.0)), scale_db=0)
    np.testing.assert_allclose(result.levels[0].error_db, 10 * np.log10(2.0))
    np.testing.assert_allclose(result.levels[1].error_db, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.level_rms_db,
                               np.sqrt((10 * np.log10(2.0)) ** 2 / 2))


def test_the_cross_terms_out_are_counted_at_each_line():
    coherence = np.full(F.shape, 0.64)
    coherence[:3] = 0.3
    phase = np.full(F.shape, 30.0)
    phase[2:6] = 60.0
    result = compare_matrix(SPEC, measured(coherence, phase), scale_db=0)
    assert result.cross_out.tolist()[:7] == [1, 1, 1, 1, 1, 1, 0]


def test_the_scale_moves_the_levels_and_not_the_cross_terms():
    run = measured(autos=(0.25, 1.0), coherence=0.64, phase=30.0)
    flat = compare_matrix(SPEC, run, scale_db=0)
    assert all(level.state == 'out' for level in flat.levels)
    scaled = compare_matrix(SPEC, run, scale_db=6)
    assert scaled.scale_db == 6
    assert all(level.state == 'in' for level in scaled.levels)
    np.testing.assert_array_equal(flat.pair(0, 1).measured_coherence,
                                  scaled.pair(0, 1).measured_coherence)


def test_a_detected_scale_is_warned():
    from visualdynamics.core.compliance import ScaleWarning

    run = measured(autos=(0.25, 1.0))
    with pytest.warns(ScaleWarning, match='scaled by \\+6 dB'):
        result = compare_matrix(SPEC, run)
    assert result.scale_db == 6


# ---- what cannot be judged ---------------------------------------------------


def test_a_set_of_autospectra_is_refused():
    autos = Psd(F, np.ones((2, F.size)), response_dof=DOFS[:2],
                ordinate_dim=['acceleration**2/frequency'] * 2)
    with pytest.raises(ValueError, match='needs a cross-spectral matrix'):
        compare_matrix(SPEC, autos)


def test_fewer_than_two_shared_channels_are_refused():
    run = matrix([1.0, 4.0], {(0, 1): term(0.5, 0.0, 1.0, 4.0)})
    run.response_dof = ['1X+', '9X+', '1X+', '9X+']
    run.reference_dof = ['1X+', '1X+', '9X+', '9X+']
    with pytest.raises(ValueError, match='at least two control channels.*'
                                         'share 1'):
        compare_matrix(SPEC, run)


def test_bands_against_lines_are_refused_with_the_comparisons_reason():
    banded = measured().to_octave(3)
    with pytest.raises(ValueError, match='octave bands'):
        compare_matrix(SPEC, banded)


def test_a_line_that_is_not_a_number_is_noted_and_not_judged():
    run = measured()
    run.ordinate[1, 5] = np.nan               # the cross term 1X+/2X+
    result = compare_matrix(SPEC, run, scale_db=0)
    pair = result.pair(0, 1)
    assert pair.coherence_lines == F.size - 1
    assert pair.state == 'in'
    assert pair.notes == (
        ('1 line of the measured 1X+/2X+ is not a number and not judged'),)


def test_a_level_with_a_line_that_is_not_a_number_is_refused():
    """Counted as no power, the line would read the channel low without a
    word."""
    run = measured()
    run.ordinate[0, 7] = np.nan
    result = compare_matrix(SPEC, run, scale_db=0)
    level = result.levels[0]
    assert level.state == 'refused'
    assert level.words() == ('not compared: 1 line of the measured 1X+ is '
                             'not a number; a level is not read across them')
    assert result.levels[1].state == 'in'


def test_a_dead_channel_fails_its_level_and_leaves_its_pairs_unjudged():
    run = matrix([1.0, np.zeros(F.shape)], {(0, 1): np.zeros(F.shape)})
    result = compare_matrix(SPEC, run, scale_db=0)
    assert result.levels[1].state == 'out'
    assert 'no power measured' in result.levels[1].words()
    pair = result.pair(0, 1)
    assert pair.state == 'not judged'
    assert pair.reason == ('the measured 2X+ recorded no power, so the pair '
                           'has no coherence')
    assert result.passed is False


def test_a_measurement_of_undeclared_units_is_refused_cell_by_cell():
    """Against a specification in declared units, a measurement whose
    units are not declared holds raw numbers in some other scale: each
    level and each cross term says so rather than read a verdict."""
    dims = {(i, j): 'unknown' for i in range(2) for j in range(2)}
    run = matrix([1.0, 4.0], {(0, 1): term(0.64, 30.0, 1.0, 4.0)}, dims=dims)
    result = compare_matrix(SPEC, run, scale_db=0)
    assert [level.state for level in result.levels] == ['refused'] * 2
    pair = result.pair(0, 1)
    assert pair.state == 'refused'
    assert pair.reason.startswith('the units of the measured one are not '
                                  'declared')
    assert result.passed is None


def test_a_cross_term_inconsistent_with_its_autospectra_is_not_read():
    """A cross term of an acceleration against a velocity is not the
    term between two acceleration channels, whatever its DOFs."""
    dims = {(i, j): 'acceleration**2/frequency' for i in range(2)
            for j in range(2)}
    dims[(0, 1)] = dims[(1, 0)] = 'acceleration*velocity/frequency'
    run = matrix([1.0, 4.0], {(0, 1): term(0.64, 30.0, 1.0, 4.0)}, dims=dims)
    pair = compare_matrix(SPEC, run, scale_db=0).pair(0, 1)
    assert pair.state == 'not measured'


# ---- the cross terms are no longer read as levels (M5) ----------------------


def test_compare_refuses_a_cross_term():
    """Read by its real part, a cross term held at its magnitude and 60
    degrees off its phase came out 3 dB low."""
    assert not autospectrum(SPEC, 1)
    run = measured(phase=90.0)
    assert compare(SPEC, run, 1, 1, scale_db=0) == {
        'lines': 0, 'refused': CROSS_TERM_REFUSAL}


def test_compare_all_compares_the_autospectra_only():
    rows = compare_all(SPEC, measured(phase=170.0), scale_db=0)
    assert [label for label, _result in rows] == ['1X+', '2X+']


def test_a_cross_term_does_not_move_the_detected_scale():
    """Every autospectrum at -6 dB, and a cross term whose real part
    reads 20 dB under its own: the scale is the autospectra's."""
    run = measured(autos=(0.25, 1.0), phase=84.0)
    assert detect_scale_db(SPEC, run) == 6


def test_an_autospectrum_held_in_a_complex_cpsd_is_compared_without_warning():
    run = measured()
    assert np.iscomplexobj(run.ordinate)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = compare(SPEC, run, 0, 0, scale_db=0)
    assert result['difference_db'] == pytest.approx(0.0, abs=1e-12)


# ---- against noise -------------------------------------------------------------


def _random_run(coherence, phase, seed, n=2 ** 18, fs=4096.0):
    """Three channels of band-limited random noise, 20 to 2000 Hz at a
    density of 1 (m/s^2)^2/Hz: the second carries the first at the
    coherence and phase asked for, the third is independent. Its CPSD
    from the default Hann window, 4096 lines a frame, half overlapped."""
    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.data import TimeHistory

    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    band = (f >= 20.0) & (f <= 2000.0)

    def spectrum():
        z = np.fft.rfft(rng.standard_normal(n))
        z[~band] = 0.0
        return z * np.sqrt(1980.0) / np.std(np.fft.irfft(z, n))

    z1, z2, z3 = spectrum(), spectrum(), spectrum()
    x2 = (np.sqrt(coherence) * np.exp(1j * np.radians(phase)) * z1
          + np.sqrt(1.0 - coherence) * z2)
    rows = [np.fft.irfft(z, n) for z in (z1, x2, z3)]
    history = TimeHistory(np.arange(n) / fs, np.vstack(rows),
                          response_dof=DOFS, ordinate_dim=['acceleration'] * 3,
                          ordinate_unit=['m/s**2'] * 3)
    history.averaging = Averaging(frame_length=4096, overlap=0.5,
                                  window='hann').filled(n, fs)
    return history.compute_cpsds()


def _sheet(coherence, phase):
    from visualdynamics.core.author import SpecificationDraft

    draft = SpecificationDraft.at_dofs(DOFS, 'acceleration')
    return dataclasses.replace(
        draft, pairs={(0, 1): (coherence, phase), (0, 2): (0.0, 0.0),
                      (1, 2): None}).make()


@pytest.mark.parametrize('coherence', [0.8, COHERENCE_FLOOR])
def test_a_run_that_is_its_specification_passes_against_noise(coherence):
    """About 127 frames, half overlapped. At a coherence of 0.8 one
    standard deviation of the coherence estimate is about 0.02 and of
    the phase about 2 degrees, far inside 0.1 and 10. At the floor, 0.5,
    the coherence estimate scatters by about 0.05, so a run that is its
    specification puts a few percent of its lines past 0.1 by chance
    (2.3 percent on this seed): the reason a cross term is judged by a
    share of its lines and not by its worst one. The independent pair's
    coherence estimate sits near 1 / 127, far inside 0.1."""
    result = compare_matrix(_sheet(coherence, 30.0),
                            _random_run(coherence, 30.0, seed=11),
                            scale_db=0)
    assert result.passed is True, result.describe()
    first = result.pair(0, 1)
    assert first.coherence_percent < 5.0 and first.phase_percent < 5.0
    if coherence > COHERENCE_FLOOR:
        assert first.coherence_percent == first.phase_percent == 0.0
    assert result.pair(0, 2).coherence_percent == 0.0
    assert result.pair(1, 2).state == 'not specified'
    assert COHERENCE_TOLERANCE == 0.1


def test_a_run_off_its_specification_fails_against_noise():
    result = compare_matrix(_sheet(0.8, 30.0), _random_run(0.5, 50.0, seed=12),
                            scale_db=0)
    pair = result.pair(0, 1)
    assert (pair.coherence_state, pair.phase_state) == ('out', 'out')
    assert pair.coherence_percent > 95.0 and pair.phase_percent > 95.0
    assert all(level.state == 'in' for level in result.levels)


# ---- the edges -----------------------------------------------------------------


def test_the_widest_settings_are_accepted():
    lines = cross_term_lines([0.0], [0.0], [1.0], [180.0],
                             coherence_tolerance=1.0, phase_tolerance=180.0,
                             coherence_floor=0.0)
    assert not lines['coherence_out'].any() and not lines['phase_out'].any()
    assert lines['phase_judged'].all(), 'a floor of zero judges every phase'


def test_a_cross_term_between_breakpoints_is_read_as_the_sheet_reads_it():
    """From 1 at 100 Hz to 100 at 90 degrees at 1 kHz: log-log, 10 at
    316.2 Hz, the phase a quarter of the way, 21.6 degrees. From there to
    zero at 2 kHz, straight: 50 at 1.5 kHz, the phase halfway back, 45.
    Nothing outside the points."""
    from visualdynamics.core.matrix_comparison import _between

    f = np.array([1000.0, 100.0, 2000.0])
    values = np.array([100j, 1.0, 0.0])
    out = _between(f, values, np.array([10 ** 2.5, 1500.0, 50.0, 2500.0,
                                        100.0, 2000.0]))
    np.testing.assert_allclose(np.abs(out[:2]), [10.0, 50.0], rtol=1e-12)
    np.testing.assert_allclose(np.degrees(np.angle(out[:2])),
                               [(10 ** 2.5 - 100.0) / 900.0 * 90.0, 45.0])
    assert np.isnan(out[2:4]).all()
    np.testing.assert_allclose(out[4:], [1.0, 0.0], atol=1e-12)
    # and straight up out of zero: 20 halfway from 0 at 2 kHz to 40
    rising = _between(np.array([2000.0, 3000.0]), np.array([0.0, 40.0]),
                      np.array([2500.0]))
    np.testing.assert_allclose(np.abs(rising), [20.0])
    assert np.isnan(_between(f[:1], values[:1], np.array([100.0]))).all()


def test_a_line_on_a_bins_edge_belongs_to_the_bin_above():
    """On 20 Hz lines a bin runs from 10 Hz below its line up to, not
    including, 10 Hz above: 210 Hz is the 220 Hz line's, and a line
    below the first bin has no value."""
    from visualdynamics.core.matrix_comparison import _read_at

    coarse = np.arange(20.0, 440.0, 20.0)
    spec = matrix([coarse, 4.0], {(0, 1): term(0.64, 30.0, coarse, 4.0)},
                  Specification, f=coarse)
    read = _read_at(spec, 0, np.array([5.0, 10.0, 29.99, 30.0, 210.0,
                                       429.0, 430.0]))
    np.testing.assert_array_equal(read[1:6], [20.0, 20.0, 40.0, 220.0, 420.0])
    assert np.isnan(read[[0, 6]]).all()


def test_every_measured_line_inside_a_specification_on_lines_is_judged():
    coarse = np.arange(20.0, 440.0, 20.0)
    spec = matrix([1.0, 4.0], {(0, 1): term(0.64, 30.0, 1.0, 4.0)},
                  Specification, f=coarse)
    pair = compare_matrix(spec, measured(), scale_db=0).pair(0, 1)
    assert pair.coherence_lines == pair.phase_lines == F.size


def test_a_line_that_is_not_a_number_outside_the_band_does_not_refuse_a_level():
    narrow = matrix([1.0, 4.0], {(0, 1): term(0.64, 30.0, 1.0, 4.0)},
                    Specification, f=np.arange(100.0, 310.0, 10.0))
    run = measured()
    # 20 Hz, well under the band (95 to 305 Hz), and 90 and 310 Hz,
    # whose bins end where it begins and begin where it ends
    run.ordinate[0, [0, 7, 29]] = np.nan
    result = compare_matrix(narrow, run, scale_db=0)
    assert result.levels[0].state == 'in'
    run.ordinate[0, 8] = np.nan                  # 100 Hz, inside
    assert compare_matrix(narrow, run, scale_db=0).levels[0].state == 'refused'


def test_a_channel_with_no_measured_autospectrum_leaves_its_pairs_unread():
    crosses = {(0, 1): term(0.64, 30.0, 1.0, 4.0),
               (0, 2): term(0.5, 0.0, 1.0, 2.0),
               (1, 2): term(0.5, 0.0, 4.0, 2.0)}
    spec = matrix([1.0, 4.0, 2.0], crosses, Specification)
    run = matrix([1.0, 4.0, 2.0], crosses)
    keep = [i for i in range(run.num_records)
            if (run.response_dof[i], run.reference_dof[i]) != ('3X+', '3X+')]
    run = Psd(F, run.ordinate[keep],
              response_dof=[run.response_dof[i] for i in keep],
              reference_dof=[run.reference_dof[i] for i in keep],
              ordinate_dim=[run.ordinate_dim[i] for i in keep])
    result = compare_matrix(spec, run, scale_db=0)
    assert result.levels[2].state == 'not measured'
    assert result.pair(0, 2).reason == ('the measurement holds no '
                                        'autospectrum for 3X+')
    assert result.pair(1, 2).state == 'not measured'
    assert result.pair(0, 1).state == 'in'
    assert result.passed is True


def test_the_summary_skips_a_line_no_channel_was_judged_on():
    result = compare_matrix(SPEC, measured(autos=(2.0, 4.0)), scale_db=0)
    errors = [level.error_db for level in result.levels]
    errors[0][3] = np.nan
    errors[1][3] = np.nan
    errors[1][4] = np.nan
    rms = result.level_rms_db
    assert np.isnan(rms[3])
    assert rms[4] == pytest.approx(10 * np.log10(2.0))
    assert rms[5] == pytest.approx(10 * np.log10(2.0) / np.sqrt(2.0))


def test_a_level_exactly_on_its_tolerance_is_in():
    """Ten percent of the band outside abort is allowed, and is the
    share a cross term is held to."""
    autos = np.full(F.shape, 1.0)
    autos[:4] = 10.0                             # 4 of 40 lines over abort
    scale = 10 ** (2.9 / 10.0) / (np.sum(autos) / 40.0)
    result = compare_matrix(SPEC, measured(autos=(autos * scale, 4.0)),
                            scale_db=0)
    level = result.levels[0]
    assert level.result['abort_percent'] == pytest.approx(10.0)
    assert level.result['difference_db'] == pytest.approx(2.9)
    assert level.state == 'in'


def test_a_comparison_judged_again_at_other_tolerances_is_a_fresh_one():
    """The readings are kept and the judgement moves: the same states and
    lines as comparing afresh at the new tolerances."""
    run = measured(coherence=np.linspace(0.5, 0.64, F.size),
                   phase=np.linspace(20.0, 45.0, F.size))
    first = compare_matrix(SPEC, run, scale_db=0)
    again = first.with_tolerances(coherence_tolerance=0.05,
                                  phase_tolerance=12.0, coherence_floor=0.6)
    fresh = compare_matrix(SPEC, run, scale_db=0, coherence_tolerance=0.05,
                           phase_tolerance=12.0, coherence_floor=0.6)
    assert again.levels is first.levels
    for mine, theirs in zip(again.pairs, fresh.pairs, strict=True):
        assert (mine.coherence_state, mine.phase_state) == (
            theirs.coherence_state, theirs.phase_state)
        for key, value in theirs.lines.items():
            np.testing.assert_array_equal(mine.lines[key], value)
    assert (again.coherence_tolerance, again.phase_tolerance,
            again.coherence_floor) == (0.05, 12.0, 0.6)
    assert again.with_tolerances().coherence_tolerance == 0.05
    with pytest.raises(ValueError, match='phase tolerance'):
        first.with_tolerances(phase_tolerance=0.0)


def test_a_pair_with_no_readings_is_kept_when_judged_again():
    spec = matrix([1.0, 4.0, 2.0], {(0, 1): term(0.64, 30.0, 1.0, 4.0)},
                  Specification)
    run = matrix([1.0, 4.0, 2.0], {(0, 1): term(0.64, 30.0, 1.0, 4.0)})
    first = compare_matrix(spec, run, scale_db=0)
    again = first.with_tolerances(coherence_tolerance=0.2)
    assert again.pairs[1] is first.pairs[1]
    assert again.pairs[1].state == 'not specified'


def test_a_level_error_is_written_without_a_negative_zero():
    from visualdynamics.core.matrix_comparison import level_text

    assert level_text(-0.04) == '+0.0 dB'
    assert level_text(-1.26) == '-1.3 dB'
    assert level_text(3.01) == '+3.0 dB'
    assert level_text(float('nan')) == '—'


def test_a_level_out_says_both_its_numbers():
    result = compare_matrix(SPEC, measured(autos=(2.0, 4.0)), scale_db=0)
    assert result.levels[0].words() == (
        'out: 0.0 % of lines outside abort, +3.0 dB rms')


def test_a_pair_not_judged_says_why_in_both_its_cells():
    run = matrix([1.0, np.zeros(F.shape)], {(0, 1): np.zeros(F.shape)})
    result = compare_matrix(SPEC, run, scale_db=0)
    reason = 'the measured 2X+ recorded no power, so the pair has no coherence'
    assert result.cell(0, 1).words == f'coherence not judged: {reason}'
    assert result.cell(1, 0).words == f'phase not judged: {reason}'
    pair = result.pair(0, 1)
    assert pair.coherence_lines == 0
    assert np.isnan(pair.coherence_percent)


def test_a_pair_not_specified_counts_no_lines():
    spec = matrix([1.0, 4.0, 2.0], {(0, 1): term(0.64, 30.0, 1.0, 4.0)},
                  Specification)
    run = matrix([1.0, 4.0, 2.0], {(0, 1): term(0.64, 30.0, 1.0, 4.0)})
    pair = compare_matrix(spec, run, scale_db=0).pair(0, 2)
    assert (pair.coherence_lines, pair.phase_lines) == (0, 0)
    assert np.isnan(pair.phase_percent)


def test_a_measured_term_held_only_the_other_way_is_its_conjugate():
    """The measurement holds 2X+ against 1X+ only (phase -30)."""
    run = measured()
    keep = [0, 2, 3]
    run = Psd(F, run.ordinate[keep],
              response_dof=[run.response_dof[i] for i in keep],
              reference_dof=[run.reference_dof[i] for i in keep],
              ordinate_dim=[run.ordinate_dim[i] for i in keep])
    pair = compare_matrix(SPEC, run, scale_db=0).pair(0, 1)
    np.testing.assert_allclose(pair.measured_phase, 30.0)
    assert pair.state == 'in'


def test_a_dead_line_leaves_the_summary_to_the_other_channels():
    """A line with no power has no level error in dB; the RMS there is
    the other channel's alone."""
    autos = np.full(F.shape, 2.0)
    autos[6] = 0.0
    result = compare_matrix(SPEC, measured(autos=(autos, 8.0)), scale_db=0)
    assert np.isnan(result.levels[0].error_db[6])
    assert result.level_rms_db[6] == pytest.approx(10 * np.log10(2.0))


def test_undeclared_units_on_both_sides_are_compared_and_noted_once():
    from visualdynamics.core.compliance import RAW_NOTE

    dims = {(i, j): 'unknown' for i in range(2) for j in range(2)}
    spec = matrix([1.0, 4.0], {(0, 1): term(0.64, 30.0, 1.0, 4.0)},
                  Specification, dims=dims)
    run = matrix([1.0, 4.0], {(0, 1): term(0.64, 30.0, 1.0, 4.0)}, dims=dims)
    result = compare_matrix(spec, run, scale_db=0)
    assert result.passed is True
    assert result.notes == (RAW_NOTE,)


def test_the_shared_channels_are_counted_in_the_refusal():
    spec = matrix([1.0, 4.0, 2.0], {}, Specification)
    run = matrix([1.0, 4.0], {(0, 1): term(0.5, 0.0, 1.0, 4.0)})
    run.response_dof = ['1X+', '9X+', '1X+', '9X+']
    run.reference_dof = ['1X+', '1X+', '9X+', '9X+']
    with pytest.raises(ValueError, match='these share 1$'):
        compare_matrix(spec, run)


def test_a_detected_scale_is_warned_at_the_callers_line():
    from visualdynamics.core.compliance import ScaleWarning

    with pytest.warns(ScaleWarning) as caught:
        compare_matrix(SPEC, measured(autos=(0.25, 1.0)))
    assert caught[0].filename == __file__
