"""The limit and notch prediction: where each limit channel would exceed
its limit profile at full level, and the notch the controller would cut
in the control spectrum to hold it there.

Pinned against arithmetic worked by hand, independent of the code. On
1 Hz lines with a flat reference of 0.04 g^2/Hz and an abort tolerance
of minus 3 dB (0.02 g^2/Hz):

- a limit channel whose transmissibility is 3 over 240 to 260 Hz and 1
  elsewhere responds at 9 x 0.04 = 0.36 g^2/Hz there; against a flat
  profile of 0.18 g^2/Hz it is twice over, 3.01 dB, and the control has
  to come down to 0.18 / 9 = 0.02 g^2/Hz: a notch 3.01 dB deep, a
  ratio of a half, sitting exactly on the abort edge, which is inside;
- with a transmissibility of 4 the response is 0.64 g^2/Hz, 5.51 dB
  over, and the control 0.18 / 16 = 0.01125 g^2/Hz: a notch 5.51 dB
  deep, under the abort edge by 10 log10(0.02 / 0.01125) = 2.50 dB;
- a second channel of transmissibility 3 over 250 to 270 Hz with a
  profile of 0.09 g^2/Hz allows 0.01 g^2/Hz there, so the two cut one
  notch from 240 to 270 Hz, 6.02 dB deep from 250 to 260 Hz, where the
  second sets it.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.averaging import Averaging
from visualdynamics.core.compliance import exceedances
from visualdynamics.core.data import Frf, Psd, Specification, TimeHistory
from visualdynamics.core.limit_prediction import (
    MOST_OF_BAND,
    NARROW_LINES,
    _runs,
    predict_limits,
    response_through,
)

G = 9.80665
LEVEL = 0.04 * G ** 2
F = np.arange(0.0, 2001.0, 1.0)


def bump(gain, low, high, f=F):
    """A transmissibility of `gain` from `low` to `high` Hz, 1 elsewhere."""
    t = np.ones(len(f), dtype=complex)
    t[(f >= low) & (f <= high)] = gain
    return t


def frf(*rows, dofs=None, reference='1Z+', dim='acceleration/acceleration',
        f=F):
    dofs = dofs or [f'{k + 2}Z+' for k in range(len(rows))]
    return Frf(f, np.array(rows), response_dof=dofs,
               reference_dof=[reference] * len(rows), ordinate_dim=dim)


def spec(*profiles, dofs=None, lower=0.5, band=(20.0, 2000.0),
         dim='acceleration**2/frequency', control='1Z+'):
    """The reference at 1Z+ with an abort pair of `lower` and 2 times it,
    and flat limit profiles (in g^2/Hz) at 2Z+, 3Z+, ... beside it."""
    dofs = [control] + (dofs or [f'{k + 2}Z+' for k in range(len(profiles))])
    levels = [LEVEL] + [p * G ** 2 for p in profiles]
    values = np.array([[v, v] for v in levels])
    nan = np.full(2, np.nan)
    limits = {}
    if lower is not None:
        limits = {
            'abort_lower': np.array([values[0] * lower]
                                    + [nan] * len(profiles)),
            'abort_upper': np.array([values[0] * 2.0]
                                    + [nan] * len(profiles))}
    return Specification(np.array(band), values, response_dof=dofs,
                         ordinate_dim=dim, **limits)


def at(prediction, frequency):
    return int(np.argmin(np.abs(prediction.frequency - frequency)))


# ---- the hand values ----------------------------------------------------------

def test_a_channel_twice_over_cuts_a_notch_three_db_deep_on_the_edge():
    p = predict_limits(frf(bump(3.0, 240, 260)), spec(0.18))
    (notch,) = p.notches
    assert (notch.low, notch.high, notch.lines) == (240.0, 260.0, 21)
    assert notch.depth_db == pytest.approx(10 * np.log10(2.0), abs=1e-9)
    assert notch.ratio == pytest.approx(0.5)
    assert notch.level == pytest.approx(0.02 * G ** 2)
    assert notch.reference == pytest.approx(LEVEL)
    assert notch.channels == ('2Z+',)
    # exactly on the abort edge is inside it
    assert notch.outside is False and notch.outside_lines == 0
    assert p.within_tolerance is True
    (over,) = p.channels[0].exceedances
    assert (over.low, over.high, over.lines) == (240.0, 260.0, 21)
    assert over.over_db == pytest.approx(10 * np.log10(2.0))
    np.testing.assert_allclose(p.channels[0].response[at(p, 250)],
                               0.36 * G ** 2)
    assert p.verdict() == (
        '1 notch in the control at full level, the deepest 3.0 dB at '
        '240 Hz for 2Z+; the control stays inside the abort tolerance')
    # the property reads the same edge the notch was read on
    assert p.outside.sum() == 0
    assert p.notched.sum() == 21
    assert not any('in effect the control' in note for note in p.notes)
    assert p.describe()[1] == (
        'notch from 240 to 260 Hz (21 lines): 3.0 dB deep at 240 Hz, a '
        'ratio of 0.5, the control left at 0.02 g²/Hz, for 2Z+; inside the '
        'abort tolerance')
    # nothing else moves
    assert not p.notched[at(p, 239)] and not p.notched[at(p, 261)]
    assert p.control[at(p, 1000)] == pytest.approx(LEVEL)


def test_a_deeper_notch_takes_the_control_outside_the_tolerance():
    p = predict_limits(frf(bump(4.0, 240, 260)), spec(0.18))
    (notch,) = p.notches
    assert notch.depth_db == pytest.approx(10 * np.log10(0.04 / 0.01125))
    assert notch.level == pytest.approx(0.01125 * G ** 2)
    assert notch.outside is True and notch.outside_lines == 21
    assert notch.beyond_db == pytest.approx(10 * np.log10(0.02 / 0.01125))
    assert p.within_tolerance is False
    assert p.verdict().endswith(
        '; it takes the control outside the abort tolerance')
    assert p.outside.sum() == 21
    rows = p.table()
    assert rows[0]['band'] == '240–260'
    assert rows[0]['depth'] == '5.5 (0.28)'
    assert rows[0]['tolerance'] == 'outside'
    assert rows[0]['level'] == '0.01125 g²/Hz'


def test_the_warning_pair_is_read_when_asked_for():
    s = spec(0.18)
    s.limits['warning_lower'] = s.limits['abort_lower'] * 1.2
    p = predict_limits(frf(bump(3.0, 240, 260)), s, tolerance='warning')
    assert p.notches[0].outside is True
    assert p.tolerance == 'warning'
    with pytest.raises(ValueError, match="'abort' or 'warning', not 'loose'"):
        predict_limits(frf(bump(3.0, 240, 260)), s, tolerance='loose')


def test_two_channels_cut_one_notch_and_the_deeper_one_names_it():
    p = predict_limits(frf(bump(3.0, 240, 260), bump(3.0, 250, 270)),
                       spec(0.18, 0.09))
    (notch,) = p.notches
    assert (notch.low, notch.high) == (240.0, 270.0)
    assert notch.frequency == 250.0
    assert notch.depth_db == pytest.approx(10 * np.log10(4.0))
    assert notch.channels == ('3Z+', '2Z+')
    assert p.control[at(p, 245)] == pytest.approx(0.02 * G ** 2)
    assert p.control[at(p, 255)] == pytest.approx(0.01 * G ** 2)
    assert p.control[at(p, 265)] == pytest.approx(0.01 * G ** 2)
    np.testing.assert_allclose(p.depth_db[at(p, 245)], 10 * np.log10(2.0))


def test_separate_bands_are_separate_notches_in_frequency_order():
    t = bump(3.0, 600, 610) * bump(4.0, 100, 120)
    p = predict_limits(frf(t), spec(0.18))
    assert [(n.low, n.high) for n in p.notches] == [(100.0, 120.0),
                                                    (600.0, 610.0)]
    assert [n.outside for n in p.notches] == [True, False]
    assert p.verdict() == (
        '2 notches in the control at full level, the deepest 5.5 dB at '
        '100 Hz for 2Z+; 1 of them takes the control outside the abort '
        'tolerance')
    both = predict_limits(frf(bump(4.0, 600, 610) * bump(4.0, 100, 120)),
                          spec(0.18))
    assert both.verdict().endswith('; 2 of them take the control outside '
                                   'the abort tolerance')


def test_a_one_line_notch_and_one_line_outside_are_said_in_the_singular():
    p = predict_limits(frf(bump(4.0, 300, 300)), spec(0.18))
    (notch,) = p.notches
    assert notch.describe(p.dimension, 'abort').startswith(
        'notch at 300 Hz (1 line): 5.5 dB deep at 300 Hz')
    assert notch.tolerance_words('abort') == (
        'outside the abort tolerance at 1 line, by up to 2.5 dB')
    assert p.table()[0]['band'] == '300'
    assert p.channels[0].exceedances[0].describe() == (
        '5.5 dB over its limit profile at 300 Hz, at 300 Hz')


def test_an_exceedance_names_its_peak_line_wherever_it_falls_in_the_run():
    t = np.ones(len(F), dtype=complex)
    band = (F >= 240) & (F <= 260)
    t[band] = np.linspace(2.5, 4.0, band.sum())        # rising to the end
    p = predict_limits(frf(t), spec(0.18))
    (over,) = p.channels[0].exceedances
    assert over.frequency == 260.0 and over.low == 240.0
    assert p.notches[0].frequency == 260.0


def test_a_channel_exactly_on_its_profile_is_not_over_it():
    p = predict_limits(frf(bump(3.0, 240, 260)), spec(0.36))
    assert p.channels[0].exceedances == ()
    assert p.channels[0].worst is None
    assert p.notches == ()


def test_two_lines_are_enough():
    f = np.array([100.0, 101.0])
    p = predict_limits(frf(np.array([4.0, 1.0 + 0j]), f=f), spec(0.18))
    assert p.notches[0].low == 100.0


def test_no_notch_says_how_close_the_nearest_channel_comes():
    p = predict_limits(frf(bump(1.5, 240, 260)), spec(0.18))
    assert p.notches == ()
    assert p.within_tolerance is True
    assert p.channels[0].worst is None
    # no notch, so no rms line: the control is the reference
    assert not any('rms' in line for line in p.describe())
    # 2.25 x 0.04 = 0.09 against 0.18: 3.01 dB under, first at 240 Hz
    assert p.verdict() == (
        'no notch: the limit channel stays under its limit profile at '
        'full level, 2Z+ closest, 3.0 dB under at 240 Hz')
    assert p.channels[0].describe(p.frequency) == (
        '2Z+: under its limit profile everywhere, 3.0 dB under at its '
        'closest, 240 Hz')


def test_the_closest_of_two_channels_is_the_one_with_least_margin():
    """2Z+ comes within 3.0 dB at 240 Hz, 3Z+ within 6.0 dB at 900 Hz:
    the verdict names 2Z+, the nearer to its limit, not the higher in
    frequency."""
    p = predict_limits(frf(bump(1.5, 240, 260), bump(1.5, 900, 920)),
                       spec(0.18, 0.36))
    assert p.verdict() == (
        'no notch: 2 limit channels stay under their limit profiles at '
        'full level, 2Z+ closest, 3.0 dB under at 240 Hz')


def test_a_channel_with_no_profile_is_predicted_and_does_not_notch():
    p = predict_limits(frf(bump(3.0, 240, 260), bump(9.0, 500, 520)),
                       spec(0.18))
    monitor = p.channels[1]
    assert not monitor.limited and monitor.exceedances == ()
    np.testing.assert_allclose(monitor.response[at(p, 510)], 81 * LEVEL)
    assert [n.channels for n in p.notches] == [('2Z+',)]
    assert monitor.describe(p.frequency).startswith(
        '3Z+: no limit profile, predicted only, highest 3.24 g²/Hz at '
        '500 Hz')


def test_the_profiles_may_be_held_apart_from_the_specification():
    profiles = Psd(np.array([20.0, 2000.0]),
                   np.array([[0.18, 0.18]]) * G ** 2, response_dof=['2Z+'],
                   ordinate_dim='acceleration**2/frequency')
    profiles.interpolation = 'log_log'
    p = predict_limits(frf(bump(3.0, 240, 260)), spec(), profiles)
    assert p.notches[0].depth_db == pytest.approx(10 * np.log10(2.0))


def test_a_sloped_profile_is_read_log_log_over_each_line():
    """A profile rising 0.18 to 0.72 g^2/Hz from 200 to 400 Hz is a power
    law of slope 2: at 250 Hz, 0.18 (250/200)^2 = 0.28125, which a
    transmissibility of 3 over the 0.04 reference (0.36) exceeds by
    1.07 dB, and at 249 Hz, the deepest line, by 1.11 dB; the line's bin
    average differs from the point by 1e-6."""
    s = spec()
    profile = Specification(np.array([200.0, 400.0]),
                            np.array([[0.18, 0.72]]) * G ** 2,
                            response_dof=['2Z+'],
                            ordinate_dim='acceleration**2/frequency')
    p = predict_limits(frf(bump(3.0, 249, 251)), s, profile)
    assert p.notches[0].depth_db == pytest.approx(
        10 * np.log10(0.36 / 0.18 / (249 / 200) ** 2), abs=1e-4)
    # outside the profile nothing limits it, and that is said
    assert any('not written at' in note for note in p.notes)


def test_the_reference_s_own_breakpoints_are_read_and_nothing_outside():
    p = predict_limits(frf(bump(3.0, 240, 260)),
                       spec(0.18, band=(20.0, 1500.0)))
    assert np.isnan(p.reference[at(p, 10)]) and np.isnan(p.control[at(p, 10)])
    assert np.isnan(p.channels[0].response[at(p, 1600)])
    # the 1500 Hz line's bin is half inside, and read over that half
    assert p.reference[at(p, 1500)] == pytest.approx(LEVEL)
    assert np.isnan(p.reference[at(p, 1501)])


def test_a_notch_at_the_first_line_of_the_band():
    """The band starts at 20 Hz; the 20 Hz line's bin is cut to the half
    inside it, and still predicts the full depth."""
    p = predict_limits(frf(bump(3.0, 0, 22)), spec(0.18))
    (notch,) = p.notches
    assert notch.low == 20.0 and notch.high == 22.0
    assert notch.depth_db == pytest.approx(10 * np.log10(2.0))
    assert notch.narrow is False


def test_a_notch_narrower_than_a_controller_cuts_is_said_to_be():
    p = predict_limits(frf(bump(3.0, 300, 301)), spec(0.18))
    assert p.notches[0].lines == 2 < NARROW_LINES
    assert p.notches[0].narrow
    assert any(note.startswith('a notch fewer than 3 lines wide (300 Hz)')
               for note in p.notes)


def test_a_notch_over_most_of_the_band_is_named_as_the_control():
    p = predict_limits(frf(np.full(len(F), 2.0 + 0j)), spec(0.08))
    assert len(p.notches) == 1
    assert p.notched.sum() > MOST_OF_BAND * np.isfinite(p.control).sum()
    assert any(note.startswith('the notches cover 100 % of the predicted '
                               'lines: the limit channels are in effect '
                               'the control') for note in p.notes)
    assert p.control_rms / p.reference_rms == pytest.approx(np.sqrt(0.5))
    assert p.describe()[-2] == ("the control rms -3.01 dB from the "
                                "reference's over the predicted lines")


def test_a_specification_with_no_lower_edge_judges_no_tolerance():
    p = predict_limits(frf(bump(4.0, 240, 260)), spec(0.18, lower=None))
    assert p.lower is None
    assert p.notches[0].outside is None
    assert p.notches[0].outside_lines == 0
    assert np.isnan(p.notches[0].beyond_db)
    assert p.notches[0].tolerance_words('abort') == (
        'the abort tolerance is not written over it')
    assert p.within_tolerance is None
    assert p.verdict().endswith('the specification carries no lower abort '
                                'limit, so the tolerance is not judged')
    assert p.table()[0]['tolerance'] == 'not judged'


def test_the_tolerance_verdict_is_the_comparison_s_own():
    """The predicted control, judged as any measured spectrum is against
    the specification's abort pair, falls out at exactly the lines the
    prediction says."""
    p = predict_limits(frf(bump(4.0, 240, 260), bump(2.5, 700, 720)),
                       spec(0.18, 0.15))
    control = Psd(p.frequency, np.where(np.isfinite(p.control), p.control,
                                        0.0)[None], response_dof=['1Z+'],
                  ordinate_dim='acceleration**2/frequency')
    _over, under = exceedances(spec(0.18, 0.15), control, 0, 0, 'abort',
                               scale_db=0)
    inside = np.isfinite(p.reference)
    np.testing.assert_array_equal(under[inside], p.outside[inside])


def test_declared_units_are_carried_and_the_level_said_in_them():
    from visualdynamics.units import SI

    p = predict_limits(frf(bump(4.0, 240, 260)), spec(0.18))
    words = p.notches[0].describe(p.dimension, 'abort', SI)
    # 0.01125 g^2/Hz is 1.082 (m/s^2)^2/Hz in SI, and 0.01125 in g
    assert 'the control left at 1.082 (m/s²)²/Hz,' in words
    assert 'the control left at 0.01125 g²/Hz,' in p.notches[0].describe(
        p.dimension, 'abort')


def test_raw_numbers_on_both_sides_are_compared_and_said_to_be():
    p = predict_limits(frf(bump(3.0, 240, 260), dim='unknown'),
                       spec(0.18, dim='unknown'))
    assert p.notches[0].depth_db == pytest.approx(10 * np.log10(2.0))
    assert any('raw numbers' in note for note in p.notes)


# ---- dead, not-a-number and empty inputs -------------------------------------

def test_a_dead_limit_channel_is_not_judged_and_never_a_pass():
    p = predict_limits(frf(np.zeros(len(F), dtype=complex)), spec(0.18))
    assert p.channels[0].judged is False
    assert p.notches == () and p.within_tolerance is None
    assert p.verdict().startswith('nothing judged: its transmissibility '
                                  'predicts nothing')
    assert any(note.startswith('the transmissibility to 2Z+ is zero at '
                               'every predicted line') for note in p.notes)


def test_a_dead_channel_beside_a_live_one_is_named_and_left_out():
    p = predict_limits(frf(bump(3.0, 240, 260), np.zeros(len(F))),
                       spec(0.18, 0.18))
    assert [c.judged for c in p.channels] == [True, False]
    assert len(p.notches) == 1
    assert p.channels[1].describe(p.frequency) == (
        '3Z+: not judged, its transmissibility predicts nothing')


def test_a_line_that_is_not_a_number_is_not_predicted_and_said():
    t = bump(3.0, 240, 260)
    t[250] = np.nan
    p = predict_limits(frf(t), spec(0.18))
    assert np.isnan(p.control[at(p, 250)])
    # the notch is broken at the line nothing can be said of
    assert [(n.low, n.high) for n in p.notches] == [(240.0, 249.0),
                                                    (251.0, 260.0)]
    assert any(note.startswith('the transmissibility to 2Z+ is not a '
                               'number at 1 of the 1981 predicted lines')
               for note in p.notes)


def test_a_record_holding_one_nan_sample_predicts_nothing_and_says_why():
    history = survey(nan_at=1000)
    t = history.compute_frfs(references=['1Z+'], method='H1')
    p = predict_limits(t, survey_spec())
    assert p.verdict().startswith('nothing judged')
    assert any('is not a number at' in note and 'holds a value that is '
               'not one' in note for note in p.notes)


def test_an_empty_transmissibility_is_refused():
    empty = Frf(F, np.zeros((0, len(F)), dtype=complex), response_dof=[],
                reference_dof=[], ordinate_dim=[])
    with pytest.raises(ValueError, match='holds no record to predict from'):
        predict_limits(empty, spec(0.18))


# ---- refusals ----------------------------------------------------------------

def test_transmissibilities_against_two_channels_are_refused():
    t = Frf(F, np.array([bump(3, 240, 260), bump(3, 240, 260)]),
            response_dof=['2Z+', '3Z+'], reference_dof=['1Z+', '4Z+'],
            ordinate_dim='acceleration/acceleration')
    with pytest.raises(ValueError, match='taken against 1Z\\+ and 4Z\\+'):
        predict_limits(t, spec(0.18, 0.18))


def test_a_specification_with_nothing_at_the_control_channel_is_refused():
    with pytest.raises(ValueError, match='no autospectrum at 5Z\\+, the '
                                         'channel the transmissibilities'):
        predict_limits(frf(bump(3, 240, 260), reference='5Z+'), spec(0.18))


def test_an_frf_against_a_force_is_not_a_transmissibility_from_control():
    with pytest.raises(ValueError, match='the specification is of '
                                         'acceleration and the control '
                                         'channel of the transmissibility '
                                         'of force'):
        predict_limits(frf(bump(3, 240, 260), dim='acceleration/force'),
                       spec(0.18))


def test_a_force_limit_channel_is_read_in_force():
    """An FRF from the control acceleration to an interface force, 50 N
    per m/s^2, against a force profile: 2500 x 0.04 g^2/Hz in N^2/Hz is
    predicted, and a profile of half that notches 3.01 dB."""
    t = frf(np.full(len(F), 50.0 + 0j), dim='force/acceleration')
    need = 2500.0 * LEVEL
    profiles = Specification(np.array([20.0, 2000.0]),
                             np.array([[need / 2, need / 2]]),
                             response_dof=['2Z+'],
                             ordinate_dim='force**2/frequency')
    p = predict_limits(t, spec(), profiles)
    np.testing.assert_allclose(p.channels[0].response[at(p, 100)], need)
    assert p.notches[0].depth_db == pytest.approx(10 * np.log10(2.0))
    assert p.channels[0].dimension == 'force**2/frequency'


def test_a_profile_of_another_quantity_is_refused():
    profiles = Specification(np.array([20.0, 2000.0]),
                             np.array([[1.0, 1.0]]), response_dof=['2Z+'],
                             ordinate_dim='force**2/frequency')
    with pytest.raises(ValueError, match='the limit profile at 2Z\\+ is of '
                                         'force and the transmissibility to '
                                         'it of acceleration'):
        predict_limits(frf(bump(3, 240, 260)), spec(), profiles)


def test_one_side_s_units_undeclared_is_refused():
    with pytest.raises(ValueError, match='are not declared'):
        predict_limits(frf(bump(3, 240, 260), dim='unknown'), spec(0.18))


def test_no_profile_at_any_channel_is_refused_with_how_to_give_one():
    with pytest.raises(ValueError, match='no response channel the '
                                         'transmissibilities reach \\(2Z\\+'
                                         '\\) has a limit profile in the '
                                         'specification'):
        predict_limits(frf(bump(3, 240, 260)), spec())


def test_two_records_to_one_channel_are_refused():
    with pytest.raises(ValueError, match='two records to 2Z\\+'):
        predict_limits(frf(bump(3, 240, 260), bump(2, 240, 260),
                           dofs=['2Z+', '2Z+']), spec(0.18))


def test_a_specification_on_octave_bands_is_refused():
    banded = spec(0.18).to_octave(3)
    with pytest.raises(ValueError, match='octave bands'):
        predict_limits(frf(bump(3, 240, 260)), banded)


def test_a_specification_over_none_of_the_lines_is_refused():
    with pytest.raises(ValueError, match='written over none'):
        predict_limits(frf(bump(3, 240, 260)), spec(0.18,
                                                    band=(5000.0, 8000.0)))


def test_one_line_is_too_few():
    with pytest.raises(ValueError, match='at least two lines'):
        predict_limits(frf(np.array([3.0 + 0j]), f=np.array([250.0])),
                       spec(0.18))


def test_the_control_over_itself_is_not_a_limit_channel():
    t = Frf(F, np.array([np.ones(len(F)), bump(3, 240, 260)]),
            response_dof=['1Z+', '2Z+'], reference_dof=['1Z+', '1Z+'],
            ordinate_dim='acceleration/acceleration')
    p = predict_limits(t, spec(0.18))
    assert [c.label for c in p.channels] == ['2Z+']


# ---- the pieces -------------------------------------------------------------

def test_the_response_is_the_squared_magnitude_times_the_density():
    out = response_through(np.array([3j, 1 + 1j, np.nan]),
                           np.array([0.04, 2.0, 1.0]))
    np.testing.assert_allclose(out[:2], [0.36, 4.0])
    assert np.isnan(out[2])


def test_runs_are_the_adjacent_true_stretches():
    assert _runs(np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], bool)) == [
        (0, 1), (3, 3), (6, 8)]
    assert _runs(np.zeros(4, bool)) == []


# ---- from a measured transmissibility, under the default window --------------

FS = 5120.0
FN, Q = 300.0, 20.0


def truth(f):
    r = np.asarray(f, dtype=float) / FN
    z = 1.0 / (2.0 * Q)
    return (1.0 + 2j * z * r) / (1.0 - r ** 2 + 2j * z * r)


def survey(seconds=24.0, noise=0.05, nan_at=None):
    """Random at 0.04 g^2/Hz from 20 to 2000 Hz on 1Z+ and its response
    through a 300 Hz, Q 20 resonance on 2Z+, with `noise` g rms the input
    did not cause, on a record that repeats, framed under Hann."""
    n = int(seconds * FS)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    spectrum = np.fft.rfft(np.random.default_rng(2604).standard_normal(n))
    spectrum[(f < 20.0) | (f > 2000.0)] = 0.0
    x = np.fft.irfft(spectrum, n)
    x *= np.sqrt(0.04 * 1980.0) / np.std(x)
    y = np.fft.irfft(np.fft.rfft(x) * truth(f), n)
    y += np.random.default_rng(2605).standard_normal(n) * noise
    if nan_at is not None:
        y[nan_at] = np.nan
    history = TimeHistory(np.arange(n) / FS + 3.0, np.vstack([x, y]) * G,
                          response_dof=['1Z+', '2Z+'],
                          ordinate_dim='acceleration')
    history.averaging = Averaging(frame_length=4096, overlap=0.5).filled(
        n, FS)
    return history


def survey_spec():
    """The reference at 1Z+ and a profile of 4 g^2/Hz at 2Z+, 100 times
    the input: the resonance, 401 times the input at its peak
    (1 + (1/Q)^2 over (1/Q)^2), is 16.04 g^2/Hz there, 6.03 dB over."""
    return spec(4.0)


def test_a_measured_h1_predicts_the_notch_the_resonance_asks_for():
    """Under the default window, with noise on the response and a record
    that does not start at zero, the depth read from the estimated H1 is
    the hand value to within the estimate's own scatter at the peak."""
    history = survey()
    assert history.averaging.window == 'hann'
    t = history.compute_frfs(references=['1Z+'], method='H1')
    p = predict_limits(t, survey_spec())
    deepest = p.deepest
    expected = 10 * np.log10(0.04 * np.abs(truth(deepest.frequency)) ** 2
                             / 4.0)
    assert deepest.depth_db == pytest.approx(expected, abs=0.3)
    assert abs(deepest.frequency - FN) < 2.0
    assert deepest.channels == ('2Z+',)
    # the true peak is 6.03 dB over, the line nearest it no more
    assert 5.5 < deepest.depth_db < 6.1


def test_noise_alone_on_a_response_never_reads_as_a_notch():
    """A response that is only noise the input did not cause has an H1
    near zero, and against a profile at the input's own level it trips
    no line: what is not caused by the drive cannot be notched out."""
    history = survey(noise=3.0)
    history.ordinate[1] = np.random.default_rng(2606).standard_normal(
        history.ordinate.shape[1]) * 3.0 * G
    t = history.compute_frfs(references=['1Z+'], method='H1')
    p = predict_limits(t, spec(0.04))
    assert p.notches == ()


def test_a_profile_line_of_zero_is_not_a_limit_of_zero():
    """A profile on lines that holds zero at a line has said nothing
    there; read as a limit it would ask for a notch of no control."""
    level = np.full(len(F), 0.18 * G ** 2)
    level[250] = 0.0
    profiles = Psd(F, level[None], response_dof=['2Z+'],
                   ordinate_dim='acceleration**2/frequency')
    p = predict_limits(frf(bump(3.0, 240, 260)), spec(), profiles)
    assert np.isnan(p.channels[0].profile[250])
    assert np.isfinite(p.control[250]) and not p.notched[250]


def test_the_matching_quantity_is_chosen_among_records_at_one_channel():
    """Two profiles at 2Z+, a force first: the acceleration one is
    read."""
    profiles = Specification(np.array([20.0, 2000.0]),
                             np.array([[1.0, 1.0], [0.18 * G ** 2] * 2]),
                             response_dof=['2Z+', '2Z+'],
                             ordinate_dim=['force**2/frequency',
                                           'acceleration**2/frequency'])
    p = predict_limits(frf(bump(3.0, 240, 260)), spec(), profiles)
    assert p.notches[0].depth_db == pytest.approx(10 * np.log10(2.0))


def test_an_undeclared_profile_is_preferred_over_another_quantity():
    """And with only a force and an undeclared one, the undeclared one is
    taken, and refused for its undeclared units, the nearer reason."""
    profiles = Specification(np.array([20.0, 2000.0]),
                             np.array([[1.0, 1.0], [1.0, 1.0]]),
                             response_dof=['2Z+', '2Z+'],
                             ordinate_dim=['force**2/frequency', 'unknown'])
    with pytest.raises(ValueError, match='the units of the limit profile at '
                                         '2Z\\+ are not declared'):
        predict_limits(frf(bump(3.0, 240, 260)), spec(), profiles)


def test_one_control_channel_in_two_quantities_is_refused():
    t = Frf(F, np.array([bump(3, 240, 260), bump(3, 240, 260)]),
            response_dof=['2Z+', '3Z+'], reference_dof=['1Z+', '1Z+'],
            ordinate_dim=['acceleration/acceleration', 'acceleration/force'])
    with pytest.raises(ValueError, match='in more than one quantity'):
        predict_limits(t, spec(0.18, 0.18))


def test_profiles_on_octave_bands_are_refused():
    profiles = Specification(np.array([20.0, 2000.0]),
                             np.array([[0.18, 0.18]]) * G ** 2,
                             response_dof=['2Z+'],
                             ordinate_dim='acceleration**2/frequency'
                             ).to_octave(3)
    with pytest.raises(ValueError, match='octave bands'):
        predict_limits(frf(bump(3, 240, 260)), spec(), profiles)
