"""Two signatures of one article compared mode by mode.

Held to a worked case whose answer is known from its construction: a
two-mode article measured at one point before and after a test, its
transmissibility computed exactly from two resonators on a moving base
(a global first mode with 85 percent of the motion there, a local mode
with 15). The first mode reads 84.0 Hz and a height of 18.0 before and
81.2 Hz and 13.0 after, a frequency shift of minus 3.3 percent and an
amplitude shift of minus 28, so under limits of 5 and 20 percent it
fails on amplitude alone. The local mode goes from 412 to 377 Hz and is
reported, not judged, when the caller says it does not count. Around
that case: the limits' edges, limits written as a pair, the window
each mode is read in, a sweep read downward, the records paired by DOF,
the modes found when none is named, and the reading in the window.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.data import Frf, Psd, Spectrum
from visualdynamics.core.signature import (
    AMPLITUDE_LIMIT,
    FREQUENCY_LIMIT,
    SignatureMode,
    as_modes,
    compare_signatures,
    find_modes,
    matching_record,
    mode_windows,
    read_peak,
    signature_shifts,
)

#: natural frequency, Q and share of the motion at the point, per mode
BEFORE = [(84.07, 21.1, 0.85), (412.0, 30.0, 0.15)]
AFTER = [(81.32, 15.25, 0.85), (377.0, 20.0, 0.15)]
#: quarter-hertz lines to 1600 Hz
F = np.arange(1, 6401) * 0.25
#: the modes as the analyst names them: the first counts, the local
#: mode does not
MODES = [84.0, (412.0, False)]


def transmissibility(f, modes):
    """Response over base motion at a point that sees each mode by its
    share: the sum of each resonator's absolute transmissibility."""
    out = np.zeros(np.shape(f), dtype=complex)
    for fn, q, share in modes:
        r, zeta = np.asarray(f) / fn, 0.5 / q
        out += share * (1 + 2j * zeta * r) / (1 - r ** 2 + 2j * zeta * r)
    return out


def signature(modes, f=F, dof='101X+', cls=Frf):
    """The point's transmissibility as an FRF, or its magnitude as a
    spectrum (what a tracked sweep's level is)."""
    if cls is Frf:
        return Frf(f, transmissibility(f, modes)[None], response_dof=[dof],
                   reference_dof=['1X+'],
                   ordinate_dim=['acceleration/acceleration'])
    return Spectrum(f, np.abs(transmissibility(f, modes))[None],
                    response_dof=[dof])


# ---- the worked case ---------------------------------------------------------


def test_the_worked_case_fails_on_amplitude_alone():
    result = compare_signatures(signature(BEFORE), signature(AFTER), MODES)
    first = result.shifts[0]
    assert round(first.before.frequency, 1) == 84.0
    assert round(first.after.frequency, 1) == 81.2
    assert round(first.before.height, 1) == 18.0
    assert round(first.after.height, 1) == 13.0
    assert round(first.frequency_shift, 1) == -3.3
    assert round(first.amplitude_shift) == -28
    assert first.frequency_inside and not first.amplitude_inside
    assert first.passed is False and first.verdict == 'flag'
    assert result.passed is False
    assert result.flagged == (first,)
    assert result.dof == '101X+'
    assert result.describe() == (
        'fails: 1 of 1 mode that counts flagged; 1 not judged')
    assert first.describe() == (
        '84.0 to 81.2 Hz (−3.3 %), height 18.0 to 13.0 '
        '(−27.6 %, outside): flag')


def test_a_local_mode_that_does_not_count_is_reported_not_judged():
    """412 to 377 Hz and a third of its height gone, past both limits,
    and still no verdict: the caller left it out."""
    result = compare_signatures(signature(BEFORE), signature(AFTER), MODES)
    local = result.shifts[1]
    assert round(local.before.frequency) == 412
    assert round(local.after.frequency) == 377
    assert round(local.frequency_shift, 1) == -8.5
    assert round(local.amplitude_shift) == -33
    assert not local.frequency_inside and not local.amplitude_inside
    assert local.passed is None and local.verdict == 'not judged'
    assert local not in result.flagged
    alone = compare_signatures(signature(BEFORE), signature(AFTER),
                               [(412.0, False)])
    assert alone.passed is None
    assert alone.describe() == 'no mode counts; 1 not judged'


def test_the_same_local_mode_counted_is_flagged():
    result = compare_signatures(signature(BEFORE), signature(AFTER),
                                [84.0, 412.0])
    assert [s.verdict for s in result.shifts] == ['flag', 'flag']
    assert result.describe() == 'fails: 2 of 2 modes that count flagged'


def test_each_mode_reads_its_q_before_and_after():
    """The half-power Q of the first mode, against the resonator's Q
    the point sees it through: 21.1 before and 15.25 after, read a
    little low on a mode that shares the point with another."""
    first = compare_signatures(signature(BEFORE), signature(AFTER),
                               MODES).shifts[0]
    assert first.before.q == pytest.approx(21.1, rel=0.01)
    assert first.after.q == pytest.approx(15.25, rel=0.01)
    assert first.before.notes == ()


def test_an_unchanged_article_passes_every_mode():
    result = compare_signatures(signature(BEFORE), signature(BEFORE),
                                [84.0, 412.0])
    assert result.passed is True
    assert [s.frequency_shift for s in result.shifts] == [0.0, 0.0]
    assert [s.amplitude_shift for s in result.shifts] == [0.0, 0.0]
    assert result.describe() == 'passes: 0 of 2 modes that count flagged'


# ---- the limits --------------------------------------------------------------


def test_the_default_limits_are_five_and_twenty_percent_either_way():
    assert (FREQUENCY_LIMIT, AMPLITUDE_LIMIT) == (5.0, 20.0)
    result = compare_signatures(signature(BEFORE), signature(AFTER), MODES)
    assert result.frequency_limit == (-5.0, 5.0)
    assert result.amplitude_limit == (-20.0, 20.0)


def test_a_shift_on_a_limits_edge_is_inside():
    """The curve after is the curve before, 20 percent taller and 5
    percent higher in frequency: exactly on both edges, and inside."""
    m = np.abs(transmissibility(F, BEFORE))
    on = signature_shifts(F, m, 1.2 * m, [84.0], after_frequencies=1.05 * F)
    shift = on.shifts[0]
    assert shift.frequency_shift == pytest.approx(5.0, abs=1e-9)
    assert shift.amplitude_shift == pytest.approx(20.0, abs=1e-9)
    assert shift.passed is True
    past = signature_shifts(F, m, 1.2001 * m, [84.0],
                            after_frequencies=1.05 * F).shifts[0]
    assert past.frequency_inside and not past.amplitude_inside
    assert past.passed is False


def test_a_shift_on_a_limits_lower_edge_is_inside():
    """The curve after 20 percent lower and 5 percent lower in
    frequency: the amplitude shift computes as -19.999999999999996 and
    the frequency shift as -4.999999999999993, on both lower edges, and
    inside; a hair further down in height is outside."""
    m = np.abs(transmissibility(F, BEFORE))
    on = signature_shifts(F, m, 0.8 * m, [84.0], after_frequencies=0.95 * F)
    shift = on.shifts[0]
    assert shift.frequency_shift == pytest.approx(-5.0, abs=1e-9)
    assert shift.amplitude_shift == pytest.approx(-20.0, abs=1e-9)
    assert shift.frequency_inside and shift.amplitude_inside
    past = signature_shifts(F, m, 0.7999 * m, [84.0],
                            after_frequencies=0.95 * F).shifts[0]
    assert past.frequency_inside and not past.amplitude_inside
    assert past.passed is False


def test_a_limit_written_as_a_pair_holds_a_drop_and_a_rise_apart():
    """A plan that allows a 30 percent drop in height but only a 10
    percent rise passes the worked case, and fails the same article
    read the other way round, when its first mode rises 38 percent."""
    before, after = signature(BEFORE), signature(AFTER)
    loose = compare_signatures(before, after, MODES,
                               amplitude_limit=(-30.0, 10.0))
    assert loose.passed is True
    assert loose.amplitude_limit == (-30.0, 10.0)
    backward = compare_signatures(after, before, MODES,
                                  amplitude_limit=(-30.0, 10.0))
    assert round(backward.shifts[0].amplitude_shift) == 38
    assert backward.passed is False


def test_the_box_is_the_limits_about_the_peak_before():
    shift = compare_signatures(signature(BEFORE), signature(AFTER),
                               MODES).shifts[0]
    f, h = shift.before.frequency, shift.before.height
    assert shift.box() == pytest.approx((0.95 * f, 1.05 * f, 0.8 * h, 1.2 * h))


@pytest.mark.parametrize('kwargs, message', [
    ({'frequency_limit': (1.0, 5.0)}, 'does not hold zero'),
    ({'amplitude_limit': 'wide'}, 'is a percentage or a pair'),
    ({'amplitude_limit': float('nan')}, 'does not hold zero'),
    ({'search': 1.5}, 'is not a share'),
])
def test_a_limit_or_search_that_cannot_be_read_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        compare_signatures(signature(BEFORE), signature(AFTER), MODES,
                           **kwargs)


# ---- naming the modes and reading them --------------------------------------


def test_modes_are_read_from_numbers_pairs_named_modes_and_shape_sets():
    class Shapes:
        frequency = np.array([412.0, 84.0])

    assert as_modes(84.0) == [SignatureMode(84.0)]
    assert as_modes([(412.0, False), 84]) == [
        SignatureMode(84.0), SignatureMode(412.0, counts=False)]
    assert as_modes(Shapes()) == [SignatureMode(84.0), SignatureMode(412.0)]
    named = SignatureMode(84.0, name='first bending')
    assert as_modes(named)[0].label() == 'first bending'
    assert SignatureMode(84.0).label() == '84 Hz'
    with pytest.raises(ValueError, match='positive frequency'):
        as_modes([0.0])


def test_a_shape_sets_rigid_body_mode_is_left_out():
    """A free-free fit holds a mode at 0 Hz, which has no peak to
    compare: the shape set names the others."""
    class Shapes:
        frequency = np.array([0.0, 412.0, 84.0])

    assert as_modes(Shapes()) == [SignatureMode(84.0), SignatureMode(412.0)]


def test_a_window_stops_halfway_to_the_next_mode_named():
    """A fifth either side, cut at the geometric mean of two
    neighbors, so two modes named close together are read apart."""
    modes = as_modes([100.0, 110.0, 400.0])
    windows = mode_windows(modes)
    middle = np.sqrt(100.0 * 110.0)
    assert windows[0] == pytest.approx((80.0, middle))
    assert windows[1] == pytest.approx((middle, 132.0))
    assert windows[2] == pytest.approx((320.0, 480.0))
    own = mode_windows([SignatureMode(84.0, within=(100.0, 70.0))])
    assert own == [(70.0, 100.0)]


def test_two_modes_named_close_are_each_read_on_their_own_peak():
    pair = [(100.0, 50.0, 0.5), (110.0, 50.0, 0.5)]
    result = compare_signatures(signature(pair), signature(pair),
                                [100.0, 110.0])
    assert [round(s.before.frequency) for s in result.shifts] == [100, 110]


def test_a_peak_outside_its_window_is_said_to_be():
    reading = read_peak(F, transmissibility(F, BEFORE), (90.0, 120.0))
    assert reading.frequency == 90.0
    assert any('window\'s edge' in note for note in reading.notes)


def test_a_top_placed_outside_its_window_is_said_to_be():
    """Two modes named at 100 and 110 Hz, the first moved to 106 after
    the test: its window stops at 104.9 Hz, the tallest line is on that
    edge, and the top placed between lines lands at 105 Hz, past it."""
    m = np.abs(transmissibility(F, [(100.0, 30.0, 0.5), (110.0, 30.0, 0.5)]))
    moved = np.abs(transmissibility(F, [(106.0, 30.0, 0.5),
                                        (110.0, 30.0, 0.5)]))
    first = signature_shifts(F, m, moved, [100.0, 110.0]).shifts[0]
    assert first.after.frequency == 105.0
    assert first.window[1] == pytest.approx(104.88, abs=0.01)
    assert first.after.notes == (
        ("the tallest line between 80 and 104.9 Hz is on the window's "
         'edge, at 104.8 Hz: the peak there may be outside the window, and '
         'its top, placed between lines, is at 105 Hz, outside it'),)


def test_two_modes_read_on_one_peak_are_said_to_be():
    """The same pair: after the test the second mode's window holds the
    first mode's peak, taller than its own, and reads a shift of -4.1
    percent that passes; the shift is not the second mode's, and both
    modes say they were read on one peak."""
    m = np.abs(transmissibility(F, [(100.0, 30.0, 0.5), (110.0, 30.0, 0.5)]))
    moved = np.abs(transmissibility(F, [(106.0, 30.0, 0.5),
                                        (110.0, 30.0, 0.5)]))
    result = signature_shifts(F, m, moved, [100.0, 110.0])
    second = result.shifts[1]
    assert round(second.frequency_shift, 1) == -4.1
    assert second.passed is True
    assert second.notes == (
        ('after the test: read on the same peak as 100 Hz, which tops at '
         '105.7 Hz'),)
    assert result.shifts[0].notes[-1] == (
        'after the test: read on the same peak as 110 Hz, which tops at '
        '105.7 Hz')
    assert result.notes[-1] == (
        '110 Hz, after the test: read on the same peak as 100 Hz, which '
        'tops at 105.7 Hz')
    assert len(result.notes) == 3
    assert result.describe() == (
        'fails: 1 of 2 modes that count flagged; 3 notes')
    # read apart before the test, so nothing is said of it there
    assert not any('before' in note for note in result.notes)


def test_a_mode_moved_out_of_its_window_carries_its_note_to_the_comparison():
    """100 Hz moved to 75, past the window's low end at 80: the reading
    after is the window's edge, a shift of -20 percent where the mode
    moved 25, and the comparison says so rather than reading clean."""
    m = np.abs(transmissibility(F, [(100.0, 30.0, 1.0)]))
    moved = np.abs(transmissibility(F, [(75.0, 30.0, 1.0)]))
    result = signature_shifts(F, m, moved, [100.0])
    assert round(result.shifts[0].frequency_shift) == -20
    assert result.notes == (
        ("100 Hz, after the test: the tallest line between 80 and 120 Hz is "
         "on the window's edge, at 80 Hz: the peak there may be outside the "
         'window'),)
    assert result.describe().endswith('; 1 note')
    records = compare_signatures(
        Spectrum(F, m[None], response_dof=['101X+']),
        Spectrum(F, moved[None], response_dof=['101X+']), [100.0])
    assert records.notes == result.notes


def test_a_line_with_no_value_in_the_window_is_said():
    """A NaN on the peak's flank is read as no part of it, and said: the
    crossing beside it is placed at the last line read."""
    m = np.abs(transmissibility(F, BEFORE))
    m[np.argmin(np.abs(F - 83.0))] = np.nan
    reading = read_peak(F, m, (70.0, 100.0))
    assert reading.low == 83.25
    assert reading.notes == (
        ('1 line between 70 and 100 Hz holds no value (NaN or infinite) '
         'and is read as no part of the peak'),)
    m[np.argmin(np.abs(F - 90.0))] = np.inf
    assert read_peak(F, m, (70.0, 100.0)).notes[0].startswith(
        '2 lines between 70 and 100 Hz hold no value')


def test_a_window_with_nothing_above_zero_is_refused():
    """A curve before the test that is zero across the mode's window has
    no peak, and its height of zero would divide the amplitude shift."""
    zero = np.abs(transmissibility(F, BEFORE))
    zero[(F >= 60.0) & (F <= 110.0)] = 0.0
    with pytest.raises(ValueError, match='no line there stands above zero'):
        signature_shifts(F, zero, np.abs(transmissibility(F, AFTER)), [84.0])


def test_signatures_of_different_quantities_are_refused():
    """An accelerance against a transmissibility: the heights are not
    one scale, and the amplitude shift would be a number of nothing."""
    accelerance = Frf(F, transmissibility(F, AFTER)[None],
                      response_dof=['101X+'], reference_dof=['1X+'],
                      ordinate_dim=['acceleration/force'])
    with pytest.raises(ValueError, match='acceleration/acceleration and the '
                       'one after is acceleration/force'):
        compare_signatures(signature(BEFORE), accelerance, MODES)
    # a record with undeclared units is taken as the other's
    level = signature(AFTER, cls=Spectrum)
    assert compare_signatures(signature(BEFORE), level, MODES).passed is False


def test_a_window_with_no_line_in_it_is_refused():
    with pytest.raises(ValueError, match='no line of the curve'):
        compare_signatures(signature(BEFORE), signature(AFTER),
                           [SignatureMode(84.0, within=(2000.0, 3000.0))])


def test_a_peak_with_no_half_power_crossing_reads_no_q():
    """A curve that ends on the peak's flank has no width to read."""
    keep = F <= 85.0
    m = np.abs(transmissibility(F, BEFORE))[keep]
    reading = read_peak(F[keep], m, (70.0, 85.0))
    assert reading.high is None and reading.q is None
    assert reading.notes == (
        ('the curve ends at 85 Hz before falling to half power above the '
         'peak at 84 Hz; no width or Q is read'),)


def test_a_tracked_sweep_on_an_uneven_axis_reads_the_worked_case():
    """A log sweep's levels: lines a hundredth of an octave apart,
    magnitude only. The same verdict, the peaks placed between lines."""
    f = 20.0 * 2.0 ** (np.arange(0, 6.0, 1.0 / 400.0))
    result = compare_signatures(signature(BEFORE, f, cls=Spectrum),
                                signature(AFTER, f, cls=Spectrum), MODES)
    first = result.shifts[0]
    assert first.frequency_shift == pytest.approx(-3.34, abs=0.05)
    assert round(first.amplitude_shift) == -28
    assert result.passed is False


def test_a_sweep_read_downward_reads_the_same():
    m_before = np.abs(transmissibility(F, BEFORE))
    m_after = np.abs(transmissibility(F, AFTER))
    up = signature_shifts(F, m_before, m_after, MODES)
    down = signature_shifts(F[::-1], m_before[::-1], m_after[::-1], MODES)
    assert [s.describe() for s in down.shifts] == [
        s.describe() for s in up.shifts]


def test_the_record_after_is_the_one_at_the_same_dof():
    """Two points measured after the test, in the other order: the
    record compared is the one at the point compared before."""
    before = signature(BEFORE, dof='101X+')
    other = transmissibility(F, [(84.07, 21.1, 0.85), (412.0, 30.0, 0.15)])
    after = Frf(F, np.vstack([other, transmissibility(F, AFTER)]),
                response_dof=['202X+', '101X+'],
                reference_dof=['1X+', '1X+'],
                ordinate_dim=['acceleration/acceleration'] * 2)
    result = compare_signatures(before, after, MODES)
    assert result.passed is False                # read at 101X+, not 202X+
    forced = compare_signatures(before, after, MODES, after_record=0)
    assert forced.passed is True


def test_with_no_record_at_the_same_dof_the_same_index_is_compared():
    before = signature(BEFORE, dof='101X+')
    after = signature(AFTER, dof='303X+')
    assert matching_record(before, 0, after) == 0
    assert compare_signatures(before, after, MODES).passed is False
    two = Frf(F, np.vstack([transmissibility(F, BEFORE)] * 2),
              response_dof=['101X+', '102X+'], reference_dof=['1X+'] * 2,
              ordinate_dim=['acceleration/acceleration'] * 2)
    with pytest.raises(ValueError, match=r'no record at 102X\+ and no '
                       'record 1'):
        matching_record(two, 1, after)


# ---- finding the modes -------------------------------------------------------


def test_the_modes_are_found_on_the_signature_before():
    found = find_modes(signature(BEFORE))
    assert [round(f, 1) for f in found] == [84.0, 411.9]
    assert find_modes(signature(BEFORE, cls=Spectrum)) == found
    assert [round(f) for f in find_modes(signature(BEFORE), count=1)] == [84]
    assert [round(f) for f in find_modes(signature(BEFORE), count=5)] == [
        84, 412]
    assert find_modes(signature(BEFORE), bounds=(200.0, 600.0),
                      count=1) == pytest.approx([found[1]])


def test_the_modes_are_ranked_by_prominence_not_height():
    """A sharp mode low in the band beside a broad hump that stands
    taller: the sharp mode is the first found, as the modal fit would
    offer it."""
    sharp = np.abs(transmissibility(F, [(60.0, 40.0, 1.0)]))
    hump = 45.0 * np.exp(-0.5 * ((F - 900.0) / 200.0) ** 2)
    values = sharp + hump
    curve = Spectrum(F, values[None], response_dof=['101X+'])
    assert values[F > 500].max() > values[F < 100].max()
    assert [round(f) for f in find_modes(curve, count=1)] == [60]
    assert [round(f) for f in find_modes(curve, count=2)] == [60, 900]


def test_the_modes_a_modal_fit_confirmed_can_be_compared():
    """A shape set from the fitting screen's loop names the modes."""
    import visualdynamics

    project = visualdynamics.Project()
    project.add('Before', signature(BEFORE))
    project.add('After', signature(AFTER))
    shapes = project[project.fit_modes('Before', limit=2)]
    result = compare_signatures(project['Before'], project['After'], shapes)
    assert [round(s.before.frequency) for s in result.shifts] == [84, 412]
    assert [s.verdict for s in result.shifts] == ['flag', 'flag']


def test_a_peak_inside_the_width_of_one_taken_is_the_same_mode():
    """A ripple on the flank of a broad mode, inside its half-power
    width, is a local peak and stands out from its surroundings; it is
    the same mode, and is passed over. (The broad mode's top reads 99
    Hz: a Q of 5 sits that far below its natural frequency.)"""
    broad = np.abs(transmissibility(F, [(100.0, 5.0, 1.0)]))
    ripple = broad * (1.0 + 0.1 * np.exp(-0.5 * ((F - 106.0) / 0.4) ** 2))
    curve = Spectrum(F, ripple[None], response_dof=['101X+'])
    i = int(np.argmin(np.abs(F - 106.0)))
    assert ripple[i] > ripple[i - 1] and ripple[i] > ripple[i + 1]
    assert [round(f) for f in find_modes(curve, count=2)] == [99]


def test_the_modes_are_found_within_bounds():
    """Bounds leave out a peak outside them, given either way round."""
    curve = signature(BEFORE)
    assert [round(f) for f in find_modes(curve, bounds=(600.0, 200.0))] == [
        412]
    assert [round(f) for f in find_modes(curve, bounds=(50.0, 100.0))] == [84]
    assert find_modes(curve, bounds=(500.0, 1000.0)) == []


def test_a_curve_with_no_peak_finds_none():
    flat = Spectrum(F, np.ones((1, len(F))), response_dof=['101X+'])
    assert find_modes(flat) == []


# ---- the view ----------------------------------------------------------------


def test_the_view_draws_headless(tmp_path):
    from visualdynamics.plot import plot_signature

    path = tmp_path / 'signature.png'
    plot_signature(signature(BEFORE), signature(AFTER),
                   [84.0, SignatureMode(412.0, counts=False)],
                   path=str(path), show=False)
    assert path.exists() and path.stat().st_size > 2000


def test_the_view_draws_a_window_on_each_mode_and_says_its_verdict(qt_app):
    """A solid window on the mode that counts and a dotted one on the
    mode that does not, each at its limits about the peak before, its
    verdict written over it, both peaks marked, and the table beside."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from visualdynamics.plot.signature import build_signature

    layout = pg.GraphicsLayoutWidget()
    try:
        result = build_signature(layout, signature(BEFORE),
                                 signature(AFTER), MODES)
        plot = layout.getItem(0, 0)
        windows = [item for item in plot.listDataItems()
                   if getattr(item, 'is_signature_window', False)]
        assert [w.mode_counts for w in windows] == [True, False]
        assert [w.opts['pen'].style() for w in windows] == [
            Qt.PenStyle.SolidLine, Qt.PenStyle.DotLine]
        f_low, f_high, h_low, h_high = result.shifts[0].box()
        assert np.allclose(windows[0].xData, [f_low, f_high, f_high, f_low,
                                              f_low])
        assert np.allclose(windows[0].yData, [h_low, h_low, h_high, h_high,
                                              h_low])
        peaks = [item for item in plot.listDataItems()
                 if getattr(item, 'is_signature_peak', False)]
        assert [p.opts['symbol'] for p in peaks] == ['o', 's', 'o', 's']
        labels = [item.toPlainText() for item in plot.items
                  if getattr(item, 'is_signature_label', False)]
        assert labels == ['84 Hz: flag', '412 Hz: not judged']
        names = [label.text for _sample, label in plot.legend.items]
        assert names[:2] == ['before', 'after']
        table = layout.getItem(0, 1)
        assert table.is_signature_table
        assert 'not judged' in table.text and '−27.6 %' in table.text
        assert 'fails: 1 of 1 mode that counts flagged' in table.text
        assert 'height before, after [g/g]' in table.text
    finally:
        layout.close()


def test_the_view_leaves_the_table_to_the_panel_when_asked(qt_app):
    import pyqtgraph as pg

    from visualdynamics.plot.signature import build_signature

    layout = pg.GraphicsLayoutWidget()
    try:
        build_signature(layout, signature(BEFORE), signature(AFTER), MODES,
                        table=False)
        assert layout.getItem(0, 1) is None
    finally:
        layout.close()


# ---- the reading in the window -----------------------------------------------


def _signatures(window, pump, before=BEFORE, after=AFTER, cls=Frf):
    """Two signatures in the window, selected, on the Signature
    reading."""
    from conftest import select_objects

    window.add_object('Before', signature(before, cls=cls))
    window.add_object('After', signature(after, cls=cls))
    select_objects(window, pump, 'Before', 'After')
    pane = window.data_pane
    assert pane.signature_action.isVisible()
    pane.signature_action.trigger()
    pump()
    return pane.signature_panel


def test_the_signature_is_a_reading_of_two_frfs(window, pump):
    """Offered beside the overlay; choosing it puts the panel up, and
    with no mode named the two most prominent peaks are found and
    judged."""
    panel = _signatures(window, pump)
    assert panel.isVisible()
    assert window.data_pane.signature_action.text() == 'Signature'
    assert panel.derived['verdict'].text() == 'fails'
    assert panel.readout.text().splitlines()[:4] == [
        '84 Hz: flag',
        '  84.0 to 81.2 Hz, −3.3 %',
        '  height 18.0 to 13.0 g/g, −27.6 %',
        '  Q 21.1 to 15.2']
    assert window._status_text == (
        'Before before, After after, at 101X+ (modes found): '
        'fails: 2 of 2 modes that count flagged')
    plot = window.data_pane.graphics.getItem(0, 0)
    assert [item for item in plot.listDataItems()
            if getattr(item, 'is_signature_window', False)]


def test_the_panels_modes_and_limits_redraw_the_comparison(window, pump):
    panel = _signatures(window, pump)
    panel.modes_edit.setText('84, (412)')
    panel.modes_edit.editingFinished.emit()
    pump()
    assert not panel.find_box.isVisibleTo(panel)
    assert panel.readout.text().splitlines()[4] == '412 Hz: not judged'
    assert 'fails: 1 of 1 mode that counts flagged; 1 not judged' in \
        window._status_text
    panel.amplitude_box.setValue(30.0)
    pump()
    assert panel.derived['verdict'].text() == 'passes'
    panel.frequency_box.setValue(3.0)
    pump()
    assert panel.derived['verdict'].text() == 'fails'
    assert panel.readout.text().startswith('84 Hz: flag')


def test_either_selected_can_be_the_signature_before(window, pump):
    panel = _signatures(window, pump)
    panel.set_settings(before='After', modes='84')
    pump()
    assert panel.readout.text().splitlines()[1] == (
        '  81.2 to 84.0 Hz, +3.5 %')
    assert window._status_text.startswith('After before, Before after')


def test_a_modes_field_that_cannot_be_read_says_why(window, pump):
    panel = _signatures(window, pump)
    panel.set_settings(modes='84, eighty')
    pump()
    assert panel.derived['verdict'].text() == '—'
    assert window._status_text == "Signature: 'eighty' is not a frequency"


def test_the_signature_reading_is_journaled_as_its_plot_call(window, pump):
    panel = _signatures(window, pump)
    panel.set_settings(modes='84, (412)', amplitude_limit=25.0)
    pump()
    lines = [line for line in window.project.journal
             if line.startswith('visualdynamics.plot.plot_signature(')]
    assert lines == [
        ("visualdynamics.plot.plot_signature(project['Before'], "
         "project['After'], [(84.0, True), (412.0, False)], record=0, "
         'frequency_limit=5.0, amplitude_limit=25.0, '
         "names=('Before', 'After'), path='signature.png', show=False)")]


def test_tracked_sweep_levels_read_the_same_way(window, pump):
    panel = _signatures(window, pump, cls=Spectrum)
    panel.set_settings(modes='84, (412)')
    pump()
    assert panel.readout.text().startswith('84 Hz: flag')


def test_the_signature_is_not_offered_for_other_pairs(window, pump):
    """An FRF beside a spectrum is not two signatures of one kind, and
    two densities have their own readings."""
    from conftest import select_objects

    window.add_object('Before', signature(BEFORE))
    window.add_object('Level', signature(AFTER, cls=Spectrum))
    window.add_object('Noise', Psd(F, np.ones((1, len(F))),
                                   response_dof=['101X+'],
                                   ordinate_dim=['acceleration']))
    window.add_object('Noise 2', Psd(F, np.ones((1, len(F))),
                                     response_dof=['101X+'],
                                     ordinate_dim=['acceleration']))
    pane = window.data_pane
    for pair in (('Before', 'Level'), ('Noise', 'Noise 2')):
        select_objects(window, pump, *pair)
        assert not pane.signature_action.isVisible()
        assert not pane.signature_panel.isVisibleTo(pane)


def test_the_event_box_picks_the_point_compared(window, pump):
    """A signature at two points: the box lists both, and the second
    is compared against its own record after the test."""
    from conftest import select_objects

    def two_points(first, second):
        return Frf(F, np.vstack([transmissibility(F, first),
                                 transmissibility(F, second)]),
                   response_dof=['101X+', '202X+'],
                   reference_dof=['1X+', '1X+'],
                   ordinate_dim=['acceleration/acceleration'] * 2)

    window.add_object('Before', two_points(BEFORE, BEFORE))
    window.add_object('After', two_points(AFTER, BEFORE))
    select_objects(window, pump, 'Before', 'After')
    pane = window.data_pane
    pane.signature_action.trigger()
    pane.signature_panel.set_settings(modes='84, (412)')
    pump()
    assert [pane.event_box.itemText(i) for i in range(2)] == [
        '101X+ / 1X+', '202X+ / 1X+']
    assert pane.signature_panel.derived['verdict'].text() == 'fails'
    pane.event_box.setCurrentIndex(1)
    pump()
    assert 'at 202X+' in window._status_text
    assert pane.signature_panel.derived['verdict'].text() == 'passes'


def test_a_readings_notes_are_in_the_panel_the_status_and_the_table(
        window, pump):
    """The mode named at 100 Hz moved to 75, out of its window: the
    panel lists the note under the mode, the status line carries it
    after the verdict, and the table the view draws ends with it."""
    import pyqtgraph as pg

    from visualdynamics.plot.signature import build_signature

    note = ("100 Hz, after the test: the tallest line between 80 and 120 Hz "
            "is on the window's edge, at 80 Hz: the peak there may be "
            'outside the window')
    panel = _signatures(window, pump, before=[(100.0, 30.0, 1.0)],
                        after=[(75.0, 30.0, 1.0)])
    panel.set_settings(modes='100')
    pump()
    assert panel.readout.text().splitlines()[4] == (
        '  after the test: ' + note.split('after the test: ')[1])
    assert window._status_text == (
        'Before before, After after, at 101X+: fails: 1 of 1 mode that '
        f'counts flagged; 1 note. {note}')
    layout = pg.GraphicsLayoutWidget()
    try:
        build_signature(layout, signature([(100.0, 30.0, 1.0)]),
                        signature([(75.0, 30.0, 1.0)]), [100.0])
        assert layout.getItem(0, 1).text.endswith(
            '<br/>' + note.replace("'", '&#x27;') + '</span>')
    finally:
        layout.close()


def test_signatures_of_different_quantities_say_why_in_the_window(
        window, pump):
    from conftest import select_objects

    window.add_object('Before', signature(BEFORE))
    window.add_object('After', Frf(F, transmissibility(F, AFTER)[None],
                                   response_dof=['101X+'],
                                   reference_dof=['1X+'],
                                   ordinate_dim=['acceleration/force']))
    select_objects(window, pump, 'Before', 'After')
    window.data_pane.signature_action.trigger()
    pump()
    panel = window.data_pane.signature_panel
    assert panel.derived['verdict'].text() == '—'
    assert window._status_text == (
        'Signature: the signature before is acceleration/acceleration and '
        'the one after is acceleration/force: two signatures of one '
        'quantity are compared')


def test_a_new_pair_clears_the_modes_named_for_the_last(window, pump):
    """Modes typed for one article are not carried to the next pair
    selected: its own are found, and the peak count is offered again."""
    from conftest import select_objects

    panel = _signatures(window, pump)
    panel.set_settings(modes='84, (412)')
    pump()
    assert panel.names['find'].text() == 'Peaks found'
    assert not panel.find_box.isVisibleTo(panel)
    window.add_object('Again', signature(AFTER))
    select_objects(window, pump, 'Before', 'Again')
    pump()
    assert panel.modes_edit.text() == ''
    assert panel.find_box.isVisibleTo(panel)
    assert '(modes found)' in window._status_text
