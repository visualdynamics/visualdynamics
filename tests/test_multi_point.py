"""Multi-point control: the composite of several control channels, and
the two judgements a run controlled that way is accepted on.

Every case is built so its numbers are known from the construction: a
flat reference, control channels that are the reference times a known
factor at every line, so the composite under each strategy is a hand
value, and the dB each channel stands from the reference is ten log ten
of its factor. The cases that matter most are the ones the composite
cannot see: a dead channel inside an average, which leaves the
composite in tolerance while the live channels run high, and a hot
spot diluted by the channels around it.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.compliance import RAW_NOTE, compare
from visualdynamics.core.data import Psd, Specification
from visualdynamics.core.multi_point import (
    COMPOSITE_DB,
    INDIVIDUAL_DB,
    composite,
    control_channels,
    multi_point,
)

#: the reference level, (m/s²)²/Hz, flat from 20 Hz to 2 kHz
LEVEL = 0.4
PSD_DIM = 'acceleration**2/frequency'
LINES = np.arange(0.0, 2001.0, 5.0)
#: the lines inside the reference's band, each judged once
JUDGED = int(((LINES >= 20.0) & (LINES <= 2000.0)).sum())


def reference(channels=('1Z+',), db=3.0, dim=PSD_DIM, levels=None):
    """A flat reference with warning limits `db` either side, one record
    per channel named (the same level on each unless `levels` says)."""
    levels = [LEVEL] * len(channels) if levels is None else levels
    rows = np.array([[v, v] for v in levels])
    factor = 10.0 ** (db / 10.0)
    return Specification([20.0, 2000.0], rows, response_dof=list(channels),
                         ordinate_dim=[dim] * len(channels),
                         warning_lower=rows / factor,
                         warning_upper=rows * factor)


def run(*factors, dofs=None, dims=None, nan_at=None):
    """A PSD whose channel k is the reference times `factors[k]` at every
    line, at DOFs 1Z+, 2Z+... unless named."""
    dofs = dofs or [f'{k + 1}Z+' for k in range(len(factors))]
    values = np.array([np.full(LINES.size, LEVEL * f) for f in factors])
    if nan_at is not None:
        values[nan_at[0], nan_at[1]] = np.nan
    return Psd(LINES, values, response_dof=dofs,
               ordinate_dim=dims or [PSD_DIM] * len(factors))


def db(factor):
    return 10.0 * np.log10(factor)


# ---- the composite, by strategy -------------------------------------------


def test_each_strategy_combines_the_densities_line_by_line():
    """Four times and a quarter: the average is 2.125 (just over twice,
    the hot spot visible and far smaller than it is), the maximum 4,
    the minimum a quarter, and weighted three to one, 3.0625."""
    values = np.array([[4.0, 4.0], [0.25, 0.25]])
    assert composite(values, 'average') == pytest.approx([2.125, 2.125])
    assert composite(values, 'maximum') == pytest.approx([4.0, 4.0])
    assert composite(values, 'minimum') == pytest.approx([0.25, 0.25])
    assert composite(values, 'weighted', [3, 1]) == pytest.approx(
        [3.0625, 3.0625])


def test_the_average_is_of_the_densities_not_their_decibels():
    """A mean in dB of +6 and -6 would be 0 dB; on the densities it is
    +3.3 dB, the number a controller holds."""
    values = np.array([[4.0], [0.25]])
    assert db(composite(values)[0]) == pytest.approx(3.274, abs=1e-3)


def test_a_line_not_formed_in_one_channel_is_not_formed_at_all():
    """A NaN or a negative density in any channel leaves the line
    without a composite, rather than an average of the others; a zero
    is a reading and is counted."""
    values = np.array([[1.0, 1.0, 1.0, 1.0], [np.nan, -1.0, 0.0, 3.0]])
    out = composite(values)
    assert np.isnan(out[:2]).all()
    assert out[2:] == pytest.approx([0.5, 2.0])


@pytest.mark.parametrize('strategy,weights,reason', [
    ('mean', None, 'the strategy is one of average, weighted'),
    ('average', [1, 1], 'weights are for the weighted strategy'),
    ('maximum', [1, 1], 'maximum takes none'),
    ('weighted', None, 'needs a weight per channel'),
    ('weighted', [1, 1, 1], '3 weights for 2 channels'),
    ('weighted', [1, -1], 'zero or more'),
    ('weighted', [1, np.nan], 'zero or more'),
    ('weighted', [0, 0], 'add up to zero'),
])
def test_a_strategy_or_weights_that_cannot_be_used_are_refused(
        strategy, weights, reason):
    with pytest.raises(ValueError, match=reason):
        composite(np.ones((2, 3)), strategy, weights)


# ---- the two judgements -----------------------------------------------------


def test_a_dead_channel_leaves_the_composite_inside_and_is_found():
    """Two live channels at 1.5 times the reference and one reading
    zero: the average is the reference at every line, so the composite
    is in tolerance; the live channels run +1.8 dB, inside their own
    band; and the dead one is outside its own band at every line, each
    of them a line the composite is inside at."""
    reading = multi_point(reference(), run(1.5, 1.5, 0.0),
                          channels=['1Z+', '2Z+', '3Z+'], scale_db=0)
    assert reading.composite.ordinate[0][LINES >= 20] == pytest.approx(LEVEL)
    assert reading.outside_lines == 0
    assert reading.result['difference_db'] == pytest.approx(0.0, abs=1e-9)
    live, other, dead = reading.channels
    assert live.result['difference_db'] == pytest.approx(
        db(1.5), abs=1e-9)
    assert (live.outside_lines, other.outside_lines) == (0, 0)
    assert dead.outside_lines == dead.hidden_lines == JUDGED
    assert reading.hiding == (dead,)
    assert reading.notes == (('3Z+ reads zero at every line, as a dead or '
                             'disconnected sensor does; the composite '
                             'counts it as it reads'),)


def test_a_channel_far_below_the_others_is_noted_as_dead():
    """Not zero but forty dB down, as a disconnected sensor's noise
    reads: said with its distance."""
    reading = multi_point(reference(), run(1.0, 1.0, 1e-4),
                          channels=[0, 1, 2], scale_db=0)
    assert reading.notes == (('3Z+ reads 40 dB below the other channels at '
                             'its median line, as a dead or disconnected '
                             'sensor does; the composite counts it as it '
                             'reads'),)


def test_a_hot_spot_diluted_four_ways_is_outside_only_on_its_own():
    """One channel of four at 4.4 times the reference: the average is
    1.85 times, +2.7 dB, inside -3/+3; the hot channel is +6.4 dB,
    outside -6/+6, at every line the composite is inside at."""
    reading = multi_point(reference(), run(4.4, 1.0, 1.0, 1.0),
                          channels=[0, 1, 2, 3], scale_db=0)
    assert reading.result['difference_db'] == pytest.approx(
        db(1.85), abs=1e-9)
    assert reading.outside_lines == 0
    hot = reading.channels[0]
    assert hot.result['difference_db'] == pytest.approx(db(4.4), abs=1e-9)
    assert hot.over.sum() == hot.hidden_lines == JUDGED
    assert not hot.under.any()
    assert [c.outside_lines for c in reading.channels[1:]] == [0, 0, 0]


def test_a_composite_outside_its_band_hides_nothing():
    """Two of two channels high: the composite is out at every line, so
    no line of a channel's is one the composite was inside at."""
    reading = multi_point(reference(), run(4.4, 4.4), channels=[0, 1],
                          scale_db=0)
    assert reading.outside_lines == JUDGED
    assert reading.over.sum() == JUDGED and not reading.under.any()
    assert [c.outside_lines for c in reading.channels] == [JUDGED, JUDGED]
    assert reading.hiding == ()


def test_maximum_control_holds_the_highest_and_says_which_it_was():
    """At 1, a half and a quarter, maximum control's composite is the
    first channel, which is the one taken at every line, and the others
    are under it by 3 and 6 dB; minimum control takes the last."""
    measured = run(1.0, 0.5, 0.25)
    high = multi_point(reference(), measured, channels=[0, 1, 2],
                       strategy='maximum', scale_db=0)
    assert high.result['difference_db'] == pytest.approx(0.0, abs=1e-9)
    assert [c.held_lines for c in high.channels] == [LINES.size, 0, 0]
    assert all(np.isnan(c.weight) for c in high.channels)
    low = multi_point(reference(), measured, channels=[0, 1, 2],
                      strategy='minimum', scale_db=0)
    assert low.result['difference_db'] == pytest.approx(db(0.25), abs=1e-9)
    assert [c.held_lines for c in low.channels] == [0, 0, LINES.size]


def test_weights_are_normalized_and_a_zero_weight_is_noted():
    reading = multi_point(reference(), run(3.0, 1.0, 1.0),
                          channels=[0, 1, 2], strategy='weighted',
                          weights=[1, 1, 0], scale_db=0)
    assert [c.weight for c in reading.channels] == [0.5, 0.5, 0.0]
    assert reading.result['difference_db'] == pytest.approx(db(2.0),
                                                            abs=1e-9)
    assert reading.notes == (('3Z+ has weight zero and is not in the '
                             'composite; it is still judged against its '
                             'own band'),)
    assert [c.held_lines for c in reading.channels] == [None] * 3


def test_the_composite_is_judged_by_the_comparison_every_reading_uses():
    """The composite's numbers are `compare`'s on the composite itself,
    against the specification's own warning limits: one judge, not a
    second copy of it."""
    spec = reference()
    reading = multi_point(spec, run(2.5, 1.2, 0.9), channels=[0, 1, 2],
                          scale_db=0)
    direct = compare(spec, reading.composite, 0, 0, scale_db=0)
    assert reading.result['difference_db'] == direct['difference_db']
    assert reading.result['measured_rms'] == direct['measured_rms']
    assert reading.result['outside_lines'] == direct['warning_lines']
    assert reading.result['outside_percent'] == direct['warning_percent']
    assert reading.composite_text == 'the warning limits'


def test_a_band_in_db_reads_the_same_as_limits_written_at_it():
    """-3/+3 dB typed is the reference's ±3 dB warning limits."""
    spec, measured = reference(db=3.0), run(2.1, 1.9)
    named = multi_point(spec, measured, channels=[0, 1], scale_db=0)
    typed = multi_point(spec, measured, channels=[0, 1], scale_db=0,
                        composite_band=(-3.0, 3.0))
    assert typed.composite_text == '−3/+3 dB'
    assert np.array_equal(named.over, typed.over)
    assert named.result['outside_lines'] == typed.result['outside_lines'] > 0
    # ten log ten on a density: 1.7 times is +2.3 dB, inside +3, and a
    # channel at 3.5 times is +5.4 dB, inside +6 (twenty log ten would
    # put both out)
    near = multi_point(spec, run(3.5, 0.5, 1.1), channels=[0, 1, 2],
                       scale_db=0, composite_band=(-3.0, 3.0))
    assert near.result['difference_db'] == pytest.approx(db(1.7))
    assert near.outside_lines == 0
    assert near.channels[0].outside_lines == 0


def test_an_open_edge_is_no_edge_on_that_side():
    """With no ceiling on a channel's band, a channel far over it is
    not outside, and one far under is."""
    reading = multi_point(reference(), run(9.0, 0.1), channels=[0, 1],
                          scale_db=0, individual_band=(-6.0, np.inf))
    assert reading.individual_text == '−6/up dB'
    hot, cold = reading.channels
    assert hot.outside_lines == 0
    assert cold.under.sum() == JUDGED


def test_the_defaults_are_the_starting_points_named():
    assert INDIVIDUAL_DB == (-6.0, 6.0) and COMPOSITE_DB == (-3.0, 3.0)
    reading = multi_point(reference(), run(1.0, 1.0), channels=[0, 1],
                          scale_db=0)
    assert reading.individual_text == '−6/+6 dB'
    assert reading.strategy == 'average'


def test_a_line_not_formed_is_not_judged_and_is_noted():
    """A NaN at five lines of one channel (and the other 7 dB under the
    reference, the composite on it): the composite is not formed
    there and those lines are not judged, for the composite or as
    hidden lines, and the note says how many."""
    reading = multi_point(reference(), run(1.8, 0.2, nan_at=(0, slice(100, 105))),
                          channels=[0, 1], scale_db=0)
    assert np.isnan(reading.composite.ordinate[0][100:105]).all()
    assert reading.result['lines'] == JUDGED - 5
    dead = reading.channels[1]
    assert dead.outside_lines == JUDGED
    assert dead.hidden_lines == JUDGED - 5
    assert ('1Z+ has 5 lines that are not a finite, non-negative density; '
            'the composite is not formed or judged there') in reading.notes


# ---- which channels, and against which reference ---------------------------


def test_the_channels_default_to_those_the_specification_pairs_with():
    """A reference written at three control channels, a run holding
    those and a response: the composite is of the three."""
    spec = reference(('1Z+', '2Z+', '3Z+'))
    measured = run(1.0, 1.0, 1.0, 9.0, dofs=['1Z+', '2Z+', '3Z+', '9Z+'])
    assert control_channels(spec, measured) == [0, 1, 2]
    reading = multi_point(spec, measured, scale_db=0)
    assert [c.label for c in reading.channels] == ['1Z+', '2Z+', '3Z+']
    assert reading.composite.comment == ['average of 1Z+, 2Z+, 3Z+']


def test_a_reference_that_differs_between_channels_is_refused():
    spec = reference(('1Z+', '2Z+'), levels=[LEVEL, 2 * LEVEL])
    with pytest.raises(ValueError, match='asks 2Z\\+ for another level '
                                         'than 1Z\\+'):
        multi_point(spec, run(1.0, 1.0), scale_db=0)
    # named, it is the one held to
    reading = multi_point(spec, run(1.0, 1.0), spec_record=1, scale_db=0)
    assert reading.spec_record == 1
    assert reading.result['difference_db'] == pytest.approx(db(0.5))


@pytest.mark.parametrize('channels,reason', [
    (['1Z+'], 'needs two or more control channels; one is a single-point'),
    ([0, 0], '1Z\\+ is named twice'),
    (['7Z+', '1Z+'], 'no autospectrum at 7Z\\+'),
    ([0, 9], 'there is no record 9'),
])
def test_channels_that_cannot_make_a_composite_are_refused(channels, reason):
    with pytest.raises(ValueError, match=reason):
        multi_point(reference(), run(1.0, 1.0), channels=channels,
                    scale_db=0)


def test_with_nothing_paired_the_channels_must_be_named():
    with pytest.raises(ValueError, match='pairs with the specification, '
                                         'so name them'):
        control_channels(reference(('5Z+',)), run(1.0, 1.0))


def test_a_cross_term_is_not_a_control_channel():
    from visualdynamics.core.data import Psd as Cpsd

    values = np.full((3, LINES.size), LEVEL, dtype=complex)
    cpsd = Cpsd(LINES, values, response_dof=['1Z+', '2Z+', '1Z+'],
                reference_dof=['1Z+', '2Z+', '2Z+'],
                ordinate_dim=[PSD_DIM] * 3)
    with pytest.raises(ValueError, match='record 2 is a cross term, '
                                         '1Z\\+/2Z\\+'):
        control_channels(reference(), cpsd, [0, 2])
    assert control_channels(reference(('1Z+', '2Z+')), cpsd) == [0, 1]


def test_two_records_at_one_dof_are_named_by_record():
    measured = run(1.0, 1.0, dofs=['1Z+', '1Z+'],
                   dims=[PSD_DIM, 'force**2/frequency'])
    with pytest.raises(ValueError, match='2 autospectra at 1Z\\+'):
        control_channels(reference(), measured, ['1Z+', '1Z+'])


# ---- quantities and units ----------------------------------------------------


def test_channels_of_different_quantities_are_not_combined():
    measured = run(1.0, 1.0, dims=[PSD_DIM, 'velocity**2/frequency'])
    with pytest.raises(ValueError,
                       match='channel 1Z\\+ is of acceleration\\*\\*2/'
                             'frequency and channel 2Z\\+ of velocity'
                             '.*cannot be combined'):
        multi_point(reference(), measured, channels=[0, 1], scale_db=0)


def test_a_channel_with_undeclared_units_beside_a_declared_one_is_refused():
    measured = run(1.0, 1.0, dims=[PSD_DIM, 'unknown'])
    with pytest.raises(ValueError, match='units of channel 2Z\\+ are not '
                                         'declared'):
        multi_point(reference(), measured, channels=[0, 1], scale_db=0)


def test_a_reference_of_another_quantity_is_refused_by_the_comparison():
    measured = run(1.0, 1.0, dims=['velocity**2/frequency'] * 2)
    with pytest.raises(ValueError, match='the specification is of '
                                         'acceleration'):
        multi_point(reference(), measured, channels=[0, 1], scale_db=0)


def test_raw_numbers_on_both_sides_are_compared_and_say_so():
    spec = reference(dim='unknown')
    measured = run(1.0, 1.0, dims=['unknown', 'unknown'])
    reading = multi_point(spec, measured, channels=[0, 1], scale_db=0)
    assert reading.notes == (RAW_NOTE,)


# ---- the bands and the scale ---------------------------------------------------


@pytest.mark.parametrize('band,reason', [
    ('abort', 'carries no abort limits'),
    ('tolerance', "is 'warning' or 'abort'"),
    ((3.0, 6.0), 'lower edge is at or below the reference'),
    ((-3.0, np.nan), 'lower edge is at or below'),
    ((-np.inf, np.inf), 'no edge on either side'),
    ((1.0,), 'a pair of dB'),
])
def test_a_band_that_cannot_be_judged_against_is_refused(band, reason):
    with pytest.raises(ValueError, match=reason):
        multi_point(reference(), run(1.0, 1.0), channels=[0, 1],
                    composite_band=band, scale_db=0)


def test_a_run_at_a_lower_level_is_scaled_once_for_every_channel():
    """Run at -6 dB, as a step-up is: detected from the data, every
    channel and the composite scaled by the same +6 dB, and said."""
    measured = run(10 ** -0.6, 10 ** -0.6, 10 ** -0.6)
    reading = multi_point(reference(), measured, channels=[0, 1, 2])
    assert reading.scale_db == 6.0
    assert reading.result['difference_db'] == pytest.approx(0.0, abs=1e-9)
    assert [c.outside_lines for c in reading.channels] == [0, 0, 0]
    assert reading.notes == (('the measurement was scaled by +6 dB, detected '
                             'from the data as a run at -6 dB'),)
    held = multi_point(reference(), measured, channels=[0, 1, 2],
                       scale_db=0)
    assert held.scale_db == 0.0 and held.notes == ()
    assert held.result['difference_db'] == pytest.approx(-6.0, abs=1e-9)


def test_the_reference_must_be_a_specification_and_the_run_a_psd():
    with pytest.raises(TypeError, match='a random specification'):
        multi_point(run(1.0), run(1.0, 1.0))
    with pytest.raises(TypeError, match='a measured PSD'):
        multi_point(reference(), reference())
