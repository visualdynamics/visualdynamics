"""The overload check: each channel of a record read against its range.

Pinned against numbers that do not depend on the implementation: the
range a chain reads to (10 V over 100 mV/g is 100 g), the decibel
(20 log10 of the range over the peak), and a worked example: 12 seconds
of the flat 0.04 g^2/Hz profile from 20 to 2,000 Hz at 5,120 samples a
second, 8.88 grms with its largest peak at 37 g. At minus 6 dB its
largest peak is 18.49 g, 2.62 dB under a 25 g range, and scaled up 6 dB
that is 37 g, past it. Through the 25 g range at full level it reads
287 samples on the rail, 0.47 %, about 24 a second, a crest factor of
2.83 and a kurtosis of 2.86, which the kurtosis bars' band of 2 to 4
passes and this check does not. The range is read from a channel table:
10 V over 400 mV/g. A table row whose Range is the column's default of
5 V is read but marked, since a blank, zero or fractional Range reads
back as 5 V too: a channel clipped at 2.5 g on a 0.5 V input over
200 mV/g is read against 25 g, 20 dB of headroom it does not have.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from visualdynamics.core.channel_table import ChannelTable
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.kurtosis import HIGH, LOW, channel_kurtosis
from visualdynamics.core.overload import (
    DEFAULT_RANGE,
    OVERLOADS,
    STEP_DB,
    channel_overload,
    headroom_rows,
    notes_for,
    range_in_unit,
    table_full_scales,
)

G = 9.80665
RATE = 5120.0


def flat_random(seconds, rate, seed, low=20.0, high=2000.0, level=0.04):
    """A Gaussian record in g whose one-sided ASD is `level` g^2/Hz from
    `low` to `high`: white noise shaped line by line in frequency."""
    n = round(seconds * rate)
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / rate)
    # white noise of unit variance has a one-sided ASD of 2/rate
    gain = np.where((f >= low) & (f <= high),
                    np.sqrt(level / (2.0 / rate)), 0.0)
    return np.fft.irfft(spectrum * gain, n)


@pytest.fixture(scope='module')
def clean():
    """The worked example's record at full level: its largest peak 37 g."""
    return flat_random(12.0, RATE, 1403)


def history(*rows, dofs=None):
    rows = np.vstack(rows) * G
    count = len(rows)
    return TimeHistory(np.arange(rows.shape[1]) / RATE, rows,
                       response_dof=dofs or [f'{k + 1}Z+'
                                             for k in range(count)],
                       ordinate_dim=['acceleration'] * count,
                       ordinate_unit=['g'] * count)


def table(nodes, ranges, sensitivities, units=None, types=None):
    count = len(nodes)
    return ChannelTable({
        'channel': list(range(1, count + 1)), 'node': list(nodes),
        'direction': ['Z+'] * count,
        'unit': units or ['g'] * count,
        'channel_type': types or ['acceleration'] * count,
        'sensitivity': list(sensitivities), 'range': list(ranges)})


@pytest.fixture(scope='module')
def run(clean):
    """Three channels: the record at minus 6 dB, the record through a
    25 g range at full level, and the record at minus 12 dB."""
    return history(clean / 2.0, np.clip(clean, -25.0, 25.0), clean / 4.0)


@pytest.fixture(scope='module')
def channels():
    """Each channel 10 V over 400 mV/g: a 25 g range, on no default."""
    return table([1, 2, 3], [10, 10, 10], [400, 400, 400])


# ---- the range from the table ------------------------------------------

def test_a_10_volt_input_on_100_millivolts_a_g_reads_to_100_g():
    assert range_in_unit(10, 100.0) == 100.0
    assert range_in_unit(5, 200.0) == 25.0


def test_a_sensitivity_that_is_not_a_number_gives_no_range():
    assert math.isnan(range_in_unit(5, float('nan')))
    assert math.isnan(range_in_unit(5, 0.0))


def test_the_tables_range_is_read_in_si(run, channels):
    scales, reasons, defaults, notes = table_full_scales(channels, run)
    assert scales == pytest.approx([25.0 * G] * 3)
    assert reasons == ['', '', '']
    assert defaults == [False] * 3 and notes == [''] * 3


def test_a_range_in_si_units_is_its_volts_over_its_sensitivity(run):
    # 10 V over 51 mV per m/s^2 is 196.08 m/s^2
    scales = table_full_scales(
        table([1, 2, 3], [10, 10, 10], [51.0] * 3, units=['m/s^2'] * 3),
        run).full_scales
    assert scales == pytest.approx([10000.0 / 51.0] * 3)


def test_each_channel_without_a_range_says_why(run):
    described = table([1, 2, 7], [5, 5, 5], [200.0, float('nan'), 200.0],
                      units=['g', 'g', 'g'])
    scales, reasons, _defaults, _notes = table_full_scales(described, run)
    assert scales[0] == pytest.approx(25.0 * G)
    assert np.isnan(scales[1:]).all()
    assert reasons == ['', 'no sensitivity in the channel table',
                       'not in the channel table']


def test_a_unit_of_another_quantity_gives_no_range(run):
    described = table([1, 2, 3], [5, 5, 5], [200.0] * 3,
                      units=['g', 'lbf', 'g'])
    scales, reasons, _defaults, _notes = table_full_scales(described, run)
    assert np.isnan(scales[1])
    assert reasons[1] == "the table's unit 'lbf' is not acceleration"


def test_a_row_with_no_unit_says_so(run):
    """A blank unit is named as blank, not as a unit of another quantity
    (the guard before the unit is converted)."""
    described = table([1, 2, 3], [10, 10, 10], [400.0] * 3,
                      units=['g', '', 'g'])
    scales, reasons, _defaults, _notes = table_full_scales(described, run)
    assert np.isnan(scales[1])
    assert reasons == ['', 'no unit in the channel table', '']


def test_a_row_with_no_type_names_the_one_channel_at_its_dof(run):
    # a blank unit too, or the table reads the type off the unit
    described = table([1, 2, 3], [5, 5, 5], [200.0] * 3,
                      units=['', '', ''], types=['', '', ''])
    assert described.types() == ['', '', '']
    assert described.identities(run) == [
        ('1Z+', 'acceleration'), ('2Z+', 'acceleration'),
        ('3Z+', 'acceleration')]
    assert table_full_scales(described, run).reasons == [
        'no unit in the channel table'] * 3


def test_a_row_with_no_type_names_no_channel_where_two_share_its_dof(clean):
    """With acceleration and force both at 1Z+, a row naming 1Z+ with no
    type names neither: there is no one channel there to take."""
    record = TimeHistory(np.arange(clean.size) / RATE,
                         np.vstack([clean, clean]) * G,
                         response_dof=['1Z+', '1Z+'],
                         ordinate_dim=['acceleration', 'force'],
                         ordinate_unit=['g', 'N'])
    # a blank unit too, or the table reads the type off the unit
    described = table([1], [10], [400.0], units=[''], types=[''])
    assert described.types() == ['']
    assert described.identities(record) == [None]
    found = table_full_scales(described, record)
    assert np.isnan(found.full_scales).all()
    assert found.reasons == ['not in the channel table'] * 2


def test_two_rows_naming_one_channel_must_agree(run):
    """Rows 2 and 4 both name 2Z+. Alike, the range is read and a note
    says which rows; different, there is no range, and the reason says
    why, rather than the first row winning unsaid."""
    alike = table([1, 2, 3, 2], [10, 10, 10, 5], [400.0, 400.0, 400.0, 200.0])
    found = table_full_scales(alike, run)
    assert found.full_scales == pytest.approx([25.0 * G] * 3)
    assert found.notes == ['', ('rows 2 and 4 of the channel table name '
                                'it, with the same range'), '']
    check = run.overload_check(alike)
    assert check.channels[1].note.startswith('rows 2 and 4')
    assert check.channels[1].describe('g', G).endswith(
        'Rows 2 and 4 of the channel table name it, with the same range.')
    differ = table([1, 2, 3, 2], [10, 10, 10, 10], [400.0] * 3 + [200.0])
    found = table_full_scales(differ, run)
    assert np.isnan(found.full_scales[1])
    assert found.reasons[1] == ('rows 2 and 4 of the channel table name it '
                                'with different ranges')
    assert not run.overload_check(differ).channels[1].has_range


def test_a_range_on_the_columns_default_is_read_and_marked(clean):
    """The Range column holds whole volts with a default of 5: 0.5 V, a
    blank and a 0 all read back as 5 V. A channel clipped at 2.5 g on a
    0.5 V input over 200 mV/g is read against 25 g with 20 dB of
    headroom, so the words say the range is the default and ask for it
    to be confirmed."""
    assert DEFAULT_RANGE == 5
    record = history(np.clip(clean, -2.5, 2.5), clean / 4.0)
    described = table([1, 2], ['0.5', ''], [200.0, 200.0])
    assert described.ranges().tolist() == [5, 5]
    found = table_full_scales(described, record)
    assert found.on_default == [True, True]
    check = record.overload_check(described)
    clipped, _quiet = check.channels
    assert clipped.headroom_db == pytest.approx(20.0, abs=0.01)
    assert clipped.flagged == [] and clipped.on_default
    assert clipped.describe('g', G) == (
        "1Z+: 20.0 dB of headroom, 14.0 dB at +6 dB. Its range is the "
        "Range column's default of 5 V, which a blank, zero or fractional "
        "Range also reads as.")
    assert check.describe() == (
        "All 2 channels inside their ranges, at +6 dB as well. 2 channels "
        "(1Z+, 2Z+) read against the Range column's default of 5 V, which a "
        "blank, zero or fractional Range also reads as: confirm they are "
        "the input's.")
    # a range given in SI is not the table's and carries no such mark
    assert not record.overload_check(25.0 * G).channels[0].on_default
    one = table([1, 2], [5, 10], [200.0, 400.0])
    assert record.overload_check(one).describe().endswith(
        "1 channel (1Z+) read against the Range column's default of 5 V, "
        "which a blank, zero or fractional Range also reads as: confirm it "
        "is the input's.")


def test_the_tables_rows_name_the_historys_channels(run, channels):
    assert channels.identities(run) == [('1Z+', 'acceleration'),
                                        ('2Z+', 'acceleration'),
                                        ('3Z+', 'acceleration')]
    other = table([9], [5], [200.0])
    assert other.identities(run) == [None]
    assert other.identities() == [('9Z+', 'acceleration')]


# ---- the worked example -------------------------------------------------

def test_the_clipped_channel_reads_the_worked_examples_numbers(run, channels):
    check = run.overload_check(channels)
    clipped = check.channels[1]
    assert clipped.label == '2Z+'
    assert clipped.rail == 287
    assert 100 * clipped.rail_share == pytest.approx(0.47, abs=0.005)
    assert clipped.rail_rate == pytest.approx(287 / 12.0)
    assert round(clipped.rail_rate) == 24
    assert clipped.crest == pytest.approx(2.83, abs=0.005)
    assert clipped.headroom_db == pytest.approx(0.0, abs=1e-9)
    assert clipped.flagged == ['rail']


def test_the_kurtosis_band_passes_the_clip_this_check_catches(run, channels):
    check = run.overload_check(channels)
    clipped = check.channels[1]
    assert clipped.statistics.kurtosis == pytest.approx(2.86, abs=0.005)
    assert LOW <= clipped.statistics.kurtosis <= HIGH
    assert 'kurtosis' not in clipped.statistics.flagged
    assert 'rail' in clipped.flagged


def test_the_kurtosis_is_the_kurtosis_bars(run, channels):
    check = run.overload_check(channels)
    bars = dict(channel_kurtosis(run))
    for channel in check.channels:
        assert channel.statistics.kurtosis == bars[channel.label]


def test_minus_6_db_reaches_its_range_at_the_next_level(run, channels):
    low = run.overload_check(channels).channels[0]
    assert low.peak / G == pytest.approx(18.49, abs=0.005)
    assert low.headroom_db == pytest.approx(20 * math.log10(25.0 / 18.4949),
                                            abs=1e-3)
    assert low.headroom_db == pytest.approx(2.62, abs=0.005)
    # 6 dB up is 1.995 times, about double, against 25 g
    assert low.next_peak / G == pytest.approx(18.4949 * 10 ** 0.3, abs=1e-3)
    assert low.next_peak / G == pytest.approx(36.90, abs=0.005)
    assert low.next_headroom_db == pytest.approx(low.headroom_db - 6.0)
    assert low.rail == 0
    assert low.flagged == ['next']


def test_a_smaller_step_holds(run, channels):
    low = run.overload_check(channels, step_db=2.0).channels[0]
    assert low.next_peak / G == pytest.approx(18.4949 * 10 ** 0.1, abs=1e-3)
    assert low.flagged == []


def test_minus_12_db_holds_the_next_level_too(run, channels):
    quiet = run.overload_check(channels).channels[2]
    assert quiet.headroom_db == pytest.approx(2.62 + 6.02, abs=0.01)
    assert quiet.flagged == []
    assert quiet.describe('g', G) == (
        '3Z+: 8.6 dB of headroom, 2.6 dB at +6 dB.')


def test_the_judgment_names_each_channel_for_what_it_is(run, channels):
    check = run.overload_check(channels)
    assert check.describe() == ('Of 3 channels: 1 on the rail (2Z+); '
                                '1 reaches its range at +6 dB (1Z+).')
    assert [channel.label for channel in check.flagged] == ['1Z+', '2Z+']
    assert check.worst.label == '2Z+'


def test_the_words_state_the_numbers_in_the_units_asked(run, channels):
    check = run.overload_check(channels)
    assert check.channels[1].describe('g', G) == (
        '2Z+: 287 samples on the rail (0.47 %, 23.9 a second), crest '
        'factor 2.83.')
    assert check.channels[0].describe('g', G) == (
        '1Z+: 2.6 dB of headroom; +6 dB puts its peak at 36.9 g against '
        'a range of 25 g.')


def test_without_the_worst_on_the_rail_the_least_headroom_is_worst(clean):
    record = history(clean / 4.0, clean / 2.0)
    check = record.overload_check(25.0 * G)
    assert check.worst.label == '2Z+'


def test_every_channel_inside_its_range_is_said_so(clean):
    check = history(clean / 4.0).overload_check(25.0 * G)
    assert check.describe() == ('All 1 channel inside their ranges, at '
                                '+6 dB as well.')


# ---- what the table can get wrong ---------------------------------------

def test_samples_beyond_the_range_say_the_table_is_wrong(clean):
    # 2 V over 100 mV/g is 20 g, and the record holds samples to 25 g
    record = history(np.clip(clean, -25.0, 25.0))
    check = record.overload_check(table([1], [2], [100.0]))
    channel = check.channels[0]
    assert channel.full_scale == pytest.approx(20.0 * G)
    assert channel.beyond == int(np.count_nonzero(
        np.abs(np.clip(clean, -25.0, 25.0)) > 1.001 * 20.0))
    assert channel.beyond > 0
    assert channel.flagged == ['beyond', 'rail']
    assert 'not the one the record was taken with' in channel.describe()
    assert check.describe().startswith('Of 1 channel: 1 beyond its range')


def test_flat_tops_inside_the_range_are_left_to_the_statistics(clean):
    # the sensor clipped at 20 g on an input that reads to 50: the
    # range held, and the flat tops are the time statistics' to judge
    record = history(np.clip(clean, -20.0, 20.0))
    channel = record.overload_check(50.0 * G).channels[0]
    assert channel.rail == 0
    assert channel.flagged == []
    assert 'at_peak' in channel.statistics.flagged


def test_the_rail_is_judged_as_the_statistics_judge_it(run, channels):
    allowed = run.overload_check(channels, rail_samples=300).channels[1]
    assert allowed.rail == 287
    assert allowed.flagged == []
    assert 'rail' not in allowed.statistics.flagged


def test_a_channel_with_no_range_is_not_judged(run):
    check = run.overload_check([25.0 * G, None, float('nan')])
    assert [channel.has_range for channel in check.channels] == [
        True, False, False]
    assert check.channels[1].reason == 'no range given'
    assert check.channels[1].flags() == {}
    assert check.describe().endswith('2 channels with no range.')
    assert notes_for(check) == {1: 'no range: no range given',
                                2: 'no range: no range given'}
    rows = headroom_rows(check)
    assert rows[0][0] == '1Z+' and np.isnan(rows[1][1])


def test_one_range_serves_every_channel(run):
    check = run.overload_check(25.0 * G)
    assert [channel.full_scale for channel in check.channels] == [
        25.0 * G] * 3


def test_some_channels_can_be_read_alone(run):
    check = run.overload_check(25.0 * G, records=[2, 0])
    assert [channel.record for channel in check.channels] == [2, 0]


@pytest.mark.parametrize('ranges, options, match', [
    ([1.0, 2.0], {}, '2 ranges for 3 channels'),
    (-1.0, {}, 'has to be above zero'),
    (25.0, {'margin': 1.5}, 'margin is a share'),
    (25.0, {'step_db': float('nan')}, 'number of dB'),
    (25.0, {'records': [3]}, 'no channel 3'),
])
def test_a_setting_that_says_nothing_is_refused(run, ranges, options, match):
    with pytest.raises(ValueError, match=match):
        run.overload_check(ranges, **options)


def test_a_dead_channel_says_why_and_is_not_judged(clean):
    dead = np.full(clean.size, np.nan)
    check = history(clean / 4.0, dead).overload_check(25.0 * G)
    assert check.channels[1].statistics is None
    assert 'two finite samples' in check.channels[1].reason
    assert check.channels[1].flagged == []


def test_a_plain_array_is_read_the_same(clean):
    clipped = np.clip(clean, -25.0, 25.0)
    channel = channel_overload(clipped, RATE, 25.0, label='clipped')
    assert channel.rail == 287
    assert channel.flagged == ['rail']
    assert channel.headroom_db == pytest.approx(0.0, abs=1e-9)
    unranged = channel_overload(clipped, RATE, None)
    assert unranged.flags() == {} and unranged.reason == 'no range given'
    with pytest.raises(ValueError, match='above zero'):
        channel_overload(clipped, RATE, 0.0)


def test_every_flag_has_a_name():
    assert list(OVERLOADS) == ['beyond', 'rail', 'next']
    assert STEP_DB == 6.0


# ---- the project ----------------------------------------------------------

def test_the_project_finds_the_table_that_describes_a_record(run, channels):
    from visualdynamics.project import Project

    project = Project()
    project.add('Run', run)
    project.add('Channels', channels)
    project.add('Other', table([9], [5], [200.0]))
    assert project.channel_table_for('Run') is None
    project.link('Run', 'Other')
    assert project.channel_table_for('Run') is None, (
        'a table naming none of its channels describes another recording')
    project.link('Run', 'Channels')
    name, found = project.channel_table_for('Run')
    assert name == 'Channels' and found is channels
    assert project.channel_table_for('Channels') is None


# ---- the view -------------------------------------------------------------

def _plots(layout):
    import pyqtgraph as pg

    return [item for item in layout.ci.items
            if isinstance(item, pg.PlotItem)]


def _words(plot):
    import pyqtgraph as pg

    return [item.toPlainText() for item in plot.items
            if isinstance(item, pg.TextItem)]


def test_the_view_draws_the_range_the_rail_and_the_bars(qt_app, run,
                                                        channels):
    import pyqtgraph as pg

    from visualdynamics.plot.overload import build_overload
    from visualdynamics.units import SI

    check = run.overload_check(channels)
    layout = pg.GraphicsLayoutWidget()
    headroom, kurtosis = build_overload(
        layout, run, check, unit_system=SI.with_units(acceleration='g'))
    trace, left, right = _plots(layout)
    # the worst channel is drawn, and its title is its verdict
    assert '2Z+: 287 samples on the rail' in trace.titleLabel.text
    rails = sorted(item.value() for item in trace.items
                   if isinstance(item, pg.InfiniteLine))
    # the range either way, in g; a clipped record's peak scaled up
    # predicts nothing, so no next-level lines
    assert rails == pytest.approx([-25.0, 25.0])
    crosses = [item for item in trace.listDataItems()
               if item.opts.get('symbol') == 'x']
    assert len(crosses) == 1 and crosses[0].getData()[1].size == 287
    assert crosses[0].name() == 'on the rail (287)'
    # a bar per channel, each flagged one with its finding in words
    assert [label for label, _v in headroom.rows] == ['1Z+', '2Z+', '3Z+']
    assert headroom.low == STEP_DB and headroom.side == 'under'
    assert sorted(_words(left)) == ['on the rail, 287',
                                    'reaches its range at +6 dB']
    # beside the kurtosis bars, which pass the clipped channel
    assert [label for label, _v in kurtosis.rows] == ['1Z+', '2Z+', '3Z+']
    assert kurtosis.beyond(dict(kurtosis.rows)['2Z+']) is None
    assert '2Z+ on the rail and inside 2 to 4' in right.titleLabel.text
    layout.close()


def test_a_channel_that_held_shows_its_peak_at_the_next_level(qt_app, run,
                                                              channels):
    import pyqtgraph as pg

    from visualdynamics.plot.overload import build_overload
    from visualdynamics.units import SI

    check = run.overload_check(channels)
    layout = pg.GraphicsLayoutWidget()
    build_overload(layout, run, check, 0,
                   unit_system=SI.with_units(acceleration='g'))
    trace = _plots(layout)[0]
    lines = {round(item.value(), 2): item.pen.style() for item in trace.items
             if isinstance(item, pg.InfiniteLine)}
    dash, dot = pg.QtCore.Qt.PenStyle.DashLine, pg.QtCore.Qt.PenStyle.DotLine
    assert lines == {25.0: dash, -25.0: dash, 36.9: dot, -36.9: dot}
    assert not [item for item in trace.listDataItems()
                if item.opts.get('symbol') == 'x']
    assert '1Z+: 2.6 dB of headroom' in trace.titleLabel.text
    layout.close()


def test_a_channel_with_no_range_says_why_on_its_row(qt_app, run):
    import pyqtgraph as pg

    from visualdynamics.plot.overload import build_overload

    check = run.overload_check(table([1, 2], [5, 5], [200.0, 200.0]))
    layout = pg.GraphicsLayoutWidget()
    headroom, _kurtosis = build_overload(layout, run, check)
    assert headroom.notes == {2: 'no range: not in the channel table'}
    layout.close()


def test_the_view_draws_to_a_file(qt_app, tmp_path, run, channels):
    from visualdynamics.plot import plot_overload

    path = tmp_path / 'overload.png'
    plot_overload(run, channels, path=str(path), show=False)
    assert path.exists() and path.stat().st_size > 2000


# ---- in the app -----------------------------------------------------------

def _overload(window, pump, record, described=None):
    name = window.add_object('Run', record)
    if described is not None:
        window.add_object('Channels', described)
        window.project.link('Run', 'Channels')
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object(name)
    pump()
    pane = window.data_pane
    if not pane.overload_action.isChecked():
        pane.overload_action.trigger()
    pump()
    return pane


def test_the_reading_is_offered_once_a_table_describes_the_record(
        window, pump, run, channels):
    """No button on a time history with no linked table, or with a
    linked table that names none of its channels: there is no range to
    read against (Brandon, 2026-10-23)."""
    window.add_object('Run', run)
    window.data_pane.waterfall_action.setChecked(False)
    window.show_object('Run')
    pump()
    pane = window.data_pane
    assert not pane.overload_action.isVisible()
    assert not pane.showing('overload')
    window.add_object('Other', table([9], [10], [400.0]))
    window.project.link('Run', 'Other')
    window.show_object('Run')
    pump()
    assert not pane.overload_action.isVisible()
    window.add_object('Channels', channels)
    window.project.link('Run', 'Channels')
    window.show_object('Run')
    pump()
    assert pane.overload_action.isVisible()
    pane.overload_action.trigger()
    pump()
    assert pane.showing('overload')
    assert pane.overload_panel.source.text() == 'Ranges from Channels'
    # and with none, the panel says how to give it one
    pane.overload_panel.show_check(None)
    assert pane.overload_panel.source.text().startswith(
        'No channel table linked')


def test_the_linked_tables_ranges_are_read(window, pump, run, channels):
    pane = _overload(window, pump, run, channels)
    panel = pane.overload_panel
    assert panel.source.text() == 'Ranges from Channels'
    assert panel.table.rowCount() == 3
    cells = [[panel.table.item(row, column).text() for column in range(3)]
             for row in range(3)]
    assert cells == [['1Z+', '2.6 dB', '0'], ['2Z+', '0.0 dB', '287'],
                     ['3Z+', '8.6 dB', '0']]
    bold = [panel.table.item(row, 0).font().bold() for row in range(3)]
    assert bold == [True, True, False]
    # the numbers are the worst channel's, the one the trace draws
    assert panel.values['channel'].text() == '2Z+'
    assert panel.values['rail'].text() == '287 (0.47 %)'
    assert panel.values['range'].text() == '25 g'
    assert panel.values['range'].toolTip() == ''
    assert panel.values['kurtosis'].text() == '2.860'
    assert panel.verdict.text() == ('Of 3 channels: 1 on the rail (2Z+); '
                                    '1 reaches its range at +6 dB (1Z+).')
    line = window.project.journal[-1]
    assert line.startswith("visualdynamics.plot.plot_overload(project['Run'], "
                           "project['Channels'], None, step_db=6.0")


def test_a_step_set_in_the_panel_rejudges_the_next_level(window, pump, run,
                                                         channels):
    pane = _overload(window, pump, run, channels)
    panel = pane.overload_panel
    panel.step_box.setValue(2.0)
    pump()
    assert not panel.table.item(0, 0).font().bold()
    assert panel.verdict.text() == 'Of 3 channels: 1 on the rail (2Z+).'
    assert 'step_db=2.0' in window.project.journal[-1]
    window.render_current()
    pump()
    assert panel.step_box.value() == 2.0, 'the step is sticky'


def test_a_row_clicked_draws_that_channels_trace(window, pump, run,
                                                 channels):
    pane = _overload(window, pump, run, channels)
    panel = pane.overload_panel
    panel.table.cellClicked.emit(0, 0)
    pump()
    assert panel.values['channel'].text() == '1Z+'
    assert panel.values['next_peak'].text() == '36.9 g'
    trace = _plots(pane.graphics)[0]
    assert '1Z+: 2.6 dB of headroom' in trace.titleLabel.text
    assert "project['Channels'], 0, step_db" in window.project.journal[-1]


# ---- dead, not finite, empty, a late clock and the worker -----------------

def test_a_channel_of_zeros_is_not_read_and_never_passes(clean):
    """Zero throughout has infinite headroom, which would read as a
    channel well inside its range: it is not read, and says why."""
    check = history(clean / 4.0, np.zeros(clean.size)).overload_check(
        25.0 * G)
    dead = check.channels[1]
    assert dead.statistics is None and not dead.has_range
    assert dead.flagged == [] and dead.reason.startswith('zero throughout')
    assert dead.describe() == ('2Z+: not read (zero throughout: a dead '
                               'channel has no peak to read against its '
                               'range).')
    assert check.describe() == (
        'All 1 channel inside their ranges, at +6 dB as well. Not read: '
        '2Z+, zero throughout: a dead channel has no peak to read against '
        'its range.')
    assert notes_for(check) == {1: 'not read: zero throughout: a dead '
                                   'channel has no peak to read against its '
                                   'range'}
    alone = channel_overload(np.zeros(64), RATE, 25.0)
    assert alone.statistics is None and alone.flags() == {}


def test_one_sample_not_finite_is_said_and_left_out(clean):
    holed = clean / 4.0
    holed[1000] = np.nan
    check = history(holed).overload_check(25.0 * G)
    channel, = check.channels
    assert channel.statistics.nonfinite == 1 and channel.has_range
    assert '1 sample is not finite and left out' in channel.cautions()
    assert channel.describe().endswith(' 1 sample is not finite and left '
                                       'out.')
    assert check.describe().endswith(
        'Samples not finite, left out: 1Z+ 1.')


def test_an_empty_selection_is_refused(run):
    with pytest.raises(ValueError, match='no channel asked for'):
        run.overload_check(25.0 * G, records=[])


def test_a_record_whose_clock_starts_late_reads_the_same(run):
    late = TimeHistory(np.asarray(run.abscissa) + 3.25, run.ordinate,
                       response_dof=list(run.response_dof),
                       ordinate_dim=list(run.ordinate_dim),
                       ordinate_unit=['g'] * 3)
    early, later = run.overload_check(25.0 * G), late.overload_check(25.0 * G)
    assert later.describe() == early.describe()
    assert [c.headroom_db for c in later.channels] == pytest.approx(
        [c.headroom_db for c in early.channels])
    assert later.channels[1].rail == early.channels[1].rail == 287


def test_a_long_record_is_read_on_the_worker(window, pump, run, channels,
                                             monkeypatch):
    """Past `OVERLOAD_INLINE_SAMPLES` the reading runs through
    `_run_long` with the bar up, and is shown as an inline one is."""
    from visualdynamics.gui import main_window as module

    ran = []
    original = window._run_long

    def spy(status, verb, *args, **kwargs):
        ran.append(status)
        return original(status, verb, *args, **kwargs)

    monkeypatch.setattr(window, '_run_long', spy)
    monkeypatch.setattr(module, 'OVERLOAD_INLINE_SAMPLES', 1000)
    pane = _overload(window, pump, run, channels)
    assert ran and ran[-1] == 'Reading Run against its ranges…'
    assert pane.overload_panel.verdict.text() == (
        'Of 3 channels: 1 on the rail (2Z+); 1 reaches its range at +6 dB '
        '(1Z+).')
