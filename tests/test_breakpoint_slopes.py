"""A random specification written by slope, and its table read by segment.

Many written tables give a level, a slope in dB per octave and the next
frequency, and leave the level there to the reader. The slope is an
exponent n = N / 3.0103, the level it reaches is W1 (f2 / f1)**n, and
each segment's area has a closed form with a logarithmic case at
n = -1 (minus 3.0103 dB per octave). The hand values below are worked
on a workmanship-style profile: 0.01 g²/Hz at 20 Hz, plus 3 dB per
octave to 80 Hz, flat at 0.04 to 500 Hz, minus 3 dB per octave to
2000 Hz. Its areas are 1.50, 16.80 and 27.73 g² and its running rms
1.22, 4.28 and 6.78 g, every one checked by hand:

    rise   0.04 x 80 / 2 x (1 - (20/80)**2)  = 1.6 x 0.9375 = 1.5
    flat   0.04 x (500 - 80)                 = 16.8
    fall   0.04 x 500 x ln 4                 = 27.7259
    rms    sqrt(46.0259)                     = 6.7842
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path
from conftest import select_objects as _select
from PySide6.QtCore import Qt

import visualdynamics
from visualdynamics.core.author import SpecificationDraft
from visualdynamics.core.breakpoints import (
    OCTAVE_DB,
    level_at_slope,
    levels_from_slopes,
    levels_holding_slopes,
    levels_with_slope,
    read_breakpoints,
    running_rms,
    segment_areas,
    segment_slopes,
    slope_db,
    slope_exponent,
)
from visualdynamics.core.compliance import specification_rms

G = 9.80665
#: the profile's breakpoints and levels, in Hz and g²/Hz, with its
#: slopes exactly an exponent of one (3.0103 dB per octave)
F = [20.0, 80.0, 500.0, 2000.0]
W = [0.01, 0.04, 0.04, 0.01]
AREAS = [1.5, 16.8, 0.04 * 500.0 * np.log(4.0)]
RUNNING = [np.sqrt(1.5), np.sqrt(18.3), np.sqrt(18.3 + AREAS[2])]


def _sheet(frequencies=None, levels=None, channels=('1Z+',)):
    """A one-or-more-channel draft in SI, levels given in g²/Hz."""
    frequencies = F if frequencies is None else frequencies
    levels = W if levels is None else levels
    return SpecificationDraft(
        list(channels), ['acceleration'] * len(channels), list(frequencies),
        [[v * G * G for v in levels] for _ in channels],
        {(i, j): (0.0, 0.0) for i in range(len(channels))
         for j in range(i + 1, len(channels))})


# ---- the core: slopes, levels, areas -------------------------------------------


def test_a_slope_is_an_exponent_and_the_documents_three_is_not_quite_one():
    assert OCTAVE_DB == pytest.approx(3.0103, abs=1e-4)
    assert float(slope_exponent(3.0)) == pytest.approx(0.99658, abs=1e-5)
    assert float(slope_exponent(-6.0)) == pytest.approx(-1.99316, abs=1e-5)
    assert float(slope_exponent(OCTAVE_DB)) == 1.0
    assert float(slope_db(slope_exponent(4.5))) == pytest.approx(4.5)
    # the unprinted levels: two octaves at the documents' 3 dB reach
    # 0.0398 and 0.01005, the table's 0.04 and 0.01 to half a percent
    assert level_at_slope(20, 0.01, 80, 3) == pytest.approx(0.039811, abs=1e-6)
    assert level_at_slope(500, 0.04, 2000, -3) == pytest.approx(0.0100475, abs=1e-7)
    assert level_at_slope(20, 0.01, 80, OCTAVE_DB) == pytest.approx(0.04, rel=1e-14)
    # followed downward too, and flat is flat
    assert level_at_slope(80, 0.04, 20, OCTAVE_DB) == pytest.approx(0.01, rel=1e-14)
    assert level_at_slope(80, 0.04, 500, 0.0) == 0.04
    with pytest.raises(ValueError, match='positive'):
        level_at_slope(0, 0.01, 80, 3)
    with pytest.raises(ValueError, match='level'):
        level_at_slope(20, -0.01, 80, 3)


def test_the_segment_areas_are_the_closed_form_by_hand():
    areas = segment_areas(F, W)
    assert areas == pytest.approx(AREAS, rel=1e-12)
    assert areas[2] == pytest.approx(27.7259, abs=1e-4)
    reading = read_breakpoints(F, W)
    assert reading.running == pytest.approx(RUNNING, rel=1e-12)
    assert reading.rms == pytest.approx(6.7842, abs=1e-4)
    assert reading.total_area == pytest.approx(46.0259, abs=1e-4)
    assert reading.slopes == pytest.approx([OCTAVE_DB, 0.0, -OCTAVE_DB])
    # the falling segment, short on log axes, is 60 percent of the area
    assert reading.shares == pytest.approx([0.0326, 0.3650, 0.6024], abs=1e-4)
    assert reading.shares.sum() == pytest.approx(1.0)
    # the floor's shorthand, one rectangle: 79.2 g², 8.90 grms
    flat = read_breakpoints([20.0, 2000.0], [0.04, 0.04])
    assert flat.areas == pytest.approx([79.2])
    assert flat.rms == pytest.approx(8.8994, abs=1e-4)


def test_minus_one_takes_the_logarithm_and_minus_three_db_the_general_form():
    """An exponent of exactly minus one divides the general form by zero;
    typed as minus 3.0103 dB per octave the levels make that exponent to
    the last bit and the area is the logarithm. Typed as the documents'
    minus 3 the exponent is -0.9966 and the general form is right."""
    exact = levels_from_slopes([500.0, 2000.0], 0.04, [-OCTAVE_DB])
    assert exact[1] == pytest.approx(0.01, rel=1e-14)
    assert segment_areas([500.0, 2000.0], exact)[0] == pytest.approx(
        0.04 * 500.0 * np.log(4.0), rel=1e-12)
    typed = levels_from_slopes([500.0, 2000.0], 0.04, [-3.0])
    n = -3.0 / OCTAVE_DB
    by_hand = 0.04 * 500.0 * ((2000.0 / 500.0) ** (n + 1.0) - 1.0) / (n + 1.0)
    assert segment_areas([500.0, 2000.0], typed)[0] == pytest.approx(by_hand, rel=1e-9)
    assert by_hand == pytest.approx(27.7917, abs=1e-4)
    # and the trapezoid is wrong there: 37.5 against 27.7
    assert 0.5 * (0.04 + 0.01) * 1500.0 == pytest.approx(37.5)


def test_a_minus_one_segment_followed_by_another_takes_the_logarithm_whole():
    """The logarithmic case has two homes in `log_log_areas`: the whole
    segment, accumulated into the running total at the breakpoints, and
    the partial segment a band ends inside. A minus-one segment that is
    the last of its table only ever reaches the second, so the table
    here carries one on to a flat segment after it."""
    f = [*F, 4000.0]
    w = [*W, 0.01]
    assert segment_slopes(f, w)[2] == pytest.approx(-OCTAVE_DB, rel=1e-13)
    areas = segment_areas(f, w)
    assert areas == pytest.approx([*AREAS, 0.01 * 2000.0], rel=1e-12)
    # the running rms at 2000 Hz is the cumulative area through the
    # minus-one segment whole, and past it the flat segment adds 20 g²
    tail = running_rms(f, w, [2000.0, 3000.0, 4000.0])
    assert tail ** 2 == pytest.approx(
        [sum(AREAS), sum(AREAS) + 10.0, sum(AREAS) + 20.0], rel=1e-12)
    reading = read_breakpoints(f, w)
    assert reading.running[2] == pytest.approx(RUNNING[2], rel=1e-12)
    assert reading.rms ** 2 == pytest.approx(sum(AREAS) + 20.0, rel=1e-12)
    # and from slopes: minus 3.0103 into 2000 Hz, then flat to 4000 Hz
    typed = levels_from_slopes(f, 0.01, [OCTAVE_DB, 0.0, -OCTAVE_DB, 0.0])
    assert segment_areas(f, typed) == pytest.approx(areas, rel=1e-12)


def test_the_cited_worked_example_reads_the_same():
    """The worked example of the reference in `core.breakpoints`
    (Irvine, Integration of the Power Spectral Density Function, Rev B):
    10 Hz at 0.002, 100 and 1000 Hz at 0.04, 2000 Hz at 0.02 g²/Hz. It
    prints areas of 1.726, 36.000 and 27.726 g² and 8.09 grms. Its first
    exponent is rounded there to 1.3 from log(20)/log(10) = 1.30103,
    which is the whole of the difference from the 1.7296 here."""
    f = [10.0, 100.0, 1000.0, 2000.0]
    w = [0.002, 0.04, 0.04, 0.02]
    areas = segment_areas(f, w)
    assert areas == pytest.approx([1.7296, 36.0, 27.7259], abs=1e-4)
    rounded = 0.002 / 10.0 ** 1.3 / 2.3 * (100.0 ** 2.3 - 10.0 ** 2.3)
    assert rounded == pytest.approx(1.726, abs=5e-4), 'its own figure, at n = 1.3'
    assert read_breakpoints(f, w).rms == pytest.approx(8.09, abs=5e-3)
    # its Appendix A: 0.04 at 350 Hz, minus 3 dB per octave to 2000 Hz
    # is 0.007, and plus 3 dB per octave down from 80 to 20 Hz is 0.01
    assert level_at_slope(350.0, 0.04, 2000.0, -3.0) == pytest.approx(0.007, abs=5e-4)
    assert level_at_slope(80.0, 0.04, 20.0, 3.0) == pytest.approx(0.01, abs=5e-4)


def test_the_running_rms_climbs_from_zero_to_the_total():
    at = [10.0, 20.0, 80.0, 500.0, 2000.0, 3000.0]
    assert running_rms(F, W, at) == pytest.approx([0.0, 0.0, *RUNNING,
                                                   RUNNING[-1]], rel=1e-12)
    assert running_rms(F, W) == pytest.approx([0.0, *RUNNING], rel=1e-12)
    fine = np.geomspace(20.0, 2000.0, 500)
    climb = running_rms(F, W, fine)
    assert np.all(np.diff(climb) >= 0.0), 'an area only accumulates'
    # inside a segment it is the partial area by the same power law
    mid = running_rms(F, W, [40.0])[0]
    assert mid ** 2 == pytest.approx(0.01 / 20.0 * (40.0 ** 2 - 20.0 ** 2) / 2.0,
                                     rel=1e-12)


def test_slopes_and_levels_convert_both_ways():
    slopes = segment_slopes(F, W)
    assert levels_from_slopes(F, W[0], slopes) == pytest.approx(W, rel=1e-13)
    assert levels_from_slopes(F, 0.01, [3.0, 0.0, -3.0]) == pytest.approx(
        [0.01, 0.039811, 0.039811, 0.0100002], abs=1e-6)
    with pytest.raises(ValueError, match='2 slopes for 3 segments'):
        levels_from_slopes(F, 0.01, [3.0, 0.0])
    with pytest.raises(ValueError, match='increase'):
        segment_slopes([20.0, 20.0, 80.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match='positive and finite'):
        segment_slopes(F, [0.01, 0.0, 0.04, 0.01])
    with pytest.raises(ValueError, match='at least two'):
        read_breakpoints([20.0], [0.01])


def test_a_typed_slope_holds_every_slope_after_it():
    after = levels_with_slope(F, W, 0, 2 * OCTAVE_DB)
    assert after[0] == W[0], 'the levels before it, exactly'
    assert after[1] == pytest.approx(0.16, rel=1e-13), 'two octaves at +6.02'
    assert segment_slopes(F, after)[1:] == pytest.approx(segment_slopes(F, W)[1:])
    assert after == pytest.approx([0.01, 0.16, 0.16, 0.04], rel=1e-13)
    middle = levels_with_slope(F, W, 2, -2 * OCTAVE_DB)
    assert list(middle[:3]) == W[:3]
    assert middle[3] == pytest.approx(0.0025, rel=1e-13)
    with pytest.raises(ValueError, match='no segment 4'):
        levels_with_slope(F, W, 3, 3.0)


def test_a_typed_level_holds_every_slope_after_it():
    after = levels_holding_slopes(F, W, 1, 0.08)
    assert after == pytest.approx([0.01, 0.08, 0.08, 0.02], rel=1e-13)
    assert after[1] == 0.08, 'exactly as typed'
    assert segment_slopes(F, after)[1:] == pytest.approx(segment_slopes(F, W)[1:])
    whole = levels_holding_slopes(F, W, 0, 0.02)
    assert whole == pytest.approx([0.02, 0.08, 0.08, 0.02], rel=1e-13)
    with pytest.raises(ValueError, match='no breakpoint 5'):
        levels_holding_slopes(F, W, 4, 0.1)
    with pytest.raises(ValueError, match='positive'):
        levels_holding_slopes(F, W, 1, -1.0)


# ---- the draft: the entry, and a breakpoint by slope ----------------------------


def test_a_table_entered_by_slope_is_the_table():
    draft = _sheet([20.0, 80.0], [0.01, 0.04])
    draft = draft.with_slope_breakpoint(500.0, 0.0)
    draft = draft.with_slope_breakpoint(2000.0, -OCTAVE_DB)
    assert draft.frequencies == F
    assert [v / G ** 2 for v in draft.levels[0]] == pytest.approx(W, rel=1e-13)
    assert draft.slopes_db()[0] == pytest.approx([OCTAVE_DB, 0.0, -OCTAVE_DB])
    reading = draft.reading('1Z+')
    assert reading.areas / G ** 2 == pytest.approx(AREAS, rel=1e-12)
    assert reading.rms / G == pytest.approx(6.7842, abs=1e-4)
    # the specification made from it carries the same rms, by the
    # same closed form
    assert specification_rms(draft.make()) / G == pytest.approx(reading.rms / G,
                                                                rel=1e-12)
    # three segments, three bands each
    assert len(draft.bands[0]['warning']) == 3
    # the documents' 3 dB rather than 3.0103: 0.01005 at the top
    typed = (_sheet([20.0, 80.0], [0.01, 0.04])
             .with_slope_breakpoint(500.0, 0.0)
             .with_slope_breakpoint(2000.0, -3.0))
    assert typed.levels[0][-1] / G ** 2 == pytest.approx(0.0100475, abs=1e-7)


def test_a_slope_breakpoint_inside_a_segment_splits_it_and_holds_the_rest():
    draft = _sheet()
    split = draft.with_slope_breakpoint(1000.0, -2 * OCTAVE_DB)
    assert split.frequencies == [20.0, 80.0, 500.0, 1000.0, 2000.0]
    # the lower half at the typed slope: 0.04 falls 6.02 dB in an octave
    assert split.levels[0][3] / G ** 2 == pytest.approx(0.01, rel=1e-13)
    # the upper half keeps the slope the segment had, -3.0103
    assert split.slopes_db()[0][3] == pytest.approx(-OCTAVE_DB)
    assert split.levels[0][4] / G ** 2 == pytest.approx(0.005, rel=1e-13)
    assert len(split.bands[0]['abort']) == 4, 'both halves wear its band'
    # at a frequency already a breakpoint, the slope into it is typed
    again = draft.with_slope_breakpoint(80.0, 2 * OCTAVE_DB)
    assert again.frequencies == F
    assert again.levels[0][1] / G ** 2 == pytest.approx(0.16, rel=1e-13)
    with pytest.raises(ValueError, match='not above the first breakpoint'):
        draft.with_slope_breakpoint(20.0, 3.0)
    with pytest.raises(ValueError, match='not above the first breakpoint'):
        draft.with_slope_breakpoint(10.0, 3.0)


def test_the_channels_named_take_the_slope_and_the_others_the_power_law():
    draft = _sheet(channels=('1X+', '1Y+'))
    out = draft.with_slope_breakpoint(4000.0, -2 * OCTAVE_DB, channels=['1X+'])
    assert out.levels[0][-1] / G ** 2 == pytest.approx(0.0025, rel=1e-13)
    assert out.levels[1][-1] == out.levels[1][-2], 'past the end: flat'
    sloped = draft.with_slope(0, 0.0, channels=['1Y+'])
    assert sloped.levels[0] == draft.levels[0]
    assert sloped.levels[1][1] == pytest.approx(draft.levels[1][0])
    with pytest.raises(ValueError, match="no channel '9Z\\+'"):
        draft.with_slope(0, 3.0, channels=['9Z+'])


def test_the_entry_decides_what_a_typed_level_holds():
    draft = _sheet()
    assert draft.entry == 'levels'
    alone = draft.with_level(1, 0.08 * G * G)
    assert [v / G ** 2 for v in alone.levels[0]] == pytest.approx(
        [0.01, 0.08, 0.04, 0.01])
    by_slope = draft.with_entry('slopes')
    assert by_slope.entry == 'slopes' and draft.entry == 'levels'
    assert by_slope.levels == draft.levels, 'the requirement is the same'
    held = by_slope.with_level(1, 0.08 * G * G)
    assert [v / G ** 2 for v in held.levels[0]] == pytest.approx(
        [0.01, 0.08, 0.08, 0.02])
    assert held.entry == 'slopes', 'an operation keeps the entry'
    assert np.allclose(np.real(by_slope.make().ordinate),
                       np.real(draft.make().ordinate)), \
        'the entry changes nothing in the specification made'
    with pytest.raises(ValueError, match="'levels' or 'slopes'"):
        draft.with_entry('decibels')
    with pytest.raises(ValueError, match='no breakpoint 9'):
        draft.with_level(8, 1.0)
    with pytest.raises(ValueError, match='no channel 3'):
        draft.reading(2)


def test_the_entry_rides_the_dict_form():
    draft = _sheet().with_entry('slopes')
    data = draft.as_dict()
    assert data['entry'] == 'slopes'
    assert SpecificationDraft.from_dict(data) == draft
    del data['entry']
    assert SpecificationDraft.from_dict(data).entry == 'levels', \
        'a record written before the entry existed reads as levels'


# ---- the view ----------------------------------------------------------------------


def _layout(qt_app):
    import pyqtgraph as pg

    widget = pg.GraphicsLayoutWidget()
    widget.resize(900, 700)
    return widget


def test_the_view_writes_each_segment_and_dashes_the_running_rms(qt_app):
    from visualdynamics.plot.breakpoints import build_breakpoints

    widget = _layout(qt_app)
    reading = _sheet().reading(0)
    plot, running = build_breakpoints(widget, reading, 'acceleration', name='1Z+')
    labels = [item for item in plot.items if getattr(item, 'is_segment_label', False)]
    texts = [label.textItem.toPlainText() for label in labels]
    assert len(texts) == 3
    assert texts[0].startswith('+3.0 dB/oct') and '1.5 g' in texts[0]
    assert texts[1].startswith('flat') and '16.8 g' in texts[1]
    assert texts[2].startswith('−3.0 dB/oct') and '27.7 g' in texts[2]
    assert '(60 %)' in texts[2], 'its share of the whole'
    # placed in log coordinates, over the middle of each segment
    assert labels[1].pos().x() == pytest.approx(np.log10(np.sqrt(80.0 * 500.0)))
    assert labels[1].pos().y() == pytest.approx(np.log10(0.04))
    curve = next(item for item in running.listDataItems()
                 if getattr(item, 'is_running_rms', False))
    assert curve.opts['pen'].style() == Qt.PenStyle.DashLine
    x, y = curve.getOriginalDataset()
    assert x[0] == 20.0 and x[-1] == 2000.0
    assert y[0] == 0.0 and y[-1] == pytest.approx(6.7842, abs=1e-4), 'in g'
    spec_pen = plot.listDataItems()[0].opts['pen']
    assert spec_pen.style() == Qt.PenStyle.SolidLine, 'told apart by style'
    assert 'dashed' in running.titleLabel.text
    assert '6.784 g at 2000 Hz' in running.titleLabel.text
    assert '6.784 g rms' in plot.titleLabel.text
    assert running.getViewBox().state['logMode'][0], 'the frequency axis shared'
    assert running.getViewBox().linkedView(0) is plot.getViewBox()


def test_a_long_table_is_read_but_not_labeled(qt_app):
    from visualdynamics.plot.breakpoints import MOST_LABELS, build_breakpoints

    f = np.linspace(20.0, 2000.0, MOST_LABELS + 10)
    reading = read_breakpoints(f, np.full(f.shape, 0.04 * G * G))
    plot, running = build_breakpoints(_layout(qt_app), reading, 'acceleration')
    assert not [item for item in plot.items
                if getattr(item, 'is_segment_label', False)]
    assert 'too many to label' in plot.titleLabel.text
    assert '8.899 g at 2000 Hz' in running.titleLabel.text


def test_the_readout_is_the_reading_in_words():
    from visualdynamics.plot.breakpoints import readout

    rows = readout(_sheet().reading(0), 'acceleration')
    assert [row['segment'] for row in rows] == [
        '20–80 Hz', '80–500 Hz', '500–2000 Hz', 'total']
    assert [row['slope'] for row in rows[:3]] == [
        '+3.0 dB/oct', 'flat', '−3.0 dB/oct']
    assert [row['area'] for row in rows] == [
        '1.5 g²', '16.8 g²', '27.73 g²', '46.03 g²']
    assert [row['running'] for row in rows] == [
        '1.225 g', '4.278 g', '6.784 g', '6.784 g rms']
    assert [row['share'] for row in rows] == ['3 %', '37 %', '60 %', '100 %']


def test_the_standalone_plot_writes_a_file_from_a_sheet_or_an_object(tmp_path, qt_app):
    from visualdynamics.plot import plot_breakpoints

    draft = _sheet(channels=('1X+', '1Y+'))
    path = plot_breakpoints(draft, channel='1Y+', show=False,
                            path=tmp_path / 'sheet.png')
    assert (tmp_path / 'sheet.png').stat().st_size > 2000 and str(path).endswith('.png')
    plot_breakpoints(draft.make(), show=False, path=tmp_path / 'object.png')
    assert (tmp_path / 'object.png').stat().st_size > 2000
    with pytest.raises(ValueError, match="no channel '9Z\\+'"):
        plot_breakpoints(draft, channel='9Z+', show=False,
                         path=tmp_path / 'none.png')


def test_a_density_per_bin_is_refused_rather_than_read_as_a_power_law(tmp_path, qt_app):
    """A controller's target on its lines is a density per line,
    integrated as bins (`interpolation` 'bin'). Read as a power law
    between those lines the areas add to an rms the object does not
    have. As bins the profile on 20 Hz lines holds the curve's own
    6.784 g; drawn as a power law through those lines it would read
    6.775 g."""
    from visualdynamics.plot import plot_breakpoints

    lines = _sheet().to_lines(20.0)
    assert lines.form == 'lines'
    binned = lines.make("a controller's target")
    assert binned.interpolation == 'bin'
    assert specification_rms(binned) / G == pytest.approx(6.7842, abs=1e-4)
    assert read_breakpoints(lines.frequencies, lines.levels[0]).rms / G == \
        pytest.approx(6.7746, abs=1e-4), 'what the view would have drawn'
    with pytest.raises(ValueError, match='density per line'):
        plot_breakpoints(binned, show=False, path=tmp_path / 'binned.png')
    with pytest.raises(ValueError, match='breakpoint table'):
        lines.reading(0)
    # its breakpoints read as the table they are, on lines fine enough
    # that each segment has whole lines to read its power law from
    table = SpecificationDraft.from_specification(
        _sheet().to_lines(5.0).make('finer')).to_breakpoints()
    assert table.frequencies == F
    assert table.reading(0).rms / G == pytest.approx(6.7842, abs=1e-4)
    # an object written at breakpoints on an even grid is still a power
    # law between its points, and its areas add to its own rms
    even = lines._copy(form='breakpoints').make('breakpoints on an even grid')
    assert even.interpolation == 'log_log'
    plot_breakpoints(even, show=False, path=tmp_path / 'even.png')
    assert (tmp_path / 'even.png').stat().st_size > 2000
    assert specification_rms(even) / G == pytest.approx(6.7746, abs=1e-4)


# ---- the sheet in the window ----------------------------------------------------

CONTROLS = ['101X+', '101Y+', '101Z+', '102X+', '102Y+', '102Z+']


def _populate(window, pump):
    rng = np.random.default_rng(1)
    geometry = visualdynamics.Geometry(np.arange(101, 106),
                                       rng.uniform(-1.0, 1.0, (5, 3)),
                                       length_unit='m')
    window.add_object('Plate', geometry)
    window.add_object('Channels', visualdynamics.ChannelTable({
        'channel': list(range(1, 7)),
        'node': [dof[:3] for dof in CONTROLS],
        'direction': [dof[3:] for dof in CONTROLS],
        'unit': ['g'] * 6, 'control': ['True'] * 6}))
    window.project.link('Plate', 'Channels')
    pump()


def _open(window, pump):
    _populate(window, pump)
    _select(window, pump, 'Channels')
    window.author_action.setChecked(True)
    window._author_toggled(True)
    pump()
    pump()
    return window.data_pane.author_panel


def _header(view, column):
    return view.model().headerData(column, Qt.Orientation.Horizontal,
                                   Qt.ItemDataRole.DisplayRole)


def _text(view, row, column):
    return view.model().data(view.model().index(row, column),
                             Qt.ItemDataRole.DisplayRole)


def _type(view, row, column, text):
    model = view.model()
    return model.setData(model.index(row, column), text, Qt.ItemDataRole.EditRole)


def test_slopes_put_a_slope_beside_each_level_and_write_nothing(window, pump):
    panel = _open(window, pump)
    key = window._author_door[1]
    assert panel.levels_button.isChecked() and not panel.slopes_button.isChecked()
    assert panel.points.model().columnCount() == 7
    before = window.project['Channels Specification']
    journal = len(window.project.journal)
    panel.slopes_button.click()
    pump()
    assert window._drafts[key].entry == 'slopes'
    assert panel.slopes_button.isChecked()
    assert panel.points.model().columnCount() == 13, 'a level and a slope a channel'
    assert _header(panel.points, 2) == '101X+\ndB/oct'
    assert _text(panel.points, 0, 2) == '', 'nothing slopes into the first'
    assert _text(panel.points, 1, 2) == '0', 'the starter is flat'
    assert window.project['Channels Specification'] is before, \
        'the entry is how the sheet is typed, not a change to the object'
    assert len(window.project.journal) == journal
    panel.levels_button.click()
    pump()
    assert window._drafts[key].entry == 'levels'
    assert panel.points.model().columnCount() == 7


def test_a_typed_slope_computes_the_level_and_lands_on_the_object(window, pump):
    panel = _open(window, pump)
    key = window._author_door[1]
    panel.slopes_button.click()
    pump()
    first = window._drafts[key].levels[0][0]
    # 20 to 2000 Hz is log2(100) octaves; at minus 3.0103 dB an octave
    # the level falls by the frequency ratio
    assert _type(panel.points, 1, 2, str(-OCTAVE_DB))
    pump()
    draft = window._drafts[key]
    assert draft.levels[0][1] == pytest.approx(first / 100.0, rel=1e-12)
    assert draft.levels[1] == window._drafts[key].levels[1], 'one channel'
    made = window.project['Channels Specification']
    assert np.real(made.ordinate[0][-1]) == pytest.approx(first / 100.0, rel=1e-12)
    assert _text(panel.points, 1, 1) == f'{float(window.unit_system.from_si(first / 100.0, "acceleration**2/frequency")):.6g}', \
        'the level the slope reaches, shown beside it'
    # the first row takes no slope, and says why
    assert not _type(panel.points, 0, 2, '3')
    assert 'no slope into it' in panel.problem.text()
    # a slope into a later row runs from the row above it, and the
    # rows above keep their levels
    panel.points.setCurrentIndex(panel.points.model().index(1, 0))
    panel.add_button.click()
    pump()
    assert window._drafts[key].frequencies == [20.0, 2000.0, 4000.0]
    assert _type(panel.points, 2, 2, str(-2 * OCTAVE_DB))
    pump()
    draft = window._drafts[key]
    assert draft.levels[0][:2] == pytest.approx([first, first / 100.0], rel=1e-12)
    assert draft.levels[0][2] == pytest.approx(first / 400.0, rel=1e-12)


def test_a_level_typed_by_slope_holds_the_slopes_after_it(window, pump):
    panel = _open(window, pump)
    key = window._author_door[1]
    panel.slopes_button.click()
    pump()
    assert _type(panel.points, 1, 2, '3')
    pump()
    slope = window._drafts[key].slopes_db()[0][0]
    assert slope == pytest.approx(3.0)
    assert _type(panel.points, 0, 1, '0.5')
    pump()
    draft = window._drafts[key]
    assert draft.levels[0][0] == pytest.approx(
        float(window.unit_system.to_si(0.5, 'acceleration**2/frequency')))
    assert draft.slopes_db()[0][0] == pytest.approx(3.0), 'held'
    # a frequency moved within its neighbors keeps the slope too
    assert _type(panel.points, 1, 0, '1000')
    pump()
    draft = window._drafts[key]
    assert draft.frequencies == [20.0, 1000.0]
    assert draft.slopes_db()[0][0] == pytest.approx(3.0)


def test_scale_selected_under_slopes_scales_levels_and_passes_over_slopes(window, pump):
    """Under Slopes the grid holds a level and a slope a channel. Scale
    selected scales the selected levels as they stand, so the slopes
    into and out of a scaled row follow from the levels, and a slope
    cell in the selection is not a level and is passed over."""
    panel = _open(window, pump)
    key = window._author_door[1]
    panel.slopes_button.click()
    pump()
    assert _type(panel.points, 1, 2, '-3')
    pump()
    before = [list(row) for row in window._drafts[key].levels]
    slopes = window._drafts[key].slopes_db()
    # every cell, slopes among them: every level up by 3 dB, every slope
    # held, and a slope is not multiplied as though it were a level
    panel.points.selectAll()
    panel.scale_box.setValue(3.0)
    panel.scale_button.click()
    pump()
    draft = window._drafts[key]
    for k, row in enumerate(before):
        assert draft.levels[k] == pytest.approx([v * 10 ** 0.3 for v in row],
                                                rel=1e-12)
    assert np.asarray(draft.slopes_db()) == pytest.approx(np.asarray(slopes), abs=1e-12)
    assert draft.slopes_db()[0][0] == pytest.approx(-3.0)
    made = window.project['Channels Specification']
    assert np.real(made.ordinate[0][1]) == pytest.approx(before[0][1] * 10 ** 0.3,
                                                         rel=1e-12)
    # one breakpoint's row: its levels down 6 dB, and the slope into
    # it, shown beside them, steepens by 6 dB over log2(100) octaves
    scaled = [list(row) for row in draft.levels]
    panel.points.clearSelection()
    panel.points.selectRow(1)
    panel.scale_box.setValue(-6.0)
    panel.scale_button.click()
    pump()
    draft = window._drafts[key]
    assert draft.levels[0] == pytest.approx(
        [scaled[0][0], scaled[0][1] * 10 ** -0.6], rel=1e-12)
    steeper = -3.0 - 6.0 / np.log2(100.0)
    assert draft.slopes_db()[0][0] == pytest.approx(steeper, rel=1e-9)
    assert float(_text(panel.points, 1, 2)) == pytest.approx(steeper, rel=1e-4)
    # a slope column alone holds no level to scale
    panel.points.clearSelection()
    panel.points.selectColumn(2)
    panel.scale_box.setValue(6.0)
    panel.scale_button.click()
    pump()
    assert 'select the levels' in panel.problem.text()
    assert window._drafts[key].levels == draft.levels, 'nothing scaled'


def test_areas_write_on_the_plot_draw_the_running_rms_and_read_out(window, pump):
    import pyqtgraph as pg

    panel = _open(window, pump)
    key = window._author_door[1]
    assert _type(panel.points, 0, 1, '0.04')
    pump()
    assert not panel.reading.isVisibleTo(panel)

    def plots():
        return [item for item in window.data_pane.graphics.ci.items
                if isinstance(item, pg.PlotItem)]

    alone = len(plots())
    panel.areas_box.setChecked(True)
    pump()
    drawn = plots()
    assert len(drawn) == alone + 1
    running = [p for p in drawn if getattr(p, 'is_running_rms', False)]
    assert len(running) == 1
    channel = window._plotted_pair[0]
    reading = window._drafts[key].reading(channel)
    rms = float(window.unit_system.from_si(reading.rms, 'acceleration'))
    assert f'{rms:.4g} g at 2000 Hz' in running[0].titleLabel.text
    labels = [item for item in drawn[0].items
              if getattr(item, 'is_segment_label', False)]
    assert len(labels) == 1, 'one segment, one label'
    assert panel.reading.isVisibleTo(panel)
    assert f'<b>{channel}</b>' in panel.reading.text()
    assert f'{rms:.4g} g rms' in panel.reading.text()
    # an edit restates the reading at once
    assert _type(panel.points, 1, 1 + window._drafts[key].channels.index(channel), '0.04')
    pump()
    flat = float(window.unit_system.from_si(
        window._drafts[key].reading(channel).rms, 'acceleration'))
    assert f'{flat:.4g} g rms' in panel.reading.text()
    panel.areas_box.setChecked(False)
    pump()
    assert len(plots()) == alone
    assert not panel.reading.isVisibleTo(panel)


def test_slopes_apply_to_the_breakpoints_and_the_entry_survives_the_form(window, pump):
    window.import_paths([fixture_path('plate', 'random.nc4')])
    pump()
    _select(window, pump, 'Specification')
    window.data_pane.waterfall_action.setChecked(False)
    window.author_data_action.setChecked(True)
    window._author_toggled(True)
    pump()
    pump()
    panel = window.data_pane.author_panel
    key = window._author_door[1]
    assert window._drafts[key].form == 'lines'
    assert not panel.slopes_button.isVisibleTo(panel), \
        'a slope into each of a controller\'s lines is not offered'
    assert not panel.areas_box.isVisibleTo(panel), \
        'nor the areas, which read a breakpoint table'
    panel.areas_box.setChecked(True)
    pump()
    assert not panel.areas_wanted
    assert not panel.reading.isVisibleTo(panel)
    assert not [item for item in window.data_pane.graphics.ci.items
                if getattr(item, 'is_running_rms', False)]
    panel.areas_box.setChecked(False)
    pump()
    window._author_entry_asked('slopes')
    pump()
    assert window._drafts[key].entry == 'slopes'
    assert panel.points.model().columnCount() == 9, 'still a level a channel'
    panel.breakpoints_button.click()
    pump()
    draft = window._drafts[key]
    assert draft.form == 'breakpoints' and draft.entry == 'slopes'
    assert panel.slopes_button.isVisibleTo(panel) and panel.slopes_button.isChecked()
    assert panel.areas_box.isVisibleTo(panel)
    assert panel.points.model().columnCount() == 17
