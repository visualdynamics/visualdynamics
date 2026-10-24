"""The sine distortion check: the record less the tracked tones against
the tones.

The numbers are closed forms. A harmonic at h of the tone's amplitude
leaves a residual of h times the tone's RMS, so the distortion reads h
itself and the ratio it stands for, sqrt(1 + h**2), reads 1.005 at
h = 0.10, 1.020 at 0.20 and 1.050 at 0.32. The record is a 2 m/s**2
sweep from 100 to 500 Hz at 100 Hz/s sampled at 4096 Hz, so its third
harmonic stays under 1500 Hz, under the Nyquist frequency.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from conftest import select_objects

from visualdynamics.core.data import TimeHistory
from visualdynamics.core.sine import (
    LOG,
    SineLevel,
    SineLevelSet,
    SineSweepSpecification,
    SineTone,
    extract_sine,
)
from visualdynamics.core.sine_distortion import (
    DISTORTION_EXCESS,
    check_distortion,
    decibels,
    distortion,
    distortion_limit,
)

FS = 4096.0
AMPLITUDE = 2.0
CONTROL = '101Z+'
LIMIT = np.sqrt(1.05 ** 2 - 1.0)            # 0.3202


def _sweep(harmonic=0.0, band=None, clip=None, offset=0.0, noise=0.0,
           top=500.0, seconds=6.0, second_tone=False):
    """The sweep with a third harmonic at `harmonic` of the tone (only
    where the drive is inside `band`, Hz, when given), clipped at `clip`,
    on an `offset`, under white `noise`; with `second_tone`, a 1 m/s**2
    tone at 1000 Hz plays alongside it from 1 to 3 s of the environment."""
    up = SineTone('Up', 0.5, [100.0, top], [[AMPLITUDE]] * 2, [0], [100.0])
    tones = [up]
    if second_tone:
        tones.append(SineTone('High', 1.0, [1000.0, 1000.0 + 2.0],
                              [[1.0]] * 2, [0], [1.0]))
    spec = SineSweepSpecification(tones, [CONTROL], ordinate_unit='m/s**2')
    n = int(seconds * FS)
    rng = np.random.default_rng(3)
    x = rng.standard_normal(n) * noise + offset
    argument = up.argument(1.0 / FS)
    start = int(0.5 * FS)
    m = min(len(argument), n - start)
    drive = up.frequency_at(np.arange(m) / FS)
    wave = np.cos(argument[:m])
    third = harmonic * np.cos(3.0 * argument[:m])
    if band is not None:
        third = third * ((drive >= band[0]) & (drive <= band[1]))
    wave = AMPLITUDE * (wave + third)
    if clip is not None:
        wave = np.clip(wave, -clip, clip)
    x[start:start + m] += wave
    if second_tone:
        high = tones[1].argument(1.0 / FS)
        at = int(1.0 * FS)
        x[at:at + len(high)] += np.cos(high)
    history = TimeHistory(np.arange(n) / FS, x[None, :],
                          response_dof=[CONTROL],
                          ordinate_dim=['acceleration'],
                          ordinate_unit=['m/s**2'])
    return history, spec


def _checked(**kwargs):
    history, spec = _sweep(**kwargs)
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    return check_distortion(history, spec, levels), history, spec, levels


def _inside(check, low=150.0, high=450.0):
    return (check.frequency > low) & (check.frequency < high)


# ---- the arithmetic ---------------------------------------------------------


def test_the_distortion_is_what_is_left_over_the_tone():
    assert float(distortion(1.05, 1.0)) == pytest.approx(0.32016, abs=1e-5)
    assert float(distortion(np.sqrt(1.01), 1.0)) == pytest.approx(0.1)
    assert float(distortion(2.0, 2.0)) == 0.0
    assert float(distortion(0.99, 1.0)) == 0.0, \
        'a ratio under one is the readings\' scatter, not a negative power'
    assert np.isnan(distortion(1.0, 0.0))
    assert distortion([1.0, 1.05], [1.0, 1.0]) == pytest.approx([0.0, LIMIT])


def test_a_tolerance_on_the_ratio_is_a_distortion_limit():
    assert DISTORTION_EXCESS == 0.05
    assert distortion_limit() == pytest.approx(0.32016, abs=1e-5)
    assert distortion_limit(0.10) == pytest.approx(np.sqrt(1.21 - 1.0))
    assert float(decibels(distortion_limit())) == pytest.approx(-9.892,
                                                                abs=1e-3)
    assert float(decibels(0.1)) == pytest.approx(-20.0)
    assert decibels(0.0) == -np.inf
    with pytest.raises(ValueError, match='more than zero'):
        distortion_limit(0.0)


# ---- the check on a sweep ---------------------------------------------------


@pytest.mark.parametrize('harmonic,ratio', [(0.10, 1.005), (0.20, 1.020),
                                            (0.32, 1.050)])
def test_a_harmonic_reads_as_its_own_share_of_the_tone(harmonic, ratio):
    check, *_rest = _checked(harmonic=harmonic)
    inside = _inside(check)
    assert check.ratio[0][inside] == pytest.approx(ratio, abs=6e-4)
    assert check.distortion[0][inside] == pytest.approx(harmonic, abs=4e-3)
    assert check.fundamental[0][inside] == pytest.approx(
        AMPLITUDE / np.sqrt(2.0), rel=2e-4), 'the tone, and only the tone'


def test_a_clean_sweep_reads_near_zero_and_passes():
    check, *_rest = _checked()
    assert check.distortion.max() < 0.02
    assert check.judged.all() and check.passed
    assert check.notes == ()
    assert check.cycles == 10.0, "read over the extraction's own smoothing"
    assert check.dofs == [CONTROL] and check.tone == 'Up'


def test_a_harmonic_in_one_band_is_flagged_there_and_nowhere_else():
    check, *_rest = _checked(harmonic=0.5, band=(250.0, 350.0))
    over = check.over[0]
    assert not check.passed
    assert over.any()
    assert check.frequency[over].min() > 240.0
    assert check.frequency[over].max() < 360.0
    clear = (check.frequency < 230.0) | (check.frequency > 370.0)
    assert check.distortion[0][clear].max() < 0.05
    frequency, worst = check.worst()
    assert 250.0 < frequency < 352.0
    assert worst == pytest.approx(0.5, abs=0.01)
    assert check.describe(CONTROL).startswith(
        f'Up at {CONTROL}: worst 50.0 % (-6.0 dB) at ')
    assert check.describe().endswith('over the 32.0 % limit')


def test_a_looser_tolerance_lets_the_same_band_through():
    check, *_rest = _checked(harmonic=0.5, band=(250.0, 350.0))
    looser = replace(check, excess=0.15)
    assert looser.limit == pytest.approx(np.sqrt(1.15 ** 2 - 1.0))
    assert looser.passed, 'a 50 % harmonic is under a 57 % limit'
    assert looser.describe().endswith('within the 56.8 % limit')


def test_clipping_hardly_moves_the_check():
    """A sine clipped at four fifths of its peak puts most of the lost
    power back into the fundamental: the ratio reads about 1.004."""
    check, *_rest = _checked(clip=0.8 * AMPLITUDE)
    inside = _inside(check, 150.0, 300.0)
    assert np.median(check.ratio[0][inside]) == pytest.approx(1.004,
                                                              abs=1e-3)
    assert check.passed


def test_an_offset_is_taken_out_before_the_rms():
    clean, *_rest = _checked()
    offset, *_rest = _checked(offset=0.7)
    assert offset.unfiltered == pytest.approx(clean.unfiltered, abs=1e-9)
    assert offset.passed


def test_noise_is_counted_as_everything_that_is_not_the_tone():
    """White noise of 0.2 RMS beside a tone of 1.414 RMS is a distortion
    of 0.2 / 1.414, 14 %, read in the band the reconstruction holds."""
    check, *_rest = _checked(noise=0.2)
    inside = _inside(check, 150.0, 300.0)
    assert np.median(check.distortion[0][inside]) == pytest.approx(
        0.2 / np.sqrt(2.0), abs=0.01)


def test_two_tones_at_once_are_both_the_fundamental():
    """With both tones extracted, both are rebuilt and taken away
    together, and a clean record reads clean where they overlap and
    across the second tone's ends."""
    check, *_rest = _checked(second_tone=True)
    assert check.judged.all()
    overlap = (check.seconds > 1.6) & (check.seconds < 2.9)
    assert overlap.any()
    # the RMS of the two together over a short span, which beats about
    # their root-sum-square
    assert check.fundamental[0][overlap] == pytest.approx(
        np.sqrt(AMPLITUDE ** 2 + 1.0) / np.sqrt(2.0), rel=1e-2)
    # read as two RMS readings, the beat and the second tone's scatter
    # read a few percent, and counted by the line's center alone the
    # spans across the second tone's ends read 29 %
    assert np.median(check.distortion[0][overlap]) < 0.002
    ends = (np.abs(check.seconds - 1.0) < 0.05) | (np.abs(check.seconds
                                                          - 3.0) < 0.05)
    assert ends.any()
    assert check.distortion[0][ends].max() < 0.05
    assert check.distortion.max() < 0.05
    assert check.passed


def test_a_tone_the_levels_do_not_hold_leaves_its_lines_unjudged():
    history, spec = _sweep(second_tone=True)
    levels = extract_sine(history, spec, tones=['Up'], cycles=10.0,
                          workers=1)
    check = check_distortion(history, spec, levels)
    shared = ~check.judged[0]
    assert shared.any()
    # a span reaching the other tone at all is shared: ten cycles of
    # the sweep at 150 to 350 Hz is under 0.07 s, half each side
    assert 1.0 - 0.035 <= check.seconds[shared].min() < 1.0
    assert 3.0 < check.seconds[shared].max() <= 3.0 + 0.035, \
        'a span with only its edge in the other tone is shared too'
    assert check.passed, 'the unjudged lines carry the other tone'
    assert check.ratio[0][shared].max() > 1.1, \
        'they would have been flagged if judged'
    assert any('shared with a tone the levels do not hold' in note
               for note in check.notes)


def test_a_line_under_the_noise_floor_is_not_judged():
    history, spec = _sweep(harmonic=0.5)
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    level = levels.levels[0]
    below = np.zeros(np.shape(level.ordinate), dtype=bool)
    below[0, :20] = True
    flagged = SineLevel(level.abscissa, level.ordinate,
                        response_dof=list(level.response_dof),
                        ordinate_dim=list(level.ordinate_dim),
                        ordinate_unit=list(level.ordinate_unit),
                        tone=level.tone, onset=level.onset,
                        seconds=level.seconds, floor=level.floor,
                        below_floor=below)
    check = check_distortion(history, spec,
                             SineLevelSet([flagged], cycles=levels.cycles))
    assert not check.judged[0, :20].any() and check.judged[0, 20:].all()
    assert int(check.over[0].sum()) == len(check.frequency) - 20
    assert check.notes == ('20 lines under the noise floor, not judged',)


def test_without_levels_the_extraction_is_run_with_the_records_setting():
    from visualdynamics.core.sine import SineExtraction

    history, spec = _sweep(harmonic=0.2)
    history.sine_extraction = SineExtraction(cycles=10.0)
    run = check_distortion(history, spec)
    given = check_distortion(history, spec,
                             extract_sine(history, spec, cycles=10.0))
    assert np.array_equal(run.frequency, given.frequency)
    assert run.distortion == pytest.approx(given.distortion, rel=1e-9)


def test_a_wider_span_reads_the_same_steady_harmonic():
    history, spec = _sweep(harmonic=0.2)
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    wide = check_distortion(history, spec, levels, cycles=40.0)
    assert wide.cycles == 40.0
    assert wide.distortion[0][_inside(wide)] == pytest.approx(0.2, abs=4e-3)


def test_a_sweep_whose_third_harmonic_passes_nyquist_says_so():
    check, *_rest = _checked(top=800.0, seconds=9.0)
    reach = FS / 2.0 / 3.0
    note = (f'above {reach:.4g} Hz the third harmonic is past the Nyquist '
            'frequency, so the record cannot hold it and the check cannot '
            'see it')
    assert check.notes == (note,)


def test_a_harmonic_up_near_nyquist_reads_in_full():
    # a 20 % third harmonic of a tone at 600 to 650 Hz is at 1800 to
    # 1950 Hz, 0.88 to 0.95 of the Nyquist frequency: counted on the
    # samples it reads in full, where the band-limited reading this
    # replaced under-read it past 0.8
    check, *_rest = _checked(harmonic=0.2, top=680.0, seconds=7.0)
    near = (check.frequency > 600.0) & (check.frequency < 650.0)
    assert near.sum() > 5
    assert check.distortion[0][near] == pytest.approx(0.2, abs=0.01)


def test_what_cannot_be_checked_is_refused():
    history, spec = _sweep()
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    with pytest.raises(ValueError, match='more than zero'):
        check_distortion(history, spec, levels, excess=0.0)
    with pytest.raises(ValueError, match='more than zero cycles'):
        check_distortion(history, spec, levels, cycles=0.0)
    with pytest.raises(ValueError, match="no tone 'Down'"):
        check_distortion(history, spec, levels, tone='Down')
    with pytest.raises(ValueError, match='no reading for'):
        check_distortion(history, spec, levels, channels=['999X+'])
    with pytest.raises(ValueError, match='do not say what smoothing'):
        check_distortion(history, spec, SineLevelSet(levels.levels))
    with pytest.raises(ValueError, match='less than the 5000'):
        check_distortion(history, spec, levels, cycles=5000.0)
    with pytest.raises(ValueError, match='no tone; extract them first'):
        check_distortion(history, spec, SineLevelSet([], cycles=10.0))


# ---- the residual read directly (Brandon, 2026-10-23) ----------------------


def _record(x):
    return TimeHistory(np.arange(len(x)) / FS, x[None, :],
                       response_dof=[CONTROL], ordinate_dim=['acceleration'],
                       ordinate_unit=['m/s**2'])


def test_two_clean_tones_crossing_read_clean_where_they_cross():
    """Two tones sweeping through each other are not orthogonal where
    they cross: the root-sum-square of the two levels read against the
    unfiltered RMS put D near 100 % there. Rebuilt together and taken
    away together, they leave nothing."""
    up = SineTone('Up', 0.5, [100.0, 500.0], [[2.0]] * 2, [0], [100.0])
    down = SineTone('Down', 0.5, [500.0, 100.0], [[1.5]] * 2, [0], [-100.0])
    spec = SineSweepSpecification([up, down], [CONTROL],
                                  ordinate_unit='m/s**2')
    n = int(5.0 * FS)
    x = np.zeros(n)
    start = int(0.5 * FS)
    for tone, amplitude in ((up, 2.0), (down, 1.5)):
        argument = tone.argument(1.0 / FS)
        m = min(len(argument), n - start)
        x[start:start + m] += amplitude * np.cos(argument[:m])
    history = _record(x)
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    for name in ('Up', 'Down'):
        check = check_distortion(history, spec, levels, tone=name)
        crossing = np.abs(check.frequency - 300.0) < 20.0
        assert crossing.sum() > 5
        assert check.distortion[0][crossing].max() < 0.01
        assert check.distortion.max() < 0.02
        assert check.passed


def _resonance(q, rate, low=200.0, high=400.0, fn=300.0):
    """A clean linear sweep through a base-driven resonance at `fn` and
    quality factor `q`: the control channel is the response, whose level
    climbs and falls through the sweep."""
    from scipy.signal import bilinear, lfilter

    up = SineTone('Up', 0.5, [low, high], [[2.0]] * 2, [0], [rate])
    spec = SineSweepSpecification([up], [CONTROL], ordinate_unit='m/s**2')
    n = int((1.0 + (high - low) / rate) * FS)
    drive = np.zeros(n)
    start = int(0.5 * FS)
    argument = up.argument(1.0 / FS)
    m = min(len(argument), n - start)
    drive[start:start + m] = np.cos(argument[:m])
    wn, zeta = 2.0 * np.pi * fn, 1.0 / (2.0 * q)
    b, a = bilinear([2.0 * zeta * wn, wn ** 2],
                    [1.0, 2.0 * zeta * wn, wn ** 2], FS)
    return _record(lfilter(b, a, drive)), spec


def test_a_clean_sweep_through_a_resonance_reads_clean():
    """A Q of 50 swept at 20 Hz/s: the level climbs twentyfold through
    the resonance. Read as the level at an instant against the RMS over
    a span, the two disagreed wherever the level moved inside the span
    and D read 8.0 % at the peak; the residual over the same span as
    the rebuilt tone reads under 1 %."""
    history, spec = _resonance(50.0, 20.0)
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    check = check_distortion(history, spec, levels)
    peak = np.abs(check.frequency - 300.0) < 15.0
    assert peak.sum() > 10
    level = np.abs(np.asarray(levels.levels[0].ordinate))[0]
    assert level.max() / level.min() > 10.0, 'the level moves a lot'
    assert check.distortion[0][peak].max() < 0.01
    # away from the ends, where the drive starting and stopping at full
    # level rings the resonance at its own frequency, which is not the
    # tone and is counted
    middle = (check.frequency > 220.0) & (check.frequency < 380.0)
    assert check.distortion[0][middle].max() < 0.01


def test_a_drifted_sweep_clock_is_rebuilt_on_its_tone():
    """A record swept at 106 Hz/s against a specification of 100: the
    extraction corrects the clock (drift_hz), and so must the rebuild,
    or the tone is rebuilt off itself and read as distortion (91 % at
    the median uncorrected)."""
    spec_tone = SineTone('Up', 0.5, [100.0, 500.0], [[2.0]] * 2, [0],
                         [100.0])
    run = SineTone('Up', 0.5, [100.0, 524.0], [[2.0]] * 2, [0], [106.0])
    spec = SineSweepSpecification([spec_tone], [CONTROL],
                                  ordinate_unit='m/s**2')
    n = int(5.5 * FS)
    x = np.zeros(n)
    start = int(0.5 * FS)
    argument = run.argument(1.0 / FS)
    m = min(len(argument), n - start)
    x[start:start + m] = AMPLITUDE * np.cos(argument[:m])
    history = _record(x)
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    assert abs(levels.levels[0].drift_hz) > 10.0
    check = check_distortion(history, spec, levels)
    assert check.distortion[0][_inside(check)].max() < 0.01
    assert check.passed


def test_a_log_sweep_off_its_schedule_is_rebuilt_on_its_tone():
    """A 10 to 1000 Hz log sweep at 4 octaves a minute recorded 1.5 %
    above its schedule: the extraction moves the clock, and the rebuild
    must follow the tone the whole way, or the tone is rebuilt off itself
    and read as distortion (more than 500 times the tone at its worst,
    fitted as a quadratic in time, 2026-10-24)."""
    spec_tone = SineTone('Log', 0.5, [10.0, 1000.0], [[2.0]] * 2, [LOG],
                         [4.0])
    spec = SineSweepSpecification([spec_tone], [CONTROL],
                                  ordinate_unit='m/s**2')
    n = int((0.5 + spec_tone.duration() + 0.5) * FS)
    x = np.zeros(n)
    start = int(0.5 * FS)
    argument = spec_tone.argument(1.0 / FS)
    m = min(len(argument), n - start)
    x[start:start + m] = AMPLITUDE * np.cos(1.015 * argument[:m])
    history = _record(x)
    levels = extract_sine(history, spec, cycles=10.0, workers=1,
                          onsets={'Log': 0.5})
    assert levels.levels[0].drift_hz > 10.0
    check = check_distortion(history, spec, levels)
    assert np.isfinite(check.distortion[0]).all()
    assert check.distortion[0].max() < 0.01


def test_a_dead_channel_is_not_judged_and_does_not_pass():
    history, spec = _sweep()
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    dead = _record(np.zeros(history.ordinate.shape[1]))
    check = check_distortion(dead, spec, levels)
    assert not check.judged.any()
    assert not check.passed, 'nothing judged is not a pass'
    assert check.describe() == 'Up: no line could be judged'
    assert any('no tone on the channel to judge against' in note
               for note in check.notes)


def test_a_sample_that_is_not_a_number_leaves_the_channel_unjudged():
    history, spec = _sweep(harmonic=0.5)
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    x = np.array(history.ordinate[0], copy=True)
    x[int(2.0 * FS)] = np.nan
    check = check_distortion(_record(x), spec, levels)
    assert not check.judged.any()
    assert not check.passed
    assert check.notes[0] == (f'{CONTROL}: 1 sample over the tone\'s '
                              'stretch is not a finite number, so no '
                              'line is judged')


def test_a_channel_is_found_by_its_quantity_as_well_as_its_dof():
    """Levels read from the accelerometer at 101Z+ are checked against
    it, not against a force gauge at the same DOF listed first."""
    history, spec = _sweep()
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    gauge = np.random.default_rng(9).standard_normal(
        history.ordinate.shape[1])
    both = TimeHistory(history.abscissa,
                       np.vstack([gauge, history.ordinate[0]]),
                       response_dof=[CONTROL, CONTROL],
                       ordinate_dim=['force', 'acceleration'],
                       ordinate_unit=['N', 'm/s**2'])
    check = check_distortion(both, spec, levels)
    assert check.dimensions == ['acceleration']
    assert check.passed and check.distortion.max() < 0.02
    only = TimeHistory(history.abscissa, gauge[None, :],
                       response_dof=[CONTROL], ordinate_dim=['force'],
                       ordinate_unit=['N'])
    with pytest.raises(ValueError, match='read in acceleration'):
        check_distortion(only, spec, levels)


def test_levels_that_do_not_say_when_they_were_read_are_refused():
    """Levels built through the API or loaded without their times have
    no span to check; the refusal is a ValueError the panel shows."""
    history, spec = _sweep()
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    levels.levels[0].seconds = None
    with pytest.raises(ValueError, match='do not say when each line'):
        check_distortion(history, spec, levels)


def test_levels_read_from_another_record_are_refused():
    """Levels whose lines fall outside the tone's stretch of the record
    are not of it: a record cut short since they were extracted."""
    history, spec = _sweep()
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    keep = int(3.0 * FS)
    shorter = TimeHistory(history.abscissa[:keep],
                          history.ordinate[:, :keep],
                          response_dof=[CONTROL],
                          ordinate_dim=['acceleration'],
                          ordinate_unit=['m/s**2'])
    with pytest.raises(ValueError, match='Extract them again'):
        check_distortion(shorter, spec, levels)


# ---- the plot -----------------------------------------------------------------


def test_the_plot_names_the_limit_and_the_verdict_in_words(qt_app):
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from visualdynamics.plot.sine_distortion import SYMBOLS, build_distortion

    check, *_rest = _checked(harmonic=0.5, band=(250.0, 350.0))
    layout = pg.GraphicsLayoutWidget()
    plot = build_distortion(layout, check)
    names = [label.text for _sample, label in plot.legend.items]
    assert 'limit 32.0 % (-9.9 dB): unfiltered 5 % over the tone' in names
    over = int(check.over[0].sum())
    assert any(name.startswith(f'{CONTROL}: worst 50.0 % at ')
               and name.endswith(f'{over} lines over the limit')
               for name in names), names
    limit, = [item for item in plot.listDataItems()
              if item.name() and item.name().startswith('limit')]
    assert limit.opts['pen'].style() == Qt.DashLine, \
        'the limit reads by its line style, not its hue'
    assert limit.getData()[1].tolist() == pytest.approx([100.0 * LIMIT] * 2)
    marked = [item for item in plot.listDataItems()
              if item.opts['symbol'] is not None]
    assert {item.opts['symbol'] for item in marked} == {SYMBOLS[0]}
    filled = [item for item in marked if item.opts['symbolBrush'] is not None]
    assert len(filled) == 1 and len(filled[0].getData()[0]) == over, \
        'a line over the limit is the same marker, filled'
    assert 'over the limit' in plot.titleLabel.text


def test_the_plot_draws_to_a_file_with_no_window(tmp_path):
    from visualdynamics.plot import plot_sine_distortion

    history, spec = _sweep(harmonic=0.2)
    levels = extract_sine(history, spec, cycles=10.0, workers=1)
    path = tmp_path / 'distortion.png'
    plot_sine_distortion(history, spec, levels, path=str(path), show=False)
    assert path.stat().st_size > 5_000


# ---- the readout in the window --------------------------------------------------


def _journaled(window):
    """The distortion readout's line in the journal: the frequency
    readout under it journals its own line too, each settling in place."""
    lines = [line for line in window.project.journal
             if line.startswith('visualdynamics.plot.plot_sine_distortion(')]
    assert len(lines) == 1, lines
    return lines[0]


def _shown(window, pump, history, spec, extract=True):
    window.add_object('Record', history)
    window.add_object('Sweep', spec)
    if extract:
        from visualdynamics.core.sine import SineExtraction

        history.sine_extraction = SineExtraction(cycles=10.0)
        window.project.extract_sine('Record', 'Sweep')
    select_objects(window, pump, 'Record')
    window.render_current()
    pump()
    pane = window.data_pane
    pane.sine_action.setChecked(True)
    pane._choose_sine(True)
    pump()
    pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    return pane.sine_panel.distortion


def test_the_readout_checks_the_extracted_levels_in_the_sine_view(window,
                                                                   pump):
    panel = _shown(window, pump, *_sweep(harmonic=0.5, band=(250.0, 350.0)))
    assert panel.isVisible()
    assert panel.channel_box.currentText() == CONTROL
    assert not panel.tone_box.isVisible(), 'one tone: no tone to choose'
    assert panel.derived['limit'].text() == '32.0 % (-9.9 dB)'
    assert panel.derived['worst'].text().startswith('50.0 % (-6.0 dB) at ')
    assert panel.derived['verdict'].text() == 'over the limit'
    assert panel.derived['over'].text().endswith(' Hz')
    assert panel.preview.items, 'the distortion against frequency is drawn'
    assert _journaled(window) == (
        "visualdynamics.plot.plot_sine_distortion(project['Record'], "
        "project['Sweep'], project['Sine Levels'], tone='Up', "
        "channel='101Z+', excess=0.05, path='distortion.png', show=False)")


def test_the_readout_stays_away_until_the_levels_are_extracted(window, pump):
    panel = _shown(window, pump, *_sweep(), extract=False)
    assert window.data_pane.sine_panel.isVisible()
    assert not panel.isVisible(), 'nothing to check yet (principle 3)'


def test_a_tolerance_edit_reads_again_without_reading_the_record(window,
                                                                 pump):
    panel = _shown(window, pump, *_sweep(harmonic=0.5, band=(250.0, 350.0)))
    made = panel.check
    panel.excess_box.setValue(15.0)
    pump()
    assert panel.check.excess == pytest.approx(0.15)
    assert panel.check.unfiltered is made.unfiltered, \
        'the RMS readings are kept; only the judgement moved'
    assert panel.derived['verdict'].text() == 'within the limit'
    assert panel.derived['limit'].text() == '56.8 % (-4.9 dB)'
    assert 'excess=0.15,' in _journaled(window)


def test_stale_levels_are_said_to_be_stale_and_not_checked(window, pump):
    from visualdynamics.core.sine import SineExtraction

    panel = _shown(window, pump, *_sweep(harmonic=0.5, band=(250.0, 350.0)))
    assert panel.check is not None
    # the smoothing moved since the levels were read: the project marks
    # them stale, and the readout checks nothing against them
    window.project['Record'].sine_extraction = SineExtraction(cycles=20.0)
    window._refresh_stale_badges()
    window.render_current()
    pump()
    assert panel.isVisible()
    assert panel.check is None
    assert 'These levels are stale' in panel.notes.text()
    assert panel.derived['verdict'].text() == '—'
    panel.excess_box.setValue(15.0)
    pump()
    assert panel.check is None, 'an edit does not check stale levels'
