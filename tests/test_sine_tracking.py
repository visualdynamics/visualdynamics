"""A sweep read through a tracking filter and a detector.

The readings a controller takes of a swept tone, held to closed forms
on synthetic records: a clean sweep, the same with a third harmonic
riding it (a clipped drive's signature), and the same under broadband
noise. A pure sine's peak, its RMS times root two and its mean
absolute value times pi/2 are all its amplitude; a harmonic in phase
adds its own amplitude to the peak and its power to the RMS; white
noise adds its variance to the mean square. The filter sees none of
it — and pays in time, which the step tests pin.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.data import TimeHistory
from visualdynamics.core.sine import LOG, SineSweepSpecification, SineTone
from visualdynamics.core.sine_tracking import (
    SineTracking,
    track_sine,
)

RATE = 16384.0
#: where the tone's sweep begins in the record: after a quiet lead-in,
#: as a controller's recording starts before its sweep does
ONSET = 0.5
AMPLITUDE = 3.0          # m/s**2
HARMONIC = 0.3           # of the amplitude, at three times the drive
NOISE = 0.3              # standard deviation, of the amplitude
DOF = '101Z+'


def _tone():
    # 20 to 320 Hz at 60 oct/min: four octaves in four seconds, and the
    # top of it under a fiftieth of the sample rate, where a sampled
    # peak reads within 0.2 % of the true one
    return SineTone('Sine Tone 1', ONSET, [20.0, 320.0],
                    [[AMPLITUDE], [AMPLITUDE]], [LOG], [60.0])


@pytest.fixture(scope='module')
def spec():
    return SineSweepSpecification([_tone()], [DOF], ordinate_unit='m/s**2')


def _record(content='clean', seed=0):
    tone = _tone()
    t = np.arange(int((ONSET + tone.duration() + 0.25) * RATE)) / RATE
    local = np.clip(t - ONSET, 0.0, None)
    inside = (t >= ONSET) & (local <= tone.duration())
    phase = tone.phase_at(local)
    wave = np.cos(phase)
    if content == 'harmonic':
        wave = wave + HARMONIC * np.cos(3.0 * phase)
    x = np.where(inside, AMPLITUDE * wave, 0.0)
    if content == 'noise':
        x = x + AMPLITUDE * NOISE * np.random.default_rng(seed).standard_normal(len(t))
    return TimeHistory(t, x[None], response_dof=[DOF],
                       ordinate_dim=['acceleration'])


@pytest.fixture(scope='module')
def clean():
    return _record('clean')


@pytest.fixture(scope='module')
def harmonic():
    return _record('harmonic')


@pytest.fixture(scope='module')
def noisy():
    return _record('noise')


def _read(level):
    """A level's readings on its one channel, as amplitudes."""
    return np.abs(level.ordinate[0])


FILTERED = [SineTracking(proportional=0.5), SineTracking(proportional=0.1),
            SineTracking(fixed=10.0)]


# ---- the setting ---------------------------------------------------------


def test_a_band_is_proportional_or_fixed_never_both():
    with pytest.raises(ValueError, match='not both'):
        SineTracking(proportional=0.5, fixed=10.0)
    assert SineTracking(proportional=0.5).kind == 'proportional'
    assert SineTracking(fixed=10.0).kind == 'fixed'
    assert SineTracking(detector='peak').kind == 'unfiltered'


def test_the_filters_own_output_needs_a_filter():
    with pytest.raises(ValueError, match='needs a filter'):
        SineTracking()
    with pytest.raises(ValueError, match='not a detector'):
        SineTracking(detector='quasi-peak', proportional=0.5)


def test_a_proportional_band_must_stay_about_the_drive():
    """Half of a band twice the drive reaches zero frequency: past
    that it is not about the drive anywhere."""
    with pytest.raises(ValueError, match='between zero and two'):
        SineTracking(proportional=2.0)
    with pytest.raises(ValueError, match='between zero and two'):
        SineTracking(proportional=0.0)
    with pytest.raises(ValueError, match='wider than zero'):
        SineTracking(fixed=-5.0)


def test_the_band_and_its_settling_follow_the_drive():
    proportional = SineTracking(proportional=0.5)
    assert proportional.bandwidth(200.0) == pytest.approx(100.0)
    assert proportional.settling(200.0) == pytest.approx(0.01)
    fixed = SineTracking(fixed=10.0)
    assert np.allclose(fixed.bandwidth([20.0, 300.0]), 10.0)
    with pytest.raises(ValueError, match='no band'):
        SineTracking(detector='rms').bandwidth(100.0)


def test_a_reading_names_itself():
    assert SineTracking(detector='peak').describe() == 'peak, unfiltered'
    assert SineTracking(proportional=0.5).describe() == \
        'filter output, 50 % proportional'
    assert SineTracking(fixed=10.0, detector='rms').describe() == \
        'rms as peak, 10 Hz fixed'
    assert SineTracking(proportional=0.07, detector='mean').describe() == \
        'mean as peak, 7 % proportional'


# ---- a clean tone --------------------------------------------------------


def test_every_reading_of_a_clean_tone_is_its_amplitude(clean, spec):
    """Every detector is exact for a pure sine, filtered or not —
    which is why the differences on a real record are worth seeing."""
    settings = FILTERED + [SineTracking(detector=d)
                           for d in ('peak', 'rms', 'mean')]
    for setting, level in zip(settings, track_sine(clean, spec, settings,
                                                   onset=ONSET)):
        read = _read(level)
        assert np.max(np.abs(read / AMPLITUDE - 1.0)) < 0.015, \
            setting.describe()
        assert np.median(read) == pytest.approx(AMPLITUDE, rel=1e-3), \
            setting.describe()


def test_the_filtered_level_recovers_the_tone_within_one_percent(clean,
                                                                  spec):
    """Fixed and proportional both: every line within 1 % of the
    amplitude the record was made with, from the first line to the
    last."""
    for setting, level in zip(FILTERED, track_sine(clean, spec, FILTERED,
                                                   onset=ONSET)):
        assert np.max(np.abs(_read(level) / AMPLITUDE - 1.0)) < 0.01, \
            setting.describe()


def test_the_onset_is_found_when_not_given(clean, spec):
    level, = track_sine(clean, spec, SineTracking(proportional=0.5))
    assert level.onset == pytest.approx(ONSET, abs=2.0 / RATE)
    assert np.median(_read(level)) == pytest.approx(AMPLITUDE, rel=1e-3)


# ---- a harmonic ----------------------------------------------------------


def test_an_unfiltered_peak_counts_the_harmonic(harmonic, spec):
    """A third harmonic in phase with its tone peaks with it, so the
    unfiltered peak reads the sum of their amplitudes."""
    level, = track_sine(harmonic, spec, SineTracking(detector='peak'),
                        onset=ONSET)
    assert np.median(_read(level)) == pytest.approx(
        AMPLITUDE * (1.0 + HARMONIC), rel=0.005)


def test_an_unfiltered_rms_counts_the_harmonics_power(harmonic, spec):
    level, = track_sine(harmonic, spec, SineTracking(detector='rms'),
                        onset=ONSET)
    assert np.median(_read(level)) == pytest.approx(
        AMPLITUDE * np.sqrt(1.0 + HARMONIC ** 2), rel=0.005)


def test_the_filtered_level_does_not_see_the_harmonic(harmonic, spec):
    for setting, level in zip(FILTERED, track_sine(harmonic, spec, FILTERED,
                                                   onset=ONSET)):
        assert np.max(np.abs(_read(level) / AMPLITUDE - 1.0)) < 0.01, \
            setting.describe()


def test_a_detector_after_the_filter_does_not_see_it_either(harmonic, spec):
    """The filter is what rejects the harmonic, not the detector: a
    peak read after the band is the tone's peak."""
    filtered, raw = track_sine(
        harmonic, spec, [SineTracking(proportional=0.5, detector='peak'),
                         SineTracking(detector='peak')], onset=ONSET)
    assert np.median(_read(filtered)) == pytest.approx(AMPLITUDE, rel=0.005)
    assert np.median(_read(raw)) > 1.25 * AMPLITUDE


# ---- broadband noise -----------------------------------------------------


def test_an_unfiltered_rms_counts_the_noise_power(noisy, spec):
    """The mean square of tone plus white noise is A**2/2 + sigma**2,
    so the RMS as peak reads A * sqrt(1 + 2 (sigma/A)**2)."""
    level, = track_sine(noisy, spec, SineTracking(detector='rms'),
                        onset=ONSET)
    assert np.median(_read(level)) == pytest.approx(
        AMPLITUDE * np.sqrt(1.0 + 2.0 * NOISE ** 2), rel=0.01)


def test_the_narrow_band_rejects_the_noise(noisy, spec):
    """The noise a band lets through is its share of the whole: a
    tenth of the drive at a few hundred Hz is a few hundredths of the
    noise. The filtered level sits on the tone; the unfiltered peak
    does not."""
    narrow, raw = track_sine(
        noisy, spec, [SineTracking(proportional=0.1),
                      SineTracking(detector='peak')], onset=ONSET)
    assert np.median(_read(narrow)) == pytest.approx(AMPLITUDE, rel=0.005)
    assert np.percentile(np.abs(_read(narrow) / AMPLITUDE - 1.0), 95) < 0.03
    assert np.median(_read(raw)) > 1.4 * AMPLITUDE


# ---- the price: settling -------------------------------------------------


@pytest.fixture(scope='module')
def stepped():
    """A linear sweep whose level doubles two seconds in: 50 to 450 Hz
    at 100 Hz/s, so the step lands at 250 Hz."""
    tone = SineTone('Sine Tone 1', 0.0, [50.0, 450.0], [[1.0], [1.0]],
                    [0], [100.0])
    t = np.arange(int(tone.duration() * 8192.0) + 1) / 8192.0
    x = np.where(t < 2.0, 1.0, 2.0) * np.cos(tone.phase_at(t))
    history = TimeHistory(t, x[None], response_dof=[DOF],
                          ordinate_dim=['acceleration'])
    return history, SineSweepSpecification([tone], [DOF])


def _half_way(level):
    """Seconds after the step at which the reading first reaches half
    of it."""
    after = (level.seconds > 2.0) & (_read(level) >= 1.5)
    return float(level.seconds[np.flatnonzero(after)[0]] - 2.0)


def test_a_narrow_band_follows_a_step_later(stepped):
    history, spec = stepped
    wide, narrow = track_sine(
        history, spec, [SineTracking(proportional=0.5),
                        SineTracking(proportional=0.1)],
        onset=0.0, lines=2000)
    assert _half_way(narrow) > 3.0 * _half_way(wide), \
        'a fifth of the band, several times the lag'


def test_the_lag_is_about_one_over_the_bandwidth(stepped):
    """The rule of thumb, held loosely: the fourth-order filter is half
    way at 0.9 of 1/B."""
    history, spec = stepped
    for setting in (SineTracking(proportional=0.1),
                    SineTracking(fixed=25.0)):
        level, = track_sine(history, spec, setting, onset=0.0, lines=2000)
        bandwidth = float(setting.bandwidth(250.0))
        assert _half_way(level) * bandwidth == pytest.approx(0.9, abs=0.15), \
            setting.describe()


def test_an_unfiltered_reading_follows_within_its_cycle(stepped):
    history, spec = stepped
    level, = track_sine(history, spec, SineTracking(detector='peak'),
                        onset=0.0, lines=2000)
    assert _half_way(level) < 1.5 / 250.0


# ---- what the levels carry -----------------------------------------------


def test_the_levels_carry_units_tone_and_reading(clean, spec):
    settings = [SineTracking(detector='peak'), SineTracking(fixed=10.0)]
    levels = track_sine(clean, spec, settings, onset=ONSET, lines=50)
    assert len(levels) == 2
    for setting, level in zip(settings, levels):
        assert level.ordinate_dim == ['acceleration']
        assert level.ordinate_unit == ['m/s**2']
        assert level.response_dof == [DOF]
        assert level.tone == 'Sine Tone 1'
        assert level.onset == ONSET
        assert level.comment == [f'Sine Tone 1 at {DOF}, {setting.describe()}']
        assert len(level.abscissa) == 50
        assert np.all(np.diff(level.abscissa) > 0), 'ascending in frequency'
        tone = spec.tone('Sine Tone 1')
        assert np.allclose(tone.frequency_at(level.seconds - ONSET),
                           level.abscissa), \
            'each line is stamped with the second it was read'


def test_a_descending_sweep_reads_ascending(spec):
    tone = SineTone('Down', 0.0, [300.0, 30.0], [[1.0], [1.0]], [0],
                    [-135.0])
    t = np.arange(int(tone.duration() * 8192.0) + 1) / 8192.0
    history = TimeHistory(t, np.cos(tone.phase_at(t))[None],
                          response_dof=[DOF], ordinate_dim=['acceleration'])
    level, = track_sine(history, SineSweepSpecification([tone], [DOF]),
                        SineTracking(proportional=0.5), onset=0.0)
    assert np.all(np.diff(level.abscissa) >= 0)
    assert np.all(np.diff(level.seconds) < 0), 'read late at the low end'
    assert np.max(np.abs(_read(level) - 1.0)) < 0.01


def test_a_named_channel_is_read_and_a_missing_one_refused(clean, spec):
    level, = track_sine(clean, spec, SineTracking(proportional=0.5),
                        onset=ONSET, channels=[DOF])
    assert level.response_dof == [DOF]
    with pytest.raises(ValueError, match='no record'):
        track_sine(clean, spec, SineTracking(proportional=0.5),
                   onset=ONSET, channels=['999X+'])


# ---- the plot ------------------------------------------------------------


def test_the_readings_draw_headless(harmonic, spec, tmp_path):
    from visualdynamics.plot import plot_sine_tracking

    path = tmp_path / 'tracking.png'
    plot_sine_tracking(harmonic, spec,
                       [SineTracking(detector='peak'),
                        SineTracking(proportional=0.5),
                        SineTracking(proportional=0.1)],
                       onset=ONSET, path=str(path), show=False)
    assert path.exists() and path.stat().st_size > 2000


def test_each_reading_wears_its_own_marker_and_name(qt_app, harmonic, spec):
    """Told apart by shape as well as color, and named in the legend
    the way the reading names itself."""
    import pyqtgraph as pg

    from visualdynamics.plot.sine_tracking import (
        SYMBOLS,
        build_sine_tracking,
    )

    settings = [SineTracking(detector='peak'),
                SineTracking(proportional=0.5),
                SineTracking(proportional=0.1)]
    levels = track_sine(harmonic, spec, settings, onset=ONSET, lines=100)
    layout = pg.GraphicsLayoutWidget()
    try:
        drawn = build_sine_tracking(layout, levels, settings,
                                    specification=spec)
        assert drawn == 3
        plot = layout.getItem(0, 0)
        names = [label.text for _sample, label in plot.legend.items]
        assert names == ['specification'] + [s.describe() for s in settings]
        symbols = [item.opts['symbol'] for item in plot.listDataItems()
                   if item.opts.get('symbol') is not None]
        assert symbols == list(SYMBOLS[:3])
        markers = [item for item in plot.listDataItems()
                   if item.opts.get('symbol') is not None]
        assert all(4 <= len(item.getData()[0]) <= 12 for item in markers)
    finally:
        layout.close()


def test_levels_and_settings_are_drawn_in_pairs(qt_app, clean, spec):
    import pyqtgraph as pg

    from visualdynamics.plot.sine_tracking import build_sine_tracking

    levels = track_sine(clean, spec, SineTracking(proportional=0.5),
                        onset=ONSET, lines=20)
    layout = pg.GraphicsLayoutWidget()
    try:
        with pytest.raises(ValueError, match='in pairs'):
            build_sine_tracking(layout, levels, [])
    finally:
        layout.close()
