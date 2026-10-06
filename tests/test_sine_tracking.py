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


# ---- the band's shape, its weight, and the waveform it passes -------------


def test_the_bands_shape_is_three_db_down_at_its_edges():
    """The drive plus and minus half the bandwidth are the Butterworth's
    corners after demodulation: −3 dB, by definition."""
    for setting, drive in ((SineTracking(proportional=0.5), 100.0),
                           (SineTracking(proportional=0.1), 250.0),
                           (SineTracking(fixed=5.0), 300.0)):
        half = float(setting.bandwidth(drive)) / 2.0
        shape = setting.response(drive, [drive - half, drive, drive + half])
        assert shape == pytest.approx([-3.0103, 0.0, -3.0103], abs=1e-3), \
            setting.describe()


def test_the_shape_drawn_is_the_shape_the_band_passes():
    """The response is the band's own: a steady tone a given offset
    from the drive comes through `track_sine`'s filter at the level
    the shape reads there."""
    from visualdynamics.core.sine_tracking import _band_passed

    rate = 8192.0
    t = np.arange(int(3.0 * rate)) / rate
    setting = SineTracking(fixed=50.0)
    for offset in (10.0, 25.0, 40.0):
        through = _band_passed(np.exp(2j * np.pi * offset * t),
                               np.full(len(t), 25.0), rate, setting.order,
                               0j)
        measured = 20.0 * np.log10(np.mean(np.abs(through[-2000:])))
        assert measured == pytest.approx(
            float(setting.response(100.0, 100.0 + offset)), abs=0.01)


def test_the_weight_is_the_bands_settling():
    """The impulse response integrates to one, and its running integral
    — the step response — is half way at 0.9 of one over the band and
    overshoots by about a tenth, the numbers the guide states."""
    for setting in (SineTracking(proportional=0.1), SineTracking(fixed=5.0)):
        width = float(setting.bandwidth(250.0))
        lags = np.linspace(0.0, 20.0 / width, 200001)
        step = np.cumsum(setting.weighting(250.0, lags)) * (lags[1] - lags[0])
        assert step[-1] == pytest.approx(1.0, abs=1e-4)
        half = lags[np.argmax(step >= 0.5)] * width
        assert half == pytest.approx(0.9, abs=0.01), setting.describe()
        assert step.max() == pytest.approx(1.108, abs=0.005)
    assert SineTracking(fixed=5.0).weighting(250.0, -0.1) == 0.0, \
        'the band cannot see ahead'


def test_the_waveform_is_read_through_the_same_band_as_the_levels(harmonic,
                                                                   spec):
    """One implementation: the waveform's output amplitude at the
    instants a level was read is that level."""
    from visualdynamics.core.sine_tracking import track_waveform

    for setting in FILTERED:
        level, = track_sine(harmonic, spec, setting, onset=ONSET, lines=50)
        waveform = track_waveform(harmonic, spec, setting, onset=ONSET)
        at = np.searchsorted(waveform.time, level.seconds - 0.5 / RATE)
        assert np.allclose(waveform.level[at], _read(level), rtol=1e-12), \
            setting.describe()


def test_the_band_passes_the_tone_and_not_its_harmonic(harmonic, spec):
    """What comes through the band is the tone alone, sample by sample:
    the harmonic's 30 % is gone from the waveform, not only from the
    level."""
    from visualdynamics.core.sine_tracking import track_waveform

    waveform = track_waveform(harmonic, spec,
                              SineTracking(proportional=0.5), onset=ONSET)
    tone = _tone()
    clean = AMPLITUDE * np.cos(tone.phase_at(waveform.time - ONSET))
    assert np.max(np.abs(waveform.passed - clean)) < 0.01 * AMPLITUDE
    raw = harmonic.ordinate[0, np.searchsorted(harmonic.abscissa,
                                               waveform.time[0]):]
    assert np.max(np.abs(raw[:len(clean)] - clean)) > 0.25 * AMPLITUDE
    assert waveform.unit == 'm/s**2' or waveform.unit is None
    assert (waveform.dof, waveform.dimension) == (DOF, 'acceleration')


def test_the_instant_a_drive_passes_a_frequency(clean, spec):
    """An octave a second from 20 Hz: 100 Hz is log2(5) seconds in."""
    from visualdynamics.core.sine_tracking import track_waveform

    waveform = track_waveform(clean, spec, SineTracking(fixed=5.0),
                              onset=ONSET)
    assert waveform.instant(100.0) == pytest.approx(
        ONSET + np.log2(5.0), abs=1.0 / RATE)
    with pytest.raises(ValueError, match='never passes 1000 Hz'):
        waveform.instant(1000.0)


def test_a_waveform_needs_a_band(clean, spec):
    from visualdynamics.core.sine_tracking import track_waveform

    with pytest.raises(ValueError, match='no band'):
        track_waveform(clean, spec, SineTracking(detector='peak'),
                       onset=ONSET)


# ---- the band drawn on its record ----------------------------------------


@pytest.fixture(scope='module')
def dirty():
    """The guide's record: the tone, its third harmonic and broadband
    noise together."""
    record = _record('harmonic')
    noise = AMPLITUDE * 0.2 * np.random.default_rng(3).standard_normal(
        record.ordinate.shape[1])
    return TimeHistory(record.abscissa, record.ordinate + noise,
                       response_dof=[DOF], ordinate_dim=['acceleration'])


def _view(history, spec, setting, **kwargs):
    """The view built on a laid-out widget, as a window shows it."""
    import pyqtgraph as pg

    from visualdynamics.core.sine_tracking import track_waveform
    from visualdynamics.plot.tracking_filter import build_tracking_filter

    widget = pg.GraphicsLayoutWidget(size=(1100, 760))
    widget.resize(1100, 760)
    waveform = track_waveform(history, spec, setting, onset=ONSET)
    view = build_tracking_filter(widget, history, waveform, **kwargs)
    widget.show()
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
    return widget, view


def _edges(view):
    """The corridor as drawn: (seconds, lower Hz, upper Hz)."""
    lower, upper = view.picture.corridor
    t, low = lower.getData()
    _t, high = upper.getData()
    return t, 10.0 ** low, 10.0 ** high


@pytest.mark.parametrize('setting', [SineTracking(proportional=0.5),
                                     SineTracking(fixed=5.0)],
                         ids=['proportional', 'fixed'])
def test_the_corridor_is_the_bandwidth_about_the_tone(qt_app, dirty, spec,
                                                      setting):
    """At every time the corridor is drawn, its edges are the tone's
    own frequency — from the sweep's law, not the view's — plus and
    minus half the bandwidth: a constant share of the drive for a
    proportional band, a constant width in Hz for a fixed one. The
    tone is inside it and its third harmonic outside it."""
    widget, view = _view(dirty, spec, setting)
    try:
        t, low, high = _edges(view)
        tone = _tone().frequency_at(t - ONSET)
        assert len(t) > 500
        assert np.allclose(high - low, setting.bandwidth(tone), rtol=1e-9)
        assert np.allclose((low + high) / 2.0, tone, rtol=1e-9)
        if setting.kind == 'proportional':
            assert np.allclose((high - low) / tone, 0.5)
        else:
            assert np.allclose(high - low, 5.0)
        assert np.all((low < tone) & (tone < high))
        assert np.all(3.0 * tone > high)
    finally:
        widget.close()


def test_the_picture_puts_the_tone_inside_the_corridor(qt_app, dirty, spec):
    """Registration, read off the picture itself: at sampled columns
    the scalogram's ridge falls between the corridor's edges, and the
    loudest thing above the corridor is the harmonic, at three times
    the drive."""
    import pyqtgraph as pg

    widget, view = _view(dirty, spec, SineTracking(proportional=0.5))
    try:
        image = next(item for item in view.picture.items
                     if isinstance(item, pg.ImageItem))
        decibels = image.image                       # (times, rows)
        assert list(image.getLevels()) == [-60.0, 0.0], \
            'decibels below the largest, so the noise has a color'
        rect = image.mapRectToParent(image.boundingRect())
        columns = rect.left() + (np.arange(decibels.shape[0]) + 0.5) \
            * rect.width() / decibels.shape[0]
        rows = 10.0 ** (rect.top() + (np.arange(decibels.shape[1]) + 0.5)
                        * rect.height() / decibels.shape[1])
        t, low, high = _edges(view)
        checked = 0
        for at in np.linspace(ONSET + 0.4, ONSET + 3.6, 17):
            column = int(np.argmin(np.abs(columns - at)))
            drive = float(_tone().frequency_at(columns[column] - ONSET))
            lower = float(np.interp(columns[column], t, low))
            upper = float(np.interp(columns[column], t, high))
            ridge = rows[int(np.argmax(decibels[column]))]
            assert lower < ridge < upper, f'{at:.2f} s: ridge {ridge:.1f}'
            outside = rows > upper * 1.2
            loudest = rows[outside][int(np.argmax(decibels[column][outside]))]
            assert loudest == pytest.approx(3.0 * drive, rel=0.08)
            checked += 1
        assert checked == 17
    finally:
        widget.close()


def test_the_cursor_reads_the_band_at_its_instant(qt_app, dirty, spec):
    """At the instant the tone passes 100 Hz, a 50 % band is 75 to
    125 Hz and settles in 20 ms, and the shape marks the drive at 0 dB
    and its third harmonic 72 dB down — on both panels, at the same
    frequencies."""
    from visualdynamics.core.sine_tracking import track_waveform

    setting = SineTracking(proportional=0.5)
    at = track_waveform(dirty, spec, setting, onset=ONSET).instant(100.0)
    widget, view = _view(dirty, spec, setting, cursor=at)
    try:
        assert view.cursor == pytest.approx(at)
        assert view.drive_hz == pytest.approx(100.0, rel=1e-6)
        assert view.edges_hz == pytest.approx((75.0, 125.0), rel=1e-6)
        assert view.harmonic_db[3] == pytest.approx(-72.25, abs=0.01)
        x, y = view.shape_drive.getData()
        assert (x[0], 10.0 ** y[0]) == pytest.approx((0.0, 100.0))
        x, y = view.shape_harmonics.getData()
        assert (x[0], 10.0 ** y[0]) == pytest.approx((-72.25, 300.0),
                                                     abs=0.01)
        x, y = view.picture_drive.getData()
        assert (x[0], 10.0 ** y[0]) == pytest.approx((at, 100.0))
        x, y = view.picture_harmonics.getData()
        assert (x[0], 10.0 ** y[0]) == pytest.approx((at, 300.0))
        assert [10.0 ** edge.value() for edge in view.shape_edges] == \
            pytest.approx([75.0, 125.0])
        assert view.text.splitlines() == [
            f'at {at:.3f} s, drive 100 Hz',
            'band 50 Hz wide: 75 to 125 Hz',
            'settles in about 20 ms (1 / band)',
            '3 × drive, 300 Hz: −72 dB']
    finally:
        widget.close()


def test_a_fixed_band_reads_its_harmonic_below_the_floor(qt_app, dirty,
                                                         spec):
    widget, view = _view(dirty, spec, SineTracking(fixed=5.0), cursor=2.0)
    try:
        assert view.bandwidth_hz == 5.0
        assert view.edges_hz[1] - view.edges_hz[0] == pytest.approx(5.0)
        assert view.harmonic_db[3] < -80.0
        assert view.text.splitlines()[-1].endswith('below −80 dB')
        x, _y = view.shape_harmonics.getData()
        assert x[0] == -80.0, 'the mark sits on the floor, not off the panel'
    finally:
        widget.close()


def test_the_weight_reaches_back_from_the_cursor(qt_app, dirty, spec):
    """The band's weight on the record ends at the cursor and reaches
    back a few settling times: five times as far for a band a fifth as
    wide."""
    reaches = []
    for setting in (SineTracking(fixed=25.0), SineTracking(fixed=5.0)):
        widget, view = _view(dirty, spec, setting, cursor=2.0)
        try:
            x, _y = view.weight.getData()
            assert x[1] == pytest.approx(2.0)
            reaches.append(x[1] - x[0])
            assert 2.0 < view.reach * setting.fixed < 6.0
        finally:
            widget.close()
    assert reaches[1] / reaches[0] == pytest.approx(5.0, rel=1e-6)


def test_dragging_the_cursor_restates_the_band(qt_app, dirty, spec):
    """A drag on either time axis moves both cursors and reads the band
    again where it stopped."""
    widget, view = _view(dirty, spec, SineTracking(proportional=0.5),
                         cursor=1.0)
    try:
        record_line, picture_line = view.lines
        assert record_line.movable and picture_line.movable
        picture_line.setValue(ONSET + 3.0)           # 160 Hz
        assert view.cursor == pytest.approx(ONSET + 3.0)
        assert record_line.value() == pytest.approx(ONSET + 3.0)
        assert view.drive_hz == pytest.approx(160.0, rel=1e-4)
        assert view.edges_hz == pytest.approx((120.0, 200.0), rel=1e-4)
        record_line.setValue(-5.0)                   # before the tone
        assert view.cursor == pytest.approx(ONSET), 'held to the span'
    finally:
        widget.close()


def test_the_panels_share_their_axes_pixel_for_pixel(qt_app, dirty, spec):
    """The record and the picture have one time axis — the same left
    and right edges, so the cursor is one vertical — and the shape
    stands on the picture's frequency axis: the same top and bottom."""
    widget, view = _view(dirty, spec, SineTracking(proportional=0.5))
    try:
        record = view.record_plot.getViewBox().sceneBoundingRect()
        picture = view.picture.getViewBox().sceneBoundingRect()
        shape = view.shape.getViewBox().sceneBoundingRect()
        assert abs(record.left() - picture.left()) < 1.0
        assert abs(record.right() - picture.right()) < 1.0
        assert abs(shape.top() - picture.top()) < 1.0
        assert abs(shape.bottom() - picture.bottom()) < 1.0
        assert view.record_plot.getViewBox().viewRange()[0] == \
            pytest.approx(view.picture.getViewBox().viewRange()[0])
        assert view.shape.getViewBox().viewRange()[1] == \
            pytest.approx(view.picture.getViewBox().viewRange()[1])
    finally:
        widget.close()


def test_the_view_names_what_it_draws(qt_app, dirty, spec):
    widget, view = _view(dirty, spec, SineTracking(proportional=0.5),
                         harmonics=(2, 3))
    try:
        names = [label.text for _sample, label in view.picture.legend.items]
        assert names == ['record', 'through the band',
                         'band edges, −3 dB',
                         "the band's weight on the record", 'drive',
                         '2 × drive', '3 × drive']
        assert view.record_plot.titleLabel.text == \
            'Sine Tone 1 at 101Z+: filter output, 50 % proportional'
        symbols = [view.picture_drive.opts['symbol'],
                   view.picture_harmonics.opts['symbol']]
        assert symbols == ['o', 't'], 'told apart by shape'
        dashed = view.picture.corridor[0].opts['pen'].style()
        from PySide6.QtCore import Qt
        assert dashed == Qt.PenStyle.DashLine
    finally:
        widget.close()


def test_the_view_draws_headless(dirty, spec, tmp_path):
    from visualdynamics.plot import plot_tracking_filter

    path = tmp_path / 'band.png'
    plot_tracking_filter(dirty, spec, SineTracking(proportional=0.5),
                         onset=ONSET, cursor_hz=100.0, path=str(path),
                         show=False)
    assert path.exists() and path.stat().st_size > 20000
    with pytest.raises(ValueError, match='not both'):
        plot_tracking_filter(dirty, spec, SineTracking(fixed=5.0),
                             onset=ONSET, cursor=2.0, cursor_hz=100.0,
                             path=str(path), show=False)


def test_rms_and_mean_are_exact_with_few_samples_a_cycle():
    """At a controller's sample rate a cycle near the top of the band
    is a few samples long. Read off the samples, RMS was 3.6 % off at
    10 samples a cycle and mean 9 % at 5.5, on a clean tone (found in
    review, 2026-10-05; the fixtures above run at 51). Read from the
    band-limited waveform over exactly the span, both are exact; the
    peak stays the peak of the samples, by design, and reads low."""
    rate, amplitude = 4096.0, 2.0
    tone = SineTone('Up', 0.5, [100.0, 800.0], [[amplitude]] * 2, [0],
                    [100.0])
    spec = SineSweepSpecification([tone], [DOF], ordinate_unit='m/s**2')
    t = np.arange(int(9.0 * rate)) / rate
    local = np.clip(t - 0.5, 0.0, None)
    inside = (t >= 0.5) & (local <= tone.duration())
    x = np.where(inside, amplitude * np.cos(tone.phase_at(local)), 0.0)
    history = TimeHistory(t, x[None], response_dof=[DOF],
                          ordinate_dim=['acceleration'])
    settings = [SineTracking(detector='rms', proportional=0.1),
                SineTracking(detector='mean', proportional=0.1),
                SineTracking(detector='rms'), SineTracking(detector='mean'),
                SineTracking(detector='peak')]
    levels = track_sine(history, spec, settings, onset=0.5)
    top = (levels[0].abscissa > 400.0) & (levels[0].abscissa < 780.0)
    error = [np.max(np.abs(np.abs(np.asarray(level.ordinate[0]))[top]
                           / amplitude - 1.0)) for level in levels]
    assert error[0] < 5e-4 and error[2] < 5e-4, f'rms {error[0]}, {error[2]}'
    assert error[1] < 1.5e-3 and error[3] < 1.5e-3, f'mean {error[1]}, {error[3]}'
    assert error[4] > 0.05, 'the peak still reads the samples, low up here'
