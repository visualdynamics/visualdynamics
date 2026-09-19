"""Cutting a time history into frames before averaging it.

Five independent numbers — start, frame length, overlap, window, frame
count — and everything else derived from them. The math is held to
scipy's Welch, which is the reference every text agrees on; the rest of
these hold the derivations and the refusals.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core.averaging import Averaging, normalize_window, window_shape
from visualdynamics.core.data import TimeHistory

RATE = 1024.0
SAMPLES = 8192


def noisy_tone(channels=1, seed=3):
    t = np.arange(SAMPLES) / RATE
    rng = np.random.default_rng(seed)
    rows = [np.sin(2 * np.pi * 97 * t) + 0.4 * rng.standard_normal(SAMPLES)
            for _ in range(channels)]
    return TimeHistory(abscissa=t, ordinate=np.array(rows),
                       response_dof=[f'10{i + 1}X+' for i in range(channels)],
                       ordinate_dim=['acceleration'] * channels)


@pytest.mark.parametrize('window', ['hann', 'hamming', 'blackman',
                                    'flattop', 'boxcar'])
@pytest.mark.parametrize('overlap', [0.0, 0.5, 0.75])
def test_it_agrees_with_welch(window, overlap):
    """The whole point of the exercise. Every window, every overlap."""
    welch = pytest.importorskip('scipy.signal').welch
    get_window = pytest.importorskip('scipy.signal').get_window

    history = noisy_tone()
    averaging = Averaging(frame_length=1024, overlap=overlap,
                          window=window, frames=1).filled(SAMPLES, RATE)
    ours = history.compute_psds(averaging).ordinate[0].real
    _f, theirs = welch(
        history.ordinate[0], fs=RATE,
        window=get_window('boxcar' if window == 'boxcar' else window,
                          1024, fftbins=True),
        nperseg=1024, noverlap=1024 - averaging.hop, nfft=1024,
        detrend=False, scaling='density', average='mean')
    assert np.allclose(ours, theirs, rtol=1e-12, atol=0)


def test_no_averaging_is_every_record_its_own_frame():
    """What it always did, said explicitly: a stored capture is already
    an average, so the two spellings have to agree exactly."""
    history = noisy_tone()
    implied = history.compute_psds()
    spelled = history.compute_psds(Averaging.for_records(SAMPLES))
    assert np.allclose(implied.ordinate, spelled.ordinate)


def test_frames_are_pooled_across_records():
    """A burst-random run of 20 captures cut three ways is 60 averages,
    not 3 and not 20."""
    t = np.arange(1024) / RATE
    rng = np.random.default_rng(5)
    rows = np.array([rng.standard_normal(1024) for _ in range(4)])
    history = TimeHistory(abscissa=t, ordinate=rows,
                          response_dof=['101X+'] * 4,
                          ordinate_dim=['acceleration'] * 4)
    averaging = Averaging(frame_length=256, overlap=0.0, window='hann',
                          frames=4)
    _f, _scale, groups = history._spectral_frame(averaging)
    assert next(iter(groups.values())).shape == (16, 256), '4 records x 4'


def test_a_window_is_periodic_not_symmetric():
    """The spectral convention — scipy's fftbins=True — where a frame is
    one period of an assumed-repeating signal."""
    get_window = pytest.importorskip('scipy.signal').get_window

    for name in ('hann', 'hamming', 'blackman', 'flattop'):
        assert np.allclose(window_shape(name, 64),
                           get_window(name, 64, fftbins=True))


def test_a_window_is_named_however_a_file_names_it():
    assert normalize_window('Hann') == 'hann'
    assert normalize_window('hanning') == 'hann'
    # scipy's spelling is the canonical one since 2026-08-28; the
    # engineering spelling stays an alias
    assert normalize_window('rectangle') == 'boxcar'
    assert normalize_window('boxcar') == 'boxcar'
    assert normalize_window(None) == 'boxcar', 'no window is no window'
    with pytest.raises(ValueError, match='unknown window'):
        normalize_window('gaussian')


def test_the_hop_and_the_span_follow_from_the_rest():
    averaging = Averaging(frame_length=1000, overlap=0.5, frames=4)
    assert averaging.hop == 500
    assert averaging.span == 1000 + 3 * 500
    assert averaging.stop(1000.0) == pytest.approx(2.5)


def test_the_stop_time_moves_with_the_start():
    averaging = Averaging(frame_length=1000, overlap=0.0, frames=2, start=1.0)
    assert averaging.stop(1000.0) == pytest.approx(3.0)


def test_the_frames_are_where_the_plot_will_shade_them():
    bounds = Averaging(frame_length=1000, overlap=0.5,
                       frames=3).frame_bounds(1000.0)
    assert bounds == [(0.0, 1.0), (0.5, 1.5), (1.0, 2.0)]


def test_only_whole_frames_are_counted():
    """What the drag handle snaps to: room for most of a frame is room
    for no frame."""
    averaging = Averaging(frame_length=1000, overlap=0.0, frames=1)
    assert averaging.most_frames(2999, 1000.0) == 2
    assert averaging.most_frames(3000, 1000.0) == 3
    assert averaging.most_frames(999, 1000.0) == 0


def test_filling_takes_as_many_as_fit():
    averaging = Averaging(frame_length=1000, overlap=0.5, frames=1)
    assert averaging.filled(3000, 1000.0).frames == 5


def test_asking_for_more_than_there_is_says_how_many_there_are():
    history = noisy_tone()
    averaging = Averaging(frame_length=1024, overlap=0.0, frames=99)
    with pytest.raises(ValueError, match='room for 8'):
        history.compute_psds(averaging)


@pytest.mark.parametrize('bad, message', [
    ({'frame_length': 1}, 'at least two samples'),
    ({'overlap': 1.0}, 'fraction of a frame'),
    ({'overlap': -0.1}, 'fraction of a frame'),
    ({'frames': 0}, 'at least one frame'),
    ({'start': -1.0}, 'cannot start before'),
])
def test_what_it_refuses(bad, message):
    fields = {'frame_length': 512, 'overlap': 0.5, 'frames': 2, 'start': 0.0}
    with pytest.raises(ValueError, match=message):
        Averaging(**{**fields, **bad})


def test_cpsds_average_the_same_way():
    """The diagonal of a windowed CPSD is the windowed PSD, as it is
    without any windowing."""
    history = noisy_tone(channels=2)
    averaging = Averaging(frame_length=1024, overlap=0.5,
                          window='hann', frames=1).filled(SAMPLES, RATE)
    psds = history.compute_psds(averaging)
    cpsds = history.compute_cpsds(averaging)
    for k, (response, reference) in enumerate(zip(cpsds.response_dof,
                                                  cpsds.reference_dof)):
        if response != reference:
            continue
        row = list(psds.response_dof).index(response)
        assert np.allclose(cpsds.ordinate[k], psds.ordinate[row])


def test_the_start_time_actually_moves_the_analysis():
    """Otherwise every parameter but one would be decoration."""
    history = noisy_tone()
    early = Averaging(frame_length=1024, overlap=0.0, window='hann', frames=2)
    late = Averaging(frame_length=1024, overlap=0.0, window='hann', frames=2,
                     start=4.0)
    assert not np.allclose(history.compute_psds(early).ordinate,
                           history.compute_psds(late).ordinate)


# ---- what the file itself says ------------------------------------------


def imported(name):
    from conftest import fixture_path

    import visualdynamics

    loaded = visualdynamics.import_file(fixture_path('plate', name))
    return next(value for value in loaded.values()
                if type(value).__name__ == 'TimeHistory')


def test_a_modal_run_brings_its_own_averaging():
    """samples_per_frame, num_averages, overlap, frf_window."""
    averaging = imported('modal.nc4').averaging
    assert averaging.frame_length == 2048
    assert averaging.frames == 20
    assert averaging.overlap == 0.0
    assert averaging.window == 'boxcar'


def test_a_random_run_brings_the_control_loop_settings():
    """samples_per_frame, cpsd_overlap, cpsd_window — and 'Hann'
    normalized. The count is not the file's frames_in_cpsd (10, the
    controller's running buffer) but what the run's full-level stretch
    holds: the import answers as Detect does (Brandon, 2026-09-19)."""
    history = imported('random.nc4')
    averaging = history.averaging
    assert averaging.frame_length == 2048
    assert averaging.overlap == 0.5
    assert averaging.window == 'hann'
    assert averaging.frames > 10, 'the stretch holds more than the buffer did'
    assert history.suggest_averaging() == averaging, \
        'the import is the Detect answer, exactly'


def test_the_system_id_settings_count_too():
    """A file that describes only its system identification still says
    how it averaged, and that is worth reading."""
    from visualdynamics.io.rattlesnake import AVERAGING_KEYS

    assert ('sysid_frame_size', 'sysid_averages',
            'sysid_overlap', 'sysid_window') in AVERAGING_KEYS


def test_the_frame_count_is_trimmed_to_what_was_saved(tmp_path):
    """A control loop may average hundreds of frames over a run whose
    saved capture holds a handful. Taking the stated count at its word
    would make an imported history refuse to compute at all — the very
    thing reading the parameters was meant to make easy."""
    from test_rattlesnake_files import write_nc4

    import visualdynamics

    # 4096 samples of 512 is room for eight frames; the file claims 500
    path = write_nc4(str(tmp_path / 'greedy.nc4'), 4096, spectral=False,
                     per_frame=512, averages=500)
    loaded = visualdynamics.import_file(path)
    history = next(value for value in loaded.values()
                   if type(value).__name__ == 'TimeHistory')
    assert history.averaging.frames == 8, 'trimmed to what is there'
    assert history.compute_psds().num_records == 2, 'and it computes'


def test_computing_uses_what_the_file_said():
    """No argument means the history's own averaging, which is the whole
    point of reading it: the calculator computes with what is set."""
    history = imported('random.nc4')
    explicit = history.compute_psds(history.averaging)
    implied = history.compute_psds()
    assert np.allclose(implied.ordinate, explicit.ordinate)
    assert len(implied.abscissa) == history.averaging.frame_length // 2 + 1


# ---- a capture that is already cut into frames --------------------------


def test_a_frame_per_record_run_is_recognized_as_already_split():
    """The controller saved each average as its own record, so the frame
    length and the count are the file's, not the user's."""
    history = imported('modal_spectra.nc4')
    assert history.split_into_frames
    assert history.averaging.frame_length == len(history.abscissa)
    assert history.averaging.frames == 1, 'one frame per record'
    assert history.averaging.overlap == 0.0, 'nothing to overlap with'


@pytest.mark.parametrize('name', ['modal.nc4', 'random.nc4'])
def test_a_continuous_capture_is_not_already_split(name):
    """A long record is the case the five parameters exist for; calling
    it split would freeze every one of them."""
    assert not imported(name).split_into_frames


def test_one_whole_record_is_not_split_however_it_is_described():
    """The state a freshly imported history starts in: one record per
    DOF, the whole of it one frame. That is not a capture cut into
    averages, it is a capture nobody has cut yet — and freezing its
    parameters would take away the only thing worth doing to it.
    """
    history = noisy_tone()
    assert not history.split_into_frames, 'no averaging set'
    history.averaging = Averaging.for_records(SAMPLES)
    assert not history.split_into_frames, 'and none implied'


def test_the_average_count_comes_from_the_records():
    """Twenty captures is twenty averages, for every channel alike."""
    history = imported('modal_spectra.nc4')
    assert history.average_counts == (20, 20)
    counts = history.records_per_channel
    assert set(counts.values()) == {20}
    assert len(counts) == 13, '13 channels across 260 records'


def test_a_drive_point_is_two_channels_and_not_one():
    """The one that got this wrong. 101Z+ and 1004Z+ carry a force
    record and an acceleration record apiece, so counting by DOF alone
    finds 11 channels and credits those two with twice the averages — a
    number that is true of nothing. What is measured is part of the
    channel.
    """
    history = imported('modal_spectra.nc4')
    counts = history.records_per_channel
    at_drive = {key: n for key, n in counts.items() if key[0] == '101Z+'}
    assert sorted(key[1] for key in at_drive) == ['acceleration', 'force']
    assert set(at_drive.values()) == {20}, 'twenty each, not forty between'
    assert len({key[0] for key in counts}) == 11, '11 DOFs, 13 channels'


def test_counting_channels_and_averaging_them_agree():
    """They are the same question asked twice, and used to be answered
    by two copies of the key."""
    history = imported('modal_spectra.nc4')
    _f, _scale, groups = history._spectral_frame()
    assert set(groups) == set(history.records_per_channel)
    assert all(len(groups[key]) == history.records_per_channel[key]
               for key in groups)


def test_a_continuous_capture_counts_its_frames_per_record():
    """One record each, so the count is what the averaging asks for."""
    history = imported('random.nc4')
    frames = history.averaging.frames
    assert history.average_counts == (frames, frames)


def test_the_window_is_still_the_users_when_the_frames_are_fixed():
    """The one parameter left to choose. Without it a saved capture
    could only ever be averaged unwindowed."""
    history = imported('modal_spectra.nc4')
    samples = len(history.abscissa)
    plain = history.compute_psds(Averaging.for_records(samples))
    hann = history.compute_psds(Averaging.for_records(samples, 'hann'))
    assert not np.allclose(plain.ordinate, hann.ordinate), (
        'the window reached the math')


def test_a_record_shaped_averaging_leaves_nothing_else_to_choose():
    averaging = Averaging.for_records(512, 'blackman')
    assert (averaging.frames, averaging.overlap, averaging.start) == (1, 0.0,
                                                                      0.0)
    assert averaging.window == 'blackman'
    assert averaging.span == 512, 'the frame is the record'


def test_an_averaging_survives_the_round_trip(tmp_path):
    from visualdynamics import io

    history = noisy_tone()
    history.averaging = Averaging(frame_length=512, overlap=0.25,
                                  window='flattop', frames=3, start=0.5)
    path = tmp_path / 'history.vdyn'
    io.save(history, str(path))
    back = io.load(str(path))
    assert back.averaging == history.averaging


# ---- where in the record to start ---------------------------------------


def rattlesnake_run(path, profile, kind=1, per_frame=512, averages=5,
                    overlap=0.5, window='hann', rate=256.0, channels=4):
    """A Rattlesnake-shaped random run stepping through levels.

    `profile` is [(amplitude, seconds)]; `kind` is the controller's own
    EnvironmentType — 1 random, 6 modal.
    """
    import netCDF4

    rng = np.random.default_rng(2)
    data = np.concatenate(
        [amp * rng.standard_normal((channels, int(secs * rate)))
         for amp, secs in profile], axis=1)
    with netCDF4.Dataset(path, 'w', format='NETCDF4') as ds:
        ds.sample_rate = rate
        ds.createDimension('response_channels', channels)
        ds.createDimension('time_samples', data.shape[1])
        ds.createVariable('time_data', 'f8',
                          ('response_channels', 'time_samples'))[...] = data
        group = ds.createGroup('channels')
        for name, values in (('node_number',
                              [str(100 + i) for i in range(channels)]),
                             ('node_direction', ['Z+'] * channels),
                             ('unit', ['m/s^2'] * channels)):
            group.createVariable(name, str, ('response_channels',))[:] = \
                np.array(values, dtype=object)
        ds.createDimension('num_environments', 1)
        ds.createVariable('environment_names', str,
                          ('num_environments',))[0] = 'Env'
        ds.createVariable('environment_types', int,
                          ('num_environments',))[0] = kind
        env = ds.createGroup('Env')
        env.samples_per_frame = per_frame
        env.frames_in_cpsd = averages
        env.cpsd_overlap = overlap
        env.cpsd_window = window
    return str(path)


def loaded(path):
    import visualdynamics

    return next(value for value in visualdynamics.import_file(path).values()
                if type(value).__name__ == 'TimeHistory')


def test_a_random_run_starts_where_the_record_is_worth_averaging(tmp_path):
    """The file settles the frame length, the count, the overlap and the
    window. The one thing it never says is when — and left at zero the
    analysis starts on the shaker coming up, which is the one stretch of
    a run that is certainly not the test."""
    path = rattlesnake_run(tmp_path / 'random.nc4',
                           [(0.05, 20), (1.0, 60), (0.05, 20)])
    averaging = loaded(path).averaging
    assert averaging.start > 20.0, 'past the low-level opening'
    assert averaging.stop(256.0) < 80.0, 'and done before the level drops'


def test_the_recipe_is_the_controllers_and_the_count_is_the_stretchs(tmp_path):
    """The frame length, overlap and window are the controller's own
    account of the run and outrank anything worked out from the
    samples; the count is as many frames as the full-level stretch
    holds — the file's five were its running buffer, not the record
    (Brandon, 2026-09-19)."""
    path = rattlesnake_run(tmp_path / 'random.nc4',
                           [(0.05, 20), (1.0, 60), (0.05, 20)],
                           per_frame=512, averages=5, overlap=0.5,
                           window='hann')
    averaging = loaded(path).averaging
    assert averaging.frame_length == 512
    assert averaging.overlap == 0.5
    assert averaging.window == 'hann'
    assert averaging.frames > 5, 'sixty seconds at level hold far more than five'
    assert averaging.stop(256.0) <= 80.0 + 2.0, 'and none of it past the level drop'


def test_a_modal_run_is_left_where_it_was(tmp_path):
    """A modal survey averages bursts, and a burst record has no
    stationary stretch to find. Asked for one on the survey's modal
    capture the detector returns sixteen seconds holding four of the
    twenty averages the file asked for — worse than starting at zero."""
    path = rattlesnake_run(tmp_path / 'modal.nc4', [(0.05, 20), (1.0, 60)],
                           kind=6)
    assert loaded(path).averaging.start == 0.0
    assert imported('modal.nc4').averaging.start == 0.0


def test_a_short_stretch_is_used_with_the_frames_it_holds(tmp_path):
    """Until 2026-09-19 a stretch short of the file's count was not
    used at all, and the analysis stayed at zero with the shaker coming
    up in it. The stretch is what is at level; the count follows it."""
    # 8 s of full level cannot hold 20 frames of 2 s at half overlap
    path = rattlesnake_run(tmp_path / 'brief.nc4',
                           [(0.05, 40), (1.0, 8), (0.05, 40)],
                           per_frame=512, averages=20)
    averaging = loaded(path).averaging
    assert 38.0 <= averaging.start <= 42.0, 'onto the eight seconds at level'
    assert 1 <= averaging.frames < 20, 'as many as fit, no more'
    assert averaging.frame_length == 512


def test_detect_keeps_the_recipe_a_record_carries():
    """Detect works out the start and the count; the frame length,
    window, overlap and detrend stay the controller's (Brandon,
    2026-09-19). A bare record gets the detector's own recipe."""
    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.data import TimeHistory

    rate = 1024.0
    t = np.arange(int(rate * 30)) / rate
    rng = np.random.default_rng(4)
    level = np.where((t > 5.0) & (t < 25.0), 1.0, 0.05)
    history = TimeHistory(
        t, rng.standard_normal((2, len(t))) * level, response_dof=['1Z+', '2Z+'],
        ordinate_dim='acceleration')
    bare = history.suggest_averaging()
    history.averaging = Averaging(frame_length=2048, overlap=0.25,
                                  window='rectangle', frames=3, detrend='mean')
    kept = history.suggest_averaging()
    assert (kept.frame_length, kept.overlap, kept.window, kept.detrend) == \
        (2048, 0.25, normalize_window('rectangle'), 'mean')
    assert kept.frames > 3 and 5.0 <= kept.start <= 10.0, 'on the level stretch, trimmed at the front'
    assert bare.frame_length != 2048 or bare.window != 'rectangle', \
        'the bare record was given the detector\'s own recipe'
    assert history.suggest_averaging(window='hann').window == 'hann', \
        'an explicit choice still wins'


def test_the_moved_analysis_still_fits_the_record(tmp_path):
    path = rattlesnake_run(tmp_path / 'random.nc4',
                           [(0.05, 20), (1.0, 60), (0.05, 20)])
    history = loaded(path)
    assert history.averaging.fits(len(history.abscissa), history.sample_rate)
    history.compute_psds()          # and it computes without refusing


def test_a_file_with_no_averaging_at_all_still_has_none(tmp_path):
    """Nothing to move."""
    import netCDF4

    path = rattlesnake_run(tmp_path / 'bare.nc4', [(1.0, 30)])
    with netCDF4.Dataset(path, 'a') as ds:
        for name in ('samples_per_frame', 'frames_in_cpsd'):
            ds.groups['Env'].delncattr(name)
    assert loaded(path).averaging is None


def test_detrend_levels_each_frame_before_its_window():
    """scipy parity (Brandon, 2026-08-29): 'none' is the convention
    here and in sdynpy — the oracle pins it — 'mean' is what scipy's
    welch does by default, 'linear' takes a drift line out. The DC
    bin says which happened."""
    from visualdynamics.core.data import TimeHistory

    rng = np.random.default_rng(5)
    t = np.arange(8192) / 1024.0
    signal = rng.standard_normal(len(t)) + 7.0 + 0.5 * t
    history = TimeHistory(t, np.atleast_2d(signal),
                          response_dof=['101Z+'],
                          ordinate_dim='acceleration')

    def dc(detrend):
        history.averaging = Averaging(frame_length=1024, frames=8,
                                      window='hann', detrend=detrend)
        return float(history.compute_psds().ordinate[0][0].real)

    raw, mean, linear = dc('none'), dc('mean'), dc('linear')
    assert raw > 10.0, 'the offset and drift dominate the DC bin'
    assert mean < raw * 1e-3, 'the mean is gone'
    assert linear < mean, 'and the drift line with it'


def test_detrend_mean_matches_scipy_welch():
    """The whole point of offering it under scipy's meaning: same
    frames, same window, same leveling, same numbers."""
    from scipy.signal import welch

    from visualdynamics.core.data import TimeHistory

    rng = np.random.default_rng(9)
    rate = 2048.0
    t = np.arange(16384) / rate
    signal = rng.standard_normal(len(t)) + 3.0
    history = TimeHistory(t, np.atleast_2d(signal),
                          response_dof=['101Z+'],
                          ordinate_dim='acceleration')
    history.averaging = Averaging(frame_length=2048, frames=15,
                                  overlap=0.5, window='hann',
                                  detrend='mean')
    ours = history.compute_psds()
    frequencies, theirs = welch(signal, fs=rate, window='hann',
                                nperseg=2048, noverlap=1024,
                                detrend='constant')
    assert np.allclose(ours.abscissa, frequencies)
    assert np.allclose(ours.ordinate[0].real, theirs, rtol=1e-10)


def test_the_blackman_harris_window_joined_under_scipys_name():
    from scipy.signal import get_window

    assert normalize_window('blackman-harris') == 'blackmanharris'
    assert np.array_equal(window_shape('blackmanharris', 512),
                          get_window('blackmanharris', 512, fftbins=True))


def test_detrend_survives_the_file(tmp_path):
    import visualdynamics
    from visualdynamics.core.data import TimeHistory

    t = np.arange(4096) / 1024.0
    history = TimeHistory(t, np.atleast_2d(np.sin(2 * np.pi * 50 * t)),
                          response_dof=['101Z+'],
                          ordinate_dim='acceleration')
    history.averaging = Averaging(frame_length=1024, frames=4,
                                  detrend='linear')
    project = visualdynamics.Project('D')
    project.add('Run', history)
    back = visualdynamics.Project.open(
        project.save(tmp_path / 'detrend.vdyn'))
    assert back['Run'].averaging.detrend == 'linear'


def test_kaiser_psds_match_scipy_welch_end_to_end():
    """The parameter travels the whole pipeline: same frames, same
    (kaiser, beta) window, same numbers as scipy's welch."""
    from scipy.signal import welch

    from visualdynamics.core.data import TimeHistory

    rng = np.random.default_rng(3)
    rate = 2048.0
    signal = rng.standard_normal(16384)
    history = TimeHistory(np.arange(len(signal)) / rate,
                          np.atleast_2d(signal),
                          response_dof=['101Z+'],
                          ordinate_dim='acceleration')
    history.averaging = Averaging(frame_length=2048, frames=15,
                                  overlap=0.5, window='kaiser',
                                  window_parameter=8.0)
    ours = history.compute_psds()
    _f, theirs = welch(signal, fs=rate, window=('kaiser', 8.0),
                       nperseg=2048, noverlap=1024, detrend=False)
    assert np.allclose(ours.ordinate[0].real, theirs, rtol=1e-10)


def test_the_window_parameter_survives_the_file(tmp_path):
    import visualdynamics
    from visualdynamics.core.data import TimeHistory

    t = np.arange(4096) / 1024.0
    history = TimeHistory(t, np.atleast_2d(np.sin(2 * np.pi * 50 * t)),
                          response_dof=['101Z+'],
                          ordinate_dim='acceleration')
    history.averaging = Averaging(frame_length=1024, frames=4,
                                  window='tukey', window_parameter=0.3)
    project = visualdynamics.Project('W')
    project.add('Run', history)
    back = visualdynamics.Project.open(
        project.save(tmp_path / 'tukey.vdyn'))
    assert back['Run'].averaging.window == 'tukey'
    assert back['Run'].averaging.window_parameter == 0.3
