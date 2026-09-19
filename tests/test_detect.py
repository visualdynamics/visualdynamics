"""Finding the stretch of a record worth averaging.

A run holds the shaker coming up, a reduced-level check, the full level,
and whatever was still recording afterwards. These hold that the right
stretch comes out, that the number of averages comes from the record
rather than from a number chosen in advance, and that the compression to
one level per hop — which is what makes it quick enough for three
hundred channels — does not throw away what the answer depends on.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from conftest import fixture_path

from visualdynamics.core import detect
from visualdynamics.core.data import TimeHistory

RATE = 256.0


def history(profile, channels=8, seed=0, rate=RATE, offset=0.0):
    """A record stepping through (amplitude, seconds) stretches."""
    rng = np.random.default_rng(seed)
    parts = [amplitude * rng.standard_normal((channels, int(seconds * rate)))
             for amplitude, seconds in profile]
    ordinate = np.concatenate(parts, axis=1) + offset
    t = np.arange(ordinate.shape[1]) / rate
    return TimeHistory(abscissa=t, ordinate=ordinate,
                       response_dof=[f'{101 + i}Z+' for i in range(channels)],
                       ordinate_dim=['acceleration'] * channels)


def seconds(history, window):
    return tuple(round(edge / history.sample_rate, 1) for edge in window)


# ---- the frame, and the number of averages ------------------------------


def test_a_frame_is_about_a_second_and_a_power_of_two():
    """A power of two for the FFT, a second because that is the
    resolution a random vibration test is usually specified at."""
    assert detect.frame_length_for(256) == 256
    assert detect.frame_length_for(4096) == 4096
    assert detect.frame_length_for(1000) == 1024, 'the nearest power of two'
    assert detect.frame_length_for(48000) == 65536


def test_the_number_of_averages_comes_from_the_record():
    """Not from a number chosen in advance. Twice the settled record is
    twice the averages, and nothing had to be told so."""
    short = detect.suggest(history([(0.1, 5), (1.0, 40), (0.1, 5)]))
    long = detect.suggest(history([(0.1, 5), (1.0, 120), (0.1, 5)]))
    assert long.frames > 2.5 * short.frames
    assert short.frames >= 30, 'and both are worth averaging'


def test_a_record_too_short_for_the_target_still_answers():
    """Better a short average than none. The caller can see how many it
    got and decide whether to believe it."""
    averaging = detect.suggest(history([(1.0, 6)]))
    assert averaging.frames >= 1
    assert averaging.fits(6 * int(RATE), RATE)


# ---- which stretch ------------------------------------------------------


def test_the_full_level_stretch_is_the_one_found():
    data = history([(0.1, 20), (0.5, 20), (1.0, 60), (0.0, 20)])
    assert seconds(data, detect.analysis_window(data, keep=(0.0, 1.0))) == (
        pytest.approx(40.0, abs=1.0), pytest.approx(100.0, abs=1.0))


def test_an_overload_does_not_become_the_analysis():
    """A ten-second burst in the middle of a ninety-second test is the
    highest level in the record and the last thing worth averaging.
    Duration comes first, level second."""
    data = history([(0.1, 20), (1.0, 40), (3.0, 10), (1.0, 40)])
    first, last = detect.analysis_window(data, keep=(0.0, 1.0))
    assert seconds(data, (first, last)) == (pytest.approx(20.0, abs=1.0),
                                            pytest.approx(110.0, abs=1.0))
    assert (last - first) / RATE > 60, 'the test, not the burst'


def test_a_lower_tail_is_left_out():
    data = history([(1.0, 50), (0.3, 40)])
    _first, last = detect.analysis_window(data, keep=(0.0, 1.0))
    assert last / RATE == pytest.approx(50.0, abs=1.0)


def test_the_highest_of_several_rising_levels_wins():
    data = history([(0.1, 30), (0.5, 30), (1.0, 50)])
    first, _last = detect.analysis_window(data, keep=(0.0, 1.0))
    assert first / RATE == pytest.approx(60.0, abs=1.0)


# ---- noise floor data is data ------------------------------------------


def test_a_noise_floor_capture_is_not_refused():
    """Noise floor is collected on purpose, alongside the system ID, and
    the point of finding a window in it is to be able to put a PSD on
    it. A detector that held out for excitation would refuse the one
    record whose whole purpose is to have none."""
    data = history([(1e-4, 60)])
    averaging = detect.suggest(data)
    assert averaging.frames >= 30
    psd = data.compute_psds(averaging)
    assert np.all(np.isfinite(psd.ordinate.real))
    assert psd.ordinate.real.max() > 0


def test_the_answer_does_not_depend_on_the_absolute_level():
    """What the dB scale buys. In power, scatter grows with level, so
    the flattest stretch of a record is reliably its quietest — the
    opposite of what is wanted — and a noise floor capture would be
    judged by a different standard from a full-level one."""
    quiet = history([(1e-5, 20), (1e-4, 60), (1e-5, 20)], seed=7)
    loud = history([(1e-5 * 1e6, 20), (1e-4 * 1e6, 60), (1e-5 * 1e6, 20)],
                   seed=7)
    assert (detect.analysis_window(quiet)
            == detect.analysis_window(loud))


def test_a_record_of_nothing_at_all_is_taken_whole():
    """Every channel flat. There is no level to find, and reporting a
    floor would be inventing one."""
    t = np.arange(2048) / RATE
    data = TimeHistory(abscissa=t, ordinate=np.zeros((4, 2048)),
                       response_dof=['101Z+', '102Z+', '103Z+', '104Z+'],
                       ordinate_dim=['acceleration'] * 4)
    assert detect.analysis_window(data) == (0, 2048)
    assert detect.suggest(data).frames >= 1


# ---- the settling trim --------------------------------------------------


def test_the_settled_middle_is_what_is_kept():
    """Closed-loop control does not arrive at a level at once — under
    match-trace it walks in over several frames — so the front of a
    stretch is not yet the level the rest of it is."""
    assert detect.trim(0, 100) == (20, 90)
    assert detect.trim(50, 150) == (70, 140)


def test_more_is_trimmed_from_the_front_than_the_back():
    """The settling is at the start; the tail only risks the level
    coming off early."""
    first, last = detect.trim(0, 100)
    assert first - 0 > 100 - last


def test_the_trim_shows_up_in_the_window():
    data = history([(0.1, 10), (1.0, 100), (0.1, 10)])
    whole = detect.analysis_window(data, keep=(0.0, 1.0))
    kept = detect.analysis_window(data)
    assert kept[0] > whole[0] and kept[1] < whole[1]
    span, inner = whole[1] - whole[0], kept[1] - kept[0]
    assert inner / span == pytest.approx(0.70, abs=0.02)


def test_a_stretch_too_short_to_trim_is_used_whole():
    """Better the whole of a short stretch than nothing of it."""
    assert detect.trim(10, 11) == (10, 11)
    assert detect.trim(0, 0) == (0, 0)


def test_the_trim_is_the_callers_to_set():
    data = history([(0.1, 10), (1.0, 60), (0.1, 10)])
    whole = detect.analysis_window(data, keep=(0.0, 1.0))
    half = detect.analysis_window(data, keep=(0.25, 0.75))
    assert (half[1] - half[0]) / (whole[1] - whole[0]) == pytest.approx(
        0.5, abs=0.02)


def test_what_the_trim_refuses():
    for bad in ((0.9, 0.2), (-0.1, 0.5), (0.2, 1.5)):
        with pytest.raises(ValueError, match='fractions'):
            detect.trim(0, 100, bad)


# ---- what the compression must not lose ---------------------------------


def test_one_hot_channel_does_not_become_the_answer():
    """A set whose levels span orders of magnitude — a stiff mount and a
    panel — is compressed to a consensus, not to a portrait of the
    loudest channel. Each channel is normalized by its own median first.
    """
    data = history([(0.1, 20), (1.0, 60), (0.1, 20)], channels=8, seed=3)
    wanted = detect.analysis_window(data)
    # one channel a thousand times hotter, and stepping the other way
    loud = history([(1000.0, 20), (100.0, 60), (1000.0, 20)],
                   channels=1, seed=4)
    mixed = TimeHistory(
        abscissa=data.abscissa,
        ordinate=np.vstack([data.ordinate, loud.ordinate]),
        response_dof=[*data.response_dof, '999Z+'],
        ordinate_dim=[*data.ordinate_dim, 'acceleration'])
    assert detect.analysis_window(mixed) == wanted


def test_a_channel_that_drops_out_moves_nothing():
    """The consensus is the median across channels, so a dead cable is
    one vote and not a level change."""
    data = history([(0.1, 20), (1.0, 60), (0.1, 20)], channels=9, seed=5)
    wanted = detect.analysis_window(data)
    broken = data.ordinate.copy()
    broken[0, int(40 * RATE):] = 0.0        # a cable pulled mid-test
    dropped = TimeHistory(abscissa=data.abscissa, ordinate=broken,
                          response_dof=list(data.response_dof),
                          ordinate_dim=list(data.ordinate_dim))
    assert detect.analysis_window(dropped) == wanted


def test_a_dc_offset_is_not_a_level():
    """The mean is removed inside each hop, so a bias on a channel — or
    a slow thermal drift — is not read as excitation."""
    plain = history([(0.1, 20), (1.0, 60), (0.1, 20)], seed=6)
    biased = history([(0.1, 20), (1.0, 60), (0.1, 20)], seed=6, offset=50.0)
    assert detect.analysis_window(biased) == detect.analysis_window(plain)


def test_a_flat_stretch_scatters_no_more_than_it_must():
    """The chi-squared floor is what the flatness test is measured
    against, so it has to be the real one: a stationary stretch of a
    record really does scatter about this much and no more."""
    data = history([(1.0, 120)], channels=16, seed=11)
    level = detect.hop_levels(data.ordinate, 128)
    assert np.std(level) < 1.5 * detect.scatter_floor(128)
    assert np.std(level) > 0.2 * detect.scatter_floor(128)


# ---- the real record ----------------------------------------------------


STRESS_RUN = fixture_path('..', 'stressdata', 'plate', 'random.nc4')


@pytest.mark.skipif(not os.path.exists(STRESS_RUN),
                    reason='stressdata is not in the repository')
def test_the_real_run_lands_on_its_full_level_stretch():
    """The run steps -6, -3 and 0 dB and then stops. Full level is
    61-160 s, and what comes back is the settled middle of it.

    The record is 21 MB of streamed time data and lives outside the
    repository, so this is skipped wherever it has not been generated.
    """
    import visualdynamics

    data = next(v for v in visualdynamics.import_file(STRESS_RUN).values()
                if type(v).__name__ == 'TimeHistory')
    averaging = detect.suggest(data)
    rate = data.sample_rate
    assert averaging.frame_length == 256, 'one second at 256 Hz'
    assert averaging.window == 'hann' and averaging.overlap == 0.5
    # inside the full-level stretch, not on either edge of it. Which 
    # part of it is a judgment the numbers should not be pinned to;
    # that it is all of one level, and enough of it, is the claim.
    assert averaging.start >= 61.0, averaging.start
    assert averaging.stop(rate) <= 160.0, averaging.stop(rate)
    assert averaging.frames >= 30, averaging.frames
    assert averaging.fits(len(data.abscissa), rate)


@pytest.mark.skipif(not os.path.exists(STRESS_RUN),
                    reason='stressdata is not in the repository')
def test_most_of_the_full_level_stretch_is_found():
    """What the smoothing buys, and the only thing that shows it. The
    full-level stretch of a closed-loop run wanders by as much as the
    step to the next level, so read hop by hop it breaks into fragments
    and a fifth of it comes back. Median-filtered over a few seconds it
    comes out nearly whole — which is averages, and averages are the
    point."""
    import visualdynamics

    data = next(v for v in visualdynamics.import_file(STRESS_RUN).values()
                if type(v).__name__ == 'TimeHistory')
    first, last = detect.analysis_window(data, keep=(0.0, 1.0))
    rate = data.sample_rate
    covered = (min(last / rate, 160.0) - max(first / rate, 61.0)) / 99.0
    assert covered > 0.75, f'only {covered:.0%} of the full-level stretch'


@pytest.mark.skipif(not os.path.exists(STRESS_RUN),
                    reason='stressdata is not in the repository')
def test_the_real_run_is_not_averaged_across_a_level_change():
    """The point of the exercise. Every frame has to be at one level, so
    the window must not straddle 61 s, where the run steps to full."""
    import visualdynamics

    data = next(v for v in visualdynamics.import_file(STRESS_RUN).values()
                if type(v).__name__ == 'TimeHistory')
    rate = data.sample_rate
    for edge in (31.0, 61.0, 163.0):
        window = detect.suggest(data)
        assert not (window.start < edge < window.stop(rate)), (
            f'the window straddles the step at {edge} s')


def test_a_short_test_beats_a_long_quiet():
    """Eight seconds at level inside eighty of quiet hold no thirty frames
    at level, and the sweep used to carry on down to the quiet and find
    thirty frames of noise floor there — the exact mistake the exercise
    is meant to avoid, caught by the importer's guard until the import
    became the Detect answer (2026-09-19). The sweep stops a margin under
    the top; the test is found, with the frames it holds."""
    data = history([(0.05, 40), (1.0, 8), (0.05, 40)])
    first, last = detect.analysis_window(data, keep=(0.0, 1.0))
    assert seconds(data, (first, last)) == (pytest.approx(40.0, abs=1.5),
                                            pytest.approx(48.0, abs=1.5))
    found = detect.suggest(data)
    assert 1 <= found.frames < 30 and 39.0 <= found.start <= 42.0
