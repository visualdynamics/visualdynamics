"""The rms check: the rms from the area under an autospectrum beside the
rms of the time history it came from, the gap in dB, and the verdict.

Pinned against hand values (a flat density whose area is known, a sine's
rms), against a random record averaged every way the check has to tell
apart (frames end to end with no window, which is exact; Hann with and
without overlap, which is an estimate within hundredths of a decibel and
carries the frames' own mean square exactly), and against the scaling
errors the check exists to catch, each made on purpose and named: the
window's amplitude correction on a density, the one-sided doubling left
out or applied twice, and a factor of standard gravity. Then the span
and the captures read, what is refused, the project's record of where a
spectrum came from, and the label and plot that write the check.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import Psd, TimeHistory
from visualdynamics.core.rms_check import (
    GAP_DB,
    RmsCheck,
    _coverage,
    matching_records,
    refusal,
    rms_check,
    time_rms,
)
from visualdynamics.core.tone_readout import noise_bandwidth

G = 9.80665
FS = 2048.0


def _record(seconds=16.0, seed=1502, grms=2.0, offset=0.0):
    """Gaussian noise flat from 20 to 800 Hz at `grms` plus `offset`, in
    g, stored as one channel in m/s^2."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / FS)
    spectrum[(f < 20.0) | (f > 800.0)] = 0.0
    x = np.fft.irfft(spectrum, n)
    x = x * grms / np.sqrt(np.mean(x ** 2)) + offset
    t = np.arange(n) / FS
    return TimeHistory(t, x[None] * G, response_dof=['1Z+'],
                       ordinate_dim=['acceleration'], ordinate_unit=['g'])


def _frames(history, block=1024, window='hann', overlap=0.5, **kw):
    """Frames of `block` samples filling the record."""
    return Averaging(frame_length=block, overlap=overlap, window=window,
                     **kw).filled(len(history.abscissa), FS)


def _scaled(psd, factor):
    """The spectrum with every line multiplied by `factor`: a scaling
    error made on purpose."""
    out = copy.deepcopy(psd)
    out.ordinate = psd.ordinate * factor
    return out


def replace_check(check, **changes):
    """The check with `changes` made: a case set up on its numbers."""
    return replace(check, **changes)


@pytest.fixture(scope='module')
def record():
    return _record()


@pytest.fixture(scope='module')
def hann(record):
    """The record's ASD, Hann at half overlap, and its averaging."""
    averaging = _frames(record)
    return record.compute_psds(averaging), averaging


# ---- hand values --------------------------------------------------------

def test_a_flat_density_s_area_is_its_level_times_its_band():
    """101 lines 1 Hz apart at 0.01 per hertz hold 1.01 of mean square:
    every line, the two ends included, stands for a whole bin."""
    f = np.arange(0.0, 101.0)
    check = rms_check(f, np.full(101, 0.01), np.full(400, np.sqrt(1.01)))
    assert check.spectrum_rms == pytest.approx(np.sqrt(1.01), rel=1e-12)
    assert check.time_rms == pytest.approx(np.sqrt(1.01), rel=1e-12)
    assert check.gap_db == pytest.approx(0.0, abs=1e-10)
    assert check.passed and check.lines == 101 and check.spacing == 1.0
    assert check.samples == 400 and check.framed_rms is None


def test_a_sine_s_rms_is_its_amplitude_over_root_two():
    t = np.arange(4096) / 4096.0
    assert time_rms(3.0 * np.sin(2.0 * np.pi * 64.0 * t)) == \
        pytest.approx(3.0 / np.sqrt(2.0), rel=1e-12)
    # captures are pooled sample for sample, whatever their lengths
    assert time_rms([np.full(3, 2.0), np.full(1, 6.0)]) == \
        pytest.approx(np.sqrt((3 * 4.0 + 36.0) / 4.0))


def test_the_gap_is_signed_high_when_the_spectrum_reads_high():
    f = np.arange(0.0, 11.0)
    high = rms_check(f, np.full(11, 2.0), np.full(10, np.sqrt(11.0)))
    assert high.gap_db == pytest.approx(10.0 * np.log10(2.0))
    assert high.verdict.startswith('the spectrum reads 3.01 dB high')
    low = rms_check(f, np.full(11, 0.5), np.full(10, np.sqrt(11.0)))
    assert low.gap_db == pytest.approx(-10.0 * np.log10(2.0))
    assert 'low' in low.verdict


# ---- the record, averaged every way --------------------------------------

def test_frames_end_to_end_with_no_window_agree_exactly(record):
    """Every sample used once at one weight: Parseval holds sample for
    sample."""
    averaging = _frames(record, window='boxcar', overlap=0.0)
    check = record.compute_psds(averaging).rms_check(record,
                                                     averaging=averaging)
    assert check.gap_db == pytest.approx(0.0, abs=1e-9)
    assert check.framed_rms == pytest.approx(check.time_rms, rel=1e-12)
    assert check.time_rms / G == pytest.approx(2.0, rel=1e-9)
    assert check.passed and not check.explained and check.cause is None


@pytest.mark.parametrize('overlap', [0.0, 0.5, 0.75])
def test_a_windowed_average_is_an_estimate_within_hundredths(record,
                                                             overlap):
    """Hann frames weight the record unevenly, so the spectrum is an
    estimate of its mean square: within hundredths of a decibel, and
    closest at half overlap or more, where the windows' squares nearly
    tile the record. Laid end to end they drop each frame's ends, and on
    this 16 s record that reads 0.052 dB low, over the default: the
    check flags it and says the window is the whole of it."""
    averaging = _frames(record, overlap=overlap)
    check = record.compute_psds(averaging).rms_check(record,
                                                     averaging=averaging)
    assert 0.0 < abs(check.gap_db) < 0.1, \
        'an estimate rather than an identity, and a close one'
    assert check.passed or check.explained
    if overlap >= 0.5:
        assert check.passed and abs(check.gap_db) < 0.02
    # and the spectrum carries the frames' own mean square exactly
    assert check.framed_gap_db == pytest.approx(0.0, abs=1e-9)
    assert check.window_gap_db == pytest.approx(check.gap_db, abs=1e-9)
    assert check.window == 'hann' and check.noise_bandwidth == \
        pytest.approx(1.5, rel=1e-3)


def test_a_gap_that_is_all_the_window_s_is_said_to_be(hann, record):
    psd, averaging = hann
    check = psd.rms_check(record, averaging=averaging, threshold_db=1e-4)
    assert not check.passed, 'over a threshold set below the estimate'
    assert check.explained and check.cause is None
    assert "all of it the window's" in check.verdict
    assert 'the scaling is right' in check.verdict


# ---- the scaling errors, made on purpose ---------------------------------

def test_the_amplitude_correction_on_a_density_is_named(hann, record):
    """The commonest error in a hand-written script: Hann's amplitude
    correction squared, 4, where the energy correction is 8/3, which is
    every line 1.5 times too high, its noise bandwidth."""
    psd, averaging = hann
    wrong = _scaled(psd, noise_bandwidth(averaging))
    check = wrong.rms_check(record, averaging=averaging)
    assert check.gap_db == pytest.approx(10.0 * np.log10(1.5), abs=0.02)
    assert not check.passed and not check.explained
    assert check.cause.startswith("the window's amplitude correction "
                                  'applied to a density')
    assert 'hann window passes 1.5 lines' in check.cause
    assert check.verdict.startswith(
        f'the spectrum reads {check.gap_db:.2f} dB high: the window')


def test_the_one_sided_doubling_is_named_either_way(hann, record):
    psd, averaging = hann
    twice = _scaled(psd, 2.0).rms_check(record, averaging=averaging)
    assert twice.gap_db == pytest.approx(3.01, abs=0.03)
    assert twice.cause.startswith('twice the mean square')
    half = _scaled(psd, 0.5).rms_check(record, averaging=averaging)
    assert half.gap_db == pytest.approx(-3.01, abs=0.03)
    assert half.cause.startswith('half the mean square: a two-sided')


def test_a_factor_of_gravity_is_named(hann, record):
    psd, averaging = hann
    check = _scaled(psd, 1.0 / G ** 2).rms_check(record, averaging=averaging)
    assert check.gap_db == pytest.approx(-19.83, abs=0.03)
    assert check.cause == ('a factor of standard gravity: one side in g, '
                           'the other in m/s²')


def test_an_unnamed_gap_is_sent_to_look_for_a_scaling_error(hann, record):
    psd, averaging = hann
    off = 10.0 ** (0.5 / 10.0)                         # half a decibel
    check = _scaled(psd, off).rms_check(record, averaging=averaging)
    assert check.cause is None and not check.explained
    assert check.verdict.endswith('outside 0.05 dB: look for a scaling '
                                  'error before its plot is read')


def test_the_threshold_is_an_argument(record):
    """A gap of 0.03 dB passes the default and fails 0.02."""
    averaging = _frames(record, window='boxcar', overlap=0.0)
    psd = _scaled(record.compute_psds(averaging), 10.0 ** (0.03 / 10.0))
    loose = psd.rms_check(record, averaging=averaging)
    assert loose.gap_db == pytest.approx(0.03, abs=1e-6) and loose.passed
    assert loose.threshold_db == GAP_DB == 0.05
    tight = psd.rms_check(record, averaging=averaging, threshold_db=0.02)
    assert not tight.passed and not tight.explained, \
        'the frames carry no such gap: it is the scaling'
    assert tight.verdict.startswith('the spectrum reads 0.03 dB high')


def test_the_amplitude_error_is_not_named_without_a_window(hann, record):
    """No averaging known with the spectrum's lines, no window, and so
    no noise bandwidth to name the error by."""
    psd, _averaging = hann
    wrong = _scaled(psd, 1.5)
    elsewhere = _frames(record, block=512)
    check = wrong.rms_check(record, averaging=elsewhere)
    assert check.window is None and check.framed_rms is None
    assert check.cause is None
    assert any('was not averaged this way' in note for note in check.notes)


# ---- the span, the captures and the leveling ------------------------------

def test_the_record_is_read_over_the_frames_span_alone():
    history = _record(seconds=8.0)
    loud = history.ordinate.copy()
    loud[0, :int(2 * FS)] *= 10.0              # a loud start, not analyzed
    history = TimeHistory(history.abscissa, loud, response_dof=['1Z+'],
                          ordinate_dim=['acceleration'],
                          ordinate_unit=['g'])
    averaging = Averaging(frame_length=1024, overlap=0.0, window='boxcar',
                          frames=8, start=2.0)
    check = history.compute_psds(averaging).rms_check(history,
                                                      averaging=averaging)
    span = loud[0, int(2 * FS):int(2 * FS) + 8 * 1024]
    assert check.time_rms == pytest.approx(np.sqrt(np.mean(span ** 2)))
    assert check.span == pytest.approx((2.0, 6.0))
    assert check.samples == 8 * 1024
    assert check.gap_db == pytest.approx(0.0, abs=1e-9)


def test_every_capture_of_the_channel_is_pooled():
    one, two = _record(seconds=4.0, seed=1), _record(seconds=4.0, seed=2,
                                                     grms=3.0)
    both = TimeHistory(one.abscissa,
                       np.vstack([one.ordinate, two.ordinate]),
                       response_dof=['1Z+', '1Z+'],
                       ordinate_dim=['acceleration'] * 2,
                       ordinate_unit=['g'] * 2)
    averaging = _frames(both, window='boxcar', overlap=0.0)
    check = both.compute_psds(averaging).rms_check(both, averaging=averaging)
    assert check.time_rms / G == pytest.approx(np.sqrt((4.0 + 9.0) / 2.0),
                                               rel=1e-9)
    assert check.gap_db == pytest.approx(0.0, abs=1e-9)
    assert check.samples == 2 * len(one.abscissa)
    assert '2 captures of the channel' in check.notes[0]


@pytest.mark.parametrize('detrend', ['mean', 'linear'])
def test_a_leveled_average_is_checked_against_a_leveled_span(detrend):
    """A 5 g offset on a 2 grms record: leveled frames carry none of it,
    and neither does the span, so the two still agree."""
    history = _record(seconds=8.0, offset=5.0)
    averaging = _frames(history, window='boxcar', overlap=0.0,
                        detrend=detrend)
    check = history.compute_psds(averaging).rms_check(history,
                                                      averaging=averaging)
    assert check.time_rms / G == pytest.approx(2.0, rel=2e-3)
    assert abs(check.gap_db) < GAP_DB
    assert check.framed_gap_db == pytest.approx(0.0, abs=1e-9)
    assert any(f'leveled ({detrend})' in note for note in check.notes)


def test_an_offset_left_in_is_in_both(record):
    history = _record(seconds=8.0, offset=5.0)
    averaging = _frames(history, window='boxcar', overlap=0.0)
    check = history.compute_psds(averaging).rms_check(history,
                                                      averaging=averaging)
    assert check.time_rms / G == pytest.approx(np.sqrt(4.0 + 25.0),
                                               rel=1e-6)
    assert check.gap_db == pytest.approx(0.0, abs=1e-9)
    assert not check.notes


def test_no_averaging_is_each_record_whole_unwindowed():
    history = _record(seconds=2.0)
    check = history.compute_psds().rms_check(history)
    assert check.window == 'boxcar' and check.span == pytest.approx(
        (0.0, 2.0))
    assert check.gap_db == pytest.approx(0.0, abs=1e-9)


# ---- what is refused -------------------------------------------------------

def test_what_cannot_be_checked_is_refused(hann, record):
    psd, _averaging = hann
    banded = psd.to_octave(3)
    assert 'banded' in refusal(banded)
    with pytest.raises(ValueError, match='banded'):
        banded.rms_check(record)
    f = np.arange(0.0, 64.0)
    cpsd = Psd(f, np.ones((2, 64)), response_dof=['1Z+', '1Z+'],
               reference_dof=['1Z+', '2Z+'],
               ordinate_dim=['acceleration**2/frequency',
                             'acceleration*acceleration/frequency'])
    assert refusal(cpsd, 0) is None
    assert 'cross spectrum' in refusal(cpsd, 1)
    # force against acceleration at a drive point names 1Z+ twice and
    # has no phase when real, and is still not an autospectrum
    drive = Psd(f, np.ones((1, 64)), response_dof=['1Z+'],
                reference_dof=['1Z+'],
                ordinate_dim=['acceleration*force/frequency'])
    assert 'cross spectrum' in refusal(drive, 0)
    elsewhere = Psd(psd.abscissa, psd.ordinate, response_dof=['2X+'],
                    ordinate_dim=['acceleration**2/frequency'])
    with pytest.raises(ValueError, match='no 2X\\+ channel'):
        elsewhere.rms_check(record)
    long = Averaging(frame_length=1024, frames=10_000)
    with pytest.raises(ValueError, match='do not fit'):
        psd.rms_check(record, averaging=long)
    with pytest.raises(ValueError, match='zero or more'):
        rms_check(f, np.ones(64), np.ones(8), threshold_db=-0.1)
    assert np.isnan(time_rms(np.array([np.nan, np.inf]))), \
        'no finite sample, no rms'
    with pytest.raises(ValueError, match='two lines'):
        rms_check([1.0], [1.0], np.ones(8))


def test_a_force_at_the_same_dof_is_not_the_channel():
    t = np.arange(64) / 64.0
    history = TimeHistory(t, np.ones((2, 64)), response_dof=['1Z+', '1Z+'],
                          ordinate_dim=['force', 'acceleration'],
                          ordinate_unit=['N', 'g'])
    assert matching_records(history, '1Z+', 'acceleration**2/frequency') \
        == [1]
    assert matching_records(history, '1Z+', 'force**2/frequency') == [0]
    assert matching_records(history, '1Z+',
                            'acceleration*force/frequency') == []


def test_a_record_with_no_mean_square_has_no_gap():
    f = np.arange(0.0, 11.0)
    check = rms_check(f, np.ones(11), np.zeros(32))
    assert np.isnan(check.gap_db) and not check.passed
    assert check.verdict.startswith('no gap to judge')
    assert 'no gap' in check.describe()


# ---- the project -----------------------------------------------------------

def test_the_project_checks_a_spectrum_against_what_it_was_made_from():
    history = _record(seconds=8.0)
    history.averaging = _frames(history)
    project = visualdynamics.Project()
    project.add('Run', history)
    name = project.compute_psds('Run')
    assert project.rms_source(name) == 'Run'
    made = project.rms_check(name)
    assert made.window == 'hann' and made.passed
    assert made.framed_gap_db == pytest.approx(0.0, abs=1e-9)
    # the record's averaging moves on; the spectrum is still read as it
    # was made, over its own frames
    history.averaging = _frames(history, block=256, window='boxcar')
    again = project.rms_check(name)
    assert again.window == 'hann'
    assert again.framed_gap_db == pytest.approx(0.0, abs=1e-9)


def test_an_imported_spectrum_names_its_record():
    history = _record(seconds=8.0)
    averaging = _frames(history)
    psd = history.compute_psds(averaging)
    project = visualdynamics.Project()
    project.add('Run', history)
    project.add('Controller', Psd(psd.abscissa, psd.ordinate,
                                  response_dof=['1Z+'],
                                  ordinate_dim=['acceleration**2/frequency']))
    assert project.rms_source('Controller') is None
    with pytest.raises(ValueError, match='name the one to check it'):
        project.rms_check('Controller')
    # checked against the run with no averaging known: the run whole,
    # whose lines are not the spectrum's, so the frames are not read
    unknown = project.rms_check('Controller', history='Run')
    assert unknown.framed_rms is None and unknown.passed
    # with the run's averaging set to the one that made it, they are
    history.averaging = averaging
    known = project.rms_check('Controller', history=history)
    assert known.framed_gap_db == pytest.approx(0.0, abs=1e-9)


# ---- the words, the label and the plot -------------------------------------

def test_the_check_describes_itself_in_display_units(hann, record):
    psd, averaging = hann
    check = psd.rms_check(record, averaging=averaging)
    line = check.describe('g', 1.0 / G)
    assert line.startswith(f'{check.spectrum_rms / G:.4g} g rms from the '
                           f'spectrum, {check.time_rms / G:.4g} g rms from '
                           'the time history, ')
    assert line.endswith(f'{check.gap_db:+.3f} dB: agree within 0.05 dB: '
                         "the spectrum carries the record's mean square")


def test_the_label_writes_both_numbers_the_gap_and_the_verdict(hann, record):
    from visualdynamics.plot.rms_check import label_html, label_lines

    psd, averaging = hann
    check = psd.rms_check(record, averaging=averaging)
    lines = label_lines(check, 'g', 1.0 / G)
    assert lines[:4] == [
        'rms check: agree',
        f'from the spectrum: {check.spectrum_rms / G:.4g} g rms',
        f'from the time history: {check.time_rms / G:.4g} g rms',
        f'gap {check.gap_db:+.3f} dB, threshold 0.05 dB']
    assert ' '.join(lines[4:]) == check.verdict
    assert label_lines(check, 'g', 1.0 / G, brief=True) == \
        [lines[0], lines[3]]
    assert label_html(check, 'g', 1.0 / G).startswith(
        '<b>rms check: agree</b><br>from the spectrum')
    # a gap that is the window's says its share
    tight = psd.rms_check(record, averaging=averaging, threshold_db=1e-4)
    said = label_lines(tight, 'g', 1.0 / G)
    assert said[0] == "rms check: the window's gap"
    wrong = _scaled(psd, 2.0).rms_check(record, averaging=averaging)
    assert label_lines(wrong)[0] == 'rms check: flagged'
    assert said[4] == f"the window's share: {tight.window_gap_db:+.3f} dB"


def test_the_label_stays_in_the_corner_and_out_of_the_bounds(hann, record,
                                                             qt_app):
    import pyqtgraph as pg

    from visualdynamics.plot.rms_check import RmsCheckLabel

    psd, averaging = hann
    check = psd.rms_check(record, averaging=averaging)
    plot = pg.PlotItem()
    plot.plot([1.0, 2.0], [1.0, 2.0])
    before = plot.getViewBox().childrenBounds()
    label = RmsCheckLabel(plot, brief=True)
    label.show(check, unit='g', scale=1.0 / G)
    assert label.label.parentItem() is plot.getViewBox()
    assert label.label not in plot.getViewBox().addedItems
    assert plot.getViewBox().childrenBounds() == before
    assert 'rms check: agree' in label.text()
    assert 'from the spectrum' not in label.text(), 'brief'
    label.clear()
    assert label.text() == '' and not label.label.isVisible()
    label.remove()
    assert label.label.parentItem() is None


def test_the_plot_writes_the_check_over_the_spectrum(hann, record, tmp_path,
                                                     qt_app):
    from visualdynamics.plot import plot_rms_check

    psd, averaging = hann
    path = tmp_path / 'rms.png'
    written = plot_rms_check(psd, record, averaging=averaging,
                             path=str(path), show=False)
    assert written == str(path) and path.stat().st_size > 10_000
    pane = plot_rms_check(psd, record, averaging=averaging, show=False)
    graphics = pane.graphics
    plot = next(item for item in graphics.ci.items
                if hasattr(item, 'addItem'))
    labels = [item for item in plot.getViewBox().childItems()
              if isinstance(item, __import__('pyqtgraph').LabelItem)]
    assert len(labels) == 1 and 'rms check: agree' in labels[0].text
    # headroom over the trace for the words
    top = plot.viewRange()[1][1]
    highest = np.log10(np.max(psd.ordinate[0]) / G ** 2)
    assert top > highest + 0.2
    assert pane.windowTitle().startswith('1Z+: rms check, ')


def test_the_result_is_a_plain_record_of_numbers(hann, record):
    psd, averaging = hann
    check = psd.rms_check(record, averaging=averaging)
    assert isinstance(check, RmsCheck)
    with pytest.raises(AttributeError):
        check.time_rms = 1.0


def _undeclared_pair(seconds=4.0):
    """Two channels at 1Z+ with no declared units, told apart only by
    their hints: an acceleration of 2 (raw) rms and a force a million
    times smaller."""
    accel = _record(seconds=seconds, seed=7).ordinate[0] / G
    force = 1e-6 * _record(seconds=seconds, seed=8).ordinate[0] / G
    t = np.arange(accel.size) / FS
    return TimeHistory(t, np.vstack([accel, force]),
                       response_dof=['1Z+', '1Z+'],
                       dimension_hint=['acceleration', 'force'])


def test_two_undeclared_channels_at_one_dof_are_not_pooled():
    """The hint is part of what makes a channel (`channel_key`): an
    acceleration and a force at 1Z+, neither declared, are two channels,
    and a correct acceleration spectrum is not said to be doubled."""
    history = _undeclared_pair()
    averaging = _frames(history, window='boxcar', overlap=0.0)
    psd = history.compute_psds(averaging)
    assert psd.num_records == 2
    for record in (0, 1):
        check = psd.rms_check(history, record, averaging=averaging)
        assert check.gap_db == pytest.approx(0.0, abs=1e-9)
        assert check.passed and check.cause is None
        assert check.samples == history.abscissa.size
        assert not any('captures' in note for note in check.notes)
    assert matching_records(history, '1Z+', psd.ordinate_dim[0],
                            psd.dimension_hint[0]) == [0]
    assert matching_records(history, '1Z+', psd.ordinate_dim[1],
                            psd.dimension_hint[1]) == [1]


def test_an_unhinted_spectrum_against_two_channels_is_ambiguous():
    history = _undeclared_pair()
    psd = history.compute_psds(_frames(history, window='boxcar',
                                       overlap=0.0))
    bare = Psd(psd.abscissa, psd.ordinate[:1], response_dof=['1Z+'])
    assert matching_records(history, '1Z+', bare.ordinate_dim[0]) == []
    with pytest.raises(ValueError, match=r'two channels at 1Z\+'):
        bare.rms_check(history)


def test_a_hint_that_matches_no_channel_is_named_in_the_refusal():
    history = _undeclared_pair()
    f = np.arange(0.0, 1025.0) * 2.0
    velocity = Psd(f, np.ones((1, f.size)), response_dof=['1Z+'],
                   dimension_hint=['velocity**2/frequency'])
    with pytest.raises(ValueError,
                       match=r'what velocity\*\*2/frequency is the') as err:
        velocity.rms_check(history)
    assert 'unknown' not in str(err.value)


def test_a_band_limited_spectrum_is_not_read_as_a_scaling_error(hann,
                                                                 record):
    """A controller's spectrum on 10 to 1000 Hz of a record sampled at
    2048: the record's power outside those lines is not in the area, so
    a gap is not a scaling error and none is named."""
    psd, averaging = hann
    keep = (psd.abscissa >= 10.0) & (psd.abscissa <= 1000.0)
    limited = Psd(psd.abscissa[keep], psd.ordinate[:, keep],
                  response_dof=['1Z+'],
                  ordinate_dim=['acceleration**2/frequency'])
    check = limited.rms_check(record, averaging=averaging)
    assert check.coverage is not None
    assert '10 to 1000 Hz' in check.coverage
    assert check.cause is None and not check.explained
    # the record is 20 to 800 Hz, so this one agrees, but it says why
    assert check.passed
    assert any('outside' in note for note in check.notes)
    # cut inside the record's band, it reads low and says why
    narrow = (psd.abscissa >= 100.0) & (psd.abscissa <= 400.0)
    cut = Psd(psd.abscissa[narrow], psd.ordinate[:, narrow],
              response_dof=['1Z+'],
              ordinate_dim=['acceleration**2/frequency'])
    check = cut.rms_check(record, averaging=averaging)
    assert not check.passed and check.cause is None
    assert 'not a test of its scaling' in check.verdict
    # the full spectrum covers 0 to half the rate
    assert psd.rms_check(record, averaging=averaging).coverage is None
    # half the power cut away reads -3.01 dB, which is not a two-sided
    # density when the lines cover part of the band
    halved = replace_check(check, spectrum_rms=check.time_rms / np.sqrt(2.0))
    assert halved.gap_db == pytest.approx(-3.0103, abs=1e-4)
    assert halved.cause is None
    assert replace_check(halved, coverage=None).cause is not None


@pytest.mark.parametrize('block, cut, short', [
    (1024, slice(None), False), (1024, slice(1, None), True),
    (1024, slice(None, -1), True), (1023, slice(None), False),
    (1023, slice(None, -1), True)])
def test_the_coverage_is_said_a_line_short_at_either_end(block, cut, short):
    """Lines from 0 to half the rate cover it all, and so does an odd
    frame's, whose top line sits half a line under it; without the DC
    line or the top line they are a line short and say so."""
    f = np.fft.rfftfreq(block, 1.0 / FS)[cut]
    assert (_coverage(f, FS) is not None) is short


def test_lines_left_out_of_the_area_are_said(hann, record):
    psd, averaging = hann
    holed = copy.deepcopy(psd)
    holed.ordinate = psd.ordinate.copy()
    holed.ordinate[0, 100] = np.nan
    holed.ordinate[0, 101] = -1.0
    check = holed.rms_check(record, averaging=averaging)
    assert any('2 lines' in note and 'left out' in note
               for note in check.notes)
    # a line of exactly nothing is a line of the area, not one left out
    quiet = copy.deepcopy(psd)
    quiet.ordinate = psd.ordinate.copy()
    quiet.ordinate[0, 100] = 0.0
    check = quiet.rms_check(record, averaging=averaging)
    assert not any('left out' in note for note in check.notes)


def test_samples_not_finite_are_left_out_and_counted():
    """The rule `time_statistics` reads a record by: a sample that is not
    finite is left out and counted, not a refusal of the whole record."""
    from visualdynamics.core.time_statistics import time_statistics

    x = np.array([1.0, np.nan, -1.0, 1.0, np.inf])
    assert time_rms(x) == pytest.approx(1.0)
    assert time_statistics(x, 4.0).rms == pytest.approx(time_rms(x))
    check = rms_check(np.arange(0.0, 11.0), np.ones(11), x)
    assert any('2 samples' in note and 'not finite' in note
               for note in check.notes)


def test_the_window_s_gap_does_not_promise_a_longer_record_helps(hann,
                                                                 record):
    psd, averaging = hann
    check = replace_check(psd.rms_check(record, averaging=averaging),
                          threshold_db=0.0)
    assert check.explained
    assert 'longer record' not in check.verdict


def test_a_band_limited_gap_is_not_said_to_be_the_window_s(hann, record):
    """The frames' own mean square agrees, but lines that do not cover
    the record cannot say the window is the whole of a gap."""
    psd, averaging = hann
    check = replace_check(psd.rms_check(record, averaging=averaging),
                          threshold_db=0.0)
    assert check.explained
    limited = replace_check(check, coverage='its lines run 10 to 1000 Hz')
    assert not limited.explained


@pytest.mark.parametrize('off, named', [(0.09, True), (0.11, False),
                                        (0.5, False), (-0.09, True)])
def test_a_cause_is_named_only_within_a_tenth_of_it(hann, record, off,
                                                    named):
    psd, averaging = hann
    factor = 10.0 ** ((10.0 * np.log10(2.0) + off) / 10.0)
    check = psd.rms_check(record, averaging=averaging)
    doubled = replace_check(check, spectrum_rms=check.time_rms
                            * np.sqrt(factor), framed_rms=None)
    assert (doubled.cause is not None) is named


def test_a_negative_record_is_refused(hann, record):
    psd, averaging = hann
    with pytest.raises(ValueError, match='record -1'):
        psd.rms_check(record, -1, averaging=averaging)
    assert 'record 5' in refusal(psd, 5)
    assert f'record {psd.num_records}' in refusal(psd, psd.num_records)


# ---- dead, not finite, empty and late records -------------------------------

def _one_channel(x, start=0.0):
    """`x`, in g, as a one-channel record at 1Z+ whose clock starts at
    `start`."""
    t = start + np.arange(len(x)) / FS
    return TimeHistory(t, np.asarray(x, dtype=float)[None] * G,
                       response_dof=['1Z+'], ordinate_dim=['acceleration'],
                       ordinate_unit=['g'])


def test_a_dead_channel_is_not_judged_and_says_why():
    history = _one_channel(np.zeros(int(8 * FS)))
    averaging = _frames(history)
    check = history.compute_psds(averaging).rms_check(history,
                                                      averaging=averaging)
    assert not check.passed and check.cause is None
    assert check.verdict == ('no gap to judge: the record is zero over the '
                             'span read')
    from visualdynamics.plot.rms_check import heading
    assert heading(check) == 'rms check: not judged'


def test_one_sample_not_finite_is_named_as_the_cause(record):
    x = record.ordinate[0] / G
    x[1000] = np.nan
    history = _one_channel(x)
    averaging = _frames(history)
    check = history.compute_psds(averaging).rms_check(history,
                                                      averaging=averaging)
    assert not check.passed and not check.explained and check.cause is None
    assert check.verdict.startswith(
        "no gap to judge: none of the spectrum's lines is a finite number")
    assert '1 sample of the record is not finite' in ' '.join(check.notes)


def test_a_spectrum_of_zeros_and_a_record_of_nans_are_each_named():
    f = np.arange(0.0, 11.0)
    assert rms_check(f, np.zeros(11), np.ones(32)).verdict == (
        "no gap to judge: the spectrum's lines are all zero")
    assert rms_check(f, np.ones(11), np.full(32, np.nan)).verdict == (
        "no gap to judge: none of the record's samples is a finite number")


def test_an_empty_record_and_an_empty_spectrum_are_refused():
    with pytest.raises(ValueError, match='no samples'):
        rms_check(np.arange(0.0, 11.0), np.ones(11), np.array([]))
    with pytest.raises(ValueError, match='two lines or more'):
        rms_check(np.array([1.0]), np.ones(1), np.ones(32))


def test_a_record_whose_clock_starts_late_reads_the_same(record):
    x = record.ordinate[0] / G
    early, late = _one_channel(x), _one_channel(x, start=3.25)
    averaging = Averaging(frame_length=1024, frames=20, start=1.0)
    first = early.compute_psds(averaging).rms_check(early,
                                                    averaging=averaging)
    second = late.compute_psds(averaging).rms_check(late,
                                                    averaging=averaging)
    assert second.span == pytest.approx((3.25 + 1.0, 3.25 + 11.0))
    assert first.span == pytest.approx((1.0, 11.0))
    assert second.gap_db == pytest.approx(first.gap_db, abs=1e-12)
    assert second.time_rms == pytest.approx(first.time_rms, rel=1e-12)


def test_only_the_checked_channel_is_framed(monkeypatch, record):
    """The other channels of a many-channel record are not read, so they
    are not framed: on a long record framing them was most of the time."""
    x = record.ordinate[0] / G
    two = TimeHistory(record.abscissa, np.vstack([x, 2.0 * x]) * G,
                      response_dof=['1Z+', '2Z+'],
                      ordinate_dim=['acceleration'] * 2,
                      ordinate_unit=['g'] * 2)
    averaging = _frames(two)
    psd = two.compute_psds(averaging)
    framed = []
    original = TimeHistory._spectral_frame

    def spy(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        framed.append(set(out[2]))
        return out

    monkeypatch.setattr(TimeHistory, '_spectral_frame', spy)
    check = psd.rms_check(two, 1, averaging=averaging)
    assert framed == [{two.channel_key(1)}]
    assert check.passed and check.dof == '2Z+'


def test_white_noise_at_the_defaults_is_never_called_a_scaling_error():
    """Correctly scaled spectra of white noise under the default window
    (Hann, frames end to end): the gap scatters with the window's
    weighting, and over 0.05 dB it is said to be the window's, never
    flagged and never given a cause."""
    flagged, windows = 0, 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        history = _one_channel(rng.standard_normal(int(8 * FS)))
        averaging = Averaging(frame_length=1024).filled(
            len(history.abscissa), FS)
        check = history.compute_psds(averaging).rms_check(
            history, averaging=averaging)
        assert check.window == 'hann'
        assert check.cause is None
        flagged += not (check.passed or check.explained)
        windows += check.explained
    assert flagged == 0
    assert windows > 0, 'the default threshold is reached by the window alone'


# ---- the survivors of tools/mutate.py, each pinned -------------------------

def test_a_boxcar_names_no_amplitude_correction():
    """A rectangle's noise bandwidth is one line, so its amplitude
    correction is no correction and is never the cause of a gap."""
    f = np.arange(0.0, 11.0)
    values = np.full(11, 1.0 / 10.5)
    # the record read 0.05 dB under the spectrum's own rms
    carried = rms_check(f, values, np.ones(8)).spectrum_rms
    record = np.ones(64) * carried / 10.0 ** (0.05 / 20.0)
    check = rms_check(f, values, record, threshold_db=0.01, window='boxcar')
    assert check.gap_db == pytest.approx(0.05, abs=1e-9)
    assert not check.passed and check.noise_bandwidth == pytest.approx(1.0)
    assert check.cause is None


def test_one_finite_sample_has_an_rms_and_a_zero_threshold_is_allowed():
    assert time_rms(np.array([np.nan, -3.0, np.inf])) == pytest.approx(3.0)
    f = np.arange(0.0, 11.0)
    assert rms_check(f, np.ones(11), np.ones(8),
                     threshold_db=0.0).threshold_db == 0.0


def test_two_lines_are_enough_for_an_area():
    check = rms_check([0.0, 1.0], [1.0, 1.0], np.ones(8))
    assert check.lines == 2 and np.isfinite(check.gap_db)
    two = Psd(np.array([0.0, 1.0]), np.ones((1, 2)), response_dof=['1Z+'],
              ordinate_dim=['acceleration**2/frequency'])
    assert refusal(two) is None


def test_one_line_left_out_is_said_in_the_singular():
    f = np.arange(0.0, 11.0)
    values = np.ones(11)
    values[3] = np.nan
    check = rms_check(f, values, np.ones(8))
    assert check.notes == (('1 line of the spectrum is not finite or below '
                            'zero, and left out of its area'),)


def test_a_declared_spectrum_names_its_own_quantity_in_the_refusal():
    t = np.arange(64) / 64.0
    history = TimeHistory(t, np.ones((1, 64)), response_dof=['1Z+'],
                          ordinate_dim=['acceleration'], ordinate_unit=['g'])
    f = np.arange(0.0, 33.0)
    force = Psd(f, np.ones((1, f.size)), response_dof=['1Z+'],
                ordinate_dim=['force**2/frequency'])
    with pytest.raises(ValueError, match=r'what force\*\*2/frequency is'):
        force.rms_check(history)


def test_the_ambiguity_names_each_channel_s_hint():
    history = _undeclared_pair()
    psd = history.compute_psds(_frames(history, window='boxcar',
                                       overlap=0.0))
    bare = Psd(psd.abscissa, psd.ordinate[:1], response_dof=['1Z+'])
    with pytest.raises(ValueError, match=r'\(acceleration, force\)'):
        bare.rms_check(history)


def test_the_first_record_is_checked_when_none_is_named(hann, record):
    from visualdynamics.core.rms_check import check_record

    psd, averaging = hann
    assert check_record(psd, record, averaging=averaging) == psd.rms_check(
        record, 0, averaging=averaging)


def test_a_fine_spectrum_a_line_short_at_the_bottom_is_said_to_be():
    """At half-hertz lines the first line at 0.5 Hz is one short of DC."""
    assert _coverage(np.arange(1.0, 2049.0) * 0.5, 2048.0) == (
        'its lines run 0.5 to 1024 Hz of the 0 to 1024 Hz the record holds')
    assert _coverage(np.arange(0.0, 2049.0) * 0.5, 2048.0) is None
