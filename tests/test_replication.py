"""How closely a transient replicated the waveform it was controlled to.

A transient specification is one waveform played over and over, so the
comparison begins by cutting the record back into those repeats. Where
they are is derived, not read: the file describes the repeats nowhere,
and the parameters it *does* carry describe a different measurement.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import (
    ShockSpecification,
    TimeHistory,
    TransientSpecification,
)
from visualdynamics.core.replication import (
    SRS_FLOOR_DB,
    ZERO_TARGET_DB,
    averaging_for,
    compare,
    lag_of,
)
from visualdynamics.core.replication import leftover as rep_leftover
from visualdynamics.core.replication import playings as rep_playings

RATE = 2048.0
FRAME = 1024


pytestmark = pytest.mark.usefixtures('flat_reading')

def _pulse(samples=FRAME, onset=None, peak=50.0, decay=90.0, ring=180.0):
    """A decaying impulse response — the shape a transient target is."""
    onset = samples // 8 if onset is None else onset
    t = np.arange(samples) / RATE
    out = np.zeros(samples)
    after = t[onset:] - t[onset]
    out[onset:] = peak * np.exp(-decay * after) * np.sin(2 * np.pi * ring
                                                         * after)
    return out


def _target(dofs=('1Z+', '2Z+'), scales=(1.0, 0.4)):
    rows = np.array([_pulse() * s for s in scales])
    spec = TransientSpecification(np.arange(FRAME) / RATE, rows,
                                  response_dof=list(dofs))
    spec.define_units('m/s^2')
    return spec


def _measured(spec, repeats=4, gain=1.0, lag=0, extra=0, noise=None):
    """The target played `repeats` times, optionally shifted and scaled."""
    rows = []
    for row in spec.ordinate:
        record = np.tile(row * gain, repeats)
        if extra:
            record = np.concatenate([record, row[:extra] * gain])
        rows.append(record)
    rows = np.asarray(rows)
    if lag:
        rows = np.concatenate([np.zeros((rows.shape[0], lag)), rows], axis=1)
    if noise is not None:
        rows = rows + noise
    history = TimeHistory(np.arange(rows.shape[1]) / RATE, rows,
                          response_dof=list(spec.response_dof))
    history.define_units('m/s^2')
    return history


# --- where the repeats are ------------------------------------------

def test_the_frame_is_the_specification_and_nothing_else_is_guessed():
    """Four of the five averaging numbers are settled by construction.

    A repeat plays the control signal, so the frame is its length; the
    repeats run back to back, so there is no overlap; a taper would
    distort the waveform being replicated, so the window is
    rectangular; and the record starts where it starts. Only the count
    is measured, and measuring it is division.
    """
    spec = _target()
    averaging = averaging_for(_measured(spec, repeats=4), spec)
    assert averaging == Averaging(frame_length=FRAME, overlap=0.0,
                                  window='rectangle', frames=4, start=0.0)


def test_a_part_played_repeat_is_not_an_average():
    """A profile whose stop lands near a frame boundary leaves a
    fraction of a repeat on the end. A fraction of a frame is not an
    average, and `Averaging` will not hold one either."""
    spec = _target()
    history = _measured(spec, repeats=3, extra=FRAME // 2)
    averaging = averaging_for(history, spec)
    assert averaging.frames == 3, 'the half repeat is not a fourth average'
    assert averaging.span == 3 * FRAME
    assert history.ordinate.shape[1] - averaging.span == FRAME // 2


def test_a_record_shorter_than_one_repeat_has_no_averaging():
    spec = _target()
    short = TimeHistory(np.arange(FRAME // 2) / RATE,
                        spec.ordinate[:, :FRAME // 2],
                        response_dof=list(spec.response_dof))
    assert averaging_for(short, spec) is None
    assert compare(short, spec) == []


# --- alignment ------------------------------------------------------

@pytest.mark.parametrize('shift', [0, 1, 3, 7])
def test_the_shift_between_target_and_record_is_recovered(shift):
    spec = _target()
    history = _measured(spec, repeats=3, lag=shift)
    assert lag_of(history, spec) == shift


def test_one_shift_is_found_for_every_channel_together():
    """The delay belongs to the acquisition, not to a channel.

    Given its own shift each channel would hunt for the alignment that
    flattered it most and report the error it found there, which is a
    measurement of the search rather than of the test.
    """
    spec = _target()
    history = _measured(spec, repeats=3, lag=4)
    # wreck the second channel; the shared shift must not move
    history.ordinate[1] = np.roll(history.ordinate[1], 9)
    assert lag_of(history, spec) == 4


def test_alignment_is_applied_before_anything_is_compared():
    spec = _target()
    shifted = _measured(spec, repeats=3, lag=5)
    aligned = compare(shifted, spec)
    assert max(r['waveform'] for r in aligned) < 1e-6
    ignored = compare(shifted, spec, lag=0)
    assert min(r['waveform'] for r in ignored) > 10.0, (
        'unaligned, a perfect replication reads as a bad one')


def test_a_channel_whose_target_is_silent_cannot_drag_the_shift():
    spec = _target(scales=(1.0, 0.0))
    history = _measured(spec, repeats=3, lag=2)
    assert lag_of(history, spec) == 2


# --- the metrics ----------------------------------------------------

def test_a_perfect_replication_scores_zero_on_all_three():
    spec = _target()
    rows = compare(_measured(spec, repeats=4), spec)
    assert [r['label'] for r in rows] == ['1Z+'] * 4 + ['2Z+'] * 4, (
        'every repeat is reported and none is singled out')
    assert sorted({r['frame'] for r in rows}) == [0, 1, 2, 3]
    for row in rows:
        assert row['waveform'] == pytest.approx(0.0, abs=1e-9)
        assert row['srs'] == pytest.approx(0.0, abs=1e-9)
        assert row['level'] == pytest.approx(0.0, abs=1e-9)


def test_level_reads_a_scaling_and_waveform_reads_it_too():
    """Doubling every sample is 6.02 dB, and the two readings part
    company: the level is exactly the scaling, while the waveform error
    is the whole target's worth of difference."""
    spec = _target()
    rows = compare(_measured(spec, repeats=3, gain=2.0), spec)
    for row in rows:
        assert row['level'] == pytest.approx(20 * np.log10(2.0), abs=1e-6)
        assert row['srs'] == pytest.approx(20 * np.log10(2.0), abs=1e-6)
        assert row['waveform'] == pytest.approx(100.0, abs=1e-6)


def test_level_is_blind_to_a_shape_the_waveform_error_catches():
    """The two readings earn their places by disagreeing.

    Reversing a frame in time keeps every sample and so keeps the
    level exactly; the waveform is a different waveform, and says so.
    """
    spec = _target()
    history = _measured(spec, repeats=2)
    history.ordinate[:, FRAME:2 * FRAME] = history.ordinate[:, :FRAME][:, ::-1]
    for row in compare(history, spec, frame=1):
        assert row['level'] == pytest.approx(0.0, abs=1e-9), (
            'the same samples in another order hold the same power')
        assert row['waveform'] > 50.0, 'and are not the same waveform'


def test_every_repeat_is_reported_and_none_is_singled_out():
    """Which playing was the bad one is a judgment about what the
    article is for, not a measurement. An earlier version reduced these
    to the worst repeat and labeled it so."""
    spec = _target()
    history = _measured(spec, repeats=4)
    history.ordinate[:, 2 * FRAME:3 * FRAME] *= 2.0
    rows = compare(history, spec)
    for row in rows:
        expected = 100.0 if row['frame'] == 2 else 0.0
        assert row['waveform'] == pytest.approx(expected, abs=1e-6)
    assert not any(key.endswith('_frame') for row in rows for key in row), (
        'no reading carries a repeat it decided was the worst')


# --- what cannot be scored -------------------------------------------

def test_a_target_that_asks_for_nothing_is_not_scored_against():
    """Every metric here is a ratio against the target. Against a
    target of nothing they do not become large so much as stop meaning
    anything: the survey's lateral control channel is 230 dB below
    the loudest and read 1.3e12 percent before this existed."""
    spec = _target(scales=(1.0, 0.0))
    rows = compare(_measured(spec, repeats=3), spec)
    quiet = next(r for r in rows if r['label'] == '2Z+')
    assert quiet['zero_target'] is True
    for key in ('waveform', 'srs', 'level'):
        assert np.isnan(quiet[key])
    loud = next(r for r in rows if r['label'] == '1Z+')
    assert loud['zero_target'] is False
    assert loud['waveform'] == pytest.approx(0.0, abs=1e-9), (
        'and the silent channel does not spoil the one beside it')


def test_the_silence_threshold_is_relative_to_the_loudest_target():
    """Quiet is not silent. A control channel genuinely asked for at a
    low level is still asked for, and is scored."""
    quiet = 10.0 ** (-(ZERO_TARGET_DB - 20.0) / 20.0)
    spec = _target(scales=(1.0, quiet))
    rows = compare(_measured(spec, repeats=3), spec)
    assert not any(r['zero_target'] for r in rows)


def test_the_srs_is_compared_only_where_the_target_asks_for_something():
    """The bottom of an SRS band answers what a very low oscillator did
    during a short record, which for a shock is residual drift rather
    than the shock. On the survey the unmasked band reported 21 dB of
    deviation from a target 80 dB below its own peak; masked, the same
    channel reads 6.9 dB, which is the real answer.
    """
    spec = _target()
    history = _measured(spec, repeats=2)
    # a slow drift, well below the ringing the target is made of
    t = np.arange(history.ordinate.shape[1]) / RATE
    history.ordinate += 0.2 * np.sin(2 * np.pi * 2.2 * t)
    masked = max(abs(r['srs']) for r in compare(history, spec))

    from visualdynamics.core import replication
    original = replication.SRS_FLOOR_DB
    try:
        replication.SRS_FLOOR_DB = 1e6          # compare everywhere
        everywhere = max(abs(r['srs']) for r in compare(history, spec))
    finally:
        replication.SRS_FLOOR_DB = original
    assert everywhere > 10.0, (
        'unmasked, the drift is read as a replication failure')
    assert masked < everywhere - 5.0, (
        'the mask drops the bands where the target asks for nothing')
    assert original == SRS_FLOOR_DB


# --- against the real run -------------------------------------------

def _transient_file(path, repeats=3, frame=128, rate=256.0, extra=0):
    """A streamed transient run, carrying the `sysid_*` trap.

    Built here rather than taken from a fixture so the numbers going in
    are known — and so the sysid attributes are certainly present, this
    being the thing the importer must not read.
    """
    import netCDF4

    signal = np.stack([_pulse(frame, peak=50.0),
                       _pulse(frame, peak=20.0)])
    record = np.tile(signal, repeats)
    if extra:
        record = np.concatenate([record, signal[:, :extra]], axis=1)
    with netCDF4.Dataset(path, 'w') as ds:
        ds.sample_rate = rate
        ds.createDimension('response_channels', 2)
        ds.createDimension('time_samples', record.shape[1])
        ds.createDimension('num_environments', 1)
        channels = ds.createGroup('channels')
        for name, values in (('node_number', ['101', '111']),
                             ('node_direction', ['Z+', 'Z+']),
                             ('channel_type', ['acceleration'] * 2),
                             ('unit', ['m/s^2'] * 2)):
            variable = channels.createVariable(name, str,
                                               ('response_channels',))
            for i, value in enumerate(values):
                variable[i] = value
        names = ds.createVariable('environment_names', str,
                                  ('num_environments',))
        names[0] = 'Drop'
        kinds = ds.createVariable('environment_types', 'i8',
                                  ('num_environments',))
        kinds[0] = 2                       # ENVIRONMENT_KINDS: transient
        data = ds.createVariable('time_data', 'f8',
                                 ('response_channels', 'time_samples'))
        data[:] = record
        group = ds.createGroup('Drop')
        # exactly the shelf Rattlesnake writes, describing the random
        # excitation it ran beforehand and not this test at all
        group.sysid_frame_size = frame // 4
        group.sysid_averages = 10
        group.sysid_overlap = 0.5
        group.sysid_window = 'Hann'
        group.createDimension('specification_channels', 2)
        group.createDimension('signal_samples', frame)
        control = group.createVariable(
            'control_signal', 'f8',
            ('specification_channels', 'signal_samples'))
        control[:] = signal
        indices = group.createVariable('control_channel_indices', 'i4',
                                       ('specification_channels',))
        indices[:] = [0, 1]
    return signal


def test_a_rattlesnake_transient_run_reads_its_own_frames(tmp_path):
    """The imported record carries the averaging its repeats imply,
    not the one the file's `sysid_*` attributes describe.

    Those attributes are the trap: Hann, half overlap and a frame a
    quarter of the true one. They belong to the random excitation
    Rattlesnake runs beforehand to measure the FRF it inverts, and the
    importer would have taken them — they are the third entry in
    `AVERAGING_KEYS` and nothing else in a transient file answers.
    """
    path = tmp_path / 'run.nc4'
    _transient_file(path, repeats=3, frame=128)
    project = visualdynamics.Project()
    project.import_file(str(path))
    history = next(obj for obj in project.values()
                   if type(obj) is TimeHistory)
    spec = next(obj for obj in project.values()
                if isinstance(obj, TransientSpecification))
    assert history.averaging == Averaging(
        frame_length=128, overlap=0.0, window='rectangle', frames=3)
    assert history.averaging == averaging_for(history, spec)
    assert history.averaging.frame_length != 32, 'that is the sysid frame'
    assert history.averaging.window != 'hann', 'that is the sysid window'


def test_the_imported_run_compares_against_its_own_specification(tmp_path):
    """End to end: the file goes in, the three readings come out."""
    path = tmp_path / 'run.nc4'
    _transient_file(path, repeats=3, frame=128)
    project = visualdynamics.Project()
    project.import_file(str(path))
    history = next(obj for obj in project.values()
                   if type(obj) is TimeHistory)
    spec = next(obj for obj in project.values()
                if isinstance(obj, TransientSpecification))
    rows = compare(history, spec)
    assert [r['label'] for r in rows] == ['101Z+'] * 3 + ['111Z+'] * 3
    for row in rows:
        assert row['waveform'] == pytest.approx(0.0, abs=1e-9)
        assert row['srs'] == pytest.approx(0.0, abs=1e-9)
        assert row['level'] == pytest.approx(0.0, abs=1e-9)


def test_a_random_run_still_reads_its_own_averaging():
    """The transient route must not capture a file that has no
    transient in it."""
    project = visualdynamics.Project()
    project.import_file(fixture_path('plate', 'random.nc4'))
    history = next(obj for obj in project.values()
                   if type(obj) is TimeHistory)
    assert history.averaging is not None
    assert history.averaging.window != 'rectangle' or (
        history.averaging.overlap != 0.0)


# --- as bars ---------------------------------------------------------

def _bars(mode, history, spec):
    from visualdynamics.report import _build_block

    block = {'kind': 'bars', 'mode': mode, 'source': 'spec',
             'measured': 'time', 'caption': 'c'}
    return _build_block(block, {'spec': spec, 'time': history},
                        visualdynamics.SI, [])


def test_the_waveform_error_draws_as_bars():
    spec = _target()
    built = _bars('waveform', _measured(spec, repeats=3, gain=2.0), spec)
    assert built['kind'] == 'bars'
    assert built['labels'] == ['1Z+ e1', '1Z+ e2', '1Z+ e3',
                               '2Z+ e1', '2Z+ e2', '2Z+ e3'], (
        'a bar per channel per playing, the repeat named because there '
        'is more than one to tell apart')
    assert all(v == pytest.approx(100.0, abs=1e-4) for v in built['values'])


def test_one_repeat_needs_no_repeat_in_its_label():
    spec = _target()
    built = _bars('waveform', _measured(spec, repeats=1), spec)
    assert built['labels'] == ['1Z+', '2Z+']


@pytest.mark.parametrize('mode', ['srs', 'level'])
def test_only_the_waveform_error_is_read_from_the_time_data(mode):
    """A level wants two PSDs and an SRS deviation two spectra. Both
    are reached by computing them and comparing the pair — where a
    specification's own PSD is a Specification — so offering them here
    too would be two ways to one number, agreeing only by luck."""
    spec = _target()
    assert _bars(mode, _measured(spec, repeats=3), spec) is None


def test_a_one_sided_reading_carries_its_threshold_where_the_page_looks():
    """The page tells one-sided from two-sided by `high` being null,
    and reads the only threshold out of `low`. A waveform error put in
    `high` would shade nothing at all."""
    from visualdynamics.core.replication import WAVEFORM_PERCENT

    spec = _target()
    history = _measured(spec, repeats=3)
    waveform = _bars('waveform', history, spec)
    assert waveform['high'] is None
    assert waveform['low'] == WAVEFORM_PERCENT
    assert waveform['floor'] == 0.0, 'an error in percent cannot go below 0'


def test_an_unscorable_channel_is_named_rather_than_dropped_silently():
    """A bar of zero would read as a channel that replicated perfectly,
    which is the opposite of what is known about it."""
    spec = _target(scales=(1.0, 0.0))
    built = _bars('waveform', _measured(spec, repeats=1), spec)
    assert built['labels'] == ['1Z+'], 'no bar it cannot earn'
    assert '2Z+' in built['caption']
    assert 'asked for nothing' in built['caption']


def test_an_unknown_reading_binds_to_nothing():
    spec = _target()
    assert _bars('trac', _measured(spec, repeats=3), spec) is None


def test_the_transient_report_asks_for_the_waveform_error(tmp_path):
    """The template's bars must bind on a real imported run, not just
    build in isolation."""
    from visualdynamics.core.report import transient_template
    from visualdynamics.report import _build_block

    path = tmp_path / 'run.nc4'
    _transient_file(path, repeats=3, frame=128)
    project = visualdynamics.Project()
    project.import_file(str(path))
    report = transient_template(project, links=project.links)
    bars = [b for b in report.blocks if b['kind'] == 'bars']
    # no 'srs': that judgment moved to the shock report, where the
    # SRS is the test's own measure rather than a borrowed one
    # error first: spectral data and its RMS reading precede the
    # waveform judgment in the standing order (Brandon, 2026-08-23)
    assert [b['mode'] for b in bars] == ['error', 'waveform',
                                         'kurtosis']
    built = _build_block(bars[1], dict(project), visualdynamics.SI, project.links)
    assert built is not None
    assert built['labels'] == ['101Z+ e1', '101Z+ e2', '101Z+ e3',
                               '111Z+ e1', '111Z+ e2', '111Z+ e3']
    # the spectral pair binds once the spectra exist, and not before
    assert _build_block(bars[0], dict(project), visualdynamics.SI,
                        project.links) is None

    # the overlay: the first playing over its target, channel by channel
    overlay = next(b for b in report.blocks
                   if b.get('mode') == 'overlay')
    built = _build_block(overlay, dict(project), visualdynamics.SI,
                         project.links)
    assert built is not None
    assert built['logy'] is False, 'waveforms draw on linear axes'
    assert [ch['label'] for ch in built['channels']] == ['101Z+', '111Z+']
    assert 'playing 1 of 3' in built['curves'][1]['label']
    assert all(ch['zones'] == [] for ch in built['channels']), (
        'a target waveform carries no limits to shade')

    # the SRS deviation bars live in the shock template now — bound
    # here against the same run once its spectra exist, because a
    # transient run is the shock template's data too once the engineer
    # declares it one
    from visualdynamics.core.report import shock_template

    for source in list(project):
        if type(project[source]).__name__ in ('TimeHistory',
                                              'TransientSpecification'):
            project.compute_srs(source)
    shock = shock_template(project, links=project.links)
    srs_bars = next(b for b in shock.blocks
                    if b.get('kind') == 'bars' and b.get('mode') == 'srs')
    srs = _build_block(srs_bars, dict(project), visualdynamics.SI,
                       project.links)
    assert srs is not None
    # the SRS reading, not the PSD one falling through: a spectrum has
    # no area to integrate, so an RMS-error-over-a-band label here would
    # mean the wrong comparison had been made
    assert srs['ylabel'] == 'SRS deviation, RMS across the band [dB]'
    # both sides: the deviation is signed now — an under-hit shock used
    # to read as an over-test (Brandon, 2026-08-20)
    from visualdynamics.core.compliance import SRS_ERROR_DB

    assert srs['low'] == -SRS_ERROR_DB and srs['high'] == SRS_ERROR_DB
    assert 'floor' not in srs, 'a signed deviation reaches below zero'
    assert len(srs['labels']) == 2 * 3, 'a channel per playing'


# --- on screen -------------------------------------------------------

def _load_transient(window, tmp_path, repeats=3, frame=128):
    path = tmp_path / 'run.nc4'
    _transient_file(path, repeats=repeats, frame=frame)
    window.import_paths([str(path)])
    window.tree.clearSelection()
    for name, obj in window.objects.items():
        if isinstance(obj, (TimeHistory, TransientSpecification)):
            window._item_for_object(name).setSelected(True)
    return window


def test_selecting_the_record_and_its_target_offers_the_replication(
        window, pump, tmp_path):
    """Which is the target is settled by type. The user never says."""
    _load_transient(window, tmp_path)
    window.render_current()
    pane = window.data_pane
    assert all(a.isVisible() for a in pane.replication_actions.values())
    assert not any(a.isVisible()
                   for a in pane.comparison_actions.values()), (
        'a waveform error read against an abort band that does not '
        'exist is worse than no reading')


def test_the_record_alone_offers_nothing_to_compare_against(window, pump,
                                                            tmp_path):
    _load_transient(window, tmp_path)
    window.tree.clearSelection()
    name = next(n for n, o in window.objects.items()
                if type(o) is TimeHistory)
    window._item_for_object(name).setSelected(True)
    window.render_current()
    assert not any(a.isVisible() for a in
                   window.data_pane.replication_actions.values())


def test_the_event_box_lists_every_repeat_and_marks_none(window, pump,
                                                         tmp_path):
    """Which repeat is the bad one depends on the reading you care
    about and on what the article is for. Naming one here would put a
    judgment in a list of facts."""
    _load_transient(window, tmp_path, repeats=4)
    history = next(o for o in window.objects.values()
                   if type(o) is TimeHistory)
    history.ordinate[:, 2 * 128:3 * 128] *= 3.0
    window.render_current()
    box = window.data_pane.event_box
    texts = [box.itemText(i) for i in range(box.count())]
    assert texts == [f'Event {i + 1} of 4' for i in range(4)]
    assert window.data_pane.chosen_event() == 0


def test_the_bars_follow_the_event_box(window, pump, tmp_path):
    """The box moves the playing and takes the channels along.

    The grid pins both a channel and a playing, so left alone it would
    outrank the box and the box would do nothing — the trap the channel
    box fell into first."""
    _load_transient(window, tmp_path, repeats=3)
    history = next(o for o in window.objects.values()
                   if type(o) is TimeHistory)
    history.ordinate[:, 128:2 * 128] *= 2.0
    window.data_pane.replication_view = 'waveform'
    window.render_current()
    quiet = [v for _label, v in window.bar_chart.rows]
    assert max(abs(v) for v in quiet) < 1e-6, 'event 0 was untouched'
    window.data_pane.step_event(1)
    window.render_current()
    loud = [v for _label, v in window.bar_chart.rows]
    assert min(loud) == pytest.approx(100.0, abs=1e-6), (
        'the bars moved to the event the box points at')


def test_stepping_stops_at_either_end(window, pump, tmp_path):
    """The repeats are in time order; running off the last one back to
    the first would read as having gone forwards."""
    _load_transient(window, tmp_path, repeats=3)
    window.render_current()
    pane = window.data_pane
    pane.step_event(-1)
    assert pane.chosen_event() == 0
    for _ in range(5):
        pane.step_event(1)
    assert pane.chosen_event() == 2


def test_the_overlay_draws_the_event_and_the_target_together(window, pump,
                                                             tmp_path):
    _load_transient(window, tmp_path, repeats=3)
    window.data_pane.replication_view = 'overlay'
    window.render_current()
    assert 'against the specification' in window._status_text


def test_the_alignment_correction_is_reported_not_hidden(window, pump,
                                                         tmp_path):
    """Asked for explicitly: align the two, then say by how much."""
    path = tmp_path / 'lagged.nc4'
    signal = _transient_file(path, repeats=3, frame=128)
    assert signal is not None
    project = visualdynamics.Project()
    project.import_file(str(path))
    history = next(o for o in project.values() if type(o) is TimeHistory)
    spec = next(o for o in project.values()
                if isinstance(o, TransientSpecification))
    history.ordinate = np.concatenate(
        [np.zeros((history.ordinate.shape[0], 3)), history.ordinate], axis=1)
    history.abscissa = np.arange(history.ordinate.shape[1]) / 256.0
    assert lag_of(history, spec) == 3


def test_the_overlay_draws_one_channel_at_a_time(window, pump, tmp_path):
    """Six measured curves over six targets share one set of colors,
    and nothing on the plot then says which curve is the target. The
    channel box beside the event box is how the others are reached —
    the same box the specification comparison uses."""
    _load_transient(window, tmp_path, repeats=3)
    window.data_pane.replication_view = 'overlay'
    window.render_current()
    box = window.data_pane.pair_box
    labels = [box.itemText(i) for i in range(box.count())]
    assert labels == ['101Z+', '111Z+']
    assert '101Z+' in window._status_text
    window.data_pane.step_pair(1)
    window.render_current()
    assert '111Z+' in window._status_text, 'the plot followed the box'


def test_the_channel_box_goes_away_for_the_bars(window, pump, tmp_path):
    """Every channel is a bar there, so there is nothing to choose."""
    _load_transient(window, tmp_path, repeats=3)
    window.data_pane.replication_view = 'overlay'
    window.render_current()
    assert window.data_pane.pair_action.isVisible()
    window.data_pane.replication_view = 'waveform'
    window.render_current()
    assert not window.data_pane.pair_action.isVisible()


def test_the_headless_route_reaches_all_four_readings(tmp_path):
    """A view the window can put up that a script cannot is the gap
    `test_headless` exists to catch."""
    from visualdynamics.plot import plot_replication

    path = tmp_path / 'run.nc4'
    _transient_file(path, repeats=3, frame=128)
    project = visualdynamics.Project()
    project.import_file(str(path))
    history = next(o for o in project.values() if type(o) is TimeHistory)
    spec = next(o for o in project.values()
                if isinstance(o, TransientSpecification))
    for mode in ('overlay', 'waveform', 'srs', 'level'):
        out = tmp_path / f'{mode}.png'
        plot_replication(history, spec, mode, path=str(out), show=False)
        assert out.stat().st_size > 0
    with pytest.raises(ValueError, match='not a reading'):
        plot_replication(history, spec, 'trac', show=False)
    with pytest.raises(ValueError, match='not a control channel'):
        plot_replication(history, spec, 'overlay', channel='999X+',
                         show=False)


def test_the_channels_picked_in_the_grid_are_the_ones_drawn(window, pump,
                                                            tmp_path):
    """Picking channels and being shown every channel anyway is the
    grid saying one thing and the plot another.

    The two grids are pooled: a channel chosen on the target and the
    same channel chosen on the record are the same request.
    """
    _load_transient(window, tmp_path, repeats=3)
    window.data_pane.replication_view = 'overlay'
    window.render_current()
    assert '101Z+' in window._status_text
    grid = window.record_grids.get('Time History')
    assert grid is not None, 'the record grid is where channels are picked'
    grid.select_records([1])
    window.render_current()
    assert '111Z+' in window._status_text
    assert '101Z+' not in window._status_text


def test_several_picked_channels_are_all_drawn(window, pump, tmp_path):
    _load_transient(window, tmp_path, repeats=3)
    window.data_pane.replication_view = 'overlay'
    window.record_grids['Time History'].select_records([0, 1])
    window.render_current()
    assert '101Z+' in window._status_text
    assert '111Z+' in window._status_text


# --- the grid under the plot -----------------------------------------

def _grid(window):
    model = window.table.model()
    return [[model.data(model.index(r, c))
             for c in range(model.columnCount())]
            for r in range(model.rowCount())]


def _headers(window):
    from PySide6.QtCore import Qt

    model = window.table.model()
    return [model.headerData(c, Qt.Orientation.Horizontal)
            for c in range(model.columnCount())]


def _pick(window, cells, pump):
    from PySide6.QtCore import QItemSelectionModel

    selection = window.table.selectionModel()
    model = window.table.model()
    selection.clearSelection()
    for row, column in cells:
        selection.select(model.index(row, column),
                         QItemSelectionModel.SelectionFlag.Select)
    for _ in range(3):
        pump()


def test_the_grid_is_channels_down_and_playings_across(window, pump,
                                                       tmp_path):
    """A transient record is many attempts at one target, so the
    numbers are two-dimensional. Laid out this way a channel bad
    everywhere and a repeat bad for everything look different at a
    glance, which one column cannot show."""
    _load_transient(window, tmp_path, repeats=3)
    window.data_pane.replication_view = 'overlay'
    window.render_current()
    pump()
    assert window.data_pane.isVisible()
    assert window.table_pane.isVisible()
    assert _headers(window) == ['Channel', 'Event 1 [%]', 'Event 2 [%]',
                                'Event 3 [%]']
    rows = _grid(window)
    assert [row[0] for row in rows] == ['101Z+', '111Z+']
    assert all(float(cell) == pytest.approx(0.0, abs=1e-6)
               for row in rows for cell in row[1:])


def test_a_cell_that_cannot_be_scored_is_blank_not_zero(window, pump,
                                                        tmp_path):
    """A zero would read as a channel that replicated perfectly, which
    is the opposite of what is known about it."""
    path = tmp_path / 'quiet.nc4'
    _transient_file(path, repeats=2, frame=128)
    window.import_paths([str(path)])
    spec = next(o for o in window.objects.values()
                if isinstance(o, TransientSpecification))
    spec.ordinate[1] = 0.0
    window.tree.clearSelection()
    for name, obj in window.objects.items():
        if isinstance(obj, (TimeHistory, TransientSpecification)):
            window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    rows = _grid(window)
    assert rows[1][0] == '111Z+'
    assert all(cell in ('', None) for cell in rows[1][1:])


def test_each_playing_gets_its_own_column(window, pump, tmp_path):
    """And the columns disagree when the repeats do — which is what
    makes a grid of identical numbers a fact about the data rather than
    a stale view."""
    _load_transient(window, tmp_path, repeats=3)
    history = next(o for o in window.objects.values()
                   if type(o) is TimeHistory)
    history.ordinate[:, 128:2 * 128] *= 2.0
    window.render_current()
    pump()
    for row in _grid(window):
        assert float(row[1]) == pytest.approx(0.0, abs=1e-6)
        assert float(row[2]) == pytest.approx(100.0, abs=1e-6)
        assert float(row[3]) == pytest.approx(0.0, abs=1e-6)


def test_one_cell_names_a_channel_and_a_playing(window, pump, tmp_path):
    _load_transient(window, tmp_path, repeats=3)
    window.data_pane.replication_view = 'overlay'
    window.render_current()
    pump()
    _pick(window, [(1, 3)], pump)
    assert '111Z+' in window._status_text
    assert 'event 3' in window._status_text
    assert '101Z+' not in window._status_text


def test_the_name_column_asks_for_that_channel_everywhere(window, pump,
                                                          tmp_path):
    """Column zero is the channel's name rather than a reading of it,
    so picking there is the row as a whole."""
    _load_transient(window, tmp_path, repeats=3)
    window.data_pane.replication_view = 'overlay'
    window.render_current()
    pump()
    _pick(window, [(0, 0)], pump)
    assert '101Z+' in window._status_text
    assert 'events 1, 2, 3' in window._status_text


def test_a_column_is_every_channel_at_one_playing(window, pump, tmp_path):
    _load_transient(window, tmp_path, repeats=3)
    window.data_pane.replication_view = 'overlay'
    window.render_current()
    pump()
    _pick(window, [(0, 2), (1, 2)], pump)
    assert '101Z+' in window._status_text
    assert '111Z+' in window._status_text
    assert 'event 2 of 3' in window._status_text


def test_a_scatter_of_cells_is_exactly_what_it_looks_like(window, pump,
                                                          tmp_path):
    """Any combination, not just whole rows or whole columns."""
    _load_transient(window, tmp_path, repeats=3)
    window.data_pane.replication_view = 'overlay'
    window.render_current()
    pump()
    _pick(window, [(0, 1), (1, 3)], pump)
    assert '101Z+' in window._status_text
    assert '111Z+' in window._status_text
    assert 'events 1, 3' in window._status_text


def test_the_bars_are_the_picked_cells(window, pump, tmp_path):
    """And the repeat earns a place in a label only when more than one
    is up — `101Z+ e3` on every bar of a single playing is noise."""
    _load_transient(window, tmp_path, repeats=3)
    window.data_pane.replication_view = 'waveform'
    window.render_current()
    pump()
    _pick(window, [(0, 1), (1, 3)], pump)
    assert [label for label, _v in window.bar_chart.rows] == [
        '101Z+ e1', '111Z+ e3']
    _pick(window, [(0, 2), (1, 2)], pump)
    assert [label for label, _v in window.bar_chart.rows] == [
        '101Z+', '111Z+'], 'one playing needs no repeat in the label'


def test_the_grid_is_up_for_the_bars_too(window, pump, tmp_path):
    """It is the account of every channel at every playing, and a bar
    chart of the picked cells is never that."""
    _load_transient(window, tmp_path, repeats=3)
    window.data_pane.replication_view = 'waveform'
    window.render_current()
    pump()
    assert window.table_pane.isVisible()
    assert _headers(window)[0] == 'Channel'
    assert len(_headers(window)) == 4


def test_the_event_box_moves_the_playing_and_keeps_the_channels(
        window, pump, tmp_path):
    """Both controls stay meaningful: the box answers which repeat, the
    grid answers which of these, and neither gives up its half."""
    _load_transient(window, tmp_path, repeats=3)
    window.data_pane.replication_view = 'overlay'
    window.render_current()
    pump()
    _pick(window, [(1, 1)], pump)
    assert '111Z+' in window._status_text and 'event 1' in window._status_text
    window.data_pane.step_event(1)
    pump()
    assert '111Z+' in window._status_text, 'the channel came along'
    assert 'event 2 of 3' in window._status_text


def test_the_grid_shows_what_is_drawn_as_selected(window, pump, tmp_path):
    """A grid of a hundred cells has to say which of them is on screen."""
    from PySide6.QtCore import Qt

    _load_transient(window, tmp_path, repeats=3)
    window.data_pane.replication_view = 'overlay'
    window.render_current()
    pump()
    _pick(window, [(0, 2), (1, 3)], pump)
    picked = {(i.row(), i.column())
              for i in window.table.selectionModel().selectedIndexes()
              if i.column() > 0}
    assert picked == {(0, 2), (1, 3)}
    model = window.table.model()
    assert model.headerData(2, Qt.Orientation.Horizontal) == 'Event 2 [%]'


def test_adding_cells_never_takes_others_away(window, pump, tmp_path):
    """Command-clicking a cell used to drop others.

    Every drawing rebuilt the model, cleared the selection and put it
    back from `_replication_pairs` — and a drawing is what a pick
    *causes*, so the selection just made was torn down and rebuilt from
    this end's idea of it. Anything that idea could not express, like a
    column-zero pick standing for a whole row, vanished. The selection
    is the input here, not the output.
    """
    from PySide6.QtCore import QItemSelectionModel

    _load_transient(window, tmp_path, repeats=4)
    window.data_pane.replication_view = 'overlay'
    window.render_current()
    pump()
    selection = window.table.selectionModel()
    model = window.table.model()
    selection.clearSelection()
    added = []
    for row, column in [(0, 1), (1, 3), (0, 4), (1, 2)]:
        selection.select(model.index(row, column),
                         QItemSelectionModel.SelectionFlag.Select)
        for _ in range(3):
            pump()
        added.append((row, column))
        held = {(i.row(), i.column())
                for i in window.table.selectionModel().selectedIndexes()}
        assert held == set(added), f'adding {(row, column)} lost cells'


def test_an_unchanged_grid_keeps_its_model(window, pump, tmp_path):
    """Rebuilding a table that says the same thing is what let the
    selection be rewritten underneath the user."""
    _load_transient(window, tmp_path, repeats=3)
    window.data_pane.replication_view = 'overlay'
    window.render_current()
    pump()
    first = window.table.model()
    window.render_current()
    pump()
    assert window.table.model() is first


def test_the_grid_costs_no_shock_spectra(window, pump, tmp_path,
                                          monkeypatch):
    """The grid shows waveform errors, and an SRS is three orders of
    magnitude dearer to work out. Asking for one per cell cost two
    seconds an update on the survey run."""
    from visualdynamics.core import srs as srs_module

    called = []
    real = srs_module.maximax
    monkeypatch.setattr(srs_module, 'maximax',
                        lambda *a, **k: called.append(1) or real(*a, **k))
    _load_transient(window, tmp_path, repeats=3)
    window.data_pane.replication_view = 'overlay'
    window.render_current()
    pump()
    assert called == [], 'the time-data view computed a shock spectrum'
    window.data_pane.replication_view = 'waveform'
    window.render_current()
    pump()
    assert called == [], 'and the bars did too'


# --- a playing the recording cut off ---------------------------------

def test_a_part_played_repeat_is_not_analyzed():
    """Rattlesnake repeats until told to stop, so a record rarely ends
    on a boundary. What is left is not a playing: a waveform error over
    part of a window is taken against a different stretch of the
    target, and a column of those invites comparing things that do not
    compare.
    """
    spec = _target()
    history = _measured(spec, repeats=3, extra=FRAME // 2)
    assert [p['frame'] for p in rep_playings(history, spec)] == [0, 1, 2]
    assert sorted({r['frame'] for r in
                   compare(history, spec, metrics=('waveform',))}) == [0, 1, 2]


def test_every_view_of_the_record_counts_the_same_playings():
    """The bug this replaced: the grid counted a part-played one as a
    seventh event while the shading, which comes from
    `Averaging.frame_bounds`, knew only about whole frames. Two views
    of one record giving two answers."""
    spec = _target()
    history = _measured(spec, repeats=3, extra=FRAME // 2)
    averaging = averaging_for(history, spec)
    assert (len(rep_playings(history, spec))
            == averaging.frames
            == len(averaging.frame_bounds(RATE))
            == 3)


def test_what_was_left_over_is_still_reported():
    """Not analyzed is not the same as not mentioned, which is where
    this started: a run that recorded two seconds of a further event
    should not have it vanish between a frame count and a file size."""
    spec = _target()
    history = _measured(spec, repeats=3, extra=FRAME // 2)
    samples, share = rep_leftover(history, spec)
    assert samples == FRAME // 2
    assert share == pytest.approx(0.5)


def test_a_record_ending_on_a_boundary_has_nothing_left_over():
    spec = _target()
    assert rep_leftover(_measured(spec, repeats=3), spec) is None


def test_the_grid_columns_are_the_whole_playings(window, pump, tmp_path):
    from PySide6.QtCore import Qt

    path = tmp_path / 'clipped.nc4'
    _transient_file(path, repeats=2, frame=128, extra=64)
    window.import_paths([str(path)])
    window.tree.clearSelection()
    for name, obj in window.objects.items():
        if isinstance(obj, (TimeHistory, TransientSpecification)):
            window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    model = window.table.model()
    headers = [model.headerData(c, Qt.Orientation.Horizontal)
               for c in range(model.columnCount())]
    assert headers == ['Channel', 'Event 1 [%]', 'Event 2 [%]']
    assert [window.data_pane.event_box.itemText(i)
            for i in range(window.data_pane.event_box.count())] == [
        'Event 1 of 2', 'Event 2 of 2']
    assert 'further event' in window._status_text, (
        'the fragment is excluded, not hidden')


# --- the PSD comparison ----------------------------------------------

def _psd_pair(window, tmp_path, pump):
    path = tmp_path / 'run.nc4'
    _transient_file(path, repeats=3, frame=128)
    window.import_paths([str(path)])
    for name, obj in list(window.objects.items()):
        if isinstance(obj, (TimeHistory, TransientSpecification)):
            window.tree.clearSelection()
            item = window._item_for_object(name)
            item.setSelected(True)
            window.tree.setCurrentItem(item)
            window.compute_psds()
    window.tree.clearSelection()
    for name in [n for n in window.objects if 'PSD' in n]:
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()


def test_a_psd_of_a_target_is_averaged_over_every_playing(tmp_path):
    """Which is why the PSD comparison needs no event chooser: the
    repeats are already in it. `compute_psds` averages over exactly the
    frames the averaging defines, and for a transient run those frames
    are the playings."""
    from dataclasses import replace

    path = tmp_path / 'run.nc4'
    _transient_file(path, repeats=3, frame=128)
    project = visualdynamics.Project()
    project.import_file(str(path))
    history = next(o for o in project.values() if type(o) is TimeHistory)
    assert history.averaging.frames == 3
    history.ordinate[:, 128:2 * 128] *= 4.0      # spoil the middle one
    averaged = history.compute_psds()
    alone = history.compute_psds(replace(history.averaging, frames=1))
    assert not np.allclose(averaged.ordinate, alone.ordinate), (
        'the average of three playings is not the first of them'
    )


def test_the_psd_comparison_puts_a_channel_table_under_the_plot(
        window, pump, tmp_path):
    """The same arrangement the time comparison has."""
    from PySide6.QtCore import Qt

    _psd_pair(window, tmp_path, pump)
    assert window.data_pane.isVisible()
    assert window.table_pane.isVisible()
    model = window.table.model()
    headers = [model.headerData(c, Qt.Orientation.Horizontal)
               for c in range(model.columnCount())]
    assert headers == ['Channel', 'RMS error [dB]', 'Outside abort [%]']
    assert model.rowCount() >= 1
    for row in range(model.rowCount()):
        assert np.isfinite(float(model.data(model.index(row, 1)))), (
            'a dense specification used to integrate to NaN'
        )


def test_the_psd_comparison_offers_no_event_chooser(window, pump,
                                                    tmp_path):
    """A PSD is already an average over the playings, so there is no
    repeat left to choose between."""
    _psd_pair(window, tmp_path, pump)
    assert not window.data_pane.event_action.isVisible()


def test_picking_psd_rows_draws_those_channels(window, pump, tmp_path):
    """Matched by label and not by tuple: `channel_errors` says '121Z+'
    where `specification_pairs` says ('121Z+', '121Z+'), and compared
    as tuples every pick fell back to the first channel."""
    from PySide6.QtCore import QItemSelectionModel

    _psd_pair(window, tmp_path, pump)
    model = window.table.model()
    if model.rowCount() < 2:
        pytest.skip('needs two control channels to tell a pick apart')
    selection = window.table.selectionModel()
    selection.clearSelection()
    selection.select(model.index(1, 0),
                     QItemSelectionModel.SelectionFlag.Select
                     | QItemSelectionModel.SelectionFlag.Rows)
    for _ in range(4):
        pump()
    assert window._compliance_channels == [model.data(model.index(1, 0))]
    assert window._plotted_pair[0] == model.data(model.index(1, 0))


# --- the averaging panel drives the comparison -----------------------

def test_narrowing_the_averaging_narrows_the_comparison():
    """The panel is where a user says which events to analyze. Left
    unread, the shading on the time history said three and every number
    beside it was still worked out from six."""
    from dataclasses import replace

    spec = _target()
    history = _measured(spec, repeats=5)
    assert len(rep_playings(history, spec)) == 5
    history.averaging = replace(averaging_for(history, spec), frames=3)
    assert [p['frame'] for p in rep_playings(history, spec)] == [0, 1, 2]
    assert sorted({r['frame'] for r in
                   compare(history, spec, metrics=('waveform',))}) == [0, 1, 2]


def test_the_start_of_the_analysis_moves_the_playings():
    from dataclasses import replace

    spec = _target()
    history = _measured(spec, repeats=4)
    history.averaging = replace(averaging_for(history, spec), frames=2,
                                start=FRAME / RATE)
    starts = [p['start'] for p in rep_playings(history, spec)]
    assert starts == [FRAME, 2 * FRAME]


def test_an_averaging_about_something_else_is_not_honored():
    """A frame that is not the specification's length does not line up
    with a playing of it, and honoring it would compare each event
    against a slice of the target chosen by accident."""
    from dataclasses import replace

    spec = _target()
    history = _measured(spec, repeats=4)
    history.averaging = replace(averaging_for(history, spec),
                                frame_length=FRAME // 4, frames=9)
    derived = averaging_for(history, spec)
    assert derived.frame_length == FRAME
    assert derived.frames == 4, 'it fell back to what the playings are'


def test_what_was_left_over_does_not_move_with_the_choice():
    """Choosing to look at three of six is a decision and needs no
    announcing; a recording that stopped part way through a playing is
    a fact about the file."""
    from dataclasses import replace

    spec = _target()
    history = _measured(spec, repeats=4, extra=FRAME // 2)
    before = rep_leftover(history, spec)
    history.averaging = replace(averaging_for(history, spec), frames=2)
    assert rep_leftover(history, spec) == before == (FRAME // 2, 0.5)


def test_the_event_box_belongs_to_the_comparison_alone(window, pump,
                                                       tmp_path):
    """Plotting the record on its own shows every event at once, so
    there is nothing to choose between.

    `reset_controls` promises that a drawing with no use for the bar
    need not remember to put away what the last one left up — a promise
    each control has to be registered there to keep. The event box was
    added later and never was, so once a transient comparison raised
    it, it stayed up over the record alone and over a channel table,
    which has no events at all.
    """
    _load_transient(window, tmp_path, repeats=3)
    window.render_current()
    pump()
    assert window.data_pane.event_action.isVisible()

    history = next(n for n, o in window.objects.items()
                   if type(o) is TimeHistory)
    for name in (history, 'Channel Table'):
        if name not in window.objects:
            continue
        window.tree.clearSelection()
        window._item_for_object(name).setSelected(True)
        window.render_current()
        pump()
        assert not window.data_pane.event_action.isVisible(), (
            f'the event box stayed up over {name}')
        assert not any(a.isVisible() for a in
                       window.data_pane.replication_actions.values())

    # and it comes back when the comparison does
    window.tree.clearSelection()
    for name, obj in window.objects.items():
        if isinstance(obj, (TimeHistory, TransientSpecification)):
            window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    assert window.data_pane.event_action.isVisible()


# --- frames or events, never both ------------------------------------

def test_no_project_type_preselects_a_reading(qt_app):
    """The default is the record itself — the plain stage, no reading
    pre-selected, whatever the project type (Brandon, 2026-08-30;
    random and transient used to open with the averaging view up,
    marks before the data had been looked at, and shock opened on
    detection). The toggles are sticky once touched; that is the whole
    mechanism now."""
    from visualdynamics.gui.panes import DataPane

    pane = DataPane('dark')
    assert not pane.averaging_wanted and not pane.shocks_wanted
    assert not hasattr(pane, 'read_record_as'), \
        'the type-to-reading hook is gone, not dormant'


def test_the_srs_of_a_transient_run_is_windowed_on_its_frames():
    """One curve per channel per playing, from the frames the averaging
    already defines — not from whatever a detector makes of the same
    record."""
    spec = _target()
    history = _measured(spec, repeats=4)
    history.averaging = averaging_for(history, spec)
    srs = history.compute_srs()
    assert srs.num_records == 4 * len(history.response_dof)
    assert sorted(set(srs.block)) == ['shock 1', 'shock 2', 'shock 3',
                                      'shock 4']


def test_the_srs_of_a_specification_is_one_playing():
    """It is a single playing of a waveform, whole."""
    spec = _target()
    assert spec.averaging is None and not spec.shocks
    srs = spec.compute_srs()
    assert srs.num_records == len(spec.response_dof)
    assert isinstance(srs, ShockSpecification)


def test_detected_shocks_still_win_where_there_are_any():
    """A shock record's events really do have to be found."""
    from visualdynamics.core.shocks import Shock

    spec = _target()
    history = _measured(spec, repeats=4)
    history.averaging = averaging_for(history, spec)
    history.shocks = (Shock(0.0, FRAME / RATE),)
    assert history.compute_srs().num_records == len(history.response_dof)


# --- shock spectra against the one required --------------------------

def _srs_pair(window, tmp_path, pump):
    path = tmp_path / 'run.nc4'
    _transient_file(path, repeats=3, frame=128)
    window.import_paths([str(path)])
    for name, obj in list(window.objects.items()):
        if isinstance(obj, (TimeHistory, TransientSpecification)):
            window.tree.clearSelection()
            item = window._item_for_object(name)
            item.setSelected(True)
            window.tree.setCurrentItem(item)
            window.compute_srs()
    window.tree.clearSelection()
    for name in [n for n in window.objects if 'SRS' in n]:
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()


def test_computing_an_srs_does_not_go_hunting_for_events():
    """Detection is the last resort, not the first. A record read as
    frames already says where its events are, and set loose on the
    survey run the detector answered with thirty-one where there were
    six — and cut the target, which is one playing, into three."""
    spec = _target()
    history = _measured(spec, repeats=3)
    history.averaging = averaging_for(history, spec)
    project = visualdynamics.Project()
    project.add('Response', history)
    project.add('Target', spec)
    project.compute_srs('Response')
    project.compute_srs('Target')
    assert not history.shocks, 'the frames already said where they are'
    assert not spec.shocks, 'a target is one playing by definition'
    assert project['Response SRS'].num_records == 3 * len(history.response_dof)
    assert project['Target SRS'].num_records == len(spec.response_dof)


def test_the_srs_comparison_has_the_same_layout(window, pump, tmp_path):
    """Plot on top, grid beneath, and a button for the statistic."""
    from PySide6.QtCore import Qt

    _srs_pair(window, tmp_path, pump)
    assert window.data_pane.isVisible()
    assert window.table_pane.isVisible()
    assert all(a.isVisible() for a in window.data_pane.srs_actions.values())
    model = window.table.model()
    headers = [model.headerData(c, Qt.Orientation.Horizontal)
               for c in range(model.columnCount())]
    assert headers == ['Channel', 'Event 1 [dB]', 'Event 2 [dB]',
                       'Event 3 [dB]']


def test_the_srs_statistic_is_rms_decibels(window, pump, tmp_path):
    """In dB because an SRS spans decades and is read on log axes: the
    linear distance a waveform gets is dominated by whatever bands are
    loudest and goes blind to a large ratio error where the target is
    small. On the survey it passed every channel at 20% while one sat
    25 dB over."""
    from visualdynamics.core.compliance import SRS_ERROR_DB

    _srs_pair(window, tmp_path, pump)
    window.data_pane.srs_view = 'error'
    window.render_current()
    pump()
    assert window.bar_chart.low == -SRS_ERROR_DB
    assert window.bar_chart.high == SRS_ERROR_DB, (
        'bounded both sides: under-testing is a fault of its own')
    assert window.bar_chart.units == 'dB'


def test_a_perfect_replication_has_no_srs_deviation():
    from visualdynamics.core.compliance import srs_errors

    spec = _target()
    history = _measured(spec, repeats=2)
    history.averaging = averaging_for(history, spec)
    for _dof, _block, value in srs_errors(spec.compute_srs(),
                                          history.compute_srs()):
        assert value == pytest.approx(0.0, abs=1e-9)


# --- a computed spectrum is a density, whatever class it wears -------

def test_every_computed_spectrum_is_drawn_flat_in_band(tmp_path):
    """A PSD is a density sampled once per bin, so it is drawn flat
    across each — the area drawn is the area counted when it is
    integrated.

    The shape used to be chosen by class, and `Specification` means
    breakpoints of a power law. True of one written by hand; false of
    one computed from a record. The PSD of a transient target is four
    thousand lines of density and was being drawn as a power law
    through themselves.
    """
    from visualdynamics.core.data import Psd, Specification

    path = tmp_path / 'run.nc4'
    _transient_file(path, repeats=3, frame=128)
    project = visualdynamics.Project()
    project.import_file(str(path))
    for name, obj in list(project.items()):
        if isinstance(obj, TimeHistory):
            project.compute_psds(name)
    project.compute_cpsds(
        next(n for n, o in project.items() if type(o) is TimeHistory))
    computed = [o for o in project.values() if isinstance(o, Psd)]
    assert len(computed) >= 3
    for spectrum in computed:
        assert spectrum.interpolation == 'bin', (
            f'{type(spectrum).__name__} is a density')
    # and the target's own PSD is still a Specification, so the app
    # still knows which of the two is the requirement
    assert any(isinstance(o, Specification) for o in computed)


def test_a_written_specification_stays_a_power_law():
    """Its points are breakpoints, and drawing them as steps would make
    a staircase of a slope. A controller's target on its lines is the
    other thing — a density per line, stepped — and the spacing tells
    them apart at import (Brandon, 2026-09-19); this test pinned the
    lines one as breakpoints until then."""
    from visualdynamics.core.data import Specification

    project = visualdynamics.Project()
    project.import_file(fixture_path('plate', 'random.nc4'))
    lines = next(o for o in project.values() if type(o) is Specification)
    assert lines.interpolation == 'bin', "the controller's lines"
    written = Specification(np.array([20.0, 80.0, 800.0, 2000.0]),
                            np.array([[1e-3, 4e-3, 4e-3, 1e-3]]),
                            response_dof=['101Z+'],
                            ordinate_dim=['acceleration**2/frequency'])
    assert written.interpolation == 'log_log'


def test_whether_it_is_a_density_survives_the_project_file(tmp_path):
    """It cannot be worked out again from the numbers: a specification
    computed from a record and one written by hand can both arrive as
    several hundred evenly spaced lines, and only the first is a
    density. Left behind, a reopened project drew one as a power law.
    """
    path = tmp_path / 'run.nc4'
    _transient_file(path, repeats=3, frame=128)
    project = visualdynamics.Project()
    project.import_file(str(path))
    target = next(n for n, o in project.items()
                  if isinstance(o, TransientSpecification))
    added = project.compute_psds(target)
    assert project[added].interpolation == 'bin'
    saved = project.save(tmp_path / 'p.vdyn')
    assert visualdynamics.Project.open(saved)[added].interpolation == 'bin'


def test_the_dof_arrows_reach_the_scene_a_history_is_animated_on(
        window, pump, tmp_path):
    """There are two scene builders and only one of them drew these.

    A geometry on its own goes through `_render_geometries`; a geometry
    with a time history on it builds its scene through the animator
    instead. So turning the arrows on beside a time history — the one
    selection that makes the button appear at all — did nothing.
    """
    window.import_paths([fixture_path('plate', 'time.npz'),
                         fixture_path('plate', 'geometry.npz')])
    pump()
    geometry = next(n for n, o in window.objects.items()
                    if type(o).__name__ == 'Geometry')
    history = next(n for n, o in window.objects.items()
                   if type(o).__name__ == 'TimeHistory')
    # the fixture arrives with no units, and a quantity is what the
    # arrows are grouped by — with none there is nothing to draw, which
    # is right and is not what this is about
    window.objects[history].define_units('m/s^2')
    window.tree.clearSelection()
    for name in (geometry, history):
        window._item_for_object(name).setSelected(True)
    window.render_current()
    pump()
    assert window.dofs_action.isVisible(), (
        'a geometry and the data to read DOFs from are both selected')
    bare = len(window.scene.plotter.renderer.actors)

    window.dofs_action.setChecked(True)
    window.render_current()
    pump()
    assert len(window.scene.plotter.renderer.actors) > bare, (
        'the arrows never reached the animated scene'
    )

    window.dofs_action.setChecked(False)
    window.render_current()
    pump()
    assert len(window.scene.plotter.renderer.actors) == bare, (
        'and they go away again'
    )


def test_a_geometry_alone_marks_its_own_degrees_of_freedom(window, pump):
    """Every node's own X, Y and Z.

    There is no measurement to read a quantity from, so the drop-down
    has nothing to offer and stays away — what is drawn is the
    geometry's own axes rather than what something happened to measure
    along them. Written as plain DOF strings, so a node whose
    displacement coordinate system is turned gets its arrows turned
    with it.
    """
    window.import_paths([fixture_path('plate', 'geometry.npz')])
    pump()
    geometry = next(n for n, o in window.objects.items()
                    if type(o).__name__ == 'Geometry')
    window.tree.clearSelection()
    window._item_for_object(geometry).setSelected(True)
    window.render_current()
    pump()
    assert window.dofs_action.isVisible(), 'a geometry has DOFs to mark'
    assert not window.dofs_combo_action.isVisible(), (
        'and nothing measured to choose a quantity from')
    bare = len(window.scene.plotter.renderer.actors)

    window.dofs_action.setChecked(True)
    window.render_current()
    pump()
    nodes = len(window.objects[geometry].node_id)
    drawn = len(window.scene.plotter.renderer.actors) - bare
    assert drawn == 3 * nodes, f'{drawn} arrows for {nodes} nodes'

    window.dofs_action.setChecked(False)
    window.render_current()
    pump()
    assert len(window.scene.plotter.renderer.actors) == bare


def test_a_lone_specification_reads_one_channel_at_a_time(window, pump,
                                                          tmp_path):
    """A transient specification selected on its own: its waveforms
    stacked are a thicket with no comparison in them, so one channel
    draws and the drop-down reaches the rest — the same reading the
    report gives it."""
    from visualdynamics.plot import data_curves

    path = tmp_path / 'run.nc4'
    _transient_file(path, repeats=3, frame=128)
    window.import_paths([str(path)])
    window.tree.clearSelection()
    name = next(n for n, o in window.objects.items()
                if isinstance(o, TransientSpecification))
    window._item_for_object(name).setSelected(True)
    window.tree.setCurrentItem(window._item_for_object(name))
    pump()
    assert window.data_pane.pair_action.isVisible(), (
        'the channel drop-down is the way to the others')
    box = window.data_pane.pair_box
    assert box.count() == 2, 'one entry per control channel'
    plots = [item for item in window.data_pane.graphics.ci.items
             if hasattr(item, 'listDataItems')]
    assert sum(len(data_curves(p)) for p in plots) == 1, (
        'one channel on screen, not the thicket')
    # picking the other channel redraws it
    box.setCurrentIndex(1)
    pump()
    plots = [item for item in window.data_pane.graphics.ci.items
             if hasattr(item, 'listDataItems')]
    assert sum(len(data_curves(p)) for p in plots) == 1


# ---- the deviation is signed, and the two SRS readings compose -----------


def test_an_under_hit_shock_reads_negative_not_over():
    """Brandon, on the drone stress set: the bar chart said the SRS was
    outside 3 dB *high* when the spectrum was actually low. The RMS was
    unsigned — a spectrum 6 dB low everywhere reported +6, shaded past
    the ceiling, and read as an over-test."""
    import copy

    from visualdynamics.core.compliance import srs_errors

    spec = _target().compute_srs()
    quiet = copy.deepcopy(spec)
    quiet.ordinate = spec.ordinate * 10 ** (-6.0 / 20.0)   # 6 dB low
    loud = copy.deepcopy(spec)
    loud.ordinate = spec.ordinate * 10 ** (6.0 / 20.0)     # 6 dB high

    for _dof, _block, value in srs_errors(spec, quiet):
        assert value == pytest.approx(-6.0, abs=1e-6), \
            'an under-hit reads negative'
    for _dof, _block, value in srs_errors(spec, loud):
        assert value == pytest.approx(6.0, abs=1e-6), \
            'an over-test reads positive'


def test_the_bars_show_every_channel_of_every_event(window, pump,
                                                    tmp_path):
    """One signed number per channel per event, all at once: the bar
    chart exists to say at a glance where the whole test stands, and a
    view that pages is a view that hides."""
    _srs_pair(window, tmp_path, pump)
    window.data_pane.srs_view = 'error'
    window.render_current()
    pump()
    labels = [label for label, _v in window.bar_chart.rows]
    assert len(labels) == 2 * 3, 'two channels, three events, six bars'
    assert not window.data_pane.event_action.isVisible(), \
        'nothing to page when everything is shown'


def test_the_curves_show_one_channel_all_events(window, pump, tmp_path):
    """The events are what a shock series compares, so they share the
    plot; the box picks which control channel's spectra are up."""
    _srs_pair(window, tmp_path, pump)
    window.data_pane.srs_view = 'curves'
    window.render_current()
    pump()
    box = window.data_pane.event_box
    assert box.count() == 2, 'the box lists the control channels'
    chosen = box.itemText(box.currentIndex())
    curves = [item for item in
              window.data_pane.graphics.ci.items
              if hasattr(item, 'listDataItems')]
    drawn = sum(len(plot.listDataItems()) for plot in curves)
    assert drawn >= 3 + 1, (
        f'{chosen}: all three events and the specification, '
        f'got {drawn} curves')
