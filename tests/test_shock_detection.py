"""Finding the shocks in a recording.

Built on synthesized records where the answer is known by construction,
because that is the only way to say a detector found the right events
rather than merely some events. The real file is checked too, at the
end, against what the generator put in it.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core import shocks
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.shocks import Shock

RATE = 8192.0
NOISE = 1.0            # the floor everything is measured against
PEAK = 400.0           # a shock, 52 dB over it


def record(events=(0.3, 1.0, 1.7), peak=PEAK, span=2.4, noise=NOISE,
           channels=3, decay=0.02, ring=78.0, seed=0):
    """A recording with shocks at known times.

    Each is a short pulse and a ringdown, on a noise floor that runs the
    whole length — a record that is digitally silent between events has
    no floor to measure them against, which is a different problem.
    """
    rng = np.random.default_rng(seed)
    samples = round(span * RATE)
    t = np.arange(samples) / RATE
    rows = []
    for c in range(channels):
        row = rng.normal(0.0, noise, samples)
        for at in events:
            since = t - at
            live = since >= 0
            row[live] += peak * (1.0 + 0.2 * c) * np.exp(
                -decay * 2 * np.pi * ring * since[live]) * np.sin(
                2 * np.pi * ring * since[live])
        rows.append(row)
    return TimeHistory(t, np.array(rows),
                       response_dof=[f'10{c}Z+' for c in range(channels)],
                       ordinate_dim=['acceleration'] * channels)


def starts(found):
    return [round(s.start, 3) for s in found]


# ---- the events ---------------------------------------------------------


def test_it_finds_one_event_per_shock():
    found = shocks.find(record(events=(0.3, 1.0, 1.7)))
    assert len(found) == 3


def test_each_window_opens_before_its_own_shock():
    """The lead-in is not decoration. An SRS oscillator starts from
    rest, and a window opening mid-pulse invents a step that was never
    in the record."""
    events = (0.3, 1.0, 1.7)
    found = shocks.find(record(events=events))
    for at, window in zip(events, found):
        assert window.start < at, 'opens before the rise'
        assert window.stop > at, 'and does not close on it'


def test_a_ringdown_is_one_event_and_not_forty():
    """What the hysteresis is for. A decaying sinusoid crosses any
    single threshold on every cycle; a lightly damped one crosses it
    hundreds of times."""
    found = shocks.find(record(events=(1.0,), decay=0.003))
    assert len(found) == 1


def test_the_window_holds_the_whole_ringdown():
    """A window cut at the pulse would leave the low-frequency
    oscillators still answering after it closed."""
    found = shocks.find(record(events=(0.6,), decay=0.02, ring=78.0, span=2.0))
    assert len(found) == 1
    # 1/(zeta*omega) is 0.10 s here, so a window that stops in under
    # two time constants has thrown away the part still ringing
    assert found[0].stop - 0.6 > 0.2


def test_nothing_is_found_in_a_stationary_record():
    """A random vibration run has no quiet, so nothing rises over it.
    Reporting the whole record as one shock would be worse than saying
    nothing."""
    rng = np.random.default_rng(1)
    samples = round(2.0 * RATE)
    t = np.arange(samples) / RATE
    steady = TimeHistory(t, rng.normal(0.0, 5.0, (3, samples)),
                         response_dof=['101Z+', '102Z+', '103Z+'],
                         ordinate_dim=['acceleration'] * 3)
    assert shocks.find(steady) == ()


def test_a_record_with_no_shocks_still_offers_the_whole_of_itself():
    """`suggest` is for the caller that needs something to analyze. The
    honest fallback is the record, not nothing."""
    rng = np.random.default_rng(1)
    samples = round(2.0 * RATE)
    t = np.arange(samples) / RATE
    steady = TimeHistory(t, rng.normal(0.0, 5.0, (1, samples)),
                         response_dof=['101Z+'], ordinate_dim=['acceleration'])
    suggested = shocks.suggest(steady)
    assert len(suggested) == 1
    assert suggested[0].start == 0.0
    assert suggested[0].duration == pytest.approx(samples / RATE)


# ---- what makes it robust ------------------------------------------------


def test_a_walked_up_series_is_found_whole():
    """Shock series are walked up in level, so the last event can be
    many times the first. Thresholds taken from the floor find them all;
    thresholds taken from the record's peak find only the big ones."""
    rng = np.random.default_rng(2)
    samples = round(2.4 * RATE)
    t = np.arange(samples) / RATE
    row = rng.normal(0.0, NOISE, samples)
    for k, at in enumerate((0.3, 1.0, 1.7)):
        since = t - at
        live = since >= 0
        row[live] += PEAK * (0.1 * 4 ** k) * np.exp(
            -0.02 * 2 * np.pi * 78.0 * since[live]) * np.sin(
            2 * np.pi * 78.0 * since[live])
    walked = TimeHistory(t, row[None, :], response_dof=['101Z+'],
                         ordinate_dim=['acceleration'])
    assert len(shocks.find(walked)) == 3


def test_one_dead_channel_does_not_hide_the_shocks():
    """The consensus is a median across channels, so a channel that
    flatlines is one vote."""
    live = record(events=(0.3, 1.0), channels=3)
    values = np.array(live.ordinate)
    values[1] = 0.0
    with_dead = TimeHistory(live.abscissa, values,
                            response_dof=list(live.response_dof),
                            ordinate_dim=list(live.ordinate_dim))
    assert len(shocks.find(with_dead)) == 2


def test_one_screaming_channel_does_not_invent_them():
    """And the same median the other way: a channel forty dB over its
    neighbors does not get to answer for the record."""
    quiet = record(events=(), channels=3)
    values = np.array(quiet.ordinate)
    values[0] *= 100.0
    with_loud = TimeHistory(quiet.abscissa, values,
                            response_dof=list(quiet.response_dof),
                            ordinate_dim=list(quiet.ordinate_dim))
    assert shocks.find(with_loud) == ()


def test_a_drifting_baseline_is_not_a_shock():
    """The mean is removed within each hop, so a slow drift or a DC
    offset carries no level."""
    drifting = record(events=(1.0,))
    values = np.array(drifting.ordinate)
    values += 50.0 * np.linspace(0.0, 1.0, values.shape[1])[None, :]
    with_drift = TimeHistory(drifting.abscissa, values,
                             response_dof=list(drifting.response_dof),
                             ordinate_dim=list(drifting.ordinate_dim))
    assert len(shocks.find(with_drift)) == 1


def test_digital_silence_between_events_still_gives_a_usable_floor():
    """A synthesized or gated record has exact zeros in the gaps, so its
    quantile floor is negative infinity and every threshold taken from
    it arms on the first bit of dither. The dynamic-range guard is what
    keeps that record readable."""
    samples = round(2.4 * RATE)
    t = np.arange(samples) / RATE
    row = np.zeros(samples)
    for at in (0.3, 1.0, 1.7):
        since = t - at
        live = (since >= 0) & (since < 0.25)
        row[live] += PEAK * np.exp(
            -0.02 * 2 * np.pi * 78.0 * since[live]) * np.sin(
            2 * np.pi * 78.0 * since[live])
    silent = TimeHistory(t, row[None, :], response_dof=['101Z+'],
                         ordinate_dim=['acceleration'])
    level, _hop = shocks.envelope(silent)
    assert level.min() < -200.0, 'the quiet really is digital silence'
    assert shocks.floor_of(level) > level.max() - shocks.DYNAMIC - 1e-9
    assert len(shocks.find(silent)) == 3


def test_two_hits_close_together_are_one_window():
    """A double hit, and a ringdown that dips through the release for a
    moment, look the same from here. Two windows would analyze the
    second half of one shock as though it were a shock of its own."""
    close = record(events=(1.0, 1.01))
    assert len(shocks.find(close)) == 1


def test_windows_never_overlap():
    """Two windows sharing samples would count one shock's ringdown as
    part of the next shock's event."""
    found = shocks.find(record(events=(0.3, 0.9, 1.5, 2.1), span=2.8))
    for earlier, later in pairwise(found):
        assert earlier.stop <= later.start


# ---- the window itself ---------------------------------------------------


def test_a_window_knows_its_samples():
    window = Shock(1.0, 0.5)
    assert window.stop == 1.5
    assert window.bounds(1000.0) == (1000, 1500)
    assert window.fits(2000, 1000.0)
    assert not window.fits(1200, 1000.0)


def test_a_window_is_pulled_inside_its_record():
    clipped = Shock(1.8, 0.5).clipped(2000, 1000.0)
    assert clipped.stop == pytest.approx(2.0)
    assert clipped.start == pytest.approx(1.8)


def test_a_window_with_no_length_is_refused():
    with pytest.raises(ValueError, match='some length'):
        Shock(0.0, 0.0)


def test_a_window_before_the_record_is_refused():
    with pytest.raises(ValueError, match='before the record'):
        Shock(-1.0, 0.5)


# ---- against the real file ----------------------------------------------


@pytest.fixture
def run():
    return visualdynamics.import_file(fixture_path('plate', 'shock.nc4'))


def test_the_four_shocks_in_the_run_are_found(run):
    found = shocks.find(run['time_data'])
    assert len(found) == 4


def test_each_found_window_holds_the_pulse_the_generator_put_there(run):
    """The generator spaces the events evenly and puts each pulse a
    fixed way into its own segment; the windows have to bracket those."""
    from testdata.generate_plate import PULSE, QUIET, SHOCKS
    from testdata.generate_plate import RATE as GEN_RATE

    per_shock = round((PULSE + 2 * QUIET) * GEN_RATE)
    found = shocks.find(run['time_data'])
    for k in range(SHOCKS):
        at = (k * per_shock + round(QUIET * GEN_RATE)) / GEN_RATE
        assert found[k].start < at < found[k].stop


def test_the_run_gives_one_spectrum_per_channel_per_shock(run):
    history = run['time_data']
    history.shocks = shocks.find(history)
    spectra = history.compute_srs()
    assert spectra.num_records == history.num_records * 4
    assert spectra.block[:4] == ['shock 1', 'shock 2', 'shock 3', 'shock 4']


def test_the_spectra_step_up_with_the_shocks(run):
    """The generator walks the level up by a tenth each event. The
    filter is linear, so the spectra have to walk up with it — which is
    a statement about the windows as much as about the filter, since a
    window that clipped an event would break it."""
    history = run['time_data']
    history.shocks = shocks.find(history)
    spectra = history.compute_srs()
    rows = [i for i, dof in enumerate(spectra.response_dof) if dof == '113Z+']
    peaks = [spectra.ordinate[i].max() for i in rows]
    ratios = [p / peaks[0] for p in peaks]
    assert ratios == pytest.approx([1.0, 8 / 7, 9 / 7, 10 / 7], rel=0.02)


def test_the_windows_survive_a_round_trip(run, tmp_path):
    history = run['time_data']
    history.shocks = shocks.find(history)
    visualdynamics.save(history, tmp_path / 'run')
    back = visualdynamics.load(tmp_path / 'run.vdyn')
    assert back.shocks == history.shocks


def test_a_history_that_was_never_looked_at_has_no_shocks(run, tmp_path):
    history = run['time_data']
    visualdynamics.save(history, tmp_path / 'run')
    assert visualdynamics.load(tmp_path / 'run.vdyn').shocks is None


# ---- one length for the series ------------------------------------------


def _leveled(peaks=(0.25, 0.55, 1.0)):
    """A series walked up in level, which is how shock tests are run.

    Same pulse, same ringdown, different amplitude — so the events are
    the same length physically and differ only in how long each takes
    to fall back to the floor, which is what makes their windows
    disagree when each is sized on its own.
    """
    rng = np.random.default_rng(3)
    samples = round(2.4 * RATE)
    t = np.arange(samples) / RATE
    row = rng.normal(0.0, NOISE, samples)
    for at, scale in zip((0.3, 1.0, 1.7), peaks):
        since = t - at
        live = since >= 0
        row[live] += PEAK * scale * np.exp(
            -0.02 * 2 * np.pi * 78.0 * since[live]) * np.sin(
            2 * np.pi * 78.0 * since[live])
    return TimeHistory(t, np.array([row, row]),
                       response_dof=['101Z+', '102Z+'],
                       ordinate_dim=['acceleration'] * 2)


def test_a_series_walked_up_in_level_gets_windows_of_one_length():
    """The point of COMMON_LENGTH. The run above RELEASE is measured
    against the floor, so a harder hit takes longer to fall back even
    though the damping that sets the ringdown never changed — leaving
    window length tracking level instead of physics."""
    history = _leveled()
    per_event = shocks.find(history, common=False)
    assert len(per_event) == 3
    spread = max(s.duration for s in per_event) \
        - min(s.duration for s in per_event)
    assert spread > 0.01, 'the levels really do pull the windows apart'

    together = shocks.find(history, common=True)
    assert len(together) == 3
    assert shocks.uniform(together), 'and one length puts them back'


def test_the_common_length_is_the_longest_event_never_the_shortest():
    """A shared length that truncates any event is strictly worse than
    per-event windows: the event it cuts is the one whose spectrum then
    reads low at the bottom of the band."""
    history = _leveled()
    per_event = shocks.find(history, common=False)
    together = shocks.find(history, common=True)
    assert together[0].duration >= max(s.duration for s in per_event)


def test_each_window_still_opens_before_its_own_shock():
    """One length is shared; the starts stay with their events."""
    events = (0.3, 1.0, 1.7)
    for window, at in zip(shocks.find(_leveled(), common=True), events):
        assert window.start < at < window.stop


def test_windows_of_one_length_still_do_not_overlap():
    history = _leveled()
    found = shocks.find(history, common=True)
    for one, next_one in pairwise(found):
        assert one.stop <= next_one.start


def test_close_events_cap_the_length_for_all_of_them():
    """The tightest gap is everybody's ceiling — capping just the pair
    that collided would put the series back where it started."""
    # sized against the detector rather than guessed: an isolated event
    # of this shape wants a window wider than the spacing below, so the
    # neighbors are what decides the length
    alone = shocks.find(record(events=(1.0,), decay=0.03, span=2.4))
    wanted = alone[0].duration
    events = (0.3, 1.3, round(1.3 + wanted * 0.93, 4))
    found = shocks.find(record(events=events, decay=0.03, span=2.6))
    assert len(found) == 3, 'three events, not two merged'
    assert shocks.uniform(found), 'the close pair shortened all of them'
    assert found[0].duration < wanted, 'and shortened them below what '\
        'the same event alone would have been given'
    for one, next_one in pairwise(found):
        assert one.stop <= next_one.start


# ---- holding them apart --------------------------------------------------


def test_a_moved_window_is_stopped_at_its_neighbor():
    """Drag an edge into the next event and it stops at the edge. The
    event nobody touched does not move."""
    was = (Shock(0.0, 0.2), Shock(0.5, 0.2), Shock(1.0, 0.2))
    pulled = (was[0], Shock(0.1, 0.6), was[2])     # left edge back into #1
    held = shocks.held_apart(pulled, moved=1)
    assert held[0] == was[0], 'the untouched neighbor stayed put'
    assert held[1].start == pytest.approx(0.2), 'stopped at the edge'
    assert held[1].stop == pytest.approx(0.7)


def test_an_overlapping_pair_with_no_author_cuts_the_earlier_one():
    """A list arriving from somewhere with no single edit behind it."""
    held = shocks.held_apart((Shock(0.0, 0.8), Shock(0.5, 0.2)))
    assert held[0].stop == pytest.approx(0.5)
    assert held[1] == Shock(0.5, 0.2), 'the later one is untouched'


def test_holding_them_apart_never_cascades():
    """One bad edit must not walk down the whole record."""
    held = shocks.held_apart(
        (Shock(0.0, 5.0), Shock(0.5, 0.2), Shock(1.0, 0.2)))
    assert held[1] == Shock(0.5, 0.2)
    assert held[2] == Shock(1.0, 0.2)


def test_a_series_at_one_length_widens_together_until_it_runs_out():
    """What a drag does in common mode."""
    was = (Shock(0.0, 0.2), Shock(0.5, 0.2), Shock(1.0, 0.2))
    wider = shocks.same_length(was, 0.4)
    assert [s.duration for s in wider] == [pytest.approx(0.4)] * 3

    too_wide = shocks.same_length(was, 5.0)
    assert shocks.uniform(too_wide)
    assert too_wide[0].duration == pytest.approx(0.5), 'the tightest gap'


def test_a_shared_length_never_moves_a_start():
    """Only the length is the series'. A start is the event's own, so
    resizing one window must not walk the others along the record."""
    was = (Shock(0.3, 0.2), Shock(0.9, 0.2), Shock(1.5, 0.2))
    wider = shocks.same_length(was, 0.35)
    assert [s.start for s in wider] == [0.3, 0.9, 1.5]


def test_the_shared_length_stops_at_the_end_of_the_record():
    was = (Shock(0.0, 0.2), Shock(1.0, 0.2))
    capped = shocks.same_length(was, 5.0, limit=1.3)
    assert capped[1].stop == pytest.approx(1.3)
    assert shocks.uniform(capped)


# ---- the ringdown is followed, not fractioned ----------------------------


def _one_channel_still_ringing(events=(0.5, 1.9), span=3.2):
    """Most channels die fast; one lightly damped low mode rings on.

    The drone stress set in miniature: the median envelope settles in
    tens of milliseconds while a single channel's 60 Hz mode is still
    writing the low end of its own spectrum. A window sized on the
    median cuts that channel's SRS short — measurably, which is what
    the test below measures.
    """
    rng = np.random.default_rng(11)
    samples = round(span * RATE)
    t = np.arange(samples) / RATE
    rows = []
    for c in range(4):
        row = rng.normal(0.0, NOISE, samples)
        for at in events:
            since = t - at
            live = since >= 0
            row[live] += PEAK * np.exp(
                -0.05 * 2 * np.pi * 400.0 * since[live]) * np.sin(
                2 * np.pi * 400.0 * since[live])
        rows.append(row)
    ringer = rng.normal(0.0, NOISE, samples)
    for at in events:
        since = t - at
        live = since >= 0
        # a tenth of the broadband peak, decaying ten times slower:
        # invisible to the median, decisive for its own SRS
        ringer[live] += 0.1 * PEAK * np.exp(
            -0.02 * 2 * np.pi * 60.0 * since[live]) * np.sin(
            2 * np.pi * 60.0 * since[live])
    rows.append(ringer)
    return TimeHistory(t, np.array(rows),
                       response_dof=[f'20{c}Z+' for c in range(5)],
                       ordinate_dim=['acceleration'] * 5)


def test_the_window_follows_one_channel_ringing_under_the_median():
    """The reading that matters: the ringing channel's SRS from the
    found window, against the same channel given all the room there
    is. A tail taken as a fraction of the run left this 7 dB low on
    the drone stress set; the follow closes it."""
    from visualdynamics.core import srs

    history = _one_channel_still_ringing()
    found = shocks.find(history)
    assert len(found) == 2
    values = np.asarray(history.ordinate)
    rate = float(history.sample_rate)
    frequencies = srs.octave_frequencies(20, 500)
    worst = 0.0
    edges = [s.start for s in found] + [values.shape[-1] / rate]
    for k, shock in enumerate(found):
        first = round(shock.start * rate)
        clear = round(edges[k + 1] * rate)
        room = srs.maximax(values[-1, first:clear], frequencies, rate)
        got = srs.maximax(
            values[-1, first:first + shock.samples(rate)],
            frequencies, rate)
        worst = min(worst, float(20 * np.log10(got / room).min()))
    assert worst > -0.5, (
        f'the ringing channel reads {worst:.2f} dB low — its ringdown '
        'was cut where the median went quiet')


def test_an_impulse_like_record_keeps_its_modest_windows():
    """The follow must cost nothing where there is nothing to follow:
    a record whose ringing dies inside tens of milliseconds gets the
    tail-fraction margin it always had, not a stretched window. (The
    default `record()` is *not* such a record — its ringdown is
    visible for hundreds of milliseconds, and the follow now honestly
    keeps it; hence the tenfold damping here.)"""
    found = shocks.find(record(events=(0.3, 1.0, 1.7), decay=0.2))
    for window in found:
        assert window.duration < 0.15, (
            f'{window.duration:.3f} s for an event whose ringdown is '
            'over in tens of milliseconds')

