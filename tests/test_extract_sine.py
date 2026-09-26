"""Sine extraction: the tracking-filter reading, graded two ways.

The synthetic tests plant known tones in known noise and hold the
recovered numbers; the oracle tests (skipped where the stressdata is
absent) grade the same code against real controller runs and the
controller's own tracked-amplitude record.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.sine import (
    SineLevel,
    SineSweepSpecification,
    SineTone,
    extract_sine,
    find_tone,
)

FS = 4096.0
STRESS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'stressdata', 'plate')


def _spec():
    up = SineTone('Up', 1.0, [100.0, 800.0],
                  [[2.0, 3.0], [2.0, 3.0]], [0], [100.0])
    down = SineTone('Down', 2.0, [600.0, 200.0],
                    [[1.0, 1.0], [1.0, 1.0]], [0], [-80.0])
    return SineSweepSpecification([up, down], ['101Z+', '104Z+'],
                                  ordinate_unit='m/s**2')


def _recording(spec, seconds=12.0, noise=0.0, seed=7):
    """The specification, played: each tone at its own start time with
    its own amplitude, plus gaussian noise at `noise` RMS."""
    rng = np.random.default_rng(seed)
    samples = int(seconds * FS)
    signal = rng.standard_normal((2, samples)) * noise
    for tone in spec.tones:
        arg = tone.argument(1.0 / FS)
        start = int(tone.start_time * FS)
        n = min(len(arg), samples - start)
        for channel in range(2):
            amplitude = tone.amplitude[0, channel]
            signal[channel, start:start + n] += (amplitude
                                                 * np.cos(arg[:n]))
    return TimeHistory(
        abscissa=np.arange(samples) / FS, ordinate=signal,
        response_dof=['101Z+', '104Z+'],
        ordinate_dim=['acceleration'] * 2,
        ordinate_unit=['m/s**2'] * 2)


def test_planted_tones_come_back_at_their_amplitudes():
    spec = _spec()
    levels = extract_sine(_recording(spec), spec)
    assert levels.tone_names == ['Up', 'Down'], \
        'one extraction, one object, the tones inside'
    for level, planted in zip(levels, ([2.0, 3.0], [1.0, 1.0])):
        assert level.onset == pytest.approx(
            spec.tone(level.tone).start_time, abs=2e-3), \
            'the matched filter lands on the planted start'
        mid = slice(len(level.abscissa) // 4, -len(level.abscissa) // 4)
        for row, amplitude in enumerate(planted):
            measured = np.abs(level.ordinate[row, mid])
            # the two tones cross in frequency; under the tracking
            # demodulation each read the other as noise around the
            # crossing (0.4 dB, and this tolerance was 5%) — the joint
            # Vold-Kalman solve separates them, so every line holds
            assert np.median(measured) == pytest.approx(
                amplitude, rel=0.002), \
                f'{level.tone} channel {row}: the planted amplitude'
            assert np.abs(measured - amplitude).max() < 0.03 * amplitude, \
                f'{level.tone} channel {row}: worst line off by more than 3%'
    assert np.all(np.diff(levels.tone('Down').abscissa) > 0), \
        'a descending sweep still reads ascending in frequency'
    for level in levels:
        assert level.seconds is not None and \
            np.all(np.diff(level.seconds[np.argsort(level.seconds)]) >= 0), \
            'every line is stamped with the second it was measured'


def test_two_tones_that_stay_close_are_separated():
    """The case the Vold-Kalman solve was built for (Brandon,
    2026-09-03: the last analysis before a report should use the best
    estimator there is). The real sine run's Sweep Up (100-800 Hz at
    50 Hz/s) and Log Octave (200-400 Hz at 10 oct/min, from 1 s) sit
    within 20 Hz of each other for two seconds and within 50 Hz for
    six — inside each other's tracking filter for a long stretch, not
    a crossing instant. Planted at known amplitudes, the tracking
    demodulation read both 2 to 2.5 dB high through that stretch with
    excursions to -23 dB (and so does the controller's own tracker,
    which is why the oracle test below compares only where tones are
    apart). The joint solve recovers both to 0.000 dB."""
    up = SineTone('Up', 0.0, [100.0, 800.0], [[2.5, 2.5], [2.5, 2.5]],
                  [0], [50.0])
    log = SineTone('Log', 1.0, [200.0, 400.0], [[2.3, 2.3], [2.3, 2.3]],
                   [1], [10.0])
    spec = SineSweepSpecification([up, log], ['101Z+', '104Z+'],
                                  ordinate_unit='m/s**2')
    levels = extract_sine(_recording(spec, seconds=14.0), spec,
                          onsets={'Up': 0.0, 'Log': 1.0})
    dt = 1.0 / FS
    clock = np.arange(0.0, 14.0, dt)
    paths = {t.name: np.interp(clock, t.trajectory(dt)[0] + t.start_time,
                               t.trajectory(dt)[1], left=np.nan,
                               right=np.nan) for t in spec.tones}
    for level, amplitude in zip(levels, (2.5, 2.3)):
        other = next(n for n in paths if n != level.tone)
        gap = np.abs(np.interp(level.seconds, clock, paths[other],
                               left=np.nan, right=np.nan) - level.abscissa)
        close = gap < 20.0
        assert close.sum() > 50, 'the planted pair really is close for a while'
        error_db = 20 * np.log10(np.abs(level.ordinate[0]) / amplitude)
        assert np.abs(error_db[close]).max() < 0.1, (
            f'{level.tone}: {np.abs(error_db[close]).max():.2f} dB off where '
            'the other tone sits within 20 Hz')
        assert np.abs(error_db).max() < 0.1, f'{level.tone}: off somewhere'


def test_extraction_holds_under_loud_noise():
    """Noise 4x the quieter tone's amplitude: the tracking filter's
    whole job. Medians hold to a fraction of a dB."""
    spec = _spec()
    levels = extract_sine(_recording(spec, noise=4.0), spec)
    down = levels.tone('Down')
    mid = slice(len(down.abscissa) // 4, -len(down.abscissa) // 4)
    measured = np.median(np.abs(down.ordinate[0, mid]))
    error_db = 20 * np.log10(measured / 1.0)
    assert abs(error_db) < 1.0, \
        f'planted 1.0 under 4.0-RMS noise read back {error_db:+.2f} dB off'


def test_a_short_recording_reports_its_coverage():
    """The record ends mid-sweep: the level covers what was run, and
    says nothing further — the coverage rule, not an error."""
    spec = _spec()
    levels = extract_sine(_recording(spec, seconds=5.0), spec,
                          tones=['Up'])
    top = levels.tone('Up').abscissa.max()
    assert top < 500.0, 'the sweep was cut off well before 800 Hz'
    assert top > 250.0, 'but a real stretch of it was recorded'


def test_missing_control_channels_refuse_by_name():
    spec = _spec()
    history = _recording(spec)
    short = TimeHistory(
        abscissa=history.abscissa, ordinate=history.ordinate[:1],
        response_dof=['101Z+'], ordinate_dim=['acceleration'],
        ordinate_unit=['m/s**2'])
    with pytest.raises(ValueError, match='104Z'):
        extract_sine(short, spec)


def test_an_onset_override_skips_the_search():
    spec = _spec()
    history = _recording(spec)
    levels = extract_sine(history, spec, tones=['Up'],
                          onsets={'Up': 1.0})
    assert levels.tone('Up').onset == 1.0


def test_find_tone_bounds_its_search_when_told():
    spec = _spec()
    history = _recording(spec)
    onset = find_tone(np.asarray(history.ordinate), 1.0 / FS,
                      spec.tone('Up'), search=(0.5, 1.5))
    assert onset == pytest.approx(1.0, abs=2e-3)


def test_the_level_set_round_trips_through_native(tmp_path):
    from visualdynamics.core.sine import SineLevelSet

    spec = _spec()
    levels = extract_sine(_recording(spec), spec)
    path = str(tmp_path / 'levels.vdyn')
    visualdynamics.io.save(levels, path)
    back = visualdynamics.io.load(path)
    assert isinstance(back, SineLevelSet)
    assert back == levels
    up = back.tone('Up')
    assert isinstance(up, SineLevel)
    assert up.onset == pytest.approx(levels.tone('Up').onset)
    assert np.allclose(up.seconds, levels.tone('Up').seconds)


def test_the_project_verb_extracts_and_links():
    spec = _spec()
    project = visualdynamics.Project('sine run')
    project.add('Time History', _recording(spec))
    project.add('Sine Specification', spec)
    names = project.extract_sine('Time History')
    assert names == ['Sine Levels'], 'one extraction, one object'
    assert project['Sine Levels'].tone_names == ['Up', 'Down']
    assert 'Time History' in (project.group_of('Sine Levels') or [])


# ---- the oracle: real controller runs -----------------------------------

needs_stress = pytest.mark.skipif(
    not os.path.exists(os.path.join(STRESS, 'sine.nc4')),
    reason='stressdata not generated on this machine')


def _controller_levels(control_npz, tone_index):
    """The controller's own tracked amplitude, laid over frequency.

    Its per-block amplitude records and its frequency trajectory share
    a clock, so amplitude-versus-frequency needs no time alignment —
    both are functions of the sweep's own monotonic frequency."""
    truth = np.load(control_npz)
    keys = sorted((int(k.rsplit('_', 1)[1]), k) for k in truth.files
                  if k.startswith('control_response_amplitudes'))
    tracked = np.concatenate([truth[k][tone_index] for _, k in keys],
                             axis=-1)
    frequency = truth['control_response_frequencies'][tone_index]
    n = min(tracked.shape[-1], len(frequency))
    return frequency[:n], tracked[:, :n]


def _against_controller(level, frequency, tracked, keep=None):
    """Median dB between the extraction and the controller's tracker,
    per channel, compared where both actually swept — and, with
    `keep`, only on the lines it marks."""
    moving = np.flatnonzero(np.abs(np.diff(frequency)) > 0)
    # margins by span, not by absolute frequency: a 390-410 near-dwell
    # has no room for percentage margins
    span = frequency[moving].max() - frequency[moving].min()
    lo = frequency[moving].min() + 0.1 * span
    hi = frequency[moving].max() - 0.1 * span
    inside = (level.abscissa > lo) & (level.abscissa < hi)
    if keep is not None:
        inside &= keep
    out = []
    for row in range(level.ordinate.shape[0]):
        order = np.argsort(frequency[moving])
        reference = np.interp(level.abscissa[inside],
                              frequency[moving][order],
                              tracked[row][moving][order])
        ratio = np.abs(level.ordinate[row, inside]) / np.maximum(
            reference, 1e-12)
        out.append(20 * np.log10(np.median(ratio)))
    return out


@needs_stress
def test_the_real_sine_run_sits_at_or_under_the_controller():
    """Four real tones over eight control channels and four shakers.

    The controller's live tracker reads *high* — its quadrature filter
    counts the other tones' leakage as amplitude — so the structural
    claim is one-sided: the extraction sits at or below the tracker,
    never meaningfully above. An *agreement* oracle, not a truth
    oracle (AGENTS.md): where the nearest other tone is more than
    50 Hz away the two read the same thing and agree within the
    envelope measured here (-0.9 to -4.8 dB when the tracking
    demodulation was the reader; the joint solve sits a little closer);
    where another tone is within 20 Hz the tracker double-counts and
    the extraction reads the truth, as the planted close-tones test
    proves — there the only claim is at-or-under. Point-for-point
    agreement is never the claim: on a resonant plate under live
    control the physical amplitude swings several dB within small
    frequency spans."""
    project = visualdynamics.Project('real')
    project.import_file(os.path.join(STRESS, 'sine.nc4'))
    project.extract_sine(project.time_history)
    levels = project['Sine Levels']
    assert len(levels) == 4
    spec = project.sine_sweep_specification
    control_npz = os.path.join(STRESS, 'sine_control.npz')
    order = [str(n) for n in np.load(control_npz)['names']]
    dt = 1.0 / FS
    clock = np.arange(0.0, 16.0, dt)
    paths = {t.name: np.interp(clock, t.trajectory(dt)[0] + t.start_time,
                               t.trajectory(dt)[1], left=np.nan,
                               right=np.nan) for t in spec.tones}
    for level in levels:
        frequency, tracked = _controller_levels(
            control_npz, order.index(level.tone))
        tone = spec.tone(level.tone)
        seconds = level.seconds - (level.onset - tone.start_time)
        gap = np.nanmin(np.stack([
            np.abs(np.interp(seconds, clock, paths[o], left=np.nan,
                             right=np.nan) - level.abscissa)
            for o in paths if o != level.tone]), axis=0)
        apart, close = gap > 50.0, gap < 20.0
        if apart.sum() >= 50:
            for row, error in enumerate(
                    _against_controller(level, frequency, tracked, apart)):
                assert -6.0 < error < 0.5, (
                    f'{level.tone} channel {row}: {error:+.2f} dB from '
                    "the controller's own tracker, tones apart")
        for row, error in enumerate(
                _against_controller(level, frequency, tracked, close)):
            assert error < 0.5, (
                f'{level.tone} channel {row}: {error:+.2f} dB *above* '
                'the tracker where it double-counts a close tone')


@needs_stress
def test_the_real_mixed_run_reads_the_sweep_under_the_random():
    """The 0.5-amplitude sweep under a random 13.5 dB louder — the
    problem the arc exists for, on real data.

    Under that much noise the controller's tracker bias grows to
    several dB (it reads the random as sine and backs its drive off
    accordingly), so the one-sided envelope is wider. The physical
    check is the second assertion: the extracted levels bracket the
    0.5 target the way an eight-channel four-shaker compromise must —
    some channels above, some below, the median within a few dB."""
    project = visualdynamics.Project('mixed')
    project.import_file(os.path.join(STRESS, 'mixed.nc4'))
    project.extract_sine(project.time_history)
    level = project['Sine Levels'].levels[0]
    control_npz = os.path.join(STRESS, 'mixed_sine_control.npz')
    frequency, tracked = _controller_levels(control_npz, 0)
    errors = _against_controller(level, frequency, tracked)
    for row, error in enumerate(errors):
        # -12: the joint solve's narrower noise bandwidth carries less
        # positive noise bias than the tracking average, so it sits
        # half a dB further under a tracker that reads the random as
        # sine (measured 2026-09-03: -4.2 to -8.9 dB by channel)
        assert -12.0 < error < 0.5, (
            f'channel {row}: {error:+.2f} dB from the controller '
            'under the random')
    mid = (level.abscissa > 170) & (level.abscissa < 730)
    medians = [np.median(np.abs(level.ordinate[row, mid]))
               for row in range(level.ordinate.shape[0])]
    spread_db = 20 * np.log10(np.median(medians) / 0.5)
    assert abs(spread_db) < 5.0, (
        f'physically implausible: the channel-median level sits '
        f'{spread_db:+.2f} dB from the 0.5 target')


@needs_stress
def test_a_sweep_over_a_virtual_point_is_read_at_its_rows():
    """The same mixed run, with the sweep controlling a virtual point —
    three rows of a response transformation over the eight control
    channels — while the random controls the raw channels (Brandon,
    2026-09-24: a run like this did not load). The levels are read from
    the rows' own time histories, which ride the recording beside the
    raw channels, and graded against the controller's tracker on the
    same rows: the untransformed run's envelope holds (measured -3.5 to
    -3.8 dB by row).

    Regenerate with `generate_plate_sine.py mixed_transformed` in the
    generators repository."""
    path = os.path.join(STRESS, 'mixed_transformed.nc4')
    if not os.path.exists(path):
        pytest.skip('regenerate with generate_plate_sine.py mixed_transformed')
    project = visualdynamics.Project('mixed_transformed')
    project.import_file(path)
    spec = project.sine_sweep_specification
    assert spec.response_dof == ['1', '2', '3']
    project.extract_sine(project.time_history)
    level = project['Sine Levels'].levels[0]
    assert level.ordinate.shape[0] == 3, 'one level per row'
    control_npz = os.path.join(STRESS, 'mixed_transformed_sine_control.npz')
    frequency, tracked = _controller_levels(control_npz, 0)
    for row, error in enumerate(_against_controller(level, frequency,
                                                    tracked)):
        assert -12.0 < error < 0.5, (
            f'row {row + 1}: {error:+.2f} dB from the controller '
            'under the random')


# ---- the GUI reading ----------------------------------------------------

def _curve_count(window):
    import pyqtgraph as pg

    plots = [item for item in window.data_pane.graphics.ci.items
             if isinstance(item, pg.PlotItem)]
    return sum(1 for plot in plots for item in plot.listDataItems()
               if not getattr(item, 'is_zone_edge', False))


def test_the_spec_draws_its_tones_with_bands(window, pump):
    spec = _spec()
    for tone in spec.tones:
        tone.limits['warning_lower'] = tone.amplitude * 10 ** (-3 / 20)
        tone.limits['warning_upper'] = tone.amplitude * 10 ** (3 / 20)
    window.add_object('Sine Specification', spec)
    window._item_for_object('Sine Specification').setSelected(True)
    # the stage is the spec's default reading; this test holds the
    # flat one. A programmatic setChecked emits no click, so the
    # re-render a real click causes is asked for explicitly.
    window.data_pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    assert _curve_count(window) == 1, \
        'one tone at one DOF: the tone from its box, the DOF from the pair box'
    box = window.data_pane.event_box
    assert [box.itemText(i) for i in range(box.count())] == ['Up', 'Down']
    assert window.data_pane.pair_box.count() == 2


def test_levels_draw_over_the_spec_and_the_bars_judge_them(window, pump):
    spec = _spec()
    for tone in spec.tones:
        tone.limits['warning_lower'] = tone.amplitude * 10 ** (-3 / 20)
        tone.limits['warning_upper'] = tone.amplitude * 10 ** (3 / 20)
    window.add_object('Sine Specification', spec)
    window.add_object('Time History', _recording(spec))
    window.add_object('Sine Levels',
                      extract_sine(window.project['Time History'], spec))
    window.tree.clearSelection()
    window._item_for_object('Sine Specification').setSelected(True)
    window._item_for_object('Sine Levels').setSelected(True)
    pump()
    # the comparison's own default is the stage: measured paths over
    # the muted targets and their zones
    actors = list(window.data_pane.waterfall_plotter.actors)
    assert any(name.startswith('sine-level-') for name in actors), \
        'the measured paths ride the stage'
    assert any('zone' in name for name in actors), \
        'over the specification zones'
    window.data_pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    assert _curve_count(window) >= 2, \
        'flat: the level and the requirement of one DOF share the plot'
    pairs = window.data_pane.pair_box
    assert pairs.count() == 2, 'and the other DOF is a pick away'
    window.data_pane.srs_view = 'error'
    window.render_current()
    pump()
    assert window.bar_chart is not None
    text = window.bar_chart.summary.toPlainText()
    assert 'within' in text or 'dB' in text


def test_every_sine_reading_with_two_forms_offers_the_2d_3d_toggle(window,
                                                                  pump):
    """The toggle is on the bar wherever the stage and the flat plot
    are both readings: the specification alone, the levels alone, and
    the two together. The tests above flip it with `setChecked`, which
    a hidden button accepts just the same — and the button was hidden:
    the renderer branched on the toggle without ever offering it, so
    the drawing followed whatever the last object had left it at
    (Brandon, 2026-09-25). The bars are one flat picture and do not
    offer it."""
    spec = _spec()
    for tone in spec.tones:
        tone.limits['warning_lower'] = tone.amplitude * 10 ** (-3 / 20)
        tone.limits['warning_upper'] = tone.amplitude * 10 ** (3 / 20)
    window.add_object('Sine Specification', spec)
    window.add_object('Time History', _recording(spec))
    window.add_object('Sine Levels',
                      extract_sine(window.project['Time History'], spec))
    pane = window.data_pane
    toggle = pane.waterfall_action

    def show(*names):
        window.tree.clearSelection()
        for name in names:
            window._item_for_object(name).setSelected(True)
        pump()

    # a flat reading first, so the button's state is not the default
    show('Time History')
    for names in (('Sine Specification',), ('Sine Levels',),
                  ('Sine Specification', 'Sine Levels')):
        show(*names)
        assert toggle.isVisible(), f'no 2D/3D button for {names}'
        # and it works both ways from the bar
        toggle.trigger()
        pump()
        flat = not pane._waterfall_page.isVisible()
        toggle.trigger()
        pump()
        assert flat != (not pane._waterfall_page.isVisible()), names
    pane.srs_view = 'error'
    window.render_current()
    pump()
    assert window.bar_chart is not None
    assert not toggle.isVisible(), 'the bars have no depth to add'


# ---- the report ---------------------------------------------------------

def test_the_sine_report_renders_its_comparison_and_bars():
    import json

    from visualdynamics.report import render_html

    spec = _spec()
    for tone in spec.tones:
        tone.limits['warning_lower'] = tone.amplitude * 10 ** (-3 / 20)
        tone.limits['warning_upper'] = tone.amplitude * 10 ** (3 / 20)
        tone.limits['abort_lower'] = tone.amplitude * 10 ** (-6 / 20)
        tone.limits['abort_upper'] = tone.amplitude * 10 ** (6 / 20)
    project = visualdynamics.Project('sine run')
    project.add('Time History', _recording(spec))
    project.add('Sine Specification', spec)
    project.extract_sine('Time History')
    project.set_basis(*list(project))
    name = project.generate_report('sine')
    html = render_html(project[name], project)
    payload = json.loads(html.split('type="application/json">')[1]
                         .split('</script>')[0])
    figures = [block for block in payload['blocks']
               if block.get('kind') == 'plot'
               and block.get('channels')]
    assert len(figures) >= 2, 'one comparison figure per tone'
    first = figures[-1]
    assert first['channels'][0]['zones'], \
        "the tone's warning and abort bands shade the figure"
    assert first['channels'][0]['responses'], \
        'the extracted level rides over the requirement'
    bars = [block for block in payload['blocks']
            if block.get('kind') == 'bars']
    assert bars and len(bars[0]['values']) == 4, \
        'two tones x two control channels, one signed bar each'
    assert bars[0]['low'] == -3.0 and bars[0]['high'] == 3.0


def test_the_calculator_offers_extraction_when_a_spec_exists(window, pump):
    """On the time data's calculator, gated by the presence of a sine
    specification — a fact about the objects, not the project type: a
    mixed run types Random Vibration and still carries the sweep."""
    spec = _spec()
    window.add_object('Time History', _recording(spec))
    labels = [label for _v, label, *_rest
              in window.acts_for(['Time History'])]
    assert 'Extract Sine Levels' not in labels, \
        'no specification, nothing to extract against'
    window.add_object('Sine Specification', spec)
    labels = [label for _v, label, *_rest
              in window.acts_for(['Time History'])]
    assert 'Extract Sine Levels' in labels
    window.extract_sine_levels('Time History')
    pump()
    assert 'Sine Levels' in window.objects
    from visualdynamics.core.sine import SineLevelSet
    assert isinstance(window.objects['Sine Levels'], SineLevelSet)
    assert window.objects['Sine Levels'].tone_names == ['Up', 'Down']


def test_the_spec_alone_takes_the_stage(window, pump):
    """Frequency, time, amplitude: the specification's own 3-D view,
    with the 3D toggle standing down to the flat reading."""
    spec = _spec()
    for tone in spec.tones:
        tone.limits['warning_lower'] = tone.amplitude * 10 ** (-3 / 20)
        tone.limits['warning_upper'] = tone.amplitude * 10 ** (3 / 20)
    window.add_object('Sine Specification', spec)
    window._item_for_object('Sine Specification').setSelected(True)
    pump()
    pane = window.data_pane
    assert pane._waterfall_page is not None and \
        pane._waterfall_page.isVisible(), 'the stage is the default view'
    names = [name for name in pane.waterfall_plotter.actors
             if name.startswith('sine-')]
    assert any('target' in name for name in names)
    assert any('warning_upper' in name for name in names), \
        'the band rides the stage with the tone'
    assert any('zone' in name for name in names), \
        'the warning zone shades the stage as it shades the flat plot'
    # the toggle stands down to the flat per-channel reading
    pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    assert not pane._waterfall_page.isVisible()
    assert _curve_count(window) == 1, \
        'the flat curve comes back: one tone at one DOF'



def test_the_rows_are_the_dofs_and_the_tone_is_on_the_bar(window, pump):
    """The specification expands into one row per control DOF, as
    every other object's rows are its channels, and the tone is the
    drop-down on the bar. Until 2026-09-25 it was the other way
    round: Brandon expanded a three-DOF specification expecting three
    rows and found one tone. Picking rows plots just those DOFs —
    records and modes habits — and the drop-down picks the tone."""
    spec = _spec()
    window.add_object('Sine Specification', spec)
    item = window._item_for_object('Sine Specification')
    item.setSelected(True)
    pump()
    grid = window.record_grids['Sine Specification']
    assert grid.kind == 'dof'
    assert grid.responses == ['101Z+', '104Z+']
    box = window.data_pane.event_box
    assert [box.itemText(i) for i in range(box.count())] == ['Up', 'Down']

    def targets():
        return [name for name in window.data_pane.waterfall_plotter.actors
                if name.startswith('sine-') and name.endswith('-target')]

    assert len(targets()) == 2, 'one tone, both DOFs on the stage'
    # flat, several DOFs are a thicket: the pair box on the bar picks
    # the one drawn, as the spectra comparison's does
    window.data_pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    pairs = window.data_pane.pair_box
    assert [pairs.itemData(i)[0] for i in range(pairs.count())] == \
        ['101Z+', '104Z+']
    assert _curve_count(window) == 1, 'one DOF at a time flat'
    pairs.setCurrentIndex(1)
    pump()
    assert '104Z+' in window.statusBar().currentMessage(), \
        'the pair box moves the flat reading to the other DOF'
    window.data_pane.waterfall_action.setChecked(True)
    grid.select_records([1])
    window.render_current()
    pump()
    assert len(targets()) == 1, 'one DOF picked, one requirement'
    assert 'the requirement at 1 DOF' in window.statusBar().currentMessage()
    window.data_pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    assert _curve_count(window) == 1, 'the flat reading honors the same pick'
    assert not window.data_pane.pair_action.isVisible(), \
        'one DOF chosen, nothing for the pair box to choose'
    # the drop-down moves the tone, the pick stays
    box.setCurrentIndex(1)
    pump()
    assert window.statusBar().currentMessage().startswith('Down at ')
    assert _curve_count(window) == 1


def test_the_comparison_stage_aligns_the_two_clocks():
    """A level's clock is recording seconds (onset included); the
    specification's is the environment's. On one stage the level's
    path must land on its tone's own start time, or the comparison
    draws a lie about when."""
    import pyvista as pv

    from visualdynamics.viz.sinespec import add_sine_stage

    spec = _spec()
    plain = _recording(spec)
    # two seconds of dead lead-in: the recording's clock now runs two
    # seconds ahead of the environment's, which is exactly the gap
    # the alignment exists to close — a fixture where the clocks
    # agree cannot falsify it (they did, and the broken shift passed)
    lead = int(2.0 * FS)
    shifted = np.concatenate(
        [np.zeros((2, lead)), np.asarray(plain.ordinate)], axis=1)
    history = TimeHistory(
        abscissa=np.arange(shifted.shape[1]) / FS, ordinate=shifted,
        response_dof=list(plain.response_dof),
        ordinate_dim=list(plain.ordinate_dim),
        ordinate_unit=list(plain.ordinate_unit))
    levels = extract_sine(history, spec)
    up = levels.tone('Up')
    assert up.onset == pytest.approx(3.0, abs=2e-3), \
        'found in recording time: the lead-in plus the start time'
    plotter = pv.Plotter(off_screen=True)
    arrays = add_sine_stage(plotter, specification=spec, levels=[up],
                            dof='101Z+')
    plotter.close()
    path = arrays['levels'][0]
    start = spec.tone('Up').start_time
    assert path['t'].min() == pytest.approx(start, abs=0.1), \
        "the level's path begins at its tone's start time"
    tone = next(t for t in arrays['tones'] if t['name'] == 'Up')
    assert abs(path['t'].min() - tone['t'].min()) < 0.1, \
        'measured and required share the clock'


def test_exceeding_lines_wear_the_abort_colors_on_the_stage():
    """A level driven over the upper abort marks those lines red on
    the stage, exactly as the 2-D comparison boxes them — and a level
    inside the band marks nothing."""
    import pyvista as pv

    from visualdynamics.viz.sinespec import add_sine_stage

    # one tone alone: where two tones cross, the documented debias
    # dip is real data briefly under -3 dB, and marking it would be
    # honest — this half needs a level that truly meets its spec
    only = SineTone('Up', 1.0, [100.0, 800.0],
                    [[2.0, 3.0], [2.0, 3.0]], [0], [100.0])
    spec = SineSweepSpecification([only], ['101Z+', '104Z+'],
                                  ordinate_unit='m/s**2')
    for tone in spec.tones:
        tone.limits['abort_upper'] = tone.amplitude * 10 ** (3 / 20)
        tone.limits['abort_lower'] = tone.amplitude * 10 ** (-3 / 20)
    history = _recording(spec)
    levels = extract_sine(history, spec)
    up = levels.tone('Up')

    def stripes(levels_shown):
        plotter = pv.Plotter(off_screen=True)
        add_sine_stage(plotter, specification=spec,
                       levels=levels_shown, dof='101Z+')
        found = {}
        for name, actor in plotter.actors.items():
            if 'exceed' not in name:
                continue
            mesh = actor.mapper.dataset
            # rectangles, not wedges: every cell is a quad whose four
            # corners sit on exactly two heights — the limit and the
            # stage's edge — two corners on each
            for c in range(mesh.n_cells):
                heights = np.round(
                    np.asarray(mesh.get_cell(c).points)[:, 2], 6)
                values, counts = np.unique(heights,
                                           return_counts=True)
                assert len(values) == 2 and list(counts) == [2, 2], \
                    f'cell {c} is not a rectangle: heights {heights}'
            bounds = mesh.bounds
            found[name.rsplit('-', 1)[-1]] = bounds[5] - bounds[4]
        plotter.close()
        return found

    assert not stripes([up]), \
        'the planted level meets its spec: no stripes at all'

    def scaled(factor):
        return SineLevel(
            abscissa=up.abscissa, ordinate=up.ordinate * factor,
            response_dof=list(up.response_dof),
            ordinate_dim=list(up.ordinate_dim),
            ordinate_unit=list(up.ordinate_unit),
            tone=up.tone, onset=up.onset, seconds=up.seconds)

    loud = stripes([scaled(3.0)])
    assert list(loud) == ['exceed_over'] and loud['exceed_over'] > 0.1, \
        'three times the target: a red stripe with real height, ' \
        'running from the abort limit to the ceiling'
    quiet = stripes([scaled(1 / 3.0)])
    assert list(quiet) == ['exceed_under'] and \
        quiet['exceed_under'] > 0.1, \
        'a third of the target: the blue stripe to the floor'
