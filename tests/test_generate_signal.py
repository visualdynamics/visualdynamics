"""Generate Signal: the project's verb, and its pane in the window.

Brandon, 2026-10-24: signals at the reference DOFs, made from a
specification or a channel table, before responses through a system.
The verb records its settings and the seed it drew, so Recompute makes
the same record and the journal replays it; the pane offers only the
kinds the object makes, previews as a setting changes, and its button
runs the verb (principle 13).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import bar_acts, select_objects

import visualdynamics
from visualdynamics.core import signals
from visualdynamics.core.channel_table import ChannelTable
from visualdynamics.core.sine import LOG, SineSweepSpecification, SineTone


def _target():
    return signals.flat_spectrum(['1X+', '2X+'], quantity='force',
                                 band=(20.0, 2000.0), level=1.0)


def _table():
    return ChannelTable({
        'channel': [1, 2, 3],
        'node': ['1', '2', '101'],
        'direction': ['X+', 'Y+', 'Z+'],
        'unit': ['N', 'N', 'g'],
        'role': ['reference', 'reference', 'response'],
        'channel_type': ['force', 'force', 'acceleration']})


def _sweep():
    g = 9.80665
    tone = SineTone('Up', 0.0, [20.0, 200.0], [[g], [2 * g]], [LOG], [8.0])
    return SineSweepSpecification([tone], ['1X+'], ordinate_unit='m/s**2')


# ---- the verb ---------------------------------------------------------------

def test_the_verb_records_its_settings_and_the_seed_it_drew():
    project = visualdynamics.Project()
    project.add('Target', _target())
    name = project.generate_signal('Target', 'random', duration=2.0,
                                   sample_rate=8192.0)
    made = project[name]
    assert made.ordinate.shape == (2, 16384)
    assert made.ordinate_dim == ['force', 'force']
    record = project.provenance[name]
    assert record['verb'] == 'generate_signal' and record['source'] == 'Target'
    seed = record['params']['seed']
    assert isinstance(seed, int), 'a seed drawn is kept'
    again = signals.generate(project['Target'],
                             signals.SignalSettings.from_dict(record['params']))
    assert np.array_equal(again.ordinate, made.ordinate)
    assert name in project.object_group_of('Target')


def test_recompute_makes_the_same_record_and_a_changed_target_is_stale():
    project = visualdynamics.Project()
    project.add('Target', _target())
    name = project.generate_signal('Target', 'burst', bursts=3,
                                   frame_length=1024, sample_rate=8192.0,
                                   seed=4)
    first = project[name].ordinate.copy()
    assert name not in project.stale()
    project['Target'].ordinate *= 2.0
    assert name in project.stale(), 'the levels it was drawn to moved'
    project.refresh(name)
    assert np.allclose(project[name].ordinate, np.sqrt(2.0) * first)
    assert name not in project.stale()


def test_a_channel_table_makes_an_impulse_at_its_references():
    project = visualdynamics.Project()
    project.add('Channels', _table())
    name = project.generate_signal('Channels', 'impulse', duration=1.0,
                                   sample_rate=10000.0, at=0.1, peak=5.0,
                                   width=0.004)
    made = project[name]
    assert made.response_dof == ['1X+', '2Y+'], 'the reference channels only'
    assert made.ordinate_dim == ['force', 'force']
    # the sampled half-sine of N = 40 intervals sums to cot(π/2N)
    assert made.ordinate[0].sum() == pytest.approx(
        5.0 / np.tan(np.pi / 80), rel=1e-9)
    pulse = project.generate_signal('Channels', 'pulse', duration=0.1,
                                    sample_rate=20000.0, peak=100.0,
                                    pulse_duration=0.005)
    assert project[pulse].ordinate.max() == pytest.approx(100.0, rel=1e-3)
    typed = project.generate_signal('Channels', 'random', duration=4.0,
                                    sample_rate=8192.0, band=(20.0, 500.0),
                                    level=2.0, seed=1)
    assert project[typed].ordinate[0].std() == pytest.approx(
        np.sqrt(2.0 * 480.0), rel=0.03)


def test_a_sine_specification_makes_its_tone():
    project = visualdynamics.Project()
    project.add('Sweep', _sweep())
    name = project.generate_signal('Sweep', 'sine', sample_rate=4096.0)
    assert project[name].response_dof == ['1X+']
    assert project.provenance[name]['params']['kind'] == 'sine'


def test_what_the_verb_refuses():
    project = visualdynamics.Project()
    project.add('Target', _target())
    project.add('Channels', ChannelTable({'channel': [1], 'node': ['1'],
                                          'direction': ['X+'], 'unit': ['N']}))
    with pytest.raises(ValueError, match='is not made from this object'):
        project.generate_signal('Target', 'impulse')
    with pytest.raises(ValueError, match='Nyquist'):
        project.generate_signal('Target', 'random', sample_rate=3000.0)
    with pytest.raises(ValueError, match="no channel's role is reference"):
        project.generate_signal('Channels', 'impulse')
    with pytest.raises(TypeError):
        project.generate_signal('Target', 'random', durations=3.0)
    assert 'generate_signal' in dict(project.verbs('Target'))
    assert 'generate_signal' not in dict(project.verbs(
        visualdynamics.TimeHistory(np.arange(8.0), np.zeros((1, 8)),
                                   response_dof='1X+')))


def test_settings_written_by_another_version_still_read():
    values = signals.SignalSettings(kind='burst', bursts=4).to_dict()
    values['a_setting_from_later'] = 3
    del values['taper']
    read = signals.SignalSettings.from_dict(values)
    assert read.kind == 'burst' and read.bursts == 4
    assert read.taper == signals.BURST_TAPER


# ---- the pane ---------------------------------------------------------------

def _lbf_table(units=('lbf', 'lbf')):
    return ChannelTable({
        'channel': [1, 2, 3],
        'node': ['1', '2', '101'],
        'direction': ['X+', 'Y+', 'Z+'],
        'unit': [*units, 'g'],
        'role': ['reference', 'reference', 'response'],
        'channel_type': ['force', 'force', 'acceleration']})


LBF = 4.4482216152605      # newtons in a pound-force


def test_a_typed_level_and_peak_are_in_each_channels_own_unit():
    # 1.0 typed for an lbf table gave 1 N**2/Hz: 63.1 N RMS over
    # 20-4000 Hz where 63.1 lbf was meant (the BARC table, 2026-10-29)
    project = visualdynamics.Project()
    project.add('Channels', _lbf_table(('lbf', 'N')))
    typed = project.generate_signal('Channels', 'random', duration=4.0,
                                    sample_rate=8192.0, band=(20.0, 500.0),
                                    level=2.0, seed=1)
    made = project[typed]
    rms = np.sqrt(2.0 * 480.0)
    assert made.ordinate[0].std() == pytest.approx(rms * LBF, rel=0.03)
    assert made.ordinate[1].std() == pytest.approx(rms, rel=0.03), \
        'the newton channel in newtons'
    impulse = project.generate_signal('Channels', 'impulse', duration=1.0,
                                      sample_rate=10000.0, at=0.1, peak=5.0,
                                      width=0.004)
    peaks = project[impulse].ordinate.max(axis=1)
    assert peaks == pytest.approx([5.0 * LBF, 5.0], rel=1e-9)
    pulse = project.generate_signal('Channels', 'pulse', duration=0.1,
                                    sample_rate=20000.0, peak=100.0,
                                    pulse_duration=0.005)
    assert project[pulse].ordinate.max(axis=1) == pytest.approx(
        [100.0 * LBF, 100.0], rel=1e-3)
    target = signals.target_of(project['Channels'], signals.SignalSettings(
        band=(20.0, 500.0), level=2.0))
    assert target.ordinate[:, 0] == pytest.approx([2.0 * LBF ** 2, 2.0])


def test_the_reference_unit_is_the_one_they_share():
    assert signals.reference_unit(_lbf_table()) == 'lbf'
    assert signals.reference_unit(_lbf_table(('lbf', 'N'))) == ''
    assert signals.reference_unit(_lbf_table(('', ''))) == 'N', \
        "SI's where none is written, as the level is then read"
    assert signals.reference_unit(_lbf_table(), ['1X+']) == 'lbf'


def _open(window, pump, name):
    select_objects(window, pump, name)
    window.generate_signal_act()
    pump()
    return window.data_pane.signal_panel


def test_the_act_is_on_the_bar_for_a_spectrum_and_opens_the_pane(window,
                                                                  pump):
    window.add_object('Target', _target())
    pump()
    select_objects(window, pump, 'Target')
    assert 'Generate Signal' in bar_acts(window.data_pane)
    panel = _open(window, pump, 'Target')
    assert panel.isVisible()
    kinds = [panel.kind_box.itemData(i) for i in range(panel.kind_box.count())]
    assert kinds == ['random', 'burst', 'periodic']
    assert panel.settings().sample_rate == 5120.0, '2.56 times 2000 Hz'
    assert 'Generate Signal' in window.statusBar().currentMessage()
    curves = [item for plot in window.data_pane.graphics.ci.items
              if hasattr(plot, 'listDataItems')
              for item in plot.listDataItems()]
    assert len(curves) >= 2, 'the target and the preview drawn'


def test_the_rows_follow_the_kind(window, pump):
    window.add_object('Target', _target())
    pump()
    panel = _open(window, pump, 'Target')

    def shown():
        return {k for k, e in panel._editors.items() if e.isVisibleTo(panel)}

    assert {'duration', 'frame', 'seed'} <= shown()
    assert 'bursts' not in shown() and 'peak' not in shown()
    panel.kind_box.setCurrentIndex(panel.kind_box.findData('burst'))
    pump()
    assert {'bursts', 'burst', 'taper'} <= shown()
    assert 'duration' not in shown()
    assert 'correlated' not in shown(), 'no cross terms to draw'


def test_generate_makes_the_object_with_its_seed_and_journals(window, pump):
    window.add_object('Target', _target())
    pump()
    panel = _open(window, pump, 'Target')
    panel._editors['duration'].setValue(2.0)
    pump()
    panel.apply_button.click()
    pump()
    made = [n for n in window.objects if n.startswith('Target Random')]
    assert made, 'the signal was made'
    record = window.project.provenance[made[0]]
    assert record['params']['duration'] == 2.0
    assert isinstance(record['params']['seed'], int)
    assert 'seed' in window.statusBar().currentMessage()
    line = next(entry for entry in window.project.journal
                if 'generate_signal' in entry)
    assert "'random'" in line and 'duration=2.0' in line
    assert f"seed={record['params']['seed']}" in line, \
        'the seed drawn is in the line, so the replay is the same record'
    assert 'taper' not in line, 'only what was chosen is written'
    room = {'visualdynamics': visualdynamics,
            'project': visualdynamics.Project()}
    room['project'].add('Target', _target())
    exec(line, room)                                     # noqa: S102
    assert np.array_equal(room['project'][made[0]].ordinate,
                          window.objects[made[0]].ordinate)


def test_a_refusal_is_said_in_the_pane_not_raised(window, pump):
    window.add_object('Target', _target())
    pump()
    panel = _open(window, pump, 'Target')
    panel._editors['rate'].setValue(3000.0)
    pump()
    assert 'Nyquist' in panel.note.text()
    assert 'Nyquist' in window.statusBar().currentMessage()


def test_the_pane_stands_down_when_the_selection_moves(window, pump):
    window.add_object('Target', _target())
    window.add_object('Sweep', _sweep())
    pump()
    panel = _open(window, pump, 'Target')
    assert panel.isVisible()
    select_objects(window, pump, 'Sweep')
    assert not panel.isVisible() and window._signaling is None
    panel = _open(window, pump, 'Sweep')
    assert [panel.kind_box.itemData(i)
            for i in range(panel.kind_box.count())] == ['sine']
    assert not panel.kind_box.isVisibleTo(panel), 'one kind: no choice shown'
    window.generate_signal_act()
    pump()
    assert not panel.isVisible(), 'the act again puts it away'


def test_a_channel_table_offers_an_impulse_and_previews_its_rolloff(window,
                                                                     pump):
    """The impulse's width sets its band; the pane reads where it rolls
    off and the plot draws its trace and its spectrum."""
    window.add_object('Channels', _table())
    pump()
    panel = _open(window, pump, 'Channels')
    kinds = [panel.kind_box.itemData(i) for i in range(panel.kind_box.count())]
    assert kinds == ['random', 'burst', 'periodic', 'impulse', 'pulse']
    panel.kind_box.setCurrentIndex(panel.kind_box.findData('impulse'))
    pump()
    assert 'at 1X+, 2Y+' in window.statusBar().currentMessage()
    for shown in ('peak', 'width', 'at', 'duration'):
        assert panel._editors[shown].isVisibleTo(panel), shown
    assert not panel._editors['shape'].isVisibleTo(panel)
    assert 'area' not in panel._editors
    assert panel._derived['down3'][1].text() == '593.5 Hz'
    assert panel._derived['down10'][1].text() == '1019 Hz'
    assert panel._derived['zero'][1].text() == '1500 Hz'
    keys = [getattr(window.data_pane.graphics.getItem(row, 0), 'series_key',
                    None) for row in range(6)]
    drawn = [key[1] for key in keys if key]
    assert drawn == ['force', 'force*time'], 'the trace, then its spectrum'
    panel._editors['width'].setValue(0.25)
    pump()
    assert panel.note.text() == ('an impulse 0.25 ms wide is under two '
                                 'samples at 5120 Hz; the narrowest is '
                                 '0.3906 ms')
    panel._editors['width'].setValue(0.4)
    pump()
    assert panel._derived['down3'][1].text() == '1484 Hz'
    assert panel._derived['down10'][1].text() == '2548 Hz'
    assert panel._derived['zero'][1].text() == 'above Nyquist', \
        '3750 Hz past a 2560 Hz Nyquist'
    panel.kind_box.setCurrentIndex(panel.kind_box.findData('random'))
    pump()
    assert panel._editors['level'].isVisibleTo(panel), 'a typed target'


def test_a_generated_burst_opens_in_its_own_frames(window, pump, tmp_path):
    """The analysis settings shown on a generated burst are its bursts:
    50 rectangular frames, not one Hann frame of the whole record, and a
    save keeps them."""
    window.add_object('Target', _target())
    pump()
    added = window.project.generate_signal('Target', 'burst', bursts=50,
                                           frame_length=4096,
                                           sample_rate=8192.0, seed=3)
    window.show_object(added)
    pump()
    window.data_pane.averaging_action.trigger()
    pump()
    panel = window.data_pane.averaging_panel
    assert panel.isVisible()
    assert panel.frames_box.value() == 50
    assert panel.length_box.value() == 4096
    assert panel.window_box.currentData() == 'boxcar'
    assert panel.overlap_box.value() == 0.0
    path = tmp_path / 'signals.vdyn'
    window.project.save(path)
    reopened = visualdynamics.Project.open(path)
    a = reopened[added].averaging
    assert (a.frame_length, a.window, a.frames) == (4096, 'boxcar', 50)


def test_the_pane_shows_the_level_in_the_channels_unit(window, pump):
    """Not the display units: the level typed is the channels' own, as
    the verb reads it, and the box says which."""
    window.add_object('Channels', _lbf_table())
    pump()
    panel = _open(window, pump, 'Channels')
    panel.kind_box.setCurrentIndex(panel.kind_box.findData('random'))
    pump()
    assert panel._editors['level'].suffix() == ' lbf²/Hz'
    panel._editors['level'].setValue(3.0)
    assert panel.settings().level == pytest.approx(3.0), 'read back as typed'
    panel.kind_box.setCurrentIndex(panel.kind_box.findData('impulse'))
    pump()
    assert panel._editors['peak'].suffix() == ' lbf'
