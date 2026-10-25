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
                                   sample_rate=1000.0, at=0.1, area=5.0)
    made = project[name]
    assert made.response_dof == ['1X+', '2Y+'], 'the reference channels only'
    assert made.ordinate_dim == ['force', 'force']
    assert made.ordinate[0].sum() / 1000.0 == pytest.approx(5.0)
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


def test_a_channel_table_offers_an_impulse_and_previews_its_trace(window,
                                                                   pump):
    window.add_object('Channels', _table())
    pump()
    panel = _open(window, pump, 'Channels')
    kinds = [panel.kind_box.itemData(i) for i in range(panel.kind_box.count())]
    assert kinds == ['random', 'burst', 'periodic', 'impulse', 'pulse']
    panel.kind_box.setCurrentIndex(panel.kind_box.findData('impulse'))
    pump()
    assert 'at 1X+, 2Y+' in window.statusBar().currentMessage()
    assert panel._editors['area'].isVisibleTo(panel)
    panel.kind_box.setCurrentIndex(panel.kind_box.findData('random'))
    pump()
    assert panel._editors['level'].isVisibleTo(panel), 'a typed target'
