"""The tree bar's Automatic (Brandon, 2026-09-30): one press computes
every gray slot the project type expects that the loaded data can
fill, in the workflow's order, and generates the report. The rule is
the project's (`Project.work_up`); the window shows what it made.
"""
from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.data import (
    Frf,
    MultipleCoherence,
    Psd,
    Specification,
    TimeHistory,
)
from visualdynamics.core.report import Report, is_banded
from visualdynamics.core.shapes import ShapeSet

RANDOM = ('plate', 'random.nc4')


def _placeholders(window):
    from visualdynamics.gui.main_window import ROLE_REFERENCE

    out = []
    for i in range(window.test_item.childCount()):
        child = window.test_item.child(i)
        reference = child.data(0, ROLE_REFERENCE)
        if reference is not None and reference[0] == 'placeholder':
            out.append(child.text(0))
    return out


def _kinds(project):
    return {type(obj).__name__ for obj in project.objects}


def _response_only(seconds=8.0, rate=1024.0, seed=1):
    """A recording with no drive channel: PSDs can be averaged from
    it, a coherence cannot."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    return TimeHistory(t, rng.standard_normal((2, len(t))),
                       response_dof=['1Z+', '2Z+'],
                       ordinate_dim='acceleration', ordinate_unit='g')


# ---- the button -----------------------------------------------------------


def test_the_button_shows_only_on_a_typed_project(window, pump):
    """An untyped project has no skeleton, so there is nothing for the
    button to fill and it is not shown."""
    assert not window.automatic_action.isVisible()
    window.set_project_type('Random Vibration')
    assert window.automatic_action.isVisible()
    window.set_project_type(None)
    assert not window.automatic_action.isVisible()


def test_a_run_that_types_the_project_brings_the_button(window, pump):
    window.import_paths([fixture_path(*RANDOM)])
    pump()
    assert window.project_type == 'Random Vibration'
    assert window.automatic_action.isVisible()


def test_the_glyph_is_its_own(window):
    from visualdynamics.gui.icons import control_icon

    ours = control_icon('automatic').pixmap(32).toImage()
    fallback = control_icon('no-such-glyph').pixmap(32).toImage()
    assert ours != fallback


# ---- a random run, the case that asked for it ----------------------------


def test_a_random_run_is_worked_up_to_its_report(window, pump):
    """The run brings the time data, the channel table and the
    specification; one press makes the PSDs, the octave-band PSDs and
    specification, the multiple coherence and the report, and opens
    the report. What the data cannot make — the geometry, the
    photographs — stays gray."""
    window.import_paths([fixture_path(*RANDOM)])
    pump()
    assert _placeholders(window).count('PSD') == 2
    window.automatic_action.trigger()
    pump()
    assert _placeholders(window) == ['Geometry', 'Photos']
    objects = list(window.objects.values())
    assert {'Psd', 'MultipleCoherence', 'Report'} <= _kinds(window.project)
    assert any(is_banded(obj) and not isinstance(obj, Specification)
               for obj in objects if isinstance(obj, Psd)), 'octave PSD'
    assert any(is_banded(obj) for obj in objects
               if isinstance(obj, Specification)), 'octave specification'
    assert isinstance(window.current_object(), Report), 'the report opens'
    status = window.statusBar().currentMessage()
    assert status.startswith('Automatic: added ')
    assert 'still missing Geometry, Photos' in status
    assert window.project.journal[-1] == 'project.work_up()'
    # everything made joins the run's group, where the report reads it
    basis = set(window.project.basis.names)
    assert {name for name, obj in window.objects.items()
            if isinstance(obj, (Psd, MultipleCoherence))} <= basis


def test_pressed_twice_the_second_press_adds_nothing(window, pump):
    window.import_paths([fixture_path(*RANDOM)])
    pump()
    window.automatic_action.trigger()
    pump()
    names = list(window.objects)
    window.automatic_action.trigger()
    pump()
    assert list(window.objects) == names
    status = window.statusBar().currentMessage()
    assert status.startswith('Automatic: nothing to compute')
    assert 'Geometry, Photos must be imported' in status


def test_the_one_journal_line_replays(window, pump):
    """`project.work_up()` is the line, and running it rebuilds the
    project — the journal rule every workflow keeps."""
    from test_workflow_journals import _replay

    window.import_paths([fixture_path(*RANDOM)])
    pump()
    window.automatic_action.trigger()
    pump()
    assert window.project.journal.count('project.work_up()') == 1
    _replay(window)


def test_the_verb_makes_what_the_one_call_makes():
    """`work_up` on an imported run and `random_vibration_run` on its
    file are the same workflow, and say so with the same objects."""
    project = visualdynamics.Project()
    project.import_file(fixture_path(*RANDOM))
    added = project.work_up()
    whole = visualdynamics.random_vibration_run(fixture_path(*RANDOM))
    assert set(project.names) == set(whole.names) | {'Report'}
    assert added[-1] == 'Report'
    for name in whole.names:
        ordinate = getattr(whole[name], 'ordinate', None)
        if ordinate is not None:
            assert np.allclose(np.asarray(ordinate),
                               np.asarray(project[name].ordinate),
                               equal_nan=True), name


# ---- the rule, type by type -------------------------------------------


def test_an_untyped_project_has_nothing_to_work_up():
    project = visualdynamics.Project()
    project.add('Time History', _response_only())
    with pytest.raises(ValueError, match='no type'):
        project.work_up()
    assert project.missing() == []


def test_what_the_data_refuses_stays_gray():
    """No drive channel, no coherence — and the PSDs, the bands and the
    report are still made around the refusal."""
    project = visualdynamics.Project()
    project.add('Time History', _response_only())
    project.project_type = 'Random Vibration'
    added = project.work_up()
    assert 'Report' in added
    assert 'MultipleCoherence' not in _kinds(project)
    assert sum(isinstance(obj, Psd) for obj in project.objects) == 2
    labels = [slot[0] for slot in project.missing()]
    assert 'Multiple Coherence' in labels and 'Report' not in labels


def test_a_modal_run_gets_its_frfs_and_a_fitted_shape_set():
    project = visualdynamics.Project()
    project.import_file(fixture_path('plate', 'modal.nc4'))
    assert project.project_type == 'Modal Test'
    project.work_up()
    assert {'Psd', 'Frf', 'MultipleCoherence', 'ShapeSet', 'Report'} \
        <= _kinds(project)
    frf = next(name for name, obj in project.items() if isinstance(obj, Frf))
    shapes = next(name for name, obj in project.items()
                  if isinstance(obj, ShapeSet))
    assert shapes in project.group_of(frf), 'fitted from the FRFs made'
    # the other side and the matched pairs are best practice the data
    # cannot supply; only they are left
    assert [slot[0] for slot in project.missing()] == \
        ['Geometry', 'Photos', 'Geometry', 'Matched Modes']


def test_a_shock_run_is_filtered_before_its_srs_and_motion_chain():
    """The recommended order the skeleton teaches: the SRS and the
    velocity and displacement come from the *filtered* record, so the
    filtering is done first although its slot is listed after."""
    project = visualdynamics.Project()
    project.import_file(fixture_path('plate', 'shock.nc4'))
    project.project_type = 'Shock'
    project.work_up()
    made = {name: (info['verb'], info['source'])
            for name, info in project.provenance.items()}
    filtered = next(n for n, (v, _s) in made.items() if v == 'filter_data')
    srs = next(n for n, (v, _s) in made.items() if v == 'compute_srs')
    assert made[srs][1] == filtered
    velocity = next(n for n, (v, s) in made.items()
                    if v == 'integrate' and s == filtered)
    assert any(v == 'integrate' and s == velocity
               for v, s in made.values()), 'displacement from the velocity'
    assert [slot[0] for slot in project.missing()] == \
        ['Geometry', 'Photos', 'Shock Specification']


def test_a_transient_run_reads_its_target_and_its_record(tmp_path):
    from test_replication import _transient_file

    path = str(tmp_path / 'transient.nc4')
    _transient_file(path, repeats=4, frame=256)
    project = visualdynamics.Project()
    project.import_file(path)
    assert project.project_type == 'Transient'
    project.work_up()
    kinds = {name: type(obj).__name__ for name, obj in project.items()}
    assert kinds['Transient Specification PSDs'] == 'Specification'
    assert kinds['Time History PSDs'] == 'Psd'
    assert [slot[0] for slot in project.missing()] == ['Geometry', 'Photos']


def test_a_sine_sweep_extracts_its_levels():
    from test_extract_sine import _recording, _spec

    spec = _spec()
    project = visualdynamics.Project()
    project.add('Time History', _recording(spec))
    project.add('Sine Specification', spec)
    project.project_type = 'Sine Sweep'
    added = project.work_up()
    assert added == ['Sine Levels', 'Report']
    assert [slot[0] for slot in project.missing()] == \
        ['Geometry', 'Photos', 'Channel Table']


def test_a_system_id_averages_both_streams_on_shared_frames(tmp_path):
    """The ambient borrows the excitation's frames, as the one-call
    workup does: the signal-to-noise the report reads is a ratio of
    densities, defined only on lines both hold. The plant and the
    coherence need a reference channel the streamed save does not
    mark, so they stay gray until one is declared."""
    from test_sysid_package import _write_streamed_sysid

    project = visualdynamics.Project()
    project.project_type = 'System ID'
    project.import_file(str(_write_streamed_sysid(tmp_path / 'sysid.nc4')))
    project.work_up()
    histories = [name for name, obj in project.items()
                 if isinstance(obj, TimeHistory)]
    assert len(histories) == 2
    assert project[histories[0]].averaging == project[histories[1]].averaging
    assert sum(isinstance(obj, Psd) for obj in project.objects) == 2
    assert 'Report' in project.names
    assert [slot[0] for slot in project.missing() if not slot[4]] == \
        ['FRF', 'Multiple Coherence']


def test_a_random_and_sine_run_works_up_both_halves():
    """Automatic on a Random and Sine project fills the random half's
    slots and the sine half's — the PSDs, the octave bands of both the
    data and the requirement, the sine levels — and makes the type's
    two reports last, the random's then the sine's (Brandon,
    2026-10-02: "make sure Compute everything works for Sine and
    Random and Sine"; two reports since 2026-10-05). Pressed again it
    adds nothing."""
    import numpy as np
    from test_extract_sine import _recording, _spec

    from visualdynamics.core.data import Specification

    spec = _spec()
    history = _recording(spec, noise=0.5)
    frequencies = np.logspace(1, 3, 25)
    level = np.full((2, len(frequencies)), 1e-3)
    random = Specification(
        abscissa=frequencies, ordinate=level,
        response_dof=list(history.response_dof),
        ordinate_dim=['acceleration**2/frequency'] * 2,
        ordinate_unit=['(m/s**2)**2/Hz'] * 2,
        abort_upper=level * 2.0, abort_lower=level * 0.5)
    project = visualdynamics.Project()
    project.add('Time History', history)
    project.add('PSD Specification', random)
    project.add('Sine Specification', spec)
    project.project_type = 'Random and Sine'
    added = project.work_up()
    kinds = [type(project[name]).__name__ for name in added]
    assert 'Psd' in kinds and 'SineLevelSet' in kinds
    reports = [project[name] for name in added[-2:]]
    assert [r.title for r in reports] == ['Random Vibration Test Report',
                                          'Sine Sweep Test Report']
    random_captions = ' '.join(b.get('caption', '') for b in reports[0].blocks)
    sine_captions = ' '.join(b.get('caption', '') for b in reports[1].blocks)
    assert 'Control against specification' in random_captions
    assert 'extracted level against the requirement' in sine_captions
    assert project.work_up() == [], 'pressed twice, nothing new'
    assert not [slot[0] for slot in project.missing()
                if not slot[4] and slot[0] not in
                ('Geometry', 'Photos', 'Channel Table', 'Multiple Coherence')]


def test_the_file_menu_offers_every_report(window):
    """The File menu's Generate Report lists the same templates the
    bar's menu does — the sine report was missing there (Brandon,
    2026-10-02). No random-and-sine template since 2026-10-05: that
    project makes a random report and a sine report."""
    file_menu = next(a.menu() for a in window.menuBar().actions()
                     if a.text().replace('&', '') == 'File')
    report_menu = next(a.menu() for a in file_menu.actions()
                       if a.menu() is not None
                       and a.text().replace('&', '') == 'Generate Report')
    labels = [a.text().replace('&', '') for a in report_menu.actions()]
    assert labels == ['Modal Test', 'Random Vibration', 'Transient', 'Shock',
                      'Sine Sweep', 'System ID', 'Empty']
