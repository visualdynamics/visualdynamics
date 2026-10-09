"""Every documented workflow journals as a script that replays it.

Brandon's rule (2026-08-30): nothing in the console may be
not-replayable, by design. Each driver below walks one project type's
workflow through the window's own handlers — the acts the guide
teaches — then holds the journal to the rule and *runs* it: exec the
session script, compare the projects. The stressdata-fed drivers skip
where the local run data is absent, the pattern `test_extract_sine`
set.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from conftest import fixture_path
from conftest import select_objects as _select

STRESS = os.path.join(os.path.dirname(__file__), '..', 'stressdata',
                      'plate')


def _replay(window):
    """The rule and the proof: no comments, and the script rebuilds
    the project."""
    journal = window.project.journal
    bad = [line for line in journal if line.lstrip().startswith('#')]
    assert not bad, f'not replayable: {bad}'
    room: dict = {}
    exec(window.project.session_script(), room)         # noqa: S102
    replayed = room['project']
    assert set(replayed.names) == set(window.project.names), (
        set(replayed.names) ^ set(window.project.names))
    for name in window.project.names:
        ours, theirs = window.project[name], replayed[name]
        assert type(ours) is type(theirs), name
        ordinate = getattr(ours, 'ordinate', None)
        if ordinate is not None:
            assert np.allclose(np.asarray(ordinate),
                               np.asarray(theirs.ordinate),
                               equal_nan=True), name
        matrix = getattr(ours, 'shape_matrix', None)
        if matrix is not None:
            assert np.allclose(matrix, theirs.shape_matrix), name
            assert np.allclose(ours.frequency, theirs.frequency), name
    return replayed


def test_the_modal_workflow_replays(window, pump, tmp_path, monkeypatch):
    """Import, declare, fit interactively, refine, report, save — the
    modal guide's own arc, journaled and replayed."""
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4'),
                         fixture_path('plate', 'test_geometry.npz')])
    pump()
    # declare the geometry's units through the pane
    _select(window, pump, 'Geometry')
    window.define_units()
    pump()
    model = window.units_table.model()
    from PySide6.QtCore import Qt

    model.setData(model.index(0, model.columnCount() - 1), 'm',
                  Qt.ItemDataRole.EditRole)
    window._restate_after_units()
    pump()
    # PSDs over the run's own captures
    _select(window, pump, 'Time History')
    window.compute_psds()
    pump()
    # the interactive fit: two confirms and a refine... the screen's
    # own buttons, which is what the journal must replay
    _select(window, pump, 'FRF')
    window.start_modal_fit()
    pump()
    window.confirm_mode_button.click()
    pump()
    window.find_mode_button.click()
    pump()
    window.confirm_mode_button.click()
    pump()
    window.refine_all_modes()
    pump()
    window.stop_fitting()
    pump()
    # report and save, dialogs answered
    window.generate_report('modal')
    pump()
    from PySide6.QtWidgets import QFileDialog

    out = str(tmp_path / 'modal.vdyn')
    monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: (out, '')))
    window.save_test()
    pump()
    replayed = _replay(window)
    assert 'FRF Modes' in replayed.names, 'the interactive fit replayed'


def test_the_random_workflow_replays(window, pump):
    window.import_paths([fixture_path('plate', 'random_spectra.nc4')])
    pump()
    psd = next(n for n, o in window.objects.items()
               if type(o).__name__ == 'Psd')
    _select(window, pump, psd)
    window.compute_octave()
    pump()
    window.generate_report('random')
    pump()
    _replay(window)


def test_the_shock_workflow_replays(window, pump):
    window.import_paths([fixture_path('plate', 'shock.nc4')])
    pump()
    name = next(n for n, o in window.objects.items()
                if type(o).__name__ == 'TimeHistory')
    _select(window, pump, name)
    window._detect_shocks()
    pump()
    if window.objects[name].shocks:
        window.compute_srs()
        pump()
    window.generate_report('shock')
    pump()
    _replay(window)


def test_the_transient_workflow_replays(window, pump, tmp_path):
    from test_replication import _transient_file

    from visualdynamics.core.averaging import Averaging

    path = str(tmp_path / 'transient.nc4')
    _transient_file(path, repeats=4, frame=256)
    window.import_paths([path])
    pump()
    name = next(n for n, o in window.objects.items()
                if type(o).__name__ == 'TimeHistory')
    _select(window, pump, name)
    window._averaging_edited(Averaging(frame_length=256, frames=1))
    pump()
    window.compute_psds()
    pump()
    window.generate_report('transient')
    pump()
    _replay(window)


def test_the_sysid_workflow_replays(window, pump, tmp_path, monkeypatch):
    from test_sysid_package import _write_streamed_sysid

    path = str(_write_streamed_sysid(tmp_path / 'sysid_stream.nc4'))
    # declared first, the way the guide opens: the import then names
    # the streams without asking (the question box would wedge a
    # headless run, and a person who declared the type already
    # answered it)
    window.set_project_type('System ID')
    window.import_paths([path])
    pump()
    name = next(n for n, o in window.objects.items()
                if type(o).__name__ == 'TimeHistory')
    _select(window, pump, name)
    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, 'getItem',
                        staticmethod(lambda *a, **k: (a[3][0], True)))
    window.compute_frfs()
    pump()
    window.generate_report('sysid')
    pump()
    _replay(window)


@pytest.mark.skipif(not os.path.exists(os.path.join(STRESS, 'sine.nc4')),
                    reason='local sine run absent (generators repo)')
def test_the_sine_workflow_replays(window, pump):
    window.import_paths([os.path.join(STRESS, 'sine.nc4')])
    pump()
    name = next(n for n, o in window.objects.items()
                if type(o).__name__ == 'TimeHistory')
    _select(window, pump, name)
    window.extract_sine_levels()
    pump()
    window.generate_report('sine')
    pump()
    _replay(window)


def test_a_fit_of_picked_records_weighted_by_coherence_replays(window,
                                                               pump):
    """The fitting screen fits the records picked in the tree and weights
    them by the project's coherence; the journal said neither until
    2026-10-09, so the replay fitted every record unweighted — another
    fit under the same name. A coherence that distrusts one response
    badly is what makes the weighting visible in the numbers."""
    window.import_paths([fixture_path('plate', 'modal_spectra.nc4')])
    pump()
    coherence = window.objects['Multiple Coherence']
    distrusted = coherence.response_dof.index('107Z+')
    coherence.ordinate[distrusted] = 0.2
    item = window._item_for_object('FRF')
    window.tree.clearSelection()
    item.setExpanded(True)
    pump()
    picked = list(range(16))
    window.record_grids['FRF'].select_records(picked)
    item.setSelected(True)
    pump()
    window.start_modal_fit()
    pump()
    assert window._fit_coherence_name == 'Multiple Coherence'
    # enough modes, refined twice, for the weighting to move the shapes
    for _ in range(8):
        window.confirm_mode_button.click()
        pump()
    for _ in range(2):
        window.refine_all_modes()
        pump()
    window.stop_fitting()
    pump()
    line = next(entry for entry in window.project.journal
                if entry.startswith("project.fit_modes('FRF'"))
    assert f'records={picked!r}' in line
    assert "coherence='Multiple Coherence'" in line
    # the line alone, on a fresh project holding the same FRF and the
    # same distrustful coherence (that edit was a raw write the journal
    # cannot see), must be this fit
    import visualdynamics

    def replayed(text):
        fresh = visualdynamics.Project('Replay')
        fresh.import_file(fixture_path('plate', 'modal_spectra.nc4'))
        fresh['Multiple Coherence'].ordinate[distrusted] = 0.2
        exec(text, {'project': fresh})                   # noqa: S102
        return fresh['FRF Modes']

    ours, theirs = window.objects['FRF Modes'], replayed(line)
    assert np.allclose(ours.frequency, theirs.frequency)
    assert np.allclose(ours.shape_matrix, theirs.shape_matrix)
    # and the test can tell: the same line without either is another fit
    for dropped in (f', records={picked!r}',
                    ", coherence='Multiple Coherence'"):
        other = replayed(line.replace(dropped, ''))
        assert (other.shape_matrix.shape != ours.shape_matrix.shape
                or not np.allclose(other.shape_matrix, ours.shape_matrix)), \
            f'dropping {dropped!r} changed nothing: the test cannot bite'
