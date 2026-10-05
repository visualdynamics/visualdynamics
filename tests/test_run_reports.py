"""Reports from Runs (2026-10-04): `run_report` over a batch of
controller runs, from a script as a list and from the window's File
menu through one dialog, on the long-verb runner with the bar and
Cancel."""

from __future__ import annotations

import os

import pytest
from conftest import fixture_path
from test_rattlesnake_sine import _write_run

import visualdynamics


def _campaign(tmp_path):
    """A sweep and a random run: two kinds, one batch."""
    return [_write_run(tmp_path / 'sweep.nc4', seconds=18.0),
            fixture_path('plate', 'random.nc4')]


def test_a_list_of_runs_is_a_batch_with_progress(tmp_path):
    out = tmp_path / 'reports'
    out.mkdir()
    ticks = []
    written = visualdynamics.run_report(
        _campaign(tmp_path), str(out) + '/', marking='BATCH',
        progress=lambda done, total: ticks.append((done, total)))
    assert sorted(os.path.basename(p) for p in written) == \
        ['random.html', 'sweep.html']
    assert ticks == [(0, 2), (1, 2), (2, 2)], 'told before, and after each run'


def test_a_list_of_one_still_answers_a_list(tmp_path):
    written = visualdynamics.run_report(
        [fixture_path('plate', 'random.nc4')], str(tmp_path) + '/')
    assert isinstance(written, list) and len(written) == 1


def test_a_batch_with_an_unreportable_run_writes_nothing(tmp_path):
    """Every kind is read before the first report: a modal survey
    third in line refuses the batch, not after two reports."""
    out = tmp_path / 'reports'
    out.mkdir()
    runs = [*_campaign(tmp_path), fixture_path('plate', 'modal.nc4')]
    with pytest.raises(ValueError, match='no one-call report'):
        visualdynamics.run_report(runs, str(out) + '/')
    assert list(out.iterdir()) == []


def test_the_dialog_lists_each_runs_report_and_leaves_out_the_rest(
        qt_app, tmp_path):
    from PySide6.QtCore import Qt

    from visualdynamics.gui.run_reports import DEFAULT_MARKING, RunReportsDialog

    runs = [*_campaign(tmp_path), fixture_path('plate', 'modal.nc4')]
    dialog = RunReportsDialog(runs)
    texts = [dialog.list.item(i).text() for i in range(dialog.list.count())]
    assert texts[0] == 'sweep.nc4 — Sine Sweep'
    assert texts[1] == 'random.nc4 — Random Vibration'
    assert 'no one-call report' in texts[2]
    assert not dialog.list.item(2).flags() & Qt.ItemFlag.ItemIsEnabled
    assert dialog.write_button.text() == 'Write 2 Reports'
    chosen = dialog.choices()
    assert chosen['runs'] == runs[:2], 'the modal survey is left out'
    assert chosen['path'] is None, 'beside each run, as run_report does'
    assert chosen['geometry'] is None
    assert chosen['marking'] == DEFAULT_MARKING
    # one folder: a trailing separator, which run_report reads as a folder
    dialog.folder = str(tmp_path)
    dialog.in_folder.setChecked(True)
    assert dialog.choices()['path'] == os.path.join(str(tmp_path), '')


def test_the_file_menu_writes_the_reports(window, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    from visualdynamics.gui import run_reports

    runs = _campaign(tmp_path)
    out = tmp_path / 'reports'
    out.mkdir()
    monkeypatch.setattr(QFileDialog, 'getOpenFileNames',
                        staticmethod(lambda *a, **k: (runs, '')))
    monkeypatch.setattr(run_reports, 'ask_run_reports', lambda parent, paths: {
        'runs': list(paths), 'path': str(out) + os.sep,
        'geometry': None, 'marking': 'BATCH'})
    assert any(action.text() == 'Reports from R&uns…'
               for action in window.menuBar().actions()[0].menu().actions())
    before = set(window.project)
    window.reports_from_runs()
    assert sorted(p.name for p in out.iterdir()) == \
        ['random.html', 'sweep.html']
    assert 'BATCH' in (out / 'random.html').read_text(encoding='utf-8')
    assert window.statusBar().currentMessage().startswith('Wrote 2 reports to')
    assert set(window.project) == before, 'the open project is untouched'


def test_a_cancel_stops_the_batch_and_says_so(window, monkeypatch, tmp_path):
    """Cancel lands between runs: the runner's progress hook raises at
    the next tick, and the window says the written ones are kept."""
    from PySide6.QtWidgets import QFileDialog

    import visualdynamics.project as project_module
    from visualdynamics.gui import run_reports

    seen = []

    def two_runs(runs, path, progress=None, **_kw):
        progress(0, 2)
        seen.append('first')
        window._cancel_asked = True
        progress(1, 2)
        seen.append('second')

    monkeypatch.setattr(project_module, 'run_report', two_runs)
    monkeypatch.setattr(QFileDialog, 'getOpenFileNames',
                        staticmethod(lambda *a, **k: (['a.nc4', 'b.nc4'], '')))
    monkeypatch.setattr(run_reports, 'ask_run_reports', lambda parent, paths: {
        'runs': list(paths), 'path': None, 'geometry': None,
        'marking': 'BATCH'})
    window.reports_from_runs()
    assert seen == ['first'], 'the second run never started'
    assert 'cancelled' in window.statusBar().currentMessage()
