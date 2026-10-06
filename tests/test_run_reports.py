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
    from visualdynamics.gui.run_reports import DEFAULT_MARKING, RunReportsDialog

    runs = [*_campaign(tmp_path), fixture_path('plate', 'modal.nc4')]
    dialog = RunReportsDialog(runs)
    names = [dialog.table.item(row, 0).text() for row in range(3)]
    assert names[:2] == ['sweep.nc4', 'random.nc4']
    assert 'no one-call report' in names[2], 'said, not hidden'
    assert [box.currentText() for box in dialog.reports] == [
        'Sine Sweep', 'Random Vibration', 'Leave out']
    assert dialog.write_button.text() == 'Write 2 Reports'
    chosen = dialog.choices()
    assert chosen['runs'] == runs[:2], 'the modal survey is left out'
    assert chosen['kinds'] == {runs[0]: 'sine', runs[1]: 'random'}
    assert chosen['last'] == {runs[0]: None, runs[1]: None}, 'whole runs'
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

    # a random-and-sine run the dialog was told to give a random report
    # alone: one file where its type would write two, so the window is
    # seen to pass the dialog's per-run choice on (2026-10-06)
    both = _write_run(tmp_path / 'both.nc4', random=True)
    runs = [*_campaign(tmp_path), both]
    out = tmp_path / 'reports'
    out.mkdir()
    monkeypatch.setattr(QFileDialog, 'getOpenFileNames',
                        staticmethod(lambda *a, **k: (runs, '')))
    monkeypatch.setattr(run_reports, 'ask_run_reports', lambda parent, paths: {
        'runs': list(paths), 'path': str(out) + os.sep,
        'geometry': None, 'marking': 'BATCH',
        'kinds': {both: 'random'}, 'last': {}})
    assert any(action.text() == 'Reports from R&uns…'
               for action in window.menuBar().actions()[0].menu().actions())
    before = set(window.project)
    window.reports_from_runs()
    assert sorted(p.name for p in out.iterdir()) == \
        ['both.html', 'random.html', 'sweep.html']
    assert 'BATCH' in (out / 'random.html').read_text(encoding='utf-8')
    assert window.statusBar().currentMessage().startswith('Wrote 3 reports to')
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


def test_each_run_can_be_given_its_report_and_its_last(qt_app, tmp_path):
    """The detected report is a default: a run is given another or left
    out, and the button counts what will be written (2026-10-06). The
    one box for every random report sets the random rows' Last and
    leaves the sweep whole; a row's own Last is still its own."""
    from visualdynamics.gui.run_reports import RunReportsDialog

    both = _write_run(tmp_path / 'both.nc4', random=True)
    runs = [*_campaign(tmp_path), both]
    dialog = RunReportsDialog(runs)
    assert [box.currentData() for box in dialog.reports] == \
        ['sine', 'random', 'mixed']
    assert dialog.write_button.text() == 'Write 4 Reports', \
        'a random-and-sine run is two reports'
    dialog.every_random.setValue(120.0)
    assert [span.value() for span in dialog.lasts] == [0.0, 120.0, 120.0]
    dialog.lasts[1].setValue(60.0)
    _sine, random, mixed = dialog.reports
    random.setCurrentIndex(random.findText('Leave out'))
    mixed.setCurrentIndex(mixed.findData('random'))
    assert dialog.write_button.text() == 'Write 2 Reports'
    chosen = dialog.choices()
    assert chosen['runs'] == [runs[0], both]
    assert chosen['kinds'] == {runs[0]: 'sine', both: 'random'}
    assert chosen['last'] == {runs[0]: None, both: 120.0}


def test_run_report_takes_each_runs_report_and_last(tmp_path):
    """`kinds` overrides the file's own type run by run, `last` may be
    given run by run, and a kind that is no report is refused before
    anything is written."""
    import re

    both = _write_run(tmp_path / 'both.nc4', random=True, seconds=18.0)
    out = tmp_path / 'reports'
    out.mkdir()
    with pytest.raises(ValueError, match='no report is called'):
        visualdynamics.run_report([both], str(out) + '/',
                                  kinds={both: 'combined'})
    assert list(out.iterdir()) == []
    written = visualdynamics.run_report(
        [both], str(out) + '/', kinds={both: 'random'}, last={both: 5.0})
    assert [os.path.basename(p) for p in written] == ['both.html'], \
        'one random report, not the two its type would get'
    with open(written[0], encoding='utf-8') as handle:
        html = handle.read()
    assert 'Random Vibration Test Report' in html
    assert float(re.search(r'over ([0-9.]+) ?s', html).group(1)) == \
        pytest.approx(5.0, abs=0.01)


def test_a_channel_with_no_reading_is_a_gap_not_a_failed_report(tmp_path):
    """A one-second random-and-sine run: some channel's RMS error has no
    number. It went into the page as NaN, which the page's JSON refuses,
    and the whole report failed to write (2026-10-06, found by the File
    menu test above). It is a null now, a labeled row with no bar."""
    import json

    from visualdynamics.report import render_html

    both = _write_run(tmp_path / 'both.nc4', random=True)
    project = visualdynamics.random_vibration_run(both)
    name = project.generate_report('random')
    html = render_html(project[name], project, None, links=project.links)
    payload = json.loads(html.split('type="application/json">')[1]
                         .split('</script>')[0])
    errors = [b for b in payload['blocks'] if b.get('kind') == 'bars'
              and b.get('ylabel', '').startswith('RMS error')]
    assert errors, 'the RMS error chart is on the page'
    assert any(value is None for chart in errors for value in chart['values'])
