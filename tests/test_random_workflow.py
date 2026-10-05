"""A controller run in, a report out — and the headless path it takes.

`random_vibration_report` is the whole random vibration workflow in one
call, and `random_vibration_run` is the project it does that from. What
matters about them is not that they run but that they take the same
steps the window takes and reach the same numbers, so a report written
from a script and one written from the app are the same report.

The plots are here too. Everything the app draws for a random vibration
run has a call that renders it to a file with no window: the comparison
against the specification, both bar charts, the coherence map. A gap
there is a workflow that has to be finished by hand in the GUI.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.data import MultipleCoherence, Psd, Specification, TimeHistory
from visualdynamics.plot import plot_bars, plot_comparison, plot_series

RUN = 'random.nc4'


@pytest.fixture(scope='module')
def worked_up():
    """The run, worked up once. Every test here reads it and none
    changes it — the FFTs are the expensive part and doing them per
    test buys nothing."""
    return visualdynamics.random_vibration_run(fixture_path('plate', RUN))


def _measured(project):
    """(narrowband PSDs, octave-band PSDs) — the specification is a Psd
    too, and the banded one carries its own bin widths."""
    psds = [obj for obj in project.psds
            if not isinstance(obj, Specification)]
    narrow = [obj for obj in psds if obj.bandwidth is None]
    banded = [obj for obj in psds if obj.bandwidth is not None]
    return narrow[0], banded[0]


# ---- what the workflow builds -------------------------------------------


def test_the_run_arrives_worked_up(worked_up):
    """Import, PSDs, octave bands, coherence — the four steps, and the
    project typed by what the file says it is."""
    assert worked_up.project_type == 'Random Vibration'
    assert isinstance(worked_up.time_history, TimeHistory)
    assert isinstance(worked_up.specifications[0], Specification)
    assert isinstance(worked_up.coherence, MultipleCoherence)
    narrow, banded = _measured(worked_up)
    assert narrow.num_records == worked_up.time_history.num_records
    assert banded.num_records == narrow.num_records
    assert len(banded.abscissa) < len(narrow.abscissa), (
        'bands are wider than lines, so there are fewer of them')


def test_the_frames_are_detected_rather_than_demanded(worked_up):
    """The file does not carry an averaging, so one is worked out — and
    stored, because a coherence averaged differently from the PSD beside
    it would describe a different measurement."""
    averaging = worked_up.time_history.averaging
    assert averaging is not None
    assert averaging.frames > 1, (
        'a single average fits every reference exactly and reads 1.0 at '
        'every line')
    assert (worked_up.coherence.ordinate <= 1.0 + 1e-9).all()
    assert worked_up.coherence.ordinate.min() < 0.99, (
        'a coherence that is 1.0 everywhere was not averaged at all')


def test_banding_keeps_the_power_it_was_given(worked_up):
    """The octave bands are the same spectrum rearranged: the area under
    it, which is the mean square the channel carries, does not move."""
    narrow, banded = _measured(worked_up)
    for record in range(narrow.num_records):
        left, right = narrow.bin_bounds()
        whole = float(np.nansum(narrow.ordinate[record] * (right - left)))
        left, right = banded.bin_bounds()
        bands = float(np.nansum(banded.ordinate[record] * (right - left)))
        assert bands == pytest.approx(whole, rel=2e-3), (
            f'record {record} lost power on the way onto bands')


def test_the_report_binds_the_narrowband_set_not_the_bands(worked_up):
    """Both are PSDs, and the report bands for itself.

    Bound to the octave set the report would band an already-banded
    spectrum for its octave section and compare bands against lines in
    its first — a wrong report that renders perfectly.
    """
    from visualdynamics.core.report import resolve_binding

    narrow, _banded = _measured(worked_up)
    name = resolve_binding('@basis:Psd', worked_up, worked_up.links)
    assert worked_up[name] is narrow


# ---- the one call -------------------------------------------------------


def test_one_call_writes_the_report(tmp_path):
    path = visualdynamics.random_vibration_report(fixture_path('plate', RUN),
                                         tmp_path / 'run.html')
    html = (tmp_path / 'run.html').read_text(encoding='utf-8')
    assert str(path) == str(tmp_path / 'run.html')
    assert 'Random Vibration Test Report' in html
    for caption in ('Control against specification',
                    'RMS error by control channel',
                    'Band outside the abort limits, by control channel',
                    'Control against specification, octave bands',
                    'Multiple coherence',
                    'Instrumentation'):
        assert caption in html, caption
    # the specification is drawn only against a measurement, never
    # alone (Brandon, 2026-09-21)
    assert 'Test specification' not in html


def test_the_report_lands_beside_the_run_by_default(tmp_path):
    import shutil

    run = tmp_path / 'shaker.nc4'
    shutil.copy(fixture_path('plate', RUN), run)
    path = visualdynamics.random_vibration_report(run)
    assert path == str(tmp_path / 'shaker.html')
    assert (tmp_path / 'shaker.html').exists()


def test_a_file_with_no_time_data_says_so(tmp_path):
    with pytest.raises(ValueError, match='no time data'):
        visualdynamics.random_vibration_report(
            fixture_path('plate', 'geometry.npz'),
            tmp_path / 'nothing.html')


def test_the_last_seconds_of_a_run_are_what_is_imported():
    """`last=` is the import dialog's Last field for a script: the
    window opens that many seconds before the end and runs to it, and
    the work-up is the same afterwards (Brandon, 2026-09-19)."""
    whole = visualdynamics.random_vibration_run(fixture_path('plate', RUN))
    part = visualdynamics.random_vibration_run(fixture_path('plate', RUN),
                                               last=10.0)
    full, cut = whole.time_history.abscissa, part.time_history.abscissa
    assert cut[-1] == pytest.approx(full[-1])
    assert cut[0] == pytest.approx(full[-1] - 10.0)
    assert cut.size < full.size
    assert set(part.names) == set(whole.names), 'worked up the same'


def test_a_last_longer_than_the_run_takes_it_whole():
    """Asking for the last 100 s of a 28 s run is asking for all of it
    (Brandon, 2026-09-19), not a mistake to refuse."""
    whole = visualdynamics.random_vibration_run(fixture_path('plate', RUN))
    part = visualdynamics.random_vibration_run(fixture_path('plate', RUN),
                                               last=100.0)
    assert part.time_history.abscissa.size == whole.time_history.abscissa.size
    assert part.time_history.abscissa[0] == pytest.approx(
        whole.time_history.abscissa[0])


def test_last_must_be_a_positive_span():
    with pytest.raises(ValueError, match='positive'):
        visualdynamics.random_vibration_run(fixture_path('plate', RUN), last=0)


def _photo(path, color):
    from PySide6.QtGui import QImage

    image = QImage(64, 48, QImage.Format.Format_RGB32)
    image.fill(color)
    assert image.save(str(path), 'PNG')
    return path


def test_the_geometry_and_the_photos_ride_along(qt_app, tmp_path):
    """The rest of what the tree asks for, given to the one call: the
    geometry with its unit declared, the photographs from a folder —
    all linked into the run's group, which is what the report reads
    them through."""
    folder = tmp_path / 'photos'
    folder.mkdir()
    _photo(folder / 'front.png', 0x336699)
    _photo(folder / 'side.png', 0x996633)
    (folder / 'notes.txt').write_text('not a photo')
    project = visualdynamics.random_vibration_run(
        fixture_path('plate', RUN), geometry=fixture_path('plate', 'geometry.npz'),
        length_unit='m', photos=folder)
    assert project['Geometry'].length_unit == 'm'
    assert project['Photos'].names == ['front', 'side']
    [group] = project.links
    assert {'Time History', 'Geometry', 'Photos'} <= set(group['members'])
    path = visualdynamics.random_vibration_report(
        fixture_path('plate', RUN), tmp_path / 'run.html', last=10.0,
        geometry=fixture_path('plate', 'geometry.npz'),
        photos=[folder / 'side.png', folder / 'front.png'])
    html = (tmp_path / 'run.html').read_text(encoding='utf-8')
    assert str(path) == str(tmp_path / 'run.html')
    # the page carries its blocks as JSON, the caption's dash escaped
    # to \u2014, which is what the file holds
    assert html.index('Test setup \\u2014 side') < html.index('Test setup \\u2014 front'), \
        'listed files appear in the order given'
    assert 'Test geometry and measurement locations' in html


def test_a_folder_with_no_photos_says_so(tmp_path):
    (tmp_path / 'empty').mkdir()
    with pytest.raises(ValueError, match='no png or jpeg'):
        visualdynamics.random_vibration_run(fixture_path('plate', RUN),
                                            photos=tmp_path / 'empty')


# ---- the plots, with no window ------------------------------------------


def test_every_comparison_plot_renders_headless(worked_up, tmp_path):
    """The three readings the app's toolbar offers, each to a file."""
    narrow, banded = _measured(worked_up)
    spec = worked_up.specifications[0]
    drawn = {
        'curves': lambda p: plot_comparison(narrow, spec, path=p, show=False),
        'octave': lambda p: plot_comparison(banded, spec, path=p, show=False),
        'error': lambda p: plot_bars(narrow, spec, 'error', path=p,
                                     show=False),
        'lines': lambda p: plot_bars(narrow, spec, 'lines', path=p,
                                     show=False),
        'both': lambda p: plot_series([('PSD', narrow), ('Spec', spec)],
                                      path=p, show=False),
    }
    for name, call in drawn.items():
        path = tmp_path / f'{name}.png'
        call(str(path))
        assert path.stat().st_size > 2000, f'{name} rendered to nothing'


def test_a_comparison_draws_the_channel_it_was_asked_for(worked_up, tmp_path):
    narrow, _banded = _measured(worked_up)
    spec = worked_up.specifications[0]
    pair = (spec.response_dof[2], spec.reference_dof[2])
    path = tmp_path / 'one.png'
    plot_comparison(narrow, spec, pair=pair, path=str(path), show=False)
    assert path.exists()
    with pytest.raises(ValueError, match='not one of the channels'):
        plot_comparison(narrow, spec, pair=('999X+', '999X+'),
                        path=str(path), show=False)


def test_a_reading_that_is_not_one_says_which_are(worked_up, tmp_path):
    narrow, _banded = _measured(worked_up)
    with pytest.raises(ValueError,
                       match="'error', 'lines' or 'srs'"):
        plot_bars(narrow, worked_up.specifications[0], 'rms',
                  path=str(tmp_path / 'x.png'), show=False)


def test_the_bar_labels_are_the_channels(worked_up, qt_app):
    """A bar chart with no channel names on it is a picture of six bars.

    Left to size itself the axis settles on the width a *number* wants —
    35 px, where '101Z+' needs 39 — and pyqtgraph silently draws no tick
    text that will not fit. So every label vanished from every exported
    chart, and would vanish in the app too for any name long enough.

    The width asked for is what is checked, not the width the axis has:
    a widget that was never laid out still carries its old geometry, and
    the exporter renders from the ask.
    """
    import pyqtgraph as pg
    from PySide6.QtGui import QFontMetrics

    from visualdynamics.core.compliance import channel_errors, compare_all
    from visualdynamics.plot.bars import error_chart
    from visualdynamics.theme import theme

    narrow, _banded = _measured(worked_up)
    rows = channel_errors(compare_all(worked_up.specifications[0], narrow))
    widget = pg.GraphicsLayoutWidget()
    chart = error_chart(widget.addPlot(row=0, col=0), rows, theme(None))
    axis = chart.plot.getAxis('left')
    metrics = QFontMetrics(axis.font())
    widest = max(metrics.horizontalAdvance(label) for label, _v in chart.rows)
    assert axis.maximumWidth() > widest, (
        'the labels do not fit, so pyqtgraph will draw none of them')
    labels = {label for _position, label in axis._tickLevels[0]}
    assert labels == {label for label, _v in chart.rows}
    widget.close()


def test_the_workflow_needs_no_application_shell(tmp_path):
    """The promise the modal example makes, made for this one too: the
    plots are Qt's job whether a window opens or not, but the app is
    never imported."""
    import pathlib
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    probe = (
        'import os, sys\n'
        'os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")\n'
        'import visualdynamics\n'
        f'visualdynamics.random_vibration_report({str(root / "testdata" / "plate" / RUN)!r},'
        f' {str(tmp_path / "probe.html")!r})\n'
        'print("app imported:", "visualdynamics.gui" in sys.modules)\n'
    )
    result = subprocess.run([sys.executable, '-c', probe], cwd=root,
                            capture_output=True, text=True, timeout=900,
                            check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'app imported: False' in result.stdout
    assert (tmp_path / 'probe.html').exists()


# ---- and the same numbers the window reaches ----------------------------


def test_the_script_and_the_window_agree(worked_up, window, pump):
    """A project built by clicking and one built by calling are the same
    project — the point of every verb living on `Project`."""
    window.import_paths([fixture_path('plate', RUN)])
    pump()
    assert window.project_type == 'Random Vibration'
    name = next(n for n, obj in window.objects.items()
                if isinstance(obj, TimeHistory))
    window.tree.setCurrentItem(window._item_for_object(name))
    window.compute_psds()
    # the calculator shows what it made, so the history is selected
    # again for the second computation — as it would be by hand
    window.tree.setCurrentItem(window._item_for_object(name))
    window.compute_multiple_coherence()
    clicked = next(obj for obj in window.objects.values()
                   if isinstance(obj, Psd)
                   and not isinstance(obj, Specification))
    called, _banded = _measured(worked_up)
    assert np.allclose(clicked.ordinate, called.ordinate)
    coherence = next(obj for obj in window.objects.values()
                     if isinstance(obj, MultipleCoherence))
    assert np.allclose(coherence.ordinate, worked_up.coherence.ordinate)


def test_a_folder_takes_the_run_s_own_name(tmp_path):
    """`path` may be the file to write or the folder to write into.

    A campaign is a folder of runs and nobody wants to name each
    report after its run by hand (Brandon, 2026-09-22)."""
    folder = tmp_path / 'reports'
    folder.mkdir()
    written = visualdynamics.random_vibration_report(
        fixture_path('plate', RUN), folder)
    assert pathlib.Path(written) == folder / 'random.html'
    assert pathlib.Path(written).is_file()
    # and a name that is not a folder is still the file to write
    named = visualdynamics.random_vibration_report(
        fixture_path('plate', RUN), tmp_path / 'named.html')
    assert pathlib.Path(named) == tmp_path / 'named.html'


def test_a_folder_that_does_not_exist_yet_is_a_file_name(tmp_path):
    """The one case with no right answer, decided and written down:
    there is no telling a folder nobody has made from a file nobody
    has written, and guessing folder would put the report where
    nothing looks for it."""
    written = visualdynamics.random_vibration_report(
        fixture_path('plate', RUN), tmp_path / 'no_extension')
    assert pathlib.Path(written) == tmp_path / 'no_extension'
    assert pathlib.Path(written).is_file()


def test_left_out_it_asks_once_and_reports_on_everything_chosen(tmp_path,
                                                                monkeypatch):
    """No run given: a dialog, and one report per run chosen.

    **One geometry for the batch, asked for once.** A campaign is a
    folder of runs on one article (Brandon, 2026-09-22), so the
    question is put before the loop and its answer is used by every
    report. Counted rather than merely observed: a test that only
    checks a dialog happened cannot tell one from three.
    """
    import shutil

    from visualdynamics.gui import ask

    runs = []
    for name in ('run_a', 'run_b', 'run_c'):
        runs.append(tmp_path / f'{name}.nc4')
        shutil.copy(fixture_path('plate', RUN), runs[-1])
    calls = {'runs': 0, 'geometry': 0}

    def chose_runs(*a, **k):
        calls['runs'] += 1
        return [str(r) for r in runs]

    def chose_geometry(*a, **k):
        calls['geometry'] += 1
        return str(fixture_path('plate', 'geometry.npz'))

    monkeypatch.setattr(ask, 'for_runs', chose_runs)
    monkeypatch.setattr(ask, 'for_geometry', chose_geometry)
    folder = tmp_path / 'out'
    folder.mkdir()
    written = visualdynamics.random_vibration_report(path=folder)
    assert calls == {'runs': 1, 'geometry': 1}, (
        f'asked once each for three runs, not per run: {calls}'
    )
    assert [pathlib.Path(p).name for p in written] == [
        'run_a.html', 'run_b.html', 'run_c.html']
    assert all(pathlib.Path(p).is_file() for p in written)
    assert all('Test geometry and measurement locations'
               in pathlib.Path(p).read_text() for p in written), (
        'and that one geometry is in every report of the batch'
    )


def test_one_run_chosen_still_answers_with_one_path(tmp_path, monkeypatch):
    from visualdynamics.gui import ask

    monkeypatch.setattr(ask, 'for_runs',
                        lambda *a, **k: [str(fixture_path('plate', RUN))])
    monkeypatch.setattr(ask, 'for_geometry', lambda *a, **k: None)
    written = visualdynamics.random_vibration_report(path=tmp_path)
    assert isinstance(written, str), 'a list is for a batch, not for one'
    assert pathlib.Path(written) == tmp_path / 'random.html'


def test_several_runs_refuse_one_file_name(tmp_path, monkeypatch):
    from visualdynamics.gui import ask

    monkeypatch.setattr(ask, 'for_runs',
                        lambda *a, **k: [str(fixture_path('plate', RUN))] * 2)
    monkeypatch.setattr(ask, 'for_geometry', lambda *a, **k: None)
    with pytest.raises(ValueError, match='cannot be written to one file'):
        visualdynamics.random_vibration_report(path=tmp_path / 'one.html')


def test_choosing_nothing_says_so(monkeypatch):
    from visualdynamics.gui import ask

    monkeypatch.setattr(ask, 'for_runs', lambda *a, **k: [])
    with pytest.raises(ValueError, match='no run chosen'):
        visualdynamics.random_vibration_report()


def test_a_named_run_without_a_geometry_asks_nothing(tmp_path, monkeypatch):
    """The contract this function was built on: a script that names its
    run and leaves the geometry out means *no geometry*, and nothing
    opens. `ASK` is how it says otherwise."""
    from visualdynamics.gui import ask

    def refuse(*a, **k):
        raise AssertionError('a dialog opened in a scripted call')

    monkeypatch.setattr(ask, 'for_runs', refuse)
    monkeypatch.setattr(ask, 'for_geometry', refuse)
    written = visualdynamics.random_vibration_report(
        fixture_path('plate', RUN), tmp_path / 'quiet.html')
    assert pathlib.Path(written).is_file()

    monkeypatch.setattr(ask, 'for_geometry',
                        lambda *a, **k: str(fixture_path('plate',
                                                         'geometry.npz')))
    asked = visualdynamics.random_vibration_report(
        fixture_path('plate', RUN), tmp_path / 'asked.html',
        geometry=visualdynamics.ASK)
    assert 'Test geometry and measurement locations' in \
        pathlib.Path(asked).read_text()


def test_cancelling_the_geometry_is_asked_once_for_the_whole_batch(
        tmp_path, monkeypatch):
    """The state where asking per run actually bites.

    Answer the geometry dialog with a file and the question stops
    asking itself either way, because the answer is no longer missing.
    **Cancel** is the case: None is what "no geometry" looks like, so a
    question put inside the loop would be put again for every run of
    the batch — three dialogs to say no once.
    """
    import shutil

    from visualdynamics.gui import ask

    runs = []
    for name in ('one', 'two', 'three'):
        runs.append(tmp_path / f'{name}.nc4')
        shutil.copy(fixture_path('plate', RUN), runs[-1])
    calls = {'geometry': 0}

    def cancelled(*a, **k):
        calls['geometry'] += 1

    monkeypatch.setattr(ask, 'for_runs', lambda *a, **k: [str(r) for r in runs])
    monkeypatch.setattr(ask, 'for_geometry', cancelled)
    folder = tmp_path / 'out'
    folder.mkdir()
    written = visualdynamics.random_vibration_report(path=folder)
    assert calls['geometry'] == 1, (
        f'cancel is an answer, given once for the batch: {calls}'
    )
    assert len(written) == 3
    assert all(pathlib.Path(p).is_file() for p in written)
    assert not any('Test geometry and measurement locations'
                   in pathlib.Path(p).read_text() for p in written), (
        'and no report pretends to a geometry nobody chose'
    )


def test_a_scene_names_no_unit_and_the_report_asks_for_none(tmp_path):
    """A scene draws a shape, never a coordinate or a real scale.

    The hint under each geometry figure used to end "coordinates in
    <unit>", and the only thing a declared length unit changed in a
    whole report was that label and the numbers behind it — no
    spectrum, no compliance number, no verdict (Brandon, 2026-09-22).
    Both are gone: the scenes say nothing about units, and the report
    call no longer takes one.
    """
    import inspect
    import json
    import re

    assert 'length_unit' not in inspect.signature(
        visualdynamics.random_vibration_report).parameters, (
        'the report takes no length unit'
    )
    # while the run, whose project lives on, still declares one
    assert 'length_unit' in inspect.signature(
        visualdynamics.random_vibration_run).parameters

    written = visualdynamics.random_vibration_report(
        fixture_path('plate', RUN), tmp_path / 'scenes.html',
        geometry=fixture_path('plate', 'geometry.npz'))
    page = pathlib.Path(written).read_text(encoding='utf-8')
    assert 'coordinates in ' not in page, 'no scene names a unit'
    assert 'units undefined' not in page, (
        'nor says it lacks one, having stopped claiming to need it'
    )
    payload = json.loads(re.search(r'<script id="data"[^>]*>(.*?)</script>',
                                   page, re.DOTALL).group(1))
    scenes = [b for b in payload['blocks'] if b['kind'] == 'scene']
    assert len(scenes) == 3, 'geometry, excitation DOFs, response DOFs'
    assert not any('unit' in scene for scene in scenes), (
        'and none of them carries the field that fed the label'
    )


def _history(level, dof='101Z+'):
    from visualdynamics.core.data import TimeHistory

    rng = np.random.default_rng(0)
    return TimeHistory(np.arange(0.0, 1.0, 0.01),
                       rng.normal(0.0, level, (1, 100)),
                       response_dof=[dof], ordinate_dim='acceleration',
                       ordinate_unit='m/s**2')


def test_the_quieter_recording_is_the_ambient_one():
    """A system identification is two recordings — the room, then the
    room with the shakers running — and the file does not say which is
    which. The quieter one is the ambient, by the only measure always
    there (Brandon, 2026-09-22)."""
    from visualdynamics.project import _quiet_and_driven

    project = visualdynamics.Project()
    project.add('First', _history(1.0))
    project.add('Second', _history(0.01))
    assert _quiet_and_driven(project) == ('Second', 'First')

    # and the other way round, so the order in the file proves nothing
    other = visualdynamics.Project()
    other.add('First', _history(0.01))
    other.add('Second', _history(1.0))
    assert _quiet_and_driven(other) == ('First', 'Second')

    # one recording has no ambient at all
    alone = visualdynamics.Project()
    alone.add('Only', _history(1.0))
    assert _quiet_and_driven(alone) == (None, 'Only')

    with pytest.raises(ValueError, match='no time data'):
        _quiet_and_driven(visualdynamics.Project())


def test_a_system_id_run_names_its_streams_and_measures_the_plant(tmp_path):
    """The workup, on the one recording a committed fixture holds: the
    stream named for what it is, the plant by H1, the coherence and a
    density. With no ambient the signal-to-noise stands down on its
    own rather than refusing the report."""
    project = visualdynamics.system_id_run(fixture_path('plate', RUN))
    assert project.project_type == 'System ID'
    assert 'Excitation Time History' in project
    assert 'Noise Time History' not in project, 'this file holds one stream'
    kinds = {type(obj).__name__ for obj in project.values()}
    assert {'Frf', 'MultipleCoherence', 'Psd'} <= kinds

    written = visualdynamics.system_id_report(
        fixture_path('plate', RUN), tmp_path / 'sysid.html')
    page = pathlib.Path(written).read_text(encoding='utf-8')
    assert 'System ID Report' in page
    assert 'The measured plant' in page


def test_the_system_id_report_asks_and_batches_like_the_random_one(
        tmp_path, monkeypatch):
    """The asking, the batch and the folder are one implementation
    shared by both one-call reports, so this proves the sharing."""
    import shutil

    from visualdynamics.gui import ask

    runs = []
    for name in ('sysid_a', 'sysid_b'):
        runs.append(tmp_path / f'{name}.nc4')
        shutil.copy(fixture_path('plate', RUN), runs[-1])
    calls = {'geometry': 0}

    def chose_geometry(*a, **k):
        calls['geometry'] += 1

    monkeypatch.setattr(ask, 'for_runs', lambda *a, **k: [str(r) for r in runs])
    monkeypatch.setattr(ask, 'for_geometry', chose_geometry)
    folder = tmp_path / 'out'
    folder.mkdir()
    written = visualdynamics.system_id_report(path=folder)
    assert [pathlib.Path(p).name for p in written] == ['sysid_a.html',
                                                       'sysid_b.html']
    assert calls['geometry'] == 1, 'asked once for the batch'
    assert all('System ID Report' in pathlib.Path(p).read_text()
               for p in written)


def test_the_ambient_stream_borrows_the_excitation_s_frames():
    """A ratio of densities only exists on lines both hold.

    The signal-to-noise the report draws is the driven density over
    the ambient one, so the two must be averaged on the same frames —
    and nothing in the file makes that happen. No committed recording
    holds the two streams a system identification is made of, so the
    second is built here from the first, quieter by a factor of a
    hundred, and the work is done through the seam that does not read
    a file.
    """
    import copy
    import dataclasses

    from visualdynamics.core.data import TimeHistory
    from visualdynamics.project import work_up_system_id

    project = visualdynamics.Project()
    project.import_file(fixture_path('plate', RUN))
    driven = next(name for name, obj in project.items()
                  if isinstance(obj, TimeHistory))
    quiet = copy.deepcopy(project[driven])
    quiet.ordinate = np.asarray(quiet.ordinate, dtype=float) / 100.0
    # framed differently to begin with, or borrowing the excitation's
    # frames would be a line that changed nothing and a test that
    # proved nothing
    quiet.averaging = dataclasses.replace(quiet.averaging,
                                          frame_length=1024, overlap=0.0)
    project.add('Ambient', quiet)
    assert quiet.averaging != project[driven].averaging

    noise, excitation = work_up_system_id(project)
    assert (noise, excitation) == ('Noise Time History',
                                   'Excitation Time History')
    assert project[noise].averaging == project[excitation].averaging, (
        'the ambient is averaged on the frames the excitation was'
    )
    psds = [name for name, obj in project.items()
            if type(obj).__name__ == 'Psd']
    assert len(psds) == 2, 'a density from each stream'
    first, second = (np.asarray(project[name].abscissa) for name in psds)
    assert np.array_equal(first, second), (
        'and on one set of lines, or the ratio has nowhere to live'
    )
