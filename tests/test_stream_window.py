"""A long run is imported through a window chosen on a preview.

Brandon (2026-09-18): a 22.5 GB stream fits a 64 GB machine and not a
32 GB one, so the import of a stream that would take a large share
of the machine asks how much of it to read — one channel's envelope
over the whole run to choose from, the truncate reading's own span
over it, and the cost of the window in samples and gigabytes against
the machine's memory. The window is read as a slice of the file, on
the run's own clock, and a script says the same thing the dialog
asks: `import_file(path, start=, stop=, channels=)`.
"""

import tracemalloc

import netCDF4
import numpy as np
import pytest
from test_large_records import _stream
from test_rattlesnake_transformation import CONTROL, MATRIX, MEASURED, write_run

import visualdynamics
from visualdynamics.io import rattlesnake

MB = 2 ** 20
G = 9.80665


def _raw(path, variable='time_data'):
    with netCDF4.Dataset(path) as ds:
        return np.asarray(ds.variables[variable][()])


# ---- the window, through the API --------------------------------------------

def test_a_window_is_the_slice_on_the_runs_own_clock(tmp_path):
    path = _stream(str(tmp_path / 'run.nc4'), samples=1 << 14)   # 4 s at 4096
    raw = _raw(path)
    history = visualdynamics.import_file(path, start=1.0, stop=2.0)['time_data']
    assert history.ordinate.shape == (8, 4097), 'inclusive at both instants'
    assert history.abscissa[0] == 1.0 and history.abscissa[-1] == 2.0, \
        'the clock is the run\'s, as a truncation keeps it'
    assert np.allclose(history.ordinate.real, raw[:, 4096:8193] * G)
    assert history.ordinate_unit[0] == 'g'
    assert history.comment[0] == '1 to 2 s of run.nc4'


def test_either_end_of_the_window_may_be_open(tmp_path):
    path = _stream(str(tmp_path / 'run.nc4'), samples=1 << 14)
    late = visualdynamics.import_file(path, start=3.0)['time_data']
    early = visualdynamics.import_file(path, stop=0.5)['time_data']
    assert late.abscissa[0] == 3.0 and late.ordinate.shape[1] == 16384 - 12288
    assert early.abscissa[0] == 0.0 and early.ordinate.shape[1] == 2049


def test_a_window_that_misses_a_stream_leaves_it_out(tmp_path):
    """A system-ID stream a few seconds long, asked for its second
    minute, is not an empty history: it is not there, and everything
    else in the file still is."""
    path, _raw_run = write_run(str(tmp_path / 'run.nc4'))      # 2 s at 256
    out = visualdynamics.import_file(path, start=5.0, stop=6.0)
    assert 'time_data' not in out
    assert 'Random_specification' in out and 'channel_table' in out


def test_a_backward_or_empty_window_is_refused_by_name(tmp_path):
    path = _stream(str(tmp_path / 'run.nc4'), samples=1 << 12)
    with pytest.raises(ValueError, match='runs forward'):
        visualdynamics.import_file(path, start=2.0, stop=1.0)
    with pytest.raises(ValueError, match='finite'):
        visualdynamics.import_file(path, start=float('nan'))


def test_channels_import_by_coordinate_or_row_in_the_order_asked(tmp_path):
    path = _stream(str(tmp_path / 'run.nc4'), samples=1 << 12)
    raw = _raw(path)
    history = visualdynamics.import_file(path, channels=['103Z+', 0, '103Z+'])['time_data']
    assert history.response_dof == ['103Z+', '101Z+'], 'once each, as asked'
    assert np.allclose(history.ordinate.real, raw[[2, 0]] * G)
    assert history.comment == ['', ''], 'a subset is not a window'
    with pytest.raises(ValueError, match=r"'999X\+' is not in the table"):
        visualdynamics.import_file(path, channels=['999X+'])


def test_a_windowed_subset_reads_only_what_it_keeps(tmp_path, monkeypatch):
    """The window is a slice of the file, not the stream read and cut:
    a quarter of two channels costs a quarter of two channels."""
    path = _stream(str(tmp_path / 'run.nc4'))                  # 8 × 1M, 64 MB
    monkeypatch.setattr(rattlesnake, 'READ_SAMPLES', 1 << 16)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        history = visualdynamics.import_file(
            path, start=0.0, stop=64.0, channels=['101Z+', '105Z+'])['time_data']
        _now, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert history.ordinate.shape == (2, 262145)
    assert peak < 3 * history.ordinate.nbytes, \
        f'peak {peak / MB:.0f} MB for a {history.ordinate.nbytes / MB:.0f} MB window'


def test_virtual_rows_follow_only_when_every_control_channel_does(tmp_path):
    path, raw = write_run(str(tmp_path / 'run.nc4'))
    whole = visualdynamics.import_file(
        path, channels=[f'{101 + i}Z+' for i in range(MEASURED)])['time_data']
    assert whole.num_records == MEASURED + 2, 'the controls, and the rows over them'
    assert np.allclose(whole.ordinate[-2:].real, MATRIX @ raw[CONTROL])
    part = visualdynamics.import_file(path, channels=['101Z+', '102Z+'])['time_data']
    assert part.num_records == 2, 'a row over a missing channel is left out, not made up'
    windowed = visualdynamics.import_file(path, start=0.5, stop=1.0)['time_data']
    assert windowed.comment[-1].startswith('0.5 to 1 s of run.nc4; row 2 of the Random')


# ---- what the dialog reads ---------------------------------------------------

def test_the_summary_costs_nothing_to_read(tmp_path):
    path = _stream(str(tmp_path / 'run.nc4'), streams=2)        # 2 × 64 MB
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        summary = rattlesnake.stream_summary(path)
        _now, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * MB, f'peak {peak / MB:.1f} MB reading a summary'
    assert summary['sample_rate'] == 4096.0
    assert summary['channels'] == [f'{101 + i}Z+' for i in range(8)]
    assert [s['key'] for s in summary['streams']] == ['time_data', 'time_data_2']
    assert summary['streams'][0]['bytes'] == 8 * (1 << 20) * 8
    assert summary['streams'][0]['seconds'] == ((1 << 20) - 1) / 4096.0
    assert summary['memory'] == rattlesnake.machine_memory() > 0


def test_a_package_has_no_streams_to_summarize(tmp_path):
    import netCDF4 as nc

    path = str(tmp_path / 'package.nc4')
    with nc.Dataset(path, 'w') as ds:
        ds.createGroup('Random').sysid_sample_rate = 256.0
    assert rattlesnake.stream_summary(path)['streams'] == []


def test_the_preview_is_the_channels_envelope_in_the_files_unit(tmp_path):
    path = _stream(str(tmp_path / 'run.nc4'), samples=1 << 16)
    raw = _raw(path)
    preview = rattlesnake.stream_preview(path, '103Z+', points=64)
    grid = raw[2].reshape(64, 1024)
    assert np.array_equal(preview['low'], grid.min(axis=1))
    assert np.array_equal(preview['high'], grid.max(axis=1))
    assert preview['dof'] == '103Z+' and preview['unit'] == 'g'
    assert preview['times'][0] == pytest.approx(1023 / 2 / 4096.0)
    assert preview['times'][-1] <= ((1 << 16) - 1) / 4096.0
    assert preview['samples'] == 1 << 16 and preview['sample_rate'] == 4096.0


def test_the_preview_is_read_a_slab_at_a_time(tmp_path, monkeypatch):
    """A 22 GB run's preview holds a slab, never the channel."""
    path = _stream(str(tmp_path / 'run.nc4'), channels=2)        # 2 × 1M
    monkeypatch.setattr(rattlesnake, 'READ_SAMPLES', 1 << 14)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        preview = rattlesnake.stream_preview(path, 1, points=1000)
        _now, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    channel = (1 << 20) * 8
    assert len(preview['low']) == 1000 and np.all(preview['low'] <= preview['high'])
    assert peak < channel / 8, f'peak {peak / MB:.1f} MB against an {channel / MB:.0f} MB channel'


def test_a_lead_in_of_nan_is_a_gap_in_the_envelope(tmp_path):
    """The stress stream opens with a hundred seconds of NaN — what
    the controller writes before the first sample arrives — and the
    preview said so with a numpy warning per slab. A bucket of nothing
    is NaN, quietly."""
    import warnings

    path = _stream(str(tmp_path / 'run.nc4'), samples=1 << 12)
    with netCDF4.Dataset(path, 'a') as ds:
        ds.variables['time_data'][:, :1024] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        preview = rattlesnake.stream_preview(path, 0, points=8)
    assert np.isnan(preview['low'][:2]).all() and np.isnan(preview['high'][:2]).all()
    assert np.isfinite(preview['low'][2:]).all()


def test_a_stream_shorter_than_the_points_previews_whole(tmp_path):
    path = _stream(str(tmp_path / 'run.nc4'), samples=100)
    preview = rattlesnake.stream_preview(path, 0)
    assert len(preview['low']) == 100
    assert np.array_equal(preview['low'], preview['high'])
    assert preview['times'][-1] == 99 / 4096.0


def test_the_machine_answers_its_memory():
    assert rattlesnake.machine_memory() > 2 ** 30
    assert 0 < rattlesnake.LARGE_STREAM_SHARE < 1


# ---- the dialog ---------------------------------------------------------------

def _dialog(tmp_path, window, **stream):
    from visualdynamics.gui.stream_window import StreamWindowDialog
    from visualdynamics.theme import theme as resolve_theme

    path = _stream(str(tmp_path / 'run.nc4'), **stream)
    summary = rattlesnake.stream_summary(path)
    return path, StreamWindowDialog(path, summary, resolve_theme(window.theme_name))


def test_the_dialog_offers_the_whole_run_and_says_what_it_costs(window, tmp_path):
    _path, dialog = _dialog(tmp_path, window, samples=1 << 14)
    assert dialog.options() == {}, 'nothing cut: the plain import'
    assert 'run.nc4 holds 8 channels for 4 s at 4096 Hz' in dialog.header.text()
    assert dialog.cost.text().startswith('16,384 samples × 8 channels = 0.00 GB')
    assert dialog.import_button.isEnabled()
    assert dialog.preview['dof'] == '101Z+'
    assert '101Z+ [g]' in dialog.plot.getAxis('left').labelText
    assert dialog.panel.derived['samples'].text() == '16384 of 16384'
    assert not dialog.stream_box.isVisibleTo(dialog), 'one stream needs no chooser'


def test_a_span_set_on_the_panel_is_the_window(window, tmp_path):
    _path, dialog = _dialog(tmp_path, window, samples=1 << 14)
    dialog.panel.start_box.setValue(1.0)
    dialog.panel.stop_box.setValue(2.0)
    assert dialog.options() == {'start': 1.0, 'stop': 2.0}
    assert dialog.overlay.region.getRegion() == (1.0, 2.0), 'the marks follow'
    assert dialog.cost.text().startswith('4,097 samples × 8 channels')
    assert dialog.panel.derived['samples'].text() == '4097 of 16384'


def test_a_dragged_span_restates_the_panel(window, pump, tmp_path):
    _path, dialog = _dialog(tmp_path, window, samples=1 << 14)
    dialog.overlay.region.setRegion((0.5, 1.5))
    pump()
    assert dialog.panel.start_box.value() == pytest.approx(0.5)
    assert dialog.panel.stop_box.value() == pytest.approx(1.5)
    assert dialog.options() == {'start': 0.5, 'stop': 1.5}


def test_unchecked_channels_are_left_out(window, tmp_path):
    from PySide6.QtCore import Qt

    _path, dialog = _dialog(tmp_path, window, samples=1 << 12)
    for row in (0, 1):
        dialog.channel_list.item(row).setCheckState(Qt.CheckState.Unchecked)
    assert dialog.options() == {'channels': [f'{103 + i}Z+' for i in range(6)]}
    assert '× 6 channels' in dialog.cost.text()
    for row in range(2, 8):
        dialog.channel_list.item(row).setCheckState(Qt.CheckState.Unchecked)
    assert not dialog.import_button.isEnabled()
    assert 'no channel is checked' in dialog.cost.text()


def test_the_preview_is_one_trace_of_each_buckets_extremes(window, tmp_path):
    """Two curves — the lows and the highs — read as two noisy
    signals of opposite sign (Brandon, 2026-09-18). One trace, least
    and greatest at each bucket's center, reads as the record."""
    _path, dialog = _dialog(tmp_path, window, samples=1 << 12)
    assert len(dialog._curves) == 1
    x, y = dialog._curves[0].getData()
    preview = dialog.preview
    assert np.array_equal(x, np.repeat(preview['times'], 2))
    assert np.array_equal(y[0::2], preview['low'])
    assert np.array_equal(y[1::2], preview['high'])


def test_the_preview_channel_is_the_users_to_pick(window, tmp_path):
    path, dialog = _dialog(tmp_path, window, samples=1 << 12)
    dialog.channel_box.setCurrentIndex(2)
    assert dialog.preview['dof'] == '103Z+'
    expected = rattlesnake.stream_preview(path, 2)
    assert np.array_equal(dialog.preview['high'], expected['high'])
    assert '103Z+ [g]' in dialog.plot.getAxis('left').labelText


def test_the_largest_stream_is_previewed_and_the_window_is_the_streams(window, tmp_path):
    _path, dialog = _dialog(tmp_path, window, samples=1 << 12, streams=2)
    assert dialog.stream_box.isVisibleTo(dialog)
    assert dialog.stream['key'] == 'time_data', 'equal sizes: the first'
    dialog.stream_box.setCurrentIndex(1)
    assert dialog.stream['key'] == 'time_data_2'
    assert dialog.preview['high'].max() > 5.0, 'the second stream is ten times louder'


# ---- the window asks only when it matters --------------------------------------

def _histories(window):
    from visualdynamics.core.data import TimeHistory

    return [obj for obj in window.project.values() if isinstance(obj, TimeHistory)]


def test_a_small_stream_imports_whole_without_a_question(window, pump, tmp_path,
                                                          monkeypatch):
    from visualdynamics.gui import stream_window

    path = _stream(str(tmp_path / 'run.nc4'), samples=1 << 12)
    monkeypatch.setattr(stream_window, 'ask_stream_window',
                        lambda *a: pytest.fail('a small stream was asked about'))
    monkeypatch.setattr(rattlesnake, 'machine_memory', lambda: 1 << 40)
    window.import_paths([path])
    pump()
    assert _histories(window)[0].ordinate.shape == (8, 1 << 12)
    assert f'project.import_file({path!r})' in window.project.journal


def test_a_large_stream_imports_the_window_chosen(window, pump, tmp_path,
                                                  monkeypatch):
    """The gate is the stream's share of the machine; the dialog's
    answer goes to the importer as the keywords a script would say,
    and the journal replays exactly that."""
    from visualdynamics.gui import stream_window

    path = _stream(str(tmp_path / 'run.nc4'), samples=1 << 14)
    monkeypatch.setattr(rattlesnake, 'machine_memory', lambda: 1)
    asked = []

    def choose(parent, asked_path, summary, colors):
        asked.append((asked_path, summary['streams'][0]['bytes']))
        return {'start': 1.0, 'stop': 2.0, 'channels': ['101Z+', '104Z+']}

    monkeypatch.setattr(stream_window, 'ask_stream_window', choose)
    window.import_paths([path])
    pump()
    assert asked == [(path, 8 * (1 << 14) * 8)]
    history = _histories(window)[0]
    assert history.response_dof == ['101Z+', '104Z+']
    assert history.abscissa[0] == 1.0 and history.ordinate.shape[1] == 4097
    assert (f"project.import_file({path!r}, start=1.0, stop=2.0, "
            "channels=['101Z+', '104Z+'])") in window.project.journal


def test_the_dialog_is_asked_under_the_arrow_not_the_wait_cursor(
        window, pump, tmp_path, monkeypatch):
    """A drop's wait cursor stays up for the import but comes down
    while the window is being chosen, and goes back up after."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from visualdynamics.gui import stream_window

    path = _stream(str(tmp_path / 'run.nc4'), samples=1 << 12)
    monkeypatch.setattr(rattlesnake, 'machine_memory', lambda: 1)
    seen = []

    def choose(*_args):
        seen.append(QApplication.overrideCursor())
        return {}

    monkeypatch.setattr(stream_window, 'ask_stream_window', choose)
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        window.import_paths([path])
        pump()
        assert seen == [None], 'the arrow while asking'
        assert QApplication.overrideCursor() is not None, 'and busy again after'
    finally:
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
    assert len(_histories(window)) == 1


def test_cancelling_the_dialog_skips_the_file(window, pump, tmp_path, monkeypatch):
    from visualdynamics.gui import stream_window

    path = _stream(str(tmp_path / 'run.nc4'), samples=1 << 12)
    monkeypatch.setattr(rattlesnake, 'machine_memory', lambda: 1)
    monkeypatch.setattr(stream_window, 'ask_stream_window', lambda *a: None)
    window.import_paths([path])
    pump()
    assert _histories(window) == []
