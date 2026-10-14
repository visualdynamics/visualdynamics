"""A Rattlesnake run cut to a stretch and written back, nothing else
changed (Brandon, 2026-10-13: "load one in, truncate the time data, and
write it back out", every other piece of metadata kept).

`io.rattlesnake.save_window` copies the run group by group, every
attribute, dimension and variable as the file has them, but the
streams, each cut to the samples an import of the same window reads.
Pinned: the copy against the run, field by field, on the plate's real
controller files; importing the copy is importing the window; each
stream on its own clock; the refusals; a stopped copy leaves nothing;
and the two doors in the app — Save Truncated Copy in the import
window, and File → Truncate Rattlesnake File.
"""

import netCDF4
import numpy as np
import pytest
from conftest import fixture_path
from test_large_records import _stream

import visualdynamics
from visualdynamics.io import rattlesnake

RUNS = ['random', 'modal', 'shock']


def _differences(a, b, where=''):
    """Every attribute, dimension, variable, filter and value two groups
    do not share, but the time data's."""
    found = []
    attrs = lambda item: {k: np.asarray(item.getncattr(k)).tolist()
                          for k in item.ncattrs()}
    if attrs(a) != attrs(b):
        found.append(f'{where}@attrs')
    if list(a.dimensions) != list(b.dimensions) or list(a.variables) != list(b.variables):
        found.append(f'{where} names')
    for name, dim in a.dimensions.items():
        if dim.isunlimited() != b.dimensions[name].isunlimited():
            found.append(f'{where}{name} unlimited')
        if not name.startswith('time_samples') and len(dim) != len(b.dimensions[name]):
            found.append(f'{where}{name} length')
    for name, var in a.variables.items():
        other = b.variables[name]
        if (var.dtype != other.dtype or var.dimensions != other.dimensions
                or var.filters() != other.filters() or var.chunking() != other.chunking()
                or attrs(var) != attrs(other)):
            found.append(f'{where}{name} layout')
        elif not any(d.startswith('time_samples') for d in var.dimensions) and not \
                np.array_equal(np.asarray(var[...]), np.asarray(other[...]),
                               equal_nan=var.dtype != str):
            found.append(f'{where}{name} values')
    for name, group in a.groups.items():
        found += _differences(group, b.groups[name], f'{where}{name}/')
    return found


@pytest.mark.parametrize('run', RUNS)
def test_the_copy_is_the_run_but_its_time_data(tmp_path, run):
    """Every setting, channel field and specification as the run has
    them, the dimensions unlimited where the run's are; the time data
    the run's own samples over the window, both instants included."""
    path = str(fixture_path('plate', f'{run}.nc4'))
    target = str(tmp_path / 'cut.nc4')
    kept = rattlesnake.save_window(path, target, 2.0, 3.5)
    with netCDF4.Dataset(path) as a, netCDF4.Dataset(target) as b:
        assert _differences(a, b) == []
        rate = float(a.sample_rate)
        first = int(np.ceil(2.0 * rate))
        last = min(int(np.floor(3.5 * rate)) + 1, len(a.dimensions['time_samples']))
        assert kept == {'time_data': last - first}
        assert len(b.dimensions['time_samples']) == last - first
        assert np.array_equal(a['time_data'][:, first:last], b['time_data'][...])


@pytest.mark.parametrize('run', RUNS)
def test_importing_the_copy_is_importing_the_window(tmp_path, run):
    """The same samples a windowed import reads, on a clock that starts
    at zero: the file has nothing to say where the stretch began."""
    path = str(fixture_path('plate', f'{run}.nc4'))
    target = str(tmp_path / 'cut.nc4')
    rattlesnake.save_window(path, target, 2.0, 3.5)
    window = visualdynamics.import_file(path, start=2.0, stop=3.5)['time_data']
    copy = visualdynamics.import_file(target)['time_data']
    assert np.array_equal(copy.ordinate, window.ordinate)
    assert copy.abscissa[0] == 0.0
    assert np.allclose(copy.abscissa, window.abscissa - window.abscissa[0])


def test_a_variables_attributes_and_storage_go_with_it(tmp_path):
    """What the plate's files happen not to use, a run may: attributes on
    a variable, compression and chunking — a stream's included — a fill
    value, a group inside a group. Each copied as it is."""
    path = str(tmp_path / 'run.nc4')
    with netCDF4.Dataset(path, 'w', format='NETCDF4') as ds:
        ds.sample_rate = 1024.0
        ds.createDimension('response_channels', 2)
        ds.createDimension('time_samples', None)
        channels = ds.createGroup('channels')
        for name, values in (('node_number', ['1', '2']), ('node_direction', ['X+', 'Y+']),
                             ('unit', ['g', 'g'])):
            channels.createVariable(name, str, ('response_channels',))[:] = \
                np.array(values, dtype=object)
        stream = ds.createVariable('time_data', 'f8', ('response_channels', 'time_samples'),
                                   zlib=True, complevel=6, shuffle=True,
                                   chunksizes=(1, 512))
        stream[:, :4096] = np.arange(8192.0).reshape(2, 4096)
        stream.long_name = 'the stream'
        deep = ds.createGroup('Random').createGroup('extra')
        deep.createDimension('lines', 3)
        table = deep.createVariable('table', 'f4', ('lines',), fill_value=-1.0,
                                    zlib=True, fletcher32=True)
        table[:2] = [1.0, 2.0]
        table.units = 'g**2/Hz'
    target = str(tmp_path / 'cut.nc4')
    assert rattlesnake.save_window(path, target, 1.0, 2.0) == {'time_data': 1025}
    with netCDF4.Dataset(path) as a, netCDF4.Dataset(target) as b:
        assert _differences(a, b) == []
        assert b['time_data'].filters()['complevel'] == 6
        assert b['time_data'].long_name == 'the stream'
        assert np.array_equal(b['time_data'][...], a['time_data'][:, 1024:2049])


def test_each_stream_is_cut_on_its_own_clock(tmp_path):
    """A system-ID run streams twice; the window is each stream's, as the
    import reads it, and one it misses is left with no samples — as a
    spectral file's stream is, and as the import leaves it out."""
    path = _stream(str(tmp_path / 'run.nc4'), channels=2, samples=4 * 4096)
    with netCDF4.Dataset(path, 'a') as ds:
        # a second stream, a second long
        ds.createDimension('time_samples_1', 4096)
        ds.createVariable('time_data_1', 'f8',
                          ('response_channels', 'time_samples_1'))[...] = 1.0
    target = str(tmp_path / 'cut.nc4')
    assert rattlesnake.save_window(path, target, 2.0, 3.0) == {
        'time_data': 4097, 'time_data_1': 0}
    out = visualdynamics.import_file(target)
    assert out['time_data'].ordinate.shape == (2, 4097)
    assert 'time_data_2' not in out


def test_what_cannot_be_cut_is_refused_by_name(tmp_path):
    import shutil

    # a copy of the run: a refusal that failed to refuse writes over it,
    # and once wrote over the fixture itself (2026-10-13)
    path = str(tmp_path / 'random.nc4')
    shutil.copy(fixture_path('plate', 'random.nc4'), path)
    with pytest.raises(ValueError, match='fewer than two samples'):
        rattlesnake.save_window(path, str(tmp_path / 'a.nc4'), 500.0, 600.0)
    with pytest.raises(ValueError, match='runs forward'):
        rattlesnake.save_window(path, str(tmp_path / 'a.nc4'), 3.0, 2.0)
    with pytest.raises(ValueError, match='holds no time data'):
        rattlesnake.save_window(str(fixture_path('plate', 'random_spectra.nc4')),
                                str(tmp_path / 'a.nc4'), 1.0, 2.0)
    with pytest.raises(ValueError, match='over the run it is cut from'):
        rattlesnake.save_window(path, path, 1.0, 2.0)
    assert not (tmp_path / 'a.nc4').exists()
    assert visualdynamics.import_file(path)['time_data'].ordinate.shape[1] == 114688, \
        'the run untouched'



def test_a_copy_stopped_part_way_leaves_nothing(tmp_path, monkeypatch):
    """Told its progress slab by slab, and a stop raised from the callback
    — Cancel in the app — removes the half-written file."""
    from visualdynamics.core import progress

    class Every(progress.Ticker):
        # every slab told, not one each 50 ms: the stop lands part way
        def __init__(self, callback):
            super().__init__(callback, interval=0.0)

    monkeypatch.setattr(progress, 'Ticker', Every)
    monkeypatch.setattr(rattlesnake, 'READ_SAMPLES', 1024)
    path = str(fixture_path('plate', 'random.nc4'))
    target = tmp_path / 'cut.nc4'
    told = []
    rattlesnake.save_window(path, str(target), 1.0, 3.0,
                            progress=lambda done, total: told.append((done, total)))
    assert told[-1][0] == told[-1][1] == 8193 and len(told) > 2

    def stop(done, total):
        if done > 2048:
            assert target.exists(), 'stopped while it was being written'
            raise KeyboardInterrupt

    target.unlink()
    with pytest.raises(KeyboardInterrupt):
        rattlesnake.save_window(path, str(target), 1.0, 3.0, progress=stop)
    assert not target.exists()


def test_the_copy_is_named_for_its_window():
    assert rattlesnake.window_name('/runs/random.nc4', 30.0, 60.0) == \
        '/runs/random_30-60s.nc4'
    assert rattlesnake.window_name('/runs/random.nc4', 12.5, None) == \
        '/runs/random_12.5s-end.nc4'


# ---- the app --------------------------------------------------------------------

def _save_to(monkeypatch, target):
    from PySide6.QtWidgets import QFileDialog

    asked = {}

    def fake(parent, title, start, filters):
        asked['start'] = start
        return str(target), filters

    monkeypatch.setattr(QFileDialog, 'getSaveFileName', staticmethod(fake))
    return asked


def test_the_import_window_saves_a_truncated_copy(window, pump, tmp_path, monkeypatch):
    """Offered once the window is less than the run, named for it, and
    the run's copy written through the window's runner, journaled as the
    call a script makes; the dialog stays up."""
    from visualdynamics.gui.stream_window import StreamWindowDialog
    from visualdynamics.theme import theme as resolve_theme

    path = _stream(str(tmp_path / 'run.nc4'), samples=1 << 14)
    dialog = StreamWindowDialog(path, rattlesnake.stream_summary(path),
                                resolve_theme(window.theme_name), window,
                                save=window._save_rattlesnake_window)
    assert not dialog.save_button.isVisibleTo(dialog), 'the whole run is the run'
    dialog.panel.start_box.setValue(1.0)
    dialog.panel.stop_box.setValue(2.0)
    assert dialog.save_button.isVisibleTo(dialog)
    target = tmp_path / 'kept.nc4'
    asked = _save_to(monkeypatch, target)
    dialog.save_button.click()
    pump()
    assert asked['start'] == str(tmp_path / 'run_1-2s.nc4')
    assert target.exists()
    assert visualdynamics.import_file(str(target))['time_data'].ordinate.shape == (8, 4097)
    assert window.project.journal[-1] == (
        f'visualdynamics.io.rattlesnake.save_window({path!r}, {str(target)!r}, 1.0, 2.0)')
    assert dialog.cost.text().startswith('Saved kept.nc4: 4,097 samples of each channel')


def test_file_truncate_opens_the_window_with_no_import(window, pump, tmp_path,
                                                       monkeypatch):
    """Any run, whatever its size, without Import or the channel picker:
    the copy keeps every channel. A file with no stream is said so."""
    from PySide6.QtWidgets import QFileDialog

    from visualdynamics.gui.stream_window import StreamWindowDialog

    path = _stream(str(tmp_path / 'run.nc4'), samples=1 << 12)
    picked = [path]
    monkeypatch.setattr(QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a: (picked[0], '')))
    seen = []

    def shown(dialog):
        seen.append(dialog)
        return 0

    monkeypatch.setattr(StreamWindowDialog, 'exec', shown)
    window.truncate_rattlesnake_file()
    (dialog,) = seen
    assert dialog.windowTitle() == 'Truncate run.nc4'
    assert not dialog.import_button.isVisibleTo(dialog)
    assert not dialog.channel_list.isVisibleTo(dialog)
    assert 'the copy keeps every channel and every setting' in dialog.header.text()
    assert dialog.cost.text() == '4,096 samples of each channel, 1 s'
    picked[0] = str(fixture_path('plate', 'random_spectra.nc4'))
    window.truncate_rattlesnake_file()
    assert len(seen) == 1
    assert 'holds no streamed time data' in window.statusBar().currentMessage()
