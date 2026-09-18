"""A long run costs its own size, not several times it.

A 22.5 GB Rattlesnake stream imported and then took the application
down at 67 GB when its time history went onto the 3-D stage (Brandon,
2026-09-18). Measured on the 1.3 GB stress stream: the import held
1.17 GB and peaked at 2.4 GB, the stage arrays peaked at 4.9 GB and
the flat plot at 4.8 GB — the importer scaled into a second whole
copy, the display conversion gathered and scaled into two more, and
the decimator padded a third. These tests hold every one of those
paths to a bounded transient, measured with tracemalloc on data big
enough to see and small enough to run in seconds.
"""

import tracemalloc

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core import data as core_data
from visualdynamics.units import SYSTEMS

MB = 2 ** 20


def _history(records=16, samples=1 << 20):
    """A history of `records` × `samples` float64: 128 MB at the defaults,
    rows of 8 MB."""
    rng = np.random.default_rng(11)
    t = np.arange(samples) / 4096.0
    return visualdynamics.TimeHistory(
        t, rng.standard_normal((records, samples)),
        response_dof=[f'{101 + k}Z+' for k in range(records)],
        ordinate_dim='acceleration', ordinate_unit='m/s**2')


def _peak(fn):
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        result = fn()
        _now, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


# ---- the display conversion ------------------------------------------------

def test_the_conversion_holds_one_block_beside_its_answer(monkeypatch):
    """Converting whole gathered every asked-for record and scaled the
    gather: two whole copies beside the output. In blocks, the output
    is the only whole array."""
    history = _history()
    size = history.ordinate.nbytes
    row = history.ordinate.shape[1] * 8
    monkeypatch.setattr(core_data, 'DISPLAY_BLOCK_BYTES', 4 * MB)   # one row
    out, peak = _peak(lambda: history.display_ordinate(SYSTEMS['in-slinch-lbf-s']))
    assert out.shape == history.ordinate.shape
    # the answer, plus a block's gather, its scaling and the block —
    # a handful of rows, never a second whole copy
    assert peak < size + 6 * row, f'peak {peak / MB:.0f} MB for {size / MB:.0f} MB'
    assert np.allclose(out, history.ordinate / 0.0254)


def test_blocks_convert_exactly_what_the_whole_conversion_does(monkeypatch):
    history = _history(records=5, samples=1 << 16)
    us = SYSTEMS['in-slinch-lbf-s']
    whole = history.display_ordinate(us, [4, 0, 2])
    pieces = list(history.display_blocks(us, [4, 0, 2], block_bytes=1))
    assert [(a, b) for a, b, _ in pieces] == [(0, 1), (1, 2), (2, 3)], \
        'a row is never split, and a block is at least a row'
    assert np.array_equal(np.vstack([block for _, _, block in pieces]), whole)


# ---- the 3-D stage ---------------------------------------------------------

def test_the_stage_thins_a_long_run_a_block_at_a_time(monkeypatch):
    """The stage never holds a converted copy of the whole object:
    each block is converted, thinned and let go."""
    from visualdynamics.viz.waterfall import waterfall_arrays

    history = _history()
    size = history.ordinate.nbytes
    row = history.ordinate.shape[1] * 8
    monkeypatch.setattr(core_data, 'DISPLAY_BLOCK_BYTES', 8 * MB)   # one row
    arrays, peak = _peak(lambda: waterfall_arrays(
        history, list(range(history.num_records)), SYSTEMS['in-slinch-lbf-s']))
    assert len(arrays['curves']) == history.num_records
    assert all(len(cx) <= 4096 for cx, _cz in arrays['curves'])
    # a block, its conversion and the decimator's padding and search:
    # a handful of rows, whatever the object weighs — it was four
    # times the object
    assert peak < 12 * row, f'peak {peak / MB:.0f} MB for {size / MB:.0f} MB'
    assert peak < 0.75 * size


def test_the_stage_draws_the_same_curves_however_it_is_blocked(monkeypatch):
    from visualdynamics.viz.waterfall import waterfall_arrays

    history = _history(records=6, samples=1 << 17)
    us = SYSTEMS['in-slinch-lbf-s']
    records = list(range(history.num_records))
    monkeypatch.setattr(core_data, 'DISPLAY_BLOCK_BYTES', 1 << 40)
    whole = waterfall_arrays(history, records, us)['curves']
    monkeypatch.setattr(core_data, 'DISPLAY_BLOCK_BYTES', 1)
    blocked = waterfall_arrays(history, records, us)['curves']
    assert len(whole) == len(blocked) == len(records)
    for (wx, wz), (bx, bz) in zip(whole, blocked):
        assert np.array_equal(wx, bx) and np.array_equal(wz, bz)


# ---- the import ------------------------------------------------------------

def _stream(path, channels=8, samples=1 << 20, streams=1):
    """A Rattlesnake-shaped stream: `channels` × `samples` float64 per
    stream, 64 MB at the defaults."""
    import netCDF4

    rng = np.random.default_rng(3)
    with netCDF4.Dataset(path, 'w', format='NETCDF4') as ds:
        ds.sample_rate = 4096.0
        ds.createDimension('response_channels', channels)
        group = ds.createGroup('channels')
        for name, values in (
                ('node_number', [str(101 + i) for i in range(channels)]),
                ('node_direction', ['Z+'] * channels),
                ('unit', ['g'] * channels)):
            group.createVariable(name, str, ('response_channels',))[:] = \
                np.array(values, dtype=object)
        for k in range(streams):
            suffix = '' if k == 0 else f'_{k}'
            ds.createDimension(f'time_samples{suffix}', samples)
            var = ds.createVariable(f'time_data{suffix}', 'f8',
                                    ('response_channels', f'time_samples{suffix}'))
            var[...] = rng.standard_normal((channels, samples)) * (1.0 + 9.0 * k)
    return path


def test_the_import_holds_the_stream_once(tmp_path, monkeypatch):
    """The file's read is scaled in place and becomes the history; a
    scaled copy beside it was the second whole copy an import cost."""
    from visualdynamics.io import rattlesnake

    path = _stream(str(tmp_path / 'long.nc4'))
    size = 8 * (1 << 20) * 8
    monkeypatch.setattr(rattlesnake, 'READ_SAMPLES', 1 << 16)
    out, peak = _peak(lambda: visualdynamics.import_file(path))
    history = out['time_data']
    assert history.ordinate.nbytes == size
    assert history.ordinate_unit[0] == 'g'
    # the stream once, its clock, and a slab or two in flight; a whole
    # read double-buffered the stream
    row, slab = (1 << 20) * 8, 8 * (1 << 16) * 8
    assert peak < size + 2 * row + 4 * slab, \
        f'peak {peak / MB:.0f} MB for {size / MB:.0f} MB'


def test_the_system_id_sniff_reads_a_slice_not_the_stream(tmp_path):
    """Two streams, quiet then loud: the question is answered from the
    first seconds of each, not from reading both whole twice over."""
    from visualdynamics.io.rattlesnake import SNIFF_SAMPLES, streamed_sysid_candidate

    path = _stream(str(tmp_path / 'sysid.nc4'), streams=2)
    answer, peak = _peak(lambda: streamed_sysid_candidate(path))
    assert answer is True
    slice_bytes = 8 * SNIFF_SAMPLES * 8
    assert peak < 3 * slice_bytes, \
        f'peak {peak / MB:.0f} MB; a slice is {slice_bytes / MB:.0f} MB'


@pytest.mark.parametrize('samples', [1 << 10])
def test_a_short_stream_still_answers_the_sniff(tmp_path, samples):
    """Shorter than the slice is fine: the slice is a ceiling."""
    from visualdynamics.io.rattlesnake import streamed_sysid_candidate

    path = _stream(str(tmp_path / 'short.nc4'), samples=samples, streams=2)
    assert streamed_sysid_candidate(path) is True


# ---- the 2-D plot -----------------------------------------------------------

def test_the_plot_converts_only_the_curves_it_draws(qt_app, monkeypatch):
    """The 2-D plot has to hold a converted copy of every row it draws
    — pyqtgraph thins them on the fly as the view zooms, so it needs
    the whole row — but it drew from a conversion of the whole
    selection, made before the curve budget said which few of them
    would be drawn. The rows are converted after the budget now, in
    blocks, for the drawn curves alone."""
    import pyqtgraph as pg

    from visualdynamics.plot import build_plots

    history = _history()
    row = history.ordinate.shape[1] * 8
    monkeypatch.setattr(core_data, 'DISPLAY_BLOCK_BYTES', 8 * MB)   # one row
    layout = pg.GraphicsLayoutWidget()
    (drawn, requested), peak = _peak(lambda: build_plots(
        layout, [('run', history, None)], unit_system=SYSTEMS['in-slinch-lbf-s'],
        max_records=4))
    assert (drawn, requested) == (4, 16)
    # the four drawn rows, a block and a curve's transients; converting
    # the selection cost sixteen rows before a curve was chosen
    assert peak < 9 * row, f'peak {peak / MB:.0f} MB for {row / MB:.0f} MB rows'
    layout.clear()
