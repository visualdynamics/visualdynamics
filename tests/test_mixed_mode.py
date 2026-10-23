"""A mixed-mode test as one specification, and its data sheet.

The numbers are a rotor composite's: a 1 grms floor written flat from 5
to 400 Hz, made from noise flat from 3 to 480 Hz so that no line of the
written band sits on the noise's own edge, with tones of 1.944, 2.5 and
1.5 g at 19.44, 38.88 and 58.32 Hz. By hand, the tones' mean squares are
1.89, 3.13 and 1.13 g², 2.48 g rms together, and the total 2.67 grms;
four tones with the 0.12 g one at 4.86 Hz and a 1.0 grms floor give the
data sheet's 1.00, 2.48 and 2.67. A fixed narrowband plus or minus 5
percent about 300 Hz runs from 285 to 315 Hz, 30 Hz wide.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import select_objects

from visualdynamics.core.data import Specification, TimeHistory
from visualdynamics.core.mixed_mode import (
    EXCLUSION,
    MixedSpecification,
    Narrowband,
    Tone,
    judge_mixed,
    settling_seconds,
)
from visualdynamics.core.sine import LOG, SineSweepSpecification
from visualdynamics.core.sine_tracking import SineTracking

G = 9.80665
FS = 1024.0
SECONDS = 20.0
CHANNEL = '1Z+'
TONES = ((19.44, 1.944), (38.88, 2.50), (58.32, 1.50))


def _noise(low, high, grms, rng, n):
    """Gaussian noise flat from `low` to `high` Hz at `grms`, in g."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / FS)
    spectrum[(f < low) | (f > high)] = 0.0
    x = np.fft.irfft(spectrum, n)
    return x * grms / np.sqrt(np.mean(x ** 2))


def _floor(level=1.0 / 395.0, low=5.0, high=400.0, **extra):
    """A flat floor written at two breakpoints, `level` in g²/Hz."""
    return Specification(np.array([low, high]),
                         np.array([[level, level]]) * G ** 2,
                         response_dof=[CHANNEL],
                         ordinate_dim=['acceleration**2/frequency'],
                         **extra)


def _record(tones=TONES, seconds=SECONDS, seed=33, extra=None):
    """The floor's noise (1 grms over 5 to 400 Hz, flat 3 to 480) with
    `tones` (Hz, g peak) added and `extra` (g) on top, in m/s²."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    # 1 grms over the written 395 Hz is 1/395 g²/Hz; spread flat over
    # 597 Hz that is sqrt(597/395) grms in all
    x = _noise(3.0, 480.0, np.sqrt(477.0 / 395.0), rng, n)
    t = np.arange(n) / FS
    for f, amplitude in tones:
        x = x + amplitude * np.sin(2.0 * np.pi * f * t + f)
    if extra is not None:
        x = x + extra(t, rng)
    return TimeHistory(t, x[None] * G, response_dof=[CHANNEL],
                       ordinate_dim=['acceleration'], ordinate_unit=['g'])


def _spec(tones=TONES, floor=None, **options):
    return MixedSpecification(_floor() if floor is None else floor,
                              [Tone(f, a * G) for f, a in tones],
                              seconds=SECONDS, **options)


@pytest.fixture(scope='module')
def composite():
    """The three-tone composite read against its own specification."""
    history = _record()
    spec = _spec()
    return history, spec, judge_mixed(history, spec)


# ---- the parts, by hand ---------------------------------------------------

def test_a_band_written_plus_or_minus_five_percent_runs_its_full_width():
    band = Narrowband.plus_minus(400.0, 5.0, 0.01 * G ** 2)
    assert band.edges() == pytest.approx((380.0, 420.0))
    assert band.width_at() == pytest.approx(40.0)
    assert band.rms == pytest.approx(np.sqrt(0.4) * G)
    # a share widens with the center, as a harmonic's band does
    assert band.width_at(800.0) == pytest.approx(80.0)
    edges = Narrowband.between(380.0, 420.0, 0.01 * G ** 2)
    assert edges.center == 400.0 and edges.width_at(800.0) == 40.0
    assert edges.label == '380-420 Hz band'


def test_a_swept_band_goes_and_comes_back():
    """56 to 93 Hz at 2 Hz/s is an 18.5 s leg, then back."""
    band = Narrowband(56.0, 1.0, width=4.0, sweep_to=93.0, rate=2.0)
    assert band.leg_seconds() == pytest.approx(18.5)
    assert band.center_at([0.0, 9.25, 18.5, 27.75, 37.0]) == pytest.approx(
        [56.0, 74.5, 93.0, 74.5, 56.0])
    assert band.span() == pytest.approx((54.0, 95.0))
    assert band.crossing_seconds(70.0) == pytest.approx(2.0)
    log = Narrowband.plus_minus(56.0, 5.0, 1.0, sweep_to=112.0, rate=6.0,
                                mode=LOG)
    assert log.leg_seconds() == pytest.approx(10.0)
    assert log.center_at(5.0) == pytest.approx(56.0 * 2 ** 0.5)
    assert Narrowband(56.0, 1.0, width=4.0).crossing_seconds(56.0) == np.inf


@pytest.mark.parametrize('fields,said', [
    ({'width': None}, 'one width'),
    ({'share': 0.1}, 'one width'),
    ({'level': 0.0}, 'level'),
    ({'sweep_to': 93.0}, 'both sweep_to and rate'),
    ({'sweep_to': 93.0, 'rate': -2.0}, 'rate is above zero'),
    ({'width': 200.0}, 'below zero frequency'),
    ({'mode': 7}, 'LINEAR or LOG'),
])
def test_a_band_that_cannot_be_run_is_refused(fields, said):
    given = {'center': 56.0, 'level': 1.0, 'width': 4.0, **fields}
    with pytest.raises(ValueError, match=said):
        Narrowband(**given)


def test_a_fixed_tone_is_written_as_a_dwell_and_read_back_fixed():
    """No segment type is added: the tone the file holds is the dwell a
    controller writes, and the tone table reads it back as fixed."""
    row = Tone(19.44, 1.944 * G, name='blade passage')
    tone = row.sine_tone(60.0, 2)
    assert tone.dwell_segments() == [0]
    assert tone.duration() == pytest.approx(60.0)
    assert tone.amplitude.shape == (2, 2)
    assert Tone.of(tone) == row
    with pytest.raises(ValueError, match='seconds it holds'):
        row.sine_tone()


def test_a_swept_tone_keeps_its_direction_in_the_rate_s_sign():
    down = Tone(40.0, G, sweep_to=20.0, rate=0.5)
    tone = down.sine_tone()
    assert tone.segment_rate[0] == -0.5
    assert tone.duration() == pytest.approx(40.0)
    assert Tone.of(tone) == Tone(40.0, G, sweep_to=20.0, rate=0.5,
                                 name='40-20 Hz')
    assert not Tone.of(tone).fixed


def test_the_three_figures_come_from_the_parts():
    """The data sheet's arithmetic: four tones of 0.12, 1.944, 2.5 and
    1.5 g peak over a 1.0 grms floor are 1.00, 2.48 and 2.67."""
    four = ((4.86, 0.12),) + TONES
    spec = MixedSpecification(_floor(1.0 / 495.0, 5.0, 500.0),
                              [Tone(f, a * G) for f, a in four],
                              seconds=60.0)
    written = {key: value / G for key, value in spec.reference_rms().items()}
    assert written['floor'] == pytest.approx(1.0)
    assert written['tones'] == pytest.approx(2.479, abs=5e-4)
    assert written['total'] == pytest.approx(2.673, abs=5e-4)
    band = Narrowband.between(380.0, 420.0, 0.01 * G ** 2)
    with_band = MixedSpecification(spec.floor, spec.tones, [band])
    assert with_band.reference_rms()['narrowbands'] == pytest.approx(
        np.sqrt(0.4) * G)
    assert with_band.reference_rms()['total'] / G == pytest.approx(
        np.sqrt(1.0 + 6.1489 + 0.4), abs=5e-4)


def test_the_lines_left_out_follow_each_tone_and_its_sweep():
    spec = MixedSpecification(_floor(), [Tone(20.0, G),
                                         Tone(100.0, G, sweep_to=120.0,
                                              rate=1.0)],
                              seconds=SECONDS, exclusion=1.5)
    f = np.arange(0.0, 200.0, 0.5)
    out = spec.left_out(f)
    assert f[out & (f < 50)].tolist() == [18.5, 19.0, 19.5, 20.0, 20.5,
                                          21.0, 21.5]
    assert f[out & (f > 50)][[0, -1]].tolist() == [98.5, 121.5]
    assert MixedSpecification(_floor()).left_out(f).sum() == 0


def test_narrowbands_and_the_width_go_on_a_copy_of_the_floor():
    """Asking a question of a project's floor never changes it."""
    floor = _floor()
    band = Narrowband.between(380.0, 390.0, 1.0)
    spec = MixedSpecification(floor, None, [band], 3.0)
    assert spec.narrowbands == (band,) and spec.exclusion == 3.0
    assert floor.narrowbands == () and floor.tone_exclusion is None
    assert MixedSpecification(floor).exclusion == EXCLUSION
    # and read back from the floor when it carries them
    assert MixedSpecification(spec.floor).narrowbands == (band,)


def test_tones_at_a_channel_the_floor_is_not_written_at_are_refused():
    sweep = SineSweepSpecification([Tone(20.0, G).sine_tone(10.0)],
                                   ['2X+'])
    with pytest.raises(ValueError, match='not written at'):
        MixedSpecification(_floor(), sweep)


def test_the_band_settles_in_about_one_over_its_width():
    assert settling_seconds(SineTracking(fixed=0.5)) == 2.0
    assert settling_seconds(SineTracking(proportional=0.1),
                            [Tone(20.0, G).sine_tone(1.0)]) == 0.5


# ---- the record read against it ----------------------------------------------

def test_each_tone_reads_its_amplitude_through_the_band(composite):
    _history, _spec, reading = composite
    for row, (f, amplitude) in zip(reading.tones, TONES):
        assert row.fixed and row.low == pytest.approx(f)
        assert row.reference == pytest.approx(amplitude * G)
        assert row.measured == pytest.approx(amplitude * G, rel=0.02)
        assert row.passed and row.inside == 1.0
        assert row.readings > 300
    assert reading.tones_reference / G == pytest.approx(2.4777, abs=1e-3)
    assert reading.tones_measured == pytest.approx(reading.tones_reference,
                                                   rel=0.01)


def test_the_tones_taken_out_leave_the_floor(composite):
    """The residual's rms is the floor's, where the record's is the
    total; the lines left out are 4 a tone at 1 Hz lines."""
    _history, _spec, reading = composite
    floor = reading.floor
    assert floor.spacing == 1.0
    assert floor.residual / G == pytest.approx(np.sqrt(477.0 / 395.0),
                                               rel=0.02)
    assert floor.measured == pytest.approx(floor.reference, rel=0.03)
    assert floor.reference / G == pytest.approx(np.sqrt(384.0 / 395.0),
                                                rel=0.005)
    assert floor.judged == 384 and floor.outside == 0 and floor.passed
    assert floor.left_out == pytest.approx(12.0 / 396.0)
    assert reading.left_out.sum() == 12
    # the composite still holds the tones
    assert reading.composite[reading.left_out].max() > 100 * \
        reading.residual[reading.left_out].max()
    assert reading.start == 2.0 and reading.passed


def test_the_total_is_every_line_and_carries_no_verdict(composite):
    _history, spec, reading = composite
    total = reading.total
    assert total.measured == pytest.approx(total.record, rel=0.005)
    assert total.measured / G == pytest.approx(
        np.sqrt(477.0 / 395.0 + 2.4777 ** 2), rel=0.01)
    assert total.reference / G == pytest.approx(np.sqrt(1.0 + 2.4777 ** 2),
                                                rel=1e-3)
    assert total.reference == pytest.approx(spec.reference_rms()['total'])
    assert not hasattr(total, 'passed')
    assert 'every part passes' in reading.describe('g', 1.0 / G)


def test_a_tone_written_fifteen_percent_high_fails_alone():
    history = _record()
    wrong = ((19.44, 1.944 * 1.15),) + TONES[1:]
    reading = judge_mixed(history, _spec(wrong))
    first, *rest = reading.tones
    assert first.error == pytest.approx(-13.0, abs=1.0)
    assert not first.passed and all(row.passed for row in rest)
    assert reading.floor.passed and not reading.passed
    assert 'outside: 19.44 Hz' in reading.describe()


def test_a_floor_written_two_decibels_low_fails_on_its_rms():
    reading = judge_mixed(_record(), _spec(floor=_floor(1.0 / 395.0
                                                        / 10 ** 0.2)))
    assert reading.floor.error == pytest.approx(25.9, abs=3.0)
    assert not reading.floor.passed
    assert all(row.passed for row in reading.tones)


def test_a_feature_inside_the_left_out_width_is_not_judged():
    """A 1 Hz wide bump 10 dB over the floor beside a tone: inside the
    width left out it is not judged; with nothing left out it fails."""
    def bump(t, rng):
        return _noise(39.5, 40.5, np.sqrt(10.0 / 395.0), rng, len(t))

    history = _record(extra=bump)
    kept = judge_mixed(history, _spec(exclusion=3.0))
    assert kept.floor.outside == 0 and kept.floor.passed
    bare = judge_mixed(history, _spec(exclusion=0.0))
    assert bare.floor.outside >= 2 and not bare.floor.passed


def test_without_tones_the_record_is_its_own_residual():
    """Random on random: nothing to take out."""
    history = _record(tones=())
    reading = judge_mixed(history, MixedSpecification(_floor()))
    assert reading.tones == ()
    assert np.array_equal(reading.residual, reading.composite)
    assert reading.floor.left_out == 0.0 and reading.passed
    assert reading.start == 0.0, 'no band to settle'


def test_a_fixed_band_is_judged_at_the_degrees_of_freedom_of_its_lines():
    """0.02 g²/Hz over 285 to 315 Hz on the floor: 0.775 grms."""
    def band(t, rng):
        return _noise(285.0, 315.0, np.sqrt(0.02 * 30.0), rng, len(t))

    narrow = Narrowband.plus_minus(300.0, 5.0, 0.02 * G ** 2,
                                   name='engine')
    reading = judge_mixed(_record(extra=band),
                          _spec(narrowbands=[narrow]))
    row, = reading.narrowbands
    assert (row.low, row.high) == pytest.approx((285.0, 315.0))
    assert row.reference / G == pytest.approx(np.sqrt(0.6))
    assert row.measured / G == pytest.approx(np.sqrt(0.6), rel=0.05)
    # 31 lines under Hann's 1.5-line noise bandwidth, at the floor's
    # per-line count
    assert row.dof == pytest.approx(reading.floor.dof * 31 / 1.5, rel=1e-6)
    assert (row.low_db, row.high_db) == (-3.0, 3.0)
    assert row.passed and reading.passed
    # its lines are the band's, not the floor's
    assert reading.band_lines.sum() == 31
    assert reading.floor.judged == 384 - 31


def test_a_band_missing_from_the_record_fails():
    narrow = Narrowband.between(285.0, 315.0, 0.02 * G ** 2)
    row, = judge_mixed(_record(), _spec(narrowbands=[narrow])).narrowbands
    # the floor alone, 1/395 g²/Hz, where 0.0225 is written: 9.5 dB short
    assert row.error_db == pytest.approx(-9.5, abs=0.5) and not row.passed


def _swept(level, width, c0, c1, rate):
    """A band `width` Hz wide at `level` g²/Hz whose center sweeps from
    `c0` to `c1` and back at `rate` Hz/s: complex noise flat over half
    the width either side of zero, moved up along the sweep."""
    def extra(t, rng):
        n = len(t)
        base = np.fft.fft(rng.standard_normal(n)
                          + 1j * rng.standard_normal(n))
        f = np.fft.fftfreq(n, 1.0 / FS)
        base[np.abs(f) > 0.5 * width] = 0.0
        moved = np.fft.ifft(base)
        band = Narrowband(c0, 1.0, width=width, sweep_to=c1, rate=rate)
        phase = 2.0 * np.pi * np.cumsum(band.center_at(t)) / FS
        y = np.real(moved * np.exp(1j * phase))
        return y * np.sqrt(level * width / np.mean(y ** 2))
    return extra


def test_a_swept_band_is_read_in_slices_at_its_own_scatter():
    """A 16 Hz band at 0.05 g²/Hz sweeping 150 to 250 Hz at 10 Hz/s: it
    covers any one frequency for 1.6 s, so it is read in 0.8 s slices,
    each worth a few degrees of freedom."""
    level = 0.05
    narrow = Narrowband(150.0, level * G ** 2, width=16.0, sweep_to=250.0,
                        rate=10.0, name='track')
    history = _record(extra=_swept(level, 16.0, 150.0, 250.0, 10.0))
    reading = judge_mixed(history, _spec(narrowbands=[narrow]))
    row, = reading.narrowbands
    assert row.swept and row.span == pytest.approx((142.0, 258.0))
    assert row.slices >= 20
    assert row.slice_dof < reading.floor.dof / 5, 'a slice is worth few'
    # one slice's scatter at so few degrees of freedom is wider than 3 dB
    from visualdynamics.core.mixed_mode import _scatter_band

    low, high = _scatter_band(3.0, row.slice_dof, 0.9)
    assert high - low > 6.0
    # and pooled over the sweep the band is judged at their sum
    assert row.dof == pytest.approx(row.slices * row.slice_dof, rel=0.2)
    assert abs(row.error_db) < 1.0
    assert row.passed and row.inside > 0.6
    # the floor is not judged where the band travelled
    f = reading.frequencies
    assert reading.band_lines[(f > 142.0) & (f < 258.0)].all()


def test_a_swept_band_run_at_a_quarter_of_its_level_fails():
    level = 0.05
    narrow = Narrowband(150.0, 4 * level * G ** 2, width=16.0,
                        sweep_to=250.0, rate=10.0, tolerance_db=3.0)
    history = _record(extra=_swept(level, 16.0, 150.0, 250.0, 10.0))
    row, = judge_mixed(history, _spec(narrowbands=[narrow])).narrowbands
    assert row.error_db < -4.0 and not row.passed


def test_a_swept_tone_is_tracked_along_its_sweep():
    tone = Tone(20.0, G, sweep_to=40.0, rate=1.0)
    spec = MixedSpecification(_floor(), [tone])

    def sweep(t, _rng):
        return np.cos(tone.sine_tone().phase_at(t))

    reading = judge_mixed(_record(tones=(), extra=sweep), spec)
    row, = reading.tones
    assert not row.fixed and (row.low, row.high) == (20.0, 40.0)
    assert row.measured / G == pytest.approx(1.0, rel=0.02) and row.passed
    assert reading.floor.left_out == pytest.approx(25.0 / 396.0)


def test_a_channel_neither_holds_is_refused():
    with pytest.raises(ValueError, match='not written at'):
        judge_mixed(_record(), _spec(), channel='9X+')


# ---- the file ----------------------------------------------------------------

def test_the_narrowbands_and_the_width_travel_with_the_floor(tmp_path):
    """Saved as attributes of the floor's own group: no new group."""
    import h5py

    from visualdynamics.io import native

    bands = [Narrowband.plus_minus(400.0, 5.0, 0.2, tolerance_db=4.0,
                                   name='engine'),
             Narrowband(56.0, 0.5, width=4.0, sweep_to=93.0, rate=2.0,
                        mode=LOG)]
    spec = MixedSpecification(_floor(), None, bands, 2.5)
    path = tmp_path / 'floor.vdyn'
    native.save(spec.floor, path)
    back = native.load(path)
    assert isinstance(back, Specification)
    assert back.narrowbands == tuple(bands)
    assert back.tone_exclusion == 2.5
    assert MixedSpecification(back).narrowbands == tuple(bands)
    with h5py.File(path, 'r') as f:
        groups = []
        f.visit(groups.append)
    assert not any('narrow' in name for name in groups)
    # a floor that never had them reads as a plain random specification
    native.save(_floor(), tmp_path / 'plain.vdyn')
    plain = native.load(tmp_path / 'plain.vdyn')
    assert plain.narrowbands == () and plain.tone_exclusion is None


# ---- the view ----------------------------------------------------------------

def test_the_sheet_has_a_row_for_each_part(composite):
    from visualdynamics.plot.mixed_mode import sheet_html, sheet_rows
    from visualdynamics.units import IN_LBF_S

    us = IN_LBF_S.with_units('g', acceleration='g')
    _history, _spec, reading = composite
    rows = sheet_rows(reading, us)
    assert [row[0].strip() for row in rows] == [
        'Floor, tone and band lines left out', 'Tones, 3 together',
        '19.44 Hz', '38.88 Hz', '58.32 Hz', 'Total, the whole drive']
    assert rows[2][2] == '1.94 g peak' and rows[2][-1] == 'passes'
    assert rows[0][2].endswith('g rms') and rows[-1][-2].startswith('none')
    text = sheet_html(reading, '#000000', us)
    assert '<b>fails</b>' not in text
    assert '±2 Hz left out about each tone' in text
    assert '−0.0' not in text, 'no minus zero'


def _drawn(plot):
    """Every item on a plot, those kept out of its bounds included."""
    box = plot.getViewBox()
    return list(dict.fromkeys([*plot.items, *box.addedItems,
                               *box.childGroup.childItems()]))


def test_the_view_hatches_the_left_out_lines_and_marks_each_tone(composite):
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from visualdynamics.plot.mixed_mode import build_mixed_mode

    _history, _spec, reading = composite
    layout = pg.GraphicsLayoutWidget()
    plot = build_mixed_mode(layout, reading)
    regions = [item for item in _drawn(plot)
               if isinstance(item, pg.LinearRegionItem)]
    assert len(regions) == 3
    assert all(region.brush.style() == Qt.BrushStyle.BDiagPattern
               for region in regions), 'a pattern, not a hue'
    words = [item.toPlainText() for item in _drawn(plot)
             if isinstance(item, pg.TextItem)]
    assert words.count('left out') == 3
    assert sum('g peak' in text for text in words) == 3
    names = [item.name() for item in plot.listDataItems() if item.name()]
    assert 'residual, tones taken out' in names
    sheet, = [item for item in layout.ci.items if getattr(item,
                                                          'is_data_sheet',
                                                          False)]
    assert 'Total, the whole drive' in sheet.text


def test_a_narrowband_s_edges_are_drawn_and_named():
    import pyqtgraph as pg

    from visualdynamics.plot.mixed_mode import build_mixed_mode

    def band(t, rng):
        return _noise(285.0, 315.0, np.sqrt(0.6), rng, len(t))

    narrow = Narrowband.between(285.0, 315.0, 0.02 * G ** 2, name='engine')
    reading = judge_mixed(_record(extra=band), _spec(narrowbands=[narrow]))
    layout = pg.GraphicsLayoutWidget()
    plot = build_mixed_mode(layout, reading)
    edges = sorted(item.band_edge for item in _drawn(plot)
                   if hasattr(item, 'band_edge'))
    assert edges == [285.0, 315.0]


def test_a_failing_part_is_bold_in_the_sheet():
    from visualdynamics.plot.mixed_mode import sheet_html

    wrong = ((19.44, 1.944 * 1.15),) + TONES[1:]
    reading = judge_mixed(_record(), _spec(wrong))
    assert sheet_html(reading, '#000').count('<b>fails</b>') == 1


def test_the_plot_draws_to_a_file_with_no_window(tmp_path):
    from visualdynamics.plot import plot_mixed_mode

    path = tmp_path / 'sheet.png'
    spec = _spec()
    plot_mixed_mode(_record(), spec.floor, spec.tones, exclusion=3.0,
                    path=str(path), show=False)
    assert path.stat().st_size > 10_000
    assert spec.floor.tone_exclusion is None, 'the floor is not changed'


# ---- the readout in the window ---------------------------------------------------

def _shown(window, pump, floor=True):
    spec = _spec()
    window.add_object('Record', _record())
    if floor:
        window.add_object('Floor', spec.floor)
    window.add_object('Sweep', spec.tones)
    select_objects(window, pump, 'Record')
    window.render_current()
    pump()
    pane = window.data_pane
    pane.sine_action.setChecked(True)
    pane._choose_sine(True)
    pump()
    pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    return pane.sine_panel.mixed


def test_the_readout_reads_the_sheet_in_the_sine_view(window, pump):
    panel = _shown(window, pump)
    assert panel.isVisible() and panel.reading is None
    assert 'Read Data Sheet' in panel.sheet.text()
    panel.read_button.click()
    pump()
    reading = panel.reading
    assert reading.passed and len(reading.tones) == 3
    text = panel.sheet.text()
    assert 'Tones, 3 together' in text and 'Total, the whole drive' in text
    assert panel.preview.isVisible() and panel.preview.items
    assert window.project.journal[-1] == (
        "visualdynamics.plot.plot_mixed_mode(project['Record'], "
        "project['Floor'], project['Sweep'], channel='1Z+', exclusion=2.0, "
        "band=visualdynamics.core.sine_tracking.SineTracking(fixed=0.5), "
        "tone_tolerance=10.0, path='mixed_mode.png', show=False)")


def test_an_edit_reads_the_sheet_again(window, pump):
    panel = _shown(window, pump)
    panel.read_button.click()
    pump()
    first = panel.reading
    panel.exclusion_box.setValue(3.0)
    pump()
    assert panel.reading is not first
    assert panel.reading.floor.exclusion == 3.0
    assert 'exclusion=3.0,' in window.project.journal[-1]
    assert window.project['Floor'].tone_exclusion is None, \
        'reading never changes the project\'s floor'


def test_the_readout_stays_away_without_a_floor(window, pump):
    panel = _shown(window, pump, floor=False)
    assert window.data_pane.sine_panel.isVisible()
    assert not panel.isVisible(), 'no floor, no mixed test (principle 3)'
