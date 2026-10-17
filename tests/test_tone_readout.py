"""A tone read off an averaged autospectrum, in amplitude.

The reading sums the lines a tone leaks into, less the floor beside
them, times the spacing: the tone's mean square whatever the spacing and
window, where the spike's height moves with both. Pinned against hand
values (Harris's noise bandwidths and scalloping losses), against a
composite of four tones on a 1 grms floor read at 1 Hz and 4 Hz lines,
and against single tones of known amplitude in a known floor, under
every window the averaging offers. Then the plot, the project's record
of the window, and the reading in the window.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import Psd, TimeHistory
from visualdynamics.core.tone_readout import (
    ToneWarning,
    leakage,
    noise_bandwidth,
    read_tone,
    spike_height,
)

G = 9.80665
FS = 2048.0
#: four tones of a rotor test, Hz and g peak: one small at the floor's
#: lower edge, three standing well clear of it
TONES = ((4.86, 0.12), (19.44, 1.944), (38.88, 2.50), (58.32, 1.50))


def _floor(seconds, low=5.0, high=500.0, grms=1.0, seed=3304):
    """Gaussian noise flat from `low` to `high` Hz at `grms`, in g."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / FS)
    spectrum[(f < low) | (f > high)] = 0.0
    x = np.fft.irfft(spectrum, n)
    return x * grms / np.sqrt(np.mean(x ** 2))


def _history(tones=TONES, seconds=60.0, seed=3304):
    """A floor of 1 grms with `tones` added, stored in m/s^2."""
    x = _floor(seconds, seed=seed)
    t = np.arange(len(x)) / FS
    for f, amplitude in tones:
        x = x + amplitude * np.sin(2.0 * np.pi * f * t + f)
    return TimeHistory(t, x[None] * G, response_dof=['1Z+'],
                       ordinate_dim=['acceleration'], ordinate_unit=['g'])


def _averaging(history, block, window='hann'):
    """Half-overlapped frames of `block` samples filling the record."""
    return Averaging(frame_length=block, overlap=0.5,
                     window=window).filled(len(history.abscissa), FS)


def _asd(history, block, window='hann'):
    averaging = _averaging(history, block, window)
    return history.compute_psds(averaging), averaging


@pytest.fixture(scope='module')
def composite():
    """The four-tone composite's ASD at 1 Hz and at 4 Hz lines, Hann."""
    history = _history()
    return {block: _asd(history, block) for block in (2048, 512)}


# ---- the window's leakage, against Harris ------------------------------

@pytest.mark.parametrize('window,bandwidth', [
    ('boxcar', 1.0), ('hann', 1.5), ('hamming', 1.36), ('blackman', 1.73),
    ('blackmanharris', 2.00), ('flattop', 3.77)])
def test_the_noise_bandwidth_is_the_window_s_own(window, bandwidth):
    assert noise_bandwidth(window) == pytest.approx(bandwidth, abs=0.005)


@pytest.mark.parametrize('window,loss_db', [
    ('boxcar', 3.92), ('hann', 1.42), ('hamming', 1.75), ('blackman', 1.10)])
def test_a_tone_midway_between_lines_loses_harris_s_scalloping(window,
                                                               loss_db):
    """The spike of a tone half a line off reads short by the window's
    scalloping loss, Harris's table."""
    lost = -10.0 * np.log10(leakage(window).scallop(0.5))
    assert lost == pytest.approx(loss_db, abs=0.02)


@pytest.mark.parametrize('window,half', [
    ('boxcar', 1), ('hann', 2), ('hamming', 2), ('blackman', 3),
    ('blackmanharris', 4), ('flattop', 5)])
def test_the_lines_summed_reach_across_the_main_lobe(window, half):
    """Out to the first null: Hann's lies on the second line."""
    assert leakage(window).half_width == half


@pytest.mark.parametrize('beta,half', [(0.0, 1), (14.0, 5), (40.0, 13)])
def test_a_kaiser_s_lobe_widens_with_its_beta(beta, half):
    """Its first null is at sqrt(1 + (beta/pi)^2) lines: 1, 4.6, 12.8."""
    assert leakage('kaiser', parameter=beta).half_width == half


def test_the_lines_of_any_placement_hold_the_whole_tone():
    """Parseval: every line's share, summed, is the tone, wherever it
    sits between lines; that is why the sum reads the amplitude."""
    kernel = leakage('hann', 256)
    for offset in (0.0, 0.17, 0.5):
        shares = kernel.share(np.arange(256) - 128 - offset)
        assert shares.sum() == pytest.approx(1.0, rel=1e-9)


def test_the_spike_height_goes_as_one_over_the_spacing():
    """The guide's arithmetic: 1.944 g peak is 1.89 g², which stands
    at 1.26 g²/Hz at 1 Hz lines under Hann and a quarter of that at
    4 Hz."""
    assert spike_height(1.944, 1.0) == pytest.approx(1.2598, abs=1e-4)
    assert spike_height(1.944, 4.0) == pytest.approx(1.2598 / 4, abs=1e-4)
    assert spike_height(1.944, 1.0, 'boxcar') == pytest.approx(1.8896,
                                                              abs=1e-4)
    # and between lines it reads short, by the scalloping
    assert spike_height(1.944, 1.0, offset=0.5) == pytest.approx(
        1.944 ** 2 / 2 / 1.5 * leakage('hann').scallop(0.5), rel=1e-9)
    with pytest.raises(ValueError, match='positive'):
        spike_height(1.0, 0.0)


# ---- the guide's composite ------------------------------------------

def test_the_same_tone_reads_the_same_at_either_spacing(composite):
    """The blade-passage tone at 1 Hz and at 4 Hz lines: its spike reads
    0.98 and 0.31 g²/Hz, its amplitude the same within a percent."""
    readings = {}
    for block, (asd, averaging) in composite.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ToneWarning)
            readings[block] = asd.read_tone(19.44, window=averaging)
    fine, coarse = readings[2048], readings[512]
    assert fine.spacing == 1.0 and coarse.spacing == 4.0
    assert fine.spike / G ** 2 == pytest.approx(0.98, abs=0.01)
    assert coarse.spike / G ** 2 == pytest.approx(0.31, abs=0.01)
    assert fine.amplitude == pytest.approx(coarse.amplitude, rel=0.01)
    for reading in (fine, coarse):
        assert reading.amplitude / G == pytest.approx(1.944, rel=0.02)
        assert reading.frequency == pytest.approx(19.44, abs=0.05)
        assert reading.noise_bandwidth == 1.5


def test_every_clear_tone_reads_within_two_percent_at_either_spacing(
        composite):
    for block, (asd, averaging) in composite.items():
        for f, amplitude in TONES[1:]:
            reading = asd.read_tone(f, window=averaging, warn=False)
            assert reading.amplitude / G == pytest.approx(amplitude,
                                                          rel=0.02), (block, f)
            assert not reading.uncertain, reading.notes


def test_the_spike_is_drawn_where_the_spacing_and_window_put_it(composite):
    """The height the reading says this tone stands at, over the floor,
    is the spike measured: within 2 % at both spacings."""
    for asd, averaging in composite.values():
        for f, _amplitude in TONES[1:]:
            reading = asd.read_tone(f, window=averaging, warn=False)
            assert reading.floor + reading.height == pytest.approx(
                reading.spike, rel=0.02)
            assert reading.height < reading.on_line + 1e-12


def test_a_tone_between_lines_is_warned_and_its_amplitude_is_not(
        composite):
    """19.44 Hz sits 0.44 of a line off at 1 Hz lines: its spike reads
    a fifth short, which is said; 38.88 Hz, 0.12 off, is not."""
    asd, averaging = composite[2048]
    with pytest.warns(ToneWarning, match='0.44 of a line'):
        reading = asd.read_tone(19.44, window=averaging)
    assert reading.between_lines and not reading.uncertain
    assert reading.offset == pytest.approx(0.44, abs=0.02)
    assert reading.spike / G ** 2 / (reading.on_line / G ** 2
                                     + reading.floor / G ** 2) \
        == pytest.approx(0.78, abs=0.02)
    near = asd.read_tone(38.88, window=averaging, warn=False)
    assert not near.between_lines
    assert not any('of a line' in note for note in near.notes)


def test_a_tone_at_the_floor_s_edge_is_called_uncertain(composite):
    """The 0.12 g tone at 4.86 Hz sits at the floor's lower edge, with no
    floor under it on one side: the guide's 'not a reading'."""
    asd, averaging = composite[2048]
    reading = asd.read_tone(4.86, window=averaging, warn=False)
    assert reading.uncertain
    assert any('does not run under this tone' in note
               for note in reading.notes)


def test_a_tone_gone_into_the_floor_is_said_to_be(composite):
    """At 4 Hz lines the 0.12 g tone stands no higher than the floor."""
    asd, averaging = composite[512]
    reading = asd.read_tone(4.86, window=averaging, snap=False, warn=False)
    assert reading.uncertain
    assert any('stands clear' in note for note in reading.notes)


def test_a_frequency_with_no_tone_reads_none(composite):
    asd, averaging = composite[2048]
    reading = asd.read_tone(300.0, window=averaging, warn=False)
    assert reading.amplitude / G < 0.1, 'the floor alone is no tone'
    assert reading.uncertain
    assert any('stands clear' in note for note in reading.notes)


def test_a_flat_top_s_lobe_swallowing_a_neighbor_is_said(composite):
    """Under a flat top at 4 Hz lines the main lobe is five lines either
    side, and the tones 19.44 Hz apart fall inside each other's."""
    history = _history()
    asd, averaging = _asd(history, 512, 'flattop')
    reading = asd.read_tone(38.88, window=averaging, warn=False)
    assert reading.uncertain
    assert any('another spike' in note or 'larger spike' in note
               for note in reading.notes)


# ---- single tones of known amplitude in a known floor ------------------

@pytest.mark.parametrize('window', ['hann', 'hamming', 'blackman',
                                    'blackmanharris', 'flattop', 'boxcar'])
@pytest.mark.parametrize('block', [2048, 512])
def test_a_known_tone_reads_back_within_two_percent(window, block):
    """1.5 g peak at 47.3 Hz, a third of a line off at 1 Hz and between
    lines at 4 Hz, on a 1 grms floor, under every window."""
    history = _history(tones=((47.3, 1.5),), seconds=40.0, seed=11)
    asd, averaging = _asd(history, block, window)
    reading = asd.read_tone(47.3, window=averaging, warn=False)
    assert reading.amplitude / G == pytest.approx(1.5, rel=0.02)
    assert reading.rms == pytest.approx(reading.amplitude / np.sqrt(2.0))


@pytest.mark.parametrize('block', [2048, 512])
def test_the_floor_under_a_small_tone_is_taken_away(block):
    """A 0.3 g tone on a 1 grms floor: at 4 Hz lines the floor in its
    four lines is two thirds as much mean square as the tone, and
    leaving it in would read 0.39 g. The reading is held to 5 %, since
    four lines of floor at 160 averages scatter by a few percent of the
    tone's mean square."""
    history = _history(tones=((131.0, 0.3),), seconds=40.0, seed=5)
    asd, averaging = _asd(history, block)
    reading = asd.read_tone(131.0, window=averaging, warn=False)
    assert reading.amplitude / G == pytest.approx(0.3, rel=0.05)
    kept = np.sqrt(2.0 * (reading.mean_square + reading.floor
                          * reading.spacing
                          * (reading.lines[1] - reading.lines[0] + 1)))
    assert kept / G > 0.33, 'the floor is a real share of these lines'


def test_snap_false_reads_at_the_frequency_given(composite):
    asd, averaging = composite[2048]
    reading = asd.read_tone(19.44, window=averaging, snap=False, warn=False)
    assert reading.frequency == pytest.approx(19.44, abs=1e-9)
    assert reading.amplitude / G == pytest.approx(1.944, rel=0.02)


def test_the_cursor_may_stand_beside_the_spike(composite):
    """A frequency a line or so off the tone still reads the tone."""
    asd, averaging = composite[2048]
    reading = asd.read_tone(18.2, window=averaging, warn=False)
    assert reading.frequency == pytest.approx(19.44, abs=0.05)


def test_read_tone_works_on_bare_arrays():
    """The reading is of any evenly spaced density, not only a Psd."""
    f = np.arange(0.0, 200.0, 1.0)
    density = np.full(len(f), 0.002)
    shares = leakage('hann').share(np.arange(len(f)) - 60.25)
    density = density + 2.0 ** 2 / 2.0 * shares
    reading = read_tone(f, density, 60.0, window='hann', warn=False)
    assert reading.amplitude == pytest.approx(2.0, rel=1e-3)
    assert reading.frequency == pytest.approx(60.25, abs=0.01)
    assert reading.floor == pytest.approx(0.002, rel=1e-4)


def test_what_cannot_be_read_is_refused(composite):
    asd, averaging = composite[2048]
    with pytest.raises(ValueError, match='banded'):
        asd.to_octave(3).read_tone(19.44)
    with pytest.raises(ValueError, match='outside the spectrum'):
        asd.read_tone(5000.0, window=averaging)
    with pytest.raises(ValueError, match='evenly spaced'):
        read_tone([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0],
                  np.ones(9), 8.0, warn=False)
    with pytest.raises(ValueError, match='frequencies for'):
        read_tone(np.arange(10.0), np.ones(9), 5.0)


def test_the_reading_describes_itself_in_display_units(composite):
    asd, averaging = composite[2048]
    reading = asd.read_tone(19.44, window=averaging, warn=False)
    text = reading.describe('g', 1.0 / G)
    assert text.startswith('19.44 Hz: 1.94 g peak; its spike 0.978 g²/Hz')
    assert 'is not the tone' in text


# ---- the project records the window ------------------------------------

def test_the_project_gives_the_averaging_a_spectrum_was_made_with(tmp_path):
    history = _history(seconds=8.0)
    project = visualdynamics.Project()
    project.add('Run', history)
    whole = project.compute_psds('Run')
    assert project.averaging_of(whole).window == 'boxcar', \
        'no averaging set: each record whole, unwindowed'
    history.averaging = _averaging(history, 512, 'flattop')
    framed = project.compute_psds('Run')
    assert project.averaging_of(framed) == history.averaging
    # and it is the averaging it was made with, not the record's now
    history.averaging = _averaging(history, 1024, 'hann')
    assert project.averaging_of(framed).window == 'flattop'
    path = tmp_path / 'tones.vdyn'
    project.save(path)
    reopened = visualdynamics.Project.open(path)
    assert reopened.averaging_of(framed).frame_length == 512
    imported = Psd(np.arange(0.0, 64.0), np.ones((1, 64)),
                   response_dof=['1Z+'],
                   ordinate_dim='acceleration**2/frequency')
    project.add('Imported', imported)
    assert project.averaging_of('Imported') is None


# ---- the plot ----------------------------------------------------------

def test_the_plot_draws_each_tone_at_its_spike(composite, tmp_path, qt_app):
    from visualdynamics.plot import plot_tone_readout

    asd, averaging = composite[2048]
    path = tmp_path / 'tones.png'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ToneWarning)
        written = plot_tone_readout(asd, [19.44, 58.32], window=averaging,
                                    band=(2.0, 200.0), path=str(path),
                                    show=False)
    assert written == str(path) and path.stat().st_size > 10_000


def test_the_overlay_stands_at_the_height_in_display_units(composite,
                                                           qt_app):
    import pyqtgraph as pg

    from visualdynamics.plot.tone_readout import ToneOverlay, label_text

    asd, averaging = composite[2048]
    reading = asd.read_tone(19.44, window=averaging, warn=False)
    plot = pg.PlotItem()
    overlay = ToneOverlay(plot)
    overlay.show(reading, asd.abscissa, scale=1.0 / G, unit='g')
    top = overlay.top.yData[0]
    assert top == pytest.approx((reading.floor + reading.height) / G ** 2)
    assert overlay.stem.yData[0] == pytest.approx(reading.floor / G ** 2)
    assert list(overlay.floor.xData) == [18.0, 21.0], \
        'the floor line spans the lines summed'
    text = label_text(reading, 'g', 1.0 / G)
    assert text.splitlines() == ['1.94 g peak', 'spike 0.978 g²/Hz',
                                 'between lines']
    overlay.clear()
    assert not overlay.top.isVisible()


# ---- the notes' units, what is refused, and the window's parameter -----

def test_the_notes_name_their_densities_in_the_units_shown(composite):
    """The no-tone note gives the floor: in SI from a script, as the
    spectrum is stored, and in g²/Hz when the pane shows g, the same
    number the floor row shows."""
    asd, averaging = composite[2048]
    reading = asd.read_tone(300.0, window=averaging, warn=False)
    note = next(note for note in reading.notes if 'stands clear' in note)
    assert f'({reading.floor:.3g} (m/s²)²/Hz)' in note
    shown = next(note for note in reading.notes_in('g', 1.0 / G)
                 if 'stands clear' in note)
    assert f'({reading.floor / G ** 2:.3g} g²/Hz)' in shown
    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter('always', ToneWarning)
        asd.read_tone(300.0, window=averaging)
    assert any(f'({reading.floor:.3g} (m/s²)²/Hz)' in str(w.message)
               for w in warned), 'the warning says what the note says'


def test_the_floor_s_two_sides_are_written_in_the_units_shown():
    """A tone at a step in the floor: both sides named in g²/Hz."""
    f = np.arange(0.0, 200.0, 1.0)
    density = np.where(f < 60.0, 0.001, 0.004) * G ** 2
    density = density + (2.0 * G) ** 2 / 2.0 * leakage('hann').share(
        np.arange(len(f)) - 60.0)
    reading = read_tone(f, density, 60.0, unit='m/s²', warn=False)
    note = next(note for note in reading.notes_in('g', 1.0 / G)
                if 'differs either side' in note)
    assert '(0.001 g²/Hz below, 0.004 g²/Hz above)' in note
    assert '(m/s²)²/Hz below' in next(
        note for note in reading.notes if 'differs either side' in note)


def test_a_compound_unit_is_squared_whole():
    from visualdynamics.core.tone_readout import density_label

    assert density_label('g') == 'g²/Hz'
    assert density_label('m/s²') == '(m/s²)²/Hz'
    assert density_label('') == '/Hz'
    assert density_label('µm') == 'µm²/Hz'


def test_a_cross_spectrum_is_refused():
    """A CPSD's off-diagonal record is not a density of one signal,
    though its values are real here; its diagonal is read."""
    f = np.arange(0.0, 64.0)
    cpsd = Psd(f, np.ones((2, 64)), response_dof=['1Z+', '1Z+'],
               reference_dof=['1Z+', '2Z+'],
               ordinate_dim=['acceleration**2/frequency',
                             'acceleration*acceleration/frequency'])
    assert cpsd.can_read_tone(0) and not cpsd.can_read_tone(1)
    with pytest.raises(ValueError, match='cross spectrum'):
        cpsd.read_tone(20.0, 1, warn=False)
    cpsd.read_tone(20.0, 0, warn=False)


def test_lines_without_a_number_are_said_or_refused():
    f = np.arange(0.0, 200.0, 1.0)
    density = 0.002 + 2.0 ** 2 / 2.0 * leakage('hann').share(
        np.arange(len(f)) - 60.25)
    holed = density.copy()
    holed[64] = np.nan
    reading = read_tone(f, holed, 60.0, warn=False)
    assert reading.uncertain
    assert any('1 of the lines read' in note for note in reading.notes)
    far = density.copy()
    far[180] = np.nan
    clear = read_tone(f, far, 60.0, warn=False)
    assert not clear.uncertain and not clear.notes
    with pytest.raises(ValueError, match='holds a number'):
        read_tone(f, np.full(len(f), np.nan), 60.0, warn=False)


def test_a_kaiser_s_beta_reaches_the_reading(composite):
    asd, _averaging = composite[2048]
    six = asd.read_tone(19.44, window='kaiser', parameter=6.0, warn=False)
    fourteen = asd.read_tone(19.44, window='kaiser', warn=False)
    assert six.noise_bandwidth == pytest.approx(
        noise_bandwidth('kaiser', parameter=6.0))
    assert six.noise_bandwidth < fourteen.noise_bandwidth - 0.2


def test_the_plot_s_title_comes_from_the_spectrum_with_no_tones(composite,
                                                                qt_app):
    from visualdynamics.plot import plot_tone_readout

    asd, averaging = composite[512]
    pane = plot_tone_readout(asd, [], window=averaging, show=False)
    try:
        assert pane.windowTitle() == '1Z+: tones at 4 Hz lines, hann window'
    finally:
        pane.close()
        pane.deleteLater()
        qt_app.processEvents()


def test_the_display_units_of_a_record(composite):
    from visualdynamics.plot.tone_readout import display_units
    from visualdynamics.units import DEFAULT_SYSTEM, SI

    asd, _averaging = composite[2048]
    scale, dense, unit = display_units(asd, 0, DEFAULT_SYSTEM)
    assert unit == 'g' and scale == pytest.approx(1.0 / G)
    assert dense == pytest.approx(1.0 / G ** 2)
    assert display_units(asd, 0, SI) == (1.0, 1.0, 'm/s²')
    cross = Psd(np.arange(0.0, 64.0), np.ones((1, 64)),
                response_dof=['1Z+'], reference_dof=['2X+'],
                ordinate_dim='acceleration*force/frequency')
    assert display_units(cross, 0, DEFAULT_SYSTEM) == (1.0, 1.0, '')
