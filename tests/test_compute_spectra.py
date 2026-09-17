"""Compute Spectra: averaged spectra from multi-average time data.

Rattlesnake writes the time frames of a burst random modal test but not
their spectra; right-clicking the time history fills that gap. The
convention is sdynpy's `TimeHistoryArray.fft` with frames, on Brandon's
explicit ask: a raw single-sided rfft (norm='backward' — a sine of
amplitude A on a bin reads A*N/2) and a *complex mean* across frames,
so phase is preserved and random-phase content averages toward zero.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from visualdynamics.core.data import Spectrum, TimeHistory
from visualdynamics.core.report import Report, insert_options
from visualdynamics.report import render_html

N = 256          # 1 s at 256 Hz: df = 1 Hz, so test tones sit on bins


def _burst_time(frames=4, random_phase=False):
    """Two channels of exact-bin sines: 1X+ is 1 m/s^2 at 16 Hz, 2X+ is
    2 N at 24 Hz — every frame in phase, or each at a random phase."""
    t = np.arange(N) / N
    rng = np.random.default_rng(3)
    rows, dofs, blocks = [], [], []
    for dof, amplitude, f0 in (('1X+', 1.0, 16), ('2X+', 2.0, 24)):
        for k in range(frames):
            phase = rng.uniform(0, 2 * np.pi) if random_phase else 0.0
            rows.append(amplitude * np.sin(2 * np.pi * f0 * t + phase))
            dofs.append(dof)
            blocks.append(f'avg {k + 1}')
    time = TimeHistory(t, np.array(rows), response_dof=dofs, block=blocks)
    time.define_units({i: 'm/s^2' for i in range(frames)}
                      | {i + frames: 'N' for i in range(frames)})
    return time


def test_spectra_are_the_complex_mean_of_the_frames():
    time = _burst_time()
    spectra = time.compute_spectra()
    assert isinstance(spectra, Spectrum)
    assert spectra.num_records == 2, 'one record per channel, not per frame'
    assert spectra.response_dof == ['1X+', '2X+']
    assert spectra.ordinate_dim == ['acceleration', 'force']
    # in-phase frames survive the average at the raw rfft's own scale:
    # a sine of amplitude A on a bin reads A*N/2
    magnitudes = np.abs(spectra.ordinate)
    assert magnitudes[0, 16] == pytest.approx(1.0 * N / 2, rel=1e-9)
    assert magnitudes[1, 24] == pytest.approx(2.0 * N / 2, rel=1e-9)
    # and the phase rides along: rfft of a sine is purely -i at its bin
    assert spectra.ordinate[0, 16].imag == pytest.approx(-N / 2, rel=1e-9)
    off = magnitudes.copy()
    off[0, 16] = off[1, 24] = 0.0
    assert off.max() < 1e-7, 'exact-bin tones leak nowhere'
    assert float(spectra.abscissa[16]) == pytest.approx(16.0)


def test_random_phase_frames_average_toward_zero():
    """The behavior Brandon asked for by name: a complex average zeroes
    content whose phase is random frame to frame, exactly as sdynpy's
    fft documents. An RMS-magnitude average would keep it at full
    amplitude and fail this."""
    single = np.abs(_burst_time(frames=1).compute_spectra().ordinate)
    averaged = np.abs(
        _burst_time(frames=32, random_phase=True).compute_spectra().ordinate)
    assert averaged[0, 16] < 0.3 * single[0, 16]
    assert averaged[1, 24] < 0.3 * single[1, 24]


def test_the_spectrum_is_the_raw_single_sided_rfft():
    """The convention, pinned: sdynpy's. A sine with a peak-to-peak
    amplitude of 5 (2.5 peak) reads 2.5*N/2 at its own frequency —
    numpy's unscaled rfft, single-sided, rectangular window."""
    for n in (256, 501):                    # even and odd frame lengths
        t = np.arange(n) / n                # df = 1 Hz, 40 Hz on a bin
        wave = 2.5 * np.sin(2 * np.pi * 40 * t)     # 5 peak to peak
        assert wave.max() - wave.min() == pytest.approx(5.0, rel=1e-3)
        time = TimeHistory(t, wave[None, :], response_dof=['1X+'])
        spectra = time.compute_spectra()
        bin40 = int(np.argmin(np.abs(spectra.abscissa - 40.0)))
        assert float(spectra.abscissa[bin40]) == pytest.approx(40.0)
        assert np.abs(spectra.ordinate[0, bin40]) == pytest.approx(
            2.5 * n / 2, rel=1e-9)


def test_psds_are_the_welch_average_of_the_frames():
    """One-sided auto-power density, rectangular window, powers
    averaged — scipy's Welch with each average as its own frame, and
    Parseval holds: the PSD integrates to the signal's mean square."""
    import pytest

    # a dev extra, so CI has it; skipped rather than errored for anyone
    # running the suite without them
    welch = pytest.importorskip('scipy.signal').welch

    from visualdynamics.core.data import Psd

    time = _burst_time(frames=8, random_phase=True)
    psds = time.compute_psds()
    assert isinstance(psds, Psd)
    assert psds.num_records == 2
    assert psds.ordinate_dim == ['acceleration**2/frequency',
                                 'force**2/frequency']
    # random phase costs nothing: power is phase-insensitive, so the
    # tone keeps its full A^2/(2 df) in its bin
    assert np.abs(psds.ordinate[0, 16]) == pytest.approx(0.5, rel=1e-9)
    assert np.abs(psds.ordinate[1, 24]) == pytest.approx(2.0, rel=1e-9)
    # scipy agrees, frame for frame
    rows = [i for i, d in enumerate(time.response_dof) if d == '1X+']
    signal = np.concatenate([time.ordinate[i] for i in rows])
    _freqs, reference = welch(signal, fs=N, nperseg=N, noverlap=0,
                             window='boxcar', detrend=False)
    assert np.allclose(np.abs(psds.ordinate[0]), reference, rtol=1e-9)
    # Parseval: integrated density equals the mean square of the frames
    mean_square = np.mean([time.ordinate[i] ** 2 for i in rows])
    df = float(psds.abscissa[1] - psds.abscissa[0])
    assert np.sum(np.abs(psds.ordinate[0])) * df == pytest.approx(
        mean_square, rel=1e-9)


def test_uneven_sampling_is_refused():
    time = TimeHistory(np.array([0.0, 0.1, 0.3]), np.ones((1, 3)),
                       response_dof=['1X+'])
    with pytest.raises(ValueError, match='evenly spaced'):
        time.compute_spectra()


def test_the_context_action_adds_the_spectra_object(window, pump):
    window.add_object('Time', _burst_time())
    window.tree.setCurrentItem(window._item_for_object('Time'))
    window.compute_spectra()
    spectra = window.objects['Time Spectra']
    assert isinstance(spectra, Spectrum)
    assert spectra.num_records == 2
    assert 'averaged over 4 frames' in window.statusBar().currentMessage()
    assert window.linked_group('Time Spectra') == ['Time', 'Time Spectra'], (
        'what it was computed from is not a guess — linked by default')


def test_the_bar_offers_the_acts_an_object_can_take(window, pump, survey):
    """Every act lives on the bar (Brandon, 2026-09-04): the tree's
    column carries no calculator any more, and the right-click no
    computations. A time history offers Integrate, an FRF Fit Modal
    Model, a spectrum nothing."""
    _shapes, frfs = survey
    window.add_object('Time', _burst_time())
    window.add_object('FRF', frfs)
    window.add_object('Spec', window.objects['Time'].compute_spectra())
    for name in ('Time', 'FRF', 'Spec'):
        assert window._item_for_object(name).icon(1).isNull(), name
    assert [label for _v, label, *_rest in window.acts_for(['Time'])] == \
        ['Integrate'], \
        'record-level transforms only: every verb a view '\
        'parameterizes lives on that view (Brandon, 2026-08-28)'
    assert [label for _v, label, *_rest in window.acts_for(['FRF'])] == \
        ['Fit Modal Model']
    assert window.acts_for(['Spec']) == [], \
        'nothing computes from a spectrum, so no acts'
def test_spectra_blocks_split_by_quantity_like_time_blocks(window, pump):
    time = _burst_time()
    spectra = time.compute_spectra()
    objects = {'Time': time, 'Spec': spectra}
    options = insert_options(objects)
    assert ('spectrum:force', 'Spectra (Force)') in options
    assert ('spectrum:acceleration', 'Spectra (Acceleration)') in options
    report = Report('R', [{'kind': 'plot', 'source': 'Spec',
                           'mode': 'curves', 'select': 'dim:force',
                           'caption': ''}])
    html = render_html(report, objects)
    payload = json.loads(
        html.split('type="application/json">')[1].split('</script>')[0])
    block = payload['blocks'][0]
    assert [c['label'] for c in block['curves']] == ['2X+']
    assert block['logy'] is True, 'a spectrum reads in log magnitude'
    # the axis wears the units of the records actually plotted — the
    # acceleration channel sits first in the object, and a force block
    # labeled with its units is exactly the bug this pins down
    from visualdynamics.units import DEFAULT_SYSTEM
    assert block['ylabel'] == DEFAULT_SYSTEM.label_text('force')
    # the inserts bind to the right kind of data: a Spectra insert
    # never grabs the time history, nor the other way around
    window.add_object('Time', time)
    window.add_object('Time Spectra', spectra)
    name = window.generate_report('empty')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    editor = window.report_editor
    editor._operate({'op': 'insert', 'at': 0, 'kind': 'spectrum:force'})
    editor._operate({'op': 'insert', 'at': 0, 'kind': 'time:force'})
    built = window.objects[name].blocks
    assert built[0]['source'] == 'Time'
    assert built[1]['source'] == 'Time Spectra'
    assert built[1]['select'] == 'dim:force'


def test_psd_blocks_split_by_quantity_too(window, pump):
    time = _burst_time()
    psds = time.compute_psds()
    objects = {'Time': time, 'PSD': psds}
    options = insert_options(objects)
    assert ('psd:force', 'PSD (Force)') in options
    assert ('psd:acceleration', 'PSD (Acceleration)') in options
    report = Report('R', [{'kind': 'plot', 'source': 'PSD',
                           'mode': 'curves', 'select': 'dim:force',
                           'caption': ''}])
    payload = json.loads(render_html(report, objects).split(
        'type="application/json">')[1].split('</script>')[0])
    block = payload['blocks'][0]
    assert [c['label'] for c in block['curves']] == ['2X+'], (
        'the **2/frequency dimension still filters by its base quantity')
    window.add_object('Time', time)
    window.add_object('Time PSDs', psds)
    name = window.generate_report('empty')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    window.report_editor._operate({'op': 'insert', 'at': 0,
                                   'kind': 'psd:acceleration'})
    built = window.objects[name].blocks[0]
    assert built['source'] == 'Time PSDs'
    assert built['select'] == 'dim:acceleration'


def test_the_context_action_adds_the_psds_object(window, pump):
    from visualdynamics.core.data import Psd

    window.add_object('Time', _burst_time())
    window.tree.setCurrentItem(window._item_for_object('Time'))
    window.compute_psds()
    psds = window.objects['Time PSDs']
    assert isinstance(psds, Psd)
    assert psds.num_records == 2
    assert 'averaged over 4 frames' in window.statusBar().currentMessage()
    assert window.linked_group('Time PSDs') == ['Time', 'Time PSDs']


def _two_channel_time(frames=6):
    """The same signal on two channels, several frames of each."""
    import numpy as np

    from visualdynamics.core.data import TimeHistory

    fs, n = 1000.0, 512
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    rows, dofs = [], []
    for dof in ('101X+', '102X+'):
        for _ in range(frames):
            rows.append(np.sin(2 * np.pi * 50.0 * t)
                        + 0.1 * rng.standard_normal(n))
            dofs.append(dof)
    return TimeHistory(abscissa=t, ordinate=np.array(rows),
                       response_dof=dofs,
                       ordinate_dim=['acceleration'] * len(rows))


def test_cpsds_are_the_whole_matrix_not_the_diagonal():
    history = _two_channel_time()
    cpsds = history.compute_cpsds()
    assert cpsds.num_records == 4, 'two channels make a 2x2'
    assert list(zip(cpsds.response_dof, cpsds.reference_dof)) == [
        ('101X+', '101X+'), ('101X+', '102X+'),
        ('102X+', '101X+'), ('102X+', '102X+')]


def test_the_diagonal_is_exactly_compute_psds():
    """Two ways of saying the same thing must say it identically, or
    which one you asked for changes the answer."""
    import numpy as np

    history = _two_channel_time()
    psds, cpsds = history.compute_psds(), history.compute_cpsds()
    for k, (response, reference) in enumerate(zip(cpsds.response_dof,
                                                  cpsds.reference_dof)):
        if response != reference:
            continue
        row = list(psds.response_dof).index(response)
        assert np.allclose(cpsds.ordinate[k], psds.ordinate[row])


def test_an_autospectrum_is_stored_real():
    """A channel against itself is a magnitude squared. Held complex it
    carried an imaginary part of exactly nothing at twice the size, and
    every caller asking "is this complex?" — the component selector
    above all — got the answer of the type rather than of the data."""
    psds = _two_channel_time().compute_psds()
    assert psds.ordinate.dtype == np.float64


def test_a_cross_spectrum_keeps_its_phase():
    """The other half of the same rule: demoting a CPSD would discard
    the phase between channels, which is most of what it is for."""
    cpsds = _two_channel_time().compute_cpsds()
    assert cpsds.ordinate.dtype == np.complex128
    cross = [k for k, (r, f) in enumerate(zip(cpsds.response_dof,
                                              cpsds.reference_dof)) if r != f]
    assert np.abs(cpsds.ordinate[cross].imag).max() > 0.0


def test_a_cpsd_of_one_channel_is_real():
    """The rule reads the values, not the class: a 1x1 CPSD is its own
    diagonal, so there is no cross term and nothing to hold complex."""
    history = _two_channel_time()
    keep = [k for k, dof in enumerate(history.response_dof) if dof == '101X+']
    one = TimeHistory(history.abscissa, history.ordinate[keep],
                      response_dof=[history.response_dof[k] for k in keep])
    assert one.compute_cpsds().ordinate.dtype == np.float64


def test_the_matrix_is_hermitian():
    """Gxy is the conjugate of Gyx — the property that says the cross
    terms are a cross-spectrum and not two unrelated numbers."""
    import numpy as np

    cpsds = _two_channel_time().compute_cpsds()
    assert np.allclose(cpsds.ordinate[1], np.conj(cpsds.ordinate[2]))


def test_a_cross_term_carries_the_product_dimension():
    from visualdynamics.core.data import TimeHistory

    history = _two_channel_time()
    history.ordinate_dim = ['acceleration'] * 6 + ['force'] * 6
    cpsds = TimeHistory(
        abscissa=history.abscissa, ordinate=history.ordinate,
        response_dof=history.response_dof,
        ordinate_dim=history.ordinate_dim).compute_cpsds()
    dims = dict(zip(zip(cpsds.response_dof, cpsds.reference_dof),
                    cpsds.ordinate_dim))
    assert dims[('101X+', '101X+')] == 'acceleration**2/frequency'
    assert dims[('101X+', '102X+')] == 'acceleration*force/frequency'


def test_the_project_keeps_and_links_them(window, pump):
    window.add_object('Time', _two_channel_time())
    added = window.project.compute_cpsds('Time')
    assert added == 'Time CPSDs'
    assert window.objects[added].num_records == 4
    assert set(window.project.group_of('Time')) >= {'Time', added}


def test_the_calculator_computes_them(window, pump):
    window.add_object('Time', _two_channel_time())
    window.tree.setCurrentItem(window._item_for_object('Time'))
    pump()
    window.compute_cpsds()
    pump()
    assert 'Time CPSDs' in window.objects
    assert '2x2 cross-spectral matrix' in window.statusBar().currentMessage()


def test_the_gui_status_counts_the_windows_frames(window, pump):
    """A continuous recording cut by the averaging window was announced
    as 'averaged over 1 frame' — the record-count ratio, which is the
    frame count only for a stack of captures. The count and the span
    are the one glance that says whether the window the user dragged
    is the window that was used."""
    from visualdynamics.core.averaging import Averaging

    t = np.arange(4096) / 256.0
    rng = np.random.default_rng(7)
    time = TimeHistory(t, rng.standard_normal((2, 4096)),
                       response_dof=['1X+', '2X+'])
    time.averaging = Averaging(frame_length=512, overlap=0.5,
                               window='hann', frames=5, start=2.0)
    window.add_object('Run', time)
    window.tree.setCurrentItem(window._item_for_object('Run'))
    window.compute_psds()
    message = window.statusBar().currentMessage()
    # 5 frames of 512 at half overlap: 2.0 s + 1536 samples at 256 Hz
    assert 'averaged over 5 frames (2.0–8.0 s)' in message, message
