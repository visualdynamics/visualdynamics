"""Generated reference signals (`core.signals`), read back with the
package's own estimators under the default window.

Brandon, 2026-10-24: random, burst random and an impulse at the
reference DOFs first, the response through a system after. Every claim
of the module docstring is a number here: a PSD that comes back as its
target, Gaussian samples, coherence at chance between DOFs drawn apart,
a correlated pair's cross term in magnitude and phase, frames with no
seam, bursts that are quiet when they say, periods that repeat exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.core import signals
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.classical_pulse import ClassicalPulse
from visualdynamics.core.data import Specification
from visualdynamics.core.sine import LOG, SineSweepSpecification, SineTone

FS = 8192.0
BAND = (20.0, 2000.0)


def _flat(dofs=('1X+',), level=1.0):
    return signals.flat_spectrum(list(dofs), quantity='force', band=BAND,
                                 level=level)


def _read(history, frame=4096):
    history.averaging = Averaging(frame_length=frame, overlap=0.5,
                                  window='hann').filled(
        history.ordinate.shape[1], FS)
    return history


def _inside(spectrum, low=60.0, high=1900.0):
    return (spectrum.abscissa > low) & (spectrum.abscissa < high)


CROSS = 0.6 * np.exp(1j * np.pi / 3)


def _pair(level=1.0, cross=CROSS):
    """A 2x2 force target: autos `level`, the record (1X+, 2X+) `cross`."""
    f = np.array(BAND)
    return Specification(
        f, np.array([[level] * 2, [cross] * 2, [level] * 2]),
        response_dof=['1X+', '1X+', '2X+'],
        reference_dof=['1X+', '2X+', '2X+'],
        ordinate_dim='force**2/frequency')


# ---- random: the target, the samples, the record ----------------------------

def test_random_comes_back_as_its_target():
    history = signals.random_history(_flat(level=2.0), duration=30.0,
                                     sample_rate=FS, seed=1)
    psd = _read(history).compute_psds()
    inside = _inside(psd)
    assert np.real(psd.ordinate[0, inside]).mean() == pytest.approx(2.0,
                                                                    rel=0.02)
    rms = np.sqrt(2.0 * (BAND[1] - BAND[0]))
    assert history.ordinate[0].std() == pytest.approx(rms, rel=0.01)
    outside = (psd.abscissa > 2200.0) | (psd.abscissa < 10.0)
    assert np.real(psd.ordinate[0, outside]).max() < 0.01, \
        'nothing drawn outside the band'


def test_random_samples_are_gaussian():
    x = signals.random_history(_flat(), duration=30.0, sample_rate=FS,
                               seed=2).ordinate[0]
    kurtosis = np.mean((x - x.mean()) ** 4) / np.var(x) ** 2
    assert kurtosis == pytest.approx(3.0, abs=0.08)
    skew = np.mean((x - x.mean()) ** 3) / np.std(x) ** 3
    assert abs(skew) < 0.05


def test_random_has_no_seam_and_holds_its_level():
    """Overlapped half-cycle sine windows square to one at every sample:
    the variance at each position within a hop is the same, the first
    third of the record as loud as the last."""
    history = signals.random_history(_flat(), duration=60.0, sample_rate=FS,
                                     frame_length=1024, seed=3)
    x = history.ordinate[0]
    hop = 512
    by_position = x[:len(x) // hop * hop].reshape(-1, hop).var(axis=0)
    # where one window is at its peak and the other at zero, against
    # where both are at sin(pi/4): a Hann window, whose squares do not
    # sum to one, reads 2 here
    edge = np.concatenate([by_position[:16], by_position[-16:]]).mean()
    middle = by_position[hop // 2 - 16:hop // 2 + 16].mean()
    assert edge / middle == pytest.approx(1.0, abs=0.06), \
        'the frame edge is as loud as its middle'
    assert by_position.max() / by_position.min() < 1.6
    thirds = np.array_split(x, 3)
    assert thirds[0].std() / thirds[-1].std() == pytest.approx(1.0, abs=0.03)
    assert len(x) == round(60.0 * FS)


def test_a_breakpoint_slope_is_drawn_as_a_power_law():
    """+6 dB/oct from 20 to 160 Hz, flat to 1600, read log-log between:
    at the slope's geometric middle the power law reads 0.125 where a
    straight line between the breakpoints would read 0.27."""
    spec = Specification([20.0, 160.0, 1600.0], [[1.0 / 64, 1.0, 1.0]],
                         response_dof='1X+', ordinate_dim='force**2/frequency')
    psd = _read(signals.random_history(spec, duration=40.0, sample_rate=FS,
                                       seed=4)).compute_psds()

    def near(f0):
        return np.real(psd.ordinate[0][(psd.abscissa > f0 * 0.95)
                                       & (psd.abscissa < f0 * 1.05)]).mean()

    assert near(56.57) == pytest.approx(0.125, rel=0.08)
    assert near(800.0) == pytest.approx(1.0, rel=0.04)


def test_a_controllers_lines_are_read_per_bin():
    """Bins 10 Hz wide that alternate between 1 and 4, from 0 Hz: each
    is read over its own width, the DC line is left at zero."""
    f = np.arange(0.0, 1010.0, 10.0)
    level = np.where((f // 10) % 2 == 0, 1.0, 4.0)
    spec = Specification(f, level[None], response_dof='1X+',
                         ordinate_dim='force**2/frequency')
    # how an importer marks a controller's target (`reading_of`)
    spec.interpolation = Specification.reading_of(f)
    assert spec.interpolation == 'bin'
    history = signals.random_history(spec, duration=60.0, sample_rate=FS,
                                     frame_length=8192, seed=5)
    psd = _read(history, frame=8192).compute_psds()
    centers = f[(f >= 100) & (f <= 900)]
    read = np.array([np.real(psd.ordinate[0][np.abs(psd.abscissa - c) <= 2.0])
                     .mean() for c in centers])
    want = np.where((centers // 10) % 2 == 0, 1.0, 4.0)
    assert np.corrcoef(read, want)[0, 1] > 0.9
    assert read[want == 4.0].mean() / read[want == 1.0].mean() > 2.5
    # a period's mean is its DC line exactly: none drawn, though the
    # first bin is written from 0 Hz
    period = signals.periodic_random_history(spec, periods=1, sample_rate=FS,
                                             frame_length=8192, seed=6)
    x = period.ordinate[0]
    assert abs(x.mean()) < 1e-12 * x.std(), 'no DC drawn'


def test_the_same_seed_gives_the_same_record():
    a = signals.random_history(_flat(), duration=2.0, sample_rate=FS, seed=7)
    b = signals.random_history(_flat(), duration=2.0, sample_rate=FS, seed=7)
    c = signals.random_history(_flat(), duration=2.0, sample_rate=FS, seed=8)
    assert np.array_equal(a.ordinate, b.ordinate)
    assert not np.allclose(a.ordinate, c.ordinate)


# ---- several DOFs: apart, or together ---------------------------------------

def test_dofs_drawn_apart_are_coherent_only_by_chance():
    history = _read(signals.random_history(_flat(('1X+', '2X+')),
                                           duration=40.0, sample_rate=FS,
                                           seed=9))
    coherence = history.compute_cpsds().coherence()
    inside = _inside(coherence)
    row = next(i for i in range(coherence.num_records)
               if coherence.record_pair(i) == ('1X+', '2X+'))
    averages = history.averaging.frames
    assert np.real(coherence.ordinate[row, inside]).mean() < 3.0 / averages


def test_a_correlated_pair_comes_back_with_its_cross_term():
    """The cross term read back through compute_cpsds, magnitude and
    phase, the record (1X+, 2X+) as stored and its mirror the conjugate:
    the one check a real-valued target cannot pass by accident."""
    history = _read(signals.random_history(_pair(), duration=40.0,
                                           sample_rate=FS, seed=10))
    cpsd = history.compute_cpsds()
    inside = _inside(cpsd)
    read = {cpsd.record_pair(i): cpsd.ordinate[i, inside].mean()
            for i in range(cpsd.num_records)}
    assert abs(read[('1X+', '2X+')]) == pytest.approx(0.6, rel=0.03)
    assert np.degrees(np.angle(read[('1X+', '2X+')])) \
        == pytest.approx(60.0, abs=1.5)
    assert read[('2X+', '1X+')] == pytest.approx(np.conj(read[('1X+', '2X+')]),
                                                 rel=0.03)
    assert np.real(read[('1X+', '1X+')]) == pytest.approx(1.0, rel=0.03)


def test_correlated_false_draws_each_dof_alone():
    history = _read(signals.random_history(_pair(), duration=40.0,
                                           sample_rate=FS, correlated=False,
                                           seed=11))
    coherence = history.compute_cpsds().coherence()
    row = next(i for i in range(coherence.num_records)
               if coherence.record_pair(i) == ('1X+', '2X+'))
    assert np.real(coherence.ordinate[row, _inside(coherence)]).mean() < 0.05


def test_a_fully_coherent_pair_is_drawn_and_an_impossible_one_refused():
    together = signals.random_history(_pair(cross=1.0), duration=4.0,
                                      sample_rate=FS, seed=12)
    assert np.allclose(together.ordinate[0], together.ordinate[1],
                       atol=1e-9 * np.abs(together.ordinate).max()), \
        'coherence one at zero phase: the same signal twice'
    with pytest.raises(ValueError, match='not a matrix any set of signals'):
        signals.random_history(_pair(cross=1.3), duration=1.0, sample_rate=FS)


def test_a_target_is_refused_where_the_realizability_check_fails_it():
    """Two channels 80 dB apart with a coherence of 1.01 between them.
    The raw matrix's smallest eigenvalue is about 2e-10 of its largest,
    small enough to pass for rounding against the loud channel, but the
    coherence matrix's is -0.01: no set of signals has it, the
    realizability check says so, and the generator refuses it too."""
    from visualdynamics.core.matrix_checks import read_realizability

    quiet = 1e-8
    target = Specification(
        np.array(BAND), np.array([[1.0] * 2, [1.01 * np.sqrt(quiet)] * 2,
                                  [quiet] * 2]),
        response_dof=['1X+', '1X+', '2X+'],
        reference_dof=['1X+', '2X+', '2X+'],
        ordinate_dim='force**2/frequency')
    assert not read_realizability(target).realizable
    with pytest.raises(ValueError, match='not a matrix any set of signals'):
        signals.random_history(target, duration=1.0, sample_rate=FS)


def test_dofs_are_drawn_in_the_order_asked():
    history = signals.random_history(_pair(), duration=1.0, sample_rate=FS,
                                     dofs=['2X+'], seed=13)
    assert history.response_dof == ['2X+']
    assert history.ordinate_dim == ['force']
    with pytest.raises(ValueError, match='no autospectrum at 3X\\+'):
        signals.random_history(_pair(), duration=1.0, sample_rate=FS,
                               dofs=['3X+'])


# ---- what is refused --------------------------------------------------------

def test_what_cannot_be_drawn_is_refused():
    with pytest.raises(ValueError, match='Nyquist'):
        signals.random_history(_flat(), duration=1.0, sample_rate=3000.0)
    not_density = Specification(BAND, [[1.0, 1.0]], response_dof='1X+',
                                ordinate_dim='force')
    with pytest.raises(ValueError, match='not one'):
        signals.random_history(not_density, duration=1.0, sample_rate=FS)
    undeclared = Specification(BAND, [[1.0, 1.0]], response_dof='1X+')
    with pytest.raises(ValueError, match='declare its units'):
        signals.random_history(undeclared, duration=1.0, sample_rate=FS)
    with pytest.raises(ValueError, match='fewer than two samples'):
        signals.random_history(_flat(), duration=0.0, sample_rate=FS)
    with pytest.raises(ValueError, match='even'):
        signals.random_history(_flat(), duration=1.0, sample_rate=FS,
                               frame_length=1001)
    with pytest.raises(ValueError, match='below the highest'):
        signals.flat_spectrum('1X+', quantity='force', band=(100, 10),
                              level=1.0)
    with pytest.raises(ValueError, match='above zero'):
        signals.flat_spectrum('1X+', quantity='force', band=BAND, level=0.0)


# ---- burst random -----------------------------------------------------------

def test_a_burst_is_on_for_its_share_and_quiet_after():
    n, share = 8192, 0.4
    history = signals.burst_random_history(_flat(), bursts=6, sample_rate=FS,
                                           frame_length=n, burst=share,
                                           seed=14)
    frames = history.ordinate[0].reshape(6, n)
    on = round(share * n)
    assert np.all(frames[:, on:] == 0.0), 'quiet after the burst, exactly'
    ramp = round(signals.BURST_TAPER * on)
    middle = frames[:, ramp:on - ramp]
    assert middle.std() == pytest.approx(np.sqrt(BAND[1] - BAND[0]), rel=0.03)
    assert abs(frames[:, 0]).max() < 0.05 * middle.std(), 'tapered on'
    assert not np.allclose(frames[0], frames[1]), 'each burst its own'


def test_a_burst_without_taper_and_the_refusals():
    history = signals.burst_random_history(_flat(), bursts=2, sample_rate=FS,
                                           frame_length=1024, burst=1.0,
                                           taper=0.0, seed=15)
    assert np.count_nonzero(history.ordinate[0]) == 2048
    for kwargs, words in (({'burst': 0.0}, 'above 0'),
                          ({'taper': 0.5}, 'under one half'),
                          ({'bursts': 0}, 'at least one burst')):
        args = {'bursts': 2, 'sample_rate': FS, 'frame_length': 1024} | kwargs
        with pytest.raises(ValueError, match=words):
            signals.burst_random_history(_flat(), **args)


# ---- periodic random --------------------------------------------------------

def test_pseudo_random_repeats_and_is_leakage_free():
    """One period's transform holds exactly the target's amplitude at
    every line in the band and nothing between: the period is a whole
    number of every line's cycles."""
    n = 4096
    history = signals.periodic_random_history(_flat(level=0.5), periods=3,
                                              sample_rate=FS, frame_length=n,
                                              seed=16)
    periods = history.ordinate[0].reshape(3, n)
    assert np.array_equal(periods[0], periods[2])
    spectrum = np.abs(np.fft.rfft(periods[0]))
    f = np.arange(n // 2 + 1) * FS / n
    inside = (f >= BAND[0]) & (f <= BAND[1])
    assert np.allclose(spectrum[inside], n * np.sqrt(0.5 * FS / n / 2.0),
                       rtol=1e-9)
    assert spectrum[~inside].max() < 1e-9 * spectrum[inside].max()


def test_periodic_random_draws_a_new_period_every_few():
    history = signals.periodic_random_history(_flat(), periods=6,
                                              sample_rate=FS,
                                              frame_length=1024, repeats=2,
                                              seed=17)
    p = history.ordinate[0].reshape(6, 1024)
    assert np.array_equal(p[0], p[1]) and np.array_equal(p[4], p[5])
    assert not np.allclose(p[1], p[2])
    assert history.comment[0] == 'periodic random, 2 periods each'


# ---- impulse and pulses -----------------------------------------------------

def test_an_impulse_is_a_half_sine_of_its_width_and_peak():
    """Brandon, 2026-10-25: a half-sine whose width sets the band it
    excites, not one sample flat to the Nyquist frequency."""
    rate, width, peak = 20000.0, 0.002, 300.0
    history = signals.impulse_history(['1X+', '2Y+'], quantity='force',
                                      duration=0.5, sample_rate=rate,
                                      at=0.1, peak=peak, width=width)
    row = history.ordinate[0]
    t = np.arange(row.size) / rate - 0.1
    inside = (t >= 0.0) & (t <= width)
    assert np.allclose(row[inside], peak * np.sin(np.pi * t[inside] / width),
                       atol=1e-9)
    assert np.all(row[~inside] == 0.0)
    # the sampled half-sine of N = 40 intervals sums to cot(π/2N), the
    # continuous area 2PT/π less a part in (π²/12N²)
    assert row.sum() == pytest.approx(peak / np.tan(np.pi / 80), rel=1e-9)
    assert np.array_equal(history.ordinate[0], history.ordinate[1])
    assert history.ordinate_dim == ['force', 'force']
    assert history.comment[0] == 'impulse, 2 ms half-sine'


def test_the_rolloff_points_are_the_half_sines():
    """The −3 dB and −10 dB points solve |cos(πfT)/(1 − 4f²T²)| for
    their levels, and the first zero is 1.5/T; the record's own spectrum,
    finely sampled, falls by those amounts there."""
    width = 0.001
    down3, down10, zero = signals.impulse_rolloff(width)
    ratio = signals._half_sine_ratio
    assert ratio(down3 * width) == pytest.approx(10 ** (-3 / 20), rel=1e-9)
    assert ratio(down10 * width) == pytest.approx(10 ** (-10 / 20), rel=1e-9)
    assert (down3, down10, zero) == pytest.approx((593.51, 1019.37, 1500.0),
                                                  abs=0.01)
    history = signals.impulse_history('1X+', quantity='force', duration=0.2,
                                      sample_rate=200000.0, width=width,
                                      peak=50.0)
    spectrum = signals.impulse_spectrum(history)
    f, magnitude = spectrum.abscissa, np.abs(spectrum.ordinate[0])
    assert spectrum.ordinate_dim == ['force*time']
    assert magnitude[0] == pytest.approx(2 * 50.0 * width / np.pi, rel=1e-4)

    def db(at):
        return 20 * np.log10(np.interp(at, f, magnitude) / magnitude[0])

    assert db(down3) == pytest.approx(-3.0, abs=0.01)
    assert db(down10) == pytest.approx(-10.0, abs=0.01)
    assert db(zero) < -50.0


def test_a_classical_pulse_is_the_pulse_from_its_start():
    pulse = ClassicalPulse('half-sine', 490.3325, 0.011)
    history = signals.impulse_history('1X+', quantity='acceleration',
                                      duration=0.1, sample_rate=20000.0,
                                      at=0.02, pulse=pulse)
    t = np.arange(2000) / 20000.0
    assert np.allclose(history.ordinate[0], pulse.at(t - 0.02))
    assert history.ordinate[0].max() == pytest.approx(490.3325, rel=1e-3)
    assert np.all(history.ordinate[0][t < 0.02] == 0.0)


def test_an_impulse_out_of_the_record_or_of_nothing_is_refused():
    with pytest.raises(ValueError, match='outside the record'):
        signals.impulse_history('1X+', quantity='force', duration=1.0,
                                sample_rate=100.0, at=1.0)
    with pytest.raises(ValueError, match='other than zero'):
        signals.impulse_history('1X+', quantity='force', duration=1.0,
                                sample_rate=1000.0, peak=0.0, width=0.002)
    with pytest.raises(ValueError, match='under two samples at 1000 Hz; '
                                         'the narrowest is 2 ms'):
        signals.impulse_history('1X+', quantity='force', duration=1.0,
                                sample_rate=1000.0, width=0.0015)
    signals.impulse_history('1X+', quantity='force', duration=1.0,
                            sample_rate=1000.0, width=0.002)
    with pytest.raises(ValueError, match='runs past the end of the record'):
        signals.impulse_history('1X+', quantity='force', duration=1.0,
                                sample_rate=1000.0, at=0.995, width=0.01)


# ---- sine -------------------------------------------------------------------

def _sweep(g=9.80665):
    up = SineTone('Up', 0.0, [20.0, 200.0],
                  [[1.0 * g, 1.0 * g], [2.0 * g, 2.0 * g]],
                  [LOG], [4.0], phase=[[0.0, np.pi / 2]] * 2)
    return SineSweepSpecification([up], ['1X+', '2X+'], ordinate_unit='m/s**2')


def test_a_sine_follows_its_tone():
    spec = _sweep()
    history = signals.sine_history(spec, sample_rate=FS)
    tone = spec.tones[0]
    assert history.ordinate.shape[1] == len(tone.argument(1.0 / FS))
    g = 9.80665
    head, tail = history.ordinate[:, :1200], history.ordinate[:, -1200:]
    assert np.abs(head[0]).max() / g == pytest.approx(1.0, abs=0.01)
    assert np.abs(tail[0]).max() / g == pytest.approx(2.0, abs=0.01)
    crossings = np.flatnonzero(np.diff(np.sign(history.ordinate[0][:4096])))
    assert len(crossings) / 2 / (4096 / FS) == pytest.approx(20.0, rel=0.05), \
        'starts at 20 Hz'
    # the second DOF a quarter cycle ahead: cos(a + pi/2) = -sin(a)
    a = tone.argument(1.0 / FS)[:100]
    assert np.allclose(history.ordinate[1][:100] / history.ordinate[0][:100].max(),
                       -np.sin(a), atol=0.02)


def test_a_sine_reaching_nyquist_or_an_unknown_tone_is_refused():
    with pytest.raises(ValueError, match='Nyquist'):
        signals.sine_history(_sweep(), sample_rate=300.0)
    with pytest.raises(ValueError, match="no tone 'Down'"):
        signals.sine_history(_sweep(), sample_rate=FS, tone='Down')


# ---- the reading, line by line: a period holds each line exactly ------------

def _lines(spec, n=8192, seed=20):
    """Each line's density as drawn: one period's transform, which a
    periodic record holds exactly, as G = 2|X|²/(N² df)."""
    period = signals.periodic_random_history(spec, periods=1, sample_rate=FS,
                                             frame_length=n, seed=seed)
    spectrum = np.fft.rfft(period.ordinate, axis=1)
    df = FS / n
    return np.arange(n // 2 + 1) * df, 2.0 * np.abs(spectrum) ** 2 / (n * n * df)


def test_a_bin_reaches_half_way_to_its_neighbor_at_either_end():
    f = np.arange(100.0, 210.0, 10.0)
    spec = Specification(f, np.ones((1, len(f))), response_dof='1X+',
                         ordinate_dim='force**2/frequency')
    spec.interpolation = Specification.reading_of(f)
    lines, level = _lines(spec)
    held = lines[level[0] > 0.5]
    assert held.min() == 95.0 and held.max() == 204.0, \
        'from 5 Hz below the first line to 5 Hz above the last'


def test_breakpoints_are_read_log_log_exactly_and_zero_outside():
    spec = Specification([20.0, 160.0, 1600.0], [[1.0 / 64, 1.0, 1.0]],
                         response_dof='1X+', ordinate_dim='force**2/frequency')
    lines, level = _lines(spec)
    at = {f: level[0][np.argmin(np.abs(lines - f))] for f in (40.0, 80.0, 1000.0)}
    assert at[40.0] == pytest.approx((40.0 / 160.0) ** 2, rel=1e-9)
    assert at[80.0] == pytest.approx(0.25, rel=1e-9)
    assert at[1000.0] == pytest.approx(1.0, rel=1e-9)
    assert level[0][(lines < 20.0) | (lines > 1600.0)].max() < 1e-20


def test_a_target_written_as_zeros_above_its_band_is_drawn():
    """A controller writes zero on the lines it does not control; zeros
    up to 5 kHz are not a band reaching past the Nyquist frequency."""
    f = np.arange(0.0, 5000.0, 10.0)
    level = np.where((f >= 20.0) & (f <= 2000.0), 1.0, 0.0)
    spec = Specification(f, level[None], response_dof='1X+',
                         ordinate_dim='force**2/frequency')
    spec.interpolation = Specification.reading_of(f)
    lines, drawn = _lines(spec)
    assert drawn[0][lines > 2010.0].max() < 1e-20


def test_the_refusal_names_the_first_line_that_cannot_hold():
    with pytest.raises(ValueError, match=r'at 20 Hz the target'):
        signals.periodic_random_history(_pair(cross=1.3), periods=1,
                                        sample_rate=FS, frame_length=8192)
    with pytest.raises(ValueError, match='even and at least 4'):
        signals.random_history(_flat(), duration=1.0, sample_rate=FS,
                               frame_length=2)


def test_a_target_from_zero_hertz_takes_the_longest_default_frame():
    """A target written down to 0 Hz asks for lines without end; the
    default stops at `DEFAULT_FRAME_MAX`."""
    f = np.arange(0.0, 1010.0, 10.0)
    spec = Specification(f, np.ones((1, len(f))), response_dof='1X+',
                         ordinate_dim='force**2/frequency')
    _, _, frame, _ = signals._plan(spec, FS, None, None, True)
    assert frame == signals.DEFAULT_FRAME_MAX


def test_frames_made_in_turn_are_the_frames_made_at_once(monkeypatch):
    """How many frames are made at a time changes the memory a record
    takes, not the record."""
    one = signals.random_history(_flat(), duration=3.0, sample_rate=FS,
                                 frame_length=512, seed=26)
    bursts = signals.burst_random_history(_flat(), bursts=5, sample_rate=FS,
                                          frame_length=512, seed=27)
    monkeypatch.setattr(signals, 'FRAMES_AT_ONCE', 3)
    assert np.array_equal(one.ordinate, signals.random_history(
        _flat(), duration=3.0, sample_rate=FS, frame_length=512,
        seed=26).ordinate)
    assert np.array_equal(bursts.ordinate, signals.burst_random_history(
        _flat(), bursts=5, sample_rate=FS, frame_length=512,
        seed=27).ordinate)


def test_the_default_frame_has_lines_of_a_quarter_the_lowest_frequency():
    spec = signals.flat_spectrum('1X+', quantity='force', band=(2.0, 200.0),
                                 level=1.0)
    _, _, frame, _ = signals._plan(spec, 1024.0, None, None, True)
    assert frame == 2048, '0.5 Hz lines, a quarter of 2 Hz'
    _, _, frame, _ = signals._plan(_flat(), FS, None, None, True)
    assert frame == 8192, 'never coarser than 1 Hz'


def test_the_taper_is_a_cosine_over_its_share_of_the_burst():
    """The same seed draws the same frames, so a tapered burst over an
    untapered one is the gate itself."""
    kwargs = {'bursts': 2, 'sample_rate': FS, 'frame_length': 4096,
              'burst': 0.5, 'seed': 21}
    plain = signals.burst_random_history(_flat(), taper=0.0, **kwargs)
    tapered = signals.burst_random_history(_flat(), taper=0.1, **kwargs)
    keep = np.abs(plain.ordinate[0]) > 1e-6
    gate = np.zeros_like(plain.ordinate[0])
    gate[keep] = tapered.ordinate[0][keep] / plain.ordinate[0][keep]
    gate = gate.reshape(2, 4096)[0]
    ramp = round(0.1 * 2048)
    want = 0.5 - 0.5 * np.cos(np.pi * (np.arange(ramp) + 0.5) / ramp)
    assert np.allclose(gate[:ramp], want, atol=1e-9)
    assert np.allclose(gate[2048 - ramp:2048], want[::-1], atol=1e-9)
    assert np.allclose(gate[ramp:2048 - ramp], 1.0)


def test_several_dofs_keep_their_own_frames_in_order():
    burst = signals.burst_random_history(_flat(('1X+', '2X+')), bursts=3,
                                         sample_rate=FS, frame_length=1024,
                                         burst=0.5, seed=22)
    for row in burst.ordinate:
        frames = row.reshape(3, 1024)
        assert np.all(frames[:, 512:] == 0.0) and np.all(frames[:, 100] != 0.0)
    assert not np.allclose(burst.ordinate[0], burst.ordinate[1])
    periodic = signals.periodic_random_history(_flat(('1X+', '2X+')),
                                               periods=4, sample_rate=FS,
                                               frame_length=1024, repeats=2,
                                               seed=23)
    for row in periodic.ordinate:
        p = row.reshape(4, 1024)
        assert np.array_equal(p[0], p[1]) and np.array_equal(p[2], p[3])
        assert not np.allclose(p[1], p[2])


def test_a_phase_that_wraps_between_breakpoints_is_unwrapped():
    """3.0 rad, then -3.0: the short way round, through pi, not the long
    way through zero."""
    g = 9.80665
    tone = SineTone('Up', 0.0, [20.0, 40.0], [[g], [g]], [LOG], [4.0],
                    phase=[[3.0], [-3.0]])
    spec = SineSweepSpecification([tone], ['1X+'], ordinate_unit='m/s**2')
    history = signals.sine_history(spec, sample_rate=FS)
    a = tone.argument(1.0 / FS)
    mid = len(a) // 2
    window = slice(mid - 200, mid + 200)
    shift = np.angle(np.sum(history.ordinate[0][window] / g
                            * np.exp(-1j * a[window])))
    assert abs(abs(shift) - np.pi) < 0.1, 'half way between is pi'


def test_a_zero_breakpoint_bounds_the_band():
    """A breakpoint written as zero is where the band stops, not a level
    to read the next one back from."""
    spec = Specification([10.0, 20.0, 2000.0], [[0.0, 1.0, 1.0]],
                         response_dof='1X+', ordinate_dim='force**2/frequency')
    lines, level = _lines(spec)
    assert level[0][lines < 20.0].max() < 1e-20
    assert level[0][np.argmin(np.abs(lines - 21.0))] == pytest.approx(1.0)


def test_the_phases_go_all_the_way_round():
    """Uniform on the circle: the lines' unit phasors of one period
    average to nothing beyond their scatter, 1/sqrt(lines)."""
    period = signals.periodic_random_history(_flat(), periods=1,
                                             sample_rate=FS,
                                             frame_length=8192, seed=24)
    x = np.fft.rfft(period.ordinate[0])
    f = np.arange(len(x)) * FS / 8192
    inside = (f >= BAND[0]) & (f <= BAND[1])
    phasors = x[inside] / np.abs(x[inside])
    assert abs(phasors.mean()) < 3.0 / np.sqrt(inside.sum())


def test_the_last_hop_of_a_record_is_as_loud_as_the_rest():
    x = signals.random_history(_flat(), duration=30.0, sample_rate=FS,
                               frame_length=2048, seed=25).ordinate[0]
    assert x[-1024:].std() / x.std() == pytest.approx(1.0, abs=0.15)
    assert x[:1024].std() / x.std() == pytest.approx(1.0, abs=0.15)


def test_the_edges_of_what_is_accepted():
    g = signals.flat_spectrum('1X+', quantity='force', band=(20.0, 4096.0),
                              level=1.0)
    with pytest.raises(ValueError, match='Nyquist'):
        signals.random_history(g, duration=1.0, sample_rate=FS)
    assert signals.random_history(_flat(), duration=1.0, sample_rate=FS,
                                  frame_length=4, seed=1).ordinate.shape[0] == 1
    assert signals.random_history(_flat(), duration=2.0 / FS, sample_rate=FS,
                                  seed=1).ordinate.shape[1] == 2
    with pytest.raises(ValueError, match='fewer than two samples'):
        signals.random_history(_flat(), duration=1.0 / FS, sample_rate=FS)
    one = signals.burst_random_history(_flat(), bursts=1, sample_rate=FS,
                                       frame_length=1024, seed=1)
    assert one.ordinate.shape[1] == 1024
    with pytest.raises(ValueError, match='below the highest'):
        signals.flat_spectrum('1X+', quantity='force', band=(100.0, 100.0),
                              level=1.0)
    with pytest.raises(ValueError, match='above zero'):
        signals.random_history(_flat(), duration=1.0, sample_rate=0.0)
    with pytest.raises(ValueError, match='above zero'):
        signals.sine_history(_sweep(), sample_rate=0.0)
    silent = Specification(BAND, [[0.0, 0.0]], response_dof='1X+',
                           ordinate_dim='force**2/frequency')
    with pytest.raises(ValueError, match='zero everywhere'):
        signals.random_history(silent, duration=1.0, sample_rate=FS)
    pseudo = signals.periodic_random_history(_flat(), periods=2,
                                             sample_rate=FS, frame_length=64,
                                             repeats=2, seed=1)
    assert pseudo.comment[0] == 'pseudo-random', 'every period one frame'


def test_a_burst_at_no_rate_a_tone_at_nyquist_and_a_band_from_zero_are_refused():
    with pytest.raises(ValueError, match='above zero'):
        signals.burst_random_history(_flat(), bursts=1, sample_rate=0.0,
                                     frame_length=64)
    g = 9.80665
    at_nyquist = SineSweepSpecification(
        [SineTone('Up', 0.0, [20.0, 4096.0], [[g], [g]], [LOG], [4.0])],
        ['1X+'], ordinate_unit='m/s**2')
    with pytest.raises(ValueError, match='Nyquist'):
        signals.sine_history(at_nyquist, sample_rate=FS)
    with pytest.raises(ValueError, match='above zero and below'):
        signals.flat_spectrum('1X+', quantity='force', band=(0.0, 100.0),
                              level=1.0)


def test_the_target_is_drawn_on_the_lines_it_was_read_at():
    """`target_at` is the level the generator reads at each line, the
    autospectra only, and nothing outside the band."""
    spec = Specification([20.0, 160.0, 1600.0], [[1.0 / 64, 1.0, 1.0]],
                         response_dof='1X+', ordinate_dim='force**2/frequency')
    lines = np.arange(0.0, 2000.0, 1.0)
    target = signals.target_at(spec, lines)
    assert target.ordinate_dim == ['force**2/frequency']
    assert np.real(target.ordinate[0][40]) == pytest.approx(1.0 / 16, rel=1e-9)
    assert np.isnan(target.ordinate[0][[10, 1700]]).all()
    pair = signals.target_at(_pair(), lines)
    assert pair.response_dof == ['1X+', '2X+'], 'the autospectra only'


# ---- read the way it was made -----------------------------------------------

def test_each_signal_carries_the_averaging_it_is_read_with():
    """Brandon, 2026-10-25: 50 bursts opened as one average under Hann.
    A signal made in frames is read in those frames."""
    bursts = signals.burst_random_history(_flat(), bursts=50, sample_rate=FS,
                                          frame_length=2048, seed=30)
    a = bursts.averaging
    assert (a.frame_length, a.overlap, a.window, a.frames) == \
        (2048, 0.0, 'boxcar', 50)
    periodic = signals.periodic_random_history(_flat(), periods=12,
                                               sample_rate=FS,
                                               frame_length=1024, seed=31)
    a = periodic.averaging
    assert (a.frame_length, a.overlap, a.window, a.frames) == \
        (1024, 0.0, 'boxcar', 12)
    stationary = signals.random_history(_flat(), duration=4.0, sample_rate=FS,
                                        frame_length=4096, seed=32)
    a = stationary.averaging
    assert (a.frame_length, a.overlap, a.window) == (4096, 0.5, 'hann')
    assert a.frames == 15, 'as many half-overlapped frames as fit'
    tap = signals.impulse_history('1X+', quantity='force', duration=0.5,
                                  sample_rate=FS)
    a = tap.averaging
    assert (a.frame_length, a.window, a.frames) == (4096, 'boxcar', 1)
    assert signals.sine_history(_sweep(), sample_rate=FS).averaging is None


def test_bursts_read_in_their_own_frames_are_the_target_times_the_share_on():
    level, share = 2.0, 0.5
    history = signals.burst_random_history(_flat(level=level), bursts=40,
                                           sample_rate=FS, frame_length=4096,
                                           burst=share, taper=0.0, seed=33)
    psd = history.compute_psds()
    inside = _inside(psd)
    assert np.real(psd.ordinate[0, inside]).mean() == pytest.approx(
        level * share, rel=0.03)
