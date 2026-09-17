"""The motion chain: low-pass, integrate, differentiate.

The shock workflow's derivations (Brandon, 2026-08-24): filtered
acceleration, then velocity, then displacement, each a derived object
with provenance. The physics here is checkable against closed forms —
a sine's integral is a cosine over omega — so most of these tests pin
amplitudes, not shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.filters import (
    DRIFT_CORNER,
    Filtering,
    differentiate,
    filtered,
    integrate,
)

RATE = 4096.0
SAMPLES = 4 * 4096


def _history(rows, dims=None, dofs=None, units=None):
    n = np.atleast_2d(np.asarray(rows)).shape[0]
    return TimeHistory(
        np.arange(SAMPLES) / RATE, rows,
        response_dof=dofs or [f'{101 + i}Z+' for i in range(n)],
        ordinate_dim=dims or ['acceleration'] * n,
        ordinate_unit=units)


def _sine(freq, amplitude=1.0):
    return amplitude * np.sin(2 * np.pi * freq * np.arange(SAMPLES) / RATE)


# ---- the filter ----------------------------------------------------------


def test_the_low_pass_keeps_the_band_and_kills_the_rest():
    data = _history([_sine(50.0) + _sine(1500.0)])
    out = filtered(data, Filtering(high=400.0))
    spectrum = np.abs(np.fft.rfft(out.ordinate[0]))
    lines = np.fft.rfftfreq(SAMPLES, 1 / RATE)
    kept = spectrum[np.argmin(np.abs(lines - 50.0))]
    killed = spectrum[np.argmin(np.abs(lines - 1500.0))]
    assert kept == pytest.approx(SAMPLES / 2, rel=0.01), \
        'the 50 Hz tone passes at full amplitude'
    assert killed < kept * 1e-4, 'the 1500 Hz tone is gone'


def test_the_filter_is_zero_phase_so_the_peak_stays_put():
    """A causal filter would shift a pulse by its group delay, and the
    report's time axis would be quietly wrong by that much."""
    t = np.arange(SAMPLES) / RATE
    pulse = np.exp(-0.5 * ((t - 2.0) / 0.005) ** 2)
    out = filtered(_history([pulse]), Filtering(high=200.0))
    assert abs(int(np.argmax(out.ordinate[0])) - int(np.argmax(pulse))) <= 1


def test_the_filter_takes_every_channel_whatever_it_measures():
    """The drive force belongs at the same bandwidth as the responses
    it drove."""
    data = _history([_sine(50.0), _sine(50.0)],
                    dims=['acceleration', 'force'])
    out = filtered(data, Filtering(high=400.0))
    assert out.num_records == 2
    assert out.ordinate_dim == ['acceleration', 'force']
    assert out.ordinate_unit == ['m/s**2', 'N']


def test_the_marks_ride_along():
    """Same abscissa, so the averaging frames and shock windows still
    say what they said — the report numbers the same events on the
    velocity it numbered on the acceleration."""
    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.shocks import Shock

    data = _history([_sine(50.0)])
    data.averaging = Averaging(frame_length=1024, frames=4)
    data.shocks = [Shock(0.5, 0.1)]
    for out in (filtered(data, Filtering(high=400.0)),
                integrate(data), differentiate(integrate(data))):
        assert out.averaging == data.averaging
        assert tuple(out.shocks) == tuple(data.shocks)
        assert out.shocks is not data.shocks, 'a copy, not the same list'


def test_a_corner_at_or_above_nyquist_is_refused():
    with pytest.raises(ValueError, match='Nyquist'):
        filtered(_history([_sine(50.0)]), Filtering(high=RATE / 2))


def test_the_settings_validate_at_entry():
    with pytest.raises(ValueError):
        Filtering(high=0.0)
    with pytest.raises(ValueError):
        Filtering(high=100.0, order=0)


def test_uneven_sampling_is_refused():
    data = _history([_sine(50.0)])
    data.abscissa = data.abscissa ** 1.01
    with pytest.raises(ValueError):
        filtered(data, Filtering(high=400.0))
    with pytest.raises(ValueError):
        integrate(data)


# ---- integration ---------------------------------------------------------


def test_a_sine_integrates_to_its_closed_form():
    """A*sin(wt) integrates to -(A/w)*cos(wt): the amplitude comes out
    divided by omega, which pins the dx and the sign at once."""
    freq, amp = 80.0, 3.0
    out = integrate(_history([_sine(freq, amp)]))
    omega = 2 * np.pi * freq
    middle = out.ordinate[0][SAMPLES // 4: -SAMPLES // 4]
    assert np.max(np.abs(middle)) == pytest.approx(amp / omega, rel=0.01)
    assert out.ordinate_dim == ['velocity']
    assert out.ordinate_unit == ['m/s']


def test_integration_walks_the_chain_to_displacement():
    out = integrate(integrate(_history([_sine(80.0)])))
    assert out.ordinate_dim == ['length']
    assert out.ordinate_unit == ['m']


def test_a_bias_does_not_become_a_ramp():
    """The drift control is the whole reason the corner exists: a
    0.05 m/s**2 offset walked the design measurement past a meter of
    displacement, and the reading is meaningless with it in."""
    rng = np.random.default_rng(11)
    biased = _sine(80.0, 2.0) + 0.05 + rng.standard_normal(SAMPLES) * 0.1
    dirty = integrate(integrate(_history([biased]), drift_corner=None),
                      drift_corner=None)
    clean = integrate(integrate(_history([biased])))
    assert abs(dirty.ordinate[0][-1]) > 0.1, 'raw integration drifts'
    assert np.max(np.abs(clean.ordinate[0])) < 1e-3, \
        'controlled integration does not'


def test_none_means_raw_on_purpose():
    """A constant integrates to a ramp, and asking for no drift control
    must actually deliver one — None is a choice, not an absence."""
    out = integrate(_history([np.ones(SAMPLES)]), drift_corner=None)
    t = np.arange(SAMPLES) / RATE
    assert np.allclose(out.ordinate[0], t, atol=1e-6), \
        'the integral of 1 is t, untouched'


def test_unmappable_channels_are_left_out():
    data = _history([_sine(80.0), _sine(80.0)],
                    dims=['acceleration', 'force'],
                    dofs=['101Z+', '201Z+'])
    out = integrate(data)
    assert out.num_records == 1
    assert out.response_dof == ['101Z+']
    with pytest.raises(ValueError, match='nothing here integrates'):
        integrate(_history([_sine(80.0)], dims=['force']))


def test_a_bad_drift_corner_is_refused():
    with pytest.raises(ValueError, match='drift corner'):
        integrate(_history([_sine(80.0)]), drift_corner=RATE)


# ---- differentiation -----------------------------------------------------


def test_a_sine_differentiates_to_its_closed_form():
    """A*sin(wt) differentiates to A*w*cos(wt)."""
    freq, amp = 80.0, 3.0
    out = differentiate(_history([_sine(freq, amp)], dims=['velocity']))
    omega = 2 * np.pi * freq
    middle = out.ordinate[0][SAMPLES // 4: -SAMPLES // 4]
    assert np.max(np.abs(middle)) == pytest.approx(amp * omega, rel=0.01)
    assert out.ordinate_dim == ['acceleration']
    assert out.ordinate_unit == ['m/s**2']


def test_differentiation_refuses_an_acceleration():
    """The chain stops where the quantities do — jerk is not a report
    quantity here."""
    with pytest.raises(ValueError, match='nothing here differentiates'):
        differentiate(_history([_sine(80.0)]))


# ---- the round trip ------------------------------------------------------


def test_the_round_trip_comes_back():
    """acceleration -> velocity -> displacement -> velocity ->
    acceleration, within the band the filters keep. Shock-like content
    measured ~2% RMS with the corner at a tenth of the rate
    (2026-08-24); this fixture is *white* to the corner, which stacks
    energy exactly where the central difference bites and measured
    6.4% — the honest worst case. Pinned at 8%: a frequency-domain
    wrap-around or a broken dx shows up at tens of percent, and
    scipy's own numerics have room to breathe."""
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(21)
    raw = _history([rng.standard_normal(SAMPLES) * 10.0])
    accel = filtered(raw, Filtering(high=RATE / 10))
    disp = integrate(integrate(accel))
    back = differentiate(differentiate(disp))
    assert back.ordinate_dim == ['acceleration']
    band = butter(4, [2 * DRIFT_CORNER, RATE / 10], 'bandpass',
                  fs=RATE, output='sos')
    lens = lambda x: sosfiltfilt(band, x)[SAMPLES // 8: -SAMPLES // 8]
    reference, returned = lens(accel.ordinate[0]), lens(back.ordinate[0])
    error = np.sqrt(np.mean((returned - reference) ** 2)) / np.std(reference)
    assert error < 0.08, f'round trip error {100 * error:.2f}%'


# ---- the project verbs ---------------------------------------------------


def _project():
    project = visualdynamics.Project('Chain')
    data = _history([_sine(80.0, 2.0)])
    project.add('Run', data)
    return project


def test_the_verbs_name_by_what_the_result_is():
    project = _project()
    filtered = project.filter_data('Run')
    assert filtered == 'Run Filtered'
    velocity = project.integrate(filtered)
    assert velocity == 'Run Filtered Velocity'
    displacement = project.integrate(velocity)
    assert displacement == 'Run Filtered Displacement', \
        'the quantity word is swapped, not stacked'
    assert project.differentiate(displacement) == 'Run Filtered Velocity (2)'


def test_filter_data_adopts_the_suggestion_once():
    project = _project()
    project.filter_data('Run')
    filtering = project['Run'].filtering
    assert filtering is not None
    assert filtering.kind == 'low-pass'
    assert filtering.high == pytest.approx(RATE / 10)


def test_the_chain_carries_provenance_and_goes_stale_together():
    """Change the filter and the filtered record goes stale; refresh it
    and the velocity built on it goes stale in turn — the content
    fingerprint is the cascade, nobody walks a graph."""
    project = _project()
    filtered = project.filter_data('Run')
    velocity = project.integrate(filtered)
    assert not project.stale()
    project['Run'].filtering = Filtering(high=100.0)
    stale = project.stale()
    assert filtered in stale and 'low-pass' in stale[filtered]
    assert velocity not in stale, 'the velocity is stale only once its ' \
        'source is actually recomputed'
    project.refresh(filtered)
    assert velocity in project.stale()
    project.refresh(velocity)
    assert not project.stale()


def test_refresh_recomputes_with_the_recorded_drift_corner():
    project = _project()
    filtered = project.filter_data('Run')
    velocity = project.integrate(filtered, drift_corner=None)
    assert project.provenance[velocity]['params']['drift_corner'] is None
    project['Run'].filtering = Filtering(high=100.0)
    project.refresh_stale()
    # raw integration of a (zero-mean) sine still carries no high-pass:
    # its integral holds the -(A/w)*cos offset structure untouched at
    # the very start, where a high-pass would have bent it to zero mean
    assert not project.stale()


def test_the_settings_survive_a_save_and_a_load(tmp_path):
    project = _project()
    project['Run'].filtering = Filtering(high=300.0, order=6)
    filtered = project.filter_data('Run')
    velocity = project.integrate(filtered)
    path = project.save(tmp_path / 'chain.vdyn')
    back = visualdynamics.Project.open(path)
    assert back['Run'].filtering == Filtering(high=300.0, order=6)
    assert back[velocity].ordinate_dim == ['velocity']
    assert not back.stale(), 'nothing moved, so nothing is stale'
    back['Run'].filtering = Filtering(high=200.0)
    assert filtered in back.stale(), \
        'the fingerprint survived the file, so the badge still works'


# ---- the other kinds -----------------------------------------------------


def test_the_high_pass_keeps_the_band_and_kills_the_rest():
    data = _history([_sine(50.0) + _sine(400.0)])
    out = filtered(data, Filtering(low=200.0))
    kept = np.fft.rfft(out.ordinate[0])
    lines = np.fft.rfftfreq(SAMPLES, 1.0 / RATE)
    high = np.abs(kept[np.argmin(np.abs(lines - 400.0))])
    low = np.abs(kept[np.argmin(np.abs(lines - 50.0))])
    assert high > 0.4 * SAMPLES, 'the band above the corner survives'
    assert low < 1e-3 * high, 'and the band below it is gone'


def test_the_band_pass_keeps_the_middle_only():
    data = _history([_sine(30.0) + _sine(200.0) + _sine(900.0)])
    out = filtered(data, Filtering(low=100.0, high=500.0))
    kept = np.fft.rfft(out.ordinate[0])
    lines = np.fft.rfftfreq(SAMPLES, 1.0 / RATE)

    def at(hz):
        return np.abs(kept[np.argmin(np.abs(lines - hz))])

    assert at(200.0) > 0.4 * SAMPLES, 'the middle survives'
    assert at(30.0) < 1e-3 * at(200.0), 'below the band is gone'
    assert at(900.0) < 1e-3 * at(200.0), 'and above it too'


def test_the_kind_is_derived_from_which_edges_exist():
    """Never stored beside the edges, so it cannot disagree with
    them — an invalid pairing cannot be stated at all."""
    assert Filtering(high=500.0).kind == 'low-pass'
    assert Filtering(low=5.0).kind == 'high-pass'
    assert Filtering(low=5.0, high=500.0).kind == 'band-pass'
    with pytest.raises(ValueError, match='at least one'):
        Filtering()
    with pytest.raises(ValueError, match='runs upward'):
        Filtering(low=500.0, high=5.0)
    with pytest.raises(TypeError):
        # keyword-only on purpose: the first field was `corner` until
        # 2026-08-28, and a bare Filtering(200.0) quietly flipping
        # from low-pass to high-pass must not survive a call site
        Filtering(200.0)


def test_describe_is_the_one_wording_everywhere():
    """The status line, the staleness story and the report prose all
    take this string, so it is pinned once."""
    assert Filtering(high=500.0).describe() == 'low-pass at 500 Hz'
    assert Filtering(low=5.0).describe() == 'high-pass at 5 Hz'
    assert Filtering(low=5.0, high=500.0).describe() == \
        'band-pass from 5 to 500 Hz'


def test_every_kind_is_minus_six_dB_at_its_nominal_edges():
    """The zero-phase pair squares the magnitude, so a Butterworth's
    own −3 dB corner reads −6.02 on what the data sees — for every
    kind, at every edge it has."""
    from visualdynamics.core.filters import response

    def at(filtering, hz):
        frequencies, magnitude = response(filtering, RATE)
        return float(np.interp(hz, frequencies, magnitude))

    assert at(Filtering(low=50.0), 50.0) == pytest.approx(-6.02, abs=0.05)
    band = Filtering(low=50.0, high=400.0)
    assert at(band, 50.0) == pytest.approx(-6.02, abs=0.05)
    assert at(band, 400.0) == pytest.approx(-6.02, abs=0.05)
    assert at(band, 140.0) == pytest.approx(0.0, abs=0.05), \
        'and flat in the middle of the pass band'


def test_every_kind_survives_a_save_and_a_load(tmp_path):
    """An absent edge is an absent attribute in the file, and comes
    back as None — the file says what the class says."""
    for filtering in (Filtering(low=5.0),
                      Filtering(low=5.0, high=300.0, order=6)):
        project = _project()
        project['Run'].filtering = filtering
        path = project.save(tmp_path / f'{filtering.kind}.vdyn')
        back = visualdynamics.Project.open(path)
        assert back['Run'].filtering == filtering
