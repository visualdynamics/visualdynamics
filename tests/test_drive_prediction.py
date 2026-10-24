"""The drive prediction: the drives a specification asks for through an
FRF, the responses they make, and what they ask of the rig.

Pinned against arithmetic worked by hand, independent of the code. One
drive and one channel: a plant of 2 g/V and a target of 0.04 g^2/Hz
need a drive of 0.04 / 2^2 = 0.01 V^2/Hz at every line, whatever the
plant's phase; over 20 to 2000 Hz on 5 Hz lines that is 397 lines of
5 Hz each, 4.455 V rms. Tikhonov at a weight of a quarter on a flat
plant scales the drive by 1 / (1 + 1/4)^2, to 0.0064 V^2/Hz, and the
control channel then reads 1.94 dB under the target. Two drives and
two channels, the plant [[1, 0.5], [0, 1]] g/V (drive 2 reaches
channel 1 at half strength) and the target [[1, 0.5j], [-0.5j, 2]]
times 0.01 g^2/Hz, need the drives

    Z S Z^H = [[1.5, -1 + 0.5j], [-1 - 0.5j, 2]] times 0.01 V^2/Hz,

with Z = [[1, -0.5], [0, 1]] V/g. The shaker's velocity and travel are
pinned to the closed forms of a flat acceleration autospectrum divided
by (2 pi f)^2 and (2 pi f)^4.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from visualdynamics.core.data import Frf, Psd, Specification
from visualdynamics.core.drive_prediction import (
    DURATION,
    UNMET_DB,
    predict_drives,
    regularized_inverse,
    runs,
    through,
)
from visualdynamics.core.shaker import ShakerRating

G = 9.80665
LEVEL = 0.04                                   # g^2/Hz
BAND = (20.0, 2000.0)


def lines(step=5.0, top=2000.0):
    return np.arange(0.0, top + step / 2, step)


def siso_frf(gain=2.0, f=None, phase=0.0, dof='1Z+', reference='1Z+'):
    """A flat plant of `gain` g/V at every line."""
    f = lines() if f is None else f
    values = np.full(len(f), gain * G * np.exp(1j * phase))
    return Frf(f, values[None], response_dof=dof, reference_dof=reference,
               ordinate_dim='acceleration/voltage')


def flat_spec(level=LEVEL, dof='1Z+', band=BAND):
    return Specification(np.array(band), np.array([[level, level]]) * G ** 2,
                         response_dof=[dof],
                         ordinate_dim='acceleration**2/frequency')


def matrix_frf(plant, f=None, quantity='acceleration/voltage'):
    """An FRF holding `plant` (rows x drives, in g/V) at every line,
    rows at 1Z+, 2Z+, ... and drives at 1Z+, 2Z+, ...."""
    f = lines() if f is None else f
    plant = np.asarray(plant, dtype=complex)
    rows, responses, references = [], [], []
    for i in range(plant.shape[0]):
        for j in range(plant.shape[1]):
            rows.append(np.full(len(f), plant[i, j] * G))
            responses.append(f'{i + 1}Z+')
            references.append(f'{j + 1}Z+')
    return Frf(f, np.array(rows), response_dof=responses,
               reference_dof=references, ordinate_dim=quantity)


def matrix_spec(matrix, band=BAND, skip=()):
    """A specification of `matrix` (in g^2/Hz) at every breakpoint,
    every pair written as its own record except the ones in `skip`."""
    matrix = np.asarray(matrix, dtype=complex)
    records, responses, references = [], [], []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if (i, j) in skip:
                continue
            records.append(np.full(len(band), matrix[i, j] * G ** 2))
            responses.append(f'{i + 1}Z+')
            references.append(f'{j + 1}Z+')
    return Specification(np.array(band), np.array(records),
                         response_dof=responses, reference_dof=references,
                         ordinate_dim='acceleration**2/frequency')


TWO = np.array([[1.0, 0.5], [0.0, 1.0]])
TARGET = np.array([[1.0, 0.5j], [-0.5j, 2.0]]) * 0.01
DRIVES = np.array([[1.5, -1.0 + 0.5j], [-1.0 - 0.5j, 2.0]]) * 0.01


def at(prediction, frequency):
    return int(np.flatnonzero(prediction.frequency == frequency)[0])


# ---- the inverse -------------------------------------------------------------


def test_a_well_conditioned_plant_inverts_exactly_either_way():
    plant = (np.array([[2.0, 1.0j], [0.5, 3.0]]) * G)[None]
    exact = np.linalg.inv(plant[0])
    truncated, kept = regularized_inverse(plant)
    assert truncated[0] == pytest.approx(exact, rel=1e-12)
    assert kept.tolist() == [2]
    # Tikhonov is the inverse only as the weight goes to zero
    nearly, _kept = regularized_inverse(plant, method='tikhonov',
                                        weight=1e-12)
    assert nearly[0] == pytest.approx(exact, rel=1e-9)


def test_truncation_drops_the_direction_under_the_tolerance():
    """Singular values 1 and 0.005: at a tolerance of 0.01 the weak one
    goes and the inverse holds only the strong one; at 0.001 it stays."""
    plant = np.diag([1.0, 0.005]).astype(complex)[None]
    inverse, kept = regularized_inverse(plant, tolerance=0.01)
    assert kept.tolist() == [1]
    assert inverse[0] == pytest.approx(np.diag([1.0, 0.0]))
    inverse, kept = regularized_inverse(plant, tolerance=0.001)
    assert kept.tolist() == [2]
    assert inverse[0] == pytest.approx(np.diag([1.0, 200.0]))


def test_tikhonov_is_the_book_formula_against_the_band_largest_gain():
    """(H^H H + lambda I)^-1 H^H, lambda the weight times the largest
    squared singular value over every line, not the line's own."""
    rng = np.random.default_rng(7)
    plant = (rng.standard_normal((3, 2, 2))
             + 1j * rng.standard_normal((3, 2, 2)))
    plant[1] *= 0.01                     # a weak line
    weight = 0.05
    scale = max(np.linalg.svd(p, compute_uv=False)[0] ** 2 for p in plant)
    inverse, _kept = regularized_inverse(plant, method='tikhonov',
                                         weight=weight)
    for line in range(3):
        h = plant[line]
        book = np.linalg.solve(h.conj().T @ h + weight * scale * np.eye(2),
                               h.conj().T)
        assert inverse[line] == pytest.approx(book, rel=1e-10, abs=1e-14)
    # on one channel, H* / (|H|^2 + lambda)
    single = np.array([2.0, 0.1, 0.0], dtype=complex)[:, None, None]
    inverse, kept = regularized_inverse(single, method='tikhonov',
                                        weight=0.25)
    lam = 0.25 * 4.0
    assert inverse[:, 0, 0] == pytest.approx(
        [2.0 / (4.0 + lam), 0.1 / (0.01 + lam), 0.0])
    assert kept.tolist() == [1, 0, 0]


def test_a_line_not_a_number_has_no_inverse_and_a_zero_line_one_of_zero():
    plant = np.array([[[np.nan]], [[0.0]], [[2.0]]], dtype=complex)
    for method in ('truncated', 'tikhonov'):
        inverse, kept = regularized_inverse(plant, method=method)
        assert np.isnan(inverse[0, 0, 0])
        assert inverse[1, 0, 0] == 0.0
        assert kept.tolist() == [0, 0, 1]


@pytest.mark.parametrize('settings, words', [
    ({'method': 'pinv'}, "'truncated' or 'tikhonov', not 'pinv'"),
    ({'tolerance': 0.0}, 'above 0 and under 1, not 0'),
    ({'tolerance': 1.0}, 'above 0 and under 1, not 1'),
    ({'method': 'tikhonov', 'weight': 0.0}, 'above zero and finite, not 0'),
    ({'method': 'tikhonov', 'weight': np.nan}, 'not nan'),
])
def test_an_inverse_out_of_range_is_refused(settings, words):
    with pytest.raises(ValueError, match=words.replace('(', r'\(')):
        regularized_inverse(np.ones((1, 1, 1)), **settings)


def test_through_is_h_s_h_conjugate_transposed():
    h = np.array([[[1.0, 2.0j], [0.0, 1.0]]])
    s = np.array([[[2.0, 0.5], [0.5, 1.0]]], dtype=complex)
    assert through(h, s)[0] == pytest.approx(h[0] @ s[0] @ h[0].conj().T)


def test_runs_are_the_first_and_last_index_of_each_stretch():
    assert runs([0, 1, 1, 0, 1, 0, 0, 1]) == [(1, 2), (4, 4), (7, 7)]
    assert runs([1, 1]) == [(0, 1)]
    assert runs([0, 0]) == []


# ---- one drive, one channel ---------------------------------------------------


@pytest.mark.parametrize('phase', [0.0, 0.7, -2.0])
def test_one_drive_is_the_target_over_the_plant_squared(phase):
    prediction = predict_drives(siso_frf(phase=phase), flat_spec())
    k = at(prediction, 100.0)
    assert prediction.drive_asd(0)[k] == pytest.approx(0.01, rel=1e-12)
    assert prediction.single
    # the drive reproduces the target, line for line
    assert prediction.shortfall_db[prediction.judged] == pytest.approx(
        0.0, abs=1e-9)
    assert not prediction.unmet.any()


def test_the_drive_rms_and_peak_are_worked_by_hand():
    """397 lines of 5 Hz at 0.01 V^2/Hz: 4.455 V rms; at a stated peak
    factor of three, 13.37 V against a 10 V limit, 134 %."""
    prediction = predict_drives(siso_frf(), flat_spec(), drive_limit=10.0,
                                peak_factor=3.0)
    drive = prediction.demands[0]
    assert drive.kind == 'drive' and drive.channel == '1Z+'
    assert drive.rms == pytest.approx(math.sqrt(0.01 * 397 * 5.0))
    assert drive.peak == pytest.approx(3.0 * drive.rms)
    assert drive.share == pytest.approx(3.0 * math.sqrt(19.85) / 10.0)
    assert drive.exceeds
    assert prediction.first_out is drive
    assert prediction.fits is False
    assert prediction.headroom_db == pytest.approx(
        -20.0 * math.log10(drive.share))
    assert prediction.describe()[0] == (
        '1 drive for 1 control channel: the drive at 1Z+ runs out first, '
        'at 134 % of its rating, most at 20 Hz')


def test_the_expected_peak_follows_the_crossing_rate_and_the_duration():
    """A flat band from 20 to 2000 Hz crosses its mean about
    sqrt((f2^3 - f1^3) / (3 (f2 - f1))) = 1,156 times a second each
    way; over a minute the expected largest peak of either sign is
    sqrt(2 ln(2 x 1156 x 60)) = 4.87 standard deviations, over an hour
    5.68."""
    rate = math.sqrt((2000.0 ** 3 - 20.0 ** 3) / (3 * 1980.0))
    for duration in (DURATION, 3600.0):
        prediction = predict_drives(siso_frf(f=lines(0.5)), flat_spec(),
                                    duration=duration)
        drive = prediction.demands[0]
        factor = math.sqrt(2.0 * math.log(2.0 * rate * duration))
        assert drive.peak / drive.rms == pytest.approx(factor, rel=2e-3)


def test_tikhonov_holds_a_flat_plant_by_one_over_one_plus_the_weight_squared():
    prediction = predict_drives(siso_frf(), flat_spec(), method='tikhonov',
                                weight=0.25)
    k = at(prediction, 500.0)
    assert prediction.drive_asd(0)[k] == pytest.approx(0.0064)
    assert prediction.shortfall_db[k, 0] == pytest.approx(
        10.0 * math.log10(0.64))
    # 1.94 dB under is past the 1 dB the inverse is held to
    assert UNMET_DB < 1.94
    assert prediction.unmet_bands == [(20.0, 2000.0)]
    assert 'target not met by the inverse within 1 dB at 20 to 2000 Hz' \
        in prediction.describe()


def test_tikhonov_stays_small_at_a_notch_where_the_plain_inverse_does_not():
    """A plant that falls to a thousandth of its gain at one line: the
    plain inverse asks a million times the drive there; Tikhonov at a
    weight of 1e-4 holds it near the target times |H|^2 / lambda^2."""
    f = lines()
    gains = np.full(len(f), 2.0)
    notch = int(np.flatnonzero(f == 300.0)[0])
    gains[notch] = 0.002
    frf = Frf(f, (gains * G)[None].astype(complex), response_dof='1Z+',
              reference_dof='1Z+', ordinate_dim='acceleration/voltage')
    plain = predict_drives(frf, flat_spec())
    held = predict_drives(frf, flat_spec(), method='tikhonov', weight=1e-4)
    assert plain.drive_asd(0)[notch] == pytest.approx(0.04 / 0.002 ** 2)
    lam = 1e-4 * 4.0
    assert held.drive_asd(0)[notch] == pytest.approx(
        0.04 * 0.002 ** 2 / (0.002 ** 2 + lam) ** 2)
    assert held.unmet_bands == [(300.0, 300.0)]


def test_a_breakpoint_specification_is_read_log_log_at_the_frf_lines():
    """0.01 at 20 Hz to 0.04 at 80 Hz is a power law of slope one: 0.02
    at 40 Hz, so the drive there is 0.02 / 4."""
    spec = Specification(np.array([20.0, 80.0, 2000.0]),
                         np.array([[0.01, 0.04, 0.04]]) * G ** 2,
                         response_dof=['1Z+'],
                         ordinate_dim='acceleration**2/frequency')
    prediction = predict_drives(siso_frf(), spec)
    assert prediction.drive_asd(0)[at(prediction, 40.0)] == pytest.approx(
        0.005)
    assert not prediction.judged[at(prediction, 15.0)]


def test_a_specification_on_lines_is_read_per_bin():
    """A density on 10 Hz lines holds its value across each line's bin,
    so the FRF's 5 Hz lines inside a bin read that bin's level."""
    f10 = np.arange(20.0, 2001.0, 10.0)
    levels = np.where(f10 < 1000.0, 0.04, 0.01) * G ** 2
    spec = Specification(f10, levels[None], response_dof=['1Z+'],
                         ordinate_dim='acceleration**2/frequency')
    spec.interpolation = 'bin'
    prediction = predict_drives(siso_frf(), spec)
    assert prediction.drive_asd(0)[at(prediction, 500.0)] == \
        pytest.approx(0.01)
    assert prediction.drive_asd(0)[at(prediction, 1500.0)] == \
        pytest.approx(0.0025)
    assert prediction.drive_asd(0)[at(prediction, 1505.0)] == \
        pytest.approx(0.0025)


def test_a_line_where_the_frf_is_not_a_number_is_left_out_and_said():
    frf = siso_frf()
    frf.ordinate[0, at(predict_drives(frf, flat_spec()), 100.0)] = np.nan
    prediction = predict_drives(frf, flat_spec())
    k = at(prediction, 100.0)
    assert not prediction.judged[k]
    assert np.isnan(prediction.drive_asd(0)[k])
    assert '1 line where the FRF is not a number not predicted, first at 1Z+/1Z+' in \
        prediction.notes
    # the line adds nothing to the rms: 396 lines of 5 Hz
    assert prediction.demands[0].rms == pytest.approx(
        math.sqrt(0.01 * 396 * 5.0))


def test_a_plant_zero_at_some_lines_says_nothing_can_be_driven_there():
    frf = siso_frf()
    frf.ordinate[0, 40:42] = 0.0
    prediction = predict_drives(frf, flat_spec())
    assert 'the plant is zero at 2 lines: nothing can be driven there' in \
        prediction.notes
    assert prediction.drive_asd(0)[40] == 0.0
    assert prediction.unmet[40] and prediction.unmet[41]


# ---- two drives, two channels ---------------------------------------------------


def test_two_drives_are_z_s_z_conjugate_transposed_by_hand():
    prediction = predict_drives(matrix_frf(TWO), matrix_spec(TARGET))
    k = at(prediction, 100.0)
    assert prediction.drive_csd[k] == pytest.approx(DRIVES, rel=1e-12)
    assert prediction.control[k] == pytest.approx(TARGET * G ** 2,
                                                  rel=1e-12)
    assert prediction.drives == ['1Z+', '2Z+']
    assert prediction.title == '2 drives for 2 control channels'
    # |(-1 + 0.5j)|^2 / (1.5 x 2) = 1.25 / 3
    assert prediction.drive_coherence(0, '2Z+')[k] == pytest.approx(
        1.25 / 3.0)


def test_a_cross_term_held_only_by_its_mirror_is_its_conjugate():
    full = predict_drives(matrix_frf(TWO), matrix_spec(TARGET))
    mirrored = predict_drives(matrix_frf(TWO),
                              matrix_spec(TARGET, skip={(0, 1)}))
    k = at(full, 300.0)
    assert mirrored.drive_csd[k] == pytest.approx(full.drive_csd[k])


def test_any_well_conditioned_plant_reproduces_its_target_exactly():
    rng = np.random.default_rng(31)
    plant = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    plant += 3.0 * np.eye(3)
    root = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    target = root @ root.conj().T * 0.01
    prediction = predict_drives(matrix_frf(plant), matrix_spec(target))
    k = at(prediction, 1000.0)
    assert prediction.control[k] == pytest.approx(target * G ** 2,
                                                  rel=1e-10)
    assert not prediction.unmet.any()


def test_two_nearly_identical_drives_lose_a_direction_and_say_so():
    """Two drives reaching both channels in proportions a thousandth
    apart: a condition number past 1000, so the truncated inverse keeps
    one direction, the target is not met, and the reading says where."""
    plant = np.array([[1.0, 1.0], [1.0, 1.001]])
    prediction = predict_drives(matrix_frf(plant), matrix_spec(TARGET))
    assert set(prediction.kept[prediction.judged]) == {1}
    assert prediction.condition[at(prediction, 100.0)] > 1000.0
    assert prediction.unmet_bands == [(20.0, 2000.0)]
    assert ('the inverse set aside a direction at 397 lines where the '
            'plant is ill-conditioned') in prediction.notes
    # the plain inverse would have needed drives thousands of times larger
    loose = predict_drives(matrix_frf(plant), matrix_spec(TARGET),
                           tolerance=1e-6)
    k = at(prediction, 100.0)
    assert loose.drive_asd(0)[k] > 1e4 * prediction.drive_asd(0)[k]


def test_more_drives_than_channels_takes_the_least_power_and_says_so():
    """Two equal drives on one channel: each takes a quarter of the
    one-drive answer, the least-power way of meeting the target."""
    prediction = predict_drives(matrix_frf([[2.0, 2.0]]), flat_spec())
    k = at(prediction, 100.0)
    assert prediction.drive_asd(0)[k] == pytest.approx(0.04 / 16.0)
    assert prediction.drive_coherence(0, 1)[k] == pytest.approx(1.0)
    assert any(note.startswith('more drives than control channels')
               for note in prediction.notes)


# ---- what the drives ask of the shaker -------------------------------------------


def test_the_shakers_force_velocity_and_travel_follow_the_closed_forms():
    """A flat 0.04 g^2/Hz from 20 to 2000 Hz: 8.90 g rms, so 60 kg needs
    60 x 8.90 g rms; the velocity's mean square is G / (4 pi^2) times
    (1/f1 - 1/f2) and the displacement's G / (16 pi^4) / 3 times
    (1/f1^3 - 1/f2^3). A body four times the moving mass adds a quarter
    to the velocity and the travel."""
    f = lines(0.25)
    rating = ShakerRating(force=30000.0, stroke=0.0508, velocity=1.8,
                          moving_mass=60.0, body_mass=240.0)
    prediction = predict_drives(siso_frf(f=f), flat_spec(), shakers=rating,
                                peak_factor=3.0)
    force, velocity, travel = prediction.demands[1:]
    level = LEVEL * G ** 2
    grms = math.sqrt(level * 1980.0)
    assert force.rms == pytest.approx(60.0 * grms, rel=1e-3)
    assert force.peak == pytest.approx(3.0 * 60.0 * grms, rel=1e-3)
    assert force.share == pytest.approx(force.peak / 30000.0)
    vrms = math.sqrt(level / (4 * math.pi ** 2) * (1 / 20 - 1 / 2000))
    assert velocity.rms == pytest.approx(1.25 * vrms, rel=5e-3)
    assert velocity.peak == pytest.approx(1.25 * 3.0 * vrms, rel=5e-3)
    assert velocity.share == pytest.approx(velocity.peak / 1.8)
    drms = math.sqrt(level / (16 * math.pi ** 4) / 3
                     * (1 / 20 ** 3 - 1 / 2000 ** 3))
    assert travel.peak == pytest.approx(1.25 * 3.0 * drms, rel=1e-2)
    assert travel.rating == pytest.approx(0.0254)
    assert travel.share == pytest.approx(travel.peak / 0.0254)
    # the travel lives at the bottom of the band: half of the integral
    # of f^-4 from 20 Hz up lies under 20 x 2^(1/3) = 25.2 Hz. The force
    # is flat, half of it under the middle of the band, 1010 Hz
    assert travel.frequency == 20.0
    assert travel.band == pytest.approx((20.0, 20.0 * 2 ** (1 / 3)),
                                        abs=0.25)
    assert force.band == pytest.approx((20.0, 1010.0), abs=0.25)


def test_a_force_rating_without_the_moving_mass_is_not_judged_and_said():
    prediction = predict_drives(siso_frf(), flat_spec(),
                                shakers=ShakerRating(force=20000.0))
    assert [d.kind for d in prediction.demands] == ['drive']
    assert ('the force at 1Z+ is not judged: the force rating needs the '
            'moving mass') in prediction.notes


def test_a_shaker_is_read_at_the_acceleration_at_its_drives_dof():
    plant = np.array([[1.0, 0.5], [0.2, 1.0]])
    rating = ShakerRating(moving_mass=10.0)
    prediction = predict_drives(matrix_frf(plant), matrix_spec(TARGET),
                                shakers={'2Z+': rating})
    force = [d for d in prediction.demands if d.kind == 'force']
    assert [d.channel for d in force] == ['2Z+']
    expected = 10.0 * math.sqrt(2.0 * 0.01 * G ** 2 * 397 * 5.0)
    assert force[0].rms == pytest.approx(expected)


def test_a_shaker_with_no_acceleration_at_its_dof_needs_a_table():
    frf = matrix_frf(TWO)
    frf.reference_dof = ['11Z+', '12Z+'] * 2
    with pytest.raises(ValueError, match='the shaker at 11Z\\+ has no '
                       'acceleration measured at 11Z\\+; name the channel'):
        predict_drives(frf, matrix_spec(TARGET),
                       shakers=ShakerRating(moving_mass=10.0))
    prediction = predict_drives(frf, matrix_spec(TARGET),
                                shakers={'11Z+': ShakerRating(
                                    moving_mass=10.0)},
                                tables={'11Z+': '1Z+'})
    assert 'the shaker at 11Z+ is read at 1Z+' in prediction.notes


def test_another_channel_is_predicted_and_judged_against_its_limit():
    """A force gauge at the drive point, 30 N/V: 900 times the drive's
    0.01 V^2/Hz, 30 times its rms, against a 500 N limit."""
    f = lines()
    rows = np.array([np.full(len(f), 2.0 * G), np.full(len(f), 30.0)],
                    dtype=complex)
    frf = Frf(f, rows, response_dof=['1Z+', '1Z+'],
              reference_dof=['1Z+', '1Z+'],
              ordinate_dim=['acceleration/voltage', 'force/voltage'])
    prediction = predict_drives(frf, flat_spec(), limits={'1Z+ force': 500.0},
                                peak_factor=3.0)
    assert prediction.rows == ['1Z+ acceleration', '1Z+ force']
    gauge = prediction.demands[-1]
    assert gauge.kind == 'response' and gauge.dimension == 'force'
    assert gauge.rms == pytest.approx(30.0 * math.sqrt(0.01 * 397 * 5.0))
    assert gauge.share == pytest.approx(3.0 * gauge.rms / 500.0)
    assert prediction.asd[at(prediction, 100.0), 1] == pytest.approx(9.0)


def test_the_describe_lines_name_each_demand_in_words():
    from visualdynamics.units import SI

    prediction = predict_drives(siso_frf(), flat_spec(), drive_limit=30.0,
                                peak_factor=3.0)
    assert prediction.describe(SI)[0].startswith(
        '1 drive for 1 control channel: within every rating; the drive at '
        '1Z+ is nearest, at 45 % (7.0 dB of headroom)')
    assert prediction.describe(SI)[1] == (
        'drive at 1Z+: 4.46 V rms, 13.4 V expected peak against 30 V '
        '(45 %); most at 20 Hz, half from 20 to 1010 Hz')
    bare = predict_drives(siso_frf(), flat_spec())
    assert bare.fits is None and bare.first_out is None
    assert bare.describe()[0] == '1 drive for 1 control channel: no ratings given'


# ---- refusals -------------------------------------------------------------------


def _refused(match, frf=None, spec=None, **settings):
    with pytest.raises(ValueError, match=match):
        predict_drives(siso_frf() if frf is None else frf,
                       flat_spec() if spec is None else spec, **settings)


def test_the_wrong_objects_are_refused():
    with pytest.raises(TypeError, match='reads an FRF, not a Specification'):
        predict_drives(flat_spec(), flat_spec())
    with pytest.raises(TypeError, match='reads a specification, not a Frf'):
        predict_drives(siso_frf(), siso_frf())
    with pytest.raises(TypeError,
                       match='the shaker at 1Z\\+ is a ShakerRating, not a '
                             'float'):
        predict_drives(siso_frf(), flat_spec(), shakers={'1Z+': 3.0})


def test_a_specification_of_zeros_asks_for_nothing_and_is_refused():
    """A controller writes zero on the lines it does not control, so a
    target of zeros is written nowhere."""
    _refused('share no line where both are written',
             spec=flat_spec(level=0.0))


def test_undeclared_units_are_refused_on_either_side():
    f = lines()
    bare = Frf(f, np.ones((1, len(f)), complex), response_dof='1Z+',
               reference_dof='1Z+')
    _refused('the units of the FRF 1Z\\+ over 1Z\\+ are not declared',
             frf=bare)
    raw = Specification(np.array(BAND), np.array([[1.0, 1.0]]),
                        response_dof=['1Z+'])
    _refused('the units of the specification at 1Z\\+ are not declared',
             spec=raw)


def test_drives_of_two_quantities_are_refused():
    f = lines()
    frf = Frf(f, np.ones((2, len(f)), complex), response_dof=['1Z+', '1Z+'],
              reference_dof=['1Z+', '2Z+'],
              ordinate_dim=['acceleration/voltage', 'acceleration/force'])
    _refused('the FRF is against drives of force and voltage', frf=frf)


def test_a_control_channel_the_frf_does_not_hold_is_refused():
    _refused('the specification asks for 5Z\\+, and the FRF has no '
             'response there', spec=flat_spec(dof='5Z+'))
    velocity = Frf(lines(), np.ones((1, len(lines())), complex),
                   response_dof='1Z+', reference_dof='1Z+',
                   ordinate_dim='velocity/voltage')
    _refused('at 1Z\\+, the specification is of acceleration and the FRF '
             'of velocity, which cannot be compared', frf=velocity)


def test_a_missing_cross_term_or_autospectrum_is_refused():
    _refused('no cross term between 1Z\\+ and 2Z\\+: with several control '
             'channels', frf=matrix_frf(TWO),
             spec=matrix_spec(TARGET, skip={(0, 1), (1, 0)}))
    _refused('the specification has no autospectrum at 2Z\\+',
             frf=matrix_frf(TWO), spec=matrix_spec(TARGET, skip={(1, 1)}))


def test_a_control_channel_not_measured_against_every_drive_is_refused():
    frf = matrix_frf(TWO)
    keep = [0, 1, 2]                  # 2Z+ over 2Z+ left out
    partial = Frf(frf.abscissa, frf.ordinate[keep],
                  response_dof=[frf.response_dof[i] for i in keep],
                  reference_dof=[frf.reference_dof[i] for i in keep],
                  ordinate_dim='acceleration/voltage')
    _refused('the control channel 2Z\\+ is not measured against 2Z\\+',
             frf=partial, spec=matrix_spec(TARGET))


def test_no_shared_line_a_dead_plant_and_a_plant_of_nothing_are_refused():
    _refused('share no line where both are written',
             spec=flat_spec(band=(3000.0, 4000.0)))
    frf = siso_frf()
    frf.ordinate[:] = 0.0
    _refused('the FRF is zero at every line the specification is written '
             'over: the drives reach nothing', frf=frf)
    frf.ordinate[:] = np.nan
    _refused('the FRF is not a number at every line', frf=frf)
    empty = Psd(np.array(BAND), np.zeros((0, 2)), response_dof=[])
    _refused('nothing to predict: the specification holds no records',
             spec=empty)


@pytest.mark.parametrize('settings, words', [
    ({'drive_limit': -1.0},
     'the drive limit at 1Z\\+ has to be above zero, not -1'),
    ({'duration': 0.0}, 'the duration has to be above zero, not 0'),
    ({'peak_factor': np.inf},
     'the peak factor has to be above zero and finite, not inf'),
    ({'drive_limit': {'9Z+': 10.0}},
     ('the drive limit names 9Z\\+, which is not a drive of the FRF '
      '\\(1Z\\+\\)')),
    ({'limits': {'9Z+': 10.0}},
     'a limit is given at 9Z\\+, which is not a channel of the FRF'),
    ({'shakers': ShakerRating(moving_mass=1.0), 'tables': {'1Z+': '9Z+'}},
     'the table of the shaker at 1Z\\+ is 9Z\\+, which is not a channel'),
    ({'method': 'exact'}, "not 'exact'"),
])
def test_a_setting_out_of_range_is_refused_by_name(settings, words):
    _refused(words, **settings)


def test_a_table_that_is_not_acceleration_is_refused():
    f = lines()
    rows = np.array([np.full(len(f), 2.0 * G), np.full(len(f), 30.0)],
                    dtype=complex)
    frf = Frf(f, rows, response_dof=['1Z+', '7Z+'],
              reference_dof=['1Z+', '1Z+'],
              ordinate_dim=['acceleration/voltage', 'force/voltage'])
    _refused('the table of the shaker at 1Z\\+ is read in acceleration, and '
             '7Z\\+ holds force', frf=frf,
             shakers=ShakerRating(moving_mass=1.0), tables={'1Z+': '7Z+'})


# ---- the view ---------------------------------------------------------------------


def test_the_view_draws_the_drives_the_channel_and_the_shares(qt_app):
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    from visualdynamics.plot.drive_prediction import build_drive_prediction
    from visualdynamics.units import SI

    prediction = predict_drives(
        matrix_frf(TWO), matrix_spec(TARGET), drive_limit=10.0,
        method='tikhonov', weight=0.25, peak_factor=3.0,
        shakers={'1Z+': ShakerRating(moving_mass=10.0, force=500.0)})
    layout = pg.GraphicsLayoutWidget()
    try:
        drives, middle, shares = build_drive_prediction(
            layout, prediction, unit_system=SI)
        # two drives told apart by line style as well as color
        assert [curve.opts['pen'].style() for curve in
                drives.listDataItems()] == [Qt.PenStyle.SolidLine,
                                            Qt.PenStyle.DashLine]
        assert [label.text for _s, label in drives.legend.items] == [
            'drive at 1Z+', 'drive at 2Z+']
        assert drives.titleLabel.text.startswith('drive at 2Z+')
        # the control channel predicted solid, its target dashed, and the
        # lines the inverse missed hatched
        assert [curve.opts['pen'].style() for curve in
                middle.listDataItems()] == [Qt.PenStyle.SolidLine,
                                            Qt.PenStyle.DashLine]
        hatched = [item for item in middle.items
                   if isinstance(item, pg.LinearRegionItem)]
        assert len(hatched) == 1
        assert hatched[0].brush.style() == Qt.BrushStyle.BDiagPattern
        assert 'misses the target by more than 1 dB at 397 lines' in \
            middle.titleLabel.text
        assert shares.titleLabel.text == prediction.describe()[0]
        assert 'runs out first' in shares.titleLabel.text
    finally:
        layout.close()


def test_the_view_reads_another_channel_against_its_limit(qt_app):
    import pyqtgraph as pg

    from visualdynamics.plot.drive_prediction import build_drive_prediction

    f = lines()
    rows = np.array([np.full(len(f), 2.0 * G), np.full(len(f), 30.0)],
                    dtype=complex)
    frf = Frf(f, rows, response_dof=['1Z+', '3Z+'],
              reference_dof=['1Z+', '1Z+'],
              ordinate_dim=['acceleration/voltage', 'force/voltage'])
    prediction = predict_drives(frf, flat_spec(), limits={'3Z+': 50.0})
    layout = pg.GraphicsLayoutWidget()
    try:
        _drives, middle, _shares = build_drive_prediction(
            layout, prediction, channel='3Z+')
        assert middle.titleLabel.text.startswith('response at 3Z+')
        assert '(' in middle.titleLabel.text and 'over' in \
            middle.titleLabel.text
        assert len(middle.listDataItems()) == 1
        with pytest.raises(ValueError, match='9Z\\+ is not a channel'):
            build_drive_prediction(layout, prediction, channel='9Z+')
    finally:
        layout.close()


def test_the_view_draws_to_a_file(qt_app, tmp_path):
    from visualdynamics.plot import plot_drive_prediction

    path = tmp_path / 'drives.png'
    plot_drive_prediction(siso_frf(), flat_spec(), drive_limit=10.0,
                          path=str(path), show=False)
    assert path.exists() and path.stat().st_size > 2000
    with pytest.raises(ValueError, match='channel 4 is not one'):
        plot_drive_prediction(siso_frf(), flat_spec(), channel=4,
                              path=str(path), show=False)


# ---- the edges the first tests left open --------------------------------------


def test_a_singular_value_exactly_at_the_tolerance_is_kept():
    _inverse, kept = regularized_inverse(np.diag([1.0, 0.01])[None]
                                         .astype(complex), tolerance=0.01)
    assert kept.tolist() == [2]


def test_tikhonov_keeps_the_directions_whose_square_reaches_lambda():
    """A largest gain of 2 at a weight of 0.04 puts lambda at 0.16: a
    gain of 0.5 (0.25 squared) is kept, 0.3 (0.09) is not. A plant of
    zeros has nothing to scale lambda by, keeps nothing and inverts to
    zero."""
    plant = np.array([2.0, 0.5, 0.3], dtype=complex)[:, None, None]
    _inverse, kept = regularized_inverse(plant, method='tikhonov',
                                         weight=0.04)
    assert kept.tolist() == [1, 1, 0]
    inverse, kept = regularized_inverse(np.zeros((2, 2, 2), complex),
                                        method='tikhonov')
    assert kept.tolist() == [0, 0]
    assert np.all(inverse == 0.0)


def test_a_specification_on_lines_is_not_read_past_its_last_bin():
    """10 Hz lines from 20 to 1000 Hz: their bins run 15 to 1005 Hz, so
    the FRF's line at 1005 Hz is outside them, as is everything above."""
    f10 = np.arange(20.0, 1001.0, 10.0)
    spec = Specification(f10, np.full((1, len(f10)), 0.04 * G ** 2),
                         response_dof=['1Z+'],
                         ordinate_dim='acceleration**2/frequency')
    spec.interpolation = 'bin'
    prediction = predict_drives(siso_frf(), spec)
    assert prediction.judged[at(prediction, 1000.0)]
    assert not prediction.judged[at(prediction, 1005.0)]
    assert not prediction.judged[at(prediction, 1500.0)]


def test_a_controllers_zero_lines_and_its_dc_line_are_not_asked_for():
    """A target on lines from 0 Hz with zeros written where it does not
    control: the zero lines are no requirement, and the DC line, where
    a velocity and a travel have no meaning, is never predicted."""
    f = lines()
    levels = np.full(len(f), 0.04 * G ** 2)
    levels[(f > 500.0) & (f < 600.0)] = 0.0
    spec = Specification(f, levels[None], response_dof=['1Z+'],
                         ordinate_dim='acceleration**2/frequency')
    spec.interpolation = 'bin'
    prediction = predict_drives(siso_frf(), spec,
                                shakers=ShakerRating(stroke=0.05))
    assert not prediction.judged[0]
    assert not prediction.judged[at(prediction, 550.0)]
    assert prediction.judged[at(prediction, 5.0)]
    assert np.isfinite(prediction.demands[-1].peak)


def test_a_dead_channel_reads_zero_and_names_no_line():
    """A channel the drives do not reach: zero rms, a share of zero, no
    line it is most at, and no headroom to speak of."""
    f = lines()
    rows = np.array([np.full(len(f), 2.0 * G), np.zeros(len(f))],
                    dtype=complex)
    frf = Frf(f, rows, response_dof=['1Z+', '3Z+'],
              reference_dof=['1Z+', '1Z+'],
              ordinate_dim=['acceleration/voltage', 'force/voltage'])
    prediction = predict_drives(frf, flat_spec(), limits={'3Z+': 10.0})
    dead = prediction.demands[-1]
    assert dead.rms == 0.0 and dead.share == 0.0
    assert np.isnan(dead.frequency)
    assert prediction.first_out is dead
    assert prediction.headroom_db is None
    assert not dead.exceeds


def test_the_dominant_band_is_the_fewest_lines_holding_half():
    """Contributions 4, 3, 2 and 1: the first two hold 7 of 10, the
    first alone 4, so the band is the first two lines."""
    from visualdynamics.core.drive_prediction import _dominant

    f = np.array([10.0, 20.0, 30.0, 40.0])
    top, band = _dominant(f, np.array([4.0, 3.0, 2.0, 1.0]))
    assert top == 10.0
    assert band == (10.0, 20.0)
    top, band = _dominant(f, np.array([1.0, 2.0, 3.0, 6.0]))
    assert (top, band) == (40.0, (40.0, 40.0))
    # five equal lines: half is reached at the third
    top, band = _dominant(np.arange(10.0, 51.0, 10.0), np.full(5, 2.0))
    assert (top, band) == (10.0, (10.0, 30.0))


def test_a_peak_exactly_at_its_rating_is_within_it():
    first = predict_drives(siso_frf(), flat_spec(), peak_factor=3.0)
    peak = first.demands[0].peak
    at_limit = predict_drives(siso_frf(), flat_spec(), peak_factor=3.0,
                              drive_limit=peak)
    drive = at_limit.demands[0]
    assert drive.share == 1.0
    assert drive.exceeds is False
    assert at_limit.fits is True
    assert '(100 %)' in drive.describe()


def test_two_drives_and_one_channel_still_need_a_table_for_each_shaker():
    """The one control channel is the table only when there is one
    drive: with two, a drive with no acceleration at its DOF is
    refused."""
    with pytest.raises(ValueError, match='the shaker at 2Z\\+ has no '
                       'acceleration measured at 2Z\\+'):
        predict_drives(matrix_frf([[2.0, 2.0]]), flat_spec(),
                       shakers=ShakerRating(moving_mass=10.0))


def test_one_drive_away_from_its_control_point_reads_the_control_channel():
    """A drive channel at 9Z+ and the control accelerometer at 1Z+: the
    single-input test's table is its control channel."""
    prediction = predict_drives(siso_frf(reference='9Z+'), flat_spec(),
                                shakers=ShakerRating(moving_mass=10.0))
    force = prediction.demands[-1]
    assert force.kind == 'force'
    assert force.rms == pytest.approx(10.0 * math.sqrt(LEVEL * G ** 2
                                                       * 397 * 5.0))
    assert 'the shaker at 9Z+ is read at 1Z+' in prediction.notes


def test_a_square_well_conditioned_prediction_has_nothing_to_note():
    prediction = predict_drives(matrix_frf(TWO), matrix_spec(TARGET))
    assert prediction.notes == []
    k = at(prediction, 100.0)
    assert prediction.drive_asd()[k] == pytest.approx(0.015)


def test_one_ill_conditioned_line_is_counted_in_the_singular():
    frf = matrix_frf(TWO)
    k = at(predict_drives(frf, matrix_spec(TARGET)), 300.0)
    near = np.array([[1.0, 1.0], [1.0, 1.0001]]) * G
    for row, (i, j) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        frf.ordinate[row, k] = near[i, j]
    prediction = predict_drives(frf, matrix_spec(TARGET))
    assert ('the inverse set aside a direction at 1 line where the plant '
            'is ill-conditioned') in prediction.notes
    assert prediction.unmet_bands == [(300.0, 300.0)]


def test_a_row_not_measured_against_every_drive_is_left_out_and_said():
    frf = matrix_frf(TWO)
    extra = Frf(frf.abscissa,
                np.vstack([frf.ordinate,
                           np.full((1, len(frf.abscissa)), 5.0)]),
                response_dof=[*frf.response_dof, '7Z+'],
                reference_dof=[*frf.reference_dof, '1Z+'],
                ordinate_dim='acceleration/voltage')
    prediction = predict_drives(extra, matrix_spec(TARGET))
    assert prediction.notes == [
        '7Z+ not predicted: not measured against every drive']
    assert np.isnan(prediction.asd[:, 2]).all()


@pytest.mark.parametrize('notches, words', [
    (3, '100 Hz, 300 Hz, 500 Hz'),
    (4, '100 Hz, 300 Hz, 500 Hz and 1 more band'),
    (5, '100 Hz, 300 Hz, 500 Hz and 2 more bands'),
])
def test_many_unmet_bands_are_counted_past_the_third(notches, words):
    """Notches the Tikhonov inverse holds down: three named, the rest
    counted."""
    f = lines()
    gains = np.full(len(f), 2.0)
    for notch in (100.0, 300.0, 500.0, 700.0, 900.0)[:notches]:
        gains[f == notch] = 0.002
    frf = Frf(f, (gains * G)[None].astype(complex), response_dof='1Z+',
              reference_dof='1Z+', ordinate_dim='acceleration/voltage')
    held = predict_drives(frf, flat_spec(), method='tikhonov', weight=1e-4)
    assert f'target not met by the inverse within 1 dB at {words}' in \
        held.describe()


def test_a_dof_holding_two_channels_has_to_be_named_in_full():
    f = lines()
    rows = np.array([np.full(len(f), 2.0 * G), np.full(len(f), 30.0)],
                    dtype=complex)
    frf = Frf(f, rows, response_dof=['1Z+', '1Z+'],
              reference_dof=['1Z+', '1Z+'],
              ordinate_dim=['acceleration/voltage', 'force/voltage'])
    _refused('a limit is given at 1Z\\+, which is not a channel', frf=frf,
             limits={'1Z+': 10.0})


# ---- dead channels, a measured FRF at the edges, the peaks against noise -----


def test_a_dead_drive_and_a_dead_control_channel_are_named():
    dead_drive = predict_drives(matrix_frf([[1.0, 0.0], [0.5, 0.0]]),
                                matrix_spec(TARGET))
    assert ('the drive 2Z+ reaches no control channel at any line: it is '
            'predicted to do nothing') in dead_drive.notes
    dead_control = predict_drives(matrix_frf([[1.0, 0.5], [0.0, 0.0]]),
                                  matrix_spec(TARGET))
    assert ('no drive reaches the control channel 2Z+ at any line: its '
            'target cannot be met') in dead_control.notes
    assert dead_control.unmet.any()


def test_an_frf_from_a_record_with_one_nan_sample_is_refused_by_name():
    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.data import TimeHistory

    rng = np.random.default_rng(31)
    fs, n = 4096.0, 2 ** 15
    drive = rng.standard_normal(n)
    response = 2.0 * G * drive
    response[400] = np.nan
    history = TimeHistory(np.arange(n) / fs, np.vstack([drive, response]),
                          response_dof=['1Z+', '2Z+'],
                          ordinate_dim=['voltage', 'acceleration'])
    history.averaging = Averaging(frame_length=1024, overlap=0.5).filled(
        n, fs)
    frf = history.compute_frfs(references=['1Z+'])
    with pytest.raises(ValueError, match=r'not a number at every line the '
                                         r'specification is written over, '
                                         r'first at 2Z\+/1Z\+'):
        predict_drives(frf, flat_spec(dof='2Z+'))


def test_a_measured_frf_under_hann_predicts_the_drive_to_the_edges():
    """A plant estimated from records with noise on the drive and on the
    response, under the default Hann window, from a record that starts
    40 s in: the drive predicted for a flat target is the target over
    the true plant squared at every line of the band, its first and its
    last included, to the estimate's own scatter."""
    from visualdynamics.core.averaging import Averaging
    from visualdynamics.core.data import TimeHistory

    rng = np.random.default_rng(32)
    fs, n = 4096.0, 2 ** 18
    drive = rng.standard_normal(n)
    # a plant of 2 g/V with a one-sample delay: a phase that turns
    # across the band, the magnitude flat
    response = 2.0 * G * np.roll(drive, 1)
    measured_drive = drive + 0.05 * rng.standard_normal(n)
    measured_response = response + 0.05 * 2.0 * G * rng.standard_normal(n)
    history = TimeHistory(40.0 + np.arange(n) / fs,
                          np.vstack([measured_drive, measured_response]),
                          response_dof=['1Z+', '2Z+'],
                          ordinate_dim=['voltage', 'acceleration'])
    history.averaging = Averaging(frame_length=1024, overlap=0.5).filled(
        n, fs)
    assert history.averaging.window == 'hann'
    frf = history.compute_frfs(references=['1Z+'])
    response_row = [i for i in range(frf.num_records)
                    if frf.response_dof[i] == '2Z+']
    plant = Frf(frf.abscissa, frf.ordinate[response_row],
                response_dof=['2Z+'], reference_dof=['1Z+'],
                ordinate_dim=['acceleration/voltage'])
    prediction = predict_drives(plant, flat_spec(dof='2Z+'))
    f = prediction.frequency
    band = (f >= BAND[0]) & (f <= BAND[1])
    assert prediction.judged[band].all() and not prediction.judged[~band].any()
    drive_asd = prediction.drive_asd(0)
    edges = [int(np.flatnonzero(band)[0]), int(np.flatnonzero(band)[-1])]
    assert f[edges[0]] == 20.0 and f[edges[1]] == 2000.0
    expected = LEVEL / 2.0 ** 2
    np.testing.assert_allclose(drive_asd[band], expected, rtol=0.03)
    np.testing.assert_allclose(drive_asd[edges], expected, rtol=0.03)
    assert not prediction.unmet.any()


def test_the_expected_largest_peak_holds_against_simulated_noise():
    """The peak a drive is judged on, checked against what Gaussian
    noise does: twenty runs of the predicted drive, band-limited white
    noise of the predicted rms over the default duration, and the mean
    of their largest peaks of either sign against the expected largest
    peak. Counted one way, the expected peak sat 3 percent under the
    runs' mean; counted both ways it sits on it."""
    prediction = predict_drives(siso_frf(), flat_spec())
    drive = next(d for d in prediction.demands if d.kind == 'drive')
    fs = 8192.0
    n = int(fs * DURATION)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    band = (f >= BAND[0]) & (f <= BAND[1])
    peaks = []
    for seed in range(20):
        rng = np.random.default_rng(400 + seed)
        z = np.fft.rfft(rng.standard_normal(n))
        z[~band] = 0.0
        x = np.fft.irfft(z, n)
        x *= drive.rms / np.std(x)
        peaks.append(np.max(np.abs(x)))
    assert np.mean(peaks) == pytest.approx(drive.peak, rel=0.03)
    assert max(peaks) < 1.25 * drive.peak


def test_progress_is_told_and_a_raise_in_it_stops_the_prediction():
    told = []
    predict_drives(siso_frf(), flat_spec(),
                   progress=lambda done, total: told.append((done, total)))
    assert told[-1] == (4, 4)

    class Stop(Exception):
        pass

    def stop(_done, _total):
        raise Stop

    with pytest.raises(Stop):
        predict_drives(siso_frf(), flat_spec(), progress=stop)


# ---- what a mutation run found untested ----------------------------------------


def test_tikhonov_keeps_a_direction_whose_square_is_exactly_lambda():
    """A largest gain of 2 at a weight of a quarter puts lambda at 1: a
    gain of 1 reaches it exactly and is kept."""
    plant = np.array([2.0, 1.0], dtype=complex)[:, None, None]
    _inverse, kept = regularized_inverse(plant, method='tikhonov',
                                         weight=0.25)
    assert kept.tolist() == [1, 1]


def test_the_last_bin_of_a_specification_on_lines_is_its_own():
    """The FRF's line at 1000 Hz lies in the last 10 Hz bin, written at
    twice the level of the rest: the drive there is that level over the
    plant squared."""
    f10 = np.arange(20.0, 1001.0, 10.0)
    level = np.full((1, len(f10)), 0.04 * G ** 2)
    level[0, -1] *= 2.0
    spec = Specification(f10, level, response_dof=['1Z+'],
                         ordinate_dim='acceleration**2/frequency')
    spec.interpolation = 'bin'
    prediction = predict_drives(siso_frf(), spec)
    drive = prediction.drive_asd(0)
    assert drive[at(prediction, 1000.0)] == pytest.approx(0.02)
    assert drive[at(prediction, 990.0)] == pytest.approx(0.01)


def test_a_line_s_share_of_the_mean_square_is_its_density_times_its_width():
    """On lines spaced evenly in log frequency a flat drive's widest
    bin is the last, so the line contributing most is the top one."""
    f = np.geomspace(20.0, 2000.0, 200)
    prediction = predict_drives(siso_frf(f=f), flat_spec())
    assert prediction.demands[0].frequency == pytest.approx(2000.0)
