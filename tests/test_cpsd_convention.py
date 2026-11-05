"""A cross term means the same thing inside the program and in the file.

This package stores a CPSD as conj(X)·Y (`compute_cpsds`). Rattlesnake
and sdynpy hold X·conj(Y), the elementwise conjugate, and neither file
says which it holds. Until 2026-11-04 every one of these crossings wrote
and read the values unconverted, so the controller was handed each cross
term with its phase reversed.

A round trip cannot catch this: the writer and the reader had the same
gap, and the two errors cancelled. So these tests pin what is *in the
file* against a phase worked out by hand, or against a matrix formed
outside the code being tested. Reported by a session driving the plate
through the controller, where the file's phases were what the plate got.
"""

from __future__ import annotations

import shutil

import netCDF4
import numpy as np
import pytest
from conftest import fixture_path
from scipy.linalg import eigh

import visualdynamics
from visualdynamics.core import modal_response as mr
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import Frf, Psd, TimeHistory
from visualdynamics.core.shapes import ShapeSet
from visualdynamics.core.transform import to_physical
from visualdynamics.io import rattlesnake_specification, sdynpy_data

RATE, TONE = 1024.0, 100.0
CHANNELS = ['1X+', '2X+']


def _second_leads_first():
    """Channel 2 leads channel 1 by 90 degrees: sin and cos of one tone."""
    t = np.arange(8192) / RATE
    return np.array([np.sin(2 * np.pi * TONE * t),
                     np.cos(2 * np.pi * TONE * t)]), t


def _cpsds():
    signal, t = _second_leads_first()
    return TimeHistory(t, signal, response_dof=CHANNELS).compute_cpsds()


def _degrees(values, abscissa):
    """The phase in degrees at the tone's line."""
    line = int(np.argmin(np.abs(np.asarray(abscissa) - TONE)))
    return np.degrees(np.angle(values[line]))


def _pair(data, response, reference):
    return next(i for i in range(data.num_records)
                if data.response_dof[i] == response
                and data.reference_dof[i] == reference)


def test_the_package_holds_the_first_channel_conjugated():
    """What every other test here leans on, by hand: conj(X1)·X2 with X2
    leading X1 by 90 degrees is +90, and X1·conj(X2) would be -90."""
    cpsd = _cpsds()
    ordinate = cpsd.ordinate[_pair(cpsd, '1X+', '2X+')]
    assert _degrees(ordinate, cpsd.abscissa) == pytest.approx(90.0, abs=1e-6)


def test_the_specification_file_holds_the_controllers_form(tmp_path):
    cpsd = _cpsds()
    path = tmp_path / 'spec.npz'
    rattlesnake_specification.save(cpsd.as_specification(), path)
    with np.load(path) as d:
        phase = _degrees(d['cpsd'][:, 0, 1], d['f'])
        lower = _degrees(d['cpsd'][:, 1, 0], d['f'])
    assert phase == pytest.approx(-90.0, abs=1e-6)
    assert lower == pytest.approx(90.0, abs=1e-6)


def test_a_filled_half_of_the_specification_is_in_the_controllers_form_too(
        tmp_path):
    """The half the object did not hold is filled from the half it did,
    and has to land on the right side of the conversion."""
    cpsd = _cpsds()
    keep = [_pair(cpsd, '1X+', '1X+'), _pair(cpsd, '2X+', '2X+'),
            _pair(cpsd, '1X+', '2X+')]
    held = Psd(cpsd.abscissa, cpsd.ordinate[keep],
               response_dof=[cpsd.response_dof[i] for i in keep],
               reference_dof=[cpsd.reference_dof[i] for i in keep])
    path = tmp_path / 'spec.npz'
    rattlesnake_specification.save(held.as_specification(), path)
    with np.load(path) as d:
        assert _degrees(d['cpsd'][:, 0, 1], d['f']) == pytest.approx(
            -90.0, abs=1e-6)
        assert _degrees(d['cpsd'][:, 1, 0], d['f']) == pytest.approx(
            90.0, abs=1e-6)


def test_a_modal_response_exports_as_the_controller_forms_it(tmp_path):
    """Known H and S_ff, through Compute Modal Responses and Expand: the
    file holds H S_ff Hᴴ, with H the modal sum in +jωt, formed here by
    hand, which is the law the controller runs by."""
    mass = np.diag([2.0, 1.0, 1.5, 1.0])
    stiffness = 1e5 * np.array([[1.0, -1.0, 0.0, 0.0], [-1.0, 3.0, -2.0, 0.0],
                                [0.0, -2.0, 3.0, -1.0], [0.0, 0.0, -1.0, 1.0]])
    dofs, drives = ['1X+', '2X+', '3X+', '4X+'], ['1X+', '3X+']
    lines = np.linspace(10.0, 200.0, 96)
    lam, phi = eigh(stiffness, mass)
    damping = np.array([0.01, 0.02, 0.03, 0.01])
    omega_n = np.sqrt(np.clip(lam, 0.0, None))   # the rigid mode can land below 0
    shapes = ShapeSet(omega_n / (2 * np.pi), damping, dofs, phi.T,
                      mass_unit='kg')
    cross = 0.7 * np.exp(1j * lines / 40.0)
    forces = np.empty((lines.size, 2, 2), dtype=complex)
    forces[:, 0, 0], forces[:, 1, 1] = 1.0, 2.0
    forces[:, 0, 1], forces[:, 1, 0] = cross, np.conj(cross)
    # the force density as this package holds it, conj(F_a)·F_b
    stored = np.conj(forces)
    density = Psd(lines,
                  np.array([stored[:, a, b] for a in range(2)
                            for b in range(2)]),
                  response_dof=[drives[a] for a in range(2) for _ in range(2)],
                  reference_dof=[drives[b] for _ in range(2)
                                 for b in range(2)],
                  ordinate_dim=['force**2/frequency' if a == b
                                else 'force*force/frequency'
                                for a in range(2) for b in range(2)])
    modal = mr.modal_response_density(density, shapes)
    physical, _report = to_physical(modal, shapes)
    path = tmp_path / 'spec.npz'
    rattlesnake_specification.save(physical.as_specification(), path)
    with np.load(path) as d:
        written, channels = d['cpsd'], [str(c) for c in d['dof']]
    order = [dofs.index(c) for c in channels]
    omega = 2 * np.pi * lines
    drive = [dofs.index(d) for d in drives]
    h = np.zeros((lines.size, len(dofs), 2), dtype=complex)
    for k, w in enumerate(omega):
        receptance = 1.0 / (omega_n ** 2 - w ** 2
                            + 2j * damping * omega_n * w)
        h[k] = -w ** 2 * (phi * receptance) @ phi[drive].T
    # S_ff here is in the controller's form: the density above is its
    # conjugate, so this is exactly what the controller would compute
    want = h @ forces @ h.conj().transpose(0, 2, 1)
    want = want[:, order][:, :, order]
    scale = np.abs(want).max()
    assert np.abs(written - want).max() < 1e-10 * scale
    assert np.abs(written - np.conj(want)).max() > 1e-3 * scale, (
        'the cross terms must carry phase, or this test cannot tell')


def test_a_specification_file_comes_back_as_it_went_out(tmp_path):
    """Not the bug's test (it passed with the bug), but the round trip
    has to survive the fix."""
    spec = _cpsds().as_specification()
    path = tmp_path / 'spec.npz'
    rattlesnake_specification.save(spec, path)
    back = rattlesnake_specification.load(path)
    for i in range(spec.num_records):
        at = _pair(back, spec.response_dof[i], spec.reference_dof[i])
        assert np.allclose(back.ordinate[at], spec.ordinate[i])


def test_a_file_in_the_controllers_form_reads_in_this_packages(tmp_path):
    """The controller's own target: entry (1, 2) is X1·conj(X2)."""
    frequencies = np.array([TONE])
    matrix = np.ones((1, 2, 2), dtype=complex)
    matrix[0, 0, 1], matrix[0, 1, 0] = -1j, 1j
    path = tmp_path / 'theirs.npz'
    np.savez(path, f=frequencies, cpsd=matrix, dof=np.array(CHANNELS))
    spec = rattlesnake_specification.load(path)
    assert np.degrees(np.angle(
        spec.ordinate[_pair(spec, '1X+', '2X+'), 0])) == pytest.approx(90.0)


@pytest.mark.parametrize('name', ['response_cpsd', 'drive_cpsd'])
def test_a_recorded_cpsd_reads_in_this_packages_form(tmp_path, name):
    """A cross term the controller recorded as X1·conj(X2), written into a
    copy of a real run: it reads back as `compute_cpsds` would have it."""
    copy = tmp_path / 'random_spectra.nc4'
    shutil.copy(fixture_path('plate', 'random_spectra.nc4'), copy)
    with netCDF4.Dataset(copy, 'a') as ds:
        group = next(g for g in ds.groups.values()
                     if f'{name}_real' in g.variables)
        environment = next(key for key, g in ds.groups.items() if g is group)
        real = group.variables[f'{name}_real']
        imag = group.variables[f'{name}_imag']
        values = np.zeros(real.shape)
        real[:] = values
        values[:, 0, 1] = -1.0                 # entry (1, 2): -90 degrees
        values[:, 1, 0] = 1.0
        imag[:] = values
    data = visualdynamics.import_file(copy)[f'{environment}_{name}']
    channels = list(dict.fromkeys(data.response_dof))
    ordinate = data.ordinate[_pair(data, channels[0], channels[1])]
    assert np.degrees(np.angle(ordinate[1])) == pytest.approx(90.0)
    ordinate = data.ordinate[_pair(data, channels[1], channels[0])]
    assert np.degrees(np.angle(ordinate[1])) == pytest.approx(-90.0)


def test_the_specification_in_a_run_reads_in_this_packages_form(tmp_path):
    copy = tmp_path / 'random.nc4'
    shutil.copy(fixture_path('plate', 'random.nc4'), copy)
    with netCDF4.Dataset(copy, 'a') as ds:
        group = next(g for g in ds.groups.values()
                     if 'specification_cpsd_matrix_real' in g.variables)
        real = group.variables['specification_cpsd_matrix_real']
        imag = group.variables['specification_cpsd_matrix_imag']
        values = np.zeros(imag.shape)
        values[:, 0, 1], values[:, 1, 0] = -1.0, 1.0
        imag[:] = values
        matrix = np.zeros(real.shape)
        matrix[:, range(real.shape[1]), range(real.shape[1])] = 1.0
        real[:] = matrix
    spec = next(v for k, v in visualdynamics.import_file(copy).items()
                if k.endswith('_specification'))
    channels = list(dict.fromkeys(spec.response_dof))
    ordinate = spec.ordinate[_pair(spec, channels[0], channels[1])]
    assert np.degrees(np.angle(ordinate[1])) == pytest.approx(90.0)


def test_the_recorded_cross_terms_agree_with_the_stream_of_the_same_run():
    """The real measurement, with no number of ours in it: the controller's
    saved response CPSD against `compute_cpsds` on its own streamed time
    data, at the same frame length. The two are not the same frames, and
    the agreement is weak (a median correlation near 0.35), so this asks
    only the sign: each cross term agrees better as read than conjugated.
    """
    saved = visualdynamics.import_file(
        fixture_path('plate', 'random_spectra.nc4'))['Random_response_cpsd']
    stream = visualdynamics.import_file(
        fixture_path('plate', 'random.nc4'))['time_data']
    channels = list(dict.fromkeys(saved.response_dof))
    keep = [stream.response_dof.index(c) for c in channels
            if stream.ordinate_dim[stream.response_dof.index(c)]
            == 'acceleration']
    tail = stream.ordinate[keep, -stream.ordinate.shape[1] // 3:].real
    dt = float(stream.abscissa[1] - stream.abscissa[0])
    names = [stream.response_dof[k] for k in keep]
    computed = TimeHistory(
        dt * np.arange(tail.shape[1]), tail, response_dof=names,
    ).compute_cpsds(Averaging(frame_length=2048, overlap=0.5, window='hann'))
    assert np.allclose(computed.abscissa, saved.abscissa)
    better, total = 0, 0
    for a in names:
        for b in names:
            if a == b:
                continue
            theirs = saved.ordinate[_pair(saved, a, b)]
            ours = computed.ordinate[_pair(computed, a, b)]
            total += 1
            better += (np.real(np.vdot(ours, theirs))
                       > np.real(np.vdot(ours, np.conj(theirs))))
    assert total == 56
    assert better == total, f'{better} of {total} cross terms'


def test_an_sdynpy_file_holds_a_cross_term_in_sdynpys_form(tmp_path):
    cpsd = _cpsds()
    path = tmp_path / 'cpsd.npz'
    sdynpy_data.save(cpsd, path)
    with np.load(path, allow_pickle=True) as d:
        records = d['data']
    at = _pair(cpsd, '1X+', '2X+')
    assert _degrees(records['ordinate'][at], cpsd.abscissa) == pytest.approx(
        -90.0, abs=1e-6)


def test_a_cross_term_in_an_sdynpy_file_reads_in_this_packages_form(tmp_path):
    cpsd = _cpsds()
    path = tmp_path / 'cpsd.npz'
    sdynpy_data.save(cpsd, path)
    with np.load(path, allow_pickle=True) as d:
        records, function_type = d['data'], d['function_type']
    records['ordinate'] = np.conj(cpsd.ordinate)     # sdynpy's X·conj(Y)
    np.savez(path, data=records, function_type=function_type)
    back = sdynpy_data.load(path)
    assert np.allclose(back.ordinate, cpsd.ordinate)


def test_sdynpy_data_that_is_not_a_density_is_left_alone(tmp_path):
    """An FRF has a phase of its own, and a conjugate there is a new bug."""
    lines = np.array([1.0, 2.0, 3.0])
    frf = Frf(
        lines, np.array([1j * lines]), response_dof=['1X+'],
        reference_dof=['2X+'])
    path = tmp_path / 'frf.npz'
    sdynpy_data.save(frf, path)
    with np.load(path, allow_pickle=True) as d:
        assert np.allclose(d['data']['ordinate'][0], 1j * lines)
    assert np.allclose(sdynpy_data.load(path).ordinate, frf.ordinate)
