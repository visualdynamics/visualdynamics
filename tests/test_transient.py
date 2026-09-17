"""Transient tests, which are not shock tests.

The two get called by each other's names and the difference matters. A
**shock** specification is an SRS: a required response spectrum, met by
producing *some* transient whose spectrum lands inside the band. A
**transient** specification is a waveform: this acceleration, sample by
sample, and the controller inverts the structure's transfer function to
reproduce it.

Rattlesnake can run the second today. The first it cannot — so a
transient run is not a shock test, however much its pulses look like one.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.data import ShockSpecification, TimeHistory, TransientSpecification
from visualdynamics.core.report import (
    PROJECT_TEMPLATES,
    PROJECT_TYPES,
    binding_types,
    project_expectations,
    transient_template,
)


def _target(samples=64, rate=256.0):
    t = np.arange(samples) / rate
    pulse = np.where(t < 0.02, np.sin(np.pi * t / 0.02), 0.0)
    return TransientSpecification(
        t, np.stack([pulse, 0.5 * pulse]), response_dof=['101Z+', '111Z+'],
        ordinate_dim='acceleration', ordinate_unit='m/s**2')


# ---- the object ---------------------------------------------------------


def test_a_transient_specification_is_a_time_history():
    """Because that is what it is: a waveform someone asked for."""
    target = _target()
    assert isinstance(target, TimeHistory)
    assert not isinstance(target, ShockSpecification)
    assert target.abscissa_dim == 'time'
    assert target.num_records == 2


def test_it_carries_no_limits():
    """A tolerance on a waveform is not a settled idea the way a
    tolerance on a spectrum is, and inventing one here would be
    inventing a convention rather than reading one."""
    assert not hasattr(_target(), 'limits')


def test_it_survives_a_project_file(tmp_path):
    """As itself: the native format names the class, because dataset 58
    has no code for a control target."""
    project = visualdynamics.Project('Transient')
    project.add('Target', _target())
    project.project_type = 'Transient'
    path = project.save(tmp_path / 'transient.vdyn')
    back = visualdynamics.Project.open(path)
    assert back.project_type == 'Transient'
    assert isinstance(back['Target'], TransientSpecification)
    assert np.allclose(back['Target'].ordinate, _target().ordinate)


def test_the_project_reaches_it_by_type():
    project = visualdynamics.Project()
    project.add('Target', _target())
    assert isinstance(project.transient_specification, TransientSpecification)


# ---- the project type ---------------------------------------------------


def test_transient_and_shock_are_both_types_and_are_not_each_other():
    assert 'Transient' in PROJECT_TYPES
    assert 'Shock' in PROJECT_TYPES
    assert PROJECT_TEMPLATES['Transient'] == 'transient'
    assert PROJECT_TEMPLATES['Shock'] == 'shock'


def test_each_type_expects_its_own_kind_of_target():
    transient = {name for name, *_rest in project_expectations('Transient')}
    shock = {name for name, *_rest in project_expectations('Shock')}
    assert 'Transient Specification' in transient
    assert 'Shock Specification' not in transient
    assert 'Shock Specification' in shock
    assert 'Transient Specification' not in shock
    assert 'SRS' in shock, 'a shock test is judged on a spectrum'
    assert 'SRS' not in transient, (
        'a transient is judged on its waveform and its level — a shock '
        'response spectrum is how a *shock* is judged, and expecting '
        'one here blurred the very distinction the two types draw')
    assert 'Shock Specification' in shock
    assert 'SRS Specification' not in transient


def test_a_binding_can_name_it():
    assert binding_types()['TransientSpecification'] is TransientSpecification


def test_the_report_compares_two_waveforms():
    """There is no band to fall outside and no share of a spectrum to
    count — there is a target waveform and a measured one.

    The waveform error is the transient's own, and the one thing the
    time data alone can answer. The level comes from PSDs computed off
    both sides — and a spectrum of the specification is still a
    specification, so that comparison is the same one a random test
    makes.

    What stays absent: the share of a band outside abort limits (that
    wants limits, and a target waveform carries none), and any SRS —
    that is the shock report's judgment, and its deviation bars live
    there now.
    """
    report = transient_template({})
    assert report.title == 'Transient Test Report'
    sources = [block.get('source') for block in report.blocks]
    assert '@basis:TransientSpecification' in sources
    assert '@basis:TimeHistory' in sources
    assert '@basis:Srs' not in sources
    modes = [block.get('mode') for block in report.blocks
             if block.get('kind') == 'bars']
    # the level's RMS reading follows its spectra, and the waveform
    # bars come after — spectral data before computed judgments
    # (Brandon, 2026-08-23)
    # and the kurtosis of the record itself, which every report
    # carries now (Brandon, 2026-08-24)
    assert modes == ['error', 'waveform', 'kurtosis']
    assert 'lines' not in modes, (
        'a target waveform carries no abort band to fall outside')


# ---- and the run it comes from ------------------------------------------


def test_a_transient_run_imports_its_specification():
    """Rattlesnake writes the target into the streamed file as
    `control_signal`, and it was going unread."""
    project = visualdynamics.Project()
    project.import_file(fixture_path('plate', 'shock.nc4'))
    assert project.project_type == 'Transient'


def test_the_control_signal_becomes_the_specification(tmp_path):
    """Built here rather than taken from a fixture, so the numbers going
    in are known."""
    import netCDF4

    path = tmp_path / 'transient.nc4'
    rate, samples = 256.0, 128
    signal = np.stack([np.sin(np.arange(samples) / 8.0),
                       np.cos(np.arange(samples) / 8.0)])
    with netCDF4.Dataset(path, 'w') as ds:
        ds.sample_rate = rate
        ds.createDimension('response_channels', 2)
        ds.createDimension('time_samples', samples)
        ds.createDimension('num_environments', 1)
        channels = ds.createGroup('channels')
        for name, values in (('node_number', ['101', '111']),
                             ('node_direction', ['Z+', 'Z+']),
                             ('channel_type', ['acceleration'] * 2),
                             ('unit', ['m/s^2'] * 2)):
            variable = channels.createVariable(name, str,
                                               ('response_channels',))
            for i, value in enumerate(values):
                variable[i] = value
        names = ds.createVariable('environment_names', str,
                                  ('num_environments',))
        names[0] = 'Drop'
        kinds = ds.createVariable('environment_types', str,
                                  ('num_environments',))
        kinds[0] = 'transient'
        data = ds.createVariable('time_data', 'f8',
                                 ('response_channels', 'time_samples'))
        data[:] = np.zeros((2, samples))
        group = ds.createGroup('Drop')
        group.createDimension('specification_channels', 2)
        group.createDimension('signal_samples', samples)
        control = group.createVariable(
            'control_signal', 'f8',
            ('specification_channels', 'signal_samples'))
        control[:] = signal
        indices = group.createVariable('control_channel_indices', 'i4',
                                       ('specification_channels',))
        indices[:] = [0, 1]

    loaded = visualdynamics.import_file(str(path))
    target = next(obj for obj in loaded.values()
                  if isinstance(obj, TransientSpecification))
    assert target.num_records == 2
    assert list(map(str, target.response_dof)) == ['101Z+', '111Z+']
    assert np.allclose(target.ordinate, signal)
    assert target.sample_rate == pytest.approx(rate)
    assert target.abscissa[-1] == pytest.approx((samples - 1) / rate)


def test_a_random_run_grows_no_transient_specification():
    """The reader looks for `control_signal`, which a random environment
    does not have — its target is a CPSD."""
    loaded = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))
    assert not any(isinstance(obj, TransientSpecification)
                   for obj in loaded.values())
    assert any(isinstance(obj, visualdynamics.Specification) for obj in loaded.values())
