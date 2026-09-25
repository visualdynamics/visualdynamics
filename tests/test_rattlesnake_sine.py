"""The rattlesnake importer reads a sine environment's tone set.

Fixtures are synthesized in the controller's own layout — the one the
Phase A runs pinned (environment group, `specifications` subgroup per
tone, padded segment arrays, (breakpoint, lower/upper, left/right,
channel) band arrays) — with planted values the assertions name.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.sine import SineSweepSpecification
from visualdynamics.io import rattlesnake


def _write_run(path, sine=True, random=False, band=True,
               split_band=False, sine_transformation=None):
    import netCDF4 as nc

    ds = nc.Dataset(str(path), 'w')
    n_channels = 3
    ds.file_version = '3.0.0'
    ds.sample_rate = 4096.0
    ds.createDimension('response_channels', n_channels)
    ds.createDimension('time_samples', 4096)
    time_data = ds.createVariable('time_data', 'f8',
                                  ('response_channels', 'time_samples'))
    time_data[...] = np.zeros((n_channels, 4096))

    ch = ds.createGroup('channels')
    ch.createDimension('n', n_channels)
    values = {'node_number': ['101', '104', '901'],
              'node_direction': ['Z+', 'Z+', 'Z+'],
              'unit': ['m/s^2', 'm/s^2', 'N'],
              'channel_type': ['Acceleration', 'Acceleration', 'Force'],
              'feedback_device': ['', '', 'Input']}
    for name, column in values.items():
        var = ch.createVariable(name, str, ('n',))
        for i, value in enumerate(column):
            var[i] = value

    environments = []
    if sine:
        environments.append(('Sine', 3))
    if random:
        environments.append(('Random', 1))
    ds.createDimension('num_environments', len(environments))
    names = ds.createVariable('environment_names', str,
                              ('num_environments',))
    types = ds.createVariable('environment_types', int,
                              ('num_environments',))
    for i, (name, code) in enumerate(environments):
        names[i] = name
        types[i] = code

    if sine:
        g = ds.createGroup('Sine')
        # the sysid attributes every sysid-capable environment writes:
        # they describe the plant-measurement phase, and the importer
        # must not read them as the recording's averaging
        g.sysid_frame_size = 1024
        g.sysid_averages = 20
        g.sysid_overlap = 0.5
        g.sysid_window = 'Hann'
        g.createDimension('control_channels', 2)
        var = g.createVariable('control_channel_indices', 'i4',
                               ('control_channels',))
        var[...] = [0, 1]
        # the sweep's target per specification channel: the control
        # channels, or the rows of a response transformation over them,
        # laid out as the controller writes it
        base_row = [2.0, 4.0]
        if sine_transformation is not None:
            matrix = np.asarray(sine_transformation, dtype=float)
            g.createDimension('response_transformation_rows', matrix.shape[0])
            g.createDimension('response_transformation_cols', matrix.shape[1])
            g.createVariable(
                'response_transformation_matrix', 'f8',
                ('response_transformation_rows',
                 'response_transformation_cols'))[...] = matrix
            base_row = [3.0] * matrix.shape[0]
        n_spec = len(base_row)
        specs = g.createGroup('specifications')
        tone = specs.createGroup('Sweep Up')
        tone.start_time = 2.0
        tone.createDimension('num_breakpoints', 2)
        tone.createDimension('specification_channels', n_spec)
        tone.createDimension('two', 2)
        tone.createVariable('spec_frequency', 'f8',
                            ('num_breakpoints',))[...] = [100.0, 800.0]
        tone.createVariable(
            'spec_amplitude', 'f8',
            ('num_breakpoints', 'specification_channels'))[...] = \
            [base_row, base_row]
        tone.createVariable(
            'spec_phase', 'f8',
            ('num_breakpoints', 'specification_channels'))[...] = 0.0
        # padded to breakpoint length, trailing entry dead — the
        # leading-rate convention the controller writes
        tone.createVariable('spec_sweep_type', 'i1',
                            ('num_breakpoints',))[...] = [0, 1]
        tone.createVariable('spec_sweep_rate', 'f8',
                            ('num_breakpoints',))[...] = [50.0, 99.0]
        for kind in ('warning', 'abort'):
            var = tone.createVariable(
                f'spec_{kind}', 'f8',
                ('num_breakpoints', 'two', 'two',
                 'specification_channels'))
            values = np.full((2, 2, 2, n_spec), np.nan)
            if band:
                width = 3.0 if kind == 'warning' else 6.0
                base = np.array([base_row, base_row])
                values[:, 0, :, :] = (base
                                      * 10 ** (-width / 20))[:, None, :]
                values[:, 1, :, :] = (base
                                      * 10 ** (width / 20))[:, None, :]
                if split_band:
                    values[1, 1, 1, :] *= 2.0    # right side disagrees
            var[...] = values

    if random:
        g = ds.createGroup('Random')
        g.samples_per_frame = 512
        g.frames_in_cpsd = 6
        g.cpsd_overlap = 0.5
        g.cpsd_window = 'hann'
        g.createDimension('control_channels', 2)
        g.createDimension('lines', 4)
        var = g.createVariable('control_channel_indices', 'i4',
                               ('control_channels',))
        var[...] = [0, 1]
        g.createVariable('specification_frequency_lines', 'f8',
                         ('lines',))[...] = [50.0, 100.0, 200.0, 400.0]
        for name in ('specification_cpsd_matrix_real',
                     'specification_cpsd_matrix_imag'):
            var = g.createVariable(
                name, 'f8', ('lines', 'control_channels',
                             'control_channels'))
            var[...] = np.broadcast_to(
                np.eye(2) * 1e-3 if 'real' in name else np.zeros((2, 2)),
                (4, 2, 2))
    ds.close()
    return str(path)


def test_a_sine_run_imports_its_tone_set(tmp_path):
    path = _write_run(tmp_path / 'sine.nc4')
    out = visualdynamics.import_file(path)
    spec = out['Sine_specification']
    assert isinstance(spec, SineSweepSpecification)
    assert spec.response_dof == ['101Z+', '104Z+']
    tone = spec.tone('Sweep Up')
    assert tone.start_time == 2.0
    assert list(tone.segment_rate) == [50.0], \
        'the padded trailing entry is dead and stays behind'
    assert tone.duration() == pytest.approx(14.0)
    assert tone.amplitude[0, 1] == pytest.approx(4.0)
    assert tone.limits['warning_upper'][0, 0] == \
        pytest.approx(2.0 * 10 ** (3 / 20))
    assert tone.limits['abort_lower'][0, 1] == \
        pytest.approx(4.0 * 10 ** (-6 / 20))
    assert spec.ordinate_unit == 'm/s**2'
    assert spec.ordinate_dim == 'acceleration'


def test_the_run_declares_itself_a_sine_sweep(tmp_path):
    path = _write_run(tmp_path / 'sine.nc4')
    assert rattlesnake.run_kind(path) == 'sine'
    assert rattlesnake.project_type(path) == 'Sine Sweep'
    project = visualdynamics.Project('run')
    project.import_file(path)
    assert project.project_type == 'Sine Sweep'
    assert project.sine_sweep_specification is not None


def test_a_mixed_run_imports_both_specifications(tmp_path):
    path = _write_run(tmp_path / 'mixed.nc4', random=True)
    assert rattlesnake.run_kind(path) == 'mixed'
    assert rattlesnake.project_type(path) == 'Random and Sine', \
        'the one mixed run that is a type of its own'
    out = visualdynamics.import_file(path)
    assert isinstance(out['Sine_specification'], SineSweepSpecification)
    assert type(out['Random_specification']).__name__ == 'Specification'


def test_a_sweep_over_a_virtual_point_beside_a_plain_random(tmp_path):
    """Brandon's run (2026-09-24): the sine sweep controlled a virtual
    point through a response transformation and the random controlled
    the raw channels. The sine specification is over the matrix's rows,
    and was scaled by the control channels' scales instead — one per
    channel against one per row — so the file did not load at all.

    It is over the rows now, numbered as the random reads them; the
    rows' time histories ride the recording beside the raw channels,
    which is what the sweep's levels are read from."""
    matrix = [[0.5, 0.5]]
    path = _write_run(tmp_path / 'mixed.nc4', random=True,
                      sine_transformation=matrix)
    out = visualdynamics.import_file(path)
    sine = out['Sine_specification']
    assert sine.response_dof == ['1'], 'the row, not the channels'
    tone = sine.tone('Sweep Up')
    assert tone.amplitude.shape == (2, 1)
    assert tone.amplitude[0, 0] == pytest.approx(3.0)
    assert tone.limits['warning_upper'][0, 0] == \
        pytest.approx(3.0 * 10 ** (3 / 20))
    assert sine.ordinate_unit == 'm/s**2', \
        'the row shares its control channels\' unit'
    assert out['Random_specification'].response_dof[:2] == \
        ['101Z+', '104Z+'], 'the random still controls the raw channels'
    history = out['time_data']
    assert '1' in history.response_dof, 'the row is recorded as well'


def test_control_channels_are_marked_from_the_sine_spec(tmp_path):
    path = _write_run(tmp_path / 'sine.nc4')
    out = visualdynamics.import_file(path)
    table = out['channel_table']
    controls = dict(zip(table.dof_strings(), table.controls()))
    assert controls['101Z+'] and controls['104Z+']
    assert not controls['901Z+'], 'the drive is not a control channel'


def test_a_bandless_tone_carries_no_limits(tmp_path):
    path = _write_run(tmp_path / 'sine.nc4', band=False)
    out = visualdynamics.import_file(path)
    assert not out['Sine_specification'].tone('Sweep Up').limits


def test_split_band_sides_refuse_by_name(tmp_path):
    path = _write_run(tmp_path / 'sine.nc4', split_band=True)
    with pytest.raises(ValueError, match='left and right'):
        visualdynamics.import_file(path)


def test_a_sine_run_carries_no_averaging_frames(tmp_path):
    """The sweep is read sample by sample through a tracking filter;
    the only frame-shaped attributes in a sine run's file are the
    sysid ones, which describe the plant measurement, not the sweep —
    shading their frames on the recording claimed an analysis that
    never happens."""
    path = _write_run(tmp_path / 'sine.nc4')
    out = visualdynamics.import_file(path)
    assert out['time_data'].averaging is None


def test_a_mixed_run_keeps_the_random_loops_own_frames(tmp_path):
    """And the sine group's sysid attributes must not shadow them,
    whatever order the file lists its groups."""
    path = _write_run(tmp_path / 'mixed.nc4', random=True)
    out = visualdynamics.import_file(path)
    averaging = out['time_data'].averaging
    assert averaging is not None
    assert averaging.frame_length == 512, \
        "the random loop's frames, not the sysid phase's 1024"
    assert averaging.window == 'hann'
