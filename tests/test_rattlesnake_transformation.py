"""A Rattlesnake run that controlled a *virtual* response.

A random, sine or transient environment may put a response
transformation matrix over its control channels, and then everything
it computes and saves — the specification, the FRF, the coherence, the
control CPSD — is over the matrix's rows, while the streamed time data
stays in hardware channels. Brandon's file (2026-09-18): twelve
accelerometers transformed to three virtual degrees of freedom, and
the importer died on the specification with "boolean index did not
match indexed array along axis 1; size of axis is 3 but size of
corresponding boolean axis is 12". The rows are unnamed in the file
and import numbered from 1; the transformed responses come in as more
records of the same time history, after the raw channels, so one pass
of averaging gives the PSDs the specification was judged against.
"""

import numpy as np
import pytest

import visualdynamics

RATE = 256.0
SAMPLES = 512
MEASURED, DRIVES = 4, 2
CONTROL = [0, 1, 2, 3]
#: two virtual rows over four channels: their mean, and a rocking pair
MATRIX = np.array([[0.25, 0.25, 0.25, 0.25],
                   [0.5, 0.5, -0.5, -0.5]])
LINES = 5


def write_run(path, *, units=None, with_transformation=True, spectra=True,
              matrix=MATRIX):
    """A stream and its environment group in one file, the layout the
    controller writes: raw channels streamed, the specification and
    the spectra over the transformation's rows."""
    import netCDF4

    channels = MEASURED + DRIVES
    units = units or ['m/s^2'] * MEASURED + ['N'] * DRIVES
    rows = matrix.shape[0] if with_transformation else len(CONTROL)
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((channels, SAMPLES))
    with netCDF4.Dataset(path, 'w', format='NETCDF4') as ds:
        ds.sample_rate = RATE
        ds.createDimension('response_channels', channels)
        ds.createDimension('time_samples', SAMPLES)
        ds.createVariable('time_data', 'f8',
                          ('response_channels', 'time_samples'))[...] = raw
        group = ds.createGroup('channels')
        for name, values in (
                ('node_number', [str(101 + i) for i in range(channels)]),
                ('node_direction', ['Z+'] * channels),
                ('unit', units),
                ('feedback_device', [''] * MEASURED + ['Input'] * DRIVES)):
            group.createVariable(name, str, ('response_channels',))[:] = \
                np.array(values, dtype=object)
        ds.createDimension('num_environments', 1)
        ds.createVariable('environment_names', str,
                          ('num_environments',))[0] = 'Random'
        ds.createVariable('environment_types', int,
                          ('num_environments',))[0] = 0
        env = ds.createGroup('Random')
        env.samples_per_frame = 128
        env.frames_in_cpsd = 4
        env.createDimension('control_channels', len(CONTROL))
        env.createVariable('control_channel_indices', 'i4',
                           ('control_channels',))[...] = CONTROL
        if with_transformation:
            env.createDimension('response_transformation_rows', rows)
            env.createDimension('response_transformation_cols',
                                matrix.shape[1])
            env.createVariable('response_transformation_matrix', 'f8',
                               ('response_transformation_rows',
                                'response_transformation_cols'))[...] = matrix
        env.createDimension('fft_lines', LINES)
        env.createDimension('two', 2)
        env.createDimension('specification_channels', rows)
        env.createVariable('specification_frequency_lines', 'f8',
                           ('fft_lines',))[...] = np.arange(LINES) * 2.0
        cpsd = np.zeros((LINES, rows, rows))
        for k in range(rows):
            cpsd[:, k, k] = 1.0 + k
        for name, values in (('specification_cpsd_matrix_real', cpsd),
                             ('specification_cpsd_matrix_imag', 0 * cpsd)):
            env.createVariable(name, 'f8', ('fft_lines', 'specification_channels',
                                            'specification_channels'))[...] = values
        for name, factor in (('specification_warning_matrix', (0.5, 2.0)),
                             ('specification_abort_matrix', (0.25, 4.0))):
            var = env.createVariable(name, 'f8', ('two', 'fft_lines',
                                                  'specification_channels'))
            diagonal = np.stack([cpsd[:, k, k] for k in range(rows)], axis=1)
            var[0] = diagonal * factor[0]
            var[1] = diagonal * factor[1]
        if spectra:
            env.createDimension('drive_channels', DRIVES)
            for name in ('frf_data_real', 'frf_data_imag'):
                env.createVariable(name, 'f8', ('fft_lines',
                                                'specification_channels',
                                                'drive_channels'))[...] = 1.0
            env.createVariable('frf_coherence', 'f8',
                               ('fft_lines', 'specification_channels'))[...] = 0.9
            for name in ('response_cpsd_real', 'response_cpsd_imag'):
                env.createVariable(name, 'f8', ('fft_lines',
                                                'specification_channels',
                                                'specification_channels'))[...] = 2.0
    return path, raw


def test_the_run_imports_at_all(tmp_path):
    """The reported failure: twelve control channels, a three-row
    specification, and a boolean mask sized for the twelve."""
    path, _raw = write_run(str(tmp_path / 'virtual.nc4'))
    visualdynamics.import_file(path)


def test_the_specification_is_over_the_transformations_rows(tmp_path):
    path, _raw = write_run(str(tmp_path / 'virtual.nc4'))
    spec = visualdynamics.import_file(path)['Random_specification']
    assert spec.response_dof == ['1', '2'], 'rows numbered from 1'
    assert spec.ordinate_dim[0] == 'acceleration**2/frequency'
    assert spec.ordinate_unit[0] == 'm/s**2', 'the control channels share one'
    assert np.allclose(spec.ordinate[1].real, 2.0)
    assert np.allclose(spec.limits['abort_upper'][1], 8.0)


def test_the_transformed_responses_are_records_of_the_same_history(tmp_path):
    """The controller applied `T @ frame`; the same rows over the raw
    channels come in after them, in the control channels' unit, in the
    one history — so they are averaged, filtered and banded with the
    raw channels rather than processed twice (Brandon, 2026-09-18,
    reversing the separate object he first asked for)."""
    path, raw = write_run(str(tmp_path / 'virtual.nc4'))
    out = visualdynamics.import_file(path)
    assert 'Random_transformed' not in out, 'no second object'
    history = out['time_data']
    assert history.num_records == MEASURED + DRIVES + 2
    assert history.response_dof[-2:] == ['1', '2']
    assert history.ordinate_dim[-2:] == ['acceleration', 'acceleration']
    assert history.ordinate_unit[-2:] == ['m/s**2', 'm/s**2']
    assert np.allclose(history.ordinate[-2:].real, MATRIX @ raw[CONTROL])
    assert np.allclose(history.ordinate[:MEASURED + DRIVES].real, raw), \
        'the raw channels first, untouched'
    assert history.comment[-1].startswith('row 2 of the Random response transformation')
    assert history.comment[0] == ''


def test_a_transformation_over_mixed_units_imports_raw(tmp_path):
    """A row combining g and V is a number in no unit; it is imported
    undeclared rather than labelled with either."""
    path, raw = write_run(str(tmp_path / 'mixed.nc4'),
                          units=['g', 'g', 'V', 'V', 'N', 'N'])
    out = visualdynamics.import_file(path)
    history = out['time_data']
    assert history.ordinate_dim[-2:] == ['unknown', 'unknown']
    assert history.ordinate_unit[-2:] == [None, None]
    assert np.allclose(history.ordinate[-2:].real, MATRIX @ raw[CONTROL]), \
        'the raw combination, unscaled'
    spec = out['Random_specification']
    assert spec.ordinate_dim[0] == 'unknown'


def test_the_spectra_are_over_the_rows_too(tmp_path):
    path, _raw = write_run(str(tmp_path / 'virtual.nc4'))
    out = visualdynamics.import_file(path)
    frf = out['Random_frf']
    assert sorted(set(frf.response_dof)) == ['1', '2']
    assert set(frf.reference_dof) == {'105Z+', '106Z+'}, 'the drives, by table'
    assert frf.ordinate_dim[0] == 'acceleration/force'
    coherence = out['Random_coherence']
    assert coherence.response_dof == ['1', '2']
    cpsd = out['Random_response_cpsd']
    assert cpsd.response_dof == ['1', '1', '2', '2']


def test_a_transformation_that_does_not_fit_its_channels_is_refused(tmp_path):
    """A matrix with three columns over four control channels is the
    shape of a mistake, said by name rather than mis-multiplied."""
    path, _raw = write_run(str(tmp_path / 'bad.nc4'), matrix=MATRIX[:, :3])
    with pytest.raises(ValueError, match=r'\(2, 3\) but the environment controls 4'):
        visualdynamics.import_file(path)


def test_a_run_without_a_transformation_is_as_it_was(tmp_path):
    path, _raw = write_run(str(tmp_path / 'plain.nc4'), with_transformation=False)
    out = visualdynamics.import_file(path)
    assert out['time_data'].num_records == MEASURED + DRIVES
    assert out['Random_specification'].response_dof[:2] == ['101Z+', '102Z+']


def test_a_system_id_package_numbers_the_rows(tmp_path):
    """A package saved from a transformed environment holds the FRF
    over the rows; its control channel indices name more channels
    than the FRF has rows, and the rows are numbered from 1."""
    import netCDF4

    path = str(tmp_path / 'package.nc4')
    rows = MATRIX.shape[0]
    with netCDF4.Dataset(path, 'w', format='NETCDF4') as ds:
        env = ds.createGroup('Random')
        env.sysid_sample_rate = RATE
        env.sysid_frame_size = 8
        env.createDimension('fft_lines', LINES)
        env.createDimension('specification_channels', rows)
        env.createDimension('drive_channels', DRIVES)
        env.createDimension('control_channels', len(CONTROL))
        env.createVariable('control_channel_indices', 'i4',
                           ('control_channels',))[...] = CONTROL
        env.createDimension('response_transformation_rows', rows)
        env.createDimension('response_transformation_cols', len(CONTROL))
        env.createVariable('response_transformation_matrix', 'f8',
                           ('response_transformation_rows',
                            'response_transformation_cols'))[...] = MATRIX
        for name in ('frf_data_real', 'frf_data_imag'):
            env.createVariable(name, 'f8', ('fft_lines', 'specification_channels',
                                            'drive_channels'))[...] = 1.0
    frf = visualdynamics.import_file(path)['Random_frf']
    assert sorted(set(frf.response_dof)) == ['1', '2']
    assert frf.num_records == rows * DRIVES
