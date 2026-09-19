"""Reading real Rattlesnake output.

`plate/random.nc4` and `modal.nc4` came out of the Rattlesnake
controller actually running — see `generate_plate_runs.py` in the
visualdynamics-generators repository. Every
other file in testdata/ visualdynamics either wrote itself or synthesized from sdynpy,
so these are the only check that the `.nc4` reader copes with a real file
rather than an idea of one.

They are also the only unit-bearing files here. A Rattlesnake channel table
carries engineering units per channel, so these import fully defined and the
user is never asked — which is the whole point of the format being read at
all.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics

RANDOM = fixture_path('plate', 'random.nc4')
MODAL = fixture_path('plate', 'modal.nc4')


@pytest.fixture(params=[RANDOM, MODAL], ids=['random', 'modal'])
def run(request):
    return visualdynamics.import_file(request.param)


def test_a_rattlesnake_run_imports_with_its_units_already_known(run):
    """No Define Units step: the file says what its channels are in."""
    data = run['time_data']
    assert data.units_defined
    assert data.undefined_records == []
    assert set(data.ordinate_dim) == {'acceleration', 'force'}
    assert set(data.ordinate_unit) == {'m/s**2', 'N'}


def test_the_accelerometers_and_the_load_cells_are_told_apart(run):
    """One object, two quantities — which is why units are per record."""
    data = run['time_data']
    accelerations = [i for i, dim in enumerate(data.ordinate_dim)
                     if dim == 'acceleration']
    forces = [i for i, dim in enumerate(data.ordinate_dim) if dim == 'force']
    assert accelerations and forces
    # the drives come last, as the channel table orders them
    assert forces == list(range(data.num_records - len(forces),
                                data.num_records))


def test_the_channel_table_comes_across_whole(run):
    table = run['channel_table']
    data = run['time_data']
    assert table.num_channels == data.num_records
    assert set(table['unit']) == {'m/s^2', 'N'}


def test_the_dofs_are_real_airplane_nodes(run):
    """Node numbers and directions survive, so this lands on the geometry."""
    from visualdynamics.core.data import parse_dof

    geometry = visualdynamics.import_file(fixture_path('plate', 'geometry.npz'))
    nodes = set(geometry.node_id.tolist())
    for dof in run['time_data'].response_dof:
        node, direction = parse_dof(dof)
        assert node in nodes, dof
        assert direction in ('X+', 'Y+', 'Z+')


def test_there_is_actually_data_in_it(run):
    """A streamed file can be structurally perfect and hold no samples —
    which is what a frequency-spacing mismatch between the system ID and the
    specification produced, silently, for a whole run."""
    data = run['time_data']
    assert data.ordinate.shape[1] > 1000
    assert np.all(np.isfinite(data.ordinate))
    assert (np.abs(data.ordinate).max(axis=1) > 0).all(), 'a dead channel'


def test_the_random_run_carries_the_specification_it_controlled_to():
    """Rattlesnake stores the target CPSD in the environment group."""
    spec = visualdynamics.import_file(RANDOM)['Random_specification']
    assert spec.units_defined
    assert set(spec.ordinate_dim) == {'acceleration**2/frequency'}
    assert spec.num_records == 8, 'one per control channel'


def test_the_random_run_actually_converged_on_that_specification():
    """Otherwise this is a recording of a controller failing, which is
    not the realistic test case it is here to be. Judged as a run, not
    as one channel: the ensemble median must sit on the specification,
    and no single channel may be wildly off — a channel a few dB out
    is what the compliance report exists to show, not a broken
    fixture."""
    run = visualdynamics.import_file(RANDOM)
    data, spec = run['time_data'], run['Random_specification']
    step = data.abscissa[1] - data.abscissa[0]
    samples = 2048
    window = np.hanning(samples)
    frequency = np.fft.rfftfreq(samples, step)
    band = (frequency >= 60) & (frequency <= 1400)
    spec_dofs = {dof: i for i, dof in enumerate(spec.response_dof)}
    medians = []
    for row, dof in enumerate(data.response_dof):
        if dof not in spec_dofs or data.ordinate_dim[row] != 'acceleration':
            continue
        # measure against the tail, after the loop has settled
        tail = data.ordinate[row][data.ordinate.shape[1] // 2:]
        frames = [tail[i:i + samples]
                  for i in range(0, len(tail) - samples, samples // 2)]
        psd = np.mean([np.abs(np.fft.rfft(frame * window)) ** 2
                       for frame in frames], axis=0)
        psd *= 2 * step / (window ** 2).sum()
        target = np.interp(frequency, spec.abscissa,
                           spec.ordinate[spec_dofs[dof]].real)
        medians.append(float(np.median(
            10 * np.log10(psd[band] / target[band]))))
    assert len(medians) == spec.num_records, 'every control channel judged'
    ensemble = float(np.median(medians))
    assert abs(ensemble) < 2.0, f'{ensemble:+.2f} dB off as a run'
    assert max(abs(m) for m in medians) < 5.0, (
        f'worst channel {max(medians, key=abs):+.2f} dB — a compliance '
        'story, not a convergence one, up to a point')


def test_a_modal_run_measures_more_than_it_drives():
    """A modal survey covers the structure; the drives are a couple of
    shakers. Both directions of the survey are measured."""
    data = visualdynamics.import_file(MODAL)['time_data']
    forces = sum(dim == 'force' for dim in data.ordinate_dim)
    assert data.num_records - forces > 4 * forces
    assert {dof[-2:] for dof in data.response_dof} > {'Z+'}, 'not only Z'


# ---- the specification, and the band around it ------------------------------

def test_a_specification_is_a_psd_with_bounds():
    """It is a PSD in every respect — abscissa, quantity, conversions — with
    four more curves saying how far the response may stray. So it inherits."""
    from visualdynamics.core.data import Psd, Specification

    spec = visualdynamics.import_file(RANDOM)['Random_specification']
    assert isinstance(spec, Specification)
    assert isinstance(spec, Psd), 'a specification is a kind of PSD'
    assert spec.units_defined
    assert set(spec.ordinate_dim) == {'acceleration**2/frequency'}


def test_the_warning_and_abort_limits_come_across():
    """They are in the file; visualdynamics used to drop them."""
    spec = visualdynamics.import_file(RANDOM)['Random_specification']
    assert spec.has_limits
    assert set(spec.limits) == {'warning_lower', 'warning_upper',
                                'abort_lower', 'abort_upper'}
    for values in spec.limits.values():
        assert values.shape == spec.ordinate.shape


@pytest.mark.parametrize('bound,decibels', [
    ('warning_lower', -3.0), ('warning_upper', 3.0),
    ('abort_lower', -6.0), ('abort_upper', 6.0)])
def test_each_limit_sits_where_it_was_written(bound, decibels):
    """The generator asked for +/-3 dB warning and +/-6 dB abort, and lower
    is index 0 of rattlesnake's matrix — which is how its own plotting reads
    it, and the one thing that would silently swap the band."""
    spec = visualdynamics.import_file(RANDOM)['Random_specification']
    target = spec.ordinate[0].real
    inside = target > 0
    ratio = spec.limits[bound][0][inside] / target[inside]
    assert 10 * np.log10(np.median(ratio)) == pytest.approx(decibels, abs=0.1)


def test_the_limits_convert_with_the_ordinate():
    """A limit is the same quantity in the same units, so a declaration that
    rescales the ordinate must rescale the bounds by exactly as much —
    otherwise a spec shown in g would be bounded in m/s^2."""
    spec = visualdynamics.import_file(RANDOM)['Random_specification']
    before = spec.ordinate[0].real / spec.limits['abort_upper'][0]
    spec.define_units('g')                      # reinterpret, do not rescale
    after = spec.ordinate[0].real / spec.limits['abort_upper'][0]
    assert np.allclose(before, after, equal_nan=True), 'the band moved'
    assert spec.ordinate_unit[0] == 'g'


def test_withdrawing_the_units_restores_the_limits_too():
    spec = visualdynamics.import_file(RANDOM)['Random_specification']
    raw_ordinate = spec.ordinate.copy()
    raw_limits = {k: v.copy() for k, v in spec.limits.items()}
    spec.define_units('g')
    spec.undefine_units()
    assert np.allclose(spec.ordinate, raw_ordinate)
    for name, values in raw_limits.items():
        assert np.allclose(spec.limits[name], values, equal_nan=True), name


def test_a_limit_can_be_taken_as_a_psd_of_its_own():
    """For plotting or exporting one on its own terms."""
    from visualdynamics.core.data import Psd

    spec = visualdynamics.import_file(RANDOM)['Random_specification']
    upper = spec.limit('abort_upper')
    assert isinstance(upper, Psd)
    assert np.allclose(upper.ordinate, spec.limits['abort_upper'],
                       equal_nan=True)
    assert upper.abscissa is spec.abscissa or np.allclose(upper.abscissa,
                                                          spec.abscissa)
    assert spec.limit('not a bound') is None


def test_a_specification_survives_the_native_format(tmp_path):
    """It shares the PSD function type, so the file has to name the class."""
    from visualdynamics.core.data import Specification

    spec = visualdynamics.import_file(RANDOM)['Random_specification']
    path = str(tmp_path / 'spec.vdyn')
    spec.save(path)
    back = visualdynamics.load(path)
    assert isinstance(back, Specification)
    assert back == spec
    assert set(back.limits) == set(spec.limits)


def test_cross_spectral_records_have_no_limits_of_their_own():
    """Limits are per control channel. A cross term gets NaN, which plots as
    a gap rather than as a line along zero."""
    spec = visualdynamics.import_file(RANDOM, full_cpsd=True)['Random_specification']
    autos = [i for i, (r, c) in
             enumerate(zip(spec.response_dof, spec.reference_dof)) if r == c]
    crosses = [i for i in range(spec.num_records) if i not in autos]
    assert autos and crosses
    upper = spec.limits['abort_upper']
    # an auto row is partly NaN too — rattlesnake leaves the lines outside
    # the control band unbounded — so the test is that it has a band at all
    assert all(np.isfinite(upper[i]).any() for i in autos)
    assert np.isnan(upper[crosses]).all()


# ---- the other save: what the environment computed ---------------------------

RANDOM_SPECTRA = fixture_path('plate', 'random_spectra.nc4')
MODAL_SPECTRA = fixture_path('plate', 'modal_spectra.nc4')


@pytest.fixture(params=[RANDOM_SPECTRA, MODAL_SPECTRA],
                ids=['random', 'modal'])
def spectra(request):
    return visualdynamics.import_file(request.param), request.param


def test_a_spectral_save_is_recognized_without_any_time_data():
    """A sniffer that insists on time data rejects half of what rattlesnake
    writes. The random environment's spectral save has none — it carries a
    `time_samples` dimension of zero, which its metadata writer puts there,
    and an empty history is not a history.

    Not parametrized over both any more: a *modal* spectral save does carry
    time data, and it is not a recording but the stack of accepted average
    frames. See `test_a_spectral_saves_time_data_splits_into_its_captures`.
    """
    out = visualdynamics.import_file(RANDOM_SPECTRA)
    assert 'time_data' not in out
    assert 'channel_table' in out


def test_a_modal_spectral_save_does_carry_its_frames():
    """The other half of the pair above, so neither file's shape is assumed
    of the other."""
    out = visualdynamics.import_file(MODAL_SPECTRA)
    assert 'time_data' in out
    assert out['time_data'].block is not None, 'a stack, keyed by average'


def test_the_spectra_arrive_unit_aware(spectra):
    """The channel table is what carries the units, and rattlesnake's own
    save drops it — the generator puts it back for exactly this reason."""
    out, _path = spectra
    environment = 'Random' if 'random' in _path else 'Modal'
    frf = out[f'{environment}_frf']
    assert frf.units_defined
    assert 'acceleration/force' in set(frf.ordinate_dim)
    assert 'm/s**2' in set(frf.ordinate_unit)
    assert 'N' in set(frf.reference_unit)


def test_coherence_comes_across_as_its_own_type(spectra):
    """And as the *right* one. Rattlesnake reports multiple coherence — one
    curve per response with the references summed over — which UFF calls 26,
    not the ordinary type 6 that promises a reference DOF per record."""
    from visualdynamics.core.data import Coherence, MultipleCoherence

    out, path = spectra
    environment = 'Random' if 'random' in path else 'Modal'
    coherence = out[f'{environment}_coherence']
    assert isinstance(coherence, MultipleCoherence)
    assert not isinstance(coherence, Coherence)
    assert coherence.reference_dof is None
    assert coherence.units_defined, 'a ratio needs no unit declared'
    assert set(coherence.ordinate_dim) == {'dimensionless'}


def test_the_coherence_is_a_measurement_and_not_noise(spectra):
    """Two shakers on a clean virtual plant: coherence should sit near one
    across the band, dipping only at anti-resonances."""
    out, path = spectra
    environment = 'Random' if 'random' in path else 'Modal'
    values = out[f'{environment}_coherence'].ordinate.real
    assert values.min() >= 0.0 and values.max() <= 1.0 + 1e-9
    frequency = out[f'{environment}_coherence'].abscissa
    # from below the first mode to the band edge. The plate's sparse
    # low spectrum has deep anti-resonance valleys where coherence
    # honestly dips (and the survey's in-plane channel measures
    # nothing at all), so the bar sits lower than a dense structure's
    # would — the sys-id-instead-of-measurement bug this guards
    # against reads as noise, nowhere near it
    band = values[:, (frequency >= 400) & (frequency <= 1400)]
    assert np.median(band) > 0.75, f'median {np.median(band):.3f}'


def test_a_coherence_plots_linear_against_its_own_bounds():
    """A 0..1 ratio says nothing on a log axis, which is what every other
    frequency-domain array gets."""
    from visualdynamics.core.data import Coherence

    assert Coherence.log_ordinate is False
    assert Coherence.ordinate_limits == (0.0, 1.05)


def test_the_random_spectral_save_carries_the_cross_spectra_too():
    """Its environment writes response and drive CPSDs beside the FRFs."""
    out = visualdynamics.import_file(RANDOM_SPECTRA)
    response = out['Random_response_cpsd']
    drive = out['Random_drive_cpsd']
    assert set(response.ordinate_dim) == {'acceleration**2/frequency'}
    assert set(drive.ordinate_dim) == {'force**2/frequency'}
    # the whole matrix, not its diagonal: 4 shakers is 16 terms
    assert drive.num_records == 16, 'every shaker against every shaker'
    assert len(set(drive.reference_dof)) == 4


def test_the_saved_spectra_are_the_measurement_and_not_the_system_id():
    """The check nobody had, and the one that matters.

    Rattlesnake's `SAVE_CONTROL_DATA` writes the system identification —
    measured once before the profile at its own excitation — under names
    that read like control data, and for a while the fixture was that.
    Every existing test passed: the shapes, units and dimensions were all
    correct, and the numbers were 82 dB from the truth.

    So tie the two files together. The response CPSD integrated over the
    band is an RMS, and the streamed time history at the same level has one
    too; nothing but the real measurement makes those agree.
    """
    time = visualdynamics.import_file(RANDOM)['time_data']
    cpsd = visualdynamics.import_file(RANDOM_SPECTRA)['Random_response_cpsd']
    df = float(cpsd.abscissa[1] - cpsd.abscissa[0])
    autos = {response: i for i, (response, reference)
             in enumerate(zip(cpsd.response_dof, cpsd.reference_dof))
             if response == reference}
    # the streamed file spans both levels; its last third is at full level
    tail = time.ordinate[:, -time.ordinate.shape[1] // 3:].real
    for row, dof in enumerate(time.response_dof):
        if dof not in autos:
            continue
        if time.ordinate_dim[row] != 'acceleration':
            # both drives are control channels here, so their force
            # records share a DOF name with an acceleration CPSD —
            # comparing those was newtons against a (m/s^2)^2 spectrum
            continue
        measured = float(np.sqrt(np.mean(tail[row] ** 2)))
        integrated = float(np.sqrt(
            np.sum(cpsd.ordinate[autos[dof]].real.clip(0)) * df))
        assert 0.5 < integrated / measured < 2.0, (
            f'{dof}: {integrated:.4f} from the spectra against '
            f'{measured:.4f} from the time data')


def test_the_modal_run_was_burst_random_with_twenty_averages():
    """Burst random is only correct as a set: the excitation, a rectangle
    window because the burst windows itself, and no overlap because one
    frame is one burst. The file records all three, so the fixture can be
    asked what it is rather than trusted to be what the generator meant."""
    import netCDF4

    with netCDF4.Dataset(MODAL) as handle:
        group = handle.groups['Modal']
        assert group.signal_generator_type == 'burst'
        assert group.num_averages == 20
        assert group.frf_window == 'rectangle'
        assert group.overlap == 0.0


def test_a_modal_frf_covers_every_response_against_every_drive():
    out = visualdynamics.import_file(MODAL_SPECTRA)
    frf = out['Modal_frf']
    table = out['channel_table']
    drives = 2
    assert frf.num_records == (table.num_channels - drives) * drives
    assert len(set(frf.reference_dof)) == drives


def test_the_drives_are_not_reported_as_responses_against_themselves():
    """A modal environment counts every enabled channel as a response and
    never excludes its references, so the saved matrix carries each drive
    *force* against the drives: |H| exactly 1 against itself and
    numerically zero against the other, coherence exactly 1. Arithmetic,
    not measurement — the importer drops those rows. The drive DOFs do
    appear as responses (the survey measures an accelerometer there:
    the drive points), but only ever as accelerations."""
    out = visualdynamics.import_file(MODAL_SPECTRA)
    frf, coherence = out['Modal_frf'], out['Modal_coherence']
    drives = set(frf.reference_dof)
    assert drives, 'the file does name references'
    assert set(frf.ordinate_dim) == {'acceleration/force'}, 'no force/force'
    assert drives <= set(frf.response_dof), (
        'the drive points are measured — as accelerations')
    assert drives <= set(coherence.response_dof)
    assert frf.num_records == \
        len(set(frf.response_dof)) * len(drives), (
        'every response against every drive, force rows dropped')


def test_those_rows_really_are_trivial():
    """The reason for dropping them, checked against the file rather than
    asserted: if they ever carried information this would fail."""
    import netCDF4

    with netCDF4.Dataset(MODAL_SPECTRA) as handle:
        group = handle.groups['Modal']
        matrix = (np.asarray(group['frf_data_real'][()])
                  + 1j * np.asarray(group['frf_data_imag'][()]))
        responses = np.asarray(group['response_channel_indices'][()])
        references = np.asarray(group['reference_channel_indices'][()])
    band = slice(20, 200)
    for j, reference in enumerate(references):
        i = int(np.where(responses == reference)[0][0])
        against_itself = np.abs(matrix[band, i, j])
        assert np.allclose(against_itself, 1.0), 'not an identity after all'
    other = np.abs(matrix[band, int(np.where(responses == references[0])[0][0]), 1])
    assert np.nanmax(other) < 1e-9, 'the drives are not uncorrelated'



# ---- a stack of captures is not a recording ---------------------------------

def write_nc4(path, samples, *, spectral, per_frame=None, averages=None,
              channels=2, units=None):
    """A minimal Rattlesnake-shaped file, to isolate what the split keys on."""
    import netCDF4

    units = units or ['m/s^2'] * channels
    with netCDF4.Dataset(path, 'w', format='NETCDF4') as ds:
        ds.sample_rate = 256.0
        ds.createDimension('response_channels', channels)
        ds.createDimension('time_samples', samples)
        ds.createVariable('time_data', 'f8',
                          ('response_channels', 'time_samples'))[...] = (
            np.arange(channels * samples, dtype=float).reshape(channels, samples))
        group = ds.createGroup('channels')
        for name, values in (('node_number', [str(100 + i) for i in range(channels)]),
                             ('node_direction', ['Z+'] * channels),
                             ('unit', units)):
            group.createVariable(name, str, ('response_channels',))[:] = \
                np.array(values, dtype=object)
        env = ds.createGroup('Modal')
        if per_frame is not None:
            env.samples_per_frame = per_frame
            env.num_averages = averages
        if spectral:
            env.createDimension('fft_lines', 3)
            env.createDimension('response_channels', channels)
            env.createDimension('reference_channels', 1)
            for name in ('frf_data_real', 'frf_data_imag'):
                env.createVariable(name, 'f8', ('fft_lines', 'response_channels',
                                                'reference_channels'))[...] = 0.0
    return path


def test_the_drives_come_from_the_channel_table_not_from_elimination(tmp_path):
    """A random environment names its control channels and not its drives.

    Inferring the drives as "every channel that is not a control channel"
    reads correctly only while the file holds nothing but controls and
    drives. Instrument more than you control — which is how a real random
    test is run — and it returns hundreds, and the FRF slicing walks off the
    end of its drive axis. The channel table marks a drive with a feedback
    device; that is the answer.
    """
    import netCDF4

    controls, measured, drives = 2, 5, 2
    channels = measured + drives
    path = str(tmp_path / 'random.nc4')
    with netCDF4.Dataset(path, 'w', format='NETCDF4') as ds:
        ds.sample_rate = 256.0
        ds.createDimension('response_channels', channels)
        group = ds.createGroup('channels')
        for name, values in (
                ('node_number', [str(100 + i) for i in range(channels)]),
                ('node_direction', ['Z+'] * channels),
                ('unit', ['m/s^2'] * measured + ['N'] * drives),
                ('feedback_device', [''] * measured + ['Input'] * drives)):
            group.createVariable(name, str, ('response_channels',))[:] = \
                np.array(values, dtype=object)
        env = ds.createGroup('Random')
        env.createDimension('fft_lines', 3)
        env.createDimension('specification_channels', controls)
        env.createDimension('drive_channels', drives)
        env.createVariable('control_channel_indices', 'i4',
                           ('specification_channels',))[...] = range(controls)
        for name in ('frf_data_real', 'frf_data_imag'):
            env.createVariable(name, 'f8', ('fft_lines',
                                            'specification_channels',
                                            'drive_channels'))[...] = 1.0
    frf = visualdynamics.import_file(path)['Random_frf']
    assert frf.num_records == controls * drives
    assert set(frf.reference_dof) == {'105Z+', '106Z+'}, 'the feedback pair'


def test_a_spectral_saves_time_data_splits_into_its_captures(tmp_path):
    """Its UI appends one accepted average's frame at a time, so the samples
    are separate captures laid end to end."""
    path = write_nc4(str(tmp_path / 'stack.nc4'), 40, spectral=True,
                     per_frame=10, averages=4)
    data = visualdynamics.import_file(path)['time_data']
    assert data.num_records == 8, '2 channels x 4 averages'
    assert len(data.abscissa) == 10, 'each capture starts again at zero'
    assert data.block == ['avg 1', 'avg 2', 'avg 3', 'avg 4'] * 2
    assert data.record_label(1) == '100Z+ avg 2'


def test_the_split_copies_nothing(tmp_path):
    """Splitting is a reshape of a contiguous array. If it ever becomes a
    copy, a 50%-overlap run doubles its memory for no reason."""
    path = write_nc4(str(tmp_path / 'stack.nc4'), 40, spectral=True,
                     per_frame=10, averages=4)
    data = visualdynamics.import_file(path)['time_data']
    assert data.ordinate.base is not None


def test_each_record_is_the_capture_it_claims_to_be(tmp_path):
    path = write_nc4(str(tmp_path / 'stack.nc4'), 40, spectral=True,
                     per_frame=10, averages=4)
    data = visualdynamics.import_file(path)['time_data']
    raw = np.arange(2 * 40, dtype=float).reshape(2, 40)
    for channel in range(2):
        for average in range(4):
            record = channel * 4 + average
            assert np.allclose(data.ordinate[record].real,
                               raw[channel, average * 10:(average + 1) * 10])


def test_a_streamed_recording_is_left_whole(tmp_path):
    """No spectral results means nothing framed it, so the samples are one
    continuous span however they happen to divide."""
    path = write_nc4(str(tmp_path / 'stream.nc4'), 40, spectral=False,
                     per_frame=10, averages=4)
    data = visualdynamics.import_file(path)['time_data']
    assert data.num_records == 2 and len(data.abscissa) == 40
    assert data.block is None


def test_a_channel_tables_units_are_read_in_the_case_they_were_typed(tmp_path):
    """A real run's table said `G` for its accelerometers and `V` for its
    voltage channels; the accelerometers arrived unit-less and the
    voltages did not (Brandon, 2026-09-17). pint reads `G` as gauss.
    The importer reads the table's case as the typist's, and the record
    carries the toolset's own spelling."""
    path = write_nc4(str(tmp_path / 'typed.nc4'), 40, spectral=False,
                     units=['G', 'Volts'])
    data = visualdynamics.import_file(path)['time_data']
    assert data.ordinate_dim == ['acceleration', 'voltage']
    assert data.ordinate_unit == ['g', 'volts']
    raw = np.arange(2 * 40, dtype=float).reshape(2, 40)
    assert np.allclose(data.ordinate[0].real, raw[0] * 9.80665)
    assert np.allclose(data.ordinate[1].real, raw[1])


def test_a_sample_count_that_divides_by_luck_is_not_a_stack(tmp_path):
    """A recording can be an exact multiple of the frame size by accident;
    that alone must not be read as a stack of captures."""
    path = write_nc4(str(tmp_path / 'stream.nc4'), 40, spectral=False)
    data = visualdynamics.import_file(path)['time_data']
    assert data.block is None


def test_a_stack_expands_into_a_grid_of_channels_by_averages(tmp_path, window,
                                                             pump):
    from visualdynamics.gui.record_grid import grid_axes

    path = write_nc4(str(tmp_path / 'stack.nc4'), 40, spectral=True,
                     per_frame=10, averages=4)
    window.import_paths([path])
    name = next(key for key in window.objects if key.endswith('Time History'))
    responses, columns = grid_axes(window.objects[name])
    assert len(responses) == 2 and columns == ['avg 1', 'avg 2', 'avg 3', 'avg 4']
    grid = window.record_grids[name]
    grid.item(1, 2).setSelected(True)
    pump()
    assert grid.selected_records() == [6], 'channel 2, average 3'


def test_a_target_on_the_controllers_lines_reads_as_a_density_per_line():
    """A narrowband specification on one-hertz lines drew as the law
    between its points; it is a density per line and steps, like the
    octave-band one (Brandon, 2026-09-19). A written breakpoint curve —
    a few unevenly spaced points — is still the law."""
    import numpy as np

    from visualdynamics.core.data import Specification
    from visualdynamics.plot import drawing_shape

    spec = visualdynamics.import_file(fixture_path('plate', 'random.nc4'))['Random_specification']
    assert spec.interpolation == 'bin', 'the controller\'s lines'
    assert spec.bandwidth is None, 'and not banded: the plain specification still'
    assert drawing_shape(spec) == 'steps'
    assert Specification.reading_of(np.array([20.0, 80.0, 800.0, 2000.0])) == 'log_log'
    assert Specification.reading_of(np.arange(1.0, 2001.0)) == 'bin'
    assert Specification.reading_of(np.linspace(10.0, 20.0, 5)) == 'log_log', \
        'too few to be lines, however even'
