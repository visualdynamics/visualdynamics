"""One controller file is one measurement, and imports as one group.

The type's expectations place a run's arrivals into the Basis, and
whatever the Basis has no slot for joins them there. A run whose
kinds the Basis already holds — the same file imported again through
a window, a second run beside the first — placed nothing, and its
arrivals were left unlinked (Brandon, 2026-09-18: "I would think they
should import as a linked set"). They link as a group of their own.
"""

from test_rattlesnake_transformation import write_run


def _groups(window):
    return [(group['role'], list(group['members'])) for group in window.project.links]


def test_a_second_run_links_as_its_own_group(window, pump, tmp_path):
    """A stream file: channel table, time history, specification. (A
    run carrying spectra places one of them into a free Basis slot
    and the rest follow it there, the older rule; that is not this
    case.)"""
    path, _raw = write_run(str(tmp_path / 'run.nc4'), spectra=False)
    window.set_project_type('Random Vibration')   # the type's slots place the first
    window.import_paths([path])
    pump()
    groups = _groups(window)
    assert len(groups) == 1 and groups[0][0] == 'Basis'
    first = groups[0][1]
    assert set(first) == set(window.objects), 'the whole run in the Basis'
    window.import_paths([path])
    pump()
    groups = _groups(window)
    assert groups[0] == ('Basis', first), 'the first keeps the Basis'
    assert groups[1] == (None, [f'{name} (2)' for name in first]), \
        'the second is its own group, whole'


def test_a_lone_arrival_is_not_a_group(window, pump, tmp_path):
    """Linking takes two: a run that brought one object stays a single
    unlinked object when the Basis has no place for it, as before."""
    import netCDF4

    path = str(tmp_path / 'package.nc4')
    with netCDF4.Dataset(path, 'w', format='NETCDF4') as ds:
        env = ds.createGroup('Random')
        env.sysid_sample_rate = 256.0
        env.sysid_frame_size = 8
        env.createDimension('fft_lines', 5)
        env.createDimension('specification_channels', 2)
        env.createDimension('drive_channels', 1)
        env.createDimension('control_channels', 2)
        env.createVariable('control_channel_indices', 'i4',
                           ('control_channels',))[...] = [0, 1]
        for name in ('frf_data_real', 'frf_data_imag'):
            env.createVariable(name, 'f8', ('fft_lines', 'specification_channels',
                                            'drive_channels'))[...] = 1.0
    window.import_paths([path])
    pump()
    window.import_paths([path])
    pump()
    assert 'FRF (2)' in window.objects
    assert all('FRF (2)' not in members for _r, members in _groups(window))


def test_a_run_into_a_project_with_no_type_waits_for_the_type(window, pump, tmp_path):
    """A run whose kind the importer cannot name sets no type, and the
    type, when it is set, builds the Basis from what is unlinked.
    Grouped on arrival, the run's objects counted as placed elsewhere
    and the Basis stayed empty beside an orange bracket (Brandon,
    2026-09-19). The run stays unlinked until the type comes."""
    from test_run_kind import _run_with_environments

    # an environment the importer can name no type for: a playback
    path = _run_with_environments(str(tmp_path / 'run.nc4'),
                                  [('Playback', 6, 'time')])
    window.import_paths([path])
    pump()
    assert window.project.project_type is None
    assert _groups(window) == [], 'nothing grouped ahead of the type'
    window.set_project_type('Random Vibration')
    pump()
    assert _groups(window) == [('Basis', ['Channel Table', 'Time History'])]
