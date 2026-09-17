"""Exodus beyond the mesh: results read with the axis declared.

The file cannot say what its step axis means — a modal run, a
transient and a spectral convention all use `time_whole` — so
`load(steps=...)` declares it, and everything else follows: names
become DOFs and hints, `_RE`/`_IM` pairs go complex only on a
frequency axis, global variables ride as their own object.
"""

import numpy as np
import pytest

from visualdynamics.core.data import Spectrum, TimeHistory
from visualdynamics.io import exodus
from visualdynamics.io.exodus import _parse_variable
from visualdynamics.units import UNKNOWN

NODES = np.array([101, 102, 103])
STEPS = np.linspace(0.0, 0.5, 6)


def write_exodus(path, nodal=(), global_=(), times=STEPS):
    """A minimal exodus file: three nodes, a step axis, the variables
    asked for — built with netCDF4 the way the format documents itself,
    so the reader is held to the layout and not to our own writer."""
    import netCDF4

    def set_name(variable, i, name):
        variable[i] = np.frombuffer(name.ljust(33)[:33].encode(),
                                    dtype='S1')

    with netCDF4.Dataset(path, 'w', format='NETCDF3_CLASSIC') as ds:
        ds.createDimension('num_nodes', len(NODES))
        ds.createDimension('num_dim', 3)
        ds.createDimension('time_step', None)
        ds.createDimension('len_name', 33)
        coord = ds.createVariable('coord', 'f8', ('num_dim', 'num_nodes'))
        coord[:] = np.zeros((3, len(NODES)))
        node_map = ds.createVariable('node_num_map', 'i4', ('num_nodes',))
        node_map[:] = NODES
        whole = ds.createVariable('time_whole', 'f8', ('time_step',))
        whole[:] = times
        if nodal:
            ds.createDimension('num_nod_var', len(nodal))
            names = ds.createVariable('name_nod_var', 'S1',
                                      ('num_nod_var', 'len_name'))
            for i, (name, values) in enumerate(nodal):
                set_name(names, i, name)
                var = ds.createVariable(f'vals_nod_var{i + 1}', 'f8',
                                        ('time_step', 'num_nodes'))
                var[:] = values
        if global_:
            ds.createDimension('num_glo_var', len(global_))
            names = ds.createVariable('name_glo_var', 'S1',
                                      ('num_glo_var', 'len_name'))
            columns = np.column_stack([values for _n, values in global_])
            for i, (name, _values) in enumerate(global_):
                set_name(names, i, name)
            var = ds.createVariable('vals_glo_var', 'f8',
                                    ('time_step', 'num_glo_var'))
            var[:] = columns
    return str(path)


def ramps(scale=1.0):
    """(steps, nodes) values, distinct per node so transposes show."""
    return scale * np.arange(len(STEPS))[:, None] * (1.0 + np.arange(3))


# ---- how a variable's name reads ----------------------------------------


def test_names_read_as_direction_and_quantity():
    assert _parse_variable('AccX') == ('AccX', '', 'X+', 'acceleration')
    assert _parse_variable('vel_y') == ('vel_y', '', 'Y+', 'velocity')
    assert _parse_variable('DispZ') == ('DispZ', '', 'Z+', 'length')
    assert _parse_variable('RotX') == ('RotX', '', 'RX+', None), \
        'a rotation about x, never an x translation'
    assert _parse_variable('pressure') == ('pressure', '', '', 'pressure')
    # exact stems only: 'force_total' is a force to a human, but a
    # prefix match would also read 'velocity_error' as a velocity —
    # a hint that cannot be trusted is worse than none
    assert _parse_variable('force_total')[3] is None


def test_complex_suffixes_need_their_separator():
    """'pressure' ends in 're': a bare-suffix match would read it as
    the real half of a complex pressure."""
    assert _parse_variable('DispX_RE') == ('DispX', 're', 'X+', 'length')
    assert _parse_variable('DispX_imag') == ('DispX', 'im', 'X+', 'length')
    assert _parse_variable('pressure')[1] == ''
    assert _parse_variable('grim')[1] == ''


# ---- the axis is declared, never guessed --------------------------------


def test_a_time_reading_returns_a_time_history(tmp_path):
    path = write_exodus(tmp_path / 'run.exo',
                        nodal=[('AccX', ramps()), ('AccY', ramps(2.0))])
    result = exodus.load(path, steps='time')
    data = result['time']
    assert isinstance(data, TimeHistory)
    assert data.num_records == 6, 'two variables across three nodes'
    assert np.allclose(data.abscissa, STEPS)
    assert set(data.response_dof) == {'101X+', '102X+', '103X+',
                                      '101Y+', '102Y+', '103Y+'}
    row = data.response_dof.index('102Y+')
    assert np.allclose(data.ordinate[row], ramps(2.0)[:, 1])
    assert data.ordinate_dim[row] == UNKNOWN, 'exodus has no units'
    assert data.dimension_hint[row] == 'acceleration'


def test_displacements_on_a_time_axis_are_not_modes(tmp_path):
    """The same DispX that reads as a mode per step under the default
    reads as a transient when the caller says the axis is time."""
    path = write_exodus(tmp_path / 'run.exo',
                        nodal=[('DispX', ramps())])
    default = exodus.load(path)
    assert 'shapes' in default, "the default reading is today's"
    declared = exodus.load(path, steps='time')
    assert 'shapes' not in declared
    assert isinstance(declared['time'], TimeHistory)


def test_a_frequency_reading_pairs_re_and_im(tmp_path):
    lines = np.linspace(10.0, 60.0, 6)
    path = write_exodus(tmp_path / 'run.exo', times=lines,
                        nodal=[('DispX_RE', ramps()),
                               ('DispX_IM', ramps(3.0)),
                               ('pressure', ramps(5.0))])
    result = exodus.load(path, steps='frequency')
    data = result['spectra']
    assert isinstance(data, Spectrum)
    assert data.num_records == 6, 'three complex, three real'
    assert np.allclose(data.abscissa, lines)
    complex_row = data.response_dof.index('101X+')
    assert np.allclose(data.ordinate[complex_row],
                       ramps()[:, 0] + 1j * ramps(3.0)[:, 0])
    real_row = data.response_dof.index('101')
    assert np.allclose(data.ordinate[real_row], ramps(5.0)[:, 0])
    assert data.dimension_hint[real_row] == 'pressure'


def test_a_lone_half_stays_real(tmp_path):
    """Half a pair is not evidence of the other half."""
    lines = np.linspace(10.0, 60.0, 6)
    path = write_exodus(tmp_path / 'run.exo', times=lines,
                        nodal=[('DispX_RE', ramps())])
    data = exodus.load(path, steps='frequency')['spectra']
    assert np.allclose(np.imag(data.ordinate), 0.0)


def test_global_variables_ride_as_their_own_object(tmp_path):
    ke = np.linspace(0.0, 9.0, 6)
    path = write_exodus(tmp_path / 'run.exo',
                        nodal=[('AccX', ramps())],
                        global_=[('KE', ke), ('force', 2 * ke)])
    result = exodus.load(path, steps='time')
    extra = result['global']
    assert isinstance(extra, TimeHistory)
    assert extra.num_records == 2
    assert extra.record_label(0) == 'KE', \
        'no node, so the name is the whole identity'
    assert np.allclose(extra.ordinate[0], ke)
    assert extra.dimension_hint[1] == 'force'
    assert extra.dimension_hint[0] is None, 'KE names no known stem'


def test_nodes_narrows_the_records(tmp_path):
    path = write_exodus(tmp_path / 'run.exo',
                        nodal=[('AccX', ramps())])
    data = exodus.load(path, steps='time', nodes=[102])['time']
    assert data.response_dof == ['102X+']
    assert np.allclose(data.ordinate[0], ramps()[:, 1])


def test_refusals_say_why(tmp_path):
    path = write_exodus(tmp_path / 'bare.exo')
    with pytest.raises(ValueError, match='no nodal or global variables'):
        exodus.load(path, steps='time')
    with pytest.raises(ValueError, match="choose 'modes'"):
        exodus.load(path, steps='sideways')
    full = write_exodus(tmp_path / 'run.exo', nodal=[('AccX', ramps())])
    with pytest.raises(ValueError, match='no node'):
        exodus.load(full, steps='time', nodes=[999])


# ---- the writer, held to its own reader ---------------------------------


def plate():
    from visualdynamics.core.geometry import Geometry

    return Geometry(node_id=NODES,
                    node_xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                    length_unit='m')


def test_a_time_history_rides_out_and_back(tmp_path):
    values = ramps()
    history = TimeHistory(
        STEPS, np.vstack([values[:, 0], values[:, 1], 3 * values[:, 2]]),
        response_dof=['101X+', '102X+', '102Z+'],
        ordinate_dim=['acceleration', 'acceleration', 'force'],
        ordinate_unit=['m/s**2', 'm/s**2', 'N'])
    path = str(tmp_path / 'out.exo')
    exodus.save(history, path, geometry=plate())
    back = exodus.load(path, steps='time')['time']
    assert sorted(back.response_dof) == ['101X+', '102X+', '102Z+'], \
        'NaN fill must not come back as channels'
    for dof, hint in (('101X+', 'acceleration'), ('102Z+', 'force')):
        row = back.response_dof.index(dof)
        original = history.response_dof.index(dof)
        assert np.allclose(back.ordinate[row], history.ordinate[original])
        assert back.dimension_hint[row] == hint, \
            'the quantity survives as a hint — exodus has no units'
    assert np.allclose(back.abscissa, STEPS)


def test_a_complex_spectrum_rides_out_and_back(tmp_path):
    lines = np.linspace(10.0, 60.0, 6)
    spectrum = Spectrum(
        lines, np.vstack([ramps()[:, 0] + 1j * ramps(2.0)[:, 0]]),
        response_dof=['101X+'],
        ordinate_dim=['acceleration'], ordinate_unit=['m/s**2'])
    path = str(tmp_path / 'out.exo')
    exodus.save(spectrum, path, geometry=plate())
    back = exodus.load(path, steps='frequency')['spectra']
    row = back.response_dof.index('101X+')
    assert np.allclose(back.ordinate[row], spectrum.ordinate[0]), \
        'the _RE/_IM pair carries the phase'
    assert np.allclose(back.abscissa, lines)


def test_global_records_ride_out_and_back(tmp_path):
    ke = np.linspace(0.0, 9.0, 6)
    history = TimeHistory(STEPS, np.vstack([ke]), response_dof=[''],
                          block=['KE'])
    path = str(tmp_path / 'out.exo')
    exodus.save(history, path, geometry=plate())
    back = exodus.load(path, steps='time')['global']
    assert back.record_label(0) == 'KE'
    assert np.allclose(back.ordinate[0], ke)


def test_a_stacked_object_is_refused_with_the_collision_named(tmp_path):
    stacked = TimeHistory(STEPS, np.vstack([ramps()[:, 0]] * 2),
                          response_dof=['101X+', '101X+'],
                          block=['avg 1', 'avg 2'])
    with pytest.raises(ValueError, match='both write ValX at node 101'):
        exodus.save(stacked, str(tmp_path / 'out.exo'), geometry=plate())


def test_data_alone_is_refused(tmp_path):
    history = TimeHistory(STEPS, np.vstack([ramps()[:, 0]]),
                          response_dof=['101X+'])
    with pytest.raises(ValueError, match='needs a geometry'):
        exodus.save(history, str(tmp_path / 'out.exo'))


def test_local_frames_rotate_to_global(tmp_path):
    """A node whose displacement system is rotated 90 deg about Z: its
    local X reading must land in the file as global Y, or a reader
    with no idea the frame existed reads the wrong axis."""
    from visualdynamics.core.geometry import Geometry

    geometry = Geometry(
        node_id=NODES, node_xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        cs_id=[5], cs_matrix=[[[0, 1, 0], [-1, 0, 0], [0, 0, 1],
                               [0, 0, 0]]],
        node_disp_cs=[5, 0, 0], length_unit='m')
    wave = ramps()[:, 0]
    history = TimeHistory(
        STEPS, np.vstack([wave, np.zeros_like(wave), np.zeros_like(wave)]),
        response_dof=['101X+', '101Y+', '101Z+'],
        ordinate_dim=['acceleration'] * 3, ordinate_unit=['m/s**2'] * 3)
    path = str(tmp_path / 'out.exo')
    exodus.save(history, path, geometry=geometry)
    back = exodus.load(path, steps='time')['time']
    # rows of a cs matrix are its axes in global coordinates and a
    # local vector maps out as v @ rotation, so local X (axis (0,1,0))
    # lands exactly on global +Y
    global_x = back.ordinate[back.response_dof.index('101X+')]
    global_y = back.ordinate[back.response_dof.index('101Y+')]
    assert np.allclose(global_y, wave), \
        'the local X reading lands on global Y'
    assert np.allclose(global_x, 0.0)


# ---- the window asks; a script declares ---------------------------------


def add_node_set(path, set_id, name, node_ids):
    """A node set appended to an existing exodus fixture file."""
    import netCDF4

    with netCDF4.Dataset(path, 'a') as ds:
        ds.createDimension('num_node_sets', 1)
        ds.createDimension('num_nod_ns1', len(node_ids))
        prop = ds.createVariable('ns_prop1', 'i4', ('num_node_sets',))
        prop[:] = [set_id]
        names = ds.createVariable('ns_names', 'S1',
                                  ('num_node_sets', 'len_name'))
        names[0] = np.frombuffer(name.ljust(33)[:33].encode(), dtype='S1')
        local = {int(n): i + 1 for i, n in enumerate(NODES)}
        members = ds.createVariable('node_ns1', 'i4', ('num_nod_ns1',))
        members[:] = [local[int(n)] for n in node_ids]


def test_the_summary_names_what_there_is_to_ask_about(tmp_path):
    bare = write_exodus(tmp_path / 'bare.exo')
    summary = exodus.result_summary(bare)
    assert summary['nodal'] == [] and summary['global'] == []
    full = write_exodus(tmp_path / 'run.exo', nodal=[('AccX', ramps())],
                        global_=[('KE', np.zeros(6))])
    add_node_set(full, 7, 'accels', [101, 103])
    summary = exodus.result_summary(full)
    assert summary['nodal'] == ['AccX']
    assert summary['global'] == ['KE']
    assert summary['steps'] == 6 and summary['nodes'] == 3
    assert summary['node_sets'] == [(7, 'accels', 2)]
    assert list(exodus.node_set_nodes(full, 7)) == [101, 103]


def test_the_window_asks_and_the_answer_reaches_the_reader(
        window, pump, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    path = write_exodus(tmp_path / 'run.exo', nodal=[('AccX', ramps())])
    add_node_set(path, 7, 'accels', [101, 103])
    asked = []

    def get_item(_parent, title, _label, items, *args, **kwargs):
        asked.append(title)
        if 'nodes' in title.lower():
            return items[1], True     # the 'accels' set
        return items[1], True         # 'Time data'
    monkeypatch.setattr(QInputDialog, 'getItem', staticmethod(get_item))
    window.import_paths([path])
    pump()
    assert len(asked) == 2, 'the axis, then the nodes'
    data = window.objects['Time History']
    assert isinstance(data, TimeHistory)
    assert sorted(data.response_dof) == ['101X+', '103X+'], \
        'the chosen node set narrowed the channels'


def test_canceling_the_question_skips_the_file(window, pump, tmp_path,
                                                monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    path = write_exodus(tmp_path / 'run.exo', nodal=[('AccX', ramps())])
    monkeypatch.setattr(QInputDialog, 'getItem',
                        staticmethod(lambda *a, **k: ('', False)))
    added = window.import_paths([path])
    pump()
    assert not [name for name in added if name]
    assert 'Time History' not in window.objects


def test_a_bare_mesh_asks_nothing(window, pump, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QInputDialog, QMessageBox

    def refuse(*_a, **_k):
        raise AssertionError('a file with no results has nothing to ask')
    monkeypatch.setattr(QInputDialog, 'getItem', staticmethod(refuse))
    # and the failure dialog too: if the refusal fires, the import
    # loop catches it and reports through a modal box, which headless
    # means a hang instead of a failed assert
    monkeypatch.setattr(QMessageBox, 'warning',
                        staticmethod(lambda *_a, **_k: None))
    path = write_exodus(tmp_path / 'bare.exo')
    window.import_paths([path])
    pump()
    assert 'Geometry' in window.objects


def test_a_script_declares_what_the_window_asks(tmp_path):
    from visualdynamics import Project

    path = write_exodus(tmp_path / 'run.exo', nodal=[('AccX', ramps())])
    project = Project('exodus import')
    added = project.import_file(path, steps='time', nodes=[102])
    data = next(obj for name, obj in project.items()
                if isinstance(obj, TimeHistory))
    assert data.response_dof == ['102X+']
    assert len(added) == 2, 'the geometry and the data'


def test_parabolic_solids_and_pyramids_round_trip_exodus(tmp_path):
    """TET10, WEDGE15 and the pyramids — the transition element of
    every hex-dominant mesh — read and write by their exodus names."""
    import numpy as np

    import visualdynamics
    from visualdynamics.io import exodus

    geometry = visualdynamics.Geometry(
        node_id=list(range(1, 16)),
        node_xyz=np.random.default_rng(7).normal(size=(15, 3)),
        elem_id=[1, 2, 3],
        elem_type=[118, 113, 201],
        elem_color=[1, 1, 1],
        elem_conn=[np.arange(1, 11), np.arange(1, 16), np.arange(1, 6)])
    geometry.define_units('m')
    exodus.save(geometry, str(tmp_path / 'solids.exo'))
    back = visualdynamics.import_file(str(tmp_path / 'solids.exo'),
                                      length_unit='m')
    assert sorted(back.elem_type) == [113, 118, 201]
    by_type = {int(t): c for t, c in zip(back.elem_type, back.elem_conn)}
    assert len(by_type[118]) == 10 and len(by_type[113]) == 15
    assert len(by_type[201]) == 5, 'the pyramid survives where UNV refuses'
