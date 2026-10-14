"""MATLAB `.mat` files are the project file in MATLAB's container.

A `.mat` written here holds exactly what a `.vdyn` holds, so the test
that matters is the one below: the frozen corpus, which exercises
every object kind and every optional field, goes out as `.mat`, comes
back, and the two projects write byte-for-byte the same `.vdyn` tree.
The rest pins what MATLAB sees, the refusals, and the doors.
"""

import os

import h5py
import numpy as np
import pytest
from conftest import fixture_path
from scipy.io import loadmat, savemat

import visualdynamics
from visualdynamics.io import export_file, import_file, matlab, native

CORPUS = str(fixture_path('vdyn_corpus', 'schema1.vdyn'))


def vdyn_tree(path):
    """Every attribute and dataset in a .vdyn, keyed by path."""
    out = {}
    with h5py.File(path) as f:
        def visit(name, obj):
            for key, value in obj.attrs.items():
                out[f'{name}@{key}'] = value
            if isinstance(obj, h5py.Dataset):
                out[name] = obj[()]
        f.visititems(visit)
        for key, value in f.attrs.items():
            out[f'@{key}'] = value
    return out


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False
        if a.dtype.kind in 'OSU' or b.dtype.kind in 'OSU':
            return [text(x) for x in a.ravel()] == [text(x) for x in b.ravel()]
        return a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, (str, bytes)) or isinstance(b, (str, bytes)):
        return text(a) == text(b)
    return type(a) is type(b) and a == b


# ---- the round trip is the format ------------------------------------------

def test_the_whole_corpus_survives_matlab_exactly(tmp_path):
    """Every object kind and every optional field the schema has: the
    project that comes back writes the identical .vdyn tree — names,
    kinds, attributes, datasets, dtypes, shapes, strings and all. This
    is the claim that the .mat *is* the project file."""
    project = native.load(CORPUS)
    mat = str(tmp_path / 'corpus.mat')
    export_file(project, mat)
    back = import_file(mat)
    assert isinstance(back, native.Project)
    assert back.name == project.name
    assert back.project_type == project.project_type
    assert back.active_geometry == project.active_geometry
    assert list(back) == list(project)
    project.save(str(tmp_path / 'a.vdyn'))
    back.save(str(tmp_path / 'b.vdyn'))
    a, b = vdyn_tree(tmp_path / 'a.vdyn'), vdyn_tree(tmp_path / 'b.vdyn')
    assert set(a) == set(b), (set(a) ^ set(b))
    differing = [key for key in a if not same(a[key], b[key])]
    assert not differing, differing


def test_a_single_object_goes_out_and_comes_back(tmp_path):
    """One object saved alone is one struct named for its kind — the
    same rule the .vdyn loader recognises a file by."""
    frf = visualdynamics.import_file(fixture_path('plate', 'frfs.npz'),
                                     response_unit='m/s**2',
                                     reference_unit='N')
    path = str(tmp_path / 'frf.mat')
    export_file(frf, path)
    variables = loadmat(path, squeeze_me=True, struct_as_record=False)
    assert 'data' in variables and 'visualdynamics_schema' in variables
    back = import_file(path)
    assert type(back).__name__ == 'Frf'
    assert np.array_equal(back.ordinate, frf.ordinate)
    assert back.response_dof == frf.response_dof
    assert back.ordinate_unit == frf.ordinate_unit
    assert back.reference_unit == frf.reference_unit


def test_a_one_line_geometry_keeps_its_one_line(tmp_path):
    """A cell holding one entry of four nodes came back as four entries
    of one node, because numpy makes a matrix of a list of equal-length
    arrays; the cell is built by hand now. The line is a block of
    three beams since 2026-09-30, and comes back as that chain."""
    from visualdynamics.core.geometry import Geometry

    geometry = Geometry(node_id=[1, 2, 3, 4],
                        node_xyz=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    geometry.add_beams([1, 2, 3, 4])
    path = str(tmp_path / 'g.mat')
    export_file(geometry, path)
    back = import_file(path)
    assert back == geometry
    assert [line['chains'] for line in back.drawn_lines()] == [[[1, 2, 3, 4]]]


def test_a_one_channel_sine_tone_keeps_its_matrix(tmp_path):
    """MATLAB has no one-dimensional array, so a matrix with one column
    would read back as a vector; the struct that holds one names it."""
    project = native.load(CORPUS)
    spec = project.sine_sweep_specification
    path = str(tmp_path / 'sine.mat')
    export_file(spec, path)
    raw = loadmat(path, squeeze_me=False, struct_as_record=False)
    tone = raw['sine_specification'][0, 0].tones.ravel()[0][0, 0]
    assert tone.amplitude.ndim == 2 and tone.amplitude.shape[1] == 1
    assert 'amplitude' in [str(x[0]) for x in tone.matrix_fields.ravel()]
    back = import_file(path)
    assert back.tones[0].amplitude.shape == spec.tones[0].amplitude.shape


# ---- what MATLAB sees -------------------------------------------------------

def test_matlab_sees_structs_cells_and_columns(tmp_path):
    project = native.load(CORPUS)
    path = str(tmp_path / 'corpus.mat')
    export_file(project, path)
    variables = loadmat(path, squeeze_me=True, struct_as_record=False,
                        chars_as_strings=True)
    assert variables['visualdynamics_schema'] == native.SCHEMA_VERSION
    assert variables['test_name'] == project.name
    objects = variables['objects']
    assert len(objects) == len(project)
    assert [o.name for o in objects] == list(project)
    geometry = next(o for o in objects if o.kind == 'geometry')
    # ragged connectivity is a cell, not a flat-plus-offsets pair
    assert not hasattr(geometry, 'elem_conn_flat')
    assert hasattr(geometry, 'elem_conn')
    data = next(o for o in objects if o.kind == 'data')
    raw = loadmat(path, squeeze_me=False, struct_as_record=False)
    raw_data = next(o for o in raw['objects'].ravel()
                    if o[0, 0].kind == 'data')[0, 0]
    assert raw_data.abscissa.shape[1] == 1, 'a vector is a column'
    assert raw_data.ordinate.ndim == 2, 'records × samples stays a matrix'
    assert list(data.response_dof) == project[data.name].response_dof


def test_values_are_in_the_display_units_and_labeled_with_them(tmp_path):
    """Brandon (2026-10-13): a time history declared in g, exported with
    in-slinch-lbf-s (g) on, came out in m/s² beside an ordinate_unit of
    'g'. Now its numbers are in g and say so, the file names the system,
    and it reads back to the same SI values."""
    from visualdynamics.core.data import TimeHistory
    from visualdynamics.units import SYSTEMS

    history = TimeHistory(np.arange(5) * 0.01, [[0.0, 1.0, -2.0, 0.5, 3.0]], ['1X+'])
    history.define_units('g')
    path = str(tmp_path / 'history.mat')
    export_file(history, path, unit_system=SYSTEMS['in-slinch-lbf-s (g)'])
    variables = loadmat(path, squeeze_me=True, struct_as_record=False)
    assert np.allclose(variables['data'].ordinate, [0.0, 1.0, -2.0, 0.5, 3.0])
    assert variables['data'].ordinate_unit == 'g'
    assert variables['unit_system'] == 'in-slinch-lbf-s (g)'
    back = import_file(path)
    assert np.allclose(back.ordinate, history.ordinate, rtol=1e-12)
    assert back.ordinate_unit == ['g'] and back.ordinate_dim == ['acceleration']


def test_an_frf_is_written_in_the_systems_units_over_its_reference(tmp_path):
    """Declared in m/s² per N, written in in/s² per lbf: the numbers the
    view shows, each half of the unit named."""
    from visualdynamics.units import SYSTEMS

    system = SYSTEMS['in-slinch-lbf-s']
    frf = visualdynamics.import_file(fixture_path('plate', 'frfs.npz'),
                                     response_unit='m/s**2',
                                     reference_unit='N')
    path = str(tmp_path / 'frf.mat')
    export_file(frf, path, unit_system=system)
    variables = loadmat(path, squeeze_me=True, struct_as_record=False)
    assert np.allclose(variables['data'].ordinate, frf.display_ordinate(system))
    assert not np.allclose(variables['data'].ordinate, frf.ordinate)
    assert variables['data'].ordinate_unit[0] == 'in/s**2'
    assert variables['data'].reference_unit[0] == 'lbf'
    back = import_file(path)
    assert np.allclose(back.ordinate, frf.ordinate, rtol=1e-12)


def test_without_a_display_system_the_file_is_the_project_file(tmp_path):
    """SI exactly, with the declared units, and no unit_system: what
    Project.save writes, and how a file from before 2026-10-13 reads."""
    frf = visualdynamics.import_file(fixture_path('plate', 'frfs.npz'),
                                     response_unit='m/s**2',
                                     reference_unit='N')
    path = str(tmp_path / 'frf.mat')
    export_file(frf, path)
    variables = loadmat(path, squeeze_me=True, struct_as_record=False)
    assert 'unit_system' not in variables
    assert np.array_equal(variables['data'].ordinate, frf.ordinate)
    assert np.array_equal(import_file(path).ordinate, frf.ordinate)


@pytest.mark.parametrize('system', ['in-slinch-lbf-s (g)', 'mm-kg-N-s'])
def test_the_whole_corpus_comes_back_from_a_display_system(tmp_path, system):
    """Every kind with units — records, specifications and their limits,
    geometry, shapes, sine tones and levels — written in a display
    system and read back: the same SI values, only the declared units
    moved to the ones written in."""
    from visualdynamics.units import SYSTEMS

    project = native.load(CORPUS)
    mat = str(tmp_path / 'corpus.mat')
    export_file(project, mat, unit_system=SYSTEMS[system])
    back = import_file(mat)
    project.save(str(tmp_path / 'a.vdyn'))
    back.save(str(tmp_path / 'b.vdyn'))
    a, b = vdyn_tree(tmp_path / 'a.vdyn'), vdyn_tree(tmp_path / 'b.vdyn')
    assert set(a) == set(b), (set(a) ^ set(b))
    labels = ('ordinate_unit', 'reference_unit', 'length_unit', 'mass_unit')
    moved = [key for key in a if not same(a[key], b[key])]
    numbers = [key for key in moved if not key.split('/')[-1].split('@')[-1] in labels]
    assert numbers, 'some values were converted and back'
    for key in numbers:
        assert np.allclose(np.asarray(a[key]), np.asarray(b[key]), rtol=1e-9,
                           atol=0), key
    assert any(key.endswith('ordinate_unit') for key in moved), 'labels moved'


# ---- refusals, said by name -------------------------------------------------

def test_a_strangers_mat_is_refused_by_name(tmp_path):
    path = str(tmp_path / 'workspace.mat')
    savemat(path, {'x': np.arange(3), 'y': np.eye(2)})
    with pytest.raises(ValueError, match='not a Visual Dynamics file.*x, y'):
        import_file(path)


def test_a_struct_built_to_the_layout_is_read_without_the_stamp(tmp_path):
    """A user's own struct with the documented fields is the one
    unstamped shape accepted — and it comes in as the object it says."""
    path = str(tmp_path / 'own.mat')
    savemat(path, {'th': {
        'data_class': 'TimeHistory', 'function_type': 1,
        'abscissa': np.linspace(0, 1, 8)[:, None],
        'ordinate': np.ones((2, 8)),
        'response_dof': np.array(['101Z+', '102Z+'], dtype=object),
        'ordinate_dim': np.array(['acceleration'] * 2, dtype=object),
        'comment': np.array(['', ''], dtype=object),
        'ordinate_unit': np.array(['g', 'g'], dtype=object),
        'reference_unit': np.array(['', ''], dtype=object),
        'dimension_hint': np.array(['', ''], dtype=object)}},
        oned_as='column')
    back = import_file(path)
    assert type(back).__name__ == 'TimeHistory'
    assert back.response_dof == ['101Z+', '102Z+']
    assert back.ordinate_unit == ['g', 'g']


def test_a_newer_stamp_is_refused(tmp_path):
    path = str(tmp_path / 'future.mat')
    savemat(path, {'visualdynamics_schema': native.SCHEMA_VERSION + 1,
                   'data': {'abscissa': np.zeros((2, 1))}})
    with pytest.raises(ValueError, match='newer Visual Dynamics'):
        import_file(path)


def test_a_record_too_big_for_the_format_is_refused_first(tmp_path,
                                                           monkeypatch):
    """scipy raises after writing most of a too-large file; the check
    here runs before a byte is written and names the array."""
    monkeypatch.setattr(matlab, 'MAX_BYTES', 1000)
    frf = visualdynamics.import_file(fixture_path('plate', 'frfs.npz'))
    path = str(tmp_path / 'big.mat')
    with pytest.raises(ValueError, match=r'data\.(abscissa|ordinate) is .*GB'):
        export_file(frf, path)
    assert not os.path.exists(path)


# ---- the doors --------------------------------------------------------------

def test_every_object_kind_is_offered_matlab():
    project = native.load(CORPUS)
    for name, obj in project.items():
        assert 'matlab' in [e.name for e in visualdynamics.io.exporters(obj)], name
    assert 'matlab' in [e.name for e in visualdynamics.io.exporters(project)]


def test_the_project_saves_and_opens_as_mat(tmp_path):
    project = native.load(CORPUS)
    path = project.save(str(tmp_path / 'corpus.mat'))
    assert path.endswith('.mat') and os.path.exists(path)
    back = visualdynamics.Project.open(path)
    assert list(back) == list(project)
    assert back.journal == [f'project = visualdynamics.Project.open({path!r})']


def test_the_window_offers_matlab_for_the_project(window, pump, monkeypatch,
                                                  tmp_path):
    """Save Project offers the MATLAB filter, and picking it writes the
    .mat through the same verb a script calls."""
    from PySide6.QtWidgets import QFileDialog

    window.import_paths([CORPUS])
    pump()
    target = str(tmp_path / 'saved')
    seen = {}

    def fake(parent, title, start, filters):
        seen['filters'] = filters
        return target, 'MATLAB file (*.mat)'

    monkeypatch.setattr(QFileDialog, 'getSaveFileName', staticmethod(fake))
    window.save_test()
    assert '*.mat' in seen['filters']
    assert os.path.exists(target + '.mat')
    assert isinstance(import_file(target + '.mat'), native.Project)


def test_a_suffix_in_capitals_is_still_matlab(tmp_path):
    """A `.MAT` from a Windows tool was offered to no importer."""
    frf = visualdynamics.import_file(fixture_path('plate', 'frfs.npz'),
                                     response_unit='m/s**2', reference_unit='N')
    path = str(tmp_path / 'FRF.MAT')
    matlab.save(frf, str(tmp_path / 'frf.mat'))
    os.replace(str(tmp_path / 'frf.mat'), path)
    assert matlab.sniff(path)
    assert np.array_equal(import_file(path).ordinate, frf.ordinate)
