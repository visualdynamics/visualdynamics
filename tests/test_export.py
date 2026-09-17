"""Anything visualdynamics can read, it can write — and reading it back agrees.

Round trips are the point of these tests: an exporter that writes a file
nobody can read again is worse than no exporter.
"""

import pathlib

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics


def plate(name):
    return fixture_path('plate', name)


def survey(name):
    return fixture_path('plate', name)


DATA_FILES = [
    ('frfs.npz', {'response_unit': 'm/s**2', 'reference_unit': 'N'}),
    ('time.npz', {'ordinate_unit': 'm/s**2'}),
    ('psd.npz', {'ordinate_unit': 'm/s**2'}),
    ('spectrum.npz', {'ordinate_unit': 'm/s**2'}),
]


def same_geometry(a, b, elements=True, tracelines=True):
    assert np.array_equal(a.node_id, b.node_id)
    assert np.allclose(a.node_xyz, b.node_xyz)
    if tracelines:
        assert len(a.traceline_conn) == len(b.traceline_conn)
        assert all(np.array_equal(x, y)
                   for x, y in zip(a.traceline_conn, b.traceline_conn))
    if elements:
        assert np.array_equal(np.asarray(a.elem_type, dtype=int),
                              np.asarray(b.elem_type, dtype=int))
        assert all(np.array_equal(x, y)
                   for x, y in zip(a.elem_conn, b.elem_conn))
    b.validate()


# ---- what can be written at all -------------------------------------------

def test_every_object_type_has_somewhere_to_go():
    geometry = visualdynamics.import_file(survey('geometry.npz'))
    shapes = visualdynamics.import_file(survey('shapes.npy'))
    data = visualdynamics.import_file(plate('frfs.npz'))
    for obj in (geometry, shapes, data):
        assert visualdynamics.io.exporters(obj), f'{type(obj).__name__} cannot be written'


def test_an_exporter_says_what_it_cannot_take():
    data = visualdynamics.import_file(plate('frfs.npz'))
    with pytest.raises(ValueError, match='exodus cannot write'):
        visualdynamics.export_file(data, '/tmp/unused.exo', format='exodus')


def test_an_unknown_suffix_names_the_ones_that_would_work(tmp_path):
    shapes = visualdynamics.import_file(survey('shapes.npy'))
    with pytest.raises(ValueError, match='sdynpy_shapes'):
        visualdynamics.export_file(shapes, str(tmp_path / 'shapes.nonsense'))


def test_rattlesnake_is_read_only():
    """Its files record a controller run visualdynamics was never part of."""
    assert any(i.name == 'rattlesnake' for i in visualdynamics.io.importers())
    assert not any(e.name == 'rattlesnake' for e in visualdynamics.io.exporters())


# ---- sdynpy ----------------------------------------------------------------

def test_geometry_round_trips_through_sdynpy_npz(tmp_path):
    source = visualdynamics.import_file(survey('geometry.npz'))
    path = str(tmp_path / 'out.npz')
    visualdynamics.export_file(source, path, format='sdynpy_geometry')
    same_geometry(source, visualdynamics.import_file(path))


def test_geometry_keeps_its_coordinate_systems_through_sdynpy(tmp_path):
    source = visualdynamics.import_file(survey('geometry.npz'))
    source.add_coordinate_system(origin=[0.1, 0.0, 0.0], name='second')
    source.add_coordinate_system(origin=[0.0, 0.1, 0.0], name='third')
    source.cs_type[:] = [0, 1, 2]
    path = str(tmp_path / 'cs.npz')
    visualdynamics.export_file(source, path, format='sdynpy_geometry')
    back = visualdynamics.import_file(path)
    assert np.array_equal(source.cs_id, back.cs_id)
    assert np.array_equal(source.cs_type, back.cs_type)
    assert np.allclose(source.cs_matrix, back.cs_matrix)


@pytest.mark.parametrize(('name', 'units'), DATA_FILES)
def test_data_round_trips_through_sdynpy_npz(tmp_path, name, units):
    source = visualdynamics.import_file(plate(name), **units)
    path = str(tmp_path / 'out.npz')
    visualdynamics.export_file(source, path, format='sdynpy_data')
    back = visualdynamics.import_file(path, **units)
    assert type(back) is type(source)
    assert back.num_records == source.num_records
    assert np.allclose(back.abscissa, source.abscissa)
    assert np.allclose(back.ordinate, source.ordinate)
    assert back.response_dof == source.response_dof
    assert back.reference_dof == source.reference_dof


def test_shapes_round_trip_through_sdynpy_npy(tmp_path):
    source = visualdynamics.import_file(survey('shapes.npy'), mass_unit='kg')
    path = str(tmp_path / 'out.npy')
    visualdynamics.export_file(source, path, format='sdynpy_shapes')
    back = visualdynamics.import_file(path, mass_unit='kg')
    assert back.num_shapes == source.num_shapes
    assert np.allclose(back.frequency, source.frequency)
    assert np.allclose(back.damping, source.damping)
    assert list(back.coordinate) == list(source.coordinate)
    assert np.allclose(np.real(back.shape_matrix),
                       np.real(source.shape_matrix))


def test_a_typed_description_survives_as_the_files_comment(tmp_path):
    """The formats have one comment field where visualdynamics has two."""
    source = visualdynamics.import_file(survey('shapes.npy'))
    source.description[0] = 'first bending'
    path = str(tmp_path / 'described.npy')
    visualdynamics.export_file(source, path, format='sdynpy_shapes')
    assert visualdynamics.import_file(path).comment[0] == 'first bending'


# ---- exodus ----------------------------------------------------------------

def test_geometry_round_trips_through_exodus(tmp_path):
    source = visualdynamics.import_file(plate('geometry.exo'))
    path = str(tmp_path / 'out.exo')
    visualdynamics.export_file(source, path, format='exodus')
    same_geometry(source, visualdynamics.import_file(path), tracelines=False)


def test_exodus_carries_the_coordinate_systems_as_frames(tmp_path):
    """Exodus defines coordinate frames — an id, an origin, a point on
    the local Z axis, a point in the local XZ plane, and a tag — and
    nothing in the format refers to them, so the *definitions* round
    trip and the *assignment* does not (Brandon asked, 2026-09-12:
    the module had said the format has no coordinate systems at all)."""
    import netCDF4
    from scipy.spatial.transform import Rotation

    source = visualdynamics.import_file(plate('geometry.exo'))
    turned = Rotation.from_euler('z', 30, degrees=True).as_matrix()
    local = source.add_coordinate_system(origin=[0.1, 0.2, 0.3],
                                         rotation=turned, name='turned')
    barrel = source.add_coordinate_system(origin=[0.0, 0.5, 0.0],
                                          cs_type=1, name='barrel')
    first = int(source.node_id[0])
    source.node_disp_cs[0] = local
    path = str(tmp_path / 'frames.exo')
    visualdynamics.export_file(source, path, format='exodus')

    with netCDF4.Dataset(path) as ds:
        ids = list(ds.variables['frame_ids'][:])
        coords = np.asarray(ds.variables['frame_coordinates'][:]).reshape(-1, 9)
        tags = [t.decode() for t in ds.variables['frame_tags'][:]]
    assert ids == [1, local, barrel]
    assert tags == ['R', 'R', 'C']
    row = coords[ids.index(local)]
    assert np.allclose(row[0:3], [0.1, 0.2, 0.3]), 'the origin'
    assert np.allclose(row[3:6] - row[0:3], turned[2]), 'a point on Z'
    assert np.allclose(row[6:9] - row[0:3], turned[0]), 'a point in XZ'

    back = visualdynamics.import_file(path)
    assert list(back.cs_id) == [1, local, barrel]
    assert list(back.cs_type) == [0, 0, 1]
    assert np.allclose(back.cs_matrix, source.cs_matrix)
    assert back.cs_name == ['', '', ''], 'a frame carries no name'
    # no exodus node or variable can name a frame; the assignment rides
    # as a named node set instead (the next test), so it comes back
    i = list(back.node_id).index(first)
    assert back.node_disp_cs[i] == local and back.node_def_cs[i] == 1


def test_exodus_node_sets_carry_which_node_is_in_which_system(tmp_path):
    """Nothing in exodus lets a node name a frame, but a *node set* is
    standard, named, and shown harmlessly by every exodus tool — so a
    set per local system, named for it, carries the assignment: which
    nodes are placed in it (`def_cs_<id>`) and which are measured in it
    (`disp_cs_<id>`). The global system needs no set. Those sets are
    markers, not instrumented nodes, so the import dialog's list of
    node sets leaves them out."""
    import netCDF4

    source = visualdynamics.import_file(plate('geometry.exo'))
    local = source.add_coordinate_system(origin=[0.1, 0.0, 0.0])
    other = source.add_coordinate_system(origin=[0.0, 0.1, 0.0], cs_type=1)
    source.node_disp_cs[:3] = local
    source.node_def_cs[1:4] = other
    source.node_disp_cs[10] = other
    path = str(tmp_path / 'assigned.exo')
    visualdynamics.export_file(source, path, format='exodus')

    with netCDF4.Dataset(path) as ds:
        names = [''.join(c.decode() for c in row if c).strip()
                 for row in ds.variables['ns_names'][:]]
        members = {name: list(ds.variables[f'node_ns{i + 1}'][:])
                   for i, name in enumerate(names)}
    assert set(names) == {f'disp_cs_{local}', f'def_cs_{other}',
                          f'disp_cs_{other}'}
    assert members[f'disp_cs_{local}'] == [1, 2, 3], '1-based indices'
    assert members[f'def_cs_{other}'] == [2, 3, 4]
    assert members[f'disp_cs_{other}'] == [11]

    back = visualdynamics.import_file(path)
    assert np.array_equal(back.node_disp_cs, source.node_disp_cs)
    assert np.array_equal(back.node_def_cs, source.node_def_cs)
    from visualdynamics.io.exodus import result_summary
    assert result_summary(path)['node_sets'] == [], \
        'markers are not offered as instrumented nodes'


def test_an_exodus_whose_frames_are_all_local_still_gets_a_global_system(
        tmp_path):
    """A file from elsewhere may define frames and none of them the
    identity; the nodes still need a global system to be placed in."""
    import netCDF4

    source = visualdynamics.import_file(plate('geometry.exo'))
    path = str(tmp_path / 'local.exo')
    visualdynamics.export_file(source, path, format='exodus')
    with netCDF4.Dataset(path, 'a') as ds:
        ds.variables['frame_ids'][:] = np.array([7], dtype=np.int32)
        ds.variables['frame_coordinates'][:] = np.array(
            [1.0, 0, 0, 1, 0, 1, 2, 0, 0])
    back = visualdynamics.import_file(path)
    assert list(back.cs_id) == [1, 7]
    assert np.allclose(back.cs_matrix[0], np.vstack([np.eye(3),
                                                     np.zeros(3)]))
    assert np.allclose(back.cs_matrix[1, 3], [1.0, 0, 0])
    assert set(back.node_def_cs) == {1} and set(back.node_disp_cs) == {1}


def test_exodus_writes_tracelines_as_beams(tmp_path):
    """Exodus has no traceline, so they go as runs of two-node beams."""
    source = visualdynamics.import_file(survey('geometry.npz'))
    path = str(tmp_path / 'beams.exo')
    visualdynamics.export_file(source, path, format='exodus')
    back = visualdynamics.import_file(path)
    segments = sum(len(line) - 1 for line in source.traceline_conn)
    assert len(back.elem_conn) == len(source.elem_conn) + segments
    assert sum(1 for code in back.elem_type if int(code) == 21) == segments
    back.validate()


def test_exodus_can_leave_tracelines_out(tmp_path):
    source = visualdynamics.import_file(survey('geometry.npz'))
    path = str(tmp_path / 'plain.exo')
    visualdynamics.io.exodus.save(source, path, tracelines_as_beams=False)
    assert len(visualdynamics.import_file(path).elem_conn) == len(source.elem_conn)


# ---- unv -------------------------------------------------------------------

def test_geometry_round_trips_through_unv(tmp_path):
    source = visualdynamics.import_file(survey('geometry.npz'))
    path = str(tmp_path / 'out.unv')
    visualdynamics.export_file(source, path, format='unv')
    same_geometry(source, visualdynamics.import_file(path))


@pytest.mark.parametrize(('name', 'units'), DATA_FILES)
def test_data_round_trips_through_unv(tmp_path, name, units):
    source = visualdynamics.import_file(plate(name), **units)
    path = str(tmp_path / 'out.unv')
    visualdynamics.export_file(source, path, format='unv')
    back = visualdynamics.import_file(path)
    assert type(back) is type(source)
    assert back.num_records == source.num_records
    assert np.allclose(back.abscissa, source.abscissa)
    assert np.allclose(back.ordinate, source.ordinate, rtol=1e-4)
    assert back.response_dof == source.response_dof
    assert back.reference_dof == source.reference_dof


def test_unv_declares_units_only_when_they_are_known(tmp_path):
    """A file that says SI is making a claim; only make it when it is true."""
    declared = visualdynamics.import_file(plate('frfs.npz'), response_unit='m/s**2',
                                reference_unit='N')
    path = str(tmp_path / 'declared.unv')
    visualdynamics.export_file(declared, path, format='unv')
    assert visualdynamics.import_file(path).units_defined
    assert set(visualdynamics.import_file(path).ordinate_dim) == {'acceleration/force'}

    unitless = visualdynamics.import_file(plate('frfs.npz'))
    path = str(tmp_path / 'unitless.unv')
    visualdynamics.export_file(unitless, path, format='unv')
    back = visualdynamics.import_file(path)
    assert not back.units_defined, 'a unit-less object must not claim units'
    assert set(back.ordinate_dim) == {'unknown'}


# ---- native ----------------------------------------------------------------

def test_the_native_format_is_the_one_that_keeps_everything(tmp_path):
    """Which is why Save and Load use it, and export is for other tools."""
    source = visualdynamics.import_file(plate('frfs.npz'), response_unit='m/s**2',
                              reference_unit='N')
    path = str(tmp_path / 'kept.vdyn')
    visualdynamics.save(source, path)
    back = visualdynamics.load(path)
    assert back.units_defined
    assert back.ordinate_unit == source.ordinate_unit
    assert back.reference_unit == source.reference_unit
    assert np.allclose(back.ordinate, source.ordinate)


# ---- mode shapes in the formats that can hold them -------------------------

def test_shapes_round_trip_through_unv(tmp_path):
    """Dataset 55 carries frequency, damping and modal mass as well."""
    source = visualdynamics.import_file(survey('shapes.npy'), mass_unit='kg')
    path = str(tmp_path / 'modes.unv')
    visualdynamics.export_file(source, path, format='unv')
    back = visualdynamics.import_file(path)
    assert back.num_shapes == source.num_shapes
    assert back.num_dofs == source.num_dofs
    assert list(back.coordinate) == list(source.coordinate)
    assert np.allclose(back.frequency, source.frequency, rtol=1e-4)
    assert np.allclose(back.damping, source.damping, rtol=1e-4)
    assert np.allclose(back.modal_mass, source.modal_mass, rtol=1e-4)
    assert np.allclose(np.real(back.shape_matrix),
                       np.real(source.shape_matrix), rtol=1e-3)


def test_shapes_round_trip_through_exodus(tmp_path):
    """Exodus stores a mode per time step, the frequency as its time —
    always riding a geometry, because the file is a mesh with results."""
    source = visualdynamics.import_file(survey('shapes.npy'), mass_unit='kg')
    mesh = visualdynamics.import_file(survey('geometry.npz'))
    path = str(tmp_path / 'modes.exo')
    visualdynamics.export_file(source, path, format='exodus', geometry=mesh)
    back = visualdynamics.import_file(path)
    shapes = back['shapes']
    assert shapes.num_shapes == source.num_shapes
    # atol: coincident frequencies are deliberately nudged a hair apart
    # (1e-9 of the top frequency) so ParaView keeps every mode as its
    # own time step — at the plate's span that hair is microhertz
    assert np.allclose(shapes.frequency, source.frequency, atol=1e-3)
    for dof in ('101X+', '707Y+', '1313Z+'):
        assert (shapes.shape_matrix[0, shapes.coordinate.index(dof)]
                == pytest.approx(float(np.real(
                    source.shape_matrix[0, source.coordinate.index(dof)]))))


def test_exodus_can_carry_the_mesh_with_the_modes(tmp_path):
    source = visualdynamics.import_file(survey('shapes.npy'))
    mesh = visualdynamics.import_file(survey('geometry.npz'))
    path = str(tmp_path / 'both.exo')
    visualdynamics.io.exodus.save(source, path, geometry=mesh)
    back = visualdynamics.import_file(path)
    assert set(back) == {'geometry', 'shapes'}
    assert back['geometry'].num_nodes == mesh.num_nodes
    assert np.allclose(back['geometry'].node_xyz, mesh.node_xyz)


def test_exodus_carries_damping_and_modal_mass_as_global_variables(
        tmp_path):
    """A global variable is a scalar per time step, and a step is a
    mode — exactly where a mode's damping and modal mass belong. They
    were lost until 2026-09-12 (the earlier test said so plainly);
    now the modal round trip is whole, and ParaView ignores them."""
    import netCDF4

    source = visualdynamics.import_file(survey('shapes.npy'), mass_unit='kg')
    source.damping[:] = np.linspace(0.01, 0.05, source.num_shapes)
    source.modal_mass[:] = np.linspace(1.5, 3.0, source.num_shapes)
    mesh = visualdynamics.import_file(survey('geometry.npz'))
    path = str(tmp_path / 'damped.exo')
    visualdynamics.export_file(source, path, format='exodus', geometry=mesh)
    with netCDF4.Dataset(path) as ds:
        names = [''.join(c.decode() for c in row if c).strip()
                 for row in ds.variables['name_glo_var'][:]]
    assert names == ['Damping', 'ModalMass']
    back = visualdynamics.import_file(path)['shapes']
    assert np.allclose(back.damping, source.damping)
    assert np.allclose(back.modal_mass, source.modal_mass)


def test_a_complex_shape_round_trips_through_exodus(tmp_path):
    """The real part stays DispX, so ParaView still assembles its Disp
    vector and animates; the imaginary part rides beside it as
    DispX_IM, the pairing the spectrum export already uses, and the
    reader puts the two back together. A real shape set writes no
    _IM variables at all — nothing for a ParaView user to wonder at."""
    import netCDF4

    def variable_names(path):
        with netCDF4.Dataset(path) as ds:
            return [''.join(c.decode() for c in row if c).strip()
                    for row in ds.variables['name_nod_var'][:]]

    mesh = visualdynamics.import_file(survey('geometry.npz'))
    real = visualdynamics.import_file(survey('shapes.npy'), mass_unit='kg')
    plain = str(tmp_path / 'real.exo')
    visualdynamics.export_file(real, plain, format='exodus', geometry=mesh)
    assert not any(n.endswith('_IM') for n in variable_names(plain))

    source = visualdynamics.import_file(survey('shapes.npy'), mass_unit='kg')
    turned = np.exp(1j * np.linspace(0.0, 1.0, source.num_shapes))
    source.shape_matrix = source.shape_matrix * turned[:, np.newaxis]
    assert np.iscomplexobj(source.shape_matrix)
    path = str(tmp_path / 'complex.exo')
    visualdynamics.export_file(source, path, format='exodus', geometry=mesh)
    names = variable_names(path)
    assert 'DispX' in names and 'DispX_IM' in names
    back = visualdynamics.import_file(path)['shapes']
    assert np.iscomplexobj(back.shape_matrix)
    for dof in ('101X+', '707Y+', '1313Z+'):
        assert back.shape_matrix[2, back.coordinate.index(dof)] == \
            pytest.approx(source.shape_matrix[2, source.coordinate.index(dof)])


def test_a_geometry_only_exodus_still_reads_as_one_object(tmp_path):
    """Adding shapes must not change what a mesh-only file returns."""
    source = visualdynamics.import_file(plate('geometry.exo'))
    path = str(tmp_path / 'mesh.exo')
    visualdynamics.export_file(source, path, format='exodus')
    assert isinstance(visualdynamics.import_file(path), visualdynamics.Geometry)


def test_shapes_can_be_written_to_every_format_that_holds_them():
    shapes = visualdynamics.import_file(survey('shapes.npy'))
    names = {e.name for e in visualdynamics.io.exporters(shapes)}
    assert names == {'sdynpy_shapes', 'unv', 'exodus', 'adf_shapes'}


# ---- exodus has no coordinate systems, so results go out global ------------

def turned_survey(axis=2, angle=np.pi / 2):
    """The plate with a second frame a quarter turn round, and the
    first grid row displaced in it — the multi-frame case the airplane
    fixture used to ship with, built here instead of shipped."""
    from visualdynamics.rotate import rotate_frame

    geometry = visualdynamics.import_file(survey('geometry.npz'), length_unit='m')
    turned = geometry.add_coordinate_system(origin=[0.0, 0.0, 0.0],
                                            name='turned frame')
    geometry.cs_matrix[-1] = rotate_frame(geometry.cs_matrix[-1], axis, angle)
    first_row = [i for i, n in enumerate(geometry.node_id)
                 if int(n) // 100 == 1]
    geometry.node_disp_cs[first_row] = turned
    geometry.validate()
    return geometry, turned


def test_shapes_are_rotated_into_the_global_frame_for_exodus(tmp_path):
    """A value left in a node's own frame would read back as global."""
    geometry, frame = turned_survey()
    shapes = visualdynamics.import_file(survey('shapes.npy'))
    path = str(tmp_path / 'global.exo')
    visualdynamics.io.exodus.save(shapes, path, geometry=geometry)
    written = visualdynamics.import_file(path)['shapes']

    dofs = list(shapes.coordinate)
    turned = int(next(n for n, c in zip(geometry.node_id, geometry.node_disp_cs)
                      if int(c) == frame))
    x, y = dofs.index(f'{turned}X+'), dofs.index(f'{turned}Y+')
    local = np.real(shapes.shape_matrix[3][[x, y]])
    globals_ = written.shape_matrix[3][[x, y]]
    assert np.allclose(globals_, [-local[1], local[0]]) or \
        np.allclose(globals_, [local[1], -local[0]]), 'a quarter turn'
    assert np.isclose(np.hypot(*local), np.hypot(*globals_)), 'same magnitude'


def test_a_node_already_in_the_global_frame_is_left_alone(tmp_path):
    geometry, frame = turned_survey()
    shapes = visualdynamics.import_file(survey('shapes.npy'))
    path = str(tmp_path / 'mixed.exo')
    visualdynamics.io.exodus.save(shapes, path, geometry=geometry)
    written = visualdynamics.import_file(path)['shapes']

    dofs = list(shapes.coordinate)
    plain = int(next(n for n, c in zip(geometry.node_id, geometry.node_disp_cs)
                     if int(c) != frame))
    column = dofs.index(f'{plain}X+')
    assert np.isclose(np.real(shapes.shape_matrix[3][column]),
                      written.shape_matrix[3][column])


def test_without_a_geometry_shapes_are_refused(tmp_path):
    """The shapes alone say neither what frame they are in nor what mesh
    they deflect — a mesh-less modal exodus was a file ParaView could do
    nothing with, so it is refused rather than written."""
    shapes = visualdynamics.import_file(survey('shapes.npy'))
    with pytest.raises(ValueError, match='needs a geometry'):
        visualdynamics.export_file(shapes, str(tmp_path / 'asis.exo'),
                         format='exodus')


# ---- unv keeps local frames, because it can --------------------------------

def test_unv_writes_the_coordinate_systems_the_nodes_refer_to(tmp_path):
    """Dataset 2420, without which local directions mean nothing."""
    from visualdynamics.rotate import rotate_frame

    source = visualdynamics.import_file(survey('geometry.npz'), length_unit='m')
    source.add_coordinate_system(origin=[0.1, 0.0, 0.0], name='second')
    source.add_coordinate_system(origin=[0.0, 0.1, 0.0], name='third')
    source.cs_type[:] = [0, 1, 2]
    source.cs_matrix[1] = rotate_frame(source.cs_matrix[1], 2, np.pi / 2)
    source.cs_name[1] = 'edge frame'

    path = str(tmp_path / 'frames.unv')
    visualdynamics.export_file(source, path, format='unv')
    back = visualdynamics.import_file(path, length_unit='m')

    assert np.array_equal(back.cs_id, source.cs_id)
    assert np.array_equal(back.cs_type, source.cs_type)
    assert list(back.cs_name) == list(source.cs_name)
    assert np.allclose(back.cs_matrix, source.cs_matrix)
    assert np.array_equal(back.node_disp_cs, source.node_disp_cs)
    assert np.array_equal(back.node_def_cs, source.node_def_cs)


def test_unv_shapes_stay_in_their_own_frames(tmp_path):
    """Unlike exodus: dataset 55 means the node's displacement system, and
    the systems are written alongside, so nothing has to be flattened."""
    source = visualdynamics.import_file(survey('shapes.npy'))
    path = str(tmp_path / 'local.unv')
    visualdynamics.export_file(source, path, format='unv')
    back = visualdynamics.import_file(path)
    assert np.allclose(np.real(back.shape_matrix),
                       np.real(source.shape_matrix), rtol=1e-3)


def test_a_geometry_with_no_named_systems_round_trips_too(tmp_path):
    source = visualdynamics.import_file(survey('geometry.npz'), length_unit='m')
    path = str(tmp_path / 'plain.unv')
    visualdynamics.export_file(source, path, format='unv')
    back = visualdynamics.import_file(path, length_unit='m')
    assert list(back.cs_name) == list(source.cs_name), 'a blank name is a name'
    same_geometry(source, back)


# ---- one exodus file, both objects -----------------------------------------

def test_an_exodus_with_a_mesh_and_modes_imports_both(tmp_path):
    mesh = visualdynamics.import_file(survey('geometry.npz'), length_unit='m')
    shapes = visualdynamics.import_file(survey('shapes.npy'))
    path = str(tmp_path / 'modal.exo')
    visualdynamics.io.exodus.save(shapes, path, geometry=mesh)

    result = visualdynamics.import_file(path)
    assert set(result) == {'geometry', 'shapes'}
    assert isinstance(result['geometry'], visualdynamics.Geometry)
    assert result['shapes'].num_shapes == shapes.num_shapes
    assert result['geometry'].num_nodes == mesh.num_nodes


# ---- exports are written in the display unit system -------------------------

SYSTEMS = ['m-kg-N-s', 'mm-kg-N-s', 'in-slinch-lbf-s', 'ft-slug-lbf-s']


#: the plate file's raw values are meters; reading them as inches makes
#: a deliberately silly 0.3048-inch article whose numbers expose which
#: system an export was written in
RAW = 0.3048


def inch_plate():
    """The plate's meter-valued file read as inches, on purpose."""
    return visualdynamics.import_file(plate('geometry.exo'), length_unit='in')


@pytest.mark.parametrize('system', SYSTEMS)
def test_a_unv_round_trips_whatever_system_it_was_written_in(tmp_path, system):
    """It declares its units, so the numbers come home to the same SI."""
    source = inch_plate()
    path = str(tmp_path / f'{system}.unv')
    visualdynamics.export_file(source, path, format='unv',
                     unit_system=visualdynamics.units.SYSTEMS[system])
    back = visualdynamics.import_file(path)
    assert back.units_defined
    assert np.allclose(back.node_xyz, source.node_xyz), system


@pytest.mark.parametrize(('system', 'expected'), [
    ('m-kg-N-s', RAW * 0.0254), ('mm-kg-N-s', RAW * 25.4),
    ('in-slinch-lbf-s', RAW),
])
def test_the_silent_formats_hold_the_numbers_you_were_looking_at(
        tmp_path, system, expected):
    """sdynpy records no units, so what it holds is what you asked for."""
    source = inch_plate()
    path = str(tmp_path / 'out.npz')
    visualdynamics.export_file(source, path, format='sdynpy_geometry',
                     unit_system=visualdynamics.units.SYSTEMS[system])
    with np.load(path, allow_pickle=True) as archive:
        assert archive['node']['coordinate'][:, 0].max() == pytest.approx(
            expected, rel=1e-9)


def test_exodus_follows_the_same_rule(tmp_path):
    source = inch_plate()
    path = str(tmp_path / 'out.exo')
    visualdynamics.export_file(source, path, format='exodus',
                     unit_system=visualdynamics.units.SYSTEMS['in-slinch-lbf-s'])
    assert visualdynamics.import_file(path).node_xyz[:, 0].max() == pytest.approx(RAW)


def test_data_is_converted_too(tmp_path):
    source = visualdynamics.import_file(plate('frfs.npz'), response_unit='m/s**2',
                              reference_unit='N')
    path = str(tmp_path / 'data.unv')
    visualdynamics.export_file(source, path, format='unv',
                     unit_system=visualdynamics.units.SYSTEMS['in-slinch-lbf-s'])
    back = visualdynamics.import_file(path)
    assert np.allclose(back.ordinate, source.ordinate, rtol=1e-4), \
        'declared units bring it back to the same SI'


def test_the_164_says_which_system_the_file_is_in(tmp_path):
    source = inch_plate()
    path = str(tmp_path / 'inches.unv')
    visualdynamics.export_file(source, path, format='unv',
                     unit_system=visualdynamics.units.SYSTEMS['in-slinch-lbf-s'])
    text = pathlib.Path(path).read_text(encoding='utf-8')
    header = text.split('   164')[1].splitlines()[1]
    assert header.strip().startswith('7'), 'code 7 is inch (pound f)'
    assert 'Inch' in header
    factors = text.split('   164')[1].splitlines()[2].replace('D', 'E').split()
    assert float(factors[0]) == pytest.approx(1 / 0.0254, rel=1e-9), \
        'inches per meter'
    assert float(factors[1]) == pytest.approx(1 / 4.4482216152605, rel=1e-9), \
        'pounds force per newton'


def test_a_unitless_object_ignores_the_unit_system(tmp_path):
    """There is no factor to convert it by, so nothing is applied."""
    source = visualdynamics.import_file(plate('geometry.exo'))      # no units declared
    assert not source.units_defined
    path = str(tmp_path / 'raw.unv')
    visualdynamics.export_file(source, path, format='unv',
                     unit_system=visualdynamics.units.SYSTEMS['in-slinch-lbf-s'])
    text = pathlib.Path(path).read_text(encoding='utf-8')
    assert '\n   164\n' not in text, 'nothing to declare'
    back = visualdynamics.import_file(path)
    assert not back.units_defined
    assert np.allclose(back.node_xyz, source.node_xyz), 'values untouched'


def test_without_a_system_the_stored_values_go_out(tmp_path):
    """Scripting default: what is stored is what is written."""
    source = inch_plate()
    path = str(tmp_path / 'stored.npz')
    visualdynamics.export_file(source, path, format='sdynpy_geometry')
    with np.load(path, allow_pickle=True) as archive:
        assert archive['node']['coordinate'][:, 0].max() == pytest.approx(
            RAW * 0.0254, rel=1e-9)


def test_reading_in_g_does_not_put_g_in_the_file(tmp_path):
    """The display override is a convenience of the screen. No format here
    can record 'g', so exports use the coherent system underneath."""
    source = visualdynamics.import_file(plate('time.npz'), ordinate_unit='g')
    in_g = visualdynamics.units.SYSTEMS['in-slinch-lbf-s (g)']
    plain = visualdynamics.units.SYSTEMS['in-slinch-lbf-s']

    # on screen: in/s**2 -> SI is x0.0254, SI -> g is /9.80665
    assert np.abs(source.display_ordinate(in_g)).max() == pytest.approx(
        np.abs(source.display_ordinate(plain)).max() * 0.0254 / 9.80665,
        rel=1e-6)

    written = {}
    for system in (in_g, plain):
        path = str(tmp_path / f'{system.name}.npz')
        visualdynamics.export_file(source, path, format='sdynpy_data',
                         unit_system=system)
        with np.load(path, allow_pickle=True) as archive:
            written[system.name] = np.abs(archive['data']['ordinate']).max()
    assert written[in_g.name] == pytest.approx(written[plain.name]), \
        'both write in/s**2'


def test_the_164_names_the_coherent_system(tmp_path):
    source = inch_plate()
    path = str(tmp_path / 'g.unv')
    visualdynamics.export_file(source, path, format='unv',
                     unit_system=visualdynamics.units.SYSTEMS['in-slinch-lbf-s (g)'])
    header = pathlib.Path(path).read_text(encoding='utf-8').split('   164')[1].splitlines()[1]
    assert header.strip().startswith('7'), 'inch (pound f), not user-defined'


def test_shapes_ride_a_geometry_through_exodus(tmp_path):
    """ParaView wants a mesh with results: one time step per mode, the
    frequency as the step's time, over the geometry's own mesh."""
    geometry = visualdynamics.import_file(survey('geometry.npz'))
    shapes = visualdynamics.import_file(survey('shapes.npy'))
    path = str(tmp_path / 'modal.exo')
    visualdynamics.export_file(shapes, path, format='exodus', geometry=geometry)
    back = visualdynamics.import_file(path)
    assert set(back) == {'geometry', 'shapes'}
    assert back['geometry'].num_nodes == geometry.num_nodes
    assert np.allclose(back['shapes'].frequency, shapes.frequency,
                       atol=1e-3)
    # a shape value survives the trip: node 101 X, first mode
    source_column = shapes.coordinate.index('101X+')
    returned = back['shapes']
    got = returned.shape_matrix[0, returned.coordinate.index('101X+')]
    assert got == pytest.approx(
        float(np.real(shapes.shape_matrix[0, source_column])))



def test_paraviews_own_reader_accepts_the_modal_exodus(tmp_path):
    """VTK's exodus reader — the one inside ParaView — must read the file.

    The writer declared the large-model layout (file_size = 1), where
    coordinates are three separate variables, but wrote the old combined
    'coord' array: the exodus library found nothing and ParaView
    segfaulted on the empty result.
    """
    from vtkmodules.vtkIOExodus import vtkExodusIIReader

    geometry = visualdynamics.import_file(survey('geometry.npz'))
    shapes = visualdynamics.import_file(survey('shapes.npy'))
    path = str(tmp_path / 'modal.exo')
    visualdynamics.export_file(shapes, path, format='exodus', geometry=geometry)

    reader = vtkExodusIIReader()
    reader.SetFileName(path)
    reader.UpdateInformation()
    assert reader.GetNumberOfTimeSteps() == shapes.num_shapes
    reader.SetAllArrayStatus(vtkExodusIIReader.NODAL, 1)
    reader.Update()
    element_blocks = reader.GetOutput().GetBlock(0)
    meshes = [element_blocks.GetBlock(i)
              for i in range(element_blocks.GetNumberOfBlocks())]
    beams = sum(len(line) - 1 for line in geometry.traceline_conn)
    assert sum(m.GetNumberOfCells() for m in meshes) == \
        len(geometry.elem_conn) + beams, 'a mesh, not a point cloud'
    arrays = [meshes[0].GetPointData().GetArrayName(i)
              for i in range(meshes[0].GetPointData().GetNumberOfArrays())]
    assert 'Disp' in arrays, 'the vector ParaView warps by'


def test_coincident_mode_frequencies_stay_distinct_time_steps(tmp_path):
    """ParaView collapses duplicate time values, so six rigid-body modes
    at exactly 0 Hz read back as one; the written times separate them by
    a hair."""
    import netCDF4

    geometry = visualdynamics.import_file(survey('geometry.npz'))
    shapes = visualdynamics.import_file(survey('shapes.npy'))
    assert (shapes.frequency == 0.0).sum() > 1, 'the fixture has the case'
    path = str(tmp_path / 'rigid.exo')
    visualdynamics.export_file(shapes, path, format='exodus', geometry=geometry)
    with netCDF4.Dataset(path) as ds:
        times = np.asarray(ds.variables['time_whole'][:])
    assert len(np.unique(times)) == shapes.num_shapes, 'no two collapse'
    assert (np.diff(times) > 0).all(), 'strictly increasing'
    assert np.allclose(times, shapes.frequency, atol=1e-3), (
        'still readable as frequencies')


# ---- Rattlesnake's specification file ---------------------------------------
#
# Not the .nc4 (a run that happened) but the target the Random
# environment loads before a test: f, a whole cpsd matrix per line, the
# bands, and node/direction coordinates it uses to order the matrix.

def _stated_target(project):
    """The controller's own target with its pairs stated independent."""
    from visualdynamics.core.author import SpecificationDraft

    project.import_file(plate('random.nc4'))
    draft = SpecificationDraft.from_specification(
        project['Specification']).with_all_pairs(0.0)
    project.author_specification('Specification', draft, replace=True)
    return project['Specification']


def test_autos_alone_go_out_with_zeros_and_come_back_as_autos(tmp_path):
    """Most random tests run on autospectra alone (Brandon,
    2026-09-05): the controller's own targets carry zeros off the
    diagonal, so that is what is written — and `held` says they were
    placeholders, so nothing comes back claiming independence."""
    project = visualdynamics.Project('t')
    project.import_file(plate('random.nc4'))
    spec = project['Specification']
    assert 'rattlesnake_specification' in [
        e.name for e in visualdynamics.io.exporters(spec)]
    out = tmp_path / 'spec.npz'
    visualdynamics.export_file(spec, str(out),
                               format='rattlesnake_specification')
    with np.load(out) as d:
        assert d['cpsd'].shape == (1025, 8, 8)
        assert np.all(d['cpsd'][:, ~np.eye(8, dtype=bool)] == 0)
        assert np.array_equal(d['held'], np.eye(8, dtype=bool))
    back = visualdynamics.import_file(str(out))
    assert back.num_records == 8, 'placeholders, not statements'
    assert back.response_dof == spec.response_dof


def test_the_target_goes_out_as_the_controller_reads_it(tmp_path):
    project = visualdynamics.Project('t')
    spec = _stated_target(project)
    out = tmp_path / 'spec.npz'
    visualdynamics.export_file(spec, str(out),
                               format='rattlesnake_specification')
    with np.load(out) as d:
        assert set(d.files) == {'f', 'cpsd', 'dof', 'held', 'coordinate',
                                'warning_lower', 'warning_upper',
                                'abort_lower', 'abort_upper'}
        assert d['held'].all(), 'every pair stated'
        assert np.array_equal(d['f'], spec.abscissa)
        cpsd = d['cpsd']
        # the whole sheet written back gives the object the sheet's
        # lines — the band, 726 of the controller's 1025 (2026-09-06)
        assert cpsd.shape == (726, 8, 8) and cpsd.dtype == np.complex128
        autos = [i for i in range(spec.num_records)
                 if spec.response_dof[i] == spec.reference_dof[i]]
        for k, i in enumerate(autos):
            assert np.allclose(cpsd[:, k, k], spec.ordinate[i])
        assert np.all(cpsd[:, ~np.eye(8, dtype=bool)] == 0), 'stated zeros'
        assert d['warning_upper'].shape == (726, 8)
        assert np.allclose(d['abort_lower'][:, 0],
                           spec.limits['abort_lower'][autos[0]], equal_nan=True)
        coordinate = d['coordinate']
        assert coordinate.shape == (8, 8, 2)
        assert coordinate.dtype.names == ('node', 'direction')
        assert tuple(coordinate[0, 1, 0]) == (101, 3), 'response 101Z+'
        assert tuple(coordinate[0, 1, 1]) == (104, 3), 'reference 104Z+'
        assert list(d['dof']) == [spec.response_dof[i] for i in autos]


def test_the_target_round_trips_with_its_stated_zeros(tmp_path):
    project = visualdynamics.Project('t')
    spec = _stated_target(project)
    out = tmp_path / 'spec.npz'
    visualdynamics.export_file(spec, str(out),
                               format='rattlesnake_specification')
    assert any(i.name == 'rattlesnake_specification' and i.sniff(str(out))
               for i in visualdynamics.io.importers())
    back = visualdynamics.import_file(str(out), ordinate_unit='m/s**2')
    assert isinstance(back, visualdynamics.Specification)
    assert back.num_records == 64, 'a stated zero is a cross term, kept'
    pairs = {(r, f): i for i, (r, f) in enumerate(
        zip(back.response_dof, back.reference_dof))}
    for i, (r, f) in enumerate(zip(spec.response_dof, spec.reference_dof)):
        assert np.allclose(back.ordinate[pairs[(r, f)]], spec.ordinate[i])
    for name, values in spec.limits.items():
        for i, (r, f) in enumerate(zip(spec.response_dof, spec.reference_dof)):
            assert np.allclose(back.limits[name][pairs[(r, f)]], values[i],
                               equal_nan=True), name
    assert back.ordinate_dim[0] == 'acceleration**2/frequency'


def test_a_hermitian_half_is_completed_and_the_units_are_the_systems(tmp_path):
    """A specification holding one of each pair writes the conjugate for
    the other, as the transform allows; values go out in the coherent
    system asked for, and the file cannot say which."""
    from visualdynamics.core.author import SpecificationDraft

    draft = SpecificationDraft.at_dofs(['1Z+', '2Z+']).with_all_pairs(
        0.5, 30.0)
    spec = draft.make()
    keep = [i for i, (r, f) in enumerate(zip(spec.response_dof,
                                              spec.reference_dof))
            if not (r == '2Z+' and f == '1Z+')]
    half = visualdynamics.Specification(
        spec.abscissa, spec.ordinate[keep],
        response_dof=[spec.response_dof[i] for i in keep],
        reference_dof=[spec.reference_dof[i] for i in keep],
        ordinate_dim=[spec.ordinate_dim[i] for i in keep],
        **{k: v[keep] for k, v in spec.limits.items()})
    assert half.num_records == 3
    out = tmp_path / 'half.npz'
    system = visualdynamics.units.SYSTEMS['in-slinch-lbf-s (g)']
    visualdynamics.export_file(half, str(out),
                               format='rattlesnake_specification',
                               unit_system=system)
    with np.load(out) as d:
        cpsd = d['cpsd']
        assert np.allclose(cpsd[:, 1, 0], np.conj(cpsd[:, 0, 1]))
        assert d['held'].all(), 'a Hermitian half is the whole pair'
        factor = float(system.coherent.from_si(1.0, 'acceleration**2/frequency'))
        assert factor != 1.0
        assert np.allclose(cpsd[:, 0, 0], np.real(spec.ordinate[0]) * factor), \
            'the coherent system, in/s², not the display g'
        assert np.allclose(d['abort_upper'][:, 0],
                           spec.limits['abort_upper'][0] * factor)


def test_a_modal_specification_writes_no_coordinate(tmp_path):
    """A modal coordinate has no node and no direction to write, so the
    controller gets the matrix in order and no coordinate array — and
    the dof list brings it back by name."""
    from visualdynamics.core.author import SpecificationDraft

    spec = SpecificationDraft.at_dofs(['M1', 'M2'],
                                      'acceleration').with_all_pairs(1.0).make()
    out = tmp_path / 'modal.npz'
    visualdynamics.export_file(spec, str(out),
                               format='rattlesnake_specification')
    with np.load(out) as d:
        assert 'coordinate' not in d.files
        assert list(d['dof']) == ['M1', 'M2']
    back = visualdynamics.import_file(str(out))
    assert back.response_dof[:2] == ['M1', 'M1']
    assert back.num_records == 4


def test_a_controllers_own_file_reads_placeholders_as_autos(tmp_path):
    """A file without a dof list is the controller's or sdynpy's: an
    off-diagonal all zero or NaN is its placeholder, not a statement,
    and the channels come from the coordinate diagonal."""
    coordinate = np.zeros((2, 2, 2), dtype=[('node', '<u8'),
                                            ('direction', 'i1')])
    for i, node in enumerate((101, 104)):
        for j, other in enumerate((101, 104)):
            coordinate[i, j, 0] = (node, 3)
            coordinate[i, j, 1] = (other, 3)
    f = np.arange(20.0, 60.0, 10.0)
    cpsd = np.zeros((4, 2, 2), dtype=complex)
    cpsd[:, 0, 0] = 1.0
    cpsd[:, 1, 1] = 2.0
    out = tmp_path / 'controller.npz'
    np.savez(out, f=f, cpsd=cpsd, coordinate=coordinate,
             warning_upper=np.full((4, 2), 3.0))
    back = visualdynamics.import_file(str(out))
    assert back.num_records == 2
    assert back.response_dof == ['101Z+', '104Z+']
    assert np.allclose(back.limits['warning_upper'], 3.0)
    assert set(back.limits) == {'warning_upper'}
    cpsd[:, 0, 1] = 0.5j
    np.savez(out, f=f, cpsd=cpsd, coordinate=coordinate)
    assert visualdynamics.import_file(str(out)).num_records == 4, \
        'one meaningful number and every pair is read'
    np.savez(out, f=f, cpsd=cpsd)
    with pytest.raises(ValueError, match='names no channels'):
        visualdynamics.import_file(str(out))
    np.savez(out, f=f, cpsd=cpsd[:, :1, :])
    with pytest.raises(ValueError, match='not \\(lines, n, n\\)'):
        visualdynamics.import_file(str(out))


def test_an_exodus_frame_tag_this_reader_does_not_know_reads_as_rectangular(
        tmp_path):
    """The tags are R, C and S; a spherical frame comes back spherical,
    and a tag from a tool with its own vocabulary is read as
    rectangular rather than refused — the frame's axes are still
    there to place nodes with."""
    import netCDF4

    source = visualdynamics.import_file(plate('geometry.exo'))
    source.add_coordinate_system(origin=[0.1, 0.0, 0.0], cs_type=2)
    path = str(tmp_path / 'tags.exo')
    visualdynamics.export_file(source, path, format='exodus')
    with netCDF4.Dataset(path) as ds:
        assert [t.decode() for t in ds.variables['frame_tags'][:]] == ['R', 'S']
    assert list(visualdynamics.import_file(path).cs_type) == [0, 2]
    with netCDF4.Dataset(path, 'a') as ds:
        ds.variables['frame_tags'][1] = b'Z'
    assert list(visualdynamics.import_file(path).cs_type) == [0, 0]


def test_an_exodus_frame_with_a_point_on_its_origin_is_refused_by_name(
        tmp_path):
    import netCDF4

    source = visualdynamics.import_file(plate('geometry.exo'))
    frame = source.add_coordinate_system(origin=[0.1, 0.0, 0.0])
    path = str(tmp_path / 'degenerate.exo')
    visualdynamics.export_file(source, path, format='exodus')
    with netCDF4.Dataset(path, 'a') as ds:
        coords = np.asarray(ds.variables['frame_coordinates'][:]).reshape(-1, 9)
        coords[1, 3:6] = coords[1, 0:3]          # the Z point on the origin
        ds.variables['frame_coordinates'][:] = coords.ravel()
    with pytest.raises(ValueError, match=f'frame {frame} .*on its origin'):
        visualdynamics.import_file(path)
    with netCDF4.Dataset(path, 'a') as ds:
        coords = np.asarray(ds.variables['frame_coordinates'][:]).reshape(-1, 9)
        coords[1, 3:6] = coords[1, 0:3] + [0, 0, 1]
        coords[1, 6:9] = coords[1, 0:3] + [0, 0, 2]   # the XZ point on Z
        ds.variables['frame_coordinates'][:] = coords.ravel()
    with pytest.raises(ValueError, match=f'frame {frame} .*on its Z axis'):
        visualdynamics.import_file(path)


def test_an_assignment_set_naming_a_frame_that_is_not_there_is_ignored(
        tmp_path):
    """A node set called `def_cs_9999` in a file whose frames never
    defined 9999 is somebody else's set: the nodes stay global."""
    import netCDF4

    source = visualdynamics.import_file(plate('geometry.exo'))
    local = source.add_coordinate_system(origin=[0.1, 0.0, 0.0])
    source.node_def_cs[:3] = local
    path = str(tmp_path / 'stray.exo')
    visualdynamics.export_file(source, path, format='exodus')
    with netCDF4.Dataset(path, 'a') as ds:
        width = ds.variables['ns_names'].shape[1]
        ds.variables['ns_names'][0] = np.array(
            list('def_cs_9999'.ljust(width, '\0')), dtype='S1')
    back = visualdynamics.import_file(path)
    assert set(back.node_def_cs) == {1}, 'nobody is placed in a frame the file lacks'
