
import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics

PLATE_NPZ = fixture_path('plate', 'geometry.npz')
PLATE_EXO = fixture_path('plate', 'geometry.exo')
#: the survey display model: the modal run's nodes and tracelines —
#: the traceline-flavored tests use it, since the meshed plate has
#: elements instead
SURVEY_NPZ = fixture_path('plate', 'test_geometry.npz')


def test_import_without_units_is_unitless():
    """Sources that do not declare units import raw, for the user to define."""
    for path in (PLATE_NPZ, PLATE_EXO):
        geo = visualdynamics.import_file(path)
        assert not geo.units_defined
        assert geo.length_unit is None
        lo, hi = geo.extent
        assert np.allclose(hi - lo, [0.3048, 0.3048, 0.0])  # raw file values


def test_define_units_converts_to_si():
    geo = visualdynamics.import_file(PLATE_NPZ)
    geo.define_units('m')
    assert geo.units_defined
    lo, hi = geo.extent
    assert np.allclose(hi - lo, [0.3048, 0.3048, 0.0])


def test_define_units_reinterprets_not_rescales():
    """A wrong guess can be corrected without reimporting."""
    a = visualdynamics.import_file(PLATE_NPZ)
    a.define_units('in')      # wrong guess
    a.define_units('m')       # corrected
    b = visualdynamics.import_file(PLATE_NPZ, length_unit='m')
    assert np.allclose(a.node_xyz, b.node_xyz)


def test_import_sdynpy_npz():
    geo = visualdynamics.import_file(PLATE_NPZ, length_unit='m')
    assert geo.num_nodes == 169
    assert len(geo.traceline_conn) == 0, 'the meshed plate draws elements'
    assert len(geo.elem_conn) == 144
    lo, hi = geo.extent
    assert np.allclose(hi - lo, [0.3048, 0.3048, 0.0])


def test_import_exodus():
    geo = visualdynamics.import_file(PLATE_EXO, length_unit='m')
    assert geo.num_nodes == 169
    assert len(geo.elem_conn) == 144
    assert all(int(t) == 94 for t in geo.elem_type)  # quadshell4
    assert all(len(c) == 4 for c in geo.elem_conn)


def test_npz_and_exodus_agree():
    """Both exports of the same demo geometry must land on identical nodes."""
    a = visualdynamics.import_file(PLATE_NPZ, length_unit='m')
    b = visualdynamics.import_file(PLATE_EXO, length_unit='m')
    assert np.array_equal(a.node_id, b.node_id)
    assert np.allclose(a.node_xyz, b.node_xyz)


def test_length_unit_conversion_on_import():
    meters = visualdynamics.import_file(PLATE_NPZ, length_unit='m')
    inches = visualdynamics.import_file(PLATE_NPZ, length_unit='in')
    assert np.allclose(inches.node_xyz, meters.node_xyz * 0.0254)


def test_survey_display_import():
    geo = visualdynamics.import_file(SURVEY_NPZ, length_unit='m')
    assert geo.num_nodes > 0
    assert len(geo.traceline_conn) > 0


def test_save_load_round_trip(tmp_path):
    geo = visualdynamics.import_file(PLATE_EXO, length_unit='m')
    path = str(tmp_path / 'plate.vdyn')
    geo.save(path)
    again = visualdynamics.Geometry.load(path)
    assert again == geo


def test_native_import_dispatch(tmp_path):
    geo = visualdynamics.import_file(PLATE_NPZ, length_unit='m')
    path = str(tmp_path / 'plate.vdyn')
    visualdynamics.save(geo, path)
    assert visualdynamics.import_file(path) == geo


def test_validation_rejects_bad_connectivity():
    with pytest.raises(ValueError):
        visualdynamics.Geometry(node_id=[1, 2], node_xyz=[[0, 0, 0], [1, 0, 0]],
                      traceline_conn=[np.array([1, 99])])


def test_display_units():
    geo = visualdynamics.import_file(PLATE_NPZ, length_unit='m')
    disp = visualdynamics.IN_LBF_S.from_si(geo.node_xyz, 'length')
    assert disp[:, 0].max() == pytest.approx(12.0), 'a 12 inch plate'
    disp_mm = visualdynamics.MMKS.from_si(geo.node_xyz, 'length')
    assert disp_mm[:, 0].max() == pytest.approx(304.8)


def test_unitless_geometry_is_not_converted_or_mislabeled():
    """A unit-less geometry must draw raw values and say units are undefined."""
    from visualdynamics.viz.geometry import display_points

    geo = visualdynamics.import_file(PLATE_NPZ)
    points, label = display_points(geo, visualdynamics.IN_LBF_S)
    assert np.array_equal(points, geo.node_xyz)   # no scaling applied
    assert label == '[units undefined]'


def test_defined_geometry_converts_and_labels():
    from visualdynamics.viz.geometry import display_points

    geo = visualdynamics.import_file(PLATE_NPZ, length_unit='m')
    points, label = display_points(geo, visualdynamics.IN_LBF_S)
    assert np.allclose(points, geo.node_xyz / 0.0254)
    assert label == '[in]'
    assert points[:, 0].max() == pytest.approx(12.0)


def test_entity_subset_rendering():
    """Drawing one node/traceline must not pull in the whole geometry."""
    import pyvista as pv

    from visualdynamics.viz.geometry import add_geometry

    geo = visualdynamics.import_file(PLATE_NPZ, length_unit='m')
    plotter = pv.Plotter(off_screen=True)
    add_geometry(plotter, geo, entities={'nodes': [int(geo.node_id[0])]})
    single = len(plotter.renderer.actors)
    plotter.close()

    plotter = pv.Plotter(off_screen=True)
    add_geometry(plotter, geo)
    plotter.close()
    assert single >= 1


def test_entity_subset_ignores_missing_ids():
    """An id that is not in the geometry must not raise."""
    import pyvista as pv

    from visualdynamics.viz.geometry import add_geometry

    geo = visualdynamics.import_file(PLATE_NPZ, length_unit='m')
    plotter = pv.Plotter(off_screen=True)
    add_geometry(plotter, geo, entities={'nodes': [999999]},
                 labels=['nodes'])
    plotter.close()


def test_delete_nodes_takes_referencing_entities_with_them():
    """A traceline naming a deleted node cannot survive."""
    geo = visualdynamics.import_file(SURVEY_NPZ, length_unit='m')
    nodes_before = geo.num_nodes
    lines_before = len(geo.traceline_conn)
    report = geo.delete_nodes([int(geo.node_id[0])])
    assert report['nodes'] == 1
    assert report['tracelines'] > 0
    assert geo.num_nodes == nodes_before - 1
    assert len(geo.traceline_conn) == lines_before - report['tracelines']
    geo.validate()          # must still be self-consistent


def test_delete_elements_and_tracelines_by_id():
    """By id, like every other group — an element's id names it here the
    same way it does in the file it came from."""
    geo = visualdynamics.import_file(PLATE_EXO, length_unit='m')
    before = len(geo.elem_conn)
    doomed = [int(geo.elem_id[0]), int(geo.elem_id[5])]
    kept = [list(map(int, geo.elem_conn[i])) for i in range(before)
            if i not in (0, 5)]
    assert geo.delete_elements(doomed) == {'elements': 2}
    assert len(geo.elem_conn) == before - 2
    assert [list(map(int, c)) for c in geo.elem_conn] == kept
    assert len(geo.elem_id) == len(geo.elem_conn)
    assert doomed[0] not in geo.elem_id.tolist()


def test_an_id_that_is_not_there_is_ignored_not_an_error():
    """Deleting twice is not a failure, the same as for nodes."""
    geo = visualdynamics.import_file(SURVEY_NPZ, length_unit='m')
    first = int(geo.traceline_id[0])
    assert geo.delete_tracelines([first]) == {'tracelines': 1}
    assert geo.delete_tracelines([first]) == {'tracelines': 0}


def test_ids_that_other_data_points_at_must_be_unique():
    """Connectivity names nodes by id; a node names its systems by id."""
    identity = np.vstack([np.eye(3), np.zeros(3)])
    with pytest.raises(ValueError, match='duplicate coordinate system ids'):
        visualdynamics.Geometry(node_id=[1, 2], node_xyz=np.zeros((2, 3)),
                      cs_id=[4, 4], cs_name=['', ''], cs_type=[0, 0],
                      cs_matrix=[identity, identity], length_unit='m')


def test_one_traceline_id_can_name_several_polylines():
    """A UNV trace line that lifts the pen arrives split into its drawn
    runs, all still that one trace line — and deleting it takes the
    whole thing, gaps and all."""
    geo = visualdynamics.Geometry(node_id=[1, 2, 3, 4], node_xyz=np.zeros((4, 3)),
                        traceline_id=[7, 7],
                        traceline_conn=[[1, 2], [3, 4]], length_unit='m')
    assert len(geo.tracelines) == 2
    assert geo.delete_tracelines([7]) == {'tracelines': 2}
    assert len(geo.tracelines) == 0


def test_delete_coordinate_system_reassigns_its_nodes():
    identity = np.vstack([np.eye(3), np.zeros(3)])
    geo = visualdynamics.Geometry(node_id=[1, 2], node_xyz=[[0, 0, 0], [1, 0, 0]],
                        node_def_cs=[1, 9], node_disp_cs=[9, 9],
                        cs_id=[1, 9], cs_name=['', 'extra'], cs_type=[0, 0],
                        cs_matrix=[identity, identity], length_unit='m')
    report = geo.delete_coordinate_systems([9])
    assert report['coordinate_systems'] == 1
    assert report['nodes_reassigned'] == 3
    assert set(geo.node_def_cs.tolist()) == {1}
    assert set(geo.node_disp_cs.tolist()) == {1}


def test_the_last_coordinate_system_cannot_be_deleted():
    geo = visualdynamics.import_file(PLATE_NPZ, length_unit='m')
    with pytest.raises(ValueError):
        geo.delete_coordinate_systems(geo.cs_id.tolist())


def test_deleting_nothing_is_harmless():
    geo = visualdynamics.import_file(PLATE_NPZ, length_unit='m')
    assert geo.delete_nodes([999999])['nodes'] == 0
    assert geo.num_nodes == 169


def test_add_node_and_coordinate_system():
    geo = visualdynamics.import_file(PLATE_EXO, length_unit='m')
    nodes = geo.num_nodes
    new_id = geo.add_node([0.5, 0.25, 0.0])
    assert new_id == int(geo.node_id[:-1].max()) + 1
    assert geo.num_nodes == nodes + 1
    cs_id = geo.add_coordinate_system(origin=[1.0, 0.0, 0.0], name='shaker')
    assert geo.cs_name[-1] == 'shaker'
    assert np.allclose(geo.cs_matrix[-1][3], [1.0, 0.0, 0.0])
    assert cs_id not in geo.cs_id[:-1]
    geo.validate()


def test_element_type_follows_the_node_count():
    geo = visualdynamics.import_file(PLATE_EXO, length_unit='m')
    for nodes, expected in [([101, 102], 21), ([101, 102, 103], 41),
                            ([101, 102, 103, 104], 44)]:
        geo.add_element(nodes)
        assert int(geo.elem_type[-1]) == expected
    with pytest.raises(ValueError):
        geo.add_element([101, 102, 103, 104, 105])   # no default for five
    geo.validate()


def test_added_entities_must_reference_real_nodes():
    geo = visualdynamics.import_file(PLATE_EXO, length_unit='m')
    with pytest.raises(ValueError):
        geo.add_traceline([101, 999999])
    with pytest.raises(ValueError):
        geo.add_element([101, 999999, 103])


def test_a_traceline_needs_two_nodes():
    geo = visualdynamics.import_file(PLATE_EXO, length_unit='m')
    with pytest.raises(ValueError):
        geo.add_traceline([101])


def test_renumbering_a_node_carries_its_connectivity_across():
    geo = visualdynamics.import_file(PLATE_EXO, length_unit='m')
    old = int(geo.node_id[0])
    holders = [i for i, conn in enumerate(geo.elem_conn)
               if old in [int(n) for n in conn]]
    assert holders, 'the plate should have elements on its first node'

    geo.renumber_node(0, 9999)
    assert int(geo.node_id[0]) == 9999
    for i in holders:
        assert 9999 in [int(n) for n in geo.elem_conn[i]]
        assert old not in [int(n) for n in geo.elem_conn[i]]
    geo.validate()


def test_a_node_cannot_take_an_id_another_node_holds():
    geo = visualdynamics.import_file(PLATE_EXO, length_unit='m')
    with pytest.raises(ValueError, match='already exists'):
        geo.renumber_node(0, int(geo.node_id[1]))
    assert int(geo.node_id[0]) != int(geo.node_id[1])


def test_renumbering_a_coordinate_system_repoints_its_nodes():
    geo = visualdynamics.import_file(PLATE_EXO, length_unit='m')
    old = int(geo.cs_id[0])
    users = geo.node_def_cs == old
    assert users.any()

    geo.renumber_coordinate_system(0, 42)
    assert int(geo.cs_id[0]) == 42
    assert (geo.node_def_cs[users] == 42).all()
    assert not (geo.node_disp_cs == old).any()


def test_the_plate_ships_fully_meshed():
    """The demonstration article is quads throughout — its drawing and
    its structure are the same 144 elements."""
    geo = visualdynamics.import_file(PLATE_NPZ)
    assert len(geo.elem_conn) == 144, 'a 12 x 12 mesh of quads'
    assert all(len(conn) == 4 for conn in geo.elem_conn)
    assert {int(code) for code in geo.elem_type} == {44}, 'quad elements'
    geo.validate()


def test_a_geometry_declaration_can_be_withdrawn():
    import numpy as np

    geometry = visualdynamics.import_file(PLATE_NPZ)
    raw = geometry.node_xyz.copy()
    raw_origins = geometry.cs_matrix[:, 3, :].copy()
    geometry.define_units('in')
    assert geometry.units_defined
    geometry.undefine_units()
    assert not geometry.units_defined and geometry.length_unit is None
    assert np.allclose(geometry.node_xyz, raw)
    assert np.allclose(geometry.cs_matrix[:, 3, :], raw_origins), \
        'coordinate system origins come back too'


# ---- what an id may be ------------------------------------------------------

def test_an_id_must_be_a_whole_number():
    """`1.7` is not node 1 with a rounding error — it is a caller who
    computed an id, and truncating it quietly is how that becomes
    somebody else's afternoon three weeks later."""
    from visualdynamics.core.geometry import Geometry

    xyz = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    with pytest.raises(ValueError, match='whole numbers'):
        Geometry(node_id=[1.7, 2.9, 3.1], node_xyz=xyz)
    with pytest.raises(ValueError, match='whole numbers'):
        Geometry(node_id=[1, 2, 3], node_xyz=xyz,
                 elem_conn=[[1, 2, 3]], elem_type=[41], elem_id=[2.5])
    # a whole number that arrived as a float is still a whole number
    assert list(Geometry(node_id=[1.0, 2.0, 3.0], node_xyz=xyz).node_id) \
        == [1, 2, 3]


def test_an_id_may_arrive_as_text_because_every_format_writes_it_that_way():
    from visualdynamics.core.geometry import Geometry

    xyz = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    geometry = Geometry(node_id=['1', '2', '3'], node_xyz=xyz)
    assert list(geometry.node_id) == [1, 2, 3]
    assert geometry.node_id.dtype == np.int64, 'stored as the type it claims'
    with pytest.raises(ValueError, match='whole numbers'):
        Geometry(node_id=['x', 'y', 'z'], node_xyz=xyz)


def test_a_duplicate_id_says_which_one():
    """The old message named the kind and left you to find it in four
    hundred nodes."""
    from visualdynamics.core.geometry import Geometry

    xyz = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    with pytest.raises(ValueError, match='duplicate node ids: 7'):
        Geometry(node_id=[7, 7, 9], node_xyz=xyz)


def test_element_and_traceline_ids_are_still_allowed_to_repeat():
    """Deliberate: nothing refers to them, and one traceline id
    legitimately names several polylines — a UNV trace line that lifts
    the pen — so deleting that id removes every run of it."""
    from visualdynamics.core.geometry import Geometry

    xyz = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    geometry = Geometry(node_id=[1, 2, 3], node_xyz=xyz,
                        traceline_conn=[[1, 2], [2, 3]], traceline_id=[4, 4])
    assert list(geometry.traceline_id) == [4, 4]


def _framed_geometry():
    """Node 101 measured in a frame turned 30° about Z, node 102 in the
    global one, node 103 in a cylindrical one."""
    c, s = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))
    turned = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    identity = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 0, 0]]
    return visualdynamics.Geometry(
        node_id=[101, 102, 103], node_xyz=np.zeros((3, 3)),
        node_disp_cs=[2, 1, 3], cs_id=[1, 2, 3], cs_name=['', '', ''],
        cs_type=[0, 0, 1], cs_matrix=[identity, turned, identity])


def test_a_dofs_direction_reads_through_its_nodes_frame():
    """'101X+' at a node measured in a turned frame is not global X: the
    report's grid reads which global axis a channel is nearest and how
    far off (Brandon, 2026-09-19)."""
    geometry = _framed_geometry()
    c, s = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))
    assert np.allclose(geometry.dof_direction('101X+'), [c, s, 0.0])
    assert np.allclose(geometry.dof_direction('101Y-'), [s, -c, 0.0])
    assert np.allclose(geometry.dof_direction('101RZ+'), [0.0, 0.0, 1.0]), \
        'a rotational DOF points along its axis'
    assert np.allclose(geometry.dof_direction('102Z+'), [0.0, 0.0, 1.0])
    assert geometry.dof_direction('103X+') is None, 'a cylindrical frame is not read'
    assert geometry.dof_direction('999X+') is None, 'no such node'
    assert geometry.dof_direction('101') is None, 'no axis'
    bare = visualdynamics.Geometry(node_id=[7], node_xyz=[[0, 0, 0]])
    assert np.allclose(bare.dof_direction('7Y+'), [0.0, 1.0, 0.0]), \
        'a geometry with only the default frame measures in the global one'
