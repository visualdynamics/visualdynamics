"""Nastran springs and supports, and ground in chosen directions
(2026-10-08). A CELAS1 or CELAS2 reads as a spring element carrying its
stiffness, one element group per stiffness and direction; an SPC or
SPC1 reads as ground points held in its components; a deck written back
says both. And a ground holds the directions it is given, not all six
only, so a pinned support is a ground of X, Y and Z."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import edit_category, select_objects
from test_geometry_springs import POINT, _beams_geometry, _group
from test_solve_modes import _column, _set
from test_springs import _whole_beam

import visualdynamics
from visualdynamics.core.fem import AXES, GroupProperties, Model
from visualdynamics.gui.object_tables import element_group_table_model
from visualdynamics.io import nastran

DECK = """BEGIN BULK
GRID,1,,0.,0.,0.
GRID,2,,1.,0.,0.
GRID,3,,2.,0.,0.
CONM2,11,1,0,2.0
CONM2,12,2,0,2.0
CONM2,13,3,0,2.0
CELAS2,21,8.+4,1,1,2,1
CELAS1,22,7,2,1,3,1
PELAS,7,3.+4
CELAS2,23,1.+4,1,1
CELAS2,24,5.+2,3,5
SPC1,1,23456,1,2,3
ENDDATA
"""


def _springs(model):
    """(ends, stiffness) for every spring, comparable across models."""
    return sorted(((s.node_a, s.direction_a, s.node_b or 0, s.direction_b),
                   s.stiffness) for s in model.springs)


def test_celas_cards_read_as_springs_with_their_stiffness(tmp_path):
    deck = tmp_path / 'springs.bdf'
    deck.write_text(DECK)
    geometry = nastran.load(str(deck))
    names = dict(zip(geometry.group_id.tolist(), geometry.group_name))
    of = {int(e): int(g) for e, g in zip(geometry.elem_id, geometry.elem_group)}
    kind = {int(e): int(t) for e, t in zip(geometry.elem_id, geometry.elem_type)}
    conn = {int(e): list(map(int, c)) for e, c in zip(geometry.elem_id,
                                                       geometry.elem_conn)}
    assert conn[21] == [1, 2] and kind[21] == 136, 'two grids, both kept'
    assert conn[22] == [2, 3], 'a CELAS1 grid is its third field'
    assert conn[23] == [1] and kind[23] == 138, 'one grid: to ground'
    assert kind[24] == 139, 'a rotation to ground'
    assert names[of[21]] == 'CELAS 80000 X'
    assert names[of[22]] == 'CELAS 30000 X', 'the PELAS stiffness'
    props = geometry.group_properties[of[24]]
    assert props.stiffness == (None, None, None, None, 500.0, None)
    # what it builds is what add_spring would
    model = Model.from_geometry(geometry)
    expected = [((1, 1, 2, 1), 8e4), ((1, 1, 0, 0), 1e4),
                ((2, 1, 3, 1), 3e4), ((3, 5, 0, 0), 500.0)]
    got = _springs(model)
    assert sorted(got) == sorted(expected)


def test_a_deck_written_back_carries_its_springs_and_supports(tmp_path):
    deck = tmp_path / 'springs.bdf'
    deck.write_text(DECK)
    geometry = nastran.load(str(deck))
    out = tmp_path / 'out.bdf'
    nastran.save(geometry, str(out))
    text = out.read_text()
    assert 'CELAS2' in text and 'SPC1' in text
    back = nastran.load(str(out))
    first, again = Model.from_geometry(geometry), Model.from_geometry(back)
    assert _springs(again) == _springs(first)
    assert again.grounds == first.grounds
    assert again.eigensolution().frequency == pytest.approx(
        first.eigensolution().frequency, rel=1e-6, abs=1e-9)


def test_a_stiffness_written_short_reads_back_whole(tmp_path):
    """Eight characters cut '1.23457E+07' to '1.23457E', which no reader
    parses back; the short Nastran form keeps the exponent."""
    geometry = _beams_geometry(_whole_beam())
    _group(geometry, 'mount', [[1]], POINT,
           GroupProperties(stiffness=(None, None, 12345678.9, None, None,
                                      None)))
    out = tmp_path / 'k.bdf'
    nastran.save(geometry, str(out))
    back = nastran.load(str(out))
    stiffness = [p.stiffness[2] for p in back.group_properties.values()
                 if p.kind == 'spring']
    assert stiffness == [pytest.approx(12345678.9, rel=1e-4)]


def test_spc_cards_read_as_ground_in_their_components(tmp_path):
    deck = tmp_path / 'held.bdf'
    deck.write_text(DECK)
    geometry = nastran.load(str(deck))
    grounds = [(geometry.group_name[k], geometry.group_properties[int(g)])
               for k, g in enumerate(geometry.group_id)
               if geometry.group_properties.get(int(g)) is not None
               and geometry.group_properties[int(g)].kind == 'ground']
    assert [name for name, _p in grounds] == ['SPC 23456']
    assert grounds[0][1].ground == ('Y', 'Z', 'RX', 'RY', 'RZ')
    assert Model.from_geometry(geometry).grounds == {
        node: (1, 2, 3, 4, 5) for node in (1, 2, 3)}


def test_spc_thru_and_the_cards_refused(tmp_path):
    deck = tmp_path / 'thru.bdf'
    deck.write_text('BEGIN BULK\n' + ''.join(
        f'GRID,{n},,{n}.,0.,0.\n' for n in range(1, 6))
        + 'SPC1,1,123,2,THRU,4\nENDDATA\n')
    geometry = nastran.load(str(deck))
    held = sorted(int(c[0]) for c, t in zip(geometry.elem_conn,
                                            geometry.elem_type)
                  if int(t) == POINT)
    assert held == [2, 3, 4]
    for body, said in (
            ('SPC,1,1,123,0.5\n', 'enforces a displacement'),
            ('SPC1,1,123,1\nSPC1,2,123,2\n', 'SPC sets 1, 2'),
            ('CELAS2,9,1.,1,1,2,2\n',
             'joins component 1 of grid 1 to component 2'),
            ('CELAS2,9,1.,1,7\n', 'components 1 to 6')):
        bad = tmp_path / 'bad.bdf'
        bad.write_text('BEGIN BULK\nGRID,1,,0.,0.,0.\nGRID,2,,1.,0.,0.\n'
                       + body + 'ENDDATA\n')
        with pytest.raises(ValueError, match=said):
            nastran.load(str(bad))


def test_a_ground_holds_the_directions_it_is_given():
    """A beam pinned at both ends (translations held, rotations free) is
    the beam with `fixed` naming those translations; a rotational spring
    on a node held in that rotation is no orphan."""
    assert GroupProperties(ground=True).ground == AXES
    assert GroupProperties(ground=('z', 'X+', 'x')).ground == ('X', 'Z')
    assert GroupProperties(ground=False).kind != 'ground'
    with pytest.raises(ValueError, match='not a direction to hold'):
        GroupProperties(ground=('Q',))
    free = _whole_beam()
    geometry = _beams_geometry(free)
    _group(geometry, 'pins', [[1], [9]], POINT,
           GroupProperties(ground=('X', 'Y', 'Z', 'RX')))
    built = Model.from_geometry(geometry)
    fixed = [f'{n}{d}' for n in (1, 9) for d in ('X+', 'Y+', 'Z+', 'RX+')]
    assert built.eigensolution(num_modes=6).frequency == pytest.approx(
        free.eigensolution(num_modes=6, fixed=fixed).frequency, rel=1e-9)
    hung = visualdynamics.Geometry(
        node_id=[1, 2], node_xyz=[[0, 0, 0], [0, 0, 0.1]], length_unit='m')
    _group(hung, 'held', [[1]], POINT, GroupProperties(ground=True))
    _group(hung, 'twist', [[2]], POINT, GroupProperties(ground=('RX', 'RZ')))
    _group(hung, 'mass', [[2]], POINT, GroupProperties(mass=1.0))
    _group(hung, 'mount', [[1, 2]], 21,
           GroupProperties(stiffness=(1e3, 1e3, 1e3, None, 5.0, None)))
    with pytest.raises(ValueError, match='spring acts on a rotation of node 2'):
        Model.from_geometry(hung).eigensolution()
    hung.group_properties[int(hung.group_id[1])] = GroupProperties(
        ground=('RX', 'RY', 'RZ'))
    assert len(Model.from_geometry(hung).eigensolution().frequency) == 3


def test_ground_directions_ride_the_file_and_a39_reads_as_six(tmp_path):
    import h5py

    geometry = _beams_geometry(_whole_beam())
    pins = _group(geometry, 'pins', [[1]], POINT,
                  GroupProperties(ground=('X', 'Y', 'Z')))
    path = tmp_path / 'g.vdyn'
    visualdynamics.save(geometry, path)
    assert visualdynamics.load(path).group_properties[pins].ground == \
        ('X', 'Y', 'Z')
    with h5py.File(path, 'r+') as f:
        found = [g for name, g in f.items()
                 if isinstance(g, h5py.Group) and 'block_properties' in g]
        holder = (found[0] if found else f)['block_properties'][str(pins)]
        holder.attrs['ground'] = True          # how a39 wrote it
    assert visualdynamics.load(path).group_properties[pins].ground == AXES


def test_the_table_takes_ground_directions(qt_app):
    geometry = _beams_geometry(_whole_beam())
    group = _group(geometry, 'pins', [[1]], POINT, GroupProperties(mass=1.0))
    row = list(geometry.group_id).index(group)
    model = element_group_table_model(geometry)
    lines = []
    model.edit_journaled.connect(lines.append)
    _set(model, 'Ground', 'x, y, z', row=row)
    assert geometry.group_properties[group].ground == ('X', 'Y', 'Z')
    assert model.data(model.index(row, _column(model, 'Ground'))) == 'X, Y, Z'
    assert lines[-1] == (f'.group_properties[{group}] = '
                         "GroupProperties(ground=('X', 'Y', 'Z'))")
    _set(model, 'Ground', 'all', row=row)
    assert model.data(model.index(row, _column(model, 'Ground'))) == 'all'
    assert lines[-1].endswith('GroupProperties(ground=True)')
    _set(model, 'Ground', '', row=row)
    assert group not in geometry.group_properties


def test_two_clicks_on_one_spot_take_both_coincident_nodes(window, pump):
    """Coincident nodes draw as one and answer one pixel: the first
    extending click takes the one not picked yet, then the other, then
    unpicks them last first; the caption says how many are there."""
    halves = _whole_beam()
    geometry = halves.geometry()
    twin = geometry.add_node(geometry.node_xyz[geometry.node_index([5])[0]])
    window.add_object('Geometry', geometry)
    select_objects(window, pump, 'Geometry')
    edit_category(window, pump, 'Beams')
    window.set_add_mode(True)
    screen = window._projector.screen()[0]
    spot = screen[geometry.node_index([5])[0]]
    hovered = window.hover_at(*spot)
    assert hovered in (5, twin)
    other = twin if hovered == 5 else 5
    assert window._hover_caption(geometry) == f'{hovered}, {other} (2 here)'
    window._add_at(*spot)
    window._add_at(*spot, extend=True)
    assert sorted(window._picked_nodes) == sorted([5, twin]), 'both, in turn'
    window._add_at(*spot, extend=True)
    assert window._picked_nodes == [hovered], 'the last one unpicked first'
    window.hover_at(*screen[0])
    assert window._hover_caption(geometry) == '1', 'one node, its id alone'
    assert np.array_equal(geometry.node_xyz[-1],
                          geometry.node_xyz[geometry.node_index([5])[0]])


def test_an_exodus_spring_of_one_node_is_to_ground(tmp_path):
    """Exodus names a spring SPRING whether it has one node or two; one
    node is a spring to ground (138), two a spring between them (136)."""
    import netCDF4

    from visualdynamics.io import exodus

    path = tmp_path / 'springs.exo'
    with netCDF4.Dataset(path, 'w') as ds:
        ds.createDimension('num_dim', 3)
        ds.createDimension('num_nodes', 3)
        ds.createDimension('num_el_blk', 2)
        ds.createDimension('num_el_in_blk1', 1)
        ds.createDimension('num_nod_per_el1', 2)
        ds.createDimension('num_el_in_blk2', 1)
        ds.createDimension('num_nod_per_el2', 1)
        for axis, values in zip('xyz', ([0, 1, 2], [0, 0, 0], [0, 0, 0])):
            ds.createVariable(f'coord{axis}', 'f8', ('num_nodes',))[:] = values
        ds.createVariable('eb_prop1', 'i4', ('num_el_blk',))[:] = [1, 2]
        between = ds.createVariable('connect1', 'i4',
                                    ('num_el_in_blk1', 'num_nod_per_el1'))
        between.elem_type = 'SPRING'
        between[:] = [[1, 2]]
        grounded = ds.createVariable('connect2', 'i4',
                                     ('num_el_in_blk2', 'num_nod_per_el2'))
        grounded.elem_type = 'SPRING'
        grounded[:] = [[3]]
    geometry = exodus.load(path)
    assert sorted(int(t) for t in geometry.elem_type) == [136, 138]


def test_escdf_writes_a_spring_between_two_nodes_as_a_bar(tmp_path):
    """136 is two nodes now; written as the standard's one-node sphere it
    would have lost one."""
    from visualdynamics.io.escdf_objects import _ELEMENT_NAMES

    assert _ELEMENT_NAMES[136] == 'bar2' and _ELEMENT_NAMES[138] == 'sphere1'
