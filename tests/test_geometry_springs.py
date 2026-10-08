"""Springs and ground on a geometry (Brandon, 2026-10-08: a spring is a
kind of line element, ground a kind of point, so the two-beam
substructuring cases can be built in the window as well as by script).

A group of two-node lines given a stiffness in some of the six global
directions is a set of springs, `Model.add_spring` per direction; a
group of points given a stiffness is springs to ground; a group of
points given ground holds its nodes in all six directions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import edit_category, select_objects
from PySide6.QtCore import Qt
from test_fem import ALUMINUM
from test_solve_modes import _column, _set
from test_springs import ROD, _two_beams, _whole_beam

import visualdynamics
from visualdynamics.core.fem import GroupProperties, Model
from visualdynamics.gui.object_tables import element_group_table_model

EDIT = Qt.ItemDataRole.EditRole
POINT, LINE = 161, 21
SIX = ('X+', 'Y+', 'Z+', 'RX+', 'RY+', 'RZ+')


def _beams_geometry(model):
    """A model's beams as a geometry, every beam group given the rod."""
    geometry = model.geometry()
    geometry.group_properties = {int(g): GroupProperties(ALUMINUM, section=ROD)
                                 for g in geometry.group_id}
    return geometry


def _group(geometry, name, nodes, code, props):
    """A new element group of one element per node list, given `props`."""
    group = geometry.add_group(name)
    geometry.add_elements(nodes, [code] * len(nodes), [group] * len(nodes))
    geometry.group_properties[group] = props
    return group


def test_a_spring_line_is_the_springs_add_spring_makes():
    """The two half-beams joined at their coincident nodes by a spring
    group of five directions solve to the model built with add_spring,
    to the last digit: one line, five springs."""
    joints = ('X+', 'Y+', 'Z+', 'RX+', 'RZ+')
    by_script = _two_beams(joints, stiffness=1e5)
    halves = _two_beams((), stiffness=1e5)
    geometry = _beams_geometry(halves)
    stiffness = tuple(1e5 if d in joints else None for d in SIX)
    _group(geometry, 'joint', [[104, 200]], LINE,
           GroupProperties(stiffness=stiffness))
    built = Model.from_geometry(geometry)
    assert len(built.springs) == 5
    assert built.eigensolution(num_modes=14).frequency == pytest.approx(
        by_script.eigensolution(num_modes=14).frequency, rel=1e-9, abs=1e-6)


def test_a_ground_point_holds_its_node_as_fixed_does():
    """A ground point on a beam's end node makes the cantilever, the
    same modes as the free beam solved with that node fixed."""
    free = _whole_beam()
    geometry = _beams_geometry(free)
    _group(geometry, 'support', [[1]], POINT, GroupProperties(ground=True))
    built = Model.from_geometry(geometry)
    assert built.grounds == {1: (0, 1, 2, 3, 4, 5)}
    expected = free.eigensolution(num_modes=8, fixed=['1']).frequency
    got = built.eigensolution(num_modes=8).frequency
    assert got == pytest.approx(expected, rel=1e-9)
    assert got[0] > 0.0, 'no rigid-body modes left'


def test_a_spring_to_ground_is_a_point_or_a_line_to_a_ground_point():
    """Two ways to draw one thing: a point in a group of springs at the
    beam's end, or a line from that end to a ground point at a node of
    its own. Both are add_spring with no second end."""
    k = 4e3
    expected_model = _whole_beam()
    for node in (1, 9):
        expected_model.add_spring(f'{node}Z+', None, k)
    expected = expected_model.eigensolution(num_modes=10).frequency

    on_points = _beams_geometry(_whole_beam())
    _group(on_points, 'end springs', [[1], [9]], POINT,
           GroupProperties(stiffness=(None, None, k, None, None, None)))
    assert Model.from_geometry(on_points).eigensolution(
        num_modes=10).frequency == pytest.approx(expected, rel=1e-9, abs=1e-6)

    to_ground = _beams_geometry(_whole_beam())
    far = [to_ground.add_node(to_ground.node_xyz[to_ground.node_index([n])[0]])
           for n in (1, 9)]
    _group(to_ground, 'ground', [[n] for n in far], POINT,
           GroupProperties(ground=True))
    _group(to_ground, 'end springs', [[1, far[0]], [9, far[1]]], LINE,
           GroupProperties(stiffness=(None, None, k, None, None, None)))
    assert Model.from_geometry(to_ground).eigensolution(
        num_modes=10).frequency == pytest.approx(expected, rel=1e-9, abs=1e-6)


def test_a_mass_on_springs_rings_where_the_textbook_says():
    """A point mass hung from a ground point by a spring line stiff in
    X, Y and Z: three modes at sqrt(k/m)/2pi. Its node is no beam's, so
    its rotations have nothing to act on and are grounded, as a solid's
    are; left free, the solve could not factor."""
    m, k = 0.5, 2e4
    geometry = visualdynamics.Geometry(
        node_id=[1, 2], node_xyz=[[0, 0, 0], [0, 0, 0.1]], length_unit='m')
    _group(geometry, 'ground', [[1]], POINT, GroupProperties(ground=True))
    _group(geometry, 'mass', [[2]], POINT, GroupProperties(mass=m))
    _group(geometry, 'mount', [[1, 2]], LINE,
           GroupProperties(stiffness=(k, k, k, None, None, None)))
    frequency = Model.from_geometry(geometry).eigensolution().frequency
    assert frequency == pytest.approx([math.sqrt(k / m) / (2 * math.pi)] * 3)


def test_springs_and_ground_are_refused_by_what_is_wrong():
    geometry = _beams_geometry(_whole_beam())
    group = _group(geometry, 'joint', [[1, 2]], LINE,
                   GroupProperties(stiffness=(None,) * 6))
    with pytest.raises(ValueError, match='no stiffness in any direction'):
        Model.from_geometry(geometry)
    geometry.group_properties[group] = GroupProperties(ground=True)
    with pytest.raises(ValueError, match='a ground group holds point elements'):
        Model.from_geometry(geometry)
    beams = int(geometry.group_id[0])
    geometry.group_properties[group] = GroupProperties(ALUMINUM, section=ROD)
    geometry.group_properties[beams] = GroupProperties(
        stiffness=(1.0,) + (None,) * 5)
    plate = visualdynamics.Geometry(
        node_id=[1, 2, 3, 4], node_xyz=[[0, 0, 0], [1, 0, 0], [1, 1, 0],
                                        [0, 1, 0]], length_unit='m')
    _group(plate, 'skin', [[1, 2, 3, 4]], 44,
           GroupProperties(stiffness=(1.0,) + (None,) * 5))
    with pytest.raises(ValueError, match='a group of springs holds two-node'):
        Model.from_geometry(plate)
    # a rotation spring needs a node something rotational holds, or ground
    hung = visualdynamics.Geometry(
        node_id=[1, 2], node_xyz=[[0, 0, 0], [0, 0, 0.1]], length_unit='m')
    _group(hung, 'ground', [[1]], POINT, GroupProperties(ground=True))
    _group(hung, 'mass', [[2]], POINT, GroupProperties(mass=1.0))
    _group(hung, 'mount', [[1, 2]], LINE,
           GroupProperties(stiffness=(1e3, 1e3, 1e3, None, 5.0, None)))
    with pytest.raises(ValueError, match='spring acts on a rotation of node 2'):
        Model.from_geometry(hung).eigensolution()


def test_springs_and_ground_ride_the_native_file(tmp_path):
    geometry = _beams_geometry(_whole_beam())
    springs = _group(geometry, 'springs', [[1]], POINT,
                     GroupProperties(stiffness=(None, None, 4e3, None, 7.0,
                                                None)))
    ground = _group(geometry, 'ground', [[9]], POINT,
                    GroupProperties(ground=True))
    visualdynamics.save(geometry, tmp_path / 'g.vdyn')
    back = visualdynamics.load(tmp_path / 'g.vdyn')
    assert back.group_properties[springs] == geometry.group_properties[springs]
    assert back.group_properties[ground] == GroupProperties(ground=True)
    assert back.group_properties[ground].kind == 'ground'


def test_the_element_groups_table_takes_a_stiffness_and_ground(qt_app):
    """Typed by direction in the display units, held in SI, journaled as
    the line that rebuilds it; ground is picked from the Material list
    beside rigid, and a material picked over it makes the group that
    material again."""
    from visualdynamics.core.fem import MATERIALS
    from visualdynamics.gui.object_tables import GROUND
    from visualdynamics.units import SYSTEMS

    inch = SYSTEMS['in-slinch-lbf-s']
    geometry = _beams_geometry(_whole_beam())
    group = _group(geometry, 'springs', [[1]], POINT, GroupProperties(mass=1.0))
    row = list(geometry.group_id).index(group)
    model = element_group_table_model(geometry, inch)
    title = 'Stiffness [lbf/in, lbf·in/rad]'
    lines = []
    model.edit_journaled.connect(lines.append)
    _set(model, title, 'Kz=100, Kry=50', row=row)
    along = inch.to_si(100.0, 'force/length')
    about = inch.to_si(50.0, 'force*length')
    props = geometry.group_properties[group]
    assert props.kind == 'spring'
    assert props.stiffness[2] == pytest.approx(along)
    assert props.stiffness[4] == pytest.approx(about)
    assert props.stiffness[0] is None, 'a direction not named is free'
    assert model.data(model.index(row, _column(model, title))) == \
        'Kz=100, Kry=50'
    assert lines[-1] == (f'.group_properties[{group}] = '
                         f'GroupProperties(stiffness={props.stiffness!r})')
    said = []
    model.edit_rejected.connect(said.append)
    assert not model.setData(model.index(row, _column(model, title)),
                             'Kq=5', EDIT)
    assert 'given by direction' in said[-1]
    assert not model.setData(model.index(row, _column(model, 'Material')),
                             'my alloy', EDIT)
    assert 'takes a stiffness alone' in said[-1]
    # ground, from the Material list
    _set(model, title, '', row=row)
    assert group not in geometry.group_properties
    material = _column(model, 'Material')
    assert GROUND in model.columns[material].choices_for(geometry, row)
    _set(model, 'Material', GROUND, row=row)
    assert geometry.group_properties[group] == GroupProperties(ground=True)
    assert lines[-1] == f'.group_properties[{group}] = GroupProperties(ground=True)'
    assert model.data(model.index(row, material)) == GROUND
    _set(model, 'Material', '6061-T6', row=row)
    assert geometry.group_properties[group].kind == 'solid'
    assert geometry.group_properties[group].material is MATERIALS['6061-T6']


def test_the_tree_offers_points_before_there_are_any(window, pump):
    """A ground point or a spring to ground starts from a point, and a
    family the tree hid until it held one could not be added to."""
    geometry = _beams_geometry(_whole_beam())
    assert POINT not in geometry.elem_type.tolist()
    window.add_object('Geometry', geometry)
    select_objects(window, pump, 'Geometry')
    edit_category(window, pump, 'Points')
    window.set_add_mode(True)
    assert window.element_type == (POINT, 1)
    screen = window._projector.screen()[0]
    window.hover_at(*screen[0])
    window._add_at(*screen[0])
    assert int(geometry.elem_type[-1]) == POINT


def test_the_tie_menu_and_ties_leave_springs_and_ground_alone(window, pump,
                                                               monkeypatch):
    from test_tie import _floor_and_foot, _select_rows

    from visualdynamics.core import mesh

    geometry = _floor_and_foot()
    _group(geometry, 'mounts', [[int(geometry.node_id[0])]], POINT,
           GroupProperties(stiffness=(1.0,) + (None,) * 5))
    _group(geometry, 'support', [[int(geometry.node_id[1])]], POINT,
           GroupProperties(ground=True))
    with pytest.raises(ValueError, match="'mounts' is springs, not rigid"):
        mesh.tie(geometry, geometry.elements_in('foot')[:1], 'floor',
                 group='mounts')
    window.add_object('Geometry', geometry)
    select_objects(window, pump, 'Geometry')
    edit_category(window, pump, 'Quads')
    rows = [int(np.flatnonzero(geometry.elem_id == e)[0])
            for e in geometry.elements_in('foot')]
    _select_rows(window, pump, rows[:1])
    popped = []
    monkeypatch.setattr(window, '_pop_menu', popped.append)
    window.tie_action.trigger()
    labels = [action.text() for action in popped[0].actions()]
    assert labels == ['Tie to the nearest nodes of', 'floor', 'foot', '',
                      'A second selection…']
