"""Sensors drawn on the geometry: each body centered on its node, turned by
the channel table alone, and moving with its node in an animation.

Brandon, 2026-10-26: a sensor whose make and model are listed is drawn
in the geometry window, aligned with the node's local coordinates, with
nothing added to the model. Brandon, 2026-10-27: the node is the body's
center, so the lines through the centers of a triax's X, Y and Z faces
cross at the node and a uniaxial's axis runs through it; the mounting
surface is never read; the positive face of each sensing axis is drawn
in that axis's arrow color, so a sign wrong in the table shows.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core import sensors as placing
from visualdynamics.core.channel_table import ChannelTable
from visualdynamics.core.sensor_catalog import GENERIC, lookup, models_for
from visualdynamics.units import SI
from visualdynamics.viz import sensors as drawing
from visualdynamics.viz.geometry import AXIS_COLORS

C, S = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))
HERE = np.array([0.3, -0.2, 0.5])


def _geometry(units=True):
    """Node 101 at HERE measured in a frame turned 30 degrees about Z,
    node 102 in the global frame, node 103 in a cylindrical one."""
    turned = [[C, S, 0.0], [-S, C, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    identity = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 0, 0]]
    geometry = visualdynamics.Geometry(
        node_id=[101, 102, 103],
        node_xyz=[HERE, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        node_disp_cs=[2, 1, 3], cs_id=[1, 2, 3], cs_name=['', '', ''],
        cs_type=[0, 0, 1], cs_matrix=[identity, turned, identity])
    if units:
        geometry.define_units('m')
    return geometry


def _table(rows):
    """A table from (node, direction, type, model, triax DOF) rows."""
    columns = list(zip(*rows))
    count = len(rows)
    return ChannelTable({
        'channel': list(range(1, count + 1)), 'node': list(columns[0]),
        'direction': list(columns[1]), 'channel_type': list(columns[2]),
        'unit': ['g' if t == 'acceleration' else 'N' if t == 'force' else 'V'
                 for t in columns[2]],
        'model': list(columns[3]), 'triax_dof': list(columns[4])})


def _triax(node=101, model='356A02', directions=('X+', 'Y+', 'Z+'),
           legs=('X', 'Y', 'Z')):
    return [(node, d, 'acceleration', model, leg)
            for d, leg in zip(directions, legs)]


def _body_points(sensor):
    """The body's points in geometry units, the connector left out: the
    box or prism the node is the center of."""
    points, triangles, parts = drawing.body(sensor.entry)
    used = sorted({int(v) for t, part in zip(triangles, parts)
                   if part != 'connector' for v in t})
    return sensor.center + (points[used] * 1e-3) @ sensor.frame.T


# ---- placement ---------------------------------------------------------------

def test_a_triax_is_centered_on_its_node_and_turned_by_its_legs():
    geometry = _geometry()
    (sensor,), notes = placing.place(_table(_triax()), geometry)
    assert notes == []
    assert sensor.kind == 'triax' and not sensor.generic
    assert sensor.entry is lookup('356A02')
    assert np.array_equal(sensor.center, HERE)
    for column, leg in enumerate('XYZ'):
        assert np.allclose(sensor.frame[:, column],
                           geometry.dof_direction(f'101{leg}+'))
    body = _body_points(sensor)
    assert np.allclose(body.mean(axis=0), HERE, rtol=0, atol=1e-15), \
        'the node is the center of the body'
    # the line through the center of each face, along its leg, meets the
    # node: the face centers sit at +-half the size along each leg
    for column, extent in enumerate(lookup('356A02').size):
        reach = (body - HERE) @ sensor.frame[:, column]
        assert reach.max() == pytest.approx(extent / 2 * 1e-3, abs=1e-12)
        assert reach.min() == pytest.approx(-extent / 2 * 1e-3, abs=1e-12)


def test_two_legs_give_the_third_and_one_leg_stands_on_the_nodes_axes():
    geometry = _geometry()
    two = _triax(directions=('Y+', 'Z+'), legs=('Y', 'Z'))
    (sensor,), _notes = placing.place(_table(two), geometry)
    assert np.allclose(sensor.frame[:, 0], geometry.dof_direction('101X+'))
    for wired, legs, third in ((('X+', 'Z+'), ('X', 'Z'), 1),
                               (('Y+', 'X+'), ('Y', 'X'), 2)):
        (sensor,), _notes = placing.place(
            _table(_triax(directions=wired, legs=legs)), geometry)
        letter = 'XYZ'[third]
        assert np.allclose(sensor.frame[:, third],
                           geometry.dof_direction(f'101{letter}+')), \
            f'the {letter} leg from the other two'
    one = _triax(node=102, directions=('Z-',), legs=('Z',))
    (sensor,), _notes = placing.place(_table(one), geometry)
    assert np.allclose(sensor.frame[:, 2], [0, 0, -1])
    assert np.allclose(sensor.frame[:, 0], [1, 0, 0])
    assert np.linalg.det(sensor.frame) == pytest.approx(1.0)


def test_a_blank_triax_dof_reads_the_rows_own_letter():
    geometry = _geometry()
    rows = _triax(node=102, model='356A01', directions=('Y-', 'X+', 'Z-'),
                  legs=('', '', ''))
    (sensor,), notes = placing.place(_table(rows), geometry)
    assert notes == []
    assert np.allclose(sensor.frame, [[1, 0, 0], [0, -1, 0], [0, 0, -1]])


def test_a_left_handed_or_skewed_triax_is_drawn_generic_and_named():
    geometry = _geometry()
    flipped = _triax(node=102, directions=('X+', 'Y+', 'Z-'))
    (sensor,), notes = placing.place(_table(flipped), geometry)
    assert sensor.generic and sensor.positive == {}
    assert np.allclose(sensor.frame, np.eye(3)), "on the node's own axes"
    assert notes == [('node 102: the triax legs make a left-handed frame, so '
                     "one sign in the table is wrong; drawn on the node's "
                     'own axes')]
    skewed = _triax(node=102, directions=('X+', 'X+', 'Z+'))
    (sensor,), notes = placing.place(_table(skewed), geometry)
    assert sensor.generic
    assert notes == [("node 102: the triax legs are not perpendicular; drawn "
                     "on the node's own axes")]


def test_a_uniaxials_axis_runs_through_its_node():
    geometry = _geometry()
    rows = [(101, 'Y-', 'acceleration', '352A24', ''),
            (102, 'X+', 'force', '208C03', '')]
    (accel, force), notes = placing.place(_table(rows), geometry)
    assert notes == []
    assert (accel.kind, force.kind) == ('uniaxial', 'force')
    assert np.allclose(accel.frame[:, 2], geometry.dof_direction('101Y-'))
    assert accel.positive == {2: 'Y'} and force.positive == {2: 'X'}
    for sensor in (accel, force):
        body = _body_points(sensor)
        axis = sensor.frame[:, 2]
        assert np.allclose(body.mean(axis=0), sensor.center, rtol=0,
                           atol=1e-15)
        # the sensing axis is the line through the center along z: the
        # node is on it, at zero distance
        offset = sensor.center - body.mean(axis=0)
        assert np.linalg.norm(offset - (offset @ axis) * axis) < 1e-15
    # the turn about the axis: the connector toward the node's local +X,
    # or +Y when the axis is along X
    assert np.allclose(accel.frame[:, 0], geometry.dof_direction('101X+'))
    assert np.allclose(force.frame[:, 0], [0, 1, 0])


def test_what_counts_as_one_sensor():
    geometry = _geometry()
    rows = (_triax(node=102, model='356a02')
            + [(102, 'Z+', 'acceleration', '352A24', ''),
               (102, 'X+', 'acceleration', '352A24', ''),
               (101, 'Z+', 'acceleration', 'ACME-9', 'Z'),
               (101, 'X+', 'acceleration', 'ACME-9', 'X'),
               (101, 'Z+', 'acceleration', 'BETA-2', ''),
               (101, 'Y+', 'acceleration', '', ''),
               (102, 'X+', 'force', '', ''),
               (102, 'Z+', 'voltage', 'V-1', '')])
    found, notes = placing.place(_table(rows), geometry)
    kinds = [(s.node, s.kind, s.rows) for s in found]
    assert kinds == [(102, 'triax', (0, 1, 2)), (102, 'uniaxial', (3,)),
                     (102, 'uniaxial', (4,)), (101, 'triax', (5, 6)),
                     (101, 'uniaxial', (7,))], \
        'a case-blind model; two uniaxials apart; a triax of an unknown ' \
        'model by its Triax DOF; nothing for a row with no model, nor for ' \
        'a voltage'
    assert found[3].entry is GENERIC['triax'] and found[3].generic
    assert found[4].entry is GENERIC['uniaxial']
    assert notes == [('not in the catalog, drawn as a generic triaxial '
                     'accelerometer: ACME-9'),
                     ('not in the catalog, drawn as a generic uniaxial '
                     'accelerometer: BETA-2')]


def test_rows_that_cannot_be_placed_are_counted():
    geometry = _geometry()
    rows = [('', 'Z+', 'acceleration', '352A24', ''),
            (999, 'Z+', 'acceleration', '352A24', ''),
            (101, '', 'force', '208C01', ''),
            (103, 'Z+', 'acceleration', '352A24', ''),
            (101, 'Z+', 'force', '352A24', '')]
    found, notes = placing.place(_table(rows), geometry)
    assert [s.entry for s in found] == [GENERIC['force']], \
        'an accelerometer model on a force row draws a generic force gauge'
    assert notes == [('352A24 is a uniaxial accelerometer, on a force row at '
                     'node 101: drawn as a generic force transducer'),
                     '1 sensor row not drawn: no node',
                     '1 sensor row not drawn: node not in the geometry',
                     '1 sensor row not drawn: no direction',
                     ('1 sensor row not drawn: node in a cylindrical or '
                     'spherical system')]


def test_the_catalog():
    assert lookup(' 356 a02 ') is lookup('356A02')
    assert lookup('nothing') is None
    assert {m.model for m in models_for('force')} == {
        f'208C0{k}' for k in range(1, 6)}
    assert 'force' not in {m.kind for m in models_for('acceleration')}
    assert models_for('voltage') == []


def test_the_project_places_a_linked_tables_sensors():
    project = visualdynamics.Project()
    project.add('Plate', _geometry())
    project.add('Channels', _table(_triax()))
    project.link('Plate', 'Channels')
    (sensor,), notes = project.sensors('Channels')
    assert sensor.node == 101 and notes == []
    with pytest.raises(TypeError, match='not a channel table'):
        project.sensors('Plate')


# ---- the bodies --------------------------------------------------------------

def _colors(mesh):
    return {tuple(int(v) for v in rgb) for rgb in mesh.mesh.cell_data['colors']}


def _rgb(color):
    return tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))


def test_the_mesh_is_true_size_in_display_units_about_the_node():
    geometry = _geometry()
    found, _notes = placing.place(_table(_triax()), geometry)
    drawn = drawing.sensor_mesh(found, geometry, SI)
    mm = drawing.sensor_mesh(found, geometry, visualdynamics.units.MMKS)
    assert np.allclose(mm.base, drawn.base * 1000.0)
    sized = drawing.sensor_mesh(found, geometry, SI, size=10.0)
    assert np.allclose(sized.base - HERE, (drawn.base - HERE) * 10.0), \
        'the size factor scales about the node, which stays the center'
    assert np.array_equal(drawn.node_rows, np.zeros(len(drawn.base)))
    assert drawn.notes == []
    assert _colors(drawn) == {_rgb(c) for c in AXIS_COLORS} | {
        _rgb(drawing.BODY_COLOR), _rgb(drawing.CONNECTOR_COLOR)}


def test_each_positive_face_is_on_its_legs_positive_side():
    geometry = _geometry()
    for sign in ('+', '-'):
        rows = [(102, f'Z{sign}', 'acceleration', '352A24', '')]
        found, _notes = placing.place(_table(rows), geometry)
        drawn = drawing.sensor_mesh(found, geometry, SI)
        faces = drawn.mesh.faces.reshape(-1, 4)[:, 1:]
        blue = np.all(drawn.mesh.cell_data['colors']
                      == _rgb(AXIS_COLORS[2]), axis=1)
        heights = drawn.base[faces[blue]][..., 2] - 0.0
        assert np.all(np.sign(heights) == (1 if sign == '+' else -1)), \
            'the blue face is on the side the row says is positive'


def test_with_no_units_the_bodies_scale_with_the_model():
    geometry = _geometry(units=False)
    found, _notes = placing.place(_table(_triax(node=102, model='ACME-9')),
                                  geometry)
    drawn = drawing.sensor_mesh(found, geometry, SI)
    assert drawn.notes == [drawing.NOMINAL_NOTE]
    extent = np.linalg.norm(np.ptp(geometry.node_xyz, axis=0))
    body = drawn.base[:8]                      # the box's corners
    assert np.ptp(body[:, 0]) == pytest.approx(
        10.0 * drawing.NOMINAL_MM * extent)


def test_no_sensors_no_mesh():
    geometry = _geometry()
    assert drawing.sensor_mesh([], geometry, SI) is None


# ---- moving with the node ----------------------------------------------------

def test_an_animation_carries_each_body_with_its_node():
    import pyvista as pv

    from visualdynamics.deform import ShapeDeflection
    from visualdynamics.viz.animate import GeometryAnimator

    geometry = _geometry()
    deflection = ShapeDeflection(geometry, ['101Z+', '102X+'],
                                 np.array([1.0, 0.5]))
    plotter = pv.Plotter(off_screen=True)
    try:
        animator = GeometryAnimator(plotter, geometry, deflection,
                                    unit_system=SI)
        found, _notes = placing.place(_table(
            _triax() + [(102, 'Z+', 'acceleration', '352A24', '')]), geometry)
        drawn = drawing.sensor_mesh(found, geometry, SI)
        animator.follow(drawn)
        animator.set_parameter(np.pi / 3)
        moved = animator.view[drawn.node_rows] - animator.base[drawn.node_rows]
        assert np.abs(moved).max() > 0, 'the nodes did move'
        assert np.allclose(drawn.view, drawn.base + moved, rtol=0,
                           atol=1e-12), 'each point by its own node'
        held = drawn.view.ctypes.data
        animator.set_parameter(np.pi / 2)
        assert drawn.view.ctypes.data == held, 'written in place'
        animator.reset()
        assert np.allclose(drawn.view, drawn.base, rtol=0, atol=1e-12)
    finally:
        plotter.close()


# ---- the window --------------------------------------------------------------

def _linked(window, pump, rows=None, shapes=False):
    """A plate of two nodes, a channel table, and with `shapes` a shape
    set, all linked; the window holding them."""
    from visualdynamics.core.shapes import ShapeSet

    geometry = visualdynamics.Geometry(node_id=[1, 2],
                                       node_xyz=[[0, 0, 0], [0.5, 0, 0]])
    geometry.define_units('m')
    table = _table(rows or (_triax(node=1)
                            + [(2, 'Z+', 'acceleration', '352A24', '')]))
    window.add_object('Plate', geometry)
    window.add_object('Channels', table)
    names = ['Plate', 'Channels']
    if shapes:
        dofs = ['1X+', '1Y+', '1Z+', '2X+', '2Y+', '2Z+']
        window.add_object('Modes', ShapeSet(
            [10.0], [0.01], dofs, np.array([[0, 0, 1.0, 0, 0, -1.0]]),
            mass_unit='kg'))
        names.append('Modes')
    window.project.link(*names)
    pump()
    return geometry, table


def _sensor_points(window, index=0):
    actor = window.scene.plotter.actors.get(f'sensors-{index}')
    return None if actor is None else np.asarray(actor.mapper.dataset.points)


def test_the_toggle_is_offered_only_where_a_sensor_is_placed(window, pump):
    from conftest import select_objects

    _linked(window, pump)
    select_objects(window, pump, 'Plate', 'Channels')
    assert window.sensors_action.isVisible()
    assert window.sensor_size_action.isVisible()
    assert _sensor_points(window) is not None
    select_objects(window, pump, 'Channels')
    assert not window.sensors_action.isVisible(), 'no geometry drawn'
    window.add_object('Bare', visualdynamics.Geometry(
        node_id=[7], node_xyz=[[0, 0, 0]]))
    pump()
    select_objects(window, pump, 'Bare')
    assert not window.sensors_action.isVisible(), 'no table linked with it'
    window.add_object('Volts', _table([(7, 'Z+', 'voltage', '', '')]))
    window.project.link('Bare', 'Volts')
    pump()
    select_objects(window, pump, 'Bare')
    assert not window.sensors_action.isVisible(), 'a table with no sensor'
    select_objects(window, pump, 'Plate')
    assert window.sensors_action.isVisible(), 'the linked table is read'
    window.sensors_action.setChecked(False)
    pump()
    assert _sensor_points(window) is None
    assert not window.sensor_size_action.isVisible()


def test_the_size_is_a_multiple_of_true_size_about_each_node(window, pump):
    from conftest import select_objects

    _linked(window, pump)
    select_objects(window, pump, 'Plate')
    true = _sensor_points(window)
    window.sensor_size_combo.setCurrentIndex(
        window.sensor_size_combo.findData(10.0))
    pump()
    big = _sensor_points(window)
    rows = np.asarray(window.objects['Plate'].node_xyz)
    us = window.unit_system
    node = np.asarray(us.from_si(rows[0], 'length'), dtype=float)
    first = slice(0, len(true) // 2)            # the triax at node 1
    assert np.allclose(big[first] - node, (true[first] - node) * 10.0)
    span = np.ptp(true[:8, 0])
    assert span == pytest.approx(float(us.from_si(0.0203, 'length')))


def test_a_cell_typed_in_the_table_redraws_them(window, pump):
    from conftest import select_objects

    _geometry, table = _linked(window, pump)
    select_objects(window, pump, 'Plate', 'Channels')
    before = _sensor_points(window).copy()
    table.set_cell('direction', 3, 'Z-')
    # the table view's model reports an edit through this, as a typed
    # cell does
    edited = type('Model', (), {'obj': table})()
    window._journal_table_edit(edited, ".set_cell('direction', 3, 'Z-')")
    pump()
    after = _sensor_points(window)
    half = len(before) // 2
    assert np.allclose(before[:half], after[:half]), 'the triax stays'
    assert np.allclose(after[half:, 2], -before[half:, 2]), \
        'the uniaxial at a node on z = 0 is drawn upside down'


def test_a_mode_shape_carries_them_with_their_nodes(window, pump):
    from conftest import select_objects

    _linked(window, pump, shapes=True)
    select_objects(window, pump, 'Plate', 'Modes')
    animator = window.animator
    assert animator is not None
    animator.reset()
    rest = _sensor_points(window).copy()
    animator.set_parameter(np.pi / 3)
    moved = _sensor_points(window)
    offsets = animator.view - animator.base
    assert np.abs(offsets).max() > 0
    half = len(rest) // 2
    assert np.allclose(moved[:half] - rest[:half], offsets[0], atol=1e-12)
    assert np.allclose(moved[half:] - rest[half:], offsets[1], atol=1e-12)


def test_each_table_is_drawn_once_on_its_own_geometry(window, pump):
    """An instrumentation plan keeps the model's node numbers, each node
    measured in a turned frame of its own. With the plan's geometry and
    the model selected together, the plan's sensors were drawn on both,
    the same sensors twice and turned two ways (Brandon, 2026-10-27)."""
    from conftest import select_objects

    turned = _geometry()                       # node 101 in a turned frame
    model = visualdynamics.Geometry(node_id=[101, 102, 103],
                                    node_xyz=turned.node_xyz.copy())
    model.define_units('m')
    window.add_object('Plan', turned)
    window.add_object('Plan Channels', _table(_triax()))
    window.add_object('Model', model)
    window.project.link('Plan', 'Plan Channels')
    pump()

    def drawn():
        return sorted(str(name) for name in window.scene.plotter.actors
                      if str(name).startswith('sensors-'))

    expected = drawing.sensor_mesh(
        placing.place(window.objects['Plan Channels'], turned)[0], turned,
        window.unit_system).base
    for picked in (('Plan', 'Model'), ('Model', 'Plan'),
                   ('Plan Channels', 'Model', 'Plan')):
        select_objects(window, pump, *picked)
        (name,) = drawn()
        points = np.asarray(
            window.scene.plotter.actors[name].mapper.dataset.points)
        assert np.allclose(points, expected), \
            f'{picked}: once, turned by the plan\'s own frame'
    window.add_object('Model Channels', _table(
        [(102, 'Z+', 'acceleration', '352A24', '')]))
    window.project.link('Model', 'Model Channels')
    pump()
    select_objects(window, pump, 'Plan', 'Model')
    assert len(drawn()) == 2, 'each geometry its own table'


# ---- the Make and Model cells ------------------------------------------------

def test_a_catalog_model_fills_a_blank_make():
    table = _table([(101, 'Z+', 'acceleration', '', ''),
                    (102, 'Z+', 'acceleration', '', ''),
                    (103, 'Z+', 'acceleration', '', '')])
    table.set_cell('model', 0, '356A02')
    assert table['make'][0] == 'PCB'
    table.set_cell('make', 1, 'Lab stock')
    table.set_cell('model', 1, '352A24')
    assert table['make'][1] == 'Lab stock', 'a make already written is kept'
    table.set_cell('model', 2, 'ACME-9')
    assert table['make'][2] == '', 'nothing to fill from'


def test_the_cells_offer_the_catalog_for_what_the_row_measures(qt_app):
    from PySide6.QtCore import Qt

    from visualdynamics.gui.object_tables import channel_table_model

    table = _table([(101, 'Z+', 'acceleration', '', ''),
                    (102, 'X+', 'force', '', ''),
                    (103, 'Z+', 'voltage', '', '')])
    model = channel_table_model(table)
    titles = [column.title for column in model.columns]
    make, chosen = (model.columns[titles.index(t)] for t in ('Make', 'Model'))
    assert chosen.choices_editable and make.choices_editable, \
        'offered, not imposed'
    accelerometers = chosen.choices_for(table, 0)
    assert '356A02' in accelerometers and '208C01' not in accelerometers
    assert chosen.choices_for(table, 1) == [
        f'208C0{k}' for k in range(1, 6)]
    assert not chosen.choices_for(table, 2), 'no list for a voltage'
    assert make.choices_for(table, 0) == ['PCB']
    cell = model.index(0, titles.index('Model'))
    assert model.setData(cell, '356A03', Qt.ItemDataRole.EditRole)
    assert table['make'][0] == 'PCB'
    assert model.setData(model.index(2, titles.index('Model')), 'DAQ-7',
                         Qt.ItemDataRole.EditRole), 'any text is accepted'
    assert table['model'][2] == 'DAQ-7'


def test_a_triaxs_identity_is_written_once_for_its_legs():
    """The other acceleration rows at the node get the triax's make,
    model and serial number where theirs are blank; a force gauge there
    is never filled, nor a leg already holding another sensor's model."""
    table = _table([(101, 'X+', 'acceleration', '', 'X'),
                    (101, 'Y+', 'acceleration', '', 'Y'),
                    (101, 'Z+', 'acceleration', '', 'Z'),
                    (101, 'Z+', 'force', '', ''),
                    (101, 'Z-', 'acceleration', '352A24', ''),
                    (102, 'Z+', 'acceleration', '', '')])
    table.set_cell('model', 1, '356A02')
    assert list(table['model']) == ['356A02', '356A02', '356A02', '',
                                    '352A24', '']
    assert list(table['make']) == ['PCB', 'PCB', 'PCB', '', '', '']
    table.set_cell('serial_number', 2, 'LW1234')
    assert list(table['serial_number']) == ['LW1234'] * 3 + ['', '', '']
    table.set_cell('serial_number', 0, 'LW9999')
    assert list(table['serial_number'][:3]) == ['LW9999', 'LW1234',
                                                'LW1234'], \
        'a serial already written is kept'
    table.set_cell('model', 3, '208C02')
    assert table['model'][0] == '356A02', 'a force model fills no leg'


def test_one_or_two_legs_wired_fill_only_what_is_there():
    table = _table([(101, 'X+', 'acceleration', '', 'X'),
                    (101, 'Z+', 'acceleration', '', 'Z')])
    table.set_cell('model', 0, '356A43')
    assert list(table['model']) == ['356A43', '356A43']
    assert table.num_channels == 2, 'no leg is added'
    uniaxial = _table([(101, 'X+', 'acceleration', '', ''),
                       (101, 'Y+', 'acceleration', '', '')])
    uniaxial.set_cell('model', 0, '352A24')
    assert list(uniaxial['model']) == ['352A24', ''], \
        'only a triax is spread'


def test_the_table_view_shows_the_other_legs_filled(qt_app):
    from PySide6.QtCore import Qt

    from visualdynamics.gui.object_tables import channel_table_model

    table = _table(_triax(node=101, model=''))
    model = channel_table_model(table)
    titles = [column.title for column in model.columns]
    seen = []
    model.dataChanged.connect(lambda top, bottom: seen.append(
        (top.row(), bottom.row())))
    assert model.setData(model.index(0, titles.index('Model')), '356A02',
                         Qt.ItemDataRole.EditRole)
    assert seen == [(0, 2)], 'every row redrawn, so the legs show'
    assert [model.index(r, titles.index('Model')).data() for r in range(3)] \
        == ['356A02'] * 3
