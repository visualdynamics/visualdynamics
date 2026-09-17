"""The geometry tables' own columns: units, colors, ids and references.

`test_table_conventions.py` checks that every table *behaves* the same;
this is what these particular columns mean. Four of them refuse rather
than write — an id that is taken, a coordinate system that does not
exist, a color outside the palette, a frame type that is not one of the
three — and a refusal that quietly wrote would leave a geometry that
cannot be validated.

Coordinates are the other half: they read and write in the *display*
system, so typing 25 into a millimeter view has to arrive as 0.025 m and
come back as 25.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.gui.object_tables import (
    coordinate_system_table_model,
    element_table_model,
    node_table_model,
)
from visualdynamics.units import SYSTEMS


@pytest.fixture
def plate(qt_app):
    return visualdynamics.import_file(fixture_path('plate', 'geometry.exo'),
                            length_unit='m')


def column_of(model, title):
    return [c.title for c in model.columns].index(title)


def test_coordinates_read_and_write_in_the_display_system(plate):
    """The number in the cell is the number on the axis."""
    mmks = SYSTEMS['mm-kg-N-s']
    model = node_table_model(plate, mmks)
    x = column_of(model, next(c.title for c in model.columns
                              if c.title.startswith('X [')))
    meters = float(plate.node_xyz[0, 0])
    assert float(model.data(model.index(0, x))) == pytest.approx(
        meters * 1000.0), 'shown in millimeters'
    assert model.setData(model.index(0, x), '25')
    assert float(plate.node_xyz[0, 0]) == pytest.approx(0.025), (
        'and stored in meters')


def test_an_id_that_is_taken_is_refused(plate):
    model = node_table_model(plate, SYSTEMS['m-kg-N-s'])
    reasons = []
    model.edit_rejected.connect(reasons.append)
    taken = int(plate.node_id[1])
    before = list(plate.node_id)
    assert model.setData(model.index(0, 0), str(taken)) is False
    assert list(plate.node_id) == before, 'refused, not written'
    assert reasons and 'exists' in reasons[0]
    assert model.setData(model.index(0, 0), '0') is False, 'nor a zero id'


def test_a_node_can_only_point_at_a_coordinate_system_that_exists(plate):
    model = node_table_model(plate, SYSTEMS['m-kg-N-s'])
    column = column_of(model, 'Placement CS')
    reasons = []
    model.edit_rejected.connect(reasons.append)
    assert model.setData(model.index(0, column), '99') is False
    assert reasons and 'no coordinate system 99' in reasons[0]
    assert model.setData(model.index(0, column), str(int(plate.cs_id[0])))


def test_a_color_is_a_palette_name_or_its_index(plate):
    model = element_table_model(plate, SYSTEMS['m-kg-N-s'])
    column = column_of(model, 'Color')
    assert model.setData(model.index(0, column), 'red')
    assert model.data(model.index(0, column)) == 'red', (
        'stored as the palette index, read back as the name')
    assert model.setData(model.index(0, column), '3'), 'the index still works'
    reasons = []
    model.edit_rejected.connect(reasons.append)
    assert model.setData(model.index(0, column), 'puce') is False
    assert model.setData(model.index(0, column), '999') is False
    assert all('palette name' in reason for reason in reasons)


def test_a_frame_is_one_of_the_three_kinds_there_are(plate):
    model = coordinate_system_table_model(plate, SYSTEMS['m-kg-N-s'])
    column = column_of(model, 'Type')
    assert model.columns[column].choices == ['cartesian', 'cylindrical',
                                             'spherical']
    assert model.setData(model.index(0, column), 'cylindrical')
    assert int(plate.cs_type[0]) == 1
    reasons = []
    model.edit_rejected.connect(reasons.append)
    assert model.setData(model.index(0, column), 'polar') is False
    assert reasons and 'expected one of' in reasons[0]


def test_a_coordinate_system_origin_moves_in_display_units(plate):
    mmks = SYSTEMS['mm-kg-N-s']
    model = coordinate_system_table_model(plate, mmks)
    column = column_of(model, next(c.title for c in model.columns
                                   if c.title.startswith('Origin X [')))
    assert model.setData(model.index(0, column), '50')
    assert float(plate.cs_matrix[0, 3, 0]) == pytest.approx(0.05)
    assert float(model.data(model.index(0, column))) == pytest.approx(50.0)


def test_switching_the_theme_repaints_what_is_on_screen(window, pump):
    """Live, with a model up: the plot's foreground, the scene's
    background and the panes all follow, and nothing has to be reopened."""
    from visualdynamics.theme import theme as resolve_theme

    window.import_paths([fixture_path('plate', 'geometry.exo')])
    pump()
    for name in ('light', 'dark'):
        window.apply_theme(name)
        pump()
        assert window.theme_name == name
        colors = resolve_theme(name)
        background = window.scene.plotter.renderer.GetBackground()
        expected = tuple(int(colors['scene_background'].lstrip('#')[i:i + 2],
                             16) / 255 for i in (0, 2, 4))
        assert np.allclose(background, expected, atol=0.02), (
            f'{name}: the 3D view kept the other theme')
