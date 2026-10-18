"""The weight on the status line: a geometry whose element groups carry
a material, or the element groups picked in the tree, weighed as the
solve weighs them, in lbf for an imperial display system and kg for SI.

Two quads of 1 m square, aluminum at 2700 kg/m^3: the wing 10 mm thick
(27 kg), the tail 20 mm (54 kg).
"""

from __future__ import annotations

import pytest
from conftest import group_row
from test_element_blocks import _show_geometry
from test_fem import ALUMINUM

from visualdynamics.core.fem import GroupProperties
from visualdynamics.core.geometry import Geometry
from visualdynamics.units import IN_LBF_S, SI

#: one kilogram's weight in pounds-force, 9.80665 / 4.4482216152605
LBF_PER_KG = 2.2046226218


@pytest.fixture
def panel():
    return Geometry(
        node_id=[1, 2, 3, 4, 5, 6],
        node_xyz=[[0, 0, 0], [1, 0, 0], [2, 0, 0],
                  [0, 1, 0], [1, 1, 0], [2, 1, 0]],
        elem_id=[10, 11],
        elem_conn=[[1, 2, 5, 4], [2, 3, 6, 5]],
        elem_type=[44, 44],
        elem_group=[7, 9],
        group_id=[7, 9],
        group_name=['wing', 'tail'],
        length_unit='m',
        group_properties={7: GroupProperties(ALUMINUM, 0.01),
                          9: GroupProperties(ALUMINUM, 0.02)})


def _status(window):
    return window.statusBar().currentMessage()


def test_a_geometry_with_materials_says_its_weight_in_kg_for_si(
        window, pump, panel):
    window.unit_system = SI
    _show_geometry(window, pump, panel)
    assert _status(window).endswith(' — weight 81 kg'), _status(window)


def test_imperial_says_it_in_pounds_force(window, pump, panel):
    """The imperial system's mass unit is the slinch, which nobody
    reads as weight; lbf is how weight is said there."""
    window.unit_system = IN_LBF_S
    _show_geometry(window, pump, panel)
    assert _status(window).endswith(
        f' — weight {81 * LBF_PER_KG:.4g} lbf'), _status(window)


def test_picked_element_groups_say_their_own_weight_summed(
        window, pump, panel):
    window.unit_system = SI
    _show_geometry(window, pump, panel)
    window.tree.setCurrentItem(group_row(window, 9))
    pump()
    assert _status(window).endswith(' — weight 54 kg'), _status(window)
    group_row(window, 7).setSelected(True)
    pump()
    assert _status(window).endswith(' — weight 81 kg'), _status(window)


def test_a_group_without_a_material_is_said_not_counted(window, pump, panel):
    window.unit_system = SI
    panel.group_properties = {7: GroupProperties(ALUMINUM, 0.01)}
    _show_geometry(window, pump, panel)
    assert _status(window).endswith(
        ' — weight 27 kg (1 of 2 element groups have a material)'), \
        _status(window)


def test_no_material_anywhere_says_no_weight(window, pump, panel):
    panel.group_properties = {}
    _show_geometry(window, pump, panel)
    assert 'weight' not in _status(window)


def test_an_edit_to_a_material_is_weighed_again(window, pump, panel):
    """The masses are kept between clicks, and a change to anything
    they depend on is weighed afresh."""
    window.unit_system = SI
    _show_geometry(window, pump, panel)
    assert _status(window).endswith(' — weight 81 kg')
    panel.group_properties = {7: GroupProperties(ALUMINUM, 0.03),
                              9: GroupProperties(ALUMINUM, 0.02)}
    window.render_current()
    pump()
    assert _status(window).endswith(' — weight 135 kg'), _status(window)
