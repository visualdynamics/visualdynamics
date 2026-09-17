"""The rigid-body reading: six shapes previewed where they are set.

The toggle on the 3-D view's bar is offered to one whole geometry and
nothing else; its pane seeds the centroid in the display units and
hands every edit to the geometry as a `MassProperties` (which is what
the journal records and the generated set's badge fingerprints); the
scene animates the proposed set through the same animator a selected
set uses, with the reference point marked; and the button makes the
set through the project verb, linked into the geometry's group.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import select_objects as _select

import visualdynamics
from visualdynamics.core.rigid import MassProperties
from visualdynamics.viz.rigid import NAME as MARKER


def _geometry(unit='m'):
    rng = np.random.default_rng(3)
    return visualdynamics.Geometry(np.arange(101, 108),
                                   rng.uniform(-1.0, 1.0, (7, 3)),
                                   length_unit=unit)


def _plate(window, pump, unit='m'):
    geometry = _geometry(unit)
    window.add_object('Plate', geometry)
    _select(window, pump, 'Plate')
    return geometry


def _toggle(window, pump):
    scene = window.scene
    scene.rigid_action.setChecked(True)
    scene._choose_rigid(True)
    pump()
    return scene


def _marker_up(window):
    return MARKER in window.scene.plotter.actors


# ---- offered to one whole geometry ----------------------------------------


def test_the_toggle_is_offered_to_a_lone_geometry_only(window, pump):
    _plate(window, pump)
    scene = window.scene
    assert scene.rigid_action.isVisible()
    assert not scene.rigid_action.isChecked(), 'off until asked for'
    assert not scene.rigid_panel.isVisible()

    t = np.arange(2048) / 2048.0
    window.add_object('Record', visualdynamics.TimeHistory(
        t, np.zeros((1, len(t))), response_dof=['101Z+'],
        ordinate_dim=['acceleration']))
    _select(window, pump, 'Plate', 'Record')
    assert not scene.rigid_action.isVisible(), \
        'data riding the geometry is another reading of the scene'
    _select(window, pump, 'Record')
    assert not scene.rigid_action.isVisible()
    window.tree.clearSelection()
    pump()
    assert not scene.rigid_action.isVisible(), 'nothing selected, nothing offered'


def test_the_wish_survives_the_selection(window, pump):
    _plate(window, pump)
    scene = _toggle(window, pump)
    window.add_object('Other', _geometry())
    _select(window, pump, 'Other')
    assert scene.rigid_action.isChecked(), \
        'the reading comes back up on the next geometry, the data pane\'s rule'
    assert window._rigid_preview[0] == 'Other'
    _select(window, pump, 'Plate', 'Other')
    assert not scene.rigid_action.isVisible()
    assert not scene.rigid_panel.isVisible()
    assert window._rigid_preview is None


# ---- the pane and the preview ---------------------------------------------


def test_the_pane_seeds_the_centroid_in_display_units(window, pump):
    geometry = _plate(window, pump)
    scene = _toggle(window, pump)
    panel = scene.rigid_panel
    assert panel.isVisible()
    centroid = window.unit_system.from_si(geometry.node_xyz.mean(axis=0),
                                          'length')
    for box, value in zip(panel.point_boxes, centroid):
        assert box.value() == pytest.approx(value, abs=1e-4)
    assert panel.point_boxes[0].suffix().strip() == \
        window.unit_system.label_text('length')
    assert panel.derived['modes'].text() == '6'
    assert panel.derived['dofs'].text() == '3 × 7'
    assert panel.derived['axes'].text() == 'X, Y, Z'
    assert panel.derived['scaling'].text() == 'unit shapes'
    assert not panel.scale_panel.isVisible()
    assert geometry.mass_properties is None, \
        'looking is not setting: nothing is stored until an edit or the act'


def test_the_scene_previews_the_six_modes_with_the_point_marked(window, pump):
    _plate(window, pump)
    scene = _toggle(window, pump)
    assert window.animator is not None
    assert window._shape_source[1].num_shapes == 6
    assert window.mode_box.value() == 1 and window.mode_box.maximum() == 6
    assert window.play_action.isEnabled()
    assert _marker_up(window)
    # mode 1 is the X translation: every node moves the same way
    offsets = window.animator.deflection.offsets(0.0)
    assert np.allclose(offsets, offsets[0])
    assert offsets[0, 0] > 0 and np.allclose(offsets[0, 1:], 0.0)
    assert '(preview)' in window.statusBar().currentMessage() or \
        'Generate Rigid Body Mode Shapes' in window.statusBar().currentMessage()

    window.mode_box.setValue(4)
    pump()
    assert _marker_up(window), 'stepping modes clears the scene; the marker returns'
    turned = window.animator.deflection.offsets(0.0)
    assert not np.allclose(turned, turned[0]), 'a rotation moves nodes differently'
    assert scene.rigid_action.isChecked()


def test_an_edit_lands_on_the_geometry_and_moves_the_preview(window, pump):
    geometry = _plate(window, pump)
    scene = _toggle(window, pump)
    window.mode_box.setValue(6)                 # rotation about Z
    pump()
    before = window.animator.deflection.offsets(0.0).copy()
    scene.rigid_panel.point_boxes[0].setValue(20.0)
    pump()
    stored = geometry.mass_properties
    assert stored is not None
    assert stored.point[0] == pytest.approx(
        float(window.unit_system.to_si(20.0, 'length')))
    assert window.project.journal[-1].startswith(
        "project['Plate'].mass_properties = MassProperties(")
    after = window.animator.deflection.offsets(0.0)
    assert not np.allclose(after, before), \
        'a rotation about a moved point moves the nodes differently'
    assert window.mode_box.value() == 6, 'the mode was kept'
    assert _marker_up(window)


def test_mass_normalizing_reveals_the_terms_and_refuses_a_bad_tensor(
        window, pump):
    geometry = _plate(window, pump)
    scene = _toggle(window, pump)
    panel = scene.rigid_panel
    assert panel.scale_check.text() == 'Mass properties', \
        'named for what it reveals, not for the result'
    panel.scale_check.setChecked(True)
    pump()
    assert panel.scale_panel.isVisible()
    assert geometry.mass_properties.scaled
    assert geometry.mass_properties.mass == pytest.approx(
        float(window.unit_system.to_si(1.0, 'mass')))
    assert panel.derived['scaling'].text() == 'mass-normalized'
    assert panel.mass_box.suffix().strip() == \
        window.unit_system.label_text('mass')
    assert 'slinch' in panel.inertia_boxes['Ixx'].suffix()
    assert window._shape_source[1].mass_unit == 'kg'

    panel.inertia_boxes['Ixy'].setValue(5.0)
    pump()
    assert 'positive definite' in panel.problem.text()
    assert not panel.apply_button.isEnabled()
    assert geometry.mass_properties.inertia[3] == 0.0, \
        'a tensor no body could have is never stored'
    panel.inertia_boxes['Ixy'].setValue(0.0)
    pump()
    assert panel.problem.text() == ''
    assert panel.apply_button.isEnabled()


def test_undeclared_units_cannot_mass_normalize(window, pump):
    _plate(window, pump, unit=None)
    scene = _toggle(window, pump)
    panel = scene.rigid_panel
    assert not panel.scale_check.isVisible()
    assert panel.scale_note.isVisible()
    assert panel.point_boxes[0].suffix() == ''
    assert window.animator is not None, 'unit shapes need no unit'
    assert 'units undefined' in window.statusBar().currentMessage()


def test_a_unit_system_change_restates_the_pane(window, pump):
    geometry = _plate(window, pump)
    scene = _toggle(window, pump)
    panel = scene.rigid_panel
    window.unit_combo.setCurrentText('m-kg-N-s')
    pump()
    assert panel.point_boxes[0].suffix().strip() == 'm'
    assert panel.point_boxes[0].value() == pytest.approx(
        geometry.node_xyz.mean(axis=0)[0], abs=1e-4)


# ---- the act -------------------------------------------------------------


def test_generate_makes_the_set_in_the_group(window, pump):
    geometry = _plate(window, pump)
    scene = _toggle(window, pump)
    scene.rigid_panel.point_boxes[2].setValue(3.0)
    pump()
    scene.rigid_panel.apply_button.click()
    pump()
    assert 'Plate Rigid Body Modes' in window.project
    assert window.project.links == [
        {'members': ['Plate', 'Plate Rigid Body Modes'], 'role': None}]
    made = window.project['Plate Rigid Body Modes']
    assert made.num_shapes == 6 and made.unscaled
    assert window.project.provenance['Plate Rigid Body Modes']['source'] \
        == 'Plate'
    assert window.project.journal[-1] == \
        "project.generate_rigid_body_modes('Plate')"
    status = window.statusBar().currentMessage()
    assert status.startswith('Plate Rigid Body Modes: 6 modes about (')
    assert 'linked to Plate' in status
    # the result is shown, as every act shows what it made; the
    # reading stands down with it and comes back on the geometry
    assert window.current_object() is made
    assert not scene.rigid_action.isVisible()
    _select(window, pump, 'Plate')
    assert scene.rigid_action.isChecked()
    assert geometry.mass_properties.point[2] == pytest.approx(
        float(window.unit_system.to_si(3.0, 'length')))
    # and the badge: moving the point after the act marks the set
    scene.rigid_panel.point_boxes[2].setValue(4.0)
    pump()
    assert 'Plate Rigid Body Modes' in window.project.stale()


def test_the_act_refuses_without_a_geometry(window, pump):
    t = np.arange(2048) / 2048.0
    window.add_object('Record', visualdynamics.TimeHistory(
        t, np.zeros((1, len(t))), response_dof=['101Z+'],
        ordinate_dim=['acceleration']))
    _select(window, pump, 'Record')
    window.generate_rigid_body_modes()
    assert 'Select a geometry' in window.statusBar().currentMessage()
    assert 'Record Rigid Body Modes' not in window.project


def test_the_icon_is_drawn(qt_app):
    from visualdynamics.gui.icons import control_icon

    assert not control_icon('rigid').isNull()
    ours = control_icon('rigid').pixmap(32).toImage()
    fallback = control_icon('no-such-glyph').pixmap(32).toImage()
    assert ours != fallback, 'the glyph is its own, not the default dot'


def test_the_properties_object_is_what_the_pane_hands_over(window, pump):
    _plate(window, pump)
    scene = _toggle(window, pump)
    handed = []
    scene.rigid_panel.changed.connect(handed.append)
    scene.rigid_panel.point_boxes[1].setValue(-2.0)
    pump()
    assert len(handed) == 1 and isinstance(handed[0], MassProperties)
