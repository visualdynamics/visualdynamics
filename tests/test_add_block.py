"""Add Block: a meshed box of bricks typed into a geometry, the project
verb and the app's dialog (2026-09-30, for the four-unit frame example).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import select_objects

import visualdynamics
from visualdynamics.core import mesh
from visualdynamics.core.fem import BlockProperties, Model, material

INCH = 0.0254


def test_a_block_is_a_box_of_bricks():
    bar = mesh.block((0, 0, 0), (0.4, 0, 0), (0, 0.1, 0), (0, 0, 0.02), 0.02,
                     'bar')
    assert bar.num_nodes == 21 * 6 * 2 and len(bar.elem_conn) == 20 * 5
    assert set(bar.elem_type.tolist()) == {115}
    assert list(bar.block_name) == ['bar'] and bar.length_unit == 'm'
    with pytest.raises(ValueError, match='not perpendicular'):
        mesh.block((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 1), 0.5, 'skew')
    with pytest.raises(ValueError, match='no length'):
        mesh.block((0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 0, 1), 0.5, 'flat')


def test_a_block_weighs_and_bends_as_its_solids():
    bar = mesh.block((0, 0, 0), (0.4, 0, 0), (0, 0.02, 0), (0, 0, 0.02),
                     0.02, 'bar')
    bar.block_properties = {1: BlockProperties(material('6061-T6'))}
    model = Model.from_geometry(bar)
    assert model.structural_mass == pytest.approx(
        0.4 * 0.02 * 0.02 * material('6061-T6').density)
    shapes = model.eigensolution(num_modes=8)
    assert int(np.sum(shapes.frequency == 0.0)) == 6
    assert shapes.frequency[6] == pytest.approx(shapes.frequency[7], rel=1e-6), \
        'a square bar bends the same either way'


def test_holes_leave_the_block_or_go_to_an_insert_block():
    box = mesh.block((0, 0, 0), (0.4, 0, 0), (0, 0.1, 0), (0, 0, 0.02), 0.01,
                     'bar', holes=[((0.1, 0.05, 0), 0.02, 2)])
    whole = 40 * 10 * 2
    assert len(box.elem_conn) < whole and list(box.block_name) == ['bar']
    assert box.num_nodes < 41 * 11 * 3, 'the hole\'s own nodes went too'
    centers = np.array([box.node_xyz[box.node_index(c)].mean(axis=0)
                        for c in box.elem_conn])
    assert np.linalg.norm(centers[:, :2] - [0.1, 0.05], axis=1).min() > 0.02
    inserts = mesh.block((0, 0, 0), (0.4, 0, 0), (0, 0.1, 0), (0, 0, 0.02),
                         0.01, 'bar', holes=[((0.1, 0.05, 0), 0.02, 2),
                                             ((0.3, 0.05, 0.02), 0.02, 2, 0.012)],
                         hole_name='inserts')
    assert list(inserts.block_name) == ['bar', 'inserts']
    assert len(inserts.elem_conn) == whole, 'nothing removed, only moved'
    insert_rows = np.flatnonzero(inserts.elem_block == 2)
    depths = centers_z = np.array([inserts.node_xyz[inserts.node_index(
        inserts.elem_conn[r])].mean(axis=0) for r in insert_rows])
    through = depths[np.abs(centers_z[:, 0] - 0.1) < 0.021]
    blind = depths[np.abs(centers_z[:, 0] - 0.3) < 0.021]
    assert through[:, 2].min() < 0.01 < through[:, 2].max()
    assert blind[:, 2].min() > 0.02 - 0.012, 'blind from the far face'
    with pytest.raises(ValueError, match='leave nothing'):
        mesh.block((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 0.5, 'x',
                   holes=[((0.5, 0.5, 0), 5, 2)])


def test_the_project_verb_is_journaled_and_ties_blocks_that_meet():
    project = visualdynamics.Project('p')
    name = project.new_geometry('Frame', unit='in')
    first = project.add_block(name, (0, 0, 0), (4, 0, 0), (0, 0.5, 0),
                              (0, 0, 0.5), 0.5, 'frame', unit='in')
    second = project.add_block(name, (0, 0.5, 0), (0.5, 0, 0), (0, 2, 0),
                               (0, 0, 0.5), 0.5, 'frame', unit='in')
    assert first['elements'] == 8 and second['elements'] == 4
    assert second['shared'] == 4, 'the upright stands on the rail\'s face'
    frame = project[name]
    assert list(frame.block_name) == ['frame']
    assert np.allclose(frame.node_xyz.max(axis=0), [4 * INCH, 2.5 * INCH,
                                                    0.5 * INCH])
    assert project.journal[-1] == (
        "project.add_block('Frame', (0, 0.5, 0), (0.5, 0, 0), (0, 2, 0), "
        "(0, 0, 0.5), 0.5, 'frame', unit='in')")
    frame.block_properties = {1: BlockProperties(material('6061-T6'))}
    modes = project.solve_modes(name, num_modes=8)
    assert int(np.sum(project[modes].frequency == 0.0)) == 6, 'one piece'


def test_add_block_types_blocks_in_display_units(window, pump):
    """The dialog reads in the display unit, previews the block by its
    skin, says what it shares, and each Add is the project's verb."""
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    window.project.new_geometry('Frame', unit='in')
    window.show_object('Frame')
    select_objects(window, pump, 'Frame')
    assert 'add_block' in [act[0] for act in window.acts_for(['Frame'])]
    window.add_block_act()
    pump()
    dialog = window.block_dialog
    from PySide6.QtWidgets import QLabel

    labels = [label.text() for label in dialog.findChildren(QLabel)]
    assert {'Corner [in]', 'Edge A [in]', 'Edge B [in]', 'Edge C [in]',
            'Element size [in]'} <= set(labels)
    dialog.set_values(block='rail', corner=(0, 0, 0), edge_a=(4, 0, 0),
                      edge_b=(0, 1, 0), edge_c=(0, 0, 0.5), size=0.5)
    pump()
    assert dialog.reading_label.text() == (
        "16 bricks of 0.5 by 0.5 by 0.5 in, into a new block 'rail': "
        '54 nodes to add, 0 on nodes already there.')
    assert 'plane-preview' in window.scene.plotter.actors
    dialog.add_button.click()
    pump()
    frame = window.objects['Frame']
    assert np.allclose(frame.node_xyz.max(axis=0), [4 * INCH, INCH, 0.5 * INCH])
    assert 'added 16 bricks — 54 nodes, 0 shared' in \
        window.statusBar().currentMessage()
    dialog.set_values(edge_a=(4, 0, 1))
    pump()
    assert 'not perpendicular' in dialog.reading_label.text()
    assert not dialog.add_button.isEnabled()
    dialog.close()
    pump()
    assert 'plane-preview' not in window.scene.plotter.actors
