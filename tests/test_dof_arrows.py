"""DOF arrows: labeled direction markers on the geometry.

Force DOFs come from wherever the project says force was applied —
including an FRF's reference DOFs — and acceleration DOFs from every
measured response. Arrows scale to the geometry and carry their DOF
name, in the GUI's 3D scene and in report scenes alike.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics.core.report import (
    Report,
    dof_quantities,
    insert_options,
    modal_template,
    quantity_dofs,
    series_dof_quantities,
    series_quantity_dofs,
)
from visualdynamics.report import render_html


@pytest.fixture
def project(survey):
    shapes, frfs = survey
    frfs.define_units('m/s^2', reference_units='N')
    return {'Geometry': visualdynamics.import_file(
                fixture_path('plate', 'geometry.npz')),
            'Shape Set': shapes, 'FRF': frfs}


def test_quantities_and_dofs_come_from_the_whole_project(project):
    assert dof_quantities(project) == ['acceleration', 'force']
    frf = project['FRF']
    assert quantity_dofs(project, 'force') == sorted(
        set(frf.reference_dof)), 'an FRF per force names its force DOFs'
    accelerations = quantity_dofs(project, 'acceleration')
    assert set(accelerations) == set(frf.response_dof)
    assert quantity_dofs(project, 'voltage') == []


def test_a_record_pick_narrows_the_dofs(project):
    """The selection is the source of truth: picked grid records give
    arrows for exactly those records' DOFs."""
    frf = project['FRF']
    series = [('FRF', frf, [0])]
    assert series_quantity_dofs(series, 'acceleration') == [
        frf.response_dof[0]]
    assert series_quantity_dofs(series, 'force') == [frf.reference_dof[0]]
    assert series_dof_quantities(series) == ['acceleration', 'force']


def test_arrows_land_on_the_scene(project):
    import pyvista as pv

    from visualdynamics.viz.geometry import add_dof_arrows

    plotter = pv.Plotter(off_screen=True)
    drawn = add_dof_arrows(plotter, project['Geometry'],
                           quantity_dofs(project, 'force')
                           + ['9999Z+'])       # a node the geometry lacks
    assert drawn == len(quantity_dofs(project, 'force')), (
        'one arrow per force DOF; the unknown node is skipped')
    plotter.close()


def test_a_dofs_scene_carries_labeled_arrows(project):
    report = Report('R', [
        {'kind': 'scene', 'geometry': 'Geometry', 'shapes': '',
         'dofs': 'force', 'dofs_source': 'FRF',
         'caption': 'Excitation DOFs'},
        {'kind': 'scene', 'geometry': 'Geometry', 'shapes': '',
         'dofs': 'acceleration', 'dofs_source': 'FRF',
         'caption': 'Response DOFs'},
        {'kind': 'scene', 'geometry': 'Geometry', 'shapes': '',
         'dofs': 'voltage', 'dofs_source': 'FRF',
         'caption': 'nothing measured as voltage'},
    ])
    payload = json.loads(render_html(report, project).split(
        'type="application/json">')[1].split('</script>')[0])
    assert len(payload['blocks']) == 2, (
        'no voltage anywhere, so that scene renders nothing')
    force, response = payload['blocks']
    assert [a['label'] for a in force['arrows']] == quantity_dofs(
        project, 'force')
    geometry = project['Geometry']
    rows = {int(n): i for i, n in enumerate(geometry.node_id)}
    axis_color = {'X': '#e5534b', 'Y': '#3fb950', 'Z': '#4c92d9'}
    for arrow in force['arrows'] + response['arrows']:
        node = int(arrow['label'].rstrip('+-XYZR'))
        assert arrow['node'] == rows[node]
        assert np.linalg.norm(arrow['vector']) == pytest.approx(1.0)
        letter = arrow['label'].rstrip('+-').lstrip('0123456789R')[:1]
        assert arrow['color'] == axis_color[letter], (
            'colored by axis, the orientation marker convention')
    # forces point into the node; responses leave it
    assert force['arrows_incoming'] is True
    assert response['arrows_incoming'] is False
    # one shared length per plot. On this survey both sets sit at the
    # 12%-of-extent cap — the grid is not dense enough to shrink below
    # it — so equality is allowed; the shrink rule itself is pinned in
    # test_dense_arrows_shrink_below_the_cap.
    assert len(response['arrows']) > len(force['arrows'])
    assert response['arrow_length'] <= force['arrow_length']


def test_dense_arrows_shrink_below_the_cap():
    """The rule the scenes rely on: arrows never overlap a close
    neighbor, however coarse the 12%-of-extent default would be."""
    from visualdynamics.viz.geometry import dof_arrow_length

    sparse = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert dof_arrow_length(sparse, extent=1.0) == pytest.approx(0.12)
    dense = [[0.01 * i, 0.0, 0.0] for i in range(11)]
    assert dof_arrow_length(dense, extent=1.0) == pytest.approx(0.009)


def test_the_dofs_insert_binds_the_geometry(window, pump, project):
    for name, obj in project.items():
        window.add_object(name, obj)
    assert ('dofs:force', 'DOFs (Force)') in insert_options(window.objects)
    name = window.generate_report('empty')
    window.tree.setCurrentItem(window._item_for_object(name))
    window.render_current()
    pump()
    window.report_editor._operate({'op': 'insert', 'at': 0,
                                   'kind': 'dofs:force'})
    block = window.objects[name].blocks[0]
    assert block == {'kind': 'scene', 'geometry': 'Geometry',
                     'shapes': '', 'dofs': 'force', 'dofs_source': 'FRF',
                     'caption': ''}


def test_the_template_includes_excitation_and_response_dofs(project):
    """The DOFs figures read from the FRFs by default — the excitation
    from its references, the responses from its responses — in the
    standing order: excitations first, then the responses (Brandon,
    2026-08-23). Only the quantities the source measures get a scene
    (2026-09-08): an FRF per unit force names force and acceleration
    and no voltage, so no voltage card is left empty in the editor."""
    template = modal_template(project)
    blocks = [b for b in template.blocks
              if b.get('kind') == 'scene' and b.get('dofs')]
    assert [b['dofs'] for b in blocks] == ['force', 'acceleration']
    assert all(b['dofs_source'] == '@basis:Frf' for b in blocks)
    # a stale source makes the block an unbound slot, never an error
    template.blocks[template.blocks.index(blocks[0])]['dofs_source'] = 'X'
    assert template.blocks.index(blocks[0]) in template.unbound(project)


def test_the_quantity_drop_down_needs_geometry_plus_data(window, pump,
                                                         project):
    """The arrows read the selection.

    A geometry alone marks its own degrees of freedom — every node's X,
    Y and Z — and offers no quantity, because there is nothing measured
    to have a quantity. Beside data it offers that data's, and marks
    what was measured instead.
    """
    for name, obj in project.items():
        window.add_object(name, obj)
    window.tree.clearSelection()
    item = window._item_for_object('Geometry')
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    assert window.dofs_action.isVisible(), (
        'a geometry has DOFs of its own to mark')
    assert not window.dofs_combo_action.isVisible(), (
        'and nothing measured to choose a quantity from')
    window._item_for_object('FRF').setSelected(True)
    window.render_current()
    pump()
    assert window.dofs_action.isVisible()
    window.dofs_action.setChecked(True)
    pump()
    quantities = [window.dofs_combo.itemData(i)
                  for i in range(window.dofs_combo.count())]
    assert quantities == ['acceleration', 'force']
    assert window.dofs_combo_action.isVisible()
    window.dofs_combo.setCurrentIndex(1)      # Force
    pump()
    window.dofs_action.setChecked(False)
    pump()
    assert not window.dofs_combo_action.isVisible()
