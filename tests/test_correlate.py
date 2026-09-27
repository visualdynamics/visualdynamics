"""Test-analysis correlation: FEM shapes projected onto test DOFs.

The FEM is a unit cube with an analytically known displacement field,
the test a couple of uniaxial sensors — one in a rotated local frame —
so the projected values are checkable by hand: matching, rotation into
the local frame, and the uniaxial dot product all pin exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.correlate import project_shapes
from visualdynamics.core.shapes import ShapeSet


def _fem():
    corners = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    geometry = visualdynamics.Geometry(node_id=list(range(1, 9)), node_xyz=corners)
    dofs = [f'{n}{d}+' for n in range(1, 9) for d in 'XYZ']
    # mode 1 displaces every node by (x+10, y+20, z+30); mode 2 purely
    # in Y by the node's own x — both fields checkable by hand
    m1 = np.array([v for (x, y, z) in corners
                   for v in (x + 10.0, y + 20.0, z + 30.0)])
    m2 = np.array([v for (x, y, _z) in corners for v in (0.0, x, 0.0)])
    shapes = ShapeSet([10.0, 20.0], [0.01, 0.02], dofs,
                      np.vstack([m1, m2]))
    return geometry, shapes


def _test_model(extra_node=False):
    node_id = [101, 102] + ([103] if extra_node else [])
    node_xyz = [[0.001, 0.0, 0.0], [1.0, 1.0, 0.999]] \
        + ([[5.0, 5.0, 5.0]] if extra_node else [])
    # node 102 measures along a local X that is global Y
    geometry = visualdynamics.Geometry(
        node_id=node_id, node_xyz=node_xyz,
        node_disp_cs=[0, 5] + ([0] if extra_node else []),
        cs_id=[5], cs_type=[0], cs_name=['rotated'],
        cs_matrix=[[[0, 1, 0], [-1, 0, 0], [0, 0, 1], [0, 0, 0]]])
    dofs = ['101X+', '102X+'] + (['103X+'] if extra_node else [])
    shapes = ShapeSet([1.0, 2.0], [0.01, 0.01], dofs,
                      np.ones((2, len(dofs))))
    return geometry, shapes


def test_projection_matches_rotates_and_samples():
    fem_geometry, fem_shapes = _fem()
    test_geometry, test_shapes = _test_model()
    projected, report = project_shapes(fem_shapes, fem_geometry,
                                       test_shapes, test_geometry)
    assert projected.coordinate == ['101X+', '102X+']
    assert list(projected.frequency) == [10.0, 20.0], (
        "the projected set keeps the FEM's own modal parameters")
    # 101X+ reads the global X displacement at corner (0,0,0);
    # 102X+ is a local X that is global Y, read at corner (1,1,1)
    assert projected.shape_matrix[0] == pytest.approx([10.0, 21.0])
    assert projected.shape_matrix[1] == pytest.approx([0.0, 1.0])
    assert report['matched'] == 2 and report['total'] == 2
    assert report['dropped'] == []
    assert report['worst'] == pytest.approx(0.001, rel=0.2)


def test_the_tolerance_drops_far_nodes_until_loosened():
    fem_geometry, fem_shapes = _fem()
    test_geometry, test_shapes = _test_model(extra_node=True)
    projected, report = project_shapes(fem_shapes, fem_geometry,
                                       test_shapes, test_geometry)
    assert '103X+' in report['dropped'], 'far outside the tolerance'
    assert projected.coordinate == ['101X+', '102X+']
    loosened, report = project_shapes(fem_shapes, fem_geometry,
                                      test_shapes, test_geometry,
                                      tolerance=2.0)
    assert loosened.coordinate == ['101X+', '102X+', '103X+']
    assert report['dropped'] == []


def test_mismatched_unit_definitions_are_refused():
    fem_geometry, fem_shapes = _fem()
    test_geometry, test_shapes = _test_model()
    test_geometry.define_units('m')
    with pytest.raises(ValueError, match='units defined'):
        project_shapes(fem_shapes, fem_geometry,
                       test_shapes, test_geometry)


def test_the_reduced_test_geometry_matches_the_modal_run():
    """test_geometry.npz is the survey cut to the 13 nodes the
    Rattlesnake modal test measured — a strict subset of the model,
    with tracelines threaded through the sensors."""
    from conftest import fixture_path

    from visualdynamics.core.data import parse_dof

    reduced = visualdynamics.import_file(fixture_path('plate',
                                            'test_geometry.npz'))
    full = visualdynamics.import_file(fixture_path('plate',
                                         'geometry.npz'))
    contents = visualdynamics.import_file(fixture_path('plate',
                                             'modal_spectra.nc4'))
    measured = set()
    for obj in contents.values():
        for dof in getattr(obj, 'response_dof', []) or []:
            measured.add(parse_dof(dof)[0])
        for dof in getattr(obj, 'reference_dof', None) or []:
            measured.add(parse_dof(dof)[0])
        if hasattr(obj, 'dof_strings'):
            measured |= {parse_dof(dof)[0] for dof in obj.dof_strings()}
    measured.discard(None)
    reduced_nodes = {int(node) for node in reduced.node_id}
    assert reduced_nodes == measured
    assert reduced_nodes < {int(node) for node in full.node_id}
    assert len(reduced.traceline_conn) >= 2, (
        'the sensors thread into tracelines, not a dot cloud')


def _select(window, *names):
    window.tree.clearSelection()
    for name in names:
        window._item_for_object(name).setSelected(True)


def _populate(window):
    fem_geometry, fem_shapes = _fem()
    test_geometry, test_shapes = _test_model()
    window.add_object('Test Geometry', test_geometry)
    window.add_object('Test Shapes', test_shapes)
    window.add_object('FEM Geometry', fem_geometry)
    window.add_object('FEM Shapes', fem_shapes)
    _select(window, 'Test Geometry', 'Test Shapes')
    window.link_selected()
    _select(window, 'FEM Geometry', 'FEM Shapes')
    window.link_selected()
    _select(window, 'Test Shapes', 'FEM Shapes')


def test_the_action_needs_the_links_never_a_guess(window, pump):
    _populate(window)
    candidates = window._correlation_candidates()
    assert candidates is not None
    assert candidates['basis'][0] == 'Test Shapes', (
        'without a declared Basis, the sparser set stands in')
    assert candidates['other'][0] == 'FEM Shapes'
    assert candidates['basis'][3] is window.objects['Test Geometry'], (
        'the geometry comes from the link, not node coverage')
    # a declared Basis outranks the sparser-set convention entirely
    window.set_link_role('FEM Shapes', 'Basis')
    swapped = window._correlation_candidates()
    assert swapped['basis'][0] == 'FEM Shapes', (
        'the Basis decides, however the DOF counts fall')
    window.set_link_role('Test Shapes', 'Basis')
    # break one link: the action withdraws — explicit, not inferred
    _select(window, 'FEM Shapes')
    window.unlink_selected()
    _select(window, 'Test Shapes', 'FEM Shapes')
    assert window._correlation_candidates() is None


def test_the_action_adds_the_projected_set(window, pump, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    _populate(window)      # ends with the two shape sets selected
    monkeypatch.setattr(QInputDialog, 'getDouble',
                        staticmethod(lambda *a, **k: (2.0, True)))
    window.project_onto_basis()
    projected = window.objects['FEM Shapes @ Basis DOFs']
    assert projected.coordinate == ['101X+', '102X+']
    assert '2 of 2 basis nodes matched' in \
        window.statusBar().currentMessage()
    assert 'FEM Shapes @ Basis DOFs' in \
        window.linked_group('Test Shapes'), (
        'the projection lives at the basis DOFs, so it joins that side')


def test_comparing_across_geometries_always_projects(window, pump):
    """Selecting two sets on different geometries never yields a raw
    name-matched MAC: the other set is projected onto the basis DOFs
    first. These fixtures share no DOF names at all — without the
    projection there is no MAC to show."""
    from visualdynamics.core.shapes import cross_mac

    _populate(window)      # ends with the two shape sets selected
    window.set_link_role('Test Shapes', 'Basis')
    window.render_current()
    pump()
    matrix = window._compare['matrix']
    projected, _report = project_shapes(
        window.objects['FEM Shapes'], window.objects['FEM Geometry'],
        window.objects['Test Shapes'], window.objects['Test Geometry'])
    expected = cross_mac(window.objects['Test Shapes'], projected)
    assert matrix == pytest.approx(expected), (
        'the displayed MAC is the projected comparison')
    assert window._compare['pair'] == ('Test Shapes', 'FEM Shapes'), (
        'the sparser set leads: rows, table, and baseline')
    assert 'projected onto Test Shapes DOFs' in \
        window.statusBar().currentMessage()
    # the overlay animates each set on its own linked geometry —
    # basis shapes on the basis mesh, FEM shapes on the FEM mesh —
    # phase-aligned to each other through the projection
    from visualdynamics.viz.animate import PairedAnimator

    assert isinstance(window.animator, PairedAnimator)
    assert window.animator.first.geometry is \
        window.objects['Test Geometry']
    assert window.animator.second.geometry is \
        window.objects['FEM Geometry']
    # each mesh wears its bracket's color: the basis blue, the FEM
    # group the first non-basis color of the curve palette
    from visualdynamics.plot import curve_color

    assert window._compare['colors'] == (
        window.LINK_ROLE_COLORS['Basis'], curve_color(1))
    # the taskbar toggle switches the second set to its projection on
    # the basis mesh, and back
    assert window.project_action.isVisible()
    window.project_action.setChecked(True)
    window.render_current()
    pump()
    assert window.animator.second.geometry is \
        window.objects['Test Geometry'], (
        'projected: both sets animate on the basis mesh')
    window.project_action.setChecked(False)
    window.render_current()
    pump()
    assert window.animator.second.geometry is \
        window.objects['FEM Geometry']


def test_the_report_animates_the_matched_overlay(window, pump):
    """The overlay block: both geometries in one scene — the basis
    side blue, the model orange — the second set phase-aligned
    through the projection, each part normalized to its own peak, and
    corner info for both halves of the pair."""
    import json

    from visualdynamics.core.report import Report
    from visualdynamics.report import render_html

    _populate(window)
    window.render_current()
    pump()
    window.add_matches()
    matched = window.objects['Matched Modes']
    assert matched.first_geometry == 'Test Geometry', (
        'the matches remember the geometries they were committed on')
    assert matched.second_geometry == 'FEM Geometry'
    report = Report('r', [{'kind': 'overlay', 'source': 'Matched Modes',
                           'caption': ''}])
    payload = json.loads(render_html(report, window.objects).split(
        'type="application/json">')[1].split('</script>')[0])
    scene = payload['blocks'][0]
    assert scene['kind'] == 'scene' and scene['flat'] is True
    assert len(scene['points']) == 2 + 8, 'both geometries, one scene'
    assert scene['node_colors'][0] == '#4c92d9', 'basis side blue'
    assert scene['node_colors'][-1] == '#ff8c2b', 'model side orange'
    mode = scene['modes'][0]
    assert 'MAC' in mode['label']
    assert [entry['set'] for entry in mode['info']] == [
        'Test Shapes', 'FEM Shapes']
    real = np.asarray(mode['real'])
    assert real.shape == (10, 3)
    # each part swings to its own unit peak: comparable amplitudes
    # whatever the sets' arbitrary scalings
    assert np.max(np.linalg.norm(real[:2], axis=1)) == pytest.approx(1.0)
    assert np.max(np.linalg.norm(real[2:], axis=1)) == pytest.approx(1.0)
    # matches committed before the object remembered its geometries
    # still animate: the renderer falls back to the link groups
    matched.first_geometry = None
    matched.second_geometry = None
    payload = json.loads(
        render_html(report, window.objects, links=window.links).split(
            'type="application/json">')[1].split('</script>')[0])
    aged = payload['blocks'][0]
    assert aged['kind'] == 'scene' and len(aged['points']) == 10


def test_the_report_cross_mac_binds_the_projected_set():
    """The modal template's correlation block: unbound without a second
    set, the cross-MAC against the projection when it exists."""
    import json

    from visualdynamics.core.report import modal_template
    from visualdynamics.core.shapes import cross_mac
    from visualdynamics.report import render_html

    fem_geometry, fem_shapes = _fem()
    test_geometry, test_shapes = _test_model()
    projected, _report = project_shapes(fem_shapes, fem_geometry,
                                        test_shapes, test_geometry)
    bare = {'Geometry': test_geometry, 'Shape Set': test_shapes}
    template = modal_template(bare)
    correlation = next(b for b in template.blocks
                       if b.get('mode') == 'mac' and b.get('shapes'))
    assert correlation['shapes'] == 'FEM Shapes @ Basis DOFs', (
        'bound to the name the projection produces — a slot until then')
    payload = json.loads(render_html(template, bare).split(
        'type="application/json">')[1].split('</script>')[0])
    macs = [b for b in payload['blocks'] if b['kind'] == 'mac']
    assert len(macs) == 1, 'without the projection, only the auto-MAC'

    # the old '@ Test DOFs' suffix still binds — saved projects keep it
    full = dict(bare)
    full['FEM Shapes @ Test DOFs'] = projected
    payload = json.loads(render_html(modal_template(full), full).split(
        'type="application/json">')[1].split('</script>')[0])
    macs = [b for b in payload['blocks'] if b['kind'] == 'mac']
    assert len(macs) == 2, 'the cross-MAC renders beside the auto-MAC'
    cross = macs[1]
    expected = cross_mac(test_shapes, projected)
    assert np.asarray(cross['matrix']) == pytest.approx(expected)
    assert len(cross['rows']) == test_shapes.num_shapes
    assert len(cross['columns']) == projected.num_shapes

def test_sensors_on_a_plates_surface_are_found_on_its_mid_surface():
    """A plate model's nodes are on mid-surfaces and its sensors on
    surfaces, half a thickness away. The match limit allows half the
    thickest plate beyond its fraction of the extent — without it, the
    BARC's box lost every sensor on its quarter-inch walls
    (2026-09-26)."""
    import numpy as np

    from visualdynamics import fem, mesh
    from visualdynamics.core.correlate import project_shapes
    from visualdynamics.core.geometry import Geometry
    from visualdynamics.core.shapes import ShapeSet

    plate = mesh.plane((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.25, 'plate')
    plate.block_properties = {1: fem.BlockProperties(
        fem.material('6061-T6'), 0.1)}
    dofs = [f'{n}Z+' for n in plate.node_id]
    fem_shapes = ShapeSet([10.0], [0.0], dofs,
                          plate.node_xyz[:, 0][np.newaxis] + 1.0)
    corners = np.array([[0.0, 0.0, 0.05], [1.0, 0.0, 0.05],
                        [0.0, 1.0, 0.05], [1.0, 1.0, 0.05]])
    sensors = Geometry(node_id=[1, 2, 3, 4], node_xyz=corners,
                       length_unit='m')
    test_shapes = ShapeSet([10.0], [0.0], ['1Z+', '2Z+', '3Z+', '4Z+'],
                           np.ones((1, 4)))
    _projected, report = project_shapes(fem_shapes, plate, test_shapes,
                                        sensors)
    assert report['matched'] == report['total'] == 4
    assert not report['dropped']
