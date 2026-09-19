"""The Project: the GUI's own container, scriptable.

A Project holds what the project tree shows — named objects, link
groups, the Basis, the type, the active geometry — and gives scripts
the same verbs the GUI's buttons call, so neither side has to
hand-assemble dicts of members and roles.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from conftest import fixture_path

import visualdynamics
from visualdynamics import io
from visualdynamics.core.matches import MatchedModes
from visualdynamics.core.report import Report
from visualdynamics.core.shapes import ShapeSet


def _geometry(node=101):
    return visualdynamics.Geometry(node_id=[node], node_xyz=[[0.0, 0.0, 0.0]])


def _shapes(node=101):
    return ShapeSet([10.0], [0.01], [f'{node}X+'], np.ones((1, 1)))


def test_an_empty_project_is_a_named_empty_structure():
    project = visualdynamics.Project('Beam Airplane Modal Test')
    assert project.name == 'Beam Airplane Modal Test'
    assert list(project) == [] and len(project) == 0
    assert project.links == [] and not project.basis
    assert project.project_type is None
    assert project.active_geometry is None
    assert isinstance(project, dict), 'a project *is* its objects'


def test_adding_numbers_a_clash_and_adopts_the_first_geometry():
    project = visualdynamics.Project()
    assert project.add('Geometry', _geometry()) == 'Geometry'
    assert project.active_geometry == 'Geometry', 'the first one in'
    assert project.add('Geometry', _geometry(102)) == 'Geometry (2)', (
        'a clash is numbered, never overwritten'
    )
    assert project.active_geometry == 'Geometry', 'and it stays'
    project['FRF'] = _shapes()      # the plain mapping still works
    assert set(project) == {'Geometry', 'Geometry (2)', 'FRF'}


def test_linking_merges_groups_and_refuses_a_second_geometry():
    project = visualdynamics.Project()
    project.add('Geometry', _geometry())
    project.add('Shapes', _shapes())
    project.add('More Shapes', _shapes())
    project.link('Geometry', 'Shapes')
    project.link('Shapes', 'More Shapes')        # merges into one group
    assert set(project.group_of('Geometry')) == {
        'Geometry', 'Shapes', 'More Shapes'}
    project.add('Other Geometry', _geometry(900))
    with pytest.raises(ValueError, match='one geometry'):
        project.link('Geometry', 'Other Geometry')
    with pytest.raises(KeyError):
        project.link('Geometry', 'Nothing')


def test_linking_refuses_dofs_the_geometry_lacks():
    """A link says these objects describe one structure; shapes at
    nodes the geometry has never heard of are not that."""
    project = visualdynamics.Project()
    project.add('Geometry', _geometry(101))
    project.add('Stranger Shapes', _shapes(900))
    with pytest.raises(ValueError, match='cannot link'):
        project.link('Geometry', 'Stranger Shapes')
    assert project.links == [], 'a refused link changes nothing'


def test_the_basis_is_declared_once_and_moves():
    project = visualdynamics.Project()
    project.add('Geometry', _geometry())
    project.add('Shapes', _shapes())
    project.add('FEM Geometry', _geometry(900))
    project.add('FEM Shapes', _shapes(900))
    assert project.set_basis('Geometry', 'Shapes') == ['Geometry', 'Shapes']
    assert project.role_of('Shapes') == 'Basis'
    project.link('FEM Geometry', 'FEM Shapes')
    assert project.role_of('FEM Shapes') is None
    project.set_basis('FEM Shapes')
    assert project.basis.names == ['FEM Geometry', 'FEM Shapes']
    assert project.role_of('Shapes') is None, 'the role is unique'


def test_the_geometry_an_object_answers_to():
    project = visualdynamics.Project()
    project.add('Geometry', _geometry())
    project.add('FEM Geometry', _geometry(900))
    project.add('FEM Shapes', _shapes(900))
    project.link('FEM Geometry', 'FEM Shapes')
    assert project.geometry_for('FEM Shapes')[0] == 'FEM Geometry', (
        'its link group, not the active geometry')
    project.add('Stranger', _shapes())
    assert project.geometry_for('Stranger')[0] == 'Geometry', (
        'unlinked: the active one')


def test_renaming_carries_every_reference_with_it():
    project = visualdynamics.Project()
    project.add('Geometry', _geometry())
    project.add('Shapes', _shapes())
    project.link('Geometry', 'Shapes')
    project.add('Matched', MatchedModes('Shapes', 'Other', [[0, 0]], [1.0]))
    project.add('Report', Report('r', [{'kind': 'plot', 'source': 'Shapes',
                                        'mode': 'mac', 'caption': ''}]))
    project.rename('Shapes', 'Experimental Modes')
    assert 'Shapes' not in project and 'Experimental Modes' in project
    assert project.group_of('Geometry') == ['Geometry',
                                            'Experimental Modes']
    assert project['Matched'].first == 'Experimental Modes'
    assert project['Report'].blocks[0]['source'] == 'Experimental Modes'
    assert list(project).index('Experimental Modes') == 1, 'order held'
    with pytest.raises(ValueError, match='already in use'):
        project.rename('Geometry', 'Experimental Modes')


def test_removing_prunes_the_links():
    project = visualdynamics.Project()
    project.add('Geometry', _geometry())
    project.add('Shapes', _shapes())
    project.add('More Shapes', _shapes())
    project.link('Geometry', 'Shapes', 'More Shapes')
    project.remove('Shapes')
    assert project.group_of('Geometry') == ['Geometry', 'More Shapes']
    project.remove('More Shapes')
    assert project.links == [], 'a group of one dissolves'
    project.remove('Geometry')
    assert project.active_geometry is None


def test_a_project_round_trips_through_the_vibe_file(tmp_path):
    project = visualdynamics.Project('Scripted')
    project.project_type = 'Modal Test'
    project.add('Geometry', _geometry())
    project.add('Shapes', _shapes())
    project.set_basis('Geometry', 'Shapes')
    project.save(tmp_path / 'p.vdyn')

    back = visualdynamics.Project.open(tmp_path / 'p.vdyn')
    assert isinstance(back, visualdynamics.Project)
    assert back.name == 'Scripted'
    assert back.project_type == 'Modal Test'
    assert back.basis.names == ['Geometry', 'Shapes']
    assert back.active_geometry == 'Geometry'
    assert list(back) == ['Geometry', 'Shapes']
    # import_file and io.load give the same thing — a project is a
    # project however it is opened
    assert isinstance(visualdynamics.import_file(tmp_path / 'p.vdyn'), visualdynamics.Project)
    assert isinstance(io.load(str(tmp_path / 'p.vdyn')), visualdynamics.Project)


def test_a_project_prints_as_its_tree():
    """Typing a project's name at a prompt answers what the tree
    answers: what is in here, how it groups, how big each piece is."""
    project = visualdynamics.Project('Beam Airplane Modal Test')
    project.project_type = 'Modal Test'
    project.add('Shapes', _shapes())            # out of order on purpose
    project.add('Geometry', _geometry())
    project.add('FEM Geometry', _geometry(900))
    project.add('FEM Shapes', _shapes(900))
    project.set_basis('Geometry', 'Shapes')
    project.link('FEM Geometry', 'FEM Shapes')

    text = repr(project)
    lines = [line.rstrip() for line in text.splitlines()]
    assert lines[0] == 'Project: Beam Airplane Modal Test  [Modal Test]'
    assert lines[1].strip() == 'Basis', 'the Basis group reads first'
    assert 'Geometry' in lines[2] and 'ShapeSet' not in lines[2], (
        'the canonical type order, whatever order they arrived in')
    assert '1 node' in lines[2], 'and what it holds'
    assert '*' in lines[2], 'the active geometry is marked'
    assert lines[4].strip() == 'linked', 'then the unroled groups'
    assert project.ordered_names() == ['Geometry', 'Shapes',
                                       'FEM Geometry', 'FEM Shapes']
    assert repr(visualdynamics.Project('Empty')).endswith('(empty)')


def test_the_tree_order_is_the_projects_order(window, pump, survey):
    """The GUI's tree and a Project's repr read from the same
    ordering, so they cannot disagree."""
    shapes, frfs = survey
    geometry = visualdynamics.import_file(fixture_path('plate',
                                             'geometry.npz'))
    for name, obj in (('Shapes', shapes), ('FRF', frfs),
                      ('Geometry', geometry)):
        window.add_object(name, obj)
    pump()
    from_tree = [window.test_item.child(i).text(0)
                 for i in range(window.test_item.childCount())]
    project = visualdynamics.Project('x', objects=dict(window.objects),
                           links=window.links)
    assert project.ordered_names() == from_tree


def test_importing_data_files_names_them_as_the_tree_would():
    project = visualdynamics.Project()
    added = project.import_file(fixture_path('plate',
                                             'geometry.npz'))
    assert added == ['Geometry']
    # a reader's keys are for code; the names are what the objects are
    added = project.import_file(fixture_path('plate',
                                             'modal_spectra.nc4'))
    assert 'FRF' in added and 'Time History' in added
    assert 'Multiple Coherence' in added and 'Channel Table' in added


def test_importing_a_project_brings_its_structure(tmp_path):
    """A saved project imported into an empty one arrives whole: its
    links, Basis, type and name."""
    source = visualdynamics.Project('Saved Test')
    source.project_type = 'Modal Test'
    source.add('Geometry', _geometry())
    source.add('Shapes', _shapes())
    source.set_basis('Geometry', 'Shapes')
    source.save(tmp_path / 'saved.vdyn')

    project = visualdynamics.Project()
    added = project.import_file(tmp_path / 'saved.vdyn')
    assert added == ['Geometry', 'Shapes']
    assert project.basis.names == ['Geometry', 'Shapes']
    assert project.name == 'Saved Test'
    assert project.project_type == 'Modal Test'
    assert project.active_geometry == 'Geometry'


def test_importing_a_project_into_a_full_one_keeps_both(tmp_path):
    """The clash is numbered, the imported group follows its objects
    to their new names, and the Basis already spoken for stays put."""
    source = visualdynamics.Project('Second Run')
    source.add('Geometry', _geometry())
    source.add('Shapes', _shapes())
    source.add('Matched', MatchedModes('Shapes', 'Shapes', [[0, 0]], [1.0]))
    source.set_basis('Geometry', 'Shapes')
    source.save(tmp_path / 'second.vdyn')

    project = visualdynamics.Project('First Run')
    project.add('Geometry', _geometry())
    project.add('Shapes', _shapes())
    project.set_basis('Geometry', 'Shapes')

    added = project.import_file(tmp_path / 'second.vdyn')
    assert added == ['Geometry (2)', 'Shapes (2)', 'Matched']
    assert project.group_of('Geometry (2)') == ['Geometry (2)',
                                                'Shapes (2)'], (
        'the imported links came with it, renamed')
    assert project.role_of('Geometry (2)') is None, (
        'the Basis is the project\'s, not the file\'s')
    assert project.basis.names == ['Geometry', 'Shapes']
    assert project['Matched'].first == 'Shapes (2)', (
        'references inside the imported objects followed too')
    assert project.name == 'First Run', 'identity is not overwritten'


def test_a_single_object_file_is_not_a_project(tmp_path):
    io.save(_geometry(), str(tmp_path / 'g.vdyn'))
    with pytest.raises(ValueError, match='not a project'):
        visualdynamics.Project.open(tmp_path / 'g.vdyn')


def test_the_gui_and_a_script_build_the_same_project(window, pump,
                                                     tmp_path,
                                                     monkeypatch):
    """What the GUI saves, a script opens as a Project — and what a
    script saves, the GUI imports as its own project, links and all."""
    from PySide6.QtWidgets import QFileDialog

    geometry = visualdynamics.import_file(fixture_path('plate',
                                             'geometry.npz'))
    shapes = visualdynamics.import_file(fixture_path('plate', 'shapes.npy'))
    window.add_object('Geometry', geometry)
    window.add_object('Shapes', shapes)
    window.tree.clearSelection()
    for name in ('Geometry', 'Shapes'):
        window._item_for_object(name).setSelected(True)
    window.link_selected()
    window.set_link_role('Shapes', 'Basis')
    window.set_project_type('Modal Test')
    path = str(tmp_path / 'from_gui.vdyn')
    monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: (path, '')))
    window.save_test()

    project = visualdynamics.Project.open(path)
    assert project.basis.names == ['Geometry', 'Shapes']
    assert project.project_type == 'Modal Test'
    assert project.role_of('Geometry') == 'Basis'

    # and back the other way: the script's own file into the GUI
    project.rename('Shapes', 'Experimental Modes')
    scripted = str(tmp_path / 'from_script.vdyn')
    project.save(scripted)
    fresh = window.__class__(offscreen_3d=True)
    fresh.import_paths([scripted])
    pump()
    assert fresh.linked_group('Geometry') == ['Geometry',
                                              'Experimental Modes']
    assert fresh.link_role('Geometry') == 'Basis'
    assert fresh.project_type == 'Modal Test'


# ---- the workflow verbs -----------------------------------------------------

def test_the_workflow_verbs_derive_and_link(survey, tmp_path):
    """Each verb adds its result and links it to what it came from,
    the way the GUI's buttons do."""
    _shape_set, frfs = survey
    project = visualdynamics.Project('Scripted')
    project.add('Geometry', visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz')))
    project.add('FRF', frfs)
    project.set_basis('Geometry', 'FRF')

    modes = project.fit_modes('FRF', bounds=(5.0, 60.0), limit=3)
    assert modes == 'FRF Modes'
    assert project[modes].num_shapes >= 1
    assert modes in project.group_of('FRF'), 'linked to what it came from'

    # a report, bound symbolically, exported as one file
    report = project.generate_report('modal')
    block = next(b for b in project[report].blocks
                 if b.get('mode') == 'cmif')
    assert block['source'] == '@basis:Frf', 'no name in the binding'
    path = project.export_report(report, tmp_path / 'r.html')
    assert pathlib.Path(path).read_text(
        encoding='utf-8').lstrip().startswith('<')


def test_matching_uses_the_displayed_comparison(survey):
    """match_modes projects across geometries exactly as the
    comparison screen does, and keeps the MAC values it showed."""
    fem_shapes, _frfs = survey
    test_geometry = visualdynamics.import_file(
        fixture_path('plate', 'test_geometry.npz'))
    fem_geometry = visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz'))
    # a stand-in measured set: the model's own motion at the measured
    # nodes, which is what a good test would have found
    from visualdynamics.core.correlate import project_shapes

    dofs = [f'{int(node)}Z+' for node in test_geometry.node_id]
    stub = ShapeSet(fem_shapes.frequency[:6], fem_shapes.damping[:6],
                    dofs, np.ones((6, len(dofs))))
    measured, _report = project_shapes(fem_shapes, fem_geometry, stub,
                                       test_geometry)

    project = visualdynamics.Project()
    project.add('Test Geometry', test_geometry)
    project.add('Test Modes', measured)
    project.add('FEM Geometry', fem_geometry)
    project.add('FEM Modes', fem_shapes)
    project.link('Test Geometry', 'Test Modes')
    project.link('FEM Geometry', 'FEM Modes')
    project.set_basis('Test Modes')

    matrix = project.comparison_mac('Test Modes', 'FEM Modes')
    assert matrix.shape[0] == measured.num_shapes, (
        'the basis set makes the rows')
    name = project.match_modes('Test Modes', 'FEM Modes', threshold=0.9)
    matched = project[name]
    assert matched.num_matches, 'a set projected from the model matches it'
    assert all(mac >= 0.9 for mac in matched.macs)
    assert matched.first_geometry == 'Test Geometry'
    assert matched.second_geometry == 'FEM Geometry'
    assert project.group_of(name) is None, (
        'a comparison names a set on each side, so it joins neither')
    assert matched.first_geometry == 'Test Geometry', (
        'and needs no group to find its geometries — it holds them')


def test_every_gui_plot_has_a_call(survey, tmp_path):
    """The plots the app draws, drawn from a script: curves, MAC,
    cross-MAC, CMIF with the synthesis over it, the coherence map, the
    geometry, its DOF arrows, and a mode shape."""
    import os

    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    shapes, frfs = survey
    contents = visualdynamics.import_file(fixture_path('plate',
                                             'modal_spectra.nc4'))
    geometry = visualdynamics.import_file(fixture_path('plate',
                                             'geometry.npz'))
    drawings = {
        'curves.png': lambda p: frfs.plot(records=[0], path=p),
        'mac.png': lambda p: shapes.plot_mac(path=p),
        'cross.png': lambda p: shapes.plot_mac(shapes, path=p),
        'cmif.png': lambda p: frfs.plot_cmif(shapes, path=p),
        'map.png': lambda p: contents['Modal_coherence'].plot_map(path=p),
        'geom.png': lambda p: geometry.plot(screenshot=p),
        'dofs.png': lambda p: geometry.plot_dofs(frfs, 'force',
                                                 screenshot=p),
        'mode.png': lambda p: shapes.animate(geometry, 1, screenshot=p),
    }
    for name, draw in drawings.items():
        draw(str(tmp_path / name))
        assert (tmp_path / name).stat().st_size > 1000, name


def test_merge_and_export_are_verbs_too(tmp_path):
    """The last two things the app's menus do: combining compatible
    objects, and writing one out to a foreign format."""
    project = visualdynamics.Project()
    project.add('Front', visualdynamics.Geometry(node_id=[1, 2],
                                       node_xyz=[[0, 0, 0], [1, 0, 0]]))
    project.add('Back', visualdynamics.Geometry(node_id=[3, 4],
                                      node_xyz=[[2, 0, 0], [3, 0, 0]]))
    whole = project.merge('Front', 'Back', name='Airframe')
    assert list(project) == ['Airframe']
    assert project['Airframe'].num_nodes == 4

    path = project.export('Airframe', tmp_path / 'airframe.unv')
    assert pathlib.Path(path).stat().st_size > 100
    assert visualdynamics.import_file(path).num_nodes == 4, 'and it reads back'
    assert whole == 'Airframe'


def test_the_two_front_ends_reach_the_same_project(window, pump):
    """The whole point, checked: the same job done by clicking and by
    calling lands on the same project — same objects, same links, same
    Basis, same identified modes."""

    def sources():
        contents = visualdynamics.import_file(fixture_path('plate',
                                                 'modal_spectra.nc4'))
        geometry = visualdynamics.import_file(fixture_path('plate',
                                                 'test_geometry.npz'))
        geometry.define_units('m')
        return contents, geometry

    # --- by calling ---
    contents, geometry = sources()
    scripted = visualdynamics.Project('Beam Airplane')
    scripted.project_type = 'Modal Test'
    scripted.add('Geometry', geometry)
    scripted.add('FRF', contents['Modal_frf'])
    scripted.add('Time History', contents['time_data'])
    scripted.set_basis('Geometry', 'FRF', 'Time History')
    scripted.compute_psds('Time History')
    scripted.fit_modes('FRF', bounds=(5.0, 60.0), limit=4)

    # --- by clicking ---
    contents, geometry = sources()
    window.set_project_type('Modal Test')
    window.add_object('Geometry', geometry)
    window.add_object('FRF', contents['Modal_frf'])
    window.add_object('Time History', contents['time_data'])
    window.tree.clearSelection()
    for name in ('Geometry', 'FRF', 'Time History'):
        window._item_for_object(name).setSelected(True)
    window.link_selected()
    window.set_link_role('FRF', 'Basis')
    window.tree.setCurrentItem(window._item_for_object('Time History'))
    window.compute_psds()
    # the fitting screen's loop is the same verb underneath
    window.project.fit_modes('FRF', bounds=(5.0, 60.0), limit=4)
    pump()

    clicked = window.project
    assert set(clicked) == set(scripted)
    assert clicked.ordered_names() == scripted.ordered_names()
    assert sorted(clicked.basis.names) == sorted(scripted.basis.names)
    assert clicked.project_type == scripted.project_type
    assert clicked.active_geometry == scripted.active_geometry
    assert np.allclose(clicked['FRF Modes'].frequency,
                       scripted['FRF Modes'].frequency)
    assert (clicked.group_of('Time History PSDs')
            == scripted.group_of('Time History PSDs'))


# ---- reaching objects by what they are -------------------------------------

def test_objects_are_reached_by_type_not_by_name(survey):
    """A name is arbitrary and goes stale; a type does not."""
    shapes, frfs = survey
    project = visualdynamics.Project()
    project.add('Geometry', visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz')))
    project.add('Measured FRFs', frfs)          # any name at all
    project.add('Identified Modes', shapes)
    project.set_basis('Geometry', 'Measured FRFs', 'Identified Modes')

    assert project.frf is frfs, 'across the project'
    assert project.basis.frf is frfs, 'and within a group'
    assert project.basis.shapes is shapes
    assert project.basis.geometry.num_nodes == 169
    assert project.frfs == [frfs], 'the plural is always a list'
    assert project.specifications == [], 'even when there are none'


def test_the_singular_says_which_ones_it_found(survey):
    """Sometimes-an-object-sometimes-a-list would break code the day a
    second object arrives; this refuses and names them instead."""
    shapes, _frfs = survey
    project = visualdynamics.Project()
    project.add('First', shapes)
    assert project.shapes is shapes
    project.add('Second', shapes)
    with pytest.raises(AttributeError) as raised:
        _ = project.shapes
    message = str(raised.value)
    assert "'First'" in message and "'Second'" in message
    assert '.shape_sets[i]' in message, 'and how to pick one'
    assert len(project.shape_sets) == 2, 'the plural still answers'
    with pytest.raises(AttributeError, match='no frf in'):
        _ = project.frf


def test_completion_offers_what_the_project_holds(survey):
    shapes, frfs = survey
    project = visualdynamics.Project()
    project.add('Shapes', shapes)
    offered = set(dir(project))
    assert {'shapes', 'shape_sets'} <= offered
    assert 'frf' not in offered, 'nothing here is an FRF yet'
    project.add('FRF', frfs)
    assert {'frf', 'frfs'} <= set(dir(project)), 'and now there is'
    # the scoped view offers only what is in scope
    project.set_basis('Shapes', 'FRF')
    assert {'shapes', 'frf', 'names'} <= set(dir(project.basis))


def test_the_view_is_never_a_snapshot(survey):
    """Attributes are computed on access, so adds, renames and deletes
    show up without anything to invalidate."""
    shapes, frfs = survey
    project = visualdynamics.Project()
    project.add('Shapes', shapes)
    assert project.shapes is shapes
    project.rename('Shapes', 'Truth')
    assert project.shapes is shapes, 'renaming changes the name, not the kind'
    project.add('FRF', frfs)
    assert project.frf is frfs
    project.remove('Truth')
    with pytest.raises(AttributeError):
        _ = project.shapes


def test_groups_are_reached_the_same_way(survey):
    shapes, frfs = survey
    project = visualdynamics.Project()
    project.add('Test Geometry', visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz')))
    project.add('FRF', frfs)
    project.add('FEM Geometry', visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz')))
    project.add('FEM Modes', shapes)
    project.set_basis('Test Geometry', 'FRF')
    project.link('FEM Geometry', 'FEM Modes')

    assert project.basis.frf is frfs
    assert project.other.shapes is shapes, 'the one group that is not it'
    assert [group.names for group in project.groups] == [
        ['Test Geometry', 'FRF'], ['FEM Geometry', 'FEM Modes']]
    assert not visualdynamics.Project().basis, 'no Basis declared: empty, not everything'


def test_verbs_take_objects_as_well_as_names():
    """So the two halves compose: reach an object by type, hand it
    straight to the verb that stores and links the result."""
    time = _time_history()
    project = visualdynamics.Project()
    project.add('Time History', time)
    added = project.compute_psds(project.time_history)
    assert added == 'Time History PSDs'
    assert project.psd is project[added]
    assert added in project.group_of(project.time_history)
    assert project.name_of(project.psd) == added
    with pytest.raises(ValueError, match='not in this project'):
        project.name_of(_time_history())


def _time_history():
    from visualdynamics.core.data import TimeHistory

    t = np.arange(256) / 256.0
    rows = [np.sin(2 * np.pi * 16 * t)] * 2
    time = TimeHistory(t, np.array(rows), response_dof=['1X+', '1X+'],
                       block=['avg 1', 'avg 2'])
    time.define_units('m/s^2')
    return time


def test_a_refusing_property_keeps_its_own_message():
    """Python routes an AttributeError raised *inside* a property to
    __getattr__, which would answer 'no such kind of object' — the
    wrong reason entirely."""
    project = visualdynamics.Project()
    for tag, node in (('Test', 101), ('FEM', 200), ('Updated', 300)):
        project.add(f'{tag} Geometry', _geometry(node))
        project.add(f'{tag} Modes', _shapes(node))
        project.link(f'{tag} Geometry', f'{tag} Modes')
    project.set_basis('Test Modes')
    with pytest.raises(AttributeError, match='besides the Basis'):
        _ = project.other
    assert len(project.groups) == 3, 'and every one is reachable'
    assert project.groups[2].shapes is project['Updated Modes']


def test_no_object_type_sorts_after_the_report():
    """Every kind of object reads above the report that reports on it.

    `type_rank` walks TYPE_ORDER and takes the first `isinstance` match,
    so a class that subclasses nothing listed falls through to
    `len(TYPE_ORDER)` — past the end, and past Report with it. That is
    not a wrong position, it is *no* position, and it is silent: the
    type appears, sorts last, and nothing says why.

    It happened. `Srs` subclasses no listed class, so a shock project
    listed Report above both its SRS and its shock specification — the
    report over the two things the report is about. Both project types
    that run a shock had it.

    Written against `NAMED_CLASSES` — the native format's own list of
    every persistable data class — plus the non-data types, so adding a
    class anywhere makes this fail rather than making a tree quietly
    wrong.
    """
    from visualdynamics.core.channel_table import ChannelTable
    from visualdynamics.core.data import NAMED_CLASSES
    from visualdynamics.core.geometry import Geometry
    from visualdynamics.core.matches import MatchedModes
    from visualdynamics.core.photos import Photos
    from visualdynamics.core.report import Report
    from visualdynamics.core.shapes import ShapeSet
    from visualdynamics.project import TYPE_ORDER, type_rank

    everything = [*NAMED_CLASSES.values(), Geometry, Photos, ChannelTable,
                  ShapeSet, MatchedModes, Report]
    homeless = [cls.__name__ for cls in everything
                if type_rank(cls.__new__(cls)) >= len(TYPE_ORDER)]
    assert not homeless, (
        f'{homeless} match nothing in TYPE_ORDER, so they sort after '
        'Report. Give the class an entry, or place it under one it '
        'subclasses.')

    report = type_rank(Report.__new__(Report))
    below = [cls.__name__ for cls in everything
             if cls is not Report and type_rank(cls.__new__(cls)) >= report]
    assert not below, f'{below} sort at or after the Report'


def test_a_shock_project_reads_in_derivation_order():
    """The bug the rank guard is about, from the outside: the tree."""
    import numpy as np

    from visualdynamics.core.data import ShockSpecification, Srs, TimeHistory
    from visualdynamics.core.report import Report

    f = np.linspace(10.0, 2000.0, 40)
    units = {'response_dof': ['1Z+'], 'ordinate_dim': 'acceleration',
             'ordinate_unit': 'm/s**2'}
    t = np.linspace(0.0, 1.0, 64)
    project = visualdynamics.Project('Shock')
    project.add('Report', Report('R'))
    project.add('SRS', Srs(f, np.ones((1, 40)), **units))
    project.add('Shock Specification',
                ShockSpecification(f, np.ones((1, 40)) * 2.0, **units))
    project.add('Time Data', TimeHistory(t, np.sin(t)[None], **units))
    assert project.ordered_names() == [
        'Time Data', 'SRS', 'Shock Specification', 'Report'], (
        'the report reads last, under the two things it is about')


def test_a_kind_never_answers_with_what_a_narrower_kind_claims():
    """The same rule the report's symbolic bindings live by, arrived at
    the same way. A TransientSpecification *is* a TimeHistory, so in a
    transient project `project.time_history` found two and refused,
    though there is exactly one thing anyone asking means; and
    `project.psds` handed back the specification alongside the measured
    PSDs, so every caller filtered it back out by hand.
    """
    import numpy as np

    from visualdynamics.core.data import (
        Psd,
        Specification,
        TimeHistory,
        TransientSpecification,
    )

    project = visualdynamics.Project()
    project.add('History', TimeHistory(
        np.arange(8) / 256.0, np.ones((1, 8)), response_dof=['101Z+']))
    project.add('Target', TransientSpecification(
        np.arange(8) / 256.0, np.ones((1, 8)), response_dof=['101Z+']))
    assert project.time_history is project['History'], (
        'the broad kind means the thing that is only that kind')
    assert project.transient_specification is project['Target']
    assert [project.name_of(h) for h in project.time_histories] == \
        ['History'], 'the plural excludes it too'

    project.add('PSD', Psd(np.arange(4.0), np.ones((1, 4)),
                           response_dof=['101Z+']))
    project.add('Spec', Specification(np.arange(4.0), np.ones((1, 4)),
                                      response_dof=['101Z+']))
    assert project.psd is project['PSD']
    assert project.specification is project['Spec']


def test_plot_mac_draws_the_projected_comparison(survey, tmp_path,
                                                 monkeypatch):
    """Anything the interface can do the API can do: the comparison
    screen draws a MAC projected across geometries, and until this
    verb existed a script could only compute it (`comparison_mac`) or
    draw the wrong one — a shape set's own `plot_mac` compares by DOF
    name, which across two geometries shares nothing (Brandon,
    2026-09-02, after asking where the MAC's math lives)."""
    from visualdynamics.core.correlate import project_shapes
    from visualdynamics.core.shapes import cross_mac

    fem_shapes, _frfs = survey
    test_geometry = visualdynamics.import_file(
        fixture_path('plate', 'test_geometry.npz'))
    fem_geometry = visualdynamics.import_file(
        fixture_path('plate', 'geometry.npz'))
    # a test lab numbers its sensors its own way: nothing shares a name
    # with the model, and only the geometry says which is where
    for row in range(len(test_geometry.node_id)):
        test_geometry.renumber_node(row, int(test_geometry.node_id[row]) + 5000)
    dofs = [f'{int(node)}Z+' for node in test_geometry.node_id]
    stub = ShapeSet(fem_shapes.frequency[:6], fem_shapes.damping[:6],
                    dofs, np.ones((6, len(dofs))))
    measured, _report = project_shapes(fem_shapes, fem_geometry, stub,
                                       test_geometry)
    project = visualdynamics.Project()
    project.add('Test Geometry', test_geometry)
    project.add('Test Modes', measured)
    project.add('FEM Geometry', fem_geometry)
    project.add('FEM Modes', fem_shapes)
    project.link('Test Geometry', 'Test Modes')
    project.link('FEM Geometry', 'FEM Modes')

    # the name-based comparison is not the displayed one here
    with pytest.raises(ValueError, match='share no DOFs'):
        cross_mac(measured, fem_shapes)

    import visualdynamics.plot as plotting

    drawn = {}

    def capture(frequencies, matrix, **kwargs):
        drawn.update(frequencies=frequencies, matrix=matrix, **kwargs)
        return 'drawn'

    monkeypatch.setattr(plotting, 'plot_mac_matrix', capture)
    assert project.plot_mac('Test Modes', 'FEM Modes', title='x') == 'drawn'
    assert np.allclose(drawn['matrix'],
                       project.comparison_mac('Test Modes', 'FEM Modes'))
    assert np.allclose(drawn['frequencies'], measured.frequency)
    assert np.allclose(drawn['column_frequencies'], fem_shapes.frequency)
    assert drawn['title'] == 'x', 'the drawing keywords pass through'

    monkeypatch.undo()
    out = tmp_path / 'crossmac.png'
    project.plot_mac('Test Modes', 'FEM Modes', path=out, show=False)
    assert out.stat().st_size > 0, 'and it renders headless like every plot'


def test_the_singular_accessor_is_the_plain_object_beside_its_banded_one():
    """A worked-up random project holds the requirement on lines and on
    octave bands: `.specification` is the one on lines, as
    '@basis:Specification' is in a report; two on lines still refuse."""
    import numpy as np

    freq = np.linspace(10.0, 2000.0, 200)
    level = np.full((1, 200), 1e-3)
    spec = visualdynamics.Specification(
        freq, level, response_dof=['101Z+'],
        ordinate_dim=['acceleration**2/frequency'], ordinate_unit=['m/s**2'])
    psd = visualdynamics.Psd(freq, level, response_dof=['101Z+'],
                             ordinate_dim=['acceleration**2/frequency'],
                             ordinate_unit=['m/s**2'])
    project = visualdynamics.Project('t')
    project.add('Spec', spec)
    project.add('Octave Spec', spec.to_octave(6))
    project.add('PSD', psd)
    project.add('Octave PSD', psd.to_octave(6))
    assert project.specification is project['Spec']
    assert project.psd is project['PSD']
    assert len(project.specifications) == 2 and len(project.psds) == 2
    project.add('Spec 2', spec)
    with pytest.raises(AttributeError, match='3 in project'):
        _ = project.specification
