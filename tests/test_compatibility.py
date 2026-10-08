
import numpy as np
from conftest import fixture_path

import visualdynamics


def load(model, name):
    return visualdynamics.import_file(fixture_path(model, name))


def survey_test():
    return {
        'survey geometry': load('plate', 'geometry.npz'),
        'survey shapes': load('plate', 'shapes.npy'),
        'survey frfs': load('plate', 'frfs.npz'),
    }


def foreign_geometry():
    """A mesh from some other campaign entirely: node ids the plate
    never uses, so everything measured on the plate is missing here.
    Synthesized rather than shipped — the fixtures all describe one
    article now, and incompatibility needs a second one."""
    geometry = visualdynamics.Geometry(
        node_id=[9001, 9002, 9003, 9004],
        node_xyz=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        length_unit='m')
    geometry.add_beams([9001, 9002, 9003, 9004])
    return geometry


def foreign_frfs():
    return visualdynamics.Frf(
        abscissa=[1.0, 2.0], ordinate=[[1 + 1j, 2 - 1j]],
        response_dof='9001Z+', reference_dof='9002Z+',
        ordinate_dim='acceleration/force')


def test_matching_model_is_compatible():
    report = visualdynamics.check_compatibility(survey_test(), 'survey geometry')
    assert report.incompatible_names == []
    assert report.is_compatible('survey shapes')


def test_a_linked_geometry_answers_for_its_group():
    """A FEM shape set beside its own FEM mesh is consistent whichever
    geometry is active: the link, not the active geometry, says which
    structure an object describes."""
    objects = {
        'test geometry': load('plate', 'test_geometry.npz'),
        'FEM geometry': load('plate', 'geometry.npz'),
        'FEM shapes': load('plate', 'shapes.npy'),
    }
    object_groups = [{'members': ['FEM geometry', 'FEM shapes'], 'role': None}]
    report = visualdynamics.check_compatibility(objects, 'test geometry',
                                      object_groups=object_groups)
    assert report.is_compatible('FEM shapes'), (
        'judged against the linked FEM mesh, not the active test one')
    # unlinked, the active geometry judges — and flags it
    unlinked = visualdynamics.check_compatibility(objects, 'test geometry')
    assert not unlinked.is_compatible('FEM shapes')


def test_shapes_from_another_model_are_incompatible():
    objects = survey_test()
    objects['foreign geometry'] = foreign_geometry()
    report = visualdynamics.check_compatibility(objects, 'foreign geometry')
    issue = report.issue_for('survey shapes')
    assert issue is not None
    assert '1014 of 1014 shape DOFs' in issue.message
    assert 'foreign geometry' in issue.message
    assert '101X+' in issue.missing_dofs


def test_every_mode_is_flagged_together():
    """Modes share one DOF set, so a mismatch marks all of them."""
    objects = {'foreign geometry': foreign_geometry(),
               'survey shapes': load('plate', 'shapes.npy')}
    report = visualdynamics.check_compatibility(objects, 'foreign geometry')
    shapes = objects['survey shapes']
    issue = report.issue_for('survey shapes')
    assert issue.sub_items == list(range(shapes.num_shapes))
    assert report.sub_item_flagged('survey shapes', 0)
    assert report.sub_item_flagged('survey shapes', shapes.num_shapes - 1)


def test_geometry_is_never_flagged():
    objects = {'survey geometry': load('plate', 'geometry.npz'),
               'foreign geometry': foreign_geometry()}
    report = visualdynamics.check_compatibility(objects, 'survey geometry')
    assert report.incompatible_names == []


def test_no_geometry_flags_nothing():
    """Importing data before its geometry must not paint everything red."""
    objects = {'survey shapes': load('plate', 'shapes.npy')}
    report = visualdynamics.check_compatibility(objects)
    assert report.geometry_name is None
    assert report.incompatible_names == []


def test_only_the_offending_records_are_flagged():
    geometry = load('plate', 'geometry.npz')
    frfs = load('plate', 'frfs.npz')
    frfs.response_dof[2] = '9999Z+'          # a node the geometry lacks
    report = visualdynamics.check_compatibility(
        {'g': geometry, 'frfs': frfs}, 'g')
    issue = report.issue_for('frfs')
    assert issue.sub_items == [2]
    assert issue.missing_dofs == ['9999Z+']
    assert not report.sub_item_flagged('frfs', 0)
    assert report.sub_item_flagged('frfs', 2)


def test_reference_dof_is_checked_too():
    geometry = load('plate', 'geometry.npz')
    frfs = load('plate', 'frfs.npz')
    frfs.reference_dof[1] = '4242X+'
    report = visualdynamics.check_compatibility({'g': geometry, 'frfs': frfs}, 'g')
    assert report.issue_for('frfs').sub_items == [1]


def test_channel_table_channels_are_checked():
    geometry = load('plate', 'geometry.npz')
    table = visualdynamics.ChannelTable({
        'channel': [1, 2, 3],
        'node': ['101', '5150', '110'],
        'direction': ['Z+', 'Z+', 'Z+'],
        'unit': ['m/s^2'] * 3})
    report = visualdynamics.check_compatibility({'g': geometry, 'table': table}, 'g')
    issue = report.issue_for('table')
    assert issue.sub_items == [1]
    assert issue.missing_dofs == ['5150Z+']


def test_active_geometry_decides():
    objects = survey_test()
    objects['foreign geometry'] = foreign_geometry()
    objects['foreign frfs'] = foreign_frfs()

    survey_active = visualdynamics.check_compatibility(objects, 'survey geometry')
    assert survey_active.incompatible_names == ['foreign frfs']

    foreign_active = visualdynamics.check_compatibility(objects,
                                                        'foreign geometry')
    assert set(foreign_active.incompatible_names) == {'survey shapes',
                                                      'survey frfs'}


def test_unknown_active_geometry_falls_back():
    objects = survey_test()
    report = visualdynamics.check_compatibility(objects, 'no such geometry')
    assert report.geometry_name == 'survey geometry'


def test_message_summarizes_long_dof_lists():
    objects = {'foreign geometry': foreign_geometry(),
               'survey shapes': load('plate', 'shapes.npy')}
    message = visualdynamics.check_compatibility(
        objects, 'foreign geometry').issue_for('survey shapes').message
    assert 'and 1008 more' in message


def test_parse_dof_round_trips():
    from visualdynamics.core.data import dof_string, parse_dof

    for node, direction in [(101, 'X+'), (7, 'Z-'), (55, 'RY+')]:
        code = {'X+': 1, 'Z-': -3, 'RY+': 5}[direction]
        assert parse_dof(dof_string(node, code)) == (node, direction)


def test_geometry_membership_helpers():
    geometry = load('plate', 'geometry.npz')
    assert np.array_equal(geometry.contains_nodes([101, 999999]),
                          [True, False])
    assert geometry.missing_dofs(['101X+', '999999Z+']) == ['999999Z+']
