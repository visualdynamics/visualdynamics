"""Links: objects explicitly declared to belong together.

Explicit rather than implicit: a link says which geometry a shape set
belongs to, the tree draws the group joined by a bracket and floats it
to the top, and everything that used to guess an association now reads
the link instead.
"""

from __future__ import annotations

import numpy as np
from conftest import fixture_path

import visualdynamics
from visualdynamics import io
from visualdynamics.core.shapes import ShapeSet


def _select(window, *names):
    window.tree.clearSelection()
    for name in names:
        window._item_for_object(name).setSelected(True)


def _tree_names(window):
    from visualdynamics.gui.main_window import ROLE_REFERENCE

    out = []
    for i in range(window.test_item.childCount()):
        child = window.test_item.child(i)
        reference = child.data(0, ROLE_REFERENCE)
        if reference is not None and reference[0] == 'object':
            out.append(child.text(0))
    return out


def _airplane_pair(window, survey):
    shapes, frfs = survey
    geometry = visualdynamics.import_file(fixture_path('plate',
                                             'geometry.npz'))
    window.add_object('Geometry', geometry)
    window.add_object('FRF', frfs)
    window.add_object('Shapes', shapes)
    return geometry, shapes, frfs


def test_link_and_unlink_maintain_the_groups(window, pump, survey):
    _airplane_pair(window, survey)
    _select(window, 'Geometry', 'Shapes')
    window.link_selected()
    assert window.linked_group('Shapes') == ['Geometry', 'Shapes']
    assert window.link_role('Shapes') is None, 'no role until named'
    assert window.linked_geometry('Shapes')[0] == 'Geometry'
    # linking another member merges into the existing group
    _select(window, 'Shapes', 'FRF')
    window.link_selected()
    assert window.linked_group('FRF') == ['Geometry', 'Shapes', 'FRF']
    assert len(window.tree.link_spans) == 1
    _select(window, 'FRF')
    window.unlink_selected()
    assert window.linked_group('Shapes') == ['Geometry', 'Shapes']
    _select(window, 'Shapes')
    window.unlink_selected()
    assert window.links == [], 'one member left dissolves the group'


def test_linked_groups_float_to_the_top(window, pump, survey):
    _airplane_pair(window, survey)
    assert _tree_names(window) == ['Geometry', 'FRF', 'Shapes']
    _select(window, 'FRF', 'Shapes')
    window.link_selected()
    assert _tree_names(window) == ['FRF', 'Shapes', 'Geometry'], (
        'the linked group leads, members adjacent, unlinked after')


def test_a_link_holds_one_geometry_and_reports_rather_than_refuses(window,
                                                                   pump,
                                                                   survey):
    """Two geometries in one link is still refused: it is a statement
    about structure, not about nodes. Shapes naming nodes the geometry
    lacks are not — they link, and the indicator carries the news
    (Brandon, 2026-09-21)."""
    _airplane_pair(window, survey)
    second = visualdynamics.Geometry(node_id=[900], node_xyz=[[0.0, 0.0, 0.0]])
    window.add_object('Other Geometry', second)
    _select(window, 'Geometry', 'Other Geometry')
    window.link_selected()
    assert window.links == [], 'two geometries cannot share a link'
    assert 'one geometry' in window.statusBar().currentMessage()
    stranger = ShapeSet([1.0], [0.01], ['999X+'], np.ones((1, 1)))
    window.add_object('Stranger', stranger)
    _select(window, 'Geometry', 'Stranger')
    window.link_selected()
    pump()
    assert any('Stranger' in group['members'] for group in window.links), (
        'shapes naming nodes the geometry lacks link and are flagged')


def test_the_window_links_across_a_test_s_virtual_points(window, pump,
                                                        survey):
    """The same rule the API keeps: shapes naming a handful of places
    the model does not draw are this structure with extra points, and
    the window links them and leaves the indicator to say so."""
    _airplane_pair(window, survey)
    geometry = window.project['Geometry']
    on_the_model = str(geometry.node_id[0])
    partly = ShapeSet([1.0], [0.01], [f'{on_the_model}X+', '999X+'],
                      np.ones((1, 2)))
    window.add_object('Mostly Ours', partly)
    _select(window, 'Geometry', 'Mostly Ours')
    window.link_selected()
    pump()
    assert any('Mostly Ours' in group['members'] and 'Geometry' in group['members']
               for group in window.links), 'linked despite the virtual point'


def test_renames_and_deletions_keep_links_honest(window, pump, survey):
    _airplane_pair(window, survey)
    _select(window, 'Geometry', 'Shapes')
    window.link_selected()
    item = window._item_for_object('Shapes')
    item.setText(0, 'Truth Shapes')
    window._item_renamed(item, 0)
    assert window.linked_group('Geometry') == ['Geometry', 'Truth Shapes']
    assert window.linked_geometry('Truth Shapes') is not None
    _select(window, 'Truth Shapes')
    window.delete_selected()
    assert window.links == [], 'a deleted member dissolves its pair'


def test_links_ride_the_project_file(tmp_path, window, window_factory,
                                     pump, survey):
    _airplane_pair(window, survey)
    _select(window, 'Geometry', 'Shapes')
    window.link_selected()
    window.set_link_role('Shapes', 'Basis')
    path = tmp_path / 'linked.vdyn'
    io.save_test(str(path), 'Linked', dict(window.objects),
                 links=window.links)
    contents = io.load(str(path))
    assert contents.links == [
        {'members': ['Geometry', 'Shapes'], 'role': 'Basis'}]
    other = window_factory()
    other.import_paths([str(path)])
    assert other.linked_group('Shapes') == ['Geometry', 'Shapes']
    assert other.link_role('Shapes') == 'Basis', (
        'the role rides the project file too')
    assert _tree_names(other)[:2] == ['Geometry', 'Shapes']


def test_the_toolbar_offers_link_only_when_it_applies(window, pump,
                                                      survey):
    _airplane_pair(window, survey)
    _select(window, 'Geometry')
    window.render_current()
    pump()
    assert not window.link_action.isVisible(), 'one object links nothing'
    _select(window, 'Geometry', 'Shapes')
    window.render_current()
    pump()
    assert window.link_action.isVisible()
    assert not window.unlink_action.isVisible(), 'nothing linked yet'
    window.link_selected()
    window.render_current()
    pump()
    assert window.unlink_action.isVisible()


def test_the_basis_is_unique_and_paints_its_bracket(window, pump,
                                                    survey):
    """The Basis group reads by its bracket alone: blue and bold (for
    colorblind readers), no text; other groups take other colors. The
    role is unique, and the Basis group floats to the very top."""
    _airplane_pair(window, survey)
    _select(window, 'Geometry', 'Shapes')
    window.link_selected()
    window.set_link_role('Shapes', 'Basis')
    assert window.link_role('Geometry') == 'Basis'
    color, _items, bold = window.tree.link_spans[0]
    assert bold is True
    assert color == window.LINK_ROLE_COLORS['Basis']
    # the Basis is unique: taking it takes it from the other group
    second = visualdynamics.Geometry(node_id=[900], node_xyz=[[0.0, 0.0, 0.0]])
    window.add_object('Other Geometry', second)
    stranger = ShapeSet([1.0], [0.01], ['900X+'], np.ones((1, 1)))
    window.add_object('Other Shapes', stranger)
    _select(window, 'Other Geometry', 'Other Shapes')
    window.link_selected()
    color, _items, bold = window.tree.link_spans[1]
    assert bold is False
    assert color != window.LINK_ROLE_COLORS['Basis'], (
        'an unroled bracket never wears the Basis blue')
    window.set_link_role('Other Shapes', 'Basis')
    assert window.link_role('Other Shapes') == 'Basis'
    assert window.link_role('Shapes') is None, 'the role moved'
    pump()
    assert _tree_names(window)[:2] == ['Other Geometry', 'Other Shapes'], (
        'the Basis group reads first in the tree')


def test_the_bracket_right_click_offers_the_basis(window, pump,
                                                  survey):
    """The Basis toggle lives on the bracket itself: right-clicking the
    painted span opens the group's menu, not the object's."""
    from PySide6.QtCore import QPoint

    _airplane_pair(window, survey)
    _select(window, 'Geometry', 'Shapes')
    window.link_selected()
    pump()
    _color, items, _bold = window.tree.link_spans[0]
    middle = window.tree.visualItemRect(items[0]).center().y()
    span = window.tree.span_at(QPoint(5, middle))
    assert span == 0, 'the gutter click lands on the bracket'
    assert window.tree.span_at(QPoint(200, middle)) is None, (
        'clicks on the objects themselves stay theirs')
    # drive the menu's action directly: exec would block the test
    group = window.links[span]
    assert group['role'] is None
    window.set_link_role(group['members'][0], 'Basis')
    assert window.link_role('Shapes') == 'Basis'



def test_the_tree_keeps_the_canonical_type_order(window, pump, survey):
    """Types read in one order everywhere — geometry, measurements,
    shapes — however they arrived, and the same order holds inside a
    linked group."""
    shapes, frfs = survey
    window.add_object('Shapes', shapes)
    window.add_object('FRF', frfs)
    geometry = visualdynamics.import_file(fixture_path('plate',
                                             'geometry.npz'))
    window.add_object('Geometry', geometry)
    assert _tree_names(window) == ['Geometry', 'FRF', 'Shapes'], (
        'canonical order, not arrival order')
    _select(window, 'Shapes', 'Geometry')
    window.link_selected()
    assert _tree_names(window) == ['Geometry', 'Shapes', 'FRF'], (
        'the group leads and keeps the canonical order inside')


def test_a_runs_leftovers_join_its_group(window, pump):
    """One controller file is one measurement: a random run's
    system-ID FRF has no slot in the Random Vibration structure, but it
    belongs beside the rest of the run rather than outside the
    bracket."""
    window.import_paths([fixture_path('plate', 'random_spectra.nc4')])
    pump()
    assert window.link_role('FRF') == 'Basis', (
        'the FRF joined the side its file filled')
    others = [name for name in window.objects if name != 'FRF']
    assert all(window.link_role(name) == 'Basis' for name in others)


def test_a_saved_projects_unlinked_objects_stay_unlinked(tmp_path,
                                                         window,
                                                         window_factory,
                                                         pump):
    """A .vdyn brings its author's own arrangement: an object left out
    of every group was left out on purpose, and the kin rule is for
    runs, not for projects."""
    import visualdynamics

    window.import_paths([fixture_path('plate', 'random_spectra.nc4')])
    pump()
    window.project.unlink('FRF')
    path = str(tmp_path / 'arranged.vdyn')
    visualdynamics.io.save_test(path, 'Arranged', dict(window.objects),
                                project_type=window.project_type,
                                links=window.links)
    fresh = window_factory()
    fresh.import_paths([path])
    pump()
    assert fresh.link_role('FRF') is None, (
        'the author unlinked it, and it stays that way')


def test_computing_from_a_member_leaves_it_in_place(window, pump,
                                                    monkeypatch):
    """Deriving an object links it to its source, and the source keeps
    its place in its group: the tree keeps arrival order within a
    type, and a time history jumping down its group the moment FRFs
    were computed from it read as a deletion (Brandon, 2026-08-29 —
    `link` used to move a re-linked member to the back)."""
    from PySide6.QtWidgets import QInputDialog

    rng = np.random.default_rng(3)
    t = np.arange(4096) / 1024.0
    for k in (1, 2, 3):
        window.add_object(
            f'Pass {k}', visualdynamics.TimeHistory(
                t, rng.standard_normal((2, len(t))),
                response_dof=['101Z+', '9001X+'],
                ordinate_dim=['acceleration', 'force']))
    window.project.link('Pass 1', 'Pass 2', 'Pass 3', role='Basis')
    window._paint_links()
    _select(window, 'Pass 2')
    window.tree.setCurrentItem(window._item_for_object('Pass 2'))
    monkeypatch.setattr(QInputDialog, 'getItem',
                        staticmethod(lambda *a, **k: (a[3][0], True)))
    window.compute_frfs()
    pump()
    names = _tree_names(window)
    assert names[:3] == ['Pass 1', 'Pass 2', 'Pass 3'], names
    assert 'Pass 2 Hv FRFs' in names
    assert window.link_role('Pass 2 Hv FRFs') == 'Basis', \
        'the derived object joined its source\'s group'
