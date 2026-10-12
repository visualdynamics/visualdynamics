"""Solids in the app (Brandon, 2026-10-11): the Solids row under every
geometry, Add Solid Block and Add Cylinder in the add pane, the
booleans on picked parts, Take Apart, Delete, and the parts drawn
see-through in the view.

The geometry's own verbs are pinned in `test_solids.py`; here, that the
window reaches them as a person would and shows what they did.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import select_objects

from visualdynamics.core import solids

pytest.importorskip('OCP')

INCH = 0.0254


def _rows(window, kind):
    """The tree rows whose reference is of `kind`, in tree order."""
    found = []

    def walk(item):
        for i in range(item.childCount()):
            child = item.child(i)
            reference = child.data(0, 257)       # ROLE_REFERENCE
            if reference is not None and reference[0] == kind:
                found.append(child)
            walk(child)

    walk(window.tree.invisibleRootItem())
    return found


def _pick(window, pump, rows):
    window.tree.clearSelection()
    window.tree.setCurrentItem(rows[0])
    for row in rows[1:]:
        row.setSelected(True)
    pump()


def _acts(window):
    return {act[0]: act for act in window.acts_for()}


def _part(window, pump):
    window.unit_combo.setCurrentText('in-slinch-lbf-s')
    window.project.new_geometry('Part', unit='in')
    window.show_object('Part')
    select_objects(window, pump, 'Part')
    return window.objects['Part']


def _add(window, pump, verb, **values):
    _pick(window, pump, _rows(window, 'solids'))
    _acts(window)[verb][3]()
    pump()
    panel = window.scene.mesh_panel
    panel.set_values(**values)
    pump()
    return panel


def test_the_solids_row_adds_blocks_and_cylinders_in_the_pane(window, pump):
    geometry = _part(window, pump)
    (holder,) = _rows(window, 'solids')
    assert holder.text(0) == 'Solids (0)', 'there before the first solid'
    _pick(window, pump, [holder])
    assert list(_acts(window)) == ['add_solid_block', 'add_solid_cylinder']
    panel = _add(window, pump, 'add_solid_block', center=(0, 0, -0.25),
                 widths=(4, 4, 0.5), group='plate')
    assert panel.title.text() == 'Add Solid Block'
    assert panel.group_label.text() == 'Name'
    assert panel.size.isHidden() and panel.radius.isHidden()
    assert panel.reading_label.text() == (
        'A solid block 4 by 4 by 0.5 in, 8 in³ — a shape, meshed later.')
    assert 'plane-preview' in window.scene.plotter.actors
    panel.add_button.click()
    pump()
    assert window.project.journal[-1] == (
        "project.add_solid_block('Part', (0.0, 0.0, -0.25), (4.0, 4.0, 0.5), "
        "(0.0, 0.0, 0.0), 'plate', unit='in')")
    panel = _add(window, pump, 'add_solid_cylinder', center=(1, 0, -0.25),
                 radius=0.25, length=1.0)
    assert panel.title.text() == 'Add Cylinder'
    assert panel.widths[0].isHidden() and not panel.radius.isHidden()
    assert panel.reading_label.text().startswith(
        'A cylinder 0.5 in across and 1 in long, 0.1963 in³')
    panel.add_button.click()
    pump()
    plate, cylinder = geometry.solids
    assert plate.name == 'plate' and cylinder.kind == 'cylinder'
    assert np.allclose(cylinder.center, np.array([1, 0, -0.25]) * INCH)
    assert cylinder.radius == pytest.approx(0.25 * INCH)
    assert [row.text(0) for row in _rows(window, 'solid')] == ['plate', 'Cylinder 2']
    panel.set_values(radius=0.0)
    pump()
    assert 'a radius and a length' in panel.reading_label.text()
    assert not panel.add_button.isEnabled()


def _plate_with_two_cylinders(window, pump):
    geometry = _part(window, pump)
    window.project.add_solid_block('Part', (0, 0, -0.25), (4, 4, 0.5), name='plate',
                                   unit='in')
    for x in (-1, 1):
        window.project.add_solid_cylinder('Part', (x, 0, -0.25), 0.25, 1.0, unit='in')
    window._solids_changed('Part')
    pump()
    return geometry


def test_picked_parts_subtract_from_the_first_and_draw(window, pump):
    """Three parts picked offer the booleans; Subtract names its body,
    the first in the tree, takes the cylinders from it, and the new
    solid is picked and drawn in the picked color, its cylinders listed
    under it."""
    geometry = _plate_with_two_cylinders(window, pump)
    _pick(window, pump, _rows(window, 'solid'))
    acts = _acts(window)
    assert list(acts) == ['union_solids', 'subtract_solids', 'intersect_solids',
                          'pattern_solids', 'mesh_solids']
    assert acts['subtract_solids'][4] == 'Take the others from plate, the first in the tree'
    acts['subtract_solids'][3]()
    pump()
    (part,) = solids.top_level(geometry)
    assert part.kind == 'subtract' and part.operands == (1, 2, 3)
    assert solids.volume(geometry, part.id) / INCH ** 3 == pytest.approx(
        8 - 2 * np.pi * 0.25 ** 2 * 0.5)
    rows = _rows(window, 'solid')
    # the subtraction takes its first solid's name (2026-10-11)
    assert [row.text(0) for row in rows] == ['plate', 'plate',
                                             'Cylinder 2', 'Cylinder 3']
    assert [row.isSelected() for row in rows] == [True, False, False, False]
    actors = window.scene.plotter.actors
    assert 'solid-4' in actors and 'solid-edges-4' in actors
    assert 'solid-1' not in actors, 'a part is drawn whole, not its operands'
    assert actors['solid-4'].GetProperty().GetOpacity() == pytest.approx(0.6)
    assert list(_acts(window)) == ['pattern_solids', 'mesh_solids',
                                   'separate_solids']
    # a cylinder picked inside it is drawn over it
    _pick(window, pump, [rows[2]])
    assert 'solid-2' in window.scene.plotter.actors
    assert not _acts(window), 'nothing to combine a solid inside another with'


def test_take_apart_and_delete(window, pump):
    geometry = _plate_with_two_cylinders(window, pump)
    _pick(window, pump, _rows(window, 'solid'))
    _acts(window)['subtract_solids'][3]()
    pump()
    _acts(window)['separate_solids'][3]()
    pump()
    assert [s.id for s in solids.top_level(geometry)] == [1, 2, 3]
    assert [row.isSelected() for row in _rows(window, 'solid')] == [True] * 3
    assert window.project.journal[-1] == "project.separate_solids('Part', [4])"
    # Delete takes a part whole; one inside another is refused
    _pick(window, pump, _rows(window, 'solid')[:2])
    _acts(window)['union_solids'][3]()
    pump()
    rows = _rows(window, 'solid')
    # the union, named for the plate, and the plate inside it
    assert [row.text(0) for row in rows] == ['Cylinder 3', 'plate', 'plate',
                                             'Cylinder 2']
    _pick(window, pump, [rows[2]])
    window.delete_selected()
    pump()
    assert 'take that apart first' in window.statusBar().currentMessage()
    _pick(window, pump, [_rows(window, 'solid')[1]])
    window.delete_selected()
    pump()
    assert [s.id for s in geometry.solids] == [3]
    assert 'deleted 3 solids' in window.statusBar().currentMessage()


def test_pattern_previews_the_copies_and_makes_them(window, pump):
    geometry = _part(window, pump)
    window.project.add_solid_cylinder('Part', (-7.75, -2.75, -0.25), 0.1451, 1.0,
                                      name='hole', unit='in')
    window._solids_changed('Part')
    pump()
    _pick(window, pump, _rows(window, 'solid'))
    _acts(window)['pattern_solids'][3]()
    pump()
    panel = window.scene.pattern_panel
    assert panel.isVisibleTo(window) and panel.spacing_label.text() == 'Spacing [in]'
    assert not panel.pattern_button.isEnabled(), 'one of each makes nothing'
    panel.set_values(counts=(13, 1, 1))
    pump()
    assert panel.reading_label.text() == 'More than one along X needs a spacing there.'
    panel.set_values(spacing=(15.5 / 12, 0, 0))
    pump()
    assert panel.reading_label.text() == '12 copies — 12 of each solid patterned.'
    assert 'plane-preview' in window.scene.plotter.actors
    panel.pattern_button.click()
    pump()
    assert not panel.isVisibleTo(window)
    # one solid, the hole under it (Brandon, 2026-10-11), picked to
    # combine next
    (made,) = solids.top_level(geometry)
    assert made.kind == 'pattern' and made.operands == (1,)
    rows = _rows(window, 'solid')
    assert [row.text(0) for row in rows] == ['hole', 'hole']
    assert rows[1].parent() is rows[0], 'the hole under its pattern'
    assert [row.isSelected() for row in rows] == [True, False]
    assert 'Take Apart' in [act[1] for act in window.acts_for()]


def test_a_patterns_pencil_changes_its_counts_and_the_originals_changes_every_copy(
        window, pump):
    """The pattern's pencil opens Pattern on its counts and spacing, to
    change them; the hole's pencil, under it, changes every copy."""
    geometry = _part(window, pump)
    hole = window.project.add_solid_cylinder('Part', (0, 0, -0.25), 0.1, 1.0,
                                             name='hole', unit='in')
    made = window.project.pattern_solids('Part', [hole], (3, 1, 1), (1, 0, 0),
                                         unit='in')
    window._solids_changed('Part')
    pump()
    pattern_row, hole_row = _rows(window, 'solid')
    assert pattern_row.toolTip(1).startswith('Change how many copies')
    window._tree_item_clicked(pattern_row, 1)
    pump()
    panel = window.scene.pattern_panel
    assert panel.isVisibleTo(window) and panel.title.text() == 'Edit Pattern'
    assert panel.pattern_button.text() == 'Apply'
    assert panel.values()['counts'] == (3, 1, 1)
    assert panel.values()['spacing'] == pytest.approx((1, 0, 0))
    assert not panel.pattern_button.isEnabled(), 'nothing to apply yet'
    panel.set_values(counts=(5, 1, 1))
    pump()
    panel.pattern_button.click()
    pump()
    assert window.project.journal[-1] == (
        f"project.edit_solid('Part', {made}, counts=(5, 1, 1), unit='in')")
    assert solids.solid(geometry, made).counts == (5, 1, 1)
    assert not panel.isVisibleTo(window)
    # the hole's own pencil: every copy larger
    hole_row = _rows(window, 'solid')[1]
    window._tree_item_clicked(hole_row, 1)
    pump()
    window.scene.mesh_panel.set_values(radius=0.2)
    pump()
    window.scene.mesh_panel.add_button.click()
    pump()
    radii = [leaf.radius for leaf in solids._leaves(geometry, made)]
    assert radii == pytest.approx([0.2 * INCH] * 5)
    assert len(solids._leaves(geometry, made)) == 5


def test_a_solid_block_takes_a_corner_radius_in_the_pane(window, pump):
    geometry = _part(window, pump)
    panel = _add(window, pump, 'add_solid_block', center=(0, 0, 0),
                 widths=(3.375, 5, 0.5), corner=0.138)
    assert not panel.corner.isHidden()
    assert panel.corner_label.text() == 'Corner radius [in]'
    assert panel.reading_label.text().startswith(
        'A solid block 3.375 by 5 by 0.5 in, its corners rounded to 0.138 in, ')
    panel.add_button.click()
    pump()
    assert window.project.journal[-1].endswith("corner=0.138)")
    assert geometry.solids[0].corner == pytest.approx(0.138 * INCH)
    panel.set_values(corner=3.0)
    pump()
    assert panel.reading_label.text().startswith('A corner radius from zero to half')
    panel = _add(window, pump, 'add_solid_cylinder')
    assert panel.corner.isHidden(), 'a cylinder has no corners'


def _pencil(window, pump, row):
    """Click a row's pencil, the tree's icon column."""
    window._tree_item_clicked(row, 1)
    pump()


def test_a_solids_pencil_opens_the_pane_on_it_and_apply_changes_it(window, pump):
    """Brandon (2026-10-11): a solid moved, sized or reshaped after it is
    made. Its row's pencil opens the add pane on its numbers, titled to
    edit and with Apply; Apply waits for a change, then journals only
    what moved; the pane stays on it, its pencil drawn on, and a second
    click on the pencil puts it away."""
    geometry = _plate_with_two_cylinders(window, pump)
    plate_row = _rows(window, 'solid')[0]
    assert plate_row.toolTip(1) == 'Move it, turn it or change its size'
    _pencil(window, pump, plate_row)
    panel = window.scene.mesh_panel
    assert panel.isVisibleTo(window) and panel.title.text() == 'Edit Solid Block'
    assert panel.add_button.text() == 'Apply'
    values = panel.values()
    assert values['center'] == pytest.approx((0, 0, -0.25))
    assert values['widths'] == pytest.approx((4, 4, 0.5)) and values['group'] == 'plate'
    assert not panel.add_button.isEnabled(), 'nothing to apply yet'
    assert 'As it is' in panel.reading_label.text()
    assert 'plane-preview' in window.scene.plotter.actors
    panel.set_values(widths=(5, 4, 0.5))
    pump()
    assert panel.add_button.isEnabled()
    panel.add_button.click()
    pump()
    assert window.project.journal[-1] == (
        "project.edit_solid('Part', 1, widths=(5.0, 4.0, 0.5), unit='in')")
    assert geometry.solids[0].widths[0] == pytest.approx(5 * INCH)
    assert panel.isVisibleTo(window) and not panel.add_button.isEnabled()
    from visualdynamics.gui.main_window import ROLE_PENCIL_ON

    (row,) = [r for r in _rows(window, 'solid') if r.data(0, 257)[2] == 1]
    assert row.isSelected() and row.data(1, ROLE_PENCIL_ON) is True, 'its pencil drawn on'
    _pencil(window, pump, row)
    assert not panel.isVisibleTo(window)
    assert row.data(1, ROLE_PENCIL_ON) is False


def test_a_cylinders_pencil_changes_its_radius_and_name(window, pump):
    geometry = _plate_with_two_cylinders(window, pump)
    _pencil(window, pump, _rows(window, 'solid')[1])
    panel = window.scene.mesh_panel
    assert panel.title.text() == 'Edit Cylinder'
    # its own row picked keeps the pane on it; another solid's puts it away
    _pick(window, pump, [_rows(window, 'solid')[1]])
    assert panel.isVisibleTo(window)
    _pick(window, pump, [_rows(window, 'solid')[2]])
    assert not panel.isVisibleTo(window)
    _pencil(window, pump, _rows(window, 'solid')[1])
    assert panel.values()['radius'] == pytest.approx(0.25)
    panel.set_values(radius=0.375, group='bolt hole')
    pump()
    panel.add_button.click()
    pump()
    assert window.project.journal[-1] == (
        "project.edit_solid('Part', 2, radius=0.375, name='bolt hole', unit='in')")
    assert geometry.solids[1].radius == pytest.approx(0.375 * INCH)
    assert [row.text(0) for row in _rows(window, 'solid')][1] == 'bolt hole'


def test_a_parts_pencil_moves_and_turns_it_whole(window, pump):
    """A plate less two holes: the boolean's pane is Move Part, centered
    on its extent, its fields the slide and the turn; Apply moves every
    solid in it, the holes going round with the plate."""
    geometry = _plate_with_two_cylinders(window, pump)
    _pick(window, pump, _rows(window, 'solid'))
    _acts(window)['subtract_solids'][3]()
    pump()
    part_row = _rows(window, 'solid')[0]
    assert part_row.toolTip(1).startswith('Move or turn it whole')
    _pencil(window, pump, part_row)
    panel = window.scene.mesh_panel
    assert panel.title.text() == 'Move Part' and panel.add_button.text() == 'Apply'
    assert panel.widths[0].isHidden() and panel.radius.isHidden()
    assert panel.values()['center'] == pytest.approx((0, 0, -0.25))
    assert not panel.add_button.isEnabled()
    panel.set_values(center=(10, 0, -0.25), angles=(0, 0, 90))
    pump()
    assert panel.reading_label.text() == (
        'Moved (10, 0, 0) in and turned (0, 0, 90)°, the 3 solids it is made of '
        'with it.')
    panel.add_button.click()
    pump()
    assert window.project.journal[-1] == (
        "project.move_solids('Part', [4], (10.0, 0.0, 0.0), turn=(0.0, 0.0, 90.0), "
        "unit='in')")
    centers = [np.asarray(s.center) / INCH for s in geometry.solids[:3]]
    assert np.allclose(centers, [(10, 0, -0.25), (10, -1, -0.25), (10, 1, -0.25)],
                       atol=1e-9)
    assert panel.values()['angles'] == pytest.approx((0, 0, 0)), 'the turn is done'
