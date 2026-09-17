"""Selection is universal: the 3D view picks the way everything picks.

A plain click selects only what was clicked, the modifier extends, and
the modifier on something already picked takes it back out — for nodes,
coordinate systems, tracelines and elements alike, and for the
half-picked nodes of an element being built. One rule, four groups, and
`test_pick.py` covers only the geometry underneath it (which entity is
under a pixel), not the window's use of it.

Two details that are easy to lose and were covered only by the
end-to-end script: a pick made in the 3D view selects rows in the table
while the *focus* stays in the 3D pane, so the delete shortcut has to
reach that pane and the selection must not be drawn as an inactive
(grayed) one. Both were bugs.
"""

from __future__ import annotations

import pytest
from conftest import edit_category as edit
from conftest import fixture_path


@pytest.fixture
def plate(window, pump):
    window.import_paths([fixture_path('plate', 'geometry.exo')])
    pump()
    geometry = window.objects['Geometry']
    # the exodus plate is nodes and quads; give it lines to pick as well
    geometry.add_traceline([int(n) for n in geometry.node_id[:2]])
    geometry.add_traceline([int(n) for n in geometry.node_id[2:4]])
    return geometry


def rows(window):
    return sorted({index.row() for index
                   in window.table.selectionModel().selectedIndexes()})


def test_every_entity_kind_picks_by_the_same_three_rules(plate, window, pump):
    for label, picks in (
            ('Nodes', [int(n) for n in plate.node_id[:3]]),
            ('Coordinate systems', [int(c) for c in plate.cs_id[:1]]),
            ('Tracelines', [0, 1]),
            ('Elements', [0, 1])):
        edit(window, pump, label)
        window.select_entity(picks[0])
        assert rows(window) == [0], (label, rows(window))
        for pick in picks[1:]:
            window.select_entity(pick, extend=True)
        assert len(rows(window)) == len(picks), (label, rows(window))
        window.select_entity(picks[-1], extend=True)
        assert len(rows(window)) == len(picks) - 1, (
            f'{label}: the modifier on a picked one takes it back out')
        window.select_entity(picks[0])
        assert rows(window) == [0], f'{label}: a plain click starts over'


def test_the_nodes_of_a_half_built_element_pick_the_same_way(plate, window,
                                                             pump):
    """The rule holds for a selection that is not a table selection at
    all — the nodes gathered so far for an element that has not committed."""
    edit(window, pump, 'Elements')
    window.set_add_mode(True)
    window.element_type_actions['quad'].trigger()
    screen = window._projector.screen()[0]

    def click(row, extend=False):
        window.hover_at(*screen[row])
        window._add_at(*screen[row], extend=extend)

    click(0)
    click(1)
    assert window._picked_nodes == [int(plate.node_id[1])], (
        'a plain click replaces rather than adds')
    click(6, extend=True)
    click(5, extend=True)
    assert len(window._picked_nodes) == 3
    already = window._picked_nodes[1]
    click(6, extend=True)
    assert already not in window._picked_nodes, 'a second pick unpicks it'
    assert len(window._picked_nodes) == 2
    click(8)
    assert window._picked_nodes == [int(plate.node_id[8])], 'starting over'


def test_the_modifier_is_read_from_the_click_s_own_layer_too(plate, window,
                                                             pump):
    """A click arrives through VTK, which carries Shift and Ctrl but knows
    nothing of Command; Qt knows Command but only as of the last event it
    handled. Both layers get a say."""
    edit(window, pump, 'Elements')
    window.set_add_mode(True)
    assert not window._extend_pressed()
    interactor = getattr(window.scene.plotter, 'iren', None)
    if interactor is None:
        pytest.skip('no interactor on this offscreen plotter')
    interactor.interactor.SetShiftKey(1)
    try:
        assert window._extend_pressed(), 'a shift-click through VTK extends'
    finally:
        interactor.interactor.SetShiftKey(0)


def test_delete_reaches_the_pane_that_holds_the_focus(plate, window, pump):
    """Picking in the 3D view selects table rows while the focus stays in
    the 3D pane, so Delete has to be an action *of* that pane — and of
    its children, since the pane's focus lives in a child widget."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence

    assert window.delete_action in window.scene.actions()
    assert (window.delete_action.shortcutContext()
            == Qt.ShortcutContext.WidgetWithChildrenShortcut)
    assert QKeySequence(Qt.Key.Key_Backspace) in window.delete_action.shortcuts()


def test_a_selection_made_from_the_view_is_not_drawn_grayed_out():
    """Qt draws a selection in a widget without focus as 'inactive',
    which for rows selected by clicking the model reads as disabled."""
    from PySide6.QtGui import QColor, QPalette

    from visualdynamics.gui.main_window import _keep_selection_vivid
    from visualdynamics.gui.tables import CopyPasteTableView

    probe = CopyPasteTableView()
    palette = probe.palette()
    palette.setColor(QPalette.ColorGroup.Active,
                     QPalette.ColorRole.Highlight, QColor('#0640c6'))
    palette.setColor(QPalette.ColorGroup.Inactive,
                     QPalette.ColorRole.Highlight, QColor('#314f78'))
    probe.setPalette(palette)
    _keep_selection_vivid(probe)
    assert (probe.palette().color(QPalette.ColorGroup.Inactive,
                                  QPalette.ColorRole.Highlight)
            == QColor('#0640c6'))
