"""Every act lives on the bar, and nowhere else.

An act with no settings — Integrate, Transform, Merge, Generate Report,
Recompute — is a push button on the bar of the pane showing what it
changes, labeled, in its own fenced group; a reading with settings
keeps its toggle and its pane. The bar follows the selection: one
object's own acts, or the acts a combination can take and nothing that
applies to one of its members alone. The tree's column carries the
refresh badge only, and the right-click no computation (Brandon,
2026-09-04: the three homes an act used to have confused a user, and
one rule replaces them).
"""

from __future__ import annotations

import numpy as np
from conftest import select_objects as _select
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton

import visualdynamics
from visualdynamics.core.rigid import MassProperties, rigid_body_shapes
from visualdynamics.project import _SELECTION_APPLIES, _VERB_APPLIES

T = np.arange(4096) / 2048.0

#: the verbs whose act is the apply button of a reading's pane — the
#: four parts of principle 13, so not a bar button of their own
PANE_VERBS = {'filter_data', 'truncate_data', 'detect_shocks',
              'compute_spectra', 'compute_psds', 'compute_cpsds',
              'compute_srs', 'compute_frfs', 'compute_multiple_coherence',
              'compute_octave', 'generate_rigid_body_modes',
              'author_specification'}
#: the comparison table's own act: two sets are matched where they
#: are compared, pair by pair
TABLE_VERBS = {'match_modes'}


def _populate(window, pump):
    rng = np.random.default_rng(1)
    geometry = visualdynamics.Geometry(np.arange(101, 106),
                                       rng.uniform(-1.0, 1.0, (5, 3)),
                                       length_unit='m')
    rigid = rigid_body_shapes(geometry, MassProperties((0, 0, 0)))
    run = visualdynamics.TimeHistory(
        T, rigid.shape_matrix.T @ rng.standard_normal((6, len(T))),
        response_dof=list(rigid.coordinate), ordinate_dim='acceleration')
    window.add_object('Plate', geometry)
    window.add_object('Modes', rigid)
    window.project.link('Plate', 'Modes')
    window.add_object('Run', run)
    pump()
    return run


def _bar(pane):
    """The act buttons showing on a pane's bar, by label."""
    actions = pane.__dict__.get('_acts', {}).get('actions', {}).values()
    return [a.text() for a in pane.toolbar.actions()
            if a in actions and a.isVisible()]


def test_a_lone_object_offers_its_own_acts_on_the_plots_bar(window, pump):
    _populate(window, pump)
    _select(window, pump, 'Run')
    assert _bar(window.data_pane) == ['Integrate', 'Copy']
    assert _bar(window.scene) == []
    assert window._item_for_object('Run').icon(1).isNull(), \
        'no calculator in the tree'
    button = window.data_pane.toolbar.widgetForAction(
        window.data_pane._acts['actions']['integrate'])
    assert isinstance(button, QToolButton)
    assert button.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly, \
        'an icon, like every bar button; the words are in the tooltip ' \
        '(Brandon, 2026-09-12)'
    assert button.toolTip().startswith('Integrate — '), \
        'the verb leads the tooltip, since the icon is all that shows'
    _select(window, pump, 'Modes')
    # no verb applies to a shape set alone, but what is drawn copies —
    # on whichever bar is up
    assert _bar(window.data_pane) + _bar(window.scene) == ['Copy']


def test_a_combination_offers_only_what_it_can_take_together(window, pump):
    _populate(window, pump)
    _select(window, pump, 'Run', 'Modes')
    assert _bar(window.data_pane) == ['Transform to Modal Responses', 'Copy'], \
        'not Integrate: that applies to one member alone'
    window.data_pane._acts['actions']['transform'].trigger()
    pump()
    assert 'Run Modal Responses' in window.project, \
        'the button runs the same verb the API has'
    _select(window, pump, 'Run Modal Responses', 'Modes')
    assert _bar(window.data_pane) == ['Expand to Physical Responses', 'Copy']


def test_siblings_of_one_type_offer_merge(window, pump):
    _populate(window, pump)
    other = visualdynamics.TimeHistory(
        T, np.zeros((2, len(T))), response_dof=['201X+', '202X+'],
        ordinate_dim='acceleration')
    window.add_object('Run 2', other)
    _select(window, pump, 'Run', 'Run 2')
    assert _bar(window.data_pane) == ['Merge into One', 'Copy']
    window.merge_selected()
    pump()
    assert 'Run' not in window.project and 'Run 2' not in window.project


def test_the_project_row_offers_its_report(window, pump, monkeypatch):
    _populate(window, pump)
    window.tree.clearSelection()
    window.test_item.setSelected(True)
    window.tree.setCurrentItem(window.test_item)
    pump()
    assert 'Generate Report' in _bar(window.data_pane) + _bar(window.scene)
    assert window.test_item.icon(1).isNull(), 'the column carries no act'
    # the row alone shows nothing on the right: the act was landing a
    # third of the way down the window on the time data's bar under
    # everything the project holds drawn at once (Brandon, 2026-09-09)
    assert not window.data_pane.isVisibleTo(window), 'no plot for the project row'
    assert not window.table.isVisibleTo(window)
    # and the project's **+**, somewhere to build a model (2026-09-26)
    assert _bar(window.scene) == ['Generate Report', 'New Geometry'], \
        "the 3-D view's bar, holding the space"
    assert window.scene.isVisibleTo(window)
    assert 'Generate Report is on the bar' in window.statusBar().currentMessage()
    popped = []
    monkeypatch.setattr(window, '_pop_menu', lambda menu: popped.append(menu))
    window.generate_report_act()
    # seven: the combined random-and-sine template retired 2026-10-05
    assert popped and len(popped[0].actions()) == 7, \
        'untyped: the choice of templates, one per built-in and Empty'
    window.set_project_type('Modal Test')
    pump()
    window.generate_report_act()
    pump()
    from visualdynamics.core.report import Report

    assert any(isinstance(obj, Report) for obj in window.project.values()), \
        'typed: one click, one report'


def test_a_stale_object_offers_recompute_first(window, pump):
    from dataclasses import replace

    run = _populate(window, pump)
    run.averaging = run.suggest_averaging()
    name = window.project.compute_psds('Run')
    window.show_object(name)
    pump()
    run.averaging = replace(run.averaging, frames=max(1, run.averaging.frames - 1))
    window._refresh_stale_badges()
    _select(window, pump, name)
    assert _bar(window.data_pane)[0] == 'Recompute'
    assert not window._item_for_object(name).icon(1).isNull(), \
        'the badge stays in the tree: it is status, and the tree is '\
        'where the objects are'
    window.data_pane._acts['actions']['refresh'].trigger()
    pump()
    assert window._stale == {}
    assert 'Recompute' not in _bar(window.data_pane)


def test_the_right_click_carries_no_act(window, pump, qt_app):
    from PySide6.QtCore import QPoint, QTimer
    from PySide6.QtWidgets import QMenu

    _populate(window, pump)
    other = visualdynamics.TimeHistory(
        T, np.zeros((2, len(T))), response_dof=['201X+', '202X+'],
        ordinate_dim='acceleration')
    window.add_object('Run 2', other)
    _select(window, pump, 'Run', 'Run 2')
    item = window._item_for_object('Run')
    window.tree.scrollToItem(item)
    qt_app.processEvents()
    rect = window.tree.visualItemRect(item)
    position = QPoint(rect.center().x(), rect.center().y())
    shown = []

    def grab():
        for widget in qt_app.topLevelWidgets():
            if isinstance(widget, QMenu) and widget.isVisible():
                shown.extend(a.text() for a in widget.actions() if a.text())
                widget.close()

    QTimer.singleShot(0, grab)
    window._show_tree_menu(position)
    qt_app.processEvents()
    assert shown, 'the menu still opens for what is not an act'
    assert not any('Merge' in e or 'Transform' in e or 'Project onto' in e
                   for e in shown), shown


def test_every_verb_has_a_home_on_a_bar_or_a_pane(window):
    """Nothing is reachable only by menu: every processing verb is a
    bar act, a reading's apply button, or the comparison table's own."""
    acts = {verb for verb, *_rest in window.ACTS}
    single = {verb for verb, _applies in _VERB_APPLIES}
    paired = {verb for verb, _applies in _SELECTION_APPLIES}
    assert single | paired == acts | PANE_VERBS | TABLE_VERBS, \
        (single | paired) ^ (acts | PANE_VERBS | TABLE_VERBS)
    assert not hasattr(window, 'CALCULATOR_MENU')
    assert not hasattr(window, '_computations_for')


def test_no_two_acts_share_an_icon(qt_app):
    """An icon says what a button does. Merge Coincident Nodes first wore
    Merge into One's (2026-09-26): which one showed depended on the
    selection, and a person who knew the icon expected the wrong act.
    Every act draws a glyph of its own, and none is the fallback."""
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice

    from visualdynamics.gui.icons import control_icon
    from visualdynamics.gui.main_window import MainWindow

    def pixels(name):
        data = QByteArray()
        buffer = QBuffer(data)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        control_icon(name).pixmap(64, 64).toImage().save(buffer, 'PNG')
        return bytes(data)

    icons = [icon for _verb, _label, icon, _handler in MainWindow.ACTS]
    assert len(icons) == len(set(icons)), sorted(
        icon for icon in set(icons) if icons.count(icon) > 1)
    fallback = pixels('no such glyph')
    drawn = {icon: pixels(icon) for icon in icons}
    assert all(image != fallback for image in drawn.values()), [
        icon for icon, image in drawn.items() if image == fallback]
    assert len(set(drawn.values())) == len(drawn), 'two names, one picture'


def test_no_act_wears_an_edit_bar_buttons_glyph():
    """Solve Modes wore the quadrilateral element's square, and read as
    nothing but a square (Brandon, 2026-09-27). An act has a glyph of its
    own, not one of the edit bar's element types or modes."""
    from visualdynamics.gui.main_window import MainWindow

    # 'edit', the pencil, is left out: Fit Modal Model wears it still,
    # which is Brandon's call (STATUS.md, 2026-09-27)
    borrowed = {'beam', 'tri', 'quad', 'add', 'rotate', 'reset', 'tie'}
    icons = {verb: icon for verb, _label, icon, _handler in MainWindow.ACTS}
    assert not borrowed & set(icons.values()), icons
    assert icons['solve_modes'] == 'solve_modes'


def test_merge_blocks_has_a_glyph_of_its_own(qt_app):
    """Merge Blocks is offered for a block selection, off the ACTS list,
    so the uniqueness test above does not see it: its glyph is drawn and
    is neither Merge into One's nor Merge Coincident Nodes'."""
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice

    from visualdynamics.gui.icons import control_icon

    def pixels(name):
        data = QByteArray()
        buffer = QBuffer(data)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        control_icon(name).pixmap(64, 64).toImage().save(buffer, 'PNG')
        return bytes(data)

    drawn = {name: pixels(name) for name in
             ('merge_groups', 'merge', 'merge_nodes', 'no such glyph')}
    assert len(set(drawn.values())) == 4
