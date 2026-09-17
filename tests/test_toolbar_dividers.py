"""Dividers that mean something, on every bar.

A control bar holds two kinds of thing and used to run them together:
*choices*, where one of a set is in force and picking one drops the
last, and *settings*, each independent of the rest. The 2-D/3-D toggle
is the clearest case of the second kind — it is not a way of reading a
record, it is how any reading is drawn — and it sat in an undivided row
with four things that do turn each other off (Brandon, 2026-08-27).

`gui/toolbars.fence` puts a divider around every group at build time and
`tidy` decides at show time which of them still divide anything, so the
rule is one implementation rather than a habit each bar has to keep.

The collapsing case is the one that was wrong first and is easy to get
wrong again: two groups fenced side by side put two separators
together, and hiding *both* — which the first version did — runs the
groups into each other, which is the whole thing this exists to stop.
"""

from __future__ import annotations

import pytest
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QToolBar

from visualdynamics.gui.toolbars import fence, shows_anything, tidy


@pytest.fixture
def bar(qt_app):
    toolbar = QToolBar()
    actions = []
    for name in ('one', 'two', 'three', 'four'):
        action = QAction(name)
        toolbar.addAction(action)
        actions.append(action)
    return toolbar, actions


def shape(toolbar):
    """The bar as a reader sees it: '|' for a visible divider."""
    out = []
    for action in toolbar.actions():
        if action.isSeparator():
            if action.isVisible():
                out.append('|')
        elif action.isVisible():
            out.append(action.text())
    return out


def test_a_group_is_fenced_on_both_sides(bar):
    toolbar, actions = bar
    fence(toolbar, [actions[1:3]])
    tidy(toolbar)
    assert shape(toolbar) == ['one', '|', 'two', 'three', '|', 'four']


def test_two_groups_side_by_side_keep_one_divider_between_them(bar):
    """The bug in the first version: both separators were dropped and
    the groups ran together."""
    toolbar, actions = bar
    fence(toolbar, [actions[0:2], actions[2:4]])
    tidy(toolbar)
    assert shape(toolbar) == ['one', 'two', '|', 'three', 'four']


def test_a_divider_at_either_end_is_dropped(bar):
    """A fence needs a field on both sides."""
    toolbar, actions = bar
    fence(toolbar, [actions[0:2]])
    tidy(toolbar)
    assert shape(toolbar)[0] != '|'
    fence(toolbar, [actions[2:4]])
    tidy(toolbar)
    assert shape(toolbar)[-1] != '|'


def test_a_group_hidden_away_leaves_no_fence_around_nothing(bar):
    """The bar is built with everything on it and hidden down to what
    the selection can use, so most groups are absent most of the time.
    A fence around an empty field would read as a group gone missing.

    Its two dividers collapse to one rather than to none — the same
    collapsing rule as two groups meeting. What is left is a single
    divider between the neighbors, not a gap where a group used to be;
    on the real bar those neighbors are themselves different groups,
    so it is doing an honest job.
    """
    toolbar, actions = bar
    fence(toolbar, [actions[1:3]])
    for action in actions[1:3]:
        action.setVisible(False)
    tidy(toolbar)
    assert shape(toolbar) == ['one', '|', 'four']
    assert shape(toolbar).count('|') == 1, 'no empty compartment'


def test_a_bar_of_nothing_but_dividers_is_not_showing_anything(bar):
    """Separators are visible whether or not anything is around them, so
    a plain any(isVisible) kept an empty bar up — which is how the first
    fenced group broke `show_controls`."""
    toolbar, actions = bar
    fence(toolbar, [actions[1:3]])
    for action in actions:
        action.setVisible(False)
    tidy(toolbar)
    assert not shows_anything(toolbar)
    assert shape(toolbar) == []


def test_a_lone_setting_can_be_fenced_by_itself(bar):
    """Which is what the 2-D/3-D toggle is: fenced not because it is a
    group but because it is *not* one of the choices beside it."""
    toolbar, actions = bar
    fence(toolbar, [[actions[2]]])
    tidy(toolbar)
    assert shape(toolbar) == ['one', 'two', '|', 'three', '|', 'four']


# ---- and the real bar ---------------------------------------------------


@pytest.fixture
def pane(qt_app):
    from visualdynamics.gui.panes import DataPane

    pane = DataPane('light')
    for action in pane.toolbar.actions():
        if not action.isSeparator():
            action.setVisible(True)
    tidy(pane.toolbar)
    return pane


def test_the_3d_toggle_stands_apart_from_the_choices(pane):
    """Brandon's ask, on the real bar: fenced off from the choices,
    because it applies to whichever choice is in force. It leads the
    bar now, so the left edge is its opening fence — the same
    allowance every group at the bar's ends gets."""
    visible = [a for a in pane.toolbar.actions() if a.isVisible()]
    at = visible.index(pane.waterfall_action)
    assert at == 0 or visible[at - 1].isSeparator(), 'fenced before'
    assert visible[at + 1].isSeparator(), 'and after'


def test_every_exclusive_group_is_fenced(pane):
    """All of them, not the ones somebody remembered."""
    groups = [name for name in dir(pane) if name.endswith('_group')]
    assert len(groups) >= 6, 'the bar really does hold this many'
    visible = [a for a in pane.toolbar.actions() if a.isVisible()]
    for name in groups:
        group = getattr(pane, name)
        members = [a for a in group.actions() if a in visible]
        if not members:
            continue
        first = min(visible.index(a) for a in members)
        last = max(visible.index(a) for a in members)
        # the ends of the bar are fences in themselves
        assert first == 0 or visible[first - 1].isSeparator(), (
            f'{name} opens undivided')
        assert last + 1 == len(visible) or visible[last + 1].isSeparator(), (
            f'{name} closes undivided')


def test_no_choice_shares_a_compartment_with_a_setting(pane):
    """The rule stated as the reader experiences it: between any two
    dividers, either everything is a member of one group or nothing
    is."""
    visible = [a for a in pane.toolbar.actions() if a.isVisible()]
    groups = [getattr(pane, name) for name in dir(pane)
              if name.endswith('_group')]
    compartment: list = []
    for action in visible + [None]:
        if action is None or action.isSeparator():
            owners = {next((g for g in groups if a in g.actions()), None)
                      for a in compartment}
            assert len(owners) <= 1, (
                'a compartment mixes a choice with something else: '
                + ', '.join(a.text() or '<widget>' for a in compartment))
            compartment = []
        else:
            compartment.append(action)


def test_the_3d_toggle_leads_the_bar(pane):
    """Always furthest left (Brandon, 2026-08-28): the toggle is how
    any reading is drawn, not one of the readings, so it takes a fixed
    place the hand can find whatever the selection has hidden or shown
    — and first is the one position that cannot drift as controls
    appear and disappear around it."""
    first = next(action for action in pane.toolbar.actions()
                 if not action.isSeparator())
    assert first is pane.waterfall_action
