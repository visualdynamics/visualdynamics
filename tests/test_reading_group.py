"""The readings of a record are one choice, and the bar says so.

Averaging, filter, kurtosis, shocks and the wavelet are ways of reading
one record, and a record is read one way at a time. That was always true and
the bar did not show it: four buttons in a row, indistinguishable from
the independent toggles beside them, with the exclusion implemented four
times over — each button standing its siblings down by name. The rule
being written four times is how it came to disagree with itself once,
leaving shocks up with the filter still on.

Now they are an exclusive action group fenced by separators (Brandon,
2026-08-27). The fence is the part the user reads; the group is the part
that cannot be got wrong.

**ExclusiveOptional, not Exclusive.** None of them is a legitimate state
— the plain unmarked trace is the most common thing to want — so
clicking the checked button turns it off. The separators say "one of
these"; they do not say "you must".

The 2-D/3-D toggle stays outside the fence on purpose: it is not a fifth
reading, it is how any of them is drawn.
"""

from __future__ import annotations

import pytest
from conftest import fixture_path
from PySide6.QtGui import QActionGroup

MODAL = ('plate', 'modal.nc4')


def looking_at(window, pump, fixture=MODAL, name='Time History'):
    window.import_paths([fixture_path(*fixture)])
    item = window._item_for_object(name)
    window.tree.clearSelection()
    item.setSelected(True)
    window.tree.setCurrentItem(item)
    window.render_current()
    pump()
    return window.objects[name]


@pytest.fixture
def pane(window, pump):
    looking_at(window, pump)
    return window.data_pane


def readings(pane):
    return {'averaging': pane.averaging_action, 'filter': pane.filter_action,
            'sine': pane.sine_action,
            'truncate': pane.truncate_action,
            'kurtosis': pane.kurtosis_action, 'shocks': pane.shocks_action,
            'wavelet': pane.wavelet_action}


def test_the_readings_are_one_exclusive_group(pane):
    group = pane.reading_group
    assert isinstance(group, QActionGroup)
    assert set(group.actions()) == set(readings(pane).values())
    assert (group.exclusionPolicy()
            == QActionGroup.ExclusionPolicy.ExclusiveOptional)


def test_the_group_is_fenced_by_separators(pane):
    """What the user actually sees: a divider either side, so the four
    read as a set rather than as four independent switches."""
    actions = pane.toolbar.actions()
    positions = [actions.index(action) for action in readings(pane).values()]
    before = [i for i in range(min(positions)) if actions[i].isSeparator()]
    after = [i for i in range(max(positions) + 1, len(actions))
             if actions[i].isSeparator()]
    assert before, 'a divider opens the group'
    assert after, 'and one closes it'
    assert max(before) == min(positions) - 1, 'immediately before the first'
    assert min(after) == max(positions) + 1, 'immediately after the last'


def test_nothing_else_sits_inside_the_fence(pane):
    """A stray toggle between the dividers would read as a fifth
    reading and be neither exclusive nor independent."""
    actions = pane.toolbar.actions()
    positions = [actions.index(action) for action in readings(pane).values()]
    inside = actions[min(positions):max(positions) + 1]
    assert set(inside) == set(readings(pane).values())


@pytest.mark.parametrize('standing', ['averaging', 'filter', 'truncate',
                                      'kurtosis', 'shocks', 'wavelet'])
@pytest.mark.parametrize('chosen', ['averaging', 'filter', 'truncate',
                                    'kurtosis', 'shocks', 'wavelet'])
def test_choosing_one_puts_the_others_down(pane, pump, standing, chosen):
    """The exclusion itself, from every direction to every other.

    Every ordered pair, because the bug that got through when this was
    four hand-written lists was one *direction* that had been missed —
    shocks stood the others down, and the filter did not stand down
    shocks. A test that starts from all-off cannot see that at all: it
    is the state where one is already up that the exclusion is for.
    """
    if standing == chosen:
        pytest.skip('no exclusion to test against itself')
    for action in readings(pane).values():
        action.setVisible(True)

    readings(pane)[standing].trigger()
    pump()
    assert readings(pane)[standing].isChecked(), 'the first one is up'

    readings(pane)[chosen].trigger()
    pump()
    for name, action in readings(pane).items():
        assert action.isChecked() == (name == chosen), (
            f'{standing} was up, {chosen} was chosen, and {name} is wrong')


@pytest.mark.parametrize('chosen', ['averaging', 'filter', 'truncate',
                                    'kurtosis', 'shocks', 'wavelet'])
def test_the_flags_follow_the_buttons(pane, pump, chosen):
    """The flags the drawing code reads, which the old version updated
    by hand in each handler."""
    for action in readings(pane).values():
        action.setVisible(True)
    readings(pane)[chosen].trigger()
    pump()

    flags = {'averaging': pane.averaging_wanted, 'filter': pane.filter_wanted,
             'truncate': pane.truncate_wanted,
             'kurtosis': pane.kurtosis_wanted, 'shocks': pane.shocks_wanted,
             'wavelet': pane.wavelet_wanted}
    assert flags == {name: name == chosen for name in flags}


def test_clicking_the_chosen_one_again_leaves_none_chosen(pane, pump):
    """ExclusiveOptional: the plain trace is a state worth being able to
    get back to, and it is the most common one."""
    pane.averaging_action.trigger()
    pump()
    assert pane.averaging_wanted

    pane.averaging_action.trigger()
    pump()
    assert not pane.averaging_wanted
    assert not any(action.isChecked() for action in readings(pane).values())


def test_the_panels_go_away_with_their_readings(pane, pump):
    """Standing a reading down has to take its panel with it, or the
    parameters of a view nobody is looking at stay beside the plot."""
    pane.averaging_action.trigger()
    pump()
    assert pane.averaging_panel.isVisible()

    pane.filter_action.setVisible(True)
    pane.filter_action.trigger()
    pump()
    assert not pane.averaging_panel.isVisible(), 'its panel went with it'


def test_the_3d_toggle_is_not_one_of_the_readings(pane):
    """It is how a reading is drawn, not a reading — so it is outside
    the fence and stays independent of whichever is chosen."""
    assert pane.waterfall_action not in pane.reading_group.actions()
    actions = pane.toolbar.actions()
    positions = [actions.index(action) for action in readings(pane).values()]
    assert not (min(positions) < actions.index(pane.waterfall_action)
                < max(positions))


def test_every_reading_panel_lives_inside_the_pane(pane):
    """A panel constructed but never added to the plot row is a
    parentless widget, and Qt shows one of those as its own top-level
    window — which is how the truncate panel first shipped (Brandon,
    2026-08-28). The guard for the next reading added: its panel must
    be a child of the pane, beside the plot, like every other."""
    for _action, _wanted, panel_name in type(pane)._READINGS:
        if panel_name is None:
            continue
        panel = getattr(pane, panel_name)
        assert panel.parent() is pane, \
            f'{panel_name} is not in the pane; shown, it would be ' \
            'a separate window'
        assert not panel.isWindow()
