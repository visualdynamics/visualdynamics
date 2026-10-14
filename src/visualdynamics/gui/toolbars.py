"""Dividers on a plot's control bar, which show how its controls group.

A control bar holds two kinds of control, and they behave differently:

- Choices. Curves or Map; Averaging, Filter, Kurtosis or Shocks. Exactly
  one of each set is in force, and picking one turns off the last, so
  the reader needs to know which controls form a set.
- Settings. The 2-D/3-D toggle, which quantity is on the stage, which
  page of channels, Drive points, CMIF. Each is independent of the
  others and of whichever choice is in force. 3-D is how any of the
  choices is drawn, and is not a choice of its own.

Without dividers, a person has to click an icon to find out whether it
turns another one off. A divider shows the grouping before they click.

The bars are built once with every control and then hidden down to
what the current selection can use, so there are two functions:
`fence` puts the dividers in at build time, and `tidy` decides at show
time which of them still divide anything. Every bar uses the same
rule, so the bars all read the same way.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

# 3-D is a setting, orthogonal to the choices, and not a fifth way of
# reading a record (Brandon, 2026-08-27).


def fence(toolbar: Any, runs: Iterable[Sequence[Any]]) -> None:
    """Put a divider on each side of every run of actions.

    Parameters
    ----------
    toolbar : QToolBar
        The bar to add dividers to. Separators are inserted between its
        actions, and no action changes place.
    runs : iterable of sequence of QAction
        Each run is a group that belongs together: the actions of a
        `QActionGroup`, or a single setting that should stand apart. A
        run whose actions are not adjacent on the bar is fenced from its
        first action to its last, so add a group's actions together.
        `None` entries are skipped, and a run with an action that is not
        on the bar is skipped entirely.

    Notes
    -----
    `tidy` hides the dividers that turn out to divide nothing, so a
    fence around a group that the current selection hides does not leave
    a stray line. That is why every group can be fenced, and not only
    the ones that seem to need it.
    """
    for run in runs:
        actions = [action for action in run if action is not None]
        if not actions:
            continue
        current = toolbar.actions()
        try:
            first = min(current.index(action) for action in actions)
            last = max(current.index(action) for action in actions)
        except ValueError:            # an action that is not on this bar
            continue
        toolbar.insertSeparator(current[first])
        if last + 1 < len(current):
            toolbar.insertSeparator(current[last + 1])
        else:
            toolbar.addSeparator()


def tidy(toolbar: Any) -> None:
    """Show only the dividers that still divide something.

    The bar is built with every control on it and then hidden down to
    what the current selection can use, so which groups are left differs
    each time. A divider with nothing on one side reads as a group that
    is missing, when the group does not apply.

    A separator is kept when a visible control is somewhere before it
    and somewhere after it, and no separator has been kept since the
    last visible control. That drops leading and trailing dividers and
    reduces a run of adjacent dividers to one. Two groups fenced side by
    side put two separators together, and dropping both would run the
    groups into each other.
    """
    # Reducing adjacent dividers to one, and not to none, came from
    # looking at the bar: Curves/Map ran straight into the 3-D toggle.
    actions = toolbar.actions()
    # is there a visible control still to come, at each position?
    ahead = [False] * (len(actions) + 1)
    for index in range(len(actions) - 1, -1, -1):
        action = actions[index]
        ahead[index] = ahead[index + 1] or (
            not action.isSeparator() and action.isVisible())

    pending = False          # a visible control since the last divider kept
    for index, action in enumerate(actions):
        if action.isSeparator():
            keep = pending and ahead[index + 1]
            action.setVisible(keep)
            if keep:
                pending = False
        elif action.isVisible():
            pending = True


def shows_anything(toolbar: Any) -> bool:
    """Whether the bar has a visible control on it, not counting
    dividers.

    A separator is visible whether or not anything is around it, so a
    bar of nothing but dividers would pass a plain `any(isVisible)` and
    stay up. Run `tidy` first, then this.
    """
    return any(action.isVisible() for action in toolbar.actions()
               if not action.isSeparator())
