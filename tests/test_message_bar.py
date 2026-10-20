"""The status line is selectable text (Brandon, 2026-10-20: a refusal
could be read but not copied with Cmd+C)."""

from __future__ import annotations

from PySide6.QtCore import Qt


def test_the_status_line_can_be_selected_and_copied(window, pump):
    """A message shown is a label's text that the mouse and keyboard can
    select, which Cmd+C or the label's own menu copies; what reads the
    bar reads the same message as before. The chord itself is not typed
    here: a modifier typed with QTest stays held for the rest of the
    run."""
    from visualdynamics.gui.message_bar import MessageBar

    bar = window.statusBar()
    assert isinstance(bar, MessageBar)
    said = 'Geometry: element group 3 (Subtract 10) holds solid elements'
    window._show_status(said)
    pump()
    assert bar.currentMessage() == said and bar.label.text() == said
    flags = bar.label.textInteractionFlags()
    assert flags & Qt.TextInteractionFlag.TextSelectableByMouse
    assert flags & Qt.TextInteractionFlag.TextSelectableByKeyboard
    assert bar.label.isVisible() and bar.label.toolTip() == said
    bar.label.setSelection(0, len(said))
    assert bar.label.selectedText() == said
    # the window's own Copy belongs to the tree alone, so the label keeps
    # the chord to itself when it has the focus
    assert window.copy_action.shortcutContext() == Qt.ShortcutContext.WidgetShortcut
    assert window.copy_action not in bar.label.actions()


def test_a_long_message_does_not_widen_the_window(window, pump):
    width = window.width()
    window._show_status('word ' * 400)
    pump()
    assert window.width() == width
    window.statusBar().clearMessage()
    assert window.statusBar().currentMessage() == '' and window.statusBar().label.text() == ''
