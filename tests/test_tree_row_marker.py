"""The project tree draws no selection marker in the link brackets' gutter.

Qt's Windows 11 style marks each selected tree row with a short bar in
the accent color, 3.5 px from the left edge, where the link brackets
are drawn; on a friend's Windows machine (2026-11-12) it read as stray
pieces of a blue bracket beside an orange and a green group. The tests
run on Fusion, so the style is stood in for by one that records the
accent each row is drawn with.
"""

from __future__ import annotations

from PySide6.QtWidgets import QProxyStyle, QStyle, QTreeWidgetItem

from visualdynamics.gui.project_tree import (
    NoRowMarkerStyle,
    ProjectTree,
    drop_row_marker,
)


class _Recording(QProxyStyle):
    """Fusion, noting the accent alpha and selection of every row."""

    def __init__(self) -> None:
        super().__init__('Fusion')
        self.rows: list[tuple[bool, int]] = []

    def drawPrimitive(self, element, option, painter, widget=None):
        if element == QStyle.PrimitiveElement.PE_PanelItemViewRow:
            self.rows.append((bool(option.state & QStyle.StateFlag.State_Selected),
                              option.palette.accent().color().alpha()))
        super().drawPrimitive(element, option, painter, widget)


def _drawn(qt_app, style):
    tree = ProjectTree()
    tree.setStyle(style)
    for name in ('Geometry', 'Shapes', 'FRFs'):
        QTreeWidgetItem(tree, [name])
    tree.topLevelItem(1).setSelected(True)
    tree.resize(240, 160)
    tree.grab()
    return tree


def test_a_selected_row_is_drawn_without_the_accent(qt_app):
    recording = _Recording()
    tree = _drawn(qt_app, NoRowMarkerStyle(recording))
    selected = [alpha for chosen, alpha in recording.rows if chosen]
    assert selected, 'a selected row was drawn'
    assert set(selected) == {0}, 'the marker color is transparent'
    tree.deleteLater()


def test_without_it_the_accent_reaches_the_row(qt_app):
    """The stand-in does see the accent when nothing removes it, so the
    test above is not passing on a palette that never had one."""
    recording = _Recording()
    tree = _drawn(qt_app, recording)
    selected = [alpha for chosen, alpha in recording.rows if chosen]
    assert selected and min(selected) > 0
    tree.deleteLater()


def test_only_the_windows_11_style_is_wrapped(qt_app):
    class _Eleven(QProxyStyle):
        def name(self):
            return 'windows11'

    tree = ProjectTree()
    assert not isinstance(tree.style(), NoRowMarkerStyle), 'Fusion here'
    assert drop_row_marker(tree) is False
    tree.setStyle(_Eleven('Fusion'))
    assert drop_row_marker(tree) is True
    assert isinstance(tree.style(), NoRowMarkerStyle)
    assert drop_row_marker(tree) is False, 'once is enough'
    tree.deleteLater()
