"""The application's own copy buttons: the plot bar, the 3-D bar, and
every table (Brandon, 2026-09-24: everything should copy the way a
chat's code blocks do). The report's are in `test_report_copy.py`.
"""

from __future__ import annotations

import numpy as np
from test_acts import _bar
from test_table_machinery import Sheet, sheet_model

import visualdynamics
from visualdynamics.gui.tables import CopyPasteTableView

T = np.arange(1024) / 1024.0


def _time_history():
    return visualdynamics.TimeHistory(
        T, np.random.default_rng(4).standard_normal((2, len(T))),
        response_dof=['101Z+', '104Z+'], ordinate_dim='acceleration')


def test_the_plot_bar_copies_the_flat_plot_at_screen_density(window, pump):
    from PySide6.QtWidgets import QApplication

    window.add_object('Run', _time_history())
    window.tree.setCurrentItem(window._item_for_object('Run'))
    window.data_pane.waterfall_action.setChecked(False)
    window.render_current()
    pump()
    QApplication.clipboard().clear()
    assert _bar(window.data_pane)[-1] == 'Copy', 'last on the bar'
    assert window.data_pane.copy_view()
    image = QApplication.clipboard().image()
    assert not image.isNull()
    graphics = window.data_pane.graphics
    assert image.width() == int(graphics.width() * graphics.devicePixelRatioF())
    assert 'copied to the clipboard' in window.statusBar().currentMessage()


def test_the_plot_bar_copies_the_stage_when_it_is_up(window, pump):
    from PySide6.QtWidgets import QApplication

    window.add_object('Run', _time_history())
    window.tree.setCurrentItem(window._item_for_object('Run'))
    window.data_pane.waterfall_action.setChecked(True)
    window.render_current()
    pump()
    if window.data_pane.waterfall_plotter is None:
        # the stage never built for this reading; the flat plot copies
        assert window.data_pane.copy_view()
        return
    QApplication.clipboard().clear()
    assert window.data_pane.copy_view()
    assert not QApplication.clipboard().image().isNull()
    assert 'Stage copied' in window.statusBar().currentMessage()


def test_the_3d_bar_copies_the_scene(window, pump):
    from PySide6.QtWidgets import QApplication

    geometry = visualdynamics.Geometry(
        np.arange(1, 5), np.random.default_rng(5).uniform(-1, 1, (4, 3)),
        length_unit='m')
    window.add_object('Plate', geometry)
    window.tree.setCurrentItem(window._item_for_object('Plate'))
    window.render_current()
    pump()
    QApplication.clipboard().clear()
    assert _bar(window.scene) == ['Copy'], 'the 3-D bar offers it'
    assert window.scene.copy_view()
    image = QApplication.clipboard().image()
    assert not image.isNull() and image.width() > 0
    assert '3-D view copied' in window.statusBar().currentMessage()


def test_a_table_copies_whole_with_headers_as_cells_and_as_a_table(qt_app):
    from PySide6.QtWidgets import QApplication

    view = CopyPasteTableView()
    view.setModel(sheet_model(Sheet(rows=2, columns=3)))
    try:
        assert view.copy_table()
        mime = QApplication.clipboard().mimeData()
        assert mime.text() == 'C0\tC1\tC2\nr0c0\tr0c1\tr0c2\nr1c0\tr1c1\tr1c2'
        assert mime.hasHtml()
        assert mime.html().count('<th>') == 3 and mime.html().count('<tr>') == 3
    finally:
        view.deleteLater()


def test_the_tables_copy_button_shows_under_the_pointer(qt_app):
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QEnterEvent

    view = CopyPasteTableView()
    view.setModel(sheet_model(Sheet(rows=2, columns=3)))
    view.resize(400, 200)
    view.show()
    qt_app.processEvents()
    try:
        button = view._copy_button
        # the offscreen platform's pointer sits at the origin, inside a
        # window shown there, so the view may already count as entered;
        # a leave puts it in the state a pointer elsewhere would
        view.leaveEvent(QEvent(QEvent.Type.Leave))
        assert not button.isVisible(), 'hidden until the pointer arrives'
        view.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        assert button.isVisible()
        assert button.geometry().right() <= view.viewport().geometry().right()
        assert button.toolTip().startswith('Copy the whole table')
        view.leaveEvent(QEvent(QEvent.Type.Leave))
        assert not button.isVisible()
    finally:
        view.close()
        view.deleteLater()


def test_a_table_the_owner_marked_working_surface_has_no_button(qt_app):
    """Geometry editing turns whole-table copy off; the button follows."""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QEnterEvent

    view = CopyPasteTableView()
    view.setModel(sheet_model(Sheet()))
    view.whole_table_copy = False
    view.show()
    qt_app.processEvents()
    try:
        view.enterEvent(QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(5, 5)))
        assert not view._copy_button.isVisible()
    finally:
        view.close()
        view.deleteLater()
