"""A number counts when it is finished, not while it is being typed.

Typing `50` into the overlap field applied 5 first: Qt reports every
keystroke unless told not to, and here every report is a real edit — it
landed on the object, moved the shading, and restated the clamped value,
which took the selection away mid-number. 50% overlap could not be
entered at all, and frame length behaved the same way (Brandon,
2026-08-27).

The panel tests below pin the two fields that were reported. The sweep
at the end is the one that matters afterwards: it walks every spin box
the application builds, so a tenth one added next year cannot quietly
arrive with the old behavior.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFocusEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractSpinBox, QApplication

from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import TimeHistory
from visualdynamics.core.filters import Filtering
from visualdynamics.gui.averaging_panel import AveragingPanel
from visualdynamics.gui.filter_panel import FilterPanel


@pytest.fixture
def history():
    t = np.arange(4096) / 512.0
    return TimeHistory(t, np.atleast_2d(np.sin(2 * np.pi * 20.0 * t)),
                       response_dof=['101Z+'])


@pytest.fixture
def panel(qt_app, history):
    panel = AveragingPanel()
    panel.show_history(history, Averaging(frame_length=1024, overlap=0.0,
                                          window='hann', frames=4,
                                          start=0.0))
    # shown, because focus is what half of this file is about and an
    # unshown widget cannot take it
    panel.show()
    return panel


def type_into(box, text):
    """Type a number the way a person does: select what is there, then
    the digits, one key event each."""
    box.setFocus()
    box.selectAll()
    QTest.keyClicks(box.lineEdit(), text)


def test_typing_fifty_percent_overlap_reports_fifty_and_not_five(panel):
    """The bug, exactly as reported."""
    seen = []
    panel.changed.connect(lambda averaging: seen.append(averaging.overlap))

    type_into(panel.overlap_box, '50')

    assert 0.05 not in seen, 'the half-typed 5% was applied'
    assert not seen, 'nothing is reported until the edit is finished'
    QTest.keyClick(panel.overlap_box.lineEdit(), Qt.Key.Key_Return)
    assert seen == [0.5]


def test_typing_a_frame_length_reports_it_once_it_is_finished(panel):
    """The second field Brandon named. 2048 would have arrived as 2,
    then 20, then 204 — and 2 is below the box's own minimum, so the
    clamp restated it and the rest of the number went nowhere."""
    seen = []
    panel.changed.connect(lambda averaging: seen.append(averaging.frame_length))

    type_into(panel.length_box, '2048')
    assert not seen
    QTest.keyClick(panel.length_box.lineEdit(), Qt.Key.Key_Return)
    assert seen == [2048]


def test_the_arrows_still_report_immediately(panel):
    """Stepping is a finished edit each time — the value on screen is
    always one the user asked for, so nothing is deferred."""
    seen = []
    panel.changed.connect(lambda averaging: seen.append(averaging.overlap))
    panel.overlap_box.stepUp()
    assert seen, 'a step is a whole edit and reports at once'


def test_leaving_the_field_reports_it_too(panel):
    """Enter is not the only way a person finishes: clicking away is
    the other, and a value typed and abandoned must not be lost.

    The focus-out event is delivered rather than provoked by moving
    focus: the offscreen platform this suite runs under never activates
    a window, so `setFocus` lands nowhere and a real click cannot be
    staged. This is the event Qt itself sends when focus leaves, and it
    is what `QAbstractSpinBox` acts on.
    """
    seen = []
    panel.changed.connect(lambda averaging: seen.append(averaging.overlap))
    type_into(panel.overlap_box, '25')
    assert not seen, 'still being typed'

    QApplication.sendEvent(
        panel.overlap_box,
        QFocusEvent(QEvent.Type.FocusOut, Qt.FocusReason.MouseFocusReason))
    assert seen == [0.25], 'the finished value arrives when focus goes'
    assert panel.overlap_box.value() == 25.0


def test_the_filter_corner_waits_as_well(qt_app, history):
    """1000 Hz would have applied as 1, 10 and 100 Hz on the way — three
    filter designs nobody asked for, and the record refiltered each
    time."""
    panel = FilterPanel()
    panel.show_history(history, Filtering(high=100.0, order=4))
    seen = []
    panel.changed.connect(lambda filtering: seen.append(filtering.high))
    type_into(panel.corner_box, '250')
    assert not seen
    QTest.keyClick(panel.corner_box.lineEdit(), Qt.Key.Key_Return)
    assert seen == [250.0]


def test_every_spin_box_in_the_window_waits_for_a_finished_edit(window):
    """The guard for the next one added.

    Keyboard tracking is on by default, so a new spin box arrives with
    the bug unless its author knows this. Nobody should have to: this
    fails the moment one does.
    """
    boxes = window.findChildren(QAbstractSpinBox)
    assert boxes, 'the window builds spin boxes at all'
    typing = [box for box in boxes if box.keyboardTracking()]
    assert not typing, (
        'these report every keystroke; pass them to '
        'gui.editors.commit_on_enter: '
        + ', '.join(box.objectName() or box.__class__.__name__
                    for box in typing))


# ---- a number over the maximum is corrected, not refused ----------------


def over_type(box, text):
    """Type a number into a box that already holds one."""
    box.setFocus()
    box.selectAll()
    QTest.keyClicks(box.lineEdit(), text)
    shown = box.lineEdit().text()
    QTest.keyClick(box.lineEdit(), Qt.Key.Key_Return)
    return shown, box.value()


def test_a_number_above_the_maximum_can_be_typed_at_all(qt_app):
    """Qt refuses the keystroke: with a range ending at 1638, typing
    2000 stops after the third digit, because the fourth would make a
    value out of range and the validator calls that invalid. The number
    never appears and nothing says why — the box just stops responding
    (Brandon, 2026-08-28).
    """
    from visualdynamics.gui.editors import DoubleSpinBox

    box = DoubleSpinBox()
    box.setRange(0.0, 1638.0)
    box.setDecimals(2)
    box.setSuffix(' Hz')
    box.show()

    shown, value = over_type(box, '2000')
    assert shown.startswith('2000'), 'the whole number was typeable'
    assert value == pytest.approx(1638.0), 'and then corrected to the maximum'


def test_the_same_for_whole_numbers(qt_app):
    from visualdynamics.gui.editors import SpinBox

    box = SpinBox()
    box.setRange(2, 4096)
    box.show()
    shown, value = over_type(box, '99999')
    assert shown.startswith('99999')
    assert value == 4096


def test_a_number_below_the_minimum_is_corrected_too(qt_app):
    from visualdynamics.gui.editors import DoubleSpinBox

    box = DoubleSpinBox()
    box.setRange(10.0, 1638.0)
    box.setDecimals(2)
    box.show()
    _shown, value = over_type(box, '1')
    assert value == pytest.approx(10.0)


def test_a_number_inside_the_range_is_left_alone(qt_app):
    """The correction is a clamp, not a rounding: what fits is what was
    typed."""
    from visualdynamics.gui.editors import DoubleSpinBox

    box = DoubleSpinBox()
    box.setRange(0.0, 1638.0)
    box.setDecimals(2)
    box.show()
    _shown, value = over_type(box, '250.5')
    assert value == pytest.approx(250.5)


def test_nonsense_is_still_refused(qt_app):
    """Permissive about range, not about what a number is."""
    from visualdynamics.gui.editors import DoubleSpinBox

    box = DoubleSpinBox()
    box.setRange(0.0, 1638.0)
    box.show()
    box.setValue(100.0)
    box.setFocus()
    box.selectAll()
    QTest.keyClicks(box.lineEdit(), 'abc')
    QTest.keyClick(box.lineEdit(), Qt.Key.Key_Return)
    assert box.value() == pytest.approx(100.0), 'the last good value stands'


def test_every_spin_box_in_the_window_corrects_rather_than_refuses(window):
    """The guard for the next one added, beside the keyboard-tracking
    sweep above: a plain QSpinBox would refuse the keystroke."""
    from visualdynamics.gui.editors import _Clamping

    boxes = window.findChildren(QAbstractSpinBox)
    assert boxes, 'the window builds spin boxes at all'
    plain = [box for box in boxes if not isinstance(box, _Clamping)]
    assert not plain, (
        'these refuse a number above their maximum instead of correcting '
        'it; build them from gui.editors.SpinBox / DoubleSpinBox: '
        + ', '.join(box.objectName() or box.__class__.__name__
                    for box in plain))
