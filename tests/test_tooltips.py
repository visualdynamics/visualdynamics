"""Long tooltips wrap (Brandon, 2026-10-17: "sometimes the length of the
text in the tooltips makes the tooltip quite wide"). Qt draws a plain
tooltip on one line; the application's filter shows a long one as a
paragraph, and leaves a short one alone."""

from __future__ import annotations

from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QToolTip

from visualdynamics.gui.tooltips import WRAP_CHARACTERS


def test_a_long_tooltip_wraps_and_a_short_one_does_not(window):
    from PySide6.QtWidgets import QPushButton

    button = QPushButton('x', window)
    long_text = ('Whether a shaker can play this record: the force at its '
                 'largest peak, and its travel each way and peak velocity at '
                 'each candidate high-pass cutoff, against the ratings')
    button.setToolTip(long_text)
    metrics = QFontMetrics(QToolTip.font())
    assert button.toolTip().startswith('<table width='), button.toolTip()
    assert f'width="{WRAP_CHARACTERS * metrics.averageCharWidth()}"' in button.toolTip()
    assert 'the ratings' in button.toolTip()
    button.setToolTip('Averaging')
    assert button.toolTip() == 'Averaging', 'a short tooltip is left alone'
    # a bar button takes its action's tooltip, so it wraps with it
    pane = window.data_pane
    pane.averaging_action.setToolTip(long_text)
    shown = pane.toolbar.widgetForAction(pane.averaging_action).toolTip()
    assert shown.startswith('<table width='), shown


def test_text_in_a_long_tooltip_is_escaped():
    from visualdynamics.gui.tooltips import wrapped

    out = wrapped('a < b & c\nnext', 300)
    assert 'a &lt; b &amp; c<br>next' in out


def test_every_control_in_the_window_says_what_it_does(window, pump):
    """Brandon (2026-10-17): "sometimes I feel like I don't see one when
    I should." Every button, box, drop-down, field and slider in the
    window and its panes shows a tooltip on hover; a bar button whose
    tooltip is only its own name does not count. Qt's own parts (a
    table's corner button, a toolbar's overflow arrow) are left out."""
    from PySide6.QtWidgets import (
        QAbstractButton,
        QAbstractSpinBox,
        QComboBox,
        QLineEdit,
        QSlider,
        QToolButton,
        QWidget,
    )

    kinds = (QAbstractButton, QComboBox, QAbstractSpinBox, QLineEdit, QSlider)
    silent = []
    for widget in window.findChildren(QWidget):
        if not isinstance(widget, kinds) or widget.objectName().startswith('qt_'):
            continue
        if isinstance(widget.parentWidget(), (QAbstractSpinBox, QComboBox)):
            continue            # a spin box's own line edit, a combo's view
        tip = widget.toolTip()
        if isinstance(widget, QToolButton) and widget.defaultAction() is not None:
            action = widget.defaultAction()
            tip = action.toolTip()
            if tip.replace('&', '') == action.text().replace('&', ''):
                tip = ''
        if not tip.strip():
            label = getattr(widget, 'text', lambda: '')() if isinstance(
                widget, QAbstractButton) else type(widget).__name__
            silent.append(f'{type(widget.window()).__name__}/'
                          f'{type(widget.parentWidget()).__name__}: {label!r}')
    assert not silent, 'no tooltip on:\n  ' + '\n  '.join(silent)


def test_a_wrapped_tooltip_reads_back_as_its_words():
    from visualdynamics.gui.tooltips import plain, wrapped

    text = 'Integrate — a < b & c\nthen d'
    assert plain(wrapped(text, 420)) == text
    assert plain('Averaging') == 'Averaging'
