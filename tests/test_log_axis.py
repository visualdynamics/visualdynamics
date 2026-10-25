"""Log-axis tick labels fitted to the axis they are drawn on.

Brandon, 2026-10-25: the impulse preview's spectrum, two decades on a
plot about 100 pixels tall, carried eighteen labels on top of each
other, and '10¹' for 1e-5 beside a '×1e-06' on the axis label. The
labels are now the densest set that fits, and a log axis takes no SI
prefix.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualdynamics.plot import log_axis
from visualdynamics.plot.log_axis import choose, label


def test_labels_are_plain_in_the_middle_and_powers_outside():
    assert [label(v) for v in (1e-6, 3e-6, 2.5e-5, 1e-4, 0.001)] == [
        '10⁻⁶', '3·10⁻⁶', '2.5·10⁻⁵', '10⁻⁴', '10⁻³']
    assert [label(v) for v in (0.01, 0.0125, 0.5, 200, 1500, 9999)] == [
        '0.01', '0.0125', '0.5', '200', '1500', '9999']
    assert [label(v) for v in (1e4, 2e5, 1e6)] == ['10⁴', '2·10⁵', '10⁶']


@pytest.mark.parametrize(('decades', 'pixels', 'room', 'chosen'), [
    (1.0, 600.0, 20.0, ((1, 2, 3, 4, 5, 6, 7, 8, 9), 1)),  # 27 px at 9 to 10
    (2.0, 400.0, 20.0, ((1, 2, 5), 1)),                    # 60 px from 1 to 2
    (2.0, 100.0, 20.0, ((1, 3), 1)),                       # 24 px from 1 to 3
    (2.0, 30.0, 14.0, ((1,), 1)),                          # 15 px a decade
    (6.0, 100.0, 20.0, ((1,), 2)),                         # 17 px a decade
    (40.0, 100.0, 20.0, ((1,), 10)),                       # 2.5 px a decade
])
def test_the_densest_set_that_fits_is_chosen(decades, pixels, room, chosen):
    assert choose(0.0, decades, pixels, room) == chosen


def _log_plot(qt_app, low, high, height, width=500, horizontal=False):
    """A plot drawn with a log axis from `low` to `high` (values), its
    log axis the one returned, laid out at `width` × `height`."""
    import pyqtgraph as pg

    log_axis.install()
    widget = pg.PlotWidget()
    widget.resize(width, height)
    x = np.geomspace(low, high, 200)
    if horizontal:
        widget.plotItem.setLogMode(x=True, y=False)
        widget.plot(x, np.linspace(0.0, 1.0, x.size))
    else:
        widget.plotItem.setLogMode(x=False, y=True)
        widget.plot(np.linspace(0.0, 1.0, x.size), x)
    widget.plotItem.getAxis('left').setLabel('force', units='N*s')
    widget.show()
    for _ in range(5):
        qt_app.processEvents()
    side = 'bottom' if horizontal else 'left'
    return widget, widget.plotItem.getAxis(side)


def _drawn(axis):
    """The labels the axis draws, from its own tick levels."""
    lo, hi = axis.range
    geometry = axis.geometry()
    length = (geometry.width() if axis.orientation in ('top', 'bottom')
              else geometry.height())
    strings = []
    for spacing, values in axis.tickValues(lo, hi, length):
        strings += [s for s in axis.tickStrings(values, axis.autoSIPrefixScale,
                                                spacing or 1.0) if s]
    return strings, length


def test_a_short_log_axis_carries_few_labels_and_no_si_prefix(qt_app):
    """The impulse preview's case: two decades in about 100 pixels."""
    widget, axis = _log_plot(qt_app, 3e-6, 1.5e-4, height=160)
    strings, length = _drawn(axis)
    try:
        assert 2 <= len(strings) <= length / 20.0, (strings, length)
        assert all('·10' in s or s.startswith('10') for s in strings)
        assert axis.autoSIPrefixScale == 1.0
        assert 'x1e' not in axis.labelString(), 'no SI prefix on a log axis'
    finally:
        widget.close()


def test_a_tall_log_axis_carries_more_and_ticks_every_mantissa(qt_app):
    widget, axis = _log_plot(qt_app, 1e-7, 3e-4, height=520)
    try:
        strings, _length = _drawn(axis)
        assert '2·10⁻⁵' in strings and '5·10⁻⁵' in strings, strings
        lo, hi = axis.range
        levels = axis.tickValues(lo, hi, axis.geometry().height())
        ticks = sum(len(values) for _spacing, values in levels)
        assert ticks >= 9 * 3, 'every mantissa still ticks the grid'
    finally:
        widget.close()


def test_a_horizontal_log_axis_fits_its_labels_by_width(qt_app):
    widget, axis = _log_plot(qt_app, 10.0, 1e5, height=200, width=260,
                             horizontal=True)
    try:
        strings, length = _drawn(axis)
        assert strings and len(strings) <= 5, (strings, length)
    finally:
        widget.close()


def test_a_span_narrower_than_round_mantissas_gets_round_values(qt_app):
    widget, axis = _log_plot(qt_app, 3.1e-5, 3.4e-5, height=300)
    try:
        strings, _length = _drawn(axis)
        assert len(strings) >= 2, strings
        assert all(s.endswith('10⁻⁵') for s in strings), strings
    finally:
        widget.close()


def test_a_linear_axis_keeps_its_si_prefix(qt_app):
    import pyqtgraph as pg

    log_axis.install()
    widget = pg.PlotWidget()
    widget.plot([0.0, 1.0], [0.0, 0.002])
    widget.plotItem.getAxis('left').setLabel('force', units='N')
    widget.show()
    for _ in range(5):
        qt_app.processEvents()
    try:
        assert widget.plotItem.getAxis('left').autoSIPrefixScale == 1000.0
    finally:
        widget.close()


@pytest.mark.parametrize('entry', [
    'import visualdynamics.gui.panes',
    'from visualdynamics.plot import _application; _application()',
])
def test_the_window_and_a_script_both_install_it(entry):
    """In a fresh process, so no other test's install hides a missing
    one: the window's panes and a script's first plot each put it on
    every axis."""
    import os
    import subprocess
    import sys

    check = (f'{entry}\n'
             'import pyqtgraph as pg\n'
             'from visualdynamics.plot import log_axis\n'
             'print(pg.AxisItem.logTickValues is log_axis.log_tick_values)\n')
    done = subprocess.run([sys.executable, '-c', check], check=False,
                          capture_output=True,
                          text=True, env={**os.environ,
                                          'QT_QPA_PLATFORM': 'offscreen'},
                          timeout=120)
    assert done.stdout.strip().splitlines()[-1] == 'True', done.stderr
