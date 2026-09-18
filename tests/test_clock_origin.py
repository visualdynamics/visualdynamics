"""Marks on a record whose clock does not begin at zero.

An averaging's start and a shock's start count from the record's
beginning — a sample count in disguise — and the plots draw on the
record's own clock, which a truncation keeps and an import window
reads from the run. Brandon (2026-09-18): the last hundred seconds of
a run imported through the window, beginning at 1060 s, and the
averaging reading showed nothing: its frames were drawn at 0 s, off
the plot, while the panel said "start 0 s" against an axis that began
at 1060. Every reading in seconds takes the record's first instant
as its origin, once, in `Averaging.stop`, `frame_bounds` and
`from_span`; the overlays, the stage's geometry, the dragger, the
report and the panels hand it over.
"""

import numpy as np
import pytest

import visualdynamics
from visualdynamics.core.averaging import Averaging, from_span
from visualdynamics.core.shocks import Shock
from visualdynamics.theme import theme as resolve_theme

RATE = 1024.0
ORIGIN = 1060.0
SAMPLES = 8192


def _late_history(origin=ORIGIN, samples=SAMPLES, channels=2):
    t = origin + np.arange(samples) / RATE
    return visualdynamics.TimeHistory(
        t, np.random.default_rng(2).standard_normal((channels, samples)),
        response_dof=[f'{101 + k}Z+' for k in range(channels)],
        ordinate_dim='acceleration', ordinate_unit='m/s**2')


# ---- the one implementation ---------------------------------------------------

def test_the_averaging_reads_its_seconds_on_the_clock():
    averaging = Averaging(frame_length=1024, overlap=0.5, frames=3, start=1.0)
    assert averaging.stop(RATE) == pytest.approx(3.0)
    assert averaging.stop(RATE, ORIGIN) == pytest.approx(ORIGIN + 3.0)
    plain = averaging.frame_bounds(RATE)
    clocked = averaging.frame_bounds(RATE, ORIGIN)
    assert plain[0] == (1.0, 2.0)
    assert clocked == [(a + ORIGIN, b + ORIGIN) for a, b in plain]


def test_a_drag_on_the_clock_means_a_start_from_the_beginning():
    averaging = Averaging(frame_length=1024, overlap=0.0, frames=1)
    dragged = from_span(averaging, ORIGIN + 2.0, ORIGIN + 5.0, RATE, SAMPLES,
                        origin=ORIGIN)
    assert dragged.start == pytest.approx(2.0)
    assert dragged.frames == 3
    assert from_span(averaging, 2.0, 5.0, RATE, SAMPLES) == dragged, \
        'the same drag on a record that begins at zero'


# ---- the 2-D overlays -------------------------------------------------------------

def _plot(history):
    import pyqtgraph as pg

    layout = pg.GraphicsLayoutWidget()
    plot = layout.addPlot()
    plot.plot(history.abscissa, history.ordinate[0].real)
    plot.getViewBox().setYRange(-3.0, 3.0, padding=0)
    return layout, plot


def test_the_averaging_overlay_sits_on_the_record(qt_app, pump):
    from visualdynamics.plot.averaging import AveragingOverlay

    history = _late_history()
    layout, plot = _plot(history)
    averaging = Averaging(frame_length=1024, overlap=0.0, frames=2, start=1.0)
    overlay = AveragingOverlay(plot, averaging, RATE, SAMPLES,
                               resolve_theme('light'), origin=ORIGIN)
    assert overlay.region.getRegion() == pytest.approx((ORIGIN + 1.0, ORIGIN + 3.0))
    low, high = overlay.bands[0].getData()[0][[0, -1]]
    assert (low, high) == pytest.approx((ORIGIN + 1.0, ORIGIN + 2.0))
    heard = []
    overlay.changed.connect(heard.append)
    overlay.region.setRegion((ORIGIN + 2.0, ORIGIN + 4.0))
    pump()
    assert heard and heard[-1].start == pytest.approx(2.0), \
        'dragged on the clock, stored from the beginning'
    overlay.remove()
    del layout


def test_the_shock_overlay_sits_on_the_record(qt_app, pump):
    from visualdynamics.plot.shocks import ShockOverlay

    history = _late_history()
    layout, plot = _plot(history)
    shocks = (Shock(1.0, 0.5), Shock(3.0, 0.5))
    heard = []
    overlay = ShockOverlay(plot, shocks, resolve_theme('light'),
                           changed=heard.append, origin=ORIGIN)
    assert overlay.regions[0].getRegion() == pytest.approx((ORIGIN + 1.0, ORIGIN + 1.5))
    overlay.regions[1].setRegion((ORIGIN + 4.0, ORIGIN + 4.5))
    overlay.regions[1].sigRegionChangeFinished.emit(overlay.regions[1])
    pump()
    assert heard and heard[-1][1].start == pytest.approx(4.0)
    overlay.remove()
    del layout


# ---- the stage, the dragger, the report -----------------------------------------

def test_the_stage_geometry_is_on_the_clock():
    from visualdynamics.viz.marks import (
        averaging_stage_geometry,
        shock_stage_geometry,
    )

    extents = (ORIGIN, ORIGIN + (SAMPLES - 1) / RATE, -1.0, 1.0)
    averaging = Averaging(frame_length=1024, overlap=0.0, frames=2, start=1.0)
    geometry = averaging_stage_geometry(averaging, RATE, extents, ORIGIN)
    off = averaging_stage_geometry(averaging, RATE, extents)
    span = geometry['span']
    assert 0.0 <= span[0] < span[1], 'inside the stage'
    assert span != off['span'], 'the origin moved it'
    windows = shock_stage_geometry((Shock(1.0, 0.5),), extents, ORIGIN)['windows']
    assert windows[0]['span'] == pytest.approx(
        averaging_stage_geometry(
            Averaging(frame_length=512, overlap=0.0, frames=1, start=1.0),
            RATE, extents, ORIGIN)['span'])


def test_the_stage_dragger_commits_from_the_beginning():
    from visualdynamics.gui.stage_drag import StageMarksDragger

    dragger = StageMarksDragger(plotter=None)
    averaging = Averaging(frame_length=1024, overlap=0.0, frames=1, start=1.0)
    extents = (ORIGIN, ORIGIN + (SAMPLES - 1) / RATE, -1.0, 1.0)
    seen = []
    dragger.arm_averaging(averaging, RATE, SAMPLES, extents,
                          lambda a: None, seen.append, origin=ORIGIN)
    assert dragger.begin('marks-averaging-handle-start', at_seconds=ORIGIN + 1.0)
    dragger.drag_to(ORIGIN + 2.0)
    dragger.finish(ORIGIN + 2.0)
    assert seen and seen[-1].start == pytest.approx(2.0)


def test_the_report_frames_are_on_the_clock():
    from visualdynamics.report import _averaging_marks

    history = _late_history()
    history.averaging = Averaging(frame_length=1024, overlap=0.0, frames=2,
                                  start=1.0)
    frames = _averaging_marks(history)['frames']
    assert frames[0] == pytest.approx([ORIGIN + 1.0, ORIGIN + 2.0])


# ---- the panels ------------------------------------------------------------------

def test_the_averaging_panel_speaks_the_clock(qt_app):
    from visualdynamics.gui.averaging_panel import AveragingPanel

    history = _late_history()
    panel = AveragingPanel()
    averaging = Averaging(frame_length=1024, overlap=0.0, frames=2, start=1.0)
    panel.show_history(history, averaging)
    assert panel.start_box.value() == pytest.approx(ORIGIN + 1.0)
    assert panel.start_box.minimum() == pytest.approx(ORIGIN)
    assert panel.derived['stop'].text() == f'{ORIGIN + 3.0:.4g} s'
    panel.start_box.setValue(ORIGIN + 2.0)
    assert panel.averaging().start == pytest.approx(2.0)


def test_the_shock_panel_speaks_the_clock(qt_app):
    from visualdynamics.gui.shock_panel import ShockPanel

    panel = ShockPanel()
    panel.origin = ORIGIN
    panel.set_shocks((Shock(1.0, 0.5),))
    assert float(panel.table.item(0, 0).text()) == pytest.approx(ORIGIN + 1.0)
    heard = []
    panel.changed.connect(heard.append)
    panel.table.item(0, 0).setText(f'{ORIGIN + 2.0:.4f}')
    assert heard and heard[-1][0].start == pytest.approx(2.0)


# ---- the whole thing, as Brandon saw it ----------------------------------------------

def test_the_reading_on_a_windowed_record_lands_on_its_plot(window, pump):
    """The averaging reading on the last stretch of a run: the region
    inside the plot's own range, the panel starting where the record
    does."""
    history = _late_history()
    window.add_object('Late', history)
    window.show_object('Late')
    pump()
    pane = window.data_pane
    if pane.waterfall_action.isChecked():
        pane.waterfall_action.trigger()
        pump()
    if not pane.averaging_action.isChecked():
        pane.averaging_action.trigger()
        pump()
    assert window.averaging_overlays, 'the reading is up'
    low, high = window.averaging_overlays[0].region.getRegion()
    first, last = float(history.abscissa[0]), float(history.abscissa[-1])
    # the stop is one sample past the last instant, as it always was
    assert first <= low < high <= last + 1 / RATE + 1e-9
    assert pane.averaging_panel.start_box.value() >= first
