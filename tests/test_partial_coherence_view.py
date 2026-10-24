"""The partial coherence reading in the window: on a multiple coherence
computed from a time history with two or more drives, the multiple
coherence over each drive's partial coherence, the coherence between
the drives and the principal drives' powers, with the summary in the
pane.

A reading of a coherence that exists, so a toggle on the bar with no
act, offered on such a coherence and nowhere else.
"""

from __future__ import annotations

import numpy as np

import visualdynamics
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.data import MultipleCoherence

RATE = 1024.0


def _plots(pane):
    """The plots the pane draws, its legends aside."""
    import pyqtgraph as pg

    return [item for item in pane.graphics.ci.items
            if isinstance(item, pg.PlotItem)]


def _record(drives=2, coupled=False):
    rng = np.random.default_rng(31)
    n = 1 << 14
    x = rng.standard_normal((drives, n))
    if coupled and drives > 1:
        x[1] = x[0] + 0.1 * rng.standard_normal(n)
    y = np.vstack([x[0] + 0.3 * x[-1] + 0.2 * rng.standard_normal(n),
                   0.5 * x[0] - x[-1] + 0.2 * rng.standard_normal(n)])
    history = visualdynamics.TimeHistory(
        np.arange(n) / RATE, np.vstack([x, y]),
        response_dof=[f'{k + 1}Z+' for k in range(drives)]
        + ['101Z+', '102Z+'],
        ordinate_dim=['force'] * drives + ['acceleration'] * 2)
    history.averaging = Averaging(frame_length=512, overlap=0.5,
                                  window='hann', frames=60)
    return history


def _coherence(window, pump, **kwargs):
    window.add_object('Run', _record(**kwargs))
    name = window.project.compute_multiple_coherence('Run')
    window.show_object(name)
    pump()
    window.render_current()
    pump()
    return name, window.data_pane


def _reading_on(window, pump, **kwargs):
    name, pane = _coherence(window, pump, **kwargs)
    pane.partial_coherence_action.trigger()
    pump()
    return name, pane


def test_the_reading_is_offered_on_a_multiple_coherence_with_drives(window,
                                                                     pump):
    name, pane = _coherence(window, pump)
    assert pane.partial_coherence_action.isVisible()
    # the same numbers imported: no history behind them to condition
    coherence = window.project[name]
    window.add_object('Imported', MultipleCoherence(
        coherence.abscissa, coherence.ordinate,
        response_dof=list(coherence.response_dof)))
    window.show_object('Imported')
    pump()
    window.render_current()
    pump()
    assert not pane.partial_coherence_action.isVisible()


def test_one_drive_has_nothing_more_to_read(window, pump):
    _name, pane = _coherence(window, pump, drives=1)
    assert not pane.partial_coherence_action.isVisible(), \
        'with one drive the partial coherence is the multiple'


def test_the_toggle_draws_three_plots_and_the_pane(window, pump):
    name, pane = _reading_on(window, pump)
    assert pane.partial_coherence_panel.isVisible()
    plots = _plots(pane)
    assert len(plots) == 3
    names = [item.name() for item in plots[0].listDataItems()]
    assert 'multiple, every reference' in names
    assert 'partial with 1Z+, the others removed' in names
    panel = pane.partial_coherence_panel
    assert panel.derived['response'].text() == '101Z+'
    assert panel.derived['references'].text() == '1Z+, 2Z+'
    assert panel.derived['explains'].text() == '1Z+'
    assert panel.derived['drives'].text() == '2 of 2'
    assert '1Z+' in panel.derived['partials'].text()
    assert any(line.startswith('visualdynamics.plot.plot_partial_coherence('
                               f'project.read_partial_coherence({name!r})')
               for line in window.project.journal)


def test_the_response_follows_the_record_picked(window, pump):
    name, pane = _reading_on(window, pump)
    window._select_records(name, [1])
    window.render_current()
    pump()
    assert pane.partial_coherence_panel.derived['response'].text() == '102Z+'
    assert pane.partial_coherence_panel.derived['explains'].text() == '2Z+'


def test_coupled_drives_are_named_in_the_pane(window, pump):
    _name, pane = _reading_on(window, pump, coupled=True)
    panel = pane.partial_coherence_panel
    assert panel.derived['coupled'].text().startswith('1Z+, 2Z+: 0.9')
    assert 'cannot tell them apart' in panel.notes.text()


def test_put_away_the_coherence_is_drawn_again(window, pump):
    _name, pane = _reading_on(window, pump)
    pane.partial_coherence_action.trigger()
    pump()
    assert not pane.partial_coherence_panel.isVisible()
    plots = _plots(pane)
    assert len(plots) != 3 or not any(
        'multiple, every reference' == item.name()
        for item in plots[0].listDataItems())
