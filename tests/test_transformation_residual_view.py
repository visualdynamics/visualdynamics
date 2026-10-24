"""The rigid-body residual reading in the window: on a density
transformed through six rigid-body shapes, the measured power at the
channels, the part the six motions explain and the residual, with the
transformation's conditioning and where it holds in the pane.

A reading of a transform that exists, so a toggle on the bar with no
act, offered on such a density and nowhere else.
"""

from __future__ import annotations

import numpy as np

import visualdynamics
from visualdynamics.core.averaging import Averaging
from visualdynamics.core.rigid import MassProperties, rigid_body_shapes


def _plots(pane):
    """The plots the pane draws, its legends aside."""
    import pyqtgraph as pg

    return [item for item in pane.graphics.ci.items
            if isinstance(item, pg.PlotItem)]


def _objects(flex_above=500.0):
    rng = np.random.default_rng(5)
    geometry = visualdynamics.Geometry(np.arange(101, 106),
                                       rng.uniform(-1.0, 1.0, (5, 3)),
                                       length_unit='m')
    rigid = rigid_body_shapes(geometry, MassProperties((0.0, 0.0, 0.0)))
    n, rate = 16384, 2048.0
    q = rng.standard_normal((6, n))
    noise = rng.standard_normal((15, n))
    spectrum = np.fft.rfft(noise, axis=1)
    spectrum[:, np.fft.rfftfreq(n, 1.0 / rate) < flex_above] = 0.0
    u = rigid.shape_matrix.T @ q + 2.0 * np.fft.irfft(spectrum, n, axis=1)
    run = visualdynamics.TimeHistory(np.arange(n) / rate, u,
                                     response_dof=list(rigid.coordinate),
                                     ordinate_dim='acceleration')
    run.averaging = Averaging(frame_length=1024, frames=15)
    return rigid, run


def _transformed(window, pump, density=True):
    rigid, run = _objects()
    window.add_object('Run', run)
    window.add_object('Rigid', rigid)
    source = window.project.compute_cpsds('Run') if density else 'Run'
    name = window.project.transform(source, 'Rigid')
    window.show_object(name)
    pump()
    window.render_current()
    pump()
    return name, window.data_pane


def test_the_reading_is_offered_on_a_transformed_density(window, pump):
    _name, pane = _transformed(window, pump)
    assert pane.transformation_action.isVisible()


def test_a_transformed_time_history_has_no_density_to_split(window, pump):
    _name, pane = _transformed(window, pump, density=False)
    assert not pane.transformation_action.isVisible()


def test_the_toggle_draws_the_split_and_the_pane(window, pump):
    name, pane = _transformed(window, pump)
    pane.transformation_action.trigger()
    pump()
    panel = pane.transformation_panel
    assert panel.isVisible()
    plots = _plots(pane)
    assert len(plots) == 2
    names = [item.name() for item in plots[0].listDataItems()]
    assert names == ['measured, summed over the inputs',
                     'explained by the outputs',
                     'residual, what the outputs leave']
    assert panel.derived['channels'].text() == '15 to 6 motions'
    reading = window.project.read_transformation(name)
    assert panel.derived['condition'].text() == \
        f'{reading.transformation.condition:.3g}'
    holds = panel.derived['holds'].text()
    assert holds.endswith(' Hz') and 480.0 <= float(holds[:-3]) <= 500.0
    assert any(line.startswith(
        'visualdynamics.plot.plot_transformation_residual('
        f'project.read_transformation({name!r})')
        for line in window.project.journal)
    # and put away, the panel goes with it
    pane.transformation_action.trigger()
    pump()
    assert not panel.isVisible()


def test_the_length_scale_is_in_the_display_unit(window, pump):
    from visualdynamics.units import SYSTEMS

    _name, pane = _transformed(window, pump)
    english = next(system for system in SYSTEMS.values()
                   if system.label_text('length') == 'in')
    window.unit_system = english
    pane.transformation_action.trigger()
    pump()
    text = pane.transformation_panel.derived['length'].text()
    assert text.endswith(' in')


def test_the_residual_is_read_off_the_window_thread_and_held(window, pump,
                                                            monkeypatch):
    """Splitting a long record's matrix is a second and more: the reading
    is made on the worker, drawn again from the one held while the
    density stands, and read again when it changes."""
    import threading

    threads = []
    real = window.project.read_transformation

    def recorded(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(window.project, 'read_transformation', recorded)
    name, pane = _transformed(window, pump)
    pane.transformation_action.trigger()
    pump()
    assert len(threads) == 1
    assert threads[0] is not threading.main_thread()
    window.render_current()
    pump()
    assert len(threads) == 1, 'the reading held is drawn again'
    density = window.project[name]
    density.ordinate = np.array(density.ordinate)
    window.render_current()
    pump()
    assert len(threads) == 2, 'a density changed is read again'
