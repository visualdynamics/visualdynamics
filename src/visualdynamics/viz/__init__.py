"""The 3-D scene: geometry, DOF arrows and moving mode shapes.

Drawn with PyVista over VTK, either in the app's own pane or off
screen to an image. Plots on a 2-D axis are in `visualdynamics.plot`;
this package is the half with a camera.
"""

from .geometry import geometry_scene, plot_geometry
from .waterfall import plot_waterfall, waterfall_scene


def undeferred(plotter):
    """Make a Qt plotter render synchronously, and return it.

    On macOS pyvistaqt runs ``render()`` through a worker thread that
    signals back to the GUI thread, so every render is queued and
    delivered later by the event loop. That is only needed when
    ``render()`` is called from another thread, which nothing here does,
    and it allows a crash: if the widget's native window is destroyed
    between the signal and its delivery, the queued render reaches
    ``vtkCocoaRenderWindow`` with no window and the process dies with
    SIGSEGV. This replaces ``render`` with a direct call to the
    unthreaded path, and sets ``_rendered`` the way pyvistaqt's threaded
    version does (``BasePlotter.render`` does not set it when it skips a
    render). It also stops the widget's automatic render timer, since
    every render here is explicit.

    The change is made on this instance only, not on the class, so any
    other use of pyvistaqt in the process keeps its own behavior.
    """
    # The SIGSEGV was seen in the field: QPlatformWindow::window()
    # dereferencing null out of NSOpenGLContext update. pyvistaqt's own
    # source calls the _rendered flag "crucial".
    unthreaded = plotter._render

    def render():
        plotter._rendered = True
        unthreaded()

    plotter.render = render
    # And nothing renders behind the app's back: QtInteractor defaults
    # to auto_update=5.0, a QTimer calling render() five times a second
    # for the life of the widget. Every draw here is explicit, so those
    # ticks bought nothing — and the first one to fire after the native
    # window died walked into vtkCocoaRenderWindow with a dead
    # QPlatformWindow, the same SIGSEGV as the threaded render, arriving
    # by timer instead of by queued metacall (Brandon's crash report,
    # 2026-08-30).
    timer = getattr(plotter, 'render_timer', None)
    if timer is not None:
        timer.stop()
    return plotter


__all__ = ['geometry_scene', 'plot_geometry', 'plot_waterfall',
           'undeferred', 'waterfall_scene']
