"""The 3-D scene: geometry, DOF arrows and moving mode shapes.

PyVista over VTK, shown in the app's own pane or rendered
off-screen to an image. The plots on an axis are `visualdynamics.plot`;
this is the half that has a camera in it.
"""

from .geometry import geometry_scene, plot_geometry
from .waterfall import plot_waterfall, waterfall_scene


def undeferred(plotter):
    """Make this Qt plotter render synchronously, and return it.

    On macOS pyvistaqt wraps ``render()`` in a worker thread that emits
    a signal back to the GUI thread, so every render becomes a queued
    metacall delivered later by the event loop. That indirection exists
    to make ``render()`` safe to call *from* another thread — which
    nothing here ever does — and it opens a race that nothing here can
    close: if the widget's native window goes away between the emit and
    the delivery, the queued call walks into ``vtkCocoaRenderWindow``
    with a dead ``QPlatformWindow`` and the process dies with SIGSEGV
    (seen in the field: ``QPlatformWindow::window()`` dereferencing
    null out of ``NSOpenGLContext update``).

    Calling the unthreaded path directly is what the queued delivery
    would have done, minus the window between emit and delivery in
    which the target can die. ``_rendered`` is forced the way the
    threaded override forces it, because ``BasePlotter.render`` skips
    the flag when it skips the render, and pyvistaqt marked that flag
    "crucial".

    Bound on the instance at construction rather than patched on the
    class, so a pyvistaqt used by anything else in the process keeps
    its own behavior.
    """
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
