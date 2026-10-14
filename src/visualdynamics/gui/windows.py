"""A pane in a window of its own.

`geometry.plot()` from a script opens the same 3-D view the app shows,
with the same bar over it, so the labeled axes and the orientation triad
are toggles instead of arguments to know about in advance. Each pane
owns its controls, so the window and a one-line script show the same
thing.

Nothing here is imported unless a plot is shown on screen. Rendering to
a file does not use this module, so a headless run never builds a
widget.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..plot import _application
from .panes import DataPane, ScenePane


def _run(app: Any, widget: Any, show: bool) -> Any:
    widget.raise_()
    widget.activateWindow()
    # inside the app's own event loop the window is just another window;
    # from a script it is the only one, so this call is what keeps it up
    if show and not getattr(app, '_vibe_loop_running', False):
        app.exec()
    return widget


def scene_window(draw: Callable[[Any], Any], *, theme: Any = None,
                 title: str | None = None, axis_unit: str = '',
                 size: tuple[int, int] = (1000, 700),
                 show: bool = True, view: Any = None) -> ScenePane:
    """Draw into a ScenePane of its own and show it on screen.

    `draw(plotter)` adds whatever is being shown. `axis_unit` is the unit
    the labeled axes show; the pane needs it to label them again when
    the bar's toggles change. `view` is the geometry's `View` the window
    opens on and Reset View returns to. Without one (a MAC's bars, a
    waterfall), the camera keeps the direction `draw` gave it and is
    fitted to the scene. With
    `show` and no event loop already running, this runs the event loop
    and returns when the window is closed. Returns the pane; its
    `.plotter` is the PyVista plotter.
    """
    app = _application()
    pane = ScenePane(theme)
    pane.setWindowTitle(title or '3D View')
    pane.resize(*size)
    # on screen before the VTK widget is built: created earlier on macOS
    # it leaves the window 0x0 and never shown
    pane.show()
    app.processEvents()
    pane.create_plotter()
    pane.axis_unit = axis_unit
    draw(pane.plotter)
    pane.apply_annotations()
    pane.set_home_view(view)
    if view is None:
        pane.plotter.reset_camera()
    else:
        pane.reset_view()
    pane.plotter.render()
    return _run(app, pane, show)


def data_window(draw: Callable[[Any], Any], *, theme: Any = None,
                title: str | None = None, complex_data: bool = False,
                size: tuple[int, int] = (1000, 700),
                on_pane: Callable[[DataPane], Any] | None = None,
                show: bool = True) -> DataPane:
    """Draw into a DataPane of its own and show it on screen.

    `draw(graphics)` builds the plot. With `complex_data` the component
    box is shown, and choosing a component redraws the plot, the same
    control the app shows over an FRF. The rest of the app's bar is left
    out because none of it applies here: filtering to drive points is a
    record selection, and Edit Fit opens a fitting session.

    `on_pane` is called with the pane before anything is drawn, so a
    caller whose drawing depends on a control (which component to plot)
    can read it on the first pass as well as on every redraw. `show`
    behaves as in `scene_window`. Returns the pane.
    """
    app = _application()
    pane = DataPane(theme)
    if on_pane is not None:
        on_pane(pane)
    pane.setWindowTitle(title or 'Plot')
    pane.resize(*size)

    def redraw() -> None:
        pane.graphics.clear()
        draw(pane.graphics)

    pane.reread.connect(redraw)
    draw(pane.graphics)
    pane.show_controls(map_wanted=None, diagonal=None, cmif=False,
                       complex_data=complex_data, pair=False)
    pane.show()
    return _run(app, pane, show)
