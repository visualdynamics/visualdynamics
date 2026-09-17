"""A pane in a window of its own.

`geometry.plot()` from a script opens the same 3-D view the app shows,
with the same bar over it — the labeled axes and the orientation triad
are toggles rather than arguments you have to know about beforehand.
That is the whole point of the panes owning their own controls: the
window and a one-line script put up the same thing.

Nothing here is imported unless a plot is actually shown. Rendering to a
file never comes through here, so a headless run never builds a widget.
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
                 show: bool = True) -> ScenePane:
    """Draw into a ScenePane of its own and put it on screen.

    `draw(plotter)` adds whatever is being shown. `axis_unit` is what the
    labeled axes say, which the pane needs to re-annotate when the bar's
    toggles move. Returns the pane; its `.plotter` is the PyVista one.
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
    pane.plotter.reset_camera()
    pane.plotter.render()
    return _run(app, pane, show)


def data_window(draw: Callable[[Any], Any], *, theme: Any = None,
                title: str | None = None, complex_data: bool = False,
                size: tuple[int, int] = (1000, 700),
                on_pane: Callable[[DataPane], Any] | None = None,
                show: bool = True) -> DataPane:
    """Draw into a DataPane of its own and put it on screen.

    `draw(graphics)` builds the plot. With `complex_data` the component
    box is offered, and choosing a part redraws — the same control the
    app puts over an FRF. The rest of the app's bar is absent because it
    is not a live option here: filtering to drive points is a record
    selection, and Edit Fit opens a fitting session.

    `on_pane` is handed the pane before anything is drawn, so a caller
    whose drawing depends on a control — which component to plot — can
    read it on the first pass as well as on every redraw.
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
