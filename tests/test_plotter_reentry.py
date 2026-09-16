"""The 3-D view is built once, even when its own construction shows the
window again.

On Windows, making the VTK view's native window re-shows the main
window, whose showEvent asks the scene pane for a plotter again while
the first QtInteractor is half-built — and the first launch of
0.1.0a1 died in that loop, reading `renderers` off an interactor that
had none yet (the release smoke test, 2026-09-14). The pane refuses
the nested request; only the outer construction lands.
"""

from __future__ import annotations


def test_a_nested_request_while_building_answers_none(window, monkeypatch):
    from visualdynamics.gui import panes

    pane = window.scene
    pane.plotter = None             # as on screen: not built yet
    seen = {'nested': None, 'built': 0}

    class Interactor:
        def __init__(self, parent):
            seen['built'] += 1
            # what Windows does: the window shows again mid-construction
            seen['nested'] = pane.create_plotter()
            self.renderers = []

        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr('pyvistaqt.QtInteractor', Interactor)
    monkeypatch.setattr(panes, 'undeferred', lambda plotter: plotter,
                        raising=False)
    try:
        pane.create_plotter()
    except AttributeError:
        pass                    # the fake stops at the first real call
    assert seen['built'] == 1, 'one construction, not one per show'
    assert seen['nested'] is None, 'the nested request was refused'
    # the guard is for the *duration* of a construction: one that
    # failed must not leave the pane refusing every later request
    assert pane._creating_plotter is False, 'a failed construction latched the guard'

    class Anything:
        """Answers every attribute and call with itself: the rest of
        `create_plotter` wires the real view up, and none of it is
        what this test is about."""
        def __getattr__(self, name):
            return self

        def __call__(self, *args, **kwargs):
            return self

    from PySide6.QtWidgets import QWidget

    class Working(QWidget):
        """A real widget, because the pane puts the view into its
        layout; everything VTK-shaped answers with `Anything`."""
        def __init__(self, parent):
            super().__init__(parent)
            self.renderers = []

        def __getattr__(self, name):
            return Anything()

    monkeypatch.setattr('pyvistaqt.QtInteractor', Working)
    pane.plotter = None
    assert pane.create_plotter() is not None, 'the next request builds'
