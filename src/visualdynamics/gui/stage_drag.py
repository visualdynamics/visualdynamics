"""Dragging the averaging span, the truncation span and the shock
windows on the 3-D stage.

The handles are spheres `viz.marks` puts on each slab's floor edge: the
left edge, the center and the right edge, which resize, move and
resize, as in the 2-D region. This class turns interactor events into
those drags. The averaging and shock drags are settled through the same
rules the 2-D overlays use (`core.averaging.from_span`,
`core.shocks.drag_settled`), so the two editors cannot disagree about
what a drag means. A truncation drag is kept inside the record's first
and last times.

The class is split into a layer that says what a drag means (`begin`,
`drag_to`, `finish`) and the VTK code that feeds it. The tests drive
the first layer, because a synthetic mouse cannot reach an offscreen
interactor. While a drag is live the interactor style is removed, so
the camera does not orbit under the handle, and the press is aborted in
this class's observer so the style never starts a rotation.

During a drag only the marks are redrawn (their actors are replaced by
name, a few hundred points), and the change is committed on release, so
the expensive recompute happens once, at the end.
"""

from __future__ import annotations

import re
from typing import Any

from ..core.averaging import Averaging, from_span
from ..core.shocks import Shock, drag_settled

# Committing only on release is the rule the modal fit's drags follow
# too: the expensive recompute happens once, at the end of the drag.

_HANDLE = re.compile(
    r'^marks-(?:(averaging)|(truncation)|shock-(\d+))-handle-'
    r'(start|move|stop|open|close)$')


class StageMarksDragger:
    """One per stage plotter, armed with whatever the marks currently
    mean."""

    def __init__(self, plotter: Any) -> None:
        self.plotter: Any = plotter
        self._context: dict[str, Any] | None = None
        self._grab: dict[str, Any] | None = None
        self._style = None
        iren = getattr(plotter, 'iren', None)
        self._iren = getattr(iren, 'interactor', None)
        if self._iren is not None:
            # priority above the interactor style's, so a grabbed press
            # can be aborted before the style starts a camera rotation
            self._tags = [
                self._iren.AddObserver('LeftButtonPressEvent',
                                       self._pressed, 10.0),
                self._iren.AddObserver('MouseMoveEvent',
                                       self._moved, 10.0),
                self._iren.AddObserver('LeftButtonReleaseEvent',
                                       self._released, 10.0)]

    # ---- what the marks mean right now ----------------------------------

    def arm_averaging(self, averaging: Averaging, sample_rate: float,
                      samples: int, extents, preview, commit,
                      origin: float = 0.0) -> None:
        """Arm for an averaging span. `preview(averaging)` redraws the
        marks, and `commit(averaging)` is the window's own drag handler.
        `origin` is the record's first time: the handles are on the
        record's clock, and the averaging counts from its beginning."""
        self._context = {
            'kind': 'averaging', 'averaging': averaging,
            'rate': float(sample_rate), 'samples': int(samples),
            'extents': tuple(extents), 'preview': preview,
            'commit': commit, 'origin': float(origin)}

    def arm_truncation(self, truncation, first: float, last: float,
                       extents, preview, commit) -> None:
        """Arm for a truncation span. `preview(truncation)` redraws the
        marks, and `commit(truncation)` is the window's own drag
        handler. `first` and `last` are the record's first and last
        times, which a drag cannot go past."""
        self._context = {
            'kind': 'truncation', 'truncation': truncation,
            'first': float(first), 'last': float(last),
            'extents': tuple(extents), 'preview': preview,
            'commit': commit}

    def arm_shocks(self, shocks, common: bool, limit, extents,
                   preview, commit, origin: float = 0.0) -> None:
        """Arm for shock windows. `preview(shocks)` redraws the marks,
        and `commit(shocks)` stores the settled series. `common` and
        `limit` are passed to `core.shocks.drag_settled`, and `origin`
        is as for `arm_averaging`."""
        self._context = {
            'kind': 'shocks', 'shocks': tuple(shocks),
            'common': bool(common), 'limit': limit,
            'extents': tuple(extents), 'preview': preview,
            'commit': commit, 'origin': float(origin)}

    def disarm(self) -> None:
        self._context = None
        self._grab = None

    # ---- the semantics (what the tests drive) ----------------------------

    def begin(self, name: str, at_seconds: float | None = None) -> bool:
        """Grab the named handle. Returns True when `name` is a handle
        of what the dragger is armed for. `at_seconds` is where the
        pointer took hold; it defaults to the middle of the span."""
        context = self._context
        if context is None:
            return False
        matched = _HANDLE.match(name or '')
        if matched is None:
            return False
        averaging_kind, truncation_kind, index, role = matched.groups()
        handle_kind = ('averaging' if averaging_kind else
                       'truncation' if truncation_kind else 'shocks')
        if handle_kind != context['kind']:
            return False        # a stale handle from another view
        origin = context.get('origin', 0.0)
        if context['kind'] == 'averaging':
            averaging = context['averaging']
            rate = context['rate']
            low = origin + averaging.start_sample(rate) / rate
            high = averaging.stop(rate, origin)
        elif context['kind'] == 'truncation':
            truncation = context['truncation']
            low, high = truncation.start, truncation.stop
        else:
            index = int(index)
            if index >= len(context['shocks']):
                return False
            shock = context['shocks'][index]
            low, high = origin + shock.start, origin + shock.stop
        self._grab = {
            'role': role, 'index': None if index is None else int(index),
            'low': low, 'high': high,
            'anchor': (low + high) / 2.0 if at_seconds is None
            else float(at_seconds)}
        return True

    def _span_for(self, seconds: float) -> tuple[float, float]:
        """The span the pointer at `seconds` asks for."""
        grab = self._grab
        low, high = grab['low'], grab['high']
        least = 1e-9
        if grab['role'] in ('start', 'open'):
            return min(float(seconds), high - least), high
        if grab['role'] in ('stop', 'close'):
            return low, max(float(seconds), low + least)
        delta = float(seconds) - grab['anchor']
        return low + delta, high + delta

    def _proposal(self, seconds: float):
        context, low_high = self._context, self._span_for(seconds)
        origin = context.get('origin', 0.0)
        if context['kind'] == 'averaging':
            return from_span(context['averaging'], *low_high,
                             context['rate'], context['samples'],
                             origin=origin)
        if context['kind'] == 'truncation':
            return self._truncation_for(*low_high)
        low, high = (v - origin for v in low_high)
        index = self._grab['index']
        raw = Shock(max(low, 0.0), max(high - max(low, 0.0), 1e-9))
        return tuple(raw if k == index else shock
                     for k, shock in enumerate(context['shocks']))

    def _truncation_for(self, low: float, high: float):
        """The span a truncation drag asks for, kept inside the
        record: an edge clamps to the walls, and a slide that runs
        out of record stops at the wall with its width kept."""
        from ..core.truncate import Truncation

        context = self._context
        first, last = context['first'], context['last']
        if self._grab['role'] == 'move':
            width = high - low
            low = min(max(low, first), last - width)
            high = low + width
        least = (last - first) * 1e-3
        low = min(max(low, first), last - least)
        high = max(min(high, last), low + least)
        return Truncation(low, high)

    def drag_to(self, seconds: float) -> None:
        """The pointer moved: redraw the marks where it asks."""
        if self._grab is None or self._context is None:
            return
        self._context['preview'](self._proposal(seconds))

    def finish(self, seconds: float) -> None:
        """The pointer was released: settle the drag and commit it
        through the shared rules, or put the marks back when the drag
        asked for something impossible."""
        grab, context = self._grab, self._context
        if grab is None or context is None:
            self._grab = None
            return
        try:
            if context['kind'] in ('averaging', 'truncation'):
                context['commit'](self._proposal(seconds))
                return
            low, high = self._span_for(seconds)
            settled = drag_settled(context['shocks'], grab['index'],
                                   low, high, context['common'],
                                   context['limit'])
            if settled is None:
                # nothing changed, or no room: the preview may have
                # moved the slab, so the marks go back to the stored
                context['preview'](context['shocks'])
            else:
                context['commit'](settled)
        finally:
            # the commit re-renders, which re-arms through
            # _stage_marks; released after so a half-finished drag
            # can never read a cleared grab
            self._grab = None

    # ---- the VTK plumbing -------------------------------------------------

    def _handle_under(self, x: int, y: int) -> str | None:
        import vtk

        picker = vtk.vtkPropPicker()
        if not picker.Pick(x, y, 0, self.plotter.renderer):
            return None
        picked = picker.GetActor()
        if picked is None:
            return None
        for name, actor in self.plotter.renderer.actors.items():
            if actor is picked and _HANDLE.match(name):
                return name
        return None

    def _pointer_seconds(self) -> float | None:
        """The pointer's position, projected onto the time axis.

        The display point at two depths gives the mouse ray; the
        closest point between that ray and the handles' own line (the
        stage's front floor edge, direction +x) names a stage x, and
        the extents turn it back into seconds.
        """
        import numpy as np

        renderer = self.plotter.renderer
        x, y = self._iren.GetEventPosition()
        ends = []
        for depth in (0.0, 1.0):
            renderer.SetDisplayPoint(float(x), float(y), depth)
            renderer.DisplayToWorld()
            wx, wy, wz, w = renderer.GetWorldPoint()
            if w == 0:
                return None
            ends.append(np.array([wx / w, wy / w, wz / w]))
        origin, direction = ends[0], ends[1] - ends[0]
        norm = float(np.linalg.norm(direction))
        if norm == 0:
            return None
        direction /= norm
        # closest approach between the ray and the axis y=z=0, d1=+x
        b = float(direction[0])
        w0 = -origin
        d = float(w0[0])
        e = float(w0 @ direction)
        denominator = 1.0 - b * b
        if abs(denominator) < 1e-12:
            return None       # sighting straight down the axis
        s = (b * e - d) / denominator
        from ..viz.waterfall import STAGE

        x0, x1, *_rest = self._context['extents']
        return x0 + s / STAGE[0] * (x1 - x0)

    def _pressed(self, caller, _event) -> None:
        if self._context is None or self._iren is None:
            return
        x, y = self._iren.GetEventPosition()
        name = self._handle_under(x, y)
        if name is None:
            return
        seconds = self._pointer_seconds()
        if seconds is None or not self.begin(name, seconds):
            return
        # the style must not also read this press as the start of a
        # camera rotation: abort the event, and stand the style down
        # for the drag so the moves stay ours too
        command = caller.GetCommand(self._tags[0])
        if command is not None:
            command.AbortFlagOn()
        self._style = self._iren.GetInteractorStyle()
        self._iren.SetInteractorStyle(None)

    def _moved(self, _caller, _event) -> None:
        if self._grab is None:
            return
        seconds = self._pointer_seconds()
        if seconds is not None:
            self.drag_to(seconds)

    def _released(self, _caller, _event) -> None:
        if self._grab is None:
            return
        if self._style is not None:
            self._iren.SetInteractorStyle(self._style)
            self._style = None
        seconds = self._pointer_seconds()
        self.finish(seconds if seconds is not None
                    else (self._grab['low'] + self._grab['high']) / 2.0)
