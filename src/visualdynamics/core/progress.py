"""Progress through a long computation, reported to a callback.

Every long verb reports progress the same way: a callable is called
with ``(done, total)`` as the work goes on. The window uses it to
update its progress bar, and a script can print it. Each stage of a
computation adds its count once that count is known (a matched filter
per tone, an assembly per element, a solve's pieces per channel), so
the total increases stage by stage and the bar is accurate within
each stage. Anything the callback raises passes out of the
computation, which is how a cancel works: the callback raises.
"""

from __future__ import annotations

import time
from collections.abc import Callable

# One progress callback shape for every long verb (2026-10-01).


class Cancelled(Exception):
    """Raised by a progress callback to stop the computation it reports on.

    The window's Cancel button sets a flag that the callback reads on
    the next tick. Nothing between the callback and the caller of the
    computation catches this exception.
    """


class Ticker:
    """Counts progress and reports it to a callback.

    Call `add` with a stage's count when it is known and `tick` as each
    piece finishes. The callback is called with ``(done, total)`` at
    most once every `interval` seconds (default 0.05 s), and always when
    `done` reaches `total`, because a window processes its events
    inside the callback. `Ticker(None)` does nothing.
    """

    def __init__(self, callback: Callable[[int, int], None] | None,
                 interval: float = 0.05) -> None:
        self.callback, self.interval = callback, float(interval)
        self.done: int = 0
        self.total: int = 0
        self._told = -1.0

    def add(self, count: int) -> None:
        self.total += int(count)
        self._tell()

    def tick(self, count: int = 1) -> None:
        self.done += int(count)
        self._tell()

    def extend_to(self, count: int) -> None:
        """Raise the total to `count` past what is done, if it is lower.

        Used for a stage whose length is only an estimate (the
        eigenvalue iterations), so the bar does not reach full before
        the stage ends.
        """
        if self.done + int(count) > self.total:
            self.total = self.done + int(count)
            self._tell()

    def _tell(self) -> None:
        if self.callback is None:
            return
        now = time.monotonic()
        if self.done >= self.total or now - self._told >= self.interval:
            self._told = now
            self.callback(self.done, self.total)
