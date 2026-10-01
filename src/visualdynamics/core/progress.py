"""Progress through a long computation, told to a callback.

One shape for every long verb (2026-10-01): a callable told
``(done, total)`` as the work advances. The window feeds its strip bar
with it; a script prints it. Each stage of a computation adds its
count as it is sized — a matched filter per tone, an assembly per
element, a solve's pieces per channel — so the total grows by stage
and a bar is determinate within one. Anything the callback raises
propagates out of the computation: a cancel is the callback raising.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class Ticker:
    """`add` a stage's count, `tick` as it goes; the callback hears
    `(done, total)` at most every `interval` seconds and always on the
    last tick, since a window pumps its events in it. A `Ticker(None)`
    costs nothing and tells no one."""

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
        """A stage whose length is an estimate — the eigen iterations —
        keeps the bar short of full: the total is raised to `count`
        past what is done whenever it would be reached."""
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
