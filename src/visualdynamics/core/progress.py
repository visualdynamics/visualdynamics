"""Progress through a long computation, reported to a callback.

Every long verb reports progress the same way: a callable is called
with ``(done, total)`` as the work goes on. The window uses it to
update its progress bar, and a script can print it. Each stage of a
computation adds its count once that count is known (a matched filter
per tone, an assembly per element, a solve's pieces per channel), so
the total increases stage by stage and the bar is accurate within
each stage. Anything the callback raises passes out of the
computation, which is how a cancel works: the callback raises.

A stage that cannot say how far along it is, one long call into a
library, is announced with a total of `BUSY`: ``(done, BUSY)`` means
work is going on and no count can be given, and the window shows a
moving bar until the next ordinary report.
"""

from __future__ import annotations

import time
from collections.abc import Callable

# One progress callback shape for every long verb (2026-10-01).


#: the total that says a stage is running and cannot count itself
BUSY = -1


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

    def busy(self) -> None:
        """Say that a stage is starting that cannot report its progress:
        the callback is told ``(done, BUSY)`` at once. The next `tick`
        or `add` reports the count again."""
        if self.callback is not None:
            self._told = time.monotonic()
            self.callback(self.done, BUSY)

    def extend_to(self, count: int) -> None:
        """Raise the total to `count` past what is done, if it is lower.

        Used for a stage whose length is only an estimate (the
        eigenvalue iterations), so the bar does not reach full before
        the stage ends.
        """
        if self.done + int(count) > self.total:
            self.total = self.done + int(count)
            self._tell()

    def share(self, units: int) -> Share:
        """A stage given a fixed `units` of this ticker's total, whatever
        its own count turns out to be.

        Stages counted in their own pieces are not comparable: a model's
        assembly ticks once per element and its eigen iterations once
        per solve, so 190,000 elements left a few hundred solves a
        sliver of the bar, and the bar read full for the whole of the
        solve. Each share is added to the total at once, so the bar's
        length is known from the start, and the stage's own count is
        spread over its units.
        """
        return Share(self, units)

    def _tell(self) -> None:
        if self.callback is None:
            return
        now = time.monotonic()
        if self.done >= self.total or now - self._told >= self.interval:
            self._told = now
            self.callback(self.done, self.total)


class Share:
    """A stage's own count, spread over a fixed number of its parent's
    units (`Ticker.share`). It answers `add`, `tick`, `extend_to` and
    `busy` as a `Ticker` does, so a stage takes either. The parent is
    never ticked back: a stage whose estimate grows holds the bar where
    it is until its count catches up."""

    def __init__(self, parent: Ticker, units: int) -> None:
        self.parent, self.units = parent, int(units)
        self.done: int = 0
        self.total: int = 0
        self._given = 0
        parent.add(self.units)

    def add(self, count: int) -> None:
        self.total += int(count)
        self._pass()

    def tick(self, count: int = 1) -> None:
        self.done += int(count)
        self._pass()

    def extend_to(self, count: int) -> None:
        if self.done + int(count) > self.total:
            self.total = self.done + int(count)
            self._pass()

    def busy(self) -> None:
        self.parent.busy()

    def _pass(self) -> None:
        reached = (self.units * min(self.done, self.total) // self.total
                   if self.total > 0 else 0)
        # ticked even when nothing is owed, so a count after `busy`
        # brings the bar back from its moving state
        owed = max(0, reached - self._given)
        self._given += owed
        self.parent.tick(owed)
