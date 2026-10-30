"""The controller-file watch in the window: `core.run_log_files` on a
thread of its own, its proposals carried to the window's thread.

The long-verb runner (`MainWindow._run_long`) shuts out the keyboard
and mouse while it works, which is right for a verb the user is waiting
on and wrong here: the log keeper types the next entry while a stream
of many gigabytes is hashed. So the watch runs on a thread that gates
nothing, and what it finds comes back as queued signals, delivered on
the window's thread between whatever else the window is doing, the way
the update check's answer comes back.

The watch stops when the window closes, before Qt destroys anything, so
no thread is left emitting into a deleted object.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import QEvent, QFileSystemWatcher, QObject, Signal

from ..core.run_log_files import ControllerWatch


class ControllerFilesWatch(QObject):
    """A `ControllerWatch` running for a window.

    `proposed` carries each `Proposal` and `problem` each
    `WatchProblem`, emitted on the watch's thread and delivered on the
    window's. A notice from the operating system that a folder changed
    makes the next look come at once; the folders are still listed every
    `poll_s` seconds, since a network share often sends no notice.

    Made with the widget it serves as its parent (a dock, the window),
    so it is deleted with it; it stops when that widget's window closes.
    The keyword settings are `ControllerWatch`'s.

    Parameters
    ----------
    folders : str, os.PathLike, or a sequence of them
        The controller output folders.
    parent : QObject
        The widget it serves.
    **settings
        `ControllerWatch`'s settings: settle_s, poll_s, rate, limit,
        existing.

    Raises
    ------
    ControllerFilesError
        For a folder that does not exist, or a setting it cannot take,
        before anything is made.
    """

    proposed = Signal(object)
    problem = Signal(object)

    def __init__(self, folders: str | os.PathLike
                 | Sequence[str | os.PathLike], parent: QObject,
                 **settings: Any) -> None:
        # refused before any Qt object exists, so a missing folder leaves
        # nothing behind to tear down
        watch = ControllerWatch(folders, **settings)
        super().__init__(parent)
        self.watch: ControllerWatch = watch
        self._notices = QFileSystemWatcher(list(watch.folders), self)
        self._notices.directoryChanged.connect(self._changed)
        window = parent.window() if hasattr(parent, 'window') else parent
        self._window = window
        window.installEventFilter(self)
        watch.start(on_proposal=self.proposed.emit,
                    on_problem=self.problem.emit)

    def _changed(self, _folder: str) -> None:
        self.watch.poke()

    @property
    def running(self) -> bool:
        """Whether the watch's thread is running."""
        return self.watch.running

    def archive(self) -> None:
        """Hash every file seen and not yet proposed, on the watch's
        thread, at once: what a test's close asks. The proposals come
        through `proposed` as any other."""
        self.watch.ask_archive()

    def stop(self) -> bool:
        """End the watch's thread and stop listening for notices; True
        when the thread has ended. Safe to call again."""
        if self._window is not None:
            self._notices.directoryChanged.disconnect(self._changed)
            paths = self._notices.directories()
            if paths:
                self._notices.removePaths(paths)
            self._window.removeEventFilter(self)
            self._window = None
        return self.watch.stop()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Stop with the window: on its close, before Qt destroys it."""
        if watched is self._window and event.type() == QEvent.Type.Close:
            self.stop()
        return False
