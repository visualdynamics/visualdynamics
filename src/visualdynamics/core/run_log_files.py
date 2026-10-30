"""Controller files: what a controller writes while a test runs,
proposed for the run log, and the log's files in a report's data index.

A controller writes its records (a run's stream, its spectra, its
settings) into an output folder. `ControllerWatch` watches one or more
such folders and **proposes** each file that appears: it never attaches
anything to the log itself. The log keeper accepts a proposal
(`accept`), which writes an entry naming the file by SHA-256, with its
format and what it holds, in the run that was current when the file
appeared (`place_at`). A proposal not accepted leaves no trace in the
log.

**Only settled files are hashed.** A file is settled when its size and
modification time have stayed the same over `settle_s` seconds, seen on
at least two looks at the folder; a file still being written would hash
to a value its finished bytes do not have. A file that changes while it
is hashed is not proposed and waits to settle again, and one removed
before it settles is forgotten. Hashing is throttled to `rate` bytes a
second, because the file sits on the controller's own disk or share and
the next run is writing there too, and a stream can be tens of
gigabytes. At a test's close everything seen and not yet hashed is
hashed at once (`ControllerWatch.archive`).

**Watched and polled.** The folders are listed every `poll_s` seconds,
whether or not anything says they changed, because a network share
often sends no notice of a change; a front end that is told of a change
(`ControllerWatch.poke`) only makes the next look come sooner. All of
it, looking and hashing, runs on a thread of its own
(`ControllerWatch.start`), which never holds up the program that writes
the log: proposals are handed to a callback or to a queue that is
safe to read from any thread.

**Format and holdings.** Each file is described as a data index
describes a file (`data_index.IndexEntry`): its name, where it is, its
size, its SHA-256, its format and what it holds, read through Visual
Dynamics' own importers. A Rattlesnake run is read from its dimensions
alone (`rattlesnake.stream_summary`), so a run of any size gives its
channels, its sample rate and its stream's length without reading the
stream; another format an importer recognizes is read whole when it is
no larger than `HOLDINGS_LIMIT`; and the run, axis and level its name
gives are added (`runs.read_name`). A format nothing recognizes is
recorded by name, size and hash with the format 'unknown': it is never
refused.

**The stream and the counter.** A Rattlesnake stream's length is
compared with the time the run's entries counted (`check_stream`). The
stream holds the ramps up and down, and any lower levels run inside the
same run, so it is longer than the counter by however long those take;
the comparison is within a stated tolerance, and a difference beyond it
is a finding for the log keeper, never a refusal.

**One data index.** The log's files go into the project's data index
through the index's own entry (`DataIndex.add`, the one the import and
`Project.index_file` use), with the checksum the log recorded, so
nothing is hashed twice and a report lists them beside every other file
it rests on (`index_run_log`). The log's own file goes in with each
test's head hash (`RunLog.head`) in what it holds, which anchors the
chain inside the file to a record outside it.
"""

from __future__ import annotations

import math
import os
import queue
import re
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, NamedTuple

from . import data_index
from .data_index import DataIndex, IndexEntry
from .progress import Cancelled
from .run_log import SUFFIX, Entry, FileRef, RunLog

# One module for watching, describing, accepting and indexing, all on
# plain values: the front end's worker (gui/run_log_watch.py) only runs
# `ControllerWatch` on its thread and carries the proposals back.

#: what the data index records as having added the log's entries
SOURCE = 'run log'

#: the format recorded for a file no importer recognizes
UNKNOWN = 'unknown'

#: the format a run log is recorded under in the data index
LOG_FORMAT = f'Visual Dynamics run log ({SUFFIX})'

#: seconds a file's size and modification time must stay the same
#: before it is hashed
SETTLE_S = 5.0

#: seconds between looks at the folders
POLL_S = 2.0

#: bytes a second hashing reads at most: a 20 GB stream takes about ten
#: minutes, and the controller's next run keeps most of its disk
RATE = 32 * 1024 * 1024

#: the largest file read whole for its holdings; a larger one of a
#: format other than Rattlesnake's is recorded by its format alone
HOLDINGS_LIMIT = 64 * 1024 * 1024

#: seconds a stream may differ from the run's counter before the
#: difference is a finding: room for the ramps up and down
STREAM_TOLERANCE_S = 30.0

#: names in a watched folder that are never proposed: a run log, its
#: lock and its mirror's staging copy, should the log sit in the folder
_IGNORED = (SUFFIX, f'{SUFFIX}.lock', f'{SUFFIX}.partial')


class ControllerFilesError(ValueError):
    """A folder, a setting or a request the controller-file watch cannot
    take; the message names the cause."""


class Place(NamedTuple):
    """Where in the log an entry goes: a test and a run by number, an
    axis by name, any of them None."""

    test: int | None
    axis: str | None
    run: int | None


@dataclass(frozen=True)
class Stream:
    """A Rattlesnake time stream's size, from its dimensions
    (`rattlesnake.stream_summary`): its key, sample rate in hertz,
    samples, length in seconds and channels."""

    key: str
    sample_rate: float
    samples: int
    seconds: float
    channels: int


@dataclass(frozen=True)
class Description:
    """What a file is: its format ('unknown' when nothing recognizes
    it), what it holds in words, its channels, the run identity its name
    gives (`runs.read_name`), a Rattlesnake time stream when it has one,
    and why anything was not read."""

    format: str
    holds: str
    channels: tuple[str, ...] = ()
    run: dict[str, Any] = field(default_factory=dict)
    stream: Stream | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Proposal:
    """A controller file proposed for the log.

    `file` is the file as a data index records it (`IndexEntry`, its
    source the run log), hashed once it settled. `folder` is the watched
    folder it appeared in and `seen` when it was first seen there, with
    its zone; the run current at `seen` is where it is proposed to go
    (`place_at`). `archived` marks a file hashed at a test's close
    rather than once it had settled. The rest is its `Description`.
    """

    file: IndexEntry
    folder: str
    seen: datetime
    channels: tuple[str, ...] = ()
    run: dict[str, Any] = field(default_factory=dict)
    stream: Stream | None = None
    notes: tuple[str, ...] = ()
    archived: bool = False

    @property
    def name(self) -> str:
        """The file's name."""
        return self.file.name

    @property
    def path(self) -> str:
        """Where the file was when it was hashed."""
        return self.file.path

    @property
    def sha256(self) -> str:
        """The file's SHA-256, as 64 lower-case hex digits."""
        return self.file.sha256

    def file_ref(self) -> FileRef:
        """The file as a log entry names it: name, SHA-256, size, format,
        holdings and path."""
        return FileRef(self.file.name, self.file.sha256, self.file.size,
                       self.file.format, self.file.holds, self.file.path)


@dataclass(frozen=True)
class WatchProblem:
    """Something the watch could not do: the folder or file and why.
    The watch goes on; the problem is reported once until it clears."""

    path: str
    reason: str

    def __str__(self) -> str:
        return f'{self.path}: {self.reason}'


@dataclass(frozen=True)
class StreamCheck:
    """A stream's length against the run's counter.

    `ok` is True within the tolerance, False beyond it, and None when
    nothing was compared (no time stream, no run, no counter), which is
    never a pass. `stream_s` and `sample_rate` are the stream's,
    `counted_s` the counter's, `tolerance_s` the difference allowed, and
    `message` says what was found in words."""

    ok: bool | None
    message: str
    tolerance_s: float
    stream_s: float | None = None
    sample_rate: float | None = None
    counted_s: float | None = None


@dataclass(frozen=True)
class IndexedLog:
    """What `index_run_log` put into a data index: the log's own entry,
    each file's entry, and each file not found, with why."""

    log: IndexEntry
    files: tuple[IndexEntry, ...]
    missing: tuple[tuple[FileRef, str], ...]


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _finite(value: Any, what: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value < 0 \
            or (positive and value == 0):
        bound = 'more than 0' if positive else '0 or more'
        raise ControllerFilesError(f'{what} is a number of seconds, '
                                   f'{bound}, not {value!r}')
    return float(value)


# ---- describing a file --------------------------------------------------------


def _channel_words(channels: Sequence[str]) -> str:
    if len(channels) == 1:
        return f'1 channel, {channels[0]}'
    return f'{len(channels)} channels, {channels[0]} to {channels[-1]}'


def _name_words(fields: dict[str, Any]) -> str:
    from .runs import level_text

    words = []
    if 'run' in fields:
        words.append(f'run {fields["run"]}')
    if 'axis' in fields:
        words.append(f'axis {fields["axis"]}')
    if 'level_db' in fields:
        words.append(level_text(fields['level_db']))
    if 'start' in fields:
        words.append(f'started {fields["start"]}')
    return ', '.join(words) + ' (from its name)' if words else ''


def _rattlesnake(path: str) -> tuple[list[str], tuple[str, ...], Stream | None]:
    """(words, channels, first time stream) of a Rattlesnake run, from its
    dimensions alone."""
    from ..io import rattlesnake

    summary = rattlesnake.stream_summary(path)
    channels = tuple(str(dof) for dof in summary['channels'])
    words = []
    try:
        kind = rattlesnake.run_kind(path)
    except (OSError, ValueError, KeyError):
        kind = None
    words.append(f'Rattlesnake {kind} run' if kind else 'Rattlesnake run')
    if channels:
        words.append(_channel_words(channels))
    if summary['sample_rate']:
        words.append(f'{summary["sample_rate"]:.6g} Hz')
    stream = None
    if summary['streams']:
        first = summary['streams'][0]
        stream = Stream(str(first['key']), float(summary['sample_rate']),
                        int(first['samples']), float(first['seconds']),
                        int(first['channels']))
        words.append(f'time stream {stream.seconds:.6g} s')
    else:
        words.append('no time stream')
    return words, channels, stream


def _imported(path: str) -> tuple[list[str], tuple[str, ...]]:
    """(words, channels) of a file read whole through its importer."""
    from .. import io
    from ..project import holds
    from .data import DataArray

    result = io.import_file(path)
    objects = result if isinstance(result, dict) else {'': result}
    channels: list[str] = []
    for obj in objects.values():
        if isinstance(obj, DataArray) and obj.num_records:
            channels += [str(dof) for dof in obj.response_dof]
    said = holds(objects)
    return ([said] if said else []), tuple(dict.fromkeys(channels))


def describe(path: str | os.PathLike, *,
             limit: int = HOLDINGS_LIMIT) -> Description:
    """What the file at `path` is and holds, read through Visual
    Dynamics' importers.

    A Rattlesnake run gives its kind, channels, sample rate and time
    stream from its dimensions, whatever its size. Another format an
    importer recognizes is read whole when it is at most `limit` bytes,
    and its holdings are the data index's (`project.holds`). A file
    nothing recognizes has the format 'unknown'. The run, axis, level
    and start its name gives are added. Nothing here raises for a file
    that cannot be read: what was not read, and why, is in `notes` and
    in the holdings.

    Parameters
    ----------
    path : str or os.PathLike
        The file.
    limit : int, optional
        The largest file read whole for its holdings, in bytes.

    Returns
    -------
    Description
        Its format, holdings, channels, run identity and stream.
    """
    from .. import io
    from ..io import rattlesnake
    from .runs import read_name

    path = os.fspath(path)
    name = os.path.basename(path)
    run, _refused = read_name(name)
    known = io.known_format(path)
    words: list[str] = []
    notes: list[str] = []
    channels: tuple[str, ...] = ()
    stream = None
    try:
        is_rattlesnake = known is not None and rattlesnake.sniff(path)
    except (OSError, ValueError):
        is_rattlesnake = False
    try:
        if known is None:
            notes.append('no importer recognizes its format; it is recorded '
                         'by name, size and SHA-256')
        elif is_rattlesnake:
            words, channels, stream = _rattlesnake(path)
        elif os.path.getsize(path) > limit:
            notes.append(f'larger than {limit} bytes, so it was not read for '
                         'its channels')
        else:
            words, channels = _imported(path)
    except Exception as error:  # noqa: BLE001 — any reader's failure is a note
        notes.append(f'its holdings could not be read ({error})')
    said = _name_words(run)
    if said:
        words.append(said)
    holds = '; '.join(words) or 'not read: ' + '; '.join(notes)
    return Description(known or UNKNOWN, holds, channels, dict(run), stream,
                       tuple(notes))


# ---- watching ---------------------------------------------------------------


@dataclass
class _Seen:
    """A file seen and not yet proposed: its (size, mtime) when last
    seen, since when it has had them on the watch's clock, how many
    looks since then have found them unchanged, and when it was first
    seen, with its zone."""

    key: tuple[int, int]
    since: float
    seen: datetime
    looks: int = 0


def _key(stat: os.stat_result) -> tuple[int, int]:
    return stat.st_size, stat.st_mtime_ns


class ControllerWatch:
    """A watch over one or more controller output folders.

    Each look (`scan`) lists the folders, notes each file that is new or
    has changed since the watch began, and hashes each one that has
    settled, returning a `Proposal` for it; what could not be done comes
    back as a `WatchProblem`. `start` runs the looks on a thread of the
    watch's own every `poll_s` seconds and hands what they find to
    callbacks, or to a queue read with `take`; `stop` ends it.

    The files already in a folder when the watch begins are the records
    of earlier runs and are not proposed, unless `existing` is set; one
    of them that changes afterwards is.

    Parameters
    ----------
    folders : str, os.PathLike, or a sequence of them
        The folders to watch. Each must exist when the watch begins.
    settle_s : float, optional
        Seconds a file's size and modification time must stay the same
        before it is hashed.
    poll_s : float, optional
        Seconds between looks when the watch runs on its thread.
    rate : float or None, optional
        Bytes a second hashing reads at most; None for no limit.
    limit : int, optional
        The largest file read whole for its holdings (`describe`).
    existing : bool, optional
        Propose the files already in the folders as well.
    clock : callable, optional
        Seconds on a monotonic clock, which settling is timed by.
    now : callable, optional
        The time with its zone, which a file is recorded as first seen
        at.

    Raises
    ------
    ControllerFilesError
        When no folder is named, a folder does not exist or is a file,
        or a setting is not a number it can be.
    """

    def __init__(self, folders: str | os.PathLike
                 | Sequence[str | os.PathLike], *,
                 settle_s: float = SETTLE_S, poll_s: float = POLL_S,
                 rate: float | None = RATE, limit: int = HOLDINGS_LIMIT,
                 existing: bool = False,
                 clock: Callable[[], float] = time.monotonic,
                 now: Callable[[], datetime] = _local_now) -> None:
        if isinstance(folders, (str, os.PathLike)):
            folders = [folders]
        named = list(dict.fromkeys(os.path.abspath(os.fspath(folder))
                                   for folder in folders))
        if not named:
            raise ControllerFilesError('name at least one controller output '
                                       'folder to watch')
        for folder in named:
            if os.path.isfile(folder):
                raise ControllerFilesError(f'{folder} is a file, not a '
                                           'folder')
            if not os.path.isdir(folder):
                raise ControllerFilesError(
                    f'no folder at {folder}: a controller output folder is '
                    'watched once it exists')
        self.folders: tuple[str, ...] = tuple(named)
        self.settle_s: float = _finite(settle_s, 'the settling time')
        self.poll_s: float = _finite(poll_s, 'the polling interval', positive=True)
        if rate is not None:
            _finite(rate, 'the hashing rate', positive=True)
        self.rate: float | None = None if rate is None else float(rate)
        self.limit: int = limit
        self._clock = clock
        self._now = now
        self._known: dict[str, tuple[int, int]] = {}
        self._pending: dict[str, _Seen] = {}
        self._failing: set[str] = set()
        self._looking = threading.Lock()
        self._stopping = threading.Event()
        self._wake = threading.Event()
        self._archive_asked = threading.Event()
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue = queue.Queue()
        self._on_proposal: Callable[[Proposal], Any] | None = None
        self._on_problem: Callable[[WatchProblem], Any] | None = None
        if not existing:
            for folder in self.folders:
                for path, stat in self._listing(folder) or ():
                    self._known[path] = _key(stat)

    def __repr__(self) -> str:
        return (f'ControllerWatch({list(self.folders)!r}, '
                f'{len(self._pending)} pending)')

    # ---- one look -------------------------------------------------------------

    @property
    def pending(self) -> tuple[str, ...]:
        """The files seen and not yet proposed, by path."""
        return tuple(self._pending)

    def _listing(self, folder: str) -> list[tuple[str, os.stat_result]] | None:
        """(path, stat) of every file directly in `folder`, or None when
        the folder cannot be read."""
        try:
            with os.scandir(folder) as found:
                listing = []
                for item in found:
                    if item.name.endswith(_IGNORED):
                        continue
                    try:
                        # the file's own stat, not the listing's: Windows
                        # updates a folder's copy of a file's size and time
                        # lazily, and the hash is checked against os.stat,
                        # so the two would disagree on a file just written
                        if item.is_file():
                            listing.append((item.path, os.stat(item.path)))
                    except OSError:
                        # gone between the listing and the look
                        continue
                return listing
        except OSError:
            return None

    def _problem(self, path: str, reason: str) -> list[WatchProblem]:
        """A problem the first time it is met; nothing while it lasts."""
        if path in self._failing:
            return []
        self._failing.add(path)
        return [WatchProblem(path, reason)]

    def _look(self) -> list[WatchProblem]:
        """List the folders and bring what is pending up to date."""
        found: list[WatchProblem] = []
        now = self._clock()
        for folder in self.folders:
            listing = self._listing(folder)
            if listing is None:
                found += self._problem(
                    folder, 'the folder cannot be read (moved, or its share '
                    'is away); it is looked for again at each poll')
                continue
            self._failing.discard(folder)
            here = set()
            for path, stat in listing:
                here.add(path)
                key = _key(stat)
                if self._known.get(path) == key:
                    continue
                seen = self._pending.get(path)
                if seen is None:
                    self._pending[path] = _Seen(key, now, self._now())
                elif seen.key != key:
                    seen.key, seen.since, seen.looks = key, now, 0
                else:
                    seen.looks += 1
            for path in [path for path in self._pending
                         if os.path.dirname(path) == folder
                         and path not in here]:
                # removed before it was hashed: nothing to propose
                del self._pending[path]
                self._failing.discard(path)
        return found

    def _settled(self) -> list[str]:
        now = self._clock()
        # settled only once a later look has found it unchanged, so even
        # with no settling time a file is seen twice before it is read
        return [path for path, seen in self._pending.items()
                if seen.looks >= 1 and now - seen.since >= self.settle_s]

    def scan(self) -> list[Proposal | WatchProblem]:
        """Look at the folders once, and hash and propose each file that
        has settled.

        Returns
        -------
        list of Proposal and WatchProblem
            What this look found, in the order found.
        """
        with self._looking:
            found: list[Proposal | WatchProblem] = list(self._look())
            for path in self._settled():
                found += self._hash(path, archived=False)
            return found

    def archive(self) -> list[Proposal | WatchProblem]:
        """Look at the folders once more and hash every file seen and not
        yet proposed, settled or not: what a test's close asks, so the
        files of its last run are named before it is archived. A file
        that changes while it is hashed is reported, not proposed.

        Returns
        -------
        list of Proposal and WatchProblem
            What was found, in the order found.
        """
        with self._looking:
            found: list[Proposal | WatchProblem] = list(self._look())
            for path in list(self._pending):
                found += self._hash(path, archived=True)
            return found

    def _hash(self, path: str, *, archived: bool) -> list[Proposal | WatchProblem]:
        seen = self._pending[path]
        name = os.path.basename(path)
        try:
            size, sha = data_index.digest(path, self._throttle())
            after = _key(os.stat(path))
        except FileNotFoundError:
            del self._pending[path]
            return []
        except OSError as error:
            return list(self._problem(path, f'{name} could not be read '
                                      f'({error.strerror or error}); it is '
                                      'tried again at the next look'))
        if after != seen.key or size != seen.key[0]:
            # still being written: it settles again from now
            seen.key, seen.since, seen.looks = after, self._clock(), 0
            if archived:
                return list(self._problem(
                    path, f'{name} changed while it was being hashed at the '
                    'close; it is proposed once it settles'))
            return []
        # described before the file leaves the pending list, so a reader
        # that fails leaves it to be tried at the next look
        described = describe(path, limit=self.limit)
        self._failing.discard(path)
        del self._pending[path]
        self._known[path] = after
        folder = os.path.dirname(path)
        entry = IndexEntry(name, path, size, sha, described.holds,
                           described.format, source=SOURCE,
                           indexed=data_index.now())
        return [Proposal(entry, folder, seen.seen, described.channels,
                         described.run, described.stream, described.notes,
                         archived)]

    def _throttle(self) -> Callable[[int, int], None]:
        """A progress callback for one file's hash: it holds the reading
        to `rate`, and ends it when the watch is stopped."""
        began = time.monotonic()

        def told(done: int, _total: int) -> None:
            if self._stopping.is_set():
                raise Cancelled('the controller-file watch was stopped')
            if self.rate is None:
                return
            ahead = done / self.rate - (time.monotonic() - began)
            if ahead > 0 and self._stopping.wait(ahead):
                raise Cancelled('the controller-file watch was stopped')

        return told

    # ---- on a thread of its own ---------------------------------------------

    @property
    def running(self) -> bool:
        """Whether the watch's thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def start(self, on_proposal: Callable[[Proposal], Any] | None = None,
              on_problem: Callable[[WatchProblem], Any] | None = None
              ) -> None:
        """Run the looks on a thread of the watch's own, every `poll_s`
        seconds, until `stop`.

        The callbacks are called on that thread; a front end hands what
        they get to its own thread (a queued signal). Without them, what
        is found goes to a queue that `take` reads from any thread.

        Parameters
        ----------
        on_proposal : callable, optional
            Called with each `Proposal`.
        on_problem : callable, optional
            Called with each `WatchProblem`.

        Returns
        -------
        None

        Raises
        ------
        ControllerFilesError
            When the watch is running already.
        """
        if self.running:
            raise ControllerFilesError('the watch is running already')
        self._on_proposal, self._on_problem = on_proposal, on_problem
        self._stopping.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='run-log-files')
        self._thread.start()

    def poke(self) -> None:
        """Look now rather than at the next poll: what a notice that a
        folder changed asks. Safe from any thread."""
        self._wake.set()

    def ask_archive(self) -> None:
        """Have the thread hash every file not yet proposed at its next
        look, which comes at once (`archive`). Safe from any thread."""
        self._archive_asked.set()
        self._wake.set()

    def stop(self, timeout: float | None = 10.0) -> bool:
        """End the thread: a hash in progress is abandoned at its next
        chunk. Safe to call when it is not running.

        Parameters
        ----------
        timeout : float or None, optional
            Seconds to wait for the thread to end; None to wait as long
            as it takes.

        Returns
        -------
        bool
            Whether the thread has ended.
        """
        self._stopping.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        ended = thread is None or not thread.is_alive()
        if ended:
            self._thread = None
            # the callbacks hold the front end's objects; let go of them
            self._on_proposal = self._on_problem = None
        return ended

    def take(self) -> list[Proposal | WatchProblem]:
        """Everything the thread has found since the last call, when it
        runs without callbacks. Safe from any thread.

        Returns
        -------
        list of Proposal and WatchProblem
            In the order found.
        """
        taken = []
        while True:
            try:
                taken.append(self._queue.get_nowait())
            except queue.Empty:
                return taken

    def _run(self) -> None:
        while not self._stopping.is_set():
            self._wake.clear()
            try:
                if self._archive_asked.is_set():
                    self._archive_asked.clear()
                    found = self.archive()
                else:
                    found = self.scan()
            except Cancelled:
                return
            except Exception as error:  # noqa: BLE001 — the thread reports, never dies silent
                found = [WatchProblem(', '.join(self.folders),
                                      f'the watch failed ({error}); it goes on '
                                      'at the next poll')]
            for item in found:
                if not self._deliver(item):
                    return
            self._wake.wait(self.poll_s)

    def _deliver(self, item: Proposal | WatchProblem) -> bool:
        """Hand one finding on; False when its receiver is gone."""
        call = (self._on_proposal if isinstance(item, Proposal)
                else self._on_problem)
        if call is None:
            self._queue.put(item)
            return True
        try:
            call(item)
        except RuntimeError:
            # the front end's object was deleted under the thread (its
            # window closed without stopping the watch): nothing is left
            # to hand anything to
            self._stopping.set()
            return False
        return True


# ---- the log ----------------------------------------------------------------


def place_at(log: RunLog, when: datetime) -> Place:
    """The place in `log` that was current at `when`: the run started
    last at or before it (its test, axis and run), or the test opened
    last at or before it when that came after, or no place.

    A file that appears after a run stops still belongs to that run,
    since a controller writes its records as the run ends; one that
    appears after the next test opens belongs to that test.

    Parameters
    ----------
    log : RunLog
        The log, read in its corrected view.
    when : datetime
        The time, with its zone.

    Returns
    -------
    Place
        The test, axis and run, any of them None.
    """
    best: Entry | None = None
    for entry in log.view():
        if entry.kind not in ('run_start', 'test_opened') \
                or entry.event_time > when:
            continue
        if best is None or entry.event_time >= best.event_time:
            best = entry
    if best is None:
        return Place(None, None, None)
    if best.kind == 'test_opened':
        return Place(best.test, None, None)
    return Place(best.test, best.axis, best.run)


_CLOCK = re.compile(r'^(?:(\d+):)?(\d+):(\d+(?:\.\d*)?)$')
_AMOUNT = re.compile(r'^(\d+(?:\.\d*)?|\.\d+)\s*([a-z]*)$', re.IGNORECASE)
_UNITS = {'': 1.0, 's': 1.0, 'sec': 1.0, 'secs': 1.0, 'second': 1.0,
          'seconds': 1.0, 'min': 60.0, 'mins': 60.0, 'minute': 60.0,
          'minutes': 60.0, 'h': 3600.0, 'hr': 3600.0, 'hrs': 3600.0,
          'hour': 3600.0, 'hours': 3600.0}


def counter_seconds(text: str) -> float:
    """A counter as the log keeper typed it, in seconds: '1:02:03' (hours,
    minutes, seconds), '12:30' (minutes, seconds), '750', '750 s',
    '12.5 min' or '1 h'.

    Parameters
    ----------
    text : str
        The counter.

    Returns
    -------
    float
        Seconds.

    Raises
    ------
    ControllerFilesError
        When the text is none of those forms, or a clock's minutes or
        seconds are 60 or more; the message names the text.
    """
    said = str(text).strip()
    clock = _CLOCK.fullmatch(said)
    if clock:
        hours, minutes, seconds = clock.groups()
        if (hours is not None and int(minutes) >= 60) or float(seconds) >= 60:
            raise ControllerFilesError(f'counter {text!r}: minutes and '
                                       'seconds on a clock run to 59')
        return (int(hours or 0) * 3600.0 + int(minutes) * 60.0
                + float(seconds))
    amount = _AMOUNT.fullmatch(said)
    if amount and amount.group(2).lower() in _UNITS:
        return float(amount.group(1)) * _UNITS[amount.group(2).lower()]
    raise ControllerFilesError(
        f'counter {text!r} is not a time: write it as 1:02:03, 12:30, '
        '750 s or 12.5 min')


def counted_seconds(log: RunLog, test: int, run: int
                    ) -> tuple[float | None, str]:
    """(seconds, '') the counter ran in run `run` of test `test`, from the
    run's entries: the last counter recorded at a stop or abort less the
    counter at its start (0 when its start has none). (None, why) when it
    cannot be told.

    Parameters
    ----------
    log : RunLog
        The log, read in its corrected view.
    test, run : int
        The test and run, by number.

    Returns
    -------
    tuple of (float or None, str)
        The seconds counted, or None and the reason.
    """
    start = 0.0
    end: float | None = None
    found = False
    for entry in log.view():
        if entry.test != test or entry.run != run:
            continue
        found = True
        counter = entry.fields.get('counter')
        if counter is None or entry.kind not in ('run_start', 'run_stop',
                                                 'abort'):
            continue
        try:
            seconds = counter_seconds(counter)
        except ControllerFilesError as error:
            return None, f'entry #{entry.id}: {error}'
        if entry.kind == 'run_start':
            start = seconds
        else:
            end = seconds
    if not found:
        return None, f'the log has no entries for run {run} of test {test}'
    if end is None:
        return None, (f'run {run} of test {test} has no counter recorded at '
                      'its stop')
    return end - start, ''


def check_stream(log: RunLog, proposal: Proposal, *,
                 place: Place | None = None,
                 tolerance_s: float = STREAM_TOLERANCE_S) -> StreamCheck:
    """A Rattlesnake stream's length against the time its run counted.

    The stream runs from the controller's start to its stop, ramps
    included, so it is expected to be somewhat longer than the counter.
    A difference of more than `tolerance_s` either way is a finding for
    the log keeper: shorter, the stream may not hold the whole run;
    longer, more ran than the counter and its ramps account for. Nothing
    is refused, and when nothing can be compared the check says so and
    is not a pass.

    Parameters
    ----------
    log : RunLog
        The log the run's counter is read from (`counted_seconds`).
    proposal : Proposal
        The file, with its stream (`describe`).
    place : Place, optional
        The run it belongs to; the run current when it was first seen
        (`place_at`) unless given.
    tolerance_s : float, optional
        Seconds the stream may differ from the counter.

    Returns
    -------
    StreamCheck
        What was compared and found.

    Raises
    ------
    ControllerFilesError
        When the tolerance is not a number of seconds, 0 or more.
    """
    tolerance = _finite(tolerance_s, 'the tolerance')
    name = proposal.name
    stream = proposal.stream
    if stream is None:
        return StreamCheck(None, f'{name} holds no time stream, so nothing '
                           'was compared with the counter', tolerance)
    where = place if place is not None else place_at(log, proposal.seen)
    rate, length = stream.sample_rate, stream.seconds
    if where.test is None or where.run is None:
        return StreamCheck(None, f'{name} is not in a run of the log, so its '
                           f'{length:.1f} s stream was not compared with a '
                           'counter', tolerance, length, rate)
    counted, why = counted_seconds(log, where.test, where.run)
    if counted is None:
        return StreamCheck(None, f'{name}: {why}, so its {length:.1f} s '
                           'stream was not compared with a counter',
                           tolerance, length, rate)
    held = (f'{name}: the stream holds {length:.1f} s at {rate:.6g} Hz '
            f'against {counted:.1f} s counted in run {where.run} of test '
            f'{where.test}')
    over = length - counted
    if over < -tolerance:
        return StreamCheck(False, f'{held}, {-over:.1f} s shorter, more than '
                           f'the {tolerance:g} s allowed: the stream may not '
                           'hold the whole run', tolerance, length, rate,
                           counted)
    if over > tolerance:
        return StreamCheck(False, f'{held}, {over:.1f} s longer, more than '
                           f'the {tolerance:g} s allowed for its ramps',
                           tolerance, length, rate, counted)
    return StreamCheck(True, f'{held}, within the {tolerance:g} s allowed',
                       tolerance, length, rate, counted)


def accept(log: RunLog, proposal: Proposal, *, author: str,
           place: Place | None = None, kind: str = 'note',
           text: str = '') -> Entry:
    """Write a proposed file into the log: an entry naming it by SHA-256,
    with its size, format, holdings and path, at the time it was first
    seen, in the run current then.

    Parameters
    ----------
    log : RunLog
        The log, open for writing.
    proposal : Proposal
        The file.
    author : str
        The log keeper's initials.
    place : Place, optional
        Where it goes; the run current when it was first seen
        (`place_at`) unless given.
    kind : str, optional
        The kind of entry, a note unless given.
    text : str, optional
        Free text.

    Returns
    -------
    Entry
        The entry written.

    Raises
    ------
    ControllerFilesError
        When the log already names this file (the same name and
        SHA-256); the message names the entry.
    RunLogError
        For anything the log refuses (`RunLog.append`).
    """
    for entry in log.entries:
        for ref in entry.files:
            if ref.sha256 == proposal.sha256 and ref.name == proposal.name:
                raise ControllerFilesError(f'{proposal.name} is in the log '
                                           f'already, entry #{entry.id}')
    where = place if place is not None else place_at(log, proposal.seen)
    return log.append(kind, author=author, text=text,
                      event_time=proposal.seen, test=where.test,
                      axis=where.axis, run=where.run,
                      files=[proposal.file_ref()])


# ---- the data index -----------------------------------------------------------


def _found(ref: FileRef, search: Sequence[str]) -> str | None:
    candidates = ([ref.path] if ref.path else []) + [
        os.path.join(folder, ref.name) for folder in search]
    return next((path for path in candidates if os.path.isfile(path)), None)


def _heads(log: RunLog, tests: Sequence[int | None]) -> str:
    title = f'run log, {log.header.title}' if log.header.title else 'run log'
    heads = [f'{"between tests" if test is None else f"test {test}"} head '
             f'{log.head(test)}' for test in tests]
    return '; '.join([title, *heads]) if heads else f'{title}; no test yet'


def index_run_log(index: DataIndex, log: RunLog, *,
                  tests: Iterable[int | None] | None = None,
                  search: Sequence[str | os.PathLike] = ()) -> IndexedLog:
    """Put a run log's files, and the log with each test's head hash, into
    a data index (`Project.data_index`), so a report lists them.

    Each file the log names is entered through the index's own entry
    (`DataIndex.add`) with the size and SHA-256 the log recorded, so it
    is not hashed again; the index's check before a report is issued
    (`DataIndex.verify`) hashes it and says whether it still matches. A
    file is looked for where it was when it was hashed, then by its name
    in each folder of `search`, then beside the log; a file found
    nowhere is listed as missing, since an index entry names a file
    that is there. The log's own file is entered as a run log holding
    each test's head (`RunLog.head`), replacing the run log's earlier
    entry for the same file, since the log grows.

    Parameters
    ----------
    index : DataIndex
        The index, the project's data index in a project.
    log : RunLog
        The log.
    tests : iterable of int or None, optional
        The tests whose files and heads go in (None for the entries
        between tests); every test, and the entries between tests, unless
        given.
    search : sequence of str or os.PathLike, optional
        Folders a file is looked for in by its name.

    Returns
    -------
    IndexedLog
        The log's entry, each file's entry, and each file not found.

    Raises
    ------
    ControllerFilesError
        When a test named is not in the log.
    """
    from .. import io

    if tests is None:
        chosen: list[int | None] = list(log.tests())
        if 'log' in log.chains():
            chosen.append(None)
    else:
        chosen = list(dict.fromkeys(tests))
        held = log.tests()
        for test in chosen:
            if test is not None and test not in held:
                raise ControllerFilesError(f'the log holds no test {test}')
    folders = [os.fspath(folder) for folder in search]
    folders.append(os.path.dirname(os.path.abspath(log.path)))
    added: list[IndexEntry] = []
    missing: list[tuple[FileRef, str]] = []
    done: set[tuple[str, str]] = set()
    for entry in log.view():
        if entry.test not in chosen:
            continue
        for ref in entry.files:
            if (ref.name, ref.sha256) in done:
                continue
            done.add((ref.name, ref.sha256))
            path = _found(ref, folders)
            if path is None:
                where = f'at {ref.path} or ' if ref.path else ''
                reason = (f'{ref.name} (entry #{entry.id}) was not found '
                          f'{where}by its name in {", ".join(folders)}')
                missing.append((ref, reason))
                continue
            size = ref.size if ref.size is not None else os.path.getsize(path)
            added.append(index.add(
                path, ref.holds or f'named in the run log, entry #{entry.id}',
                format=ref.format or io.format_of(path), source=SOURCE,
                checksum=(size, ref.sha256)))
    own = os.path.abspath(log.path)
    index.entries = [item for item in index.entries
                     if not (item.source == SOURCE and os.path.normcase(
                         item.path) == os.path.normcase(own))]
    entry = index.add(own, _heads(log, chosen), format=LOG_FORMAT,
                      source=SOURCE)
    return IndexedLog(entry, tuple(added), tuple(missing))
