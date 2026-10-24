"""The data index: every file a project rests on, by name, size and
SHA-256.

A report names the evidence it was written from. The index is that
naming: for each file, its name, where it was, what it holds, its
format, its size in bytes and its SHA-256 checksum, the fingerprint
that changes if a single byte of the file changes, so that a copy can
be shown identical to the original years later. A project keeps one
index (`Project.data_index`), whatever adds to it: an import records
the file it read, the author attaches a document (a settings list, a
log, a scanned procedure), and any other source of files adds its rows
the same way (`DataIndex.add`).

Each entry is either **referenced** (named precisely, kept in the
archive) or **attached** (small enough to travel with the report). A
small text file that is attached keeps its contents, so the report can
print it.

`DataIndex.verify` hashes every file again where it can be found and
marks each entry as matching, changed or not found; that is the check
a report is given before it is issued. Hashing reads files in chunks
and reports its progress, so a front end can run it off its event loop
and show a bar.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

# One index per project, whatever adds to it: two indexes of the same
# files would drift, and a report would have to say which one it
# printed.

#: how an entry travels with the report: named and kept in the archive,
#: or sent with it
ROLES = ('referenced', 'attached')

#: what a check found, in the words the report prints. An entry never
#: checked has no status.
MATCHES = 'matches'
CHANGED = 'changed'
NOT_FOUND = 'not found'
STATUSES = ('', MATCHES, CHANGED, NOT_FOUND)

#: bytes read at a time while hashing: large enough that the hash, not
#: the loop, is the cost, and small enough that progress moves
CHUNK = 4 * 1024 * 1024

#: the largest attached text file whose contents are kept for the
#: report to print: a settings list or a short log, not a data file
TEXT_LIMIT = 64 * 1024

_HEX = re.compile(r'[0-9a-f]{64}')


def now() -> str:
    """The time, in UTC, to the second, as an index records it."""
    return datetime.now(UTC).isoformat(timespec='seconds')


def digest(path: str | os.PathLike,
           progress: Callable[[int, int], None] | None = None
           ) -> tuple[int, str]:
    """(size in bytes, SHA-256 as 64 lower-case hex digits) of a file.

    The file is read in chunks of `CHUNK` bytes, so a file of any size
    is hashed in constant memory; `progress`, when given, is called as
    ``progress(bytes read, bytes expected)`` before the first chunk and
    after each one. The size returned is the bytes actually read, so it
    always describes the bytes that were hashed.

    Raises
    ------
    FileNotFoundError
        When there is no file at `path`.
    IsADirectoryError
        When `path` is a folder.
    TypeError
        When `path` is not a path.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f'a file is named by its path, not '
                        f'{type(path).__name__}')
    path = os.fspath(path)
    if os.path.isdir(path):
        raise IsADirectoryError(f'{path} is a folder, not a file')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'no file at {path}')
    total = os.path.getsize(path)
    hasher = hashlib.sha256()
    done = 0
    if progress is not None:
        progress(0, total)
    with open(path, 'rb') as handle:
        while chunk := handle.read(CHUNK):
            hasher.update(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done, max(total, done))
    return done, hasher.hexdigest()


def _same_file(a: str, b: str) -> bool:
    return (os.path.normcase(os.path.abspath(a))
            == os.path.normcase(os.path.abspath(b)))


@dataclass(frozen=True)
class IndexEntry:
    """One file of the index.

    `name` is the file's name as the report prints it, and `path` where
    it was when it was indexed. `size` is in bytes and `sha256` is the
    checksum, as 64 lower-case hex digits. `format` names the file's
    format and `holds` says what is in it: the records and channels of
    a data file, or what a document is ("settings list"). `role` is
    'referenced' or 'attached' (`ROLES`). `source` says what added the
    entry: 'import' for a file read into the project, 'author' for one
    added by hand, or the name of another source. `objects` are the
    project objects read from the file. `indexed` is when the file was
    hashed, in UTC. `text` is the contents of a small attached text
    file, kept so the report can print it, and otherwise ''.

    `status` is what the last check found (`MATCHES`, `CHANGED`,
    `NOT_FOUND`, or '' before any check), `checked` when, and `found`
    where the file was found when that was not `path`.
    """

    name: str
    path: str
    size: int
    sha256: str
    holds: str
    format: str = ''
    role: str = 'referenced'
    source: str = 'import'
    objects: tuple[str, ...] = ()
    indexed: str = ''
    text: str = ''
    status: str = ''
    checked: str = ''
    found: str = ''

    def __post_init__(self) -> None:
        for label in ('name', 'path', 'holds', 'source'):
            value = getattr(self, label)
            if not isinstance(value, str):
                raise TypeError(f'a data index entry\'s {label} is a str, '
                                f'not {type(value).__name__}')
        if not self.name.strip():
            raise ValueError('a data index entry needs the file\'s name')
        if not self.holds.strip():
            raise ValueError(f'{self.name}: say what the file holds; an '
                             'index entry that does not cannot be asked for')
        if not self.source.strip():
            raise ValueError(f'{self.name}: say what added the entry')
        if self.role not in ROLES:
            raise ValueError(f'{self.name}: role {self.role!r} is not one of '
                             f'{", ".join(ROLES)}')
        if (isinstance(self.size, bool) or not isinstance(self.size, int)
                or self.size < 0):
            raise ValueError(f'{self.name}: the size is a whole number of '
                             f'bytes, not {self.size!r}')
        if not isinstance(self.sha256, str) or not _HEX.fullmatch(self.sha256):
            raise ValueError(f'{self.name}: {self.sha256!r} is not a SHA-256 '
                             'checksum (64 lower-case hex digits)')
        if self.status not in STATUSES:
            raise ValueError(f'{self.name}: status {self.status!r} is not '
                             f'one of {", ".join(map(repr, STATUSES))}')
        object.__setattr__(self, 'objects',
                           tuple(str(name) for name in self.objects))

    def record(self) -> dict[str, Any]:
        """The entry as plain values, for the project file; an empty
        field is left out."""
        out: dict[str, Any] = {}
        for key in ('name', 'path', 'size', 'sha256', 'holds', 'format',
                    'role', 'source', 'objects', 'indexed', 'text',
                    'status', 'checked', 'found'):
            value = getattr(self, key)
            if key == 'objects':
                value = list(value)
            if value or key in ('size',):
                out[key] = value
        return out

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> IndexEntry:
        """An entry read back from `record`. A key this version does
        not know is ignored, so a file written by a later version still
        opens; a malformed entry raises `ValueError`, and one that is
        not a mapping `TypeError`."""
        if not isinstance(record, Mapping):
            raise TypeError(f'a data index entry is a mapping, not '
                             f'{type(record).__name__}')
        known = {key: record[key] for key in cls.__dataclass_fields__
                 if key in record}
        missing = [key for key in ('name', 'path', 'size', 'sha256', 'holds')
                   if key not in known]
        if missing:
            raise ValueError(f'a data index entry is missing '
                             f'{", ".join(missing)}')
        known['objects'] = tuple(known.get('objects', ()))
        return cls(**known)


@dataclass
class DataIndex:
    """Every file a project rests on, in the order each was indexed.

    Iterating gives the entries (`IndexEntry`). `project_file` is
    (name, size, SHA-256, when) of the project's own file at the last
    check, or None: kept in memory only, since a file cannot carry its
    own checksum.
    """

    entries: list[IndexEntry] = field(default_factory=list)
    project_file: tuple[str, int, str, str] | None = None

    def __post_init__(self) -> None:
        entries = list(self.entries)
        for entry in entries:
            if not isinstance(entry, IndexEntry):
                raise TypeError(f'a data index holds IndexEntry, not '
                                f'{type(entry).__name__}')
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[IndexEntry]:
        return iter(self.entries)

    def __getitem__(self, index: int) -> IndexEntry:
        return self.entries[index]

    def add(self, path: str | os.PathLike, holds: str, *, format: str = '',
            role: str = 'referenced', source: str = 'import',
            objects: Iterable[str] = (),
            checksum: tuple[int, str] | None = None,
            progress: Callable[[int, int], None] | None = None) -> IndexEntry:
        """Index the file at `path` and return its entry.

        The file is hashed (`digest`, with `progress`) unless `checksum`
        gives (size, SHA-256) already computed, for example on a worker
        thread. An attached text file of at most `TEXT_LIMIT` bytes
        keeps its contents. The same file with the same checksum
        indexed again is one entry, which gains the new `objects`; the
        same path with different contents is a second entry, since it
        is a different file.

        Refused, before anything changes: `holds` empty or not a
        string, an unknown `role`, an empty `source`, no file at
        `path`, a folder, and an attached text file whose contents
        changed between the checksum and the read.

        Raises
        ------
        ValueError
            For a refusal; the message names the file and the reason.
        FileNotFoundError, IsADirectoryError
            When `path` is not a file.
        TypeError
            When `path` or `holds` is not of the right kind.
        """
        if not isinstance(holds, str):
            raise TypeError(f'holds is a str, not {type(holds).__name__}')
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError(f'a file is named by its path, not '
                            f'{type(path).__name__}')
        path = os.path.abspath(os.fspath(path))
        name = os.path.basename(path)
        # the entry's own checks first, on a stand-in checksum, so a
        # refusal costs no read of the file
        IndexEntry(name or '?', path, 0, '0' * 64, holds, str(format), role,
                   source)
        if os.path.isdir(path):
            raise IsADirectoryError(f'{path} is a folder, not a file')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'no file at {path}')
        text = ''
        if checksum is None:
            size, sha = digest(path, progress)
        else:
            size, sha = int(checksum[0]), str(checksum[1])
        if role == 'attached' and size <= TEXT_LIMIT:
            text = _text_of(path, sha)
        made = IndexEntry(name, path, size, sha, holds.strip(), str(format),
                          role, source, tuple(objects), now(), text)
        for at, entry in enumerate(self.entries):
            if entry.sha256 == sha and _same_file(entry.path, path):
                merged = tuple(dict.fromkeys((*entry.objects,
                                              *made.objects)))
                self.entries[at] = replace(entry, objects=merged)
                return self.entries[at]
        self.entries.append(made)
        return made

    def remove(self, name: str) -> IndexEntry:
        """Take the entry for the file named `name` (its name, or its
        path) out of the index, and return it. Raises `ValueError` when
        no entry, or more than one, has that name."""
        found = [at for at, entry in enumerate(self.entries)
                 if name in (entry.name, entry.path)]
        if not found:
            raise ValueError(f'no file named {name!r} in the data index')
        if len(found) > 1:
            raise ValueError(f'{len(found)} entries are named {name!r}; '
                             'name the one by its path')
        return self.entries.pop(found[0])

    def rename_object(self, old: str, new: str) -> None:
        """Follow a project object renamed from `old` to `new`."""
        self.entries = [
            replace(entry, objects=tuple(new if name == old else name
                                         for name in entry.objects))
            if old in entry.objects else entry for entry in self.entries]

    def forget_objects(self, names: Iterable[str],
                       keep: Callable[[IndexEntry], bool] | None = None
                       ) -> list[IndexEntry]:
        """Follow project objects removed: each entry loses `names`, and
        an imported entry left with no object is dropped unless
        `keep(entry)` says something still rests on it. Returns the
        entries dropped."""
        gone = set(names)
        kept, dropped = [], []
        for entry in self.entries:
            if not gone & set(entry.objects):
                kept.append(entry)
                continue
            left = replace(entry, objects=tuple(
                name for name in entry.objects if name not in gone))
            if (entry.source == 'import' and not left.objects
                    and not (keep is not None and keep(entry))):
                dropped.append(entry)
            else:
                kept.append(left)
        self.entries = kept
        return dropped

    def adopt(self, other: DataIndex | None,
              mapping: Mapping[str, str] | None = None) -> None:
        """Take in the entries of another project's index, its objects
        renamed through `mapping` ({old: new}) the way the objects
        were; an entry already here is merged, not repeated."""
        for entry in (other or ()):
            objects = tuple(dict.fromkeys(
                (mapping or {}).get(name, name) for name in entry.objects
                if mapping is None or name in mapping))
            moved = replace(entry, objects=objects)
            for at, mine in enumerate(self.entries):
                if mine.sha256 == moved.sha256 and _same_file(mine.path,
                                                              moved.path):
                    self.entries[at] = replace(mine, objects=tuple(
                        dict.fromkeys((*mine.objects, *moved.objects))))
                    break
            else:
                self.entries.append(moved)

    def verify(self, search: Sequence[str | os.PathLike] = (),
               progress: Callable[[int, int], None] | None = None
               ) -> tuple[IndexEntry, ...]:
        """Hash every indexed file again and record what was found.

        Each file is looked for where it was indexed, then by its name
        in each folder of `search` in turn (an archive copy, the folder
        the project now sits in). Found with the same size and
        checksum, it `MATCHES`; found otherwise, it has `CHANGED`; not
        found anywhere, it is `NOT_FOUND`. `progress` is called as
        (bytes read, bytes expected) across the whole index. The
        entries are replaced only once every file has been read, so a
        check stopped part way (a cancel raised from `progress`) leaves
        the index as it was.

        Returns
        -------
        tuple of IndexEntry
            The entries afterwards, in order.
        """
        when = now()
        # what the bar counts to: the bytes indexed, never zero, and a
        # file grown since cannot carry the count past it
        whole = max(sum(entry.size for entry in self.entries), 1)
        base = 0
        out = []
        for entry in self.entries:
            candidates = [entry.path] + [os.path.join(os.fspath(folder),
                                                      entry.name)
                                         for folder in search]
            where = next((path for path in candidates
                          if os.path.isfile(path)), None)
            if where is None:
                out.append(replace(entry, status=NOT_FOUND, checked=when,
                                   found=''))
                base += entry.size
                continue

            def told(done, _expected, base=base):
                if progress is not None:
                    progress(min(base + done, whole), whole)

            size, sha = digest(where, told if progress is not None else None)
            base += entry.size
            status = MATCHES if (size, sha) == (entry.size,
                                                entry.sha256) else CHANGED
            out.append(replace(
                entry, status=status, checked=when,
                found='' if _same_file(where, entry.path) else where))
        self.entries = out
        return tuple(out)

    def records(self) -> list[dict[str, Any]]:
        """Every entry as plain values, for the project file."""
        return [entry.record() for entry in self.entries]

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]] | None
                     ) -> DataIndex:
        """An index read back from `records` (`records()`)."""
        return cls([IndexEntry.from_record(record)
                    for record in (records or ())])


def _text_of(path: str, sha: str) -> str:
    """The contents of a small attached file, when it is text; '' when
    it is not (it still travels, unprinted). Refuses a file whose bytes
    no longer match `sha`."""
    with open(path, 'rb') as handle:
        data = handle.read(TEXT_LIMIT + 1)
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError(f'{os.path.basename(path)} changed while it was '
                         'being indexed; index it again')
    if b'\x00' in data:
        return ''
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return ''
