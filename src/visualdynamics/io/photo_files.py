"""Write photos out as image files.

A `Photos` object holds each file's bytes exactly as they were read,
so writing them out is a copy with no re-encoding: a JPEG written here
is the same JPEG, byte for byte.

The photos are written into a folder, which is the one way this is not
the inverse of reading them: a Photos object is ordered and named, and
a folder is neither. The names are kept as the filenames; the order is
not, and on reading back is only what the file manager's sorting gives.
The object holds nothing else, so nothing else is lost.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                                    # pragma: no cover
    from ..core.photos import Photos
    from ..units import UnitSystem

#: the extension each stored format is written under. An ESCDF import
#: names a photo's format by its attachment's suffix, so 'jpg', 'gif' and
#: 'bmp' arrive as well, and each was written as '.png' around bytes that
#: are not PNG (2026-10-13, found by the docstring pass).
SUFFIX = {'png': '.png', 'jpeg': '.jpg', 'jpg': '.jpg', 'gif': '.gif',
          'bmp': '.bmp', 'heic': '.heic'}

#: characters a filename cannot carry on some platform or other. A photo
#: may be called anything — 'Fixture 3/4 view' is a fine caption.
UNSAFE = '/\\:*?"<>|'


def handles(obj: Any) -> bool:
    from ..core.photos import Photos

    return isinstance(obj, Photos)


def file_name(name: str, fmt: str, taken: set[str]) -> str:
    """A filename for one photo: its own name, with characters some
    platform cannot use in a filename replaced by '-', and made unique
    among the names in `taken` (compared ignoring case), which it is added
    to.

    Two photos in one object may have the same name, and a folder cannot
    hold two files with one name, so the second is numbered ('name
    (2).jpg') instead of replacing the first. An empty name becomes
    'photo', and a format with no known suffix is given '.png'.
    """
    stem = ''.join('-' if c in UNSAFE or ord(c) < 32 else c
                   for c in str(name)).strip(' .') or 'photo'
    suffix = SUFFIX.get(str(fmt).lower(), '.png')
    candidate, n = f'{stem}{suffix}', 1
    while candidate.lower() in taken:
        n += 1
        candidate = f'{stem} ({n}){suffix}'
    taken.add(candidate.lower())
    return candidate


def save(photos: Photos, path: str | os.PathLike,
         unit_system: UnitSystem | None = None, **_kwargs: Any) -> list[str]:
    """Write every photo into the folder `path`, which is created if it
    does not exist.

    It is a folder rather than a file because the object is several
    pictures and an image format holds one. A `path` with a suffix is
    taken as a file name, and the folder is that path without the
    suffix. Returns the paths written, which a caller preparing files
    for a drag needs.

    `unit_system` is accepted and not used, since a photograph has no
    units.
    """
    folder = str(path)
    # a caller who named a file rather than a folder means the folder it
    # would have sat in — dragging out asks for one, Export As the other
    if os.path.splitext(folder)[1]:
        folder = os.path.splitext(folder)[0]
    os.makedirs(folder, exist_ok=True)

    written, taken = [], set()
    for name, fmt, image in zip(photos.names, photos.formats, photos.images):
        target = os.path.join(folder, file_name(name, fmt, taken))
        with open(target, 'wb') as out:
            out.write(image)
        written.append(target)
    return written
