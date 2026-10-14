"""The checks every sniffer makes before it reads anything.

A sniffer answers "is this file one this importer reads?" and must
never raise while answering: a file it cannot read is not its file,
whatever the reason. The open and the exception handling for that are
here, once, for every importer.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

# Each text importer used to carry its own copy of the try/open/except, and
# each npz importer its own copy of the open/except.


def text_head(path: str | os.PathLike, size: int = 65536) -> str | None:
    """The first `size` characters of a text file, or None when the file
    cannot be read.

    Undecodable bytes are replaced rather than raised, because a sniff
    judges the shape of the text, and one undecodable byte must not
    reject a file the loader would read.
    """
    try:
        with open(path, errors='replace') as f:
            return f.read(size)
    except OSError:
        return None


def npz_has(path: str | os.PathLike, keys: Iterable[str], *,
            allow_pickle: bool = True) -> bool:
    """True when `path` is an `.npz` holding every one of `keys`; False
    for any other file, or one that cannot be read.

    `allow_pickle` is the caller's choice: the Rattlesnake specification
    is plain arrays and refuses pickles, while the sdynpy formats hold
    object arrays.
    """
    if not str(path).endswith('.npz'):
        return False
    try:
        import numpy as np
        with np.load(path, allow_pickle=allow_pickle) as d:
            return set(keys) <= set(d.files)
    except Exception:  # noqa: BLE001 - sniffers must not raise on foreign files
        return False


def npz_summary(path: str | os.PathLike) -> str | None:
    """A one-line description of what an `.npz` holds (the number of
    arrays and the first eight names), for an error that would otherwise
    say only that no importer recognized it. None when the file is not
    an `.npz`.
    """
    if not str(path).lower().endswith('.npz'):
        return None
    try:
        import numpy as np

        with np.load(path, allow_pickle=True) as d:
            names = sorted(d.files)
        said = f'an npz archive holding {len(names)} array'
        said += 's' if len(names) != 1 else ''
        if names:
            said += ': ' + ', '.join(names[:8])
            said += ', …' if len(names) > 8 else ''
        return said
    except Exception:  # noqa: BLE001
        return 'an npz archive that could not be read'


def describe(path: str | os.PathLike) -> str | None:
    """What a file is, for an error message: `npz_summary` for an `.npz`,
    None for anything else.

    An `.npz` is a container whose contents decide which reader takes it,
    and an error that names only the path cannot tell the wrong file
    from an unsupported one.
    """
    return npz_summary(path)
