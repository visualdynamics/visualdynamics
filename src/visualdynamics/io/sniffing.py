"""What every sniffer does before it reads anything.

A sniffer answers "is this file yours?" and must never raise doing it —
an unreadable file is simply not ours, whatever the reason. Each text
importer was carrying its own copy of that try/open/except; the rule
lives here once.
"""

from __future__ import annotations

import os
from collections.abc import Iterable


def text_head(path: str | os.PathLike, size: int = 65536) -> str | None:
    """The first `size` characters of a text file, or None where the
    file cannot be read. `errors='replace'` because a sniff judges the
    shape of the text, and one undecodable byte must not veto a file
    the loader would take."""
    try:
        with open(path, errors='replace') as f:
            return f.read(size)
    except OSError:
        return None


def npz_has(path: str | os.PathLike, keys: Iterable[str], *,
            allow_pickle: bool = True) -> bool:
    """Whether an `.npz` at `path` carries every one of `keys` — the
    question the three npz importers ask, each of which once carried
    its own copy of the open/except. `allow_pickle` is the caller's
    call: the Rattlesnake specification is plain arrays and refuses
    pickles, sdynpy's saves carry object arrays."""
    if not str(path).endswith('.npz'):
        return False
    try:
        import numpy as np
        with np.load(path, allow_pickle=allow_pickle) as d:
            return set(keys) <= set(d.files)
    except Exception:  # noqa: BLE001 - sniffers must not raise on foreign files
        return False


def npz_summary(path: str | os.PathLike) -> str | None:
    """A one-line description of what an `.npz` holds, for a refusal
    that would otherwise say only that nothing recognized it. None
    when the file is not an npz archive."""
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
    """What a file is, for a refusal to say. An `.npz` is a container
    whose contents decide which reader wants it, and a refusal that
    names only the path cannot tell the wrong file from an
    unsupported one."""
    return npz_summary(path)
