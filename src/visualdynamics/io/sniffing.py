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


def npz_has(path: str | os.PathLike, keys: Iterable[str]) -> bool:
    """True when `path` is an `.npz` holding every one of `keys`; False
    for any other file, or one that cannot be read.

    Only the archive's list of names is read. A sniff never unpickles: it
    runs on any file that is dropped or listed, before anyone has asked
    for it to be read.
    """
    if not str(path).endswith('.npz'):
        return False
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as d:
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

        with np.load(path, allow_pickle=False) as d:
            names = sorted(d.files)
        said = f'an npz archive holding {len(names)} array'
        said += 's' if len(names) != 1 else ''
        if names:
            said += ': ' + ', '.join(names[:8])
            said += ', …' if len(names) > 8 else ''
        return said
    except Exception:  # noqa: BLE001
        return 'an npz archive that could not be read'


def npy_summary(path: str | os.PathLike) -> str | None:
    """Why a `.npy` is not read, when that is something other than being
    the wrong array: it holds objects, or is not a NumPy file at all. None
    for anything else."""
    if not str(path).lower().endswith('.npy'):
        return None
    from .plain_numpy import npy_dtype

    try:
        holds_objects = npy_dtype(path).hasobject
    except OSError:
        return None
    except ValueError:
        return 'a .npy that is not a NumPy file'
    if holds_objects:
        return ('a NumPy file of Python objects, which could run code from '
                'the file if read, so it was not')
    return None


def describe(path: str | os.PathLike) -> str | None:
    """What a file is, for an error message: `npz_summary` for an `.npz`,
    `npy_summary` for a `.npy`, None for anything else.

    An `.npz` is a container whose contents decide which reader takes it,
    and an error that names only the path cannot tell the wrong file
    from an unsupported one.
    """
    return npz_summary(path) or npy_summary(path)
