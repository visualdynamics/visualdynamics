"""Read NumPy files without running anything in them.

A `.npy` or `.npz` member with an object array holds a pickle, and
unpickling runs whatever the pickle says. A file is something a person
was sent and drops on the window, so nothing here hands a file's pickle
to `pickle.load` or `allow_pickle=True`: an array of objects is refused
in words, and the one kind a real file needs, the node id arrays in a
geometry's `traceline` and `element` (`PlainNpz.with_objects`), is
rebuilt from a pickle that may name nothing but numpy's arrays.
"""

# Brought over from Hertzforge, where it closed the same hole (Brandon,
# 2026-11-07). Brandon asked for it here on 2026-11-11 ("Yes, port it"),
# for parity: the sdynpy readers here had handed every file's pickles to
# np.load(allow_pickle=True).

from __future__ import annotations

import math
import os
import pickle
import re
import zipfile
from typing import IO, Any, Self

import numpy as np


def _holds_objects(where: Any, asked_for: str = '') -> ValueError:
    asked = f' (it asks for {asked_for})' if asked_for else ''
    return ValueError(f'{where} holds Python objects, not plain numbers and '
                      f'text{asked}. Reading them could run code from the '
                      'file, so it was not read.')


def _header(fp: IO[bytes]) -> tuple[tuple[int, ...], np.dtype]:
    """The shape and dtype a NumPy array stream declares, leaving `fp` at
    its data. The header is parsed as a literal, so it runs nothing."""
    version = np.lib.format.read_magic(fp)
    read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
            else np.lib.format.read_array_header_2_0)
    shape, _fortran, dtype = read(fp)
    return shape, dtype


def npy_dtype(path: str | os.PathLike) -> np.dtype:
    """What a `.npy` declares it holds, from its header alone. Raises
    ValueError for a file that is not a NumPy file."""
    with open(path, 'rb') as fp:
        return _header(fp)[1]


def read_npy(path: str | os.PathLike) -> np.ndarray:
    """A `.npy` of plain data; ValueError for an array of objects, or for
    a file that is not a NumPy file."""
    try:
        dtype = npy_dtype(path)
    except ValueError:
        raise ValueError(f'{path} is not a NumPy file') from None
    if dtype.hasobject:
        raise _holds_objects(path)
    return np.load(path, allow_pickle=False)


class PlainNpz:
    """An open `.npz`, used as `np.load`'s result is, whose arrays of
    objects are refused in words."""

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = path
        try:
            self._npz = np.load(path, allow_pickle=False)
            self.files: list[str] = self._npz.files
        except (ValueError, AttributeError, zipfile.BadZipFile):
            raise ValueError(f'{path} is not an npz archive: it could not be read') from None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self._npz.close()

    def __contains__(self, name: str) -> bool:
        return name in self._npz

    def _open(self, name: str) -> IO[bytes]:
        if name not in self._npz:
            raise KeyError(f'{name} is not a file in the archive')
        member = f'{name}.npy'
        names = self._npz.zip.namelist()
        return self._npz.zip.open(member if member in names else name)

    def __getitem__(self, name: str) -> np.ndarray:
        where = f'{self._path}: {name}'
        with self._open(name) as fp:
            try:
                _shape, dtype = _header(fp)
            except ValueError:
                raise ValueError(f'{where} is not a NumPy array') from None
        if dtype.hasobject:
            raise _holds_objects(where)
        return self._npz[name]

    def with_objects(self, name: str) -> np.ndarray:
        """An array that may hold objects, rebuilt from plain data only."""
        where = f'{self._path}: {name}'
        with self._open(name) as fp:
            try:
                shape, dtype = _header(fp)
            except ValueError:
                raise ValueError(f'{where} is not a NumPy array') from None
            if dtype.hasobject:
                return _read_objects(fp, shape, dtype, where)
        return self._npz[name]


# what numpy writes an array of objects as: an array is
# `_reconstruct(ndarray, ...)` and its type a `dtype(...)`. The module moved
# in numpy 2.0, and a file keeps the name it was saved under.
_NUMPY_CALLS = {('numpy', 'ndarray'): 'ndarray', ('numpy', 'dtype'): 'dtype',
                ('numpy.core.multiarray', '_reconstruct'): 'array',
                ('numpy._core.multiarray', '_reconstruct'): 'array'}

_NUMBER_OR_TEXT = re.compile(r'[biufcSU]\d+')


class _NotPlain(Exception):
    pass


class _Call:
    """A numpy call the pickle asked for, written down and not made.

    numpy's own `_reconstruct` and the `__setstate__` of an array or a
    dtype trust what they are handed: a dtype's pickled state can claim a
    field holds no objects while giving it raw bytes, and those bytes are
    then read as pointers. So nothing in the pickle reaches numpy until
    it has been checked to be a number or text.
    """

    __slots__ = ('args', 'kind', 'state')

    def __init__(self, kind: str, args: tuple) -> None:
        self.kind: str = kind
        self.args: tuple = args
        self.state: Any = None

    def __setstate__(self, state: Any) -> None:
        self.state = state


class _PlainUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        kind = _NUMPY_CALLS.get((module, name))
        if kind is None:
            raise _NotPlain(f'{module}.{name}')
        # a new callable each time: a pickle can set attributes on what
        # it gets from here, and there is then nothing shared to change
        return lambda *args: _Call(kind, args)


def _plain_dtype(call: Any) -> np.dtype:
    """The real dtype of a pickled `dtype(...)`, a number or text with no
    fields of its own."""
    if not isinstance(call, _Call) or call.kind != 'dtype' or not call.args:
        raise _NotPlain('an array of something other than numbers or text')
    spec, state = call.args[0], call.state
    if not (isinstance(spec, str) and _NUMBER_OR_TEXT.fullmatch(spec)
            and isinstance(state, tuple) and len(state) >= 5
            and state[1] in ('<', '>', '|', '=')
            and all(part is None for part in state[2:5])):
        raise _NotPlain('an array of something other than numbers or text')
    return np.dtype(state[1] + spec)


def _plain_value(value: Any) -> Any:
    """One entry of an array of objects, with what was pickled as a numpy
    array made into one."""
    if isinstance(value, _Call):
        if value.kind == 'array' and isinstance(value.state, tuple) and len(value.state) == 5:
            _version, shape, dtype, fortran, raw = value.state
            if isinstance(raw, bytes):
                return (np.frombuffer(raw, _plain_dtype(dtype))
                        .reshape(shape, order='F' if fortran else 'C').copy())
        raise _NotPlain('an array of something other than numbers or text')
    if value is None or isinstance(value, bool | int | float | str | bytes):
        return value
    if isinstance(value, list | tuple) and all(isinstance(n, int | float) for n in value):
        return value
    raise _NotPlain(f'a {type(value).__name__}')


def _read_objects(fp: IO[bytes], shape: tuple[int, ...], dtype: np.dtype,
                  where: str) -> np.ndarray:
    """The array of objects whose pickle `fp` is at, of the `shape` and
    `dtype` its header declared."""
    try:
        whole = _PlainUnpickler(fp).load()
        # an array of objects pickles as `_reconstruct`, with its entries
        # as a list in place of the bytes
        if not (isinstance(whole, _Call) and whole.kind == 'array'
                and isinstance(whole.state, tuple) and len(whole.state) == 5
                and isinstance(whole.state[4], list)
                and len(whole.state[4]) == math.prod(shape)):
            raise _NotPlain('something other than an array')
        out = np.empty(shape, dtype)
        flat = out.reshape(-1)
        for i, entry in enumerate(whole.state[4]):
            if dtype.names is None:
                flat[i] = _plain_value(entry)
            else:
                if not isinstance(entry, tuple):
                    raise _NotPlain('a record that is not one')
                flat[i] = tuple(_plain_value(value) for value in entry)
    except _NotPlain as refused:
        raise _holds_objects(where, str(refused)) from None
    except Exception:  # noqa: BLE001 - a broken pickle says nothing a person can act on
        raise _holds_objects(where) from None
    return out
