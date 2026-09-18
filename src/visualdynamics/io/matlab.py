"""MATLAB `.mat` files: the project file in MATLAB's container.

A `.mat` written here is a `.vdyn` respelled — the same schema, the
same names, the same values (`docs/vdyn-format.md`) — so a MATLAB user
gets `load('modal.mat')` and structs with the right shapes and complex
numbers assembled, rather than HDF5 datasets to walk with `h5read`
(which a `.vdyn` already allows). Everything a project holds has a
MATLAB form, so the round trip is lossless and the reader takes its
own files back exactly.

There is one implementation of the layout, and it is not here. The
writer runs the native `.vdyn` writer into an HDF5 file held in memory
and carries that tree into `.mat` variables; the reader rebuilds an
in-memory HDF5 tree from the `.mat` and hands it to the native loader.
A field the schema grows tomorrow travels tomorrow with no change
here, and the two files cannot drift apart because neither this module
nor `native.py` knows two layouts (principle 9). The price is one
in-memory HDF5 round trip per file, which is cheap.

Three things MATLAB spells differently from HDF5, and the rules for
them, applied mechanically both ways:

- **Ragged arrays** — the `<name>_flat` / `<name>_offsets` pairs
  (traceline and element connectivity) — are one cell array `<name>`,
  one row vector per entry, which is how MATLAB holds a ragged list.
- **Numbered groups** — the project's `objects/0000, 0001, …` and a
  sine specification's `tones/0, 1, …` — are a cell array of structs
  in the same order (`p.objects{3}.name`); each object struct keeps
  the `name` and `kind` its group's attributes carried.
- **Vectors are columns.** A one-dimensional dataset is written N×1,
  MATLAB's own vector orientation, so it reads back one-dimensional;
  a genuinely two-dimensional array (`ordinate`, records × samples)
  stays as it is even with one row. MATLAB has no one-dimensional
  array, so a matrix that happens to have one column (a sine tone's
  amplitude over one channel) would read back as a vector; a struct
  holding one carries a `matrix_fields` cellstr naming it, written
  only then, and read if present.

HDF5 attributes and datasets both become plain fields — `frf.function_type`
beside `frf.ordinate` — because a MATLAB struct has no second kind of
member and a hidden marker would be clutter. On the way back the
native loader asks for each by the kind it expects, so the rebuild
writes every field as a dataset and, where it is small enough to be
one, as an attribute too; the loader reads the one it wants and the
other is never looked at. Strings are char arrays, lists of strings
are cellstr, and `''` (an undeclared unit) is MATLAB's empty char.

**Values are SI**, exactly as in a `.vdyn`, with every record's unit
named beside it — the one other format that stores SI regardless of
the display system is ADF, whose definition requires it; here it is
what keeps the file the project file. A record whose units were never
declared is written as it stands, marked `''`, and comes back
undeclared. Convert in MATLAB from the unit strings if a display
system is wanted.

The stamp `visualdynamics_schema` travels as a variable, and the
reader keeps the `.vdyn` contract: a newer stamp is refused, an older
one always loads. A `.mat` with no stamp is accepted in one case only —
a single struct in the documented layout of one object (a data array
with `abscissa`, `ordinate` and `data_class`, a geometry with
`node_xyz`, a shape set with `shape_matrix`), which is a user's own
struct built to the page — and refused by name otherwise: reading a
stranger's workspace would be guessing.

`.mat` version 5 (what `save -v7` writes, and what scipy reads and
writes) holds at most 4 GB per variable. A record larger than that is
refused before anything is written, naming the array; version 7.3
files are HDF5 and are not read or written here yet.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from . import native

if TYPE_CHECKING:                                    # pragma: no cover
    import h5py

SUFFIX = '.mat'
STAMP = 'visualdynamics_schema'
MATRIX_FIELDS = 'matrix_fields'
#: scipy's MAT-5 writer raises once a matrix passes 2**32 bytes — after
#: writing most of the file. Refuse first, and say which array.
MAX_BYTES = 2 ** 32 - 1
#: an HDF5 attribute is refused past 64 KB; anything a loader reads as
#: an attribute is far under this
_ATTR_BYTES = 16 * 1024
#: a struct built by hand to the documented layout, recognised by the
#: one field each kind cannot lack
_BARE_KINDS = (('data', ('abscissa', 'ordinate')),
               ('geometry', ('node_xyz',)),
               ('shapes', ('shape_matrix',)),
               ('channel_table', ('column_order',)))
_DIGITS = re.compile(r'^\d+$')


def handles(obj: Any) -> bool:
    """Every object the project file holds, and a whole project."""
    from ..project import Project

    if isinstance(obj, Project):
        return True
    try:
        native._saver_for(obj)
    except TypeError:
        return False
    return True


def sniff(path: str | os.PathLike) -> bool:
    """A `.mat` by suffix; whether it is ours is `load`'s to say, by
    name, since a wrong-format error beats "no importer recognises"."""
    return str(path).endswith(SUFFIX)


# ---- HDF5 tree -> MATLAB variables ------------------------------------------

def _memory_file() -> h5py.File:
    import h5py

    # the core driver with no backing store is an HDF5 file that never
    # touches disk; the name is a label
    return h5py.File('visualdynamics.mat', 'w', driver='core',
                     backing_store=False)


def _cell(items: list) -> np.ndarray:
    """A one-dimensional object array holding exactly these items —
    built by hand, because `np.array` given a list of equal-length
    arrays makes a matrix of their elements instead of a cell of
    arrays (one traceline of four nodes came back as four of one)."""
    cell = np.empty(len(items), dtype=object)
    for k, item in enumerate(items):
        cell[k] = item
    return cell


def _attr_out(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.ndarray) and value.dtype.kind in 'OSU':
        return np.array([str(v) for v in value], dtype=object)
    return value


def _dataset_out(dataset: h5py.Dataset) -> Any:
    import h5py

    if h5py.check_string_dtype(dataset.dtype) is not None:
        value = dataset[()]
        if isinstance(value, bytes):                 # one string, not a list
            return value.decode('utf-8')
        return np.array([v.decode('utf-8') for v in value], dtype=object)
    return dataset[()]


def _tree_out(group: h5py.Group) -> dict[str, Any]:
    """A group as a dict of MATLAB-ready values, the three respellings
    applied."""
    import h5py

    out: dict[str, Any] = {}
    for name, value in group.attrs.items():
        out[name] = _attr_out(value)
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            members = list(item.items())
            if members and all(isinstance(sub, h5py.Group)
                               and _DIGITS.match(key)
                               for key, sub in members):
                ordered = sorted(members, key=lambda kv: int(kv[0]))
                out[name] = _cell([_tree_out(sub) for _, sub in ordered])
            elif not members and name == 'objects':
                out[name] = _cell([])
            else:
                out[name] = _tree_out(item)
        else:
            out[name] = _dataset_out(item)
    for name in [n for n in out if n.endswith('_flat')]:
        stem = name[:-len('_flat')]
        if f'{stem}_offsets' in out:
            flat, offsets = out.pop(name), out.pop(f'{stem}_offsets')
            out[stem] = _cell([flat[offsets[i]:offsets[i + 1]]
                               for i in range(len(offsets) - 1)])
    one_column = [name for name, value in out.items()
                  if isinstance(value, np.ndarray) and value.dtype != object
                  and value.ndim == 2 and value.shape[1] == 1]
    if one_column:
        out[MATRIX_FIELDS] = np.array(one_column, dtype=object)
    return out


def _check_size(tree: dict[str, Any], where: str = '') -> None:
    for name, value in tree.items():
        label = f'{where}.{name}' if where else name
        if isinstance(value, dict):
            _check_size(value, label)
        elif isinstance(value, np.ndarray) and value.dtype == object:
            for k, item in enumerate(value):
                if isinstance(item, dict):
                    _check_size(item, f'{label}{{{k + 1}}}')
        elif isinstance(value, np.ndarray) and value.nbytes > MAX_BYTES:
            raise ValueError(
                f'{label} is {value.nbytes / 2 ** 30:.1f} GB, and a MATLAB '
                'version 5 file holds at most 4 GB per variable; save the '
                'project as .vdyn, which MATLAB reads with h5read')


def save(obj: Any, path: str | os.PathLike, unit_system: Any = None,
         **_ignored: Any) -> None:
    """Write an object, or a whole project, as a `.mat` file.

    `unit_system` is taken for the exporter's uniform signature and
    unused: the file stores SI with the units named, like the project
    file it is (see the module docstring).
    """
    from scipy.io import savemat

    from ..project import Project

    with _memory_file() as f:
        if isinstance(obj, Project):
            native.save_test_into(f, obj.name, dict(obj),
                                  active_geometry=obj.active_geometry,
                                  project_type=obj.project_type,
                                  links=obj.links, provenance=obj.provenance)
        else:
            native.save_into(obj, f)
        tree = _tree_out(f)
    _check_size(tree)
    path = str(path)
    if not path.endswith(SUFFIX):
        path += SUFFIX
    savemat(path, tree, do_compression=True, long_field_names=True,
            oned_as='column')


# ---- MATLAB variables -> HDF5 tree ------------------------------------------

def _is_struct(value: Any) -> bool:
    return hasattr(value, '_fieldnames')


def _unwrapped(item: Any) -> Any:
    while (isinstance(item, np.ndarray) and item.dtype == object
           and item.size == 1):
        item = item.ravel()[0]
    return item


def _value_in(value: Any) -> Any:
    """One loaded value in the vocabulary the rebuild speaks: dict for
    a struct, str, list[str], list[dict], list[np.ndarray] for the
    kinds of cell, np.ndarray otherwise."""
    if _is_struct(value):
        return {name: _value_in(getattr(value, name))
                for name in value._fieldnames}
    if not isinstance(value, np.ndarray):
        return value
    if value.dtype.kind == 'U':
        if value.size == 0:
            return ''
        return str(value[0]) if value.shape[0] == 1 else [str(r) for r in value]
    if value.dtype == object:
        items = list(value.ravel())
        if not items:
            return []
        if all(_is_struct(item) for item in items):
            # a struct array: one struct is a struct, and a struct with
            # several elements is read as the list it is
            structs = [_value_in(item) for item in items]
            return structs[0] if len(structs) == 1 else structs
        # a cell: each struct in one comes back as a 1×1 struct array
        # wrapping it, which is what tells a cell of one struct from a
        # struct — so the wrapping is looked at before it is removed
        items = [_unwrapped(item) for item in items]
        if all(_is_struct(item) for item in items):
            return [_value_in(item) for item in items]
        if all(isinstance(item, np.ndarray) and item.dtype.kind == 'U'
               for item in items):
            return [str(item[0]) if item.size else '' for item in items]
        return [np.asarray(item).ravel() for item in items]
    return value


def _write_field(group: h5py.Group, name: str, value: Any,
                 matrices: frozenset[str] = frozenset()) -> None:
    """One field back into HDF5, as whatever the loader may ask for."""
    import h5py

    if isinstance(value, dict):
        _write_tree(group.create_group(name), value)
    elif isinstance(value, str):
        group.attrs[name] = value
        group.create_dataset(name, data=value,
                             dtype=h5py.string_dtype('utf-8'))
    elif isinstance(value, list):
        if not value:
            # an empty cell could have been an empty list of strings,
            # an empty ragged array or an empty container; write what
            # each loader would look for
            if name == 'objects':
                group.create_group(name)
                return
            group.create_dataset(name, data=[], dtype=h5py.string_dtype('utf-8'))
            group.create_dataset(f'{name}_flat', data=np.empty(0, dtype=np.int64))
            group.create_dataset(f'{name}_offsets', data=np.zeros(1, dtype=np.int64))
        elif isinstance(value[0], dict):
            container = group.create_group(name)
            width = 4 if name == 'objects' else 1
            for k, item in enumerate(value):
                _write_tree(container.create_group(f'{k:0{width}d}'), item)
        elif isinstance(value[0], str):
            group.attrs[name] = value
            group.create_dataset(name, data=value,
                                 dtype=h5py.string_dtype('utf-8'))
        else:
            native._write_ragged(group, name, [np.asarray(v) for v in value])
    else:
        array = np.asarray(value)
        if array.ndim == 2 and array.shape == (0, 0):
            array = array.reshape(0)
        elif array.ndim == 2 and array.shape[1] == 1 and name not in matrices:
            array = array.reshape(-1)               # a column is a vector
        if array.size == 1 and array.ndim <= 1:
            group.attrs[name] = array.reshape(())[()]
        elif array.nbytes <= _ATTR_BYTES:
            group.attrs[name] = array
        group.create_dataset(name, data=array)


def _write_tree(group: h5py.Group, tree: dict[str, Any]) -> None:
    noted = tree.get(MATRIX_FIELDS, [])
    matrices = frozenset([noted] if isinstance(noted, str) else noted)
    for name, value in tree.items():
        if name != MATRIX_FIELDS:
            _write_field(group, name, value, matrices)


def _bare_kind(variables: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The one unstamped shape accepted: a single struct built to the
    documented layout of one object."""
    structs = {name: value for name, value in variables.items()
               if isinstance(value, dict)}
    if len(structs) != 1:
        return None
    (fields,) = structs.values()
    for kind, needed in _BARE_KINDS:
        if all(field in fields for field in needed):
            return kind, fields
    return None


def load(path: str | os.PathLike, **_ignored: Any) -> Any:
    """Read a `.mat` written here (or one struct built to the layout):
    the object it holds, or a whole project."""
    from scipy.io import loadmat

    path = str(path)
    try:
        raw = loadmat(path, squeeze_me=False, struct_as_record=False,
                      chars_as_strings=True)
    except NotImplementedError:
        raise ValueError(
            f'{path} is a MATLAB version 7.3 file, which is HDF5; save it '
            "with '-v7', or save the data as .vdyn") from None
    variables = {name: _value_in(value) for name, value in raw.items()
                 if not name.startswith('__')}
    if STAMP not in variables:
        bare = _bare_kind(variables)
        if bare is None:
            found = ', '.join(sorted(variables)) or 'nothing'
            raise ValueError(
                f'{path} is not a Visual Dynamics file (no {STAMP} '
                f'variable; it holds {found}). A struct built to the '
                'documented layout of one object is read too.')
        kind, fields = bare
        variables = {STAMP: native.SCHEMA_VERSION, kind: fields}
    with _memory_file() as f:
        _write_tree(f, variables)
        return native.load_from(f, path=path)
