"""MATLAB `.mat` files: the project file in MATLAB's container.

A `.mat` written here has the `.vdyn` schema and names
(`docs/vdyn-format.md`), so a MATLAB user gets `load('modal.mat')` and
structs with the right shapes and complex numbers assembled, rather
than HDF5 datasets to read one at a time with `h5read` (which a
`.vdyn` already allows). Everything a project holds has a MATLAB form,
so the reader reads its own files back completely.

The layout is implemented once, and not in this module. The writer
runs the native `.vdyn` writer into an HDF5 file held in memory and
converts that tree into `.mat` variables; the reader rebuilds an
in-memory HDF5 tree from the `.mat` and passes it to the native
loader. A field added to the schema is carried with no change here,
and the two file types cannot drift apart, because neither this module
nor `native.py` knows two layouts (principle 9). It costs one
in-memory HDF5 round trip per file, which is fast.

MATLAB holds three things differently from HDF5, and the same rules
convert them in both directions:

- Ragged arrays, the `<name>_flat` / `<name>_offsets` pairs (element
  connectivity), are one cell array `<name>` with one row vector per
  entry, which is how MATLAB holds a ragged list.
- Numbered groups, the project's `objects/0000, 0001, …` and a sine
  specification's `tones/0, 1, …`, are a cell array of structs in the
  same order (`p.objects{3}.name`); each object struct keeps the
  `name` and `kind` its group's attributes held.
- Vectors are columns. A one-dimensional dataset is written N×1,
  MATLAB's own vector orientation, so it reads back one-dimensional;
  a two-dimensional array (`ordinate`, records × samples) stays as it
  is even with one row. MATLAB has no one-dimensional array, so a
  matrix with one column (a sine tone's amplitude over one channel)
  would read back as a vector. A struct holding one has a
  `matrix_fields` cellstr naming it, written only in that case and
  read if present.

HDF5 attributes and datasets both become plain fields
(`frf.function_type` beside `frf.ordinate`), because a MATLAB struct
has only one kind of member and a hidden marker would be clutter. On
the way back the native loader asks for each field as the kind it
expects, so the rebuild writes every field as a dataset and, where it
is small enough, as an attribute too; the loader reads the one it
wants. Strings are char arrays, lists of strings are cellstr, and
`''` (an undeclared unit) is MATLAB's empty char.

Values are in the display units, and every unit field names the unit
its values are in. A `.mat` exported in a display system holds each
record in that system's unit for its quantity, labeled with it: g
values with an `ordinate_unit` of `'g'`, `(in/s**2)/lbf` with an FRF
in those units, a geometry's coordinates in inches with `length_unit`
`'in'`, shapes in slinch. The file names the system in a
`unit_system` variable, and the reader converts each object back to SI
from its own labels, so on reading the declared unit is the one it was
written in. Some values stay SI in every system, because the project
file defines them as SI: material and section properties, and the
abscissa, which is seconds or hertz. A record whose units were never
declared is written as it is, marked `''`, and is read back
undeclared. Without a display system (`Project.save` to `.mat`, or a
script that names none) the file holds exactly what the project file
holds, in SI, and a file with no `unit_system` variable is read as SI.

The conversion is the core's own, run on a copy: each object is
labeled with the display unit and `undefine_units` converts its values
to it, the inverse of how a declaration converts a file's numbers to
SI; the reader clears the labels and declares them again
(`_in_display`, `_from_display`).

The stamp `visualdynamics_schema` is written as a variable, and the
reader follows the `.vdyn` rule: a newer stamp is refused, and an
older one always loads. A `.mat` with no stamp is accepted in one case
only, a single struct in the documented layout of one object (a data
array with `abscissa`, `ordinate` and `data_class` or `function_type`,
a geometry with `node_xyz`, a shape set with `shape_matrix`, a channel
table with `column_order`), which is a user's own struct built to that
layout. Any other `.mat` is refused with a ValueError naming the
variables it holds, because reading an arbitrary workspace would be
guessing.

`.mat` version 5 (what `save -v7` writes, and what scipy reads and
writes) holds at most 4 GB per variable. A record larger than that is
refused before anything is written, with the array named; version 7.3
files are HDF5 and are not read or written here yet.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from . import native

# Values in the display units, labeled (Brandon, 2026-10-13): a time
# history declared in g, exported with in-slinch-lbf-s (g) on, came out
# in m/s² beside an `ordinate_unit` of 'g'. That is the project file's
# field, which names the unit a record was declared in over values that
# are always SI. A file from before 2026-10-13 has no unit_system and
# is read as SI.

if TYPE_CHECKING:                                    # pragma: no cover
    import h5py

SUFFIX = '.mat'
STAMP = 'visualdynamics_schema'
#: the display system a file's values are in, by name; absent, SI
SYSTEM = 'unit_system'
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
    """Whether `obj` can be written as a `.mat`: any object the project
    file holds, or a whole project."""
    from ..project import Project

    if isinstance(obj, Project):
        return True
    try:
        native._saver_for(obj)
    except TypeError:
        return False
    return True


def sniff(path: str | os.PathLike) -> bool:
    """Whether `path` ends in `.mat`, in any case (`.MAT` too).

    The suffix is the only check. Whether the file was written here is
    decided by `load`, which raises a ValueError saying what is wrong;
    that is more useful than a message that no importer recognized the
    file."""
    return str(path).lower().endswith(SUFFIX)


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
    arrays (one line of four nodes came back as four of one)."""
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

    With a `unit_system`, the values are in that system's units and
    each unit field names them; with None, the values are SI, exactly
    as the project file holds them (see the module docstring). `.mat`
    is added to `path` when it does not end in it. Raises ValueError,
    before anything is written, when an array is larger than the 4 GB
    a version 5 file allows per variable.
    """
    from scipy.io import savemat

    from ..project import Project

    def shown(item):
        return item if unit_system is None else _in_display(item, unit_system)

    with _memory_file() as f:
        if isinstance(obj, Project):
            native.save_test_into(f, obj.name,
                                  {name: shown(item) for name, item in obj.items()},
                                  active_geometry=obj.active_geometry,
                                  project_type=obj.project_type,
                                  object_groups=obj.object_groups, provenance=obj.provenance)
        else:
            native.save_into(shown(obj), f)
        tree = _tree_out(f)
    if unit_system is not None:
        tree[SYSTEM] = unit_system.name
    _check_size(tree)
    path = str(path)
    if not path.endswith(SUFFIX):
        path += SUFFIX
    savemat(path, tree, do_compression=True, long_field_names=True,
            oned_as='column')


# ---- the display units --------------------------------------------------------

def _in_display(obj: Any, system: Any) -> Any:
    """A copy of `obj` with its values in `system`'s units and its unit
    fields naming them; `obj` itself if it has no units, or none
    declared. Each kind is labeled with the display unit and its own
    `undefine_units` takes its SI values there."""
    import copy

    from ..core.data import DataArray
    from ..core.geometry import Geometry
    from ..core.shapes import ShapeSet
    from ..core.sine import SineLevelSet, SineSweepSpecification
    from ..units import si_transform

    if isinstance(obj, DataArray):
        rows = [i for i, unit in enumerate(obj.ordinate_unit) if unit is not None]
        if not rows:
            return obj
        out = copy.deepcopy(obj)
        labels = {i: type(out).units_for(out.ordinate_dim[i], system) for i in rows}
        dims = list(out.ordinate_dim)
        for i, (unit, reference) in labels.items():
            out.ordinate_unit[i], out.reference_unit[i] = unit, reference
        out.undefine_units(rows)
        for i, (unit, reference) in labels.items():
            out.ordinate_unit[i], out.reference_unit[i] = unit, reference
            out.ordinate_dim[i] = dims[i]
        return out
    if isinstance(obj, Geometry) and obj.length_unit is not None:
        out = copy.deepcopy(obj)
        out.length_unit = system.unit('length')
        out.undefine_units()
        out.length_unit = system.unit('length')
        return out
    if isinstance(obj, ShapeSet) and obj.mass_unit is not None:
        out = copy.deepcopy(obj)
        out.mass_unit = system.unit('mass')
        out.undefine_units()
        out.mass_unit = system.unit('mass')
        return out
    if isinstance(obj, SineLevelSet):
        out = copy.copy(obj)
        out.levels = [_in_display(level, system) for level in obj.levels]
        return out
    if isinstance(obj, SineSweepSpecification) and obj.ordinate_unit is not None:
        # its amplitudes are held in their SI unit, named: no declaration
        # to take back, so scaled here by the one transform
        out = copy.deepcopy(obj)
        out.ordinate_unit = system.unit(obj.ordinate_dim)
        scale, offset = si_transform(out.ordinate_unit)
        for tone in out.tones:
            tone.amplitude = (tone.amplitude - offset) / scale
            tone.limits = {name: (values - offset) / scale
                           for name, values in tone.limits.items()}
        return out
    return obj


def _from_display(obj: Any) -> Any:
    """`obj`, read from a file written in a display system, back in SI
    in place: each label cleared and declared again, so the values are
    taken from the unit it names — the inverse of `_in_display`."""
    from ..core.data import DataArray
    from ..core.geometry import Geometry
    from ..core.shapes import ShapeSet
    from ..core.sine import SineLevelSet, SineSweepSpecification
    from ..units import SI, si_transform

    if isinstance(obj, DataArray):
        units, references = list(obj.ordinate_unit), list(obj.reference_unit)
        obj.ordinate_unit = [None] * obj.num_records
        obj.reference_unit = [None] * obj.num_records
        obj.define_units(units, references)
    elif isinstance(obj, Geometry) and obj.length_unit is not None:
        unit, obj.length_unit = obj.length_unit, None
        obj.define_units(unit)
    elif isinstance(obj, ShapeSet) and obj.mass_unit is not None:
        unit, obj.mass_unit = obj.mass_unit, None
        obj.define_units(unit)
    elif isinstance(obj, SineLevelSet):
        for level in obj.levels:
            _from_display(level)
    elif isinstance(obj, SineSweepSpecification) and obj.ordinate_unit is not None:
        scale, offset = si_transform(obj.ordinate_unit)
        for tone in obj.tones:
            tone.amplitude = tone.amplitude * scale + offset
            tone.limits = {name: values * scale + offset
                           for name, values in tone.limits.items()}
        obj.ordinate_unit = SI.unit(obj.ordinate_dim)
    return obj


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
    """Read a `.mat` written here, or a single struct built to the
    documented layout of one object.

    Returns the object the file holds, or a `Project` for a whole
    project. Values written in a display system are converted
    back to SI. Raises ValueError for a version 7.3 file, for a schema
    stamp newer than this version reads, and for a file that has no
    `visualdynamics_schema` stamp and is not a single struct in the
    documented layout."""
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
    shown = variables.pop(SYSTEM, None)
    with _memory_file() as f:
        _write_tree(f, variables)
        loaded = native.load_from(f, path=path)
    if shown is None:
        return loaded
    for item in (loaded.values() if isinstance(loaded, dict) else [loaded]):
        _from_display(item)
    return loaded
