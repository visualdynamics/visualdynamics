"""The Engineering Sciences Common Data Format, read and written by its
specification.

ESCDF (github.com/sandialabs/Engineering-Sciences-Common-Data-Format,
BSD-3-Clause) is one HDF5 file: metadata groups at the root — a
geometry, a channel table, a set of parameters — and an `activities`
group with one group per test or analysis, holding the results it
produced and a `parameters` list naming the metadata it links to. Every
dataset is a group stamped with the name of its specification, a
descriptive name and a version, and carries one HDF5 dataset per
property, each stamped with its type. What the types are is written in
plain-text specification files, one per type, and those files are the
standard: a reader that parses them reads every type there is, and a
type added tomorrow is a file added tomorrow. The files ship here as
package data (`escdf_specifications/`, with their license), and this
module reads them the way the reference implementation does, from the
grammar its documentation states — and was held to the reference's own
files, both ways, as it was written (2026-09-30; PLAN.md "ESCDF: import
and export first").

What this module is: the generic layer. `Specification` is a parsed
file; `Dataset` is one group's worth of values against its
specification, validated the way the reference validates — every
property named, every shared dimension the same size wherever it
appears, one choice of every either-or group, an enumeration honored —
and `File` is the whole file with its activities. `read` and `write`
move a `File` to and from disk on h5py. What it is not: the mapping
from these to Visual Dynamics' own objects, which is `escdf_objects`.

The on-disk conventions, as measured from files the reference wrote:
strings are variable-length UTF-8, a scalar string a zero-dimensional
dataset; bytes are variable-length uint8; a `variable_length` property
is a one-dimensional dataset of variable-length rows of its type;
numbers are the type the specification names, `u8` meaning eight
*bytes*, so `uint64`, and `c16` `complex128`; every property dataset
carries a `data_type` attribute naming that type; every group carries
`_specification_name`, `_descriptive_name` and `_version` (three
integers); an activity group carries `activity_name` and
`activity_date` (ISO 8601, UTC) and a `parameters` string dataset of
the metadata names it links to; the file carries `created_by` and
`created_date`. A choice group is written as whichever alternative was
chosen, under the shared property name, and read back by which shape
and type is found.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import h5py
import numpy as np

__all__ = [
    'Activity',
    'Dataset',
    'File',
    'Property',
    'Specification',
    'read',
    'specifications',
    'valid_identifier',
    'write',
]

#: the version a file written here claims for each dataset: the
#: specifications' own, read from their first line
_NUMPY_TYPES = {
    'u1': np.uint8, 'u2': np.uint16, 'u4': np.uint32, 'u8': np.uint64,
    'i1': np.int8, 'i2': np.int16, 'i4': np.int32, 'i8': np.int64,
    'f4': np.float32, 'f8': np.float64,
    'c8': np.complex64, 'c16': np.complex128,
}

_IDENTIFIER = re.compile(r'^[a-zA-Z][a-zA-Z0-9_]*$')


def valid_identifier(name: str, prefix: str = 'item_') -> str:
    """The format's name for a group: ASCII letters, digits and
    underscores, starting with a letter — whitespace to underscores,
    anything else dropped, `prefix` in front if what is left does not
    start with a letter (the reference's own repair, so a name repaired
    here is the name it would give).

    Parameters
    ----------
    name : str
        Any text.
    prefix : str, default 'item_'
        What to put in front when the repaired name starts with a digit
        or is empty.

    Returns
    -------
    str
    """
    repaired = re.sub(r'[^a-zA-Z0-9_]', '', re.sub(r'\s+', '_', name))
    if not _IDENTIFIER.match(repaired):
        repaired = prefix + repaired
    return repaired


@dataclass(frozen=True)
class Property:
    """One property of a specification: its name, type, shape and options.

    `shape` is the specification's own words: named dimensions
    ('num_nodes'), literal sizes (3), or () for a scalar. `choice` is
    (group, alternative) for a property in an either-or group, where
    exactly one alternative of the group is written.
    """

    name: str
    dtype: str                                  #: 'f8', 'u8', 'str', 'bytes'...
    shape: tuple[str | int, ...] = ()
    optional: bool = False
    variable_length: bool = False
    enum: str | None = None
    regex: str | None = None
    choice: tuple[str, str] | None = None

    @property
    def key(self) -> str:
        """What the on-disk dataset is called: the name, shared by the
        alternatives of a choice group."""
        return self.name


@dataclass
class Specification:
    """One parsed specification file.

    Attributes:
        name: The type's name, the file's first word.
        version: Three integers from the first line's 'vX.Y.Z'.
        extends: The type this one inherits from, or None.
        properties: This file's own properties, in order.
        enumerations: Each enumeration's allowed values.
        doc: The prose between the header and the properties block.
    """

    name: str
    version: tuple[int, int, int]
    extends: str | None
    properties: list[Property] = field(default_factory=list)
    enumerations: dict[str, list[str]] = field(default_factory=dict)
    doc: str = ''
    _parents: dict[str, Specification] = field(default_factory=dict, repr=False)

    def all_properties(self) -> list[Property]:
        """Every property, inherited ones first, a child's redefinition
        of a parent's name replacing the parent's (a `data_type` that
        narrows its enumeration, as `response_spectrum` does)."""
        chain: list[Specification] = []
        spec: Specification | None = self
        while spec is not None:
            chain.append(spec)
            spec = self._parents.get(spec.extends) if spec.extends else None
        merged: dict[str, list[Property]] = {}
        for spec in reversed(chain):
            own: dict[str, list[Property]] = {}
            for prop in spec.properties:
                own.setdefault(prop.name, []).append(prop)
            for name, props in own.items():
                merged[name] = props
        return [prop for props in merged.values() for prop in props]

    def all_enumerations(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        spec: Specification | None = self
        chain: list[Specification] = []
        while spec is not None:
            chain.append(spec)
            spec = self._parents.get(spec.extends) if spec.extends else None
        for spec in reversed(chain):
            out.update(spec.enumerations)
        return out

    def is_a(self, name: str) -> bool:
        """Whether this type is `name` or extends it."""
        spec: Specification | None = self
        while spec is not None:
            if spec.name == name:
                return True
            spec = self._parents.get(spec.extends) if spec.extends else None
        return False


def parse_specification(text: str) -> Specification:
    """A specification file's text, parsed by the documented grammar.

    The first line is 'name - vX.Y.Z', underlined; an 'extends: other'
    line names the parent; prose follows until a 'properties' header;
    each property line is 'name - type [- shape [- options]]', the
    fields split on ' - ', the shape comma-separated names or numbers or
    the word 'scalar', the options comma-separated from 'optional',
    'variable_length', 'enum:<name>', 'regex:<pattern>' and
    'or:<group>:<alternative>'; an 'enumerations' block lists
    'name - a, b, c'; a 'notes' or 'Examples' block ends it.

    Parameters
    ----------
    text : str
        The file's contents.

    Returns
    -------
    Specification
    """
    lines = text.splitlines()
    head = next(line for line in lines if line.strip())
    name, _, version = head.partition(' - ')
    numbers = tuple(int(v) for v in version.strip().lstrip('v').split('.'))
    spec = Specification(name.strip(), (numbers + (0, 0, 0))[:3], None)
    section = 'doc'
    doc: list[str] = []
    for i, line in enumerate(lines[1:]):
        stripped = line.strip()
        if set(stripped) == {'-'} and stripped:
            continue                                  # an underline
        lowered = stripped.lower()
        if lowered in ('properties', 'enumerations', 'notes', 'examples'):
            section = lowered
            continue
        if section == 'doc':
            if lowered.startswith('extends:'):
                parent = stripped.split(':', 1)[1].strip()
                spec.extends = None if parent.lower() == 'none' else parent
            elif stripped:
                doc.append(stripped)
        elif section == 'properties' and stripped:
            spec.properties.append(_parse_property(stripped))
        elif section == 'enumerations' and stripped:
            enum_name, _, values = stripped.partition(' - ')
            spec.enumerations[enum_name.strip()] = [
                v.strip() for v in values.split(',') if v.strip()]
    spec.doc = ' '.join(doc)
    return spec


def _parse_property(line: str) -> Property:
    fields = [f.strip() for f in line.split(' - ')]
    if len(fields) < 2:
        raise ValueError(f'a property line needs a name and a type: {line!r}')
    name, dtype = fields[0], fields[1]
    shape: tuple[str | int, ...] = ()
    options = ''
    if len(fields) >= 3:
        # a third field that is a size is the shape; one that reads as
        # options ('optional', 'or:...') on a scalar property is options
        third = fields[2]
        if _looks_like_shape(third):
            shape = tuple(int(s) if s.strip().isdigit() else s.strip()
                          for s in third.split(',') if s.strip() and s.strip() != 'scalar')
            options = ' - '.join(fields[3:])
        else:
            options = ' - '.join(fields[2:])
    prop = {'optional': False, 'variable_length': False, 'enum': None,
            'regex': None, 'choice': None}
    for option in _split_options(options):
        if option == 'optional':
            prop['optional'] = True
        elif option == 'variable_length':
            prop['variable_length'] = True
        elif option.startswith('enum:'):
            prop['enum'] = option[5:].strip()
        elif option.startswith('regex:'):
            prop['regex'] = option[6:].strip()
        elif option.startswith('or:'):
            _, group, alternative = option.split(':', 2)
            prop['choice'] = (group.strip(), alternative.strip())
        elif option:
            raise ValueError(f'unknown property option {option!r} in {line!r}')
    return Property(name, dtype, shape, **prop)   # type: ignore[arg-type]


def _looks_like_shape(text: str) -> bool:
    parts = [p.strip() for p in text.split(',')]
    return all(p == 'scalar' or p.isdigit() or _IDENTIFIER.match(p) and not (
        p in ('optional', 'variable_length') or ':' in p) for p in parts)


def _split_options(text: str) -> list[str]:
    """Options split on commas, except inside a regex, which runs to the
    end of the line and may hold commas of its own."""
    out: list[str] = []
    rest = text.strip()
    while rest:
        if rest.startswith('regex:'):
            out.append(rest)
            break
        item, _, rest = rest.partition(',')
        rest = rest.strip()
        if item.strip():
            out.append(item.strip())
    return out


def _package_specifications() -> dict[str, Specification]:
    folder = resources.files('visualdynamics.io').joinpath('escdf_specifications')
    specs: dict[str, Specification] = {}
    for entry in folder.iterdir():
        if entry.name.endswith('.txt'):
            spec = parse_specification(entry.read_text(encoding='utf-8'))
            specs[spec.name] = spec
    for spec in specs.values():
        spec._parents = specs
    return specs


_SPECIFICATIONS: dict[str, Specification] | None = None


def specifications() -> dict[str, Specification]:
    """Every type the vendored specification files define, by name,
    each knowing its parents.

    Returns
    -------
    dict of str to Specification
    """
    global _SPECIFICATIONS
    if _SPECIFICATIONS is None:
        _SPECIFICATIONS = _package_specifications()
    return _SPECIFICATIONS


@dataclass
class Dataset:
    """One dataset: a group's values against its specification.

    Attributes:
        name: The group's name, a valid identifier.
        kind: The specification's name ('geometry', 'data', ...).
        descriptive_name: Free text, the reference's `_descriptive_name`.
        values: Property name to value. A string property is a str or
            an array of str; a bytes property a list of uint8 arrays; a
            variable-length property a list of arrays; numbers arrays.
        version: The specification version the group claims, or None
            to write the vendored specification's own.
        extras: Datasets in the group the specification does not name,
            kept as read so a re-export loses nothing.
    """

    name: str
    kind: str
    descriptive_name: str = ''
    values: dict[str, Any] = field(default_factory=dict)
    version: tuple[int, int, int] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def specification(self) -> Specification | None:
        return specifications().get(self.kind)

    def problems(self) -> list[str]:
        """What keeps this dataset from validating against its
        specification, as the reference would refuse it: a required
        property missing, a choice group with none or two alternatives
        chosen, a shared dimension of two sizes, a type the value cannot
        take, an enumeration or pattern not honored. Empty when valid;
        an unknown type is a problem of its own.

        Returns
        -------
        list of str
        """
        spec = self.specification
        if spec is None:
            # a type nobody defined is kept as read, everything of it in
            # extras — the reference loads one as `unknown` and keeps its
            # properties too (measured 2026-09-30, at the root and in an
            # activity alike); values it has no specification for cannot
            # be checked, so it may carry none
            unknown = (f'{self.kind!r} is not a type the specifications define, '
                       'so it can carry nothing but extras')
            return [unknown] if self.values else []
        problems: list[str] = []
        props = spec.all_properties()
        enums = spec.all_enumerations()
        by_name: dict[str, list[Property]] = {}
        for prop in props:
            by_name.setdefault(prop.name, []).append(prop)
        dims: dict[str, int] = {}
        chosen: dict[str, list[str]] = {}
        for name, alternatives in by_name.items():
            value = self.values.get(name)
            if value is None:
                if all(p.optional for p in alternatives):
                    continue
                if alternatives[0].choice is None:
                    problems.append(f'{name} is required')
                continue
            matches = [p for p in alternatives if _fits(p, value)]
            if not matches:
                problems.append(f'{name}: {_describe(value)} does not fit '
                                + ' or '.join(_shape_text(p) for p in alternatives))
                continue
            prop = matches[0]
            if prop.choice is not None:
                chosen.setdefault(prop.choice[0], []).append(prop.choice[1])
            shape = _value_shape(value, prop)
            for axis, size in zip(prop.shape, shape):
                if isinstance(axis, str) and dims.setdefault(axis, size) != size:
                    problems.append(f'{name}: {axis} is {size} here and '
                                    f'{dims[axis]} elsewhere')
            if prop.enum is not None:
                allowed = set(enums.get(prop.enum, []))
                for item in np.asarray(value, dtype=object).ravel():
                    if str(item) not in allowed:
                        problems.append(f'{name}: {item!r} is not one of '
                                        + ', '.join(sorted(allowed)))
                        break
            if prop.regex is not None:
                pattern = re.compile(prop.regex)
                for item in np.asarray(value, dtype=object).ravel():
                    if not pattern.match(str(item)):
                        problems.append(f'{name}: {item!r} does not match '
                                        f'{prop.regex}')
                        break
        groups: dict[str, set[str]] = {}
        for prop in props:
            if prop.choice is not None:
                groups.setdefault(prop.choice[0], set()).add(prop.choice[1])
        for group, alternatives in groups.items():
            picked = set(chosen.get(group, []))
            # one alternative may cover several properties (even spacing
            # is a start and a step); every property of it must be there
            if not picked:
                if any(p.optional for p in props if p.choice and p.choice[0] == group):
                    continue
                problems.append(f'choice {group}: none of '
                                + ', '.join(sorted(alternatives)) + ' is given')
            elif len(picked) > 1:
                problems.append(f'choice {group}: both ' + ' and '.join(sorted(picked)))
            else:
                alternative = next(iter(picked))
                wanted = [p.name for p in props
                          if p.choice == (group, alternative)]
                missing = [n for n in wanted if n not in self.values]
                if missing:
                    problems.append(f'choice {group}/{alternative} also needs '
                                    + ', '.join(missing))
        return problems


def _describe(value: Any) -> str:
    array = np.asarray(value, dtype=object) if isinstance(value, list) else np.asarray(value)
    return f'shape {array.shape}, {array.dtype}'


def _shape_text(prop: Property) -> str:
    return f'{prop.dtype} {",".join(str(s) for s in prop.shape) or "scalar"}'


def _value_shape(value: Any, prop: Property) -> tuple[int, ...]:
    if prop.dtype == 'bytes' or prop.variable_length:
        return (len(value),)
    if prop.dtype == 'str':
        return np.asarray(value, dtype=object).shape if not isinstance(value, str) else ()
    return np.asarray(value).shape


def _fits(prop: Property, value: Any) -> bool:
    """Whether a value can be written as this property: the right rank,
    literal sizes matched, and a type the value casts to safely — a
    complex value cannot be a real property, a string not a number."""
    try:
        shape = _value_shape(value, prop)
    except (TypeError, ValueError):
        return False
    if len(shape) != len(prop.shape):
        return False
    for axis, size in zip(prop.shape, shape):
        if isinstance(axis, int) and axis != size:
            return False
    if prop.dtype == 'str':
        items = [value] if isinstance(value, str) else np.asarray(value, dtype=object).ravel()
        return all(isinstance(item, str) for item in items)
    if prop.dtype == 'bytes':
        return all(np.asarray(item).dtype == np.uint8 for item in value)
    numpy_type = _NUMPY_TYPES.get(prop.dtype)
    if numpy_type is None:
        return False
    items = value if prop.variable_length else [value]
    for item in items:
        array = np.asarray(item)
        if array.dtype == object or array.dtype.kind in 'USb':
            return False
        if array.dtype.kind == 'c' and numpy_type(0).dtype.kind != 'c':
            return False
        if array.dtype.kind == 'f' and numpy_type(0).dtype.kind in 'iu':
            return False
        if array.dtype.kind in 'iu' and numpy_type(0).dtype.kind == 'u' and array.size and (array < 0).any():
            return False
    return True


@dataclass
class Activity:
    """One activity: a test or an analysis, its results and the metadata
    it links to."""

    name: str
    descriptive_name: str = ''
    date: dt.datetime | None = None
    links: list[str] = field(default_factory=list)
    data: dict[str, Dataset] = field(default_factory=dict)


@dataclass
class File:
    """A whole file: who made it and when, the metadata at its root, and
    its activities."""

    created_by: str = ''
    created_date: dt.datetime | None = None
    metadata: dict[str, Dataset] = field(default_factory=dict)
    activities: dict[str, Activity] = field(default_factory=dict)

    def problems(self) -> list[str]:
        """Every dataset's problems, each prefixed with where it is, and
        a link that names no metadata."""
        out = [f'{name}: {p}' for name, ds in self.metadata.items()
               for p in ds.problems()]
        specs = specifications()
        for activity in self.activities.values():
            for link in activity.links:
                if link not in self.metadata:
                    out.append(f'{activity.name} links to {link!r}, which is not '
                               'in the metadata')
            for name, ds in activity.data.items():
                out.extend(f'{activity.name}/{name}: {p}' for p in ds.problems())
                # the reference refuses a known type that is not a result
                # inside an activity (a parameter set belongs at the root,
                # linked); an unknown type it takes either place
                spec = specs.get(ds.kind)
                if spec is not None and not spec.is_a('activity_result'):
                    out.append(f'{activity.name}/{name}: {ds.kind} is not an '
                               'activity result; it belongs in the metadata, linked')
        return out


# ---- disk ----------------------------------------------------------------

def _iso(when: dt.datetime | None) -> str:
    when = when or dt.datetime.now(dt.UTC)
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.UTC)
    return when.astimezone(dt.UTC).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def _from_iso(text: Any) -> dt.datetime | None:
    if isinstance(text, (bytes, np.bytes_)):
        text = text.decode()
    if not text:
        return None
    text = str(text).strip()
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        when = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=dt.UTC)


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def _write_value(group: h5py.Group, prop: Property, value: Any) -> None:
    if prop.dtype == 'str':
        dtype = h5py.string_dtype('utf-8')
        if isinstance(value, str):
            dataset = group.create_dataset(prop.name, (), dtype=dtype)
            dataset[()] = value
        else:
            array = np.asarray(value, dtype=object)
            dataset = group.create_dataset(prop.name, array.shape, dtype=dtype)
            if array.size:
                dataset[...] = array
    elif prop.dtype == 'bytes':
        dtype = h5py.vlen_dtype(np.dtype('uint8'))
        dataset = group.create_dataset(prop.name, (len(value),), dtype=dtype)
        for i, item in enumerate(value):
            dataset[i] = np.asarray(item, dtype=np.uint8)
    elif prop.variable_length:
        numpy_type = _NUMPY_TYPES[prop.dtype]
        dataset = group.create_dataset(prop.name, (len(value),),
                                       dtype=h5py.vlen_dtype(numpy_type))
        for i, item in enumerate(value):
            dataset[i] = np.asarray(item, dtype=numpy_type)
    else:
        array = np.asarray(value, dtype=_NUMPY_TYPES[prop.dtype])
        dataset = group.create_dataset(prop.name, array.shape,
                                       dtype=array.dtype)
        dataset[()] = array
    dataset.attrs['data_type'] = prop.dtype


def _read_value(dataset: h5py.Dataset) -> Any:
    data_type = _text(dataset.attrs.get('data_type', ''))
    raw = dataset[()]
    if data_type == 'str' or h5py.check_string_dtype(dataset.dtype) is not None:
        if dataset.shape == ():
            return _text(raw)
        return np.array([_text(v) for v in np.asarray(raw, dtype=object).ravel()],
                        dtype=object).reshape(dataset.shape)
    if data_type == 'bytes':
        return [np.asarray(item, dtype=np.uint8) for item in raw]
    if h5py.check_vlen_dtype(dataset.dtype) is not None:
        base = h5py.check_vlen_dtype(dataset.dtype)
        return [np.asarray(item, dtype=base) for item in raw]
    return raw


def _write_dataset(parent: h5py.Group, dataset: Dataset) -> None:
    spec = dataset.specification
    group = parent.create_group(dataset.name)
    group.attrs['_specification_name'] = dataset.kind
    group.attrs['_descriptive_name'] = dataset.descriptive_name
    version = dataset.version or (spec.version if spec else (0, 0, 0))
    group.attrs['_version'] = np.array(version, dtype=np.int64)
    props = spec.all_properties() if spec else []
    by_name: dict[str, list[Property]] = {}
    for prop in props:
        by_name.setdefault(prop.name, []).append(prop)
    for name, value in dataset.values.items():
        alternatives = by_name.get(name)
        if not alternatives:
            raise ValueError(f'{dataset.name}: {name} is not a property of '
                             f'{dataset.kind}; put what the type does not '
                             'define in extras')
        chosen = next((p for p in alternatives if _fits(p, value)), None)
        if chosen is None:
            raise ValueError(f'{dataset.name}: {name} {_describe(value)} fits '
                             'none of ' + ' or '.join(_shape_text(p) for p in alternatives))
        _write_value(group, chosen, value)
    for name, (data_type, value) in dataset.extras.items():
        extra = Property(name, data_type, variable_length=isinstance(value, list)
                         and data_type not in ('str', 'bytes'))
        _write_value(group, extra, value)


def _read_dataset(group: h5py.Group) -> Dataset:
    kind = _text(group.attrs.get('_specification_name', 'unknown'))
    dataset = Dataset(group.name.rsplit('/', 1)[-1], kind,
                      _text(group.attrs.get('_descriptive_name', '')))
    version = group.attrs.get('_version')
    if version is not None and len(version) == 3:
        dataset.version = tuple(int(v) for v in version)
    spec = dataset.specification
    known = {p.name for p in spec.all_properties()} if spec else set()
    for name, item in group.items():
        if not isinstance(item, h5py.Dataset):
            continue
        value = _read_value(item)
        if name in known:
            dataset.values[name] = value
        else:
            dataset.extras[name] = (_text(item.attrs.get('data_type', '')), value)
    return dataset


def write(file: File, path: str | os.PathLike) -> None:
    """Write a `File` as an ESCDF file, validating first.

    Parameters
    ----------
    file : File
        What to write; `file.problems()` must be empty.
    path : path-like
        The file to write; replaced if it exists.
    """
    problems = file.problems()
    if problems:
        raise ValueError('not a valid ESCDF file:\n  ' + '\n  '.join(problems))
    with h5py.File(path, 'w') as h5:
        h5.attrs['created_by'] = file.created_by or 'unknown'
        h5.attrs['created_date'] = _iso(file.created_date)
        for dataset in file.metadata.values():
            _write_dataset(h5, dataset)
        activities = h5.create_group('activities')
        for activity in file.activities.values():
            group = activities.create_group(activity.name)
            group.attrs['activity_name'] = activity.descriptive_name
            group.attrs['activity_date'] = _iso(activity.date)
            links = group.create_dataset('parameters', (len(activity.links),),
                                         dtype=h5py.string_dtype('utf-8'))
            if activity.links:
                links[...] = np.asarray(activity.links, dtype=object)
            links.attrs['data_type'] = 'str'
            for dataset in activity.data.values():
                _write_dataset(group, dataset)


def read(path: str | os.PathLike) -> File:
    """Read an ESCDF file whole: every dataset against its
    specification, unknown types and undefined properties kept as they
    are.

    Parameters
    ----------
    path : path-like
        The file.

    Returns
    -------
    File
    """
    file = File()
    with h5py.File(path, 'r') as h5:
        file.created_by = _text(h5.attrs.get('created_by', ''))
        file.created_date = _from_iso(h5.attrs.get('created_date', ''))
        for name, item in h5.items():
            if name == 'activities' or not isinstance(item, h5py.Group):
                continue
            file.metadata[name] = _read_dataset(item)
        for name, group in h5.get('activities', {}).items():
            if not isinstance(group, h5py.Group):
                continue
            activity = Activity(name, _text(group.attrs.get('activity_name', '')),
                                _from_iso(group.attrs.get('activity_date', '')))
            if 'parameters' in group:
                activity.links = [_text(v) for v in np.asarray(group['parameters'][()]).ravel()]
            for item_name, item in group.items():
                if isinstance(item, h5py.Group):
                    activity.data[item_name] = _read_dataset(item)
            file.activities[name] = activity
    return file


def sniff(path: str | os.PathLike) -> bool:
    """An HDF5 file with an `activities` group and the file's own
    creation stamps."""
    try:
        with h5py.File(path, 'r') as h5:
            return 'activities' in h5 and 'created_by' in h5.attrs
    except (OSError, ValueError):
        return False


def datasets_of(file: File, kind: str) -> Iterable[tuple[str | None, Dataset]]:
    """Every dataset in the file that is `kind` or extends it, with the
    activity it belongs to (None for metadata), in file order."""
    specs = specifications()
    for dataset in file.metadata.values():
        spec = specs.get(dataset.kind)
        if spec is not None and spec.is_a(kind):
            yield None, dataset
    for activity in file.activities.values():
        for dataset in activity.data.values():
            spec = specs.get(dataset.kind)
            if spec is not None and spec.is_a(kind):
                yield activity.name, dataset

