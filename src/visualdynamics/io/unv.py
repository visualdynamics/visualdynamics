"""Importer for Universal Files (.unv/.uff), per the SDRL specifications
(https://www.ceas3.uc.edu/sdrluff/ — local copies in docs/uff_spec).

Geometry datasets supported:

- 164  units: length/force/temperature factors; per spec, file values are
       DIVIDED by the factors to get SI, so a 164 makes the import unit-aware
- 15   nodes (old single-precision form, one line per node)
- 2411 nodes (double precision: 4I10 record + 3D25.16 record per node)
- 82   tracelines: node list where 0 means pen-up (move without drawing);
       pen-up runs are split into separate polylines
- 2412 elements: FE descriptor codes match visualdynamics's element vocabulary; beam-
       class descriptors carry an extra orientation record before the nodes
- 58b  the binary form of 58: the same eleven ASCII header records, then
       the values as raw floats. Read in either byte order at either
       precision, and written little-endian IEEE 754 doubles when
       `save(..., binary=True)` is asked for. Only IEEE 754 floats are
       decoded — DEC VMS and IBM 5/370 floats are refused by name rather
       than read as IEEE, which would return wrong numbers instead of
       failing.
- 58   functions at nodal DOFs (time histories, spectra, FRFs, PSDs). The
       ordinate scale to SI comes from the axis data-type unit exponents
       combined with the 164 factors; without a 164, unit-bearing data
       imports unit-less unless ordinate_unit is declared. Functions are
       grouped by type into one visualdynamics data object per type; unsupported
       function types (coherence, etc.) are skipped.

A file with several objects (e.g. geometry + FRFs) returns a dict keyed
'geometry', 'time', 'spectrum', 'frf', 'psd'; a file with one object returns
it directly. Unrecognized datasets are skipped (a UNV file is a stream of
independent datasets delimited by '-1' lines).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                                    # pragma: no cover
    from ..units import UnitSystem

import re

import numpy as np

from ..core.data import direction_code
from ..core.geometry import ELEMENT_TYPES, to_global, to_local
from .sniffing import text_head

_FLOAT = re.compile(r'[-+]?\d*\.?\d+(?:[DdEe][-+]?\d+)?')

# 2412 FE descriptors with the extra beam-definition record before the nodes
_BEAM_DESCRIPTORS = {11, 21, 22, 23, 24}

# Dataset 58 specific data type -> (dimension, length exponent, force exponent).
# The exponents express the quantity in UFF unit-system terms (length L, force
# F; time and temperature never rescale between UFF unit systems).
_58_DATA_TYPES = {
    0: ('unknown', 0, 0),  # type 0 is 'unknown' in the spec, and visualdynamics has a
                           # dimension for exactly that; calling it
                           # dimensionless would claim knowledge the file
                           # does not carry
    1: (None, 0, 0),  # general: exponents come from the record itself
    2: ('pressure', -2, 1),
    3: ('dimensionless', 0, 0),
    8: ('length', 1, 0),
    9: ('force', 0, 1),
    11: ('velocity', 1, 0),
    12: ('acceleration', 1, 0),
    13: ('force', 0, 1),
    15: ('pressure', -2, 1),
    16: ('mass', -1, 1),
    17: ('time', 0, 0),
    18: ('frequency', 0, 0),
}

# dataset 58's function types, and the name each becomes on import. 6 and 26
# are both coherence and are not interchangeable: 6 is one response against
# one reference, 26 is one response against all of them at once.
_58_NAMES = {1: 'time', 4: 'frf', 6: 'coherence', 9: 'psd', 12: 'spectrum',
             24: 'srs', 26: 'multiple_coherence'}

EXTENSIONS = ('.unv', '.uff', '.uf')


def _floats(line):
    return [float(tok.replace('D', 'E').replace('d', 'e'))
            for tok in _FLOAT.findall(line)]


def _ints(line):
    return [int(tok) for tok in line.split()]


#: The marker line of a binary dataset: the number, a lowercase 'b',
#: then the fields that say how to read what follows. Format
#: (I6,1A1,I6,I6,I12,I12,I6,I6,I12,I12) — byte ordering, floating point
#: format, the count of ASCII lines, and the count of bytes after them
#: (docs/uff_spec/58b.asc).
_BINARY_MARKER = re.compile(r'\s*(\d+)b\s*(-?\d+)\s+(-?\d+)\s+'
                            r'(-?\d+)\s+(-?\d+)')

#: Byte ordering method, field 3.
LITTLE_ENDIAN, BIG_ENDIAN = 1, 2

#: Floating point format, field 4. Only IEEE 754 is decoded: DEC VMS's
#: F/G float and IBM 5/370's hexadecimal float are different
#: representations, not different byte orders, and reading either as
#: IEEE would return plausible wrong numbers rather than fail.
IEEE_754 = 2

_FP_FORMATS = {1: 'DEC VMS', 2: 'IEEE 754', 3: 'IBM 5/370'}


@dataclass(frozen=True)
class Binary:
    """The binary half of a dataset 58b: how to read it, and it.

    `order` and `fp_format` are the marker line's own fields, carried
    with the bytes rather than read again from somewhere else — a blob
    whose byte order has been separated from it is a blob that will one
    day be decoded by the wrong one.
    """

    order: int
    fp_format: int
    data: bytes


def iter_datasets(
        data: str | bytes) -> Iterator[tuple[int | None, list[str], bytes]]:
    """Yield (dataset_number, record_lines, payload) for each dataset.

    `payload` is None for an ordinary ASCII dataset and a `Binary` for
    a binary one (dataset 58b), whose eleven header records stay ASCII
    and come back in `record_lines` like any other.

    Bytes rather than text, and a cursor rather than `splitlines`,
    because a binary payload is not text: it contains newlines that are
    float bytes, and it can contain the byte pattern of a '-1' delimiter
    line. Splitting first and interpreting afterwards would cut a
    dataset in half at a number that happened to look like a delimiter.
    The payload is therefore taken by the byte count the marker line
    declares, and never scanned.
    """
    if isinstance(data, str):
        data = data.encode('utf-8', 'surrogateescape')
    position, number, records, state = 0, None, [], 'between'
    payload = None
    while position < len(data):
        stop = data.find(b'\n', position)
        if stop == -1:
            stop = len(data)
        line = data[position:stop].decode('utf-8', 'replace').rstrip('\r')
        position = stop + 1

        if line.strip() == '-1':
            if state == 'in' and number is not None:
                yield number, records, payload
            number, records, payload = None, [], None
            state = 'delim' if state == 'between' else 'between'
            continue
        if state == 'delim':
            state = 'in'
            binary = _BINARY_MARKER.match(line)
            if binary:
                number = int(binary.group(1))
                order, fp, ascii_lines, count = (int(binary.group(i))
                                                 for i in (2, 3, 4, 5))
                for _ in range(ascii_lines):
                    stop = data.find(b'\n', position)
                    if stop == -1:
                        stop = len(data)
                    records.append(
                        data[position:stop].decode('utf-8', 'replace')
                        .rstrip('\r'))
                    position = stop + 1
                blob = data[position:position + count]
                position += count
                # a writer may or may not end the blob with a newline;
                # skipping one keeps an empty record out of the list
                if data[position:position + 1] == b'\n':
                    position += 1
                payload = Binary(order, fp, blob)
                continue
            try:
                number = int(line.strip())
            except ValueError:
                number = None
        elif state == 'in':
            records.append(line)
    if state == 'in' and number is not None:
        yield number, records, payload


def sniff(path: str | os.PathLike) -> bool:
    if not str(path).lower().endswith(EXTENSIONS):
        return False
    head = text_head(path, 2048)
    return head is not None and head.lstrip().startswith('-1')


def _parse_2411(records, nodes):
    for i in range(0, len(records) - 1, 2):
        label, def_cs, disp_cs, color = _ints(records[i])[:4]
        xyz = _floats(records[i + 1])[:3]
        nodes.append((label, def_cs, disp_cs, color, xyz))


def _parse_15(records, nodes):
    for line in records:
        parts = line.split()
        if len(parts) < 7:
            continue
        label, def_cs, disp_cs, color = (int(p) for p in parts[:4])
        xyz = [float(p.replace('D', 'E')) for p in parts[4:7]]
        nodes.append((label, def_cs, disp_cs, color, xyz))


def _parse_2420(records, systems):
    """Coordinate systems: label, type, name and a 4x3 transform each.

    Rows 0-2 of the transform are the system's axes in global coordinates
    and row 3 is its origin, which is exactly how visualdynamics holds one.

    NOTE: the published spec for this dataset could not be reached when
    this was written (the SDRL pages 404, including for datasets already
    mirrored under docs/uff_spec). The layout below — a part header, then
    per system an integer record, a name record and four matrix records —
    round trips through visualdynamics, but has not been checked against the
    standard or against another tool's file.
    """
    if len(records) < 2:
        return
    body = records[2:]                     # skip the part uid and part name
    i = 0
    while i + 5 < len(body) + 1 and i + 1 < len(body):
        header = _ints(body[i])
        if len(header) < 2:
            break
        label, cs_type = header[0], header[1]
        color = header[2] if len(header) > 2 else 1
        name = body[i + 1].strip()
        rows = []
        j = i + 2
        while len(rows) < 4 and j < len(body):
            values = _floats(body[j])
            if len(values) >= 3:
                rows.append(values[:3])
            j += 1
        if len(rows) < 4:
            break
        systems.append((label, cs_type, color, name, np.array(rows)))
        i = j


def _parse_82(records, tracelines):
    tl_id, num_nodes, color = (_ints(records[0]) + [1, 1])[:3]
    desc = records[1].strip() if len(records) > 1 else ''
    entries = []
    for line in records[2:]:
        entries.extend(_ints(line))
        if len(entries) >= num_nodes:
            break
    # 0 = pen up: split into separate polylines
    run = []
    for entry in entries[:num_nodes]:
        if entry == 0:
            if len(run) >= 2:
                tracelines.append((tl_id, color, desc, run))
            run = []
        else:
            run.append(entry)
    if len(run) >= 2:
        tracelines.append((tl_id, color, desc, run))


def _parse_2412(records, elements, path):
    i = 0
    while i < len(records):
        fields = _ints(records[i])
        if len(fields) < 6:
            i += 1
            continue
        label, descriptor, _phys, _mat, color, num_nodes = fields[:6]
        i += 1
        if descriptor in _BEAM_DESCRIPTORS:
            i += 1  # beam orientation / cross-section record
        conn = []
        while len(conn) < num_nodes and i < len(records):
            conn.extend(_ints(records[i]))
            i += 1
        if descriptor not in ELEMENT_TYPES:
            raise ValueError(
                f"Unsupported FE descriptor {descriptor} "
                f"(element {label}) in {path}")
        elements.append((label, descriptor, color, conn[:num_nodes]))


def _int_slice(line, start, stop, default=0):
    token = line[start:stop].strip()
    try:
        return int(token)
    except ValueError:
        return default


# dataset 55 writes whole nodal vectors: three translations, or those plus
# the three rotations
_55_COMPONENTS = ('X+', 'Y+', 'Z+', 'RX+', 'RY+', 'RZ+')


def _parse_55(records, modes, path):
    """Parse one dataset 55 — one mode's shape over the nodes — into `modes`.

    Two analysis types are mode shapes, and both are read. Type 2, a
    normal mode, carries frequency, modal mass and damping ratio. Type
    3, a complex eigenvalue of the first-order problem, carries the
    eigenvalue itself (rad/s), modal A and modal B, each as a real and
    imaginary pair; it is how a curve fitter exports complex modes. The
    eigenvalue l becomes frequency |l| / 2 pi and damping -Re(l) / |l|,
    which is the pole `ShapeSet.synthesize_frf` rebuilds; modal A goes
    into `modal_mass`, which is what that synthesis reads it as for
    complex shapes, and modal B into `modal_damping`. The other analysis
    types (static, transient, buckling) are not mode shapes and would
    need somewhere else to go.
    """
    definition = _ints(records[5])
    if len(definition) < 6:
        raise ValueError(f'{path}: dataset 55 has a short definition record')
    analysis_type, characteristic = definition[1], definition[2]
    data_type, values_per_node = definition[4], definition[5]
    if analysis_type not in (2, 3):
        return                      # not a mode shape; nothing to build here
    # The characteristic says what a node's vector *is* — 2 a 3-DOF
    # translation, 3 translation and rotation — and so how many values
    # it has; the count field only repeats it. Where the two disagree
    # the count is the writer's slip: sdynpy writes 1 for a 3-DOF
    # vector, and trusting it read one value per node and dropped the
    # rest without a word (2026-09-23).
    values_per_node = {2: 3, 3: 6}.get(characteristic, values_per_node)
    if values_per_node > len(_55_COMPONENTS):
        raise ValueError(
            f'{path}: dataset 55 has {values_per_node} values per node; '
            f'visualdynamics understands up to {len(_55_COMPONENTS)}')

    integers = _ints(records[6])
    reals = _floats(records[7])
    mode_number = integers[3] if len(integers) > 3 else len(modes) + 1
    modal_b = 0.0
    if analysis_type == 3:
        pairs = [complex(reals[k], reals[k + 1]) if len(reals) > k + 1
                 else 0j for k in (0, 2, 4)]
        eigenvalue, modal_mass, modal_b = pairs
        frequency = abs(eigenvalue) / (2.0 * np.pi)
        damping = (-eigenvalue.real / abs(eigenvalue)
                   if eigenvalue != 0 else 0.0)
        # a writer with no scaling to give leaves modal A zero; unity is
        # what the synthesis assumes of a set that carries none
        if modal_mass == 0:
            modal_mass = 1.0
    else:
        frequency = reals[0] if reals else 0.0
        modal_mass = reals[1] if len(reals) > 1 else 1.0
        damping = reals[2] if len(reals) > 2 else 0.0

    complex_values = data_type == 5
    per_node = values_per_node * (2 if complex_values else 1)
    node_values = {}
    i = 8
    while i < len(records):
        node = _ints(records[i])
        if not node:
            break
        values = []
        i += 1
        while len(values) < per_node and i < len(records):
            values.extend(_floats(records[i]))
            i += 1
        values = values[:per_node]
        if complex_values:
            values = [values[k] + 1j * values[k + 1]
                      for k in range(0, len(values), 2)]
        node_values[node[0]] = values
    modes.append({'mode': mode_number, 'analysis': analysis_type,
                  'frequency': frequency, 'modal_mass': modal_mass,
                  'modal_b': modal_b, 'damping': damping,
                  'values_per_node': values_per_node, 'nodes': node_values,
                  'comment': records[0].strip() if records else ''})


def _build_shapes(modes, path):
    """One ShapeSet from the dataset 55 blocks, ordered by mode number."""
    from ..core.shapes import ShapeSet

    modes = sorted(modes, key=lambda m: m['mode'])
    # the two carry different scalings — a normal mode's modal mass and
    # a complex mode's modal A — and one set holds one of them
    if len({m['analysis'] for m in modes}) > 1:
        raise ValueError(
            f'{path}: dataset 55 mixes normal modes (analysis type 2) with '
            'complex eigenvalues (type 3); a visualdynamics ShapeSet is '
            'one or the other')
    first = modes[0]
    nodes = list(first['nodes'])
    width = first['values_per_node']
    for mode in modes[1:]:
        if list(mode['nodes']) != nodes or mode['values_per_node'] != width:
            raise ValueError(
                f'{path}: modes cover differing nodes; a visualdynamics ShapeSet '
                'shares one set of coordinates')

    coordinate = [f'{node}{_55_COMPONENTS[k]}'
                  for node in nodes for k in range(width)]
    shape_matrix = np.array(
        [[value for node in nodes for value in mode['nodes'][node]]
         for mode in modes])
    return ShapeSet(
        frequency=[m['frequency'] for m in modes],
        damping=[m['damping'] for m in modes],
        coordinate=coordinate,
        shape_matrix=shape_matrix,
        modal_mass=[m['modal_mass'] for m in modes],
        comment=[m['comment'] for m in modes],
        modal_damping=([m['modal_b'] for m in modes]
                       if any(m['modal_b'] != 0 for m in modes) else None),
    )


#: Ordinate data type (record 7, field 1) -> bytes per float. 2 and 5
#: are single precision, 4 and 6 double; 5 and 6 are the complex pair,
#: which is two floats rather than a wider one.
_58_PRECISION = {2: 4, 4: 8, 5: 4, 6: 8}


def _decode_58b(payload, ordinate_type):
    """The floats a binary dataset 58 carries, in file order."""
    if payload.fp_format != IEEE_754:
        raise ValueError(
            f'this universal file stores its binary data as '
            f'{_FP_FORMATS.get(payload.fp_format, payload.fp_format)} '
            f'floating point, and only IEEE 754 is decoded. Reading it '
            f'as IEEE would not fail — it would return wrong numbers.')
    width = _58_PRECISION.get(ordinate_type)
    if width is None:
        raise ValueError(
            f'ordinate data type {ordinate_type} has no defined float '
            f'width, so the binary data cannot be read')
    end = '<' if payload.order == LITTLE_ENDIAN else '>'
    dtype = np.dtype(f'{end}f{width}')
    usable = len(payload.data) - len(payload.data) % width
    return np.frombuffer(payload.data[:usable], dtype=dtype).astype(float)


def _parse_58(records, functions, payload=None):
    """Parse one dataset 58 into an entry dict appended to `functions`.

    `payload` is the `Binary` of a 58b, whose values are the same
    numbers in the same order as the ASCII records 12 onward would
    hold — so only how they are *read* differs, and everything after
    that is the one implementation both forms go through.
    """
    from ..core.data import dof_string

    dof_line = records[5]
    function_type = _int_slice(dof_line, 0, 5)
    response = dof_string(_int_slice(dof_line, 41, 51),
                          _int_slice(dof_line, 51, 55))
    reference_node = _int_slice(dof_line, 66, 76)
    # node 0 is how the format says there is no reference; reading it as a
    # DOF named '0' would invent one
    reference = (dof_string(reference_node, _int_slice(dof_line, 76, 80))
                 if reference_node else '')

    form = _ints(records[6][:30]) + _floats(records[6][30:])
    ordinate_type, num_values, even = form[0], form[1], form[2]
    abscissa_min, abscissa_inc = form[3], form[4]
    is_complex = ordinate_type in (5, 6)

    def axis(record: str) -> tuple[str, int, int]:
        data_type = _int_slice(record, 0, 10)
        dim, lexp, fexp = _58_DATA_TYPES.get(data_type, ('dimensionless', 0, 0))
        if dim is None:  # general: exponents stored in the record
            dim = 'dimensionless'
            lexp = _int_slice(record, 10, 15)
            fexp = _int_slice(record, 15, 20)
        return dim, lexp, fexp

    num_dim, num_l, num_f = axis(records[8])
    den_type = _int_slice(records[9], 0, 10)
    if den_type:
        den_dim, den_l, den_f = axis(records[9])
        ordinate_dim = f'{num_dim}/{den_dim}'
        lexp, fexp = num_l - den_l, num_f - den_f
    else:
        ordinate_dim = num_dim
        lexp, fexp = num_l, num_f

    if payload is None:
        values = []
        for line in records[11:]:
            values.extend(_floats(line))
        values = np.array(values)
    else:
        values = _decode_58b(payload, ordinate_type)

    if even:
        ordinate = values
        abscissa = abscissa_min + abscissa_inc * np.arange(num_values)
    else:
        step = 3 if is_complex else 2
        values = values[:num_values * step].reshape(num_values, step)
        abscissa = values[:, 0]
        ordinate = values[:, 1:]
    if is_complex:
        flat = ordinate.reshape(-1)
        ordinate = flat[0::2] + 1j * flat[1::2]
    elif ordinate.ndim > 1:
        # unevenly spaced real data arrives as (n, 1) pairs; a record is one
        # series, not a series of one-element rows
        ordinate = ordinate[:, 0]
    ordinate = ordinate[:num_values]

    functions.append({
        'function_type': function_type,
        'response': response,
        'reference': reference,
        'abscissa': abscissa,
        'ordinate': ordinate,
        'ordinate_dim': ordinate_dim,
        'lexp': lexp,
        'fexp': fexp,
        'comment': records[0].strip() if records else '',
    })


def _build_data_objects(functions, factors, ordinate_unit, path):
    """Group parsed 58 entries by function type into visualdynamics data objects.

    With a 164 present the ordinate scale to SI is known from the unit
    exponents, so the data is unit-aware. Without one, unit-bearing records
    import unit-less unless `ordinate_unit` was declared.
    """
    out = {}
    for code, name in _58_NAMES.items():
        typed = [f for f in functions if f['function_type'] == code]
        if not typed:
            continue
        # Records of one type need not share an axis: a writer holds
        # them happily in one file (an awkward real-world sample
        # pinned it), but a visualdynamics data array does
        # share one — so each distinct abscissa becomes its own
        # object. This used to refuse the whole file, which turned
        # one odd record into no import at all.
        grouped = []
        for entry in typed:
            for group in grouped:
                first = group[0]['abscissa']
                if (len(entry['abscissa']) == len(first)
                        and np.allclose(entry['abscissa'], first)):
                    group.append(entry)
                    break
            else:
                grouped.append([entry])
        for which, entries in enumerate(grouped):
            _build_one(out, name if which == 0 else f'{name} ({which + 1})',
                       code, entries, factors, ordinate_unit)
    return out


def _build_one(out, name, code, entries, factors, ordinate_unit):
    from ..core.data import class_for_function_type
    from ..units import UNKNOWN, si_factor

    abscissa = entries[0]['abscissa']
    scales, dims, hints = [], [], []
    for entry in entries:
        if factors is not None:
            length_factor, force_factor = factors
            scales.append(1.0 / (length_factor ** entry['lexp']
                                 * force_factor ** entry['fexp']))
            dims.append(entry['ordinate_dim'])
            hints.append(None)
        elif entry['lexp'] == 0 and entry['fexp'] == 0:
            scales.append(1.0)
            dims.append(entry['ordinate_dim'])
            hints.append(None)
        elif ordinate_unit is not None:
            scales.append(si_factor(ordinate_unit))
            dims.append(entry['ordinate_dim'])
            hints.append(None)
        else:
            # The 58 names the quantity but the missing 164 sizes it, so
            # the values stay raw. Keep the name: it is still true, and
            # it is most of what someone needs to declare the unit later.
            scales.append(1.0)
            dims.append(UNKNOWN)
            hints.append(entry['ordinate_dim'])
    cls = class_for_function_type(code)
    out[name] = cls(
        abscissa=abscissa,
        ordinate=np.array([e['ordinate'] * s
                           for e, s in zip(entries, scales)]),
        response_dof=[e['response'] for e in entries],
        reference_dof=([e['reference'] for e in entries]
                       if cls.needs_reference
                       or any(e['reference'].strip() for e in entries)
                       else None),
        ordinate_dim=dims,
        dimension_hint=hints,
        comment=[e['comment'] for e in entries],
    )


def load(path: str | os.PathLike, length_unit: str | None = None,
         ordinate_unit: str | None = None) -> Any:
    from ..units import si_factor

    # bytes, because a 58b dataset holds raw floats that are not text
    # and must not be decoded, replaced or newline-translated on the way
    # in (Brandon, 2026-08-25)
    with open(path, 'rb') as f:
        text = f.read()

    factors = None
    nodes, tracelines, elements = [], [], []
    functions, mode_shapes, systems = [], [], []
    for number, records, payload in iter_datasets(text):
        if number == 164:
            fields = _floats(' '.join(records[1:]))
            factors = (fields[0], fields[1])  # length, force
        elif number == 2411:
            _parse_2411(records, nodes)
        elif number == 15:
            _parse_15(records, nodes)
        elif number == 82:
            _parse_82(records, tracelines)
        elif number == 2412:
            _parse_2412(records, elements, path)
        elif number == 58:
            _parse_58(records, functions, payload)
        elif number == 55:
            _parse_55(records, mode_shapes, path)
        elif number == 2420:
            _parse_2420(records, systems)

    out = _build_data_objects(functions, factors, ordinate_unit, path)
    if mode_shapes:
        out['shapes'] = _build_shapes(mode_shapes, path)

    if nodes:
        if factors is not None:
            scale, geometry_unit = 1.0 / factors[0], 'm'
        elif length_unit is not None:
            scale, geometry_unit = si_factor(length_unit, 'length'), length_unit
        else:
            scale, geometry_unit = 1.0, None  # unit-less: raw coordinates
        out['geometry'] = _build_geometry(nodes, tracelines, elements, scale,
                                          geometry_unit, systems)
    elif not out:
        raise ValueError(f"No supported datasets found in {path}")

    return next(iter(out.values())) if len(out) == 1 else out


def _origin_scale(scale):
    """Scale that leaves the axes alone and converts only the origin row."""
    factors = np.ones((4, 3))
    factors[3] = scale
    return factors


def _build_geometry(nodes, tracelines, elements, scale, length_unit,
                    systems=()):
    from ..core.geometry import Geometry

    node_ids = [n[0] for n in nodes]
    # a 2411 record states a node's position in the frame it is placed
    # in — the record's own second column — and the frames are in the
    # 2420 beside it. Read as written they land wherever those numbers
    # happen to point in global (Brandon, 2026-09-20). Resolved before
    # the unit scale, because an angle is not a length.
    frames = ([s[0] for s in systems], [s[1] for s in systems],
              np.array([s[4] for s in systems])) if systems else None
    written = np.array([n[4] for n in nodes], dtype=float)
    placement = [n[1] for n in nodes]
    return Geometry(
        length_unit=length_unit,
        node_id=node_ids,
        node_xyz=(written if frames is None
                  else to_global(written, placement, *frames)) * scale,
        node_def_cs=placement,
        node_disp_cs=[n[2] for n in nodes],
        node_color=[n[3] for n in nodes],
        cs_id=[s[0] for s in systems] or None,
        cs_type=[s[1] for s in systems] or None,
        cs_name=[s[3] for s in systems] or None,
        cs_matrix=(np.array([s[4] for s in systems]) * _origin_scale(scale)
                   if systems else None),
        traceline_id=[t[0] for t in tracelines],
        traceline_color=[t[1] for t in tracelines],
        traceline_desc=[t[2] for t in tracelines],
        traceline_conn=[np.array(t[3]) for t in tracelines],
        elem_id=[e[0] for e in elements],
        elem_type=[e[1] for e in elements],
        elem_color=[e[2] for e in elements],
        elem_conn=[np.array(e[3]) for e in elements],
    )


# ---- writing --------------------------------------------------------------

def handles(obj: Any) -> bool:
    from ..core.data import DataArray
    from ..core.geometry import Geometry
    from ..core.shapes import ShapeSet

    return isinstance(obj, (Geometry, DataArray, ShapeSet))


def handles_binary(obj: Any) -> bool:
    """Only function data, because only dataset 58 has a binary form.

    A geometry or a mode shape offered as "binary" would write exactly
    the ASCII file the other exporter writes, under a name promising
    something else — 2411, 82, 2412 and 55 have no `b` variant, and
    inventing one would produce a file nothing else reads. Caught by
    `test_shapes_can_be_written_to_every_format_that_holds_them`, which
    already knew what a ShapeSet may be written as (2026-08-25).
    """
    from ..core.data import DataArray

    return isinstance(obj, DataArray)


# visualdynamics dimension -> the dataset 58 specific data type that means it
_58_CODES = {'length': 8, 'force': 9, 'velocity': 11, 'acceleration': 12,
             'pressure': 15, 'mass': 16, 'time': 17, 'frequency': 18,
             'dimensionless': 3}

DELIMITER = '    -1\n'


def _block(number, body):
    """One dataset, delimited the way a UNV file delimits them."""
    return f'{DELIMITER}{number:6d}\n{body}{DELIMITER}'


def _binary_block(number, header, blob):
    """One binary dataset: the marker line, the ASCII header records,
    then the bytes.

    Written little endian and IEEE 754 — the only pair every machine
    this runs on speaks natively, and both are declared on the marker
    line rather than assumed, which is what lets the reader take a file
    written the other way round.

    The marker's format is (I6,1A1,I6,I6,I12,I12,I6,I6,I12,I12); the
    last four fields are reserved and written as zeros
    (docs/uff_spec/58b.asc).
    """
    lines = header.count('\n')
    marker = (f'{number:6d}b{LITTLE_ENDIAN:6d}{IEEE_754:6d}'
              f'{lines:12d}{len(blob):12d}'
              f'{0:6d}{0:6d}{0:12d}{0:12d}\n')
    return (DELIMITER.encode() + marker.encode() + header.encode()
            + blob + b'\n' + DELIMITER.encode())


def _geometry_datasets(geometry, unit_system=None):
    """Nodes as 2411, tracelines as 82, elements as 2412."""
    from .exporters import geometry_values

    points, matrices = geometry_values(geometry, unit_system)
    out = [_coordinate_system_dataset(geometry, matrices)]

    # written back into each node's own placement frame, or the file
    # would state global coordinates under a local frame and the next
    # reader would resolve them a second time
    written = to_local(points, geometry.node_def_cs, geometry.cs_id,
                       geometry.cs_type, matrices)
    nodes = []
    for i, node in enumerate(geometry.node_id):
        nodes.append(f'{int(node):10d}{int(geometry.node_def_cs[i]):10d}'
                     f'{int(geometry.node_disp_cs[i]):10d}'
                     f'{int(geometry.node_color[i]):10d}\n')
        nodes.append(''.join(f'{v:25.16E}' for v in written[i]) + '\n')
    out.append(_block(2411, ''.join(nodes)))

    for i, conn in enumerate(geometry.traceline_conn):
        ids = [int(n) for n in conn]
        body = (f'{int(geometry.traceline_id[i]):10d}{len(ids):10d}'
                f'{int(geometry.traceline_color[i]):10d}\n'
                f'{str(geometry.traceline_desc[i])[:80]}\n')
        body += ''.join(
            ''.join(f'{n:10d}' for n in ids[start:start + 8]) + '\n'
            for start in range(0, len(ids), 8))
        out.append(_block(82, body))

    if len(geometry.elem_conn):
        elements = []
        for i, conn in enumerate(geometry.elem_conn):
            ids = [int(n) for n in conn]
            descriptor = int(geometry.elem_type[i])
            if descriptor >= 200:
                from ..core.geometry import ELEMENT_TYPES
                raise ValueError(
                    f'universal files cannot carry a '
                    f'{ELEMENT_TYPES[descriptor][0]} (element '
                    f'{int(geometry.elem_id[i])}): UFF 2412 has no '
                    'pyramid descriptor. Export to exodus instead.')
            elements.append(
                f'{int(geometry.elem_id[i]):10d}{descriptor:10d}'
                f'{1:10d}{1:10d}{int(geometry.elem_color[i]):10d}'
                f'{len(ids):10d}\n')
            if descriptor in _BEAM_DESCRIPTORS:
                elements.append(f'{0:10d}{1:10d}{1:10d}\n')
            elements.extend(
                ''.join(f'{n:10d}' for n in ids[start:start + 8]) + '\n'
                for start in range(0, len(ids), 8))
        out.append(_block(2412, ''.join(elements)))
    return out


def _58_axis_record(dimension, label='', units=''):
    code = _58_CODES.get(dimension, 0)
    return (f'{code:10d}{0:5d}{0:5d}{0:5d}  '
            f'{label[:20]:<20}{units[:20]:<20}\n')


def _58_dataset(data, record, unit_system=None, binary=False):
    """One function, written as dataset 58 the way this module reads one."""
    from ..core.data import parse_dof

    response_node, response_direction = parse_dof(data.response_dof[record])
    reference_node, reference_direction = (
        parse_dof(data.reference_dof[record])
        if data.reference_dof is not None else (0, ''))
    from .exporters import data_values

    abscissa, ordinates = data_values(data, unit_system)
    ordinate = np.atleast_1d(ordinates[record])
    complex_data = np.iscomplexobj(ordinate)

    lines = [f'{data.record_label(record)[:80]}\n', 'NONE\n', 'NONE\n',
             'NONE\n', 'NONE\n']
    lines.append(
        f'{data.function_type:5d}{record + 1:10d}{0:5d}{0:10d} '
        f'{"NONE":<10}{response_node or 0:10d}'
        f'{direction_code(response_direction) if response_direction else 0:4d} '
        f'{"NONE":<10}{reference_node or 0:10d}'
        f'{direction_code(reference_direction) if reference_direction else 0:4d}\n')
    # written unevenly spaced, with every abscissa value present: an even
    # record would have to assume a constant step this abscissa may not have
    lines.append(f'{6 if complex_data else 4:10d}{len(ordinate):10d}{0:10d}'
                 f'{0.0:13.5E}{0.0:13.5E}{0.0:13.5E}\n')
    lines.append(_58_axis_record(data.abscissa_dim))

    # dataset 58 names a dimension from a fixed vocabulary. A PSD's
    # acceleration**2/frequency is not in it, so rather than write half of
    # it — which reads back as a dimension the units engine rejects — the
    # record says nothing and the data returns unit-less.
    # A record whose unit is undefined may still know what it is, and this
    # is one of the few things a 58 can say without a 164 to size it: name
    # the quantity, write no scale. Reading it back recovers the hint.
    dimension = data.known_dim(record) or 'unknown'
    numerator, _, denominator = dimension.partition('/')
    nameable = numerator in _58_CODES and (not denominator
                                           or denominator in _58_CODES)
    blank = f'{0:10d}{0:5d}{0:5d}{0:5d}  {"":<20}{"":<20}\n'
    lines.append(_58_axis_record(numerator) if nameable else blank)
    lines.append(_58_axis_record(denominator)
                 if nameable and denominator else blank)
    lines.append(f'{0:10d}{0:5d}{0:5d}{0:5d}  {"":<20}{"":<20}\n')

    values = []
    for x, y in zip(abscissa, ordinate):
        values.append(x)
        values.extend((y.real, y.imag) if complex_data else (float(y),))
    if binary:
        # float64 because the header two records above declares ordinate
        # type 4 or 6 — the double-precision pair. The two must agree,
        # and the header is the one that says so.
        header = ''.join(lines)
        blob = np.asarray(values, dtype='<f8').tobytes()
        return _binary_block(58, header, blob)
    per_line = 6
    for start in range(0, len(values), per_line):
        lines.append(''.join(f'{v:13.5E}'
                             for v in values[start:start + per_line]) + '\n')
    return _block(58, ''.join(lines))


# visualdynamics unit system -> (dataset 164 code, the description that goes with it)
_164_SYSTEMS = {
    'm-kg-N-s': (1, 'SI - meters (newton)'),
    'mm-kg-N-s': (10, 'mm (newton)'),
    'in-slinch-lbf-s': (7, 'Inch (pound f)'),
    'ft-slug-lbf-s': (2, 'BG - Foot (pound f)'),
}


def _units_dataset(unit_system=None):
    """Dataset 164: which units the file is in, and how to leave them.

    The spec's factors convert *out* of the file's units by division, so
    each is the number of file units in one SI unit. Temperature takes an
    offset as well, which is why a file in degrees Fahrenheit is expressible
    at all.
    """
    from ..units import SI, si_factor, si_transform

    unit_system = (unit_system or SI).coherent
    code, description = _164_SYSTEMS.get(unit_system.name,
                                         (9, unit_system.name[:20]))
    length = 1.0 / si_factor(unit_system.unit('length'), 'length')
    force = 1.0 / si_factor(unit_system.unit('force'), 'force')
    scale, offset = si_transform(unit_system.unit('temperature'),
                                 'temperature')
    return _block(164,
                  f'{code:10d}{description:<20}{2:10d}\n'
                  + ''.join(f'{v:25.17E}'.replace('E', 'D')
                            for v in (length, force, 1.0 / scale)) + '\n'
                  + f'{offset / scale:25.17E}'.replace('E', 'D') + '\n')


def _coordinate_system_dataset(geometry, matrices=None):
    """Dataset 2420, so the frames the nodes refer to are actually defined.

    Without it a file's local displacement directions mean nothing to
    whoever reads it, and mode shapes written in those directions would be
    taken for global ones.
    """
    lines = [f'{1:10d}\n', 'visualdynamics\n']
    for i, cs_id in enumerate(geometry.cs_id):
        lines.append(f'{int(cs_id):10d}{int(geometry.cs_type[i]):10d}'
                     f'{1:10d}\n')
        # written even when blank: an empty name is a name, and the
        # reader takes this record's line whatever is on it
        lines.append(f'{str(geometry.cs_name[i])[:40]}\n')
        rows = (geometry.cs_matrix if matrices is None else matrices)[i]
        lines.extend(''.join(f'{v:25.16E}' for v in row) + '\n'
                     for row in rows)
    return _block(2420, ''.join(lines))


def _shape_datasets(shapes, unit_system=None):
    """One dataset 55 per mode, whole nodal vectors at a time.

    The dataset stores a vector per node, so the DOFs are grouped by node
    and any the shape does not carry are written as zero — which means a
    shape covering only some directions comes back covering all of them.
    """
    from ..core.data import parse_dof
    from .exporters import shape_values

    coefficients = shape_values(shapes, unit_system)
    order, components = [], {}
    for i, dof in enumerate(shapes.coordinate):
        node, direction = parse_dof(dof)
        if node is None or direction.upper() not in _55_COMPONENTS:
            raise ValueError(f'{dof} is not a direction dataset 55 can hold')
        if node not in components:
            order.append(node)
            components[node] = {}
        components[node][_55_COMPONENTS.index(direction.upper())] = i

    width = 6 if any(k > 2 for node in order
                     for k in components[node]) else 3
    complex_values = bool(np.iscomplexobj(shapes.shape_matrix))

    blocks = []
    for mode in range(shapes.num_shapes):
        values = coefficients[mode]
        lines = [f'{shapes.mode_label(mode)[:80]}\n', 'NONE\n', 'NONE\n',
                 'NONE\n', 'NONE\n']
        # a complex set goes out as analysis type 3, the complex
        # eigenvalue, which is where every reader looks for one and the
        # only record with room for a complex modal A; a real set as
        # type 2, the normal mode
        lines.append(f'{1:10d}{3 if complex_values else 2:10d}'
                     f'{2 if width == 3 else 3:10d}'
                     f'{8:10d}{5 if complex_values else 2:10d}'
                     f'{width:10d}\n')
        if complex_values:
            eigenvalue, modal_a, modal_b = _complex_mode_record(shapes, mode)
            lines.append(f'{2:10d}{6:10d}{1:10d}{mode + 1:10d}\n')
            lines.append(''.join(f'{v:13.5E}' for pair in
                                 (eigenvalue, modal_a, modal_b)
                                 for v in (pair.real, pair.imag)) + '\n')
        else:
            lines.append(f'{2:10d}{4:10d}{1:10d}{mode + 1:10d}\n')
            lines.append(f'{float(shapes.frequency[mode]):13.5E}'
                         f'{float(np.real(shapes.modal_mass[mode])):13.5E}'
                         f'{float(shapes.damping[mode]):13.5E}'
                         f'{0.0:13.5E}\n')
        for node in order:
            lines.append(f'{node:10d}\n')
            row = []
            for k in range(width):
                index = components[node].get(k)
                value = 0.0 if index is None else values[index]
                row.extend((value.real, value.imag) if complex_values
                           else (float(np.real(value)),))
            lines.append(''.join(f'{v:13.5E}' for v in row) + '\n')
        blocks.append(_block(55, ''.join(lines)))
    return blocks


def _complex_mode_record(shapes, mode):
    """(eigenvalue, modal A, modal B) for one complex mode's record.

    The eigenvalue is the pole `synthesize_frf` uses, -z w + i w
    sqrt(1 - z^2) in rad/s. Modal B is the set's own where it carried
    one; otherwise it is -l A, which is what B is for the first-order
    problem (l = -B / A), rather than a zero that would claim the pole
    sits at the origin.
    """
    omega = 2.0 * np.pi * float(shapes.frequency[mode])
    zeta = float(shapes.damping[mode])
    eigenvalue = complex(-zeta * omega,
                         omega * np.sqrt(complex(1.0 - zeta ** 2)).real)
    modal_a = complex(shapes.modal_mass[mode])
    carried = shapes.modal_damping
    modal_b = (complex(carried[mode]) if carried is not None
               else -eigenvalue * modal_a)
    return eigenvalue, modal_a, modal_b


def save(obj: Any, path: str | os.PathLike,
         unit_system: UnitSystem | None = None, binary: bool = False) -> None:
    """Write a geometry or a data array as a universal file.

    Geometry goes out as 2420 coordinate systems, 2411 nodes, 82 tracelines
    and 2412 elements; data as one dataset 58 per record; mode shapes as
    one dataset 55 per mode.

    Shape values stay in each node's own displacement system, which is what
    dataset 55 means by them — the coordinate systems written alongside are
    what make that readable.

    A dataset 164 declaring SI is written when the object's units have been
    defined, and left out when they have not. Stored values are SI either
    way, but only in the first case is that a fact about the data rather
    than an assumption — and a file that says so reads back with its units,
    where one that does not reads back unit-less.

    `binary` writes the functions as dataset **58b** instead of 58: the
    same eleven ASCII header records, and then the values as raw IEEE
    754 doubles rather than as `13.5E` text. It is roughly a third of
    the size and it is exact — a written-and-read float comes back the
    float that went in, where five decimal places do not. Only the
    functions change form; geometry, mode shapes and the units dataset
    have no binary variant and go out as they always do.
    """
    from ..core.geometry import Geometry

    blocks = []
    if getattr(obj, 'units_defined', False):
        blocks.append(_units_dataset(unit_system))
    from ..core.shapes import ShapeSet

    if isinstance(obj, Geometry):
        blocks += _geometry_datasets(obj, unit_system)
    elif isinstance(obj, ShapeSet):
        blocks += _shape_datasets(obj, unit_system)
    else:
        blocks += [_58_dataset(obj, i, unit_system, binary)
                   for i in range(obj.num_records)]
    # bytes throughout: a 58b block is bytes, the rest are text, and the
    # file is one stream of both
    with open(path, 'wb') as handle:
        handle.writelines(block if isinstance(block, bytes)
                          else block.encode() for block in blocks)


def save_binary(obj: Any, path: str | os.PathLike,
                unit_system: UnitSystem | None = None) -> None:
    """`save` in the binary form — the registry's entry for 58b.

    A separate exporter rather than a checkbox on the other one,
    because the Export menu is a list of formats and this is one: a
    file another tool will or will not read. Choosing by suffix still
    gets the ASCII form, which is the one to hand a stranger.
    """
    save(obj, path, unit_system, binary=True)
