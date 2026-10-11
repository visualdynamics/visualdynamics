"""I-DEAS Associated Data Files — .afu, .ati and .ash, read natively.

ADF is the binary sibling of the Universal File's dataset 58: the same
function types, the same data-type codes, the same interleaved-uneven
convention, in 512-byte blocks of little-endian float32. Shape files
add a file-wide node table and per-mode records. The formats originate
with SDRC I-DEAS and live on in Siemens NX. No public specification
exists, so this reader was **derived from paired sample files and
from published documentation — never from anyone's source code**.

Two facts a reader of the data should know, both established against
those samples: **ADF files store SI regardless of the writing
session's unit system** (round-tripped through all nine unit systems
the format admits, with and without the G's flag), so imports arrive
with units defined; and **the container stores float32 only** — data
written as double precision was narrowed by the writer, and no reader
can get it back.

The share with `unv.py` is deliberate: parsed function records become
the same entry dicts UNV's dataset-58 parser produces and are grouped
by `_build_data_objects` with SI factors, so an `.afu` and the
equivalent `.unv` import through one code path and cannot disagree.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .unv import _58_DATA_TYPES, _build_data_objects

#: the container's block size; every ADF is a whole number of these
BLOCK = 512

#: the record id of a shape file's node table
NODE_TABLE = 10000

#: shape header code -> CSType name (`ash_attrs` pinned the codes)
CS_TYPES = {0: 'Displacement', 1: 'Basic', 2: 'Other'}


def sniff(path: str | os.PathLike) -> bool:
    path = str(path)
    if os.path.splitext(path)[1].lower() not in ('.afu', '.ati', '.ash'):
        return False
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            magic = f.read(4)
    except OSError:
        return False
    return size >= 3 * BLOCK and size % BLOCK == 0 \
        and magic == b'\xff\xff\xff\xff'


def load(path: str | os.PathLike) -> Any:
    """Read an ADF; functions and time histories group exactly like a
    universal file's dataset 58s, shapes become a `ShapeSet`."""
    path = str(path)
    with open(path, 'rb') as f:
        raw = f.read()
    if raw[:4] != b'\xff\xff\xff\xff' or len(raw) % BLOCK:
        raise ValueError(f'{path} is not an Associated Data File')
    entries = _directory(raw)
    if not entries:
        raise ValueError(f'{path}: no records in the ADF directory')
    if any(e['record'] == NODE_TABLE for e in entries):
        return _load_shapes(raw, entries, path)
    return _load_functions(raw, entries, path)


# ---- the container -----------------------------------------------------

def _i32(raw, offset, count):
    return np.frombuffer(raw, dtype='<i4', count=count, offset=offset)


def _f32(raw, offset, count):
    return np.frombuffer(raw, dtype='<f4', count=count, offset=offset)


def _packed_direction(value: int) -> str:
    """Two ASCII characters packed little-endian: 0x2B58 -> 'X+'."""
    return bytes([value & 0xFF, (value >> 8) & 0xFF]) \
        .decode('ascii', 'replace').strip('\x00 ')


def _directory(raw):
    """The chained record directory: 15 entries of eight int32 per
    block starting in block 2, the block's final int32 naming the next
    directory block (1-based; 0 ends the chain)."""
    entries, block, seen = [], 2, set()
    while block not in seen:
        seen.add(block)
        base = block * BLOCK
        if base + BLOCK > len(raw):
            break
        for slot in range(15):
            chunk = _i32(raw, base + 32 * slot, 8)
            if chunk[0] <= 0:
                break
            entries.append({
                'record': int(chunk[0]),
                'start': int(chunk[1]) - 1,      # to 0-based blocks
                'blocks': int(chunk[2]),
                'response_node': int(chunk[3]),
                'response_dir': _packed_direction(int(chunk[4])),
                'reference_node': int(chunk[5]),
                'reference_dir': _packed_direction(int(chunk[6])),
            })
        chain = int(_i32(raw, base + BLOCK - 4, 1)[0])
        if chain <= 0:
            break
        block = chain - 1
    return entries


def _extremes(values):
    """(max, min) of a record's values, NaN passed over; (0, 0) when
    nothing is left."""
    kept = values[~np.isnan(values)]
    return (float(kept.max()), float(kept.min())) if kept.size else (0.0, 0.0)


def _char_field(raw, base, offset, length):
    text = raw[base + offset: base + offset + length]
    return text.decode('latin1').rstrip('\x00 ')


# ---- functions (.afu / .ati) -------------------------------------------

def _load_functions(raw, entries, path):
    functions = []
    for e in entries:
        base = e['start'] * BLOCK
        i = _i32(raw, base, 128)
        f = _f32(raw, base, 128)
        storage, spacing, n = int(i[11]), int(i[12]), int(i[13])
        if storage not in (2, 5):
            # the corpus only ever produced these two; a code outside
            # them is a format variant nobody has shown us
            raise ValueError(
                f'{path}: record {e["record"]} stores ordinate type '
                f'{storage}, which this reader has no evidence for')
        is_complex = storage == 5

        def axis(code):
            dim, lexp, fexp = _58_DATA_TYPES.get(code,
                                                 ('dimensionless', 0, 0))
            if dim is None:
                # the 'general' data type, whose exponents dataset 58
                # carries in-record; the corpus only ever wrote zeros
                # there, so until a file shows otherwise they are zeros
                dim, lexp, fexp = 'dimensionless', 0, 0
            return dim, lexp, fexp

        num_dim, num_l, num_f = axis(int(i[19]))
        den_code = int(i[24])
        function_type = int(i[4])
        if den_code:
            den_dim, den_l, den_f = axis(den_code)
            ordinate_dim = f'{num_dim}/{den_dim}'
            lexp, fexp = num_l - den_l, num_f - den_f
        elif function_type == 4:
            # an FRF is a ratio by definition; a record that names no
            # denominator still is one, and the class holds it to that
            ordinate_dim = f'{num_dim}/dimensionless'
            lexp, fexp = num_l, num_f
        else:
            ordinate_dim, lexp, fexp = num_dim, num_l, num_f

        # +500 flags the NX12+ extended character header (afu_nx12
        # against a plain record): an extra block after the char block
        # holding a long function name, with the data one block later
        extended = int(i[125])
        data_at = (e['start'] + (3 if extended else 2)) * BLOCK
        width = (2 if is_complex else 1) + (0 if spacing == 1 else 1)
        flat = _f32(raw, data_at, width * n).astype(float)
        if spacing == 1:
            abscissa = float(f[96]) + float(f[97]) * np.arange(n)
            values = flat
        else:
            grid = flat.reshape(n, width)
            abscissa = grid[:, 0]
            values = grid[:, 1:].ravel()
        ordinate = values[0::2] + 1j * values[1::2] if is_complex else values

        node, direction = int(i[5]), _packed_direction(int(i[6]))
        reference_node = int(i[8])
        if function_type == 0:
            # a 'general' function is x-y data with no stronger claim;
            # the abscissa decides what it is here — time makes it a
            # time history, anything else reads as a spectrum (UFF
            # abscissa data type 17 is time). Complex data can never be
            # a time history: the class stores real, and casting would
            # silently discard the imaginary half.
            function_type = (1 if int(i[14]) == 17 and not is_complex
                             else 12)
        functions.append({
            'function_type': function_type,
            'response': f'{node}{direction}',
            'reference': (f'{reference_node}'
                          f'{_packed_direction(int(i[9]))}'
                          if reference_node else ''),
            'abscissa': abscissa,
            'ordinate': ordinate,
            'ordinate_dim': ordinate_dim,
            'lexp': lexp,
            'fexp': fexp,
            # an NX12 record's long name stands in for a blank IDLine1,
            # the same trade readadf documents for NX-written files
            'comment': (_char_field(raw, base, BLOCK, int(i[39]))
                        or (_char_field(raw, base, 2 * BLOCK, 128).strip()
                            if extended else '')),
            # ADF's SI conversion happens per data type, so a record
            # typed Unknown/General was never converted — its values
            # are whatever the writing session used, and the ordinate
            # units label is the producer's only statement of what
            # they are (afu_strings pinned the offsets)
            # ...and a code the table has never heard of converted
            # nothing either, so its label is kept the same way
            'unit_label': (_char_field(raw, base, 972, int(i[47]))
                           if (int(i[19]) in (0, 1)
                               or int(i[19]) not in _58_DATA_TYPES)
                           and not den_code
                           else ''),
        })
    # factors (1, 1): the file is SI by construction, which is exactly
    # what a universal file's dataset 164 written in SI would declare
    out = _build_data_objects(functions, (1.0, 1.0), None, path)
    _apply_unit_labels(out, functions)
    return next(iter(out.values())) if len(out) == 1 else out


def _apply_unit_labels(objects, functions):
    """Untyped records keep their ordinate units label as a hint.

    The data-type code is what converts a record to SI on write; a
    record typed Unknown or General bypassed that, so its values are
    raw and the label is the only statement of what they might be.
    The label used to *define* the units when it parsed, by analogy
    to SEP 005's unit_str — a bad analogy, walked back: SEP 005's
    field is first-class and its values are in those units by
    definition, while an ADF label is free text that decades of
    tools filled with display units, stale units or nothing, and it
    is often simply wrong. A wrong label promoted to a definition
    scales a report by a false 9.81 or 386 with full confidence. So
    the claim narrows the shortlist and names the axis, and a person
    confirms it in the Imported Units pane before anything scales —
    interface rule 11, the same standing an unscaled UNV claim gets.
    The grouping preserves per-type record order, which is what maps
    an entry back to its record.
    """
    from .unv import _58_NAMES

    for code, name in _58_NAMES.items():
        obj = objects.get(name)
        if obj is None:
            continue
        entries = [f for f in functions if f['function_type'] == code]
        for k, entry in enumerate(entries):
            if entry['ordinate_dim'] not in ('dimensionless',
                                            'unknown'):
                continue
            # imported as 'dimensionless, defined' would claim a
            # conversion nobody performed — the values go back to raw
            obj.undefine_units([k])
            label = entry['unit_label']
            if label and obj.dimension_hint[k] is None:
                obj.dimension_hint[k] = label


# ---- shapes (.ash) -----------------------------------------------------

#: DOF directions per node, in storage order
DIRECTIONS_3 = ('X+', 'Y+', 'Z+')
DIRECTIONS_6 = ('X+', 'Y+', 'Z+', 'RX+', 'RY+', 'RZ+')


def _load_shapes(raw, entries, path):
    from ..core.shapes import ShapeSet

    table_entry = next(e for e in entries if e['record'] == NODE_TABLE)
    shape_entries = [e for e in entries if e['record'] != NODE_TABLE]
    if not shape_entries:
        raise ValueError(f'{path}: a shape ADF with no shapes')

    # the node table: triples (node id, second id, insertion ordinal),
    # stored sorted by id. Its length is not in its own header; the
    # largest write-time size any shape record carries is the count.
    counts = [int(_i32(raw, e['start'] * BLOCK + 4, 1)[0])
              for e in shape_entries]
    current = max(counts)
    triples = _i32(raw, (table_entry['start'] + 2) * BLOCK,
                   3 * current).reshape(current, 3)
    ins_to_row = {int(row[2]): index for index, row in enumerate(triples)}

    frequencies, dampings, masses, modal_dampings = [], [], [], []
    comments, rows = [], []
    dof_per_node = None
    for e in shape_entries:
        base = e['start'] * BLOCK
        i = _i32(raw, base, 128)
        f = _f32(raw, base, 128)
        stored, dof = int(i[1]), int(i[4])
        if dof_per_node is None:
            dof_per_node = dof
        elif dof != dof_per_node:
            raise ValueError(f'{path}: shapes mix {dof_per_node}- and '
                             f'{dof}-DOF records')
        # +12 is the shape type code: 1 real, 2 complex (pinned by
        # ash_complex against ash_3dof_real)
        is_complex = int(i[3]) == 2
        packed = _f32(raw, base + 2 * BLOCK,
                      stored * dof * (2 if is_complex else 1)).astype(float)
        if is_complex:
            packed = packed[0::2] + 1j * packed[1::2]
        # positions are insertion ordinals, zero-filled to the
        # write-time table; map through the table's third column
        row = np.zeros(current * dof, dtype=packed.dtype)
        for p in range(stored):
            at = ins_to_row.get(p + 1)
            if at is not None:
                row[dof * at: dof * at + dof] = packed[dof * p: dof * p + dof]
        rows.append(row)
        frequencies.append(float(f[5]))
        dampings.append(float(f[6]) / 100.0)   # stored as a percentage
        # the modal quad (ash_attrs pinned the offsets): mass and
        # damping each real+imaginary, read in as measured — a
        # complex-mode identification's estimates are complex, and
        # dropping the imaginary halves would misreport the fit
        masses.append(complex(float(f[8]), float(f[9])))
        modal_dampings.append(complex(float(f[10]), float(f[11])))
        comments.append(_char_field(raw, base, BLOCK, int(i[39])))

    directions = DIRECTIONS_3 if dof_per_node == 3 else DIRECTIONS_6
    coordinate = [f'{int(node)}{direction}'
                  for node in triples[:, 0] for direction in directions]
    matrix = np.array(rows)

    # The format writes zeros for DOFs that were never measured and
    # keeps no record of which — the format documentation says so
    # outright. A column of exact zeros across every mode is therefore
    # fill from a uniaxial or biaxial layout, not a measurement:
    # float32 data never lands on 0.0 exactly, and even if it could,
    # an all-zero column holds no dynamics — while keeping it poisons
    # every MAC against a model with zeros nobody measured. Columns
    # that carry data in any mode stay whole; a zero there is
    # plausibly a node line and is kept as the measurement it claims
    # to be.
    measured = np.any(matrix != 0, axis=0)
    dropped = int(np.count_nonzero(~measured))
    if dropped and measured.any():
        matrix = matrix[:, measured]
        coordinate = [dof for dof, keep in zip(coordinate, measured)
                      if keep]

    # what the write-time table proves per mode: a node inserted after
    # a record was written is *undefined* there, not zero — knowledge
    # the format itself forgets, kept in the mode's description
    descriptions = []
    for e, stored in zip(shape_entries, counts):
        absent = current - stored
        descriptions.append(f'{absent} of {current} nodes not yet in '
                            'the file when this shape was written'
                            if absent else '')

    return ShapeSet(
        frequency=frequencies, damping=dampings, coordinate=coordinate,
        shape_matrix=matrix,
        modal_mass=(masses if any(m != 0 for m in masses) else None),
        comment=comments,
        description=(descriptions if any(descriptions) else None),
        modal_damping=(modal_dampings
                       if any(d != 0 for d in modal_dampings) else None))


# ---- writing -----------------------------------------------------------

#: block 0's fourth int32 says what the file holds (corpus block-0
#: survey: every .afu carries 1, .ati 4, .ash 9)
FILE_KIND = {'.afu': 1, '.ati': 4, '.ash': 9}

#: visualdynamics class name -> UFF 58 function type for the writer
_WRITE_TYPES = {'TimeHistory': 1, 'Frf': 4, 'Coherence': 6, 'Psd': 9,
                'Spectrum': 12}


def _pack_direction_code(text):
    text = (text or '').ljust(2)[:2]
    return text.encode('ascii')[0] | (text.encode('ascii')[1] << 8)


def _assemble(path, kind, records):
    """Lay out the container: header, zeros, then the chained directory
    interleaved with records exactly as the format lays it — a directory
    block, up to fifteen records, the next directory block right after
    the record that filled the previous one."""
    chunks = {}          # block index -> bytes
    header = np.zeros(128, dtype='<i4')
    header[0], header[1], header[3], header[4], header[5] = -1, 4, kind, 1, 7
    chunks[1] = b'\x00' * BLOCK

    groups = [records[k:k + 15] for k in range(0, len(records), 15)]
    directory_at, cursor = [], 2
    for group in groups:
        directory_at.append(cursor)
        cursor += 1
        for record in group:
            record['start'] = cursor + 1          # 1-based
            span = 2 + max(1, (len(record['data']) + BLOCK - 1) // BLOCK)
            record['blocks'] = span
            chunks[cursor] = record['header']
            chunks[cursor + 1] = record['chars']
            data = record['data'].ljust((span - 2) * BLOCK, b'\x00')
            for j in range(span - 2):
                chunks[cursor + 2 + j] = data[j * BLOCK:(j + 1) * BLOCK]
            cursor += span

    for g, group in enumerate(groups):
        block = np.zeros(128, dtype='<i4')
        for slot, record in enumerate(group):
            entry = np.array([
                record['id'], record['start'], record['blocks'],
                record.get('response_node', 0),
                _pack_direction_code(record.get('response_dir', '')),
                record.get('reference_node', 0),
                _pack_direction_code(record.get('reference_dir', '')),
                record.get('version', 0)], dtype='<i4')
            block[slot * 8: slot * 8 + 8] = entry
        if g + 1 < len(groups):
            block[127] = directory_at[g + 1] + 1   # 1-based chain
        chunks[directory_at[g]] = block.tobytes()

    # Block 0's tail, without which the reference tooling refuses
    # the file outright
    # ('0 bits per integer format not supported'): three creation
    # stamps (date, owner-length, owner), then the format descriptor
    # and the file's own accounting — record count, total blocks,
    # directory blocks — all pinned by diffing corpus block 0s.
    import getpass
    import time

    stamp = time.strftime('%d-%b-%y   %H:%M:%S')[:20].ljust(20)
    owner = getpass.getuser()[:16]
    head = bytearray(header.tobytes())
    for offset, tail in ((112, len(owner).to_bytes(4, 'little')
                          + owner.ljust(16).encode('ascii')),
                         (152, len(owner).to_bytes(4, 'little')
                          + owner.ljust(16).encode('ascii')),
                         (192, (1).to_bytes(4, 'little'))):
        text = stamp.encode('ascii') + tail
        head[offset:offset + len(text)] = text
    total_blocks = max(chunks) + 1
    for offset, value in ((460, 1), (464, 32), (476, 15), (480, 8),
                          (500, len(records)), (504, total_blocks),
                          (508, len(groups))):
        head[offset:offset + 4] = int(value).to_bytes(4, 'little')
    chunks[0] = bytes(head)

    with open(path, 'wb') as f:
        for k in range(max(chunks) + 1):
            piece = chunks.get(k, b'\x00' * BLOCK)
            f.write(piece.ljust(BLOCK, b'\x00'))


def _char_block(comment='', axis_labels=('', '', '', ''), owner=''):
    chars = bytearray(BLOCK)
    def put(offset, text, width):
        raw = str(text)[:width].encode('latin1', 'replace')
        chars[offset:offset + len(raw)] = raw
    put(0, comment, 80)                     # IDLine1
    put(400, axis_labels[0], 20)            # abscissa axis label
    put(420, axis_labels[1], 20)            # abscissa units label
    put(440, axis_labels[2], 20)            # ordinate axis label
    put(460, axis_labels[3], 20)            # ordinate units label
    put(480, owner, 16)
    return bytes(chars)


def _function_record(data_obj, index, record_number):
    """One record of a DataArray as an ADF function record."""
    from ..core.data import TimeHistory, parse_dof
    from ..units import SI

    kind = _WRITE_TYPES.get(type(data_obj).__name__)
    if kind is None:
        raise ValueError(f'{type(data_obj).__name__} has no ADF function '
                         'type; UNV or .vdyn carries it instead')

    abscissa = np.asarray(data_obj.abscissa, dtype=float)
    values = np.asarray(data_obj.ordinate[index])
    is_complex = bool(np.iscomplexobj(values) and np.any(values.imag))
    steps = np.diff(abscissa)
    even = bool(steps.size == 0
                or np.allclose(steps, steps[0], rtol=1e-6, atol=1e-12))

    dim = data_obj.ordinate_dim[index]
    defined = data_obj.ordinate_unit[index] is not None
    num_code = den_code = 0
    if defined and '/' in dim:
        num, den = dim.split('/', 1)
        num_code = _write_data_code(num)
        # every FRF the corpus holds writes its force denominator as
        # excitation force (13), not force (9)
        den_code = 13 if den == 'force' else _write_data_code(den)
    elif defined:
        num_code = _write_data_code(dim)

    node, direction = parse_dof(data_obj.response_dof[index])
    reference = (data_obj.reference_dof[index]
                 if data_obj.reference_dof is not None else '')
    ref_node, ref_dir = parse_dof(reference) if reference else (0, '')

    header = np.zeros(128, dtype='<i4')
    floats = header.view('<f4')
    header[0] = record_number
    header[1] = 1
    header[4] = kind
    header[5] = node or 0
    header[6] = _pack_direction_code(direction)
    header[8] = ref_node or 0
    header[9] = _pack_direction_code(ref_dir)
    header[11] = 5 if is_complex else 2
    header[12] = 1 if even else 0
    header[13] = len(values)
    header[14] = 17 if isinstance(data_obj, TimeHistory) else 18
    header[19] = num_code
    header[21] = 1
    header[24] = den_code
    header[27] = 1 if den_code else 0

    comment = data_obj.comment[index] if data_obj.comment else ''
    unit_text = SI.label(dim.split('/', 1)[0]) if defined else \
        (data_obj.dimension_hint[index] or '')
    axis_labels = ('Time' if header[14] == 17 else 'Frequency',
                   's' if header[14] == 17 else 'Hz',
                   dim.split('/', 1)[0] if defined else '',
                   unit_text)
    header[39] = len(comment[:80])
    header[44] = len(axis_labels[0][:20])
    header[45] = len(axis_labels[1][:20])
    header[46] = len(axis_labels[2][:20])
    header[47] = len(axis_labels[3][:20])

    if even:
        floats[96] = abscissa[0] if len(abscissa) else 0.0
        floats[97] = steps[0] if steps.size else 0.0
        payload = (np.column_stack([values.real, values.imag]).ravel()
                   if is_complex else values.real)
    else:
        if is_complex:
            payload = np.column_stack([abscissa, values.real,
                                       values.imag]).ravel()
        else:
            payload = np.column_stack([abscissa, values.real]).ravel()

    # the extremes of each part as the reference tooling writes them:
    # NaN passed over, an infinity kept, nothing at all as zero.
    # Clamped to zero until 2026-10-11, which wrote an all-positive
    # record's minimum as 0 and an all-negative record's maximum as 0
    # (the corpus files say otherwise)
    floats[100], floats[102] = _extremes(values.real.astype('<f4'))
    floats[101], floats[103] = (_extremes(values.imag.astype('<f4'))
                                if is_complex else (0.0, 0.0))
    floats[106] = 1.0
    floats[107] = 1.0

    return {
        'id': record_number,
        'header': header.tobytes(),
        'chars': _char_block(comment, axis_labels),
        'data': payload.astype('<f4').tobytes(),
        'response_node': node or 0, 'response_dir': direction,
        'reference_node': ref_node or 0, 'reference_dir': ref_dir,
        'version': 1,
    }


def _write_data_code(dim):
    from .unv import _58_CODES
    return _58_CODES.get(dim, 0)


def handles_functions(obj):
    from ..core.data import TimeHistory
    return (type(obj).__name__ in _WRITE_TYPES
            and not isinstance(obj, TimeHistory))


def handles_time(obj):
    from ..core.data import TimeHistory
    return isinstance(obj, TimeHistory)


def handles_shapes(obj):
    from ..core.shapes import ShapeSet
    return isinstance(obj, ShapeSet)


def save(obj, path, unit_system=None):
    """Write an ADF. Values go out in SI, which is what the format
    stores by definition — and in float32, which is all it has."""
    from ..core.shapes import ShapeSet

    path = str(path)
    extension = os.path.splitext(path)[1].lower()
    if isinstance(obj, ShapeSet):
        if extension != '.ash':
            raise ValueError('shapes write to .ash')
        return _save_shapes(obj, path)
    kind = FILE_KIND.get(extension)
    if kind is None:
        raise ValueError(f'{extension} is not an ADF extension')
    records = [_function_record(obj, k, k + 1)
               for k in range(obj.num_records)]
    _assemble(path, kind, records)


def _save_shapes(shapes, path):
    from ..core.data import parse_dof

    directions_6 = list(DIRECTIONS_6)
    nodes, seen_dirs = [], set()
    for dof in shapes.coordinate:
        node, direction = parse_dof(dof)
        if node is None:
            raise ValueError(f'{dof} names no node an ADF can store')
        base = direction.rstrip('+-') + '+'
        if base not in directions_6:
            raise ValueError(f'{dof} is not a direction an ADF can hold')
        if node not in nodes:
            nodes.append(node)
        seen_dirs.add(base)
    six = any(base.startswith('R') for base in seen_dirs)
    directions = DIRECTIONS_6 if six else DIRECTIONS_3
    order = sorted(nodes)
    dof = len(directions)
    complex_shapes = bool(np.iscomplexobj(shapes.shape_matrix)
                          and np.any(shapes.shape_matrix.imag))

    # densify: (node, direction) -> value, signs folded positive the
    # way I-DEAS keeps shape records
    index = {}
    for j, text in enumerate(shapes.coordinate):
        node, direction = parse_dof(text)
        sign = -1.0 if direction.endswith('-') else 1.0
        index[(node, direction.rstrip('+-') + '+')] = (j, sign)

    records = []
    table_header = np.zeros(128, dtype='<i4')
    capacity = max(250, len(order))
    table_header[0], table_header[1], table_header[2] = (NODE_TABLE,
                                                         len(order),
                                                         capacity)
    triples = np.zeros((len(order), 3), dtype='<i4')
    for k, node in enumerate(order):
        triples[k] = (node, node, k + 1)
    records.append({
        'id': NODE_TABLE,
        'header': table_header.tobytes(),
        'chars': b'\x00' * BLOCK,
        'data': triples.tobytes(),
        'version': 0,
    })

    for m in range(shapes.num_shapes):
        header = np.zeros(128, dtype='<i4')
        floats = header.view('<f4')
        header[0] = m + 1
        header[1] = len(order)
        header[2] = capacity
        # +12 is the shape type code (1 real, 2 complex), and the pair
        # at +48/+52 rides 1/1 only on real records — ash_complex vs
        # ash_3dof_real pins all three
        header[3] = 2 if complex_shapes else 1
        header[4] = dof
        floats[5] = float(shapes.frequency[m])
        floats[6] = float(shapes.damping[m]) * 100.0    # a percentage
        mass = complex(shapes.modal_mass[m])
        floats[8], floats[9] = mass.real, mass.imag
        if shapes.modal_damping is not None:
            md = complex(shapes.modal_damping[m])
            floats[10], floats[11] = md.real, md.imag
        header[12] = 0 if complex_shapes else 1
        header[13] = 0 if complex_shapes else 1
        comment = shapes.comment[m] if shapes.comment else ''
        header[39] = len(comment[:80])

        row = np.zeros(len(order) * dof,
                       dtype=complex if complex_shapes else float)
        for k, node in enumerate(order):
            for a, direction in enumerate(directions):
                hit = index.get((node, direction))
                if hit is not None:
                    j, sign = hit
                    row[k * dof + a] = sign * shapes.shape_matrix[m, j]
        payload = (np.column_stack([row.real, row.imag]).ravel()
                   if complex_shapes else row.real)
        records.append({
            'id': m + 1,
            'header': header.tobytes(),
            'chars': _char_block(comment),
            'data': payload.astype('<f4').tobytes(),
            'version': 0,
        })

    _assemble(path, FILE_KIND['.ash'], records)
