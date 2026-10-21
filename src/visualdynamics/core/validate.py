"""The checks on what an object accepts: ids, DOFs, frequencies and
damping.

A type annotation tells an editor, a type checker and the API reference
what a value is, and does nothing at run time:
`node_id: NDArray[np.int64]` does not stop a string being passed. Each
annotation here has a function that enforces it, kept in the same
module so the two do not drift apart.

The errors these catch are the ones that would otherwise pass without
notice. `'x'` as a node id already fails inside numpy. `1.7` would
become node 1, and a mode shape would be drawn on the wrong node with
nothing in the file to say why.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: The ids of nodes, coordinate systems, elements and
#: blocks: whole numbers, stored as int64. Ids are *labels* — they are
#: what connectivity, placement and element blocks refer to by — so
#: nothing is inferred from their value and nothing is computed with
#: them.
IdArray = NDArray[np.int64]

#: What may be handed in where an `IdArray` is stored.
Ids = Sequence[int] | NDArray[np.integer]

#: A DOF string: a node id and a direction, concatenated — `'101RX+'`.
#: Deliberately a string and not a class. A geometry has two hundred
#: thousand nodes and a matrix of FRFs has thousands of DOFs; one object
#: per node would be a hundred thousand Python objects where an int64
#: array is what VTK, the FFT and HDF5 all want. It is also the
#: interchange vocabulary — UFF writes `101RX+`, and so does everything
#: that reads a UFF — so a class would have to be unwrapped at every
#: boundary. The rules live here instead, once, and the objects call
#: them: `Dof` says what a string means, `dofs()` says what is allowed.
Dof = str


def ids(values: ArrayLike, label: str, unique: bool = False) -> IdArray:
    """`values` as an int64 id array, or a ValueError naming the values
    that are wrong.

    Whole numbers only. `1.7` is not node 1 with a rounding error; it
    means the caller computed an id, and truncating it without notice
    would hide that. Numeric strings are accepted, because the file
    formats in this field write ids as text, and requiring every reader
    to convert them first would mean many places to get the conversion
    wrong.

    Ids cannot be negative: an id is a label, and a negative one has no
    meaning. Zero is allowed, because real files do number from zero.

    Set `unique` where something refers to these by id: nodes,
    coordinate systems and blocks. Element ids are labels nothing refers
    to (a UNV trace line that lifts the pen, for example), so they are
    not checked for duplicates. `label` names the values in the error
    message.
    """
    array = np.asarray(values)
    if array.size == 0:
        return np.asarray([], dtype=np.int64)
    if array.dtype.kind == 'f':
        whole = np.isfinite(array) & (array == np.floor(array))
        if not whole.all():
            bad = array[~whole][:3]
            raise ValueError(
                f'{label} must be whole numbers; got '
                + ', '.join(repr(float(v)) for v in bad))
    elif array.dtype.kind not in 'iu':
        # strings and objects: let numpy say which entry is not a number
        try:
            array = array.astype(np.int64)
        except (TypeError, ValueError) as bad:
            raise ValueError(f'{label} must be whole numbers: {bad}') from None
    out = array.astype(np.int64)
    if (out < 0).any():
        bad = out[out < 0][:3]
        raise ValueError(
            f'{label} cannot be negative; got '
            + ', '.join(str(int(v)) for v in bad))
    if unique and len(np.unique(out)) != len(out):
        seen, repeated = set(), []
        for value in out.tolist():
            if value in seen and value not in repeated:
                repeated.append(value)
            seen.add(value)
        raise ValueError(
            f'duplicate {label}: ' + ', '.join(str(v) for v in repeated[:5]))
    return out


def damping(values: ArrayLike,
            label: str = 'damping') -> NDArray[np.float64]:
    """Damping as a fraction of critical, per mode: finite and zero or
    more.

    There is no upper bound. Above 1.0 is overdamped, which can be
    measured even if it is not a mode anyone will animate, so the check
    is a lower limit and not a range. A negative value comes from a
    failed fit or an unintended sign convention, and a negative damping
    synthesizes an FRF that grows without bound.
    """
    return _nonnegative(values, f'{label} is a fraction of critical and '
                        'cannot be negative', label)


def frequency(values: ArrayLike,
              label: str = 'frequency') -> NDArray[np.float64]:
    """Modal frequencies in Hz, per mode: finite and zero or more.

    Exactly zero is a rigid-body mode, and the FRF synthesis tests for
    it to cancel its 0/0, so zero is allowed and has a meaning. A
    negative value is not a frequency.
    """
    return _nonnegative(values, f'{label} cannot be negative', label)


#: The axes a DOF may name, in UFF code order: 1..6 are the three
#: translations then the three rotations, and a negative code is the
#: same axis the other way. Every format this package reads and writes
#: speaks this vocabulary, which is why it is one list and not one per
#: reader — `data._DIRECTIONS` builds its code map from this, and
#: `validate.dofs` checks against it.
AXES: tuple[str, ...] = ('X', 'Y', 'Z', 'RX', 'RY', 'RZ')

#: Those axes with their signs — what a DOF string may end with.
DIRECTIONS: tuple[str, ...] = tuple(
    f'{axis}{sign}' for axis in AXES for sign in '+-')

#: {direction: UFF code}. Positive codes count up from 1 in `AXES`
#: order; the negative direction of an axis is the negative code.
DIRECTION_CODES: dict[str, int] = {
    f'{axis}{sign}': code * (1 if sign == '+' else -1)
    for code, axis in enumerate(AXES, start=1) for sign in '+-'}


#: the spelling of a modal coordinate: the letter, then the mode's
#: 1-based index — `'M3'` is the third mode of whatever set the object
#: was transformed through (its provenance says which). Visibly not a
#: node, and one rule for every set, rigid or fitted (Brandon,
#: 2026-09-04: a virtual node's X…RZ was one set's special case).
MODAL_PREFIX = 'M'


def modal_coordinate(text: str) -> int | None:
    """The mode index a modal coordinate names, or None for anything
    else. This is the one function that reads the `M<index>` spelling.

    Parameters
    ----------
    text : str
        A DOF string.

    Returns
    -------
    int or None
        The 1-based mode index, or None when the string is not a
        modal coordinate.
    """
    text = str(text).strip().upper()
    if len(text) > 1 and text[0] == MODAL_PREFIX and text[1:].isdigit():
        index = int(text[1:])
        return index if index > 0 else None
    return None


def dofs(values: Sequence[str], label: str = 'DOF',
         allow_unknown: bool = False) -> list[str]:
    """DOF strings, checked and normalized: a node id, then one of
    `DIRECTIONS` (`'101RX+'`), or a modal coordinate (`'M3'`, see
    `modal_coordinate`).

    A record finds its node on a geometry by its DOF, a shape finds its
    column by it, and a channel says what it measures with it. A DOF
    that cannot be parsed would not fail where it was written; it would
    show up later as a record matched to no node on the geometry. So it
    is refused here, with a ValueError naming the string.

    An unsigned direction is taken as the positive one (`'101Z'` is
    `'101Z+'`), because that is how people write them by hand. The
    direction is upper-cased and the node id written without leading
    zeros, so that later comparisons are between identical strings.
    A single string is accepted as a list of one.

    `allow_unknown` lets an incomplete DOF through, because a channel
    table is written by people and people leave fields out. A DOF
    string is the node and the direction concatenated, so whatever was
    recorded is what comes out:

    | node | direction | DOF     |
    |------|-----------|---------|
    | 101  | Z+        | `101Z+` |
    | 101  | —         | `101`   |
    | —    | Z+        | `Z+`    |
    | —    | —         | `''`    |

    All four import. Refusing the file would leave the user nothing to
    fix, while an imported record is shown as matching no geometry and
    can be corrected in the channel table. A direction that is not a
    direction is still refused either way: `101Q+` is a typo, not an
    omission, and Q is not an axis.
    """
    items = [values] if isinstance(values, str) else list(values)
    normalized = _dofs_normalized(items)
    if normalized is not None:
        return normalized
    return _dofs_each(items, label, allow_unknown)


#: one DOF string already in the normalized spelling `dofs` produces:
#: a node id with no leading zero (zero itself allowed), then a signed
#: upper-case direction — or a modal coordinate with a positive index.
#: ASCII digits only: `\d` would also match the other scripts' digits,
#: which `str.isdigit` accepts and `int` converts, so the slow path
#: rewrites them and this one must not pass them through.
_NORMALIZED = re.compile(
    r'(?:(?:0|[1-9][0-9]*)(?:' + '|'.join(re.escape(d) for d in DIRECTIONS)
    + r')|' + MODAL_PREFIX + r'[1-9][0-9]*)\n')
_ALL_NORMALIZED = re.compile(r'(?:' + _NORMALIZED.pattern + r')*')


def _dofs_normalized(items: list) -> list[str] | None:
    """`items` as a list of str when every one is already normalized,
    else None.

    The fast path: a solved model's coordinates are hundreds of
    thousands of strings its own `dof_strings` wrote in exactly this
    spelling, and `_dofs_each` over them, one Python call per string,
    was the largest cost left in a `ShapeSet`. One regex over the
    strings joined with newlines reads them in C; anything it does not
    recognize — a sign left off, a lower-case axis, a leading zero, a
    blank, a non-string — goes to `_dofs_each`, which is the rule. The
    join refuses a non-string by raising, and the separator count
    catches a string with a newline inside it, which would otherwise
    read as two good ones.
    """
    try:
        joined = '\n'.join(items)
    except TypeError:
        return None
    if joined.count('\n') != len(items) - 1:
        return None
    if not _ALL_NORMALIZED.fullmatch(joined + '\n'):
        return None
    return [str(item) for item in items]


def _dofs_each(values: Sequence[str], label: str,
               allow_unknown: bool) -> list[str]:
    """`dofs`, one string at a time: the reference the fast path must
    agree with (`tests/test_validate_rules.py` holds it to that)."""
    out = []
    for value in values:
        text = '' if value is None else str(value).strip()
        if not text:
            if allow_unknown:
                out.append('')
                continue
            raise ValueError(f'{label} is empty')
        index = modal_coordinate(text)
        if index is not None:
            out.append(f'{MODAL_PREFIX}{index}')
            continue
        digits = 0
        while digits < len(text) and text[digits].isdigit():
            digits += 1
        node, direction = text[:digits], text[digits:].upper()
        if not digits and not allow_unknown:
            raise ValueError(
                f'{label} {value!r} does not start with a node id')
        if direction and direction not in DIRECTIONS:
            if direction + '+' in DIRECTIONS:
                direction += '+'
            else:
                raise ValueError(
                    f'{label} {value!r} names no direction: expected one of '
                    + ', '.join(DIRECTIONS))
        if not direction and not allow_unknown:
            raise ValueError(
                f'{label} {value!r} names no direction: expected one of '
                + ', '.join(DIRECTIONS))
        out.append(f'{int(node)}{direction}' if node else direction)
    return out


def _nonnegative(values: ArrayLike, refusal: str,
                 label: str) -> NDArray[np.float64]:
    """Finite and zero or more, per element — the floor `damping` and
    `frequency` share; each states its own reason above."""
    array = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if not np.isfinite(array).all():
        raise ValueError(f'{label} must be a number')
    if (array < 0).any():
        bad = array[array < 0][:3]
        raise ValueError(
            f'{refusal}; got ' + ', '.join(f'{float(v):g}' for v in bad))
    return array
