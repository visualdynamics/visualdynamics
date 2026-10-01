"""What an object will accept, in one place.

A type annotation is a *claim* — it tells an editor, a type checker and
the API reference what a thing is, and it does nothing at all at run
time. `node_id: NDArray[np.int64]` will not stop a string arriving. So
each claim needs an enforcement beside it, and the two belong in one
place or they drift: the helper here is what makes the annotation true,
and the annotation is what tells a reader the helper exists.

The failures worth catching are not the absurd ones. `'x'` for a node id
already fails, loudly, inside numpy. What silently succeeds is `1.7`
becoming node 1 — a mode shape drawn on the wrong node three weeks
later, with nothing in the file to say why.
"""

from __future__ import annotations

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
    """`values` as ids, or a refusal saying which one was wrong.

    Whole numbers only. `1.7` is not node 1 with a rounding error — it
    is a caller who has computed an id, and truncating it quietly is how
    that becomes somebody else's afternoon. Numeric strings *are*
    accepted: every file format in this field writes ids as text, and a
    reader that has to convert them before calling is a reader that will
    convert them wrongly.

    **Not negative.** An id is a label, and there is no meaning to give
    a negative one; zero is allowed, because files in the wild do number
    from zero and refusing them would be refusing real data.

    `unique` where something refers to these by id — nodes, coordinate
    systems and blocks. Element ids are labels nothing refers to
    (a UNV trace line that lifts the pen), so they are not held to it.
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
    """Damping as a fraction of critical: zero or more, per mode.

    No upper bound. Above 1.0 is overdamped — a real thing to measure
    even if it is not a mode anyone will animate — so the rule is a
    floor and not a range. Below zero is a fit that has gone wrong or a
    sign convention nobody meant, and a negative damping synthesizes an
    FRF that grows without bound.
    """
    return _nonnegative(values, f'{label} is a fraction of critical and '
                        'cannot be negative', label)


def frequency(values: ArrayLike,
              label: str = 'frequency') -> NDArray[np.float64]:
    """Modal frequencies in Hz: zero or more.

    Exactly zero is a rigid-body mode and is load-bearing — the FRF
    synthesis cancels its 0/0 by testing for it — so zero is not merely
    allowed, it is meaningful. Negative is not a frequency.
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
    else — the one reader of the `M<index>` spelling.

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
    """DOF strings: a node id, then one of `DIRECTIONS`. `'101RX+'`.
    Or a modal coordinate, `'M3'` (`modal_coordinate`).

    A DOF is the whole vocabulary of this toolset — it is how a record
    finds its node on a geometry, how a shape finds its column, how a
    channel finds what it measures. An unparseable one does not fail
    where it is written; it fails as a record that quietly measures
    nothing, on a geometry that quietly has no such node, which is a
    much worse afternoon than a refusal at the door.

    An unsigned direction is taken as the positive one — `'101Z'` is
    `'101Z+'` — because that is how people write them by hand, and the
    string is returned normalized so that everything downstream compares
    like with like.

    `allow_unknown` lets an **incomplete** DOF through, because a
    channel table is written by people and people leave fields out. A
    DOF string is the node and the direction concatenated, so whatever
    was recorded is what comes out:

    | node | direction | DOF     |
    |------|-----------|---------|
    | 101  | Z+        | `101Z+` |
    | 101  | —         | `101`   |
    | —    | Z+        | `Z+`    |
    | —    | —         | `''`    |

    All four import. Refusing the file would leave the user nothing to
    fix, where an imported record is visibly incompatible with any
    geometry and the channel table is right there to correct it in. What
    is still refused either way is a direction that is not a direction:
    `101Q+` is a typo, not an omission, and no amount of fixing the
    channel table will make Q an axis.
    """
    out = []
    for value in ([values] if isinstance(values, str) else values):
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
