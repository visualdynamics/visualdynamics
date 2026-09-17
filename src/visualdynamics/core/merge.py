"""Merging several objects of one type into one.

`mergeable` says whether — and, when not, why, which is what a test or a
tooltip wants; the GUI simply does not offer a merge that would refuse.
`merge` builds the combined object and touches nothing it was given.

The ground rules, per Brandon:

- Only objects of the same concrete type merge. Channel tables merge
  when their rows agree — a multi-pass survey carries one table per
  pass, with the reference rows repeated verbatim and the response
  rows disjoint. Identical rows (numbering aside) collapse to one,
  the rest concatenate, and channel numbers renumber: they are the
  wiring of one acquisition, not the identity of a measurement point.
  The same point with the same role described two different ways is a
  disagreement, and refuses by naming the point.
- A geometry merge must have disjoint node and coordinate-system ids —
  colliding numbers are a decision, not a default. Traceline and element
  ids only name themselves, so those renumber quietly.
- FRFs (and every other frequency-domain array) refuse when two records
  share a (response, reference) pair: the same measurement twice is not
  one set. Time histories are the exception — a repeated DOF is another
  average, and lands in its own averaging column (`block`) — but only
  two time-data merges tell a true story: repeat runs (every channel
  shared) or one acquisition split across files (no channel shared,
  equal capture counts). A mixture — a roving survey's passes — refuses
  (see `_time_refusal`).
- Units gate per object where units are object-wide (geometry, shapes):
  everything defined, or everything raw — mixing SI values with raw ones
  would add meters to numbers. Data-array units are per record and simply
  ride along.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .channel_table import ChannelTable
from .data import DataArray, Specification, TimeHistory
from .geometry import Geometry
from .shapes import ShapeSet


def mergeable(objects: Sequence[Any]) -> str | None:
    """None when the objects can merge; otherwise the reason they cannot."""
    objects = list(objects)
    if len(objects) < 2:
        return 'merging takes at least two objects'
    first = type(objects[0])
    if any(type(obj) is not first for obj in objects):
        return 'only objects of the same type merge'
    if first is ChannelTable:
        return _table_refusal(objects)
    if issubclass(first, Geometry):
        return _geometry_refusal(objects)
    if issubclass(first, ShapeSet):
        return _shapes_refusal(objects)
    if issubclass(first, DataArray):
        return _data_refusal(objects)
    return f'{first.__name__} does not merge'


def merge(objects: Sequence[Any]) -> Any:
    """The combined object; call `mergeable` first, this trusts it."""
    objects = list(objects)
    reason = mergeable(objects)
    if reason is not None:
        raise ValueError(reason)
    if isinstance(objects[0], ChannelTable):
        return _merge_tables(objects)
    if isinstance(objects[0], Geometry):
        return _merge_geometry(objects)
    if isinstance(objects[0], ShapeSet):
        return _merge_shapes(objects)
    return _merge_data(objects)


# ---- channel tables ---------------------------------------------------------

def _described(objects):
    """All rows of all tables, one frame, with the columns that describe
    the measurement — everything but the channel number, which is the
    wiring of one acquisition rather than the identity of a point."""
    import pandas as pd

    whole = pd.concat([obj.frame for obj in objects], ignore_index=True)
    describing = [name for name in whole.columns if name != 'channel']
    return whole, describing


def _table_refusal(objects):
    whole, describing = _described(objects)
    unique = whole.drop_duplicates(subset=describing)
    # after identical rows collapse, one point in one role must have
    # exactly one description — two sensitivities for the same channel
    # is a disagreement to resolve, not a table to build
    twice = unique[unique.duplicated(subset=['node', 'direction', 'role'],
                                     keep=False)]
    if len(twice):
        row = twice.iloc[0]
        return (f'the tables disagree about {row["node"]}{row["direction"]} '
                f'({row["role"]}): same point, different descriptions')
    return None


def _merge_tables(objects):
    whole, describing = _described(objects)
    unique = whole.drop_duplicates(subset=describing).reset_index(drop=True)
    unique['channel'] = np.arange(1, len(unique) + 1)
    return ChannelTable(unique)


# ---- geometry ---------------------------------------------------------------

def _geometry_refusal(objects):
    nodes = [{int(n) for n in obj.node_id} for obj in objects]
    if sum(len(s) for s in nodes) != len(set().union(*nodes)):
        return 'node numbers collide'
    # every geometry carries the global system, so shared ids are the
    # normal case — a collision is one id meaning two different frames
    seen = {}
    for obj in objects:
        for i, cs in enumerate(obj.cs_id):
            entry = (int(obj.cs_type[i]), obj.cs_matrix[i])
            known = seen.get(int(cs))
            if known is None:
                seen[int(cs)] = entry
            elif known[0] != entry[0] or not np.allclose(known[1],
                                                        entry[1]):
                return 'coordinate system numbers collide'
    defined = [obj.units_defined for obj in objects]
    if any(defined) and not all(defined):
        return 'some have units defined and some do not'
    return None


def _merge_geometry(objects):
    def concat(attribute: str) -> np.ndarray:
        return np.concatenate([getattr(obj, attribute) for obj in objects])

    def listed(attribute: str) -> list[Any]:
        return [entry for obj in objects for entry in getattr(obj, attribute)]

    unique_cs, names, types, matrices = [], {}, {}, {}
    for obj in objects:
        for i, cs in enumerate(obj.cs_id):
            cs = int(cs)
            if cs not in names:
                unique_cs.append(cs)
                names[cs] = obj.cs_name[i]
                types[cs] = int(obj.cs_type[i])
                matrices[cs] = obj.cs_matrix[i]

    return Geometry(
        node_id=concat('node_id'), node_xyz=concat('node_xyz'),
        node_def_cs=concat('node_def_cs'),
        node_disp_cs=concat('node_disp_cs'),
        node_color=concat('node_color'),
        # identical systems dedupe; the refusal above already guaranteed
        # a shared id means the same frame
        cs_id=[cs for cs in unique_cs],
        cs_name=[names[cs] for cs in unique_cs],
        cs_type=[types[cs] for cs in unique_cs],
        cs_matrix=[matrices[cs] for cs in unique_cs],
        # traceline and element ids only name themselves: renumber
        traceline_color=concat('traceline_color'),
        traceline_desc=listed('traceline_desc'),
        traceline_conn=listed('traceline_conn'),
        elem_type=concat('elem_type'), elem_color=concat('elem_color'),
        elem_conn=listed('elem_conn'),
        length_unit=objects[0].length_unit)


# ---- shapes -----------------------------------------------------------------

def _shapes_refusal(objects):
    dofs = set(objects[0].coordinate)
    if any(set(obj.coordinate) != dofs for obj in objects[1:]):
        return 'the shape sets cover different DOFs'
    defined = [obj.units_defined for obj in objects]
    if any(defined) and not all(defined):
        return 'some have units defined and some do not'
    return None


def _merge_shapes(objects):
    order = list(objects[0].coordinate)
    columns = []
    for obj in objects:
        position = {dof: i for i, dof in enumerate(obj.coordinate)}
        columns.append(obj.shape_matrix[:, [position[dof] for dof in order]])
    merged = ShapeSet(
        frequency=np.concatenate([obj.frequency for obj in objects]),
        damping=np.concatenate([obj.damping for obj in objects]),
        coordinate=order,
        shape_matrix=np.vstack(columns),
        modal_mass=np.concatenate([obj.modal_mass for obj in objects]),
        comment=[c for obj in objects for c in obj.comment],
        description=[d for obj in objects for d in obj.description],
        mass_unit=objects[0].mass_unit,
        unscaled=any(getattr(obj, 'unscaled', False)
                     for obj in objects))
    # a mode table reads by frequency, whatever order the sets arrived in
    sorted_order = np.argsort(merged.frequency, kind='stable')
    merged.frequency = merged.frequency[sorted_order]
    merged.damping = merged.damping[sorted_order]
    merged.modal_mass = merged.modal_mass[sorted_order]
    merged.shape_matrix = merged.shape_matrix[sorted_order]
    merged.comment = [merged.comment[i] for i in sorted_order]
    merged.description = [merged.description[i] for i in sorted_order]
    return merged


# ---- data arrays ------------------------------------------------------------

def _pairs(obj):
    reference = obj.reference_dof or obj.response_dof
    return [(obj.response_dof[i], reference[i], obj.known_dim(i))
            for i in range(obj.num_records)]


def _data_refusal(objects):
    first = objects[0]
    for obj in objects[1:]:
        if not np.array_equal(obj.abscissa, first.abscissa):
            return 'the abscissas differ'
    if isinstance(first, Specification):
        keys = set(first.limits)
        if any(set(obj.limits) != keys for obj in objects[1:]):
            return 'the specifications carry different limit curves'
    if isinstance(first, TimeHistory):
        return _time_refusal(objects)
    pairs = [pair for obj in objects for pair in _pairs(obj)]
    if len(set(pairs)) != len(pairs):
        return 'the same response/reference pair appears more than once'
    return None


def _time_refusal(objects):
    """Only two merges of time data tell a true story (Brandon,
    2026-08-29). Repeat runs — every channel shared — where each
    record is another average of the same setup. Or one acquisition
    split across files — no channel shared — where merging claims the
    records played at the same time. A mixture is neither: a roving
    survey's passes share their references and nothing else, and
    merging them built forces numbered 1-120 against accelerations
    numbered 1-20, with which-caused-which kept only as row order.

    Both true stories leave every channel holding the same number of
    captures, so that is checked too: frame-by-frame pairing is what
    every cross-spectrum does with this object, and unequal counts
    would pair captures that never played together.
    """
    keysets = [{obj.channel_key(i) for i in range(obj.num_records)}
               for obj in objects]
    repeat = all(keys == keysets[0] for keys in keysets[1:])
    union = set().union(*keysets)
    if not repeat and sum(len(keys) for keys in keysets) != len(union):
        shared = next(iter(keysets[0] & set().union(*keysets[1:])))
        alone = next(iter(union - set.intersection(*keysets)))
        return (f'the histories share {shared[0]} but not {alone[0]}: '
                'time data merges as repeat runs (every channel shared) '
                'or as one acquisition split across files (none shared), '
                'and this is neither')
    counts: dict = {}
    for obj in objects:
        for i in range(obj.num_records):
            key = obj.channel_key(i)
            counts[key] = counts.get(key, 0) + 1
    if len(set(counts.values())) > 1:
        least = min(counts, key=counts.get)
        most = max(counts, key=counts.get)
        return (f'every channel of a merged history must hold the same '
                f'number of captures — {most[0]} would hold '
                f'{counts[most]} and {least[0]} {counts[least]}, and '
                'pairing them capture by capture would pair records '
                'that never played together')
    return None


def _merged_blocks(objects):
    """One averaging column per repeat of a (DOF, quantity) pair.

    Original block labels survive where a pair appears once; repeats are
    renumbered 'avg 1', 'avg 2', ... in arrival order, which is what two
    single-hit sets or two 20-average sets both want.
    """
    totals = {}
    for obj in objects:
        for pair in _pairs(obj):
            totals[pair[:2] + (pair[2],)] = totals.get(
                pair[:2] + (pair[2],), 0) + 1
    if (all(count == 1 for count in totals.values())
            and all(obj.block is None for obj in objects)):
        return None
    blocks, seen = [], {}
    for obj in objects:
        for i in range(obj.num_records):
            key = (obj.response_dof[i], obj.known_dim(i))
            seen[key] = seen.get(key, 0) + 1
            original = obj.block[i] if obj.block is not None else None
            repeated = totals.get(
                (obj.response_dof[i],
                 (obj.reference_dof or obj.response_dof)[i],
                 obj.known_dim(i)), 1) > 1
            blocks.append(f'avg {seen[key]}' if repeated
                          else (original or 'avg 1'))
    return blocks


def _merge_data(objects):
    first = objects[0]
    cls = type(first)

    def listed(attribute: str) -> list[Any]:
        out = []
        for obj in objects:
            values = getattr(obj, attribute)
            out.extend(values if values is not None
                       else [None] * obj.num_records)
        return out

    references = None
    if any(obj.reference_dof is not None for obj in objects):
        references = [dof for obj in objects
                      for dof in (obj.reference_dof or obj.response_dof)]

    extras = {}
    if isinstance(first, TimeHistory):
        blocks = _merged_blocks(objects)
        if blocks is not None:
            extras['block'] = blocks
    elif all(obj.block is not None for obj in objects):
        # duplicate pairs were refused above, so existing repeat labels
        # (a spectral save's frames) simply ride along
        extras['block'] = [label for obj in objects for label in obj.block]
    if isinstance(first, Specification):
        for name in first.limits:
            extras[name] = np.vstack([obj.limits[name] for obj in objects])

    merged = cls(
        first.abscissa,
        np.vstack([obj.ordinate for obj in objects]),
        response_dof=listed('response_dof'),
        reference_dof=references,
        ordinate_dim=listed('ordinate_dim'),
        ordinate_unit=listed('ordinate_unit'),
        reference_unit=listed('reference_unit'),
        comment=listed('comment'),
        dimension_hint=listed('dimension_hint'),
        **extras)
    return merged
