"""A geometry's entities, grouped the way the project tree lists them.

`geometry.nodes`, `.coordinate_systems`, `.elements` and
`.groups` each hand back a view: the same rows you can expand in the
tree, reachable from a script under the same names. A view is a window onto
the geometry's own arrays and never a copy of them, so writing through a
row writes into the geometry:

    geometry.nodes[0].xyz = [0.0, 0.1, 0.0]     # moves the node
    geometry.nodes.xyz[:, 2] += 0.5             # and so does this

Columns keep their plural name on the view and their singular on a row —
`nodes.ids` is every id, `nodes[3].id` is one. Adding and deleting are
forwarded to the geometry, which is what keeps ids unique and
connectivity honest; everything is deleted by id, in every group.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from .validate import Ids

if TYPE_CHECKING:                                    # pragma: no cover
    from .geometry import Geometry

# (plural on the view, singular on a row, what the geometry stores it as)
Column = tuple[str, str, str]

COLUMNS: dict[str, tuple[Column, ...]] = {
    'nodes': (
        ('ids', 'id', 'node_id'),
        ('xyz', 'xyz', 'node_xyz'),
        ('colors', 'color', 'node_color'),
        ('placement_systems', 'placement_system', 'node_def_cs'),
        ('displacement_systems', 'displacement_system', 'node_disp_cs'),
    ),
    'coordinate_systems': (
        ('ids', 'id', 'cs_id'),
        ('names', 'name', 'cs_name'),
        ('types', 'type', 'cs_type'),
        ('matrices', 'matrix', 'cs_matrix'),
    ),
    'elements': (
        ('ids', 'id', 'elem_id'),
        ('types', 'type', 'elem_type'),
        ('colors', 'color', 'elem_color'),
        ('groups', 'group', 'elem_group'),
        ('nodes', 'nodes', 'elem_conn'),
    ),
    # An element group is a *grouping* of elements rather than a thing in space: it
    # has an id and a name and nothing else, and what it holds is read off
    # the elements that name it (`elements_in`). Listing it beside the
    # other three is what the file says — exodus writes a mesh as element blocks —
    # and it is the only one of the four nothing is drawn for.
    'groups': (
        ('ids', 'id', 'group_id'),
        ('names', 'name', 'group_name'),
    ),
}

# what the geometry calls adding and deleting one of these
VERBS: dict[str, tuple[str, str]] = {
    'nodes': ('add_node', 'delete_nodes'),
    'coordinate_systems': ('add_coordinate_system',
                           'delete_coordinate_systems'),
    'elements': ('add_element', 'delete_elements'),
    'groups': ('add_group', 'delete_groups'),
}

LABELS: dict[str, str] = {
    'nodes': 'node',
    'coordinate_systems': 'coordinate system',
    'elements': 'element',
    'groups': 'element group',
}


class EntityRow:
    """One node, coordinate system, element or element group, in place.

    Holds a row number rather than the values, so it reads and writes the
    geometry as it stands. Delete rows out from under one and it will be
    describing whatever moved up into its place — take the values you
    need rather than keeping the row.
    """

    __slots__ = ('_row', '_view')

    def __init__(self, view: EntityView, row: int) -> None:
        object.__setattr__(self, '_view', view)
        object.__setattr__(self, '_row', row)

    @property
    def row(self) -> int:
        """Where this sits in the geometry's arrays."""
        return self._row

    def __getattr__(self, name: str) -> Any:
        attr = self._view._by_singular.get(name)
        if attr is None:
            raise AttributeError(
                f'{LABELS[self._view.kind]} has no {name!r}; it has '
                + ', '.join(repr(single) for _p, single, _a
                            in COLUMNS[self._view.kind]))
        return getattr(self._view.geometry, attr)[self._row]

    def __setattr__(self, name: str, value: Any) -> None:
        attr = self._view._by_singular.get(name)
        if attr is None:
            raise AttributeError(
                f'cannot set {name!r} on a {LABELS[self._view.kind]}')
        getattr(self._view.geometry, attr)[self._row] = value

    def __repr__(self) -> str:
        parts = []
        for _plural, single, _attr in COLUMNS[self._view.kind]:
            if single == 'matrix':                 # 4x3, not worth printing
                continue
            value = getattr(self, single)
            if hasattr(value, 'tolist'):
                value = value.tolist()
            parts.append(f'{single}={value!r}')
        return f'<{LABELS[self._view.kind]} {", ".join(parts)}>'


class EntityView:
    """One of a geometry's four groups of entities.

    Reached as `geometry.nodes` and friends. Iterating gives one
    `EntityRow` per entity; a column is reached by its plural name and is
    the geometry's own array, so slicing and assigning into it edit the
    geometry in place.
    """

    def __init__(self, geometry: Geometry, kind: str) -> None:
        self.geometry: Geometry = geometry
        self.kind: str = kind
        self._by_plural = {plural: attr
                           for plural, _single, attr in COLUMNS[kind]}
        self._by_singular = {single: attr
                             for _plural, single, attr in COLUMNS[kind]}

    def __len__(self) -> int:
        first = COLUMNS[self.kind][0][2]
        return len(getattr(self.geometry, first))

    def __iter__(self) -> Iterator[EntityRow]:
        return (EntityRow(self, row) for row in range(len(self)))

    def __getitem__(self,
                    index: int | slice) -> EntityRow | list[EntityRow]:
        if isinstance(index, slice):
            return [EntityRow(self, row)
                    for row in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(
                f'{self.kind}[{index}]: there are {len(self)}')
        return EntityRow(self, index)

    def __getattr__(self, name: str) -> Any:
        # __getattr__ runs only when normal lookup fails, so the real
        # attributes set in __init__ never come through here
        attr = self._by_plural.get(name)
        if attr is None:
            raise AttributeError(
                f'{self.kind} has no column {name!r}; it has '
                + ', '.join(repr(plural) for plural, _s, _a
                            in COLUMNS[self.kind]))
        return getattr(self.geometry, attr)

    @property
    def columns(self) -> tuple[str, ...]:
        """What this view can be asked for, in table order."""
        return tuple(plural for plural, _s, _a in COLUMNS[self.kind])

    def add(self, *args: Any, **kwargs: Any) -> int:
        """Add one, by the geometry's rules — it keeps ids unique and
        checks that connectivity names nodes that exist."""
        return getattr(self.geometry, VERBS[self.kind][0])(*args, **kwargs)

    def delete(self, ids: Ids) -> None:
        """Delete by id, in every group — an id names an entity here the
        same way it does everywhere else. Ids that are not there are
        ignored, so deleting twice is not an error."""
        getattr(self.geometry, VERBS[self.kind][1])(ids)

    def __repr__(self) -> str:
        count = len(self)
        label = LABELS[self.kind]
        return f'<{count} {label}{"s" * (count != 1)}>'
