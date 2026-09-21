"""Do the objects in a test belong together?

Everything carrying DOFs — mode shapes, data records, channel table rows —
names nodes. Those nodes have to exist in the geometry the test is being
interpreted against, or the object describes a different structure: airplane
mode shapes beside a plate geometry are not a test, they are a mistake.

Compatibility is judged against the geometry each object **answers
to**: its link group's geometry when it has one — a FEM shape set
beside its own FEM mesh is consistent, whichever geometry is active —
and the one **active geometry** otherwise, so a test holding two models
gives a definite answer instead of "compatible with something". Objects
with no DOFs (a geometry itself) are never flagged, and with no active
geometry nothing is checked — importing data before its geometry must
not paint everything red.

A data object whose DOFs are **modal coordinates** (`M1` … `Mn`,
`validate.modal_coordinate`) names no node at all: it answers to a
*shape set*, the one it was transformed through, and fits wherever a
set with at least that many modes is — its link group's set when it
has one, any set in the project otherwise (Brandon, 2026-09-04: the
modal responses belong in the group with the shapes and the record
they came from).

Deliberately not part of this:

- units, which have their own indicator
- direction validity: a geometry does not restrict directions, so 101RX+ is
  fine wherever node 101 exists
- whether two data objects can be compared with each other
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                                    # pragma: no cover
    from .core.geometry import Geometry

MAX_LISTED = 6  # DOFs named in a message before it summarizes


@dataclass
class Issue:
    """Why one object does not fit the active geometry."""

    name: str
    kind: str
    message: str
    missing_dofs: list = field(default_factory=list)
    sub_items: list = field(default_factory=list)  # indices to mark


@dataclass
class Report:
    """Compatibility of every object in a test."""

    geometry_name: str | None = None
    issues: dict = field(default_factory=dict)

    def is_compatible(self, name: str) -> bool:
        """Whether one object fits the geometry it is linked to.

        Parameters
        ----------
        name : str
            The object to ask about.

        Returns
        -------
        bool
            Whether it fits the geometry it is linked to.
        """
        return name not in self.issues

    def issue_for(self, name: str) -> Issue | None:
        """What is wrong with one object, or None if nothing is.

        Parameters
        ----------
        name : str
            The object to ask about.

        Returns
        -------
        Issue or None
            What is wrong with it, or None if nothing is.
        """
        return self.issues.get(name)

    def sub_item_flagged(self, name: str, index: int) -> bool:
        """Whether one record within an object is incompatible.

        Parameters
        ----------
        name : str
            The object to ask about.
        index : int
            Which record within it.

        Returns
        -------
        bool
            Whether that record is one of the incompatible ones.
        """
        issue = self.issues.get(name)
        return bool(issue) and index in issue.sub_items

    @property
    def incompatible_names(self) -> list[str]:
        """The objects that do not fit the geometry, by name — what
        the tree marks in red and a link refuses over."""
        return list(self.issues)


def _listed(dofs: Sequence[str]) -> str:
    """'501X+, 501Y+ and 3 more' — enough to identify, short enough to read."""
    text = ', '.join(dofs[:MAX_LISTED])
    if len(dofs) > MAX_LISTED:
        text += f' and {len(dofs) - MAX_LISTED} more'
    return text


def modal_object(obj: Any) -> int | None:
    """The highest mode a data object's coordinates name, when every
    one of its DOFs is a modal coordinate — else None."""
    from .core.data import DataArray
    from .core.validate import modal_coordinate

    if not isinstance(obj, DataArray) or not obj.num_records:
        return None
    indices = [modal_coordinate(dof) for dof in obj.response_dof]
    if any(index is None for index in indices):
        return None
    return max(indices)


def check_modal(name: str, obj: Any, companions: Mapping[str, Any]
                ) -> Issue | None:
    """The Issue for a modal object against the shape sets among its
    companions, or None when one of them has every mode it names."""
    from .core.shapes import ShapeSet

    highest = modal_object(obj)
    sets = {other: value for other, value in companions.items()
            if isinstance(value, ShapeSet) and other != name}
    if any(value.num_shapes >= highest for value in sets.values()):
        return None
    beyond = [i for i, dof in enumerate(obj.response_dof)
              if _index(dof) > max((v.num_shapes for v in sets.values()),
                                   default=0)]
    if not sets:
        message = (f'modal coordinates up to M{highest}, and no shape set '
                   'to answer to — link the set they were transformed '
                   'through')
    else:
        most = max(sets.items(), key=lambda item: item[1].num_shapes)
        message = (f'modal coordinates up to M{highest}, but shape set '
                   f'{most[0]!r} has {most[1].num_shapes} modes')
    return Issue(name=name, kind='modes-not-in-set', message=message,
                 missing_dofs=[obj.response_dof[i] for i in beyond],
                 sub_items=beyond)


def _index(dof: str) -> int:
    from .core.validate import modal_coordinate

    return modal_coordinate(dof) or 0


def blocks_a_link(issue: Issue | None) -> bool:
    """Whether an issue is reason to refuse a link outright.

    Naming nodes the geometry lacks is **not**. It is reported, in
    red, by the indicator this module exists to feed — and that is
    the whole of the answer it gets. A link is a statement about
    which objects belong together, and a test names places the model
    does not draw as a matter of course: drive points, control
    coordinates, computed locations, virtual channels. Twice in two
    days a real report died on them (Brandon, 2026-09-20 and
    2026-09-21), the second time on an object whose channels were
    *all* virtual — which is why the first attempt at this, refusing
    only when nothing overlapped, was not a line that could be drawn.
    A controller's control object and a foreign mode shape look the
    same from here; the person reading the red indicator can tell
    them apart, and this function cannot.

    What still blocks is what is not about nodes at all: a modal
    record with no shape set to answer to has nothing to be drawn
    against in any geometry. (Two geometries in one link is refused
    before this is ever called.)
    """
    return issue is not None and issue.kind != 'dofs-not-in-geometry'


def check_object(name: str, obj: Any, geometry: Geometry | None,
                 geometry_name: str = 'the active geometry',
                 companions: Mapping[str, Any] | None = None) -> Issue | None:
    """The Issue for one object against a geometry, or None if it fits.

    `companions` are the objects it is grouped with, for a modal
    object, which answers to a shape set among them rather than to
    the geometry.
    """
    from .core.channel_table import ChannelTable
    from .core.data import DataArray
    from .core.geometry import Geometry
    from .core.shapes import ShapeSet

    if isinstance(obj, Geometry):
        return None
    if modal_object(obj) is not None:
        return check_modal(name, obj, companions or {})
    if geometry is None:
        return None

    if isinstance(obj, ShapeSet):
        missing = geometry.missing_dofs(obj.coordinate)
        if not missing:
            return None
        return Issue(
            name=name, kind='dofs-not-in-geometry',
            message=(f'{len(missing)} of {obj.num_dofs} shape DOFs are not in '
                     f'geometry {geometry_name!r}: {_listed(missing)}'),
            missing_dofs=missing,
            # every mode spans the same DOFs, so all of them are affected
            sub_items=list(range(obj.num_shapes)))

    if isinstance(obj, DataArray):
        missing, flagged = [], []
        for i in range(obj.num_records):
            dofs = [obj.response_dof[i]]
            if obj.reference_dof is not None:
                dofs.append(obj.reference_dof[i])
            gone = geometry.missing_dofs(dofs)
            if gone:
                flagged.append(i)
                missing.extend(d for d in gone if d not in missing)
        if not flagged:
            return None
        return Issue(
            name=name, kind='dofs-not-in-geometry',
            message=(f'{len(flagged)} of {obj.num_records} records reference '
                     f'DOFs not in geometry {geometry_name!r}: '
                     f'{_listed(missing)}'),
            missing_dofs=missing, sub_items=flagged)

    if isinstance(obj, ChannelTable):
        dofs = obj.dof_strings()
        flagged = [i for i, dof in enumerate(dofs)
                   if geometry.missing_dofs([dof])]
        if not flagged:
            return None
        missing = [dofs[i] for i in flagged]
        return Issue(
            name=name, kind='dofs-not-in-geometry',
            message=(f'{len(flagged)} of {len(dofs)} channels are not in '
                     f'geometry {geometry_name!r}: {_listed(missing)}'),
            missing_dofs=missing, sub_items=flagged)

    return None


def check_compatibility(objects: Mapping[str, Any],
                        geometry_name: str | None = None,
                        links: Sequence[Mapping[str, Any]] | None = None
                        ) -> Report:
    """Check every object in a test against the geometry it answers to.

    `objects` maps name -> object. An object linked into a group that
    holds a geometry is judged against *that* geometry — a FEM shape
    set beside its own FEM mesh is consistent, whichever geometry is
    active. Everything else is judged against `geometry_name` (the
    active geometry; without it the first geometry found). Returns a
    Report.
    """
    from .core.geometry import Geometry

    geometries = {name: obj for name, obj in objects.items()
                  if isinstance(obj, Geometry)}
    if geometry_name not in geometries:
        geometry_name = next(iter(geometries), None)

    def home(name: str) -> str | None:
        for group in links or []:
            if name in group.get('members', ()):
                linked = next((member for member in group['members']
                               if member in geometries), None)
                if linked is not None:
                    return linked
                break
        return geometry_name

    def companions(name: str) -> Mapping[str, Any]:
        # a modal object answers to its group's shape sets, or to any
        # in the project while it is not yet linked
        for group in links or []:
            if name in group.get('members', ()):
                return {member: objects[member] for member in group['members']
                        if member in objects}
        return objects

    report = Report(geometry_name=geometry_name)
    for name, obj in objects.items():
        if modal_object(obj) is not None:
            issue = check_modal(name, obj, companions(name))
        elif geometry_name is None:
            continue
        else:
            against = home(name)
            issue = check_object(name, obj, geometries.get(against), against)
        if issue is not None:
            report.issues[name] = issue
    return report
