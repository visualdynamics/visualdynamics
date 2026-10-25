"""Check that the objects in a test belong together.

Everything that carries DOFs (mode shapes, data records, channel table
rows) names nodes, and this module checks those nodes against the
geometry each object answers to. An object that names nodes the geometry
does not have is reported, and the app marks it red. It can still be
linked: a test often has drive points, control coordinates or virtual
channels the model does not draw. Mode shapes from one structure beside
the geometry of another are reported the same way, and the person
reading the indicator can tell the two cases apart.

An object answers to its object group's geometry when the group has
one, so a FEM shape set linked with its own FEM mesh is consistent
whichever geometry is active. Otherwise it answers to the active
geometry, so a test holding two models still gives one answer for each
object. Objects with no DOFs (a geometry itself) are never flagged. With
no geometry in the project only modal objects (below) are checked, so
importing data before its geometry does not mark everything red.

A data object whose DOFs are all modal coordinates (`M1` ... `Mn`, see
`validate.modal_coordinate`) names no node. It answers to a shape set,
the one it was transformed or computed through, and fits wherever there
is a set with at least as many modes as it names: a set in its object
group when it is in one, any set in the project otherwise. A time
history can hold both kinds of record, forces with the modal responses
computed from them appended: its physical records answer to the
geometry and its modal records to the shape sets, each by its own rule.

Not checked here:

- units, which have their own indicator
- direction: a geometry does not restrict directions, so 101RX+ is
  fine wherever node 101 exists
- whether two data objects can be compared with each other
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                                    # pragma: no cover
    from .core.geometry import Geometry

# A modal object answers to the shape sets in its own group (Brandon,
# 2026-09-04: the modal responses belong in the group with the shapes
# and the record they came from).

MAX_LISTED = 6  # DOFs named in a message before it summarizes


@dataclass
class Issue:
    """Why one object does not fit the geometry or shape set it answers to.

    `kind` is 'dofs-not-in-geometry', 'modes-not-in-set' (an object of
    modal coordinates alone), or 'modes-beyond-set' (modal records beside
    physical ones, the physical ones fitting). `missing_dofs` lists the
    DOFs at fault, and `sub_items` the indices of the records, shapes or
    channels that name them.
    """

    name: str
    kind: str
    message: str
    missing_dofs: list = field(default_factory=list)
    sub_items: list = field(default_factory=list)  # indices to mark


@dataclass
class Report:
    """The result of `check_compatibility`: an `Issue` for each object
    that does not fit, by name, and the geometry the rest were judged
    against."""

    geometry_name: str | None = None
    issues: dict = field(default_factory=dict)

    def is_compatible(self, name: str) -> bool:
        """Whether one object fits the geometry or shape set it answers to.

        Parameters
        ----------
        name : str
            The object to ask about.

        Returns
        -------
        bool
            True when no issue was found for it.
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
        """Whether one record, shape or channel within an object is
        flagged.

        Parameters
        ----------
        name : str
            The object to ask about.
        index : int
            Its index within the object.

        Returns
        -------
        bool
            True when that item names a DOF the geometry lacks, or a
            mode no shape set has.
        """
        issue = self.issues.get(name)
        return bool(issue) and index in issue.sub_items

    @property
    def incompatible_names(self) -> list[str]:
        """The names of the objects that have an issue, which the tree
        marks in red."""
        return list(self.issues)


def _listed(dofs: Sequence[str]) -> str:
    """'501X+, 501Y+ and 3 more' — enough to identify, short enough to read."""
    text = ', '.join(dofs[:MAX_LISTED])
    if len(dofs) > MAX_LISTED:
        text += f' and {len(dofs) - MAX_LISTED} more'
    return text


def _channel_named(table: Any, row: int) -> str:
    """A channel as the issue lists it: its DOF, or, before it has one,
    its number and what it still lacks ('channel 3 (no node yet)'). A
    channel with a node and no direction is listed only when the node is
    not on the geometry, so its node is named: 'channel 5 at node 9 (no
    direction yet)'."""
    # A channel added blank listed as '', so the message ended at its
    # colon with nothing named (Brandon, 2026-10-22).
    node, direction = str(table['node'][row]), str(table['direction'][row])
    number = int(table['channel'][row])
    if node and direction:
        return table.dof_strings()[row]
    if node:
        return f'channel {number} at node {node} (no direction yet)'
    lacking = 'node' if direction else 'node or direction'
    return f'channel {number} (no {lacking} yet)'


def modal_object(obj: Any) -> int | None:
    """The highest mode a data object's coordinates name, when every
    one of its DOFs is a modal coordinate, else None."""
    from .core.data import DataArray
    from .core.validate import modal_coordinate

    if not isinstance(obj, DataArray) or not obj.num_records:
        return None
    indices = [modal_coordinate(dof) for dof in obj.response_dof]
    if any(index is None for index in indices):
        return None
    return max(indices)


def modal_records(obj: Any) -> list[int]:
    """The records of a data object at modal coordinates; empty for
    anything else."""
    from .core.data import DataArray

    if not isinstance(obj, DataArray):
        return []
    return [i for i, dof in enumerate(obj.response_dof) if _index(dof)]


def check_modal(name: str, obj: Any, companions: Mapping[str, Any]
                ) -> Issue | None:
    """The Issue for a modal object against the shape sets among its
    companions, or None when one of them has every mode it names.

    The Issue's kind is 'modes-not-in-set', and its `sub_items` are the
    records that name a mode beyond the largest set. A mixed object's
    physical records name no mode and are never among them.
    """
    from .core.shapes import ShapeSet

    records = range(obj.num_records)
    highest = max(_index(obj.response_dof[i]) for i in records)
    sets = {other: value for other, value in companions.items()
            if isinstance(value, ShapeSet) and other != name}
    if any(value.num_shapes >= highest for value in sets.values()):
        return None
    beyond = [i for i in records
              if _index(obj.response_dof[i])
              > max((v.num_shapes for v in sets.values()), default=0)]
    if not sets:
        message = (f'modal coordinates up to M{highest}, and no shape set '
                   'to answer to — link the set they belong to')
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
    """Whether an issue is reason to refuse a link.

    Naming nodes the geometry lacks is not. It is reported in red by the
    compatibility indicator and the link goes ahead, because a test
    names places the model does not draw as a matter of course: drive
    points, control coordinates, computed locations, virtual channels.
    A controller's control object and a mode shape from another
    structure look the same from here, and only the person reading the
    indicator can tell them apart.

    What blocks is a modal object with no shape set that has as many
    modes as it names, since it has nothing to be drawn against in any
    geometry. A time history holding modal records beside physical ones
    does not block: its physical records are drawn, and the shortfall of
    its modal ones is reported ('modes-beyond-set'). (`Project.link` refuses two geometries in one group before
    this is called.)

    Parameters
    ----------
    issue : Issue or None
        An object's issue, from `check_object`.

    Returns
    -------
    bool
        True for a 'modes-not-in-set' issue, False for None or any
        other kind.
    """
    # Missing nodes stopped blocking after a real report failed on them
    # twice in two days (Brandon, 2026-09-20 and 2026-09-21), the second
    # time on an object whose channels were all virtual. That is why the
    # first attempt, refusing only when nothing overlapped, was not a
    # line that could be drawn either.
    return issue is not None and issue.kind == 'modes-not-in-set'


def check_object(name: str, obj: Any, geometry: Geometry | None,
                 geometry_name: str = 'the active geometry',
                 companions: Mapping[str, Any] | None = None) -> Issue | None:
    """The Issue for one object against a geometry, or None if it fits.

    A geometry, an object of a type that names no DOFs, and any
    non-modal object when `geometry` is None all return None.
    `companions` are the objects it is grouped with: a modal object is
    checked against the shape sets among them instead of the geometry.
    `geometry_name` is used only in the message.
    """
    from .core.channel_table import ChannelTable
    from .core.data import DataArray
    from .core.geometry import Geometry
    from .core.shapes import ShapeSet

    if isinstance(obj, Geometry):
        return None
    if modal_object(obj) is not None:
        return check_modal(name, obj, companions or {})
    modal = modal_records(obj)
    if geometry is None and not modal:
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
        # modal records beside physical ones (Brandon, 2026-10-25: forces
        # with their modal responses appended went red, every M record
        # read as a node the geometry lacked): each kind by its own rule
        skipped = set(modal)
        missing, flagged = [], []
        for i in range(obj.num_records):
            if geometry is None or i in skipped:
                continue
            dofs = [obj.response_dof[i]]
            if obj.reference_dof is not None:
                dofs.append(obj.reference_dof[i])
            gone = geometry.missing_dofs(dofs)
            if gone:
                flagged.append(i)
                missing.extend(d for d in gone if d not in missing)
        short = check_modal(name, obj, companions or {}) if modal else None
        if not flagged and short is None:
            return None
        said, kind = [], 'modes-beyond-set'
        if flagged:
            kind = 'dofs-not-in-geometry'
            said.append(f'{len(flagged)} of {obj.num_records} records '
                        f'reference DOFs not in geometry {geometry_name!r}: '
                        f'{_listed(missing)}')
        if short is not None:
            said.append(short.message)
            missing += short.missing_dofs
            flagged = sorted(flagged + short.sub_items)
        return Issue(name=name, kind=kind, message='; '.join(said),
                     missing_dofs=missing, sub_items=flagged)

    if isinstance(obj, ChannelTable):
        dofs = obj.dof_strings()
        flagged = [i for i, dof in enumerate(dofs)
                   if geometry.missing_dofs([dof])]
        if not flagged:
            return None
        named = [_channel_named(obj, i) for i in flagged]
        return Issue(
            name=name, kind='dofs-not-in-geometry',
            message=(f'{len(flagged)} of {len(dofs)} channels are not in '
                     f'geometry {geometry_name!r}: {_listed(named)}'),
            # only what is a DOF: a channel not yet placed names none
            missing_dofs=[dofs[i] for i in flagged
                          if str(obj['node'][i]) and str(obj['direction'][i])],
            sub_items=flagged)

    return None


def check_compatibility(objects: Mapping[str, Any],
                        geometry_name: str | None = None,
                        object_groups: Sequence[Mapping[str, Any]] | None = None
                        ) -> Report:
    """Check every object in a test against the geometry it answers to.

    An object linked into a group that holds a geometry is judged
    against that geometry, so a FEM shape set beside its own FEM mesh is
    consistent whichever geometry is active. Everything else is judged
    against `geometry_name`. A modal object is judged against the shape
    sets in its group, or against every set in the project when it is
    not linked.

    Parameters
    ----------
    objects : Mapping[str, Any]
        The test's objects, by name.
    geometry_name : str, optional
        The geometry everything not linked to one is judged against,
        normally the active geometry. When it is omitted or names no
        geometry in `objects`, the first geometry found is used.
    object_groups : sequence of Mapping, optional
        The project's object groups, which say which geometry each object
        answers to.

    Returns
    -------
    Report
        What is consistent and what is not, object by object.
    """
    from .core.geometry import Geometry

    geometries = {name: obj for name, obj in objects.items()
                  if isinstance(obj, Geometry)}
    if geometry_name not in geometries:
        geometry_name = next(iter(geometries), None)

    def home(name: str) -> str | None:
        for group in object_groups or []:
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
        for group in object_groups or []:
            if name in group.get('members', ()):
                return {member: objects[member] for member in group['members']
                        if member in objects}
        return objects

    report = Report(geometry_name=geometry_name)
    for name, obj in objects.items():
        if modal_object(obj) is not None:
            issue = check_modal(name, obj, companions(name))
        elif geometry_name is None and not modal_records(obj):
            continue
        else:
            against = home(name)
            issue = check_object(name, obj, geometries.get(against), against,
                                 companions=companions(name))
        if issue is not None:
            report.issues[name] = issue
    return report
