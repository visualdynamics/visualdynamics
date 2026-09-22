"""The project: every object a test holds, and how they belong together.

The GUI's window keeps exactly what a `Project` keeps — the named
objects, the link groups, which group is the Basis, the project type,
the active geometry — so a project built by clicking and one built in
a script are the same thing, and either saves to the same `.vdyn`
file. Scripts get the GUI's own verbs (`add`, `link`, `set_basis`,
`rename`, `save`) instead of hand-assembling dicts and lists.

`random_vibration_report` at the bottom is the whole of one of those
workflows in a single call — a Rattlesnake run in, an HTML report out —
and `random_vibration_run` is the project it does that from.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from .core.channel_table import ChannelTable
from .core.data import (
    DataArray,
    Frf,
    Psd,
    ShockSpecification,
    Specification,
    Spectrum,
    Srs,
    TimeHistory,
    TransientSpecification,
    _CoherenceBase,
)
from .core.geometry import Geometry
from .core.matches import MatchedModes
from .core.photos import Photos
from .core.report import Report
from .core.shapes import ShapeSet
from .core.sine import SineLevelSet, SineSweepSpecification
from .names import display_name

# A link group: the objects declared to belong together, and what the
# group is — 'Basis', or None for every other group. One field says it
# all: 'side' used to ride beside the role saying the same thing in a
# second vocabulary, and was folded in (Brandon, 2026-08-30); a 'FEM'
# role named the comparison pair until 2026-09-02, when it went too —
# a project knows its Basis, and everything else is other (Brandon).
LinkGroup = dict[str, Any]

# How a test reads: the structure, its photos and instrumentation, the
# measurements in the order they get derived, then what was identified
# from them. The tree shows this order and so does a Project's repr.
#
# Subclasses inherit their parent's place, which is what puts a
# Specification beside the PSD it bounds and a TransientSpecification
# beside the time data — a target reads next to the measurement it is a
# target for, not off in a section of its own.
#
# Srs earns its own entry because it subclasses nothing here. Left out,
# it and ShockSpecification fell off the end and sorted *after* Report,
# so a shock project listed its report above the two things the report
# is about. `test_no_object_type_sorts_after_the_report` is the guard.
#
# The three single-signal spectra come before the two-signal pair: a
# spectrum, a density and a shock spectrum are each derived from one
# record, where an FRF and a coherence need a reference as well.
TYPE_ORDER = (Geometry, Photos, ChannelTable, TimeHistory, Spectrum, Psd,
              Srs, SineSweepSpecification, SineLevelSet, Frf,
              _CoherenceBase, ShapeSet, MatchedModes,
              Report)


def type_rank(obj: Any) -> int:
    """Where an object sits in the canonical order."""
    return next((rank for rank, cls in enumerate(TYPE_ORDER)
                 if isinstance(obj, cls)), len(TYPE_ORDER))


def describe(obj: Any) -> str:
    """A few words about what an object holds — how big it is, not what
    it is called. Empty categories are left out rather than reported as
    zero."""
    if isinstance(obj, Geometry):
        counts = [(obj.num_nodes, 'node'),
                  (len(obj.traceline_conn), 'traceline'),
                  (len(obj.elem_conn), 'element')]
        if len(obj.cs_id) > 1:
            counts.insert(1, (len(obj.cs_id), 'coordinate system'))
    elif isinstance(obj, ShapeSet):
        span = (f', {obj.frequency.min():.4g}-{obj.frequency.max():.4g} Hz'
                if obj.num_shapes else '')
        note = (' — unscaled: no drive point measured, modal masses '
                'are not physical'
                if getattr(obj, 'unscaled', False) else '')
        return f'{obj.num_shapes} modes{span}{note}'
    elif isinstance(obj, SineLevelSet):
        return (f'{len(obj.levels)} tone{"s" * (len(obj.levels) != 1)}, '
                f'{len(obj.response_dof)} channel'
                f'{"s" * (len(obj.response_dof) != 1)}')
    elif isinstance(obj, SineSweepSpecification):
        lo = min(tone.frequency.min() for tone in obj.tones)
        hi = max(tone.frequency.max() for tone in obj.tones)
        return (f'{len(obj.tones)} tone{"s" * (len(obj.tones) != 1)}, '
                f'{lo:.4g}-{hi:.4g} Hz, {len(obj.response_dof)} control '
                f'channel{"s" * (len(obj.response_dof) != 1)}')
    elif isinstance(obj, DataArray):
        counts = [(obj.num_records, 'record'), (len(obj.abscissa), 'sample')]
    elif isinstance(obj, ChannelTable):
        counts = [(obj.num_channels, 'channel')]
    elif isinstance(obj, Photos):
        counts = [(obj.num_photos, 'photo')]
    elif isinstance(obj, MatchedModes):
        return (f'{obj.num_matches} matched pairs, '
                f'{obj.first} against {obj.second}')
    elif isinstance(obj, Report):
        counts = [(obj.num_blocks, 'block')]
    else:
        return ''
    return ', '.join(f'{n} {word}{"s" * (n != 1)}'
                     for n, word in counts if n)


def remap_links(links: Iterable[LinkGroup],
                mapping: dict[str, str],
                roles_taken: Iterable[str] = ()) -> list[LinkGroup]:
    """Link groups translated through a {old name: new name} mapping.

    Importing a project into one that already holds objects renames
    what clashes; its groups have to follow, or the structure the file
    carried is lost. A role already spoken for stays with the group
    that has it — the Basis is the project's, not the file's.
    """
    taken, out = set(roles_taken), []
    for group in links or ():
        members = [mapping[name] for name in group.get('members', ())
                   if name in mapping]
        if len(members) < 2:
            continue
        role = group.get('role')
        role = None if role in taken else role
        if role is not None:
            taken.add(role)
        out.append({'members': members, 'role': role})
    return out


def retarget(obj: Any, mapping: dict[str, str]) -> None:
    """Point an object's references at renamed objects.

    Objects that name others — matched modes name their two shape
    sets, a report's blocks name what they draw from — go stale the
    moment a name changes under them, and a stale binding is an
    unbound figure or a bracket that vanished.
    """
    follow = getattr(obj, 'rename_source', None)
    if callable(follow):                            # MatchedModes
        for old, new in mapping.items():
            follow(old, new)
    # the type, not the attribute: a Geometry grew a `blocks` of its own
    # (its element blocks), and duck-typing walked those instead — a
    # rename then died inside a Qt signal, where the traceback goes
    # nowhere and the rename simply does not happen
    if isinstance(obj, Report):
        for block in obj.blocks:
            for key in ('source', 'geometry', 'dofs_source', 'shapes'):
                if block.get(key) in mapping:
                    block[key] = mapping[block[key]]


# What each kind of object is called when it is reached by type
# rather than by name: (singular, plural, class). The singular gives
# you the one there is and says so when there are several; the plural
# is always a list. Both complete on tab.
KINDS: tuple[tuple[str, str, type], ...] = (
    ('geometry', 'geometries', Geometry),
    ('photos', 'photo_sets', Photos),
    ('channel_table', 'channel_tables', ChannelTable),
    ('time_history', 'time_histories', TimeHistory),
    ('transient_specification', 'transient_specifications',
     TransientSpecification),
    ('spectrum', 'spectra', Spectrum),
    ('specification', 'specifications', Specification),
    ('psd', 'psds', Psd),
    ('srs', 'srs_sets', Srs),
    ('shock_specification', 'shock_specifications', ShockSpecification),
    ('sine_sweep_specification', 'sine_sweep_specifications',
     SineSweepSpecification),
    ('sine_levels', 'sine_level_sets', SineLevelSet),
    ('frf', 'frfs', Frf),
    ('coherence', 'coherences', _CoherenceBase),
    ('shapes', 'shape_sets', ShapeSet),
    ('matches', 'match_sets', MatchedModes),
    ('report', 'reports', Report),
)

_SINGULAR = {singular: (plural, cls) for singular, plural, cls in KINDS}
_PLURAL = {plural: cls for _singular, plural, cls in KINDS}

# A kind never answers with what a more specific kind would claim — the
# same rule the report's symbolic bindings live by, arrived at the same
# way. A TransientSpecification *is* a TimeHistory, so in a transient
# project `project.time_history` found two and refused, though there is
# exactly one thing anyone asking means; `project.psds` handed back the
# specification alongside the measured PSDs, and every caller filtered
# it back out by hand. The specification kinds have their own names —
# asking for the broad kind means the thing that is *only* that kind.
_NARROWER = {cls: tuple(other for _s, _p, other in KINDS
                        if other is not cls and issubclass(other, cls))
             for _singular, _plural, cls in KINDS}


class Selection:
    """Objects reached by *what they are* instead of what they were named.

        project.geometry            # anywhere in the project
        project.basis.frf           # in the Basis group
        project.basis.psds[0]       # when you mean to choose

    A name is arbitrary — it came from whatever an importer or a user
    called it, and importing a second geometry renames nothing but
    makes 'Geometry' ambiguous. A type is not, so this is the path
    worth typing, and the only one an editor can complete.

    The singular gives the one object of that kind, and says which
    ones it found when there are several. The plural is always a
    list, so code written against one FRF keeps working when a second
    arrives.
    """

    if TYPE_CHECKING:
        # Declarations, not assignments: they give an editor real
        # completion and real types for names __getattr__ serves at
        # run time. Without them every one of these would be `Any`,
        # and type checking would quietly stop at the dot.
        geometry: Geometry
        geometries: list[Geometry]
        photos: Photos
        photo_sets: list[Photos]
        channel_table: ChannelTable
        channel_tables: list[ChannelTable]
        time_history: TimeHistory
        time_histories: list[TimeHistory]
        transient_specification: TransientSpecification
        transient_specifications: list[TransientSpecification]
        shock_specification: ShockSpecification
        shock_specifications: list[ShockSpecification]
        srs: Srs
        srs_sets: list[Srs]
        spectrum: Spectrum
        spectra: list[Spectrum]
        specification: Specification
        specifications: list[Specification]
        psd: Psd
        psds: list[Psd]
        frf: Frf
        frfs: list[Frf]
        coherence: _CoherenceBase
        coherences: list[_CoherenceBase]
        shapes: ShapeSet
        shape_sets: list[ShapeSet]
        matches: MatchedModes
        match_sets: list[MatchedModes]
        report: Report
        reports: list[Report]

    def __init__(self, project: Project, names: Iterable[str] | None = None,
                 label: str = 'project') -> None:
        self._project = project
        # None scopes to the whole project, and keeps doing so as
        # objects arrive: nothing here is a snapshot
        self._scope = None if names is None else list(names)
        self._label = label

    @property
    def names(self) -> list[str]:
        """The names in scope, in the order the tree shows them."""
        order = self._project.ordered_names()
        if self._scope is None:
            return order
        wanted = set(self._scope)
        return [name for name in order if name in wanted]

    @property
    def objects(self) -> list[Any]:
        return [self._project[name] for name in self.names]

    def of_type(self, cls: type) -> list[str]:
        """The names in scope holding objects of this class — and not
        of a class registered as its own kind beneath it (see _NARROWER:
        a transient specification is not the answer to `time_history`)."""
        narrower = _NARROWER.get(cls, ())
        return [name for name in self.names
                if isinstance(self._project[name], cls)
                and not isinstance(self._project[name], narrower)]

    def __getattr__(self, attribute: str) -> Any:
        # leading underscores are this object's own business, and
        # answering them here would recurse during copy or unpickling
        if attribute.startswith('_'):
            raise AttributeError(attribute)
        if attribute in _PLURAL:
            return [self._project[name]
                    for name in self.of_type(_PLURAL[attribute])]
        if attribute not in _SINGULAR:
            raise AttributeError(
                f'{attribute!r} is not a kind of object; '
                f'{self._label} holds {", ".join(self._kinds()) or "nothing"}')
        plural, cls = _SINGULAR[attribute]
        found = self.of_type(cls)
        if len(found) > 1:
            # a banded flavor answers only to a request that names it:
            # `.specification` beside the same requirement on octave
            # bands is the one on lines, as '@basis:Specification' is
            # in a report (core.report.resolve_binding) — the worked-up
            # random project holds both (2026-09-18)
            plain = [name for name in found
                     if getattr(self._project[name], 'bandwidth', None) is None]
            if len(plain) == 1:
                found = plain
        if len(found) == 1:
            return self._project[found[0]]
        if not found:
            raise AttributeError(
                f'no {attribute.replace("_", " ")} in {self._label}')
        listed = ', '.join(repr(name) for name in found)
        raise AttributeError(
            f'{len(found)} in {self._label} — {listed}. Use '
            f'.{plural}[i] to pick one, or the name directly.')

    def _kinds(self) -> list[str]:
        """The singular names that would resolve right now."""
        return [singular for singular, _plural, cls in KINDS
                if self.of_type(cls)]

    def __dir__(self) -> list[str]:
        # completion shows what this project actually holds, not every
        # kind visualdynamics knows about
        present = {singular: plural for singular, plural, cls in KINDS
                   if self.of_type(cls)}
        return sorted({*super().__dir__(), *present, *present.values()})

    def __bool__(self) -> bool:
        return bool(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.objects)

    def __contains__(self, item: Any) -> bool:
        return item in self.names or any(obj is item for obj in self.objects)

    def __repr__(self) -> str:
        if not self.names:
            return f'<{self._label}: empty>'
        width = max(len(name) for name in self.names)
        lines = [f'{self._label}:']
        for name in self.names:
            obj = self._project[name]
            lines.append(f'  {name:<{width}}  {type(obj).__name__:<13} '
                         f'{describe(obj)}'.rstrip())
        return '\n'.join(lines)


class Project(dict):
    """Every object in one test, by name, plus the structure around them.

    A Project *is* the name-to-object mapping, so `project['FRF']`,
    `list(project)` and `project.items()` read the way the tree reads.

    It is also what the desktop app holds: the window's `objects`,
    `links`, `project_type` and `active_geometry` are properties over one
    of these, and its buttons call the verbs below. A project built by
    clicking and one built by calling are the same object, and open in
    each other.

    Objects are usually reached **by type** rather than by name —
    `project.geometry`, `project.basis.frf`, `project.other.shapes`. The
    singular gives the one there is and says so when several qualify; the
    plural is always a list.

    Attributes:
        links: The groups objects have been declared to belong to, as
            `{'members': [...], 'role': 'Basis' | None}`.
            Association is explicit here, never inferred from names.
        project_type: What kind of test this is — 'Modal Test', 'Random
            Vibration', 'Shock', 'Transient' — which decides the report
            template and the skeleton of slots the tree shows.
        active_geometry: The name of the geometry data is drawn on when
            nothing says otherwise.
        name: What the project is called, which is what a saved `.vdyn`
            and a rendered report are titled.
    """

    if TYPE_CHECKING:
        # the same declarations Selection carries, so `project.frf`
        # completes and types at the top level too
        geometry: Geometry
        geometries: list[Geometry]
        photos: Photos
        photo_sets: list[Photos]
        channel_table: ChannelTable
        channel_tables: list[ChannelTable]
        time_history: TimeHistory
        time_histories: list[TimeHistory]
        transient_specification: TransientSpecification
        transient_specifications: list[TransientSpecification]
        shock_specification: ShockSpecification
        shock_specifications: list[ShockSpecification]
        srs: Srs
        srs_sets: list[Srs]
        spectrum: Spectrum
        spectra: list[Spectrum]
        specification: Specification
        specifications: list[Specification]
        psd: Psd
        psds: list[Psd]
        frf: Frf
        frfs: list[Frf]
        coherence: _CoherenceBase
        coherences: list[_CoherenceBase]
        shapes: ShapeSet
        shape_sets: list[ShapeSet]
        matches: MatchedModes
        match_sets: list[MatchedModes]
        report: Report
        reports: list[Report]

    def __init__(self, name: str = 'Project',
                 objects: dict[str, Any] | None = None,
                 active_geometry: str | None = None,
                 project_type: str | None = None,
                 links: Iterable[LinkGroup] | None = None,
                 provenance: dict[str, dict[str, Any]] | None = None
                 ) -> None:
        super().__init__(objects or {})
        self.name: str = str(name)
        self.active_geometry: str | None = active_geometry
        self.project_type: str | None = project_type
        # [{'members': [...], 'role': 'Basis' | None}] — a loaded file
        # may still carry a 'side' key from before the two vocabularies
        # merged, or a 'FEM' role from before that role was retired;
        # both read as the Basis where they meant it and as nothing
        # otherwise, and are never stored again
        self.links: list[dict[str, Any]] = [
            {'members': list(group['members']),
             'role': 'Basis' if (group.get('role') == 'Basis'
                                 or group.get('side') == 'experimental'
                                 and not group.get('role')) else None}
            for group in (links or [])]
        #: how each derived object was computed — {name: {'verb',
        #: 'source', 'params', 'state'}} with 'state' the fingerprint
        #: of the analysis settings read at compute time. Staleness is
        #: the recorded state disagreeing with the source's current
        #: one; a mismatch is exactly the recompute the refresh badge
        #: offers (Brandon, 2026-08-23: explicit, never automatic —
        #: a report must not rewrite itself).
        self.provenance: dict[str, dict[str, Any]] = dict(provenance or {})
        #: this sitting's acts, each a runnable line of Python — what
        #: the console shows and `session_script` exports (Brandon,
        #: 2026-08-30). In memory only, and deliberately so: the
        #: journal is a record of *this session*, where the `.vdyn`
        #: file's provenance records tell the durable story. Guarded
        #: by `_journal_depth` so a verb calling other verbs — merge
        #: adds and links, refresh recomputes — records once, as the
        #: line the user could have typed.
        self.journal: list[str] = [
            f'project = visualdynamics.Project({str(name)!r})']
        self._journal_depth: int = 0

    def __repr__(self) -> str:
        """The project tree, as the GUI shows it.

        Typing the name of a project at a prompt should answer the
        question the tree answers: what is in here, how does it group,
        and how big is each piece.
        """
        head = f'Project: {self.name}'
        if self.project_type:
            head += f'  [{self.project_type}]'
        if not self:
            return head + '\n  (empty)'
        width = max(len(name) for name in self)
        lines = [head]
        for group, members in self.grouped_names():
            if group is not None:
                role = group['role'] or 'linked'
                lines.append(f'  {role}')
            for name in members:
                obj = self[name]
                mark = ' *' if name == self.active_geometry else '  '
                lines.append(
                    f'{"    " if group is not None else "  "}'
                    f'{name:<{width}}{mark} {type(obj).__name__:<13} '
                    f'{describe(obj)}'.rstrip())
        if self.active_geometry:
            lines.append('  * the active geometry')
        return '\n'.join(lines)

    def grouped_names(self) -> list[tuple[LinkGroup | None, list[str]]]:
        """[(group or None, [names])] in the order the tree shows them:
        the Basis group first, then the other link groups, then what is
        unlinked — each in the canonical type order, and objects of one
        type in the order they arrived."""
        def ordered(names: Iterable[str]) -> list[str]:
            return sorted(names, key=lambda name: type_rank(self[name]))

        groups = sorted(self.links, key=lambda g: g['role'] != 'Basis')
        out, linked = [], set()
        for group in groups:
            members = [name for name in group['members'] if name in self]
            if members:
                out.append((group, ordered(members)))
                linked |= set(members)
        loose = ordered(name for name in self if name not in linked)
        return out + ([(None, loose)] if loose else [])

    def ordered_names(self) -> list[str]:
        """Every name, flat, in the order the tree shows them."""
        return [name for _group, members in self.grouped_names()
                for name in members]

    # ---- objects -----------------------------------------------------------

    def add(self, name: str, obj: Any) -> str:
        """Add an object under a unique name; returns the name used.

        A clash is numbered rather than refused or overwritten, exactly
        as importing twice does in the GUI. The first geometry added
        becomes the active one.

        Parameters
        ----------
        name : str
            What to call it. A clash gets a numbered suffix.
        obj : object
            Any object the project can hold.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix.
        """
        unique, n = str(name), 1
        while unique in self:
            n += 1
            unique = f'{name} ({n})'
        self[unique] = obj
        if self.active_geometry is None and isinstance(obj, Geometry):
            self.active_geometry = unique
        return unique

    def duplicate(self, *names: Any) -> list[str]:
        """Copies of objects, added beside them (Copy, then Paste, in
        the tree): each under its own name with ' copy', numbered when
        that is taken. Returns the names added.

        Independent objects, not views: a copy's arrays are its own,
        so editing one leaves the other as it was. Links and
        provenance stay with the originals — a copy is a fresh object
        that happens to hold the same numbers, and what it is for is
        the user's to say.

        Parameters
        ----------
        *names : str or object
            The objects to copy, by name or as the objects.

        Returns
        -------
        list of str
            The names the copies were added under, in order.
        """
        import copy

        added = []
        for name in names:
            name = self.name_of(name)
            added.append(self.add(f'{name} copy', copy.deepcopy(self[name])))
        return added

    def import_file(self, path: str | os.PathLike,
                    **options: Any) -> list[str]:
        """Import a file into this project; returns the names added.

        Anything visualdynamics reads: a geometry, a Rattlesnake run, or a whole
        saved project. A project brings its structure with it — its
        link groups follow the objects even when a name clash renamed
        them — and, into an empty project, its name, type and active
        geometry too. Foreign readers' keys become readable names
        ('Modal_frf' is an FRF), the way the tree spells them.

        A file that knows what kind of test it was says so: a
        controller's own save records which environment drove the run,
        and adopting it here settles the project type in a script the
        same way importing one settles it in the window.

        `options` pass through to the format's reader — an exodus
        file's `steps='time'` and `nodes=[...]`, a geometry's
        `length_unit='m'` — so a script can declare what the
        window asks about in a dialog.

        Parameters
        ----------
        path : str or os.PathLike
            The file to read. The importer is chosen by content
            and extension.
        **options
            Passed through to the importer.

        Returns
        -------
        list of str
            The names of every object added, in the order added.
        """
        from .io import import_file as read
        from .io import project_type_of

        was_empty = not self
        result = read(str(path), **options)
        self.project_type = project_type_of(str(path)) or self.project_type
        if isinstance(result, Project):
            mapping = {name: self.add(name, obj)
                       for name, obj in result.items()}
            if any(old != new for old, new in mapping.items()):
                # a clash renamed something: the references the
                # imported objects carry follow it
                for obj in result.values():
                    retarget(obj, mapping)
            self.links += remap_links(
                result.links, mapping,
                {group['role'] for group in self.links if group['role']})
            if was_empty:
                self.name = result.name or self.name
                self.project_type = result.project_type
                if result.active_geometry in mapping:
                    self.active_geometry = mapping[result.active_geometry]
            return list(mapping.values())
        if isinstance(result, dict):
            # a reader's keys are for code; the name is what the object
            # *is*, and the key stays available on the result
            added = [self.add(display_name(type(obj).__name__), obj)
                     for obj in result.values()]
            # a controller's run arrives as a channel table beside the
            # data it describes: one file, so one group (Brandon,
            # 2026-09-06 — linked, and otherwise two independent
            # objects: a coordinate corrected on one is not corrected on
            # the other)
            if len(added) > 1 and any(isinstance(obj, ChannelTable)
                                      for obj in result.values()):
                self.link(*added)
            return added
        return [self.add(display_name(type(result).__name__), result)]

    def remove(self, *names: str) -> None:
        """Delete objects, pruning them out of every link group.

        Parameters
        ----------
        *names : str
            The objects to act on, by name.

        Returns
        -------
        None
        """
        for name in names:
            self.pop(name, None)
            if self.active_geometry == name:
                self.active_geometry = next(
                    (n for n, obj in self.items()
                     if isinstance(obj, Geometry)), None)
        self._prune_links()

    def rename(self, old: str, new: str) -> str:
        """Rename an object; every reference to it follows.

        Link groups, matched-modes sets and report block bindings all
        name their objects, and a rename that left any of them pointing
        at the old name would strand a figure or a bracket.

        Parameters
        ----------
        old : str
            The current name.
        new : str
            The name to give it.

        Returns
        -------
        str
            The name actually used, which may carry a suffix.
        """
        if old not in self:
            raise KeyError(f'no object named {old!r}')
        if new in self:
            raise ValueError(f'name {new!r} is already in use')
        if not str(new).strip():
            raise ValueError('name cannot be empty')
        # rebuilt rather than popped, so the order the tree shows holds
        items = [(new if name == old else name, obj)
                 for name, obj in self.items()]
        self.clear()
        self.update(items)
        if self.active_geometry == old:
            self.active_geometry = new
        for group in self.links:
            group['members'] = [new if member == old else member
                                for member in group['members']]
        for obj in self.values():
            retarget(obj, {old: new})
        if old in self.provenance:
            self.provenance[new] = self.provenance.pop(old)
        for record in self.provenance.values():
            if record.get('source') == old:
                record['source'] = new
        return new

    def rename_dof(self, source: Any, old: str, new: str,
                   quantity: str | None = None) -> list[str]:
        """Correct a channel's coordinate on an object and on everything
        derived from it (double-click a row or reference column of the
        grid and type).

        The channel, not the point: a force labeled at the wrong node
        moves without taking the accelerometer at that node with it
        (Brandon, 2026-09-06 — the other is changed explicitly if it
        should be). The spectra computed from a time history inherited
        its channels, so one found mislabeled is mislabeled in every
        one of them; the correction follows the derivation chain rather
        than leaving each derived object to be fixed by hand or
        recomputed. An object downstream that does not carry the
        channel is left alone.

        Parameters
        ----------
        source : str or object
            The object whose grid row or column was edited.
        old : str
            The coordinate as it is, '101Z+'.
        new : str
            The coordinate to give it, normalized the way every DOF is.
        quantity : str, optional
            Which channel at `old` — 'acceleration', 'force' …; every
            channel at the coordinate when omitted.

        Returns
        -------
        list of str
            The names of the objects changed, the source first.
        """
        name = self.name_of(source)
        obj = self[name]
        if not hasattr(obj, 'rename_dof'):
            raise TypeError(f'{name} has no coordinates to rename')
        obj.rename_dof(old, new, quantity)
        changed = [name]
        for other in list(self.provenance):
            if other not in self or not self._descends_from(other, name):
                continue
            derived = self[other]
            if not hasattr(derived, 'rename_dof'):
                continue
            try:
                if derived.rename_dof(old, new, quantity):
                    changed.append(other)
            except ValueError:
                # the channel is not on this one — a modal
                # transformation carries none of the source's
                continue
        return changed

    def _descends_from(self, name: str, ancestor: str) -> bool:
        """Whether `name` was derived, through any number of steps,
        from `ancestor`."""
        seen = set()
        while name in self.provenance and name not in seen:
            seen.add(name)
            name = self.provenance[name].get('source')
            if name == ancestor:
                return True
        return False

    # ---- links -------------------------------------------------------------

    def link(self, *names: str, role: str | None = None) -> list[str]:
        """Declare objects part of one group, merging any they are in.

        A group holds at most one geometry — its members are read
        against it — and a member naming nodes that geometry lacks is
        refused, because the link would be a claim that is not true.
        Returns the group's members.

        Parameters
        ----------
        *names : str
            The objects to act on, by name.
        role : str, optional
            The role to give the group — 'Basis', or None.

        Returns
        -------
        list of str
            The group's members after linking.
        """
        names = [str(name) for name in names]
        if len(names) < 2:
            raise ValueError('linking takes two or more objects')
        missing = [name for name in names if name not in self]
        if missing:
            raise KeyError(f'no object named {missing[0]!r}')
        # Existing members keep their places; only genuinely new names
        # append. The first version seeded `merged` with `names` and
        # prepended what each touched group already held — which moved
        # a member being re-linked to the back of its own group, and
        # the tree (which keeps arrival order within a type, read from
        # this list) showed a time history jumping down its group the
        # moment FRFs were computed from it (Brandon, 2026-08-29).
        merged, kept = [], []
        for group in self.links:
            if any(name in group['members'] for name in names):
                merged += [name for name in group['members']
                           if name not in merged]
                role = role or group['role']
            else:
                kept.append(group)
        merged += [name for name in names if name not in merged]
        geometries = [name for name in merged
                      if isinstance(self.get(name), Geometry)]
        if len(geometries) > 1:
            raise ValueError('a link holds one geometry — '
                             f'{" and ".join(geometries)} cannot share')
        from .compatibility import blocks_a_link, check_object, modal_object

        # a modal member answers to a shape set in the group, whether
        # or not the group holds a geometry; everything else answers
        # to the geometry when there is one
        companions = {member: self[member] for member in merged}
        against = (self[geometries[0]], geometries[0]) if geometries else None
        for member in merged:
            if against is None and modal_object(self[member]) is None:
                continue
            issue = check_object(
                member, self[member],
                None if against is None else against[0],
                'the active geometry' if against is None else against[1],
                companions=companions)
            # a few points the geometry does not draw are a test's
            # virtual channels, not a different structure: the
            # indicator names them and the link stands
            if blocks_a_link(issue):
                raise ValueError(f'cannot link: {issue.message}')
        self.links = kept + [{'members': merged, 'role': role}]
        if role is not None:
            self.set_role(merged[0], role)
        return merged

    def unlink(self, *names: str) -> None:
        """Take objects out of their groups; a group of one dissolves.

        Parameters
        ----------
        *names : str
            The objects to act on, by name.

        Returns
        -------
        None
        """
        names = {str(name) for name in names}
        self.links = [
            {'members': kept, 'role': group['role']}
            for group, kept in
            ((group, [member for member in group['members']
                      if member not in names])
             for group in self.links)
            # A *named* group holds one object quite happily: the FEM
            # group of a modal test starts as a lone geometry, and
            # dissolving it is why the group could never be built up one
            # drag at a time.
            if len(kept) > 1 or (kept and group['role'])]

    def relink(self, name: Any, target: Any = None) -> list[str]:
        """Move one object into the group holding `target`.

        `link` *merges* the groups its arguments are in, which is right
        for declaring two things related and wrong for moving one thing
        between groups — linking a geometry to the other side would pull
        its whole group across with it. This takes the object out first,
        so only it moves; `target` of None just takes it out.

        The group it lands in keeps its role, so dropping something into
        the Basis makes it part of the Basis rather than dissolving it.
        Returns the members of the group it ends up in, empty if none.

        Parameters
        ----------
        name : str or object
            The object to move.
        target : str or object, optional
            An object whose group it should join. None removes it
            from its current group.

        Returns
        -------
        list of str
            The group's members afterwards.
        """
        name = self.name_of(name)
        if target is None:
            self.unlink(name)
            return []
        target = self.name_of(target)
        if target == name:
            raise ValueError('an object is already in its own group')
        members = self.group_of(target) or [target]
        if name in members:
            return list(members)
        role = self.role_of(target)
        # `link` refuses a second geometry or a member the group's
        # geometry cannot carry, and by then the object has already left
        # where it was: a refused move must leave the links untouched
        # rather than half applied.
        before = [dict(group) for group in self.links]
        self.unlink(name)
        try:
            # `name` last: it joins the group it landed in, and a group
            # reads in the order its members arrived
            return self.link(*members, name, role=role)
        except (ValueError, KeyError):
            self.links = before
            raise

    def group_of(self, name: Any) -> list[str] | None:
        """The members linked with `name`, or None.

        Parameters
        ----------
        name : str or object
            The object to look up.

        Returns
        -------
        list of str, or None
            The names sharing its link group, or None when it is in
            no group.
        """
        group = self._group(self.name_of(name))
        return None if group is None else list(group['members'])

    def role_of(self, name: Any) -> str | None:
        """'Basis', or None for an unroled or unlinked object.

        Parameters
        ----------
        name : str or object
            The object to look up.

        Returns
        -------
        str or None
            Its link group's role, or None if it has none.
        """
        group = self._group(self.name_of(name))
        return None if group is None else group['role']

    def placed(self) -> dict[str, list[str]]:
        """{role: [members]} — which objects are in each *named* group.

        Nothing is guessed here. An object nobody has placed is in no
        named group, which is what makes the tree's gray slots mean
        anything: a slot is filled by an object in its own group, so a
        modal test whose only geometry is the model's still shows a
        slot for the measured one.
        """
        out: dict[str, list[str]] = {}
        for group in self.links:
            if group['role']:
                out.setdefault(group['role'],
                               []).extend(group['members'])
        return out

    def role_group(self, role: str) -> LinkGroup | None:
        """The link group carrying a role, or None.

        Parameters
        ----------
        role : str
            Which named group to fetch — 'Basis' is the only name.

        Returns
        -------
        LinkGroup or None
            That group, or None if unset.
        """
        return next((group for group in self.links
                     if group['role'] == role), None)

    def place(self, name: Any, role: str) -> list[str]:
        """Put one object into the named group, making it if need be.

        Unlike an ordinary link this takes a single object, because a
        named group is a *declaration* rather than an observed relation
        — and needing two members before the group can exist at all is
        what made it impossible to move a wrongly-sorted pair across
        one at a time. Returns the group's members.

        Parameters
        ----------
        name : str or object
            The object being placed.
        role : str or None
            'Basis', or None for the *other* group — the one group
            that is not the Basis, made if there is none yet. With
            several groups besides the Basis there is no one other,
            and this refuses: relink onto a member of the one meant.

        Returns
        -------
        list of str
            The group's members.
        """
        name = self.name_of(name)
        if role is None:
            others = [g for g in self.links if g['role'] != 'Basis'
                      and name not in g['members']]
            mine = self._group(name)
            if mine is not None and mine['role'] != 'Basis':
                return list(mine['members'])
            if len(others) > 1:
                raise ValueError(
                    f'{len(others)} groups besides the Basis — drop '
                    'onto a member of the one meant')
            before = [dict(existing) for existing in self.links]
            self.unlink(name)
            try:
                if not others:
                    self.links = self.links + [
                        {'members': [name], 'role': None}]
                    return [name]
                return self.link(*others[0]['members'], name)
            except (ValueError, KeyError):
                self.links = before
                raise
        group = self.role_group(role)
        if group is not None and name in group['members']:
            return list(group['members'])
        before = [dict(existing) for existing in self.links]
        self.unlink(name)
        group = self.role_group(role)
        try:
            if group is None:
                self.links = self.links + [
                    {'members': [name], 'role': None}]
                self.set_role(name, role)
                return [name]
            return self.link(*group['members'], name, role=role)
        except (ValueError, KeyError):
            self.links = before
            raise

    def set_role(self, name: str, role: str | None) -> None:
        """Name what a group is. The Basis is unique: taking the role
        takes it from whatever group held it.

        Parameters
        ----------
        name : str
            The object whose group is being labeled.
        role : str or None
            The role, or None to clear it.

        Returns
        -------
        None
        """
        group = self._group(name)
        if group is None:
            raise ValueError(f'{name!r} is not in a link group')
        if role is not None:
            for other in self.links:
                if other is not group and other['role'] == role:
                    other['role'] = None
        group['role'] = role

    def set_basis(self, *names: Any) -> list[str]:
        """Declare the Basis of comparisons: the group whose DOFs
        comparisons happen in, whose modes are the MAC rows and the
        frequency-error baseline. One name marks that object's group;
        several link them first.

        Parameters
        ----------
        *names : str or object
            The objects that form the basis set.

        Returns
        -------
        list of str
            The basis group's members.
        """
        names = tuple(self.name_of(name) for name in names)
        if len(names) > 1:
            self.link(*names, role='Basis')
        elif names:
            self.set_role(names[0], 'Basis')
        return self.basis.names

    @property
    def basis(self) -> Selection:
        """The Basis group, reached by type: `project.basis.frf`.

        Empty (and falsy) when no group has been declared the Basis,
        so `if project.basis:` still asks the question it reads as.
        Its member names are `project.basis.names`.
        """
        # [] not None: None is the whole project, and a project with
        # no Basis declared has an *empty* one, not every object in it
        members = next((group['members'] for group in self.links
                        if group['role'] == 'Basis'), [])
        return Selection(self, members, 'the Basis group')

    @property
    def groups(self) -> list[Selection]:
        """Every link group, the Basis first, each reached by type."""
        return [Selection(self, group['members'],
                          f'the {group["role"] or "linked"} group')
                for group in sorted(self.links,
                                    key=lambda g: g['role'] != 'Basis')]

    @property
    def other(self) -> Selection:
        """The one link group that is not the Basis — the model side of
        a correlation, usually. Says so when there are several."""
        others = [group for group in self.links if group['role'] != 'Basis']
        if len(others) == 1:
            return Selection(self, others[0]['members'], 'the other group')
        if not others:
            return Selection(self, [], 'the other group')
        raise AttributeError(
            f'{len(others)} groups besides the Basis — use .groups[i]')

    def __getattr__(self, attribute: str) -> Any:
        """Objects reached by type across the whole project.

        Only reached for names the class does not define, so a verb
        never has to compete with a kind of object.
        """
        if attribute.startswith('_'):
            raise AttributeError(attribute)
        # A property that raises AttributeError lands here too — Python
        # cannot tell "no such attribute" from "the attribute refused" —
        # and its own message is the useful one, so run it again and let
        # that error out rather than reporting a missing kind of object.
        descriptor = getattr(type(self), attribute, None)
        if isinstance(descriptor, property):
            return descriptor.fget(self)
        return getattr(Selection(self), attribute)

    def __dir__(self) -> list[str]:
        return sorted({*super().__dir__(), *dir(Selection(self))})

    def geometry_for(self, name: Any) -> tuple[str, Geometry] | None:
        """(name, geometry) the object answers to: its group's, else
        the active one. What it is drawn on, and checked against.

        Parameters
        ----------
        name : str or object
            The object whose geometry is wanted.

        Returns
        -------
        tuple of (str, Geometry), or None
            The geometry's name and the geometry itself, or None when
            the object is not linked to one.
        """
        name = self.name_of(name)
        for member in self.group_of(name) or ():
            if isinstance(self.get(member), Geometry):
                return member, self[member]
        active = self.active_geometry
        if active is not None and active in self:
            return active, self[active]
        return None

    def _group(self, name: str) -> LinkGroup | None:
        return next((group for group in self.links
                     if name in group['members']), None)

    def absorb_links(self, groups: Iterable[LinkGroup]) -> None:
        """Take on the link groups of a project being imported.

        The arriving objects have *already* been placed in named
        groups by the project type's rules — one at a time, as each arrived,
        which is a guess made without the file's own structure to go
        on. The file knows better, so it goes last and `_prune_links`
        lets it win.

        Parameters
        ----------
        groups : iterable of LinkGroup
            Link groups from another project, merged into this one's.

        Returns
        -------
        None
        """
        self.links = list(self.links) + [dict(group) for group in groups]
        self._prune_links()

    def _prune_links(self) -> None:
        """Members that no longer exist go, and so do groups left with
        nothing in them.

        **An object belongs to one group**, and this is where that is
        made true. Everything downstream assumes it: `group_of` answers
        with the first match, a bracket is painted over a contiguous
        run of rows, and the tree orders its rows by walking the groups
        — an object named twice put its row in that order twice, ran
        the positions past the end of the tree, and `insertChild` past
        the end silently drops a child that has already been taken out.
        Seven rows vanished from a project that still held all
        forty-five of its objects.

        **The later group wins.** The order is the order things were
        declared in, and the last word is the most informed one: an
        imported file's own structure arrives after the guess the type
        rules made about its objects one at a time. A group emptied
        this way still goes if it carries no role.
        """
        seen: set[str] = set()
        groups = []
        for group in reversed(self.links):
            kept = [member for member in group['members']
                    if member in self and member not in seen]
            seen.update(kept)
            groups.append({'members': kept, 'role': group['role']})
        self.links = [group for group in reversed(groups)
                      if len(group['members']) > 1
                      or (group['members'] and group['role'])]

    # ---- the workflow ------------------------------------------------------
    #
    # One verb per thing the GUI's buttons do. Each adds its result
    # under a name, links it to what it came from — a derived object
    # belongs with its source, and saying so is not a guess — and
    # returns the name, so a script reads as a sequence of steps.

    def verbs(self, source: Any = None) -> list[tuple[str, str]]:
        """The processing verbs that apply to an object, each with its
        one-line reading — how a script writer discovers what can be
        done with what (Brandon, 2026-08-31: a flat method list says
        nothing about what `integrate` is *for*).

        The applicability table is the same one the window's bar
        reads, so the two surfaces cannot disagree; the summaries
        are the first paragraph of each verb's own docstring, so this
        and the API reference cannot disagree either.

        Parameters
        ----------
        source : str or object, optional
            The object to ask about — a name looks it up here, an
            object answers for itself whether or not it has been
            added (what applies to a result is knowable before it is
            kept). Omitted, every processing verb is listed.

        Returns
        -------
        list of (str, str)
            ``(verb, summary)`` pairs, in the order the verbs are
            declared. Call the verb as ``getattr(project, verb)``, or
            just read the list and type the name.
        """
        import inspect

        obj = self[source] if isinstance(source, str) else source
        out = []
        for verb, applies in _VERB_APPLIES:
            if obj is not None and not applies(self, obj):
                continue
            doc = inspect.cleandoc(getattr(type(self), verb).__doc__)
            out.append((verb, ' '.join(doc.split('\n\n', 1)[0].split())))
        return out

    def selection_verbs(self, *names: Any) -> list[tuple[str, str]]:
        """The processing verbs a *selection* can act on, each with its
        one-line reading — what the window's bar offers (Brandon,
        2026-09-04: every act on the bar, none behind a menu).

        One object: its own verbs, less the ones that need a partner
        (`transform` needs a shape set beside the record). Several: the
        partner verbs that apply to exactly that combination — a
        record and a shape set transform or expand, two shape sets on
        two geometries project, siblings of one type merge — and none
        of the verbs that apply to one of them alone. The same table
        `verbs` reads, so the bar and the API cannot disagree.

        Parameters
        ----------
        *names : str or object
            The selection, by name or as the objects themselves.

        Returns
        -------
        list of (str, str)
            ``(verb, summary)`` pairs, in the order the verbs are
            declared.
        """
        import inspect

        objects = [self[self.name_of(name)] for name in names]
        if not objects:
            return []
        if len(objects) == 1:
            return [(verb, summary) for verb, summary in self.verbs(objects[0])
                    if verb not in PARTNER_VERBS]
        out = []
        for verb, applies in _SELECTION_APPLIES:
            if applies(self, objects):
                doc = inspect.cleandoc(getattr(type(self), verb).__doc__)
                out.append((verb, ' '.join(doc.split('\n\n', 1)[0].split())))
        return out

    def compute_spectra(self, source: Any) -> str:
        """Spectra from a time history's averages (the averaging
        view's Compute Spectra).

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself;
            `name_of` resolves either.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix.
        """
        source = self.name_of(source)
        return self._derive(source, self[source].compute_spectra(),
                            f'{source} Spectra', recipe=('compute_spectra', {}))

    def compute_psds(self, source: Any) -> str:
        """PSDs from a time history's averages (Compute PSDs).

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself;
            `name_of` resolves either.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix.
        """
        source = self.name_of(source)
        return self._derive(source, self[source].compute_psds(),
                            f'{source} PSDs', recipe=('compute_psds', {}))

    def compute_octave(self, source: Any, per_octave: int | None = None
                       ) -> str:
        """A spectrum integrated onto proportional bands (Compute
        Octave Bands) — the same power, arranged the way it is read;
        a specification with its warning and abort limits banded the
        same way.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself;
            `name_of` resolves either.
        per_octave : int, optional
            Bands per octave. Defaults to `core.octave.PER_OCTAVE`.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix.
        """
        from .core.octave import PER_OCTAVE

        per_octave = PER_OCTAVE if per_octave is None else int(per_octave)
        source = self.name_of(source)
        return self._derive(
            source, self[source].to_octave(per_octave),
            f'{source} 1/{per_octave} Octave',
            recipe=('compute_octave', {'per_octave': per_octave}))

    def compute_frfs(self, source: Any, method: str = 'Hv') -> str:
        """Frequency response functions from a time history (Compute
        FRFs) — one per response and drive, over the frames a PSD uses.

        `method` is 'Hv', 'H1' or 'H2': where the noise is assumed to
        be, which is the one thing the three estimators disagree about.
        The name goes on the object, since two FRF sets from one history
        differ in nothing else a reader can see.

        The frames are detected if the history has none, the same way
        the coherence does it, so the two describe one measurement.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself;
            `name_of` resolves either.
        method : str, default 'Hv'
            Which estimator: 'Hv', 'H1' or 'H2' — where the noise is
            assumed to be, which is the one thing they disagree about.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix.
        """
        source = self.name_of(source)
        history = self[source]
        if history.averaging is None:
            history.averaging = history.suggest_averaging()
        return self._derive(source, history.compute_frfs(method=method),
                            f'{source} {method} FRFs',
                            recipe=('compute_frfs', {'method': method}))

    def compute_multiple_coherence(self, source: Any) -> str:
        """Multiple coherence from a time history (Compute Multiple
        Coherence) — how much of each response the drives account for.

        Averaged over frames, and the frames are detected if the history
        has none. Computed over the whole selection instead, the
        reference set fits every response exactly and the answer is 1.0
        at every line — a number that says nothing, arrived at honestly.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself;
            `name_of` resolves either.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix.
        """
        source = self.name_of(source)
        history = self[source]
        if history.averaging is None:
            history.averaging = history.suggest_averaging()
        return self._derive(source, history.compute_multiple_coherence(),
                            f'{source} Multiple Coherence',
                            recipe=('compute_multiple_coherence', {}))

    def compute_srs(self, source: Any, *,
                    per_octave: int | None = None,
                    q: float | None = None,
                    kind: str = 'maximax') -> str:
        """Shock response spectra from a time history's shocks (Compute
        SRS) — one curve per channel per event.

        The events are the history's own — detected only when there is
        nothing else to say where they are, so this answers rather than
        asking the caller to go and find them.

        Detection is the last resort and not the first. A record being
        read as frames already says where its events are: a transient
        run's playings *are* its averaging, and set loose on one the
        detector answered with thirty-one events where there were six.
        A target is one playing by definition and gets no detector at
        all — it was being cut into three.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself;
            `name_of` resolves either.
        per_octave : int, optional
            Natural-frequency lines per octave. Defaults to the
            module's convention (12).
        q : float, optional
            The oscillator amplification, Q = 1/(2ζ); 10 — 5%
            damping, the shock-test convention — when omitted.
        kind : str, default 'maximax'
            Which peak each oscillator reports: 'maximax' (largest
            magnitude of either sign), 'positive' or 'negative'.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix.
        """
        from .core.data import TransientSpecification

        source = self.name_of(source)
        history = self[source]
        if (not history.shocks and history.averaging is None
                and not isinstance(history, TransientSpecification)):
            from .core.shocks import suggest

            history.shocks = suggest(history)
        # the analysis choices are parameters of the act, recorded in
        # the recipe like integrate's drift corner — a refresh after
        # the windows move recomputes at the same Q, spacing and kind
        return self._derive(source,
                            history.compute_srs(per_octave=per_octave,
                                                q=q, kind=kind),
                            f'{source} SRS',
                            recipe=('compute_srs',
                                    {'per_octave': per_octave,
                                     'q': q, 'kind': kind}))

    def detect_shocks(self, source: Any) -> int:
        """Find the events in a time history and mark them on it (the
        shock view's Detect), returning how many.

        The verb the API was missing (Brandon, 2026-08-25). `compute_srs`
        detects as a side effect when a record carries no windows, which
        served while the SRS came straight off the recording — but the
        recommended shock workflow filters first, and then the detection
        happened on the *filtered* record and the recording itself was
        left unmarked. The events belong to the recording: mark them
        there and every derivation carries them forward, because
        `core.filters` copies the marks onto whatever it makes.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself;
            `name_of` resolves either.

        Returns
        -------
        int
            How many events were found and marked on the record.
        """
        source = self.name_of(source)
        from .core.shocks import find

        history = self[source]
        history.shocks = find(history)
        return len(history.shocks or ())

    def filter_data(self, source: Any) -> str:
        """A time history through its low-pass (the filter view's
        Apply Filter) — every channel, zero phase, so the peaks stay put.

        The settings are the history's own `filtering`, set in the
        filter view; with none set, the suggestion is adopted the way
        `compute_frfs` adopts a suggested averaging, so the button
        works before the view has been visited.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself;
            `name_of` resolves either.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix.
        """
        source = self.name_of(source)
        history = self[source]
        if history.filtering is None:
            history.filtering = history.suggest_filtering()
        return self._derive(source, history.filter(),
                            f'{source} Filtered',
                            recipe=('filter_data', {}))

    def truncate_data(self, source: Any) -> str:
        """A time history cut to its truncation's span (the
        truncate view's Apply Truncation) — every channel between start and
        stop, the clock kept.

        The span is the history's own `truncation`, set in the
        truncate view. Unlike Filter Data there is no suggestion to
        adopt: the whole record is the only neutral span and keeping
        all of it is not an act, so with none set this refuses and
        says where to set one.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself;
            `name_of` resolves either.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix.
        """
        source = self.name_of(source)
        history = self[source]
        if history.truncation is None:
            raise ValueError('no span set: drag one in the truncate '
                             'view first')
        return self._derive(source, history.truncate(),
                            f'{source} Truncated',
                            recipe=('truncate_data', {}))

    def integrate(self, source: Any, drift_corner: Any = ...) -> str:
        """One integration of a time history (Integrate): acceleration
        channels become velocity, velocity becomes displacement.

        Whole record, never the shock windows — the reasons live in
        `core.filters`. The drift corner is a parameter of the act,
        recorded in the recipe: ``...`` takes the default, ``None``
        integrates raw.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself;
            `name_of` resolves either.
        drift_corner : float or None, optional
            High-pass corner in Hz applied after integration, to stop a
            sensor bias becoming a ramp. `...` takes the default;
            `None` integrates raw, drift and all.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix.
        """
        from .core.filters import DRIFT_CORNER

        corner = DRIFT_CORNER if drift_corner is ... else drift_corner
        source = self.name_of(source)
        result = self[source].integrate(corner)
        return self._derive(source, result,
                            _motion_name(source, result, 'Integrated'),
                            recipe=('integrate', {'drift_corner': corner}))

    def differentiate(self, source: Any) -> str:
        """One differentiation of a time history (Differentiate):
        displacement channels become velocity, velocity becomes
        acceleration.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself;
            `name_of` resolves either.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix.
        """
        source = self.name_of(source)
        result = self[source].differentiate()
        return self._derive(source, result,
                            _motion_name(source, result, 'Differentiated'),
                            recipe=('differentiate', {}))

    def compute_cpsds(self, source: Any) -> str:
        """The full cross-spectral matrix from a time history's averages
        (Compute CPSDs) — every channel against every channel.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself;
            `name_of` resolves either.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix.
        """
        source = self.name_of(source)
        return self._derive(source, self[source].compute_cpsds(),
                            f'{source} CPSDs', recipe=('compute_cpsds', {}))

    def transform(self, source: Any, shapes: Any, *,
                  records: Sequence[int] | None = None,
                  name: str | None = None) -> str:
        """Physical responses through a shape set to modal responses
        (Transform to Modal Responses) — `q = Φ⁺u` for the motions,
        `Φᵀf` for the forces, one record per mode and quantity at the
        modal coordinates `M1` … `Mn`.

        Any set serves: the six rigid-body shapes of a geometry make
        this the virtual point transformation. The result stands alone
        in the tree — its DOFs are on no geometry — with its
        provenance naming both the record and the set, and a
        `transform_report` saying what was shared, dropped and left
        unexplained.

        Parameters
        ----------
        source : str or object
            The time history, by name or as the object itself;
            `name_of` resolves either.
        shapes : str or object
            The shape set to transform through, by name or as itself.
        records : sequence of int, optional
            Which of the record's channels to carry through — the ones
            picked in the tree. All of them when omitted.
        name : str, optional
            What to call the result. Defaults to the record's name
            followed by 'Modal Responses'.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix.
        """
        from .core.transform import to_modal

        source, shapes = self.name_of(source), self.name_of(shapes)
        data, shape_set = self._transform_pair(source, shapes)
        picked = None if records is None else [int(i) for i in records]
        result, report = to_modal(data, shape_set, picked)
        result.transform_report = report
        params = {'shapes': shapes}
        if picked is not None:
            params['records'] = picked
        return self._derive(source, result,
                            name or f'{source} Modal Responses',
                            recipe=('transform', params))

    def expand(self, source: Any, shapes: Any, *,
               records: Sequence[int] | None = None,
               name: str | None = None) -> str:
        """Modal responses back through a shape set to physical
        responses (Expand to Physical Responses) — `u = Φq` at every
        DOF the set covers, linked into the set's group so the result
        animates on the geometry. A pick of modes expands those modes'
        contribution alone, and the name says which.

        Parameters
        ----------
        source : str or object
            The modal time history, by name or as the object itself;
            `name_of` resolves either.
        shapes : str or object
            The shape set it was transformed through, by name or as
            itself.
        records : sequence of int, optional
            Which modal records to expand — the modes picked in the
            tree. All of them when omitted.
        name : str, optional
            What to call the result. Defaults to the record's name
            with 'Modal Responses' read as 'Physical Responses', and
            the modes carried in brackets when they are not all.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix.
        """
        from .core.transform import carried_modes, to_physical

        source, shapes = self.name_of(source), self.name_of(shapes)
        data, shape_set = self._transform_pair(source, shapes)
        picked = None if records is None else [int(i) for i in records]
        result, report = to_physical(data, shape_set, picked)
        result.transform_report = report
        params = {'shapes': shapes}
        if picked is not None:
            params['records'] = picked
        if name is None:
            name = _physical_name(source, result)
            carried = carried_modes(report, shape_set)
            if carried:
                name += f' ({", ".join(carried)})'
        # its DOFs are the set's, so it belongs where the set is —
        # with the geometry it animates on — and not with its modal
        # source, which no geometry group can hold
        added = self._derive(source, result, name,
                             recipe=('expand', params), link=False)
        with contextlib.suppress(ValueError):
            self.link(shapes, added)
        return added

    def _transform_pair(self, source, shapes):
        data, shape_set = self[source], self[shapes]
        if not isinstance(data, DataArray):
            raise TypeError(f'{source!r} is not a data object')
        if not isinstance(shape_set, ShapeSet):
            raise TypeError(f'{shapes!r} is not a shape set')
        return data, shape_set

    def author_specification(self, source: Any, draft: Any, *,
                             name: str | None = None,
                             replace: bool = False) -> str:
        """A specification written from a sheet (the Specification
        reading's Make Specification) — autospectra at breakpoints,
        every cross term from a stated coherence and phase, bands in
        decibels — beside the object the sheet was opened on: a shape
        set (at its modal coordinates, ready to expand through it), a
        channel table (at its control channels), or a specification.
        With `replace`, the sheet is written *into* the specification
        it was opened from under its own name, so links and report
        slots hold — or into several at once, from a sheet that spans
        them. A sheet holding every channel of the specification
        rewrites it in the sheet's own form, its breakpoints or its
        lines becoming the object's; a sheet holding a picked subset
        merges back at the specification's own lines with the other
        channels untouched. A pair the sheet leaves unstated is absent
        from the result — nothing is assumed for it.

        Parameters
        ----------
        source : str, object, or list of str
            The object the sheet was opened on, by name or as itself;
            the specifications' names when the sheet spans several.
        draft : SpecificationDraft
            What the author stated (`core.author`). A draft with a
            pair unstated is refused by name.
        name : str, optional
            What to call the result. Defaults to the source's name
            followed by 'Specification'.
        replace : bool, default False
            Write the sheet into `source` — a specification, or
            several — in place.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix — or
            the source's own name when replaced (the first of them
            when several).
        """
        from .core.author import SpecificationDraft

        if isinstance(draft, dict):
            draft = SpecificationDraft.from_dict(draft)
        if replace:
            names = ([self.name_of(s) for s in source]
                     if isinstance(source, (list, tuple))
                     else [self.name_of(source)])
            for each in names:
                if not isinstance(self[each], Specification):
                    raise TypeError(f'{each!r} is not a specification to '
                                    'replace')
            spanning = len(set(draft.sources)) > 1 or (
                bool(draft.sources) and len(names) > 1)
            for each in names:
                own = draft.for_source(each) if spanning else draft
                if own.sources:
                    own = own._copy(sources=[])
                self[each] = _rewritten(self[each], own,
                                        f'edited as a sheet on {each}')
                self.provenance[each] = {
                    'verb': 'author_specification', 'source': each,
                    'params': {'draft': own.as_dict(), 'into': True},
                    'state': None}
            return names[0]
        source = self.name_of(source)
        result = draft.make(f'written on {source}')
        return self._derive(source, result,
                            name or f'{source} Specification',
                            recipe=('author_specification',
                                    {'draft': draft.as_dict()}))

    def generate_rigid_body_modes(self, source: Any, *,
                                  name: str | None = None) -> str:
        """The six rigid-body mode shapes of a geometry (Generate Rigid
        Body Mode Shapes) — three translations and three rotations
        about its reference point, as a shape set in the geometry's
        group.

        The point, and the mass and inertia that mass-normalize the
        set, are the geometry's own `mass_properties`, set in the
        rigid-body view. With none set the centroid is adopted, unit
        shapes about the middle of the model — a real answer, unlike
        a whole-record truncation, and the one the virtual-point
        transformation wants most often — and stored, so the
        staleness fingerprint records what was actually used.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself; `name_of`
            resolves either.
        name : str, optional
            What to call the result. Defaults to the geometry's name
            followed by 'Rigid Body Modes'.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix.
        """
        from .core.rigid import rigid_body_shapes

        source = self.name_of(source)
        geometry = self[source]
        if not isinstance(geometry, Geometry):
            raise TypeError(f'{source!r} is not a geometry')
        if geometry.mass_properties is None:
            geometry.mass_properties = geometry.suggest_mass_properties()
        return self._derive(
            source, rigid_body_shapes(geometry, geometry.mass_properties),
            name or f'{source} Rigid Body Modes',
            recipe=('generate_rigid_body_modes', {}))

    def fit_modes(self, source: str, *, bounds: tuple[float, float] | None
                  = None, limit: int = 30, name: str | None = None,
                  at: Sequence[tuple[float, float]] | None = None,
                  refine: int = 0) -> str:
        """Fit a modal model to an FRF set (the fitting screen).

        The screen's loop, scripted: confirm the suggestion, take the
        next, `limit` times. The session's own suggestion logic is the
        whole judgment — a confirmed peak is spoken for unless the
        shape standing there is somebody else's — so the loop adds no
        second opinion. It used to: a proximity guard here vetoed any
        suggestion within 1 Hz of a confirmed mode, which was the same
        ridge-trap bandaid the session has since outgrown, and it
        silently skipped the repeated pair's second tooth that
        `suggest` had deliberately offered. On the screen the person
        stops the loop; scripted, `limit` is that judgment, and the
        plate demo's own cap is the worked example of choosing it.

        On the screen the equivalent of `bounds` is the zoom — what is
        on the plot is what gets searched. A script has no plot, so it
        says so here.

        Parameters
        ----------
        source : str
            The FRF set to fit.
        bounds : tuple of float, optional
            (low, high) frequency limits to fit within. Defaults to
            the whole band.
        limit : int, default 30
            The most modes to accept.
        name : str, optional
            What to call the shape set.
        at : sequence of tuple, optional
            Explicit (frequency, damping) picks — each optionally
            (frequency, damping, description) — confirmed in the
            order given — the residual is peeled sequentially, so the
            order is part of the fit. This is how an interactive
            session replays: the fitting screen journals its confirms
            as exactly this call. `bounds` and `limit` are ignored
            when picks are given.
        refine : int, default 0
            Times to run the joint residue refinement after the
            confirms — the screen's Refine All, counted.

        Returns
        -------
        str
            The name the result was added under, unique within
            the project — a clash gets a numbered suffix.
        """
        from .core.modal_fit import ModalFitSession

        source = self.name_of(source)
        session = ModalFitSession(self[source])
        if at is not None:
            for pick in at:
                frequency, damping, *described = pick
                session.confirm(frequency=float(frequency),
                                damping=float(damping),
                                description=(described[0] if described
                                             else None))
        else:
            low, high = bounds or (float(self[source].abscissa[0]),
                                   float(self[source].abscissa[-1]))
            session.suggest((low, high))
            for _ in range(limit):
                session.confirm()
                session.suggest((low, high))
        for _ in range(int(refine)):
            session.refine_residues()
        return self._derive(source, session.shape_set(),
                            name or f'{source} Modes')

    def project_onto_basis(self, source: str, *, onto: str | None = None,
                           tolerance: float = 0.02,
                           name: str | None = None) -> str:
        """A shape set sampled at the Basis set's DOFs (Project onto
        Basis DOFs): nearest node within `tolerance` of the basis
        model's extent, the displacement there dotted with each basis
        DOF's direction. Returns the new set's name.

        Parameters
        ----------
        source : str
            The shape set to project.
        onto : str, optional
            The basis to project onto. Defaults to the project's basis.
        tolerance : float, default 0.02
            The residual a fit may leave.
        name : str, optional
            What to call the result.

        Returns
        -------
        str
            The name the result was added under, unique within
            the project — a clash gets a numbered suffix.
        """
        from .core.correlate import project_shapes

        source = self.name_of(source)
        onto = None if onto is None else self.name_of(onto)
        basis = ((onto, self[onto]) if onto is not None
                 else self._basis_shapes())
        if basis is None:
            raise ValueError('name the set to project onto, or declare a '
                             'Basis holding one')
        basis_name, basis_shapes = basis
        home = self.geometry_for(basis_name)
        theirs = self.geometry_for(source)
        if home is None or theirs is None:
            raise ValueError('both shape sets need a geometry — link one '
                             'to each side')
        projected, report = project_shapes(self[source], theirs[1],
                                           basis_shapes, home[1],
                                           tolerance=tolerance)
        # how it was made travels with it: matched and dropped counts,
        # the worst distance accepted
        projected.projection_report = report
        return self._derive(basis_name, projected,
                            name or f'{source} @ Basis DOFs')

    def match_modes(self, first: str, second: str, *,
                    pairs: Iterable[tuple[int, int]] | None = None,
                    macs: Iterable[float] | None = None,
                    threshold: float = 0.7,
                    name: str = 'Matched Modes') -> str:
        """Commit matched mode pairs (the comparison screen's **+**).

        With `pairs` those pairs exactly; otherwise each of `first`'s
        modes takes its best partner in `second` when the MAC clears
        `threshold`. Comparing across geometries goes through the
        projection first, as the screen does.

        Parameters
        ----------
        first : str
            One shape set, by name.
        second : str
            The other shape set, by name.
        pairs : iterable of tuple of int, optional
            Explicit (first, second) index pairs, overriding the
            automatic matching.
        macs : iterable of float, optional
            MAC values for those pairs.
        threshold : float, default 0.7
            The lowest MAC an automatic pairing may have.
        name : str, default 'Matched Modes'
            What to call the result.

        Returns
        -------
        str
            The name the result was added under, unique within
            the project — a clash gets a numbered suffix.
        """
        import numpy as np

        from .core.matches import MatchedModes

        first, second = self.name_of(first), self.name_of(second)
        if pairs is None or macs is None:
            matrix = self.comparison_mac(first, second)
            if pairs is None:
                pairs = [(row, int(np.argmax(matrix[row])))
                         for row in range(matrix.shape[0])
                         if matrix[row].max() >= threshold]
            pairs = [(int(row), int(column)) for row, column in pairs]
            # the displayed comparison's own values, which a
            # name-matched recompute would not reproduce
            macs = [float(matrix[row, column]) for row, column in pairs]
        pairs = [(int(row), int(column)) for row, column in pairs]
        macs = [float(mac) for mac in macs]
        first_home = self.geometry_for(first)
        second_home = self.geometry_for(second)
        matched = MatchedModes(
            first, second, pairs, macs,
            first_geometry=first_home[0] if first_home else None,
            second_geometry=second_home[0] if second_home else None)
        # Unlinked, unlike every other derived object. A link group is a
        # side of the comparison, and this object *is* the comparison —
        # it names a set on each side, so putting it in one of them
        # claims it belongs to the half it is measuring against the
        # other. It carries its own two geometries (`first_geometry`,
        # `second_geometry`), so it needs no group to find them.
        #
        # Linking it by hand still works, for anyone who wants it
        # bracketed with a side.
        return self.add(name, matched)

    def comparison_mac(self, first: str, second: str) -> np.ndarray:
        """The MAC between two shape sets as the comparison screen
        shows it: across geometries the second set is projected onto
        the first's DOFs, because matching DOF *names* across
        geometries would trust them to mean the same directions.

        Parameters
        ----------
        first : str
            One shape set, by name.
        second : str
            The other shape set, by name.

        Returns
        -------
        numpy.ndarray
            The MAC matrix, first's shapes down the
            rows and second's across the columns.
        """
        from .core.correlate import project_shapes
        from .core.shapes import cross_mac

        first, second = self.name_of(first), self.name_of(second)
        home, theirs = self.geometry_for(first), self.geometry_for(second)
        if (home is None or theirs is None or home[1] is theirs[1]):
            return cross_mac(self[first], self[second])
        projected, _report = project_shapes(self[second], theirs[1],
                                            self[first], home[1])
        return cross_mac(self[first], projected)

    def plot_mac(self, first: Any, second: Any = None,
                 **kwargs: Any) -> Any:
        """The MAC picture the comparison screen draws: `first`
        against itself, or against `second` — projected across
        geometries exactly as `comparison_mac` does it, which a
        shape set's own `plot_mac` cannot, since it compares by DOF
        name and knows no geometry.

        Parameters
        ----------
        first : str or ShapeSet
            One shape set, by name or as the object.
        second : str or ShapeSet, optional
            The other. Absent, the auto-MAC.
        **kwargs
            Passed to the drawing: `path=` renders to a file,
            `bars=True` is the 3-D reading (then `screenshot=`),
            `theme`, `title`, `size`, `show`.

        Returns
        -------
        object
            Whatever the drawing returns — a window, an image.
        """
        from .plot import plot_mac_matrix

        first = self.name_of(first)
        if second is None:
            return plot_mac_matrix(self[first].frequency,
                                   self[first].auto_mac(), **kwargs)
        second = self.name_of(second)
        return plot_mac_matrix(self[first].frequency,
                               self.comparison_mac(first, second),
                               column_frequencies=self[second].frequency,
                               **kwargs)

    def merge(self, *names: str, name: str | None = None) -> str:
        """Combine compatible objects into one (Merge).

        Same concrete type only, and each kind has its own rule about
        what may join: geometries need disjoint node ids, shape sets
        the same DOF cover, data arrays an identical abscissa. The
        merged object replaces its parts.

        Parameters
        ----------
        *names : str
            The objects to act on, by name.
        name : str, optional
            What to call the merged object.

        Returns
        -------
        str
            The name the result was added under, unique within
            the project — a clash gets a numbered suffix.
        """
        from .core.merge import merge as merge_objects

        names = tuple(self.name_of(n) for n in names)
        merged = merge_objects([self[n] for n in names])
        group = self.group_of(names[0]) or []
        self.remove(*names)
        added = self.add(name or names[0], merged)
        rest = [member for member in group if member in self]
        if rest:
            self.link(added, *rest)
        return added

    def export(self, name: str, path: str | os.PathLike,
               unit_system: Any = None, **kwargs: Any) -> str:
        """Write an object to a foreign format, chosen by suffix —
        every registered writer, `.unv`, `.exo`, `.npz`, `.bdf`,
        `.afu`/`.ati`/`.ash`, `.xlsx`, `.3mf`, `.stl`, `.vdreport` and
        the rest of `io.exporters()` (Export).

        Parameters
        ----------
        name : str
            The object to write.
        path : str or os.PathLike
            Where to write it. The format follows the extension.
        unit_system : UnitSystem, optional
            Units to write in. Defaults to the project's own.
        **kwargs
            Passed through to the exporter.

        Returns
        -------
        str
            The path written.
        """
        from .io import export_file

        name = self.name_of(name)
        export_file(self[name], str(path), unit_system=unit_system,
                    **kwargs)
        return str(path)

    def generate_report(self, template: str = 'modal',
                        name: str = 'Report') -> str:
        """Build a report from a starter template, bound symbolically
        to this project's structure (Generate Report).

        Parameters
        ----------
        template : str, default 'modal'
            Which starter to build: 'modal', 'random', 'shock',
            'transient', 'sine', 'sysid' or 'empty' — or a saved
            template, by the name it was saved under in the templates
            folder or by the path of a `.vdreport` file (a report
            exported on its own; `io.report_template`).
        name : str, default 'Report'
            What to call the report object.

        Returns
        -------
        str
            The name the result was added under, which is unique
            within the project — a clash gets a numbered suffix.
        """
        from .core.report import (
            Report,
            modal_template,
            random_template,
            shock_template,
            sine_template,
            sysid_template,
            transient_template,
        )

        builders = {'modal': modal_template, 'random': random_template,
                    'shock': shock_template, 'transient': transient_template,
                    'sine': sine_template, 'sysid': sysid_template}
        if template in builders:
            report = builders[template](self, links=self.links)
        elif template == 'empty':
            report = Report('Report')
        else:
            import pathlib

            from .io import report_template

            saved = dict(report_template.saved_templates())
            path = pathlib.Path(saved.get(str(template), str(template)))
            if not (path.name.endswith(report_template.SUFFIX)
                    and path.is_file()):
                raise ValueError(
                    f'unknown template {template!r}: not a built-in '
                    f'({", ".join([*builders, "empty"])}), a saved '
                    f'template ({", ".join(saved) or "none saved"}) or a '
                    f'{report_template.SUFFIX} file')
            report = report_template.load(path)
        return self.add(name, report)

    def export_report(self, name: str, path: str | os.PathLike,
                      unit_system: Any = None) -> str:
        """Write a report as one self-contained HTML file (Export).

        Parameters
        ----------
        name : str
            The report object to render.
        path : str or os.PathLike
            Where to write the self-contained HTML file.
        unit_system : UnitSystem, optional
            Units to render in. Defaults to the project's own.

        Returns
        -------
        str
            The path written.
        """
        from .report import render_html

        name = self.name_of(name)
        html = render_html(self[name], self, unit_system, links=self.links)
        path = os.path.expanduser(str(path))
        with open(path, 'w', encoding='utf-8') as out:
            out.write(html)
        return path

    def table(self, name: Any) -> tuple[list[str], list[list[str]]]:
        """(headers, rows) for an object that reads as a table.

        The instrumentation of a channel table, the identified
        parameters of a shape set — the same rows the report prints,
        so a script and a report cannot disagree about what is in one.

        Parameters
        ----------
        name : str or object
            The object to tabulate.

        Returns
        -------
        tuple of (list of str, list of list of str)
            The column headings and the rows, both as text.
        """
        from .core.tables import table_of

        name = self.name_of(name)
        built = table_of(self[name])
        if built is None:
            raise ValueError(f'{name} is a {type(self[name]).__name__}, '
                             'which does not read as a table')
        return built

    def plot(self, name: str, **kwargs: Any) -> Any:
        """Plot an object the way the GUI plots it: data as curves, a
        geometry as its scene, a shape set as its auto-MAC.

        Parameters
        ----------
        name : str
            The object to draw.
        **kwargs
            Passed through to the object's own plot method.

        Returns
        -------
        object
            Whatever the underlying plot call returns.
        """
        obj = self[self.name_of(name)]
        plot = getattr(obj, 'plot', None)
        if plot is None:
            raise TypeError(f'{name!r} is a {type(obj).__name__}, which '
                            'has no plot')
        return plot(**kwargs)

    def animate(self, name: str, mode: int = 0, **kwargs: Any) -> Any:
        """A mode shape — or a complex spectrum's operating deflection —
        moving on the geometry it answers to.

        Parameters
        ----------
        name : str
            The shape set to animate.
        mode : int, default 0
            Which mode, by index.
        **kwargs
            Passed through to the scene.

        Returns
        -------
        object
            The plotter the animation is running in.
        """
        name = self.name_of(name)
        home = self.geometry_for(name)
        if home is None:
            raise ValueError(f'{name!r} has no geometry — link one')
        target = self[name]
        if isinstance(target, ShapeSet):
            return target.animate(home[1], mode, **kwargs)
        # spectra pick a line by `frequency=`, not a mode number, and a
        # positional 0 must not read as 0 Hz
        return target.animate(home[1], **kwargs)

    @property
    def names(self) -> list[str]:
        """Every object's name, in the order the tree shows them."""
        return self.ordered_names()

    def name_of(self, target: Any) -> str:
        """The name an object goes by here; a name passes through.

        Verbs take either, so `project.compute_psds('Time History')`
        and `project.compute_psds(project.basis.time_history)` are the
        same call.

        Parameters
        ----------
        target : str or object
            A name, or an object the project holds.

        Returns
        -------
        str
            The name it is stored under.
        """
        if isinstance(target, str):
            return target
        for name, obj in self.items():
            if obj is target:
                return name
        raise ValueError(f'that {type(target).__name__} is not in this '
                         f'project — add it first')

    def _basis_shapes(self) -> tuple[str, Any] | None:
        for member in self.basis.names:
            if isinstance(self.get(member), ShapeSet):
                return member, self[member]
        return None

    def extract_sine(self, source: Any,
                     specification: Any = None) -> list[str]:
        """Each specification tone's level, read out of a recording
        (Extract Sine Levels) — one object per tone, because each tone
        sweeps its own frequencies on its own clock.

        The specification is found in the project when not named —
        the one SineSweepSpecification there is — and each result is
        linked to the recording it was read from.

        Parameters
        ----------
        source : str or object
            The sine level set or run to read.
        specification : str or object, optional
            The sweep specification to extract against. Defaults to
            the project's own, when it holds exactly one.

        Returns
        -------
        list of str
            The names of the levels added, one per tone.
        """
        from .core.sine import extract_sine

        source = self.name_of(source)
        if specification is None:
            spec = self.sine_sweep_specification
        else:
            spec = self[self.name_of(specification)]
        levels = extract_sine(self[source], spec)
        return [self._derive(source, levels, 'Sine Levels')]

    def _derive(self, source: str, obj: Any, name: str,
                recipe: tuple[str, dict[str, Any]] | None = None,
                link: bool = True) -> str:
        added = self.add(name, obj)
        if link:
            try:
                self.link(source, added)
            except ValueError:
                # a derived object that will not share its source's
                # geometry still belongs in the project; the link is
                # the claim that failed, not the result
                pass
        if recipe is not None:
            verb, params = recipe
            self.provenance[added] = {
                'verb': verb, 'source': source, 'params': dict(params),
                'state': self._analysis_state(source, verb, params)}
        return added

    # ---- staleness: what a change to the analysis settings owes ---------

    #: which part of a source each verb reads — the fingerprint follows
    #: the dependency, so changing the shocks never marks a PSD stale
    _VERB_READS: ClassVar[dict[str, str]] = {
        'compute_spectra': 'averaging', 'compute_psds': 'averaging',
        'compute_cpsds': 'averaging', 'compute_frfs': 'averaging',
        'compute_multiple_coherence': 'averaging',
        'compute_srs': 'shocks', 'compute_octave': 'content',
        # the filter reads the settings riding the source; integration
        # and differentiation read the record's bytes, so refreshing a
        # re-filtered source cascades down the whole motion chain
        'filter_data': 'filtering', 'truncate_data': 'truncation',
        'integrate': 'content', 'differentiate': 'content',
        # the settings and the nodes both: a moved node, or a turned
        # displacement system, changes the shapes as surely as a moved
        # reference point does
        'generate_rigid_body_modes': 'rigid',
        # the record and the set both: a refitted set badges the modal
        # responses made through it
        'transform': 'pair', 'expand': 'pair'}

    def _analysis_state(self, source: str, verb: str,
                        params: Mapping[str, Any] | None = None) -> Any:
        """The fingerprint of what `verb` would read from `source` now.

        Settings, never timestamps: the badge must appear exactly when
        the numbers would differ, and survive a save and a load.
        `params` is the recipe's, for a verb that reads a second
        object named there.
        """
        from dataclasses import asdict

        reads = self._VERB_READS.get(verb)
        obj = self.get(source)
        if reads is None or obj is None:
            return None
        if reads == 'pair':
            other = self.get((params or {}).get('shapes'))
            if other is None:
                return None
            import hashlib

            digest = hashlib.blake2b(digest_size=16)
            for array in (obj.abscissa, obj.ordinate, other.shape_matrix,
                          other.frequency):
                digest.update(np.ascontiguousarray(array).tobytes())
            digest.update(','.join(obj.response_dof).encode())
            digest.update(','.join(other.coordinate).encode())
            return ('content', digest.hexdigest())
        if reads == 'averaging':
            averaging = getattr(obj, 'averaging', None)
            return ('averaging', None if averaging is None
                    else asdict(averaging))
        if reads == 'filtering':
            filtering = getattr(obj, 'filtering', None)
            return ('filtering', None if filtering is None
                    else asdict(filtering))
        if reads == 'truncation':
            truncation = getattr(obj, 'truncation', None)
            return ('truncation', None if truncation is None
                    else asdict(truncation))
        if reads == 'shocks':
            shocks = getattr(obj, 'shocks', None) or ()
            return ('shocks', tuple((float(s.start), float(s.duration))
                                    for s in shocks))
        if reads == 'rigid':
            import hashlib

            properties = getattr(obj, 'mass_properties', None)
            digest = hashlib.blake2b(digest_size=16)
            for field in ('node_id', 'node_xyz', 'node_disp_cs',
                          'cs_id', 'cs_matrix'):
                digest.update(np.ascontiguousarray(
                    getattr(obj, field)).tobytes())
            return ('rigid', {
                'properties': (None if properties is None
                               else asdict(properties)),
                'nodes': digest.hexdigest()})
        # content: the source is itself derived (an octave banding of a
        # PSD), so a recompute upstream changes these bytes and the
        # staleness cascades without anyone walking a graph
        import hashlib

        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.ascontiguousarray(obj.abscissa).tobytes())
        digest.update(np.ascontiguousarray(obj.ordinate).tobytes())
        return ('content', digest.hexdigest())

    def stale(self) -> dict[str, str]:
        """{derived name: why} for everything whose source's settings
        have moved since it was computed.

        A missing source, or a derivation this bookkeeping predates,
        answers nothing — absence of evidence is not staleness.
        """
        out = {}
        for name, record in self.provenance.items():
            if name not in self or record.get('source') not in self:
                continue
            now = self._analysis_state(record['source'], record['verb'],
                                       record.get('params'))
            was = record.get('state')
            if was is None or now is None:
                continue
            if _state_tuple(now) != _state_tuple(was):
                out[name] = _state_story(was, now)
        return out

    def refresh(self, name: Any) -> str:
        """Recompute a derived object in place, under its own name.

        The links, the report's bindings and the grids all key on the
        name, so replacing the value under it is what keeps every
        reference honest. Anything derived from *this* object goes
        stale by content, which is the cascade — refreshed one badge
        at a time, or all at once, but always by a person.

        Parameters
        ----------
        name : str or object
            The derived object to recompute, in place and under
            its own name.

        Returns
        -------
        str
            The name refreshed.
        """
        name = self.name_of(name)
        record = self.provenance.get(name)
        if record is None:
            raise ValueError(f'{name!r} records no derivation to re-run')
        source = record['source']
        if source not in self:
            raise ValueError(
                f"{name!r} was computed from {source!r}, which is gone")
        verb, params = record['verb'], record.get('params', {})
        rebuilt = _RECOMPUTE[verb](self, source, params)
        self[name] = rebuilt
        record['state'] = self._analysis_state(source, verb, params)
        return name

    def refresh_stale(self) -> list[str]:
        """Refresh everything stale, sources before their dependents,
        until nothing is — the project row's one click."""
        done: list[str] = []
        # bounded: each pass refreshes at least one or stops, and a
        # refresh can only newly stale things derived from it
        for _ in range(len(self.provenance) + 1):
            waiting = self.stale()
            if not waiting:
                break
            for name in self.ordered_names():
                if name in waiting:
                    done.append(self.refresh(name))
        return done

    # ---- the file ----------------------------------------------------------

    def save(self, path: str | os.PathLike) -> str:
        """Write the whole project to one file: `.vdyn`, or `.mat` for
        the same layout in MATLAB's container.

        Parameters
        ----------
        path : str or os.PathLike
            Where to write the file. A `.mat` suffix writes the project
            as MATLAB structs (`io.matlab`); anything else is `.vdyn`.

        Returns
        -------
        str
            The path written.
        """
        from .io import export_file, save_test

        if str(path).endswith('.mat'):
            export_file(self, str(path))
            return str(path)
        save_test(str(path), self.name, dict(self),
                  active_geometry=self.active_geometry,
                  project_type=self.project_type, links=self.links,
                  provenance=self.provenance)
        return str(path)

    @classmethod
    def open(cls, path: str | os.PathLike) -> Project:
        """Read a project back, from `.vdyn` or from `.mat`."""
        from .io import import_file, load

        loaded = (import_file(str(path)) if str(path).endswith('.mat')
                  else load(str(path)))
        if isinstance(loaded, Project):
            # whatever the loader did on the way — construct, add,
            # link — the session's story starts here: one line that
            # reproduces this state exactly
            loaded.journal = [
                f'project = visualdynamics.Project.open({str(path)!r})']
            return loaded
        raise ValueError(f'{path} holds a single object, not a project')

    def _journal_arg(self, value: Any) -> str:
        """One argument as the source text a script would pass.

        An object handed to a verb — `project.basis.time_history` —
        is not eval-able by repr, but its *name* is what the verb
        resolves it to and what a script would say; everything plain
        reprs as itself.
        """
        if isinstance(value, os.PathLike):
            return repr(str(value))
        if isinstance(value, (str, int, float, bool, type(None),
                              tuple, list)):
            return repr(value)
        try:
            return repr(self.name_of(value))
        except (KeyError, ValueError):
            return repr(value)

    @contextlib.contextmanager
    def journal_as(self, line: str | None):
        """Record a stretch of front-end work as one replaying line.

        The GUI imports a file by building the objects itself and
        adding them one by one; journaled verb by verb, that stretch
        is a pile of not-replayable comments — when the honest record
        is the single `import_file` call a script would make. Inside
        the stretch every verb stays quiet, exactly as verbs nested in
        verbs do; the line lands only when the stretch succeeds.

        Parameters
        ----------
        line : str or None
            The line that replays the stretch — None to record
            nothing at all.

        Returns
        -------
        None
        """
        self._journal_depth += 1
        try:
            yield
        finally:
            self._journal_depth -= 1
        if line is not None:
            self.journal.append(line)

    def record_setting(self, target: Any, attribute: str,
                       value: Any) -> None:
        """A settings write, journaled the way a script would make it.

        The front ends' funnel: the GUI stores analysis settings by
        assignment — a dragged averaging span, a filter corner, the
        shock windows — and those writes are session acts as much as
        any verb. A repeated write to the same slot replaces its own
        last line, so a session of nudging settles to the one
        assignment that stands rather than a line per keystroke.

        Parameters
        ----------
        target : str or object
            The object written to, by name or as itself.
        attribute : str
            Which settings attribute was stored.
        value : Any
            What was stored; its repr must rebuild it, which every
            settings dataclass here guarantees.

        Returns
        -------
        None
        """
        try:
            name = self.name_of(target)
        except (KeyError, ValueError):
            return                     # not (or no longer) in the project
        prefix = f'project[{name!r}].{attribute} = '
        line = prefix + repr(value)
        if self.journal and self.journal[-1].startswith(prefix):
            self.journal[-1] = line
        else:
            self.journal.append(line)

    def record_call(self, target: Any, method: str, *args: Any,
                    **kwargs: Any) -> None:
        """A method call on an object, journaled as a script makes it.

        The front ends' funnel for object verbs that are not Project
        verbs — a traceline added to a geometry, a photo renamed —
        each an act of the session the console must speak (Brandon,
        2026-08-30: adding a traceline said nothing).

        Parameters
        ----------
        target : str or object
            The object acted on, by name or as itself.
        method : str
            The method a script would call.
        *args : Any
            The call's arguments; their reprs must rebuild them.
        **kwargs : Any
            Keyword arguments, same rule.

        Returns
        -------
        None
        """
        try:
            name = self.name_of(target)
        except (KeyError, ValueError):
            return
        shown = [self._journal_arg(a) for a in args]
        shown += [f'{key}={self._journal_arg(value)}'
                  for key, value in kwargs.items()]
        self.journal.append(
            f'project[{name!r}].{method}({", ".join(shown)})')

    #: what a settings repr needs in scope, added to `session_script`
    #: only when a journal line uses it
    _SCRIPT_IMPORTS: ClassVar[tuple[tuple[str, str], ...]] = (
        ('Averaging(', 'from visualdynamics.core.averaging import Averaging'),
        ('Filtering(', 'from visualdynamics.core.filters import Filtering'),
        ('Truncation(', 'from visualdynamics.core.truncate import Truncation'),
        ('Shock(', 'from visualdynamics.core.shocks import Shock'),
        ('Photos(', 'from visualdynamics.core.photos import Photos'),
        ('MassProperties(',
         'from visualdynamics.core.rigid import MassProperties'),
        ('SpecificationDraft(',
         'from visualdynamics.core.author import SpecificationDraft'),
        ('np.array(', 'import numpy as np'),
    )

    def session_script(self) -> str:
        """This sitting's acts as a runnable Python script.

        The journal joined under its imports: every verb that ran and
        every setting stored — clicked in the GUI or called from a
        script — recorded as the line that reproduces it, so a session
        worked up by hand can be replayed, adapted, or kept. Reads and
        refusals are absent on purpose: the script is what *happened
        to the project*, and a verb that raised changed nothing.

        Returns
        -------
        str
            A Python script; running it rebuilds this session's
            project from the same inputs.
        """
        head = ['import visualdynamics']
        head += [imported for token, imported in self._SCRIPT_IMPORTS
                 if any(token in line for line in self.journal)]
        return '\n'.join([*head, '', *self.journal])


def _journaled(verb):
    """Record a successful verb call as the line that would replay it.

    After success, never before: a refused verb changed nothing, and a
    script replaying the session must not replay the refusal. The
    depth guard keeps a verb that calls other verbs — merge removes,
    adds and links; refresh recomputes — down to the one line the user
    could have typed.
    """
    import functools

    @functools.wraps(verb)
    def recorded(self, *args, **kwargs):
        self._journal_depth += 1
        try:
            result = verb(self, *args, **kwargs)
        finally:
            self._journal_depth -= 1
        if not self._journal_depth:
            # `add` takes the object itself, and an object built in
            # this session has no source text to replay it from — the
            # honest line is a comment, so the reader sees the gap and
            # exec skips it, rather than a plausible call that adds
            # the wrong thing (name_of would have resolved the object
            # to its own just-added name).
            plain = (str, int, float, bool, type(None))
            unsayable = (verb.__name__ == 'add' and len(args) > 1
                         and not isinstance(args[1], plain))
            if unsayable:
                built = (f'<{type(args[1]).__name__} built in '
                         'this session>')
                shown = [self._journal_arg(args[0]), built]
            else:
                shown = [self._journal_arg(a) for a in args]
            shown += [f'{key}={self._journal_arg(value)}'
                      for key, value in kwargs.items()]
            line = f'project.{verb.__name__}({", ".join(shown)})'
            if unsayable:
                line = f'# {line} — not replayable'
            # the sheet edits a specification live, one verb call per
            # keystroke (2026-09-06): a run of edits on one object
            # settles to the line that stands, as a nudged setting
            # does in `record_setting`, rather than a line per key
            prefix = (f'project.{verb.__name__}({shown[0]}, '
                      if verb.__name__ in _SETTLING and shown else None)
            if (prefix and kwargs.get('replace') and self.journal
                    and self.journal[-1].startswith(prefix)
                    and self.journal[-1].endswith(', replace=True)')):
                self.journal[-1] = line
            else:
                self.journal.append(line)
        return result
    return recorded


#: verbs whose consecutive calls on one object replace their own last
#: journal line — the live edits of an editor, not acts in sequence
_SETTLING = frozenset({'author_specification'})


#: every verb that changes the project, or writes one of its outputs —
#: what the journal records. Reads (names, stale, table) say nothing a
#: script needs to replay. Wrapped in one pass rather than decorated at
#: thirty definition sites, so the list of what journals is one list.
def _is_time(project, obj):
    return isinstance(obj, TimeHistory)


def _has_dims(obj, wanted):
    return isinstance(obj, TimeHistory) and bool(
        {obj.known_dim(i) for i in range(obj.num_records)} & wanted)


def _two_shape_sets(project, obj):
    return isinstance(obj, ShapeSet) and sum(
        isinstance(other, ShapeSet) for other in project.values()) >= 2


def _integrable(project, obj):
    from .core.filters import INTEGRATED

    return _has_dims(obj, INTEGRATED.keys())


def _differentiable(project, obj):
    from .core.filters import DIFFERENTIATED

    return _has_dims(obj, DIFFERENTIATED.keys())


#: verb → does it apply to this object — the one applicability table,
#: read by `Project.verbs` and by the window's bar, so the
#: API and the interface cannot disagree about what an object offers.
#: Each predicate mirrors the verb's own cheap refusals (a time history
#: with no acceleration cannot integrate; FRFs need drive channels),
#: never its deep ones — a predicate that re-ran the verb to answer
#: would make asking as expensive as doing.
_VERB_APPLIES: tuple = (
    ('integrate', _integrable),
    ('differentiate', _differentiable),
    ('filter_data', _is_time),
    ('truncate_data', _is_time),
    ('detect_shocks', _is_time),
    ('compute_spectra', _is_time),
    ('compute_psds', _is_time),
    ('compute_cpsds', _is_time),
    ('compute_srs', _is_time),
    ('compute_frfs', lambda p, o: (_is_time(p, o)
                                   and bool(o.drive_dofs()))),
    ('compute_multiple_coherence', lambda p, o: (
        _is_time(p, o) and bool(o.drive_dofs()))),
    ('extract_sine', lambda p, o: (_is_time(p, o) and any(
        isinstance(other, SineSweepSpecification)
        for other in p.values()))),
    # a specification bands too, limits and all (Brandon, 2026-09-18)
    ('compute_octave', lambda p, o: isinstance(o, Psd)),
    ('fit_modes', lambda p, o: isinstance(o, Frf)),
    ('transform', lambda p, o: _reads_as(p, o, 'physical')),
    ('expand', lambda p, o: _reads_as(p, o, 'modal')),
    ('generate_rigid_body_modes', lambda p, o: isinstance(o, Geometry)),
    ('author_specification', lambda p, o: isinstance(
        o, (ShapeSet, Specification, ChannelTable))),
    ('project_onto_basis', _two_shape_sets),
    ('match_modes', _two_shape_sets),
    ('merge', lambda p, o: any(type(other) is type(o) and other is not o
                               for other in p.values())),
)


def _reads_as(project, obj, direction):
    """Whether some shape set in the project reads `obj` as physical
    (transformable) or modal (expandable) — the applicability of a
    two-object verb, answered against every partner available."""
    from .core.transform import reads_as

    if not isinstance(obj, DataArray):
        return False
    return any(reads_as(obj, other) == direction
               for other in project.values() if isinstance(other, ShapeSet))


def _physical_name(source, result=None):
    """'Run Modal Responses' expands to 'Run Physical Responses'; a
    specification to '… Physical Specification'; anything else gets
    the responses suffix."""
    suffix = ' Modal Responses'
    if source.endswith(suffix):
        return source[:-len(suffix)] + ' Physical Responses'
    if isinstance(result, Specification):
        return f'{source} at Physical DOFs'
    return f'{source} Physical Responses'


def _pair_reads(project, objects, direction):
    """One data object and one shape set — a geometry may come along
    — and the record reads through the set the given way."""
    from .core.transform import reads_as

    data = [o for o in objects if isinstance(o, DataArray)]
    sets = [o for o in objects if isinstance(o, ShapeSet)]
    rest = [o for o in objects
            if not isinstance(o, (DataArray, ShapeSet, Geometry))]
    return (len(data) == 1 and len(sets) == 1 and not rest
            and reads_as(data[0], sets[0]) == direction)


def _two_sets_on_two_geometries(project, objects):
    """Two shape sets, each answering to its own geometry — the
    projection samples one model's field at the other's points."""
    if len(objects) != 2 or not all(isinstance(o, ShapeSet) for o in objects):
        return False
    homes = [project.geometry_for(o) for o in objects]
    return (all(home is not None for home in homes)
            and homes[0][1] is not homes[1][1])


def _mergeable(project, objects):
    from .core.merge import mergeable

    return len(objects) >= 2 and mergeable(list(objects)) is None


#: the verbs that act on a *combination* of objects, with what makes
#: one — the bar's answer for a multi-selection (`selection_verbs`)
_SELECTION_APPLIES: tuple = (
    ('transform', lambda p, o: _pair_reads(p, o, 'physical')),
    ('expand', lambda p, o: _pair_reads(p, o, 'modal')),
    ('project_onto_basis', _two_sets_on_two_geometries),
    ('match_modes', lambda p, o: (len(o) == 2 and all(
        isinstance(x, ShapeSet) for x in o))),
    ('merge', _mergeable),
)

#: the verbs that need a partner, so a lone selection is not offered them
PARTNER_VERBS = frozenset(verb for verb, _applies in _SELECTION_APPLIES)


_JOURNALED_VERBS = (
    'add', 'import_file', 'remove', 'rename', 'rename_dof', 'link', 'unlink',
    'relink',
    'place', 'set_role', 'set_basis', 'merge',
    'compute_spectra', 'compute_psds', 'compute_cpsds', 'compute_octave',
    'compute_frfs', 'compute_multiple_coherence', 'compute_srs',
    'detect_shocks', 'filter_data', 'truncate_data', 'integrate',
    'differentiate', 'fit_modes', 'generate_rigid_body_modes',
    'author_specification',
    'transform', 'expand', 'project_onto_basis', 'match_modes',
    'extract_sine', 'refresh', 'refresh_stale',
    'generate_report', 'export_report', 'export', 'save', 'duplicate',
)
for _verb in _JOURNALED_VERBS:
    setattr(Project, _verb, _journaled(getattr(Project, _verb)))


#: how each verb rebuilds its object from the source, for `refresh` —
#: the same core calls the verbs make, minus the add
_RECOMPUTE = {
    'compute_spectra': lambda p, s, k: p[s].compute_spectra(),
    'compute_psds': lambda p, s, k: p[s].compute_psds(),
    'compute_cpsds': lambda p, s, k: p[s].compute_cpsds(),
    'compute_frfs': lambda p, s, k: p[s].compute_frfs(
        method=k.get('method', 'Hv')),
    'compute_multiple_coherence':
        lambda p, s, k: p[s].compute_multiple_coherence(),
    'compute_srs': lambda p, s, k: p[s].compute_srs(
        per_octave=k.get('per_octave'), q=k.get('q'),
        kind=k.get('kind', 'maximax')),
    'compute_octave': lambda p, s, k: p[s].to_octave(
        k.get('per_octave', 6)),
    'filter_data': lambda p, s, k: p[s].filter(),
    'truncate_data': lambda p, s, k: p[s].truncate(),
    # `get` with the module default, so a recipe that recorded None
    # stays raw — None is a choice here, not an absence
    'integrate': lambda p, s, k: p[s].integrate(
        k.get('drift_corner', _default_drift())),
    'differentiate': lambda p, s, k: p[s].differentiate(),
    'generate_rigid_body_modes': lambda p, s, k: _rigid_shapes(p[s]),
    'transform': lambda p, s, k: _through(p, s, k, 'to_modal'),
    'expand': lambda p, s, k: _through(p, s, k, 'to_physical'),
    'author_specification': lambda p, s, k: _authored(p, s, k),
}


def _rewritten(spec, draft, comment):
    """The specification as the sheet leaves it: the sheet's own
    form when it holds every channel (fold a target to its
    breakpoints and the object *is* the breakpoints), merged at the
    object's lines when it holds a picked subset."""
    references = spec.reference_dof or list(spec.response_dof)
    autos = {spec.response_dof[i] for i in range(spec.num_records)
             if references[i] == spec.response_dof[i]}
    if set(draft.channels) == autos:
        return draft.make(comment)
    return draft.written_into(spec, comment)


def _authored(project, source, params):
    from .core.author import SpecificationDraft

    draft = SpecificationDraft.from_dict(params['draft'])
    if params.get('into'):
        # the sheet written into the specification again: the same
        # levels land on the same lines, so a refresh changes nothing
        return _rewritten(project[source], draft,
                          f'edited as a sheet on {source}')
    return draft.make(f'written on {source}')


def _through(project, source, params, which):
    from .core import transform

    shapes = params.get('shapes')
    if shapes not in project:
        raise ValueError(f'{source!r} was transformed through {shapes!r}, '
                         'which is gone')
    records = params.get('records')
    if which == 'to_modal':
        result, report = transform.to_modal(project[source],
                                            project[shapes], records)
    else:
        result, report = transform.to_physical(project[source],
                                               project[shapes], records)
    result.transform_report = report
    return result


def _rigid_shapes(geometry):
    from .core.rigid import rigid_body_shapes

    return rigid_body_shapes(
        geometry, geometry.mass_properties
        or geometry.suggest_mass_properties())


def _default_drift():
    from .core.filters import DRIFT_CORNER

    return DRIFT_CORNER


#: what the motion chain's quantities are called in an object's name
_QUANTITY_WORDS = {'acceleration': 'Acceleration',
                   'velocity': 'Velocity', 'length': 'Displacement'}


def _motion_name(source: str, result: Any, fallback: str) -> str:
    """What to call an integrated or differentiated history.

    Named by what it *is*, not by what was done: 'Shock Run Velocity',
    and integrating that gives 'Shock Run Displacement' rather than
    'Shock Run Velocity Displacement' — the trailing quantity word is
    swapped, not stacked. A result of mixed quantities (a record
    carrying both accelerations and velocities) has no one word and
    falls back to the verb.
    """
    dims = {result.known_dim(i) for i in range(result.num_records)}
    if len(dims) != 1:
        return f'{source} {fallback}'
    word = _QUANTITY_WORDS[next(iter(dims))]
    for old in _QUANTITY_WORDS.values():
        if source.endswith(f' {old}'):
            return f'{source[:-len(old)]}{word}'
    return f'{source} {word}'


#: the settings classes a fingerprint can be of, by the name it records
_STATE_CLASSES = {'averaging': 'visualdynamics.core.averaging:Averaging',
                  'filtering': 'visualdynamics.core.filters:Filtering',
                  'truncation': 'visualdynamics.core.truncate:Truncation'}


def _named_state(kind, value):
    """An averaging or filtering fingerprint as {field: value}, whatever
    shape it was recorded in.

    Fingerprints used to be positional — `astuple` — and that made them
    a record of the *class* as much as of the settings: adding `pad` to
    `Averaging` (2026-08-27) lengthened the tuple, so every project
    saved before it opened with a refresh badge on everything derived,
    over a story that said the same thing twice. Nothing had changed but
    the arity.

    Names fix it, and an absent field reads as the class's own default,
    which is right rather than merely convenient: a project saved before
    padding existed was computed unpadded, and `pad=1` is exactly what
    it meant. It is also the rule the `.vdyn` format already states for
    optional fields — absent, not sentinel-valued.

    The values then go through the class itself, because a *rename* is
    the other way a fingerprint drifts without the settings moving:
    'rectangle' became 'boxcar' (2026-08-29, the scipy spelling), the
    loader normalizes the object's own averaging through `Averaging`,
    and a fingerprint recorded before the rename compared unequal to
    the identical computation — every older project opened with a
    refresh badge over a story that named the same window twice
    (Brandon, 2026-08-30). One implementation of the normalization:
    the class's, here as at load.
    """
    import importlib
    from dataclasses import MISSING, asdict, fields

    module, _, name = _STATE_CLASSES[kind].partition(':')
    cls = getattr(importlib.import_module(module), name)
    spec = {field.name: (None if field.default is MISSING else field.default)
            for field in fields(cls)}
    if (kind == 'filtering' and not isinstance(value, dict)
            and len(value) == 2):
        # a positional filtering fingerprint of two values predates
        # 2026-08-28, when `Filtering` was `(corner, order)` and only a
        # low-pass; laid onto today's (low, high, order) it read the
        # corner as `low` and the order as `high`, and the story
        # builder then refused its own fingerprint — a saved shock
        # project raised from `stale()` (2026-09-08)
        corner, order = value
        value = {'low': None, 'high': corner, 'order': order}
    named = dict(value) if isinstance(value, dict) else dict(zip(spec, value))
    named = {field: named.get(field, default)
             for field, default in spec.items()}
    try:
        return asdict(cls(**named))
    except (TypeError, ValueError):
        # a fingerprint the class refuses outright (recorded by some
        # future format this code predates) compares as recorded —
        # absence of evidence is not staleness
        return named


def _state_tuple(state):
    """The fingerprint in a form two of them can be compared in.

    JSON round-trips tuples as lists, so the comparison is shape-blind;
    and an averaging or filtering fingerprint is named first, so one
    recorded before a field existed compares equal to one recorded after
    when nothing about the settings moved. See `_named_state`.
    """
    import json

    kind, value = state
    if kind in _STATE_CLASSES and value is not None:
        value = _named_state(kind, value)
    return json.loads(json.dumps([kind, value]))


def _summarize_state(state):
    kind, value = state
    if kind == 'averaging':
        if value is None:
            return 'no framing'
        named = _named_state(kind, value)
        return (f'{named["frames"]} frames of {named["frame_length"]} '
                f'({named["window"]})')
    if kind == 'shocks':
        return f'{len(value)} window{"s" * (len(value) != 1)}'
    if kind == 'filtering':
        if value is None:
            return 'no filter'
        from .core.filters import Filtering

        named = _named_state(kind, value)
        # rebuilt rather than re-worded: `describe` is the one
        # implementation of a filter said in words
        try:
            return ('a ' + Filtering(low=named['low'], high=named['high'],
                                     order=named['order']).describe()
                    + f', order {named["order"]}')
        except (TypeError, ValueError):
            # a fingerprint the class refuses is still a story to tell,
            # not a reason to refuse the whole staleness reading
            return ('a filter ' + ', '.join(
                f'{field} {named[field]}' for field in ('low', 'high', 'order')
                if named.get(field) is not None))
    if kind == 'truncation':
        if value is None:
            return 'no span'
        from .core.truncate import Truncation

        named = _named_state(kind, value)
        return Truncation(named['start'], named['stop']).describe()
    if kind == 'rigid':
        from .core.rigid import MassProperties

        properties = value.get('properties')
        if properties is None:
            return 'the centroid, unit shapes'
        return MassProperties(**properties).describe()
    return 'its source'


def _state_story(was, now):
    was, now = _state_tuple(was), _state_tuple(now)
    if was[0] == 'content':
        return 'its source was recomputed'
    if was[0] == 'rigid':
        if was[1]['properties'] == now[1]['properties']:
            return 'computed on nodes that have since moved'
        return (f'computed {_summarize_state(was)}; the geometry now '
                f'says {_summarize_state(now)}')
    return (f'computed with {_summarize_state(was)}; the recording '
            f'now says {_summarize_state(now)}')


def _last_window(run, last):
    """The import options for the last `last` seconds of a run: empty
    when the run is no longer than that, so a short run is imported
    whole and the journal line stays the plain one."""
    from .io.rattlesnake import last_window, stream_summary

    streams = stream_summary(run)['streams']
    if not streams:
        return {}
    start = last_window(max(s['seconds'] for s in streams), last)
    return {} if start is None else {'start': start}


def _photos_from(photos) -> Photos:
    """A `Photos` from a folder, one file, or a sequence of files."""
    from .core.photos import FORMATS

    if isinstance(photos, (str, os.PathLike)):
        if os.path.isdir(photos):
            files = [os.path.join(photos, name)
                     for name in sorted(os.listdir(photos))
                     if os.path.splitext(name)[1].lower() in FORMATS]
            if not files:
                raise ValueError(f'{photos} holds no png or jpeg to add')
        else:
            files = [photos]
    else:
        files = [str(p) for p in photos]
    out = Photos()
    for file in files:
        out.add_file(file)
    return out


def random_vibration_run(run: str | os.PathLike,
                         per_octave: int | None = None, *,
                         last: float | None = None,
                         geometry: str | os.PathLike | None = None,
                         length_unit: str | None = None,
                         photos: Any = None) -> Project:
    """A Rattlesnake random vibration run, worked up into a project.

        project = visualdynamics.random_vibration_run('run.nc4')
        project = visualdynamics.random_vibration_run(
            'run.nc4', last=100.0, geometry='article.stp', length_unit='mm',
            photos='setup_photos/')

    Every step the window would take on the way from a controller file
    to a finished project, in the order it takes them: import the run,
    average PSDs from the control time histories, band those onto
    proportional bands — and the specification onto the same bands,
    limits and all, since the report reads the banded measurement
    against the banded requirement (Brandon, 2026-09-18) — and
    measure how much of each response the drives account for. The run
    says it is a random vibration test, so the project comes back
    declared as one.

    The averaging is the Detect answer: the controller's own frame
    length, window and overlap from the file, with the start and the
    count worked out from the record — where the run is at level and
    how many frames that stretch holds — so a script, an import and a
    click on Detect reach the same numbers (Brandon, 2026-09-19).

    The rest is what the window's tree asks for after the run is in,
    given here so a script never has to open it (Brandon,
    2026-09-19): the article's geometry, with its length unit
    declared when the file does not carry one; the setup photographs,
    as a folder of png or jpeg files, one file, or a list of files in
    the order they should appear; and, for a run too long to hold,
    `last` — the seconds before the end to import, the same window the
    import dialog's *Last* field sets, and a run shorter than that is
    taken whole. The geometry and the photographs are linked into the
    run's own group, which is what lets the report read them against
    the channel table.

    Parameters
    ----------
    run : str or os.PathLike
        The controller's `.nc4`.
    per_octave : int, optional
        Bands per octave for the banded PSD and specification; the
        project's default (a sixth) when omitted.
    last : float, optional
        Import only the last `last` seconds of the run's streams.
    geometry : str or os.PathLike, optional
        A geometry file to import and link to the run.
    length_unit : str, optional
        The geometry's length unit, for a file that does not say.
    photos : str, os.PathLike or sequence of them, optional
        A folder of photographs, one photograph, or several.
    """
    project = Project()
    project.import_file(run, **(_last_window(run, last) if last is not None
                                else {}))
    history = next((name for name, obj in project.items()
                    if isinstance(obj, TimeHistory)), None)
    if history is None:
        raise ValueError(f'{run} holds no time data to work up')
    psds = project.compute_psds(history)
    project.compute_octave(psds, per_octave)
    from .core.data import Specification
    specification = next((name for name, obj in project.items()
                          if isinstance(obj, Specification)), None)
    if specification is not None:
        project.compute_octave(specification, per_octave)
    project.compute_multiple_coherence(history)
    extras: list[str] = []
    if geometry is not None:
        options = {} if length_unit is None else {'length_unit': length_unit}
        extras += project.import_file(geometry, **options)
    if photos is not None:
        extras.append(project.add('Photos', _photos_from(photos)))
    if extras:
        project.link(history, *extras)
    return project


def _quiet_and_driven(project: Project) -> tuple[str | None, str]:
    """(the ambient recording, the driven one) among the time histories.

    A system identification is made from two recordings — the room on
    its own, then the room with the shakers running — and the file
    does not say which is which. The quieter one is the ambient, by
    the only measure that is always there: how much the signal moves.
    A file holding one recording has no ambient, and the report's
    signal-to-noise stands down on its own.
    """
    names = [name for name, obj in project.items()
             if isinstance(obj, TimeHistory)]
    if not names:
        raise ValueError('no time data to work up')
    ranked = sorted(names, key=lambda name: float(
        np.std(np.asarray(project[name].ordinate, dtype=float))))
    return (ranked[0] if len(ranked) > 1 else None), ranked[-1]


def work_up_system_id(project: Project) -> tuple[str | None, str]:
    """An imported system identification, worked up in place.

    The seam between reading a file and doing the work, so the work
    can be exercised on a recording built by hand: no file a test can
    commit holds the *two* streams a system identification is made of,
    and a rule that is never exercised is a rule that is not tested
    (2026-09-22).

    Returns the two stream names, the ambient one None where the
    recording holds only the driven stream.
    """
    noise, driven = _quiet_and_driven(project)
    project.project_type = 'System ID'
    driven = project.rename(driven, 'Excitation Time History')
    if noise is not None:
        noise = project.rename(noise, 'Noise Time History')
    # H1: the plant is measured driving through a known excitation, so
    # what noise there is sits on the response
    project.compute_frfs(driven, 'H1')
    project.compute_multiple_coherence(driven)
    project.compute_psds(driven)
    if noise is not None:
        # the ambient borrows the excitation's frames on purpose: the
        # ratio the report reads is only defined on lines both hold
        project[noise].averaging = project[driven].averaging
        project.compute_psds(noise)
    return noise, driven


def system_id_run(run: str | os.PathLike, *,
                  last: float | None = None,
                  geometry: str | os.PathLike | None = None,
                  length_unit: str | None = None,
                  photos: Any = None) -> Project:
    """A Rattlesnake system identification, worked up into a project.

        project = visualdynamics.system_id_run('sysid.nc4')

    The steps the window would take, in its order: import the
    recording, name the two streams for what they are, measure the
    plant from the driven one by H1 — the controller's own estimator,
    the excitation being known and the noise on the response — take
    the multiple coherence, and average a density from each stream on
    **the same frames**, since the signal-to-noise is a ratio of
    densities and a ratio only exists on shared lines. The project
    comes back declared a System ID.

    `last`, `geometry`, `length_unit` and `photos` are
    `random_vibration_run`'s, and mean the same things.
    """
    project = Project()
    project.import_file(run, **(_last_window(run, last) if last is not None
                                else {}))
    try:
        noise, driven = work_up_system_id(project)
    except ValueError as exc:
        raise ValueError(f'{run} holds no time data to work up') from exc
    extras: list[str] = []
    if geometry is not None:
        options = {} if length_unit is None else {'length_unit': length_unit}
        extras += project.import_file(geometry, **options)
    if photos is not None:
        extras.append(project.add('Photos', _photos_from(photos)))
    linked = [name for name in (noise, *extras) if name]
    if linked:
        project.link(driven, *linked)
    return project


class _Ask:
    """The value that means "open a dialog and ask me".

    A sentinel rather than None, because None already means something
    on every one of these keywords: no geometry, no photographs, the
    whole run. `visualdynamics.ASK` is the public name.
    """

    def __repr__(self) -> str:           # pragma: no cover - a label
        return 'visualdynamics.ASK'

    def __bool__(self) -> bool:
        return False


#: pass this where a path goes to be asked for it instead
ASK = _Ask()


def _report_path(run: str | os.PathLike,
                 path: str | os.PathLike | None) -> str:
    """Where one run's report is written.

    A *folder* is taken as a folder — the report lands in it under the
    run's own name — and anything else is the file to write (Brandon,
    2026-09-22). Without a path at all it lands beside the run, which
    is what it always did. A name that does not exist yet and does not
    end in a separator is a file: there is no way to tell a folder
    nobody has made from a file nobody has written, and guessing wrong
    writes a report where nothing will look for it.
    """
    if path is None:
        return os.path.splitext(os.path.expanduser(str(run)))[0] + '.html'
    path = os.path.expanduser(str(path))
    if os.path.isdir(path) or path.endswith((os.sep, '/')):
        stem = os.path.splitext(os.path.basename(os.path.expanduser(
            str(run))))[0]
        return os.path.join(path, stem + '.html')
    return path


def _one_call_reports(run: Any, path: Any, geometry: Any,
                      unit_system: Any, kind: str,
                      work_up: Any) -> Any:
    """What every one-call report function does around its workup.

    The asking, the batch and the destination, in one place because
    they are one rule (PRINCIPLES.md, 9): a run left out is asked for,
    several may be chosen, the geometry is asked for only when the run
    was, and `path` may be a folder each report lands in under its
    run's own name.
    """
    asked = run is ASK
    if asked:
        from .gui.ask import for_runs
        runs = for_runs()
        if not runs:
            raise ValueError('no run chosen')
    else:
        runs = [run]
    # the geometry is asked for only when the run was: a script that
    # names its run and leaves the geometry out means "no geometry",
    # and has meant it since these functions existed. `ASK` says so on
    # purpose either way. Asked once, before the loop: a campaign is a
    # folder of runs on one article (Brandon, 2026-09-22).
    if geometry is ASK or (asked and geometry is None):
        from .gui.ask import for_geometry
        geometry = for_geometry()
    if len(runs) > 1 and path is not None and not (
            os.path.isdir(os.path.expanduser(str(path)))
            or str(path).endswith((os.sep, '/'))):
        raise ValueError(
            f'{len(runs)} runs cannot be written to one file '
            f'{str(path)!r} — give a folder, or no path at all')
    written = []
    for one in runs:
        project = work_up(one, geometry)
        report = project.generate_report(kind, name='Report')
        written.append(project.export_report(
            report, _report_path(one, path), unit_system))
    return written if len(written) > 1 else written[0]


def system_id_report(run: Any = ASK,
                     path: str | os.PathLike | None = None, *,
                     last: float | None = None,
                     geometry: Any = None,
                     photos: Any = None,
                     unit_system: Any = None) -> Any:
    """A Rattlesnake system identification in, an HTML report out.

        visualdynamics.system_id_report('sysid.nc4', 'sysid.html')
        visualdynamics.system_id_report()            # ask for both

    `system_id_run` followed by the System ID report: the measured
    plant as a CMIF, the coherence map, and the signal-to-noise of the
    measurement — where that ratio approaches one, the plant is the
    room.

    Everything about being asked, writing a batch and taking a folder
    is `random_vibration_report`'s, and means the same things: left
    out, the run is asked for and as many may be chosen as there are
    reports wanted; the geometry is asked for once for the whole
    batch when the run was asked for, Cancel meaning none;
    `visualdynamics.ASK` forces either question; and `path` may be the
    file to write or the folder to write into.
    """
    return _one_call_reports(
        run, path, geometry, unit_system, 'sysid',
        lambda one, geo: system_id_run(one, last=last, geometry=geo,
                                       photos=photos))


def random_vibration_report(run: Any = ASK,
                            path: str | os.PathLike | None = None, *,
                            last: float | None = None,
                            geometry: Any = None,
                            photos: Any = None,
                            per_octave: int | None = None,
                            unit_system: Any = None) -> Any:
    """A Rattlesnake random vibration run in, an HTML report out.

        visualdynamics.random_vibration_report('run.nc4', 'report.html')
        visualdynamics.random_vibration_report(
            'run.nc4', 'report.html', last=100.0,
            geometry='article.stp', photos='setup_photos/')
        visualdynamics.random_vibration_report()          # ask for both
        visualdynamics.random_vibration_report('run.nc4', 'reports/')

    **Left out, it asks.** Called with no run, it opens a file dialog
    and takes as many runs as are chosen, writing one report each and
    returning the list; a run chosen that way is asked about its
    geometry too, where Cancel means none. `visualdynamics.ASK` in
    either place forces the question, so a script with a run in hand
    can still be asked for the geometry. A run given without a
    geometry keyword means *no geometry*, as it always has, and
    nothing opens.

    **A folder is a folder.** `path` may be the file to write, or a
    folder the report lands in under the run's own name with an
    `.html` extension. Without a path it lands beside the run. Several
    runs need a folder or no path, never one file name.

    The whole workflow in one call, with nothing to click (Brandon,
    2026-09-19): import the run — or only its last `last` seconds, a
    shorter run taken whole — detect the averaging, compute the PSDs,
    band them and the specification onto octave bands, compute the
    multiple coherence, bring in the geometry and the photographs when
    they are given, generate the Random Vibration report and write it
    as one self-contained HTML file. Returns the path written, or the
    list of them when several runs were chosen. The keywords are
    `random_vibration_run`'s, plus `unit_system` for the units the
    report is written in.

    Everything it does is `random_vibration_run` followed by
    `generate_report` and `export_report`; reach for those instead when
    the project is wanted afterwards — to write the test summary, or to
    save it as `.vdyn`.
    """
    # no length unit reaches the workup: the report draws the geometry
    # as a shape and never states a coordinate or a scale, so declaring
    # what the file's numbers meant changed one label and nothing else
    # (Brandon, 2026-09-22). The `_run` calls still take it, for a
    # project that lives on.
    return _one_call_reports(
        run, path, geometry, unit_system, 'random',
        lambda one, geo: random_vibration_run(one, per_octave, last=last,
                                              geometry=geo, photos=photos))
