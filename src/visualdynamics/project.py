"""The project: every object in a test, and how the objects are grouped.

The desktop app's window holds the same things a `Project` holds: the
named objects, the object groups, which group is the Basis, the project
type and the active geometry. A project built in the app and one built
in a script are therefore the same, and both save to the same `.vdyn`
file. A script uses the app's own verbs (`add`, `link`, `set_basis`,
`rename`, `save`) instead of building dicts and lists by hand.

The functions near the end of this module run a whole workflow in one
call. `random_vibration_report`, for example, reads a Rattlesnake run
and writes an HTML report, and `random_vibration_run` builds the project
that report is made from.
"""

from __future__ import annotations

import contextlib
import os
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from .core.calibration import CalibrationCheck
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
from .core.data_index import DataIndex, IndexEntry
from .core.data_index import digest as digest_file
from .core.data_index import now as _index_now
from .core.field_limit import CONFIDENCE as _FIELD_CONFIDENCE
from .core.field_limit import COVERAGE as _FIELD_COVERAGE
from .core.field_limit import MAX_BREAKPOINTS as _FIELD_BREAKPOINTS
from .core.field_limit import OVERSHOOT_DB as _FIELD_OVERSHOOT
from .core.geometry import Geometry, View
from .core.matches import MatchedModes
from .core.photos import Photos
from .core.prediction import Predictions
from .core.report import Report
from .core.shapes import ShapeSet
from .core.sine import SineLevelSet, SineSweepSpecification
from .names import display_name
from .report.features import FeatureSelection

# An object group: the objects declared to belong together, and what the
# group is — 'Basis', or None for every other group. One field says it
# all: 'side' used to ride beside the role saying the same thing in a
# second vocabulary, and was folded in (Brandon, 2026-08-30); a 'FEM'
# role named the comparison pair until 2026-09-02, when it went too —
# a project knows its Basis, and everything else is other (Brandon).
ObjectGroup = dict[str, Any]

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
              _CoherenceBase, ShapeSet, MatchedModes, Predictions,
              Report)


def type_rank(obj: Any) -> int:
    """The position of an object's type in the order the tree lists types."""
    return next((rank for rank, cls in enumerate(TYPE_ORDER)
                 if isinstance(obj, cls)), len(TYPE_ORDER))


def describe(obj: Any) -> str:
    """A few words on what an object holds: its size, not its name.

    A count of zero is left out of the text.
    """
    if isinstance(obj, Geometry):
        counts = [(obj.num_nodes, 'node'),
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
    elif isinstance(obj, Predictions):
        counts = [(len(obj), 'channel')]
    elif isinstance(obj, Report):
        counts = [(obj.num_blocks, 'block')]
    else:
        return ''
    return ', '.join(f'{n} {word}{"s" * (n != 1)}'
                     for n, word in counts if n)


def holds(objects: Mapping[str, Any]) -> str:
    """What a file held, in words, from the objects read out of it: for
    each, its kind and size (`describe`), and for records the channels
    they span and a time history's sample rate. The data index prints
    this as the file's holdings, so a reader can ask for the file that
    holds a channel.
    """
    parts = []
    for obj in objects.values():
        words = [describe(obj)] if describe(obj) else []
        if isinstance(obj, DataArray) and obj.num_records:
            # the DOF each record measures, which is what a reader asks
            # for a file by (the quantity at a DOF is in the kind)
            dofs = list(dict.fromkeys(str(dof) for dof in obj.response_dof))
            words.append(f'channel {dofs[0]}' if len(dofs) == 1
                         else f'channels {dofs[0]} to {dofs[-1]}')
            if isinstance(obj, TimeHistory) and len(obj.abscissa) > 1:
                try:
                    words.append(f'{obj.sample_rate:.6g} Hz')
                except ValueError:
                    # no one rate to state; the file is indexed all the
                    # same, since what is in it is not the index's to judge
                    words.append('samples not evenly spaced')
        kind = display_name(type(obj).__name__)
        parts.append(f'{kind}: {", ".join(words)}' if words else kind)
    return '; '.join(parts)


def holds_of_names(project: Mapping[str, Any], names: Sequence[str]) -> str:
    """`holds` for the objects of `project` called `names`; refuses an
    empty selection, since an index entry has to say what it holds."""
    if not names:
        raise ValueError('say what the file holds: give holds, or the '
                         'objects read from it')
    return holds({name: project[name] for name in names})


def remap_links(object_groups: Iterable[ObjectGroup],
                mapping: dict[str, str],
                roles_taken: Iterable[str] = ()) -> list[ObjectGroup]:
    """Object groups translated through a {old name: new name} mapping.

    When a project is imported into one that already holds objects,
    the names that clash are renamed, and the groups have to be renamed
    with them or the file's grouping is lost. A member missing from the
    mapping is dropped, and a group left with fewer than two members is
    dropped. A role in `roles_taken` stays with the group that already
    has it, because the Basis belongs to the project, not to the file.
    """
    taken, out = set(roles_taken), []
    for group in object_groups or ():
        members = [mapping[name] for name in group.get('members', ())
                   if name in mapping]
        if len(members) < 2:
            continue
        role = group.get('role')
        role = None if role in taken else role
        if role is not None:
            taken.add(role)
        out.append({'members': members, 'role': role,
                    **({'name': group['name']} if group.get('name') else {})})
    return out


def retarget(obj: Any, mapping: dict[str, str]) -> None:
    """Point an object's references at renamed objects.

    Matched modes name their two shape sets, and a report's blocks name
    the objects they draw from. When one of those objects is renamed,
    the reference has to be changed with it, or the report figure has
    nothing to draw. The object is changed in place.
    """
    follow = getattr(obj, 'rename_source', None)
    if callable(follow):                            # MatchedModes
        for old, new in mapping.items():
            follow(old, new)
    # the type, not the attribute: a Geometry grew a `blocks` of its own
    # (its element groups, now `groups`), and duck-typing walked those instead — a
    # rename then died inside a Qt signal, where the traceback goes
    # nowhere and the rename simply does not happen
    if isinstance(obj, Report):
        for block in obj.blocks:
            for key in ('source', 'geometry', 'dofs_source', 'shapes',
                        'measured'):
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
    ('predictions', 'prediction_sets', Predictions),
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
    """Objects reached by type instead of by name.

        project.geometry            # anywhere in the project
        project.basis.frf           # in the Basis group
        project.basis.psds[0]       # one of several

    A name is whatever an importer or a user called the object, and
    importing a second geometry makes 'Geometry' ambiguous without
    renaming anything. A type does not change, and an editor can
    complete it.

    The singular gives the one object of that kind. When there are
    several it raises AttributeError listing their names, except that
    when exactly one of them is not on octave bands, that one is
    returned. The plural is always a list, so code written for one FRF
    still works when a second one is added. A kind leaves out objects
    of a narrower kind: `time_history` does not return a transient
    specification, and `psds` does not return a specification.
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
        predictions: Predictions
        prediction_sets: list[Predictions]
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
        """The names in scope holding objects of this class.

        Objects of a narrower kind are left out: a transient
        specification is not returned for `TimeHistory`.
        """
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
    """Every object in one test, by name, and how the objects are grouped.

    A Project is a dict from name to object, so `project['FRF']`,
    `list(project)` and `project.items()` give what the tree shows.

    The desktop app holds one of these: the window's `objects`,
    `project_type` and `active_geometry` are properties over it, and its
    buttons call the verbs below. A project built in the app and one
    built in a script are the same object, and each opens in the other.

    Objects are usually reached by type rather than by name, as in
    `project.geometry`, `project.basis.frf` or `project.other.shapes`.
    The singular gives the one object of that kind and raises an error
    when several qualify; the plural is always a list.

    Attributes:
        object_groups: The groups objects have been linked into, each
            `{'members': [...], 'role': 'Basis' | None}` with an
            optional 'name'. Objects are grouped only by `link`, never
            by matching their names.
        project_type: The kind of test: 'Modal Test', 'Random
            Vibration', 'Transient', 'Shock', 'Sine Sweep', 'Random and
            Sine' or 'System ID'. It sets the report template and the
            empty slots the tree shows.
        active_geometry: The name of the geometry data is drawn on when
            nothing else is named.
        name: The project's name, used as the title of a saved `.vdyn`
            and of a rendered report.
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
        predictions: Predictions
        prediction_sets: list[Predictions]
        report: Report
        reports: list[Report]

    def __init__(self, name: str = 'Project',
                 objects: dict[str, Any] | None = None,
                 active_geometry: str | None = None,
                 project_type: str | None = None,
                 object_groups: Iterable[ObjectGroup] | None = None,
                 provenance: dict[str, dict[str, Any]] | None = None,
                 data_index: DataIndex | Iterable[Mapping[str, Any]]
                 | None = None
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
        # ... and, since 2026-09-30, an optional 'name': what the group
        # is called, which an ESCDF activity needs and a tree can show
        self.object_groups: list[dict[str, Any]] = [
            {'members': list(group['members']),
             'role': 'Basis' if (group.get('role') == 'Basis'
                                 or group.get('side') == 'experimental'
                                 and not group.get('role')) else None,
             **({'name': str(group['name'])} if group.get('name') else {})}
            for group in (object_groups or [])]
        #: the roles the last link moved, (history, channel, was, now)
        #: — a front end says so rather than letting FRFs move silently
        self.last_role_changes: list = []
        #: how each derived object was computed — {name: {'verb',
        #: 'source', 'params', 'state'}} with 'state' the fingerprint
        #: of the analysis settings read at compute time. Staleness is
        #: the recorded state disagreeing with the source's current
        #: one; a mismatch is exactly the recompute the refresh badge
        #: offers (Brandon, 2026-08-23: explicit, never automatic —
        #: a report must not rewrite itself).
        self.provenance: dict[str, dict[str, Any]] = dict(provenance or {})
        #: every file the project rests on, by name, size and SHA-256:
        #: each file imported, and each document attached (`index_file`).
        #: One index, whatever adds to it; saved in the `.vdyn`, and
        #: printed by the report feature `data_index`
        self.data_index: DataIndex = (
            data_index if isinstance(data_index, DataIndex)
            else DataIndex.from_records(data_index))
        #: the file this project was last opened from or saved to, or
        #: None: where a check of the data index looks for moved files,
        #: and the file whose checksum the report states
        self.file: str | None = None
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
        #: (what the last settling call was, the line it wrote): a
        #: settling `record_call` of the same call replaces that line
        self._settling: tuple[tuple, str] | None = None

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

    def grouped_names(self) -> list[tuple[ObjectGroup | None, list[str]]]:
        """[(group or None, [names])] in the order the tree shows them.

        The Basis group comes first, then the other object groups, then
        the unlinked objects under None. Within each, objects are in the
        tree's type order, and objects of one type are in the order they
        were added.
        """
        def ordered(names: Iterable[str]) -> list[str]:
            return sorted(names, key=lambda name: type_rank(self[name]))

        groups = sorted(self.object_groups, key=lambda g: g['role'] != 'Basis')
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
        """Add an object under a unique name and return the name used.

        A name that is already taken gets a numbered suffix, the same as
        importing a file twice in the app; nothing is overwritten. The
        first geometry added becomes the active geometry.

        Parameters
        ----------
        name : str
            What to call the object.
        obj : object
            Any object the project can hold.

        Returns
        -------
        str
            The name the object was added under, unique within the
            project.
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
        """Copy objects and add the copies beside them, as Copy and Paste
        do in the tree. Returns the names added.

        Each copy is named '<name> copy', with a number added when that
        name is taken. A copy's arrays are its own, so editing one leaves
        the other unchanged. Links and provenance stay with the
        originals: a copy is in no object group and has no record of how
        it was computed.

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
        """Import a file into this project and return the names added.

        The file can be anything visualdynamics reads: a geometry, a
        Rattlesnake run or a saved project. A saved project brings its
        object groups, which follow the objects even when a name clash
        renamed them, and when this project is empty it also brings its
        name, type and active geometry. When a reader returns several
        objects, each is named by its type the way the tree shows it (a
        reader's key such as 'Modal_frf' becomes 'FRF'), and if one of
        them is a channel table they are all linked into one group.

        A file that records what kind of test it was sets the project
        type. A controller's save records which environment ran, and
        importing it here sets the type the same way importing it in the
        app does.

        `options` are passed to the format's reader, for example an
        Exodus file's `steps='time'` and `nodes=[...]` or a geometry's
        `length_unit='m'`, so a script can give what the app asks for in
        a dialog.

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
        # the file's checksum before anything is added: a file that
        # cannot be hashed (gone since it was read) refuses the import
        # whole rather than leave objects the index cannot trace.
        # Opening a project into an empty one is opening it, not
        # importing a file it rests on; its own index comes with it
        indexed = not (isinstance(result, Project) and was_empty)
        local = os.path.expanduser(str(path))
        checksum = digest_file(local) if indexed else None
        self.project_type = project_type_of(str(path)) or self.project_type
        if isinstance(result, Project):
            mapping = {name: self.add(name, obj)
                       for name, obj in result.items()}
            self.data_index.adopt(result.data_index, mapping)
            if indexed:
                self._index_import(local, result, list(mapping.values()),
                                   checksum)
            if any(old != new for old, new in mapping.items()):
                # a clash renamed something: the references the
                # imported objects carry follow it
                for obj in result.values():
                    retarget(obj, mapping)
            self.object_groups += remap_links(
                result.object_groups, mapping,
                {group['role'] for group in self.object_groups if group['role']})
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
        else:
            added = [self.add(display_name(type(result).__name__), result)]
        self._index_import(local, result, added, checksum)
        return added

    def _index_import(self, path: str, result: Any, added: Sequence[str],
                      checksum: tuple[int, str]) -> IndexEntry:
        """Record an imported file in the data index: `result` is what
        the reader returned, `added` the names its objects took, and
        `checksum` the file's (size, SHA-256), taken before the objects
        were added. The import's one implementation of the entry, for
        `import_file` and for the app's own import alike."""
        from .io import format_of

        if isinstance(result, Project):
            said = f'Visual Dynamics project: {len(result)} objects'
        else:
            said = holds(result if isinstance(result, dict)
                         else {'': result})
        return self.data_index.add(
            path, said or 'nothing Visual Dynamics reads',
            format=format_of(path), objects=added, checksum=checksum)

    def remove(self, *names: str) -> None:
        """Delete objects and take them out of every object group.

        A name not in the project is ignored. When the active geometry
        is removed, the first remaining geometry becomes the active one.

        Parameters
        ----------
        *names : str
            The objects to act on, by name.

        Returns
        -------
        None
        """
        # a file's entry goes with the last object read from it, unless
        # an object still here was derived from one of them: a PSD
        # kept after its time history was removed still rests on the
        # file. Asked before the removal, while the names still resolve
        gone = [name for name in names if name in self]
        staying = [name for name in self if name not in gone]
        self.data_index.forget_objects(gone, keep=lambda entry: any(
            self._descends_from(name, source) for name in staying
            for source in entry.objects))
        for name in names:
            self.pop(name, None)
            if self.active_geometry == name:
                self.active_geometry = next(
                    (n for n, obj in self.items()
                     if isinstance(obj, Geometry)), None)
        self._prune_links()

    def rename(self, old: str, new: str) -> str:
        """Rename an object and change every reference to it.

        Object groups, matched-mode sets, report blocks, provenance
        records and the active geometry all refer to objects by name,
        and each is updated so that no report figure is left pointing
        at the old name. An unknown `old` raises KeyError; a `new` that
        is empty or already in use raises ValueError.

        Parameters
        ----------
        old : str
            The current name.
        new : str
            The name to give it.

        Returns
        -------
        str
            The new name.
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
        for group in self.object_groups:
            group['members'] = [new if member == old else member
                                for member in group['members']]
        for obj in self.values():
            retarget(obj, {old: new})
        if old in self.provenance:
            self.provenance[new] = self.provenance.pop(old)
        for record in self.provenance.values():
            if record.get('source') == old:
                record['source'] = new
        self.data_index.rename_object(old, new)
        return new

    def rename_dof(self, source: Any, old: str, new: str,
                   quantity: str | None = None) -> list[str]:
        """Correct a channel's coordinate on an object and on every object
        derived from it. In the app, double-click a row or reference
        column of the grid and type.

        Only the channel changes, not the point: correcting a force
        labeled at the wrong node leaves the accelerometer at that node
        where it is. Spectra computed from a time history carry its
        channels, so the correction is also applied to every object
        derived from the source, through any number of steps, instead of
        leaving each one to be fixed by hand or recomputed. A derived
        object that does not carry the channel is left alone. An object
        with no coordinates raises TypeError.

        Parameters
        ----------
        source : str or object
            The object whose grid row or column was edited.
        old : str
            The coordinate as it is, for example '101Z+'.
        new : str
            The coordinate to give it, normalized the same way as every
            coordinate.
        quantity : str, optional
            Which channel at `old`, for example 'acceleration' or
            'force'. Every channel at the coordinate when omitted.

        Returns
        -------
        list of str
            The names of the objects changed, the source first.
        """
        # The channel, not the point (Brandon, 2026-09-06): the other
        # channel at that node is changed explicitly if it should be.
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

    def link(self, *names: str, role: str | None = None,
             name: str | None = None) -> list[str]:
        """Link objects into one group, merging any groups they are
        already in. Returns the group's members.

        A group holds at most one geometry, and its members are checked
        against it. A member that names nodes the geometry does not have
        is still linked, and the app marks its row red so the mismatch
        can be seen; a test often has drive points, control coordinates
        or virtual channels the model does not draw. A modal object with
        no shape set to be drawn against is refused with ValueError, as
        are fewer than two names and a second geometry.

        Parameters
        ----------
        *names : str
            The objects to link, by name.
        role : str, optional
            The role to give the group: 'Basis', or None. Left out, a
            group being merged into keeps the role it had.
        name : str, optional
            What to call the group. A named group is exported to the
            Engineering Sciences Common Data Format as an activity with
            that name. Left out, a group being merged into keeps the
            name it had.

        Returns
        -------
        list of str
            The group's members after linking.
        """
        # A group can be named (Brandon, 2026-09-30): an activity in the
        # Engineering Sciences Common Data Format has a name, and a named
        # group exports as one that keeps its identity.
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
        for group in self.object_groups:
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
        kept_name = name or next(
            (group.get('name') for group in self.object_groups
             if any(n in group['members'] for n in names) and group.get('name')),
            None)
        self.object_groups = kept + [{'members': merged, 'role': role,
                              **({'name': kept_name} if kept_name else {})}]
        if role is not None:
            self.set_role(merged[0], role)
        # a channel table joining time data it describes brings its
        # roles; the history is the truth from then on
        self._sync_linked_roles(merged)
        return merged

    def unlink(self, *names: str) -> None:
        """Take objects out of their object groups. A group left with one
        member is dissolved unless it has a role.

        Parameters
        ----------
        *names : str
            The objects to act on, by name.

        Returns
        -------
        None
        """
        names = {str(name) for name in names}
        # the group's name kept with it: rebuilt as members and role
        # alone, every group lost its name, the untouched ones too
        # (2026-10-13, found by the docstring pass)
        self.object_groups = [
            {**group, 'members': kept}
            for group, kept in
            ((group, [member for member in group['members']
                      if member not in names])
             for group in self.object_groups)
            # A *named* group holds one object quite happily: the FEM
            # group of a modal test starts as a lone geometry, and
            # dissolving it is why the group could never be built up one
            # drag at a time.
            if len(kept) > 1 or (kept and group['role'])]

    def relink(self, name: Any, target: Any = None) -> list[str]:
        """Move one object into the group holding `target`.

        `link` merges the groups of all its arguments, which is right for
        saying two objects belong together and wrong for moving one
        object: linking a geometry into the other group would bring its
        whole group with it. This method takes the object out of its
        group first, so only it moves. A `target` of None only takes it
        out.

        The group it joins keeps its role, so an object moved into the
        Basis becomes part of the Basis. A move that `link` refuses
        leaves the groups as they were and raises the same error.

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
            The members of the group the object ends up in, or an empty
            list when `target` is None.
        """
        name = self.name_of(name)
        if target is None:
            self.unlink(name)
            return []
        target = self.name_of(target)
        if target == name:
            raise ValueError('an object is already in its own group')
        members = self.object_group_of(target) or [target]
        if name in members:
            return list(members)
        role = self.role_of(target)
        # `link` refuses a second geometry or a member the group's
        # geometry cannot carry, and by then the object has already left
        # where it was: a refused move must leave the links untouched
        # rather than half applied.
        before = [dict(group) for group in self.object_groups]
        self.unlink(name)
        try:
            # `name` last: it joins the group it landed in, and a group
            # reads in the order its members arrived
            return self.link(*members, name, role=role)
        except (ValueError, KeyError):
            self.object_groups = before
            raise

    def name_object_group(self, member: str, name: str | None) -> list[str]:
        """Name the object group an object belongs to, or remove its name.

        An object in no group raises KeyError.

        Parameters
        ----------
        member : str
            Any member of the group, by name.
        name : str or None
            The group's new name; None removes it.

        Returns
        -------
        list of str
            The group's members.
        """
        for group in self.object_groups:
            if member in group['members']:
                if name:
                    group['name'] = str(name)
                else:
                    group.pop('name', None)
                return list(group['members'])
        raise KeyError(f'{member!r} is in no object group')

    def object_group_of(self, name: Any) -> list[str] | None:
        """The members linked with `name`, or None.

        Parameters
        ----------
        name : str or object
            The object to look up.

        Returns
        -------
        list of str, or None
            The names sharing its object group, or None when it is in
            no group.
        """
        group = self._group(self.name_of(name))
        return None if group is None else list(group['members'])

    def role_of(self, name: Any) -> str | None:
        """'Basis' for an object in the Basis group, otherwise None.

        Parameters
        ----------
        name : str or object
            The object to look up.

        Returns
        -------
        str or None
            Its object group's role, or None if it has none.
        """
        group = self._group(self.name_of(name))
        return None if group is None else group['role']

    def placed(self) -> dict[str, list[str]]:
        """{role: [members]}: the objects in each group that has a role.

        Nothing is inferred here. An object that has not been placed is
        in no group with a role, and that is what makes the tree's gray
        slots meaningful: a slot is filled only by an object in its own
        group, so a modal test whose only geometry is the model's still
        shows a slot for the measured one.
        """
        out: dict[str, list[str]] = {}
        for group in self.object_groups:
            if group['role']:
                out.setdefault(group['role'],
                               []).extend(group['members'])
        return out

    def sides(self) -> dict[str, list[str]]:
        """{side: [object names]} for the project type's slots.

        The Basis is listed under its role. Every member of every other
        group is listed under `OTHER_SIDE`, because that side has no role
        of its own and is read from the groups instead. The tree's gray
        slots and `missing` are computed from this.

        Returns
        -------
        dict of str to list of str
        """
        from .core.report import OTHER_SIDE

        sides = dict(self.placed())
        others = [name for group in self.object_groups if group['role'] != 'Basis'
                  for name in group['members']]
        if others:
            sides[OTHER_SIDE] = others
        return sides

    def missing(self) -> list[tuple[str, type, str, int, bool, str | None]]:
        """The project type's empty slots, the ones the tree shows gray.

        Each is a tuple (label, class, icon key, ordinal, optional,
        side), as `project_expectations` gives them. A project with no
        type expects nothing. Until a Basis is set, every object counts
        toward every slot: the app places objects as they arrive, a
        script may never place them, and a script's project with
        everything in it and no groups should not show every slot empty.

        Returns
        -------
        list of tuple
        """
        from .core.report import missing_expectations

        if not self.project_type:
            return []
        sides = self.sides()
        return missing_expectations(self.project_type, dict(self.items()),
                                    sides if 'Basis' in sides else None)

    def object_group_with_role(self, role: str) -> ObjectGroup | None:
        """The object group carrying a role, or None.

        Parameters
        ----------
        role : str
            Which group to fetch. 'Basis' is the only role.

        Returns
        -------
        ObjectGroup or None
            That group, or None when no group has the role.
        """
        return next((group for group in self.object_groups
                     if group['role'] == role), None)

    def place(self, name: Any, role: str) -> list[str]:
        """Put one object into the Basis or the other group, creating the
        group if needed. Returns the group's members.

        Unlike `link`, this takes a single object. If a group needed two
        members before it could exist, a pair sorted into the wrong
        group could not be moved across one object at a time. A move
        that is refused leaves the groups as they were.

        Parameters
        ----------
        name : str or object
            The object being placed.
        role : str or None
            'Basis', or None for the other group: the one group that is
            not the Basis, created if there is none yet. When there are
            several groups besides the Basis there is no single other
            group, and this raises ValueError; use `relink` onto a
            member of the group meant.

        Returns
        -------
        list of str
            The group's members.
        """
        name = self.name_of(name)
        if role is None:
            others = [g for g in self.object_groups if g['role'] != 'Basis'
                      and name not in g['members']]
            mine = self._group(name)
            if mine is not None and mine['role'] != 'Basis':
                return list(mine['members'])
            if len(others) > 1:
                raise ValueError(
                    f'{len(others)} groups besides the Basis — drop '
                    'onto a member of the one meant')
            before = [dict(existing) for existing in self.object_groups]
            self.unlink(name)
            try:
                if not others:
                    self.object_groups = self.object_groups + [
                        {'members': [name], 'role': None}]
                    return [name]
                return self.link(*others[0]['members'], name)
            except (ValueError, KeyError):
                self.object_groups = before
                raise
        group = self.object_group_with_role(role)
        if group is not None and name in group['members']:
            return list(group['members'])
        before = [dict(existing) for existing in self.object_groups]
        self.unlink(name)
        group = self.object_group_with_role(role)
        try:
            if group is None:
                self.object_groups = self.object_groups + [
                    {'members': [name], 'role': None}]
                self.set_role(name, role)
                return [name]
            return self.link(*group['members'], name, role=role)
        except (ValueError, KeyError):
            self.object_groups = before
            raise

    def set_channel_role(self, source: Any, dof: str, quantity: str,
                         role: str) -> list[str]:
        """Give one channel the role reference, response or monitor, on a
        time history or a channel table and on whatever is linked with it.

        The time history's roles are the ones FRFs and multiple
        coherence use. A channel table linked with a history whose
        channels it describes shows the history's roles, so a change
        made on either is made on both. A channel table with no such
        history keeps its own roles. An unknown role or channel raises
        ValueError, and an object that is neither raises TypeError.

        Parameters
        ----------
        source : str or object
            The time history or channel table, by name or as the object.
        dof : str
            The channel's degree of freedom, for example '101Z+'.
        quantity : str
            What it measures, as a history's `ordinate_dim` or a
            table's channel type names it ('acceleration', 'force', or
            'length' for a displacement).
        role : str
            'reference', 'response' or 'monitor'.

        Returns
        -------
        list of str
            The objects whose roles changed.
        """
        # The history's role is the truth (Brandon, 2026-09-25: "the
        # channel table is just a pointer to the linked time data's
        # role").
        from .core.channel_table import ROLES

        if role not in ROLES:
            raise ValueError(f'{role!r} is not a role: ' + ', '.join(ROLES))
        name = self.name_of(source)
        obj = self[name]
        asked = (str(dof), str(quantity))
        if isinstance(obj, TimeHistory):
            targets = [(name, asked)]
        elif isinstance(obj, ChannelTable):
            rows = [row for row, own in enumerate(_table_identities(obj, None))
                    if own == asked]
            if not rows:
                raise ValueError(f'{name} has no channel {dof} ({quantity})')
            # the table row names a channel of each linked history —
            # through its channel type, or its DOF alone when the type
            # is blank and the DOF has one channel there
            targets = [(other, mapped[row]) for other, mapped
                       in self._role_partners(name) for row in rows
                       if mapped[row] is not None]
            if not targets:
                # no time data it describes: the table's own record
                for row in rows:
                    obj.set_cell('role', row, role)
                return [name]
        else:
            raise TypeError(f'{name!r} is neither time data nor a channel '
                            'table')
        for history_name, identity in targets:
            if identity not in self[history_name].channel_identities():
                raise ValueError(f'{history_name} has no channel {identity[0]} '
                                 f'({identity[1]})')
        changed = []
        for history_name, identity in targets:
            history = self[history_name]
            roles = history.channel_roles()
            roles[identity] = role
            history.roles = roles
            changed.append(history_name)
            changed += self._push_roles(history_name)
        return list(dict.fromkeys(changed))

    def transformed_rows(self, name: Any) -> list[tuple[str, str, list[str]]]:
        """Which of a data object's response DOFs are rows of a response
        transformation in its channel table.

        A transformed environment's specification and spectra are over
        the matrix's rows, which the controller's file names only by
        number, so they import as '1', '2', ... (`io.rattlesnake`). This
        ties such a row back to the matrix: the window's note on the
        object, and a script's.

        Parameters
        ----------
        name : str or object
            The data object.

        Returns
        -------
        list of (str, str, list of str)
            `(table name, environment, rows)` for each environment of
            the linked channel table (`channel_table_for`) whose
            response transformation has a row among the object's
            response DOFs, in the table's order. Empty when none has.
        """
        name = self.name_of(name)
        dofs = getattr(self.get(name), 'response_dof', None)
        found = self.channel_table_for(name)
        if dofs is None or found is None:
            return []
        table_name, table = found
        asked = set(dofs)
        out = []
        for env_name, environment in table.environments.items():
            matrix = environment.response_transformation
            if matrix is None:
                continue
            rows = [str(k + 1) for k in range(matrix.shape[0])]
            hit = [row for row in rows if row in asked]
            if hit:
                out.append((table_name, env_name, hit))
        return out

    def _role_partners(self, name: str) -> list[tuple[str, list]]:
        """The objects in `name`'s group on the other side of the
        history/table pairing, each with the table's rows mapped to the
        history's channels — only the compatible ones: a table none of
        whose rows names one of the history's channels describes some
        other recording, and is left alone."""
        group = self.object_group_of(name) or []
        obj = self[name]
        out = []
        for other in group:
            if other == name:
                continue
            partner = self[other]
            if isinstance(obj, TimeHistory) and isinstance(partner, ChannelTable):
                history, table = obj, partner
            elif isinstance(obj, ChannelTable) and isinstance(partner,
                                                                TimeHistory):
                history, table = partner, obj
            else:
                continue
            rows = _table_identities(table, history)
            if any(found is not None for found in rows):
                out.append((other, rows))
        return out

    def _push_roles(self, history_name: str) -> list[str]:
        """Show a history's roles on every compatible channel table
        linked with it; returns the tables changed."""
        roles = self[history_name].channel_roles()
        changed = []
        for table_name, rows in self._role_partners(history_name):
            table = self[table_name]
            current = table.roles()
            for row, identity in enumerate(rows):
                if identity is not None and current[row] != roles[identity]:
                    table.set_cell('role', row, roles[identity])
                    if table_name not in changed:
                        changed.append(table_name)
        return changed

    def _sync_linked_roles(self, members: Sequence[str]
                           ) -> list[tuple[str, tuple[str, str], str, str]]:
        """Linking is the moment a channel table starts describing time
        data, and it is treated as an import is (Brandon, 2026-09-25):
        every role the table *declares* is given to the history's channel
        it names; a channel the table leaves blank keeps the history's
        role. From then on the history's roles are the truth and the
        table shows them — so a table linked after the fact, with the
        monitors marked on it, is not overwritten by the roles the data
        already had.

        Returns the roles the link changed, (history, channel, was,
        now), so the front end can say so: a role moving silently would
        move FRFs silently.
        """
        changes = []
        for name in members:
            history = self.get(name)
            if not isinstance(history, TimeHistory):
                continue
            for table_name, rows in self._role_partners(name):
                stated = self[table_name].roles()
                current = history.channel_roles()
                declared = {identity: stated[row]
                            for row, identity in enumerate(rows)
                            if identity is not None and stated[row]}
                moved = {identity: role for identity, role in declared.items()
                         if current[identity] != role}
                if declared and (moved or history.roles is None):
                    history.roles = {**current, **declared}
                changes += [(name, identity, current[identity], role)
                            for identity, role in moved.items()]
            self._push_roles(name)
        self.last_role_changes = changes
        return changes

    def set_role(self, name: str, role: str | None) -> None:
        """Set the role of the group an object is in. The Basis is
        unique, so giving it to one group takes it from any other.

        An object in no group raises ValueError.

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
            raise ValueError(f'{name!r} is not in an object group')
        if role is not None:
            for other in self.object_groups:
                if other is not group and other['role'] == role:
                    other['role'] = None
        group['role'] = role

    def set_basis(self, *names: Any) -> list[str]:
        """Set the Basis for comparisons, the group the other objects are
        compared against.

        Comparisons are made in the Basis group's DOFs, its modes are
        the rows of the MAC, and its frequencies are the baseline for
        frequency error. One name marks that object's group as the
        Basis; several names are linked first.

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
        """The Basis group, reached by type, as in `project.basis.frf`.

        It is empty, and false, when no group is the Basis, so
        `if project.basis:` works as it reads. Its member names are
        `project.basis.names`.
        """
        # [] not None: None is the whole project, and a project with
        # no Basis declared has an *empty* one, not every object in it
        members = next((group['members'] for group in self.object_groups
                        if group['role'] == 'Basis'), [])
        return Selection(self, members, 'the Basis group')

    @property
    def object_group_selections(self) -> list[Selection]:
        """Every object group, the Basis first, each reached by type."""
        return [Selection(self, group['members'],
                          f'the {group["role"] or "linked"} group')
                for group in sorted(self.object_groups,
                                    key=lambda g: g['role'] != 'Basis')]

    @property
    def other(self) -> Selection:
        """The one object group that is not the Basis, usually the model
        side of a correlation.

        Empty when there is none. Raises AttributeError when there are
        several; use `object_group_selections` then.
        """
        others = [group for group in self.object_groups if group['role'] != 'Basis']
        if len(others) == 1:
            return Selection(self, others[0]['members'], 'the other group')
        if not others:
            return Selection(self, [], 'the other group')
        raise AttributeError(
            f'{len(others)} groups besides the Basis — use .object_group_selections[i]')

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

    def modal_basis(self, name: Any,
                    records: Sequence[int] | None = None) -> str | None:
        """The shape set an object's modal records are coordinates of,
        by name: the set whose shapes turn them into motion.

        Read from where the records came from. Modal responses computed
        from forces (`compute_modal_responses`) name their set in each
        comment; a transform's result names it in its provenance; an
        object derived from either (filtered, decimated, cropped) keeps
        its source's set. None when nothing says, when the records named
        come from two sets, or when the set is gone or has fewer modes
        than the records.

        Parameters
        ----------
        name : str or object
            The object, by name or as the object itself.
        records : sequence of int, optional
            Read only these records; every record at a modal DOF when
            omitted.

        Returns
        -------
        str or None
            The shape set's name.
        """
        from .core.validate import modal_coordinate

        name = self.name_of(name)
        data = self[name]
        if not isinstance(data, DataArray):
            return None
        picked = (range(data.num_records) if records is None else records)
        modal = [i for i in picked
                 if modal_coordinate(data.response_dof[i]) is not None]
        if not modal:
            return None
        named = set()
        for i in modal:
            label, _colon, rest = (data.comment[i] or '').partition(': ')
            if rest.startswith('modal ') and isinstance(self.get(label),
                                                        ShapeSet):
                named.add(label)
        if not named:
            # a transform's result, or something derived from one
            seen, step = set(), name
            while step not in seen and step in self.provenance:
                seen.add(step)
                entry = self.provenance[step]
                shapes = entry.get('params', {}).get('shapes')
                if entry.get('verb') == 'transform' and shapes:
                    named.add(shapes)
                    break
                step = entry.get('source')
        if len(named) != 1:
            return None
        basis = named.pop()
        modes = self.get(basis)
        highest = max(modal_coordinate(data.response_dof[i]) for i in modal)
        if not isinstance(modes, ShapeSet) or modes.num_shapes < highest:
            return None
        return basis

    def geometry_for(self, name: Any) -> tuple[str, Geometry] | None:
        """(name, geometry) for the geometry an object is drawn on and
        checked against: the one in its group, otherwise the active one.

        Parameters
        ----------
        name : str or object
            The object whose geometry is wanted.

        Returns
        -------
        tuple of (str, Geometry), or None
            The geometry's name and the geometry itself, or None when
            there is no geometry in the object's group and no active
            geometry.
        """
        name = self.name_of(name)
        for member in self.object_group_of(name) or ():
            if isinstance(self.get(member), Geometry):
                return member, self[member]
        active = self.active_geometry
        if active is not None and active in self:
            return active, self[active]
        return None

    def sensors(self, name: Any) -> tuple[list, list[str]]:
        """The sensors a channel table places on its geometry, and notes on
        what could not be placed (`core.sensors.place`): the bodies the
        3-D view draws when Sensors is on.

        Parameters
        ----------
        name : str or object
            The channel table.

        Returns
        -------
        sensors : list of Sensor
            One per physical sensor: its node, model, catalog entry,
            center and frame.
        notes : list of str
            Rows left out and why, models drawn as generic bodies, and
            triax legs the table gets wrong.

        Raises
        ------
        TypeError
            If the object is not a channel table.
        ValueError
            If there is no geometry to place them on.
        """
        from .core.sensors import place

        name = self.name_of(name)
        table = self[name]
        if not isinstance(table, ChannelTable):
            raise TypeError(f'{name!r} is not a channel table')
        home = self.geometry_for(name)
        if home is None:
            raise ValueError(f'{name!r} has no geometry — link one')
        return place(table, home[1])

    def channel_table_for(self, name: Any) -> tuple[str, ChannelTable] | None:
        """(name, table) for the channel table linked with an object:
        the one lookup the overload check and the shape values share.

        For a time history, the first table in its group with a row
        naming one of its channels: a table in the group that names
        none of them describes some other recording and is passed over,
        as the role pairing passes it over. For anything else (a shape
        set, whose values are read at a table's channels), the first
        table in its group. A table has none of its own.

        Parameters
        ----------
        name : str or object
            The object.

        Returns
        -------
        tuple of (str, ChannelTable), or None
            The table's name and the table, or None when no linked table
            fits.
        """
        # one lookup for both readers, so the overload check and the
        # shape values cannot find different tables for one object
        name = self.name_of(name)
        obj = self.get(name)
        if isinstance(obj, TimeHistory):
            for other, _rows in self._role_partners(name):
                return other, self[other]
            return None
        if obj is None or isinstance(obj, ChannelTable):
            return None
        for member in self.object_group_of(name) or []:
            if member != name and isinstance(self[member], ChannelTable):
                return member, self[member]
        return None

    def _group(self, name: str) -> ObjectGroup | None:
        return next((group for group in self.object_groups
                     if name in group['members']), None)

    def absorb_links(self, groups: Iterable[ObjectGroup]) -> None:
        """Add the object groups of a project being imported.

        By this point the arriving objects have already been placed in
        groups one at a time by the project type's rules, without the
        file's own grouping to go on. The file's groups are added last,
        and where an object is in two groups the later one is kept, so
        the file's grouping replaces the earlier placement.

        Parameters
        ----------
        groups : iterable of ObjectGroup
            Object groups from another project, merged into this one's.

        Returns
        -------
        None
        """
        self.object_groups = list(self.object_groups) + [dict(group) for group in groups]
        self._prune_links()

    def _prune_links(self) -> None:
        """Members that no longer exist go, and so do groups left with
        nothing in them.

        **An object belongs to one group**, and this is where that is
        made true. Everything downstream assumes it: `object_group_of` answers
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
        for group in reversed(self.object_groups):
            kept = [member for member in group['members']
                    if member in self and member not in seen]
            seen.update(kept)
            groups.append({**group, 'members': kept})
        self.object_groups = [group for group in reversed(groups)
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
        one-line summary, so a script can find what can be done with it.

        The table of which verb applies to what is the same one the
        app's bar reads, so the two always agree. Each summary is the
        first paragraph of the verb's own docstring, so it also matches
        the API reference.

        Parameters
        ----------
        source : str or object, optional
            The object to ask about. A name is looked up in the project;
            an object is checked as it is, whether or not it has been
            added, so a result can be asked about before it is kept.
            Omitted, every processing verb is listed.

        Returns
        -------
        list of (str, str)
            ``(verb, summary)`` pairs, in the order the verbs are
            declared. Call a verb as ``getattr(project, verb)``.
        """
        # Brandon, 2026-08-31: a flat method list says nothing about
        # what `integrate` is *for*, so each verb comes with its reading.
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
        """The processing verbs a selection can act on, each with its
        one-line summary. This is what the app's bar offers.

        For one object, these are its own verbs less the ones that need
        a second object (`transform` needs a shape set beside the
        record). For several objects, these are the verbs that apply to
        exactly that combination: a record and a shape set can be
        transformed or expanded, two shape sets on two geometries can be
        projected, and objects of one type can be merged. Verbs that
        apply to only one of the objects are not listed. The table is
        the same one `verbs` reads, so the bar and the API agree.

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
        # Brandon, 2026-09-04: every act on the bar, none behind a menu.
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
        """Compute spectra from a time history's averages, as Compute
        Spectra in the averaging view does.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
        """
        source = self.name_of(source)
        return self._derive(source, self[source].compute_spectra(),
                            f'{source} Spectra', recipe=('compute_spectra', {}))

    def compute_psds(self, source: Any) -> str:
        """Compute PSDs from a time history's averages (Compute PSDs).

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
        """
        source = self.name_of(source)
        return self._derive(source, self[source].compute_psds(),
                            f'{source} PSDs', recipe=('compute_psds', {}))

    def averaging_of(self, name: Any) -> Any:
        """The `Averaging` a computed spectrum was made with, from its
        provenance, or None when it records none (an imported spectrum).
        A spectrum computed from a record with no averaging set was each
        record whole under a rectangle, and that is what is returned.

        A spectrum does not carry its window, and a tone read off it
        needs one (`Psd.read_tone`): the window sets how far the tone leaks
        and how tall its spike stands. The provenance holds the
        averaging as it was when the spectrum was computed, which is the
        one that made it even if the record's own has moved since. An
        FRF and a coherence are read the same way: the transmissibility
        behind an FRF (`read_transmissibility`) is read from the frames
        the FRF was.

        Parameters
        ----------
        name : str or object
            The spectrum, by name or as the object itself.

        Returns
        -------
        Averaging or None
        """
        from .core.averaging import Averaging

        record = self.provenance.get(self.name_of(name), {})
        state = record.get('state')
        if (record.get('verb') not in ('compute_psds', 'compute_cpsds',
                                       'compute_spectra', 'compute_frfs',
                                       'compute_multiple_coherence')
                or not state or state[0] not in ('averaging',
                                                 'averaging+roles')):
            return None
        settings = state[1]
        if state[0] == 'averaging+roles' and settings is not None:
            settings = settings.get('averaging')
        if not settings:
            # no averaging set: each record was its own frame, whole and
            # unwindowed (`TimeHistory._spectral_frame`)
            source = self.get(record.get('source'))
            spectrum = self.get(self.name_of(name))
            samples = (len(source.abscissa) if source is not None
                       else 2 * (len(spectrum.abscissa) - 1))
            return Averaging.for_records(samples)
        known = {key: settings[key] for key in (
            'frame_length', 'overlap', 'window', 'frames', 'start',
            'detrend', 'window_parameter') if key in settings}
        try:
            return Averaging(**known)
        except (TypeError, ValueError):
            return None

    def rms_source(self, name: Any) -> str | None:
        """The name of the time history a computed autospectrum was made
        from, from its provenance, or None when it records none (an
        imported spectrum) or that record is no longer in the project.

        Parameters
        ----------
        name : str or object
            The spectrum, by name or as the object itself.

        Returns
        -------
        str or None
        """
        from .core.data import TimeHistory

        record = self.provenance.get(self.name_of(name), {})
        source = record.get('source')
        if (record.get('verb') != 'compute_psds' or source not in self
                or not isinstance(self[source], TimeHistory)):
            return None
        return source

    def rms_check(self, spectrum: Any, record: int = 0, *,
                  history: Any = None,
                  threshold_db: float | None = None) -> Any:
        """Check one record of an autospectrum against the time history
        it came from (the rms check): the rms from the area under the
        spectrum beside the record's own rms, the gap in dB, and the
        verdict.

        A computed spectrum is checked against the record its
        provenance names, over the frames it was averaged with
        (`averaging_of`), so a record whose averaging has moved since
        is still read as the spectrum saw it. An imported spectrum, or
        one checked against another record, takes `history` and that
        record's own averaging. `Psd.rms_check` does the reading.

        Parameters
        ----------
        spectrum : str or object
            The autospectrum, by name or as the object itself.
        record : int, default 0
            Which record of the spectrum.
        history : str or object, optional
            The time history to check against; the one the spectrum
            was computed from when omitted.
        threshold_db : float, optional
            The gap, in dB either way, beyond which the two disagree;
            `core.rms_check.GAP_DB` when omitted.

        Returns
        -------
        RmsCheck

        Raises
        ------
        ValueError
            When no time history is given and the spectrum records
            none, or when `Psd.rms_check` refuses.
        """
        name = self.name_of(spectrum)
        source = self.rms_source(name)
        if history is None:
            if source is None:
                raise ValueError(
                    f'{name} records no time history it was computed '
                    'from; name the one to check it against')
            history = source
        history_name = self.name_of(history)
        averaging = (self.averaging_of(name) if history_name == source
                     else None)
        return self[name].rms_check(self[history_name], record,
                                    averaging=averaging,
                                    threshold_db=threshold_db)

    def caption_of(self, name: Any, records: int | Sequence[int] = 0, *,
                   specification: Any = None, per_octave: int | None = None,
                   unit_system: Any = None, tone: str | None = None) -> Any:
        """The caption a report plot of a measured record carries, read
        off the record and the project around it (`core.report_caption`):
        a spectrum's, a shock response spectrum's, a sine level's or a
        time history's (`report_caption.caption_kind`).

            print(project.caption_of('Time History PSDs', 2).text())

        A shock response spectrum's run is the time history it was
        computed from, with that run's sample rate and the filters
        applied in the project on the way to it; against a requirement
        each record is judged by the points rule in the polarity drawn
        and in each other polarity the project holds of the same events
        at the same Q and spacing. A sine level's run is the recording
        it was extracted from, its sweep is the tone as the requirement
        specifies it, and its distortion is the check the report's check
        blocks have already made of the tone (`check_blocks.kept`); the
        check rebuilds every tone and is not made for a caption. A time
        history is its own run.

        Every row comes from the project, none is typed. The run is the
        time history the spectrum was computed from, and the stretch of
        it averaged; the window, overlap and frames are the averaging
        the spectrum was made with (`averaging_of`). A banded spectrum
        reads these, and its line spacing and degrees of freedom, from
        the narrowband spectrum it was banded from. The channel number
        and comment come from a channel table that names the channel,
        the one linked with the spectrum or its run first, and the
        global axis from the geometry linked with them, or the Basis
        geometry. An imported spectrum has no run and no averaging the
        project knows of, so its run, window and frames read "not
        recorded" (a block label, where its records carry one, is the
        run): the name an import gave the object is not a run.

        `per_octave` is for a narrowband spectrum drawn on bands without
        being stored banded, as a report block that bands its data
        draws it: the caption is then of the spectrum on those bands,
        compared with the specification on the same bands, and the
        comparison's scale is resolved on the narrowband spectrum, as
        the report resolves it.

        Parameters
        ----------
        name : str or object
            The spectrum, by name or as the object itself.
        records : int or sequence of int, default 0
            The records the plot draws.
        specification : str or object, optional
            The requirement drawn with the spectrum, by name or as the
            object itself.
        per_octave : int, optional
            Bands per octave the spectrum is drawn on, for one banded
            only for the plot.
        unit_system : UnitSystem, optional
            The display units the rms values and ranges are written in.
        tone : str, optional
            For a sine level set, the tone drawn; its first when omitted.

        Returns
        -------
        RecordCaption, SrsCaption, SineCaption or TimeCaption

        Raises
        ------
        TypeError
            If the object is not a measured record a caption reads.
        """
        from .core.report_caption import caption_kind, record_caption

        name = self.name_of(name)
        spectrum = self[name]
        kind = caption_kind(spectrum)
        if kind is None:
            raise TypeError(f'{name!r} is not a measured record a caption '
                            'reads (a spectrum, a shock response spectrum, '
                            'a sine level or a time history)')
        if kind == 'srs':
            return self._srs_caption_of(name, records, specification)
        if kind == 'sine':
            return self._sine_caption_of(name, records, specification, tone)
        if kind == 'time':
            return self._time_caption_of(name, records, unit_system)
        # a banded spectrum is read through the narrowband one it was
        # banded from: the averaging, the run, the lines and the DOF
        # are all that spectrum's
        narrow_name, narrowband, stored_bands = name, None, None
        made = self.provenance.get(name, {})
        if made.get('verb') == 'compute_octave' \
                and isinstance(self.get(made.get('source')), Psd):
            narrow_name = made['source']
            narrowband = self[narrow_name]
            stored_bands = (made.get('params') or {}).get('per_octave')
        averaging = self.averaging_of(narrow_name)
        computed = self.provenance.get(narrow_name, {})
        history_name = (computed.get('source')
                        if computed.get('verb') in ('compute_psds',
                                                    'compute_cpsds')
                        else None)
        history = self.get(history_name) if history_name else None
        if not isinstance(history, TimeHistory):
            history_name, history = None, None
        # an imported spectrum names no run: the object's name is what
        # the import or a person called it, which may be the file's,
        # the channel's or anything else. A block label, where the
        # records carry one, is read by `record_caption`
        run = history_name
        sample_rate, origin = None, 0.0
        if history is not None and len(history.abscissa) > 1:
            sample_rate = history.sample_rate
            origin = float(history.abscissa[0])
        table, geometry = self._caption_surroundings(
            spectrum, records, (name, narrow_name, history_name))
        bound, requirement = self._caption_requirement(specification)
        scale_db = None
        drawn = spectrum
        if per_octave and getattr(spectrum, 'bandwidth', None) is None:
            from .core.compliance import comparison_scale_db

            # the report's rule (`report._banded_pair`): bands compare
            # only with the same bands, so the specification is banded
            # with the data unless it is on bands already; and the scale
            # is the narrowband one, since detecting it again on the
            # banded copy can round to a different decibel
            if bound is not None:
                scale_db = comparison_scale_db(bound, spectrum)
                if getattr(bound, 'bandwidth', None) is None:
                    bound = bound.to_octave(int(per_octave))
            drawn, narrowband = spectrum.to_octave(int(per_octave)), spectrum
            per_octave = int(per_octave)
        else:
            # stored banded, or narrowband and drawn so
            per_octave = stored_bands
        return record_caption(
            drawn, records, averaging=averaging, run=run,
            sample_rate=sample_rate, origin=origin, channel_table=table,
            geometry=geometry, narrowband=narrowband, per_octave=per_octave,
            specification=bound, requirement=requirement, scale_db=scale_db,
            unit_system=unit_system)

    def _field_members(self, sources: Any,
                       records: Sequence[int] | None) -> tuple[Any, list[str]]:
        """What `core.field_limit.gather` takes for `sources`, and the
        names of the objects it reads: one name or object (with
        `records`), a list of them (every record of each), or a list of
        (name, record) pairs."""
        if isinstance(sources, (list, tuple)):
            members, names = [], []
            for item in sources:
                if isinstance(item, (list, tuple)):
                    name = self.name_of(item[0])
                    members.append((self[name], int(item[1])))
                else:
                    name = self.name_of(item)
                    members.append(self[name])
                if name not in names:
                    names.append(name)
            return members, names
        name = self.name_of(sources)
        return self[name], [name]

    def field_limit(self, sources: Any, records: Sequence[int] | None = None,
                    *, statistic: str = 'tolerance',
                    coverage: float = _FIELD_COVERAGE,
                    confidence: float = _FIELD_CONFIDENCE,
                    k: float | None = None, sigma_db: float | None = None,
                    band: tuple[float, float] | None = None,
                    max_overshoot_db: float = _FIELD_OVERSHOOT,
                    rms: str = 'exceed',
                    max_breakpoints: int | None = _FIELD_BREAKPOINTS,
                    progress: Any = None) -> Any:
        """The statistical limit over a set of measured PSDs or SRSs and
        the straight-line envelope over it (`core.field_limit.derive`),
        without adding anything to the project.

        Parameters
        ----------
        sources : str, object, or list
            One PSD or SRS by name or as the object (its records are the
            set), a list of them (every record of each), or a list of
            (name, record) pairs.
        records : sequence of int, optional
            With one source, the records that make the set.
        statistic, coverage, confidence, k, sigma_db, band
            As `core.field_limit.derive` takes them, with its defaults.
        max_overshoot_db, rms, max_breakpoints
            As `core.field_limit.fit_envelope` takes them, with its
            defaults; `max_breakpoints` None for no limit.
        progress : callable, optional
            Called with ``(done, total)``.

        Returns
        -------
        EnvelopeFit
            The envelope, its `limit` the limit and `limit.fieldset` the
            set.
        """
        from .core import field_limit as fl

        members, _names = self._field_members(sources, records)
        return fl.derive(
            members, records, statistic=statistic, coverage=coverage,
            confidence=confidence, k=k, sigma_db=sigma_db,
            band=None if band is None else tuple(band),
            max_overshoot_db=max_overshoot_db, rms=rms,
            max_breakpoints=max_breakpoints, progress=progress)

    def derive_specification(self, sources: Any,
                             records: Sequence[int] | None = None, *,
                             statistic: str = 'tolerance',
                             coverage: float = _FIELD_COVERAGE,
                             confidence: float = _FIELD_CONFIDENCE,
                             k: float | None = None,
                             sigma_db: float | None = None,
                             band: tuple[float, float] | None = None,
                             max_overshoot_db: float = _FIELD_OVERSHOOT,
                             rms: str = 'exceed',
                             max_breakpoints: int | None = _FIELD_BREAKPOINTS,
                             margin_db: float = 0.0,
                             dof: str | None = None,
                             name: str | None = None,
                             progress: Any = None) -> str:
        """Write the specification a set of measured PSDs or SRSs gives
        (Derive Specification): the limit over the set, the
        straight-line envelope over the limit, and `margin_db` added,
        as a `Specification` at the breakpoints for PSDs or a
        `ShockSpecification` for SRSs (`EnvelopeFit.specification`).

        Parameters
        ----------
        sources, records, statistic, coverage, confidence, k, sigma_db,
        band, max_overshoot_db, rms, max_breakpoints
            As `field_limit` takes them.
        margin_db : float, default 0
            Decibels added to every breakpoint, on the set's convention
            (10 log10 for a PSD, 20 log10 for an SRS).
        dof : str, optional
            The DOF the specification is written at; the set's own when
            it has one.
        name : str, optional
            What to call it; the first source's name followed by
            'Derived Specification' when omitted.
        progress : callable, optional
            Called with ``(done, total)``.

        Returns
        -------
        str
            The name it was added under, unique within the project (a
            name already taken gets a numbered suffix).
        """
        settings = {'statistic': statistic, 'coverage': coverage,
                    'confidence': confidence, 'k': k, 'sigma_db': sigma_db,
                    'band': None if band is None else list(band),
                    'max_overshoot_db': max_overshoot_db, 'rms': rms,
                    'max_breakpoints': max_breakpoints}
        fit = self.field_limit(sources, records, progress=progress,
                               **settings)
        spec = fit.specification(margin_db=margin_db, dof=dof)
        _members, names = self._field_members(sources, records)
        listed = ([list(item) if isinstance(item, (list, tuple))
                   else self.name_of(item) for item in sources]
                  if isinstance(sources, (list, tuple)) else names[0])
        params = {'sources': listed,
                  'records': None if records is None else
                  [int(r) for r in records],
                  **settings, 'margin_db': float(margin_db), 'dof': dof}
        return self._derive(names[0], spec,
                            name or f'{names[0]} Derived Specification',
                            recipe=('derive_specification', params))

    def _caption_surroundings(self, data: Any, records: Any,
                              members: Sequence[str | None]) -> tuple[Any, Any]:
        """(channel table, geometry) for a caption of `records` of
        `data`. The table and the geometry linked with any of `members`
        (the record, what it was made from, its run) come first, since a
        project can hold two tests' tables; then any table that names a
        channel drawn, and the Basis geometry."""
        from .core.report import resolve_binding

        linked: list[str] = []
        for member in members:
            if member:
                linked.extend(self.object_group_of(member) or [])
        candidates = linked + [other for other in self.names
                               if other not in linked]
        wanted = ([int(records)] if np.ndim(records) == 0
                  else [int(record) for record in records])
        dofs = {str(data.response_dof[record]) for record in wanted
                if 0 <= record < data.num_records}
        table = next((self[other] for other in candidates
                      if isinstance(self[other], ChannelTable)
                      and dofs & set(self[other].dof_strings())), None)
        geometry = next((self[other] for other in linked
                         if isinstance(self[other], Geometry)), None)
        if geometry is None:
            basis = resolve_binding('@basis:Geometry', self, self.object_groups)
            geometry = self.get(basis) if basis else None
        return table, geometry

    def _caption_requirement(self, specification: Any) -> tuple[Any, Any]:
        """(requirement, its name in the project) for a caption, given
        by name or as the object; (None, None) for none."""
        if specification is None:
            return None, None
        bound = (self[specification] if isinstance(specification, str)
                 else specification)
        return bound, next((other for other, obj in self.items()
                            if obj is bound), None)

    def _made_from(self, name: str, verb: str) -> str | None:
        """The time history `name` was made from by `verb`, from its
        provenance, or None (an import, or a source since removed)."""
        made = self.provenance.get(name, {})
        source = made.get('source')
        if made.get('verb') == verb and isinstance(self.get(source),
                                                   TimeHistory):
            return source
        return None

    def _filters_on_the_way(self, run: str) -> tuple[str, ...]:
        """Each filter applied in the project on the way to `run`, the
        first applied first, in words: the setting recorded when the
        filtered record was made, its order and its zero-phase pass
        (`filter_data`)."""
        from .core.filters import Filtering

        out: list[str] = []
        seen: set[str] = set()
        current = run
        while current and current not in seen:
            seen.add(current)
            made = self.provenance.get(current) or {}
            if made.get('verb') == 'filter_data':
                state = made.get('state') or (None, None)
                try:
                    setting = Filtering(**state[1])
                except (TypeError, ValueError):
                    out.append('a filter whose setting is not recorded')
                else:
                    out.append(f'{setting.describe()}, order '
                               f'{setting.order}, zero phase')
            current = made.get('source')
        return tuple(reversed(out))

    def _srs_caption_of(self, name: str, records: Any,
                        specification: Any) -> Any:
        """`caption_of` for a shock response spectrum."""
        from .core.report_caption import caption_kind, srs_caption

        srs = self[name]
        run = self._made_from(name, 'compute_srs')
        history = self.get(run) if run else None
        sample_rate = None
        if history is not None and len(history.abscissa) > 1:
            try:
                sample_rate = history.sample_rate
            except ValueError:
                sample_rate = None      # not evenly sampled: no one rate
        made = self.provenance.get(name, {})
        # the other polarities of the same events: computed from the
        # same run over the same windows, at the same Q and spacing
        polarities: list[Any] = []
        if run is not None:
            for other, obj in self.items():
                theirs = self.provenance.get(other, {})
                if (other != name and caption_kind(obj) == 'srs'
                        and theirs.get('verb') == 'compute_srs'
                        and theirs.get('source') == run
                        and theirs.get('state') == made.get('state')
                        and obj.q == srs.q and obj.kind != srs.kind
                        and obj.num_records == srs.num_records
                        and np.array_equal(obj.abscissa, srs.abscissa)
                        and all(obj.kind != held.kind
                                for held in polarities)):
                    polarities.append(obj)
        order = ('maximax', 'positive', 'negative')
        polarities.sort(key=lambda obj: (order.index(obj.kind)
                                         if obj.kind in order else len(order)))
        table, geometry = self._caption_surroundings(srs, records,
                                                     (name, run))
        bound, requirement = self._caption_requirement(specification)
        return srs_caption(
            srs, records, run=run, sample_rate=sample_rate,
            filters=None if run is None else self._filters_on_the_way(run),
            channel_table=table, geometry=geometry, specification=bound,
            requirement=requirement, polarities=polarities)

    def _sine_caption_of(self, name: str, records: Any, specification: Any,
                         tone: str | None) -> Any:
        """`caption_of` for a sine level or a set of them."""
        from .core.report_caption import sine_caption
        from .core.sine import SineLevelSet
        from .report.check_blocks import Refusal, kept

        levels = self[name]
        run = self._made_from(name, 'extract_sine')
        bound, _requirement = self._caption_requirement(specification)
        if bound is None and run is not None:
            # the requirement the levels were extracted against
            spec_name = (self.provenance[name].get('params') or {}).get(
                'specification')
            bound = self.get(spec_name) if spec_name else None
        level = levels
        if isinstance(levels, SineLevelSet) and levels.levels:
            level = levels.tone(tone) if tone else levels.levels[0]
        table, geometry = (self._caption_surroundings(level, records,
                                                      (name, run))
                           if not isinstance(level, SineLevelSet)
                           else (None, None))
        # the distortion check already made of this tone over the run
        # the levels came from; a check of another recording is not
        # this level's
        distortion = None
        if (run is not None and bound is not None
                and not isinstance(level, SineLevelSet)):
            reading = kept('distortion',
                           {'source': self[run], 'specification': bound,
                            'measured': levels},
                           {'tone': level.tone})
            if reading is not None:
                distortion = (reading.reason if isinstance(reading, Refusal)
                              else reading)
        return sine_caption(levels, records, tone=tone, run=run,
                            channel_table=table, geometry=geometry,
                            specification=bound, distortion=distortion)

    def _time_caption_of(self, name: str, records: Any,
                         unit_system: Any) -> Any:
        """`caption_of` for a time history, which is its own run."""
        from .core.report_caption import time_caption

        history = self[name]
        table, geometry = self._caption_surroundings(history, records,
                                                     (name,))
        return time_caption(history, records, run=name, channel_table=table,
                            geometry=geometry, unit_system=unit_system)

    def compute_octave(self, source: Any, per_octave: int | None = None
                       ) -> str:
        """Integrate a spectrum onto proportional octave bands (Compute
        Octave Bands). A specification's warning and abort limits are
        banded the same way.

        The total power is unchanged; it is only arranged in bands, the
        way such a spectrum is usually read.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself.
        per_octave : int, optional
            Bands per octave. Defaults to `core.octave.PER_OCTAVE`, 6.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
        """
        from .core.octave import PER_OCTAVE

        per_octave = PER_OCTAVE if per_octave is None else int(per_octave)
        source = self.name_of(source)
        return self._derive(
            source, self[source].to_octave(per_octave),
            f'{source} 1/{per_octave} Octave',
            recipe=('compute_octave', {'per_octave': per_octave}))

    def make_realizable(self, source: Any, *, name: str | None = None,
                        tolerance: float | None = None,
                        keep_diagonal: bool = True,
                        progress: Any = None) -> str:
        """Repair a cross-spectral matrix to the nearest realizable one
        (Make Realizable in the Realizability reading), and add the
        result beside it.

        At every line where the matrix is not Hermitian and positive
        semi-definite, so that no set of signals has it, the nearest
        matrix that is replaces it (`Psd.made_realizable`); the lines
        that pass are unchanged. Holding the diagonal, the default, the
        autospectra stay exactly as written and only the cross terms
        move. `Psd.realizability` on the source says what moved and by
        how much.

        Parameters
        ----------
        source : str or object
            The CSD or specification, by name or as the object itself.
        name : str, optional
            What to call the result. Defaults to the source's name
            followed by 'Realizable'.
        tolerance : float, optional
            How far below zero the smallest eigenvalue of the coherence
            matrix may lie, as a fraction of the largest;
            `core.matrix_checks.REALIZABLE_TOLERANCE` when omitted.
        keep_diagonal : bool, default True
            Hold the autospectra where they are.
        progress : callable, optional
            Called with ``(done, total)`` as the repair goes on. The
            app's progress bar uses it; a script can print it.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.

        Raises
        ------
        ValueError
            If the object has no cross terms or its matrix is
            incomplete, if no line could be judged, or if it is
            realizable already, in which case there is nothing to
            repair.
        """
        from .core.matrix_checks import REALIZABLE_TOLERANCE, made_realizable

        tolerance = (REALIZABLE_TOLERANCE if tolerance is None
                     else float(tolerance))
        source = self.name_of(source)
        repaired, reading = made_realizable(
            self[source], tolerance=tolerance, keep_diagonal=keep_diagonal,
            progress=progress)
        if not reading.judged.any():
            raise ValueError(f'{source!r}: no line could be judged, so there '
                             'is nothing to repair; '
                             + '; '.join(reading.notes))
        if reading.realizable:
            raise ValueError(f'{source!r} is realizable at every line; '
                             'there is nothing to repair')
        return self._derive(
            source, repaired, name or f'{source} Realizable',
            recipe=('make_realizable', {'tolerance': tolerance,
                                        'keep_diagonal': bool(keep_diagonal)}))

    def compute_frfs(self, source: Any, method: str = 'Hv') -> str:
        """Compute frequency response functions from a time history
        (Compute FRFs), one for each response and drive, over the
        frames a PSD uses.

        The three estimators differ only in where the noise is assumed
        to be. The method is put in the result's name, since two FRF
        sets from one history differ in nothing else a reader can see.

        If the history has no averaging set, the frames are detected the
        same way the coherence detects them, so the FRFs and the
        coherence come from the same frames. The detected averaging is
        stored on the history.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself.
        method : str, default 'Hv'
            Which estimator: 'Hv', 'H1' or 'H2'.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
        """
        source = self.name_of(source)
        history = self[source]
        if history.averaging is None:
            history.averaging = history.suggest_averaging()
        return self._derive(source, history.compute_frfs(method=method),
                            f'{source} {method} FRFs',
                            recipe=('compute_frfs', {'method': method}))

    def read_transmissibility(self, name: Any, record: int = 0) -> Any:
        """The ASD-ratio transmissibility behind one record of an FRF,
        beside its H1, with H1's coherence and random error.

        The FRF must have been computed in this project from a time
        history (Compute FRFs), against one reference. The reading is
        made from that history, for the record's response and reference,
        with the averaging the FRF was computed with (`averaging_of`),
        so it is of the frames the FRF was. Its H1 is the H1 of that
        pair whichever estimator the FRF used; the ratio, the coherence
        and the error are what `TimeHistory.read_transmissibility` says.

        Parameters
        ----------
        name : str or object
            The FRF, by name or as the object itself.
        record : int, default 0
            Which record of the FRF.

        Returns
        -------
        TransmissibilityReading
            H1, the ASD ratio and the coherence at every line, in SI,
            with the frames and the averages.

        Raises
        ------
        ValueError
            If the object is not an FRF computed from a time history in
            this project, or it was computed against more than one
            reference, where a ratio of two autospectra would credit one
            input with what several caused.
        """
        from .core.data import channel_quantities

        name = self.name_of(name)
        frf = self[name]
        provenance = self.provenance.get(name, {})
        source = self.get(provenance.get('source'))
        if (not isinstance(frf, Frf)
                or provenance.get('verb') != 'compute_frfs'
                or not isinstance(source, TimeHistory)):
            raise ValueError(
                f'{name!r} is not an FRF computed from a time history in '
                'this project, so there are no autospectra behind it to '
                'divide')
        references = sorted({str(dof) for dof in frf.reference_dof})
        if len(references) != 1:
            raise ValueError(
                f'{name!r} was computed against {len(references)} '
                'references; the ASD ratio is one response over one '
                'input, and with several it would credit one with what '
                'all of them caused')
        record = int(record)
        response = str(frf.response_dof[record])
        reference = str(frf.reference_dof[record])
        dimension = str(frf.ordinate_dim[record])
        if '/' in dimension:
            # (DOF, quantity) pairs, so a drive point's accelerometer and
            # its force gauge are not taken one for the other
            out, into = channel_quantities(dimension)
            response_channel: Any = (response, out)
            reference_channel: Any = (reference, into)
        else:
            response_channel, reference_channel = response, reference
        return source.read_transmissibility(
            response_channel, reference_channel,
            averaging=self.averaging_of(name))

    def read_partial_coherence(self, name: Any, *,
                               progress: Any = None) -> Any:
        """Which drive explains each response, and whether the drives
        were independent: the partial, reference and virtual coherence
        (`core.partial_coherence`).

        `name` is a time history, read against its reference channels
        with its own averaging, or a multiple coherence computed in this
        project (Compute Multiple Coherence), read from its history with
        the averaging it was computed with (`averaging_of`), so the
        reading's multiple coherence is that object's.

        Parameters
        ----------
        name : str or object
            The time history or the multiple coherence, by name or as
            the object itself.
        progress : callable, optional
            Called with ``(0, core.progress.BUSY)`` as the work starts,
            which cannot count its own stages; the app's progress bar
            uses it.

        Returns
        -------
        PartialCoherence
            Every response against every reference, at every line.

        Raises
        ------
        ValueError
            If the object is neither, or the history has no reference or
            no response channels, or no more averages than references.
        """
        from .core.data import MultipleCoherence
        from .core.progress import Ticker

        Ticker(progress).busy()
        name = self.name_of(name)
        data = self[name]
        if isinstance(data, TimeHistory):
            return data.read_partial_coherence()
        provenance = self.provenance.get(name, {})
        source = self.get(provenance.get('source'))
        if (not isinstance(data, MultipleCoherence)
                or provenance.get('verb') != 'compute_multiple_coherence'
                or not isinstance(source, TimeHistory)):
            raise ValueError(
                f'{name!r} is neither a time history nor a multiple '
                'coherence computed from one in this project, so there are '
                'no spectra behind it to condition')
        return source.read_partial_coherence(
            averaging=self.averaging_of(name))

    def compute_multiple_coherence(self, source: Any) -> str:
        """Compute the multiple coherence from a time history (Compute
        Multiple Coherence): how much of each response the drives
        account for.

        It is averaged over frames. If the history has no averaging set,
        the frames are detected and stored on the history. Computed from
        the whole record as one frame, the drives would fit every
        response exactly and the coherence would be 1.0 at every line,
        which says nothing about the measurement.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
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
        """Compute shock response spectra from a time history's shocks
        (Compute SRS), one curve per channel per event.

        The events are the ones marked on the history. They are detected,
        and marked on the history, only when it has no shocks marked and
        no averaging set and is not a transient specification. A record
        read as frames already says where its events are (in a transient
        run each playing is one frame), and a specification is one
        playing by definition.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself.
        per_octave : int, optional
            Natural-frequency lines per octave. Defaults to 12.
        q : float, optional
            The oscillator amplification, Q = 1/(2ζ). Defaults to 10
            (5% damping), the usual value for shock tests.
        kind : str, default 'maximax'
            Which peak each oscillator reports: 'maximax' (largest
            magnitude of either sign), 'positive' or 'negative'.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
        """
        # Detection is the last resort. Set loose on a transient run, whose
        # playings are its averaging, the detector found thirty-one
        # events where there were six, and a target (one playing by
        # definition) was being cut into three.
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
        """Find the events in a time history and mark them on it (Detect
        in the shock view). Returns how many were found.

        Any events already marked are replaced. Mark the events on the
        recording before filtering it: every object derived from the
        recording then carries them, because `core.filters` copies the
        marks onto what it makes. `compute_srs` detects events only when
        a record has none, so if the filtered record were the first to
        be detected, the recording itself would be left unmarked.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself.

        Returns
        -------
        int
            How many events were found and marked on the record.
        """
        # The verb the API was missing (Brandon, 2026-08-25): the shock
        # workflow filters first, and detection inside compute_srs then
        # happened on the filtered record and left the recording unmarked.
        source = self.name_of(source)
        from .core.shocks import find

        history = self[source]
        history.shocks = find(history)
        return len(history.shocks or ())

    def filter_data(self, source: Any) -> str:
        """Filter a time history with its own filter settings (Apply
        Filter in the filter view), every channel at zero phase so the
        peaks stay where they are.

        The settings are the history's `filtering`, set in the filter
        view as a low-, high- or band-pass. When none is set, the
        suggested low-pass (a tenth of the sample rate, order 4) is
        stored on the history and used, the way `compute_frfs` uses a
        suggested averaging, so this works before the filter view has
        been opened.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
        """
        source = self.name_of(source)
        history = self[source]
        if history.filtering is None:
            history.filtering = history.suggest_filtering()
        return self._derive(source, history.filter(),
                            f'{source} Filtered',
                            recipe=('filter_data', {}))

    def resample(self, source: Any, rate: float, *, anti_alias: bool = True,
                 order: int | None = None) -> str:
        """Take a time history to a lower sample rate (Apply Resample in
        the resample view) and add the result beside it.

        Every channel goes through the zero-phase anti-alias low-pass
        and then every q-th sample is kept (`TimeHistory.resample`,
        which says what the filter is). The resample view shows what
        the resampling does to a channel's spectrum before it is made.

        Parameters
        ----------
        source : str or object
            The time history, by name or as the object itself.
        rate : float
            The new sample rate, in Hz, a whole division of the
            record's.
        anti_alias : bool, default True
            Filter before decimating.
        order : int, optional
            The anti-alias filter's design order,
            `core.resample.ORDER` when omitted.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
        """
        # The reading had no apply and no verb (2026-10-28): a script
        # could make the record but the app could not, and a record a
        # script made had no provenance. Processing is select, see, set,
        # apply (principle 13).
        source = self.name_of(source)
        history = self[source]
        made = history.resample(float(rate), anti_alias=anti_alias,
                                order=order)
        return self._derive(source, made, f'{source} at {rate:g} Hz',
                            recipe=('resample', {
                                'rate': float(rate),
                                'anti_alias': bool(anti_alias),
                                'order': order}))

    def truncate_data(self, source: Any) -> str:
        """Cut a time history to its truncation span (Apply Truncation in
        the truncate view): every channel between start and stop, on the
        record's original clock.

        The span is the history's `truncation`, set in the truncate
        view. Unlike `filter_data`, there is no suggested value to fall
        back on, because the only neutral span is the whole record and
        keeping all of it changes nothing. With no span set this raises
        ValueError.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
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
        """Integrate a time history once (Integrate): acceleration
        channels become velocity, and velocity channels become
        displacement.

        The whole record is integrated, never the shock windows alone;
        `core.filters` gives the reasons. The drift corner is stored in
        the result's provenance, so a refresh uses the same one.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself.
        drift_corner : float or None, optional
            High-pass corner in Hz applied after integration, so that a
            sensor bias does not become a ramp. ``...``, the default,
            uses `core.filters.DRIFT_CORNER` (2 Hz). None applies no
            high-pass, so any drift is kept.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
        """
        from .core.filters import DRIFT_CORNER

        corner = DRIFT_CORNER if drift_corner is ... else drift_corner
        source = self.name_of(source)
        result = self[source].integrate(corner)
        return self._derive(source, result,
                            _motion_name(source, result, 'Integrated'),
                            recipe=('integrate', {'drift_corner': corner}))

    def differentiate(self, source: Any) -> str:
        """Differentiate a time history once (Differentiate):
        displacement channels become velocity, and velocity channels
        become acceleration.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
        """
        source = self.name_of(source)
        result = self[source].differentiate()
        return self._derive(source, result,
                            _motion_name(source, result, 'Differentiated'),
                            recipe=('differentiate', {}))

    def compute_cpsds(self, source: Any) -> str:
        """Compute the full cross-spectral matrix from a time history's
        averages (Compute CPSDs), every channel against every channel.

        Parameters
        ----------
        source : str or object
            The object to read, by name or as the object itself.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
        """
        source = self.name_of(source)
        return self._derive(source, self[source].compute_cpsds(),
                            f'{source} CPSDs', recipe=('compute_cpsds', {}))

    def transform(self, source: Any, shapes: Any, *,
                  records: Sequence[int] | None = None,
                  name: str | None = None) -> str:
        """Transform a record's channels through a shape set to modal
        coordinates (Transform to Modal Coordinates), one record per mode
        and quantity at `M1` … `Mn`.

        The motions are transformed as `q = Φ⁺u` and the forces as
        `Φᵀf`. It is a change of coordinates only: the set's
        frequencies, damping and masses are not used (for the response
        a mode makes to forces, see `compute_modal_responses`). Any shape
        set can be used; with the six rigid-body shapes of a
        geometry this is the virtual point transformation. The result is
        linked into the record's group when a shape set there has every
        mode it names, and is otherwise in no group. Its provenance
        names both the record and the shape set, and its
        `transform_report` says what was shared, what was dropped and
        what was left unexplained. A source that is not a data object,
        or `shapes` that is not a shape set, raises TypeError.

        Parameters
        ----------
        source : str or object
            The time history, by name or as the object itself.
        shapes : str or object
            The shape set to transform through, by name or as the object.
        records : sequence of int, optional
            Which of the record's channels to use, the ones picked in
            the tree. All of them when omitted.
        name : str, optional
            What to call the result. Defaults to the record's name
            followed by 'Modal Coordinates'.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
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
                            name or f'{source} Modal Coordinates',
                            recipe=('transform', params))

    def expand(self, source: Any, shapes: Any, *,
               records: Sequence[int] | None = None,
               name: str | None = None) -> str:
        """Expand modal responses through a shape set back to physical
        responses (Expand to Physical Responses): `u = Φq` at every DOF
        the shape set covers.

        The result is linked into the shape set's group, so it animates
        on that group's geometry. Expanding some of the modes gives
        their contribution alone, and the result's name lists them.

        Parameters
        ----------
        source : str or object
            The modal time history, by name or as the object itself.
        shapes : str or object
            The shape set it was transformed through, by name or as the
            object.
        records : sequence of int, optional
            Which modal records to expand, the modes picked in the
            tree. All of them when omitted.
        name : str, optional
            What to call the result. Defaults to the record's name with
            'Modal Coordinates' replaced by 'Physical Responses', and the
            modes listed in parentheses when not all of them are used.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
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
        from .core.sine import SineLevelSet
        from .core.transform import _cannot

        data, shape_set = self[source], self[shapes]
        if isinstance(data, SineLevelSet):
            raise TypeError(_cannot(data))
        if not isinstance(data, DataArray):
            raise TypeError(f'{source!r} is not a data object')
        if not isinstance(shape_set, ShapeSet):
            raise TypeError(f'{shapes!r} is not a shape set')
        return data, shape_set

    def author_specification(self, source: Any, draft: Any, *,
                             name: str | None = None,
                             replace: bool = False) -> str:
        """Write a specification from a sheet (Make Specification in the
        Specification reading) and add it beside the object the sheet
        was opened on.

        The sheet gives autospectra at breakpoints, each cross term as a
        coherence and phase, and the limit bands in decibels. It can be
        opened on a shape set (the specification is then at the set's
        modal coordinates, ready to expand through it), on a channel
        table (at its control channels) or on a specification. A pair
        the sheet does not state is left out of the result; nothing is
        assumed for it.

        With `replace`, the sheet is written into the specification it
        was opened from, under that specification's name, so its links
        and report slots still hold. A sheet that spans several
        specifications is written into each of them. A sheet holding
        every channel of a specification replaces it in the sheet's
        form, with the sheet's breakpoints or lines. A sheet holding some
        of the channels is merged back at the specification's own lines,
        and the other channels are unchanged. Replacing an object that
        is not a specification raises TypeError.

        Parameters
        ----------
        source : str, object, or list of str
            The object the sheet was opened on, by name or as the
            object, or the specifications' names when the sheet spans
            several.
        draft : SpecificationDraft or dict
            What the author stated (`core.author`), or its dict form.
        name : str, optional
            What to call the result. Defaults to the source's name
            followed by 'Specification'.
        replace : bool, default False
            Write the sheet into `source` (a specification, or several)
            in place.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project (a name already taken gets a numbered suffix). With
            `replace`, the source's own name, or the first of them when
            there are several.
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
        """Generate the six rigid-body mode shapes of a geometry (Generate
        Rigid Body Mode Shapes): three translations and three rotations
        about its reference point, as a shape set in the geometry's
        group.

        The reference point, and the mass and inertia that
        mass-normalize the shapes, are the geometry's `mass_properties`,
        set in the rigid-body view. When none are set, unit shapes about
        the centroid are used, which is the case the virtual point
        transformation needs most often. Those properties are then
        stored on the geometry, so the result's provenance records what
        was used. An object that is not a geometry raises TypeError.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        name : str, optional
            What to call the result. Defaults to the geometry's name
            followed by 'Rigid Body Modes'.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
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

    def generate_signal(self, source: Any, kind: str = 'random', *,
                        name: str | None = None, progress: Any = None,
                        **settings: Any) -> str:
        """Generate a reference signal from a spectrum, a sine
        specification, a shock specification or a channel table
        (Generate Signal), as a time history in the source's group.

        A spectrum gives random, burst random or periodic random at its
        DOFs, drawn to it; a sine specification gives its tone; a shock
        specification gives a shock synthesized to its spectrum and
        checked against the shaker's shock ratings given
        (`core.shock_synthesis`); a channel table gives the random kinds
        at its reference channels, to a typed flat `band` and `level`,
        or an impulse or a classical pulse (`core.signals`). The
        settings are recorded with the result, and the random seed with
        them (one is drawn when none is given), so Recompute makes the
        same record again.

            project.generate_signal('Requirement', 'srs',
                                    sample_rate=20480.0, stroke=0.0508,
                                    velocity=1.78)

        Parameters
        ----------
        source : str or object
            The spectrum, sine specification, shock specification or
            channel table, by name or as the object itself.
        kind : str, default 'random'
            One of `core.signals.KINDS`: 'random', 'burst', 'periodic',
            'impulse', 'pulse', 'sine' or 'srs'.
        name : str, optional
            What to call the result. Defaults to the source's name and
            the kind.
        progress : callable, optional
            Called with ``(done, total)`` as an SRS synthesis iterates;
            the app's progress bar uses it.
        **settings
            The fields of `core.signals.SignalSettings`: `duration`,
            `sample_rate`, `frame_length`, `seed`, `correlated`, `dofs`,
            `bursts`, `burst`, `taper`, `periods`, `repeats`, `at`,
            `area`, `shape`, `peak`, `pulse_duration`, `edge`, `band`,
            `level`, `tone`, and for an SRS synthesis `half_cycles` and
            the shaker's `stroke`, `velocity`, `force`, `moving_mass` and
            `body_mass` in SI.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.

        Raises
        ------
        TypeError
            If a setting is not one `SignalSettings` has.
        ValueError
            If the kind is not one the source makes, or as the signal's
            generator refuses (a band at or above the Nyquist frequency,
            units not declared, no reference channel, a shock spectrum
            no synthesis meets or no synthesis the shaker can hold).
        """
        from dataclasses import replace

        from .core.signals import KINDS, SignalSettings, generate

        source = self.name_of(source)
        if settings.get('dofs') is not None:
            settings['dofs'] = tuple(settings['dofs'])
        if settings.get('band') is not None:
            settings['band'] = tuple(float(v) for v in settings['band'])
        chosen = SignalSettings(kind=kind, **settings)
        if chosen.seed is None:
            # drawn here and kept, so the record can be made again
            chosen = replace(chosen, seed=int(
                np.random.SeedSequence().generate_state(1)[0]))
        made = generate(self[source], chosen, progress=progress)
        return self._derive(source, made,
                            name or f'{source} {KINDS.get(kind, kind)}',
                            recipe=('generate_signal', chosen.to_dict()))

    def compute_modal_responses(self, source: Any, shapes: Any, *,
                                quantity: str = 'acceleration',
                                modes: Sequence[int] | None = None,
                                records: Sequence[int] | None = None,
                                name: str | None = None) -> str:
        """Compute how each mode of a shape set answers the forces in a
        time history, as a new time history (Compute Modal Responses).

        The history's force channels are made modal through the set
        (pᵣ = Σ φₖᵣ fₖ; channels already holding the set's modal forces
        are used as they are), and each mode's equation is solved
        exactly for a force linear between samples, from rest
        (`core.modal_response`). The result holds one record per mode,
        `M1` … `Mn`, each comment naming the set, on the history's time
        axis with its averaging and shock windows. It is linked into the
        history's group, and the history itself is not changed; `merge`
        puts the forces and the responses back in one object.

        Parameters
        ----------
        source : str or object
            The time history, by name or as the object itself.
        shapes : str or object
            The shape set: real modes with frequency, damping and modal
            mass, by name or as the object itself.
        quantity : str, default 'acceleration'
            'displacement', 'velocity' or 'acceleration' of each modal
            coordinate.
        modes : sequence of int, optional
            Which modes, by index in the set from zero; all of them when
            omitted.
        records : sequence of int, optional
            Which of the history's records are the forces; every force
            channel when omitted.
        name : str, optional
            What to call the result. Defaults to the history's name
            followed by 'Modal Responses'.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.

        Raises
        ------
        TypeError
            If `source` is not a time history or `shapes` not a shape
            set.
        ValueError
            As `core.modal_response.modal_responses`: no force on a DOF
            the set covers, a complex set, a mode not in the set, uneven
            sampling, or a quantity it does not know.
        """
        # The responses were appended to the force history until Brandon
        # asked for an object of their own beside it, the way Transform
        # to Modal Coordinates leaves its result (2026-10-27): the
        # results stay organized, and Merge puts them back together.
        source = self.name_of(source)
        set_name = self.name_of(shapes)
        params: dict[str, Any] = {'shapes': set_name, 'quantity': quantity}
        if modes is not None:
            params['modes'] = sorted({int(k) for k in modes})
        if records is not None:
            params['records'] = [int(i) for i in records]
        return self._derive(source, _answered(self, source, params),
                            name or f'{source} Modal Responses',
                            recipe=('compute_modal_responses', params))

    def merge_coincident_nodes(self, source: Any,
                               tolerance: float | None = None,
                               groups: Sequence[int] | None = None,
                               exclude: Sequence[int] | None = None) -> dict:
        """Merge each set of a geometry's coincident nodes into one node
        (Merge Coincident Nodes). Elements are reconnected to the lowest
        node id at each point, and the other nodes are removed.

        Plates are connected only where they share nodes, so this is
        what ties together planes meshed separately, along the lines
        where they meet.

        With `groups`, only the nodes used by those element groups'
        elements are merged. The app offers the merge only on element
        groups picked in the tree, never on the whole geometry, because
        merging every coincident node in a model ties together parts
        that were meant to stay separate. With `groups` left out, the
        whole geometry is merged.

        While an object linked with the geometry names a node the merge
        would remove, the merge raises ValueError, since that object's
        data would refer to a node that no longer exists. Merge first,
        then measure or solve. An object that is not a geometry raises
        TypeError.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        tolerance : float, optional
            How close two nodes must be to be merged, in meters (the
            geometry holds its coordinates in meters). Defaults to a
            millionth of the diagonal of the geometry's extent.
        groups : sequence of int, optional
            The element groups whose nodes are merged, by id. Every node
            of the geometry when omitted.
        exclude : sequence of int, optional
            Nodes left out of the merge, by id: the places deselected in
            the app's pane before Merge was pressed.

        Returns
        -------
        dict
            'merged', the nodes removed, 'into', the nodes they became,
            'duplicates', the elements removed for filling a cell another
            already filled, and 'links', the line elements removed for
            joining nodes that became one (`Geometry.merge_collapses`).
        """
        # Brandon, 2026-10-11: the app merges only element groups picked
        # in the tree. With `groups` left out the whole geometry is
        # merged, as scripts and journals written before then expect.
        name = self.name_of(source)
        geometry = self[name]
        if not isinstance(geometry, Geometry):
            raise TypeError(f'{name!r} is not a geometry')
        if tolerance is None:
            low, high = geometry.extent
            tolerance = 1e-6 * float(np.linalg.norm(high - low) or 1.0)
        going = {node for spot in geometry.coincident_spots(
            tolerance, groups, exclude) for node in spot[1:]}
        self._refuse_linked(name, going, 'the merge would remove — merge')
        return geometry.merge_coincident_nodes(tolerance, groups, exclude)

    def _refuse_linked(self, name: str, nodes, what: str) -> None:
        """Refuse a change to geometry `name`'s `nodes` while an object
        linked with it names one: its data would point at a node that is
        gone, or somewhere it was not measured or solved. `what` finishes
        the sentence: what the change does to it, and what to do first."""
        for other in self.object_group_of(name) or []:
            if other == name:
                continue
            clash = sorted(_dof_nodes(self.get(other)) & set(nodes))
            if clash:
                raise ValueError(
                    f'{other} names node {clash[0]}, which {what} before '
                    'measuring or solving, or unlink it first')

    def align_meshes(self, source: Any, reach: float | None = None,
                     groups: Sequence[int] | None = None,
                     exclude: Sequence[int] | None = None) -> dict:
        """Move the coarser mesh's nodes onto the finer mesh's nodes where
        parts meshed separately meet or overlap (Align Meshes), so Merge
        Coincident Nodes can join them next.

        Each node of the coarser piece within `reach` of the finer
        piece's surface is moved onto its nearest node there
        (`Geometry.align_moves`). The app offers this on element groups
        picked in the tree, as it does the merge. While an object linked
        with the geometry names a node that would move, this raises
        ValueError, since that object's data was measured or solved
        where the node was.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        reach : float, optional
            How far a node may move, in meters (the geometry holds its
            coordinates in meters). Defaults to half the coarsest piece's
            spacing (`Geometry.align_reach`).
        groups : sequence of int, optional
            The element groups whose pieces are aligned, by id. Every
            element of the geometry when omitted.
        exclude : sequence of int, optional
            Nodes left out, by id: the moves deselected in the app's pane
            before Align was pressed, both nodes of each.

        Returns
        -------
        dict
            'moved', how many nodes moved; 'farthest', the longest move,
            in meters.
        """
        # Align Meshes added by Brandon, 2026-10-11.
        name = self.name_of(source)
        geometry = self[name]
        if not isinstance(geometry, Geometry):
            raise TypeError(f'{name!r} is not a geometry')
        if reach is None:
            reach = geometry.align_reach(groups) or 0.0
        moving = {a for a, _b in geometry.align_moves(reach, groups, exclude)}
        self._refuse_linked(name, moving, 'aligning would move — align')
        return geometry.align_meshes(reach, groups, exclude)

    def new_geometry(self, name: str = 'Geometry', *,
                     unit: str = 'm') -> str:
        """Add an empty geometry to build a model in (the project's +
        button). Its parts are added as solids (`add_solid_plate`,
        `add_solid_block` and the rest) and meshed with `mesh_solids`,
        and links in the app's add mode.

        Parameters
        ----------
        name : str, default 'Geometry'
            What to call it.
        unit : str, default 'm'
            Its length unit. Coordinates are held in SI either way; the
            geometry records this as the unit it was built in.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
        """
        return self.add(name, Geometry(node_id=[], node_xyz=np.empty((0, 3)),
                                       length_unit=unit))

    def new_channel_table(self, name: str = 'Channel Table') -> str:
        """Add an empty channel table to fill in by hand (the project's
        New Channel Table). Channels are added with `add_channels`, and
        their cells are typed in the app or set with `set_cell`.

        Parameters
        ----------
        name : str, default 'Channel Table'
            What to call it.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
        """
        # Brandon, 2026-10-22: as an empty geometry can be made.
        return self.add(name, ChannelTable({column: [] for column
                                            in ChannelTable.CORE}))

    def add_channels(self, source: Any, count: int = 1) -> list[int]:
        """Add blank channels to a channel table, numbered on from its
        highest channel.

        Parameters
        ----------
        source : str or ChannelTable
            The channel table, by name or as the object itself.
        count : int, default 1
            How many channels to add.

        Returns
        -------
        list of int
            The new channels' numbers.

        Raises
        ------
        TypeError
            If the object is not a channel table.
        """
        name = self.name_of(source)
        table = self[name]
        if not isinstance(table, ChannelTable):
            raise TypeError(f'{name!r} is not a channel table')
        return table.add_channels(count)

    def set_view(self, source: Any, view: View | None = None) -> None:
        """Set the view a geometry opens on in 3-D, in the app, the report
        and every exported figure.

        In the app, Set Default View captures the view on screen; from a
        script the view is given, for example a model built Y-up seen
        from the front and above. Everything drawn on the geometry (its
        shapes, an ODS, the DOF arrows) opens on this view too, and Reset
        View returns to it. The nodes are not rotated; only the viewing
        direction changes. An object that is not a geometry, or a `view`
        that is not a View, raises TypeError.

        Parameters
        ----------
        source : str or Geometry
            The geometry, by name or as the object itself.
        view : View, optional
            Where the eye is, seen from the model's center, and which way
            is up. None returns to the default isometric view (from
            +X+Y+Z, Z up).

        Returns
        -------
        None

        Examples
        --------
        >>> from visualdynamics import View
        >>> project.set_view('BARC', View(eye=(1, 1, -1), up=(0, 1, 0)))  # doctest: +SKIP
        """
        # The nodes are not turned, only how the model is looked at
        # (Brandon, 2026-09-27).
        name = self.name_of(source)
        geometry = self[name]
        if not isinstance(geometry, Geometry):
            raise TypeError(f'{name!r} is not a geometry')
        if view is not None and not isinstance(view, View):
            raise TypeError('view must be a View, or None for the default')
        geometry.view = view

    def tie_elements(self, source: Any, elements: Any, to: Any,
                     group: str | None = None) -> dict:
        """Tie a patch of a geometry's elements rigidly to the part under
        it (Tie), as for a bolted joint modeled from the elements its
        washer covers.

        Each node of the patch is joined by a rigid, massless link to the
        nearest node of `to` (`mesh.tie`).

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        elements : sequence of int
            The patch, by element id.
        to : str, int or sequence of int
            An element group, by name or id, or a second patch, by
            element ids.
        group : str, optional
            The element group the links go into, made rigid if new.
            Defaults to the geometry's first rigid element group, or a
            new one named 'ties'.

        Returns
        -------
        dict
            'links', how many were added; 'shared', patch nodes the
            target already holds; 'group', where the links went.
        """
        from .core import mesh

        name = self.name_of(source)
        geometry = self[name]
        if not isinstance(geometry, Geometry):
            raise TypeError(f'{name!r} is not a geometry')
        to = to if isinstance(to, (str, int)) else [int(e) for e in to]
        return mesh.tie(geometry, [int(e) for e in elements], to, group)

    def merge_groups(self, source: Any, groups: Any) -> dict:
        """Merge a geometry's element groups into one (Merge Element
        Groups). The first group keeps its id, name and properties, and
        the other groups' elements move into it.

        The merge is refused unless the groups hold the same element
        types and have the same material and thickness or section
        (`Geometry.merge_refusal`).

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        groups : sequence of int
            The element groups, by id; the first is the one kept.

        Returns
        -------
        dict
            'into', the element group kept; 'groups', how many merged
            into it; 'elements', how many elements moved.
        """
        name = self.name_of(source)
        geometry = self[name]
        if not isinstance(geometry, Geometry):
            raise TypeError(f'{name!r} is not a geometry')
        return geometry.merge_groups([int(b) for b in groups])

    def _solid_geometry(self, source: Any) -> tuple[str, Geometry]:
        name = self.name_of(source)
        geometry = self[name]
        if not isinstance(geometry, Geometry):
            raise TypeError(f'{name!r} is not a geometry')
        return name, geometry

    @staticmethod
    def _solid_scale(geometry: Geometry, unit: str) -> float:
        """What turns lengths typed in `unit` into the geometry's own:
        meters once its units are defined, as given before."""
        from .units import si_transform

        return (si_transform(unit, 'length')[0]
                if geometry.length_unit is not None else 1.0)

    def add_solid_block(self, source: Any, center: Any, widths: Any,
                        angles: Any = (0.0, 0.0, 0.0), name: str = '', *,
                        corner: float = 0.0, unit: str = 'm') -> int:
        """Add a block to a geometry's solids (`core.solids`). A solid is
        a shape to build a part from and mesh later; it adds no elements.

        The block is `widths` across along its own axes, centered at
        `center` and rotated by `angles`.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        center, widths : array_like
            Three numbers each, in `unit`.
        angles : array_like, default (0, 0, 0)
            Degrees about the geometry's X, then Y, then Z axes.
        name : str, optional
            What to call it.
        corner : float, default 0
            The radius its four edges along its own Z are rounded to, in
            `unit`. A window cut with a rounded block leaves fillets.
        unit : str, default 'm'
            The unit the lengths are in. A geometry whose units are not
            defined takes them as given.

        Returns
        -------
        int
            The solid's id.
        """
        from .core import solids

        _name, geometry = self._solid_geometry(source)
        scale = self._solid_scale(geometry, unit)
        return solids.add_block(geometry, np.asarray(center, dtype=float) * scale,
                                np.asarray(widths, dtype=float) * scale,
                                angles, name, float(corner) * scale)

    def add_solid_cylinder(self, source: Any, center: Any, radius: float,
                           length: float, angles: Any = (0.0, 0.0, 0.0),
                           name: str = '', *, unit: str = 'm') -> int:
        """Add a cylinder to a geometry's solids (`core.solids`), of
        `radius` about its own axis and `length` along it, centered at
        `center`.

        With no `angles` its axis is the geometry's Z. A cylinder
        subtracted from a part (`combine_solids`) makes a hole.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        center : array_like
            Three numbers, in `unit`.
        radius, length : float
            In `unit`.
        angles : array_like, default (0, 0, 0)
            Degrees about the geometry's X, then Y, then Z axes.
        name : str, optional
            What to call it.
        unit : str, default 'm'
            The unit the lengths are in. A geometry whose units are not
            defined takes them as given.

        Returns
        -------
        int
            The solid's id.
        """
        from .core import solids

        _name, geometry = self._solid_geometry(source)
        scale = self._solid_scale(geometry, unit)
        return solids.add_cylinder(geometry,
                                   np.asarray(center, dtype=float) * scale,
                                   float(radius) * scale, float(length) * scale,
                                   angles, name)

    def add_solid_cone(self, source: Any, center: Any, base_radius: float,
                       top_radius: float, height: float, angles: Any = (0.0, 0.0, 0.0),
                       name: str = '', *, unit: str = 'm') -> int:
        """Add a cone to a geometry's solids (`core.solids.add_cone`): a
        frustum `height` along its own axis, `base_radius` at its base
        and `top_radius` at its top, either zero for a point, centered at
        `center`.

        With no `angles` its axis is the geometry's Z, its base below.
        Bricks are fitted to it, since no sweep makes a cone.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        center : array_like
            Three numbers, in `unit`.
        base_radius, top_radius, height : float
            In `unit`.
        angles : array_like, default (0, 0, 0)
            Degrees about the geometry's X, then Y, then Z axes.
        name : str, optional
            What to call it.
        unit : str, default 'm'
            The unit the lengths are in. A geometry whose units are not
            defined takes them as given.

        Returns
        -------
        int
            The solid's id.
        """
        from .core import solids

        _name, geometry = self._solid_geometry(source)
        scale = self._solid_scale(geometry, unit)
        return solids.add_cone(geometry, np.asarray(center, dtype=float) * scale,
                               float(base_radius) * scale, float(top_radius) * scale,
                               float(height) * scale, angles, name)

    def add_solid_sphere(self, source: Any, center: Any, radius: float, name: str = '', *,
                         unit: str = 'm') -> int:
        """Add a sphere to a geometry's solids (`core.solids.add_sphere`),
        `radius` about `center`. Bricks are fitted to it.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        center : array_like
            Three numbers, in `unit`.
        radius : float
            In `unit`.
        name : str, optional
            What to call it.
        unit : str, default 'm'
            The unit the lengths are in. A geometry whose units are not
            defined takes them as given.

        Returns
        -------
        int
            The solid's id.
        """
        from .core import solids

        _name, geometry = self._solid_geometry(source)
        scale = self._solid_scale(geometry, unit)
        return solids.add_sphere(geometry, np.asarray(center, dtype=float) * scale,
                                 float(radius) * scale, name)

    def add_solid_plate(self, source: Any, center: Any, widths: Any,
                        angles: Any = (0.0, 0.0, 0.0), name: str = '', *,
                        unit: str = 'm') -> int:
        """Add a plate to a geometry's solids (`core.solids.add_plate`): a
        flat rectangle `widths` across along its own X and Y, centered at
        `center`.

        With no `angles` it lies in the geometry's XY plane. A plate has
        no thickness of its own: it is meshed into plates (`mesh_solids`
        with shells), and their element group takes the thickness. A
        cylinder subtracted from it (`combine_solids`) makes a hole.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        center : array_like
            Three numbers, in `unit`.
        widths : array_like
            Two numbers, in `unit`: along its own X, then Y.
        angles : array_like, default (0, 0, 0)
            Degrees about the geometry's X, then Y, then Z axes.
        name : str, optional
            What to call it.
        unit : str, default 'm'
            The unit the lengths are in. A geometry whose units are not
            defined takes them as given.

        Returns
        -------
        int
            The solid's id.
        """
        from .core import solids

        _name, geometry = self._solid_geometry(source)
        scale = self._solid_scale(geometry, unit)
        return solids.add_plate(geometry, np.asarray(center, dtype=float) * scale,
                                np.ravel(np.asarray(widths, dtype=float))[:2] * scale,
                                angles, name)

    def add_solid_disk(self, source: Any, center: Any, radius: float,
                       angles: Any = (0.0, 0.0, 0.0), name: str = '', *,
                       unit: str = 'm') -> int:
        """Add a disk to a geometry's solids (`core.solids.add_disk`): a
        flat circle of `radius` in its own XY plane, centered at
        `center`.

        With no `angles` it lies in the geometry's XY plane. As a
        plate's, its thickness is set on the element group it is meshed
        into (`mesh_solids` with shells).

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        center : array_like
            Three numbers, in `unit`.
        radius : float
            In `unit`.
        angles : array_like, default (0, 0, 0)
            Degrees about the geometry's X, then Y, then Z axes.
        name : str, optional
            What to call it.
        unit : str, default 'm'
            The unit the lengths are in. A geometry whose units are not
            defined takes them as given.

        Returns
        -------
        int
            The solid's id.
        """
        from .core import solids

        _name, geometry = self._solid_geometry(source)
        scale = self._solid_scale(geometry, unit)
        return solids.add_disk(geometry, np.asarray(center, dtype=float) * scale,
                               float(radius) * scale, angles, name)

    def add_solid_section(self, source: Any, shape: str, center: Any,
                          dimensions: Any, length: float,
                          angles: Any = (0.0, 0.0, 0.0), name: str = '', *,
                          unit: str = 'm') -> int:
        """Add a structural section to a geometry's solids
        (`core.solids.add_section`): an I-beam, channel, tee, angle,
        rectangular tube or round tube, `length` along its own axis.

        It is sized by its own dimensions, which `edit_solid` can change
        later.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        shape : str
            One of `core.solids.SECTIONS`: 'I-beam', 'channel', 'tee',
            'angle', 'rectangular tube' or 'round tube'.
        center : array_like
            Three numbers, in `unit`: the center of the box it fills.
        dimensions : sequence of float
            In `unit`, in the order `SECTIONS[shape]` names them. For an
            I-beam these are the depth, flange width, flange thickness
            and web thickness, the same as the beam element's section.
        length : float
            In `unit`.
        angles : array_like, default (0, 0, 0)
            Degrees about the geometry's X, then Y, then Z axes. With
            none it runs along Z.
        name : str, optional
            What to call it.
        unit : str, default 'm'
            The unit the lengths are in. A geometry whose units are not
            defined takes them as given.

        Returns
        -------
        int
            The solid's id.
        """
        from .core import solids

        _name, geometry = self._solid_geometry(source)
        scale = self._solid_scale(geometry, unit)
        return solids.add_section(geometry, shape,
                                  np.asarray(center, dtype=float) * scale,
                                  np.asarray(dimensions, dtype=float) * scale,
                                  float(length) * scale, angles, name)

    def add_solid_hole(self, source: Any, center: Any, diameter: float,
                       depth: float | None = None, angles: Any = (0.0, 0.0, 0.0),
                       name: str = '', *, counterbore: Any = None,
                       unit: str = 'm') -> int:
        """Add a hole to a geometry's solids (`core.solids.add_hole`), to
        be subtracted from a part with `combine_solids`.

        The hole enters at `center` and runs `depth` along its own -Z,
        or through the part it is cut from when `depth` is None. A
        counterbore makes it wider at the entry.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        center : array_like
            Where it enters, three numbers in `unit`.
        diameter : float
            In `unit`.
        depth : float, optional
            In `unit`. None for a through hole.
        angles : array_like, default (0, 0, 0)
            Degrees about the geometry's X, then Y, then Z axes. With
            none it runs down Z.
        name : str, optional
            What to call it.
        counterbore : (float, float), optional
            Its diameter and depth, in `unit`.
        unit : str, default 'm'
            The unit the lengths are in. A geometry whose units are not
            defined takes them as given.

        Returns
        -------
        int
            The solid's id.
        """
        from .core import solids

        _name, geometry = self._solid_geometry(source)
        scale = self._solid_scale(geometry, unit)
        return solids.add_hole(
            geometry, np.asarray(center, dtype=float) * scale, float(diameter) * scale,
            None if depth is None else float(depth) * scale, angles, name,
            None if counterbore is None else
            tuple(float(v) * scale for v in counterbore))

    def _data_nodes(self) -> set[int]:
        """Every node some data in the project is measured or solved at:
        the nodes its DOFs name — records' responses and references,
        shapes' coordinates, the channel table's nodes."""
        found: set[int] = set()
        for obj in self.values():
            found |= _dof_nodes(obj)
            # a channel table's nodes: its frame's column, blanks left out
            frame = getattr(obj, 'frame', None)
            if frame is not None and 'node' in getattr(frame, 'columns', ()):
                found.update(int(n) for n in frame['node'].dropna())
        return found

    def delete_elements(self, source: Any, ids: Any) -> dict:
        """Delete a geometry's elements and the nodes only they used,
        except nodes where data in the project is measured or solved
        (`Geometry.delete_elements`).

        Deleting a meshed part's bricks removes its nodes, while deleting
        a test geometry's panels leaves its measurement points.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        ids : sequence of int
            The elements, by id.

        Returns
        -------
        dict
            'elements' and 'nodes', how many were deleted.
        """
        # Nodes data uses are kept (Brandon, 2026-10-12).
        _name, geometry = self._solid_geometry(source)
        return geometry.delete_elements([int(i) for i in ids], nodes=True,
                                        keep=sorted(self._data_nodes()))

    def delete_element_groups(self, source: Any, ids: Any) -> dict:
        """Delete a geometry's element groups with their elements and the
        nodes only they used, except nodes where data in the project is
        measured or solved (`Geometry.delete_groups`).

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        ids : sequence of int
            The element groups, by id.

        Returns
        -------
        dict
            'groups', 'elements' and 'nodes', how many were deleted.
        """
        _name, geometry = self._solid_geometry(source)
        return geometry.delete_groups([int(i) for i in ids],
                                      keep=sorted(self._data_nodes()))

    def edit_solid(self, source: Any, solid_id: int, *, center: Any = None,
                   widths: Any = None, angles: Any = None, radius: Any = None,
                   length: Any = None, corner: Any = None, counts: Any = None,
                   spacing: Any = None, dimensions: Any = None, name: Any = None,
                   top_radius: Any = None, unit: str = 'm') -> None:
        """Change a solid after it is made (`core.solids.edit`). Anything
        not given is kept.

        The values that can be changed are a block's center, widths,
        angles and corner; a cylinder's center, radius, length and
        angles; a cone's center, radius (at its base), top radius,
        length (its height) and angles; a sphere's center and radius; a
        plate's center, widths and angles; a disk's center, radius and
        angles; a section's center, dimensions, length and angles; a
        hole's center, dimensions and angles; a pattern's counts and
        spacing; and any solid's name. A boolean the solid is part of
        changes with it. A mesh made from it does not, until Mesh Solids
        meshes it again.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        solid_id : int
            Which solid.
        center, widths, angles, radius, length, corner, name : optional
            As `add_solid_block` and `add_solid_cylinder` take them,
            lengths in `unit`.
        counts, spacing : optional
            As `pattern_solids` takes them.
        top_radius : float, optional
            A cone's, in `unit`.
        dimensions : sequence of float, optional
            A section's, as `add_solid_section` takes them, or a hole's:
            its diameter, its depth (zero for a through hole) and its
            counterbore's diameter and depth (zeros for none), in `unit`.
        unit : str, default 'm'
            The unit the lengths are in. A geometry whose units are not
            defined takes them as given.

        Returns
        -------
        None
        """
        from .core import solids

        _name, geometry = self._solid_geometry(source)
        scale = self._solid_scale(geometry, unit)
        changes = {key: value for key, value in (
            ('center', center), ('widths', widths), ('angles', angles),
            ('radius', radius), ('length', length), ('corner', corner),
            ('counts', counts), ('spacing', spacing), ('dimensions', dimensions),
            ('top_radius', top_radius), ('name', name)) if value is not None}
        for key in ('center', 'widths', 'spacing', 'dimensions'):
            if key in changes:
                changes[key] = np.asarray(changes[key], dtype=float) * scale
        for key in ('radius', 'length', 'corner', 'top_radius'):
            if key in changes:
                changes[key] = float(changes[key]) * scale
        solids.edit(geometry, int(solid_id), **changes)

    def move_solids(self, source: Any, ids: Any, offset: Any = (0.0, 0.0, 0.0),
                    turn: Any = (0.0, 0.0, 0.0), *, unit: str = 'm') -> int:
        """Move whole solids (`core.solids.move`): a part and every solid
        it is made of, rotated by `turn` about the center of their extent
        and then translated by `offset`.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        ids : sequence of int
            The solids, by id.
        offset : array_like, default (0, 0, 0)
            The translation, in `unit`.
        turn : array_like, default (0, 0, 0)
            Degrees about the geometry's X, then Y, then Z axes.
        unit : str, default 'm'
            The unit the offset is in. A geometry whose units are not
            defined takes it as given.

        Returns
        -------
        int
            How many primitive solids moved (blocks, cylinders, sections
            and holes).
        """
        from .core import solids

        _name, geometry = self._solid_geometry(source)
        scale = self._solid_scale(geometry, unit)
        return solids.move(geometry, ids, np.asarray(offset, dtype=float) * scale,
                           turn)

    def combine_solids(self, source: Any, operation: str, ids: Any,
                       name: str = '') -> int:
        """Add a boolean of a geometry's solids: 'union', 'subtract' (the
        others subtracted from the first) or 'intersect'.

        The solids combined become part of the new solid, and only the
        new solid is listed as a part. A solid already part of another
        boolean is refused.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        operation : str
            'union', 'subtract' or 'intersect'.
        ids : sequence of int
            Two or more solids, by id, in order.
        name : str, optional
            What to call it. Defaults to the first solid's name.

        Returns
        -------
        int
            The new solid's id.
        """
        from .core import solids

        _name, geometry = self._solid_geometry(source)
        return solids.combine(geometry, operation, ids, name)

    def reorder_solids(self, source: Any, ids: Any, *,
                       within: int | None = None) -> None:
        """Put a geometry's solids in a new order (a drag in the tree):
        its parts, or with `within` one boolean's own solids. A
        subtraction cuts every solid after its first from the first, so
        the order decides what is cut (`core.solids.reorder`).

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        ids : sequence of int
            Every one of the solids being ordered, in the new order.
        within : int, optional
            The boolean or pattern whose solids these are; the parts
            when omitted.

        Returns
        -------
        None
            The geometry is changed in place.
        """
        from .core import solids

        _name, geometry = self._solid_geometry(source)
        solids.reorder(geometry, ids, within)

    def copy_solids(self, source: Any, ids: Any, *,
                    into: Any = None) -> list[int]:
        """Copy a geometry's solids, recipe and all, into the same
        geometry or another (Copy and Paste on solid rows). Returns the
        new top-level ids.

        A boolean or a pattern is copied with every solid it is made of
        (`core.solids.copy`), so the copy is edited on its own. A named
        solid's copy is '<name> copy'.

        Parameters
        ----------
        source : str or object
            The geometry the solids are in, by name or as the object.
        ids : sequence of int
            The solids, by id.
        into : str or object, optional
            The geometry the copies go into; the same one when omitted.
            Both geometries hold their coordinates the same way, with
            units defined or not, or the copy is refused.

        Returns
        -------
        list of int
            The new solids' ids, in the order of `ids`.
        """
        from .core import solids

        _name, geometry = self._solid_geometry(source)
        target = geometry if into is None else self._solid_geometry(into)[1]
        return solids.copy(geometry, ids, into=target)

    def pattern_solids(self, source: Any, ids: Any, counts: Any, spacing: Any,
                       *, unit: str = 'm') -> int:
        """Pattern a geometry's solids in rows, columns and layers
        (Pattern): `counts` along X, Y and Z, the originals included,
        `spacing` apart.

        The pattern is one solid made from the originals
        (`core.solids.pattern`), so changing an original changes every
        copy. Its counts and spacing are changed with `edit_solid`.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        ids : sequence of int
            The solids to copy, by id.
        counts : sequence of int
            How many along X, Y and Z.
        spacing : sequence of float
            How far apart along X, Y and Z, in `unit`.
        unit : str, default 'm'
            The unit the spacing is in. A geometry whose units are not
            defined takes it as given.

        Returns
        -------
        int
            The pattern's id.
        """
        from .core import solids

        _name, geometry = self._solid_geometry(source)
        scale = self._solid_scale(geometry, unit)
        return solids.pattern(geometry, ids, counts,
                              np.asarray(spacing, dtype=float) * scale)

    def mesh_solids(self, source: Any, ids: Any, size: float | None = None, *,
                    unit: str = 'm', apart: bool = False,
                    parts_apart: bool = False, elements: str = 'hexes',
                    progress: Any = None, sizes: Any = None) -> dict:
        """Mesh a geometry's solids into bricks, tetrahedra or shells
        (Mesh Solids). Each part becomes an element group named for it.

        With `elements` 'hexes', parts that are an extrusion along X, Y
        or Z are meshed on one grid cut at every flat face, with their
        holes rounded (`core.solids.extrusion_mesh`); a union that is
        no extrusion in pieces, each operand swept where it is one and
        fitted where it is not, at its own size, joined where the
        grids meet and tied at the seams where they do not; and any
        other shape on a grid fitted to it (`core.hexmesh.overlay_mesh`);
        `core.solids.brick_mesh` picks. With 'tetras', the parts are
        meshed into ten-node tetrahedra from their recipe's signed
        distance (`core.tetmesh.tet_mesh`). With 'quads' or 'triangles'
        they are meshed into shells (`core.shells.shell_mesh`): a plate
        or a disk, with what is cut from it, on its plane, and a solid
        with a volume on its outer surface, the surface of its bricks
        for quads and of its tetrahedra for triangles; the thickness is
        set on the element group. A plate or a disk is meshed into
        shells only. Parts
        meshed together share nodes where they touch. The solids are
        kept so they can be meshed again, and a part meshed again
        replaces its earlier mesh: the elements in the element group of
        its name are deleted, and the group and its material are kept.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        ids : sequence of int
            The parts, by id: solids no other solid uses.
        size : float, optional
            The target element size, in `unit`. Defaults to half the
            thinnest feature (`core.solids.mesh_size`; a plate's or a
            disk's measured across its plane, `core.shells.mesh_size`).
        unit : str, default 'm'
            The unit the size is in. A geometry whose units are not
            defined takes it as given.
        apart : bool, default False
            Join the bricks only to nodes of their own element groups
            (`mesh.join`), so a part meshed beside an existing one keeps
            nodes of its own where they touch.
        parts_apart : bool, default False
            Give each part meshed here its own nodes where the parts
            touch each other. They are still on one grid, so their nodes
            coincide and can be merged later where bolts go. Bricks
            only.
        elements : {'hexes', 'tetras', 'quads', 'triangles'}, default 'hexes'
            The element type, as the tree names the families; 'bricks'
            and 'tets' are taken to mean the same. 'quads' and
            'triangles' are shells.
        progress : callable, optional
            Called with ``(done, total)`` as the mesh goes
            (`core.solids.extrusion_mesh`, `core.tetmesh.tet_mesh`). The
            app's progress bar uses it; a script can print it.
        sizes : mapping of int to float, optional
            A size per piece of a union meshed in pieces, by solid id,
            in `unit`; a piece without one takes `size`. Bricks only.

        Returns
        -------
        dict
            For bricks, what `brick_mesh` gives, 'method' included (in
            pieces, 'pieces' lists each piece's readings, 'seam_shared'
            the nodes shared at the seams, 'ties' the nodes joined there
            by rigid links and 'seams' the mated surfaces that join
            them); for tetrahedra, 'tets', 'nodes', 'steps', 'quality',
            'poor', 'slivers' and 'shortest' as `tet_mesh` gives them;
            for shells, what `shell_mesh` gives; and every way 'added', 'shared' and 'groups' as `mesh.join`
            gives them, 'replaced', how many elements of the parts'
            earlier mesh were replaced, 'mates_dropped', how many
            mated surfaces on that mesh went with it, and 'cleared',
            {group name: what was taken off it} for each group the
            solids went into that carried a section, a thickness or an
            orientation (solids take a material alone, which is kept).
        """
        from .core import mesh, shells, solids, tetmesh

        elements = {'bricks': 'hexes', 'tets': 'tetras'}.get(elements, elements)
        if elements not in ('hexes', 'tetras', 'quads', 'triangles'):
            raise ValueError("elements is 'hexes', 'tetras', 'quads' or 'triangles', "
                             f'not {elements!r}')
        _name, geometry = self._solid_geometry(source)
        ids = [int(i) for i in ids]
        shell = elements in ('quads', 'triangles')
        flat = [i for i in ids if solids.is_flat(geometry, i)]
        if flat and not shell:
            raise ValueError(f'{solids.piece_name(geometry, flat[0])} is flat: a plate '
                             'or a disk is meshed into shells, quads or triangles')
        held = ((shells.mesh_size if shell else solids.mesh_size)(geometry, ids)
                if size is None
                else float(size) * self._solid_scale(geometry, unit))
        scale = self._solid_scale(geometry, unit)
        if shell:
            part, made = shells.shell_mesh(geometry, ids, held,
                                           triangles=elements == 'triangles',
                                           progress=progress)
        elif elements == 'tetras':
            part, made = tetmesh.tet_mesh(geometry, ids, held, progress=progress)
        else:
            part, made = solids.brick_mesh(
                geometry, ids, held, parts_apart, progress=progress,
                sizes={int(k): float(v) * scale for k, v in (sizes or {}).items()})
        # meshed again, a part replaces its earlier mesh (Brandon,
        # 2026-10-12): the bricks of the element groups this mesh joins
        # by name go first, with the nodes only they used unless data
        # uses them. The groups stay, and their materials with them.
        # Left in, the new bricks that landed on old ones were left out
        # as already there, and a table meshed again without its legs
        # kept the legs' bricks. The rigid links on those bricks' nodes,
        # the seams' ties above all, go with them: left in, they would
        # hold the old nodes and point the new mesh at them
        replaced = [int(e) for e in solids.meshed_elements(geometry, part.group_name)]
        # and the mated surfaces on them: their nodes are the old mesh's
        # (a union meshed in pieces brings its seams again, in `part`)
        from .core import mates

        names = list(geometry.group_name)
        remeshed = [int(geometry.group_id[names.index(n)]) for n in part.group_name
                    if n and n in names]
        dropped = mates.delete(geometry, mates.naming_groups(geometry, remeshed))
        if replaced:
            replaced += [int(e) for e in mesh.links_on(geometry, replaced)]
            geometry.delete_elements(replaced, nodes=True, keep=sorted(self._data_nodes()))
        found = mesh.join(geometry, part, apart=apart)
        cleared = _solids_take_a_material_alone(geometry, found['groups'])
        return {**made, 'added': found['added'], 'shared': found['shared'],
                'groups': found['groups'], 'replaced': len(replaced),
                'mates_dropped': dropped, 'cleared': cleared}

    def detect_mated_surfaces(self, source: Any, groups: Any = None,
                              tolerance: float | None = None, *,
                              unit: str = 'm') -> list[dict]:
        """The patches where element groups' surfaces mate
        (`core.mates.detect`): the nodes on one group's outer faces that
        lie on another's, within `tolerance`, one patch per connected
        piece. The finer side is the first, and follows.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        groups : sequence of int or str, optional
            The element groups, by id or name; every group of plates or
            solids when omitted.
        tolerance : float, optional
            How far a node may be from the other's faces, in `unit`; a
            millionth of the geometry's size when omitted.
        unit : str, default 'm'
            The unit the tolerance is in.

        Returns
        -------
        list of dict
            Each patch's 'groups' (first side's and second side's ids),
            'nodes_a' and 'nodes_b'.
        """
        from .core import mates

        _name, geometry = self._solid_geometry(source)
        held = (None if tolerance is None
                else float(tolerance) * self._solid_scale(geometry, unit))
        ids = None if groups is None else [self._group_of(geometry, g) for g in groups]
        return mates.detect(geometry, ids, held)

    @staticmethod
    def _group_of(geometry: Geometry, group: Any) -> int:
        """An element group's id, from its id or its name."""
        if isinstance(group, str):
            names = list(geometry.group_name)
            if group not in names:
                raise ValueError(f'no element group {group!r}')
            return int(geometry.group_id[names.index(group)])
        return int(group)

    def add_mated_surfaces(self, source: Any, groups: Any, nodes_a: Any, nodes_b: Any, *,
                           normal_mode: str = 'rigid', shear_mode: str = 'rigid',
                           normal_stiffness: float = 0.0, shear_stiffness: float = 0.0,
                           name: str | None = None) -> int:
        """Join two element groups where their surfaces mate (Mate
        Surfaces): when a model is built, each of `nodes_a` is tied to the
        nearest of `nodes_b`, along the normal of the surface it lies on
        and in that surface's plane, each held rigidly, by a stiffness or
        not at all (`fem.Model.add_tie`). The set is kept on the geometry
        (`geometry.mates`), to be edited or deleted later.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        groups : (int or str, int or str)
            The first side's element group and the second's, by id or
            name.
        nodes_a, nodes_b : sequence of int
            The first side's nodes, which follow, and the second's.
        normal_mode, shear_mode : {'rigid', 'stiffness', 'free'}
            How the normal and the plane are held.
        normal_stiffness, shear_stiffness : float
            Each node's stiffness along the normal and in the plane, in
            N/m, where its mode is 'stiffness'.
        name : str, optional
            What to call the set; "first on second" when omitted.

        Returns
        -------
        int
            The set's id.
        """
        from .core import mates

        _name, geometry = self._solid_geometry(source)
        pair = tuple(self._group_of(geometry, g) for g in groups)
        return mates.add(geometry, pair, nodes_a, nodes_b, normal_mode, shear_mode,
                         normal_stiffness, shear_stiffness, name).id

    def mate_surfaces(self, source: Any, groups: Any = None, tolerance: float | None = None,
                      *, unit: str = 'm', normal_mode: str = 'rigid',
                      shear_mode: str = 'rigid', normal_stiffness: float = 0.0,
                      shear_stiffness: float = 0.0) -> list[int]:
        """Join every patch where the element groups' surfaces mate, as
        found (`detect_mated_surfaces`), one set each
        (`add_mated_surfaces`).

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        groups : sequence of int or str, optional
            The element groups; every group of plates or solids when
            omitted.
        tolerance : float, optional
            As `detect_mated_surfaces` takes it, in `unit`.
        unit : str, default 'm'
            The unit the tolerance is in.
        normal_mode, shear_mode, normal_stiffness, shear_stiffness
            As `add_mated_surfaces` takes them.

        Returns
        -------
        list of int
            The sets' ids. None found raises ValueError.
        """
        found = self.detect_mated_surfaces(source, groups, tolerance, unit=unit)
        if not found:
            raise ValueError('no surface of these element groups lies on another')
        return [self.add_mated_surfaces(source, patch['groups'], patch['nodes_a'],
                                        patch['nodes_b'], normal_mode=normal_mode,
                                        shear_mode=shear_mode,
                                        normal_stiffness=normal_stiffness,
                                        shear_stiffness=shear_stiffness)
                for patch in found]

    def edit_mated_surfaces(self, source: Any, mate_id: int, **fields: Any) -> None:
        """Change a set of mated surfaces: its `name`, `nodes_a`,
        `nodes_b`, `normal_mode`, `shear_mode`, `normal_stiffness` or
        `shear_stiffness`
        (`core.mates.change`), refused as `add_mated_surfaces` refuses.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        mate_id : int
            The set.
        **fields
            The fields to change, stiffnesses in N/m.

        Returns
        -------
        None
        """
        from .core import mates

        _name, geometry = self._solid_geometry(source)
        mates.change(geometry, mate_id, **fields)

    def delete_mated_surfaces(self, source: Any, ids: Any) -> int:
        """Delete sets of mated surfaces, by id; returns how many went.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        ids : sequence of int
            The sets.

        Returns
        -------
        int
        """
        from .core import mates

        _name, geometry = self._solid_geometry(source)
        return mates.delete(geometry, ids)

    def delete_solids(self, source: Any, ids: Any) -> int:
        """Delete a geometry's solids, each with every solid it is made
        of (`core.solids.delete`).

        A solid that a boolean still uses is refused unless the boolean
        is deleted with it.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        ids : sequence of int
            The solids, by id.

        Returns
        -------
        int
            How many solids were deleted, the ones inside included.
        """
        from .core import solids

        _name, geometry = self._solid_geometry(source)
        return solids.delete(geometry, ids)

    def separate_solids(self, source: Any, ids: Any) -> int:
        """Take a geometry's booleans and patterns apart (Take Apart).
        Each one is removed, and its operands are parts again
        (`core.solids.separate`).

        A primitive solid, or one that is still part of another, is
        refused.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        ids : sequence of int
            The booleans or patterns, by id.

        Returns
        -------
        int
            How many solids are parts again.
        """
        from .core import solids

        _name, geometry = self._solid_geometry(source)
        return solids.separate(geometry, ids)

    def solve_modes(self, source: Any, *,
                    maximum_frequency: float | None = None,
                    num_modes: int | None = None, damping: float = 0.0,
                    name: str | None = None, progress: Any = None) -> str:
        """Solve the normal modes of a geometry whose element groups have
        properties (Solve Modes). The shapes are added in the geometry's
        group.

        The finite element model is built from the geometry
        (`fem.Model.from_geometry`). Each element group has a material
        and a thickness or a section (`fem.GroupProperties`, set in the
        Element Groups table or on `geometry.group_properties`); each
        quad becomes a plate element, each triangle a triangular plate
        element and each two-node line a beam. The model is solved
        free-free, so the six rigid-body modes come back at 0 Hz,
        followed by the elastic modes. A geometry with no element group
        properties raises ValueError.

        Parameters
        ----------
        source : str or object
            The geometry, by name or as the object itself.
        maximum_frequency : float, optional
            Solve for every mode up to this frequency, in Hz.
        num_modes : int, optional
            Solve for this many modes instead, rigid-body modes included.
        damping : float, default 0.0
            The fraction of critical damping given to every mode. A
            finite element model has no damping of its own.
        name : str, optional
            What to call the result. Defaults to the geometry's name
            with ' Modes' after it.
        progress : callable, optional
            Called with ``(done, total)`` as the solve advances, and with
            a total of `core.progress.BUSY` while a stage runs that
            cannot count itself. The app's progress bar uses it; a
            script can print it.

        Returns
        -------
        str
            The name the shape set was added under.
        """
        from .core.fem import Model

        source = self.name_of(source)
        geometry = self[source]
        if not isinstance(geometry, Geometry):
            raise TypeError(f'{source!r} is not a geometry')
        if not geometry.group_properties:
            raise ValueError(
                f'{source} has no element group properties: give each element group a '
                'material and a thickness or a section, in the Element Groups '
                'table or on geometry.group_properties')
        model = Model.from_geometry(geometry, name=source)
        shapes = model.eigensolution(maximum_frequency=maximum_frequency,
                                     num_modes=num_modes, damping=damping,
                                     progress=progress)
        params = {'maximum_frequency': maximum_frequency,
                  'num_modes': num_modes, 'damping': damping}
        return self._derive(source, shapes, name or f'{source} Modes',
                            recipe=('solve_modes', params))

    def plan_instrumentation(self, source: Any, triaxial: int, uniaxial: int = 0, *,
                             modes: Any = None, groups: Any = None,
                             surfaces: Any = None, edge_angle: float | None = None,
                             axes: Any = (2,), name: str | None = None,
                             progress: Any = None, edge_distance: float | None = None,
                             unit: str = 'm') -> dict[str, str]:
        """Plan a test's accelerometers from finite element modes, by
        Effective Independence.

        The model's outer surface is split into surfaces at its sharp
        edges (`core.instrumentation.skin`), and every place on the
        surfaces kept a sensor can sit is a candidate, measured in its own system with +Z out of the surface
        and +X in the plane of a system axis (`core.instrumentation.plan`).
        Tri-axial accelerometers are placed first, then uni-axial ones
        along the surface normal. Three objects are added and linked: the
        experimental geometry of the chosen places, the channel table, and
        the modes at exactly those channels.

        Parameters
        ----------
        source : str or object
            The shape set, by name or as the object itself. It is read
            on the geometry in its object group, or the active one.
        triaxial : int
            How many tri-axial accelerometers.
        uniaxial : int, default 0
            How many uni-axial accelerometers.
        modes : sequence of int, optional
            The modes to measure, by index. Every mode above 0 Hz when
            omitted.
        groups : sequence of int or str, optional
            The element groups sensors may go on, by id or name. All
            when omitted.
        surfaces : sequence of int, optional
            The surfaces sensors may go on, by their number in
            `core.instrumentation.skin`. When omitted, every surface of
            `groups` of at least 1 % of the outside's area
            (`core.instrumentation.default_surfaces`).
        edge_angle : float, optional
            Degrees between two neighboring faces' normals beyond which
            they are on separate surfaces;
            `core.instrumentation.EDGE_ANGLE` when omitted. A coarse
            curved mesh may need more.
        axes : sequence of int, default (2,)
            The directions a uni-axial accelerometer may measure, 0, 1, 2
            for X, Y, Z of its place: Z, the surface normal, by default.
        name : str, optional
            The start of the three names. The shape set's name with
            ' Plan' after it when omitted.
        progress : callable, optional
            Called with ``(0, core.progress.BUSY)`` as the work starts,
            which cannot count its own stages; the app's progress bar
            uses it.
        edge_distance : float, optional
            How near a sensor may be to the edge of its surface, in
            `unit`: a sharp edge, a free edge, a hole or where parts
            touch. A quarter of an inch when omitted
            (`core.instrumentation.EDGE_DISTANCE`), and 0 for a geometry
            whose units are not defined; 0 allows a sensor on the edge.
        unit : str, default 'm'
            The unit `edge_distance` is in. A geometry whose units are
            not defined takes it as given.

        Returns
        -------
        dict
            The names added, under 'geometry', 'channels' and 'shapes'.

        Raises
        ------
        TypeError
            When `source` is not a shape set.
        ValueError
            When it has no geometry to plan on, or the plan cannot be
            made (no sensors, fewer places than sensors, no mode chosen).

        Warns
        -----
        core.instrumentation.FewerDirectionsWarning
            When the sensors measure fewer directions than there are
            modes: the plan is made for the lowest modes that many, and
            the shapes kept still hold every mode asked for, so their
            auto-MAC shows which the sensors cannot tell apart.
        """
        from .core.instrumentation import FewerDirectionsWarning, plan, shortfall
        from .core.progress import BUSY

        if progress is not None:
            progress(0, BUSY)
        source = self.name_of(source)
        shapes = self[source]
        if not isinstance(shapes, ShapeSet):
            raise TypeError(f'{source!r} is not a shape set')
        found = self.geometry_for(source)
        if found is None:
            raise ValueError(f'{source} has no geometry to plan the instrumentation on')
        geometry_name, geometry = found
        ids = None if groups is None else [self._group_of(geometry, g) for g in groups]
        kept = None if surfaces is None else [int(s) for s in surfaces]
        if edge_distance is not None:
            edge_distance = float(edge_distance) * self._solid_scale(geometry, unit)
        made = plan(shapes, geometry, triaxial, uniaxial, modes=modes, groups=ids,
                    surfaces=kept, edge_angle=edge_angle, axes=axes,
                    edge_distance=edge_distance)
        short = shortfall(made.placement, made.shapes.num_shapes)
        if short:
            warnings.warn(FewerDirectionsWarning(f'{source}: {short}'), stacklevel=2)
        base = name or f'{source} Plan'
        params = {'triaxial': int(triaxial), 'uniaxial': int(uniaxial),
                  'modes': None if modes is None else [int(m) for m in modes],
                  'groups': ids, 'surfaces': kept,
                  'edge_angle': None if edge_angle is None else float(edge_angle),
                  'edge_distance': None if edge_distance is None else float(edge_distance),
                  'axes': [int(a) for a in axes],
                  'geometry': geometry_name}
        added = {
            'geometry': self._derive(source, made.geometry, f'{base} Geometry',
                                     recipe=('plan_instrumentation', params), link=False),
        }
        added['channels'] = self.add(f'{base} Channels', made.channels)
        added['shapes'] = self.add(f'{base} Shapes', made.shapes)
        self.link(added['geometry'], added['channels'], added['shapes'])
        return added

    def convert_to_specification(self, name: Any) -> str:
        """Convert a PSD or CPSD to a specification, the requirement
        measurements are judged against, keeping its name, object group
        and every reference to it.

        Some formats have no way to mark a requirement (the ESCDF data
        types do not), so a file can only suggest one by its name, and a
        name is not evidence. The person who knows marks it here
        instead. An object that is not a PSD, or is already a
        specification, raises TypeError.

        Parameters
        ----------
        name : str or Psd
            The PSD.

        Returns
        -------
        str
            Its name.
        """
        name = self.name_of(name)
        obj = self[name]
        if not isinstance(obj, Psd) or isinstance(obj, Specification):
            raise TypeError(f'{name} is not a PSD read as data')
        self[name] = obj.as_specification()
        return name

    def convert_to_psd(self, name: Any) -> str:
        """Convert a specification to a plain PSD or CPSD, under its own
        name. This undoes `convert_to_specification`, and makes a
        target written on the Specification sheet a CPSD to drive or
        predict with.

        A PSD cannot hold warning and abort limits, so a specification's
        limits are left out; asking for the conversion is asking for
        that (`Specification.limit` takes one out as a PSD of its own
        first). An object that is not a specification raises TypeError.

        Parameters
        ----------
        name : str or Specification
            The specification.

        Returns
        -------
        str
            Its name.
        """
        name = self.name_of(name)
        obj = self[name]
        if not isinstance(obj, Specification):
            raise TypeError(f'{name} is not a specification')
        self[name] = obj.as_psd(drop_limits=True)
        return name

    def scale_family(self, measured: Any) -> list[str]:
        """The measured PSDs that share one comparison scale: the object
        and every PSD in its object group that is not a specification,
        such as its source or the octave bands made from it.

        The family has one scale because an octave-band PSD is the same
        measurement on another frequency grid, and two scales for one
        run would contradict each other.

        Parameters
        ----------
        measured : str or Psd
            A measured PSD.

        Returns
        -------
        list of str
            The family's names, `measured` among them.
        """
        from .core.data import Psd, Specification

        name = self.name_of(measured)
        group = self.object_group_of(name) or [name]
        if name not in group:
            group = [name, *group]
        return [n for n in group
                if isinstance(self.get(n), Psd)
                and not isinstance(self.get(n), Specification)]

    def comparison_scale(self, specification: Any, measured: Any) -> int:
        """The decibels a measured PSD is scaled by when it is compared
        with a specification. The comparison's plot, bars and report all
        use this value.

        A scale held on any member of the family is used
        (`set_comparison_scale` keeps the family equal). Otherwise it is
        detected on the family's narrowband member, so the same data on
        octave bands cannot round to a different decibel than the lines
        it came from.

        Parameters
        ----------
        specification : str or Specification
            What is compared against.
        measured : str or Psd
            The measurement.

        Returns
        -------
        int
            Whole decibels.
        """
        from .core.compliance import comparison_scale_db

        # an object as itself: the window hands over a specification it
        # may have banded for the plot, which is in no project
        spec = (self[specification] if isinstance(specification, str)
                else specification)
        name = self.name_of(measured)
        family = self.scale_family(name)
        held = [self[n].scale_db for n in family
                if self[n].scale_db is not None]
        if held:
            return int(held[0])
        roots = [n for n in family
                 if getattr(self[n], 'bandwidth', None) is None]
        return int(comparison_scale_db(spec, self[roots[0]] if roots
                                       else self[name]))

    def set_comparison_scale(self, measured: Any,
                             db: int | None) -> list[str]:
        """Hold a comparison scale on a measured PSD and its whole family
        (`scale_family`), or with None, go back to detecting it from the
        data.

        The scale is stored on the objects, not on a view, so the report
        compares exactly what the screen shows and a saved project keeps
        the choice. The data itself is never changed. A value that is
        not a whole number raises ValueError.

        Parameters
        ----------
        measured : str or Psd
            A measured PSD.
        db : int or None
            Whole decibels, or None for automatic.

        Returns
        -------
        list of str
            The PSDs the value was set on.
        """
        if db is not None:
            if isinstance(db, float) and not db.is_integer():
                raise ValueError('a comparison scale is whole decibels')
            db = int(db)
        family = self.scale_family(measured)
        for name in family:
            self[name].scale_db = db
        return family

    def fit_coherence(self, source: Any,
                      records: Sequence[int] | None = None) -> str | None:
        """The coherence the fitting screen weights a fit of `source` by,
        or None when no coherence covers any of the fit's response DOFs.

        It is the coherence covering the most of those DOFs, and one in
        the FRF's own object group is chosen in a tie.

        Parameters
        ----------
        source : str or Frf
            The FRF set being fitted.
        records : sequence of int, optional
            The records fitted; every record when omitted.

        Returns
        -------
        str or None
            The coherence's name, for `fit_modes(coherence=...)`.
        """
        from .core.data import _CoherenceBase

        source = self.name_of(source)
        frf = self[source]
        rows = range(frf.num_records) if records is None else records
        wanted = {frf.response_dof[int(i)] for i in rows}
        linked = set(self.object_group_of(source) or ())
        best, best_key = None, (0, False)
        for name in self.ordered_names():
            obj = self[name]
            if not isinstance(obj, _CoherenceBase):
                continue
            covered = len(set(obj.response_dof) & wanted)
            key = (covered, name in linked)
            if covered and key > best_key:
                best, best_key = name, key
        return best

    def fit_modes(self, source: str, *, bounds: tuple[float, float] | None
                  = None, limit: int = 30, name: str | None = None,
                  at: Sequence[tuple[float, float]] | None = None,
                  refine: int = 0, records: Sequence[int] | None = None,
                  coherence: Any = None) -> str:
        """Fit a modal model to an FRF set, as the fitting screen does.

        This runs the screen's loop: confirm the suggested mode, then
        take the next suggestion, `limit` times. The session's
        suggestion logic makes every choice (a peak already confirmed is
        not offered again unless a different shape is found there, as
        for a repeated pair), and the loop adds no check of its own. On
        the screen the person stops the loop; in a script `limit` does,
        and the plate demonstration's cap is a worked example of
        choosing it.

        On the screen the zoom does the job of `bounds`: what is on the
        plot is what gets searched. A script has no plot, so it gives
        `bounds` instead.

        Parameters
        ----------
        source : str
            The FRF set to fit.
        bounds : tuple of float, optional
            (low, high) frequency limits to fit within, in Hz. Defaults
            to the whole band.
        limit : int, default 30
            The most modes to accept.
        name : str, optional
            What to call the shape set. Defaults to the FRF set's name
            followed by 'Modes'.
        at : sequence of tuple, optional
            Explicit (frequency, damping) picks, each optionally
            (frequency, damping, description), confirmed in the order
            given. Each mode is removed from the residual before the
            next is fitted, so the order is part of the fit. The fitting
            screen journals its confirms as this call, which is how an
            interactive session is replayed. `bounds` and `limit` are
            ignored when picks are given.
        refine : int, default 0
            How many times to run the joint residue refinement after the
            confirms (the screen's Refine All).
        records : sequence of int, optional
            Fit these records only; the screen fits the records picked
            in the tree. Every record when omitted.
        coherence : str or Coherence, optional
            Weight each response's least-squares fit by this coherence,
            line by line. The screen weights by
            `fit_coherence(source, records)` when the project has one.
            Unweighted when omitted.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
        """
        # The loop adds no second opinion. It used to: a proximity guard
        # here vetoed any suggestion within 1 Hz of a confirmed mode, the
        # same ridge-trap fix the session has since outgrown, and it
        # silently skipped the second mode of a repeated pair that
        # `suggest` had offered on purpose.
        from .core.modal_fit import ModalFitSession

        source = self.name_of(source)
        if isinstance(coherence, str):
            coherence = self[self.name_of(coherence)]
        session = ModalFitSession(self[source], records, coherence)
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
        """Sample a shape set at the Basis shape set's DOFs, or at a
        channel table's (Project onto Basis DOFs). Returns the new shape
        set's name.

        For each target DOF, the nearest node of the source's geometry
        is found, and the displacement there is dotted with the DOF's
        direction on the target's geometry, its node's coordinate system
        and sign included. A DOF with no node within `tolerance` is
        dropped rather than estimated. The result is linked into the
        target's group. A channel table as the target puts a model's
        modes at a test's sensors and drives before there are test modes
        to compare with: its channels' DOFs, on the geometry the table
        answers to. With no target, or the source or the target without
        a geometry, this raises ValueError.

        Parameters
        ----------
        source : str
            The shape set to project.
        onto : str, optional
            The shape set or channel table to project onto. Defaults to
            the shape set in the project's Basis.
        tolerance : float, default 0.02
            How far the nearest node may be, as a fraction of the Basis
            geometry's extent (the largest distance of a node from the
            nodes' mean position).
        name : str, optional
            What to call the result. Defaults to the source's name
            followed by '@ Basis DOFs', or by '@' and the table's name.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
        """
        from .core.correlate import project_shapes

        source = self.name_of(source)
        onto = None if onto is None else self.name_of(onto)
        if onto is not None and isinstance(self[onto], ChannelTable):
            # Brandon, 2026-10-29: the BARC's FE modes at its test
            # geometry's sensors took a placeholder shape set by script.
            # Only the DOFs are read from the target
            dofs = list(dict.fromkeys(d for d in self[onto].dof_strings() if d))
            if not dofs:
                raise ValueError(f'{onto} has no channel with a node and '
                                 'direction')
            target = ShapeSet([], [], dofs, np.zeros((0, len(dofs))))
            name = name or f'{source} @ {onto}'
        else:
            target = None
        basis = ((onto, target if target is not None else self[onto])
                 if onto is not None else self._basis_shapes())
        if basis is None:
            raise ValueError('name the set to project onto, or declare a '
                             'Basis holding one')
        basis_name, basis_shapes = basis
        home = self.geometry_for(basis_name)
        theirs = self.geometry_for(source)
        if home is None or theirs is None:
            raise ValueError(f'{source} and {basis_name} each need a '
                             'geometry — link one to each side')
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
        """Save matched mode pairs as an object (the comparison screen's +
        button).

        With `pairs`, exactly those pairs are kept. Otherwise each of
        `first`'s modes is paired with the mode in `second` that has the
        highest MAC, when that MAC is at least `threshold`. Across
        geometries the MAC is computed after projection, as the screen
        does it (`comparison_mac`). The result is in no object group,
        since it names a shape set in each; it records both geometries
        itself.

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
            MAC values for those pairs. Computed from `comparison_mac`
            when omitted.
        threshold : float, default 0.7
            The lowest MAC an automatic pairing may have.
        name : str, default 'Matched Modes'
            What to call the result.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
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
        # Unlinked, unlike every other derived object. An object group is a
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
        """The MAC between two shape sets, as the comparison screen shows
        it.

        When the sets are on different geometries, the second set is
        projected onto the first set's DOFs, because DOFs with the same
        name on two geometries need not point in the same direction.

        Parameters
        ----------
        first : str
            One shape set, by name.
        second : str
            The other shape set, by name.

        Returns
        -------
        numpy.ndarray
            The MAC matrix, with `first`'s shapes down the rows and
            `second`'s across the columns.
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
        """Draw the MAC the comparison screen draws: `first` against
        itself, or against `second`.

        Across geometries the second set is projected exactly as
        `comparison_mac` does it. A shape set's own `plot_mac` compares
        by DOF name and has no geometry, so it cannot do this.

        Parameters
        ----------
        first : str or ShapeSet
            One shape set, by name or as the object.
        second : str or ShapeSet, optional
            The other shape set. Omitted, the auto-MAC of `first` is
            drawn.
        **kwargs
            Passed to the drawing: `path=` renders to a file,
            `bars=True` draws 3-D bars (then `screenshot=`), and
            `theme`, `title`, `size` and `show`.

        Returns
        -------
        object
            Whatever the drawing returns: a window or an image.
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
        """Combine compatible objects into one (Merge). The merged object
        replaces its parts.

        Only objects of the same type can be merged, and each kind has
        its own rule: geometries need disjoint node ids, shape sets the
        same DOFs, and data arrays an identical abscissa. The merged
        object is linked into the first part's object group.

        Parameters
        ----------
        *names : str
            The objects to merge, by name.
        name : str, optional
            What to call the merged object. Defaults to the first
            part's name.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
        """
        from .core.merge import merge as merge_objects

        names = tuple(self.name_of(n) for n in names)
        merged = merge_objects([self[n] for n in names])
        group = self.object_group_of(names[0]) or []
        # the merged object rests on every file its parts were read from
        held = DataIndex([entry for entry in self.data_index
                          if set(entry.objects) & set(names)])
        self.remove(*names)
        added = self.add(name or names[0], merged)
        self.data_index.adopt(held, dict.fromkeys(names, added))
        rest = [member for member in group if member in self]
        if rest:
            self.link(added, *rest)
        return added

    def export(self, name: str, path: str | os.PathLike,
               unit_system: Any = None, **kwargs: Any) -> str:
        """Write an object to another file format, chosen by the file's
        suffix (Export).

        Any registered writer can be used: `.unv`, `.exo`, `.npz`,
        `.bdf`, `.afu`/`.ati`/`.ash`, `.xlsx`, `.3mf`, `.stl`,
        `.vdreport` and the rest of `io.exporters()`.

        Parameters
        ----------
        name : str
            The object to write.
        path : str or os.PathLike
            Where to write it. The format follows the extension.
        unit_system : UnitSystem, optional
            Units to write in. Omitted, the stored (SI) values are
            written as they are.
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
                        name: str = 'Report', marking: str | None = None,
                        marking_color: str | None = None) -> str:
        """Build a report from a starter template, bound by type to this
        project's objects (Generate Report).

        An unknown template raises ValueError.

        Parameters
        ----------
        template : str, default 'modal'
            Which starter to build: 'modal', 'random', 'shock',
            'transient', 'sine', 'sysid', 'mimo' (a multiple-input
            random run, judged on its control matrix) or 'empty'. (In
            the app, a
            Random and Sine project makes 'random' and 'sine', one
            report each.) It can also be a saved template, by the name
            it was saved under in the templates folder, or the path of a
            `.vdreport` file, a report exported on its own
            (`io.report_template`).
        name : str, default 'Report'
            What to call the report object.
        marking : str, optional
            The banner across the top and bottom of every page.
            'UNCLASSIFIED' when omitted.
        marking_color : str, optional
            'ink' or 'red'.

        Returns
        -------
        str
            The name the result was added under, unique within the
            project. A name already taken gets a numbered suffix.
        """
        # `marking` lets a batch stamp its own banner (Brandon, 2026-10-02).
        from .core.report import (
            TEMPLATE_BUILDERS,
            Report,
        )

        builders = TEMPLATE_BUILDERS
        if template in builders:
            report = builders[template](self, object_groups=self.object_groups)
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
        if marking is not None:
            report.marking = str(marking)
        if marking_color is not None:
            report.marking_color = marking_color
        return self.add(name, report)

    def report_features(self, name: str, *, enable: Sequence[str] = (),
                        disable: Sequence[str] = (),
                        preset: str | None = None) -> FeatureSelection:
        """Turn a report's optional features on or off (the report
        editor's Report Features panel).

        A report feature is an addition to the generated report, off
        until it is enabled; with none enabled the report is exactly
        what Visual Dynamics writes otherwise. The features, their
        groups and what each one needs are listed in
        `visualdynamics.report.features.FEATURES`, and the presets in
        `PRESETS`. The choice is saved with the report and in a
        `.vdreport` template.

        Parameters
        ----------
        name : str
            The report.
        enable : sequence of str, optional
            Feature keys to turn on. A feature that needs another can
            be enabled in the same call as the one it needs.
        disable : sequence of str, optional
            Feature keys to turn off. A feature that needs one turned
            off goes with it, and the notes say so.
        preset : str, optional
            'default' (no features), 'test_report' or 'flash_report':
            the report's features are replaced by the preset's before
            `disable` and `enable` apply. A feature the preset names
            that is not available yet is skipped with a note.

        Returns
        -------
        FeatureSelection
            `enabled`, the report's features afterwards, and `notes`,
            what was skipped or turned off along the way.

        Raises
        ------
        ValueError
            For an unknown feature or preset, a feature not yet
            available, or a feature whose requirement is not met.
            Nothing is changed.
        TypeError
            When `name` is not a report, or a key is not a string.
        """
        from .report.features import configure

        return configure(self._report(name), enable=enable,
                         disable=disable, preset=preset)

    def report_section(self, name: str, section: str,
                       text: str | None = None, *, none: bool = False,
                       by: str | None = None) -> None:
        """Write one skeleton section of a report (the report editor's
        pane on a section's block).

        With the report feature `skeleton_sections` on, the report is
        laid out on the sections in
        `visualdynamics.report.skeleton.SECTIONS`, and each section only
        the author can write is shown as owed until it says something.
        This sets what it says.

        Parameters
        ----------
        name : str
            The report.
        section : str
            The section's key: 'identification', 'summary',
            'requirement', 'article', 'setup', 'instrumentation',
            'chronology', 'results', 'deviations', 'discussion',
            'conclusion' or 'record'.
        text : str, optional
            The section's words, in the report's Markdown, replacing
            what it said.
        none : bool, default False
            Write the section's own wording for nothing to report, in
            one step ('deviations' and 'discussion' only): "No
            deviations. No anomalies. No nonconformances." With the
            report feature `findings` on and findings entered, the
            deviations' None says "none" only of each kind with
            nothing entered (`visualdynamics.report.findings.none`).
        by : str, optional
            Who made the interpretation ('discussion' only); an empty
            string clears it.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            For an unknown section, `text` and `none` together, `none`
            for a section that always has something to say, or `by`
            outside the discussion. Nothing is changed.
        TypeError
            When `name` is not a report, or a value is not a string.
        """
        from .report import findings
        from .report.features import enabled
        from .report.skeleton import write

        report = self._report(name)
        if (none and text is None and by is None
                and section == findings.SECTION
                and findings.FEATURE in enabled(report)):
            # the register's None: "none" only of the kinds with nothing
            # entered, so it cannot contradict a finding
            findings.none(report)
            return
        write(report, section, text, none=none, by=by)

    def report_check_block(self, name: str, check: str, *,
                           source: str | None = None,
                           measured: str | None = None,
                           specification: str | None = None,
                           shapes: str | None = None,
                           **settings: Any) -> int:
        """Give a report a check of its own, with what it reads and its
        settings (report feature `check_blocks`).

        With the feature on, the report carries a block for every check
        it can make from what it shows. A check that needs a fact no
        object holds is made from a block of its own: the classical
        pulse ('pulse', with its nominal shape, peak and duration), the
        sweep rate against the modes the test plan names ('sweep_rate',
        `modes`), two signatures before and after the test
        ('signature', `source` before and `measured` after) and a
        replicated record's segments ('segments', `segments`). A check
        made from what the report shows, given a block of its own, is
        made from the block, so its settings hold. The checks and what
        each reads are `visualdynamics.report.check_blocks.CHECKS`.

        Parameters
        ----------
        name : str
            The report.
        check : str
            'time_statistics', 'rms', 'chance', 'shock_validity',
            'srs_points', 'pulse', 'distortion', 'frequency',
            'sweep_rate', 'signature' or 'segments'.
        source, measured, specification, shapes : str, optional
            What the check reads, by object name or symbolic binding
            ('@basis:TimeHistory'); each field the check takes defaults
            to its own binding.
        **settings
            The keyword arguments of the check's reading in `core`, such
            as `shape`, `peak` (SI) and `duration` for the pulse, or
            `modes` as [[frequency, Q], ...] for the sweep rate.

        Returns
        -------
        int
            The index of the check's block in the report.

        Raises
        ------
        ValueError
            For an unknown check, a field or setting the check does not
            take, a required setting left out, or settings that cannot
            be saved. Nothing is changed.
        TypeError
            When `name` is not a report.
        """
        from .report.check_blocks import add

        bindings = {key: value for key, value in (
            ('source', source), ('measured', measured),
            ('specification', specification), ('shapes', shapes))
            if value is not None}
        return add(self._report(name), check, bindings, settings)

    def report_check(self, name: str) -> list[str]:
        """What a report still owes before it is issued, one line per
        section, in order (what Export lists and asks about).

        A line names the section and why: not yet written; nothing in
        it (Instrumentation or Results with nothing generated or
        written); or a discussion that does not say by whom. With the
        report feature `findings` on, section 9 is read off the
        register: a kind neither entered nor set to None, a kind both
        entered and said to be none, and each finding's blank parts,
        such as '9. Deviations, Anomalies and Nonconformances: D-1:
        cause not yet written'. An empty list means every section of
        the skeleton says something. With the report feature
        `flash_report` on, the lines are what the flash report's one
        page still owes (`visualdynamics.report.flash.check`), such as
        'Condition on release: not yet written', since the page draws
        none of the skeleton's sections.

        Parameters
        ----------
        name : str
            The report.

        Returns
        -------
        list of str
            One line per owed section, such as '4. Test Article: not
            yet written', in the skeleton's order.

        Raises
        ------
        ValueError
            When the report does not have the report feature
            `skeleton_sections` enabled: without the skeleton there is
            nothing to check against, and an empty list would read as a
            complete report.
        TypeError
            When `name` is not a report.
        """
        from .report import findings, flash
        from .report.features import enabled
        from .report.skeleton import check

        report = self._report(name)
        if flash.FEATURE in enabled(report):
            return flash.check(report, self, self.object_groups)
        return [str(owed) for owed in findings.check(
            report, check(report, self, self.object_groups))]

    def report_flash(self, name: str, *, condition: str | None = None,
                     next_steps: str | None = None,
                     report_date: str | None = None,
                     opinions: Sequence[Any] | None = None,
                     opinions_none: bool | None = None) -> None:
        """Type the parts of a flash report only the author can write.

        With the report feature `flash_report` on, the report is one
        preliminary page drawn from the identity, the result table and
        the findings register (`visualdynamics.report.flash`). What the
        project does not hold is owed on the page until it is typed
        here.

        Parameters
        ----------
        name : str
            The report.
        condition : str, optional
            The article's condition on release: the post-test functional
            and inspection results as far as they go, and who received
            it ('' clears it).
        next_steps : str, optional
            What is still to be done before the result is final.
        report_date : str, optional
            The date the test report will be issued, as the lab writes
            it.
        opinions : sequence, optional
            The opinions given at the console, replacing those entered,
            each (by, text) or a mapping of 'by' and 'text'. Each is
            drawn marked as an opinion, with who gave it.
        opinions_none : bool, optional
            True to say no opinion was given; False takes it back.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            When the feature is off, for an opinion without its words or
            without who gave it, and for opinions entered while None is
            said of them. Nothing is changed.
        TypeError
            When `name` is not a report, or a value is of the wrong
            type.
        """
        from .report.flash import enter

        enter(self._report(name), condition=condition,
              next_steps=next_steps, report_date=report_date,
              opinions=opinions, opinions_none=opinions_none)

    def report_finding(self, name: str, number: str | None = None, *,
                       kind: str | None = None, run: str | None = None,
                       channel: str | None = None,
                       band: Sequence[float] | None = None,
                       fact: str | None = None, cause: str | None = None,
                       effect: str | None = None,
                       approval: str | None = None,
                       status: str | None = None, accept: bool = False,
                       discard: bool = False,
                       remove: bool = False) -> str:
        """Enter, change or remove a finding in a report's findings
        register (report feature `findings`; the report editor's pane
        on section 9).

        A finding is a deviation, an anomaly or a nonconformance,
        numbered by kind (D-1, A-1, NC-1) and written in four parts:
        the fact, the cause, the effect, and the approval (of a
        deviation) or disposition (of an anomaly or nonconformance).
        Section 9 of the report draws each one, and the summary lists
        them by number. A blank part is drawn as owed, and
        `report_check` lists it.

        Parameters
        ----------
        name : str
            The report.
        number : str, optional
            The finding to change. Without it a finding of `kind` is
            entered under the next number of its kind.
        kind : str, optional
            'deviation', 'anomaly' or 'nonconformance', for a new
            finding only: a finding's kind is its number's.
        run, channel : str, optional
            The run and the channel the finding concerns.
        band : (float, float), optional
            The band it concerns, low and high in Hz; an empty
            sequence clears it.
        fact, cause, effect, approval : str, optional
            The four parts, in the author's words.
        status : str, optional
            'open', 'closed' or '' (not stated).
        accept : bool, default False
            Make the fact draft offered on the finding its fact
            (`draft_finding_fact`); with `fact`, as edited.
        discard : bool, default False
            Drop the fact draft offered on the finding.
        remove : bool, default False
            Take the finding out of the register. Its number is not
            given again.

        Returns
        -------
        str
            The finding's number.

        Raises
        ------
        ValueError
            When the report does not have the feature enabled; for a
            kind on an existing finding or none for a new one, an
            unknown number, a band that is not two ordered frequencies,
            a status other than open or closed, `accept` with no draft
            waiting, `discard` with none, more than one of `accept`,
            `discard` and `remove`, or `remove` with fields. Nothing is
            changed.
        TypeError
            When `name` is not a report, or a text field is not a
            string.
        """
        from .report import findings

        report = self._report(name)
        fields = {'run': run, 'channel': channel, 'band': band,
                  'fact': fact, 'cause': cause, 'effect': effect,
                  'approval': approval, 'status': status}
        given = {key: value for key, value in fields.items()
                 if value is not None}
        if sum((accept, discard, remove)) > 1:
            raise ValueError('accept, discard and remove are one at a time')
        if number is None:
            if accept or discard or remove:
                raise ValueError('name the finding to accept, discard or '
                                 'remove')
            if kind is None:
                raise ValueError('a new finding needs its kind: deviation, '
                                 'anomaly or nonconformance')
            return findings.add(report, kind, **given)
        if kind is not None:
            raise ValueError(f"{number}: a finding's kind is its number's; "
                             'enter it again under the other kind')
        if remove:
            if given:
                raise ValueError(f'{number}: remove takes no fields')
            findings.remove(report, number)
            return number
        if accept:
            # the accepted fact, as offered or as edited, then the rest
            findings.accept(report, number, given.pop('fact', None))
        elif discard:
            findings.discard(report, number)
        if given:
            findings.edit(report, number, **given)
        elif not accept and not discard:
            findings.find(report, number)
        return number

    def draft_finding_fact(self, name: str, number: str, measured: str,
                           specification: str, channel: str | None = None,
                           *, band: Sequence[float] | None = None,
                           limits: str = 'abort',
                           progress: Any = None
                           ) -> str:
        """Offer a finding a draft of its fact, read off a comparison the
        project holds.

        The draft names the channel, each stretch where `measured`
        went outside `specification`'s `limits` (its band, its side,
        how far beyond at worst in dB, how many lines), the record and
        the time history it was computed from, and what was not
        judged. It is read on the same cells, at the same scale, as the
        comparison figure (`visualdynamics.core.findings.excursions`).
        The draft fills nothing: the author accepts it, edits it or
        discards it (`report_finding(..., accept=True)`). Only the
        fact is ever offered; a channel inside its limits gives no
        draft.

        Parameters
        ----------
        name : str
            The report.
        number : str
            The finding.
        measured, specification : str
            The measured spectrum and the specification it is compared
            with.
        channel : str, optional
            The channel, such as '2Y+'; the finding's own when omitted.
        band : (float, float), optional
            The band to read, in Hz; the finding's own when omitted,
            and the whole comparison when it has none.
        limits : str, default 'abort'
            'abort' or 'warning'.
        progress : callable, optional
            Accepted so the app can run the reading on its worker; not
            called.

        Returns
        -------
        str
            The draft offered.

        Raises
        ------
        ValueError
            When the report does not have the feature enabled; for an
            unknown finding, a name that is not a spectrum or a
            specification of the project, a channel they do not share,
            no channel, a pair of limits the specification does not
            carry, a comparison refused (different quantities, units on
            one side only), nothing judged in the band, an all-zero
            channel, and a channel inside its limits wherever it was
            judged. Nothing is changed.
        TypeError
            When `name` is not a report.
        """
        from .report import findings

        return findings.draft_from_comparison(
            self._report(name), number, self, measured, specification,
            channel, band=band, pair=limits, provenance=self.provenance).text

    def draft_finding_from_requirement(self, name: str, number: str,
                                       measured: str, specification: str,
                                       criterion: int | str) -> str:
        """Offer a finding a draft of its fact, read off a row of a
        requirement's result table.

        A row offers a fact where the run departed from the plan: a
        limit channel's notch the plan does not name (its band, its
        line count and how far the control fell under the reference),
        and a cross-axis channel over its limit (its worst percentage,
        the lines over and their reach). The draft names the channel
        and the record and the time history it was computed from, and
        reaches the finding as `draft_finding_fact`'s does, through the
        register's one entry point for drafts
        (`visualdynamics.report.findings.offer`): it fills nothing
        until the author accepts it, edits it or discards it
        (`report_finding(..., accept=True)`).

        Parameters
        ----------
        name : str
            The report, with the features `test_requirement` and
            `findings` on.
        number : str
            The finding.
        measured, specification : str
            The run and the specification the requirement is bound to
            (`report_requirement`).
        criterion : int or str
            The row: its number in the result table from 1, or its name,
            such as 'Limit channel 3Z+'.

        Returns
        -------
        str
            The draft offered.

        Raises
        ------
        ValueError
            When the report does not have both features enabled; for an
            unknown finding, no requirement bound to the pair, one that
            cannot be read, a row number out of range, a name no row has,
            a name two offering rows share, and a row that offers no
            finding. Nothing is changed.
        TypeError
            When `name` is not a report.
        """
        from .report import requirement

        return requirement.draft_finding(
            self._report(name), number, self, measured, specification,
            criterion, self.object_groups, provenance=self.provenance).text

    def report_test_date(self, name: str, start: str | None,
                         end: str | None = None) -> None:
        """Set the dates of the test a report covers, which its sensors'
        calibration is judged against (the dates in the report editor's
        pane on the channel table).

        With the report feature `calibration_in_date` on, the report's
        channel table carries each sensor's calibration status on these
        dates: in date, expired before the test, lapsing during it, or
        no date given (`visualdynamics.core.calibration`). The dates are
        kept with the report's channel table blocks, and a `.vdreport`
        template leaves them out.

        Parameters
        ----------
        name : str
            The report.
        start : str or None
            The first test day, YYYY-MM-DD. None or '' clears the dates.
        end : str, optional
            The last test day, YYYY-MM-DD, for a test over several
            days; the first day when omitted.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            When the report does not have `calibration_in_date` enabled
            or has no channel table block, or for a date not written
            YYYY-MM-DD (a slashed date reads two ways) or a last day
            before the first. Nothing is changed.
        TypeError
            When `name` is not a report, or a date is not a string.
        """
        from .report.calibration import set_dates

        set_dates(self._report(name), self, self.object_groups, start, end)

    def report_calibration(self, name: str) -> CalibrationCheck:
        """Each sensor of a report's channel table against its test
        dates (what the report's Calibration column shows).

        Parameters
        ----------
        name : str
            The report.

        Returns
        -------
        CalibrationCheck
            `span`, the test dates; `channels`, one entry per channel
            with its `status` ('in date', 'expired', 'lapses during the
            test' or 'no date given') and `days`; `summary()`, the
            check in words.

        Raises
        ------
        ValueError
            When the report does not have `calibration_in_date` enabled,
            has no channel table block or no test date (there is then
            nothing to judge against, and no answer would read as in
            date), or when a date or an Expiration cell cannot be read
            as a date; the message names the cell.
        TypeError
            When `name` is not a report.
        """
        from .report.calibration import checked

        return checked(self._report(name), self, self.object_groups)[0]

    def report_calibration_text(self, name: str) -> str:
        """The equipment list with each sensor's calibration status on
        the test date, as Markdown offered for the report's Setup and
        Method section (the section pane's Insert Calibration Status).

        The section is the author's, so nothing is written into it
        here; put the text in with `report_section`:

            text = project.report_calibration_text('Report')
            project.report_section('Report', 'setup', text)

        Parameters
        ----------
        name : str
            The report.

        Returns
        -------
        str
            A sentence, a pipe table of the sensors (channel, DOF,
            quantity, make, model, serial number, expiration and
            calibration status) and the lines of the check.

        Raises
        ------
        ValueError
            As `report_calibration` refuses.
        TypeError
            When `name` is not a report.
        """
        from .report.calibration import draft

        return draft(self._report(name), self, self.object_groups)

    def predict(self, name: str, channel: str, quantity: str,
                peaks: Sequence[Any], *, unit: str | None = None,
                level_db: float | None = None,
                curve: tuple[Sequence[float], Sequence[float]] | None = None,
                reference: str | None = None, source: str = '',
                customer: bool = False,
                frequency_uncertainty: float | None = None,
                level_uncertainty: float | None = None) -> str:
        """Enter one response channel's predicted response, the
        analyst's numbers, into the predictions object `name`, made
        when the project has nothing of that name. A prediction already
        held for the channel and quantity is replaced.

        A report reads a measurement against these with the report
        feature `prediction` (`visualdynamics.report.prediction`), and
        a script with `core.prediction.compare`.

        Parameters
        ----------
        name : str
            The predictions object.
        channel : str
            The response DOF, as the records name it ('201Z+').
        quantity : str
            'asd', 'amplitude' or 'transmissibility'.
        peaks : sequence
            The predicted peaks, each (frequency [Hz], level, Q, band):
            the level in `unit` (squared per hertz for an ASD), Q
            optional, and the band (low, high) in Hz to look for the
            measured peak in, optional when the model states its
            frequency uncertainty. For a `curve`, each mode's frequency
            alone.
        unit : str, optional
            The unit of the quantity: 'g' for an ASD in g²/Hz or an
            amplitude in g. A transmissibility has none.
        level_db : float, optional
            The input level the prediction is stated at, dB relative to
            full level (0 at full level). A transmissibility has none.
        curve : (frequencies, values), optional
            The predicted curve, in `unit`; the peaks' levels and Q are
            read off it. Given as lists, the journal's line replays it.
        reference : str, optional
            The input DOF of a transmissibility.
        source : str, optional
            The analysis it comes from, with its document and revision.
        customer : bool, default False
            Whether the customer supplied it.
        frequency_uncertainty : float, optional
            The model's stated frequency uncertainty, plus or minus, in
            percent.
        level_uncertainty : float, optional
            The model's stated level uncertainty, plus or minus, in dB.

        Returns
        -------
        str
            The name of the predictions object.

        Raises
        ------
        ValueError
            For what a prediction cannot hold
            (`core.prediction.Prediction` lists it). Nothing is changed.
        TypeError
            When `name` is an object that is not a predictions object.
        """
        from .core.prediction import Prediction

        held = self.get(name)
        if held is not None and not isinstance(held, Predictions):
            raise TypeError(f'{name!r} is a {type(held).__name__}, not a '
                            'predictions object')
        prediction = Prediction(
            channel, quantity, peaks, unit=unit, level_db=level_db,
            curve=curve, reference=reference, source=source,
            customer=customer, frequency_uncertainty=frequency_uncertainty,
            level_uncertainty=level_uncertainty)
        if held is None:
            return self.add(name, Predictions([prediction]))
        held.put(prediction)
        return name

    def report_prediction(self, name: str, measured: Any, predictions: Any,
                          *, level: float | None = None,
                          caption: str = '') -> int:
        """Add a block reading `measured` against `predictions` at the
        end of the report `name` (the report editor's Insert, Responses
        Against Prediction), and return its index.

        While the report feature `prediction` is on, the block draws a
        figure per response channel, the measurement over its
        prediction scaled to the level run, and a table of each
        predicted peak against the measured one.

        Parameters
        ----------
        name : str
            The report.
        measured : str or object
            The response: a PSD, an amplitude spectrum or sine level, or
            a transmissibility (an FRF).
        predictions : str or object
            The predictions object.
        level : float, optional
            The level `measured` was run at, dB relative to full level
            (-12 for a run at minus 12 dB). Required for an ASD or an
            amplitude, unless the PSD holds its scaling; refused for a
            transmissibility.
        caption : str, optional
            The table's caption.

        Returns
        -------
        int
            The block's index in the report.

        Raises
        ------
        ValueError
            When the comparison cannot be read; `core.prediction.compare`
            says why. Nothing is changed.
        TypeError
            When `name` is not a report or `predictions` not a
            predictions object.
        """
        from .core.prediction import compare

        report = self._report(name)
        measured_name = self.name_of(measured)
        predictions_name = self.name_of(predictions)
        held = self[predictions_name]
        if not isinstance(held, Predictions):
            raise TypeError(f'{predictions_name!r} is a '
                            f'{type(held).__name__}, not a predictions '
                            'object')
        compare(held, self[measured_name], level)
        report.add({'kind': 'prediction', 'source': predictions_name,
                    'measured': measured_name,
                    'level': None if level is None else float(level),
                    'caption': str(caption)})
        return report.num_blocks - 1

    def report_requirement(self, name: str, specification: Any,
                           measured: Any, document: str,
                           composite_bands: Sequence[Any], *,
                           revision: str = '', clause: str = '',
                           strategy: str = 'single',
                           channels: Sequence[str] = (),
                           weights: Sequence[float] | None = None,
                           individual_bands: Sequence[Any] = (),
                           grms_percent: float | None = None,
                           duration: float | None = None,
                           duration_tolerance: Sequence[float] | None = None,
                           min_dof: float | None = None,
                           max_line_spacing: float | None = None,
                           cross_axis: Sequence[Any] = (),
                           cross_axis_channels: Mapping[str, str] | None = None,
                           limit_channels: Sequence[Any] = (),
                           rule: str = 'any_line', combine: int = 1,
                           sources: Mapping[str, Any] | None = None,
                           uncertainty: Mapping[str, Any] | None = None,
                           duration_run: float | None = None,
                           caption: str = '') -> int:
        """Bind a test requirement to a run in the report `name`, and
        return its block's index.

        While the report feature `test_requirement` is on, the block
        draws the requirement in words and a result table with one row
        per criterion, worded met, not met or not judged, and the
        report's pass/fail box for the same specification and run reads
        from it (`visualdynamics.report.requirement`). The requirement's
        fields are `core.requirement.TestRequirement`'s; a band is
        (low, high, below_db, above_db), low or high None for the
        specification's own end.

        Parameters
        ----------
        name : str
            The report.
        specification : str or object
            The specification the run was controlled to; a symbolic
            binding ('@basis:Specification') is kept as given, so a
            template carries it.
        measured : str or object
            The run's PSD, likewise.
        document : str
            The governing document.
        composite_bands : sequence
            The control band: the one channel's, or the composite's
            under a multi-point strategy; one band per frequency range.
        revision, clause : str, optional
            The document's revision, and the tolerance clause as the
            plan copied it.
        strategy : str, default 'single'
            'single', or a multi-point composite: 'average', 'weighted',
            'maximum' or 'minimum'.
        channels : sequence of str, optional
            The control channels by response DOF; by default the
            channel table's Control flags, else the records that pair
            with the specification.
        weights : sequence of float, optional
            One per channel, under 'weighted' only.
        individual_bands : sequence, optional
            Each control channel's band, under a multi-point strategy.
        grms_percent : float, optional
            The overall rms tolerance, plus or minus, percent.
        duration : float, optional
            The nominal duration at level, seconds.
        duration_tolerance : (float, float), optional
            Its tolerance below and above, percent.
        min_dof : float, optional
            The fewest degrees of freedom the control may have.
        max_line_spacing : float, optional
            The widest line spacing the control may have, Hz.
        cross_axis : sequence, optional
            The cross-axis limits, each (low, high, percent).
        cross_axis_channels : mapping, optional
            Each cross-axis channel's response DOF, to the in-axis
            control channel it is read against as a percentage.
        limit_channels : sequence, optional
            The response channels held under a limit profile, each
            (channel, profile, notches): the name of the object holding
            its profile ('' for the specification's own record at the
            channel) and the notches the plan names for it, each
            (low, high) Hz.
        rule : str, default 'any_line'
            The decision rule: 'any_line' or 'combined'.
        combine : int, default 1
            How many adjacent lines are combined under 'combined'.
        sources : mapping, optional
            Where each number came from, by field: (origin, owner), the
            origin 'laboratory', 'customer' or 'derived'.
        uncertainty : mapping, optional
            Each control channel's measurement uncertainty, by response
            DOF: a list of contributions, each (name, percent, coverage,
            source), an expanded uncertainty in percent on the
            acceleration with its coverage factor (2 when left out),
            drawn against one third of the tolerance.
        duration_run : float, optional
            How long the run was at level, seconds: the controller's
            counter, which the duration row is judged on.
        caption : str, optional
            The result table's caption.

        Returns
        -------
        int
            The block's index in the report.

        Raises
        ------
        ValueError
            For a requirement `TestRequirement` refuses, or a run it
            cannot be read against at all (`core.requirement.evaluate`
            says why). Nothing is changed.
        TypeError
            When `name` is not a report, or the specification or the run
            is not of its kind.
        """
        from .core.report import resolve_binding
        from .core.requirement import TestRequirement, evaluate
        from .report.requirement import add, channel_table

        report = self._report(name)
        bound = []
        for given in (specification, measured):
            if isinstance(given, str) and given.startswith('@'):
                # a symbolic binding is kept as given, so a template
                # carries it; it must resolve here
                held = resolve_binding(given, self, self.object_groups)
                if held not in self:
                    raise ValueError(f'{given!r} names nothing in this '
                                     'project')
                bound.append((given, held))
            else:
                held = self.name_of(given)
                bound.append((held, held))
        (specification_name, spec_held), (measured_name, run_held) = bound
        requirement = TestRequirement(
            document, tuple(composite_bands), revision=revision,
            clause=clause, strategy=strategy, channels=tuple(channels),
            weights=None if weights is None else tuple(weights),
            individual_bands=tuple(individual_bands),
            grms_percent=grms_percent, duration=duration,
            duration_tolerance=(None if duration_tolerance is None
                                else tuple(duration_tolerance)),
            min_dof=min_dof, max_line_spacing=max_line_spacing,
            cross_axis=tuple(cross_axis),
            cross_axis_channels=dict(cross_axis_channels or {}),
            limit_channels=tuple(limit_channels), rule=rule, combine=combine,
            sources=dict(sources or {}), uncertainty=dict(uncertainty or {}))
        evaluate(requirement, self[spec_held], self[run_held],
                 table=channel_table(self), duration_run=duration_run,
                 profiles={c.profile: self[resolve_binding(
                     c.profile, self, self.object_groups)]
                     for c in requirement.limit_channels
                     if resolve_binding(c.profile, self,
                                        self.object_groups) in self})
        return add(report, requirement, source=specification_name,
                   measured=measured_name, duration_run=duration_run,
                   caption=caption)

    def index_file(self, path: str | os.PathLike, holds: str | None = None,
                   *, role: str = 'referenced', objects: Sequence[str] = (),
                   source: str = 'author',
                   progress: Any | None = None) -> IndexEntry:
        """Add a file to the project's data index (the report editor's
        Insert, Attached File).

        The data index lists every file the project rests on, by name,
        what it holds, format, size and SHA-256, and the report feature
        `data_index` prints it in the report's record. A file imported
        into the project is indexed by the import; this adds any other
        file: a document that goes with the report (the settings list,
        the log, the as-run procedure, the flash report), or a data
        file kept in the archive that was not imported. Another source
        of files adds its entries through this verb as well, naming
        itself in `source`, so the project keeps one index.

        Parameters
        ----------
        path : str or os.PathLike
            The file. It is hashed now.
        holds : str, optional
            What the file is or holds, in words ("settings list", "log,
            Day 1 and Day 2"). Taken from `objects` when omitted.
        role : str, default 'referenced'
            'attached' for a document sent with the report, or
            'referenced' for a file named in it and kept in the archive.
            An attached text file of up to 64 KiB is kept whole, so the
            report can print it; a CSV prints as a table.
        objects : sequence of str, optional
            The project objects read from the file, by name.
        source : str, default 'author'
            What added the entry, as the index records it.
        progress : callable, optional
            Called as ``progress(bytes read, bytes in the file)`` while
            the file is hashed.

        Returns
        -------
        IndexEntry
            The file's entry. The same file indexed again with the same
            checksum is the same entry.

        Raises
        ------
        ValueError
            When neither `holds` nor `objects` says what the file holds,
            for an unknown role, or for an object not in the project.
            Nothing is changed.
        FileNotFoundError, IsADirectoryError
            When `path` is not a file.
        """
        from .io import format_of

        names = [self.name_of(name) for name in objects]
        if holds is None:
            holds = holds_of_names(self, names)
        local = os.path.expanduser(str(path))
        return self.data_index.add(local, holds, format=format_of(local),
                                   role=role, source=source, objects=names,
                                   progress=progress)

    def unindex_file(self, name: str) -> IndexEntry:
        """Take a file out of the data index (the report editor's
        Insert, Remove from Index).

        For a document added by mistake, or one that no longer goes
        with the report. A file imported into the project leaves the
        index on its own when the last object read from it is removed.

        Parameters
        ----------
        name : str
            The file's name as the index prints it, or its path when
            two entries share a name.

        Returns
        -------
        IndexEntry
            The entry taken out.

        Raises
        ------
        ValueError
            When no entry, or more than one, has that name. Nothing is
            changed.
        """
        return self.data_index.remove(name)

    def verify_data_index(self, search: Sequence[str | os.PathLike] = (),
                          progress: Any | None = None) -> list[str]:
        """Hash every file in the data index again and record whether
        it matches (what Export does when the report prints the index).

        A file is looked for where it was indexed, then by its name in
        each folder of `search`, then beside the project's own file. It
        matches when its size and SHA-256 are the ones indexed; it has
        changed when they are not; it is not found when it is nowhere.
        The result is kept with each entry and printed in the report.
        The project's own file, when it has one, is hashed too, and the
        report states it.

        Parameters
        ----------
        search : sequence of str, optional
            Folders to look in for files that are not where they were
            indexed, for example the archive copy.
        progress : callable, optional
            Called as ``progress(bytes read, bytes to read)`` across the
            whole index.

        Returns
        -------
        list of str
            One line per file, in the index's order: its name and what
            the check found, such as 'run11.nc4: matches'. Empty when the
            index is empty.
        """
        folders = [os.path.expanduser(str(folder)) for folder in search]
        if self.file:
            folders.append(os.path.dirname(self.file))
        entries = self.data_index.verify(folders, progress)
        self.data_index.project_file = None
        if self.file and os.path.isfile(self.file):
            size, sha = digest_file(self.file)
            self.data_index.project_file = (
                os.path.basename(self.file), size, sha, entries[0].checked
                if entries else _index_now())
        return [f'{entry.name}: {entry.status}' for entry in entries]

    def report_identity(self, name: str, **fields: Any) -> None:
        """Enter a report's identification (the report editor's Identity
        and Issue fields).

        With the report feature `report_identity` on, every printed page
        carries the laboratory, the report number and revision, the
        title and page n of N, and the first page the identification
        block. Every field is the author's: the program never writes a
        number, a date or a name of its own, and a field left blank
        shows as not entered.

        Parameters
        ----------
        name : str
            The report.
        **fields
            Any of 'number', 'revision', 'laboratory', 'address',
            'location', 'customer', 'contact', 'received', 'tested',
            'issued' (the dates as the lab writes them) and
            'distribution', each a str ('' clears it); 'authorized',
            the people authorizing the report, a list of names or one
            per line ("name, role"), printed each with a blank line for
            the signature; and 'full_only', whether the first page says
            the report is not reproduced except in full (True unless
            turned off).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            For an unknown field, the status or issue record (changed
            only by issuing and amending), or the number or revision of
            a report already issued. Nothing is changed.
        TypeError
            When `name` is not a report, or a value is of the wrong
            type.
        """
        from .report.identity import identify

        identify(self._report(name), **fields)

    def export_report_pdf(self, name: str, path: str | os.PathLike,
                          unit_system: Any = None,
                          page_size: str = 'letter') -> str:
        """Write a report as a paginated PDF, printed from the same page
        `export_report` writes.

        A draft prints as often as wanted and says it is a draft on
        every page. An issued report prints again only unchanged: a copy
        of what was issued. With the report feature `report_identity`
        on, each page carries the identity and page n of N. With the
        report feature `data_index` on, every indexed file is hashed
        again first, as for `export_report`, so the PDF states what the
        check found today.

        Parameters
        ----------
        name : str
            The report.
        path : str or os.PathLike
            Where to write the PDF.
        unit_system : UnitSystem, optional
            Units to render in; `units.DEFAULT_SYSTEM` when left out.
        page_size : str, default 'letter'
            'letter' or 'a4'.

        Returns
        -------
        str
            The path written.

        Raises
        ------
        ValueError
            When the report was issued and has changed since (a change
            after issue is an amendment), or for an unknown page size.
        OSError
            When the page cannot be printed.
        """
        self._verify_index_for(name)
        return self._export_pdf(name, path, unit_system, page_size)

    def _export_pdf(self, name: str, path: str | os.PathLike,
                    unit_system: Any = None,
                    page_size: str = 'letter') -> str:
        """`export_report_pdf` without the check of the data index; the
        window calls it after it has checked the index and asked."""
        from .report.pdf import print_pdf

        page = self._export_page(name, unit_system)
        path = os.path.expanduser(str(path))
        print_pdf(page, path, page_size=page_size)
        return path

    def issue_report(self, name: str, path: str | os.PathLike,
                     unit_system: Any = None,
                     page_size: str = 'letter') -> str:
        """Issue a report: print it as a PDF under its number and
        revision, and record the issue.

        The report's status becomes issued (or amended, for an
        amendment), and the issue record gains the number, revision,
        status and date of issue as entered, with the SHA-256 of the
        PDF and of the page it was printed from. From then on the report
        is not written again under that identity unless it is
        unchanged; a change is an amendment (`amend_report`). With the
        report feature `data_index` on, every indexed file is hashed
        again first, as for `export_report`, so the issued PDF states
        what the check found today.

        Parameters
        ----------
        name : str
            The report, with the report feature `report_identity` on.
        path : str or os.PathLike
            Where to write the issued PDF.
        unit_system : UnitSystem, optional
            Units to render in; `units.DEFAULT_SYSTEM` when left out.
        page_size : str, default 'letter'
            'letter' or 'a4'.

        Returns
        -------
        str
            The path written.

        Raises
        ------
        ValueError
            When the feature is off, the report is already issued, a
            field the identification needs is not entered, or (with
            skeleton sections on) a section is still owed; or for an
            unknown page size. Nothing is changed or written.
        OSError
            When the PDF cannot be printed; the report stays a draft.
        """
        from .report.identity import issue_ready
        from .report.pdf import PAGE_SIZES

        if page_size not in PAGE_SIZES:
            raise ValueError(f'unknown page size {page_size!r}; the sizes '
                             f'are: {", ".join(PAGE_SIZES)}')
        # refused before the files are read: a report that cannot be
        # issued is not worth hashing an index for, and a refusal
        # changes nothing, the check's results included
        issue_ready(self._report(name), self, self.object_groups)
        self._verify_index_for(name)
        return self._issue_pdf(name, path, unit_system, page_size)

    def _issue_pdf(self, name: str, path: str | os.PathLike,
                   unit_system: Any = None, page_size: str = 'letter') -> str:
        """`issue_report` without the check of the data index; the window
        calls it after it has checked the index and asked."""
        from .report.pdf import print_pdf

        report = self._report(name)
        path = os.path.expanduser(str(path))
        self._issue(report, unit_system,
                    lambda page: print_pdf(page, path, page_size=page_size),
                    os.path.basename(path))
        return path

    def _issue(self, report: Report, unit_system: Any, write: Any,
               file: str) -> None:
        """Mark `report` issued, render it, hand the page to `write`
        (which returns the PDF's bytes), and record the issue; on any
        failure the report is put back as it was. The one sequence the
        script verb and the window share."""
        from .report import render_html
        from .report.identity import (
            issue_ready,
            mark_issued,
            record_issue,
            unmark,
        )

        status = issue_ready(report, self, self.object_groups)
        before = mark_issued(report, status)
        try:
            page = render_html(report, self, unit_system,
                               object_groups=self.object_groups)
            pdf = write(page)
        except BaseException:
            unmark(report, before)
            raise
        record_issue(report, page, pdf, file)

    def amend_report(self, name: str, changes: Sequence[Mapping[str, str]],
                     *, revision: str, number: str | None = None) -> None:
        """Make an issued report an amendment, to be issued in its turn
        (`issue_report`).

        The amendment carries its own revision (and, if given, its own
        number), refers to the revision it amends by number, revision
        and date of issue, and prints a table of what it changes. It
        ends "End of amendment.". The original's record is kept.

        Parameters
        ----------
        name : str
            The report.
        changes : sequence of mapping
            Each change: 'where' (page or section), 'was' (what it
            said), 'now' (what it says) and, where appropriate,
            'reason'.
        revision : str
            The amendment's revision, entered by the author.
        number : str, optional
            The amendment's own report number; the original's when left
            out.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            When the report has not been issued, a change lacks where,
            was or now, there are no changes, or the number and revision
            were already issued. Nothing is changed.
        TypeError
            When `name` is not a report, or a value is of the wrong
            type.
        """
        from .report.identity import amend

        amend(self._report(name), changes, revision=revision, number=number)

    def _verify_index_for(self, name: str) -> None:
        """With the report feature `data_index` on report `name`, hash
        every indexed file again (`verify_data_index`), so what is
        written states what the check found today, not at the last
        check; nothing is read with the feature off."""
        from .report.features import enabled

        if 'data_index' in enabled(self._report(name)):
            self.verify_data_index()

    def _export_page(self, name: str, unit_system: Any = None) -> str:
        """The report rendered for export, refused when it was issued and
        has changed since (`report.identity.guard`)."""
        from .report import render_html
        from .report.identity import guard

        report = self._report(name)
        page = render_html(report, self, unit_system,
                           object_groups=self.object_groups)
        guard(report, page)
        return page

    def run_identity(self, name: str) -> Any:
        """Which run of the test a time history is, as a report reads it.

        The axis, run, level against full and start the history carries
        (read from its file's name on import, or entered with
        `set_run_identity`), and the level the comparison with its
        specification detects when nothing else gives one. A field read
        two ways is not read, and `refused` says why, naming the
        record.

        Parameters
        ----------
        name : str
            A time history, or an object computed from one (which
            answers with that run's identity).

        Returns
        -------
        RunIdentity
            `axis`, `run`, `level_db`, `start` and `comment`; `sources`,
            where each came from; `refused`, what was not read and why.

        Raises
        ------
        ValueError
            When `name` is not in the project.
        TypeError
            When `name` is not a time history and comes from none.
        """
        from .report.runs import identity

        return identity(self, self.name_of(name), self.object_groups)

    def set_run_identity(self, name: str, *, axis: str | None = None,
                         run: str | int | None = None,
                         level_db: float | str | None = None,
                         start: Any = None,
                         comment: str | None = None) -> Any:
        """Enter which run of the test a time history is.

        An entered field stands over what the file's name or the
        comparison gave. A field left out (None) is unchanged; an empty
        string clears it, back to what the record shows. The identity
        is saved with the history in the project file.

        Parameters
        ----------
        name : str
            The time history: one imported, not one computed from
            another (that is part of the run it came from).
        axis : str, optional
            The test axis, 'X', 'Y', 'Z' or a lab's own name.
        run : str or int, optional
            The run's identifier.
        level_db : float, optional
            The level against full, in decibels: 0 for full level.
        start : str, datetime or date, optional
            When the run started, ISO 8601 text such as
            '2026-10-11T11:11:05'.
        comment : str, optional
            A free note on the run.

        Returns
        -------
        RunIdentity
            What the history carries afterwards.

        Raises
        ------
        ValueError
            For a history computed from another, a level that is not a
            finite number, a start that is not a date, or an axis or
            run on more than one line; the message names the record.
            Nothing is changed.
        TypeError
            For an object that is not a time history, or a value of the
            wrong type.
        """
        from .report.runs import set_identity

        given = {key: value for key, value in (
            ('axis', axis), ('run', run), ('level_db', level_db),
            ('start', start), ('comment', comment)) if value is not None}
        return set_identity(self, self.name_of(name), **given)

    def report_chronology(self, name: str, *, window_s: float = 0.5,
                          drop_db: float = 6.0, min_s: float = 1.0,
                          progress: Callable[[int, int], None] | None = None
                          ) -> str:
        """Draft the chronology of the project's runs and offer it to a
        report's Chronology section (the editor's Draft from the Runs).

        Each run in the order it was run, with its axis, level, start,
        the time its record was at level and the interruptions found in
        it, as a Markdown table, then how they were read. The chronology
        is the author's: the draft is offered beside the section, marked
        as drafted on the editing page, and the section stays owed until
        the author accepts it (`accept_chronology`) or writes the
        section (`report_section`). The outcome of each run and the
        cause of each interruption are left for the author.

        Parameters
        ----------
        name : str
            The report, with the report feature `run_sections` on.
        window_s : float, default 0.5
            The running-RMS window, in seconds.
        drop_db : float, default 6.0
            How far below the run's level an interruption reaches.
        min_s : float, default 1.0
            How long an interruption lasts at least, in seconds.
        progress : callable, optional
            Called with (runs read, runs) as each record is read.

        Returns
        -------
        str
            The draft offered, in the report's Markdown.

        Raises
        ------
        ValueError
            When the report does not have `run_sections` enabled, the
            project holds no time history, the order of the runs cannot
            be read (the records with neither a start nor a run number
            are named), or a setting is not one the reading can use.
        TypeError
            When `name` is not a report.
        """
        from .report.features import enabled
        from .report.runs import FEATURE, chronology, draft, offer

        report = self._report(name)
        if FEATURE not in enabled(report):
            raise ValueError(f'report {report.title!r} does not have Runs and '
                             'chronology enabled')
        text = draft(chronology(self, self.object_groups, window_s, drop_db,
                                min_s, progress))
        offer(report, text)
        return text

    def accept_chronology(self, name: str, text: str | None = None) -> None:
        """Accept the drafted chronology as a report's Chronology section,
        as it was drafted or as the author edited it.

        Parameters
        ----------
        name : str
            The report.
        text : str, optional
            The draft as the author edited it; the draft on offer when
            omitted.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            When no draft is on offer and no `text` is given.
        TypeError
            When `name` is not a report, or `text` is not a string.
        """
        from .report.runs import accept

        accept(self._report(name), text)

    def report_runs(self, name: str) -> list[str]:
        """Repeat a report's Results for every run (the editor's Results
        for Every Run).

        The report's generated Results sections drawn from its first
        run are added again for each run it does not draw on yet, each
        figure bound by name to the object computed from that run the
        same way, its caption naming the run. A section whose figure
        has nothing computed from a run is not added for that run, and
        the line for the run says what to compute. A second call adds
        nothing for a run already drawn on.

        Parameters
        ----------
        name : str
            The report, with the report feature `run_sections` on.

        Returns
        -------
        list of str
            A line per run: the figures added, and what was not added
            and why.

        Raises
        ------
        ValueError
            When the report does not have `run_sections` enabled, or
            draws on no run.
        TypeError
            When `name` is not a report.
        """
        from .report.features import enabled
        from .report.runs import FEATURE, repeat

        report = self._report(name)
        if FEATURE not in enabled(report):
            raise ValueError(f'report {report.title!r} does not have Runs and '
                             'chronology enabled')
        return repeat(report, self, self.object_groups)

    def _report(self, name: str) -> Report:
        """The report called `name`; `TypeError` for anything else."""
        name = self.name_of(name)
        report = self[name]
        if not isinstance(report, Report):
            raise TypeError(f'{name!r} is a {type(report).__name__}, not a '
                            'report')
        return report

    def work_up(self) -> list[str]:
        """Compute every missing object the project's type expects from
        what is loaded, with the report last (Automatic on the tree's
        bar).

        The empty slots are filled in their own order, each from the
        objects already there, the way the workflow guide makes them:
        PSDs from the time data, octave bands from the PSDs and the
        specification, and the coherence from the time data; the FRFs, a
        fitted shape set and the matched modes for a modal test; the
        levels for a sine sweep; the filtered record, its SRS and the
        integrated motion for a shock; the two streams' PSDs on shared
        frames and the H1 FRFs for a system identification; and then
        the type's report. A slot the loaded data cannot fill (a
        geometry, the photographs, a specification) is left gray. A
        computation the data refuses (no drive channel for a coherence)
        is skipped, and the rest still run. Nothing already present is
        made again, so calling this twice adds nothing the second time.

        It is journaled as one line, `project.work_up()`, the way
        `merge` is. The verbs it calls are the ones the guide describes,
        and a script that wants them one at a time can call them.

        Returns
        -------
        list of str
            The names added, in the order made; empty when the
            skeleton was already full.

        Raises
        ------
        ValueError
            When the project has no type, so there are no slots to
            fill.
        """
        # Automatic on the tree bar (Brandon, 2026-09-30).
        from .core.data import Psd, Specification, TransientSpecification
        from .core.report import (
            PROJECT_TEMPLATES,
            TEMPLATE_BUILDERS,
            is_banded,
            project_expectations,
        )
        from .core.shapes import ShapeSet

        project_type = self.project_type
        if not project_type:
            raise ValueError('the project has no type, so there is nothing '
                             'to work up — Set Project Type says what a '
                             'finished project holds')
        added: list[str] = []

        def make(verb, *args, **kwargs):
            names = verb(*args, **kwargs)
            names = names if isinstance(names, list) else [names]
            added.extend(names)
            return names

        def on(side, cls):
            sides = self.sides()
            pool = sides.get(side, []) if 'Basis' in sides else self.names
            return [name for name in pool if isinstance(self[name], cls)]

        def source_of(name):
            return self.provenance.get(name, {}).get('source')

        def verb_of(name):
            return self.provenance.get(name, {}).get('verb')

        def derived(name, verb):
            return next((other for other in self.names
                         if verb_of(other) == verb
                         and source_of(other) == name), None)

        def recorded(side):
            # the imported record, ahead of anything derived from one
            histories = on(side, TimeHistory)
            histories = [name for name in histories
                         if not isinstance(self[name], TransientSpecification)]
            return next((name for name in histories
                         if verb_of(name) is None), None) or (
                histories[0] if histories else None)

        def filtered(side):
            record = recorded(side)
            if record is None:
                return None
            done = derived(record, 'filter_data')
            return done or make(self.filter_data, record)[0]

        for slot in project_expectations(project_type):
            if slot not in self.missing():
                continue
            _label, _cls, icon, ordinal, _optional, side = slot
            side = side or 'Basis'
            try:
                if icon == 'Psd':
                    if project_type == 'System ID':
                        noise, driven = _quiet_and_driven(self)
                        # the excitation's frames are the detected ones,
                        # as `compute_frfs` adopts them, and the ambient
                        # borrows them on purpose: the ratio the report
                        # reads is only defined on lines both hold
                        if self[driven].averaging is None:
                            self[driven].averaging = \
                                self[driven].suggest_averaging()
                        if derived(driven, 'compute_psds') is None:
                            make(self.compute_psds, driven)
                        if noise and derived(noise, 'compute_psds') is None:
                            self[noise].averaging = self[driven].averaging
                            make(self.compute_psds, noise)
                        continue
                    record = recorded(side)
                    if record is not None:
                        make(self.compute_psds, record)
                elif icon == 'Specification':
                    # a transient's target is a recording too, and its
                    # spectrum is the requirement the run is read against
                    target = next((name for name in on(side, TransientSpecification)
                                   if derived(name, 'compute_psds') is None),
                                  None)
                    if target is not None:
                        make(self.compute_psds, target)
                elif icon == 'OctavePsd':
                    plain = [name for name in on(side, Psd)
                             if not isinstance(self[name], Specification)
                             and not is_banded(self[name])]
                    if plain:
                        make(self.compute_octave, plain[0])
                elif icon == 'OctaveSpecification':
                    plain = [name for name in on(side, Specification)
                             if not is_banded(self[name])]
                    if plain:
                        make(self.compute_octave, plain[0])
                elif icon == 'MultipleCoherence':
                    record = (_quiet_and_driven(self)[1]
                              if project_type == 'System ID'
                              else recorded(side))
                    if record is not None:
                        make(self.compute_multiple_coherence, record)
                elif icon == 'Frf':
                    record = (_quiet_and_driven(self)[1]
                              if project_type == 'System ID'
                              else recorded(side))
                    if record is not None:
                        # H1 for a system identification: the excitation
                        # is known and what noise there is sits on the
                        # response
                        make(self.compute_frfs, record,
                             *(['H1'] if project_type == 'System ID' else []))
                elif icon == 'ShapeSet' and side == 'Basis':
                    frfs = on(side, Frf)
                    if frfs:
                        make(self.fit_modes, frfs[0])
                elif icon == 'MatchedModes':
                    from .core.report import OTHER_SIDE
                    ours, theirs = on('Basis', ShapeSet), on(OTHER_SIDE, ShapeSet)
                    if ours and theirs and ours[0] != theirs[0]:
                        make(self.match_modes, ours[0], theirs[0])
                elif icon == 'SineLevelSet':
                    record = recorded(side)
                    if record is not None:
                        make(self.extract_sine, record)
                elif icon == 'Srs':
                    # the recommended order: the SRS is read from the
                    # filtered record, so the filtering comes first even
                    # though its slot is listed after
                    record = filtered(side) if project_type == 'Shock' \
                        else recorded(side)
                    if record is not None:
                        make(self.compute_srs, record)
                elif icon == 'TimeHistory' and ordinal > 1:
                    # the shock's motion chain: filtered, then velocity,
                    # then displacement, each from the one before
                    chain = filtered(side)
                    for _step in range(ordinal - 2):
                        if chain is None:
                            break
                        chain = derived(chain, 'integrate') or \
                            make(self.integrate, chain)[0]
                elif icon == 'Report':
                    # the type's reports in order, each made once: chosen
                    # by which are already here (by title), not by the
                    # slot's number, so a random-and-sine project holding
                    # its sine report gets the random one, not a second
                    # sine (2026-10-05, two reports for that type)
                    made = {self[name].title for name in self.names
                            if isinstance(self[name], Report)}
                    wanted = next(
                        (template for template
                         in PROJECT_TEMPLATES[project_type]
                         if TEMPLATE_BUILDERS[template]({}, object_groups=[]).title
                         not in made), None)
                    if wanted is not None:
                        make(self.generate_report, wanted)
            except ValueError:
                # what the data refuses (a coherence with no drive
                # channel) stays gray; the slots after it still fill
                continue
        return added

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
            Units to render in. Defaults to `units.DEFAULT_SYSTEM`,
            'in-slinch-lbf-s (g)'.

        Returns
        -------
        str
            The path written.

        Raises
        ------
        ValueError
            With the report feature `report_identity` on, when the
            report was issued and has changed since: an issued report
            is not written again under its identity, and a change after
            issue is an amendment (`amend_report`).
        """
        name = self.name_of(name)
        self._verify_index_for(name)
        html = self._export_page(name, unit_system)
        path = os.path.expanduser(str(path))
        with open(path, 'w', encoding='utf-8') as out:
            out.write(html)
        return path

    def table(self, name: Any, *, values: bool = False, dofs: Any = None,
              form: str = 'shapes', environment: str | None = None
              ) -> tuple[list[str], list[list[str]]]:
        """(headers, rows) for an object that can be shown as a table.

        The rows are a channel table's instrumentation, a shape set's
        identified parameters, or a matched-modes object's pairs beside
        both sets' parameters. They are the same rows the report prints,
        so a script and a report always agree. With `values`, a shape
        set gives its shape values instead, as the app's Shape Values
        table shows them: at every DOF of the set, or at `dofs`, and as
        the shapes or one of the transformations through them
        (`core.transform.matrix_at`). With `environment`, a channel
        table's Control column is that control loop's, and with `form`
        'response' or 'reference' as well the rows are that loop's
        transformation instead, as the app shows it with the
        environment and the view picked on the table's bar. An object
        with no table raises ValueError.

        Parameters
        ----------
        name : str or object
            The object to tabulate.
        values : bool, default False
            For a shape set, its values rather than its modes.
        dofs : str or sequence of str, optional
            With `values`, where: a list of DOFs, or a channel choice
            ('all', 'control', 'reference', 'response' or 'monitor') of
            the channel table in the set's object group. Every DOF of
            the set when left out.
        form : str, default 'shapes'
            With `values`, what: 'shapes' (`Φ`, a row per DOF),
            'response' (`Φ⁺`, a row per mode) or 'reference' (`Φᵀ`).
            With `environment`, 'response' or 'reference' for that
            loop's transformation (`ChannelTable.transformation`).
        environment : str, optional
            For a channel table, one of its environments
            (`ChannelTable.environments`).

        Returns
        -------
        tuple of (list of str, list of list of str)
            The column headings and the rows, both as text.
        """
        from .core.channel_table import TRANSFORMATION_FORMS
        from .core.shapes import ShapeSet
        from .core.tables import shape_values, table_of, transformation_table

        name = self.name_of(name)
        if values:
            if not isinstance(self[name], ShapeSet):
                raise ValueError(f'{name} is a {type(self[name]).__name__}; '
                                 'only a shape set has shape values')
            return shape_values(self[name], self._shape_dofs(name, dofs), form)
        if environment is not None:
            if not isinstance(self[name], ChannelTable):
                raise ValueError(f'{name} is a {type(self[name]).__name__}; '
                                 'only a channel table has environments')
            if form in TRANSFORMATION_FORMS:
                return transformation_table(self[name], environment, form)
        built = table_of(self[name], objects=self, environment=environment)
        if built is None:
            raise ValueError(f'{name} is a {type(self[name]).__name__}, '
                             'which does not read as a table')
        return built

    def plot_transformation(self, name: Any, *, dofs: Any = None,
                            form: str = 'reference', row: int = 0,
                            **kwargs: Any) -> Any:
        """Draw one row of a shape set's transformation on its geometry,
        as arrows scaled per DOF, as the app's Shape Values view does
        with a transformation picked.

        The row is a mode; each arrow is at one of `dofs`, along it, its
        length the row's value there as a share of the row's largest
        (`viz.geometry.transformation_arrows`), pointing the other way
        for a negative value. The geometry is the one in the set's
        object group (`geometry_for`). A set with no geometry, or the
        'shapes' form, whose rows are DOFs rather than modes, raises
        ValueError.

        Parameters
        ----------
        name : str or ShapeSet
            The shape set.
        dofs : str or sequence of str, optional
            Where, as for `table`: a list of DOFs or a channel choice
            ('reference', 'response', ...) of the channel table in the
            set's object group. Every DOF of the set when left out.
        form : str, default 'reference'
            'reference' (`Φᵀ`) or 'response' (`Φ⁺`).
        row : int, default 0
            Which mode's row, from 0.
        **kwargs
            Passed to `viz.geometry.plot_transformation`, such as
            `screenshot=`.

        Returns
        -------
        object
            The plotter, or the image when rendered to `screenshot`.
        """
        from .core.transform import matrix_at
        from .viz.geometry import plot_transformation

        name = self.name_of(name)
        if form not in ('reference', 'response'):
            raise ValueError(f"{form!r} has no rows to draw; use 'reference' "
                             "or 'response'")
        found = self.geometry_for(name)
        if found is None:
            raise ValueError(f'{name} has no geometry to draw on')
        view = matrix_at(self[name], self._shape_dofs(name, dofs), form)
        return plot_transformation(found[1], view, row, **kwargs)

    def transformation_of(self, name: Any, dofs: Any = None, *,
                          max_condition: float | None = None) -> Any:
        """A rigid set's transformation at a choice of DOFs, as an object
        (`core.transformation.Transformation.from_shapes`): the matrix
        from motions at the DOFs to the set's six coordinates, with its
        conditioning, its residual projector and its table.

        Parameters
        ----------
        name : str or ShapeSet
            Six unscaled rigid-body shapes.
        dofs : str or sequence of str, optional
            Where, as for `table`: a list of DOFs or a channel choice
            ('control', 'response', ...) of the channel table in the
            set's object group. Every DOF of the set when left out.
        max_condition : float, optional
            The conditioning past which it is refused; the package's
            limit (`core.transformation.MAX_CONDITION`) when left out.

        Returns
        -------
        Transformation

        Raises
        ------
        ValueError
            If the set is not unscaled rigid shapes, a DOF is not in it,
            or the DOFs resolve too little or too poorly.
        TypeError
            If the object is not a shape set.
        """
        from .core.transformation import MAX_CONDITION, Transformation

        name = self.name_of(name)
        shapes = self[name]
        if not isinstance(shapes, ShapeSet):
            raise TypeError(f'{name} is a {type(shapes).__name__}, not a '
                            'shape set')
        chosen = self._shape_dofs(name, dofs)
        return Transformation.from_shapes(
            shapes, list(shapes.coordinate) if chosen is None else chosen,
            max_condition=(MAX_CONDITION if max_condition is None
                           else float(max_condition)))

    def read_transformation(self, name: Any, *,
                            threshold: float | None = None,
                            progress: Any = None) -> Any:
        """What a rigid-body transformation leaves unexplained of the
        density it was applied to, line by line
        (`core.transformation.read_residual`).

        `name` is a density transformed in this project (Transform to
        Modal Responses) from a cross-spectral density through six
        unscaled rigid-body shapes. The transformation is rebuilt from
        the set at the channels the transform used, and the source's
        measured power at those channels is split into the part the six
        motions explain and the residual, with the frequency up to which
        the residual stays under `threshold` of it. The conditioning is
        reported, not refused: a reading of a poor layout is the point.

        Parameters
        ----------
        name : str or object
            The transformed density, by name or as the object itself.
        threshold : float, optional
            The residual's share of the measured power at which the
            rigid-body assumption is said to stop holding; a tenth when
            left out (`core.transformation.RESIDUAL_THRESHOLD`).
        progress : callable, optional
            Called with ``(0, core.progress.BUSY)`` as the work starts,
            which cannot count its own stages; the app's progress bar
            uses it.

        Returns
        -------
        TransformationResidual

        Raises
        ------
        ValueError
            If the object was not transformed in this project from a
            cross-spectral density through rigid shapes, or no motion
            channel of the source is in the set.
        """
        from .core.progress import Ticker
        from .core.transform import MOTIONS, to_modal
        from .core.transformation import (
            RESIDUAL_THRESHOLD,
            Transformation,
            read_residual,
        )

        Ticker(progress).busy()
        name = self.name_of(name)
        provenance = self.provenance.get(name, {})
        params = provenance.get('params', {}) or {}
        source = self.get(provenance.get('source'))
        shapes = self.get(params.get('shapes'))
        if (provenance.get('verb') != 'transform'
                or not isinstance(source, Psd)
                or source.reference_dof is None
                or not isinstance(shapes, ShapeSet)):
            raise ValueError(
                f'{name!r} was not transformed in this project from a '
                'cross-spectral density, so there is no measured matrix '
                'behind it to split')
        picked = params.get('records')
        _modal, report = to_modal(source, shapes, picked)
        motions = [quantity for quantity in report.shared
                   if quantity in MOTIONS]
        if not motions:
            raise ValueError(f'no motion density of {provenance["source"]!r} '
                             f'is at a DOF of {params["shapes"]!r}')
        quantity = motions[0]
        transformation = Transformation.from_shapes(
            shapes, report.shared[quantity], max_condition=np.inf)
        return read_residual(
            transformation, source, picked, quantity=quantity,
            threshold=(RESIDUAL_THRESHOLD if threshold is None
                       else float(threshold)))

    def _shape_dofs(self, name: str, dofs: Any) -> list[str] | None:
        """`dofs` as DOF strings for the shape values of `name`: a list as
        given, None for every DOF of the set, or a channel choice read
        from the channel table in its object group, in table order and
        with a channel's sign as the table writes it. One reading for
        the window's drop-down and a script's `table`."""
        if dofs is None or not isinstance(dofs, str):
            return None if dofs is None else [str(dof) for dof in dofs]
        found = self.channel_table_for(name)
        if found is None:
            raise ValueError(f'{name} has no channel table in its object '
                             f'group to take {dofs!r} channels from')
        table = found[1]
        strings = table.dof_strings()
        # a channel with no node or direction yet names no DOF
        return [strings[i] for i in table.chosen(dofs)
                if str(table['node'][i]) and str(table['direction'][i])]

    def plot(self, name: str, **kwargs: Any) -> Any:
        """Plot an object the way the app plots it: data as curves, a
        geometry as its scene, a shape set as its auto-MAC.

        An object with no plot raises TypeError.

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
        """Animate a mode shape, a complex spectrum's operating
        deflection shape, or a time history, on the object's geometry.

        The geometry is the one `geometry_for` gives; with none, this
        raises ValueError. A spectrum's line is picked with
        `frequency=` in `kwargs`. A time history whose modal records
        have a shape set behind them (`modal_basis`) plays those records
        through the set's shapes on the set's geometry, unless
        `quantity=` picks one of its physical measurements.

        Parameters
        ----------
        name : str
            The shape set or spectrum to animate.
        mode : int, default 0
            Which mode, by index. Not used for a spectrum.
        **kwargs
            Passed through to the scene. `sensors=True` draws the sensors
            of the channel table linked with the object
            (`channel_table_for`) at their nodes, moving with them, and
            `sensor_size` multiplies their true size.

        Returns
        -------
        object
            The plotter the animation is running in.
        """
        name = self.name_of(name)
        target = self[name]
        drawn_on = name
        if kwargs.get('sensors') is True:
            found = self.channel_table_for(name)
            if found is None:
                raise ValueError(f'{name!r} has no channel table linked to '
                                 'draw sensors from')
            kwargs = {**kwargs, 'sensors': found[1]}
        quantity = kwargs.get('quantity')
        if (isinstance(target, TimeHistory) and 'shapes' not in kwargs
                and (quantity is None or quantity.startswith('modal_'))):
            basis = self.modal_basis(name, kwargs.get('records'))
            if basis is not None:
                # the modes move the set's geometry, not the forces'
                drawn_on, kwargs = basis, {**kwargs, 'shapes': self[basis]}
        home = self.geometry_for(drawn_on)
        if home is None:
            raise ValueError(f'{drawn_on!r} has no geometry — link one')
        if isinstance(target, ShapeSet):
            return target.animate(home[1], mode, **kwargs)
        # spectra pick a line by `frequency=`, not a mode number, and a
        # positional 0 must not read as 0 Hz
        return target.animate(home[1], **kwargs)

    def animate_pair(self, first: Any, mode: int, second: Any,
                     other_mode: int, **kwargs: Any) -> Any:
        """Animate two shape sets' modes overlaid on their geometries, as
        the app does for a picked cell of their cross-MAC.

        Each set is drawn on its own geometry (`geometry_for`), or on
        the first set's when the second has none; the first must have
        one, or this raises ValueError. The Basis set is drawn solid and
        the other partly transparent over it (`theme.overlay_alphas`),
        whichever order they are given in; with no Basis between them,
        the first is drawn solid. The caption gives both names.

        Parameters
        ----------
        first, second : str or ShapeSet
            The two shape sets.
        mode, other_mode : int
            Their modes, by index.
        **kwargs
            Passed through to `ShapeSet.animate_pair`: `on_basis`,
            `screenshot=`, `movie=`, `scale`, `colors`.

        Returns
        -------
        object
            The plotter, the still, or the movie's path.
        """
        from .theme import overlay_alphas

        first, second = self.name_of(first), self.name_of(second)
        homes = [self.geometry_for(name) for name in (first, second)]
        if homes[0] is None:
            raise ValueError(f'{first!r} has no geometry — link one')
        kwargs.setdefault('alphas', overlay_alphas(self.role_of(first),
                                                   self.role_of(second)))
        return self[first].animate_pair(
            homes[0][1], mode, self[second], other_mode,
            (homes[1] or homes[0])[1], names=(first, second), **kwargs)

    @property
    def names(self) -> list[str]:
        """Every object's name, in the order the tree shows them."""
        return self.ordered_names()

    def name_of(self, target: Any) -> str:
        """The name an object is stored under; a string is returned as
        it is.

        Every verb accepts either, so `project.compute_psds('Time
        History')` and `project.compute_psds(project.basis.time_history)`
        are the same call. An object not in the project raises
        ValueError.

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
                     specification: Any = None,
                     progress: Any = None) -> list[str]:
        """Read each specification tone's level out of a recording
        (Extract Sine Levels), as one sine level set holding every tone.

        Each tone is read against its own sweep, since each sweeps its
        own frequencies on its own clock. When no specification is
        named, the project's one sine sweep specification is used. The
        settings are the recording's `sine_extraction`; when none is
        set, the suggested setting is used, as `filter_data` uses its
        suggested filter, and the setting with the smoothing chosen is
        stored on the recording. The result is linked to the recording.

        Parameters
        ----------
        source : str or object
            The recording (a time history) to read.
        specification : str or object, optional
            The sweep specification to extract against. Defaults to the
            project's own, when it holds exactly one.
        progress : callable, optional
            Called with ``(done, total)`` as the extraction advances:
            the matched filter for each tone, the smoothing steps, then
            the solve's pieces for each channel. The app's progress bar
            uses it; a script can print it. Anything it raises stops
            the extraction.

        Returns
        -------
        list of str
            A list holding the name of the sine level set added.
        """
        from dataclasses import replace

        from .core.sine import extract_sine

        source = self.name_of(source)
        if specification is None:
            spec = self.sine_sweep_specification
            spec_name = self.name_of(spec)
        else:
            spec_name = self.name_of(specification)
            spec = self[spec_name]
        history = self[source]
        # the settings are the history's own; with none set, the
        # suggestion is adopted the way `filter_data` adopts its filter,
        # so the button works before the sine view has been visited
        setting = history.sine_extraction
        if setting is None:
            setting = history.suggest_sine_extraction()
        levels = extract_sine(history, spec, cycles=setting.cycles,
                              refine=setting.refine,
                              target_db=setting.target_db,
                              progress=progress)
        # what the automatic chose rides the setting, so the view and
        # the journal say what was used
        history.sine_extraction = replace(setting, chosen=levels.cycles)
        return [self._derive(source, levels, 'Sine Levels',
                             recipe=('extract_sine',
                                     {'specification': spec_name}))]

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
        'compute_cpsds': 'averaging',
        # the two read the channels' roles as well as the framing; the
        # fingerprint stays an 'averaging' one with the roles folded in
        # only when set, so a project saved before roles existed opens
        # with nothing marked stale
        'compute_frfs': 'averaging+roles',
        'compute_multiple_coherence': 'averaging+roles',
        'compute_srs': 'shocks', 'compute_octave': 'content',
        'make_realizable': 'content',
        # the levels of the set; only the first source is fingerprinted,
        # which is the one the result is filed under
        'derive_specification': 'content',
        # the filter reads the settings riding the source; integration
        # and differentiation read the record's bytes, so refreshing a
        # re-filtered source cascades down the whole motion chain
        'filter_data': 'filtering', 'truncate_data': 'truncation',
        # the smoothing and the refinement the levels were read with
        'extract_sine': 'sine_extraction',
        'integrate': 'content', 'differentiate': 'content',
        'resample': 'content',
        # the settings and the nodes both: a moved node, or a turned
        # displacement system, changes the shapes as surely as a moved
        # reference point does
        'generate_rigid_body_modes': 'rigid',
        # what the signal is drawn from: the spectrum's levels, the
        # tone's schedule, the table's reference channels
        'generate_signal': 'signal',
        # the record and the set both: a refitted set badges the modal
        # responses made through it
        'transform': 'pair', 'expand': 'pair',
        # the pair and the set's damping and modal masses, which a
        # transform never reads and a response does
        'compute_modal_responses': 'pair+model'}

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
        if reads in ('pair', 'pair+model'):
            other = self.get((params or {}).get('shapes'))
            if other is None:
                return None
            import hashlib

            digest = hashlib.blake2b(digest_size=16)
            arrays = [obj.abscissa, obj.ordinate, other.shape_matrix,
                      other.frequency]
            if reads == 'pair+model':
                arrays += [other.damping, other.modal_mass]
            for array in arrays:
                digest.update(np.ascontiguousarray(array).tobytes())
            digest.update(','.join(obj.response_dof).encode())
            digest.update(','.join(other.coordinate).encode())
            return ('content', digest.hexdigest())
        if reads in ('averaging', 'averaging+roles'):
            averaging = getattr(obj, 'averaging', None)
            state = None if averaging is None else asdict(averaging)
            resolved = (obj.channel_roles()
                        if reads.endswith('roles')
                        and getattr(obj, 'roles', None) is not None else None)
            if resolved is not None and resolved != obj.channel_roles(
                    stated=False):
                # the composite kind only where the roles differ from the
                # guess: otherwise the fingerprint is the plain averaging
                # one it always was, so an older project opens with
                # nothing stale, and stating what the guess already said
                # (an import seeding from its table) changes nothing
                return ('averaging+roles',
                        {'averaging': state,
                         'roles': sorted([dof, quantity, role] for
                                         (dof, quantity), role
                                         in resolved.items())})
            return ('averaging', state)
        if reads == 'filtering':
            filtering = getattr(obj, 'filtering', None)
            return ('filtering', None if filtering is None
                    else asdict(filtering))
        if reads == 'sine_extraction':
            setting = getattr(obj, 'sine_extraction', None)
            # `chosen` is what the automatic found, not a choice: a
            # fingerprint that moved when the data answered would
            # badge a result for having been computed
            return ('sine_extraction', None if setting is None else
                    (setting.cycles, setting.target_db, setting.refine))
        if reads == 'truncation':
            truncation = getattr(obj, 'truncation', None)
            return ('truncation', None if truncation is None
                    else asdict(truncation))
        if reads == 'shocks':
            shocks = getattr(obj, 'shocks', None) or ()
            return ('shocks', tuple((float(s.start), float(s.duration))
                                    for s in shocks))
        if reads == 'signal':
            return ('signal', _signal_source_digest(obj))
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
        """{derived name: reason} for every derived object whose source's
        settings have changed since it was computed.

        An object whose source is gone, or whose provenance records no
        state to compare, is not reported.
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

        Links, the report's figures and the grids all refer to the
        object by name, so replacing the value under that name keeps
        every reference correct. Objects derived from this one then
        become stale by content, and are refreshed one at a time or all
        together with `refresh_stale`, never automatically. An object
        with no recorded derivation, or whose source is gone, raises
        ValueError.

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
        """Refresh every stale object, sources before the objects derived
        from them, until nothing is stale. Returns the names refreshed.

        This is the refresh on the project's row in the tree.
        """
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

    def save(self, path: str | os.PathLike, **options: Any) -> str:
        """Write the whole project to one file: `.vdyn`, `.mat` for the
        same layout in a MATLAB file, or `.h5` for the Engineering
        Sciences Common Data Format.

        Parameters
        ----------
        path : str or os.PathLike
            Where to write the file. A `.mat` suffix writes the project
            as MATLAB structs (`io.matlab`), and `.h5`, `.hdf5` or
            `.escdf` writes the standard's types with each object stored
            whole in an attachment (`io.escdf_objects`). Any other
            suffix writes a `.vdyn` file.
        **options
            Passed to the MATLAB or ESCDF writer, for example an ESCDF
            file's `created_by`. Not used for a `.vdyn` file.

        Returns
        -------
        str
            The path written.
        """
        from .io import export_file, save_test
        from .io.escdf_objects import SUFFIXES

        if str(path).endswith('.mat'):
            export_file(self, str(path), **options)
            return str(path)
        if str(path).endswith(SUFFIXES):
            # by name: the writer's own suffix is `.h5`, and picking by
            # suffix would refuse the other two it keeps
            export_file(self, str(path), format='escdf', **options)
            return str(path)
        save_test(str(path), self.name, dict(self),
                  active_geometry=self.active_geometry,
                  project_type=self.project_type, object_groups=self.object_groups,
                  provenance=self.provenance,
                  data_index=self.data_index.records())
        written = str(path) if str(path).endswith('.vdyn') else f'{path}.vdyn'
        self.file = os.path.abspath(os.path.expanduser(written))
        return str(path)

    @classmethod
    def open(cls, path: str | os.PathLike) -> Project:
        """Read a project from a `.vdyn`, `.mat` or ESCDF `.h5` (`.hdf5`,
        `.escdf`) file.

        The journal starts over with the one line that opens the file. A
        file holding a single object rather than a project raises
        ValueError.
        """
        from .io import import_file, load
        from .io.escdf_objects import SUFFIXES

        loaded = (import_file(str(path))
                  if str(path).endswith(('.mat', *SUFFIXES))
                  else load(str(path)))
        if isinstance(loaded, Project):
            # whatever the loader did on the way — construct, add,
            # link — the session's story starts here: one line that
            # reproduces this state exactly
            loaded.journal = [
                f'project = visualdynamics.Project.open({str(path)!r})']
            loaded.file = os.path.abspath(os.path.expanduser(str(path)))
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
                              tuple, list, View)):
            return repr(value)
        try:
            return repr(self.name_of(value))
        except (KeyError, ValueError):
            return repr(value)

    @contextlib.contextmanager
    def journal_as(self, line: str | None):
        """Record a stretch of front-end work as the one line that
        replays it.

        The app imports a file by building the objects itself and
        adding them one at a time. Journaled verb by verb, that stretch
        would be lines that do not replay, when the accurate record is
        the single `import_file` call a script would make. Inside the
        stretch, verbs record nothing, the same as verbs called from
        other verbs, and the line is added only when the stretch
        finishes without an error.

        Parameters
        ----------
        line : str or None
            The line that replays the stretch, or None to record
            nothing.

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
        """Journal a settings write the way a script would make it.

        The app stores analysis settings by assignment (a dragged
        averaging span, a filter corner, the shock windows), and those
        writes are part of the session as much as any verb. A repeated
        write to the same attribute replaces its own last line, so a
        setting adjusted many times leaves one assignment, not a line
        for each change. An object not in the project is ignored.

        Parameters
        ----------
        target : str or object
            The object written to, by name or as the object.
        attribute : str
            Which settings attribute was stored.
        value : Any
            What was stored. Its repr must rebuild it, as every settings
            dataclass in visualdynamics does.

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
                    settle: bool = False, **kwargs: Any) -> None:
        """Journal a method call on an object the way a script would make
        it.

        This is for object methods that are not Project verbs, such as a
        line added to a geometry or a photo renamed, so the console
        shows every change made in the session. An object not in the
        project is ignored.

        Parameters
        ----------
        target : str or object
            The object acted on, by name or as the object.
        method : str
            The method a script would call.
        *args : Any
            The call's arguments. Their reprs must rebuild them.
        settle : bool, default False
            Replace the last line when it was the same call (the same
            object, method, positional arguments and keyword names, with
            only the keyword values different), so a gesture adjusted
            again leaves one line, not one per adjustment.
        **kwargs : Any
            Keyword arguments, under the same rule as `args`.

        Returns
        -------
        None
        """
        # Brandon, 2026-08-30: adding a line said nothing in the console.
        try:
            name = self.name_of(target)
        except (KeyError, ValueError):
            return
        shown = [self._journal_arg(a) for a in args]
        key = (name, method, tuple(shown), tuple(kwargs))
        shown += [f'{word}={self._journal_arg(value)}'
                  for word, value in kwargs.items()]
        line = f'project[{name!r}].{method}({", ".join(shown)})'
        last = self._settling
        if (settle and last is not None and last[0] == key
                and self.journal and self.journal[-1] == last[1]):
            self.journal[-1] = line
        else:
            self.journal.append(line)
        self._settling = (key, line) if settle else None

    #: what a settings repr needs in scope, added to `session_script`
    #: only when a journal line uses it
    _SCRIPT_IMPORTS: ClassVar[tuple[tuple[str, str], ...]] = (
        ('Averaging(', 'from visualdynamics.core.averaging import Averaging'),
        ('Filtering(', 'from visualdynamics.core.filters import Filtering'),
        ('SineExtraction(',
         'from visualdynamics.core.sine import SineExtraction'),
        ('Truncation(', 'from visualdynamics.core.truncate import Truncation'),
        ('Shock(', 'from visualdynamics.core.shocks import Shock'),
        ('Segment(',
         'from visualdynamics.core.replication_segments import Segment'),
        ('Photos(', 'from visualdynamics.core.photos import Photos'),
        ('MassProperties(',
         'from visualdynamics.core.rigid import MassProperties'),
        ('View(', 'from visualdynamics import View'),
        ('SpecificationDraft(',
         'from visualdynamics.core.author import SpecificationDraft'),
        ('np.array(', 'import numpy as np'),
        ('visualdynamics.viz.scalogram.',
         'import visualdynamics.viz.scalogram'),
        ('visualdynamics.core.sweep_rate.',
         'import visualdynamics.core.sweep_rate'),
        ('GroupProperties(',
         ('from visualdynamics.core.fem import RIGID, GroupProperties, '
          'Material, Section')),
    )

    def session_script(self) -> str:
        """This session's changes to the project as a runnable Python
        script.

        It is the journal under the imports it needs: every verb that
        ran and every setting stored, whether clicked in the app or
        called from a script, as the line that reproduces it. A session
        worked up by hand can then be replayed, adapted or kept. Reads
        and refused calls are left out, because the script records what
        changed the project, and a verb that raised changed nothing.

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


def _journaled(verb, name=None):
    """Record a successful verb call as the line that would replay it,
    under `name` when given (a private step journaled as its verb).

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
            # a callback — the window's progress bar — is this session's,
            # not the script's: a replay has no bar to feed, and the
            # line reads as the user could have typed it (2026-10-01)
            shown += [f'{key}={self._journal_arg(value)}'
                      for key, value in kwargs.items()
                      if not callable(value)]
            called = name or verb.__name__
            line = f'project.{called}({", ".join(shown)})'
            if unsayable:
                line = f'# {line} — not replayable'
            # the sheet edits a specification live, one verb call per
            # keystroke (2026-09-06): a run of edits on one object
            # settles to the line that stands, as a nudged setting
            # does in `record_setting`, rather than a line per key
            prefix = (f'project.{called}({shown[0]}, '
                      if called in _SETTLING and shown else None)
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
def _has_both_sides(history) -> bool:
    """Whether a history's roles give an FRF something to estimate."""
    return bool(history.reference_channels()) and bool(
        history.response_channels())


def _table_identities(table, history) -> list:
    """Each channel-table row as the (DOF, quantity) channel it names,
    or None where it names none of `history`'s channels; with no
    history, each row's own (DOF, type). `ChannelTable.identities`
    holds the rule."""
    return table.identities(history)


def _is_time(project, obj):
    return isinstance(obj, TimeHistory)


def _has_dims(obj, wanted):
    return isinstance(obj, TimeHistory) and bool(
        {obj.known_dim(i) for i in range(obj.num_records)} & wanted)


def _plannable(project, obj) -> bool:
    """Whether a shape set can plan a test's instrumentation: it has a
    mode above 0 Hz and is drawn on a geometry with solids, plates or
    beams to place sensors on."""
    from .core.geometry import ELEMENT_TYPES

    if not isinstance(obj, ShapeSet) or not np.any(np.asarray(obj.frequency) > 0):
        return False
    try:
        found = project.geometry_for(project.name_of(obj))
    except (KeyError, ValueError, TypeError):
        return False
    if found is None:
        return False
    kinds = {ELEMENT_TYPES.get(int(code), ('', 0, ''))[2]
             for code in np.unique(found[1].elem_type)}
    return bool(kinds & {'volume', 'face', 'line'})


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
def _makes_signals(obj: Any) -> bool:
    """Whether a signal can be generated from `obj` (`generate_signal`)."""
    from .core.signals import kinds_for

    return bool(kinds_for(obj))


def _generated_again(source: Any, params: Mapping[str, Any]) -> Any:
    """A generated signal made again from its recorded settings."""
    from .core.signals import SignalSettings, generate

    return generate(source, SignalSettings.from_dict(params))


def _signal_source_digest(obj: Any) -> str:
    """What a generated signal reads from its source, as a digest: a
    spectrum's lines and levels, a sine specification's tones, a channel
    table's channels, roles and types."""
    import hashlib

    from .core.sine import SineSweepSpecification

    digest = hashlib.blake2b(digest_size=16)
    if isinstance(obj, SineSweepSpecification):
        for tone in obj.tones:
            for array in (tone.frequency, tone.amplitude, tone.segment_rate):
                digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(','.join(obj.response_dof).encode())
    elif isinstance(obj, ChannelTable):
        digest.update('|'.join(obj.dof_strings()).encode())
        digest.update('|'.join(obj.roles()).encode())
        digest.update('|'.join(obj.types()).encode())
    else:
        digest.update(np.ascontiguousarray(obj.abscissa).tobytes())
        digest.update(np.ascontiguousarray(obj.ordinate).tobytes())
        digest.update(','.join(obj.ordinate_dim).encode())
        if isinstance(obj, ShockSpecification):
            # a synthesis puts every point inside the warning band, at
            # the requirement's Q and of its type
            digest.update(f'{obj.q!r}|{obj.kind}'.encode())
            for key in sorted(obj.limits or {}):
                digest.update(key.encode())
                digest.update(np.ascontiguousarray(obj.limits[key]).tobytes())
    return digest.hexdigest()


_VERB_APPLIES: tuple = (
    ('integrate', _integrable),
    ('differentiate', _differentiable),
    ('filter_data', _is_time),
    ('truncate_data', _is_time),
    ('resample', _is_time),
    ('detect_shocks', _is_time),
    ('compute_spectra', _is_time),
    ('compute_psds', _is_time),
    ('compute_cpsds', _is_time),
    ('compute_srs', _is_time),
    # offered where the roles give both sides: a history with no
    # reference, or none but references and monitors, has nothing to
    # estimate
    ('compute_frfs', lambda p, o: (_is_time(p, o) and _has_both_sides(o))),
    ('compute_multiple_coherence', lambda p, o: (
        _is_time(p, o) and _has_both_sides(o))),
    ('extract_sine', lambda p, o: (_is_time(p, o) and any(
        isinstance(other, SineSweepSpecification)
        for other in p.values()))),
    # a specification bands too, limits and all (Brandon, 2026-09-18)
    ('compute_octave', lambda p, o: isinstance(o, Psd)),
    # a cross-spectral matrix repaired where no signals could have it;
    # the Realizability reading's apply button
    ('make_realizable', lambda p, o: (isinstance(o, Psd)
                                      and o.reference_dof is not None)),
    # what a file could not say: a PSD that is a requirement, and back.
    # A specification with limits is converted too, the limits left
    # out (Brandon, 2026-10-29): the Specification sheet always writes
    # its bands, so a target made there could never become a CPSD
    ('convert_to_specification', lambda p, o: (
        isinstance(o, Psd) and not isinstance(o, Specification))),
    ('convert_to_psd', lambda p, o: isinstance(o, Specification)),
    ('fit_modes', lambda p, o: isinstance(o, Frf)),
    ('transform', lambda p, o: _reads_as(p, o, 'physical')),
    ('expand', lambda p, o: _reads_as(p, o, 'modal')),
    ('generate_rigid_body_modes', lambda p, o: isinstance(o, Geometry)),
    ('generate_signal', lambda p, o: _makes_signals(o)),
    # a geometry is a model once its element groups say what they are made of
    ('solve_modes', lambda p, o: (isinstance(o, Geometry)
                                  and bool(o.group_properties))),
    # finite element modes on a model to put sensors on (Brandon,
    # 2026-10-21, Plan Instrumentation)
    ('plan_instrumentation', _plannable),
    # coincident nodes matter to connectivity, so a geometry of elements;
    # a sensor layout of bare nodes has none to tie
    ('merge_coincident_nodes', lambda p, o: (isinstance(o, Geometry)
                                             and len(o.elem_conn) > 0)),
    # where parts meshed apart meet, the step before the merge
    ('align_meshes', lambda p, o: (isinstance(o, Geometry)
                                   and len(o.elem_conn) > 0)),
    # surfaces to mate: two element groups or more; the app offers it on
    # the groups picked, as it does the merge (2026-10-20)
    ('mate_surfaces', lambda p, o: (isinstance(o, Geometry)
                                    and len(o.group_id) > 1 and len(o.elem_conn) > 0)),
    # solids go into any geometry, an empty one above all (2026-10-11;
    # Add Plane and Add Block, which put elements straight in, retired
    # 2026-10-26 for plates and blocks meshed by Mesh Solids)
    ('add_solid_block', lambda p, o: isinstance(o, Geometry)),
    ('add_solid_cylinder', lambda p, o: isinstance(o, Geometry)),
    ('add_solid_cone', lambda p, o: isinstance(o, Geometry)),
    ('add_solid_sphere', lambda p, o: isinstance(o, Geometry)),
    ('add_solid_section', lambda p, o: isinstance(o, Geometry)),
    ('add_solid_plate', lambda p, o: isinstance(o, Geometry)),
    ('add_solid_disk', lambda p, o: isinstance(o, Geometry)),
    ('set_view', lambda p, o: isinstance(o, Geometry)),
    # a channel table takes rows, an empty one above all (2026-10-22)
    ('add_channels', lambda p, o: isinstance(o, ChannelTable)),
    # a channel table's control channels, or a specification to edit;
    # a shape set's modal coordinates came off the bar (Brandon,
    # 2026-10-22), and a script can still write one through the verb
    ('author_specification', lambda p, o: isinstance(
        o, (Specification, ChannelTable))),
    ('project_onto_basis', _two_shape_sets),
    ('match_modes', _two_shape_sets),
    ('merge', lambda p, o: any(type(other) is type(o) and other is not o
                               for other in p.values())),
)


def _dof_nodes(obj) -> set[int]:
    """The nodes an object's DOFs name: a record's responses and
    references, a shape set's coordinates — what a node deleted or moved
    under data would leave pointing nowhere."""
    from .core.data import parse_dof

    found = set()
    for dofs in (getattr(obj, 'response_dof', None), getattr(obj, 'reference_dof', None),
                 getattr(obj, 'coordinate', None)):
        for dof in dofs if dofs is not None else ():
            node, _direction = parse_dof(dof)
            if node is not None:
                found.add(node)
    return found


def _reads_as(project, obj, direction):
    """Whether some shape set in the project reads `obj` as physical
    (transformable) or modal (expandable) — the applicability of a
    two-object verb, answered against every partner available."""
    if not isinstance(obj, DataArray):
        return False
    return any(_goes(obj, other, direction)
               for other in project.values() if isinstance(other, ShapeSet))


def _goes(data, shapes, direction):
    """Whether `data` transforms through `shapes` the given way. Modal
    is wider than reading as modal: a history of forces with modal
    responses appended reads physical, and its modal records expand."""
    from .core.transform import expands, reads_as

    if direction == 'modal':
        return expands(data, shapes)
    return reads_as(data, shapes) == direction


def _physical_name(source, result=None):
    """'Run Modal Coordinates' expands to 'Run Physical Responses'; a
    specification to '… Physical Specification'; anything else gets
    the responses suffix."""
    # 'Modal Responses' is the name transforms took until Brandon renamed
    # the act Transform to Modal Coordinates (2026-10-25), and the name
    # Compute Modal Responses gives its result (2026-10-27)
    for suffix in (' Modal Coordinates', ' Modal Responses'):
        if source.endswith(suffix):
            return source[:-len(suffix)] + ' Physical Responses'
    if isinstance(result, Specification):
        return f'{source} at Physical DOFs'
    return f'{source} Physical Responses'


def _pair_reads(project, objects, direction):
    """One data object and one shape set — a geometry may come along
    — and the record reads through the set the given way."""
    data = [o for o in objects if isinstance(o, DataArray)]
    sets = [o for o in objects if isinstance(o, ShapeSet)]
    rest = [o for o in objects
            if not isinstance(o, (DataArray, ShapeSet, Geometry))]
    return (len(data) == 1 and len(sets) == 1 and not rest
            and _goes(data[0], sets[0], direction))


def _two_sets_on_two_geometries(project, objects):
    """Two shape sets, each answering to its own geometry — the
    projection samples one model's field at the other's points."""
    if len(objects) != 2 or not all(isinstance(o, ShapeSet) for o in objects):
        return False
    homes = [project.geometry_for(o) for o in objects]
    return (all(home is not None for home in homes)
            and homes[0][1] is not homes[1][1])


def _set_and_table_apart(project, objects):
    """A shape set and a channel table, each answering to its own
    geometry: the model's modes put at the test's channels."""
    sets = [o for o in objects if isinstance(o, ShapeSet)]
    tables = [o for o in objects if isinstance(o, ChannelTable)]
    if len(objects) != 2 or len(sets) != 1 or len(tables) != 1:
        return False
    homes = [project.geometry_for(sets[0]), project.geometry_for(tables[0])]
    return (all(home is not None for home in homes)
            and homes[0][1] is not homes[1][1])


def _mergeable(project, objects):
    from .core.merge import mergeable

    return len(objects) >= 2 and mergeable(list(objects)) is None


#: the verbs that act on a *combination* of objects, with what makes
#: one — the bar's answer for a multi-selection (`selection_verbs`)
def _forces_and_modes(objects: Sequence[Any]) -> bool:
    """One time history holding a force channel and one real shape set
    (`compute_modal_responses`)."""
    histories = [o for o in objects if isinstance(o, TimeHistory)]
    sets = [o for o in objects if isinstance(o, ShapeSet)]
    if len(objects) != 2 or len(histories) != 1 or len(sets) != 1:
        return False
    if sets[0].num_shapes == 0 or sets[0].is_complex:
        return False
    return 'force' in histories[0].ordinate_dim or any(
        dim == 'modal_force' for dim in histories[0].ordinate_dim)


_SELECTION_APPLIES: tuple = (
    ('compute_modal_responses', lambda p, o: _forces_and_modes(o)),
    ('transform', lambda p, o: _pair_reads(p, o, 'physical')),
    ('expand', lambda p, o: _pair_reads(p, o, 'modal')),
    ('project_onto_basis', lambda p, o: (_two_sets_on_two_geometries(p, o)
                                         or _set_and_table_apart(p, o))),
    ('match_modes', lambda p, o: (len(o) == 2 and all(
        isinstance(x, ShapeSet) for x in o))),
    ('merge', _mergeable),
)

#: the verbs that need a partner, so a lone selection is not offered them
PARTNER_VERBS = frozenset(verb for verb, _applies in _SELECTION_APPLIES)


def _solids_take_a_material_alone(geometry: Geometry, groups: Sequence[int]) -> dict:
    """Take a section, a thickness or an orientation off each of these
    element groups that holds solids alone, keeping its material, and
    say what went: {group name: 'a beam section and an orientation'}.

    A group keeps its properties when its part is meshed again, so one
    given a beam's section by mistake gave it to the new bricks too, and
    the solve refused them (Brandon, 2026-10-20: he deleted the mesh and
    meshed again twice, and the round tube came back each time)."""
    from dataclasses import replace

    from .core.geometry import SOLID_FAMILIES
    from .core.mesh import group_families

    cleared = {}
    names = dict(zip((int(g) for g in geometry.group_id), geometry.group_name))
    for group in groups:
        props = geometry.group_properties.get(int(group))
        families = group_families(geometry, int(group))
        if props is None or not families or not families <= SOLID_FAMILIES:
            continue
        went = [what for what, there in (('a beam section', props.section is not None),
                                         ('a thickness', props.thickness is not None),
                                         ('an orientation', props.orientation is not None))
                if there]
        if not went:
            continue
        geometry.group_properties[int(group)] = replace(props, section=None, thickness=None,
                                                        orientation=None)
        label = str(names.get(int(group), '')).strip() or f'element group {group}'
        cleared[label] = ' and '.join([', '.join(went[:-1]), went[-1]] if len(went) > 1
                                      else went)
    return cleared


_JOURNALED_VERBS = (
    'add', 'import_file', 'remove', 'rename', 'rename_dof', 'link', 'unlink',
    'relink', 'name_object_group',
    'place', 'set_role', 'set_basis', 'merge', 'set_channel_role',
    'compute_spectra', 'compute_psds', 'compute_cpsds', 'compute_octave',
    'make_realizable',
    'compute_frfs', 'compute_multiple_coherence', 'compute_srs',
    'detect_shocks', 'filter_data', 'truncate_data', 'resample', 'integrate',
    'differentiate', 'fit_modes', 'generate_rigid_body_modes', 'solve_modes',
    'generate_signal', 'compute_modal_responses',
    'plan_instrumentation',
    'merge_coincident_nodes', 'align_meshes', 'new_geometry',
    'new_channel_table', 'add_channels',
    'add_solid_block', 'add_solid_cylinder', 'add_solid_cone',
    'add_solid_sphere', 'add_solid_section', 'add_solid_plate', 'add_solid_disk',
    'add_solid_hole', 'combine_solids',
    'delete_solids', 'separate_solids', 'mesh_solids', 'pattern_solids',
    'copy_solids', 'reorder_solids',
    'edit_solid', 'move_solids', 'delete_elements', 'delete_element_groups',
    'tie_elements', 'mate_surfaces', 'add_mated_surfaces',
    'edit_mated_surfaces', 'delete_mated_surfaces',
    'merge_groups', 'set_view', 'set_comparison_scale',
    'convert_to_specification', 'convert_to_psd',
    'author_specification', 'derive_specification',
    'transform', 'expand', 'project_onto_basis', 'match_modes',
    'extract_sine', 'refresh', 'refresh_stale', 'work_up',
    'generate_report', 'report_features', 'report_section',
    'report_test_date', 'predict', 'report_prediction', 'export_report',
    'index_file', 'unindex_file', 'verify_data_index', 'report_check_block',
    'report_identity', 'export_report_pdf', 'issue_report', 'amend_report',
    'set_run_identity', 'report_runs', 'report_chronology',
    'accept_chronology',
    'report_finding', 'draft_finding_fact', 'report_flash',
    'report_requirement', 'draft_finding_from_requirement',
    'export', 'save', 'duplicate',
)
for _verb in _JOURNALED_VERBS:
    setattr(Project, _verb, _journaled(getattr(Project, _verb)))
# the window checks the data index and asks before it writes, then calls
# the write alone; the line it leaves is the verb a script would call
Project._export_pdf = _journaled(Project._export_pdf, 'export_report_pdf')
Project._issue_pdf = _journaled(Project._issue_pdf, 'issue_report')


def _extract_again(project, source, params):
    """The sine levels read again with the history's current setting
    — what `refresh` does for a level set whose smoothing moved."""
    from .core.sine import extract_sine

    history = project[source]
    setting = history.sine_extraction or history.suggest_sine_extraction()
    return extract_sine(history, project[params['specification']],
                        cycles=setting.cycles, refine=setting.refine,
                        target_db=setting.target_db)


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
    'make_realizable': lambda p, s, k: p[s].made_realizable(
        tolerance=k.get('tolerance'),
        keep_diagonal=k.get('keep_diagonal', True)),
    'compute_octave': lambda p, s, k: p[s].to_octave(
        k.get('per_octave', 6)),
    'filter_data': lambda p, s, k: p[s].filter(),
    'extract_sine': lambda p, s, k: _extract_again(p, s, k),
    'truncate_data': lambda p, s, k: p[s].truncate(),
    'resample': lambda p, s, k: p[s].resample(
        k['rate'], anti_alias=k.get('anti_alias', True), order=k.get('order')),
    # `get` with the module default, so a recipe that recorded None
    # stays raw — None is a choice here, not an absence
    'integrate': lambda p, s, k: p[s].integrate(
        k.get('drift_corner', _default_drift())),
    'differentiate': lambda p, s, k: p[s].differentiate(),
    'generate_rigid_body_modes': lambda p, s, k: _rigid_shapes(p[s]),
    'generate_signal': lambda p, s, k: _generated_again(p[s], k),
    'transform': lambda p, s, k: _through(p, s, k, 'to_modal'),
    'expand': lambda p, s, k: _through(p, s, k, 'to_physical'),
    'compute_modal_responses': lambda p, s, k: _answered(p, s, k),
    'author_specification': lambda p, s, k: _authored(p, s, k),
    'derive_specification': lambda p, s, k: _derived_again(p, k),
}


def _derived_again(project, params):
    """The specification a set gives, worked out again from the
    recipe: what `refresh` does when the set's levels have moved."""
    settings = {key: params[key] for key in (
        'statistic', 'coverage', 'confidence', 'k', 'sigma_db', 'band',
        'max_overshoot_db', 'rms', 'max_breakpoints') if key in params}
    fit = project.field_limit(params['sources'], params.get('records'),
                              **settings)
    return fit.specification(margin_db=params.get('margin_db', 0.0),
                             dof=params.get('dof'))


def _rewritten(spec, draft, comment):
    """The specification as the sheet leaves it: the sheet's own
    form when it holds every channel (fold a target to its
    breakpoints and the object *is* the breakpoints), merged at the
    object's lines when it holds a picked subset."""
    references = spec.reference_dof or list(spec.response_dof)
    autos = {spec.response_dof[i] for i in range(spec.num_records)
             if references[i] == spec.response_dof[i]}
    if set(draft.channels) != autos:
        return draft.written_into(spec, comment)
    out = draft.make(comment)
    # what the sheet does not edit rides through it: a mixed-mode
    # floor's narrowbands and the width left out about its tones
    out.narrowbands = tuple(getattr(spec, 'narrowbands', ()) or ())
    out.tone_exclusion = getattr(spec, 'tone_exclusion', None)
    return out


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


def _answered(project, source, params):
    """The modal responses `compute_modal_responses` makes, from its
    recipe: computed first and when refreshed."""
    from .core.filters import carrying_marks
    from .core.modal_response import modal_responses

    shapes = params.get('shapes')
    if shapes not in project:
        raise ValueError(f'{source!r} was answered through {shapes!r}, '
                         'which is gone')
    history, modal_set = project[source], project[shapes]
    if not isinstance(history, TimeHistory):
        raise TypeError(f'{source!r} is not a time history')
    if not isinstance(modal_set, ShapeSet):
        raise TypeError(f'{shapes!r} is not a shape set')
    responses = modal_responses(
        history, modal_set, modes=params.get('modes'),
        quantity=params.get('quantity', 'acceleration'),
        records=params.get('records'), label=shapes)
    return carrying_marks(history, responses)


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
    elif kind == 'averaging+roles' and value['averaging'] is not None:
        # the framing inside normalizes the way a bare one does
        value = dict(value, averaging=_named_state('averaging',
                                                   value['averaging']))
    return json.loads(json.dumps([kind, value]))


def _summarize_state(state):
    kind, value = state
    if kind == 'averaging':
        if value is None:
            return 'no framing'
        named = _named_state(kind, value)
        return (f'{named["frames"]} frames of {named["frame_length"]} '
                f'({named["window"]})')
    if kind == 'averaging+roles':
        framing = _summarize_state(('averaging', value['averaging']))
        named = ', '.join(f'{dof} {quantity}' for dof, quantity, role
                          in value['roles'] if role == 'reference')
        return f'{framing}, references {named or "none"}'
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
    """Work a Rattlesnake random vibration run up into a project.

        project = visualdynamics.random_vibration_run('run.nc4')
        project = visualdynamics.random_vibration_run(
            'run.nc4', last=100.0, geometry='article.stp', length_unit='mm',
            photos='setup_photos/')

    These are the steps the app takes from a controller file to a
    finished project, in the same order: import the run, average PSDs
    from the control time histories, put those on proportional bands,
    put the specification on the same bands with its limits (the report
    compares the banded measurement with the banded requirement), and
    compute how much of each response the drives account for. The run
    records that it is a random vibration test, so the project's type
    is set to Random Vibration.

    The averaging is what Detect finds: the controller's own frame
    length, window and overlap from the file, with the start and the
    number of frames worked out from the record (where the run is at
    level and how many frames that stretch holds). A script, an import
    and a click on Detect therefore give the same numbers.

    The other arguments are what the app's tree asks for after the run
    is imported, so a script never has to open the app: the article's
    geometry, with its length unit given when the file does not carry
    one; the setup photographs, as a folder of png or jpeg files, one
    file, or a list of files in the order they should appear; and, for
    a run too long to hold in memory, `last`, the number of seconds
    before the end to import. `last` is the same window the import
    dialog's Last field sets, and a run shorter than that is imported
    whole. The geometry and the photographs are linked into the run's
    own group, so the report can read them against the channel table.
    A run with no time data raises ValueError.

    Parameters
    ----------
    run : str or os.PathLike
        The controller's `.nc4`.
    per_octave : int, optional
        Bands per octave for the banded PSD and specification. Defaults
        to 6.
    last : float, optional
        Import only the last `last` seconds of the run's streams.
    geometry : str or os.PathLike, optional
        A geometry file to import and link to the run.
    length_unit : str, optional
        The geometry's length unit, for a file that does not say.
    photos : str, os.PathLike or sequence of them, optional
        A folder of photographs, one photograph, or several.

    Returns
    -------
    Project
        The worked-up project, of type Random Vibration.
    """
    # The specification is banded too, limits and all, because the
    # report reads the banded measurement against the banded
    # requirement (Brandon, 2026-09-18). The averaging is the Detect
    # answer, and the geometry, photos and `last` are taken here so a
    # script never has to open the window (Brandon, 2026-09-19).
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
    """Work up an imported system identification in place.

    The project type is set to System ID, the driven stream is renamed
    'Excitation Time History' and the ambient one 'Noise Time History',
    and the H1 FRFs, the multiple coherence and both streams' PSDs (on
    the excitation's frames) are computed. Returns the two stream
    names, with the ambient one None when the recording holds only the
    driven stream.
    """
    # Kept apart from reading the file so the work can be tested on a
    # recording built by hand: no file a test can commit holds the two
    # streams a system identification is made of (2026-09-22).
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
    """Work a Rattlesnake system identification up into a project.

        project = visualdynamics.system_id_run('sysid.nc4')

    These are the steps the app takes, in the same order: import the
    recording, name the two streams (the ambient and the driven), and
    compute the FRFs from the driven stream by H1, the controller's own
    estimator, since the excitation is known and the noise is on the
    response. Then compute the multiple coherence, and average a PSD
    from each stream on the same frames, since the signal-to-noise
    ratio is a ratio of densities and is only defined on lines both
    hold. The project's type is set to System ID. A recording with no
    time data raises ValueError.

    `last`, `geometry`, `length_unit` and `photos` mean the same as in
    `random_vibration_run`. The ambient stream, the geometry and the
    photographs are linked into the driven stream's group.

    Parameters
    ----------
    run : str or os.PathLike
        The controller's `.nc4`.
    last : float, optional
        Import only the last `last` seconds of the run's streams; a
        shorter run is imported whole.
    geometry : str or os.PathLike, optional
        A geometry file to import and link to the run.
    length_unit : str, optional
        The geometry's length unit, for a file that does not say.
    photos : str, os.PathLike or sequence of them, optional
        A folder of photographs, one photograph, or several in order.

    Returns
    -------
    Project
        The worked-up project, of type System ID.
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
        # a name ending in a separator says folder, so one not made yet
        # is made: it was taken as a folder and then not found when the
        # report was written (2026-10-13, found by the website pass)
        os.makedirs(path, exist_ok=True)
        stem = os.path.splitext(os.path.basename(os.path.expanduser(
            str(run))))[0]
        return os.path.join(path, stem + '.html')
    return path


def _runs_of(run: Any) -> tuple[list[Any], bool, bool]:
    """(runs, asked, many) for what a one-call report was handed.

    `ASK` opens the dialog; one path is one run; a list or tuple of
    paths is a batch, the form the window hands over after its own
    dialog (2026-10-04) and a script's way of naming a campaign.
    `many` says whether the answer is a list: always for a list given,
    and for an asked batch of more than one, as it always was.
    """
    if run is ASK:
        from .gui.ask import for_runs
        runs = for_runs()
        if not runs:
            raise ValueError('no run chosen')
        return runs, True, len(runs) > 1
    if isinstance(run, (str, os.PathLike)):
        return [run], False, False
    runs = list(run)
    if not runs:
        raise ValueError('no run given')
    return runs, False, True


def _one_call_reports(run: Any, path: Any, geometry: Any,
                      unit_system: Any, kind: str,
                      work_up: Any, marking: str | None = None) -> Any:
    """What every one-call report function does around its workup.

    The asking, the batch and the destination, in one place because
    they are one rule (PRINCIPLES.md, 9): a run left out is asked for,
    several may be chosen or given as a list, the geometry is asked
    for only when the run was, and `path` may be a folder each report
    lands in under its run's own name.
    """
    runs, asked, many = _runs_of(run)
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
        report = project.generate_report(kind, name='Report', marking=marking)
        written.append(project.export_report(
            report, _report_path(one, path), unit_system))
    return written if many else written[0]



def mixed_run(run: str | os.PathLike, per_octave: int | None = None, *,
              last: float | None = None,
              geometry: str | os.PathLike | None = None,
              length_unit: str | None = None,
              photos: Any = None) -> Project:
    """Work a Rattlesnake random run with a sine sweep under it up into a
    project, both halves.

        project = visualdynamics.mixed_run('run.nc4')

    This does everything `random_vibration_run` does (imports the run,
    averages and bands the PSDs, bands the specification, computes the
    coherence, and brings in the geometry and photographs), and then the
    sine half: each tone's level is extracted from the same recording
    against the sweep specification the file carries. The run records
    that it is both, so the project's type is set to Random and Sine. A
    run with no sine sweep specification raises ValueError instead of
    being worked up as half of what was asked for; use
    `random_vibration_run` for it. The keyword arguments are the same
    as `random_vibration_run`'s.

    Parameters
    ----------
    run : str or os.PathLike
        The controller's `.nc4`.
    per_octave : int, optional
        Bands per octave for the banded PSD and specification. Defaults
        to 6.
    last : float, optional
        Import only the last `last` seconds of the run's streams; a
        shorter run is imported whole.
    geometry : str or os.PathLike, optional
        A geometry file to import and link to the run.
    length_unit : str, optional
        The geometry's length unit, for a file that does not say.
    photos : str, os.PathLike or sequence of them, optional
        A folder of photographs, one photograph, or several in order.

    Returns
    -------
    Project
        The worked-up project, of type Random and Sine.
    """
    project = random_vibration_run(run, per_octave, last=last,
                                   geometry=geometry,
                                   length_unit=length_unit, photos=photos)
    from .core.sine import SineSweepSpecification

    if not any(isinstance(obj, SineSweepSpecification)
               for _name, obj in project.items()):
        raise ValueError(
            f'{run} holds no sine sweep specification: it is not a random '
            'and sine run — random_vibration_run is the call for it')
    project.extract_sine(project.time_history)
    return project


def sine_run(run: str | os.PathLike, *,
             last: float | None = None,
             geometry: str | os.PathLike | None = None,
             length_unit: str | None = None,
             photos: Any = None) -> Project:
    """Work a Rattlesnake sine sweep run up into a project.

        project = visualdynamics.sine_run('sweep.nc4')

    The run is imported, each tone's level is extracted from the
    control channels against the sweep specification the file carries,
    and the geometry and photographs are imported and linked the way
    `random_vibration_run` does it. The run records that it is a sine
    sweep, so the project's type is set to Sine Sweep. A run with no
    sine sweep specification, or no time data, raises ValueError; use
    `random_vibration_run` or `mixed_run` for a run without a sweep. The
    keyword arguments are `random_vibration_run`'s except `per_octave`,
    which a sweep does not use.

    Parameters
    ----------
    run : str or os.PathLike
        The controller's `.nc4`.
    last : float, optional
        Import only the last `last` seconds of the run's streams; a
        shorter run is imported whole.
    geometry : str or os.PathLike, optional
        A geometry file to import and link to the run.
    length_unit : str, optional
        The geometry's length unit, for a file that does not say.
    photos : str, os.PathLike or sequence of them, optional
        A folder of photographs, one photograph, or several in order.

    Returns
    -------
    Project
        The worked-up project, of type Sine Sweep.
    """
    project = Project()
    project.import_file(run, **(_last_window(run, last) if last is not None
                                else {}))
    from .core.sine import SineSweepSpecification

    if not any(isinstance(obj, SineSweepSpecification)
               for _name, obj in project.items()):
        raise ValueError(
            f'{run} holds no sine sweep specification: it is not a sine '
            'sweep run — random_vibration_run or mixed_run is the call for it')
    history = next((name for name, obj in project.items()
                    if isinstance(obj, TimeHistory)), None)
    if history is None:
        raise ValueError(f'{run} holds no time data to work up')
    project.extract_sine(history)
    extras: list[str] = []
    if geometry is not None:
        options = {} if length_unit is None else {'length_unit': length_unit}
        extras += project.import_file(geometry, **options)
    if photos is not None:
        extras.append(project.add('Photos', _photos_from(photos)))
    if extras:
        project.link(history, *extras)
    return project


def sine_report(run: Any = ASK, path: str | os.PathLike | None = None, *,
                last: float | None = None,
                geometry: Any = None,
                photos: Any = None,
                unit_system: Any = None,
                marking: str | None = None) -> Any:
    """Read a Rattlesnake sine sweep run and write an HTML report.

        visualdynamics.sine_report('sweep.nc4', 'report.html')
        visualdynamics.sine_report()                   # ask for both
        visualdynamics.sine_report('sweep.nc4', 'reports/')

    This is `random_vibration_report` for a sweep: a run left out is
    asked for, the batch and folder rules are the same, the project is
    worked up by `sine_run`, and the Sine Sweep report is written as one
    self-contained HTML file. Returns the path written, or the list of
    them when several runs were chosen. `marking` is the banner across
    every page of the report, 'UNCLASSIFIED' when omitted.

    Parameters
    ----------
    run : str, os.PathLike, sequence of them, or `visualdynamics.ASK`
        The controller's `.nc4`, or a list of them for a batch. When
        omitted, a file dialog asks for as many as wanted.
    path : str, os.PathLike or None, optional
        The file to write, or a folder (one that exists, or a name ending
        in a separator) the report is written into under the run's own
        name. Beside the run when omitted. Several runs need a folder or
        None.
    last : float, optional
        Import only the last `last` seconds of the run; a shorter run is
        imported whole.
    geometry : str, os.PathLike, `visualdynamics.ASK` or None, optional
        A geometry file to import and link to the run. Asked for once for
        the whole batch when the run was asked for, with Cancel meaning
        none; `ASK` asks even when the run was given.
    photos : str, os.PathLike or sequence of them, optional
        A folder of photographs, one photograph, or several in order.
    unit_system : UnitSystem, optional
        The units the report is written in. Defaults to the package
        default, 'in-slinch-lbf-s (g)'.
    marking : str, optional
        The banner across the top and bottom of every page.
        'UNCLASSIFIED' when omitted.

    Returns
    -------
    str or list of str
        The path written, or a list of them when several runs were
        chosen or a list of runs was given.
    """
    # Brandon, 2026-10-02: the batch tool stopped at a run that was sine
    # alone, and a batch wanted its own banner (`marking`, on every
    # one-call report since).
    return _one_call_reports(
        run, path, geometry, unit_system, 'sine',
        lambda one, geo: sine_run(one, last=last, geometry=geo, photos=photos),
        marking=marking)


#: the one-call report each kind of run gets, by the project type the
#: run declares: (template, workup, report function)
_REPORTS_BY_TYPE = {
    'Random Vibration': 'random', 'Random and Sine': 'mixed',
    'Sine Sweep': 'sine', 'System ID': 'sysid'}


def report_kind(run: str | os.PathLike) -> str:
    """Which one-call reports a Rattlesnake run gets: 'random', 'mixed',
    'sine' or 'sysid', from the project type the file records.

    'mixed' is a random run with a sweep under it, which gets both a
    random report and a sine report (`run_report`). 'sysid' is also
    given for a streamed save shaped like a system identification (two
    streams, a quiet one and then a loud one); the app asks in that
    case, and this function chooses 'sysid'. A run of a type with no
    one-call report, such as a modal or transient run, raises
    ValueError.

    Parameters
    ----------
    run : str or os.PathLike
        The controller's `.nc4`.

    Returns
    -------
    str
        'random', 'mixed', 'sine' or 'sysid'.
    """
    # Brandon, 2026-10-02: a batch defaults to the likeliest reading.
    from .io.rattlesnake import project_type, streamed_sysid_candidate

    declared = project_type(run)
    if declared == 'System ID' or streamed_sysid_candidate(run):
        return 'sysid'
    kind = _REPORTS_BY_TYPE.get(declared or '')
    if kind is None:
        raise ValueError(
            f'{run} is {declared or "of no type visualdynamics reports on"}: '
            'no one-call report exists for it — a project worked up in the '
            'window can still generate its report')
    return kind


#: the reports `run_report` can be told a run gets, by `kinds`
REPORT_KINDS = ('random', 'mixed', 'sine', 'sysid')


def run_report(run: Any = ASK, path: str | os.PathLike | None = None, *,
           last: Any = None,
           kinds: Any = None,
           geometry: Any = None,
           photos: Any = None,
           per_octave: int | None = None,
           unit_system: Any = None,
           marking: str | None = None,
           progress: Any = None) -> Any:
    """Read a Rattlesnake run and write the report its type calls for.

        visualdynamics.run_report('run.nc4', 'report.html')
        visualdynamics.run_report()               # ask for the runs and the geometry
        visualdynamics.run_report('run.nc4', 'reports/')
        visualdynamics.run_report(['a.nc4', 'b.nc4'], 'reports/',
                                  geometry='article.stp')

    This one-call report reads the run's own type (`report_kind`) and
    writes that report. A random run gets `random_vibration_report`'s
    report, a sweep `sine_report`'s, a system identification
    `system_id_report`'s, and a random and sine run gets both a random
    and a sine report, as `<name>_random.html` and `<name>_sine.html`.
    For that run the random report reads only the last `last` seconds
    when they are given, and the sine report always reads the whole
    run, since a sweep cut short loses tones. A streamed save that
    looks like a system identification (two streams, quiet and then
    loud) is treated as one without asking. The function is named for
    what it takes, since `visualdynamics.report` is the rendering
    package.

    The asking, the batch and the folder follow the same rules as the
    typed functions, and a batch may mix kinds. The keyword arguments
    are the union of theirs, and `per_octave` is used only for the
    random reports.

    Every run's kind is read before any report is written, so a batch
    holding a run with no one-call report raises ValueError before
    writing anything instead of stopping partway through. `kinds`
    overrides the reading run by run, for example for a restarted
    random run that was taken for a system identification or a file
    that records no type, and `last` can be given run by run too.
    `progress(done, total)` is called after each run, which is what the
    app's progress bar and Cancel use; a run is the smallest step, so a
    cancel takes effect between runs.

    Parameters
    ----------
    run : str, os.PathLike, sequence of them, or `visualdynamics.ASK`
        The controller's `.nc4`, or a list of them for a batch. When
        omitted, a file dialog asks for as many as wanted.
    path : str, os.PathLike or None, optional
        The file to write, or a folder (one that exists, or a name ending
        in a separator) the report is written into under the run's own
        name. Beside the run when omitted. Several runs need a folder or
        None.
    last : float or Mapping, optional
        Import only the last `last` seconds of the run; a shorter run is
        imported whole. For a random and sine run it applies to the
        random report only, and the sine report reads the whole run. A
        mapping gives each run its own value, by path; a run it leaves
        out, or maps to None, is read whole.
    kinds : Mapping, optional
        The report each run gets, by path: 'random', 'mixed' (a random
        and a sine report), 'sine' or 'sysid'. Any other value raises
        ValueError. A run it leaves out gets the report its file
        records (`report_kind`).
    geometry : str, os.PathLike, `visualdynamics.ASK` or None, optional
        A geometry file to import and link to the run. Asked for once for
        the whole batch when the run was asked for, with Cancel meaning
        none; `ASK` asks even when the run was given.
    photos : str, os.PathLike or sequence of them, optional
        A folder of photographs, one photograph, or several in order.
    per_octave : int, optional
        Bands per octave for the banded PSD and specification. Defaults
        to 6. Only the random reports use it.
    unit_system : UnitSystem, optional
        The units the report is written in. Defaults to the package
        default, 'in-slinch-lbf-s (g)'.
    marking : str, optional
        The banner across the top and bottom of every page.
        'UNCLASSIFIED' when omitted.
    progress : callable, optional
        Called as ``progress(done, total)`` before the first run and
        after each one. It may raise to stop the batch between runs.

    Returns
    -------
    str or list of str
        The path written, or a list of them when several runs were
        chosen, a list of runs was given, or a run was random and sine
        (two reports).
    """
    # The combined random-and-sine report was retired on 2026-10-05.
    # `kinds` is the per-run choice in Reports from Runs (2026-10-06),
    # and `progress` the hook its bar and Cancel use (2026-10-04).
    runs, asked, many = _runs_of(run)
    chosen = {str(key): value for key, value in (kinds or {}).items()}
    wrong = sorted({str(value) for value in chosen.values()
                    if value not in REPORT_KINDS})
    if wrong:
        raise ValueError(f'no report is called {", ".join(wrong)}: '
                         f'the kinds are {", ".join(REPORT_KINDS)}')
    kinds = [chosen.get(str(one)) or report_kind(one) for one in runs]

    def seconds(one):
        # one number for the batch, or each run's own; read whole when
        # a mapping says nothing about it
        if isinstance(last, Mapping):
            return last.get(str(one))
        return last

    if geometry is ASK or (asked and geometry is None):
        from .gui.ask import for_geometry
        geometry = for_geometry()
    if len(runs) > 1 and path is not None and not (
            os.path.isdir(os.path.expanduser(str(path)))
            or str(path).endswith((os.sep, '/'))):
        raise ValueError(
            f'{len(runs)} runs cannot be written to one file '
            f'{str(path)!r} — give a folder, or no path at all')
    workups = {
        'random': lambda one, geo: random_vibration_run(
            one, per_octave, last=seconds(one), geometry=geo, photos=photos),
        'sine': lambda one, geo: sine_run(one, last=seconds(one),
                                          geometry=geo, photos=photos),
        'sysid': lambda one, geo: system_id_run(one, last=seconds(one),
                                                geometry=geo, photos=photos),
    }
    written = []
    if progress is not None:
        progress(0, len(runs))
    for done, (one, kind) in enumerate(zip(runs, kinds), start=1):
        if kind == 'mixed':
            # two reports, each read the way its own test is: the
            # random from the last `last` seconds when given, the sweep
            # from the whole run, since a sweep cut short loses tones
            # (Brandon, 2026-10-05, when the combined report retired)
            stem, suffix = os.path.splitext(_report_path(one, path))
            suffix = suffix or '.html'
            written.append(_one_call_reports(
                one, f'{stem}_random{suffix}', geometry, unit_system,
                'random', workups['random'], marking=marking))
            written.append(_one_call_reports(
                one, f'{stem}_sine{suffix}', geometry, unit_system, 'sine',
                lambda whole, geo: sine_run(whole, geometry=geo,
                                            photos=photos),
                marking=marking))
        else:
            written.append(_one_call_reports(one, path, geometry,
                                             unit_system, kind,
                                             workups[kind], marking=marking))
        if progress is not None:
            progress(done, len(runs))
    return written if many or len(written) > 1 else written[0]


def system_id_report(run: Any = ASK,
                     path: str | os.PathLike | None = None, *,
                     last: float | None = None,
                     geometry: Any = None,
                     photos: Any = None,
                     unit_system: Any = None,
                     marking: str | None = None) -> Any:
    """Read a Rattlesnake system identification and write an HTML
    report.

        visualdynamics.system_id_report('sysid.nc4', 'sysid.html')
        visualdynamics.system_id_report()            # ask for both

    This is `system_id_run` followed by the System ID report: the
    measured FRFs, the coherence map, and the signal-to-noise ratio of
    the measurement at each line and for each channel. Where the ratio
    falls to zero decibels, the measurement is of the ambient noise,
    not of the structure.

    Asking, batches and folders work as in `random_vibration_report`.
    When the run is left out it is asked for, and as many runs may be
    chosen as there are reports wanted; the geometry is asked for once
    for the whole batch when the run was asked for, with Cancel meaning
    none; `visualdynamics.ASK` forces either question; and `path` may
    be the file to write or the folder to write into.

    Parameters
    ----------
    run : str, os.PathLike, sequence of them, or `visualdynamics.ASK`
        The controller's `.nc4`, or a list of them for a batch. When
        omitted, a file dialog asks for as many as wanted.
    path : str, os.PathLike or None, optional
        The file to write, or a folder (one that exists, or a name ending
        in a separator) the report is written into under the run's own
        name. Beside the run when omitted. Several runs need a folder or
        None.
    last : float, optional
        Import only the last `last` seconds of the run; a shorter run is
        imported whole.
    geometry : str, os.PathLike, `visualdynamics.ASK` or None, optional
        A geometry file to import and link to the run. Asked for once for
        the whole batch when the run was asked for, with Cancel meaning
        none; `ASK` asks even when the run was given.
    photos : str, os.PathLike or sequence of them, optional
        A folder of photographs, one photograph, or several in order.
    unit_system : UnitSystem, optional
        The units the report is written in. Defaults to the package
        default, 'in-slinch-lbf-s (g)'.
    marking : str, optional
        The banner across the top and bottom of every page.
        'UNCLASSIFIED' when omitted.

    Returns
    -------
    str or list of str
        The path written, or a list of them when several runs were
        chosen or a list of runs was given.
    """
    return _one_call_reports(
        run, path, geometry, unit_system, 'sysid',
        lambda one, geo: system_id_run(one, last=last, geometry=geo,
                                       photos=photos),
        marking=marking)


def random_vibration_report(run: Any = ASK,
                            path: str | os.PathLike | None = None, *,
                            last: float | None = None,
                            geometry: Any = None,
                            photos: Any = None,
                            per_octave: int | None = None,
                            unit_system: Any = None,
                            marking: str | None = None) -> Any:
    """Read a Rattlesnake random vibration run and write an HTML report.

        visualdynamics.random_vibration_report('run.nc4', 'report.html')
        visualdynamics.random_vibration_report(
            'run.nc4', 'report.html', last=100.0,
            geometry='article.stp', photos='setup_photos/')
        visualdynamics.random_vibration_report()          # ask for both
        visualdynamics.random_vibration_report('run.nc4', 'reports/')

    Called with no run, it opens a file dialog and takes as many runs as
    are chosen, writes one report for each and returns the list. A run
    chosen that way is asked about its geometry too, with Cancel meaning
    none. `visualdynamics.ASK` in either place forces the question, so a
    script with a run in hand can still be asked for the geometry. A
    run given without a `geometry` argument means no geometry, and no
    dialog opens.

    `path` may be the file to write, or a folder the report is written
    into under the run's own name with an `.html` extension. Without a
    path the report is written beside the run. Several runs need a
    folder or no path; a single file name raises ValueError.

    The whole workflow runs in one call: import the run (or only its
    last `last` seconds; a shorter run is imported whole), detect the
    averaging, compute the PSDs, put them and the specification on
    octave bands, compute the multiple coherence, bring in the geometry
    and the photographs when they are given, generate the Random
    Vibration report, and write it as one self-contained HTML file. The
    keyword arguments are `random_vibration_run`'s except `length_unit`,
    plus `unit_system` and `marking` for the report.

    It is `random_vibration_run` followed by `generate_report` and
    `export_report`. Use those instead when the project is wanted
    afterwards, to write the test summary or to save it as `.vdyn`.

    Parameters
    ----------
    run : str, os.PathLike, sequence of them, or `visualdynamics.ASK`
        The controller's `.nc4`, or a list of them for a batch. When
        omitted, a file dialog asks for as many as wanted.
    path : str, os.PathLike or None, optional
        The file to write, or a folder (one that exists, or a name ending
        in a separator) the report is written into under the run's own
        name. Beside the run when omitted. Several runs need a folder or
        None.
    last : float, optional
        Import only the last `last` seconds of the run; a shorter run is
        imported whole.
    geometry : str, os.PathLike, `visualdynamics.ASK` or None, optional
        A geometry file to import and link to the run. Asked for once for
        the whole batch when the run was asked for, with Cancel meaning
        none; `ASK` asks even when the run was given.
    photos : str, os.PathLike or sequence of them, optional
        A folder of photographs, one photograph, or several in order.
    per_octave : int, optional
        Bands per octave for the banded PSD and specification. Defaults
        to 6.
    unit_system : UnitSystem, optional
        The units the report is written in. Defaults to the package
        default, 'in-slinch-lbf-s (g)'.
    marking : str, optional
        The banner across the top and bottom of every page.
        'UNCLASSIFIED' when omitted.

    Returns
    -------
    str or list of str
        The path written, or a list of them when several runs were
        chosen or a list of runs was given.
    """
    # The whole workflow in one call, with nothing to click (Brandon,
    # 2026-09-19).
    #
    # no length unit reaches the workup: the report draws the geometry
    # as a shape and never states a coordinate or a scale, so declaring
    # what the file's numbers meant changed one label and nothing else
    # (Brandon, 2026-09-22). The `_run` calls still take it, for a
    # project that lives on.
    return _one_call_reports(
        run, path, geometry, unit_system, 'random',
        lambda one, geo: random_vibration_run(one, per_octave, last=last,
                                              geometry=geo, photos=photos),
        marking=marking)
