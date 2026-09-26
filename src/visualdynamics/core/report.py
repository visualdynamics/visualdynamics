"""Reports: an ordered story told from a project's objects.

A Report is a list of blocks — text the user wrote, plots, animated
geometry scenes, tables — each naming the object it draws from. It holds
*references*, never data. A reference is a literal object name, or a
symbolic selector like '@basis:Frf' resolved against the project's link
groups at render time (`resolve_binding`) — which is what makes one
saved report a template for the next project: nobody has to name their
objects anything in particular. An unresolvable block sits unbound in
the builder until pointed at a compatible object, and an unbound block
simply does not render.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

BLOCK_KINDS = ('text', 'plot', 'scene', 'table', 'photo', 'pairs',
               'overlay', 'bars', 'verdict')


class Report:
    """An ordered list of blocks, each a plain dict.

    Block shapes:
      {'kind': 'text',  'text': str}
      {'kind': 'plot',  'source': name, 'mode': curves|cmif|mac|map,
                        'select': ''|drive|dim:<quantity>,
                        'component': magnitude|real|imag|phase,
                        'caption': str}
      {'kind': 'scene', 'geometry': name, 'shapes': name-or-'',
                        'dofs': quantity, 'dofs_source': name,
                        'caption': str}
        ('dofs' + 'dofs_source' make a DOFs scene: labeled arrows at
        every DOF that one data object measures as that quantity)
      {'kind': 'table', 'source': name, 'caption': str}
      {'kind': 'photo', 'source': Photos-name, 'photo': photo-name,
                        'caption': str}
      {'kind': 'pairs', 'source': MatchedModes-name, 'caption': str}
        (the matched-modes table: the committed matches, both sets'
        parameters side by side)
      {'kind': 'overlay', 'source': MatchedModes-name, 'caption': str}
        (the matched pairs animated over each other: both sets'
        geometries in one scene, resolved through the object's own
        geometry references, each pair phase-aligned)
      {'kind': 'bars', 'source': name, 'mode': kurtosis|srs|sine,
                       'caption': str}
        (a bar reading: kurtosis per channel, SRS levels against the
        target, or sine levels against their tones)
      {'kind': 'verdict', 'source': Specification-name,
                          'measured': Psd-name, 'level_label': str}
        (the pass/fail box: whether the run passed, read off the
        octave-band comparison of `measured` against `source` —
        `compliance.verdict` — with the test level it was judged at
        beside it, under `level_label`, 'Test level' when absent)

    'select' filters a curves plot to the records the GUI's own filters
    would pick: 'drive' is the response==reference diagonal of an FRF,
    and 'dim:<dimension>' keeps only records of that quantity — how a
    time history splits into Time (Acceleration) and Time (Force).
    'map' is the coherence map — frequency across, channel down,
    pinned 0..1.
    """

    def __init__(self, title: str = 'Report',
                 blocks: Sequence[Mapping[str, Any]] | None = None,
                 marking: str = 'UNCLASSIFIED',
                 marking_color: str = 'ink') -> None:
        self.title: str = str(title)
        #: the classification marking, carried as a banner across the
        #: top and bottom of the rendered document — one value, two
        #: places, because markings that can disagree are markings that
        #: will. 'UNCLASSIFIED' by default: work that carries markings
        #: is the intended use, and a document that says its level
        #: outright beats one that says nothing. Empty means unmarked —
        #: no banner at all.
        self.marking: str = str(marking)
        #: 'ink' draws the banner in the page's own black-or-white;
        #: 'red' is the conventional color for the levels that demand
        #: attention
        self.marking_color: str = (marking_color
                                   if marking_color in ('ink', 'red')
                                   else 'ink')
        self.blocks: list[dict[str, Any]] = [
            dict(block) for block in (blocks or [])]
        for block in self.blocks:
            if block.get('kind') not in BLOCK_KINDS:
                raise ValueError(f"unknown block kind {block.get('kind')!r}")

    def __repr__(self) -> str:
        count = len(self.blocks)
        return (f'Report({self.title!r}, {count} '
                f'block{"s" * (count != 1)})')

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def add(self, block: Mapping[str, Any], at: int | None = None) -> None:
        if block.get('kind') not in BLOCK_KINDS:
            raise ValueError(f"unknown block kind {block.get('kind')!r}")
        if at is None:
            self.blocks.append(dict(block))
        else:
            self.blocks.insert(int(at), dict(block))

    def remove(self, indices: Sequence[int]) -> None:
        doomed = {int(i) for i in indices}
        self.blocks = [block for i, block in enumerate(self.blocks)
                       if i not in doomed]

    def move(self, rows: Sequence[int], to: int) -> None:
        """Move the blocks at `rows` so the first lands at index `to` —
        the drop half of drag-to-reorder."""
        rows = sorted({int(r) for r in rows})
        lifted = [self.blocks[r] for r in rows]
        rest = [block for i, block in enumerate(self.blocks)
                if i not in set(rows)]
        to = int(to) - sum(1 for r in rows if r < to)
        to = max(0, min(to, len(rest)))
        self.blocks = rest[:to] + lifted + rest[to:]

    def unbound(self, objects: Mapping[str, Any],
                links: Sequence[Mapping[str, Any]] | None = None) -> list[int]:
        """Block indices whose references the project cannot resolve —
        a missing name, or a symbolic selector with nothing to pick."""
        out = []
        for i, block in enumerate(self.blocks):
            needed = [block.get(key)
                      for key in ('source', 'geometry', 'dofs_source',
                                  'shapes')
                      if block.get(key)]
            if any(resolve_binding(name, objects, links) not in objects
                   for name in needed):
                out.append(i)
        return out


def _wanted_classes(block):
    """The classes a block of this kind draws from, or None."""
    from .channel_table import ChannelTable
    from .data import DataArray, Frf, TimeHistory, _CoherenceBase
    from .geometry import Geometry
    from .shapes import ShapeSet

    kind = block.get('kind')
    if kind == 'plot':
        mode = block.get('mode')
        select = block.get('select', '')
        if mode == 'mac':
            return ShapeSet
        if mode == 'map':
            return _CoherenceBase
        if mode == 'cmif' or select == 'drive':
            return Frf
        if select.startswith('dim:'):
            from .data import Psd, Spectrum
            return (TimeHistory, Spectrum, Psd)
        return DataArray
    if kind == 'scene':
        return Geometry
    if kind == 'table':
        return (ShapeSet, ChannelTable)
    if kind == 'photo':
        from .photos import Photos
        return Photos
    if kind == 'pairs':
        from .matches import MatchedModes
        return MatchedModes
    if kind == 'overlay':
        from .matches import MatchedModes
        return MatchedModes
    if kind == 'verdict':
        from .data import Specification
        return Specification
    return None


def source_options(block: Mapping[str, Any], objects: Mapping[str, Any]) -> list[str]:
    """Object names a block of this kind could draw from."""
    wanted = _wanted_classes(block)
    if wanted is None:
        return []
    return [name for name, obj in objects.items()
            if isinstance(obj, wanted)]


#: the bindings that name a *banded* flavor of a class: an octave-band
#: PSD is a Psd with `bandwidth` set and an octave-band specification a
#: Specification with it, and a report that compares the banded
#: measurement against the banded requirement has to be able to ask
#: for each by name (Brandon, 2026-09-18). The plain token never
#: answers with a banded object — a banded flavor is claimed by its own
#: token, the way a bound is claimed by its own class.
BANDED_TOKENS = {'OctavePsd': 'Psd', 'OctaveSpecification': 'Specification'}


def binding_tokens() -> dict[str, tuple[type, bool | None]]:
    """{token: (class, banded)} — every symbolic token: the plain
    types (banded None, meaning not banded), and the banded flavors."""
    types = binding_types()
    out: dict[str, tuple[type, bool | None]] = {
        token: (cls, None) for token, cls in types.items()}
    for token, plain in BANDED_TOKENS.items():
        out[token] = (types[plain], True)
    return out


def is_banded(obj: Any) -> bool:
    """Does this object live on bands — an octave-band PSD or
    specification rather than one on lines?"""
    return getattr(obj, 'bandwidth', None) is not None


def binding_types() -> dict[str, type]:
    """{token: class} — the types a symbolic binding can name."""
    from .channel_table import ChannelTable
    from .data import (
        Frf,
        MultipleCoherence,
        Psd,
        ShockSpecification,
        Specification,
        Spectrum,
        Srs,
        TimeHistory,
        TransientSpecification,
        _CoherenceBase,
    )
    from .geometry import Geometry
    from .matches import MatchedModes
    from .photos import Photos
    from .shapes import ShapeSet
    from .sine import SineLevel, SineLevelSet, SineSweepSpecification

    return {'Geometry': Geometry, 'Photos': Photos,
            'ChannelTable': ChannelTable, 'TimeHistory': TimeHistory,
            'Spectrum': Spectrum, 'Psd': Psd, 'Frf': Frf,
            'Specification': Specification,
            'Srs': Srs, 'ShockSpecification': ShockSpecification,
            'TransientSpecification': TransientSpecification,
            'SineSweepSpecification': SineSweepSpecification,
            'SineLevel': SineLevel,
            'SineLevelSet': SineLevelSet,
            'Coherence': _CoherenceBase,
            'MultipleCoherence': MultipleCoherence,
            'ShapeSet': ShapeSet, 'MatchedModes': MatchedModes}


def resolve_binding(value: str | None, objects: Mapping[str, Any],
                    links: Sequence[Mapping[str, Any]] | None = None) -> str | None:
    """The object name a binding field means.

    A literal name passes through untouched. A symbolic selector —
    '@basis:Frf', '@other:ShapeSet', '@any:MatchedModes' — resolves
    against the project at render time, so a report never depends on
    what anyone named their objects: 'basis' is the first object of
    the type in the Basis link group, 'other' the first outside it,
    'any' the first anywhere. Without a Basis declared, 'basis'
    degrades to the first of the type and 'other' to the second, so a
    small unlinked project still renders. Returns the name, or None
    when nothing fits.
    """
    value = str(value or '')
    if not value.startswith('@'):
        return value or None
    role, _, token = value[1:].partition(':')
    cls, banded = binding_tokens().get(token, (None, None))
    if cls is None or role not in ('basis', 'other', 'any'):
        return None
    # A token never returns what a more specific token would claim —
    # with one distinction, learned one half at a time. A *bound* never
    # answers for the measurement it bounds: Specification subclasses
    # Psd, so '@basis:Psd' was answering with the specification, and a
    # report comparing the control PSDs against their specification
    # quietly compared it against itself, drew one curve where there
    # should be two, and reported a perfect match whatever the run had
    # done. But a *flavor* of the same measurement is only preferred
    # against, not refused: multiple coherence is coherence with the
    # references summed over, the skeleton's one Coherence slot takes
    # either kind, and refusing it cost a Rattlesnake test's report its
    # coherence figure with no word said — every figure after it
    # quietly renumbered.
    from .data import Bounded, TransientSpecification

    narrower = [other for other in binding_types().values()
                if other is not cls and issubclass(other, cls)]
    bounds = tuple(other for other in narrower
                   if issubclass(other, (Bounded, TransientSpecification)))
    flavors = tuple(other for other in narrower if other not in bounds)
    # ordered, not filtered: the plain kind first, flavors after, so
    # every role below prefers exact and degrades to the flavor
    # and a banded flavor answers only to its own token: '@basis:Psd'
    # is the narrowband PSD, '@basis:OctavePsd' the banded one
    names = sorted(
        (name for name, obj in objects.items()
         if isinstance(obj, cls)
         and not any(isinstance(obj, kind) for kind in bounds)
         and is_banded(obj) == bool(banded)),
        key=lambda name: isinstance(objects[name], flavors))
    basis = next((set(group['members']) for group in (links or [])
                  if group.get('role') == 'Basis'), None)
    if role == 'basis':
        if basis is None:
            return names[0] if names else None
        inside = [name for name in names if name in basis]
        return inside[0] if inside else None
    if role == 'other':
        if basis is None:
            return names[1] if len(names) > 1 else None
        outside = [name for name in names if name not in basis]
        return outside[0] if outside else None
    return names[0] if names else None


def binding_label(value: str) -> str:
    """A human reading of a binding: 'Basis FRF (auto)' for
    '@basis:Frf'; a literal name reads as itself."""
    value = str(value)
    if not value.startswith('@'):
        return value
    role, _, token = value[1:].partition(':')
    spaced = {'Frf': 'FRF', 'Psd': 'PSD', 'OctavePsd': 'Octave PSD'}.get(
        token, re.sub(r'(?<!^)(?=[A-Z])', ' ', token))
    return f'{role.title()} {spaced} (auto)'


def symbolic_options(block: Mapping[str, Any]) -> list[tuple[str, str]]:
    """[(selector, label)] this block could bind instead of a name —
    how a report stays independent of what objects are called."""
    wanted = _wanted_classes(block)
    if wanted is None:
        return []
    classes = wanted if isinstance(wanted, tuple) else (wanted,)
    from .data import DataArray
    if any(cls is DataArray for cls in classes):
        # a plain curves plot could draw from anything; offer the
        # useful few rather than every subclass
        tokens = ['TimeHistory', 'Spectrum', 'Psd', 'OctavePsd', 'Frf']
    else:
        tokens = [token for token, (cls, _banded) in binding_tokens().items()
                  if any(cls is wanted_cls for wanted_cls in classes)]
    return [(f'@{role}:{token}', binding_label(f'@{role}:{token}'))
            for token in tokens for role in ('basis', 'other')]


def photo_options(block: Mapping[str, Any], objects: Mapping[str, Any],
                  links: Sequence[Mapping[str, Any]] | None = None) -> list[str]:
    """The photo names a photo block's bound Photos object holds."""
    from .photos import Photos

    owner = objects.get(
        resolve_binding(block.get('source'), objects, links) or '')
    return list(owner.names) if isinstance(owner, Photos) else []


def shape_options(objects: Mapping[str, Any]) -> list[str]:
    from .shapes import ShapeSet

    return [name for name, obj in objects.items()
            if isinstance(obj, ShapeSet)]


def base_quantity(dim: str) -> str:
    """The quantity a record's dimension names: 'acceleration' whether
    the record holds accelerations or their square-per-Hz."""
    base = dim.partition('/')[0]
    return base.removesuffix('**2')


def record_dimensions(obj: Any) -> set[str]:
    """The quantities one data array's records hold — unknowns and
    cross terms (a CPSD's acceleration*force) dropped."""
    from ..units import UNKNOWN

    dims = {base_quantity(obj.known_dim(i))
            for i in range(obj.num_records)}
    dims.discard(UNKNOWN)
    return {dim for dim in dims if '*' not in dim}


def _held_dimensions(objects: Mapping[str, Any],
                     wanted: type | tuple[type, ...]) -> list[str]:
    dims = set()
    for obj in objects.values():
        if isinstance(obj, wanted):
            dims.update(record_dimensions(obj))
    return sorted(dims)


def time_dimensions(objects: Mapping[str, Any]) -> list[str]:
    """The quantities the project's time data actually holds, sorted —
    what the insert menu's Time (…) options are made of."""
    from .data import TimeHistory

    return _held_dimensions(objects, TimeHistory)


def spectrum_dimensions(objects: Mapping[str, Any]) -> list[str]:
    """The quantities the project's spectra hold — the Spectra (…)
    options, exactly as the time ones."""
    from .data import Spectrum

    return _held_dimensions(objects, Spectrum)


def psd_dimensions(objects: Mapping[str, Any]) -> list[str]:
    """The quantities the project's PSDs hold — the PSD (…) options."""
    from .data import Psd

    return _held_dimensions(objects, Psd)


def series_dof_quantities(
        series: Sequence[tuple[str, Any, Sequence[int] | None]]
        ) -> list[str]:
    """Every quantity the selected data measures at some DOF, sorted.

    `series` is [(name, data, records)] with records None for the whole
    object — the same shape the GUI's selection speaks. References
    count: an FRF per unit force names its force DOFs even when no
    force channel was kept."""
    from ..units import UNKNOWN

    dims = set()
    for _name, data, records in series:
        wanted = records if records is not None else range(data.num_records)
        for i in wanted:
            dim = data.known_dim(i)
            dims.add(base_quantity(dim))
            denominator = dim.partition('/')[2]
            if denominator and data.reference_dof is not None:
                dims.add(denominator)
    dims -= {UNKNOWN, 'frequency', 'dimensionless'}
    return sorted(dim for dim in dims if '*' not in dim)


def series_quantity_dofs(
        series: Sequence[tuple[str, Any, Sequence[int] | None]],
        quantity: str) -> list[str]:
    """Every DOF the selected data measures as `quantity`: response
    DOFs of records holding it, and reference DOFs of records
    referenced to it. Deduped and ordered by node."""
    from .data import parse_dof

    dofs = []
    for _name, data, records in series:
        wanted = records if records is not None else range(data.num_records)
        for i in wanted:
            dim = data.known_dim(i)
            if base_quantity(dim) == quantity:
                dofs.append(data.response_dof[i])
            if (dim.partition('/')[2] == quantity
                    and data.reference_dof is not None):
                dofs.append(data.reference_dof[i])

    def order(dof: str) -> tuple[int, str]:
        node, direction = parse_dof(dof)
        return (node if node is not None else -1, direction)

    return sorted(set(dofs), key=order)


def _whole_project_series(objects: Mapping[str, Any]
                          ) -> list[tuple[str, Any, None]]:
    from .data import DataArray

    return [(name, obj, None) for name, obj in objects.items()
            if isinstance(obj, DataArray)]


def dof_quantities(objects: Mapping[str, Any]) -> list[str]:
    """Every quantity the project measures anywhere — the DOFs (…)
    insert options."""
    return series_dof_quantities(_whole_project_series(objects))


def quantity_dofs(objects: Mapping[str, Any], quantity: str) -> list[str]:
    """Every DOF in the whole project measured as `quantity`."""
    return series_quantity_dofs(_whole_project_series(objects), quantity)


def dofs_source_options(block: Mapping[str, Any], objects: Mapping[str, Any]) -> list[str]:
    """Data objects a DOFs scene could read its arrows from — those
    that actually measure the block's quantity at some DOF."""
    from .data import DataArray

    quantity = block.get('dofs', '')
    return [name for name, obj in objects.items()
            if isinstance(obj, DataArray)
            and quantity in series_dof_quantities([(name, obj, None)])]


def insert_options(objects: Mapping[str, Any]) -> list[tuple[str, str]]:
    """[(kind, label)] for the editor's Result drop-down. The Time
    entries are per quantity, and only for quantities the project's
    time data actually holds."""
    return ([('verdict', 'Pass/Fail Box'),
             ('geometry', 'Geometry'),
             ('modes', 'Mode Shape Animations')]
            + [(f'dofs:{quantity}', f'DOFs ({quantity.title()})')
               for quantity in dof_quantities(objects)]
            + [('photos', 'Photos'),
             ('channel_table', 'Channel Table'),
             ('cmif', 'CMIF')]
            + [(f'time:{dim}', f'Time ({dim.title()})')
               for dim in time_dimensions(objects)]
            + [(f'spectrum:{dim}', f'Spectra ({dim.title()})')
               for dim in spectrum_dimensions(objects)]
            + [(f'psd:{dim}', f'PSD ({dim.title()})')
               for dim in psd_dimensions(objects)]
            + [('coherence', 'Coherence'),
               ('stage', '3-D Stage'),
               ('mac', 'Auto-MAC'),
               ('matches', 'Matched Modes'),
               ('overlay', 'Matched Mode Animations')]
            + [(f'frf_drive:{component}', f'FRF - Drive Point - {label}')
               for component, label in (('magnitude', 'Magnitude'),
                                        ('real', 'Real'),
                                        ('imag', 'Imaginary'),
                                        ('phase', 'Phase'))])


def takes_shapes(block: Mapping[str, Any]) -> bool:
    """May this block name a shape set? Scenes animate them, a CMIF
    overlays their synthesis, a MAC crosses against them, and a drive
    point plot marks their frequencies."""
    return (block.get('kind') in ('scene', 'pairs')
            or (block.get('kind') == 'plot'
                and (block.get('mode') in ('cmif', 'mac')
                     or block.get('select') == 'drive')))


# A **shock** test is controlled to an SRS: a required response spectrum,
# met by producing some transient whose spectrum lands inside the band. A
# **transient** test is controlled to a waveform: this acceleration,
# sample by sample. They are different tests with different targets, and
# calling both 'shock' is what makes the difference hard to see. A
# controller can run the second today; the first it cannot.
# A **random and sine** test is the two run at once on the same shakers
# — a random environment held to its PSD with a sweep riding under it,
# the qualification-with-a-tracked-tone case, and the only way Brandon
# has ever run a sweep. Each half is judged by its own workflow, and the
# type exists so one project shows both halves' slots and one report
# holds both judgments (Brandon, 2026-09-24). Until then a mixed run
# took the random type and carried the sine half beside it.
PROJECT_TYPES = ('Modal Test', 'Random Vibration', 'Transient', 'Shock',
                 'Sine Sweep', 'Random and Sine', 'System ID')
# which template a typed project's generic Generate Report uses
PROJECT_TEMPLATES = {'Modal Test': 'modal', 'Random Vibration': 'random',
                     'Transient': 'transient', 'Shock': 'shock',
                     'Sine Sweep': 'sine', 'Random and Sine': 'mixed',
                     'System ID': 'sysid'}


#: the side of a typed project's skeleton that has no name: every
#: group that is not the Basis. A tag for the tree's sake — the
#: project itself calls that role None.
OTHER_SIDE = 'other'


#: the instrumentation section the transient and random templates share
#: word for word
_INSTRUMENTATION_TEXT = (
    '## Test Article and Instrumentation\n\nThe test geometry '
    '({{figure:Test geometry and measurement locations}}) shows '
    'where the article was instrumented. The excitation degrees '
    'of freedom ({{figure:Excitation degrees of freedom '
    '(force)}}) mark where and in which direction it was '
    'driven; the response degrees of freedom '
    '({{figure:Response degrees of freedom (acceleration)}}) '
    'mark every measured accelerometer direction. The channel '
    'table ({{table:Instrumentation}}) lists the '
    'instrumentation at each location.')


def instrumentation_text(objects: Mapping[str, Any],
                         links: Sequence[Mapping[str, Any]] | None = None
                         ) -> str:
    """The front matter's prose, saying only what the project holds.

    The standing text names the geometry, the DOF scenes and the
    channel table. A run imported without a geometry has none of the
    first three, and a paragraph about "the test geometry" over a
    channel table alone reads as a report with holes in it (Brandon,
    2026-09-19). With a geometry bound the text is the standing one;
    without, it is the channel table's sentence, and the photographs'
    when there are any. An empty project keeps the standing text —
    there it is the template being read.
    """
    if not objects:
        return _INSTRUMENTATION_TEXT
    if resolve_binding('@basis:Geometry', objects, links) in objects:
        return _INSTRUMENTATION_TEXT
    photos = resolve_binding('@basis:Photos', objects, links)
    sentences = []
    if photos in objects and getattr(objects[photos], 'names', None):
        sentences.append('The photographs record the article as it was '
                         'instrumented.')
    sentences.append('The channel table ({{table:Instrumentation}}) '
                     'lists the instrumentation at each location.')
    return ('## Test Article and Instrumentation\n\n'
            + ' '.join(sentences))


def project_expectations(
        project_type: str
        ) -> list[tuple[str, type, str, int, bool, str | None]]:
    """[(label, class, icon key, count, optional, side)] a report of
    this type draws from — the gray slots a typed project shows until
    filled.

    The label is the type's own name and nothing more (Brandon,
    2026-09-02: every slot named for the object it waits for, so a
    skeleton reads the same in every project type); a type that
    expects several of one class shows the same label several times,
    and `count` says which instance satisfies which slot. The side is
    'Basis', `OTHER_SIDE`, or None for a slot on neither side (the
    report, the matched modes). A project knows its Basis and
    everything else is *other*: the other side is a place in the
    tree, not a role — its group carries no name and is never guessed
    into, since a second test is as good a comparison as a model. The
    FEM tag that once named that pair went 2026-09-02. `optional`
    marks best-practice slots a report renders without.

    Every type ends with the report itself. It is the one object the
    others exist to produce, and a project that has everything a report
    needs but no report is not finished — so it is a slot like the rest,
    gray until it is generated. It carries no link group: a report reads
    the whole project and belongs to no one side of it.
    """
    from ..names import display_name

    def raw(project_type):
        from .channel_table import ChannelTable
        from .data import (
            Frf,
            Psd,
            ShockSpecification,
            Specification,
            Srs,
            TimeHistory,
            TransientSpecification,
            _CoherenceBase,
        )
        from .geometry import Geometry
        from .matches import MatchedModes
        from .photos import Photos
        from .shapes import ShapeSet

        if project_type == 'Modal Test':
            return [('Geometry', Geometry, 'Geometry', 1, False,
                     'Basis'),
                    ('Photos', Photos, 'Photos', 1, False, 'Basis'),
                    ('Channel Table', ChannelTable, 'ChannelTable', 1,
                     False, 'Basis'),
                    ('Time Data', TimeHistory, 'TimeHistory', 1, False,
                     'Basis'),
                    ('PSD', Psd, 'Psd', 1, False, 'Basis'),
                    ('FRF', Frf, 'Frf', 1, False, 'Basis'),
                    ('Coherence', _CoherenceBase, 'MultipleCoherence',
                     1, False, 'Basis'),
                    ('Shape Set', ShapeSet, 'ShapeSet', 1, False,
                     'Basis'),
                    # test-analysis correlation is best practice, not
                    # required: the other side's pair to compare against
                    ('Other Geometry', Geometry, 'Geometry', 2, True,
                     OTHER_SIDE),
                    ('Other Shape Set', ShapeSet, 'ShapeSet', 2, True,
                     OTHER_SIDE),
                    # ...and the matched pairs, which sit outside both
                    # groups and after them. A group is one *side* of the
                    # comparison; this object names a set on each side, so
                    # it belongs to neither — the same reasoning as the
                    # report below it. It has to come after the FEM pair
                    # rather than between the two groups, because a group's
                    # rows are drawn as one bracket and have to stay
                    # contiguous.
                    ('Matched Modes', MatchedModes, 'MatchedModes', 1,
                     True, None),
                    ('Report', Report, 'Report', 1, False, None)]
        if project_type == 'Random Vibration':
            return [('Geometry', Geometry, 'Geometry', 1, False,
                     'Basis'),
                    ('Photos', Photos, 'Photos', 1, False, 'Basis'),
                    ('Channel Table', ChannelTable, 'ChannelTable', 1,
                     False, 'Basis'),
                    ('Time Data', TimeHistory, 'TimeHistory', 1, False,
                     'Basis'),
                    ('Specification', Specification, 'Specification',
                     1, False, 'Basis'),
                    # The same requirement on octave bands, limits and
                    # all — `Specification.to_octave` — the *second*
                    # Specification in the group, the way the octave PSD
                    # is the second Psd. The report's octave section
                    # compares the banded measurement against it
                    # (Brandon, 2026-09-18).
                    ('Octave Band Specification', Specification,
                     'OctaveSpecification', 2, False, 'Basis'),
                    ('PSD', Psd, 'Psd', 1, False, 'Basis'),
                    # The sixth-octave view of that PSD, and **required**: a
                    # random vibration report is not finished without it.
                    # The count is 2 because it is the *second* Psd in the
                    # group — `Psd.to_octave` returns a Psd, deliberately,
                    # since banding conserves the area and changes nothing
                    # about what the object is. What tells the two apart is
                    # `bandwidth`, which is also what `object_icon` asks
                    # before drawing bars instead of a filled curve.
                    ('Octave Band PSD', Psd, 'OctavePsd', 2, False,
                     'Basis'),
                    ('Multiple Coherence', _CoherenceBase,
                     'MultipleCoherence', 1, False, 'Basis'),
                    ('Report', Report, 'Report', 1, False, None)]
        if project_type == 'Transient':
            # no coherence slot, for the reason the shock type has none: a
            # transient is one event, and coherence is a statement about a
            # stationary average
            return [('Geometry', Geometry, 'Geometry', 1, False,
                     'Basis'),
                    ('Photos', Photos, 'Photos', 1, False, 'Basis'),
                    ('Channel Table', ChannelTable, 'ChannelTable', 1,
                     False, 'Basis'),
                    ('Time Data', TimeHistory, 'TimeHistory', 1, False,
                     'Basis'),
                    ('Transient Specification', TransientSpecification,
                     'TransientSpecification', 1, False, 'Basis'),
                    # the spectra the report reads the run a second way
                    # with: a target and a measurement. A Specification is
                    # a Psd, so the plain slot asks for the *second* one —
                    # the first is the target's own spectrum and the slot
                    # would otherwise count itself as filled the moment
                    # that arrived. No SRS pair: a shock response spectrum
                    # is how a *shock* test is judged, and reading one into
                    # a replication test blurred the very distinction the
                    # two project types exist to draw.
                    ('PSD Specification', Specification, 'Specification',
                     1, False, 'Basis'),
                    ('PSD', Psd, 'Psd', 2, False, 'Basis'),
                    ('Report', Report, 'Report', 1, False, None)]
        if project_type == 'Sine Sweep':
            # no coherence slot for the same reason the shock type has
            # none only in reverse: coherence is defined for a stationary
            # random average, and a swept tone is neither. The SRS the
            # shock type ends with becomes the sine specification here —
            # the extracted-sweep objects join the list when the arc's
            # Phase D gives them a class.
            from .sine import SineLevelSet, SineSweepSpecification
            return [('Geometry', Geometry, 'Geometry', 1, False,
                     'Basis'),
                    ('Photos', Photos, 'Photos', 1, False, 'Basis'),
                    ('Channel Table', ChannelTable, 'ChannelTable', 1,
                     False, 'Basis'),
                    ('Time Data', TimeHistory, 'TimeHistory', 1, False,
                     'Basis'),
                    ('Sine Specification', SineSweepSpecification,
                     'SineSweepSpecification', 1, False, 'Basis'),
                    ('Sine Levels', SineLevelSet, 'SineLevelSet', 1,
                     False, 'Basis'),
                    ('Report', Report, 'Report', 1, False, None)]
        if project_type == 'Random and Sine':
            # the union of the two halves' slots: the random's whole,
            # then what the sine's adds — its specification and its
            # levels — and the one report last. Composed rather than
            # retyped, so a slot added to either half arrives here too.
            random = raw('Random Vibration')
            have = {icon for _label, _cls, icon, _n, _opt, _side in random}
            added = [slot for slot in raw('Sine Sweep') if slot[2] not in have]
            return random[:-1] + added + [random[-1]]
        if project_type == 'System ID':
            # the two save forms fill this differently, and the slots have
            # to serve both: a time-data save brings the channel table and
            # the two streams (ambient, then driven) and computes the
            # rest; a spectral package brings the FRF, the coherence and
            # the signal-to-noise directly and nothing else — so the
            # setup slots are optional, and the three measured readings
            # are what a finished identification must hold. No shape-set
            # slot: fitting the identified plant is a modal project's job,
            # and typing this run Modal Test is one right-click away.
            return [('Geometry', Geometry, 'Geometry', 1, True,
                     'Basis'),
                    ('Photos', Photos, 'Photos', 1, True, 'Basis'),
                    ('Channel Table', ChannelTable, 'ChannelTable', 1,
                     True, 'Basis'),
                    # the two recordings a time-data save streams, named
                    # for their phases — the same names the import gives
                    # the objects, so they slide into their own slots
                    ('Noise Time History', TimeHistory, 'TimeHistory', 1,
                     True, 'Basis'),
                    ('Excitation Time History', TimeHistory, 'TimeHistory',
                     2, True, 'Basis'),
                    # the ratio is a *reading* of these two, not a third
                    # object (Brandon, 2026-08-25): select both and the
                    # bar offers overlay or ratio-in-dB. Their names match
                    # what Compute PSDs derives from the renamed streams.
                    ('Noise Time History PSDs', Psd, 'Psd', 1, False,
                     'Basis'),
                    ('Excitation Time History PSDs', Psd, 'Psd', 2, False,
                     'Basis'),
                    ('FRF', Frf, 'Frf', 1, False, 'Basis'),
                    ('Multiple Coherence', _CoherenceBase,
                     'MultipleCoherence', 1, False, 'Basis'),
                    ('Report', Report, 'Report', 1, False, None)]
        if project_type == 'Shock':
            # no coherence slot: a shock is one transient event, and
            # coherence is a statement about a stationary average
            return [('Geometry', Geometry, 'Geometry', 1, False,
                     'Basis'),
                    ('Photos', Photos, 'Photos', 1, False, 'Basis'),
                    ('Channel Table', ChannelTable, 'ChannelTable', 1,
                     False, 'Basis'),
                    ('Time Data', TimeHistory, 'TimeHistory', 1, False,
                     'Basis'),
                    ('Shock Specification', ShockSpecification,
                     'ShockSpecification', 1, False, 'Basis'),
                    ('SRS', Srs, 'Srs', 1, False, 'Basis'),
                    # The recommended workflow, slot by slot (Brandon,
                    # 2026-08-25): the raw record is filtered, and the
                    # SRS and the motion chain are computed from the
                    # *filtered* record — the skeleton is where a user is
                    # reminded of that order, so each step gets a slot to
                    # sit in. Best practice rather than required, because
                    # the report renders without them; each is another
                    # TimeHistory, hence the ordinals. There is no PSD
                    # slot any more (Brandon, 2026-08-29): a density of a
                    # transient recording carries a level set by how much
                    # quiet air was captured, and the report's scalogram
                    # answers the where-in-frequency question with *when*
                    # still attached.
                    ('Filtered Time Data', TimeHistory, 'TimeHistory', 2,
                     True, 'Basis'),
                    ('Velocity', TimeHistory, 'TimeHistory', 3, True,
                     'Basis'),
                    ('Displacement', TimeHistory, 'TimeHistory', 4, True,
                     'Basis'),
                    ('Report', Report, 'Report', 1, False, None)]
        return []


    return [(display_name(icon if cls.__name__.startswith('_') else
                          cls.__name__),
             cls, icon, count, optional, side)
            for _label, cls, icon, count, optional, side in raw(project_type)]

def _expected_narrowly(obj: Any, cls: type, classes: Sequence[type],
                       banded: bool | None = None) -> bool:
    """Does `obj` satisfy an expectation of `cls`, read the way the
    type's own list narrows it?

    Raw isinstance is wrong exactly where the type expects a subclass
    *separately*: a Specification is a Psd, so a random project's spec
    filled the 'PSD' slot the moment it arrived, and the spec plus the
    computed PSDs filled 'Octave Band PSD' — the one standing
    deliverable — with no octave object anywhere. Within a type, an
    object that is an instance of a more specific expected class
    belongs to that class's slots and no other's. Same-class ordinals
    ('Octave Band PSD' is the *second* Psd) are untouched.

    And a specification never answers for the plain measurement it
    bounds, whether or not the type expects the specification — the
    same rule `resolve_binding` holds for report bindings. Learned
    the unconditional half in a System ID project: the type expects
    no Specification, so the stream's imported spec slid past the
    per-type narrowing into the 'Noise Time History PSDs' slot, and
    deleting the PSDs left one slot filled by a requirement (Brandon,
    2026-08-23).
    """
    if not isinstance(obj, cls):
        return False
    # and a banded slot — 'Octave Band PSD', 'Octave Band
    # Specification' — wants the object on bands, a plain slot the one
    # on lines: a second narrowband specification imported beside the
    # first was filling the octave slot (2026-09-18), the same rule
    # the report's banded tokens hold
    if banded is not None and is_banded(obj) != banded:
        return False
    from .data import Bounded, TransientSpecification

    written = (Bounded, TransientSpecification)
    if isinstance(obj, written) and not issubclass(cls, written):
        return False
    return not any(
        other is not cls and issubclass(other, cls)
        and isinstance(obj, other) for other in classes)


def missing_expectations(
        project_type: str, objects: Mapping[str, Any],
        placed: Mapping[str, Sequence[str]] | None = None
        ) -> list[tuple[str, type, str, int, bool, str | None]]:
    """The expectations the project does not satisfy yet.

    `placed` is {side: [object names]} — which objects are in each
    side's groups: `Project.placed()` for the Basis, and under
    `OTHER_SIDE` every member of every other group (the window adds
    that key). A slot is filled by an object *on its own side*, never
    by one anywhere in the project: a modal test
    whose only geometry is the model's still needs a measured one, and
    counting globally made the Basis slots vanish the moment any
    geometry at all arrived, leaving a skeleton with nothing in it and
    no way to see what was wrong.

    A slot with no group of its own — the report — is still answered by
    the whole project, since that is what a report reads. Passing
    nothing counts everything for everything, which is what this did
    before and is kept for callers that have no links to offer.
    """
    expectations = project_expectations(project_type)
    classes = [expectation[1] for expectation in expectations]
    out, ordinals = [], {}
    for expectation in expectations:
        _name, cls, icon, _count, _optional, tag = expectation
        # the n-th slot of a class on a side wants an n-th object of it,
        # so a side expecting two shape sets is not filled by one — and
        # a banded slot counts the banded objects, a plain one the rest
        banded = icon in BANDED_TOKENS
        key = (tag, cls, banded)
        ordinals[key] = ordinal = ordinals.get(key, 0) + 1
        if placed is None or tag is None:
            pool = list(objects.values())
        else:
            pool = [objects[name] for name in placed.get(tag, ())
                    if name in objects]
        if sum(1 for obj in pool
               if _expected_narrowly(obj, cls, classes, banded)) < ordinal:
            out.append(expectation)
    return out


def expectation_satisfiers(
        project_type: str, objects: Mapping[str, Any],
        placed: Mapping[str, Sequence[str]] | None = None
        ) -> dict[str, list[str]]:
    """{link group: [object names]} — which objects satisfy each of
    the type's *named* expectations, so the Basis members link
    together. The count-th instance of a class (in project order)
    satisfies a count-th expectation. Slots on the other side carry no
    name and are never guessed into a group: a second geometry and
    shape set are linked by hand, or by the file that brought them.

    That ordering is a *guess*: the first geometry imported could belong
    to either group, and nothing about the object says which. Objects
    already placed are therefore taken as given and left out of the
    guessing — pass `placed` (from `Project.placed()`) and each one
    counts only for the group it is in.
    """
    homes = {name: tag for tag, names in (placed or {}).items()
             for name in names}
    groups = {tag: list(names) for tag, names in (placed or {}).items()}
    expectations = project_expectations(project_type)
    classes = [expectation[1] for expectation in expectations]
    for _name, cls, icon, count, _optional, tag in expectations:
        if tag != 'Basis':
            continue          # the other side is never guessed into
        # The ordinal is over *every* instance in the project, placed or
        # not. Counting only the unplaced ones renumbers them, and the
        # model's geometry becomes the project's first — which is the
        # Basis slot, the very group it was just dragged off.
        banded = icon in BANDED_TOKENS
        instances = [obj_name for obj_name, obj in objects.items()
                     if _expected_narrowly(obj, cls, classes, banded)]
        # a banded slot is the first of its own kind, not the second
        # of the class: the octave PSD is one object, however many
        # narrowband PSDs stand beside it
        wanted = 1 if banded else count
        if len(instances) < wanted:
            continue
        candidate = instances[wanted - 1]
        if homes.get(candidate, tag) != tag:
            continue        # somebody has put it elsewhere
        groups.setdefault(tag, [])
        if candidate not in groups[tag]:
            groups[tag].append(candidate)
    return groups


def _fem_shapes(objects: Mapping[str, Any], test_shapes: str) -> str:
    """The projected FEM shape set to correlate against, by preference:
    one named for the projection, else any second set, else the name
    the projection would produce — an unbound slot until it exists."""
    from .shapes import ShapeSet

    names = [name for name, obj in objects.items()
             if isinstance(obj, ShapeSet) and name != test_shapes]
    projected = [name for name in names if name.endswith(' DOFs')]
    if projected:
        return projected[0]
    return names[0] if names else 'FEM Shapes @ Basis DOFs'


#: the DOF scenes every report opens with, in reading order: the
#: excitations first — the voltage into the shakers, then the force
#: into the article — then the measured responses (Brandon,
#: 2026-08-23). A scene whose source measures none of that quantity
#: does not render, so the fixed list costs nothing where a type is
#: absent.
DOF_SCENE_QUANTITIES = ('voltage', 'force', 'acceleration')

#: which quantities are excitations. This is also what decides an
#: arrow's reading everywhere DOFs are drawn: an excitation ends *on*
#: its node (something arriving), a response points outward
#: (something measured leaving).
EXCITATION_QUANTITIES = frozenset({'voltage', 'force'})


def quantity_order(dims: set[str]) -> list[str]:
    """The quantities in reading order: excitations first — voltage,
    then force — then the responses, alphabetical past the standing
    three. The same order the DOF scenes read in."""
    known = [q for q in DOF_SCENE_QUANTITIES if q in dims]
    return known + sorted(dims - set(DOF_SCENE_QUANTITIES))


def time_data_blocks(source: str, objects: Mapping[str, Any],
                     links: Sequence[Mapping[str, Any]] | None,
                     caption: str, per_quantity: str,
                     mode: str = 'curves') -> list[dict[str, Any]]:
    """The time plots for one history: one figure per quantity it
    measures (Brandon, 2026-08-23 — an axis holds one quantity, so a
    stream carrying drive voltages beside response accelerations must
    not overlay them). `per_quantity` is the caption with a
    ``{quantity}`` slot; a single-quantity or unbound source keeps the
    plain `caption`, one figure, as before.

    Filled by `replace`, never `str.format` (Brandon, 2026-08-25): a
    caption may carry a live reference — `{{Object.field}}` — and
    `format` reads the doubled braces as an escape and collapses them
    to single ones, which the reference resolver then does not match.
    The shock report shipped 'at {Time History.filter_corner}' under
    two figures exactly that way, and only in the two-quantity case,
    because the one-quantity path never calls format at all.

    Flat, not the stage (Brandon, 2026-09-20). A stage figure carries
    `MAX_FIGURE_POINTS` — two million points — and a report's time
    histories are the largest thing in it by an order of magnitude:
    the plate's modal report was 36 MB, 88% of it two such figures,
    and a 144-channel run at 16 kHz made a 200 MB report. The flat
    figure pages its channels two dozen at a time and draws each page
    as an envelope on one shared time axis, which is the same picture
    at every zoom the page offers. `mode='stage'` is still there for a
    template that wants depth."""
    name = resolve_binding(source, objects, links)
    obj = objects.get(name)
    dims = record_dimensions(obj) if obj is not None else set()
    if len(dims) < 2:
        return [{'kind': 'plot', 'source': source, 'mode': mode,
                 'caption': caption}]
    return [{'kind': 'plot', 'source': source, 'mode': mode,
             'select': f'dim:{quantity}',
             'caption': per_quantity.replace('{quantity}', quantity)}
            for quantity in quantity_order(dims)]


def kurtosis_block(source: str = '@basis:TimeHistory',
                   caption: str = '') -> dict[str, Any]:
    """How Gaussian every channel of a record is, as a bar apiece.

    In every report (Brandon, 2026-08-24), because it answers a
    question a spectrum cannot: two records with identical PSDs can be
    a smooth hiss and a train of rare hard peaks, and only the second
    fatigues an article the way the peaks suggest. One chart for all
    channels whatever they measure — kurtosis is dimensionless, so
    this is the one reading here where mixing quantities is honest.

    A computed judgment of the record, so it sits with the other
    judgments rather than beside the traces (the standing order:
    geometry, table, time data, spectra, then what was computed).
    """
    return {'kind': 'bars', 'mode': 'kurtosis', 'source': source,
            'caption': caption or
            'Pearson kurtosis by channel — 3 is Gaussian; above the '
            'band the record carries peaks the spectrum does not '
            'predict, below it the record is clipped or not random'}


def paired_density_blocks(loud: str, quiet: str,
                          objects: Mapping[str, Any],
                          caption: str,
                          per_quantity: str) -> list[dict[str, Any]]:
    """The two densities overlaid, one figure per quantity they share
    — the flat reading of the app's paired stage (Brandon,
    2026-08-23). Same one-quantity-per-axis rule as the time data,
    and the same reading order.
    """
    if not loud or not quiet:
        return []
    shared = (record_dimensions(objects[loud])
              & record_dimensions(objects[quiet]))
    if len(shared) < 2:
        return [{'kind': 'plot', 'mode': 'stage',
                 'source': loud, 'floor': quiet, 'caption': caption}]
    return [{'kind': 'plot', 'mode': 'stage', 'source': loud,
             'floor': quiet, 'quantity': quantity,
             'caption': per_quantity.replace('{quantity}', quantity)}
            for quantity in quantity_order(shared)]


def front_matter(objects: Mapping[str, Any],
                 links: Sequence[Mapping[str, Any]] | None,
                 dofs_source: str) -> list[dict[str, Any]]:
    """The blocks every report opens with, in one order (Brandon,
    2026-08-23): the test geometry, a DOF scene per data type —
    excitations before responses — the photographs, and the channel
    table. What was tested and where it was instrumented, before any
    data means anything. Every slot is symbolic and an unbound or
    empty one does not render, so a bare project keeps the outline
    without blank figures.
    """
    geometry = '@basis:Geometry'
    photos = '@basis:Photos'
    photos_name = resolve_binding(photos, objects, links)
    if photos_name in objects and getattr(objects[photos_name],
                                          'names', None):
        photo_blocks = [
            {'kind': 'photo', 'source': photos_name, 'photo': name,
             'caption': f'Test setup — {name}'}
            for name in objects[photos_name].names]
    else:
        photo_blocks = [{'kind': 'photo', 'source': photos, 'photo': '',
                         'caption': 'Test article as instrumented'}]
    blocks: list[dict[str, Any]] = [
        {'kind': 'scene', 'geometry': geometry, 'shapes': '',
         'caption': 'Test geometry and measurement locations'}]
    # a DOF scene per quantity the source actually measures — a run
    # whose excitation is a load cell has no voltage scene to draw, and
    # the empty card it left in the editor read as a fault (Brandon,
    # 2026-09-08). Every quantity when the source is not there to ask:
    # a bare project keeps the outline as slots to fill.
    source_name = resolve_binding(dofs_source, objects, links)
    source = objects.get(source_name) if source_name else None
    scenes = list(DOF_SCENE_QUANTITIES)
    if source is not None and hasattr(source, 'known_dim'):
        held = set(series_dof_quantities([('', source, None)]))
        scenes = [quantity for quantity in scenes if quantity in held]
    for quantity in scenes:
        role = ('Excitation' if quantity in EXCITATION_QUANTITIES
                else 'Response')
        blocks.append(
            {'kind': 'scene', 'geometry': geometry, 'shapes': '',
             'dofs': quantity, 'dofs_source': dofs_source,
             'caption': f'{role} degrees of freedom ({quantity})'})
    blocks += photo_blocks
    blocks.append({'kind': 'table', 'source': '@basis:ChannelTable',
                   'caption': 'Instrumentation'})
    return blocks


def modal_template(objects: Mapping[str, Any], links: Sequence[Mapping[str, Any]] | None = None) -> Report:
    """A modal-test report from whatever the project holds.

    Sources bind *symbolically* — '@basis:Frf' and friends, resolved
    against the link groups at render time — so the report never
    depends on what anyone named their objects. Blocks whose type the
    project lacks are still added, unbound, so the outline of a proper
    modal report is there to fill in.
    """
    geometry = '@basis:Geometry'
    shapes = '@basis:ShapeSet'
    frf = '@basis:Frf'
    time = '@basis:TimeHistory'
    psd = '@basis:Psd'
    coherence = '@basis:Coherence'

    def ref(name: str, field: str) -> str:
        return '{{' + f'{name}.{field}' + '}}'

    return Report('Modal Test Report', [
        {'kind': 'text', 'text':
            '## Test Summary\n\n'
            'An experimental modal test was performed on the test '
            'article to identify its natural frequencies, damping, and '
            'mode shapes. This report documents the instrumentation, '
            'the measured data and its quality checks, and the modal '
            'parameters identified from it.\n\n'
            f'Time data was acquired on {ref(time, "num_channels")} '
            f'channels at a sample rate of {ref(time, "sample_rate")}, '
            f'with {ref(time, "num_samples")} samples per frame '
            f'({ref(time, "frequency_resolution")} frequency '
            f'resolution), averaged over {ref(time, "num_averages")} '
            f'frames. {ref(shapes, "num_modes")} modes were identified '
            f'between {ref(shapes, "min_frequency")} and '
            f'{ref(shapes, "max_frequency")}.\n\n'
            'The geometry, the instrumentation and the excitation '
            'directions are documented in the figures below, which '
            'are the record of how the article was set up and what '
            'was measured where.'},
        {'kind': 'text', 'text':
            '## Test Setup\n\n'
            'The test geometry ({{figure:Test geometry}}) shows every '
            'measurement location and the wireframe used to visualize '
            'the mode shapes. The excitation degrees of freedom '
            '({{figure:Excitation degrees of freedom}}) mark where and '
            'in which direction the structure was driven; the response '
            'degrees of freedom ({{figure:Response degrees of '
            'freedom}}) mark every measured accelerometer direction. '
            'The channel table ({{table:Instrumentation}}) lists the '
            'instrumentation at each location: sensor, sensitivity, '
            'and calibration status.'},
        *front_matter(objects, links, frf),
        {'kind': 'text', 'text':
            '## Measured Data\n\n'
            'The force and response time histories '
            '({{figure:Excitation force time histories}} and '
            '{{figure:Response acceleration time histories}}) show the '
            'excitation bursts and the structure’s response for '
            'every averaged frame — a check that the excitation was '
            'consistent and the responses died out within each frame. '
            'The averaged auto-power spectral densities '
            '({{figure:Averaged excitation force}} and '
            '{{figure:Averaged response acceleration}}) show how the '
            'excitation energy was distributed over frequency and '
            'which bands the structure responded in.'},
        # flat, for the reason `time_data_blocks` gives: these two were
        # 88% of a 36 MB page (Brandon, 2026-09-20). The spectra below
        # keep the stage — a thousand lines a channel is a figure the
        # depth helps and the file can hold.
        {'kind': 'plot', 'source': time, 'mode': 'curves',
         'select': 'dim:force', 'caption': 'Excitation force time '
         'histories, all frames'},
        {'kind': 'plot', 'source': time, 'mode': 'curves',
         'select': 'dim:acceleration', 'caption': 'Response '
         'acceleration time histories, all frames'},
        {'kind': 'plot', 'source': psd, 'mode': 'stage',
         'select': 'dim:force', 'caption': 'Averaged excitation force '
         'auto-power spectral density'},
        {'kind': 'plot', 'source': psd, 'mode': 'stage',
         'select': 'dim:acceleration', 'caption': 'Averaged response '
         'acceleration auto-power spectral densities'},
        {'kind': 'text', 'text':
            '## Data Quality\n\n'
            'The drive point FRF magnitudes '
            '({{figure:Drive point FRF magnitudes}}) should show a '
            'clear resonance-antiresonance alternation, the signature '
            'of a true drive point measurement. The imaginary parts '
            '({{figure:Drive point FRF imaginary}}) should be '
            'single-sided (one sign at every resonance): a sign '
            'flip indicates a sensor or excitation polarity error, and '
            'the peak magnitudes indicate how well each mode was '
            'excited from that drive point. The multiple coherence '
            'map ({{figure:Multiple coherence}}) shows, channel by '
            'channel and frequency by frequency, '
            'how much of each response the excitation accounts for — '
            'dark rows are suspect channels, dark bands are '
            'poorly-excited frequency ranges.'},
        {'kind': 'plot', 'source': frf, 'mode': 'curves',
         'select': 'drive', 'component': 'magnitude',
         'shapes': shapes,
         'caption': 'Drive point FRF magnitudes, the identified mode '
         'frequencies marked'},
        {'kind': 'plot', 'source': frf, 'mode': 'curves',
         'select': 'drive', 'component': 'imag',
         'shapes': shapes,
         'caption': 'Drive point FRF imaginary parts, the identified '
         'mode frequencies marked — single-sided '
         'peaks confirm drive point behavior; peak size shows how '
         'well each mode was excited'},
        {'kind': 'plot', 'source': coherence, 'mode': 'stage',
         'caption': 'Multiple coherence, every channel over frequency '
         '(1.0 means the excitation fully accounts for the response)'},
        {'kind': 'text', 'text':
            '## Modal Analysis\n\n'
            'The Complex Mode Indicator Function '
            '({{figure:Complex Mode Indicator Function}}) condenses '
            'the full FRF matrix into its singular values at every '
            'frequency line; peaks mark the structure’s modes, and '
            'repeated peaks in the lower curves reveal closely spaced '
            'or repeated roots. The dashed curves are the CMIF '
            'resynthesized from the identified modal model — how '
            'closely they track the measurement is the quality of the '
            'fit, and the dashed vertical markers sit at each '
            'identified frequency. The identified parameters are '
            'listed in {{table:Identified modal parameters}}. The '
            'auto-MAC ({{figure:Auto-MAC}}) checks the identified '
            'shapes’ independence: large off-diagonal terms mean '
            'two shapes are not truly distinct. The animated mode '
            'shapes ({{figure:Identified mode shapes}}) follow, '
            'colored by displacement.'},
        {'kind': 'plot', 'source': frf, 'mode': 'cmif', 'shapes': shapes,
         'caption': 'Complex Mode Indicator Function, measured (solid) '
         'against the modal model resynthesis (dashed)'},
        {'kind': 'table', 'source': shapes,
         'caption': 'Identified modal parameters'},
        {'kind': 'plot', 'source': shapes, 'mode': 'mac',
         'caption': 'Auto-MAC of the identified mode shapes — an '
         'identity-like matrix confirms independent shapes'},
        {'kind': 'scene', 'geometry': geometry, 'shapes': shapes,
         'caption': 'Identified mode shapes (pick a mode to animate '
         'it, colored by displacement)'},
        # test-analysis correlation is optional: with no projected FEM
        # shapes in the project this block is an unbound slot and the
        # report renders without it
        {'kind': 'text', 'text':
            '## Test-Analysis Correlation\n\n'
            'Where a finite element model of the same article is in '
            'the project, the identified modes are compared against '
            'it. The cross-MAC ({{figure:Test-analysis correlation}}) '
            'scores every test mode against every model mode: a value '
            'near one says two shapes describe the same motion, and '
            'the pattern of the matrix says whether the model orders '
            'its modes the way the article does. The comparison is '
            'always made in the test\'s own degrees of freedom, '
            'because that is where the measurement exists — a model '
            'has motion everywhere and a test has it where the '
            'accelerometers were.\n\n'
            'The matched pairs are then listed with both sets\' '
            'frequencies and damping side by side ({{table:Matched '
            'modes}}), and animated over each other '
            '({{figure:Matched mode pairs}}) so a pair that scores '
            'well but moves differently is visible as well as '
            'scored. Each shape in the animation is drawn to its own '
            'peak, so what is being compared is shape and not scale — '
            'a scale difference between two sets is almost always a '
            'mass unit or a normalization convention rather than the '
            'structure, and the figure says so in words when it '
            'finds one.\n\n'
            'These figures are unbound and absent when the project '
            'holds no model to compare against, which is the ordinary '
            'case for a test run on its own.'},
        {'kind': 'plot', 'source': shapes, 'mode': 'mac',
         'shapes': _fem_shapes(objects,
                               resolve_binding(shapes, objects, links)),
         'caption': 'Test-analysis correlation: cross-MAC between the '
         'identified test modes and the finite element model '
         'projected onto the test DOFs'},
        # ...as is the matched-modes table: an unbound slot until the
        # user commits matches on the comparison screen.
        #
        # `@any` and not `@basis`: a comparison names a set on each
        # side, so it is linked into neither group and a basis-scoped
        # selector would never find it.
        {'kind': 'pairs', 'source': '@any:MatchedModes',
         'caption': 'Matched modes: both sets’ parameters side by '
         'side, frequency error against the test set, and the MAC of '
         'each matched pair'},
        {'kind': 'overlay', 'source': '@any:MatchedModes',
         'caption': 'Matched mode pairs animated over each other — '
         'the test set in blue, the model in orange, phase-aligned '
         'to move together'},
        # no kurtosis figure: a modal test is driven with burst random,
        # whose kurtosis read whole is about three over its duty and
        # says nothing about the article (Brandon, 2026-09-08: "We
        # don't need it"). The toolbar's reading is still there for a
        # record someone wants to look at.
        {'kind': 'text', 'text':
            '## Conclusions\n\n'
            f'{ref(shapes, "num_modes")} modes were identified between '
            f'{ref(shapes, "min_frequency")} and '
            f'{ref(shapes, "max_frequency")}, and their frequencies, '
            'damping and shapes are listed in '
            '{{table:Identified modal parameters}}.\n\n'
            'The quality of that identification is read off three '
            'figures. The resynthesized CMIF '
            '({{figure:Complex Mode Indicator Function}}) shows how '
            'closely a model built from those modes reproduces the '
            'measurement it was fitted to — where the dashed curve '
            'leaves the solid one, the model does not account for '
            'what was measured. The auto-MAC ({{figure:Auto-MAC}}) '
            'shows whether the identified shapes are independent of '
            'each other; large off-diagonal terms mean two modes are '
            'not truly distinct, which is the usual sign of a '
            'repeated root or an over-fit. The multiple coherence '
            '({{figure:Multiple coherence}}) shows which channels and '
            'which bands the identification had good data in to begin '
            'with.\n\n'
            'A mode identified in a band where the coherence is poor, '
            'or one whose shape correlates strongly with its '
            'neighbor, is one to re-examine before it is used — in a '
            'correlation against a model, or as the basis of anything '
            'downstream.'},
    ])


def shock_template(objects: Mapping[str, Any], links: Sequence[Mapping[str, Any]] | None = None) -> Report:
    """A shock report: the transients, and the SRS against its target.

    The time histories lead the data and are not decoration. A shock
    is a single event and its whole character — how long, how many,
    whether the article was still ringing when the next one landed —
    is in the trace and nowhere in the spectrum. The SRS answers a
    narrower question and answers it after. Front matter first, like
    every report here (Brandon, 2026-08-23).
    """
    time = '@basis:TimeHistory'
    srs = '@basis:Srs'

    def ref(name: str, field: str) -> str:
        return '{{' + f'{name}.{field}' + '}}'

    # The motion chain, bound by name like the sysid streams — a
    # generated template knows its own project's names. The filtered
    # record is found by provenance (`objects` is the project when the
    # app generates, and provenance is the truth about what was
    # filtered); velocity and displacement are what they measure, so
    # their dimensions are the honest finder whoever named them.
    from .data import TimeHistory as _TimeHistory
    provenance = getattr(objects, 'provenance', {}) or {}
    filtered = next(
        (name for name, record in provenance.items()
         if record.get('verb') == 'filter_data' and name in objects
         and isinstance(objects[name], _TimeHistory)), '')
    raw = next(
        (record['source'] for name, record in provenance.items()
         if name == filtered), '')
    velocity = next(
        (name for name, obj in objects.items()
         if isinstance(obj, _TimeHistory)
         and record_dimensions(obj) == {'velocity'}), '')
    displacement = next(
        (name for name, obj in objects.items()
         if isinstance(obj, _TimeHistory)
         and record_dimensions(obj) == {'length'}), '')
    # The figures first, then the prose that points at them
    # (Brandon, 2026-08-25). A `{{figure:...}}` matches the *start* of
    # a real caption, and a caption written by `time_data_blocks`
    # carries the quantity when a record measures more than one — so
    # a reference guessed from the caption template read
    # 'Low-pass filtered transients' against a figure captioned
    # 'Low-pass filtered voltage transients' and quietly stayed
    # unresolved in the shipped document. Referencing what was
    # actually built cannot drift from it.
    def figures(source: str, caption: str, per_quantity: str
                ) -> list[dict[str, Any]]:
        return (time_data_blocks(source, objects, links, caption,
                                 per_quantity) if source else [])

    filtered_blocks = figures(
        filtered,
        'Filtered transients, '
        f'{ref(raw or filtered, "filter_description")}',
        'Filtered {quantity} transients, '
        f'{ref(raw or filtered, "filter_description")}')
    velocity_blocks = figures(velocity, 'Integrated velocity, drift '
                              'high-passed',
                              'Integrated velocity, drift high-passed')
    displacement_blocks = figures(
        displacement, 'Integrated displacement, drift high-passed',
        'Integrated displacement, drift high-passed')

    def first_reference(blocks: list[dict[str, Any]]) -> str:
        """A reference to the first of a group, by its own caption."""
        if not blocks:
            return ''
        return '{{figure:' + blocks[0]['caption'].split(',')[0] + '}}'

    motion_blocks: list[dict[str, Any]] = []
    if filtered_blocks or velocity_blocks or displacement_blocks:
        named = ' and '.join(
            reference for reference in
            (first_reference(filtered_blocks),
             first_reference(velocity_blocks),
             first_reference(displacement_blocks)) if reference)
        motion_blocks.append({'kind': 'text', 'text':
            '## The Motion\n\n'
            f'The derived motion ({named}). '
            + ('The filtered transients are the measured record '
               'through a zero-phase '
               f'{ref(raw or filtered, "filter_description")}, order '
               f'{ref(raw or filtered, "filter_order")} — zero phase '
               'so every peak stays at its measured instant, where a '
               'causal filter would shift it by the group delay. '
               'Everything judged below is computed from this '
               'filtered record rather than the raw one, so the '
               'spectra, the motion and the response spectrum all '
               'describe the same signal. '
               if filtered_blocks else '')
            + ('Velocity and displacement are '
               if velocity_blocks and displacement_blocks else
               'The velocity is ' if velocity_blocks else
               'The displacement is ' if displacement_blocks else '')
            + ('integrated in the time domain over the whole record, '
               'with the channel mean removed and a gentle zero-phase '
               'high-pass after each integration: a sensor bias '
               'integrates to a ramp that would dwarf the motion, and '
               'nothing below the drift corner can tell bias from '
               'true motion, so the corner is a declared judgment '
               'rather than a detection. The shock windows carry over '
               'unchanged — the events numbered here are the events '
               'numbered on the measured traces. Velocity and '
               'displacement are what the article was actually made '
               'to do: an SRS reads the damage potential, but a '
               'fixture stroke or a clearance is checked against '
               'displacement, and pyroshock severity criteria are '
               'often written in velocity.'
               if velocity_blocks or displacement_blocks else '')})
    motion_blocks += (filtered_blocks + velocity_blocks
                      + displacement_blocks)

    # The scalogram replaced the density here (Brandon, 2026-08-29):
    # a PSD of a transient recording carries a level that depends on
    # how much quiet air was recorded around the events — energy per
    # unit time, over a duration that is an accident of the capture —
    # and nothing in the shock judgment reads it. What the density
    # was kept for (naming the source of an unexpected SRS peak — a
    # fixture resonance, a programmer ringing, a bolt buzzing) the
    # scalogram answers better, with *when* still attached: each
    # event's ring is a streak at its frequency, its decay legible
    # along time, so a resonance that only one event excited is
    # visible as exactly that.
    spectral_blocks: list[dict[str, Any]] = [
        {'kind': 'text', 'text':
            '## Where The Energy Is In Time\n\n'
            'The scalogram ({{figure:Shock scalogram}}) reads the '
            'filtered record against time *and* frequency at once — '
            'time across, frequency receding, amplitude standing up '
            'out of the floor. The SRS above says what each event '
            'would do to a structure; this says where in frequency '
            'the energy that did it actually sat, and *when* — which '
            'is what names the source of an unexpected SRS peak: a '
            'fixture resonance rings at one frequency across every '
            'event, a programmer rings only while it is struck, a '
            'loose bolt buzzes between them. A density would average '
            'all of that away, and its level would depend on how '
            'much quiet air the recording happened to hold. One '
            'channel is drawn, the way the application\'s wavelet '
            'view reads one at a time; the lowest rows lean on the '
            'record\'s ends for the span the caption gives, because '
            'a low-frequency wavelet is long, and inside that span '
            'the picture is shaped by where the record was cut '
            'rather than by the event.'},
        {'kind': 'plot',
         'source': filtered or '@basis:TimeHistory',
         'mode': 'scalogram', 'select': 'dim:acceleration',
         'caption': 'Shock scalogram'}]

    return Report('Shock Test Report', [
        {'kind': 'text', 'text':
            '## Test Summary\n\n'
            'A shock test was run to subject the article to a required '
            'response spectrum. A shock specification names the '
            'response, not the waveform: any transient whose shock '
            'response spectrum lands inside the tolerance band meets '
            'it, which is what separates a shock test from a transient '
            'replication. This report documents the instrumentation, '
            'the events as they were recorded, the spectrum each '
            'produced, and how far each sits from what was '
            'required.\n\n'
            f'{ref(time, "num_shocks")} events were recorded across '
            f'{ref(time, "num_channels")} channels at '
            f'{ref(time, "sample_rate")} over '
            f'{ref(time, "duration")} of record. The spectra are '
            f'computed from {ref(srs, "min_frequency")} to '
            f'{ref(srs, "max_frequency")}.'},
        {'kind': 'text', 'text':
            '## Test Article and Instrumentation\n\n'
            'The test geometry ({{figure:Test geometry}}) shows where '
            'the article was instrumented. The excitation degrees of '
            'freedom ({{figure:Excitation degrees of freedom}}) mark '
            'where and in which direction the shock was applied, and '
            'the response degrees of freedom ({{figure:Response '
            'degrees of freedom}}) every measured direction — a shock '
            'is judged where it was measured, and a channel not on '
            'these figures is a channel this report says nothing '
            'about. The photographs record the article as it was '
            'actually mounted, which for a shock test is the account '
            'of the fixture and the programmer that no number carries. '
            'The channel table ({{table:Instrumentation}}) lists the '
            'instrumentation at each location, including the '
            'measurement range — a shock is where an under-ranged '
            'accelerometer clips, and the table is where that is '
            'caught.'},
        *front_matter(objects, links, '@basis:TimeHistory'),
        {'kind': 'text', 'text':
            '## The Events\n\n'
            'The measured transients ({{figure:Measured}}), with the '
            'window each spectrum is computed over marked across every '
            'channel and numbered. A shock is a single event and its '
            'whole character is in the trace: how long it lasted, how '
            'hard it hit, whether the article was still ringing when '
            'the next one landed, and whether any channel clipped. '
            'None of that survives into a spectrum, which is why the '
            'traces come first and are not decoration.'},
        *time_data_blocks(
            '@basis:TimeHistory', objects, links,
            'Measured shock transients',
            'Measured {quantity} shock transients'),
        *motion_blocks,
        *spectral_blocks,
        {'kind': 'text', 'text':
            '## What Was Required\n\n'
            'The required response spectrum and the band around it '
            '({{figure:Shock specification}}). This is the whole of '
            'what the test had to achieve: a transient meets it by '
            'producing a shock response spectrum inside the band, '
            'however the machine and programmer arrived at that '
            'transient. It is shown alone before anything is compared '
            'against it, because it is the requirement rather than a '
            'result — and reading it first is what makes the '
            'comparison that follows a judgment instead of a '
            'picture.'},
        {'kind': 'plot', 'source': '@basis:ShockSpecification',
         'mode': 'curves', 'caption': 'Shock specification, with its '
         'tolerance band'},
        {'kind': 'text', 'text':
            '## What The Article Saw\n\n'
            'The measured shock response spectra against the '
            'requirement ({{figure:Measured shock response spectra}}), '
            'one curve per channel per event. The shock response '
            'spectrum is the peak response a single-degree-of-freedom '
            'oscillator would have reached at each natural frequency, '
            'so it says what the event did to a structure rather than '
            'what the acceleration did in time — which is why a '
            'requirement is written in these terms at all.\n\n'
            'The deviation bars ({{figure:SRS deviation}}) are the '
            'same comparison as a number: RMS across the band in '
            'decibels, one bar per channel per event, signed. Over '
            'the band the event was too hard and under it the event '
            'under-tested, and those are different faults — an '
            'over-test may have damaged the article, an under-test '
            'proves less than it claims. RMS across the band rather '
            'than a linear error because a spectrum spans decades on '
            'log axes: a linear reading follows whichever bands are '
            'loudest and goes blind to a large ratio error where the '
            'requirement is small.'},
        {'kind': 'plot', 'source': '@basis:Srs', 'mode': 'curves',
         'specification': '@basis:ShockSpecification',
         'caption': 'Measured shock response spectra against the '
         'specification'},
        # the judgment, not just the picture: RMS deviation across the
        # band in decibels, one bar per channel. This block lived in
        # the transient template until the SRS pair left that type —
        # the SRS is how a *shock* is judged, so its deviation reads
        # here and nowhere else.
        {'kind': 'bars', 'mode': 'srs',
         'source': '@basis:ShockSpecification', 'measured': '@basis:Srs',
         'caption': 'SRS deviation by channel'},
        # no kurtosis here (Brandon, 2026-08-24): a shock record is a
        # transient in a long quiet stretch and never reads near
        # three — the plate's own reads 16 to 27 over its windows —
        # so a band drawn at 2 to 4 would mark every channel of every
        # shock test in red and say nothing anyone could act on. The
        # toolbar toggle still offers the reading to anyone who wants
        # to look; what a *report* asserts is the judgment.
        {'kind': 'text', 'text':
            '## Conclusions\n\n'
            f'{ref(time, "num_shocks")} events were delivered and are '
            'documented above. Whether each was accepted is read off '
            '{{figure:Measured shock response spectra}} and '
            '{{figure:SRS deviation}} together: a channel whose '
            'spectrum sits inside the band met the requirement at '
            'that channel, and the deviation bars say by how much and '
            'in which direction where it did not. The traces in '
            '{{figure:Measured}} are what a disputed channel is '
            'settled against — a spectrum computed over a window that '
            'caught a neighboring event, or over a channel that '
            'clipped, will sit wrong in a way no spectral figure can '
            'reveal on its own.'},
    ])


def _sine_level_blocks(objects: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The sine half's judgment: one comparison figure per tone, the
    prose that reads them, and the deviation bars — shared by the sine
    report and the random-and-sine one, so the two cannot drift.

    One figure per tone, because each tone sweeps its own frequencies
    on its own clock; overlaying them would draw different requirements
    over each other. The extracted levels present in the project name
    the figures; a project not yet extracted gets one symbolic figure so
    the outline shows what is missing.
    """
    from .sine import SineLevel, SineLevelSet

    blocks: list[dict[str, Any]] = []
    grouped = [(name, obj) for name, obj in objects.items()
               if isinstance(obj, SineLevelSet)]
    loose = [name for name, obj in objects.items()
             if isinstance(obj, SineLevel)]
    if grouped:
        name, level_set = grouped[0]
        for tone in level_set.tone_names:
            blocks.append(
                {'kind': 'plot', 'source': name, 'tone': tone,
                 'mode': 'curves',
                 'specification': '@basis:SineSweepSpecification',
                 'caption': f'{tone}: extracted level against the '
                 'requirement'})
    elif loose:
        for name in loose:
            blocks.append(
                {'kind': 'plot', 'source': name, 'mode': 'curves',
                 'specification': '@basis:SineSweepSpecification',
                 'caption': f'{objects[name].tone}: extracted level '
                 'against the requirement'})
    else:
        blocks.append(
            {'kind': 'plot', 'source': '@basis:SineLevelSet',
             'mode': 'curves',
             'specification': '@basis:SineSweepSpecification',
             'caption': 'Extracted levels against the requirement'})
    blocks.append({'kind': 'text', 'text':
        '## Level Against Requirement\n\n'
        'One figure per tone, because each tone sweeps its own '
        'frequencies on its own clock and overlaying them would draw '
        'different requirements over each other. Each shows the level '
        'extracted from the recording against the amplitude that tone '
        'was required to hold, with the warning and abort bands drawn '
        'around it.\n\n'
        'The frequency coverage in each figure is the coverage: a '
        'sweep stopped early, or a tone the instructions windowed, '
        'extracts fewer lines, and the comparison shows exactly how '
        'much of the required range was actually run rather than '
        'reporting that everything was. The level itself is '
        'demodulated from the recording and corrected for the noise '
        'inside each tracking window — a controller\'s own live '
        'display reads high under a noisy background, because it '
        'counts that noise as signal and backs its drive off '
        'accordingly.'})
    blocks.append({'kind': 'text', 'text':
        '## Deviation\n\n'
        'The same comparison as a number ({{figure:Sine level '
        'deviation}}): signed RMS decibels, a bar per tone per '
        'control channel. Signed, because over-testing and '
        'under-testing are different faults rather than degrees '
        'of one — an over-test may have damaged the article, an '
        'under-test proves less than it claims. This is where a '
        'run with many tones and many channels is scanned: the '
        'figures above are read one at a time, and this one '
        'answers which of them to read first.'})
    blocks.append({'kind': 'bars', 'mode': 'sine',
                   'source': '@basis:SineSweepSpecification',
                   'measured': '@basis:SineLevelSet',
                   'caption': 'Sine level deviation by tone and channel'})
    return blocks


def sine_template(objects: Mapping[str, Any],
                  links: Sequence[Mapping[str, Any]] | None = None) -> Report:
    """A sine sweep report: the tones, and the levels they reached.

    The level figures and the deviation bars are `_sine_level_blocks`,
    shared with the random-and-sine report: the bars judge every tone
    at every control channel in one chart, the same signed
    three-decibel reading the random and shock reports give.
    """
    time = '@basis:TimeHistory'

    def ref(name: str, field: str) -> str:
        return '{{' + f'{name}.{field}' + '}}'

    blocks: list[dict[str, Any]] = [
        {'kind': 'text', 'text':
            '## Test Summary\n\n'
            'A sine sweep test was run: one or more tones swept across '
            'a frequency range at a controlled amplitude, so that the '
            'article sees each frequency in turn at a required level '
            'rather than all of them at once. This report documents '
            'the instrumentation, the recording, the level each tone '
            'actually reached against what it was required to reach, '
            'and how far apart those two are.\n\n'
            f'The run was recorded on {ref(time, "num_channels")} '
            f'channels at {ref(time, "sample_rate")} over '
            f'{ref(time, "duration")}.'},
        {'kind': 'text', 'text':
            '## Test Article and Instrumentation\n\n'
            'The test geometry ({{figure:Test geometry}}) shows where '
            'the article was instrumented, the excitation degrees of '
            'freedom ({{figure:Excitation degrees of freedom}}) where '
            'and in which direction it was driven, and the response '
            'degrees of freedom ({{figure:Response degrees of '
            'freedom}}) every measured direction. The photographs '
            'record the article as it was actually instrumented, and '
            'the channel table ({{table:Instrumentation}}) lists what '
            'was at each location — for a swept test the control '
            'channels are the ones the levels below are read from, '
            'and the table is where they are identified.'},
        *front_matter(objects, links, '@basis:TimeHistory'),
        {'kind': 'text', 'text':
            '## The Recording\n\n'
            'The measured time histories ({{figure:Measured}}). A '
            'sweep is read in the time domain first because its faults '
            'live there: a tone that lost control, a dwell that sat '
            'longer than intended, an amplitude that stepped rather '
            'than swept, or a channel that dropped out partway. The '
            'levels extracted below are demodulated from exactly this '
            'record, so anything visible here is inherited by every '
            'figure after it.'},
        *time_data_blocks(
            '@basis:TimeHistory', objects, links,
            'Measured time histories',
            'Measured {quantity} time histories'),
    ]
    blocks += _sine_level_blocks(objects)
    blocks += [
        {'kind': 'text', 'text':
            '## The Shape Of The Recording\n\n'
            'Pearson kurtosis ({{figure:Pearson kurtosis}}) reads the '
            'distribution of each channel rather than its level — '
            'three is Gaussian. A swept sine is not Gaussian and is '
            'not meant to be: a pure tone reads 1.5, and a sweep '
            'mixed with a random background sits between. What the '
            'figure is watched for is a channel far from its '
            'neighbors, which is a channel that clipped or rattled '
            'while the tracked level was extracted from it '
            'regardless.'},
        kurtosis_block(),
        {'kind': 'text', 'text':
            '## Conclusions\n\n'
            'Which tones were run and how much of each required range '
            'was covered is read off the per-tone figures above, where '
            'the extracted level spans exactly the frequencies the '
            'sweep reached. How closely each was held is read off '
            '{{figure:Sine level deviation}}, tone by tone and channel '
            'by channel. Where a tone falls short of its range or a '
            'channel sits outside its band, the recording in '
            '{{figure:Measured}} is what says whether the cause was '
            'the article, the control, or a run that stopped — and '
            'the kurtosis figure is what says whether the recording '
            'those levels were extracted from was clean enough to '
            'trust.'},
    ]
    return Report('Sine Sweep Test Report', blocks)


def sysid_template(objects: Mapping[str, Any],
                   links: Sequence[Mapping[str, Any]] | None = None) -> Report:
    """A system identification report: the measured plant, and whether
    to believe it.

    The CMIF reads the whole FRF matrix at a glance — a package's
    3 x 9 is twenty-seven curves nobody reads one by one — the
    coherence map says where the estimate held together, and the
    signal-to-noise says whether there was a measurement at all:
    where it approaches one, the plant is the room.

    Ordered the way every report here is (Brandon, 2026-08-23): the
    front matter — geometry, DOF scenes, photographs, channel table —
    then the time data, then the spectra, then the computed
    judgments: coherence and signal-to-noise. The time data is the
    ambient (noise) stream first and the driven one after it, each
    drawn under its own averaging frames; a spectral package has no
    streams, and the block stays an unbound slot that does not render.
    """
    import numpy as np

    from .data import Psd, Specification, TimeHistory
    densities = [(name, obj) for name, obj in objects.items()
                 if isinstance(obj, Psd)
                 and not isinstance(obj, Specification)]
    # the ratio binds the pair by name, louder over quieter — there is
    # no symbolic selector for "the second Psd", and a generated
    # template knows its own project's names
    loud_density = quiet_density = ''
    if len(densities) >= 2:
        ranked = sorted(
            densities,
            key=lambda pair: -float(np.nanmean(np.real(
                np.asarray(pair[1].ordinate)))))
        loud_density, quiet_density = ranked[0][0], ranked[1][0]
    # the two streams bind by name the same way: the quiet one is the
    # ambient (noise) recording, the loud one the driven, and each
    # draws with its own averaging frames so the figure says which
    # part of each stream the densities came from
    histories = [(name, obj) for name, obj in objects.items()
                 if isinstance(obj, TimeHistory)]
    if len(histories) >= 2:
        by_level = sorted(
            histories,
            key=lambda pair: float(np.nanmean(np.abs(
                np.asarray(pair[1].ordinate)))))
        # both streams are read for shape, not just the driven one
        # (Brandon, 2026-08-24): an ambient recording that is not
        # Gaussian is a noisy room or a rattling fixture, and a drive
        # that is not is a different fault entirely — the two are
        # worth telling apart, and the pair of figures is how
        kurtosis_blocks = [
            kurtosis_block(
                by_level[0][0],
                'Pearson kurtosis of the ambient recording, by channel '
                '— 3 is Gaussian; a noise floor that is not says the '
                'room or the fixture, not the article'),
            kurtosis_block(
                by_level[-1][0],
                'Pearson kurtosis of the excitation, by channel — 3 is '
                'Gaussian; above the band the drive carries peaks the '
                'spectrum does not predict, below it the drive is '
                'clipped')]
        time_blocks: list[dict[str, Any]] = [
            *time_data_blocks(
                by_level[0][0], objects, links,
                'Ambient (noise) time history, with the frames '
                'the noise densities are averaged over',
                'Ambient (noise) {quantity} time histories, with the '
                'frames the noise densities are averaged over'),
            *time_data_blocks(
                by_level[-1][0], objects, links,
                'Excitation time history, with the frames the '
                'driven densities are averaged over',
                'Excitation {quantity} time histories, with the '
                'frames the driven densities are averaged over')]
    else:
        # a spectral package has no streams; the slots keep the outline
        kurtosis_blocks = [kurtosis_block()]
        time_blocks = [
            {'kind': 'plot', 'source': '@basis:TimeHistory',
             'mode': 'curves',
             'caption': 'Measured time histories — the ambient stream, '
             'then the driven one'}]
    time = '@basis:TimeHistory'
    frf = '@basis:Frf'

    def ref(name: str, field: str) -> str:
        return '{{' + f'{name}.{field}' + '}}'

    return Report('System ID Report', [
        {'kind': 'text', 'text':
            '## Test Summary\n\n'
            'A system identification was run to measure the plant — '
            'the frequency response of each control channel to each '
            'drive — that a subsequent test will be controlled '
            'through. This report documents the instrumentation, the '
            'two recordings the identification was made from, the '
            'plant it produced, and the three readings that say '
            'whether to believe it.\n\n'
            f'The identification measured {ref(frf, "num_records")} '
            f'frequency response functions over '
            f'{ref(frf, "num_channels")} response channels, at a '
            f'resolution of {ref(frf, "frequency_resolution")} out to '
            f'{ref(frf, "max_frequency")}. The recordings were taken '
            f'at {ref(time, "sample_rate")} and each analyzed over '
            f'{ref(time, "num_frames")} frames of '
            f'{ref(time, "frame_length")} samples with a '
            f'{ref(time, "window")} window at {ref(time, "overlap")} '
            'overlap.'},
        {'kind': 'text', 'text':
            '## Test Article and Instrumentation\n\n'
            'The test geometry ({{figure:Test geometry}}) shows every '
            'measurement location and the wireframe the article is '
            'drawn on. The excitation degrees of freedom '
            '({{figure:Excitation degrees of freedom}}) mark where '
            'and in which direction the article was driven, and the '
            'response degrees of freedom ({{figure:Response degrees '
            'of freedom}}) mark every measured direction — together '
            'they are the identification\'s reach, since a plant is '
            'only measured where something was driven and something '
            'was listening. The photographs record the article as it '
            'was actually instrumented, which is the only account of '
            'cable routing, fixture condition and sensor mounting '
            'that survives the test. The channel table '
            '({{table:Instrumentation}}) lists what was at each '
            'location: sensor, sensitivity, and calibration '
            'status.'},
        *front_matter(objects, links, '@basis:TimeHistory'
                      if any(isinstance(obj, TimeHistory)
                             for obj in objects.values())
                      else '@basis:Frf'),
        {'kind': 'text', 'text':
            '## The Recordings\n\n'
            'A system identification is made from two recordings, and '
            'both are shown because the answer is a ratio of them. '
            'The ambient recording is taken with the shakers still: '
            'it is the noise floor of the whole measurement chain — '
            'the room, the amplifiers, the cabling and the '
            'instrumentation — and it sets the level below which the '
            'driven measurement means nothing. The excitation '
            'recording is the same channels with the drives running. '
            'Each is drawn one figure per quantity, with the frames '
            'its spectra are averaged over marked on it, so the '
            'figures say which stretch of each recording the rest of '
            'this report is computed from.'},
        *time_blocks,
        {'kind': 'text', 'text':
            '## The Measured Plant\n\n'
            'The densities of the two recordings are drawn together '
            'per channel, the driven level colored and the ambient '
            'one stood back in gray behind it: the distance between '
            'them at any frequency is the margin the identification '
            'had to work with there, and where they meet there was '
            'nothing to measure. The plant itself follows — every '
            'frequency response function on one stage, channels '
            'receding — which is the deliverable of the run and what '
            'a controller will invert.'},
        *paired_density_blocks(
            loud_density, quiet_density, objects,
            'Excitation and ambient densities overlaid, per channel',
            'Excitation and ambient {quantity} densities on the '
            'stage, channel by channel — the driven level colored, '
            'the ambient one stood back in gray behind it'),
        {'kind': 'plot', 'source': '@basis:Frf', 'mode': 'stage',
         'caption': 'The measured plant: every FRF on the stage, '
         'channels receding, colored by level'},
        {'kind': 'text', 'text':
            '## Whether To Believe It\n\n'
            'Two readings judge the identification, and they fail in '
            'different ways. Multiple coherence '
            '({{figure:Multiple coherence}}) says how much of each '
            'response the drives account for: at one, the channel '
            'moved because the shakers moved it; well below one, '
            'something else did, and the frequency response measured '
            'there is a coincidence rather than a plant. A dark row '
            'is a suspect channel and a dark band a poorly-driven '
            'frequency range.\n\n'
            'Signal to noise ({{figure:Signal to noise}}) says '
            'whether there was a measurement at all: it is the '
            'driven density over the ambient one, in decibels, so it '
            'is the margin above the noise floor channel by channel '
            'and line by line. Where it approaches zero the '
            'identification is measuring the room. The two are worth '
            'reading together — a channel can be well above its '
            'noise floor and still incoherent, which points at a '
            'rattle or a loose sensor rather than at the level.'},
        {'kind': 'plot', 'source': '@basis:MultipleCoherence',
         'mode': 'stage',
         'caption': 'Multiple coherence, every channel on the stage'},
        {'kind': 'plot', 'mode': 'stage', 'reading': 'ratio',
         'source': loud_density, 'floor': quiet_density,
         'caption': 'Signal to noise: the driven density over the '
         'ambient one, per channel, in decibels'},
        {'kind': 'text', 'text':
            '## The Shape Of The Recordings\n\n'
            'A spectrum says nothing about the shape of the '
            'distribution that produced it: two recordings with '
            'identical densities can be a smooth hiss and a train of '
            'rare hard peaks. Pearson kurtosis is the number that '
            'tells them apart — three for a Gaussian — and both '
            'recordings are read for it, because they fail '
            'differently. An ambient recording that is not Gaussian '
            'is a noisy room, a rattling fixture or an intermittent '
            'connection, none of which the level alone would show. A '
            'drive that is not Gaussian is a drive carrying peaks its '
            'own spectrum does not predict, or one being clipped by '
            'the amplifier — and a clipped drive makes a plant that '
            'is not the article\'s.'},
        *kurtosis_blocks,
        {'kind': 'text', 'text':
            '## Conclusions\n\n'
            'What this identification is good for is read off the '
            'three judgments above. The band it covers is the band '
            'the plant was measured over, out to '
            f'{ref(frf, "max_frequency")} at '
            f'{ref(frf, "frequency_resolution")}; within that band it '
            'resolves the channels whose coherence sits near one and '
            'whose signal-to-noise stands clear of zero, and it does '
            'not resolve the rest. A controller inverting this plant '
            'will be as good as the channels it was measured on, so '
            'any channel marked out by the coherence, the '
            'signal-to-noise or the kurtosis figures is one to fix '
            'before running the test this identification was made '
            'for, rather than one to control through.'},
    ])


def transient_template(objects: Mapping[str, Any],
                       links: Sequence[Mapping[str, Any]] | None = None) -> Report:
    """A transient replication report: what was asked for, and what came
    back — two waveforms of the same thing.

    A transient test is judged differently from a random one, and the
    report has to be shaped that way. There is no band to fall outside
    and no share of the spectrum to count: there is a target waveform, a
    measured one, and the difference between them. So the specification
    and the response go on the same axes, at one control channel at a
    time, and the reader compares them by eye — which is what everyone
    does with a shock trace anyway.

    Deliberately shorter than the random report. There is no settled
    numerical comparison for two transients — RMS error over the event,
    peak error, an SRS of each — and inventing one here would be
    inventing a convention rather than reporting a measurement. The
    plots come first; the numbers can follow once they are agreed.
    """
    time = '@basis:TimeHistory'

    def ref(name: str, field: str) -> str:
        return '{{' + f'{name}.{field}' + '}}'

    return Report('Transient Test Report', [
        {'kind': 'text', 'text':
            '## Test Summary\n\n'
            'A transient replication test was run: the controller was '
            'given a waveform and asked to reproduce it at the control '
            'points, sample by sample, by inverting the structure\'s '
            'own response. That is what separates it from a shock '
            'test, which names a required response spectrum and leaves '
            'the waveform to the machine. This report documents the '
            'instrumentation, the waveform that was asked for, what '
            'the article actually saw, and the two readings of how '
            'far apart those are.\n\n'
            f'The run was recorded on {ref(time, "num_channels")} '
            f'channels at {ref(time, "sample_rate")} over '
            f'{ref(time, "duration")}.'},
        {'kind': 'text', 'text': instrumentation_text(objects, links)},
        *front_matter(objects, links, time),
        {'kind': 'text', 'text':
            '## Specification\n\nThe waveform the controller was asked '
            'to reproduce at each control point '
            '({{figure:Transient specification}}). A transient test '
            'names the motion itself, sample by sample — where a shock '
            'test names a response spectrum and leaves the waveform to '
            'the controller.'},
        {'kind': 'plot', 'source': '@basis:TransientSpecification',
         'mode': 'curves', 'caption': 'Transient specification'},
        {'kind': 'text', 'text':
            '## Control\n\nWhat the article actually did '
            '({{figure:Measured control response}}), against what was '
            'asked of it. The controller played the specification over '
            'and over, so the record is cut back into those repeats '
            'before anything is compared — the frame is the '
            "specification's own length, the repeats run back to back, "
            'and the window is rectangular, because a taper would '
            'distort the very waveform being replicated.\n\n'
            'The overlay ({{figure:First playing against the target '
            'waveform}}) puts one playing directly over the waveform '
            'it was aiming at, channel by channel. It is the only '
            'figure here that shows the comparison the way it is '
            'actually judged — by eye, sample against sample — and '
            'amplitude, phase and shape errors each look different on '
            'it in a way no single number distinguishes.\n\n'
            'The drive forces ({{figure:Drive forces}}) are what the '
            'controller had to produce to get that response. They are '
            'reported because a replication that succeeded by driving '
            'the shakers to their limit is one that will not repeat '
            'on a heavier article or a colder day, and because a '
            'drive that clipped is visible here and nowhere else.'},
        {'kind': 'plot', 'source': '@basis:TimeHistory', 'mode': 'curves',
         'select': 'dim:acceleration',
         'caption': 'Measured control response'},
        # the comparison itself: a playing over the waveform it was
        # aiming at, channel by channel — the app's overlay view
        {'kind': 'plot', 'mode': 'overlay',
         'source': '@basis:TimeHistory',
         'specification': '@basis:TransientSpecification',
         'caption': 'First playing against the target waveform'},
        {'kind': 'plot', 'source': '@basis:TimeHistory', 'mode': 'curves',
         'select': 'dim:force',
         'caption': 'Drive forces the controller produced'},
        {'kind': 'text', 'text':
            '## Where The Energy Is In Time\n\n'
            'The scalogram ({{figure:Control scalogram}}) reads the '
            'control response against time *and* frequency at once — '
            'time across, frequency receding, amplitude standing up '
            'out of the floor. A transient is the reading\'s natural '
            'subject: the spectra below say what frequencies the event '
            'contained and average away *when*, and for a shock\'s '
            'ring-down or a resonance the event excited on its way '
            'through, when is the story. One channel is drawn, the '
            'way the application\'s wavelet view reads one at a '
            'time; the lowest rows lean on the record\'s ends for '
            'the span the caption gives, because a low-frequency '
            'wavelet is long, and inside that span the picture is '
            'shaped by where the record was cut rather than by the '
            'event.'},
        {'kind': 'plot', 'source': '@basis:TimeHistory',
         'mode': 'scalogram', 'select': 'dim:acceleration',
         'caption': 'Control scalogram'},
        {'kind': 'text', 'text':
            '## Level\n\nThe comparison read as spectra '
            '({{figure:Control spectra against the specification}}). A '
            'PSD is already an average over the playings — it is taken '
            'over the very frames the events are — so there is one '
            'curve per channel here however many times the waveform '
            'was played.\n\nThis is the scale with shape and phase '
            'divided out ({{figure:RMS error by control channel}}): a '
            'channel can hold the right level and still follow the '
            'waveform badly, which is why the waveform reading follows '
            'and neither stands in for the other.'},
        {'kind': 'plot', 'source': '@basis:Psd',
         'specification': '@basis:Specification', 'mode': 'curves',
         'caption': 'Control spectra against the specification'},
        {'kind': 'bars', 'mode': 'error', 'source': '@basis:Specification',
         'measured': '@basis:Psd',
         'caption': 'RMS error by control channel'},
        {'kind': 'text', 'text':
            '## Replication\n\nHow far each control channel is from '
            'the waveform it was asked for '
            '({{figure:Waveform error by control channel}}): the '
            'difference between the two waveforms against the size of '
            'the target. Amplitude, phase and shape all move it, and '
            'it is what the controller is minimizing — the level above '
            'is the scale alone; this is everything at once.\n\nOne '
            'bar per channel per playing of the waveform, and none of '
            'them singled out — which repeat was the bad one depends '
            'on the reading you care about and on what the article is '
            'for, so they are all here. The threshold is a line for '
            'the eye and nothing more: there is no accepted tolerance '
            'on how closely a replicated transient must follow its '
            'target.'},
        {'kind': 'bars', 'mode': 'waveform',
         'source': '@basis:TransientSpecification',
         'measured': '@basis:TimeHistory',
         'caption': 'Waveform error by control channel'},
        kurtosis_block(),
        {'kind': 'text', 'text':
            '## The Shape Of The Record\n\n'
            'Pearson kurtosis ({{figure:Pearson kurtosis}}) reads the '
            'record\'s distribution rather than its level or its '
            'shape in time — three is Gaussian. A replicated transient '
            'is not Gaussian and is not meant to be: it is a waveform '
            'played over and over, so the reading sits well above '
            'three and is expected to. What it is watched for is '
            'change: a channel far from its neighbors, or a run far '
            'from the last one, is a channel where something clipped '
            'or rattled rather than reproduced.'},
        {'kind': 'text', 'text':
            '## Conclusions\n\n'
            'How closely the run reproduced its specification is read '
            'two ways, and both are here because they answer '
            'different questions. {{figure:RMS error by control '
            'channel}} is the level with shape and phase divided out: '
            'a channel can hold exactly the right level and still '
            'follow the waveform badly. {{figure:Waveform error by '
            'control channel}} is everything at once — amplitude, '
            'phase and shape — which is what the controller was '
            'minimizing, and it is reported per playing because which '
            'repeat was the bad one depends on what the article is '
            'for.\n\n'
            'There is no accepted tolerance on how closely a '
            'replicated transient must follow its target, so no line '
            'here is a pass mark: the threshold on the waveform chart '
            'is a line for the eye. What the figures support is a '
            'comparison — between channels, between playings, and '
            'against whatever was accepted the last time this article '
            'was run.'},
    ])


def control_channels_of(spec: Any) -> list[str]:
    """The control channels a specification names, as the comparison
    figure labels them: its autospectra, in its own order. A cross
    term has no measured response to stand against it. Empty for
    anything that is not a specification."""
    if spec is None or not hasattr(spec, 'record_pair'):
        return []
    labels: list[str] = []
    for i in range(spec.num_records):
        response, reference = spec.record_pair(i)
        if response == reference and response not in labels:
            labels.append(response)
    return labels


def control_channel_labels(objects: Mapping[str, Any],
                           links: Sequence[Mapping[str, Any]] | None,
                           token: str = '@basis:Specification') -> list[str]:
    """The control channels the specification a token binds names, as
    the comparison figure labels them — each its own figure in the
    random report (Brandon, 2026-09-18: one figure per control
    channel rather than one figure with a drop-down). Empty when the
    token binds nothing yet, and the template writes one figure that
    picks."""
    name = resolve_binding(token, objects, links)
    return control_channels_of(objects[name] if name and name in objects
                               else None)


#: above this many control channels the random report lays their
#: figures out as a grid — a row per node, a column per direction —
#: rather than one figure after another (Brandon, 2026-09-19); four
#: still read as a sequence
GRID_ABOVE = 4

GRID_AXES = ('X', 'Y', 'Z')


def channel_grid(labels: Sequence[str],
                 geometry: Any = None) -> dict[str, Any]:
    """The grid the random report lays many control channels out on.

    A row per node, in the order the channels first name them; a
    column per direction. With a geometry that can place a channel
    (`Geometry.dof_direction`), the columns are the global axes —
    'Global X', 'Global Y', 'Global Z' — and a channel goes under the
    axis its measured direction is nearest, with how far off it sits
    noted when it is not on it (Brandon, 2026-09-19). Without one, or
    for a node the geometry cannot place, the columns are the DOF's
    own letters; a channel with no direction at all — a node number
    alone — stands in a 'Channel' column. Only the columns in use
    appear, in that order.

    Parameters
    ----------
    labels : sequence of str
        The control channels, as `control_channels_of` lists them.
    geometry : Geometry, optional
        The geometry the channels are measured on.

    Returns
    -------
    dict
        'columns', the column headings; 'rows', one dict per node with
        'label' and 'cells' — a list per column of the channels in it,
        each {'channel': str, 'note': str}.
    """
    import numpy as np

    from .data import parse_dof

    placed: list[tuple[str, tuple[int, int], str, str]] = []
    for label in labels:
        node, direction = parse_dof(label)
        row = f'Node {node}' if node is not None else str(label)
        letter = str(direction).upper().lstrip('R')[:1]
        vector = geometry.dof_direction(label) if geometry is not None else None
        note = ''
        if vector is not None:
            axis = int(np.argmax(np.abs(vector)))
            angle = float(np.degrees(np.arccos(min(1.0, abs(float(vector[axis]))))))
            column = (0, axis)
            if angle >= 0.5:
                note = f'{angle:.0f}° off global {GRID_AXES[axis]}'
        elif letter in GRID_AXES:
            column = (1, GRID_AXES.index(letter))
        else:
            column = (2, 0)
        placed.append((row, column, str(label), note))
    columns = sorted({column for _row, column, _label, _note in placed})
    names = {0: 'Global {}', 1: '{}', 2: 'Channel'}
    headings = [names[kind].format(GRID_AXES[axis] if kind < 2 else '')
                for kind, axis in columns]
    rows: list[dict[str, Any]] = []
    for row, column, label, note in placed:
        entry = next((r for r in rows if r['label'] == row), None)
        if entry is None:
            entry = {'label': row, 'cells': [[] for _ in columns]}
            rows.append(entry)
        entry['cells'][columns.index(column)].append(
            {'channel': label, 'note': note})
    return {'columns': headings, 'rows': rows}


def _figures_text(labels: Sequence[str], sequence: str, grid: str) -> str:
    """How the prose describes the control figures: one after another
    up to `GRID_ABOVE` channels, a grid above."""
    return grid if len(labels) > GRID_ABOVE else sequence


def prune_unbound(report: Report, objects: Mapping[str, Any],
                  links: Sequence[Mapping[str, Any]] | None = None
                  ) -> Report:
    """Drop what the project cannot fill, once it holds anything
    (Brandon, 2026-09-19): every figure block that fails to bind, and
    every section whose figures all did.

    A section is a text block opening with a `## ` heading and what
    follows it up to the next; its figures are the blocks in it that
    are not text. When none of them can resolve — the section's
    objects are simply not in the project — the whole section goes,
    heading and prose included, rather than a heading over nothing;
    when some can, the ones that cannot go on their own, so the editor
    shows no unbound cards for a run that has no geometry. A section
    with no figures (the summary, the conclusions) is prose and stays.
    An empty project keeps the whole outline: that is the template
    being read, not a report being written.
    """
    if not objects:
        return report
    keys = ('source', 'geometry', 'dofs_source', 'shapes', 'measured',
            'specification')

    def bound(block):
        needed = [block.get(key) for key in keys if block.get(key)]
        return all(resolve_binding(name, objects, links) in objects
                   for name in needed)

    def heading(block):
        return (block.get('kind') == 'text'
                and block.get('text', '').lstrip().startswith('## '))

    sections: list[list[int]] = []
    for i, block in enumerate(report.blocks):
        if heading(block) or not sections:
            sections.append([])
        sections[-1].append(i)
    doomed = []
    for section in sections:
        figures = [i for i in section
                   if report.blocks[i].get('kind') != 'text']
        if figures and not any(bound(report.blocks[i]) for i in figures):
            doomed.extend(section)
        else:
            doomed.extend(i for i in figures if not bound(report.blocks[i]))
    report.remove(doomed)
    return report


def _per_channel(token: str, caption: str, objects: Mapping[str, Any],
                 links: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """A specification's own figures, one per control channel — no
    drop-down anywhere in the random report (Brandon, 2026-09-18);
    one figure that picks when nothing binds yet."""
    labels = control_channel_labels(objects, links, token)
    if not labels:
        return [{'kind': 'plot', 'source': token, 'mode': 'curves',
                 'caption': caption}]
    if len(labels) > GRID_ABOVE:
        # many channels read as a grid, a row per node and a column per
        # direction, one figure (Brandon, 2026-09-19)
        return [{'kind': 'plot', 'source': token, 'mode': 'curves',
                 'grid': True, 'caption': caption}]
    return [{'kind': 'plot', 'source': token, 'mode': 'curves',
             'channel': label, 'caption': f'{caption} — {label}'}
            for label in labels]


def _comparisons(psd_token: str, spec_token: str, caption: str,
                 objects: Mapping[str, Any],
                 links: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """The control-against-specification figures: one per control
    channel the bound specification names, each opening on its own
    channel; one figure that picks when nothing binds yet."""
    labels = control_channel_labels(objects, links, spec_token)
    if not labels:
        return [{'kind': 'plot', 'source': psd_token,
                 'specification': spec_token, 'mode': 'curves',
                 'caption': caption}]
    if len(labels) > GRID_ABOVE:
        return [{'kind': 'plot', 'source': psd_token,
                 'specification': spec_token, 'mode': 'curves',
                 'grid': True, 'caption': caption}]
    return [{'kind': 'plot', 'source': psd_token,
             'specification': spec_token, 'mode': 'curves',
             'channel': label, 'caption': f'{caption} — {label}'}
            for label in labels]


def _random_control_blocks(objects: Mapping[str, Any],
                           links: Sequence[Mapping[str, Any]] | None
                           ) -> list[dict[str, Any]]:
    """The random half's judgment: the control spectra against the
    specification, the two compliance readings, and the same on octave
    bands — shared by the random report and the random-and-sine one, so
    the two cannot drift.

    The specification is not drawn on its own, in either form. It is
    the shaded band behind every comparison, and a figure of the
    requirement with nothing measured against it was a page the reader
    had already seen (Brandon, 2026-09-21).
    """
    controls = control_channel_labels(objects, links)
    return [
        {'kind': 'text', 'text':
            '## Control\n\nThe measured control spectra against the '
            'specification, '
            + _figures_text(
                controls,
                'one figure per control channel '
                '({{figure:Control against specification}} and following). ',
                'laid out by control channel — a row per node, a column '
                'per direction ({{figure:Control against specification}}). ')
            + 'Each carries the requirement with it — the target '
            'drawn through the measurement, its warning and abort '
            'limits shaded around it — so the specification is read '
            'where the response is. Each opens on the '
            'specification\'s own frequency band; the measurement '
            'beyond it is a zoom away. Shading marks lines outside '
            'an abort limit — red above the upper, blue below the '
            'lower.'},
        *_comparisons('@basis:Psd', '@basis:Specification',
                      'Control against specification', objects, links),
        {'kind': 'text', 'text':
            '## Compliance\n\nThe same comparison read two ways, a bar '
            'per control channel. **RMS error** '
            '({{figure:RMS error by control channel}}) is the level: '
            'how far each channel sits from what was asked for. '
            '**Lines outside abort** '
            '({{figure:Band outside the abort limits, by control '
            'channel}}) is the shape: a channel can sit at exactly the '
            'right level and still be out of tolerance across half its '
            'band. The thresholds drawn on both are a tolerance '
            'someone chose — plus or minus three decibels and a tenth '
            'of the band are where most specifications land, not '
            'where they all do — so each chart is read against the '
            'line it carries rather than as a pass mark of its '
            'own.'},
        {'kind': 'bars', 'mode': 'error', 'source': '@basis:Specification',
         'measured': '@basis:Psd',
         'caption': 'RMS error by control channel'},
        {'kind': 'bars', 'mode': 'lines', 'source': '@basis:Specification',
         'measured': '@basis:Psd',
         'caption': 'Band outside the abort limits, by control channel'},
        {'kind': 'text', 'text':
            '## Octave Band Comparison\n\nThe same comparison on '
            'octave bands: the control spectra integrated onto bands '
            '({{figure:Control against specification, octave bands}}'
            + _figures_text(controls,
                            ' and following, one per control channel) ',
                            ', by node and direction) ')
            + 'against the specification banded the same way, its '
            'warning and abort limits banded with it, and the same '
            'two readings of them. The bands are wider as the frequency '
            'rises, which is how a response is usually specified and '
            'how it is usually read.\n\n'
            'Banding conserves the area under each curve, so the RMS '
            'error below is the same number as above; what changes is '
            'the share of the band outside abort, which is counted '
            'over bands rather than over lines, against limits that '
            'are themselves per band.'},
        # the banded objects themselves, not the report banding the
        # narrowband ones for itself (which `'octave': N` on a block
        # still does, for a template that asks): the comparison is
        # read between the two objects the project holds (Brandon,
        # 2026-09-18). The banded specification is drawn only there,
        # as the narrowband one is — see above.
        *_comparisons('@basis:OctavePsd', '@basis:OctaveSpecification',
                      'Control against specification, octave bands',
                      objects, links),
        {'kind': 'bars', 'mode': 'error',
         'source': '@basis:OctaveSpecification',
         'measured': '@basis:OctavePsd',
         'caption': 'RMS error by control channel, octave bands'},
        {'kind': 'bars', 'mode': 'lines',
         'source': '@basis:OctaveSpecification',
         'measured': '@basis:OctavePsd',
         'caption': 'Band outside the abort limits, by control channel, '
                    'octave bands'},
    ]


def random_template(objects: Mapping[str, Any], links: Sequence[Mapping[str, Any]] | None = None) -> Report:
    """A random vibration report: what was asked for, what arrived, and
    by how much they differ.

    Ordered the standing way (Brandon, 2026-08-23): the front matter —
    geometry, the DOF scenes excitation-first, photographs, the
    channel table — then the time data, then the spectra, then the
    computed judgments. Within the spectra the specification still
    leads: it is what the article was required to see, and the control
    PSDs against it are the whole claim. Coherence stays last among
    the judgments — it is how you tell a real exceedance from a bad
    channel, a question asked only once the comparison has raised it.

    Sources bind symbolically, like the modal template's, so the report
    finds its objects under the names the typed project's tree
    suggested, and every unbound slot is invisible — a run with no
    geometry still renders.
    """
    time = '@basis:TimeHistory'
    psd = '@basis:Psd'
    coherence = '@basis:MultipleCoherence'

    def ref(name: str, field: str) -> str:
        return '{{' + f'{name}.{field}' + '}}'

    def present(token: str) -> bool:
        """Whether the prose may speak of what a token binds: yes in
        an empty project (the outline), otherwise only when it binds.
        A reference to what is not there stays on the page as written
        — the page's rule for an unmatched reference, right for an
        author and wrong for a reader (Brandon, 2026-09-19) — so the
        prose says only what the project holds."""
        return not objects or resolve_binding(token, objects, links) in objects

    return prune_unbound(Report('Random Vibration Test Report', [
        # the verdict first, under the title: whether the environment
        # passed, read off the octave-band comparison, so a reader
        # opening the report knows before reading a word (Brandon,
        # 2026-09-20)
        {'kind': 'verdict', 'source': '@basis:OctaveSpecification',
         'measured': '@basis:OctavePsd'},
        {'kind': 'text', 'text':
            '## Test Summary\n\n'
            'A random vibration test was run: the article was driven '
            'with a broadband random excitation held to a required '
            'power spectral density at the control channels. This '
            'report documents the instrumentation, the record the '
            'analysis was made from, the specification the run was '
            'controlled to, how closely each control channel held it, '
            'and the checks that say whether those numbers can be '
            'trusted.\n\n'
            f'The run was recorded on {ref(time, "num_channels")} '
            f'channels at {ref(time, "sample_rate")} over '
            f'{ref(time, "duration")}. The control spectra are '
            f'averaged over {ref(time, "num_frames")} frames of '
            f'{ref(time, "frame_length")} samples with a '
            f'{ref(time, "window")} window at {ref(time, "overlap")} '
            'overlap'
            + (f', giving {ref(psd, "frequency_resolution")} resolution '
               f'out to {ref(psd, "max_frequency")}.' if present(psd)
               else '.')},
        {'kind': 'text', 'text': instrumentation_text(objects, links)},
        *front_matter(objects, links, time),
        {'kind': 'text', 'text':
            '## Measured Data\n\nThe measured time histories, one '
            'figure per quantity, with the frames a PSD is averaged '
            'over marked on them ({{figure:Measured}}) — which part '
            'of the run was analyzed, and under what window.'},
        *time_data_blocks(
            time, objects, links,
            'Measured time histories, with the frames the spectra '
            'are averaged over',
            'Measured {quantity} time histories, with the frames '
            'the spectra are averaged over'),
        *_random_control_blocks(objects, links),
        {'kind': 'text', 'text':
            '## Data Quality\n\n'
            + ('Two checks stand behind every number above, and a '
               'channel that fails either is one whose exceedance may '
               'be the measurement rather than the article.\n\n'
               if present(coherence) else
               'One check stands behind every number above, and a '
               'channel that fails it is one whose exceedance may be '
               'the measurement rather than the article.\n\n')
            + ('Multiple coherence ({{figure:Multiple coherence}}) says '
               'how much of each response the drives account for. At '
               'one the channel moved because the shakers moved it; '
               'well below one something else did, and a control '
               'channel in that state was being held to a level it '
               'was not wholly responsible for.\n\n'
               if present(coherence) else '')
            + 'Pearson kurtosis ({{figure:Pearson kurtosis}}) says '
            'whether the excitation had the shape a random test '
            'assumes. A spectrum cannot answer this: two records with '
            'identical densities can be a smooth hiss and a train of '
            'rare hard peaks, and the second fatigues an article in a '
            'way the first does not. Three is Gaussian. Above the band '
            'the record carried peaks the specification never asked '
            'for; below it the record was clipped, or was not random. '
            'Either way the article saw something other than the '
            'specified test while the spectrum looked correct.'},
        # the stage, not curves: a dozen channels of coherence stacked
        # on one 2-D axis is a thicket. The map answered that first
        # (frequency across, channel down); the stage answers it the
        # way the app now does, spreading the channels in depth and
        # keeping each one's own trace readable (Brandon, 2026-08-23)
        {'kind': 'plot', 'source': '@basis:MultipleCoherence',
         'mode': 'stage', 'caption': 'Multiple coherence'},
        kurtosis_block(),
        # no figure references here (Brandon, 2026-09-19): the
        # conclusions stand whatever the project holds, and a reference
        # to a figure a thin project does not have would stay on the
        # page as written — the page's rule for an unmatched reference,
        # right for an author and wrong for a reader
        {'kind': 'text', 'text':
            '## Conclusions\n\n'
            'Whether the run met its specification is read off the '
            'control comparison and the two compliance charts under '
            'it: the RMS error says how far each channel sat from the '
            'level it was asked for, and the band outside abort how '
            'much of each channel\'s band fell outside tolerance. A '
            'channel can pass one and fail the other — a channel 2 dB '
            'low everywhere may never cross an abort limit, and one at '
            'exactly the right level may be out across half its band — '
            'which is why both are there. The octave-band figures read '
            'the same run the way a requirement is usually written and '
            'usually argued.\n\n'
            'Where a channel is out, the two data-quality figures are '
            'what decide whether the article or the measurement is at '
            'fault. A channel clean on both and still outside '
            'tolerance is a real exceedance.'},
    ]), objects, links)


def mixed_template(objects: Mapping[str, Any],
                   links: Sequence[Mapping[str, Any]] | None = None) -> Report:
    """A random-and-sine report: both halves of a run that was both.

    A random environment held to its PSD with a sweep riding under it
    — the qualification-with-a-tracked-tone case, and the only way a
    sweep has been run here. The two halves are judged by their own
    workflows, and the report holds both judgments in the standing
    order: the front matter once, the recording once, then the random's
    control comparisons and compliance (`_random_control_blocks`), then
    the sine's levels and deviation (`_sine_level_blocks`), then the
    data-quality readings and the conclusions, each written for a run
    that was both. Nothing here is a third copy of either half: the
    judgment blocks are the very functions the two single-environment
    reports call (Brandon, 2026-09-24).

    Two things a reader of one half's report would not have to know are
    said here. The control spectra are of the whole recording, sweep
    included, so a tone adds its power across the band it swept and a
    channel reading high there may be the tone rather than the random.
    And the recording is not Gaussian by design: a pure tone's kurtosis
    is 1.5, so a channel under three is the sweep, not clipping.
    """
    time = '@basis:TimeHistory'
    psd = '@basis:Psd'
    coherence = '@basis:MultipleCoherence'

    def ref(name: str, field: str) -> str:
        return '{{' + f'{name}.{field}' + '}}'

    def present(token: str) -> bool:
        return not objects or resolve_binding(token, objects, links) in objects

    return prune_unbound(Report('Random and Sine Test Report', [
        # the verdict is the random half's, read off the octave-band
        # comparison as the random report reads it; the sine half is
        # judged tone by tone in its own figures and bars below. So is
        # the level beside it, and it says so (Brandon, 2026-09-26): the
        # sine is compared as measured, never scaled
        {'kind': 'verdict', 'source': '@basis:OctaveSpecification',
         'measured': '@basis:OctavePsd',
         'level_label': 'Random test level'},
        {'kind': 'text', 'text':
            '## Test Summary\n\n'
            'A random vibration test was run with a sine sweep under '
            'it: the article was driven with a broadband random '
            'excitation held to a required power spectral density at '
            'the control channels, while one or more tones swept across '
            'a frequency range at a controlled amplitude on the same '
            'shakers. The two are judged separately here — the random '
            'against its specification, each tone against the level it '
            'was required to hold — because they are two requirements '
            'met by one recording. This report documents the '
            'instrumentation, the record both analyses were made from, '
            'how closely the control channels held the random '
            'specification, the level each tone actually reached '
            'against what it was required to reach, and the checks that '
            'say whether those numbers can be trusted.\n\n'
            f'The run was recorded on {ref(time, "num_channels")} '
            f'channels at {ref(time, "sample_rate")} over '
            f'{ref(time, "duration")}. The control spectra are '
            f'averaged over {ref(time, "num_frames")} frames of '
            f'{ref(time, "frame_length")} samples with a '
            f'{ref(time, "window")} window at {ref(time, "overlap")} '
            'overlap'
            + (f', giving {ref(psd, "frequency_resolution")} resolution '
               f'out to {ref(psd, "max_frequency")}.' if present(psd)
               else '.')},
        {'kind': 'text', 'text': instrumentation_text(objects, links)},
        *front_matter(objects, links, time),
        {'kind': 'text', 'text':
            '## Measured Data\n\nThe measured time histories, one '
            'figure per quantity, with the frames a PSD is averaged '
            'over marked on them ({{figure:Measured}}) — which part '
            'of the run was analyzed for the random half, and under '
            'what window. The sweep is read from the same record, '
            'sample by sample along each tone\'s own trajectory rather '
            'than by frames, so its faults show here first: a tone that '
            'lost control, a dwell that sat too long, an amplitude that '
            'stepped rather than swept.'},
        *time_data_blocks(
            time, objects, links,
            'Measured time histories, with the frames the spectra '
            'are averaged over',
            'Measured {quantity} time histories, with the frames '
            'the spectra are averaged over'),
        *_random_control_blocks(objects, links),
        {'kind': 'text', 'text':
            '## The Sweep\n\n'
            'The sine half of the same run. The control spectra above '
            'are of the whole recording, sweep included: a tone adds '
            'its power across the band it swept, so a control channel '
            'reading high there may be the tone rather than the random. '
            'The tone itself is judged below, along its own trajectory, '
            'with the random treated as the noise it is extracted '
            'from.'},
        *_sine_level_blocks(objects),
        {'kind': 'text', 'text':
            '## Data Quality\n\n'
            + ('Two checks stand behind every number above, and a '
               'channel that fails either is one whose exceedance may '
               'be the measurement rather than the article.\n\n'
               if present(coherence) else
               'One check stands behind every number above, and a '
               'channel that fails it is one whose exceedance may be '
               'the measurement rather than the article.\n\n')
            + ('Multiple coherence ({{figure:Multiple coherence}}) says '
               'how much of each response the drives account for. At '
               'one the channel moved because the shakers moved it; '
               'well below one something else did, and a control '
               'channel in that state was being held to a level it '
               'was not wholly responsible for. The sweep is driven by '
               'the same shakers, so it does not lower the coherence; '
               'what does is a channel moving on its own.\n\n'
               if present(coherence) else '')
            + 'Pearson kurtosis ({{figure:Pearson kurtosis}}) reads the '
            'distribution of each channel rather than its level. Three '
            'is Gaussian, which the random half alone would be; a pure '
            'tone reads 1.5, and a sweep mixed with a random background '
            'sits between, lower the louder the tone. So a channel '
            'under three is the sweep, by design, and what the figure is '
            'watched for is a channel far from its neighbors — one that '
            'clipped or rattled — and a channel well above three, which '
            'carried peaks neither requirement asked for.'},
        {'kind': 'plot', 'source': '@basis:MultipleCoherence',
         'mode': 'stage', 'caption': 'Multiple coherence'},
        kurtosis_block(),
        {'kind': 'text', 'text':
            '## Conclusions\n\n'
            'Whether the random half met its specification is read off '
            'the control comparison and the two compliance charts under '
            'it: the RMS error says how far each channel sat from the '
            'level it was asked for, and the band outside abort how '
            'much of each channel\'s band fell outside tolerance. A '
            'channel can pass one and fail the other, which is why both '
            'are there; the octave-band figures read the same run the '
            'way a requirement is usually written. Where a control '
            'channel is out only across the band the sweep passed '
            'through, the tone is the first suspect.\n\n'
            'Whether the sweep met its requirement is read off the '
            'per-tone figures, where the extracted level spans exactly '
            'the frequencies the sweep reached, and off the sine '
            'deviation bars, tone by tone and channel by channel.\n\n'
            'Where either half is out, the data-quality figures decide '
            'whether the article or the measurement is at fault, read '
            'knowing that a kurtosis under three is the sweep. A channel '
            'clean on both and still outside tolerance is a real '
            'exceedance.'},
    ]), objects, links)


#: every built-in report template by the key `Project.generate_report`
#: takes — the one list the project, the menus and the tests read
TEMPLATE_BUILDERS = {'modal': modal_template, 'random': random_template,
                     'shock': shock_template, 'transient': transient_template,
                     'sine': sine_template, 'mixed': mixed_template,
                     'sysid': sysid_template}
