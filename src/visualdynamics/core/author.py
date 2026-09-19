"""Writing and editing a specification, at any coordinates.

A specification is a sheet: breakpoints, a level per channel at each,
bands in decibels either side, and a cross term per pair of channels.
None of that depends on where the channels are, so one draft serves
every door in (Brandon, 2026-09-04: a generic editor, not a modal
one) — new at a shape set's modal coordinates, new at chosen physical
DOFs, or opened from a specification that already exists — and every
way out: a new object, or the one it was opened from, replaced.

The virtual point arc's third tool is one of those doors. Written at
`M1 … Mn` and expanded through the set, a specification becomes an
exact control-channel CPSD, phase and coherence included, with no
assumption in it: every number came from something the author
stated.

What an author states, per `SpecificationDraft`:

- **Channels**, each with the quantity its density is of.
- **Breakpoints**: frequencies and a level per channel at each, the
  power law between breakpoints being how a written specification
  reads (`Specification.interpolation = 'log_log'`).
- **Every pair's cross term**, as a coherence and a phase, from which
  `S_ij = √(γ² S_ii S_jj) e^{iφ}`. A pair left unstated is a refusal
  to make the specification, not a zero: independent channels are a
  statement too, and the author makes it (Brandon, 2026-09-04: no
  assumptions about cross terms, ever). `with_all_pairs` states one
  answer for every pair in one gesture.
- **Bands**: warning and abort, as decibels either side of the
  target, the same on every channel — which is what lets them carry
  exactly through a transform.

The operations on a draft (`scaled_db`, `with_bands`,
`with_breakpoint`, `without_breakpoint`, `with_all_pairs`) each give
a new draft and leave this one alone, so a script reads as a sequence
of statements and a session journals as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..units import UNKNOWN
from .transform import MOTIONS, _modal_dim, modal_dofs

if TYPE_CHECKING:                                    # pragma: no cover
    from collections.abc import Sequence

    from .channel_table import ChannelTable
    from .data import Specification
    from .shapes import ShapeSet

#: the bands a starter draft wears, in decibels below and above the
#: target — the customary ±3 dB warning and ±6 dB abort
WARNING_DB = (-3.0, 3.0)
ABORT_DB = (-6.0, 6.0)
#: a starter's breakpoints and level: a flat line across the usual
#: random band, at a round number in the channel's SI density, for the
#: author to pull into shape
STARTER_FREQUENCIES = (20.0, 2000.0)
STARTER_LEVEL = 1.0
#: how far a specification's own cross terms or bands may wander over
#: frequency and still be read as one coherence, one phase, one band
CONSTANT_TOLERANCE = 1e-6


@dataclass
class SpecificationDraft:
    """A specification being written, at any channels.

    Attributes:
        channels: The DOF strings, one per channel — modal coordinates
            (`M1`), physical DOFs (`101Z+`), whatever the door opened
            on.
        dims: The quantity each channel's density is of
            ('acceleration', 'angular_acceleration', …), one per
            channel; the density's dimension is its square per hertz.
        frequencies: The breakpoints, in Hz, increasing.
        levels: One list per channel, a density at each breakpoint,
            in the channel's SI density unit.
        pairs: `{(i, j): (coherence, phase_degrees)}` for every pair
            `i < j` of channels, or None where the pair has not been
            stated yet.
        bands: One entry per channel: `{'warning': [(below, above)
            per segment], 'abort': [...], 'symmetric': bool, 'uniform':
            bool}`, the bands in decibels either side of the target,
            a segment being the span between two neighboring
            breakpoints. A channel's own constraints: `symmetric`
            holds above at minus below, `uniform` holds every segment
            at one band (Brandon, 2026-09-06: each channel its own
            settings, the bands dragged on the plot per section, not
            typed in a pane). A breakpoint belongs to the segment on
            its right, the last one to the segment on its left.
        notes: What a door could not carry over — cross terms that
            varied with frequency, bands that differed — said rather
            than silently dropped.
        sources: Which specification each channel was opened from, one
            per channel, when the sheet spans several (`across`);
            empty when it is one object's. Channels of different
            sources have no cross term between them — the pair is not
            offered rather than unstated.
        form: 'breakpoints' — the few points a specification is
            written from — or 'lines', the same requirement read onto
            evenly spaced frequency lines, as a controller writes its
            target. `to_breakpoints` and `to_lines` convert; the
            levels are edited on the breakpoints (Brandon,
            2026-09-06).
        spacing: The frequency spacing, in Hz, the lines form has or
            the sheet remembers from the specification it opened —
            what `to_lines` reads onto when not told otherwise.
    """

    channels: list[str]
    dims: list[str]
    frequencies: list[float]
    levels: list[list[float]]
    pairs: dict[tuple[int, int], tuple[float, float] | None] = field(
        default_factory=dict)
    bands: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    form: str = 'breakpoints'
    spacing: float | None = None

    def __post_init__(self) -> None:
        from .validate import dofs as _dofs

        # what the object accepts, the sheet accepts: a virtual point's
        # rows come in numbered '1', '2', '3' with no direction, since
        # the controller's file names them nowhere, and the sheet
        # refused to open on them (Brandon, 2026-09-18). Only an empty
        # name is refused — a channel has to be called something
        self.channels = _dofs(list(self.channels), 'channel',
                              allow_unknown=True)
        if any(not name for name in self.channels):
            raise ValueError('every channel needs a name')
        self.dims = [str(d) for d in self.dims]
        self.frequencies = [float(f) for f in self.frequencies]
        self.levels = [[float(v) for v in row] for row in self.levels]
        self.pairs = {(int(i), int(j)): (None if value is None else
                                         (float(value[0]), float(value[1])))
                      for (i, j), value in self.pairs.items()}
        segments = max(len(self.frequencies) - 1, 0)
        if not self.bands:
            self.bands = [default_bands(segments) for _ in self.channels]
        self.bands = [_normal_bands(entry) for entry in self.bands]
        self.notes = [str(n) for n in self.notes]
        self.sources = [str(s) for s in self.sources]
        self.form = str(self.form)
        self.spacing = None if self.spacing is None else float(self.spacing)
        self.validate()

    # ---- the doors in ---------------------------------------------------------

    @classmethod
    def at_modal_coordinates(cls, shapes: ShapeSet,
                             quantity: str = 'acceleration',
                             modes: Sequence[int] | None = None
                             ) -> SpecificationDraft:
        """A starter at a shape set's modal coordinates `M1 … Mn`, each
        channel's quantity by the transform's unit rule — a unit rigid
        set's rotations per radian, a mass-normalized set's modal
        quantity. A set with no mass unit gives its coordinates no
        unit, and a specification in nothing is refused.

        Parameters
        ----------
        shapes : ShapeSet
            The set the specification is written at.
        quantity : str, default 'acceleration'
            The physical quantity the target is a density of.
        modes : sequence of int, optional
            Which shapes get a channel, by index from 0 — the ones
            picked in the tree. All of them when omitted. A virtual
            point's target is usually its three translations alone
            (Brandon, 2026-09-06): expanded through the set, a shape
            with no channel contributes nothing, which is what a zero
            would say, and a sheet holds positive levels.

        Returns
        -------
        SpecificationDraft
            Every pair unstated.
        """
        which = (list(range(shapes.num_shapes)) if modes is None
                 else [int(k) for k in modes])
        if not which:
            raise ValueError('no shape picked to write a channel at')
        out_of_range = [k for k in which if not 0 <= k < shapes.num_shapes]
        if out_of_range:
            raise ValueError(f'the set has {shapes.num_shapes} shapes; no '
                             f'shape {out_of_range[0] + 1}')
        names = modal_dofs(shapes)
        dims = []
        for k in which:
            dim, _hint = _modal_dim(shapes, quantity, k)
            if dim == UNKNOWN:
                raise ValueError(
                    'the shape set has no mass unit, so its modal '
                    'coordinates have no unit and a specification written '
                    'at them would be in nothing — declare the set\'s mass '
                    'unit first (Define Units)')
            dims.append(dim)
        return cls.at_dofs([names[k] for k in which], dims)

    @classmethod
    def at_dofs(cls, dofs: list[str], quantity: str | list[str]
                = 'acceleration') -> SpecificationDraft:
        """A starter at chosen DOFs: a flat line across 20–2000 Hz at
        one SI unit of density on every channel, every pair unstated.

        Parameters
        ----------
        dofs : list of str
            The channels.
        quantity : str or list of str, default 'acceleration'
            The quantity each channel's density is of, one for all or
            one per channel.

        Returns
        -------
        SpecificationDraft
            The sheet to fill.
        """
        n = len(dofs)
        dims = [quantity] * n if isinstance(quantity, str) else list(quantity)
        return cls(list(dofs), dims, list(STARTER_FREQUENCIES),
                   [[STARTER_LEVEL] * len(STARTER_FREQUENCIES)
                    for _ in range(n)],
                   {(i, j): None for i in range(n) for j in range(i + 1, n)})

    @classmethod
    def at_control_channels(cls, table: ChannelTable) -> SpecificationDraft:
        """A starter at a channel table's control channels — every
        motion channel when none is flagged — each in the quantity
        its channel type declares.

        Parameters
        ----------
        table : ChannelTable
            The table.

        Returns
        -------
        SpecificationDraft
            The sheet to fill.
        """
        dofs, types = table.dof_strings(), table.types()
        flagged = table.controls()
        rows = [i for i in range(len(dofs))
                if (flagged[i] if flagged.any() else True)
                and types[i] in MOTIONS and dofs[i]]
        if not rows:
            raise ValueError('the channel table has no motion channel to '
                             'write a specification at — declare the '
                             'channel types, or flag the control channels')
        return cls.at_dofs([dofs[i] for i in rows], [types[i] for i in rows])

    @classmethod
    def from_specification(cls, spec: Specification,
                           channels: list[str] | None = None,
                           source: str = '') -> SpecificationDraft:
        """The sheet a specification already is: its autospectra's lines
        as breakpoints, its cross terms as a coherence and phase where
        they hold one over frequency, its bands as decibels where they
        do — and a note for whatever did not fit the sheet.

        Parameters
        ----------
        spec : Specification
            The specification to open.
        channels : list of str, optional
            Only these of its channels — the ones picked in the tree.
            All of them when omitted.
        source : str, default ''
            The specification's name, for a sheet that will span
            several (`across`).

        Returns
        -------
        SpecificationDraft
            Ready to edit; `written_into` writes it back, `make` makes
            a new specification of it alone.
        """
        from .data import channel_quantities

        references = spec.reference_dof or list(spec.response_dof)
        autos = [i for i in range(spec.num_records)
                 if references[i] == spec.response_dof[i]]
        if channels is not None:
            wanted = set(channels)
            missing = wanted - {spec.response_dof[i] for i in autos}
            if missing:
                raise ValueError(f'{min(missing)} has no autospectrum in the '
                                 'specification')
            autos = [i for i in autos if spec.response_dof[i] in wanted]
        if not autos:
            raise ValueError('the specification holds no autospectrum to '
                             'open as a sheet')
        channels = [spec.response_dof[i] for i in autos]
        dims = [channel_quantities(spec.known_dim(i))[0] for i in autos]
        notes = []
        # a sheet holds positive levels at positive frequencies; a
        # controller's target starts at 0 Hz and rolls off to nothing at
        # the edges, and those lines are put aside rather than refused
        full = np.real(spec.ordinate[autos])
        lit = (np.asarray(spec.abscissa, dtype=float) > 0) & np.all(
            np.isfinite(full) & (full > 0), axis=0)
        if lit.sum() < 2:
            raise ValueError('the specification has fewer than two lines '
                             'where every autospectrum is positive')
        if not lit.all():
            dropped = int((~lit).sum())
            notes.append(f'{dropped} line{"s" * (dropped != 1)} at 0 Hz or '
                         'with a zero level left out — a sheet holds '
                         'positive levels at positive frequencies')
        frequencies = [float(f) for f in np.asarray(spec.abscissa)[lit]]
        target = full[:, lit]
        levels = [[float(v) for v in row] for row in target]
        n = len(autos)
        pairs: dict[tuple[int, int], tuple[float, float] | None] = {}
        by_pair = {(spec.response_dof[i], references[i]): i
                   for i in range(spec.num_records)}
        varying = []
        for a in range(n):
            for b in range(a + 1, n):
                i = by_pair.get((channels[a], channels[b]))
                j = by_pair.get((channels[b], channels[a]))
                if i is None and j is None:
                    pairs[(a, b)] = None
                    continue
                cross = (spec.ordinate[i] if i is not None
                         else np.conj(spec.ordinate[j]))[lit]
                with np.errstate(divide='ignore', invalid='ignore'):
                    coherence = np.abs(cross) ** 2 / (target[a] * target[b])
                phase = np.degrees(np.angle(cross))
                if not np.isfinite(coherence).any():
                    pairs[(a, b)] = (0.0, 0.0)
                    continue
                c, p = coherence, phase
                if np.allclose(c, c[0], atol=CONSTANT_TOLERANCE) and (
                        c[0] == 0.0 or np.allclose(p, p[0], atol=1e-6)):
                    pairs[(a, b)] = (float(min(max(c[0], 0.0), 1.0)),
                                     float(p[0]) if c[0] else 0.0)
                else:
                    pairs[(a, b)] = None
                    varying.append(f'{channels[a]}–{channels[b]}')
        if varying:
            notes.append(
                f'{len(varying)} cross term{"s" * (len(varying) != 1)} vary '
                'with frequency and cannot be one coherence and phase '
                f'({varying[0]}' + (f' and {len(varying) - 1} more'
                                    if len(varying) > 1 else '') + ')')
        # a specification with no limits opens wearing the defaults, so
        # the bands are on the plot to drag into being; `band_notes`
        # says which are not on the object yet
        # the bands, per channel and per segment: a segment runs from
        # one line to the next and takes the ratio at its first line,
        # since a line belongs to the segment on its right
        bands = []
        segments = max(len(frequencies) - 1, 0)
        for a, i in enumerate(autos):
            entry: dict[str, Any] = {}
            for kind, (fallback_low, fallback_high) in (
                    ('warning', WARNING_DB), ('abort', ABORT_DB)):
                sides = []
                for side, fallback in (('lower', fallback_low),
                                       ('upper', fallback_high)):
                    limit = spec.limits.get(f'{kind}_{side}')
                    if limit is None:
                        sides.append(np.full(segments, fallback))
                        continue
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = np.real(limit[i])[lit] / target[a]
                        decibels = 10 * np.log10(ratio)
                    decibels = np.where(np.isfinite(decibels) & (
                        (decibels < 0) if side == 'lower' else (decibels > 0)),
                        decibels, fallback)
                    # to the microdecibel: a limit stored as a ratio comes
                    # back as 6.000000000000001 on one line and
                    # 5.999999999999999 on the next, and read exactly
                    # that made every line its own band run — sixteen
                    # thousand handles on the drone target (2026-09-06)
                    sides.append(np.round(decibels[:segments], 6))
                entry[kind] = [(float(low), float(high))
                               for low, high in zip(sides[0], sides[1])]
            every = np.asarray(entry['warning'] + entry['abort'], dtype=float)
            entry['symmetric'] = bool(
                np.allclose(every[:, 0], -every[:, 1]) if len(every) else True)
            entry['uniform'] = all(
                np.allclose(entry[kind], entry[kind][0])
                for kind in ('warning', 'abort')) if segments else True
            # a constraint the object carries outranks what the ratios
            # happen to show: a channel whose bands still mirror after
            # Symmetric was switched off is not symmetric (2026-09-06)
            carried = (getattr(spec, 'band_constraints', None) or {}).get(
                spec.response_dof[i])
            if carried:
                entry['symmetric'] = bool(carried.get('symmetric',
                                                      entry['symmetric']))
                entry['uniform'] = bool(carried.get('uniform', entry['uniform']))
            # a rule read off the ratios holds to rounding; the bands
            # are settled onto it exactly, as the constraint will hold
            # them from here
            for kind in ('warning', 'abort'):
                found = entry[kind]
                if entry['uniform'] and found:
                    found[:] = [found[0]] * len(found)
                if entry['symmetric']:
                    found[:] = [(-abs(high), abs(high)) for _low, high in found]
            bands.append(entry)
        # a controller's target is written on evenly spaced lines; the
        # sheet remembers the spacing so the breakpoints it is edited
        # as can go back onto the same lines
        spacing = uniform_spacing(spec.abscissa)
        draft = cls(channels, dims, frequencies, levels, pairs, bands, notes,
                    [source] * len(channels) if source else [],
                    'lines' if spacing is not None and len(frequencies) > 2
                    else 'breakpoints', spacing)
        # A helper line — a hair before a breakpoint, carrying the left
        # segment's band up to it (`with_band_corners`) — is how the
        # object holds a band step, not a breakpoint anyone wrote. Read
        # as one it showed as a second row at 40 Hz and a second at
        # 200 Hz, and with Uniform off gave each sliver its own handle,
        # too thin to grab and left behind by a drag on the section
        # beside it (Brandon, 2026-09-06). Folded away: the merged
        # segment keeps the band the helper carried, which is the band
        # of the segment it started in, so nothing is lost — and `make`
        # writes the helper again if the step is still there.
        for frequency in _helper_lines(frequencies):
            draft = draft.without_breakpoint(frequency)
        return draft

    @classmethod
    def across(cls, drafts: dict[str, SpecificationDraft]
               ) -> SpecificationDraft:
        """One sheet spanning several specifications' sheets, so an
        edit — a scale, a band — lands on all of them at once.

        The sheets meet on one breakpoint list: the union of theirs
        over the range they share, each read off its own power law
        there (exact at its own breakpoints). Pairs are offered within
        a specification only. `for_source` gives each its own sheet
        back, and `written_into` writes it at its own lines.

        Parameters
        ----------
        drafts : dict of str to SpecificationDraft
            Each specification's sheet by its name.

        Returns
        -------
        SpecificationDraft
            The combined sheet, `sources` naming each channel's.
        """
        if not drafts:
            raise ValueError('no sheets to span')
        if len(drafts) == 1:
            (name, draft), = drafts.items()
            return draft._copy(sources=[name] * draft.num_channels)
        low = max(d.frequencies[0] for d in drafts.values())
        high = min(d.frequencies[-1] for d in drafts.values())
        union = sorted({f for d in drafts.values() for f in d.frequencies
                        if low <= f <= high})
        if len(union) < 2:
            raise ValueError('the specifications share fewer than two '
                             'breakpoints; open them one at a time')
        channels, dims, levels, sources, notes, bands = [], [], [], [], [], []
        pairs: dict[tuple[int, int], tuple[float, float] | None] = {}
        log_u = np.log(np.asarray(union))
        for name, draft in drafts.items():
            offset = len(channels)
            log_f = np.log(np.asarray(draft.frequencies))
            for row in draft.levels:
                levels.append([float(v) for v in np.exp(
                    np.interp(log_u, log_f, np.log(np.asarray(row))))])
            channels.extend(draft.channels)
            dims.extend(draft.dims)
            sources.extend([name] * draft.num_channels)
            bands.extend(draft._bands_on(union))
            for (i, j), value in draft.pairs.items():
                pairs[(i + offset, j + offset)] = value
            notes.extend(f'{name}: {note}' for note in draft.notes)
        forms = {d.form for d in drafts.values()}
        spacings = {d.spacing for d in drafts.values()}
        return cls(channels, dims, union, levels, pairs, bands, notes, sources,
                   'lines' if forms == {'lines'} and len(spacings) == 1
                   else 'breakpoints',
                   spacings.pop() if len(spacings) == 1 else None)

    def for_source(self, source: str) -> SpecificationDraft:
        """The sheet of one of the specifications a combined sheet
        spans — its channels and pairs alone, unnamed.

        Parameters
        ----------
        source : str
            The specification's name, as `sources` has it.

        Returns
        -------
        SpecificationDraft
            That specification's own sheet.
        """
        rows = [i for i, s in enumerate(self.sources) if s == source]
        if not rows:
            raise ValueError(f'the sheet has no channel of {source!r}')
        index = {old: new for new, old in enumerate(rows)}
        return self._copy(
            channels=[self.channels[i] for i in rows],
            dims=[self.dims[i] for i in rows],
            levels=[list(self.levels[i]) for i in rows],
            bands=[_normal_bands(self.bands[i]) for i in rows],
            pairs={(index[i], index[j]): value
                   for (i, j), value in self.pairs.items()
                   if i in index and j in index},
            notes=[n[len(source) + 2:] for n in self.notes
                   if n.startswith(f'{source}: ')],
            sources=[])

    # ---- what the draft is ---------------------------------------------------

    @property
    def num_channels(self) -> int:
        """How many channels the draft covers."""
        return len(self.channels)

    def validate(self) -> None:
        """Refuse what no specification could hold: unordered or
        repeated breakpoints, a level that is not positive, a
        coherence outside [0, 1], a band the wrong way round."""
        n = self.num_channels
        if not n:
            raise ValueError('a specification needs at least one channel')
        if self.sources and len(self.sources) != n:
            raise ValueError(f'{len(self.sources)} sources for {n} channels')
        if len(set(zip(self._sources(), self.channels))) != n:
            raise ValueError('a channel is named twice')
        if len(self.dims) != n:
            raise ValueError(f'{len(self.dims)} quantities for {n} channels')
        for dim in self.dims:
            if dim == UNKNOWN or not dim:
                raise ValueError('every channel needs a quantity')
        if len(self.frequencies) < 2:
            raise ValueError('a specification needs at least two breakpoints')
        if any(f <= 0 for f in self.frequencies):
            raise ValueError('breakpoint frequencies must be positive')
        if any(b <= a for a, b in zip(self.frequencies, self.frequencies[1:])):
            raise ValueError('breakpoint frequencies must increase')
        if len(self.levels) != n:
            raise ValueError(f'{len(self.levels)} level rows for {n} channels')
        for name, row in zip(self.channels, self.levels):
            if len(row) != len(self.frequencies):
                raise ValueError(f'{name} has {len(row)} levels for '
                                 f'{len(self.frequencies)} breakpoints')
            if any(not np.isfinite(v) or v <= 0 for v in row):
                raise ValueError(f'{name} needs a positive level at every '
                                 'breakpoint')
        offered = set(self.pairable())
        for (i, j), value in self.pairs.items():
            if not 0 <= i < j < n:
                raise ValueError(f'no pair ({i}, {j}) among {n} channels')
            if (i, j) not in offered:
                raise ValueError(
                    f'{self.channels[i]} of {self.sources[i]} and '
                    f'{self.channels[j]} of {self.sources[j]} have no cross '
                    'term: they are in different specifications')
            if value is not None and not 0.0 <= value[0] <= 1.0:
                raise ValueError(
                    f'the coherence of {self.channels[i]} and '
                    f'{self.channels[j]} is {value[0]:g}; a coherence lies '
                    'in [0, 1]')
        segments = len(self.frequencies) - 1
        if len(self.bands) != n:
            raise ValueError(f'{len(self.bands)} band entries for {n} channels')
        for name, entry in zip(self.channels, self.bands):
            for kind in ('warning', 'abort'):
                pairs = entry[kind]
                if len(pairs) != segments:
                    raise ValueError(f'{name} has {len(pairs)} {kind} bands '
                                     f'for {segments} segments')
                if not pairs:
                    continue
                # as arrays, once: a controller's target has four
                # thousand segments a channel, and a check per pair in
                # Python cost a second per copy of the draft
                arr = np.asarray(pairs, dtype=float)
                wrong = np.flatnonzero(~((arr[:, 0] < 0) & (arr[:, 1] > 0)))
                if len(wrong):
                    low, high = pairs[wrong[0]]
                    raise ValueError(
                        f'the {kind} band of {name} is {low:g} to '
                        f'{high:g} dB; it lies below and above the target')
                if entry['symmetric'] and not np.allclose(arr[:, 0], -arr[:, 1]):
                    raise ValueError(f'{name} is symmetric, and its {kind} '
                                     'band is not')
                if entry['uniform'] and not np.allclose(arr, arr[0]):
                    raise ValueError(f'{name} is uniform, and its {kind} '
                                     'band differs between segments')
        if self.form not in ('breakpoints', 'lines'):
            raise ValueError(f"a sheet is 'breakpoints' or 'lines', not "
                             f'{self.form!r}')
        if self.spacing is not None and not self.spacing > 0:
            raise ValueError('a frequency spacing is positive')

    def _sources(self) -> list[str]:
        return self.sources or [''] * self.num_channels

    def pairable(self) -> list[tuple[int, int]]:
        """Every pair `(i, j)`, `i < j`, that has a cross term: two
        channels of the same specification."""
        sources = self._sources()
        n = self.num_channels
        return [(i, j) for i in range(n) for j in range(i + 1, n)
                if sources[i] == sources[j]]

    def unset_pairs(self) -> list[tuple[int, int]]:
        """The pairs no cross term has been stated for."""
        return [pair for pair in self.pairable()
                if self.pairs.get(pair) is None]

    def _copy(self, **changes: Any) -> SpecificationDraft:
        fields = {'channels': list(self.channels), 'dims': list(self.dims),
                  'frequencies': list(self.frequencies),
                  'levels': [list(row) for row in self.levels],
                  'pairs': dict(self.pairs),
                  'bands': [_normal_bands(entry) for entry in self.bands],
                  'notes': list(self.notes),
                  'sources': list(self.sources), 'form': self.form,
                  'spacing': self.spacing}
        fields.update(changes)
        return SpecificationDraft(**fields)

    # ---- the operations, each a new draft ----------------------------------------

    def with_all_pairs(self, coherence: float,
                       phase_degrees: float = 0.0) -> SpecificationDraft:
        """The same draft with one answer stated for every pair.

        Parameters
        ----------
        coherence : float
            The coherence between every two channels, 0 for
            independent, 1 for fully coherent.
        phase_degrees : float, default 0.0
            The phase of the later channel against the earlier.

        Returns
        -------
        SpecificationDraft
            A new draft; this one is unchanged.
        """
        return self._copy(pairs={
            pair: (float(coherence), float(phase_degrees))
            for pair in self.pairable()})

    def scaled_db(self, decibels: float,
                  channels: list[str] | None = None) -> SpecificationDraft:
        """The same draft with the target raised or lowered by
        `decibels` — on every channel, or the named ones.

        Parameters
        ----------
        decibels : float
            The change, positive to raise.
        channels : list of str, optional
            Which channels; all of them when omitted.

        Returns
        -------
        SpecificationDraft
            A new draft; this one is unchanged.
        """
        factor = 10 ** (float(decibels) / 10.0)
        chosen = set(self.channels if channels is None else channels)
        unknown = chosen - set(self.channels)
        if unknown:
            raise ValueError(f'no channel {min(unknown)!r} in the draft')
        return self._copy(levels=[
            [v * factor for v in row] if name in chosen else list(row)
            for name, row in zip(self.channels, self.levels)])

    def segment_of(self, frequencies: Any = None) -> np.ndarray:
        """Which segment each frequency lies in: a breakpoint belongs to
        the segment on its right, the last breakpoint (and anything
        past the end) to the last segment."""
        f = np.asarray(self.frequencies)
        last = max(len(f) - 2, 0)
        if frequencies is None:
            return np.minimum(np.arange(len(f)), last)
        at = np.atleast_1d(np.asarray(frequencies, dtype=float))
        return np.clip(np.searchsorted(f, at, side='right') - 1, 0, last)

    def band_db(self, kind: str, channel: int,
                frequencies: Any = None) -> tuple[np.ndarray, np.ndarray]:
        """(below, above) in decibels at each frequency for one
        channel: the band of the segment each lies in.

        Parameters
        ----------
        kind : {'warning', 'abort'}
            Which band.
        channel : int
            Which channel, by index.
        frequencies : array-like, optional
            Where to answer; the breakpoints themselves when omitted.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The band below and above the target, per frequency.
        """
        pairs = np.asarray(self.bands[channel][kind], dtype=float)
        if not len(pairs):
            return np.empty(0), np.empty(0)
        which = self.segment_of(frequencies)
        return pairs[which, 0], pairs[which, 1]

    def _bands_on(self, frequencies: Any) -> list[dict[str, Any]]:
        """Every channel's bands re-cut onto another breakpoint list:
        each new segment takes the band of the old segment its start
        lies in."""
        starts = np.asarray(frequencies, dtype=float)[:-1]
        which = self.segment_of(starts) if len(self.frequencies) > 1 else []
        out = []
        for entry in self.bands:
            cut = {'symmetric': entry['symmetric'], 'uniform': entry['uniform']}
            for kind in ('warning', 'abort'):
                pairs = entry[kind]
                cut[kind] = ([pairs[k] for k in which] if len(pairs)
                             else [WARNING_DB if kind == 'warning' else ABORT_DB]
                             * len(starts))
            out.append(cut)
        return out

    def _channel_indices(self, channels: Any) -> list[int]:
        if channels is None:
            return list(range(self.num_channels))
        unknown = [c for c in channels if c not in self.channels]
        if unknown:
            raise ValueError(f'no channel {unknown[0]!r} in the draft')
        return [self.channels.index(c) for c in channels]

    def with_band(self, kind: str, below: float, above: float,
                  channels: Any = None,
                  segments: Any = None) -> SpecificationDraft:
        """The same draft with a band stated outright.

        Parameters
        ----------
        kind : {'warning', 'abort'}
            Which band.
        below, above : float
            Decibels below (negative) and above (positive) the target.
        channels : list of str, optional
            Which channels; all of them when omitted.
        segments : list of int, optional
            Which segments, by index; all of them when omitted, and
            all of them regardless on a uniform channel.

        Returns
        -------
        SpecificationDraft
            A new draft; this one is unchanged. Refused where the
            band breaks a channel's own constraint.
        """
        bands = [_normal_bands(entry) for entry in self.bands]
        for k in self._channel_indices(channels):
            entry = bands[k]
            if entry['symmetric'] and not np.isclose(below, -above):
                raise ValueError(f'{self.channels[k]} is symmetric: a band '
                                 f'{below:g} to {above:g} dB is not')
            which = (range(len(entry[kind]))
                     if segments is None or entry['uniform'] else segments)
            for s in which:
                entry[kind][s] = (float(below), float(above))
        return self._copy(bands=bands)

    def shifted_band(self, kind: str, side: str, segments: Any,
                     decibels: float, channels: Any = None,
                     step: float = 1.0) -> SpecificationDraft:
        """The same draft with one edge of a band moved — the drag on
        the plot, landed. A symmetric channel moves the other edge to
        match; a uniform channel moves every segment. The edge lands
        on a multiple of `step` (Brandon, 2026-09-06: whole decibels,
        never 1.5), and never nearer the target than one step.

        Parameters
        ----------
        kind : {'warning', 'abort'}
            Which band.
        side : {'lower', 'upper'}
            Which edge was dragged.
        segments : list of int
            The segments under the drag.
        decibels : float
            How far, positive up.
        channels : list of str, optional
            Which channels; all of them when omitted — the channels
            selected are the channels edited.
        step : float, default 1.0
            The decibel grid the edge lands on.

        Returns
        -------
        SpecificationDraft
            A new draft; this one is unchanged.
        """
        return self._band_edge(kind, side, segments, channels, step,
                               lambda was: was + decibels)

    def with_band_edge(self, kind: str, side: str, segments: Any,
                       decibels: float, channels: Any = None,
                       step: float = 1.0) -> SpecificationDraft:
        """The same draft with one edge of a band set to `decibels` on
        every channel — the drag on the plot, landed (Brandon,
        2026-09-06: a drag from 3 to 4 dB puts every selected channel
        at 4 dB, not each one decibel further from where it was, so a
        channel at 3 and one at 4 both come out at 4). A symmetric
        channel sets the other edge to match; a uniform channel sets
        every segment. Lands on a multiple of `step`, never nearer the
        target than one step.

        Parameters
        ----------
        kind : {'warning', 'abort'}
            Which band.
        side : {'lower', 'upper'}
            Which edge.
        segments : list of int
            The segments under the drag.
        decibels : float
            The edge's level, positive above the target.
        channels : list of str, optional
            Which channels; all of them when omitted.
        step : float, default 1.0
            The decibel grid the edge lands on.

        Returns
        -------
        SpecificationDraft
            A new draft; this one is unchanged.
        """
        return self._band_edge(kind, side, segments, channels, step,
                               lambda was: np.full_like(was, decibels))

    def _band_edge(self, kind, side, segments, channels, step, landing):
        """One edge of a band, on each channel's own constraints, at
        `landing(was)` — the one implementation under `shifted_band`
        and `with_band_edge`."""
        bands = [_normal_bands(entry) for entry in self.bands]
        for k in self._channel_indices(channels):
            entry = bands[k]
            pairs = entry[kind]
            if not pairs:
                continue
            # the segments under the drag, as one array: a controller's
            # target has four thousand a channel, and a round per pair
            # in Python was a third of a second per drag
            which = (np.arange(len(pairs)) if entry['uniform']
                     else np.asarray(list(segments), dtype=int))
            arr = np.asarray(pairs, dtype=float)
            moved = np.round(
                landing(arr[which, 0 if side == 'lower' else 1]) / step) * step
            if side == 'lower':
                arr[which, 0] = np.minimum(moved, -step)
                if entry['symmetric']:
                    arr[which, 1] = -arr[which, 0]
            else:
                arr[which, 1] = np.maximum(moved, step)
                if entry['symmetric']:
                    arr[which, 0] = -arr[which, 1]
            entry[kind] = [(float(low), float(high)) for low, high in arr]
        return self._copy(bands=bands)

    def constrained(self, channels: Any = None, symmetric: bool | None = None,
                    uniform: bool | None = None) -> SpecificationDraft:
        """The same draft with channels' constraints set. Turning
        symmetric on mirrors the band above to the one below; turning
        uniform on gives every segment the first one's band.

        Parameters
        ----------
        channels : list of str, optional
            Which channels; all of them when omitted.
        symmetric, uniform : bool, optional
            The constraint to set; kept when omitted.

        Returns
        -------
        SpecificationDraft
            A new draft; this one is unchanged.
        """
        bands = [_normal_bands(entry) for entry in self.bands]
        for k in self._channel_indices(channels):
            entry = bands[k]
            if symmetric is not None:
                entry['symmetric'] = bool(symmetric)
            if uniform is not None:
                entry['uniform'] = bool(uniform)
            for kind in ('warning', 'abort'):
                pairs = entry[kind]
                if entry['uniform'] and pairs:
                    pairs[:] = [pairs[0]] * len(pairs)
                if entry['symmetric']:
                    pairs[:] = [(-abs(high), abs(high)) for _low, high in pairs]
        return self._copy(bands=bands)

    def sections(self, channel: int, tolerance: float = 1e-6) -> list[list[int]]:
        """One channel's segments grouped into the linear sections of
        its requirement: every segment on its own in the breakpoints
        form, and in the interpolated form the runs of one power law
        — the lines between bends, as `to_breakpoints` finds them."""
        n = len(self.frequencies)
        if n < 2:
            return []
        if self.form != 'lines':
            return [[s] for s in range(n - 1)]
        log_f = np.log10(np.asarray(self.frequencies))
        slope = np.diff(np.log10(np.asarray(self.levels[channel]))) / np.diff(log_f)
        bend = ~np.isclose(slope[1:], slope[:-1], rtol=tolerance, atol=tolerance)
        runs: list[list[int]] = [[0]]
        for s in range(1, n - 1):
            if bend[s - 1]:
                runs.append([s])
            else:
                runs[-1].append(s)
        return runs

    def band_runs(self, channel: int, kind: str) -> list[list[int]]:
        """The segments of one channel grouped as the plot draws its
        handles: one run over the whole range while the channel is
        uniform, else one per linear section of the requirement
        (Brandon, 2026-09-06: Uniform off should show a level per
        section, each moved on its own), split again wherever the
        band already differs within a section."""
        pairs = self.bands[channel][kind]
        if not pairs:
            return []
        if self.bands[channel]['uniform']:
            return [list(range(len(pairs)))]
        runs: list[list[int]] = []
        for section in self.sections(channel):
            for s in section:
                if runs and s != section[0] and pairs[runs[-1][0]] == pairs[s]:
                    runs[-1].append(s)
                else:
                    runs.append([s])
        return runs

    def with_breakpoint(self, frequency: float) -> SpecificationDraft:
        """The same draft with a breakpoint at `frequency`, its levels
        read off the power law between its neighbors — or the end
        breakpoint's, past the ends.

        Parameters
        ----------
        frequency : float
            Where, in Hz.

        Returns
        -------
        SpecificationDraft
            A new draft; this one is unchanged.
        """
        frequency = float(frequency)
        if frequency <= 0:
            raise ValueError('a breakpoint frequency is positive')
        if frequency in self.frequencies:
            return self._copy()
        log_f = np.log(np.asarray(self.frequencies))
        levels = []
        for row in self.levels:
            value = float(np.exp(np.interp(np.log(frequency), log_f,
                                           np.log(np.asarray(row)))))
            levels.append(row + [value])
        order = np.argsort(self.frequencies + [frequency], kind='stable')
        frequencies = [(self.frequencies + [frequency])[k] for k in order]
        # the split segment's band on both halves
        return self._copy(
            frequencies=frequencies,
            levels=[[row[k] for k in order] for row in levels],
            bands=self._bands_on(frequencies))

    def without_breakpoint(self, frequency: float) -> SpecificationDraft:
        """The same draft without the breakpoint at `frequency`.

        Parameters
        ----------
        frequency : float
            Which, in Hz; it has to be one of the breakpoints, and not
            one of the last two.

        Returns
        -------
        SpecificationDraft
            A new draft; this one is unchanged.
        """
        frequency = float(frequency)
        if frequency not in self.frequencies:
            raise ValueError(f'no breakpoint at {frequency:g} Hz')
        if len(self.frequencies) <= 2:
            raise ValueError('a specification keeps at least two breakpoints')
        k = self.frequencies.index(frequency)
        frequencies = [f for i, f in enumerate(self.frequencies) if i != k]
        # the merged span keeps the band of the segment it starts in
        return self._copy(
            frequencies=frequencies,
            levels=[[v for i, v in enumerate(row) if i != k]
                    for row in self.levels],
            bands=self._bands_on(frequencies))

    def to_breakpoints(self, tolerance: float = 1e-6) -> SpecificationDraft:
        """The same requirement as the few points it was written from:
        the first and last line, and every line where the log-log
        slope of any channel changes. Exact for a specification that
        was read onto lines from breakpoints (a controller's target);
        a specification with no such structure keeps every line, and
        the notes say so rather than the sheet inventing a
        simplification.

        Parameters
        ----------
        tolerance : float, default 1e-6
            How much the slope may wander between adjacent segments
            and still be one power law.

        Returns
        -------
        SpecificationDraft
            A new draft, `form` 'breakpoints'; this one is unchanged.
        """
        n = len(self.frequencies)
        if n <= 2:
            return self._copy(form='breakpoints')
        log_f = np.log10(np.asarray(self.frequencies))
        keep = np.zeros(n, dtype=bool)
        keep[[0, -1]] = True
        for row in self.levels:
            slope = np.diff(np.log10(np.asarray(row))) / np.diff(log_f)
            bend = ~np.isclose(slope[1:], slope[:-1], rtol=tolerance,
                               atol=tolerance)
            keep[1:-1] |= bend
        # a band that steps at a line is a bend in the limit, and the
        # line stays for it even where the target runs straight through
        for entry in self.bands:
            for kind in ('warning', 'abort'):
                pairs = entry[kind]
                for s in range(len(pairs) - 1):
                    if pairs[s] != pairs[s + 1]:
                        keep[s + 1] = True
        kept = [i for i in range(n) if keep[i]]
        notes = list(self.notes)
        if len(kept) == n:
            notes.append(f'no power-law structure in its {n} lines: every '
                         'line kept as a breakpoint')
        else:
            notes.append(f'{n} lines read as {len(kept)} breakpoints')
        frequencies = [self.frequencies[i] for i in kept]
        if any(len({self.bands[k][kind][s]
                    for s in range(kept[r], kept[r + 1])}) > 1
               for k in range(self.num_channels) for kind in ('warning', 'abort')
               for r in range(len(kept) - 1)):
            notes.append('a band that changed within a folded segment was '
                         'read as the band at its start')
        return self._copy(
            frequencies=frequencies,
            levels=[[row[i] for i in kept] for row in self.levels],
            bands=self._bands_on(frequencies), notes=notes, form='breakpoints')

    def to_lines(self, spacing: float | None = None) -> SpecificationDraft:
        """The same requirement read onto evenly spaced frequency
        lines — multiples of `spacing` from 0 Hz, as a controller lays
        its lines — over the band the breakpoints cover, each channel
        on its own power law between them.

        Only the lines inside the band: a sheet holds positive levels,
        and the zeros a controller writes outside the band are its own
        convention, filled in by its loader for any line the file
        does not name.

        Parameters
        ----------
        spacing : float, optional
            The line spacing in Hz. Defaults to the spacing the sheet
            remembers; with neither, refused — nothing is guessed.

        Returns
        -------
        SpecificationDraft
            A new draft, `form` 'lines'; this one is unchanged.
        """
        spacing = self.spacing if spacing is None else float(spacing)
        if spacing is None:
            raise ValueError('no frequency spacing to read the breakpoints '
                             'onto — give the spacing the lines are at')
        if not spacing > 0:
            raise ValueError('a frequency spacing is positive')
        low, high = self.frequencies[0], self.frequencies[-1]
        first = int(np.ceil(low / spacing - 1e-9))
        last = int(np.floor(high / spacing + 1e-9))
        if last - first < 1:
            raise ValueError(f'fewer than two lines at {spacing:g} Hz fall '
                             f'between {low:g} and {high:g} Hz')
        lines = np.arange(first, last + 1) * spacing
        log_f = np.log(np.asarray(self.frequencies))
        levels = [[float(v) for v in np.exp(np.interp(
            np.log(lines), log_f, np.log(np.asarray(row))))]
                  for row in self.levels]
        notes = list(self.notes)
        edges = [f'{f:g}' for f, k in ((low, first), (high, last))
                 if abs(k * spacing - f) > 1e-9 * max(1.0, f)]
        if edges:
            notes.append(f'the band edge{"s" * (len(edges) != 1)} at '
                         f'{" and ".join(edges)} Hz {"are" if len(edges) != 1 else "is"} '
                         f'not on the {spacing:g} Hz grid; the lines run '
                         f'{lines[0]:g} to {lines[-1]:g} Hz')
        return self._copy(frequencies=[float(f) for f in lines],
                          levels=levels, notes=notes, form='lines',
                          spacing=spacing, bands=self._bands_on(lines))

    # ---- the ways out ------------------------------------------------------------

    def bands_differ(self) -> bool:
        """Whether any channel's band changes from one segment to the
        next — where the object needs a helper line to carry the step."""
        return any(entry[kind][s] != entry[kind][s + 1]
                   for entry in self.bands for kind in ('warning', 'abort')
                   for s in range(len(entry[kind]) - 1))

    def with_band_corners(self, part: float = 1e-6) -> SpecificationDraft:
        """The same draft with a helper breakpoint a hair before each
        breakpoint whose two segments carry different bands, so the
        object made of it can carry the step.

        A limit array holds one value per line and a breakpoint
        belongs to the segment on its right, so the segment on its
        left needs a line of its own at the corner to show its band up
        to it. The helper sits `part` of the breakpoint's frequency
        before it, on the same power law, and folds away again —
        `to_breakpoints` keeps only lines that bend.

        Returns
        -------
        SpecificationDraft
            A new draft; the same one when no helper is needed.
        """
        f = self.frequencies
        corners = sorted({
            f[s + 1] * (1 - part)
            for entry in self.bands for kind in ('warning', 'abort')
            for s in range(len(entry[kind]) - 1)
            if entry[kind][s] != entry[kind][s + 1]})
        out = self
        for frequency in corners:
            if frequency not in out.frequencies:
                out = out.with_breakpoint(frequency)
        return out

    def make(self, comment: str = '') -> Specification:
        """The specification, with every cross term.

        Parameters
        ----------
        comment : str, default ''
            A comment for the records — where the sheet came from.

        Returns
        -------
        Specification
            Autospectra and cross terms at the breakpoints, a power law
            between them, the bands on every autospectrum.
        """
        from .data import Specification

        if len(set(self.sources)) > 1:
            raise ValueError('the sheet spans several specifications; write '
                             'it into them, or make one of them alone '
                             '(for_source)')
        if self.bands_differ():
            cornered = self.with_band_corners()
            if cornered is not self:
                return cornered.make(comment)
        n = self.num_channels
        autos = np.asarray(self.levels, dtype=np.float64)          # n × points
        rows, resp, refs, record_dims, limits = [], [], [], [], {
            'warning_lower': [], 'warning_upper': [],
            'abort_lower': [], 'abort_upper': []}
        blank = np.full(len(self.frequencies), np.nan)
        for i in range(n):
            for j in range(n):
                if i == j:
                    values = autos[i].astype(np.complex128)
                else:
                    a, b = min(i, j), max(i, j)
                    if self.pairs.get((a, b)) is None:
                        # unstated is absent: the object holds its
                        # autospectra and the pairs that were stated,
                        # and nothing is assumed for the rest (the
                        # sheet edits the object live, 2026-09-06, so
                        # an unstated pair cannot be a refusal)
                        continue
                    coherence, phase = self.pairs[(a, b)]
                    sign = 1.0 if i < j else -1.0
                    values = (np.sqrt(coherence * autos[i] * autos[j])
                              * np.exp(1j * sign * np.radians(phase)))
                rows.append(values)
                resp.append(self.channels[i])
                refs.append(self.channels[j])
                record_dims.append(
                    f'{self.dims[i]}**2/frequency'
                    if self.dims[i] == self.dims[j]
                    else f'{self.dims[i]}*{self.dims[j]}/frequency')
                for kind in ('warning', 'abort'):
                    below, above = self.band_db(kind, i)
                    for side, decibels in (('lower', below), ('upper', above)):
                        limits[f'{kind}_{side}'].append(
                            autos[i] * 10 ** (decibels / 10.0)
                            if i == j else blank)
        spec = Specification(
            np.asarray(self.frequencies), np.asarray(rows),
            response_dof=resp, reference_dof=refs, ordinate_dim=record_dims,
            comment=comment or 'written as a specification sheet',
            **{key: np.asarray(values) for key, values in limits.items()})
        spec.interpolation = 'log_log'
        spec.band_constraints = self.constraints()
        return spec

    def constraints(self) -> dict[str, dict[str, bool]]:
        """Each channel's constraints, by channel — what the object
        carries beside its limits so a sheet reopens as it was set."""
        return {name: {'symmetric': entry['symmetric'],
                       'uniform': entry['uniform']}
                for name, entry in zip(self.channels, self.bands)}

    def written_into(self, spec: Specification,
                     comment: str = '') -> Specification:
        """The specification with this sheet's channels rewritten in
        it, at its own lines.

        The sheet's channels take the sheet's levels, cross terms and
        bands over the sheet's range, read off its power law at each
        line; a breakpoint that is not a line becomes one. Lines the
        sheet left out — 0 Hz, a zero level — keep what they had, and
        a cross term the sheet states but the specification never
        held is made there from its own autospectra. Channels the
        sheet does not hold are untouched, read off their own curve
        at any new line.

        Parameters
        ----------
        spec : Specification
            The specification the sheet was opened from.
        comment : str, default ''
            Appended to the record's comment.

        Returns
        -------
        Specification
            A new object; `spec` is unchanged.
        """
        from .data import Specification

        if len(set(self.sources)) > 1:
            raise ValueError('the sheet spans several specifications; write '
                             'each in with for_source')
        old_f = np.asarray(spec.abscissa, dtype=float)
        lines = np.asarray(sorted(set(old_f.tolist()) | set(self.frequencies)))
        inside = (lines >= self.frequencies[0]) & (lines <= self.frequencies[-1])
        log_f = np.log(np.asarray(self.frequencies))
        with np.errstate(divide='ignore'):
            log_lines = np.log(lines)
        # the sheet's autospectra at every line inside its range
        sheet = {}
        for name, row in zip(self.channels, self.levels):
            values = np.full(len(lines), np.nan)
            values[inside] = np.exp(np.interp(log_lines[inside], log_f,
                                              np.log(np.asarray(row))))
            sheet[name] = values
        index = {name: k for k, name in enumerate(self.channels)}
        references = spec.reference_dof or list(spec.response_dof)
        held = {(spec.response_dof[i], references[i]): i
                for i in range(spec.num_records)}
        resp, refs, dims, rows = [], [], [], []
        limits: dict[str, list[np.ndarray]] = {
            key: [] for key in Specification.LIMITS}
        def band_at(key, name):
            kind, side = key.split('_')
            below, above = self.band_db(kind, index[name], lines[inside])
            return below if side == 'lower' else above

        def carried(values):
            return read_at_lines(old_f, values, lines)

        def autos_at(name, i):
            # the sheet inside its range, the specification's own
            # autospectrum elsewhere (an edited channel's old curve
            # stands where the sheet left the line out)
            own = carried(np.real(spec.ordinate[i])) if i is not None else None
            values = sheet[name].copy()
            if own is not None:
                values[~inside] = own[~inside]
            return values

        def cross_of(a, b):
            coherence, phase = self.pairs[(min(index[a], index[b]),
                                           max(index[a], index[b]))]
            sign = 1.0 if index[a] < index[b] else -1.0
            sa, sb = autos_at(a, held.get((a, a))), autos_at(b, held.get((b, b)))
            with np.errstate(invalid='ignore'):
                magnitude = np.sqrt(coherence * sa * sb)
            magnitude = np.where(np.isfinite(magnitude), magnitude, 0.0)
            return magnitude * np.exp(1j * sign * np.radians(phase))

        for i in range(spec.num_records):
            a, b = spec.response_dof[i], references[i]
            resp.append(a)
            refs.append(b)
            dims.append(spec.ordinate_dim[i])
            stated = (a in index and b in index and (
                a == b or self.pairs.get((min(index[a], index[b]),
                                          max(index[a], index[b])))
                is not None))
            if stated:
                if a == b:
                    values = autos_at(a, i).astype(np.complex128)
                else:
                    values = cross_of(a, b)
                rows.append(values)
                for key in Specification.LIMITS:
                    old = spec.limits.get(key)
                    kept = (carried(np.real(old[i])) if old is not None
                            else np.full(len(lines), np.nan))
                    if a == b:
                        kept = kept.copy()
                        kept[inside] = (np.real(values)[inside]
                                        * 10 ** (band_at(key, a) / 10.0))
                    limits[key].append(kept)
            else:
                rows.append(carried(spec.ordinate[i]))
                for key in Specification.LIMITS:
                    old = spec.limits.get(key)
                    limits[key].append(carried(np.real(old[i]))
                                       if old is not None
                                       else np.full(len(lines), np.nan))
        blank = np.full(len(lines), np.nan)
        comments = list(spec.comment)
        for a in self.channels:
            for b in self.channels:
                if a != b and (a, b) not in held and self.pairs.get(
                        (min(index[a], index[b]), max(index[a], index[b]))) \
                        is not None:
                    comments.append(comment or 'written as a specification '
                                    'sheet')
                    resp.append(a)
                    refs.append(b)
                    da, db = self.dims[index[a]], self.dims[index[b]]
                    dims.append(f'{da}**2/frequency' if da == db
                                else f'{da}*{db}/frequency')
                    rows.append(cross_of(a, b))
                    for key in Specification.LIMITS:
                        limits[key].append(blank)
        keep = {key: values for key, values in limits.items()
                if key in spec.limits or any(
                    np.isfinite(v).any() for v in values)}
        out = Specification(
            lines, np.asarray(rows), response_dof=resp, reference_dof=refs,
            ordinate_dim=dims, comment=comments,
            **{key: np.asarray(values) for key, values in keep.items()})
        out.interpolation = spec.interpolation
        carried = dict(getattr(spec, 'band_constraints', None) or {})
        carried.update(self.constraints())
        out.band_constraints = carried
        return out

    def preview(self) -> Specification:
        """The autospectra and their bands alone — what the plot shows
        beside the sheet."""
        from .data import Specification

        if len(set(self.sources)) > 1:
            raise ValueError('the sheet spans several specifications; '
                             'preview each with for_source')
        spec = self._copy(sources=[]).make()
        autos = [i for i in range(spec.num_records)
                 if spec.response_dof[i] == spec.reference_dof[i]]
        preview = Specification(
            spec.abscissa, np.real(spec.ordinate[autos]),
            response_dof=[spec.response_dof[i] for i in autos],
            reference_dof=[spec.reference_dof[i] for i in autos],
            ordinate_dim=[spec.ordinate_dim[i] for i in autos],
            **{key: values[autos] for key, values in spec.limits.items()})
        preview.interpolation = 'log_log'
        return preview

    def as_dict(self) -> dict[str, Any]:
        """A JSON-friendly form, for the provenance record."""
        return {'channels': list(self.channels), 'dims': list(self.dims),
                'frequencies': list(self.frequencies),
                'levels': [list(row) for row in self.levels],
                'pairs': [[i, j, None if value is None else list(value)]
                          for (i, j), value in sorted(self.pairs.items())],
                'bands': [{'warning': [list(p) for p in entry['warning']],
                           'abort': [list(p) for p in entry['abort']],
                           'symmetric': entry['symmetric'],
                           'uniform': entry['uniform']}
                          for entry in self.bands],
                'notes': list(self.notes), 'sources': list(self.sources),
                'form': self.form, 'spacing': self.spacing}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpecificationDraft:
        """The draft a provenance record holds."""
        return cls(data['channels'], data['dims'], data['frequencies'],
                   data['levels'],
                   {(int(i), int(j)): (None if value is None
                                       else (value[0], value[1]))
                    for i, j, value in data.get('pairs', [])},
                   data.get('bands', []),
                   data.get('notes', []), data.get('sources', []),
                   data.get('form', 'breakpoints'), data.get('spacing'))


def default_bands(segments: int) -> dict[str, Any]:
    """A channel's starting bands: ±3 dB warning, ±6 dB abort on every
    segment, symmetric and uniform."""
    return {'warning': [WARNING_DB] * segments, 'abort': [ABORT_DB] * segments,
            'symmetric': True, 'uniform': True}


def _normal_bands(entry: Any) -> dict[str, Any]:
    """One channel's band entry as floats in tuples, a fresh copy."""
    return {'warning': [(float(p[0]), float(p[1])) for p in entry['warning']],
            'abort': [(float(p[0]), float(p[1])) for p in entry['abort']],
            'symmetric': bool(entry.get('symmetric', True)),
            'uniform': bool(entry.get('uniform', True))}


def band_notes(spec: Specification) -> list[str]:
    """What a sheet opened on `spec` should say about bands it does
    not have: a specification with no warning or abort limits opens
    wearing the defaults, there on the plot to drag, and the first
    edit writes them (Brandon, 2026-09-06: where there are none, the
    sheet is where they are added). Said from the object rather than
    kept on the draft, so the saying stops when the limits exist.

    Parameters
    ----------
    spec : Specification
        The specification the sheet is open on.

    Returns
    -------
    list of str
        One note per band kind missing; empty when both are there.
    """
    limits = getattr(spec, 'limits', None) or {}
    return [f'no {kind} band on the specification — the sheet shows '
            f'{low:+.0f}/{high:+.0f} dB, written with the first edit'
            for kind, (low, high) in (('warning', WARNING_DB),
                                      ('abort', ABORT_DB))
            if all(limits.get(f'{kind}_{side}') is None
                   for side in ('lower', 'upper'))]


def _helper_lines(frequencies: Any) -> list[float]:
    """The lines a hair before another — `with_band_corners` puts one
    `1e-6` of a breakpoint's frequency before it; nothing else writes
    two lines closer than a hundred-thousandth of their frequency."""
    f = np.asarray(frequencies, dtype=float)
    if len(f) < 3:
        return []
    helper = np.zeros(len(f), dtype=bool)
    helper[:-1] = np.diff(f) < 1e-5 * f[1:]
    return [float(v) for v in f[helper]]


def uniform_spacing(frequencies: Any) -> float | None:
    """The spacing of evenly spaced lines, or None when they are not
    — a controller writes its target at multiples of one spacing, a
    breakpoint specification at whatever frequencies it was written.

    A helper line — one a hair before a breakpoint, carrying a band
    step (`with_band_corners`) — is not a line of the grid and is
    left out of the reckoning: read as one, two helpers made a
    controller's target open as four thousand breakpoints, each its
    own section (2026-09-06)."""
    f = np.asarray(frequencies, dtype=float)
    if len(f) < 3:
        return None
    f = np.asarray([v for v in f if v not in _helper_lines(f)])
    if len(f) < 3:
        return None
    steps = np.diff(f)
    if not steps[0] > 0 or not np.allclose(steps, steps[0], rtol=1e-6,
                                            atol=1e-9 * steps[0]):
        return None
    return float(steps[0])


def read_at_lines(old_f: np.ndarray, values: np.ndarray,
              lines: np.ndarray) -> np.ndarray:
    """A specification's record read at new lines: its own value where
    a line is one of its own, and between them the power law a
    specification means — magnitude log–log, phase straight — so a
    channel the sheet does not hold is unchanged in meaning by a line
    the sheet added. Past its ends, its end values."""
    values = np.asarray(values)
    exact = np.isin(lines, old_f)
    if exact.all():
        return values[np.searchsorted(old_f, lines)]
    with np.errstate(divide='ignore'):
        log_old, log_new = np.log(old_f), np.log(lines)
    magnitude = np.abs(values)
    positive = np.isfinite(magnitude) & (magnitude > 0) & np.isfinite(log_old)
    out = np.full(len(lines), np.nan, dtype=values.dtype)
    out[exact] = values[np.searchsorted(old_f, lines[exact])]
    new = ~exact
    if positive.sum() >= 2:
        size = np.exp(np.interp(log_new[new], log_old[positive],
                                np.log(magnitude[positive])))
    else:
        size = np.interp(lines[new], old_f, np.nan_to_num(magnitude))
    if np.iscomplexobj(values):
        phase = np.interp(lines[new], old_f,
                          np.unwrap(np.angle(values)))
        out[new] = size * np.exp(1j * phase)
    else:
        out[new] = size * np.sign(np.interp(lines[new], old_f,
                                            np.nan_to_num(values)))
    return out
