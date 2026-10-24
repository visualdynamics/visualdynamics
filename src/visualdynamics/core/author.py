"""Writing and editing a specification sheet, at any channels.

A specification sheet is a set of breakpoints, a level per channel at
each breakpoint, warning and abort bands in decibels below and above the
target, and a cross term for each pair of channels. None of that depends
on where the channels are, so one draft class is used for every way a
sheet is opened (new at a shape set's modal coordinates, new at chosen
physical DOFs or at a channel table's control channels, or opened from
an existing specification) and for every way it is saved (as a new
specification, or written back into the one it was opened from).

A specification written at the modal coordinates `M1 … Mn` and expanded
through the shape set becomes an exact control-channel CPSD, phase and
coherence included. The expansion makes no assumption: every number in
it comes from a value the author entered.

What an author enters in a `SpecificationDraft`:

- The channels, each with the quantity its density is of.
- The breakpoints: frequencies, and a level per channel at each.
  Between breakpoints the level follows a power law, which is how a
  written specification is read (`Specification.interpolation =
  'log_log'`).
- Each pair's cross term, as a coherence and a phase, from which
  `S_ij = √(γ² S_ii S_jj) e^{iφ}`. The entries must be able to hold
  together, which each in [0, 1] does not ensure: a sheet whose entries
  no set of signals has is refused (`core.matrix_checks`). A pair that
  has not been entered is not assumed to be zero; the specification
  made from the draft has no cross term for it. Independent channels
  are a statement too, and the author has to make it. `with_all_pairs`
  enters one coherence and phase for every pair at once.
- The bands: warning and abort, in decibels below and above the
  target, set per channel and per segment between breakpoints, under
  each channel's `symmetric` and `uniform` constraints.
- How the levels are typed (`entry`): a level at every breakpoint, or
  a level at the first and a slope in decibels per octave into each
  breakpoint after it, the level there computed, as many written
  tables give them (`core.breakpoints`). The draft holds levels either
  way; the entry decides what an edit holds still.

The operations on a draft (`scaled_db`, `with_band`, `with_band_edge`,
`shifted_band`, `constrained`, `with_breakpoint`, `without_breakpoint`,
`with_all_pairs`, `to_breakpoints`, `to_lines`, `with_entry`,
`with_slope`, `with_level`, `with_slope_breakpoint`) each return a new draft
and leave the original unchanged, so a script reads as a sequence of
statements, and so does a session's journal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..units import UNKNOWN
from .matrix_checks import entries_refusal
from .transform import MOTIONS, _modal_dim, modal_dofs

if TYPE_CHECKING:                                    # pragma: no cover
    from collections.abc import Sequence

    from .breakpoints import BreakpointReading
    from .channel_table import ChannelTable
    from .data import Specification
    from .shapes import ShapeSet

# One draft for every way in and out (Brandon, 2026-09-04: a generic
# editor, not a modal one). The modal-coordinate way in is the virtual
# point workflow's third tool. No assumptions about cross terms, ever
# (Brandon, 2026-09-04).

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


#: the two ways a sheet's levels are typed: a level at every
#: breakpoint, or a level at the first and a slope in decibels per
#: octave into each one after it
ENTRIES = ('levels', 'slopes')


#: the grid a dragged or stepped limit lands on, in decibels. A
#: whole decibel (Brandon, 2026-09-06: "never 1.5") was too coarse
#: to place a limit where it was wanted (Brandon, 2026-09-18); the
#: plot's handles snap to the same grid (`plot.bands.STEP`).
LIMIT_STEP_DB = 0.25


@dataclass
class SpecificationDraft:
    """A specification being written, at any channels.

    Attributes:
        channels: The DOF strings, one per channel: modal coordinates
            (`M1`), physical DOFs (`101Z+`), or whatever the sheet was
            opened at. Any name is accepted except an empty one.
        dims: The quantity each channel's density is of
            ('acceleration', 'angular_acceleration', …), one per
            channel; the density's dimension is its square per hertz.
        frequencies: The breakpoints, in Hz, increasing.
        levels: One list per channel, a density at each breakpoint,
            in the channel's SI density unit.
        pairs: `{(i, j): (coherence, phase_degrees)}` for every pair
            `i < j` of channels, or None where the pair has not been
            entered yet.
        bands: One entry per channel: `{'warning': [(below, above)
            per segment], 'abort': [...], 'symmetric': bool, 'uniform':
            bool}`, the bands in decibels below and above the target,
            where a segment is the span between two neighboring
            breakpoints. The constraints belong to each channel:
            `symmetric` holds the upper edge at minus the lower one,
            and `uniform` holds every segment at one band. A breakpoint
            belongs to the segment on its right, and the last
            breakpoint to the segment on its left.
        notes: What could not be carried over when the sheet was
            opened (cross terms that varied with frequency, bands that
            differed), listed so that nothing is dropped without
            notice.
        sources: The name of the specification each channel was opened
            from, one per channel, when the sheet spans several
            (`across`); empty when the sheet is one specification's.
            Channels from different specifications have no cross term
            between them, and the pair is not offered at all.
        form: 'breakpoints', the few points a specification is written
            from, or 'lines', the same requirement read onto evenly
            spaced frequency lines, as a controller writes its target.
            `to_breakpoints` and `to_lines` convert between the two.
            The levels are edited on the breakpoints.
        spacing: The frequency spacing in Hz of the lines form, or the
            spacing remembered from the specification the sheet was
            opened from. `to_lines` uses it when no spacing is given.
        entry: 'levels', a level typed at every breakpoint, or
            'slopes', a level at the first breakpoint and a slope in
            decibels per octave into each one after it, the level
            there computed. The levels are held either way. Under
            'slopes', `with_level` holds every slope after the level
            it changes, as `with_slope` does; under 'levels' it
            changes that level alone.
    """
    # Each channel has its own band settings, and the bands are dragged
    # on the plot per section, not typed in a pane (Brandon,
    # 2026-09-06). The levels are edited on the breakpoints (Brandon,
    # 2026-09-06).

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
    entry: str = 'levels'

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
        self.entry = str(self.entry)
        self.validate()

    # ---- the doors in ---------------------------------------------------------

    @classmethod
    def at_modal_coordinates(cls, shapes: ShapeSet,
                             quantity: str = 'acceleration',
                             modes: Sequence[int] | None = None
                             ) -> SpecificationDraft:
        """A starter sheet at a shape set's modal coordinates `M1 … Mn`.

        The starter is the one `at_dofs` makes. Each channel's quantity
        follows the transform's unit rule: per radian for a unit rigid
        set's rotations, the modal quantity for a mass-normalized set.
        A set with no mass unit gives its coordinates no unit, and a
        specification with no unit is refused.

        Parameters
        ----------
        shapes : ShapeSet
            The set the specification is written at.
        quantity : str, default 'acceleration'
            The physical quantity the target is a density of.
        modes : sequence of int, optional
            Which shapes get a channel, by index from 0 (the ones
            selected in the tree). All of them when omitted. A virtual
            point's target is usually its three translations alone.
            Expanded through the set, a shape with no channel
            contributes nothing, which is the same as a zero level, and
            a sheet holds only positive levels.

        Returns
        -------
        SpecificationDraft
            No pair entered.

        Raises
        ------
        ValueError
            When no shape is selected, an index is out of range, or the
            set has no mass unit.
        """
        # A virtual point's target is usually its three translations
        # alone (Brandon, 2026-09-06).
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
        """A starter sheet at chosen DOFs, with no pair entered.

        The starter is a flat line from 20 to 2000 Hz at one SI unit of
        density on every channel.

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
        """A starter sheet at a channel table's control channels.

        Each channel is in the quantity its channel type declares, and
        the starter is the one `at_dofs` makes. A flagged control
        channel of any declared type is used: a test can be controlled
        to a force or a voltage as well as to a motion, and a
        specification written at a force is a force specification.
        When the table flags no control channel, every motion channel
        is used, since those are the likely control channels.

        Parameters
        ----------
        table : ChannelTable
            The table.

        Returns
        -------
        SpecificationDraft
            The sheet to fill.

        Raises
        ------
        ValueError
            When the flagged channels declare no channel type, or when
            nothing is flagged and the table has no motion channel.
        """
        # Any declared type, not only motions: a test can be controlled
        # to a force or a voltage (Brandon, 2026-09-26).
        dofs, types = table.dof_strings(), table.types()
        flagged = table.controls()
        if flagged.any():
            rows = [i for i in range(len(dofs))
                    if flagged[i] and types[i] and dofs[i]]
            if not rows:
                raise ValueError('the channel table\'s control channels '
                                 'declare no channel type, so a '
                                 'specification at them would be in '
                                 'nothing — declare their types')
        else:
            rows = [i for i in range(len(dofs))
                    if types[i] in MOTIONS and dofs[i]]
            if not rows:
                raise ValueError('the channel table has no motion channel to '
                                 'write a specification at — declare the '
                                 'channel types, or flag the control '
                                 'channels')
        return cls.at_dofs([dofs[i] for i in rows], [types[i] for i in rows])

    @classmethod
    def from_specification(cls, spec: Specification,
                           channels: list[str] | None = None,
                           source: str = '') -> SpecificationDraft:
        """A sheet opened from an existing specification.

        The autospectra's lines become the breakpoints. Lines at 0 Hz,
        or where any autospectrum is not positive, are left out, with
        a note. A cross term that is one coherence and phase over
        frequency becomes that pair's entry; one that varies is left
        unentered, with a note. The warning and abort limits become
        bands in decibels per segment, and a specification with no
        limits opens with the default bands. A specification on evenly
        spaced lines opens in the 'lines' form and remembers its
        spacing.

        Parameters
        ----------
        spec : Specification
            The specification to open.
        channels : list of str, optional
            Only these of its channels (the ones selected in the tree).
            All of them when omitted.
        source : str, default ''
            The specification's name, for a sheet that will span
            several (`across`).

        Returns
        -------
        SpecificationDraft
            Ready to edit. `written_into` writes it back, and `make`
            makes a new specification from it alone.

        Raises
        ------
        ValueError
            When a named channel has no autospectrum, the specification
            has no autospectrum, a channel's units are not declared, or
            fewer than two lines have every autospectrum positive.
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
        # said here, with what to do: the draft's own check said only
        # "every channel needs a quantity", and the window cleared the
        # plot under it (Brandon, 2026-10-14)
        undeclared = [ch for ch, dim in zip(channels, dims) if dim == UNKNOWN or not dim]
        if undeclared:
            shown = ', '.join(undeclared[:3]) + (', …' if len(undeclared) > 3 else '')
            raise ValueError(
                f'{len(undeclared)} of its channels ({shown}) have no declared '
                'units, so the sheet cannot tell what its levels are; declare '
                'them in the Imported Units pane, then press Edit again')
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
        """One sheet spanning several specifications' sheets.

        An edit such as a scale or a band is then applied to all of
        them at once. The sheets share one breakpoint list: the union
        of their breakpoints over the range they all cover, each
        sheet's levels read off its own power law (exact at its own
        breakpoints). Pairs are offered only within one specification.
        `for_source` returns each specification's own sheet, which
        `written_into` writes back at that specification's own lines.
        Raises `ValueError` when there are no sheets, or when they
        share fewer than two breakpoints.

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
        """The sheet of one of the specifications a combined sheet spans.

        It has that specification's channels and pairs only, and its
        `sources` is empty. Raises `ValueError` when no channel is
        from `source`.

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
        """Raise `ValueError` for anything no specification could hold.

        That is: no channels, a channel named twice, a channel with no
        quantity, fewer than two breakpoints or breakpoints that are
        not positive and increasing, a level that is not finite and
        positive, a coherence outside [0, 1], coherence and phase
        entries that cannot hold together (no set of signals has them;
        the reason names the channels and the nearest entries that would
        hold), a pair across two specifications, a band that is not
        below and above the target, a band that breaks its channel's
        `symmetric` or `uniform` constraint, a form other than
        'breakpoints' or 'lines', a spacing that is not positive, or an
        entry other than 'levels' or 'slopes'. Every draft is checked
        when it is made.
        """
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
        # the entries together, not only each on its own: three pairs
        # each in [0, 1] can still ask for a motion no signals make
        # (core.matrix_checks), and a sheet that held them would make a
        # specification no controller can play
        refusal = entries_refusal(self.channels, self.pairs,
                                  self.frequencies)
        if refusal is not None:
            raise ValueError(refusal)
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
        if self.entry not in ENTRIES:
            raise ValueError(f"levels are typed as 'levels' or 'slopes', not "
                             f'{self.entry!r}')

    def _sources(self) -> list[str]:
        return self.sources or [''] * self.num_channels

    def pairable(self) -> list[tuple[int, int]]:
        """Every pair `(i, j)`, `i < j`, that can have a cross term.

        Those are the pairs of channels from the same specification.
        """
        sources = self._sources()
        n = self.num_channels
        return [(i, j) for i in range(n) for j in range(i + 1, n)
                if sources[i] == sources[j]]

    def unset_pairs(self) -> list[tuple[int, int]]:
        """The pairs that have no cross term entered yet."""
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
                  'spacing': self.spacing, 'entry': self.entry}
        fields.update(changes)
        return SpecificationDraft(**fields)

    # ---- the operations, each a new draft ----------------------------------------

    def with_all_pairs(self, coherence: float,
                       phase_degrees: float = 0.0) -> SpecificationDraft:
        """The same draft with one coherence and phase entered for every pair.

        Parameters
        ----------
        coherence : float
            The coherence between every two channels, 0 for
            independent, 1 for fully coherent.
        phase_degrees : float, default 0.0
            The phase of the later channel against the earlier, in
            degrees.

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
        """The same draft with the target raised or lowered by `decibels`.

        The change is applied to every channel, or to the named ones.
        Raises `ValueError` for a channel the draft does not have.

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
        """The index of the segment each frequency lies in.

        A breakpoint belongs to the segment on its right, and the last
        breakpoint (and anything past the end) to the last segment.
        With no frequencies, the answer is for each breakpoint.
        """
        f = np.asarray(self.frequencies)
        last = max(len(f) - 2, 0)
        if frequencies is None:
            return np.minimum(np.arange(len(f)), last)
        at = np.atleast_1d(np.asarray(frequencies, dtype=float))
        return np.clip(np.searchsorted(f, at, side='right') - 1, 0, last)

    def band_db(self, kind: str, channel: int,
                frequencies: Any = None) -> tuple[np.ndarray, np.ndarray]:
        """One channel's band, (below, above) in decibels, at each frequency.

        Each frequency takes the band of the segment it lies in.

        Parameters
        ----------
        kind : {'warning', 'abort'}
            Which band.
        channel : int
            Which channel, by index.
        frequencies : array-like, optional
            The frequencies in Hz; the breakpoints themselves when
            omitted.

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
        """The same draft with a band set to the given values.

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
            A new draft; this one is unchanged. `ValueError` is raised
            where the band breaks a channel's own constraint.
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
                     step: float = LIMIT_STEP_DB) -> SpecificationDraft:
        """The same draft with one edge of a band moved by `decibels`.

        This is a drag on the plot, applied. On a symmetric channel the
        other edge moves to match, and on a uniform channel every
        segment moves. The edge lands on a multiple of `step` (by
        default `LIMIT_STEP_DB`, a quarter decibel) and is never nearer
        the target than one step.

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
            Which channels; all of them when omitted. The selected
            channels are the ones edited.
        step : float, default LIMIT_STEP_DB
            The decibel grid the edge lands on.

        Returns
        -------
        SpecificationDraft
            A new draft; this one is unchanged.
        """
        # A quarter decibel (Brandon, 2026-09-18), from the whole
        # decibel of 2026-09-06.
        return self._band_edge(kind, side, segments, channels, step,
                               lambda was: was + decibels)

    def with_band_edge(self, kind: str, side: str, segments: Any,
                       decibels: float, channels: Any = None,
                       step: float = LIMIT_STEP_DB) -> SpecificationDraft:
        """The same draft with one edge of a band set to `decibels`.

        This is a drag on the plot, applied. Every selected channel is
        set to the same level: a drag from 3 to 4 dB puts a channel at
        3 dB and one at 4 dB both at 4 dB, instead of moving each one
        by 1 dB. On a symmetric channel the other edge is set to match,
        and on a uniform channel every segment is set. The edge lands
        on a multiple of `step` and is never nearer the target than
        one step.

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
        step : float, default LIMIT_STEP_DB
            The decibel grid the edge lands on.

        Returns
        -------
        SpecificationDraft
            A new draft; this one is unchanged.
        """
        # Brandon, 2026-09-06: a drag sets every selected channel to the
        # level it was dragged to.
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
        """The same draft with the channels' constraints set.

        Turning symmetric on mirrors the band above the target to the
        one below, and turning uniform on gives every segment the first
        segment's band.

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
        """One channel's segments grouped into its requirement's straight sections.

        In the breakpoints form each segment is its own section. In the
        lines form a section is a run of segments on one power law,
        between the bends that `to_breakpoints` finds; `tolerance` is
        how much the log-log slope may change and still be one section.
        """
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
        """One channel's segments grouped as the plot draws its band handles.

        A uniform channel has one run over the whole range. Otherwise
        there is one run per section of the requirement (`sections`),
        split again wherever the band already differs within a section.
        """
        # Brandon, 2026-09-06: Uniform off should show a level per
        # section, each moved on its own.
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
        """The same draft with a breakpoint added at `frequency`.

        Its levels are read off the power law between its neighbors, or
        are the end breakpoint's level outside the range. The segment it
        splits keeps its band on both halves. A frequency that is
        already a breakpoint gives an unchanged copy, and one that is
        not positive raises `ValueError`.

        Parameters
        ----------
        frequency : float
            The frequency in Hz.

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
            The frequency in Hz. It has to be one of the breakpoints,
            and the draft has to have more than two; otherwise
            `ValueError` is raised. The merged segment keeps the band
            of the segment it starts in.

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

    # ---- written by slope ---------------------------------------------------

    def with_entry(self, entry: str) -> SpecificationDraft:
        """The same draft with its levels typed another way.

        Nothing in the requirement changes: the draft holds levels
        either way, and the entry decides what `with_level` holds
        still.

        Parameters
        ----------
        entry : {'levels', 'slopes'}
            'levels', a level at every breakpoint, or 'slopes', a level
            at the first and a slope in decibels per octave into each
            breakpoint after it.

        Returns
        -------
        SpecificationDraft
            A new draft; this one is unchanged. Any other entry raises
            `ValueError`.
        """
        return self._copy(entry=str(entry))

    def slopes_db(self) -> list[list[float]]:
        """Each channel's slope in decibels per octave on each segment,
        one row per channel, one fewer than the breakpoints: the table
        as it is written by slope (`core.breakpoints.segment_slopes`)."""
        from .breakpoints import segment_slopes

        return [[float(s) for s in segment_slopes(self.frequencies, row)]
                for row in self.levels]

    def with_slope(self, segment: int, db_per_octave: float,
                   channels: list[str] | None = None) -> SpecificationDraft:
        """The same draft with one segment's slope typed.

        The level at the segment's end is the one the slope reaches
        from its start, and every segment after it keeps its slope, so
        the levels past it move together. Bands, pairs and the levels
        before the segment are unchanged.

        Parameters
        ----------
        segment : int
            The segment by index from 0: segment k runs from breakpoint
            k to breakpoint k + 1.
        db_per_octave : float
            The slope. Plus 3 is the documents' plus 3 dB per octave,
            an exponent of 0.9966; `core.breakpoints.OCTAVE_DB` is an
            exponent of exactly one.
        channels : list of str, optional
            Which channels; all of them when omitted.

        Returns
        -------
        SpecificationDraft
            A new draft; this one is unchanged. A segment the draft
            does not have, or a channel it does not have, raises
            `ValueError`.
        """
        from .breakpoints import levels_with_slope

        levels = [list(row) for row in self.levels]
        for k in self._channel_indices(channels):
            levels[k] = [float(v) for v in levels_with_slope(
                self.frequencies, levels[k], segment, db_per_octave)]
        return self._copy(levels=levels)

    def with_level(self, breakpoint: int, level: float,
                   channels: list[str] | None = None) -> SpecificationDraft:
        """The same draft with one breakpoint's level typed, the way the
        draft's `entry` says.

        Under 'levels' that level changes and nothing else. Under
        'slopes' the slope into the breakpoint changes and every slope
        after it is held, so the levels past it move by the same factor;
        typed at the first breakpoint, the whole table moves with it.

        Parameters
        ----------
        breakpoint : int
            The breakpoint by index from 0.
        level : float
            The level, in the channel's SI density unit, positive.
        channels : list of str, optional
            Which channels; all of them when omitted.

        Returns
        -------
        SpecificationDraft
            A new draft; this one is unchanged. A breakpoint or channel
            the draft does not have, or a level that is not positive,
            raises `ValueError`.
        """
        from .breakpoints import levels_holding_slopes

        breakpoint, level = int(breakpoint), float(level)
        if not 0 <= breakpoint < len(self.frequencies):
            raise ValueError(f'the draft has {len(self.frequencies)} '
                             f'breakpoints; no breakpoint {breakpoint + 1}')
        if not (np.isfinite(level) and level > 0):
            raise ValueError('a level is positive and finite')
        levels = [list(row) for row in self.levels]
        for k in self._channel_indices(channels):
            if self.entry == 'slopes':
                levels[k] = [float(v) for v in levels_holding_slopes(
                    self.frequencies, levels[k], breakpoint, level)]
            else:
                levels[k][breakpoint] = level
        return self._copy(levels=levels)

    def with_slope_breakpoint(self, frequency: float, db_per_octave: float,
                              channels: list[str] | None = None
                              ) -> SpecificationDraft:
        """The same draft with a breakpoint entered as a slope and the
        next frequency, the level there computed.

        This is the line of a table that reads "plus 3 dB per octave to
        80 Hz": the slope runs from the breakpoint below `frequency` to
        it. Past the last breakpoint the table grows by one segment.
        Inside a segment the breakpoint splits it (`with_breakpoint`,
        both halves wearing its band) and the slope is typed on the
        lower half (`with_slope`), so the segments above keep their
        slopes. At a frequency that is already a breakpoint, the slope
        into it is typed. The channels not named take the level the
        power law between their neighbors gives, or the last
        breakpoint's past the end.

        Parameters
        ----------
        frequency : float
            The next frequency, in Hz, above the first breakpoint.
        db_per_octave : float
            The slope from the breakpoint below it.
        channels : list of str, optional
            Which channels the slope is typed for; all of them when
            omitted.

        Returns
        -------
        SpecificationDraft
            A new draft; this one is unchanged. A frequency at or below
            the first breakpoint raises `ValueError`, since there is no
            breakpoint below it for the slope to run from.
        """
        frequency = float(frequency)
        if not frequency > self.frequencies[0]:
            raise ValueError(
                f'a slope runs up from the breakpoint below it, and '
                f'{frequency:g} Hz is not above the first breakpoint at '
                f'{self.frequencies[0]:g} Hz')
        split = self.with_breakpoint(frequency)
        segment = split.frequencies.index(frequency) - 1
        return split.with_slope(segment, db_per_octave, channels)

    def reading(self, channel: int | str = 0) -> BreakpointReading:
        """One channel's table read: each segment's slope in decibels
        per octave, its area by the closed form, and the rms
        accumulated from the lowest breakpoint
        (`core.breakpoints.read_breakpoints`), in the channel's SI
        units: areas in the square of its quantity, rms in its
        quantity.

        Parameters
        ----------
        channel : int or str, default 0
            The channel by index or by name.

        Returns
        -------
        BreakpointReading
            The slopes, the areas and the running rms.

        Raises
        ------
        ValueError
            On the 'lines' form. A controller's lines are a density per
            line, integrated as bins, and read as a power law between
            them the areas add to an rms the specification does not
            have; `to_breakpoints` gives the table to read.
        """
        from .breakpoints import read_breakpoints

        if self.form == 'lines':
            raise ValueError(
                'the areas and the running rms read a breakpoint table, and '
                'the sheet is on its lines, a density per line; read the '
                'breakpoints instead (to_breakpoints)')
        if isinstance(channel, (int, np.integer)):
            k = int(channel)
            if not 0 <= k < self.num_channels:
                raise ValueError(f'the draft has {self.num_channels} '
                                 f'channels; no channel {k + 1}')
        else:
            k = self._channel_indices([channel])[0]
        return read_breakpoints(self.frequencies, self.levels[k])

    def to_breakpoints(self, tolerance: float = 1e-6) -> SpecificationDraft:
        """The same requirement as the few points it was written from.

        The breakpoints kept are the first and last line, every line
        where the log-log slope of any channel changes, and every line
        where a band steps. This is exact for a specification that was
        read onto lines from breakpoints (a controller's target). A
        specification with no such structure keeps every line. Either
        way a note says how many lines became how many breakpoints, and
        another note is added if a band changed within a merged
        segment, which then takes the band at its start.

        Parameters
        ----------
        tolerance : float, default 1e-6
            How much the slope may change between adjacent segments and
            still be one power law.

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
        """The same requirement read onto evenly spaced frequency lines.

        The lines are multiples of `spacing` from 0 Hz, as a controller
        lays out its lines, over the range the breakpoints cover, with
        each channel on its own power law between breakpoints. Only the
        lines inside that range are made: a sheet holds positive
        levels, and the zeros a controller writes outside the band are
        its own convention, filled in by its loader for any line the
        file does not name. A note is added when an end breakpoint is
        not on the grid.

        Parameters
        ----------
        spacing : float, optional
            The line spacing in Hz. Defaults to the spacing the sheet
            remembers. With neither, `ValueError` is raised; the spacing
            is not guessed. It is also raised when the spacing is not
            positive or fewer than two lines fall in the range.

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
        """Whether any channel's band changes from one segment to the next.

        Where it does, the specification needs a helper line to hold
        the step (`with_band_corners`).
        """
        return any(entry[kind][s] != entry[kind][s + 1]
                   for entry in self.bands for kind in ('warning', 'abort')
                   for s in range(len(entry[kind]) - 1))

    def with_band_corners(self, part: float = 1e-6) -> SpecificationDraft:
        """The same draft with helper breakpoints where a band steps.

        A helper is added just below each breakpoint whose two segments
        have different bands, so the specification made from the draft
        can hold the step. A limit array holds one value per line and a
        breakpoint belongs to the segment on its right, so the segment
        on its left needs its own line at the corner to carry its band
        up to it. The helpers are removed again when the specification
        is opened as a sheet, and `to_breakpoints` drops them, since
        they lie on a straight section.

        Parameters
        ----------
        part : float, default 1e-6
            How far below the breakpoint the helper is, as a fraction
            of the breakpoint's frequency. The helper is on the same
            power law.

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
        """The specification this sheet describes.

        A pair with no cross term entered is left out of the
        specification, not assumed to be zero. Where a band steps
        between segments, helper lines are added first
        (`with_band_corners`). The channels' `symmetric` and `uniform`
        constraints are kept on the specification as
        `band_constraints`, so a sheet reopens as it was set. Raises
        `ValueError` when the sheet spans several specifications.

        Parameters
        ----------
        comment : str, default ''
            A comment for the records, such as where the sheet came
            from. Defaults to 'written as a specification sheet'.

        Returns
        -------
        Specification
            The autospectra and the entered cross terms at the
            breakpoints, interpolated log-log between them, with the
            warning and abort limits on every autospectrum.
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
        """Each channel's `symmetric` and `uniform` constraints, by name.

        The specification keeps these beside its limits, so a sheet
        reopens as it was set.
        """
        return {name: {'symmetric': entry['symmetric'],
                       'uniform': entry['uniform']}
                for name, entry in zip(self.channels, self.bands)}

    def written_into(self, spec: Specification,
                     comment: str = '') -> Specification:
        """`spec` with this sheet's channels rewritten in it, at its own lines.

        The sheet's channels take the sheet's levels, cross terms and
        bands over the sheet's range, read off its power law at each
        line, and a breakpoint that is not one of the specification's
        lines becomes a new line. Lines the sheet left out (0 Hz, a
        zero level) keep what they had. A cross term the sheet has but
        the specification never held is added, made from the
        autospectra. Channels the sheet does not hold are unchanged,
        read off their own curves at any new line. Raises `ValueError`
        when the sheet spans several specifications; write each one
        with `for_source`.

        Parameters
        ----------
        spec : Specification
            The specification the sheet was opened from.
        comment : str, default ''
            The comment for each cross-term record the sheet adds;
            'written as a specification sheet' when empty. Records the
            specification already had keep their own comments.

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
        """The autospectra and their bands only, as the plot beside the sheet shows.

        Raises `ValueError` when the sheet spans several specifications.
        """
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
                'form': self.form, 'spacing': self.spacing,
                'entry': self.entry}

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
                   data.get('form', 'breakpoints'), data.get('spacing'),
                   data.get('entry', 'levels'))


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
    """The notes a sheet opened on `spec` shows about bands it does not have.

    A specification with no warning or abort limits opens with the
    default bands on the plot, ready to drag, and the first edit writes
    them to it. The notes are computed from the specification instead
    of being stored on the draft, so they stop once the limits exist.

    Parameters
    ----------
    spec : Specification
        The specification the sheet is open on.

    Returns
    -------
    list of str
        One note per band kind missing; empty when both are there.
    """
    # Brandon, 2026-09-06: where there are none, the sheet is where
    # they are added.
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
    """The spacing of evenly spaced lines, in Hz, or None when they are not.

    A controller writes its target at multiples of one spacing; a
    breakpoint specification is at whatever frequencies it was written
    at. Helper lines (just below a breakpoint, holding a band step; see
    `SpecificationDraft.with_band_corners`) are not on the grid and are
    ignored. Fewer than three lines gives None.
    """
    # Counted as grid lines, two helpers made a controller's target open
    # as four thousand breakpoints, each its own section (2026-09-06).
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
    """A specification record read at new lines.

    At its own lines the record keeps its own values. Between them it
    is interpolated the way a specification is read, the magnitude
    log-log and the phase linearly, so a line the sheet added does not
    change what a channel the sheet does not hold means. Outside its
    range it takes its end values.
    """
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
